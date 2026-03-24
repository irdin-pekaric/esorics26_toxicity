from __future__ import annotations

import os
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import csv
import json
import inspect
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import onnxruntime as ort

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    DataCollatorWithPadding, get_scheduler
)

from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import onnx
    HAVE_ONNX = True
except Exception:
    HAVE_ONNX = False

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAVE_QUANT = True
except Exception:
    HAVE_QUANT = False

try:
    from huggingface_hub import snapshot_download
    HAVE_HF_HUB = True
except Exception:
    HAVE_HF_HUB = False

BASE_DIR = Path(__file__).resolve().parent

TRAIN_CSV_PATH  = BASE_DIR / "2_16000_chatlogs_english_only_train_split.csv"
TEST_CSV_PATH   = BASE_DIR / "2_16000_chatlogs_english_only_test_split.csv"

TEXT_COL  = "message"
LABEL_COL = "label"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

USE_HF_ONNX_BASELINE = False
HF_REPO_ID = None
HF_ALLOW_PATTERNS = []

LOCAL_MODELS_ROOT = None
LOCAL_ONNX_DIR = None
LOCAL_ONNX_NAME = None
LOCAL_ONNX_PATH = None
BASELINE_TOKENIZER_DIR = None

PYTORCH_MODEL_ID = "unitary/toxic-bert"

MAX_LEN = 192
BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 2
RANDOM_SEED = 42

DO_THRESHOLD_SWEEP = True
THRESHOLD_MIN = 0.50
THRESHOLD_MAX = 0.90
THRESHOLD_STEPS = 41
TARGET_PRECISION = 0.85
MIN_RECALL = 0.30
FP_BUDGET = 60

BASELINE_2CLASS_SOFTMAX_THRESHOLD = 0.50
BASELINE_6CLASS_SIGMOID_THRESHOLD = 0.50

NEG_POS_CLASS_WEIGHTS = (1.15, 1.0)

EXPORT_ONNX = True
ONNX_OPSET = 14
EXPORT_DIR = BASE_DIR / "output"
CHECKPOINT_PATH = EXPORT_DIR / "best_model.pt"
FINE_TUNED_DIR  = EXPORT_DIR / "finetuned_model"
ONNX_DIR        = EXPORT_DIR / "onnx_ft"
ONNX_PATH       = ONNX_DIR / "toxic_bert_finetuned.onnx"
EXPORT_MAX_LEN  = MAX_LEN

APPLY_DYNAMIC_QUANT = True
ONNX_QUANT_PATH     = ONNX_DIR / "toxic_bert_finetuned.qdynamic.onnx"

OUTPUT_BASELINE_PRED   = EXPORT_DIR / "baseline_onnx_predictions.csv"
OUTPUT_FT_PRED         = EXPORT_DIR / "finetuned_predictions.csv"
OUTPUT_MTX_TXT         = EXPORT_DIR / "metrics.txt"
OUTPUT_SUMMARY_JSON    = EXPORT_DIR / "summary.json"
OUTPUT_BASELINE_CM_PNG = EXPORT_DIR / "baseline_onnx_confusion_matrix.png"
OUTPUT_FT_CM_PNG       = EXPORT_DIR / "finetuned_confusion_matrix.png"

VERBOSE = True

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def read_csv_auto(path: Path) -> pd.DataFrame:
    path = Path(path)
    with open(path, "r", encoding="latin1", newline="") as f:
        sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except csv.Error:
            sep = ","
    return pd.read_csv(path, sep=sep, encoding="latin1")


def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = read_csv_auto(csv_path)
    if LABEL_COL not in df.columns or TEXT_COL not in df.columns:
        raise KeyError(f"CSV must contain '{TEXT_COL}' and '{LABEL_COL}' columns.")
    df = df[df[LABEL_COL] != 2].copy()  # drop uncertain class if present
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int).clip(0,1)
    df = df[df[TEXT_COL].astype(str).str.len().fillna(0) >= 1]
    return df.reset_index(drop=True)


def load_train_test(train_csv: Path, test_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not train_csv or not test_csv:
        raise ValueError("Please set TRAIN_CSV_PATH and TEST_CSV_PATH.")
    train_df = load_and_preprocess(train_csv)
    test_df  = load_and_preprocess(test_csv)
    return train_df, test_df


def plot_cm(y_true, y_pred, title, out_png: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.xticks([0,1], ["Non-toxic", "Toxic"])
    plt.yticks([0,1], ["Non-toxic", "Toxic"])
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def metrics_from_preds(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm


def save_metrics_block(name: str, acc, prec, rec, f1, cm_vals, out_path: Path, threshold=None):
    tn, fp, fn, tp = cm_vals.ravel()
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{name}]\n")
        if threshold is not None:
            f.write(f"Threshold : {threshold:.4f}\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")
        f.write(f"Confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

def _pick_providers() -> List[str]:
    avail = ort.get_available_providers()
    return ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]


def _make_session(onnx_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = _pick_providers()
    if VERBOSE:
        print("ONNX providers:", providers)
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def _load_tokenizer_for_dir_or_model(dir_path: Optional[Path], fallback_model_id: str):
    raise RuntimeError("Baseline ONNX tokenizer loader removed; finetuning uses Unitary model only.")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _prepare_onnx_feeds(tokenizer, texts, max_len, session: ort.InferenceSession):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    input_names = {i.name for i in session.get_inputs()}
    feeds = {
        "input_ids": enc["input_ids"].astype("int64"),
        "attention_mask": enc["attention_mask"].astype("int64"),
    }
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = (
            enc["token_type_ids"].astype("int64")
            if "token_type_ids" in enc
            else np.zeros_like(feeds["input_ids"], dtype="int64")
        )
    feeds = {k: v for k, v in feeds.items() if k in input_names}
    return feeds


def _ensure_baseline_onnx_and_tokenizer() -> tuple[str, str]:
    raise RuntimeError("Baseline ONNX support removed; enable USE_HF_ONNX_BASELINE only if re-adding baseline code.")


def evaluate_baseline_onnx(test_df: pd.DataFrame, onnx_path: str, tokenizer_dir: Optional[str]) -> tuple:
    raise RuntimeError("Baseline ONNX evaluation is disabled in this repository. Use the finetuned model evaluation instead.")

class ChatDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)


@torch.no_grad()
def evaluate_loader_probs(model, dataloader, device):
    model.eval()
    probs_all, labels_all, losses = [], [], []
    for batch in dataloader:
        labels = batch["labels"].numpy().tolist()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        losses.append(float(outputs.loss.item()))
        p = torch.softmax(outputs.logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()
        probs_all.extend(p)
        labels_all.extend(labels)
    return np.array(labels_all, dtype=int), np.array(probs_all, dtype=float), float(np.mean(losses) if losses else 0.0)


def select_threshold(y_true, probs, sweep, target_prec, min_rec, fp_budget, out_csv: Path):
    rows = []
    best_f1_row = None
    for t in sweep:
        y_pred = (np.asarray(probs) >= float(t)).astype(int)
        acc, prec, rec, f1, cm = metrics_from_preds(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold": float(t), "acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
        if best_f1_row is None or f1 > best_f1_row["f1"]:
            best_f1_row = {"threshold": t, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm}
    df = pd.DataFrame(rows); out_csv.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out_csv, index=False)

    cand = df[(df["prec"] >= target_prec) & (df["rec"] >= min_rec)]
    if not cand.empty:
        r = cand.loc[cand["f1"].idxmax()]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=int)
        return float(r["threshold"]), float(r["acc"]), float(r["prec"]), float(r["rec"]), float(r["f1"]), cm

    if fp_budget is not None:
        fp_cand = df[df["fp"] <= fp_budget]
        if not fp_cand.empty:
            r = fp_cand.loc[fp_cand["f1"].idxmax()]
            cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=int)
            return float(r["threshold"]), float(r["acc"]), float(r["prec"]), float(r["rec"]), float(r["f1"]), cm

    r = best_f1_row
    return float(r["threshold"]), float(r["acc"]), float(r["prec"]), float(r["rec"]), float(r["f1"]), r["cm"]


def train_finetune(train_df, test_df, device):
    tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_ID)
    train_enc = tokenizer(train_df[TEXT_COL].astype(str).tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    test_enc  = tokenizer(test_df[TEXT_COL].astype(str).tolist(),  truncation=True, padding=True, max_length=MAX_LEN)

    train_ds = ChatDataset(train_enc, train_df[LABEL_COL].astype(int).tolist())
    val_ds   = ChatDataset(test_enc,  test_df[LABEL_COL].astype(int).tolist())

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    config = AutoConfig.from_pretrained(PYTORCH_MODEL_ID, num_labels=2, problem_type="single_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(
        PYTORCH_MODEL_ID, config=config, ignore_mismatched_sizes=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    class_weights = torch.tensor(NEG_POS_CLASS_WEIGHTS, dtype=torch.float32, device=device)
    best_loss, patience = float("inf"), 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in loop:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**inputs)
            loss = F.cross_entropy(out.logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        y_true_val, probs_val, _ = evaluate_loader_probs(model, val_loader, device)
        with torch.no_grad():
            val_loss = 0.0; n = 0
            for batch in val_loader:
                labels = batch["labels"].to(device)
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                o = model(**inputs)
                ce = F.cross_entropy(o.logits, labels, weight=class_weights)
                val_loss += float(ce.item()); n += 1
            val_loss = val_loss / max(1, n)
        y_pred_val = (probs_val >= 0.5).astype(int)
        acc, prec, rec, f1, _ = metrics_from_preds(y_true_val, y_pred_val)
        print(f"Validation: loss={val_loss:.4f} | acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")

        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            print("Saving best finetuned weights…")
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            FINE_TUNED_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(FINE_TUNED_DIR)
            tokenizer.save_pretrained(FINE_TUNED_DIR)
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model, val_loader

class _ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return out.logits


def export_to_onnx(model, tokenizer, out_path: Path, opset: int, max_len: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = tokenizer(["dummy export", "short"], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    device = next(model.parameters()).device
    dummy = {k: v.to(device) for k, v in dummy.items()}

    input_names = ["input_ids", "attention_mask"]
    inputs = (dummy["input_ids"], dummy["attention_mask"])
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    }
    if "token_type_ids" in dummy:
        input_names.append("token_type_ids")
        inputs = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq"}

    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        pass

    wrapper = _ONNXWrapper(model).to(device)
    wrapper.eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            inputs,
            f=str(out_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
    print(f"✅ Exported ONNX → {out_path}")


def maybe_quantize_dynamic(src_path: Path, dst_path: Path):
    if not HAVE_QUANT:
        print("Note: onnxruntime.quantization not installed → skipping dynamic quantization.")
        return

    sig = inspect.signature(quantize_dynamic)
    kwargs = {}
    if "weight_type" in sig.parameters:
        kwargs["weight_type"] = QuantType.QInt8
    if "per_channel" in sig.parameters:
        kwargs["per_channel"] = False
    if "reduce_range" in sig.parameters:
        kwargs["reduce_range"] = False
    if "optimize_model" in sig.parameters:
        kwargs["optimize_model"] = True
    if "op_types_to_quantize" in sig.parameters:
        kwargs["op_types_to_quantize"] = None
    if "extra_options" in sig.parameters:
        kwargs["extra_options"] = {}

    try:
        quantize_dynamic(
            model_input=str(src_path),
            model_output=str(dst_path),
            **kwargs,
        )
    except TypeError:
        quantize_dynamic(str(src_path), str(dst_path), weight_type=QuantType.QInt8)

    print(f"✅ Dynamic-quantized ONNX → {dst_path}")

def evaluate_exported_onnx(df: pd.DataFrame, onnx_path: str, tokenizer_dir: str, threshold: float):
    session = _make_session(onnx_path)
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    texts = df[TEXT_COL].astype(str).tolist()
    y_true = df[LABEL_COL].astype(int).to_numpy()

    logits_list = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        feeds = _prepare_onnx_feeds(tok, batch, MAX_LEN, session)
        out = session.run([session.get_outputs()[0].name], feeds)[0]
        logits_list.append(out)
    logits = np.concatenate(logits_list, axis=0)

    probs = _softmax(logits)[:, 1]
    y_pred = (np.asarray(probs) >= float(threshold)).astype(int)
    return metrics_from_preds(y_true, y_pred)

def main():
    train_csv = Path(TRAIN_CSV_PATH)
    test_csv  = Path(TEST_CSV_PATH)
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Set TRAIN_CSV_PATH and TEST_CSV_PATH to valid files.\n"
            f"TRAIN_CSV_PATH={train_csv}\nTEST_CSV_PATH={test_csv}"
        )

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    set_seeds(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", ("CUDA " + torch.cuda.get_device_name(0)) if device.type=="cuda" else "CPU")

    # --- Data ---
    train_df, test_df = load_train_test(train_csv, test_csv)
    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    print("Label dist (train):", train_df[LABEL_COL].value_counts().to_dict())
    print("Label dist (test) :", test_df[LABEL_COL].value_counts().to_dict())

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_MTX_TXT, "w", encoding="utf-8") as f:
        f.write("Toxicity evaluation — Baseline ONNX vs Finetuned PyTorch\n")
        f.write(f"PyTorch model id: {PYTORCH_MODEL_ID}\n")

    if USE_HF_ONNX_BASELINE:
        onnx_baseline_path, tok_dir_for_baseline = _ensure_baseline_onnx_and_tokenizer()

        base_acc, base_prec, base_rec, base_f1, base_true, base_pred, base_cm = evaluate_baseline_onnx(
            test_df, onnx_baseline_path, tok_dir_for_baseline
        )
        save_metrics_block("Baseline (ONNX quantized)", base_acc, base_prec, base_rec, base_f1, base_cm, OUTPUT_MTX_TXT)
    else:
        onnx_baseline_path = None
        tok_dir_for_baseline = None
        base_acc = base_prec = base_rec = base_f1 = 0.0
        base_true = base_pred = np.array([], dtype=int)
        base_cm = np.array([[0, 0], [0, 0]])

    model, val_loader = train_finetune(train_df, test_df, device)
    y_true_ft, probs_ft, _ = evaluate_loader_probs(model, val_loader, device)

    if DO_THRESHOLD_SWEEP:
        sweep = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)
        sweep_csv = EXPORT_DIR / "threshold_sweep.csv"
        sel_t, sel_acc, sel_prec, sel_rec, sel_f1, sel_cm = select_threshold(
            y_true_ft, probs_ft, sweep, TARGET_PRECISION, MIN_RECALL, FP_BUDGET, sweep_csv
        )
        chosen_threshold = float(sel_t)
    else:
        chosen_threshold = 0.75
        y_pred_tmp = (probs_ft >= chosen_threshold).astype(int)
        sel_acc, sel_prec, sel_rec, sel_f1, sel_cm = metrics_from_preds(y_true_ft, y_pred_tmp)

    print(f"Chosen threshold (finetuned): {chosen_threshold:.4f}")
    with open(OUTPUT_MTX_TXT, "a", encoding="utf-8") as f:
        f.write(f"\nChosen threshold (finetuned): {chosen_threshold:.4f}\n")

    y_pred_ft = (probs_ft >= chosen_threshold).astype(int)
    ft_acc, ft_prec, ft_rec, ft_f1, ft_cm = metrics_from_preds(y_true_ft, y_pred_ft)

    ft_pred_df = test_df.copy()
    ft_pred_df["predicted_label"] = y_pred_ft
    ft_pred_df.to_csv(OUTPUT_FT_PRED, index=False, encoding="utf-8")
    plot_cm(y_true_ft, y_pred_ft, f"Finetuned @ thr={chosen_threshold:.2f}", OUTPUT_FT_CM_PNG)

    save_metrics_block("Finetuned (PyTorch)", ft_acc, ft_prec, ft_rec, ft_f1, ft_cm, OUTPUT_MTX_TXT, threshold=chosen_threshold)

    exported_paths = {}
    parity_metrics = {}

    if EXPORT_ONNX:
        tokenizer = AutoTokenizer.from_pretrained(str(FINE_TUNED_DIR) if FINE_TUNED_DIR.exists() else PYTORCH_MODEL_ID, use_fast=True)
        export_to_onnx(model, tokenizer, ONNX_PATH, opset=ONNX_OPSET, max_len=EXPORT_MAX_LEN)

        if HAVE_ONNX:
            try:
                m = onnx.load(str(ONNX_PATH))
                onnx.checker.check_model(m)
                print("ONNX checker: OK")
            except Exception as e:
                print("ONNX checker warning:", e)

        tokenizer.save_pretrained(str(ONNX_DIR))
        model.config.save_pretrained(str(ONNX_DIR))
        exported_paths["onnx"] = str(ONNX_PATH)

        sample = test_df.sample(n=min(512, len(test_df)), random_state=RANDOM_SEED)
        acc_o, prec_o, rec_o, f1_o, _ = evaluate_exported_onnx(sample, str(ONNX_PATH), str(ONNX_DIR), chosen_threshold)
        print(f"ONNX parity (exported, n={len(sample)}): acc={acc_o:.4f} | prec={prec_o:.4f} | rec={rec_o:.4f} | f1={f1_o:.4f}")
        parity_metrics["onnx_fp32"] = {"acc": float(acc_o), "prec": float(prec_o), "rec": float(rec_o), "f1": float(f1_o)}

        if APPLY_DYNAMIC_QUANT:
            maybe_quantize_dynamic(ONNX_PATH, ONNX_QUANT_PATH)
            exported_paths["onnx_dynamic_quant"] = str(ONNX_QUANT_PATH)

            acc_q, prec_q, rec_q, f1_q, _ = evaluate_exported_onnx(sample, str(ONNX_QUANT_PATH), str(ONNX_DIR), chosen_threshold)
            print(f"ONNX parity (dynamic-quant, n={len(sample)}): acc={acc_q:.4f} | prec={prec_q:.4f} | rec={rec_q:.4f} | f1={f1_q:.4f}")
            parity_metrics["onnx_qdynamic"] = {"acc": float(acc_q), "prec": float(prec_q), "rec": float(rec_q), "f1": float(f1_q)}

    summary = {
        "baseline_onnx_repo": HF_REPO_ID if USE_HF_ONNX_BASELINE else None,
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "threshold_sweep_used": DO_THRESHOLD_SWEEP,
        "chosen_threshold_finetuned": float(chosen_threshold),
        "metrics": {
            "baseline_onnx": {"acc": float(base_acc), "prec": float(base_prec), "rec": float(base_rec), "f1": float(base_f1)},
            "finetuned": {"acc": float(ft_acc), "prec": float(ft_prec), "rec": float(ft_rec), "f1": float(ft_f1)},
            **({"parity_exported": parity_metrics} if parity_metrics else {}),
        },
        "artifacts": {
            "baseline_predictions_csv": str(OUTPUT_BASELINE_PRED) if USE_HF_ONNX_BASELINE else None,
            "finetuned_predictions_csv": str(OUTPUT_FT_PRED),
            "baseline_cm_png": str(OUTPUT_BASELINE_CM_PNG),
            "finetuned_cm_png": str(OUTPUT_FT_CM_PNG),
            "metrics_txt": str(OUTPUT_MTX_TXT),
            "checkpoint_pt": str(CHECKPOINT_PATH),
            "finetuned_dir": str(FINE_TUNED_DIR),
            **exported_paths
        }
    }
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if USE_HF_ONNX_BASELINE:
        print("\n=== Baseline (ONNX quantized) ===")
        print(f"Acc={base_acc:.4f} | Prec={base_prec:.4f} | Rec={base_rec:.4f} | F1={base_f1:.4f}")
        print("Saved:", OUTPUT_BASELINE_PRED, OUTPUT_BASELINE_CM_PNG)

    print("\n=== Finetuned (PyTorch, exported to ONNX) ===")
    print(f"Acc={ft_acc:.4f} | Prec={ft_prec:.4f} | Rec={ft_rec:.4f} | F1={ft_f1:.4f}")
    print("Chosen threshold:", f"{chosen_threshold:.4f}")
    print("Saved:", OUTPUT_FT_PRED, OUTPUT_FT_CM_PNG)

    print("\nArtifacts summary at:", OUTPUT_SUMMARY_JSON)


if __name__ == "__main__":
    main()
