from __future__ import annotations

import argparse
from pathlib import Path

from typing import List

import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


TEXT_COL = "message"
LABEL_COL = "label"
DEFAULT_TEXT_COL = TEXT_COL
DEFAULT_LABEL_COL = LABEL_COL
BATCH_SIZE = 64
EXPORT_MAX_LEN = 192
THRESHOLD_DEFAULT = 0.58

def read_csv_flex(path: Path) -> pd.DataFrame:
    encs = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    seps = [",", ";", "\t", "|"]
    last_err = None
    best_df = None
    best_meta = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
                if best_df is None or df.shape[1] > best_df.shape[1]:
                    best_df = df
                    best_meta = (enc, sep, df.shape)
            except Exception as e:
                last_err = e
                continue
    if best_df is not None:
        enc, sep, shape = best_meta
        print(f"Loaded CSV {path} with encoding={enc}, sep='{sep}' → shape={shape}")
        return best_df
    raise last_err or RuntimeError(f"Could not read {path}")

def resolve_col(df: pd.DataFrame, desired: str) -> str:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    key = desired.lower()
    if key in lower:
        return lower[key]
    squeeze = {c.lower().replace(" ", "").replace("_", ""): c for c in cols}
    k2 = key.replace(" ", "").replace("_", "")
    if k2 in squeeze:
        return squeeze[k2]
    raise KeyError(f"Column '{desired}' not found in CSV columns: {cols}")

def labels_to_binary(series: pd.Series, numeric_threshold: int = 1) -> pd.Series:
    s = series.copy()
    try:
        s_int = s.astype(int)
        if set(pd.unique(s_int)) <= {0, 1}:
            return s_int
    except Exception:
        pass
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        uniq = set(pd.unique(s_num.dropna()))
        if uniq <= {0, 1}:
            return s_num.fillna(0).astype(int)
        return (s_num.fillna(0) >= float(numeric_threshold)).astype(int)
    lowered = s.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "toxic", "yes", "y"}
    return lowered.isin(truthy).astype(int)

DATA_PATHS = [
    ("2_16000_chatlogs_english_only_test_split.csv", {}),
    ("3_16000_chatlogs_grouped_match_level_test.csv", {}),
    ("4_16000_chatlogs_grouped_following_messages_test.csv", {}),
    ("gosu_ai_english_dota_2_game_chats.csv", {
        "rename_map": {"text": "message", "target": "label"},
        "text_col": "message",
        "label_col": "label",
        "id_col": "id",
        "numeric_threshold": 1,
    }),
    ("youtoxic_english_1000.csv", {
        "rename_map": {"IsToxic": "label", "Text": "message", "CommentId": "id"},
        "text_col": "message",
        "label_col": "label",
        "id_col": "id",
        "numeric_threshold": 1,
    }),
]

DEFAULT_ONNX_CANDIDATES = [
    "toxic_bert_finetuned.onnx",
]


def _pick_providers() -> List[str]:
    avail = ort.get_available_providers()
    return ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]


def _prep_inputs(tokenizer, texts: List[str], max_len: int):
    enc = tokenizer(list(texts), return_tensors="np", padding="max_length", truncation=True, max_length=max_len)
    ort_inputs = {}
    for k, v in enc.items():
        if hasattr(v, "dtype") and v.dtype != np.int64 and v.dtype.kind in ("i", "u"):
            ort_inputs[k] = v.astype(np.int64)
        else:
            ort_inputs[k] = v
    return ort_inputs


def _run_onnx(onnx_path: Path, tokenizer, texts: List[str], batch_size: int = BATCH_SIZE, max_len: int = EXPORT_MAX_LEN):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = _pick_providers()
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)

    probs = np.zeros(len(texts), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        ort_inputs = _prep_inputs(tokenizer, chunk, max_len)
        outs = sess.run(None, ort_inputs)
        logits = outs[0]

        if isinstance(logits, np.ndarray):
            if logits.ndim == 2:
                if logits.shape[1] == 1:
                    p = 1.0 / (1.0 + np.exp(-logits[:, 0]))
                else:
                    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
                    sm = ex / ex.sum(axis=1, keepdims=True)
                    idx = 1 if logits.shape[1] > 1 else 0
                    p = sm[:, min(idx, sm.shape[1] - 1)]
            elif logits.ndim == 1:
                p = 1.0 / (1.0 + np.exp(-logits))
            else:
                p = 1.0 / (1.0 + np.exp(-logits.reshape(len(chunk), -1)[:, 0]))
        else:
            p = np.array([0.0] * len(chunk), dtype=np.float32)

        probs[i:i + len(chunk)] = p.astype(np.float32)

    return probs


def evaluate_dataset(onnx_path: Path, tokenizer, data_path: Path, cfg: dict, threshold: float, out_root: Path):
    print(f"\n--- Evaluating {data_path} with finetuned ONNX {onnx_path.name} ---")
    df = read_csv_flex(data_path)

    if cfg.get("rename_map"):
        df = df.rename(columns=cfg["rename_map"])

    if "id" not in df.columns:
        df = df.copy()
        df.insert(0, "id", np.arange(1, len(df) + 1, dtype=int))

    text_col_name = cfg.get("text_col", DEFAULT_TEXT_COL)
    label_col_name = cfg.get("label_col", DEFAULT_LABEL_COL)

    try:
        text_col = resolve_col(df, text_col_name)
    except KeyError:
        print(f"Text column for {data_path.stem} not found (tried '{text_col_name}'), skipping.")
        return

    label_col = None
    if label_col_name:
        try:
            label_col = resolve_col(df, label_col_name)
        except KeyError:
            label_col = None

    texts = df[text_col].astype(str).tolist()
    ids = df["id"].tolist()

    probs = _run_onnx(onnx_path, tokenizer, texts)

    out_df = pd.DataFrame({"id": ids, "toxicity_score": probs})
    out_df["is_toxic"] = out_df["toxicity_score"] >= threshold
    out_df["toxicity_label"] = out_df["is_toxic"].map({True: "toxic", False: "non_toxic"})

    dataset_out = out_root / data_path.stem
    dataset_out.mkdir(parents=True, exist_ok=True)
    pred_file = dataset_out / f"finetuned_{onnx_path.stem}_predictions.csv"
    out_df.to_csv(pred_file, index=False)
    print(f"Predictions saved → {pred_file}")

    if label_col and label_col in df.columns and not df[label_col].isna().all():
        y_true = labels_to_binary(df[label_col], cfg.get("numeric_threshold", 1))
        y_pred = labels_to_binary(out_df["is_toxic"].astype(int), cfg.get("numeric_threshold", 1))

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]

        metrics_file = dataset_out / "metrics_finetuned_onnx.csv"
        dfm = pd.DataFrame([{"model": onnx_path.stem, "threshold": threshold, "accuracy": acc,
                             "precision": prec, "recall": rec, "f1": f1,
                             "tn": tn, "fp": fp, "fn": fn, "tp": tp}])
        if metrics_file.exists():
            existing = pd.read_csv(metrics_file)
            dfm = pd.concat([existing, dfm], ignore_index=True)
        dfm.to_csv(metrics_file, index=False)
        print(f"Metrics written → {metrics_file}")
    else:
        print("No label column found; predictions saved but no metrics computed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Path to finetuned ONNX model. If omitted, use training config default.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT, help="Threshold to binarize scores (default 0.5)")
    default_out = (Path(__file__).resolve().parent / "evaluations").resolve()
    parser.add_argument("--out-root", default=default_out, help="Base output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.is_absolute():
            onnx_path = (script_dir / onnx_path).resolve()
    else:
        onnx_path = None
        for cand in DEFAULT_ONNX_CANDIDATES:
            candidate_path = (script_dir / cand)
            if candidate_path.exists():
                onnx_path = candidate_path
                break
            for parent in script_dir.parents:
                found = list(parent.rglob(Path(cand).name))
                if found:
                    onnx_path = found[0]
                    break
            if onnx_path is not None:
                break
        if onnx_path is None:
            onnx_path = (script_dir / DEFAULT_ONNX_CANDIDATES[0]).resolve()

    if onnx_path is None or not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Please provide --onnx or place a model next to this script.")

    tok_dir = onnx_path.parent
    tokenizer = None
    try:
        if any((tok_dir / f).exists() for f in ("tokenizer.json", "vocab.txt", "vocab.json", "merges.txt")):
            tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        else:
            tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for entry in DATA_PATHS:
        if isinstance(entry, (list, tuple)):
            p = Path(entry[0])
            cfg = dict(entry[1]) if len(entry) > 1 and isinstance(entry[1], dict) else {}
        elif isinstance(entry, dict):
            if "path" in entry:
                p = Path(entry["path"])
                cfg = {k: v for k, v in entry.items() if k != "path"}
            else:
                raise TypeError(f"DATA_PATHS dict entries must contain 'path' key: {entry}")
        elif isinstance(entry, (str, Path)):
            p = Path(entry)
            cfg = {}
        else:
            raise TypeError(f"Unsupported DATA_PATHS entry type: {type(entry)} -> {entry}")

        if not p.is_absolute():
            p = (script_dir / p).resolve()

        if not p.exists():
            candidates = []
            for parent in [script_dir] + list(script_dir.parents):
                found = list(parent.rglob(p.name))
                if found:
                    candidates.extend(found)
            if candidates:
                p = candidates[0]
                print(f"Resolved missing path to: {p}")
            else:
                print(f"Skipping missing input path: {p}")
                continue

        evaluate_dataset(onnx_path, tokenizer, p, cfg, args.threshold, out_root)


if __name__ == "__main__":
    main()
