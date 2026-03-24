from __future__ import annotations

import warnings
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional, Union, List
import argparse
import inspect

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEXT_COL = "message"
LABEL_COL = "label"
NUMERIC_TOXIC_THRESHOLD = 1

DATA_PATHS: List[Union[Path, str, tuple, dict]] = [
    REPO_ROOT / "1_dataset" / "2_16000_chatlogs_english_only_test_split.csv",
    REPO_ROOT / "1_dataset" / "3_16000_chatlogs_grouped_match_level_test.csv",
    REPO_ROOT / "1_dataset" / "4_16000_chatlogs_grouped_following_messages_test.csv",
    (
        REPO_ROOT / "1_dataset" / "gosu_ai_english_dota_2_game_chats.csv",
        {
            "rename_map": {"text": "message", "target": "label"},
            "text_col": "message",
            "label_col": "label",
            "id_col": "id",
            "numeric_threshold": 1,
        },
    ),
     (
         REPO_ROOT / "1_dataset" / "youtoxic_english_1000.csv",
         {
             "rename_map": {
                "IsToxic": "label",
                "is_toxic": "label",
                "Text": "message",
                "text": "message",
                "CommentId": "id",
                "commentid": "id",
         },
             "text_col": "message",
             "label_col": "label",
             "id_col": "id",
             "numeric_threshold": 1,
             "keep_columns": ["id", "message", "label"],
         },
     ),
 ]

DATA_GLOB: Optional[Union[Path, str]] = None

OUT_DIR: Optional[Union[Path, str]] = None

DATASET_CONFIG = {

}

print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")

try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

from models.garak_llm_roberta_toxicity_classifier import (
    run_model as run_garak_roberta, MODEL_NAME as GARAK_MODEL,
)
from models.junglelee_bert_toxic_comment_classification import (
    run_model as run_junglelee_bert, MODEL_NAME as JUNGLE_MODEL,
)
from models.martin_ha_toxic_comment_model import (
    run_model as run_martin_ha, MODEL_NAME as MARTIN_HA_MODEL,
)
from models.nicholas_kluge_toxicity_model import (
    run_model as run_nicholas_kluge, MODEL_NAME as NICHOLAS_MODEL,
)
from models.protectai_unbiased_toxic_roberta_onnx import (
    run_model as run_protectai_roberta, MODEL_NAME as PROTECTAI_ONNX_MODEL,
)
from models.unitary_toxic_bert import (
    run_model as run_unitary_toxic_bert, MODEL_NAME as UNITARY_MODEL,
)

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
        print(f"Loaded CSV with encoding={enc}, sep='{sep}' → shape={shape}")
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

def labels_to_binary(series: pd.Series, numeric_threshold: Optional[float] = None) -> pd.Series:
    if numeric_threshold is None:
        numeric_threshold = NUMERIC_TOXIC_THRESHOLD

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

def evaluate_and_append(metrics_file: Path, model_name: str, y_true: pd.Series, y_pred: pd.Series):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    row = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    if metrics_file.exists():
        existing = pd.read_csv(metrics_file)
        out_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        out_df = pd.DataFrame([row])
    out_df.to_csv(metrics_file, index=False)
    print(f"[{model_name}] metrics appended → {metrics_file}")

def save_predictions_with_truth(preds: pd.DataFrame, df: pd.DataFrame, out_dir: Path,
                                model_name: str, text_col: str, label_col: str | None,
                                numeric_threshold: Optional[float] = None):
    pred_file = out_dir / f"{model_name}_predictions.csv"
    if label_col and label_col in df.columns:
        keep_cols = ["id", text_col, label_col]
        merged = preds.merge(df[keep_cols], on="id", how="left")
        merged = merged.rename(columns={label_col: "true_label", text_col: "message"})
        merged["true_label_bin"] = labels_to_binary(merged["true_label"], numeric_threshold)
        merged["pred_bin"] = labels_to_binary(merged["is_toxic"], numeric_threshold)
        merged["is_error"] = merged["pred_bin"] != merged["true_label_bin"]
        order = ["id", "message", "true_label", "true_label_bin",
                 "toxicity_label", "toxicity_score", "pred_bin", "is_error"]
        order = [c for c in order if c in merged.columns]
        merged[order].to_csv(pred_file, index=False)
    else:
        preds.to_csv(pred_file, index=False)
    print(f"[{model_name}] predictions saved → {pred_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate toxicity models on one or more CSV datasets")
    parser.add_argument("--data", "-d", action="append", help="Input file or directory (can be supplied multiple times). Can be a file path or directory containing CSVs.")
    parser.add_argument("--glob", "-g", help="Glob pattern (relative to repo root or absolute) to match input CSV files, e.g. 'evaluations/*/*.csv'")
    parser.add_argument("--out-dir", "-o", help="Base output directory where per-dataset folders will be created. If omitted, defaults to BASE_DIR / 'evaluations'.")
    parser.add_argument("--score-threshold", type=float, default=0.45, help="Score threshold to interpret probability as toxic (default: 0.45)")
    args = parser.parse_args()

    def collect_input_files() -> list[tuple[Path, dict]]:
        files: list[tuple[Path, dict]] = []
        def add_path(p: Path, cfg: dict | None = None):
            cfg = cfg or {}
            if p.exists():
                if p.is_dir():
                    for f in sorted([f for f in p.glob("*.csv") if f.is_file()]):
                        files.append((f, cfg))
                else:
                    files.append((p, cfg))
            else:
                s = str(p)
                if "*" in s:
                    for f in sorted([f for f in REPO_ROOT.glob(s) if f.is_file()]):
                        files.append((f, cfg))
                else:
                    alt = REPO_ROOT / p
                    if alt.exists():
                        if alt.is_dir():
                            for f in sorted([f for f in alt.glob("*.csv") if f.is_file()]):
                                files.append((f, cfg))
                        else:
                            files.append((alt, cfg))
                    else:
                        print(f"Warning: input path not found: {p}")
        data_sources = args.data if args.data else DATA_PATHS or []
        for item in data_sources:
            if isinstance(item, (list, tuple)):
                p = Path(item[0])
                cfg = dict(item[1]) if len(item) > 1 and isinstance(item[1], dict) else {}
                add_path(p, cfg)
            elif isinstance(item, dict):
                if "path" in item:
                    p = Path(item["path"])
                    cfg = {k: v for k, v in item.items() if k != "path"}
                    add_path(p, cfg)
                else:
                    print(f"Warning: dict entry in DATA_PATHS missing 'path' key: {item}")
            else:
                add_path(Path(item), {})

        if args.glob is not None:
            add_path(Path(args.glob))
        elif DATA_GLOB is not None:
            add_path(Path(DATA_GLOB))
        seen = set()
        out = []
        for f, cfg in files:
            fp = f.resolve()
            if fp not in seen:
                seen.add(fp)
                out.append((fp, cfg))
        return out

    data_files = collect_input_files()  # list of (Path, cfg)
    if not data_files:
        raise FileNotFoundError("No input CSV files found. Check DATA_PATHS / DATA_GLOB configuration or supply --data / --glob arguments.")

    if args.out_dir:
        base_out_root = Path(args.out_dir)
    elif OUT_DIR:
        base_out_root = Path(OUT_DIR)
    else:
        base_out_root = BASE_DIR / "evaluations"
    base_out_root.mkdir(parents=True, exist_ok=True)
    print(f"Base output directory: {base_out_root}")

    for data_path, inline_cfg in data_files:
        print(f"\n--- Evaluating file: {data_path} ---")

        if not data_path.exists():
            print(f"Skipping missing file: {data_path}")
            continue

        out_dir = base_out_root / data_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        df = read_csv_flex(data_path)

        dataset_key = data_path.stem
        cfg_global = DATASET_CONFIG.get(dataset_key, {})
        cfg = {**cfg_global, **(inline_cfg or {})}
        if cfg.get("rename_map"):
            df = df.rename(columns=cfg["rename_map"])

        if "id" not in df.columns:
            df = df.copy()
            df.insert(0, "id", np.arange(1, len(df) + 1, dtype=int))
            print("No 'id' column found; synthesized sequential IDs starting at 1.")

        dataset_key = data_path.stem
        cfg_global = DATASET_CONFIG.get(dataset_key, {})
        cfg = {**cfg_global, **(inline_cfg or {})}
        if cfg.get("rename_map"):
            df = df.rename(columns=cfg["rename_map"])

        if cfg.get("keep_columns"):
            resolved_keep: List[str] = []
            for desired in cfg["keep_columns"]:
                try:
                    col = resolve_col(df, desired)
                    if col not in resolved_keep:
                        resolved_keep.append(col)
                except KeyError:
                    print(f"Note: requested keep_column '{desired}' not found in dataset; skipping.")

            if "id" in df.columns and "id" not in resolved_keep:
                resolved_keep.insert(0, "id")

            desired_text = cfg.get("text_col", TEXT_COL)
            try:
                txt_col = resolve_col(df, desired_text)
                if txt_col not in resolved_keep:
                    resolved_keep.append(txt_col)
            except KeyError:
                pass

            if resolved_keep:
                df = df[resolved_keep]
                print(f"Restricted columns to: {resolved_keep}")
            else:
                print(f"Requested keep_columns {cfg['keep_columns']} resolved to none - leaving all columns intact.")

        text_col_name = cfg.get("text_col", TEXT_COL)
        label_col_name = cfg.get("label_col", LABEL_COL)
        id_col_name = cfg.get("id_col", "id")
        numeric_threshold = cfg.get("numeric_threshold", NUMERIC_TOXIC_THRESHOLD)

        try:
            text_col = resolve_col(df, text_col_name)
        except KeyError as e:
            raise KeyError(f"Text column for dataset '{dataset_key}' not found (tried '{text_col_name}'): {e}")

        label_col = None
        if label_col_name:
            try:
                label_col = resolve_col(df, label_col_name)
            except KeyError:
                label_col = None
                print(f"⚠️ Label column '{label_col_name}' not found for dataset '{dataset_key}'; proceeding without labels.")

        y_true = None
        if label_col and label_col in df.columns and not df[label_col].isna().all():
            y_true = labels_to_binary(df[label_col], numeric_threshold)

        models_to_run = [
            (GARAK_MODEL, run_garak_roberta),
            (JUNGLE_MODEL, run_junglelee_bert),
            (MARTIN_HA_MODEL, run_martin_ha),
            (NICHOLAS_MODEL, run_nicholas_kluge),
            (PROTECTAI_ONNX_MODEL, run_protectai_roberta),
            (UNITARY_MODEL, run_unitary_toxic_bert),
        ]

        metrics_file = out_dir / "metrics_all_models.csv"

        for model_name, runner in models_to_run:
            print(f"\n=== Running {model_name} ===")
            try:
                try:
                    sig = inspect.signature(runner)
                    if "score_threshold" in sig.parameters:
                        preds = runner(df, text_col=text_col, score_threshold=args.score_threshold)
                    else:
                        preds = runner(df, text_col=text_col)
                except ValueError:
                    preds = runner(df, text_col=text_col)
            except Exception as e:
                err_msg = f"Failed to run model {model_name}: {e}"
                print(err_msg)
                try:
                    (out_dir / f"{model_name}_error.txt").write_text(err_msg)
                except Exception:
                    pass
                continue

            save_predictions_with_truth(preds, df, out_dir, model_name, text_col, label_col, numeric_threshold)

            if y_true is not None:
                try:
                    y_pred = labels_to_binary(preds["is_toxic"], numeric_threshold)
                    evaluate_and_append(metrics_file, model_name, y_true, y_pred)
                except Exception as e:
                    m_err = f"Failed to evaluate model {model_name}: {e}"
                    print(m_err)
                    try:
                        (out_dir / f"{model_name}_metrics_error.txt").write_text(m_err)
                    except Exception:
                        pass
            else:
                print(f"[{model_name}] No LABEL_COL found; skipping metrics.")

        print(f"\nFinished evaluation for: {data_path}. Outputs → {out_dir}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
