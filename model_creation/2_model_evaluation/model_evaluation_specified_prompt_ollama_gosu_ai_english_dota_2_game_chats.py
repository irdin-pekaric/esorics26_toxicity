from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS: List[tuple[Path, dict]] = [
    (REPO_ROOT / "1_dataset" / "gosu_ai_english_dota_2_game_chats.csv", {
        "rename_map": {"text": "message", "target": "label"},
        "text_col": "message",
        "label_col": "label",
        "id_col": "id",
        "numeric_threshold": 1,
    }),
]

DEFAULT_MODEL = "llama3.2"
DEFAULT_PROMPT = (
    'Please read the following chat messages from a Dota match and decide whether it is TOXIC.\n'
    'Respond with a JSON object only, with the keys: "answer" (true or false) and "explanation" (a short string).\n'
    'Do NOT include any extra text outside the JSON.\n\n'
    'Text:\n\n{text}\n\n'
    'Example response: {"answer": true, "explanation": "contains insults"}\n'
)
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "evaluations_llama_specified_prompt"
DEFAULT_MAX_SAMPLES = None

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
        best_df.columns = [str(c).lstrip('\ufeff').strip() for c in best_df.columns]
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

class OllamaClient:
    def __init__(self, model: str = "llama2", debug: bool = False):
        self.model = model
        self.debug = debug
        self.ollama_cmd = os.environ.get("OLLAMA_CLI") or shutil.which("ollama")
        if not self.ollama_cmd:
            for c in ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama", "/usr/bin/ollama"):
                if Path(c).exists():
                    self.ollama_cmd = c
                    break
        if not self.ollama_cmd:
            path_info = os.environ.get("PATH", "")
            raise RuntimeError(
                "'ollama' CLI not found. Ensure ollama is installed and on PATH, or set OLLAMA_CLI=/path/to/ollama.\n"
                f"PATH={path_info}"
            )
        try:
            proc = subprocess.run([self.ollama_cmd, "version"], capture_output=True, text=True, check=True)
            print(f"ollama at {self.ollama_cmd}: {proc.stdout.strip()}")
        except Exception:
            pass

    def generate(self, prompt: str, timeout: int = 300) -> str:
        cmd = [self.ollama_cmd, "run", self.model]
        try:
            if self.debug:
                print(f"OLLAMA CMD: {cmd}")
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, check=False, timeout=timeout)
            out = proc.stdout.strip()
            err = (proc.stderr or "").strip()
            if self.debug:
                print(f"OLLAMA rc={proc.returncode} stdout=<{len(out)} chars> stderr=<{len(err)} chars>")
                if out:
                    print("OLLAMA STDOUT:\n", out)
                if err:
                    print("OLLAMA STDERR:\n", err)
            if proc.returncode != 0:
                raise RuntimeError(f"ollama run failed (rc={proc.returncode}): {err}")
            return out
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            raise RuntimeError(f"ollama run failed: {stderr}") from e
        except Exception as e:
            raise RuntimeError(f"ollama invocation error: {e}") from e

_BOOL_RE = re.compile(r"\b(true|false|yes|no|1|0)\b", re.IGNORECASE)

def parse_bool_from_text(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and 'answer' in obj:
                    ans = obj['answer']
                    if isinstance(ans, bool):
                        return ans
                    if isinstance(ans, (int, float)):
                        return bool(ans)
                    s = str(ans).strip().lower()
                    if s in ('true', 'yes', '1'):
                        return True
                    if s in ('false', 'no', '0'):
                        return False
            except Exception:
                pass
    except Exception:
        pass

    m = _BOOL_RE.search(text)
    if m:
        tok = m.group(1).lower()
        if tok in ("true", "yes", "1"):
            return True
        if tok in ("false", "no", "0"):
            return False
    low = text.lower()
    if "true" in low and "false" not in low:
        return True
    if "false" in low and "true" not in low:
        return False
    return None

def compute_metrics(y_true: List[Optional[int]], y_pred: List[Optional[int]]) -> Dict[str, Any]:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not pairs:
        return {"valid": 0}
    ys, yp = zip(*pairs)
    ys = np.array(ys, dtype=int)
    yp = np.array(yp, dtype=int)
    acc = float(accuracy_score(ys, yp))
    prec = float(precision_score(ys, yp, zero_division=0))
    rec = float(recall_score(ys, yp, zero_division=0))
    f1 = float(f1_score(ys, yp, zero_division=0))
    cm = confusion_matrix(ys, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    return {"valid": len(pairs), "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def evaluate_dataset(client: OllamaClient, data_path: Path, cfg: dict, prompt_template: str, out_root: Path, max_samples: Optional[int]):
    print(f"\n--- Evaluating file: {data_path} ---")
    df = read_csv_flex(data_path)

    if cfg.get("rename_map"):
        df = df.rename(columns=cfg["rename_map"])

    if "id" not in df.columns:
        df = df.copy()
        df.insert(0, "id", np.arange(1, len(df) + 1, dtype=int))
        print("No 'id' column found; synthesized sequential IDs starting at 1.")

    text_col = cfg.get("text_col", "message")
    label_col = cfg.get("label_col", "label")

    try:
        text_col_res = resolve_col(df, text_col)
    except KeyError as e:
        print(f"Text column for dataset '{data_path.stem}' not found (tried '{text_col}'): {e}. Skipping.")
        return

    label_col_res = None
    if label_col:
        try:
            label_col_res = resolve_col(df, label_col)
        except KeyError:
            label_col_res = None
            print(f"Label column '{label_col}' not found; running unlabeled evaluation.")

    it = list(df.iterrows())
    if max_samples:
        it = it[:max_samples]

    rows = []
    y_true: List[Optional[int]] = []
    y_pred: List[Optional[int]] = []

    for idx, row in it:
        text = str(row[text_col_res])
        prompt = prompt_template.replace("{text}", text)
        try:
            out = client.generate(prompt)
        except Exception as e:
            out = f"<generation_error: {e}>"
        pred_bool = parse_bool_from_text(out)
        pred_bin = int(pred_bool) if pred_bool is not None else None
        label_bin = None
        if label_col_res and label_col_res in row.index:
            try:
                label_bin = int(labels_to_binary(pd.Series([row[label_col_res]]))[0])
            except Exception:
                label_bin = None
        rows.append({
            "id": row.get("id", idx),
            "text": text,
            "prompt": prompt,
            "llm_output": out,
            "prediction": pred_bin,
            "label": label_bin,
        })
        y_true.append(label_bin)
        y_pred.append(pred_bin)

    out_dir = out_root / data_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / f"ollama_{client.model}_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_file, index=False)

    metrics = compute_metrics(y_true, y_pred)
    (out_dir / "metrics_ollama.json").write_text(json.dumps(metrics, indent=2))

    print(f"Predictions saved → {pred_file}")
    print("Metrics:", json.dumps(metrics, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ollama model name to run (default: llama3.2)")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT, help="Prompt template containing {text} placeholder")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output directory (repo-relative)")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES, help="Limit number of samples per dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    client = None
    try:
        client = OllamaClient(model=args.model, debug=args.debug)
    except Exception as e:
        print(f"Failed to initialize Ollama client: {e}")
        print("Ensure ollama is installed and 'ollama run <model>' works in your shell.")
        return

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for data_path, cfg in DATASETS:
        if not data_path.exists():
            candidate = REPO_ROOT / Path(data_path)
            if candidate.exists():
                data_path = candidate
            else:
                print(f"Skipping missing dataset: {data_path}")
                continue
        evaluate_dataset(client, data_path, dict(cfg), args.prompt_template, out_root, args.max_samples)


if __name__ == "__main__":
    main()
