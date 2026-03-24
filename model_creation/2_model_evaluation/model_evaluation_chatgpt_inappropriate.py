from __future__ import annotations

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import openai
except Exception:
    openai = None

# ----------------------
# Configuration
# ----------------------
OPENAI_API_KEY = "ENTER_YOUR_OPENAI_API_KEY"
if os.environ.get("OPENAI_API_KEY"):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS: List[tuple[Path, dict]] = [
    (REPO_ROOT / "1_dataset" / "2_16000_chatlogs_english_only_test_split.csv", {}),
]

DEFAULT_MODEL = "gpt-4o"
DEFAULT_PROMPT = (
    "You are a classifier that detects toxic behavior in gaming chats.\n"
    "Classify the following message as either `inappropriate' or 'non-inappropriate'.\n"
    "The message may contain slang, sarcasm, abbreviations, or profanity.\n"
    "Your response must be only: inappropriate OR non-inappropriate.\n\n"
    "Message:\n\n{text}\n\n"
    "Example response: inappropriate\n"
)
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "evaluations_chatgpt_inappropriate"
DEFAULT_MAX_SAMPLES = None

# ----------------------
# Script
# ----------------------
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
    truthy = {"1", "true", "toxic", "inappropriate", "yes", "y"}
    return lowered.isin(truthy).astype(int)


class ChatGPTClient:
    def __init__(self, model: str = "gpt-4o", debug: bool = False):
        self.model = model
        self.debug = debug
        if openai is None:
            raise RuntimeError("The 'openai' package is required but not installed. Install it with 'pip install openai'.")
        if not OPENAI_API_KEY:
            raise RuntimeError("OpenAI API key not set. Edit OPENAI_API_KEY at the top of this file or set OPENAI_API_KEY env var.")
        self._new_client = None
        try:
            if hasattr(openai, "OpenAI"):
                try:
                    self._new_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                except TypeError:
                    self._new_client = openai.OpenAI()
                if self.debug:
                    print("Using new OpenAI client API (openai.OpenAI).")
            else:
                # fallback to setting legacy global api_key
                openai.api_key = OPENAI_API_KEY
                if self.debug:
                    print("Using legacy openai module API (openai.ChatCompletion).")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    def generate(self, prompt: str, timeout: int = 300) -> tuple[str, Dict[str, int]]:
        try:
            if self.debug:
                print(f"Requesting model={self.model} prompt_len={len(prompt)}")
            if self._new_client is not None and hasattr(self._new_client, "chat"):
                resp = self._new_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                try:
                    choices = getattr(resp, "choices", None) or resp.get("choices")
                except Exception:
                    choices = None
                if not choices:
                    raise RuntimeError(f"No choices returned from OpenAI (new client): {resp}")
                first = choices[0]
                content = None
                if hasattr(first, "message"):
                    content = getattr(first.message, "content", None) or (getattr(first.message, "content", "") if hasattr(first.message, "content") else None)
                if not content:
                    content = first.get("message", {}).get("content", "") if isinstance(first, dict) else (first.get("delta", {}).get("content", "") if isinstance(first, dict) else "")
                content = (content or "").strip()
                usage = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else {})
                try:
                    usage = {k: int(v) for k, v in usage.items()} if isinstance(usage, dict) else dict(usage) if usage is not None else {}
                except Exception:
                    usage = usage or {}
                if self.debug:
                    print("OpenAI (new) response:", content[:400])
                    print("OpenAI (new) usage:", usage)
                return content, usage

            if hasattr(openai, "ChatCompletion"):
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                    timeout=timeout,
                )
                choices = resp.get("choices") or []
                if not choices:
                    raise RuntimeError(f"No choices returned from OpenAI (legacy): {resp}")
                content = choices[0].get("message", {}).get("content", "").strip()
                usage = resp.get("usage") or {}
                try:
                    usage = {k: int(v) for k, v in usage.items()} if isinstance(usage, dict) else {}
                except Exception:
                    usage = usage or {}
                if self.debug:
                    print("OpenAI (legacy) response:", content[:400])
                    print("OpenAI (legacy) usage:", usage)
                return content, usage

            raise RuntimeError("Could not find a supported OpenAI API interface in the installed openai package.")
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e


def parse_toxicity_from_text(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    s = str(text).lower()
    s = s.replace('`', ' ')
    s = re.sub(r'\bnon\s*[-_]?\s*inappropriate\b', 'noninappropriate', s)
    s = re.sub(r'(?<=\w)[-_](?=\w)', '', s)
    s = re.sub(r'[^a-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        return None
    m = re.search(r"\b(noninappropriate|inappropriate)\b", s)
    if not m:
        return None
    token = m.group(1)
    if token == 'inappropriate':
        return 1
    if token == 'noninappropriate':
        return 0
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


def evaluate_dataset(client: ChatGPTClient, data_path: Path, cfg: dict, prompt_template: str, out_root: Path, max_samples: Optional[int]):
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
    total_tokens = 0
    token_count_samples = 0

    for idx, row in it:
        text = str(row[text_col_res])
        prompt = prompt_template.replace("{text}", text)
        try:
            out, usage = client.generate(prompt)
        except Exception as e:
            out = f"<generation_error: {e}>"
            usage = None
        pred_bin = parse_toxicity_from_text(out)
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
            "usage_prompt_tokens": int(usage.get("prompt_tokens")) if usage and usage.get("prompt_tokens") is not None else None,
            "usage_completion_tokens": int(usage.get("completion_tokens")) if usage and usage.get("completion_tokens") is not None else None,
            "usage_total_tokens": int(usage.get("total_tokens")) if usage and usage.get("total_tokens") is not None else None,
        })
        if usage and usage.get("total_tokens") is not None:
            try:
                total_tokens += int(usage.get("total_tokens"))
                token_count_samples += 1
            except Exception:
                pass
        y_true.append(label_bin)
        y_pred.append(pred_bin)

    out_dir = out_root / data_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / f"chatgpt_{client.model}_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_file, index=False)

    metrics = compute_metrics(y_true, y_pred)
    metrics["token_samples_with_usage"] = int(token_count_samples)
    metrics["total_tokens"] = int(total_tokens)
    metrics["avg_tokens_per_response"] = float(total_tokens / token_count_samples) if token_count_samples else None
    (out_dir / "metrics_chatgpt.json").write_text(json.dumps(metrics, indent=2))

    print(f"Predictions saved → {pred_file}")
    print("Metrics:", json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="chat model name to run (default: gpt-4o)")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT, help="Prompt template containing {text} placeholder")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output directory (repo-relative)")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES, help="Limit number of samples per dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    client = None
    try:
        client = ChatGPTClient(model=args.model, debug=args.debug)
    except Exception as e:
        print(f"Failed to initialize ChatGPT client: {e}")
        print("Ensure OPENAI_API_KEY is set in the environment or edit OPENAI_API_KEY at the top of this file, and the 'openai' package is installed.")
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
