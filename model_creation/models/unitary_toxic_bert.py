import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

MODEL_NAME = "unitary_toxic_bert"

_pipe = None

def get_pipeline(device: torch.device | None = None):
    global _pipe
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_device = 0 if (device.type == "cuda") else -1
    if _pipe is None:
        _pipe = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            tokenizer="unitary/toxic-bert",
            device=hf_device,
            truncation=True,
        )
    return _pipe


def _interpret_label(pred_label: str, score: float, threshold: float = 0.5) -> bool:
    try:
        if score is not None:
            return float(score) >= float(threshold)
    except Exception:
        pass

    if pred_label is None:
        return False
    lbl = str(pred_label).strip().lower()
    if lbl in {"label_1", "1", "toxic", "true", "yes"}:
        return True
    if lbl in {"label_0", "0", "non-toxic", "non_toxic", "false", "no"}:
        return False
    return False


def run_model(df: pd.DataFrame, text_col: str = "message", device: torch.device | None = None, batch_size: int = 32, score_threshold: float = 0.45) -> pd.DataFrame:
    if "id" not in df.columns:
        ids = pd.Series(range(1, len(df) + 1), name="id")
        df_local = pd.concat([ids, df.reset_index(drop=True)], axis=1)
    else:
        df_local = df.reset_index(drop=True)

    pipe = get_pipeline(device)

    tqdm.pandas()

    def _single_predict(text: str):
        try:
            out = pipe(str(text))
        except Exception as e:
            return {"label": None, "score": 0.0}

        if isinstance(out, list):
            candidate = out[0] if out else {}
        elif isinstance(out, dict):
            candidate = out
        else:
            return {"label": str(out), "score": 0.0}

        label = None
        score = 0.0
        if isinstance(candidate, dict):
            if "label" in candidate and "score" in candidate:
                label = candidate.get("label")
                try:
                    score = float(candidate.get("score", 0.0))
                except Exception:
                    score = 0.0
            else:
                try:
                    kv = max(candidate.items(), key=lambda kv: float(kv[1]))
                    label = kv[0]
                    score = float(kv[1])
                except Exception:
                    label = str(candidate)
        else:
            label = str(candidate)

        return {"label": label, "score": score}

    preds_series = df_local[text_col].astype(str).progress_apply(_single_predict)

    labels = preds_series.apply(lambda p: p.get("label"))
    scores = preds_series.apply(lambda p: float(p.get("score", 0.0)))
    is_toxic = [ _interpret_label(l, s, threshold=score_threshold) for l, s in zip(labels, scores) ]

    out_df = pd.DataFrame({
        "id": df_local["id"],
        "toxicity_label": labels,
        "toxicity_score": scores,
        "is_toxic": is_toxic,
    })
    return out_df
