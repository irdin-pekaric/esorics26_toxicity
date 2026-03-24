import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
from transformers.utils import logging as hf_logging

MODEL_NAME = "martin_ha_toxic_comment_model"
TOXIC_THRESHOLD = 0.5

hf_logging.set_verbosity_error()
_pipe = None

def _get_pipeline():
    global _pipe
    if _pipe is None:
        device = 0 if torch.cuda.is_available() else -1
        print("Device:", "CUDA" if device == 0 else "CPU")
        _pipe = pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-model",
            tokenizer="martin-ha/toxic-comment-model",
            device=device,
            truncation=True,
            top_k=None,
        )
    return _pipe

def _toxic_score_from_list(items) -> float:
    for d in items:
        if d.get("label", "").lower() == "toxic":
            return float(d.get("score", 0.0))
    return 0.0

def run_model(df: pd.DataFrame, text_col: str = "message") -> pd.DataFrame:
    tqdm.pandas(mininterval=0.3)
    pipe = _get_pipeline()

    texts = df[text_col].astype(str).tolist()
    with torch.inference_mode():
        outputs = pipe(texts, batch_size=32, truncation=True)  # list[list[dict]]

    scores = [_toxic_score_from_list(lst) for lst in outputs]
    out = pd.DataFrame({
        "id": df["id"] if "id" in df.columns else range(1, len(df) + 1),
        "toxicity_score": scores,
    })
    out["is_toxic"] = out["toxicity_score"] >= TOXIC_THRESHOLD
    out["toxicity_label"] = out["is_toxic"].map({True: "toxic", False: "non_toxic"})
    return out
