import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
from transformers.utils import logging as hf_logging

MODEL_NAME = "junglelee_bert_toxic_comment_classification"

hf_logging.set_verbosity_error()
_pipe = None

def _get_pipeline():
    global _pipe
    if _pipe is None:
        device = 0 if torch.cuda.is_available() else -1
        print("Device:", "CUDA" if device == 0 else "CPU")
        _pipe = pipeline(
            "text-classification",
            model="JungleLee/bert-toxic-comment-classification",
            tokenizer="JungleLee/bert-toxic-comment-classification",
            device=device,
            truncation=True,
        )
    return _pipe

def run_model(df: pd.DataFrame, text_col: str = "message") -> pd.DataFrame:
    tqdm.pandas(mininterval=0.3)
    pipe = _get_pipeline()

    texts = df[text_col].astype(str).tolist()
    with torch.inference_mode():
        outputs = pipe(texts, batch_size=32, truncation=True)

    s = pd.Series(outputs)
    out = pd.DataFrame({
        "id": df["id"] if "id" in df.columns else range(1, len(df) + 1),
        "toxicity_label": s.apply(lambda x: x["label"]),
        "toxicity_score": s.apply(lambda x: float(x["score"])),
    })
    out["is_toxic"] = out["toxicity_label"].str.lower().isin(["label_1", "toxic"])
    return out
