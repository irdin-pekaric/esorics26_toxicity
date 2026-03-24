import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
from transformers.utils import logging as hf_logging

MODEL_NAME = "nicholas_kluge_toxicity_model"
THRESHOLD = 0.5

hf_logging.set_verbosity_error()
_pipe = None

def _get_pipeline():
    global _pipe
    if _pipe is None:
        device = 0 if torch.cuda.is_available() else -1
        print("Device:", "CUDA" if device == 0 else "CPU")
        _pipe = pipeline(
            "text-classification",
            model="nicholasKluge/ToxicityModel",
            tokenizer="nicholasKluge/ToxicityModel",
            device=device,
            truncation=True,
        )
    return _pipe

def _is_label_toxic(label_lower: str) -> bool:
    if "toxic" in label_lower and ("non" not in label_lower and "not" not in label_lower):
        return True
    if label_lower in {"label_1"}:
        return True
    return False

def run_model(df: pd.DataFrame, text_col: str = "message") -> pd.DataFrame:
    tqdm.pandas(mininterval=0.3)
    pipe = _get_pipeline()

    texts = df[text_col].astype(str).tolist()
    with torch.inference_mode():
        outputs = pipe(texts, batch_size=32, truncation=True)  # list of {"label","score"}

    labels = []
    tox_probs = []
    for o in outputs:
        lab = str(o["label"])
        score = float(o["score"])
        lab_lower = lab.lower()

        if _is_label_toxic(lab_lower):
            p_toxic = score
        elif "toxic" in lab_lower:
            p_toxic = 1.0 - score
        else:
            p_toxic = 1.0 - score

        labels.append(lab)
        tox_probs.append(p_toxic)

    out = pd.DataFrame({
        "id": df["id"] if "id" in df.columns else range(1, len(df) + 1),
        "toxicity_label": labels,
        "toxicity_score": tox_probs,
    })
    out["is_toxic"] = out["toxicity_score"] >= THRESHOLD
    return out
