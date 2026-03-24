import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

MODEL_NAME = "garak_llm_roberta_toxicity_classifier"

_pipe = None
def get_pipeline(device: torch.device | None = None):
    global _pipe
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_device = 0 if (device.type == "cuda") else -1
    if _pipe is None:
        _pipe = pipeline(
            "text-classification",
            model="garak-llm/roberta_toxicity_classifier",
            tokenizer="garak-llm/roberta_toxicity_classifier",
            device=hf_device,
            torch_dtype=torch.float16 if hf_device >= 0 else None,
            truncation=True,
        )
    return _pipe

def run_model(df: pd.DataFrame, text_col: str = "message", device: torch.device | None = None) -> pd.DataFrame:
    tqdm.pandas()
    pipe = get_pipeline(device)
    preds = df[text_col].astype(str).progress_apply(lambda x: pipe(x)[0])
    out = pd.DataFrame({
        "id": df["id"],
        "toxicity_label": preds.apply(lambda x: x["label"]),
        "toxicity_score": preds.apply(lambda x: float(x["score"])),
    })
    out["is_toxic"] = out["toxicity_label"].isin(["LABEL_1", "toxic"])
    return out
