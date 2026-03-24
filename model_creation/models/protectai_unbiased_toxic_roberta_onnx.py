from __future__ import annotations

from pathlib import Path
import os
from typing import Tuple, List, Iterable

import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

MODEL_NAME = "protectai_unbiased_toxic_roberta_onnx"

MAX_LEN = 128
BATCH_SIZE = 32
THRESHOLD = float(os.environ.get("ONNX_THRESHOLD", 0.5))  # use 0.5 by default to match expected behavior

_DEFAULT_ROOT = (Path(__file__).resolve().parents[1] / "models" / "unbiased-toxic-roberta-onnx").resolve()
_DEFAULT_ONNX_PATH = _DEFAULT_ROOT / "model.onnx"

_DEFAULT_REPO_ID = os.environ.get("ONNX_REPO_ID", "protectai/unbiased-toxic-roberta-onnx")

# If a local tokenizer is not present in the ONNX snapshot, prefer the unitary tokenizer
_FALLBACK_TOKENIZER_ID = "unitary/unbiased-toxic-roberta"
_FALLBACK_MODEL_ID = os.environ.get("ONNX_FALLBACK_MODEL", "protectai/unbiased-toxic-roberta-onnx")

_session = None
_tokenizer = None
_input_names: List[str] | None = None
_toxic_idx = 1
_cfg_source = "default"

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _find_first_onnx(root: Path) -> Path | None:
    candidates = list(root.rglob("*.onnx"))
    return candidates[0] if candidates else None

def _download_snapshot(repo_id: str, local_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            f"[{MODEL_NAME}] 'huggingface_hub' is required for auto-download.\n"
            f"Install with: pip install huggingface_hub"
        ) from e

    local_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        "*.onnx",
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "config.json",
        "special_tokens_map.json",
    ]
    print(f"[{MODEL_NAME}] Downloading from hf://{repo_id} to {local_dir} …")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )

def _ensure_assets(onnx_path: Path, local_root: Path, repo_id: str) -> Tuple[Path, Path]:
    onnx_ok = onnx_path.exists()
    tok_ok = local_root.exists() and any(
        (local_root / f).exists()
        for f in ("tokenizer.json", "vocab.txt", "vocab.json", "merges.txt")
    )

    if not (onnx_ok and tok_ok):
        print(f"[{MODEL_NAME}] Assets missing → attempting to download '{repo_id}' …")
        _download_snapshot(repo_id, local_root)

    if not onnx_path.exists():
        discovered = _find_first_onnx(local_root)
        if not discovered:
            raise FileNotFoundError(
                f"[{MODEL_NAME}] Downloaded repo but no .onnx file was found under {local_root}"
            )
        onnx_path = discovered

    print(f"[{MODEL_NAME}] ✅ Assets ready at onnx={onnx_path}, tokenizer_dir={local_root}")
    return onnx_path, local_root

def _resolve_onnx_path() -> Path:
    env_path = os.environ.get("ONNX_MODEL_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    if _DEFAULT_ONNX_PATH.exists():
        return _DEFAULT_ONNX_PATH

    discovered = _find_first_onnx(_DEFAULT_ROOT)
    if discovered:
        return discovered

    return _DEFAULT_ONNX_PATH

def _pick_providers() -> List[str]:
    available = ort.get_available_providers()
    return ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

def _prep_inputs(texts: Iterable[str], tokenizer: AutoTokenizer, input_names: List[str]) -> dict:
    # Use tokenizer exactly like the original working script and pass all outputs through to ONNX
    enc = tokenizer(
        list(texts),
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    # Convert arrays to int64 where appropriate, but pass all keys (some ONNX snapshots expect token_type_ids etc.)
    ort_inputs = {k: (v.astype(np.int64) if hasattr(v, 'dtype') and v.dtype != np.int64 and v.dtype.kind in ('i','u') else v) for k, v in enc.items()}
    return ort_inputs

def _batch(iterable: List[str], n: int):
    for i in range(0, len(iterable), n):
        yield i, iterable[i:i + n]

def _get_session_tokenizer_and_label():
    global _session, _tokenizer, _input_names, _toxic_idx, _cfg_source
    if _session is not None:
        return _session, _tokenizer, _input_names, _toxic_idx

    onnx_path = _resolve_onnx_path()
    onnx_path, local_root = _ensure_assets(onnx_path, _DEFAULT_ROOT, _DEFAULT_REPO_ID)

    providers = _pick_providers()
    _session = ort.InferenceSession(str(onnx_path), providers=providers)
    _input_names = [i.name for i in _session.get_inputs()]
    print(f"[{MODEL_NAME}] ONNX: {onnx_path.name} | providers: {providers} | inputs: {_input_names}")

    has_local_tok = any((local_root / f).exists() for f in ["tokenizer.json", "vocab.json", "vocab.txt", "merges.txt", "config.json"])
    if has_local_tok:
        _tokenizer = AutoTokenizer.from_pretrained(local_root)
        try:
            cfg = AutoConfig.from_pretrained(local_root)
            _cfg_source = "onnx_snapshot"
        except Exception:
            # fallback to model id config if snapshot doesn't include config
            try:
                cfg = AutoConfig.from_pretrained(_FALLBACK_MODEL_ID)
                _cfg_source = f"fallback: {_FALLBACK_MODEL_ID}"
            except Exception:
                cfg = None
                _cfg_source = "onnx_snapshot (no config)"
    else:
        # prefer the unitary tokenizer for compatibility with earlier working code
        try:
            _tokenizer = AutoTokenizer.from_pretrained(_FALLBACK_TOKENIZER_ID)
            cfg = AutoConfig.from_pretrained(_FALLBACK_TOKENIZER_ID)
            _cfg_source = f"hub: {_FALLBACK_TOKENIZER_ID}"
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(_FALLBACK_MODEL_ID)
            cfg = AutoConfig.from_pretrained(_FALLBACK_MODEL_ID)
            _cfg_source = f"hub: {_FALLBACK_MODEL_ID}"

    try:
        id2label = getattr(cfg, "id2label", None) or {}
        id2label = {int(k): str(v).lower() for k, v in id2label.items()} if id2label else {}
        toxic_idx = None
        for k, v in id2label.items():
            if "toxic" in v and "non" not in v:
                toxic_idx = k
                break
        # If we couldn't find mapping, assume binary single-logit or two-class with toxic at index 1
        if toxic_idx is None:
            toxic_idx = 1
        _toxic_idx = int(toxic_idx)
    except Exception:
        _toxic_idx = 1

    print(f"[{MODEL_NAME}] tokenizer: {_cfg_source} | toxic_idx={_toxic_idx}")
    return _session, _tokenizer, _input_names, _toxic_idx

def _extract_first_logit(onnx_outputs) -> float:
    """Extract a single logit value from ONNX outputs similar to original script's ort_outputs[0][0][0]."""
    try:
        out0 = onnx_outputs[0]
        arr = np.asarray(out0)
        flat = arr.flatten()
        return float(flat[0])
    except Exception:
        # fallback
        try:
            return float(np.asarray(onnx_outputs).flatten()[0])
        except Exception:
            return 0.0

def run_model(df: pd.DataFrame, text_col: str = "message", per_sample: bool = True) -> pd.DataFrame:
    """Run the ONNX model. By default use per-sample calls to mirror the user's original script.

    If per_sample=False, uses batched execution (faster) with shape-aware logic.
    """

    session, tokenizer, input_names, toxic_idx = _get_session_tokenizer_and_label()

    ids = df["id"] if "id" in df.columns else pd.Series(np.arange(1, len(df) + 1, dtype=int), name="id")
    texts = df[text_col].astype(str).tolist()

    if per_sample:
        probs = np.zeros(len(texts), dtype=np.float32)
        for i, t in enumerate(tqdm(texts, desc=f"[{MODEL_NAME}] per-sample", unit="it", mininterval=0.3)):
            enc = tokenizer(str(t), return_tensors="np", padding="max_length", truncation=True, max_length=MAX_LEN)
            ort_inputs = {k: v for k, v in enc.items()}
            try:
                onnx_outs = session.run(None, ort_inputs)
            except Exception as e:
                # fallback to zeros
                probs[i] = 0.0
                continue
            logit = _extract_first_logit(onnx_outs)
            probs[i] = _sigmoid(np.array(logit, dtype=float))
    else:
        # batched path (existing behaviour)
        probs = np.zeros(len(texts), dtype=np.float32)
        for i, chunk in _batch(texts, BATCH_SIZE):
            ort_inputs = _prep_inputs(chunk, tokenizer, input_names)
            logits = session.run(None, ort_inputs)[0]

            if logits.ndim == 2:
                if logits.shape[1] == 1:
                    p_toxic = _sigmoid(logits[:, 0])
                else:
                    p = _softmax(logits, axis=1)
                    idx = toxic_idx if toxic_idx < p.shape[1] else min(1, p.shape[1] - 1)
                    p_toxic = p[:, idx]
            elif logits.ndim == 1:
                p_toxic = _sigmoid(logits)
            else:
                p_toxic = _sigmoid(logits.reshape(len(chunk), -1)[:, 0])

            probs[i:i + len(chunk)] = p_toxic.astype(np.float32)

    pct_pos_05 = float((probs >= 0.5).mean())
    pct_pos_T = float((probs >= THRESHOLD).mean())
    print(
        f"[{MODEL_NAME}] probs: min={probs.min():.3f} mean={probs.mean():.3f} max={probs.max():.3f} | "
        f">=0.5: {pct_pos_05:.1%} | >=THR({THRESHOLD}): {pct_pos_T:.1%}"
    )

    out = pd.DataFrame({
        "id": ids.values,
        "toxicity_score": probs.astype(float),
    })
    out["is_toxic"] = out["toxicity_score"] >= THRESHOLD
    out["toxicity_label"] = out["is_toxic"].map({True: "toxic", False: "non_toxic"})
    return out
