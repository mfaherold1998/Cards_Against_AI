from __future__ import annotations
from typing import Dict, Iterable, List, Optional
import numpy as np
import pandas as pd
import torch
from detoxify import Detoxify

DEFAULT_EXPECTED = [
    "toxicity", "severe_toxicity", "obscene", "threat",
    "insult", "identity_attack", "sexual_explicit"
]
DEFAULT_ALIASES = {"sexually_explicit": "sexual_explicit"}

def get_available_labels(model: Detoxify, aliases: Dict[str, str]) -> set:

    """
    Query the checkpoint for the tags it actually returns and normalize them with aliases.
    """

    # Run a test prediction (["probe"]) to find out which keys the loaded checkpoint returns.
    with torch.no_grad():
        probe = model.predict(["probe"]) 
    # normalizes names with aliases
    return {aliases.get(k, k) for k in probe.keys()}

def detoxify_batch(
    texts: List[str],
    model: Detoxify,
    tags: Iterable[str],
    aliases: Dict[str, str],
    batch_size: int = 64
    ) -> Dict[str, np.ndarray]:
    
    """
    Runs Detoxify in batches and returns a dict{label: np.array} aligned with texts.
    """
    
    # Input sentences
    clean = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    n_rows = len(clean)
    # Dict of scores
    out: Dict[str, np.ndarray] = {k: np.full(n_rows, np.nan, dtype=np.float32) for k in tags}

    with torch.no_grad():
        for i in range(0, n_rows, batch_size):
            chunk = clean[i:i+batch_size]
            preds = model.predict(chunk)  
            # normalizes names with aliases
            preds = {aliases.get(k, k): v for k, v in preds.items()}
            for k in tags:
                if k in preds:  # only write if category exists at this checkpoint
                    out[k][i:i+len(chunk)] = np.asarray(preds[k], dtype=np.float32)
    return out

# Detoxify Model => [original (english), unbiased, multilingual]
def add_detoxify_scores(
    df: pd.DataFrame,
    text_col: str,
    model: str | Detoxify = "original",
    batch_size: int = 64,
    prefix: str = "detox_",
    expected: Optional[Iterable[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
    device: Optional[str] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    
    """
    Adds toxicity columns to df based on the `text_col` column.
    - 'model': Checkpoint name ("original", "unbiased", "multilingual") or already created Detoxify instance.
    - 'expected'/ 'aliases': Lists of expected tags and optional mappings.
    - 'inplace': If False, works on a copy of the DataFrame.
    """
    
    assert text_col in df.columns, f"Column '{text_col}' is missing"

    if not inplace:
        df = df.copy()

    if isinstance(model, Detoxify):
        tox_model = model
    else:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tox_model = Detoxify(model, device=device)

    aliases = aliases or DEFAULT_ALIASES
    expected = list(expected or DEFAULT_EXPECTED)
    available = get_available_labels(tox_model, aliases)
    
    targets = [t for t in expected if t in available]
    if not targets:
        raise RuntimeError(
            f"No matching labels. Available from model: {sorted(available)}; expected: {sorted(expected)}"
        )

    scores = detoxify_batch(df[text_col].astype(str).tolist(), tox_model, targets, aliases, batch_size=batch_size)

    for k, arr in scores.items():
        df[f"{prefix}{k}"] = np.clip(arr, 0.0, 1.0)

    tox_col = f"{prefix}toxicity"
    if tox_col in df.columns:
        df[f"{prefix}tox_gt_05"] = (df[tox_col] >= 0.5).astype(int)
        df[f"{prefix}tox_gt_08"] = (df[tox_col] >= 0.8).astype(int)

    return df
