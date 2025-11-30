from __future__ import annotations
from typing import Dict, Iterable, List, Optional
import numpy as np
import pandas as pd
import torch
from detoxify import Detoxify
from src.utils.utils import ToxicityAttributes

from src.utils.logging import create_logger
logger = create_logger (log_name="main")

DEFAULT_EXPECTED = [i.value for i in ToxicityAttributes]
DEFAULT_ALIASES = {"sexually_explicit": "sexual_explicit"}

def _get_available_labels(model: Detoxify, aliases: Dict[str, str]) -> set:

    """
    Query the checkpoint for the tags it actually returns and normalize them with aliases.
    """

    # Run a test prediction (["probe"]) to find out which keys the loaded checkpoint returns.
    with torch.no_grad():
        probe = model.predict(["probe"]) 
    # normalizes names with aliases
    return {aliases.get(k, k) for k in probe.keys()}

def _detoxify_batch(
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
    model: str | Detoxify,
    device: Optional[str] = None,
    batch_size: int = 64,
    prefix: str = "detox_",
    expected: Optional[Iterable[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    
    """
    Adds toxicity columns to df based on the `text_col` column.
    - 'model': Checkpoint name ("original", "unbiased", "multilingual") or already created Detoxify instance.
    - 'expected'/ 'aliases': Lists of expected tags and optional mappings.
    - 'inplace': If False, works on a copy of the DataFrame.
    """

    df_temp = df.copy()
    
    # Check that the column with the sentences exists
    assert text_col in df_temp.columns, logger.error(f"Column '{text_col}' is missing")

    #Set the Detoxify model
    if isinstance(model, Detoxify):
        tox_model = model
    else:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tox_model = Detoxify(model, device=device)

    # Determining the Deroxify model tags
    aliases = aliases or DEFAULT_ALIASES
    expected = list(expected or DEFAULT_EXPECTED)
    available = _get_available_labels(tox_model, aliases)
    targets = [t for t in expected if t in available]    
    if not targets:
        error_message = f"No matching labels. Available from model: {sorted(available)}; expected: {sorted(expected)}"
        logger.error(error_message)
        raise RuntimeError(error_message)

    # Getting scores
    scores = _detoxify_batch(df_temp[text_col].astype(str).tolist(), tox_model, targets, aliases, batch_size=batch_size)

    for k, arr in scores.items():
        df_temp[k] = np.clip(arr, 0.0, 1.0)

    TOXICITY = ToxicityAttributes.SEVERE_TOXICITY.value
    if TOXICITY in df_temp.columns:
        df_temp[f"{TOXICITY}_gt_05"] = (df_temp[TOXICITY] >= 0.5).astype(int)
        df_temp[f"{TOXICITY}_gt_08"] = (df_temp[TOXICITY] >= 0.8).astype(int)

    return df_temp
