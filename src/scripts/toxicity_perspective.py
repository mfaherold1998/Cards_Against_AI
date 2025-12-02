from __future__ import annotations

import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Iterable, List, Dict, Optional, Sequence

from src.utils.utils import ToxicityAttributes
from dotenv import load_dotenv

from googleapiclient import discovery
from googleapiclient.errors import HttpError

from src.utils.logging import create_logger
logger = create_logger (log_name="main")

load_dotenv()

PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
if not PERSPECTIVE_API_KEY:
    error_message = "PERSPECTIVE_API_KEY is not defined."
    logger.error(error_message)
    raise RuntimeError(error_message)

PERSPECTIVE_DISCOVERY_URL = "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"

DEFAULT_ATTRIBUTES = [i.value.upper() for i in ToxicityAttributes]

def _build_perspective_client(api_key: str):
    """
    Create the Perspective client (static discovery).
    """
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl=PERSPECTIVE_DISCOVERY_URL,
        static_discovery=False,
        cache_discovery=False,
    )
    return client

def analyze_texts(
    texts: Iterable[str],
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    langs: Optional[Sequence[str]] = None,
    max_retries: int = 5,
    base_backoff: float = 1.0,
) -> List[Dict]:
    """
    Parses a text sequence and returns a list of dicts with:
    - original_text
    - attributeScores (each attribute with summaryScore.value)
    - requestedAttributes, languages ​​(echo)
    With exponential retries for 429/5xx.
    """
    batch_size = 60
    wait_time = 90

    client = _build_perspective_client(PERSPECTIVE_API_KEY)
    results: List[Dict] = []

    texts_list = list(texts)
    num_texts = len(texts_list)

    for batch_start in tqdm(range(0, num_texts, batch_size), desc="Analazing elements", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]", colour="green", ascii=True):
        
        batch_end = min(batch_start + batch_size, num_texts)
        current_batch = texts_list[batch_start:batch_end]

        for text_idx_in_batch, text in enumerate(current_batch):

            global_idx = batch_start + text_idx_in_batch
        
            analyze_request = {
                "comment": {"text": str(text) if text is not None else ""},
                "requestedAttributes": {attr: {} for attr in attributes},
            }
            if langs:
                analyze_request["languages"] = list(langs)

            attempt = 0
            
            while True:
                try:
                    # Get the scores from Perspective per sentence
                    res = client.comments().analyze(body=analyze_request).execute()
                    # Include original text in response for traceability
                    res["original_text"] = text
                    res["requestedAttributes"] = list(attributes)
                    if langs:
                        res["languages"] = list(langs)
                    results.append(res)
                    
                    # Preparing to Logging
                    #prefix = (text or "")[:40].replace("\n", " ")
                    #print(f"[{idx+1}] Analizing: {prefix!r}...")
                    break

                except HttpError as e:
                    # Possible errors from the API
                    status = getattr(e, "status_code", None) or getattr(e.resp, "status", None)
                    if status and int(status) in (429, 500, 502, 503, 504) and attempt < max_retries:
                        # Implemeting sleep for retraying
                        sleep_s = base_backoff * (2**attempt)
                        logger.info(f"HTTP {status} – retrying in {sleep_s:.1f}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_s)
                        attempt += 1
                        continue                
                    # Unrecoverable error or retries exhausted
                    logger.error(f"Error in text #{global_idx+1}: {e}")
                    results.append({
                        "original_text": text,
                        "error": f"HttpError {status}",
                    })
                    break

                except Exception as e:
                    logger.error(f"Error in text #{global_idx+1}: {e}")
                    results.append({
                        "original_text": text,
                        "error": str(e),
                    })
                    break
        
        if batch_end < num_texts:
            logger.info(f"Batch {len(current_batch)} complete. Waiting {wait_time} seconds to avoid API limits...")
            time.sleep(wait_time)

    return results

def _scores_to_dataframe(
    responses: List[Dict],
    attributes: Optional[List] = [],
    text_col: str = "original_text",
) -> pd.DataFrame:
    """
    Converts the Perspective response list to a flat DataFrame:
    columns = [text_col] + attributes.
    If there are errors, set NaNs to the corresponding attributes.
    """
    rows = []
    for r in responses:
        row = {text_col: r.get("original_text")}
        if "attributeScores" in r and isinstance(r["attributeScores"], dict):
            attrs = attributes or list(r["attributeScores"].keys())
            for a in attrs:
                try:
                    row[a.lower()] = r["attributeScores"][a]["summaryScore"]["value"]
                except Exception:
                    row[a.lower()] = float("nan")
        else:
            # No attributeScores (calling error)
            if attributes:
                for a in attributes:
                    row[a] = float("nan")
        rows.append(row)

        df = pd.DataFrame(rows)

        TOXICITY = ToxicityAttributes.SEVERE_TOXICITY.value.lower()
        if TOXICITY in df.columns:
            df[f"{TOXICITY}_gt_05"] = (df[TOXICITY] >= 0.5).astype(int)
            df[f"{TOXICITY}_gt_08"] = (df[TOXICITY] >= 0.8).astype(int)

    return df

def add_perspective_scores(
    df: pd.DataFrame,
    responses: List[Dict],
    text_col: str = "sentence",
) -> pd.DataFrame:
    """
    Concatenate the scores assuming the same order.
    """
    df_scores = _scores_to_dataframe(responses, text_col=text_col)

    # If lengths match, concatenate by index
    if len(df_scores) == len(df):
        return pd.concat([df.reset_index(drop=True), df_scores.drop(columns=[text_col], errors="ignore")], axis=1)

    # Fallback: merge by text
    return df.merge(df_scores, on=text_col, how="left")

