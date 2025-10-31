from __future__ import annotations

import json
import os
import time
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Sequence

from src.utils import load_dotenv

from googleapiclient import discovery
from googleapiclient.errors import HttpError

load_dotenv()

PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
if not PERSPECTIVE_API_KEY:
    raise RuntimeError("PERSPECTIVE_API_KEY is not defined.")

PERSPECTIVE_DISCOVERY_URL = "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"

DEFAULT_ATTRIBUTES: Sequence[str] = (
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT",
    "SEXUALLY_EXPLICIT"
)

def save_json(scores, results_dir: Path) -> None:
    with results_dir.open("w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("The Json file must be a list of objects.")
    return obj

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
    client = _build_perspective_client(PERSPECTIVE_API_KEY)
    results: List[Dict] = []

    for idx, text in enumerate(texts):
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
                prefix = (text or "")[:40].replace("\n", " ")
                print(f"[{idx+1}] Analizing: {prefix!r}...")
                break

            except HttpError as e:
                # Possible errors from the API
                status = getattr(e, "status_code", None) or getattr(e.resp, "status", None)
                if status and int(status) in (429, 500, 502, 503, 504) and attempt < max_retries:
                    # Implemeting sleep for retraying
                    sleep_s = base_backoff * (2**attempt)
                    print(f"HTTP {status} – retrying in {sleep_s:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_s)
                    attempt += 1
                    continue                
                # Unrecoverable error or retries exhausted
                print(f"ERROR in text #{idx+1}: {e}")
                results.append({
                    "original_text": text,
                    "error": f"HttpError {status}",
                })
                break

            except Exception as e:
                print(f"ERROR in text #{idx+1}: {e}")
                results.append({
                    "original_text": text,
                    "error": str(e),
                })
                break

    return results

def _scores_to_dataframe(
    responses: List[Dict],
    attributes: Optional[Sequence[str]] = None,
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
                    row[a] = r["attributeScores"][a]["summaryScore"]["value"]
                except Exception:
                    row[a] = float("nan")
        else:
            # No attributeScores (calling error)
            if attributes:
                for a in attributes:
                    row[a] = float("nan")
        rows.append(row)

        df = pd.DataFrame(rows)

        if "TOXICITY" in df.columns:
            df[f"tox_gt_05"] = (df["TOXICITY"] >= 0.5).astype(int)
            df[f"tox_gt_08"] = (df["TOXICITY"] >= 0.8).astype(int)

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
