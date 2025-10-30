from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Sequence

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from googleapiclient import discovery
from googleapiclient.errors import HttpError

from src.utils import load_last_data

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

def _load_env_once() -> None:
    
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found)
        return

cwd_path = Path(os.getcwd())
variables_env_path = cwd_path.parent / '.env'

_load_env_once()

PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
if not PERSPECTIVE_API_KEY:
    raise RuntimeError("PERSPECTIVE_API_KEY is not defined.")


def load_df_texts(results_dir) -> pd.DataFrame :
    df = load_last_data(results_dir, name_file ='configs')
    if 'sentence' not in df.columns:
        raise KeyError(f"There is not 'sentence' collumn in the file: {list(df.columns)}")
    return df

def save_json(scores, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_perspective_client(api_key: str):
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
    languages: Optional[Sequence[str]] = None,
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
    client = build_perspective_client(PERSPECTIVE_API_KEY)
    results: List[Dict] = []

    for idx, text in enumerate(texts):
        analyze_request = {
            "comment": {"text": str(text) if text is not None else ""},
            "requestedAttributes": {attr: {} for attr in attributes},
        }
        if languages:
            analyze_request["languages"] = list(languages)

        attempt = 0
        
        while True:
            try:
                resp = client.comments().analyze(body=analyze_request).execute()
                resp["original_text"] = text
                resp["requestedAttributes"] = list(attributes)
                if languages:
                    resp["languages"] = list(languages)
                results.append(resp)
                # Log
                prefix = (text or "")[:40].replace("\n", " ")
                print(f"[{idx+1}] Analizing: {prefix!r}...")
                break
            except HttpError as e:
                status = getattr(e, "status_code", None) or getattr(e.resp, "status", None)
                if status and int(status) in (429, 500, 502, 503, 504) and attempt < max_retries:
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

def scores_to_dataframe(
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
    return pd.DataFrame(rows)

def attach_perspective_scores(
    df: pd.DataFrame,
    responses: List[Dict],
    text_col: str = "sentence",
) -> pd.DataFrame:
    """
    Concatenate the scores assuming the same order.
    """
    df_scores = scores_to_dataframe(responses, text_col=text_col)

    # If lengths match, concatenate by index
    if len(df_scores) == len(df):
        return pd.concat([df.reset_index(drop=True), df_scores.drop(columns=[text_col], errors="ignore")], axis=1)

    # Fallback: merge by text
    return df.merge(df_scores, on=text_col, how="left")

def save_perspective_responses(responses: List[Dict], out_path: Path) -> None:
    """
    Save the list of dicts in JSON.
    """
    save_json(responses, out_path)

def load_perspective_responses(path: Path) -> List[Dict]:
    obj = load_json(path)
    if not isinstance(obj, list):
        raise ValueError("The Json file must be a list of objects.")
    return obj
