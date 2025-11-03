from pathlib import Path
from typing import Literal
import pandas as pd
import re
from enum import Enum

from dotenv import load_dotenv, find_dotenv

class PointerFile(Enum):
    """Standard filenames for trace pointers."""
    LATEST_RUN = "LATEST_RUN"
    LATEST_PROCESS = "LATEST_PROCESS"

class ResultsName(Enum):
    """Standard filenames for results files."""
    LLM_RESPONSES = "all_models_responses"
    GOOD_RESPONSES = "all_games_good_results"
    MISMATCH_RESPONSES = "all_games_mismatch"
    NO_ID_RESPONSES = "all_games_no_id_detected"
    DETOXIFY_SCORES = "all_games_detoxify_scores"
    PERSPECTIVE_SCORES = "all_games_perspective_scores"

def load_env() -> None:
    
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found)
        return

def _smart_read_file (file_root:Path | str, file_type:str):

    file_path = f"{file_root}.{file_type}"
    if file_type == 'xlsx':
        df = pd.read_excel(file_path)
    elif file_type == 'csv':
        df = pd.read_csv(file_path, sep=',')
    else:
        raise ValueError("file_type must be 'xlsx' or 'csv'")
    
    return df

def _unique_path(p: Path) -> Path:
    """If p exists, add suffixes -1, -2, ... until you find a free one."""
    if not p.exists():
        return p
    base = p.stem
    suffix = p.suffix  # normalmente vacío para carpetas
    i = 1
    while True:
        candidate = p.with_name(f"{base}-{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def demote_previous_last_runs(results_dir: Path) -> None:
    """
    Find last_run_* folders and remove the 'last_' from their names
    so they are all run_* (so there is only ONE valid last_run_*).
    """
    for d in results_dir.glob("last_run_*"):
        if d.is_dir():
            # Remove only the first 'last_' to avoid side effects
            new_name = re.sub(r"^last_", "", d.name, count=1)  # last_run_... -> run_...
            target = d.with_name(new_name)
            target = _unique_path(target)
            try:
                d.rename(target)
            except Exception as e:
                print(f"[WARN] It cannot be renamed: {d.name} -> {target.name}: {e}")

def write_latest_pointer(results_dir: Path, target_path: Path | str, pointer_type: str) -> None:
    """
    Write a pointer file with the absolute path of a completed process.
    """
    try:
        pointer_filename = f"{pointer_type}.txt"
        pointer_file = Path(f"{results_dir}/{pointer_filename}")
        content = str(target_path.resolve())
        pointer_file.write_text(content, encoding="utf-8")
    except Exception as e:
        print(f"[WARN] The pointer {pointer_type} could not be written: {e}")

def get_last_pointer_dir (results_dir:Path, pointer_type: str) -> Path:
    
    pointer_filename = f"{pointer_type}.txt"
    last_process_file = Path(f"{results_dir}/{pointer_filename}")
    if not last_process_file.exists():
        raise FileNotFoundError(f"File {last_process_file} not found.")
    
    rund_dir = Path(last_process_file.read_text(encoding="utf-8").strip())
    
    if not rund_dir.exists():
        raise FileNotFoundError(f"The last process folder does not exist: {rund_dir}")
    
    return rund_dir

def load_last_data(last_dir: Path, file_name:ResultsName, file_type:Literal['xlsx', 'csv'] = 'xlsx') -> Path:
    """Returns the path to the last  csv or xlsx responses file to get the data."""
    
    if not last_dir.exists():
        raise FileNotFoundError(f"Folder {last_dir} not found.")
    
    file_root = last_dir / file_name
    df = _smart_read_file(file_root,file_type)
    
    return df


