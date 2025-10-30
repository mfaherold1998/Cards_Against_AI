from pathlib import Path
from typing import Literal
import pandas as pd
import re

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

def write_latest_pointer(results_dir: Path, run_dir: Path) -> None:
    """
    Writes a file with the absolute path of the last run.
    """
    try:
        (results_dir / "LATEST_RUN.txt").write_text(str(run_dir.resolve()), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Couldn't write LATEST_RUN.txt: {e}")

def load_last_run_data(results_dir: Path, file_type:Literal['xlsx', 'csv'] = 'xlsx') -> Path:
    """Returns the path to the last run of models to get the csv or xlsx responses file."""
    
    last_run_file = results_dir / "LATEST_RUN.txt"
    if not last_run_file.exists():
        raise FileNotFoundError(f"File {last_run_file} not found.")
    
    lats_run_path = Path(last_run_file.read_text(encoding="utf-8").strip())
    if not lats_run_path.exists():
        raise FileNotFoundError(f"The last run folder does not exist: {lats_run_path}")
    
    file_path = lats_run_path / f"all_models_responses.{file_type}"
    if file_type == 'xlsx':
        df = pd.read_excel(file_path)
    elif file_type == 'csv':
        df = pd.read_csv(file_path, sep=',')
    else:
        raise ValueError("file_type must be 'xlsx' or 'csv'")
    
    return df