from pathlib import Path
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
            # Quita solo el primer 'last_' para evitar efectos colaterales
            new_name = re.sub(r"^last_", "", d.name, count=1)  # last_run_... -> run_...
            target = d.with_name(new_name)
            target = _unique_path(target)
            try:
                d.rename(target)
            except Exception as e:
                print(f"[WARN] No pude renombrar {d.name} -> {target.name}: {e}")


def write_latest_pointer(results_dir: Path, run_dir: Path) -> None:
    """
    Writes a file with the absolute path of the last run.
    Useful for other modules to quickly determine where to read.
    """
    try:
        (results_dir / "LATEST_RUN.txt").write_text(str(run_dir.resolve()), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Couldn't write LATEST_RUN.txt: {e}")


def get_last_run_path_csv(results_dir: Path) -> Path:
    """Returns the path to the last run of models responses csv file."""
    
    latest_file = results_dir / "LATEST_RUN.txt"
    if not latest_file.exists():
        raise FileNotFoundError(f"File {latest_file} not found.")
    
    latest_run_path = Path(latest_file.read_text(encoding="utf-8").strip())
    if not latest_run_path.exists():
        raise FileNotFoundError(f"The last run folder does not exist: {latest_run_path}")
    
    return latest_run_path / "all_responses_results.xlsx"