from src.logging import create_logger
logger = create_logger (log_name="main")

logger.info("Importing Libraries")
from pathlib import Path
from datetime import datetime
import csv

from src.args_parser import get_args
from src.utils import get_last_pointer_dir, load_last_data, PointerFile, ResultsName

logger.info("Parsing config.json file to get parameters...")

config_params = get_args()
results_dir = Path(config_params.get("results_dir", "./results"))

# Get latest processing directory
RUN_DIR = get_last_pointer_dir(results_dir, PointerFile.LATEST_PROCESS.value)
logger.debug(f"Current process folder: {RUN_DIR}")

file_to_process_name = ResultsName.GOOD_RESPONSES.value
file_type = config_params.get("file_type", "xlsx")

logger.info(f"Loading data: {file_to_process_name}...")

df_responses = load_last_data(RUN_DIR, file_to_process_name, file_type)
if 'sentence' not in df_responses.columns:
    raise KeyError(f"There is not 'sentence' column in the file: {list(df_responses.columns)}")

logger.info(f"Loaded {len(df_responses)} rows.")