from src.logging import create_logger
logger = create_logger (log_name="main")

logger.info("Importing Libraries")
from pathlib import Path
import csv

from src.args_parser import get_args
from src.analysis import *
from src.utils import get_last_pointer_dir, load_last_data, PointerFile, ResultsName

logger.info("Parsing config.json file to get parameters...")

config_params = get_args()
results_dir = Path(config_params.get("results_dir", "./results"))

# Get latest processing directory
RUN_DIR = get_last_pointer_dir(results_dir, PointerFile.LATEST_PROCESS.value)
logger.debug(f"Current process folder: {RUN_DIR}")

# Load models responses
file_to_process_name = ResultsName.GOOD_RESPONSES.value
file_type = config_params.get("file_type", "xlsx")

logger.info(f"Loading data: {file_to_process_name}...")
df_responses = load_last_data(RUN_DIR, file_to_process_name, file_type)
logger.info(f"Loaded {len(df_responses)} rows.")

file_to_process_name = f"{ResultsName.DETOXIFY_SCORES.value}_all_combinations"
logger.info(f"Loading data: {file_to_process_name}...")
df_all_combinations_detox = load_last_data(RUN_DIR, file_to_process_name, file_type)
logger.info(f"Loaded {len(df_all_combinations_detox)} rows.")

file_to_process_name = f"{ResultsName.PERSPECTIVE_SCORES.value}_all_combinations"
logger.info(f"Loading data: {file_to_process_name}...")
df_all_combinations_perspective = load_last_data(RUN_DIR, file_to_process_name, file_type)
logger.info(f"Loaded {len(df_all_combinations_perspective)} rows.")

# Initializing analisis
logger.info("Analyzing the frequency of model choices (how many times a card is chosen compared to how many times it is drawn)")

df_freq = frequencies_election(df_responses)

logger.info("Analyzing the inconsistency of the models' choices (how many times they choose different cards for the same play)")

df_incons = inconcistencies_elections(df_responses)

logger.info("Calculating the delta toxicity (how much more toxic are the options that didn't win compared to the one that did)")

df_delta_tox_detox = relative_toxicity_comparison(df_all_combinations_detox)
df_delta_tox_perspective = relative_toxicity_comparison(df_all_combinations_perspective)

# Saving results

logger.info(f"Saving results in {RUN_DIR.resolve()}...")

freq_xlsx_path = RUN_DIR / f"{ResultsName.ELECTION_FREQ.value}.xlsx"
incons_path = RUN_DIR / f"{ResultsName.ELECTION_INCONSISTENCY.value}.xlsx"
comb_path = RUN_DIR / f"{ResultsName.DELTA_TOX.value}_detox.xlsx"
perspective_path = RUN_DIR / f"{ResultsName.DELTA_TOX.value}_perspective.xlsx"

df_freq.to_excel(freq_xlsx_path, index=False, header=True, sheet_name="frequencies")
df_incons.to_excel(incons_path, index=False, header=True, sheet_name="inconsistencies")
df_delta_tox_detox.to_excel(comb_path, index=False, header=True, sheet_name="delta_tox")
df_delta_tox_perspective.to_excel(perspective_path, index=False, header=True, sheet_name="delta_tox")

logger.info("END")