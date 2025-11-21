from src.logging import create_logger
logger = create_logger (log_name="main")

logger.info("Importing Libraries")

from pathlib import Path
import csv
import time
from src.args_parser import get_args
from src.toxicity_perspective import analyze_texts, add_perspective_scores
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

file_to_process_name = ResultsName.ALL_POSIBLE_COMBINATIONS.value

logger.info(f"Loading data: {file_to_process_name}...")

df_all_combinations = load_last_data(RUN_DIR, file_to_process_name, file_type)
if 'sentence' not in df_responses.columns:
    raise KeyError(f"There is not 'sentence' column in the file: {list(df_responses.columns)}")

logger.info(f"Loaded {len(df_all_combinations)} rows.")

logger.info("Clasifying Toxicity with Perspective (Google clasifier)...")
logger.info("Adding scores to sentences...")

# Getting the responses from the API -> List[Dict]
perspectives_scores_responses = analyze_texts(df_responses["sentence"])
wait_time_between_files = 90 
logger.info(f"First file processed. Waiting {wait_time_between_files} seconds to ensure API limits reset before processing the second file.")
time.sleep(wait_time_between_files)
perspectives_scores_comb = analyze_texts(df_all_combinations["sentence"])

df_perspectives_scores = add_perspective_scores(df_responses, perspectives_scores_responses, text_col="sentence")
df_perspectives_scores_comb = add_perspective_scores(df_all_combinations, perspectives_scores_comb, text_col="sentence")

logger.info(f"Saving results in {RUN_DIR.resolve()}...")

perspective_scores_xlsx_path = RUN_DIR / f"{ResultsName.PERSPECTIVE_SCORES.value}.xlsx"
perspective_scores_comb_path = RUN_DIR / f"{ResultsName.PERSPECTIVE_SCORES.value}_all_combinations.xlsx"
#perspective_scores_csv_path  = RUN_DIR / f"{ResultsName.PERSPECTIVE_SCORES.value}.csv"
df_perspectives_scores.to_excel(perspective_scores_xlsx_path, index=False, header=True, sheet_name="toxicity_scores")
df_perspectives_scores_comb.to_excel(perspective_scores_comb_path, index=False, header=True, sheet_name="toxicity_scores")
#df_perspectives_scores.to_csv(perspective_scores_csv_path, index=False, quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')

logger.info("END")
