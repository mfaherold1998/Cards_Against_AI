from src.logging import create_logger
logger = create_logger (log_name="main")

logger.info("Importing Libraries")

from pathlib import Path
import csv
from src.args_parser import get_args
from src.toxicity_detox import add_detoxify_scores
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

logger.info("Clasifying Toxicity with Detoxify (local clasifier)...")
logger.info("Adding scores to sentences...")

device = config_params.get("device", "cpu")
batch = config_params.get("batch", 64)

df_detoxify_scores_responses = add_detoxify_scores(
    df=df_responses, 
    text_col='sentence', 
    model=config_params.get("detoxify_model", "original"),
    device=device,
    batch_size=batch)

df_detoxify_scores_combinations = add_detoxify_scores(
    df=df_all_combinations, 
    text_col='sentence', 
    model=config_params.get("detoxify_model", "original"),
    device=device,
    batch_size=batch)

# Remove columns of NAN values in case some category is not present
df_detoxify_scores_responses = df_detoxify_scores_responses.dropna(axis=1, how='all')
df_detoxify_scores_combinations = df_detoxify_scores_combinations.dropna(axis=1, how='all')

logger.info(f"Saving results in {RUN_DIR.resolve()}...")

detoxify_scores_xlsx_path = RUN_DIR / f"{ResultsName.DETOXIFY_SCORES.value}.xlsx"
all_combinations_xlsx_path = RUN_DIR / f"{ResultsName.DETOXIFY_SCORES.value}_all_combinations.xlsx"
#detoxify_scores_csv_path  = RUN_DIR / f"{ResultsName.DETOXIFY_SCORES.value}.csv"
df_detoxify_scores_responses.to_excel(detoxify_scores_xlsx_path, index=False, header=True, sheet_name="toxicity_scores")
df_detoxify_scores_combinations.to_excel(all_combinations_xlsx_path, index=False, header=True, sheet_name="toxicity_scores")
#df_detoxify_scores.to_csv(detoxify_scores_csv_path, index=False, quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')

logger.info("END")
