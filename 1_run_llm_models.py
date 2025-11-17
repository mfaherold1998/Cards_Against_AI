from src.logging import create_logger
logger = create_logger (log_name="main")

logger.info("Importing Libraries")
from pathlib import Path
from datetime import datetime
import json
import csv

from src.args_parser import get_args
from src.data_loader import load_cards, load_games
from src.model_runner import run_models
from src.utils import demote_previous_last_runs, write_latest_pointer, PointerFile, ResultsName

logger.info("Parsing config.json file to get parameters...")

config_params = get_args()

# Set or create results folder path
results_dir = Path(config_params.get("results_dir", "./results"))
results_dir.mkdir(parents=True, exist_ok=True)

# Remove the 'last_' prefix from previous executions to uniquely mark the last one
demote_previous_last_runs(results_dir)  # From utils.py

# Create the last_run directory
date_tag = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
dir_name = f"last_run_{date_tag}"
RUN_DIR = results_dir / dir_name
RUN_DIR.mkdir(parents=True, exist_ok=True)

logger.debug(f"Current run folder: {dir_name}")

# Write a pointer to the path of the last run
write_latest_pointer(results_dir, RUN_DIR, PointerFile.LATEST_RUN.value)  # From utils.py

# Save used configuration in RUN_DIR
with open(RUN_DIR / "used_config.json", "w", encoding="utf-8") as f:
    json.dump(config_params, f, ensure_ascii=False, indent=2)

logger.info("Loading BLACK and WHITE cards text...")

cards_text_dir = config_params.get("cards_texts_dir", "./cards_dataset")
langs = config_params.get("languages", ["EN"])
DIC_ALL_CARDS = load_cards(cards_text_dir, langs)  # file_type xlsx by default

logger.info("Loading games configurations...")

DIC_ALL_GAMES = load_games(
    data_dir=cards_text_dir,
    langs=langs,
    dataset=config_params.get("dataset_size", "test"),    
    subset_rows=config_params.get("subset_rows", 2),
    files_names=config_params.get("files_names", "")
    )  # file_type xlsx by default


logger.info("Running ollama models...")

df_results = run_models(
    n_rounds=int(config_params.get("rounds", 1)),
    models=list(config_params.get("models", ["gemma3:4b"])),
    temperatures=list(config_params.get("temperatures", [0.8])),
    games=DIC_ALL_GAMES,
    cards=DIC_ALL_CARDS
)

logger.info(f"Saving results in {RUN_DIR.resolve()}...")
xlsx_path = RUN_DIR / f"{ResultsName.LLM_RESPONSES.value}.xlsx"
#csv_path  = RUN_DIR / f"{ResultsName.LLM_RESPONSES.value}.csv"
df_results.to_excel(xlsx_path, index=False, header=True, sheet_name="responses")
#df_results.to_csv(csv_path, index=False, encoding='utf-8')

logger.info("END")


