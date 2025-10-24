print("Importing Libraries")
from pathlib import Path
from datetime import datetime
import json


from src.data_loader import load_cards, config_games
from src.model_runner import run_models
from src.args_parser import get_args
from src.utils import *

print("Parsing config.json file to get parameters...")

config_params = get_args()

results_dir = Path(config_params.get("save_dir", "./results"))
results_dir.mkdir(parents=True, exist_ok=True)
demote_previous_last_runs(results_dir)  # From utils.py

date_tag = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
RUN_DIR = results_dir / f"last_run_{date_tag}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

write_latest_pointer(results_dir, RUN_DIR)  # From utils.py

# Save used configuration in results
with open(RUN_DIR / "config_used.json", "w", encoding="utf-8") as f:
    json.dump(config_params, f, ensure_ascii=False, indent=2)

print("Loading BLACK and WHITE cards text...")

data_dir = config_params.get("data_dir", "./cards_dataset")
DIC_ALL_CARDS = load_cards(data_dir)

print("Loading games configurations...")

DIC_ALL_GAMES = config_games(
    dataset=config_params.get("dataset", "test"),
    data_dir=data_dir,
    subset_n=config_params.get("subset_n"))  # It could be None

print("Running ollama models...")

df_results = run_models(
    n_rounds=int(config_params.get("rounds", 1)),
    models=list(config_params.get("models", [])),
    temperatures=list(config_params.get("temperatures", [])),
    games=DIC_ALL_GAMES,
    cards=DIC_ALL_CARDS
)

print("Saving results...")

xlsx_path = RUN_DIR / "all_responses_results.xlsx"
csv_path  = RUN_DIR / "all_responses_results.csv"
df_results.to_excel(xlsx_path, index=False, header=True, sheet_name="responses")
df_results.to_csv(csv_path, index=False)

print(f"Finish. Run folder: {RUN_DIR.resolve()}")


