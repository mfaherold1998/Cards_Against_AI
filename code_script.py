print("Importing Libraries")
from pathlib import Path
from datetime import datetime
import json

from src.data_loader import load_cards, config_games
from src.model_runner import run_models
from src.args_parser import get_args
from src.response_processing import get_winners_id, build_sentence
from src.toxicity_detox import add_detoxify_scores
from src.plotting import plot_all, plot_all_configs


print("Parsing config.json file to get parameters...")

config_params = get_args()

date_tag = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
RUN_DIR = Path(config_params.get("save_dir", "./results")) / f"run_{date_tag}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

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
    pick=bool(config_params.get("pick_more_than_2", False)),
    games=DIC_ALL_GAMES,
    cards=DIC_ALL_CARDS
)

print("Starting MODEL RESPONSES PROCESSING ...")

# df with rows where the answer does not contains a white card ID
no_id_detected = get_winners_id(df_results)  
df_results['sentence'] = df_results.apply(build_sentence, axis=1, args=(DIC_ALL_CARDS,))

# How many times the model choose less or more cards than blank spaces 
df_inconsistencies = df_results["sentence"].str.contains(r"\[WARN").copy()

print("Saving results...")

xlsx_path = RUN_DIR / "all_configurations_results.xlsx"
csv_path  = RUN_DIR / "all_configurations_results.csv"
noresp_csv = RUN_DIR / "no_id_detected_rows.csv"
df_results.to_excel(xlsx_path, index=False, header=True, sheet_name="results")
df_results.to_csv(csv_path, index=False)
if not no_id_detected.empty:
    no_id_detected.to_csv(noresp_csv, index=False)

print("Clasifying Toxicity with Detoxify (local clasifier)...")
print("Adding scores to sentences...")

df_results_detoxify_scores = add_detoxify_scores(
    df_results, 
    text_col='sentence', 
    model=config_params.get("detoxify_model", "original"), 
    inplace=True)

# Remove columns of NAN values in case some category is not present
df_results_detoxify_scores = df_results_detoxify_scores.dropna(axis=1, how='all')

print("Creating Graphics (saving .png pictures)...")

plot_paths = plot_all(df_results_detoxify_scores, outdir=RUN_DIR)
plot_all_configs(df_results_detoxify_scores, outdir=RUN_DIR)

with open(RUN_DIR / "generated_plots.txt", "w", encoding="utf-8") as f:
    for p in plot_paths:
        f.write(str(p) + "\n")

print(f"Finish. Run folder: {RUN_DIR.resolve()}")


