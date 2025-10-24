print("Importing Libraries")
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.data_loader import load_cards
from src.args_parser import get_args
from src.response_processing import get_winners_id, build_sentence
from src.toxicity_detox import add_detoxify_scores
from src.plotting import plot_all, plot_all_configs
from src.utils import *

print("Parsing config.json file to get parameters...")

config_params = get_args()

date_tag = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
results_dir = Path(config_params.get("save_dir", "./results"))
RUN_DIR = results_dir / f"process_{date_tag}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("Loading BLACK and WHITE cards text...")

data_dir = config_params.get("data_dir", "./cards_dataset")
DIC_ALL_CARDS = load_cards(data_dir)

print("Loading model responses...")

process_run = config_params.get("process_run","last")

if process_run == "last":
    df_results = pd.read_excel(get_last_run_path_csv(results_dir))
elif process_run == "all":
    pass  # Write this part
else:
    raise ValueError(f"Not valid value for 'process_run': {process_run}")

print(f"Rows loaded: {len(df_results)}")

print("Starting MODEL RESPONSES PROCESSING ...")

# df with rows where the answer does not contains a white card ID
no_id_detected = get_winners_id(df_results)
if not no_id_detected.empty:
    print(f"Rows without ID detected: {len(no_id_detected)}")

df_results['sentence'] = df_results.apply(build_sentence, axis=1, args=(DIC_ALL_CARDS,))

# How many times the model choose less or more cards than blank spaces
mask_inconsist = df_results["sentence"].str.contains(r"\[WARN", na=False)
df_inconsistencies = df_results.loc[mask_inconsist].copy()
if not df_inconsistencies.empty:
    print(f"Inconsistent rows (WARN) detected: {len(df_inconsistencies)}")
df_results = df_results.loc[~mask_inconsist].copy()

print(f"Rows after cleaning: {len(df_results)}")

print("Saving results...")

results_xlsx = RUN_DIR / "all_configurations_results.xlsx"
results_csv  = RUN_DIR / "all_configurations_results.csv"
noid_csv = RUN_DIR / "no_id_detected_rows.csv"
inconsistencies_csv = RUN_DIR / "inconsistent_rows.csv"

df_results.to_excel(results_xlsx, index=False, header=True, sheet_name="results")
df_results.to_csv(results_csv, index=False)

if not no_id_detected.empty:
    no_id_detected.to_csv(noid_csv, index=False)
if not df_inconsistencies.empty:
    df_inconsistencies.to_csv(inconsistencies_csv, index=False)

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

if df_results_detoxify_scores.empty:
    print("[WARN] There are no rows to plot after preprocessing..")

plot_paths = plot_all(df_results_detoxify_scores, outdir=RUN_DIR)
plot_all_configs(df_results_detoxify_scores, outdir=RUN_DIR)

with open(RUN_DIR / "generated_plots.txt", "w", encoding="utf-8") as f:
    for p in plot_paths:
        f.write(str(p) + "\n")

print(f"Finish. Run folder: {RUN_DIR.resolve()}")