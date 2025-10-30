print("Importing Libraries")
from pathlib import Path
from datetime import datetime
#import pandas as pd

from src.args_parser import get_args
from src.data_loader import load_cards
from src.response_processing import split_responses, build_sentence
from src.utils import load_last_run_data

print("Parsing config.json file to get parameters...")

config_params = get_args()

date_tag = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
results_dir = Path(config_params.get("results_dir", "./results"))
RUN_DIR = results_dir / f"processing_{date_tag}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("Loading BLACK and WHITE cards text...")

cards_text_dir = config_params.get("cards_texts_dir", "./cards_dataset")
langs = config_params.get("languages", ["EN"])
DIC_ALL_CARDS = load_cards(cards_text_dir, langs, file_type='xlsx')

print("Loading model responses...")

run_to_process = config_params.get("run_to_process","last")  # "last", "all"
file_type = config_params.get("file_type","xlsx")

if run_to_process == "last":
    df_results = load_last_run_data(results_dir)  # file_type xlsx by default
elif run_to_process == "all":
    all_runs = []
    print("IN process...")    
    pass  # Write this part
else:
    raise ValueError(f"Not valid value for 'run_to_process': {run_to_process}. It must be last or all.")

print(f"Rows loaded: {len(df_results)}")

print("Starting model responses processing...")

print("Filtering good responses...")

# Spliting the dataset into good answers and answers with problems
df_winners_id, df_no_id_detected, df_mismatch_id_spaces = split_responses(df_results, DIC_ALL_CARDS)

if not df_no_id_detected.empty:
    print(f"Rows without cards id detected: {len(df_no_id_detected)}")
    no_id_xlsx_path = RUN_DIR / "all_games_no_id_detected.xlsx"
    no_id_csv_path = RUN_DIR / "all_games_no_id_detected.csv"
    df_no_id_detected.to_excel(no_id_xlsx_path, index=False, header=True, sheet_name="no_id_results")
    df_no_id_detected.to_csv(no_id_csv_path, index=False)

if not df_mismatch_id_spaces.empty:
    print(f"Rows rows where the count between ids and spaces does not match detected: {len(df_mismatch_id_spaces)}")
    mismacht_xlsx_path = RUN_DIR / "all_games_mismatch.xlsx"
    mismacht_csv_path = RUN_DIR / "all_games_mismatch.csv"
    df_mismatch_id_spaces.to_excel(mismacht_xlsx_path, index=False, header=True, sheet_name="mismatch_results")
    df_mismatch_id_spaces.to_csv(mismacht_csv_path, index=False)

print(f"Results dataframe rows after cleaning detected: {len(df_winners_id)}")

print("Building sentences...")

df_winners_id['sentence'] = df_winners_id.apply(build_sentence, axis=1, args=(DIC_ALL_CARDS,))

print(f"Saving results in {RUN_DIR.resolve()}...")

good_results_xlsx_path = RUN_DIR / "all_games_good_results.xlsx"
good_results_csv_path  = RUN_DIR / "all_games_good_results.csv"
df_winners_id.to_excel(good_results_xlsx_path, index=False, header=True, sheet_name="good_results")
df_winners_id.to_csv(good_results_csv_path, index=False)

print("[END]")