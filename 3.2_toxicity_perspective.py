print("Importing Libraries")

from pathlib import Path
from src.args_parser import get_args
from src.toxicity_perspective import analyze_texts, add_perspective_scores
from src.utils import get_last_pointer_dir, load_last_data, PointerFile, ResultsName

print("Parsing config.json file to get parameters...")

config_params = get_args()
results_dir = Path(config_params.get("results_dir", "./results"))

# Get latest processing directory
RUN_DIR = get_last_pointer_dir(results_dir, PointerFile.LATEST_PROCESS.value)
file_to_process_name = ResultsName.GOOD_RESPONSES.value
file_type = config_params.get("file_type", "xlsx")

print(f"Loading data: {file_to_process_name}...")

df_responses = load_last_data(RUN_DIR, file_to_process_name, file_type)
if 'sentence' not in df_responses.columns:
    raise KeyError(f"There is not 'sentence' column in the file: {list(df_responses.columns)}")

print("Clasifying Toxicity with Perspective (Google clasifier)...")
print("Adding scores to sentences...")

# Getting the responses from the API -> List[Dict]
perspectives_scores = analyze_texts(df_responses["sentence"])

df_perspectives_scores = add_perspective_scores(df_responses, perspectives_scores, text_col="sentence")

# Remove columns of NAN values in case some category is not present
#df_perspectives_scores = df_perspectives_scores.dropna(axis=1, how='all')

print(f"Saving results in {RUN_DIR.resolve()}...")

perspective_scores_xlsx_path = RUN_DIR / f"{ResultsName.PERSPECTIVE_SCORES.value}.xlsx"
perspective_scores_csv_path  = RUN_DIR / f"{ResultsName.PERSPECTIVE_SCORES.value}.csv"
df_perspectives_scores.to_excel(perspective_scores_xlsx_path, index=False, header=True, sheet_name="toxicity_scores")
df_perspectives_scores.to_csv(perspective_scores_csv_path, index=False)

print("[END]")

