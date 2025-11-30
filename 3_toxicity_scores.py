from src.utils.logging import create_logger
logger = create_logger (log_name="main")

logger.info("IMPORTING LIBRARIES")

from pathlib import Path
import json
import time
from src.utils.args_parser import get_args
from src.data.data_loader import load_data
from src.scripts.toxicity_detox import add_detoxify_scores
from src.scripts.toxicity_perspective import analyze_texts, add_perspective_scores
from src.utils.utils import DirNames, ResultsNames

def main():

    logger.info("PARSING config.json FILE TO GET PARAMETERS...")

    # 1. Get parameters from json config
    config_params = get_args(3)
    run_id = config_params.get("run_id")
    run_config_path = config_params.get("run_config_dir")
    detoxify_model = config_params.get("detoxify_model")
    device = config_params.get("device")
    batch_size = config_params.get("batch_size")

    # 2. Get configuration from run_config
    jfile ={}
    with open(run_config_path, 'r', encoding='utf-8') as archivo_json:
        jfile = json.load(archivo_json)

    results_dir = jfile["results_dir"] # ("./results")
    languages = jfile["languages"] # (["EN"])
    file_type = jfile["file_type"] # ("xlsx")

    # 3. Create the dir for toxicity scores data
    results_dir = Path(results_dir)
    run_dir = results_dir / run_id
    TOXICITY_SCORES_DIR = results_dir / run_id / DirNames.LLL_TOXICITY_SCORES.value
    TOXICITY_SCORES_DIR.mkdir(parents=True, exist_ok=True)

    logger.debug(f"CURRENT TOCIXITY SCORES DIR: {TOXICITY_SCORES_DIR}")

    # 3.1 Create the dir to storage share results for analysis module
    ANALYSIS_DIR = results_dir / DirNames.ANALYSIS.value
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    sentences_dir = run_dir / DirNames.LLL_PROCESSED_DATA.value

    logger.info(f"LOADING FILES TO PROCESS FROM DIR: {sentences_dir}...")

    # 4. Load files with winners and all sentences
    df_sentences = {}
    for file_path in sentences_dir.glob(f"*.{file_type}"):
        df, errors = load_data(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path.stem}.")
        df_sentences[file_path.stem] = df

    df_winners = df_sentences["winners_sentences"].copy()
    df_combinations = df_sentences["all_combination_sentences"].copy()

    logger.info("CLASIFYING TOXICITY WITH DETOXIFY (LOCAL CLASIFYIER)...")

    # 5. Calculate detoxify scores
    logger.info("Adding scores to sentences...")
    df_detoxify_scores_winners = add_detoxify_scores(
        df=df_winners, 
        text_col='sentence', 
        model=detoxify_model,
        device=device,
        batch_size=batch_size)

    df_detoxify_scores_combinations = add_detoxify_scores(
        df=df_combinations, 
        text_col='sentence', 
        model=detoxify_model,
        device=device,
        batch_size=batch_size)

    # 6. Remove columns of NAN values in case some category is not present
    df_detoxify_scores_winners = df_detoxify_scores_winners.dropna(axis=1, how='all')
    df_detoxify_scores_combinations = df_detoxify_scores_combinations.dropna(axis=1, how='all')

    logger.info(f"SAVING RESULTS IN: {TOXICITY_SCORES_DIR.resolve()} ...")

    # 7. Saving Detoxify scores results
    winners_scores_path = TOXICITY_SCORES_DIR / f"{ResultsNames.DETOXIFY_SCORES_WINNERS.value}.{file_type}"
    all_combinations_path = TOXICITY_SCORES_DIR / f"{ResultsNames.DETOXIFY_SCORES_COMBINATIONS.value}.{file_type}"

    # Analysis copies
    winners_analysis = ANALYSIS_DIR / f"{ResultsNames.DETOXIFY_SCORES_WINNERS.value}_{run_id}.{file_type}"
    all_comb_analysis = ANALYSIS_DIR / f"{ResultsNames.DETOXIFY_SCORES_COMBINATIONS.value}_{run_id}.{file_type}"

    if file_type == "xlsx":    
        df_detoxify_scores_winners.to_excel(winners_scores_path, index=False, header=True, sheet_name="toxicity_scores")
        df_detoxify_scores_combinations.to_excel(all_combinations_path, index=False, header=True, sheet_name="toxicity_scores")
        df_detoxify_scores_winners.to_excel(winners_analysis, index=False, header=True, sheet_name="toxicity_scores")
        df_detoxify_scores_combinations.to_excel(all_comb_analysis, index=False, header=True, sheet_name="toxicity_scores")
    elif file_type == "csv":    
        df_detoxify_scores_winners.to_csv(winners_scores_path, index=False, quotechar='"', encoding='utf-8')
        df_detoxify_scores_combinations.to_csv(all_combinations_path, index=False, quotechar='"', encoding='utf-8')
        df_detoxify_scores_winners.to_csv(winners_analysis, index=False, quotechar='"', encoding='utf-8')
        df_detoxify_scores_combinations.to_csv(all_comb_analysis, index=False, quotechar='"', encoding='utf-8')


    logger.info("CLASIFYING TOXICITY WITH PERSPECTIVE (GOOGLE CLASIFYIER)...")
    logger.info("Adding scores to sentences...")

    # 8. Getting the responses from the API
    perspectives_scores_winners = analyze_texts(df_winners["sentence"])
    
    wait_time_between_files = 90
    logger.info(f"First file processed. Waiting {wait_time_between_files} seconds to ensure API limits reset before processing the second file.")
    time.sleep(wait_time_between_files)

    perspectives_scores_combinations = analyze_texts(df_combinations["sentence"])

    # 9. Adding the scores to the df
    df_perspectives_winners = add_perspective_scores(df_winners, perspectives_scores_winners, text_col="sentence")
    df_perspectives_combinations = add_perspective_scores(df_combinations, perspectives_scores_combinations, text_col="sentence")

    logger.info(f"Saving results in {TOXICITY_SCORES_DIR.resolve()}...")

    perspective_winners_path = TOXICITY_SCORES_DIR / f"{ResultsNames.PERSPECTIVE_SCORES_WINNERS.value}.{file_type}"
    perspective_combinations_path = TOXICITY_SCORES_DIR / f"{ResultsNames.PERSPECTIVE_SCORES_COMBINATIONS.value}.{file_type}"

    # Analysis copies
    winners_analysis = ANALYSIS_DIR / f"{ResultsNames.PERSPECTIVE_SCORES_WINNERS.value}_{run_id}.{file_type}"
    all_comb_analysis = ANALYSIS_DIR / f"{ResultsNames.PERSPECTIVE_SCORES_COMBINATIONS.value}_{run_id}.{file_type}"
    
    if file_type == "xlsx":    
        df_perspectives_winners.to_excel(perspective_winners_path, index=False, header=True, sheet_name="toxicity_scores")
        df_perspectives_combinations.to_excel(perspective_combinations_path, index=False, header=True, sheet_name="toxicity_scores")
        df_perspectives_winners.to_excel(winners_analysis, index=False, header=True, sheet_name="toxicity_scores")
        df_perspectives_combinations.to_excel(all_comb_analysis, index=False, header=True, sheet_name="toxicity_scores")
    elif file_type == "csv":    
        df_perspectives_winners.to_csv(perspective_winners_path, index=False, quotechar='"', encoding='utf-8')
        df_perspectives_combinations.to_csv(perspective_combinations_path, index=False, quotechar='"', encoding='utf-8')
        df_perspectives_winners.to_csv(winners_analysis, index=False, quotechar='"', encoding='utf-8')
        df_perspectives_combinations.to_csv(all_comb_analysis, index=False, quotechar='"', encoding='utf-8')
    
    logger.info("END")

if __name__ == "__main__":
    main()