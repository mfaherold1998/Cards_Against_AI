from src.utils.logging import create_logger
logger = create_logger (log_name="main")

logger.info("IMPORTING LIBRARIES")
from pathlib import Path
import json

from src.utils.args_parser import get_args
from src.data.data_loader import load_data
from src.scripts.build_responses import split_responses, build_sentence, build_all_combinations
from src.utils.utils import FilesNames, DirNames, ResultsNames

def main():
    logger.info("PARSING config.json FILE TO GET PARAMETERS...")

    # 1. Get parameters from json config
    config_params = get_args(2)
    run_id = config_params.get("run_id")
    run_config_path = config_params.get("run_config_dir")

    # 2. Get configuration from run_config
    jfile ={}
    with open(run_config_path, 'r', encoding='utf-8') as archivo_json:
        jfile = json.load(archivo_json)

    data_dir = jfile["cards_dir"] # ("./data")
    results_dir = jfile["results_dir"] # ("./results")
    languages = jfile["languages"] # (["EN"])
    file_type = jfile["file_type"] # ("xlsx")

    # 3. Create the dir for processed data
    results_dir = Path(results_dir)
    run_dir = results_dir / run_id
    PROCESSED_DATA_DIR = results_dir / run_id / DirNames.LLL_PROCESSED_DATA.value
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.debug(f"CURRENT PROCESSED DATA DIR: {PROCESSED_DATA_DIR}")

    logger.info("LOADING BLACK AND WHITE CARDS TEXTS...")

    DIC_ALL_CARDS = {}

    # 4. Loading Cards Text
    for lang in languages:
            cards_text_dir = f"{data_dir}/{lang}/{DirNames.CARDS_DIR.value}"
            black_cards_path = f"{cards_text_dir}/{FilesNames.BLACK_CARDS.value}.{file_type}"
            white_cards_path = f"{cards_text_dir}/{FilesNames.WHITE_CARDS.value}.{file_type}"

            df_black_card, errors_b = load_data(black_cards_path)
            df_white_card, errors_w = load_data(white_cards_path)

            DIC_ALL_CARDS[lang] = {"BLACK" : df_black_card.set_index('card_id'), "WHITE" : df_white_card.set_index('card_id')}

    logger.info("LOADING MODEL RESPONSES...")

    # 5. Loading raw responses from models
    raw_data_dir = run_dir / DirNames.LLM_RAW_RESPONSES.value
    raw_responses_path = raw_data_dir / f"{ResultsNames.LLM_RAW_RESPONSES.value}.{file_type}"
    df_results, errors = load_data(raw_responses_path)

    logger.info(f"ROWS LOADED: {len(df_results)}")

    logger.info("STARTING MODEL RESPONSES PROCESSING...")

    logger.info("FILTERING GOOD RESPONSES...")

    # 6. Spliting the dataset into good answers and answers with problems
    df_all_good_responses, df_no_id_detected, df_mismatch_id_spaces = split_responses(df_results, DIC_ALL_CARDS)

    # 7. Saving no good results
    if not df_no_id_detected.empty:
        logger.info(f"Rows without card id detected: {len(df_no_id_detected)}")
        no_id_path = raw_data_dir / f"{ResultsNames.NO_ID_RESPONSES.value}.{file_type}"
        if file_type == "xlsx":    
            df_no_id_detected.to_excel(no_id_path, index=False, header=True, sheet_name="no_id_results")
        elif file_type == "csv":    
            df_no_id_detected.to_csv(no_id_path, index=False, quotechar='"', encoding='utf-8')

    if not df_mismatch_id_spaces.empty:
        logger.info(f"Rows where the count between ids and spaces does not match detected: {len(df_mismatch_id_spaces)}")
        mismacht_path = raw_data_dir / f"{ResultsNames.MISMATCH_RESPONSES.value}.{file_type}"
        if file_type == "xlsx":    
            df_mismatch_id_spaces.to_excel(mismacht_path, index=False, header=True, sheet_name="mismatch_results")
        elif file_type == "csv":    
            df_mismatch_id_spaces.to_csv(mismacht_path, index=False, quotechar='"', encoding='utf-8')


    logger.info(f"RESULTS DF ROWS AFTER CLEANING: {len(df_all_good_responses)}")

    logger.info("BUILDING SENTENCES...")

    # 8. Save original raw responses
    df_original = df_all_good_responses.copy()

    # 9. Building sentences of winners
    df_all_good_responses['sentence'] = df_all_good_responses.apply(build_sentence, axis=1, args=(DIC_ALL_CARDS,))

    logger.info("BUILDING ALL POSSIBLE COMBINATION SENTENCES...")

    # 10. Building all possible combinations sentences
    df_all_combinations = build_all_combinations(df_original, DIC_ALL_CARDS, build_sentence)

    logger.info(f"SAVING RESULTS IN {PROCESSED_DATA_DIR.resolve()}...")

    # 11. Saving results of winners and all combinations
    good_results_path = PROCESSED_DATA_DIR / f"{ResultsNames.GOOD_RESPONSES.value}.{file_type}"
    all_posible_combinations_path = PROCESSED_DATA_DIR / f"{ResultsNames.ALL_POSIBLE_COMBINATIONS.value}.{file_type}"
    if file_type == "xlsx":    
        df_all_good_responses.to_excel(good_results_path, index=False, header=True, sheet_name="winner_sentences")
        df_all_combinations.to_excel(all_posible_combinations_path, index=False, header=True, sheet_name="all_sentences")
    elif file_type == "csv":    
        df_all_good_responses.to_csv(good_results_path, index=False, quotechar='"', encoding='utf-8')
        df_all_combinations.to_csv(good_results_path, index=False, quotechar='"', encoding='utf-8')

    logger.info("END")

if __name__ == "__main__":
    main()