#from src.utils.logging import create_logger
#logger = create_logger (log_name="main")

#logger.info("IMPORTING LIBRARIES")
from pathlib import Path
from datetime import datetime
import json

from src.utils.args_parser import get_args
from src.data.data_loader import load_data
from src.scripts.run_models import run_models
from src.utils.utils import FilesNames, DirNames, ResultsNames

def main():
    
    #logger.info("PARSING config.json FILE TO GET PARAMETERS...")

    # 1. Get parameters from json config
    config_params = get_args(1) #json number file 1_run_config.json

    # 2. Set or create results folder path
    results_dir = Path(config_params.get("results_dir"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create the current run directory with its run_id
    date_tag = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
    prompt_type = config_params.get("prompt_type")
    run_id = f"run_{prompt_type}_{date_tag}"
    print(run_id)
    RUN_DIR = results_dir / run_id
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    #logger.debug(f"CURRENT RUN DIR (RUN_ID): {run_id}")

    # 4. Save used configuration in RUN_DIR
    with open(RUN_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_params, f, ensure_ascii=False, indent=2)

    #logger.info("LOADING BLACK AND WHITE CARDS TEXTS...")

    DIC_ALL_CARDS = {} # Dict{ "EN":{BLACK:pd.DataFrame, WHITE:pd.dataFrame}, "IT":{BLACK:pd.DataFrame, WHITE:pd.dataFrame}, ... }
    DIC_ALL_GAMES = {} # Dict{ "EN": {"game1":pd.DataFrame, "game2":pd.DataFrame,...}, "IT": {"game1":pd.DataFrame, "game2":pd.DataFrame,...}, ...}

    data_dir = config_params.get("cards_dir")
    langs = config_params.get("languages")
    file_type = config_params.get("file_type")
    dataset_mode = config_params.get("dataset_mode")
    subset_rows = config_params.get("test_num_rows")


    # 5. Loading card texts data
    for lang in langs:
        cards_text_dir = f"{data_dir}/{lang}/{DirNames.CARDS_DIR.value}"
        black_cards_path = f"{cards_text_dir}/{FilesNames.BLACK_CARDS.value}.{file_type}"
        white_cards_path = f"{cards_text_dir}/{FilesNames.WHITE_CARDS.value}.{file_type}"

        df_black_card, errors_b = load_data(black_cards_path)
        df_white_card, errors_w = load_data(white_cards_path)

        DIC_ALL_CARDS[lang] = {"BLACK" : df_black_card.set_index('card_id'), "WHITE" : df_white_card.set_index('card_id')} 

    #logger.info("LOADING GAMES CONFIGURATIONS...")

    # 6. Loading games configuration data
    if prompt_type == "prompt_player":
        games_config_dir = Path(f"{data_dir}/{lang}/{DirNames.GAMES_DIR.value}")
    elif prompt_type == "prompt_judge":
        games_config_dir = Path(f"{data_dir}/{lang}/{DirNames.TO_JUDGE_DIR.value}")

    for lang in langs:
        DIC_ALL_GAMES[lang] = {}
        for file_path in games_config_dir.glob(f"*.{file_type}"):
            df, errors = load_data(file_path)
            DIC_ALL_GAMES[lang][file_path.stem] = df

    if dataset_mode == "test" :
        for game in DIC_ALL_GAMES[lang].keys():
            DIC_ALL_GAMES[lang][game] = DIC_ALL_GAMES[lang][game][:int(subset_rows)]

    #logger.info("RUNNING OLLAMA MODELS...")

    # 7. Running LLM models
    df_results = run_models(
        n_rounds=config_params.get("rounds"),
        models=config_params.get("models"),
        temperatures=config_params.get("temperatures"),
        games=DIC_ALL_GAMES,
        cards=DIC_ALL_CARDS,
        run_langs=langs,
        prompt_to_use= prompt_type,
        character_description=config_params.get("character_description")
    )

    #logger.info(f"SAVING RESULTS IN {RUN_DIR.resolve()}...")

    # 8. Saving Results in raw data
    results_dir = RUN_DIR / DirNames.LLM_RAW_RESPONSES.value
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{ResultsNames.LLM_RAW_RESPONSES.value}.{file_type}"
    
    if file_type == "xlsx":    
        df_results.to_excel(results_path, index=False, header=True, sheet_name="responses")
    elif file_type == "csv":    
        df_results.to_csv(results_path, index=False, encoding='utf-8')

    #logger.info("END")


if __name__ == "__main__":
    main()