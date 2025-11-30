from src.utils.logging import create_logger
logger = create_logger (log_name="main")

logger.info("IMPORTING LIBRARIES")

from pathlib import Path
import json
from src.utils.args_parser import get_args
from src.data.data_loader import load_data
from src.scripts.plotting import plot_all_prompt_player
from src.utils.utils import DirNames

def main():
    
    logger.info("PARSING config.json FILE TO GET PARAMETERS...")

    # 1. Get parameters from json config
    config_params = get_args(4)
    run_id = config_params.get("run_id")
    run_config_path = config_params.get("run_config_dir")

    # 2. Get configuration from run_config
    jfile ={}
    with open(run_config_path, 'r', encoding='utf-8') as archivo_json:
        jfile = json.load(archivo_json)

    results_dir = jfile["results_dir"] # ("./results")
    file_type = jfile["file_type"] # ("xlsx")
    prompt_type = jfile['prompt_type'] # ('prompt_player')

    # 3. Create the dir for plots
    results_dir = Path(results_dir)
    run_dir = results_dir / run_id
    PLOTTING_DIR = results_dir / run_id / DirNames.PLOTS.value
    PLOTTING_DIR.mkdir(parents=True, exist_ok=True)

    logger.debug(f"CURRENT PLOTS DIR: {PLOTTING_DIR}")

    tox_dir = run_dir / DirNames.LLL_TOXICITY_SCORES.value

    logger.info(f"LOADING FILES TO PROCESS FROM DIR: {tox_dir}...")

    # 4. Load all toxicity scores files
    dict_tox_scores = {}
    for file_path in tox_dir.glob(f"*.{file_type}"):
        df, errors = load_data(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path.stem}.")
        dict_tox_scores[file_path.stem] = df

    logger.info("CREATING PLOTS FOR EVERY FILE...")

    # 5. Create all plots
    if prompt_type == "prompt_player":
        for name, df in dict_tox_scores.items():

            file_name = Path(name)

            plot_all_prompt_player(df, outdir=PLOTTING_DIR, classifier_name=file_name.stem)
    
    elif prompt_type == "prompt_judge":
        for name, df in dict_tox_scores.items():

            file_name = Path(name)

            plot_all_prompt_player(df, outdir=PLOTTING_DIR, classifier_name=file_name.stem)

    logger.info("END")

if __name__ == "__main__":
    main()
        