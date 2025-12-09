from src.utils.logging import create_logger
logger = create_logger (log_name="main")

logger.info("IMPORTING LIBRARIES")
from pathlib import Path

from src.utils.args_parser import get_args
from src.data.data_loader import load_data
from src.scripts.analysis import *
from src.utils.utils import DirNames

def main():
    
    logger.info("PARSING config.json FILE TO GET PARAMETERS...")

    # 1. Get parameters from json config
    config_params = get_args(5)
    analysis_dir = config_params.get("analysis_dir")
    analysis_plot_dir = config_params.get("analysis_plot_dir")
    file_type = config_params.get("file_type")
    results_dir = config_params.get("results_dir")

    # 2. Create the dir for analysis results
    analysis_dir = Path(analysis_dir)
    ANALYSIS_RESULTS_DIR = analysis_dir / DirNames.ANALYSIS_RES.value
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.debug(f"CURRENT RESULTS DIR: {ANALYSIS_RESULTS_DIR}")

    logger.info(f"LOADING PLAYERS AND JUDGES FILES TO PROCESS FROM: {analysis_dir}...")

    # 3. Loading the players and judges results from analysis directory
    players_files_dict = {"winners":{}, "combinations": {}}
    judges_files_dict = {"winners":{}, "combinations": {}}

    for file_path in analysis_dir.glob(f"*.{file_type}"):
        
        df, errors = load_data(file_path)
        name = file_path.stem
        logger.info(f"Loaded {len(df)} rows from {name}.")
        
        if 'player' in name: 
            if 'winners' in name:         
                players_files_dict['winners'][name] = df
            elif 'combinations' in name:
                 players_files_dict['combinations'][name] = df
        
        elif 'judge' in name:
            if 'winners' in name:         
                judges_files_dict['winners'][name] = df
            elif 'combinations' in name:
                 judges_files_dict['combinations'][name] = df

    logger.info(f"ANALIZING FILES AND CREATING GRAPHS...")
    # 4. Analyzing players files

    # Winners files
    inconsistencies_players = {}    # { "file_name" : pd.DataFrame, ...}
    success_rate_players = {}       # { "file_name" : pd.DataFrame, ...}
    for name, df in players_files_dict['winners'].items():        
        inconsistencies_players[name] = calculate_models_inconsistencies(df)
        success_rate_players[name] = calculate_success_rate(df)
    
    # Combination files
    overall_toxicity_players ={}
    for name, df in players_files_dict['combinations'].items():
        overall_toxicity_players[name] = calculate_overall_toxicity(df)
    
    # 5. Analyzing judges files
    
    # Winners files
    inconsistencies_judges = {}    # { "file_name" : pd.DataFrame, ...}
    success_rate_judges = {}       # { "file_name" : pd.DataFrame, ...}
    for name, df in judges_files_dict['winners'].items():
        inconsistencies_judges[name] = calculate_models_inconsistencies(df)
        success_rate_judges[name] = calculate_success_rate(df)
    
    # Combination files
    overall_toxicity_judges ={}
    for name, df in judges_files_dict['combinations'].items():
        overall_toxicity_judges[name] = calculate_overall_toxicity(df)
    

    # 6. Judge descriptions Comparisons
    df_judge_description_tox_players = judge_description_comparison_mean_toxicity(players_files_dict['winners'], results_dir)
    df_judge_description_tox_judges = judge_description_comparison_mean_toxicity(judges_files_dict['winners'], results_dir)
    
    logger.info("END")


if __name__ == "__main__":
    main()