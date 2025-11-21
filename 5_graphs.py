from src.logging import create_logger
logger = create_logger (log_name="main")

logger.info("Importing Libraries")

from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
from src.args_parser import get_args
from src.utils import get_last_pointer_dir, load_last_data, PointerFile, ensure_outdir
from src.plotting import (
    plot_all, 
    plot_election_frequency, 
    plot_inconsistency_ratio, 
    plot_toxicity_delta_distribution, 
    plot_toxicity_delta_boxplot
)

# File names constants
FILE_SCORES_DETOX = "all_games_detoxify_scores"
FILE_SCORES_PERSPECTIVE = "all_games_perspective_scores"
FILE_FREQ = "all_games_election_frequencies"
FILE_INCONSISTENCIES = "all_games_election_inconsistencies"
FILE_DELTA_DETOX = "all_games_delta_toxicity_detox"
FILE_DELTA_PERSPECTIVE = "all_games_delta_toxicity_perspective"

def setup_and_load() -> Tuple[Path, Path, List[Tuple[str, pd.DataFrame]], Dict[str, pd.DataFrame]]:
    """
    Configure directories and load all scores and analysis DataFrames.
    """
    logger.info("Parsing config.json file to get parameters...")
    config_params = get_args()
    results_dir = Path(config_params.get("results_dir", "./results"))
    
    # Get latest processing directory
    RUN_DIR = get_last_pointer_dir(results_dir, PointerFile.LATEST_PROCESS.value)
    logger.debug(f"Current process folder: {RUN_DIR}")
    plot_dir = RUN_DIR / "plots"
    file_type = config_params.get("file_type", "xlsx")

    logger.info("Loading general score data...")
    # Load all scores files (Detoxify, Perspective...)
    datasets = [] # List[(name, df)]
    for name in [FILE_SCORES_DETOX, FILE_SCORES_PERSPECTIVE]:
        df = load_last_data(RUN_DIR, name, file_type)
        datasets.append((name, df))
        logger.info(f"Loaded {len(df)} rows for {name}.")

    logger.info("Loading analysis data...")
    # Load specific analysis files
    analysis_data = {
        'df_frequencies': load_last_data(RUN_DIR, FILE_FREQ, file_type),
        'df_inconsistencies': load_last_data(RUN_DIR, FILE_INCONSISTENCIES, file_type),
        'df_delta_tox_detox': load_last_data(RUN_DIR, FILE_DELTA_DETOX, file_type),
        'df_delta_tox_perspective': load_last_data(RUN_DIR, FILE_DELTA_PERSPECTIVE, file_type),
    }
    logger.info("All analysis data loaded.")

    return RUN_DIR, plot_dir, datasets, analysis_data


def generate_score_plots(datasets: List[Tuple[str, pd.DataFrame]], plot_dir: Path):
    """
    Generates overall score charts using the original 'plot_all' function.
    """
    logger.info("Creating Graphics for General Scores (using plot_all)...")
    for name, df in datasets:

        if df.empty:
            logger.warning(f"There are no rows to plot from {name}..")
            continue

        # Extract the classifier name
        classifier_name = name.split('_')[2] 

        plot_paths = plot_all(df, outdir=plot_dir, classifier_name=classifier_name)    
        
        # Log paths
        with open(plot_dir / f"{classifier_name}_generated_plots.txt", "w", encoding="utf-8") as f:
            for p in plot_paths:
                f.write(str(p) + "\n")


def generate_analysis_plots(analysis_data: Dict[str, pd.DataFrame], plot_dir: Path):
    """
    Generates graphs of specific analysis results.
    """
    logger.info("Creating Graphics for Analysis Results (Frequencies, Inconsistencies, Delta)...")
    
    paths: List[Path] = []
    
    # 1. Frequency Graph (use df_frequencies)
    df_freq = analysis_data['df_frequencies']
    if not df_freq.empty:
        paths.append(plot_election_frequency(df_freq, plot_dir, 'all_models'))
        logger.info("Generated Election Frequency plot.")

    # 2. Inconsistencies graph (use df_inconsistencies)
    df_inconsistencies = analysis_data['df_inconsistencies']
    if not df_inconsistencies.empty:
        paths.append(plot_inconsistency_ratio(df_inconsistencies, plot_dir, 'all_models'))
        logger.info("Generated Inconsistency Ratio plot.")
        
    # 3. Delta of Toxicity Charts (loop for Detoxify and Perspective)
    for key in ['df_delta_tox_detox', 'df_delta_tox_perspective']:
        df_delta = analysis_data[key]
        if not df_delta.empty:
            classifier_name = key.split('_')[3] # 'detox' o 'perspective'

            # Distribution Chart (C)
            paths.append(plot_toxicity_delta_distribution(df_delta, plot_dir, classifier_name))
            
            # Box Plot Chart (D)
            paths.append(plot_toxicity_delta_boxplot(df_delta, plot_dir, classifier_name))
            
            logger.info(f"Generated Delta Toxicity plots for {classifier_name}.")

    # Log all analysis paths
    with open(plot_dir / "analysis_generated_plots.txt", "w", encoding="utf-8") as f:
        for p in [p for p in paths if p is not None]:
            f.write(str(p) + "\n")


def main():
    """Flujo principal de ejecución."""
    
    # 1. Configuration and Data Loading
    RUN_DIR, plot_dir, datasets, analysis_data = setup_and_load()
    ensure_outdir(plot_dir)
    
    # 2. Graphing General Scores
    generate_score_plots(datasets, plot_dir)
    
    # 3. Graphing Analysis Results
    generate_analysis_plots(analysis_data, plot_dir)
    
    logger.info("END")

if __name__ == "__main__":
    main()