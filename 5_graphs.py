from src.logging import create_logger
logger = create_logger (log_name="main")

logger.info("Importing Libraries")

from pathlib import Path
from src.args_parser import get_args
from src.utils import get_last_pointer_dir, load_last_data, PointerFile, ensure_outdir
from src.plotting import plot_all

logger.info("Parsing config.json file to get parameters...")

config_params = get_args()
results_dir = Path(config_params.get("results_dir", "./results"))

# Get latest processing directory
RUN_DIR = get_last_pointer_dir(results_dir, PointerFile.LATEST_PROCESS.value)
logger.debug(f"Current process folder: {RUN_DIR}")
plot_dir = RUN_DIR / "plots"
file_type = config_params.get("file_type", "xlsx")
file_names = config_params.get("file_names", [])

logger.info(f"Loading data from: {file_names}...")

# Load all scores files (Detoxify, Perspective...)
datasets = [] # List[pd.DataFrame()]
for name in file_names:
    df = load_last_data(RUN_DIR, name, file_type)
    res = (name,df)
    datasets.append(res)

logger.info("Creating Graphics (saving .png pictures)...")

for name, df in datasets:

    logger.info(f"Generating all plots for {name} file...")
    
    if df.empty:
        logger.warning(f"There are no rows to plot from {name}..")
        continue

    name = name.split('_')
    name = name[2]

    plot_paths_1 = plot_all(df, outdir=plot_dir, classifier_name=name)    
    
    with open(plot_dir / f"{name}_generated_plots.txt", "w", encoding="utf-8") as f:
        for p in plot_paths_1:
            f.write(str(p) + "\n")
        
logger.info(f"Plots saved in {plot_dir.resolve()}...")
logger.info("END")