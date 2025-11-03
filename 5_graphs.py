print("Importing Libraries")

from pathlib import Path
from src.args_parser import get_args
from src.utils import get_last_pointer_dir, load_last_data, PointerFile, ensure_outdir
from src.plotting import plot_all, plot_all_configs

print("Parsing config.json file to get parameters...")

config_params = get_args()
results_dir = Path(config_params.get("results_dir", "./results"))

# Get latest processing directory
RUN_DIR = get_last_pointer_dir(results_dir, PointerFile.LATEST_PROCESS.value)
plot_dir = RUN_DIR / "plots"
file_type = config_params.get("file_type", "xlsx")
file_names = config_params.get("file_names", [])

print(f"Loading data from: {file_names}...")

# Load all scores files (Detoxify, Perspective...)
datasets = [] # List[pd.DataFrame()]
for name in file_names:
    df = load_last_data(RUN_DIR, name, file_type)
    res = (name,df)
    datasets.append(res)

print("Creating Graphics (saving .png pictures)...")

for name, df in datasets:

    print (f"Processing {name} file...")
    
    if df.empty:
        print(f"[WARN] There are no rows to plot from {name}..")
        continue

    name = name.split('_')
    name = name[2]

    plot_paths_1 = plot_all(df, outdir=plot_dir, classifier_name=name)    
    
    with open(plot_dir / f"{name}_generated_plots.txt", "w", encoding="utf-8") as f:
        for p in plot_paths_1:
            f.write(str(p) + "\n")
        
print(f"Graph saved in {plot_dir.resolve()}...")
print("[END]")