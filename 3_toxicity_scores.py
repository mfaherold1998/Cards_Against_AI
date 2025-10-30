
from src.toxicity_detox import add_detoxify_scores
from src.plotting import plot_all, plot_all_configs
from src.toxicity_perspective import *

print("Clasifying Toxicity with Detoxify (local clasifier)...")
print("Adding scores to sentences...")

df_results_detoxify_scores = add_detoxify_scores(
    df_results, 
    text_col='sentence', 
    model=config_params.get("detoxify_model", "original"))

# Remove columns of NAN values in case some category is not present
df_results_detoxify_scores = df_results_detoxify_scores.dropna(axis=1, how='all')

print("Clasifying Toxicity with Perspective (Google clasifier)...")
print("Adding scores to sentences...")

perspective_responses = analyze_texts(df_results["sentence"], attributes=DEFAULT_ATTRIBUTES)
save_perspective_responses(perspective_responses, results_dir / "perspective_analysis.json")

df_with_scores = attach_perspective_scores(df_results, perspective_responses, text_col="sentence")

df_with_scores.to_csv(results_dir / "perspective_results.csv", index=False)
print(f"Saved in: {results_dir}")

print("Creating Graphics (saving .png pictures)...")

if df_results_detoxify_scores.empty:
    print("[WARN] There are no rows to plot after preprocessing..")

plot_paths = plot_all(df_results_detoxify_scores, outdir=RUN_DIR)
plot_all_configs(df_results_detoxify_scores, outdir=RUN_DIR)

with open(RUN_DIR / "generated_plots.txt", "w", encoding="utf-8") as f:
    for p in plot_paths:
        f.write(str(p) + "\n")