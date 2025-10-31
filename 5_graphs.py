print("Creating Graphics (saving .png pictures)...")

if df_results_detoxify_scores.empty:
    print("[WARN] There are no rows to plot after preprocessing..")

plot_paths = plot_all(df_results_detoxify_scores, outdir=RUN_DIR)
plot_all_configs(df_results_detoxify_scores, outdir=RUN_DIR)

with open(RUN_DIR / "generated_plots.txt", "w", encoding="utf-8") as f:
    for p in plot_paths:
        f.write(str(p) + "\n")