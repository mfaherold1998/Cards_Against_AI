from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.utils import build_play_key

from src.utils.logging import create_logger
logger = create_logger (log_name="main")

# Columns of dataframes
CONFIG = "config"
LANG = "lang"
MODEL = "model"
TEMPERATURE = "temperature"
BLACKID = "black_id"
PLAY = "play"
WINNERS = "winners"
SENTENCE = "sentence"
TOXICITY = "toxicity"
SEVERE_TOXICITY = "severe_toxicity"
OBSCENE = "obscene"
THREAT = "threat" 
INSULT = "insult"
IDENTITY_ATTACK = "identity_attack"
SEXUALLY_EXPLICIT = "sexually_explicit"
PROFANITY = "profanity"
ELECTION_FREQ = "election_freq"
INCONSISTENCY_RATIO = "inconsistency_ratio"
TOXICITY_DELTA = "toxicity_delta"
SEVERE_TOXICITY_DELTA = "severe_soxicity_delta"

ATTRIBUTE_COLUMNS = [TOXICITY, SEVERE_TOXICITY, OBSCENE, THREAT, INSULT, IDENTITY_ATTACK, SEXUALLY_EXPLICIT, PROFANITY]

#------- Players Plots --------#

def plot_toxicity_vs_temperature(df: pd.DataFrame, outdir: Path | str, classifier_name:str) -> Optional[Path]:
    
    g = (df.groupby([MODEL, TEMPERATURE])[TOXICITY]
           .agg(["mean", "count", "std"]).reset_index())
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))

    plt.figure()
    for m, sub in g.groupby(MODEL):
        sub = sub.sort_values(TEMPERATURE)
        x = pd.to_numeric(sub[TEMPERATURE], errors='coerce').to_numpy()
        y = pd.to_numeric(sub["mean"], errors='coerce').to_numpy()
        se = pd.to_numeric(sub["sem"], errors='coerce').fillna(0).to_numpy()
        plt.errorbar(x, y, yerr=1.96*se, marker="o", label=str(m))
    plt.xlabel(TEMPERATURE); plt.ylabel(TOXICITY); plt.legend(); plt.title("Toxicity vs Temperature")
    path = Path(outdir) / f"1_{classifier_name}_toxicity_vs_temperature.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_distribution_by_model(df: pd.DataFrame, outdir: Path | str, classifier_name:str) -> Optional[Path]:
    
    models = df[MODEL].dropna().astype(str).unique().tolist()
    data = [df.loc[df[MODEL] == m, TOXICITY].dropna().values for m in models]
    if not any(len(d) for d in data):
        return None

    plt.figure()
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(models)+1), models, rotation=30)
    plt.ylabel(TOXICITY); plt.title("Distribution per model")
    path = Path(outdir) / f"2_{classifier_name}_distribution_by_model.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_temperature_curve_per_model(df: pd.DataFrame, outdir: Path | str, classifier_name:str) -> List[Path]:
    
    g = (df.groupby([MODEL, TEMPERATURE])[TOXICITY]
           .agg(["mean", "count", "std"]).reset_index())
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))

    paths: List[Path] = []
    for m, sub in g.groupby(MODEL):
        sub = sub.sort_values(TEMPERATURE)
        plt.figure()
        x = pd.to_numeric(sub[TEMPERATURE], errors='coerce').to_numpy()
        y = pd.to_numeric(sub["mean"], errors='coerce').to_numpy()
        se = pd.to_numeric(sub["sem"], errors='coerce').fillna(0).to_numpy()
        plt.plot(x, y, marker="o")
        plt.fill_between(x, y-1.96*se, y+1.96*se, alpha=0.2)
        plt.title(f"{m}: Toxicity vs Temperature")
        plt.xlabel("Temperature"); plt.ylabel("Mean toxicity")
        safe_m = str(m).replace("/", "_").replace(":", "_")
        path = Path(outdir) / f"3_{classifier_name}_temperature_curve_{safe_m}.png"
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
        paths.append(path)
    return paths

def plot_rates_above_threshold(df: pd.DataFrame, outdir: Path | str, classifier_name:str, thr: float = 0.8) -> Optional[Path]:
     
    tmp = df.copy()
    tmp["tail"] = (tmp[TOXICITY] >= thr).astype(int)

    rate = (tmp.groupby([MODEL, TEMPERATURE])["tail"]
              .mean().mul(100).reset_index())

    plt.figure()
    for m, sub in rate.groupby(MODEL):
        sub = sub.sort_values(TEMPERATURE)
        plt.plot(sub[TEMPERATURE], sub["tail"], marker="o", label=str(m))
    plt.xlabel("Temperature"); plt.ylabel(f"% ≥ {thr}"); plt.legend(); plt.title("High tail by model")
    path = Path(outdir) / f"4_{classifier_name}_rates_above_threshold.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_black_card_triggers(df: pd.DataFrame, outdir: Path | str, classifier_name:str, top_k: int = 30) -> Optional[Path]:
    
    tmp = df.copy()
    top_black = (tmp.groupby("black_id")[TOXICITY]
                   .mean().sort_values(ascending=False).head(top_k).index)

    mat = (tmp[tmp["black_id"].isin(top_black)]
           .pivot_table(index="black_id", columns=MODEL,
                        values=TOXICITY, aggfunc="mean"))
    if mat.empty:
        return None

    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, cmap="viridis", cbar_kws={"label": "Mean toxicity"})
    plt.title("Top toxic black cards (mean per model)")
    path = Path(outdir) / f"5_{classifier_name}_black_card_triggers.png"
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path

def plot_top_plays_heatmap(df: pd.DataFrame, outdir: Path | str, classifier_name:str, top_k: int = 30) -> Optional[Path]:
    
    tmp = df.copy()
    tmp["play_key"] = [build_play_key(b, w) for b, w in zip(tmp[BLACKID], tmp[WINNERS])]

    top_keys = (tmp.groupby("play_key")[TOXICITY]
                  .mean().sort_values(ascending=False).head(top_k).index)
    mat = (tmp[tmp["play_key"].isin(top_keys)]
           .pivot_table(index="play_key", columns=MODEL,
                        values=TOXICITY, aggfunc="mean"))
    if mat.empty:
        return None

    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, cmap="magma", cbar_kws={"label": "Mean toxicity"})
    plt.title("Top toxic plays (black + white) — mean per model")
    path = Path(outdir) / f"6_{classifier_name}_top_plays_heatmap.png"
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path

def plot_instability(df: pd.DataFrame, outdir: Path | str, classifier_name:str, top_n: int = 10) -> List[Path]:
   
    tmp = df.copy()
    tmp["play_key"] = [build_play_key(b, w) for b, w in zip(tmp[BLACKID], tmp[WINNERS])]

    stability = (tmp.groupby([MODEL, TEMPERATURE, "play_key"])[TOXICITY]
                   .agg(["mean", "std", "count"]).reset_index()
                   .rename(columns={"mean": "tox_mean", "std": "tox_std", "count": "n"}))
    if stability.empty:
        return []

    # Scatter
    stability["size"] = stability["n"] * 500 / max(stability["n"].max(), 1)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=stability, x="tox_mean", y="tox_std",
        hue=MODEL, size="size", sizes=(50, 500), alpha=0.7, legend="full"
    )
    plt.title("Mean Toxicity vs. Instability (Standard Deviation)")
    plt.xlabel("Average Toxicity (tox_mean)"); plt.ylabel("Instability (tox_std)")
    p1 = Path(outdir) / f"7_{classifier_name}_instability_scatter.png"
    plt.tight_layout(); plt.savefig(p1, dpi=200); plt.close()

    # Top-N bar
    top_unstable = stability.sort_values("tox_std", ascending=False).head(top_n).copy()
    #top_unstable["ranking"] = [f"Play_{i+1}" for i in range(len(top_unstable))]
    plt.figure(figsize=(10, max(7, top_n * 0.7)))
    sns.barplot(data=top_unstable, x="tox_std", y="play_key", hue=MODEL, dodge=False)
    plt.title(f"Top {top_n} Most Unstable Combinations (High STD)")
    plt.xlabel("STD of Toxicity (tox_std)"); plt.ylabel("Play")
    plt.legend(title="Model"); plt.grid(axis="x", linestyle="--", alpha=0.7)
    p2 = Path(outdir) / f"7_{classifier_name}_instability_bar.png"
    plt.tight_layout(); plt.savefig(p2, dpi=200); plt.close()

    return [p1, p2]

def plot_category_comparison(df: pd.DataFrame, outdir: Path | str, classifier_name:str) -> Optional[Path]:
   
    present_attributes = [col for col in ATTRIBUTE_COLUMNS if col in df.columns]

    agg = df.groupby(MODEL)[present_attributes].mean()

    ax = agg.plot(kind="bar")
    plt.ylabel("Average per attribute"); plt.title("Profile of attributes per model"); plt.xticks(rotation=30)
    path = Path(outdir) / f"8_{classifier_name}_category_comparison.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_language_risk(df: pd.DataFrame, outdir: Path | str, classifier_name:str) -> List[Path]:
  
    def summarize_toxicity(d: pd.DataFrame) -> pd.DataFrame:
        return (d.groupby(MODEL)[TOXICITY]
                  .agg(mean="mean",
                       p50=lambda s: s.quantile(.5),
                       p90=lambda s: s.quantile(.9),
                       p95=lambda s: s.quantile(.95))
                  .round(3))

    paths: List[Path] = []
    for L in df[LANG].dropna().unique():
        sub = df[df[LANG] == L]
        out = summarize_toxicity(sub)
        ax = out[["mean", "p50", "p90", "p95"]].plot(kind="bar")
        plt.title(f"Toxicity profile ({L}) per model")
        plt.ylabel("Score"); plt.xlabel("Model"); plt.xticks(rotation=30)
        path = Path(outdir) / f"9_{classifier_name}_language_risk_{L}.png"
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
        paths.append(path)
    return paths

def plot_config_heatmap(df: pd.DataFrame, outdir: Path | str, classifier_name:str, min_n: int = 1) -> Optional[Path]:
   
    CLEAN_CONFIG = 'clean_config'
    df[CLEAN_CONFIG] = df[CONFIG].str.replace("_configurations", "", regex=False)
    
    stats = (df.groupby([CLEAN_CONFIG, MODEL])[TOXICITY]
               .agg(["mean","count"]).reset_index())
    stats = stats[stats["count"] >= min_n]
    if stats.empty:
        return None

    mat = stats.pivot(index=CLEAN_CONFIG, columns=MODEL, values="mean")
    if mat.empty:
        return None

    plt.figure(figsize=(10, max(6, 0.5*len(mat.index))))
    sns.heatmap(mat, cmap="viridis", cbar_kws={"label": "Mean toxicity"})
    plt.title("Mean toxicity by configuration × model")
    path = Path(outdir) / f"10_{classifier_name}_config_heatmap_mean_toxicity.png"
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path

def plot_config_distribution(df: pd.DataFrame, outdir: Path | str, classifier_name:str, by_model: bool = False) -> Optional[Path]:
    
    CLEAN_CONFIG = 'clean_config'
    df[CLEAN_CONFIG] = df[CONFIG].str.replace("_configurations", "", regex=False)

    plt.figure(figsize=(12, max(6, 0.5 * df[CLEAN_CONFIG].nunique())))
    if by_model:
        sns.violinplot(data=df, x=TOXICITY, y=CLEAN_CONFIG,
                       hue=MODEL, cut=0, inner="quartile", density_norm="width")
        plt.legend(title="Model", bbox_to_anchor=(1.04, 1), loc="upper left")
        title = "Toxicity distribution by configuration (split by model)"
        fname = f"{classifier_name}_config_distribution_by_model.png"
    else:
        sns.violinplot(data=df, x=TOXICITY, y=CLEAN_CONFIG,
                       cut=0, inner="quartile", density_norm="width")
        title = "Toxicity distribution by configuration (all models)"
        fname = f"11_{classifier_name}_config_distribution_model_{by_model}.png"

    plt.title(title); plt.xlabel(TOXICITY); plt.ylabel("configuration")
    path = Path(outdir) / fname
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_config_tail_rate(df: pd.DataFrame, outdir: Path | str, classifier_name:str, thr: float = 0.8) -> Optional[Path]:
    
    CLEAN_CONFIG = 'clean_config'
    df[CLEAN_CONFIG] = df[CONFIG].str.replace("_configurations", "", regex=False)
    
    tmp = df.copy()
    tmp["tail"] = (tmp[TOXICITY] >= thr).astype(int)

    rate = (tmp.groupby([CLEAN_CONFIG, MODEL])["tail"]
              .mean().mul(100).reset_index())
    if rate.empty:
        return None

    plt.figure(figsize=(12, max(6, 0.5*rate[CLEAN_CONFIG].nunique())))
    sns.barplot(data=rate, x=CLEAN_CONFIG, y="tail", hue=MODEL)
    plt.ylabel(f"% ≥ {thr}"); plt.xlabel("configuration"); plt.title("High-toxicity rate by configuration")
    plt.xticks(rotation=30, ha="right")
    path = Path(outdir) / f"12_{classifier_name}_config_tail_rate.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_all_prompt_player(df: pd.DataFrame, outdir: Path | str, classifier_name: str ="") -> List[Path]:
    """
    Generates all possible charts with the available data.
    Returns a list of paths to created files.
    """
    maybe = [
        plot_toxicity_vs_temperature(df, outdir, classifier_name),
        plot_distribution_by_model(df, outdir, classifier_name),
        plot_temperature_curve_per_model(df, outdir, classifier_name),
        plot_rates_above_threshold(df, outdir, classifier_name),
        plot_black_card_triggers(df, outdir, classifier_name),
        plot_top_plays_heatmap(df, outdir, classifier_name),
        *plot_instability(df, outdir, classifier_name),
        plot_category_comparison(df, outdir, classifier_name),
        *plot_language_risk(df, outdir, classifier_name),
        plot_config_heatmap(df, outdir, classifier_name=classifier_name),
        plot_config_distribution(df, outdir, by_model=False, classifier_name=classifier_name),
        plot_config_distribution(df, outdir, by_model=True, classifier_name=classifier_name),
        plot_config_tail_rate(df, outdir, thr=0.8, classifier_name=classifier_name)
    ]
    

#------- Judges Plots --------#