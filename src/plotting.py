from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def _ensure_outdir(dirpath: Path | str) -> Path:
    out = Path(dirpath)
    out.mkdir(parents=True, exist_ok=True)
    return out

def _has_cols(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(c in df.columns for c in cols)

# Graph 1:Toxicity vs Temperature
def plot_toxicity_vs_temperature(df: pd.DataFrame, outdir: Path | str) -> Optional[Path]:
    cols = ["model", "temperature", "detox_toxicity"]
    if not _has_cols(df, cols):
        return None
    outdir = _ensure_outdir(outdir)

    g = (df.groupby(["model", "temperature"])["detox_toxicity"]
           .agg(["mean", "count", "std"])
           .reset_index())
    g["sem"] = g["std"] / np.sqrt(g["count"])

    plt.figure()
    for m, sub in g.groupby("model"):
        sub = sub.sort_values("temperature")
        plt.errorbar(sub["temperature"], sub["mean"], yerr=1.96*sub["sem"], marker="o", label=m)
    plt.xlabel("Temperature"); plt.ylabel("Toxicity"); plt.legend(); plt.title("Toxicity vs Temperature")
    path = outdir / "toxicity_vs_temperature.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

# Graph 2: Full shape of the distribution by model
def plot_distribution_by_model(df: pd.DataFrame, outdir: Path | str) -> Optional[Path]:
    cols = ["model", "detox_toxicity"]
    if not _has_cols(df, cols):
        return None
    outdir = _ensure_outdir(outdir)

    models = df["model"].dropna().unique().tolist()
    data = [df.loc[df["model"] == m, "detox_toxicity"].dropna().values for m in models]

    plt.figure()
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(models)+1), models, rotation=30)
    plt.ylabel("detox_toxicity"); plt.title("Distribution per model")
    path = outdir / "distribution_by_model.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

# Graph 3: Temperature curve
def plot_temperature_curve_per_model(df: pd.DataFrame, outdir: Path | str) -> List[Path]:
    cols = ["model", "temperature", "detox_toxicity"]
    if not _has_cols(df, cols):
        return []
    outdir = _ensure_outdir(outdir)

    g = (df.groupby(["model", "temperature"])["detox_toxicity"]
           .agg(["mean", "count", "std"])
           .reset_index())
    g["sem"] = g["std"] / np.sqrt(g["count"])

    paths: List[Path] = []
    for m in g["model"].unique():
        sub = g[g["model"] == m].sort_values("temperature")
        plt.figure()
        plt.plot(sub["temperature"], sub["mean"], marker="o")
        plt.fill_between(sub["temperature"], sub["mean"]-1.96*sub["sem"], sub["mean"]+1.96*sub["sem"], alpha=0.2)
        plt.title(f"{m}: Toxicity vs Temperature"); plt.xlabel("Temperature"); plt.ylabel("Mean")
        path = outdir / f"temperature_curve_{m}.png"
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
        paths.append(path)
    return paths

# Graph 4: Rates above threshold
def plot_rates_above_threshold(df: pd.DataFrame, outdir: Path | str) -> Optional[Path]:
    cols = ["model", "temperature", "detox_tox_gt_08"]
    if not _has_cols(df, cols):
        return None
    outdir = _ensure_outdir(outdir)

    rate = (df.groupby(["model", "temperature"])[["detox_tox_gt_08"]]
              .mean().mul(100).reset_index())

    plt.figure()
    for m, sub in rate.groupby("model"):
        sub = sub.sort_values("temperature")
        plt.plot(sub["temperature"], sub["detox_tox_gt_08"], marker="o", label=m)
    plt.xlabel("Temperature"); plt.ylabel("% ≥ 0.8"); plt.legend(); plt.title("High tail by model")
    path = outdir / "rates_above_threshold.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

# Graph 5: Black card triggers
def _parse_play_row(x: Sequence[str]) -> Tuple[Optional[str], Tuple[str, ...]]:
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        return None, tuple()
    b = x[0]
    w = tuple(sorted(x[1:])) if len(x) > 1 else tuple()
    return b, w

def plot_black_card_triggers(df: pd.DataFrame, outdir: Path | str) -> Optional[Path]:
    cols = ["play", "model", "detox_toxicity"]
    if not _has_cols(df, cols):
        return None
    outdir = _ensure_outdir(outdir)

    tmp = df.copy()
    parsed = tmp["play"].apply(_parse_play_row)
    tmp["black_id"] = parsed.apply(lambda t: t[0])

    top_black = (tmp.groupby("black_id")["detox_toxicity"]
                   .mean()
                   .sort_values(ascending=False)
                   .head(30).index)

    mat = (tmp[tmp["black_id"].isin(top_black)]
           .pivot_table(index="black_id", columns="model",
                        values="detox_toxicity", aggfunc="mean"))
    if mat.empty:
        return None

    plt.figure(figsize=(10, 8))
    plt.imshow(mat.values, aspect="auto")
    plt.colorbar(label="Mean Toxicity", shrink=0.7)
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=30)
    plt.title("Top toxic black cards (mean per model)")
    path = outdir / "black_card_triggers.png"
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path

def plot_top_plays_heatmap(df: pd.DataFrame, outdir: Path | str) -> Optional[Path]:
    cols = ["play", "model", "detox_toxicity"]
    if not _has_cols(df, cols):
        return None
    outdir = _ensure_outdir(outdir)

    tmp = df.copy()
    parsed = tmp["play"].apply(_parse_play_row)
    tmp["black_id"] = parsed.apply(lambda t: t[0])
    tmp["white_ids"] = parsed.apply(lambda t: t[1])
    tmp["play_key"] = [f"B:{b or ''}|W:{','.join(w)}" for b, w in parsed]

    top_keys = (tmp.groupby("play_key")["detox_toxicity"]
                  .mean().sort_values(ascending=False).head(30).index)
    mat = (tmp[tmp["play_key"].isin(top_keys)]
           .pivot_table(index="play_key", columns="model",
                        values="detox_toxicity", aggfunc="mean"))
    if mat.empty:
        return None

    plt.figure(figsize=(10, 8))
    plt.imshow(mat.values, aspect="auto")
    plt.colorbar(label="Mean Toxicity", shrink=0.7)
    plt.yticks(range(len(mat.index)), range(len(mat.index)))
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=30)
    plt.title("Top plays (black + white) more toxic (mean per model)")
    path = outdir / "top_plays_heatmap.png"
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path

# Graph 6: Stability per round
def plot_instability(df: pd.DataFrame, outdir: Path | str, top_n: int = 10) -> List[Path]:
    cols = ["model", "temperature", "play", "detox_toxicity"]
    if not _has_cols(df, cols):
        return []
    outdir = _ensure_outdir(outdir)

    tmp = df.copy()
    parsed = tmp["play"].apply(_parse_play_row)
    tmp["play_key"] = [f"B:{b or ''}|W:{','.join(w)}" for b, w in parsed]

    stability = (tmp.groupby(["model", "temperature", "play_key"])["detox_toxicity"]
                   .agg(["mean", "std", "count"])
                   .reset_index()
                   .rename(columns={"mean": "tox_mean", "std": "tox_std", "count": "n"}))
    if stability.empty:
        return []

    # Scatter
    stability["size"] = stability["n"] * 500 / max(stability["n"].max(), 1)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=stability, x="tox_mean", y="tox_std",
        hue="model", size="size", sizes=(50, 500), alpha=0.7, legend="full"
    )
    plt.title("Mean Toxicity vs. Instability (Standard Deviation)")
    plt.xlabel("Average Toxicity (tox_mean)"); plt.ylabel("Instability (tox_std)")
    p1 = outdir / "instability_scatter.png"
    plt.tight_layout(); plt.savefig(p1, dpi=200); plt.close()

    # Top-N bar
    top_unstable = stability.sort_values("tox_std", ascending=False).head(top_n).copy()
    top_unstable["ranking"] = [f"Play_{i+1}" for i in range(len(top_unstable))]
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_unstable, x="tox_std", y="ranking", hue="model", dodge=False)
    plt.title(f"Top {top_n} Most Unstable Combinations (High STD)")
    plt.xlabel("STD of Toxicity (tox_std)"); plt.ylabel("Play")
    plt.legend(title="Model"); plt.grid(axis="x", linestyle="--", alpha=0.7)
    p2 = outdir / "instability_bar.png"
    plt.tight_layout(); plt.savefig(p2, dpi=200); plt.close()

    return [p1, p2]

# Graph 7: Category comparison (insult, threat, identity_attack…)
def plot_category_comparison(df: pd.DataFrame, outdir: Path | str) -> Optional[Path]:
    attrs = ["detox_insult", "detox_threat", "detox_identity_attack", "detox_obscene", "detox_severe_toxicity"]
    cols = ["model"] + attrs
    if not _has_cols(df, cols):
        return None
    outdir = _ensure_outdir(outdir)

    # Asegura que no falten columnas (si alguna no la produjo el checkpoint)
    present = [c for c in attrs if c in df.columns]
    if not present:
        return None

    agg = df.groupby("model")[present].mean()
    ax = agg.plot(kind="bar")
    plt.ylabel("Average per attribute"); plt.title("Profile of attributes per model"); plt.xticks(rotation=30)
    path = outdir / "category_comparison.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

# Graph 8: Language risk : how models handle extreme cases (the tail of the distribution) per-language.
def plot_language_risk(df: pd.DataFrame, outdir: Path | str) -> List[Path]:
    cols = ["language", "model", "detox_toxicity"]
    if not _has_cols(df, cols):
        return []
    outdir = _ensure_outdir(outdir)

    def summarize_toxicity(d: pd.DataFrame) -> pd.DataFrame:
        return (d.groupby("model")["detox_toxicity"]
                  .agg(mean="mean",
                       p50=lambda s: s.quantile(.5),
                       p90=lambda s: s.quantile(.9),
                       p95=lambda s: s.quantile(.95))
                  .round(3))

    paths: List[Path] = []
    for L in df["language"].dropna().unique():
        sub = df[df["language"] == L]
        out = summarize_toxicity(sub)
        ax = out[["mean", "p50", "p90", "p95"]].plot(kind="bar")
        plt.title(f"Toxicity profile ({L}) per model")
        plt.ylabel("Score"); plt.xlabel("Model"); plt.xticks(rotation=30)
        path = outdir / f"language_risk_{L}.png"
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
        paths.append(path)
    return paths

def plot_all(df: pd.DataFrame, outdir: Path | str = "./plots") -> List[Path]:
    """
    Generates all possible charts with the available data.
    Returns a list of paths to created files.
    """

    timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S") 
    outdir = Path(outdir)
    outdir = outdir.parent / f"{outdir.name}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []

    maybe = [
        plot_toxicity_vs_temperature(df, outdir),
        plot_distribution_by_model(df, outdir),
        *plot_temperature_curve_per_model(df, outdir),
        plot_rates_above_threshold(df, outdir),
        plot_black_card_triggers(df, outdir),
        plot_top_plays_heatmap(df, outdir),
        *plot_instability(df, outdir),
        plot_category_comparison(df, outdir),
        *plot_language_risk(df, outdir),
    ]
    paths.extend([p for p in maybe if p is not None])
    return paths