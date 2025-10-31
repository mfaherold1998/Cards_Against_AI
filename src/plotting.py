
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _ensure_outdir(dirpath: Path | str) -> Path:
    out = Path(dirpath)
    out.mkdir(parents=True, exist_ok=True)
    return out

def _resolve_cols(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect and normalize key column names so plotting code works across datasets.
    Returns a dict with keys:
      toxicity_col, language_col, config_col, model_col, temp_col, play_col
      attrs: list of attribute columns available for category plots.
    """
    # toxicity column (Detoxify vs Perspective)
    toxicity_candidates = ["detox_toxicity", "TOXICITY", "toxicity"]
    tox_col = next((c for c in toxicity_candidates if c in df.columns), None)

    # language
    lang_col = "language" if "language" in df.columns else ("lang" if "lang" in df.columns else None)

    # config / configuration
    cfg_col = "config_name" if "config_name" in df.columns else ("config" if "config" in df.columns else None)

    # model, temperature, play
    model_col = "model" if "model" in df.columns else None
    temp_col = "temperature" if "temperature" in df.columns else None
    play_col = "play" if "play" in df.columns else None

    # attribute columns (Detoxify prefixed vs Perspective ALLCAPS)
    detox_attrs = ["detox_insult", "detox_threat", "detox_identity_attack",
                   "detox_obscene", "detox_severe_toxicity"]
    persp_attrs = ["INSULT", "THREAT", "IDENTITY_ATTACK",
                   "PROFANITY", "SEVERE_TOXICITY", "SEXUALLY_EXPLICIT"]
    attrs_present = [c for c in detox_attrs + persp_attrs if c in df.columns]

    return dict(
        toxicity_col=tox_col, language_col=lang_col, config_col=cfg_col,
        model_col=model_col, temp_col=temp_col, play_col=play_col,
        attrs=attrs_present
    )

def _parse_play_row(x) -> Tuple[Optional[str], Tuple[str, ...]]:
    """
    Normaliza 'play' que puede venir como lista real [B, W1, W2, ...]
    o como string representando esa lista. Devuelve (black_id, tuple(white_ids_sorted)).
    """
    # Try literal_eval if it's a string
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            # fallback: try to split rudimentariamente
            return None, tuple()
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        return None, tuple()
    b = x[0]
    w = tuple(sorted(str(i) for i in x[1:])) if len(x) > 1 else tuple()
    return str(b), w

def _build_play_key(black_id: Optional[str], white_ids: Tuple[str, ...]) -> str:
    return f"B:{black_id or ''}|W:{','.join(white_ids)}"


def plot_toxicity_vs_temperature(df: pd.DataFrame, outdir: Path | str, model_name:str) -> Optional[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["model_col"] or not cols["temp_col"]:
        return None
    outdir = _ensure_outdir(outdir)

    tox = cols["toxicity_col"]
    g = (df.groupby([cols["model_col"], cols["temp_col"]])[tox]
           .agg(["mean", "count", "std"]).reset_index())
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))

    plt.figure()
    for m, sub in g.groupby(cols["model_col"]):
        sub = sub.sort_values(cols["temp_col"])
        x = pd.to_numeric(sub[cols["temp_col"]], errors='coerce').to_numpy()
        y = pd.to_numeric(sub["mean"], errors='coerce').to_numpy()
        se = pd.to_numeric(sub["sem"], errors='coerce').fillna(0).to_numpy()
        plt.errorbar(x, y, yerr=1.96*se, marker="o", label=str(m))
    plt.xlabel("Temperature"); plt.ylabel("Toxicity"); plt.legend(); plt.title("Toxicity vs Temperature")
    path = Path(outdir) / f"{model_name}_toxicity_vs_temperature.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_distribution_by_model(df: pd.DataFrame, outdir: Path | str, model_name:str) -> Optional[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["model_col"]:
        return None
    outdir = _ensure_outdir(outdir)

    tox = cols["toxicity_col"]
    models = df[cols["model_col"]].dropna().astype(str).unique().tolist()
    data = [df.loc[df[cols["model_col"]] == m, tox].dropna().values for m in models]
    if not any(len(d) for d in data):
        return None

    plt.figure()
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(models)+1), models, rotation=30)
    plt.ylabel(tox); plt.title("Distribution per model")
    path = Path(outdir) / f"{model_name}_distribution_by_model.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_temperature_curve_per_model(df: pd.DataFrame, outdir: Path | str, model_name:str) -> List[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["model_col"] or not cols["temp_col"]:
        return []
    outdir = _ensure_outdir(outdir)

    tox = cols["toxicity_col"]
    g = (df.groupby([cols["model_col"], cols["temp_col"]])[tox]
           .agg(["mean", "count", "std"]).reset_index())
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))

    paths: List[Path] = []
    for m, sub in g.groupby(cols["model_col"]):
        sub = sub.sort_values(cols["temp_col"])
        plt.figure()
        x = pd.to_numeric(sub[cols["temp_col"]], errors='coerce').to_numpy()
        y = pd.to_numeric(sub["mean"], errors='coerce').to_numpy()
        se = pd.to_numeric(sub["sem"], errors='coerce').fillna(0).to_numpy()
        plt.plot(x, y, marker="o")
        plt.fill_between(x, y-1.96*se, y+1.96*se, alpha=0.2)
        plt.title(f"{m}: Toxicity vs Temperature")
        plt.xlabel("Temperature"); plt.ylabel("Mean toxicity")
        safe_m = str(m).replace("/", "_").replace(":", "_")
        path = Path(outdir) / f"{model_name}_temperature_curve_{safe_m}.png"
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
        paths.append(path)
    return paths

def plot_rates_above_threshold(df: pd.DataFrame, outdir: Path | str, model_name:str, thr: float = 0.8) -> Optional[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["model_col"] or not cols["temp_col"]:
        return None
    outdir = _ensure_outdir(outdir)
    tox = cols["toxicity_col"]

    tmp = df.copy()
    tmp["tail"] = (tmp[tox] >= thr).astype(int)

    rate = (tmp.groupby([cols["model_col"], cols["temp_col"]])["tail"]
              .mean().mul(100).reset_index())

    plt.figure()
    for m, sub in rate.groupby(cols["model_col"]):
        sub = sub.sort_values(cols["temp_col"])
        plt.plot(sub[cols["temp_col"]], sub["tail"], marker="o", label=str(m))
    plt.xlabel("Temperature"); plt.ylabel(f"% ≥ {thr}"); plt.legend(); plt.title("High tail by model")
    path = Path(outdir) / f"{model_name}_rates_above_threshold.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_black_card_triggers(df: pd.DataFrame, outdir: Path | str, model_name:str, top_k: int = 30) -> Optional[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["model_col"] or not cols["play_col"]:
        return None
    outdir = _ensure_outdir(outdir)
    tox = cols["toxicity_col"]

    tmp = df.copy()
    parsed = tmp[cols["play_col"]].apply(_parse_play_row)
    tmp["black_id"] = parsed.apply(lambda t: t[0])

    top_black = (tmp.groupby("black_id")[tox]
                   .mean().sort_values(ascending=False).head(top_k).index)

    mat = (tmp[tmp["black_id"].isin(top_black)]
           .pivot_table(index="black_id", columns=cols["model_col"],
                        values=tox, aggfunc="mean"))
    if mat.empty:
        return None

    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, cmap="viridis", cbar_kws={"label": "Mean toxicity"})
    plt.title("Top toxic black cards (mean per model)")
    path = Path(outdir) / f"{model_name}_black_card_triggers.png"
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path

def plot_top_plays_heatmap(df: pd.DataFrame, outdir: Path | str, model_name:str, top_k: int = 30) -> Optional[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["model_col"] or not cols["play_col"]:
        return None
    outdir = _ensure_outdir(outdir)
    tox = cols["toxicity_col"]

    tmp = df.copy()
    parsed = tmp[cols["play_col"]].apply(_parse_play_row)
    tmp["black_id"] = parsed.apply(lambda t: t[0])
    tmp["white_ids"] = parsed.apply(lambda t: t[1])
    tmp["play_key"] = [ _build_play_key(b, w) for b, w in parsed ]

    top_keys = (tmp.groupby("play_key")[tox]
                  .mean().sort_values(ascending=False).head(top_k).index)
    mat = (tmp[tmp["play_key"].isin(top_keys)]
           .pivot_table(index="play_key", columns=cols["model_col"],
                        values=tox, aggfunc="mean"))
    if mat.empty:
        return None

    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, cmap="magma", cbar_kws={"label": "Mean toxicity"})
    plt.title("Top toxic plays (black + white) — mean per model")
    path = Path(outdir) / f"{model_name}_top_plays_heatmap.png"
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path

def plot_instability(df: pd.DataFrame, outdir: Path | str, model_name:str, top_n: int = 10) -> List[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["model_col"] or not cols["temp_col"] or not cols["play_col"]:
        return []
    outdir = _ensure_outdir(outdir)
    tox = cols["toxicity_col"]

    tmp = df.copy()
    parsed = tmp[cols["play_col"]].apply(_parse_play_row)
    tmp["play_key"] = [ _build_play_key(b, w) for b, w in parsed ]

    stability = (tmp.groupby([cols["model_col"], cols["temp_col"], "play_key"])[tox]
                   .agg(["mean", "std", "count"]).reset_index()
                   .rename(columns={"mean": "tox_mean", "std": "tox_std", "count": "n"}))
    if stability.empty:
        return []

    # Scatter
    stability["size"] = stability["n"] * 500 / max(stability["n"].max(), 1)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=stability, x="tox_mean", y="tox_std",
        hue=cols["model_col"], size="size", sizes=(50, 500), alpha=0.7, legend="full"
    )
    plt.title("Mean Toxicity vs. Instability (Standard Deviation)")
    plt.xlabel("Average Toxicity (tox_mean)"); plt.ylabel("Instability (tox_std)")
    p1 = Path(outdir) / f"{model_name}_instability_scatter.png"
    plt.tight_layout(); plt.savefig(p1, dpi=200); plt.close()

    # Top-N bar
    top_unstable = stability.sort_values("tox_std", ascending=False).head(top_n).copy()
    top_unstable["ranking"] = [f"Play_{i+1}" for i in range(len(top_unstable))]
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_unstable, x="tox_std", y="ranking", hue=cols["model_col"], dodge=False)
    plt.title(f"Top {top_n} Most Unstable Combinations (High STD)")
    plt.xlabel("STD of Toxicity (tox_std)"); plt.ylabel("Play")
    plt.legend(title="Model"); plt.grid(axis="x", linestyle="--", alpha=0.7)
    p2 = Path(outdir) / f"{model_name}_instability_bar.png"
    plt.tight_layout(); plt.savefig(p2, dpi=200); plt.close()

    return [p1, p2]

def plot_category_comparison(df: pd.DataFrame, outdir: Path | str, model_name:str) -> Optional[Path]:
    cols = _resolve_cols(df)
    if not cols["model_col"] or not cols["attrs"]:
        return None
    outdir = _ensure_outdir(outdir)

    present = cols["attrs"]
    agg = df.groupby(cols["model_col"])[present].mean()

    # Make nice display names
    pretty = {c: (c.replace("detox_", "").replace("_", " ").title()) for c in present}
    agg = agg.rename(columns=pretty)

    ax = agg.plot(kind="bar")
    plt.ylabel("Average per attribute"); plt.title("Profile of attributes per model"); plt.xticks(rotation=30)
    path = Path(outdir) / f"{model_name}_category_comparison.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_language_risk(df: pd.DataFrame, outdir: Path | str, model_name:str) -> List[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["language_col"] or not cols["model_col"]:
        return []
    outdir = _ensure_outdir(outdir)
    tox = cols["toxicity_col"]

    def summarize_toxicity(d: pd.DataFrame) -> pd.DataFrame:
        return (d.groupby(cols["model_col"])[tox]
                  .agg(mean="mean",
                       p50=lambda s: s.quantile(.5),
                       p90=lambda s: s.quantile(.9),
                       p95=lambda s: s.quantile(.95))
                  .round(3))

    paths: List[Path] = []
    for L in df[cols["language_col"]].dropna().unique():
        sub = df[df[cols["language_col"]] == L]
        out = summarize_toxicity(sub)
        ax = out[["mean", "p50", "p90", "p95"]].plot(kind="bar")
        plt.title(f"Toxicity profile ({L}) per model")
        plt.ylabel("Score"); plt.xlabel("Model"); plt.xticks(rotation=30)
        path = Path(outdir) / f"{model_name}_language_risk_{L}.png"
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
        paths.append(path)
    return paths

def plot_config_heatmap(df: pd.DataFrame, outdir: Path | str, model_name:str, min_n: int = 1) -> Optional[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["config_col"] or not cols["model_col"]:
        return None
    outdir = _ensure_outdir(outdir)
    tox = cols["toxicity_col"]

    stats = (df.groupby([cols["config_col"], cols["model_col"]])[tox]
               .agg(["mean","count"]).reset_index())
    stats = stats[stats["count"] >= min_n]
    if stats.empty:
        return None

    mat = stats.pivot(index=cols["config_col"], columns=cols["model_col"], values="mean")
    if mat.empty:
        return None

    plt.figure(figsize=(10, max(6, 0.5*len(mat.index))))
    sns.heatmap(mat, cmap="viridis", cbar_kws={"label": "Mean toxicity"})
    plt.title("Mean toxicity by configuration × model")
    path = Path(outdir) / f"{model_name}_config_heatmap_mean_toxicity.png"
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path

def plot_config_distribution(df: pd.DataFrame, outdir: Path | str, model_name:str, by_model: bool = False) -> Optional[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["config_col"]:
        return None
    outdir = _ensure_outdir(outdir)

    plt.figure(figsize=(12, max(6, 0.5 * df[cols['config_col']].nunique())))
    if by_model and cols["model_col"]:
        sns.violinplot(data=df, x=cols["toxicity_col"], y=cols["config_col"],
                       hue=cols["model_col"], cut=0, inner="quartile", density_norm="width")
        plt.legend(title="Model", bbox_to_anchor=(1.04, 1), loc="upper left")
        title = "Toxicity distribution by configuration (split by model)"
        fname = f"{model_name}_config_distribution_by_model.png"
    else:
        sns.violinplot(data=df, x=cols["toxicity_col"], y=cols["config_col"],
                       cut=0, inner="quartile", density_norm="width")
        title = "Toxicity distribution by configuration (all models)"
        fname = f"{model_name}_config_distribution.png"

    plt.title(title); plt.xlabel(cols["toxicity_col"]); plt.ylabel("configuration")
    path = Path(outdir) / fname
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_config_tail_rate(df: pd.DataFrame, outdir: Path | str, model_name:str, thr: float = 0.8) -> Optional[Path]:
    cols = _resolve_cols(df)
    if not cols["toxicity_col"] or not cols["config_col"] or not cols["model_col"]:
        return None
    outdir = _ensure_outdir(outdir)
    tox = cols["toxicity_col"]

    tmp = df.copy()
    tmp["tail"] = (tmp[tox] >= thr).astype(int)

    rate = (tmp.groupby([cols["config_col"], cols["model_col"]])["tail"]
              .mean().mul(100).reset_index())
    if rate.empty:
        return None

    plt.figure(figsize=(12, max(6, 0.5*rate[cols["config_col"]].nunique())))
    sns.barplot(data=rate, x=cols["config_col"], y="tail", hue=cols["model_col"])
    plt.ylabel(f"% ≥ {thr}"); plt.xlabel("configuration"); plt.title("High-toxicity rate by configuration")
    plt.xticks(rotation=30, ha="right")
    path = Path(outdir) / f"{model_name}_config_tail_rate.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path


def plot_all(df: pd.DataFrame, outdir: Path | str = "./plots", model_name: str ="") -> List[Path]:
    """
    Generates all possible charts with the available data.
    Returns a list of paths to created files.
    """
    outdir = _ensure_outdir(outdir)
    paths: List[Path] = []

    maybe = [
        plot_toxicity_vs_temperature(df, outdir, model_name),
        plot_distribution_by_model(df, outdir, model_name),
        *plot_temperature_curve_per_model(df, outdir, model_name),
        plot_rates_above_threshold(df, outdir, model_name),
        plot_black_card_triggers(df, outdir, model_name),
        plot_top_plays_heatmap(df, outdir, model_name),
        *plot_instability(df, outdir, model_name),
        plot_category_comparison(df, outdir, model_name),
        *plot_language_risk(df, outdir, model_name),
    ]
    paths.extend([p for p in maybe if p is not None])
    return paths

def plot_all_configs(df: pd.DataFrame, outdir: Path | str = "./plots", model_name: str ="") -> List[Path]:
    outdir = _ensure_outdir(outdir)
    paths: List[Path] = []

    maybe = [
        plot_config_heatmap(df, outdir, model_name=model_name),
        plot_config_distribution(df, outdir, by_model=False, model_name=model_name),
        plot_config_distribution(df, outdir, by_model=True, model_name=model_name),
        plot_config_tail_rate(df, outdir, thr=0.8, model_name=model_name),
    ]
    paths.extend([p for p in maybe if p is not None])
    return paths
