"""
src/visualization.py
=====================
Publication-style figures for the Streamlit "Reports" tab and for the
standalone report-generation scripts. Every function takes a dataframe (or
plain dict/list) plus a save_path and writes a PNG file there.

Color palette matches the rest of the project deliverables (forest green /
rust) for visual consistency across the proposal, slides, and this app.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN_DARK = "#1E3F26"
GREEN = "#2F5D3A"
RUST = "#A85B32"
LIGHT = "#F2F1EC"
INK = "#2A2A28"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#C9C4B8",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def _empty_axis_message(ax, message: str):
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#6B6B63", wrap=True)
    ax.set_axis_off()


_PALETTE = [RUST, GREEN_DARK, "#7A8B99", "#C9A227", "#6B4C9A", "#3E7C8C", "#B5533C", "#5B7A4A"]


def plot_mixture_triangle(df: pd.DataFrame, save_path: str, materials: Sequence[str] = None) -> str:
    """
    Composition plot for ANY number of mixture materials: one horizontal
    stacked bar per DOE run, segments = each material's fraction. Works
    uniformly whether there are 2 materials or 6. Function name kept as
    plot_mixture_triangle so the rest of the app (app.py) does not need to
    change.
    """
    if materials is None:
        from .data_utils import infer_mixture_components
        materials = infer_mixture_components(df)
    materials = [m for m in materials if m in df.columns]

    n_runs = len(df)
    fig_h = max(2.2, 0.35 * n_runs + 1.0)
    fig, ax = plt.subplots(figsize=(8, fig_h), dpi=150)
    if len(materials) < 2 or df.empty:
        _empty_axis_message(ax, "No mixture composition data available yet.")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    comp = df[materials].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    totals = comp.sum(axis=1).replace(0, 1.0)
    comp = comp.div(totals, axis=0)

    run_labels = df["Run"].astype(str).values if "Run" in df.columns else [f"#{i+1}" for i in range(n_runs)]
    y_pos = np.arange(n_runs)
    left = np.zeros(n_runs)
    for i, mat in enumerate(materials):
        vals = comp[mat].values
        ax.barh(y_pos, vals, left=left, color=_PALETTE[i % len(_PALETTE)], label=mat, height=0.65)
        left = left + vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(run_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of dry mixture")
    ax.set_title("Mixture Composition by Run", fontsize=12, color=GREEN_DARK, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(materials), 5), frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_property_ranking(ranked_df: pd.DataFrame, property_name: str, save_path: str, top_n: int = 10) -> str:
    """Horizontal bar chart: the top_n ranked formulations' predicted value for one chosen property."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    pred_col = f"Predicted_{property_name}"
    if ranked_df is None or ranked_df.empty or pred_col not in ranked_df.columns:
        _empty_axis_message(ax, f"No optimization results yet for '{property_name}'.")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    top = ranked_df.nlargest(min(top_n, len(ranked_df)), "score")
    labels = [f"#{i+1}" for i in range(len(top))]
    values = top[pred_col].astype(float).values[::-1]
    labels = labels[::-1]

    ax.barh(labels, values, color=GREEN_DARK)
    ax.set_xlabel(property_name)
    ax.set_title(f"Top Formulations \u2014 Predicted {property_name}", fontsize=12, color=GREEN_DARK, fontweight="bold")
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3g}", va="center", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_radar_comparison(ranked_df: pd.DataFrame, properties: List[str], save_path: str, top_n: int = 5) -> str:
    """
    Spider/radar chart comparing the top_n ranked formulations across the
    chosen properties. Each property axis is min-max normalized across the
    shown formulations (0 = lowest shown, 1 = highest shown) purely for
    plotting on a shared 0-1 scale -- raw predicted values are not comparable
    across different units otherwise.
    """
    fig = plt.figure(figsize=(6.5, 6.5), dpi=150)
    pred_cols = [f"Predicted_{p}" for p in properties]
    available = [p for p, c in zip(properties, pred_cols) if ranked_df is not None and c in getattr(ranked_df, "columns", [])]
    if ranked_df is None or ranked_df.empty or len(available) < 3:
        ax = fig.add_subplot(111)
        _empty_axis_message(ax, "Pick at least 3 properties with optimization results for a radar chart.")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    top = ranked_df.nlargest(min(top_n, len(ranked_df)), "score").reset_index(drop=True)
    pred_cols = [f"Predicted_{p}" for p in available]
    norm = top[pred_cols].astype(float).copy()
    for c in pred_cols:
        lo, hi = norm[c].min(), norm[c].max()
        norm[c] = 0.5 if hi <= lo else (norm[c] - lo) / (hi - lo)

    n = len(available)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax = fig.add_subplot(111, polar=True)
    for i in range(len(top)):
        values = norm.loc[i, pred_cols].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, color=_PALETTE[i % len(_PALETTE)], label=f"#{i+1}")
        ax.fill(angles, values, color=_PALETTE[i % len(_PALETTE)], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Top Formulations \u2014 Property Comparison", fontsize=12, color=GREEN_DARK, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_response_surface(df: pd.DataFrame, target: str, save_path: str,
                           x_col: str = None, y_col: str = None) -> str:
    """Filled contour of `target` over (x_col, y_col), interpolated from scattered runs.
    If x_col/y_col are not given, the first two detected mixture materials are used."""
    if x_col is None or y_col is None:
        from .data_utils import infer_mixture_components
        materials = infer_mixture_components(df)
        if len(materials) >= 2:
            x_col, y_col = materials[0], materials[1]
        else:
            x_col, y_col = x_col or "SAC", y_col or "Lime"
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    needed = {x_col, y_col, target}
    valid = df.dropna(subset=[c for c in needed if c in df.columns]) if needed.issubset(df.columns) else pd.DataFrame()
    valid = valid[pd.to_numeric(valid.get(target), errors="coerce").notna()] if not valid.empty else valid

    if valid.empty or len(valid) < 4:
        _empty_axis_message(ax, f"Not enough data yet to plot a response surface for '{target}'.")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    x = valid[x_col].astype(float).values
    y = valid[y_col].astype(float).values
    z = pd.to_numeric(valid[target], errors="coerce").astype(float).values

    try:
        contour = ax.tricontourf(x, y, z, levels=14, cmap="YlOrBr")
        fig.colorbar(contour, ax=ax, label=target)
    except Exception:
        pass
    ax.scatter(x, y, c=GREEN_DARK, s=18, zorder=3)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Response Surface \u2014 {target}", fontsize=12, color=GREEN_DARK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str) -> str:
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    if numeric.shape[1] < 2 or numeric.shape[0] < 3:
        _empty_axis_message(ax, "Not enough numeric data yet for a correlation heatmap.")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    corr = numeric.corr()
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color=INK)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation Heatmap \u2014 Composition, Process & Responses",
                 fontsize=12, color=GREEN_DARK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_importance(importances: Dict[str, float], save_path: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    if not importances:
        _empty_axis_message(ax, "No feature-importance data available yet.")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    items = sorted(importances.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    ax.barh(names, values, color=RUST)
    ax.set_xlabel("Relative importance")
    ax.set_title("Feature Importance", fontsize=12, color=GREEN_DARK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_optimization_progress(progress: List[float], save_path: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    if not progress:
        _empty_axis_message(ax, "No optimization run yet.")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    ax.plot(range(len(progress)), progress, color=GREEN_DARK, linewidth=2)
    ax.fill_between(range(len(progress)), progress, color=GREEN, alpha=0.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best overall desirability (0\u20131)")
    ax.set_title("Genetic Algorithm Optimization Progress", fontsize=12, color=GREEN_DARK, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
