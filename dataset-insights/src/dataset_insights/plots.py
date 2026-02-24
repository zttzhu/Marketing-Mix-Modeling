"""Plot generators: histogram, correlation heatmap, missingness bar chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering — must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


_MAX_HISTOGRAM_COLS = 6  # hard cap on subplots


def _ensure_plots_dir(outdir: Path) -> Path:
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_distribution_histogram(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Histograms for up to 6 numeric columns. Returns None if no numeric cols."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    cols = numeric_cols[:_MAX_HISTOGRAM_COLS]
    n = len(cols)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Distribution of Numeric Columns", fontsize=14, y=1.02)

    for idx, col in enumerate(cols):
        ax = axes[idx // ncols][idx % ncols]
        df[col].dropna().plot(kind="hist", bins=30, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out_path = _ensure_plots_dir(outdir) / "distribution_histogram.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Correlation heatmap for numeric columns. Returns None if fewer than 2 numeric cols."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap", fontsize=13)
    plt.tight_layout()
    out_path = _ensure_plots_dir(outdir) / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_missingness_bar(df: pd.DataFrame, outdir: Path) -> Path:
    """Bar chart of missing % per column."""
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(missing_pct) * 0.5), 5))
    colors = ["#e74c3c" if v > 0 else "#95a5a6" for v in missing_pct.values]
    ax.bar(missing_pct.index, missing_pct.values, color=colors)
    ax.set_title("Missing Data (%) per Column", fontsize=13)
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Column")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(y=0, color="black", linewidth=0.8)
    plt.tight_layout()

    out_path = _ensure_plots_dir(outdir) / "missingness_bar.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path
