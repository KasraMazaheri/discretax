"""Hyperparameter study visualizations for discretax sweeps.

Each "study" extracts a subset of the hyperparameter space from one or more
sweeps, aggregates over nuisance variables (seeds, discretization, etc.),
and produces charts. Studies are grouped by dataset in ``STUDIES``:

- cifar_init:      RT/AG initialization grids + undamped baseline (accuracy)
- cifar_disc:      discretization bar chart conditioned on AG (accuracy)
- cifar_multihead: num_heads bars across damped/undamped × bare/gated
- ppg_init:        AG grids (coarse + fine) + undamped baseline (MSE)
- ppg_multihead:   num_heads bars across bare/proj/gated for PPG damped

Usage
-----
    python cluster/study_hparams.py sweeps/
    python cluster/study_hparams.py sweeps/ --studies cifar_init
    python cluster/study_hparams.py sweeps/ --output figures/
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reuse the loader from analyze_sweep
# ---------------------------------------------------------------------------

# Allow running from repo root: add cluster/ to path so we can import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sweep import load_job_results

# ---------------------------------------------------------------------------
# Metric specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """Describes how to extract, display, and color a metric in charts."""

    column: str  # DataFrame column holding the per-run value
    label: str  # Axis/colorbar label, e.g. "Test Accuracy"
    higher_is_better: bool  # True for accuracy, False for MSE/loss
    cmap: str  # Matplotlib colormap name
    value_format: Callable[[float], str]
    std_format: Callable[[float], str]
    vmin: float | None = None  # Optional fixed color scale
    vmax: float | None = None


ACCURACY_SPEC = MetricSpec(
    column="test_accuracy",
    label="Test Accuracy",
    higher_is_better=True,
    cmap="RdYlGn",
    value_format=lambda v: f"{v * 100:.1f}%",
    std_format=lambda s: f"±{s * 100:.1f}",
    vmin=0.5,
    vmax=0.7,
)

MSE_SPEC = MetricSpec(
    column="test_mse",
    label="Test MSE",
    higher_is_better=False,
    cmap="RdYlGn_r",
    value_format=lambda v: f"{v:.4f}",
    std_format=lambda s: f"±{s:.4f}",
    vmin=0.05,
    vmax=0.15,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Columns produced by analyze_sweep.load_job_results that describe the *run
# outcome* rather than a hyperparameter. Anything matching these names (or the
# seed/path/wandb prefixes handled separately) must be excluded from nuisance
# detection so it isn't treated as a configuration axis.
RESULT_COLS: frozenset[str] = frozenset(
    {
        "job_id",
        "job_dir",
        "score",
        "test_metric",
        "test_accuracy",
        "test_loss",
        "test_mse",
        "test_mae",
        "best_metric",
        "final_step",
        "duration_seconds",
        "parameter_count",
        "median_train_examples_per_second",
        "median_eval_examples_per_second",
    }
)


def _safe_nunique(series: pd.Series) -> int:
    """Like Series.nunique() but handles unhashable types (lists, dicts)."""
    try:
        return series.nunique()
    except TypeError:
        return series.apply(str).nunique()


def _load_sweep(sweeps_root: Path, sweep_name: str) -> pd.DataFrame:
    """Load a single sweep into a DataFrame.

    ``load_job_results`` now returns ``(results, metric_key, higher_is_better)``;
    we discard the auto-detected metric here since each study pins its own
    metric via a MetricSpec. Metric columns are coerced so ±Inf (failed runs)
    becomes NaN — otherwise those runs poison mean aggregations.
    """
    sweep_dir = sweeps_root / sweep_name
    if not sweep_dir.exists():
        print(f"Error: sweep directory {sweep_dir} not found")
        return pd.DataFrame()
    results, _metric_key, _higher_is_better = load_job_results(sweep_dir)
    if not results:
        print(f"Warning: no completed results in {sweep_dir}")
        return pd.DataFrame()
    df = pd.DataFrame(results)
    for col in ("test_accuracy", "test_loss", "test_mse", "test_mae", "best_metric", "score"):
        if col in df.columns:
            df[col] = df[col].replace([float("inf"), float("-inf")], float("nan"))
    return df


def _pivot_metric_grid(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    nuisance_cols: list[str],
    spec: MetricSpec,
    agg: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-stage aggregation over a 2D grid.

    Stage 1: Average over seeds to get a per-configuration score.
             A "configuration" is defined by (row_col, col_col, *nuisance_cols).
    Stage 2: Aggregate across nuisance_cols using `agg` (mean, max, or min)
             to collapse down to the (row_col, col_col) grid.

    For lower-is-better metrics callers should pass agg="min" to select the
    best nuisance configuration per cell.

    Returns:
    -------
    (primary_pivot, std_pivot) — std from stage 2 for annotation.
    """
    value_col = spec.column
    present_nuisance = [c for c in nuisance_cols if c in df.columns]
    if present_nuisance:
        # Stage 1: mean over seeds within each full configuration
        config_cols = [row_col, col_col] + present_nuisance
        seed_avg = df.groupby(config_cols, dropna=False)[value_col].mean().reset_index()
        seed_avg = seed_avg.rename(columns={value_col: "config_score"})
        # Stage 2: aggregate across nuisance axes
        stage2 = (
            seed_avg.groupby([row_col, col_col])["config_score"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
    else:
        # No nuisance axes — compute std directly across seeds.
        stage2 = (
            df.groupby([row_col, col_col], dropna=False)[value_col]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

    primary_pivot = stage2.pivot(index=row_col, columns=col_col, values=agg)
    std_pivot = stage2.pivot(index=row_col, columns=col_col, values="std")

    # Sort axes numerically
    primary_pivot = primary_pivot.sort_index(ascending=True).reindex(
        sorted(primary_pivot.columns), axis=1
    )
    std_pivot = std_pivot.sort_index(ascending=True).reindex(sorted(std_pivot.columns), axis=1)
    return primary_pivot, std_pivot


def _count_grid(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str = "test_accuracy",
) -> pd.DataFrame:
    """Count runs per cell."""
    grouped = df.groupby([row_col, col_col])[value_col].count().reset_index()
    return (
        grouped.pivot(index=row_col, columns=col_col, values=value_col)
        .sort_index(ascending=True)
        .reindex(sorted(grouped[col_col].unique()), axis=1)
    )


def _best_agg(spec: MetricSpec) -> str:
    """Name of the stage-2 aggregation for picking the best config per cell."""
    return "max" if spec.higher_is_better else "min"


def _compute_nuisance(df: pd.DataFrame, study_axes: set[str]) -> list[str]:
    """Config columns that vary but aren't study axes, seeds, or result columns."""
    seed_cols = {c for c in df.columns if c.endswith("_seed") or c == "trainer_seed"}
    excluded_prefixes = ("paths_", "wandb_", "name")
    return [
        c
        for c in df.columns
        if c not in RESULT_COLS
        and c not in seed_cols
        and c not in study_axes
        and not any(c.startswith(p) for p in excluded_prefixes)
        and _safe_nunique(df[c]) > 1
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Publication-quality defaults
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


def plot_metric_heatmap(
    primary_pivot: pd.DataFrame,
    std_pivot: pd.DataFrame | None,
    spec: MetricSpec,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """Draw a heatmap of a metric with optional ± std annotations."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    data = primary_pivot.values.astype(float)
    std_data = std_pivot.values.astype(float) if std_pivot is not None else None

    vmin = spec.vmin if spec.vmin is not None else float(np.nanmin(data)) - 0.005
    vmax = spec.vmax if spec.vmax is not None else float(np.nanmax(data)) + 0.005

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(spec.cmap)

    im = ax.imshow(data, cmap=cmap_obj, norm=norm, aspect="auto")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "—"
            else:
                val_str = spec.value_format(val)
                if std_data is not None and not np.isnan(std_data[i, j]):
                    text = f"{val_str}\n{spec.std_format(std_data[i, j])}"
                else:
                    text = val_str
            # Choose text color based on background luminance
            bg_color = cmap_obj(norm(val)) if not np.isnan(val) else (1, 1, 1, 1)
            lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            text_color = "white" if lum < 0.5 else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="medium",
            )

    ax.set_xticks(range(len(primary_pivot.columns)))
    ax.set_xticklabels([f"{v}" for v in primary_pivot.columns])
    ax.set_yticks(range(len(primary_pivot.index)))
    ax.set_yticklabels([f"{v}" for v in primary_pivot.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(spec.label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Study definitions
# ---------------------------------------------------------------------------

_HPARAM_PREFIX = "model_backbone_kwargs_"


def _short(col: str) -> str:
    """Strip the model_backbone_kwargs_ prefix for use as an axis label."""
    return col[len(_HPARAM_PREFIX) :] if col.startswith(_HPARAM_PREFIX) else col


def _study_metric_grid(
    sweeps_root: Path,
    output_dir: Path,
    *,
    sweep_name: str,
    row_col: str,
    col_col: str,
    spec: MetricSpec,
    title: str,
    file_tag: str,
) -> None:
    """Run a 2D hyperparameter grid study: aggregate seeds + nuisance, plot heatmap."""
    print("\n" + "=" * 60)
    print(f"STUDY: {title}  [{spec.label}]")
    print("=" * 60)

    df = _load_sweep(sweeps_root, sweep_name)
    if df.empty:
        return
    for col in (row_col, col_col):
        if col not in df.columns:
            print(f"Error: column '{col}' not found. Available columns:\n  {list(df.columns)}")
            return

    row_label, col_label = _short(row_col), _short(col_col)
    print(f"\n  Using: row={row_col}, col={col_col}  metric={spec.column}")
    print(f"  {row_label} values: {sorted(df[row_col].dropna().unique())}")
    print(f"  {col_label} values: {sorted(df[col_col].dropna().unique())}")
    print(f"  Total runs: {len(df)}")

    nuisance = _compute_nuisance(df, {row_col, col_col})
    print(f"  Aggregating over seeds, then {', '.join(nuisance) or '(nothing else)'}")

    best_agg = _best_agg(spec)
    mean_pivot, std_pivot = _pivot_metric_grid(df, row_col, col_col, nuisance, spec, agg="mean")
    best_pivot, _ = _pivot_metric_grid(df, row_col, col_col, nuisance, spec, agg=best_agg)
    count_pivot = _count_grid(df, row_col, col_col, value_col=spec.column)

    print(f"\n  Grid shape: {mean_pivot.shape[0]} × {mean_pivot.shape[1]}")
    print(f"  Runs per cell: {count_pivot.min().min():.0f}–{count_pivot.max().max():.0f}")
    print(
        f"  Mean {spec.label} range: {mean_pivot.min().min():.4f} – {mean_pivot.max().max():.4f}"
    )
    lo, hi = best_pivot.min().min(), best_pivot.max().max()
    print(f"  Best {spec.label} ({best_agg}) range: {lo:.4f} – {hi:.4f}")

    nuisance_str = ", ".join(nuisance + ["seeds"]) if nuisance else "seeds"
    for agg_name, pivot, show_std in [
        ("mean", mean_pivot, std_pivot),
        (best_agg, best_pivot, None),
    ]:
        fig = plot_metric_heatmap(
            pivot,
            show_std,
            spec,
            title=f"{title}\n({agg_name} {spec.label.lower()}, aggregated over {nuisance_str})",
            xlabel=col_label,
            ylabel=row_label,
        )
        out_path = output_dir / f"study_{file_tag}_{agg_name}.png"
        fig.savefig(out_path)
        print(f"  Saved: {out_path}")
        plt.close(fig)


def study_cifar_init_rt(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 damped LinOSS — RT initialization grid (r_min × theta_max)."""
    _study_metric_grid(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-damped-sweep",
        row_col=f"{_HPARAM_PREFIX}r_min",
        col_col=f"{_HPARAM_PREFIX}theta_max",
        spec=ACCURACY_SPEC,
        title="Damped LinOSS — RT Initialization",
        file_tag="damped_init_rt",
    )


def study_cifar_init_ag(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 damped LinOSS — AG initialization grid (A_max × G_max)."""
    _study_metric_grid(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-damped-sweep-ag",
        row_col=f"{_HPARAM_PREFIX}A_max",
        col_col=f"{_HPARAM_PREFIX}G_max",
        spec=ACCURACY_SPEC,
        title="Damped LinOSS — AG Initialization",
        file_tag="damped_init_ag",
    )


def study_ppg_init_ag(sweeps_root: Path, output_dir: Path) -> None:
    """PPG regression — AG initialization grid (A_max × G_max)."""
    _study_metric_grid(
        sweeps_root,
        output_dir,
        sweep_name="ppg/damped/init-coarse",
        row_col=f"{_HPARAM_PREFIX}A_max",
        col_col=f"{_HPARAM_PREFIX}G_max",
        spec=MSE_SPEC,
        title="PPG Damped LinOSS — AG Initialization",
        file_tag="ppg_init_ag_coarse",
    )
    _study_metric_grid(
        sweeps_root,
        output_dir,
        sweep_name="ppg/damped/init-fine",
        row_col=f"{_HPARAM_PREFIX}A_max",
        col_col=f"{_HPARAM_PREFIX}G_max",
        spec=MSE_SPEC,
        title="PPG Damped LinOSS — AG Initialization",
        file_tag="ppg_init_ag_fine",
    )


def _bar_metric(
    df: pd.DataFrame,
    category_col: str,
    nuisance_cols: list[str],
    spec: MetricSpec,
    agg: str = "mean",
) -> pd.DataFrame:
    """Two-stage aggregation for a single categorical axis.

    Stage 1: Average over seeds within each full configuration.
    Stage 2: Aggregate across nuisance_cols using `agg` ("mean", "max", or "min").

    Returns a DataFrame with columns: [category_col, "value", "std", "count"].
    """
    value_col = spec.column
    present_nuisance = [c for c in nuisance_cols if c in df.columns]
    if present_nuisance:
        config_cols = [category_col] + present_nuisance
        seed_avg = df.groupby(config_cols, dropna=False)[value_col].mean().reset_index()
        seed_avg = seed_avg.rename(columns={value_col: "config_score"})
        stage2 = (
            seed_avg.groupby(category_col)["config_score"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
    else:
        stage2 = (
            df.groupby(category_col, dropna=False)[value_col]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

    stage2["value"] = stage2[agg]
    return stage2[[category_col, "value", "std", "count"]]


def plot_metric_bars(
    bar_df: pd.DataFrame,
    category_col: str,
    spec: MetricSpec,
    *,
    title: str,
    xlabel: str,
    figsize: tuple[float, float] = (7, 5),
    ylim: tuple[float | None, float | None] | None = None,
    show_std: bool = True,
    category_order: list[str] | None = None,
    params_values: np.ndarray | None = None,
) -> plt.Figure:
    """Bar chart of a metric, with an optional param-count step line on a twin axis.

    ``spec.vmin/vmax`` controls the *color* scale (for comparability across
    studies). ``ylim`` controls the y-axis range independently; leave it None
    to auto-scale so outlier values don't explode the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if category_order is not None:
        bar_df = bar_df.set_index(category_col).loc[category_order].reset_index()

    x = np.arange(len(bar_df))
    values = bar_df["value"].values
    stds = bar_df["std"].values if show_std else None

    cmap_obj = plt.get_cmap(spec.cmap)
    color_vmin = spec.vmin if spec.vmin is not None else float(np.nanmin(values)) - 0.005
    color_vmax = spec.vmax if spec.vmax is not None else float(np.nanmax(values)) + 0.005
    norm = mcolors.Normalize(vmin=color_vmin, vmax=color_vmax)
    colors = [cmap_obj(norm(v)) for v in values]

    bars = ax.bar(
        x,
        values,
        width=0.8,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        yerr=stds if show_std else None,
        capsize=4,
        error_kw={"linewidth": 1.2},
    )

    # Annotate bars above the error whisker (or bar top if no std).
    data_max = float(np.nanmax(values)) if values.size else 1.0
    data_min = float(np.nanmin(values)) if values.size else 0.0
    label_pad = max(abs(data_max - data_min) * 0.01, 1e-6)
    for i, (bar, val) in enumerate(zip(bars, values)):
        if np.isnan(val):
            continue
        label = spec.value_format(val)
        whisker_top = bar.get_height()
        if show_std and stds is not None and not np.isnan(stds[i]):
            label += f"\n{spec.std_format(stds[i])}"
            whisker_top += stds[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            whisker_top + label_pad,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bar_df[category_col].values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(spec.label, color="black")
    ax.set_title(title, pad=12)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.spines["top"].set_visible(False)
    if params_values is None:
        ax.spines["right"].set_visible(False)
    else:
        _add_params_axis(ax, x, params_values)

    fig.tight_layout()
    return fig


def _add_params_axis(ax: plt.Axes, x: np.ndarray, params_values: np.ndarray) -> None:
    """Add a twin y-axis with a step line showing parameter counts."""
    ax3 = ax.twinx()
    step_x = np.concatenate([[x[0] - 0.5], x + 0.5])
    step_y = np.concatenate([[params_values[0]], params_values])
    ax3.step(step_x, step_y, where="pre", color="gray", linewidth=1.5, alpha=0.8)
    for xi, pv in zip(x, params_values):
        if pv >= 1e6:
            lbl = f"{pv / 1e6:.2f}M"
        elif pv >= 1e3:
            lbl = f"{pv / 1e3:.1f}k"
        else:
            lbl = f"{pv:.0f}"
        ax3.text(xi, pv, lbl, ha="center", va="bottom", fontsize=8, color="gray")
    ax3.set_ylabel("# Parameters", color="gray")
    ax3.tick_params(axis="y", labelcolor="gray")
    ax3.spines["top"].set_visible(False)
    pmin, pmax = float(np.min(params_values)), float(np.max(params_values))
    pad = max((pmax - pmin) * 0.25, pmax * 0.05, 1.0)
    ax3.set_ylim(max(0, pmin - pad), pmax + pad)


def _study_metric_bars_1d(
    sweeps_root: Path,
    output_dir: Path,
    *,
    sweep_name: str,
    category_col: str,
    spec: MetricSpec,
    title: str,
    file_tag: str,
    xlabel: str | None = None,
    category_order: list[str] | None = None,
) -> None:
    """Run a single-axis study: aggregate seeds + nuisance, plot a bar chart per cell."""
    print("\n" + "=" * 60)
    print(f"STUDY: {title}  [{spec.label}]")
    print("=" * 60)

    df = _load_sweep(sweeps_root, sweep_name)
    if df.empty:
        return
    if category_col not in df.columns:
        print(
            f"Error: column '{category_col}' not found. Available columns:\n  {list(df.columns)}"
        )
        return

    category_label = _short(category_col)
    print(f"\n  {category_label} values: {sorted(df[category_col].dropna().unique())}")
    print(f"  Total runs: {len(df)}")

    nuisance = _compute_nuisance(df, {category_col})
    print(f"  Aggregating over seeds, then {', '.join(nuisance) or '(nothing else)'}")

    nuisance_str = ", ".join(nuisance + ["seeds"]) if nuisance else "seeds"
    best_agg = _best_agg(spec)
    for agg_name in ["mean", best_agg]:
        bar_df = _bar_metric(df, category_col, nuisance, spec, agg=agg_name)
        fig = plot_metric_bars(
            bar_df,
            category_col,
            spec,
            title=f"{title}\n({agg_name} {spec.label.lower()}, aggregated over {nuisance_str})",
            xlabel=xlabel if xlabel is not None else category_label,
            show_std=(agg_name == "mean"),
            category_order=category_order,
        )
        out_path = output_dir / f"study_{file_tag}_{agg_name}.png"
        fig.savefig(out_path)
        print(f"  Saved: {out_path}")
        plt.close(fig)


def study_cifar_baseline_init(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 baseline (undamped) LinOSS — initialization bar chart over A_max."""
    _study_metric_bars_1d(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-sweep",
        category_col=f"{_HPARAM_PREFIX}A_max",
        spec=ACCURACY_SPEC,
        title="Baseline LinOSS — Initialization",
        file_tag="baseline_init",
    )


def study_cifar_discretization(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 damped LinOSS — discretization comparison (conditioned on AG init)."""
    _study_metric_bars_1d(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-damped-sweep-ag",
        category_col=f"{_HPARAM_PREFIX}discretization",
        spec=ACCURACY_SPEC,
        title="Damped LinOSS — Discretization Comparison (AG init)",
        file_tag="discretization_ag",
        xlabel="Discretization",
        category_order=["EX", "IMEX", "IMEX2", "IMEX3", "IM"],
    )


def study_ppg_baseline_init(sweeps_root: Path, output_dir: Path) -> None:
    """PPG regression baseline (undamped) LinOSS — initialization bar chart over A_max."""
    _study_metric_bars_1d(
        sweeps_root,
        output_dir,
        sweep_name="ppg/undamped/oscillatory-init-sweep",
        category_col=f"{_HPARAM_PREFIX}A_max",
        spec=MSE_SPEC,
        title="PPG Baseline LinOSS — Initialization",
        file_tag="ppg_baseline_init",
    )


def _study_multihead(
    sweeps_root: Path,
    output_dir: Path,
    sweep_name: str,
    label: str,
    file_tag: str,
    *,
    use_gating: bool,
    use_output_proj: bool,
    ema_enabled: bool | None = None,
    spec: MetricSpec = ACCURACY_SPEC,
    ylim: tuple[float | None, float | None] | None = None,
    figsize: tuple[float, float] = (7, 5),
) -> None:
    """Bar chart of num_heads effect with specified gating/output_projection filter."""
    gate_str = "Gated" if use_gating else "No Gating"
    proj_str = "Output Proj" if use_output_proj else "No Output Proj"
    ema_str = "" if ema_enabled is None else f", EMA={ema_enabled}"
    print(f"\n{'=' * 60}")
    print(f"STUDY: Multi-Head ({label}) — {gate_str}, {proj_str}{ema_str}  [{spec.label}]")
    print("=" * 60)

    df = _load_sweep(sweeps_root, sweep_name)
    if df.empty:
        return

    heads_col = "model_backbone_kwargs_num_heads"
    gate_col = "model_backbone_kwargs_use_head_gating"
    proj_col = "model_backbone_kwargs_use_head_output_projection"
    ema_col = "ema_enabled"

    required = [heads_col, gate_col, proj_col] + ([ema_col] if ema_enabled is not None else [])
    for col in required:
        if col not in df.columns:
            print(f"Error: column '{col}' not found. Available columns:\n  {list(df.columns)}")
            return

    # Filter to specified gate/proj (and optionally ema) subset
    mask = (df[gate_col] == use_gating) & (df[proj_col] == use_output_proj)  # noqa: E712
    if ema_enabled is not None:
        mask = mask & (df[ema_col] == ema_enabled)  # noqa: E712
    df_filtered = df[mask].copy()
    print(
        f"\n  Total runs: {len(df)}  |  After filtering "
        f"(gate={use_gating}, proj={use_output_proj}{ema_str}): {len(df_filtered)}"
    )
    print(f"  num_heads values: {sorted(df_filtered[heads_col].dropna().unique())}")

    if df_filtered.empty:
        print("  No runs match the filter criteria, skipping")
        return

    # Merge filter columns into study_axes so they aren't treated as nuisance
    # (they're constant within df_filtered).
    study_axes = {heads_col, gate_col, proj_col}
    if ema_enabled is not None:
        study_axes.add(ema_col)
    nuisance = _compute_nuisance(df_filtered, study_axes)
    # Drop nuisance cols that are 1:1 with heads_col — they're co-varying
    # design axes, not true nuisances. Keeping them collapses seed variance
    # in stage-1 aggregation and leaves stage-2 std as NaN.
    nuisance = [c for c in nuisance if df_filtered.groupby(heads_col)[c].nunique().max() > 1]
    print(f"  Aggregating over seeds, then {', '.join(nuisance) or '(nothing else)'}")

    # Parameter count per num_heads (mean across remaining configs)
    params_by_heads: pd.Series | None = None
    if "parameter_count" in df_filtered.columns:
        params_by_heads = df_filtered.groupby(heads_col)["parameter_count"].mean().sort_index()

    best_agg = _best_agg(spec)
    for agg_name in ["mean", best_agg]:
        bar_df = _bar_metric(df_filtered, heads_col, nuisance, spec, agg=agg_name)
        show_std = agg_name == "mean"

        params_values = None
        if params_by_heads is not None:
            params_values = params_by_heads.reindex(bar_df[heads_col].values).values

        fig = plot_metric_bars(
            bar_df,
            heads_col,
            spec,
            title=(
                f"Multi-Head ({gate_str}, {proj_str}{ema_str})\n"
                f"({agg_name} {spec.label.lower()}, agg over remaining hparams & seeds)"
            ),
            xlabel="num_heads",
            ylim=ylim,
            show_std=show_std,
            figsize=figsize,
            params_values=params_values,
        )
        out_path = output_dir / f"study_multihead_{file_tag}_{agg_name}.png"
        fig.savefig(out_path)
        print(f"  Saved: {out_path}")
        plt.close(fig)


def study_cifar_multihead_bare_damped(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 damped LinOSS — num_heads bar chart, no gating or output projection."""
    _study_multihead(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-damped-gate-multihead-sweep",
        label="Damped",
        file_tag="cifar_damped_bare",
        use_gating=False,
        use_output_proj=False,
    )


def study_cifar_multihead_bare_undamped(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 undamped LinOSS — num_heads bar chart, no gating or output projection."""
    _study_multihead(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-gate-multihead-sweep",
        label="Undamped",
        file_tag="cifar_undamped_bare",
        use_gating=False,
        use_output_proj=False,
    )


def study_cifar_multihead_gated_damped(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 damped LinOSS — num_heads bar chart with gating and output projection."""
    _study_multihead(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-damped-gate-multihead-sweep",
        label="Damped",
        file_tag="cifar_damped_gated",
        use_gating=True,
        use_output_proj=True,
    )


def study_cifar_multihead_gated_undamped(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 undamped LinOSS — num_heads bar chart with gating and output projection."""
    _study_multihead(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-gate-multihead-sweep",
        label="Undamped",
        file_tag="cifar_undamped_gated",
        use_gating=True,
        use_output_proj=True,
    )


def study_ppg_multihead_bare_damped(sweeps_root: Path, output_dir: Path) -> None:
    """PPG damped LinOSS — num_heads bar chart, no gating or output projection."""
    _study_multihead(
        sweeps_root,
        output_dir,
        sweep_name="ppg/damped/param-equal-multiheading",
        label="PPG Damped",
        file_tag="ppg_damped_bare",
        use_gating=False,
        use_output_proj=False,
        spec=MSE_SPEC,
    )


# def study_ppg_multihead_bare_undamped(sweeps_root: Path, output_dir: Path) -> None:
#     _study_multihead(
#         sweeps_root, output_dir,
#         sweep_name="ppg-undamped-multiheading",
#         label="PPG Undamped", file_tag="ppg_undamped_bare",
#         use_gating=False, use_output_proj=False,
#         spec=MSE_SPEC,
#     )


def study_ppg_multihead_proj_damped(sweeps_root: Path, output_dir: Path) -> None:
    """PPG damped LinOSS — num_heads bar chart with output projection, no gating."""
    _study_multihead(
        sweeps_root,
        output_dir,
        sweep_name="ppg/damped/param-equal-multiheading",
        label="PPG Damped",
        file_tag="ppg_damped_proj",
        use_gating=False,
        use_output_proj=True,
        spec=MSE_SPEC,
    )


# def study_ppg_multihead_proj_undamped(sweeps_root: Path, output_dir: Path) -> None:
#     _study_multihead(
#         sweeps_root, output_dir,
#         sweep_name="ppg-undamped-multiheading",
#         label="PPG Undamped", file_tag="ppg_undamped_proj",
#         use_gating=False, use_output_proj=True,
#         spec=MSE_SPEC,
#     )


def study_ppg_multihead_gated_damped(sweeps_root: Path, output_dir: Path) -> None:
    """PPG damped LinOSS — num_heads bar chart with gating and output projection."""
    _study_multihead(
        sweeps_root,
        output_dir,
        sweep_name="ppg/damped/param-equal-gating-multiheading",
        label="PPG Damped",
        file_tag="ppg_damped_gated",
        use_gating=True,
        use_output_proj=True,
        spec=MSE_SPEC,
    )


# def study_ppg_multihead_gated_undamped(sweeps_root: Path, output_dir: Path) -> None:
#     _study_multihead(
#         sweeps_root, output_dir,
#         sweep_name="ppg-undamped-gating-multiheading",
#         label="PPG Undamped", file_tag="ppg_undamped_gated",
#         use_gating=True, use_output_proj=True,
#         spec=MSE_SPEC,
#     )


# ---------------------------------------------------------------------------
# Registry & CLI
# ---------------------------------------------------------------------------

STUDIES: dict[str, list[Any]] = {
    "cifar_init": [study_cifar_init_rt, study_cifar_init_ag, study_cifar_baseline_init],
    "cifar_disc": [study_cifar_discretization],
    "cifar_multihead": [
        study_cifar_multihead_bare_damped,
        study_cifar_multihead_bare_undamped,
        study_cifar_multihead_gated_damped,
        study_cifar_multihead_gated_undamped,
    ],
    "ppg_init": [study_ppg_init_ag, study_ppg_baseline_init],
    "ppg_multihead": [
        study_ppg_multihead_bare_damped,
        study_ppg_multihead_proj_damped,
        study_ppg_multihead_gated_damped,
    ],
}


def main() -> None:
    """Run selected hyperparameter studies and save figures."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter study visualizations for discretax sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sweeps_root",
        type=Path,
        help="Root directory containing sweep subdirectories (e.g. sweeps/)",
    )
    parser.add_argument(
        "--studies",
        nargs="*",
        default=None,
        help=f"Which studies to run. Options: {', '.join(STUDIES.keys())}. Default: all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures (default: figures/)",
    )
    args = parser.parse_args()

    if not args.sweeps_root.exists():
        print(f"Error: sweeps root {args.sweeps_root} not found")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # Determine which studies to run
    if args.studies is None:
        study_names = list(STUDIES.keys())
    else:
        study_names = args.studies
        for name in study_names:
            if name not in STUDIES:
                print(f"Error: unknown study '{name}'. Options: {', '.join(STUDIES.keys())}")
                return

    for name in study_names:
        for study_fn in STUDIES[name]:
            study_fn(args.sweeps_root, args.output)

    print("\n" + "=" * 60)
    print(f"All figures saved to {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
