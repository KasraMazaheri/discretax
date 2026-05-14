"""Hyperparameter study visualizations for discretax sweeps.

Each "study" extracts a subset of the hyperparameter space from one or more
sweeps, aggregates over nuisance variables (seeds, discretization, etc.),
and produces charts.

- <dataset>_init:      AG init grid + undamped baseline (driven by DATASETS)
- cifar_init_rt:       CIFAR-only RT init grid
- cifar_disc:          discretization bar chart conditioned on AG (CIFAR)
- <dataset>_multihead: num_heads bars (driven by MULTIHEAD_STUDIES)
- <dataset>_scenarios: damping × heads × output-proj (driven by SCENARIO_DATASETS)

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

# Allow running from repo root: add cluster/ to path so we can import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sweep import load_job_results

# ---------------------------------------------------------------------------
# Metric specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """How to extract, display, and color a metric in charts."""

    column: str
    label: str
    higher_is_better: bool
    cmap: str
    value_format: Callable[[float], str]
    std_format: Callable[[float], str]
    vmin: float | None = None
    vmax: float | None = None


ACCURACY_SPEC = MetricSpec(
    column="test_metric",
    label="Test Accuracy",
    higher_is_better=True,
    cmap="RdYlGn",
    value_format=lambda v: f"{v * 100:.1f}%",
    std_format=lambda s: f"±{s * 100:.1f}",
    vmin=0.0,
    vmax=1.0,
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

MAE_SPEC = MetricSpec(
    column="test_mae",
    label="Test MAE",
    higher_is_better=False,
    cmap="RdYlGn_r",
    value_format=lambda v: f"{v:.4f}",
    std_format=lambda s: f"±{s:.4f}",
    vmin=0.1,
    vmax=1.0,
)

LTSF_MSE_SPEC = MetricSpec(
    column="test_mse",
    label="Test MSE",
    higher_is_better=False,
    cmap="RdYlGn_r",
    value_format=lambda v: f"{v:.4f}",
    std_format=lambda s: f"±{s:.4f}",
    vmin=0.1,
    vmax=1.5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HPARAM_PREFIX = "model_backbone_kwargs_"


def _bk(name: str) -> str:
    """Build a model_backbone_kwargs_<name> column key."""
    return f"{_HPARAM_PREFIX}{name}"


def _short(col: str) -> str:
    """Strip the model_backbone_kwargs_ prefix for use as an axis label."""
    return col[len(_HPARAM_PREFIX) :] if col.startswith(_HPARAM_PREFIX) else col


# Columns produced by analyze_sweep.load_job_results that describe the *run
# outcome* rather than a hyperparameter. Excluded from nuisance detection.
RESULT_COLS: frozenset[str] = frozenset(
    {
        "job_id",
        "job_dir",
        "score",
        "test_metric",
        "test_loss",
        "test_mse",
        "test_mae",
        "test_accuracy",
        "best_val_metric",
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
    """Load a single sweep into a DataFrame, coercing ±Inf to NaN."""
    sweep_dir = sweeps_root / sweep_name
    if not sweep_dir.exists():
        print(f"Error: sweep directory {sweep_dir} not found")
        return pd.DataFrame()
    results, _, _ = load_job_results(sweep_dir)
    if not results:
        print(f"Warning: no completed results in {sweep_dir}")
        return pd.DataFrame()
    df = pd.DataFrame(results)
    for col in ("test_metric", "best_val_metric", "score"):
        if col in df.columns:
            df[col] = df[col].replace([float("inf"), float("-inf")], float("nan"))
    return df


def _best_agg(spec: MetricSpec) -> str:
    """Stage-2 aggregation name for picking the best config per cell."""
    return "max" if spec.higher_is_better else "min"


def _compute_nuisance(df: pd.DataFrame, study_axes: set[str]) -> list[str]:
    """Config columns that vary but aren't study axes, seeds, or result columns."""
    seed_cols = {
        c for c in df.columns if c.endswith("_seed") or c == "trainer_seed" or c == "seed"
    }
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


def _two_stage_agg(
    df: pd.DataFrame,
    group_cols: list[str],
    nuisance_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    """Aggregate seed-mean per config, then aggregate across nuisance.

    Stage 1: mean over seeds within each (group + nuisance) config.
    Stage 2: aggregate across nuisance to get mean/std/min/max/count per group.
    """
    present_nuisance = [c for c in nuisance_cols if c in df.columns]
    if present_nuisance:
        config_cols = group_cols + present_nuisance
        seed_avg = df.groupby(config_cols, dropna=False)[value_col].mean().reset_index()
        seed_avg = seed_avg.rename(columns={value_col: "config_score"})
        return (
            seed_avg.groupby(group_cols)["config_score"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
    return (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )


def _pivot_metric_grid(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    nuisance_cols: list[str],
    spec: MetricSpec,
    agg: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-stage aggregation over a 2D grid; returns (primary_pivot, std_pivot)."""
    stage2 = _two_stage_agg(df, [row_col, col_col], nuisance_cols, spec.column)
    primary = stage2.pivot(index=row_col, columns=col_col, values=agg)
    stds = stage2.pivot(index=row_col, columns=col_col, values="std")
    primary = primary.sort_index().reindex(sorted(primary.columns), axis=1)
    stds = stds.sort_index().reindex(sorted(stds.columns), axis=1)
    return primary, stds


def _bar_metric(
    df: pd.DataFrame,
    category_col: str,
    nuisance_cols: list[str],
    spec: MetricSpec,
    agg: str = "mean",
) -> pd.DataFrame:
    """Two-stage aggregation along one axis; returns [category, value, std, count]."""
    stage2 = _two_stage_agg(df, [category_col], nuisance_cols, spec.column)
    stage2["value"] = stage2[agg]
    return stage2[[category_col, "value", "std", "count"]]


def _count_grid(df: pd.DataFrame, row_col: str, col_col: str, value_col: str) -> pd.DataFrame:
    """Count non-null `value_col` runs per cell."""
    grouped = df.groupby([row_col, col_col])[value_col].count().reset_index()
    return (
        grouped.pivot(index=row_col, columns=col_col, values=value_col)
        .sort_index()
        .reindex(sorted(grouped[col_col].unique()), axis=1)
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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


def _norm_for(spec: MetricSpec, values: np.ndarray) -> mcolors.Normalize:
    """Color normalization from spec, falling back to data range."""
    vmin = spec.vmin if spec.vmin is not None else float(np.nanmin(values)) - 0.005
    vmax = spec.vmax if spec.vmax is not None else float(np.nanmax(values)) + 0.005
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def plot_metric_heatmap(
    primary_pivot: pd.DataFrame,
    std_pivot: pd.DataFrame | None,
    spec: MetricSpec,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """Heatmap of a metric with optional ± std annotations."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    data = primary_pivot.values.astype(float)
    std_data = std_pivot.values.astype(float) if std_pivot is not None else None
    norm = _norm_for(spec, data)
    cmap_obj = plt.get_cmap(spec.cmap)
    im = ax.imshow(data, cmap=cmap_obj, norm=norm, aspect="auto")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "—"
                bg = (1, 1, 1, 1)
            else:
                text = spec.value_format(val)
                if std_data is not None and not np.isnan(std_data[i, j]):
                    text += f"\n{spec.std_format(std_data[i, j])}"
                bg = cmap_obj(norm(val))
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if lum < 0.5 else "black",
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


def _add_params_axis(ax: plt.Axes, x: np.ndarray, params_values: np.ndarray) -> None:
    """Twin y-axis with a step line showing parameter counts."""
    ax3 = ax.twinx()
    step_x = np.concatenate([[x[0] - 0.5], x + 0.5])
    step_y = np.concatenate([[params_values[0]], params_values])
    ax3.step(step_x, step_y, where="pre", color="gray", linewidth=1.5, alpha=0.8)
    for xi, pv in zip(x, params_values):
        lbl = f"{pv / 1e6:.2f}M" if pv >= 1e6 else f"{pv / 1e3:.1f}k" if pv >= 1e3 else f"{pv:.0f}"
        ax3.text(xi, pv, lbl, ha="center", va="bottom", fontsize=8, color="gray")
    ax3.set_ylabel("# Parameters", color="gray")
    ax3.tick_params(axis="y", labelcolor="gray")
    ax3.spines["top"].set_visible(False)
    pmin, pmax = float(np.min(params_values)), float(np.max(params_values))
    pad = max((pmax - pmin) * 0.25, pmax * 0.05, 1.0)
    ax3.set_ylim(max(0, pmin - pad), pmax + pad)


def plot_ag_surface_3d(
    row_vals: np.ndarray,
    col_vals: np.ndarray,
    mean_grid: np.ndarray,
    std_grid: np.ndarray,
    spec: MetricSpec,
    *,
    title: str,
    row_label: str,
    col_label: str,
    log_scale: bool = False,
) -> plt.Figure:
    """3D scatter of (row, col, mean_metric) with a 1/σ²-weighted plane of best fit.

    When log_scale=True the plane is fit in log₁₀(row) × log₁₀(col) space and
    the scatter plot uses log₁₀-transformed coordinates on the x/y axes.
    Points with missing std are still shown but excluded from the plane fit.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    R_grid, C_grid = np.meshgrid(row_vals, col_vals, indexing="ij")
    flat_mask = ~np.isnan(mean_grid)
    fit_mask = flat_mask & ~np.isnan(std_grid) & (std_grid > 0)

    R_all = R_grid[flat_mask].astype(float)
    C_all = C_grid[flat_mask].astype(float)
    Z_all = mean_grid[flat_mask].astype(float)
    R_fit = R_grid[fit_mask].astype(float)
    C_fit = C_grid[fit_mask].astype(float)
    Z_fit = mean_grid[fit_mask].astype(float)
    S_fit = std_grid[fit_mask].astype(float)

    if log_scale:
        R_plot = np.log10(R_all)
        C_plot = np.log10(C_all)
        R_fit_t = np.log10(R_fit)
        C_fit_t = np.log10(C_fit)
        plot_row_label = f"log₁₀({row_label})"
        plot_col_label = f"log₁₀({col_label})"
        r_min = float(np.log10(row_vals.min()))
        r_max = float(np.log10(row_vals.max()))
        c_min = float(np.log10(col_vals.min()))
        c_max = float(np.log10(col_vals.max()))
    else:
        R_plot = R_all
        C_plot = C_all
        R_fit_t = R_fit
        C_fit_t = C_fit
        plot_row_label = row_label
        plot_col_label = col_label
        r_min, r_max = float(row_vals.min()), float(row_vals.max())
        c_min, c_max = float(col_vals.min()), float(col_vals.max())

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    norm = _norm_for(spec, mean_grid)
    scatter = ax.scatter(
        R_plot,
        C_plot,
        Z_all,
        c=Z_all,
        cmap=spec.cmap,
        norm=norm,
        s=70,
        edgecolors="black",
        linewidth=0.4,
        depthshade=True,
        zorder=5,
    )

    plane_label = ""
    if len(R_fit_t) >= 3:
        W_sqrt = np.sqrt(1.0 / S_fit**2)
        X = np.column_stack([R_fit_t, C_fit_t, np.ones_like(R_fit_t)])
        coeffs, _, _, _ = np.linalg.lstsq(X * W_sqrt[:, None], Z_fit * W_sqrt, rcond=None)
        a, b, c = coeffs

        R_range = np.linspace(r_min, r_max, 30)
        C_range = np.linspace(c_min, c_max, 30)
        R_mesh, C_mesh = np.meshgrid(R_range, C_range)
        ax.plot_surface(
            R_mesh,
            C_mesh,
            a * R_mesh + b * C_mesh + c,
            alpha=0.25,
            color="steelblue",
            zorder=1,
        )
        plane_label = (
            f"\nplane: {spec.label} = {a:+.4f}·{plot_row_label} "
            f"{b:+.4f}·{plot_col_label} {c:+.4f}  (weighted by 1/σ²)"
        )

    ax.set_xlabel(plot_row_label)
    ax.set_ylabel(plot_col_label)
    ax.set_zlabel(spec.label)
    ax.set_title(title + plane_label, pad=12)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.1)
    cbar.set_label(spec.label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


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

    ``spec.vmin/vmax`` controls *color*; ``ylim`` controls the y-axis range.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if category_order is not None:
        bar_df = bar_df.set_index(category_col).loc[category_order].reset_index()

    x = np.arange(len(bar_df))
    values = bar_df["value"].values
    stds = bar_df["std"].values if show_std else None

    cmap_obj = plt.get_cmap(spec.cmap)
    norm = _norm_for(spec, values)
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


# ---------------------------------------------------------------------------
# Generic study runners
# ---------------------------------------------------------------------------


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
    plot_3d: bool = False,
    log_scale: bool = False,
) -> None:
    """2D hyperparameter grid: aggregate seeds + nuisance, plot mean & best heatmaps."""
    print(f"\n{'=' * 60}\nSTUDY: {title}  [{spec.label}]\n{'=' * 60}")
    df = _load_sweep(sweeps_root, sweep_name)
    if df.empty:
        return
    for col in (row_col, col_col):
        if col not in df.columns:
            print(f"Error: column '{col}' not found. Available:\n  {list(df.columns)}")
            return

    row_label, col_label = _short(row_col), _short(col_col)
    print(f"\n  {row_label} values: {sorted(df[row_col].dropna().unique())}")
    print(f"  {col_label} values: {sorted(df[col_col].dropna().unique())}")
    print(f"  Total runs: {len(df)}")

    nuisance = _compute_nuisance(df, {row_col, col_col})
    print(f"  Aggregating over seeds, then {', '.join(nuisance) or '(nothing else)'}")
    nuisance_str = ", ".join(nuisance + ["seeds"]) if nuisance else "seeds"

    best_agg = _best_agg(spec)
    mean_pivot, std_pivot = _pivot_metric_grid(df, row_col, col_col, nuisance, spec, "mean")
    best_pivot, _ = _pivot_metric_grid(df, row_col, col_col, nuisance, spec, best_agg)
    count_pivot = _count_grid(df, row_col, col_col, spec.column)

    print(f"\n  Grid shape: {mean_pivot.shape[0]} × {mean_pivot.shape[1]}")
    print(f"  Runs per cell: {count_pivot.min().min():.0f}–{count_pivot.max().max():.0f}")
    print(f"  Mean range: {mean_pivot.min().min():.4f} – {mean_pivot.max().max():.4f}")
    print(
        f"  Best ({best_agg}) range: {best_pivot.min().min():.4f} – {best_pivot.max().max():.4f}"
    )

    for agg_name, pivot, show_std in [
        ("mean", mean_pivot, std_pivot),
        (best_agg, best_pivot, None),
    ]:
        fig = plot_metric_heatmap(
            pivot,
            show_std,
            spec,
            title=f"{title}\n({agg_name} {spec.label.lower()}, agg over {nuisance_str})",
            xlabel=col_label,
            ylabel=row_label,
        )
        out_path = output_dir / f"study_{file_tag}_{agg_name}.png"
        fig.savefig(out_path)
        print(f"  Saved: {out_path}")
        plt.close(fig)

    if plot_3d:
        row_vals = np.array(mean_pivot.index, dtype=float)
        col_vals = np.array(mean_pivot.columns, dtype=float)
        fig = plot_ag_surface_3d(
            row_vals,
            col_vals,
            mean_pivot.values.astype(float),
            std_pivot.values.astype(float),
            spec,
            title=f"{title}\n(mean {spec.label.lower()}, agg over {nuisance_str})",
            row_label=row_label,
            col_label=col_label,
            log_scale=log_scale,
        )
        out_path = output_dir / f"study_{file_tag}_3d.png"
        fig.savefig(out_path)
        print(f"  Saved: {out_path}")
        plt.close(fig)


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
    """Single-axis study: aggregate seeds + nuisance, plot mean & best bars."""
    print(f"\n{'=' * 60}\nSTUDY: {title}  [{spec.label}]\n{'=' * 60}")
    df = _load_sweep(sweeps_root, sweep_name)
    if df.empty:
        return
    if category_col not in df.columns:
        print(f"Error: column '{category_col}' not found. Available:\n  {list(df.columns)}")
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
            title=f"{title}\n({agg_name} {spec.label.lower()}, agg over {nuisance_str})",
            xlabel=xlabel if xlabel is not None else category_label,
            show_std=(agg_name == "mean"),
            category_order=category_order,
        )
        out_path = output_dir / f"study_{file_tag}_{agg_name}.png"
        fig.savefig(out_path)
        print(f"  Saved: {out_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-dataset init/baseline studies (config-driven)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset config for the standard AG-init grid + undamped baseline studies."""

    display: str  # used in figure titles
    spec: MetricSpec
    ag_sweep: str  # AG-init grid sweep
    baseline_sweep: str | None = None


# Adding a new dataset: drop a DatasetConfig in here. file_tag is the dict key.
DATASETS: dict[str, DatasetConfig] = {
    "cifar": DatasetConfig(
        "Damped LinOSS", ACCURACY_SPEC, "cifar10-linoss-damped-sweep-ag", "cifar10-linoss-sweep"
    ),
    "ppg": DatasetConfig(
        "PPG Damped LinOSS", MSE_SPEC, "ppg-init", "ppg/undamped/oscillatory-init-sweep"
    ),
    "weather": DatasetConfig("Weather Damped LinOSS", LTSF_MSE_SPEC, "weather-init", None),
    "etth1": DatasetConfig("ETTh1 Damped LinOSS", LTSF_MSE_SPEC, "etth1-init", None),
    "etth2": DatasetConfig("ETTh2 Damped LinOSS", LTSF_MSE_SPEC, "etth2-init", None),
    "ettm1": DatasetConfig("ETTm1 Damped LinOSS", LTSF_MSE_SPEC, "ettm1-init", None),
    "electricity": DatasetConfig(
        "Electricity Damped LinOSS", LTSF_MSE_SPEC, "electricity-init", None
    ),
    "scp1": DatasetConfig(
        "SCP1 Damped LinOSS", ACCURACY_SPEC, "scp1-init-sweep", "scp1-undamped-init"
    ),
    "ethanol": DatasetConfig(
        "Ethanol Damped LinOSS", ACCURACY_SPEC, "ethanol-init-sweep", "ethanol-undamped-init"
    ),
    "heartbeat": DatasetConfig(
        "Heartbeat Damped LinOSS", ACCURACY_SPEC, "heartbeat-init-sweep", "heartbeat/undamped/init"
    ),
    "scp2": DatasetConfig(
        "SCP2 Damped LinOSS", ACCURACY_SPEC, "scp2-init-sweep", "scp2/undamped/init"
    ),
    "motor": DatasetConfig(
        "Motor Damped LinOSS", ACCURACY_SPEC, "motor-init-sweep", "motor/undamped/init"
    ),
    "eigenworms": DatasetConfig(
        "EigenWorms Damped LinOSS",
        ACCURACY_SPEC,
        "eigenworms-init-sweep",
        "eigenworms/undamped/init",
    ),
}


def make_init_study(key: str) -> Callable[[Path, Path], None]:
    """Build a study fn that runs the AG-init grid + undamped baseline bars."""
    cfg = DATASETS[key]

    def run(sweeps_root: Path, output_dir: Path) -> None:
        _study_metric_grid(
            sweeps_root,
            output_dir,
            sweep_name=cfg.ag_sweep,
            row_col=_bk("A_max"),
            col_col=_bk("G_max"),
            spec=cfg.spec,
            title=f"{cfg.display} — AG Initialization",
            file_tag=f"{key}_init_ag",
            plot_3d=True,
            log_scale=True,
        )
        if cfg.baseline_sweep is not None:
            _study_metric_bars_1d(
                sweeps_root,
                output_dir,
                sweep_name=cfg.baseline_sweep,
                category_col=_bk("A_max"),
                spec=cfg.spec,
                title=f"{cfg.display.replace('Damped', 'Baseline')} — Initialization",
                file_tag=f"{key}_baseline_init",
            )

    run.__name__ = f"study_{key}_init"
    run.__doc__ = f"{cfg.display} — AG init grid + undamped baseline init."
    return run


# ---------------------------------------------------------------------------
# CIFAR-only studies
# ---------------------------------------------------------------------------


def study_cifar_init_rt(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 damped LinOSS — RT init grid (r_min × theta_max)."""
    _study_metric_grid(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-damped-sweep",
        row_col=_bk("r_min"),
        col_col=_bk("theta_max"),
        spec=ACCURACY_SPEC,
        title="Damped LinOSS — RT Initialization",
        file_tag="damped_init_rt",
    )


def study_cifar_discretization(sweeps_root: Path, output_dir: Path) -> None:
    """CIFAR-10 damped LinOSS — discretization comparison (conditioned on AG init)."""
    _study_metric_bars_1d(
        sweeps_root,
        output_dir,
        sweep_name="cifar10-linoss-damped-sweep-ag",
        category_col=_bk("discretization"),
        spec=ACCURACY_SPEC,
        title="Damped LinOSS — Discretization Comparison (AG init)",
        file_tag="discretization_ag",
        xlabel="Discretization",
        category_order=["EX", "IMEX", "IMEX2", "IMEX3", "IM"],
    )


# ---------------------------------------------------------------------------
# Multi-head studies (config-driven)
# ---------------------------------------------------------------------------


def _study_multihead(
    sweeps_root: Path,
    output_dir: Path,
    *,
    sweep_name: str,
    label: str,
    file_tag: str,
    use_gating: bool,
    use_output_proj: bool,
    spec: MetricSpec = ACCURACY_SPEC,
    ylim: tuple[float | None, float | None] | None = None,
) -> None:
    """Bar chart of num_heads effect with a gating/output_projection filter."""
    gate_str = "Gated" if use_gating else "No Gating"
    proj_str = "Output Proj" if use_output_proj else "No Output Proj"
    header = f"STUDY: Multi-Head ({label}) — {gate_str}, {proj_str}  [{spec.label}]"
    print(f"\n{'=' * 60}\n{header}\n{'=' * 60}")

    df = _load_sweep(sweeps_root, sweep_name)
    if df.empty:
        return

    heads_col = _bk("num_heads")
    gate_col = _bk("use_head_gating")
    proj_col = _bk("use_head_output_projection")
    for col in (heads_col, gate_col, proj_col):
        if col not in df.columns:
            print(f"Error: column '{col}' not found. Available:\n  {list(df.columns)}")
            return

    df_filt = df[(df[gate_col] == use_gating) & (df[proj_col] == use_output_proj)].copy()
    print(f"\n  Total runs: {len(df)}  |  After filtering: {len(df_filt)}")
    print(f"  num_heads values: {sorted(df_filt[heads_col].dropna().unique())}")
    if df_filt.empty:
        print("  No runs match the filter, skipping")
        return

    nuisance = _compute_nuisance(df_filt, {heads_col, gate_col, proj_col})
    # Drop nuisance cols 1:1 with heads_col — they're co-varying design axes,
    # not true nuisances. Keeping them collapses seed variance.
    nuisance = [c for c in nuisance if df_filt.groupby(heads_col)[c].nunique().max() > 1]
    print(f"  Aggregating over seeds, then {', '.join(nuisance) or '(nothing else)'}")

    params_by_heads = (
        df_filt.groupby(heads_col)["parameter_count"].mean().sort_index()
        if "parameter_count" in df_filt.columns
        else None
    )

    best_agg = _best_agg(spec)
    for agg_name in ["mean", best_agg]:
        bar_df = _bar_metric(df_filt, heads_col, nuisance, spec, agg=agg_name)
        params_values = (
            params_by_heads.reindex(bar_df[heads_col].values).values
            if params_by_heads is not None
            else None
        )
        fig = plot_metric_bars(
            bar_df,
            heads_col,
            spec,
            title=f"Multi-Head ({gate_str}, {proj_str})\n"
            f"({agg_name} {spec.label.lower()}, agg over remaining hparams & seeds)",
            xlabel="num_heads",
            ylim=ylim,
            show_std=(agg_name == "mean"),
            params_values=params_values,
        )
        out_path = output_dir / f"study_multihead_{file_tag}_{agg_name}.png"
        fig.savefig(out_path)
        print(f"  Saved: {out_path}")
        plt.close(fig)


@dataclass(frozen=True)
class MultiheadStudy:
    """One multi-head study: a sweep slice filtered by (gating, output_proj)."""

    file_tag: str
    sweep_name: str
    label: str
    use_gating: bool
    use_output_proj: bool
    spec: MetricSpec = ACCURACY_SPEC
    ylim: tuple[float | None, float | None] | None = None


_CIFAR_DAMPED_MH = "cifar10-linoss-damped-gate-multihead-sweep"
_CIFAR_UNDAMPED_MH = "cifar10-linoss-gate-multihead-sweep"

# Each registry key bundles the multihead studies that share a chart family.
MULTIHEAD_STUDIES: dict[str, list[MultiheadStudy]] = {
    "cifar_multihead": [
        MultiheadStudy(
            "cifar_damped_bare", _CIFAR_DAMPED_MH, "Damped", False, False, ylim=(0.0, 1.0)
        ),
        MultiheadStudy("cifar_undamped_bare", _CIFAR_UNDAMPED_MH, "Undamped", False, False),
        MultiheadStudy("cifar_damped_gated", _CIFAR_DAMPED_MH, "Damped", True, True),
        MultiheadStudy("cifar_undamped_gated", _CIFAR_UNDAMPED_MH, "Undamped", True, True),
    ],
    "ppg_multihead": [
        MultiheadStudy(
            "ppg_damped_bare",
            "ppg/damped/hidden-param-equal-multiheading",
            "PPG Damped",
            False,
            False,
            spec=MSE_SPEC,
        ),
        MultiheadStudy(
            "ppg_damped_proj",
            "ppg/damped/hidden-param-equal-multiheading-proj",
            "PPG Damped",
            False,
            True,
            spec=MSE_SPEC,
        ),
        MultiheadStudy(
            "ppg_damped_gated",
            "ppg/damped/param-equal-gating-multiheading",
            "PPG Damped",
            True,
            True,
            spec=MSE_SPEC,
        ),
    ],
}


def make_multihead_study(study: MultiheadStudy) -> Callable[[Path, Path], None]:
    """Build a study fn from a MultiheadStudy spec."""

    def run(sweeps_root: Path, output_dir: Path) -> None:
        _study_multihead(
            sweeps_root,
            output_dir,
            sweep_name=study.sweep_name,
            label=study.label,
            file_tag=study.file_tag,
            use_gating=study.use_gating,
            use_output_proj=study.use_output_proj,
            spec=study.spec,
            ylim=study.ylim,
        )

    run.__name__ = f"study_multihead_{study.file_tag}"
    return run


# ---------------------------------------------------------------------------
# Scenario comparison: damping × multi-head × output-projection
# ---------------------------------------------------------------------------
# A "scenario" picks out a slice of one sweep that shares a particular
# (damping mode, num_heads, use_output_proj) combination. The set of scenarios
# for a given dataset defines what gets compared in the plots below.
#
# Adding a new dataset: build a list of ScenarioDef entries and register a
# ScenarioDataset in SCENARIO_DATASETS. For datasets that use the same sweep
# layout as SCP1/Ethanol, default_scenarios(slug) does it in one line.

# Hyperparameters that define a "configuration" within a scenario sweep — the
# axes we average over (along with seeds) when computing per-config seed-mean.
_SCENARIO_CONFIG_COLS: tuple[str, ...] = (
    "model_hidden_dim",
    _bk("state_dim"),
    _bk("num_blocks"),
    "optimizer_learning_rate",
)
_SCENARIO_VAL_COL = "best_val_metric"


@dataclass(frozen=True)
class ScenarioDef:
    """One scenario: a slice of a sweep selected by (heads, output_proj)."""

    label: str  # e.g. "damped\nH=2, proj"
    sweep_name: str
    num_heads: int
    use_output_proj: bool


@dataclass(frozen=True)
class ScenarioDataset:
    """A dataset's scenario comparison: display name, file prefix, scenarios."""

    display: str
    file_prefix: str
    scenarios: tuple[ScenarioDef, ...]
    spec: MetricSpec = ACCURACY_SPEC
    config_cols: tuple[str, ...] = _SCENARIO_CONFIG_COLS
    n_configs: int = 81  # SCP1/Ethanol both have 3×3×3×3 = 81 configs


def default_scenarios(slug: str) -> tuple[ScenarioDef, ...]:
    """Build the standard 6-scenario tuple.

    Assumes sweeps live at:

      {slug}/undamped/notime-sweep
      {slug}/damped/notime-sweep
      {slug}/damped/notime-multiheading-sweep

    For other layouts, write the ScenarioDef list out by hand.
    """
    return (
        ScenarioDef("undamped\nH=1", f"{slug}/undamped/notime-sweep", 1, False),
        ScenarioDef("damped\nH=1", f"{slug}/damped/notime-sweep", 1, False),
        ScenarioDef("damped\nH=2", f"{slug}/damped/notime-multiheading-sweep", 2, False),
        ScenarioDef("damped\nH=2, proj", f"{slug}/damped/notime-multiheading-sweep", 2, True),
        ScenarioDef("damped\nH=4", f"{slug}/damped/notime-multiheading-sweep", 4, False),
        ScenarioDef("damped\nH=4, proj", f"{slug}/damped/notime-multiheading-sweep", 4, True),
    )


SCENARIO_DATASETS: dict[str, ScenarioDataset] = {
    "scp1": ScenarioDataset("SCP1", "scp1", default_scenarios("scp1")),
    "ethanol": ScenarioDataset("Ethanol", "ethanol", default_scenarios("ethanol")),
}


def _scenario_config_means(
    df: pd.DataFrame, test_col: str, config_cols: tuple[str, ...]
) -> pd.DataFrame:
    """Seed-mean per configuration for both test and val.

    Returns a DataFrame with columns ("test", "test_std", "val"). Rows where
    the test mean is NaN are dropped. "val" may be NaN for older runs without
    best_val_metric — that only disables val-based selection.
    """
    missing = [c for c in config_cols if c not in df.columns]
    if missing:
        raise KeyError(f"scenario sweep missing config columns: {missing}")
    has_val = _SCENARIO_VAL_COL in df.columns
    gb = df.groupby(list(config_cols), dropna=False)
    test_stats = gb[test_col].agg(test="mean", test_std="std")
    if has_val:
        grouped = test_stats.join(gb[_SCENARIO_VAL_COL].mean().rename("val"))
    else:
        grouped = test_stats.assign(val=float("nan"))
    return grouped.dropna(subset=["test"])


def _select_idx(scores: pd.Series, higher_is_better: bool) -> Any:
    """Index of the best (or worst) score, skipping NaN."""
    clean = scores.dropna()
    if clean.empty:
        return None
    return clean.idxmax() if higher_is_better else clean.idxmin()


def _load_scenarios(
    sweeps_root: Path, dataset: ScenarioDataset, *, select_by: str
) -> dict[str, pd.DataFrame]:
    """Load per-config seed-mean scores for every scenario.

    Each value has columns ("test", "test_std", "val"), one row per config.
    """
    heads_col = _bk("num_heads")
    proj_col = _bk("use_head_output_projection")
    spec = dataset.spec
    value_col = spec.column

    # Cache per-sweep loads since e.g. damped multiheading is reused across
    # multiple scenarios.
    cache: dict[str, pd.DataFrame] = {}
    scenarios: dict[str, pd.DataFrame] = {}
    for sc in dataset.scenarios:
        if sc.sweep_name not in cache:
            cache[sc.sweep_name] = _load_sweep(sweeps_root, sc.sweep_name)
        df = cache[sc.sweep_name]
        flat_label = sc.label.replace("\n", " ")
        if df.empty:
            print(f"  [{flat_label}] sweep empty, skipping")
            continue

        mask = pd.Series(True, index=df.index)
        if heads_col in df.columns:
            mask &= df[heads_col] == sc.num_heads
        if proj_col in df.columns:
            mask &= df[proj_col] == sc.use_output_proj
        sub = df[mask]
        if sub.empty:
            print(
                f"  [{flat_label}] no runs match heads={sc.num_heads}, proj={sc.use_output_proj}"
            )
            continue

        config_scores = _scenario_config_means(sub, value_col, dataset.config_cols)
        scenarios[sc.label] = config_scores

        ranking_col = "val" if select_by == "val" else "test"
        idx = _select_idx(config_scores[ranking_col], spec.higher_is_better)
        chosen_test = config_scores.loc[idx, "test"] if idx is not None else float("nan")
        print(
            f"  [{flat_label}] runs={len(sub)}  configs={len(config_scores)}  "
            f"chosen test (by {ranking_col})={spec.value_format(chosen_test)}"
        )
    return scenarios


def _scenario_selected_test(
    scenarios: dict[str, pd.DataFrame], select_by: str, higher_is_better: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Per scenario, return (test_mean, test_std) for the selected config.

    ``select_by="val"`` is the standard "model selection on val" protocol.
    """
    ranking_col = "val" if select_by == "val" else "test"
    means, stds = [], []
    for df in scenarios.values():
        idx = _select_idx(df[ranking_col], higher_is_better)
        if idx is None:
            means.append(float("nan"))
            stds.append(float("nan"))
        else:
            means.append(float(df.loc[idx, "test"]))
            stds.append(float(df.loc[idx, "test_std"]))
    return np.array(means), np.array(stds)


def _plot_scenario_best_bars(
    scenarios: dict[str, pd.DataFrame],
    spec: MetricSpec,
    title: str,
    *,
    select_by: str,
) -> plt.Figure:
    """Bar chart of the selected per-config seed-mean test score per scenario."""
    labels = list(scenarios.keys())
    best, best_std = _scenario_selected_test(scenarios, select_by, spec.higher_is_better)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    cmap_obj = plt.get_cmap(spec.cmap)
    norm = _norm_for(spec, best)
    colors = [cmap_obj(norm(v)) for v in best]
    yerr = np.where(np.isnan(best_std), 0.0, best_std)
    bars = ax.bar(
        x,
        best,
        width=0.75,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        yerr=yerr,
        capsize=4,
        error_kw={"linewidth": 1.2, "ecolor": "black"},
    )
    pad = max(abs(np.nanmax(best) - np.nanmin(best)) * 0.01, 1e-6)
    for bar, v, s in zip(bars, best, best_std):
        if np.isnan(v):
            continue
        whisker_top = bar.get_height() + (s if not np.isnan(s) else 0.0)
        label = spec.value_format(v) + (f"\n{spec.std_format(s)}" if not np.isnan(s) else "")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            whisker_top + pad,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
        )

    selection_desc = "max val" if select_by == "val" else "max test"
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{spec.label} of config selected by {selection_desc}")
    ax.set_title(title, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_scenario_histograms(
    scenarios: dict[str, pd.DataFrame],
    spec: MetricSpec,
    title: str,
    *,
    select_by: str,
    n_configs: int,
) -> plt.Figure:
    """Small-multiples histogram of per-config seed-mean test scores."""
    labels = list(scenarios.keys())
    n = len(labels)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))

    all_vals = np.concatenate([df["test"].values for df in scenarios.values()])
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = max((hi - lo) * 0.05, 1e-3)
    bins = np.linspace(lo - pad, hi + pad, 21)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).flatten()

    ranking_col = "val" if select_by == "val" else "test"
    for ax, label in zip(axes, labels):
        df = scenarios[label]
        test_values = df["test"].values
        ax.hist(test_values, bins=bins, color="#4c72b0", edgecolor="black", linewidth=0.4)
        idx = _select_idx(df[ranking_col], spec.higher_is_better)
        chosen = float("nan") if idx is None else float(df.loc[idx, "test"])
        mean_val = float(np.nanmean(test_values))
        if not np.isnan(chosen):
            ax.axvline(chosen, color="crimson", linestyle="--", linewidth=1.2, label="selected")
        if not np.isnan(mean_val):
            ax.axvline(mean_val, color="black", linestyle=":", linewidth=1.2, label="mean")
        ax.text(
            0.98,
            0.95,
            f"selected={spec.value_format(chosen)}\nmean={spec.value_format(mean_val)}\nn={len(test_values)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.set_title(label.replace("\n", " — "), fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel(f"Per-config seed-mean {spec.label.lower()}")
    for ax in axes[::ncols]:
        ax.set_ylabel(f"Count (of {n_configs} configs)")

    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def _plot_scenario_histogram_overlay(
    scenarios: dict[str, pd.DataFrame],
    spec: MetricSpec,
    title: str,
    *,
    select_by: str,
    n_configs: int,
) -> plt.Figure:
    """All scenarios overlaid on a single axis for direct distribution comparison."""
    labels = list(scenarios.keys())
    all_vals = np.concatenate([df["test"].values for df in scenarios.values()])
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = max((hi - lo) * 0.05, 1e-3)
    bins = np.linspace(lo - pad, hi + pad, 25)

    cmap_obj = plt.get_cmap("tab10")
    colors = [cmap_obj(i) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ranking_col = "val" if select_by == "val" else "test"
    for label, color in zip(labels, colors):
        df = scenarios[label]
        test_values = df["test"].values
        ax.hist(
            test_values,
            bins=bins,
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=1.0,
            label=label.replace("\n", " "),
        )
        idx = _select_idx(df[ranking_col], spec.higher_is_better)
        if idx is not None:
            ax.axvline(
                float(df.loc[idx, "test"]), color=color, linestyle="-", linewidth=1.6, alpha=0.9
            )
        mean_val = float(np.nanmean(test_values))
        if not np.isnan(mean_val):
            ax.axvline(mean_val, color=color, linestyle=":", linewidth=1.4, alpha=0.9)

    ax.set_xlabel(f"Per-config seed-mean {spec.label.lower()}")
    ax.set_ylabel(f"Count (of {n_configs} configs)")
    ax.set_title(title, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    scenario_legend = ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.add_artist(scenario_legend)
    style_handles = [
        plt.Line2D([], [], color="gray", linestyle="-", linewidth=1.6, label="selected"),
        plt.Line2D([], [], color="gray", linestyle=":", linewidth=1.4, label="mean"),
    ]
    ax.legend(handles=style_handles, loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def _run_scenario_study(
    sweeps_root: Path,
    output_dir: Path,
    dataset: ScenarioDataset,
    *,
    select_by: str,
) -> None:
    """Run bar + small-multiples + overlay plots for one (dataset, select_by)."""
    if select_by not in ("test", "val"):
        raise ValueError(f"select_by must be 'test' or 'val', got {select_by!r}")

    spec = dataset.spec
    print(
        f"\n{'=' * 60}\nSTUDY: {dataset.display} Scenario Comparison "
        f"(select_by={select_by})  [{spec.label}]\n{'=' * 60}"
    )

    scenarios = _load_scenarios(sweeps_root, dataset, select_by=select_by)
    if not scenarios:
        print("  No scenarios loaded, skipping")
        return

    desc = "val-selected" if select_by == "val" else "test-selected"
    fp = dataset.file_prefix

    plots = [
        (
            _plot_scenario_best_bars,
            f"{dataset.display} — {desc.capitalize()} test accuracy by scenario",
            f"study_{fp}_scenarios_best_by_{select_by}.png",
            {},
        ),
        (
            _plot_scenario_histograms,
            f"{dataset.display} — Distribution of per-config seed-mean test accuracy ({desc})",
            f"study_{fp}_scenarios_hist_by_{select_by}.png",
            {"n_configs": dataset.n_configs},
        ),
        (
            _plot_scenario_histogram_overlay,
            f"{dataset.display} — Overlaid per-config test-accuracy distributions ({desc})",
            f"study_{fp}_scenarios_hist_overlay_by_{select_by}.png",
            {"n_configs": dataset.n_configs},
        ),
    ]
    for plot_fn, title, fname, extra in plots:
        fig = plot_fn(scenarios, spec, title, select_by=select_by, **extra)
        out_path = output_dir / fname
        fig.savefig(out_path)
        print(f"  Saved: {out_path}")
        plt.close(fig)


def make_scenario_study(key: str) -> Callable[[Path, Path], None]:
    """Build a study fn for a scenario dataset (runs both selections)."""
    dataset = SCENARIO_DATASETS[key]

    def run(sweeps_root: Path, output_dir: Path) -> None:
        _run_scenario_study(sweeps_root, output_dir, dataset, select_by="test")
        _run_scenario_study(sweeps_root, output_dir, dataset, select_by="val")

    run.__name__ = f"study_{key}_scenarios"
    run.__doc__ = f"{dataset.display} — scenario comparison (test- and val-selected)."
    return run


# ---------------------------------------------------------------------------
# UEA average AG init 3D study
# ---------------------------------------------------------------------------

_UEA_AVG_DATASETS = ["scp1", "scp2", "heartbeat", "motor", "ethanol"]


def study_uea_avg_ag_3d(sweeps_root: Path, output_dir: Path) -> None:
    """3D AG init surface averaged (mean accuracy) over scp1, scp2, heartbeat, motor, ethanol."""
    row_col = _bk("A_max")
    col_col = _bk("G_max")
    spec = ACCURACY_SPEC

    print(f"\n{'=' * 60}\nSTUDY: UEA Average AG Init (3D)  [{spec.label}]\n{'=' * 60}")

    grids: list[tuple[str, pd.DataFrame]] = []
    for key in _UEA_AVG_DATASETS:
        cfg = DATASETS[key]
        df = _load_sweep(sweeps_root, cfg.ag_sweep)
        if df.empty:
            print(f"  Skipping {key}: empty sweep")
            continue
        if row_col not in df.columns or col_col not in df.columns:
            print(f"  Skipping {key}: missing A_max/G_max columns")
            continue
        nuisance = _compute_nuisance(df, {row_col, col_col})
        mean_pivot, _ = _pivot_metric_grid(df, row_col, col_col, nuisance, spec, "mean")
        grids.append((key, mean_pivot))
        print(
            f"  Loaded {key}: grid {mean_pivot.shape}, "
            f"mean range {mean_pivot.min().min():.4f}–{mean_pivot.max().max():.4f}"
        )

    if not grids:
        print("  No datasets loaded, skipping")
        return

    common_index = sorted(set.intersection(*[set(g.index) for _, g in grids]))
    common_cols = sorted(set.intersection(*[set(g.columns) for _, g in grids]))
    if not common_index or not common_cols:
        print("  No common (A_max, G_max) cells across datasets, skipping")
        return

    stacked = np.stack(
        [g.reindex(index=common_index, columns=common_cols).values.astype(float) for _, g in grids]
    )
    avg_grid = np.nanmean(stacked, axis=0)
    std_grid = np.nanstd(stacked, axis=0, ddof=1)

    row_vals = np.array(common_index, dtype=float)
    col_vals = np.array(common_cols, dtype=float)
    dataset_names = ", ".join(k for k, _ in grids)
    print(f"  Averaging over: {dataset_names}")
    print(f"  Avg range: {np.nanmin(avg_grid):.4f} – {np.nanmax(avg_grid):.4f}")

    mean_df = pd.DataFrame(avg_grid, index=common_index, columns=common_cols)
    std_df = pd.DataFrame(std_grid, index=common_index, columns=common_cols)
    fig = plot_metric_heatmap(
        mean_df,
        std_df,
        spec,
        title=f"UEA Average — AG Initialization\n({dataset_names})",
        xlabel="G_max",
        ylabel="A_max",
    )
    out_path = output_dir / "study_uea_avg_ag_mean.png"
    fig.savefig(out_path)
    print(f"  Saved: {out_path}")
    plt.close(fig)

    fig = plot_ag_surface_3d(
        row_vals,
        col_vals,
        avg_grid,
        std_grid,
        spec,
        title=f"UEA Average — AG Initialization\n({dataset_names})",
        row_label="A_max",
        col_label="G_max",
        log_scale=True,
    )
    out_path = output_dir / "study_uea_avg_ag_3d.png"
    fig.savefig(out_path)
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Registry & CLI
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, list[Callable[[Path, Path], None]]]:
    """Assemble the {study_name: [study_fns]} registry from the config tables."""
    registry: dict[str, list[Callable[[Path, Path], None]]] = {}

    # <dataset>_init for every DATASETS entry; CIFAR also gets the RT init grid.
    for key in DATASETS:
        fns = [make_init_study(key)]
        if key == "cifar":
            fns.insert(0, study_cifar_init_rt)
        registry[f"{key}_init"] = fns

    registry["cifar_disc"] = [study_cifar_discretization]

    for name, studies in MULTIHEAD_STUDIES.items():
        registry[name] = [make_multihead_study(s) for s in studies]

    for key in SCENARIO_DATASETS:
        registry[f"{key}_scenarios"] = [make_scenario_study(key)]

    registry["uea_avg_ag"] = [study_uea_avg_ag_3d]

    return registry


STUDIES: dict[str, list[Callable[[Path, Path], None]]] = _build_registry()


def main() -> None:
    """Run selected hyperparameter studies and save figures."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter study visualizations for discretax sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sweeps_root", type=Path, help="Root directory containing sweep subdirectories"
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

    study_names = list(STUDIES.keys()) if args.studies is None else args.studies
    for name in study_names:
        if name not in STUDIES:
            print(f"Error: unknown study '{name}'. Options: {', '.join(STUDIES.keys())}")
            return

    for name in study_names:
        for fn in STUDIES[name]:
            fn(args.sweeps_root, args.output)

    print(f"\n{'=' * 60}\nAll figures saved to {args.output}/\n{'=' * 60}")


if __name__ == "__main__":
    main()
