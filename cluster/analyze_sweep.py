"""Analyze results from a discretax HTCondor hyperparameter sweep.

Loads summary.json and config.yaml from each job's output directory,
aggregates metrics, and identifies best configurations.

Directory layout expected:
    sweeps/SWEEP_NAME/
        outputs/
            job-0/{timestamp-run}/
                summary.json
                run_metadata.json
                config.yaml
            job-1/...
        configs/
            config_0.txt  (raw CLI args, for reference)

Examples:
--------
    python cluster/analyze_sweep.py sweeps/my-sweep
    python cluster/analyze_sweep.py sweeps/my-sweep --top-k 20
    python cluster/analyze_sweep.py sweeps/my-sweep --output results.csv
    python cluster/analyze_sweep.py sweeps/my-sweep --no-analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "_") -> dict[str, Any]:
    """Recursively flatten a nested dict using sep-joined keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _find_run_dir(job_output_dir: Path) -> Path | None:
    """Return the single run directory inside a job output dir, or None."""
    candidates = [p for p in job_output_dir.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Multiple runs in one job dir — take the most recent by mtime
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


_METRIC_CANDIDATES: tuple[tuple[str, bool], ...] = (
    # (summary key, higher_is_better)
    ("test_metric", True),
    ("test_accuracy", True),
    ("test_loss", False),
    ("test_mse", False),
    ("test_mae", False),
)


def _detect_metric(summaries: list[dict[str, Any]]) -> tuple[str, bool]:
    """Pick the first candidate metric that is present and finite in any summary."""
    for key, higher_is_better in _METRIC_CANDIDATES:
        for s in summaries:
            v = s.get(key)
            if v is not None and pd.notna(v) and v not in (float("inf"), float("-inf")):
                return key, higher_is_better
    # Fallback: first candidate key that appears anywhere
    for key, higher_is_better in _METRIC_CANDIDATES:
        if any(key in s for s in summaries):
            return key, higher_is_better
    return "test_metric", True


def _load_single_job(
    job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load the result and summary for a single job directory, or None if skipped."""
    job_id = int(job_dir.name.removeprefix("job-"))
    run_dir = _find_run_dir(job_dir)
    if run_dir is None:
        print(f"Warning: no run directory found in {job_dir}, skipping")
        return None

    summary_file = run_dir / "summary.json"
    metadata_file = run_dir / "run_metadata.json"

    summary: dict[str, Any] = {}
    if summary_file.exists():
        with summary_file.open() as f:
            summary = json.load(f)

    # Trust summary.json over run_metadata.json: the trainer writes summary
    # AFTER finalizing metadata, so any run with a finite test metric is done
    # even if run_metadata.json wasn't updated (crash on shutdown, NFS, etc.).
    def _is_finite(v: Any) -> bool:
        return v is not None and pd.notna(v) and v not in (float("inf"), float("-inf"))

    has_test_metric = any(
        _is_finite(summary.get(k))
        for k in ("test_loss", "test_mse", "test_mae", "test_accuracy", "test_metric")
    )

    if metadata_file.exists():
        with metadata_file.open() as f:
            run_metadata = json.load(f)
    else:
        run_metadata = {}

    status = run_metadata.get("status", "unknown")
    if status != "completed" and not has_test_metric:
        print(f"  job-{job_id}: status={status}, skipping")
        return None
    if status != "completed" and has_test_metric:
        print(f"  job-{job_id}: status={status} but summary.json has test metrics — accepting")
    if not summary_file.exists():
        print(f"Warning: no summary.json in {run_dir}, skipping")
        return None

    config_file = run_dir / "config.yaml"
    flat_config: dict[str, Any] = {}
    if config_file.exists():
        with config_file.open() as f:
            raw_config = yaml.safe_load(f)
        if isinstance(raw_config, dict):
            flat_config = _flatten_dict(raw_config)

    result: dict[str, Any] = {
        "job_id": job_id,
        "job_dir": str(run_dir),
        "test_metric": summary.get("test_metric", summary.get("test_accuracy")),
        "test_loss": summary.get("test_loss"),
        "test_mse": summary.get("test_mse"),
        "test_mae": summary.get("test_mae"),
        "best_val_metric": summary.get("best_val_metric", summary.get("best_metric")),
        "final_step": summary.get("final_step"),
        "duration_seconds": summary.get("duration_seconds"),
        "parameter_count": summary.get("parameter_count"),
        "median_train_examples_per_second": summary.get("median_train_examples_per_second"),
        "median_eval_examples_per_second": summary.get("median_eval_examples_per_second"),
    }
    result.update(flat_config)
    return result, summary


def load_job_results(
    sweep_dir: Path, rank_by: str = "test"
) -> tuple[list[dict[str, Any]], str, bool, str]:
    """Load results from all jobs in a sweep's outputs directory.

    Returns (results, test_metric_key, higher_is_better, rank_metric_key).
    `score` always reflects the test metric. `rank_score` reflects whichever metric
    is selected for ranking (test or validation).
    """
    outputs_dir = sweep_dir / "outputs"
    if not outputs_dir.exists():
        print(f"Warning: outputs directory not found at {outputs_dir}")
        return [], "test_metric", True, "test_metric"

    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    job_dirs = sorted(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("job-")),
        key=lambda p: int(p.name.removeprefix("job-")),
    )

    for job_dir in job_dirs:
        loaded = _load_single_job(job_dir)
        if loaded is None:
            continue
        result, summary = loaded
        results.append(result)
        summaries.append(summary)

    metric_key, higher_is_better = _detect_metric(summaries)

    if rank_by == "val":
        # Validation metric loss-style: lower is better. (best_val_metric is the
        # monitored value at the best checkpoint — orientation matches the test loss.)
        rank_metric_key = "best_val_metric"
    else:
        rank_metric_key = metric_key

    for r, s in zip(results, summaries):
        v = s.get(metric_key)
        if v in (float("inf"), float("-inf")):
            v = float("nan")
        r["score"] = v

        rv = s.get(rank_metric_key)
        if rv in (float("inf"), float("-inf")):
            rv = float("nan")
        r["rank_score"] = rv
    return results, metric_key, higher_is_better, rank_metric_key


def _config_columns(df: pd.DataFrame, aggregate_seeds: bool = True) -> list[str]:
    """Identify flattened config columns that represent ML hyperparameters.

    Excludes result columns and per-job infrastructure (paths, wandb metadata).
    When `aggregate_seeds=True`, also excludes seed columns so that runs differing
    only by seed collapse into one group; when False, seed columns are kept so
    every run is treated as unique.
    """
    result_cols = {
        "job_id",
        "job_dir",
        "test_metric",
        "test_accuracy",
        "test_loss",
        "test_mse",
        "test_mae",
        "score",
        "rank_score",
        "best_val_metric",
        "final_step",
        "duration_seconds",
        "parameter_count",
        "median_train_examples_per_second",
        "median_eval_examples_per_second",
        "mean_score",
        "std_score",
        "min_score",
        "max_score",
        "mean_rank_score",
        "std_rank_score",
        "min_rank_score",
        "max_rank_score",
        "rank_num_runs",
        "num_runs",
        "num_nan",
        "mean_parameter_count",
        "mean_train_eps",
        "mean_eval_eps",
        "job_ids",
    }
    # Prefixes that are infrastructure/metadata, not hyperparameters
    excluded_prefixes = ("paths_", "wandb_", "name")
    # Seed columns are nuisance variables — exclude from grouping unless requested
    excluded_seed = {"seed", "trainer_seed", "dataset_seed"} if aggregate_seeds else set()

    return [
        col
        for col in df.columns
        if col not in result_cols
        and not any(col.startswith(p) for p in excluded_prefixes)
        and col not in excluded_seed
        and not (aggregate_seeds and col.endswith("_seed"))
    ]


def _varying_columns(df: pd.DataFrame, config_cols: list[str]) -> list[str]:
    """Return only config columns that differ across jobs."""
    return [col for col in config_cols if col in df.columns and df[col].nunique(dropna=False) > 1]


def group_by_config(df: pd.DataFrame, aggregate_seeds: bool = True) -> pd.DataFrame:
    """Group results by configuration and compute statistics.

    With `aggregate_seeds=True` (default), runs differing only by seed collapse
    into one row. With `aggregate_seeds=False`, the seed columns participate in
    the group key so each run becomes its own group.
    """
    config_cols = _config_columns(df, aggregate_seeds=aggregate_seeds)
    if not config_cols:
        print("Warning: no config columns found — returning ungrouped data")
        return df

    # When the caller wants every run to be unique, include job_id in the group
    # key so that re-runs sharing the same seed don't collapse together.
    if not aggregate_seeds and "job_id" in df.columns and "job_id" not in config_cols:
        config_cols = ["job_id", *config_cols]

    metric_cols = [c for c in ("score", "rank_score") if c in df.columns]
    agg: dict[str, list[str]] = {
        col: ["mean", "std", "min", "max", "count"] for col in metric_cols
    }
    for col in (
        "parameter_count",
        "median_train_examples_per_second",
        "median_eval_examples_per_second",
    ):
        if col in df.columns:
            agg[col] = ["mean"]

    present_config_cols = [c for c in config_cols if c in df.columns]
    if not present_config_cols:
        return df

    # Stringify any columns containing unhashable types (e.g. wandb_tags is a list)
    df = df.copy()
    for col in present_config_cols:
        if df[col].apply(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].apply(lambda v: str(v) if isinstance(v, (list, dict)) else v)

    grouped = df.groupby(present_config_cols, dropna=False).agg(agg)
    grouped.columns = ["_".join(col).strip() for col in grouped.columns]
    grouped = grouped.reset_index()

    # Attach the sorted list of job IDs belonging to each group.
    if "job_id" in df.columns:
        job_ids = (
            df.groupby(present_config_cols, dropna=False)["job_id"]
            .apply(lambda s: sorted(int(v) for v in s.dropna().tolist()))
            .reset_index(name="job_ids")
        )
        grouped = grouped.merge(job_ids, on=present_config_cols, how="left")

    renames = {
        "score_mean": "mean_score",
        "score_std": "std_score",
        "score_min": "min_score",
        "score_max": "max_score",
        "score_count": "num_runs",
        "rank_score_mean": "mean_rank_score",
        "rank_score_std": "std_rank_score",
        "rank_score_min": "min_rank_score",
        "rank_score_max": "max_rank_score",
        "rank_score_count": "rank_num_runs",
        "parameter_count_mean": "mean_parameter_count",
        "median_train_examples_per_second_mean": "mean_train_eps",
        "median_eval_examples_per_second_mean": "mean_eval_eps",
    }
    grouped = grouped.rename(columns={k: v for k, v in renames.items() if k in grouped.columns})
    return grouped


def print_failed_runs(df: pd.DataFrame, metric_key: str) -> None:
    """List jobs whose score is non-finite (NaN/Inf stripped to NaN upstream)."""
    if "score" not in df.columns:
        return
    failed = df[df["score"].isna()]
    if failed.empty:
        return
    print("\n" + "=" * 80)
    print(f"FAILED RUNS  —  non-finite {metric_key} ({len(failed)} of {len(df)})")
    print("=" * 80)
    df_hashable = df.copy()
    for col in df_hashable.columns:
        if df_hashable[col].apply(lambda v: isinstance(v, (list, dict))).any():
            df_hashable[col] = df_hashable[col].apply(
                lambda v: str(v) if isinstance(v, (list, dict)) else v
            )
    varying = _varying_columns(
        df_hashable, _config_columns(df_hashable, aggregate_seeds=False)
    )
    for _, row in failed.sort_values("job_id").iterrows():
        hp_str = "  ".join(f"{col}={row[col]}" for col in varying if col in row)
        raw_loss = row.get("test_loss")
        raw_str = f"test_loss={raw_loss}" if raw_loss is not None else ""
        print(f"  job-{int(row['job_id']):<4d} {raw_str}  {hp_str}")


def print_summary(df: pd.DataFrame, grouped_df: pd.DataFrame, metric_key: str) -> None:
    """Print a high-level summary of sweep results."""
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    n_nan = int(df["score"].isna().sum()) if "score" in df.columns else 0
    print(
        f"\nCompleted jobs: {len(df)}  |  Unique configurations: {len(grouped_df)}  "
        f"|  non-finite {metric_key}: {n_nan}"
    )

    if "mean_score" in grouped_df.columns:
        g = grouped_df["mean_score"].dropna()
        if not g.empty:
            print(
                f"{metric_key} (mean per config):  mean={g.mean():.4f}  std={g.std():.4f}  "
                f"min={g.min():.4f}  max={g.max():.4f}  median={g.median():.4f}"
            )

    if "duration_seconds" in df.columns:
        dur = df["duration_seconds"].dropna()
        if not dur.empty:
            print(f"Run duration:  mean {dur.mean():.0f}s  |  max {dur.max():.0f}s")


def print_top_configs(
    grouped_df: pd.DataFrame,
    metric_key: str,
    higher_is_better: bool,
    rank_metric_key: str,
    top_k: int = 10,
    aggregate_seeds: bool = True,
) -> None:
    """Print the top-k configurations ranked by `rank_metric_key` (test or val).

    Always reports the test metric (`mean_score`) alongside the ranking metric
    (`mean_rank_score`).
    """
    print("\n" + "=" * 80)
    direction = "higher is better" if higher_is_better else "lower is better"
    rank_label = (
        f"mean {rank_metric_key}"
        if rank_metric_key != metric_key
        else f"mean {metric_key} ({direction})"
    )
    print(f"TOP {top_k} CONFIGURATIONS  —  ranked by {rank_label}; reporting test {metric_key}")
    print("=" * 80)

    if "mean_score" not in grouped_df.columns:
        print("No metric data available")
        return

    sort_col = "mean_rank_score" if "mean_rank_score" in grouped_df.columns else "mean_score"
    df_sorted = grouped_df.sort_values(
        sort_col, ascending=not higher_is_better, na_position="last"
    )
    varying = _varying_columns(
        grouped_df, _config_columns(grouped_df, aggregate_seeds=aggregate_seeds)
    )
    print(f"\nVarying parameters: {', '.join(varying) or '(none)'}\n")

    show_rank = sort_col == "mean_rank_score" and rank_metric_key != metric_key

    for rank, (_, row) in enumerate(df_sorted.head(top_k).iterrows(), 1):
        mean_score = row["mean_score"]
        std_score = row.get("std_score", float("nan"))
        num_runs = int(row.get("num_runs", 0) or 0)
        std_str = f" ± {std_score:.4f}" if pd.notna(std_score) else ""
        mean_str = f"{mean_score:.4f}" if pd.notna(mean_score) else "   nan"
        rank_str = ""
        if show_rank:
            rv = row.get("mean_rank_score")
            rs = row.get("std_rank_score")
            rv_s = f"{rv:.4f}" if pd.notna(rv) else "  nan"
            rs_s = f" ± {rs:.4f}" if pd.notna(rs) else ""
            rank_str = f"  [val={rv_s}{rs_s}]"
        hp_str = "  ".join(f"{col}={row[col]}" for col in varying if col in row)
        ids = row.get("job_ids") if "job_ids" in row else None
        ids_str = f"  jobs={list(ids)}" if isinstance(ids, list) else ""
        print(f"{rank:3d}. test={mean_str}{std_str}{rank_str} (n={num_runs})  {hp_str}{ids_str}")


def analyze_parameter_impact(
    grouped_df: pd.DataFrame,
    metric_key: str,
    higher_is_better: bool,
    aggregate_seeds: bool = True,
) -> None:
    """Print per-parameter score breakdowns for all varying hyperparameters."""
    print("\n" + "=" * 80)
    print(f"PARAMETER IMPACT ANALYSIS  ({metric_key})")
    print("=" * 80)

    if "mean_score" not in grouped_df.columns:
        print("No metric data available")
        return

    config_cols = _config_columns(grouped_df, aggregate_seeds=aggregate_seeds)
    varying = _varying_columns(grouped_df, config_cols)
    if not varying:
        print("No varying hyperparameters found")
        return

    print(f"\nAnalyzing {len(varying)} varying parameter(s):\n")
    for param in varying:
        print(f"{param}:")
        grp = grouped_df.groupby(param, dropna=False)["mean_score"].agg(["mean", "std", "count"])
        grp = grp.sort_values("mean", ascending=not higher_is_better, na_position="last")
        print(grp.to_string())
        print()


def print_efficiency_comparison(
    grouped_df: pd.DataFrame, aggregate_seeds: bool = True
) -> None:
    """Print parameter count and throughput side-by-side, sorted by param count."""
    has_params = (
        "mean_parameter_count" in grouped_df.columns
        and grouped_df["mean_parameter_count"].notna().any()
    )
    has_train_eps = (
        "mean_train_eps" in grouped_df.columns and grouped_df["mean_train_eps"].notna().any()
    )
    has_eval_eps = (
        "mean_eval_eps" in grouped_df.columns and grouped_df["mean_eval_eps"].notna().any()
    )

    if not (has_params or has_train_eps or has_eval_eps):
        return

    print("\n" + "=" * 80)
    print("EFFICIENCY COMPARISON  (sorted by parameter count)")
    print("=" * 80)

    varying = _varying_columns(
        grouped_df, _config_columns(grouped_df, aggregate_seeds=aggregate_seeds)
    )
    df = grouped_df.copy()
    if has_params:
        df = df.sort_values("mean_parameter_count", ascending=True)

    for _, row in df.iterrows():
        hp_str = "  ".join(f"{col}={row[col]}" for col in varying if col in row)
        parts: list[str] = []
        if has_params and pd.notna(row.get("mean_parameter_count")):
            parts.append(f"params={int(row['mean_parameter_count']):,}")
        if has_train_eps and pd.notna(row.get("mean_train_eps")):
            parts.append(f"train={row['mean_train_eps']:.0f} ex/s")
        if has_eval_eps and pd.notna(row.get("mean_eval_eps")):
            parts.append(f"eval={row['mean_eval_eps']:.0f} ex/s")
        if "mean_score" in row and pd.notna(row["mean_score"]):
            parts.append(f"score={row['mean_score']:.4f}")
        print(f"  {hp_str}  |  {'  '.join(parts)}")


def main() -> None:
    """Entry point: parse arguments and run sweep analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze a discretax HTCondor sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Path to sweep directory (e.g. sweeps/my-sweep)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top configurations to display (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save aggregated (grouped) results to a CSV file",
    )
    parser.add_argument(
        "--output-all",
        type=str,
        help="Save all individual run results to a CSV file",
    )
    parser.add_argument(
        "--efficiency",
        action="store_true",
        help="Do efficiency analysis",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Do per-parameter impact analysis",
    )
    parser.add_argument(
        "--rank-by",
        choices=("test", "val"),
        default="test",
        help="Metric to rank configurations by (default: test). Test metric is always reported.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Treat each run as unique (no averaging over seeds or duplicate jobs).",
    )
    args = parser.parse_args()
    aggregate_seeds = not args.no_aggregate

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: sweep directory {sweep_dir} not found")
        return

    print(f"Loading results from {sweep_dir} ...")
    results, metric_key, higher_is_better, rank_metric_key = load_job_results(
        sweep_dir, rank_by=args.rank_by
    )
    if not results:
        print("No completed results found.")
        return
    print(
        f"Reporting test metric '{metric_key}'  |  ranking by '{rank_metric_key}' "
        f"({'higher' if higher_is_better else 'lower'} is better)  |  "
        f"aggregate={aggregate_seeds}"
    )

    df = pd.DataFrame(results)
    grouped_df = group_by_config(df, aggregate_seeds=aggregate_seeds)

    print_summary(df, grouped_df, metric_key)
    print_failed_runs(df, metric_key)
    print_top_configs(
        grouped_df,
        metric_key,
        higher_is_better,
        rank_metric_key,
        top_k=args.top_k,
        aggregate_seeds=aggregate_seeds,
    )

    if args.efficiency:
        print_efficiency_comparison(grouped_df, aggregate_seeds=aggregate_seeds)

    if args.analysis:
        analyze_parameter_impact(
            grouped_df, metric_key, higher_is_better, aggregate_seeds=aggregate_seeds
        )

    if args.output:
        out = Path(args.output)
        grouped_df.to_csv(out, index=False)
        print(f"\nAggregated results saved to {out}")

    if args.output_all:
        out = Path(args.output_all)
        df.to_csv(out, index=False)
        print(f"All run results saved to {out}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
