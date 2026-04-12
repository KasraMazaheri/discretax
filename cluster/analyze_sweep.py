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


def load_job_results(sweep_dir: Path) -> list[dict[str, Any]]:
    """Load results from all jobs in a sweep's outputs directory."""
    outputs_dir = sweep_dir / "outputs"
    if not outputs_dir.exists():
        print(f"Warning: outputs directory not found at {outputs_dir}")
        return []

    results = []
    job_dirs = sorted(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("job-")),
        key=lambda p: int(p.name.removeprefix("job-")),
    )

    for job_dir in job_dirs:
        job_id = int(job_dir.name.removeprefix("job-"))
        run_dir = _find_run_dir(job_dir)
        if run_dir is None:
            print(f"Warning: no run directory found in {job_dir}, skipping")
            continue

        # Load run metadata for status
        metadata_file = run_dir / "run_metadata.json"
        if not metadata_file.exists():
            print(f"Warning: no run_metadata.json in {run_dir}, skipping")
            continue
        with metadata_file.open() as f:
            run_metadata = json.load(f)

        status = run_metadata.get("status", "unknown")
        if status != "completed":
            print(f"  job-{job_id}: status={status}, skipping")
            continue

        # Load final summary
        summary_file = run_dir / "summary.json"
        if not summary_file.exists():
            print(f"Warning: no summary.json in {run_dir}, skipping")
            continue
        with summary_file.open() as f:
            summary = json.load(f)

        # Load resolved config
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
            # test_metric is the canonical name; test_accuracy kept for backward compat
            "test_accuracy": summary.get("test_metric", summary.get("test_accuracy")),
            "test_loss": summary.get("test_loss"),
            "best_metric": summary.get("best_metric"),
            "final_step": summary.get("final_step"),
            "duration_seconds": summary.get("duration_seconds"),
            "parameter_count": summary.get("parameter_count"),
            "median_train_examples_per_second": summary.get("median_train_examples_per_second"),
            "median_eval_examples_per_second": summary.get("median_eval_examples_per_second"),
        }
        result.update(flat_config)
        results.append(result)

    return results


def _config_columns(df: pd.DataFrame) -> list[str]:
    """Identify flattened config columns that represent ML hyperparameters.

    Excludes result columns, per-job infrastructure (paths, wandb metadata),
    and seed columns so that runs differing only by seed collapse into one group.
    """
    result_cols = {
        "job_id",
        "job_dir",
        "test_accuracy",
        "test_loss",
        "best_metric",
        "final_step",
        "duration_seconds",
        "parameter_count",
        "median_train_examples_per_second",
        "median_eval_examples_per_second",
        "mean_accuracy",
        "std_accuracy",
        "min_accuracy",
        "max_accuracy",
        "num_runs",
        "mean_parameter_count",
        "mean_train_eps",
        "mean_eval_eps",
    }
    # Prefixes that are infrastructure/metadata, not hyperparameters
    excluded_prefixes = ("paths_", "wandb_", "name")
    # Suffixes / exact names that identify seed columns
    excluded_seed = ("trainer_seed",)

    return [
        col
        for col in df.columns
        if col not in result_cols
        and not any(col.startswith(p) for p in excluded_prefixes)
        and col not in excluded_seed
        and not col.endswith("_seed")
    ]


def _varying_columns(df: pd.DataFrame, config_cols: list[str]) -> list[str]:
    """Return only config columns that differ across jobs."""
    return [col for col in config_cols if col in df.columns and df[col].nunique() > 1]


def group_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Group results by configuration (excluding seed) and compute statistics."""
    config_cols = _config_columns(df)
    if not config_cols:
        print("Warning: no config columns found — returning ungrouped data")
        return df

    metric_cols = [c for c in ("test_accuracy",) if c in df.columns]
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

    renames = {
        "test_accuracy_mean": "mean_accuracy",
        "test_accuracy_std": "std_accuracy",
        "test_accuracy_min": "min_accuracy",
        "test_accuracy_max": "max_accuracy",
        "test_accuracy_count": "num_runs",
        "parameter_count_mean": "mean_parameter_count",
        "median_train_examples_per_second_mean": "mean_train_eps",
        "median_eval_examples_per_second_mean": "mean_eval_eps",
    }
    grouped = grouped.rename(columns={k: v for k, v in renames.items() if k in grouped.columns})
    return grouped


def print_summary(df: pd.DataFrame, grouped_df: pd.DataFrame) -> None:
    """Print a high-level summary of sweep results."""
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    print(f"\nCompleted jobs: {len(df)}  |  Unique configurations: {len(grouped_df)}")

    if "mean_accuracy" in grouped_df.columns:
        g = grouped_df["mean_accuracy"]
        print(
            f"Test accuracy (mean per config):  mean={g.mean():.4f}  std={g.std():.4f}  "
            f"min={g.min():.4f}  max={g.max():.4f}  median={g.median():.4f}"
        )

    if "duration_seconds" in df.columns:
        dur = df["duration_seconds"].dropna()
        if not dur.empty:
            print(f"Run duration:  mean {dur.mean():.0f}s  |  max {dur.max():.0f}s")


def print_top_configs(grouped_df: pd.DataFrame, top_k: int = 10) -> None:
    """Print the top-k configurations ranked by mean accuracy."""
    print("\n" + "=" * 80)
    print(f"TOP {top_k} CONFIGURATIONS")
    print("=" * 80)

    if "mean_accuracy" not in grouped_df.columns:
        print("No accuracy data available")
        return

    df_sorted = grouped_df.sort_values("mean_accuracy", ascending=False)
    varying = _varying_columns(grouped_df, _config_columns(grouped_df))
    print(f"\nVarying parameters: {', '.join(varying) or '(none)'}\n")

    for rank, (_, row) in enumerate(df_sorted.head(top_k).iterrows(), 1):
        mean_acc = row["mean_accuracy"]
        std_acc = row.get("std_accuracy", float("nan"))
        num_runs = int(row.get("num_runs", 1))
        std_str = f" ± {std_acc:.4f}" if pd.notna(std_acc) else ""
        hp_str = "  ".join(f"{col}={row[col]}" for col in varying if col in row)
        print(f"{rank:3d}. {mean_acc:.4f}{std_str} (n={num_runs})  {hp_str}")


def analyze_parameter_impact(grouped_df: pd.DataFrame) -> None:
    """Print per-parameter accuracy breakdowns for all varying hyperparameters."""
    print("\n" + "=" * 80)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 80)

    if "mean_accuracy" not in grouped_df.columns:
        print("No accuracy data available")
        return

    config_cols = _config_columns(grouped_df)
    varying = _varying_columns(grouped_df, config_cols)
    if not varying:
        print("No varying hyperparameters found")
        return

    print(f"\nAnalyzing {len(varying)} varying parameter(s):\n")
    for param in varying:
        print(f"{param}:")
        grp = grouped_df.groupby(param)["mean_accuracy"].agg(["mean", "std", "count"])
        grp = grp.sort_values("mean", ascending=False)
        print(grp.to_string())
        print()


def print_efficiency_comparison(grouped_df: pd.DataFrame) -> None:
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

    varying = _varying_columns(grouped_df, _config_columns(grouped_df))
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
        if "mean_accuracy" in row and pd.notna(row["mean_accuracy"]):
            parts.append(f"acc={row['mean_accuracy']:.4f}")
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
        "--no-analysis",
        action="store_true",
        help="Skip per-parameter impact analysis",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: sweep directory {sweep_dir} not found")
        return

    print(f"Loading results from {sweep_dir} ...")
    results = load_job_results(sweep_dir)
    if not results:
        print("No completed results found.")
        return

    df = pd.DataFrame(results)
    grouped_df = group_by_config(df)

    print_summary(df, grouped_df)
    print_top_configs(grouped_df, top_k=args.top_k)
    print_efficiency_comparison(grouped_df)

    if not args.no_analysis:
        analyze_parameter_impact(grouped_df)

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
