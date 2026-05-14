r"""Generate HTCondor job configurations for discretax hyperparameter sweeps.

Each job runs:
    python -m discretax.training --config BASE_CONFIG [--set key=value ...]

All job outputs are directed to sweeps/SWEEP_NAME/outputs/ so they are
co-located with the sweep's config files and logs.

Example:
-------
    python cluster/generate_sweep.py \\
        --config configs/experiments/cifar10_linoss_example.yaml \\
        --sweep "optimizer.learning_rate=0.0001,0.001" \\
        --sweep "model.backbone.kwargs.state_dim=64,128,256" \\
        --sweep "seed=0,1,2" \\
        --name cifar10-lr-statedim

    condor_submit cluster/sweep.sub.generated
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HTCondor jobs for a discretax hyperparameter sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="BASE_CONFIG",
        help="Base experiment YAML config passed to every job (e.g. configs/experiments/cifar10_linoss_example.yaml).",  # noqa: E501
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="KEY=V1,V2,...",
        help="Sweep parameter as 'key=val1,val2,...'. Repeated flags produce a cartesian product.",
    )
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Fixed override applied to every job (e.g. 'trainer.max_steps=50000').",
    )
    parser.add_argument(
        "--name",
        dest="sweep_name",
        default=None,
        help="Sweep name used as the directory and HTCondor SWEEP_NAME. Defaults to current datetime.",  # noqa: E501
    )
    parser.add_argument(
        "--output-dir",
        default="sweeps",
        help="Root directory for sweep artifacts (default: sweeps/).",
    )
    parser.add_argument(
        "--submit-file",
        default="cluster/sweep.sub",
        help="HTCondor submit file template (default: cluster/sweep.sub).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing any files.",
    )
    return parser.parse_args()


def _parse_sweep_params(sweep_args: list[str]) -> dict[str, list[str]]:
    """Parse 'key=v1,v2,v3' sweep strings into a dict of key → [values]."""
    params: dict[str, list[str]] = {}
    for arg in sweep_args:
        if "=" not in arg:
            raise ValueError(f"Invalid sweep format {arg!r}: expected 'key=val1,val2,...'")
        key, values_str = arg.split("=", 1)
        params[key] = [v.strip() for v in values_str.split(",")]
    return params


def _combinations(params: dict[str, list[str]]) -> list[dict[str, str]]:
    """Return the cartesian product of all sweep parameter values."""
    if not params:
        return [{}]
    keys = list(params)
    return [dict(zip(keys, combo)) for combo in itertools.product(*params.values())]


def _format_set_flags(overrides: dict[str, Any]) -> list[str]:
    """Convert an override dict into a list of '--set key=value' strings."""
    return [f"--set {k}={v}" for k, v in overrides.items()]


def _job_args(
    base_config: str,
    sweep_combo: dict[str, str],
    fixed: list[str],
    output_root: str,
    sweep_name: str,
    job_index: int,
) -> str:
    """Build the full argument string for one job."""
    parts = [f"--config {base_config}"]
    parts += _format_set_flags(sweep_combo)
    parts += [f"--set {f}" for f in fixed]
    parts.append(f"--set paths.output_root={output_root}/job-{job_index}")
    parts.append(f"--set wandb.group={sweep_name}")
    parts.append(f"--set wandb.tags=[job-{job_index}]")
    return " ".join(parts)


def main() -> None:
    """Entry point: generate HTCondor sweep configs from command-line arguments."""
    args = _parse_arguments()

    sweep_params = _parse_sweep_params(args.sweep)
    combos = _combinations(sweep_params)
    num_jobs = len(combos)

    sweep_name = args.sweep_name or f"{datetime.now():%Y-%m-%d-%H-%M-%S}"
    sweep_dir = Path(args.output_dir) / sweep_name
    output_root = str(sweep_dir / "outputs")

    if args.dry_run:
        print(f"[DRY RUN] Sweep: {sweep_name}  |  {num_jobs} jobs")
        print(f"[DRY RUN] Sweep dir: {sweep_dir}")
        show = combos[:5]
        for i, combo in enumerate(show):
            job_str = _job_args(args.config, combo, args.fixed, output_root, sweep_name, i)
            print(f"  Job {i}: {job_str}")
        if num_jobs > 5:
            print(f"  ... and {num_jobs - 5} more")
        return

    # Create sweep directory structure
    if sweep_dir.exists():
        print(f"Removing existing sweep directory: {sweep_dir}")
        shutil.rmtree(sweep_dir)
    (sweep_dir / "configs").mkdir(parents=True)
    (sweep_dir / "logs").mkdir()
    (sweep_dir / "outputs").mkdir()
    print(f"Created sweep directory: {sweep_dir}")

    # Write per-job config files
    for i, combo in enumerate(combos):
        config_file = sweep_dir / "configs" / f"config_{i}.txt"
        config_file.write_text(
            _job_args(args.config, combo, args.fixed, output_root, sweep_name, i)
        )
    print(f"Generated {num_jobs} job config files")

    # Fill in submit file template
    submit_template = Path(args.submit_file)
    if not submit_template.exists():
        print(f"Error: submit file {submit_template} not found")
        return

    submit_content = submit_template.read_text()
    job_ids = " ".join(str(i) for i in range(num_jobs))
    submit_content = submit_content.replace("$(SWEEP_NAME)", sweep_name)
    submit_content = submit_content.replace("$(JOB_IDS)", job_ids)

    generated_submit = submit_template.with_suffix(".sub.generated")
    generated_submit.write_text(submit_content)
    print(f"Generated submit file: {generated_submit}")

    print(f"\nTo submit {num_jobs} jobs:")
    print(f"  condor_submit {generated_submit}")


if __name__ == "__main__":
    main()
