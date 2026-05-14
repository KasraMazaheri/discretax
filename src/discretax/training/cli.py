"""CLI entrypoint for configured Discretax experiment runs."""

from __future__ import annotations

import argparse

import yaml

from discretax.training.config import load_experiment_config
from discretax.training.trainer import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""
    parser = argparse.ArgumentParser(description="Run a configured Discretax experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dotted assignments like trainer.max_steps=10.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully resolved config and exit.",
    )
    parser.add_argument(
        "--resume-from",
        help="Resume from a run directory or checkpoint directory.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a checkpoint from --resume-from and run evaluation without training.",
    )
    return parser


def main() -> None:
    """Execute the training CLI."""
    args = build_parser().parse_args()
    experiment_config = load_experiment_config(args.config, overrides=args.overrides)

    if args.print_config:
        print(yaml.safe_dump(experiment_config.to_dict(), sort_keys=False))
        return

    result = run_experiment(
        experiment_config,
        resume_from=args.resume_from,
        eval_only=args.eval_only,
    )
    print(
        yaml.safe_dump(
            {
                "output_dir": str(result.output_dir),
                "best_val_metric": result.best_val_metric,
                "final_step": result.final_step,
                "test_loss": result.test_loss,
                "test_metric": result.test_metric,
                "mode": result.mode,
            },
            sort_keys=False,
        )
    )
