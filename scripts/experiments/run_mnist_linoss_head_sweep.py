"""Run and summarize a focused MNIST multi-head LinOSS sweep."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from discretax.training.config import load_experiment_config
from discretax.training.trainer import run_experiment


@dataclass(slots=True)
class SweepVariant:
    """A single experiment variant in the LinOSS sweep."""

    label: str
    hidden_dim: int
    state_dim: int
    num_heads: int
    use_head_output_projection: bool = False
    use_head_gating: bool = False


@dataclass(slots=True)
class VariantSummary:
    """Condensed result summary for a completed sweep run."""

    label: str
    hidden_dim: int
    state_dim: int
    num_heads: int
    use_head_output_projection: bool
    use_head_gating: bool
    output_dir: str
    best_metric: float
    final_step: int
    test_loss: float
    test_metric: float
    val_loss_at_50: float | None
    val_loss_at_100: float | None
    val_loss_at_200: float | None
    val_loss_at_300: float | None
    val_accuracy_at_50: float | None
    val_accuracy_at_100: float | None
    val_accuracy_at_200: float | None
    val_accuracy_at_300: float | None


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/mnist_linoss_head_sweep.yaml",
        help="Base experiment config to use for the sweep.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the sweep summary JSON output.",
    )
    parser.add_argument(
        "--max-head-runs",
        type=int,
        default=None,
        help="Optional limit for the number of baseline head-count runs.",
    )
    parser.add_argument(
        "--max-enhancement-runs",
        type=int,
        default=None,
        help="Optional limit for the number of projection/gating enhancement runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned variants without executing them.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Total hidden/state dimensions to sweep.",
    )
    return parser.parse_args()


def _history_at_or_before(history: list[dict[str, float]], step: int, key: str) -> float | None:
    """Return the last recorded metric at or before a target step."""
    candidates = [entry for entry in history if entry.get("step", -1) <= step and key in entry]
    if not candidates:
        return None
    return float(candidates[-1][key])


def _load_history(output_dir: Path) -> list[dict[str, float]]:
    """Load JSONL history records for a completed run."""
    history_path = output_dir / "history.jsonl"
    return [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]


def _variant_overrides(variant: SweepVariant) -> list[str]:
    """Build config overrides for a sweep variant."""
    tags = [
        "mnist",
        "linoss",
        "multihead",
        f"hidden-{variant.hidden_dim}",
        f"state-{variant.state_dim}",
        f"heads-{variant.num_heads}",
    ]
    if variant.use_head_output_projection:
        tags.append("projection")
    if variant.use_head_gating:
        tags.append("gating")

    return [
        f"name={variant.label}",
        f"model.hidden_dim={variant.hidden_dim}",
        f"model.backbone.kwargs.state_dim={variant.state_dim}",
        f"model.backbone.kwargs.num_heads={variant.num_heads}",
        f"model.backbone.kwargs.use_head_output_projection={str(variant.use_head_output_projection).lower()}",
        f"model.backbone.kwargs.use_head_gating={str(variant.use_head_gating).lower()}",
        f"wandb.run_name={variant.label}",
        f"wandb.tags={json.dumps(tags)}",
    ]


def _head_variants(total_dims: list[int]) -> list[SweepVariant]:
    """Return the baseline head-count sweep variants."""
    variants: list[SweepVariant] = []
    for total_dim in total_dims:
        for num_heads in (1, 2, 4, 8):
            if total_dim % num_heads != 0:
                continue
            variants.append(
                SweepVariant(
                    label=f"mnist-linoss-h{total_dim}-s{total_dim}-heads{num_heads}",
                    hidden_dim=total_dim,
                    state_dim=total_dim,
                    num_heads=num_heads,
                )
            )
    return variants


def _enhancement_variants(best_head_variant: SweepVariant) -> list[SweepVariant]:
    """Return projection and gating ablations for the strongest head-count variant."""
    return [
        SweepVariant(
            label=f"{best_head_variant.label}-projection",
            hidden_dim=best_head_variant.hidden_dim,
            state_dim=best_head_variant.state_dim,
            num_heads=best_head_variant.num_heads,
            use_head_output_projection=True,
        ),
        SweepVariant(
            label=f"{best_head_variant.label}-gating",
            hidden_dim=best_head_variant.hidden_dim,
            state_dim=best_head_variant.state_dim,
            num_heads=best_head_variant.num_heads,
            use_head_gating=True,
        ),
        SweepVariant(
            label=f"{best_head_variant.label}-projection-gating",
            hidden_dim=best_head_variant.hidden_dim,
            state_dim=best_head_variant.state_dim,
            num_heads=best_head_variant.num_heads,
            use_head_output_projection=True,
            use_head_gating=True,
        ),
    ]


def _run_variant(config_path: str, variant: SweepVariant) -> VariantSummary:
    """Execute a single sweep variant and summarize the result."""
    experiment_config = load_experiment_config(config_path, overrides=_variant_overrides(variant))
    result = run_experiment(experiment_config)
    history = _load_history(result.output_dir)
    return VariantSummary(
        label=variant.label,
        hidden_dim=variant.hidden_dim,
        state_dim=variant.state_dim,
        num_heads=variant.num_heads,
        use_head_output_projection=variant.use_head_output_projection,
        use_head_gating=variant.use_head_gating,
        output_dir=str(result.output_dir),
        best_metric=result.best_metric,
        final_step=result.final_step,
        test_loss=result.test_loss,
        test_metric=result.test_metric,
        val_loss_at_50=_history_at_or_before(history, 50, "validation_loss"),
        val_loss_at_100=_history_at_or_before(history, 100, "validation_loss"),
        val_loss_at_200=_history_at_or_before(history, 200, "validation_loss"),
        val_loss_at_300=_history_at_or_before(history, 300, "validation_loss"),
        val_accuracy_at_50=_history_at_or_before(history, 50, "validation_accuracy"),
        val_accuracy_at_100=_history_at_or_before(history, 100, "validation_accuracy"),
        val_accuracy_at_200=_history_at_or_before(history, 200, "validation_accuracy"),
        val_accuracy_at_300=_history_at_or_before(history, 300, "validation_accuracy"),
    )


def _write_summary(output_path: Path, summaries: list[VariantSummary]) -> None:
    """Write the sweep summary JSON."""
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "variants": [asdict(summary) for summary in summaries],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _default_output_path() -> Path:
    """Build the default sweep summary path."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "sweeps" / f"{timestamp}-mnist-linoss-head-sweep.json"


def main() -> None:
    """Run the configured head-count sweep and enhancement follow-up runs."""
    args = _parse_args()
    head_variants = _head_variants(args.dims)
    if args.max_head_runs is not None:
        head_variants = head_variants[: args.max_head_runs]

    if args.dry_run:
        for variant in head_variants:
            print(variant)
        return

    summaries = [_run_variant(args.config, variant) for variant in head_variants]
    multi_head_summaries = [summary for summary in summaries if summary.num_heads > 1]
    enhancement_variants: list[SweepVariant] = []
    if multi_head_summaries:
        best_multi_head_summary = min(
            multi_head_summaries, key=lambda summary: summary.best_metric
        )
        best_multi_head_variant = SweepVariant(
            label=best_multi_head_summary.label,
            hidden_dim=best_multi_head_summary.hidden_dim,
            state_dim=best_multi_head_summary.state_dim,
            num_heads=best_multi_head_summary.num_heads,
        )
        enhancement_variants = _enhancement_variants(best_multi_head_variant)
        if args.max_enhancement_runs is not None:
            enhancement_variants = enhancement_variants[: args.max_enhancement_runs]
        summaries.extend(_run_variant(args.config, variant) for variant in enhancement_variants)

    output_path = Path(args.output) if args.output else _default_output_path()
    _write_summary(output_path, summaries)

    ranked = sorted(summaries, key=lambda summary: summary.best_metric)
    print(json.dumps([asdict(summary) for summary in ranked], indent=2, sort_keys=True))
    print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()
