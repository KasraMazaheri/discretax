"""Filesystem helpers for experiment outputs and checkpoints."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import equinox as eqx
import yaml

from discretax.training.config import ExperimentConfig


def slugify(value: str) -> str:
    """Convert a run name into a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "run"


def create_run_directory(experiment_config: ExperimentConfig) -> Path:
    """Create a unique run directory for an experiment."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_slug = slugify(experiment_config.name)
    output_root = Path(experiment_config.paths.output_root)
    run_dir = output_root / f"{timestamp}-{run_slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_resolved_config(experiment_config: ExperimentConfig, output_dir: Path) -> Path:
    """Write the resolved experiment config to disk."""
    config_path = output_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(experiment_config.to_dict(), file, sort_keys=False)
    return config_path


def append_history(output_dir: Path, metrics: dict[str, Any]) -> None:
    """Append a metrics record to the JSONL run history."""
    history_path = output_dir / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, sort_keys=True) + "\n")


def write_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write a final run summary to disk."""
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    return summary_path


def write_run_metadata(output_dir: Path, metadata: dict[str, Any]) -> Path:
    """Write run metadata to disk."""
    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    return metadata_path


def resolve_checkpoint_directory(
    checkpoint_path: str | Path,
    *,
    default_checkpoint_name: str,
) -> Path:
    """Resolve a checkpoint path from either a run directory or a checkpoint directory."""
    checkpoint_path = Path(checkpoint_path)
    if (checkpoint_path / "model.eqx").exists():
        return checkpoint_path

    checkpoints_dir = checkpoint_path / "checkpoints"
    if checkpoints_dir.exists():
        candidate = checkpoints_dir / default_checkpoint_name
        if candidate.exists():
            return candidate
        if default_checkpoint_name == "latest":
            step_candidates = sorted(
                (
                    path
                    for path in checkpoints_dir.iterdir()
                    if path.is_dir() and path.name.startswith("step-")
                ),
                key=lambda path: int(path.name.removeprefix("step-")),
            )
            if step_candidates:
                return step_candidates[-1]
        raise FileNotFoundError(
            f"Checkpoint {default_checkpoint_name!r} does not exist under {checkpoints_dir}"
        )

    raise FileNotFoundError(f"Could not resolve a checkpoint directory from {checkpoint_path}")


def checkpoint_run_directory(checkpoint_dir: str | Path) -> Path:
    """Return the parent run directory for a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(
            f"Checkpoint directory is not nested under a checkpoints folder: {checkpoint_dir}"
        )
    return checkpoint_dir.parent.parent


def save_checkpoint(
    output_dir: Path,
    checkpoint_name: str,
    *,
    model: Any,
    state: Any,
    opt_state: Any,
    metadata: dict[str, Any],
) -> Path:
    """Save a model/state/optimizer checkpoint bundle."""
    checkpoint_dir = output_dir / "checkpoints" / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(checkpoint_dir / "model.eqx", model)
    eqx.tree_serialise_leaves(checkpoint_dir / "state.eqx", state)
    eqx.tree_serialise_leaves(checkpoint_dir / "opt_state.eqx", opt_state)
    with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model_like: Any,
    state_like: Any,
    opt_state_like: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load a checkpoint bundle into pre-built PyTree templates."""
    checkpoint_dir = Path(checkpoint_dir)
    model = eqx.tree_deserialise_leaves(checkpoint_dir / "model.eqx", model_like)
    state = eqx.tree_deserialise_leaves(checkpoint_dir / "state.eqx", state_like)
    opt_state = eqx.tree_deserialise_leaves(checkpoint_dir / "opt_state.eqx", opt_state_like)
    with (checkpoint_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, state, opt_state, metadata
