"""Experiment logging backends and naming helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from discretax.training.config import ExperimentConfig, flatten_config


def build_project_name(experiment_config: ExperimentConfig) -> str:
    """Resolve the W&B project name for an experiment."""
    if experiment_config.wandb.project:
        return experiment_config.wandb.project
    return f"discretax-{experiment_config.dataset.kind}"


def build_run_name(experiment_config: ExperimentConfig) -> str:
    """Build a stable run name from the experiment config."""
    if experiment_config.wandb.run_name:
        return experiment_config.wandb.run_name
    return (
        f"{experiment_config.name}-"
        f"{experiment_config.dataset.resolved_name}-"
        f"{experiment_config.model.name}-"
        f"seed{experiment_config.seed}"
    )


def build_tags(experiment_config: ExperimentConfig) -> list[str]:
    """Build the tag set for an experiment."""
    tags = [
        *experiment_config.tags,
        *experiment_config.wandb.tags,
        experiment_config.dataset.kind,
        experiment_config.model.name,
    ]
    return list(dict.fromkeys(tags))


class Tracker(Protocol):
    """Protocol for experiment trackers."""

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        """Log a step-level metrics payload."""

    def summary(self, values: dict[str, Any]) -> None:
        """Update a run summary payload."""

    def finish(self) -> None:
        """Finish the tracker cleanly."""


@dataclass(slots=True)
class NullTracker:
    """No-op tracker used when external logging is disabled."""

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        """Ignore logged metrics."""

    def summary(self, values: dict[str, Any]) -> None:
        """Ignore summary updates."""

    def finish(self) -> None:
        """No-op finish."""


@dataclass(slots=True)
class WandbTracker:
    """Thin wrapper around a W&B run object."""

    run: Any

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        """Log metrics to W&B."""
        self.run.log(metrics, step=step)

    def summary(self, values: dict[str, Any]) -> None:
        """Update W&B run summary values."""
        for key, value in values.items():
            self.run.summary[key] = value

    def finish(self) -> None:
        """Finish the wrapped W&B run."""
        self.run.finish()


def create_tracker(experiment_config: ExperimentConfig, output_dir: Path) -> Tracker:
    """Create an experiment tracker from config."""
    if not experiment_config.wandb.enabled:
        return NullTracker()

    import time

    import wandb

    init_kwargs = dict(
        project=build_project_name(experiment_config),
        entity=experiment_config.wandb.entity,
        group=experiment_config.wandb.group,
        job_type=experiment_config.wandb.job_type,
        name=build_run_name(experiment_config),
        dir=str(output_dir),
        tags=build_tags(experiment_config),
        config=flatten_config(experiment_config),
        settings=wandb.Settings(init_timeout=120),
    )

    # If mode is explicitly set (e.g. "offline" or "disabled"), respect it directly.
    if experiment_config.wandb.mode:
        run = wandb.init(mode=experiment_config.wandb.mode, **init_kwargs)
        return WandbTracker(run=run)

    # Otherwise attempt online with retries, then fall back to offline.
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            run = wandb.init(mode="online", **init_kwargs)
            return WandbTracker(run=run)
        except wandb.errors.CommError as e:
            if attempt < max_attempts - 1:
                wait = 30 * (attempt + 1)
                print(
                    f"[wandb] Init failed (attempt {attempt + 1}/{max_attempts}), "
                    f"retrying in {wait}s: {e}"
                )
                time.sleep(wait)
            else:
                print(
                    f"[wandb] Online init failed after {max_attempts} attempts, "
                    "falling back to offline mode. Sync later with: "
                    f"wandb sync {output_dir}/wandb/offline-run-*"
                )
                run = wandb.init(mode="offline", **init_kwargs)
                return WandbTracker(run=run)

    raise RuntimeError("unreachable")  # satisfy type checkers

