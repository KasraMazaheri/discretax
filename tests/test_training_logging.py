"""Tests for experiment logging helpers."""

from discretax.training import (
    CheckpointConfig,
    ComponentConfig,
    DataloaderConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    PathsConfig,
    TrainerConfig,
    WandbConfig,
)
from discretax.training.logging import build_project_name, build_run_name, build_tags


def test_wandb_naming_helpers_use_config_fields():
    """Run and project names derive from the experiment config consistently."""
    config = ExperimentConfig(
        name="mnist-smoke",
        description=None,
        tags=["smoke"],
        paths=PathsConfig(),
        loader=DataloaderConfig(),
        dataset=DatasetConfig(kind="mnist", name="mnist"),
        model=ModelConfig(
            name="linoss",
            hidden_dim=16,
            encoder=ComponentConfig(target="LinearEncoder"),
            backbone=ComponentConfig(target="LinOSS"),
            head=ComponentConfig(target="ClassificationHead"),
        ),
        optimizer=OptimizerConfig(),
        trainer=TrainerConfig(seed=7),
        checkpoint=CheckpointConfig(),
        wandb=WandbConfig(enabled=True, tags=["vision"], project=None),
    )

    assert build_project_name(config) == "discretax-mnist"
    assert build_run_name(config) == "mnist-smoke-mnist-linoss-seed7"
    assert build_tags(config) == ["smoke", "vision", "mnist", "linoss"]
