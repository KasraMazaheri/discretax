"""Tests for experiment config loading and flattening."""

from pathlib import Path

from discretax.training import flatten_config, load_experiment_config


def test_load_experiment_config_with_defaults_and_overrides(tmp_path: Path):
    """Config defaults merge recursively and dotted overrides apply last."""
    base_path = tmp_path / "base.yaml"
    dataset_path = tmp_path / "dataset.yaml"
    experiment_path = tmp_path / "experiment.yaml"

    base_path.write_text(
        """
name: base-run
paths:
  data_root: data
  output_root: outputs
loader:
  batch_size: 16
optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay_mask: exclude_1d_params
  schedule:
    name: constant
regularization:
  label_smoothing: 0.1
ema:
  enabled: true
  decay: 0.999
trainer:
  num_epochs: 5
precision:
  mode: float32
dataset:
  kind: mnist
model:
  name: lru
  hidden_dim: 32
  encoder:
    target: LinearEncoder
  backbone:
    target: LRU
    kwargs:
      num_blocks: 1
      state_dim: 8
      drop_rate: 0.0
  head:
    target: ClassificationHead
checkpoint:
  enabled: true
wandb:
  enabled: false
""".strip()
    )
    dataset_path.write_text(
        """
dataset:
  kind: uea
  name: TinyUEA
  root: raw-data
  validation_split: 0.0
""".strip()
    )
    experiment_path.write_text(
        """
defaults:
  - ./base
  - ./dataset

name: tiny-smoke
tags:
  - smoke
trainer:
  max_steps: 3
wandb:
  tags:
    - ci
""".strip()
    )

    config = load_experiment_config(
        experiment_path,
        overrides=["trainer.max_steps=7", "optimizer.learning_rate=0.01"],
    )

    assert config.name == "tiny-smoke"
    assert config.dataset.kind == "uea"
    assert config.dataset.name == "TinyUEA"
    assert config.loader.batch_size == 16
    assert config.trainer.max_steps == 7
    assert config.optimizer.learning_rate == 0.01
    assert config.optimizer.weight_decay_mask == "exclude_1d_params"
    assert config.regularization.label_smoothing == 0.1
    assert config.ema.enabled is True
    assert config.precision.mode == "float32"

    flattened = flatten_config(config)
    assert flattened["dataset.kind"] == "uea"
    assert flattened["trainer.max_steps"] == 7
    assert flattened["regularization.label_smoothing"] == 0.1
    assert flattened["ema.enabled"] is True
    assert flattened["precision.mode"] == "float32"
    assert flattened["wandb.tags"] == ["ci"]
