"""Smoke tests for the experiment training runtime."""

import pickle
from pathlib import Path

import pytest

from discretax.training import load_experiment_config
from discretax.training.trainer import run_experiment


def _write_tiny_uea_dataset(root: Path) -> None:
    """Write a tiny UEA-style dataset split for smoke testing."""
    dataset_dir = root / "processed" / "UEA" / "TinyUEA"
    dataset_dir.mkdir(parents=True)

    train_inputs = [
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.0, 0.5]],
        [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]],
        [[0.9, 0.1], [0.85, 0.15], [0.8, 0.2], [0.75, 0.25]],
    ]
    validation_inputs = [
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]],
        [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
    ]
    test_inputs = [
        [[0.15, 0.85], [0.25, 0.75], [0.35, 0.65], [0.45, 0.55]],
        [[0.75, 0.25], [0.65, 0.35], [0.55, 0.45], [0.45, 0.55]],
    ]

    payloads = {
        "X_train.pkl": train_inputs,
        "X_val.pkl": validation_inputs,
        "X_test.pkl": test_inputs,
        "y_train.pkl": [0, 1, 0, 1],
        "y_val.pkl": [0, 1],
        "y_test.pkl": [0, 1],
    }
    for name, payload in payloads.items():
        with (dataset_dir / name).open("wb") as file:
            pickle.dump(payload, file)


@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part")
def test_run_experiment_smoke(tmp_path: Path):
    """A tiny configured training run completes and writes outputs."""
    _write_tiny_uea_dataset(tmp_path)

    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        f"""
name: tiny-uea-smoke
paths:
  data_root: {tmp_path}
  output_root: {tmp_path / "outputs"}
loader:
  batch_size: 2
  eval_batch_size: 2
optimizer:
  name: adam
  learning_rate: 0.001
  schedule:
    name: constant
trainer:
  seed: 0
  num_epochs: 1
  max_steps: 2
  log_every_steps: 1
  eval_every_steps: 1
  checkpoint_every_steps: 2
dataset:
  kind: uea
  name: TinyUEA
  root: .
  normalize: true
model:
  name: lru
  hidden_dim: 8
  encoder:
    target: LinearEncoder
  backbone:
    target: LRU
    kwargs:
      num_blocks: 1
      state_dim: 4
      drop_rate: 0.0
  head:
    target: ClassificationHead
    kwargs:
      reduce: true
checkpoint:
  enabled: true
  save_best: true
  monitor: val_loss
  mode: min
wandb:
  enabled: false
""".strip()
    )

    config = load_experiment_config(config_path)
    result = run_experiment(config)

    assert result.final_step == 2
    assert result.output_dir.exists()
    assert (result.output_dir / "config.yaml").exists()
    assert (result.output_dir / "summary.json").exists()
    assert (result.output_dir / "history.jsonl").exists()
    assert (result.output_dir / "checkpoints" / "best").exists()
