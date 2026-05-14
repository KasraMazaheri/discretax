"""Smoke tests for the experiment training runtime."""

import pickle
from json import loads
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest

from discretax.datasets import build_dataset
from discretax.training import load_experiment_config
from discretax.training.factory import build_model
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


def _write_runtime_config(tmp_path: Path, *, max_steps: int) -> Path:
    """Write a tiny runtime config and return its path."""
    config_path = tmp_path / f"runtime-{max_steps}.yaml"
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
  weight_decay_mask: exclude_1d_params
  schedule:
    name: constant
regularization:
  label_smoothing: 0.1
ema:
  enabled: true
  decay: 0.99
trainer:
  seed: 0
  num_epochs: 2
  max_steps: {max_steps}
  log_every_steps: 1
  eval_every_steps: 1
  checkpoint_every_steps: 1
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
    return config_path


@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part")
def test_run_experiment_smoke(tmp_path: Path):
    """A tiny configured training run completes and writes outputs."""
    _write_tiny_uea_dataset(tmp_path)
    config = load_experiment_config(_write_runtime_config(tmp_path, max_steps=2))
    result = run_experiment(config)

    assert result.final_step == 2
    assert result.output_dir.exists()
    assert (result.output_dir / "config.yaml").exists()
    assert (result.output_dir / "summary.json").exists()
    assert (result.output_dir / "history.jsonl").exists()
    assert (result.output_dir / "run_metadata.json").exists()
    assert (result.output_dir / "checkpoints" / "best").exists()
    assert (result.output_dir / "checkpoints" / "latest").exists()
    run_metadata = loads((result.output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["status"] == "completed"
    assert run_metadata["mode"] == "train"
    assert run_metadata["ended_at"] is not None
    assert run_metadata["jax"]["device_count"] >= 1
    assert "hostname" in run_metadata
    history_records = [
        loads(line)
        for line in (result.output_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any("grad_norm" in record for record in history_records)
    assert any("mix_augmentation_lambda" in record for record in history_records)
    assert any("step_time_seconds" in record for record in history_records)
    assert any("validation_duration_seconds" in record for record in history_records)


def test_run_experiment_bfloat16_precision_policy(tmp_path: Path):
    """A tiny run can execute with a low-precision runtime policy."""
    _write_tiny_uea_dataset(tmp_path)
    config = load_experiment_config(
        _write_runtime_config(tmp_path, max_steps=2),
        overrides=[
            "model.name=linoss",
            "model.backbone.target=LinOSS",
            "precision.mode=bfloat16_mixed",
        ],
    )

    result = run_experiment(config)

    assert result.final_step == 2
    assert jnp.isfinite(result.test_loss)
    assert jnp.isfinite(result.test_metric)


def test_linoss_mixed_precision_initializes_backbone_in_target_dtype(tmp_path: Path):
    """LinOSS backbone modules are initialized directly in the mixed compute dtype."""
    _write_tiny_uea_dataset(tmp_path)
    config = load_experiment_config(
        _write_runtime_config(tmp_path, max_steps=2),
        overrides=[
            "model.name=linoss",
            "model.backbone.target=LinOSS",
            "precision.mode=bfloat16_mixed",
        ],
    )
    dataset_bundle = build_dataset(config.paths, config.dataset)
    model = build_model(config, dataset_bundle, jr.PRNGKey(0))

    encoder = model.layers[0]
    backbone = model.layers[1]
    head = model.layers[2]
    sequence_mixer = backbone.blocks[0].sequence_mixer
    channel_mixer = backbone.blocks[0].channel_mixer

    assert encoder.linear.weight.dtype == jnp.bfloat16
    assert backbone.compute_dtype == jnp.dtype(jnp.bfloat16)
    assert sequence_mixer.B.dtype == jnp.bfloat16
    assert sequence_mixer.C.dtype == jnp.bfloat16
    assert sequence_mixer.D.dtype == jnp.bfloat16
    assert channel_mixer.w1.weight.dtype == jnp.bfloat16
    assert channel_mixer.w2.weight.dtype == jnp.bfloat16
    assert head.linear.weight.dtype == jnp.bfloat16


def test_run_experiment_rejects_mixed_precision_for_non_linoss(tmp_path: Path):
    """Mixed precision is currently rejected for non-LinOSS backbones."""
    _write_tiny_uea_dataset(tmp_path)
    config = load_experiment_config(
        _write_runtime_config(tmp_path, max_steps=2),
        overrides=["precision.mode=bfloat16_mixed"],
    )

    with pytest.raises(ValueError, match="only for LinOSS"):
        run_experiment(config)


@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part")
def test_run_experiment_resume_from_run_directory(tmp_path: Path):
    """Training resumes from the latest checkpoint in an existing run directory."""
    _write_tiny_uea_dataset(tmp_path)

    initial_config = load_experiment_config(_write_runtime_config(tmp_path, max_steps=2))
    initial_result = run_experiment(initial_config)

    resumed_config = load_experiment_config(_write_runtime_config(tmp_path, max_steps=4))
    resumed_result = run_experiment(resumed_config, resume_from=initial_result.output_dir)

    assert resumed_result.output_dir == initial_result.output_dir
    assert resumed_result.final_step == 4
    assert resumed_result.mode == "train"
    history_lines = (
        (initial_result.output_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(history_lines) >= 8
    latest_metadata = loads(
        (initial_result.output_dir / "checkpoints" / "latest" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest_metadata["step"] == 4


@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part")
def test_run_experiment_eval_only_from_run_directory(tmp_path: Path):
    """Eval-only mode restores a checkpoint and writes a fresh evaluation run."""
    _write_tiny_uea_dataset(tmp_path)

    train_config = load_experiment_config(_write_runtime_config(tmp_path, max_steps=2))
    train_result = run_experiment(train_config)

    eval_result = run_experiment(train_config, resume_from=train_result.output_dir, eval_only=True)

    assert eval_result.mode == "eval_only"
    assert eval_result.output_dir != train_result.output_dir
    assert (eval_result.output_dir / "summary.json").exists()
    assert (eval_result.output_dir / "run_metadata.json").exists()
    summary = loads((eval_result.output_dir / "summary.json").read_text(encoding="utf-8"))
    run_metadata = loads(
        (eval_result.output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    restored_metadata = loads(
        (train_result.output_dir / "checkpoints" / "best" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["mode"] == "eval_only"
    assert summary["restored_step"] == restored_metadata["step"]
    assert summary["duration_seconds"] >= 0.0
    assert run_metadata["mode"] == "eval_only"
    assert run_metadata["restored_checkpoint"].endswith("/checkpoints/best")
    assert run_metadata["status"] == "completed"
