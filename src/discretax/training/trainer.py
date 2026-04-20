"""Training runtime for configured Discretax experiments."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from discretax.datasets import DatasetBundle, batch_iterator, build_dataset, count_batches
from discretax.training.config import ExperimentConfig, OptimizerConfig
from discretax.training.ema import update_ema_model
from discretax.training.factory import build_model
from discretax.training.io import (
    append_history,
    checkpoint_run_directory,
    create_run_directory,
    load_checkpoint,
    resolve_checkpoint_directory,
    save_checkpoint,
    write_resolved_config,
    write_run_metadata,
    write_summary,
)
from discretax.training.logging import create_tracker
from discretax.training.metadata import (
    collect_run_metadata,
    finalize_run_metadata,
    tracker_runtime_summary,
)
from discretax.training.regularization import apply_batch_regularization
from discretax.utils.param_count import count_params


@dataclass(slots=True)
class RunResult:
    """Summary of a completed experiment run."""

    output_dir: Path
    best_val_metric: float
    final_step: int
    test_loss: float
    test_metric: float  # accuracy for classification, MSE for regression
    mode: str = "train"


@dataclass(slots=True)
class _RuntimeState:
    """Mutable runtime state for a training or evaluation run."""

    model: eqx.nn.Sequential
    ema_model: eqx.nn.Sequential | None
    state: eqx.nn.State
    opt_state: optax.OptState
    best_val_metric: float
    best_model: eqx.nn.Sequential
    best_state: eqx.nn.State
    evals_without_improvement: int
    should_stop_early: bool
    final_step: int
    start_epoch: int
    steps_to_skip_in_epoch: int
    checkpoint_dir: Path | None
    resume_metadata: dict[str, Any] | None


def _batched_forward(
    model: eqx.nn.Sequential,
    state: eqx.nn.State,
    batch_inputs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, eqx.nn.State]:
    """Run a stateful model over a batch with a named batch axis."""
    batch_keys = jr.split(key, batch_inputs.shape[0])

    def _forward_single(
        single_inputs: jax.Array, shared_state: eqx.nn.State, single_key: jax.Array
    ) -> tuple[jax.Array, eqx.nn.State]:
        return model(single_inputs, shared_state, key=single_key)

    return jax.vmap(
        _forward_single,
        in_axes=(0, None, 0),
        out_axes=(0, None),
        axis_name="batch",
    )(
        batch_inputs,
        state,
        batch_keys,
    )


def _classification_metrics(
    log_probs: jax.Array,
    targets: jax.Array,
    *,
    target_probs: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute classification loss and accuracy."""
    log_probs = log_probs.astype(jnp.float32)
    if target_probs is None:
        loss = -jnp.mean(log_probs[jnp.arange(targets.shape[0]), targets])
    else:
        loss = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
    accuracy = jnp.mean(jnp.argmax(log_probs, axis=-1) == targets)
    return loss, accuracy


def _regression_metrics(
    predictions: jax.Array,
    targets: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute regression MSE loss and MAE."""
    predictions = predictions.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    mse = jnp.mean((predictions - targets) ** 2)
    mae = jnp.mean(jnp.abs(predictions - targets))
    return mse, mae


def _classification_loss(
    log_probs: jax.Array, targets: tuple[jax.Array, jax.Array | None]
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Cross-entropy loss with accuracy reported as an extra metric.

    `targets` is (hard_targets, target_probs); target_probs may be None.
    """
    hard_targets, target_probs = targets
    loss, accuracy = _classification_metrics(log_probs, hard_targets, target_probs=target_probs)
    return loss, {"accuracy": accuracy}


def _regression_loss(
    predictions: jax.Array, targets: jax.Array
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """MSE loss with MAE reported as an extra metric."""
    mse, mae = _regression_metrics(predictions, targets)
    return mse, {"mse": mse, "mae": mae}


def _safe_throughput(count: int, duration_seconds: float) -> float:
    """Convert a work count and duration into a stable throughput metric."""
    if duration_seconds <= 0.0:
        return 0.0
    return count / duration_seconds


def _build_schedule(optimizer_config: OptimizerConfig, total_steps: int):
    """Build an Optax learning-rate schedule."""
    schedule_config = optimizer_config.schedule
    if schedule_config.name == "constant":
        return optax.constant_schedule(optimizer_config.learning_rate)

    decay_steps = schedule_config.decay_steps or total_steps
    if schedule_config.name == "cosine_decay":
        alpha = schedule_config.end_value / optimizer_config.learning_rate
        return optax.cosine_decay_schedule(
            init_value=optimizer_config.learning_rate,
            decay_steps=decay_steps,
            alpha=alpha,
        )

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=optimizer_config.learning_rate,
        warmup_steps=schedule_config.warmup_steps,
        decay_steps=decay_steps,
        end_value=schedule_config.end_value,
    )


def _weight_decay_mask(model: eqx.nn.Sequential, optimizer_config: OptimizerConfig):
    """Build an Optax weight-decay mask for the filtered parameter tree."""
    params = eqx.filter(model, eqx.is_inexact_array)
    if optimizer_config.weight_decay_mask == "all":
        return jax.tree_util.tree_map(lambda leaf: leaf is not None, params)
    return jax.tree_util.tree_map(
        lambda leaf: leaf is not None and getattr(leaf, "ndim", 0) > 1,
        params,
    )


def _build_optimizer(
    optimizer_config: OptimizerConfig,
    total_steps: int,
    *,
    model: eqx.nn.Sequential,
):
    """Build an Optax optimizer and its learning-rate schedule."""
    learning_rate_schedule = _build_schedule(optimizer_config, total_steps)
    transforms: list[Any] = []

    if optimizer_config.grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(optimizer_config.grad_clip_norm))

    if optimizer_config.name == "adam":
        transforms.append(optax.adam(learning_rate=learning_rate_schedule))
    elif optimizer_config.name == "adamw":
        transforms.append(
            optax.adamw(
                learning_rate=learning_rate_schedule,
                weight_decay=optimizer_config.weight_decay,
                mask=_weight_decay_mask(model, optimizer_config),
            )
        )
    else:
        transforms.append(
            optax.sgd(
                learning_rate=learning_rate_schedule,
                momentum=0.9,
                nesterov=True,
            )
        )

    return optax.chain(*transforms), learning_rate_schedule


def _make_train_step(optimizer, compute_loss):
    """Create a compiled train-step function.

    `compute_loss(model_output, targets) -> (scalar_loss, metrics_dict)`.
    `targets` is any pytree; callers pass a single array for regression or a
    `(hard_targets, target_probs)` tuple for classification.
    """

    @eqx.filter_value_and_grad(has_aux=True)
    def _loss_fn(model, state, batch_inputs, targets, key):
        outputs, new_state = _batched_forward(model, state, batch_inputs, key)
        loss, metrics = compute_loss(outputs, targets)
        return loss, (new_state, metrics)

    @eqx.filter_jit
    def train_step(model, state, opt_state, batch_inputs, targets, key):
        (loss, (new_state, metrics)), grads = _loss_fn(model, state, batch_inputs, targets, key)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return (
            new_model,
            new_state,
            new_opt_state,
            {
                "loss": loss,
                **metrics,
                "grad_norm": optax.global_norm(grads),
                "update_norm": optax.global_norm(updates),
                "param_norm": optax.global_norm(eqx.filter(model, eqx.is_inexact_array)),
            },
        )

    return train_step


def _make_eval_step(compute_loss):
    """Create a compiled eval-step. Mirrors `_make_train_step` without grads."""

    @eqx.filter_jit
    def eval_step(model, state, batch_inputs, targets, key):
        inference_model = eqx.nn.inference_mode(model, value=True)
        outputs, _ = _batched_forward(inference_model, state, batch_inputs, key)
        loss, metrics = compute_loss(outputs, targets)
        return {"loss": loss, **metrics}

    return eval_step


def _evaluate(
    model: eqx.nn.Sequential,
    state: eqx.nn.State,
    dataset_bundle: DatasetBundle,
    experiment_config: ExperimentConfig,
    *,
    split_name: str,
    metric_prefix: str,
    seed: int,
    eval_step,
) -> dict[str, float]:
    """Evaluate a model on a dataset split; averages whatever `eval_step` returns.

    Metric keys are prefixed with `metric_prefix` (semantic role — "val" or
    "test") independent of `split_name` (the dataset attribute to read). This
    lets a no-validation-split run still emit `val_*` during training.
    """
    split = getattr(dataset_bundle, split_name)
    is_regression = dataset_bundle.task == "regression"
    expected_keys = ("loss", "mse", "mae") if is_regression else ("loss", "accuracy")
    if len(split) == 0:
        return {f"{metric_prefix}_{k}": float("nan") for k in expected_keys}

    sums: dict[str, float] = {}
    num_batches = 0
    for batch_index, (batch_inputs, batch_targets) in enumerate(
        batch_iterator(
            split,
            experiment_config.loader.resolved_eval_batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed,
        )
    ):
        key = jr.PRNGKey(seed + batch_index)
        targets: Any = batch_targets if is_regression else (batch_targets, None)
        metrics = eval_step(model, state, batch_inputs, targets, key)
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        num_batches += 1

    return {f"{metric_prefix}_{k}": v / num_batches for k, v in sums.items()}


def _is_improved(value: float, best_value: float, mode: str) -> bool:
    """Compare two metrics according to the checkpoint mode."""
    if mode == "min":
        return value < best_value
    return value > best_value


def _checkpoint_metadata(
    *,
    epoch: int,
    step: int,
    best_val_metric: float,
    monitor: str,
    monitor_value: float | None = None,
) -> dict[str, Any]:
    """Build checkpoint metadata consistently."""
    metadata: dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "best_val_metric": best_val_metric,
        "monitor": monitor,
    }
    if monitor_value is not None:
        metadata["monitor_value"] = monitor_value
    return metadata


def _resolve_total_steps(
    dataset_bundle: DatasetBundle,
    experiment_config: ExperimentConfig,
) -> int:
    """Resolve the total number of optimizer steps for a run."""
    epoch_steps = experiment_config.trainer.num_epochs * count_batches(
        dataset_bundle.train,
        experiment_config.loader.batch_size,
        drop_last=experiment_config.loader.drop_last_train,
    )
    if experiment_config.trainer.max_steps is None:
        return epoch_steps
    return min(experiment_config.trainer.max_steps, epoch_steps)


def _initialize_runtime_state(
    experiment_config: ExperimentConfig,
    *,
    model: eqx.nn.Sequential,
    state: eqx.nn.State,
    opt_state: optax.OptState,
    dataset_bundle: DatasetBundle,
    eval_only: bool,
    resume_from: str | Path | None,
) -> _RuntimeState:
    """Restore model runtime state when resuming from a checkpoint."""
    checkpoint_dir: Path | None = None
    resume_metadata: dict[str, Any] | None = None
    ema_model = model if experiment_config.ema.enabled else None
    if resume_from is not None:
        default_checkpoint_name = "best" if eval_only else "latest"
        checkpoint_dir = resolve_checkpoint_directory(
            resume_from,
            default_checkpoint_name=default_checkpoint_name,
        )
        model, ema_model, state, opt_state, resume_metadata = load_checkpoint(
            checkpoint_dir,
            model_like=model,
            ema_model_like=ema_model,
            state_like=state,
            opt_state_like=opt_state,
        )

    best_val_metric = float("inf") if experiment_config.checkpoint.mode == "min" else float("-inf")
    runtime_state = _RuntimeState(
        model=model,
        ema_model=ema_model,
        state=state,
        opt_state=opt_state,
        best_val_metric=best_val_metric,
        best_model=model,
        best_state=state,
        evals_without_improvement=0,
        should_stop_early=False,
        final_step=0,
        start_epoch=0,
        steps_to_skip_in_epoch=0,
        checkpoint_dir=checkpoint_dir,
        resume_metadata=resume_metadata,
    )
    if resume_metadata is None:
        return runtime_state

    runtime_state.final_step = int(resume_metadata["step"])
    runtime_state.best_val_metric = float(
        resume_metadata.get(
            "best_val_metric",
            resume_metadata.get(
                "best_metric",
                resume_metadata.get("monitor_value", runtime_state.best_val_metric),
            ),
        )
    )
    steps_per_epoch = count_batches(
        dataset_bundle.train,
        experiment_config.loader.batch_size,
        drop_last=experiment_config.loader.drop_last_train,
    )
    if steps_per_epoch == 0:
        return runtime_state

    runtime_state.start_epoch = runtime_state.final_step // steps_per_epoch
    runtime_state.steps_to_skip_in_epoch = runtime_state.final_step % steps_per_epoch
    return runtime_state


def _resolve_output_directory(
    experiment_config: ExperimentConfig,
    *,
    checkpoint_dir: Path | None,
    eval_only: bool,
) -> Path:
    """Resolve the output directory for a run."""
    if eval_only:
        return create_run_directory(experiment_config)
    if checkpoint_dir is not None:
        return checkpoint_run_directory(checkpoint_dir)
    return create_run_directory(experiment_config)


def _write_run_config(
    experiment_config: ExperimentConfig,
    *,
    output_dir: Path,
    checkpoint_dir: Path | None,
    eval_only: bool,
) -> None:
    """Write the resolved config when creating a new run directory."""
    if checkpoint_dir is not None and not eval_only:
        return
    write_resolved_config(experiment_config, output_dir)


def _log_static_summary(
    tracker: Any,
    dataset_bundle: DatasetBundle,
    model: eqx.nn.Sequential,
    run_metadata: dict[str, Any],
    experiment_config: ExperimentConfig,
) -> None:
    """Record static dataset and model metadata."""
    tracker.summary(
        {
            "dataset_name": dataset_bundle.name,
            "input_dim": dataset_bundle.input_dim,
            "sequence_length": dataset_bundle.sequence_length,
            "output_dim": dataset_bundle.output_dim,
            "parameter_count": count_params(model),
            "ema_enabled": experiment_config.ema.enabled,
            "label_smoothing": experiment_config.regularization.label_smoothing,
            "mixup_alpha": experiment_config.regularization.mixup_alpha,
            "cutmix_alpha": experiment_config.regularization.cutmix_alpha,
            **tracker_runtime_summary(run_metadata),
        }
    )


def _evaluation_model(runtime_state: _RuntimeState) -> eqx.nn.Sequential:
    """Return the model to use for evaluation and best-checkpoint tracking."""
    return runtime_state.ema_model or runtime_state.model


def _write_completed_run_artifacts(
    *,
    output_dir: Path,
    tracker: Any,
    run_metadata: dict[str, Any],
    summary: dict[str, Any],
    final_step: int,
    best_val_metric: float,
) -> None:
    """Write completion metadata and finalize the tracker."""
    finalized_metadata = finalize_run_metadata(
        run_metadata,
        ended_at=datetime.now(UTC),
        status="completed",
        final_step=final_step,
        best_val_metric=best_val_metric,
    )
    write_run_metadata(output_dir, finalized_metadata)
    tracker.summary(
        {
            "run_status": finalized_metadata["status"],
            "ended_at": finalized_metadata["ended_at"],
            "duration_seconds": finalized_metadata["duration_seconds"],
        }
    )
    write_summary(
        output_dir,
        {
            **summary,
            "started_at": run_metadata["started_at"],
            "ended_at": finalized_metadata["ended_at"],
            "duration_seconds": finalized_metadata["duration_seconds"],
        },
    )
    tracker.finish()


def _run_eval_only(
    experiment_config: ExperimentConfig,
    *,
    dataset_bundle: DatasetBundle,
    runtime_state: _RuntimeState,
    output_dir: Path,
    tracker: Any,
    run_metadata: dict[str, Any],
    eval_step,
) -> RunResult:
    """Evaluate a restored checkpoint without further training."""
    if runtime_state.checkpoint_dir is None or runtime_state.resume_metadata is None:
        raise ValueError("eval_only requires a checkpoint via resume_from")

    test_metrics = _evaluate(
        _evaluation_model(runtime_state),
        runtime_state.state,
        dataset_bundle,
        experiment_config,
        split_name="test",
        metric_prefix="test",
        seed=experiment_config.trainer.seed + runtime_state.final_step + 1,
        eval_step=eval_step,
    )
    tracker.summary(
        {
            "restored_step": runtime_state.final_step,
            **test_metrics,
        }
    )
    _write_completed_run_artifacts(
        output_dir=output_dir,
        tracker=tracker,
        run_metadata=run_metadata,
        summary={
            "mode": "eval_only",
            "restored_checkpoint": str(runtime_state.checkpoint_dir),
            "restored_step": runtime_state.final_step,
            **test_metrics,
        },
        final_step=runtime_state.final_step,
        best_val_metric=runtime_state.best_val_metric,
    )
    secondary_key = "test_mse" if dataset_bundle.task == "regression" else "test_accuracy"
    return RunResult(
        output_dir=output_dir,
        best_val_metric=runtime_state.best_val_metric,
        final_step=runtime_state.final_step,
        test_loss=test_metrics["test_loss"],
        test_metric=test_metrics[secondary_key],
        mode="eval_only",
    )


def _maybe_run_evaluation(
    experiment_config: ExperimentConfig,
    *,
    dataset_bundle: DatasetBundle,
    runtime_state: _RuntimeState,
    output_dir: Path,
    tracker: Any,
    epoch: int,
    validation_split_name: str,
    eval_step,
    total_steps: int,
) -> None:
    """Evaluate the current model, track metrics, and update best checkpoint state."""
    if (
        runtime_state.final_step % experiment_config.trainer.eval_every_steps != 0
        and runtime_state.final_step != total_steps
    ):
        return

    eval_started_at = perf_counter()
    evaluation_metrics = _evaluate(
        _evaluation_model(runtime_state),
        runtime_state.state,
        dataset_bundle,
        experiment_config,
        split_name=validation_split_name,
        metric_prefix="val",
        seed=experiment_config.trainer.seed + runtime_state.final_step,
        eval_step=eval_step,
    )
    eval_duration_seconds = perf_counter() - eval_started_at
    split = getattr(dataset_bundle, validation_split_name)
    evaluation_metrics["step"] = runtime_state.final_step
    evaluation_metrics["val_duration_seconds"] = eval_duration_seconds
    evaluation_metrics["val_examples_per_second"] = _safe_throughput(
        len(split),
        eval_duration_seconds,
    )
    append_history(output_dir, evaluation_metrics)
    tracker.log(evaluation_metrics, step=runtime_state.final_step)

    monitored_value = evaluation_metrics[experiment_config.checkpoint.monitor]
    if not _is_improved(
        monitored_value,
        runtime_state.best_val_metric,
        experiment_config.checkpoint.mode,
    ):
        runtime_state.evals_without_improvement += 1
        patience = experiment_config.trainer.early_stopping_patience
        if patience is not None and runtime_state.evals_without_improvement >= patience:
            runtime_state.should_stop_early = True
        return

    runtime_state.evals_without_improvement = 0
    runtime_state.best_val_metric = monitored_value
    runtime_state.best_model = _evaluation_model(runtime_state)
    runtime_state.best_state = runtime_state.state
    if not (experiment_config.checkpoint.enabled and experiment_config.checkpoint.save_best):
        return

    save_checkpoint(
        output_dir,
        "best",
        model=runtime_state.best_model,
        ema_model=runtime_state.ema_model if runtime_state.ema_model is not None else None,
        state=runtime_state.best_state,
        opt_state=runtime_state.opt_state,
        metadata=_checkpoint_metadata(
            epoch=epoch,
            step=runtime_state.final_step,
            best_val_metric=runtime_state.best_val_metric,
            monitor=experiment_config.checkpoint.monitor,
            monitor_value=monitored_value,
        ),
    )


def _maybe_save_progress_checkpoint(
    experiment_config: ExperimentConfig,
    *,
    runtime_state: _RuntimeState,
    output_dir: Path,
    epoch: int,
    total_steps: int,
) -> None:
    """Persist latest and step checkpoints at configured save intervals."""
    if not experiment_config.checkpoint.enabled:
        return

    save_latest = runtime_state.final_step % experiment_config.trainer.checkpoint_every_steps == 0
    save_latest = save_latest or runtime_state.final_step == total_steps
    if not save_latest:
        return

    metadata = _checkpoint_metadata(
        epoch=epoch,
        step=runtime_state.final_step,
        best_val_metric=runtime_state.best_val_metric,
        monitor=experiment_config.checkpoint.monitor,
    )
    save_checkpoint(
        output_dir,
        "latest",
        model=runtime_state.model,
        ema_model=runtime_state.ema_model if runtime_state.ema_model is not None else None,
        state=runtime_state.state,
        opt_state=runtime_state.opt_state,
        metadata=metadata,
    )
    save_checkpoint(
        output_dir,
        f"step-{runtime_state.final_step}",
        model=runtime_state.model,
        ema_model=runtime_state.ema_model if runtime_state.ema_model is not None else None,
        state=runtime_state.state,
        opt_state=runtime_state.opt_state,
        metadata=metadata,
    )


def _compute_speed_summary(output_dir: Path) -> dict[str, float]:
    """Read history.jsonl and return median throughput for train and eval.

    Skips the first few train records to avoid JIT compilation noise.
    Train records are identified by having 'train_loss'. Eval records are
    identified by having a key ending in '_examples_per_second' without
    'train' in the key name.
    """
    history_file = output_dir / "history.jsonl"
    if not history_file.exists():
        return {}

    train_eps: list[float] = []
    eval_eps: list[float] = []
    with history_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "train_loss" in record and "examples_per_second" in record:
                train_eps.append(float(record["examples_per_second"]))
            else:
                for key, value in record.items():
                    if key.endswith("_examples_per_second") and "train" not in key:
                        eval_eps.append(float(value))
                        break

    # Drop first 3 train steps to avoid JIT warmup noise
    _warmup_steps = 3
    train_eps = train_eps[_warmup_steps:]

    result: dict[str, float] = {}
    if train_eps:
        result["median_train_examples_per_second"] = statistics.median(train_eps)
    if eval_eps:
        result["median_eval_examples_per_second"] = statistics.median(eval_eps)
    return result


def _finalize_training_run(
    experiment_config: ExperimentConfig,
    *,
    dataset_bundle: DatasetBundle,
    runtime_state: _RuntimeState,
    output_dir: Path,
    tracker: Any,
    run_metadata: dict[str, Any],
    eval_step,
) -> RunResult:
    """Evaluate the best checkpoint on test data and write final outputs."""
    test_metrics = _evaluate(
        runtime_state.best_model,
        runtime_state.best_state,
        dataset_bundle,
        experiment_config,
        split_name="test",
        metric_prefix="test",
        seed=experiment_config.trainer.seed + runtime_state.final_step + 1,
        eval_step=eval_step,
    )
    speed_summary = _compute_speed_summary(output_dir)
    parameter_count = count_params(runtime_state.best_model)
    tracker.summary(
        {
            "best_val_metric": runtime_state.best_val_metric,
            **test_metrics,
            "final_step": runtime_state.final_step,
            "parameter_count": parameter_count,
            "stopped_early": runtime_state.should_stop_early,
            **speed_summary,
        }
    )
    _write_completed_run_artifacts(
        output_dir=output_dir,
        tracker=tracker,
        run_metadata=run_metadata,
        summary={
            "mode": "train",
            "best_val_metric": runtime_state.best_val_metric,
            "final_step": runtime_state.final_step,
            "parameter_count": parameter_count,
            "stopped_early": runtime_state.should_stop_early,
            **test_metrics,
            **speed_summary,
        },
        final_step=runtime_state.final_step,
        best_val_metric=runtime_state.best_val_metric,
    )
    secondary_key = "test_mse" if dataset_bundle.task == "regression" else "test_accuracy"
    return RunResult(
        output_dir=output_dir,
        best_val_metric=runtime_state.best_val_metric,
        final_step=runtime_state.final_step,
        test_loss=test_metrics["test_loss"],
        test_metric=test_metrics[secondary_key],
        mode="train",
    )


def _execute_train_step(
    runtime_state: _RuntimeState,
    batch_inputs: Any,
    batch_targets: Any,
    train_step: Any,
    is_regression: bool,
    experiment_config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    batch_aug_key: Any,
    step_key: Any,
) -> tuple[_RuntimeState, dict[str, Any]]:
    if is_regression:
        step_inputs = batch_inputs
        targets: Any = batch_targets
        aug_extras: dict[str, Any] = {}
    else:
        regularized_batch = apply_batch_regularization(
            batch_inputs,
            batch_targets,
            num_classes=dataset_bundle.output_dim,
            regularization_config=experiment_config.regularization,
            dataset_metadata=dataset_bundle.train.metadata,
            key=batch_aug_key,
        )
        step_inputs = regularized_batch.inputs
        targets = (regularized_batch.hard_targets, regularized_batch.target_probs)
        aug_extras = {
            "mix_augmentation_applied": float(regularized_batch.applied),
            "mix_augmentation_lambda": regularized_batch.lambda_value,
        }

    (
        runtime_state.model,
        runtime_state.state,
        runtime_state.opt_state,
        raw_metrics,
    ) = train_step(
        runtime_state.model,
        runtime_state.state,
        runtime_state.opt_state,
        step_inputs,
        targets,
        step_key,
    )
    if is_regression:
        extras: dict[str, Any] = {
            "train_mse": float(raw_metrics["mse"]),
            "train_mae": float(raw_metrics["mae"]),
        }
    else:
        extras = {
            "train_accuracy": float(raw_metrics["accuracy"]),
            **aug_extras,
        }
    metrics: dict[str, Any] = {
        "loss": float(raw_metrics["loss"]),
        "grad_norm": float(raw_metrics["grad_norm"]),
        "update_norm": float(raw_metrics["update_norm"]),
        "param_norm": float(raw_metrics["param_norm"]),
        "extras": extras,
    }
    return runtime_state, metrics


def run_experiment(
    experiment_config: ExperimentConfig,
    *,
    resume_from: str | Path | None = None,
    eval_only: bool = False,
) -> RunResult:
    """Execute a configured experiment end to end."""
    dataset_bundle = build_dataset(experiment_config.paths, experiment_config.dataset)

    master_key = jr.PRNGKey(experiment_config.trainer.seed)
    model_key, loop_key = jr.split(master_key)
    model = build_model(experiment_config, dataset_bundle, model_key)
    state = eqx.nn.State(model)

    total_steps = _resolve_total_steps(dataset_bundle, experiment_config)

    optimizer, learning_rate_schedule = _build_optimizer(
        experiment_config.optimizer,
        total_steps,
        model=model,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    runtime_state = _initialize_runtime_state(
        experiment_config,
        model=model,
        state=state,
        opt_state=opt_state,
        dataset_bundle=dataset_bundle,
        eval_only=eval_only,
        resume_from=resume_from,
    )
    output_dir = _resolve_output_directory(
        experiment_config,
        checkpoint_dir=runtime_state.checkpoint_dir,
        eval_only=eval_only,
    )
    run_metadata = collect_run_metadata(
        output_dir=output_dir,
        mode="eval_only" if eval_only else "train",
        started_at=datetime.now(UTC),
        resume_from=resume_from,
        restored_checkpoint=runtime_state.checkpoint_dir,
    )

    tracker = create_tracker(experiment_config, output_dir)
    _write_run_config(
        experiment_config,
        output_dir=output_dir,
        checkpoint_dir=runtime_state.checkpoint_dir,
        eval_only=eval_only,
    )
    write_run_metadata(output_dir, run_metadata)

    is_regression = dataset_bundle.task == "regression"
    compute_loss = _regression_loss if is_regression else _classification_loss
    train_step = _make_train_step(optimizer, compute_loss)
    eval_step = _make_eval_step(compute_loss)
    _log_static_summary(
        tracker,
        dataset_bundle,
        runtime_state.model,
        run_metadata,
        experiment_config,
    )

    validation_split_name = "validation" if len(dataset_bundle.validation) > 0 else "test"

    try:
        if eval_only:
            return _run_eval_only(
                experiment_config,
                dataset_bundle=dataset_bundle,
                runtime_state=runtime_state,
                output_dir=output_dir,
                tracker=tracker,
                run_metadata=run_metadata,
                eval_step=eval_step,
            )

        for epoch in range(runtime_state.start_epoch, experiment_config.trainer.num_epochs):
            epoch_seed = experiment_config.trainer.seed + epoch
            epoch_batches = list(
                batch_iterator(
                    dataset_bundle.train,
                    experiment_config.loader.batch_size,
                    shuffle=experiment_config.loader.shuffle_train,
                    drop_last=experiment_config.loader.drop_last_train,
                    seed=epoch_seed,
                )
            )
            batch_start_index = (
                runtime_state.steps_to_skip_in_epoch if epoch == runtime_state.start_epoch else 0
            )
            for batch_inputs, batch_targets in epoch_batches[batch_start_index:]:
                runtime_state.steps_to_skip_in_epoch = 0
                if runtime_state.final_step >= total_steps:
                    break

                step_started_at = perf_counter()
                loop_key, batch_aug_key, step_key = jr.split(loop_key, 3)

                runtime_state, task_step_metrics = _execute_train_step(
                    runtime_state=runtime_state,
                    batch_inputs=batch_inputs,
                    batch_targets=batch_targets,
                    train_step=train_step,
                    is_regression=is_regression,
                    experiment_config=experiment_config,
                    dataset_bundle=dataset_bundle,
                    batch_aug_key=batch_aug_key,
                    step_key=step_key,
                )

                step_duration_seconds = perf_counter() - step_started_at
                runtime_state.final_step += 1
                if runtime_state.ema_model is not None:
                    runtime_state.ema_model = update_ema_model(
                        runtime_state.ema_model,
                        runtime_state.model,
                        decay=experiment_config.ema.decay,
                    )

                step_metrics: dict[str, Any] = {
                    "epoch": epoch,
                    "step": runtime_state.final_step,
                    "learning_rate": float(learning_rate_schedule(runtime_state.final_step - 1)),
                    "train_loss": task_step_metrics["loss"],
                    "grad_norm": task_step_metrics["grad_norm"],
                    "update_norm": task_step_metrics["update_norm"],
                    "param_norm": task_step_metrics["param_norm"],
                    "step_time_seconds": step_duration_seconds,
                    "examples_per_second": _safe_throughput(
                        batch_inputs.shape[0],
                        step_duration_seconds,
                    ),
                    **task_step_metrics["extras"],
                }

                if runtime_state.final_step % experiment_config.trainer.log_every_steps == 0:
                    append_history(output_dir, step_metrics)
                    tracker.log(step_metrics, step=runtime_state.final_step)

                _maybe_run_evaluation(
                    experiment_config,
                    dataset_bundle=dataset_bundle,
                    runtime_state=runtime_state,
                    output_dir=output_dir,
                    tracker=tracker,
                    epoch=epoch,
                    validation_split_name=validation_split_name,
                    eval_step=eval_step,
                    total_steps=total_steps,
                )
                _maybe_save_progress_checkpoint(
                    experiment_config,
                    runtime_state=runtime_state,
                    output_dir=output_dir,
                    epoch=epoch,
                    total_steps=total_steps,
                )

                if runtime_state.should_stop_early:
                    break

            if runtime_state.final_step >= total_steps or runtime_state.should_stop_early:
                break

        return _finalize_training_run(
            experiment_config,
            dataset_bundle=dataset_bundle,
            runtime_state=runtime_state,
            output_dir=output_dir,
            tracker=tracker,
            run_metadata=run_metadata,
            eval_step=eval_step,
        )
    except Exception as error:
        write_run_metadata(
            output_dir,
            finalize_run_metadata(
                run_metadata,
                ended_at=datetime.now(UTC),
                status="failed",
                final_step=runtime_state.final_step,
                best_val_metric=runtime_state.best_val_metric,
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                },
            ),
        )
        tracker.summary({"run_status": "failed"})
        tracker.finish()
        raise
