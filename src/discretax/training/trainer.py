"""Training runtime for configured Discretax experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from discretax.datasets import DatasetBundle, batch_iterator, build_dataset, count_batches
from discretax.training.config import ExperimentConfig, OptimizerConfig
from discretax.training.factory import build_model
from discretax.training.io import (
    append_history,
    checkpoint_run_directory,
    create_run_directory,
    load_checkpoint,
    resolve_checkpoint_directory,
    save_checkpoint,
    write_resolved_config,
    write_summary,
)
from discretax.training.logging import create_tracker
from discretax.utils.param_count import count_params


@dataclass(slots=True)
class RunResult:
    """Summary of a completed experiment run."""

    output_dir: Path
    best_metric: float
    final_step: int
    test_loss: float
    test_accuracy: float
    mode: str = "train"


@dataclass(slots=True)
class _RuntimeState:
    """Mutable runtime state for a training or evaluation run."""

    model: eqx.nn.Sequential
    state: eqx.nn.State
    opt_state: optax.OptState
    best_metric: float
    best_model: eqx.nn.Sequential
    best_state: eqx.nn.State
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
    log_probs: jax.Array, targets: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute classification loss and accuracy."""
    loss = -jnp.mean(log_probs[jnp.arange(targets.shape[0]), targets])
    accuracy = jnp.mean(jnp.argmax(log_probs, axis=-1) == targets)
    return loss, accuracy


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


def _build_optimizer(optimizer_config: OptimizerConfig, total_steps: int):
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


def _make_train_step(optimizer):
    """Create a compiled train-step function for a specific optimizer."""

    @eqx.filter_value_and_grad(has_aux=True)
    def _loss_fn(
        model: eqx.nn.Sequential,
        state: eqx.nn.State,
        batch_inputs: jax.Array,
        batch_targets: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, tuple[eqx.nn.State, jax.Array]]:
        log_probs, new_state = _batched_forward(model, state, batch_inputs, key)
        loss, accuracy = _classification_metrics(log_probs, batch_targets)
        return loss, (new_state, accuracy)

    @eqx.filter_jit
    def train_step(
        model: eqx.nn.Sequential,
        state: eqx.nn.State,
        opt_state: optax.OptState,
        batch_inputs: jax.Array,
        batch_targets: jax.Array,
        key: jax.Array,
    ) -> tuple[eqx.nn.Sequential, eqx.nn.State, optax.OptState, dict[str, jax.Array]]:
        (loss, (new_state, accuracy)), grads = _loss_fn(
            model,
            state,
            batch_inputs,
            batch_targets,
            key,
        )
        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params=eqx.filter(model, eqx.is_inexact_array),
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, new_opt_state, {"loss": loss, "accuracy": accuracy}

    return train_step


@eqx.filter_jit
def _eval_step(
    model: eqx.nn.Sequential,
    state: eqx.nn.State,
    batch_inputs: jax.Array,
    batch_targets: jax.Array,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Run a single evaluation step."""
    inference_model = eqx.nn.inference_mode(model, value=True)
    log_probs, _ = _batched_forward(inference_model, state, batch_inputs, key)
    loss, accuracy = _classification_metrics(log_probs, batch_targets)
    return {"loss": loss, "accuracy": accuracy}


def _evaluate(
    model: eqx.nn.Sequential,
    state: eqx.nn.State,
    dataset_bundle: DatasetBundle,
    experiment_config: ExperimentConfig,
    *,
    split_name: str,
    seed: int,
) -> dict[str, float]:
    """Evaluate a model on a dataset split."""
    split = getattr(dataset_bundle, split_name)
    if len(split) == 0:
        return {f"{split_name}_loss": float("nan"), f"{split_name}_accuracy": float("nan")}

    losses: list[float] = []
    accuracies: list[float] = []
    for batch_index, (batch_inputs, batch_targets) in enumerate(
        batch_iterator(
            split,
            experiment_config.loader.resolved_eval_batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed,
        )
    ):
        metrics = _eval_step(
            model, state, batch_inputs, batch_targets, jr.PRNGKey(seed + batch_index)
        )
        losses.append(float(metrics["loss"]))
        accuracies.append(float(metrics["accuracy"]))

    return {
        f"{split_name}_loss": sum(losses) / len(losses),
        f"{split_name}_accuracy": sum(accuracies) / len(accuracies),
    }


def _is_improved(value: float, best_value: float, mode: str) -> bool:
    """Compare two metrics according to the checkpoint mode."""
    if mode == "min":
        return value < best_value
    return value > best_value


def _checkpoint_metadata(
    *,
    epoch: int,
    step: int,
    best_metric: float,
    monitor: str,
    monitor_value: float | None = None,
) -> dict[str, Any]:
    """Build checkpoint metadata consistently."""
    metadata: dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "monitor": monitor,
    }
    if monitor_value is not None:
        metadata["monitor_value"] = monitor_value
    return metadata


def _steps_per_epoch(dataset_bundle: DatasetBundle, experiment_config: ExperimentConfig) -> int:
    """Return the deterministic number of train batches per epoch."""
    return count_batches(
        dataset_bundle.train,
        experiment_config.loader.batch_size,
        drop_last=experiment_config.loader.drop_last_train,
    )


def _resolve_total_steps(
    dataset_bundle: DatasetBundle,
    experiment_config: ExperimentConfig,
) -> int:
    """Resolve the total number of optimizer steps for a run."""
    if experiment_config.trainer.max_steps is not None:
        return experiment_config.trainer.max_steps
    return experiment_config.trainer.num_epochs * _steps_per_epoch(
        dataset_bundle,
        experiment_config,
    )


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
    if resume_from is not None:
        default_checkpoint_name = "best" if eval_only else "latest"
        checkpoint_dir = resolve_checkpoint_directory(
            resume_from,
            default_checkpoint_name=default_checkpoint_name,
        )
        model, state, opt_state, resume_metadata = load_checkpoint(
            checkpoint_dir,
            model_like=model,
            state_like=state,
            opt_state_like=opt_state,
        )

    best_metric = float("inf") if experiment_config.checkpoint.mode == "min" else float("-inf")
    runtime_state = _RuntimeState(
        model=model,
        state=state,
        opt_state=opt_state,
        best_metric=best_metric,
        best_model=model,
        best_state=state,
        final_step=0,
        start_epoch=0,
        steps_to_skip_in_epoch=0,
        checkpoint_dir=checkpoint_dir,
        resume_metadata=resume_metadata,
    )
    if resume_metadata is None:
        return runtime_state

    runtime_state.final_step = int(resume_metadata["step"])
    runtime_state.best_metric = float(
        resume_metadata.get(
            "best_metric",
            resume_metadata.get("monitor_value", runtime_state.best_metric),
        )
    )
    steps_per_epoch = _steps_per_epoch(dataset_bundle, experiment_config)
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
) -> None:
    """Record static dataset and model metadata."""
    tracker.summary(
        {
            "dataset_name": dataset_bundle.name,
            "input_dim": dataset_bundle.input_dim,
            "sequence_length": dataset_bundle.sequence_length,
            "num_classes": dataset_bundle.num_classes,
            "parameter_count": count_params(model),
        }
    )


def _run_eval_only(
    experiment_config: ExperimentConfig,
    *,
    dataset_bundle: DatasetBundle,
    runtime_state: _RuntimeState,
    output_dir: Path,
    tracker: Any,
) -> RunResult:
    """Evaluate a restored checkpoint without further training."""
    if runtime_state.checkpoint_dir is None or runtime_state.resume_metadata is None:
        raise ValueError("eval_only requires a checkpoint via resume_from")

    test_metrics = _evaluate(
        runtime_state.model,
        runtime_state.state,
        dataset_bundle,
        experiment_config,
        split_name="test",
        seed=experiment_config.trainer.seed + runtime_state.final_step + 1,
    )
    tracker.summary(
        {
            "restored_step": runtime_state.final_step,
            **test_metrics,
        }
    )
    write_summary(
        output_dir,
        {
            "mode": "eval_only",
            "restored_checkpoint": str(runtime_state.checkpoint_dir),
            "restored_step": runtime_state.final_step,
            **test_metrics,
        },
    )
    tracker.finish()
    return RunResult(
        output_dir=output_dir,
        best_metric=runtime_state.best_metric,
        final_step=runtime_state.final_step,
        test_loss=test_metrics["test_loss"],
        test_accuracy=test_metrics["test_accuracy"],
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
) -> None:
    """Evaluate the current model, track metrics, and update best checkpoint state."""
    if (
        runtime_state.final_step % experiment_config.trainer.eval_every_steps != 0
        and runtime_state.final_step != _resolve_total_steps(dataset_bundle, experiment_config)
    ):
        return

    evaluation_metrics = _evaluate(
        runtime_state.model,
        runtime_state.state,
        dataset_bundle,
        experiment_config,
        split_name=validation_split_name,
        seed=experiment_config.trainer.seed + runtime_state.final_step,
    )
    evaluation_metrics["step"] = runtime_state.final_step
    append_history(output_dir, evaluation_metrics)
    tracker.log(evaluation_metrics, step=runtime_state.final_step)

    monitored_value = evaluation_metrics[
        f"{validation_split_name}_{experiment_config.checkpoint.monitor.removeprefix('val_')}"
    ]
    if not _is_improved(
        monitored_value,
        runtime_state.best_metric,
        experiment_config.checkpoint.mode,
    ):
        return

    runtime_state.best_metric = monitored_value
    runtime_state.best_model = runtime_state.model
    runtime_state.best_state = runtime_state.state
    if not (experiment_config.checkpoint.enabled and experiment_config.checkpoint.save_best):
        return

    save_checkpoint(
        output_dir,
        "best",
        model=runtime_state.best_model,
        state=runtime_state.best_state,
        opt_state=runtime_state.opt_state,
        metadata=_checkpoint_metadata(
            epoch=epoch,
            step=runtime_state.final_step,
            best_metric=runtime_state.best_metric,
            monitor=experiment_config.checkpoint.monitor,
            monitor_value=monitored_value,
        ),
    )


def _maybe_save_progress_checkpoint(
    experiment_config: ExperimentConfig,
    *,
    dataset_bundle: DatasetBundle,
    runtime_state: _RuntimeState,
    output_dir: Path,
    epoch: int,
) -> None:
    """Persist latest and step checkpoints at configured save intervals."""
    if not experiment_config.checkpoint.enabled:
        return

    total_steps = _resolve_total_steps(dataset_bundle, experiment_config)
    save_latest = runtime_state.final_step % experiment_config.trainer.checkpoint_every_steps == 0
    save_latest = save_latest or runtime_state.final_step == total_steps
    if not save_latest:
        return

    metadata = _checkpoint_metadata(
        epoch=epoch,
        step=runtime_state.final_step,
        best_metric=runtime_state.best_metric,
        monitor=experiment_config.checkpoint.monitor,
    )
    save_checkpoint(
        output_dir,
        "latest",
        model=runtime_state.model,
        state=runtime_state.state,
        opt_state=runtime_state.opt_state,
        metadata=metadata,
    )
    save_checkpoint(
        output_dir,
        f"step-{runtime_state.final_step}",
        model=runtime_state.model,
        state=runtime_state.state,
        opt_state=runtime_state.opt_state,
        metadata=metadata,
    )


def _finalize_training_run(
    experiment_config: ExperimentConfig,
    *,
    dataset_bundle: DatasetBundle,
    runtime_state: _RuntimeState,
    output_dir: Path,
    tracker: Any,
) -> RunResult:
    """Evaluate the best checkpoint on test data and write final outputs."""
    test_metrics = _evaluate(
        runtime_state.best_model,
        runtime_state.best_state,
        dataset_bundle,
        experiment_config,
        split_name="test",
        seed=experiment_config.trainer.seed + runtime_state.final_step + 1,
    )
    tracker.summary(
        {
            "best_metric": runtime_state.best_metric,
            **test_metrics,
            "final_step": runtime_state.final_step,
        }
    )
    write_summary(
        output_dir,
        {
            "mode": "train",
            "best_metric": runtime_state.best_metric,
            "final_step": runtime_state.final_step,
            **test_metrics,
        },
    )
    tracker.finish()
    return RunResult(
        output_dir=output_dir,
        best_metric=runtime_state.best_metric,
        final_step=runtime_state.final_step,
        test_loss=test_metrics["test_loss"],
        test_accuracy=test_metrics["test_accuracy"],
        mode="train",
    )


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

    optimizer, learning_rate_schedule = _build_optimizer(experiment_config.optimizer, total_steps)
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

    tracker = create_tracker(experiment_config, output_dir)
    _write_run_config(
        experiment_config,
        output_dir=output_dir,
        checkpoint_dir=runtime_state.checkpoint_dir,
        eval_only=eval_only,
    )

    train_step = _make_train_step(optimizer)
    _log_static_summary(tracker, dataset_bundle, runtime_state.model)

    validation_split_name = "validation" if len(dataset_bundle.validation) > 0 else "test"

    if eval_only:
        return _run_eval_only(
            experiment_config,
            dataset_bundle=dataset_bundle,
            runtime_state=runtime_state,
            output_dir=output_dir,
            tracker=tracker,
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

            loop_key, step_key = jr.split(loop_key)
            (
                runtime_state.model,
                runtime_state.state,
                runtime_state.opt_state,
                train_metrics,
            ) = train_step(
                runtime_state.model,
                runtime_state.state,
                runtime_state.opt_state,
                batch_inputs,
                batch_targets,
                step_key,
            )
            runtime_state.final_step += 1

            step_metrics = {
                "epoch": epoch,
                "step": runtime_state.final_step,
                "learning_rate": float(learning_rate_schedule(runtime_state.final_step - 1)),
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
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
            )
            _maybe_save_progress_checkpoint(
                experiment_config,
                dataset_bundle=dataset_bundle,
                runtime_state=runtime_state,
                output_dir=output_dir,
                epoch=epoch,
            )

        if runtime_state.final_step >= total_steps:
            break

    return _finalize_training_run(
        experiment_config,
        dataset_bundle=dataset_bundle,
        runtime_state=runtime_state,
        output_dir=output_dir,
        tracker=tracker,
    )
