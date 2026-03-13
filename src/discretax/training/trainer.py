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
    create_run_directory,
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


def _classification_metrics(log_probs: jax.Array, targets: jax.Array) -> tuple[jax.Array, jax.Array]:
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
        metrics = _eval_step(model, state, batch_inputs, batch_targets, jr.PRNGKey(seed + batch_index))
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


def run_experiment(experiment_config: ExperimentConfig) -> RunResult:
    """Execute a configured experiment end to end."""
    dataset_bundle = build_dataset(experiment_config.paths, experiment_config.dataset)
    output_dir = create_run_directory(experiment_config)
    tracker = create_tracker(experiment_config, output_dir)
    write_resolved_config(experiment_config, output_dir)

    master_key = jr.PRNGKey(experiment_config.trainer.seed)
    model_key, loop_key = jr.split(master_key)
    model = build_model(experiment_config, dataset_bundle, model_key)
    state = eqx.nn.State(model)

    total_steps = experiment_config.trainer.max_steps
    if total_steps is None:
        total_steps = experiment_config.trainer.num_epochs * count_batches(
            dataset_bundle.train,
            experiment_config.loader.batch_size,
            drop_last=experiment_config.loader.drop_last_train,
        )

    optimizer, learning_rate_schedule = _build_optimizer(experiment_config.optimizer, total_steps)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = _make_train_step(optimizer)

    tracker.summary(
        {
            "dataset_name": dataset_bundle.name,
            "input_dim": dataset_bundle.input_dim,
            "sequence_length": dataset_bundle.sequence_length,
            "num_classes": dataset_bundle.num_classes,
            "parameter_count": count_params(model),
        }
    )

    best_metric = float("inf") if experiment_config.checkpoint.mode == "min" else float("-inf")
    best_model = model
    best_state = state
    final_step = 0

    validation_split_name = "validation" if len(dataset_bundle.validation) > 0 else "test"

    for epoch in range(experiment_config.trainer.num_epochs):
        epoch_seed = experiment_config.trainer.seed + epoch
        for batch_inputs, batch_targets in batch_iterator(
            dataset_bundle.train,
            experiment_config.loader.batch_size,
            shuffle=experiment_config.loader.shuffle_train,
            drop_last=experiment_config.loader.drop_last_train,
            seed=epoch_seed,
        ):
            if final_step >= total_steps:
                break

            loop_key, step_key = jr.split(loop_key)
            model, state, opt_state, train_metrics = train_step(
                model,
                state,
                opt_state,
                batch_inputs,
                batch_targets,
                step_key,
            )
            final_step += 1

            step_metrics = {
                "epoch": epoch,
                "step": final_step,
                "learning_rate": float(learning_rate_schedule(final_step - 1)),
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
            }

            if final_step % experiment_config.trainer.log_every_steps == 0:
                append_history(output_dir, step_metrics)
                tracker.log(step_metrics, step=final_step)

            if final_step % experiment_config.trainer.eval_every_steps == 0 or final_step == total_steps:
                evaluation_metrics = _evaluate(
                    model,
                    state,
                    dataset_bundle,
                    experiment_config,
                    split_name=validation_split_name,
                    seed=experiment_config.trainer.seed + final_step,
                )
                evaluation_metrics["step"] = final_step
                append_history(output_dir, evaluation_metrics)
                tracker.log(evaluation_metrics, step=final_step)

                monitored_value = evaluation_metrics[
                    f"{validation_split_name}_{experiment_config.checkpoint.monitor.removeprefix('val_')}"
                ]
                if _is_improved(
                    monitored_value,
                    best_metric,
                    experiment_config.checkpoint.mode,
                ):
                    best_metric = monitored_value
                    best_model = model
                    best_state = state
                    if experiment_config.checkpoint.enabled and experiment_config.checkpoint.save_best:
                        save_checkpoint(
                            output_dir,
                            "best",
                            model=best_model,
                            state=best_state,
                            opt_state=opt_state,
                            metadata={
                                "step": final_step,
                                "monitor": experiment_config.checkpoint.monitor,
                                "monitor_value": monitored_value,
                            },
                        )

            if final_step % experiment_config.trainer.checkpoint_every_steps == 0:
                if experiment_config.checkpoint.enabled:
                    save_checkpoint(
                        output_dir,
                        f"step-{final_step}",
                        model=model,
                        state=state,
                        opt_state=opt_state,
                        metadata={"step": final_step},
                    )

        if final_step >= total_steps:
            break

    test_metrics = _evaluate(
        best_model,
        best_state,
        dataset_bundle,
        experiment_config,
        split_name="test",
        seed=experiment_config.trainer.seed + final_step + 1,
    )
    tracker.summary(
        {
            "best_metric": best_metric,
            **test_metrics,
            "final_step": final_step,
        }
    )
    write_summary(
        output_dir,
        {
            "best_metric": best_metric,
            "final_step": final_step,
            **test_metrics,
        },
    )
    tracker.finish()

    return RunResult(
        output_dir=output_dir,
        best_metric=best_metric,
        final_step=final_step,
        test_loss=test_metrics["test_loss"],
        test_accuracy=test_metrics["test_accuracy"],
    )
