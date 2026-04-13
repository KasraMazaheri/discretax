"""Task-specific runtime helpers for configured experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from discretax.datasets import DatasetBundle
from discretax.training.config import ExperimentConfig
from discretax.training.regularization import (
    BatchRegularizationResult,
    apply_batch_regularization,
)


@dataclass(slots=True)
class PreparedBatch:
    """Prepared training batch for non-classification tasks."""

    inputs: jax.Array
    targets: jax.Array
    applied: bool = False
    lambda_value: float = 1.0


@dataclass(slots=True)
class TaskConfig:
    """Task-specific static configuration for model construction and logging."""

    kind: str
    head_out_features: int
    head_kwargs: dict[str, Any]
    train_metric_names: tuple[str, ...]
    eval_metric_names: tuple[str, ...]
    static_summary: dict[str, Any]


@dataclass(slots=True)
class TaskOps(TaskConfig):
    """Task-specific operations used by the shared experiment runtime."""

    prepare_train_batch: Callable[[jax.Array, jax.Array, jax.Array], Any]
    train_step: Callable[
        ..., tuple[eqx.nn.Sequential, eqx.nn.State, optax.OptState, dict[str, jax.Array]]
    ]
    eval_step: Callable[..., dict[str, jax.Array]]


def _batched_forward(
    model: eqx.nn.Sequential,
    state: eqx.nn.State,
    batch_inputs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, eqx.nn.State]:
    """Run a stateful model over a batch with a named batch axis."""
    from jax import random as jr

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
    )(batch_inputs, state, batch_keys)


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


def _forecasting_metrics(
    predictions: jax.Array,
    targets: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute forecasting MSE and MAE."""
    predictions = predictions.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    mse = jnp.mean(jnp.square(predictions - targets))
    mae = jnp.mean(jnp.abs(predictions - targets))
    return mse, mae


def resolve_task_config(
    dataset_bundle: DatasetBundle,
    experiment_config: ExperimentConfig,
) -> TaskConfig:
    """Resolve the task-specific model/output configuration."""
    if dataset_bundle.task == "classification":
        num_classes = dataset_bundle.num_classes
        if num_classes is None:
            raise ValueError("Classification datasets must define num_classes")
        return TaskConfig(
            kind="classification",
            head_out_features=num_classes,
            head_kwargs={},
            train_metric_names=("loss", "accuracy"),
            eval_metric_names=("loss", "accuracy"),
            static_summary={
                "num_classes": num_classes,
                "label_smoothing": experiment_config.regularization.label_smoothing,
                "mixup_alpha": experiment_config.regularization.mixup_alpha,
                "cutmix_alpha": experiment_config.regularization.cutmix_alpha,
            },
        )

    if dataset_bundle.task == "forecasting":
        prediction_length = int(dataset_bundle.metadata["prediction_length"])
        return TaskConfig(
            kind="forecasting",
            head_out_features=dataset_bundle.output_dim,
            head_kwargs={
                "input_length": dataset_bundle.sequence_length,
                "prediction_length": prediction_length,
            },
            train_metric_names=("loss", "mse", "mae"),
            eval_metric_names=("mse", "mae"),
            static_summary={
                "output_dim": dataset_bundle.output_dim,
                "prediction_length": prediction_length,
            },
        )

    raise ValueError(f"Unsupported task kind: {dataset_bundle.task}")


def create_task_ops(
    dataset_bundle: DatasetBundle,
    experiment_config: ExperimentConfig,
    optimizer,
) -> TaskOps:
    """Resolve the task-specific runtime operations for an experiment."""
    task_config = resolve_task_config(dataset_bundle, experiment_config)
    if task_config.kind == "classification":
        return _classification_task(task_config, dataset_bundle, experiment_config, optimizer)
    return _forecasting_task(task_config, dataset_bundle, optimizer)


def _classification_task(
    task_config: TaskConfig,
    dataset_bundle: DatasetBundle,
    experiment_config: ExperimentConfig,
    optimizer,
) -> TaskOps:
    """Build classification task ops."""
    num_classes = task_config.head_out_features

    def prepare_train_batch(
        batch_inputs: jax.Array,
        batch_targets: jax.Array,
        key: jax.Array,
    ) -> BatchRegularizationResult:
        return apply_batch_regularization(
            batch_inputs,
            batch_targets,
            num_classes=num_classes,
            regularization_config=experiment_config.regularization,
            dataset_metadata=dataset_bundle.train.metadata,
            key=key,
        )

    @eqx.filter_value_and_grad(has_aux=True)
    def _loss_fn(
        model: eqx.nn.Sequential,
        state: eqx.nn.State,
        batch_inputs: jax.Array,
        hard_targets: jax.Array,
        target_probs: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, tuple[eqx.nn.State, jax.Array]]:
        log_probs, new_state = _batched_forward(model, state, batch_inputs, key)
        loss, accuracy = _classification_metrics(
            log_probs,
            hard_targets,
            target_probs=target_probs,
        )
        return loss, (new_state, accuracy)

    @eqx.filter_jit
    def train_step(
        model: eqx.nn.Sequential,
        state: eqx.nn.State,
        opt_state: optax.OptState,
        batch_inputs: jax.Array,
        batch_targets: jax.Array,
        batch_aux_targets: jax.Array,
        key: jax.Array,
    ) -> tuple[eqx.nn.Sequential, eqx.nn.State, optax.OptState, dict[str, jax.Array]]:
        (loss, (new_state, accuracy)), grads = _loss_fn(
            model,
            state,
            batch_inputs,
            batch_targets,
            batch_aux_targets,
            key,
        )
        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params=eqx.filter(model, eqx.is_inexact_array),
        )
        new_model = eqx.apply_updates(model, updates)
        return (
            new_model,
            new_state,
            new_opt_state,
            {
                "loss": loss,
                "accuracy": accuracy,
                "grad_norm": optax.global_norm(grads),
                "update_norm": optax.global_norm(updates),
                "param_norm": optax.global_norm(eqx.filter(model, eqx.is_inexact_array)),
            },
        )

    @eqx.filter_jit
    def eval_step(
        model: eqx.nn.Sequential,
        state: eqx.nn.State,
        batch_inputs: jax.Array,
        batch_targets: jax.Array,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        inference_model = eqx.nn.inference_mode(model, value=True)
        log_probs, _ = _batched_forward(inference_model, state, batch_inputs, key)
        loss, accuracy = _classification_metrics(log_probs, batch_targets)
        return {"loss": loss, "accuracy": accuracy}

    return TaskOps(
        kind=task_config.kind,
        head_out_features=task_config.head_out_features,
        head_kwargs=task_config.head_kwargs,
        train_metric_names=task_config.train_metric_names,
        eval_metric_names=task_config.eval_metric_names,
        static_summary=task_config.static_summary,
        prepare_train_batch=prepare_train_batch,
        train_step=train_step,
        eval_step=eval_step,
    )


def _forecasting_task(
    task_config: TaskConfig,
    dataset_bundle: DatasetBundle,
    optimizer,
) -> TaskOps:
    """Build forecasting task ops."""
    target_mean = jnp.asarray(
        dataset_bundle.metadata.get("target_mean", [0.0] * dataset_bundle.output_dim),
        dtype=jnp.float32,
    )
    target_std = jnp.asarray(
        dataset_bundle.metadata.get("target_std", [1.0] * dataset_bundle.output_dim),
        dtype=jnp.float32,
    )

    def prepare_train_batch(
        batch_inputs: jax.Array,
        batch_targets: jax.Array,
        key: jax.Array,
    ) -> PreparedBatch:
        del key
        return PreparedBatch(inputs=batch_inputs, targets=batch_targets)

    @eqx.filter_value_and_grad(has_aux=True)
    def _loss_fn(
        model: eqx.nn.Sequential,
        state: eqx.nn.State,
        batch_inputs: jax.Array,
        batch_targets: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, tuple[eqx.nn.State, jax.Array]]:
        predictions, new_state = _batched_forward(model, state, batch_inputs, key)
        mse, mae = _forecasting_metrics(predictions, batch_targets)
        return mse, (new_state, mae)

    @eqx.filter_jit
    def train_step(
        model: eqx.nn.Sequential,
        state: eqx.nn.State,
        opt_state: optax.OptState,
        batch_inputs: jax.Array,
        batch_targets: jax.Array,
        batch_aux_targets: jax.Array | None,
        key: jax.Array,
    ) -> tuple[eqx.nn.Sequential, eqx.nn.State, optax.OptState, dict[str, jax.Array]]:
        del batch_aux_targets
        (loss, (new_state, mae)), grads = _loss_fn(
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
        return (
            new_model,
            new_state,
            new_opt_state,
            {
                "loss": loss,
                "mse": loss,
                "mae": mae,
                "grad_norm": optax.global_norm(grads),
                "update_norm": optax.global_norm(updates),
                "param_norm": optax.global_norm(eqx.filter(model, eqx.is_inexact_array)),
            },
        )

    @eqx.filter_jit
    def eval_step(
        model: eqx.nn.Sequential,
        state: eqx.nn.State,
        batch_inputs: jax.Array,
        batch_targets: jax.Array,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        inference_model = eqx.nn.inference_mode(model, value=True)
        predictions, _ = _batched_forward(inference_model, state, batch_inputs, key)
        # predictions = predictions.astype(jnp.float32) * target_std + target_mean
        # targets = batch_targets.astype(jnp.float32) * target_std + target_mean
        # mse, mae = _forecasting_metrics(predictions, targets)
        mse, mae = _forecasting_metrics(predictions, batch_targets)
        return {"mse": mse, "mae": mae}

    return TaskOps(
        kind=task_config.kind,
        head_out_features=task_config.head_out_features,
        head_kwargs=task_config.head_kwargs,
        train_metric_names=task_config.train_metric_names,
        eval_metric_names=task_config.eval_metric_names,
        static_summary=task_config.static_summary,
        prepare_train_batch=prepare_train_batch,
        train_step=train_step,
        eval_step=eval_step,
    )
