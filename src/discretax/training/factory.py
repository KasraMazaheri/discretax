"""Factories for dataset-aware experiment model construction."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from discretax.datasets import DatasetBundle
from discretax.training.config import ComponentConfig, ExperimentConfig, PrecisionConfig
from discretax.training.tasks import resolve_task_config
from discretax.utils import resolve_target
from discretax.utils.config_mixin import Partial


def _build_partial(component_config: ComponentConfig) -> Partial:
    """Build a partial module from a component config."""
    target_cls = resolve_target(component_config.target)
    return Partial(target_cls, **component_config.kwargs)


def _resolve_dtype(name: str) -> jnp.dtype:
    """Resolve a configured dtype name to a JAX dtype."""
    supported = {
        "float32": jnp.dtype(jnp.float32),
        "float16": jnp.dtype(jnp.float16),
        "bfloat16": jnp.dtype(jnp.bfloat16),
    }
    return supported[name]


def _resolve_compute_dtype(model_name: str, precision_config: PrecisionConfig) -> jnp.dtype:
    """Resolve the compute dtype for a model build."""
    if precision_config.mode == "float32":
        return _resolve_dtype("float32")
    if model_name != "linoss":
        raise ValueError("Mixed precision is currently supported only for LinOSS models")
    if precision_config.mode == "bfloat16_mixed":
        return _resolve_dtype("bfloat16")
    return _resolve_dtype("float16")


def _with_dtype(component_config: ComponentConfig, dtype: jnp.dtype) -> ComponentConfig:
    """Return a component config with an explicit dtype override."""
    return ComponentConfig(
        target=component_config.target,
        kwargs={**component_config.kwargs, "dtype": dtype},
    )


def build_model(
    experiment_config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    key: PRNGKeyArray,
) -> eqx.nn.StatefulLayer:
    """Build an encoder-backbone-head model stack for a dataset."""
    encoder_key, backbone_key, head_key = jr.split(key, 3)
    task_config = resolve_task_config(dataset_bundle, experiment_config)
    compute_dtype = _resolve_compute_dtype(
        experiment_config.model.name,
        experiment_config.precision,
    )
    encoder_config = _with_dtype(experiment_config.model.encoder, compute_dtype)
    backbone_config = _with_dtype(experiment_config.model.backbone, compute_dtype)
    head_config = _with_dtype(experiment_config.model.head, compute_dtype)

    encoder_kwargs = {
        "in_features": dataset_bundle.input_dim,
        "out_features": experiment_config.model.hidden_dim,
        "key": encoder_key,
        "sequence_length": dataset_bundle.sequence_length,
    }
    image_shape = dataset_bundle.metadata.get("image_shape")
    if image_shape is not None:
        encoder_kwargs["image_shape"] = tuple(image_shape)
        encoder_kwargs["sequence_layout"] = dataset_bundle.metadata.get("sequence_layout", "rows")
    encoder_kwargs["value_dim"] = dataset_bundle.metadata.get("value_dim")
    encoder_kwargs["num_time_features"] = dataset_bundle.metadata.get("num_time_features", 0)
    encoder = _build_partial(encoder_config).resolve(**encoder_kwargs)
    head_input_length = int(getattr(encoder, "output_length", dataset_bundle.sequence_length))
    backbone = _build_partial(backbone_config).resolve(
        hidden_dim=experiment_config.model.hidden_dim,
        key=backbone_key,
    )
    head = _build_partial(head_config).resolve(
        in_features=experiment_config.model.hidden_dim,
        out_features=task_config.head_out_features,
        key=head_key,
        **{
            **task_config.head_kwargs,
            "input_length": head_input_length,
        },
    )
    return eqx.nn.Sequential([encoder, backbone, head])
