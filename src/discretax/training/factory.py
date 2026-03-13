"""Factories for dataset-aware experiment model construction."""

from __future__ import annotations

import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray

from discretax.datasets import DatasetBundle
from discretax.training.config import ComponentConfig, ExperimentConfig
from discretax.utils import resolve_target
from discretax.utils.config_mixin import Partial


def _build_partial(component_config: ComponentConfig) -> Partial:
    """Build a partial module from a component config."""
    target_cls = resolve_target(component_config.target)
    return Partial(target_cls, **component_config.kwargs)


def build_model(
    experiment_config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    key: PRNGKeyArray,
) -> eqx.nn.Sequential:
    """Build an encoder-backbone-head model stack for a dataset."""
    encoder_key, backbone_key, head_key = jr.split(key, 3)

    encoder = _build_partial(experiment_config.model.encoder).resolve(
        in_features=dataset_bundle.input_dim,
        out_features=experiment_config.model.hidden_dim,
        key=encoder_key,
    )
    backbone = _build_partial(experiment_config.model.backbone).resolve(
        hidden_dim=experiment_config.model.hidden_dim,
        key=backbone_key,
    )
    head = _build_partial(experiment_config.model.head).resolve(
        in_features=experiment_config.model.hidden_dim,
        out_features=dataset_bundle.num_classes,
        key=head_key,
    )
    return eqx.nn.Sequential([encoder, backbone, head])
