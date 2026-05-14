"""Exponential moving average helpers for experiment training."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax


def update_ema_model(ema_model: Any, model: Any, *, decay: float) -> Any:
    """Update an EMA copy of a model."""
    ema_params, ema_static = eqx.partition(ema_model, eqx.is_inexact_array)
    model_params, _ = eqx.partition(model, eqx.is_inexact_array)
    ema_leaves, ema_treedef = jax.tree_util.tree_flatten(ema_params)
    model_leaves, _ = jax.tree_util.tree_flatten(model_params)
    updated_leaves = [
        decay * ema_leaf + (1.0 - decay) * model_leaf
        for ema_leaf, model_leaf in zip(ema_leaves, model_leaves, strict=True)
    ]
    updated_params = ema_treedef.unflatten(updated_leaves)
    return eqx.combine(updated_params, ema_static)
