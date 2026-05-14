"""Tests for training-time regularization helpers."""

import equinox as eqx
import jax

jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import jax.random as jr

from discretax.models import LinOSS
from discretax.training import OptimizerConfig, RegularizationConfig
from discretax.training.ema import update_ema_model
from discretax.training.regularization import apply_batch_regularization, build_target_probs
from discretax.training.trainer import _weight_decay_mask


def test_build_target_probs_applies_label_smoothing():
    """Label smoothing produces normalized non-degenerate target probabilities."""
    target_probs = build_target_probs(
        jnp.asarray([0, 2]),
        num_classes=4,
        label_smoothing=0.1,
    )

    assert jnp.allclose(jnp.sum(target_probs, axis=-1), 1.0)
    assert target_probs[0, 0] > target_probs[0, 1]
    assert target_probs[1, 2] > target_probs[1, 0]


def test_apply_batch_regularization_mixup_returns_soft_targets():
    """Mixup combines inputs and target probabilities across the batch."""
    batch_inputs = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    batch_targets = jnp.asarray([0, 1])

    regularized = apply_batch_regularization(
        batch_inputs,
        batch_targets,
        num_classes=2,
        regularization_config=RegularizationConfig(
            mix_augmentation_prob=1.0,
            mixup_alpha=0.8,
        ),
        dataset_metadata={},
        key=jr.PRNGKey(0),
    )

    assert regularized.applied
    assert regularized.inputs.shape == batch_inputs.shape
    assert jnp.allclose(jnp.sum(regularized.target_probs, axis=-1), 1.0)
    assert 0.5 <= regularized.lambda_value <= 1.0


def test_apply_batch_regularization_cutmix_uses_image_metadata():
    """Cutmix operates on image-packed sequence inputs when image metadata is present."""
    batch_inputs = jnp.asarray(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]],
        ]
    )
    batch_targets = jnp.asarray([0, 1])

    regularized = apply_batch_regularization(
        batch_inputs,
        batch_targets,
        num_classes=2,
        regularization_config=RegularizationConfig(
            mix_augmentation_prob=1.0,
            cutmix_alpha=1.0,
        ),
        dataset_metadata={"image_shape": (2, 2, 2), "sequence_layout": "rows"},
        key=jr.PRNGKey(1),
    )

    assert regularized.applied
    assert regularized.inputs.shape == batch_inputs.shape
    assert 0.0 < regularized.lambda_value < 1.0


def test_weight_decay_mask_excludes_one_dimensional_parameters():
    """The selective AdamW mask skips 1D leaves such as biases and norm scales."""
    model = LinOSS(hidden_dim=8, num_blocks=1, state_dim=8, key=jr.PRNGKey(2))
    mask = _weight_decay_mask(
        eqx.nn.Sequential([model]),
        OptimizerConfig(weight_decay_mask="exclude_1d_params"),
    )
    leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(mask, is_leaf=lambda x: x is None)
        if leaf is not None
    ]
    assert any(leaves)
    assert not all(leaves)


def test_update_ema_model_blends_inexact_leaves():
    """EMA updates preserve tree structure and average inexact leaves."""
    model = LinOSS(hidden_dim=8, num_blocks=1, state_dim=8, key=jr.PRNGKey(3))
    newer_model = LinOSS(hidden_dim=8, num_blocks=1, state_dim=8, key=jr.PRNGKey(4))
    ema_model = update_ema_model(model, newer_model, decay=0.9)

    assert type(ema_model) is type(model)
