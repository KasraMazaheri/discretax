"""Training-time batch regularization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from discretax.training.config import RegularizationConfig


@dataclass(slots=True)
class BatchRegularizationResult:
    """Result of applying training-time batch regularization."""

    inputs: jax.Array
    hard_targets: jax.Array
    target_probs: jax.Array
    applied: bool
    lambda_value: float


def build_target_probs(
    targets: jax.Array,
    *,
    num_classes: int,
    label_smoothing: float,
) -> jax.Array:
    """Build one-hot or label-smoothed target probabilities."""
    targets = targets.astype(jnp.int32)
    target_probs = jnn.one_hot(targets, num_classes=num_classes, dtype=jnp.float32)
    if label_smoothing == 0.0:
        return target_probs

    off_value = label_smoothing / num_classes
    on_value = 1.0 - label_smoothing + off_value
    return target_probs * (on_value - off_value) + off_value


def apply_batch_regularization(
    batch_inputs: jax.Array,
    batch_targets: jax.Array,
    *,
    num_classes: int,
    regularization_config: RegularizationConfig,
    dataset_metadata: dict[str, Any],
    key: jax.Array,
) -> BatchRegularizationResult:
    """Apply mixup/cutmix and return the target probabilities to train against."""
    target_probs = build_target_probs(
        batch_targets,
        num_classes=num_classes,
        label_smoothing=regularization_config.label_smoothing,
    )
    if (
        regularization_config.mix_augmentation_prob == 0.0
        or (regularization_config.mixup_alpha == 0.0 and regularization_config.cutmix_alpha == 0.0)
        or batch_inputs.shape[0] < 2
    ):
        return BatchRegularizationResult(
            inputs=batch_inputs,
            hard_targets=batch_targets.astype(jnp.int32),
            target_probs=target_probs,
            applied=False,
            lambda_value=1.0,
        )

    apply_key, choice_key, perm_key, lambda_key, cutmix_key = jr.split(key, 5)
    should_apply = bool(
        jax.device_get(
            jr.bernoulli(
                apply_key,
                p=regularization_config.mix_augmentation_prob,
            )
        )
    )
    if not should_apply:
        return BatchRegularizationResult(
            inputs=batch_inputs,
            hard_targets=batch_targets.astype(jnp.int32),
            target_probs=target_probs,
            applied=False,
            lambda_value=1.0,
        )

    permutation = jr.permutation(perm_key, batch_inputs.shape[0])
    paired_inputs = batch_inputs[permutation]
    paired_target_probs = target_probs[permutation]
    use_cutmix = _use_cutmix(regularization_config, dataset_metadata, choice_key)

    if use_cutmix:
        mixed_inputs, lam = _apply_cutmix(
            batch_inputs,
            paired_inputs,
            dataset_metadata=dataset_metadata,
            alpha=regularization_config.cutmix_alpha,
            key=cutmix_key,
        )
    else:
        lam = _sample_mix_lambda(regularization_config.mixup_alpha, lambda_key)
        mixed_inputs = lam * batch_inputs + (1.0 - lam) * paired_inputs

    mixed_target_probs = lam * target_probs + (1.0 - lam) * paired_target_probs
    return BatchRegularizationResult(
        inputs=mixed_inputs,
        hard_targets=batch_targets.astype(jnp.int32),
        target_probs=mixed_target_probs,
        applied=True,
        lambda_value=float(lam),
    )


def _sample_mix_lambda(alpha: float, key: jax.Array) -> jax.Array:
    """Sample a symmetric mix coefficient."""
    if alpha <= 0.0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    lam = jr.beta(key, alpha, alpha).astype(jnp.float32)
    return jnp.maximum(lam, 1.0 - lam)


def _use_cutmix(
    regularization_config: RegularizationConfig,
    dataset_metadata: dict[str, Any],
    key: jax.Array,
) -> bool:
    """Decide whether to apply cutmix for a batch."""
    if regularization_config.cutmix_alpha <= 0.0:
        return False
    if "image_shape" not in dataset_metadata:
        return False
    if regularization_config.mixup_alpha <= 0.0:
        return True
    return bool(jax.device_get(jr.bernoulli(key, p=regularization_config.cutmix_switch_prob)))


def _apply_cutmix(
    batch_inputs: jax.Array,
    paired_inputs: jax.Array,
    *,
    dataset_metadata: dict[str, Any],
    alpha: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply cutmix to a batch of sequence-packed images."""
    lambda_key, center_y_key, center_x_key = jr.split(key, 3)
    lam = _sample_mix_lambda(alpha, lambda_key)
    images = _restore_images(batch_inputs, dataset_metadata=dataset_metadata)
    paired_images = _restore_images(paired_inputs, dataset_metadata=dataset_metadata)
    height, width, _ = dataset_metadata["image_shape"]

    cut_ratio = float(jnp.sqrt(1.0 - lam))
    cut_height = max(1, int(height * cut_ratio))
    cut_width = max(1, int(width * cut_ratio))
    center_y = int(jax.device_get(jr.randint(center_y_key, (), 0, height)))
    center_x = int(jax.device_get(jr.randint(center_x_key, (), 0, width)))

    y0 = max(0, center_y - cut_height // 2)
    x0 = max(0, center_x - cut_width // 2)
    y1 = min(height, y0 + cut_height)
    x1 = min(width, x0 + cut_width)

    mixed_images = images.at[:, y0:y1, x0:x1, :].set(paired_images[:, y0:y1, x0:x1, :])
    box_area = (y1 - y0) * (x1 - x0)
    adjusted_lam = jnp.asarray(1.0 - (box_area / (height * width)), dtype=jnp.float32)
    return _pack_images(mixed_images, dataset_metadata=dataset_metadata), adjusted_lam


def _restore_images(batch_inputs: jax.Array, *, dataset_metadata: dict[str, Any]) -> jax.Array:
    """Restore a batch of sequence-packed images to NHWC."""
    height, width, channels = dataset_metadata["image_shape"]
    sequence_layout = dataset_metadata["sequence_layout"]
    if sequence_layout == "rows":
        return batch_inputs.reshape(batch_inputs.shape[0], height, width, channels)
    if sequence_layout == "pixels":
        return batch_inputs.reshape(batch_inputs.shape[0], height, width, channels)
    raise ValueError(f"Unsupported image sequence layout: {sequence_layout}")


def _pack_images(images: jax.Array, *, dataset_metadata: dict[str, Any]) -> jax.Array:
    """Pack a batch of images back into the configured sequence layout."""
    batch_size, height, width, channels = images.shape
    sequence_layout = dataset_metadata["sequence_layout"]
    if sequence_layout == "rows":
        return images.reshape(batch_size, height, width * channels)
    if sequence_layout == "pixels":
        return images.reshape(batch_size, height * width, channels)
    raise ValueError(f"Unsupported image sequence layout: {sequence_layout}")
