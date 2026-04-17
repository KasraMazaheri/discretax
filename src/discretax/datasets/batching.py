"""Image preprocessing and batch iteration for experiment datasets."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax.numpy as jnp
import numpy as np

from discretax.datasets.base import DatasetSplit

CIFAR10_CHANNEL_MEAN = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_CHANNEL_STD = np.asarray([0.2470, 0.2435, 0.2616], dtype=np.float32)


def pack_image_sequences(images: np.ndarray, layout: str) -> np.ndarray:
    """Convert image tensors into sequence-major arrays."""
    images = images.astype(np.float32, copy=False)
    if images.ndim == 3:
        images = images[..., None]

    num_examples, height, width, channels = images.shape
    if layout == "rows":
        return images.reshape(num_examples, height, width * channels)
    if layout == "pixels":
        return images.reshape(num_examples, height * width, channels)
    raise ValueError(f"Unsupported image sequence layout: {layout}")


def unpack_image_sequences(
    inputs: np.ndarray,
    *,
    image_shape: tuple[int, int, int],
    layout: str,
) -> np.ndarray:
    """Reconstruct NHWC images from sequence-major arrays."""
    height, width, channels = image_shape
    if layout == "rows":
        return inputs.reshape(inputs.shape[0], height, width, channels)
    if layout == "pixels":
        return inputs.reshape(inputs.shape[0], height, width, channels)
    raise ValueError(f"Unsupported image sequence layout: {layout}")


def _random_crop(
    images: np.ndarray,
    *,
    padding: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply independent random crops to a batch of images."""
    batch_size, height, width, _ = images.shape
    padded = np.pad(
        images,
        ((0, 0), (padding, padding), (padding, padding), (0, 0)),
        mode="constant",
    )
    max_offset = 2 * padding
    top_offsets = rng.integers(0, max_offset + 1, size=batch_size)
    left_offsets = rng.integers(0, max_offset + 1, size=batch_size)
    cropped = np.empty_like(images)
    for index, (top, left) in enumerate(zip(top_offsets, left_offsets, strict=True)):
        cropped[index] = padded[index, top : top + height, left : left + width, :]
    return cropped


def _random_horizontal_flip(
    images: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply independent random horizontal flips to a batch of images."""
    flipped = images.copy()
    should_flip = rng.random(size=images.shape[0]) < probability
    flipped[should_flip] = flipped[should_flip, :, ::-1, :]
    return flipped


def apply_image_batch_pipeline(
    batch_inputs: np.ndarray,
    metadata: dict[str, Any],
    *,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Apply dataset-layer image preprocessing to a batch."""
    image_shape = metadata.get("image_shape")
    if image_shape is None:
        return batch_inputs.astype(np.float32, copy=False)

    layout = metadata["sequence_layout"]
    images = unpack_image_sequences(
        batch_inputs,
        image_shape=tuple(image_shape),
        layout=layout,
    ).astype(np.float32, copy=False)

    rescale = metadata.get("rescale")
    if rescale is not None:
        images *= float(rescale)

    augmentations = metadata.get("augmentations", {})
    if augmentations:
        if rng is None:
            raise ValueError("Image augmentations require a random generator")
        crop_config = augmentations.get("random_crop")
        if crop_config is not None:
            padding = int(crop_config.get("padding", 0))
            if padding > 0:
                images = _random_crop(images, padding=padding, rng=rng)
        flip_config = augmentations.get("horizontal_flip")
        if flip_config is not None:
            probability = float(flip_config.get("prob", 0.5))
            if probability > 0.0:
                images = _random_horizontal_flip(images, probability=probability, rng=rng)

    mean = metadata.get("mean")
    std = metadata.get("std")
    if mean is not None and std is not None:
        mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, -1)
        std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, -1)
        images = (images - mean_array) / std_array

    return pack_image_sequences(images, layout).astype(np.float32, copy=False)


def count_batches(dataset_split: DatasetSplit, batch_size: int, *, drop_last: bool) -> int:
    """Count the number of batches yielded for a split."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    split_size = len(dataset_split)
    if drop_last:
        return split_size // batch_size
    return (split_size + batch_size - 1) // batch_size


def batch_iterator(
    dataset_split: DatasetSplit,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield JAX arrays from an in-memory dataset split."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = np.arange(len(dataset_split))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(indices)

    stop = len(indices) if not drop_last else len(indices) - (len(indices) % batch_size)
    for start in range(0, stop, batch_size):
        batch_indices = indices[start : start + batch_size]
        if len(batch_indices) < batch_size and drop_last:
            continue
        batch_inputs = np.asarray(dataset_split.inputs[batch_indices])
        if dataset_split.metadata:
            batch_inputs = apply_image_batch_pipeline(
                batch_inputs,
                dataset_split.metadata,
                rng=rng,
            )
        yield (
            jnp.asarray(batch_inputs),
            jnp.asarray(dataset_split.targets[batch_indices]),
        )
