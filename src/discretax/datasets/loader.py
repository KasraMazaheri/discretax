"""NumPy-backed batching utilities for experiment datasets."""

from __future__ import annotations

from collections.abc import Iterator

import jax.numpy as jnp
import numpy as np

from discretax.datasets.base import DatasetSplit


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
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    stop = len(indices) if not drop_last else len(indices) - (len(indices) % batch_size)
    for start in range(0, stop, batch_size):
        batch_indices = indices[start : start + batch_size]
        if len(batch_indices) < batch_size and drop_last:
            continue
        yield (
            jnp.asarray(dataset_split.inputs[batch_indices]),
            jnp.asarray(dataset_split.targets[batch_indices]),
        )
