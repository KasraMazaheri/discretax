"""Dataset abstractions shared across experiment loaders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DatasetSplit:
    """A concrete dataset split held fully in memory."""

    inputs: np.ndarray
    targets: np.ndarray

    def __post_init__(self) -> None:
        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError("Dataset inputs and targets must have the same leading dimension")

    def __len__(self) -> int:
        """Return the number of examples in the split."""
        return int(self.inputs.shape[0])


@dataclass(slots=True)
class DatasetBundle:
    """A normalized dataset bundle for training and evaluation."""

    name: str
    task: str
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    input_dim: int
    num_classes: int
    sequence_length: int

