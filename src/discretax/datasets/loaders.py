"""Dataset registry and concrete dataset loaders."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from discretax.datasets.base import DatasetBundle, DatasetSplit
from discretax.training.config import DatasetConfig, PathsConfig


def resolve_dataset_path(paths_config: PathsConfig, dataset_config: DatasetConfig) -> Path:
    """Resolve the on-disk dataset path from global and dataset-local roots."""
    data_root = Path(paths_config.data_root)
    if dataset_config.root is None:
        return data_root

    dataset_root = Path(dataset_config.root)
    if dataset_root.is_absolute():
        return dataset_root
    return data_root / dataset_root


def _split_train_validation(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_split: float,
    seed: int,
) -> tuple[DatasetSplit, DatasetSplit]:
    """Split a training set into train and validation splits."""
    if validation_split == 0.0:
        empty_inputs = np.empty((0, *train_inputs.shape[1:]), dtype=train_inputs.dtype)
        empty_targets = np.empty((0,), dtype=train_targets.dtype)
        return DatasetSplit(train_inputs, train_targets), DatasetSplit(empty_inputs, empty_targets)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(train_inputs.shape[0])
    split_idx = int(train_inputs.shape[0] * (1.0 - validation_split))
    train_idx = permutation[:split_idx]
    validation_idx = permutation[split_idx:]
    return (
        DatasetSplit(train_inputs[train_idx], train_targets[train_idx]),
        DatasetSplit(train_inputs[validation_idx], train_targets[validation_idx]),
    )


def _standardize_with_train_stats(
    train_split: DatasetSplit,
    validation_split: DatasetSplit,
    test_split: DatasetSplit,
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Standardize all splits using training-set statistics."""
    mean = train_split.inputs.mean(axis=(0, 1), keepdims=True)
    std = train_split.inputs.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def _normalize(split: DatasetSplit) -> DatasetSplit:
        return DatasetSplit(
            inputs=((split.inputs - mean) / std).astype(np.float32),
            targets=split.targets.astype(np.int32),
        )

    return _normalize(train_split), _normalize(validation_split), _normalize(test_split)


def _prepare_image_sequences(images: np.ndarray, layout: str) -> np.ndarray:
    """Convert image tensors into sequence-major arrays."""
    images = images.astype(np.float32)
    if images.ndim == 3:
        images = images[..., None]

    num_examples, height, width, channels = images.shape
    if layout == "rows":
        return images.reshape(num_examples, height, width * channels)
    if layout == "pixels":
        return images.reshape(num_examples, height * width, channels)
    raise ValueError(f"Unsupported image sequence layout: {layout}")


def _build_mnist_dataset(
    paths_config: PathsConfig, dataset_config: DatasetConfig
) -> DatasetBundle:
    """Build the MNIST classification dataset bundle."""
    from torchvision.datasets import MNIST

    dataset_path = resolve_dataset_path(paths_config, dataset_config)
    train_dataset = MNIST(root=dataset_path, train=True, download=dataset_config.download)
    test_dataset = MNIST(root=dataset_path, train=False, download=dataset_config.download)

    train_inputs = _prepare_image_sequences(
        np.asarray(train_dataset.data), dataset_config.sequence_layout
    )
    test_inputs = _prepare_image_sequences(
        np.asarray(test_dataset.data), dataset_config.sequence_layout
    )
    if dataset_config.normalize:
        train_inputs /= 255.0
        test_inputs /= 255.0

    train_targets = np.asarray(train_dataset.targets, dtype=np.int32)
    test_targets = np.asarray(test_dataset.targets, dtype=np.int32)
    train_split, validation_split = _split_train_validation(
        train_inputs,
        train_targets,
        dataset_config.validation_split,
        dataset_config.seed,
    )
    test_split = DatasetSplit(test_inputs.astype(np.float32), test_targets)

    return DatasetBundle(
        name=dataset_config.resolved_name,
        task="classification",
        train=train_split,
        validation=validation_split,
        test=test_split,
        input_dim=int(train_split.inputs.shape[-1]),
        num_classes=10,
        sequence_length=int(train_split.inputs.shape[1]),
    )


def _build_cifar10_dataset(
    paths_config: PathsConfig, dataset_config: DatasetConfig
) -> DatasetBundle:
    """Build the CIFAR-10 classification dataset bundle."""
    from torchvision.datasets import CIFAR10

    dataset_path = resolve_dataset_path(paths_config, dataset_config)
    train_dataset = CIFAR10(root=dataset_path, train=True, download=dataset_config.download)
    test_dataset = CIFAR10(root=dataset_path, train=False, download=dataset_config.download)

    train_inputs = _prepare_image_sequences(
        np.asarray(train_dataset.data), dataset_config.sequence_layout
    )
    test_inputs = _prepare_image_sequences(
        np.asarray(test_dataset.data), dataset_config.sequence_layout
    )
    if dataset_config.normalize:
        train_inputs /= 255.0
        test_inputs /= 255.0

    train_targets = np.asarray(train_dataset.targets, dtype=np.int32)
    test_targets = np.asarray(test_dataset.targets, dtype=np.int32)
    train_split, validation_split = _split_train_validation(
        train_inputs,
        train_targets,
        dataset_config.validation_split,
        dataset_config.seed,
    )
    test_split = DatasetSplit(test_inputs.astype(np.float32), test_targets)

    return DatasetBundle(
        name=dataset_config.resolved_name,
        task="classification",
        train=train_split,
        validation=validation_split,
        test=test_split,
        input_dim=int(train_split.inputs.shape[-1]),
        num_classes=10,
        sequence_length=int(train_split.inputs.shape[1]),
    )


def _load_pickle(path: Path) -> np.ndarray:
    """Load a pickle file and convert it to a NumPy array."""
    with path.open("rb") as file:
        return np.asarray(pickle.load(file))


def _resolve_uea_directory(dataset_path: Path, dataset_name: str) -> Path:
    """Resolve a preprocessed UEA dataset directory."""
    candidates = [
        dataset_path / dataset_name,
        dataset_path / "UEA" / dataset_name,
        dataset_path / "processed" / "UEA" / dataset_name,
        dataset_path.parent / "processed" / "UEA" / dataset_name,
    ]
    if dataset_path.name == "UEA":
        candidates.extend(
            [
                dataset_path.parent / "processed" / "UEA" / dataset_name,
                dataset_path.parent / dataset_name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a preprocessed UEA dataset directory "
        f"for {dataset_name!r} under {dataset_path}"
    )


def _labels_to_int(labels: np.ndarray) -> np.ndarray:
    """Convert integer or one-hot labels into class indices."""
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels.astype(np.int32)
    if labels.ndim == 2:
        return np.argmax(labels, axis=-1).astype(np.int32)
    raise ValueError(f"Unsupported UEA label shape: {labels.shape}")


def _maybe_add_time_channel(inputs: np.ndarray) -> np.ndarray:
    """Prepend a normalized time channel to time-series inputs."""
    num_examples, sequence_length, _ = inputs.shape
    time_channel = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    time_channel = np.broadcast_to(time_channel[None, :, None], (num_examples, sequence_length, 1))
    return np.concatenate([time_channel, inputs], axis=-1)


def _build_uea_dataset(paths_config: PathsConfig, dataset_config: DatasetConfig) -> DatasetBundle:
    """Build a UEA classification dataset bundle from preprocessed pickle files."""
    dataset_root = _resolve_uea_directory(
        resolve_dataset_path(paths_config, dataset_config),
        dataset_config.resolved_name,
    )

    train_inputs = _load_pickle(dataset_root / "X_train.pkl").astype(np.float32)
    validation_inputs = _load_pickle(dataset_root / "X_val.pkl").astype(np.float32)
    test_inputs = _load_pickle(dataset_root / "X_test.pkl").astype(np.float32)
    train_targets = _labels_to_int(_load_pickle(dataset_root / "y_train.pkl"))
    validation_targets = _labels_to_int(_load_pickle(dataset_root / "y_val.pkl"))
    test_targets = _labels_to_int(_load_pickle(dataset_root / "y_test.pkl"))

    if dataset_config.params.get("include_time", False):
        train_inputs = _maybe_add_time_channel(train_inputs)
        validation_inputs = _maybe_add_time_channel(validation_inputs)
        test_inputs = _maybe_add_time_channel(test_inputs)

    train_split = DatasetSplit(train_inputs, train_targets)
    validation_split = DatasetSplit(validation_inputs, validation_targets)
    test_split = DatasetSplit(test_inputs, test_targets)

    if dataset_config.normalize:
        train_split, validation_split, test_split = _standardize_with_train_stats(
            train_split,
            validation_split,
            test_split,
        )

    num_classes = int(
        max(
            train_targets.max(initial=0),
            validation_targets.max(initial=0),
            test_targets.max(initial=0),
        )
        + 1
    )
    return DatasetBundle(
        name=dataset_config.resolved_name,
        task="classification",
        train=train_split,
        validation=validation_split,
        test=test_split,
        input_dim=int(train_split.inputs.shape[-1]),
        num_classes=num_classes,
        sequence_length=int(train_split.inputs.shape[1]),
    )


def build_dataset(paths_config: PathsConfig, dataset_config: DatasetConfig) -> DatasetBundle:
    """Build a dataset bundle from the config registry."""
    if dataset_config.kind == "mnist":
        return _build_mnist_dataset(paths_config, dataset_config)
    if dataset_config.kind == "cifar10":
        return _build_cifar10_dataset(paths_config, dataset_config)
    if dataset_config.kind == "uea":
        return _build_uea_dataset(paths_config, dataset_config)
    raise ValueError(f"Unsupported dataset kind: {dataset_config.kind}")
