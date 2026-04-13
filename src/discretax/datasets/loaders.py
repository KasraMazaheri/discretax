"""Dataset registry and concrete dataset loaders."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from discretax.datasets.base import DatasetBundle, DatasetSplit
from discretax.datasets.images import (
    CIFAR10_CHANNEL_MEAN,
    CIFAR10_CHANNEL_STD,
    pack_image_sequences,
)
from discretax.training.config import DatasetConfig, PathsConfig

_LTSF_FIXED_SPLITS = {
    "ETTh1": (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24),
    "ETTh2": (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24),
    "ETTm1": (12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4),
    "ETTm2": (12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4),
}

_LTSF_DEFAULT_FILES = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "Weather": "weather.csv",
    "Traffic": "traffic.csv",
    "Electricity": "electricity.csv",
    "Exchange": "exchange_rate.csv",
    "ILI": "national_illness.csv",
}


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


def _image_split_metadata(
    *,
    image_shape: tuple[int, int, int],
    sequence_layout: str,
    rescale: float | None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    augmentations: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build image preprocessing metadata for a dataset split."""
    metadata: dict[str, object] = {
        "image_shape": image_shape,
        "sequence_layout": sequence_layout,
    }
    if rescale is not None:
        metadata["rescale"] = rescale
    if mean is not None and std is not None:
        metadata["mean"] = mean.tolist()
        metadata["std"] = std.tolist()
    if augmentations:
        metadata["augmentations"] = augmentations
    return metadata


def _build_mnist_dataset(
    paths_config: PathsConfig, dataset_config: DatasetConfig
) -> DatasetBundle:
    """Build the MNIST classification dataset bundle."""
    from torchvision.datasets import MNIST

    dataset_path = resolve_dataset_path(paths_config, dataset_config)
    train_dataset = MNIST(root=dataset_path, train=True, download=dataset_config.download)
    test_dataset = MNIST(root=dataset_path, train=False, download=dataset_config.download)

    train_inputs = pack_image_sequences(
        np.asarray(train_dataset.data), dataset_config.sequence_layout
    )
    test_inputs = pack_image_sequences(
        np.asarray(test_dataset.data), dataset_config.sequence_layout
    )

    image_shape = (int(train_dataset.data.shape[1]), int(train_dataset.data.shape[2]), 1)
    image_metadata = _image_split_metadata(
        image_shape=image_shape,
        sequence_layout=dataset_config.sequence_layout,
        rescale=1.0 / 255.0 if dataset_config.normalize else None,
    )

    train_targets = np.asarray(train_dataset.targets, dtype=np.int32)
    test_targets = np.asarray(test_dataset.targets, dtype=np.int32)
    train_split, validation_split = _split_train_validation(
        train_inputs,
        train_targets,
        dataset_config.validation_split,
        dataset_config.seed,
    )
    train_split = DatasetSplit(
        train_split.inputs.astype(np.float32),
        train_split.targets,
        image_metadata,
    )
    validation_split = DatasetSplit(
        validation_split.inputs.astype(np.float32),
        validation_split.targets,
        dict(image_metadata),
    )
    test_split = DatasetSplit(
        test_inputs.astype(np.float32),
        test_targets,
        dict(image_metadata),
    )

    return DatasetBundle(
        name=dataset_config.resolved_name,
        task="classification",
        train=train_split,
        validation=validation_split,
        test=test_split,
        input_dim=int(train_split.inputs.shape[-1]),
        output_dim=10,
        num_classes=10,
        sequence_length=int(train_split.inputs.shape[1]),
        metadata={
            "image_shape": image_shape,
            "sequence_layout": dataset_config.sequence_layout,
        },
    )


def _build_cifar10_dataset(
    paths_config: PathsConfig, dataset_config: DatasetConfig
) -> DatasetBundle:
    """Build the CIFAR-10 classification dataset bundle."""
    from torchvision.datasets import CIFAR10

    dataset_path = resolve_dataset_path(paths_config, dataset_config)
    train_dataset = CIFAR10(root=dataset_path, train=True, download=dataset_config.download)
    test_dataset = CIFAR10(root=dataset_path, train=False, download=dataset_config.download)

    train_inputs = pack_image_sequences(
        np.asarray(train_dataset.data), dataset_config.sequence_layout
    )
    test_inputs = pack_image_sequences(
        np.asarray(test_dataset.data), dataset_config.sequence_layout
    )

    image_shape = tuple(int(value) for value in train_dataset.data.shape[1:])
    train_augmentations = dataset_config.params.get("augmentations", {})
    normalization_name = dataset_config.params.get("normalization")
    normalization_stats = (
        (CIFAR10_CHANNEL_MEAN, CIFAR10_CHANNEL_STD)
        if dataset_config.normalize and normalization_name == "cifar10_channelwise"
        else (None, None)
    )
    base_metadata = _image_split_metadata(
        image_shape=image_shape,
        sequence_layout=dataset_config.sequence_layout,
        rescale=1.0 / 255.0 if dataset_config.normalize else None,
        mean=normalization_stats[0],
        std=normalization_stats[1],
    )

    train_targets = np.asarray(train_dataset.targets, dtype=np.int32)
    test_targets = np.asarray(test_dataset.targets, dtype=np.int32)
    train_split, validation_split = _split_train_validation(
        train_inputs,
        train_targets,
        dataset_config.validation_split,
        dataset_config.seed,
    )
    train_split = DatasetSplit(
        train_split.inputs.astype(np.float32),
        train_split.targets,
        {
            **base_metadata,
            **({"augmentations": train_augmentations} if train_augmentations else {}),
        },
    )
    validation_split = DatasetSplit(
        validation_split.inputs.astype(np.float32),
        validation_split.targets,
        dict(base_metadata),
    )
    test_split = DatasetSplit(
        test_inputs.astype(np.float32),
        test_targets,
        dict(base_metadata),
    )

    return DatasetBundle(
        name=dataset_config.resolved_name,
        task="classification",
        train=train_split,
        validation=validation_split,
        test=test_split,
        input_dim=int(train_split.inputs.shape[-1]),
        output_dim=10,
        num_classes=10,
        sequence_length=int(train_split.inputs.shape[1]),
        metadata={
            "image_shape": image_shape,
            "sequence_layout": dataset_config.sequence_layout,
        },
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
        output_dim=num_classes,
        num_classes=num_classes,
        sequence_length=int(train_split.inputs.shape[1]),
    )


def _resolve_ltsf_csv_path(
    dataset_path: Path, dataset_name: str, params: dict[str, object]
) -> Path:
    """Resolve the CSV file for a long-term forecasting dataset."""
    default_file_name = _LTSF_DEFAULT_FILES.get(dataset_name, f"{dataset_name}.csv")
    file_name = str(params.get("file_name", default_file_name))
    candidate_names = [file_name]
    if file_name.lower() != file_name:
        candidate_names.append(file_name.lower())
    if dataset_path.is_file():
        return dataset_path

    candidates: list[Path] = []
    for candidate_name in candidate_names:
        candidates.extend(
            [
                dataset_path / candidate_name,
                dataset_path / dataset_name / candidate_name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    recursive_matches: list[Path] = []
    for candidate_name in candidate_names:
        recursive_matches.extend(sorted(dataset_path.rglob(candidate_name)))
    recursive_matches = sorted(set(recursive_matches))
    if len(recursive_matches) == 1:
        return recursive_matches[0]
    if recursive_matches:
        raise FileNotFoundError(
            f"Found multiple candidate files for {dataset_name!r}: {recursive_matches}"
        )

    candidate_paths = [str(path) for path in candidates]
    raise FileNotFoundError(
        "Could not find the requested LTSF dataset CSV.\n"
        f"dataset_name: {dataset_name}\n"
        f"searched_root: {dataset_path}\n"
        f"candidate_files: {candidate_names}\n"
        f"candidate_paths: {candidate_paths}\n"
        "Place the benchmark CSV under the configured root, for example "
        f"{dataset_path / file_name}, or run "
        "`uv run python scripts/datasets/download_ltsf.py --dataset "
        f"{dataset_name}`."
    )


def _calendar_time_features(values) -> np.ndarray:
    """Build simple normalized calendar features from a datetime index."""
    month = (values.dt.month.to_numpy(dtype=np.float32) - 1.0) / 11.0
    day = (values.dt.day.to_numpy(dtype=np.float32) - 1.0) / 30.0
    weekday = values.dt.weekday.to_numpy(dtype=np.float32) / 6.0
    hour = values.dt.hour.to_numpy(dtype=np.float32) / 23.0
    minute = values.dt.minute.to_numpy(dtype=np.float32) / 59.0
    return np.stack([month, day, weekday, hour, minute], axis=-1).astype(np.float32)


def _resolve_ltsf_feature_arrays(
    dataframe,
    *,
    features_mode: str,
    target_column: str,
    include_time_features: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Resolve input and target arrays for an LTSF dataset."""
    value_columns = [column for column in dataframe.columns if column != "date"]
    if target_column not in value_columns:
        raise ValueError(f"LTSF target column {target_column!r} is not present in the dataset")

    all_values = dataframe[value_columns].to_numpy(dtype=np.float32)
    target_index = value_columns.index(target_column)
    if features_mode == "S":
        input_values = all_values[:, target_index : target_index + 1]
        target_values = input_values
        target_columns = [target_column]
    elif features_mode == "MS":
        input_values = all_values
        target_values = all_values[:, target_index : target_index + 1]
        target_columns = [target_column]
    elif features_mode == "M":
        input_values = all_values
        target_values = all_values
        target_columns = value_columns
    else:
        raise ValueError(f"Unsupported LTSF features_mode: {features_mode}")

    input_columns = list(value_columns if features_mode != "S" else [target_column])
    if include_time_features:
        time_features = _calendar_time_features(dataframe["date"])
        input_values = np.concatenate([input_values, time_features], axis=-1)
        input_columns.extend(["month", "day", "weekday", "hour", "minute"])

    return input_values, target_values, input_columns, target_columns


def _resolve_ltsf_split_ranges(
    *,
    dataset_name: str,
    num_rows: int,
    sequence_length: int,
) -> dict[str, tuple[int, int]]:
    """Resolve train/validation/test borders following common LTSF conventions."""
    if dataset_name in _LTSF_FIXED_SPLITS:
        num_train, num_validation, num_test = _LTSF_FIXED_SPLITS[dataset_name]
    else:
        num_train = int(num_rows * 0.7)
        num_test = int(num_rows * 0.2)
        num_validation = num_rows - num_train - num_test

    if num_train + num_validation + num_test > num_rows:
        raise ValueError(
            f"LTSF split sizes exceed dataset length for {dataset_name}: "
            f"{num_train + num_validation + num_test} > {num_rows}"
        )

    train_end = num_train
    validation_end = num_train + num_validation
    return {
        "train": (0, train_end),
        "validation": (max(0, train_end - sequence_length), validation_end),
        "test": (max(0, validation_end - sequence_length), num_rows),
    }


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a simple feature-wise standardizer."""
    mean = values.mean(axis=0, keepdims=True).astype(np.float32)
    std = values.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply feature-wise standardization."""
    return ((values - mean) / std).astype(np.float32)


def _build_ltsf_windows(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    start: int,
    end: int,
    sequence_length: int,
    prediction_length: int,
    metadata: dict[str, object] | None = None,
) -> DatasetSplit:
    """Build sliding forecasting windows for one dataset split."""
    num_examples = end - start - sequence_length - prediction_length + 1
    if num_examples <= 0:
        raise ValueError(
            "Not enough rows to build forecasting windows with "
            f"seq_len={sequence_length} and pred_len={prediction_length}"
        )

    window_inputs = np.empty(
        (num_examples, sequence_length, inputs.shape[-1]),
        dtype=np.float32,
    )
    window_targets = np.empty(
        (num_examples, prediction_length, targets.shape[-1]),
        dtype=np.float32,
    )

    for index in range(num_examples):
        offset = start + index
        window_inputs[index] = inputs[offset : offset + sequence_length]
        target_start = offset + sequence_length
        target_end = target_start + prediction_length
        window_targets[index] = targets[target_start:target_end]

    return DatasetSplit(window_inputs, window_targets, metadata or {})


def _build_ltsf_dataset(
    paths_config: PathsConfig,
    dataset_config: DatasetConfig,
) -> DatasetBundle:
    """Build a long-term time-series forecasting dataset bundle from CSV."""
    import pandas as pd

    dataset_name = dataset_config.resolved_name
    params = dataset_config.params
    sequence_length = int(params.get("seq_len", 96))
    prediction_length = int(params.get("pred_len", 96))
    features_mode = str(params.get("features_mode", "M"))
    target_column = str(params.get("target", "OT"))
    include_time_features = params.get("time_features", "none") == "calendar"

    dataset_path = resolve_dataset_path(paths_config, dataset_config)
    csv_path = _resolve_ltsf_csv_path(dataset_path, dataset_name, params)
    dataframe = pd.read_csv(csv_path)
    if "date" not in dataframe.columns:
        raise ValueError(f"LTSF dataset {csv_path} must contain a 'date' column")
    dataframe["date"] = pd.to_datetime(dataframe["date"])

    input_values, target_values, input_columns, target_columns = _resolve_ltsf_feature_arrays(
        dataframe,
        features_mode=features_mode,
        target_column=target_column,
        include_time_features=include_time_features,
    )

    split_ranges = _resolve_ltsf_split_ranges(
        dataset_name=dataset_name,
        num_rows=len(dataframe),
        sequence_length=sequence_length,
    )
    train_input_end = split_ranges["train"][1]
    train_target_end = split_ranges["train"][1]
    input_mean, input_std = _fit_standardizer(input_values[:train_input_end])
    target_mean, target_std = _fit_standardizer(target_values[:train_target_end])
    scaled_inputs = _standardize(input_values, input_mean, input_std)
    scaled_targets = _standardize(target_values, target_mean, target_std)

    bundle_metadata = {
        "dataset_path": str(csv_path),
        "features_mode": features_mode,
        "input_columns": input_columns,
        "target_columns": target_columns,
        "prediction_length": prediction_length,
        "target_mean": target_mean.squeeze(0).astype(np.float32).tolist(),
        "target_std": target_std.squeeze(0).astype(np.float32).tolist(),
        "split_ranges": {name: [start, end] for name, (start, end) in split_ranges.items()},
    }
    split_metadata = {
        "dataset_name": dataset_name,
        "features_mode": features_mode,
        "prediction_length": prediction_length,
    }

    train_split = _build_ltsf_windows(
        scaled_inputs,
        scaled_targets,
        start=split_ranges["train"][0],
        end=split_ranges["train"][1],
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        metadata={**split_metadata, "split": "train"},
    )
    validation_split = _build_ltsf_windows(
        scaled_inputs,
        scaled_targets,
        start=split_ranges["validation"][0],
        end=split_ranges["validation"][1],
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        metadata={**split_metadata, "split": "validation"},
    )
    test_split = _build_ltsf_windows(
        scaled_inputs,
        scaled_targets,
        start=split_ranges["test"][0],
        end=split_ranges["test"][1],
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        metadata={**split_metadata, "split": "test"},
    )

    return DatasetBundle(
        name=dataset_name,
        task="forecasting",
        train=train_split,
        validation=validation_split,
        test=test_split,
        input_dim=int(train_split.inputs.shape[-1]),
        output_dim=int(train_split.targets.shape[-1]),
        sequence_length=sequence_length,
        metadata=bundle_metadata,
    )


def build_dataset(paths_config: PathsConfig, dataset_config: DatasetConfig) -> DatasetBundle:
    """Build a dataset bundle from the config registry."""
    if dataset_config.kind == "mnist":
        return _build_mnist_dataset(paths_config, dataset_config)
    if dataset_config.kind == "cifar10":
        return _build_cifar10_dataset(paths_config, dataset_config)
    if dataset_config.kind == "ltsf":
        return _build_ltsf_dataset(paths_config, dataset_config)
    if dataset_config.kind == "uea":
        return _build_uea_dataset(paths_config, dataset_config)
    raise ValueError(f"Unsupported dataset kind: {dataset_config.kind}")
