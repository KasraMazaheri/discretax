"""Dataset registry and concrete dataset loaders."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from discretax.datasets.base import DatasetBundle, DatasetSplit
from discretax.datasets.batching import (
    CIFAR10_CHANNEL_MEAN,
    CIFAR10_CHANNEL_STD,
    pack_image_sequences,
)
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


# ---------------------------------------------------------------------------
# Image datasets (MNIST, CIFAR-10)
# ---------------------------------------------------------------------------


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
        sequence_length=int(train_split.inputs.shape[1]),
        metadata={
            "image_shape": image_shape,
            "sequence_layout": dataset_config.sequence_layout,
        },
    )


# ---------------------------------------------------------------------------
# Time-series datasets (UEA, PPG)
# ---------------------------------------------------------------------------


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


def _add_time_channel(inputs: np.ndarray) -> np.ndarray:
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

    data = _load_pickle(dataset_root / "data.pkl").astype(np.float32)
    labels = _labels_to_int(_load_pickle(dataset_root / "labels.pkl"))

    rng = np.random.default_rng(dataset_config.seed)
    perm = rng.permutation(len(data))
    n = len(data)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_inputs = data[perm[:n_train]]
    train_targets = labels[perm[:n_train]]
    validation_inputs = data[perm[n_train : n_train + n_val]]
    validation_targets = labels[perm[n_train : n_train + n_val]]
    test_inputs = data[perm[n_train + n_val :]]
    test_targets = labels[perm[n_train + n_val :]]

    if dataset_config.params.get("include_time", False):
        train_inputs = _add_time_channel(train_inputs)
        validation_inputs = _add_time_channel(validation_inputs)
        test_inputs = _add_time_channel(test_inputs)

    train_split = DatasetSplit(train_inputs, train_targets)
    validation_split = DatasetSplit(validation_inputs, validation_targets)
    test_split = DatasetSplit(test_inputs, test_targets)

    if dataset_config.normalize:
        train_split, validation_split, test_split = _standardize_with_train_stats(
            train_split,
            validation_split,
            test_split,
        )

    output_dim = int(
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
        output_dim=output_dim,
        sequence_length=int(train_split.inputs.shape[1]),
    )


def _build_ppg_dataset(paths_config: PathsConfig, dataset_config: DatasetConfig) -> DatasetBundle:
    """Build the PPG-DaLiA regression dataset bundle from preprocessed pickle files."""
    dataset_root = resolve_dataset_path(paths_config, dataset_config)

    train_inputs = _load_pickle(dataset_root / "X_train.pkl").astype(np.float32)
    validation_inputs = _load_pickle(dataset_root / "X_val.pkl").astype(np.float32)
    test_inputs = _load_pickle(dataset_root / "X_test.pkl").astype(np.float32)
    # Targets from process_ppg.py are (N, T_target); expand to (N, T_target, 1)
    # so they align with the head output shape (T_target, out_features=1).
    train_targets = _load_pickle(dataset_root / "y_train.pkl").astype(np.float32)[:, :, np.newaxis]
    validation_targets = _load_pickle(dataset_root / "y_val.pkl").astype(np.float32)[
        :, :, np.newaxis
    ]
    test_targets = _load_pickle(dataset_root / "y_test.pkl").astype(np.float32)[:, :, np.newaxis]

    if dataset_config.params.get("include_time", False):
        train_inputs = _add_time_channel(train_inputs)
        validation_inputs = _add_time_channel(validation_inputs)
        test_inputs = _add_time_channel(test_inputs)

    return DatasetBundle(
        name=dataset_config.resolved_name,
        task="regression",
        train=DatasetSplit(train_inputs, train_targets),
        validation=DatasetSplit(validation_inputs, validation_targets),
        test=DatasetSplit(test_inputs, test_targets),
        input_dim=int(train_inputs.shape[-1]),
        output_dim=1,
        sequence_length=int(train_inputs.shape[1]),
    )


# ---------------------------------------------------------------------------
# LTSF (long-term time-series forecasting) datasets
# ---------------------------------------------------------------------------


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
    "Traffic": "traffic.csv",
    "Electricity": "electricity.csv",
    "Exchange": "exchange_rate.csv",
    "ILI": "national_illness.csv",
}

_LTSF_SENTINEL = -1.0


class _LtsfWindowView:
    """Memory-efficient windowed view of a standardized (T, C) array.

    Mimics the subset of the np.ndarray API that the trainer's batching
    loop needs (`shape`, `dtype`, `__len__`, fancy `__getitem__`). Each
    sample is a length-`window_length` window starting at `start + i`,
    with one half replaced by a sentinel to enforce masked seq2seq.

    `mask_role`:
      - "input": real data in the first `seq_len` steps, sentinel after.
      - "target": sentinel in the first `seq_len` steps, real data after.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        start: int,
        num_windows: int,
        seq_len: int,
        pred_len: int,
        mask_role: str,
    ) -> None:
        if mask_role not in ("input", "target"):
            raise ValueError(f"Unsupported mask_role: {mask_role!r}")
        self._data = data
        self._start = int(start)
        self._num_windows = int(num_windows)
        self._seq_len = int(seq_len)
        self._pred_len = int(pred_len)
        self._mask_role = mask_role
        self._window_length = self._seq_len + self._pred_len
        self.shape = (self._num_windows, self._window_length, int(data.shape[-1]))
        self.dtype = data.dtype

    def __len__(self) -> int:
        """Return the number of windows in the split."""
        return self._num_windows

    def __getitem__(self, index: object) -> np.ndarray:
        """Materialize the requested window(s) as a contiguous array."""
        indices = np.asarray(index)
        scalar = indices.ndim == 0
        if scalar:
            indices = indices.reshape(1)
        starts = self._start + indices.astype(np.int64)
        offsets = np.arange(self._window_length, dtype=np.int64)
        gather = starts[:, None] + offsets[None, :]
        out = self._data[gather]
        if self._mask_role == "input":
            out[:, self._seq_len :, :] = _LTSF_SENTINEL
        else:
            out[:, : self._seq_len, :] = _LTSF_SENTINEL
        return out[0] if scalar else out


def _resolve_ltsf_csv_path(dataset_root: Path, dataset_name: str, params: dict) -> Path:
    """Resolve the CSV file for a long-term forecasting dataset."""
    explicit = params.get("csv_path")
    if explicit is not None:
        path = Path(explicit)
        if not path.is_absolute():
            path = dataset_root / path
        if path.exists():
            return path
        raise FileNotFoundError(f"LTSF csv_path does not exist: {path}")

    file_name = _LTSF_DEFAULT_FILES.get(dataset_name, f"{dataset_name}.csv")
    candidates = [
        dataset_root / file_name,
        dataset_root / dataset_name / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find LTSF CSV for {dataset_name!r}. Looked in: "
        f"{[str(p) for p in candidates]}. "
        f"Run `uv run python scripts/datasets/download_ltsf.py --dataset {dataset_name}` "
        f"or set dataset.params.csv_path."
    )


def _resolve_ltsf_split_ranges(
    *, dataset_name: str, num_rows: int, seq_len: int
) -> dict[str, tuple[int, int]]:
    """Return contiguous (start, end) row ranges per split.

    Validation/test ranges include a `seq_len`-step lookback overlap into
    the previous split so that the first window's target lands at the
    canonical split boundary, matching the Informer/Autoformer convention.
    """
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
    test_end = num_train + num_validation + num_test
    return {
        "train": (0, train_end),
        "validation": (max(0, train_end - seq_len), validation_end),
        "test": (max(0, validation_end - seq_len), test_end),
    }


def _build_ltsf_dataset(paths_config: PathsConfig, dataset_config: DatasetConfig) -> DatasetBundle:
    """Build a long-term time-series forecasting dataset in masked seq2seq form.

    Loads a CSV from the Informer/Autoformer benchmark suite, fits a
    per-channel z-score on the train split only, and exposes stride-1
    sliding windows of length `seq_len + pred_len`. Inputs replace the
    last `pred_len` steps with the -1 sentinel; targets replace the first
    `seq_len` steps with the sentinel. The trainer's `loss_window` then
    restricts loss/metrics to the final `pred_len` steps.

    Windows are materialized lazily per batch via `_LtsfWindowView`, so
    only the small (T, C) standardized arrays are held in memory.
    """
    import pandas as pd

    dataset_root = resolve_dataset_path(paths_config, dataset_config)
    dataset_name = dataset_config.resolved_name
    params = dataset_config.params

    seq_len = int(params.get("seq_len", 720))
    pred_len = int(params.get("pred_len", 720))
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("LTSF seq_len and pred_len must be positive")

    csv_path = _resolve_ltsf_csv_path(dataset_root, dataset_name, params)
    dataframe = pd.read_csv(csv_path)
    if "date" in dataframe.columns:
        dataframe = dataframe.drop(columns=["date"])
    feature_columns = [c for c in dataframe.columns if c != "date"]
    data = dataframe[feature_columns].to_numpy(dtype=np.float64)

    split_ranges = _resolve_ltsf_split_ranges(
        dataset_name=dataset_name,
        num_rows=int(data.shape[0]),
        seq_len=seq_len,
    )

    train_start, train_end = split_ranges["train"]
    train_slice = data[train_start:train_end]
    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    standardized = ((data - mean) / std).astype(np.float32)

    window_length = seq_len + pred_len

    def _make_split(split_name: str) -> DatasetSplit:
        start, end = split_ranges[split_name]
        num_windows = end - start - window_length + 1
        if num_windows <= 0:
            raise ValueError(
                f"LTSF dataset {dataset_name!r} split {split_name!r} has "
                f"{end - start} rows; not enough for seq_len+pred_len={window_length}"
            )
        inputs = _LtsfWindowView(
            standardized,
            start=start,
            num_windows=num_windows,
            seq_len=seq_len,
            pred_len=pred_len,
            mask_role="input",
        )
        targets = _LtsfWindowView(
            standardized,
            start=start,
            num_windows=num_windows,
            seq_len=seq_len,
            pred_len=pred_len,
            mask_role="target",
        )
        return DatasetSplit(inputs, targets)

    train_split = _make_split("train")
    validation_split = _make_split("validation")
    test_split = _make_split("test")

    return DatasetBundle(
        name=dataset_name,
        task="regression",
        train=train_split,
        validation=validation_split,
        test=test_split,
        input_dim=int(standardized.shape[-1]),
        output_dim=int(standardized.shape[-1]),
        sequence_length=window_length,
        metadata={
            "loss_window": pred_len,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "csv_path": str(csv_path),
            "feature_columns": feature_columns,
            "channel_mean": mean.squeeze(0).astype(np.float32).tolist(),
            "channel_std": std.squeeze(0).astype(np.float32).tolist(),
        },
    )


def _build_weather_dataset(
    paths_config: PathsConfig, dataset_config: DatasetConfig
) -> DatasetBundle:
    """Build the Weather long-horizon forecasting dataset.

    Inputs and targets are both (N, 1440, 12). The first 720 timesteps of the
    input are real context and the last 720 are a -1 sentinel; the targets
    invert this. `metadata["loss_window"] = 720` tells the trainer to compute
    loss/metrics only over the last 720 timesteps.
    """
    dataset_root = resolve_dataset_path(paths_config, dataset_config)

    train_inputs = _load_pickle(dataset_root / "X_train.pkl").astype(np.float32)
    validation_inputs = _load_pickle(dataset_root / "X_val.pkl").astype(np.float32)
    test_inputs = _load_pickle(dataset_root / "X_test.pkl").astype(np.float32)
    train_targets = _load_pickle(dataset_root / "y_train.pkl").astype(np.float32)
    validation_targets = _load_pickle(dataset_root / "y_val.pkl").astype(np.float32)
    test_targets = _load_pickle(dataset_root / "y_test.pkl").astype(np.float32)

    if dataset_config.params.get("include_time", False):
        train_inputs = _add_time_channel(train_inputs)
        validation_inputs = _add_time_channel(validation_inputs)
        test_inputs = _add_time_channel(test_inputs)

    horizon = int(dataset_config.params.get("loss_window", 720))

    return DatasetBundle(
        name=dataset_config.resolved_name,
        task="regression",
        train=DatasetSplit(train_inputs, train_targets),
        validation=DatasetSplit(validation_inputs, validation_targets),
        test=DatasetSplit(test_inputs, test_targets),
        input_dim=int(train_inputs.shape[-1]),
        output_dim=int(train_targets.shape[-1]),
        sequence_length=int(train_inputs.shape[1]),
        metadata={"loss_window": horizon},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_dataset(paths_config: PathsConfig, dataset_config: DatasetConfig) -> DatasetBundle:
    """Build a dataset bundle from the config registry."""
    if dataset_config.kind == "mnist":
        return _build_mnist_dataset(paths_config, dataset_config)
    if dataset_config.kind == "cifar10":
        return _build_cifar10_dataset(paths_config, dataset_config)
    if dataset_config.kind == "uea":
        return _build_uea_dataset(paths_config, dataset_config)
    if dataset_config.kind == "ppg":
        return _build_ppg_dataset(paths_config, dataset_config)
    if dataset_config.kind == "weather":
        return _build_weather_dataset(paths_config, dataset_config)
    if dataset_config.kind == "ltsf":
        return _build_ltsf_dataset(paths_config, dataset_config)
    raise ValueError(f"Unsupported dataset kind: {dataset_config.kind}")
