"""Tests for packaged dataset loaders."""

import pickle
import sys
import types
from pathlib import Path

import numpy as np

from discretax.datasets import DatasetSplit, batch_iterator, build_dataset
from discretax.datasets.images import pack_image_sequences
from discretax.training import DatasetConfig, PathsConfig


def _install_fake_torchvision(
    monkeypatch, *, mnist_data, mnist_targets, cifar_data, cifar_targets
):
    """Install a fake torchvision module for dataset loader tests."""

    class DummyMNIST:
        def __init__(self, root, train, download):
            del root, download
            self.data = mnist_data if train else mnist_data[:2]
            self.targets = mnist_targets if train else mnist_targets[:2]

    class DummyCIFAR10:
        def __init__(self, root, train, download):
            del root, download
            self.data = cifar_data if train else cifar_data[:2]
            self.targets = cifar_targets if train else cifar_targets[:2]

    datasets_module = types.ModuleType("torchvision.datasets")
    datasets_module.MNIST = DummyMNIST
    datasets_module.CIFAR10 = DummyCIFAR10
    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.datasets = datasets_module

    monkeypatch.setitem(sys.modules, "torchvision", torchvision_module)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", datasets_module)


def test_build_mnist_dataset_with_fake_torchvision(tmp_path: Path, monkeypatch):
    """MNIST loader reshapes image rows into sequences."""
    mnist_data = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    mnist_targets = np.array([0, 1, 2, 3], dtype=np.int32)
    cifar_data = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    cifar_targets = np.array([0, 1, 2, 3], dtype=np.int32)
    _install_fake_torchvision(
        monkeypatch,
        mnist_data=mnist_data,
        mnist_targets=mnist_targets,
        cifar_data=cifar_data,
        cifar_targets=cifar_targets,
    )

    dataset_bundle = build_dataset(
        PathsConfig(data_root=str(tmp_path)),
        DatasetConfig(kind="mnist", validation_split=0.25, normalize=False),
    )

    assert dataset_bundle.sequence_length == 2
    assert dataset_bundle.input_dim == 2
    assert len(dataset_bundle.train) == 3
    assert len(dataset_bundle.validation) == 1
    assert len(dataset_bundle.test) == 2


def test_build_cifar10_dataset_with_fake_torchvision(tmp_path: Path, monkeypatch):
    """CIFAR-10 loader flattens each row into width-times-channel features."""
    mnist_data = np.zeros((4, 2, 2), dtype=np.uint8)
    mnist_targets = np.array([0, 1, 2, 3], dtype=np.int32)
    cifar_data = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
    cifar_targets = np.array([0, 1, 2, 3], dtype=np.int32)
    _install_fake_torchvision(
        monkeypatch,
        mnist_data=mnist_data,
        mnist_targets=mnist_targets,
        cifar_data=cifar_data,
        cifar_targets=cifar_targets,
    )

    dataset_bundle = build_dataset(
        PathsConfig(data_root=str(tmp_path)),
        DatasetConfig(kind="cifar10", validation_split=0.25, normalize=False),
    )

    assert dataset_bundle.sequence_length == 2
    assert dataset_bundle.input_dim == 6
    assert len(dataset_bundle.train) == 3
    assert len(dataset_bundle.validation) == 1
    assert len(dataset_bundle.test) == 2


def test_build_cifar10_dataset_attaches_image_preprocessing_metadata(tmp_path: Path, monkeypatch):
    """CIFAR-10 loader keeps image preprocessing settings in split metadata."""
    mnist_data = np.zeros((4, 2, 2), dtype=np.uint8)
    mnist_targets = np.array([0, 1, 2, 3], dtype=np.int32)
    cifar_data = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
    cifar_targets = np.array([0, 1, 2, 3], dtype=np.int32)
    _install_fake_torchvision(
        monkeypatch,
        mnist_data=mnist_data,
        mnist_targets=mnist_targets,
        cifar_data=cifar_data,
        cifar_targets=cifar_targets,
    )

    dataset_bundle = build_dataset(
        PathsConfig(data_root=str(tmp_path)),
        DatasetConfig(
            kind="cifar10",
            validation_split=0.25,
            normalize=True,
            params={
                "normalization": "cifar10_channelwise",
                "augmentations": {"horizontal_flip": {"prob": 0.5}},
            },
        ),
    )

    assert dataset_bundle.metadata["image_shape"] == (2, 2, 3)
    assert dataset_bundle.train.metadata["rescale"] == 1.0 / 255.0
    assert "augmentations" in dataset_bundle.train.metadata
    assert "augmentations" not in dataset_bundle.validation.metadata
    np.testing.assert_allclose(
        np.asarray(dataset_bundle.validation.metadata["mean"], dtype=np.float32),
        np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32),
    )


def test_batch_iterator_applies_horizontal_flip_from_split_metadata():
    """Batch iteration applies image augmentations declared on a dataset split."""
    image = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]],
        dtype=np.float32,
    )
    split = DatasetSplit(
        inputs=pack_image_sequences(image, "rows"),
        targets=np.array([0], dtype=np.int32),
        metadata={
            "image_shape": (2, 2, 3),
            "sequence_layout": "rows",
            "augmentations": {"horizontal_flip": {"prob": 1.0}},
        },
    )

    batch_inputs, batch_targets = next(
        batch_iterator(split, batch_size=1, shuffle=False, drop_last=False, seed=0)
    )

    expected = np.array(
        [[[4.0, 5.0, 6.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 7.0, 8.0, 9.0]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(batch_inputs), expected)
    np.testing.assert_array_equal(np.asarray(batch_targets), np.array([0], dtype=np.int32))


def test_batch_iterator_applies_channelwise_normalization_from_split_metadata():
    """Batch iteration applies image normalization declared on a dataset split."""
    image = np.array([[[[255.0, 128.0, 0.0]]]], dtype=np.float32)
    split = DatasetSplit(
        inputs=pack_image_sequences(image, "pixels"),
        targets=np.array([1], dtype=np.int32),
        metadata={
            "image_shape": (1, 1, 3),
            "sequence_layout": "pixels",
            "rescale": 1.0 / 255.0,
            "mean": [0.5, 0.25, 0.0],
            "std": [0.5, 0.25, 1.0],
        },
    )

    batch_inputs, _ = next(
        batch_iterator(split, batch_size=1, shuffle=False, drop_last=False, seed=0)
    )

    expected = np.array([[[1.0, 1.0078431, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(batch_inputs), expected, rtol=1e-6, atol=1e-6)


def test_build_uea_dataset_from_preprocessed_pickles(tmp_path: Path):
    """UEA loader reads the sibling-repo preprocessed split format."""
    dataset_dir = tmp_path / "processed" / "UEA" / "TinyUEA"
    dataset_dir.mkdir(parents=True)

    train_inputs = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    validation_inputs = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    test_inputs = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    train_targets = np.array([0, 1, 0, 1], dtype=np.int32)
    validation_targets = np.array([0, 1], dtype=np.int32)
    test_targets = np.array([1, 0], dtype=np.int32)

    for name, value in {
        "X_train.pkl": train_inputs,
        "X_val.pkl": validation_inputs,
        "X_test.pkl": test_inputs,
        "y_train.pkl": train_targets,
        "y_val.pkl": validation_targets,
        "y_test.pkl": test_targets,
    }.items():
        with (dataset_dir / name).open("wb") as file:
            pickle.dump(value, file)

    dataset_bundle = build_dataset(
        PathsConfig(data_root=str(tmp_path)),
        DatasetConfig(kind="uea", name="TinyUEA", root=".", normalize=True),
    )

    assert dataset_bundle.sequence_length == 3
    assert dataset_bundle.input_dim == 2
    assert dataset_bundle.output_dim == 2
    assert len(dataset_bundle.train) == 4
    assert np.isfinite(dataset_bundle.train.inputs).all()


def test_build_uea_dataset_resolves_processed_alias_from_data_root(tmp_path: Path):
    """UEA loader resolves `data/UEA`-style aliases to the processed layout."""
    dataset_dir = tmp_path / "processed" / "UEA" / "TinyUEA"
    dataset_dir.mkdir(parents=True)

    payloads = {
        "X_train.pkl": np.zeros((2, 3, 1), dtype=np.float32),
        "X_val.pkl": np.zeros((1, 3, 1), dtype=np.float32),
        "X_test.pkl": np.zeros((1, 3, 1), dtype=np.float32),
        "y_train.pkl": np.array([0, 1], dtype=np.int32),
        "y_val.pkl": np.array([0], dtype=np.int32),
        "y_test.pkl": np.array([1], dtype=np.int32),
    }
    for name, value in payloads.items():
        with (dataset_dir / name).open("wb") as file:
            pickle.dump(value, file)

    dataset_bundle = build_dataset(
        PathsConfig(data_root=str(tmp_path)),
        DatasetConfig(kind="uea", name="TinyUEA", root="UEA", normalize=False),
    )

    assert dataset_bundle.name == "TinyUEA"
    assert dataset_bundle.sequence_length == 3
