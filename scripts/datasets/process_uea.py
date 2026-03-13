"""Convert raw UEA ARFF datasets into processed train/val/test pickles."""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_from_arff_to_dataframe
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SplitArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def resolve_raw_dir(raw_root: Path) -> Path:
    """Resolve the extracted UEA raw directory across archive naming variants."""
    if raw_root.exists():
        return raw_root

    parent = raw_root.parent
    for candidate in (parent / "Multivariate2018_arff", parent / "Multivariate_arff"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Raw UEA directory does not exist: {raw_root}")


def save_pickle(path: Path, value: object) -> None:
    """Persist a Python object as a pickle."""
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def dataframe_to_numpy(dataframe: pd.DataFrame) -> np.ndarray:
    """Convert a UEA dataframe row into a dense NumPy tensor."""
    expanded = dataframe.map(lambda series: series.values).values
    return np.stack([np.vstack(row).T for row in expanded]).astype(np.float32)


def load_uea_split(train_file: Path, test_file: Path) -> SplitArrays:
    """Load a raw UEA dataset from ARFF files."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        train_frame, train_labels = load_from_arff_to_dataframe(str(train_file))
        test_frame, test_labels = load_from_arff_to_dataframe(str(test_file))

    train_inputs = dataframe_to_numpy(train_frame)
    test_inputs = dataframe_to_numpy(test_frame)

    label_encoder = LabelEncoder().fit(train_labels)
    train_targets = label_encoder.transform(train_labels).astype(np.int32)
    test_targets = label_encoder.transform(test_labels).astype(np.int32)
    return train_inputs, test_inputs, train_targets, test_targets


def create_validation_split(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the training set into train and validation partitions."""
    if validation_fraction <= 0.0:
        empty_inputs = np.empty((0, *train_inputs.shape[1:]), dtype=train_inputs.dtype)
        empty_targets = np.empty((0,), dtype=train_targets.dtype)
        return train_inputs, empty_inputs, train_targets, empty_targets

    train_x, val_x, train_y, val_y = train_test_split(
        train_inputs,
        train_targets,
        test_size=validation_fraction,
        random_state=seed,
        stratify=train_targets,
    )
    return train_x, val_x, train_y, val_y


def process_dataset(
    dataset_dir: Path,
    *,
    output_root: Path,
    validation_fraction: float,
    seed: int,
) -> None:
    """Process a single UEA dataset directory."""
    dataset_name = dataset_dir.name
    train_file = dataset_dir / f"{dataset_name}_TRAIN.arff"
    test_file = dataset_dir / f"{dataset_name}_TEST.arff"
    if not train_file.exists() or not test_file.exists():
        return

    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_inputs, test_inputs, train_targets, test_targets = load_uea_split(train_file, test_file)
    train_x, val_x, train_y, val_y = create_validation_split(
        train_inputs,
        train_targets,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    combined_inputs = np.concatenate([train_inputs, test_inputs], axis=0)
    combined_targets = np.concatenate([train_targets, test_targets], axis=0)
    original_indices = (
        np.arange(train_inputs.shape[0], dtype=np.int32),
        np.arange(train_inputs.shape[0], combined_inputs.shape[0], dtype=np.int32),
    )

    save_pickle(output_dir / "X_train.pkl", train_x)
    save_pickle(output_dir / "y_train.pkl", train_y)
    save_pickle(output_dir / "X_val.pkl", val_x)
    save_pickle(output_dir / "y_val.pkl", val_y)
    save_pickle(output_dir / "X_test.pkl", test_inputs)
    save_pickle(output_dir / "y_test.pkl", test_targets)
    save_pickle(output_dir / "data.pkl", combined_inputs)
    save_pickle(output_dir / "labels.pkl", combined_targets)
    save_pickle(output_dir / "original_idxs.pkl", original_indices)


def main() -> int:
    """CLI entrypoint for processing the UEA archive."""
    parser = argparse.ArgumentParser(description="Process raw UEA ARFF files into pickles.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "UEA" / "Multivariate_arff",
        help="Directory containing raw dataset subdirectories with *_TRAIN.arff and *_TEST.arff.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "UEA",
        help="Directory where processed dataset pickles will be written.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Process only the named dataset. Repeat the flag to process multiple datasets.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of the training set to reserve for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for train/validation splitting.",
    )
    args = parser.parse_args()

    raw_dir = resolve_raw_dir(args.raw_dir)
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in the range [0.0, 1.0)")

    dataset_dirs = [path for path in sorted(raw_dir.iterdir()) if path.is_dir()]
    if args.dataset:
        requested = set(args.dataset)
        dataset_dirs = [path for path in dataset_dirs if path.name in requested]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_dir in tqdm(dataset_dirs, desc="Processing UEA datasets"):
        process_dataset(
            dataset_dir,
            output_root=args.output_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )

    print(f"raw_dir: {raw_dir}")
    print(f"processed_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
