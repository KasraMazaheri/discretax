"""Convert the raw PPG-DaLiA dataset into processed train/val/test pickles.

This is a faithful port of the preprocessing pipeline from damped-linoss
(`damped_linoss/scripts/process_ppg.py`). The numerical operations — signal
resampling via `np.repeat`, per-channel min-max normalization, target padding,
split-variant selection, and sliding-window construction — are preserved
exactly so that the processed tensors are functionally identical.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

REPO_ROOT = Path(__file__).resolve().parents[2]

NUM_SUBJECTS = 15
INPUT_WINDOW = 49920
INPUT_STRIDE = 4992
OUTPUT_WINDOW = 390
OUTPUT_STRIDE = 39


def save_pickle(path: Path, value: object) -> None:
    """Persist a Python object as a pickle."""
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def minmax_to_unit(array: np.ndarray) -> np.ndarray:
    """Min-max scale an array to the [-1, 1] range (as in damped-linoss)."""
    return 2 * (array - np.min(array)) / (np.max(array) - np.min(array)) - 1


def split_subject(
    input_array: np.ndarray,
    output_array: np.ndarray,
    variant: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select one of the six temporal 70/15/15 split variants for a subject."""
    n_in = len(input_array)
    n_out = len(output_array)
    if variant == 0:
        train_input = input_array[: int(0.7 * n_in)]
        train_output = output_array[: int(0.7 * n_out)]
        val_input = input_array[int(0.7 * n_in) : int(0.85 * n_in)]
        val_output = output_array[int(0.7 * n_out) : int(0.85 * n_out)]
        test_input = input_array[int(0.85 * n_in) :]
        test_output = output_array[int(0.85 * n_out) :]
    elif variant == 1:
        train_input = input_array[: int(0.7 * n_in)]
        train_output = output_array[: int(0.7 * n_out)]
        val_input = input_array[int(0.85 * n_in) :]
        val_output = output_array[int(0.85 * n_out) :]
        test_input = input_array[int(0.7 * n_in) : int(0.85 * n_in)]
        test_output = output_array[int(0.7 * n_out) : int(0.85 * n_out)]
    elif variant == 2:
        train_input = input_array[int(0.15 * n_in) : int(0.85 * n_in)]
        train_output = output_array[int(0.15 * n_out) : int(0.85 * n_out)]
        val_input = input_array[: int(0.15 * n_in)]
        val_output = output_array[: int(0.15 * n_out)]
        test_input = input_array[int(0.85 * n_in) :]
        test_output = output_array[int(0.85 * n_out) :]
    elif variant == 3:
        train_input = input_array[int(0.15 * n_in) : int(0.85 * n_in)]
        train_output = output_array[int(0.15 * n_out) : int(0.85 * n_out)]
        val_input = input_array[int(0.85 * n_in) :]
        val_output = output_array[int(0.85 * n_out) :]
        test_input = input_array[: int(0.15 * n_in)]
        test_output = output_array[: int(0.15 * n_out)]
    elif variant == 4:
        train_input = input_array[int(0.30 * n_in) :]
        train_output = output_array[int(0.30 * n_out) :]
        val_input = input_array[: int(0.15 * n_in)]
        val_output = output_array[: int(0.15 * n_out)]
        test_input = input_array[int(0.15 * n_in) : int(0.30 * n_in)]
        test_output = output_array[int(0.15 * n_out) : int(0.30 * n_out)]
    elif variant == 5:
        train_input = input_array[int(0.30 * n_in) :]
        train_output = output_array[int(0.30 * n_out) :]
        val_input = input_array[int(0.15 * n_in) : int(0.30 * n_in)]
        val_output = output_array[int(0.15 * n_out) : int(0.30 * n_out)]
        test_input = input_array[: int(0.15 * n_in)]
        test_output = output_array[: int(0.15 * n_out)]
    else:
        raise ValueError(f"Unsupported split variant: {variant}")
    return (
        train_input,
        val_input,
        test_input,
        train_output,
        val_output,
        test_output,
    )


def process_subject(
    subject_index: int,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one subject's pickle and return windowed train/val/test tensors."""
    subject_file = raw_dir / f"S{subject_index}" / f"S{subject_index}.pkl"
    with subject_file.open("rb") as file:
        data = pickle.load(file, encoding="latin1")

    acc = np.repeat(data["signal"]["wrist"]["ACC"], 2, axis=0)
    bvp = data["signal"]["wrist"]["BVP"]
    eda = np.repeat(data["signal"]["wrist"]["EDA"], 16, axis=0)
    temp = np.repeat(data["signal"]["wrist"]["TEMP"], 16, axis=0)

    acc = minmax_to_unit(acc)
    bvp = minmax_to_unit(bvp)
    eda = minmax_to_unit(eda)
    temp = minmax_to_unit(temp)

    input_array = np.concatenate([acc, bvp, eda, temp], axis=1)
    output_array = data["label"]
    output_array = np.concatenate(
        [[output_array[0]], [output_array[0]], [output_array[0]], output_array],
        axis=0,
    )
    output_array = minmax_to_unit(output_array)

    variant = random.randint(0, 5)
    (
        train_input,
        val_input,
        test_input,
        train_output,
        val_output,
        test_output,
    ) = split_subject(input_array, output_array, variant)

    train_input = np.swapaxes(swv(train_input, INPUT_WINDOW, 0)[::INPUT_STRIDE], 1, 2)
    val_input = np.swapaxes(swv(val_input, INPUT_WINDOW, 0)[::INPUT_STRIDE], 1, 2)
    test_input = np.swapaxes(swv(test_input, INPUT_WINDOW, 0)[::INPUT_STRIDE], 1, 2)

    train_output = swv(train_output, OUTPUT_WINDOW, 0)[::OUTPUT_STRIDE]
    val_output = swv(val_output, OUTPUT_WINDOW, 0)[::OUTPUT_STRIDE]
    test_output = swv(test_output, OUTPUT_WINDOW, 0)[::OUTPUT_STRIDE]

    return (
        train_input,
        val_input,
        test_input,
        train_output,
        val_output,
        test_output,
    )


def process_dataset(raw_dir: Path, output_dir: Path) -> None:
    """Process every subject and save the concatenated train/val/test pickles."""
    all_train_input: list[np.ndarray] = []
    all_val_input: list[np.ndarray] = []
    all_test_input: list[np.ndarray] = []
    all_train_output: list[np.ndarray] = []
    all_val_output: list[np.ndarray] = []
    all_test_output: list[np.ndarray] = []

    for subject_index in range(1, NUM_SUBJECTS + 1):
        print(subject_index)
        (
            train_input,
            val_input,
            test_input,
            train_output,
            val_output,
            test_output,
        ) = process_subject(subject_index, raw_dir)

        all_train_input.append(train_input)
        all_val_input.append(val_input)
        all_test_input.append(test_input)
        all_train_output.append(train_output)
        all_val_output.append(val_output)
        all_test_output.append(test_output)

    train_input = np.concatenate(all_train_input, axis=0)
    val_input = np.concatenate(all_val_input, axis=0)
    test_input = np.concatenate(all_test_input, axis=0)
    train_output = np.concatenate(all_train_output, axis=0)
    val_output = np.concatenate(all_val_output, axis=0)
    test_output = np.concatenate(all_test_output, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(output_dir / "X_train.pkl", train_input)
    save_pickle(output_dir / "y_train.pkl", train_output)
    save_pickle(output_dir / "X_val.pkl", val_input)
    save_pickle(output_dir / "y_val.pkl", val_output)
    save_pickle(output_dir / "X_test.pkl", test_input)
    save_pickle(output_dir / "y_test.pkl", test_output)


def main() -> int:
    """CLI entrypoint for processing the PPG-DaLiA dataset."""
    parser = argparse.ArgumentParser(
        description="Process raw PPG-DaLiA subject pickles into train/val/test tensors.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "PPG_FieldStudy",
        help="Directory containing S{i}/S{i}.pkl subject files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "PPG" / "ppg",
        help="Directory where processed pickles will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for the per-subject split-variant selection.",
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw PPG directory does not exist: {args.raw_dir}")

    if args.seed is not None:
        random.seed(args.seed)

    process_dataset(args.raw_dir, args.output_dir)

    print(f"raw_dir: {args.raw_dir}")
    print(f"processed_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
