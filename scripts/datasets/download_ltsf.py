"""Download Long-Term Time Series Forecasting benchmark CSVs.

Mirrors the canonical Autoformer/Informer benchmark CSVs from the
`thuml/Time-Series-Library` Hugging Face dataset. These are the files
used by Autoformer, FEDformer, PatchTST, iTransformer, etc.

Note: the `Weather` dataset on this branch uses the older Informer "WTH"
(12 features, 35,064 rows) preserved as preprocessed pickles under
`data/processed/Weather/`. This script intentionally does *not* fetch
the modern 21-feature `weather.csv` so as not to clobber that lineage.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"

DATASET_PATHS: dict[str, str] = {
    "ETTh1": "ETT-small/ETTh1.csv",
    "ETTh2": "ETT-small/ETTh2.csv",
    "ETTm1": "ETT-small/ETTm1.csv",
    "ETTm2": "ETT-small/ETTm2.csv",
    "Traffic": "traffic/traffic.csv",
    "Electricity": "electricity/electricity.csv",
    "Exchange": "exchange_rate/exchange_rate.csv",
    "ILI": "illness/national_illness.csv",
}


def download_dataset(
    dataset_name: str,
    *,
    output_dir: Path,
    force: bool,
    dry_run: bool,
) -> Path:
    """Download one LTSF dataset CSV into the configured root."""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(
            f"Unsupported LTSF dataset: {dataset_name!r}. Choices: {sorted(DATASET_PATHS)}"
        )

    relative_path = Path(DATASET_PATHS[dataset_name])
    url = f"{BASE_URL}/{relative_path.as_posix()}"
    destination = output_dir / relative_path.name

    if dry_run:
        print(f"[dry-run] {dataset_name}: {url} -> {destination}")
        return destination

    output_dir.mkdir(parents=True, exist_ok=True)
    if force and destination.exists():
        destination.unlink()
    if destination.exists():
        print(f"{dataset_name}: already present at {destination}")
        return destination

    print(f"{dataset_name}: downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"{dataset_name}: saved to {destination}")
    return destination


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Download Long-Term Time Series Forecasting benchmark CSVs."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_PATHS) + ["all"],
        help="Dataset to download. Pass multiple times or use 'all'. Defaults to 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "LTSF",
        help="Root directory where the benchmark CSVs will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload datasets even if the local file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing any files.",
    )
    args = parser.parse_args()

    requested = args.dataset or ["all"]
    if "all" in requested:
        requested = sorted(DATASET_PATHS)

    for dataset_name in requested:
        download_dataset(
            dataset_name,
            output_dir=args.output_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
