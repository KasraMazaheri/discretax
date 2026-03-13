"""Download the UEA multivariate time-series archive."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = (
    "http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_arff.zip"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def download_uea_archive(*, output_dir: Path, url: str, force: bool) -> Path:
    """Download and extract the UEA archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "uea_multivariate2018_arff.zip"

    if force:
        if archive_path.exists():
            archive_path.unlink()
        for candidate in (output_dir / "Multivariate2018_arff", output_dir / "Multivariate_arff"):
            if candidate.exists():
                shutil.rmtree(candidate)

    if not archive_path.exists():
        print(f"Downloading UEA archive to {archive_path}")
        urllib.request.urlretrieve(url, archive_path)
    else:
        print(f"Archive already present at {archive_path}")

    extracted_dir = resolve_extracted_dir(output_dir)
    if extracted_dir is None:
        print(f"Extracting archive into {output_dir}")
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(output_dir)
        extracted_dir = resolve_extracted_dir(output_dir)
    else:
        print(f"Extracted data already present at {extracted_dir}")

    if extracted_dir is None:
        raise FileNotFoundError(f"Could not locate extracted UEA archive under {output_dir}")

    return extracted_dir


def resolve_extracted_dir(output_dir: Path) -> Path | None:
    """Resolve the extracted UEA archive directory."""
    for candidate in (output_dir / "Multivariate2018_arff", output_dir / "Multivariate_arff"):
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    """CLI entrypoint for downloading the UEA archive."""
    parser = argparse.ArgumentParser(description="Download the UEA dataset archive.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "UEA",
        help="Directory where the raw archive and extracted files will be stored.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Archive URL. Defaults to the public UEA multivariate archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract even if files already exist.",
    )
    args = parser.parse_args()

    extracted_dir = download_uea_archive(
        output_dir=args.output_dir,
        url=args.url,
        force=args.force,
    )
    print(f"extracted_dir: {extracted_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
