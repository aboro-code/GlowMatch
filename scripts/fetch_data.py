"""Fetch the Sephora Products and Skincare Reviews dataset into data/raw/.

Usage:
    python scripts/fetch_data.py

Resolution order:
    1. If data/raw/ already has the expected CSVs, do nothing.
    2. If a local cache directory is given via --cache (or the sibling
       ../beautyrag/data/raw exists), copy the CSVs from there instead of
       hitting the network.
    3. Otherwise download via the Kaggle API (requires ~/.kaggle/kaggle.json).

This keeps a clean-clone evaluator able to run one command and get a working
data/ directory, while avoiding a redundant few-hundred-MB download on a
machine that already has the dataset cached from another project.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

KAGGLE_DATASET = "nadyinky/sephora-products-and-skincare-reviews"

EXPECTED_FILES = [
    "product_info.csv",
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_SIBLING_CACHE = REPO_ROOT.parent / "beautyrag" / "data" / "raw"


def already_fetched() -> bool:
    return all((DATA_RAW / f).exists() for f in EXPECTED_FILES)


def copy_from_cache(cache_dir: Path) -> bool:
    if not all((cache_dir / f).exists() for f in EXPECTED_FILES):
        return False
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for f in EXPECTED_FILES:
        shutil.copy2(cache_dir / f, DATA_RAW / f)
    print(f"Copied dataset from local cache: {cache_dir}")
    return True


def download_from_kaggle() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "kaggle package not installed. Run `pip install kaggle` and "
            "make sure ~/.kaggle/kaggle.json is in place."
        ) from exc

    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_RAW), quiet=False)

    zip_path = next(DATA_RAW.glob("*.zip"), None)
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_RAW)
        zip_path.unlink()
    print(f"Downloaded and extracted into {DATA_RAW}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Local directory to copy the raw CSVs from instead of downloading.",
    )
    args = parser.parse_args()

    if already_fetched():
        print(f"data/raw/ already has all expected files, skipping fetch: {DATA_RAW}")
        return

    for candidate in filter(None, [args.cache, DEFAULT_SIBLING_CACHE]):
        if copy_from_cache(candidate):
            return

    download_from_kaggle()


if __name__ == "__main__":
    main()
