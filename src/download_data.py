"""
Downloads the selected garbage classification dataset from Kaggle.

Prerequisites:
1. Create a Kaggle account -> https://www.kaggle.com
2. Go to Account settings -> Create New API Token -> downloads kaggle.json
3. Place kaggle.json at ~/.kaggle/kaggle.json  (chmod 600 on Linux/Mac)
   or set env vars KAGGLE_USERNAME / KAGGLE_KEY

Usage:
    python download_data.py                # downloads DATASET_VARIANT from config.py
    python download_data.py --variant 12class
    python download_data.py --all           # downloads all 3 datasets
"""

import argparse
import os
import zipfile
from config import DATASETS, get_dataset_config


def download_dataset(variant: str):
    cfg = get_dataset_config(variant)
    raw_dir = cfg["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    print(f"\n[{variant}] Downloading '{cfg['name']}' -> {raw_dir}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as e:
        raise SystemExit(
            "Kaggle API credentials not found. Place kaggle.json in ~/.kaggle/ "
            "or set KAGGLE_USERNAME and KAGGLE_KEY env vars.\n" + str(e)
        )

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(cfg["kaggle_slug"], path=raw_dir, unzip=False, quiet=False)

    # Unzip
    for f in os.listdir(raw_dir):
        if f.endswith(".zip"):
            zip_path = os.path.join(raw_dir, f)
            print(f"Extracting {zip_path} ...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(raw_dir)
            os.remove(zip_path)

    print(f"[{variant}] Done. Files at: {raw_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(DATASETS.keys()), default=None,
                         help="Which dataset variant to download (default: config.DATASET_VARIANT)")
    parser.add_argument("--all", action="store_true", help="Download all 3 dataset variants")
    args = parser.parse_args()

    if args.all:
        for v in DATASETS:
            download_dataset(v)
    else:
        download_dataset(args.variant)