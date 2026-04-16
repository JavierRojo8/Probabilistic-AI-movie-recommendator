"""
Download and extract the MovieLens 1M dataset.

Usage:
    python src/ingest_data.py
"""

import os
import zipfile

import requests

# Paths relative to this file so the script works from any working directory
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_SRC_DIR, "..", "data")
URL       = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path     = os.path.join(DATA_DIR, "ml-1m.zip")
    extract_path = os.path.join(DATA_DIR, "ml-1m")

    # ---- Download --------------------------------------------------------
    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {URL} ...")
        response = requests.get(URL, stream=True, timeout=30)
        response.raise_for_status()          # raises on 4xx / 5xx

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Saved to {zip_path}")
    else:
        print(f"Zip already present: {zip_path}")

    # ---- Validate --------------------------------------------------------
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(
            f"{zip_path} is not a valid zip file. "
            "Delete it and re-run to download again."
        )

    # ---- Extract ---------------------------------------------------------
    if not os.path.exists(extract_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"  Extracted to {extract_path}")
    else:
        print(f"Already extracted: {extract_path}")


if __name__ == "__main__":
    download_movielens()
