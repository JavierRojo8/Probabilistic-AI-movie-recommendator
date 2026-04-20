import os
import zipfile
import requests

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SRC_DIR, "..", "data")
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-1m.zip")
    extract_path = os.path.join(DATA_DIR, "ml-1m")

    if not os.path.exists(zip_path):
        print(f"Downloading from {URL} ...")
        resp = requests.get(URL, stream=True, timeout=30)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Saved to {zip_path}")
    else:
        print(f"Already downloaded: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"{zip_path} doesn't look like a valid zip. Delete it and try again.")

    if not os.path.exists(extract_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"  Extracted to {extract_path}")
    else:
        print(f"Already extracted: {extract_path}")

if __name__ == "__main__":
    download_movielens()
