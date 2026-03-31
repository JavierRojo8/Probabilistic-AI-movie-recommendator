""" 

    File to download the movielens 1m dataset

"""


import os
import zipfile
import requests

DATA_DIR = "data"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-1m.zip")

    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        r = requests.get(URL)
        with open(zip_path, "wb") as f:
            f.write(r.content)

    extract_path = os.path.join(DATA_DIR, "ml-1m")
    if not os.path.exists(extract_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

download_movielens()