"""
Loads and preprocesses the MovieLens 1M dataset.

Outputs a pickled dict with:
  - train / val / test: np.array (N, 3) of [user_idx, item_idx, rating]
  - cold_test / warm_test: same, split by user training-set size
  - n_users, n_items, global_mean
  - user2idx, movie2idx: original ID -> contiguous index mappings
  - movies: DataFrame with [movie_id, title, genres, item_idx]
  - user_train_counts: dict {user_idx: n_train_ratings}
"""

import os
import pickle
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-1m")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

COLD_START_THRESHOLD = 20  # users with <= this many training ratings are "cold"


def load_ratings() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ratings.dat")
    return pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def load_movies() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "movies.dat")
    df = pd.read_csv(
        path, sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return df


def prepare_data(val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> dict:
    print("Loading ratings...")
    ratings = load_ratings()
    movies = load_movies()

    # Remap IDs to contiguous integer indices
    user_ids_sorted = sorted(ratings["user_id"].unique())
    movie_ids_sorted = sorted(ratings["movie_id"].unique())
    user2idx = {uid: idx for idx, uid in enumerate(user_ids_sorted)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids_sorted)}

    ratings["user_idx"] = ratings["user_id"].map(user2idx).astype(np.int32)
    ratings["item_idx"] = ratings["movie_id"].map(movie2idx).astype(np.int32)

    n_users = len(user2idx)
    n_items = len(movie2idx)
    print(f"  {n_users} users, {n_items} items, {len(ratings)} ratings")

    # Per-user temporal split: keep last (test+val) ratings out of training
    ratings = ratings.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    train_rows, val_rows, test_rows = [], [], []

    for _, group in ratings.groupby("user_idx", sort=False):
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            train_rows.append(group)
            continue

        train_rows.append(group.iloc[:n_train])
        val_rows.append(group.iloc[n_train : n_train + n_val])
        test_rows.append(group.iloc[n_train + n_val :])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    cols = ["user_idx", "item_idx", "rating"]
    train = train_df[cols].values.astype(np.float32)
    val = val_df[cols].values.astype(np.float32)
    test = test_df[cols].values.astype(np.float32)

    # Cold-start split based on training-set size
    user_train_counts = train_df.groupby("user_idx").size().to_dict()
    cold_users = {u for u, c in user_train_counts.items() if c <= COLD_START_THRESHOLD}
    warm_users = set(user_train_counts) - cold_users

    cold_mask = np.isin(test[:, 0].astype(int), list(cold_users))
    cold_test = test[cold_mask]
    warm_test = test[~cold_mask]

    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"  Cold users (≤{COLD_START_THRESHOLD} train ratings): {len(cold_users)}")
    print(f"  Cold test ratings: {len(cold_test)}, Warm: {len(warm_test)}")

    # Attach item index to movies DataFrame
    movies["item_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["item_idx"]).copy()
    movies["item_idx"] = movies["item_idx"].astype(int)

    data = {
        "train": train,
        "val": val,
        "test": test,
        "cold_test": cold_test,
        "warm_test": warm_test,
        "n_users": n_users,
        "n_items": n_items,
        "global_mean": float(train_df["rating"].mean()),
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "movies": movies,
        "cold_users": list(cold_users),
        "user_train_counts": user_train_counts,
    }
    return data


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data()
    out_path = os.path.join(OUT_DIR, "data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_path}")
