"""
Loads and preprocesses the MovieLens 1M dataset.

Outputs a pickled dict with:
  - train / val / test : np.array (N, 3) of [user_idx, item_idx, rating]
  - low_history_test   : test rows for users with few training ratings
  - high_history_test  : test rows for users with many training ratings
  - n_users, n_items, global_mean
  - user2idx, movie2idx : original ID -> contiguous index mappings
  - movies             : DataFrame with [movie_id, title, genres, item_idx]
  - user_train_counts  : dict {user_idx: n_train_ratings}
  - low_history_users  : list of user indices with <= LOW_HISTORY_THRESHOLD ratings

NOTE on "low-history" split
---------------------------
This is NOT a true cold-start evaluation.  All users in low_history_test have
at least some ratings in the training set (the temporal split guarantees that
every user contributes at least one row to train).  "Low-history" means the
model has seen fewer training examples for these users, making predictions
harder — but they are not genuinely unseen at inference time.
A true cold-start evaluation would require holding out entire users from
training, which is a different experimental protocol.
"""

import os
import pickle

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-1m")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Users with <= this many training ratings are considered "low-history"
LOW_HISTORY_THRESHOLD = 20


def load_ratings() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ratings.dat")
    return pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def load_movies() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "movies.dat")
    return pd.read_csv(
        path, sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )


def prepare_data(val_ratio: float = 0.1, test_ratio: float = 0.1) -> dict:
    """Load, split, and index the MovieLens 1M dataset.

    The split is purely temporal (per user, last ratings go to val/test),
    so there is no random component and no seed is needed.

    Args:
        val_ratio:  fraction of each user's ratings reserved for validation.
        test_ratio: fraction of each user's ratings reserved for testing.

    Returns:
        dict with all processed arrays, mappings, and metadata.
    """
    print("Loading ratings...")
    ratings = load_ratings()
    movies  = load_movies()

    # ---- Remap IDs to contiguous integer indices -------------------------
    user_ids_sorted  = sorted(ratings["user_id"].unique())
    movie_ids_sorted = sorted(ratings["movie_id"].unique())
    user2idx  = {uid: idx for idx, uid in enumerate(user_ids_sorted)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids_sorted)}

    ratings["user_idx"] = ratings["user_id"].map(user2idx).astype(np.int32)
    ratings["item_idx"] = ratings["movie_id"].map(movie2idx).astype(np.int32)

    n_users = len(user2idx)
    n_items = len(movie2idx)
    print(f"  {n_users} users, {n_items} items, {len(ratings)} ratings")

    # Basic integrity checks
    assert ratings["user_idx"].notna().all(), "Some user IDs failed to map"
    assert ratings["item_idx"].notna().all(), "Some item IDs failed to map"
    assert ratings["user_idx"].between(0, n_users - 1).all(), "user_idx out of range"
    assert ratings["item_idx"].between(0, n_items - 1).all(), "item_idx out of range"

    # ---- Per-user temporal split ----------------------------------------
    ratings = ratings.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    train_rows, val_rows, test_rows = [], [], []

    for _, group in ratings.groupby("user_idx", sort=False):
        n       = len(group)
        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Edge case: too few ratings to form all three splits; keep in train
            train_rows.append(group)
            continue

        train_rows.append(group.iloc[:n_train])
        val_rows.append(group.iloc[n_train : n_train + n_val])
        test_rows.append(group.iloc[n_train + n_val :])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df   = pd.concat(val_rows,   ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True)

    cols  = ["user_idx", "item_idx", "rating"]
    train = train_df[cols].values.astype(np.float32)
    val   = val_df[cols].values.astype(np.float32)
    test  = test_df[cols].values.astype(np.float32)

    # Post-split integrity checks
    assert train.shape[1] == 3 and val.shape[1] == 3 and test.shape[1] == 3
    assert len(train) > 0, "Training set is empty"
    assert len(val)   > 0, "Validation set is empty"
    assert len(test)  > 0, "Test set is empty"
    assert train[:, 2].min() >= 1.0 and train[:, 2].max() <= 5.0, \
        "Ratings outside [1, 5] in training set"

    # ---- Low-history / high-history split --------------------------------
    # See module docstring for an explanation of why this is NOT cold-start.
    user_train_counts = train_df.groupby("user_idx").size().to_dict()
    low_history_users  = {u for u, c in user_train_counts.items() if c <= LOW_HISTORY_THRESHOLD}
    high_history_users = set(user_train_counts) - low_history_users

    low_mask          = np.isin(test[:, 0].astype(int), list(low_history_users))
    low_history_test  = test[low_mask]
    high_history_test = test[~low_mask]

    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"  Low-history users (≤{LOW_HISTORY_THRESHOLD} train ratings): {len(low_history_users)}")
    print(f"  Low-history test: {len(low_history_test)}, High-history test: {len(high_history_test)}")

    # ---- Attach item index to movies DataFrame ---------------------------
    movies["item_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["item_idx"]).copy()
    movies["item_idx"] = movies["item_idx"].astype(int)

    return {
        "train":              train,
        "val":                val,
        "test":               test,
        "low_history_test":   low_history_test,
        "high_history_test":  high_history_test,
        "n_users":            n_users,
        "n_items":            n_items,
        "global_mean":        float(train_df["rating"].mean()),
        "user2idx":           user2idx,
        "movie2idx":          movie2idx,
        "movies":             movies,
        "low_history_users":  list(low_history_users),
        "user_train_counts":  user_train_counts,
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    data     = prepare_data()
    out_path = os.path.join(OUT_DIR, "data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_path}")
