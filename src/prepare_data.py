import os
import pickle

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-1m")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# users with <= this many train ratings are "low-history"
LOW_HISTORY_THRESHOLD = 20

def load_ratings():
    path = os.path.join(DATA_DIR, "ratings.dat")
    return pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

def load_movies():
    path = os.path.join(DATA_DIR, "movies.dat")
    return pd.read_csv(
        path, sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

def prepare_data(val_ratio=0.1, test_ratio=0.1):
    print("Loading ratings...")
    ratings = load_ratings()
    movies = load_movies()

    user_ids = sorted(ratings["user_id"].unique())
    movie_ids = sorted(ratings["movie_id"].unique())
    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    movie2idx = {mid: i for i, mid in enumerate(movie_ids)}

    ratings["user_idx"] = ratings["user_id"].map(user2idx).astype(np.int32)
    ratings["item_idx"] = ratings["movie_id"].map(movie2idx).astype(np.int32)

    n_users = len(user2idx)
    n_items = len(movie2idx)
    print(f"  {n_users} users, {n_items} items, {len(ratings)} ratings")

    assert ratings["user_idx"].notna().all()
    assert ratings["item_idx"].notna().all()
    assert ratings["user_idx"].between(0, n_users - 1).all()
    assert ratings["item_idx"].between(0, n_items - 1).all()

    # temporal split per user — sort by timestamp then slice off the tail
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

    assert len(train) > 0 and len(val) > 0 and len(test) > 0

    user_train_counts = train_df.groupby("user_idx").size().to_dict()
    low_history_users = {u for u, c in user_train_counts.items() if c <= LOW_HISTORY_THRESHOLD}

    low_mask = np.isin(test[:, 0].astype(int), list(low_history_users))
    low_history_test = test[low_mask]
    high_history_test = test[~low_mask]

    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"  Low-history users (≤{LOW_HISTORY_THRESHOLD} train ratings): {len(low_history_users)}")

    movies["item_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["item_idx"]).copy()
    movies["item_idx"] = movies["item_idx"].astype(int)

    return {
        "train": train,
        "val": val,
        "test": test,
        "low_history_test": low_history_test,
        "high_history_test": high_history_test,
        "n_users": n_users,
        "n_items": n_items,
        "global_mean": float(train_df["rating"].mean()),
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "movies": movies,
        "low_history_users": list(low_history_users),
        "user_train_counts": user_train_counts,
    }

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data()
    out_path = os.path.join(OUT_DIR, "data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_path}")
