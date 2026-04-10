"""
Streamlit frontend for the Bayesian Movie Recommender.

Run:  streamlit run app/app.py
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from active_learning import get_active_learning_candidates
from bpmf import BPMF

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/data.pkl")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "../checkpoints/bpmf_best.pt")

# Std threshold above which active learning is triggered
UNCERTAINTY_THRESHOLD = 1.2


# ------------------------------------------------------------------
# Cached loading
# ------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def load_model_and_data():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = BPMF(
        ckpt["n_users"],
        ckpt["n_items"],
        K=ckpt["K"],
        global_mean=ckpt["global_mean"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, data


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def get_rated_items(train_arr: np.ndarray, user_idx: int) -> set[int]:
    mask = train_arr[:, 0].astype(int) == user_idx
    return set(train_arr[mask, 1].astype(int))


def lookup_movie(movies_df, item_idx: int) -> tuple[str, str]:
    row = movies_df[movies_df["item_idx"] == item_idx]
    if len(row) == 0:
        return f"Movie {item_idx}", ""
    return row["title"].values[0], row["genres"].values[0].replace("|", " · ")


def confidence_bar(std: float, min_std: float = 0.5, max_std: float = 2.5) -> float:
    """Map std → [0, 1] confidence (inverted)."""
    return float(np.clip(1.0 - (std - min_std) / (max_std - min_std), 0.0, 1.0))


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Bayesian Movie Recommender", layout="wide")
    st.title("Bayesian Movie Recommender")
    st.caption(
        "Bayesian Probabilistic Matrix Factorization (BPMF) "
        "with Stochastic Variational Inference — uncertainty-aware recommendations."
    )

    if not os.path.exists(DATA_PATH) or not os.path.exists(CKPT_PATH):
        st.error(
            "Model or data files not found.\n\n"
            "Please run:\n"
            "```\n"
            "python src/prepare_data.py\n"
            "python src/train.py\n"
            "```"
        )
        return

    model, data = load_model_and_data()
    movies_df = data["movies"]
    train_arr = data["train"]
    n_users = data["n_users"]
    cold_users: list[int] = data.get("cold_users", [])

    # ---- Sidebar ----
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("User mode", ["Existing user", "Cold-start demo"])
    top_k = st.sidebar.slider("Recommendations to show", 5, 20, 10)

    if mode == "Existing user":
        user_idx = st.sidebar.number_input("User index", 0, n_users - 1, 0, step=1)
        user_idx = int(user_idx)
    else:
        if cold_users:
            user_idx = st.sidebar.selectbox(
                "Cold-start user (few training ratings)",
                options=cold_users[:50],
            )
        else:
            st.sidebar.info("No cold-start users found in current split.")
            user_idx = 0

    # ---- User stats ----
    rated_items = get_rated_items(train_arr, user_idx)
    n_rated = len(rated_items)
    avg_unc = model.user_uncertainty(user_idx) ** 0.5   # approx std

    c1, c2, c3 = st.columns(3)
    c1.metric("User index", user_idx)
    c2.metric("Movies rated (train)", n_rated)
    c3.metric("Mean uncertainty (std)", f"{avg_unc:.3f}")

    st.divider()

    # ---- Active learning banner ----
    is_cold = n_rated <= 20 or avg_unc > UNCERTAINTY_THRESHOLD
    if is_cold:
        st.warning(
            "**High uncertainty detected.** "
            "The model has limited data for this user. "
            "Rating a few movies below will significantly improve recommendations."
        )
        al_items, al_vars = get_active_learning_candidates(
            model, user_idx, rated_items, n_candidates=5
        )
        st.markdown("**Rate these movies to reduce uncertainty:**")
        al_cols = st.columns(len(al_items))
        for col, item_idx_al, var in zip(al_cols, al_items, al_vars):
            title, genres = lookup_movie(movies_df, int(item_idx_al))
            col.info(f"**{title}**\n\n{genres}\n\n*(uncertainty: {var**0.5:.3f})*")

        st.divider()

    # ---- Recommendations ----
    st.subheader(f"Top-{top_k} Recommendations")
    recs = model.recommend(user_idx, rated_items=list(rated_items), top_k=top_k)

    header = st.columns([0.4, 3, 2, 1, 2])
    for col, label in zip(header, ["#", "Title", "Genres", "Score", "Confidence"]):
        col.markdown(f"**{label}**")
    st.divider()

    for rank, (item_idx_r, mean_r, std_r) in enumerate(
        zip(recs["item_ids"], recs["means"], recs["stds"]), start=1
    ):
        title, genres = lookup_movie(movies_df, int(item_idx_r))
        conf = confidence_bar(std_r)
        cols = st.columns([0.4, 3, 2, 1, 2])
        cols[0].write(f"**{rank}**")
        cols[1].write(title)
        cols[2].write(genres)
        cols[3].write(f"{mean_r:.2f} ★")
        cols[4].progress(conf, text=f"{conf:.0%}")

    # ---- About ----
    with st.expander("About the model"):
        st.markdown(
            """
            **Bayesian Probabilistic Matrix Factorization (BPMF)**

            - User and item latent factors are modelled as *distributions* (Gaussian),
              not point estimates — giving us uncertainty over recommendations.
            - Training uses **Stochastic Variational Inference (SVI)** with minibatches,
              which scales to the full MovieLens 1M dataset (~1 M ratings).
            - The **confidence bar** reflects the model's epistemic uncertainty:
              wide posterior → low confidence → active-learning banner.
            - **Active Learning**: when uncertainty is high the model suggests movies
              that, if rated, would most reduce posterior variance (uncertainty sampling).
            - Compared against a standard **SVD baseline** (Surprise library);
              BPMF shows its advantage on cold-start users.
            """
        )


if __name__ == "__main__":
    main()
