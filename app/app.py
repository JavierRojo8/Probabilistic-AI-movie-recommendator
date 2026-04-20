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
from scipy.stats import norm as _scipy_norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from active_learning import get_active_learning_candidates
from baseline import SVDBaseline
from bpmf import BPMF

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DATA_PATH    = os.path.join(os.path.dirname(__file__), "../data/processed/data.pkl")
CKPT_PATH    = os.path.join(os.path.dirname(__file__), "../checkpoints/bpmf_best.pt")
SVD_CKPT_PATH = os.path.join(os.path.dirname(__file__), "../checkpoints/svd_baseline.pt")

UNCERTAINTY_THRESHOLD = 1.2


# ------------------------------------------------------------------
# Cached loading
# ------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def load_model_and_data():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = BPMF(
        ckpt["n_users"],
        ckpt["n_items"],
        K=ckpt["K"],
        global_mean=ckpt["global_mean"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Precompute inverted index once so warm-start lookups are O(1) per item
    model._build_item_user_index(data["train"])

    return model, data


@st.cache_resource(show_spinner="Loading SVD baseline…")
def load_svd() -> SVDBaseline | None:
    if not os.path.exists(SVD_CKPT_PATH):
        return None
    return SVDBaseline.load(SVD_CKPT_PATH)


# ------------------------------------------------------------------
# Shared helpers
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
    """Map std → [0, 1] confidence (inverted and clipped)."""
    return float(np.clip(1.0 - (std - min_std) / (max_std - min_std), 0.0, 1.0))


def render_rec_table(recs: dict, movies_df, sort_by: str, safe_picks_z: float = 1.96) -> None:
    """Render a recommendations dict as a Streamlit table.

    sort_by options:
      "Predicted score"                 — sort by mean (model default)
      "Confidence (most certain first)" — sort by mean - 1·std
      "Best in range (safe picks)"      — sort by mean - z·std (z from credible interval %)
    """
    ids   = recs["item_ids"].copy()
    means = recs["means"].copy()
    stds  = recs["stds"].copy()

    if sort_by == "Confidence (most certain first)":
        order = np.argsort(-(means - stds))
        ids, means, stds = ids[order], means[order], stds[order]
    elif sort_by == "Best in range (safe picks)":
        order = np.argsort(-(means - safe_picks_z * stds))
        ids, means, stds = ids[order], means[order], stds[order]

    if sort_by == "Best in range (safe picks)":
        pct = int(round(float(_scipy_norm.cdf(safe_picks_z) - 0.5) * 200))
        caption_sort = (
            f"Sorted by **{pct}% credible lower bound** "
            f"(score − {safe_picks_z:.2f}·std). "
            "Only movies the model is both enthusiastic *and* very sure about rise to the top."
        )
    elif sort_by == "Confidence (most certain first)":
        caption_sort = (
            "Sorted by **score − 1·std** (1-sigma lower bound). "
            "Promotes items the model both likes *and* is sure about."
        )
    else:
        caption_sort = "Sorted by **predicted score** (highest expected rating first)."

    st.caption(
        "**Score** — predicted rating on a 1–5 ★ scale "
        "(1 = strongly disliked, 3 = neutral, 5 = loved). "
        "**Confidence** — how certain the model is: 100 % = very sure, "
        f"low % = educated guess. {caption_sort}"
    )

    header = st.columns([0.4, 3, 2, 1.2, 2])
    for col, label in zip(header, ["#", "Title", "Genres", "Score ★", "Confidence"]):
        col.markdown(f"**{label}**")
    st.divider()

    for rank, (item_idx, mean_r, std_r) in enumerate(zip(ids, means, stds), start=1):
        title, genres = lookup_movie(movies_df, int(item_idx))
        conf = confidence_bar(std_r)
        cols = st.columns([0.4, 3, 2, 1.2, 2])
        cols[0].write(f"**{rank}**")
        cols[1].write(title)
        cols[2].write(genres)
        cols[3].write(f"{mean_r:.2f} ★")
        cols[4].progress(conf, text=f"{conf:.0%}")


def render_svd_table(svd: SVDBaseline, user_idx: int, rated_items: set[int],
                     top_k: int, movies_df) -> None:
    """Show SVD baseline recommendations for side-by-side comparison."""
    recs = svd.recommend(user_idx, svd._n_items, rated_items=rated_items, top_k=top_k)
    header = st.columns([0.4, 3, 2, 1.2])
    for col, label in zip(header, ["#", "Title", "Genres", "Score ★"]):
        col.markdown(f"**{label}**")
    st.divider()
    for rank, (item_idx, score) in enumerate(recs, start=1):
        title, genres = lookup_movie(movies_df, item_idx)
        cols = st.columns([0.4, 3, 2, 1.2])
        cols[0].write(f"**{rank}**")
        cols[1].write(title)
        cols[2].write(genres)
        cols[3].write(f"{score:.2f} ★")


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
    svd               = load_svd()
    movies_df         = data["movies"]
    train_arr         = data["train"]
    n_users           = data["n_users"]
    low_history_users: list[int] = data.get("low_history_users", data.get("cold_users", []))

    # ---- Top-level navigation in sidebar ----
    st.sidebar.header("Navigation")
    section = st.sidebar.radio(
        "Mode",
        ["👤  Existing User", "🆕  New User Demo"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    # ---- Shared display settings ----
    st.sidebar.header("Display settings")
    top_k = st.sidebar.slider("Recommendations to show", 5, 20, 10)
    sort_by = st.sidebar.radio(
        "Sort by",
        ["Predicted score", "Confidence (most certain first)", "Best in range (safe picks)"],
        help=(
            "**Predicted score**: highest expected rating first.\n\n"
            "**Confidence**: score − 1·std. Promotes items the model both likes *and* is sure about.\n\n"
            "**Best in range**: lower bound of a credible interval (score − z·std)."
        ),
    )
    safe_picks_z = 1.96
    if sort_by == "Best in range (safe picks)":
        confidence_pct = st.sidebar.slider(
            "Credible interval confidence",
            min_value=50, max_value=99, value=95, step=1, format="%d%%",
        )
        safe_picks_z = float(_scipy_norm.ppf(0.5 + confidence_pct / 200.0))

    # ==================================================================
    # SECTION 1 — Existing user
    # ==================================================================
    if section == "👤  Existing User":
        st.sidebar.divider()
        st.sidebar.header("User selection")
        mode = st.sidebar.radio("User mode", ["Existing user", "Low-history user demo"])

        if mode == "Existing user":
            user_idx = int(st.sidebar.number_input("User index", 0, n_users - 1, 0, step=1))
        else:
            if low_history_users:
                user_idx = st.sidebar.selectbox(
                    "Low-history user (≤ 20 training ratings)",
                    options=low_history_users[:50],
                )
            else:
                st.sidebar.info("No low-history users found in current split.")
                user_idx = 0

        st.header("👤 Existing User")

        rated_items = get_rated_items(train_arr, user_idx)
        n_rated     = len(rated_items)
        avg_unc     = model.user_uncertainty(user_idx) ** 0.5

        c1, c2, c3 = st.columns(3)
        c1.metric("User index", user_idx)
        c2.metric("Movies rated (train)", n_rated)
        c3.metric(
            "Mean uncertainty (std)",
            f"{avg_unc:.3f}",
            help="Average predictive std across all items. Higher = model knows less about this user.",
        )

        # Rated movies expander
        with st.expander(f"Movies already rated by this user ({n_rated} total)", expanded=False):
            if n_rated == 0:
                st.write("No training ratings found for this user.")
            else:
                # Retrieve ratings from train_arr
                user_mask = train_arr[:, 0].astype(int) == user_idx
                user_rows = train_arr[user_mask]
                # Sort by rating descending
                user_rows = user_rows[np.argsort(-user_rows[:, 2])]
                cols_header = st.columns([3, 2, 1])
                cols_header[0].markdown("**Title**")
                cols_header[1].markdown("**Genres**")
                cols_header[2].markdown("**Rating**")
                for row in user_rows:
                    item_idx_r, rating_r = int(row[1]), float(row[2])
                    title, genres = lookup_movie(movies_df, item_idx_r)
                    stars = "★" * int(rating_r) + "☆" * (5 - int(rating_r))
                    cols = st.columns([3, 2, 1])
                    cols[0].write(title)
                    cols[1].write(genres)
                    cols[2].write(f"{stars} {rating_r:.0f}")

        st.divider()

        # Active learning banner
        is_uncertain = n_rated <= 20 or avg_unc > UNCERTAINTY_THRESHOLD
        if is_uncertain:
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

        # Recommendations
        recs = model.recommend(user_idx, rated_items=list(rated_items), top_k=top_k)

        if mode == "Low-history user demo" and svd is not None:
            st.subheader(f"Top-{top_k} Recommendations — BPMF vs SVD")
            st.caption(
                "Low-history users have ≤ 20 training ratings. "
                "BPMF's uncertainty estimate helps it personalise better than SVD when data is scarce."
            )
            col_bpmf, col_svd = st.columns(2)
            with col_bpmf:
                st.markdown("#### BPMF (Bayesian)")
                render_rec_table(recs, movies_df, sort_by, safe_picks_z)
            with col_svd:
                st.markdown("#### SVD Baseline")
                render_svd_table(svd, user_idx, rated_items, top_k, movies_df)
        else:
            st.subheader(f"Top-{top_k} Recommendations")
            render_rec_table(recs, movies_df, sort_by, safe_picks_z)

        with st.expander("About the model"):
            st.markdown(
                """
                **Bayesian Probabilistic Matrix Factorization (BPMF)**

                - User and item latent factors are modelled as *distributions* (Gaussian),
                  not point estimates — giving us uncertainty over recommendations.
                - Training uses **Stochastic Variational Inference (SVI)** with minibatches,
                  which scales to the full MovieLens 1M dataset (~1 M ratings).
                - The **confidence bar** reflects epistemic uncertainty: wide posterior → low confidence.
                - **Active Learning**: when uncertainty is high, suggests movies that most reduce
                  posterior variance (uncertainty sampling).
                - **Low-history user demo**: see BPMF vs SVD side-by-side on users with few ratings.
                """
            )

    # ==================================================================
    # SECTION 2 — New User Demo
    # ==================================================================
    else:
        st.header("🆕 New User Demo")
        st.markdown(
            "Build a fresh profile by rating a few movies. "
            "Watch how recommendations and confidence evolve with each new rating."
        )

        if "demo_ratings" not in st.session_state:
            st.session_state.demo_ratings = {}

        # ---- Search & Rate ----
        st.subheader("Search & Rate")
        query = st.text_input(
            "Movie title",
            placeholder="e.g. Toy Story, Matrix, Pulp Fiction…",
            key="demo_search",
        )

        if query:
            mask    = movies_df["title"].str.contains(query, case=False, na=False)
            matches = movies_df[mask].head(10)
            if len(matches) == 0:
                st.info("No movies found — try a different title fragment.")
            else:
                labels  = [
                    f"{r['title']}  ({r['genres'].split('|')[0]})"
                    for _, r in matches.iterrows()
                ]
                indices = [int(r["item_idx"]) for _, r in matches.iterrows()]
                sel_pos = st.selectbox(
                    "Select a movie",
                    range(len(labels)),
                    format_func=lambda x: labels[x],
                    key="demo_select",
                )
                sel_idx = indices[sel_pos]

                if sel_idx in st.session_state.demo_ratings:
                    st.info(
                        f"Already rated: {st.session_state.demo_ratings[sel_idx]:.1f} ★ "
                        "— submitting again will update the rating."
                    )
                rating_val = st.slider(
                    "Your rating", 1.0, 5.0, 3.5, 0.5,
                    key="demo_rating",
                    help="1 = strongly disliked · 3 = neutral · 5 = loved",
                )
                if st.button("Add Rating", type="primary"):
                    st.session_state.demo_ratings[sel_idx] = rating_val
                    st.rerun()

        if not st.session_state.demo_ratings:
            st.info("Search for a movie above and add your first rating to see recommendations.")
            return

        st.divider()

        rated_item_ids = list(st.session_state.demo_ratings.keys())
        rated_values   = [st.session_state.demo_ratings[i] for i in rated_item_ids]

        with st.spinner("Updating model…"):
            new_user = model.fit_new_user(rated_item_ids, rated_values)

        avg_unc_new = model.mean_uncertainty_new_user(new_user)

        mc1, mc2 = st.columns(2)
        mc1.metric("Movies rated", len(rated_item_ids))
        mc2.metric(
            "Mean uncertainty (std)",
            f"{avg_unc_new:.3f}",
            help="Decreases as you rate more movies. Below ~0.9 the model is fairly confident.",
        )

        with st.expander(
            f"Your {len(rated_item_ids)} rated movie(s) — click to expand", expanded=False
        ):
            for item_idx, rating in st.session_state.demo_ratings.items():
                title, genres = lookup_movie(movies_df, item_idx)
                filled = "★" * int(rating)
                empty  = "☆" * (5 - int(rating))
                st.write(f"{filled}{empty}  **{title}** — {genres}")

        if st.button("Clear all ratings", key="demo_clear"):
            st.session_state.demo_ratings = {}
            st.rerun()

        st.divider()

        # Active learning for new user
        all_items_t = torch.arange(model.n_items, dtype=torch.long)
        _, all_vars = model.predict_new_user(new_user, all_items_t)
        all_vars_np = all_vars.numpy().copy()
        all_vars_np[rated_item_ids] = -1.0

        if avg_unc_new > UNCERTAINTY_THRESHOLD or len(rated_item_ids) <= 3:
            st.warning(
                "**High uncertainty.** "
                "Rating these movies will most reduce the model's uncertainty about you:"
            )
            top_al  = np.argsort(-all_vars_np)[:5]
            al_cols = st.columns(5)
            for col, al_idx in zip(al_cols, top_al):
                title, genres = lookup_movie(movies_df, int(al_idx))
                col.info(
                    f"**{title}**\n\n{genres}\n\n"
                    f"*(uncertainty: {all_vars_np[al_idx]**0.5:.3f})*"
                )
            st.divider()

        st.subheader(f"Top-{top_k} Recommendations")
        recs = model.recommend_new_user(
            new_user, rated_items=rated_item_ids, top_k=top_k
        )
        render_rec_table(recs, movies_df, sort_by, safe_picks_z)


if __name__ == "__main__":
    main()
