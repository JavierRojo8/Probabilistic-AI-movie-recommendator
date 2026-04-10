"""
Active Learning strategies for the BPMF recommender.

When a user has high model uncertainty, instead of serving a potentially
poor recommendation, we ask the user to rate a few movies that will
maximally reduce the epistemic uncertainty.

Strategy: Uncertainty Sampling
  Select items where the epistemic variance of the predicted rating
  (Var_q[U_i . V_j]) is highest. Rating these items will inform the
  posterior over U_i most efficiently.
"""

from __future__ import annotations

import numpy as np
import torch

from bpmf import BPMF


def get_active_learning_candidates(
    model: BPMF,
    user_idx: int,
    rated_item_ids: set[int],
    n_candidates: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Items to show the user for rating to reduce uncertainty.

    Args:
        model:           trained BPMF
        user_idx:        user index
        rated_item_ids:  items the user has already rated (to exclude)
        n_candidates:    how many items to suggest

    Returns:
        candidate_item_ids: (n_candidates,) array of item indices
        epistemic_vars:     (n_candidates,) corresponding variances
    """
    with torch.no_grad():
        var = model.item_epistemic_variance(user_idx)   # (n_items,)

    # Mask already-rated items
    if rated_item_ids:
        rated = torch.tensor(list(rated_item_ids), dtype=torch.long)
        var[rated] = -1.0

    top_idx = torch.argsort(var, descending=True)[:n_candidates]
    return top_idx.cpu().numpy(), var[top_idx].cpu().numpy()


def user_is_uncertain(
    model: BPMF,
    user_idx: int,
    threshold: float = 1.2,
) -> bool:
    """Return True if the user's mean predictive uncertainty is above threshold."""
    return model.user_uncertainty(user_idx) > threshold
