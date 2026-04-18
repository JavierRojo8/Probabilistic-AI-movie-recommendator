"""
Active Learning strategies for the BPMF recommender.

When a user has high model uncertainty, instead of serving a potentially
poor recommendation, we ask the user to rate a few movies that will
maximally reduce the epistemic uncertainty.

Strategy: Uncertainty Sampling
  Select items where the epistemic variance of the predicted rating
  (Var_q[U_i · V_j]) is highest. Rating these items will inform the
  posterior over U_i most efficiently.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
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
        candidate_item_ids: (≤ n_candidates,) array of item indices.
                            May be shorter if few unrated items remain.
        epistemic_vars:     (≤ n_candidates,) corresponding variances.
    """
    with torch.no_grad():
        var = model.item_epistemic_variance(user_idx)   # (n_items,)

    # Mask already-rated items — tensor must be on the same device as var
    if rated_item_ids:
        rated = torch.tensor(
            list(rated_item_ids), dtype=torch.long, device=var.device
        )
        var[rated] = -1.0

    # Sort only over valid (unmasked) candidates
    valid_idx = torch.where(var >= 0)[0]
    if len(valid_idx) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty.astype(np.float32)

    top_local = torch.argsort(var[valid_idx], descending=True)[:n_candidates]
    top_idx = valid_idx[top_local]
    return top_idx.cpu().numpy(), var[top_idx].cpu().numpy()
