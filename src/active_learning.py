"""
Active learning: pick items that will reduce the model's uncertainty the most.
When a user is cold or uncertain, ask them to rate these before recommending.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from bpmf import BPMF


def get_active_learning_candidates(model, user_idx, rated_item_ids, n_candidates=5):
    """Return items to show the user for rating to reduce uncertainty."""
    with torch.no_grad():
        var = model.item_epistemic_variance(user_idx)

    if rated_item_ids:
        already_rated = torch.tensor(list(rated_item_ids), dtype=torch.long, device=var.device)
        var[already_rated] = -1.0

    valid = torch.where(var >= 0)[0]
    if len(valid) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty.astype(np.float32)

    top_local = torch.argsort(var[valid], descending=True)[:n_candidates]
    top = valid[top_local]
    return top.cpu().numpy(), var[top].cpu().numpy()
