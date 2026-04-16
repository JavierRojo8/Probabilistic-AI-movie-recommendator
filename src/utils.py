"""
Shared utilities for reproducibility and common helpers.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Covers Python's random, NumPy, PyTorch CPU, and (if available) all CUDA
    devices.  Call this before creating any model, DataLoader, or split.

    Args:
        seed: integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
