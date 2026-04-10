"""
Evaluation metrics for the probabilistic recommender.

  - RMSE  : root mean squared error
  - MAE   : mean absolute error
  - NDCG@K: normalised discounted cumulative gain at rank K
  - ECE   : Expected Calibration Error for a Gaussian predictive distribution
            (measures how well uncertainty estimates are calibrated)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ------------------------------------------------------------------
# Point-estimate metrics
# ------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def ndcg_at_k(
    actual: dict[int, float],
    predicted_scores: dict[int, float],
    k: int = 10,
) -> float:
    """NDCG@K for a single user.

    Relevance is defined as max(0, rating - 2) so that ratings below 3
    contribute zero gain.

    Args:
        actual: {item_idx: true_rating}
        predicted_scores: {item_idx: predicted_score}
    """
    ranked = sorted(predicted_scores, key=lambda x: -predicted_scores[x])[:k]
    dcg = sum(
        max(0.0, actual.get(item, 0.0) - 2.0) / np.log2(rank + 2)
        for rank, item in enumerate(ranked)
    )
    ideal_rels = sorted(
        [max(0.0, r - 2.0) for r in actual.values()], reverse=True
    )[:k]
    idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg_at_k(
    test_data: np.ndarray, y_pred: np.ndarray, k: int = 10
) -> float:
    """Average NDCG@K over all users in test_data."""
    actual_by_user: dict[int, dict] = {}
    pred_by_user: dict[int, dict] = {}

    for (u, i, r), p in zip(test_data, y_pred):
        u, i = int(u), int(i)
        actual_by_user.setdefault(u, {})[i] = float(r)
        pred_by_user.setdefault(u, {})[i] = float(p)

    scores = [ndcg_at_k(actual_by_user[u], pred_by_user[u], k) for u in actual_by_user]
    return float(np.mean(scores))


# ------------------------------------------------------------------
# Uncertainty / calibration metrics
# ------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """ECE for a Gaussian predictive distribution.

    For each confidence level alpha, checks what fraction of true values
    actually fall within the alpha prediction interval. Perfect calibration
    means empirical coverage == alpha.

    Returns:
        ece: scalar
        conf_levels: (n_bins,) array
        empirical_coverage: (n_bins,) array
    """
    conf_levels = np.linspace(0.1, 0.99, n_bins)
    empirical_coverage = np.zeros(n_bins)

    for idx, alpha in enumerate(conf_levels):
        z = stats.norm.ppf(0.5 + alpha / 2.0)
        lower = y_mean - z * y_std
        upper = y_mean + z * y_std
        empirical_coverage[idx] = float(np.mean((y_true >= lower) & (y_true <= upper)))

    ece = float(np.mean(np.abs(empirical_coverage - conf_levels)))
    return ece, conf_levels, empirical_coverage


# ------------------------------------------------------------------
# Aggregated metric computation
# ------------------------------------------------------------------


def compute_all_metrics(
    test_data: np.ndarray,
    y_pred: np.ndarray,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    k: int = 10,
) -> dict[str, float]:
    """Compute all metrics for a split.

    Args:
        test_data: (N, 3) array [user_idx, item_idx, rating]
        y_pred:    (N,) clipped predictions
        y_mean:    (N,) raw predictive means (for calibration)
        y_std:     (N,) predictive standard deviations (for calibration)
    """
    y_true = test_data[:, 2]
    out: dict[str, float] = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        f"ndcg@{k}": mean_ndcg_at_k(test_data, y_pred, k),
    }
    if y_mean is not None and y_std is not None:
        ece, _, _ = expected_calibration_error(y_true, y_mean, y_std)
        out["ece"] = ece
        out["mean_uncertainty"] = float(y_std.mean())
    return out
