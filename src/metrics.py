import numpy as np
from scipy import stats

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def ndcg_at_k(actual, predicted_scores, k=10):
    # relevance = max(0, rating - 2), so only ratings >= 3 count
    ranked = sorted(predicted_scores, key=lambda x: -predicted_scores[x])[:k]
    dcg = sum(
        max(0.0, actual.get(item, 0.0) - 2.0) / np.log2(rank + 2)
        for rank, item in enumerate(ranked)
    )
    ideal = sorted([max(0.0, r - 2.0) for r in actual.values()], reverse=True)[:k]
    idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def mean_ndcg_at_k(test_data, y_pred, k=10):
    # NOTE: ranks only over each user's test items, not the full catalogue —
    # values are inflated but no problem for relative model comparison 
    actual_by_user = {}
    pred_by_user = {}

    for (u, i, r), p in zip(test_data, y_pred):
        u, i = int(u), int(i)
        actual_by_user.setdefault(u, {})[i] = float(r)
        pred_by_user.setdefault(u, {})[i] = float(p)

    scores = [ndcg_at_k(actual_by_user[u], pred_by_user[u], k) for u in actual_by_user]
    return float(np.mean(scores))

def expected_calibration_error(y_true, y_mean, y_std, n_bins=10):
    """Interval coverage ECE for a Gaussian predictive distribution."""
    y_std = np.maximum(y_std, 1e-8)

    conf_levels = np.linspace(0.1, 0.99, n_bins)
    coverage = np.zeros(n_bins)

    for idx, alpha in enumerate(conf_levels):
        z = stats.norm.ppf(0.5 + alpha / 2.0)
        lower = y_mean - z * y_std
        upper = y_mean + z * y_std
        coverage[idx] = float(np.mean((y_true >= lower) & (y_true <= upper)))

    ece = float(np.mean(np.abs(coverage - conf_levels)))
    return ece, conf_levels, coverage

def compute_all_metrics(test_data, y_pred, y_mean=None, y_std=None, k=10):
    y_true = test_data[:, 2]
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        f"ndcg@{k}": mean_ndcg_at_k(test_data, y_pred, k),
    }
    if y_mean is not None and y_std is not None:
        ece, _, _ = expected_calibration_error(y_true, y_mean, y_std)
        out["ece"] = ece
        out["mean_uncertainty"] = float(y_std.mean())
    return out