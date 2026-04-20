"""
Script to evaluate BPMF vs SVD on the test set, with a breakdown for low-history users. Hopefully this will show 
BPMF's advantages in uncertainty estimation and cold-start scenarios.

Outputs:
    results/metrics_comparison.csv
    results/calibration.png
    results/uncertainty_vs_error.png
    results/low_history_comparison.png
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline import SVDBaseline
from bpmf import BPMF
from metrics import compute_all_metrics, expected_calibration_error, rmse
from utils import set_seed

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "data.pkl")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "bpmf_best.pt")
SVD_CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "svd_baseline.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def load_bpmf(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = BPMF(ckpt["n_users"], ckpt["n_items"], K=ckpt["K"], global_mean=ckpt["global_mean"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)

def load_or_train_svd(train_arr, seed, n_users, n_items):
    os.makedirs(os.path.dirname(SVD_CKPT), exist_ok=True)
    if os.path.exists(SVD_CKPT):
        print(f"Loading cached SVD from {SVD_CKPT} ...")
        return SVDBaseline.load(SVD_CKPT)

    print("Training SVD baseline (~1-2 min)...")
    svd = SVDBaseline(n_factors=20, n_epochs=20)
    svd.fit(train_arr, seed=seed, n_users=n_users, n_items=n_items)
    svd.save(SVD_CKPT)
    print(f"  Saved to {SVD_CKPT}")
    return svd

def bpmf_predict(model, data_arr, device="cpu", batch_size=8192):
    means_list, stds_list = [], []
    for start in range(0, len(data_arr), batch_size):

        chunk = data_arr[start : start + batch_size]
        u = torch.tensor(chunk[:, 0], dtype=torch.long, device=device)
        i = torch.tensor(chunk[:, 1], dtype=torch.long, device=device)
        with torch.no_grad():
            means, variances = model.predict(u, i)
        means_list.append(means.cpu().numpy())
        stds_list.append(variances.sqrt().cpu().numpy())

    y_mean = np.concatenate(means_list)
    y_std = np.concatenate(stds_list)
    return y_mean.clip(1, 5), y_mean, y_std

def svd_predict(model, data_arr):
    u = data_arr[:, 0].astype(int)
    i = data_arr[:, 1].astype(int)
    return model.predict(u, i).clip(1, 5)

def plot_calibration(y_true, y_mean, y_std, save_path=None):
    ece, conf, emp = expected_calibration_error(y_true, y_mean, y_std)
    _, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(conf, emp, "b-o", label=f"BPMF (ECE = {ece:.4f})")
    ax.set_xlabel("Expected Coverage (confidence level)")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Uncertainty Calibration")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return ece

def plot_uncertainty_vs_error(y_true, y_pred, y_std, save_path=None):
    abs_err = np.abs(y_true - y_pred)
    edges = np.percentile(y_std, np.linspace(0, 100, 11))

    centers, errors = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_std >= lo) & (y_std <= hi)
        if mask.sum() > 0:
            centers.append(y_std[mask].mean())
            errors.append(abs_err[mask].mean())

    _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(centers)), errors, color="steelblue")
    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels([f"{c:.2f}" for c in centers], rotation=45, ha="right")
    ax.set_xlabel("Mean Predictive Std (uncertainty decile)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Uncertainty vs. Actual Error (BPMF)")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_low_history_comparison(test_data, bpmf_pred, svd_pred, user_train_counts, save_path=None):
    buckets = [(1, 5), (6, 20), (21, 50), (51, 150), (151, 10_000)]
    labels, bpmf_rmses, svd_rmses = [], [], []

    for lo, hi in buckets:
        mask = np.array([lo <= user_train_counts.get(int(u), 0) <= hi for u in test_data[:, 0]])
        if mask.sum() < 5:
            continue
        y_true = test_data[mask, 2]
        labels.append(f"{lo}–{hi}")
        bpmf_rmses.append(rmse(y_true, bpmf_pred[mask]))
        svd_rmses.append(rmse(y_true, svd_pred[mask]))

    x = np.arange(len(labels))
    w = 0.35
    _, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, bpmf_rmses, w, label="BPMF", color="steelblue")
    ax.bar(x + w / 2, svd_rmses, w, label="SVD", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Train ratings\n{l}" for l in labels])
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Training-Set Size (Low-History User Analysis)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def main(seed=42):
    set_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Data not found. Run: python src/prepare_data.py"); return
    if not os.path.exists(CKPT_PATH):
        print("Checkpoint not found. Run: python src/train.py"); return

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading BPMF model...")
    bpmf = load_bpmf(CKPT_PATH, device)
    svd = load_or_train_svd(data["train"], seed=seed, n_users=data["n_users"], n_items=data["n_items"])

    test_data = data["test"]
    low_test = data.get("low_history_test", data.get("cold_test", np.empty((0, 3), dtype=np.float32)))
    high_test = data.get("high_history_test", data.get("warm_test", np.empty((0, 3), dtype=np.float32)))
    user_train_counts = data.get("user_train_counts", {})

    rows = []
    bpmf_pred, bpmf_mean, bpmf_std = bpmf_predict(bpmf, test_data, device)
    svd_pred = svd_predict(svd, test_data)

    for name, split in [("Full Test", test_data), ("Low-History Users", low_test), ("High-History Users", high_test),]:
        if len(split) == 0:
            continue
        print(f"\n=== {name} ({len(split)} ratings) ===")

        if split is test_data:
            bp, bm, bs = bpmf_pred, bpmf_mean, bpmf_std
            sp = svd_pred
        else:
            bp, bm, bs = bpmf_predict(bpmf, split, device)
            sp = svd_predict(svd, split)
        bm_metrics = compute_all_metrics(split, bp, bm, bs)
        sm_metrics = compute_all_metrics(split, sp)
        print(f"  BPMF: {bm_metrics}")
        print(f"  SVD:  {sm_metrics}")

        for metric, val in bm_metrics.items():
            rows.append({"Split": name, "Model": "BPMF", "Metric": metric, "Value": round(val, 4)})

        for metric, val in sm_metrics.items():
            rows.append({"Split": name, "Model": "SVD", "Metric": metric, "Value": round(val, 4)})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved → {csv_path}")
    y_true = test_data[:, 2]
    plot_calibration(y_true, bpmf_mean, bpmf_std, save_path=os.path.join(RESULTS_DIR, "calibration.png"))
    print("Saved calibration.png")

    plot_uncertainty_vs_error(y_true, bpmf_pred, bpmf_std, save_path=os.path.join(RESULTS_DIR, "uncertainty_vs_error.png"))
    print("Saved uncertainty_vs_error.png")

    plot_low_history_comparison(test_data, bpmf_pred, svd_pred, user_train_counts, save_path=os.path.join(RESULTS_DIR, "low_history_comparison.png"))
    print("Saved low_history_comparison.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate BPMF vs SVD baseline")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(seed=args.seed)
