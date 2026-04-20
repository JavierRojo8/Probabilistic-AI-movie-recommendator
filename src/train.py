"""
Train BPMF with SVI. Best checkpoint (lowest val RMSE) saved to checkpoints/bpmf_best.pt.
"""
import argparse
import copy
import os
import pickle
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from bpmf import BPMF
from utils import set_seed

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "data.pkl")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

def train(data, K=20, n_epochs=20, batch_size=4096, lr=0.01, device="cpu", seed=42):
    set_seed(seed)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_arr = data["train"]
    val_arr = data["val"]
    n_users = data["n_users"]
    n_items = data["n_items"]
    global_mean = data["global_mean"]
    n_total = len(train_arr)

    user_t = torch.tensor(train_arr[:, 0], dtype=torch.long)
    item_t = torch.tensor(train_arr[:, 1], dtype=torch.long)
    rate_t = torch.tensor(train_arr[:, 2], dtype=torch.float32)
    loader = DataLoader(TensorDataset(user_t, item_t, rate_t),batch_size=batch_size,shuffle=True,num_workers=0,)

    model = BPMF(n_users, n_items, K=K, global_mean=global_mean).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    val_u = torch.tensor(val_arr[:, 0], dtype=torch.long)
    val_i = torch.tensor(val_arr[:, 1], dtype=torch.long)
    val_r = val_arr[:, 2]

    best_rmse = float("inf")
    best_state = None
    best_path = os.path.join(CKPT_DIR, "bpmf_best.pt")
    kl_anneal_epochs = min(10, n_epochs)

    print(f"Training BPMF  K={K}  epochs={n_epochs}  batch={batch_size}  lr={lr}  seed={seed}  device={device}")
    print(f"  n_users={n_users}  n_items={n_items}  n_train={n_total}")
    print(f"  {'Epoch':>6}  {'ELBO/batch':>12}  {'Val RMSE':>10}  {'sigma_obs':>10}  {'Time':>6}")
    print("  " + "-" * 55)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        total_elbo = 0.0
        n_batches = 0

        for bu, bi, br in loader:
            bu = bu.to(device)
            bi = bi.to(device)
            br = br.to(device)

            opt.zero_grad()
            # KL annealing: ramp from 0 to 1 over the first 10 epochs
            kl_weight = min(1.0, epoch / 10.0)
            elbo = model.elbo(bu, bi, br, n_total, kl_weight=kl_weight)
            (-elbo).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            total_elbo += elbo.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_means, _ = model.predict(val_u.to(device), val_i.to(device))
            val_pred = val_means.cpu().numpy().clip(1, 5)
        val_rmse = float(np.sqrt(np.mean((val_r - val_pred) ** 2)))

        elapsed = time.time() - t0
        print(
            f"  {epoch:>6d}  {total_elbo/n_batches:>12.1f}  "
            f"{val_rmse:>10.4f}  {model.sigma_obs.item():>10.4f}  {elapsed:>5.1f}s"
        )

        if epoch >= kl_anneal_epochs and val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": best_state,
                    "val_rmse": val_rmse,
                    "n_users": n_users,
                    "n_items": n_items,
                    "K": K,
                    "global_mean": global_mean,
                    "batch_size": batch_size,
                    "lr": lr,
                    "n_epochs": n_epochs,
                    "seed": seed,
                },
                best_path,
            )

    print(f"\nBest Val RMSE: {best_rmse:.4f}  →  {best_path}")

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model

def parse_args():
    p = argparse.ArgumentParser(description="Train BPMF")
    p.add_argument("--K", type=int, default=20, help="Latent factor dimension")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print(f"Loading data from {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        print("Data not found. Run: python src/prepare_data.py")
        sys.exit(1)

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    train(data, K=args.K, n_epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=device, seed=args.seed)
