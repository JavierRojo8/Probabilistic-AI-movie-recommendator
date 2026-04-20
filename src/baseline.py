"""
SVD baseline: biased matrix factorization trained with minibatch Adam.
R_ij ≈ mu + bu_i + bv_j + U_i · V_j,  loss = MSE + L2 reg
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed

class _BiasedMF(nn.Module):

    def __init__(self, n_users, n_items, K, global_mean):
        super().__init__()
        self.global_mean = global_mean
        self.bu = nn.Embedding(n_users, 1)
        self.bv = nn.Embedding(n_items, 1)
        self.U = nn.Embedding(n_users, K)
        self.V = nn.Embedding(n_items, K)

        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bv.weight)
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        return (
            self.global_mean
            + self.bu(user_ids).squeeze(-1)
            + self.bv(item_ids).squeeze(-1)
            + (self.U(user_ids) * self.V(item_ids)).sum(-1)
        )

class SVDBaseline:

    def __init__(self, n_factors=20, n_epochs=20, lr=0.005, reg=0.02, batch_size=4096, device="auto"):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self._model = None
        self._n_users = 0
        self._n_items = 0
        self._global_mean = 0.0

    def fit(self, train_array, verbose=True, seed=42, n_users=None, n_items=None):
        set_seed(seed)
        dev = self._device

        users = torch.tensor(train_array[:, 0], dtype=torch.long)
        items = torch.tensor(train_array[:, 1], dtype=torch.long)
        ratings = torch.tensor(train_array[:, 2], dtype=torch.float32)

        self._n_users = n_users if n_users is not None else int(users.max().item()) + 1
        self._n_items = n_items if n_items is not None else int(items.max().item()) + 1
        self._global_mean = float(ratings.mean().item())

        self._model = _BiasedMF(self._n_users, self._n_items, self.n_factors, self._global_mean).to(dev)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        loader = DataLoader(TensorDataset(users, items, ratings), batch_size=self.batch_size, shuffle=True)
        N = len(ratings)

        for epoch in range(1, self.n_epochs + 1):
            self._model.train()
            total_loss = 0.0
            for u_batch, i_batch, r_batch in loader:
                u_batch = u_batch.to(dev)
                i_batch = i_batch.to(dev)
                r_batch = r_batch.to(dev)

                opt.zero_grad()
                pred = self._model(u_batch, i_batch)
                mse = ((r_batch - pred) ** 2).mean()

                reg_U = self._model.U(u_batch).pow(2).sum(1).mean()
                reg_V = self._model.V(i_batch).pow(2).sum(1).mean()
                reg_bu = self._model.bu(u_batch).pow(2).mean()
                reg_bv = self._model.bv(i_batch).pow(2).mean()
                loss = mse + self.reg * (reg_U + reg_V + reg_bu + reg_bv)

                loss.backward()
                opt.step()
                total_loss += mse.item() * len(r_batch)

            if verbose:
                train_rmse = np.sqrt(total_loss / N)
                print(f"  [SVD] epoch {epoch:3d}/{self.n_epochs}  train RMSE: {train_rmse:.4f}")

        return self

    def predict(self, user_ids, item_ids):
        assert self._model is not None, "Call fit() first."
        self._model.eval()
        with torch.no_grad():
            u = torch.tensor(user_ids, dtype=torch.long, device=self._device)
            i = torch.tensor(item_ids, dtype=torch.long, device=self._device)
            pred = self._model(u, i).cpu().numpy()
        return np.clip(pred, 1.0, 5.0).astype(np.float32)

    def recommend(self, user_idx, n_items, rated_items=None, top_k=10):
        all_items = np.arange(n_items, dtype=int)
        u_arr = np.full(n_items, user_idx, dtype=int)
        scores = self.predict(u_arr, all_items)
        if rated_items:
            scores[list(rated_items)] = -np.inf
        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, path):
        assert self._model is not None, "Call fit() before save()."
        torch.save(
            {
                "config": {
                    "n_factors": self.n_factors,
                    "n_epochs": self.n_epochs,
                    "lr": self.lr,
                    "reg": self.reg,
                    "batch_size": self.batch_size,
                    "n_users": self._n_users,
                    "n_items": self._n_items,
                    "global_mean": self._global_mean,
                },
                "model_state": self._model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        obj = cls(
            n_factors=cfg["n_factors"],
            n_epochs=cfg["n_epochs"],
            lr=cfg["lr"],
            reg=cfg["reg"],
            batch_size=cfg["batch_size"],
        )
        obj._n_users = cfg["n_users"]
        obj._n_items = cfg["n_items"]
        obj._global_mean = cfg["global_mean"]
        obj._model = _BiasedMF(cfg["n_users"], cfg["n_items"], cfg["n_factors"], cfg["global_mean"])
        obj._model.load_state_dict(ckpt["model_state"])
        obj._model = obj._model.to(obj._device)
        obj._model.eval()
        return obj
