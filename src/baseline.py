"""
SVD baseline: biased matrix factorisation trained with minibatch Adam (PyTorch).

Model:  R_ij ≈ mu + bu_i + bv_j + U_i . V_j
Loss:   MSE + L2 regularisation on factors and biases

This is the classical Funk-SVD / Biased-MF baseline that BPMF is compared
against.  Using PyTorch makes training fast (same speed as BPMF) without
needing scikit-surprise.
"""

from __future__ import annotations

import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _BiasedMF(nn.Module):
    """Inner PyTorch module (point-estimate only, no uncertainty)."""

    def __init__(self, n_users: int, n_items: int, K: int, global_mean: float):
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

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        pred = (
            self.global_mean
            + self.bu(user_ids).squeeze(-1)
            + self.bv(item_ids).squeeze(-1)
            + (self.U(user_ids) * self.V(item_ids)).sum(-1)
        )
        return pred


class SVDBaseline:
    """Biased MF trained with minibatch Adam — fast on 1 M ratings."""

    def __init__(
        self,
        n_factors: int = 20,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        batch_size: int = 4096,
        device: str = "auto",
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self._device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self._model: _BiasedMF | None = None
        self._n_users = 0
        self._n_items = 0

    # ------------------------------------------------------------------

    def fit(self, train_array: np.ndarray, verbose: bool = True) -> "SVDBaseline":
        """Train on (N, 3) float32 array of [user_idx, item_idx, rating]."""
        dev = self._device

        users = torch.tensor(train_array[:, 0], dtype=torch.long)
        items = torch.tensor(train_array[:, 1], dtype=torch.long)
        ratings = torch.tensor(train_array[:, 2], dtype=torch.float32)

        n_users = int(users.max().item()) + 1
        n_items = int(items.max().item()) + 1
        self._n_users = n_users
        self._n_items = n_items
        global_mean = float(ratings.mean().item())

        self._model = _BiasedMF(n_users, n_items, self.n_factors, global_mean).to(dev)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        loader = DataLoader(
            TensorDataset(users, items, ratings),
            batch_size=self.batch_size,
            shuffle=True,
        )
        N = len(ratings)

        for epoch in range(1, self.n_epochs + 1):
            self._model.train()
            total_loss = 0.0
            for bu_, bi_, r_ in loader:
                bu_ = bu_.to(dev)
                bi_ = bi_.to(dev)
                r_ = r_.to(dev)

                optimizer.zero_grad()
                pred = self._model(bu_, bi_)
                mse = ((r_ - pred) ** 2).mean()
                # L2 regularisation on the current batch's embeddings
                reg = self.reg * (
                    self._model.U(bu_).norm() ** 2
                    + self._model.V(bi_).norm() ** 2
                    + self._model.bu(bu_).norm() ** 2
                    + self._model.bv(bi_).norm() ** 2
                )
                loss = mse + reg / len(r_)
                loss.backward()
                optimizer.step()
                total_loss += mse.item() * len(r_)

            if verbose:
                rmse = np.sqrt(total_loss / N)
                print(f"  [SVD] epoch {epoch:3d}/{self.n_epochs}  train RMSE: {rmse:.4f}")

        return self

    # ------------------------------------------------------------------

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Batch predict ratings (clipped to [1, 5])."""
        assert self._model is not None, "Call fit() first."
        self._model.eval()
        with torch.no_grad():
            u = torch.tensor(user_ids, dtype=torch.long, device=self._device)
            i = torch.tensor(item_ids, dtype=torch.long, device=self._device)
            pred = self._model(u, i).cpu().numpy()
        return np.clip(pred, 1.0, 5.0).astype(np.float32)

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        rated_items: set[int] | None = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        all_items = np.arange(n_items, dtype=int)
        u_arr = np.full(n_items, user_idx, dtype=int)
        scores = self.predict(u_arr, all_items)
        if rated_items:
            scores[list(rated_items)] = -np.inf
        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top]

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SVDBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)
