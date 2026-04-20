"""
Bayesian Probabilistic Matrix Factorization via Stochastic Variational Inference.

  U_i ~ N(0, I),  V_j ~ N(0, I)          latent factors
  b_u_i ~ N(0, 1), b_v_j ~ N(0, 1)       biases
  R_ij ~ N(mu + b_u_i + b_v_j + U_i·V_j, sigma_obs^2)

Mean-field Gaussian variational posteriors, optimised by maximising the ELBO.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

class NewUserState:
    """Variational parameters for a new user we haven't seen during training."""

    def __init__(self, mu_u, rho_u, m_bu, rho_bu):
        self.mu_u = mu_u
        self.rho_u = rho_u
        self.m_bu = m_bu
        self.rho_bu = rho_bu

    @property
    def sigma_u(self):
        return self.rho_u.exp()

    @property
    def sigma_bu(self):
        return self.rho_bu.exp()

    @classmethod
    def at_prior(cls, K, device="cpu"):
        return cls(
            mu_u=torch.zeros(K, device=device),
            rho_u=torch.full((K,), -3.0, device=device),
            m_bu=torch.tensor(0.0, device=device),
            rho_bu=torch.tensor(-3.0, device=device),
        )

class BPMF(nn.Module):

    def __init__(self, n_users, n_items, K=20, sigma_obs_init=1.0, global_mean=3.5):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.K = K
        self.global_mean = global_mean

        # learnable aleatoric noise
        self.log_sigma_obs = nn.Parameter(torch.tensor(float(np.log(sigma_obs_init))))

        self.mu_u = nn.Parameter(torch.randn(n_users, K) * 0.01)
        self.rho_u = nn.Parameter(torch.full((n_users, K), -3.0))

        self.mu_v = nn.Parameter(torch.randn(n_items, K) * 0.01)
        self.rho_v = nn.Parameter(torch.full((n_items, K), -3.0))

        self.m_bu = nn.Parameter(torch.zeros(n_users))
        self.rho_bu = nn.Parameter(torch.full((n_users,), -3.0))

        self.m_bv = nn.Parameter(torch.zeros(n_items))
        self.rho_bv = nn.Parameter(torch.full((n_items,), -3.0))

    @property
    def sigma_u(self):
        return self.rho_u.exp()

    @property
    def sigma_v(self):
        return self.rho_v.exp()

    @property
    def sigma_bu(self):
        return self.rho_bu.exp()

    @property
    def sigma_bv(self):
        return self.rho_bv.exp()

    @property
    def sigma_obs(self):
        return self.log_sigma_obs.exp()

    def predict(self, user_ids, item_ids):
        mu_u = self.mu_u[user_ids]
        mu_v = self.mu_v[item_ids]
        s_u = self.sigma_u[user_ids]
        s_v = self.sigma_v[item_ids]
        m_bu = self.m_bu[user_ids]
        m_bv = self.m_bv[item_ids]
        s_bu = self.sigma_bu[user_ids]
        s_bv = self.sigma_bv[item_ids]

        mean = self.global_mean + m_bu + m_bv + (mu_u * mu_v).sum(-1)
        var_factors = (s_u**2 * s_v**2 + mu_u**2 * s_v**2 + mu_v**2 * s_u**2).sum(-1)
        var_total = self.sigma_obs**2 + var_factors + s_bu**2 + s_bv**2

        return mean, var_total

    def elbo(self, user_ids, item_ids, ratings, n_total, kl_weight=1.0):
        B = ratings.shape[0]

        mu_u = self.mu_u[user_ids]
        mu_v = self.mu_v[item_ids]
        s_u = self.sigma_u[user_ids]
        s_v = self.sigma_v[item_ids]
        m_bu = self.m_bu[user_ids]
        m_bv = self.m_bv[item_ids]
        s_bu = self.sigma_bu[user_ids]
        s_bv = self.sigma_bv[item_ids]

        sigma_sq = self.sigma_obs**2

        # E_q[(r - r_hat)^2] = (r - E[r_hat])^2 + Var[r_hat]
        r_hat = self.global_mean + m_bu + m_bv + (mu_u * mu_v).sum(-1)
        var_r_hat = ((s_u**2 * s_v**2 + mu_u**2 * s_v**2 + mu_v**2 * s_u**2).sum(-1) + s_bu**2 + s_bv**2)

        log_lik_per = -0.5 * ((ratings - r_hat)**2 / sigma_sq + var_r_hat / sigma_sq + torch.log(2.0 * torch.pi * sigma_sq))
        log_lik = (n_total / B) * log_lik_per.sum()

        kl_u = 0.5 * (self.sigma_u**2 + self.mu_u**2 - 1.0 - 2.0 * self.rho_u).sum()
        kl_v = 0.5 * (self.sigma_v**2 + self.mu_v**2 - 1.0 - 2.0 * self.rho_v).sum()
        kl_bu = 0.5 * (self.sigma_bu**2 + self.m_bu**2 - 1.0 - 2.0 * self.rho_bu).sum()
        kl_bv = 0.5 * (self.sigma_bv**2 + self.m_bv**2 - 1.0 - 2.0 * self.rho_bv).sum()

        return log_lik - kl_weight * (kl_u + kl_v + kl_bu + kl_bv)

    @torch.no_grad()
    def recommend(self, user_idx, rated_items=None, top_k=10):
        all_items = torch.arange(self.n_items, dtype=torch.long)
        users = torch.full((self.n_items,), user_idx, dtype=torch.long)
        means, variances = self.predict(users, all_items)

        if rated_items:
            means[rated_items] = -float("inf")

        top = torch.argsort(means, descending=True)[:top_k]
        return {
            "item_ids": top.cpu().numpy(),
            "means": means[top].clamp(1.0, 5.0).cpu().numpy(),
            "variances": variances[top].cpu().numpy(),
            "stds": variances[top].sqrt().cpu().numpy(),
        }

    @torch.no_grad()
    def user_uncertainty(self, user_idx):
        s_u = self.sigma_u[user_idx]
        mu_u = self.mu_u[user_idx]
        s_v = self.sigma_v
        mu_v = self.mu_v
        var = (s_u**2 * s_v**2 + mu_u**2 * s_v**2 + mu_v**2 * s_u**2).sum(-1)
        return var.mean().item()

    @torch.no_grad()
    def item_epistemic_variance(self, user_idx):
        s_u = self.sigma_u[user_idx]
        mu_u = self.mu_u[user_idx]
        s_v = self.sigma_v
        mu_v = self.mu_v
        return (s_u**2 * s_v**2 + mu_u**2 * s_v**2 + mu_v**2 * s_u**2).sum(-1)

    def _elbo_new_user(self, mu_u, rho_u, m_bu, rho_bu, item_ids, ratings):
        sigma_u = rho_u.exp()
        sigma_bu = rho_bu.exp()

        # item parameters are frozen — no gradient flows to them
        mu_v = self.mu_v[item_ids].detach()
        s_v = self.sigma_v[item_ids].detach()
        m_bv = self.m_bv[item_ids].detach()
        s_bv = self.sigma_bv[item_ids].detach()
        sigma_sq = (self.sigma_obs**2).detach()

        r_hat = self.global_mean + m_bu + m_bv + (mu_u * mu_v).sum(-1)
        var_r_hat = ((sigma_u**2 * s_v**2 + mu_u**2 * s_v**2 + mu_v**2 * sigma_u**2).sum(-1) + sigma_bu**2 + s_bv**2)

        log_lik = -0.5 * ((ratings - r_hat)**2 / sigma_sq + var_r_hat / sigma_sq  + torch.log(2.0 * torch.pi * sigma_sq) ).sum()

        kl_u = 0.5 * (sigma_u**2 + mu_u**2 - 1.0 - 2.0 * rho_u).sum()
        kl_bu = 0.5 * (sigma_bu**2 + m_bu**2 - 1.0 - 2.0 * rho_bu)

        return log_lik - kl_u - kl_bu

    def _build_item_user_index(self, train_arr):
        index = defaultdict(list)
        for row in train_arr:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            index[i].append((u, r))
        self._item_user_ratings = dict(index)

    def _warm_start_from_neighbors(self, item_ids, ratings, n_neighbors=20):
        """Init the new user's latent vector from similar training users."""
        new_ratings = dict(zip(item_ids, ratings))

        sq_diffs = defaultdict(list)
        for item_idx, r_new in new_ratings.items():
            for uid, r_train in self._item_user_ratings.get(item_idx, []):
                sq_diffs[uid].append((r_new - r_train) ** 2)

        if not sq_diffs:
            dev = self.mu_u.device
            return torch.zeros(self.K, device=dev), torch.tensor(0.0, device=dev)

        mse_per = {u: float(np.mean(d)) for u, d in sq_diffs.items()}
        count_per = {u: len(d) for u, d in sq_diffs.items()}
        neighbors = sorted(mse_per, key=mse_per.__getitem__)[:n_neighbors]

        mse_vals = np.array([mse_per[u] for u in neighbors], dtype=np.float32)
        count_vals = np.array([count_per[u] for u in neighbors], dtype=np.float32)
        weights = count_vals / (mse_vals + 1e-3)
        weights = weights / weights.sum()

        dev = self.mu_u.device
        w = torch.tensor(weights, dtype=torch.float32, device=dev)
        idx = torch.tensor(neighbors, dtype=torch.long, device=dev)

        mu_u_init = (w.unsqueeze(1) * self.mu_u[idx].detach()).sum(0)
        m_bu_init = (w * self.m_bu[idx].detach()).sum()

        return mu_u_init, m_bu_init

    def fit_new_user(self, item_ids, ratings, n_steps=200, lr=0.05):
        if len(item_ids) == 0:
            return NewUserState.at_prior(self.K, device=self.mu_u.device)

        assert len(item_ids) == len(ratings)

        dev = self.mu_u.device

        if hasattr(self, "_item_user_ratings"):
            mu_u_init, m_bu_init = self._warm_start_from_neighbors(item_ids, ratings)
        else:
            mu_u_init = torch.zeros(self.K, device=dev)
            m_bu_init = torch.tensor(0.0, device=dev)

        mu_u = mu_u_init.clone().requires_grad_(True)
        rho_u = torch.full((self.K,), -3.0, device=dev, requires_grad=True)
        m_bu = m_bu_init.clone().requires_grad_(True)
        rho_bu = torch.tensor(-3.0, device=dev, requires_grad=True)

        item_t = torch.tensor(item_ids, dtype=torch.long, device=dev)
        rating_t = torch.tensor(ratings, dtype=torch.float32, device=dev)

        opt = torch.optim.Adam([mu_u, rho_u, m_bu, rho_bu], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()
            elbo = self._elbo_new_user(mu_u, rho_u, m_bu, rho_bu, item_t, rating_t)
            (-elbo).backward()
            opt.step()

        return NewUserState(
            mu_u=mu_u.detach(),
            rho_u=rho_u.detach(),
            m_bu=m_bu.detach(),
            rho_bu=rho_bu.detach(),
        )

    @torch.no_grad()
    def predict_new_user(self, new_user, item_ids):
        mu_u = new_user.mu_u
        sigma_u = new_user.sigma_u
        m_bu = new_user.m_bu
        sigma_bu = new_user.sigma_bu

        mu_v = self.mu_v[item_ids]
        s_v = self.sigma_v[item_ids]
        m_bv = self.m_bv[item_ids]
        s_bv = self.sigma_bv[item_ids]

        mean = self.global_mean + m_bu + m_bv + (mu_u * mu_v).sum(-1)
        var_factors = (
            sigma_u**2 * s_v**2
            + mu_u**2 * s_v**2
            + mu_v**2 * sigma_u**2
        ).sum(-1)
        var_total = self.sigma_obs**2 + var_factors + sigma_bu**2 + s_bv**2

        return mean, var_total

    @torch.no_grad()
    def recommend_new_user(self, new_user, rated_items=None, top_k=10):
        all_items = torch.arange(self.n_items, dtype=torch.long)
        means, variances = self.predict_new_user(new_user, all_items)

        if rated_items:
            means[list(rated_items)] = -float("inf")

        top = torch.argsort(means, descending=True)[:top_k]
        return {
            "item_ids": top.cpu().numpy(),
            "means": means[top].clamp(1.0, 5.0).cpu().numpy(),
            "variances": variances[top].cpu().numpy(),
            "stds": variances[top].sqrt().cpu().numpy(),
        }

    @torch.no_grad()
    def mean_uncertainty_new_user(self, new_user):
        sigma_u = new_user.sigma_u
        mu_u = new_user.mu_u
        s_v = self.sigma_v
        mu_v = self.mu_v

        var = (sigma_u**2 * s_v**2 + mu_u**2 * s_v**2 + mu_v**2 * sigma_u**2).sum(-1)
        return var.sqrt().mean().item()
