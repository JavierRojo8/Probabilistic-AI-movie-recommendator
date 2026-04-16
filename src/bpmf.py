"""
Bayesian Probabilistic Matrix Factorization (BPMF) via
Stochastic Variational Inference (SVI).

Model
-----
  U_i ~ N(0, I)          user latent factor (prior)
  V_j ~ N(0, I)          item latent factor (prior)
  b_u_i ~ N(0, 1)        user bias
  b_v_j ~ N(0, 1)        item bias
  R_ij ~ N(mu + b_u_i + b_v_j + U_i . V_j,  sigma_obs^2)

Variational distribution (mean-field Gaussian)
----------------------------------------------
  q(U_i)   = N(mu_u[i],   diag(sigma_u[i]^2))
  q(V_j)   = N(mu_v[j],   diag(sigma_v[j]^2))
  q(b_u_i) = N(m_bu[i],   s_bu[i]^2)
  q(b_v_j) = N(m_bv[j],   s_bv[j]^2)

ELBO (Evidence Lower Bound)
----------------------------
  ELBO = E_q[log p(R|U,V,b_u,b_v)] - KL(q||p)

Scaling for SVI: the expected log-likelihood over a minibatch is
multiplied by (n_total / batch_size) so the KL terms remain comparable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BPMF(nn.Module):

    def __init__(
        self,
        n_users: int,
        n_items: int,
        K: int = 20,
        sigma_obs_init: float = 1.0,
        global_mean: float = 3.5,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.K = K
        self.global_mean = global_mean

        # log sigma_obs is learnable (aleatoric noise)
        self.log_sigma_obs = nn.Parameter(torch.tensor(float(np.log(sigma_obs_init))))

        # ---- Variational parameters for latent factors ----
        # sigma = exp(rho)  (log-parameterisation guarantees positivity)
        self.mu_u = nn.Parameter(torch.randn(n_users, K) * 0.01)
        self.rho_u = nn.Parameter(torch.full((n_users, K), -3.0))

        self.mu_v = nn.Parameter(torch.randn(n_items, K) * 0.01)
        self.rho_v = nn.Parameter(torch.full((n_items, K), -3.0))

        # ---- Variational parameters for biases ----
        self.m_bu = nn.Parameter(torch.zeros(n_users))
        self.rho_bu = nn.Parameter(torch.full((n_users,), -3.0))

        self.m_bv = nn.Parameter(torch.zeros(n_items))
        self.rho_bv = nn.Parameter(torch.full((n_items,), -3.0))

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def sigma_u(self) -> torch.Tensor:
        return self.rho_u.exp()

    @property
    def sigma_v(self) -> torch.Tensor:
        return self.rho_v.exp()

    @property
    def sigma_bu(self) -> torch.Tensor:
        return self.rho_bu.exp()

    @property
    def sigma_bv(self) -> torch.Tensor:
        return self.rho_bv.exp()

    @property
    def sigma_obs(self) -> torch.Tensor:
        return self.log_sigma_obs.exp()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, total_variance) for each (user, item) pair.

        The total predictive variance decomposes as:
          Var[r_ij] = sigma_obs^2                 (aleatoric)
                    + Var[b_u_i] + Var[b_v_j]     (bias uncertainty)
                    + Var[U_i . V_j]               (epistemic from factors)

        where Var[U_i . V_j] = sum_k(
            sigma_u_k^2 * sigma_v_k^2
          + mu_u_k^2   * sigma_v_k^2
          + mu_v_k^2   * sigma_u_k^2
        )
        """
        mu_u = self.mu_u[user_ids]       # (B, K)
        mu_v = self.mu_v[item_ids]       # (B, K)
        s_u = self.sigma_u[user_ids]     # (B, K)
        s_v = self.sigma_v[item_ids]     # (B, K)
        m_bu = self.m_bu[user_ids]       # (B,)
        m_bv = self.m_bv[item_ids]       # (B,)
        s_bu = self.sigma_bu[user_ids]   # (B,)
        s_bv = self.sigma_bv[item_ids]   # (B,)

        mean = self.global_mean + m_bu + m_bv + (mu_u * mu_v).sum(-1)

        var_factors = (s_u**2 * s_v**2 + mu_u**2 * s_v**2 + mu_v**2 * s_u**2).sum(-1)
        var_total = self.sigma_obs**2 + var_factors + s_bu**2 + s_bv**2

        return mean, var_total

    # ------------------------------------------------------------------
    # ELBO (SVI objective)
    # ------------------------------------------------------------------

    def elbo(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        ratings: torch.Tensor,
        n_total: int,
        kl_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute a stochastic ELBO estimate for a minibatch.

        Args:
            user_ids: (B,)  long tensor
            item_ids: (B,)  long tensor
            ratings:  (B,)  float tensor
            n_total:  total number of training ratings (for likelihood scaling)
        """
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

        # Analytical E_q[(r - r_hat)^2] = (r - E[r_hat])^2 + Var[r_hat]
        r_hat_mean = self.global_mean + m_bu + m_bv + (mu_u * mu_v).sum(-1)
        var_r_hat = (
            (s_u**2 * s_v**2 + mu_u**2 * s_v**2 + mu_v**2 * s_u**2).sum(-1)
            + s_bu**2
            + s_bv**2
        )

        # Expected log-likelihood per datapoint (Gaussian)
        log_lik_per = -0.5 * (
            (ratings - r_hat_mean) ** 2 / sigma_sq
            + var_r_hat / sigma_sq
            + torch.log(2.0 * torch.pi * sigma_sq)
        )

        # Scale minibatch likelihood to full-dataset size
        log_lik = (n_total / B) * log_lik_per.sum()

        # KL( N(mu, diag(sigma^2)) || N(0, I) ) = 0.5 * sum(sigma^2 + mu^2 - 1 - 2*rho)
        kl_u = 0.5 * (self.sigma_u**2 + self.mu_u**2 - 1.0 - 2.0 * self.rho_u).sum()
        kl_v = 0.5 * (self.sigma_v**2 + self.mu_v**2 - 1.0 - 2.0 * self.rho_v).sum()
        kl_bu = 0.5 * (self.sigma_bu**2 + self.m_bu**2 - 1.0 - 2.0 * self.rho_bu).sum()
        kl_bv = 0.5 * (self.sigma_bv**2 + self.m_bv**2 - 1.0 - 2.0 * self.rho_bv).sum()

        return log_lik - kl_weight * (kl_u + kl_v + kl_bu + kl_bv)

    # ------------------------------------------------------------------
    # Recommendation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def recommend(
        self,
        user_idx: int,
        rated_items: list[int] | None = None,
        top_k: int = 10,
    ) -> dict:
        """Top-K recommendations for a user with uncertainty estimates."""
        all_items = torch.arange(self.n_items, dtype=torch.long)
        user_ids = torch.full((self.n_items,), user_idx, dtype=torch.long)

        means, variances = self.predict(user_ids, all_items)

        if rated_items:
            means[rated_items] = -float("inf")

        top_items = torch.argsort(means, descending=True)[:top_k]
        return {
            "item_ids": top_items.cpu().numpy(),
            "means": means[top_items].cpu().numpy(),
            "variances": variances[top_items].cpu().numpy(),
            "stds": variances[top_items].sqrt().cpu().numpy(),
        }

    @torch.no_grad()
    def user_uncertainty(self, user_idx: int) -> float:
        """Mean epistemic uncertainty for a user across all items (for active learning)."""
        s_u = self.sigma_u[user_idx]          # (K,)
        mu_u = self.mu_u[user_idx]            # (K,)
        s_v = self.sigma_v                    # (n_items, K)
        mu_v = self.mu_v                      # (n_items, K)

        var_factors = (
            s_u**2 * s_v**2
            + mu_u**2 * s_v**2
            + mu_v**2 * s_u**2
        ).sum(-1)  # (n_items,)

        return var_factors.mean().item()

    @torch.no_grad()
    def item_epistemic_variance(self, user_idx: int) -> torch.Tensor:
        """Epistemic variance for a user across every item — used for active learning."""
        s_u = self.sigma_u[user_idx]
        mu_u = self.mu_u[user_idx]
        s_v = self.sigma_v
        mu_v = self.mu_v

        return (s_u**2 * s_v**2 + mu_u**2 * s_v**2 + mu_v**2 * s_u**2).sum(-1)
