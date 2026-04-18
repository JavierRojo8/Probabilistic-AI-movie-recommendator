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
from collections import defaultdict


class NewUserState:
    """Variational posterior for a single new (out-of-vocabulary) user.

    Holds the variational parameters describing:
      q(U)   = N(mu_u,  diag(exp(rho_u)^2))   — latent factor  (K,)
      q(b_u) = N(m_bu,  exp(rho_bu)^2)         — bias           scalar

    These are plain detached tensors — no autograd.
    Produced by BPMF.fit_new_user().
    """

    def __init__(
        self,
        mu_u: torch.Tensor,    # (K,)
        rho_u: torch.Tensor,   # (K,)  log-std
        m_bu: torch.Tensor,    # ()    scalar
        rho_bu: torch.Tensor,  # ()    scalar log-std
    ):
        self.mu_u = mu_u
        self.rho_u = rho_u
        self.m_bu = m_bu
        self.rho_bu = rho_bu

    @property
    def sigma_u(self) -> torch.Tensor:
        return self.rho_u.exp()

    @property
    def sigma_bu(self) -> torch.Tensor:
        return self.rho_bu.exp()

    @classmethod
    def at_prior(cls, K: int, device: torch.device | str = "cpu") -> "NewUserState":
        """Return a state initialised at the prior q = p = N(0, I)."""
        return cls(
            mu_u=torch.zeros(K, device=device),
            rho_u=torch.full((K,), -3.0, device=device),
            m_bu=torch.tensor(0.0, device=device),
            rho_bu=torch.tensor(-3.0, device=device),
        )


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
            "means": means[top_items].clamp(1.0, 5.0).cpu().numpy(),
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

    # ------------------------------------------------------------------
    # New-user online variational inference
    # ------------------------------------------------------------------

    def _elbo_new_user(
        self,
        mu_u: torch.Tensor,    # (K,)  requires_grad
        rho_u: torch.Tensor,   # (K,)  requires_grad
        m_bu: torch.Tensor,    # ()    requires_grad
        rho_bu: torch.Tensor,  # ()    requires_grad
        item_ids: torch.Tensor,  # (N,) long
        ratings: torch.Tensor,   # (N,) float32
    ) -> torch.Tensor:
        """ELBO for a new user given their complete rated-item set.

        Item parameters and sigma_obs are detached (treated as constants).
        No minibatch scaling: we pass all of the user's ratings at once.

        ELBO = E_q[log p(R | U, V, b_u, b_v)]
               - KL(q(U)   || N(0, I))
               - KL(q(b_u) || N(0, 1))
        """
        sigma_u = rho_u.exp()    # (K,)
        sigma_bu = rho_bu.exp()  # ()

        # Frozen item parameters — no gradient flows to them
        mu_v = self.mu_v[item_ids].detach()      # (N, K)
        s_v  = self.sigma_v[item_ids].detach()   # (N, K)
        m_bv = self.m_bv[item_ids].detach()      # (N,)
        s_bv = self.sigma_bv[item_ids].detach()  # (N,)
        sigma_sq = (self.sigma_obs ** 2).detach()

        # Analytical E_q[(r - r_hat)^2] = (r - E[r_hat])^2 + Var[r_hat]
        r_hat_mean = self.global_mean + m_bu + m_bv + (mu_u * mu_v).sum(-1)
        var_r_hat = (
            (sigma_u ** 2 * s_v ** 2 + mu_u ** 2 * s_v ** 2 + mu_v ** 2 * sigma_u ** 2).sum(-1)
            + sigma_bu ** 2
            + s_bv ** 2
        )

        # Expected log-likelihood — sum over all rated items (no scaling needed)
        log_lik = -0.5 * (
            (ratings - r_hat_mean) ** 2 / sigma_sq
            + var_r_hat / sigma_sq
            + torch.log(2.0 * torch.pi * sigma_sq)
        ).sum()

        # KL( N(mu, diag(sigma^2)) || N(0, I) ) = 0.5 * sum(sigma^2 + mu^2 - 1 - 2*rho)
        kl_u  = 0.5 * (sigma_u  ** 2 + mu_u  ** 2 - 1.0 - 2.0 * rho_u).sum()
        kl_bu = 0.5 * (sigma_bu ** 2 + m_bu  ** 2 - 1.0 - 2.0 * rho_bu)

        return log_lik - kl_u - kl_bu

    def _build_item_user_index(self, train_arr: np.ndarray) -> None:
        """Build an inverted index  item_idx → [(user_idx, rating)]  from the training array.

        Called once at startup; result cached as self._item_user_ratings.
        Enables O(1) lookup of all training users who rated a given item.
        """
        index: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for row in train_arr:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            index[i].append((u, r))
        self._item_user_ratings: dict[int, list[tuple[int, float]]] = dict(index)

    def _warm_start_from_neighbors(
        self,
        item_ids: list[int],
        ratings: list[float],
        n_neighbors: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a warm-start (mu_u_init, m_bu_init) from the most similar training users.

        For every item the new user rated, we look up all training users who
        also rated it and score them by rating agreement (inverse MSE).  The
        top-n_neighbors candidates are then combined into a similarity-weighted
        average of their learned variational means.

        This gives the ELBO optimiser a starting point already close to the
        right region of latent space, so even 2–5 ratings produce meaningful
        personalisation.

        Falls back to the prior (zeros) when no co-rated items are found.
        """
        new_ratings = dict(zip(item_ids, ratings))

        # Accumulate per-item squared rating differences for each candidate user
        candidate_sq_diffs: dict[int, list[float]] = defaultdict(list)
        for item_idx, r_new in new_ratings.items():
            for user_idx, r_train in self._item_user_ratings.get(item_idx, []):
                candidate_sq_diffs[user_idx].append((r_new - r_train) ** 2)

        if not candidate_sq_diffs:
            device = self.mu_u.device
            return torch.zeros(self.K, device=device), torch.tensor(0.0, device=device)

        # Mean MSE per candidate + how many items they co-rated with the new user
        mse_per_user   = {u: float(np.mean(diffs)) for u, diffs in candidate_sq_diffs.items()}
        count_per_user = {u: len(diffs)             for u, diffs in candidate_sq_diffs.items()}
        top_users = sorted(mse_per_user, key=mse_per_user.__getitem__)[:n_neighbors]

        # Similarity weights: (co-rating count) / (mse + ε), then normalised.
        # Weighting by count rewards neighbors who share more items with the new user.
        mse_vals   = np.array([mse_per_user[u]   for u in top_users], dtype=np.float32)
        count_vals = np.array([count_per_user[u] for u in top_users], dtype=np.float32)
        weights  = count_vals / (mse_vals + 1e-3)
        weights  = weights / weights.sum()

        device = self.mu_u.device
        w   = torch.tensor(weights, dtype=torch.float32, device=device)  # (n,)
        idx = torch.tensor(top_users, dtype=torch.long,  device=device)  # (n,)

        # Weighted mean of neighbour variational means
        mu_u_init = (w.unsqueeze(1) * self.mu_u[idx].detach()).sum(0)   # (K,)
        m_bu_init = (w * self.m_bu[idx].detach()).sum()                  # ()

        return mu_u_init, m_bu_init

    def fit_new_user(
        self,
        item_ids: list[int],
        ratings: list[float],
        n_steps: int = 200,
        lr: float = 0.05,
    ) -> NewUserState:
        """Fit variational parameters for a new user via ELBO maximisation.

        All item parameters and sigma_obs are frozen; only the four new-user
        tensors (mu_u, rho_u, m_bu, rho_bu) are optimised.

        Args:
            item_ids: indices of movies the user has rated
            ratings:  corresponding ratings (same order)
            n_steps:  Adam steps — 200 is sufficient for ≤ 20 ratings
            lr:       Adam learning rate (higher than training lr is fine
                      because the problem is tiny)

        Returns:
            NewUserState with converged variational parameters.
        """
        if len(item_ids) == 0:
            return NewUserState.at_prior(self.K, device=self.mu_u.device)

        assert len(item_ids) == len(ratings), "item_ids and ratings must have equal length"

        # Warm start: initialise from similar training users if index is available,
        # otherwise fall back to the prior (zeros).
        device = self.mu_u.device

        if hasattr(self, "_item_user_ratings"):
            mu_u_init, m_bu_init = self._warm_start_from_neighbors(item_ids, ratings)
        else:
            mu_u_init = torch.zeros(self.K, device=device)
            m_bu_init = torch.tensor(0.0, device=device)

        mu_u   = mu_u_init.clone().to(device).requires_grad_(True)
        rho_u  = torch.full((self.K,), -3.0, device=device, requires_grad=True)
        m_bu   = m_bu_init.clone().to(device).requires_grad_(True)
        rho_bu = torch.tensor(-3.0, device=device, requires_grad=True)

        item_t   = torch.tensor(item_ids, dtype=torch.long,    device=device)
        rating_t = torch.tensor(ratings,  dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam([mu_u, rho_u, m_bu, rho_bu], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            elbo = self._elbo_new_user(mu_u, rho_u, m_bu, rho_bu, item_t, rating_t)
            (-elbo).backward()
            optimizer.step()

        return NewUserState(
            mu_u=mu_u.detach(),
            rho_u=rho_u.detach(),
            m_bu=m_bu.detach(),
            rho_bu=rho_bu.detach(),
        )

    @torch.no_grad()
    def predict_new_user(
        self,
        new_user: NewUserState,
        item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predictive mean and total variance for a new user over given items.

        Variance decomposes identically to predict():
          Var[r] = sigma_obs^2 + Var[b_u] + Var[b_v_j] + Var[U · V_j]
        """
        mu_u    = new_user.mu_u      # (K,)
        sigma_u = new_user.sigma_u   # (K,)
        m_bu    = new_user.m_bu      # ()
        sigma_bu = new_user.sigma_bu  # ()

        mu_v = self.mu_v[item_ids]      # (B, K)
        s_v  = self.sigma_v[item_ids]   # (B, K)
        m_bv = self.m_bv[item_ids]      # (B,)
        s_bv = self.sigma_bv[item_ids]  # (B,)

        mean = self.global_mean + m_bu + m_bv + (mu_u * mu_v).sum(-1)

        var_factors = (
            sigma_u ** 2 * s_v ** 2
            + mu_u  ** 2 * s_v ** 2
            + mu_v  ** 2 * sigma_u ** 2
        ).sum(-1)
        var_total = self.sigma_obs ** 2 + var_factors + sigma_bu ** 2 + s_bv ** 2

        return mean, var_total

    @torch.no_grad()
    def recommend_new_user(
        self,
        new_user: NewUserState,
        rated_items: list[int] | None = None,
        top_k: int = 10,
    ) -> dict:
        """Top-K recommendations for a new user with uncertainty estimates."""
        all_items = torch.arange(self.n_items, dtype=torch.long)
        means, variances = self.predict_new_user(new_user, all_items)

        if rated_items:
            means[list(rated_items)] = -float("inf")

        top_items = torch.argsort(means, descending=True)[:top_k]
        return {
            "item_ids":  top_items.cpu().numpy(),
            "means":     means[top_items].clamp(1.0, 5.0).cpu().numpy(),
            "variances": variances[top_items].cpu().numpy(),
            "stds":      variances[top_items].sqrt().cpu().numpy(),
        }

    @torch.no_grad()
    def mean_uncertainty_new_user(self, new_user: NewUserState) -> float:
        """Mean epistemic std for a new user across all items (for the UI)."""
        sigma_u = new_user.sigma_u  # (K,)
        mu_u    = new_user.mu_u     # (K,)
        s_v  = self.sigma_v         # (n_items, K)
        mu_v = self.mu_v            # (n_items, K)

        var_factors = (
            sigma_u ** 2 * s_v ** 2
            + mu_u  ** 2 * s_v ** 2
            + mu_v  ** 2 * sigma_u ** 2
        ).sum(-1)  # (n_items,)

        return var_factors.sqrt().mean().item()
