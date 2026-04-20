# Bayesian Movie Recommender

Uncertainty-aware movie recommendation system based on **Bayesian Probabilistic Matrix Factorization (BPMF)** with Stochastic Variational Inference.

Instead of predicting a single rating, the model predicts a **distribution** — giving you a score *and* a confidence level. When the model doesn't know, it says so.

**Authors:** Jorge Gómez Azor (202107216) · Javier Rojo Llorens (202113492)  
**Course:** Inteligencia Artificial Probabilística — Máster en IA, Universidad Pontificia Comillas

---

## Results

| Metric | BPMF (ours) | SVD Baseline |
|--------|-------------|--------------|
| RMSE | **0.888** | 0.923 |
| MAE | **0.695** | 0.715 |
| ECE | **0.0113** | — |
| RMSE cold-start | **0.935** | 1.092 |

Cold-start users (≤ 20 training ratings): **+14% RMSE improvement** over SVD.

---

## Quickstart — Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

**Step 1 — Generate the trained artifacts** (only needed once):

```bash
python src/prepare_data.py    # creates data/processed/data.pkl
python src/train.py           # creates checkpoints/bpmf_best.pt
```

**Step 2 — Build and run:**

```bash
docker compose up --build
```

Open **http://localhost:8501**

**Stop:**
```bash
docker compose down
```

---

## Quickstart — Local (without Docker)

**Requirements:** Python 3.11, pip

```bash
pip install -r requirements.txt
```

```bash
# 1. Download and prepare the dataset
python src/prepare_data.py

# 2. Train the BPMF model (saves best checkpoint automatically)
python src/train.py --K 20 --epochs 20 --batch 4096

# 3. Evaluate: BPMF vs SVD, calibration plots, cold-start analysis
python src/evaluate.py

# 4. Launch the Streamlit app
streamlit run app/app.py
```

---

## Project structure

```
src/
  prepare_data.py    — load ML-1M, remap IDs, temporal train/val/test split
  bpmf.py            — BPMF model (mean-field VI, analytical ELBO)
  baseline.py        — Biased-MF (Funk SVD) via minibatch Adam
  metrics.py         — RMSE, MAE, NDCG@K, ECE
  train.py           — SVI training loop with checkpointing
  active_learning.py — uncertainty-sampling candidate selection
  evaluate.py        — full BPMF vs SVD comparison + plots
app/
  app.py             — Streamlit UI
data/
  ml-1m/             — raw MovieLens 1M (downloaded by prepare_data.py)
  processed/         — preprocessed splits (generated)
checkpoints/
  bpmf_best.pt       — best model checkpoint (generated)
results/
  calibration.png
  uncertainty_vs_error.png
  cold_start_comparison.png
  metrics_comparison.csv
```

---

## How it works

### Model: BPMF

User and item latent factors are modelled as **Gaussian distributions**, not point estimates:

```
U_i ~ N(0, I)        item latent factor (prior)
V_j ~ N(0, I)        user latent factor (prior)
R_ij ~ N(mu + b_u + b_v + U_i · V_j,  sigma_obs²)
```

The variational posterior `q(U_i) = N(mu_u, diag(sigma_u²))` is trained by maximising the **ELBO** (Evidence Lower Bound).

### Scalability: Stochastic Variational Inference

Training on 1M ratings uses **SVI with minibatches** (default: 4096). The likelihood term is scaled by `n_total / batch_size` so KL terms stay in proportion. Best checkpoint reached at epoch 5 (val RMSE 0.8673).

### Uncertainty decomposition

The predictive variance decomposes as:

```
Var[r_ij] = sigma_obs²          (aleatoric — irreducible)
           + Var[b_u] + Var[b_v]
           + Var[U_i · V_j]      (epistemic — reducible with more data)
```

### Active Learning

When a user has high epistemic uncertainty, the app suggests movies to rate that would most reduce the model's posterior variance (uncertainty sampling).

### New user inference

A new user's variational parameters are optimised locally in < 1 s (200 Adam steps, frozen global model). Initialised via **neighbour warm-start**: similarity-weighted average of the latent vectors of the 20 most similar training users.

---

## Training options

```bash
python src/train.py --K 20 --epochs 20 --batch 4096 --lr 0.01 --device auto
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--K` | 20 | Latent factor dimension |
| `--epochs` | 20 | Training epochs |
| `--batch` | 4096 | Minibatch size |
| `--lr` | 0.01 | Adam learning rate |
| `--device` | auto | `cpu` or `cuda` |

---

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — 1,000,209 ratings, 6,040 users, 3,706 movies. Downloaded automatically by `prepare_data.py`.
