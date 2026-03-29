# 🎬 Uncertainty-Aware Hybrid Recommender

A movie recommendation system that doesn’t just predict what you might like—it also tells you how sure it is.

This project combines collaborative filtering with NLP in a Bayesian framework to deal with two common problems: cold start and overconfident predictions.

## Why this exists
Most recommendation systems (SVD, ALS, etc.) have two big weaknesses:

* **Cold start:** New users or items → poor recommendations
* **Overconfidence:** They always output a score, even when they’re basically guessing

This system is built to handle both.

### What’s different here
Instead of predicting a single rating, the model predicts a **distribution**.

This means:

* You get a prediction **and** a confidence level
* The system knows when it doesn’t know

## How it works
### 1. Text → embeddings

Movie descriptions are encoded.
These embeddings act as informative priors for item representations.

### 2. Bayesian Matrix Factorization

We use **BPMF**, so:

* Users and movies are not vectors → they are **distributions**
* Each latent factor has a **mean and variance**

### 3. Variational Inference

Since exact inference is not feasible, we optimize using **VI (ELBO)**.

To make this scalable, we rely on:

* **Stochastic Variational Inference (SVI)**
* **Mini-batch training**


## Practical approach
Given the computational cost, the project is developed in stages, so as to advance once the basics are reached in a scalable manner:

1. Start with a **basic probabilistic recommender (BPMF only)**
2. Ensure training and inference are stable
3. Introduce **SVI / minibatching** for scalability
4. Add **S-BERT priors** once the base model works

The goal is to validate the probabilistic model first, before adding complexity.


## Active Learning loop
When the model is uncertain, it can:

* Ask the user to rate specific movies
* Select items that maximize information gain (e.g. high uncertainty / entropy)

This helps especially in cold-start scenarios.


## 📊 Evaluation
We evaluate both **prediction quality** and **uncertainty calibration**:

### Recommendation quality

* **RMSE** → rating prediction accuracy
* **NDCG** → ranking quality

### Uncertainty quality

* **Expected Calibration Error (ECE)** → how well predicted confidence matches reality

### Baseline comparison
To validate improvements, we compare against a simple baseline:

* **SVD (Surprise library)**

This allows us to:

* Measure performance gains
* Show advantages in **high-uncertainty scenarios**
* Especially highlight improvements in **cold start**
