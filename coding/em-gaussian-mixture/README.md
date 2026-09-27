# EM for a Gaussian Mixture: Derive and Implement

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · derivation and ML implementation (NumPy) | ★★☆☆☆ | Hard | RS · RE · MLE | expectation-maximisation, gaussian-mixture, log-sum-exp, jensen-inequality, k-means, bic | 3 parts / 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

Data $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$ (equivalently an array of shape $(n, d)$) is modelled as
drawn i.i.d. from a mixture of $K$ multivariate Gaussians. Component $k$ ($1 \le k \le K$) has a *weight*
$\pi_k \ge 0$ with $\sum_{k=1}^K \pi_k = 1$, a *mean* $\mu_k \in \mathbb{R}^d$, and a *covariance*
$\Sigma_k \in \mathbb{R}^{d \times d}$ — symmetric and positive-definite ($v^\top \Sigma_k v > 0$ for every
$v \ne 0$), so $\Sigma_k$ is invertible and $|\Sigma_k| > 0$ — giving the density

$$\mathcal N(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac12 (x - \mu)^\top \Sigma^{-1} (x - \mu)\right), \qquad p(x) = \sum_{k=1}^K \pi_k\, \mathcal N(x; \mu_k, \Sigma_k).$$

Write $\theta = (\pi_{1:K}, \mu_{1:K}, \Sigma_{1:K})$ for all the parameters together. The *log-likelihood* of
$X$ under $\theta$ is

$$\ell(\theta) = \sum_{i=1}^n \log \sum_{k=1}^K \pi_k\, \mathcal N(x_i; \mu_k, \Sigma_k).$$

Example with $K = 2$, $d = 1$ (so each $\Sigma_k$ is the $1 \times 1$ matrix $[\sigma_k^2]$),
$\pi = (0.5, 0.5)$, $\mu = (-1, 1)$, $\sigma_1^2 = \sigma_2^2 = 1$, and $X = (-2, 0, 3)$:

```text
N(x; mu_k, sigma_k^2), one column per component:
  x = -2:   0.2420   0.0044
  x =  0:   0.2420   0.2420
  x =  3:   0.0001   0.0540

p(x) = 0.5 * N(x; -1, 1) + 0.5 * N(x; 1, 1):
  p(-2) = 0.1232   p(0) = 0.2420   p(3) = 0.0271

ell(theta) = log(0.1232) + log(0.2420) + log(0.0271) ~= -7.1225
```

Each part below derives or implements one piece of *EM* (expectation-maximisation), the standard algorithm
for maximising $\ell(\theta)$ by alternately estimating which component generated each point and refitting
the components to that estimate.

### Part 1 — Derive EM

Introduce a latent (unobserved) variable $z_i \in \{1, \dots, K\}$ for every point, with $p(z_i = k) = \pi_k$
and $p(x_i \mid z_i = k) = \mathcal N(x_i; \mu_k, \Sigma_k)$, so that marginalising $z_i$ out recovers
$p(x_i) = \sum_k \pi_k \mathcal N(x_i; \mu_k, \Sigma_k)$ and $\ell(\theta) = \sum_i \log p(x_i)$ as above. Fix
a current estimate $\theta^{\mathrm{old}} = (\pi^{\mathrm{old}}, \mu^{\mathrm{old}}, \Sigma^{\mathrm{old}})$.

**(a)** Using Bayes' rule, derive the *responsibility* $\gamma_{ik} = p(z_i = k \mid x_i, \theta^{\mathrm{old}})$
— the posterior probability that component $k$ generated point $i$, under $\theta^{\mathrm{old}}$.

**(b)** The *expected complete-data log-likelihood* is

$$Q(\theta; \theta^{\mathrm{old}}) = \mathbb E_{z \mid X, \theta^{\mathrm{old}}}\bigl[\log p(X, Z \mid \theta)\bigr] = \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik} \bigl[\log \pi_k + \log \mathcal N(x_i; \mu_k, \Sigma_k)\bigr],$$

with $\gamma_{ik}$ from (a) held fixed. Derive $\arg\max_\pi Q$ subject to $\sum_k \pi_k = 1$ using a Lagrange
multiplier for the constraint, and derive $\arg\max_{\mu_k} Q$ and $\arg\max_{\Sigma_k} Q$ — each $k$
independently — by setting the relevant gradient to zero.

**(c)** One *EM iteration* takes $\theta^{\mathrm{old}}$, computes $\gamma$ by (a) — the *E-step* — then sets
$\theta^{\mathrm{new}}$ to the maximiser found in (b) — the *M-step*. Prove
$\ell(\theta^{\mathrm{new}}) \ge \ell(\theta^{\mathrm{old}})$ for every $\theta^{\mathrm{old}}$: for an
arbitrary distribution $q_i$ over $\{1, \dots, K\}$ at every point ($q_i(k) \ge 0$, $\sum_k q_i(k) = 1$), use
*Jensen's inequality* — for a concave function $\varphi$ and a random variable $Y$,
$\varphi(\mathbb E[Y]) \ge \mathbb E[\varphi(Y)]$ — to construct a lower bound $F(\theta, q) \le \ell(\theta)$
that holds for every $\theta$ and every choice of $q$; show it is tight at $\theta = \theta^{\mathrm{old}}$
when $q_i(k) = \gamma_{ik}$; and show that the M-step's maximiser of $Q$ also maximises $F(\cdot, \gamma)$.

### Part 2 — Implement

```py
def fit_gmm(X: np.ndarray, K: int, n_iter: int = 200, tol: float = 1e-8, reg: float = 1e-6,
            seed: int = 0) -> dict:
    """X: (n, d). Fits a K-component full-covariance Gaussian mixture by EM. Returns a dict with
    weights (K,), means (K, d), covs (K, d, d), and log_likelihoods, a list holding ell(theta) at
    every E-step from the initial parameters through the returned ones (so log_likelihoods[-1] is
    ell at the returned weights/means/covs)."""
```

Initialise, using `np.random.default_rng(seed)` for every source of randomness: the $K$ means to $K$ distinct
rows of $X$ chosen by *k-means++* — draw the first uniformly at random from the $n$ rows, then for
$k = 2, \dots, K$ draw the next from the remaining rows with probability proportional to its squared
Euclidean distance to the nearest mean already chosen; every covariance to $\hat\Sigma + \mathrm{reg} \cdot I_d$,
where $\hat\Sigma = \frac1n \sum_i (x_i - \bar x)(x_i - \bar x)^\top$ is $X$'s own covariance and $\bar x$ its
mean; and every weight to $1/K$.

Compute each E-step in log space: form $\log \pi_k + \log \mathcal N(x_i; \mu_k, \Sigma_k)$ for every
$(i, k)$, then combine the $K$ components with the *log-sum-exp* identity — subtracting the row maximum
before exponentiating — rather than exponentiating each log-density directly, so a very negative log-density
never underflows to a $0$ that a later division cannot recover from. Add $\mathrm{reg} \cdot I_d$ to every
covariance again after each M-step. Run E-step, M-step, E-step, ..., appending each E-step's $\ell(\theta)$ to
`log_likelihoods`, and stop — without taking that iteration's M-step — as soon as $\ell$ has improved by less
than `tol` (an absolute change) from the previous E-step, or after `n_iter` E-steps, whichever comes first.

Continuing the example above ($\theta^{\mathrm{old}}$ = the $\pi, \mu, \Sigma$ given there), Part 1(a)'s
responsibilities and Part 1(b)'s updated parameters are:

```text
gamma(x=-2) = (0.9820, 0.0180)        pi_new    = (0.4948, 0.5052)
gamma(x= 0) = (0.5000, 0.5000)        mu_new    = (-1.3180, 1.9509)
gamma(x= 3) = (0.0025, 0.9975)        Sigma_new = ([[0.9238]], [[2.1654]])   # reg already added

ell(theta_old) ~= -7.1225,  ell(theta_new) ~= -6.0408   # improved, as Part 1c proves it always must
```

### Part 3 — Two connections

**(a)** Fix a shared, spherical covariance $\Sigma_k = \sigma^2 I_d$ for every $k$ (equal across components,
not updated by the M-step) and fix positive weights $\pi_k$. For a point $x$ with a *unique* nearest mean
$\mu_{k^\ast} = \arg\min_k \lVert x - \mu_k \rVert_2$ (assume no exact ties), show that the E-step
responsibility $\gamma_k(x) \to \mathbb 1[k = k^\ast]$ as $\sigma \to 0^+$, and that the M-step mean update
then converges to $\mu_k \to \frac{1}{|C_k|} \sum_{i \in C_k} x_i$, where $C_k = \{i : k^\ast(x_i) = k\}$ —
the centroid update of the *k-means* algorithm, which alternates assigning each point to its nearest of $K$
centroids and moving each centroid to the mean of the points assigned to it.

**(b)** Implement model selection by the *Bayesian information criterion*, $\mathrm{BIC} = -2\ell + p \log n$,
using a fitted model's final $\ell$, i.e. `model["log_likelihoods"][-1]`, and
$p = (K - 1) + Kd + Kd(d+1)/2$ — the number of free parameters: $K - 1$ for the weights (one degree of
freedom removed by $\sum_k \pi_k = 1$), $Kd$ for the means, and $Kd(d+1)/2$ for the $K$ symmetric
$d \times d$ covariances, each with $d(d+1)/2$ free entries. Lower BIC is preferred.

```py
def bic(X: np.ndarray, model: dict) -> float:
    """model: a dict as returned by fit_gmm, fitted on this same X. Returns the BIC score."""
```

For example, a model with $K = 2$, $d = 1$, $n = 100$ and $\ell = -140.0$ has
$p = 1 + 2 + 2 \cdot 1 \cdot 2 / 2 = 5$ and
$\mathrm{BIC} = -2 \cdot (-140.0) + 5 \log(100) \approx 280 + 23.03 = 303.03$.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points worth confirming with the interviewer before coding: that $\Sigma_k$ is a full covariance matrix
rather than constrained to diagonal, as the `covs: (K, d, d)` of the signature implies; and the
initialisation and stopping rule to use, fixed here to k-means++ means, a data-covariance-based initial
covariance, and an absolute tolerance on $\ell$.

### Part 1

**(a) The responsibility.** By Bayes' rule,

$$\gamma_{ik} = p(z_i = k \mid x_i, \theta^{\mathrm{old}}) = \frac{p(z_i = k)\, p(x_i \mid z_i = k)}{p(x_i)} = \frac{\pi_k^{\mathrm{old}}\, \mathcal N(x_i; \mu_k^{\mathrm{old}}, \Sigma_k^{\mathrm{old}})}{\sum_{j=1}^K \pi_j^{\mathrm{old}}\, \mathcal N(x_i; \mu_j^{\mathrm{old}}, \Sigma_j^{\mathrm{old}})},$$

the denominator marginalising $z_i$ out over its $K$ values, i.e. $p(x_i)$ itself; consequently
$\sum_k \gamma_{ik} = 1$ for every $i$, since $\gamma_{i,\cdot}$ is a probability distribution over the $K$
values of $z_i$.

**(b) The weights.** The terms of $Q$ depending on $\pi$ are
$\sum_i \sum_k \gamma_{ik} \log \pi_k = \sum_k N_k \log \pi_k$, writing $N_k = \sum_i \gamma_{ik}$ for the
*soft count* of component $k$ (pulling the sum over $i$ inside since $\log \pi_k$ does not depend on $i$).
Maximise subject to $\sum_k \pi_k = 1$ with a Lagrange multiplier $\lambda$:

$$\mathcal L(\pi, \lambda) = \sum_{k=1}^K N_k \log \pi_k + \lambda \Bigl(\sum_{k=1}^K \pi_k - 1\Bigr), \qquad \frac{\partial \mathcal L}{\partial \pi_k} = \frac{N_k}{\pi_k} + \lambda = 0 \ \Longrightarrow\ \pi_k = -\frac{N_k}{\lambda}.$$

Summing over $k$ and using $\sum_k \pi_k = 1$ and $\sum_k N_k = \sum_k \sum_i \gamma_{ik} = \sum_i 1 = n$:
$1 = -\sum_k N_k / \lambda = -n/\lambda$, so $\lambda = -n$ and $\pi_k = N_k / n$.

**(b) The means.** The terms of $Q$ depending on $\mu_k$ are
$\sum_i \gamma_{ik} \log \mathcal N(x_i; \mu_k, \Sigma_k) = -\frac12 \sum_i \gamma_{ik} (x_i - \mu_k)^\top \Sigma_k^{-1} (x_i - \mu_k) + \text{const}$
(the normalising constant, and every other component's terms, do not depend on $\mu_k$). Differentiate and
set to zero:

$$\frac{\partial Q}{\partial \mu_k} = \sum_i \gamma_{ik}\, \Sigma_k^{-1} (x_i - \mu_k) = 0 \ \Longrightarrow\ \sum_i \gamma_{ik}\, x_i = \Bigl(\sum_i \gamma_{ik}\Bigr) \mu_k \ \Longrightarrow\ \mu_k = \frac{\sum_i \gamma_{ik}\, x_i}{N_k},$$

multiplying through by the invertible $\Sigma_k$.

**(b) The covariances.** Work with the precision $\Lambda_k = \Sigma_k^{-1}$ rather than $\Sigma_k$ directly:
using $(x_i - \mu_k)^\top \Lambda_k (x_i - \mu_k) = \operatorname{tr}\bigl(\Lambda_k (x_i - \mu_k)(x_i - \mu_k)^\top\bigr)$
(a scalar equals its own $1 \times 1$ trace) and linearity of the trace to pull the sum over $i$ inside, the
terms of $Q$ depending on $\Lambda_k$ are

$$\sum_i \gamma_{ik} \Bigl[\tfrac12 \log |\Lambda_k| - \tfrac12 \operatorname{tr}\bigl(\Lambda_k (x_i - \mu_k)(x_i - \mu_k)^\top\bigr)\Bigr] = \frac{N_k}{2} \log |\Lambda_k| - \frac12 \operatorname{tr}(\Lambda_k S_k), \qquad S_k = \sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top.$$

Two standard matrix-derivative identities close this out: for symmetric $\Lambda$,
$\partial \log |\Lambda| / \partial \Lambda = \Lambda^{-1}$ (Jacobi's formula, $d|\Lambda| = |\Lambda| \operatorname{tr}(\Lambda^{-1} d\Lambda)$,
read off entrywise), and $\partial \operatorname{tr}(\Lambda S) / \partial \Lambda = S$ for symmetric $S$
(linear in every entry of $\Lambda$). So

$$\frac{\partial Q}{\partial \Lambda_k} = \frac{N_k}{2} \Lambda_k^{-1} - \frac12 S_k = 0 \ \Longrightarrow\ \Lambda_k^{-1} = \frac{S_k}{N_k} \ \Longrightarrow\ \Sigma_k = \frac{1}{N_k} \sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top,$$

since $\Lambda_k^{-1} = \Sigma_k$ by definition.

**(c) Monotonicity.** For an arbitrary distribution $q_i$ over $\{1, \dots, K\}$ at each point, write the
$i$-th summand of $\ell(\theta)$ as an expectation under $q_i$ and apply Jensen's inequality to the concave
$\log$:

$$\log p(x_i) = \log \sum_{k=1}^K q_i(k)\, \frac{\pi_k \mathcal N(x_i; \mu_k, \Sigma_k)}{q_i(k)} \ge \sum_{k=1}^K q_i(k)\, \log \frac{\pi_k \mathcal N(x_i; \mu_k, \Sigma_k)}{q_i(k)},$$

for every $\theta$ and every $q_i$ with $q_i(k) > 0$ wherever $\pi_k \mathcal N(x_i; \mu_k, \Sigma_k) > 0$.
Summing over $i$ gives $\ell(\theta) \ge F(\theta, q)$ for every $\theta$ and every $q = (q_1, \dots, q_n)$,
where $F(\theta, q) := \sum_i \sum_k q_i(k) \bigl[\log \pi_k + \log \mathcal N(x_i; \mu_k, \Sigma_k) - \log q_i(k)\bigr]$.

Jensen's inequality is an equality exactly when the quantity being averaged,
$\pi_k \mathcal N(x_i; \mu_k, \Sigma_k) / q_i(k)$, is the same for every $k$ with $q_i(k) > 0$ (a concave
function of a random variable that takes only one value has zero "curvature loss"). Writing that common
value $c_i$ gives $q_i(k) = \pi_k \mathcal N(x_i; \mu_k, \Sigma_k) / c_i$, and $\sum_k q_i(k) = 1$ forces
$c_i = p(x_i)$, i.e. $q_i(k) = \gamma_{ik}$ from (a). So at $\theta = \theta^{\mathrm{old}}$ and
$q = \gamma^{\mathrm{old}}$, $F(\theta^{\mathrm{old}}, \gamma^{\mathrm{old}}) = \ell(\theta^{\mathrm{old}})$
exactly.

Fixing $q = \gamma^{\mathrm{old}}$,
$F(\theta, \gamma^{\mathrm{old}}) = Q(\theta; \theta^{\mathrm{old}}) - \sum_i \sum_k \gamma_{ik}^{\mathrm{old}} \log \gamma_{ik}^{\mathrm{old}}$,
and the second term does not depend on $\theta$; so maximising $F(\cdot, \gamma^{\mathrm{old}})$ over
$\theta$ is exactly maximising $Q(\cdot; \theta^{\mathrm{old}})$ over $\theta$ — the M-step of (b). Chaining
Jensen's inequality (holds at $\theta^{\mathrm{new}}$ too), the M-step's maximising property, and equality at
$\theta^{\mathrm{old}}$:

$$\ell(\theta^{\mathrm{new}}) \ \ge\ F(\theta^{\mathrm{new}}, \gamma^{\mathrm{old}}) \ \ge\ F(\theta^{\mathrm{old}}, \gamma^{\mathrm{old}}) \ =\ \ell(\theta^{\mathrm{old}}),$$

proves $\ell(\theta^{\mathrm{new}}) \ge \ell(\theta^{\mathrm{old}})$ for every $\theta^{\mathrm{old}}$: one EM
iteration never decreases the log-likelihood.

### Part 2

Combining the $K$ components in log space uses the same identity as a numerically stable softmax: for any
constant $m$ not depending on the index summed over,

$$\log \sum_{k} e^{a_k} = m + \log \sum_k e^{a_k - m},$$

since $e^{a_k} = e^m e^{a_k - m}$ for every $k$, so $e^m$ factors out of the sum and cancels against the
outer $\log(e^m \cdot (\cdot)) = m + \log(\cdot)$. Taking $m = \max_k a_k$ keeps every exponent $\le 0$, so
`np.exp` never overflows even when a raw $\log \pi_k + \log \mathcal N(x_i; \mu_k, \Sigma_k)$ is very
negative. `make_gmm_clusters` below generates the synthetic data used for the rest of this page and for
Part 3: `n_per_cluster` points from each of three well-separated two-dimensional Gaussians sharing one
covariance.

```python
import numpy as np


def make_gmm_clusters(n_per_cluster: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, true_means): X stacks n_per_cluster points drawn from each of 3 two-dimensional
    Gaussians with well-separated means and a shared covariance; true_means holds the 3 generating
    means, shape (3, 2)."""
    rng = np.random.default_rng(seed)
    true_means = np.array([[0.0, 0.0], [8.0, 0.5], [3.5, 7.0]])
    true_cov = np.array([[1.0, 0.3], [0.3, 0.8]])
    X = np.concatenate([rng.multivariate_normal(m, true_cov, size=n_per_cluster) for m in true_means])
    return X, true_means


def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: returns K distinct row indices into X."""
    n = X.shape[0]
    centers = [int(rng.integers(n))]
    dist_sq = np.sum((X - X[centers[0]]) ** 2, axis=1)
    for _ in range(K - 1):
        probs = dist_sq / dist_sq.sum()
        # NOTE: every already-chosen point has distance 0 to itself, hence probability 0 -- never re-picked
        next_idx = int(rng.choice(n, p=probs))
        centers.append(next_idx)
        dist_sq = np.minimum(dist_sq, np.sum((X - X[next_idx]) ** 2, axis=1))
    return np.array(centers)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    # NOTE: a row that is -inf everywhere (every component underflows to 0 density for that point) would
    # otherwise subtract -inf from -inf and give nan; exp(-inf - 0) == 0 leaves the correct value, 0.
    m = np.where(np.isneginf(m), 0.0, m)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _log_gaussian_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    diff = X - mean                                          # (n, d)
    sign, logdet = np.linalg.slogdet(cov)                    # NOTE: log-determinant via slogdet, never det(cov)
    solved = np.linalg.solve(cov, diff.T)                     # NOTE: solve, not an explicit matrix inverse
    mahalanobis = np.sum(diff.T * solved, axis=0)              # (n,): one (x-mu)^T Sigma^-1 (x-mu) per row
    return -0.5 * (d * np.log(2 * np.pi) + logdet + mahalanobis)


def _e_step(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, float]:
    n, K = X.shape[0], len(weights)
    log_joint = np.empty((n, K))
    for k in range(K):
        log_joint[:, k] = np.log(weights[k]) + _log_gaussian_pdf(X, means[k], covs[k])
    log_norm = _logsumexp(log_joint, axis=1)                   # (n,): log p(x_i) under the mixture
    resp = np.exp(log_joint - log_norm[:, None])                # (n, K): gamma_ik
    return resp, float(log_norm.sum())


def _m_step(X: np.ndarray, resp: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    K = resp.shape[1]
    Nk = resp.sum(axis=0)                                       # (K,)
    weights = Nk / n
    means = (resp.T @ X) / Nk[:, None]                           # (K, d)
    covs = np.empty((K, d, d))
    for k in range(K):
        diff = X - means[k]
        covs[k] = (resp[:, k, None] * diff).T @ diff / Nk[k]
        covs[k] += reg * np.eye(d)                                # NOTE: guards a collapsing component's Sigma_k
    return weights, means, covs


def fit_gmm(X: np.ndarray, K: int, n_iter: int = 200, tol: float = 1e-8, reg: float = 1e-6,
            seed: int = 0) -> dict:
    n, d = X.shape
    rng = np.random.default_rng(seed)
    means = X[_kmeans_pp_init(X, K, rng)].copy()                 # (K, d): K distinct data points
    weights = np.full(K, 1.0 / K)
    base_cov = np.cov(X, rowvar=False, ddof=0).reshape(d, d) + reg * np.eye(d)
    covs = np.stack([base_cov.copy() for _ in range(K)])

    log_likelihoods: list = []
    prev_ll = -np.inf
    for iteration in range(n_iter):
        resp, ll = _e_step(X, weights, means, covs)
        log_likelihoods.append(ll)
        # NOTE: stop BEFORE this iteration's M-step, so the returned parameters always match
        # log_likelihoods[-1] exactly -- never one M-step ahead of the last logged value.
        if ll - prev_ll < tol or iteration == n_iter - 1:
            break
        prev_ll = ll
        weights, means, covs = _m_step(X, resp, reg)

    return {"weights": weights, "means": means, "covs": covs, "log_likelihoods": log_likelihoods}
```

**Complexity.** k-means++ initialisation costs $O(nKd)$: each of the $K - 1$ rounds after the first computes
one squared distance per remaining point, $O(nd)$. Each E-step evaluates `_log_gaussian_pdf` once per
component: factoring $\Sigma_k$ for `slogdet`/`solve` costs $O(d^3)$, and solving for all $n$ points' Mahalanobis
terms costs $O(nd^2)$, so $O(K(d^3 + nd^2))$ per E-step. Each M-step's $K$ covariance updates cost $O(nd^2)$
apiece, $O(Knd^2)$ total, dominating the $O(nKd)$ mean update. Over at most `n_iter` iterations, `fit_gmm`
costs $O(\mathrm{n\_iter} \cdot K(nd^2 + d^3))$ — linear in $n$ and $K$, cubic in $d$ through the per-component
linear algebra.

On `X, true_means = make_gmm_clusters(seed=0)` (600 points, 3 true clusters), `fit_gmm(X, 3, seed=0)`
converges in 6 iterations and recovers `true_means` to within $0.08$ (matched up to a permutation of the 3
components). Handing the same initial weights, means and precisions to
`sklearn.mixture.GaussianMixture(covariance_type="full", weights_init=..., means_init=..., precisions_init=...,
reg_covar=reg)`, the two converge to average per-sample log-likelihoods agreeing to within $10^{-6}$: sklearn
reports the *mean* log-likelihood per sample (its `lower_bound_`), where `log_likelihoods[-1]` above is the
*total*, so the comparison divides by $n$ first.

### Part 3

**(a)** With $\Sigma_k = \sigma^2 I_d$ for every $k$,
$\log \mathcal N(x; \mu_k, \sigma^2 I_d) = -\frac{d}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \lVert x - \mu_k \rVert^2$,
so component $k$'s responsibility at a point $x$ is

$$\gamma_k(x) = \frac{\pi_k \exp\bigl(-\lVert x - \mu_k \rVert^2 / 2\sigma^2\bigr)}{\sum_j \pi_j \exp\bigl(-\lVert x - \mu_j \rVert^2 / 2\sigma^2\bigr)},$$

the $-\frac{d}{2}\log(2\pi\sigma^2)$ term cancelling between numerator and denominator (the same for every
$k$). Let $k^\ast = \arg\min_k \lVert x - \mu_k \rVert^2$ (unique by assumption) and divide every term by
$\exp\bigl(-\lVert x - \mu_{k^\ast} \rVert^2 / 2\sigma^2\bigr)$:

$$\gamma_k(x) = \frac{\pi_k \exp\Bigl(-\bigl(\lVert x - \mu_k \rVert^2 - \lVert x - \mu_{k^\ast} \rVert^2\bigr) / 2\sigma^2\Bigr)}{\sum_j \pi_j \exp\Bigl(-\bigl(\lVert x - \mu_j \rVert^2 - \lVert x - \mu_{k^\ast} \rVert^2\bigr) / 2\sigma^2\Bigr)}.$$

For $k \ne k^\ast$, $\lVert x - \mu_k \rVert^2 - \lVert x - \mu_{k^\ast} \rVert^2 > 0$ strictly (uniqueness of
$k^\ast$), so that term's exponent $\to -\infty$ as $\sigma \to 0^+$ and the term $\to 0$; the $k = k^\ast$
term equals $\pi_{k^\ast}$ exactly, at every $\sigma$. Every $\pi_k > 0$ is fixed and independent of
$\sigma$, so it cannot keep pace with an exponential heading to $0$; hence
$\gamma_{k^\ast}(x) \to \pi_{k^\ast} / \pi_{k^\ast} = 1$ and $\gamma_k(x) \to 0$ for $k \ne k^\ast$:
$\gamma(x) \to \mathbb 1[k = k^\ast]$, a one-hot vector.

With $\gamma$ one-hot at every point in the limit, the M-step mean update
$\mu_k = \sum_i \gamma_{ik} x_i / \sum_i \gamma_{ik}$ becomes a sum over exactly the points with
$k^\ast(x_i) = k$, divided by their count: $\mu_k \to \frac{1}{|C_k|} \sum_{i \in C_k} x_i$,
$C_k = \{i : k^\ast(x_i) = k\}$ — the mean of the points nearest to $\mu_k$, exactly k-means's centroid
update, with argmin-distance hard assignment in place of a soft responsibility. A tied point (equidistant
from two or more means) is exactly where the argument breaks:
$\lVert x - \mu_k \rVert^2 - \lVert x - \mu_{k^\ast} \rVert^2 = 0$ for more than one $k$, and $\gamma$ stays
split between them at every $\sigma$, never reaching a one-hot vector — as in the last column below.

```text
means = (0, 0) and (3, 0), weights = (0.5, 0.5), Sigma_k = sigma^2 * I_2

               x = (1, 0)              x = (2.5, 0.5)          x = (1.5, 0), tied: 1.5 from each mean
sigma = 1.0    gamma = (0.8176, 0.1824)  gamma = (0.0474, 0.9526)  gamma = (0.5000, 0.5000)
sigma = 0.5    gamma = (0.9975, 0.0025)  gamma = (0.0000, 1.0000)  gamma = (0.5000, 0.5000)
sigma = 0.1    gamma = (1.0000, 0.0000)  gamma = (0.0000, 1.0000)  gamma = (0.5000, 0.5000)
```

**(b)** A mixture with more components can only fit the training data at least as well, so $\ell$ alone
would always favour the largest $K$ allowed; BIC penalises that gain by $p \log n$, the cost of the extra
parameters. `model["log_likelihoods"][-1]` is exactly $\ell$ at the parameters `fit_gmm` returns, by Part 2's
loop invariant.

```python
def bic(X: np.ndarray, model: dict) -> float:
    n, d = X.shape
    K = model["weights"].shape[0]
    p = (K - 1) + K * d + K * d * (d + 1) // 2
    ell = model["log_likelihoods"][-1]
    return -2 * ell + p * np.log(n)
```

On `make_gmm_clusters(seed=0)`'s 600 points:

```text
K   :  1       2       3       4       5       6
BIC :  6335.6  5132.2  4566.8  4597.0  4627.7  4651.3
```

BIC is minimised at $K = 3$, the true number of clusters, even though $\ell$ itself keeps improving slightly
as $K$ grows past 3: the $p \log n$ penalty for each additional component's $d + d(d+1)/2$ extra parameters
outweighs that gain here.

### Follow-ups

- **Singular likelihood.** Without regularisation, $\ell(\theta)$ has no finite maximum: let one component's
  mean sit exactly on a single data point, $\mu_k = x_i$, and shrink its covariance to $\Sigma_k = \varepsilon I_d$;
  then $\mathcal N(x_i; \mu_k, \Sigma_k) = (2\pi\varepsilon)^{-d/2} \to \infty$ as $\varepsilon \to 0^+$, so
  $\ell(\theta) \to \infty$ too (the other $n - 1$ points still contribute a finite, bounded amount). The
  `reg` term added to every $\Sigma_k$ after each M-step floors every eigenvalue of every $\Sigma_k$ at
  `reg`, ruling out this exact collapse and keeping $\ell$ bounded along the whole trajectory.
- **EM for missing data in general.** A Gaussian mixture is one instance of a general template: for any
  model with observed data $X$ and unobserved $Z$, Part 1(c)'s argument — Jensen's inequality, tight at the
  E-step's posterior, the M-step maximising the same bound — shows $\ell(\theta) = \log p(X \mid \theta)$
  improves every iteration for *any* meaning of $Z$, including genuinely missing entries of $X$ itself
  rather than an auxiliary cluster label; the E-step becomes $p(Z \mid X, \theta^{\mathrm{old}})$ over
  whatever is missing, and the M-step maximises the corresponding $Q$.
- **Variational inference as the generalisation.** Part 1(c)'s bound $F(\theta, q) \le \ell(\theta)$ holds
  for *every* distribution $q$, and EM picks the best one at every E-step — the exact posterior
  $q_i = \gamma_i$ — because that posterior has a closed form here (Bayes' rule over $K$ components). When
  the exact posterior is intractable, variational inference instead restricts $q$ to a tractable family (for
  example one that factorises over the unobserved variables) and maximises the same bound, now called the
  *evidence lower bound* (ELBO), over $\theta$ and $q$ jointly; EM is the special case where $q$ is
  unrestricted and the E-step reaches the bound exactly.
- **Local optima and restarts.** $\ell(\theta)$ is not concave, so Part 1(c) only shows EM does not
  *decrease* $\ell$, never that it reaches the global maximum; `fit_gmm` from different `seed`s can converge
  to different fixed points, especially without well-separated clusters. The standard mitigation is running
  several seeds and keeping the fit with the largest final `log_likelihoods[-1]` — always at a *fixed* $K$,
  restarting to find the best fit; BIC then compares across different values of $K$ using each one's best
  restart.
- **Diagonal versus full covariance.** Constraining every $\Sigma_k$ to be diagonal turns the $O(d^3)$
  `slogdet`/`solve` per component into $O(d)$ elementwise work (a sum of log-variances, and a division by
  each variance instead of a linear solve), and the M-step needs only the $d$ per-feature variances, $O(nd)$
  instead of $O(nd^2)$ — a good trade when the features are close to independent, at the cost of an
  axis-aligned covariance that cannot fit correlation between two features without several rotated,
  diagonal-looking components standing in for the one full-covariance component that would fit it directly.

<details>
<summary>Checks (runnable)</summary>

```python
def loop_e_step(X, weights, means, covs):                          # Part 1, straight from the formula
    n, d = X.shape
    K = len(weights)
    joint = np.zeros((n, K))
    for i in range(n):
        for k in range(K):
            diff = X[i] - means[k]
            inv = np.linalg.inv(covs[k])
            det = np.linalg.det(covs[k])
            density = np.exp(-0.5 * diff @ inv @ diff) / np.sqrt((2 * np.pi) ** d * det)
            joint[i, k] = weights[k] * density
    resp = joint / joint.sum(axis=1, keepdims=True)
    ll = float(np.sum(np.log(joint.sum(axis=1))))
    return resp, ll


def loop_m_step(X, resp, reg):
    n, d = X.shape
    K = resp.shape[1]
    Nk = np.array([sum(resp[i, k] for i in range(n)) for k in range(K)])
    weights = Nk / n
    means = np.zeros((K, d))
    for k in range(K):
        means[k] = sum((resp[i, k] * X[i] for i in range(n)), np.zeros(d)) / Nk[k]
    covs = np.zeros((K, d, d))
    for k in range(K):
        s = np.zeros((d, d))
        for i in range(n):
            diff = (X[i] - means[k]).reshape(-1, 1)
            s += resp[i, k] * (diff @ diff.T)
        covs[k] = s / Nk[k] + reg * np.eye(d)
    return weights, means, covs


# --- the worked example of the statement: intro densities/ell, then one E-step + one M-step ---
X0 = np.array([[-2.0], [0.0], [3.0]])
weights0 = np.array([0.5, 0.5])
means0 = np.array([[-1.0], [1.0]])
covs0 = np.array([[[1.0]], [[1.0]]])

N0 = np.array([[np.exp(_log_gaussian_pdf(X0[i:i + 1], means0[k], covs0[k])[0]) for k in range(2)] for i in range(3)])
assert np.allclose(np.round(N0, 4), [[0.2420, 0.0044], [0.2420, 0.2420], [0.0001, 0.0540]], atol=5e-5)
p0 = N0 @ weights0
assert np.allclose(np.round(p0, 4), [0.1232, 0.2420, 0.0271])
ell0 = float(np.sum(np.log(p0)))
assert round(ell0, 4) == -7.1225

resp0, ll0 = _e_step(X0, weights0, means0, covs0)
assert abs(ll0 - ell0) < 1e-9                                        # the E-step's ell matches the forward computation
assert np.allclose(np.round(resp0, 4), [[0.9820, 0.0180], [0.5000, 0.5000], [0.0025, 0.9975]], atol=5e-5)

w1, m1, c1 = _m_step(X0, resp0, 1e-6)
assert np.allclose(np.round(w1, 4), [0.4948, 0.5052])
assert np.allclose(np.round(m1.ravel(), 4), [-1.3180, 1.9509])
assert np.allclose(np.round(c1.ravel(), 4), [0.9238, 2.1654])
_, ll1 = _e_step(X0, w1, m1, c1)
assert round(ll1, 4) == -6.0408
assert ll1 > ll0                                                     # Part 1c's claim, on this example
print("worked example OK")

# --- brute-force differential test: an independent loop implementation, straight from the formulas ---
rng = np.random.default_rng(3)
for _ in range(200):
    n = int(rng.integers(3, 9))
    d = int(rng.integers(1, 4))
    K = int(rng.integers(1, 4))
    X = rng.normal(size=(n, d)) * 2
    weights = rng.dirichlet(np.ones(K))
    means = rng.normal(size=(K, d)) * 2
    A = rng.normal(size=(K, d, d))
    covs = np.einsum('kij,kaj->kia', A, A) + 0.5 * np.eye(d)[None]   # a random SPD matrix per component

    resp_ref, ll_ref = loop_e_step(X, weights, means, covs)
    resp_ours, ll_ours = _e_step(X, weights, means, covs)
    assert np.allclose(resp_ref, resp_ours, atol=1e-7)
    assert abs(ll_ref - ll_ours) < 1e-5
    assert np.allclose(resp_ours.sum(axis=1), 1.0, atol=1e-10)        # responsibilities always sum to 1

    w_ref, m_ref, c_ref = loop_m_step(X, resp_ours, 1e-6)
    w_ours, m_ours, c_ours = _m_step(X, resp_ours, 1e-6)
    assert np.allclose(w_ref, w_ours, atol=1e-7)
    assert np.allclose(m_ref, m_ours, atol=1e-7)
    assert np.allclose(c_ref, c_ours, atol=1e-7)
print("brute-force differential OK")

# --- on synthetic data: log-likelihood monotone, means recovered up to permutation, vs. sklearn ---
X, true_means = make_gmm_clusters(seed=0)
model = fit_gmm(X, 3, seed=0)
lls = model["log_likelihoods"]
assert len(lls) == 6                                                   # "converges in 6 iterations" above
assert all(lls[i + 1] >= lls[i] - 1e-9 for i in range(len(lls) - 1))
resp_final, _ = _e_step(X, model["weights"], model["means"], model["covs"])
assert np.allclose(resp_final.sum(axis=1), 1.0, atol=1e-10)

from itertools import permutations

best_err = min(np.max(np.abs(model["means"][list(perm)] - true_means)) for perm in permutations(range(3)))
assert round(best_err, 2) == 0.08                                      # "within 0.08" above
print("monotone log-likelihood + mean recovery OK, best_err =", round(best_err, 4))

from sklearn.mixture import GaussianMixture

K, reg = 3, 1e-6
rng0 = np.random.default_rng(0)                                       # NOTE: same seed fit_gmm(..., seed=0) uses,
means_init = X[_kmeans_pp_init(X, K, rng0)].copy()                     # and the same first call it makes with it,
weights_init = np.full(K, 1.0 / K)                                     # so this reproduces fit_gmm's own init exactly
d = X.shape[1]
base_cov = np.cov(X, rowvar=False, ddof=0).reshape(d, d) + reg * np.eye(d)
precisions_init = np.stack([np.linalg.inv(base_cov)] * K)

ours = fit_gmm(X, K, n_iter=300, tol=1e-12, reg=reg, seed=0)
n = X.shape[0]
ours_avg_ll = ours["log_likelihoods"][-1] / n                          # NOTE: sklearn reports the MEAN per-sample
                                                                          # log-likelihood, ours the TOTAL -- divide first
gmm = GaussianMixture(n_components=K, covariance_type="full", max_iter=300, tol=1e-12,
                       reg_covar=reg, weights_init=weights_init, means_init=means_init,
                       precisions_init=precisions_init, n_init=1)
gmm.fit(X)
assert abs(ours_avg_ll - gmm.lower_bound_) < 1e-6
assert np.allclose(np.sort(ours["means"], axis=0), np.sort(gmm.means_, axis=0), atol=1e-4)
print("sklearn agreement OK, |avg log-likelihood diff| =", abs(ours_avg_ll - gmm.lower_bound_))

# --- Part 3b: BIC formula on the worked example from the statement (K=2, d=1, n=100, ell=-140.0) ---
X_bic_example = np.zeros((100, 1))
model_bic_example = {"weights": np.array([0.5, 0.5]), "log_likelihoods": [-140.0]}
assert round(bic(X_bic_example, model_bic_example), 2) == 303.03
print("BIC worked example OK")

# --- Part 3b: BIC selects K = 3 ---
bics = [bic(X, fit_gmm(X, K, seed=0)) for K in range(1, 7)]
assert int(np.argmin(bics)) == 2                                       # K = 3 (0-indexed: K=1 is index 0)
assert bics[2] < bics[1] - 400 and bics[2] < bics[3] - 10              # a comfortable margin either side
print("BIC table (K=1..6):", [round(b, 1) for b in bics])

# --- Part 3a: the small-variance limit, including the tied point that never resolves ---
means_tiny = np.array([[0.0, 0.0], [3.0, 0.0]])
weights_tiny = np.array([0.5, 0.5])
X_tiny = np.array([[1.0, 0.0], [2.5, 0.5], [1.5, 0.0]])                 # the last point is exactly equidistant

expected = {
    1.0: [[0.8176, 0.1824], [0.0474, 0.9526], [0.5000, 0.5000]],
    0.5: [[0.9975, 0.0025], [0.0000, 1.0000], [0.5000, 0.5000]],
    0.1: [[1.0000, 0.0000], [0.0000, 1.0000], [0.5000, 0.5000]],
}
for sigma, want in expected.items():
    covs_tiny = np.stack([np.eye(2) * sigma ** 2, np.eye(2) * sigma ** 2])
    resp_tiny, _ = _e_step(X_tiny, weights_tiny, means_tiny, covs_tiny)
    assert np.allclose(np.round(resp_tiny, 4), want, atol=5e-5)
print("small-variance limit table OK")

# the same limit for the M-step mean update -> k-means centroids, on the 3-cluster data
dists = np.stack([np.linalg.norm(X - m, axis=1) for m in true_means], axis=1)
hard = np.argmin(dists, axis=1)
kmeans_centroids = np.stack([X[hard == k].mean(axis=0) for k in range(3)])
weights_eq = np.full(3, 1 / 3)
for sigma in (1.0, 0.1, 0.01, 0.001):
    covs_shared = np.stack([np.eye(2) * sigma ** 2 for _ in range(3)])
    resp_lim, _ = _e_step(X, weights_eq, true_means, covs_shared)
    _, means_lim, _ = _m_step(X, resp_lim, 1e-6)
    assert np.max(np.abs(means_lim - kmeans_centroids)) < 1e-3
print("k-means limit OK")

# --- Follow-up: without reg, the density at a collapsing component's mean is unbounded ---
x_pt = np.array([[0.0, 0.0]])
mean_pt = np.array([0.0, 0.0])
eps_values = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-12, 1e-20)
logps = [_log_gaussian_pdf(x_pt, mean_pt, eps * np.eye(2))[0] for eps in eps_values]
assert all(logps[i] < logps[i + 1] for i in range(len(logps) - 1))     # strictly increasing as eps shrinks
assert logps[-1] - logps[0] > 40                                        # genuinely diverging, not just "a bit bigger"
assert np.allclose(logps, [-np.log(2 * np.pi * eps) for eps in eps_values], atol=1e-8)
print("singular likelihood OK")

print("all checks passed")
```

</details>

</details>
