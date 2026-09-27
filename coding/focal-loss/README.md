# Focal Loss versus Cross-Entropy

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · derivation and ML implementation (NumPy) | ★★☆☆☆ | Medium | MLE · RS · RE · Applied AI | focal-loss, cross-entropy, class-imbalance, numerical-stability, initialisation, gradients | 3 parts / 45 min | Skills interview |
<!-- meta:end -->

## Problem

Consider binary classification with a single real-valued logit $z \in \mathbb{R}$ per example,
predicted probability $p = \sigma(z) = 1/(1+e^{-z})$ (the logistic sigmoid), and label
$y \in \{0, 1\}$. Write $p_t$ for the probability the model assigns to the *true* class: $p_t = p$
if $y=1$, $p_t = 1-p$ if $y=0$. Write $\alpha_t$ for a class-dependent weight, for a fixed constant
$\alpha \in (0, 1)$: $\alpha_t = \alpha$ if $y=1$, $\alpha_t = 1-\alpha$ if $y=0$. Binary
cross-entropy is $\mathrm{CE} = -\log p_t$; *focal loss* is

$$\mathrm{FL} = -\alpha_t\,(1-p_t)^\gamma\,\log p_t, \qquad \gamma \ge 0,$$

for a fixed constant $\gamma$ called the *focusing parameter*. The factor $(1-p_t)^\gamma$ is the
*modulating factor*: it is close to $1$ when $p_t$ is small (the model assigns low probability to
the true class — a *hard* example) and close to $0$ when $p_t$ is close to $1$ (an *easy*,
confidently correct example), so it scales down the loss contributed by examples the model already
gets right. Setting $\gamma=0$ recovers plain $\alpha_t$-weighted cross-entropy exactly, since
$(1-p_t)^0 = 1$ for every $p_t \in (0, 1]$.

### Part 1 — The gradient

Write $s = 2y - 1$, so $s=+1$ when $y=1$ and $s=-1$ when $y=0$; this lets $p_t$ be written as the
single expression $p_t = \sigma(sz)$, covering either label. Derive $\partial\,\mathrm{FL}/\partial z$
in closed form, as a function of $\alpha_t$, $s$, $p_t$ and $\gamma$ alone. Verify algebraically
that setting $\gamma=0$ in your result recovers $\alpha_t(p-y)$, the gradient of plain
$\alpha_t$-weighted cross-entropy. Then, at $\gamma=2$ and with $\alpha_t=1$ (to isolate the
modulating factor's own effect on the gradient, separately from the class weight $\alpha_t$),
compute the ratio $\lvert\partial\,\mathrm{FL}/\partial z\rvert \big/ \lvert\partial\,\mathrm{CE}/\partial z\rvert$
for an *easy* example with $p_t = 0.9$ and for a *hard* one with $p_t = 0.1$, and comment on what
the two ratios say about how strongly focal loss suppresses each example's contribution to the
*gradient*, as against its contribution to the *loss value*, $(1-p_t)^\gamma$ itself.

### Part 2 — A stable implementation

```py
def focal_loss_and_grad(z: np.ndarray, y: np.ndarray, alpha: float = 0.25,
                         gamma: float = 2.0) -> tuple[float, np.ndarray]:
    """z, y: 1-D arrays of the same shape (N,), one logit and one label (0.0 or 1.0) per example.
    Returns (mean_loss, grad), where mean_loss is the batch mean of FL and grad has shape (N,),
    grad[i] = d(mean_loss)/dz[i]. Finite for every finite z, including |z| up to 100."""
```

Implement `focal_loss_and_grad`: the batch mean of $\mathrm{FL}$ over the $N$ examples, and the
gradient of that mean with respect to every entry of `z`. The result must stay finite — no `nan`,
no `inf`, no overflow warning — for every $z$ with $\lvert z\rvert \le 100$ and every
$y \in \{0, 1\}$, for any $\alpha \in (0, 1)$ and $\gamma \ge 0$. Compute $\log p_t$ and $1-p_t$
each through its own numerically stable formula, rather than through subtraction from an
already-computed $p$: $\log p_t = -\mathrm{softplus}(-sz)$, where
$\mathrm{softplus}(x) = \log(1+e^x)$, and $1-p_t = \sigma(-sz)$, itself computed the same stable
way as $p_t$.

With $\alpha=0.25$, $\gamma=2.0$ (the function's defaults) and a batch of two examples, both
labelled $y=1$, one confidently correct ($z = \ln 9 \approx 2.1972$, so $p_t = 0.9$) and one
confidently wrong ($z = -\ln 9 \approx -2.1972$, so $p_t = 0.1$):

```text
z = [2.1972, -2.1972], y = [1, 1]
focal_loss_and_grad(z, y) -> (0.2333, [-0.0004, -0.1378])
```

### Part 3 — Making focal loss work in practice

**(a) Initialisation.** A classifier trained on a rare positive class of prevalence $\pi$ (for
example $\pi = 0.01$: $1\%$ of examples are positive) is usually initialised with every weight
feeding the final logit at $0$ except the bias, which is set to $b = \log\bigl(\pi/(1-\pi)\bigr)$
rather than to $0$. With every other weight at $0$, every example's logit is $z=b$ regardless of
its features. Compute the mean focal loss (at $\alpha=0.25$, $\gamma=2.0$) over a population that
is a fraction $\pi=0.01$ positive, once with $b=0$ and once with $b=\log(\pi/(1-\pi))$, and compute
what fraction of that mean loss the negatives (the abundant class) are responsible for in each
case. Explain, from these numbers, why the first gradient steps are dominated by the abundant easy
negatives when $b=0$, and why the prior-informed bias avoids this.

**(b) Where the gradient goes.** Generate a synthetic imbalanced binary classification dataset with
a $1\%$ positive rate (positives and negatives drawn from two Gaussian clusters with different
means) and fit a logistic-regression model $z = w\cdot x + b$ to it by plain (full-batch) gradient
descent — once minimising mean cross-entropy ($\alpha=0.5,\gamma=0$) and once minimising mean focal
loss ($\alpha=0.25,\gamma=2$) — starting both runs from the same initialisation ($w=0$, $b$ set as
in part (a)). At initialisation, and again after training, measure what fraction of
$\sum_i \lvert\partial(\text{mean loss})/\partial z_i\rvert$ (the total per-example gradient
magnitude) is contributed by *easy* examples, defined as those with $p_t > 0.9$. Report the four
numbers (CE/FL, at initialisation/after training) and state what they show about which examples
each loss listens to.

**(c) Choosing $\gamma$ and $\alpha$.** Using the numbers from (b), explain what $\gamma$ controls,
what $\alpha$ controls, and why the original paper's choice of $\alpha=0.25$ — which puts *less*
weight on the rare positive class than the naive balanced choice $\alpha=0.5$ — does not contradict
the goal of correcting for class imbalance.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Worth confirming with the interviewer: that this is binary, single-logit classification rather than
multi-class (the multi-class generalisation is in the Follow-ups); and that $\alpha$ weights the
positive class directly ($\alpha_t=\alpha$ when $y=1$), the convention used by the original paper,
rather than always weighting whichever class happens to be rarer in a given batch.

### Part 1

Using $p_t = \sigma(sz)$ and $\sigma'(x) = \sigma(x)(1-\sigma(x))$, the chain rule gives
$dp_t/dz = s\,p_t(1-p_t)$ (the extra factor $s$ from differentiating $sz$), and so

$$\frac{d\log p_t}{dz} = \frac{1}{p_t}\cdot s\,p_t(1-p_t) = s(1-p_t), \qquad
\frac{d(1-p_t)^\gamma}{dz} = \gamma(1-p_t)^{\gamma-1}\cdot\bigl(-s\,p_t(1-p_t)\bigr)
= -\gamma s\,p_t(1-p_t)^\gamma.$$

$\alpha_t$ does not depend on $z$, so the product rule applied to
$\mathrm{FL} = -\alpha_t(1-p_t)^\gamma\log p_t$ gives

$$\frac{\partial\,\mathrm{FL}}{\partial z}
= -\alpha_t\left[\frac{d(1-p_t)^\gamma}{dz}\log p_t + (1-p_t)^\gamma\frac{d\log p_t}{dz}\right]
= -\alpha_t\Bigl[-\gamma s\,p_t(1-p_t)^\gamma\log p_t + s(1-p_t)^{\gamma+1}\Bigr],$$

and factoring out $\alpha_t\,s\,(1-p_t)^\gamma$ from both terms in the bracket,

$$\frac{\partial\,\mathrm{FL}}{\partial z}
= \alpha_t\,s\,(1-p_t)^\gamma\,\bigl[\gamma\,p_t\log p_t - (1-p_t)\bigr].$$

**The $\gamma=0$ case.** $(1-p_t)^0=1$ and the bracket becomes $-(1-p_t)$, so
$\partial\,\mathrm{FL}/\partial z\rvert_{\gamma=0} = -\alpha_t\,s\,(1-p_t) = \alpha_t\,s\,(p_t-1)$.
For $y=1$ ($s=1,\,p_t=p$) this is $\alpha_t(p-1) = \alpha_t(p-y)$; for $y=0$ ($s=-1,\,p_t=1-p$) it
is $-\alpha_t\bigl((1-p)-1\bigr) = \alpha_t p = \alpha_t(p-y)$ — the same formula, $\alpha_t(p-y)$,
either way, which confirms the reduction (and shows, as a by-product, that $s(p_t-1)=p-y$
identically — the standard sigmoid-cross-entropy gradient).

**The two ratios.** With $\alpha_t=1$, setting $\gamma=0$ above gives
$\lvert\partial\,\mathrm{CE}/\partial z\rvert = \lvert{-s(1-p_t)}\rvert = 1-p_t$, so

$$\frac{\lvert\partial\,\mathrm{FL}/\partial z\rvert}{\lvert\partial\,\mathrm{CE}/\partial z\rvert}
= (1-p_t)^{\gamma-1}\,\bigl\lvert \gamma\,p_t\log p_t - (1-p_t) \bigr\rvert.$$

At $\gamma=2$: for the easy example ($p_t=0.9$) this is
$0.1 \times \lvert 2(0.9)\log(0.9) - 0.1\rvert \approx 0.1 \times 0.2896 \approx 0.0290$ — the focal
gradient is about $34.5\times$ smaller than the cross-entropy gradient there. For the hard example
($p_t=0.1$) it is $0.9 \times \lvert 2(0.1)\log(0.1) - 0.9\rvert \approx 0.9 \times 1.3605
\approx 1.2245$ — the focal gradient is not suppressed at all; it is slightly *larger* than the
cross-entropy gradient. This is sharper than the loss-value ratio $(1-p_t)^\gamma$ alone would
suggest ($0.01$ and $0.81$ respectively): differentiating the modulating factor contributes an
extra term that vanishes as $p_t\to1$ at least as fast as $(1-p_t)^\gamma$ itself (so an easy
example's gradient is suppressed at least as strongly as its loss value is), but stays of order $1$
as $p_t\to0$ (so a hard example's gradient is barely damped by the modulating factor at all, unlike
its loss value, which is damped very little either — $(1-p_t)^\gamma=0.81$ — but is damped some).

```python
import numpy as np


# Part 1: the closed-form gradient, and the ratio to plain cross-entropy at alpha_t = 1
def closed_form_grad(z: float, y: float, alpha: float, gamma: float) -> float:
    s = 2.0 * y - 1.0
    p = 1.0 / (1.0 + np.exp(-z))
    p_t = p if y == 1.0 else 1.0 - p
    alpha_t = alpha if y == 1.0 else 1.0 - alpha
    return alpha_t * s * (1.0 - p_t) ** gamma * (gamma * p_t * np.log(p_t) - (1.0 - p_t))


def ratio_to_ce(p_t: float, gamma: float) -> float:
    # NOTE: the alpha_t and s factors of both |dFL/dz| and |dCE/dz| are equal (alpha_t = 1 here,
    # |s| = 1 always) and cancel in the ratio; only the (1 - p_t)^(gamma - 1) * |bracket| part
    # depends on p_t and gamma.
    return (1.0 - p_t) ** (gamma - 1.0) * abs(gamma * p_t * np.log(p_t) - (1.0 - p_t))
```

Complexity: $O(1)$ per example — a handful of elementary operations, independent of everything
else in the problem.

### Part 2

Computing $\log p_t$ as `np.log(sigmoid(s * z))` is unsafe in the direction that matters here: for
$sz$ very negative (a confidently wrong prediction), $\sigma(sz)$ can round to a value whose
distance from $0$ has already lost precision, or — more importantly for the symmetric quantity
$1-p_t$ — for $sz$ very *positive* (a confidently *correct* prediction), $\sigma(sz)$ rounds to
something indistinguishable from $1.0$ once $sz \gtrsim 37$, so computing $1-p_t$ as
`1 - sigmoid(s * z)` subtracts two nearly-equal floats and returns exactly $0$ instead of the true,
tiny positive value — and `(1 - p_t) ** gamma` would then always be exactly $0$ for every
confidently correct example, silently wrong rather than merely imprecise. The identity
$\log\sigma(x) = -\log(1+e^{-x}) = -\mathrm{softplus}(-x)$ avoids this: `np.logaddexp(0, -x)`
computes $\log(1+e^{-x})$ directly, and is stable for every finite $x$, since $e^{-x}$ is never
formed when $x$ is very negative and never overflows when $x$ is very positive (`logaddexp`
subtracts the larger of its two arguments before exponentiating, internally). Both $\log p_t$ and
$1-p_t$ are then obtained by calling this one stable primitive on $sz$ and on $-sz$ respectively —
each accurate on its own side of $0.5$ — rather than by deriving one from the other.

```python
def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    # NOTE: log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x)) = -logaddexp(0, -x); this never
    # forms exp of a large positive number, unlike np.log(1 / (1 + np.exp(-x))) computed directly.
    return -np.logaddexp(0.0, -x)


def _focal_terms(z: np.ndarray, y: np.ndarray, alpha: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-example FL and per-example d(FL)/dz, stable for every finite z. Shared by Part 2's
    focal_loss_and_grad and Part 3's population- and dataset-level measurements."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s = 2.0 * y - 1.0                                        # +1 where y == 1, -1 where y == 0
    log_pt = _log_sigmoid(s * z)                               # log(p_t), stable for every finite z
    p_t = np.exp(log_pt)                                        # NOTE: from the same stable log -- never "1 - (1 - p_t)"
    one_minus_pt = np.exp(_log_sigmoid(-s * z))                   # 1 - p_t, its own stable computation
    alpha_t = np.where(y == 1.0, alpha, 1.0 - alpha)
    loss = -alpha_t * one_minus_pt ** gamma * log_pt
    # NOTE: Part 1's closed form, built only from log_pt / p_t / one_minus_pt above -- never
    # recomputes p_t or 1 - p_t by subtracting one from the other.
    grad = alpha_t * s * one_minus_pt ** gamma * (gamma * p_t * log_pt - one_minus_pt)
    return loss, grad


def focal_loss_and_grad(z: np.ndarray, y: np.ndarray, alpha: float = 0.25,
                         gamma: float = 2.0) -> tuple[float, np.ndarray]:
    loss, grad = _focal_terms(z, y, alpha, gamma)
    return float(loss.mean()), grad / len(grad)          # NOTE: mean loss's gradient -- divide by N, not the sum's
```

Complexity: $O(N)$ time and memory for a batch of $N$ examples — one vectorised pass, no
Python-level loop over examples.

### Part 3

**(a) Initialisation.** With $b=0$, every example gets $p=\sigma(0)=0.5$, so $p_t=0.5$ regardless of
label — nothing is *easy* yet, since the modulating factor $(1-p_t)^\gamma = 0.5^\gamma$ treats
every example identically. The mean loss is then a weighted average of the same per-example value,
$-\alpha_t\,0.5^\gamma\log(0.5)$, over the two classes:

$$\overline{\mathrm{FL}}\big|_{b=0} = -0.5^\gamma\log(0.5)\,\bigl[\pi\alpha + (1-\pi)(1-\alpha)\bigr]
\approx 0.1291 \quad (\pi=0.01,\ \alpha=0.25,\ \gamma=2),$$

of which the negatives — simply because they are $99\%$ of the population, not because they are
individually harder or easier than the positives — account for $99.66\%$. With the prior-informed
bias, negatives immediately get $p_t = 1-\pi = 0.99$ (already confidently correct, so
$(1-p_t)^\gamma = \pi^\gamma = 0.0001$ suppresses them) while positives get $p_t=\pi=0.01$ (still as
hard as ever); the mean loss drops to $\approx 0.01128$ — an $11.4\times$ reduction — and the
negatives' share of it collapses to $0.0066\%$. Without the prior bias, the first gradient steps
mostly adjust the model to fit the *average* status of the overwhelming block of negatives (none of
which are yet being down-weighted for being easy, since none of them are confidently classified
either way at $z=0$); with it, the negatives start already easy, so essentially the entire initial
loss — and gradient — comes from the rare positives instead, exactly the signal a detector needs
from its very first step.

**(b) Where the gradient goes.** Both models reach the same accuracy after training: $27$ of the
$30$ positives recovered, with a single false positive, under both losses. (An independently-fit
scikit-learn logistic regression on the same data recovers $28$ of $30$ — one better than either
from-scratch fit, confirming both plain-gradient-descent runs land close to what a linear model can
achieve here, not in some bug-induced local optimum.) So this is not a claim that focal loss
*classifies* better on this data — only a measurement of where each loss's gradient, in aggregate,
came from while getting there. Under cross-entropy, the easy majority's share of the total gradient
magnitude barely moves: an exact $50.0\%$ at initialisation (at $z=b$ for every example, $p=\pi$
everywhere, so positives contribute $(1-\pi)\times\pi N$ and negatives contribute
$\pi\times(1-\pi)N$ to $\sum_i\lvert p_i-y_i\rvert$ — equal, for *any* $\pi$) down to only $49.5\%$
after training, even once $99.4\%$ of all examples are individually easy — because a cross-entropy
gradient never reaches zero for a confidently correct example, it only gets small, and there are
enough of them to add back up to as much as the minority ever contributed. Under focal loss, the
easy majority's share stays close to zero throughout: $0.08\%$ at initialisation, $3.4\%$ after
training — because the extra $(1-p_t)^\gamma$ factor suppresses each easy example's gradient far
more aggressively than it suppresses its loss (Part 1), so even thousands of them cannot outweigh
the roughly thirty examples that remain hard.

**(c) Choosing $\gamma$ and $\alpha$.** $\gamma$ alone already does nearly all of the work measured
in (b): at the same prior-informed initialisation, switching from plain cross-entropy
($\alpha=0.5,\gamma=0$: negatives get $50\%$ of the gradient magnitude) to focusing alone
($\alpha=0.5,\gamma=2$, no class weighting at all) already shrinks the negatives' share to about
$0.028\%$ — smaller, in fact, than under the paper's own $\alpha=0.25,\gamma=2$, where it is
$0.084\%$. That is because $\alpha_t$ for a negative example is $1-\alpha$, which *increases* as
$\alpha$ drops below the neutral value $0.5$: moving from $\alpha=0.5$ to $\alpha=0.25$ raises
$\alpha_t$ for negatives from $0.5$ to $0.75$ (and lowers it for positives from $0.5$ to $0.25$),
partially restoring some of the gradient share that $\gamma$'s focusing had all but removed from
the negatives — without coming remotely close to cross-entropy's $50\%$. In this sense $\alpha=0.25$
is not asking for *more* attention on the rare positives than a neutral, unweighted loss already
gives them once $\gamma$ is doing the focusing; it is a mild correction in the *other* direction,
stopping $\gamma$ from suppressing the hard negatives — the false positives a detector must still
learn to reject — almost entirely.

```python
def population_shares(pi: float, b: float, alpha: float, gamma: float,
                       n: int = 100_000) -> tuple[float, float, float]:
    """n examples, an exact fraction pi of them positive, every logit fixed at b (as if every
    weight but the bias were 0). Returns (mean loss, negatives' share of total loss,
    negatives' share of total |d(mean loss)/dz|)."""
    n_pos = round(pi * n)
    y = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)])
    z = np.full(n, b)
    loss, grad = _focal_terms(z, y, alpha, gamma)
    negative = y == 0.0
    loss_share = float(loss[negative].sum() / loss.sum())
    grad_share = float(np.abs(grad)[negative].sum() / np.abs(grad).sum())
    return float(loss.mean()), loss_share, grad_share


def make_imbalanced_dataset(rng: np.random.Generator, n_neg: int, n_pos: int, d: int,
                             separation: float) -> tuple[np.ndarray, np.ndarray]:
    """n_neg negatives ~ N(0, I_d), followed by n_pos positives ~ N(separation * e_0, I_d)."""
    X_neg = rng.normal(size=(n_neg, d))
    X_pos = rng.normal(size=(n_pos, d))
    X_pos[:, 0] += separation                                  # NOTE: only feature 0 carries signal
    X = np.concatenate([X_neg, X_pos], axis=0)
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    return X, y


def train_logreg(X: np.ndarray, y: np.ndarray, alpha: float, gamma: float, steps: int, lr: float,
                  pi: float) -> tuple[np.ndarray, float]:
    """Plain full-batch gradient descent on z = X @ w + b, minimising mean FL(alpha, gamma). w
    starts at 0; b starts at the prior-informed bias log(pi / (1 - pi)) of part (a)."""
    n, d = X.shape
    w = np.zeros(d)
    b = np.log(pi / (1.0 - pi))
    for _ in range(steps):
        z = X @ w + b
        _, grad = _focal_terms(z, y, alpha, gamma)
        grad = grad / n                                         # NOTE: gradient of the MEAN loss, not the sum
        w -= lr * (X.T @ grad)
        b -= lr * grad.sum()
    return w, b


def easy_share(z: np.ndarray, y: np.ndarray, alpha: float, gamma: float) -> float:
    """Fraction of sum(|d(mean loss)/dz_i|) contributed by examples with p_t > 0.9."""
    _, grad = _focal_terms(z, y, alpha, gamma)
    s = 2.0 * y - 1.0
    p_t = np.exp(_log_sigmoid(s * z))
    magnitude = np.abs(grad)
    return float(magnitude[p_t > 0.9].sum() / magnitude.sum())
```

Complexity: `train_logreg` costs $O(\mathrm{steps}\cdot N\cdot d)$, dominated by the two matrix–vector
products per step; `population_shares` and `easy_share` cost $O(n)$ and $O(N)$ respectively, one
pass each.

### Follow-ups

- **Multi-class focal loss.** With $C$ classes and a softmax over logits, focal loss generalises to
  $\mathrm{FL} = -\alpha_c(1-p_c)^\gamma\log p_c$, where $p_c$ is the softmax probability the model
  assigns to the true class $c$ and $\alpha_c$ a per-class weight; the modulating factor still reads
  off a single scalar, the probability given to the correct class, so it needs no change, while the
  gradient now flows through the softmax Jacobian $\partial p_c/\partial z_k = p_c(\delta_{ck}-p_k)$
  into every logit, not just one.
- **Focal loss and calibration.** A model trained with focal loss tends to produce *less*
  over-confident probabilities than one trained with cross-entropy at comparable accuracy: because
  the modulating factor keeps shrinking the loss (and gradient) even as $p_t \to 1$, there is less
  pressure to keep pushing an already-correct prediction's probability all the way to the boundary,
  which is exactly where cross-entropy's own gradient, $p-y$, is still largest in absolute terms
  among correctly-classified examples close to the decision boundary.
- **Alternatives for imbalance, and when they fit.** Re-weighting by inverse class frequency (or by
  the *effective number of samples*, a softened version of it) is simple and keeps every example,
  but treats every example of a class identically regardless of how hard it individually is —
  unlike focal loss's *per-example* focusing, it cannot tell a genuinely hard negative from a
  confidently correct one. Resampling (oversampling the minority or undersampling the majority)
  works with any loss unmodified, at the cost of either training on repeated examples or discarding
  majority-class data. Hard-example mining explicitly selects the highest-loss examples of each
  batch to back-propagate, which is a discrete, batch-local relative of focal loss's continuous,
  per-example down-weighting, and adds the mining step's own overhead and variance.
- **Evaluate with average precision, not accuracy.** At $\pi=0.01$, a classifier that always
  predicts negative already scores $99\%$ accuracy — the same pathology, at evaluation time, that
  motivates focal loss during training: accuracy is dominated by how well the abundant class is
  handled and barely reflects performance on the rare one. Average precision (the area under the
  precision–recall curve) or recall at a fixed operating threshold are threshold-aware measures of
  the actual trade-off — catching positives against raising false alarms among the negatives — that
  a constant predictor cannot win by default.

<details>
<summary>Checks (runnable)</summary>

```python
import warnings

import scipy.special as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

rng = np.random.default_rng(0)

# --- Part 1: the closed-form gradient against central finite differences of FL itself ---
max_err = 0.0
for _ in range(3000):
    z = rng.uniform(-6, 6)
    y = float(rng.integers(0, 2))
    alpha = rng.uniform(0.05, 0.95)
    gamma = rng.uniform(0, 5)

    def fl_scalar(zz, y=y, alpha=alpha, gamma=gamma):                # straight from the statement's formula
        p = 1.0 / (1.0 + np.exp(-zz))
        p_t = p if y == 1.0 else 1.0 - p
        alpha_t = alpha if y == 1.0 else 1.0 - alpha
        return -alpha_t * (1.0 - p_t) ** gamma * np.log(p_t)

    eps = 1e-6
    finite_diff = (fl_scalar(z + eps) - fl_scalar(z - eps)) / (2 * eps)
    max_err = max(max_err, abs(finite_diff - closed_form_grad(z, y, alpha, gamma)))
assert max_err < 1e-4, max_err

# gamma = 0 reduces to alpha_t * (p - y), for random z, y, alpha
for _ in range(500):
    z = rng.uniform(-6, 6)
    y = float(rng.integers(0, 2))
    alpha = rng.uniform(0.05, 0.95)
    p = 1.0 / (1.0 + np.exp(-z))
    alpha_t = alpha if y == 1.0 else 1.0 - alpha
    assert abs(closed_form_grad(z, y, alpha, 0.0) - alpha_t * (p - y)) < 1e-10

# the two named ratios of the statement
r_easy, r_hard = ratio_to_ce(0.9, 2.0), ratio_to_ce(0.1, 2.0)
assert round(r_easy, 4) == 0.029
assert round(r_hard, 4) == 1.2245
assert r_hard > 1.0 > r_easy                       # robust: hard is amplified, easy is heavily damped
assert round(1.0 / r_easy, 1) == 34.5              # "about 34.5x smaller" for the easy example
# loss-value ratios (1 - p_t)^gamma, quoted in the text for contrast with the gradient ratios above
assert round((1.0 - 0.9) ** 2.0, 2) == 0.01
assert round((1.0 - 0.1) ** 2.0, 2) == 0.81

# --- Part 2: the worked example of the statement ---
z_ex = np.array([np.log(9.0), -np.log(9.0)])
y_ex = np.array([1.0, 1.0])
loss_ex, grad_ex = focal_loss_and_grad(z_ex, y_ex)
assert round(loss_ex, 4) == 0.2333
assert [round(float(v), 4) for v in grad_ex] == [-0.0004, -0.1378]


# an independent implementation, built from scipy's own stable sigmoid/log-sigmoid primitives
def focal_loss_and_grad_scipy(z, y, alpha=0.25, gamma=2.0):
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    s = 2.0 * y - 1.0
    log_pt, p_t, one_minus_pt = sc.log_expit(s * z), sc.expit(s * z), sc.expit(-s * z)
    alpha_t = np.where(y == 1.0, alpha, 1.0 - alpha)
    loss = -alpha_t * one_minus_pt ** gamma * log_pt
    grad = alpha_t * s * one_minus_pt ** gamma * (gamma * p_t * log_pt - one_minus_pt)
    return float(loss.mean()), grad / len(grad)


for _ in range(500):
    n = int(rng.integers(1, 6))
    z = rng.normal(scale=5, size=n)
    y = rng.integers(0, 2, size=n).astype(float)
    alpha, gamma = rng.uniform(0.05, 0.95), rng.uniform(0, 5)
    l1, g1 = focal_loss_and_grad(z, y, alpha, gamma)
    l2, g2 = focal_loss_and_grad_scipy(z, y, alpha, gamma)
    assert abs(l1 - l2) < 1e-9 and np.allclose(g1, g2, atol=1e-9)

# gamma = 0 reduces to alpha-weighted mean CE at the batched-implementation level too
for _ in range(200):
    n = int(rng.integers(1, 8))
    z = rng.normal(scale=4, size=n)
    y = rng.integers(0, 2, size=n).astype(float)
    alpha = rng.uniform(0.05, 0.95)
    loss, grad = focal_loss_and_grad(z, y, alpha=alpha, gamma=0.0)
    p = 1.0 / (1.0 + np.exp(-z))
    alpha_t = np.where(y == 1.0, alpha, 1.0 - alpha)
    expected_loss = np.mean(-alpha_t * np.log(np.where(y == 1.0, p, 1.0 - p)))
    assert abs(loss - expected_loss) < 1e-8
    assert np.allclose(grad, alpha_t * (p - y) / n, atol=1e-8)

# the naive computation the text warns against really does break exactly where it says: at
# sz = 37 the naive "1 - sigmoid(sz)" underflows to exactly 0.0, while the stable formula this
# page uses still returns the true, tiny positive value
naive_one_minus_p_37 = 1.0 - 1.0 / (1.0 + np.exp(-37.0))
assert naive_one_minus_p_37 == 0.0
stable_one_minus_p_37 = float(np.exp(_log_sigmoid(np.array([-37.0])))[0])
assert 0.0 < stable_one_minus_p_37 < 1e-15

# no non-finite values or warnings for |z| <= 100, across many (z, y) pairs and at the extremes
zs = np.linspace(-100, 100, 4001)
ys = (rng.random(zs.shape) < 0.5).astype(float)
with warnings.catch_warnings():
    warnings.simplefilter("error")
    loss_big, grad_big = focal_loss_and_grad(zs, ys)
    assert np.isfinite(loss_big) and np.all(np.isfinite(grad_big))
    for z_extreme in (100.0, -100.0):
        for y_extreme in (0.0, 1.0):
            l, g = focal_loss_and_grad(np.array([z_extreme]), np.array([y_extreme]))
            assert np.isfinite(l) and np.all(np.isfinite(g))

# --- Part 3(a): the initial-loss numbers ---
pi = 0.01
mean_b0, share_b0, _ = population_shares(pi, b=0.0, alpha=0.25, gamma=2.0)
mean_prior, share_prior, _ = population_shares(pi, b=np.log(pi / (1 - pi)), alpha=0.25, gamma=2.0)
assert round(mean_b0, 4) == 0.1291
assert round(mean_prior, 5) == 0.01128
assert round(share_b0 * 100, 2) == 99.66
assert round(share_prior * 100, 4) == 0.0066
assert round(mean_b0 / mean_prior, 1) == 11.4       # "an 11.4x reduction"

# --- Part 3(b): the synthetic dataset, both training runs, and the four share numbers ---
n_pos, n_neg, d, separation = 30, 2970, 4, 5.0
data_rng = np.random.default_rng(3)
X, y = make_imbalanced_dataset(data_rng, n_neg, n_pos, d, separation)
b0 = np.log(pi / (1 - pi))
z_init = X @ np.zeros(d) + b0

share_ce_init = easy_share(z_init, y, alpha=0.5, gamma=0.0)
share_fl_init = easy_share(z_init, y, alpha=0.25, gamma=2.0)
w_ce, b_ce = train_logreg(X, y, alpha=0.5, gamma=0.0, steps=2000, lr=2.0, pi=pi)
# a much larger lr for FL: Part 1 showed its gradients are far smaller in magnitude than CE's, so
# matching CE's number of steps needs a correspondingly larger step size to actually converge
w_fl, b_fl = train_logreg(X, y, alpha=0.25, gamma=2.0, steps=2000, lr=100.0, pi=pi)
z_ce, z_fl = X @ w_ce + b_ce, X @ w_fl + b_fl
share_ce_final = easy_share(z_ce, y, alpha=0.5, gamma=0.0)
share_fl_final = easy_share(z_fl, y, alpha=0.25, gamma=2.0)

assert abs(share_ce_init - 0.5) < 1e-9              # exact, by the pi/(1-pi) symmetry argument above
assert round(share_fl_init, 4) == 0.0008
assert round(share_ce_final, 4) == 0.4953
assert round(share_fl_final, 4) == 0.0337
assert share_fl_init < share_ce_init / 100           # robust: FL starts with far less than CE's share
assert share_fl_final < share_ce_final / 10           # robust: still far less after training

# "even once 99.4% of all examples are individually easy" under the trained CE model
s_ce = 2.0 * y - 1.0
frac_easy_ce_final = float((np.exp(_log_sigmoid(s_ce * z_ce)) > 0.9).mean())
assert round(frac_easy_ce_final * 100, 1) == 99.4


def positive_negative_counts(z):
    p = 1.0 / (1.0 + np.exp(-z))
    tp = int((p[y == 1] > 0.5).sum())
    fp = int((p > 0.5).sum() - tp)
    return tp, fp


assert positive_negative_counts(z_ce) == (27, 1)
assert positive_negative_counts(z_fl) == (27, 1)

# independent sanity check: a separately-fit scikit-learn logistic regression is close to both,
# confirming plain gradient descent found a genuine near-optimum rather than a training bug
oracle = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
tp_oracle = int((oracle.predict_proba(X)[:, 1][y == 1] > 0.5).sum())
assert tp_oracle >= 27 and tp_oracle - 27 <= 2

# --- Part 3(c): gamma alone, and the paper's (alpha, gamma), against plain CE ---
_, _, share_ce_pure = population_shares(pi, b=b0, alpha=0.5, gamma=0.0)
_, _, share_gamma_only = population_shares(pi, b=b0, alpha=0.5, gamma=2.0)
_, _, share_paper = population_shares(pi, b=b0, alpha=0.25, gamma=2.0)
assert round(share_ce_pure * 100, 1) == 50.0
assert round(share_gamma_only * 100, 3) == 0.028
assert round(share_paper * 100, 3) == 0.084
assert share_gamma_only < share_paper < share_ce_pure

# --- Follow-up: trivial accuracy versus average precision ---
assert round(float((y == 0).mean()), 2) == 0.99      # a constant "always negative" predictor's accuracy
ap_ce = average_precision_score(y, 1.0 / (1.0 + np.exp(-z_ce)))
ap_fl = average_precision_score(y, 1.0 / (1.0 + np.exp(-z_fl)))
assert round(ap_ce, 4) == 0.9904
assert round(ap_fl, 4) == 0.9910

print("all checks passed")
```

</details>

</details>
