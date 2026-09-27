# Backpropagation from Scratch

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · ML implementation (NumPy) | ★★★☆☆ | Medium | RS · RE · MLE · Intern | backpropagation, softmax-cross-entropy, gradient-check, initialisation, sgd-momentum | 3 parts / 45 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

A two-layer network maps a batch of inputs to class scores. Given a batch $X \in \mathbb{R}^{B\times d}$
of $B$ examples in $\mathbb{R}^d$, a hidden layer of $m$ units, and $C$ output classes,

$$Z_1 = XW_1 + b_1, \qquad H = \mathrm{ReLU}(Z_1), \qquad S = HW_2 + b_2,$$

where $\mathrm{ReLU}(z) = \max(z, 0)$ is applied elementwise, $W_1 \in \mathbb{R}^{d\times m}$,
$b_1 \in \mathbb{R}^m$, $W_2 \in \mathbb{R}^{m\times C}$, $b_2 \in \mathbb{R}^C$, and each row of
$S \in \mathbb{R}^{B\times C}$ holds one example's *scores* (equivalently *logits*). Parameters are
collected in a `dict`, `{"W1": W1, "b1": b1, "W2": W2, "b2": b2}`, each value an `np.ndarray` of the
shape just given. The predicted class probabilities are the row-wise *softmax* of $S$,
$\mathrm{probs}_{i,c} = e^{S_{i,c}} / \sum_{c'=0}^{C-1} e^{S_{i,c'}}$, and given integer labels
$y \in \{0,\dots,C-1\}^B$, the loss on the batch is the mean *softmax cross-entropy*,

$$L = \frac{1}{B}\sum_{i=0}^{B-1} -\log \mathrm{probs}_{i,y_i}.$$

### Part 1 — Forward and backward

```py
def forward_backward(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    """params: {"W1": (d, m), "b1": (m,), "W2": (m, C), "b2": (C,)}. X: (B, d). y: (B,) int labels in
    [0, C). Returns (L, grads); grads holds dL/dW1, dL/db1, dL/dW2, dL/db2, each the same shape as the
    corresponding entry of params."""
```

Before writing any code, derive every gradient in matrix form, in this order: $\partial L/\partial S$
first — it has the closed form

$$\frac{\partial L}{\partial S} = \frac{\mathrm{softmax}(S) - Y}{B}, \qquad Y \in \{0,1\}^{B\times C},\ \ Y_{i,c} = \mathbb{1}[c=y_i]$$

($Y$ the matrix of one-hot labels) — then, by the chain rule through each layer in turn,
$\partial L/\partial W_2$, $\partial L/\partial b_2$, $\partial L/\partial H$, $\partial L/\partial Z_1$,
$\partial L/\partial W_1$, $\partial L/\partial b_1$, in that order.

Worked example, $B=2$, $d=2$, $m=2$, $C=2$:

```text
X = [[ 1.0, -1.0],       y = [0, 1]
     [ 0.5,  2.0]]

W1 = [[ 0.5, -0.5],      b1 = [ 0.1, -0.2]
      [-1.0,  1.0]]

W2 = [[ 1.0, -1.0],      b2 = [ 0.0,  0.5]
      [-1.0,  2.0]]

Z1 = [[ 1.6 , -1.7 ],     H = [[1.6 , 0.  ],     S = [[ 1.6, -1.1],
      [-1.65,  1.55]]         [0.  , 1.55]]          [-1.55, 3.6]]

probs = [[0.9370, 0.0630],
         [0.0058, 0.9942]]

forward_backward(params, X, y) -> loss ~= 0.0354
```

### Part 2 — Gradient check

```py
def grad_check(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    """Returns the largest relative error, over every entry of every one of the four parameter arrays,
    between forward_backward's analytic gradient and a central finite difference of the same entry."""
```

For one scalar parameter entry $\theta$, holding every other entry of `params` fixed, the *central
finite difference* is

$$g_n[\theta] = \frac{L(\theta+\epsilon) - L(\theta-\epsilon)}{2\epsilon},$$

where $L(\theta')$ is the loss `forward_backward` reports with that one entry set to $\theta'$. Writing
$g_a[\theta]$ for the matching entry of `forward_backward`'s analytic gradient, the entry's *relative
error* is

$$\mathrm{rel\_err}[\theta] = \frac{|g_a[\theta] - g_n[\theta]|}{\max\bigl(|g_a[\theta]| + |g_n[\theta]|,\ 10^{-12}\bigr)},$$

and `grad_check` returns the maximum of $\mathrm{rel\_err}[\theta]$ over every entry $\theta$ of `W1`,
`b1`, `W2` and `b2`. As a rule of thumb, used below, a maximum under $10^{-7}$ passes and one above
$10^{-4}$ is grounds to suspect a bug. Explain the choice of the default `eps` (why not $10^{-8}$, or
$10^{-2}$?); why `grad_check` needs `float64` parameters and data rather than the `float32` a model
might otherwise train in; and how to guard against a hidden pre-activation that lies within `eps` of the
$\mathrm{ReLU}$ kink at $Z_1=0$, where a difference straddling both sides of the kink is not
trustworthy regardless of whether `forward_backward`'s code is correct.

### Part 3 — Train it, and why the initialisation matters

```py
def make_spirals(n_per_class: int, n_turns: float, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved spiral arms, C = 2 classes. t is drawn so that sqrt(t) is uniform on
    [0, n_turns * 2 * pi]; with r = t / (n_turns * 2 * pi) in [0, 1], class 0 is the point
    (r cos t, r sin t) and class 1 is the same point rotated by pi, (-r cos t, -r sin t); both get
    i.i.d. N(0, noise ** 2) coordinate noise added. Returns (X, y), X: (2 * n_per_class, 2),
    y: (2 * n_per_class,), n_per_class 0s followed by n_per_class 1s."""


def he_init(d: int, m: int, C: int, seed: int) -> dict[str, np.ndarray]:
    """W1, W2 entries drawn i.i.d. from N(0, 2 / fan_in) (fan_in = d for W1, m for W2); b1, b2 zero."""


def zero_init(d: int, m: int, C: int) -> dict[str, np.ndarray]:
    """Every entry of every one of the four parameter arrays is exactly 0.0."""


def train(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, *, steps: int, batch_size: int,
          lr: float, momentum: float, seed: int) -> dict[str, np.ndarray]:
    """Mini-batch SGD with momentum against forward_backward: every step draws batch_size examples of
    (X, y) uniformly with replacement, then updates v <- momentum * v - lr * grad and
    theta <- theta + v for every parameter array (v starts at all zeros). Returns the final params."""
```

The two arms `make_spirals` produces are not linearly separable: no single straight line has every
class-$0$ point on one side and every class-$1$ point on the other, which the checks below confirm
independently by showing that a linear classifier (logistic regression) cannot get much past chance on
it, well short of the target below. Using `he_init`, train to at least $95\%$ **training** accuracy —
the argmax of $S$ against $y$, evaluated on the same `X`, `y` used for training — with a fixed seed
throughout.

Justify `he_init`'s variance, $\mathrm{Var}(W) = 2/\mathrm{fan\_in}$ ($\mathrm{fan\_in}$ is how many
inputs each output unit reads: $d$ for `W1`, since every one of the $m$ hidden units reads all $d$ input
features, and $m$ for `W2`, since every one of the $C$ output units reads all $m$ hidden features), by
deriving what it does to the *second moment* of a layer's activations — $\mathrm{E}[h^2]$, meaningful
even when $\mathrm{E}[h]\ne0$, unlike the variance $\mathrm E[(h-\mathrm E[h])^2]$. For a single ReLU
layer $h=\mathrm{ReLU}(xW)$ ($x$'s entries have some common second moment $s$; $W$'s
$\mathrm{fan\_in}\times\mathrm{fan\_out}$ entries are drawn i.i.d., mean $0$, variance $\sigma^2$,
independent of $x$), find $\mathrm E[h^2]$ as a function of $s$, $\sigma^2$ and $\mathrm{fan\_in}$, and
the $\sigma^2$ that makes it equal $s$ again — then confirm by Monte Carlo that stacking $10$ such
layers keeps $\mathrm E[h^2]$ roughly constant with this $\sigma^2$, against what happens with
$\sigma^2=1/\mathrm{fan\_in}$ instead.

Separately, using `zero_init` in place of `he_init`: from Part 1's gradient formulas, derive exactly
what `forward_backward` returns at $W_1=b_1=W_2=b_2=0$, and what every step of `train` after that does
to each of the four parameter arrays.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Worth confirming before coding: the loss is the *mean*, not the sum, of the per-example cross-entropies
over the batch — assumed here, and it is what fixes the $1/B$ that appears directly in
$\partial L/\partial S$ below rather than leaving the batch size as a free rescaling of the learning
rate — and that `y` holds integer class indices, not pre-one-hot vectors, so `forward_backward` builds
the one-hot matrix internally.

### Part 1

Write $s_{i,c}$ for entry $(i,c)$ of $S$ and $p_{i,c}=\mathrm{probs}_{i,c}$. The per-example loss is
$\ell_i = -\log p_{i,y_i} = -s_{i,y_i} + \log\sum_{c'} e^{s_{i,c'}}$ (substitute the definition of
$p_{i,y_i}$ and take the log). Differentiating with respect to one score of that same example,

$$\frac{\partial \ell_i}{\partial s_{i,c}} = -\mathbb{1}[c=y_i] + \frac{e^{s_{i,c}}}{\sum_{c'}e^{s_{i,c'}}} = p_{i,c} - \mathbb{1}[c=y_i],$$

and $\ell_i$ does not depend on any score of a different example, so $\partial \ell_i/\partial
s_{i',c}=0$ for $i'\ne i$. Averaging over the batch, $\partial L/\partial s_{i,c} =
\frac{1}{B}(p_{i,c}-\mathbb 1[c=y_i])$ — in matrix form, with $Y$ the one-hot label matrix,
$\partial L/\partial S = (P-Y)/B$, the formula the statement gives. Call this matrix $dS$.

Every remaining gradient follows from $dS$ by the chain rule, one layer at a time. $S=HW_2+b_2$ means
$s_{i,c}=\sum_j h_{i,j}w2_{j,c}+b2_c$, so $\partial s_{i,c}/\partial w2_{j,c'}$ equals $h_{i,j}$ when
$c=c'$ and $0$ otherwise (only column $c'$ of $S$ involves $w2_{j,c'}$):

$$\frac{\partial L}{\partial w2_{j,c}} = \sum_{i,c'} \frac{\partial L}{\partial s_{i,c'}}\frac{\partial s_{i,c'}}{\partial w2_{j,c}} = \sum_i (dS)_{i,c}\, h_{i,j} = (H^\top dS)_{j,c}, \qquad\text{so}\quad \frac{\partial L}{\partial W_2} = H^\top dS.$$

The same sum with $\partial s_{i,c}/\partial b2_c = 1$ gives $\partial L/\partial b_2 = \sum_i
(dS)_{i,:}$ — column sums of $dS$, i.e. `dS.sum(axis=0)`. For $H$, $\partial s_{i,c}/\partial
h_{i,j}=w2_{j,c}$ (row $i$ of $H$ only feeds row $i$ of $S$):

$$\frac{\partial L}{\partial h_{i,j}} = \sum_c \frac{\partial L}{\partial s_{i,c}}\, w2_{j,c} = (dS\, W_2^\top)_{i,j}, \qquad\text{so}\quad dH := \frac{\partial L}{\partial H} = dS\, W_2^\top.$$

$H=\mathrm{ReLU}(Z_1)$ acts elementwise, $h_{i,j}=\max(z1_{i,j},0)$, with derivative $\partial
h_{i,j}/\partial z1_{i,j}=\mathbb 1[z1_{i,j}>0]$ (the subgradient $0$ is taken at the kink itself,
$z1_{i,j}=0$ — Part 2 returns to why this one point is a nuisance for gradient *checking*, not for
backprop itself), so $dZ_1 := \partial L/\partial Z_1 = dH \odot \mathbb 1[Z_1>0]$, an elementwise
product. Finally $Z_1=XW_1+b_1$ has exactly the shape of $S=HW_2+b_2$ with $(X,W_1,b_1,dZ_1)$ standing
in for $(H,W_2,b_2,dS)$, so the same two derivations give $\partial L/\partial W_1 = X^\top dZ_1$ and
$\partial L/\partial b_1 = \sum_i (dZ_1)_{i,:}$.

```python
import numpy as np


def _forward(params: dict[str, np.ndarray], X: np.ndarray) -> dict[str, np.ndarray]:
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = X @ W1 + b1
    H = np.maximum(Z1, 0.0)
    S = H @ W2 + b2
    row_max = S.max(axis=1, keepdims=True)          # NOTE: subtract the row max before exp -- keeps every
    exp = np.exp(S - row_max)                        #       exponent <= 0 (no overflow); softmax is unchanged,
    probs = exp / exp.sum(axis=1, keepdims=True)      #       since a per-row constant cancels in the ratio
    return {"Z1": Z1, "H": H, "S": S, "probs": probs}


def forward_backward(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    B = X.shape[0]
    cache = _forward(params, X)
    probs, H, Z1 = cache["probs"], cache["H"], cache["Z1"]
    loss = float(np.mean(-np.log(probs[np.arange(B), y] + 1e-12)))   # NOTE: +1e-12 only guards log(0);
                                                                        #       it does not appear in dS below
    onehot = np.zeros_like(probs)
    onehot[np.arange(B), y] = 1.0
    dS = (probs - onehot) / B          # NOTE: divide by B exactly once, here -- every gradient below inherits
                                        #       this factor through the chain rule; do not divide by B again
    dW2 = H.T @ dS
    db2 = dS.sum(axis=0)
    dH = dS @ params["W2"].T
    dZ1 = dH * (Z1 > 0)                # NOTE: the gate uses the PRE-activation Z1, not H -- using dH directly,
                                        #       without this factor, silently makes ReLU the identity here
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)              # NOTE: bias gradients sum over the batch axis, giving shape (m,), not (B, m)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

On the worked example above, this returns `loss ~= 0.0354` and $dS \approx \begin{pmatrix}-0.0315 &
0.0315\\ 0.0029 & -0.0029\end{pmatrix}$: row $0$ pulls class $0$'s score up and class $1$'s down, since
the true label is $0$ but $\mathrm{probs}_{0,0}=0.9370$ still slightly under-predicts it; row $1$'s pull
is much smaller, since $\mathrm{probs}_{1,1}=0.9942$ is already close to the label. The checks verify
every one of `W1`, `b1`, `W2`, `b2`'s returned gradients on this example against an independent
finite-difference loop.

The forward pass costs $O(Bdm)$ for $XW_1$ and $O(BmC)$ for $HW_2$; the backward pass repeats the same
two products transposed — $H^\top dS$ and $dS\,W_2^\top$ are each $O(BmC)$, $X^\top dZ_1$ is $O(Bdm)$ —
so `forward_backward` costs $O(B(dm+mC))$ time overall, the same order as the forward pass alone (a
general fact about backprop, revisited in the Follow-ups). Memory is $O(B(d+m+C))$ for the cached $X$,
$Z_1$, $H$ (the backward pass reads all three), on top of $O(dm+mC)$ for the parameters themselves.

### Part 2

$g_n$ approximates $g_a$ up to a *truncation error*, from stopping the Taylor expansion of $L$ at a
finite order, and a *rounding error*, from evaluating $L$ in finite-precision arithmetic; the choice of
`eps` trades one against the other. Expanding $L$ around $\theta$,

$$L(\theta \pm \epsilon) = L(\theta) \pm \epsilon L'(\theta) + \frac{\epsilon^2}{2}L''(\theta) \pm \frac{\epsilon^3}{6}L'''(\theta) + O(\epsilon^4),$$

so $L(\theta+\epsilon)-L(\theta-\epsilon) = 2\epsilon L'(\theta) + \frac{\epsilon^3}{3}L'''(\theta) +
O(\epsilon^5)$, and dividing by $2\epsilon$,

$$g_n[\theta] = L'(\theta) + \frac{\epsilon^2}{6}L'''(\theta) + O(\epsilon^4):$$

the truncation error is $O(\epsilon^2)$ — smaller `eps` helps on this term alone, and this term is why
central differences are used at all rather than a one-sided difference
$(L(\theta+\epsilon)-L(\theta))/\epsilon$, whose truncation error is only $O(\epsilon)$ (the odd-order
terms of the expansion do not cancel there). But `float64` arithmetic represents $L(\theta)$ with a
relative error of about the *machine epsilon* $u\approx2.22\times10^{-16}$, so each evaluation carries
an absolute error of about $u|L|$; subtracting two evaluations that agree to within $O(\epsilon)$ loses
precision to this error at a rate of about $u|L|/\epsilon$ — halving `eps` roughly quarters the
truncation term but roughly doubles this one, so the total error $\frac{\epsilon^2}{6}|L'''| +
\frac{u|L|}{\epsilon}$ is smallest where the two terms are comparable, $\epsilon^2 \sim u/\epsilon$,
i.e. $\epsilon \sim u^{1/3} \approx 6\times10^{-6}$ for `float64` — the order of magnitude of the
$10^{-5}$ default. `float32`'s machine epsilon is about $1.19\times10^{-7}$, nine orders of magnitude
larger, giving an optimal $\epsilon \sim (1.19\times10^{-7})^{1/3}\approx4.9\times10^{-3}$: using
$10^{-5}$ there sits deep in the rounding-dominated regime, where $u|L|/\epsilon$ swamps the signal
`grad_check` is trying to measure. The checks below demonstrate this directly, so `grad_check` requires
`float64` parameters and data, raising rather than silently returning a misleading number for `float32`
input.

The kink is a separate issue from precision. $Z_1=XW_1+b_1$ is itself a function of $W_1$ and $b_1$, so
perturbing one of their entries by $\pm\epsilon$ shifts some of $Z_1$'s entries by an amount of order
$\epsilon$; if a hidden unit's pre-activation already lies within `eps` of $0$ for some example, that
shift can flip whether $\mathrm{ReLU}$ is active there between the two evaluations $L(\theta+\epsilon)$
and $L(\theta-\epsilon)$, so $g_n$ no longer estimates the derivative of either linear piece on the two
sides of the kink — a genuine limitation of the central-difference *method* at that point, present
regardless of whether `forward_backward`'s code is correct. `grad_check` checks for this directly, using
the very pre-activations Part 1 already computes, and raises rather than returning a number a caller
could mistake for evidence of a bug.

```python
def grad_check(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    for name, arr in params.items():
        if arr.dtype != np.float64:                            # NOTE: float32 rounding error swamps the
            raise ValueError(f"grad_check requires float64 parameters, got {arr.dtype} for {name!r}")  # eps**2 truncation term derived above -- see the checks for a direct demonstration
    if np.any(np.abs(_forward(params, X)["Z1"]) < eps):          # NOTE: a pre-activation within eps of the
        raise ValueError("a hidden pre-activation lies within eps of the ReLU kink; redraw the inputs")  # kink makes the two-sided difference straddle it -- redraw rather than report a misleading error
    _, analytic = forward_backward(params, X, y)
    worst = 0.0
    for name, arr in params.items():
        grad_a = analytic[name]
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = arr[idx]
            arr[idx] = original + eps
            loss_plus, _ = forward_backward(params, X, y)
            arr[idx] = original - eps
            loss_minus, _ = forward_backward(params, X, y)
            arr[idx] = original                                  # NOTE: restore before moving to the next entry
            grad_n = (loss_plus - loss_minus) / (2 * eps)
            denom = max(abs(grad_a[idx]) + abs(grad_n), 1e-12)
            worst = max(worst, abs(grad_a[idx] - grad_n) / denom)
    return worst
```

`grad_check` calls `forward_backward` twice per parameter entry — $dm+m+mC+C$ entries in total — so it
costs $O\bigl((dm+m+mC+C)\cdot B(dm+mC)\bigr)$ time, quadratic-ish in the network's size; it recomputes
the (unused) analytic gradient at every perturbed point too, since reusing `forward_backward` rather than
a loss-only variant keeps the two code paths manifestly in sync, at a cost that only matters because
`grad_check` is meant for the tiny networks used here, never for a model actually being trained.

### Part 3

**He initialisation.** Consider one ReLU layer in isolation, $h=\mathrm{ReLU}(z)$, $z=xW$ (biases only
shift the pre-activation's distribution and are $0$ throughout, in both the derivation and `he_init`, so
they are dropped here), where $x\in\mathbb R^{\mathrm{fan\_in}}$'s entries share a second moment
$s=\mathrm E[x_j^2]$ and $W$'s $\mathrm{fan\_in}\times\mathrm{fan\_out}$ entries are drawn i.i.d. from a
distribution symmetric about $0$ (`he_init` uses a Gaussian) with variance $\sigma^2$, independent of
$x$. For one output feature $z_k=\sum_{j=1}^{\mathrm{fan\_in}} x_jW_{jk}$: since $\mathrm E[W_{jk}]=0$
and $W_{jk}$ is independent of $x_j$, $\mathrm E[z_k]=\sum_j\mathrm E[x_j]\mathrm E[W_{jk}]=0$ regardless
of $x$'s own mean, so $\mathrm{Var}(z_k)=\mathrm E[z_k^2]$. Expanding the square,

$$\mathrm E[z_k^2] = \sum_{j,j'}\mathrm E[x_jx_{j'}W_{jk}W_{j'k}];$$

for $j\ne j'$, $W_{jk}$ and $W_{j'k}$ are independent of each other and of $x_j,x_{j'}$, with
$\mathrm E[W_{jk}]=\mathrm E[W_{j'k}]=0$, so that term is $\mathrm E[x_jx_{j'}]\,\mathrm
E[W_{jk}]\,\mathrm E[W_{j'k}]=0$; for $j=j'$, independence of $W_{jk}$ and $x_j$ gives $\mathrm
E[x_j^2W_{jk}^2]=\mathrm E[x_j^2]\,\mathrm E[W_{jk}^2]=s\sigma^2$. So $\mathrm
E[z_k^2]=\mathrm{fan\_in}\cdot s\sigma^2$ — a clean recursion on second moments, regardless of $x$'s own
distribution.

Because $W_{jk}$'s distribution is symmetric about $0$ and independent of $x_j$, so is the product
$x_jW_{jk}$ (it has the same distribution as $x_j(-W_{jk})=-(x_jW_{jk})$, since $-W_{jk}$ has the same
distribution as $W_{jk}$), and a sum of independent variables each symmetric about $0$ is itself
symmetric about $0$ — so $z_k$ is symmetric about $0$ exactly, not just approximately. For a distribution
symmetric about $0$, $\mathrm E[z^2\mathbb 1[z>0]]=\mathrm E[z^2\mathbb 1[z<0]]$ (substitute $z\to-z$
inside the expectation: $z^2$ is unchanged, $\mathbb 1[z>0]$ becomes $\mathbb 1[z<0]$), and the two sum
to $\mathrm E[z^2]$ (an entry is almost surely never exactly $0$), so each is exactly half of it:

$$\mathrm E[h^2] = \mathrm E\bigl[\max(z,0)^2\bigr] = \mathrm E[z^2\mathbb 1[z>0]] = \tfrac12\mathrm E[z^2] = \tfrac12\,\mathrm{fan\_in}\cdot s\sigma^2.$$

Setting $\mathrm E[h^2]=s$ — the layer preserves the activation second moment — gives
$\sigma^2=2/\mathrm{fan\_in}$: He initialisation. Using $\sigma^2=1/\mathrm{fan\_in}$ instead (the
variance that would preserve a *linear* layer's output variance, ignoring $\mathrm{ReLU}$'s factor of
$\tfrac12$) gives $\mathrm E[h^2]=s/2$: the second moment is cut in half at every layer, so after $10$
stacked layers it is about $2^{-10}\approx0.001$ of where it started — confirmed by Monte Carlo below.

```python
def he_init(d: int, m: int, C: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(scale=np.sqrt(2.0 / d), size=(d, m)),      # fan_in = d
        "b1": np.zeros(m),
        "W2": rng.normal(scale=np.sqrt(2.0 / m), size=(m, C)),      # fan_in = m
        "b2": np.zeros(C),
    }


def zero_init(d: int, m: int, C: int) -> dict[str, np.ndarray]:
    return {"W1": np.zeros((d, m)), "b1": np.zeros(m), "W2": np.zeros((m, C)), "b2": np.zeros(C)}
```

**All-zero initialisation.** Take Part 1's gradient formulas at $W_1=b_1=W_2=b_2=0$. $Z_1=XW_1+b_1=0$
for every row regardless of $X$, so $H=\mathrm{ReLU}(0)=0$ too, for every hidden unit and every example
— every hidden unit computes the same, identically-zero function of the input, which is what "the hidden
units stay identical" means at its most literal. This propagates all the way back through
`forward_backward`: $dW_2=H^\top dS=0$ (the zero matrix times anything is zero, whatever $dS$ turns out
to be), $dH=dS\,W_2^\top=0$ (since $W_2=0$), $dZ_1=dH\odot\mathbb 1[Z_1>0]=0$ ($\mathbb 1[0>0]$ is
`False`), $dW_1=X^\top dZ_1=0$, $db_1=\sum_i(dZ_1)_{i,:}=0$. Only $db_2=\sum_i(dS)_{i,:}$ can be nonzero,
since it never multiplies by $H$ or $W_2$.

By induction, this holds at *every* training step, not just the first: if $W_1^{(t)}=b_1^{(t)}=0$ at
step $t$ (true at $t=0$ by construction), the paragraph above gives $dW_1^{(t)}=db_1^{(t)}=dW_2^{(t)}=0$
regardless of $W_2^{(t)}$, $b_2^{(t)}$, so momentum's own recursion
$v^{(t+1)}=\mathrm{momentum}\cdot v^{(t)}-\mathrm{lr}\cdot0=\mathrm{momentum}\cdot v^{(t)}$ keeps
$v_{W_1},v_{b_1},v_{W_2}$ at their starting value $\mathbf0$, so $W_1^{(t+1)}=W_1^{(t)}+0=0$ and likewise
$b_1$, $W_2$ — exactly the inductive hypothesis, one step later. So `zero_init` freezes `W1`, `b1` and
`W2` at exactly $\mathbf0$ for the entire run, and the only parameter training can move at all is `b2`,
which drifts to fit the label marginal: with $H\equiv0$, $S=HW_2+b_2=b_2$ is the same vector of scores
for every example regardless of $X$, so $b_2$ receives a gradient from $y$'s class frequencies and
nothing else. No architecture, learning rate, momentum or number of steps can break this symmetry, since
it is a property of the *gradient formulas themselves* at this one point — only the initial values can.

```python
def make_spirals(n_per_class: int, n_turns: float, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.random(n_per_class)) * n_turns * 2 * np.pi    # sqrt: denser near the centre, like arc length
    r = t / (n_turns * 2 * np.pi)                                   # r in [0, 1], grows with the angle
    arm0 = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    arm1 = np.stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)], axis=1)   # the same arm, rotated by pi
    X = np.concatenate([arm0, arm1]) + rng.normal(scale=noise, size=(2 * n_per_class, 2))
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y


def train(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, *, steps: int, batch_size: int,
          lr: float, momentum: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = {k: v.copy() for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    N = X.shape[0]
    for _ in range(steps):
        idx = rng.integers(0, N, size=batch_size)     # NOTE: with replacement -- the simplest correct mini-batching
        _, grads = forward_backward(params, X[idx], y[idx])
        for k in params:
            velocity[k] = momentum * velocity[k] - lr * grads[k]
            params[k] = params[k] + velocity[k]
    return params


def accuracy(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(_forward(params, X)["probs"], axis=1)
    return float(np.mean(pred == y))
```

Momentum changes what learning rate is safe: for a roughly steady gradient $g$, momentum's own recursion
$v^{(t+1)}=\mathrm{momentum}\cdot v^{(t)}-\mathrm{lr}\cdot g$ converges to the fixed point
$v_\infty=-\mathrm{lr}\cdot g/(1-\mathrm{momentum})$ (solve $v_\infty=\mathrm{momentum}\cdot
v_\infty-\mathrm{lr}\cdot g$ for $v_\infty$), so at $\mathrm{momentum}=0.9$ the raw `lr` acts through an
effective step ten times larger; `lr`$=0.15$ (effective step $1.5$) was chosen empirically to sit
comfortably inside the range that still converges rather than oscillates.

With `he_init(2, 64, 2, seed=1)`, `make_spirals(150, n_turns=1.5, noise=0.08, seed=0)` and
`train(..., steps=4000, batch_size=32, lr=0.15, momentum=0.9, seed=2)`, training accuracy reaches
$99\%$, comfortably past the $95\%$ target; with `zero_init(2, 64, 2)` in place of `he_init` and
everything else unchanged, `W1`, `b1` and `W2` come back bit-for-bit equal to their starting zero, and
accuracy is exactly $50\%$ — chance, on this perfectly class-balanced dataset, since the frozen network
predicts the same class for every input.

### Follow-ups

- **Reverse-mode versus forward-mode differentiation.** For a scalar loss and $P$ parameters,
  reverse-mode (what `forward_backward` does) computes the entire gradient in one backward pass costing
  about as much as the forward pass itself — $O(1)$ passes, independent of $P$ — by applying the chain
  rule output-to-input from the single scalar $\partial L/\partial L=1$. Forward-mode instead propagates
  a directional derivative alongside every intermediate value, input-to-output, needing one full pass per
  parameter (or per input direction), $O(P)$ passes; reverse-mode wins whenever there are many parameters
  and few outputs, exactly this page's setting, while forward-mode wins with few inputs and many outputs,
  such as a Jacobian-vector product along a handful of directions.
- **Memory and activation checkpointing.** The backward pass needs every intermediate activation the
  forward pass produced — `_forward` caches `Z1`, `H` and `X` because `forward_backward`'s gradients read
  all three — so memory scales with network depth times activation size, on top of the parameters
  themselves. Activation checkpointing trades this for compute: keep only a subset of activations (say,
  one every few layers) and recompute the ones in between from the nearest kept activation during the
  backward pass, turning $O(\text{depth})$ memory into roughly $O(\sqrt{\text{depth}})$ at the cost of a
  second, partial forward pass.
- **Vanishing and exploding gradients with depth.** Backprop through $L$ stacked layers multiplies
  roughly $L$ Jacobians together; if their typical scale is $\rho\ne1$, the product scales like
  $\rho^L$ — vanishing towards $0$ or blowing up exponentially in $L$ once $\rho$ drifts even slightly
  from $1$. This is the same variance-preservation argument behind He initialisation above, applied to
  the *backward* pass instead of the forward one that the Monte Carlo checks: a layer whose forward
  second moment is preserved is not automatically one whose backward gradient magnitude is too, which is
  why some initialisation schemes are stated in terms of the backward pass, or an average of the two.
- **Weight decay.** Adding $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss adds $\lambda\theta$ to
  every parameter's gradient, elementwise and independent of the data — one extra line,
  `dW1 = dW1 + lambda_ * W1` and likewise for `dW2`, before the values are handed to the optimiser. For
  plain SGD this is equivalent to shrinking the parameter directly,
  $\theta\leftarrow(1-\eta\lambda)\theta-\eta\nabla_\theta L_{\text{data}}$; the two stop being
  equivalent once momentum or Adam are involved, which is exactly the distinction *decoupled* weight
  decay (AdamW) is built to preserve.
- **Batch normalisation's backward pass.** BatchNorm centres and rescales each feature over the *batch*,
  $\hat z_j=(z_j-\mu_j)/\sqrt{\sigma_j^2+\epsilon}$, with $\mu_j,\sigma_j^2$ computed from every example
  sharing that batch — unlike $\mathrm{ReLU}$'s gate above, which acts independently on each example, the
  local Jacobian $\partial\hat z_{i,j}/\partial z_{i',j}$ is nonzero for every pair of examples $i,i'$
  through $\mu_j$ and $\sigma_j^2$, so one example's gradient at that layer genuinely depends on every
  other example currently sharing its batch.

<details>
<summary>Checks (runnable)</summary>

```python
# --- Part 1: the worked example of the statement ---
X_ex = np.array([[1.0, -1.0], [0.5, 2.0]])
y_ex = np.array([0, 1])
params_ex = {
    "W1": np.array([[0.5, -0.5], [-1.0, 1.0]]),
    "b1": np.array([0.1, -0.2]),
    "W2": np.array([[1.0, -1.0], [-1.0, 2.0]]),
    "b2": np.array([0.0, 0.5]),
}
cache_ex = _forward(params_ex, X_ex)
assert np.allclose(cache_ex["Z1"], [[1.6, -1.7], [-1.65, 1.55]])
assert np.allclose(cache_ex["H"], [[1.6, 0.0], [0.0, 1.55]])
assert np.allclose(cache_ex["S"], [[1.6, -1.1], [-1.55, 3.6]])
assert np.allclose(cache_ex["probs"], [[0.9370, 0.0630], [0.0058, 0.9942]], atol=1e-4)
loss_ex, grads_ex = forward_backward(params_ex, X_ex, y_ex)
assert round(loss_ex, 4) == 0.0354
onehot_ex = np.array([[1.0, 0.0], [0.0, 1.0]])
dS_ex = (cache_ex["probs"] - onehot_ex) / 2
assert np.allclose(dS_ex, [[-0.0315, 0.0315], [0.0029, -0.0029]], atol=1e-4)

# --- Part 1: an independent finite-difference loop, NOT calling grad_check, confirming forward_backward ---


def relative_errors(params, X, y, analytic, eps=1e-5):
    """Central-difference relative error of a GIVEN analytic-gradient dict (which may be deliberately
    wrong) against forward_backward's own loss. Independent of grad_check's implementation above."""
    out = {}
    for name, arr in params.items():
        grad_a = analytic[name]
        errs = np.zeros_like(arr, dtype=np.float64)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = arr[idx]
            eps_t = arr.dtype.type(eps)
            arr[idx] = original + eps_t
            loss_plus, _ = forward_backward(params, X, y)
            arr[idx] = original - eps_t
            loss_minus, _ = forward_backward(params, X, y)
            arr[idx] = original
            grad_n = (loss_plus - loss_minus) / (2 * eps_t)
            denom = max(abs(grad_a[idx]) + abs(grad_n), 1e-12)
            errs[idx] = abs(grad_a[idx] - grad_n) / denom
        out[name] = errs
    return out


rng = np.random.default_rng(7)
d, m, C, B = 3, 4, 3, 5
p_rand = {
    "W1": rng.normal(scale=0.5, size=(d, m)), "b1": rng.normal(scale=0.5, size=m),
    "W2": rng.normal(scale=0.5, size=(m, C)), "b2": rng.normal(scale=0.5, size=C),
}
X_rand = rng.normal(size=(B, d))
y_rand = rng.integers(0, C, size=B)
_, analytic_rand = forward_backward(p_rand, X_rand, y_rand)
errs_indep = relative_errors(p_rand, X_rand, y_rand, analytic_rand)
max_indep = max(e.max() for e in errs_indep.values())
assert max_indep < 1e-7, max_indep

# --- Part 2: grad_check itself, below 1e-7 on this same random example ---
gc = grad_check(p_rand, X_rand, y_rand)
assert gc < 1e-7, gc
assert abs(gc - max_indep) < 1e-6      # the two independent implementations agree closely

# --- Part 2: the machine-epsilon numbers behind the eps derivation above ---
u64 = np.finfo(np.float64).eps
u32 = np.finfo(np.float32).eps
assert abs(u64 - 2.22e-16) / u64 < 1e-2
assert abs(u32 - 1.19e-7) / u32 < 1e-2
assert abs(u64 ** (1 / 3) - 6e-6) / 6e-6 < 0.05
assert abs(u32 ** (1 / 3) - 4.9e-3) / 4.9e-3 < 0.05
assert round(np.log10(u32 / u64)) == 9      # "nine orders of magnitude larger"

# --- Part 2: a broader sweep of random network sizes, generous bound (still far under the "suspicious" 1e-4) ---
rng2 = np.random.default_rng(0)
worst_sweep = 0.0
n_checked = 0
for trial in range(200):
    dd, mm, CC = (int(rng2.integers(1, 6)) for _ in range(3))
    BB = int(rng2.integers(2, 6))
    rng_p = np.random.default_rng(1000 + trial)
    p = {
        "W1": rng_p.normal(scale=0.5, size=(dd, mm)), "b1": rng_p.normal(scale=0.5, size=mm),
        "W2": rng_p.normal(scale=0.5, size=(mm, CC)), "b2": rng_p.normal(scale=0.5, size=CC),
    }
    Xs = rng2.normal(size=(BB, dd))
    ys = rng2.integers(0, CC, size=BB)
    try:
        err = grad_check(p, Xs, ys)
    except ValueError:
        continue                        # a random draw landed within eps of a kink; grad_check itself refused it
    n_checked += 1
    worst_sweep = max(worst_sweep, err)
assert n_checked > 150                  # the kink guard did not eat the whole sweep
assert worst_sweep < 1e-4, worst_sweep

# --- Part 2: deliberate bug -- drop the 1/B scaling on b2's gradient only ---
bug_analytic = dict(analytic_rand)
bug_analytic["b2"] = analytic_rand["b2"] * B         # undoes the /B, exactly the bug described above
errs_bug = relative_errors(p_rand, X_rand, y_rand, bug_analytic)
assert errs_bug["b2"].max() > 1e-4                    # b2's own error is large
assert max(errs_bug[k].max() for k in ("W1", "b1", "W2")) < 1e-7   # every other parameter is untouched

# --- Part 2: float32 -- the derivation's prediction, demonstrated directly ---
p32 = {k: v.astype(np.float32) for k, v in p_rand.items()}
X32 = X_rand.astype(np.float32)
_, analytic32 = forward_backward(p32, X32, y_rand)
errs32 = relative_errors(p32, X32, y_rand, analytic32)
assert max(e.max() for e in errs32.values()) > 0.1    # float64's same eps=1e-5 gave < 1e-7 above
try:
    grad_check(p32, X32, y_rand)                       # grad_check itself refuses float32 rather than
    assert False, "expected grad_check to reject float32 parameters"
except ValueError:
    pass

# --- Part 2: the kink guard ---
d_k, m_k = 2, 2
p_kink = {"W1": np.zeros((d_k, m_k)), "b1": np.zeros(m_k), "W2": np.eye(m_k, 2), "b2": np.zeros(2)}
X_kink = np.array([[1.0, 2.0], [3.0, 4.0]])
y_kink = np.array([0, 1])
try:
    grad_check(p_kink, X_kink, y_kink)
    assert False, "expected grad_check to reject a pre-activation exactly on the kink"
except ValueError:
    pass

# --- Part 3: dataset is not linearly separable -- an independent check with scikit-learn's logistic regression ---
from sklearn.linear_model import LogisticRegression

X_sp, y_sp = make_spirals(150, n_turns=1.5, noise=0.08, seed=0)
linear_acc = LogisticRegression().fit(X_sp, y_sp).score(X_sp, y_sp)
assert linear_acc < 0.85               # a straight-line boundary falls well short of separating it

# --- Part 3: He initialisation reaches the stated accuracy; all-zero initialisation cannot move past chance ---
he_params = he_init(2, 64, 2, seed=1)
trained_he = train(he_params, X_sp, y_sp, steps=4000, batch_size=32, lr=0.15, momentum=0.9, seed=2)
acc_he = accuracy(trained_he, X_sp, y_sp)
assert acc_he >= 0.98, acc_he           # the text states 99%; a little slack for platform float differences

zero_params = zero_init(2, 64, 2)
trained_zero = train(zero_params, X_sp, y_sp, steps=4000, batch_size=32, lr=0.15, momentum=0.9, seed=2)
assert np.array_equal(trained_zero["W1"], np.zeros((2, 64)))
assert np.array_equal(trained_zero["b1"], np.zeros(64))
assert np.array_equal(trained_zero["W2"], np.zeros((64, 2)))
assert not np.allclose(trained_zero["b2"], 0.0)         # b2, unlike the other three, is free to move
assert accuracy(trained_zero, X_sp, y_sp) == 0.5        # exactly chance: one class predicted for every input

# --- Part 3: He initialisation preserves the activation second moment across 10 stacked ReLU layers; 1/fan_in halves it ---


def second_moments(n_layers, width, var_scale, n_samples, seed):
    """Independent of he_init/zero_init above: a bare stack of ReLU layers with no biases (so E[z] = 0
    exactly, matching the derivation), tracking the empirical second moment after each layer."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n_samples, width))          # "layer 0": second moment 1 by construction
    moments = [float(np.mean(h ** 2))]
    for _ in range(n_layers):
        W = rng.standard_normal((width, width)) * np.sqrt(var_scale / width)
        h = np.maximum(h @ W, 0.0)
        moments.append(float(np.mean(h ** 2)))
    return moments


he_moments = second_moments(10, 128, 2.0, 20_000, seed=3)
half_moments = second_moments(10, 128, 1.0, 20_000, seed=3)
assert all(0.5 < v < 2.0 for v in he_moments[1:])        # stays within a small constant factor of 1
ratios = [half_moments[i + 1] / half_moments[i] for i in range(10)]
assert all(0.35 < r < 0.65 for r in ratios)               # cut roughly in half at every layer
assert abs(half_moments[-1] / 2.0 ** -10 - 1.0) < 2.0     # within a factor of 3 of the derived 2 ** -10

print("all checks passed")
```

</details>

</details>
