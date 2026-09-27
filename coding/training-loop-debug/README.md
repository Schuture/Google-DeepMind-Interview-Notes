# Debugging a Classifier That Does Not Learn

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · ML debugging (NumPy) | ★★★★☆ | Medium | RE · RS · MLE · Applied AI | debugging, softmax, data-shuffling, gradient-scaling, momentum, dropout, broadcasting, sanity-checks | 3 parts / 45 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

A *two-layer perceptron* maps a standardised input $x \in \mathbb{R}^2$ to a distribution over $C = 3$
classes through one hidden layer of $32$ units. The hidden pre-activation is $z^{(1)} = xW_1 + b_1$
($W_1 \in \mathbb{R}^{2\times32}$, $b_1 \in \mathbb{R}^{32}$), the hidden activation is
$h = \mathrm{ReLU}(z^{(1)}) = \max(z^{(1)}, 0)$ elementwise, and the logits are
$z^{(2)} = h_{\mathrm{drop}}W_2 + b_2 \in \mathbb{R}^3$ ($W_2 \in \mathbb{R}^{32\times3}$, $b_2\in\mathbb R^3$),
where $h_{\mathrm{drop}}$ is $h$ after *inverted dropout*: during training, each of the $32$ hidden units
is independently zeroed with probability $p = 0.2$ and every surviving unit is rescaled by $1/(1-p)$; at
evaluation time no unit is dropped and no rescaling is applied, so $h_{\mathrm{drop}} = h$. The predicted
class probabilities are $\mathrm{probs}_c = e^{z^{(2)}_c} / \sum_{c'=0}^{2} e^{z^{(2)}_{c'}}$, and the
loss on a batch of $N$ examples with true labels $y \in \{0,1,2\}^N$ is the mean cross-entropy
$L = \frac1N\sum_{i=1}^N -\log \mathrm{probs}_{i, y_i}$.

`make_dataset` draws $600$ training points and $300$ validation points ($200$/$100$ per class) from
three Gaussian blobs in the plane, then standardises both input features using only the training
split's own mean and standard deviation, $x' = (x - \hat\mu) / \hat\sigma$, applying the same $\hat\mu,
\hat\sigma$ to the validation split. Training runs mini-batch SGD with momentum for $30$ epochs of batch
size $32$, reshuffling the training data at the start of every epoch: the velocity $v$ (one array per
parameter, initialised to $\mathbf 0$ once, before training starts) updates as
$v \leftarrow \mu v - \eta\, \nabla_\theta L$ and each parameter as $\theta \leftarrow \theta + v$, with
momentum $\mu = 0.9$ and learning rate $\eta$.

### Part 1 — Find and fix every bug

The script below is meant to train this network on `make_dataset`'s data and print the mean training
loss per epoch and the final validation accuracy. It contains exactly six bugs; none of them raises an
exception. Find and fix every one. With all six fixed, `train(seed=0)` reaches a validation accuracy of
at least $0.9$.

```py
import numpy as np

HIDDEN = 32
DROPOUT_P = 0.2
N_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.15
MOMENTUM = 0.9
N_PER_CLASS_TRAIN = 200
N_PER_CLASS_VAL = 100
CENTERS = np.array([[0.0, 2.0], [-1.7320508, -1.0], [1.7320508, -1.0]])
BLOB_STD = 0.9


def make_dataset(seed: int):
    """Three Gaussian blobs in 2-D, one per class. Returns standardised (X_train, y_train, X_val, y_val)."""
    rng = np.random.default_rng(seed)

    def blobs(n_per_class, rng):
        X = np.concatenate([CENTERS[c] + BLOB_STD * rng.standard_normal((n_per_class, 2)) for c in range(N_CLASSES)])
        y = np.repeat(np.arange(N_CLASSES), n_per_class)
        return X, y

    X_train, y_train = blobs(N_PER_CLASS_TRAIN, rng)
    X_val, y_val = blobs(N_PER_CLASS_VAL, rng)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, y_train, (X_val - mean) / std, y_val


def init_params(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        W1=rng.standard_normal((2, HIDDEN)) * np.sqrt(2.0 / 2),
        b1=np.zeros(HIDDEN),
        W2=rng.standard_normal((HIDDEN, N_CLASSES)) * 0.01,
        b2=np.zeros(N_CLASSES),
    )


def forward(params: dict, X: np.ndarray, training: bool, rng: np.random.Generator) -> dict:
    """X: (N, 2). Returns {"probs": (N, 3), "cache": ...}. training selects training- versus eval-time behaviour."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)                                       # ReLU
    keep_prob = 1.0 - DROPOUT_P
    mask = (rng.random(h.shape) < keep_prob) / keep_prob           # inverted dropout
    h_drop = h * mask
    logits = h_drop @ W2 + b2
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=0, keepdims=True)
    cache = dict(X=X, z1=z1, mask=mask, h_drop=h_drop)
    return {"probs": probs, "cache": cache}


def loss_and_grads(params: dict, X: np.ndarray, y: np.ndarray, training: bool, rng: np.random.Generator):
    """Mean softmax cross-entropy over the batch, and its gradient with respect to every parameter."""
    out = forward(params, X, training=training, rng=rng)
    probs, cache = out["probs"], out["cache"]
    N = X.shape[0]
    loss = float(np.mean(-np.log(probs[np.arange(N), y] + 1e-12)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dlogits = probs - onehot
    dW2 = cache["h_drop"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = (dlogits @ params["W2"].T) * cache["mask"]
    dz1 = dh * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def accuracy_score(probs: np.ndarray, y: np.ndarray) -> float:
    """probs: (N, 3) predicted class probabilities. y: (N,) true labels. Returns the fraction correct."""
    pred = np.argmax(probs, axis=1, keepdims=True)
    return float(np.mean(pred == y))


def sgd_momentum_step(params, velocity, grads, lr, momentum):
    new_v = {k: momentum * velocity[k] - lr * grads[k] for k in params}
    new_p = {k: params[k] + new_v[k] for k in params}
    return new_p, new_v


def train(seed: int = 0) -> dict:
    X_train, y_train, X_val, y_val = make_dataset(seed)
    params = init_params(seed)
    rng = np.random.default_rng(seed + 1000)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    N = len(X_train)
    train_losses = []
    for epoch in range(EPOCHS):
        X_shuf = X_train[rng.permutation(N)]
        y_shuf = y_train[rng.permutation(N)]
        losses = []
        for start in range(0, N, BATCH_SIZE):
            Xb, yb = X_shuf[start:start + BATCH_SIZE], y_shuf[start:start + BATCH_SIZE]
            velocity = {k: np.zeros_like(v) for k, v in params.items()}
            loss, grads = loss_and_grads(params, Xb, yb, training=True, rng=rng)
            params, velocity = sgd_momentum_step(params, velocity, grads, LR, MOMENTUM)
            losses.append(loss)
        train_losses.append(float(np.mean(losses)))
        print(f"epoch {epoch + 1:2d}  training loss {train_losses[-1]:.4f}")
    val_accuracy = accuracy_score(forward(params, X_val, training=False, rng=rng)["probs"], y_val)
    print(f"validation accuracy: {val_accuracy:.3f}")
    return {"train_losses": train_losses, "val_accuracy": val_accuracy}


if __name__ == "__main__":
    train()
```

Running it as given prints something like:

```text
epoch  1  training loss 25.5481
epoch  2  training loss 26.7524
epoch  3  training loss 26.8433
...
epoch 16  training loss nan
...
epoch 30  training loss nan
validation accuracy: 0.333
```

### Part 2 — Predict the symptom of each bug

For each of the six bugs, in isolation (every other bug already fixed), state what the mean
training-loss-per-epoch curve looks like and what the reported validation accuracy is, compared to a
fully correct run, and explain why.

### Part 3 — Sanity checks that catch these early

Implement `sanity_checks`, taking the four pieces above and a dataset, that runs five fast checks before
committing to a full run and returns a `dict[str, bool]`:

- `initial_loss_near_ln_C`: a freshly initialised network's loss on the given dataset (dropout off) is
  within $0.1$ of $\ln C$.
- `overfits_tiny_batch`: starting from a fresh initialisation, $500$ steps of plain gradient descent
  (learning rate $0.5$, dropout off) on the first $8$ examples of the dataset drive the loss below
  $0.05$.
- `gradients_match_finite_differences`: every analytic gradient (dropout off, float64) matches a central
  finite difference of step $10^{-5}$ on the first $6$ examples, relative error under $10^{-5}$.
- `eval_invariant_to_batching`: for the first $6$ examples, evaluating them together and evaluating each
  one on its own, in a shuffled order (dropout off), give identical predicted probabilities.
- `accuracy_metric_correct_on_labels`: the accuracy function reports exactly $1.0$ when handed a one-hot
  encoding of the true labels in place of predicted probabilities.

```py
def sanity_checks(init_params, forward, loss_and_grads, accuracy_score,
                   X: np.ndarray, y: np.ndarray, n_classes: int, seed: int = 0) -> dict:
    """Runs the five checks above against the given pipeline pieces and dataset."""
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points are worth confirming before coding: that dropout is meant to be off at evaluation time (the
standard inverted-dropout convention assumed throughout), and that reaching a validation accuracy
comfortably above $0.9$ — rather than any single exact number — is the bar for Part 1, since the specific
synthetic dataset is otherwise unconstrained by the statement.

### Part 1

Running the script above and fixing bugs in the order their symptoms point to, one new symptom appears
at a time. Because bugs 5 and 6 can make the *reported* accuracy wrong without the model itself being bad,
the table also tracks the *true* accuracy: the same trained parameters, re-evaluated with a fully correct
forward pass and accuracy function regardless of which bugs are still active elsewhere.

| Bugs fixed so far | Training loss, first few epochs | Reported / true validation accuracy | Points to next |
| --- | --- | --- | --- |
| none | $\approx 26$, climbing to `nan` by epoch $16$ | $0.333$ / $0.333$ | the softmax axis |
| softmax axis | $\approx 2.0$, still far above a healthy run | $0.333$ / $0.333$ | the missing $1/N$ |
| + gradient scale | $\approx 1.1$–$1.2$, the $\ln 3$ plateau | $0.333$ / $0.243$ | the mismatched shuffle |
| + shuffle | decreasing normally, epoch $1 \approx 0.39$ | $0.333$ / $0.957$ | the accuracy function |
| + accuracy function | unchanged | $0.963$ / $0.957$ | close but not exact — the momentum reset |
| + momentum | epoch $1$ improves to $\approx 0.30$ | $0.953$ / $0.957$ | still not exact — evaluation-time dropout |
| + dropout at eval (all six) | unchanged | $0.957$ / $0.957$ | — |

**Bug 2: softmax over the batch axis.** `logits.max(axis=0)` and `.sum(axis=0)` reduce over the $N$
examples of the batch for each fixed class, not over the $3$ classes of each fixed example: the result
does not sum to $1$ along a row, so it is not any example's probability distribution, and it depends on
which other examples happen to share the batch. Class $c$'s mass $\sum_i e^{z^{(2)}_{i,c}}$ is shared
out among the $N$ rows in proportion to their own logit for that class alone, so whichever row has the
largest logit for class $c$ claims nearly all of it, leaving the other $N-1$ rows a probability for
class $c$ close to $0$ — including, for most rows, their own true class — which is exactly what drives
$-\log(\mathrm{probs}_{i,y_i})$ into the tens seen above.

**Bug 3: the missing $1/N$.** The forward loss averages the $N$ per-example terms,
$L=\frac1N\sum_i \ell_i$ with $\ell_i=-\log \mathrm{probs}_{i,y_i} = -z^{(2)}_{i,y_i} + \log\sum_{c'} e^{z^{(2)}_{i,c'}}$,
so $\partial \ell_i/\partial z^{(2)}_{i,c} = \mathrm{probs}_{i,c}-\mathbb 1[c=y_i]$ and
$\partial L/\partial z^{(2)}_{i,c}=\frac1N(\mathrm{probs}_{i,c}-\mathbb 1[c=y_i])$. `dlogits = probs -
onehot` omits that $\frac1N$, leaving every gradient — and so every parameter update, since $\eta$
multiplies the gradient directly — exactly $N=32$ times too large: training behaves as though the
learning rate were $32\times 0.15=4.8$ rather than $0.15$, and the loss never settles into the healthy
run's range.

**Bug 1: mismatched shuffle.** `X_shuf = X_train[rng.permutation(N)]` and
`y_shuf = y_train[rng.permutation(N)]` draw two independent orderings of the same length, so
`X_shuf[i]` and `y_shuf[i]` almost never describe the same original example once $N>1$. Training on
inputs paired with effectively random labels leaves the network nothing to fit beyond the label
marginal: minimising cross-entropy with no usable signal points every row of `probs` towards the
empirical class frequencies, and with three balanced classes that is the uniform distribution
$(1/3,1/3,1/3)$, whose cross-entropy against any single true label is exactly $-\log(1/3)=\ln 3$ — the
plateau above.

**Bug 6: `keepdims=True` in accuracy.** `np.argmax(probs, axis=1, keepdims=True)` has shape $(N,1)$;
comparing it against `y`'s shape $(N,)$ right-aligns and broadcasts to $(N,N)$, whose entry $(i,j)$ is
$\mathrm{pred}_i=y_j$ — the $N$ correct per-example comparisons that belong on the diagonal, diluted by
$N(N-1)$ unrelated off-diagonal ones. Writing $q_c$ for the fraction of the $N$ predictions equal to
class $c$ and $q'_c$ for the true fraction, the mean over all $N^2$ pairs is
$\frac{1}{N^2}\sum_c (Nq_c)(Nq'_c)=\sum_c q_c q'_c$; for a model whose predictions land on each of $3$
balanced classes about as often as that class truly occurs, every $q_c,q'_c\approx1/3$ and the reported
number is $\approx 3\cdot(1/3)^2=1/3$ regardless of how many of the $N$ diagonal entries are actually
correct. Nothing about training is affected, so the loss curve gives no hint of it either.

**Bug 4: momentum reset every step.** Recreating `velocity` as all zeros at the top of the mini-batch
loop means every step computes $v=\mu\cdot\mathbf 0-\eta\nabla_\theta L=-\eta\nabla_\theta L$: a plain
gradient step, regardless of $\mu$. A genuinely accumulating velocity, over $k$ consecutive steps with a
roughly constant gradient $g$, reaches $v_k=-\eta g\sum_{j=0}^{k-1}\mu^j=-\eta g\,(1-\mu^k)/(1-\mu)$,
approaching $-\eta g/(1-\mu)=10\times$ the plain step as $k$ grows at $\mu=0.9$; resetting every step
pins every update at the $k=1$ term instead, so the first several epochs move noticeably more slowly
than a correctly accumulating run, even though both reach the same place once $30$ epochs have given the
un-accelerated updates enough time to catch up — a bug that changes the *shape* of the loss curve without
changing where it ends up.

**Bug 5: dropout active at evaluation.** Leaving the random mask unconditional — not gated by
`training` — means `forward(..., training=False, ...)` still zeroes $20\%$ of the hidden units at random
and rescales the rest by $1/(1-p)=1.25$: two calls on the very same `X` draw two different masks and can
disagree, so `forward` is no longer a deterministic function of its input at evaluation time. Like bug 6,
nothing about training changes, so this bug is invisible in the training-loss curve; it shows up only by
evaluating the same inputs more than once and comparing.

```python
import numpy as np

HIDDEN = 32
DROPOUT_P = 0.2
N_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.15
MOMENTUM = 0.9
N_PER_CLASS_TRAIN = 200
N_PER_CLASS_VAL = 100
CENTERS = np.array([[0.0, 2.0], [-1.7320508, -1.0], [1.7320508, -1.0]])
BLOB_STD = 0.9


def make_dataset(seed: int):
    rng = np.random.default_rng(seed)

    def blobs(n_per_class, rng):
        X = np.concatenate([CENTERS[c] + BLOB_STD * rng.standard_normal((n_per_class, 2)) for c in range(N_CLASSES)])
        y = np.repeat(np.arange(N_CLASSES), n_per_class)
        return X, y

    X_train, y_train = blobs(N_PER_CLASS_TRAIN, rng)
    X_val, y_val = blobs(N_PER_CLASS_VAL, rng)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, y_train, (X_val - mean) / std, y_val


def init_params(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        W1=rng.standard_normal((2, HIDDEN)) * np.sqrt(2.0 / 2),
        b1=np.zeros(HIDDEN),
        W2=rng.standard_normal((HIDDEN, N_CLASSES)) * 0.01,
        b2=np.zeros(N_CLASSES),
    )


def forward(params: dict, X: np.ndarray, training: bool, rng: np.random.Generator) -> dict:
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)
    if training:
        keep_prob = 1.0 - DROPOUT_P
        mask = (rng.random(h.shape) < keep_prob) / keep_prob   # NOTE: inverted dropout, drawn only while training
    else:
        mask = np.ones_like(h)                                  # NOTE: identity at eval -- no masking, no rescaling
    h_drop = h * mask
    logits = h_drop @ W2 + b2
    shifted = logits - logits.max(axis=1, keepdims=True)        # NOTE: max over classes (axis=1), not the batch
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)                 # NOTE: likewise sum over classes -- rows sum to 1
    cache = dict(X=X, z1=z1, mask=mask, h_drop=h_drop)
    return {"probs": probs, "cache": cache}


def loss_and_grads(params: dict, X: np.ndarray, y: np.ndarray, training: bool, rng: np.random.Generator):
    out = forward(params, X, training=training, rng=rng)
    probs, cache = out["probs"], out["cache"]
    N = X.shape[0]
    loss = float(np.mean(-np.log(probs[np.arange(N), y] + 1e-12)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dlogits = (probs - onehot) / N       # NOTE: /N mirrors the mean in `loss` -- omitting it scales every
                                          #       gradient (and so the effective learning rate) by N
    dW2 = cache["h_drop"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = (dlogits @ params["W2"].T) * cache["mask"]
    dz1 = dh * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def accuracy_score(probs: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)      # NOTE: no keepdims -- shape (N,), matching y, so pred == y is elementwise
    return float(np.mean(pred == y))


def sgd_momentum_step(params, velocity, grads, lr, momentum):
    new_v = {k: momentum * velocity[k] - lr * grads[k] for k in params}
    new_p = {k: params[k] + new_v[k] for k in params}
    return new_p, new_v


def train(seed: int = 0) -> dict:
    X_train, y_train, X_val, y_val = make_dataset(seed)
    params = init_params(seed)
    rng = np.random.default_rng(seed + 1000)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}   # NOTE: created once, never reset inside the loop
    N = len(X_train)
    train_losses = []
    for epoch in range(EPOCHS):
        perm = rng.permutation(N)
        X_shuf, y_shuf = X_train[perm], y_train[perm]              # NOTE: one shared permutation for X and y
        losses = []
        for start in range(0, N, BATCH_SIZE):
            Xb, yb = X_shuf[start:start + BATCH_SIZE], y_shuf[start:start + BATCH_SIZE]
            loss, grads = loss_and_grads(params, Xb, yb, training=True, rng=rng)
            params, velocity = sgd_momentum_step(params, velocity, grads, LR, MOMENTUM)
            losses.append(loss)
        train_losses.append(float(np.mean(losses)))
    val_accuracy = accuracy_score(forward(params, X_val, training=False, rng=rng)["probs"], y_val)
    return {"train_losses": train_losses, "val_accuracy": val_accuracy}
```

### Part 2

| Bug (isolated) | Training-loss curve | Reported validation accuracy |
| --- | --- | --- |
| 1 — mismatched shuffle | plateaus near $\ln 3\approx1.10$ every epoch | $\approx 1/3$ |
| 2 — softmax over the batch axis | huge, $17$–$27$, no downward trend | $\approx 1/3$ |
| 3 — missing $1/N$ | elevated and erratic, $1.1$–$4.7$, never converges | $\approx 1/3$ |
| 4 — momentum reset every step | decreasing normally, but epoch $1\approx0.39$ against $\approx0.30$ correctly | matches the correct run, $\approx 0.96$ |
| 5 — dropout at evaluation | identical to the correct run | close to the correct run on any one call, but changes between calls |
| 6 — `keepdims` in accuracy | identical to the correct run | $\approx 1/3$ regardless of how good the model is |

Bugs 1, 2 and 3 each corrupt the gradient (or the labels the gradient is computed against), so each
leaves a distinctive mark on the loss curve itself, exactly as derived in Part 1: bug 1's plateau at
$\ln 3$, bug 2's blow-up from normalising over the wrong axis, bug 3's persistently elevated loss from an
effective learning rate of $4.8$. Bug 4 also touches training, but only its *speed*: with $30$ epochs to
recover, a plain gradient step every time still reaches the same place bug 1–3 never do, so it shows up
as a slower first few epochs rather than a wrong destination. Bugs 5 and 6 touch nothing that training
ever sees — dropout-at-eval and the accuracy metric both run strictly after the last gradient step — so
the loss curve for either is bit-for-bit the correct one; only comparing repeated evaluations (bug 5) or
checking the metric against a case with a known answer (bug 6) reveals them, which is exactly the job
Part 3's checks do.

### Part 3

```python
def sanity_checks(init_params, forward, loss_and_grads, accuracy_score, X, y, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    results = {}

    # a freshly initialised net should be no more sure of anything than a uniform guess over the classes
    params = init_params(seed)
    loss0, _ = loss_and_grads(params, X, y, training=False, rng=rng)
    results["initial_loss_near_ln_C"] = bool(abs(loss0 - np.log(n_classes)) < 0.1)

    # backprop should be able to memorise a handful of examples almost exactly
    Xb, yb = X[:8], y[:8]
    p = init_params(seed + 1)
    for _ in range(500):
        _, grads = loss_and_grads(p, Xb, yb, training=False, rng=rng)
        p = {k: p[k] - 0.5 * grads[k] for k in p}
    final_loss, _ = loss_and_grads(p, Xb, yb, training=False, rng=rng)
    results["overfits_tiny_batch"] = bool(final_loss < 0.05)

    # the analytic gradient must match a central finite difference, parameter by parameter
    p64 = {k: v.astype(np.float64) for k, v in init_params(seed + 2).items()}
    Xg, yg = X[:6].astype(np.float64), y[:6]
    _, analytic = loss_and_grads(p64, Xg, yg, training=False, rng=rng)
    eps = 1e-5
    worst_rel_err = 0.0
    for name, arr in p64.items():
        grad = analytic[name]
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps
            loss_plus, _ = loss_and_grads(p64, Xg, yg, training=False, rng=rng)
            arr[idx] = orig - eps
            loss_minus, _ = loss_and_grads(p64, Xg, yg, training=False, rng=rng)
            arr[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            denom = max(abs(numeric), abs(grad[idx]), 1e-8)
            worst_rel_err = max(worst_rel_err, abs(numeric - grad[idx]) / denom)
    results["gradients_match_finite_differences"] = bool(worst_rel_err < 1e-5)

    # a prediction must not depend on which other rows happen to share its batch
    params_eval = init_params(seed + 3)
    X_sub = X[:6]
    probs_batch = forward(params_eval, X_sub, training=False, rng=rng)["probs"]
    order = np.random.default_rng(seed + 4).permutation(len(X_sub))
    probs_singleton = np.zeros_like(probs_batch)
    for i in order:
        probs_singleton[i] = forward(params_eval, X_sub[i:i + 1], training=False, rng=rng)["probs"][0]
    results["eval_invariant_to_batching"] = bool(np.allclose(probs_batch, probs_singleton, atol=1e-8))

    # the metric itself must report perfect accuracy when handed perfect predictions
    onehot = np.zeros((len(y), n_classes))
    onehot[np.arange(len(y)), y] = 1.0
    results["accuracy_metric_correct_on_labels"] = bool(accuracy_score(onehot, y) == 1.0)

    return results
```

Each check targets a different failure mode. `initial_loss_near_ln_C` catches a badly scaled
initialisation or a corrupted forward pass early, before a single gradient step is taken — including
bug 2, whose batch-axis softmax already reports a loss nowhere near $\ln 3$ on a fresh network. Because
it only needs backprop and a few hundred cheap steps on $8$ examples, `overfits_tiny_batch` is normally
the fastest way to catch a broken gradient path (a transposed weight, a swapped axis) without waiting
through even one real epoch; `gradients_match_finite_differences` is the more exhaustive version of the
same idea, checking every parameter individually rather than trusting that a decreasing loss implies a
correct gradient. `eval_invariant_to_batching` is the one built specifically for the two silent bugs of
Part 2: bug 2's axis-0 softmax gives a batch of $1$ a completely different (degenerate) normalisation
than a batch of $6$, and bug 5's unconditional dropout draws an independent random mask on every call, so
neither can keep a prediction the same when the same row is evaluated alone instead of alongside others.
`accuracy_metric_correct_on_labels` isolates the metric from the model entirely — handing it a one-hot
encoding of the true labels sidesteps needing the model to be any good at all, so bug 6's broadcasting
mistake has nowhere to hide behind noisy real predictions. Reinserting bug 2 this way in fact fails four
of the five checks, not only the batching one, since a forward pass that never produces a valid
per-example distribution corrupts the loss, the tiny-batch fit and the gradient check as well; bug 5
fails two (batching and the gradient check, since finite differences need a deterministic forward pass);
bug 6, confined entirely to the metric, fails only its own check.

### Follow-ups

- **Bugs invisible from the loss curve alone.** Bugs 5 and 6 leave every training-loss number identical
  to a fully correct run, because dropout-at-eval and the accuracy metric only run *after* the last
  gradient step. No amount of staring at the loss curve catches either one; only evaluating the same
  inputs twice (bug 5) or checking the metric on a case with a known answer (bug 6) does, which is
  exactly what `sanity_checks` is for.
- **Learning-rate range test.** Training for a few hundred steps while increasing the learning rate
  geometrically from a tiny value and plotting the loss shows a wide, stable, decreasing region followed
  by a sharp blow-up; bug 3's effective learning rate of $32\times\eta$ sits well past that point even
  though $\eta=0.15$ on its own looks unremarkable on paper.
- **Visualise a batch with its labels before training.** Plotting `Xb` coloured by `yb` for a couple of
  mini-batches, after standardisation, is the cheapest check against bug 1: a scatter plot where every
  colour is spread uniformly over the same region, rather than in the three separated blobs the data are
  actually drawn from, is visible on sight, without a loss curve at all.
- **Normalisation leakage between splits.** `make_dataset` fits `mean`/`std` on the training split alone
  and reuses them for validation; computing them on the concatenation of both splits instead would leak a
  little information about the validation set's distribution into every standardised input, inflating
  validation accuracy by an amount too small to look wrong on its own and too easy to miss because
  nothing about it raises an error.
- **Determinism.** Every source of randomness here — data generation, initialisation, shuffling, dropout
  — is seeded through an explicit `np.random.default_rng`, never a bare global call; that is what makes
  it meaningful to say `train(seed=0)` reaches a specific accuracy at all, rather than one it reaches
  "usually".

<details>
<summary>Checks (runnable)</summary>

```python
result = train(seed=0)
assert len(result["train_losses"]) == EPOCHS
assert result["val_accuracy"] >= 0.9, result["val_accuracy"]


# --- every formula quoted above, checked independently ---

def forward_loop(params, X):                              # forward's formula, written with explicit loops
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    N, Din, H, C = X.shape[0], X.shape[1], params["W1"].shape[1], params["W2"].shape[1]
    probs = np.zeros((N, C))
    for i in range(N):
        z1 = [sum(X[i, d] * W1[d, j] for d in range(Din)) + b1[j] for j in range(H)]
        h = [max(v, 0.0) for v in z1]
        logits = [sum(h[j] * W2[j, c] for j in range(H)) + b2[c] for c in range(C)]
        m = max(logits)
        exps = [np.exp(v - m) for v in logits]
        s = sum(exps)
        probs[i] = [e / s for e in exps]
    return probs


rng = np.random.default_rng(3)
params_rand = init_params(3)
X_rand = rng.normal(size=(15, 2))
assert np.allclose(forward(params_rand, X_rand, training=False, rng=rng)["probs"],
                    forward_loop(params_rand, X_rand), atol=1e-8)

# inverted dropout: E[mask] = 1 elementwise, and about DROPOUT_P of units are zeroed
rng = np.random.default_rng(4)
keep_prob = 1.0 - DROPOUT_P
mask_sample = (rng.random((200_000, HIDDEN)) < keep_prob) / keep_prob
assert abs(mask_sample.mean() - 1.0) < 0.01
assert abs(np.mean(mask_sample == 0.0) - DROPOUT_P) < 0.01

# standardisation: the training split has mean 0 and standard deviation 1 per feature
X_train, y_train, X_val, y_val = make_dataset(0)
assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-8)
assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-8)
print("forward pass, dropout statistics, and standardisation match their formulas")


# --- bug machinery: one general, parameterised reimplementation, independent of the fixed code above
#     except for the pieces the corresponding bug leaves untouched -- not string-patching the listing ---

def forward_general(params, X, training, rng, bugs=frozenset()):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)
    keep_prob = 1.0 - DROPOUT_P
    if training or (5 in bugs):                            # bug 5: dropout stays on when training is False
        mask = (rng.random(h.shape) < keep_prob) / keep_prob
    else:
        mask = np.ones_like(h)
    h_drop = h * mask
    logits = h_drop @ W2 + b2
    axis = 0 if (2 in bugs) else 1                           # bug 2: normalise over the batch, not the classes
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=axis, keepdims=True)
    return {"probs": probs, "cache": dict(X=X, z1=z1, mask=mask, h_drop=h_drop)}


def loss_and_grads_general(params, X, y, training, rng, bugs=frozenset()):
    out = forward_general(params, X, training, rng, bugs)
    probs, cache = out["probs"], out["cache"]
    N = X.shape[0]
    loss = float(np.mean(-np.log(probs[np.arange(N), y] + 1e-12)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dlogits = probs - onehot
    if 3 not in bugs:                                        # bug 3: skip the /N that undoes the batch mean
        dlogits = dlogits / N
    dW2 = cache["h_drop"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = (dlogits @ params["W2"].T) * cache["mask"]
    dz1 = dh * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def accuracy_score_general(probs, y, bugs=frozenset()):
    pred = np.argmax(probs, axis=1, keepdims=(6 in bugs))     # bug 6: keepdims broadcasts (N,1) against (N,)
    return float(np.mean(pred == y))


def train_with_bugs(bugs=frozenset(), seed=0):
    """Reruns the training loop with an arbitrary subset of the six bugs reinserted; bugs=frozenset()
    is the fully fixed pipeline. Independent of train() above."""
    X_train, y_train, X_val, y_val = make_dataset(seed)
    params = init_params(seed)
    rng = np.random.default_rng(seed + 1000)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    N = len(X_train)
    train_losses = []
    for epoch in range(EPOCHS):
        if 1 in bugs:                                          # bug 1: two independent permutations
            X_shuf, y_shuf = X_train[rng.permutation(N)], y_train[rng.permutation(N)]
        else:
            perm = rng.permutation(N)
            X_shuf, y_shuf = X_train[perm], y_train[perm]
        losses = []
        for start in range(0, N, BATCH_SIZE):
            Xb, yb = X_shuf[start:start + BATCH_SIZE], y_shuf[start:start + BATCH_SIZE]
            if 4 in bugs:                                       # bug 4: velocity reset every mini-batch
                velocity = {k: np.zeros_like(v) for k, v in params.items()}
            loss, grads = loss_and_grads_general(params, Xb, yb, True, rng, bugs)
            params, velocity = sgd_momentum_step(params, velocity, grads, LR, MOMENTUM)
            losses.append(loss)
        train_losses.append(float(np.mean(losses)))
    probs = forward_general(params, X_val, False, rng, bugs)["probs"]
    reported_acc = accuracy_score_general(probs, y_val, bugs)
    true_probs = forward_general(params, X_val, False, rng, frozenset())["probs"]
    true_acc = accuracy_score_general(true_probs, y_val, frozenset())
    return {"train_losses": train_losses, "reported_val_accuracy": reported_acc,
            "true_val_accuracy": true_acc, "params": params}


ALL_BUGS = frozenset({1, 2, 3, 4, 5, 6})
import warnings

# --- the buggy script exactly as given (all six bugs): the statement's worked example ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                            # expected overflow once bug 2 and 3 compound
    r_all = train_with_bugs(ALL_BUGS)
assert max(l for l in r_all["train_losses"] if np.isfinite(l)) > 20
assert any(not np.isfinite(l) for l in r_all["train_losses"])
assert abs(r_all["reported_val_accuracy"] - 1 / 3) < 0.05

# --- the fixing order of Part 1's table: 2, 3, 1, 6, 4, 5, one new symptom at a time ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r_fix2 = train_with_bugs(ALL_BUGS - {2})
    r_fix23 = train_with_bugs(ALL_BUGS - {2, 3})
r_fix231 = train_with_bugs(ALL_BUGS - {2, 3, 1})
r_fix2316 = train_with_bugs(ALL_BUGS - {2, 3, 1, 6})
r_fix23164 = train_with_bugs(ALL_BUGS - {2, 3, 1, 6, 4})
r_fixed_all = train_with_bugs(frozenset())

assert r_fixed_all["train_losses"] == result["train_losses"]                # train_with_bugs(frozenset()) agrees
assert r_fixed_all["reported_val_accuracy"] == result["val_accuracy"]       # with train(), bit for bit
assert 1.5 < max(r_fix2["train_losses"]) < 5.0                              # no longer huge, still clearly wrong
assert abs(r_fix23["train_losses"][0] - np.log(3)) < 0.15                    # the ln(3) plateau
assert abs(r_fix23["true_val_accuracy"] - 0.2433) < 0.02                     # shuffle still scrambled -- no better
                                                                              # than chance even off the reported metric
assert (r_fix231["train_losses"][0] < 0.5 and abs(r_fix231["reported_val_accuracy"] - 1 / 3) < 0.05
        and r_fix231["true_val_accuracy"] > 0.9)                             # healthy loss, broken reported accuracy
assert 1e-6 < abs(r_fix2316["reported_val_accuracy"] - r_fix2316["true_val_accuracy"]) < 0.05  # close, not exact
assert r_fix23164["train_losses"][0] < r_fix231["train_losses"][0]           # momentum now speeds up epoch 1
assert r_fixed_all["reported_val_accuracy"] == r_fixed_all["true_val_accuracy"]
print("the fixing order in Part 1's table reproduces its numbers one row at a time")

# --- Part 2: each bug, reinserted in isolation (every other bug fixed) ---
r1 = train_with_bugs(frozenset({1}))
assert abs(r1["reported_val_accuracy"] - 1 / 3) < 0.05 and r1["train_losses"][-1] > 0.9

r2 = train_with_bugs(frozenset({2}))
assert abs(r2["reported_val_accuracy"] - 1 / 3) < 0.05 and max(r2["train_losses"]) > 5.0

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r3 = train_with_bugs(frozenset({3}))
finite3 = [l for l in r3["train_losses"] if np.isfinite(l)]
assert abs(r3["reported_val_accuracy"] - 1 / 3) < 0.05 and min(finite3) > 1.0 and max(finite3) > 3.0

r0 = train_with_bugs(frozenset())
r4 = train_with_bugs(frozenset({4}))
assert r4["train_losses"][0] > 1.1 * r0["train_losses"][0]                   # bug 4: noticeably slower epoch 1
assert r4["reported_val_accuracy"] >= 0.9                                    # same destination by epoch 30

rng_demo = np.random.default_rng(42)
probs_a = forward_general(r0["params"], X_val, False, rng_demo, frozenset({5}))["probs"]
probs_b = forward_general(r0["params"], X_val, False, rng_demo, frozenset({5}))["probs"]
assert not np.allclose(probs_a, probs_b, atol=1e-6)                          # bug 5: two evaluations disagree

r6 = train_with_bugs(frozenset({6}))
assert abs(r6["reported_val_accuracy"] - 1 / 3) < 0.05
assert abs(r6["reported_val_accuracy"] - r6["true_val_accuracy"]) > 0.3      # bug 6: reported far from true
print("each bug reproduces its predicted symptom in isolation")


# --- Part 3: sanity_checks passes on the fixed pipeline, fails the relevant check when a bug is reinserted ---
res_fixed = sanity_checks(init_params, forward, loss_and_grads, accuracy_score, X_train, y_train, n_classes=3, seed=0)
assert all(res_fixed.values()), res_fixed


def fwd2(params, X, training, rng):
    return forward_general(params, X, training, rng, frozenset({2}))


def lg2(params, X, y, training, rng):
    return loss_and_grads_general(params, X, y, training, rng, frozenset({2}))


def fwd5(params, X, training, rng):
    return forward_general(params, X, training, rng, frozenset({5}))


def lg5(params, X, y, training, rng):
    return loss_and_grads_general(params, X, y, training, rng, frozenset({5}))


def acc6(probs, y):
    return accuracy_score_general(probs, y, frozenset({6}))


res_bug2 = sanity_checks(init_params, fwd2, lg2, accuracy_score, X_train, y_train, n_classes=3, seed=0)
assert res_bug2["eval_invariant_to_batching"] is False, res_bug2
assert sum(res_bug2.values()) == 1                          # only accuracy_metric_correct_on_labels survives

res_bug5 = sanity_checks(init_params, fwd5, lg5, accuracy_score, X_train, y_train, n_classes=3, seed=0)
assert res_bug5["eval_invariant_to_batching"] is False, res_bug5
assert sum(res_bug5.values()) == 3

res_bug6 = sanity_checks(init_params, forward, loss_and_grads, acc6, X_train, y_train, n_classes=3, seed=0)
assert res_bug6["accuracy_metric_correct_on_labels"] is False, res_bug6
assert sum(res_bug6.values()) == 4

print("sanity_checks passes on the fixed pipeline and fails the relevant check for bugs 2, 5 and 6")

print("all checks passed")
```

</details>

</details>
