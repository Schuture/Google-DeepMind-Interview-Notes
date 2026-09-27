# Code Comprehension: What Does This Code Compute?

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Quiz · reading code aloud | ★★☆☆☆ | Medium | RS · RE · Intern · SWE | code-reading, convolution, padding, numerical-stability, streaming-statistics, attention-masks, reservoir-sampling | 8 snippets / 30–45 min | Skills interview |
<!-- meta:end -->

## Problem

Each of the eight snippets below is a complete Python function (Q2's `g` calls Q1's `f`); none of them is
executed for you, and the task is to read and explain each one exactly as it stands. For every snippet, first state in one
or two sentences what it computes, using the same terminology a textbook or the documentation of the library
involved would use. Then answer its specific sub-questions: an exact traced value wherever one is asked for,
an output length or shape as a formula in the function's own parameters, a stated time or memory cost, and —
where asked — exactly which input breaks the function and how, or exactly what changes under a stated edit to
the code.

### Convolutions

**Q1.**

```py
def f(x, w):
    n, k = len(x), len(w)
    out = []
    for i in range(n - k + 1):
        s = 0.0
        for j in range(k):
            s += x[i + j] * w[j]
        out.append(s)
    return out
```

What does `f` compute, as a function of the sequence `x` and the weights `w`? Give the length of its output as
a formula in $n = \mathtt{len(x)}$ and $k = \mathtt{len(w)}$, and the number of scalar multiplications it
performs. Trace `f([1, 2, 3, 4, 5], [1, 0, -1])` by hand. `np.convolve(x, w, "valid")` computes the
mathematical convolution of `x` and `w`; state precisely how `f`'s result differs from it, and why that
difference does not matter when `w` is a set of learned weights rather than a fixed, known kernel. What does
`f` return when `len(w) > len(x)`?

**Q2.**

```py
def g(x, w):
    p = len(w) // 2
    xp = [0.0] * p + list(x) + [0.0] * p
    return f(xp, w)
```

Give the length of `g`'s output in terms of $n = \mathtt{len(x)}$, once for odd $\mathtt{len(w)}$ and once for
even $\mathtt{len(w)}$. Trace `g([1, 2, 3, 4, 5], [1, 0, -1])`, and explain why its last entry is not what the
interior entries would suggest. Name three padding schemes other than filling with zero, and state a property
the input signal must lack near its boundary for zero-padding to bias the values `g` produces there.

**Q3.**

```py
def h(x, w, stride=1, dilation=1, pad=0):
    xp = [0.0] * pad + list(x) + [0.0] * pad
    span = dilation * (len(w) - 1) + 1
    return [sum(xp[i + dilation * j] * w[j] for j in range(len(w)))
            for i in range(0, len(xp) - span + 1, stride)]
```

`h` generalises `f` with a `stride` (keep only every `stride`-th window), a `dilation` (space the
$k = \mathtt{len(w)}$ taps `dilation` apart instead of consecutively) and explicit zero-`pad`ding on each side.
Derive a closed-form formula for the length of `h`'s output in terms of $n = \mathtt{len(x)}$, $k$, $p =
\mathtt{pad}$, $s = \mathtt{stride}$ and $d = \mathtt{dilation}$, and evaluate it for $n=10,\,k=3,\,p=1,\,s=2,\,
d=2$. Define the *receptive field* of one entry of an output as the number of entries of the original input
it depends on. Now stack $L$ copies of `h`, each with stride $1$, the same odd kernel size $k$, and dilation
$2^\ell$ at the $\ell$-th copy ($\ell = 0, \dots, L-1$, so the first copy is not dilated at all), each padded
just enough to keep the sequence the same length at every stage. Derive the receptive field of one entry of
the last copy's output, far from both ends of the sequence, as a function of $k$ and $L$.

### Numerics

**Q4.**

```py
import math

def lse(z):
    m = max(z)
    s = sum(math.exp(v - m) for v in z)
    return m + math.log(s)
```

What does `lse` compute, and why does subtracting `m` before calling `exp` matter for inputs of large
magnitude, given that floating-point arithmetic can represent only a finite range of values without
overflowing to infinity? Give one input on which `lse` raises an exception, and one non-empty input on which
it silently returns `nan` instead — for the second, say exactly which sub-expression produces the `nan`, and
why — then fix the function so it handles that second case correctly (it should still raise on the first).
Given `lse`, how would you compute the softmax of `z` (the vector of the same shape as `z` whose $i$-th entry
is $e^{z_i} / \sum_k e^{z_k}$) without writing a second, separately stabilised implementation?

**Q5.**

```py
def stats(xs):
    n, mean, m2 = 0, 0.0, 0.0
    for x in xs:
        n += 1
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)
    return mean, m2 / (n - 1)
```

What does `stats` compute, reading one value of `xs` at a time and never storing the sequence itself? Name the
textbook variance formula, also computable in one pass from running sums, that this method is preferred over in
practice, and give four numbers sharing one large common offset (so that their differences are much smaller
than the numbers themselves) on which that formula is badly wrong in IEEE 754 double precision while `stats` is
not. What does `stats([5.0])` do, and what
does `stats([])` return? Which of the two is the worse failure, and why? Given the triple `(n, mean, M2)` that
`stats` computes internally for one chunk of data (where `M2` is the running sum `m2`, before the final
division) and the analogous triple for a second, disjoint chunk, give a formula for the triple that processing
the concatenation of the two chunks would have produced.

### Attention

**Q6.**

```py
import numpy as np

def weights(scores):
    T = scores.shape[0]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -1e9, scores)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
```

`scores` is a square array of shape `(T, T)`. Say, for row `i` of the result, which columns `j` receive
nonzero weight, and what the row sums to. Suppose a second mask is combined with `mask` (for instance, marking
certain columns as padding) so that every entry of one particular row ends up masked. Compare what `weights`
returns for that row if `-1e9` is replaced by `-np.inf`. Separately, say what goes wrong if `scores` is cast to
`np.float16` before the call, and how you would change the masking constant to avoid it. State the memory cost
of `scores` itself, for sequence length `T` and one attention head, as a formula in `T`.

### Sampling and geometry

**Q7.**

```py
def sample(stream, k, rng):
    res = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = rng.randrange(i + 1)
            if j < k:
                res[j] = x
    return res
```

`stream` is an iterable that need not have a known or finite length in advance, and `rng` is a source of
randomness with a method `randrange(m)` returning a uniform integer in $\{0, \dots, m-1\}$. What does `sample`
compute? For a stream of exactly $n \ge k$ items, prove that every item ends up in the returned list with the
same probability, and give that probability. What changes, quantitatively, if `rng.randrange(i + 1)` is
replaced by `rng.randrange(i)`?

**Q8.**

```py
import numpy as np

def dist(X, Y):
    sq = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.maximum(sq, 0))
```

`X` and `Y` are arrays of shape `(n, d)` and `(m, d)`: `n` and `m` points, respectively, in `d` dimensions.
What does `dist` compute? Without the `np.maximum`, `sq` would occasionally be fed to `np.sqrt` as a small
negative number; say in which regime of `X` and `Y` this happens, and what `np.sqrt` does with a negative
input. Separately, explain why the values `dist` returns can be inaccurate for two points that are close to
each other but both far from the origin, and give a fix that keeps the same $O(nmd)$-time approach. State the
time and memory cost of computing the full matrix `dist` returns.

## Reference solution

<details>
<summary>Show the reference solution</summary>

One point worth pinning down aloud before answering: each function is to be explained as written, not
rewritten; where one misbehaves on a particular input (Q4, Q5 and Q6 each do), naming that input and the exact
wrong output is part of the answer.

### Convolutions

**Q1.** `f` computes the *cross-correlation* of `x` with `w`: the same sliding-dot-product operation that
deep-learning frameworks call a (1-D) convolution, although a true mathematical convolution first reverses
`w` and this one does not. Its output has length $n-k+1$ whenever $k \le n$ (one entry per position where the
whole window of `w` fits inside `x`), at a cost of $k(n-k+1)$ scalar multiplications, one per output entry per
tap. Tracing $f([1,2,3,4,5],\,[1,0,-1])$: window $i=0$ gives $1\cdot1+2\cdot0+3\cdot(-1)=-2.0$; $i=1$ gives
$2-4=-2.0$; $i=2$ gives $3-5=-2.0$, so the result is $[-2.0,-2.0,-2.0]$. `np.convolve(x, w, "valid")` reverses
`w` before sliding it, computing $\sum_j x_{i+j}w_{k-1-j}$ rather than $f$'s $\sum_j x_{i+j}w_j$; the two agree
only when reversing `w` leaves it unchanged, and they disagree here, since $w=[1,0,-1]$ reversed is
$[-1,0,1]=-w$. The reversal is a bookkeeping convention from signal processing, where $w$ is a fixed, known
filter and flipping it keeps the composition of two convolutions associative; when $w$ is instead a set of
weights a network *learns*, nothing distinguishes "trained on $w$" from "trained on $w$ reversed" — training
simply settles on whichever orientation fits the data — so frameworks drop the flip and call the flip-free
operation "convolution" instead. When `len(w) > len(x)`, no window of `w` fits inside `x` at all:
`range(n - k + 1)` is a `range` with a non-positive stop, which is empty, so `f` returns `[]`.

**Q2.** `g` computes the same cross-correlation as `f`, after first surrounding `x` with $p=\lfloor k/2\rfloor$
zeros on each side — "same" padding, chosen so that, for odd $k$, the output lines back up one-to-one with the
original `x`. Its length is $(n+2p)-k+1$; substituting $p=\lfloor k/2\rfloor$, for odd $k$ this is exactly $n$
(the two halves of $k-1$ exactly cancel the two paddings), and for even $k$ it is $n+1$: keeping the length
needs $k-1$ zeros in total, an odd number that cannot be split equally between the two sides, so $k/2$ on each
side is one too many (an even kernel needs asymmetric padding, $k/2$ on one side and $k/2-1$ on the other).
Tracing $g([1,2,3,4,5],\,[1,0,-1])$: $p=1$, so `xp` $=[0,1,2,3,4,5,0]$, and correlating $[1,0,-1]$ against it
gives $[-2.0,-2.0,-2.0,-2.0,4.0]$. Entry $i$ is $\mathtt{xp}_i-\mathtt{xp}_{i+2}=x_{i-1}-x_{i+1}$ (0-indexed
`x`, the padded zeros standing in for $x_{-1}$ and $x_5$): entries $1$ to $3$ are exactly `f`'s output, and the
two edge entries each read one padded zero. The first, $0-x_1=-2.0$, matches the interior only because this ramp
happens to extrapolate to exactly $0$ at index $-1$; the last is $x_3-0=4.0$ instead of the $x_3-x_5=-2$ a
longer ramp would have supplied: zero-padding does not continue the data, it inserts a value of exactly $0$
next to the true edge sample, and this particular kernel reads that as a sharp drop. Reflect padding (mirror
the samples across the edge), replicate padding (repeat the edge sample) and circular padding (wrap around to
the other end of the sequence) are three alternatives; zero-padding biases the edge values whenever the signal
does not itself decay to $0$ near its boundary, because it then puts an artificial jump at the edge. Reflect
and replicate padding continue the signal at its edge level instead, which removes the jump though not every
edge effect: on this ramp they give edge values $0$ and $-1$ rather than the slope's $-2$.

**Q3.** `h` is the general 1-D convolution used throughout deep learning: `stride` keeps only every
`stride`-th output position, `dilation` spreads the $k=\mathtt{len(w)}$ taps of `w` over a window of
$d(k-1)+1$ input positions instead of $k$ consecutive ones, and `pad` adds explicit zeros on each side,
independent of $k$. One application of the kernel spans $d(k-1)+1$ positions of the padded sequence, whose own
length is $n+2p$; a copy starting at padded position $i$ needs $i+d(k-1)+1 \le n+2p$, so the valid starting
positions are $i=0,1,\dots,(n+2p-d(k-1)-1)$, and keeping every $s$-th one of these $n+2p-d(k-1)$ positions
gives $$\left\lfloor \frac{n+2p-d(k-1)-1}{s} \right\rfloor + 1$$ of them. At $n=10,k=3,p=1,s=2,d=2$: $d(k-1)=4$,
so the formula gives $\lfloor(10+2-4-1)/2\rfloor+1=\lfloor 7/2\rfloor+1=3+1=4$. For the receptive field: with
padding chosen at each layer to keep the sequence length fixed, one output entry of a single dilated layer is
a function of $k$ input entries spaced $d_\ell=2^\ell$ apart, i.e. it spans $d_\ell(k-1)+1$ consecutive
entries of *that* layer's own input — so layer $\ell$ adds $d_\ell(k-1)=(k-1)2^\ell$ to whatever receptive
field the layers before it had already built up (a length-preserving layer only widens the span the next layer
sees; it never narrows what earlier layers had already brought in). Starting from a single position (before
any layer, the receptive field is $1$) and summing the contribution of layers $\ell=0,\dots,L-1$:

$$\text{receptive field} = 1+(k-1)\sum_{\ell=0}^{L-1}2^\ell = 1+(k-1)(2^L-1),$$

using the geometric sum $\sum_{\ell=0}^{L-1}2^\ell=2^L-1$. This is the standard WaveNet argument for why
stacking layers with exponentially growing dilation reaches a receptive field that itself grows exponentially
in $L$ using only $L$ layers and $O(Lk)$ parameters, rather than the $O(2^Lk)$ a single, non-dilated layer
would need to match it.

### Numerics

**Q4.** `lse` computes $\log\sum_i e^{z_i}$ ("log-sum-exp") without ever forming $\sum_i e^{z_i}$ itself.
Subtracting $m=\max(z)$ before calling `exp` matters because Python's `math.exp`, unlike NumPy's, does not
quietly overflow to `inf`: once its argument exceeds $\ln(\texttt{sys.float\_info.max}) \approx 709.78$, it
raises `OverflowError` outright, so without the subtraction `lse` would crash the moment any single entry of
`z` was that large, rather than merely losing accuracy. After subtracting $m$, the largest exponent actually
passed to `exp` is exactly $0$ and every other one is smaller, so the sum can never overflow, whatever the
magnitude of `z`; and since the term for the maximum is $e^0=1$, the sum is at least $1$, so it cannot underflow
to $0$ either — unshifted, an input such as $[-1000,-1001]$ makes every `exp` underflow to `0.0`, and
`math.log(0.0)` raises `ValueError`. `lse([])` raises `ValueError` at its first line, since `max` of an empty
sequence is undefined. `lse([-math.inf, -math.inf])` returns `nan`: `m` is `-math.inf`, and the generator then
computes `v - m` for `v = -math.inf`, i.e. `-math.inf - (-math.inf)`, which IEEE 754 defines as `nan` rather
than $0$ — `s` becomes `nan`, and `m + math.log(s)` stays `nan`. The mathematically correct value is
$-\infty$ (the log of a sum of terms that are each exactly $0$), so this is a genuine bug, not merely
representational noise, and it is worth an explicit check: `if m == -math.inf: return -math.inf` before the
loop leaves every ordinary input unchanged (there, `m` is finite) and returns the right answer whenever every
entry of `z` is $-\infty$ (the only way `m` itself can be $-\infty$). Given `lse`, `softmax(z)_i` is
`math.exp(z_i - lse(z))`: since $\mathtt{lse}(z)=m+\log\sum_k e^{z_k-m}$, $z_i-\mathtt{lse}(z) =
(z_i-m)-\log\sum_k e^{z_k-m}$, and exponentiating gives $e^{z_i-m}/\sum_k e^{z_k-m} = e^{z_i}/\sum_k e^{z_k}$
(the common factor $e^{-m}$ cancels top and bottom exactly) — the same stabilised exponent `lse` already
computed internally, reused rather than re-derived.

**Q5.** `stats` computes the running sample mean and the *unbiased* sample variance (dividing the sum of
squared deviations by $n-1$, not $n$) of the values seen so far, updating both after every new value without
ever storing the sequence itself — this is Welford's algorithm. It is preferred to the textbook formula
$\mathrm{Var}(x)=E[x^2]-E[x]^2$ (one pass over running sums of $x$ and $x^2$) because that formula subtracts two
numbers that are each large whenever the data is, while their difference need not be: on
$x=10^9+[4,7,13,16]$, the true sample variance is exactly $30$ (shifting every value by the same constant
$10^9$ changes none of the pairwise differences variance is built from), and `stats` returns it at full float64
precision, but $E[x^2]-E[x]^2$ first squares the data into numbers of order $10^{18}$ — a magnitude at which
float64's roughly $15$–$17$ significant decimal digits no longer resolve a difference of order $30$ (one unit in
the last place there is $128$) — and in float64 it returns exactly $-128.0$: not merely imprecise, but
negative, an impossible value for a variance. `stats` never squares the offset: `m2` accumulates products of
deviations from the running mean, each the difference of two numbers of order $10^9$, which float64 resolves to
about $10^{-7}$, so every term stays accurate however large the common offset is. `stats([5.0])` divides by `n - 1 = 0`, raising
`ZeroDivisionError` — with a single point, sample variance is undefined and the function refuses to guess.
`stats([])` never enters the loop, so it returns `(0.0, 0.0 / (0 - 1))`, i.e. `(0.0, -0.0)`: a value that looks
like an unremarkable, legitimate summary of "no data" rather than an error. The empty case is the worse
failure: `ZeroDivisionError` stops the caller immediately, while `(0.0, -0.0)` is silent and could flow into
further computation (an average of variances, a check `variance > 0`) unnoticed, for exactly the input most
likely to arise by accident — an empty group in a batch. Merging two chunks' triples, after Chan, Golub and
LeVeque: with $n=n_a+n_b$ and $\delta=\mathrm{mean}_b-\mathrm{mean}_a$,

$$\mathrm{mean} = \mathrm{mean}_a+\delta\,\frac{n_b}{n}, \qquad M_2 = M_{2,a}+M_{2,b}+\delta^2\,\frac{n_a n_b}{n},$$

which reduces to exactly one step of `stats`'s own update when $n_b=1$ (there, $M_{2,b}=0$) — the formula used
to combine statistics computed independently on separate chunks of data, without ever materialising their
concatenation, in distributed or batched training.

### Attention

**Q6.** `weights` computes causal (autoregressive) self-attention weights from a $(T,T)$ score matrix: row $i$
of the result has nonzero weight only on columns $j\le i$ (the strict upper triangle, $j>i$, is masked before
the softmax), and every row sums to exactly $1$ — a valid probability distribution over the positions up to
and including $i$. Suppose a further mask fully masks one entire row (every column, including the diagonal —
this happens, for instance, when row $i$ is itself a padding query and a key-side padding mask excludes it
from attending anywhere, including to itself). With `-np.inf`, every entry of that row's `scores` becomes
`-inf`; `scores.max(...)` on that row is then `-inf` too, so every subtracted exponent is
`-inf - (-inf) = nan`, and the whole row comes out `nan`. `-1e9` avoids this: every entry of the row still
equals the same finite value $-10^9$, so the max-subtraction gives $0$ everywhere, `exp(0)=1` for all $T$
columns, and the row comes out as the *uniform* distribution $1/T$ instead — no crash, but a silent leak of
attention weight onto positions that were supposed to be excluded entirely, which then has to be zeroed out
explicitly downstream (for instance by re-applying the padding mask to the attention output) rather than
trusted to vanish on its own. Casting `scores` to `np.float16` breaks the same masking constant from the other
direction: float16's largest finite magnitude is about $65504$, so `-1e9` itself overflows to `-inf` under the
cast, silently reintroducing exactly the `nan` failure just described even though the code never mentions
`-inf` anywhere. The fix is to mask with the array's own dtype minimum, `np.finfo(scores.dtype).min`, rather
than a constant tuned for float32 or float64 — large enough in magnitude to dominate any real score, but
always representable in whatever dtype `scores` happens to be. `scores` itself costs $T^2$ elements — one
such matrix per attention head, quadratic in sequence length regardless of the model's hidden size, and the
dominant memory cost of computing attention at long sequence lengths.

### Sampling and geometry

**Q7.** `sample` computes a uniform, without-replacement sample of $k$ items from a stream of unknown or
unbounded length — *reservoir sampling* (Algorithm R): after any prefix of the stream has been seen, `res`
holds $k$ of its items, each an equally likely representative of everything seen so far. Proof that every one
of $n\ge k$ items ends up in the reservoir with probability $k/n$, by induction on the number of items
processed: the first $k$ items start in the reservoir with probability $1$. When item $i$ (0-indexed, $i\ge k$)
arrives, it is placed with probability $k/(i+1)$ — that is $P(j<k)$ for $j$ uniform on $\{0,\dots,i\}$ — and if
placed, it evicts a uniformly random one of the $k$ occupied slots (since, conditional on $j<k$, `j` itself is
uniform on $\{0,\dots,k-1\}$). Suppose, inductively, that every one of the first $i$ items is currently in the
reservoir with probability $k/i$ (true at $i=k$, where it reads $k/k=1$); a fixed such item survives item
$i$'s arrival with probability

$$1-\underbrace{\frac{k}{i+1}}_{\text{item }i\text{ is placed}}\cdot\underbrace{\frac1k}_{\text{and evicts this slot}} = \frac{i}{i+1},$$

so its probability of being in the reservoir becomes $\frac{k}{i}\cdot\frac{i}{i+1}=\frac{k}{i+1}$ — the
induction hypothesis one step later, with $i$ replaced by $i+1$ — and this already matches item $i$'s own
placement probability $k/(i+1)$, so the hypothesis holds for all $i+1$ items alike. By induction the common
probability is $k/i$ after any $i\ge k$ items have been processed, and in particular $k/n$ once the whole
stream has been: the same value for every item. (Equal inclusion probabilities alone would not make every
$k$-subset equally likely; the same induction run on whole subsets shows that each of the $\binom nk$ possible
reservoirs has probability $1/\binom nk$.) Replacing `rng.randrange(i + 1)` with `rng.randrange(i)` changes
item $i$'s own placement probability to $k/i$ and, by the identical argument with $i+1$ replaced by $i$
throughout, a surviving item's per-step survival probability to $(i-1)/i$ instead of $i/(i+1)$. The survival
factors telescope: an item of the initial fill ends in the reservoir with probability
$\prod_{i=k}^{n-1}\frac{i-1}{i}=\frac{k-1}{n-1}$, and an item $a\ge k$ with probability
$\frac{k}{a}\prod_{i=a+1}^{n-1}\frac{i-1}{i}=\frac{k}{a}\cdot\frac{a}{n-1}=\frac{k}{n-1}$, whatever its arrival
index $a$: the bug does not smoothly favour later items more and more, it favours everything after the initial
fill, equally, over the initial fill itself, and the sample is no longer uniform.

**Q8.** `dist` computes the $n\times m$ matrix of Euclidean distances between the rows of `X` and the rows of
`Y`, from the identity $\lVert x-y\rVert^2=\lVert x\rVert^2+\lVert y\rVert^2-2x^\top y$ — `sq` is exactly the
right-hand side, broadcast over every pair. Without the `np.maximum`, a negative `sq` would reach `np.sqrt`,
which returns `nan` for a negative real argument (with a warning) rather than raising. This happens whenever
`X` and `Y` contain points that are genuinely very close together (so the true squared distance is near $0$)
while $\lVert x\rVert^2$ and $\lVert y\rVert^2$ are themselves large — most predictably on the diagonal of
`dist(X, X)`, where $x=y$ makes the true value exactly $0$, but $\lVert x\rVert^2$ (computed once, via
`(X ** 2).sum(1)`) and $x^\top x$ (computed again, inside the matrix product `X @ X.T`, by a different sequence
of floating-point operations) need not round to bit-identical values, so their difference can land on either
side of $0$ from rounding alone. `np.maximum(sq, 0)` only prevents the crash; it does not restore the lost
accuracy. The actual fix is to remove the large common part before squaring anything — subtract a reference
point (for instance the mean of `X` and `Y` together) from both arrays before calling `dist`, so the identity
is applied to numbers of ordinary magnitude, keeping the same $O(nmd)$ approach; for a single pair that
matters, computing $\lVert x-y\rVert$ directly from the difference `x - y` avoids the expanded identity, and
its cancellation, entirely. Time cost is dominated by the matrix product $XY^\top$, $O(nmd)$ scalar
multiply-adds (the two squared-norm vectors cost only $O(nd)$ and $O(md)$ between them); memory cost is
$O(nm)$ for the returned matrix, plus the $O(nd+md)$ already occupied by the two inputs.

<details>
<summary>Checks (runnable)</summary>

```python
import math
import random
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp as scipy_logsumexp, softmax as scipy_softmax

rng = np.random.default_rng(2026)


# ---- Q1: 1-D cross-correlation ("conv1d" without a kernel flip)
def f(x, w):
    n, k = len(x), len(w)
    out = []
    for i in range(n - k + 1):
        s = 0.0
        for j in range(k):
            s += x[i + j] * w[j]
        out.append(s)
    return out


assert f([1, 2, 3, 4, 5], [1, 0, -1]) == [-2.0, -2.0, -2.0]
assert f([1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]) == []          # len(w) > len(x): no window fits at all

for _ in range(300):
    n, k = int(rng.integers(1, 30)), int(rng.integers(1, 8))
    x, w = rng.normal(size=n), rng.normal(size=k)
    if k > n:
        assert f(list(x), list(w)) == []
        continue
    expected = f(list(x), list(w))
    assert np.allclose(expected, np.correlate(x, w, "valid"))
    assert np.allclose(expected, np.convolve(x, w[::-1], "valid"))       # convolution undoes correlation's non-flip

asym_x, asym_w = rng.normal(size=12), np.array([1.0, 0.3, -2.0])         # w reversed != w
assert not np.allclose(f(list(asym_x), list(asym_w)), np.convolve(asym_x, asym_w, "valid"))


# ---- Q2: "same"-style zero padding around f
def g(x, w):
    p = len(w) // 2
    xp = [0.0] * p + list(x) + [0.0] * p
    return f(xp, w)


assert g([1, 2, 3, 4, 5], [1, 0, -1]) == [-2.0, -2.0, -2.0, -2.0, 4.0]
for n in (5, 8, 13):
    x = list(rng.normal(size=n))
    assert len(g(x, [1.0, 0.0, -1.0])) == n              # odd kernel: length preserved
    assert len(g(x, [1.0, 0.5, 0.0, -1.0])) == n + 1      # even kernel: one longer
    assert len(f([0.0] * 2 + x + [0.0] * 1, [1.0, 0.5, 0.0, -1.0])) == n   # asymmetric k/2 and k/2 - 1: length kept

ramp = np.arange(1.0, 6.0)                                 # other padding schemes on the same ramp and kernel
assert f(list(np.pad(ramp, 1, mode="reflect")), [1, 0, -1]) == [0.0, -2.0, -2.0, -2.0, 0.0]
assert f(list(np.pad(ramp, 1, mode="edge")), [1, 0, -1]) == [-1.0, -2.0, -2.0, -2.0, -1.0]    # replicate


# ---- Q3: general 1-D convolution with stride, dilation and explicit padding
def h(x, w, stride=1, dilation=1, pad=0):
    xp = [0.0] * pad + list(x) + [0.0] * pad
    span = dilation * (len(w) - 1) + 1
    return [sum(xp[i + dilation * j] * w[j] for j in range(len(w)))
            for i in range(0, len(xp) - span + 1, stride)]


def out_len_formula(n, k, p, s, d):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


assert out_len_formula(10, 3, 1, 2, 2) == 4

for _ in range(200):
    n, k = int(rng.integers(4, 20)), int(rng.integers(1, 5))
    p, s, d = int(rng.integers(0, 4)), int(rng.integers(1, 4)), int(rng.integers(1, 4))
    if d * (k - 1) + 1 > n + 2 * p:             # the kernel must fit at least once
        continue
    x, w = rng.normal(size=n), rng.normal(size=k)
    out = h(list(x), list(w), stride=s, dilation=d, pad=p)
    assert len(out) == out_len_formula(n, k, p, s, d)

    xp = np.concatenate([np.zeros(p), x, np.zeros(p)])           # an independent construction:
    w_dilated = np.zeros(d * (k - 1) + 1)                        # insert d - 1 zeros between consecutive taps,
    w_dilated[::d] = w                                           # then correlate at stride 1 and subsample
    full = np.correlate(xp, w_dilated, "valid")
    assert np.allclose(out, full[::s])


def receptive_field(kernel_size, num_layers, margin=200):
    signal = [0.0] * (2 * margin + 1)
    signal[margin] = 1.0                          # an impulse, far from either edge of the margin
    w = [1.0] * kernel_size                        # an all-ones kernel: only reachability matters, not cancellation
    for layer in range(num_layers):
        d = 2 ** layer
        p = d * (kernel_size - 1) // 2            # "same" padding at this layer's own dilation
        signal = h(signal, w, stride=1, dilation=d, pad=p)
    touched = [i for i, v in enumerate(signal) if v != 0.0]
    return max(touched) - min(touched) + 1


for L in (1, 2, 3, 4):
    assert receptive_field(3, L) == 1 + 2 * (2 ** L - 1)
assert receptive_field(5, 3) == 1 + 4 * (2 ** 3 - 1)


# ---- Q4: log-sum-exp
def lse(z):
    m = max(z)
    s = sum(math.exp(v - m) for v in z)
    return m + math.log(s)


for _ in range(200):
    z = list(rng.normal(scale=8.0, size=int(rng.integers(1, 10))))
    assert math.isclose(lse(z), scipy_logsumexp(np.array(z)), rel_tol=1e-9)

overflow_boundary = math.log(sys.float_info.max)              # NOTE: the ~709.78 that motivates subtracting m
assert 709 < overflow_boundary < 710 and math.isfinite(math.exp(709))
try:
    math.exp(710)
    exp_overflowed = False
except OverflowError:
    exp_overflowed = True
assert exp_overflowed                                          # unshifted exp would raise, not merely lose accuracy
assert math.isclose(lse([1000.0, 999.0, 998.0]),               # lse handles it fine: every shifted exponent is <= 0
                    scipy_logsumexp(np.array([1000.0, 999.0, 998.0])), rel_tol=1e-9)
try:
    math.log(sum(math.exp(v) for v in [-1000.0, -1001.0]))    # unshifted: every exp underflows to 0.0
    log_of_zero_raised = False
except ValueError:
    log_of_zero_raised = True
assert log_of_zero_raised
assert math.isclose(lse([-1000.0, -1001.0]), scipy_logsumexp(np.array([-1000.0, -1001.0])), rel_tol=1e-12)

try:
    lse([])
    raised_on_empty = False
except ValueError:
    raised_on_empty = True
assert raised_on_empty                                        # max([]) raises, at lse's first line

assert math.isnan(lse([-math.inf, -math.inf, -math.inf]))     # NOTE: v - m is -inf - (-inf) = nan for every v here


def lse_fixed(z):
    m = max(z)
    if m == -math.inf:            # NOTE: the only way m is -inf is if every entry is -inf; the true answer is -inf
        return -math.inf
    s = sum(math.exp(v - m) for v in z)
    return m + math.log(s)


assert lse_fixed([-math.inf, -math.inf]) == -math.inf
for _ in range(50):                                            # unaffected whenever the input is not all -inf
    z = list(rng.normal(size=5))
    assert lse_fixed(z) == lse(z)

for _ in range(50):
    z = list(rng.normal(scale=5.0, size=int(rng.integers(2, 6))))
    soft = [math.exp(v - lse(z)) for v in z]
    assert np.allclose(soft, scipy_softmax(np.array(z)))


# ---- Q5: Welford's one-pass mean and unbiased variance
def stats(xs):
    n, mean, m2 = 0, 0.0, 0.0
    for x in xs:
        n += 1
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)
    return mean, m2 / (n - 1)


for _ in range(200):
    xs = rng.normal(scale=10.0, size=int(rng.integers(2, 60)))
    mean, var = stats(list(xs))
    assert math.isclose(mean, np.mean(xs), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(var, np.var(xs, ddof=1), rel_tol=1e-9, abs_tol=1e-9)

offset_data = [1e9 + 4, 1e9 + 7, 1e9 + 13, 1e9 + 16]
welford_mean, welford_var = stats(offset_data)
assert math.isclose(welford_mean, 1e9 + 10, rel_tol=1e-12)
assert math.isclose(welford_var, 30.0, rel_tol=1e-9)           # exact: shifting by a constant leaves variance alone
arr = np.array(offset_data)
naive_var = float(np.mean(arr ** 2) - np.mean(arr) ** 2)       # NOTE: squares numbers of order 1e18 first
assert math.isclose(naive_var, -128.0, rel_tol=1e-6)           # catastrophic cancellation: negative, not just imprecise
assert np.spacing(np.mean(arr ** 2)) == 128.0                  # one unit in the last place at 1e18

try:
    stats([5.0])
    raised_on_one = False
except ZeroDivisionError:
    raised_on_one = True
assert raised_on_one

empty_mean, empty_var = stats([])
assert empty_mean == 0.0 and empty_var == 0.0
assert math.copysign(1.0, empty_var) == -1.0                   # (0.0, -0.0): silent, unlike the n=1 case above


def merge(na, mean_a, m2_a, nb, mean_b, m2_b):
    n = na + nb
    delta = mean_b - mean_a
    mean = mean_a + delta * nb / n
    m2 = m2_a + m2_b + delta ** 2 * na * nb / n
    return n, mean, m2


for _ in range(100):
    a = rng.normal(scale=5.0, size=int(rng.integers(2, 30)))
    b = rng.normal(scale=5.0, size=int(rng.integers(2, 30)))
    na, nb = len(a), len(b)
    m2_a, m2_b = np.var(a, ddof=1) * (na - 1), np.var(b, ddof=1) * (nb - 1)      # independent: straight from numpy
    n, mean, m2 = merge(na, np.mean(a), m2_a, nb, np.mean(b), m2_b)
    full = np.concatenate([a, b])
    assert n == na + nb
    assert math.isclose(mean, np.mean(full), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(m2 / (n - 1), np.var(full, ddof=1), rel_tol=1e-8, abs_tol=1e-8)


# ---- Q6: causal attention weights
def weights(scores):
    T = scores.shape[0]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -1e9, scores)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


for _ in range(50):
    T = int(rng.integers(2, 12))
    scores = rng.normal(size=(T, T))
    w_mat = weights(scores)
    assert np.allclose(w_mat.sum(axis=-1), 1.0)
    assert np.all(np.triu(w_mat, k=1) < 1e-12)
    for i in range(T):                                           # independent: per-row softmax over j <= i alone
        row = scipy_softmax(scores[i, :i + 1])
        assert np.allclose(w_mat[i, :i + 1], row, atol=1e-9)


def weights_masked(scores, row_fully_masked, fill):
    T = scores.shape[0]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    mask[row_fully_masked, :] = True                              # e.g. a padding query: every key masked too
    scores = np.where(mask, fill, scores)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


T = 6
scores = rng.normal(size=(T, T))
with np.errstate(invalid="ignore"):
    row_neg_inf = weights_masked(scores, 0, -np.inf)
assert np.isnan(row_neg_inf[0]).all()                              # -inf: max is -inf, 0/0 -- a silent nan
row_big_neg = weights_masked(scores, 0, -1e9)
assert np.allclose(row_big_neg[0], np.full(T, 1.0 / T))            # -1e9: a uniform leak, not a crash

with np.errstate(over="ignore"):
    assert np.isneginf(np.float16(-1e9))                           # NOTE: -1e9 overflows fp16's ~65504 range to -inf
fp16_min = np.finfo(np.float16).min
assert not np.isneginf(np.float16(fp16_min))                        # the dtype's own minimum survives the cast
scores16 = scores.astype(np.float16)
with np.errstate(over="ignore", invalid="ignore"):
    assert np.isnan(weights_masked(scores16, 0, -1e9)[0]).all()     # in fp16, -1e9 becomes -inf: the nan row is back
    fixed16 = weights_masked(scores16, 0, fp16_min)
assert np.allclose(fixed16[0].astype(float), 1.0 / T, atol=1e-3)    # masking with the dtype's minimum: uniform again
assert np.allclose(fixed16.astype(float).sum(axis=-1), 1.0, atol=1e-2)

sizes = {T2: weights(rng.normal(size=(T2, T2))).nbytes for T2 in (16, 64)}   # memory: T^2 entries per head
assert sizes[16] == 16 * 16 * 8 and sizes[64] == 16 * sizes[16]              # 4x the length, 16x the memory


# ---- Q7: reservoir sampling (Algorithm R)
def sample(stream, k, rng):
    res = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = rng.randrange(i + 1)
            if j < k:
                res[j] = x
    return res


def sample_biased(stream, k, rng):
    res = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = rng.randrange(i)                  # NOTE: should be randrange(i + 1); the bug under test
            if j < k:
                res[j] = x
    return res


N, K, TRIALS = 20, 5, 40000
counts, counts_biased = np.zeros(N), np.zeros(N)
py_rng = random.Random(2026)
for _ in range(TRIALS):
    for item in sample(list(range(N)), K, py_rng):
        counts[item] += 1
    for item in sample_biased(list(range(N)), K, py_rng):
        counts_biased[item] += 1
freq, freq_biased = counts / TRIALS, counts_biased / TRIALS
assert np.allclose(freq, K / N, atol=0.02)                          # correct: every position close to k / n

# buggy: a per-step survival factor of (m - 1) / m instead of m / (m + 1) telescopes to a closed form --
# every item from the initial fill lands on (k - 1) / (n - 1), every later one on the higher k / (n - 1)
assert np.allclose(freq_biased[:K], (K - 1) / (N - 1), atol=0.02)
assert np.allclose(freq_biased[K:], K / (N - 1), atol=0.02)
assert freq_biased[K:].min() > freq_biased[:K].max() + 0.03         # a visible, uniform step at the fill boundary

N_SUB, K_SUB, TRIALS_SUB = 6, 2, 30000                              # uniform over whole k-subsets, not only marginals
subset_counts = {}
for _ in range(TRIALS_SUB):
    key = tuple(sorted(sample(range(N_SUB), K_SUB, py_rng)))
    subset_counts[key] = subset_counts.get(key, 0) + 1
assert len(subset_counts) == math.comb(N_SUB, K_SUB)
assert all(abs(c / TRIALS_SUB - 1 / math.comb(N_SUB, K_SUB)) < 0.01 for c in subset_counts.values())


# ---- Q8: pairwise Euclidean distances via the norm expansion
def dist(X, Y):
    sq = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.maximum(sq, 0))


X8, Y8 = rng.normal(size=(15, 4)), rng.normal(size=(10, 4))
assert np.allclose(dist(X8, Y8), cdist(X8, Y8))

offset = rng.normal(size=8) * 1e8
Xc = offset + rng.normal(scale=50.0, size=(60, 8))                   # points spread widely, but all far from the origin
sq_raw = (Xc ** 2).sum(1)[:, None] + (Xc ** 2).sum(1)[None, :] - 2 * Xc @ Xc.T   # NOTE: computed without the clip
assert np.min(np.diag(sq_raw)) < 0.0                                 # rounding: some "self-distances" go slightly negative
d_self = dist(Xc, Xc)
assert np.all(np.isfinite(d_self))                                   # np.maximum keeps every one of them finite
off_diag = d_self[~np.eye(60, dtype=bool)]
assert np.max(d_self[np.diag_indices(60)]) < 0.3 * np.min(off_diag)  # ... though the leftover "self-distance" is small

d = 16
base = rng.normal(size=d) * 1e8
delta_x, delta_y = rng.normal(size=d), rng.normal(size=d)             # two points close together, both far from 0
X_far, Y_far = (base + delta_x)[None, :], (base + delta_y)[None, :]
true_dist = np.linalg.norm(delta_x - delta_y)                         # ground truth: a direct, well-conditioned subtraction
naive = dist(X_far, Y_far)[0, 0]
ref = np.vstack([X_far, Y_far]).mean(axis=0)                          # the fix: subtract the points' own mean first
centred = dist(X_far - ref, Y_far - ref)[0, 0]
naive_rel_err = abs(naive - true_dist) / true_dist
centred_rel_err = abs(centred - true_dist) / true_dist
assert centred_rel_err < 1e-8
assert naive_rel_err > 1000 * centred_rel_err

print("all checks passed")
```

</details>

</details>
