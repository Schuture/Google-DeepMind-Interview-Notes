# Attention with a KV Cache and an Online Softmax

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · ML implementation (NumPy) | ★★★★☆ | Hard | RS · RE · MLE | attention, kv-cache, online-softmax, flash-attention, grouped-query-attention, numerical-stability | 3 parts / 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

Causal self-attention maps an input sequence to an output sequence of the same shape, where position
$t$'s output depends only on positions $0,\dots,t$. This problem implements three views of that same
computation: a direct batched form, an autoregressive decoding path that reuses previously computed keys
and values, and a memory-bounded blockwise form.

Across all three parts, $x \in \mathbb{R}^{T\times d}$ is an input of $T$ positions (rows, indexed
$0,\dots,T-1$) with $d$ features each; weight matrices $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d\times d}$
project it to queries, keys, values and the final output. An integer number of *heads* $h$ splits the $d$
features into $h$ contiguous blocks of $d_h = d/h$ features each: head $i$ owns columns
$i\cdot d_h,\dots,(i+1)\cdot d_h - 1$ of $Q = xW_Q$, $K = xW_K$, $V = xW_V$. Writing
$Q_i, K_i, V_i \in \mathbb{R}^{T\times d_h}$ for head $i$'s slice, head $i$ computes

$$\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_h}} + M\right)V_i, \qquad
M_{ts} = \begin{cases}0 & s \le t\\ -\infty & s > t\end{cases},$$

the softmax taken over the key axis $s$, so that row $t$ of $M$ hides every key after position $t$
(*causal masking*), and computed in a numerically stable way (subtract the row maximum before
exponentiating). Concatenating $\mathrm{head}_0,\dots,\mathrm{head}_{h-1}$ along the feature axis and
multiplying by $W_O$ gives the final output, shape $(T, d)$.

### Part 1 — Causal multi-head attention

```py
def causal_mha(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
               n_heads: int) -> np.ndarray:
    """x: (T, d). Wq, Wk, Wv, Wo: (d, d). d must be divisible by n_heads; raises ValueError
    otherwise. Returns (T, d)."""
```

Compute the formula above for all `n_heads` heads and all $T$ positions at once. Raise `ValueError` if
`d % n_heads != 0`.

With `n_heads = 1` and $W_Q = W_K = W_V = W_O = I_2$ (so $Q = K = V = x$):

```text
x = [[1, 0],
     [0, 1],
     [1, 1]]

row 0 attends only to itself (causal) -> output[0] = V[0] = [1, 0]

row 1 attends to rows 0, 1:
  raw scores Q[1].K[0], Q[1].K[1] = 0, 1; scaled by 1/sqrt(2): 0, 0.7071
  weights ~= [0.3302, 0.6698]
  output[1] ~= 0.3302*[1,0] + 0.6698*[0,1] = [0.3302, 0.6698]

row 2 attends to rows 0, 1, 2:
  raw scores Q[2].K[0], Q[2].K[1], Q[2].K[2] = 1, 1, 2; scaled by 1/sqrt(2): 0.7071, 0.7071, 1.4142
  weights ~= [0.2483, 0.2483, 0.5035]
  output[2] ~= 0.2483*[1,0] + 0.2483*[0,1] + 0.5035*[1,1] = [0.7517, 0.7517]
```

### Part 2 — Incremental decoding with a KV cache

*Autoregressive decoding* produces the positions of a sequence one at a time, each new position's input
$x_t$ depending on the output already produced at position $t-1$. Recomputing `causal_mha` on the whole
prefix $x_0,\dots,x_t$ for every new $t$ repeats work: position $s<t$'s key and value do not depend on
any position after $s$, so they need computing only once. A *KV cache* holds, per head, the keys and
values of every position processed so far.

```py
class KVCache:
    def __init__(self, n_heads: int, d_head: int) -> None: ...

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        """k, v: (n_heads, d_head), one new position's keys and values. Raises ValueError
        if either does not have that shape."""

    def __len__(self) -> int: ...    # number of positions appended so far


def decode_step(x_t: np.ndarray, cache: KVCache, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                 Wo: np.ndarray, n_heads: int) -> np.ndarray:
    """x_t: (d,), one new position's input. Appends this position's keys and values to cache,
    then returns the (d,) output row for this position, attending over every position cache
    now holds (including this one)."""
```

For any $x, W_Q, W_K, W_V, W_O$, `n_heads`, constructing one fresh `KVCache` and calling
`decode_step(x[t], cache, ...)` for $t = 0,\dots,T-1$ in order must, at every step $t$, return row $t$ of
`causal_mha(x, Wq, Wk, Wv, Wo, n_heads)` to within an absolute tolerance of $10^{-10}$ — `decode_step`
never sees any row of $x$ except the one just passed in, so this is only possible because everything it
needs about earlier positions already sits in `cache`. On the same example as Part 1:

```text
cache = KVCache(n_heads=1, d_head=2)                  # len(cache) == 0

decode_step(x[0], cache, ...) -> [1, 0]                # len(cache) == 1; equals causal_mha(x, ...)[0]
decode_step(x[1], cache, ...) -> [0.3302, 0.6698]      # len(cache) == 2; equals causal_mha(x, ...)[1]
decode_step(x[2], cache, ...) -> [0.7517, 0.7517]      # len(cache) == 3; equals causal_mha(x, ...)[2]
```

Separately, implement

```py
def kv_cache_bytes(n_layers: int, n_kv_heads: int, d_head: int, seq_len: int, batch: int,
                    bytes_per_elem: int) -> int:
    """Total bytes of every layer's KV cache for `batch` sequences of length seq_len, with
    n_kv_heads key/value heads of size d_head per layer and bytes_per_elem bytes per stored
    number."""
```

For $L=32$ layers, $h_{kv}=8$ key/value heads, $d_h=128$, a sequence length of $T=32{,}768$, batch $1$
and $2$ bytes per element, `kv_cache_bytes(32, 8, 128, 32_768, 1, 2)` is exactly $2^{32}$.

### Part 3 — Attention in blocks with an online softmax

This part drops the causal mask and the weight matrices, and works directly with one head's
already-projected queries, keys and values: $Q \in \mathbb{R}^{T_q\times d_h}$,
$K, V \in \mathbb{R}^{T_k\times d_h}$, computing $O = \mathrm{softmax}(QK^\top/\sqrt{d_h})V \in
\mathbb{R}^{T_q\times d_h}$ (no mask). Implement it so that it never holds the full $T_q \times T_k$ score
matrix in memory: visit the keys in consecutive blocks of `block` positions each
($K[0{:}\mathrm{block}], K[\mathrm{block}{:}2\cdot\mathrm{block}], \dots$, the last block possibly
shorter), and after visiting a block, update, for every query row, a running maximum score $m$, a running
normaliser $\ell$ and a running unnormalised output $o \in \mathbb{R}^{d_h}$, such that once every block
has been visited, $o/\ell$ equals that query's row of $O$ exactly. Before any block is visited, $m =
-\infty$, $\ell = 0$, $o = 0$. Derive how $m, \ell, o$ must be updated from one block to the next so that
this invariant holds after every block, not only the last one, and prove it by induction on the number of
blocks visited.

```py
def attention_blockwise(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block: int) -> np.ndarray:
    """Q: (T_q, d_h). K, V: (T_k, d_h). Returns softmax(Q @ K.T / sqrt(d_h)) @ V, computed
    `block` positions of K/V at a time, for any block >= 1."""
```

The result must equal the direct computation above for every `block` from $1$ to more than $T_k$, and must
stay finite even when the raw scores $QK^\top/\sqrt{d_h}$ are of order $10^3$ or larger, where `np.exp` of
an unshifted score already overflows.

With $Q=[[1]]$, $K=[[0],[1],[3],[2]]$, $V=[[10],[20],[30],[40]]$ ($d_h=1$, one query, `block = 2`, so raw
scores $QK^\top=[0,1,3,2]$ split into two blocks $[0,1]$ and $[3,2]$):

```text
before any block: m = -inf, l = 0, o = [0]

block [0, 1]: local max = 1 -> m = 1
  l = 1.3679   (= exp(0-1) + exp(1-1))
  o = [23.6788]   (= exp(0-1)*10 + exp(1-1)*20)

block [3, 2]: local max = 3 -> m = 3 (up from 1, so block [0, 1]'s accumulators are rescaled by exp(1-3))
  l = exp(1-3)*1.3679 + (exp(3-3) + exp(2-3)) = 1.5530
  o = exp(1-3)*[23.6788] + (exp(3-3)*30 + exp(2-3)*40) = [47.9198]

output = o / l = [30.8562]
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points worth confirming with the interviewer before coding: `decode_step` and `KVCache` serve a
single sequence at a time, with no batch axis (matching `x_t`'s `(d,) -> (d,)` signature) — a batched
decoder would give every cached tensor an extra leading batch dimension, but the update rule below is
unchanged; and every array is the NumPy default `float64` unless a part says otherwise, so only Part 3's
large-score stress test and the low-precision follow-up ever touch precision at all.

### Part 1

$Q$, $K$, $V$ are each one projection, `x @ W`, shape $(T, d)$. Splitting the last axis into $h$ heads of
$d_h$ features and moving the head axis first is a `reshape` followed by a `transpose`, giving shape
$(h, T, d_h)$; NumPy's `@` then batches the per-head score and weighted-value products over that leading
head axis exactly as it would over a leading batch axis. Because position $t$ always sees itself
($M_{tt}=0$, never $-\infty$), `scores.max(axis=-1)` is never $-\infty$ here, so the stable softmax never
divides $0/0$.

```python
import numpy as np


def causal_mha(x, Wq, Wk, Wv, Wo, n_heads):
    T, d = x.shape
    if d % n_heads != 0:
        raise ValueError(f"d={d} is not divisible by n_heads={n_heads}")
    d_h = d // n_heads

    Q, K, V = x @ Wq, x @ Wk, x @ Wv                                 # each (T, d)
    Qh = Q.reshape(T, n_heads, d_h).transpose(1, 0, 2)                # (h, T, d_h)
    Kh = K.reshape(T, n_heads, d_h).transpose(1, 0, 2)                # (h, T, d_h)
    Vh = V.reshape(T, n_heads, d_h).transpose(1, 0, 2)                # (h, T, d_h)

    # NOTE: scale by sqrt(d_h), the per-head dimension -- not sqrt(d), which equals d_h only when n_heads == 1.
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(d_h)                 # (h, T, T)
    visible = np.tril(np.ones((T, T), dtype=bool))                     # visible[t, s]: position t may see key s
    # NOTE: mask before the softmax max/exp, not after normalising -- zeroing weights post hoc would
    # leave the visible weights summing to less than 1 instead of exactly 1.
    scores = np.where(visible, scores, -np.inf)

    m = scores.max(axis=-1, keepdims=True)                             # (h, T, 1); never -inf, t always sees itself
    weights = np.exp(scores - m)
    weights /= weights.sum(axis=-1, keepdims=True)

    head_out = weights @ Vh                                             # (h, T, d_h)
    merged = head_out.transpose(1, 0, 2).reshape(T, d)                  # (T, d)
    return merged @ Wo
```

Complexity: the four $(d,d)$ projections cost $O(Td^2)$ together; the per-head score and weighted-value
products cost $O(hT^2d_h) = O(T^2d)$, since $h\cdot d_h = d$; `scores`/`weights` use $O(hT^2)$ memory. For
long sequences ($T \gg d$) the $T^2$ term dominates; for short ones the $d^2$ projections do.

### Part 2

The cache stores keys and values, never queries: position $s$'s key and value are reused by every later
query that attends to $s$, but position $t$'s own query is used exactly once, to produce position $t$'s
output, so there is nothing to gain by keeping it around. `KVCache` keeps every appended position's keys
and values in a plain Python list (one entry per position, each of shape `(n_heads, d_head)`), and stacks
them into `(n_heads, t, d_head)` arrays on demand. `decode_step` projects only the new position $x_t$ (not
the whole prefix) through $W_Q, W_K, W_V$, appends the new key and value to the cache *before* attending —
so position $t$'s own key and value are already part of what it attends over, matching the formula's
$M_{tt}=0$ — then runs the same stable softmax as Part 1 over whichever positions the cache now holds, and
applies $W_O$.

```python
class KVCache:
    def __init__(self, n_heads, d_head):
        self.n_heads = n_heads
        self.d_head = d_head
        self._keys = []            # one (n_heads, d_head) entry per position appended so far
        self._values = []

    def append(self, k, v):
        shape = (self.n_heads, self.d_head)
        if k.shape != shape or v.shape != shape:
            raise ValueError(f"expected shape {shape}, got k={k.shape}, v={v.shape}")
        self._keys.append(k)
        self._values.append(v)

    def __len__(self):
        return len(self._keys)

    def arrays(self):
        # NOTE: stacking here costs O(len(self)); a production cache pre-allocates a fixed-size
        # buffer and writes into it instead, so a single step costs O(d_head), not O(t) (Follow-ups).
        K = np.stack(self._keys, axis=1)          # (n_heads, t, d_head)
        V = np.stack(self._values, axis=1)        # (n_heads, t, d_head)
        return K, V


def decode_step(x_t, cache, Wq, Wk, Wv, Wo, n_heads):
    d = x_t.shape[0]
    d_head = d // n_heads
    q_t = (x_t @ Wq).reshape(n_heads, d_head)
    k_t = (x_t @ Wk).reshape(n_heads, d_head)
    v_t = (x_t @ Wv).reshape(n_heads, d_head)
    cache.append(k_t, v_t)                          # NOTE: append before attending, so position t attends to itself

    K, V = cache.arrays()                             # (n_heads, t_so_far, d_head)
    scores = (q_t[:, None, :] * K).sum(axis=-1) / np.sqrt(d_head)   # (n_heads, t_so_far)
    m = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - m)
    weights /= weights.sum(axis=-1, keepdims=True)
    head_out = (weights[:, :, None] * V).sum(axis=1)    # (n_heads, d_head)
    return head_out.reshape(d) @ Wo


def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, batch, bytes_per_elem):
    return n_layers * 2 * n_kv_heads * d_head * seq_len * batch * bytes_per_elem
```

**Cost per step.** Projecting $x_t$ through $W_Q,W_K,W_V,W_O$ costs $O(d^2)$, independent of how many
positions are already cached. Attending $q_t$ (one query per head) against the $t+1$ keys now in the
cache — position $t$'s own key, just appended, plus the $t$ before it — costs, per head, $t+1$ dot
products and $t+1$ weighted sums, each of length $d_h$, i.e. $O(t\cdot d_h)$; summed over $h$ heads,
$O(t\cdot d_h\cdot h)=O(td)$. So one step costs $O(d^2+td)$, and $T$ steps cost $O(Td^2+T^2d)$ in
total — matching `causal_mha`'s own complexity (Part 1) exactly, just spread one token at a time instead
of requiring the whole sequence up front. Recomputing from scratch at step $t$ instead of caching —
rerunning `causal_mha` on the first $t+1$ positions — costs $O(t^2d)$ for that step alone once $t$ exceeds
$d$, since it repeats Part 1's own $O(t^2d)$ attention term on a prefix of length $t$; summing $O(t^2d)$
over $T$ steps costs $O(T^3d)$ in total, a factor of $T$ worse than caching, for generating the same $T$
tokens.

**The byte formula.** Each of the $L=$ `n_layers` transformer layers keeps its own cache. Each layer
stores, per cached position, one key vector and one value vector per KV head, each $d_h$ elements long,
over $h_{kv}=$ `n_kv_heads` heads, `seq_len` positions and `batch` sequences kept side by side. The
element count is therefore $L\cdot 2\cdot h_{kv}\cdot d_h\cdot\mathrm{seq\_len}\cdot\mathrm{batch}$ (the
factor $2$ for keys and values counted separately), and multiplying by `bytes_per_elem` gives the byte
count. At $L=32$, $h_{kv}=8$, $d_h=128$, $T=32{,}768$, batch $1$, $2$ bytes:
$32\cdot2\cdot8\cdot128\cdot32{,}768\cdot1\cdot2 = 2^5\cdot2\cdot2^3\cdot2^7\cdot2^{15}\cdot2 = 2^{32}$
bytes exactly, i.e. $4\ \mathrm{GiB}$.

**Why `n_kv_heads`, not `n_heads`.** The cache stores keys and values only, never queries, and
*grouped-query attention* gives $K$ and $V$ fewer heads than $Q$: it projects to $h_{kv}<h$ key/value
heads and has every group of $h/h_{kv}$ query heads attend using the same shared $K_i,V_i$ slice; ordinary
multi-head attention, as implemented above, is the special case $h_{kv}=h$ (every query head has its own
key/value head). Because $Q$ is never cached — only the current step's single query is ever used, then
discarded — `kv_cache_bytes` depends on $h_{kv}$ alone: shrinking $h_{kv}$ below $h$ shrinks the cache
without changing how many query heads attend (Follow-ups).

### Part 3

For one query row, write $s_j=(QK^\top)_j/\sqrt{d_h}$ for the raw score against key $j$, and, after
visiting blocks $B_1,\dots,B_n$ (a prefix of the partition of $\{0,\dots,T_k-1\}$ into consecutive
blocks), maintain the invariant

$$m_n=\max_{j\in B_1\cup\cdots\cup B_n}s_j,\qquad
\ell_n=\sum_{j\in B_1\cup\cdots\cup B_n}e^{s_j-m_n},\qquad
o_n=\sum_{j\in B_1\cup\cdots\cup B_n}e^{s_j-m_n}v_j,$$

so that $o_n/\ell_n$ is exactly the softmax-weighted sum over the keys visited so far ($m_n$ plays the
same role as Part 1's row maximum, just restricted to a prefix of the keys). The base case $n=0$ (no
blocks visited) holds by the stated initial values: an empty maximum is $-\infty$ and an empty sum is $0$.

**Inductive step.** Suppose the invariant holds after $n$ blocks; visit block $B_{n+1}$, with raw scores
$\{s_j : j\in B_{n+1}\}$. A maximum over a union of two sets is the maximum of their two maxima, so

$$m_{n+1}=\max\Bigl(m_n,\ \max_{j\in B_{n+1}}s_j\Bigr)$$

already equals $\max_{j\in B_1\cup\cdots\cup B_{n+1}}s_j$, matching the invariant's definition of
$m_{n+1}$. Split $\ell_{n+1}$'s defining sum over the two parts of $B_1\cup\cdots\cup B_{n+1}$:

$$\ell_{n+1}=\sum_{j\in B_1\cup\cdots\cup B_n}e^{s_j-m_{n+1}}+\sum_{j\in B_{n+1}}e^{s_j-m_{n+1}}.$$

In the first sum, $e^{s_j-m_{n+1}}=e^{s_j-m_n}\cdot e^{m_n-m_{n+1}}$ for every $j$, and $e^{m_n-m_{n+1}}$
does not depend on $j$, so it factors out of the sum, leaving
$e^{m_n-m_{n+1}}\sum_{j\in B_1\cup\cdots\cup B_n}e^{s_j-m_n}=e^{m_n-m_{n+1}}\ell_n$ by the inductive
hypothesis. So $\ell_{n+1}=e^{m_n-m_{n+1}}\ell_n+\sum_{j\in B_{n+1}}e^{s_j-m_{n+1}}$, exactly the claimed
update, and the identical factoring argument applied term by term (each $v_j$ just rides along unchanged)
gives $o_{n+1}=e^{m_n-m_{n+1}}o_n+\sum_{j\in B_{n+1}}e^{s_j-m_{n+1}}v_j$. By induction the invariant holds
after every block, in particular the last one, $N$, where $m_N=\max_js_j$ over every key and
$\ell_N=\sum_je^{s_j-m_N}$, so

$$\frac{o_N}{\ell_N}=\sum_j\frac{e^{s_j-m_N}}{\sum_ke^{s_k-m_N}}v_j=\sum_j\mathrm{softmax}(s)_j\,v_j,$$

the same max-subtraction identity used in Part 1, applied here to the whole row of scores rather than to
one block at a time. Writing the one-step update compactly, with $m,\ell,o$ the accumulators before a
block and $m',\ell',o'$ after it, and $s_j$ ranging over that block:

$$m'=\max\Bigl(m,\max_js_j\Bigr), \qquad \ell'=e^{m-m'}\ell+\sum_je^{s_j-m'}, \qquad
o'=e^{m-m'}o+\sum_je^{s_j-m'}v_j.$$

Every exponent above is $s_j-m'\le0$ (by definition of $m'$) or $m-m'\le0$ (since $m'=\max(m,\dots)\ge
m$), so no exponential in this recurrence ever overflows, for scores of any magnitude. The only way
$-\infty$ could cause trouble is $-\infty-(-\infty)$, which is `nan`, and that needs *both* sides of the
subtraction to be $-\infty$ at once. Here $m$ starts at $-\infty$, but $m'=\max(m,\max_js_j)$ is already
finite at the first block, since a block is never empty and its scores are finite; so $m-m'$ is only ever
$-\infty$ minus a finite number, i.e. a clean $-\infty$, and $e^{-\infty}=0$ with no guard needed.

```python
def attention_blockwise(Q, K, V, block):
    T_q, d_h = Q.shape
    T_k = K.shape[0]
    scale = np.sqrt(d_h)

    m = np.full(T_q, -np.inf)                # running max score, per query
    l = np.zeros(T_q)                         # running normaliser, per query
    o = np.zeros((T_q, d_h))                  # running unnormalised output, per query

    for start in range(0, T_k, block):
        end = min(start + block, T_k)                          # NOTE: the last block may be shorter than `block`
        s = Q @ K[start:end].T / scale                          # (T_q, end - start); one block of scores at a time
        m_new = np.maximum(m, s.max(axis=1))
        # NOTE: m starts at -inf, but m_new = max(m, block_max) is finite already at the first block (a
        # block is never empty), so m - m_new is -inf - finite = -inf, not -inf - (-inf) = nan -- exp(-inf)
        # is a clean 0, so no separate guard is needed here.
        correction = np.exp(m - m_new)                           # rescales the accumulators to the new max
        p = np.exp(s - m_new[:, None])
        l = correction * l + p.sum(axis=1)
        o = correction[:, None] * o + p @ V[start:end]
        m = m_new

    return o / l[:, None]
```

Complexity: total time is $O(T_qT_kd_h)$, the same as the direct computation — blocking changes only how
the work is scheduled, not how much of it there is. At any moment, memory holds one $(T_q,\mathrm{block})$
score block plus the $O(T_qd_h)$ accumulators $m,\ell,o$, instead of the full $(T_q,T_k)$ score matrix
Part 1's `scores` would need; this is the memory bound FlashAttention is named for, and is exactly why the
recurrence above matters: it lets `block` be chosen to fit in fast memory (cache or SRAM) regardless of
how large $T_k$ is, at no cost in either total arithmetic or in the final result.

### Follow-ups

- **Causal masking inside the blockwise version.** With a causal mask, block $[\mathrm{start},
  \mathrm{end})$ contributes nothing to query $t$ once $\mathrm{start}>t$: such a block is skipped
  outright, without computing any score in it at all, rather than masked entry by entry. Only a block
  straddling the diagonal (some keys $\le t$, some $>t$) still needs Part 1's per-entry $-\infty$ mask
  before the running $m/\ell/o$ update; every block entirely below the diagonal needs no masking at all,
  and every block entirely above it is skipped.
- **Grouped-query attention's effect on the cache.** Shrinking `n_kv_heads` from $h$ (ordinary multi-head
  attention) to $h_{kv}<h$ shrinks `kv_cache_bytes` by exactly the group size $h/h_{kv}$, since every other
  factor in the formula is unchanged. At $h=64$ query heads sharing $h_{kv}=8$ key/value heads (group size
  $8$), Part 2's $2^{32}$-byte ($4\ \mathrm{GiB}$) cache would instead need $2^{35}$ bytes ($32\
  \mathrm{GiB}$) under full multi-head attention with $h_{kv}=64$ — an $8\times$ difference that is pure
  cache footprint, since $Q$'s own size never enters the formula.
- **bf16 and why the max-subtraction matters more at low precision.** bf16 keeps float32's 8-bit exponent,
  so it overflows only where float32 does too (raw scores past about $88$); without the shift, its problem
  is precision, not range: 7 mantissa bits give only about 2 significant decimal digits, so once two of a
  row's exponentials differ by more than roughly a factor of $2^7$ (a score gap past about
  $\ln128\approx4.85$), summing them in bf16 rounds the smaller one away entirely — a plausible gap within
  one row of scores. float16 has more mantissa bits ($10$) but only a 5-bit exponent, so it overflows
  outright once a raw score passes about $11$ (already true of the modest gap used in the checks below).
  Subtracting the running maximum keeps every shifted exponential in $(0,1]$ regardless of the raw scores'
  range, sidestepping both failure modes at once, which is why production kernels apply it even when
  $\ell$ itself is accumulated in a low-precision type.
- **Sliding-window attention.** Cap the causal mask so query $t$ attends only to keys in
  $[\max(0,t-W+1),t]$ for a window $W$; combined with the cache, this bounds memory to $W$ entries per
  head, evicting the oldest key/value once the cache exceeds $W$ (a ring buffer of size $W$, in place of
  Part 2's unbounded list), and each `decode_step` costs $O(Wd)$ instead of $O(td)$, independent of how
  long the sequence has grown.
- **RoPE applied to cached keys.** Rotary position embeddings rotate $q_t$ and $k_t$ by an angle that
  depends only on the absolute position $t$ (and the feature pair being rotated), applied before the dot
  product. Because that rotation depends on nothing computed after position $t$, it can be applied once,
  to $k_t$, at the moment it is written into the cache — exactly where Part 2's `append` already sits —
  rather than re-derived from a stored, unrotated key on every later step that attends to it.

<details>
<summary>Checks (runnable)</summary>

```python
# Part 1: the worked example of the statement
x = np.array([[1., 0.], [0., 1.], [1., 1.]])
I2 = np.eye(2)
out1 = causal_mha(x, I2, I2, I2, I2, n_heads=1)
assert [round(float(v), 4) for v in out1[0]] == [1.0, 0.0]
assert [round(float(v), 4) for v in out1[1]] == [0.3302, 0.6698]
assert [round(float(v), 4) for v in out1[2]] == [0.7517, 0.7517]


def loop_causal_mha(x, Wq, Wk, Wv, Wo, n_heads):    # straight from the formula, explicit loops, no numpy tricks
    T, d = x.shape
    if d % n_heads != 0:
        raise ValueError(f"d={d} is not divisible by n_heads={n_heads}")
    d_h = d // n_heads
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    out = np.zeros((T, d))
    for h in range(n_heads):
        sl = slice(h * d_h, (h + 1) * d_h)
        for t in range(T):
            scores = [float(np.dot(Q[t, sl], K[s, sl])) / d_h ** 0.5 for s in range(t + 1)]
            m = max(scores)
            exps = [np.exp(sc - m) for sc in scores]
            denom = sum(exps)
            for s in range(t + 1):
                out[t, sl] += (exps[s] / denom) * V[s, sl]
    return out @ Wo


assert np.allclose(out1, loop_causal_mha(x, I2, I2, I2, I2, 1), atol=1e-8)

# Part 1: random trials against the independent loop implementation, several head counts
rng = np.random.default_rng(0)
for _ in range(150):
    T = int(rng.integers(1, 7))
    n_heads = int(rng.choice([1, 2, 4]))
    d = n_heads * int(rng.integers(1, 4))
    xr = rng.normal(size=(T, d)) * 0.7
    Ws = [rng.normal(size=(d, d)) * 0.6 for _ in range(4)]
    assert np.allclose(causal_mha(xr, *Ws, n_heads=n_heads), loop_causal_mha(xr, *Ws, n_heads=n_heads), atol=1e-8)

try:
    causal_mha(np.zeros((2, 3)), *[np.eye(3)] * 4, n_heads=2)
    assert False, "expected ValueError for d=3, n_heads=2"
except ValueError:
    pass

# Part 2: the worked example, continuing Part 1's -- decode_step reproduces causal_mha row by row
cache = KVCache(n_heads=1, d_head=2)
assert len(cache) == 0
step_outs = []
for t in range(3):
    o = decode_step(x[t], cache, I2, I2, I2, I2, n_heads=1)
    step_outs.append(o)
    assert len(cache) == t + 1
assert [round(float(v), 4) for v in step_outs[0]] == [1.0, 0.0]
assert [round(float(v), 4) for v in step_outs[1]] == [0.3302, 0.6698]
assert [round(float(v), 4) for v in step_outs[2]] == [0.7517, 0.7517]
assert np.allclose(np.array(step_outs), out1, atol=1e-10)

# Part 2: the byte formula, and grouped-query attention's cache reduction
assert kv_cache_bytes(32, 8, 128, 32_768, 1, 2) == 2 ** 32
mha_equivalent = kv_cache_bytes(32, 64, 128, 32_768, 1, 2)     # same shapes, but n_kv_heads == n_heads == 64
assert mha_equivalent == 2 ** 35
assert mha_equivalent == 8 * kv_cache_bytes(32, 8, 128, 32_768, 1, 2)     # group size 64 / 8 = 8x the cache

# Part 2: random trials -- decode_step fed one row at a time must match causal_mha and the independent loop
rng = np.random.default_rng(1)
for _ in range(80):
    T = int(rng.integers(1, 8))
    n_heads = int(rng.choice([1, 2, 4]))
    d = n_heads * int(rng.integers(1, 4))
    xr = rng.normal(size=(T, d)) * 0.7
    Ws = [rng.normal(size=(d, d)) * 0.6 for _ in range(4)]
    batch_out = causal_mha(xr, *Ws, n_heads=n_heads)
    loop_out = loop_causal_mha(xr, *Ws, n_heads=n_heads)
    c = KVCache(n_heads=n_heads, d_head=d // n_heads)
    incremental = np.stack([decode_step(xr[t], c, *Ws, n_heads=n_heads) for t in range(T)])
    assert np.allclose(incremental, batch_out, atol=1e-10)
    assert np.allclose(incremental, loop_out, atol=1e-8)

try:
    KVCache(n_heads=2, d_head=3).append(np.zeros((2, 3)), np.zeros((2, 4)))
    assert False, "expected ValueError for a mismatched value shape"
except ValueError:
    pass

# Part 3: the worked example of the statement
Q3 = np.array([[1.]])
K3 = np.array([[0.], [1.], [3.], [2.]])
V3 = np.array([[10.], [20.], [30.], [40.]])
out3 = attention_blockwise(Q3, K3, V3, block=2)
assert round(float(out3[0, 0]), 4) == 30.8562


def direct_attention(Q, K, V):        # no blocking, per-query python loop, straight from the formula
    T_q, d_h = Q.shape
    T_k = K.shape[0]
    out = np.zeros((T_q, d_h))
    for i in range(T_q):
        scores = [float(np.dot(Q[i], K[j])) / d_h ** 0.5 for j in range(T_k)]
        m = max(scores)
        exps = [np.exp(sc - m) for sc in scores]
        denom = sum(exps)
        for j in range(T_k):
            out[i] += (exps[j] / denom) * V[j]
    return out


assert np.allclose(out3, direct_attention(Q3, K3, V3), atol=1e-8)

# Part 3: random trials against the independent loop, for block sizes 1, 2, 3, T_k, T_k + 5 and random
rng = np.random.default_rng(2)
for _ in range(60):
    T_q = int(rng.integers(1, 5))
    T_k = int(rng.integers(1, 12))
    d_h = int(rng.integers(1, 5))
    Qr = rng.normal(size=(T_q, d_h)) * 0.8
    Kr = rng.normal(size=(T_k, d_h)) * 0.8
    Vr = rng.normal(size=(T_k, d_h)) * 0.8
    ref = direct_attention(Qr, Kr, Vr)
    for block in sorted({1, 2, 3, T_k, T_k + 5, int(rng.integers(1, T_k + 6))}):
        assert np.allclose(attention_blockwise(Qr, Kr, Vr, block), ref, atol=1e-8), (T_q, T_k, d_h, block)

# Part 3: scores of order 1e3 must not overflow, and must still match the (also stable) direct computation
Qs = np.array([[1000.0, 0.0]])
Ks = np.array([[2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
Vs = np.array([[1.0], [2.0], [3.0]])
big_scores = Qs @ Ks.T / np.sqrt(2)
assert np.max(np.abs(big_scores)) > 1000                          # genuinely of order 1e3, as the statement requires
with np.errstate(over="ignore"):
    assert np.isinf(np.exp(big_scores)).any()                      # confirms an unshifted exp really would overflow here
big_ref = direct_attention(Qs, Ks, Vs)
assert np.all(np.isfinite(big_ref))
for block in (1, 2, 3):
    got = attention_blockwise(Qs, Ks, Vs, block)
    assert np.all(np.isfinite(got))
    assert np.allclose(got, big_ref, atol=1e-6)

# Follow-up: bf16/float16 -- why the max-subtraction matters more at low precision
scores16 = np.array([0.0, 5.0, 12.0], dtype=np.float16)   # a spread that is entirely unremarkable in float64
with np.errstate(over="ignore"):
    naive16 = np.exp(scores16)                              # exp(12) ~= 162755, already past float16's ~65504 max
stable16 = np.exp(scores16 - scores16.max())                 # every exponent <= 0, so this stays finite in any dtype
assert np.isinf(naive16[-1]) and not np.isinf(naive16[0])
assert np.all(np.isfinite(stable16))

print("all checks passed")
```

</details>

</details>
