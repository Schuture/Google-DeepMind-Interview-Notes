# LLM Fundamentals: Attention, Transformers and the Training Pipeline

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Quiz · oral, with derivations | ★★★★★ | Hard | RS · RE · MLE · Applied AI · Intern | attention, transformer, rope, kv-cache, scaling-laws, perplexity, tokenisation, post-training, sampling, mixture-of-experts | 12 questions / 45–60 min | Skills interview |
<!-- meta:end -->

## Problem

Throughout, the model is a decoder-only transformer with $L$ layers, residual-stream ("width") dimension
$d$, $h$ attention heads of size $d_h = d / h$, and sequence length $T$. Self-attention is causal: position
$i$ may attend only to positions $j \le i$. Twelve questions, grouped by topic; each is answered aloud, most
with a short derivation or a computed number.

### Attention

**Q1.** Let $q, k \in \mathbb{R}^{d_h}$, and suppose all $2d_h$ components of $q$ and $k$ together are
independent of one another, each with mean $0$ and variance $1$. Derive $\mathrm{Var}(q^\top k)$. Then
explain precisely what the $1/\sqrt{d_h}$ scaling in scaled dot-product attention prevents.

**Q2.** Give the time cost and, separately, the memory used by the materialised score and weight
matrices, for one layer of causal multi-head self-attention, as functions of $T$ and $d$ — state
explicitly what you assume about how the head width $d_h$ behaves as $d$ grows. Then explain precisely
what FlashAttention changes about these two costs, and why.

**Q3.** Why does autoregressive decoding cache keys and values instead of recomputing them at every step?
Derive the size, in bytes, of the KV cache as a function of $L$, the number of key/value heads $h_{kv}$,
the head width $d_h$, the sequence length $T$, the batch size $b$, and the number of bytes used per stored
number. Evaluate it for $L = 32$, $h_{kv} = 8$, $d_h = 128$, $T = 32{,}768$, $b = 1$, in bf16 (2 bytes).
Then say how multi-query attention (MQA) and grouped-query attention (GQA) change $h_{kv}$, and through it
the cache.

**Q4.** Rotary position embeddings (RoPE) rotate the query at position $m$ and the key at position $n$ by
an angle proportional to their position before the dot product is taken. Working first in a single
two-dimensional coordinate pair with rotation frequency $\omega$, show that $q_m^\top k_n$ depends on $m$
and $n$ only through $m - n$. Then argue how this extends to all $d$ coordinates of a real query/key
vector. Contrast this with a learned or sinusoidal absolute position embedding added to the input before
the query/key projections.

### Architecture and cost

**Q5.** Count the parameters $N$ of one decoder block with residual width $d$, an MLP hidden size of $4d$,
no biases, and normalisation parameters ignored. Then give the total parameter count of an $L$-layer model
with a tied input/output embedding (the same $V \times d$ matrix used for both) over a vocabulary of size
$V$. Using only the block's non-embedding parameter count $N$, derive why training costs about $6N$
floating-point operations (FLOPs) per token and inference about $2N$, being explicit about which pieces of
the computation this approximation drops.

**Q6.** Contrast a pre-norm and a post-norm residual block: state where the normalisation sits relative to
the residual addition in each. Using a simplified linear model of the residual stream, in which each
sublayer's effect (including its normalisation) is summarised by a single scalar multiplier, derive why
pre-norm trains more stably at large depth than post-norm. Then state exactly what RMSNorm drops relative
to LayerNorm, and when the two coincide.

### Objective and data

**Q7.** Define the next-token cross-entropy loss and perplexity for a sequence, and relate perplexity to
bits per token. A model assigns probabilities $0.5,\ 0.2,\ 0.8,\ 0.25$ to the four ground-truth next
tokens of a length-4 sequence (each conditioned on the true tokens before it). Compute the cross-entropy,
in nats, and the perplexity.

**Q8.** Byte-pair encoding (BPE) builds a subword vocabulary by starting from individual characters and
repeatedly merging the most frequent adjacent symbol pair into a new symbol. Starting from the word counts
`low: 5, lower: 2, newest: 6, widest: 3` (each word pre-split into its characters plus a trailing
end-of-word marker `_`), run the first three merges by hand: state the merged pair and its count at each
step, breaking any tie by taking the lexicographically smaller pair (ordinary letters compare a–z, and the
end-of-word marker sorts after every letter). Then explain the trade-off that the number of merges
controls between vocabulary size and sequence length.

**Q9.** Compute-optimal scaling uses $C \approx 6ND$ for the pretraining compute in FLOPs (parameters $N$,
training tokens $D$) and the empirical rule of thumb $D \approx 20N$. Derive $N$ and $D$ for a compute
budget $C = 10^{23}$ FLOPs. Then say, without computing new numbers, what changes about the choice of $N$
if the total cost of inference over the model's deployment lifetime is also to be minimised, not training
compute alone.

### Post-training and inference

**Q10.** Pretraining, supervised fine-tuning (SFT) and preference optimisation (reinforcement learning from
human feedback, RLHF, with a learned reward model, or direct preference optimisation, DPO) are the three
stages of a typical LLM training pipeline.
State what each stage's loss optimises. Explain why the SFT loss is masked to the response tokens only.
Derive the DPO loss from a KL-regularised reward-maximisation objective, and explain why post-training
keeps a KL penalty to a reference model.

**Q11.** Define greedy decoding, temperature sampling, top-$k$ sampling and nucleus (top-$p$) sampling
precisely, each as a transformation of a next-token probability distribution, and state each one's effect
on the shape of the sampled distribution. A model outputs the distribution
`{a: 0.5, b: 0.2, c: 0.15, d: 0.1, e: 0.05}` over five tokens. Give the exact token set kept by top-$p$
sampling with $p = 0.7$.

**Q12.** In a mixture-of-experts (MoE) layer, a router sends each token to $k$ of $E$ experts, and the
rest of the network (attention, embeddings, the router itself) is shared by every token. Define total
parameters and active parameters per token, and compute both for $E = 8$ experts of $2 \times 10^{9}$
parameters each, $5 \times 10^{8}$ shared parameters, and $k = 2$. Explain why an auxiliary load-balancing
loss is needed, and what the capacity factor controls.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Confirm the architecture family before answering: everything below assumes a standard pre-norm,
multi-head (or grouped-query) decoder-only transformer with RoPE and tied embeddings — the family most
current open LLMs belong to — and that "parameters" means trainable weights only, never the KV cache or
optimiser state. Where a question asks for a number, say whether it is exact from a stated formula or a
widely used empirical rule of thumb before giving it.

### Attention

**Q1.** $\mathrm{Var}(q^\top k) = d_h$, and dividing by $\sqrt{d_h}$ is exactly what brings that variance
back to $1$ for every $d_h$.

Write $q^\top k = \sum_{i=1}^{d_h} q_ik_i$. Since $\mathrm{E}[q_i] = \mathrm{E}[k_i] = 0$ and $q_i \perp
k_i$, $\mathrm{E}[q_ik_i] = 0$, so $\mathrm{E}[q^\top k] = 0$ and

$$\mathrm{Var}(q^\top k) = \mathrm{E}\bigl[(q^\top k)^2\bigr] = \sum_{i=1}^{d_h}\sum_{j=1}^{d_h} \mathrm{E}[q_ik_iq_jk_j].$$

For $i \ne j$, $q_i, k_i, q_j, k_j$ are four mutually independent, zero-mean variables, so
$\mathrm{E}[q_ik_iq_jk_j] = \mathrm{E}[q_i]\,\mathrm{E}[k_i]\,\mathrm{E}[q_j]\,\mathrm{E}[k_j] = 0$; for
$i = j$, independence of $q_i$ and $k_i$ gives $\mathrm{E}[q_i^2k_i^2] = \mathrm{E}[q_i^2]\,\mathrm{E}[k_i^2]
= 1 \cdot 1 = 1$ (using $\mathrm{E}[q_i^2] = \mathrm{Var}(q_i) = 1$ since $\mathrm{E}[q_i]=0$). Summing the
$d_h$ surviving diagonal terms, $\mathrm{Var}(q^\top k) = d_h$.

So an unscaled dot product has standard deviation $\sqrt{d_h}$, growing with the head width. As $d_h$
grows, the pre-softmax scores spread further apart in absolute terms; softmax then saturates toward a
near one-hot vector (the largest score dominates the sum), and its Jacobian
$\partial p_i/\partial s_j = p_i(\delta_{ij}-p_j)$ goes to $0$ at every entry once the distribution is
that peaked — the gradient with respect to the scores, and so through $Q$ and $K$, vanishes and learning
stalls. Dividing $q^\top k$ by $\sqrt{d_h}$ rescales its variance to exactly $1$ regardless of $d_h$,
keeping softmax in a well-conditioned regime independent of head width.

**Q2.** Time: $\Theta(Td^2)$ for the four $d \times d$ projections ($Q,K,V,O$) plus $\Theta(T^2d)$ for the
score and weighted-sum contractions, so $\Theta(Td^2 + T^2d)$ overall. Memory beyond the $O(Td)$
activations: $\Theta(dT^2)$ elements for the materialised score/weight matrices, treating head width
$d_h$ as a fixed constant. FlashAttention removes exactly that memory term without changing the FLOP
count.

With $Q,K,V \in \mathbb{R}^{T \times d}$ each produced by a $d \times d$ projection, one projection costs
$2Td^2$ FLOPs (2 per multiply-add), and there are four of them (including the output projection $W_o$):
$8Td^2$. Splitting into $h$ heads of width $d_h = d/h$, the per-head score contraction $Q_hK_h^\top$ costs
$2Td_hT$ FLOPs; summed over $h$ heads this is $h \cdot 2T^2d_h = 2T^2(hd_h) = 2T^2d$ — independent of how
many heads it is split into, since $hd_h = d$ is fixed regardless of $h$. The weighted sum over $V$ is the
same shape of contraction, another $2T^2d$. Total: $8Td^2 + 4T^2d$ FLOPs, i.e. $\Theta(Td^2 + T^2d)$,
crossing over at $T \approx 2d$.

The score and weight arrays, in contrast, are $h$ separate $T \times T$ matrices — one per head, because
each head's softmax normalises independently — so they occupy $hT^2 = (d/d_h)\,T^2$ numbers, which is
*not* independent of the head split: doubling $h$ (halving $d_h$) doubles this memory even though it
leaves the FLOP totals above unchanged. Treating $d_h$ as a fixed constant (64–128, essentially
independent of model size) makes this memory $\Theta(dT^2)$: linear in $d$, quadratic in $T$, dominating
the $O(Td)$ activation memory once $T$ is large.

FlashAttention computes the same $8Td^2 + 4T^2d$ FLOPs (somewhat more in the backward pass, which
recomputes blocks instead of reading back a saved score matrix) but never materialises the full
$T \times T$ score/weight matrix. It tiles $K, V$ into blocks and keeps a running, unnormalised softmax —
a running max $m$, running sum $\ell$, and running output accumulator $\mathrm{acc}$ — updated one block
at a time. Merging a new block's local statistics $(m_2, \ell_2, \mathrm{acc}_2)$ into a running
$(m_1, \ell_1, \mathrm{acc}_1)$ uses $e^{s - m} = e^{s - m_i}\,e^{m_i - m}$ for either block $i$, so with
$m = \max(m_1, m_2)$,

$$\ell = \ell_1e^{m_1 - m} + \ell_2e^{m_2 - m}, \qquad \mathrm{acc} = \mathrm{acc}_1e^{m_1 - m} + \mathrm{acc}_2e^{m_2 - m},$$

and the final output is $\mathrm{acc}/\ell$ — exactly the softmax-weighted sum over every key seen,
computed while holding only one block's score matrix ($O(\text{block}\times d)$) at a time. Peak memory
drops from $\Theta(dT^2)$ to $O(Td)$, the same order as everything else in the layer.

**Q3.** Caching avoids recomputing $K, V$ for every earlier position at every new decoding step. Cache
size $= 2 \cdot L \cdot h_{kv} \cdot d_h \cdot T \cdot b \cdot \text{bytes}$, which is exactly $4$ GiB for
the given numbers.

Under a causal mask, position $j$'s key and value vectors never depend on any position after $j$; once
computed they are fixed for the rest of generation. Decoding token $T{+}1$ therefore needs only that one
new token's $Q, K, V$: it appends the new $K, V$ to a cache of every earlier position's, rather than
recomputing $K, V$ for the whole prefix — turning an $O(T)$-per-token, $O(T^2)$-total cost into $O(1)$ per
token. What must be stored: for each of the $L$ layers, both $K$ and $V$ (factor $2$), each of shape
$(b, h_{kv}, T, d_h)$ — using $h_{kv}$ key/value heads specifically, since with MQA/GQA fewer heads are
kept for $K, V$ than are used for $Q$ (below) — at some number of bytes per stored value. Multiplying
every factor: $2 \cdot L \cdot h_{kv} \cdot d_h \cdot T \cdot b \cdot \text{bytes}$.

With $L = 2^5$, $h_{kv} = 2^3$, $d_h = 2^7$, $T = 2^{15}$, $b = 2^0$, and bf16 $= 2^1$ bytes, every factor
is a power of two, and the leading $2$ is $2^1$: adding exponents, $1+5+3+7+15+0+1 = 32$, so the cache is
exactly $2^{32}$ bytes $= 4$ GiB.

MQA sets $h_{kv} = 1$: every query head shares one K/V head, shrinking the cache by a further factor of
$h_{kv}$ relative to this example (to $0.5$ GiB here). GQA sets $h_{kv} = g$ for some $1 < g < h$
query heads, grouped so several query heads share each of $g$ K/V heads — the worked example's
$h_{kv} = 8$ is itself a GQA configuration whenever the model has more than $8$ query heads (for instance
$h = 32$, giving $d = h d_h = 4096$): full multi-head attention, with $h_{kv} = h = 32$, would need
$4\times$ this example's cache ($16$ GiB), while MQA needs $8\times$ less ($0.5$ GiB) — GQA trades off some
of MQA's saving for keeping more than one shared K/V subspace, which tends to lose less quality.

**Q4.** $q_m^\top k_n$ depends on $m, n$ only through $m - n$, because rotating both vectors by angles
linear in position turns the dot product into a function of the angle *difference*, and that difference is
$(m-n)\omega$.

In one 2-D pair, write the rotation matrix

$$R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix},$$

which is orthogonal with $R(\theta)^\top = R(-\theta)$ and $R(\theta_1)R(\theta_2) = R(\theta_1+\theta_2)$
(composing two rotations adds their angles). RoPE rotates
the query at position $m$ by angle $m\omega$ and the key at position $n$ by angle $n\omega$, for a fixed
frequency $\omega$: $q_m = R(m\omega)\,q$, $k_n = R(n\omega)\,k$. Then

$$q_m^\top k_n = \bigl(R(m\omega)q\bigr)^\top\bigl(R(n\omega)k\bigr) = q^\top R(m\omega)^\top R(n\omega)\,k = q^\top R(n\omega - m\omega)\,k = q^\top R\bigl(-(m-n)\omega\bigr)k,$$

using $R(m\omega)^\top = R(-m\omega)$ and then $R(-m\omega)R(n\omega) = R(n\omega - m\omega)$. The
right-hand side is a fixed function of $q$, $k$, $\omega$ and the single scalar $m-n$: any two pairs
$(m,n)$ and $(m+c, n+c)$ give the same angle $-(m-n)\omega$ and so the same value, regardless of $m,n$
individually.

For a full $d$-dimensional query/key, RoPE splits the $d$ coordinates into $d/2$ pairs and applies this
rotation to pair $i$ with its own frequency $\omega_i = \Theta^{-2i/d}$ (a geometric progression, $\Theta$
typically $10^4$–$10^6$), independently of the other pairs. The full dot product is the sum of the $d/2$
pairwise dot products, $q_m^\top k_n = \sum_{i=0}^{d/2-1} q^{(i)\top}R\bigl(-(m-n)\omega_i\bigr)k^{(i)}$,
and every term depends on $m,n$ only through $m-n$ — so the whole sum does too.

Contrast: a learned or sinusoidal absolute position embedding $p_m$ is *added* to the token embedding
before the $Q,K$ projections, so $q_m = W_q(x_m + p_m) = q + W_qp_m$ (writing $q = W_qx_m$), and likewise
$k_n = k + W_kp_n$. Then

$$q_m^\top k_n = q^\top k + q^\top W_kp_n + p_m^\top W_q^\top k + p_m^\top W_q^\top W_kp_n,$$

and three of the four terms involve $p_m$ or $p_n$ directly rather than a difference — nothing in the
parameterisation forces the sum to collapse to a function of $m-n$ alone. Any relative-position behaviour
is at best learned approximately from training data, not guaranteed algebraically, and offers no
particular reason to behave sensibly at sequence lengths longer than seen in training, unlike RoPE's exact
relative-only dependence.

### Architecture and cost

**Q5.** One block has $12d^2$ parameters; the whole model has $12Ld^2 + Vd$; training costs
$\approx 6N$ and inference $\approx 2N$ FLOPs per token, with $N = 12Ld^2$, because every parameter in a
linear layer is used in exactly one multiply-add per token in the forward pass and two more matmuls of the
same size in the backward pass.

Attention has four $d \times d$ matrices ($W_q, W_k, W_v, W_o$, no biases): $4d^2$. The MLP has
$W_1 \in \mathbb{R}^{d \times 4d}$ and $W_2 \in \mathbb{R}^{4d \times d}$: $2 \cdot 4d^2 = 8d^2$. One
block: $4d^2 + 8d^2 = 12d^2$. Stacking $L$ blocks and adding one $V \times d$ embedding table, tied so it
is counted once rather than for both the input lookup and the output unembedding:
$N_{\text{total}} = 12Ld^2 + Vd$.

For the FLOPs, a linear layer $W \in \mathbb{R}^{a \times b}$ applied to one token's activation vector
performs $ab$ multiply-adds $= 2ab$ FLOPs — exactly $2$ FLOPs per parameter of that layer, since every
weight touches exactly one input feature and one output feature for a single token. Summing over every
linear layer inside the $L$ blocks ($N = 12Ld^2$ parameters), the forward pass costs $\approx 2N$ FLOPs
per token. This drops two pieces: the embedding lookup is a free gather, not a matmul, and the attention
score/weighted-sum contractions cost a further $O(Td)$ per token (Q2) — negligible next to the $O(d^2)$
here once $T \ll d$, the regime this approximation targets. Backpropagation through a linear layer computes
two more matmuls of the same size as the forward one: the gradient with respect to the layer's input, and
the gradient with respect to its weights (an outer product of the same shape); so the backward pass costs
about twice the forward pass, $\approx 4N$. Forward plus backward totals $\approx 2N + 4N = 6N$ FLOPs per
training token. Serving one new token with a KV cache — no recomputation of earlier positions — runs only
the forward pass: $\approx 2N$.

**Q6.** Post-norm normalises *after* adding the residual; pre-norm normalises only the branch feeding the
sublayer and leaves the residual addition itself un-normalised. That un-normalised addition is what keeps
pre-norm's gradients from vanishing or exploding with depth. RMSNorm drops LayerNorm's mean-centring and
its additive bias.

Post-norm: $x_{l+1} = \mathrm{Norm}\bigl(x_l + f_l(x_l)\bigr)$. Pre-norm:
$x_{l+1} = x_l + f_l\bigl(\mathrm{Norm}(x_l)\bigr)$, where $f_l$ is the attention or MLP sublayer.

Linearise each sublayer's net effect on the stream by a scalar $a_l$ (a crude but standard simplification
for this argument). In pre-norm, $x_{l+1} = x_l + a_l\,g(x_l)$ for some bounded $g$ — the normalisation
inside the branch never touches $x_l$ itself — so $\partial x_{l+1}/\partial x_l = 1 + O(a_l)$, and by the
chain rule $\partial x_L/\partial x_0 = \prod_{l=1}^{L}\bigl(1 + O(a_l)\bigr)$: an *additive* identity term
of exactly $1$ survives at every layer, so this product cannot collapse toward $0$ the way a product of
factors all below $1$ would. In post-norm, the residual sum itself is renormalised at every layer,
$x_{l+1} = \mathrm{Norm}\bigl(x_l + f_l(x_l)\bigr) \approx a_lx_l$ near a linearisation point — the
identity path is folded into the same multiplicative factor as the branch, since normalisation rescales
the whole sum together — so $\partial x_L/\partial x_0 \approx \prod_{l=1}^{L} a_l$: a plain product of
$L$ terms, which vanishes or explodes exponentially in $L$ unless every $a_l$ sits extremely close to $1$.
This is exactly why deep post-norm transformers historically needed careful learning-rate warm-up to
survive early training, while pre-norm trains stably out of the box at much greater depth.

$$\mathrm{LN}(v)_i = \frac{v_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma_i + \beta_i, \quad \mu = \tfrac1d\textstyle\sum_i v_i,\ \sigma^2 = \tfrac1d\textstyle\sum_i (v_i-\mu)^2; \qquad \mathrm{RMSNorm}(v)_i = \frac{v_i}{\mathrm{RMS}(v)}\,\gamma_i,\quad \mathrm{RMS}(v)=\sqrt{\tfrac1d\textstyle\sum_i v_i^2 + \epsilon}.$$

RMSNorm drops the mean subtraction $\mu$ (no re-centring, only rescaling) and drops the additive bias
$\beta$, keeping only the learned per-feature scale $\gamma$. When $v$ already has zero mean,
$\sigma^2 = \frac1d\sum_iv_i^2$ coincides with $\mathrm{RMS}(v)^2$ (up to $\epsilon$), so the two rescale by
the same factor and differ only in whether $\mu$ was subtracted first — RMSNorm simply skips that
reduction, one fewer pass over the $d$ features per call, and drops $\beta$'s $d$ extra parameters.

### Objective and data

**Q7.** Cross-entropy $\approx 0.978$ nats, perplexity $= \sqrt[4]{50} \approx 2.659$.

For a sequence of $T$ tokens with the model's probability $p_t = \pi_\theta(y_t \mid y_{<t})$ assigned to
the true next token at each step, cross-entropy (per token, in nats) is
$\mathcal{H} = -\frac1T\sum_{t=1}^T \ln p_t$, and perplexity is $\mathrm{PPL} = e^{\mathcal{H}}$ — the
model is, on average, as uncertain as if choosing uniformly among $\mathrm{PPL}$ options at each step,
since a uniform choice among $k$ options has $-\ln(1/k) = \ln k$ nats of surprise, matching
$\mathcal{H} = \ln k$ exactly when $\mathrm{PPL}=k$. In base 2, $\mathrm{PPL} = 2^{\mathcal{H}/\ln 2}$, so
$\mathcal{H}/\ln 2$ is the bits-per-token figure.

With $p_1,\dots,p_4 = 0.5, 0.2, 0.8, 0.25$: $\prod_t p_t = 0.5 \times 0.2 \times 0.8 \times 0.25 = 0.02 =
1/50$, so $\mathrm{PPL} = \bigl(\prod_t p_t\bigr)^{-1/T} = 50^{1/4} = \sqrt[4]{50} \approx 2.659$, and
$\mathcal{H} = \ln(\mathrm{PPL}) = \tfrac14\ln 50 \approx 0.978$ nats $\approx 1.411$ bits per token.

**Q8.** The first three merges are $(e,s)$, then $(es,t)$, then $(est,\_)$, every one with count $9$.

Split into symbols: `low` → `l o w _` (×5); `lower` → `l o w e r _` (×2); `newest` → `n e w e s t _` (×6);
`widest` → `w i d e s t _` (×3). Summing adjacent-pair counts, weighted by word frequency, across all four
words: $(l,o){=}7$, $(o,w){=}7$, $(w,\_){=}5$, $(w,e){=}8$, $(e,r){=}2$, $(r,\_){=}2$, $(n,e){=}6$,
$(e,w){=}6$, $(e,s){=}9$, $(s,t){=}9$, $(t,\_){=}9$, $(w,i){=}3$, $(i,d){=}3$, $(d,e){=}3$ — a three-way
tie for the maximum, at $9$, between $(e,s)$, $(s,t)$ and $(t,\_)$ (both `newest` and `widest` share the
exact suffix `est_`, so every pair inside it gets the same combined count, $6 + 3$). Breaking the tie
lexicographically picks $(e,s)$ first. Merging it turns `newest` into `n e w es t _` and `widest` into
`w i d es t _`; recounting, $(es,t)$ and $(t,\_)$ are now tied at $9$, and $(es,t)$ sorts first
lexicographically (`es` < `t`), so it merges second, giving `n e w est _` / `w i d est _`. Recounting once
more, only $(est,\_) = 9$ remains at the top, unique this time, and merges third: `n e w est_` /
`w i d est_`. `low` and `lower` never contain the letters that these three pairs need together and stay
untouched.

More merges shrink the vocabulary's redundancy — common multi-character chunks (`est_`, eventually whole
short words) become single tokens — which shortens the token sequence for a given piece of text, and a
shorter $T$ is cheaper for a transformer with attention cost quadratic in $T$ (Q2). Pushed far enough,
though, the vocabulary itself grows (more distinct merged symbols to embed and to produce logits over,
adding to $Vd$ in Q5), and rarer merged tokens are each seen less often during training, so their
embeddings are learned from less data; too few merges leaves sequences long and expensive, too many leaves
the tail of the vocabulary poorly trained and prevents graceful fallback to characters for
out-of-vocabulary spellings.

**Q9.** $N \approx 2.9 \times 10^{10}$ parameters, $D \approx 5.8 \times 10^{11}$ tokens.

Substituting the rule of thumb $D \approx 20N$ into $C \approx 6ND$: $C \approx 6N(20N) = 120N^2$, so

$$N \approx \sqrt{\frac{C}{120}}, \qquad D \approx 20N.$$

For $C = 10^{23}$: $N \approx \sqrt{10^{23}/120} \approx 2.9\times10^{10}$, and
$D \approx 20 \times 2.9\times10^{10} \approx 5.8\times10^{11}$.

This $N$ minimises training loss for a *fixed training-compute* budget alone; it says nothing about
inference cost, which scales roughly with $N$ per generated token (Q5) and is paid every time the model is
served — often for vastly more tokens in total, over a deployed model's lifetime, than it was trained on.
If total cost (training plus serving) is what matters, the optimum shifts toward a *smaller* $N$ trained on
*more* tokens than $D \approx 20N$: spending extra training compute (moving off the training-compute-optimal
frontier deliberately) buys a smaller, cheaper-to-serve model pushed to a lower loss than the
training-optimal ratio alone would reach at that size.

### Post-training and inference

**Q10.** Pretraining maximises next-token likelihood on raw text; SFT maximises the same next-token
likelihood but only over demonstrated response tokens, given a prompt; preference optimisation maximises
expected reward (human preference, or a learned reward model) subject to staying close, in KL divergence,
to the reference policy — and DPO is exactly the closed-form solution of that same KL-regularised
objective, needing no separate reward model or RL rollout.

Pretraining: $\mathcal{L}_{\text{pre}} = -\frac1T\sum_t \ln \pi_\theta(x_t \mid x_{<t})$ over raw text $x$.
SFT, given prompt $x$ and demonstrated response $y$: $\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|}\sum_{t \in
y} \ln \pi_\theta(y_t \mid x, y_{<t})$ — the loss on the prompt tokens is masked to exactly $0$ (in
practice, by giving them an ignored label). The prompt is fixed context the model must condition on, not
something it should be trained to *generate*: prompts vary enormously in style and are often written by a
different source (users, templates) than the desired response style, so training on them would spend
gradient signal imitating prompt phrasing rather than improving response quality, with no well-defined
"correct" way to have produced a prompt handed to the model verbatim.

For DPO, the KL-regularised objective is

$$\max_\theta\ \mathbb{E}_{x\sim\mathcal D,\ y\sim\pi_\theta(\cdot|x)}\bigl[r(x,y)\bigr] - \beta\,\mathrm{KL}\bigl(\pi_\theta(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x)\bigr).$$

For fixed $x$, maximising $\sum_y \pi(y)r(y) - \beta\sum_y\pi(y)\ln\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}$
over distributions $\pi(\cdot|x)$ subject to $\sum_y \pi(y) = 1$ has the closed-form optimum

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\,\exp\!\Bigl(\frac{r(x,y)}{\beta}\Bigr), \qquad Z(x) = \sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\Bigl(\frac{r(x,y)}{\beta}\Bigr).$$

Solving for the reward:

$$r(x,y) = \beta\ln\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\ln Z(x).$$

Substituting into the Bradley–Terry preference model $P(y_w \succ y_l \mid x) = \sigma\bigl(r(x,y_w) -
r(x,y_l)\bigr)$, the $\beta\ln Z(x)$ term is common to $y_w$ and $y_l$ (it depends only on $x$) and cancels
in the difference:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta\ln\frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\ln\frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).$$

Fitting $\pi_\theta$ by maximum likelihood to observed preference pairs under this model is exactly the DPO
loss, $\mathcal{L}_{\mathrm{DPO}} = -\ln\sigma\Bigl(\beta\ln\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
- \beta\ln\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Bigr)$: no reward model or RL rollout is
needed, because the closed form relating $\pi^*$ and $r$ substitutes the reward straight into a supervised
classification loss on $\pi_\theta$ itself.

Either route keeps the KL term because the reward signal (a learned reward model, or the implicit reward
inside DPO's preference data) is only accurate near the data distribution it was trained or collected on: a
policy free to drift arbitrarily far from $\pi_{\mathrm{ref}}$ can find outputs that score well under an
imperfect reward without being good ("reward hacking"), and drifting far from $\pi_{\mathrm{ref}}$ also
risks discarding language capability learned in pretraining/SFT. The closed form above makes this explicit:
as $\beta \to \infty$, $\pi^* \to \pi_{\mathrm{ref}}$ regardless of $r$; as $\beta \to 0$, $\pi^*$ collapses
onto the reward-maximising output(s) with no regularisation left at all.

**Q11.** Write $z$ for the next-token logits and $p = \mathrm{softmax}(z)$.

*Greedy decoding* picks $\arg\max_i p_i$ every step: deterministic, zero-entropy, the $\tau \to 0^+$ limit
of temperature sampling. *Temperature sampling* rescales the logits before softmax,
$p_i(\tau) = e^{z_i/\tau} / \sum_je^{z_j/\tau}$: $\tau < 1$ sharpens the distribution toward the arg-max,
$\tau > 1$ flattens it toward uniform, $\tau = 1$ leaves it unchanged — every token stays reachable at any
finite $\tau$, only re-weighted. *Top-$k$ sampling* keeps only the $k$ tokens with the highest probability,
renormalises them to sum to $1$, and samples from that truncated distribution (zero elsewhere): it fixes
the *size* of the kept set regardless of how the mass is distributed. *Nucleus (top-$p$) sampling* sorts
tokens by probability descending and keeps the shortest prefix whose cumulative probability is $\ge p$,
renormalising and sampling from it: it fixes the kept *probability mass* instead, so the kept set's size
adapts — small when the distribution is peaked (the model is confident), large when it is flat.

Worked example, already sorted descending — `a:0.5, b:0.2, c:0.15, d:0.1, e:0.05` — with $p=0.7$:
cumulative mass after `a` is $0.5 < 0.7$; after `a, b` it is exactly $0.7 \ge 0.7$, so the kept set is
$\{a, b\}$.

**Q12.** Total parameters $= 1.65 \times 10^{10}$; active parameters per token $= 4.5 \times 10^{9}$
($\approx 27\%$ of the total).

Total parameters are the shared parameters plus every expert's, whether or not it is used for a given
token: $\text{shared} + E \cdot P_{\text{expert}}$. Active parameters per token are the shared parameters
plus only the $k$ experts that token's router actually selects: $\text{shared} + k \cdot P_{\text{expert}}$
(the router itself adds a comparatively tiny number of parameters, usually not counted separately). For
$E=8$, $P_{\text{expert}} = 2\times10^9$, shared $=5\times10^8$, $k=2$:
$\text{total} = 5\times10^8 + 8 \times 2\times10^9 = 1.65\times10^{10}$, and
$\text{active} = 5\times10^8 + 2 \times 2\times10^9 = 4.5\times10^9$ — the model has the per-token inference
compute of a $4.5$B-parameter dense model but the capacity, and memory footprint, of a $16.5$B-parameter
one.

The router is trained purely to minimise task loss, which has no inherent pressure to spread tokens evenly
across experts — left alone, it can (and empirically does) collapse onto routing most tokens to a small
favoured subset, especially early in training. Neglected experts then receive few gradient updates and
stay undertrained, which makes the router favour them even less (a rich-get-richer loop), and because each
expert has a fixed compute/memory budget per step in an efficient parallel implementation, uneven load
either wastes capacity (idle experts) or forces dropped tokens (overloaded ones) — hurting both quality and
hardware utilisation. The auxiliary loss adds an explicit penalty, typically
$E \cdot \sum_e f_e P_e$ (experts times the dot product of the actual routed-token fraction $f_e$ with the
router's average softmax probability $P_e$ for each expert), minimised exactly when routing is uniform,
pushing the router away from collapse without supervising which expert handles which token.

The *capacity factor* $c$ (typically $1$–$2$) sets each expert's maximum tokens per batch to
$\mathrm{capacity} = c \cdot N_{\text{tokens}}/E$ — headroom above the $N_{\text{tokens}}/E$ that perfectly
even routing would send, since routing is never perfectly even in practice, while still bounding the worst
case. Tokens routed to an expert already at capacity are dropped (skipped, e.g. passed through via a
residual connection) or sent to a lower-ranked expert instead, keeping every expert's compute and memory
bounded and identical across devices in expert-parallel training, at the cost of some tokens missing their
preferred expert(s) when $c$ is small.

### Follow-ups

- **Long-context extrapolation.** RoPE's relative-position property (Q4) holds algebraically at any
  position, but the *attention patterns* the trained weights actually respond to don't extrapolate for
  free: evaluated well beyond the training length, rotation angles the model never saw during training
  degrade quality unless the frequency schedule is rescaled (for example by increasing $\Theta$ or
  interpolating positions) to keep the angles seen at inference inside the range seen in training.
- **Why split into heads at all.** Splitting $d$ into $h$ heads of width $d_h$ costs the same FLOPs and the
  same parameters as one head of width $d$ (Q2, Q5), but lets each head specialise in a different pattern
  over the sequence and normalise it with its own softmax — a saturated pattern in one head does not wash
  out a sharp one elsewhere, which a single softmax over the full $d$-wide score would.
- **Speculative decoding.** A small, fast draft model proposes several tokens; the large target model then
  verifies all of them in one forward pass, since checking a fixed sequence needs only one pass while
  generating it token by token needs one pass per token. Accepted tokens are free, and generation resumes
  from the target model's own distribution at the first rejected one — turning several sequential
  large-model steps into close to one.
- **Untying the embeddings.** Tying saves $Vd$ parameters (Q5) and links the geometry used to read a token
  with the geometry used to write one, but forecloses giving the two different capacity; once $Vd$ is a
  small fraction of $N$, untying costs relatively little extra capacity and is sometimes preferred for
  that reason.

<details>
<summary>Checks (runnable)</summary>

```python
import math
from collections import Counter

import numpy as np
from scipy import stats

rng = np.random.default_rng(2026)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ============================================================
# Q1 -- Var(q . k) = d_h, and what 1/sqrt(d_h) scaling does to softmax
# ============================================================
for d_h in (8, 16, 64, 256):
    q = rng.standard_normal((200_000, d_h))
    k = rng.standard_normal((200_000, d_h))
    dot = np.sum(q * k, axis=1)
    assert abs(dot.var() - d_h) < 0.05 * d_h                          # Var(q . k) ~= d_h, unscaled
    assert abs((dot / math.sqrt(d_h)).var() - 1.0) < 0.05              # scaled variance ~= 1, for every d_h

raw = rng.standard_normal(6)                                           # NOTE: larger pre-softmax spread saturates
entropies = [stats.entropy(softmax(raw * s)) for s in (0.25, 1.0, 4.0, 16.0)]
assert all(a > b for a, b in zip(entropies, entropies[1:]))            # softmax entropy strictly falls as spread grows
grad_mag = [np.sum(softmax(raw * s) * (1 - softmax(raw * s))) for s in (0.25, 1.0, 4.0, 16.0)]
assert all(a > b for a, b in zip(grad_mag, grad_mag[1:]))               # and the softmax Jacobian's scale shrinks too

# ============================================================
# Q2 -- attention FLOP and memory cost, independent vs. dependent on the head split
# ============================================================
def count_matmul_flops(a_shape, b_shape):
    """2 FLOPs (multiply + add) per scalar multiply-accumulate of an (..., m, k) @ (..., k, n) contraction."""
    *_, m, k = a_shape
    *_, k2, n = b_shape
    assert k == k2
    return 2 * m * k * n


def naive_causal_attention(X, Wq, Wk, Wv, Wo, n_heads, flop_counter=None):
    """One layer of causal multi-head self-attention, straight from the formula. If flop_counter is a list,
    every matmul performed appends its FLOPs (an independent, shape-based FLOP count)."""
    T, d = X.shape
    d_head = d // n_heads

    def mm(a, b):
        if flop_counter is not None:
            flop_counter.append(count_matmul_flops(a.shape, b.shape))
        return a @ b

    Q, K, V = mm(X, Wq), mm(X, Wk), mm(X, Wv)
    Qh = Q.reshape(T, n_heads, d_head).transpose(1, 0, 2)               # (h, T, d_head)
    Kh = K.reshape(T, n_heads, d_head).transpose(1, 0, 2)
    Vh = V.reshape(T, n_heads, d_head).transpose(1, 0, 2)

    visible = np.tril(np.ones((T, T), dtype=bool))
    out_heads = np.zeros((n_heads, T, d_head))
    score_matrix_elements = 0                                          # explicit element count of the materialised
    for h in range(n_heads):                                           # T x T weight matrix, one per head
        scores = mm(Qh[h], Kh[h].T) / math.sqrt(d_head)
        scores = np.where(visible, scores, -np.inf)
        m = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - m)
        weights /= weights.sum(axis=-1, keepdims=True)
        score_matrix_elements += weights.size
        out_heads[h] = mm(weights, Vh[h])
    merged = out_heads.transpose(1, 0, 2).reshape(T, d)
    out = mm(merged, Wo)
    return out, score_matrix_elements


T, d = 24, 16
X = rng.normal(size=(T, d))
for n_heads in (1, 2, 4, 8):                                            # d = 16 is divisible by each of these
    Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.3 for _ in range(4))
    flops = []
    _, score_elements = naive_causal_attention(X, Wq, Wk, Wv, Wo, n_heads, flop_counter=flops)
    closed_form_flops = 8 * T * d ** 2 + 4 * T ** 2 * d
    assert sum(flops) == closed_form_flops                              # FLOPs: exactly independent of the head split
    assert score_elements == n_heads * T ** 2                           # memory: grows linearly with the head count
_, mem_1 = naive_causal_attention(X, *(rng.normal(size=(d, d)) * 0.3 for _ in range(4)), 1)
_, mem_8 = naive_causal_attention(X, *(rng.normal(size=(d, d)) * 0.3 for _ in range(4)), 8)
assert mem_8 == 8 * mem_1                                                # unlike FLOPs, memory scales with h


def naive_head_attention(Q, K, V):
    """Reference single-head causal attention, no tiling: the full T x T matrix is built at once."""
    T, d_head = Q.shape
    scores = (Q @ K.T) / math.sqrt(d_head)
    visible = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(visible, scores, -np.inf)
    m = scores.max(axis=-1, keepdims=True)
    w = np.exp(scores - m)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ V


def flash_attention_head(Q, K, V, block_q=8, block_k=8, track_block_sizes=None):
    """FlashAttention-style tiling: an online (running) softmax, never materialising the full T x T matrix."""
    T, d_head = Q.shape
    out = np.zeros((T, d_head))
    for qs in range(0, T, block_q):
        qe = min(qs + block_q, T)
        q_blk = Q[qs:qe]
        m = np.full((qe - qs, 1), -np.inf)
        l = np.zeros((qe - qs, 1))
        acc = np.zeros((qe - qs, d_head))
        for ks in range(0, qe, block_k):                                # causal: keys beyond qe are never visible
            ke = min(ks + block_k, qe)
            k_blk, v_blk = K[ks:ke], V[ks:ke]
            s = (q_blk @ k_blk.T) / math.sqrt(d_head)
            if track_block_sizes is not None:
                track_block_sizes.append(s.size)
            causal = np.arange(qs, qe)[:, None] >= np.arange(ks, ke)[None, :]
            s = np.where(causal, s, -np.inf)
            m_new = np.maximum(m, s.max(axis=1, keepdims=True))
            correction = np.exp(m - m_new)                              # NOTE: rescale running stats to the new max,
            l = l * correction + np.exp(s - m_new).sum(axis=1, keepdims=True)   # rather than recomputing from scratch
            acc = acc * correction + np.exp(s - m_new) @ v_blk
            m = m_new
        out[qs:qe] = acc / l
    return out


for T_try, d_head_try in ((10, 6), (37, 8), (63, 5)):
    Qh, Kh, Vh = (rng.normal(size=(T_try, d_head_try)) for _ in range(3))
    ref = naive_head_attention(Qh, Kh, Vh)
    tiled = flash_attention_head(Qh, Kh, Vh, block_q=4, block_k=4)
    assert np.allclose(ref, tiled, atol=1e-10)                          # exact, not approximate: same softmax, tiled

block_sizes = []
Qh, Kh, Vh = (rng.normal(size=(256, 8)) for _ in range(3))
flash_attention_head(Qh, Kh, Vh, block_q=8, block_k=8, track_block_sizes=block_sizes)
assert max(block_sizes) == 8 * 8                                        # flash: bounded by block_q * block_k
naive_matrix_size_at_256 = 256 ** 2
naive_matrix_size_at_16 = 16 ** 2
assert (naive_matrix_size_at_256 / (8 * 8)) > 100 * (naive_matrix_size_at_16 / (8 * 8))  # the gap widens like T^2

# ============================================================
# Q3 -- KV-cache size, and the MQA / GQA reductions
# ============================================================
def kv_cache_bytes(L, h_kv, d_head, T, b, bytes_per_element):
    return 2 * L * h_kv * d_head * T * b * bytes_per_element


L, h_kv, d_head, T_ctx, b, bf16_bytes = 32, 8, 128, 32_768, 1, 2
cache = kv_cache_bytes(L, h_kv, d_head, T_ctx, b, bf16_bytes)
assert cache == 2 ** 32 == 4 * 1024 ** 3                                # exactly 4 GiB

full_mha_heads = 32                                                     # a plausible query-head count for d_head=128
mha_cache = kv_cache_bytes(L, full_mha_heads, d_head, T_ctx, b, bf16_bytes)
mqa_cache = kv_cache_bytes(L, 1, d_head, T_ctx, b, bf16_bytes)
assert mha_cache == cache * (full_mha_heads // h_kv)                    # h_kv = 8 is already a 4x GQA reduction from 32
assert mqa_cache == cache // h_kv                                       # MQA shrinks the h_kv = 8 example 8x further
assert mha_cache == mqa_cache * full_mha_heads                          # and is 32x the MQA cache

# ============================================================
# Q4 -- RoPE: q_m^T k_n depends only on m - n
# ============================================================
def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


rope_rng = np.random.default_rng(7)
omega = 0.7
q2, k2 = rope_rng.normal(size=2), rope_rng.normal(size=2)
diffs = [float((rot2d(m * omega) @ q2) @ (rot2d(n * omega) @ k2)) for m, n in [(0, -2), (5, 3), (50, 48), (1000, 998)]]
assert max(diffs) - min(diffs) < 1e-8                                    # depends only on m - n = 2, not m, n themselves
assert abs(diffs[0] - float(q2 @ (rot2d(-2 * omega) @ k2))) < 1e-8        # matches q^T R(-(m-n) omega) k exactly


def rope_rotate(x, position, freqs):
    """x: (..., d) with d = 2 * len(freqs). Rotates each 2-D coordinate pair i by angle position * freqs[i]."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    angles = position * np.asarray(freqs)
    cos, sin = np.cos(angles), np.sin(angles)
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


d_model, n_pairs = 12, 6
freqs = 10000.0 ** (-2 * np.arange(n_pairs) / d_model)                   # the usual RoPE frequency schedule
for _ in range(30):
    q, k = rope_rng.normal(size=d_model), rope_rng.normal(size=d_model)
    delta = int(rope_rng.integers(-50, 50))
    base_m = int(rope_rng.integers(0, 500))
    dots = []
    for shift in (0, 17, 250):                                          # several (m, n) pairs sharing m - n = delta
        m, n = base_m + shift, base_m + shift - delta
        dots.append(float(rope_rotate(q, m, freqs) @ rope_rotate(k, n, freqs)))
    assert max(dots) - min(dots) < 1e-8 * (1 + abs(max(dots)))

q, k = rope_rng.normal(size=d_model), rope_rng.normal(size=d_model)      # a different delta really gives a different
distinct = [float(rope_rotate(q, 300, freqs) @ rope_rotate(k, 300 - delta, freqs)) for delta in (0, 5, -5, 20)]
assert len({round(v, 6) for v in distinct}) == len(distinct)             # value -- the invariance isn't a degenerate constant

# ============================================================
# Q5 -- parameter count, and the 2N / 6N FLOPs-per-token approximations
# ============================================================
def block_shapes(d):
    return [(d, d), (d, d), (d, d), (d, d), (d, 4 * d), (4 * d, d)]      # Wq, Wk, Wv, Wo, W1, W2


for d_try, L_try, V_try in ((8, 3, 37), (16, 5, 101), (32, 2, 61)):
    shapes = block_shapes(d_try) * L_try
    explicit_block_params = sum(a * b for a, b in shapes)                # explicit sum of every matrix's size
    formula_block_params = 12 * L_try * d_try ** 2
    assert explicit_block_params == formula_block_params

    total_explicit = explicit_block_params + V_try * d_try               # tied embedding: the V x d table once
    total_formula = formula_block_params + V_try * d_try
    assert total_explicit == total_formula

    N = formula_block_params
    forward_flops = sum(count_matmul_flops((1, a), (a, b)) for a, b in shapes)     # one token through every linear
    assert forward_flops == 2 * N                                                   # layer, excluding embedding/attention

    backward_flops = 0
    for a, b in shapes:
        backward_flops += count_matmul_flops((1, b), (b, a))             # d(loss)/d(input), same shape as forward
        backward_flops += count_matmul_flops((a, 1), (1, b))             # d(loss)/d(W), an outer product
    assert backward_flops == 4 * N
    assert forward_flops + backward_flops == 6 * N

# ============================================================
# Q6 -- pre-norm vs. post-norm depth scaling (toy linear model); RMSNorm vs. LayerNorm
# ============================================================
def toy_residual_stack(depth, style, a):
    """The exact chain rule of the toy 1-D linear model in the text: an identity term survives every layer
    in pre-norm, but not in post-norm."""
    grad = 1.0
    for _ in range(depth):
        grad *= (1.0 + a) if style == "pre" else a
    return grad


for a in (0.7, 0.9, 1.1):
    for depth in (10, 40, 100):
        assert math.isclose(toy_residual_stack(depth, "pre", a), (1.0 + a) ** depth, rel_tol=1e-9)
        assert math.isclose(toy_residual_stack(depth, "post", a), a ** depth, rel_tol=1e-9)
assert toy_residual_stack(200, "post", 0.9) < 1e-8                       # post-norm: 0.9^200 vanishes
assert toy_residual_stack(200, "pre", 0.9) > 1e30                        # pre-norm: (1.9)^200 grows instead


def layer_norm_np(v, gamma, beta, eps=1e-5):
    mu = v.mean()
    var = ((v - mu) ** 2).mean()
    return (v - mu) / math.sqrt(var + eps) * gamma + beta


def rms_norm_np(v, gamma, eps=1e-5):
    rms = math.sqrt((v ** 2).mean() + eps)
    return v / rms * gamma


d_dim = 10
gamma, beta_zero = rng.normal(size=d_dim), np.zeros(d_dim)
v = rng.normal(size=d_dim)
v_centered = v - v.mean()
assert np.allclose(layer_norm_np(v_centered, gamma, beta_zero), rms_norm_np(v_centered, gamma), atol=1e-6)
assert not np.allclose(layer_norm_np(v, gamma, beta_zero), rms_norm_np(v, gamma), atol=1e-3)   # differ when not centred

# ============================================================
# Q7 -- cross-entropy and perplexity from given next-token probabilities
# ============================================================
probs = [0.5, 0.2, 0.8, 0.25]
cross_entropy = sum(-math.log(p) for p in probs) / len(probs)
perplexity = math.exp(cross_entropy)
assert math.isclose(math.prod(probs), 0.02, rel_tol=1e-9)
assert math.isclose(perplexity, 50 ** 0.25, rel_tol=1e-9)                # closed form: PPL = (prod 1/p_t)^(1/T)
assert math.isclose(cross_entropy, math.log(50) / 4, rel_tol=1e-12)
bits_per_token = cross_entropy / math.log(2)
assert math.isclose(bits_per_token, math.log2(perplexity), rel_tol=1e-9)

# ============================================================
# Q8 -- BPE: the first three merges, by an independent brute-force pair counter
# ============================================================
END = "_"
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
symbol_seqs = {word: list(word) + [END] for word in corpus}


def count_pairs(seqs, freqs):
    """Independent brute-force pair counter: does not call apply_merge or reuse any merge-loop state."""
    counts = Counter()
    for word, seq in seqs.items():
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += freqs[word]
    return counts


def apply_merge(seqs, pair):
    merged = "".join(pair)
    out = {}
    for word, seq in seqs.items():
        new_seq, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        out[word] = new_seq
    return out


def symbol_rank(sym):
    return (1, "") if sym == END else (0, sym)


def best_pair(counts):
    """Most frequent pair; ties broken by the lexicographically smaller pair (letters a-z, then '_')."""
    best_count = max(counts.values())
    candidates = [pr for pr, c in counts.items() if c == best_count]
    return min(candidates, key=lambda pr: (symbol_rank(pr[0]), symbol_rank(pr[1])))


initial_counts = count_pairs(symbol_seqs, corpus)
assert sum(initial_counts.values()) == 15 + 10 + 36 + 18 == 79           # (len(word)+1-1) pairs, weighted by frequency
assert (initial_counts[("e", "s")], initial_counts[("s", "t")], initial_counts[("t", "_")]) == (9, 9, 9)
assert (initial_counts[("l", "o")], initial_counts[("w", "e")]) == (7, 8)

seqs, merges = symbol_seqs, []
for _ in range(3):
    counts = count_pairs(seqs, corpus)
    pair = best_pair(counts)
    merges.append((pair, counts[pair]))
    seqs = apply_merge(seqs, pair)

assert merges == [(("e", "s"), 9), (("es", "t"), 9), (("est", "_"), 9)]
assert seqs["newest"] == ["n", "e", "w", "est_"]
assert seqs["widest"] == ["w", "i", "d", "est_"]
assert seqs["low"] == ["l", "o", "w", "_"]                                # untouched: no e/s/t/_ run to merge
assert seqs["lower"] == ["l", "o", "w", "e", "r", "_"]

# ============================================================
# Q9 -- compute-optimal scaling: C = 6ND, D = 20N
# ============================================================
C = 1e23
N = math.sqrt(C / 120)
D = 20 * N
assert math.isclose(N, 2.9e10, rel_tol=0.01)
assert math.isclose(D, 5.8e11, rel_tol=0.01)
assert math.isclose(6 * N * D, C, rel_tol=1e-9)                           # recovers the compute budget exactly
assert math.isclose(D / N, 20.0, rel_tol=1e-9)

# ============================================================
# Q10 -- masked SFT loss, and the DPO reward substitution
# ============================================================
prompt_logp = [-0.2, -0.5, -0.1]
response_logp = [-0.3, -0.05, -0.9, -0.2]


def sft_loss(response_logp):
    return -sum(response_logp) / len(response_logp)                       # NOTE: prompt tokens contribute 0, not their own logp


masked = sft_loss(response_logp)
all_tokens_loss = -sum(prompt_logp + response_logp) / (len(prompt_logp) + len(response_logp))
assert masked != all_tokens_loss
assert math.isclose(masked, 0.3625, rel_tol=1e-9)
mask = [0, 0, 0, 1, 1, 1, 1]                                                # an equivalent view: per-token weights
weighted = -sum(w * lp for w, lp in zip(mask, prompt_logp + response_logp)) / sum(mask)
assert math.isclose(weighted, masked, rel_tol=1e-12)

dpo_rng = np.random.default_rng(11)
for _ in range(200):
    n_outcomes = int(dpo_rng.integers(3, 8))
    logits_ref = dpo_rng.normal(size=n_outcomes)
    pi_ref = np.exp(logits_ref) / np.exp(logits_ref).sum()
    r = dpo_rng.normal(size=n_outcomes) * 2.0                              # a ground-truth reward
    beta = float(dpo_rng.uniform(0.2, 3.0))

    unnorm = pi_ref * np.exp(r / beta)
    Z = unnorm.sum()
    pi_star = unnorm / Z                                                   # the closed-form KL-regularised optimum

    recovered_r = beta * np.log(pi_star / pi_ref) + beta * np.log(Z)       # invert the closed form for r
    assert np.allclose(recovered_r, r, atol=1e-8)                          # exact recovery, every outcome

    w, l = int(dpo_rng.integers(0, n_outcomes)), int(dpo_rng.integers(0, n_outcomes))
    if w == l:
        continue
    true_pref = 1.0 / (1.0 + math.exp(-(r[w] - r[l])))                     # Bradley-Terry with the true reward
    dpo_logits = beta * math.log(pi_star[w] / pi_ref[w]) - beta * math.log(pi_star[l] / pi_ref[l])
    dpo_pref = 1.0 / (1.0 + math.exp(-dpo_logits))                          # the same probability from pi*/pi_ref alone
    assert math.isclose(true_pref, dpo_pref, rel_tol=1e-6)                  # log Z(x) cancelled, no reward model needed

# ============================================================
# Q11 -- decoding: greedy, temperature, top-k, top-p (nucleus)
# ============================================================
def softmax_from_logits(z, tau=1.0):
    z = np.asarray(z, dtype=float) / tau
    e = np.exp(z - z.max())
    return e / e.sum()


logits = rng.normal(size=8)
assert np.argmax(softmax_from_logits(logits, tau=1e-6)) == np.argmax(logits)       # tau -> 0 recovers greedy
ent = [stats.entropy(softmax_from_logits(logits, tau=t)) for t in (0.3, 1.0, 3.0)]
assert ent[0] < ent[1] < ent[2]                                                     # lower tau -> sharper -> lower entropy


def rank_order(p):
    """Descending by probability; ties broken by ascending original index (the stated tie rule)."""
    return sorted(range(len(p)), key=lambda i: (-p[i], i))


def top_k_filter(probs, k):
    idx = rank_order(probs)[:k]
    out = np.zeros_like(probs)
    out[idx] = probs[idx]
    return out / out.sum()


def top_p_filter(probs, p):
    order = rank_order(probs)
    cum = np.cumsum(probs[order])
    # NOTE: side="left" finds the first index whose cumulative mass already reaches p (rather than the last
    #       index still short of it), so +1 turns that 0-based index into the correct prefix length.
    cutoff = int(np.searchsorted(cum, p, side="left")) + 1
    kept = order[:cutoff]
    out = np.zeros_like(probs)
    out[kept] = probs[kept]
    return out / out.sum()


def brute_force_top_p(probs, p):
    """Independent definition: walk the sorted list and stop at the first prefix reaching p."""
    order, total, kept = rank_order(probs), 0.0, []
    for i in order:
        kept.append(i)
        total += probs[i]
        if total >= p - 1e-12:
            break
    out = np.zeros_like(probs)
    out[kept] = probs[kept]
    return out / out.sum()


names = ["a", "b", "c", "d", "e"]
example_probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
kept_mask = top_p_filter(example_probs, 0.7) > 0
assert set(np.array(names)[kept_mask]) == {"a", "b"}
assert math.isclose(example_probs[:2].sum(), 0.7, rel_tol=1e-9)
assert set(np.array(names)[top_k_filter(example_probs, 2) > 0]) == {"a", "b"}
assert set(np.array(names)[top_k_filter(example_probs, 3) > 0]) == {"a", "b", "c"}

tied_probs = np.array([0.3, 0.3, 0.2, 0.2])                                  # exact ties: (0, 1) and (2, 3)
assert rank_order(tied_probs) == [0, 1, 2, 3]                                # smaller original index wins a tie
assert list(np.nonzero(top_p_filter(tied_probs, 0.5) > 0)[0]) == [0, 1]      # reaches p = 0.5 exactly at index 1
assert list(np.nonzero(top_k_filter(tied_probs, 3) > 0)[0]) == [0, 1, 2]     # ties broken the same way for top-k

for _ in range(500):
    n = int(rng.integers(2, 12))
    raw_p = rng.exponential(size=n) + 1e-6
    probs_r = raw_p / raw_p.sum()
    p_r = float(rng.uniform(0.05, 0.999))
    assert np.allclose(top_p_filter(probs_r, p_r), brute_force_top_p(probs_r, p_r))

for _ in range(200):
    n = int(rng.integers(2, 12))
    probs_r = rng.dirichlet(np.ones(n))
    k = int(rng.integers(1, n + 1))
    filtered = top_k_filter(probs_r, k)
    kept, discarded = filtered > 0, filtered == 0
    assert np.count_nonzero(kept) == k
    assert math.isclose(filtered.sum(), 1.0, rel_tol=1e-9)
    # independent, definition-level property of top-k: every kept probability is >= every discarded one
    assert discarded.sum() == 0 or probs_r[kept].min() >= probs_r[discarded].max()

# ============================================================
# Q12 -- MoE: total vs. active parameters
# ============================================================
def moe_params(n_experts, params_per_expert, shared_params, k):
    return shared_params + n_experts * params_per_expert, shared_params + k * params_per_expert


total, active = moe_params(n_experts=8, params_per_expert=2e9, shared_params=5e8, k=2)
assert total == 1.65e10
assert active == 4.5e9
assert math.isclose(active / total, 4.5 / 16.5, rel_tol=1e-9)

moe_rng = np.random.default_rng(3)                                            # brute force: literal per-expert arrays
experts = [moe_rng.integers(1, 5, size=(3, 3)) for _ in range(8)]
shared_w = moe_rng.integers(1, 5, size=(2, 2))
total_bf = shared_w.size + sum(e.size for e in experts)
chosen = [0, 3]                                                                # the k = 2 experts one token routes to
active_bf = shared_w.size + sum(experts[i].size for i in chosen)
assert total_bf == shared_w.size + 8 * experts[0].size
assert active_bf == shared_w.size + 2 * experts[0].size

print("all checks passed")
```

</details>

</details>
