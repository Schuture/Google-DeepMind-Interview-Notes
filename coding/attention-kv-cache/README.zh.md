# 带 KV cache 与在线 softmax 的注意力

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 机器学习实现（NumPy） | ★★★★☆ | 困难 | RS · RE · MLE | attention, kv-cache, online-softmax, flash-attention, grouped-query-attention, numerical-stability | 3 个部分 / 45–60 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

因果自注意力（causal self-attention）把一个输入序列映射为形状相同的输出序列，其中位置 $t$ 的输出只依赖于
位置 $0,\dots,t$。这道题实现同一个计算的三种形式：一次性处理整个序列的批量形式、复用已经算好的键和值的
自回归解码路径，以及内存受限的分块（blockwise）形式。

三个部分都共用下面这些记号：$x \in \mathbb{R}^{T\times d}$ 是一个输入，有 $T$ 个位置（行，下标
$0,\dots,T-1$），每个位置 $d$ 个特征；权重矩阵 $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d\times d}$ 把它投影成
查询（query）、键（key）、值（value）和最终输出。一个整数个数的*头*（head）$h$ 把 $d$ 个特征切成 $h$ 个
连续的块，每块 $d_h = d/h$ 个特征：头 $i$ 拥有 $Q = xW_Q$、$K = xW_K$、$V = xW_V$ 的第
$i\cdot d_h,\dots,(i+1)\cdot d_h - 1$ 列。记 $Q_i, K_i, V_i \in \mathbb{R}^{T\times d_h}$ 为头 $i$ 的切片，
头 $i$ 计算

$$\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_h}} + M\right)V_i, \qquad
M_{ts} = \begin{cases}0 & s \le t\\ -\infty & s > t\end{cases},$$

softmax 沿键轴 $s$ 进行，于是 $M$ 的第 $t$ 行会挡住位置 $t$ 之后的每一个键（*因果掩码*，causal
masking），并且要以数值稳定的方式计算（在做指数运算之前先减去每行的最大值）。把
$\mathrm{head}_0,\dots,\mathrm{head}_{h-1}$ 沿特征轴拼接起来，再乘以 $W_O$，就得到最终输出，形状
$(T, d)$。

### Part 1 —— 带因果掩码的多头注意力

```py
def causal_mha(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
               n_heads: int) -> np.ndarray:
    """x: (T, d). Wq, Wk, Wv, Wo: (d, d). d must be divisible by n_heads; raises ValueError
    otherwise. Returns (T, d)."""
```

对全部 `n_heads` 个头、全部 $T$ 个位置一次性计算上面的公式。如果 `d % n_heads != 0`，抛出 `ValueError`。

取 `n_heads = 1`，$W_Q = W_K = W_V = W_O = I_2$（于是 $Q = K = V = x$）：

```text
x = [[1, 0],
     [0, 1],
     [1, 1]]

第 0 行只能关注自己（因果）-> output[0] = V[0] = [1, 0]

第 1 行可以关注第 0、1 行：
  原始得分 Q[1].K[0], Q[1].K[1] = 0, 1；乘以 1/sqrt(2)：0, 0.7071
  weights ~= [0.3302, 0.6698]
  output[1] ~= 0.3302*[1,0] + 0.6698*[0,1] = [0.3302, 0.6698]

第 2 行可以关注第 0、1、2 行：
  原始得分 Q[2].K[0], Q[2].K[1], Q[2].K[2] = 1, 1, 2；乘以 1/sqrt(2)：0.7071, 0.7071, 1.4142
  weights ~= [0.2483, 0.2483, 0.5035]
  output[2] ~= 0.2483*[1,0] + 0.2483*[0,1] + 0.5035*[1,1] = [0.7517, 0.7517]
```

### Part 2 —— 用 KV cache 做增量解码

*自回归解码*（autoregressive decoding）逐个位置地生成序列，每个新位置的输入 $x_t$ 依赖于上一步、也就是
位置 $t-1$ 已经产出的结果。如果每来一个新的 $t$ 都对整个前缀 $x_0,\dots,x_t$ 重新跑一遍 `causal_mha`，
就会重复劳动：位置 $s<t$ 的键和值不依赖它之后的任何位置，只需要算一次。*KV cache* 按头分别保存目前为止
处理过的每个位置的键和值。

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

对任意 $x, W_Q, W_K, W_V, W_O$、`n_heads`，新建一个空的 `KVCache`，依次对 $t = 0,\dots,T-1$ 调用
`decode_step(x[t], cache, ...)`，每一步都必须返回 `causal_mha(x, Wq, Wk, Wv, Wo, n_heads)` 第 $t$ 行、
误差不超过绝对容差 $10^{-10}$——`decode_step` 除了刚传进来的这一行之外看不到 $x$ 的任何其他行，能做到
这一点，全靠它需要知道的关于更早位置的一切都已经在 `cache` 里了。用和 Part 1 相同的例子：

```text
cache = KVCache(n_heads=1, d_head=2)                  # len(cache) == 0

decode_step(x[0], cache, ...) -> [1, 0]                # len(cache) == 1；等于 causal_mha(x, ...)[0]
decode_step(x[1], cache, ...) -> [0.3302, 0.6698]      # len(cache) == 2；等于 causal_mha(x, ...)[1]
decode_step(x[2], cache, ...) -> [0.7517, 0.7517]      # len(cache) == 3；等于 causal_mha(x, ...)[2]
```

另外，实现

```py
def kv_cache_bytes(n_layers: int, n_kv_heads: int, d_head: int, seq_len: int, batch: int,
                    bytes_per_elem: int) -> int:
    """Total bytes of every layer's KV cache for `batch` sequences of length seq_len, with
    n_kv_heads key/value heads of size d_head per layer and bytes_per_elem bytes per stored
    number."""
```

取 $L=32$ 层，$h_{kv}=8$ 个键/值头，$d_h=128$，序列长度 $T=32{,}768$，batch 为 $1$，每个元素 $2$ 字节，
`kv_cache_bytes(32, 8, 128, 32_768, 1, 2)` 恰好等于 $2^{32}$。

### Part 3 —— 用在线 softmax 分块计算注意力

这一部分去掉了因果掩码和权重矩阵，直接处理已经投影好的某一个头的查询、键、值：
$Q \in \mathbb{R}^{T_q\times d_h}$，$K, V \in \mathbb{R}^{T_k\times d_h}$，计算
$O = \mathrm{softmax}(QK^\top/\sqrt{d_h})V \in \mathbb{R}^{T_q\times d_h}$（不带掩码）。实现时不能在内存
中保留完整的 $T_q \times T_k$ 得分矩阵：按连续的、每块 `block` 个位置的方式访问这些键
（$K[0{:}\mathrm{block}], K[\mathrm{block}{:}2\cdot\mathrm{block}], \dots$，最后一块可能更短），每访问
完一块，就为每个查询行更新一个运行中的最大得分 $m$、运行中的归一化项 $\ell$，以及运行中的未归一化输出
$o \in \mathbb{R}^{d_h}$，使得所有块都访问完之后，$o/\ell$ 恰好等于该查询在 $O$ 中的那一行。在访问任何
一块之前，$m=-\infty$，$\ell=0$，$o=0$。推导 $m,\ell,o$ 从一块到下一块应该如何更新，使得这个不变量在
每一块之后都成立，不仅仅是最后一块，并对访问过的块数做归纳证明。

```py
def attention_blockwise(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block: int) -> np.ndarray:
    """Q: (T_q, d_h). K, V: (T_k, d_h). Returns softmax(Q @ K.T / sqrt(d_h)) @ V, computed
    `block` positions of K/V at a time, for any block >= 1."""
```

结果必须对从 $1$ 到大于 $T_k$ 的每一个 `block` 都等于上面的直接计算，并且即使原始得分
$QK^\top/\sqrt{d_h}$ 达到 $10^3$ 量级甚至更大，也要保持有限——这时不做平移的 `np.exp` 早就已经溢出了。

取 $Q=[[1]]$，$K=[[0],[1],[3],[2]]$，$V=[[10],[20],[30],[40]]$（$d_h=1$，一个查询，`block = 2`，于是
原始得分 $QK^\top=[0,1,3,2]$ 被分成两块 $[0,1]$ 和 $[3,2]$）：

```text
访问任何一块之前：m = -inf, l = 0, o = [0]

块 [0, 1]：局部最大值 = 1 -> m = 1
  l = 1.3679   (= exp(0-1) + exp(1-1))
  o = [23.6788]   (= exp(0-1)*10 + exp(1-1)*20)

块 [3, 2]：局部最大值 = 3 -> m = 3（比 1 大，于是块 [0, 1] 的累积量要乘以 exp(1-3) 重新定标）
  l = exp(1-3)*1.3679 + (exp(3-3) + exp(2-3)) = 1.5530
  o = exp(1-3)*[23.6788] + (exp(3-3)*30 + exp(2-3)*40) = [47.9198]

output = o / l = [30.8562]
```

## 参考解答

<details>
<summary>展开参考解答</summary>

写代码之前有两点值得和面试官确认：`decode_step` 和 `KVCache` 一次只服务一个序列，没有 batch 轴（这与
`x_t` 的 `(d,) -> (d,)` 签名一致）——一个支持批量的解码器会给每个缓存的张量都多加一个前导的 batch 维度，
但下面的更新规则本身不变；以及除非某一部分另有说明，所有数组都是 NumPy 默认的 `float64`，所以只有
Part 3 的大得分压力测试和精度相关的追问才会真正涉及精度问题。

### Part 1

$Q$、$K$、$V$ 各是一次投影，`x @ W`，形状 $(T, d)$。把最后一个轴拆成 $h$ 个、每个 $d_h$ 个特征的头，
再把头轴移到最前面，是一次 `reshape` 加一次 `transpose`，得到形状 $(h, T, d_h)$；接下来 NumPy 的 `@`
会把每个头的得分和加权求和都当作在头这个前导轴上分批处理，就像处理一个前导的 batch 轴一样。因为位置
$t$ 总能看到自己（$M_{tt}=0$，从不是 $-\infty$），这里的 `scores.max(axis=-1)` 永远不是 $-\infty$，
稳定 softmax 也就不会算出 $0/0$。

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

复杂度：四个 $(d,d)$ 投影合计花费 $O(Td^2)$；每个头的得分和加权求和合计花费
$O(hT^2d_h) = O(T^2d)$，因为 $h\cdot d_h = d$；`scores`/`weights` 占用 $O(hT^2)$ 内存。序列很长时
（$T \gg d$）$T^2$ 那一项占主导；序列短时，$d^2$ 的投影占主导。

### Part 2

cache 只保存键和值，从不保存查询：位置 $s$ 的键和值会被之后每一个关注它的查询复用，而位置 $t$ 自己的
查询只会被用一次、用来产出位置 $t$ 的输出，留着它不会带来任何好处。`KVCache` 把每个已追加位置的键和值
分别存在一个普通的 Python list 里（每个位置一项，形状都是 `(n_heads, d_head)`），需要时才把它们堆叠成
`(n_heads, t, d_head)` 的数组。`decode_step` 只把新位置 $x_t$（而不是整个前缀）投影一次，得到
$W_Q, W_K, W_V$ 对应的结果，*先*把新的键和值追加进 cache、*再*做注意力——这样位置 $t$ 自己的键和值
已经是它关注范围的一部分，和公式里 $M_{tt}=0$ 一致——然后对 cache 目前持有的所有位置跑一遍和 Part 1
相同的稳定 softmax，最后乘以 $W_O$。

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

**每一步的开销。** 把 $x_t$ 投影到 $W_Q,W_K,W_V,W_O$ 花费 $O(d^2)$，和已经缓存了多少个位置无关。让
$q_t$（每个头一个查询）关注 cache 里现在的 $t+1$ 个键——刚追加进去的位置 $t$ 自己的键，加上它之前的
$t$ 个——每个头要做 $t+1$ 次点积和 $t+1$ 次加权求和，每次长度都是 $d_h$，也就是 $O(t\cdot d_h)$；对
$h$ 个头求和，$O(t\cdot d_h\cdot h)=O(td)$。于是一步花费 $O(d^2+td)$，$T$
步合计 $O(Td^2+T^2d)$——和 `causal_mha` 自己的复杂度（Part 1）完全一致，只是把它摊开成一次一个 token，
而不需要一开始就拿到整个序列。如果不用 cache，而是在第 $t$ 步从头重算——对前 $t+1$ 个位置重新跑一遍
`causal_mha`——一旦 $t$ 超过 $d$，单单这一步就要花费 $O(t^2d)$，因为它把 Part 1 自己 $O(t^2d)$ 的注意力
那一项，在长度为 $t$ 的前缀上又算了一遍；把 $O(t^2d)$ 对 $T$ 步求和，总共要花费 $O(T^3d)$，比用 cache
多出整整一个 $T$ 的倍数，而生成的却是同样的 $T$ 个 token。

**字节数公式。** $L=$ `n_layers` 个 Transformer 层，每层各自维护自己的 cache。每一层为每个已缓存的
位置存一个键向量和一个值向量，每个都是 $d_h$ 个元素长，一共 $h_{kv}=$ `n_kv_heads` 个头、`seq_len`
个位置、`batch` 个并排处理的序列。于是元素总数是
$L\cdot 2\cdot h_{kv}\cdot d_h\cdot\mathrm{seq\_len}\cdot\mathrm{batch}$（因子 $2$ 是因为键和值分别
计数），再乘以 `bytes_per_elem` 就得到字节数。取 $L=32$，$h_{kv}=8$，$d_h=128$，$T=32{,}768$，batch
为 $1$，$2$ 字节：$32\cdot2\cdot8\cdot128\cdot32{,}768\cdot1\cdot2 = 2^5\cdot2\cdot2^3\cdot2^7\cdot2^{15}\cdot2 = 2^{32}$
字节，恰好，也就是 $4\ \mathrm{GiB}$。

**为什么是 `n_kv_heads`，而不是 `n_heads`。** cache 只存键和值，从不存查询，而*分组查询注意力*
（grouped-query attention）让 $K$、$V$ 的头数比 $Q$ 少：它把 $K$、$V$ 投影到 $h_{kv}<h$ 个键/值头，
让每组 $h/h_{kv}$ 个查询头共用同一份 $K_i,V_i$ 切片；上面实现的普通多头注意力，正是 $h_{kv}=h$
（每个查询头都有自己的键/值头）这个特例。因为 $Q$ 从不进 cache——每一步只用当前这一个查询，用完即
弃——`kv_cache_bytes` 只依赖 $h_{kv}$：把 $h_{kv}$ 调低到比 $h$ 小，缓存就会变小，而参与关注的查询头
数量不受影响（追问）。

### Part 3

对某一个查询行，记 $s_j=(QK^\top)_j/\sqrt{d_h}$ 为它对键 $j$ 的原始得分；访问完块 $B_1,\dots,B_n$
（对 $\{0,\dots,T_k-1\}$ 按连续块划分之后的一段前缀）之后，维持这样一个不变量：

$$m_n=\max_{j\in B_1\cup\cdots\cup B_n}s_j,\qquad
\ell_n=\sum_{j\in B_1\cup\cdots\cup B_n}e^{s_j-m_n},\qquad
o_n=\sum_{j\in B_1\cup\cdots\cup B_n}e^{s_j-m_n}v_j,$$

这样 $o_n/\ell_n$ 就恰好是目前访问过的这些键上的 softmax 加权和（$m_n$ 起的作用和 Part 1 里那一行的
最大值一样，只是限制在键的一段前缀上）。基础情形 $n=0$（还没访问任何块）由给定的初始值直接满足：
空集合的最大值是 $-\infty$，空的和是 $0$。

**归纳步骤。** 假设不变量在 $n$ 块之后成立；访问块 $B_{n+1}$，它的原始得分是 $\{s_j : j\in B_{n+1}\}$。
两个集合并集上的最大值，等于各自最大值的最大值，所以

$$m_{n+1}=\max\Bigl(m_n,\ \max_{j\in B_{n+1}}s_j\Bigr)$$

恰好等于 $\max_{j\in B_1\cup\cdots\cup B_{n+1}}s_j$，与不变量对 $m_{n+1}$ 的定义一致。把 $\ell_{n+1}$
定义中的求和，按 $B_1\cup\cdots\cup B_{n+1}$ 的这两部分拆开：

$$\ell_{n+1}=\sum_{j\in B_1\cup\cdots\cup B_n}e^{s_j-m_{n+1}}+\sum_{j\in B_{n+1}}e^{s_j-m_{n+1}}.$$

在第一个和式里，对每个 $j$ 都有 $e^{s_j-m_{n+1}}=e^{s_j-m_n}\cdot e^{m_n-m_{n+1}}$，而
$e^{m_n-m_{n+1}}$ 不依赖 $j$，可以提到求和外面，剩下
$e^{m_n-m_{n+1}}\sum_{j\in B_1\cup\cdots\cup B_n}e^{s_j-m_n}=e^{m_n-m_{n+1}}\ell_n$，用到了归纳假设。
于是 $\ell_{n+1}=e^{m_n-m_{n+1}}\ell_n+\sum_{j\in B_{n+1}}e^{s_j-m_{n+1}}$，正是要证明的更新公式；对
$o$ 逐项做完全相同的提取（每个 $v_j$ 只是原样跟着走）就得到
$o_{n+1}=e^{m_n-m_{n+1}}o_n+\sum_{j\in B_{n+1}}e^{s_j-m_{n+1}}v_j$。由归纳法，这个不变量在每一块之后
都成立，特别是在最后一块 $N$ 之后：这时 $m_N=\max_js_j$ 取遍所有键，$\ell_N=\sum_je^{s_j-m_N}$，于是

$$\frac{o_N}{\ell_N}=\sum_j\frac{e^{s_j-m_N}}{\sum_ke^{s_k-m_N}}v_j=\sum_j\mathrm{softmax}(s)_j\,v_j,$$

和 Part 1 用到的是同一个减去最大值不改变结果的等式，只不过这里是对整行得分一次性用，而不是一块一块
地用。把每一步的更新写得紧凑一些，记 $m,\ell,o$ 为一块之前的累积量，$m',\ell',o'$ 为这一块之后的
累积量，$s_j$ 取遍这一块：

$$m'=\max\Bigl(m,\max_js_j\Bigr), \qquad \ell'=e^{m-m'}\ell+\sum_je^{s_j-m'}, \qquad
o'=e^{m-m'}o+\sum_je^{s_j-m'}v_j.$$

上面每一个指数要么是 $s_j-m'\le0$（由 $m'$ 的定义），要么是 $m-m'\le0$（因为
$m'=\max(m,\dots)\ge m$），所以这个递推式里不论原始得分多大，都不会有指数运算溢出。$-\infty$ 唯一
可能惹麻烦的地方是 $-\infty-(-\infty)$，结果是 `nan`，而这需要相减的两边*同时*是 $-\infty$。这里
$m$ 一开始是 $-\infty$，但 $m'=\max(m,\max_js_j)$ 从第一块起就已经是有限值，因为一个块从不为空、
它的得分也都是有限的；于是 $m-m'$ 永远只是 $-\infty$ 减去一个有限值，也就是干干净净的 $-\infty$，
而 $e^{-\infty}=0$，不需要另外加什么保护。

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

复杂度：总时间是 $O(T_qT_kd_h)$，和直接计算一样——分块只是改变了这些运算被调度的方式，没有改变运算的
总量。任意时刻，内存里只有一块 $(T_q,\mathrm{block})$ 的得分，加上 $O(T_qd_h)$ 的累积量 $m,\ell,o$，
而不是 Part 1 里 `scores` 需要的整个 $(T_q,T_k)$ 得分矩阵；这正是 FlashAttention 这个名字所指的内存
上界，也正是上面这个递推式重要的原因：它让 `block` 可以选得足够小，装进更快的内存（缓存或 SRAM），而
不管 $T_k$ 有多大，并且在总运算量和最终结果上都不用付出任何代价。

### 追问

- **分块版本里的因果掩码。** 加上因果掩码后，一旦 $\mathrm{start}>t$，块 $[\mathrm{start},\mathrm{end})$
  对查询 $t$ 就没有任何贡献：这样的块可以整个跳过，连一个得分都不用算，而不必逐项掩码。只有跨在对角线上
  的块（一部分键 $\le t$、一部分 $>t$）才还需要 Part 1 里那种逐项的 $-\infty$ 掩码，然后再做 $m/\ell/o$
  的更新；完全落在对角线下方的块完全不需要掩码，完全落在对角线上方的块则直接跳过。
- **分组查询注意力对缓存的影响。** 把 `n_kv_heads` 从 $h$（普通多头注意力）调小到 $h_{kv}<h$，会让
  `kv_cache_bytes` 恰好缩小 $h/h_{kv}$ 倍这个分组大小，因为公式里其余每一个因子都不变。取 $h=64$ 个
  查询头共用 $h_{kv}=8$ 个键/值头（分组大小 $8$）为例，Part 2 里 $2^{32}$ 字节（$4\ \mathrm{GiB}$）的
  缓存，换成 $h_{kv}=64$ 的完整多头注意力就需要 $2^{35}$ 字节（$32\ \mathrm{GiB}$）——这 $8$ 倍的差距
  完全是缓存占用上的差距，因为 $Q$ 自身的大小从未出现在这个公式里。
- **bf16，以及为什么低精度下减去最大值更重要。** bf16 保留了 float32 的 8 位指数位，所以它只会在和
  float32 相同的地方溢出（原始得分超过大约 $88$）；不做平移时，它的问题不在范围，而在精度：只有 7
  位尾数，大约相当于 2 位有效十进制数字，于是只要同一行里两个指数值相差超过大约 $2^7$ 倍（得分差距
  超过大约 $\ln128\approx4.85$），把它们相加时较小的那个就会在 bf16 里被完全舍入掉——而这在一行得分
  里是很平常的差距。float16 的尾数位更多（$10$ 位），但只有 5 位指数位，所以原始得分一旦超过大约
  $11$ 就会直接溢出（下面验证代码里用的那个不大的得分差距就已经如此）。减去当前的运行最大值，能让
  每一个平移后的指数值都落在 $(0,1]$ 里，不论原始得分的范围有多大，两种失效方式都被一次性绕开
  了——这也是为什么即使 $\ell$ 本身是用低精度类型累加的，生产环境的 kernel 也依然要做这一步平移。
- **滑动窗口注意力（sliding-window attention）。** 把因果掩码收紧，让查询 $t$ 只能关注
  $[\max(0,t-W+1),t]$ 范围内的键，窗口大小为 $W$；配合 cache，这能把每个头的内存限制在 $W$ 项以内，
  一旦缓存超过 $W$ 就淘汰最旧的键/值（相当于一个大小为 $W$ 的环形缓冲区，取代 Part 2 里那个不设上限
  的 list），并且每次 `decode_step` 只花费 $O(Wd)$，而不是 $O(td)$，与序列已经长到多少无关。
- **对缓存中的键应用 RoPE。** 旋转位置编码（RoPE）在做点积之前，把 $q_t$ 和 $k_t$ 按一个只取决于绝对
  位置 $t$（以及被旋转的那一对特征）的角度旋转。因为这个旋转不依赖任何在位置 $t$ 之后才算出来的东西，
  它可以只做一次——在 $k_t$ 刚刚被写入 cache 的那一刻，正好就是 Part 2 里 `append` 所在的位置——而不
  必每次有新的一步关注到它时，都从一个存好的、未旋转的键重新推导一遍。

<details>
<summary>验证代码（可运行）</summary>

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
