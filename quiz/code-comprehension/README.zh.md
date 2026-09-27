# 代码理解：这段代码在计算什么？

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 知识问答 · 口述读代码 | ★★☆☆☆ | 中等 | RS · RE · Intern · SWE | code-reading, convolution, padding, numerical-stability, streaming-statistics, attention-masks, reservoir-sampling | 8 段代码 / 30–45 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

下面八段代码，每一段都是一个完整的 Python 函数（Q2 的 `g` 调用了 Q1 的 `f`）；它们都不会被运行，任务是照原样读懂并解释每一段代码。对每一段代码，先用一到两句话说明它计算的是什么，用的是教科书或相关库文档里的标准术语；然后回答它自己的子问题：凡是要求给出具体追踪值的地方，给出精确的数值；输出的长度或形状，写成该函数自身参数的公式；给出规定的时间或内存开销；以及——在题目要求时——准确说出是哪一种输入会让函数出问题、具体如何出问题，或者对代码做出指定的修改后，具体会发生什么变化。

### 卷积

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

`f` 计算的是什么，用序列 `x` 和权重 `w` 来表示？把它输出的长度写成 $n = \mathtt{len(x)}$ 和 $k = \mathtt{len(w)}$ 的
公式，并给出它执行的标量乘法次数。手动追踪 `f([1, 2, 3, 4, 5], [1, 0, -1])`。`np.convolve(x, w, "valid")` 计算的是
`x` 与 `w` 的数学卷积（convolution）；精确说明 `f` 的结果与它有何不同，以及当 `w` 是网络学到的一组权重、而不是固定
已知的核时，这个不同为什么不重要。当 `len(w) > len(x)` 时，`f` 返回什么？

**Q2.**

```py
def g(x, w):
    p = len(w) // 2
    xp = [0.0] * p + list(x) + [0.0] * p
    return f(xp, w)
```

用 $n = \mathtt{len(x)}$ 给出 `g` 输出的长度，分别针对 `len(w)` 为奇数和偶数两种情况。追踪
`g([1, 2, 3, 4, 5], [1, 0, -1])`，并解释为什么它的最后一项和内部各项显示出的规律不一致。说出除了补零之外的三种填充
（padding）方式，并说明输入信号在边界附近缺少哪种性质时，补零会让 `g` 在那里产生的值出现偏差。

**Q3.**

```py
def h(x, w, stride=1, dilation=1, pad=0):
    xp = [0.0] * pad + list(x) + [0.0] * pad
    span = dilation * (len(w) - 1) + 1
    return [sum(xp[i + dilation * j] * w[j] for j in range(len(w)))
            for i in range(0, len(xp) - span + 1, stride)]
```

`h` 用 `stride`（只保留每隔 `stride` 个的输出位置）、`dilation`（把 $k = \mathtt{len(w)}$ 个抽头之间的间隔从相邻
改成 `dilation`）和显式的两侧补零 `pad`，把 `f` 一般化了。用 $n = \mathtt{len(x)}$、$k$、$p = \mathtt{pad}$、
$s = \mathtt{stride}$、$d = \mathtt{dilation}$ 推导出 `h` 输出长度的闭式公式，并在 $n=10,\,k=3,\,p=1,\,s=2,\,d=2$
时求值。把一个输出项的*感受野*（receptive field）定义为它所依赖的原始输入项的个数。现在把 $L$ 份 `h` 叠在一起，每一份
都用 stride $1$、相同的奇数核大小 $k$，第 $\ell$ 份的 dilation 为 $2^\ell$（$\ell = 0, \dots, L-1$，所以第一份完全
不带 dilation），并且每一份都恰好用足够的 padding，使序列长度在每一级都保持不变。推导最后一份输出中、远离序列两端
的一项的感受野，写成 $k$ 和 $L$ 的函数。

### 数值计算

**Q4.**

```py
import math

def lse(z):
    m = max(z)
    s = sum(math.exp(v - m) for v in z)
    return m + math.log(s)
```

`lse` 计算的是什么？考虑到浮点数运算只能表示有限范围内的数值、超出范围就会上溢为无穷，为什么在调用 `exp` 之前先
减去 `m`，对大量级的输入很重要？给出一个会让 `lse` 抛出异常的输入，再给出一个非空、却会让它悄悄返回 `nan` 的输入——
对于后一种情况，准确说出是哪个子表达式产生了 `nan`，以及原因——然后修好这个函数，使它能正确处理第二种情况（第一种
情况仍然应该抛出异常）。有了 `lse`，如何在不另外写一份单独做数值稳定化处理的实现的前提下，计算 `z` 的 softmax（与
`z` 形状相同的向量，其第 $i$ 个分量是 $e^{z_i} / \sum_k e^{z_k}$）？

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

`stats` 逐个读取 `xs` 中的值、且从不保存整个序列，它计算的是什么？说出这种方法在实践中优先于哪个教科书上的方差
公式（它同样可以借助累积和单遍算出），并给出四个共享同一个巨大公共偏移量的数（即彼此的差远小于数值本身），使得那个
公式在 IEEE 754 双精度下严重出错，而 `stats` 不会。`stats([5.0])` 会怎样，`stats([])` 又会返回什么？这两者中哪一个是更糟糕的失败方式，为什么？
给定 `stats` 内部为一段数据算出的三元组 `(n, mean, M2)`（其中 `M2` 就是那个不断累积的 `m2`，还没做最后的除法），
以及第二段不相交数据对应的三元组，给出一个公式，算出把这两段数据拼接起来直接处理时会得到的那个三元组。

### 注意力

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

`scores` 是一个形状为 `(T, T)` 的方阵。说出结果里第 `i` 行是哪些列 `j` 获得了非零权重，以及这一行的和是多少。假设
再叠加一个掩码（mask）到 `mask` 上（比如标出某些列是填充（padding）），使得某一整行都被掩去了。如果把 `-1e9` 换成
`-np.inf`，比较一下 `weights` 在那一行上会返回什么。另外，如果在调用之前把 `scores` 转换成 `np.float16`，会出什么
问题？应该怎样修改掩码所用的常数来避免它？把 `scores` 本身、对序列长度 `T` 和单个注意力头而言的内存开销，写成关于
`T` 的公式。

### 采样与几何

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

`stream` 是一个可迭代对象（iterable），它的长度不必事先已知，也不必有限；`rng` 是一个随机源，带有方法 `randrange(m)`，返回
$\{0, \dots, m-1\}$ 上均匀分布的一个整数。`sample` 计算的是什么？对一个恰好有 $n \ge k$ 个元素的流，证明每个元素
最终出现在返回结果里的概率都相同，并给出这个概率。如果把 `rng.randrange(i + 1)` 换成 `rng.randrange(i)`，会有什么
定量的变化？

**Q8.**

```py
import numpy as np

def dist(X, Y):
    sq = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.maximum(sq, 0))
```

`X` 和 `Y` 的形状分别是 `(n, d)` 和 `(m, d)`：即分别是 `d` 维空间中的 `n` 个点和 `m` 个点。`dist` 计算的是什么？
如果没有 `np.maximum`，`sq` 有时会被当作一个很小的负数传给 `np.sqrt`；说出在 `X`、`Y` 满足什么条件时会发生这种
情况，以及 `np.sqrt` 遇到负数输入会怎样。另外，解释为什么当两个点彼此靠近、却都离原点很远时，`dist` 返回的值会不
准确，并给出一个仍保持同样 $O(nmd)$ 时间复杂度的修正办法。给出计算出完整的 `dist` 矩阵所需的时间和内存开销。

## 参考解答

<details>
<summary>展开参考解答</summary>

先口头确认一点：每个函数都要照原样解释，而不是改写；某个函数在特定输入上出错时（Q4、Q5、Q6 都是如此），准确说出
这个输入以及它给出的错误结果，本身就是答案的一部分。

### 卷积

**Q1.** `f` 计算的是 `x` 与 `w` 的*互相关*（cross-correlation）：这正是深度学习框架所说的（一维）卷积
（convolution）所用的同一种滑动点积操作，尽管真正的数学卷积会先把 `w` 反转过来，而这里没有反转。只要 $k \le n$，
它的输出长度就是 $n-k+1$（`w` 的整个窗口能放进 `x` 里的每一个位置各贡献一项），开销是 $k(n-k+1)$ 次标量乘法，
每个输出项、每个抽头各一次。追踪 $f([1,2,3,4,5],\,[1,0,-1])$：窗口 $i=0$ 给出 $1\cdot1+2\cdot0+3\cdot(-1)=-2.0$；
$i=1$ 给出 $2-4=-2.0$；$i=2$ 给出 $3-5=-2.0$，结果是 $[-2.0,-2.0,-2.0]$。`np.convolve(x, w, "valid")` 会先把 `w`
反转再滑动，计算的是 $\sum_j x_{i+j}w_{k-1-j}$ 而不是 `f` 的 $\sum_j x_{i+j}w_j$；只有当 `w` 反转后与自身相同时
两者才会一致，而这里 $w=[1,0,-1]$ 反转后是 $[-1,0,1]=-w$，二者并不一致。这个反转是信号处理里的一个记账惯例：在
那里 `w` 是固定已知的滤波器，反转能让两次卷积的复合满足结合律；而当 `w` 换成网络*学到*的一组权重时，“在 $w$ 上
训练”和“在反转后的 $w$ 上训练”没有任何区别——训练过程只会落到拟合数据的那个方向上——于是框架干脆去掉反转，把这个
不带反转的操作直接叫作“卷积”。当 `len(w) > len(x)` 时，`w` 的整个窗口根本放不进 `x`：`range(n - k + 1)`
的终止值非正，是一个空的 `range`，所以 `f` 返回 `[]`。

**Q2.** `g` 先在 `x` 两侧各补上 $p=\lfloor k/2\rfloor$ 个零，再计算和 `f` 相同的互相关——这是“same”填充，之所以
这样选取 $p$，是为了让奇数 $k$ 时输出能与原始的 `x` 一一对应。它的长度是 $(n+2p)-k+1$；代入 $p=\lfloor k/2\rfloor$，
奇数 $k$ 时恰好是 $n$（$k-1$ 的两半恰好抵消两侧的填充），偶数 $k$ 时是 $n+1$：要保持长度，总共需要补 $k-1$ 个
零，这是个奇数，无法在两侧平分，所以两侧各补 $k/2$ 个就多了一个（偶数核需要不对称的填充：一侧 $k/2$ 个，另一侧
$k/2-1$ 个）。追踪 $g([1,2,3,4,5],\,[1,0,-1])$：$p=1$，所以 `xp` $=[0,1,2,3,4,5,0]$，用 $[1,0,-1]$ 对它做互相关，
得到 $[-2.0,-2.0,-2.0,-2.0,4.0]$。第 $i$ 项是 $\mathtt{xp}_i-\mathtt{xp}_{i+2}=x_{i-1}-x_{i+1}$（`x` 从 $0$ 开始
计数，补上的零充当 $x_{-1}$ 和 $x_5$）：第 $1$ 到第 $3$ 项恰好就是 `f` 的输出，而两端的两项各读到一个补上的零。
第一项 $0-x_1=-2.0$ 之所以与内部一致，只是因为这条斜坡恰好在下标 $-1$ 处外推为 $0$；最后一项是 $x_3-0=4.0$，而
不是更长的斜坡在那个位置本该给出的 $x_3-x_5=-2$：补零并没有延续数据，而是在真实的边缘样本旁边硬塞进一个恰好为
$0$ 的值，这个特定的核就把它读成了一次陡降。镜像填充（reflect，把样本跨边界对称地翻过去）、复制填充（replicate，
重复边缘样本）和循环填充（circular，绕到序列的另一端）是另外三种方式；只要信号本身在边界附近并不衰减到 $0$，补零
就会让那里的值产生偏差，因为它会在边界处制造一个人为的跳变。镜像和复制填充则让信号在边界处延续其边缘取值，消除了
这个跳变，但并没有消除所有边缘效应：在这条斜坡上，它们给出的边缘值分别是 $0$ 和 $-1$，而不是斜率对应的 $-2$。

**Q3.** `h` 是深度学习里普遍使用的一般化一维卷积：`stride` 只保留每隔 `stride` 个的输出位置，`dilation` 把 `w` 的
$k=\mathtt{len(w)}$ 个抽头分散到跨度为 $d(k-1)+1$ 个输入位置上，而不是相邻的 $k$ 个，`pad` 则在两侧各加上与 $k$
无关的显式补零。核的一次应用要跨越补零后序列里的 $d(k-1)+1$ 个位置，而补零后序列本身的长度是 $n+2p$；从补零后
位置 $i$ 开始的一次应用需要 $i+d(k-1)+1 \le n+2p$，所以合法的起始位置是 $i=0,1,\dots,(n+2p-d(k-1)-1)$，在这
$n+2p-d(k-1)$ 个位置里每隔 $s$ 个取一个，共有

$$\left\lfloor \frac{n+2p-d(k-1)-1}{s} \right\rfloor + 1$$

个。在 $n=10,k=3,p=1,s=2,d=2$ 时：$d(k-1)=4$，代入公式得 $\lfloor(10+2-4-1)/2\rfloor+1=\lfloor 7/2\rfloor+1=3+1=4$。
关于感受野：每一级都选取恰好保持序列长度不变的 padding 时，单独一级带 dilation 的层里，一个输出项是 $k$ 个、彼此
间隔 $d_\ell=2^\ell$ 的输入项的函数，也就是跨越*那一级自己*输入里连续的 $d_\ell(k-1)+1$ 项——所以第 $\ell$ 级会
在前面各级已经积累起来的感受野之上，再加上 $d_\ell(k-1)=(k-1)2^\ell$（保持长度不变的一级，只会让下一级看到的跨度
变宽，绝不会让前面各级已经带进来的部分变窄）。从单独一个位置出发（在任何一级之前，感受野是 $1$），把
$\ell=0,\dots,L-1$ 各级的贡献加总：

$$\text{感受野} = 1+(k-1)\sum_{\ell=0}^{L-1}2^\ell = 1+(k-1)(2^L-1),$$

用到了几何级数 $\sum_{\ell=0}^{L-1}2^\ell=2^L-1$。这正是 WaveNet 里的标准论证：把 dilation 按指数增长的方式叠加
起来，只用 $L$ 级、$O(Lk)$ 个参数，就能让感受野也随 $L$ 指数增长，而不必像单独一级不带 dilation 的层那样，需要
$O(2^Lk)$ 才能达到同样的感受野。

### 数值计算

**Q4.** `lse` 计算的是 $\log\sum_i e^{z_i}$（log-sum-exp），且始终不需要真正算出 $\sum_i e^{z_i}$ 本身。在调用
`exp` 之前先减去 $m=\max(z)$，原因在于 Python 的 `math.exp` 和 NumPy 的不一样，并不会悄悄上溢为 `inf`：一旦参数
超过 $\ln(\texttt{sys.float\_info.max}) \approx 709.78$，它会直接抛出 `OverflowError`，所以不做这次减法的话，
只要 `z` 里有任何一项达到这个量级，`lse` 就会直接崩溃，而不只是损失一点精度。减去 $m$ 之后，真正传给 `exp` 的最大
指数恰好是 $0$，其余的都更小，所以无论 `z` 的量级多大，这个和都不可能上溢；而且由于最大值对应的那一项是
$e^0=1$，这个和至少为 $1$，也就不可能下溢为 $0$——如果不做平移，像 $[-1000,-1001]$ 这样的输入会让每个 `exp` 都
下溢为 `0.0`，而 `math.log(0.0)` 会抛出 `ValueError`。`lse([])` 在第一行就会抛出 `ValueError`，因为对空序列取
`max` 没有定义。`lse([-math.inf, -math.inf])` 返回
`nan`：`m` 是 `-math.inf`，随后生成器对 `v = -math.inf` 计算 `v - m`，即 `-math.inf - (-math.inf)`，IEEE 754 把
它定义为 `nan` 而不是 $0$——于是 `s` 变成 `nan`，`m + math.log(s)` 也就一直是 `nan`。数学上正确的值应该是
$-\infty$（若干个恰好为 $0$ 的项求和后取对数），所以这是一个真正的 bug，而不只是表示上的噪声，值得专门加一个
判断：在循环之前加上 `if m == -math.inf: return -math.inf`，对普通输入毫无影响（那时 `m` 是有限的），而当 `z`
的每一项都是 $-\infty$ 时（也是 `m` 本身能等于 $-\infty$ 的唯一情形）会返回正确答案。有了 `lse`，`softmax(z)_i`
就是 `math.exp(z_i - lse(z))`：因为 $\mathtt{lse}(z)=m+\log\sum_k e^{z_k-m}$，所以
$z_i-\mathtt{lse}(z) = (z_i-m)-\log\sum_k e^{z_k-m}$，取指数后得到
$e^{z_i-m}/\sum_k e^{z_k-m} = e^{z_i}/\sum_k e^{z_k}$（公共因子 $e^{-m}$ 在分子分母中恰好抵消）——用的正是 `lse`
内部已经算过的那个数值稳定的指数，直接复用，不必重新推导一遍。

**Q5.** `stats` 计算的是目前为止已经看到的值的运行中样本均值，以及*无偏*（unbiased）样本方差（把偏差平方和除以
$n-1$，而不是 $n$），每来一个新值就更新一次，且从不保存整个序列——这就是 Welford 算法。它之所以优先于教科书公式
$\mathrm{Var}(x)=E[x^2]-E[x]^2$（对 $x$ 与 $x^2$ 的累积和做单遍计算），是因为那个公式要相减两个数值本身很大的
量，而它们的差却未必大：在 $x=10^9+[4,7,13,16]$ 上，真实的样本方差恰好是 $30$（把每个值都平移同一个常数
$10^9$，不会改变方差所依赖的任何一对差值），`stats` 能以完整的 float64 精度给出这个值，但 $E[x^2]-E[x]^2$ 要先
把数据平方成量级为 $10^{18}$ 的数——在这个量级上，float64 大约 $15$ 到 $17$ 位的有效十进制数字已经分辨不出量级
为 $30$ 的差异（那里最后一位上的一个单位就是 $128$）——在 float64 下它恰好返回 $-128.0$：不只是不精确，而是
负数，这对方差来说是一个不可能的值。`stats` 从不对偏移量做平方：`m2` 累积的是相对运行均值的偏差之积，每个偏差
都是两个量级为 $10^9$ 的数之差，而 float64 在这个量级上能分辨到约 $10^{-7}$，所以无论公共偏移量有多大，每一项
都保持精确。`stats([5.0])`
会除以 `n - 1 = 0`，抛出 `ZeroDivisionError`——只有一个数据点时样本方差本来就没有定义，函数并不会去猜一个值。
`stats([])` 完全不会进入循环，于是返回 `(0.0, 0.0 / (0 - 1))`，也就是 `(0.0, -0.0)`：这看起来像是“没有数据”时
一个平淡无奇、看似合理的总结，而不像一个错误。空输入是更糟糕的那种失败：`ZeroDivisionError` 会立刻让调用者停下
来，而 `(0.0, -0.0)` 是悄无声息的，可能被带进后续的计算（比如对若干方差取平均，或者判断 `variance > 0`）却不被
察觉——而这恰恰是最容易在无意间出现的输入：批次里凑巧有一个空分组。合并两段数据各自三元组的公式，来自 Chan、
Golub 和 LeVeque：记 $n=n_a+n_b$、$\delta=\mathrm{mean}_b-\mathrm{mean}_a$，

$$\mathrm{mean} = \mathrm{mean}_a+\delta\,\frac{n_b}{n}, \qquad M_2 = M_{2,a}+M_{2,b}+\delta^2\,\frac{n_a n_b}{n},$$

当 $n_b=1$ 时（此时 $M_{2,b}=0$），它恰好化简成 `stats` 自身更新公式的一步——在分布式或分批统计里，这个公式用来
合并各自独立算出的统计量，而完全不需要真的把数据拼接起来。

### 注意力

**Q6.** `weights` 从一个 $(T,T)$ 的分数矩阵计算出因果（自回归）自注意力权重：结果里第 $i$ 行只在列 $j\le i$ 上有
非零权重（严格上三角部分，即 $j>i$，会在做 softmax 之前被掩去），且每一行的和都恰好是 $1$——是位置 $0$ 到 $i$ 上
的一个合法概率分布。假设再叠加一个掩码，使得整整一行都被完全掩去（比如第 $i$ 行本身就是一个填充（padding）位置
的查询，而键（key）侧的填充掩码把它对任何位置、包括它自己的注意力都排除掉了）。用 `-np.inf` 的话，这一行 `scores`
的每一项都变成 `-inf`；这一行的 `scores.max(...)` 也就是 `-inf`，于是每个被减去的指数都是
`-inf - (-inf) = nan`，整行都变成 `nan`。`-1e9` 则不会这样：这一行的每一项仍然是同一个有限值 $-10^9$，减去最大
值后处处得到 $0$，`exp(0)=1` 对全部 `T` 列都成立，于是这一行变成了*均匀*分布 $1/T$——不会崩溃，却悄悄把注意力
权重泄漏到了本该被完全排除的位置上，之后还得显式地把它清零（比如在注意力输出上再套用一次填充掩码），而不能指望
它自己消失。把 `scores` 转换成 `np.float16` 会从另一个方向破坏同一个掩码常数：float16 能表示的最大有限量级约为
$65504$，所以 `-1e9` 本身在转换时就会上溢为 `-inf`，即便代码里根本没出现过 `-inf`，也会悄悄重新引入前面说的那个
`nan` 故障。修法是改用该数组自己 dtype 的最小值 `np.finfo(scores.dtype).min` 来做掩码，而不是一个针对 float32
或 float64 调好的常数——量级大到足以压过任何真实分数，却总能在 `scores` 当前的 dtype 下被表示出来。`scores` 本身
的开销是 $T^2$ 个元素——每个注意力头各有一份这样的矩阵，随序列长度平方增长、与模型隐藏维度无关，是长序列下计算
注意力时主要的内存开销。

### 采样与几何

**Q7.** `sample` 计算的是从一个长度未知或无界的流中，均匀、无放回地抽取 $k$ 个元素——*水塘抽样*（reservoir
sampling，Algorithm R）：无论流的哪个前缀已经被看过，`res` 都保存着其中 $k$ 个元素，且这 $k$ 个元素中的每一个，
都同等地代表着目前为止看到的全部内容。证明 $n\ge k$ 个元素中的每一个，最终都以概率 $k/n$ 出现在水塘里，用对已
处理元素个数的归纳法：最开始的 $k$ 个元素以概率 $1$ 进入水塘。当元素 $i$（从 $0$ 开始计数，$i\ge k$）到达时，它
以概率 $k/(i+1)$ 被放入水塘——即 $j$ 在 $\{0,\dots,i\}$ 上均匀分布时 $P(j<k)$——一旦被放入，它会顶替 $k$ 个占用
槽位中均匀随机的一个（因为在 $j<k$ 的条件下，`j` 本身在 $\{0,\dots,k-1\}$ 上均匀分布）。假设归纳地，最开始的
$i$ 个元素里的每一个，此刻都以概率 $k/i$ 留在水塘里（在 $i=k$ 时成立，此时它就是 $k/k=1$）；固定其中一个这样的
元素，它熬过元素 $i$ 到达这一步的概率是

$$1-\underbrace{\frac{k}{i+1}}_{\text{元素 }i\text{ 被放入}}\cdot\underbrace{\frac1k}_{\text{且顶替了这个槽位}} = \frac{i}{i+1},$$

于是它留在水塘里的概率变成 $\frac{k}{i}\cdot\frac{i}{i+1}=\frac{k}{i+1}$——这正是把归纳假设中的 $i$ 换成 $i+1$
之后的下一步，而且它已经和元素 $i$ 自己刚刚算出的入选概率 $k/(i+1)$ 一致，所以归纳假设对全部 $i+1$ 个元素同时
成立。由归纳法，处理完任意 $i\ge k$ 个元素之后，这个公共概率都是 $k/i$；特别地，处理完整个长度为 $n$ 的流之后
就是 $k/n$：每个元素的概率都相同。（仅有相同的入选概率，并不能保证每个 $k$ 元子集都等可能；把同样的归纳法作用
在整个子集上，可以证明 $\binom nk$ 种可能的水塘内容中，每一种的概率都是 $1/\binom nk$。）如果把
`rng.randrange(i + 1)` 换成 `rng.randrange(i)`，元素 $i$ 自己的入选概率会变成 $k/i$，并且用完全相同的论证、把
式子里所有的 $i+1$ 换成 $i$，一个已在水塘里的元素熬过这一步的概率会变成 $(i-1)/i$，而不是 $i/(i+1)$。这些存活
因子连乘时逐项相消（telescoping）：初始填充中的一个元素最终留在水塘里的概率是
$\prod_{i=k}^{n-1}\frac{i-1}{i}=\frac{k-1}{n-1}$，而元素 $a\ge k$ 的概率是
$\frac{k}{a}\prod_{i=a+1}^{n-1}\frac{i-1}{i}=\frac{k}{a}\cdot\frac{a}{n-1}=\frac{k}{n-1}$，与它的到达下标 $a$
无关：这个 bug 并不是让越晚来的元素被越多地偏爱，而是让初始填充之后的所有元素，一视同仁地比初始填充本身更被偏爱，
样本也就不再是均匀的了。

**Q8.** `dist` 计算的是 `X` 各行与 `Y` 各行之间的 $n\times m$ 欧氏距离矩阵，来自恒等式
$\lVert x-y\rVert^2=\lVert x\rVert^2+\lVert y\rVert^2-2x^\top y$——`sq` 正是这个恒等式的右边，在每一对点上做了
广播。没有 `np.maximum` 的话，一个负的 `sq` 有时会被送进 `np.sqrt`，而它对负的实数输入返回的是 `nan`（并附带一
个警告），而不是抛出异常。这种情况会在 `X`、`Y` 中存在彼此确实很靠近的点（真实的平方距离接近 $0$）、而
$\lVert x\rVert^2$、$\lVert y\rVert^2$ 本身却很大时出现——最容易预见的情形是在 `dist(X, X)` 的对角线上：$x=y$
时真实值恰好是 $0$，但 $\lVert x\rVert^2$（通过 `(X ** 2).sum(1)` 算一次）和 $x^\top x$（在矩阵乘法
`X @ X.T` 内部、以一套不同的浮点运算顺序又算了一次）未必舍入到完全相同的比特，二者之差就可能仅仅因为舍入而落在
$0$ 的任意一侧。`np.maximum(sq, 0)` 只是防止了崩溃，并不能挽回已经损失掉的精度。真正的修法是在做平方之前先去掉
那个巨大的公共部分——比如把 `X`、`Y` 两者放在一起的均值当作参考点，从两个数组中都减掉它，再调用 `dist`，这样
恒等式就作用在量级普通的数上，而且保持了同样的 $O(nmd)$ 做法；如果只关心某一对具体的点，直接从差值
`x - y` 算 $\lVert x-y\rVert$，完全绕开这个展开后的恒等式，也就完全绕开了这种抵消。时间开销以矩阵乘法 $XY^\top$
为主，是 $O(nmd)$ 次标量乘加（两个平方范数向量加起来只需要 $O(nd)$ 和 $O(md)$）；内存开销方面，返回的矩阵本身
是 $O(nm)$，再加上两个输入本来就占用的 $O(nd+md)$。

<details>
<summary>验证代码（可运行）</summary>

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
