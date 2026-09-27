# Focal loss 与交叉熵

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 推导与机器学习实现（NumPy） | ★★☆☆☆ | 中等 | MLE · RS · RE · Applied AI | focal-loss, cross-entropy, class-imbalance, numerical-stability, initialisation, gradients | 3 个部分 / 45 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

考虑二分类（binary classification）问题：每个样本有一个实值 logit $z \in \mathbb{R}$，预测概率为
$p = \sigma(z) = 1/(1+e^{-z})$（即 sigmoid 函数），标签为 $y \in \{0, 1\}$。记 $p_t$ 为模型分配给
*真实*类别的概率：若 $y=1$ 则 $p_t = p$，若 $y=0$ 则 $p_t = 1-p$。记 $\alpha_t$ 为一个与类别相关的权重，
其中 $\alpha \in (0, 1)$ 是一个固定常数：若 $y=1$ 则 $\alpha_t = \alpha$，若 $y=0$ 则
$\alpha_t = 1-\alpha$。二元交叉熵（binary cross-entropy）为 $\mathrm{CE} = -\log p_t$；*focal loss*
定义为

$$\mathrm{FL} = -\alpha_t\,(1-p_t)^\gamma\,\log p_t, \qquad \gamma \ge 0,$$

其中 $\gamma$ 是一个固定常数，称为*聚焦参数*（focusing parameter）。因子 $(1-p_t)^\gamma$ 是*调制因子*
（modulating factor）：当 $p_t$ 较小时（模型给真实类别的概率很低——一个*难*样本）它接近 $1$，当 $p_t$
接近 $1$ 时（一个*易*样本，模型已经自信且正确）它接近 $0$，因此它会把模型已经分类正确的样本的损失缩小。
取 $\gamma=0$ 会精确地退化为普通的、按 $\alpha_t$ 加权的交叉熵，因为对任意 $p_t \in (0, 1]$ 都有
$(1-p_t)^0 = 1$。

### Part 1 —— 梯度

记 $s = 2y - 1$，于是 $y=1$ 时 $s=+1$，$y=0$ 时 $s=-1$；这样 $p_t$ 就可以统一写成单个表达式
$p_t = \sigma(sz)$，对两种标签都适用。以闭式（closed form）推导 $\partial\,\mathrm{FL}/\partial z$，
把结果只写成 $\alpha_t$、$s$、$p_t$、$\gamma$ 的函数。用代数方法验证：把 $\gamma=0$ 代入你的结果，会
得到 $\alpha_t(p-y)$，即普通的、按 $\alpha_t$ 加权的交叉熵的梯度。然后，在 $\gamma=2$ 且 $\alpha_t=1$
的情况下（为了把调制因子本身对梯度的影响，与类别权重 $\alpha_t$ 的影响分开），计算比值
$\lvert\partial\,\mathrm{FL}/\partial z\rvert \big/ \lvert\partial\,\mathrm{CE}/\partial z\rvert$，
分别取一个*易*样本 $p_t = 0.9$ 和一个*难*样本 $p_t = 0.1$，并说明这两个比值说明了什么：focal loss 对
每个样本*梯度*的抑制程度，相对于它对*损失值*本身的抑制程度 $(1-p_t)^\gamma$，是强是弱。

### Part 2 —— 数值稳定的实现

```py
def focal_loss_and_grad(z: np.ndarray, y: np.ndarray, alpha: float = 0.25,
                         gamma: float = 2.0) -> tuple[float, np.ndarray]:
    """z, y: 1-D arrays of the same shape (N,), one logit and one label (0.0 or 1.0) per example.
    Returns (mean_loss, grad), where mean_loss is the batch mean of FL and grad has shape (N,),
    grad[i] = d(mean_loss)/dz[i]. Finite for every finite z, including |z| up to 100."""
```

实现 `focal_loss_and_grad`：先求 $N$ 个样本上 $\mathrm{FL}$ 的批量均值，再求这个均值关于 `z` 每一个
分量的梯度。对任意满足 $\lvert z\rvert \le 100$ 的 $z$、任意 $y \in \{0, 1\}$、任意 $\alpha \in (0, 1)$
和 $\gamma \ge 0$，结果都必须保持有限——不能是 `nan`，不能是 `inf`，也不能触发溢出警告。$\log p_t$ 和
$1-p_t$ 都要各自用自己的数值稳定公式计算，而不是从已经算好的 $p$ 做减法得到：
$\log p_t = -\mathrm{softplus}(-sz)$，其中 $\mathrm{softplus}(x) = \log(1+e^x)$；
$1-p_t = \sigma(-sz)$，用和 $p_t$ 同样稳定的方式计算。

取 $\alpha=0.25$、$\gamma=2.0$（函数的默认值），batch 中两个样本都标为 $y=1$，一个自信且正确
（$z = \ln 9 \approx 2.1972$，故 $p_t = 0.9$），一个自信但错误（$z = -\ln 9 \approx -2.1972$，故
$p_t = 0.1$）：

```text
z = [2.1972, -2.1972], y = [1, 1]
focal_loss_and_grad(z, y) -> (0.2333, [-0.0004, -0.1378])
```

### Part 3 —— 让 focal loss 在实践中真正有效

**(a) 初始化。** 用于检测稀有正类（positive class）的分类器——正类的先验占比为 $\pi$（例如
$\pi = 0.01$：$1\%$ 的样本为正）——通常会把送入最终 logit 的每一个权重都初始化为 $0$，只有偏置
（bias）例外，被设为 $b = \log\bigl(\pi/(1-\pi)\bigr)$ 而不是 $0$。当其余权重都是 $0$ 时，每个样本的
logit 都是 $z=b$，与它的特征无关。分别在 $b=0$ 和 $b=\log(\pi/(1-\pi))$ 两种情况下，计算一个正类占比
为 $\pi=0.01$ 的群体上的平均 focal loss（取 $\alpha=0.25$、$\gamma=2.0$），并计算负类（negative
class，也就是数量占多数的那一类）在每种情况下要为这个平均损失负责多大比例。根据这些数字说明：为什么
$b=0$ 时，最初的梯度步会被数量庞大、却很容易分类的负类样本主导；以及为什么用先验设定的偏置可以避免
这一点。

**(b) 梯度去了哪里。** 生成一个正类占比为 $1\%$ 的合成不平衡二分类数据集（正类和负类分别取自均值不同
的两个高斯簇），用普通（全批量）梯度下降拟合一个逻辑回归模型 $z = w\cdot x + b$——一次最小化平均交叉熵
（$\alpha=0.5,\gamma=0$），一次最小化平均 focal loss（$\alpha=0.25,\gamma=2$）——两次都从同样的初始化
开始（$w=0$，$b$ 按 (a) 中的方式设定）。在初始化时，以及训练结束后，分别测量*易*样本（定义为
$p_t > 0.9$ 的样本）在 $\sum_i \lvert\partial(\text{平均损失})/\partial z_i\rvert$（全部样本梯度幅值
之和）中占的比例。报告这四个数字（CE/FL，初始化时/训练后），并说明它们反映了每种损失函数把注意力放在
了哪些样本上。

**(c) 如何选择 $\gamma$ 和 $\alpha$。** 结合 (b) 中的数字，解释 $\gamma$ 控制的是什么、$\alpha$
控制的是什么，并说明为什么原论文选择的 $\alpha=0.25$——它给稀有正类的权重反而*小于*朴素的平衡取值
$\alpha=0.5$——并不违背纠正类别不平衡这个目标。

## 参考解答

<details>
<summary>展开参考解答</summary>

值得先和面试官确认：这里是二分类、单一 logit 的设定，而不是多分类（多分类的推广见追问部分）；以及
$\alpha$ 是直接给正类加权（$y=1$ 时 $\alpha_t=\alpha$），这是原论文采用的约定，而不是在每个 batch 里
动态地给数量较少的那一类加权。

### Part 1

利用 $p_t = \sigma(sz)$ 以及 $\sigma'(x) = \sigma(x)(1-\sigma(x))$，链式法则给出
$dp_t/dz = s\,p_t(1-p_t)$（额外的因子 $s$ 来自对 $sz$ 求导），于是

$$\frac{d\log p_t}{dz} = \frac{1}{p_t}\cdot s\,p_t(1-p_t) = s(1-p_t), \qquad
\frac{d(1-p_t)^\gamma}{dz} = \gamma(1-p_t)^{\gamma-1}\cdot\bigl(-s\,p_t(1-p_t)\bigr)
= -\gamma s\,p_t(1-p_t)^\gamma.$$

$\alpha_t$ 不依赖于 $z$，所以对 $\mathrm{FL} = -\alpha_t(1-p_t)^\gamma\log p_t$ 用乘积法则，得到

$$\frac{\partial\,\mathrm{FL}}{\partial z}
= -\alpha_t\left[\frac{d(1-p_t)^\gamma}{dz}\log p_t + (1-p_t)^\gamma\frac{d\log p_t}{dz}\right]
= -\alpha_t\Bigl[-\gamma s\,p_t(1-p_t)^\gamma\log p_t + s(1-p_t)^{\gamma+1}\Bigr],$$

从括号里的两项中提出公因子 $\alpha_t\,s\,(1-p_t)^\gamma$，得到

$$\frac{\partial\,\mathrm{FL}}{\partial z}
= \alpha_t\,s\,(1-p_t)^\gamma\,\bigl[\gamma\,p_t\log p_t - (1-p_t)\bigr].$$

**$\gamma=0$ 的情形。** $(1-p_t)^0=1$，括号变成 $-(1-p_t)$，所以
$\partial\,\mathrm{FL}/\partial z\rvert_{\gamma=0} = -\alpha_t\,s\,(1-p_t) = \alpha_t\,s\,(p_t-1)$。
对 $y=1$（$s=1,\,p_t=p$）而言，这是 $\alpha_t(p-1) = \alpha_t(p-y)$；对 $y=0$（$s=-1,\,p_t=1-p$）而言，
这是 $-\alpha_t\bigl((1-p)-1\bigr) = \alpha_t p = \alpha_t(p-y)$——两种情形得到的是同一个公式
$\alpha_t(p-y)$，这就确认了这个退化关系（并顺带说明 $s(p_t-1)=p-y$ 恒成立——这正是标准的 sigmoid
交叉熵梯度）。

**两个比值。** 取 $\alpha_t=1$，把 $\gamma=0$ 代入上式得到
$\lvert\partial\,\mathrm{CE}/\partial z\rvert = \lvert{-s(1-p_t)}\rvert = 1-p_t$，于是

$$\frac{\lvert\partial\,\mathrm{FL}/\partial z\rvert}{\lvert\partial\,\mathrm{CE}/\partial z\rvert}
= (1-p_t)^{\gamma-1}\,\bigl\lvert \gamma\,p_t\log p_t - (1-p_t) \bigr\rvert.$$

在 $\gamma=2$ 时：对易样本（$p_t=0.9$），这个比值是
$0.1 \times \lvert 2(0.9)\log(0.9) - 0.1\rvert \approx 0.1 \times 0.2896 \approx 0.0290$——focal 梯度
在这里大约比交叉熵梯度小 $34.5$ 倍。对难样本（$p_t=0.1$），比值是
$0.9 \times \lvert 2(0.1)\log(0.1) - 0.9\rvert \approx 0.9 \times 1.3605 \approx 1.2245$——focal 梯度
完全没有被抑制，反而比交叉熵梯度略*大*。这比单看损失值的比值 $(1-p_t)^\gamma$（分别是 $0.01$ 和
$0.81$）更为悬殊：对调制因子本身求导会多出一项，当 $p_t\to1$ 时它至少和 $(1-p_t)^\gamma$ 本身衰减得
一样快（所以易样本的梯度至少和它的损失值被抑制得一样强），但当 $p_t\to0$ 时它仍是 $1$ 阶的（所以难
样本的梯度几乎不受调制因子影响，不像它的损失值——虽然同样没怎么被压低，$(1-p_t)^\gamma=0.81$——终究
还是被压低了一点）。

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

复杂度：每个样本 $O(1)$——只是几步初等运算，与题目里其余部分无关。

### Part 2

把 $\log p_t$ 算成 `np.log(sigmoid(s * z))` 在这里恰恰是不安全的方向：当 $sz$ 非常负（自信但错误的
预测）时，$\sigma(sz)$ 本身可能已经因为舍入而丢失了它离 $0$ 有多远的信息；更重要的是，对同样关键的量
$1-p_t$ 而言，当 $sz$ 非常*正*（自信且正确的预测）时，一旦 $sz \gtrsim 37$，$\sigma(sz)$ 就会被舍入成
与 $1.0$ 无法区分的值，于是把 $1-p_t$ 算成 `1 - sigmoid(s * z)` 就变成了两个几乎相等的浮点数相减，
结果恰好是 $0$，而不是真正的、很小的正数——`(1 - p_t) ** gamma` 也就会对每一个自信且正确的样本都恰好
算出 $0$，这不是精度损失，而是彻头彻尾地算错了。恒等式
$\log\sigma(x) = -\log(1+e^{-x}) = -\mathrm{softplus}(-x)$ 避开了这个问题：`np.logaddexp(0, -x)`
直接计算 $\log(1+e^{-x})$，对任意有限的 $x$ 都稳定——因为它在内部会先减去两个参数中较大的那个再做指数
运算，所以 $x$ 很负时不会去算 $e^{-x}$，$x$ 很正时也不会溢出。$\log p_t$ 和 $1-p_t$ 于是都通过对 $sz$
和 $-sz$ 分别调用这同一个稳定的基本函数得到——各自在 $0.5$ 的自己那一侧都是精确的——而不是从对方推
出来。

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

复杂度：对一个 $N$ 个样本的 batch 是 $O(N)$ 时间和内存——一次向量化的遍历，没有对样本逐个进行的
Python 循环。

### Part 3

**(a) 初始化。** 当 $b=0$ 时，每个样本都有 $p=\sigma(0)=0.5$，于是不论标签是什么，$p_t=0.5$——此时
还没有哪个样本是*易*的，因为调制因子 $(1-p_t)^\gamma = 0.5^\gamma$ 对每个样本都一视同仁。平均损失
就是同一个按类别加权的值 $-\alpha_t\,0.5^\gamma\log(0.5)$ 在两个类别上的加权平均：

$$\overline{\mathrm{FL}}\big|_{b=0} = -0.5^\gamma\log(0.5)\,\bigl[\pi\alpha + (1-\pi)(1-\alpha)\bigr]
\approx 0.1291 \quad (\pi=0.01,\ \alpha=0.25,\ \gamma=2),$$

其中负类要为 $99.66\%$ 负责——这仅仅是因为它们占了总体的 $99\%$，而不是因为它们个体上比正类更难或
更易。用先验设定的偏置后，负类立刻得到 $p_t = 1-\pi = 0.99$（已经自信且正确，于是
$(1-p_t)^\gamma = \pi^\gamma = 0.0001$ 把它们压低了），而正类得到 $p_t=\pi=0.01$（和原来一样难）；
平均损失降到约 $0.01128$——降低了 $11.4$ 倍——负类在其中的占比也随之跌到 $0.0066\%$。没有先验偏置时，
最初的梯度步大多是在调整模型去拟合数量庞大的负类整体的*平均*状态（因为在 $z=0$ 时，还没有哪个负类
样本因为“易”而被压低权重，不论正负类都还没被自信地分类）；有了先验偏置，负类从一开始就已经很易，
于是最初的损失、进而最初的梯度，几乎全部来自稀有的正类——这正是一个检测器从第一步起就需要的信号。

**(b) 梯度去了哪里。** 两个模型训练后达到了相同的准确率：在两种损失函数下，都从 $30$ 个正样本中找回
了 $27$ 个，只有一个假阳性（false positive）。（用 scikit-learn 独立拟合一个逻辑回归，在同一份数据上
能找回 $30$ 个里的 $28$ 个——比两个从零训练的模型都多一个，这说明两次普通梯度下降都收敛到了线性模型
在这份数据上能达到的水平附近，而不是陷入了某个因 bug 而产生的局部最优。）所以这不是说 focal loss 在
这份数据上*分类*得更好——它只是测量了两种损失函数的梯度在训练过程中、总体上都来自哪里。在交叉熵下，
易样本占总梯度幅值的比例几乎没有变化：初始化时恰好是 $50.0\%$（此时每个样本的 logit 都是 $z=b$，处处
$p=\pi$，所以正类对 $\sum_i\lvert p_i-y_i\rvert$ 贡献 $(1-\pi)\times\pi N$，负类贡献
$\pi\times(1-\pi)N$——对*任意* $\pi$ 这两者都相等），训练后也只降到 $49.5\%$，即便这时已有 $99.4\%$
的样本个体上都是易样本——因为交叉熵的梯度对一个自信且正确的样本永远不会真正降到零，只是变小，而这样
的样本足够多，加起来又能追上少数类原本贡献的量。在 focal loss 下，易样本的占比始终接近零：初始化时是
$0.08\%$，训练后是 $3.4\%$——因为额外的 $(1-p_t)^\gamma$ 因子对每个易样本梯度的抑制，比它对损失值的
抑制（Part 1）猛烈得多，哪怕有成千上万个这样的样本，也压不过那约三十个仍然困难的样本。

**(c) 如何选择 $\gamma$ 和 $\alpha$。** (b) 中测量到的效果，几乎全部由 $\gamma$ 一个人就完成了：在
同样先验设定的初始化下，从普通交叉熵（$\alpha=0.5,\gamma=0$：负类占梯度幅值的 $50\%$）切换到只有
聚焦、不加任何类别权重（$\alpha=0.5,\gamma=2$），负类的占比就已经缩小到约 $0.028\%$——事实上比论文
自己采用的 $\alpha=0.25,\gamma=2$ 下的 $0.084\%$ 还要小。这是因为负类样本的 $\alpha_t$ 是 $1-\alpha$，
$\alpha$ 越是低于中性值 $0.5$，$1-\alpha$ 就越大：从 $\alpha=0.5$ 变到 $\alpha=0.25$，会把负类的
$\alpha_t$ 从 $0.5$ 提高到 $0.75$（同时把正类的 $\alpha_t$ 从 $0.5$ 降到 $0.25$），部分地把 $\gamma$
的聚焦机制几乎夺走的那部分梯度占比还给了负类——但远远没有恢复到交叉熵的 $50\%$。从这个意义上说，
$\alpha=0.25$ 并不是要在 $\gamma$ 已经做完聚焦之后，再给稀有的正类*更多*关注；恰恰相反，它是朝*另
一个*方向做的一点温和修正，防止 $\gamma$ 把困难的负类——检测器仍然要学会拒绝的那些假阳性——的梯度
几乎完全压没。

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

复杂度：`train_logreg` 花费 $O(\mathrm{steps}\cdot N\cdot d)$，主要是每一步的两次矩阵–向量乘法；
`population_shares` 和 `easy_share` 分别是 $O(n)$ 和 $O(N)$，各自只遍历一遍。

### 追问

- **多分类 focal loss。** 有 $C$ 个类别、对 logits 做 softmax 时，focal loss 推广为
  $\mathrm{FL} = -\alpha_c(1-p_c)^\gamma\log p_c$，其中 $p_c$ 是模型分配给真实类别 $c$ 的 softmax
  概率，$\alpha_c$ 是逐类别的权重；调制因子仍然只读取一个标量——分配给正确类别的概率——所以它本身
  不需要改动，但梯度现在要通过 softmax 的雅可比 $\partial p_c/\partial z_k = p_c(\delta_{ck}-p_k)$
  流向*每一个* logit，而不只是一个。
- **focal loss 与校准（calibration）。** 在准确率相近的情况下，用 focal loss 训练出的模型给出的概率
  往往*不如*用交叉熵训练的那样过度自信：因为调制因子会在 $p_t \to 1$ 时持续缩小损失（和梯度），所以
  没有太大的压力继续把一个已经预测正确的样本的概率推向边界；而恰恰在靠近决策边界、已经分类正确的样本
  上，交叉熵自身的梯度 $p-y$，在绝对值上仍然是最大的。
- **应对不平衡的其他办法，以及各自适用的场合。** 按类别频率的倒数重新加权（或者用它的一个缓和版本，
  *有效样本数*（effective number of samples））简单，也保留了每一个样本，但它对一个类别里的每个
  样本都一视同仁，不管这个样本本身有多难——不像 focal loss 那样*逐样本*地聚焦，它分不清一个真正困难
  的负样本和一个已经自信分类正确的负样本。重采样（对少数类过采样，或对多数类欠采样）可以配合任何损失
  函数使用而不需要改动，代价是要么在重复的样本上训练，要么丢掉多数类的数据。困难样本挖掘（hard-example
  mining）显式地挑出每个 batch 里损失最高的那些样本参与反向传播，是 focal loss 那种连续的、逐样本
  降权机制的一个离散的、只在单个 batch 内起作用的近亲，还额外带来挖掘这一步本身的开销和方差。
- **用平均精度（average precision）评估，而不是准确率。** 在 $\pi=0.01$ 时，一个永远预测负类的
  分类器就已经能拿到 $99\%$ 的准确率——这正是评估阶段的同一种病态，促使训练阶段要用 focal loss：
  准确率主要由数量占多数的那一类决定，几乎反映不出模型在稀有类别上的表现。平均精度（precision–recall
  曲线下的面积）或者在某个固定判定阈值下的召回率（recall），才是对真正的权衡——在负类中制造多少误报，
  去换取抓住多少正类——敏感的度量，一个恒定输出的预测器无法靠“默认”就赢得这样的指标。

<details>
<summary>验证代码（可运行）</summary>

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
