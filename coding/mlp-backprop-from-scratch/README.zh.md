# 从零实现反向传播

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 机器学习实现（NumPy） | ★★★☆☆ | 中等 | RS · RE · MLE · Intern | backpropagation, softmax-cross-entropy, gradient-check, initialisation, sgd-momentum | 3 个部分 / 45 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

两层网络把一批输入映射为类别得分。给定一批 $X \in \mathbb{R}^{B\times d}$，即 $B$ 个 $\mathbb{R}^d$ 中的样本，一个有 $m$ 个单元的隐藏层，以及 $C$ 个输出类别：

$$Z_1 = XW_1 + b_1, \qquad H = \mathrm{ReLU}(Z_1), \qquad S = HW_2 + b_2,$$

其中 $\mathrm{ReLU}(z) = \max(z, 0)$ 逐元素作用，$W_1 \in \mathbb{R}^{d\times m}$，$b_1 \in \mathbb{R}^m$，$W_2 \in \mathbb{R}^{m\times C}$，$b_2 \in \mathbb{R}^C$，$S \in \mathbb{R}^{B\times C}$ 的每一行是一个样本的*得分*（scores，也叫 *logits*）。参数收集在一个 `dict` 里，`{"W1": W1, "b1": b1, "W2": W2, "b2": b2}`，每个值都是形状如上的 `np.ndarray`。预测的类别概率是 $S$ 逐行做 *softmax*：$\mathrm{probs}_{i,c} = e^{S_{i,c}} / \sum_{c'=0}^{C-1} e^{S_{i,c'}}$；给定整数标签 $y \in \{0,\dots,C-1\}^B$，批次上的损失是平均*softmax 交叉熵*（cross-entropy）：

$$L = \frac{1}{B}\sum_{i=0}^{B-1} -\log \mathrm{probs}_{i,y_i}.$$

### Part 1 —— 前向和反向

```py
def forward_backward(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    """params: {"W1": (d, m), "b1": (m,), "W2": (m, C), "b2": (C,)}. X: (B, d). y: (B,) int labels in
    [0, C). Returns (L, grads); grads holds dL/dW1, dL/db1, dL/dW2, dL/db2, each the same shape as the
    corresponding entry of params."""
```

在写代码之前，先按下面的顺序把每一个梯度都用矩阵形式推导出来：先是 $\partial L/\partial S$，它有一个封闭形式

$$\frac{\partial L}{\partial S} = \frac{\mathrm{softmax}(S) - Y}{B}, \qquad Y \in \{0,1\}^{B\times C},\ \ Y_{i,c} = \mathbb{1}[c=y_i]$$

（$Y$ 是标签的独热（one-hot）矩阵）——然后按链式法则（chain rule）逐层往回推，依次是 $\partial L/\partial W_2$、$\partial L/\partial b_2$、$\partial L/\partial H$、$\partial L/\partial Z_1$、$\partial L/\partial W_1$、$\partial L/\partial b_1$。

举例，$B=2$、$d=2$、$m=2$、$C=2$：

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

### Part 2 —— 梯度检查

```py
def grad_check(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    """Returns the largest relative error, over every entry of every one of the four parameter arrays,
    between forward_backward's analytic gradient and a central finite difference of the same entry."""
```

对某一个标量参数项 $\theta$，固定 `params` 里的其它每一项不变，*中心差分*（central finite difference）是

$$g_n[\theta] = \frac{L(\theta+\epsilon) - L(\theta-\epsilon)}{2\epsilon},$$

其中 $L(\theta')$ 是把这一项设为 $\theta'$ 时 `forward_backward` 报告的损失。记 $g_a[\theta]$ 为 `forward_backward` 解析梯度里对应的那一项，这一项的*相对误差*（relative error）是

$$\mathrm{rel\_err}[\theta] = \frac{|g_a[\theta] - g_n[\theta]|}{\max\bigl(|g_a[\theta]| + |g_n[\theta]|,\ 10^{-12}\bigr)},$$

`grad_check` 返回 $\mathrm{rel\_err}[\theta]$ 在 `W1`、`b1`、`W2`、`b2` 每一项 $\theta$ 上的最大值。下面会用到一条经验法则：最大值在 $10^{-7}$ 以下算通过，在 $10^{-4}$ 以上就有理由怀疑代码有 bug。解释默认 `eps` 的选取（为什么不用 $10^{-8}$，或者 $10^{-2}$？）；解释为什么 `grad_check` 需要 `float64` 的参数和数据，而不是模型平时训练可能用的 `float32`；以及如何防范某个隐藏单元的预激活值恰好落在 $\mathrm{ReLU}$ 拐点 $Z_1=0$ 的 `eps` 范围之内——此时差分横跨了拐点两侧，不论 `forward_backward` 的代码对不对，这个差分本身都不可信。

### Part 3 —— 训练它，以及初始化为什么重要

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

`make_spirals` 生成的两条臂不是线性可分的：不存在一条直线能把所有类别 $0$ 的点分到一侧、所有类别 $1$ 的点分到另一侧——下面的验证代码会用一个线性分类器（逻辑回归）独立确认这一点：它在这份数据上的准确率远远够不到下面的目标，比随机猜好不了多少。用 `he_init` 把**训练**准确率（$S$ 的 argmax 与 $y$ 在训练所用的同一份 `X`、`y` 上对比）训练到至少 $95\%$，全程使用固定的随机种子。

论证 `he_init` 的方差 $\mathrm{Var}(W) = 2/\mathrm{fan\_in}$（$\mathrm{fan\_in}$ 指每个输出单元读取多少个输入：`W1` 是 $d$，因为 $m$ 个隐藏单元里的每一个都读取全部 $d$ 个输入特征；`W2` 是 $m$，因为 $C$ 个输出单元里的每一个都读取全部 $m$ 个隐藏特征），做法是推导它对一层激活的*二阶矩*（second moment）产生了什么影响，也就是 $\mathrm{E}[h^2]$，即使 $\mathrm{E}[h]\ne0$ 时依然有意义，这一点方差 $\mathrm E[(h-\mathrm E[h])^2]$ 做不到。对单独一层 ReLU，$h=\mathrm{ReLU}(xW)$（$x$ 的各分量有共同的二阶矩 $s$；$W$ 的 $\mathrm{fan\_in}\times\mathrm{fan\_out}$ 个分量独立同分布，均值 $0$，方差 $\sigma^2$，与 $x$ 独立），求 $\mathrm E[h^2]$ 关于 $s$、$\sigma^2$、$\mathrm{fan\_in}$ 的表达式，以及能让它重新等于 $s$ 的那个 $\sigma^2$，再用蒙特卡洛（Monte Carlo）确认：堆叠 $10$ 层这样的层，用这个 $\sigma^2$ 能让 $\mathrm E[h^2]$ 大致保持不变，而换成 $\sigma^2=1/\mathrm{fan\_in}$ 又会发生什么。

另外，把 `he_init` 换成 `zero_init`：从 Part 1 的梯度公式出发，推导在 $W_1=b_1=W_2=b_2=0$ 处 `forward_backward` 恰好会返回什么，以及此后 `train` 的每一步会对这四个参数数组各自做了什么。

## 参考解答

<details>
<summary>展开参考解答</summary>

写代码之前值得先确认：损失是批次上逐样本交叉熵的*均值*，不是求和——这里就是这样假设的，也正是它决定了下面 $\partial L/\partial S$ 里直接出现的那个 $1/B$，而不是把批大小当成学习率上一个可以自由调整的倍数；以及 `y` 存的是整数类别下标，不是预先做好的独热向量，所以 `forward_backward` 要在内部自己构造独热矩阵。

### Part 1

记 $S$ 的第 $(i,c)$ 项为 $s_{i,c}$，$p_{i,c}=\mathrm{probs}_{i,c}$。单个样本的损失是 $\ell_i = -\log p_{i,y_i} = -s_{i,y_i} + \log\sum_{c'} e^{s_{i,c'}}$（代入 $p_{i,y_i}$ 的定义再取对数）。对同一个样本的某个得分求导，

$$\frac{\partial \ell_i}{\partial s_{i,c}} = -\mathbb{1}[c=y_i] + \frac{e^{s_{i,c}}}{\sum_{c'}e^{s_{i,c'}}} = p_{i,c} - \mathbb{1}[c=y_i],$$

而 $\ell_i$ 不依赖任何其它样本的得分，所以 $i'\ne i$ 时 $\partial \ell_i/\partial s_{i',c}=0$。在批次上取平均，$\partial L/\partial s_{i,c} = \frac{1}{B}(p_{i,c}-\mathbb 1[c=y_i])$，写成矩阵形式，记 $Y$ 为独热标签矩阵，就是 $\partial L/\partial S = (P-Y)/B$，正是题目给出的公式。把这个矩阵记作 $dS$。

剩下的每一个梯度都可以从 $dS$ 出发，一层一层地用链式法则推出来。$S=HW_2+b_2$ 意味着 $s_{i,c}=\sum_j h_{i,j}w2_{j,c}+b2_c$，于是 $\partial s_{i,c}/\partial w2_{j,c'}$ 在 $c=c'$ 时等于 $h_{i,j}$，否则为 $0$（$w2_{j,c'}$ 只出现在 $S$ 的第 $c'$ 列里）：

$$\frac{\partial L}{\partial w2_{j,c}} = \sum_{i,c'} \frac{\partial L}{\partial s_{i,c'}}\frac{\partial s_{i,c'}}{\partial w2_{j,c}} = \sum_i (dS)_{i,c}\, h_{i,j} = (H^\top dS)_{j,c}, \qquad\text{所以}\quad \frac{\partial L}{\partial W_2} = H^\top dS.$$

同样的求和，把 $\partial s_{i,c}/\partial b2_c = 1$ 代进去，得到 $\partial L/\partial b_2 = \sum_i (dS)_{i,:}$，也就是 $dS$ 按列求和，即 `dS.sum(axis=0)`。对 $H$，$\partial s_{i,c}/\partial h_{i,j}=w2_{j,c}$（$H$ 的第 $i$ 行只会喂给 $S$ 的第 $i$ 行）：

$$\frac{\partial L}{\partial h_{i,j}} = \sum_c \frac{\partial L}{\partial s_{i,c}}\, w2_{j,c} = (dS\, W_2^\top)_{i,j}, \qquad\text{所以}\quad dH := \frac{\partial L}{\partial H} = dS\, W_2^\top.$$

$H=\mathrm{ReLU}(Z_1)$ 是逐元素的，$h_{i,j}=\max(z1_{i,j},0)$，导数是 $\partial h_{i,j}/\partial z1_{i,j}=\mathbb 1[z1_{i,j}>0]$（在拐点本身 $z1_{i,j}=0$ 处取次梯度 $0$——Part 2 会讲到为什么这一个点对梯度*检查*（checking）来说是个麻烦，但对反向传播（backpropagation）本身却不是），于是 $dZ_1 := \partial L/\partial Z_1 = dH \odot \mathbb 1[Z_1>0]$，一次逐元素乘法。最后，$Z_1=XW_1+b_1$ 与 $S=HW_2+b_2$ 形状完全一样，只是用 $(X,W_1,b_1,dZ_1)$ 替换了 $(H,W_2,b_2,dS)$，所以同样的两步推导给出 $\partial L/\partial W_1 = X^\top dZ_1$ 和 $\partial L/\partial b_1 = \sum_i (dZ_1)_{i,:}$。

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

在上面这个例子上，它返回 `loss ~= 0.0354`，$dS \approx \begin{pmatrix}-0.0315 & 0.0315\\ 0.0029 & -0.0029\end{pmatrix}$：第 $0$ 行把类别 $0$ 的得分往上拉、类别 $1$ 的得分往下压，因为真实标签是 $0$，而 $\mathrm{probs}_{0,0}=0.9370$ 还稍微低估了它；第 $1$ 行的拉力小得多，因为 $\mathrm{probs}_{1,1}=0.9942$ 已经很接近标签了。验证代码会用一个独立的有限差分循环，核对这个例子上 `W1`、`b1`、`W2`、`b2` 返回的每一个梯度。

前向传播花费 $O(Bdm)$ 算 $XW_1$，$O(BmC)$ 算 $HW_2$；反向传播把同样两个乘积转置后各算一遍——$H^\top dS$ 和 $dS\,W_2^\top$ 各是 $O(BmC)$，$X^\top dZ_1$ 是 $O(Bdm)$——所以 `forward_backward` 总共花费 $O(B(dm+mC))$ 时间，和前向传播本身同一个量级（这是反向传播的一条一般规律，追问里还会再提到）。内存是 $O(B(d+m+C))$，用来缓存 $X$、$Z_1$、$H$（反向传播这三个都要读），再加上参数本身占用的 $O(dm+mC)$。

### Part 2

$g_n$ 逼近 $g_a$ 时会带上两种误差：把 $L$ 的泰勒展开截断在有限阶产生的*截断误差*（truncation error），以及在有限精度算术里计算 $L$ 产生的*舍入误差*（rounding error）；`eps` 的选取就是在这两者之间取舍。把 $L$ 在 $\theta$ 附近展开，

$$L(\theta \pm \epsilon) = L(\theta) \pm \epsilon L'(\theta) + \frac{\epsilon^2}{2}L''(\theta) \pm \frac{\epsilon^3}{6}L'''(\theta) + O(\epsilon^4),$$

所以 $L(\theta+\epsilon)-L(\theta-\epsilon) = 2\epsilon L'(\theta) + \frac{\epsilon^3}{3}L'''(\theta) + O(\epsilon^5)$，再除以 $2\epsilon$，

$$g_n[\theta] = L'(\theta) + \frac{\epsilon^2}{6}L'''(\theta) + O(\epsilon^4)：$$

截断误差是 $O(\epsilon^2)$，单看这一项，`eps` 越小越好，这一项也正是为什么这里用中心差分、而不是单侧差分 $(L(\theta+\epsilon)-L(\theta))/\epsilon$ 的原因：单侧差分的截断误差只有 $O(\epsilon)$（展开式里的奇数阶项在那里不会相互抵消）。但 `float64` 算术表示 $L(\theta)$ 时带有大约*机器精度*（machine epsilon）$u\approx2.22\times10^{-16}$ 的相对误差，所以每次求值都带着大约 $u|L|$ 的绝对误差；用两个只相差 $O(\epsilon)$ 的值相减，会以大约 $u|L|/\epsilon$ 的速度把精度亏给这个误差——把 `eps` 减半大致会把截断项缩小到四分之一，却把这一项放大到两倍，所以总误差 $\frac{\epsilon^2}{6}|L'''| + \frac{u|L|}{\epsilon}$ 在两项相当时最小，即 $\epsilon^2 \sim u/\epsilon$，也就是 `float64` 下 $\epsilon \sim u^{1/3} \approx 6\times10^{-6}$，与默认值 $10^{-5}$ 同一个数量级。`float32` 的机器精度大约是 $1.19\times10^{-7}$，比 `float64` 大了九个数量级，最优的 $\epsilon$ 变成 $\epsilon \sim (1.19\times10^{-7})^{1/3}\approx4.9\times10^{-3}$：在那里用 $10^{-5}$ 就深深落进了舍入误差主导的区间，$u|L|/\epsilon$ 会把 `grad_check` 想要测的信号完全淹没。下面的验证代码会直接演示这一点，所以 `grad_check` 要求 `float64` 的参数和数据，对 `float32` 的输入宁可报错，也不悄悄返回一个会误导人的数字。

拐点是和精度无关的另一个问题。$Z_1=XW_1+b_1$ 本身就是 $W_1$、$b_1$ 的函数，所以扰动它们某一项 $\pm\epsilon$，会让 $Z_1$ 的某些项跟着偏移大约 $\epsilon$ 的量；如果某个隐藏单元的预激活值本来就在某个样本上落在 `eps` 以内挨着 $0$，这个偏移就可能让 $\mathrm{ReLU}$ 在 $L(\theta+\epsilon)$ 和 $L(\theta-\epsilon)$ 这两次求值之间从激活翻转到不激活（或者反过来），于是 $g_n$ 就不再逼近拐点两侧任何一段线性分支自己的导数，这是中心差分这个*方法*本身在那一点上的真实局限，不论 `forward_backward` 的代码写得对不对都存在。`grad_check` 直接用 Part 1 已经算出的那些预激活值检查这件事，一旦发生就报错，而不是返回一个可能被误认成 bug 证据的数字。

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

`grad_check` 对每一个参数项都要调用两次 `forward_backward`，一共 $dm+m+mC+C$ 项，所以花费 $O\bigl((dm+m+mC+C)\cdot B(dm+mC)\bigr)$ 时间，大致随网络规模的平方增长；它在每个被扰动的点上也重新算了一遍用不上的解析梯度，因为复用 `forward_backward`、而不是单写一个只算损失的版本，能让两条代码路径明显保持一致，这份多余的开销只是因为 `grad_check` 本来就是给这里用到的这种小网络设计的，从不会用在真正训练的模型上，才无关紧要。

### Part 3

**He 初始化。** 单独考虑一层 ReLU，$h=\mathrm{ReLU}(z)$，$z=xW$（偏置只会平移预激活值的分布，这里推导全程和 `he_init` 里都把它设为 $0$，所以先不管它），其中 $x\in\mathbb R^{\mathrm{fan\_in}}$ 的各分量共有一个二阶矩 $s=\mathrm E[x_j^2]$，$W$ 的 $\mathrm{fan\_in}\times\mathrm{fan\_out}$ 个分量独立同分布、取自一个关于 $0$ 对称的分布（`he_init` 用高斯分布）、方差为 $\sigma^2$，且与 $x$ 独立。对某个输出特征 $z_k=\sum_{j=1}^{\mathrm{fan\_in}} x_jW_{jk}$：因为 $\mathrm E[W_{jk}]=0$ 且 $W_{jk}$ 与 $x_j$ 独立，$\mathrm E[z_k]=\sum_j\mathrm E[x_j]\mathrm E[W_{jk}]=0$，不论 $x$ 自己的均值是多少，所以 $\mathrm{Var}(z_k)=\mathrm E[z_k^2]$。展开平方，

$$\mathrm E[z_k^2] = \sum_{j,j'}\mathrm E[x_jx_{j'}W_{jk}W_{j'k}];$$

当 $j\ne j'$ 时，$W_{jk}$ 与 $W_{j'k}$ 互相独立，也都与 $x_j,x_{j'}$ 独立，且 $\mathrm E[W_{jk}]=\mathrm E[W_{j'k}]=0$，所以这一项是 $\mathrm E[x_jx_{j'}]\,\mathrm E[W_{jk}]\,\mathrm E[W_{j'k}]=0$；当 $j=j'$ 时，$W_{jk}$ 与 $x_j$ 的独立性给出 $\mathrm E[x_j^2W_{jk}^2]=\mathrm E[x_j^2]\,\mathrm E[W_{jk}^2]=s\sigma^2$。于是 $\mathrm E[z_k^2]=\mathrm{fan\_in}\cdot s\sigma^2$，一个干净的、关于二阶矩的递推式，与 $x$ 具体的分布无关。

因为 $W_{jk}$ 的分布关于 $0$ 对称，且与 $x_j$ 独立，乘积 $x_jW_{jk}$ 也关于 $0$ 对称（它与 $x_j(-W_{jk})=-(x_jW_{jk})$ 同分布，因为 $-W_{jk}$ 与 $W_{jk}$ 同分布），而若干个各自关于 $0$ 对称的独立变量之和，本身也关于 $0$ 对称，所以 $z_k$ 恰好关于 $0$ 对称，不只是近似。对一个关于 $0$ 对称的分布，$\mathrm E[z^2\mathbb 1[z>0]]=\mathrm E[z^2\mathbb 1[z<0]]$（在期望里代入 $z\to-z$：$z^2$ 不变，$\mathbb 1[z>0]$ 变成 $\mathbb 1[z<0]$），两者相加等于 $\mathrm E[z^2]$（几乎必然不会恰好等于 $0$），所以各自都恰好是它的一半：

$$\mathrm E[h^2] = \mathrm E\bigl[\max(z,0)^2\bigr] = \mathrm E[z^2\mathbb 1[z>0]] = \tfrac12\mathrm E[z^2] = \tfrac12\,\mathrm{fan\_in}\cdot s\sigma^2.$$

令 $\mathrm E[h^2]=s$，也就是这一层保住了激活的二阶矩，解出 $\sigma^2=2/\mathrm{fan\_in}$：这正是 He 初始化。换成 $\sigma^2=1/\mathrm{fan\_in}$（这是能保住*线性*层输出方差的方差，忽略了 $\mathrm{ReLU}$ 带来的那个 $\tfrac12$ 因子），会得到 $\mathrm E[h^2]=s/2$：二阶矩在每一层都被砍掉一半，堆叠 $10$ 层之后大约只剩最初的 $2^{-10}\approx0.001$，下面用蒙特卡洛确认这一点。

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

**全零初始化。** 把 Part 1 的梯度公式代入 $W_1=b_1=W_2=b_2=0$ 这一点。$Z_1=XW_1+b_1=0$ 对每一行都成立，不论 $X$ 是什么，于是 $H=\mathrm{ReLU}(0)=0$ 也对每个隐藏单元、每个样本都成立，每个隐藏单元算的都是输入的同一个（恒为零的）函数，这正是“隐藏单元保持一致”最字面的意思。这个结论会一路传回 `forward_backward` 的每一步：$dW_2=H^\top dS=0$（零矩阵乘任何东西都是零，不论 $dS$ 算出来是什么），$dH=dS\,W_2^\top=0$（因为 $W_2=0$），$dZ_1=dH\odot\mathbb 1[Z_1>0]=0$（$\mathbb 1[0>0]$ 是 `False`），$dW_1=X^\top dZ_1=0$，$db_1=\sum_i(dZ_1)_{i,:}=0$。只有 $db_2=\sum_i(dS)_{i,:}$ 可能不为零，因为它从来不乘 $H$ 或 $W_2$。

用归纳法可以证明这在*每一步*训练里都成立，不只是第一步：如果第 $t$ 步有 $W_1^{(t)}=b_1^{(t)}=0$（在 $t=0$ 时由构造成立），上一段的结论就给出 $dW_1^{(t)}=db_1^{(t)}=dW_2^{(t)}=0$，不论 $W_2^{(t)}$、$b_2^{(t)}$ 是什么；于是动量自己的递推

$$v^{(t+1)}=\mathrm{momentum}\cdot v^{(t)}-\mathrm{lr}\cdot0=\mathrm{momentum}\cdot v^{(t)}$$

让 $v_{W_1},v_{b_1},v_{W_2}$ 一直停在它们的初始值 $\mathbf0$ 上，从而 $W_1^{(t+1)}=W_1^{(t)}+0=0$，$b_1$、$W_2$ 同理，这正是下一步的归纳假设。所以 `zero_init` 会让 `W1`、`b1`、`W2` 在整个训练过程中都精确地停在 $\mathbf0$，唯一还能动的参数是 `b2`，它会去拟合标签的边际分布：因为 $H\equiv0$，$S=HW_2+b_2=b_2$ 对每个样本都是同一组得分，与 $X$ 无关，所以 `b2` 得到的梯度只来自 $y$ 的类别频率，别无其它。不论架构、学习率、动量还是训练多少步，都打不破这种对称性，因为它是*梯度公式本身*在这一点上的性质，只有初始值本身能打破它。

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

动量会改变多大的学习率才算安全：对一个大致稳定的梯度 $g$，动量自己的递推
$v^{(t+1)}=\mathrm{momentum}\cdot v^{(t)}-\mathrm{lr}\cdot g$ 会收敛到不动点
$v_\infty=-\mathrm{lr}\cdot g/(1-\mathrm{momentum})$（解 $v_\infty=\mathrm{momentum}\cdot
v_\infty-\mathrm{lr}\cdot g$ 中的 $v_\infty$ 即得），所以在 $\mathrm{momentum}=0.9$ 时，真正起作用的有效步长是原始
`lr` 的十倍；`lr`$=0.15$（有效步长 $1.5$）是凭经验选出的，稳稳落在仍然收敛、而非开始震荡的范围之内。

用 `he_init(2, 64, 2, seed=1)`、`make_spirals(150, n_turns=1.5, noise=0.08, seed=0)`、`train(..., steps=4000, batch_size=32, lr=0.15, momentum=0.9, seed=2)`，训练准确率能到 $99\%$，比 $95\%$ 的目标高出不少；把 `he_init` 换成 `zero_init(2, 64, 2)`，其它都不变，`W1`、`b1`、`W2` 逐位精确等于它们初始的零，准确率恰好是 $50\%$，在这个类别完全均衡的数据集上，这就是随机水平，因为这个被冻结的网络对每一个输入都预测同一个类别。

### 追问

- **反向模式与前向模式的自动微分。** 对一个标量损失和 $P$ 个参数，反向模式（`forward_backward` 采用的方式）用一次反向传播就能算出完整的梯度，代价和前向传播本身差不多——$O(1)$ 次遍历，与 $P$ 无关——做法是把链式法则从输出往输入方向应用，从唯一一个标量 $\partial L/\partial L=1$ 开始累积。前向模式则相反，让一个方向导数跟着每一个中间值从输入往输出方向传播，每个参数（或者每个输入方向）都要单独跑一次完整的遍历，$O(P)$ 次；参数多、输出少时反向模式占优，正是这道题的场景，输入少、输出多时则是前向模式占优，比如沿着少数几个方向求雅可比向量积（Jacobian-vector product）。
- **内存与激活重计算（activation checkpointing）。** 反向传播需要前向传播产生的每一个中间激活——`_forward` 缓存了 `Z1`、`H`、`X`，正是因为 `forward_backward` 的梯度都要读它们，所以内存开销是网络深度乘以激活大小，还要加上参数本身。激活重计算用算力换内存：只保留一部分激活（比如每隔几层留一个），反向传播时从最近保留的那个重新算出中间缺的那些，把 $O(\text{depth})$ 的内存换成大约 $O(\sqrt{\text{depth}})$，代价是多跑一次局部的前向传播。
- **梯度随深度的消失与爆炸。** 反向传播穿过 $L$ 层堆叠的层时，大致是把 $L$ 个雅可比矩阵连乘起来；如果它们典型的尺度是 $\rho\ne1$，这个乘积就会按 $\rho^L$ 变化，$\rho$ 哪怕只比 $1$ 偏离一点点，$L$ 一大就会指数级地趋于 $0$ 或者爆炸。这和上面 He 初始化背后保二阶矩的论证是同一套道理，只是这里用在*反向*传播上，而不是蒙特卡洛验证的前向传播上：前向的二阶矩保住了，并不自动意味着反向梯度的量级也保住了，这也是为什么有些初始化方案是按反向传播来定的，或者取前向、反向两者的一个折中。
- **权重衰减（weight decay）。** 在损失里加上 $\frac{\lambda}{2}\lVert\theta\rVert^2$，会给每个参数的梯度都加上 $\lambda\theta$，逐元素、与数据无关，只需要在交给优化器之前多写一行，`dW1 = dW1 + lambda_ * W1`，`dW2` 同理。对普通 SGD 来说，这等价于直接收缩参数，$\theta\leftarrow(1-\eta\lambda)\theta-\eta\nabla_\theta L_{\text{data}}$；一旦用上动量或者 Adam，这两种写法就不再等价，而这正是*解耦*（decoupled）权重衰减（AdamW）要保住的区别。
- **批归一化（batch normalisation）的反向传播。** BatchNorm 在*整个批次*上对每个特征做居中和缩放，$\hat z_j=(z_j-\mu_j)/\sqrt{\sigma_j^2+\epsilon}$，其中 $\mu_j,\sigma_j^2$ 是用共享这个批次的每一个样本算出来的，和上面 $\mathrm{ReLU}$ 那种对每个样本各自独立作用的门不同，局部雅可比 $\partial\hat z_{i,j}/\partial z_{i',j}$ 对任意一对样本 $i,i'$ 都不为零（通过 $\mu_j$ 和 $\sigma_j^2$），所以某个样本在这一层的梯度，真的依赖于当前和它共享同一批次的其它每一个样本。

<details>
<summary>验证代码（可运行）</summary>

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
