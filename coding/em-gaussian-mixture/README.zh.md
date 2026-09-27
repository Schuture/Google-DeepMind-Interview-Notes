# 高斯混合模型的 EM 算法：推导与实现

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 推导与机器学习实现（NumPy） | ★★☆☆☆ | 困难 | RS · RE · MLE | expectation-maximisation, gaussian-mixture, log-sum-exp, jensen-inequality, k-means, bic | 3 个部分 / 45–60 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

数据 $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$（等价于一个形状为 $(n, d)$ 的数组）被建模为独立同分布地采自
$K$ 个多元高斯分布组成的混合。分量 $k$（$1 \le k \le K$）有一个*权重*（weight）$\pi_k \ge 0$，满足
$\sum_{k=1}^K \pi_k = 1$；一个*均值*（mean）$\mu_k \in \mathbb{R}^d$；以及一个*协方差*（covariance）
$\Sigma_k \in \mathbb{R}^{d \times d}$——对称且正定（对任意 $v \ne 0$ 都有 $v^\top \Sigma_k v > 0$），因此
$\Sigma_k$ 可逆，$|\Sigma_k| > 0$——由此给出密度

$$\mathcal N(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac12 (x - \mu)^\top \Sigma^{-1} (x - \mu)\right), \qquad p(x) = \sum_{k=1}^K \pi_k\, \mathcal N(x; \mu_k, \Sigma_k).$$

记 $\theta = (\pi_{1:K}, \mu_{1:K}, \Sigma_{1:K})$ 为全部参数的集合。$X$ 在 $\theta$ 下的*对数似然*
（log-likelihood）是

$$\ell(\theta) = \sum_{i=1}^n \log \sum_{k=1}^K \pi_k\, \mathcal N(x_i; \mu_k, \Sigma_k).$$

举例，$K = 2$、$d = 1$（于是每个 $\Sigma_k$ 都是 $1 \times 1$ 矩阵 $[\sigma_k^2]$）、
$\pi = (0.5, 0.5)$、$\mu = (-1, 1)$、$\sigma_1^2 = \sigma_2^2 = 1$，$X = (-2, 0, 3)$：

```text
N(x; mu_k, sigma_k^2)，每列对应一个分量：
  x = -2:   0.2420   0.0044
  x =  0:   0.2420   0.2420
  x =  3:   0.0001   0.0540

p(x) = 0.5 * N(x; -1, 1) + 0.5 * N(x; 1, 1)：
  p(-2) = 0.1232   p(0) = 0.2420   p(3) = 0.0271

ell(theta) = log(0.1232) + log(0.2420) + log(0.0271) ~= -7.1225
```

下面每个部分推导或实现 *EM*（expectation-maximisation，期望最大化）的一块内容——这是最大化 $\ell(\theta)$
的标准算法，交替进行“估计每个点由哪个分量生成”和“用这个估计重新拟合各个分量”。

### Part 1 —— 推导 EM

为每个点引入一个隐变量（latent variable）$z_i \in \{1, \dots, K\}$，满足 $p(z_i = k) = \pi_k$ 和
$p(x_i \mid z_i = k) = \mathcal N(x_i; \mu_k, \Sigma_k)$，这样把 $z_i$ 边缘化掉就得到
$p(x_i) = \sum_k \pi_k \mathcal N(x_i; \mu_k, \Sigma_k)$，也就是上面的 $\ell(\theta) = \sum_i \log p(x_i)$。
固定一组当前估计 $\theta^{\mathrm{old}} = (\pi^{\mathrm{old}}, \mu^{\mathrm{old}}, \Sigma^{\mathrm{old}})$。

**(a)** 用贝叶斯公式（Bayes' rule）推导*响应度*（responsibility）
$\gamma_{ik} = p(z_i = k \mid x_i, \theta^{\mathrm{old}})$——在 $\theta^{\mathrm{old}}$ 下，点 $i$ 由分量
$k$ 生成的后验概率。

**(b)** *完全数据对数似然的期望*（expected complete-data log-likelihood）定义为

$$Q(\theta; \theta^{\mathrm{old}}) = \mathbb E_{z \mid X, \theta^{\mathrm{old}}}\bigl[\log p(X, Z \mid \theta)\bigr] = \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik} \bigl[\log \pi_k + \log \mathcal N(x_i; \mu_k, \Sigma_k)\bigr],$$

其中 (a) 里的 $\gamma_{ik}$ 保持固定。用一个拉格朗日乘子（Lagrange multiplier）处理约束
$\sum_k \pi_k = 1$，推导 $\arg\max_\pi Q$；再通过令相应梯度为零，分别推导 $\arg\max_{\mu_k} Q$ 和
$\arg\max_{\Sigma_k} Q$（每个 $k$ 各自独立求解）。

**(c)** 一次 *EM 迭代*取 $\theta^{\mathrm{old}}$，先用 (a) 算出 $\gamma$——这是 *E 步*（E-step）——再把
$\theta^{\mathrm{new}}$ 设为 (b) 求出的极大值点——这是 *M 步*（M-step）。证明对任意
$\theta^{\mathrm{old}}$ 都有 $\ell(\theta^{\mathrm{new}}) \ge \ell(\theta^{\mathrm{old}})$：对每个点上
$\{1, \dots, K\}$ 上的任意一个分布 $q_i$（$q_i(k) \ge 0$，$\sum_k q_i(k) = 1$），用 *Jensen 不等式*
（Jensen's inequality，对凹函数 $\varphi$ 和随机变量 $Y$ 有 $\varphi(\mathbb E[Y]) \ge \mathbb E[\varphi(Y)]$）
构造一个下界 $F(\theta, q) \le \ell(\theta)$，它对任意 $\theta$ 和任意 $q$ 都成立；证明当
$\theta = \theta^{\mathrm{old}}$、$q_i(k) = \gamma_{ik}$ 时这个下界恰好取等；并证明 M 步求出的 $Q$ 的极大值点，
同时也是 $F(\cdot, \gamma)$ 的极大值点。

### Part 2 —— 实现

```py
def fit_gmm(X: np.ndarray, K: int, n_iter: int = 200, tol: float = 1e-8, reg: float = 1e-6,
            seed: int = 0) -> dict:
    """X: (n, d). Fits a K-component full-covariance Gaussian mixture by EM. Returns a dict with
    weights (K,), means (K, d), covs (K, d, d), and log_likelihoods, a list holding ell(theta) at
    every E-step from the initial parameters through the returned ones (so log_likelihoods[-1] is
    ell at the returned weights/means/covs)."""
```

初始化时，所有随机性都用 `np.random.default_rng(seed)`：$K$ 个均值取 $X$ 的 $K$ 行不同的数据点，用
*k-means++* 选出——先从 $n$ 行里均匀随机抽出第一个，然后对 $k = 2, \dots, K$，从剩下的行里按“到已选出的最近均值的欧氏距离平方”成比例地抽出下一个；每个协方差都取
$\hat\Sigma + \mathrm{reg} \cdot I_d$，其中 $\hat\Sigma = \frac1n \sum_i (x_i - \bar x)(x_i - \bar x)^\top$ 是
$X$ 自己的协方差、$\bar x$ 是它的均值；每个权重都取 $1/K$。

每个 E 步都要在对数空间里计算：先对每个 $(i, k)$ 算出 $\log \pi_k + \log \mathcal N(x_i; \mu_k, \Sigma_k)$，
再用 *log-sum-exp* 恒等式——在做指数运算之前先减去每行的最大值——合并这 $K$ 个分量，而不是直接对每个
log 密度取指数，这样一个非常负的 log 密度就不会下溢成 $0$、导致后面的除法无法挽回。每次 M 步之后都要再给每个协方差加上
$\mathrm{reg} \cdot I_d$。依次跑 E 步、M 步、E 步……把每个 E 步的 $\ell(\theta)$ 追加进
`log_likelihoods`；一旦 $\ell$ 相对上一个 E 步的提升小于 `tol`（绝对值），或者已经跑了 `n_iter` 个 E 步——两者以先到的为准——就停止，且不再执行这一轮的
M 步。

接着上面的例子（$\theta^{\mathrm{old}}$ 就是那里给出的 $\pi, \mu, \Sigma$），Part 1(a) 的响应度和 Part
1(b) 更新后的参数是：

```text
gamma(x=-2) = (0.9820, 0.0180)        pi_new    = (0.4948, 0.5052)
gamma(x= 0) = (0.5000, 0.5000)        mu_new    = (-1.3180, 1.9509)
gamma(x= 3) = (0.0025, 0.9975)        Sigma_new = ([[0.9238]], [[2.1654]])   # reg 已经加上

ell(theta_old) ~= -7.1225，ell(theta_new) ~= -6.0408   # 确实提升了，正如 Part 1c 所证明的、恒成立
```

### Part 3 —— 两个联系

**(a)** 固定每个 $k$ 都共用同一个球形协方差 $\Sigma_k = \sigma^2 I_d$（各分量相等，且不被 M 步更新），并固定各个正的权重
$\pi_k$。对一个到最近均值*唯一*的点 $x$（$\mu_{k^\ast} = \arg\min_k \lVert x - \mu_k \rVert_2$，假设不存在恰好打平的情形），证明当
$\sigma \to 0^+$ 时 E 步的响应度 $\gamma_k(x) \to \mathbb 1[k = k^\ast]$，并且此时 M 步的均值更新收敛到
$\mu_k \to \frac{1}{|C_k|} \sum_{i \in C_k} x_i$，其中 $C_k = \{i : k^\ast(x_i) = k\}$——这正是
*k-means* 算法的质心更新：k-means 交替把每个点分配给离它最近的 $K$ 个质心之一、再把每个质心移到分配给它的那些点的均值处。

**(b)** 用*贝叶斯信息准则*（Bayesian information criterion）$\mathrm{BIC} = -2\ell + p \log n$ 实现模型选择，其中
$\ell$ 取一个已拟合模型最终的 `model["log_likelihoods"][-1]`，
$p = (K - 1) + Kd + Kd(d+1)/2$ 是自由参数的个数：权重占 $K - 1$ 个（$\sum_k \pi_k = 1$ 去掉一个自由度）、均值占
$Kd$ 个、$K$ 个对称的 $d \times d$ 协方差各有 $d(d+1)/2$ 个自由项，合计占 $Kd(d+1)/2$ 个。BIC
越低越好。

```py
def bic(X: np.ndarray, model: dict) -> float:
    """model: a dict as returned by fit_gmm, fitted on this same X. Returns the BIC score."""
```

举例，一个 $K = 2$、$d = 1$、$n = 100$、$\ell = -140.0$ 的模型，$p = 1 + 2 + 2 \cdot 1 \cdot 2 / 2 = 5$，
$\mathrm{BIC} = -2 \cdot (-140.0) + 5 \log(100) \approx 280 + 23.03 = 303.03$。

## 参考解答

<details>
<summary>展开参考解答</summary>

值得先和面试官确认两点：$\Sigma_k$ 是完整的协方差矩阵，而不是限制成对角矩阵——正如签名里
`covs: (K, d, d)` 所暗示的；以及要用哪种初始化和停止规则，这里固定为 k-means++ 均值、基于数据协方差的初始协方差，以及对
$\ell$ 的绝对容差。

### Part 1

**(a) 响应度。** 由贝叶斯公式，

$$\gamma_{ik} = p(z_i = k \mid x_i, \theta^{\mathrm{old}}) = \frac{p(z_i = k)\, p(x_i \mid z_i = k)}{p(x_i)} = \frac{\pi_k^{\mathrm{old}}\, \mathcal N(x_i; \mu_k^{\mathrm{old}}, \Sigma_k^{\mathrm{old}})}{\sum_{j=1}^K \pi_j^{\mathrm{old}}\, \mathcal N(x_i; \mu_j^{\mathrm{old}}, \Sigma_j^{\mathrm{old}})},$$

分母把 $z_i$ 的 $K$ 个取值边缘化掉，也就是 $p(x_i)$ 本身；因此对每个 $i$ 都有
$\sum_k \gamma_{ik} = 1$，因为 $\gamma_{i,\cdot}$ 是 $z_i$ 的 $K$ 个取值上的一个概率分布。

**(b) 权重。** $Q$ 里依赖 $\pi$ 的项是 $\sum_i \sum_k \gamma_{ik} \log \pi_k = \sum_k N_k \log \pi_k$，其中
$N_k = \sum_i \gamma_{ik}$ 是分量 $k$ 的*软计数*（soft count）（因为 $\log \pi_k$ 不依赖 $i$，可以把对 $i$
的求和挪到里面）。在约束 $\sum_k \pi_k = 1$ 下，用拉格朗日乘子 $\lambda$ 求极大：

$$\mathcal L(\pi, \lambda) = \sum_{k=1}^K N_k \log \pi_k + \lambda \Bigl(\sum_{k=1}^K \pi_k - 1\Bigr), \qquad \frac{\partial \mathcal L}{\partial \pi_k} = \frac{N_k}{\pi_k} + \lambda = 0 \ \Longrightarrow\ \pi_k = -\frac{N_k}{\lambda}.$$

对 $k$ 求和，用 $\sum_k \pi_k = 1$ 以及 $\sum_k N_k = \sum_k \sum_i \gamma_{ik} = \sum_i 1 = n$：
$1 = -\sum_k N_k / \lambda = -n/\lambda$，于是 $\lambda = -n$，$\pi_k = N_k / n$。

**(b) 均值。** $Q$ 里依赖 $\mu_k$ 的项是
$\sum_i \gamma_{ik} \log \mathcal N(x_i; \mu_k, \Sigma_k) = -\frac12 \sum_i \gamma_{ik} (x_i - \mu_k)^\top \Sigma_k^{-1} (x_i - \mu_k) + \text{常数}$
（归一化常数以及其它分量的项都不依赖 $\mu_k$）。求导并令其为零：

$$\frac{\partial Q}{\partial \mu_k} = \sum_i \gamma_{ik}\, \Sigma_k^{-1} (x_i - \mu_k) = 0 \ \Longrightarrow\ \sum_i \gamma_{ik}\, x_i = \Bigl(\sum_i \gamma_{ik}\Bigr) \mu_k \ \Longrightarrow\ \mu_k = \frac{\sum_i \gamma_{ik}\, x_i}{N_k},$$

两边同乘可逆的 $\Sigma_k$ 即得。

**(b) 协方差。** 改用精度矩阵 $\Lambda_k = \Sigma_k^{-1}$、而不是直接用 $\Sigma_k$ 来求导：用
$(x_i - \mu_k)^\top \Lambda_k (x_i - \mu_k) = \operatorname{tr}\bigl(\Lambda_k (x_i - \mu_k)(x_i - \mu_k)^\top\bigr)$
（一个标量等于它自己的 $1 \times 1$ 迹）以及迹的线性性，把对 $i$ 的求和挪到里面，$Q$ 里依赖
$\Lambda_k$ 的项就是

$$\sum_i \gamma_{ik} \Bigl[\tfrac12 \log |\Lambda_k| - \tfrac12 \operatorname{tr}\bigl(\Lambda_k (x_i - \mu_k)(x_i - \mu_k)^\top\bigr)\Bigr] = \frac{N_k}{2} \log |\Lambda_k| - \frac12 \operatorname{tr}(\Lambda_k S_k), \qquad S_k = \sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top.$$

用两个标准的矩阵求导恒等式即可收尾：对对称的 $\Lambda$，$\partial \log |\Lambda| / \partial \Lambda = \Lambda^{-1}$
（Jacobi 公式，$d|\Lambda| = |\Lambda| \operatorname{tr}(\Lambda^{-1} d\Lambda)$，逐项读出即得）；对对称的
$S$，$\partial \operatorname{tr}(\Lambda S) / \partial \Lambda = S$（$\Lambda$ 每一项都是线性的）。于是

$$\frac{\partial Q}{\partial \Lambda_k} = \frac{N_k}{2} \Lambda_k^{-1} - \frac12 S_k = 0 \ \Longrightarrow\ \Lambda_k^{-1} = \frac{S_k}{N_k} \ \Longrightarrow\ \Sigma_k = \frac{1}{N_k} \sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top,$$

因为按定义 $\Lambda_k^{-1} = \Sigma_k$。

**(c) 单调性。** 对每个点上 $\{1, \dots, K\}$ 上的任意一个分布 $q_i$，把 $\ell(\theta)$ 的第 $i$ 项写成
$q_i$ 下的期望，再对凹函数 $\log$ 用 Jensen 不等式：

$$\log p(x_i) = \log \sum_{k=1}^K q_i(k)\, \frac{\pi_k \mathcal N(x_i; \mu_k, \Sigma_k)}{q_i(k)} \ge \sum_{k=1}^K q_i(k)\, \log \frac{\pi_k \mathcal N(x_i; \mu_k, \Sigma_k)}{q_i(k)},$$

对任意 $\theta$ 和任意满足“只要 $\pi_k \mathcal N(x_i; \mu_k, \Sigma_k) > 0$ 就有 $q_i(k) > 0$”的 $q_i$
都成立。对 $i$ 求和，就得到对任意 $\theta$ 和任意 $q = (q_1, \dots, q_n)$ 都有
$\ell(\theta) \ge F(\theta, q)$，其中
$F(\theta, q) := \sum_i \sum_k q_i(k) \bigl[\log \pi_k + \log \mathcal N(x_i; \mu_k, \Sigma_k) - \log q_i(k)\bigr]$。

Jensen 不等式取等，当且仅当被平均的量 $\pi_k \mathcal N(x_i; \mu_k, \Sigma_k) / q_i(k)$ 对每个
$q_i(k) > 0$ 的 $k$ 都相同（一个凹函数作用在只取一个值的随机变量上，“凹性损失”为零）。记这个公共值为
$c_i$，则 $q_i(k) = \pi_k \mathcal N(x_i; \mu_k, \Sigma_k) / c_i$，再由 $\sum_k q_i(k) = 1$ 得
$c_i = p(x_i)$，即 $q_i(k) = \gamma_{ik}$（正是 (a) 里的响应度）。于是在
$\theta = \theta^{\mathrm{old}}$、$q = \gamma^{\mathrm{old}}$ 处，
$F(\theta^{\mathrm{old}}, \gamma^{\mathrm{old}}) = \ell(\theta^{\mathrm{old}})$ 恰好取等。

固定 $q = \gamma^{\mathrm{old}}$，
$F(\theta, \gamma^{\mathrm{old}}) = Q(\theta; \theta^{\mathrm{old}}) - \sum_i \sum_k \gamma_{ik}^{\mathrm{old}} \log \gamma_{ik}^{\mathrm{old}}$，第二项不依赖
$\theta$；所以对 $\theta$ 极大化 $F(\cdot, \gamma^{\mathrm{old}})$，正好就是对 $\theta$ 极大化
$Q(\cdot; \theta^{\mathrm{old}})$——也就是 (b) 里的 M 步。把三件事串起来：Jensen 不等式（在
$\theta^{\mathrm{new}}$ 处同样成立）、M 步求的是极大值点、以及在 $\theta^{\mathrm{old}}$ 处取等：

$$\ell(\theta^{\mathrm{new}}) \ \ge\ F(\theta^{\mathrm{new}}, \gamma^{\mathrm{old}}) \ \ge\ F(\theta^{\mathrm{old}}, \gamma^{\mathrm{old}}) \ =\ \ell(\theta^{\mathrm{old}}),$$

就证明了对任意 $\theta^{\mathrm{old}}$ 都有 $\ell(\theta^{\mathrm{new}}) \ge \ell(\theta^{\mathrm{old}})$：一次
EM 迭代永远不会让对数似然变小。

### Part 2

在对数空间里合并 $K$ 个分量，用的是和数值稳定 softmax 相同的恒等式：对任意不依赖被求和下标的常数 $m$，

$$\log \sum_{k} e^{a_k} = m + \log \sum_k e^{a_k - m},$$

因为对每个 $k$ 都有 $e^{a_k} = e^m e^{a_k - m}$，所以 $e^m$ 可以从求和里提出来，再和外层的
$\log(e^m \cdot (\cdot)) = m + \log(\cdot)$ 相消。取 $m = \max_k a_k$ 能让每个指数都 $\le 0$，这样即使某个原始的
$\log \pi_k + \log \mathcal N(x_i; \mu_k, \Sigma_k)$ 非常负，`np.exp` 也不会溢出。下面的
`make_gmm_clusters` 生成本页余下部分和 Part 3 都会用到的合成数据：从三个分离良好、共用一个协方差的二维高斯分布里各取
`n_per_cluster` 个点。

```python
import numpy as np


def make_gmm_clusters(n_per_cluster: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, true_means): X stacks n_per_cluster points drawn from each of 3 two-dimensional
    Gaussians with well-separated means and a shared covariance; true_means holds the 3 generating
    means, shape (3, 2)."""
    rng = np.random.default_rng(seed)
    true_means = np.array([[0.0, 0.0], [8.0, 0.5], [3.5, 7.0]])
    true_cov = np.array([[1.0, 0.3], [0.3, 0.8]])
    X = np.concatenate([rng.multivariate_normal(m, true_cov, size=n_per_cluster) for m in true_means])
    return X, true_means


def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: returns K distinct row indices into X."""
    n = X.shape[0]
    centers = [int(rng.integers(n))]
    dist_sq = np.sum((X - X[centers[0]]) ** 2, axis=1)
    for _ in range(K - 1):
        probs = dist_sq / dist_sq.sum()
        # NOTE: every already-chosen point has distance 0 to itself, hence probability 0 -- never re-picked
        next_idx = int(rng.choice(n, p=probs))
        centers.append(next_idx)
        dist_sq = np.minimum(dist_sq, np.sum((X - X[next_idx]) ** 2, axis=1))
    return np.array(centers)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    # NOTE: a row that is -inf everywhere (every component underflows to 0 density for that point) would
    # otherwise subtract -inf from -inf and give nan; exp(-inf - 0) == 0 leaves the correct value, 0.
    m = np.where(np.isneginf(m), 0.0, m)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _log_gaussian_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    diff = X - mean                                          # (n, d)
    sign, logdet = np.linalg.slogdet(cov)                    # NOTE: log-determinant via slogdet, never det(cov)
    solved = np.linalg.solve(cov, diff.T)                     # NOTE: solve, not an explicit matrix inverse
    mahalanobis = np.sum(diff.T * solved, axis=0)              # (n,): one (x-mu)^T Sigma^-1 (x-mu) per row
    return -0.5 * (d * np.log(2 * np.pi) + logdet + mahalanobis)


def _e_step(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, float]:
    n, K = X.shape[0], len(weights)
    log_joint = np.empty((n, K))
    for k in range(K):
        log_joint[:, k] = np.log(weights[k]) + _log_gaussian_pdf(X, means[k], covs[k])
    log_norm = _logsumexp(log_joint, axis=1)                   # (n,): log p(x_i) under the mixture
    resp = np.exp(log_joint - log_norm[:, None])                # (n, K): gamma_ik
    return resp, float(log_norm.sum())


def _m_step(X: np.ndarray, resp: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    K = resp.shape[1]
    Nk = resp.sum(axis=0)                                       # (K,)
    weights = Nk / n
    means = (resp.T @ X) / Nk[:, None]                           # (K, d)
    covs = np.empty((K, d, d))
    for k in range(K):
        diff = X - means[k]
        covs[k] = (resp[:, k, None] * diff).T @ diff / Nk[k]
        covs[k] += reg * np.eye(d)                                # NOTE: guards a collapsing component's Sigma_k
    return weights, means, covs


def fit_gmm(X: np.ndarray, K: int, n_iter: int = 200, tol: float = 1e-8, reg: float = 1e-6,
            seed: int = 0) -> dict:
    n, d = X.shape
    rng = np.random.default_rng(seed)
    means = X[_kmeans_pp_init(X, K, rng)].copy()                 # (K, d): K distinct data points
    weights = np.full(K, 1.0 / K)
    base_cov = np.cov(X, rowvar=False, ddof=0).reshape(d, d) + reg * np.eye(d)
    covs = np.stack([base_cov.copy() for _ in range(K)])

    log_likelihoods: list = []
    prev_ll = -np.inf
    for iteration in range(n_iter):
        resp, ll = _e_step(X, weights, means, covs)
        log_likelihoods.append(ll)
        # NOTE: stop BEFORE this iteration's M-step, so the returned parameters always match
        # log_likelihoods[-1] exactly -- never one M-step ahead of the last logged value.
        if ll - prev_ll < tol or iteration == n_iter - 1:
            break
        prev_ll = ll
        weights, means, covs = _m_step(X, resp, reg)

    return {"weights": weights, "means": means, "covs": covs, "log_likelihoods": log_likelihoods}
```

**复杂度。** k-means++ 初始化花费 $O(nKd)$：第一个之后的每一轮都要给剩下的每个点算一次距离平方，
$O(nd)$，一共 $K - 1$ 轮。每个 E 步都要对每个分量调用一次 `_log_gaussian_pdf`：为 `slogdet`/`solve`
分解 $\Sigma_k$ 花费 $O(d^3)$，为全部 $n$ 个点求解马氏距离项花费 $O(nd^2)$，所以每个 E 步是
$O(K(d^3 + nd^2))$。每个 M 步的 $K$ 次协方差更新各花费 $O(nd^2)$，合计 $O(Knd^2)$，超过均值更新的
$O(nKd)$。最多跑 `n_iter` 轮下来，`fit_gmm` 花费 $O(\mathrm{n\_iter} \cdot K(nd^2 + d^3))$——对
$n$ 和 $K$ 是线性的，通过每个分量自己的线性代数运算对 $d$ 是三次方的。

在 `X, true_means = make_gmm_clusters(seed=0)`（600 个点，3 个真实簇）上，`fit_gmm(X, 3, seed=0)` 跑
6 轮就收敛，恢复出的 `true_means` 误差在 $0.08$ 以内（按 3 个分量的某个排列匹配之后）。把同样的初始权重、均值、精度矩阵交给
`sklearn.mixture.GaussianMixture(covariance_type="full", weights_init=..., means_init=..., precisions_init=...,
reg_covar=reg)`，两者收敛到的每样本平均对数似然相差不超过 $10^{-6}$：sklearn 报告的是每样本的*平均*对数似然（它的
`lower_bound_`），而上面的 `log_likelihoods[-1]` 是*总*对数似然，所以比较之前要先除以 $n$。

### Part 3

**(a)** 当每个 $k$ 都取 $\Sigma_k = \sigma^2 I_d$ 时，
$\log \mathcal N(x; \mu_k, \sigma^2 I_d) = -\frac{d}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \lVert x - \mu_k \rVert^2$，
于是一个点 $x$ 在分量 $k$ 上的响应度是

$$\gamma_k(x) = \frac{\pi_k \exp\bigl(-\lVert x - \mu_k \rVert^2 / 2\sigma^2\bigr)}{\sum_j \pi_j \exp\bigl(-\lVert x - \mu_j \rVert^2 / 2\sigma^2\bigr)},$$

$-\frac{d}{2}\log(2\pi\sigma^2)$ 这一项在分子分母之间相消（对每个 $k$ 都一样）。设
$k^\ast = \arg\min_k \lVert x - \mu_k \rVert^2$（按假设唯一），把每一项都除以
$\exp\bigl(-\lVert x - \mu_{k^\ast} \rVert^2 / 2\sigma^2\bigr)$：

$$\gamma_k(x) = \frac{\pi_k \exp\Bigl(-\bigl(\lVert x - \mu_k \rVert^2 - \lVert x - \mu_{k^\ast} \rVert^2\bigr) / 2\sigma^2\Bigr)}{\sum_j \pi_j \exp\Bigl(-\bigl(\lVert x - \mu_j \rVert^2 - \lVert x - \mu_{k^\ast} \rVert^2\bigr) / 2\sigma^2\Bigr)}.$$

对 $k \ne k^\ast$，$\lVert x - \mu_k \rVert^2 - \lVert x - \mu_{k^\ast} \rVert^2 > 0$ 严格成立（$k^\ast$
唯一），所以这一项的指数在 $\sigma \to 0^+$ 时 $\to -\infty$，这一项本身 $\to 0$；而 $k = k^\ast$ 那一项在任意
$\sigma$ 下都恰好等于 $\pi_{k^\ast}$。每个 $\pi_k > 0$ 都固定不变、不依赖 $\sigma$，跟不上一个正在趋于
$0$ 的指数项的步伐；因此 $\gamma_{k^\ast}(x) \to \pi_{k^\ast} / \pi_{k^\ast} = 1$，而对 $k \ne k^\ast$ 有
$\gamma_k(x) \to 0$：$\gamma(x) \to \mathbb 1[k = k^\ast]$，成为一个 one-hot 向量。

当每个点的 $\gamma$ 在极限下都是 one-hot 时，M 步的均值更新
$\mu_k = \sum_i \gamma_{ik} x_i / \sum_i \gamma_{ik}$ 就变成只对满足 $k^\ast(x_i) = k$ 的那些点求和、再除以它们的个数：
$\mu_k \to \frac{1}{|C_k|} \sum_{i \in C_k} x_i$，$C_k = \{i : k^\ast(x_i) = k\}$——也就是离
$\mu_k$ 最近的那些点的均值，正是 k-means 的质心更新，只是把软响应度换成了按最近距离的硬分配。恰好等距的点（同时离两个或更多均值一样近）正是这个论证失效的地方：此时
$\lVert x - \mu_k \rVert^2 - \lVert x - \mu_{k^\ast} \rVert^2 = 0$ 对不止一个 $k$ 成立，$\gamma$ 在任意
$\sigma$ 下都在它们之间保持不变，永远到不了 one-hot——就像下表最后一列那样。

```text
均值为 (0, 0) 和 (3, 0)，权重为 (0.5, 0.5)，Sigma_k = sigma^2 * I_2

               x = (1, 0)                x = (2.5, 0.5)            x = (1.5, 0)，等距：与两个均值都相距 1.5
sigma = 1.0    gamma = (0.8176, 0.1824)  gamma = (0.0474, 0.9526)  gamma = (0.5000, 0.5000)
sigma = 0.5    gamma = (0.9975, 0.0025)  gamma = (0.0000, 1.0000)  gamma = (0.5000, 0.5000)
sigma = 0.1    gamma = (1.0000, 0.0000)  gamma = (0.0000, 1.0000)  gamma = (0.5000, 0.5000)
```

**(b)** 分量更多的混合模型拟合训练数据的效果只会更好或者不变，所以单看 $\ell$ 永远会偏向允许的最大
$K$；BIC 用 $p \log n$ 来惩罚这种收益，也就是多出来的那些参数的代价。根据 Part 2 的循环不变式，
`model["log_likelihoods"][-1]` 恰好就是 `fit_gmm` 返回的那组参数处的 $\ell$。

```python
def bic(X: np.ndarray, model: dict) -> float:
    n, d = X.shape
    K = model["weights"].shape[0]
    p = (K - 1) + K * d + K * d * (d + 1) // 2
    ell = model["log_likelihoods"][-1]
    return -2 * ell + p * np.log(n)
```

在 `make_gmm_clusters(seed=0)` 的 600 个点上：

```text
K   :  1       2       3       4       5       6
BIC :  6335.6  5132.2  4566.8  4597.0  4627.7  4651.3
```

BIC 在 $K = 3$——真实的簇数——处取得最小值，尽管 $\ell$ 本身在 $K$ 超过 3 之后还在小幅提升：每多一个分量带来的
$p \log n$ 惩罚——也就是那 $d + d(d+1)/2$ 个额外参数的代价——在这里超过了那点提升。

### 追问

- **奇异似然。** 不加正则化，$\ell(\theta)$ 没有有限的最大值：让某个分量的均值恰好落在一个数据点上，
  $\mu_k = x_i$，同时把它的协方差缩小成 $\Sigma_k = \varepsilon I_d$；那么
  $\mathcal N(x_i; \mu_k, \Sigma_k) = (2\pi\varepsilon)^{-d/2}$，当 $\varepsilon \to 0^+$ 时趋于
  $\infty$，$\ell(\theta)$ 也随之趋于 $\infty$（其余 $n - 1$ 个点仍然只贡献一个有限、有界的量）。每次
  M 步之后加到每个 $\Sigma_k$ 上的 `reg` 项，把每个 $\Sigma_k$ 的每个特征值都下限在 `reg`，排除了这种精确的坍缩，让
  $\ell$ 在整条轨迹上都保持有界。
- **一般缺失数据下的 EM。** 高斯混合只是一个通用模板的一个实例：对任何观测数据为 $X$、未观测量为 $Z$
  的模型，Part 1(c) 的论证——Jensen 不等式、在 E 步的后验处取等、M 步极大化同一个下界——都说明
  $\ell(\theta) = \log p(X \mid \theta)$ 每一轮都会改善，不论 $Z$ 具体是什么，包括 $X$ 本身真正缺失的条目，而不只是一个辅助的簇标签；此时
  E 步变成对缺失部分求 $p(Z \mid X, \theta^{\mathrm{old}})$，M 步极大化相应的 $Q$。
- **变分推断是它的推广。** Part 1(c) 的下界 $F(\theta, q) \le \ell(\theta)$ 对*任意*分布 $q$ 都成立，EM
  在每个 E 步都选了其中最好的那一个——精确的后验 $q_i = \gamma_i$——这是因为这里的后验有闭式解（对
  $K$ 个分量用贝叶斯公式）。当精确的后验难以计算时，变分推断改为把 $q$ 限制在一个可处理的族里（比如对未观测变量做因子分解的族），并同时对
  $\theta$ 和 $q$ 极大化同一个下界，这时它被称为*证据下界*（evidence lower bound，ELBO）；EM 正是
  $q$ 不受限制、E 步能精确达到这个下界的特殊情形。
- **局部最优与重启。** $\ell(\theta)$ 不是凹函数，所以 Part 1(c) 只说明 EM 不会让 $\ell$*变小*，从不说明它能到达全局最大值；不同
  `seed` 下的 `fit_gmm` 可能收敛到不同的不动点，尤其是在簇分离得不够好的时候。标准的缓解办法是跑几个不同的种子，保留最终
  `log_likelihoods[-1]` 最大的那次拟合——这始终是在*固定*的 $K$ 下重启、挑出最好的拟合；BIC
  再用每个 $K$ 各自最好的那次重启，去比较不同的 $K$。
- **对角协方差与完整协方差。** 把每个 $\Sigma_k$ 限制成对角矩阵，能把每个分量的 $O(d^3)$
  `slogdet`/`solve` 换成 $O(d)$ 的逐元素运算（对数方差求和，用除法代替线性求解），M 步也只需要
  $d$ 个逐特征的方差，$O(nd)$ 而不是 $O(nd^2)$——当各特征接近独立时是个划算的取舍，代价是一个轴对齐的协方差，无法拟合两个特征之间的相关性，除非用好几个旋转过的、看起来是对角的分量去凑出原本一个完整协方差的分量就能直接拟合的效果。

<details>
<summary>验证代码（可运行）</summary>

```python
def loop_e_step(X, weights, means, covs):                          # Part 1, straight from the formula
    n, d = X.shape
    K = len(weights)
    joint = np.zeros((n, K))
    for i in range(n):
        for k in range(K):
            diff = X[i] - means[k]
            inv = np.linalg.inv(covs[k])
            det = np.linalg.det(covs[k])
            density = np.exp(-0.5 * diff @ inv @ diff) / np.sqrt((2 * np.pi) ** d * det)
            joint[i, k] = weights[k] * density
    resp = joint / joint.sum(axis=1, keepdims=True)
    ll = float(np.sum(np.log(joint.sum(axis=1))))
    return resp, ll


def loop_m_step(X, resp, reg):
    n, d = X.shape
    K = resp.shape[1]
    Nk = np.array([sum(resp[i, k] for i in range(n)) for k in range(K)])
    weights = Nk / n
    means = np.zeros((K, d))
    for k in range(K):
        means[k] = sum((resp[i, k] * X[i] for i in range(n)), np.zeros(d)) / Nk[k]
    covs = np.zeros((K, d, d))
    for k in range(K):
        s = np.zeros((d, d))
        for i in range(n):
            diff = (X[i] - means[k]).reshape(-1, 1)
            s += resp[i, k] * (diff @ diff.T)
        covs[k] = s / Nk[k] + reg * np.eye(d)
    return weights, means, covs


# --- the worked example of the statement: intro densities/ell, then one E-step + one M-step ---
X0 = np.array([[-2.0], [0.0], [3.0]])
weights0 = np.array([0.5, 0.5])
means0 = np.array([[-1.0], [1.0]])
covs0 = np.array([[[1.0]], [[1.0]]])

N0 = np.array([[np.exp(_log_gaussian_pdf(X0[i:i + 1], means0[k], covs0[k])[0]) for k in range(2)] for i in range(3)])
assert np.allclose(np.round(N0, 4), [[0.2420, 0.0044], [0.2420, 0.2420], [0.0001, 0.0540]], atol=5e-5)
p0 = N0 @ weights0
assert np.allclose(np.round(p0, 4), [0.1232, 0.2420, 0.0271])
ell0 = float(np.sum(np.log(p0)))
assert round(ell0, 4) == -7.1225

resp0, ll0 = _e_step(X0, weights0, means0, covs0)
assert abs(ll0 - ell0) < 1e-9                                        # the E-step's ell matches the forward computation
assert np.allclose(np.round(resp0, 4), [[0.9820, 0.0180], [0.5000, 0.5000], [0.0025, 0.9975]], atol=5e-5)

w1, m1, c1 = _m_step(X0, resp0, 1e-6)
assert np.allclose(np.round(w1, 4), [0.4948, 0.5052])
assert np.allclose(np.round(m1.ravel(), 4), [-1.3180, 1.9509])
assert np.allclose(np.round(c1.ravel(), 4), [0.9238, 2.1654])
_, ll1 = _e_step(X0, w1, m1, c1)
assert round(ll1, 4) == -6.0408
assert ll1 > ll0                                                     # Part 1c's claim, on this example
print("worked example OK")

# --- brute-force differential test: an independent loop implementation, straight from the formulas ---
rng = np.random.default_rng(3)
for _ in range(200):
    n = int(rng.integers(3, 9))
    d = int(rng.integers(1, 4))
    K = int(rng.integers(1, 4))
    X = rng.normal(size=(n, d)) * 2
    weights = rng.dirichlet(np.ones(K))
    means = rng.normal(size=(K, d)) * 2
    A = rng.normal(size=(K, d, d))
    covs = np.einsum('kij,kaj->kia', A, A) + 0.5 * np.eye(d)[None]   # a random SPD matrix per component

    resp_ref, ll_ref = loop_e_step(X, weights, means, covs)
    resp_ours, ll_ours = _e_step(X, weights, means, covs)
    assert np.allclose(resp_ref, resp_ours, atol=1e-7)
    assert abs(ll_ref - ll_ours) < 1e-5
    assert np.allclose(resp_ours.sum(axis=1), 1.0, atol=1e-10)        # responsibilities always sum to 1

    w_ref, m_ref, c_ref = loop_m_step(X, resp_ours, 1e-6)
    w_ours, m_ours, c_ours = _m_step(X, resp_ours, 1e-6)
    assert np.allclose(w_ref, w_ours, atol=1e-7)
    assert np.allclose(m_ref, m_ours, atol=1e-7)
    assert np.allclose(c_ref, c_ours, atol=1e-7)
print("brute-force differential OK")

# --- on synthetic data: log-likelihood monotone, means recovered up to permutation, vs. sklearn ---
X, true_means = make_gmm_clusters(seed=0)
model = fit_gmm(X, 3, seed=0)
lls = model["log_likelihoods"]
assert len(lls) == 6                                                   # "converges in 6 iterations" above
assert all(lls[i + 1] >= lls[i] - 1e-9 for i in range(len(lls) - 1))
resp_final, _ = _e_step(X, model["weights"], model["means"], model["covs"])
assert np.allclose(resp_final.sum(axis=1), 1.0, atol=1e-10)

from itertools import permutations

best_err = min(np.max(np.abs(model["means"][list(perm)] - true_means)) for perm in permutations(range(3)))
assert round(best_err, 2) == 0.08                                      # "within 0.08" above
print("monotone log-likelihood + mean recovery OK, best_err =", round(best_err, 4))

from sklearn.mixture import GaussianMixture

K, reg = 3, 1e-6
rng0 = np.random.default_rng(0)                                       # NOTE: same seed fit_gmm(..., seed=0) uses,
means_init = X[_kmeans_pp_init(X, K, rng0)].copy()                     # and the same first call it makes with it,
weights_init = np.full(K, 1.0 / K)                                     # so this reproduces fit_gmm's own init exactly
d = X.shape[1]
base_cov = np.cov(X, rowvar=False, ddof=0).reshape(d, d) + reg * np.eye(d)
precisions_init = np.stack([np.linalg.inv(base_cov)] * K)

ours = fit_gmm(X, K, n_iter=300, tol=1e-12, reg=reg, seed=0)
n = X.shape[0]
ours_avg_ll = ours["log_likelihoods"][-1] / n                          # NOTE: sklearn reports the MEAN per-sample
                                                                          # log-likelihood, ours the TOTAL -- divide first
gmm = GaussianMixture(n_components=K, covariance_type="full", max_iter=300, tol=1e-12,
                       reg_covar=reg, weights_init=weights_init, means_init=means_init,
                       precisions_init=precisions_init, n_init=1)
gmm.fit(X)
assert abs(ours_avg_ll - gmm.lower_bound_) < 1e-6
assert np.allclose(np.sort(ours["means"], axis=0), np.sort(gmm.means_, axis=0), atol=1e-4)
print("sklearn agreement OK, |avg log-likelihood diff| =", abs(ours_avg_ll - gmm.lower_bound_))

# --- Part 3b: BIC formula on the worked example from the statement (K=2, d=1, n=100, ell=-140.0) ---
X_bic_example = np.zeros((100, 1))
model_bic_example = {"weights": np.array([0.5, 0.5]), "log_likelihoods": [-140.0]}
assert round(bic(X_bic_example, model_bic_example), 2) == 303.03
print("BIC worked example OK")

# --- Part 3b: BIC selects K = 3 ---
bics = [bic(X, fit_gmm(X, K, seed=0)) for K in range(1, 7)]
assert int(np.argmin(bics)) == 2                                       # K = 3 (0-indexed: K=1 is index 0)
assert bics[2] < bics[1] - 400 and bics[2] < bics[3] - 10              # a comfortable margin either side
print("BIC table (K=1..6):", [round(b, 1) for b in bics])

# --- Part 3a: the small-variance limit, including the tied point that never resolves ---
means_tiny = np.array([[0.0, 0.0], [3.0, 0.0]])
weights_tiny = np.array([0.5, 0.5])
X_tiny = np.array([[1.0, 0.0], [2.5, 0.5], [1.5, 0.0]])                 # the last point is exactly equidistant

expected = {
    1.0: [[0.8176, 0.1824], [0.0474, 0.9526], [0.5000, 0.5000]],
    0.5: [[0.9975, 0.0025], [0.0000, 1.0000], [0.5000, 0.5000]],
    0.1: [[1.0000, 0.0000], [0.0000, 1.0000], [0.5000, 0.5000]],
}
for sigma, want in expected.items():
    covs_tiny = np.stack([np.eye(2) * sigma ** 2, np.eye(2) * sigma ** 2])
    resp_tiny, _ = _e_step(X_tiny, weights_tiny, means_tiny, covs_tiny)
    assert np.allclose(np.round(resp_tiny, 4), want, atol=5e-5)
print("small-variance limit table OK")

# the same limit for the M-step mean update -> k-means centroids, on the 3-cluster data
dists = np.stack([np.linalg.norm(X - m, axis=1) for m in true_means], axis=1)
hard = np.argmin(dists, axis=1)
kmeans_centroids = np.stack([X[hard == k].mean(axis=0) for k in range(3)])
weights_eq = np.full(3, 1 / 3)
for sigma in (1.0, 0.1, 0.01, 0.001):
    covs_shared = np.stack([np.eye(2) * sigma ** 2 for _ in range(3)])
    resp_lim, _ = _e_step(X, weights_eq, true_means, covs_shared)
    _, means_lim, _ = _m_step(X, resp_lim, 1e-6)
    assert np.max(np.abs(means_lim - kmeans_centroids)) < 1e-3
print("k-means limit OK")

# --- Follow-up: without reg, the density at a collapsing component's mean is unbounded ---
x_pt = np.array([[0.0, 0.0]])
mean_pt = np.array([0.0, 0.0])
eps_values = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-12, 1e-20)
logps = [_log_gaussian_pdf(x_pt, mean_pt, eps * np.eye(2))[0] for eps in eps_values]
assert all(logps[i] < logps[i + 1] for i in range(len(logps) - 1))     # strictly increasing as eps shrinks
assert logps[-1] - logps[0] > 40                                        # genuinely diverging, not just "a bit bigger"
assert np.allclose(logps, [-np.log(2 * np.pi * eps) for eps in eps_values], atol=1e-8)
print("singular likelihood OK")

print("all checks passed")
```

</details>

</details>
