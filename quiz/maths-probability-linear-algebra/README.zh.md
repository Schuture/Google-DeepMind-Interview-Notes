# 数学问答：概率、统计与线性代数

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 知识问答 · 口述，含推导 | ★★★☆☆ | 中等 | RS · RE · MLE · Intern | bayes-theorem, expectation, markov-chains, maximum-likelihood, map-estimation, kl-divergence, svd, matrix-calculus, conditioning | 12 个问题 / 45–60 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

全文中，$\log$ 表示自然对数，$\mathbb{E}[\cdot]$ 与 $\mathrm{Var}(\cdot)$ 分别是期望和方差，
$\mathcal{N}(\mu, \sigma^2)$ 表示均值为 $\mu$、方差为 $\sigma^2$ 的高斯分布密度。

### 概率

- **Q1.** 罐子 A 中有 3 个红球和 1 个蓝球；罐子 B 中有 1 个红球和 3 个蓝球。先用一枚均匀硬币决定选哪个罐子，
  再从中摸出一球：结果是红球。求 $P(A \mid \text{red})$。接着不放回地从同一个罐子里再摸一球
  （第一个球不放回去）；求这第二个球也是红球的概率。
- **Q2.** 不断投掷一枚均匀硬币，直到图案 HH（连续两次正面）首次出现为止，求所需投掷次数的期望。
  对图案 HT 求同样的期望，并解释这两个期望为什么不同。
- **Q3.** 设 $X_1, \dots, X_n$ 是独立同分布的 $\mathrm{Uniform}(0, 1)$ 随机变量。将 $\mathbb{E}[\max_i X_i]$
  与 $\mathbb{E}[\min_i X_i]$ 表示为 $n$ 的函数并求出。
- **Q4.** 某诊断测试的灵敏度（sensitivity，患病者测出阳性的概率）为 99%，特异度（specificity，未患病者
  测出阴性的概率）为 95%。被测人群中患病率为 1%。求 $P(\text{disease} \mid \text{positive})$，并用一句话
  解释它为什么远低于灵敏度。

### 统计

- **Q5.** 设 $X_1, \dots, X_n$ 是来自 $\mathcal{N}(\mu, \sigma^2)$ 的独立同分布样本，$\mu$ 和 $\sigma^2$ 均未知。
  推导极大似然估计（maximum likelihood estimation，MLE）$\hat\mu$ 和 $\hat\sigma^2$，并证明 $\hat\sigma^2$ 是
  $\sigma^2$ 的有偏估计，$\mathbb{E}[\hat\sigma^2] = \frac{n-1}{n}\sigma^2$。
- **Q6.** 设 $y = Xw + \varepsilon$，其中 $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$，并对每个权重 $w_j$
  独立地施加 $\mathcal{N}(0, \tau^2)$ 先验。证明 $w$ 的最大后验（MAP，maximum a posteriori）估计就是岭回归
  （ridge regression）解 $\hat w = \arg\min_w \|Xw - y\|^2 + \lambda \|w\|^2$，并用 $\sigma^2$ 和 $\tau^2$
  表示出对应的 $\lambda$。再证明把高斯先验换成对每个 $w_j$ 独立的拉普拉斯（Laplace）先验后，MAP 估计变成
  L1 正则化的 Lasso 解，同样用 $\sigma^2$ 和先验参数表示出 $\lambda$。
- **Q7.** 定义同一空间上两个密度之间的 Kullback–Leibler 散度 $\mathrm{KL}(p\,\|\,q)$，证明
  $\mathrm{KL}(p\,\|\,q) \ge 0$，且等号成立当且仅当 $p = q$ 几乎处处成立。举一个两点（伯努利）分布的例子，
  说明 $\mathrm{KL}(p\,\|\,q) \ne \mathrm{KL}(q\,\|\,p)$。再考虑用单个高斯 $q$ 去拟合一个双峰（两个分量、
  峰值分离良好）高斯混合分布 $p$：给出推导，说明最小化正向散度 $\mathrm{KL}(p\,\|\,q)$ 得到的 $q$
  是什么样子（“覆盖众数”，mode-covering）；再定性说明最小化反向散度 $\mathrm{KL}(q\,\|\,p)$ 得到的 $q$
  是什么样子（“寻找众数”，mode-seeking）。
- **Q8.** 你希望用正态近似估计一个分类器的准确率，使其在 95% 置信度下的误差在 $\pm 1$ 个百分点以内。
  在真实准确率 $p$ 未知的最坏情况下，需要多少个独立同分布的测试样本？说明该正态近似成立所需的假设。

### 线性代数与微积分

- **Q9.** 对比实矩阵 $X \in \mathbb{R}^{m \times n}$ 的特征分解（eigendecomposition）与奇异值分解
  （SVD）：二者各自在什么条件下存在？$X$ 的奇异值与 $X^\top X$ 的特征值之间是什么关系？$X$ 的秩在其
  SVD 中如何体现？给定一个以观测为行的数据矩阵，主成分分析（principal component analysis，PCA）
  又是如何从中心化数据的 SVD 计算出来的？
- **Q10.** 给出实对称矩阵半正定（positive semidefinite，PSD）的定义。证明任意协方差矩阵都是半正定的。
  陈述并论证半正定的海森矩阵（Hessian）与二阶可微函数凸性之间的关系。
- **Q11.** 推导 $x \in \mathbb{R}^n$、$A \in \mathbb{R}^{n \times n}$ 时的 $\nabla_x (x^\top A x)$；
  推导 $X \in \mathbb{R}^{m \times p}$、$W \in \mathbb{R}^{p \times k}$、$Y \in \mathbb{R}^{m \times k}$ 时的
  $\nabla_W \|XW - Y\|_F^2$，即 $XW - Y$ 的平方 Frobenius 范数（矩阵所有元素平方和）的梯度；
  再推导 softmax 函数 $s = \operatorname{softmax}(z)$ 关于 $z \in \mathbb{R}^K$ 的雅可比矩阵（Jacobian）。
- **Q12.** 设 $f(x) = \tfrac12 x^\top A x$，$A \in \mathbb{R}^{n \times n}$ 对称正定，特征值为
  $0 < \lambda_1 \le \dots \le \lambda_n = \lambda_{\max}$，考虑步长 $\eta = 1/\lambda_{\max}$ 的梯度下降
  $x_{t+1} = x_t - \eta \nabla f(x_t)$。证明误差沿 $A$ 的每个特征向量方向都按几何速率收缩，第 $i$ 个特征
  方向每步收缩因子为 $1 - \lambda_i/\lambda_{\max}$，因此收敛最慢的方向以 $1 - 1/\kappa$ 收缩，其中
  $\kappa = \lambda_{\max}/\lambda_{\min}$ 是条件数（condition number）。解释条件数过大为何会拖慢收敛，
  以及特征归一化或预条件化（preconditioning）为何有帮助。

## 参考解答

<details>
<summary>展开参考解答</summary>

先向面试官确认 $\log$ 的约定（全文取自然对数），以及遇到“推导”时，是否可以直接引用标准结论，还是要从
第一性原理给出；下面的推导无论如何都给出完整过程。

### 概率

**Q1.** $P(A \mid \text{red}) = 3/4$，第二个球是红球的概率是 $1/2$。由贝叶斯定理，
$P(\text{red} \mid A) = 3/4$，$P(\text{red} \mid B) = 1/4$，于是
$P(\text{red}) = \tfrac12 \cdot \tfrac34 + \tfrac12 \cdot \tfrac14 = \tfrac12$，从而

$$P(A \mid \text{red}) = \frac{P(\text{red} \mid A)\,P(A)}{P(\text{red})} = \frac{\tfrac34 \cdot \tfrac12}{\tfrac12} = \frac34.$$

对第二次抽取，按所在的罐子分情况讨论：以 $3/4$ 的概率罐子是 A，第一个红球取走后还剩 2 红 1 蓝，
第二个球是红球的概率是 $2/3$；以 $1/4$ 的概率罐子是 B，B 唯一的红球已被取走后剩 0 红 3 蓝，
第二个球是红球的概率是 $0$。因此

$$P(\text{2nd red} \mid \text{1st red}) = \frac34 \cdot \frac23 + \frac14 \cdot 0 = \frac12.$$

**Q2.** $\mathbb{E}[\text{flips to HH}] = 6$，$\mathbb{E}[\text{flips to HT}] = 4$。用一个状态位来追踪：
$0$ 表示上一次不是正面（或尚未投掷），$H$ 表示上一次是正面。对 HH：从状态 $0$ 出发，以 $1/2$ 的概率
投出正面（转到 $H$），以 $1/2$ 的概率投出反面（留在 $0$），所以 $E_0 = 1 + \tfrac12 E_H + \tfrac12 E_0$；
从状态 $H$ 出发，以 $1/2$ 的概率投出正面（结束），以 $1/2$ 的概率投出反面（回到 $0$——刚才那次正面
就白费了），所以 $E_H = 1 + \tfrac12 E_0$。解得 $E_H = 4$，$E_0 = 6$。对 HT，状态 $0$ 出发的方程一样，
$E_0 = 1 + \tfrac12 E_H + \tfrac12 E_0$；但从状态 $H$ 出发，以 $1/2$ 的概率投出反面（结束），以 $1/2$
的概率再投出正面（留在 $H$——这个新的正面照样可以作为下一次匹配的开头），所以
$E_H = 1 + \tfrac12 E_H$，得 $E_H = 2$，$E_0 = 4$。两者不同的原因在于 HH 与自身重叠（它长度为 1 的前缀
H 同时也是它的后缀）：一次未中（正面之后来了反面）会把那个正面完全作废，搜索必须从头开始；而 HT
的一次未中（正面之后又来一个正面）什么都没有浪费，因为这个新正面本身就是一个和第一个一样好的开头。
自重叠的图案平均而言等待时间更长。

**Q3.** $\mathbb{E}[\max] = n/(n+1)$，$\mathbb{E}[\min] = 1/(n+1)$。最大值满足 $P(\max \le x) = x^n$
（$x \in [0, 1]$，因为要求每个 $X_i \le x$），因而密度为 $n x^{n-1}$，

$$\mathbb{E}[\max] = \int_0^1 x \cdot n x^{n-1}\,dx = \frac{n}{n+1}.$$

对最小值，注意若 $U \sim \mathrm{Uniform}(0,1)$，则 $1 - U$ 也是 $\mathrm{Uniform}(0,1)$，所以
$1 - \min_i X_i = \max_i (1 - X_i)$ 与 $\max_i X_i$ 同分布；两边取期望，
$1 - \mathbb{E}[\min] = n/(n+1)$，故 $\mathbb{E}[\min] = 1/(n+1)$。

**Q4.** $P(\text{disease} \mid \text{positive}) = 1/6 \approx 16.7\%$。由全概率公式（law of total
probability），$P(\text{positive}) = P(\text{pos} \mid D)P(D) + P(\text{pos} \mid \lnot D)P(\lnot D)
= 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0594$，再由贝叶斯定理，

$$P(D \mid \text{positive}) = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} = \frac16.$$

这就是基率效应（base-rate effect）：由于患病本身很罕见，健康人群（占总体 99%）远多于患病人群，
即使假阳性率只有 5%，作用在这么庞大的健康人群上产生的假阳性人数（占总体 4.95%）也超过了真阳性人数
（占总体 0.99%）——大多数阳性结果其实都是误报。

### 统计

**Q5.** $\hat\mu = \bar X = \frac1n \sum_i X_i$，$\hat\sigma^2 = \frac1n \sum_i (X_i - \bar X)^2$，
且后者以因子 $(n-1)/n$ 偏低。对数似然为
$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (X_i - \mu)^2$；
令 $\partial \ell/\partial \mu = 0$ 得 $\sum_i (X_i - \mu) = 0$，即 $\hat\mu = \bar X$；在 $\mu = \hat\mu$
处令 $\partial \ell/\partial \sigma^2 = 0$ 得 $\hat\sigma^2 = \frac1n \sum_i (X_i - \bar X)^2$。
关于偏差，围绕真实均值展开：

$$\sum_i (X_i - \bar X)^2 = \sum_i \bigl[(X_i - \mu) - (\bar X - \mu)\bigr]^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2,$$

这里用到 $\sum_i (X_i - \mu) = n (\bar X - \mu)$。两边取期望，$\mathbb{E}\sum_i (X_i - \mu)^2 = n\sigma^2$，
$\mathbb{E}[n(\bar X - \mu)^2] = n\,\mathrm{Var}(\bar X) = \sigma^2$，所以
$\mathbb{E}\sum_i (X_i - \bar X)^2 = (n-1)\sigma^2$，$\mathbb{E}[\hat\sigma^2] = \frac{n-1}{n}\sigma^2$。
这个估计量用掉了一个自由度去用 $\bar X$ 估计 $\mu$，而 $\bar X$ 平均而言比真实的 $\mu$ 更贴近样本本身，
从而低估了离散程度；把分母换成 $n - 1$ 而不是 $n$，恰好消除这个偏差。

**Q6.** 岭回归对应 $\lambda = \sigma^2/\tau^2$；对于尺度参数为 $b$ 的拉普拉斯先验，L1 惩罚对应
$\lambda = 2\sigma^2/b$。由贝叶斯定理，后验满足 $p(w \mid X, y) \propto p(y \mid X, w)\,p(w)$，
所以 MAP 估计是使 $\log p(y \mid X, w) + \log p(w)$ 最大、亦即使其负数最小的 $w$。高斯似然给出
$-\log p(y \mid X, w) = \frac{1}{2\sigma^2}\|Xw - y\|^2 + \text{常数}$，高斯先验给出
$-\log p(w) = \frac{1}{2\tau^2}\|w\|^2 + \text{常数}$，于是 MAP 的目标函数是

$$\frac{1}{2\sigma^2}\|Xw - y\|^2 + \frac{1}{2\tau^2}\|w\|^2,$$

乘以 $2\sigma^2$（不改变最小值点）得到 $\|Xw - y\|^2 + \lambda \|w\|^2$，其中 $\lambda = \sigma^2/\tau^2$——
正是岭回归。把先验换成对每个 $w_j$ 独立的 Laplace$(0, b)$，其密度为 $\frac{1}{2b} e^{-|w_j|/b}$，
给出 $-\log p(w) = \frac1b \|w\|_1 + \text{常数}$，MAP 目标函数变成
$\frac{1}{2\sigma^2}\|Xw - y\|^2 + \frac1b \|w\|_1$；同样乘以 $2\sigma^2$ 得到
$\|Xw - y\|^2 + \lambda \|w\|_1$，其中 $\lambda = 2\sigma^2/b$——正是 Lasso。高斯先验对大权重施加二次惩罚，
把每个坐标平滑地收缩向 $0$；拉普拉斯先验在 $0$ 处有一个尖峰（惩罚项在该点不可微），这个拐点处的
次梯度（subgradient）可以把某个坐标恰好锁在 $0$ 上，这就是 Lasso 会产生稀疏解、而岭回归一般不会的原因。

**Q7.** $\mathrm{KL}(p\,\|\,q) = \mathbb{E}_p\bigl[\log \frac{p(X)}{q(X)}\bigr] = \int p(x) \log
\frac{p(x)}{q(x)}\,dx \ge 0$，这就是两个密度之间的 KL 散度（Kullback–Leibler divergence）。
证明非负性：对凸函数 $-\log$ 应用詹森不等式（Jensen's inequality），取 $X \sim p$ 下的
$Y = q(X)/p(X)$：$\mathbb{E}_p[Y] = \int p(x) \frac{q(x)}{p(x)}\,dx = \int q(x)\,dx = 1$
（积分范围是 $p$ 的支撑集），于是

$$\mathrm{KL}(p\,\|\,q) = \mathbb{E}_p[-\log Y] \ge -\log \mathbb{E}_p[Y] = -\log 1 = 0,$$

且等号成立当且仅当 $Y$ 几乎必然为常数（因为 $-\log$ 是严格凸函数），即 $q = p$ 几乎处处成立。
不对称的例子：取 $p = \mathrm{Bernoulli}(0.9)$，$q = \mathrm{Bernoulli}(0.5)$，

$$\mathrm{KL}(p\,\|\,q) = 0.9 \log\frac{0.9}{0.5} + 0.1 \log\frac{0.1}{0.5} \approx 0.368, \qquad \mathrm{KL}(q\,\|\,p) = 0.5 \log\frac{0.5}{0.9} + 0.5 \log\frac{0.5}{0.1} \approx 0.511,$$

两者并不相等。对高斯拟合，记 $q = \mathcal{N}(\mu, \sigma^2)$；最小化
$\mathrm{KL}(p\,\|\,q) = -H(p) + H(p, q)$ 等价于最小化交叉熵
$H(p, q) = \mathbb{E}_p[-\log q(X)] = \frac12 \log(2\pi\sigma^2) + \frac{\mathbb{E}_p[(X - \mu)^2]}{2\sigma^2}$。
无论 $\sigma^2$ 取何值，这在 $\mu = \mathbb{E}_p[X]$ 处对 $\mu$ 取最小；随后在 $\sigma^2 = \mathrm{Var}_p(X)$
处对 $\sigma^2$ 取最小：正向 KL 的最优解正是*矩匹配*（moment matching）的结果。由于 $p$ 关于 $\pm 3$
对称、每个分量方差为 1，$\mathbb{E}_p[X] = 0$，$\mathrm{Var}_p(X) = \mathbb{E}_p[X^2] = 10$
（每个分量的 $\mathrm{Var} + \text{均值}^2 = 1 + 9 = 10$），所以正向 KL 拟合出的是
$\mathcal{N}(0, 10)$：又宽又居中，为的是让 $q$ 在 $p$ 有质量的地方都保持为正（否则
$p \log(p/q) \to \infty$）。反向散度 $\mathrm{KL}(q\,\|\,p) = \mathbb{E}_q[\log q(X) - \log p(X)]$
则是惩罚 $q$ 把质量放在 $p$ 很小的地方，因此没有动机去同时覆盖两个峰；从某一个峰附近出发做基于梯度的
拟合，会得到一个紧紧包住那一个峰的 $q$，其均值和方差都接近该峰自身，而不是同时覆盖两个峰。

**Q8.** 最坏情况下 $n \approx 9{,}604$。对 $n$ 次独立同分布的 Bernoulli$(p)$ 试验，样本比例 $\hat p$
满足 $\mathrm{Var}(\hat p) = p(1-p)/n$，由中心极限定理，当 $n$ 足够大时它近似服从
$\mathcal{N}(p,\, p(1-p)/n)$。95% 双侧置信区间的半宽为 $z_{0.975}\sqrt{p(1-p)/n}$，其中
$z_{0.975} = 1.96$；令它等于目标误差 $E = 0.01$ 并解出 $n$，得到

$$n = \frac{z_{0.975}^2\, p(1-p)}{E^2}.$$

因为对任意 $p \in [0, 1]$ 都有 $p(1-p) \le \frac14$，且在 $p = 1/2$ 处取等号，最坏情况是
$n = 1.96^2 \times 0.25 / 0.01^2 = 9{,}604$。这需要 $n$ 大到使二项分布的正态近似足够准确——
具体来说，需要测试样本是独立同分布地抽取、真实准确率 $p$ 固定不变（测试集上没有分布漂移），
并且 $np$ 与 $n(1-p)$ 都明显大于约 5–10；只要 $p$ 不是极端接近 0 或 1，9,604 都能满足这个要求。

### 线性代数与微积分

**Q9.** SVD $X = U\Sigma V^\top$（其中 $U \in \mathbb{R}^{m \times m}$、$V \in \mathbb{R}^{n \times n}$
是正交矩阵，$\Sigma$ 是长方对角矩阵，对角元 $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$）对任意形状、任意
实矩阵 $X$ 都存在，不需要任何前提条件。特征分解（eigendecomposition）$A = Q\Lambda Q^{-1}$ 则要求方阵，
只有当 $A$ 对称时才保证 $\Lambda$ 为实数、$Q$ 为正交矩阵（谱定理，spectral theorem），而一般的方阵甚至
可能根本没有特征分解：右上角为 $1$、其余元素都是 $0$ 的 $2\times 2$ 矩阵，特征值 $0$ 的代数重数为 2，
对应的特征向量空间却只有一维，找不到可逆的 $Q$ 把它对角化——这就是一个*亏损矩阵*（defective matrix）。由于
$X^\top X = V \Sigma^\top U^\top U \Sigma V^\top = V(\Sigma^\top \Sigma) V^\top$，而
$\Sigma^\top \Sigma = \mathrm{diag}(\sigma_i^2)$，这恰好就是对称半正定矩阵 $X^\top X$ 的特征分解：
它的特征值是 $\sigma_i^2$，特征向量是 $V$ 的列，即 $\sigma_i = \sqrt{\lambda_i(X^\top X)}$。
由于 $U, V$ 都可逆，$\mathrm{rank}(X) = \mathrm{rank}(\Sigma)$，也就是非零奇异值的个数。对 PCA，先将
数据中心化 $X_c = X - \bar X$（减去按行的均值），再取其 SVD $X_c = U\Sigma V^\top$；此时
$X_c^\top X_c = V \Sigma^2 V^\top$ 恰好是样本协方差矩阵的 $(n-1)$ 倍，所以主成分方向就是 $V$ 的列，
方向 $i$ 上解释的方差是 $\sigma_i^2/(n-1)$，而主成分得分（数据投影到这些方向上的坐标）就是
$X_c V = U\Sigma$。

**Q10.** 实对称矩阵 $A \in \mathbb{R}^{n \times n}$ 是半正定（PSD）的，如果对任意 $x \in \mathbb{R}^n$
都有 $x^\top A x \ge 0$（等价地，每个特征值都 $\ge 0$）。对均值为 $\mu$、协方差为
$\Sigma = \mathbb{E}[(X - \mu)(X - \mu)^\top]$ 的随机向量 $X$，以及任意 $a \in \mathbb{R}^n$，

$$a^\top \Sigma a = \mathbb{E}\bigl[a^\top (X-\mu)(X-\mu)^\top a\bigr] = \mathbb{E}\bigl[(a^\top(X-\mu))^2\bigr] \ge 0,$$

因为这是一个标量随机变量平方后的期望；把 $\mathbb{E}$ 换成对样本取平均，同样的论证对样本协方差矩阵
逐字成立。所以 $\Sigma$ 总是半正定的。一个二阶可微函数 $f$ 在凸定义域上是凸函数，当且仅当
$\nabla^2 f(x)$（海森矩阵，Hessian）在域中每一点都半正定。一个方向：若 $f$ 是凸函数，则对任意方向
$v$，函数 $g(t) = f(x + tv)$ 也是凸函数（凸函数与仿射映射的复合），所以对任意 $v$ 都有
$g''(0) = v^\top \nabla^2 f(x) v \ge 0$，即 $\nabla^2 f(x)$ 半正定。反过来，若 $\nabla^2 f$ 处处半正定，
由带拉格朗日余项的泰勒定理（Taylor's theorem），对任意 $x, y$，存在线段上某点 $\xi$ 使得
$f(y) = f(x) + \nabla f(x)^\top (y - x) + \frac12 (y-x)^\top \nabla^2 f(\xi) (y - x)$；
按假设最后一项 $\ge 0$，所以对所有 $x, y$ 都有 $f(y) \ge f(x) + \nabla f(x)^\top (y - x)$，
这正是凸性的一阶判据。

**Q11.** $\nabla_x (x^\top A x) = (A + A^\top) x$；$\nabla_W \|XW - Y\|_F^2 = 2X^\top(XW - Y)$；
softmax 的雅可比矩阵是 $\operatorname{diag}(s) - s s^\top$。把 $x^\top A x$ 写成
$\sum_{i,j} A_{ij} x_i x_j$，对 $x_k$ 求偏导：
$\partial_k (x^\top A x) = \sum_j A_{kj} x_j + \sum_i A_{ik} x_i = (Ax)_k + (A^\top x)_k$，所以
$\nabla_x(x^\top A x) = (A + A^\top)x$（当 $A$ 对称时等于 $2Ax$）。第二个式子，记 $R = XW - Y$，用微分：
$d\|R\|_F^2 = d\,\mathrm{tr}(R^\top R) = 2\,\mathrm{tr}(R^\top dR) = 2\,\mathrm{tr}(R^\top X\,dW) = 2\,
\mathrm{tr}\bigl((X^\top R)^\top dW\bigr)$；而 $d\|R\|_F^2 = \mathrm{tr}(G^\top dW)$ 正是梯度 $G$ 的定义，
读出 $G = 2X^\top R = 2X^\top(XW - Y)$。对 softmax，$s_i = e^{z_i}/Z$，$Z = \sum_k e^{z_k}$；由商法则，
$\partial s_i/\partial z_j = \delta_{ij} e^{z_i}/Z - e^{z_i} e^{z_j}/Z^2 = \delta_{ij} s_i - s_i s_j
= s_i(\delta_{ij} - s_j)$，写成矩阵形式就是 $J = \operatorname{diag}(s) - s s^\top$——它是对称矩阵，
且每行之和为 $0$（所有 logit 同时加上同一个常数，softmax 的输出不变）。

**Q12.** 最优点是 $x^\star = 0$，所以误差就是 $x_t$ 本身。$\nabla f(x) = Ax$，于是
$x_{t+1} = (I - \eta A) x_t$，其中 $\eta = 1/\lambda_{\max}$。将 $A$ 对角化为 $A = Q\Lambda Q^\top$，
记 $y_t = Q^\top x_t$ 为 $x_t$ 在特征基下的坐标；则 $y_{t+1} = (I - \eta\Lambda) y_t$ 是对角的，
逐个坐标看，

$$y_{t+1,i} = \Bigl(1 - \frac{\lambda_i}{\lambda_{\max}}\Bigr) y_{t,i} \quad\Longrightarrow\quad y_{t,i} = \Bigl(1 - \frac{\lambda_i}{\lambda_{\max}}\Bigr)^{t} y_{0,i}.$$

每个坐标都按几何速率收缩，速率只取决于自己的特征值：$\lambda_{\max}$ 方向上
$1 - \lambda_{\max}/\lambda_{\max} = 0$，走一步就恰好归零；而 $\lambda_{\min}$ 方向上的收缩因子
$1 - \lambda_{\min}/\lambda_{\max} = 1 - 1/\kappa$ 最大（最慢）。由于误差范数 $\|x_t\|$ 是这些
分方向项之和，一旦较快的方向都已衰减殆尽，它就由 $\lambda_{\min}$ 那一项主导，
$\|x_t\| / \|x_{t-1}\| \to 1 - 1/\kappa$：整体收敛速率由条件最差的那个方向决定，
不管其余方向收敛得多快。当 $\kappa \gg 1$（病态，ill-conditioning——$f$ 在某些方向上远比其他方向平坦）
时，$1 - 1/\kappa$ 接近 $1$，沿平坦方向的收敛极其缓慢，即便步长已经是针对最陡方向调到最优；
更小的步长只会让平坦方向更慢，因为 $\eta = 1/\lambda_{\max}$ 已经是沿最陡方向不发散的最大步长。
特征归一化（把输入缩放到相近的方差）和预条件化（preconditioning，把 $A$ 换成 $P^{-1}A$，其中
$P \approx A$，例如海森矩阵的对角或分块近似）都是通过缩小最大曲率与最小曲率之比来起作用的，
把 $\kappa$ 拉向 $1$，使每个方向都以相近、较快的速率收缩。

<details>
<summary>验证代码（可运行）</summary>

```python
import math
from fractions import Fraction

import numpy as np
from scipy import integrate, stats
from scipy.optimize import minimize
from sklearn.linear_model import Lasso as SkLasso

rng = np.random.default_rng(2026)

# ---- Q1: urns, Bayes' theorem, exact fractions and Monte Carlo
p_red_A, p_red_B = Fraction(3, 4), Fraction(1, 4)          # urn A: 3 red / 4; urn B: 1 red / 4
p_A_post = p_red_A * Fraction(1, 2) / (p_red_A * Fraction(1, 2) + p_red_B * Fraction(1, 2))
assert p_A_post == Fraction(3, 4)
p_second_red = p_A_post * Fraction(2, 3) + (1 - p_A_post) * Fraction(0, 3)
assert p_second_red == Fraction(1, 2)

N = 2_000_000
urn_is_A = rng.random(N) < 0.5
counts = np.where(urn_is_A[:, None], np.array([3, 1]), np.array([1, 3]))   # [red, blue] remaining
first_red = rng.random(N) < (counts[:, 0] / 4)
remaining = counts.copy()
remaining[first_red, 0] -= 1                                # NOTE: the first ball is not returned
remaining[~first_red, 1] -= 1
assert math.isclose(urn_is_A[first_red].mean(), 0.75, abs_tol=0.01)
second_red = rng.random(N) < (remaining[:, 0] / 3)
assert math.isclose(second_red[first_red].mean(), 0.5, abs_tol=0.01)

# ---- Q2: expected flips to first HH, first HT, by solving the Markov-chain equations and by simulation
# state 0: last flip was not H (or none yet); state H: last flip was H
A_hh = np.array([[0.5, -0.5], [-0.5, 1.0]])                 # E0 - EH = 2 ; -0.5 E0 + EH = 1
E0_hh, EH_hh = np.linalg.solve(A_hh, np.array([1.0, 1.0]))
assert math.isclose(E0_hh, 6.0, rel_tol=1e-9) and math.isclose(EH_hh, 4.0, rel_tol=1e-9)
A_ht = np.array([[0.5, -0.5], [0.0, 0.5]])                  # E0 - EH = 2 ; 0.5 EH = 1
E0_ht, EH_ht = np.linalg.solve(A_ht, np.array([1.0, 1.0]))
assert math.isclose(E0_ht, 4.0, rel_tol=1e-9) and math.isclose(EH_ht, 2.0, rel_tol=1e-9)

L, reps = 60, 300_000
flips = rng.integers(0, 2, size=(reps, L))                  # 1 = H, 0 = T
is_hh = (flips[:, :-1] == 1) & (flips[:, 1:] == 1)
is_ht = (flips[:, :-1] == 1) & (flips[:, 1:] == 0)
assert is_hh.any(axis=1).all() and is_ht.any(axis=1).all()  # NOTE: argmax silently returns 0 with no True value,
wait_hh, wait_ht = is_hh.argmax(axis=1) + 2, is_ht.argmax(axis=1) + 2   #      so a match must be guaranteed first
assert math.isclose(wait_hh.mean(), 6.0, abs_tol=0.05)
assert math.isclose(wait_ht.mean(), 4.0, abs_tol=0.05)

# ---- Q3: E[max] and E[min] of n iid Uniform(0,1), for several n, by quadrature and by simulation
for n3 in (1, 2, 5, 10, 30):
    e_max, e_min = n3 / (n3 + 1), 1 / (n3 + 1)
    assert math.isclose(integrate.quad(lambda x: x * n3 * x ** (n3 - 1), 0, 1)[0], e_max, rel_tol=1e-8)
    assert math.isclose(integrate.quad(lambda x: x * n3 * (1 - x) ** (n3 - 1), 0, 1)[0], e_min, rel_tol=1e-8)
    u3 = rng.random((300_000, n3))
    assert math.isclose(u3.max(axis=1).mean(), e_max, abs_tol=0.01)
    assert math.isclose(u3.min(axis=1).mean(), e_min, abs_tol=0.01)

# ---- Q4: diagnostic test, base rate, exact fractions and Monte Carlo
sens, spec, prev = Fraction(99, 100), Fraction(95, 100), Fraction(1, 100)
true_pos_rate = sens * prev                    # 0.99 x 0.01 = 0.0099, i.e. 0.99% of the whole population
false_pos_rate = (1 - spec) * (1 - prev)        # 0.05 x 0.99 = 0.0495, i.e. 4.95% of the whole population
assert true_pos_rate == Fraction(99, 10000) and false_pos_rate == Fraction(495, 10000)
p_pos = true_pos_rate + false_pos_rate
assert p_pos == Fraction(594, 10000)
p_disease_given_pos = true_pos_rate / p_pos
assert p_disease_given_pos == Fraction(1, 6)

disease = rng.random(3_000_000) < float(prev)
pos = np.where(disease, rng.random(3_000_000) < float(sens), rng.random(3_000_000) < (1 - float(spec)))
assert math.isclose(disease[pos].mean(), 1 / 6, abs_tol=0.01)

# ---- Q5: Gaussian MLE, biased variance, for several n, by simulation
mu_true, sigma_true = 2.0, 3.0
for n5 in (3, 6, 15):
    X5 = rng.normal(mu_true, sigma_true, size=(300_000, n5))
    xbar5 = X5.mean(axis=1)
    sigma2_hat = ((X5 - xbar5[:, None]) ** 2).mean(axis=1)   # NOTE: MLE divides by n, not n - 1
    assert math.isclose(xbar5.mean(), mu_true, abs_tol=0.02)
    assert math.isclose(sigma2_hat.mean(), (n5 - 1) / n5 * sigma_true ** 2, rel_tol=0.02)
    assert math.isclose(sigma2_hat.mean() * n5 / (n5 - 1), sigma_true ** 2, rel_tol=0.02)   # the n/(n-1) correction

# ---- Q6: ridge and lasso as MAP estimates
n6, p6 = 200, 5
X6 = rng.standard_normal((n6, p6))
sigma2, tau2 = 0.5, 2.0
lam_ridge = sigma2 / tau2
y6 = X6 @ np.array([1.0, -2.0, 0.5, 3.0, -1.5]) + rng.normal(scale=math.sqrt(sigma2), size=n6)
w_ridge_closed = np.linalg.solve(X6.T @ X6 + lam_ridge * np.eye(p6), X6.T @ y6)


def neg_log_posterior_gaussian(w):
    return ((X6 @ w - y6) @ (X6 @ w - y6)) / (2 * sigma2) + (w @ w) / (2 * tau2)


def grad_gaussian(w):
    return X6.T @ (X6 @ w - y6) / sigma2 + w / tau2


map_result = minimize(neg_log_posterior_gaussian, np.zeros(p6), jac=grad_gaussian, method="BFGS", tol=1e-12)
assert np.allclose(w_ridge_closed, map_result.x, atol=1e-5)

b_laplace = 0.3
lam_l1 = 2 * sigma2 / b_laplace                              # Laplace(0, b) prior -> L1 penalty lambda = 2 sigma^2 / b


def lasso_ista(X, y, lam, n_iter=4000):
    lipschitz = 2 * np.linalg.eigvalsh(X.T @ X).max()
    step = 1.0 / lipschitz
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        w = w - step * 2 * X.T @ (X @ w - y)
        w = np.sign(w) * np.maximum(np.abs(w) - step * lam, 0.0)   # proximal step: soft-thresholding
    return w


w_ista = lasso_ista(X6, y6, lam_l1)
# NOTE: sklearn's Lasso objective is (1/(2n)) ||y - Xw||^2 + alpha ||w||_1, so alpha = lambda / (2n), not lambda
w_sklearn = SkLasso(alpha=lam_l1 / (2 * n6), fit_intercept=False, tol=1e-14, max_iter=200_000).fit(X6, y6).coef_
assert np.allclose(w_ista, w_sklearn, atol=1e-3)

# ---- Q7: KL divergence: non-negativity, asymmetry, forward vs reverse fit to a bimodal mixture
def kl_discrete(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


for _ in range(2000):                                        # non-negativity on random categorical distributions
    k = rng.integers(2, 8)
    a, b = rng.random(k) + 1e-3, rng.random(k) + 1e-3
    assert kl_discrete(a / a.sum(), b / b.sum()) >= -1e-10

a2, b2 = 0.9, 0.5                                             # asymmetric two-point (Bernoulli) example
p2, q2 = np.array([a2, 1 - a2]), np.array([b2, 1 - b2])
kl_pq, kl_qp = kl_discrete(p2, q2), kl_discrete(q2, p2)
assert math.isclose(kl_pq, a2 * math.log(a2 / b2) + (1 - a2) * math.log((1 - a2) / (1 - b2)), rel_tol=1e-9)
assert math.isclose(kl_pq, 0.368, abs_tol=5e-4) and math.isclose(kl_qp, 0.511, abs_tol=5e-4)
assert kl_pq < kl_qp and not math.isclose(kl_pq, kl_qp, rel_tol=0.05)

M1, M2, SD = -3.0, 3.0, 1.0                                   # two well-separated unit-variance modes


def mixture_pdf(x):
    return 0.5 * stats.norm.pdf(x, M1, SD) + 0.5 * stats.norm.pdf(x, M2, SD)


grid = np.linspace(-20.0, 20.0, 8001)
dx = grid[1] - grid[0]
p_grid = mixture_pdf(grid)


def forward_kl_obj(params):                                  # KL(p || q), q = N(mu, exp(log_sigma)^2)
    mu, log_sigma = params
    q = np.clip(stats.norm.pdf(grid, mu, math.exp(log_sigma)), 1e-300, None)
    return np.sum(p_grid * (np.log(np.clip(p_grid, 1e-300, None)) - np.log(q))) * dx


def reverse_kl_obj(params):                                   # KL(q || p)
    mu, log_sigma = params
    q = stats.norm.pdf(grid, mu, math.exp(log_sigma))
    qc, pc = np.clip(q, 1e-300, None), np.clip(p_grid, 1e-300, None)
    return np.sum(q * (np.log(qc) - np.log(pc))) * dx


fwd = minimize(forward_kl_obj, x0=[1.0, math.log(2.0)], method="BFGS", tol=1e-12)
mu_fwd, sigma_fwd = fwd.x[0], math.exp(fwd.x[1])
assert math.isclose(mu_fwd, 0.0, abs_tol=0.02) and math.isclose(sigma_fwd, math.sqrt(10.0), rel_tol=0.01)

rev = minimize(reverse_kl_obj, x0=[2.5, math.log(1.2)], method="BFGS", tol=1e-12)   # NOTE: started near one mode;
mu_rev, sigma_rev = rev.x[0], math.exp(rev.x[1])                                    #  reverse KL is multimodal
assert math.isclose(mu_rev, M2, abs_tol=0.05) and math.isclose(sigma_rev, SD, abs_tol=0.05)
assert sigma_rev < sigma_fwd / 2

# ---- Q8: sample size for a +-1pp margin at 95% confidence
z95 = stats.norm.ppf(0.975)
assert math.isclose(z95, 1.96, abs_tol=5e-4)
n_needed = math.ceil(z95 ** 2 * 0.25 / 0.01 ** 2)
assert n_needed == 9604
p_grid8 = np.linspace(0.0, 1.0, 100_001)
assert np.all(p_grid8 * (1 - p_grid8) <= 0.25 + 1e-12)         # p(1 - p) <= 1/4, the worst case used above
phat = rng.binomial(n_needed, 0.5, size=20_000) / n_needed
assert math.isclose(np.mean(np.abs(phat - 0.5) <= 0.01), 0.95, abs_tol=0.02)   # the promised coverage, empirically

# ---- Q9: SVD vs eigendecomposition of X^T X, rank, and PCA from the SVD of centred data
X9 = rng.standard_normal((7, 5))
U9, S9, Vt9 = np.linalg.svd(X9, full_matrices=False)           # NOTE: this returns V^T, not V
eigvals9, eigvecs9 = np.linalg.eigh(X9.T @ X9)                 # ascending order
assert np.allclose(S9 ** 2, eigvals9[::-1], atol=1e-8)
signs9 = np.sign(np.sum(Vt9.T * eigvecs9[:, ::-1], axis=0))
assert np.allclose(Vt9.T, eigvecs9[:, ::-1] * signs9, atol=1e-6)

r_true = 3
A_low = rng.standard_normal((8, r_true)) @ rng.standard_normal((r_true, 6))   # an exactly rank-3 matrix
s_low = np.linalg.svd(A_low, compute_uv=False)
assert int((s_low > s_low.max() * 1e-10).sum()) == r_true == np.linalg.matrix_rank(A_low)

n9p, d9p = 300, 4
raw9 = rng.standard_normal((n9p, d9p)) @ rng.standard_normal((d9p, d9p))     # correlated columns
Xc = raw9 - raw9.mean(axis=0, keepdims=True)
Up, Sp, Vtp = np.linalg.svd(Xc, full_matrices=False)
eigvals_c, eigvecs_c = np.linalg.eigh((Xc.T @ Xc) / (n9p - 1))
assert np.allclose(Sp ** 2 / (n9p - 1), eigvals_c[::-1], atol=1e-8)          # explained variance
signs_p = np.sign(np.sum(Vtp.T * eigvecs_c[:, ::-1], axis=0))
assert np.allclose(Xc @ Vtp.T, Up * Sp, atol=1e-8)                           # scores: X_c V = U Sigma
assert np.allclose(Up * Sp, Xc @ (eigvecs_c[:, ::-1] * signs_p), atol=1e-6)  # matches the covariance eigenbasis

# ---- Q10: covariance is PSD; PSD Hessian is equivalent to convexity
data10 = rng.standard_normal((500, 6)) @ rng.standard_normal((6, 6))
centred10 = data10 - data10.mean(axis=0)
cov10 = (centred10.T @ centred10) / (len(data10) - 1)
assert np.linalg.eigvalsh(cov10).min() >= -1e-8                # NOTE: eigvalsh, not eig -- cov10 is symmetric,
for _ in range(200):                                            #  eig() can return spurious tiny imaginary parts
    a = rng.standard_normal(6)
    assert math.isclose(a @ cov10 @ a, np.mean((centred10 @ a) ** 2) * len(data10) / (len(data10) - 1), rel_tol=1e-9)


def quad_f(A, x):
    return 0.5 * x @ A @ x


Qo, _ = np.linalg.qr(rng.standard_normal((5, 5)))
A_psd = Qo @ np.diag([0.1, 1.0, 2.0, 3.0, 4.0]) @ Qo.T           # PSD: every eigenvalue positive
A_indef = Qo @ np.diag([-0.5, 1.0, 2.0, 3.0, 4.0]) @ Qo.T        # indefinite: one negative eigenvalue
for _ in range(200):                                             # exact identity: the midpoint gap is
    x, y = rng.standard_normal(5), rng.standard_normal(5)        # (1/8)(x - y)^T A (x - y)
    gap = 0.5 * (quad_f(A_psd, x) + quad_f(A_psd, y)) - quad_f(A_psd, (x + y) / 2)
    assert gap >= -1e-9 and math.isclose(gap, 0.125 * (x - y) @ A_psd @ (x - y), rel_tol=1e-9)
neg_dir = Qo[:, 0]                                                # x - y along the negative eigenvector
x0, y0 = rng.standard_normal(5), rng.standard_normal(5)
y0 = x0 - 2.0 * neg_dir
assert 0.5 * (quad_f(A_indef, x0) + quad_f(A_indef, y0)) - quad_f(A_indef, (x0 + y0) / 2) < 0

# ---- Q11: matrix-calculus identities, checked by finite differences
h = 1e-5
A11, x11 = rng.standard_normal((4, 4)), rng.standard_normal(4)
analytic11 = (A11 + A11.T) @ x11
fd11 = np.array([((x11 + h * e) @ A11 @ (x11 + h * e) - (x11 - h * e) @ A11 @ (x11 - h * e)) / (2 * h)
                  for e in np.eye(4)])
assert np.allclose(analytic11, fd11, atol=1e-4)

X11, W11, Y11 = rng.standard_normal((6, 3)), rng.standard_normal((3, 2)), rng.standard_normal((6, 2))
analytic_G = 2 * X11.T @ (X11 @ W11 - Y11)
fd_G = np.zeros_like(W11)
for i in range(3):
    for j in range(2):
        step = np.zeros_like(W11)
        step[i, j] = h
        fd_G[i, j] = (np.linalg.norm(X11 @ (W11 + step) - Y11, "fro") ** 2
                      - np.linalg.norm(X11 @ (W11 - step) - Y11, "fro") ** 2) / (2 * h)
assert np.allclose(analytic_G, fd_G, atol=1e-4)


def softmax11(z):
    e = np.exp(z - z.max())
    return e / e.sum()


z11 = rng.standard_normal(4)
s11 = softmax11(z11)
analytic_J = np.diag(s11) - np.outer(s11, s11)
fd_J = np.column_stack([(softmax11(z11 + h * e) - softmax11(z11 - h * e)) / (2 * h) for e in np.eye(4)])
assert np.allclose(analytic_J, fd_J, atol=1e-5)
assert np.allclose(analytic_J @ np.ones(4), 0.0, atol=1e-10)     # rows sum to 0: shifting every logit changes nothing

# ---- Q12: gradient descent on a quadratic, contraction along each eigendirection of A
Q12o, _ = np.linalg.qr(rng.standard_normal((4, 4)))
lambdas = np.array([1.0, 3.0, 8.0, 25.0])                        # lambda_max = 25 -> kappa = 25
A12 = Q12o @ np.diag(lambdas) @ Q12o.T
eta12 = 1.0 / lambdas.max()
x = rng.standard_normal(4)
coords = [Q12o.T @ x]
for _ in range(200):
    x = x - eta12 * (A12 @ x)
    coords.append(Q12o.T @ x)
Y12 = np.array(coords)                                            # (T + 1, 4): coordinates in the eigenbasis
predicted = 1 - lambdas / lambdas.max()
for t in range(len(Y12) - 1):
    ratio = np.divide(Y12[t + 1], Y12[t], out=np.full(4, np.nan), where=np.abs(Y12[t]) > 1e-12)
    ok = ~np.isnan(ratio)
    assert np.allclose(ratio[ok], predicted[ok], atol=1e-8)
assert math.isclose(Y12[1, np.argmax(lambdas)], 0.0, abs_tol=1e-10)   # top eigen-direction: exactly 0 after 1 step
late_ratio = np.linalg.norm(Y12[-1]) / np.linalg.norm(Y12[-2])
assert math.isclose(late_ratio, 1 - lambdas.min() / lambdas.max(), rel_tol=1e-6)   # dominated by 1 - 1/kappa

print("all checks passed")
```

</details>

</details>
