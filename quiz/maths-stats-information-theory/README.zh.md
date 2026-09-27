# 数学问答：矩阵计算、统计检验与信息论

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 知识问答 · 口述，含推导 | ★★★☆☆ | 中等 | RS · RE · MLE · Intern | matrix-multiplication, rank, matrix-inverse, pseudo-inverse, moments, central-limit-theorem, hypothesis-testing, chi-square-test, entropy, mutual-information, integration | 12 个问题 / 45–60 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

全文中，$\log$ 表示自然对数，仅当某个量以比特（bit）为单位时才写出 $\log_2$。$\|\cdot\|$ 表示向量的欧几里得范数
（Euclidean norm），$\|\cdot\|_F$ 表示矩阵的 Frobenius 范数（矩阵所有元素平方和的平方根）。对
$A \in \mathbb{R}^{m \times n}$，它的奇异值分解（singular value
decomposition，SVD）为 $A = U\Sigma V^\top$，其中 $U \in \mathbb{R}^{m \times m}$、
$V \in \mathbb{R}^{n \times n}$ 是正交矩阵，$\Sigma$ 是长方对角矩阵，对角元
$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{\min(m,n)} \ge 0$ 是 $A^\top A$ 特征值的平方根。$I_n$ 表示
$n \times n$ 单位矩阵。$\mathrm{KL}(p \,\|\, q)$ 是同一空间上两个分布之间的 Kullback–Leibler 散度；它的定义、
非负性与不对称性见[数学问答：概率、统计与线性代数](../maths-probability-linear-algebra/README.zh.md)，此处
直接使用，不再重新推导。

### 线性代数

- **Q1.** 教科书算法（schoolbook algorithm）将 $A \in \mathbb{R}^{m \times n}$ 与
  $B \in \mathbb{R}^{n \times p}$ 相乘时，$m \times p$ 乘积的每个元素都算作 $n$ 项乘积之和。求这需要多少次
  标量乘加（multiply–add）运算，从而给出两个 $n \times n$ 方阵相乘的时间复杂度——分别针对这种直接算法和
  Strassen 算法。然后，对形状为 $1000 \times 10$ 的 $A$、$10 \times 1000$ 的 $B$ 以及
  $v \in \mathbb{R}^{1000}$，按括号所示的顺序，计算求 $(AB)v$ 与 $A(Bv)$ 各自所需的乘加次数。说明这如何
  决定：(a) 一个批次上线性层前向传播的浮点运算（FLOP）代价；(b) 低秩
  更新应该按怎样的顺序相乘。
- **Q2.** 给出矩阵秩的定义，以及方阵满秩是什么意思。说明 $\mathrm{rank}(AB)$ 与 $\mathrm{rank}(A)$、
  $\mathrm{rank}(B)$ 之间的关系，并证明非零外积（outer product）$uv^\top$（$u \in \mathbb{R}^m$、
  $v \in \mathbb{R}^n$ 均非零）的秩为 1。列出关于方阵 $A \in \mathbb{R}^{n \times n}$ 的若干条件，其中每一条都与
  $A$ 可逆等价。解释为什么用 $\det(A) \ne 0$ 作为可逆性的数值判据并不可靠——以 $A = 0.1\, I_{100}$ 为例，
  它的行列式是 $10^{-100}$，但 $A$ 的数值条件（conditioning）却完美无缺——并给出一个没有这个问题的判据。
  说明这如何关系到：(a) 数据矩阵 $X$ 何时使 $X^\top X$ 可逆；(b) LoRA 更新 $BA$ 的秩。
- **Q3.** 描述求解一般（方阵、可逆）线性方程组 $Ax = b$ 的一种数值方法及其时间复杂度。给出这种方法的近似
  浮点运算次数，并与直接求出 $A^{-1}$ 再计算 $A^{-1}b$ 的浮点运算次数相比较。解释即使要为同一个 $A$ 求解
  多个右端项 $b$，为什么第一种方式仍更受青睐——无论是在运算次数上还是在数值精度上——并描述当 $A$ 是对称
  正定（symmetric positive definite）矩阵时还能进一步节省多少。说明这对以下几点意味着什么：(a) 求解线性
  回归的正规方程（normal equations）；(b) 计算高斯过程（Gaussian process）后验；(c) 求一步牛顿步
  $Hd = -g$。
- **Q4.** 从奇异值分解出发，给出 $A \in \mathbb{R}^{m \times n}$ 的 Moore–Penrose 伪逆（pseudo-inverse）
  $A^+$ 的定义。证明 $x = A^+b$ 是 $\min_x \|Ax - b\|$ 的解中、在所有取得该最小值的 $x$ 里欧几里得范数
  最小的那一个。给出当 $A$ 列满秩时 $A^+$ 的闭式，并把 $A^+$ 表示为岭回归正规方程当岭参数趋于零时的极限。
  说明这如何关系到：(a) 参数多于样本时的线性回归；(b) 在参数多于样本的最小二乘问题上，从零初始化出发的
  梯度下降收敛到什么。

### 概率与统计

- **Q5.** 对均值为 $\mu$ 的随机变量 $X$，第 $k$ 阶原点矩（raw moment）是 $E[X^k]$，第 $k$ 阶中心矩
  （central moment）是 $E[(X - \mu)^k]$；方差是二阶中心矩，偏度（skewness）是三阶中心矩除以 $\sigma^3$，
  峰度（kurtosis）是四阶中心矩除以 $\sigma^4$（超额峰度，excess kurtosis，还要再减去高斯分布对应的值 3）。
  定义矩母函数（moment-generating function）$M_X(t) = E[e^{tX}]$，并说明如何从它在 $t = 0$ 处的各阶导数
  得到各阶原点矩。举一个没有均值的分布的例子，再举一个各阶矩只存在到有限阶数的分布的例子。对密度为
  $\lambda e^{-\lambda x}$（$x \ge 0$）的 $X \sim \mathrm{Exp}(\lambda)$，推导正整数 $k$ 下的 $E[X^k]$，
  并给出它的偏度与超额峰度。说明这如何关系到：(a) Adam 内部的矩估计追踪的是什么；(b) 归一化层计算的是
  什么。
- **Q6.** 陈述独立同分布随机变量的弱大数定律（weak law of large numbers）与中心极限定理（central limit
  theorem），以及各自所需的矩假设。给出中心极限定理在可用样本量下失效或产生误导的三种方式：(a) 方差
  （或均值）无穷的分布；(b) 同分布但不独立的样本；(c) 方差有限、但在中等样本量下偏斜很强的分布，此时
  Berry–Esseen 定理把正态近似的误差界定为 $C\rho/(\sigma^3\sqrt n)$，其中 $C$ 是一个绝对常数，
  $\rho = E|X - \mu|^3$。对这三种情况分别给出一个具体例子，并定性说明问题出在哪里。说明这如何关系到：
  (a) 小批量（minibatch）梯度噪声随批大小的缩放规律；(b) 当测试准确率接近 0% 或 100% 时，正态近似置信
  区间的可靠性。
- **Q7.** 对一个统计假设检验，定义：原假设与备择假设、检验统计量、p 值、显著性水平、检验效能（power）；
  并准确说明 p 值究竟是（以及不是）哪个概率。两个模型在同一个包含 $n$ 个样本的测试集上被评估。比较：
  (i) 配对检验（paired test），对每个样本利用两个模型是否一致这一信息（例如在不一致的样本上做 McNemar
  检验，或做配对自助法，paired bootstrap）；与 (ii) 把两个准确率估计当作独立样本处理的检验；说明当两个
  模型的正确性在样本之间正相关时，哪一种检验的效能更高，并推导能解释原因的方差比较。假设独立运行 20 次
  假设检验，每次显著性水平 $\alpha = 0.05$，且每个原假设实际上都成立：计算至少有一次（错误地）拒绝原
  假设的概率。描述 Bonferroni 校正与 Benjamini–Hochberg 过程，并说明它们各自控制的是什么量。
- **Q8.** 通过投掷一枚六面骰子 600 次来检验其是否公平，按面 $1, \dots, 6$ 的顺序记录到的计数为
  $[90, 110, 95, 105, 120, 80]$。在骰子公正这一原假设下，计算卡方拟合优度统计量
  $\chi^2 = \sum_i (O_i - E_i)^2 / E_i$，给出自由度（并说明为什么不是简单的面数），并计算 5% 显著性水平下
  的临界值以及 p 值。给出结论，并说明这个检验在什么情况下不可靠。说明这如何关系到检验某个采样器的输出是否
  符合目标分布。
- **Q9.** 给定两个浮点数数组 `a` 和 `b`，分别是来自两个未知一维分布的独立同分布样本，再给定一个浮点数
  `x`。给出一条规则及其论证依据，用来判断 `x` 更可能来自哪个数组，并说明除了标签之外还应该报告什么。
  然后取 $a$ 为来自 $\mathcal N(0, 1)$ 的样本、$b$ 为来自 $\mathcal N(0, 3^2)$ 的样本：在使用真实密度、且
  两个数组大小相等的情况下，求出被判给 $a$ 的 $x$ 的集合；当 $a$ 的样本数是 $b$ 的三倍时（类先验取自
  两个数组的大小），说明这个集合如何变化。说明仅依据到最近样本均值的距离来判断的规则在什么时候会失效，
  并把它与上面这两个数组联系起来。

### 信息论

- **Q10.** 对大小为 $K$ 的有限集合上的分布 $p$，定义熵（entropy）$H(p)$；对同一集合上的第二个分布 $q$，
  定义交叉熵（cross-entropy）$H(p, q)$。推导 $H(p, q)$ 与 $H(p)$、$\mathrm{KL}(p \,\|\, q)$ 之间的恒等式，
  并说明它对固定 $p$ 时最小化交叉熵意味着什么，以及最小化交叉熵与极大似然估计之间的关系。证明对任意 $p$
  都有 $H(p) \le \log K$，且等号成立当且仅当 $p$ 是均匀分布。说明用 $\log$ 和用 $\log_2$ 度量熵之间的
  关系。把困惑度（perplexity）定义为交叉熵的一个变换，并计算：(a) 每个 token 的交叉熵损失为 2.0 nat 的
  模型的困惑度；(b) 无论输入是什么，都给 $V$ 个可能 token 中的每一个赋予概率 $1/V$ 的模型的困惑度。
- **Q11.** 把两个随机变量之间的互信息（mutual information）$I(X; Y)$ 定义为一个 Kullback–Leibler 散度，
  并推导它的另外两种标准形式：一种用（边际与条件）熵表示，另一种用三个熵 $H(X)$、$H(Y)$、$H(X, Y)$ 表示。
  证明 $I(X; Y) \ge 0$，且等号成立当且仅当 $X$ 与 $Y$ 独立；并证明 $I(X; Y) = I(Y; X)$。对如下情形计算以
  比特为单位的 $I(X; Y)$：$X$ 是均匀随机的一个比特，$Y = X \oplus N$，其中独立噪声比特 $N$ 以概率 0.1
  取值为 1，这构成一条*二元对称信道*（binary symmetric channel）。说明互信息与下列二者的关系：(a) 决策
  树中用于选择划分（split）的信息增益（information gain）；(b) InfoNCE 对比损失给出的 $I(X; Y)$ 的下界。

### 微积分

- **Q12.** 手算下列各积分，并说明所用方法：$\int x e^x\,dx$；$x > 0$ 时的 $\int \ln x\,dx$；
  $\int dx / (1 + x^2)$，并由此求出 $\int_{-\infty}^{\infty} dx / (\pi(1 + x^2))$；
  $\int_{-\infty}^{\infty} e^{-x^2/2}\,dx$；以及直接从定义出发、把 $E[X]$（$X \sim \mathrm{Exp}(\lambda)$）
  写成积分来求值。给出这类积分在机器学习中的一个用途。

## 参考解答

<details>
<summary>展开参考解答</summary>

先向面试官确认 $\log$ 的约定（全文取自然对数），以及一个具名结论（Strassen 算法、Berry–Esseen 定理、
McNemar 检验）是可以直接引用，还是需要从第一性原理推导；下面的解答直接引用它们。

### 线性代数

**Q1.** $mnp$ 次乘加（约 $2mnp$ 次浮点运算：每一项一次乘法、一次加法），方阵情形下教科书算法是
$O(n^3)$。形成 $C = AB$ 的一个元素要把 $n$ 个乘积相加，而这样的元素共有 $mp$ 个，合计 $mnp$。Strassen
算法用 7 次子乘法而不是 8 次完成两个 $2 \times 2$ 分块矩阵的乘法，给出递归式
$T(n) = 7T(n/2) + O(n^2)$；由主定理，$T(n) = O(n^{\log_2 7}) \approx O(n^{2.807})$。对 $A$
（$1000 \times 10$）、$B$（$10 \times 1000$）、$v$（$1000 \times 1$）：先形成 $AB$ 需要
$1000 \cdot 10 \cdot 1000 = 10^7$ 次乘加，$(AB)v$ 再需要 $1000 \cdot 1000 = 10^6$ 次，合计
$1.1 \times 10^7$；先形成 $Bv$ 只需 $10 \cdot 1000 = 10^4$ 次，$A(Bv)$ 再需要 $1000 \cdot 10 = 10^4$
次，合计 $2 \times 10^4$——便宜 550 倍，尽管两种顺序算出的是完全相同的向量，因为矩阵乘法满足结合律。
ML：线性层 $y = xW$ 在一批 token 上的前向传播，$x \in \mathbb R^{B \times d_{in}}$、
$W \in \mathbb R^{d_{in} \times d_{out}}$，按同样的 $mnp$ 公式需要 $Bd_{in}d_{out}$ 次乘加，即
$2Bd_{in}d_{out}$ 次浮点运算；而低秩更新 $x(AB)$（$A \in \mathbb R^{d_{in} \times r}$、
$B \in \mathbb R^{r \times d_{out}}$，$r$ 很小，如 LoRA）应该按 $(xA)B$ 计算，原因与上面第二种顺序更
便宜完全相同。

**Q2.** $A \in \mathbb R^{m \times n}$ 的秩是其列空间的维数（等价地，也是其行空间的维数）；方阵
$A \in \mathbb R^{n \times n}$ 满秩是指 $\mathrm{rank}(A) = n$。$AB$ 的每一列都是 $A$ 各列的线性组合，
所以 $AB$ 的列空间包含于 $A$ 的列空间，给出 $\mathrm{rank}(AB) \le \mathrm{rank}(A)$；对称地，$AB$ 的
每一行都是 $B$ 各行的线性组合，给出 $\mathrm{rank}(AB) \le \mathrm{rank}(B)$，合起来就是
$\mathrm{rank}(AB) \le \min(\mathrm{rank}(A), \mathrm{rank}(B))$。$uv^\top$ 的每一列都是 $u$ 的标量倍
（第 $j$ 列是 $v_j u$），所以它的列空间就是 $u$ 张成的直线：当 $u, v$ 均非零时秩为 1。对方阵 $A$，下列
各条都与 $A$ 可逆等价：$\det(A) \ne 0$；$A$ 满秩为 $n$；只有 $x = 0$ 时才有 $Ax = 0$；$A$ 的列（或行）
线性无关；$0$ 不是 $A$ 的特征值；$A$ 的每个奇异值都非零。行列式之所以是糟糕的数值判据，是因为它把尺度
和数值条件混在了一起：$\det(0.1\, I_{100}) = 0.1^{100} = 10^{-100}$，任何固定的阈值都会把它当成零（在单精度下
它会下溢为恰好 $0$，而 $\det(0.1\, I_{400}) = 10^{-400}$ 即使在双精度下也会下溢为 $0$），而 $0.1\, I_{100}$ 的
每个奇异值都是 $0.1$，条件数已经好到极致。能把两者分开的判据是最小奇异值
$\sigma_{\min}$（当 $A$ 奇异、或数值上近乎奇异时，它恰好为零，或相对 $\sigma_{\max}$ 可忽略不计），或者
等价地条件数 $\kappa(A) = \sigma_{\max}/\sigma_{\min}$，它对尺度不敏感，且恰好在 $A$ 病态时很大，与
$\det(A)$ 无关。ML：对数据矩阵 $X \in \mathbb R^{n \times d}$，$X^\top X$ 奇异当且仅当存在非零 $v$ 使
$v^\top X^\top X v = \|Xv\|^2 = 0$，即 $Xv = 0$，即 $X$ 不是列满秩——这发生在特征共线、或特征数多于
样本数（$d > n$）时；岭回归通过加上 $\lambda I$ 来保证可逆。LoRA 更新 $BA$（$B \in \mathbb R^{d \times
r}$、$A \in \mathbb R^{r \times d}$）由上面的界，秩至多为 $r$，因为 $\mathrm{rank}(B) \le r$。

**Q3.** 把 $A$ 分解一次——通过带部分主元的高斯消元（Gaussian elimination with partial pivoting），即
$LU$ 分解 $PA = LU$（$P$ 为置换矩阵，$L, U$ 为三角矩阵）——耗时 $O(n^3)$（约 $\tfrac23 n^3$ 次浮点运算），
然后用一次前代和一次回代（各 $O(n^2)$）求解 $Ax = b$。显式求出 $A^{-1}$ 大约需要 $2n^3$ 次浮点运算
（同样的 $LU$ 分解，再加上针对单位矩阵每一列各做一次三角形求解）——大约是单纯分解的三倍——之后每个
右端项上 $A^{-1}b$ 还要花 $O(n^2)$，与多做一次三角形求解同一个量级；所以预先算出 $A^{-1}$ 完全没有
节省，只有更大的前期开销。除了运算次数，先分解再求解还更精确：它是向后稳定（backward stable）的，其
残差 $\|A\hat x - b\|$ 无论 $A$ 的条件数如何都停留在机器精度的量级，而显式求出并使用 $A^{-1}$ 之后的
残差会随 $A$ 的条件数增大；前向误差 $\|\hat x - x\|$ 在两条路线上都随条件数增大，但求逆路线的更大。在条件
良好的矩阵上两条路线结果接近，但在条件不好的矩阵上（下面用 Hilbert 矩阵检验），显式求逆既多花代价、又多出误差，
什么都没换来。当 $A$ 是对称
正定矩阵时，Cholesky 分解计算 $A = LL^\top$（$L$ 为三角矩阵），利用对称性把耗时降到约 $\tfrac13 n^3$
次浮点运算——比一般的 $LU$ 分解又快了一倍。ML：正规方程 $X^\top Xw = X^\top y$（当 $X$ 列满秩时）的
系数矩阵是对称正定的，所以用 Cholesky 求解，而不是求逆 $X^\top X$（当 $X$ 病态时，则对 $X$ 本身做 QR 分解，
因为构造 $X^\top X$ 会把条件数平方）；高斯过程后验是由（对称正定的）核
矩阵的 Cholesky 因子算出的；牛顿步 $Hd = -g$ 是通过分解 $H$ 再求解得到的，从不通过求出 $H^{-1}$。

**Q4.** 记 $A = U\Sigma V^\top$；伪逆是 $A^+ = V\Sigma^+U^\top$，其中 $\Sigma^+$ 把 $\Sigma$ 的形状转置，
并把每个非零奇异值 $\sigma_i$ 换成 $1/\sigma_i$，零元素保持为零。对任意 $x \in \mathbb R^n$，写
$x = Vc$（因为 $V$ 是一组正交基），于是 $Ax = U\Sigma V^\top Vc = U\Sigma c$，又因为 $U$ 是正交矩阵，
$\|Ax - b\|^2 = \|\Sigma c - U^\top b\|^2$——按 $c$ 的各个坐标可分离。对满足 $\sigma_i > 0$ 的坐标 $i$，
项 $(\sigma_i c_i - (U^\top b)_i)^2$ 唯一地在 $c_i = (U^\top b)_i / \sigma_i$ 处取最小；对满足
$\sigma_i = 0$ 的坐标，该项根本不依赖 $c_i$，所以任何 $c_i$ 都同样让 $\|Ax - b\|$ 最小，而由于
$\|x\|^2 = \|Vc\|^2 = \|c\|^2$（同样因为 $V$ 正交），范数最小的取值是 $c_i = 0$。综合两种情形得到
$c = \Sigma^+U^\top b$，即 $x = V\Sigma^+U^\top b = A^+b$：它既是 $\|Ax - b\|$ 的一个最小值点，又是
所有最小值点里范数最小的那一个。当 $A$ 列满秩时，每个奇异值都非零，$A^\top A = V\Sigma^\top\Sigma
V^\top$ 可逆，逆为 $V(\Sigma^\top\Sigma)^{-1}V^\top$，于是
$(A^\top A)^{-1}A^\top = V(\Sigma^\top\Sigma)^{-1}\Sigma^\top U^\top = V\Sigma^+U^\top = A^+$。对岭
极限，$(A^\top A + \lambda I)^{-1}A^\top = V(\Sigma^\top\Sigma + \lambda I)^{-1}\Sigma^\top U^\top$，
其第 $i$ 个对角元是 $\sigma_i / (\sigma_i^2 + \lambda)$：当 $\lambda \to 0^+$ 时，$\sigma_i > 0$ 的项
趋于 $1/\sigma_i$，$\sigma_i = 0$ 的项趋于 $0$（因为分子本来就是 $0$），恰好与 $\Sigma^+$ 一致。ML：
对欠定的线性回归（参数多于样本），有无穷多个权重向量恰好拟合训练数据，$A^+b$ 挑出其中范数最小的那个；
从 $w_0 = 0$ 出发、在最小二乘目标上做梯度下降，每一步都只在 $A$ 的行空间内移动（每个梯度
$A^\top(Aw - b)$ 都落在其中），所以在众多零残差解里，它收敛到唯一落在行空间中的那个——正是范数最小的
解 $A^+b$：这是优化过程本身带来的、偏向小权重的隐式偏置（implicit bias），不需要任何显式正则项。

### 概率与统计

**Q5.** 对 $X \sim \mathrm{Exp}(\lambda)$，对每个正整数 $k$ 都有 $E[X^k] = k!/\lambda^k$；它的偏度是 $2$，
超额峰度是 $6$（二者都与 $\lambda$ 无关）。矩母函数是 $M_X(t) = E[e^{tX}] = \int_0^\infty e^{tx}\lambda
e^{-\lambda x}\,dx = \lambda/(\lambda - t)$（$t < \lambda$），把 $e^{tX} = \sum_k (tX)^k/k!$ 在期望内
逐项展开，得到 $M_X(t) = \sum_k E[X^k]\,t^k/k!$，于是 $E[X^k] = M_X^{(k)}(0)$，即 $M_X$ 在 $t = 0$ 处的
第 $k$ 阶导数（这正好对应泰勒系数的定义）。直接代换 $u = \lambda x$，

$$E[X^k] = \int_0^\infty x^k \lambda e^{-\lambda x}\,dx = \frac{1}{\lambda^k}\int_0^\infty u^k e^{-u}\,du = \frac{k!}{\lambda^k},$$

这里用到对整数 $k$ 有 $\Gamma(k + 1) = k!$。由原点矩，$\mu = 1/\lambda$，
$\sigma^2 = E[X^2] - \mu^2 = 2/\lambda^2 - 1/\lambda^2 = 1/\lambda^2$；把中心矩用原点矩展开，
$E[(X - \mu)^3] = E[X^3] - 3\mu E[X^2] + 2\mu^3 = (6 - 6 + 2)/\lambda^3 = 2/\lambda^3$，
$E[(X - \mu)^4] = E[X^4] - 4\mu E[X^3] + 6\mu^2 E[X^2] - 3\mu^4 = (24 - 24 + 12 - 3)/\lambda^4 = 9/\lambda^4$，
再分别除以 $\sigma^3$ 和 $\sigma^4$，得到偏度 $2$、峰度 $9$，即超额峰度 $6$。有些分布完全没有矩：标准柯西分布的密度
$1/(\pi(1 + x^2))$ 满足 $E|X| = \infty$（尾部积分 $\int^M x/(\pi(1+x^2))\,dx$ 按 $\log M$ 增长），所以
它没有均值，因而更高阶的矩也不存在。另一些分布的矩只存在到有限阶：自由度为 $\nu$ 的 Student-$t$ 分布，
其尾部按 $|x|^{-\nu - 1}$ 衰减，所以 $E[|X|^k]$ 恰好在 $k < \nu$ 时有限——例如 $t_3$ 分布均值、方差有限，
但偏度、峰度都不存在。ML：Adam 内部保存的 $m_t$、$v_t$ 分别是梯度及其逐元素平方的指数滑动平均，也就是
梯度第一、第二原点矩的滚动估计；批归一化和层归一化用在一个批次上或在特征维度上算出的均值和方差，对激活值做中心化
和缩放，而 RMSNorm 只用二阶原点矩做缩放。

**Q6.** 弱大数定律：若 $X_1, \dots, X_n$ 独立同分布，且均值 $\mu$ 有限，则当 $n \to \infty$ 时
$\bar X_n \to \mu$（依概率收敛）。中心极限定理：若进一步有 $\mathrm{Var}(X_i) = \sigma^2 < \infty$，则
$\sqrt n(\bar X_n - \mu)/\sigma$ 依分布收敛到 $N(0, 1)$。在有限的 $n$ 下，中心极限定理可能失效或产生
误导，有三种方式：(a) *方差（或均值）无穷*——对独立同分布的标准柯西样本，$\bar X_n$ 对每个 $n$ 都恰好
服从标准柯西分布（下面通过它的四分位距来检验，该四分位距始终停留在 $2$，并不随 $n$ 收缩），因为均值
和方差都不存在，大数定律与中心极限定理都无从谈起；(b) *不独立*——对具有公共方差 $\sigma^2$、两两相关
系数为 $\rho > 0$ 的等相关（equicorrelated）$X_i$，

$$\mathrm{Var}(\bar X_n) = \frac1{n^2}\Bigl(\sum_i \mathrm{Var}(X_i) + \sum_{i \ne j}\mathrm{Cov}(X_i, X_j)\Bigr) = \frac1{n^2}\bigl(n\sigma^2 + n(n-1)\rho\sigma^2\bigr) = \frac{\sigma^2}{n}\bigl(1 + (n-1)\rho\bigr) \xrightarrow{n \to \infty} \rho\sigma^2 \ne 0,$$

所以样本均值永远不会集中到 $\mu$——无论 $n$ 多大，它的方差都保持在 $\rho\sigma^2$ 之上——因为 $X_i$ 中共享的
成分永远不会被平均掉；(c) *中等样本量下的
强偏斜*——Berry–Esseen 定理把标准化和的分布函数与 $\Phi$ 之间的偏差界定为 $C\rho/(\sigma^3\sqrt n)$，
所以只要 $\rho/\sigma^3$（与偏度密切相关）很大，这个界就很大，于是相比对称分布，强偏斜的分布需要大得
多的 $n$，其标准化和才会看起来像高斯分布；具体来说，偏度为 $\gamma_1$ 的独立同分布样本，其 $\bar X_n$
本身的偏度是 $\gamma_1/\sqrt n$（下面验证），对指数分布而言，即使 $n = 20$，这个值也仍远离 $0$。ML：
小批量梯度噪声在主导阶上是 $B$ 个近似独立的单样本梯度贡献的平均，所以其标准差按 $1/\sqrt B$ 缩放（$B$ 个
独立项的均值方差为 $\sigma^2/B$），并且由中心极限定理，其分布接近高斯分布；而测试准确率 $\hat p$ 的正态近似置信区间依赖的正是同一个中心极限定理，当
$\hat p$ 接近 $0$ 或 $1$ 且 $n$ 不太大时，它对真实的、明显偏斜的二项分布近似得很差。

**Q7.** $H_0$ 是被检验的陈述（例如“骰子是公平的”）；$H_1$ 是若 $H_0$ 被拒绝后将得出的结论。检验统计量
$T$ 在 $H_0$ 下的分布已知或有很好的近似；p 值是 $P(T$ 至少与观测值一样极端 $\mid H_0)$——在原假设成立、
且实验被假想重复进行的前提下，统计量达到这么极端的概率，而不是 $H_0$ 本身为真的概率（后者需要先验）。
显著性水平 $\alpha$ 是该检验被校准到的第一类错误率，通过在 $p<\alpha$ 时拒绝原假设来实现。第二类错误率
$\beta$ 是在某个具体的 $H_1$ 成立时未能拒绝 $H_0$ 的概率，检验效能 $1-\beta$ 则是这种情况下正确拒绝
$H_0$ 的概率。

记 $C_i^{(1)}, C_i^{(2)}\in\{0,1\}$ 表示两个模型在样本 $i$ 上是否正确，$D_i=C_i^{(1)}-C_i^{(2)}$。点估计
$\bar D$ 无论哪种做法都相同，但方差不同：配对做法用的是

$$\mathrm{Var}(\bar D) = \tfrac1n\bigl(\mathrm{Var}(C^{(1)}_i)+\mathrm{Var}(C^{(2)}_i)-2\,\mathrm{Cov}(C^{(1)}_i,C^{(2)}_i)\bigr),$$

而把两个准确率当作独立样本处理时，则完全省略了 $-2\,\mathrm{Cov}/n$ 这一项。两个模型往往在哪些样本容易、
哪些样本难上超出偶然地保持一致，所以实践中 $\mathrm{Cov}>0$；独立样本方差因而高估了真实方差，使得检验
效能低于正确的配对检验（在不一致样本计数 $n_{10},n_{01}$ 上做的 McNemar 检验，统计量为
$(n_{10}-n_{01})^2/(n_{10}+n_{01})$，在 $H_0$ 下近似服从 $\chi^2_1$；或者等价的配对自助法）。如果反过来
$\mathrm{Cov}<0$，独立样本方差就偏小，会过于频繁地拒绝一个成立的 $H_0$；无论哪种情形，配对方差才是
正确的。

若独立运行 20 次假设检验，每次 $\alpha=0.05$，且每个原假设实际上都成立，则全部 20 次都不拒绝的概率是
$0.95^{20}\approx0.358$，所以至少有一次错误拒绝的概率是 $1-0.95^{20}\approx0.642$。Bonferroni 校正只在
$p_i<\alpha/m$ 时才拒绝第 $i$ 个检验；由联合界（union bound），
$P(\text{任何一次错误拒绝})\le\sum_i P(p_i<\alpha/m)=\alpha$，把族错误率（family-wise error rate）控制在
$\alpha$ 以内，代价是当 $m$ 较大时效能降低。Benjamini–Hochberg 过程则是把 p 值排序为
$p_{(1)}\le\dots\le p_{(m)}$，找出满足 $p_{(k)}\le(k/m)\alpha$ 的最大 $k$，并拒绝
$H_{(1)},\dots,H_{(k)}$；它控制的是错误发现率（false discovery rate）——被拒绝的假设中假阳性所占比例的
期望值——不超过 $\alpha$，而不是控制任何假阳性发生的概率。

**Q8.** 掷 600 次公平骰子，每个面的期望计数是 $E_i = 100$；统计量为

$$\chi^2 = \sum_{i=1}^6 \frac{(O_i - E_i)^2}{E_i} = \frac{10^2 + 10^2 + 5^2 + 5^2 + 20^2 + 20^2}{100} = \frac{100 + 100 + 25 + 25 + 400 + 400}{100} = 10.5.$$

自由度是 $6 - 1 = 5$，而不是 $6$，因为六个观测计数不能各自自由变化：它们之和必须等于掷骰总次数，所以六个偏差
满足一条线性约束 $\sum_i (O_i - E_i) = 600 - 600 = 0$（而且这里没有从数据中估计原假设分布的任何参数，否则每估计
一个参数，还要再减去一个自由度）。在 5% 水平下，临界值是
$\chi^2_{0.95, 5} \approx 11.07$，p 值是 $P(\chi^2_5 \ge 10.5) \approx 0.062$。由于 $10.5 < 11.07$
（等价地，$0.062 > 0.05$），该检验在 5% 水平下不拒绝骰子公平的原假设，不过结果已经很接近边界，若取
10% 水平则会得出不同的判断。卡方近似依赖于每个期望计数都足够大（常见经验法则：至少为 5，且最多约
五分之一的格子可以违反这一点）；当期望计数较小时，统计量的真实抽样分布会偏离 $\chi^2_5$ 曲线，此时
应改用精确的多项式检验，或对原假设做模拟。ML：同样的统计量可以用来检验某个生成式或采样过程产生的
样本，是否符合某个已知的、定义在离散结果集合上的目标分布——做法是把样本分箱，再比较观测计数与期望
计数。

**Q9.** 把数组来源看作一个潜在类别 $C\in\{a,b\}$，先验为 $\pi_a,\pi_b$（如无特别说明，取两个数组各自大小
的比例），类条件密度 $p_a,p_b$ 从各自数组估计得到——可以拟合某个参数族（例如用样本均值和方差拟合一个高
斯分布）、用核密度估计，或者换成最近邻规则；贝叶斯最优规则把 $x$ 判给后验
$P(C\mid x)\propto\pi_C p_C(x)$ 较大的类别，而应当报告的是后验本身，

$$P(a\mid x) = \frac{\pi_a p_a(x)}{\pi_a p_a(x) + \pi_b p_b(x)},$$

而不仅仅是取最大值对应的标签。若用高斯拟合且两者方差不等，取对数后边界 $\pi_a p_a(x)=\pi_b p_b(x)$ 在
$x$ 中是二次的，因此一般有两个根而非一个：一个数组占据中间区间，另一个占据两侧尾部，具体哪个占中间取决
于哪个方差更大。

取 $a \sim N(0, 1)$、$b \sim N(0, 3^2)$，且两个数组大小相等（$\pi_a = \pi_b = \tfrac12$），边界方程
$\tfrac1{\sqrt{2\pi}} e^{-x^2/2} = \tfrac1{3\sqrt{2\pi}} e^{-x^2/18}$ 两边乘以 $3\sqrt{2\pi}$ 再取对数，
化简为 $\ln 3 = x^2\bigl(\tfrac12 - \tfrac1{18}\bigr) = \tfrac{4x^2}9$，即
$x = \pm\tfrac32\sqrt{\ln 3} \approx \pm 1.572$；由于 $a$ 是更集中的密度，它在中间区间占优，所以 $x$
被判给 $a$ 当且仅当 $|x| < 1.5\sqrt{\ln 3}$。当 $a$ 的样本数是 $b$ 的三倍时，先验变为
$\pi_a = \tfrac34, \pi_b = \tfrac14$，同样的步骤给出 $\tfrac{\pi_b}{\pi_a} = \tfrac13 = 3e^{-4x^2/9}$，
即 $x^2 = \tfrac94\ln 9$，阈值扩大为 $x = \pm 1.5\sqrt{\ln 9} \approx \pm 2.223$：更有利于 $a$ 的先验
扩大了判给 $a$ 的区域。

仅依据到最近样本均值的距离这条规则，会在两个均值恰好相同时失效，正如本例（两个均值都是 $0$）：每个 $x$
到两个均值的距离都相等，规则完全无法区分，尽管这两个分布按其离散程度其实很容易分辨。同样的后验规则正是
分布外检测（out-of-distribution detection）背后的原理，也正是高斯判别分析（Gaussian discriminant
analysis）和朴素贝叶斯（naive Bayes）从各类别分别拟合的类条件密度中计算出来的东西。

### 信息论

**Q10.** 对大小为 $K$ 的有限集合上的分布 $p$，$H(p) = -\sum_x p(x)\log p(x) = E_p[-\log p(X)]$；对同一
集合上的第二个分布 $q$，$H(p, q) = -\sum_x p(x)\log q(x) = E_p[-\log q(X)]$。在求和号内把 $\log q(x)$
拆成 $\log p(x) - \log\frac{p(x)}{q(x)}$，

$$H(p, q) = -\sum_x p(x)\log p(x) - \sum_x p(x)\log\frac{q(x)}{p(x)} = H(p) + \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p) + \mathrm{KL}(p \,\|\, q).$$

由于 $H(p)$ 不依赖于 $q$，对固定的 $p$ 在 $q$ 上最小化 $H(p, q)$，恰好就是在最小化
$\mathrm{KL}(p \,\|\, q)$；当 $p$ 是真实（或经验）数据分布、$q_\theta$ 是一个模型时，大数定律给出：
对训练样本 $x_i \sim p$，$-\frac1n\sum_i\log q_\theta(x_i) \to H(p, q_\theta)$，所以在 $\theta$ 上最小化
平均负对数似然——即经验交叉熵——就是在最小化交叉熵，这正是极大似然估计。关于最大值：
$\mathrm{KL}(p \,\|\, \text{均匀分布}) = \sum_x p(x)\log\bigl(p(x)K\bigr) = -H(p) + \log K$（用到
$\sum_x p(x) = 1$），而由 KL 的非负性这个量 $\ge 0$，所以 $H(p) \le \log K$，等号成立当且仅当 $p$ 是
均匀分布（KL 取等号的情形）。用 $\log_2$（比特）度量的熵，就是用 $\log$（nat）度量的熵除以 $\log 2$，
因为 $\log_2 x = \log x/\log 2$。当交叉熵以 nat 为单位时，困惑度是 $\mathrm{PPL} = \exp\bigl(H(p,
q)\bigr)$：每个 token 的交叉熵损失为 $2.0$ nat 的模型，其困惑度为 $e^2 \approx 7.39$；而当 $q$ 是 $V$
个 token 上的均匀分布时，对*任意* $p$ 都有 $H(p, q) = -\sum_x p(x)\log(1/V) = \log V$（$\log(1/V)$
可以整体提到求和号外），所以均匀模型的困惑度恰好是 $V$，与真实的 token 分布无关。
ML：交叉熵正是用极大似然训练分类器或语言模型时被最小化的损失，而从业者实际汇报的是困惑度而不是原始
损失值，恰恰是因为困惑度可以直接读作一个有效词表大小。

**Q11.** $I(X; Y) = \mathrm{KL}\bigl(p(x,y) \,\|\, p(x)p(y)\bigr) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$。
把 $\frac{p(x,y)}{p(x)p(y)}$ 写成 $\frac{p(x \mid y)}{p(x)}$，

$$I(X; Y) = \sum_{x,y} p(x,y)\log\frac{p(x \mid y)}{p(x)} = -\sum_{x,y} p(x,y)\log p(x) + \sum_{x,y} p(x,y)\log p(x \mid y) = H(X) - H(X \mid Y),$$

其中 $H(X \mid Y) = -\sum_{x,y}p(x,y)\log p(x\mid y) = E_Y[H(X \mid Y = y)]$ 是条件熵。链式法则
$H(X, Y) = H(Y) + H(X \mid Y)$ 由 $p(x,y) = p(y)p(x\mid y)$ 两边取 $-\log$ 再对 $p(x,y)$ 求期望得到；
把 $H(X\mid Y) = H(X,Y) - H(Y)$ 代入，得到第三种形式，

$$I(X; Y) = H(X) - \bigl(H(X,Y) - H(Y)\bigr) = H(X) + H(Y) - H(X, Y),$$

这个表达式在 $X, Y$ 之间显然对称，所以 $I(X; Y) = I(Y; X)$。非负性 $I(X; Y) \ge 0$（等号成立当且仅当
$p(x,y) = p(x)p(y)$ 几乎处处成立，即 $X \perp Y$）直接来自把 KL 的非负性应用于 $p(x,y)$ 与 $p(x)p(y)$。
对于二元对称信道，$X \sim \mathrm{Bernoulli}(1/2)$，$Y = X \oplus N$，其中
$N \sim \mathrm{Bernoulli}(0.1)$ 与 $X$ 独立：由于 $X$ 均匀且信道对称，$Y$ 也均匀，所以 $H(Y) = 1$ 比特；
而无论 $X$ 取何值，$Y = X \oplus N$ 的熵恰好就是 $N$ 的熵，所以 $H(Y \mid X) = H_b(0.1)$，即 $0.1$ 的
二元熵（以比特为单位）。于是

$$I(X; Y) = H(Y) - H(Y \mid X) = 1 - H_b(0.1) \approx 1 - 0.469 = 0.531 \text{ 比特}.$$

ML：决策树中用来选择划分的信息增益，恰好就是划分变量与标签之间的互信息，
$I(\text{split}; Y) = H(Y) - H(Y \mid \text{split})$；而 InfoNCE 对比损失 $L$（对每个锚点用一个正样本
对和 $N-1$ 个负样本算出）满足 $I(X; Y) \ge \log N - L$，所以降低 $L$ 就是在推高 $X, Y$ 之间互信息的
一个下界。

### 微积分

**Q12.** $\int xe^x\,dx = (x - 1)e^x + C$：用分部积分，取 $u = x$，$dv = e^x\,dx$（于是 $du = dx$，
$v = e^x$），得到 $\int xe^x\,dx = xe^x - \int e^x\,dx = xe^x - e^x + C = (x-1)e^x + C$。$x > 0$ 时
$\int\ln x\,dx = x\ln x - x + C$：分部积分取 $u = \ln x$，$dv = dx$（$du = dx/x$，$v = x$），
$\int \ln x\,dx = x\ln x - \int x \cdot \tfrac1x\,dx = x\ln x - x + C$。$\int\frac{dx}{1 + x^2} = \arctan
x + C$，因为 $\frac{d}{dx}\arctan x = \frac1{1+x^2}$；由此柯西密度积分为 $1$，

$$\int_{-\infty}^{\infty} \frac{dx}{\pi(1 + x^2)} = \frac1\pi\Bigl[\arctan x\Bigr]_{-\infty}^{\infty} = \frac1\pi\Bigl(\frac\pi2 - \bigl(-\frac\pi2\bigr)\Bigr) = 1.$$

$\int_{-\infty}^{\infty} e^{-x^2/2}\,dx = \sqrt{2\pi}$：把这个积分记作 $I$ 并对它平方，

$$I^2 = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^2 + y^2)/2}\,dx\,dy = \int_0^{2\pi}\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta = 2\pi\Bigl[-e^{-r^2/2}\Bigr]_0^\infty = 2\pi,$$

换成极坐标（$x = r\cos\theta$，$y = r\sin\theta$，面积元 $r\,dr\,d\theta$），于是 $I = \sqrt{2\pi}$，
取正根，因为被积函数处处为正。对 $X \sim \mathrm{Exp}(\lambda)$，$E[X] = 1/\lambda$：分部积分取
$u = x$，$dv = \lambda e^{-\lambda x}\,dx$（$v = -e^{-\lambda x}$），

$$E[X] = \int_0^\infty x\lambda e^{-\lambda x}\,dx = \Bigl[-xe^{-\lambda x}\Bigr]_0^\infty + \int_0^\infty e^{-\lambda x}\,dx = 0 + \frac1\lambda = \frac1\lambda,$$

这与 Q5 中 $E[X^k] = k!/\lambda^k$ 在 $k = 1$ 时的结果一致。ML：正是这一类积分被用来对密度做归一化
（验证或推导某个密度积分为 $1$，如上面的柯西分布与高斯分布），并用于计算出现在似然函数及其梯度中的
期望。

<details>
<summary>验证代码（可运行）</summary>

```python
import math

import numpy as np
from scipy import integrate, optimize, stats
from scipy.linalg import cho_factor, cho_solve, hilbert, lu_factor, lu_solve

rng = np.random.default_rng(2026)


# ---- Q1: multiply-add counts by explicit loops and by the m n p formula, associativity, ML costs
def multiply_adds_and_product(A, B):
    """The schoolbook algorithm computed by explicit loops, counting one multiply-add per inner step."""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2
    C = np.zeros((m, p))
    count = 0
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
                count += 1
            C[i, j] = s
    return count, C


for (m1, n1, p1) in [(3, 4, 5), (5, 2, 6)]:
    A1, B1 = rng.standard_normal((m1, n1)), rng.standard_normal((n1, p1))
    count1, C1 = multiply_adds_and_product(A1, B1)
    assert count1 == m1 * n1 * p1
    assert np.allclose(C1, A1 @ B1)

assert math.isclose(math.log2(7), 2.807, abs_tol=5e-4)          # Strassen's exponent

m_big, n_big, p_big = 1000, 10, 1000
A_big = rng.standard_normal((m_big, n_big))
B_big, v_big = rng.standard_normal((n_big, p_big)), rng.standard_normal(p_big)
cost_ab_then_v = m_big * n_big * p_big + m_big * p_big          # form AB (m x p), then (AB) v
cost_bv_then_a = n_big * p_big + m_big * n_big                  # form Bv (n,), then A (Bv)
assert cost_ab_then_v == 11_000_000
assert cost_bv_then_a == 20_000
assert np.allclose((A_big @ B_big) @ v_big, A_big @ (B_big @ v_big), atol=1e-6)   # associativity

batch_tok, d_in, d_out = 6, 5, 4                                 # ML: forward pass of a linear layer
X_tok, W_layer = rng.standard_normal((batch_tok, d_in)), rng.standard_normal((d_in, d_out))
count_layer, _ = multiply_adds_and_product(X_tok, W_layer)
assert count_layer == batch_tok * d_in * d_out

batch_lr, d_lr, r_lr, d_out_lr = 50, 100, 4, 100                 # ML: low-rank update x (A B) as (x A) B
X_lr = rng.standard_normal((batch_lr, d_lr))
A_lr, B_lr = rng.standard_normal((d_lr, r_lr)), rng.standard_normal((r_lr, d_out_lr))
cost_full_lr = d_lr * r_lr * d_out_lr + batch_lr * d_lr * d_out_lr
cost_factored_lr = batch_lr * d_lr * r_lr + batch_lr * r_lr * d_out_lr
assert cost_full_lr / cost_factored_lr == 13.5
assert np.allclose(X_lr @ (A_lr @ B_lr), (X_lr @ A_lr) @ B_lr, atol=1e-6)

# ---- Q2: rank, rank(AB) <= min(rank A, rank B), outer products, det vs conditioning, ML consequences
for _ in range(200):
    m2, k2, p2 = (int(rng.integers(2, 6)) for _ in range(3))
    A2, B2 = rng.standard_normal((m2, k2)), rng.standard_normal((k2, p2))
    assert np.linalg.matrix_rank(A2 @ B2) <= min(np.linalg.matrix_rank(A2), np.linalg.matrix_rank(B2))

u_out, v_out = rng.standard_normal(5), rng.standard_normal(4)
assert np.linalg.matrix_rank(np.outer(u_out, v_out)) == 1
assert np.linalg.matrix_rank(np.outer(u_out, v_out) @ rng.standard_normal((4, 6))) == 1   # still <= 1

det_small = np.linalg.det(0.1 * np.eye(100))
assert math.isclose(det_small, 1e-100, rel_tol=1e-6)                     # representable in float64, yet "tiny"
assert np.linalg.det(0.1 * np.eye(100, dtype=np.float32)) == 0.0         # NOTE: 1e-100 underflows to 0 in float32
assert np.linalg.det(0.1 * np.eye(400)) == 0.0                           # and 1e-400 underflows even in float64
assert math.isclose(np.linalg.cond(0.1 * np.eye(100)), 1.0, rel_tol=1e-9)  # perfectly conditioned regardless
assert np.allclose(np.linalg.svd(0.1 * np.eye(100), compute_uv=False), 0.1)

n_feat, d_feat = 30, 5                                            # ML: X^T X invertible iff X full column rank
X_full = rng.standard_normal((n_feat, d_feat))
assert np.linalg.matrix_rank(X_full) == d_feat and np.linalg.cond(X_full.T @ X_full) < 1e10
X_collinear = X_full.copy()
X_collinear[:, -1] = 2.0 * X_collinear[:, 0]                      # collinear feature
assert np.linalg.matrix_rank(X_collinear) == d_feat - 1 and np.linalg.cond(X_collinear.T @ X_collinear) > 1e12
X_wide_feat = rng.standard_normal((5, 20))                         # more features than examples
assert np.linalg.matrix_rank(X_wide_feat) == 5 and np.linalg.cond(X_wide_feat.T @ X_wide_feat) > 1e12

d_lora, r_lora = 64, 4                                             # ML: a LoRA update B A has rank <= r
A_lora, B_lora = rng.standard_normal((r_lora, d_lora)), rng.standard_normal((d_lora, r_lora))
assert np.linalg.matrix_rank(B_lora @ A_lora) <= r_lora

# ---- Q3: solving A x = b, LU/Cholesky factor-and-solve vs the explicit inverse
assert (2 * 200 ** 3) / (2 * 200 ** 3 / 3) == 3.0                  # LU ~ 2n^3/3 flops, inversion ~ 2n^3

n_wc = 200
M_wc = rng.standard_normal((n_wc, n_wc))
A_wc = M_wc @ M_wc.T + n_wc * np.eye(n_wc)                         # a well-conditioned SPD matrix
x_true_wc = rng.standard_normal(n_wc)
b_wc = A_wc @ x_true_wc
lu_wc, piv_wc = lu_factor(A_wc)
x_lu_wc = lu_solve((lu_wc, piv_wc), b_wc)
chol_wc, lower_wc = cho_factor(A_wc)
x_cho_wc = cho_solve((chol_wc, lower_wc), b_wc)
x_inv_wc = np.linalg.inv(A_wc) @ b_wc
assert np.linalg.norm(A_wc @ x_lu_wc - b_wc) < 1e-8               # well conditioned: factor-and-solve...
assert np.linalg.norm(A_wc @ x_inv_wc - b_wc) < 1e-8               # ...and the explicit inverse agree closely
assert np.allclose(x_lu_wc, x_true_wc, atol=1e-8) and np.allclose(x_cho_wc, x_true_wc, atol=1e-8)

# NOTE: on a badly conditioned matrix the inverse route is both more expensive and less accurate; average
# over many right-hand sides at each size, since a single one is a noisy comparison
reps_hilbert = 200
err_lu_by_n = {}
for n_h in (6, 8, 10):
    A_h = hilbert(n_h)
    X_true_h = rng.standard_normal((n_h, reps_hilbert))
    B_h = A_h @ X_true_h
    lu_h, piv_h = lu_factor(A_h)
    X_lu_h = lu_solve((lu_h, piv_h), B_h)
    X_inv_h = np.linalg.inv(A_h) @ B_h                             # NOTE: the less accurate route -- see below
    res_lu_h = np.linalg.norm(A_h @ X_lu_h - B_h, axis=0).mean()
    res_inv_h = np.linalg.norm(A_h @ X_inv_h - B_h, axis=0).mean()
    err_lu_h = (np.linalg.norm(X_lu_h - X_true_h, axis=0) / np.linalg.norm(X_true_h, axis=0)).mean()
    err_inv_h = (np.linalg.norm(X_inv_h - X_true_h, axis=0) / np.linalg.norm(X_true_h, axis=0)).mean()
    assert res_lu_h < 1e-10                                        # LU stays backward stable regardless of cond
    assert res_inv_h > 1e4 * res_lu_h                               # the inverse route's residual does not
    assert err_inv_h > 1.8 * err_lu_h                               # and its forward error is markedly worse
    err_lu_by_n[n_h] = err_lu_h
assert err_lu_by_n[6] < err_lu_by_n[8] < err_lu_by_n[10]            # though LU's forward error also grows with cond
assert math.isclose(np.linalg.cond(X_full.T @ X_full), np.linalg.cond(X_full) ** 2, rel_tol=1e-6)  # X^T X squares it

# ---- Q4: Moore-Penrose pseudo-inverse: SVD formula, Penrose conditions, min-norm, ridge limit, GD
def pinv_from_svd(A, tol=1e-10):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.where(S > tol * S.max(), 1.0 / S, 0.0)
    return (Vt.T * S_inv) @ U.T


for shape4 in ((8, 5), (5, 8), (6, 6)):                             # tall, wide, square
    A4 = rng.standard_normal(shape4)
    Ap4 = np.linalg.pinv(A4)
    assert np.allclose(pinv_from_svd(A4), Ap4, atol=1e-8)
    assert np.allclose(A4 @ Ap4 @ A4, A4, atol=1e-8)                # the four Penrose conditions
    assert np.allclose(Ap4 @ A4 @ Ap4, Ap4, atol=1e-8)
    assert np.allclose((A4 @ Ap4).T, A4 @ Ap4, atol=1e-8)
    assert np.allclose((Ap4 @ A4).T, Ap4 @ A4, atol=1e-8)

A_tall = rng.standard_normal((9, 4))                                # full column rank: A+ = (A^T A)^-1 A^T
assert np.allclose(np.linalg.pinv(A_tall), np.linalg.inv(A_tall.T @ A_tall) @ A_tall.T, atol=1e-8)

m_wide, n_wide = 4, 9                                               # underdetermined: min-norm property
A_wide = rng.standard_normal((m_wide, n_wide))
b_wide = rng.standard_normal(m_wide)
x_pinv = np.linalg.pinv(A_wide) @ b_wide
res_pinv = np.linalg.norm(A_wide @ x_pinv - b_wide)
null_vec = np.linalg.svd(A_wide)[2][m_wide:][0]     # NOTE: svd returns V^T, so its rows are the directions of V
assert np.allclose(A_wide @ null_vec, 0, atol=1e-8)
for scale in (0.3, 1.0, -2.0):
    x_perturbed = x_pinv + scale * null_vec
    assert math.isclose(np.linalg.norm(A_wide @ x_perturbed - b_wide), res_pinv, abs_tol=1e-8)  # same residual
    assert math.isclose(np.linalg.norm(x_perturbed) ** 2, np.linalg.norm(x_pinv) ** 2 + scale ** 2,
                        rel_tol=1e-9)                                     # larger norm: orthogonal, by Pythagoras

for lam4 in (1e-2, 1e-4, 1e-6, 1e-8):                                # the ridge limit
    ridge_est = np.linalg.inv(A_wide.T @ A_wide + lam4 * np.eye(n_wide)) @ A_wide.T
    assert np.linalg.norm(ridge_est - np.linalg.pinv(A_wide)) < 5 * lam4 ** 0.5

w4 = np.zeros(n_wide)                                                # GD from 0 converges to the min-norm solution
lipschitz4 = np.linalg.eigvalsh(A_wide.T @ A_wide).max()
step4 = 1.0 / lipschitz4
for _ in range(20_000):
    w4 = w4 - step4 * (A_wide.T @ (A_wide @ w4 - b_wide))
assert np.allclose(w4, x_pinv, atol=1e-4)

# ---- Q5: raw and central moments, the MGF, distributions without moments, the exponential's moments
lam5 = 2.0


def mgf_exp(t, lam=lam5):
    return lam / (lam - t)


h5 = 1e-3
f5 = {k: mgf_exp(k * h5) for k in (-2, -1, 0, 1, 2)}
d1 = (f5[1] - f5[-1]) / (2 * h5)
d2 = (f5[1] - 2 * f5[0] + f5[-1]) / h5 ** 2
d3 = (f5[2] - 2 * f5[1] + 2 * f5[-1] - f5[-2]) / (2 * h5 ** 3)
d4 = (f5[2] - 4 * f5[1] + 6 * f5[0] - 4 * f5[-1] + f5[-2]) / h5 ** 4
for k5, dk in zip((1, 2, 3, 4), (d1, d2, d3, d4)):
    assert math.isclose(dk, math.factorial(k5) / lam5 ** k5, rel_tol=2e-2)            # MGF derivatives at 0
    quad_val5, _ = integrate.quad(lambda x, k5=k5: x ** k5 * lam5 * math.exp(-lam5 * x), 0, np.inf)
    assert math.isclose(quad_val5, math.factorial(k5) / lam5 ** k5, rel_tol=1e-8)      # and by direct integration

raw5 = {k5: math.factorial(k5) / lam5 ** k5 for k5 in (1, 2, 3, 4)}                   # central moments from raw ones
mu5 = raw5[1]
assert math.isclose(raw5[3] - 3 * mu5 * raw5[2] + 2 * mu5 ** 3, 2 / lam5 ** 3, rel_tol=1e-12)
assert math.isclose(raw5[4] - 4 * mu5 * raw5[3] + 6 * mu5 ** 2 * raw5[2] - 3 * mu5 ** 4, 9 / lam5 ** 4, rel_tol=1e-12)
_, _, skew5, kurt5 = stats.expon(scale=1 / lam5).stats(moments="mvsk")
assert math.isclose(float(skew5), 2.0, rel_tol=1e-8)
assert math.isclose(float(kurt5), 6.0, rel_tol=1e-8)                                  # excess kurtosis

mvsk_t3 = stats.t(df=3).stats(moments="mvsk")                                         # moments exist only for k < nu
assert math.isclose(float(mvsk_t3[1]), 3.0, rel_tol=1e-8)                             # variance (k=2 < 3): exists
assert not np.isfinite(float(mvsk_t3[3]))               # NOTE: a non-existent moment reports as inf/nan, not an error
cauchy_tail = [integrate.quad(lambda x: abs(x) / (math.pi * (1 + x ** 2)), -c, c)[0] for c in (1e2, 1e4, 1e6)]
assert np.all(np.diff(cauchy_tail) > 0.5)                                             # E|X| unbounded: no mean

# ---- Q6: LLN and CLT, and three ways they fail or mislead
reps6 = 8_000
for n6 in (10, 100, 1_000):
    q1c, q3c = np.percentile(rng.standard_cauchy((reps6, n6)).mean(axis=1), [25, 75])
    assert math.isclose(q3c - q1c, 2.0, abs_tol=0.2)                        # no shrinkage at all with n
    q1n, q3n = np.percentile(rng.standard_normal((reps6, n6)).mean(axis=1), [25, 75])
    assert math.isclose(q3n - q1n, 2 * stats.norm.ppf(0.75) / math.sqrt(n6), rel_tol=0.15)   # shrinks like 1/sqrt(n)

rho6 = 0.3                                                                  # strong dependence: a shared factor
reps6b = 30_000
for n6b in (5, 20, 100, 400):
    common6 = rng.standard_normal(reps6b)
    X6 = math.sqrt(rho6) * common6[:, None] + math.sqrt(1 - rho6) * rng.standard_normal((reps6b, n6b))
    var_mean6 = X6.mean(axis=1).var()
    assert math.isclose(var_mean6, (1 + (n6b - 1) * rho6) / n6b, rel_tol=0.1)
assert var_mean6 > 10 / n6b                                                 # far from the iid rate sigma^2 / n

gamma1_exp = 2.0                                                            # heavy skew at moderate n (Berry-Esseen)
reps6c = 80_000
for n6c in (5, 20, 80):
    means6c = rng.exponential(1.0, (reps6c, n6c)).mean(axis=1)
    assert math.isclose(stats.skew(means6c) * math.sqrt(n6c), gamma1_exp, rel_tol=0.15)
    # the mean of n Exp(1) variables is Gamma(n, scale 1/n): its skewness is exactly 2 / sqrt(n)
    assert math.isclose(float(stats.gamma(a=n6c, scale=1 / n6c).stats(moments="s")), gamma1_exp / math.sqrt(n6c))

# ---- Q7: p-values, familywise error and Bonferroni, and paired versus unpaired power
n7, trials7 = 30, 40_000
z7 = rng.standard_normal((trials7, n7)).mean(axis=1) * math.sqrt(n7)
p7 = 2 * (1 - stats.norm.cdf(np.abs(z7)))
assert stats.kstest(p7, "uniform").pvalue > 0.05 and math.isclose((p7 < 0.05).mean(), 0.05, abs_tol=0.01)

m7, trials7b = 20, 20_000                                                    # 20 independent true nulls
pm7 = 2 * (1 - stats.norm.cdf(np.abs(rng.standard_normal((trials7b, m7, n7)).mean(axis=2) * math.sqrt(n7))))
assert math.isclose(1 - 0.95 ** 20, 0.6415, abs_tol=5e-4)
assert math.isclose((pm7 < 0.05).any(axis=1).mean(), 1 - 0.95 ** 20, abs_tol=0.02)      # naive: much worse than 5%
assert (pm7 < 0.05 / m7).any(axis=1).mean() < 0.07                                      # Bonferroni controls it


def correlated_bernoulli(n, p_1, p_2, corr, size, rng):
    """Two Bernoulli(p_1), Bernoulli(p_2) sequences with correlated correctness, via a Gaussian copula."""
    Z = rng.multivariate_normal([0.0, 0.0], [[1.0, corr], [corr, 1.0]], size=(size, n))
    return (Z[..., 0] < stats.norm.ppf(p_1)).astype(float), (Z[..., 1] < stats.norm.ppf(p_2)).astype(float)


n7c, trials7c = 200, 4_000
c1_7, c2_7 = correlated_bernoulli(n7c, 0.75, 0.70, 0.6, trials7c, rng)
p1_hat, p2_hat = c1_7.mean(axis=1), c2_7.mean(axis=1)
diff7 = c1_7 - c2_7
se_paired = diff7.std(axis=1, ddof=1) / math.sqrt(n7c)
se_unpaired = np.sqrt(p1_hat * (1 - p1_hat) / n7c + p2_hat * (1 - p2_hat) / n7c)   # NOTE: ignores the correlation
power_paired = (np.abs(diff7.mean(axis=1) / se_paired) > stats.norm.ppf(0.975)).mean()
power_unpaired = (np.abs((p1_hat - p2_hat) / se_unpaired) > stats.norm.ppf(0.975)).mean()
assert se_paired.mean() < se_unpaired.mean()               # positive correlation shrinks the paired standard error
assert power_paired > 1.5 * power_unpaired                 # so the paired test is markedly more powerful here

rng7d = np.random.default_rng(7)                           # a separate stream, so later checks keep their draws
c1_neg, c2_neg = correlated_bernoulli(n7c, 0.70, 0.70, -0.6, trials7c, rng7d)   # H0 true, negative covariance
d_neg = c1_neg - c2_neg
se_paired_neg = d_neg.std(axis=1, ddof=1) / math.sqrt(n7c)
p1_neg, p2_neg = c1_neg.mean(axis=1), c2_neg.mean(axis=1)
se_unpaired_neg = np.sqrt(p1_neg * (1 - p1_neg) / n7c + p2_neg * (1 - p2_neg) / n7c)
size_paired = (np.abs(d_neg.mean(axis=1) / se_paired_neg) > stats.norm.ppf(0.975)).mean()
size_unpaired = (np.abs((p1_neg - p2_neg) / se_unpaired_neg) > stats.norm.ppf(0.975)).mean()
assert se_unpaired_neg.mean() < 0.95 * se_paired_neg.mean()                # the independent formula is too small
assert abs(size_paired - 0.05) < 0.015 and size_unpaired > 0.07            # so it rejects a true H0 too often

# ---- Q8: chi-square goodness of fit for the die
counts8 = np.array([90, 110, 95, 105, 120, 80])
expected8 = np.full(6, counts8.sum() / 6)
chi2_stat8 = float(((counts8 - expected8) ** 2 / expected8).sum())
assert math.isclose(chi2_stat8, 10.5, rel_tol=1e-9)
lib8 = stats.chisquare(counts8, expected8)
assert math.isclose(float(lib8.statistic), chi2_stat8, rel_tol=1e-9)
df8 = len(counts8) - 1
assert df8 == 5
crit8, pval8 = stats.chi2.ppf(0.95, df8), stats.chi2.sf(chi2_stat8, df8)
assert math.isclose(crit8, 11.07, abs_tol=5e-3) and math.isclose(pval8, 0.0622, abs_tol=5e-4)
assert math.isclose(pval8, float(lib8.pvalue), rel_tol=1e-9)
assert chi2_stat8 < crit8 and pval8 > 0.05                                   # do not reject fairness at 5%

# ---- Q9: which array does x come from -- Gaussian-discriminant thresholds and a fitted classifier
def log_posterior_gap(x, pi_a, pi_b, mu_a, sig_a, mu_b, sig_b):
    la = math.log(pi_a) - 0.5 * math.log(2 * math.pi * sig_a ** 2) - (x - mu_a) ** 2 / (2 * sig_a ** 2)
    lb = math.log(pi_b) - 0.5 * math.log(2 * math.pi * sig_b ** 2) - (x - mu_b) ** 2 / (2 * sig_b ** 2)
    return la - lb


mu9, sig_a9, sig_b9 = 0.0, 1.0, 3.0
root_equal = optimize.brentq(lambda x: log_posterior_gap(x, 0.5, 0.5, mu9, sig_a9, mu9, sig_b9), 0.1, 5.0)
assert math.isclose(root_equal, 1.5 * math.sqrt(math.log(3)), rel_tol=1e-8)
root_3to1 = optimize.brentq(lambda x: log_posterior_gap(x, 0.75, 0.25, mu9, sig_a9, mu9, sig_b9), 0.1, 5.0)
assert math.isclose(root_3to1, 1.5 * math.sqrt(math.log(9)), rel_tol=1e-8)

n9 = 4_000                                                                    # a fitted classifier on seeded samples
a9, b9 = rng.normal(mu9, sig_a9, n9), rng.normal(mu9, sig_b9, n9)
mu_a9, sig_hat_a9 = a9.mean(), a9.std(ddof=1)
mu_b9, sig_hat_b9 = b9.mean(), b9.std(ddof=1)
for x9, want_a in ((0.5, True), (4.0, False), (-4.0, False)):
    gap9 = log_posterior_gap(x9, 0.5, 0.5, mu_a9, sig_hat_a9, mu_b9, sig_hat_b9)
    assert (gap9 > 0) == want_a
def nearest_mean9(x):
    return "a" if abs(x - mu_a9) < abs(x - mu_b9) else "b"


assert "a" in (nearest_mean9(4.0), nearest_mean9(-4.0))    # both belong to b; nearest-mean sends one of them to a

# ---- Q10: entropy, cross-entropy, the H(p,q) = H(p) + KL(p||q) identity, maximum entropy, perplexity
def entropy(p):
    p = p[p > 0]                                     # NOTE: 0 log 0 is taken to be its limit, 0, not left as nan
    return float(-(p * np.log(p)).sum())


def cross_entropy(p, q):
    mask = p > 0
    return float(-(p[mask] * np.log(q[mask])).sum())


def kl_disc(p, q):
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


for _ in range(500):
    K10 = int(rng.integers(2, 10))
    p10, q10 = rng.random(K10) + 1e-3, rng.random(K10) + 1e-3
    p10, q10 = p10 / p10.sum(), q10 / q10.sum()
    assert math.isclose(cross_entropy(p10, q10), entropy(p10) + kl_disc(p10, q10), rel_tol=1e-8)

for K10b in (2, 5, 10, 50):
    unif10 = np.full(K10b, 1.0 / K10b)
    assert math.isclose(entropy(unif10), math.log(K10b), rel_tol=1e-12)
    for _ in range(50):
        p10b = rng.random(K10b) + 1e-3
        p10b /= p10b.sum()
        assert entropy(p10b) <= math.log(K10b) + 1e-9                                  # maximised by the uniform law
        assert math.isclose(kl_disc(p10b, unif10), math.log(K10b) - entropy(p10b), rel_tol=1e-8)

assert math.isclose(math.exp(2.0), 7.39, abs_tol=5e-3)                                  # perplexity at loss 2.0 nats
K10c = 500
for _ in range(30):
    p10c = rng.random(K10c) + 1e-3
    p10c /= p10c.sum()
    ce10c = cross_entropy(p10c, np.full(K10c, 1.0 / K10c))
    assert math.isclose(ce10c, math.log(K10c), rel_tol=1e-9)                            # true p is irrelevant here
    assert math.isclose(math.exp(ce10c), K10c, rel_tol=1e-8)                            # perplexity = vocabulary size

# ---- Q11: mutual information: the three formulas, independence, the BSC, information gain
def mi_definition(joint):
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    outer, mask = np.outer(px, py), joint > 1e-300
    return float((joint[mask] * np.log(joint[mask] / outer[mask])).sum())


def mi_via_cond_entropy(joint):
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    h_given_y = sum(py[j] * entropy(joint[:, j] / py[j]) for j in range(joint.shape[1]) if py[j] > 1e-300)
    return entropy(px) - h_given_y


def mi_via_joint_entropy(joint):
    return entropy(joint.sum(axis=1)) + entropy(joint.sum(axis=0)) - entropy(joint.flatten())


for _ in range(300):
    r11, c11 = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    joint11 = rng.random((r11, c11)) + 1e-3
    joint11 /= joint11.sum()
    mi1, mi2, mi3 = mi_definition(joint11), mi_via_cond_entropy(joint11), mi_via_joint_entropy(joint11)
    assert math.isclose(mi1, mi2, abs_tol=1e-8) and math.isclose(mi1, mi3, abs_tol=1e-8) and mi1 >= -1e-9

px11, py11 = rng.random(4), rng.random(3)
assert abs(mi_definition(np.outer(px11 / px11.sum(), py11 / py11.sum()))) < 1e-10        # zero on a product table


def binary_entropy_bits(p):
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


flip11 = 0.1
joint_bsc = np.array([[0.5 * (1 - flip11), 0.5 * flip11], [0.5 * flip11, 0.5 * (1 - flip11)]])
mi_bsc_bits = mi_definition(joint_bsc) / math.log(2)
assert math.isclose(mi_bsc_bits, 1 - binary_entropy_bits(flip11), rel_tol=1e-9)
assert math.isclose(mi_bsc_bits, 0.531, abs_tol=5e-4)

split11 = rng.integers(0, 2, 400)                                           # information gain = mutual information
label11 = np.where(rng.random(400) < np.where(split11 == 1, 0.8, 0.3), 1, 0)
joint_counts11 = np.zeros((2, 2))
for s11, y11 in zip(split11, label11):
    joint_counts11[s11, y11] += 1
joint_emp11 = joint_counts11 / joint_counts11.sum()
px_split11 = joint_emp11.sum(axis=1)
h_label_given_split = sum(px_split11[s11] * entropy(joint_emp11[s11] / px_split11[s11])
                          for s11 in range(2) if px_split11[s11] > 0)
info_gain11 = entropy(joint_emp11.sum(axis=0)) - h_label_given_split
assert math.isclose(info_gain11, mi_definition(joint_emp11), abs_tol=1e-9)
assert info_gain11 > 0.01

# ---- Q12: five integrals by hand, checked by differentiation or by quadrature
h12 = 1e-6


def check_antiderivative(F, f, xs):
    for x in xs:
        assert math.isclose((F(x + h12) - F(x - h12)) / (2 * h12), f(x), rel_tol=1e-4, abs_tol=1e-4)


check_antiderivative(lambda x: (x - 1) * math.exp(x), lambda x: x * math.exp(x), [-2.0, -0.5, 0.3, 1.0, 2.5])
check_antiderivative(lambda x: x * math.log(x) - x, math.log, [0.1, 0.5, 1.0, 2.0, 5.0])
check_antiderivative(math.atan, lambda x: 1 / (1 + x ** 2), [-3.0, -1.0, 0.0, 1.0, 3.0])

cauchy_integral, _ = integrate.quad(lambda x: 1 / (math.pi * (1 + x ** 2)), -np.inf, np.inf)
assert math.isclose(cauchy_integral, 1.0, rel_tol=1e-10)
gaussian_integral, _ = integrate.quad(lambda x: math.exp(-x ** 2 / 2), -np.inf, np.inf)
assert math.isclose(gaussian_integral, math.sqrt(2 * math.pi), rel_tol=1e-10)
for lam12 in (0.5, 1.0, 3.0):
    mean_exp12, _ = integrate.quad(lambda x: x * lam12 * math.exp(-lam12 * x), 0, np.inf)
    assert math.isclose(mean_exp12, 1 / lam12, rel_tol=1e-8)

print("all checks passed")
```

</details>

</details>
