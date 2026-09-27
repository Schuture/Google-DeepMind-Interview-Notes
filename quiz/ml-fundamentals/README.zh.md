# 机器学习基础：评估指标、损失函数、优化器与正则化

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 知识问答 · 口述，含推导 | ★★★★★ | 中等 | RS · RE · MLE · Applied AI · Intern | precision-recall, roc-auc, cross-entropy, logistic-regression, adam, weight-decay, bias-variance, normalisation, huber-loss, gan, distribution-shift | 15 个问题 / 60 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

全文中，$\log$ 表示自然对数，$\sigma(z) = 1/(1+e^{-z})$ 是 *sigmoid 函数*。

### 评估

**Q1.** 一个*二元分类器*（binary classifier）在 $1{,}000$ 个带标签样本上评估，得到混淆矩阵（confusion matrix）$\mathrm{TP}=40$、$\mathrm{FP}=10$、$\mathrm{FN}=20$、$\mathrm{TN}=930$（真/假正例、真/假负例的计数）。用这四个计数定义准确率（accuracy）、精确率（precision）、召回率（recall）和 F1 分数，计算这四个指标的值，并说明在这份数据上哪一个指标最容易误导人地概括这个分类器的表现，以及原因。分别给出一个会将判定阈值调向精确率优先的现实场景，和一个会调向召回率优先的现实场景。

**Q2.** 一个二元分类器给每个样本打一个实数分数；当分数超过阈值 $t$ 时，预测为正例。记 $\mathrm{TPR}(t)$ 和 $\mathrm{FPR}(t)$ 分别为阈值 $t$ 处的真正例率和假正例率——即实际正例、实际负例中分数高于 $t$ 的比例。*ROC 曲线*（ROC curve）是 $\{(\mathrm{FPR}(t), \mathrm{TPR}(t)) : t \in \mathbb R\}$，*ROC-AUC* 是它下方的面积。设 $S^+$、$S^-$ 分别是独立抽取的一个正例和一个负例的分数。推导

$$\mathrm{ROC\text{-}AUC} = P(S^+ > S^-) + \tfrac12 P(S^+ = S^-).$$

再解释为什么在类别严重不平衡时，*精确率-召回率曲线*（precision–recall curve，精确率对召回率作图，同样在这些阈值上扫描）是更有信息量的总结：说明当评估集里每个负例都被复制成十份时，ROC 曲线和精确率-召回率曲线各自会发生什么，以及原因。

### 损失函数

**Q3.** 单个样本的标签为 $y \in \{0,1\}$，模型给出一个 logit $z \in \mathbb R$，得到预测概率 $p = \sigma(z)$。对两种逐样本损失分别推导 $\partial L/\partial z$：*二元交叉熵*（binary cross-entropy）$L_{\mathrm{bce}} = -[y \log p + (1-y)\log(1-p)]$，以及*平方误差*（squared error）$L_{\mathrm{se}} = (p-y)^2$。用这两个导数解释：当模型自信地犯错时（比如 $y=1$ 而 $z \ll 0$），用平方误差训练为什么学得很慢，而交叉熵不会。

**Q4.** 一个逻辑回归（logistic regression）模型有权重 $w \in \mathbb R^d$；给定设计矩阵（design matrix）$X \in \mathbb R^{n \times d}$（第 $i$ 行是 $x_i^\top$）和标签 $y \in \{0,1\}^n$，它预测 $p_i = \sigma(x_i^\top w)$。写出负对数似然（negative log-likelihood）$L(w) = -\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$，推导它的梯度 $\nabla L(w) = X^\top(\sigma(Xw) - y)$（$\sigma$ 逐元素作用）和 Hessian 矩阵 $\nabla^2 L(w) = X^\top S X$（$S = \mathrm{diag}(p_i(1-p_i))$），并用这个 Hessian 矩阵说明 $L$ 在整个 $\mathbb R^d$ 上是凸的。现在设数据*线性可分*（linearly separable）：存在某个 $w^\star$，对每个 $i$，只要 $y_i=1$ 就有 $x_i^\top w^\star > 0$，只要 $y_i=0$ 就有 $x_i^\top w^\star < 0$。说明在这份数据上，无正则化的梯度下降中 $\inf_w L(w)$ 和 $\lVert w \rVert$ 会怎样变化，以及给 $L$ 加上 $\ell_2$ 惩罚项 $\tfrac\lambda2\lVert w \rVert^2$ 之后答案会如何改变。

**Q5.** 模型对一个 $K$ 类问题输出 logits $z \in \mathbb R^K$，$\mathrm{softmax}(z)_i = e^{z_i}/\sum_k e^{z_k}$，真实类别为 $y \in \{1,\dots,K\}$ 的样本的损失是 $L(z) = -\log \mathrm{softmax}(z)_y$。推导 $\nabla_z L = \mathrm{softmax}(z) - e_y$，其中 $e_y \in \mathbb R^K$ 是类别 $y$ 对应的标准基向量。引用使其成立的恒等式，解释为什么正确的实现在调用 `exp` 之前，会先给每个 logit 减去 $m = \max_k z_k$——无论是计算 $\mathrm{softmax}(z)$ 本身，还是计算 $\log \sum_k e^{z_k}$。

### 优化器

**Q6.** 对一个标量参数 $\theta$，第 $t$ 步的梯度为 $g_t$，学习率为 $\eta$：普通 SGD 的更新是 $\theta_{t+1} = \theta_t - \eta g_t$；带动量（momentum）$\mu$ 的 SGD 维护速度 $v_t = \mu v_{t-1} + g_t$（$v_0=0$），更新为 $\theta_{t+1} = \theta_t - \eta v_t$；Adam 维护 $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ 和 $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$（$m_0 = v_0 = 0$），偏差校正（bias correction）为 $\hat m_t = m_t/(1-\beta_1^t)$、$\hat v_t = v_t/(1-\beta_2^t)$，更新为 $\theta_{t+1} = \theta_t - \eta\, \hat m_t/(\sqrt{\hat v_t}+\epsilon)$。假设每一步都有 $\mathbb E[g_t] = \bar g$，求 $\mathbb E[m_t]$ 的闭式解，并用它解释为什么 $m_t$ 要除以 $1-\beta_1^t$。再证明：无论 $g_1$ 是多少，第一次 Adam 更新都满足 $\hat m_1/(\sqrt{\hat v_1}+\epsilon) \approx \mathrm{sign}(g_1)$——所以第一步会让每个坐标移动大约 $\eta$，无论该坐标的梯度是大是小。

**Q7.** 考虑一次 SGD 族的更新，梯度为 $g$，学习率为 $\eta$，系数为 $\lambda > 0$。*$\ell_2$ 正则化*的 SGD 通过梯度下降最小化 $L(w) + \tfrac\lambda2\lVert w\rVert^2$：$w \leftarrow w - \eta(g + \lambda w)$。*解耦的权重衰减*（decoupled weight decay）则每一步都独立于损失梯度、按比例 $\gamma$ 缩小权重：$w \leftarrow (1-\gamma) w - \eta g$。证明一旦 $\gamma = \eta\lambda$，这两者对任意 $w$、$g$ 都完全一致。现在考虑 Adam。*带 $\ell_2$ 正则化的 Adam* 在形成 $m_t, v_t$ 之前，先把 $\lambda w$ 加进 $g$，再照常走一步 Adam。*AdamW* 只用 $g$ 本身形成 $m_t, v_t$，再直接施加解耦的衰减：$w \leftarrow (1-\gamma) w - \eta\, \hat m_t/(\sqrt{\hat v_t}+\epsilon)$。结合 $\hat v_t$，解释为什么无论 $\gamma = \eta\lambda$ 还是任何其他的系数对应关系，这两者都不再一致。

**Q8.** 训练用 Adam 优化器，预热结束后的目标学习率为 $\eta$。*学习率预热*（learning-rate warmup）在最初几千步里，把学习率从 $0$ 线性升到 $\eta$，而不是从第 $1$ 步就用 $\eta$；*学习率衰减*（learning-rate decay，余弦或线性）则在训练的最后阶段把它降回 $0$ 附近。结合只经过几步之后 $\hat v_t$ 有多可靠，解释预热对 Adam、尤其是大 batch 训练有用的机制。另外，对用*恒定*学习率 $\eta$ 最小化随机目标的普通 SGD，说明迭代点最终会怎样，而不是收敛到最小值点；这个围绕最小值点的散布大小如何随 $\eta$ 变化；并用这一点解释为什么训练后期衰减 $\eta$ 有帮助。

### 泛化与正则化

**Q9.** 固定输入 $x$，设 $y = f(x) + \varepsilon$，其中 $f$ 是未知函数，噪声 $\varepsilon$ 满足 $\mathbb E[\varepsilon] = 0$、$\mathrm{Var}(\varepsilon) = \sigma^2$，且与训练数据独立。给定一个随机训练集 $D$，某个学习算法得到预测器 $\hat f_D$；记 $\bar f(x) = \mathbb E_D[\hat f_D(x)]$。推导偏差-方差分解（bias-variance decomposition）

$$\mathbb E_{D,\varepsilon}\bigl[(y - \hat f_D(x))^2\bigr] = \underbrace{(\bar f(x) - f(x))^2}_{\mathrm{偏差}^2} + \underbrace{\mathbb E_D\bigl[(\hat f_D(x) - \bar f(x))^2\bigr]}_{\mathrm{方差}} + \sigma^2,$$

并说明哪一步用到了 $\mathbb E[\varepsilon] = 0$，哪一步用到了 $\varepsilon$ 与 $D$ 的独立性。

**Q10.** 对设计矩阵 $X \in \mathbb R^{n \times d}$（$\mathrm{rank}(X) = d$）、目标 $y \in \mathbb R^n$ 和 $\lambda > 0$，*岭回归*（ridge regression）最小化 $\lVert y - Xw \rVert^2 + \lambda \lVert w \rVert^2$。推导闭式解 $\hat w_\lambda = (X^\top X + \lambda I)^{-1} X^\top y$。写出瘦身奇异值分解（singular value decomposition，SVD）$X = U \Sigma V^\top$，其奇异值为 $d_1,\dots,d_d$，用 $U, V, \Sigma, y$ 表示 $\hat w_\lambda$，并证明：在 $V$ 给出的坐标系里，岭回归把沿第 $i$ 个奇异方向的最小二乘系数缩小了 $d_i^2/(d_i^2+\lambda)$ 倍。然后取一个列正交（orthonormal columns）的设计（$X^\top X = I$）：在它上面，$\arg\min_w \lVert y-Xw\rVert^2+\lambda\lVert w\rVert^2$（如上，岭回归）分解为 $d$ 个独立的一维问题，$\arg\min_w \tfrac12\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1$（*lasso*，$\ell_1$ 惩罚的对应版本，按惯例带上前面的 $\tfrac12$ 使阈值恰好是 $\lambda$）也一样。推导两者各自的一维闭式解，并用它们解释为什么 lasso 经常把系数精确置零，而岭回归不会。

**Q11.** *倒置 dropout*（inverted dropout）以丢弃概率（drop probability）$p \in (0,1)$，独立地把一层里的每个激活值 $h$ 替换为 $\tilde h = \frac{m}{1-p} h$，其中 $m \sim \mathrm{Bernoulli}(1-p)$（即该单元以概率 $1-p$ 被保留，$m=1$）。推导 $\mathbb E[\tilde h]$，并说明如果没有 $1/(1-p)$ 这个因子，它会是多少。说明使用倒置 dropout 的一层在评估（evaluation）时计算的是什么，以及为什么那里不需要再做任何缩放。

**Q12.** 对一个包含 $N$ 个样本的*批次*（batch），每个样本是一个 $C$ 维特征向量，写成 $h \in \mathbb R^{N \times C}$，元素为 $h_{n,c}$：*批归一化*（batch normalisation）对每个通道 $c$，在 $N$ 个样本上计算均值和方差，并用它们归一化第 $c$ 列；*层归一化*（layer normalisation）对每个样本 $n$，在它自己的 $C$ 个特征上计算均值和方差，并用它们归一化第 $n$ 行。写出这两个公式（在任何可学习的仿射变换之前），并说明各自在哪个轴上归约。描述批归一化在推理（inference）时和训练时有什么不同，以及为什么需要这个不同。解释为什么 Transformer 架构用层归一化（或 RMSNorm）而不是批归一化。

### 回归损失与生成模型

**Q13.** 一个回归模型对 $n$ 个训练样本 $(x_i, y_i)$ 中的每一个都给出预测 $f(x_i)$，残差（residual）为 $r_i = y_i - f(x_i)$；两种常见的训练目标是 *L2 损失* $\sum_i r_i^2$ 和 *L1 损失* $\sum_i |r_i|$。对于常数预测器这一特殊情形，即对每个 $i$ 都有 $f(x_i) = c$，从导数出发推导使 $\sum_i (y_i-c)^2$ 最小的 $c$ 值，再从每一项的符号以及 $|\cdot|$ 在 $0$ 处的次梯度（subgradient，即与其转折点相一致的斜率区间 $[-1,1]$）出发推导使 $\sum_i |y_i-c|$ 最小的 $c$ 值；说出这两个最小值点各自对应哪个经典统计量，并说明 $\{y_i\}$ 中单个较大的离群值能把它拖多远。把 $\partial r^2/\partial r$ 以及 $|r|$ 的一个次梯度都写成残差 $r$ 的函数，并描述二者在 $r=0$ 处、以及 $|r|\to\infty$ 时的表现。说出在残差服从怎样的噪声模型时，最小化这两种损失各自恰好是最大似然估计（maximum likelihood estimation），并从该噪声模型的对数似然出发，推导其中一个对应关系。定义阈值为 $\delta>0$ 的 *Huber 损失*，

$$L_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta\left(|r| - \tfrac12\delta\right) & |r| > \delta \end{cases},$$

给出 $\partial L_\delta/\partial r$，并说明什么情况下它比单独使用 L1 或 L2 损失更合适。最后用一句话对比这两种损失*另一个*常见的角色——作为加在损失上的正则化项，而不是残差本身的损失（参见 Q10，不必重复其推导）。

**Q14.** 一个*生成对抗网络*（generative adversarial network，GAN）让生成器（generator）$G$ 与判别器（discriminator）$D$ 相互对抗，共同优化极小极大目标（minimax objective）

$$V(D,G) = \mathbb E_{x\sim p_{\mathrm{data}}}[\log D(x)] + \mathbb E_{z\sim p_z}[\log(1-D(G(z)))],$$

其中 $D$ 在固定 $G$ 时最大化 $V$，$G$ 则针对由此得到的 $D$ 最小化 $V$；记 $p_g$ 为 $z\sim p_z$ 时 $G(z)$ 的分布。固定 $G$（从而固定 $p_g$），逐点（pointwise）在每个 $x$ 上推导 $D^\star = \arg\max_D V(D,G)$——对常数 $a,b>0$，最大化 $a\log y + b\log(1-y)$（$y\in(0,1)$）——并证明它就是 $D^\star(x) = p_{\mathrm{data}}(x)/(p_{\mathrm{data}}(x)+p_g(x))$。记 $m = (p_{\mathrm{data}}+p_g)/2$，并记离散分布 $p,q$ 之间的 KL 散度（Kullback–Leibler divergence）为 $\mathrm{KL}(p\Vert q) = \sum_x p(x)\log(p(x)/q(x))$，则*Jensen–Shannon 散度*为 $\mathrm{JSD}(p\Vert q) = \tfrac12\mathrm{KL}(p\Vert m)+\tfrac12\mathrm{KL}(q\Vert m)$；证明 $V(D^\star,G) = -\log 4 + 2\,\mathrm{JSD}(p_{\mathrm{data}}\Vert p_g)$，并用 $\mathrm{JSD}\ge 0$ 说明对 $G$ 取全局最小值时，$p_g$ 应该是什么。记 $s$ 为判别器在一个生成样本上、送入 sigmoid 之前的 logit，即 $D(G(z)) = \sigma(s)$，比较 $\partial/\partial s\,\log(1-\sigma(s))$——上面目标函数中被最小化的原始生成器损失——与 $\partial/\partial s\,[-\log\sigma(s)]$——实践中实际使用的*非饱和*（non-saturating）生成器损失——在 $s \ll 0$ 时，也就是 $D(G(z))$ 接近 $0$ 时的取值，并解释为什么训练早期实践中最小化的是后者而不是前者。说出除这种梯度消失之外，GAN 训练中两种不同的失败模式，并分别给出一种具体的缓解办法。

### 从离线到线上

**Q15.** 一个新模型在离线的留出测试集（held-out test set）上分数高于当前的线上模型，但上线之后，它的线上指标（例如点击率）却比线上模型更差。列出你会去排查的几类不同原因——预期是五类——并逐一给出准确的定义（对于数据分布的变化，要按照是哪个分布发生了变化，区分它的几种标准类型）。对每一类原因，给出你会用来确认或排除它的具体检查方法。

## 参考解答

<details>
<summary>展开参考解答</summary>

开口作答前值得先确认两个约定：下面动量和 Adam 的更新规则用的是未归一化的速度/矩项，这是 ML 课程和框架里的常见写法——如果某本教材改用 $(1-\mu)$ 来加权动量项，描述的是同一个算法，只是有效学习率被重新缩放了。另外在 Q11 里，约定的是“丢弃概率 $p$”而不是“保留概率”；写公式时用 $1-p$ 还是 $p$，最好先和面试官确认清楚。

### 评估

**Q1.** 准确率是 $0.97$，精确率是 $0.8$，召回率是 $2/3 \approx 0.667$，F1 约为 $0.727$，其中准确率是最会误导人的那个。由计数直接算：$\mathrm{accuracy} = (\mathrm{TP}+\mathrm{TN})/n = 970/1000$；$\mathrm{precision} = \mathrm{TP}/(\mathrm{TP}+\mathrm{FP}) = 40/50$；$\mathrm{recall} = \mathrm{TP}/(\mathrm{TP}+\mathrm{FN}) = 40/60$；F1 是精确率与召回率的调和平均 $2\,\mathrm{precision}\cdot\mathrm{recall}/(\mathrm{precision}+\mathrm{recall})$，不是算术平均——调和平均对其中一项偏低的惩罚，比取平均严重得多。这份数据是不平衡的（$1{,}000$ 个样本里只有 $60$ 个正例），一个完全不看输入、永远预测“负例”的分类器在它上面已经能拿到 $940/1000 = 0.94$ 的准确率，所以这个分类器的 $0.97$ 比这个平凡基线也就多了一点点信息量，即便它漏掉了三分之一的真实正例（召回率 $0.667$）——这正是准确率会掩盖、而召回率会直接暴露出来的症状。当假正例是代价更高的错误时，应该偏向精确率，比如把一封正常邮件误判为垃圾邮件；当假负例是代价更高的错误时，应该偏向召回率，比如用一个便宜的检测手段筛查某种疾病，漏诊会延误治疗，而误报只是多花一次复查的代价。

**Q2.** ROC-AUC 恰好等于随机一个正例的分数高于随机一个负例分数的概率，平局按一半计——把分类阈值沿排序好的分数从高到低扫一遍，就把这个概率变成了一个面积计算。把全部 $n^++n^-$ 个分数从高到低排列，让阈值逐个不同取值地往下移动。每当新纳入一个负例 $j$，曲线就向右移动 $1/n^-$，此时的高度是 $\mathrm{TPR}$ 已经达到的值——也就是已经纳入的正例所占比例，$\#\{i : s_i^+ > s_j^-\}/n^+$；而每当新纳入一个正例，曲线向上移动，$\mathrm{FPR}$ 不变。把每一次向右移动贡献的面积加起来，得到 $\frac{1}{n^+n^-}\sum_{i,j} \mathbb 1(s_i^+ > s_j^-)$，这正是均匀随机一对 $(i,j)$ 下 $P(S^+ > S^-)$ 的值。当若干样本分数相同时，它们必须一起移动，因为没有哪个阈值能把它们分开：$a$ 个并列的正例和 $b$ 个并列的负例一起移动时，曲线沿一段直线的对角线从原处移动到新处，这个梯形的面积是 $\mathrm{TPR}\cdot(b/n^-) + \tfrac12(a/n^+)(b/n^-)$——第一项是给已经严格领先的那些配对的满分，第二项则把半分恰好平均分给新出现的 $ab$ 对并列组合。把整个扫描过程加总，就得到 $P(S^+>S^-) + \tfrac12 P(S^+=S^-)$，如下面的验证代码所示，它与 scikit-learn 的 `roc_auc_score` 在数值精度范围内完全一致。$\mathrm{TPR}(t)$ 和 $\mathrm{FPR}(t)$ 都是各自类别内部的比例，所以把每个负例复制成十份，两者都不会变：ROC 曲线和 ROC-AUC 都精确保持不变。阈值 $t$ 处的精确率是 $\mathrm{TP}(t)/(\mathrm{TP}(t)+\mathrm{FP}(t))$；复制操作把 $\mathrm{FP}(t)$ 乘以十而 $\mathrm{TP}(t)$ 不变，所以只要 $\mathrm{FP}(t)>0$，精确率在每个阈值上都会下降；在下面检查所用的数据上，平均精确率（precision–recall 曲线下的面积）下降超过 $0.1$——这正是“要正确面对多得多的负例”这一实际代价，而 ROC-AUC 因为从不除以类别数量，根本看不到它。

### 损失函数

**Q3.** 两个导数都能化成干净的闭式；交叉熵的导数恰好在平方误差的导数塌缩之处依然很大。由 $\sigma(z) = 1/(1+e^{-z})$ 用商法则立得 $\sigma'(z) = \sigma(z)(1-\sigma(z))$，从而 $\partial p/\partial z = \sigma'(z) = p(1-p)$：

$$\frac{\partial L_{\mathrm{bce}}}{\partial z} = \Bigl(-\frac{y}{p}+\frac{1-y}{1-p}\Bigr)p(1-p) = -y(1-p) + (1-y)p = p - y, \qquad \frac{\partial L_{\mathrm{se}}}{\partial z} = 2(p-y)\cdot p(1-p).$$

在 $y=1$、$z \ll 0$ 时模型是自信地犯错：$p \to 0$。交叉熵的梯度 $p - y \to -1$，是一个满幅的、推动 $z$ 增大的力；平方误差的梯度多了一个因子 $p(1-p) \to 0$：sigmoid 已经饱和，所以即便 $(p-y)$ 已经接近它最差的取值，$\partial p/\partial z$ 却很小，两者几乎相互抵消——在下面的检查中，$z=-20$ 处这两个梯度相差超过七个数量级。平方误差要等到 $p$ 已经离开饱和区之后才会真正开始用力推，所以一个错得离谱又很自信的预测，在平方误差下纠正得非常慢，在交叉熵下则很快。

**Q4.** 梯度和 Hessian 矩阵和 Q3 是同一套模式，只是对样本求了和；凸性来自 Hessian 矩阵是一组带非负权重的外积之和；可分性则破坏了有限最小值点的存在性。记 $z_i = x_i^\top w$、$p_i = \sigma(z_i)$，Q3 给出 $\partial L/\partial z_i = p_i - y_i$，由链式法则 $\partial L/\partial w = \sum_i (p_i-y_i)x_i = X^\top(\sigma(Xw)-y)$。再求一次导，$\partial p_i/\partial w = p_i(1-p_i)x_i$，于是 $\nabla^2 L(w) = \sum_i p_i(1-p_i)\,x_ix_i^\top = X^\top S X$。对任意 $v \in \mathbb R^d$，$v^\top X^\top S X v = \sum_i p_i(1-p_i)(x_i^\top v)^2 \ge 0$，因为 $p_i \in (0,1)$ 时 $p_i(1-p_i) \ge 0$，且每一个平方项都非负——这个 Hessian 矩阵处处半正定，所以 $L$ 在 $\mathbb R^d$ 上是凸的。在可分数据上，令 $w = t\,w^\star$ 并让 $t \to \infty$，会把每个带符号的间隔 $x_i^\top w$（$y_i=1$ 时取正、$y_i=0$ 时取负）都推向 $+\infty$，于是每个 $p_i \to y_i$，$L(tw^\star) \to 0$ 且单调递减：$\inf_w L(w) = 0$，但没有任何有限的 $w$ 能取到它，所以只要一直跑下去，无正则化的梯度下降就会让 $\lVert w \rVert$ 无界增长。加上 $\tfrac\lambda2\lVert w\rVert^2$ 之后，当 $\lVert w \rVert \to \infty$ 时目标函数 $L(w) + \tfrac\lambda2\lVert w\rVert^2 \to \infty$（因为 $L \ge 0$），同时它的 Hessian 矩阵 $X^\top S X + \lambda I$ 现在处处*正定*（单是 $\lambda I$ 就已经正定）：正则化后的目标函数严格凸且强制（coercive），无论数据是否可分，它都有唯一的有限最小值点，梯度下降会收敛到那里，而不是发散。

**Q5.** 梯度是 softmax 减去 one-hot 目标；减去最大值这个操作，在精确算术下完全不改变 softmax 和 log-sum-exp 的值，在浮点运算里则让每个指数都不超过 $0$。写 $L(z) = -z_y + \log\sum_k e^{z_k}$。第一项贡献 $-\partial z_y/\partial z_i = -\delta_{iy}$；第二项，$\partial/\partial z_i \log\sum_k e^{z_k} = e^{z_i}/\sum_k e^{z_k} = \mathrm{softmax}(z)_i$。所以 $\partial L/\partial z_i = \mathrm{softmax}(z)_i - \delta_{iy}$，即 $\nabla_z L = \mathrm{softmax}(z) - e_y$。对任意常数 $m$，$e^{z_k} = e^m \cdot e^{z_k - m}$，所以 $\sum_k e^{z_k} = e^m \sum_k e^{z_k-m}$：在 softmax 的比值里，$e^m$ 这个因子在分子分母中精确抵消；而 $\log \sum_k e^{z_k} = m + \log\sum_k e^{z_k-m}$ 对任意 $m$ 都精确成立——所以减去 $m = \max_k z_k$ 在数学上不改变这两个量。它改变的是浮点计算：平移后的每个指数 $z_k - m$ 都 $\le 0$，所以 `exp` 永远不会上溢；而其中最大的一项恰好是 $e^0=1$，所以这个和也永远不会下溢到 $0$——无论原始 logits 的量级多大、分布多分散都是如此。

### 优化器

**Q6.** SGD 的步长就是梯度乘以 $\eta$；动量以衰减率 $\mu$ 累积梯度；Adam 用每个坐标自己最近的梯度量级去归一化那一步，而它的偏差校正之所以存在，是因为矩估计从零开始，早期会被向零拖拽。把 $m_t$ 从 $m_0=0$ 展开：$m_t = (1-\beta_1)\sum_{k=1}^t \beta_1^{t-k}g_k$，于是在每一步都有 $\mathbb E[g_k]=\bar g$ 的假设下，

$$\mathbb E[m_t] = (1-\beta_1)\bar g \sum_{k=1}^t \beta_1^{t-k} = (1-\beta_1)\bar g\,\frac{1-\beta_1^t}{1-\beta_1} = (1-\beta_1^t)\,\bar g,$$

这里用到了有限几何级数 $\sum_{j=0}^{t-1}\beta_1^j = (1-\beta_1^t)/(1-\beta_1)$。所以 $m_t$ 以因子 $1-\beta_1^t$ 低估了 $\bar g$：$t$ 小时这个因子接近 $0$（平均才刚刚开始），$t$ 大时接近 $1$；恰好除以这个因子，$\hat m_t = m_t/(1-\beta_1^t)$，在这个平稳性假设下对每个 $t$ 都给出 $\mathbb E[\hat m_t] = \bar g$，完全消除了偏差——下面的蒙特卡洛（Monte Carlo）模拟在其抽样误差范围内证实了这一点。在 $t=1$ 时，$m_1 = (1-\beta_1)g_1$，所以 $\hat m_1 = g_1$ *精确*成立，无论 $\beta_1$ 取什么值（校正因子 $1-\beta_1^1=1-\beta_1$ 恰好把它精确抵消），同样 $\hat v_1 = g_1^2$ 也精确成立。于是 $\hat m_1/(\sqrt{\hat v_1}+\epsilon) = g_1/(|g_1|+\epsilon)$，这等于 $\mathrm{sign}(g_1)$ 加上一个随 $|g_1|/\epsilon \to \infty$ 而消失的误差：第一次 Adam 更新的步长在每个坐标上都是 $\eta\cdot|g_1|/(|g_1|+\epsilon) \approx \eta$——下面的检查对跨越六个数量级的梯度验证了这一点，误差在 $\eta$ 的 $0.01\%$ 以内——因为无论该坐标的梯度有多大或多小，第 $1$ 步的偏差校正都是精确的。

**Q7.** 对 SGD 来说，这两条更新规则只是同一个公式的不同写法；对 Adam 来说则不是，因为 $\ell_2$ 的衰减项会经过那个自适应的分母，而解耦的衰减不会。把 $\gamma = \eta\lambda$ 代入解耦衰减，$w \leftarrow (1-\eta\lambda)w - \eta g = w - \eta g - \eta\lambda w$，这恰好就是 $\ell_2$ 正则化的更新 $w - \eta(g+\lambda w)$——对任意 $w$、$g$ 两者逐项一致，没有别的附加条件，下面的检查也验证了这一点。对 Adam 来说，$\ell_2$ 正则化是在形成 $m_t, v_t$ *之前*就把 $\lambda w$ 并入 $g$，所以这个衰减项会和梯度的其余部分一起被除以 $\sqrt{\hat v_t}$：一个最近梯度量级较大（$\hat v_t$ 较大）的坐标，其有效衰减会相对于 $\hat v_t$ 较小的坐标被削弱，尽管两个权重本该按同样的比例 $\gamma$ 收缩。AdamW 则完全不让 $\lambda$ 进入 $m_t$、$v_t$，而是对每个坐标同等地施加 $(1-\gamma)$，所以它的衰减强度从不依赖那个坐标的梯度历史。由于 $\hat v_t$ 在不同坐标、不同时间上确实各不相同——这正是 Adam 逐坐标自适应的全部意义所在——没有哪一个 $\gamma$ 能让这两条更新规则在一般情况下一致，下面在不同量级梯度上的检查也证实了这一点。

**Q8.** 预热防的是训练早期 $\hat v_t$ 不可靠；衰减缩小的是恒定学习率在最小值点周围留下的、由噪声驱动的散布。$\hat v_1 = g_1^2$（Q6）：走完第一步之后，Adam 对某个坐标梯度量级的估计，完全是从一个带噪声的样本构建出来的，所以一个运气不好的、偏大或偏小的 $g_1$ 会被全盘照收，得到的步长依然 $\approx \eta$——指数平均还完全没有机会去把 minibatch 噪声平滑掉。用大 batch 训练时这一点更要紧，因为每步的学习率本身通常也会调大（用来补偿每个 epoch 步数的减少），于是早期 $\hat v_t$ 的相对不可靠会转化为更大的绝对风险；把 $\eta$ 从 $0$ 慢慢升上去，恰好能在这段不可靠的窗口期把*实际*步长压小，给 $\hat v_t$ 留出时间，在足够多步之后稳定下来。再看衰减这一半：在二次函数 $f(\theta)=\theta^2/2$（最小值在 $0$）上，用随机梯度 $g_t = \theta_t + \xi_t$（$\xi_t \sim N(0,\sigma^2)$ 独立同分布），SGD 给出 $\theta_{t+1} = (1-\eta)\theta_t - \eta\xi_t$，这是一个线性递推，其平稳方差满足 $\mathrm{Var}_\infty = (1-\eta)^2\mathrm{Var}_\infty + \eta^2\sigma^2$，即

$$\mathrm{Var}_\infty = \frac{\eta^2\sigma^2}{1-(1-\eta)^2} = \frac{\eta\sigma^2}{2-\eta}.$$

迭代点根本不会收敛到最小值点：它们依分布收敛到最小值点周围一个方差为 $\mathrm{Var}_\infty$ 的散布（下面的模拟在其所用误差范围内证实了这一点）——一个大小随 $\eta$ 增长的“噪声球”。训练后期衰减 $\eta$ 会缩小这个球——用更小的 $\eta$ 继续同一个递推，会稳定到同一公式所预言的更小方差——用大步长在早期更快、但更嘈杂的进展，换取围绕最优点更精确的最终落点。

### 泛化与正则化

**Q9.** 两次配方，一次用到 $\mathbb E[\varepsilon]=0$，一次用到 $\varepsilon$ 与 $D$ 的独立性。写 $y - \hat f_D(x) = \bigl(f(x)-\hat f_D(x)\bigr) + \varepsilon$；平方后取 $\mathbb E_{D,\varepsilon}$，并用 $\varepsilon$ 与 $D$ 的独立性把交叉项分解开，

$$\mathbb E_{D,\varepsilon}\bigl[(y-\hat f_D(x))^2\bigr] = \mathbb E_D\bigl[(f(x)-\hat f_D(x))^2\bigr] + 2\,\mathbb E_D[f(x)-\hat f_D(x)]\cdot\mathbb E[\varepsilon] + \mathbb E[\varepsilon^2];$$

$\mathbb E[\varepsilon]=0$ 直接消去中间项，$\mathbb E[\varepsilon^2]=\mathrm{Var}(\varepsilon)=\sigma^2$，剩下 $\mathbb E_D[(f(x)-\hat f_D(x))^2] + \sigma^2$。再把 $f(x)-\hat f_D(x)$ 拆成 $(f(x)-\bar f(x)) + (\bar f(x)-\hat f_D(x))$，再平方一次：

$$\mathbb E_D\bigl[(f(x)-\hat f_D(x))^2\bigr] = (f(x)-\bar f(x))^2 + 2(f(x)-\bar f(x))\underbrace{\mathbb E_D[\bar f(x)-\hat f_D(x)]}_{=\,0} + \mathbb E_D\bigl[(\hat f_D(x)-\bar f(x))^2\bigr],$$

其中中间项消失，是因为按定义 $\mathbb E_D[\hat f_D(x)] = \bar f(x)$——剩下的正好是 $\mathrm{偏差}^2 + \mathrm{方差}$，再加上前面的 $\sigma^2$，就是完整的分解。在一次模拟中，用一个明显欠拟合的模型（对一条弯曲的真实函数拟合一条直线），偏差平方项比方差项大三倍以上，而这三项之和，在下面所用的蒙特卡洛误差范围内，与实测的均方误差相符。

**Q10.** 岭回归的闭式解来自令梯度为零；它的 SVD 形式说明收缩是在每个主方向上独立作用的；在一个列正交的设计上，lasso 的闭式解是软阈值（soft-thresholding），它在恰好为零处有一整段平坦区域，这是岭回归光滑的惩罚项所没有的。$\nabla_w\bigl[\lVert y-Xw\rVert^2+\lambda\lVert w\rVert^2\bigr] = -2X^\top(y-Xw) + 2\lambda w = 0 \iff (X^\top X+\lambda I)w = X^\top y$，而 $\lambda>0$ 时 $X^\top X + \lambda I$ 正定（从而可逆），给出唯一的最小值点 $\hat w_\lambda = (X^\top X+\lambda I)^{-1}X^\top y$。设 $X=U\Sigma V^\top$，则 $X^\top X = V\Sigma^2 V^\top$，于是 $X^\top X+\lambda I = V(\Sigma^2+\lambda I)V^\top$（用到 $VV^\top=I$），从而

$$\hat w_\lambda = V(\Sigma^2+\lambda I)^{-1}\Sigma U^\top y = V\,\mathrm{diag}\Bigl(\frac{d_i}{d_i^2+\lambda}\Bigr)U^\top y.$$

普通最小二乘解是 $\lambda=0$ 的特例，$\hat w_{\mathrm{ols}} = V\,\mathrm{diag}(1/d_i)U^\top y$；在 $V$ 给出的坐标系里逐项比较，岭回归把第 $i$ 个最小二乘系数乘上 $d_i^2/(d_i^2+\lambda)$——当 $d_i^2 \gg \lambda$ 时接近 $1$（数据确定得好的方向几乎不动），当 $d_i^2 \ll \lambda$ 时接近 $0$（数据几乎确定不了的方向几乎被压到没有）。当 $X^\top X=I$ 时，记 $c = X^\top y$，同样的展开给出 $\lVert y-Xw\rVert^2 = \mathrm{常数} + \sum_i(w_i-c_i)^2$，于是 $\lVert y-Xw\rVert^2+\lambda\lVert w\rVert^2$ 逐坐标分解为 $(w_i-c_i)^2+\lambda w_i^2$；令其导数为零，$2(w_i-c_i)+2\lambda w_i=0$，得到 $w_i^\star = c_i/(1+\lambda)$——除非 $c_i$ 本身恰好是 $0$，否则岭回归总会返回一个非零的系数。对同一个设计上的 $\arg\min_w \tfrac12\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1$，同样的配方（现在多带一个整体的 $\tfrac12$）逐坐标分解为 $\tfrac12(w_i-c_i)^2+\lambda|w_i|$，其最小值点是软阈值 $w_i^\star=\mathrm{sign}(c_i)\max(|c_i|-\lambda,0)$：$\lambda|w_i|$ 在 $0$ 处的次梯度（subgradient）张成整个区间 $[-\lambda,\lambda]$，所以任何满足 $|c_i|\le\lambda$ 的 $c_i$，在 $w_i=0$ 处都已经精确满足零次梯度的最优性条件，不只是在某个极限意义下。岭回归的惩罚项没有这样的平坦区域——它的梯度 $2\lambda w_i$ 只在 $w_i=0$ 这一点上恰好为零——所以它会连续地收缩每一个系数，却从不会把哪个系数硬压到零。在下面的检查中，lasso 在一个设计上把六个系数里至少两个精确置零，而按上面闭式解算出的每个岭回归系数，量级都大于 $10^{-4}$。

**Q11.** $\mathbb E[\tilde h] = h$ 精确成立，这正是 $1/(1-p)$ 这个因子的意义所在：$\mathbb E[\tilde h] = \mathbb E[m]\,h/(1-p) = (1-p)h/(1-p) = h$，这里用到了 $m\sim\mathrm{Bernoulli}(1-p)$ 时 $\mathbb E[m]=1-p$。如果没有这个因子，光是掩码就会给出 $\mathbb E[mh] = (1-p)h$：每个激活值都会被系统性地压低 $1-p$ 这个因子，一层又一层，纯粹是 dropout 的机制造成的，而不是任何学到的东西——下面的检查里把这个“未修正”的情形也验证了一遍。因为训练时的期望前向计算已经等于不做 dropout 时的计算，评估时只需要把 dropout 关掉——每个单元都保留，不做掩码——也不需要再做任何缩放：那已经和训练在期望意义下算出的东西一致，没有什么需要再修正的了。

**Q12.** 批归一化沿*批次*（batch）维度归约，每个通道得到一对统计量；层归一化沿特征维度归约，每个样本得到一对统计量。记 $\mu_c = \frac1N\sum_n h_{n,c}$、$\sigma_c^2 = \frac1N\sum_n(h_{n,c}-\mu_c)^2$，批归一化计算 $\hat h_{n,c} = (h_{n,c}-\mu_c)/\sqrt{\sigma_c^2+\epsilon}$——这组统计量由批次里的每个样本共享，每个通道 $c$（第 $0$ 轴）一对。记 $\mu_n = \frac1C\sum_c h_{n,c}$、$\sigma_n^2 = \frac1C\sum_c(h_{n,c}-\mu_n)^2$，层归一化计算 $\hat h_{n,c} = (h_{n,c}-\mu_n)/\sqrt{\sigma_n^2+\epsilon}$——这组统计量是样本 $n$ 私有的，每一行（第 $1$ 轴）一对，与其他任何一行、以及批次的大小都无关。在推理时，批归一化不再从当前批次计算 $\mu_c,\sigma_c^2$，而是改用训练过程中积累下来的滑动平均；这是必要的，因为推理时的批次可能很小，甚至只包含一个样本，此时批次*内部*的方差是退化的——只有一个样本时方差恰好是 $0$，会让归一化后的值全部塌缩为 $0$，与输入无关，下面会展示这一点——而层归一化从来没有这个问题，因为它的统计量来自单个样本自己的特征，无论是训练还是推理都完全不需要批次的上下文信息。这也是 Transformer 架构偏爱层归一化（或者 RMSNorm，它去掉了减均值这一步，只用 $\sqrt{\frac1C\sum_c h_{n,c}^2}$ 归一化）而不是批归一化的原因：序列是变长处理的，而且常常一次解码一个 token、也就是一个样本，逐样本的统计量无论批次大小或组成如何都表现一致，而批次统计量在训练时会把互不相关的序列耦合在一起，在推理的小批次场景下则会变得不可靠或者干脆没有意义。

### 回归损失与生成模型

**Q13.** L2 损失的最小值点是均值，L1 损失的最小值点是中位数：平方项的导数无界，且随残差增大而增大；而
$|\cdot|$ 的次梯度有界于 $[-1,1]$，只编码符号——所以单个离群值可以把 L2 的最小值点拖到任意远，却几乎不会移动
L1 的最小值点。令 $\frac{d}{dc}\sum_i(y_i-c)^2=-2\sum_i(y_i-c)=0$，得到 $c^\star=\frac1n\sum_i y_i$，即均值。对
$\sum_i|y_i-c|$，在可导的地方其导数是 $\sum_i\mathrm{sign}(c-y_i)=\#\{y_i<c\}-\#\{y_i>c\}$，所以驻点要求 $c$
两侧的 $y_i$ 个数相等；在 $c$ 恰好等于某个 $y_i$ 处，次梯度是同样的计数之差，再加上 $[-1,1]$ 中的一段贡献，而
$0$ 落在其中当且仅当 $c$ 是一个中位数——当 $n$ 为奇数时唯一。直接求导，$\partial r^2/\partial r=2r$ 随
$|r|\to\infty$ 无界增长，只在 $r=0$ 处为零；而 $|r|$ 的一个次梯度，在 $r\ne0$ 时是 $\mathrm{sign}(r)=\pm1$，在
$r=0$ 处则是 $[-1,1]$ 中的任意值：一个 $|r_i|$ 极大的离群值，通过 L2 的导数会按其量级对 $c$ 产生拉力，但通过
L1 的导数却始终只贡献 $\pm1$——所以重要的只是它在哪一侧，而不是它离 $c$ 的距离，下面通过把一个离群值的取值移到
$10^6$ 验证了这一点。

设残差 $r_i=y_i-f(x_i)$ 独立同分布，密度为 $p$；对 $f$ 最小化 $-\sum_i\log p(r_i)$ 就是最大似然估计。对高斯分布
$p(r)\propto e^{-r^2/(2\sigma^2)}$，$-\log p(r)=r^2/(2\sigma^2)+\mathrm{常数}$，所以最小化它恰好就是最小化
$\sum_i r_i^2$：L2 损失在高斯噪声下就是最大似然估计。对拉普拉斯分布 $p(r)\propto e^{-|r|/b}$，同样有
$-\log p(r)=|r|/b+\mathrm{常数}$，这使得 L1 损失在拉普拉斯噪声下是最大似然估计。

Huber 损失的导数是：$|r|\le\delta$ 时为 $r$，$|r|>\delta$ 时为 $\delta\,\mathrm{sign}(r)$（在 $|r|=\delta$ 处连
续）：在拟合较好的地方是二次的，超过 $\delta$ 之后则是斜率有界的线性函数——所以它适合存在离群值污染、但污染并
不压倒性的情形：既保留了 L2 损失在最优点附近收敛快的优点，又像 L1 损失一样，给任何单个残差可能造成的破坏设了
上限。当这两个范数被用作*正则化项*而不是损失时（Q10），同样的转折点与光滑性之间的分野会再次出现：$\ell_1$
惩罚项在 $0$ 处的转折点，会把较小的系数精确压到零，而 $\ell_2$ 惩罚项光滑的梯度只会收缩系数。

**Q14.** 最优判别器是 $D^\star(x)=p_{\mathrm{data}}(x)/(p_{\mathrm{data}}(x)+p_g(x))$；此时博弈的值是 $-\log4$
加上两倍的、数据分布与生成器分布之间的 Jensen–Shannon 散度，对 $G$ 取全局最小值当且仅当 $p_g=p_{\mathrm{data}}$；
而在训练刚开始、判别器还很自信地拒绝生成样本时，原始生成器损失关于判别器 logit 的梯度会消失，而非饱和损失的梯度
不会——这正是实践中选择最小化后者的原因。

固定 $G$ 时，代入 $x=G(z)$ 可得 $V(D,G)=\int\bigl[p_{\mathrm{data}}(x)\log D(x)+p_g(x)\log(1-D(x))\bigr]\,\mathrm dx$，
于是 $D(x)$ 可以在每个 $x$ 上独立选取。对常数 $a,b>0$，在 $y\in(0,1)$ 上最大化 $a\log y+b\log(1-y)$：导数
$a/y-b/(1-y)$ 在 $y^\star=a/(a+b)$ 处为零，且此处二阶导数为负，确认这是一个最大值点；取
$a=p_{\mathrm{data}}(x)$、$b=p_g(x)$，即得到 $D^\star$。把 $D^\star$ 代回，并记 $m=(p_{\mathrm{data}}+p_g)/2$：

$$V(D^\star,G) = \mathrm{KL}(p_{\mathrm{data}}\Vert m) + \mathrm{KL}(p_g\Vert m) - \log4 = 2\,\mathrm{JSD}(p_{\mathrm{data}}\Vert p_g) - \log4,$$

这里用 $\int p_{\mathrm{data}}=\int p_g=1$ 把两个 $-\log2$ 合并，再用 JSD 的定义把两个 KL 项合成一个。因为 JSD
是两个非负 KL 散度的平均，所以 $\mathrm{JSD}\ge0$，等号成立当且仅当 $p_g=p_{\mathrm{data}}$，于是 $V(D^\star,G)$
对 $G$ 的全局最小值恰好在此处取到，值为 $-\log4$。

记 $D(G(z))=\sigma(s)$：原始生成器损失 $\log(1-\sigma(s))$ 满足 $\partial/\partial s\,\log(1-\sigma(s))=-\sigma(s)=-D(G(z))$，
而非饱和损失 $-\log\sigma(s)$ 满足 $\partial/\partial s\,[-\log\sigma(s)]=-(1-\sigma(s))=D(G(z))-1$。在
$s\ll0$、也就是 $D(G(z))\approx0$ 时——判别器自信地拒绝生成器的样本，这在训练早期很常见——第一个梯度恰好在
$G$ 最需要信号时消失，而第二个梯度则始终保持在 $\approx-1$；这正是实践中最小化 $-\mathbb E[\log D(G(z))]$
而不是 $\mathbb E[\log(1-D(G(z)))]$ 的原因。

除这种梯度消失之外，还有两种不同的失败模式：*模式坍缩*（mode collapse），即 $G$ 满足于一小部分总能骗过 $D$
的输出，而不去覆盖 $p_{\mathrm{data}}$ 的全部多样性，缓解办法是让 $D$ 能看到跨样本的批次统计量
（*minibatch discrimination*）；以及不稳定、振荡的训练过程，即交替的梯度步永远不会稳定下来，缓解办法是约束
$D$ 满足 Lipschitz 条件（例如像 WGAN-GP 那样加梯度惩罚）。

### 从离线到线上

**Q15.** 五类原因，各有各的检查方法。*训练-线上不一致*（training–serving skew）：某个特征在线上的计算方式和
训练、离线评估时不同，或者是过时的、甚至缺失的；确认方法是记录线上服务为真实请求实际算出的特征，用同样的原始
输入在离线重新计算，然后比对两者（*影子部署*（shadow deployment）在任何用户受影响之前，就在真实流量上做同样
的比对）；一个过时或缺失的特征，通常会表现为服务时的分布坍缩成离线数据里从未出现过的默认值。离线测试时段与线
上真实流量之间的*分布偏移*（distribution shift）：若 $p(x)$ 变了而 $p(y\mid x)$ 没变，是*协变量偏移*（covariate
shift）；若 $p(y)$ 变了而 $p(x\mid y)$ 没变，是*标签偏移*（label shift）；若 $p(y\mid x)$ 本身变了，则是*概念偏
移*（concept shift）；确认方法是比较这两个时段之间的特征分布和标签分布，并做一次*按时间顺序的回测*（time-based
backtest）——只用某个截止时间之前的数据训练，只在严格晚于它的数据上评估，而不是在全部历史数据上做随机划分，
因为随机划分会掩盖前向划分才能揭示的漂移（下面有检查）。当偏移具体是协变量偏移、且源域和目标域的输入密度已知
或可估计时，用 $p_{\mathrm{target}}(x)/p_{\mathrm{source}}(x)$ 对离线评估做*重要性加权*（importance weighting），
能显著改善对线上准确率的估计：由于 $p(y\mid x)$ 在两个域之间保持不变，
$\mathbb E_{x\sim\mathrm{target}}[\mathrm{correct}(x)] = \mathbb E_{x\sim\mathrm{source}}\bigl[\tfrac{p_{\mathrm{target}}(x)}{p_{\mathrm{source}}(x)}\mathrm{correct}(x)\bigr]$，
把一个源域上的期望变成了目标域上的期望（下面有检查）。使离线估计虚高的*数据泄漏*（leakage）：对按时间排序或分
组的数据采用随机划分、而不是遵循时间顺序的划分，训练集和测试集之间出现重复样本，或者*目标泄漏*（target
leakage）——某个特征编码了在服务时其实还不可得的信息；确认方法和上面的回测一样（一个随时间漂移的关系，会让随
机划分给出明显更高的准确率），目标泄漏则要逐个特征检查：在标签公布之前，这个值是否真的已经存在。*指标与目标
不匹配*：确认方法是检查，在以往的上线和 A/B 测试中，这个离线指标是否真的预测了线上收益——如果两者关系很弱，
说明它不是一个可信的代理。*反馈效应*（feedback effect），即线上模型过去自己的决策塑造了数据：确认方法是在它
没有挑选过的数据上评估两个模型——随机化的探索流量，或者按其倾向分数（propensity）重新加权的日志——如果一个
优势只在线上模型挑选过的数据上才存在，它就会在这里消失；一个带护栏指标（例如长期参与度）的 A/B 测试，就能捕
捉到任何在旧模型下收集的数据都无法揭示的循环。

<details>
<summary>验证代码（可运行）</summary>

```python
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import jensenshannon
from scipy.special import log_softmax as scipy_log_softmax, softmax as scipy_softmax
from sklearn.linear_model import Lasso, LogisticRegression, QuantileRegressor, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

rng = np.random.default_rng(2026)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---- Q1: confusion-matrix metrics
TP, FP, FN, TN = 40, 10, 20, 930
total = TP + FP + FN + TN
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)  # NOTE: harmonic mean of P, R -- not (P + R) / 2
assert (accuracy, round(precision, 4), round(recall, 4), round(f1, 4)) == (0.97, 0.8, 0.6667, 0.7273)
predict_all_negative_accuracy = (FP + TN) / total          # every actual negative is correct, every positive missed
assert math.isclose(predict_all_negative_accuracy, 0.94)
assert accuracy - predict_all_negative_accuracy < 0.04      # the classifier barely beats the trivial baseline

# ---- Q2: ROC curve and ROC-AUC
def brute_force_auc(pos_scores, neg_scores):
    pos, neg = np.asarray(pos_scores, dtype=float)[:, None], np.asarray(neg_scores, dtype=float)[None, :]
    wins = (pos > neg).sum()
    ties = (pos == neg).sum()                              # NOTE: ties count for one half, not zero or one
    return (wins + 0.5 * ties) / (pos.size * neg.size)


n_pos, n_neg = 60, 200
pos_scores = np.round(rng.normal(1.0, 1.0, n_pos), 1)       # rounded scores so real ties occur
neg_scores = np.round(rng.normal(0.0, 1.0, n_neg), 1)
y_true = np.r_[np.ones(n_pos), np.zeros(n_neg)]
y_score = np.r_[pos_scores, neg_scores]
assert (pos_scores[:, None] == neg_scores[None, :]).sum() > 0          # confirms ties are actually exercised
auc = roc_auc_score(y_true, y_score)
assert math.isclose(auc, brute_force_auc(pos_scores, neg_scores), rel_tol=1e-9)

neg_scores_dup = np.repeat(neg_scores, 10)                  # every negative duplicated ten times
y_true_dup = np.r_[np.ones(n_pos), np.zeros(neg_scores_dup.size)]
y_score_dup = np.r_[pos_scores, neg_scores_dup]
assert math.isclose(roc_auc_score(y_true_dup, y_score_dup), auc, rel_tol=1e-9)     # ROC-AUC: unchanged
ap, ap_dup = average_precision_score(y_true, y_score), average_precision_score(y_true_dup, y_score_dup)
assert ap_dup < ap - 0.1                                     # average precision: markedly lower

# ---- Q3: cross-entropy vs squared-error gradient with respect to the logit
def bce_from_logit(z, y):
    # NOTE: log(sigmoid(z)) and log(1 - sigmoid(z)) lose precision by cancellation once |z| is a few
    # units from 0; log(1 + exp(-z)) and log(1 + exp(z)) (via logaddexp) do not.
    return y * np.logaddexp(0.0, -z) + (1 - y) * np.logaddexp(0.0, z)


def bce_grad(z, y):
    return sigmoid(z) - y


def se_from_logit(z, y):
    return (sigmoid(z) - y) ** 2


def se_grad(z, y):
    p = sigmoid(z)
    return 2 * (p - y) * p * (1 - p)


h_fd = 1e-6
for _ in range(2000):
    z, y = rng.normal(scale=4.0), float(rng.integers(0, 2))
    fd_bce = (bce_from_logit(z + h_fd, y) - bce_from_logit(z - h_fd, y)) / (2 * h_fd)
    fd_se = (se_from_logit(z + h_fd, y) - se_from_logit(z - h_fd, y)) / (2 * h_fd)
    assert math.isclose(fd_bce, bce_grad(z, y), abs_tol=1e-5)
    assert math.isclose(fd_se, se_grad(z, y), abs_tol=1e-5)

z_wrong = -20.0  # confidently wrong: true label is 1, model is sure it is 0
assert abs(bce_grad(z_wrong, 1.0)) > 0.999
assert abs(se_grad(z_wrong, 1.0)) < 1e-7                      # squared error: gradient has collapsed
assert abs(bce_grad(z_wrong, 1.0)) / abs(se_grad(z_wrong, 1.0)) > 1e7    # by more than seven orders of magnitude

# ---- Q4: logistic-regression NLL, gradient, Hessian, convexity, separability
def nll(w, X, y):
    return np.sum(bce_from_logit(X @ w, y))


def nll_grad(w, X, y):
    return X.T @ (sigmoid(X @ w) - y)


def nll_hessian(w, X):
    s = sigmoid(X @ w) * (1 - sigmoid(X @ w))
    return X.T @ (X * s[:, None])


for _ in range(200):
    n, d = rng.integers(5, 15), rng.integers(2, 6)
    X, y, w = rng.normal(size=(n, d)), rng.integers(0, 2, n).astype(float), rng.normal(size=d)
    v = rng.normal(size=d)
    fd_dir = (nll(w + h_fd * v, X, y) - nll(w - h_fd * v, X, y)) / (2 * h_fd)
    assert math.isclose(fd_dir, nll_grad(w, X, y) @ v, abs_tol=1e-4, rel_tol=1e-4)
    fd_hess = (nll_grad(w + h_fd * v, X, y) - nll_grad(w - h_fd * v, X, y)) / (2 * h_fd)
    H = nll_hessian(w, X)
    assert np.allclose(fd_hess, H @ v, atol=1e-4)
    # NOTE: eigvalsh, not eig -- H is symmetric by construction, but eig() can return spurious tiny
    # imaginary parts from rounding, and comparing a complex number with >= raises.
    assert np.linalg.eigvalsh(H).min() >= -1e-10

w_dir = rng.normal(size=5)
w_dir /= np.linalg.norm(w_dir)
half, gap, radius = 100, 3.0, 0.3
pos_half = gap * w_dir + rng.uniform(-1, 1, size=(half, 5)) * radius
neg_half = -gap * w_dir + rng.uniform(-1, 1, size=(half, 5)) * radius
X_sep, y_sep = np.vstack([pos_half, neg_half]), np.r_[np.ones(half), np.zeros(half)]
# every point's margin along w_dir exceeds gap - radius*sqrt(5) > 0 by Cauchy-Schwarz, so this is separable
# for every draw of the noise, not just with high probability
assert np.array_equal((X_sep @ w_dir > 0).astype(float), y_sep)


def gradient_descent(X, y, lam, eta, steps, checkpoint_every=200):
    w = np.zeros(X.shape[1])
    norms = []
    for t in range(steps):
        w = w - eta * (nll_grad(w, X, y) + lam * w)
        if t % checkpoint_every == 0 or t == steps - 1:
            norms.append(np.linalg.norm(w))
    return norms


norms_plain = gradient_descent(X_sep, y_sep, 0.0, 0.01, 4000)
norms_ridge = gradient_descent(X_sep, y_sep, 1.0, 0.01, 4000)
assert norms_plain[-1] > norms_plain[len(norms_plain) // 2] * 1.03     # unregularised: still growing (+3% or more)
tail = norms_ridge[-len(norms_ridge) // 4:]
assert max(tail) - min(tail) < 0.02 * norms_ridge[-1]                  # regularised: has settled (within 2%)

# ---- Q5: softmax cross-entropy gradient with respect to the logits, log-sum-exp
def log_sum_exp(z):
    # NOTE: subtract m before exp, not after -- shifting after exponentiating changes nothing.
    m = np.max(z, axis=-1, keepdims=True)
    return m[..., 0] + np.log(np.sum(np.exp(z - m), axis=-1))


def softmax_stable(z):
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_ce_loss(z, y_idx):
    picked = np.take_along_axis(z, y_idx[..., None], axis=-1)[..., 0]
    return log_sum_exp(z) - picked


def softmax_ce_grad(z, y_idx):
    q = softmax_stable(z)
    onehot = np.zeros_like(z)
    np.put_along_axis(onehot, y_idx[..., None], 1.0, axis=-1)
    return q - onehot


for _ in range(500):
    K = int(rng.integers(2, 6))
    z, y_idx = rng.normal(scale=5.0, size=K), np.array(rng.integers(0, K))
    analytic = softmax_ce_grad(z[None, :], y_idx[None])[0]
    fd = np.array([(softmax_ce_loss((z + h_fd * e)[None, :], y_idx[None])[0]
                    - softmax_ce_loss((z - h_fd * e)[None, :], y_idx[None])[0]) / (2 * h_fd)
                   for e in np.eye(K)])
    assert np.allclose(analytic, fd, atol=1e-5)

for _ in range(50):                                      # matches an independent library implementation
    K = int(rng.integers(2, 8))
    z = rng.normal(scale=3.0, size=(4, K))
    assert np.allclose(softmax_stable(z), scipy_softmax(z, axis=-1))
    assert np.allclose(z - log_sum_exp(z)[:, None], scipy_log_softmax(z, axis=-1))

big_logits = np.array([1000.0, 1.0, 0.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big_logits) / np.exp(big_logits).sum()
assert np.isnan(naive).any()                              # naive softmax: breaks
stable = softmax_stable(big_logits)
assert np.all(np.isfinite(stable)) and math.isclose(stable[0], 1.0, abs_tol=1e-12)     # shifted: exact and finite

# ---- Q6: SGD, momentum, Adam; bias correction; first-step size
def adam_moments(m, v, t, g, beta1, beta2):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    mhat = m / (1 - beta1 ** t)          # NOTE: the exponent is t, the number of updates so far -- using
    vhat = v / (1 - beta2 ** t)          #       t - 1 here would leave the very first step uncorrected
    return m, v, mhat, vhat


BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def m_t_closed_form(grads, beta1):
    t = len(grads)
    weights = beta1 ** (t - 1 - np.arange(t))
    return (1 - beta1) * np.sum(weights * np.asarray(grads))


for _ in range(20):
    grads = rng.normal(size=int(rng.integers(1, 30)))
    m = 0.0
    for g in grads:
        m, _, _, _ = adam_moments(m, 0.0, 1, g, BETA1, BETA2)   # t is irrelevant to the m update itself
    assert math.isclose(m, m_t_closed_form(grads, BETA1), rel_tol=1e-9)

mu_g, sigma_g, reps = 2.5, 1.0, 20000
for t in (1, 3, 10):
    sample = rng.normal(mu_g, sigma_g, size=(reps, t))
    m = np.zeros(reps)
    for k in range(t):
        m = BETA1 * m + (1 - BETA1) * sample[:, k]
    assert math.isclose(m.mean(), (1 - BETA1 ** t) * mu_g, rel_tol=0.05, abs_tol=0.05)   # m_t is biased low
    assert math.isclose(m.mean() / (1 - BETA1 ** t), mu_g, rel_tol=0.05, abs_tol=0.05)    # correction fixes it

g1 = np.array([1e-3, 1.0, 1e3, -50.0])                    # gradients spanning six orders of magnitude
_, _, mhat1, vhat1 = adam_moments(0.0, 0.0, 1, g1, BETA1, BETA2)
assert np.allclose(mhat1, g1) and np.allclose(vhat1, g1 ** 2)      # exact at t=1, whatever beta1, beta2 are
step1 = 0.01 * mhat1 / (np.sqrt(vhat1) + ADAM_EPS)
assert np.all(np.isclose(np.abs(step1), 0.01, rtol=1e-4))          # every coordinate moves by about lr

# ---- Q7: L2 regularisation versus decoupled weight decay
def sgd_l2_step(w, g, eta, lam):
    return w - eta * g - eta * lam * w


def sgd_decoupled_step(w, g, eta, gamma):
    return (1 - gamma) * w - eta * g


for _ in range(200):
    d = int(rng.integers(1, 6))
    w, g = rng.normal(size=d), rng.normal(size=d)
    eta, lam = rng.uniform(0.001, 0.5), rng.uniform(0.0, 2.0)
    assert np.allclose(sgd_l2_step(w, g, eta, lam), sgd_decoupled_step(w, g, eta, eta * lam), atol=1e-12)


def adam_l2_run(w0, grads, eta, lam):
    w, m, v = w0.copy(), np.zeros_like(w0), np.zeros_like(w0)
    for t, g in enumerate(grads, start=1):
        m, v, mhat, vhat = adam_moments(m, v, t, g + lam * w, BETA1, BETA2)      # decay folded into g first
        w = w - eta * mhat / (np.sqrt(vhat) + ADAM_EPS)                          # -- so it is divided by
    return w                                                                     #    sqrt(vhat) too


def adamw_run(w0, grads, eta, gamma):
    w, m, v = w0.copy(), np.zeros_like(w0), np.zeros_like(w0)
    for t, g in enumerate(grads, start=1):
        m, v, mhat, vhat = adam_moments(m, v, t, g, BETA1, BETA2)                # decay never touches g,
        w = (1 - gamma) * w - eta * mhat / (np.sqrt(vhat) + ADAM_EPS)            # m or v -- applied raw
    return w


w0 = rng.normal(size=4)
eta, lam = 0.05, 0.1
grads = [rng.normal(scale=s, size=4) for s in (0.01, 1.0, 5.0)]                  # gradients of very different
assert not np.allclose(adam_l2_run(w0, grads, eta, lam),                          # scale across coordinates
                        adamw_run(w0, grads, eta, eta * lam), atol=1e-3)

# ---- Q8: warmup and later decay, mechanism checked via a noisy quadratic
NOISE_SIGMA = 2.0


def stationary_var(eta):
    return eta * NOISE_SIGMA ** 2 / (2 - eta)                # requires 0 < eta < 2 for the recursion to be stable


def run_noisy_gd(eta, steps, n_chains, theta0):
    theta = np.full(n_chains, theta0)
    for _ in range(steps):
        theta = (1 - eta) * theta - eta * rng.normal(scale=NOISE_SIGMA, size=n_chains)
    return theta


n_chains = 20000
theta_big = run_noisy_gd(0.2, 400, n_chains, 0.0)
assert math.isclose(theta_big.var(), stationary_var(0.2), rel_tol=0.1)
theta_small = run_noisy_gd(0.05, 400, n_chains, 0.0)
assert math.isclose(theta_small.var(), stationary_var(0.05), rel_tol=0.1)
theta_decayed = theta_big.copy()                              # continue the SAME chains at a smaller eta
for _ in range(400):
    theta_decayed = (1 - 0.05) * theta_decayed - 0.05 * rng.normal(scale=NOISE_SIGMA, size=n_chains)
assert math.isclose(theta_decayed.var(), stationary_var(0.05), rel_tol=0.1)
assert theta_decayed.var() < theta_big.var() / 2               # decaying eta shrinks the steady-state spread

# ---- Q9: bias-variance decomposition, by simulation
def true_f(x):
    return np.sin(3.0 * x)


x_train, x0, noise_sigma, degree = np.linspace(-1.0, 1.0, 15), 0.7, 0.3, 1
R = 6000
preds, targets = np.empty(R), np.empty(R)
for r in range(R):
    y_train = true_f(x_train) + rng.normal(scale=noise_sigma, size=x_train.size)
    preds[r] = np.polyval(np.polyfit(x_train, y_train, degree), x0)
    targets[r] = true_f(x0) + rng.normal(scale=noise_sigma)   # NOTE: a *fresh* draw of test-point noise --
                                                                #       reusing training noise would be wrong
bias2, variance = (preds.mean() - true_f(x0)) ** 2, preds.var()
measured_mse = np.mean((targets - preds) ** 2)
assert math.isclose(measured_mse, bias2 + variance + noise_sigma ** 2, rel_tol=0.1)
assert bias2 > 3 * variance                                    # the degree-1 fit is dominated by bias, as expected

# ---- Q10: ridge closed form, SVD shrinkage, lasso vs ridge sparsity
n, d = 60, 8
X = rng.normal(size=(n, d))
y = X @ rng.normal(size=d) + rng.normal(scale=0.5, size=n)
lam_ridge = 2.5
w_closed = np.linalg.solve(X.T @ X + lam_ridge * np.eye(d), X.T @ y)
assert np.allclose(w_closed, Ridge(alpha=lam_ridge, fit_intercept=False).fit(X, y).coef_, atol=1e-8)

U, svals, Vt = np.linalg.svd(X, full_matrices=False)            # NOTE: svd returns V^T, not V
w_svd = Vt.T @ ((svals / (svals ** 2 + lam_ridge)) * (U.T @ y))
assert np.allclose(w_closed, w_svd, atol=1e-8)
ols_in_v_basis = (1.0 / svals) * (U.T @ y)
ridge_in_v_basis = (svals / (svals ** 2 + lam_ridge)) * (U.T @ y)
assert np.allclose(ridge_in_v_basis / ols_in_v_basis, svals ** 2 / (svals ** 2 + lam_ridge), atol=1e-10)

n_orth = 30
Q_orth, _ = np.linalg.qr(rng.normal(size=(n_orth, 6)))           # orthonormal columns: Q^T Q = I
y_orth = Q_orth @ np.array([2.0, -1.5, 0.05, 0.0, 3.0, -0.02]) + rng.normal(scale=0.15, size=n_orth)
c = Q_orth.T @ y_orth                                             # (1/2)||y-Xw||^2 = const + (1/2) sum_i (w_i-c_i)^2

lam_l1 = 0.9                                                      # minimises (1/2)(w_i-c_i)^2 + lam_l1|w_i| per i
soft_threshold = np.sign(c) * np.maximum(np.abs(c) - lam_l1, 0.0)
# NOTE: sklearn's Lasso objective is (1/(2n))||y-Xw||^2 + alpha||w||_1; alpha = lam_l1 / n matches it exactly
# to (1/2)||y-Xw||^2 + lam_l1||w||_1 up to the positive constant factor n, which does not move the minimiser
lasso_coef = Lasso(alpha=lam_l1 / n_orth, fit_intercept=False, tol=1e-12, max_iter=100000).fit(Q_orth, y_orth).coef_
assert np.allclose(soft_threshold, lasso_coef, atol=1e-6)
assert np.sum(soft_threshold == 0.0) >= 2

lam_l2 = 0.9                                                      # ridge: minimises ||y-Xw||^2 + lam_l2||w||^2
ridge_coef = Ridge(alpha=lam_l2, fit_intercept=False).fit(Q_orth, y_orth).coef_
assert np.allclose(ridge_coef, c / (1 + lam_l2), atol=1e-8)       # closed form on this orthonormal design
assert np.all(np.abs(ridge_coef) > 1e-4)                          # never exactly zero

# ---- Q11: inverted dropout preserves the expectation
p_drop, acts, reps = 0.3, np.array([2.5, -4.0, 0.8]), 400000
keep = rng.random((reps, acts.size)) > p_drop
inverted = keep * acts / (1 - p_drop)
assert np.allclose(inverted.mean(axis=0), acts, rtol=0.02, atol=0.02)
assert np.allclose(inverted.var(axis=0), p_drop / (1 - p_drop) * acts ** 2, rtol=0.05)
uncorrected = keep * acts                                          # NOTE: without the 1/(1-p) factor the mean
assert np.allclose(uncorrected.mean(axis=0), (1 - p_drop) * acts, rtol=0.02)     #       is biased down by (1-p)

# ---- Q12: batch norm versus layer norm
def batch_norm(x, eps=1e-5):
    mu, var = x.mean(axis=0, keepdims=True), x.var(axis=0, keepdims=True)   # NOTE: axis=0 -- across the batch,
    return (x - mu) / np.sqrt(var + eps)                                   #       one pair of stats per channel


def layer_norm(x, eps=1e-5):
    mu, var = x.mean(axis=1, keepdims=True), x.var(axis=1, keepdims=True)   # NOTE: axis=1 -- across the features
    return (x - mu) / np.sqrt(var + eps)                                   #       of one example, batch-independent


N, C = 5, 4
acts2 = rng.normal(size=(N, C)) * 3 + 1
bn, ln = batch_norm(acts2), layer_norm(acts2)
assert np.allclose(bn.mean(axis=0), 0.0, atol=1e-8) and np.allclose(bn.std(axis=0), 1.0, atol=1e-3)
assert np.allclose(ln.mean(axis=1), 0.0, atol=1e-8) and np.allclose(ln.std(axis=1), 1.0, atol=1e-3)

perturbed = acts2.copy()
perturbed[0] += 5.0
assert not np.allclose(batch_norm(perturbed)[1:], bn[1:])          # batch norm: every row shares the statistics
assert np.allclose(layer_norm(perturbed)[1:], ln[1:], atol=1e-10)  # layer norm: rows are independent

single = acts2[:1]
assert np.allclose(batch_norm(single), 0.0)                        # a batch of one has zero variance: collapses
running_mean, running_var = acts2.mean(axis=0), acts2.var(axis=0)
inference_bn = (single - running_mean) / np.sqrt(running_var + 1e-5)
assert not np.allclose(inference_bn, 0.0)                          # inference uses the running statistics instead
assert np.allclose(layer_norm(single), ln[:1])                     # layer norm needs no such special-casing

# ---- Q13: L1 vs L2 regression losses, Huber loss, MLE noise models
def huber(r, delta):
    r = np.asarray(r, dtype=float)
    return np.where(np.abs(r) <= delta, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta))


def huber_grad(r, delta):
    r = np.asarray(r, dtype=float)
    return np.where(np.abs(r) <= delta, r, delta * np.sign(r))


y_reg = np.concatenate([rng.normal(10.0, 1.0, 30), [80.0]])   # 31 values (odd -- a unique median), one far outlier
bounds_c = (y_reg.min() - 10.0, 2e6)
c_l2 = minimize_scalar(lambda c: np.sum((y_reg - c) ** 2), bounds=bounds_c, method="bounded").x
c_l1 = minimize_scalar(lambda c: np.sum(np.abs(y_reg - c)), bounds=bounds_c, method="bounded").x
assert math.isclose(c_l2, y_reg.mean(), abs_tol=1e-2)
assert math.isclose(c_l1, np.median(y_reg), abs_tol=1e-2)

y_reg_moved = y_reg.copy()
y_reg_moved[-1] = 1e6                                            # push the single outlier much further away
c_l2_moved = minimize_scalar(lambda c: np.sum((y_reg_moved - c) ** 2), bounds=bounds_c, method="bounded").x
c_l1_moved = minimize_scalar(lambda c: np.sum(np.abs(y_reg_moved - c)), bounds=bounds_c, method="bounded").x
assert abs(c_l2_moved - c_l2) > 1000                              # L2 minimiser: dragged far by the outlier's size
assert math.isclose(c_l1_moved, np.median(y_reg_moved), abs_tol=1e-2)
assert abs(c_l1_moved - c_l1) < 1e-1                              # L1 minimiser: essentially unmoved

delta = 1.5
r_grid = np.concatenate([rng.uniform(-4.0, 4.0, 200), [delta, -delta]])   # NOTE: includes both sides of the kink
for r in r_grid:
    fd = (huber(r + h_fd, delta) - huber(r - h_fd, delta)) / (2 * h_fd)
    assert math.isclose(fd, float(huber_grad(r, delta)), abs_tol=1e-4)

n_reg, true_slope = 60, 2.0                                      # L1-loss linear fit versus OLS, contaminated data
x_reg = rng.normal(size=n_reg)
y_reg_lin = true_slope * x_reg + rng.normal(scale=0.2, size=n_reg)
x_outliers = np.array([2.5, -2.5, 2.8, -2.8])
y_outliers = np.array([-20.0, 20.0, -25.0, 25.0])                 # NOTE: sign deliberately opposes the true slope
x_contam = np.concatenate([x_reg, x_outliers])
y_contam = np.concatenate([y_reg_lin, y_outliers])
l1_slope = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(x_contam[:, None], y_contam).coef_[0]
ols_slope = np.polyfit(x_contam, y_contam, 1)[0]
assert abs(l1_slope - true_slope) < 0.3
assert abs(ols_slope - true_slope) > 3 * abs(l1_slope - true_slope)   # L1 fit: far closer to the true slope

# ---- Q14: GAN optimal discriminator, Jensen-Shannon divergence, saturating vs non-saturating generator loss
support_size = 6
p_data = rng.dirichlet(np.ones(support_size))
p_g = rng.dirichlet(np.ones(support_size))


def V_of_D(D_vals, p_data, p_g):
    D_vals = np.clip(D_vals, 1e-12, 1 - 1e-12)                    # NOTE: guard log() for the random D's tested below
    return np.sum(p_data * np.log(D_vals) + p_g * np.log(1 - D_vals))


D_star = p_data / (p_data + p_g)
v_star = V_of_D(D_star, p_data, p_g)
for _ in range(500):
    D_random = rng.uniform(1e-6, 1 - 1e-6, size=support_size)
    assert V_of_D(D_random, p_data, p_g) <= v_star + 1e-9          # D* maximises V pointwise, hence in total

def kl(p, q):
    return np.sum(p * np.log(p / q))


m_mix = (p_data + p_g) / 2
jsd_manual = 0.5 * kl(p_data, m_mix) + 0.5 * kl(p_g, m_mix)
jsd_scipy = jensenshannon(p_data, p_g, base=np.e) ** 2              # NOTE: scipy returns the square root of the JSD
assert math.isclose(jsd_manual, jsd_scipy, rel_tol=1e-9)
assert math.isclose(v_star, -math.log(4) + 2 * jsd_manual, rel_tol=1e-9)

p_same = rng.dirichlet(np.ones(support_size))                       # p_g = p_data: JSD collapses to 0
assert math.isclose(V_of_D(p_same / (2 * p_same), p_same, p_same), -math.log(4), abs_tol=1e-9)

d_val = 1e-3                                                        # D(G(z)) close to 0: early in training
s_val = math.log(d_val / (1 - d_val))                               # the logit with sigmoid(s_val) == d_val
assert math.isclose(sigmoid(s_val), d_val, rel_tol=1e-9)


def saturating_loss(s):
    return math.log(1 - sigmoid(s))


def nonsaturating_loss(s):
    return -math.log(sigmoid(s))


grad_saturating = (saturating_loss(s_val + h_fd) - saturating_loss(s_val - h_fd)) / (2 * h_fd)
grad_nonsaturating = (nonsaturating_loss(s_val + h_fd) - nonsaturating_loss(s_val - h_fd)) / (2 * h_fd)
assert math.isclose(grad_saturating, -d_val, abs_tol=1e-4)          # NOTE: -D(G(z)) -- vanishes as D(G(z)) -> 0
assert math.isclose(grad_nonsaturating, d_val - 1, abs_tol=1e-4)    # NOTE: D(G(z)) - 1 -- stays near -1
assert abs(grad_nonsaturating) > 500 * abs(grad_saturating)         # non-saturating: far larger gradient magnitude

# ---- Q15: offline/online gaps -- concept drift (random vs. forward split), training-serving skew, covariate shift
n_stream, d_stream = 4000, 2
frac = np.arange(n_stream) / (n_stream - 1)
w0_stream = np.array([3.0, 2.0])
theta_max = 3 * math.pi / 4                                        # NOTE: the true decision boundary rotates
theta = theta_max * frac                                          #       smoothly over time -- concept drift
cos_t, sin_t = np.cos(theta), np.sin(theta)
w_t = np.stack([w0_stream[0] * cos_t - w0_stream[1] * sin_t,
                w0_stream[0] * sin_t + w0_stream[1] * cos_t], axis=1)
X_stream = rng.normal(size=(n_stream, d_stream))
y_stream = (rng.random(n_stream) < sigmoid(np.sum(X_stream * w_t, axis=1))).astype(float)

perm = rng.permutation(n_stream)
n_train_s = int(0.7 * n_stream)
train_r, test_r = perm[:n_train_s], perm[n_train_s:]
acc_random = (LogisticRegression(max_iter=1000).fit(X_stream[train_r], y_stream[train_r])
              .score(X_stream[test_r], y_stream[test_r]))

train_f, test_f = np.arange(n_train_s), np.arange(n_train_s, n_stream)
clf_forward = LogisticRegression(max_iter=1000).fit(X_stream[train_f], y_stream[train_f])
acc_forward = clf_forward.score(X_stream[test_f], y_stream[test_f])

t_future = rng.integers(n_train_s, n_stream, size=20000)            # NOTE: fresh draws from the SAME later period
frac_future = t_future / (n_stream - 1)
theta_future = theta_max * frac_future
cos_f, sin_f = np.cos(theta_future), np.sin(theta_future)
w_future = np.stack([w0_stream[0] * cos_f - w0_stream[1] * sin_f,
                      w0_stream[0] * sin_f + w0_stream[1] * cos_f], axis=1)
X_future = rng.normal(size=(t_future.size, d_stream))
y_future = (rng.random(t_future.size) < sigmoid(np.sum(X_future * w_future, axis=1))).astype(float)
acc_future = clf_forward.score(X_future, y_future)

assert acc_random - acc_forward > 0.1         # random split: optimistic, hides the drift
assert abs(acc_forward - acc_future) < 0.03   # forward split: a faithful estimate of true future performance

n2, d2 = 3000, 3                                                     # training-serving skew
X_skew = rng.normal(size=(n2, d2))
w_true_skew = np.array([2.0, -1.5, 1.0])
y_skew = (rng.random(n2) < sigmoid(X_skew @ w_true_skew)).astype(float)
train_sk, test_sk = np.arange(n2 // 2), np.arange(n2 // 2, n2)
clf_skew = LogisticRegression(max_iter=1000).fit(X_skew[train_sk], y_skew[train_sk])
acc_clean = clf_skew.score(X_skew[test_sk], y_skew[test_sk])
X_served = X_skew[test_sk].copy()
X_served[:, 1] *= 5.0                          # NOTE: one feature computed on a different scale online than offline
acc_skewed = clf_skew.score(X_served, y_skew[test_sk])
assert acc_clean - acc_skewed > 0.05           # training-serving skew: a checked drop in accuracy


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


mu_source, mu_target, sigma_cs = 0.0, 2.0, 1.0                       # covariate shift: p(x) differs, p(y|x) shared
n_src_train, n_src_test, n_tgt_eval = 4000, 4000, 200000
x_src_train = rng.normal(mu_source, sigma_cs, n_src_train)
y_src_train = (rng.random(n_src_train) < sigmoid(2.0 * x_src_train)).astype(float)
clf_cs = LogisticRegression(max_iter=1000).fit(x_src_train[:, None], y_src_train)

x_src_test = rng.normal(mu_source, sigma_cs, n_src_test)
y_src_test = (rng.random(n_src_test) < sigmoid(2.0 * x_src_test)).astype(float)
correct_src = (clf_cs.predict(x_src_test[:, None]) == y_src_test).astype(float)
unweighted_estimate = correct_src.mean()
weights = normal_pdf(x_src_test, mu_target, sigma_cs) / normal_pdf(x_src_test, mu_source, sigma_cs)
weighted_estimate = np.sum(weights * correct_src) / np.sum(weights)

x_tgt_eval = rng.normal(mu_target, sigma_cs, n_tgt_eval)
y_tgt_eval = (rng.random(n_tgt_eval) < sigmoid(2.0 * x_tgt_eval)).astype(float)
true_target_accuracy = (clf_cs.predict(x_tgt_eval[:, None]) == y_tgt_eval).mean()

assert abs(weighted_estimate - true_target_accuracy) < abs(unweighted_estimate - true_target_accuracy)
assert abs(weighted_estimate - true_target_accuracy) < 0.02          # importance weighting: close to the truth

print("all checks passed")
```

</details>

</details>
