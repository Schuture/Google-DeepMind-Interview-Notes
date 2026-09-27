# 大模型强化学习：策略梯度、PPO、GRPO 与 DPO

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 知识问答 · 口述，含推导 | ★★★☆☆ | 困难 | RS · RE · MLE | policy-gradient, ppo, grpo, dpo, kl-regularisation, rlhf, reward-hacking, off-policy, importance-sampling | 9 个问题 / 45–60 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

约定贯穿全文的记号：提示 $x$；响应（response）$y=(y_1,\dots,y_n)$，一个长度为 $n$ 的 token 序列；策略（policy）$\pi_\theta(y\mid x) = \prod_{t=1}^n \pi_\theta(y_t \mid x, y_{<t})$，
即正在训练的语言模型，它逐个 token 地生成 $y$，每一步都以提示和已经生成的 token 为条件；参考策略（reference policy）$\pi_{\text{ref}}$，是当前训练阶
段开始之前策略的一份冻结副本；标量奖励（reward）$r(x,y)$，数值越大越好；以及 KL 系数 $\beta > 0$。

### 策略梯度

**Q1.** 固定提示 $x$，记 $J(\theta) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]$。从 $\nabla_\theta \pi_\theta(y\mid x) = \pi_\theta(y\mid x)\,\nabla_\theta \log \pi_\theta(y\mid x)$
出发，推导 $\nabla_\theta J(\theta) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\,\nabla_\theta \log \pi_\theta(y\mid x)\big]$（对
数导数技巧，log-derivative trick）。利用自回归分解，把 $\nabla_\theta \log \pi_\theta(y\mid x)$、进而把 $\nabla_\theta J(\theta)$
写成对 token $t=1,\dots,n$ 求和的形式。然后，对任意不依赖于 $y$ 的基线（baseline）$b$（可以依赖于 $x$），证明 $\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[(r(x,y) - b)\,\nabla_\theta \log \pi_\theta(y\mid x)\big] = \nabla_\theta J(\theta)$
同样成立，即减去 $b$ 不会引入偏差。说明选取得当的基线能带来什么好处。

### PPO

**Q2.** 记 $\rho = \pi_\theta(y\mid x) / \pi_{\theta_{\text{old}}}(y\mid x)$，即正在更新的策略与更新前那份固定快照 $\theta_{\text{old}}$
之间的比率（ratio），并记 $A$ 为 $(x,y)$ 的标量优势（advantage）：当 $y$ 优于当前策略对 $x$ 的平均响应时为正，劣于时为负。单个样本上 PPO 裁剪代理目标（clipped
surrogate）为

$$L^{\text{clip}}(\rho, A) = \min\big(\rho A,\ \operatorname{clip}(\rho,\, 1-\epsilon,\, 1+\epsilon)\, A\big), \qquad \operatorname{clip}(\rho, l, u) = \min(\max(\rho, l), u),$$

其中裁剪范围 $\epsilon \in (0,1)$，在期望意义下对 $\theta$ 最大化。把 $\rho$ 所在的位置分成三个区间——小于 $1-\epsilon$、落在 $[1-\epsilon, 1+\epsilon]$
内、大于 $1+\epsilon$——对每个区间与 $A$ 的符号的组合，给出 $L^{\text{clip}}$ 的取值以及 $\partial L^{\text{clip}} / \partial \rho$（一
共五种情形，因为区间内部 $A$ 的两种符号表现一致）；指出哪些情形下梯度被清零，并对这些情形用信任域（trust region）来解释裁剪保护的是什么。计算 $\epsilon = 0.2$ 时，$(\rho, A) \in \{(1.5, 2), (0.5, 2), (1.5, -2), (0.5, -2)\}$
处 $L^{\text{clip}}$ 的取值。

**Q3.** 在应用于语言模型后训练的 PPO 中，评论家（critic）$V_\phi(x, y_{<t})$ 是与策略一起训练的第二个网络。说明它估计的究竟是什么量。来自奖励模型或验证器的终局奖
励 $r(x,y)$ 只有在整条响应生成完毕后才能拿到，而针对 $\pi_{\text{ref}}$ 的逐 token KL 惩罚（乘以 $\beta$）通常在每一步都要计入；给出把二者合并成的单个逐
token 量 $\tilde r_t$，并记 TD（时序差分，temporal-difference）误差为 $\delta_t = \tilde r_t + \gamma V_\phi(x, y_{\le t}) - V_\phi(x, y_{<t})$（折
扣因子 $\gamma \in (0, 1]$，终止步 $V_\phi(x, y_{\le n}) \doteq 0$），写出广义优势估计（generalised advantage estimation，
GAE）：优势 $A_t^{\mathrm{GAE}(\gamma,\lambda)}$，对轨迹衰减参数（trace-decay parameter）$\lambda \in [0, 1]$，写成 $\delta_t, \delta_{t+1}, \dots$
按 $(\gamma\lambda)$ 幂次加权求和的形式。推导 $\lambda = 0$ 和 $\lambda = 1$ 这两个极限情形。使用评论家具体要付出什么代价？

### GRPO

**Q4.** 组相对策略优化（Group Relative Policy Optimisation，GRPO）针对一个提示 $x$，采样一组（group）$G$ 个响应 $y_1,\dots,y_G \sim \pi_{\theta_{\text{old}}}(\cdot\mid x)$，
并各自用一个标量奖励 $r_i = r(x,y_i)$ 打分（比如某个验证器的输出：$y_i$ 通过测试或与参考答案匹配则为 $1$，否则为 $0$）。响应 $i$ 的每个 token 都获得相同的优
势

$$A_i = \frac{r_i - \bar r}{\operatorname{std}(r) + \delta}, \qquad \bar r = \frac1G\sum_{j=1}^G r_j, \qquad \operatorname{std}(r) = \sqrt{\frac{1}{G-1}\sum_{j=1}^G (r_j-\bar r)^2},$$

其中 $\delta > 0$ 是一个小常数。解释为什么这不需要评论家，为什么这尤其适合可验证的奖励（单元测试、精确匹配的答案），以及当一组的 $G$ 个奖励全部相等时，每个 $A_i$ 会怎样。说出
这种做法的两个已知偏差：一个来自按每条响应自身的 token 长度对其损失贡献做归一化，另一个来自除以组内奖励的标准差。计算奖励为 $(1,0,1,0)$ 的一组和奖励为 $(1,1,1,1)$ 的一
组各自的 $A_i$。

**Q5.** 用一张表格比较 PPO 和 GRPO：各自用什么作为评论家、如何计算优势、内存与方差、各自对奖励有什么要求，以及什么情况下该选哪一个。

### KL 与偏好

**Q6.** 设 $y\sim\pi_\theta$，$u = \pi_{\text{ref}}(y)/\pi_\theta(y)$。估计量 $k_1 = -\log u$ 和 $k_3 = (u-1) - \log u$
都被用来从样本估计 $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \mathbb E_{y\sim\pi_\theta}[\log(\pi_\theta(y)/\pi_{\text{ref}}(y))]$。
证明二者都是无偏的，证明 $k_3 \ge 0$ 对每一个样本都成立（不仅是在期望意义下），并解释为什么实践中更偏好 $k_3$。

**Q7.** 直接偏好优化（Direct Preference Optimisation，DPO）。从 $\max_{\pi} \mathbb E_{y\sim\pi}[r(x,y)] - \beta\,\mathrm{KL}(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x))$
出发（对每个固定的 $x$，在所有条件分布 $\pi(\cdot\mid x)$ 上取最大值），推导闭式最大化解 $\pi^*(y\mid x) = \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)/Z(x)$，
并给出 $Z(x)$。把这个关系式反解出用 $\pi^*$、$\pi_{\text{ref}}$、$Z(x)$ 表示的 $r(x,y)$，再代入 Bradley–Terry 模型——一对响应中较优者胜
过较劣者的概率为 $P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$，其中 $\sigma(z) = 1/(1+e^{-z})$
是 logistic 函数——把 $\pi^*$ 换成可训练的 $\pi_\theta$，得到定义在偏好数据集三元组 $(x,y_w,y_l)$ 上的 DPO 损失。解释为什么 $Z(x)$ 在这个过
程中会被消掉，并说明 DPO 需要什么样的数据来训练。

**Q8.** Reward hacking（钻奖励的空子）与过度优化（over-optimisation）。在针对一个学出来的奖励模型做 RLHF 训练时，描述你会观察哪些症状来判断策略是在过度优化这个奖励模型、
而非真正在变好，并给出具体的缓解办法。

### 同策略与异策略数据

**Q9.** 当一次训练更新所使用的样本来自它正在更新的这个策略本身时，称为同策略（on-policy）；当样本来自另一个策略
时，称为异策略（off-policy）。用行为策略（behaviour policy）$\mu$——训练数据实际采样自的那个分布——和目标策略
（target policy）$\pi$——正在被评估或改进、其期望奖励为 $\mathbb E_{y\sim\pi}[r(y)]$ 的那个分布——来形式化这一
点。把下列做法放到从严格同策略到严格异策略的谱系上，并说明理由：REINFORCE（也就是 Q1 的策略梯度估计量），它在每
次梯度更新前都重新采样一个 $y\sim\pi_\theta$；PPO 和 GRPO，它们针对同一批一次性采样自
$\pi_{\theta_{\text{old}}}$ 的响应做好几次梯度更新；带经验回放缓冲区（replay buffer）的 Q-learning，它使用许多过
去策略采集到的转移（transition）；以及在一个固定的、预先收集好的偏好数据集上训练的 DPO。

写出用 $y\sim\mu$ 的独立同分布样本构造的、$\mathbb E_{y\sim\pi}[r(y)]$ 的重要性采样估计量，并证明只要在每个满足
$\pi(y)\,r(y) \ne 0$ 的 $y$ 处都有 $\mu(y) > 0$，它就是无偏的。现在把一条响应建模为从一个固定词表中独立同分布抽取
的 $T$ 个 token——去掉对历史的自回归条件依赖，从而单独看清序列长度本身对这个估计量做了什么——并记 $w_t =
\pi(y_t)/\mu(y_t)$ 为第 $t$ 个 token 处的比率，$W = \prod_{t=1}^T w_t$ 为由此得到的整条序列的权重。利用逐 token 分
布之间的卡方散度（chi-squared divergence）$\chi^2(\pi\,\|\,\mu) = \sum_y (\pi(y)-\mu(y))^2/\mu(y)$，把
$\operatorname{Var}_\mu(W)$ 推导成 $T$ 和 $\chi^2(\pi\,\|\,\mu)$ 的一个闭式函数，并解释为什么即使 $\pi$ 和 $\mu$ 在
每一个单独 token 上都很接近，它也会随 $T$ 指数增长。说明把 PPO 的裁剪代理目标（Q2）逐 token 地施加、对重要性权重
直接设一个硬上限，以及让 $\pi$ 保持接近 $\mu$ 的 KL 惩罚，各自是如何控制这个方差的，以及各自的代价是什么。最后，
说出两种大模型后训练系统在并非有意为之的情况下就变成 $\mu \ne \pi$ 的方式。

## 参考解答

<details>
<summary>展开参考解答</summary>

有两件事值得先跟面试官说清楚：下文所有 KL 项都是正向的 $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$——也就是在奖励里加一项逐 token 惩
罚时，实际估计的那个方向，而不是反过来；GRPO 的奖励 $r_i$ 是整条响应一个标量，不是逐 token 的，因为组内相对优势只有在每条采样响应恰好对应一个可比较的数值时才有意义。

### 策略梯度

**Q1.** $\nabla_\theta J(\theta) = \nabla_\theta \sum_y \pi_\theta(y\mid x)\, r(x,y) = \sum_y r(x,y)\, \nabla_\theta \pi_\theta(y\mid x)$，
因为 $r$ 不依赖于 $\theta$。在 $\pi_\theta(y\mid x) > 0$ 的地方利用 $\nabla_\theta \log \pi_\theta(y\mid x) = \nabla_\theta \pi_\theta(y\mid x)/\pi_\theta(y\mid x)$（普
通链式法则的改写，这就是整个对数导数技巧），有 $\nabla_\theta \pi_\theta(y\mid x) = \pi_\theta(y\mid x)\,\nabla_\theta \log \pi_\theta(y\mid x)$，
于是

$$\nabla_\theta J(\theta) = \sum_y \pi_\theta(y\mid x)\, r(x,y)\, \nabla_\theta \log \pi_\theta(y\mid x) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\,\nabla_\theta \log \pi_\theta(y\mid x)\big].$$

逐 token 展开：由自回归分解直接得到 $\log \pi_\theta(y\mid x) = \sum_{t=1}^n \log \pi_\theta(y_t\mid x,y_{<t})$，由梯
度的线性性，

$$\nabla_\theta J(\theta) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\Big[r(x,y)\sum_{t=1}^n \nabla_\theta \log \pi_\theta(y_t\mid x,y_{<t})\Big].$$

对于基线，取任意满足 $\nabla_\theta b = 0$ 且不依赖于 $y$（可以依赖于 $x$）的 $b$。由于对任意 $\theta$，概率分布求和都是 $1$，故 $\nabla_\theta \sum_y \pi_\theta(y\mid x) = \nabla_\theta 1 = 0$；
把等式左边按上面同样的方式展开，

$$0 = \nabla_\theta \sum_y \pi_\theta(y\mid x) = \sum_y \pi_\theta(y\mid x)\,\nabla_\theta \log \pi_\theta(y\mid x) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[\nabla_\theta \log \pi_\theta(y\mid x)\big],$$

于是 $b\,\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[\nabla_\theta \log \pi_\theta(y\mid x)] = 0$ 也成立，从奖励
中减去 $b$ 不改变任何东西：

$$\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[(r(x,y)-b)\,\nabla_\theta \log \pi_\theta(y\mid x)\big] = \nabla_\theta J(\theta) - 0 = \nabla_\theta J(\theta).$$

基线永远不会改变这个估计量收敛到的值；选取一个接近 $\mathbb E_{y\sim\pi_\theta}[r(x,y)\mid x]$ 的 $b(x)$，改变的是单个样本估计的噪声大小，而不是它收
敛到的值：它把奖励绝对量级中的大部分从与得分函数相乘的那一项里去掉——这正是使用一个学出来的评论家（Q3），或者不训练网络、直接用采样得到的组均值（Q4）的唯一理由。

### PPO

**Q2.** 裁剪的作用，是把一次已经朝其偏好方向走得够远的更新的梯度恰好清零，而一次错误的更新则永远会被完整地修正。

| $\rho$ 所在区间 | $A$ 的符号 | $L^{\text{clip}}$ | $\partial L^{\text{clip}}/\partial\rho$ |
| --- | --- | --- | --- |
| $[1-\epsilon,\,1+\epsilon]$ | 任意 | $\rho A$ | $A$ |
| $>1+\epsilon$ | $A>0$ | $(1+\epsilon)A$ | $0$ |
| $>1+\epsilon$ | $A<0$ | $\rho A$ | $A$ |
| $<1-\epsilon$ | $A>0$ | $\rho A$ | $A$ |
| $<1-\epsilon$ | $A<0$ | $(1-\epsilon)A$ | $0$ |

（区间内部，$\min$ 的两个分支对 $A$ 的两种符号都是一致的，所以算一种情形而不是两种。）梯度恰好为 $0$ 只发生在两行：$\rho$ 已经沿着 $A$ 的符号所偏好的方向移出了区间——越过
$1+\epsilon$ 且 $A>0$（一个已经不错的响应，其概率已经被推高到超出信任域），或者越过 $1-\epsilon$ 且 $A<0$（一个已经很差的响应，其概率已经被压低到超出信任域）。
其余每一行——无论是在区间内部，还是朝 $A$ 不偏好的方向移出区间，也就是一次真正的错误——梯度都是原封不动的 $A$，所以裁剪从不会阻止一次修正，只会阻止对一次已经在信任域允许范围内被利用到头的移
动的延续。

当 $\epsilon=0.2$（区间为 $[0.8,1.2]$）时：$(\rho,A)=(1.5,2)$ 落在区间之上且 $A>0$（被裁剪）：$\min(3.0,\,2.4)=2.4$。$(\rho,A)=(0.5,2)$
落在区间之下且 $A>0$（未裁剪）：$\min(1.0,\,1.6)=1.0$。$(\rho,A)=(1.5,-2)$ 落在区间之上且 $A<0$（未裁剪）：$\min(-3.0,\,-2.4)=-3.0$。
$(\rho,A)=(0.5,-2)$ 落在区间之下且 $A<0$（被裁剪）：$\min(-1.0,\,-1.6)=-1.6$。

**Q3.** $V_\phi(x,y_{<t})$ 估计的是生成完前缀 $y_{<t}$ 之后、未来（经过 KL 整形的）回报的期望——也就是从第 $t$ 步到响应结束这一段所有量的总和。终局奖励挂
在最后一个 token 上，而逐 token 的 KL 惩罚被直接并入一个每一步都要计入的量：

$$\tilde r_t = -\beta \log\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\text{ref}}(y_t\mid x,y_{<t})} + \mathbb 1[t=n]\, r(x,y),$$

于是每个 token 都要付出一份小小的 KL 代价，只有最后一个 token 还额外拿到终局奖励。记 $V_k := V_\phi(x, y_{\le k})$ 以简化记号，GAE 为

$$A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{n-t} (\gamma\lambda)^l\, \delta_{t+l}.$$

当 $\lambda=0$ 时只剩 $l=0$ 这一项，$A_t = \delta_t$：单步 TD 残差，方差低，但偏差和 $V_\phi$ 的误差一样大。当 $\lambda=1$ 时，

$$\sum_{l=0}^{n-t}\gamma^l \delta_{t+l} = \sum_{l=0}^{n-t}\gamma^l\big(\tilde r_{t+l} + \gamma V_{t+l} - V_{t+l-1}\big) = \sum_{l=0}^{n-t}\gamma^l \tilde r_{t+l} + \sum_{l=0}^{n-t}\big(\gamma^{l+1}V_{t+l} - \gamma^l V_{t+l-1}\big),$$

第二个和式是一个可以裂项相消的和：相邻各项两两抵消，只留下 $l=0$ 项里的 $-V_{t-1}$，以及最后一项里的 $\gamma^{\,n-t+1}V_n = 0$（终止值被钉在 $0$），所以
$A_t = \sum_{l=0}^{n-t}\gamma^l \tilde r_{t+l} - V_{t-1}$：即（经过 KL 整形的）蒙特卡洛回报减去基线，无论 $V_\phi$ 质量好坏都是
无偏的，但带有采样回报的完整方差。居中的 $\lambda$ 在二者之间做权衡。

具体来说，评论家要付出两个代价：它是与策略一起训练的第二个网络——对大模型而言，这通常是同一个 Transformer 主干再加一个标量输出头的副本，训练时保存的参数量和激活内存大致翻倍——而且它自身
要通过回归去拟合一个在训练过程中仍在变动的目标，所以一旦 $V_\phi$ 没校准好，就会给用到它的每一个优势都带来偏差，而这正是上面 $\lambda=1$ 那个极限情形所去掉的偏差，代价则是方差。

### GRPO

**Q4.** 不需要评论家，是因为 $\bar r$——这个提示在 $G$ 个新样本上的组内经验均值——本身就已经是 $\mathbb E_{y\sim\pi_\theta}[r(x,y)\mid x]$
的一个直接蒙特卡洛估计，正是一个学出来的基线 $V_\phi(x)$ 原本要去逼近的那个量（Q1），所以它不需要第二个网络就能把奖励的量级从估计量里去掉；再除以 $\operatorname{std}(r)+\delta$，
进一步把结果重新标定，使得奖励分散程度不同的各个提示贡献出量级相当的优势。这尤其适合可验证的奖励：一个检查器（单元测试、精确匹配打分）重复运行的代价很小、结果也是确定的，所以对同一个提示采样 $G$ 条
完整响应——这正是 GRPO 为去掉评论家而多付出的唯一代价——是负担得起的，而如果给一条响应打分本身就要跑一次昂贵的、学出来的奖励模型 $G$ 次，那就负担不起了。

当一组的 $G$ 个奖励全部相等时，$\operatorname{std}(r) = 0$ 恰好成立，$\delta>0$ 使每个 $A_i$ 都变成 $0/(0+\delta)=0$：这个提示在这一
步完全没有贡献任何梯度，不论所有响应是全对还是全错。

这种做法的两个已知偏差。第一，按每条响应自身的 token 长度 $|y_i|$ 对其累加的 token 损失做归一化（于是逐 token 的系数是 $A_i/|y_i|$），会让同样大小的 $A_i$
在长响应里每个 token 得到的拉力比短响应里每个 token 更小：长度为 $10$、$A_i=-1$ 的响应，逐 token 系数是 $-0.1$；同样 $A_i=-1$ 摊到 $40$ 个 token
上，系数是 $-0.025$，弱了四倍——于是一个错误的响应仅仅因为更长，每个 token 受到的惩罚就更轻，训练会逐渐偏向更长的错误答案。第二，除以 $\operatorname{std}(r)$ 会
按成功率偏离 $50\%$ 的程度对提示重新加权：对成功概率为 $p$ 的类伯努利奖励，$\operatorname{std}(r)\approx\sqrt{p(1-p)}$ 在 $p=0.5$ 处最
大，随 $p\to0$ 或 $p\to1$ 而缩小，所以一组成功率为 $50\%$ 的提示（$100$ 条响应，$\operatorname{std}(r)\approx0.50$）会把同样为 $1$
的奖励差转成约 $2.0$ 的优势量级差，而成功率为 $10\%$ 或 $90\%$ 的一组（$\operatorname{std}(r)\approx0.30$）会转成约 $3.3$，成功率 $2\%$
的一组（$\operatorname{std}(r)\approx0.14$）会转成约 $7.1$——是均衡组的三倍还多，而这些提示唯一的区别只是难易程度不同。

奖励为 $(1,0,1,0)$ 时：$\bar r=0.5$，$\operatorname{std}(r)=\sqrt{1/3}\approx0.5774$，所以 $A\approx(0.8660,-0.8660,0.8660,-0.8660)$。
奖励为 $(1,1,1,1)$ 时：$A=(0,0,0,0)$。

**Q5.** 二者分别站在内存、方差与评论家训练这场权衡的不同一侧：

| | PPO | GRPO |
| --- | --- | --- |
| 评论家 | 学出来的 $V_\phi(x,y_{<t})$，与策略联合用回归训练 | 无 |
| 优势 | 由针对 $V_\phi$ 的 TD 误差算出的 GAE$(\gamma,\lambda)$（Q3） | 同一提示 $G$ 次采样上做组内归一化的 $(r_i-\bar r)/(\operatorname{std}(r)+\delta)$（Q4） |
| 内存 / 算力 | 两个网络联合训练，往往都是全尺寸的 | 一个网络，但每次更新前要对每个提示做 $G$ 次完整 rollout |
| 方差 | $V_\phi$ 跟得上真实价值时更低；跟不上时有偏差 | 随 $1/\sqrt G$ 下降；没有学出来的函数带来的偏差，但见 Q4 的两个偏差 |
| 对奖励的要求 | 响应中任意位置的任意标量都可以——稠密的逐 token 塑形也没问题 | 要便宜、可重复，才负担得起对每个提示采样 $G$ 条响应；最好是可验证的 |
| 何时选用 | 奖励对每个提示打分多次代价高、有稠密塑形可用，或者第二个网络不是瓶颈 | 奖励便宜且可验证，或者评论家的内存与训练不稳定才是瓶颈 |

### KL 与偏好

**Q6.** 两者在期望意义下估计的是完全同一个量；更偏好 $k_3$，是因为它在每个单独样本上也都非负，而且在关键的地方（$\pi_\theta\approx\pi_{\text{ref}}$ 附
近）方差低得多。

$k_1$ 的无偏性：按定义 $u=\pi_{\text{ref}}(y)/\pi_\theta(y)$，所以精确地有 $-\log u = \log\big(\pi_\theta(y)/\pi_{\text{ref}}(y)\big)$，
于是

$$\mathbb E_{y\sim\pi_\theta}[k_1] = \mathbb E_{y\sim\pi_\theta}\Big[\log\frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}\Big] = \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

这就是 KL 散度的定义本身——不涉及任何近似。

$k_3$ 的无偏性：首先，$\mathbb E_{y\sim\pi_\theta}[u] = \sum_y \pi_\theta(y) \cdot \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} = \sum_y \pi_{\text{ref}}(y) = 1$，
因为 $\pi_{\text{ref}}$ 本身就是同一个支撑集上的一个概率分布。所以 $\mathbb E[u]-1=0$，于是

$$\mathbb E_{y\sim\pi_\theta}[k_3] = \mathbb E[u] - 1 - \mathbb E[\log u] = -\mathbb E[\log u] = \mathbb E[-\log u] = \mathbb E[k_1] = \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}).$$

非负性：对 $u>0$ 记 $f(u) = u - 1 - \log u$，则 $k_3=f(u)$。$f'(u) = 1 - 1/u$ 在 $u<1$ 时为负、在 $u>1$ 时为正，所以 $u=1$
是 $f$ 唯一的全局最小点，且 $f(1)=0$；因此对每个 $u>0$ 都有 $f(u)\ge0$，即 $k_3\ge0$ 对每个样本都成立，只有在 $\pi_\theta(y)=\pi_{\text{ref}}(y)$
处才取等号。

为什么更偏好 $k_3$：在最要紧的情形下——策略被 KL 惩罚本身约束得接近参考策略，所以 $\varepsilon$ 通常很小——记 $u=1+\varepsilon$。由 $\log(1+\varepsilon) = \varepsilon - \varepsilon^2/2 + O(\varepsilon^3)$，

$$k_1 = -\log u = -\varepsilon + O(\varepsilon^2), \qquad k_3 = (u-1)-\log u = \varepsilon - \big(\varepsilon - \tfrac{\varepsilon^2}{2} + O(\varepsilon^3)\big) = \tfrac{\varepsilon^2}{2} + O(\varepsilon^3),$$

于是 $k_1$ 的逐样本波动是 $\varepsilon$ 的一阶量，正负都有可能；而 $k_3$ 的波动是二阶量，永远非负，且量级要小一整阶——例如在 $u=1.01$ 处，$k_1\approx-0.00995$，
而 $k_3\approx4.97\times10^{-5}$，小了两个数量级。$k_1$ 在有限批量上取平均，读数可能是负的，尽管真实的 KL 不可能为负；$k_3$ 从不会这样，而且它更低的方差意
味着要把这个值估准所需的样本更少。

**Q7.** 对固定的 $x$，把 $\pi(\cdot\mid x)$ 当作响应上一个自由的概率分布，最大化

$$J[\pi] = \sum_y \pi(y\mid x)\,r(x,y) - \beta \sum_y \pi(y\mid x) \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$$

约束条件为 $\sum_y \pi(y\mid x)=1$。为这个约束引入 Lagrange 乘子 $\mu$，再对每个 $\pi(y\mid x)$ 分别求导，

$$\frac{\partial}{\partial \pi(y\mid x)}\Big[J[\pi] - \mu\Big(\sum_y \pi(y\mid x)-1\Big)\Big] = r(x,y) - \beta\Big(\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+1\Big) - \mu = 0,$$

于是 $\log\big(\pi(y\mid x)/\pi_{\text{ref}}(y\mid x)\big) = \big(r(x,y)-\mu\big)/\beta - 1$，即 $\pi(y\mid x) = \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)\exp(-\mu/\beta-1)$。
最后这个因子不依赖于 $y$，记它为 $1/Z(x)$，由归一化约束定出：

$$\pi^*(y\mid x) = \frac{\pi_{\text{ref}}(y\mid x)\exp\big(r(x,y)/\beta\big)}{Z(x)}, \qquad Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)\exp\big(r(x,y)/\beta\big).$$

$J$ 关于 $\pi$ 是严格凹的（KL 项是 $\pi$ 的严格凸函数，奖励项是线性的），所以这个驻点就是唯一的全局最大值点。

反解出奖励：$r(x,y) = \beta \log\dfrac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$。代入一对响
应 $(y_w,y_l)$（同一个 $x$ 下被判定更优和更劣的响应）的 Bradley–Terry 模型 $P(y_w \succ y_l\mid x) = \sigma\big(r(x,y_w)-r(x,y_l)\big)$，
奖励只通过差值 $r(x,y_w)-r(x,y_l)$ 起作用，而 $\beta\log Z(x)$ 在 $r(x,y_w)$ 和 $r(x,y_l)$ 里是完全相同的加性项（它只依赖于 $x$，与 $y$
无关），所以在相减时被消掉，根本不需要算出来：

$$r(x,y_w)-r(x,y_l) = \beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}.$$

把（未知的）最优解 $\pi^*$ 换成可训练的 $\pi_\theta$，在偏好数据集 $\mathcal D = \{(x,y_w,y_l)\}$ 上对这个 Bradley–Terry 模型做最大
似然拟合，就得到 DPO 损失

$$\mathcal L_{\text{DPO}}(\theta) = -\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right].$$

DPO 只需要一个静态的偏好三元组数据集 $(x,y_w,y_l)$——$y_w,y_l$ 可以来自任何地方，不需要在线采样，也不需要训练奖励模型——再加上一份冻结的 $\pi_{\text{ref}}$
副本，用来计算两边的对数比率。

**Q8.** 最明显的症状是代理指标和真实指标之间的差距不断拉大：奖励模型自己的打分一路走高，而一个独立的、留出（held-out）的度量——人类偏好胜率，或者在留出的可验证任务上的准确率——却先停
滞、后下降，同时响应往往会变长，因为对很多奖励模型来说，变长是一种不必真正变好、就能显得更好的捷径。其他迹象还包括措辞变得越来越重复或退化、一味迎合（sycophancy），以及输出针对奖励模型的特定
弱点做了调整，而不是针对任务本身。

缓解办法，大致按对症下药的直接程度排列：调高 $\beta$，用达到的奖励换取留在 $\pi_{\text{ref}}$ 周围、奖励模型校准最好的那片区域；使用多个独立训练的奖励模型做集成（用它们的最
小值打分，或者惩罚它们之间的分歧），让钻某一个模型特定弱点的空子不再是免费的，或者只要任务允许就把学出来的奖励换成可验证的奖励（Q4），从根源上消除这个攻击面；加入显式的长度控制或惩罚，使策略无法单靠
写得更多来取胜，这和 GRPO 自身的长度归一化偏差（Q4）针对的是同一种长度膨胀机制；以及用一个不是训练时那个奖励模型的留出裁判，在整个训练过程中持续跟踪，据此提前停止、选取训练检查点，而不是在最后
一步只相信代理奖励。

### 同策略与异策略数据

**Q9.** 同策略是指 $\mu=\pi$：正在被更新的这个策略本身就是产生数据的那个策略；异策略是指 $\mu\ne\pi$。
REINFORCE 是严格意义上的同策略：每一步都在更新前先重新采样一个 $y\sim\pi_\theta$，所以恰好有
$\mu=\pi=\pi_\theta$。PPO 和 GRPO 在设计上是同策略，但在执行上是轻度异策略的：这一批数据一次性采样自
$\mu=\pi_{\theta_{\text{old}}}$，而在针对它做的好几次梯度更新中，第一次之后 $\pi=\pi_\theta$ 就已经移动到了它之
外——这正是 Q2 中 $\rho\ne1$ 的原因，也是裁剪存在的原因。带经验回放缓冲区的 Q-learning 在设计上就是完全异策略
的：它从填充缓冲区时正在运行的、各种各样的探索策略 $\mu$ 产生的转移中，学习一个目标策略 $\pi$，而这些转移可能是许
多次更新之前采集的。DPO 是这里意义最强的异策略：$\mu$ 是产生这个固定的、预先收集好的偏好数据集的任意过程，永不重
新采样，而 $\pi=\pi_\theta$ 在整个训练过程中不断变化，完全没有重要性校正。

从 $y\sim\mu$ 构造的、$\mathbb E_{y\sim\pi}[r(y)]$ 的重要性采样估计量 $\hat J=(\pi(y)/\mu(y))\,r(y)$，只要在每个
满足 $\pi(y)r(y)\ne0$ 的 $y$ 处都有 $\mu(y)>0$，就是无偏的：

$$\mathbb E_{y\sim\mu}\Big[\frac{\pi(y)}{\mu(y)}r(y)\Big] = \sum_{y:\mu(y)>0}\pi(y)r(y) = \mathbb E_{y\sim\pi}[r(y)],$$

在这一假设下，$\mu(y)=0$ 的那些项都恰好是 $0$。

把响应建模为 $T$ 个独立同分布的 token，记 $w_t=\pi(y_t)/\mu(y_t)$，$W=\prod_{t=1}^T w_t$：$\mathbb
E_\mu[w_t]=1$ 在每个位置都成立，由独立性可得 $\mathbb E_\mu[W]=1$，同样地 $\mathbb
E_\mu[W^2]=\big(\mathbb E_\mu[w_t^2]\big)^T$。直接从定义展开 $\chi^2(\pi\,\|\,\mu)=\sum_y\pi(y)^2/\mu(y)-1$，可知
$\mathbb E_\mu[w_t^2]=1+\chi^2(\pi\,\|\,\mu)$，于是

$$\mathbb E_\mu[W^2]=\big(1+\chi^2(\pi\,\|\,\mu)\big)^T,\qquad \operatorname{Var}_\mu(W)=\big(1+\chi^2(\pi\,\|\,\mu)\big)^T-1.$$

记 $\kappa=\chi^2(\pi\,\|\,\mu)>0$——在单个 token 上可能很小——这是关于 $T$ 的指数函数：$T$ 翻倍时
$(1+\kappa)^T$ 恰好被平方，所以单个 token 上都察觉不到的偏差，到 $T=100$ 时就可能压过所有其他噪声来源。

三种控制手段，各有各的代价。PPO 的裁剪代理目标（Q2）逐 token 施加：把每个 $w_t$ 和它自己的优势配对，而不是去构造
$W$，于是没有任何东西被提升到 $T$ 次方；代价是偏差，因为逐 token 的比率忽略了 $\pi$ 的变化也会改变每个 token 所
条件于的前缀分布，而且一个已经越过 $1\pm\epsilon$ 的比率，其梯度会被直接丢弃。直接给权重设上限，
$\tilde W=\min(W,c)$，能把 $\operatorname{Var}_\mu(\tilde W)\le c^2$ 逐点地界定住；代价是只要 $\mu$ 在 $W>c$ 处
有正概率质量，就有 $\mathbb E_\mu[\tilde W]<1$，且随着该质量增大而增大。让 $\pi$ 保持接近 $\mu$ 的 KL 惩罚也会缩
小 $\chi^2(\pi\,\|\,\mu)$，因为在 $\pi=\mu$ 附近 $\mathrm{KL}\approx\tfrac12\chi^2$；代价是 $\pi$ 被限制得不能像
仅靠奖励驱动时那样远离 $\mu$。

有两种并非刻意设计、却会让 $\mu\ne\pi$ 出现的情况。其一，在异步训练中，负责生成 rollout 的 worker 所用的权重落
后于训练器，于是一批采样自某个快照的数据，要等到参数已经移动到该快照之外才被用来训练，这个差距会随生成落后训练
的程度而增大。其二，生成 rollout 的引擎对同样的权重算出的逐 token 概率，和训练器会算出的概率有微小差异（批处理
方式、算子实现或数值精度不同），所以用作 $\mu$ 的那个快照，只是 token 真正采样自的那个分布的一个近似——即便只做
一次梯度更新、训练完全同步，这个差距也依然存在。

<details>
<summary>验证代码（可运行）</summary>

```python
import math

import numpy as np

rng = np.random.default_rng(0)


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---- Q1: the score-function identity, token by token, and the effect of a baseline
# A 2-token response y = (y1, y2) from a 3-way vocabulary, autoregressive: y2's law depends on y1.
THETA1 = np.array([0.2, -0.5, 0.9])               # logits of pi(y1)
THETA2 = np.array([[0.4, 0.1, -0.3],               # logits of pi(y2 | y1 = 0)
                    [-0.2, 0.6, 0.0],              # logits of pi(y2 | y1 = 1)
                    [0.3, -0.4, 0.5]])             # logits of pi(y2 | y1 = 2)
REWARD = np.array([[1.0, -2.0, 0.5],
                    [0.3, 1.5, -1.0],
                    [-0.5, 0.8, 2.0]])             # r(y1, y2): one fixed number per outcome


def expected_reward(theta1, theta2):
    p1 = softmax(theta1)
    total = 0.0
    for y1 in range(3):
        p2 = softmax(theta2[y1])
        total += p1[y1] * float(p2 @ REWARD[y1])
    return total


def score1(theta1, y1):                            # grad_theta1 log pi(y1) = onehot(y1) - pi(y1)
    p1 = softmax(theta1)
    oh = np.zeros(3); oh[y1] = 1.0
    return oh - p1


def score2_row(theta2_row, y2):                     # grad_(theta2[y1]) log pi(y2|y1) = onehot(y2) - pi(.|y1)
    p2 = softmax(theta2_row)
    oh = np.zeros(3); oh[y2] = 1.0
    return oh - p2


def reinforce_grad_exact(theta1, theta2, reward, baseline=0.0):
    """E_{y~pi}[(r(y) - baseline) * grad log pi(y)], by enumeration; grad log pi(y) is summed token by token."""
    p1 = softmax(theta1)
    g1 = np.zeros(3)
    g2 = np.zeros((3, 3))
    for y1 in range(3):
        p2 = softmax(theta2[y1])
        for y2 in range(3):
            p = p1[y1] * p2[y2]
            adv = reward[y1, y2] - baseline
            g1 += p * adv * score1(theta1, y1)
            g2[y1] += p * adv * score2_row(theta2[y1], y2)
    return g1, g2


def fd_grad(theta1, theta2, h=1e-6):
    """Independent brute force: central finite differences directly on expected_reward, no score-function formula."""
    g1 = np.zeros(3)
    for i in range(3):
        tp, tm = theta1.copy(), theta1.copy()
        tp[i] += h; tm[i] -= h
        g1[i] = (expected_reward(tp, theta2) - expected_reward(tm, theta2)) / (2 * h)
    g2 = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            tp, tm = theta2.copy(), theta2.copy()
            tp[r, c] += h; tm[r, c] -= h
            g2[r, c] = (expected_reward(theta1, tp) - expected_reward(theta1, tm)) / (2 * h)
    return g1, g2


g1_exact, g2_exact = reinforce_grad_exact(THETA1, THETA2, REWARD)
g1_fd, g2_fd = fd_grad(THETA1, THETA2)
assert np.allclose(g1_exact, g1_fd, atol=1e-6)
assert np.allclose(g2_exact, g2_fd, atol=1e-6)

N = 200_000                                         # Monte Carlo agreement (vectorised)
p1 = softmax(THETA1)
y1_samples = rng.choice(3, size=N, p=p1)
y2_samples = np.empty(N, dtype=int)
for y1 in range(3):
    mask = y1_samples == y1
    k = int(mask.sum())
    if k:
        y2_samples[mask] = rng.choice(3, size=k, p=softmax(THETA2[y1]))
r_samples = REWARD[y1_samples, y2_samples]
score1_samples = np.eye(3)[y1_samples] - p1[None, :]
mc_g1 = (r_samples[:, None] * score1_samples).mean(axis=0)
mc_g2 = np.zeros((3, 3))
for y1 in range(3):
    mask = y1_samples == y1
    p2 = softmax(THETA2[y1])
    score2_samples = np.eye(3)[y2_samples[mask]] - p2[None, :]
    mc_g2[y1] = (r_samples[mask][:, None] * score2_samples).sum(axis=0) / N
assert np.allclose(mc_g1, g1_exact, atol=0.02)
assert np.allclose(mc_g2, g2_exact, atol=0.05)

mean_reward = expected_reward(THETA1, THETA2)                       # baseline: mean unaffected, variance changes
BASELINES = [0.0, -5.0, mean_reward, 10.0]
for b in BASELINES:
    g1_b, g2_b = reinforce_grad_exact(THETA1, THETA2, REWARD, baseline=b)
    assert np.allclose(g1_b, g1_exact, atol=1e-10)
    assert np.allclose(g2_b, g2_exact, atol=1e-10)


def estimator_second_moment(theta1, theta2, reward, baseline):
    """E[|| one-sample score-function estimator ||^2], enumerated exactly over all 9 outcomes."""
    p1 = softmax(theta1)
    total = 0.0
    for y1 in range(3):
        p2 = softmax(theta2[y1])
        for y2 in range(3):
            p = p1[y1] * p2[y2]
            adv = reward[y1, y2] - baseline
            s1, s2 = score1(theta1, y1), score2_row(theta2[y1], y2)
            total += p * (adv ** 2) * (float(s1 @ s1) + float(s2 @ s2))
    return total


mean_sq_norm = float(g1_exact @ g1_exact) + float((g2_exact * g2_exact).sum())
variances = {b: estimator_second_moment(THETA1, THETA2, REWARD, b) - mean_sq_norm for b in BASELINES}
assert variances[mean_reward] < variances[0.0] < variances[10.0]

# ---- Q2: PPO's clipped surrogate: five distinguishable cases, and finite differences in rho
EPS = 0.2


def ppo_surrogate(rho, A, eps=EPS):
    clipped = min(max(rho, 1 - eps), 1 + eps)
    return min(rho * A, clipped * A)


cases = [(1.5, 2.0), (0.5, 2.0), (1.5, -2.0), (0.5, -2.0)]
surrogate_values = [ppo_surrogate(rho, A) for rho, A in cases]
assert np.allclose(surrogate_values, [2.4, 1.0, -3.0, -1.6])

h = 1e-6
ppo_grads = [(ppo_surrogate(rho + h, A) - ppo_surrogate(rho - h, A)) / (2 * h) for rho, A in cases]
assert np.allclose(ppo_grads, [0.0, 2.0, -2.0, 0.0], atol=1e-4)

for rho in (0.85, 1.0, 1.15):                        # the fifth case: inside the interval, either sign of A
    for A in (3.0, -3.0):
        g = (ppo_surrogate(rho + h, A) - ppo_surrogate(rho - h, A)) / (2 * h)
        assert math.isclose(g, A, rel_tol=1e-4)
        assert math.isclose(ppo_surrogate(rho, A), rho * A, rel_tol=1e-9)

# ---- Q3: the per-token KL-shaped reward and GAE(gamma, lambda)
GAMMA, LAM = 0.97, 0.9
EPISODE_LEN = 6
V = rng.normal(size=EPISODE_LEN + 1)
V[EPISODE_LEN] = 0.0                       # NOTE: the terminal "value after the last token" is pinned to 0
r_tilde = rng.normal(size=EPISODE_LEN) * 0.1


def gae_recursive(r_tilde, V, gamma, lam):
    delta = r_tilde + gamma * V[1:] - V[:-1]
    steps = len(r_tilde)
    A = np.zeros(steps)
    running = 0.0
    for t in reversed(range(steps)):
        running = delta[t] + gamma * lam * running
        A[t] = running
    return A, delta


def gae_geometric_sum(r_tilde, V, gamma, lam):
    """Same recursion, unrolled explicitly as a (gamma*lam)^l-weighted sum of future deltas: an independent path."""
    delta = r_tilde + gamma * V[1:] - V[:-1]
    steps = len(r_tilde)
    A = np.zeros(steps)
    for t in range(steps):
        A[t] = sum((gamma * lam) ** l * delta[t + l] for l in range(steps - t))
    return A


A_rec, delta = gae_recursive(r_tilde, V, GAMMA, LAM)
A_sum = gae_geometric_sum(r_tilde, V, GAMMA, LAM)
assert np.allclose(A_rec, A_sum, atol=1e-10)

A_lam0, _ = gae_recursive(r_tilde, V, GAMMA, 0.0)               # lambda = 0: one-step TD residual
assert np.allclose(A_lam0, delta, atol=1e-10)

A_lam1, _ = gae_recursive(r_tilde, V, GAMMA, 1.0)                # lambda = 1: telescopes to (return-to-go - V)
return_to_go = np.array([sum(GAMMA ** l * r_tilde[t + l] for l in range(EPISODE_LEN - t))
                         for t in range(EPISODE_LEN)])
assert np.allclose(A_lam1, return_to_go - V[:-1], atol=1e-8)

# the per-token KL-shaped reward itself, worked for a 3-token response
TOK_PI_THETA = np.array([0.5, 0.25, 0.8])                   # pi_theta(y_t | x, y_<t) for t = 1, 2, 3
TOK_PI_REF = np.array([0.4, 0.3, 0.6])                      # pi_ref(y_t | x, y_<t), same tokens
BETA_TOK, TERMINAL_R = 0.1, 2.0
tilde_r_worked = -BETA_TOK * np.log(TOK_PI_THETA / TOK_PI_REF)
tilde_r_worked[-1] += TERMINAL_R                            # NOTE: only the terminal token gets the episode reward
assert math.isclose(tilde_r_worked[0], -BETA_TOK * math.log(0.5 / 0.4), rel_tol=1e-9)
assert math.isclose(tilde_r_worked[1], -BETA_TOK * math.log(0.25 / 0.3), rel_tol=1e-9)
assert math.isclose(tilde_r_worked[2], -BETA_TOK * math.log(0.8 / 0.6) + TERMINAL_R, rel_tol=1e-9)
# summed KL cost is the log of a product of ratios: an independent code path to the elementwise formula above
assert math.isclose(tilde_r_worked.sum() - TERMINAL_R, -BETA_TOK * math.log(float(np.prod(TOK_PI_THETA / TOK_PI_REF))), rel_tol=1e-9)

# ---- Q4: GRPO's group-relative advantage
ADV_DELTA = 1e-6


def grpo_advantages(rewards, delta=ADV_DELTA):
    rewards = np.asarray(rewards, dtype=float)
    mean = rewards.mean()
    std = rewards.std(ddof=1) if len(rewards) > 1 else 0.0  # NOTE: ddof=1 (sample std); numpy's default ddof=0 understates it
    return (rewards - mean) / (std + delta)


adv_mixed = grpo_advantages([1, 0, 1, 0])
expected_mixed = (np.array([1, 0, 1, 0]) - 0.5) / (math.sqrt(1 / 3) + ADV_DELTA)
assert np.allclose(adv_mixed, expected_mixed, atol=1e-6)
assert np.allclose(adv_mixed, [0.8660, -0.8660, 0.8660, -0.8660], atol=1e-3)

adv_constant = grpo_advantages([1, 1, 1, 1])
assert np.allclose(adv_constant, [0.0, 0.0, 0.0, 0.0], atol=1e-9)

for _ in range(200):                                # the group mean of the advantage is 0 whenever there is spread
    r = rng.integers(0, 2, size=8).astype(float)
    if r.std() > 0:
        assert abs(grpo_advantages(r).mean()) < 1e-8

GROUP = 100                                          # a group's success rate, exactly p * GROUP successes out of GROUP
group_stds = {}
for p in (0.5, 0.1, 0.9, 0.02):
    r = np.array([1.0] * round(p * GROUP) + [0.0] * (GROUP - round(p * GROUP)))
    group_stds[p] = r.std(ddof=1)
gaps = {p: 1.0 / (group_stds[p] + ADV_DELTA) for p in group_stds}      # |advantage| between a success and a failure
assert gaps[0.5] < gaps[0.1] < gaps[0.02]
assert gaps[0.5] < gaps[0.9]
for p, std_approx, gap_approx in [(0.5, 0.50, 2.0), (0.1, 0.30, 3.3), (0.02, 0.14, 7.1)]:  # the numbers quoted above
    assert math.isclose(group_stds[p], std_approx, abs_tol=0.01)
    assert math.isclose(gaps[p], gap_approx, rel_tol=0.02)
assert gaps[0.02] > 3 * gaps[0.5]

length_coeff = lambda A_i, length: A_i / length      # the per-token pull after dividing by the response length
assert math.isclose(length_coeff(-1.0, 10), -0.1)
assert math.isclose(length_coeff(-1.0, 40), -0.025)
assert abs(length_coeff(-1.0, 40)) < abs(length_coeff(-1.0, 10)) / 3

# ---- Q6: k1 and k3 estimators of KL(pi_theta || pi_ref)
def exact_kl(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def k1_k3_expectations(p, q):
    u = q / p  # NOTE: u = pi_ref / pi_theta = q / p; flipping this ratio breaks both estimators silently
    k1, k3 = -np.log(u), (u - 1) - np.log(u)
    return float(np.sum(p * k1)), float(np.sum(p * k3))


for _ in range(50):
    p, q = softmax(rng.normal(size=6)), softmax(rng.normal(size=6))
    kl = exact_kl(p, q)
    e_k1, e_k3 = k1_k3_expectations(p, q)
    assert math.isclose(e_k1, kl, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(e_k3, kl, rel_tol=1e-9, abs_tol=1e-9)
    u_all = q / p
    assert np.all(((u_all - 1) - np.log(u_all)) >= -1e-12)          # k3 >= 0 pointwise, not just in expectation

for eps in (1e-2, 1e-3, 1e-4):                        # near u = 1: k1 is first order, k3 is second order and >= 0
    u = 1 + eps
    assert math.isclose(-math.log(u), -eps, rel_tol=2e-2)
    assert math.isclose((u - 1) - math.log(u), eps ** 2 / 2, rel_tol=2e-2)

assert math.isclose(-math.log(1.01), -0.00995, rel_tol=1e-3)               # the u = 1.01 numbers quoted above
assert math.isclose((1.01 - 1) - math.log(1.01), 4.97e-5, rel_tol=2e-3)

# ---- Q7: the closed-form optimum of the KL-regularised objective, and the DPO loss
def closed_form_pi_star(pi_ref, reward, beta):
    unnorm = pi_ref * np.exp(reward / beta)
    return unnorm / unnorm.sum()


def kl_regularised_objective(pi, pi_ref, reward, beta):
    """J[pi] = E_pi[r] - beta * KL(pi || pi_ref)."""
    mask = pi > 0
    kl = np.sum(pi[mask] * np.log(pi[mask] / pi_ref[mask]))
    return float(np.sum(pi * reward) - beta * kl)


N_RESPONSES, BETA = 6, 0.3
pi_ref = softmax(rng.normal(size=N_RESPONSES))
reward = rng.normal(size=N_RESPONSES) * 2.0
pi_star = closed_form_pi_star(pi_ref, reward, BETA)
assert math.isclose(pi_star.sum(), 1.0, rel_tol=1e-10)
assert np.all(pi_star > 0)

best = kl_regularised_objective(pi_star, pi_ref, reward, BETA)
competitors = rng.dirichlet(np.ones(N_RESPONSES), size=2000)          # many random competing distributions
scores = np.array([kl_regularised_objective(pi, pi_ref, reward, BETA) for pi in competitors])
assert np.all(scores < best - 1e-9)
assert best - scores.max() > 1e-6                     # a real margin, not a numerical coincidence

log_Z = math.log(float((pi_ref * np.exp(reward / BETA)).sum()))
assert math.isclose(best, BETA * log_Z, rel_tol=1e-8)          # the optimal value itself is beta * log Z(x)

implicit_reward = BETA * np.log(pi_star / pi_ref)      # the DPO implicit reward, from pi_star and pi_ref alone
assert np.allclose(implicit_reward, reward - BETA * log_Z, atol=1e-8)
assert abs(BETA * log_Z) > 1e-3                          # NOTE: Z(x) is a real additive shift, not coincidentally 0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


for _ in range(30):
    yw, yl = rng.choice(N_RESPONSES, size=2, replace=False)
    assert math.isclose(implicit_reward[yw] - implicit_reward[yl], reward[yw] - reward[yl], rel_tol=1e-8)
    p_true = sigmoid(reward[yw] - reward[yl])
    p_from_policy = sigmoid(implicit_reward[yw] - implicit_reward[yl])
    assert math.isclose(p_true, p_from_policy, rel_tol=1e-8)

# ---- Q9: importance sampling across mu != pi, and the variance of a T-token product of ratios
PI_TOK = softmax(np.array([0.3, -0.1, 0.4, 0.0]))     # target policy's fixed per-token law
MU_TOK = softmax(np.array([0.1, 0.15, 0.05, 0.2]))    # behaviour policy's fixed per-token law
VOCAB = len(PI_TOK)
G_TOK = np.array([1.0, -0.5, 2.0, 0.3])               # a fixed per-token reward g(v); r(y) = sum_t g(y_t)
CHI2 = float(((PI_TOK - MU_TOK) ** 2 / MU_TOK).sum())   # chi^2(pi || mu), straight from its definition
assert math.isclose(float((PI_TOK ** 2 / MU_TOK).sum()) - 1.0, CHI2, rel_tol=1e-12)   # NOTE: E_mu[w^2] - 1 with w = pi/mu, not mu/pi
assert CHI2 > 0.02                                     # a real per-token mismatch, not a coincidental near-match


def token_weight_and_reward(tokens):
    """W = prod_t pi(y_t)/mu(y_t) and r(y) = sum_t g(y_t), vectorised over a batch of sequences (n, t_len)."""
    return (PI_TOK[tokens] / MU_TOK[tokens]).prod(axis=1), G_TOK[tokens].sum(axis=1)


def enumerate_sequences(t_len):
    """Every one of VOCAB**t_len sequences with its exact mu-probability: brute force, only for small t_len."""
    grids = np.meshgrid(*([np.arange(VOCAB)] * t_len), indexing="ij")
    tokens = np.stack([g.ravel() for g in grids], axis=1)
    return tokens, MU_TOK[tokens].prod(axis=1)


T_LIST = (1, 2, 4, 8, 16)
N_IS = 200_000
var_formula = {t: (1 + CHI2) ** t - 1 for t in T_LIST}          # the closed form derived above

for t_len in (1, 2, 4):                                          # brute force: every sequence, no IS formula involved
    tokens, mu_prob = enumerate_sequences(t_len)
    pi_prob = PI_TOK[tokens].prod(axis=1)
    exact_by_enum = float((pi_prob * G_TOK[tokens].sum(axis=1)).sum())
    assert math.isclose(exact_by_enum, t_len * float(G_TOK @ PI_TOK), rel_tol=1e-9)  # matches the i.i.d. linearity argument

var_empirical = {}
for t_len in T_LIST:
    tokens = rng.choice(VOCAB, size=(N_IS, t_len), p=MU_TOK)
    w_seq, r_seq = token_weight_and_reward(tokens)
    exact_reward = t_len * float(G_TOK @ PI_TOK)                          # independence structure: T times one token's expectation
    se_mean = math.sqrt(float((w_seq * r_seq).var(ddof=1)) / N_IS)        # NOTE: ddof=1 (sample std); numpy's default understates it
    assert abs(float((w_seq * r_seq).mean()) - exact_reward) < 6 * se_mean      # the IS estimate of E_pi[r(y)], from mu-samples alone
    se_w = math.sqrt(var_formula[t_len] / N_IS)
    assert abs(float(w_seq.mean()) - 1.0) < 8 * se_w + 1e-9                     # E_mu[W] = 1 at every length
    var_empirical[t_len] = float(w_seq.var(ddof=1))
    assert math.isclose(var_empirical[t_len], var_formula[t_len], rel_tol=0.15, abs_tol=0.03)

assert var_empirical[16] > var_empirical[8] > var_empirical[4] > var_empirical[2] > var_empirical[1]
assert (var_empirical[16] - var_empirical[8]) > 2 * (var_empirical[4] - var_empirical[2])   # accelerating: geometric, not linear
ratio_low = (var_formula[4] + 1) / (var_formula[2] + 1)
ratio_high = (var_formula[16] + 1) / (var_formula[8] + 1)
assert math.isclose(ratio_low, (1 + CHI2) ** 2, rel_tol=1e-9) and math.isclose(ratio_high, (1 + CHI2) ** 8, rel_tol=1e-9)

for eps in (1e-2, 1e-3):                          # near mu, KL and chi^2 agree to second order: KL ~ chi^2 / 2
    pi_near = (1 - eps) * MU_TOK + eps * PI_TOK
    kl_near = float((pi_near * np.log(pi_near / MU_TOK)).sum())
    chi2_near = float(((pi_near - MU_TOK) ** 2 / MU_TOK).sum())
    assert math.isclose(kl_near / chi2_near, 0.5, rel_tol=0.05)

# truncating W at a cap: lower variance, and a bias checked against the exact (enumerated) truncated mean
T_TRUNC = 8
tokens, mu_prob = enumerate_sequences(T_TRUNC)
w_all = PI_TOK[tokens].prod(axis=1) / mu_prob
assert math.isclose(float((mu_prob * w_all).sum()), 1.0, rel_tol=1e-9)    # E_mu[W] = 1 exactly, by enumeration
assert w_all.max() > 6.0                                                   # so every cap below actually binds somewhere

samp_tokens = rng.choice(VOCAB, size=(N_IS, T_TRUNC), p=MU_TOK)
w_samp, _ = token_weight_and_reward(samp_tokens)
for cap in (1.5, 3.0, 6.0):
    exact_truncated_mean = float((mu_prob * np.minimum(w_all, cap)).sum())
    assert exact_truncated_mean < 1.0 - 1e-6              # NOTE: min(W, c) <= W pointwise, so the mean can only fall
    truncated_samp = np.minimum(w_samp, cap)
    se_trunc = float(truncated_samp.std(ddof=1)) / math.sqrt(N_IS)
    assert abs(float(truncated_samp.mean()) - exact_truncated_mean) < 8 * se_trunc + 1e-9
    assert truncated_samp.var(ddof=1) < w_samp.var(ddof=1)   # capping large values can only reduce the variance
    assert truncated_samp.var(ddof=1) <= cap ** 2            # and 0 <= min(W, c) <= c bounds it by c^2 outright

print("all checks passed")
```

</details>

</details>
