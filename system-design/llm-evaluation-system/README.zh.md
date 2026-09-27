# 为大模型助手设计评测系统

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 机器学习系统设计 | ★★★★☆ | 困难 | RE · RS · MLE · Applied AI | evaluation, autoraters, side-by-side, bradley-terry, statistical-power, contamination, release-gating | 45–60 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

一个助手产品每周发布一个新的候选模型。候选模型在进入生产环境之前，必须先通过一套评测系统的检验，由它决定这个候选模型能否替换当前的生产模型——即当前实际承载产品流量的那个模型。这个产品要在 12 个任务领域上接受评测：编程、数学、摘要、多语言对话、工具使用、长文写作、指令遵循、多步推理、事实问答、多轮对话、智能体任务完成，以及代码审查。各领域现有的标注数据量并不相同，而这套系统必须在领域不断增加、产品自身行为逐周变化的情况下，持续给出可信的判断。

对每个领域而言，*黄金集*（golden set）是一份从训练数据中留出、只用于评测的固定提示词集合，其中每一条都带有足够判断一次回复所需的信息——一个参考答案、一份评分细则（rubric），或者一个可以用程序判定的结果。*自动评分器*（autorater）是一个独立于候选模型和生产模型之外的模型，专门用来给回复打分或做比较；它也被称为 LLM 评委（LLM-as-a-judge）。*并排评测*（side-by-side evaluation，SxS）把同一个提示词分别由两个模型作答后的两条回复一起交给一位评委——人类或自动评分器——由评委判断哪一条更好，或者是否打平。对一对模型、在一批提示词上跑一轮并排评测，会得到这一对模型的*胜率*（win rate）：一方在全部配对比较中获胜的比例，打平时双方各记半胜。*数据污染*（contamination）指的是黄金集里的提示词、或者它们的参考答案，逐字或近乎逐字地出现在了某个模型的训练数据里，这会让这个模型在黄金集上测出的表现，高于它在真正未见过的输入上的表现。

设计这套评测系统：一个黄金集和一个自动评分器如何变成某个领域的胜率，这 12 个领域各自的胜率又如何汇总成一个发布决定，以及这套设计如何在产品和它的各个领域不断演化的同时，让这个决定始终可信。

为下面的规模设计：

- 12 个任务领域。
- 每个领域有一个包含 2,000 条提示词的黄金集。
- 一次自动评分器调用的成本是一次候选模型调用的 1/10，耗时 3 秒。
- 人工评分的预算是每周 5,000 次评分。
- 发布决定必须在候选模型到达后 48 小时内做出。
- 目标：以 95% 置信度（双侧）、80% 检验效能（power），检测出某个领域真实的并排胜率是否偏离 50% 至少 2 个百分点。
- 有些领域除了提示词本身之外，没有任何额外的标注数据：没有参考答案，没有评分细则，除了评测系统自己产出的东西，没有任何可以拿来比较一条回复的依据。

范围内：离线评测（黄金集、自动评分器、人类与自动评分器参与的并排评测）、把 12 个领域的结果汇总成发布结论的统计决策规则、切片分析（slice analysis）、数据集／提示词／评委的版本管理、数据污染检测，以及候选模型上线之后仍在继续的在线评测（A/B 测试、用户反馈）。范围外：训练候选模型本身。

要产出：

1. 需求与规模估算：给定置信度和检验效能所需要的按领域样本量、它与各领域黄金集规模的对比及其含义；一次候选模型评测所耗费的计算量与实际耗时；每周人工评分预算的分配方式。
2. 数据模型——覆盖提示词、回复、评分、评委版本与结果——以及一次评测运行从候选模型到达到发布结论的 API 与工作流程。
3. 一张架构图，以及沿着它对一次评测运行的完整走查。
4. 深入话题：(a) 把自动评分器与人工评分对齐——一致性指标（如 Cohen's kappa）、并排比较里的位置偏差（position bias），以及如何通过交换回复顺序抵消它；(b) 把多个模型之间大量的两两比较聚合成一个统一排名——用 Bradley–Terry（或 Elo）模型，并给出自助法（bootstrap）置信区间；(c) 跨 12 个领域的关卡规则——由此产生的多重比较问题，以及非劣效性边界（non-inferiority margin）；(d) 一个没有额外标注数据的领域——合成提示词生成、基于评分细则的自动打分，以及让任务端到端运行、对照可核验结果的智能体评测（agentic evaluation）；(e) 数据污染——检测并防止黄金集泄漏进训练数据。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手设计之前值得先确认：“替换”（replace）具体是什么意思——下面假设它是指候选模型在每个领域都不劣于生产模型（容许一个小的边界），并且至少在一个领域显著更好，而不是要求在所有地方都严格更好（这个门槛哪怕对一个确实变强的模型，也可能因为偶然性在一两个领域没跨过去），也不是只要求跨领域的某种加权平均更好（这会把某个领域真实的退化，掩盖在其他领域的提升之下）——以及两条回复打平时该如何计入胜率——下面假设打平双方各记半胜，这样一个领域的胜率始终是 $[0,1]$ 里的一个比例，一次比较贡献的正是一次伯努利式的结果，这也是下面样本量推导所需要的前提。

### 需求与规模

**按领域的样本量。** 把一个领域里的并排比较看作一串独立的结果，每次比较为该领域的胜率贡献 $0$、$0.5$ 或 $1$，检验真实获胜概率是否偏离 $p_0=0.5$（与生产模型没有真实差异）达到目标的 2 个百分点，即 $p_1=0.52$（由对称性，同样的推导也覆盖 $p_1=0.48$）。在 $H_0$ 下，对二项分布做正态近似，得到 $\hat p \sim \mathcal N(p_0, p_0(1-p_0)/n)$，于是显著性水平为 $\alpha$ 的双侧检验在 $|\hat p-p_0|>z_{\alpha/2}\sqrt{p_0(1-p_0)/n}$ 时拒绝原假设。在 $H_1$ 下，$\hat p\sim\mathcal N(p_1,p_1(1-p_1)/n)$；要达到 $1-\beta$ 的检验效能，就需要拒绝域的边界在 $H_1$ 下量出来，落在比 $p_1$ 低 $z_\beta$ 个标准差的位置。让两个假设下算出的这同一条边界相等，解出 $n$：

$$n = \left(\frac{z_{\alpha/2}\sqrt{p_0(1-p_0)} + z_\beta\sqrt{p_1(1-p_1)}}{p_1-p_0}\right)^2.$$

取双侧 $\alpha=0.05$（$z_{\alpha/2}\approx1.960$）、检验效能 $0.80$（$z_\beta\approx0.842$），算出 $n\approx4{,}903$ 次比较，向上取整为 $4{,}904$（下面精确计算并核对）。

**黄金集与统计目标的对比。** 每个领域的黄金集有 2,000 条提示词，所以每条提示词跑一次比较，只能提供目标所需 4,904 次里的 2,000 次——40.8%。把同一个关系式反过来，不是解 $n$，而是解在恰好 2,000 个样本下能够分辨出的 $p_1$，得到黄金集自身能可靠分辨的最小可检测效应（minimum detectable effect）接近 3.1 个百分点，而不是 2 个。把黄金集完整跑一遍，并不能在设定的置信度和检验效能下分辨出真实的 2 个百分点的变化；下面 (c) 里搭建的关卡，被有意地按黄金集实际能够诚实分辨的程度来定标，把名义上的 2 个百分点目标当作一个需要靠增加独立比较——额外采样的生成结果，或者一个自动评分器不需要任何参考标签就能评的非黄金提示词池——去争取的数字，而不是假装跑一遍黄金集就已经达到了它。

**每个候选模型的计算量与耗时。** 一次评测运行需要为 $12\times2{,}000=24{,}000$ 条黄金集提示词中的每一条，生成一次全新的候选模型回复；与之配对的生产模型回复只在一个领域的黄金集被创建或刷新时生成一次，此后每一个候选模型都复用它，而不是每周重新生成。把自动评分器的位置偏差——它偏向某个位置上无论坐着哪条回复的倾向，推导见 (a)——用交换两种回复顺序的办法抵消掉，会让打分调用量翻倍到 $48{,}000$ 次，于是一次候选模型评测的成本是 $24{,}000 + 48{,}000\times0.1 = 28{,}800$ 个候选模型调用当量——只比单纯的生成多 20%，因为一次自动评分器调用只是生成调用价格的十分之一。按自动评分器给出的 3 秒调用延迟串行执行（假设生成调用的耗时量级与此相同，因为前提里没有给出别的信息），这 $72{,}000$ 次调用总共要花 $216{,}000$ 秒，60 小时——还没收到一条人工评分，就已经超过了 48 小时的 SLA，这正是这条流水线要跑在一个工作进程池上、而不是逐次调用的原因：要把它塞进 48 小时里的 3 小时之内，需要 $216{,}000/(3\times3{,}600)=20$ 个并行工作进程，把剩下的 45 小时留给人工评分环节、切片分析，以及发布决定本身。这份工作负载对延迟不敏感、追求高吞吐，形状更接近一个离线推理池，而不是一条实时服务路径，天然适合用产品自身服务机群在流量峰值之间的空闲或非高峰算力来承担，而不需要专门的硬件。

**人工评分预算。** 人工评分用来校准自动评分器、裁决它把握最不确定的那些判定；它们不能替代自动评分器的覆盖面，因为这一周全部 5,000 次的评分预算，仅仅覆盖一个领域自己 4,904 次比较的目标就已经捉襟见肘（$5{,}000/4{,}904\approx1.02$ 倍），何况是全部 12 个领域。每个领域固定预留 30 次评分（合计 360 次）作为底线，让每个领域的校准 kappa 值每周都保持更新；剩下的 4,640 次评分，用来裁决两种交换顺序下自动评分器的判定出现分歧的那些比较，按各领域实际产生分歧的多少分配、而不是平均分配到各领域——一个评分细则更主观的领域（比如长文写作）产生的分歧，会比一个大体靠机械检查（比如按单元测试判分的编程）的领域更多，也相应地会拿到预算中更大的一份，均分（$4{,}640/12\approx386.7$ 每个领域）只是每个领域无论自身分歧率高低都能保证拿到的底线。

### 数据模型与 API

**Prompt（提示词）** —— `prompt_id`、`domain`、`text`、`source`（`golden | synthetic | production_sample`）、`reference`（参考答案、评分细则，或者一个程序化的检查器；该领域有的话才有，否则为空）、`golden_set_version`、`created_at`、`contamination_status`（`clean | flagged | retired`）。

**Response（回复）** —— `response_id`、`prompt_id`、`model_role`（`candidate | production`）、`model_version`、`text`、`generation_params`、`generated_at`。一条生产模型的回复，每个 `golden_set_version` 只生成一次，供之后每一个候选模型复用；一条候选模型的回复则为它自己那次评测运行专门生成。

**Comparison（比较项）** —— 一条并排比较项：`comparison_id`、`run_id`、`domain`、`prompt_id`、`response_a_id`、`response_b_id`、`candidate_position`（`a | b`，候选模型这次占的是哪一侧）、`pair_key`（由同一条提示词按相反顺序展示出来的两条比较项共享，方便在汇总时把它们重新配对）。

**Rating（评分）** —— 对一条 `Comparison` 的一次判定：`rating_id`、`comparison_id`、`judge_type`（`autorater | human`）、`judge_version`、`verdict`（`a_win | tie | b_win`）、`confidence`、`rationale`、`rated_at`。

**JudgeVersion（评委版本）** —— `judge_version_id`、`domain`、`prompt_template_version`、`rubric_version`、`effective_from`、`rolling_kappa`（与人工评分的一致性，随校准评分的到来滚动更新）。

**EvaluationRun（评测运行）** —— `run_id`、`candidate_model`、`candidate_version`、`started_at`、`status`（`running | complete`）、`decision`（`release | block | pending`）、`decided_at`、`decided_by`（只在人工改判时才有值）。

**DomainResult（领域结果）** —— `run_id`、`domain`、`n_comparisons`、`win_rate`、`non_inferiority_p`、`superiority_p`、`holm_adjusted`、`verdict`（`non_inferior | regressed`）。

核心接口：

1. `POST /v1/eval-runs` —— `{candidate_model, candidate_version}` → `202 {run_id, status: "running"}`。为一个候选模型启动完整的 12 领域评测。
2. `GET /v1/eval-runs/{run_id}` —— `{status, domains: [{domain, n_comparisons, win_rate, verdict}], decision}`。整体状态与按领域汇总的结果。
3. `GET /v1/eval-runs/{run_id}/domains/{domain}/comparisons?filter=` —— 某个领域的逐条比较，可以按 `disagreement`（被路由去做人工裁决的那些）或 `calibration` 过滤。
4. `POST /v1/comparisons/{comparison_id}/ratings` —— `{judge_type, judge_version, verdict, confidence, rationale}`。自动评分器服务和人工评分工具都调用这一个接口，所以不管判定来自谁，都走同一条写入路径。
5. `POST /v1/eval-runs/{run_id}/override` —— `{decision, justification}`。发布工程师的人工改判；`justification` 是必填项，这次调用会被记入审计日志。
6. `GET /v1/judges/{domain}` —— `{judge_version, rolling_kappa, last_calibrated_at}`。当前评委及其校准状态，供下面追问里的漂移检测读取。

### 架构

```text
  候选模型
  每周到达
      |
      v
  +--------------------+          +------------------------+
  | 生成服务           |<---------| 黄金集存储             |
  | 候选模型回复；     |          | （提示词，按领域分类） |
  | 生产模型回复缓存， |          +------------------------+
  | 不重新生成         |            |
  +--------------------+            | 定期扫描
      |                           +-------------------------+
      v                           | 数据污染检测器          |
  +--------------------+          | （n-gram 重叠、金丝雀） |
  | 并排调度器         |          +-------------------------+
  | 把候选模型与缓存的 |
  | 生产模型回复配对， |
  | 两种顺序都跑，     |
  | 抽样送人工评分     |
  +--------------------+
           |                     |
           v                     v
  +----------------+      +------------+
  | 自动评分器服务 |      | 人工评分   |
  +----------------+      | 队列／工具 |
                          +------------+
           |                     |
           v                     v
  +----------+
  | 评分存储 |
  +----------+
        |
        v
  +------------------------------+
  | 聚合与统计引擎：             |
  | 胜率、kappa、Bradley-Terry、 |
  | Holm 校正的非劣效性关卡      |
  +------------------------------+
                  |
                  v
  +--------------+
  | 发布关卡决策 |
  +--------------+
           |                      |
       通过                         未通过
           v                      v
  +-----------------+      +------------+
  | 上线生产        |      | 拦截，反馈 |
  | ＋在线 A/B 挂钩 |      | 给团队     |
  +-----------------+      +------------+
```

候选模型一到达，就会创建一条 `EvaluationRun`。对 12 个领域中的每一个，生成服务都会为黄金集里的每条提示词生成一条回复——总共 24,000 次调用——同时并排调度器读取该领域缓存好的生产模型回复，而不是再去调用一次生产模型。调度器为每条提示词写入两行 `Comparison`，候选模型分别占据两个位置各一次，并把两者都发给自动评分器服务；一个 20 个工作进程的进程池（规模在上面需求与规模一节中给出）为每一条打分，并通过与人工工具相同的那个接口写入一行 `Rating`，这样聚合环节就完全不需要区分某条判定究竟来自哪一种评委。在跑完整的自动评分器这一遍的同时，调度器把一份固定的校准样本，以及所有两种交换顺序判定出现分歧的比较，都路由给人工评分队列；一条比较上的人工判定会替换掉自动评分器在这条比较里原来的判定，进入该比较的计票，而那份校准样本则单独用来更新对应领域的 `JudgeVersion.rolling_kappa`。等所有领域的比较都评完分之后，聚合引擎计算每个领域的胜率，跑 (c) 中的非劣效性检验与优越性检验，并把 Holm 校正一次性应用到全部 12 个领域上；发布关卡的决策只有在全部领域都通过非劣效性、且至少一个领域也通过优越性时才会记为 `release`，否则记为 `block`，随后通知团队，如果是发布，则顺势交给在线 A/B 挂钩。另有一个独立的、周期性运行的任务——不在任何一次候选模型评测的关键路径上——对黄金集存储跑 (e) 中的数据污染检测器，在被污染的提示词影响到未来的决定之前，把它们标记出来准备轮换。

### 深入话题

**(a) 把自动评分器与人工评分对齐。** 校准要回答两个不同的问题：自动评分器的判定是否与人类在同一批比较上给出的判定一致，以及自动评分器是否带有与质量无关的系统性偏差。

一致性用 Cohen's kappa 衡量，而不是简单的一致率，因为简单一致率会奖励一个经常默认选择多数类别的评委，而这种默认往往和它是否真的在追踪人类的判断毫无关系：在 `production_win`（生产模型胜）、`tie`（打平）、`candidate_win`（候选模型胜）这三种结果里，如果一个领域里真实的打平很常见，两个评委只要都偏向多选“打平”，就会单纯因为运气而经常一致。Cohen's kappa 用观测到的一致率 $p_o$，去对比两个边缘分布相同、却各自独立的评委仅凭运气会达到的一致率 $p_e$，来修正这一点：

$$\kappa = \frac{p_o - p_e}{1 - p_e}.$$

在下面核对代码用到的那份 200 条比较的校准混淆矩阵上，$p_o=0.80$，但 $p_e\approx0.34$，算出 $\kappa\approx0.70$——按惯例属于“实质性”（substantial）一致，还谈不上“几乎完全”（almost perfect）一致：这个水平足以信任自动评分器去做全覆盖打分，但还不足以彻底停止抽样人工判定，这正是为什么人工预算里的校准样本每周都要继续跑，而不是只跑一次。

位置偏差是另一种不同的问题：判定会因为一条回复被展示在配对中的哪一侧而发生偏移，与哪条回复真的更好无关。一个评委——不论是人类还是自动评分器——如果偏向无论哪条排在前面的回复，那么当候选模型总是排在前面时，测出来的候选模型胜率就会高于它真实的水平，而当它总是排在后面时，测出来的胜率又会偏低。要检测它，只需要把同一对回复按两种顺序各跑一遍，看判定会不会在两条回复本身毫无变化的情况下发生翻转。抵消它则是同一个道理的直接推论：如果把候选模型排在前面会给它的表观分数加上一个偏差 $b$，把它排在后面则会减去同样的 $b$（这时生产模型占据了偏差偏爱的那个位置），那么把同一条提示词在两种顺序下的分数取平均，就能恰好抵消 $b$，只留下两次判定各自独立的噪声：

$$\mathbb E\left[\frac{s_{\text{first}}+s_{\text{second}}}{2}\right] = \frac{(\text{true}+b) + (\text{true}-b)}{2} = \text{true}.$$

只用一种顺序，或者每条比较随机挑一种顺序，只能在许多条提示词的整体层面上抵消 $b$，在任何单独一条比较上都做不到，所以它的期望值和跑两种顺序的设计相同，但在单条比较这一层面上噪声更大；对每一条比较都跑两种顺序——这里完全负担得起，因为一次自动评分器调用只是一次候选模型调用价格的十分之一——正是上面计算量预算里已经假设了的做法。下面的核对代码模拟了一个带有 8 个百分点位置偏差的评委，去评一个真实胜率为 55% 的候选模型：只用一种、候选模型总在前面的顺序，测出的胜率接近 63%，而交换顺序后的估计值则回落到接近 55%。

**(b) 把大量两两比较聚合成一个排名。** 一周之内的评测恰好只比较两个模型，但同一套黄金集基础设施，在需要同时比较多个候选模型时也照常运行——比如几个互相竞争的微调版本，或者一份由历次候选模型组成的滚动排行榜——而多于两个模型之间的并排判定，并不会自动汇成一个排名：模型 A 可能赢 B，B 可能赢 C，而 A 和 C 之间却可能从未直接比过。Bradley–Terry 模型把一组两两胜负计数变成一个排名的办法，是给每个模型 $i$ 一个潜在的强度 $\pi_i>0$（等价地，$\theta_i=\log\pi_i$），并建模

$$P(i \text{ 胜过 } j) = \frac{\pi_i}{\pi_i+\pi_j} = \frac{1}{1+e^{-(\theta_i-\theta_j)}},$$

即强度差的一个 logistic 函数，打平按双方各记半胜处理，和本页别处的胜率算法完全一致。对全部观测到的比较结果求对数似然的最大值没有闭式解，但把对数似然对每个 $\theta_i$ 求导并令其为零，就能得到一个简单的不动点刻画——Zermelo/Hunter 的 MM（minorisation–maximisation）更新——它从任意正的初始值出发都会单调收敛：

$$\pi_i \leftarrow \frac{W_i}{\displaystyle\sum_{j\neq i} \dfrac{n_{ij}}{\pi_i+\pi_j}},$$

其中 $W_i$ 是模型 $i$ 对所有对手的总胜场数，$n_{ij}$ 是 $i$ 与 $j$ 之间被比较过的次数。对每个模型都做这个更新并重新归一化——这些强度只能确定到一个公共的尺度，因为把每个 $\pi_i$ 都乘上同一个常数，不会改变任何一次两两比较的概率——就能收敛到极大似然强度。下面的核对代码用这个方法拟合了 4 个模拟出来的模型，精确地恢复出了它们真实的强弱顺序。

单单一个点估计，掩盖了这个排名里有多少成分只是有限次比较带来的噪声。自助法（bootstrap）置信区间的做法是：对每一对模型各自独立地、有放回地重采样这一对的比较结果，在每个重采样样本上重新拟合 Bradley–Terry，再看某一对模型的胜率估计值在各次重采样之间的分布——不需要对抽样分布的形状做任何假设，只需要这次重采样能重现产生原始观测数据的那种随机性。下面的核对代码为某一对模型的胜率自助法算出一个 95% 区间，并确认它覆盖了模拟设定的真实值。

**(c) 跨 12 个领域的关卡规则。** 对全部 12 个领域各自独立地按 $\alpha=0.05$ 做检验，并不能把整个发布流程的总体假阳性率维持在 5%。如果一个候选模型在每一个领域里都与生产模型没有真实差异，那么 12 次独立检验里至少有一次仍然拒绝原假设的概率是

$$1-(1-\alpha)^{12} \approx 0.46$$

——这不是 5% 的判错概率，而是接近抛硬币（下面会核对这个数字）。标准的修正办法，是给每个领域用一个更严格的显著性水平。Bonferroni 校正最简单：每个领域都按 $\alpha/12$ 检验，因为对 12 个各自低于 $\alpha/12$ 的事件取并集上界，总和不会超过 $\alpha$，所以能控制住族错误率（family-wise error rate）。Holm 逐步法（step-down procedure）能控制住同一个量，同时拒绝得至少一样多：把 12 个 p 值从小到大排序，把第 $k$ 小的那个和 $\alpha/(12-k+1)$ 比较，在第一次未能通过的地方停下——在它之前的全部拒绝，从它开始（含它）的全部不拒绝。排在第一位的比较，对照的正好是 $\alpha/12$，和 Bonferroni 完全一样；它之后的每一次比较，对照的阈值都更宽松，所以 Holm 只可能拒绝一个 Bonferroni 会漏掉的假设，反过来永远不会（下面用一个小例子核对：Holm 拒绝了 3 个假设，Bonferroni 只拒绝了 1 个）。

每个领域的检验真正该问的，不是“候选模型是否更好”，因为开头关于“替换”该如何理解的确认，已经把这个过严的标准排除掉了；还因为一个按照在恰好 2 个百分点的差异处达到 80% 检验效能来定样本量的检验，按其构造，对一个真实差异恰好卡在这个点上的领域，五次里就有一次通不过——这不是一个生产关卡应该拿来卡掉一个真正打平的领域的门槛。每个领域改为跑一个单侧的非劣效性检验：$H_0$ 是候选模型真实胜率落在 $0.5-\delta$ 或更低（真实退化至少 $\delta$），$H_1$ 是它高于这个值。这里把 $\delta$ 定为 3 个百分点——比名义上的 2 个百分点目标要宽，也接近上面算出的黄金集自身 3.1 个百分点的分辨极限，宽到足以让一个真正与生产模型打平的领域，在需要同时跨过全部 12 道 Holm 校正阈值时仍有把握通过，而名义上的 2 个百分点则做不到这一点（下面会核对）。候选模型只有在全部 12 个领域都在 Holm 校正后的水平下通过非劣效性检验时，才算通过按领域的关卡；只有在此基础上，还有至少一个领域在同样的校正下显著优越于 $0.5$，才谈得上发布，而不只是“差不多”——这正是“替换”这个词该如何理解的另外一半：任何地方都不会差过 $\delta$，并且在某处确实更好。

**(d) 一个没有额外标注数据的领域。** 智能体任务完成——端到端跑完一个多步骤、需要调用工具的任务——是这份清单里最可能完全没有额外标注数据的领域：没有参考答案可以比对，有时连一套建立好的提示词都没有，因为这个领域本身比其他 11 个领域已有的黄金集基础设施还要新。从零开始搭建它分三个阶段。

合成提示词生成，负责产出一切后续工作都要依赖的提示词池：一个生成模型，在被告知任务族（task family）的形状之后（“按这些约束订一份行程”、“找到并修复一个失败的测试”），会按目的地、约束条件、起始代码仓库状态等各种参数组合，产出候选提示词，随后去掉高度相似的重复项，并经过一轮审查——人工或者一个更强的检查模型——之后，才会被信任到足以影响一次发布决定。这样做能便宜地产出数量，但产出不了正确性：一条生成出来的提示词可能含糊不清、根本无法完成，或者压根不需要用到任何工具就能回答，审查这一步存在的意义正是要挡住这些情况。

基于评分细则的自动打分，用一份对着单条回复评分的检查清单，取代了本页其他地方用的并排比较，因为这里没有参考答案可以拿来比对，而且对很多智能体类提示词来说，也不存在唯一正确的执行记录（transcript）——只有一条正确记录必须具备的若干性质。一条提示词的评分细则，是一份简短的二元判断标准清单（“订票前调用过搜索工具”、“从不编造航班号”、“总花费没有超出给定预算”）；自动评分器独立地对着执行记录逐条判断每一项标准，这条提示词的得分就是满足的标准所占的比例。这个做法不需要任何标准答案，只需要一份评分细则——比撰写一份完整的参考解答要便宜，但并不是没有成本——而且它继承了 (a) 里普通自动评分器自身需要校准这件事：一个评分细则评分器，正如一个并排自动评分器要拿人工的并排判定去衡量一样，也要拿人工在同一份评分细则上的判断去衡量。

在任务允许的地方，智能体评测更进一步：不再对执行记录本身评分，而是在任务所需环境的一份沙箱化拷贝里，把候选模型的工具调用完整跑到底，再用一个程序化的裁判去检查最终状态——它要修的那套测试是不是通过了、那份行程单的总花费是不是压在了预算之内、日历上是不是在要求的时间留出了那场会议。这样一来，对于有可核验结果的那部分任务，就完全绕开了撰写评分细则，也绕开了任何需要主观判断的打分环节，代价是只能用在“正确”有可执行定义的领域——对长文写作这样的领域，它无能为力，那里仍然只能退回基于评分细则的自动打分。用这种方式搭建起来的领域，带有一种精心整理出来的黄金集所不具备的数据污染风险：提示词生成器和审查它的模型本身也是模型，如果其中之一、或者与它们关系相近的某个模型，出现在了某个候选模型的训练数据里，那么生成出来的提示词——以及按它们确切措辞写成的评分细则或裁判逻辑——就可能和它们一起泄漏进训练数据；(e) 会讲如何检测这一点。

**(e) 数据污染。** 数据污染是黄金集和训练数据本身的属性，不是某一个候选模型的属性，所以检测它是一项针对黄金集存储的常态检查，而不是某一次评测运行里的一个步骤。最便宜的检测手段是精确重叠：把黄金文本（一条提示词，以及它的参考答案，如果有的话）和任何被怀疑的文档——一个训练数据分片，训练流水线所依赖的某次网页抓取——都切分成有重叠的词级 $n$-gram，再看黄金文本的 $n$-gram 里有多少比例，同时也出现在这份文档里。比例很高，尤其是来自一段很长的连续重合，正是黄金文本几乎逐字坐在那份文档里面的信号；已发表的数据污染研究通常取 $n$ 在 8 到 13 个词左右，长到足以让一次匹配不只是两份文档偶然共享了一个普通短语。下面的核对代码在更短的玩具字符串上用了一个更小的 $n$ 来实现这个方法，并确认它既能抓住一次逐字复制，同样也能说明问题地——完全漏掉同一句话的一次改写：一个 $n$-gram 检测器证明的始终只是文本层面的重叠，从来不是语义层面的泄漏，所以黄金提示词和训练文档之间只要经过一次改写，就能让它完全失效；补上这个缺口需要对同样的候选文档做一遍嵌入相似度扫描，单次比较的计算成本高得多，只在精确匹配筛查本身无法排除污染时才运行。

第二种廉价的技术是金丝雀字符串（canary）：生成一个独一无二、绝不做他用的字符串，插进黄金集某条提示词的一份拷贝里，而这份拷贝本身绝不发布，也不会被记录到任何训练流水线有可能收集到的地方。如果这个金丝雀字符串后来从某个候选模型嘴里冒了出来——逐字出现在一次续写里，出现在一个从未包含过它的提示词下面——那就说明黄金集是通过精确匹配扫描本来覆盖不到的某个渠道泄漏出去的，不需要拿到训练语料本身就能发现这一点。

检测只补上了这个闭环的一半，另一半是从一开始就不需要检测到泄漏。黄金集的提示词，要从任何有可能汇入未来训练语料的地方排除出去——公开发布的内容、经产品自身数据管线回传的评测流量日志——并且要做版本管理，这样一条具体提示词的来龙去脉、以及它曾经展示给过的每一个候选模型，都可以追溯回来。由于一条提示词存活得越久，暴露的风险就只会越积越多，每个领域的黄金集都按固定周期淘汰并替换掉一部分，给任何一条提示词的数据污染风险能累积多久设一个上限；另外还留出一部分，连内部发布的任何材料里都绝不收录，这样它的确切内容，也就不会通过一份引用了示例提示词的报告泄漏出去。

### 追问

- **安全性单独评测，有它自己的关卡。** 一次越狱（jailbreak）或者一次有害内容方面的退化，并不是 2 个百分点的胜率变化天生就能抓住的东西——一个候选模型完全可能在 51% 的常规质量比较里获胜，却在一个本该拒绝的请求上失手——所以安全性用它自己的黄金集、一个针对危害评分细则而不是质量比较专门调校的自动评分器，以及一条固定、不带非劣效性边界的及格线来评测：安全性基准上的任何退化，都会拦下这次发布，不管另外 12 个质量领域给出了什么结论。
- **长上下文与多模态评测改变的是成本，不是统计方法。** 10 万 token 的上下文，或者一路图像输入，会把生成与打分的延迟、以及自动评分器自身的上下文预算，都推到远超上面用到的 3 秒这个数字，这会把计算量与耗时的估算往上推，但样本量的推导、校准方法，以及关卡规则，都不受影响——这三者都是作用在一个胜率上，与底下的提示词本身有多长毫无关系。
- **成本控制：给评委分级级联。** 上面算出的 20% 自动评分器开销，假设的是每条比较都要在两种顺序下各跑一次完整的自动评分器打分；一种更省钱的策略，是先跑一个又快又便宜的自动评分器初筛，只把出现分歧的部分——两种交换顺序之间的分歧，或者自动评分器对同一对回复重复判定之间的分歧——升级给一个更慢、更强的自动评分器，或者升级给人工队列，用一小部分全覆盖精度上的损失，换来在那些本来就不难判断、用不着升级的比较上，成比例地削减自动评分器预算。
- **检测自动评分器漂移。** 自动评分器本身也是一个模型，更新它——换一个新版本，换一份新的评委提示词——会在候选模型本身毫无变化的情况下，改变同一条回复能拿到的分数，如果不加处理，这会伪装成全部领域胜率同时发生的一次偏移。给每个领域的 `JudgeVersion` 定住版本，并在一个新的评委版本接替旧版本进入生产之前，重新对它跑一遍校准样本，可以在这种变化影响到发布决定之前就把它抓出来——用的正是 (a) 里本来就在持续运行的那套机制，只是这次作用在评委的变化上，而不是候选模型的变化上。
- **线上与线下指标的相关性。** 离线关卡只有在它能预测上线后 A/B 测试会看到什么的前提下才有意义；逐次发布地跟踪某个领域离线并排胜率与它线上指标变化（任务成功率、点赞率、留存）之间的相关性，正是能够发现关卡正在悄悄偏离用户真实体验的办法——比如黄金集不再匹配真实生产流量的分布，或者自动评分器学会了奖励一些用户其实并不喜欢的东西。

<details>
<summary>估算核对（可运行）</summary>

```python
import math
import re
from itertools import combinations

import numpy as np
from scipy import stats

# ---- per-domain sample size for the stated power ----
def required_n(p1: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> float:
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(p1 * (1 - p1))) ** 2 / (p1 - p0) ** 2

n_exact = required_n(p1=0.52)
assert round(n_exact, 1) == 4903.2
n_domain = math.ceil(n_exact)
assert n_domain == 4904

domains = 12
golden_set_size = 2_000
total_prompts = domains * golden_set_size
assert total_prompts == 24_000

coverage = golden_set_size / n_domain
assert round(coverage, 3) == 0.408

def mde_at_n(n_target: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> float:
    """Invert required_n by bisection: the p1 whose required sample size is exactly n_target."""
    lo, hi = p0 + 1e-6, p0 + 0.3
    for _ in range(100):
        mid = (lo + hi) / 2
        if required_n(mid, p0, alpha, power) < n_target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

mde_pp = (mde_at_n(golden_set_size) - 0.5) * 100
assert round(mde_pp, 1) == 3.1

# ---- compute and time per candidate ----
gen_calls = total_prompts                         # one candidate response per golden prompt
autorater_calls = total_prompts * 2                # both orders, to cancel position bias -- see (a)
assert gen_calls == 24_000 and autorater_calls == 48_000

autorater_cost_ratio = 0.1                         # of one candidate-model call
compute_units = gen_calls * 1 + autorater_calls * autorater_cost_ratio
assert compute_units == 28_800
assert round(compute_units / gen_calls, 2) == 1.20

call_latency_s = 3                                 # given for an autorater call; assumed same order for generation
serial_seconds = (gen_calls + autorater_calls) * call_latency_s
assert serial_seconds == 216_000
assert serial_seconds / 3600 == 60.0

sla_hours, target_pipeline_hours = 48, 3
workers_needed = math.ceil(serial_seconds / (target_pipeline_hours * 3600))
assert workers_needed == 20
assert sla_hours - target_pipeline_hours == 45

# ---- human rating budget ----
human_budget = 5_000
calibration_per_domain = 30
calibration_total = calibration_per_domain * domains
assert calibration_total == 360
adjudication_budget = human_budget - calibration_total
assert adjudication_budget == 4_640
adjudication_per_domain = adjudication_budget / domains
assert round(adjudication_per_domain, 1) == 386.7
assert round(human_budget / n_domain, 2) == 1.02        # the whole week's human budget barely covers one domain

print("all requirements-and-scale numbers check out")

# ---- (a) Cohen's kappa and position-bias cancellation ----
def cohens_kappa(confusion: list[list[int]]) -> float:
    conf = np.array(confusion, dtype=float)
    total = conf.sum()
    p_o = np.trace(conf) / total
    row_marg, col_marg = conf.sum(axis=1) / total, conf.sum(axis=0) / total
    p_e = float(np.dot(row_marg, col_marg))
    return (p_o - p_e) / (1 - p_e)

# a calibration confusion matrix: rows = human verdict, columns = autorater verdict, over
# {production_win, tie, candidate_win}, 200 calibration comparisons
confusion = [[70, 8, 2], [10, 40, 10], [3, 7, 50]]
assert sum(sum(row) for row in confusion) == 200
kappa = cohens_kappa(confusion)
assert round(kappa, 3) == 0.696

def simulate_position_bias(seed: int, n_prompts: int, true_p: float, bias: float):
    """true_p is the candidate's true win probability under an unbiased judge; bias is the
    autorater's own boost to whichever response it sees in the first position. 'naive' always
    shows the candidate first; 'swapped' runs both orders and averages the two verdicts."""
    rng = np.random.default_rng(seed)
    p_first, p_second = min(true_p + bias, 1.0), max(true_p - bias, 0.0)
    naive = (rng.random(n_prompts) < p_first).mean()
    score_first = (rng.random(n_prompts) < p_first).astype(float)
    score_second = (rng.random(n_prompts) < p_second).astype(float)
    swapped = ((score_first + score_second) / 2).mean()
    return naive, swapped

naive_rate, swapped_rate = simulate_position_bias(seed=7, n_prompts=3_000, true_p=0.55, bias=0.08)
assert abs(naive_rate - 0.63) < 0.02 and abs(naive_rate - 0.55) > 0.05     # naive: biased by ~the position boost
assert abs(swapped_rate - 0.55) < 0.02                                    # swap-and-average: bias cancels

print("kappa and position-bias checks confirm the calibration and de-biasing claims")

# ---- (b) Bradley-Terry ranking (MM fit) and a bootstrap confidence interval ----
def simulate_pairwise(theta_true: np.ndarray, n_per_pair: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = len(theta_true)
    wins = np.zeros((m, m))
    for i, j in combinations(range(m), 2):
        p_i_beats_j = 1.0 / (1.0 + math.exp(-(theta_true[i] - theta_true[j])))
        i_wins = int((rng.random(n_per_pair) < p_i_beats_j).sum())
        wins[i, j], wins[j, i] = i_wins, n_per_pair - i_wins
    return wins

def fit_bradley_terry(wins: np.ndarray, n_iter: int = 200, eps: float = 1e-9) -> np.ndarray:
    """Zermelo/Hunter MM fixed point for the Bradley-Terry log-likelihood: pi_i is updated to
    its total wins divided by the sum, over opponents j, of n_ij / (pi_i + pi_j); iterating this
    update converges to the maximum-likelihood strengths, unique up to a common scale factor."""
    m = wins.shape[0]
    n_ij = wins + wins.T
    total_wins = wins.sum(axis=1)
    pi = np.ones(m)
    for _ in range(n_iter):
        denom = np.array([sum(n_ij[i, j] / (pi[i] + pi[j]) for j in range(m) if j != i) for i in range(m)])
        pi = total_wins / np.maximum(denom, eps)
        pi = pi / pi[0]                          # anchor model 0's strength at 1: BT strengths need a scale
    return pi

theta_true = np.array([0.0, 0.7, -0.5, 1.5])       # log-strengths of 4 models, e.g. 4 candidates in one week
wins = simulate_pairwise(theta_true, n_per_pair=200, seed=3)
theta_fit = np.log(fit_bradley_terry(wins))
assert list(np.argsort(theta_true)) == list(np.argsort(theta_fit))        # recovers the true ranking

def bootstrap_pair_ci(wins: np.ndarray, pair: tuple[int, int], n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    m = wins.shape[0]
    n_ij = wins + wins.T
    i, j = pair
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        boot_wins = wins.copy()
        for a, c in combinations(range(m), 2):                # paired bootstrap: resample within each pair
            n_ac = int(n_ij[a, c])
            resampled = rng.binomial(n_ac, wins[a, c] / n_ac)
            boot_wins[a, c], boot_wins[c, a] = resampled, n_ac - resampled
        pi_boot = fit_bradley_terry(boot_wins, n_iter=150)
        estimates[b] = pi_boot[i] / (pi_boot[i] + pi_boot[j])
    return np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])

true_p_1_beats_0 = 1.0 / (1.0 + math.exp(-(theta_true[1] - theta_true[0])))
ci_lo, ci_hi = bootstrap_pair_ci(wins, pair=(1, 0), n_boot=300, seed=11)
assert ci_lo < true_p_1_beats_0 < ci_hi
assert 0.02 < (ci_hi - ci_lo) < 0.25

print("Bradley-Terry MM fit recovers the true ranking; the bootstrap CI covers the true win probability")

# ---- (c) multiple comparisons across 12 domains and a non-inferiority gate ----
def holm_reject(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])          # ascending p-value order
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break                                              # step-down: stop at the first non-rejection
    return reject

fwer_unadjusted = 1 - (1 - 0.05) ** domains
assert round(fwer_unadjusted, 2) == 0.46

demo_p = [0.005, 0.009, 0.011, 0.02, 0.03, 0.5]
bonferroni_rejections = sum(p <= 0.05 / len(demo_p) for p in demo_p)
holm_rejections = sum(holm_reject(demo_p, 0.05))
assert bonferroni_rejections == 1 and holm_rejections == 3       # Holm is uniformly more powerful here

def one_sided_p(p_hat: float, p0: float, n: int) -> tuple[float, float]:
    z = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)
    return 1 - stats.norm.cdf(z), z

def evaluate_release(true_win_rates: list[float], seed: int, margin: float = 0.03, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    p_hats = rng.binomial(n_domain, true_win_rates) / n_domain
    ni_p = [one_sided_p(p, 0.5 - margin, n_domain)[0] for p in p_hats]     # H0: worse than production by > margin
    sup_p = [one_sided_p(p, 0.5, n_domain)[0] for p in p_hats]            # H0: no better than production
    non_inferior = holm_reject(ni_p, alpha)
    superior = holm_reject(sup_p, alpha)
    return all(non_inferior) and any(superior), non_inferior, superior

# scenario A: eleven domains genuinely tied with production, one genuinely 6 points better -> release
gate_a, ni_a, sup_a = evaluate_release([0.50] * 11 + [0.56], seed=101)
assert gate_a is True and all(ni_a) and sup_a[-1] and sum(sup_a) == 1

# scenario B: the same eleven tied domains, but one domain has genuinely regressed -> block
gate_b, ni_b, sup_b = evaluate_release([0.50] * 11 + [0.40], seed=202)
assert gate_b is False and ni_b[-1] is False and sum(ni_b) == 11

print("the multiple-comparisons gate releases a genuine improvement and blocks a genuine regression")

# ---- (e) n-gram overlap contamination detector ----
def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

def contamination_overlap(golden: str, document: str, n: int = 5) -> float:
    """Fraction of golden's n-grams that also occur in document -- 1.0 means golden's text sits
    inside document somewhere, word-for-word, in an unbroken run of at least n words."""
    g = word_ngrams(golden, n)
    return len(g & word_ngrams(document, n)) / len(g) if g else 0.0

golden_prompt = "the quick brown fox jumps over the lazy dog near the river bank"
contaminated_doc = "some preamble text before it. " + golden_prompt + ". and some text after."
unrelated_doc = "interest rates rose sharply after the central bank meeting concluded yesterday afternoon"
paraphrased_doc = "a fast auburn fox leaps above a sleepy dog beside the riverbank"

assert contamination_overlap(golden_prompt, contaminated_doc) == 1.0
assert contamination_overlap(golden_prompt, unrelated_doc) == 0.0
assert contamination_overlap(golden_prompt, paraphrased_doc) == 0.0      # exact n-grams miss paraphrase entirely

print("n-gram contamination detector flags the verbatim copy and misses the paraphrase")
print("all checks passed")
```

</details>

</details>
