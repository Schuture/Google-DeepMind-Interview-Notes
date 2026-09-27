# 机器学习设计：预测数据中心哪些机器需要更换

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 机器学习系统设计 | ★★☆☆☆ | 中等 | MLE · RE · SWE | predictive-maintenance, label-construction, censoring, class-imbalance, categorical-embeddings, survival-analysis, precision-at-k, feedback-loops | 45–60 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

设计一个系统，提前预测大型数据中心机队中哪些机器很可能需要更换硬件，让运维团队能在它们在生产环境中发生故障之前完成维护。*机器*（machine）是机队里的一台物理服务器；它的*机器型号*（machine type）是它所基于的硬件型号（CPU 代次、主板、机箱）——同时在役的型号有几百种，新型号随采购目录的更新周期性地引入。*维修工单*（repair ticket）记录了技术人员在一台机器上更换了一个或多个硬件部件，以及换了哪些部件、何时更换。*下线记录*（decommission record）记录了一台机器何时以及为何离开机队（一次计划内的更新换代、一次搬迁，或者一次严重到需要直接报废整机的故障，以及其他原因）。

为下面的规模设计：

- $1{,}000{,}000$ 台机器，分布在 $20$ 个数据中心，来自约 $300$ 种机器型号，新型号大约每季度引入一批。
- 每台机器每分钟上报约 $50$ 项遥测指标：部件温度、风扇转速、已纠正的内存错误计数、磁盘健康计数器、累计重启次数、CPU 降频事件，以及功耗。
- 维修工单记录了更换了哪个部件、何时更换；下线记录记录了一台机器何时以及为何离开机队。
- 在任意给定的 $30$ 天窗口内，约有 $0.5\%$ 的机器需要更换硬件。
- 运维团队每周最多能腾出并预先维护 $1{,}000$ 台机器。
- 一次生产环境中的计划外故障，代价约为一次预先更换代价的 $20$ 倍。

目标：每周产出一份排序列表，列出接下来 $30$ 天内最可能需要更换硬件的机器，让运维把固定的每周产能用在最要紧的地方。

范围内：整个预测系统，从原始日志到每周的排序名单，以及这份名单所制造的反馈循环。范围外：遥测采集代理本身，以及维修的物流细节（备件库存、技术人员排班）。

要产出：

1. 设计前要问清楚的问题，以及一个精确的标签定义：什么算“需要更换”，覆盖多长的时间范围。
2. 如何从上述日志构造训练集：快照、特征窗口、标签窗口、信息泄漏、删失，以及相关联的行。
3. 特征与特征工程，以及特征变换在线性模型、树集成模型与神经网络之间的差异。
4. 机器型号及其他高基数类别属性的嵌入，包括一个训练之后才引入、自身没有任何样本的型号。
5. 三个候选模型及各自的损失函数。
6. 类别不均衡如何处理。
7. 评估：用哪些指标、每个指标衡量的是什么、为什么它是该汇报的那个数字，以及回测设计。
8. 部署：预测结果如何被使用、如何监控，以及模型自身行为所制造的反馈循环。

面试官可能穿插提出的问题，在设计涉及到相应部分时提出：

- 为什么这个标签是不均衡的？选定的模型与评估指标又如何应对这一点，而不是被它误导？
- 如何为一个零训练样本的机器型号生成一个可用的嵌入？
- 每个评估指标究竟衡量的是什么？为什么该向运维团队汇报的是它，而不是另一个替代指标？

## 参考解答

<details>
<summary>展开参考解答</summary>

动手设计之前值得先确认：作为*计划内*机队更新换代的一部分而发生的硬件更换，是否应该计入标签（这里假设：不算——这个系统的目标是预测计划外的、由故障驱动的更换，一台已经处在其计划更新换代窗口内的机器，会同时被排除在正例和负例池之外）；以及一个本季度引入的机器型号，是否必须从第一天起就能被打分（这里假设需要，这也是下文“特征”一节选择从型号自身的描述性属性构建嵌入、而不是等待它本来就不会及时具备的数据的原因）。

### 需求与规模

**原始遥测数据量。** $1{,}000{,}000$ 台机器，每台每分钟上报 $50$ 项指标，一天 $1{,}440$ 次：

$$1{,}000{,}000 \times 50 \times 1{,}440 = 7.2\times10^{10} \text{ 条读数/天。}$$

每条读数按 $8$ 字节浮点数存储，就是 $7.2\times10^{10}\times8=5.76\times10^{11}$ 字节，即 $576$ GB/天——大到这个设计只保留一段滚动窗口的原始读数（几周，供审计与重新聚合），模型实际消费的一切都先聚合成按小时或按天的值，从不直接在这条每分钟的流上训练。

**训练行数。** 每台机器每周一条快照，一年就是 $1{,}000{,}000\times52=52{,}000{,}000$ 条快照行——比它所基于的原始遥测数据小好几个数量级，是一个普通表格数据流水线能轻松应付的规模。

**每周预期正例数与每周产能的对比。** 按稳定的 $0.5\%$ 的 $30$ 天更换率计算，某一周的快照预计会有
$1{,}000{,}000\times0.005=5{,}000$ 个正标签——也就是接下来 $30$ 天内需要更换的机器。运维每周只能维护其中
$1{,}000$ 台。既然一份排序列表无论排得多好，最多也只能把 $1{,}000$ 台机器纳入这份产能，recall@$1{,}000$——
这一周 $5{,}000$ 个真正例里被真正捕获的比例——就被 $1{,}000/5{,}000=20\%$ 这个上限锁死；一个达到这个上限
的模型，说明它已经把整整一周的产能都填满了真正例（precision@$1{,}000=100\%$），在这个产能之下，任何排序都
不可能比这个上限做得更好。在截断点和这一周的正例数都固定的情况下，recall@$1{,}000$ 就等于
precision@$1{,}000\times1{,}000/5{,}000$，所以两者携带的是关于一周排序的同一份信息，汇报的是
precision@$1{,}000$（见下文“评估”一节）。这个上限并不是这个设计放任发生的故障比例：一个 $30$ 天窗口大约横
跨 $30/7\approx4.3$ 次每周快照，所以相邻快照的正例大部分是重合的，新需要更换的机器每周只有大约
$5{,}000\times7/30\approx1{,}170$ 台——只要排序能在每台机器发生故障之前的某一周把它标记出来，每周 $1{,}000$
次预先维护就能覆盖这股流量的大部分。

### 标签与训练数据

**标签。** 对于在快照时间 $t$ 被观察到的一台机器，如果有一张维修工单记录了在 $(t, t+30\text{ 天}]$ 内更换了
一个**硬件**部件，标签为 $1$，否则为 $0$；一张只记录纯软件修复（刷固件、重新插好一根线缆而未更换任何部件）
的工单不计入。**计划内更新换代**——作为机队正常硬件更新换代周期的一部分而安排的更换，与观察到的任何退化都
无关——同样被排除：不仅从正例标签中排除，对处在其计划更新换代窗口内的机器，也把它从负例池中排除，因为无论
“是否发生了硬件更换”这个问题的答案是什么，都回答不了这个系统真正要回答的问题——这台机器自身的故障风
险。计划内更新换代由普通的更新换代流程处理，不归这份排序管。

**快照。** 每台机器每周一行，在固定的每周快照时间 $t$；这一行所用的特征，只用 $t$ 或之前可得的数据算出——绝
不使用之后才记录下来的任何东西，包括标签本身正要预测的那张工单。这条规则有三种容易在不经意间被打破的方式：

- **故障本身触发的诊断信息。** 一张维修工单常常带有技术人员在排查故障过程中记录下来的诊断标志或错误代码；
  这个标志是标签所标记的同一个事件的下游产物，而不是它的前兆，把它当作特征使用，等于让模型在自己的输入里
  直接看到答案。
- **用未来工单算出的“距上次工单的天数”特征。** 这个特征必须只统计严格早于 $t$ 的工单；如果只在一台机
  器的全部历史上算一次、再把同一个值套用到它的每一条快照上，就会在不知不觉间把 $t$ 之后才发生的工单也算了
  进去，包括正要预测的那一张。
- **在整个时间段上算出的机队级统计量。** 类似“这个机器型号本季度的平均错误率”这样的特征，只有在
  “本季度”指的是截至 $t$ 的数据时才是安全的；如果只在整个数据拉取范围（对大多数快照而言，这个范围会
  延伸到 $t$ 之后）上算一次，就会把模型还没走到的那些周的结果，泄漏进同一机器型号更早的每一行里。

**删失（censoring）。** 一台机器可能因为与所要预测的故障无关的原因（一次搬迁、一次无关的计划内更新换代）在
$t+30$ 天之前就下线了，数据拉取范围本身也可能在 $t+30$ 天之前就结束；不论是哪种情况，这条快照的真实标签都
是未知的，而不是负例——如果这台机器留在机队里，它本来可能需要更换，也可能不需要。把一条尚未有定论的快照悄
悄当成负例，会低估真实的更换率，因为被错贴标签的这些行，恰恰不成比例地集中在观察生涯即将结束的那些机器上。
反过来把这些快照从一个普通的分类训练集里丢弃，则会把比率往另一个方向带偏：一台在窗口早期就发生故障的机器，
即便它本来会在窗口后段离开数据，这条快照也已被确定为正例；而一台一直活到离开数据的机器，这条快照却被丢弃，
于是保留下来的行高估了故障的比例（下文的估算核对会在一个合成机队上，对照真实比率量出这两个偏差）。生存
分析建模方式（见下文“模型与损失函数”）能同时避开这两种偏差，正确地使用一台被删失的机器——只贡献它真正告
诉我们的信息：它活着离开了数据，仅此而已，不多贡献一分。

**切分。** 同一台机器相隔一周的两条快照，看到的遥测数据几乎相同，如果确实有一次故障将至，它们距离同一张工
单通常也只差寥寥几周，所以它们的标签高度相关。按行随机切分，会让其中一条落进训练集、另一条落进测试
集，一个足够有表达力、能直接认出机器本身（而不是它遥测数据里的模式）的模型，报出的验证分数，跟它在一台真
正从未见过的机器上的表现几乎没什么关系。解决办法是同时按机器和按时间切分：用较早的日历月份训练，用较晚的
月份测试，切分两侧出现的机器互不重叠——绝不按行随机切分。下文的估算核对，用同一个简单模型分别按两种方式切
分，展示了两者之间的差距。

### 特征

**特征族。** 对原始遥测数据做窗口聚合——在滚动的 $1$ 天、$7$ 天、$30$ 天窗口上取均值、最大值和斜率——把一条
每分钟的数据流变成一个固定长度的每周特征向量；已纠正内存错误及类似事件计数器的计数与速率做对数变换，因为
它们是重尾分布（大多数机器只记录寥寥几次，少数机器能记录到几千次）；差分（delta）在窗口水平之上再捕捉短期
的变化速率；静态属性（机器型号、机龄、数据中心、机架位置、固件版本）很少变化，直接按 $t$ 时刻查表即可；机
队相对特征把一台机器自己的读数拿去和同一型号的其他机器比较——针对该型号自身的均值和标准差算 $z$ 分数，而
不是针对整个机队——因为不同机器型号本身运行的基准温度和错误率就有真实差异，“比这个型号平时高 $4$°C”
是有信息量的，而“比全机队平均高 $4$°C”则不然；维修历史（工单计数、距上次工单的时间，只用 $t$ 之前的
工单算出）捕捉的是一台已经状况较差的机器往往会继续状况较差这一点。

**变换因模型族而异。** 线性或逻辑回归模型需要对输入做缩放——否则拟合出的权重、乃至任何正则化惩罚，都会被恰
好数值范围最大的那个特征主导——把偏态的计数在缩放之前先做对数变换，类别属性做独热编码或目标编码，模型该用
到的任何交互项都要显式写进去，因为线性模型自己发现不了交互项。树集成模型完全不需要这些：分裂阈值对它所分
裂的特征的任何单调变换都是不变的，所以对数变换不会改变一棵树能学到的任何东西；缺失值会被路由到训练数据更
偏好的那个分支，不需要填补步骤；类别属性可以直接当作普通整数或原生类别使用，不需要做独热展开。神经网络需
要归一化的输入以保证基于梯度的训练稳定，对于高基数的类别特征，需要一个学出来的嵌入而不是独热编码，独热编
码既会非常宽，又无法在相似类别之间共享统计强度。

**机器型号及其他类别特征的嵌入。** 数据中心（$20$ 个取值）的基数低到可以直接做独热编码；机器型号（约 $300$
个取值，每季度还在增加）和固件版本则不是，它们在神经网络里得到一张学出来的嵌入表，与模型的其余部分联合训
练。本季度才引入的型号还没有任何行可供学出它自己的嵌入，所以它的嵌入不能是一次按 ID 的裸查表；解决办法是
转而用这个型号自身的描述性属性——供应商、CPU 代次、磁盘型号、内存配置——通过一个所有型号共享的小子网络来构
建嵌入，这样一个全新型号在被引入的当天就能得到一个合理的嵌入，用的是立刻就能拿到的属性，而不用等待模型本
来就不可能及时拿到的故障历史。更粗糙的退路——一个共享的未登录（out-of-vocabulary）行，或者把型号标识符哈
希进一个固定大小的表——搭建成本更低，但会丢掉恰恰是那些让一个新型号的嵌入在它拥有自己的历史之前就有意义的
属性；回退到所属硬件家族的嵌入，是在完整属性集尚未进入目录时的一个合理折中方案。不论用哪种机制，它都要在
型号被引入的那一刻就在服务端被真正用上，因为机队会立刻开始为它上报遥测数据——远早于它自己积累出 $52$ 周的
快照。

### 模型与损失函数

三个模型，按它们直接利用时间的程度递增排列：

**(i) 逻辑回归** 预测 $p(x)=\sigma(w\cdot x+b)$，用加权二元交叉熵拟合，

$$\mathcal{L}=-\frac{1}{N}\sum_i \big[c_1\, y_i\log p_i + c_0\,(1-y_i)\log(1-p_i)\big],$$

其中 $c_1>c_0$ 给稀少的正类加权（见下文“不均衡”一节）。训练和上线都快，系数也能直接读出来，这对向技
术人员解释一台具体机器的分数很重要（见下文“部署与反馈”一节）；它的准确率上限，取决于做完上述变换之
后的这些特征，是否真的在更换的对数几率（log-odds）上是线性的。

**(ii) 梯度提升树**，逐阶段拟合：在第 $m$ 阶段，当前模型 $F_{m-1}$ 的对数损失是
$-[y\log\sigma(F)+(1-y)\log(1-\sigma(F))]$，它对 $F$ 的负梯度是 $y-\sigma(F_{m-1}(x))$——观测到的标签
减去当前预测的概率——所以下一棵树 $h_m$ 被拟合来逼近这个残差，模型随之更新为 $F_m=F_{m-1}+\eta\,h_m$。由
于树集成模型天然就能处理上文的变换（见“特征”一节），在不做额外特征工程的情况下，它通常是这里最强的
单一表格模型，代价是它的分数需要一个显式的归因步骤（见“部署与反馈”一节）才能解释，而不能像系数那样
直接读出来。

**(iii) 一个离散时间风险（hazard）模型**，把机器在机队里的每一周都当作一次试验：$h_k(x)=\sigma(w\cdot
x_k+b)$ 是它在第 $k$ 周需要更换的概率，前提是它在第 $k-1$ 周之前都不需要更换。在一台机器真正被观察到的这
些周里，似然函数正好就是一个逻辑回归的似然函数，只不过是在按机器—周（machine-week）计的每一个仍处于风险
中的行上拟合，而不是每台机器一行：

$$\text{NLL}=-\sum_i\sum_{k=1}^{K_i}\big[y_{i,k}\log h_k(x_{i,k})+(1-y_{i,k})\log(1-h_k(x_{i,k}))\big],$$

其中 $K_i$ 是机器 $i$ 被观察到的最后一周，$y_{i,k}=1$ 仅当 $k=K_i$ 且那一周确实发生了更换——所以一台被删
失的机器（见上文“标签与训练数据”一节）在它真正被观察到的每一周里，只贡献生存项 $\log(1-h_k)$，绝不
会在它离开数据的那一刻，凭空贡献一个编造出来的发生项或未发生项。这正是一条被删失的行真正携带的信息，也是
它唯一携带的信息。既然这些风险本身就是条件概率——“在第 $k$ 周被更换，前提是活到了第 $k$ 周”——连续
活过四周的概率就是分别活过每一周的概率之积，$\prod_{k=1}^4(1-h_k)$——每周一次快照，四周正是这个设计对本页
通篇所用的 $30$ 天窗口的离散化——所以这个设计用来排序的风险是

$$P(\text{30 天内被更换})=1-\prod_{k=1}^{4}(1-h_k).$$

一个可选的做法是，用一个对原始每周遥测窗口做编码的序列模型，取代手工聚合的 $x_k$，让这个模型自己学出时序
特征，代价是比手工特征版本需要更多的数据和算力。

三个模型都在同一批每周快照行上训练和评估；(iii) 额外需要下文估算核对所构建的、按机器—周展开的表格，也是三
者之中唯一一个使用了被删失机器的部分历史、而不是丢弃或错标它的模型。

### 不均衡

**类别权重。** 在上面的损失里设置 $c_1/c_0$——比如设成与各类别频率成反比——让优化器为稀少的正类付出和大量
的负类同样多的总注意力；每一行仍然都被用到，但训练算力不变，因为其中大部分仍然花在占 $99.5\%$ 的带负标签
的行上。

**降采样，以及对它做校正。** 保留每一条正例行，但只保留负例行里随机的一部分——保留率为 $r<1$——能把训练算
力大致降到原来的 $r$ 倍，但会改变拟合出的模型所估计的东西。记 $p(x)$ 为真实的更换概率，$p'(x)$ 为在降采样
数据上训练出的模型所估计的量。正例行永远被保留，负例行只有 $r$ 的概率被保留，所以由贝叶斯公式（Bayes'
rule），$p'(x)=P(y=1\mid x,\text{kept})=p(x)/\big(p(x)+r\,(1-p(x))\big)$：模型实际看到的标签的*优势比*
（odds），是真实优势比除以 $r$：

$$\frac{p'(x)}{1-p'(x)}=\frac{1}{r}\cdot\frac{p(x)}{1-p(x)}.$$

解出 $p(x)$，就得到了在上线打分时套用在每一个预测上的校正公式：

$$p(x)=\frac{r\,p'(x)}{r\,p'(x)+1-p'(x)}.$$

下文的估算核对在降采样数据上拟合一个逻辑回归，同时确认了这两点：未经校正的 $p'$ 明显高于真实比率，而套用
这个公式之后，能把它恢复到离真实比率只差零点几个百分点的范围内。

**Focal loss** 是这两者之外的另一种选择：它重新加权的是*损失*本身，而不是数据行，用的是 $-(1-p_t)^\gamma\log(p_t)$，
其中 $p_t$ 是模型对真实类别预测出的概率，这样一个已经很容易、已经判对的负例（$1-p_t$ 很小，
$(1-p_t)^\gamma$ 就更小）对梯度的贡献几乎为零，把训练集中在稀少的正例、以及模型当下判错的那些负例上。它
和降采样不同，用到了每一行数据，也不需要降采样那种特定的概率校正，但引入了自己的超参数 $\gamma$ 需要调，
并且和任何对损失的重新加权一样，在它的输出被当作概率、而不只是排序分数来信任之前，仍然需要自己的校准检
查。

### 评估

**指标必须匹配这个决策。** 运维每周对排名前 $1{,}000$ 的机器采取行动，所以在运营上真正要紧的两个数字是
**precision@$1{,}000$**（在这 $1{,}000$ 台被维护的机器里，有多大比例确实需要维护）和 **recall@$1{,}000$**
（这一周大约 $5{,}000$ 个真正例里，有多大比例被捕获）——而按上文“需求与规模”一节推出的结论，无论模
型多好，recall@$1{,}000$ 都封顶在 $20\%$，并且在一周之内它就等于 precision@$1{,}000\times1{,}000/5{,}000$，
所以作为核心数字汇报的是 precision@$1{,}000$——被维护的机器里确实需要维护的比例，取值覆盖完整的 $0$–$100\%$。
**PR-AUC**
（精确率对召回率、随截断点扫过每一个可能的排名——不只是 $1{,}000$——所围出的面积）作为一个不依赖阈值的汇
总指标，与之一起跟踪，用于比较模型，也用于在一次重训到达 top-$1{,}000$ 这个截断点之前先做回归检验。

**为什么不能只看 ROC-AUC。** ROC-AUC 是“一个随机抽到的真正例，分数排在一个随机抽到的真负例前面”的
概率，它的 $x$ 轴，假阳性率，是相对于*负例*总体算出来的比例——在 $0.5\%$ 的正例比例（prevalence）下，这几
乎就是整个机队。一个模型完全可能把它列表顶端的大部分都排错——$1{,}000$ 个名额里只塞进了区区几百个真正的正
例——却仍然显示出一个很小的假阳性率，因为这几百个假阳性，相对于那大约 $995{,}000$ 台原本就不需要更换的机
器而言，只是极小的一部分。下文的估算核对正好构造出这种情形：ROC-AUC 约为 $0.82$，precision@k 却不到
$20\%$，而同一个截断点上的假阳性率还不到千分之一。这里 ROC-AUC 并没有错——它回答的只是另一个问题（整个总
体上的排序质量），而不是运维真正要问的那个问题（固定的 $1{,}000$ 个选择里，有多少是值得的）。

**校准。** 精确率和召回率只取决于模型给出的*顺序*，与绝对概率无关；但下面的成本计算，以及上文的降采样校
正，用到的都是概率本身，所以只要有一个概率——而不只是一个排名——要喂给下游的决策，就该在排序类指标之外，再
加一个可靠性检查（预测概率与每个分数区间里实际观测到的更换率相比对）。

**预期节省的成本。** 一次预先更换的代价定为 $1$ 个单位，一次计划外故障的代价是题设的 $20$ 倍，那么每周维护
排名前 $1{,}000$ 台机器的代价固定是 $1{,}000$，不论其中究竟有几台真的需要；而其中每一个真正例，都是一次不
再以计划外方式发生的故障，省下 $20$ 个单位。相对于完全不按这一周的列表采取行动，每周节省的成本是

$$\text{节省} = 20\times\text{TP} - 1{,}000,$$

其中 TP 是前 $1{,}000$ 名里真正例的数量。一周排序完全精确（$\text{TP}=1{,}000$）能节省 $19{,}000$ 个单
位；一个不比随机挑选更好的排序（$\text{TP}\approx1{,}000\times5{,}000/1{,}000{,}000=5$），相对于什么都
不做，反而要*多亏*约 $900$ 个单位——因为产能几乎全都花在了不需要的机器上，而真正的故障照样计划外发生。真
正把这个项目从净亏损变成净节省的，是 precision@$1{,}000$，而不是有没有一个模型这件事本身。

**回测。** 单次的训练/测试切分，会低估性能随日历时间推移而漂移的程度，所以这个设计采用滚动起点回测：用
$30$ 天标签窗口在第 $W$ 周之前已经结束的那些快照训练——这正是生产环境在第 $W$ 周时手里会有的标签——在紧接
着的几周上评估，再把 $W$ 往后推、重复这个过程，汇报的是 precision@$1{,}000$、
recall@$1{,}000$ 和 PR-AUC 在各折上的分布，而不是单一的一个点估计——和训练集切分（见上文“标签与训练数
据”一节）同样的按机器与按时间的纪律，延伸出对趋势的判断，而不只是一次快照。

### 部署与反馈

```text
遥测数据、维修工单、下线记录
                |
                v
   每周快照 + 标签构造                    （标签与训练数据）
                |
                v
   训练：逻辑回归 / 梯度提升 / 风险（hazard）模型     （模型与损失函数）
                |
                v
   对全机队做每周批量打分
                |
                v
   排序列表 + 每台机器的主要贡献特征
                |
                v
   运维：预先维护排名前 1,000 台（每周产能）  ----->  一小部分随机留出的机器：永不预先维护
                |                                              |
                v                                              v
   结果：故障发生前就被更换了                        结果：自己发生故障，或没有发生故障
                |                                              |
                '--------------------> 作为新的维修工单／检修结果，反馈回上面
```

**上线与阈值。** 每周，当前模型给全部 $1{,}000{,}000$ 台机器打分，产出一份排序列表；真正起作用的阈值是
*排名*，而不是一个固定的概率截断点，因为这个设计所服务的约束是一个固定的数量——每周 $1{,}000$ 台机器——而不是
一个固定的置信水平，一个概率截断点会让运维每周拿到数量不同的候选名单，具体多少取决于那一周状况有多糟。在排
名截断之下再加一道概率下限——跳过排名前 $1{,}000$ 里那些自身预期节省成本（见上文“评估”一节）已经转负
的机器——能在特别平静的一周里兜底，避免连排名第 $1{,}000$ 的机器都不值得真去维护一次：维护一台校准后 $30$
天风险为 $p$ 的机器，期望能节省 $20p-1$ 个单位，所以这道下限位于 $p=1/20=5\%$，只有当排名第 $1{,}000$ 的机器
的风险低于它时才会真正起作用。列表上的每一台机器都附带它
的主要贡献特征——逻辑回归给出系数乘以取值的拆解，树集成模型给出可比的逐特征归因——这样技术人员拿到的是一个
写明的理由去核查，而不只是一个光秃秃的分数。

**监控漂移。** 每季度进入机队的新机器型号，需要它基于内容构建的嵌入（见上文“特征”一节）被真正用上，而
不是被悄悄地退回到默认值；一次推送到整批机器上的固件更新，会让它们的绝对遥测读数一夜之间发生变化，机队相对
的 $z$ 分数特征只有在它据以计算的参照分布本身保持更新的前提下，才能吸收这种变化；而寻常的季节性温度波动，会
让整个机队的绝对温度读数同时发生变化，同样的机队相对设计选择也能防住这一点，前提是参照窗口随季节向前滚动，
而不是冻结在训练时刻。由于结果标签比快照本身要滞后 $30$ 天，滚动回测给出的 precision@$1{,}000$ 和 PR-AUC
是一个滞后信号；逐周跟踪每个特征自身的分布，能在漂移体现到结果之前就先捕捉到它。

**反馈循环。** 一台被模型标记出来、运维又预先更换掉的机器，永远失去了自己发生故障的机会，所以唯一能证实或推
翻这次预测的那个标签——它本来会不会真的发生故障——也就永远不会产生；每一台模型成功促成了行动的机器，都从此
永久性地无法被自己未来的训练数据观察到，这是*模型自身的决策*制造出来的一种删失，而不是数据采集上的意外。如
果放任不管，日志里剩下的正例就会向模型漏掉的故障、以及排序从未触及的那些机器倾斜，这会低估模型本来已经达到
的水平，并悄悄地让模型不断在一个越来越不能代表其真实预测对象的样本上被重新训练。两种缓解办法搭配使用：机队
里固定留出一小部分随机样本，不论模型怎么说，*永远*不做预先更换，这样它们的真实结果最终总会被观察到，给出一
个无偏的精确率与召回率读数，代价是要在这部分机器上接受几次真实发生的计划外故障，以此换来一个诚实的评估信
号；而对于被预先维护掉的机器，被拆下来的部件会被送去台架检修，用一次直接的测量——它是不是真的快坏了——去替
代这个设计刻意选择不去观察的那个现场结果。

### 追问

- **用部件级别的模型，取代单一的机器级别模型。** 分别为磁盘、内存和风扇单独预测——每个都有自己的标签（更换
  了那个具体部件的工单）和自己的特征子集——给运维的是一个已经等于一个具体行动的理由（该带哪个备件、先跑哪项
  测试），而不是一个还需要自己诊断的单一“这台机器有风险”分数；代价是要维护和评估 $k$ 个独立的模型，而
  不是一个，而且机队级别的排序现在还需要一条规则，把 $k$ 个独立的风险分数合并成运维真正据以工作的那一份列
  表。
- **在线更新。** 在一个滚动窗口上每周重新训练，能让模型跟上固件变化和新机器型号，不必等待一次完整的离线周
  期，代价是模型逐周的波动更大，需要同样的滚动回测（见上文“评估”一节）来确认它没有退化，然后才能把下
  一周的列表托付给它。
- **共享的检修产能。** 当每周 $1{,}000$ 台的名额本身还要和其他维护工作（一次固件推送、一次计划内审计）共享
  时，这个设计产出的排序，就变成了进一步做产能分配的一个输入，而不是分配本身——预期节省成本这个数字（见上文
  “评估”一节），正好是能让这次分配和争抢同一批技术人员的其他工作放在同一个基准上比较的那个数字。
- **向技术人员解释预测结果。** 一个光秃秃的分数每次都会招来同一个追问——为什么是这台机器——所以随排序列表
  一起发出的逐特征归因（见上文“部署与反馈”一节）应该换算成技术人员会去核对的那种单位（一个以度为单位
  的温度、一个错误计数，而不是一个原始的特征权重），并且要随时间追踪：一台机器排名第一的原因逐周都在变，这
  本身就说明该重新审视的不只是模型的分数，还有它给出的解释。

<details>
<summary>估算核对（可运行）</summary>

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

# ---- requirements and scale ----
n_machines = 1_000_000
n_metrics = 50
minutes_per_day = 24 * 60
readings_per_day = n_machines * n_metrics * minutes_per_day
assert readings_per_day == 72_000_000_000

bytes_per_reading = 8
gb_per_day = readings_per_day * bytes_per_reading / 1e9
assert gb_per_day == 576.0

weeks_per_year = 52
snapshot_rows_per_year = n_machines * weeks_per_year
assert snapshot_rows_per_year == 52_000_000

monthly_replace_rate = 0.005
expected_positives_per_week = n_machines * monthly_replace_rate
assert expected_positives_per_week == 5_000

negative_label_fraction = 1 - monthly_replace_rate
assert negative_label_fraction == 0.995

never_needs_replacing = n_machines - expected_positives_per_week
assert never_needs_replacing == 995_000

weekly_capacity = 1_000
recall_ceiling = weekly_capacity / expected_positives_per_week
assert recall_ceiling == 0.2

snapshots_per_window = 30 / 7                  # a 30-day window spans this many weekly snapshots, so
assert round(snapshots_per_window, 1) == 4.3    # consecutive snapshots share most of their positives
new_replacements_per_week = expected_positives_per_week / snapshots_per_window
assert round(new_replacements_per_week, -1) == 1_170

print("all requirements-and-scale numbers check out")

# ---- expected cost saved, at the stated 20:1 cost ratio ----
c_pre, c_fail = 1.0, 20.0
K, P = weekly_capacity, expected_positives_per_week

break_even_risk = c_pre / c_fail   # servicing a machine of 30-day risk p saves c_fail * p - c_pre in expectation
assert break_even_risk == 0.05


def cost_saved(true_positives):
    baseline = P * c_fail                                    # do nothing: every positive fails unplanned
    policy_cost = K * c_pre + (P - true_positives) * c_fail  # service K regardless; miss the rest
    return baseline - policy_cost


saved_perfect = cost_saved(K)                # every one of the 1,000 picks is a true positive
assert saved_perfect == 19_000

tp_random = K * (P / n_machines)             # a ranking no better than picking at random
assert tp_random == 5.0
saved_random = cost_saved(tp_random)
assert saved_random == -900.0

print("expected-cost-saved figures check out (perfect precision saves 19,000; a random pick loses 900; "
      "break-even risk 5%)")


# ---- precision@k / recall@k: sorting versus argpartition, and the recall = precision * (K/P) identity ----
def topk_by_sort(scores, k):
    return np.argsort(-scores)[:k]


def topk_by_partition(scores, k):
    return np.argpartition(-scores, k - 1)[:k]  # NOTE: only the SET of the k largest is guaranteed, not their order


rng = np.random.default_rng(1)
for _ in range(200):
    n = int(rng.integers(50, 500))
    k = int(rng.integers(1, n))
    y_true = rng.binomial(1, 0.1, size=n)
    scores = rng.normal(size=n)          # continuous scores: no ties, so the top-k SET is unambiguous
    set_sort = set(topk_by_sort(scores, k).tolist())
    set_part = set(topk_by_partition(scores, k).tolist())
    assert set_sort == set_part

print("precision/recall @k: sort- and argpartition-based top-k sets agree on 200 random trials")

n_toy, P_toy, K_toy = 50_000, 500, 100
y_true = np.zeros(n_toy, dtype=int)
pos_positions = rng.choice(n_toy, size=P_toy, replace=False)
y_true[pos_positions] = 1
non_pos = np.setdiff1d(np.arange(n_toy), pos_positions, assume_unique=True)
for true_positives in (100, 70, 20, 1):
    scores = rng.normal(size=n_toy)
    chosen_pos = rng.choice(pos_positions, size=true_positives, replace=False)
    chosen_neg = rng.choice(non_pos, size=K_toy - true_positives, replace=False)
    scores[chosen_pos] = 100 + rng.random(true_positives)
    scores[chosen_neg] = 100 + rng.random(K_toy - true_positives)
    idx = topk_by_sort(scores, K_toy)
    precision = y_true[idx].sum() / K_toy
    recall = y_true[idx].sum() / P_toy
    assert y_true[idx].sum() == true_positives
    assert abs(recall - precision * (K_toy / P_toy)) < 1e-12

print("recall@K == precision@K * (K / P) confirmed whenever K and P are fixed")


# ---- down-sampling negatives inflates predicted probability; the odds-derived correction restores it ----
rng = np.random.default_rng(2)
n_total, n_features = 200_000, 4
true_w, true_b = np.array([1.2, -0.8, 0.5, 0.3]), -5.0

X = rng.normal(size=(n_total, n_features))
p_true = 1.0 / (1.0 + np.exp(-(X @ true_w + true_b)))
y = rng.binomial(1, p_true)

n_train = 150_000
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]          # held out, never down-sampled: it must reflect the real world

lr_full = LogisticRegression(C=1e6, max_iter=1000).fit(X_train, y_train)

r = 0.1  # keep-rate for negatives
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]
keep_neg = neg_idx[rng.random(len(neg_idx)) < r]
ds_idx = np.concatenate([pos_idx, keep_neg])
lr_ds = LogisticRegression(C=1e6, max_iter=1000).fit(X_train[ds_idx], y_train[ds_idx])

p_prime = lr_ds.predict_proba(X_test)[:, 1]                        # uncorrected, on the untouched test set
p_corrected = r * p_prime / (r * p_prime + (1 - p_prime))          # p = r p' / (r p' + 1 - p'), derived above
true_rate = y_test.mean()

inflation = p_prime.mean() - true_rate
residual = abs(p_corrected.mean() - true_rate)
print(f"true rate {true_rate:.4f}, mean uncorrected p' {p_prime.mean():.4f}, mean corrected {p_corrected.mean():.4f}")
assert inflation > 0.05          # clearly inflated
assert residual < 0.01           # correction restores it to within a fraction of a percentage point

# the intercept shift the derivation predicts: log(1/r), slope unchanged
assert np.allclose(lr_ds.coef_, lr_full.coef_, atol=0.05)
assert abs((lr_ds.intercept_ - lr_full.intercept_)[0] - np.log(1 / r)) < 0.05
print("down-sampling correction confirmed: inflated probability restored to within tolerance of the true rate")


# ---- ROC-AUC well above 0.5 can coexist with low precision@k, at low prevalence ----
rng = np.random.default_rng(3)
n_ranking = 200_000
prevalence = 0.005
n_pos = int(n_ranking * prevalence)
y_rank = np.zeros(n_ranking, dtype=int)
y_rank[:n_pos] = 1
rng.shuffle(y_rank)

signal = rng.normal(size=n_ranking)
score = np.where(y_rank == 1, signal + 2.0, signal) + rng.normal(scale=1.2, size=n_ranking)


def auc_from_ranks(y_true, s):
    """AUC = P(random positive scores above random negative), via the Mann-Whitney rank-sum identity --
    an independent definition-level computation, not a call into sklearn's implementation."""
    ranks = stats.rankdata(s)
    n_p, n_n = y_true.sum(), len(y_true) - y_true.sum()
    return (ranks[y_true == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)


auc_hand = auc_from_ranks(y_rank, score)
auc_lib = roc_auc_score(y_rank, score)
assert abs(auc_hand - auc_lib) < 1e-9
assert round(auc_hand, 2) == 0.82

k_rank = 200  # 0.1% of n_ranking, the same ratio as 1,000 of 1,000,000 machines
idx = np.argsort(-score)[:k_rank]
precision_k = y_rank[idx].sum() / k_rank
fpr_k = (k_rank - y_rank[idx].sum()) / (n_ranking - n_pos)
print(f"ROC-AUC {auc_hand:.3f}, precision@{k_rank} {precision_k:.3f}, "
      f"false-positive rate at the same cutoff {fpr_k:.4f}")
assert precision_k < 0.2
assert fpr_k < 0.001         # the same false positives are a tiny share of the (huge) negative population
print("ROC-AUC-versus-precision@k dichotomy at low prevalence confirmed")


# ---- a small seeded synthetic fleet: hidden degradation before failure, some unrelated decommissions ----
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate_fleet(seed, n, n_weeks, w0, w1, base_sd, slope_hi, admin_censor_p):
    rng_f = np.random.default_rng(seed)
    base = rng_f.normal(0, base_sd, size=n)
    slope = rng_f.uniform(0.0, slope_hi, size=n)
    outcome_week = np.full(n, n_weeks)
    outcome_type = np.array(["alive"] * n, dtype=object)      # alive (to end) | failed | censored
    for i in range(n):
        for k in range(1, n_weeks + 1):
            x = base[i] + slope[i] * k
            if rng_f.random() < sigmoid(w1 * x + w0):
                outcome_week[i], outcome_type[i] = k, "failed"
                break
            if rng_f.random() < admin_censor_p:
                outcome_week[i], outcome_type[i] = k, "censored"
                break
    return base, slope, outcome_week, outcome_type


SEED, N_FLEET, N_WEEKS, HORIZON = 7, 2_500, 30, 4
W0_TRUE, W1_TRUE = -4.2, 1.0
base, slope, outcome_week, outcome_type = simulate_fleet(
    SEED, N_FLEET, N_WEEKS, W0_TRUE, W1_TRUE, base_sd=0.15, slope_hi=0.3, admin_censor_p=0.025)

n_failed = int((outcome_type == "failed").sum())
n_censored = int((outcome_type == "censored").sum())
assert n_failed > 300 and n_censored > 100

# -- expanded machine-week-at-risk table: the discrete-time hazard model IS logistic regression on this table --
exp_i, exp_x, exp_event = [], [], []
for i in range(N_FLEET):
    w = outcome_week[i]
    for t in range(1, w + 1):
        exp_i.append(i)
        exp_x.append(base[i] + slope[i] * t)
        exp_event.append(1 if (t == w and outcome_type[i] == "failed") else 0)
exp_i = np.array(exp_i)
exp_x = np.array(exp_x).reshape(-1, 1)
exp_event = np.array(exp_event)

never_failed = np.where(outcome_type != "failed")[0]        # censored + administratively-ended machines
assert exp_event[np.isin(exp_i, never_failed)].sum() == 0    # they contribute ONLY survival terms, never an event

lr_hazard = LogisticRegression(C=1e6, max_iter=2000).fit(exp_x, exp_event)
w_hat, b_hat = lr_hazard.coef_[0, 0], lr_hazard.intercept_[0]
assert abs(w_hat - W1_TRUE) < 0.1 and abs(b_hat - W0_TRUE) < 0.2   # true hazard recovered, censored machines in

# the hazard NLL written per machine from its definition -- survive weeks 1..K_i-1, then fail or survive week
# K_i -- without the expanded table, equals the logistic log loss on the expanded table
nll_machine = 0.0
for i in range(N_FLEET):
    h = sigmoid(w_hat * (base[i] + slope[i] * np.arange(1, outcome_week[i] + 1)) + b_hat)
    nll_machine -= np.log(1 - h[:-1]).sum()
    nll_machine -= np.log(h[-1]) if outcome_type[i] == "failed" else np.log(1 - h[-1])
nll_table = log_loss(exp_event, lr_hazard.predict_proba(exp_x)[:, 1], normalize=False)
assert abs(nll_machine - nll_table) < 1e-6 * nll_table
print(f"fitted hazard slope {w_hat:.3f} (true {W1_TRUE}), intercept {b_hat:.3f} (true {W0_TRUE}); "
      f"per-machine NLL {nll_machine:.2f} matches the expanded table's log loss {nll_table:.2f}")

sample_x = np.array([[0.3], [0.5], [0.7], [0.9]])          # one machine's next 4 weekly feature values
h4 = lr_hazard.predict_proba(sample_x)[:, 1]
closed_form_risk = 1 - np.prod(1 - h4)
rng_mc = np.random.default_rng(99)
n_mc = 300_000
mc_risk = (rng_mc.random((n_mc, 4)) < h4[None, :]).any(axis=1).mean()
se = np.sqrt(mc_risk * (1 - mc_risk) / n_mc)
print(f"closed-form 30-day risk {closed_form_risk:.4f} vs. Monte Carlo {mc_risk:.4f} (s.e. {se:.4f})")
assert abs(mc_risk - closed_form_risk) < 6 * se
print("discrete-time hazard NLL and the 30-day risk product formula both confirmed")

# -- snapshot / classification table, with censoring-aware labels (drop what is genuinely unresolved) --
snap_i, snap_t, snap_x, snap_label = [], [], [], []
n_candidates = 0
for i in range(N_FLEET):
    w, typ = outcome_week[i], outcome_type[i]
    for t in range(1, w):
        window_end = t + HORIZON
        n_candidates += 1
        is_event_in_window = typ == "failed" and w <= window_end
        if is_event_in_window:
            snap_i.append(i); snap_t.append(t); snap_x.append(base[i] + slope[i] * t); snap_label.append(1)
        elif window_end <= w:
            snap_i.append(i); snap_t.append(t); snap_x.append(base[i] + slope[i] * t); snap_label.append(0)
        # else: the machine is censored or administratively ended strictly inside the window -- unresolved, dropped

snap_i, snap_t = np.array(snap_i), np.array(snap_t)
snap_x, snap_label = np.array(snap_x), np.array(snap_label)
print(f"kept {len(snap_label)} of {n_candidates} candidate snapshots")


# The two label-rate biases are about one percentage point each, comparable to one small fleet's sampling noise,
# so they are measured on a fleet eight times larger (same generator, another seed).
def label_rates(seed, n):
    b_, s_, w_, typ_ = simulate_fleet(seed, n, N_WEEKS, W0_TRUE, W1_TRUE, base_sd=0.15, slope_hi=0.3,
                                      admin_censor_p=0.025)
    naive_pos, kept_pos, kept, total, true_sum = 0, 0, 0, 0, 0.0
    for i in range(n):
        for t in range(1, w_[i]):
            window_end = t + HORIZON
            ahead = np.arange(t + 1, window_end + 1)   # true P(fail in the window | alive at t), uncensored
            true_sum += 1 - np.prod(1 - sigmoid(W1_TRUE * (b_[i] + s_[i] * ahead) + W0_TRUE))
            event = typ_[i] == "failed" and w_[i] <= window_end
            total += 1
            naive_pos += event
            if event or window_end <= w_[i]:
                kept += 1
                kept_pos += event
    return naive_pos / total, kept_pos / kept, true_sum / total   # unresolved as negative, dropped, true


naive_rate, dropped_rate, true_rate = label_rates(seed=8, n=8 * N_FLEET)
print(f"true rate {true_rate:.4f}, unresolved-as-negative {naive_rate:.4f}, unresolved dropped {dropped_rate:.4f}")
assert naive_rate < true_rate - 0.004    # counting unresolved rows as negatives understates the rate...
assert dropped_rate > true_rate + 0.002  # ... and dropping them overstates it: an early failure resolves its row,
                                         # while a machine that survives until it leaves the data is dropped

# -- random row split vs. split by machine AND time, for the SAME (deliberately simple, memorization-capable) model --
ID_SCALE = 1_000.0  # dominates Euclidean distance unless machine_id matches exactly
features = np.column_stack([snap_i * ID_SCALE, snap_x])
n_snap = len(snap_label)
rng_split = np.random.default_rng(11)

perm = rng_split.permutation(n_snap)
cut = int(0.7 * n_snap)
train_idx, test_idx = perm[:cut], perm[cut:]
knn_random = KNeighborsClassifier(n_neighbors=1).fit(features[train_idx], snap_label[train_idx])
auc_random = roc_auc_score(snap_label[test_idx], knn_random.predict_proba(features[test_idx])[:, 1])
nn = knn_random.kneighbors(features[test_idx], n_neighbors=1, return_distance=False).ravel()
same_machine_random = (snap_i[train_idx][nn] == snap_i[test_idx]).mean()

test_machines = rng_split.choice(N_FLEET, size=int(0.3 * N_FLEET), replace=False)
is_test_machine = np.isin(snap_i, test_machines)
time_cutoff = 18
train_mask = (~is_test_machine) & (snap_t < time_cutoff)
test_mask = is_test_machine & (snap_t >= time_cutoff)
knn_grouped = KNeighborsClassifier(n_neighbors=1).fit(features[train_mask], snap_label[train_mask])
auc_grouped = roc_auc_score(snap_label[test_mask], knn_grouped.predict_proba(features[test_mask])[:, 1])
nn2 = knn_grouped.kneighbors(features[test_mask], n_neighbors=1, return_distance=False).ravel()
same_machine_grouped = (snap_i[train_mask][nn2] == snap_i[test_mask]).mean()

print(f"random-row-split AUC {auc_random:.3f} (nearest neighbour is the same machine "
      f"{same_machine_random:.1%} of the time) vs. machine-and-time split AUC {auc_grouped:.3f} "
      f"({same_machine_grouped:.0%} of the time)")
assert same_machine_random > 0.95
assert same_machine_grouped == 0.0
assert auc_random - auc_grouped > 0.2

print("all checks passed")
```

</details>

</details>
