# 为内容信息流设计推荐系统

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 机器学习系统设计 | ★★★☆☆ | 中等 | MLE · SWE · Applied AI | candidate-generation, two-tower, ranking, multi-task-learning, feedback-loops, cold-start, ndcg | 45–60 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

设计支撑个性化短视频信息流首页的后端：用户逐条滑动浏览的一串内容，其中每一条都是专门为这个用户从大家共享的内容库中挑选出来的。*信息流请求*（feed request）是客户端在用户打开或刷新信息流时发起的一次调用，返回接下来要展示的一份有序内容列表。通过任意一次信息流请求把某一条内容展示在某个用户面前，就是一次*曝光*（impression）——无论用户是否对它做出任何反应，都要记录。

产出一次信息流请求的结果分三个阶段依次进行。*候选生成*（candidate generation）把整个内容库缩小到一个小得多的集合，里面装的是这个用户很可能感兴趣的内容，并且要便宜到能为每一次请求都跑一遍。这里选用的机制是*双塔模型*（two-tower model）：两个联合训练的神经网络，一个把用户（连同其近期行为与上下文）编码成一个定长的嵌入向量，另一个把一条内容编码成相同长度的嵌入向量，训练目标是让用户很可能与之互动的那些（用户、内容）对，其两个向量的点积更大。在线上服务时，这把候选生成变成了在一份预先算好的内容嵌入索引里，搜索与用户嵌入最接近的那些——一个*近似最近邻索引*（approximate nearest-neighbour index，ANN 索引），用一点点召回率换取速度，相对于对内容库里每一条内容都精确打分而言。*排序模型*（ranking model）随后从好几个不同的角度分别预测用户与每个存活候选互动的可能性，并把这些预测合并成一个数字。*重排*（re-ranking）拿到打分最高的这些候选后，在排序分数本身没有覆盖到的约束下重新排列——相邻内容之间的多样性、给最近上传视频的新鲜度、以及政策规则——然后系统才返回这份重排结果里靠前的部分。

有两个问题横跨这三个阶段。*冷启动*（cold start）指的是为一个几乎没有互动历史的用户或内容生成推荐——一个刚注册、还没有观看记录的新用户，或者一条几秒钟前刚上传、还没有任何互动记录的视频。*曝光偏差*（exposure bias）是只用已记录的互动来训练所带来的扭曲：日志里只会记录某个更早版本的系统选择展示过的内容上的反馈，所以如果直接用这份日志天真地训练模型，模型只会学到去偏好那个更早的系统本来就偏好的东西，永远发现不了一条它从未展示过的好内容。

为下面的规模设计：

- 日活跃用户 $200{,}000{,}000$ 人。
- 一个活跃用户平均每天打开信息流 $10$ 次；每次打开就是一次信息流请求，返回 $20$ 条内容。
- 峰值流量是日均的 $3$ 倍。
- 内容库共有 $500{,}000{,}000$ 条视频，每天新增 $5{,}000{,}000$ 条上传。
- 信息流请求的延迟目标：第 99 百分位数（p99）低于 $200$ 毫秒。
- 互动事件——曝光、观看时长、点赞、分享、跳过——必须在发生后 $1$ 小时内可供模型使用。
- 业务目标是用户的长期满意度，而不是原始的点击或观看次数。

范围内：候选生成，包括双塔模型及其他候选来源；排序模型，以及它的预测如何合并成一个分数；用于多样性、新鲜度与政策的重排；服务架构与互动日志；训练流水线——标注、负采样、延迟反馈、重新训练的节奏；新用户与新视频的冷启动；反馈循环与曝光偏差；以及离线与在线评估。范围外：主页信息流之外的推荐场景（搜索结果、通知）；视频文件本身的存储、转码与分发；决定一条视频是否允许出现在平台上的内容安全分类器（假设每个候选内容到达本系统时，已经带有一个政策合规标记）；以及身份认证。

要产出：

1. 需求与规模估算：平均和峰值的信息流请求速率；排序模型每秒打分的内容条数（说明你对它每次请求打分多少个候选的假设）；$500{,}000{,}000$ 条内容、每条 $128$ 维 `int8` 嵌入向量的检索索引所占内存；以及每天的互动日志体量（说明你对一条已记录事件大小的假设）。
2. 用户、内容与互动的数据模型，说明哪些特征必须在 $1$ 小时的新鲜度限制内可用，哪些可以按更慢的周期更新。
3. 一张架构图，以及沿着一次信息流请求走一遍的过程，覆盖候选生成（双塔检索索引，加上至少一个其他候选来源）、排序、重排，以及互动如何被记录下来用于训练。
4. 训练流水线：标签如何从互动中得出、双塔模型的负采样、延迟反馈如何处理，以及每个模型各自的重新训练节奏。
5. 深入话题：(a) 如何选择排序目标，把多个预测信号合并成一个分数；(b) 新用户与新视频的冷启动；(c) 反馈循环与曝光偏差，包括探索与记录倾向分；(d) 离线评估（候选生成用 recall@k，排序用 NDCG@k）与在线评估（A/B 测试指标与护栏指标）。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手设计之前值得先确认：“长期满意度”具体用什么来衡量（这里假设：回访率，与花在已完整观看、而非被跳过的内容上的总时长，二者的结合，并定期用更慢的在线读数加以校验，而不是永远只信任单一的代理指标——参见下面目标选择与评估两个深入话题），以及候选内容到达这个系统之前，已经有哪些约束施加在它身上（这里假设：每个候选到达时都已经带有一个政策合规标记，这个设计把它当作既定输入，而不是由自己来判断）。

### 需求与规模

**信息流请求速率。** $200{,}000{,}000$ 名日活跃用户，每人每天打开信息流 $10$ 次：

$$200{,}000{,}000 \times 10 = 2{,}000{,}000{,}000 \text{ 次请求/天，} \qquad
\frac{2{,}000{,}000{,}000}{86{,}400} \approx 23{,}148 \text{ 次请求/秒（平均）。}$$

按日均的 $3$ 倍计算，峰值吞吐量约为 $69{,}444$ 次请求/秒。

**返回的内容数与打分的内容数。** 每次请求返回 $20$ 条内容，所以响应路径——以及它的日志记录——必须支撑平均约 $23{,}148 \times 20 \approx 462{,}963$ 条/秒、峰值约 $1{,}388{,}889$ 条/秒的内容量。排序模型看到的数字更大：候选生成被设定为每次请求给出 $C = 500$ 个候选，是最终展示数量的 $25$ 倍——足够宽，让重排在多样性和新鲜度上真的有得选；又足够窄，能把排序模型每次请求的开销限制住——所以排序模型平均要为约 $23{,}148 \times 500 \approx 11{,}574{,}074$ 条/秒的内容打分，峰值约 $34{,}722{,}222$ 条/秒。

**排序机群规模。** 假设一个排序副本在一次批处理前向传播里，为一次请求的 $500$ 个候选打分，耗时约 $20$ 毫秒，吞吐量为 $500 / 0.02 = 25{,}000$ 条/秒。满足峰值需求，裸机需要 $34{,}722{,}222 / 25{,}000 \approx 1{,}389$ 个副本；再加上应对负载不均和滚动发布的 $20\%$ 余量，是 $1{,}667$ 个副本。（如果不是每次调用只处理一个请求的候选，而是把多个请求的候选打包进同一次前向传播，会提高单个副本的吞吐量，从而缩小机群规模；这里的算术采用的是更简单的单请求单调用模型。）

**候选索引内存。** 检索索引为内容库中的每一条内容保存一个 $128$ 维嵌入向量，量化为 `int8`——每一维一个字节：$500{,}000{,}000 \times 128\text{ 字节} = 64{,}000{,}000{,}000$ 字节，正好 $64$ GB。按每片约 $8$ GB 分片——足够小，能连同服务开销一起舒服地放进一台机器的内存——就是 $8$ 个分片；再按 $3$ 倍复制，以保证可用性并分摊查询压力，共 $24$ 台机器承载这个索引。内容库每天新增 $5{,}000{,}000$ 条，是其规模的 $1\%$，只带来约 $5{,}000{,}000 \times 128\text{ 字节} \approx 0.64$ GB 的新增嵌入向量——索引的内存占用是由已有的内容库大小决定的，不是由每日的新增量决定的。

**事件量。** 每一次曝光都被记录为一行：一个事件 id、把它和同一次请求里其余 $19$ 条内容归到一起的请求 id、用户 id、内容 id 和一个服务器时间戳（各 $8$ 字节，共 $40$ 字节），加上它的结果——在信息流中的位置（$1$ 字节）、以毫秒为单位的观看时长（$4$ 字节）、一个由点赞、分享等互动组成的位掩码（$1$ 字节），以及展示时所用的排序分数、留作后续分析（以 float32 存储，$4$ 字节）——合计 $50$ 字节/事件。按 $200{,}000{,}000 \times 10 \times 20 = 40{,}000{,}000{,}000$ 次曝光/天计算，就是 $40{,}000{,}000{,}000 \times 50\text{ 字节} = 2{,}000{,}000{,}000{,}000$ 字节，正好 $2$ TB/天。为训练和离线评估保留的 $30$ 天热数据窗口，因此约为 $60$ TB。

### 数据与特征

**用户** —— 一份稳定的档案（`user_id`、注册日期、声明的语言地区、引导流程中选择的话题偏好），按每日批处理周期更新；加上一份近期活动的滚动摘要（会话次数、最近 $N$ 次互动过的内容及其结果），由下文的流式处理管线更新；再加上会话内状态（本次会话中已经展示过或跳过的内容），存放在一个低延迟缓存里，在请求内部同步读取，因为它必须反映最近几秒钟的情况，而不是最近一小时。双塔模型的用户塔消费这份档案和滚动摘要，产出一个定长的*用户嵌入*（user embedding），每次请求需要时都重新计算——这很便宜，因为它只是一次小规模的前向传播，不是对预先算好的表做一次查找。

**内容** —— 上传时就固定下来的元数据（`item_id`、`creator_id`、时长、语言、话题标签、上传时间），加上滚动的互动率聚合特征（近 24 小时和近 7 天的完播率、点赞率、分享率），由流式处理管线更新。内容塔消费这两者，产出一个*内容嵌入*（item embedding）；和用户嵌入不同，它是为整个内容库预先算好、并按下文的训练节奏刷新的，因为候选生成需要检索的是一份固定的、已建好索引的向量集合，而不是在请求时现算每一条内容的嵌入。

**互动** —— 每次曝光一行，由服务路径本身写入：`event_id`、`request_id`、`user_id`、`item_id`、`position`、`served_at`、`watch_time_ms`、`engagement_flags`（点赞、分享）、以及 `ranking_score`。把本可能是五个独立事件——曝光、观看时长、点赞、分享、跳过——收拢进以曝光为键的一行，避免了为每条展示的内容写五行、之后还要把它们连接起来；*跳过*（skip）就是 `watch_time_ms` 接近零的一行，不是一种单独的事件类型。

**新鲜度。** 三层，对应每种特征变化的快慢，以及维持它保持最新的代价：会话内状态在请求内部同步读取，精确到秒；滚动的用户与内容聚合特征，由一个消费互动日志的流式作业更新，落在需求里 $1$ 小时的时限之内；用户与内容嵌入，以及训练得出的其余一切，都跟随下一节里的重新训练节奏，精确到天。一次信息流请求只会在第一层上阻塞；另外两层永远是读取后台流水线最近一次写入的结果。

### 架构

```text
客户端
  |
  | 信息流请求
  v
+----------------------------------------------------------------+
| 信息流 API（读取用户与会话特征；记录曝光）                     |
+----------------------------------------------------------------+
  |
  | 用户、上下文
  v
+----------------------------------------------------------------+
| 候选生成（Candidate generation）                               |
|                                                                |
|   * 双塔模型       -> 检索 ANN 索引                            |
|   * 新内容池       -> 按上传时间排序的最新视频                 |
|   * 关注创作者池   -> 所关注创作者尚未看过的新上传             |
|                                                                |
|   合并、去重，剔除本次会话已展示过及不合规的内容               |
+----------------------------------------------------------------+
  |
  | 约 500 个候选
  v
+----------------------------------------------------------------+
| 排序模型（Ranking model）                                      |
| 多任务：为每个候选预测多个互动信号，                           |
| 并合并为一个分数                                               |
+----------------------------------------------------------------+
  |
  | 已打分的候选
  v
+----------------------------------------------------------------+
| 重排（Re-ranking）                                             |
| 多样性、新鲜度与政策约束                                       |
+----------------------------------------------------------------+
  |
  | 最终 20 条 -> 信息流 API -> 客户端
  v
（响应已返回；每条展示的内容同时被记录，见下文）


+----------------------------------------------------------------+
| 事件日志（Event log，每条曝光一行）                            |
+----------------------------------------------------------------+
        |
        +--------------------------------+
        |                                |
        v                                v
+------------------+   +------------------------------------------+
| 流式特征聚合     |   | 离线训练流水线                           |
| （1 小时内更新） |   | （标签、负采样、重新训练——见“训练”一节） |
+------------------+   +------------------------------------------+
        |                                   |
        v                                   +----------------------+
   特征存储                                  |                      |
   （供候选生成、                             v                      v
    排序读取）                          新的内容嵌入           新的模型权重
                                       -> 检索索引              -> 排序模型
                                          （见上文）              （见上文）
```

一次信息流请求到达信息流 API，它从特征存储读取发起请求的用户的档案、滚动摘要和会话状态——这是关键路径上唯一的一次特征读取；内容特征早已由后台流水线并入了索引和排序模型的输入里。三个候选来源并行运行：双塔检索服务把用户编码成向量，在 ANN 索引里搜索按嵌入点积最接近的内容；新内容池按上传时间、在每个话题内排序，保存最近上传的内容，供那些还没来得及在嵌入索引里占据一席之地的新内容使用；关注创作者池返回用户所关注创作者尚未看过的新上传，对这层关系完全绕开相关性模型。候选合并对三路来源去重，剔除本次会话已经展示过的、以及未通过政策合规标记的内容，最终留下大约 $500$ 个存活候选。

排序模型在一次批处理前向传播里，为每一个存活候选打分，为每个候选产出若干互动概率预测并合并成一个分数（见下文目标选择的深入话题）。重排从分数最高的候选开始按顺序走一遍，在其约束下填满 $20$ 个输出位置——来自同一创作者的连续内容不超过一个较小的固定数量、留给最近一天内上传内容的位置有一个最低占比、以及其余的政策上限——跳过任何会违反约束的候选，转而选择次优的、不违反约束的那个。信息流 API 返回这最终的 $20$ 条内容，并独立于这次响应，为每条内容向事件日志记录一行互动。事件日志供两个消费者使用：让滚动特征保持在一小时新鲜度以内的流式聚合作业，以及离线训练流水线——它按“训练”一节给出的节奏，把新的内容嵌入写回检索索引，把新的模型权重写回排序服务。

### 训练

**标签。** 每一条训练数据都从一行互动记录出发（见上文“数据与特征”）：$P(\text{观看} \ge 50\%)$ 的标签在 `watch_time_ms` 达到这条内容时长一半时为 $1$；$P(\text{点赞})$ 和 $P(\text{分享})$ 直接从 `engagement_flags` 读出；$P(\text{2 秒内跳过})$ 的标签在 `watch_time_ms` 小于 $2{,}000$ 时为 $1$。这四个标签都从同一行已记录的数据算出，所以一次互动会同时产出携带全部四个标签的一条训练样本，供这个共享的多任务模型使用。

**双塔模型的负采样。** 上面的排序模型是在已经同时带有正、负隐式信号的数据行上训练的（一个低分只是意味着预测的互动概率低），但双塔检索模型需要一个显式的对比：一个正样本对是一行 $P(\text{观看} \ge 50\%)=1$ 的（用户、内容）记录，它的负样本*在批内*（in-batch）抽取——同一个训练批次里其他每一行的内容，都会充当这一行用户的负样本，不需要额外的数据加载开销，使用的损失是

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log
\frac{\exp(u_i \cdot v_i / \tau)}{\sum_{j=1}^{B} \exp(u_i \cdot v_j / \tau)},$$

其中批大小为 $B$ 的（用户、正样本内容）对，各自带有用户嵌入 $u_i$、内容嵌入 $v_i$，以及一个温度 $\tau$（下面的估算核对会在一个玩具批次上计算这个损失，并对照同一公式的逐项直接求值来验证）。批内负样本里热门内容出现的频率，会比在整个内容库上均匀采样要高，原因很简单：一个随机批次本来就会让热门内容过度出现——这是一个有用的性质，不是缺陷，因为它恰好制造出了检索在线上真正要胜过的那种难负样本（用户没有互动过、但看起来可信又受欢迎的内容）；针对由此产生的采样偏差，标准的做法是在做 softmax 之前，从每个负样本的 logit 里减去它被采样到的对数概率，这样一条内容就不会仅仅因为被采样得更频繁而受到惩罚。

**延迟反馈。** `like` 和 `share` 可能在产生它们的那次曝光之后几分钟才到达，所以标签连接（label join）如果做得太早，会低估最近这些互动里的正样本数量。这个设计在曝光发生之后、固定一个归因窗口（例如 $6$ 小时）再连接标签，连接时重新读取这行互动记录，而不是在写入时就连接；到这个窗口结束时仍然没有正信号的内容，就被当作永久负样本——用一小份、有界的噪声（来自那些极少数晚于窗口期才发生的动作）换取一次连接永远不会被无界地阻塞等待。

**重新训练节奏。** 排序模型每隔几个小时做一次微调重训，使用自上次更新以来连接好的互动数据，这样它能跟上快速变化的趋势，也能用上“数据与特征”一节里流式层刚聚合出来的特征。双塔模型每天从头完整重训一次，检索索引也用新算出的内容嵌入重建一次，因为一次完整重训能让每一条内容的嵌入都处在一套一致、可比的几何结构里，而只增量更新当天活跃内容做不到这一点；上面的新内容池和下面的冷启动平滑机制，负责覆盖一条视频从上传到下一次索引重建之间的这段空档。

### 深入话题

**(a) 选择目标并合并预测信号。** 排序模型是多任务的：共享的底层网络消费合并后候选的特征，若干个独立的输出头分别预测多个互动概率——$P(\text{观看} \ge 50\%)$、$P(\text{点赞})$、$P(\text{分享})$，以及作为显式负信号的 $P(\text{2 秒内跳过})$——都从同一批互动数据（见上文“训练”）联合训练，这让更稀疏的标签（点赞、分享）能受益于主要从更常见的观看时长标签里学到的表示，而不必各自用一个独立的小模型单独训练。把这四个预测合并成重排所使用的那一个分数，用的是一个加权和：

$$\text{分数} = w_1 P(\text{观看} \ge 50\%) + w_2 P(\text{点赞}) + w_3 P(\text{分享})
- w_4 P(\text{2 秒内跳过}),$$

权重是通过针对下面评估深入话题里的在线护栏指标和主指标做实验设定的，而不是在模型内部端到端学出来的——之所以选这个方案而不是直接针对一个复合标签训练单一模型，是因为单一模型所需要的标签，恰恰就是业务目标真正关心的长期满意度信号，而这个信号既稀疏（大多数会话都不会落到一个明确的满意度结果上），又缓慢（一个回访率标签需要好几天才能确定下来，远远晚于需要用它来学习的那次排序决策）。单纯优化原始互动——把这个加权和收缩成只剩 $P(\text{观看})$ 或一个原始的点击信号——被直接否决：它会奖励任何能抓住注意力的内容，而不管用户事后是否后悔，这一点多任务预测各自还能捕捉到（一次分享，比一条被完整看完、却再也不会回看的视频，是更强的真实价值信号），但单一的互动数字做不到这种区分。权重本身就是可调的旋钮：调高 $w_4$ 的权重，是在用一部分原始观看时长，换取更少的快速放弃，而真正验证这笔交换是否划算的，是在线护栏指标，不是离线指标。

**(b) 新用户与新视频的冷启动。** 一条刚上传的视频没有任何互动历史，所以它的滚动互动率特征（内容实体的 24 小时和 7 天聚合值）一开始是未定义的；与其把它们留成未定义或零——这两种做法排序模型都会误读为*已知不受欢迎*——不如把它们初始化为一个平滑过的先验值，再随着证据的积累向这条内容自己的实际观测率收缩：记观测到的曝光次数为 $n$、这条内容自己到目前为止的观测率为 $\hat r$，所用的特征就是 $\frac{n\hat r + \kappa \bar r}{n + \kappa}$，其中 $\bar r$ 是同类别的平均比率，常数 $\kappa$ 设定需要多少次真实曝光的证据才能压过这个先验——在 $n=0$ 时等于先验值，随着 $n$ 增大收敛到 $\hat r$。这还留下了曝光的问题：用于检索的内容嵌入完全由内容塔从上传时的元数据算出，所以一条全新的视频从上传的那一刻起，就在 ANN 索引里占有一个真实的位置，能靠普通的双塔相似度被检索到，完全不需要等待双塔模型本来就没有把它当作输入的互动数据。上面架构里的新内容池，是针对同一个问题的第二条、独立的路径：为最近上传的内容预留一小份固定配额，不管检索或排序本来会选出什么——这一点很重要，因为一条视频再好的纯内容嵌入，在排序模型的互动预测上，仍然要和那些已经有真实互动历史撑腰的内容竞争，而这些预测在还没有任何证据时，理应是悲观的。

一个新用户的会话内状态和滚动活动特征，在注册时同样是空的；引导流程里收集的信号（声明的话题偏好、语言地区、设备）先填补用户塔的输入，直到积累出足够的活动数据为止；个性化本身也是逐步引入的，而不是一开始就完全打开：候选生成和重排会把一个群体层面的热度信号，和用户嵌入自身的相似度分数混合起来，混合比例随着这个账号积累的互动次数增多而向个性化信号倾斜，这样一个新账号最初的几次会话，依靠的是普遍有效的东西，而不是一个还没积累到足够证据、尚不可靠的用户嵌入。

**(c) 反馈循环与曝光偏差。** 这个系统训练所用的互动日志，是由这个系统自己更早的决策产生的：一条从未展示过的内容不会有曝光，因此也永远不会有正标签，不管它本来有多好——单单在这份日志上训练，因此只会不断强化更早的模型本来就偏好的东西，也就是题目里定义的那种*曝光偏差*。有两种机制应对它。第一，*记录倾向分*（logging propensity）：每一次曝光都连同服务策略把这条内容放在这个位置的概率一起记录下来，而不只是记录它是否被展示过；这把训练变成了一种重要性加权学习——按记录下来的倾向分的倒数给一个样本加权——这样一条很少被展示、但展示时表现很好的内容，其权重会比它原始的曝光次数所暗示的更高，把训练样本重新校正回一份无偏日志本该有的样子。第二，*探索*（exploration）：一小部分固定比例的曝光（比如 $2\%$）不按预测分数选取，而是从合并后的候选池里均匀随机抽取，这是在有意用一点短期互动，去换取那些当前模型原本根本不会展示的内容的覆盖率——一条预测分数中等、但确有真实吸引力的内容，只有通过这条路径才有机会证明自己，因为一个纯按分数排序的位置，永远不会让它露面到足以被发现的程度。一个均匀随机位置的倾向分是精确且容易记录的（等于合格候选数量的倒数）；一个纯按分数排序的位置的倾向分则不是，因为它取决于那次请求里其他每一个候选的分数——这正是为什么即便按纸面计算，运行探索的代价是短期互动略微降低，这部分探索流量依然被保留下来的原因。

**(d) 离线与在线评估。** 两个离线指标，各对应一个阶段，因为一个阶段只能就它真正负责的事情来评估。*Recall@k* 单独评估候选生成：在一份留出的（用户、内容）对集合上——这些内容是该用户之后确实被观察到发生了互动的——统计这些对里，内容出现在检索阶段本会为该用户召回的前 $k$ 个候选里的比例；一个从未提出某条内容的候选生成器，根本不会给排序留下任何恢复它的机会，所以 recall@k 要在把责任归咎于排序之前先检查。*NDCG@k*（归一化折损累计增益）评估排序对拿到手的这些候选的排序结果，对照每个候选的一个分级相关度标签——例如跳过记 $0$，部分观看记 $1$，完整观看记 $2$，点赞记 $3$，分享记 $4$。记 $\mathrm{rel}_i$ 为模型排在第 $i$ 位（第 $1$ 位在最前）的那条内容的真实相关度：

$$\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i+1)}, \qquad
\mathrm{NDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k},$$

其中 $\mathrm{IDCG@}k$ 是在理想排序——把候选按真实相关度从高到低排序——上算出的同一个和，所以一个完美的排序得分为 $1$；当每个候选的相关度都是 $0$ 时（此时 $\mathrm{IDCG@}k = 0$），$\mathrm{NDCG@}k$ 被定义为 $0$。对于四个候选，模型给出的顺序对应的真实相关度是 $[3, 2, 3, 0]$，$\mathrm{DCG@4} \approx 12.393$；把这四个重新按 $[3,3,2,0]$ 排好得到的 $\mathrm{IDCG@4} \approx 12.917$，于是 $\mathrm{NDCG@4} \approx 0.9595$：非常接近 $1$，因为模型给出的顺序几乎、但不完全是理想顺序（两个相关度为 $3$ 的候选被调换了位置）。

在线上，一次上线是由一次随机化的 A/B 测试来判定的，而不是单靠离线指标，因为 recall@k 和 NDCG@k 都看不到业务目标真正关心的长期满意度。主指标是一个与之相关的短周期代理指标——比如实验组里在 $7$ 天内回访的比例，以及花在已完整观看、而非被跳过的内容上的总时长——并定期对照一个更慢、运行成本更高的长周期读数加以校验，而不是被盲目信任，因为每一次候选改动都要等长周期数字出来，会让迭代速度慢到无法接受。护栏指标和主指标一起运行，即便主指标获胜，也可能因为护栏指标而拦下一次上线：信息流请求的 p99 延迟必须保持在 $200$ 毫秒的目标以内；投诉或隐藏内容的比率不能恶化；曝光和观看时长里，流向热度门槛以下创作者的份额不能萎缩——这正是下面公平性追问的在线对应版本。

### 追问

- **用多模态编码器理解内容。** 把内容塔里手工构造的元数据特征，换成一个预训练编码器对视频画面、音频以及任何屏幕文字或字幕文本编码得到的嵌入，能在基于互动的相似度之外，再提供一种真正基于内容本身的相似度，这既能改善新内容的冷启动（从上传那一刻起就有一个真正基于内容的嵌入，不只是元数据），也能改善重排的多样性（两条内容即使还没有共同的互动模式，也能被识别为相似）；代价是一条更重的、按上传逐条编码的流水线，必须跟上每天 $5{,}000{,}000$ 条上传的速度，而不只是查几个元数据字段。
- **大规模下的实时特征。** 不是每个特征都能等流式层 $1$ 小时的时限：本次会话里已经展示过的内容，必须在请求内部本身就核对，所以它们存放在一个按会话键控的低延迟存储里，与更慢的流式聚合管线分开——把两者混在一起，要么会把每次请求都拖慢到流式层的延迟，要么会让会话内的核对滞后长达一小时，悄悄地重复展示内容。
- **对新创作者的公平性。** 训练日志和排序目标都会不利于一个小创作者——互动数据越少，预测就越差，带来的曝光就越少，数据就更少——所以保护新创作者需要一个明确的选择，而不是现有机制附带的效果：为粉丝数或播放量低于某个门槛的创作者，在重排里设一个曝光配额，这和反馈循环深入话题里的探索位不是一回事（那些位置的存在是为了改进模型，不是为了保证某个创作者群体的曝光），并且要把它作为每次在线测试里自己的一项护栏指标来跟踪，而不是等有人碰巧注意到问题才去检查。
- **特征上的隐私约束。** 一些看似有用的特征被原则性地排除在外，而不是因为它们缺乏预测价值：一个用户的个人观看历史，永远不会成为另一个用户请求里的特征，能用的只有聚合过、匿名化的统计量；互动日志本身也不携带任何自由文本或精确位置字段，这样一份被攻破、或者访问权限设置过宽的日志副本，暴露的只会是聚合层面的行为模式，而不会是某个具体的人某一次具体的观看过程。

<details>
<summary>估算核对（可运行）</summary>

```python
import math
import random

import numpy as np

# ---- feed-request rate ----
dau = 200_000_000
opens_per_day = 10
items_per_request = 20
peak_factor = 3

requests_per_day = dau * opens_per_day
assert requests_per_day == 2_000_000_000

avg_qps = requests_per_day / 86_400
assert round(avg_qps) == 23_148

peak_qps = avg_qps * peak_factor
assert round(peak_qps) == 69_444

# ---- items returned vs. items scored ----
items_returned_avg = avg_qps * items_per_request
items_returned_peak = peak_qps * items_per_request
assert round(items_returned_avg) == 462_963
assert round(items_returned_peak) == 1_388_889

candidates_per_request = 500
assert candidates_per_request == 25 * items_per_request

items_scored_avg = avg_qps * candidates_per_request
items_scored_peak = peak_qps * candidates_per_request
assert round(items_scored_avg) == 11_574_074
assert round(items_scored_peak) == 34_722_222

# ---- ranking fleet ----
batch_time_s = 0.020
throughput_per_replica = candidates_per_request / batch_time_s
assert throughput_per_replica == 25_000.0

raw_replicas = items_scored_peak / throughput_per_replica
assert round(raw_replicas) == 1_389
headroom = 0.2
n_rank_replicas = math.ceil(raw_replicas * (1 + headroom))
assert n_rank_replicas == 1_667

# ---- candidate index memory ----
catalogue = 500_000_000
emb_dim = 128
bytes_per_dim = 1  # int8
index_bytes = catalogue * emb_dim * bytes_per_dim
index_gb = index_bytes / 1e9
assert index_gb == 64.0

shard_gb = 8
n_shards = index_gb / shard_gb
assert n_shards == 8.0
replication = 3
n_index_machines = int(n_shards * replication)
assert n_index_machines == 24

new_items_per_day = 5_000_000
assert new_items_per_day / catalogue == 0.01
new_emb_gb_per_day = new_items_per_day * emb_dim * bytes_per_dim / 1e9
assert round(new_emb_gb_per_day, 2) == 0.64

# ---- event-log volume ----
ids_bytes = 8 * 5              # event_id, request_id, user_id, item_id, served_at
outcome_bytes = 1 + 4 + 1 + 4   # position, watch_time_ms, engagement_flags, ranking_score
row_bytes = ids_bytes + outcome_bytes
assert row_bytes == 50

impressions_per_day = dau * opens_per_day * items_per_request
assert impressions_per_day == 40_000_000_000

daily_event_bytes = impressions_per_day * row_bytes
daily_event_tb = daily_event_bytes / 1e12
assert daily_event_tb == 2.0

hot_days = 30
hot_tb = daily_event_tb * hot_days
assert hot_tb == 60.0

print("all requirements-and-scale numbers check out")


# ---- NDCG@k, checked against an independent brute-force computation from the definition ----
def ndcg_at_k(relevance, k):
    """relevance: true relevance of each candidate, already in the model's ranked order (best first)."""
    relevance = np.asarray(relevance[:k], dtype=float)
    discounts = np.log2(np.arange(2, len(relevance) + 2))   # NOTE: rank 1 -> log2(2), never log2(1) == 0
    dcg = float(np.sum((2 ** relevance - 1) / discounts))
    ideal = np.sort(relevance)[::-1]
    idcg = float(np.sum((2 ** ideal - 1) / discounts))
    return dcg / idcg if idcg > 0 else 0.0


example_rel = [3, 2, 3, 0]                 # the worked example from the text
example_ndcg = ndcg_at_k(example_rel, 4)
assert round(example_ndcg, 4) == 0.9595


def brute_ndcg(relevance, k):
    """The same definition, recomputed independently in plain Python: no NumPy, no shared helper."""
    relevance = list(relevance[:k])
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


rng_py = random.Random(0)
for _ in range(2_000):
    n = rng_py.randint(1, 12)
    k = rng_py.randint(1, n)
    rel = [rng_py.randint(0, 4) for _ in range(n)]
    assert abs(ndcg_at_k(rel, k) - brute_ndcg(rel, k)) < 1e-9

assert ndcg_at_k([0, 0, 0], 3) == 0.0      # every candidate irrelevant: IDCG@k == 0 by definition

print("NDCG@k matches an independent brute-force computation on 2,000 random relevance lists")


# ---- recall@k of a toy two-tower retrieval index against exhaustive search ----
def build_ivf_index(item_vectors, n_clusters, seed, n_iters=5):
    """A minimal inverted-file index: a few iterations of Lloyd's algorithm assign each item to the
    nearest of n_clusters centroids, used to prune the search at query time."""
    rng = np.random.default_rng(seed)
    n_items = item_vectors.shape[0]
    centroids = item_vectors[rng.choice(n_items, size=n_clusters, replace=False)].copy()
    assignment = np.zeros(n_items, dtype=int)
    for _ in range(n_iters):
        dists = ((item_vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        assignment = dists.argmin(axis=1)
        for c in range(n_clusters):
            members = item_vectors[assignment == c]
            if len(members) > 0:            # NOTE: an empty cluster keeps its old centroid, never NaNs out
                centroids[c] = members.mean(axis=0)
    dists = ((item_vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return centroids, dists.argmin(axis=1)


def ivf_topk(query, item_vectors, centroids, assignment, nprobe, k):
    """Approximate top-k by dot product, searching only the nprobe clusters whose centroid is closest
    to the query. Clustering uses squared distance; retrieval scores by dot product, matching the
    two-tower model's own training objective -- the two need not be the same metric."""
    probe = np.argsort(-(centroids @ query))[:nprobe]
    idx = np.nonzero(np.isin(assignment, probe))[0]
    order = np.argsort(-(item_vectors[idx] @ query))[:k]
    return idx[order]


rng = np.random.default_rng(0)
n_items, dim, n_clusters, k = 2_000, 32, 25, 10
item_vectors = rng.normal(size=(n_items, dim))
centroids, assignment = build_ivf_index(item_vectors, n_clusters, seed=1)
assert len(assignment) == n_items and set(assignment.tolist()) <= set(range(n_clusters))

n_queries = 60
queries = rng.normal(size=(n_queries, dim))


def recall_at(nprobe):
    total = 0.0
    for q in queries:
        exact = set(np.argsort(-(item_vectors @ q))[:k].tolist())    # brute force: never calls ivf_topk
        approx = set(ivf_topk(q, item_vectors, centroids, assignment, nprobe, k).tolist())
        total += len(exact & approx) / k
    return total / n_queries


recall_1 = recall_at(1)
recall_5 = recall_at(5)
recall_full = recall_at(n_clusters)

assert recall_full == 1.0                       # every cluster searched == exhaustive search, exactly
assert 0.10 < recall_1 < 0.30
assert 0.40 < recall_5 < 0.70
assert recall_1 < recall_5 < recall_full         # probing more clusters never recovers fewer true neighbours

print(f"recall@{k}: nprobe=1 -> {recall_1:.3f}, nprobe=5 -> {recall_5:.3f}, nprobe=full -> {recall_full:.3f}")
print("toy retrieval index recall matches exhaustive search once every cluster is probed")


# ---- in-batch softmax loss for the two-tower model, checked against a direct formula ----
def in_batch_softmax_loss(user_emb, item_emb, temperature):
    """In-batch negatives: for row i, item_emb[i] is the positive and every item_emb[j] in the same
    batch is a negative for user_emb[i]."""
    logits = (user_emb @ item_emb.T) / temperature             # logits[i, j] = u_i . v_j / tau
    # NOTE: subtract the row max before exponentiating -- without it a large logit overflows exp()
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(np.diag(log_probs)))


def direct_in_batch_loss(user_emb, item_emb, temperature):
    """The same formula, evaluated term by term with no matrix operations and no shared helper."""
    B = user_emb.shape[0]
    total = 0.0
    for i in range(B):
        num = math.exp(float(np.dot(user_emb[i], item_emb[i])) / temperature)
        den = sum(math.exp(float(np.dot(user_emb[i], item_emb[j])) / temperature) for j in range(B))
        total += -math.log(num / den)
    return total / B


rng = np.random.default_rng(3)
for _ in range(200):
    B, dim = int(rng.integers(2, 9)), int(rng.integers(2, 6))
    user_emb = rng.normal(size=(B, dim))
    item_emb = rng.normal(size=(B, dim))
    temperature = float(rng.uniform(0.05, 1.0))
    assert abs(in_batch_softmax_loss(user_emb, item_emb, temperature)
               - direct_in_batch_loss(user_emb, item_emb, temperature)) < 1e-6

print("in-batch softmax loss matches a direct, term-by-term evaluation of the same formula on 200 random batches")

print("all checks passed")
```

</details>

</details>
