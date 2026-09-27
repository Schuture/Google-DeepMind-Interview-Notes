# 机器学习系统设计：机器人操作策略的数据飞轮

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 机器学习系统设计 | ★★☆☆☆ | 困难 | RE · RS · MLE | robot-learning, data-collection, dataset-curation, vision-language-action, evaluation-statistics, deployment-safety | 45–60 分钟 | 终面（Final） |
<!-- meta:end -->

## 题目

*视觉-语言-动作*（vision-language-action，VLA）策略是这样一个模型：它在控制回路的每个时间步接收摄像头图像、机器人当前的关节状态，以及一句自然语言的任务指令，输出机器人手臂的下一个动作。设计这个策略周围的系统：一支机器人手臂机队用这个策略执行家庭场景里的操作任务——拿起一件物品、打开一个容器、放置一件物品——每个任务都由一句简短的指令给出，例如“把杯子放进水槽”。要设计的系统从机队收集数据、对数据做整理、训练策略的新版本、在新版本被信任之前对它做评估，并把它重新部署回机队。

*回合*（episode）是对单个任务的一次连续尝试，从一条固定的指令和一个初始场景布局开始，到达到某个结束条件为止——完成任务、达到时间上限，或者操作员将其中止——之后机器人复位，准备下一次尝试；它是数据收集的基本单位，持续约 60 秒。一个回合要么是*遥操作演示*（teleoperated demonstration），即人类操作员通过遥操作接口实时给出每一个动作（记录下来的动作就是操作员自己下达的指令），要么是*自主推演*（autonomous rollout），即动作来自一个闭环运行、没有人类直接操控的策略——尽管仍有一名人类安全操作员在旁监看，并可以随时叫停或接管。*成功检测器*（success detector）是一个自动分类器——这里是一个视觉-语言模型，输入是某个回合最后若干帧画面和它的任务指令——用来预测任务是否完成；正是它让给数量庞大的自主推演打标签、而不需要人工逐一复核成为可能。*介入*（intervention）是安全操作员在自主推演进行到一半时接管控制权的行为，原因是机器人即将违反某项安全限制，或者明显卡住、正在失败；*介入片段*（intervention segment）是回合中从接管那一刻到控制权交还（或者回合结束）之间的那一段。

策略不需要一次只产出一个动作：一次前向计算可以输出一个*动作块*（action chunk），即一小段连续的未来动作序列，机器人的底层控制回路随后逐个消费这些动作，同时下一个动作块正在被计算。

为下面的规模设计：

- 200 台机器人，每台每天运行 8 小时。
- 每台机器人有 3 个分辨率为 640×480、帧率为 15 帧/秒的摄像头，画面以 JPEG 格式存储，每帧约 40 KB。
- 每台机器人还以 50 Hz 记录关节状态和下达的动作；这两路低维数据合计约 1 KB/秒。
- 机器人 30% 的时间用于遥操作演示，70% 的时间用于自主推演。
- 一个回合持续约 60 秒。
- 评估套件是一组固定的、留出的 50 个任务。
- 策略运行在机器人的控制回路上，频率为 10 Hz，即每 100 毫秒消费一个动作；计算一个动作块——从触发它的那一帧画面到这个动作块准备就绪——不论其中包含多少个动作，都必须耗时不超过 100 毫秒。
- 人类操作员可以在任何时刻叫停任何一台机器人，不受策略或系统其余部分正在做什么的影响。

范围内：端到端的数据飞轮——接入、标注、整理、训练数据混合、训练、评估，以及分阶段重新部署回机队——针对单一、固定的机器人本体（embodiment：同一种机器人型号，机队内完全一致）；判断候选策略是否真的变好背后的统计方法；把介入转化为训练数据；以及服务这个策略时的安全与延迟约束。范围外：策略的模型架构与训练目标，除了已经给出的输入输出接口约定之外；下达动作之后负责执行它的底层电机控制器；遥操作硬件本身。

要产出：

1. 需求与规模估算：每机器人-天和每机队-天产生的摄像头数据量、低维数据量（要计算出来，而不是仅凭断言说它很小）、每天产生的回合数，以及一年的存储量。
2. 回合的数据模型及其存储布局：存储了什么、存储在哪里，以及为了支持整理作业的查询，建立了哪些索引。
3. 数据与策略流经的流水线——接入、标注、整理、训练数据混合、训练、评估与金丝雀发布（canary deployment）——以图的形式给出，并沿着它走一遍某个回合的完整路径。
4. 深入话题：(a) 评估统计——一个实测成功率的置信区间、需要多少评估试验才能把 60% 的成功率和 70% 的成功率区分开、如何在这个比较中控制场景重置和先后顺序的影响，以及仿真评估与真实评估的取舍；(b) 闭合这个循环——把一次介入转化为训练数据，并决定接下来该收集什么；(c) 安全与上线——动作限制、回退行为，以及在机队上的分阶段部署；(d) 延迟——机器人本地推理与服务器推理的对比，以及动作分块能带来什么。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手设计之前有两点值得先确认：什么算作成功——这里假设按回合二元判定，依据一份针对每个任务固定的简短评判标准，一次接近完成但未完成的尝试没有部分分——以及机队是否是同构的。这份设计假设全部 200 台机器人是同一种机器人型号，共享同一款手臂、同一套摄像头装置和同一种关节构型，所以一个策略——一套权重——就能服务整个机队，不需要按机器人分支处理；机器人型号不止一种的机队留给追问部分。

### 需求与规模

**摄像头数据。** 每台机器人的三个摄像头合计产生

$$3 \times 15 \times 40\text{ KB} = 1{,}800\text{ KB/秒} = 1.8\text{ MB/秒。}$$

按每天运行 8 小时算（$8\times3{,}600=28{,}800$ 秒）：

$$1.8\text{ MB/秒} \times 28{,}800\text{ 秒} = 51{,}840\text{ MB} \approx 51.84\text{ GB/机器人-天，}$$

整个机队则是 $51.84\times200=10{,}368$ GB，约 **10.4 TB/机队-天**。

**低维数据。** 关节状态和下达的动作合计以约 1 KB/秒的速率记录（题目给定），所以一个机器人-天产生
$1\text{ KB/秒}\times28{,}800\text{ 秒}=28{,}800$ KB $\approx28.8$ MB，整个机队则是 $28.8\times200=5{,}760$ MB
$=5.76$ GB/天——恰好是摄像头数据流的 $1{,}800/1=1{,}800$ 分之一，这是把“可忽略”计算出来、而不是想当然认定的结果。

**每天的回合数。** 按每个回合约 60 秒、每天运行 28,800 秒算，一台机器人每天完成 $28{,}800/60=480$ 个回合；整个
机队每天完成 $480\times200=96{,}000$ 个回合，按 30%/70% 拆分为 $28{,}800$ 个遥操作演示和 $67{,}200$ 个自主推演。

**一年的存储量。** 机队每天的总量是 $10{,}368+5.76=10{,}373.76$ GB；365 天下来，

$$10{,}373.76\text{ GB}\times365=3{,}786{,}422.4\text{ GB}\approx3.79\text{ PB/年，}$$

几乎全部来自摄像头画面。这个数字是永久保留全部原始传感器数据的结果，而这并不是真正合适的默认做法：下面的整理
环节会永久保留每个回合的元数据以及每一次标注决定的结果，但把每一帧原始画面都保留到超出它可能被重新标注或审计
的时间窗口之后，这个理由就没那么站得住脚了——一个合理的默认做法是设一个滚动的热存储层，比如保留 90 天的原始
视频，超过这个期限、且所属回合已经走完整理流程的原始帧则迁移到更便宜的冷存储或直接丢弃；这份设计不对这个具体
天数做出承诺。

### 数据模型与存储

**Task（任务）** —— `task_id`、`instruction_template`（策略收到的指令，例如“把杯子放进水槽”）、`is_eval_task`
（对留作评估套件的 50 个任务为真）、`success_rubric`（成功检测器的提示词和人工抽查者看到的同一份书面判定标准）。

**Episode（回合）** —— `episode_id`、`robot_id`、`task_id`、`episode_type`（`teleop | autonomous`）、
`policy_version`（`autonomous` 时有值，`teleop` 时为空）、`scene_config_id`（这次尝试用的是哪一种随机初始物体
布局——评估时要控制场景重置的影响，见深入话题 (a)，需要用到它）、`started_at`、`duration_s`、`outcome`
（`success | failure | unknown`）、`outcome_source`（`detector | human`）、`detector_confidence`、
`had_intervention`、`frame_ref`、`state_ref`。最后两个字段是指向下面 blob 存储的指针，绝不是画面或关节状态轨迹
本身。建立索引的字段组合是 `(task_id, episode_type, outcome, policy_version, had_intervention, started_at)`——
这正是整理作业会用来过滤的组合：策略版本 Y 下任务 X 的所有失败自主推演、过去一周里所有发生过介入的回合，等等，
全部由一个面向行的元数据存储直接回答，完全不用碰 blob 存储。

**Intervention（介入）** —— 一个回合内每次介入对应一行：`episode_id`、`operator_id`、`start_t`、`end_t`（回合内
的偏移量）、`reason_code`（`near_limit | stuck | task_error`）、`used_for_training`（由整理环节，见深入话题
(b)，决定这次纠正是否归入训练集时设置）。

摄像头画面写入对象存储，键为 `{episode_id}/{camera_id}/{frame_index}.jpg`；关节/动作轨迹是每个回合一个文件，
`{episode_id}/state.bin`，按它原生的 50 Hz 频率记录。机器人在回合进行时把数据缓冲在本地，回合一结束就上传这些
blob，只有在那之后才写入 `Episode` 这一行——先写 blob，后写元数据行——这样即便两步之间发生崩溃，最坏情况也只
是留下一个暂时没有任何东西指向它的孤立 blob，绝不会出现一行元数据指向根本没写过的数据。

接入与整理这两条路径，都是这份数据模型之上的一层简洁接口：机器人在一个回合结束、其 blob 已经持久化写入之后，
调用一次 `POST /v1/episodes`，`{robot_id, task_id, episode_type, scene_config_id, frame_ref, state_ref, ...}`
→ `{episode_id}`；整理作业调用 `POST /v1/episodes:query`，带上一个针对上面索引字段的过滤条件，分页取回匹配的
`episode_id`；标注工作者则调用 `PATCH /v1/episodes/{episode_id}`，在检测器——以及在被抽中时，人工——完成打分
之后，写入 `outcome`、`outcome_source` 以及任何 `Intervention` 行。

### 流水线

```text
    +---------------------------------------------+
    |        机队：200 台机器人运行策略 vN        |
    +---------------------------------------------+
                           |
                           v
          回合：30% 遥操作演示，70% 自主推演
                           |
                           v
    +---------------------------------------------+
    |              接入（Ingestion）              |
    |          先写 blob，再写一行元数据          |
    +---------------------------------------------+
                           |
                           v
    +---------------------------------------------+
    |              标注（Labelling）              |
    |          VLM 成功检测器 + 人工抽查          |
    |               + 介入片段打标                |
    +---------------------------------------------+
                           |
                           v
    +---------------------------------------------+
    |              整理（Curation）               |
    |          过滤 -> 去重 -> 任务配比           |
    +---------------------------------------------+
                           |
                           v
               += 网络规模视觉-语言数据
                           |
                           v
    +---------------------------------------------+
    |                训练数据混合                 |
    +---------------------------------------------+
                           |
                           v
    +---------------------------------------------+
    |          训练  -->  候选策略 vN+1           |
    +---------------------------------------------+
                           |
                           v
    +---------------------------------------------+
    |             评估（Evaluation）              |
    |     先做仿真扫描，再做真实机队测试批次      |
    +---------------------------------------------+
                           |
                           v
    +---------------------------------------------+
    |       金丝雀发布：在机队上分阶段上线        |
    +---------------------------------------------+
                           |
                           v
           vN+1 成为机队的策略；它自己产生的
              回合又会流回上面的接入环节
```

一个遥操作演示走的是较短的路径：被接入，出于一致性的考虑交给成功检测器打分（人类操作员自己完成的演示，按构造
就视为成功，除非遥操作接口本身记录了一次中止，那样会直接把 `outcome` 设为 `failure`），然后交给整理环节。

一次触发了介入的自主推演走的是更完整的路径。它和任何其他回合一样被接入，成功检测器依据任务指令给它的最后若干
帧打分；因为它包含一次介入，不论检测器给出什么结果，整理环节都会把它拆成两半区别对待。接管之前的那一段——策
略当时正要做错事的那一段——会被排除在正向训练集之外，因为它演示的是错误做法，而不是被纠正之后的轨迹，但会被
保留并打上标签，供深入话题 (b) 的失败分析使用。接管之后的那一段是操作员自己给出的纠正，一旦人工抽查确认了它
的 `reason_code` 并设置了 `used_for_training`，它就会和一段简短的遥操作演示一样被加入训练集——因为它本质上就
是：一个人类在策略当前实际到达的那个状态上，给出了正确的动作。两段被保留下来的片段之后都会照常走完整理环节剩
下的步骤，和其他任何回合一样。

给大多数回合打分的是成功检测器，而不是人工——在每天 67,200 次自主推演的规模下，这是必须的——只对其中固定*比
例*的一部分（更倾向于检测器置信度低的、以及发生过介入的，因为这些回合的标签准不准最要紧）抽样做人工复核，而
不是固定数量，这样人工复核的预算才能随机队一起扩大，而不是被机队越甩越远。

整理环节对已标注的数据流做三轮处理。过滤这一步，把检测器标记为失败的自主推演从正向训练集里剔除——和接管前的
片段一样，一次失败的推演会被保留用于分析，而不是删除，只是被排除在教策略如何成功之外——并且剔除接口标记为中
止的遥操作演示。去重这一步，在每个 `(task_id, scene_config_id)` 桶内，按轨迹和摄像头画面的嵌入对回合做聚类，
限制同一种做法的近似重复最多保留多少份；这里必须是近似相似度，而不是精确匹配的哈希，因为真实世界的传感器噪声
让两个回合完全相同这件事基本不可能发生。任务配比这一步，随后在 50 个任务之间重新采样——再叠加深入话题 (b) 里
的优先级权重，按当前策略在每个任务上已经做得多好来配比——这样一个训练轮次就不会只是简单照搬某一周里每个任务
恰好被尝试了多少次。

训练时，每个批次的数据来自两个来源：经过整理的机器人回合数据，以及固定比例的、完全不带机器人动作标签的网络规
模图文数据。每一条机器人样本都监督策略去复现记录下来的动作——对遥操作样本来说是操作员自己的动作，对被保留下
来的介入样本来说是操作员的纠正动作——条件是那个时间步的图像、关节状态和指令；网络规模样本则只监督共享的视觉-
语言部分。混合比例如果太偏向机器人数据，会在这支机队恰好拥有的这 50 个任务、这些摄像头、这种光照条件上过拟
合；如果太偏向网络数据，策略保住了宽泛的视觉-语言能力，却削弱了真正教会它行动的那一个信号。

评估环节会先让新训练出来的候选版本跑一遍这 50 个任务组成的评估套件，然后才会把它托付给整个机队——它到底需要
多大代价，由深入话题 (a) 给出估算。金丝雀发布随后把通过评估的版本，以逐步增大的比例分阶段推上机队，门槛不仅
是离线评估的分数，还要看机队上的实时指标——见深入话题 (c)。

### 深入话题

**(a) 评估统计。** 一次包含 $n$ 次试验、$k$ 次成功的评估，给出点估计 $\hat p=k/n$，但单独一个点估计掩盖了 $n$
次真实机器人试验实际上能确定到什么程度。熟悉的 Wald 区间（Wald interval）$\hat p\pm z\sqrt{\hat p(1-\hat
p)/n}$，来自把未知的 $p$ 在方差里替换成 $\hat p$ 之后的一个正态近似；Wilson 置信区间（Wilson score interval）
则不做这个替换，直接对检验统计量 $(\hat p-p)/\sqrt{p(1-p)/n}$ 求逆，通过解 $(\hat p-p)^2=z^2\,p(1-p)/n$ 这个
关于 $p$ 的二次方程得到——它的两个根是

$$p_{\pm}=\frac{\hat p+\dfrac{z^2}{2n}\pm z\sqrt{\dfrac{\hat p(1-\hat p)}{n}+\dfrac{z^2}{4n^2}}}{1+\dfrac{z^2}{n}}。$$

对 $n=60$ 次试验、$k=42$ 次成功（$\hat p=0.700$），在 95% 置信水平（$z\approx1.960$）下，这给出
$[0.575,\ 0.801]$（下面会核对），中心在 $0.688$——相对 $\hat p$ 本身向 $0.5$ 收缩，这正是它在 $\hat p$ 趋近 0
或 1、或者 $n$ 较小时仍能保持接近名义置信水平的原因，而这两种情形在真实机器人评估里都很常见，因为每一次试验
都要付出机器人的时间成本。同一份数据的 Wald 区间 $[0.584,\ 0.816]$，整体偏向更高的取值，在这里也是两者中更宽
的那一个。

正面比较两个策略版本，还需要回答第二个问题：每个版本需要多少次试验，才能把一个真实的差异和噪声区分开？把值
得可靠检测出来的最小差异定为 60% 对 70%——更小的真实差异会需要更多试验；这是这份设计给定的目标分辨率，不是一
个放之四海而皆准的常数。在 $H_0: p_1=p_2$ 下，用合并估计 $\bar p=(p_1+p_2)/2$ 给出检验统计量的方差，检验在
$|z|>z_{\alpha/2}$ 时以水平 $\alpha$ 拒绝原假设；在 $H_1$ 下，真实的 $p_1$、$p_2$ 存在差异，同一个统计量的方差
不同，要分别用 $p_1$、$p_2$ 来构造。令 $P(\text{拒绝}\mid H_1)=\text{检验效能}$ 并解出公共的每组样本量 $n$，
就得到标准的双比例样本量公式：

$$n=\frac{\left(z_{\alpha/2}\sqrt{2\bar p(1-\bar p)}+z_{\text{power}}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\right)^2}{(p_1-p_2)^2}。$$

在 $\alpha=0.05$ 双侧检验（$z_{\alpha/2}\approx1.960$）、80% 检验效能（$z_{\text{power}}\approx0.842$）下，$p_1=0.60$，$p_2=0.70$：
$n\approx355.9$，也就是**每组 356 次试验**——按每次约 60 秒算，712 次真实回合，接近 12 小时不间断的机器人时
间，而这仅仅是针对一个任务的一次正面比较。恰好在这个 $n$ 上做的一次蒙特卡洛仿真（下面会核对），在 $H_1$ 下测
得拒绝率为 0.7999，在仿真自身的误差范围内和公式给出的 80% 目标吻合。

这个比较只有在两组被比较的试验之间除了策略版本之外没有任何其他差异时才成立。把它做成分块的配对试验：对许多
次相互独立的场景重置，每次都让两个版本背靠背各跑一次再重置，并且交替让哪个版本先跑。这样就能把场景与场景之
间、以及不同时段之间的干扰方差——光照的漂移、某个物体逐渐产生的磨损、一个已经偏移了几毫米的摄像头——从比较
中剔除出去，而不是任由它放大一个表面上的差异，或者掩盖一个真实的差异；交替安排哪个版本先跑，也能抵消先手优
势（更新鲜的物体、更热身的电机）被错误地记到某个恰好先跑的版本头上。配对设计要达到同样的检验效能，所需试验
数不会比上面未配对的数字更多，通常还会更少，因为它去掉了未配对检验仍要为之付出代价的一部分方差来源——具体
能省多少，取决于场景本身对结果的影响有多强，这里不再进一步量化。

在评估套件的全部 50 个任务上都跑一遍上面这套完整的配对比较，需要 $50\times356\times2=35{,}600$ 次真实试验，
约 593 小时——接近 25 天不间断的机器人时间——这还没算上试验之间重置场景的时间：对每一个候选版本都付出这样的
代价，显然不是常规做法。真正用来吸收这份代价的是仿真：场景可以瞬间重置，环境可以并行运行，一个候选版本能以
低得多的代价在全部 50 个任务上接受筛查，在任何真实机器人介入之前就把明显的退化过滤掉——前提是仿真器自身和真
实硬件之间的差距（渲染保真度、接触与摩擦动力学、传感器噪声）本身要定期用一批真实试验去核对，而不是被盲目信
任。每组 356 次的真实试验预算，之后就花在仿真筛查标记为势均力敌、或者被列为全机队推广候选的少数任务上，而不
是照例花在全部 50 个任务上。

**(b) 闭合这个循环。** 普通的行为克隆（behaviour cloning）在专家选择去访问的状态上训练——也就是遥操作演示—
—但部署之后的策略会去到它自己的状态，其中一些是任何演示都没有覆盖过的；一个早期的小错误就可能把它带到训练
数据从未示范过该如何离开的状态，误差由此累积而不是被纠正。一次介入的纠正片段，按其构造，正好是在当前策略实
际到达、且即将处理错误的那个状态上给出的标签——这正是 DAgger（Dataset Aggregation，数据集聚合；Ross、Gordon
与 Bagnell，2011 年）背后的想法：把专家在学习者自己推演状态上给出的纠正汇聚进训练集，而不是只在专家最初的演
示分布上训练，这样每一轮都能缩小策略实际访问的状态和它已经被示范过如何应对的状态之间的差距。在这份设计里，
这种汇聚不是一个独立的离线步骤；它正是整理环节（见上面的流水线）对一次介入片段的 `used_for_training` 判定在
做的事情，持续地、在整个机队的规模上进行。

不是每个任务都需要同等的关注。一个策略已经做得很好的任务，再来一条演示收获有限；一个经常失败的任务，恰恰是
下一轮里每一条边际回合最能帮上忙的地方。把一份收集资源——一条遥操作演示，或者把一次自主推演派去尝试哪个任务
——分配给任务 $i$ 的概率，按 $w_i\propto(1-\hat p_i)+\varepsilon$ 加权，也就是它的估计失败率加上一个很小的
下限 $\varepsilon$。这个下限和权重本身同样重要：没有它，一个已经被策略打磨到接近 100% 成功率的任务会彻底停
止被采样，而这个任务之后如果发生退化——原因完全在别处，比如某次混合比例调整悄悄伤害了一项不相关的技能——就
会一直发现不了，直到它在评估里冒出来，而不是被日常收集顺带发现。下面核对的一个小型仿真显示，这个机制会把一
份固定的收集预算，更多地分给一开始更难的任务，更少地分给一开始更容易的任务，相比在所有任务上均匀分配；而且
不论怎么分，都不会让任何一个任务的分配比例降到零；跑足够多轮之后，在总共收集的回合数相同的前提下，它留下的
最差任务的状态也好于均匀分配。

因为一个任务失败得多就多往它上面派任务，这确实意味着这个任务也正是最可能很快需要一次介入的任务——这不是矛
盾，因为这正是它被修好的方式，但这也是本深入话题里的权重机制和下一个深入话题里的安全层是同一套设计、而不是
两套互不相干的设计的原因：优先级收集决定机队把自主尝试花在哪里，而动作限制与回退（深入话题 (c)）则限定了任
何一次这样的尝试在被人类或机器人自身的限制介入之前，最坏能坏到什么程度。

**(c) 安全与上线。** 动作限制是策略原始输出和机器人电机控制器之间一层硬性的、与策略无关的屏障：每一个下达的
关节速度、力矩和位置指令都被钳制在固定的物理边界内，末端执行器被限制在一个固定的笛卡尔工作空间内，指令从一
个控制周期到下一个的变化幅度也受速率限制。这些都不是学出来的，策略也无法覆盖它们，所以一个策略缺陷或者一次
分布外的输入，退化成的是被钳制、有些卡顿的动作，而不是不受约束的动作。

回退机制覆盖的是策略没能给出结果的情形，而不只是给出了错误结果的情形：一个动作块错过了它 100 毫秒的截止时间
（深入话题 (d)）、一个动作块没能通过基本的有效性检查（出现 `NaN`，或者取值大大超出钳制范围、远超一个正常边
界情形会产生的偏差），以及——这是题目本身的要求，不是一个假设情形——操作员直接叫停机器人。三种情形的回退方
式相同：冻结在最后一次下达的位置，而不是继续执行一个过时或者异常的动作；操作员的叫停信号尤其会走一条完全不
经过策略服务路径的通道直达电机控制器，这样一个变慢或者崩溃的推理服务——不论是本地还是远程——都不可能延误它。

一个新近通过评估的版本，会以逐步增大的比例分阶段推上机队，而不是一次性全量上线：比如 1%（2 台机器人）、10%
（20 台）、50%（100 台），然后是 100%（200 台），每个阶段都运行足够长的时间，在它自己的任务上积累有意义数量
的试验——这正是深入话题 (a) 里的试验数推理，只是这次用在机队的实时数据上，而不是一批专门的评估数据——然后才
进入下一阶段；每个阶段的推进门槛，除了这个版本已经通过的离线评估分数之外，还要看介入率和任何安全限制触发率
是否维持在上一版本的基线水平之内。评估套件那 50 个固定场景没有恰好覆盖到的一次退化，仍然可能在机队更丰富多
样的真实房间和光照条件下暴露出来，而分阶段上线能让这样的退化只在 2 台机器人的暴露面上被发现，而不是 200 台。
一个出现退化的阶段只会在那一个切片上回滚，而不会让整个机队的推进暂停下来等待排查。

**(d) 延迟：机器人本地推理与服务器推理，以及动作分块。** 控制回路每 100 毫秒消费一个动作，绝不能出现无动作
可用的情况。计算一个动作块——不论它的长度 $H$ 是多少——按题目所给的约束，预算都是完整的 100 毫秒，与 $H$ 无
关；随 $H$ 变化的不是这个单次调用的上限，而是到底多久才需要发起一次调用，因为一个包含 $H$ 个动作的动作块能
给控制回路留出 $H\times100$ 毫秒的缓冲余量，足够撑到下一次调用之前——这把调用频率从 10 Hz 降到了 $10/H$ Hz，
在 $H=8$ 时是 1.25 Hz。

机器人本地推理把这份预算完全花在计算上：一颗嵌入式加速器的前向计算，比数据中心级 GPU 慢，但不付出任何网络代
价，这就是全部的延迟——假设是 70 毫秒，留出 30 毫秒的余量。服务器推理把同一份预算花在网络加计算上：一次请求
离开机器人、被序列化、在更快的远程 GPU 上运行、再被序列化传回——假设本地网络往返各 4 毫秒、序列化往返各 5
毫秒、服务器端计算 30 毫秒，合计 48 毫秒，对着同一个上限还留出 52 毫秒的余量。两者在平均情况下都绰绰有余；差
别体现在偶尔慢的那一次调用上。机器人本地的计算耗时相当稳定，因为它只取决于模型和加速器本身，不涉及任何共享
资源；服务器往返则要承受网络当下正在发生的一切，即便中位数远低于 100 毫秒，偶尔也会超过这个上限。

分块正是让这种偶发的超时变得无关紧要的机制。当 $H=1$ 时，一旦当前动作被消费掉，缓冲区里就什么都不剩，所以一
次超过截止时间的调用会让控制回路一直卡住，直到它返回——这正是深入话题 (c) 里冻结这个回退方式对应的情形。当
$H=8$ 时，一旦最新的动作块到达，缓冲区里就已经有 $7\times100=700$ 毫秒的余量，所以一次飙升到比如 180 毫秒的
调用——按 100 毫秒预算的字面标准，这仍然是一次超时——会被完全吸收：控制回路此时仍在消费上一个动作块，等这次
迟到的调用返回时，下游根本不会察觉到这次超时。分块从不会提高任何单次调用所受的 100 毫秒上限；它降低的是需要
发起调用的频率，而作为这一点的直接后果，也提高了系统在超时变得可见之前，能够吸收多少次偶发超时。

### 追问

- **跨本体数据。** 一支拥有不止一种机器人型号的机队，会打破上面“一套权重、不分支”的假设，因为不同的手臂有不
  同的自由度和不同的动作空间；通常的解法是用一个共享的视觉-语言主干，搭配要么是按本体各自独立的动作头，要么
  是一种不同型号的演示都能表达出来的、与本体无关的规范动作表示（末端执行器的位姿增量和夹爪状态，而不是原始
  关节指令）。数据模型需要一个显式的 `embodiment_id` 字段，任务配比（见上面的流水线）也需要同时按本体和按任
  务配比，而不能把整支机队当成一个池子处理。
- **从人类视频里学习。** 一段人类完成任务的视频不带任何机器人动作标签，所以它没法像一条遥操作演示那样直接监
  督动作预测损失；它要么用来预训练共享的视觉-语言部分（进入训练数据混合里网络规模的那一侧，而不是带标签动作
  的那一侧），要么需要一个专门的逆动力学或者“手部姿态到动作”的重定向模型来产生一个伪标签，而这个伪标签又需
  要一套独立于上面机器人回合整理流水线的质量过滤机制。
- **家庭场景里的数据隐私。** 从受控的机队环境搬进普通人的家里，会让旁观者和私密空间进入摄像头画面；一个在家
  庭里采集的回合，需要在画面离开机器人之前就在机器人本地完成人脸遮蔽，需要比机队默认策略更严格的保留期限，
  也需要给住户一条按需删除自己回合数据的途径——在数据模型里，它还需要有自己独立的一层，日常的整理和训练作业
  默认不会读取它，而不是和受控环境机队的数据混在一个池子里。
- **机器人数量增加十倍。** 存储与接入大体按比例线性增长——每年约 37.9 PB 的原始传感器数据，而不是 3.79
  PB——这会把元数据索引从单个实例逼成按 `robot_id` 或 `task_id` 分片；标注负载也按同样比例增长，把人工抽查进
  一步从固定数量推向检测器打分量的固定比例——而这已经是这份设计默认的做法，不是被规模逼出来的改变；去重在
  200 台机器人的规模下还只是可选的余量，到了这个规模就变得不可或缺，因为一旦足够多机器人尝试同一个任务，近似
  重复的演示会比真正有价值的新增覆盖增长得更快；分阶段部署则可以在每个阶段用更小的百分比步长达到同样的统计
  置信度，因为更大机队的 1% 所对应的试验量，本来就比这支机队的 1% 更多。

<details>
<summary>估算核对（可运行）</summary>

```python
import math
import numpy as np
from scipy import stats

# ---- camera data ----
frame_bytes = 40_000
cams, fps = 3, 15
cam_bytes_per_s = cams * fps * frame_bytes
assert cam_bytes_per_s == 1_800_000

day_s = 8 * 3_600
assert day_s == 28_800

robot_day_cam_bytes = cam_bytes_per_s * day_s
robot_day_cam_gb = robot_day_cam_bytes / 1e9
assert robot_day_cam_bytes == 51_840_000_000
assert robot_day_cam_gb == 51.84

n_robots = 200
fleet_day_cam_gb = robot_day_cam_gb * n_robots
assert fleet_day_cam_gb == 10_368.0
assert round(fleet_day_cam_gb / 1000, 1) == 10.4          # TB/fleet-day, as quoted

# ---- low-dimensional data (negligible -- computed, not assumed) ----
lowdim_bytes_per_s = 1_000
robot_day_lowdim_bytes = lowdim_bytes_per_s * day_s
robot_day_lowdim_mb = robot_day_lowdim_bytes / 1e6
assert robot_day_lowdim_bytes == 28_800_000
assert robot_day_lowdim_mb == 28.8

fleet_day_lowdim_gb = robot_day_lowdim_mb * n_robots / 1000
assert fleet_day_lowdim_gb == 5.76

assert cam_bytes_per_s / lowdim_bytes_per_s == 1_800        # camera stream is 1,800x the low-dim one
assert round(fleet_day_cam_gb / fleet_day_lowdim_gb) == 1_800

# ---- episodes per day ----
episode_s = 60
episodes_per_robot_day = day_s / episode_s
assert episodes_per_robot_day == 480

episodes_fleet_day = episodes_per_robot_day * n_robots
assert episodes_fleet_day == 96_000

teleop_frac, autonomous_frac = 0.30, 0.70
teleop_episodes = episodes_fleet_day * teleop_frac
autonomous_episodes = episodes_fleet_day * autonomous_frac
assert teleop_episodes == 28_800 and autonomous_episodes == 67_200
assert teleop_episodes + autonomous_episodes == episodes_fleet_day

# ---- a year of storage ----
fleet_day_total_gb = fleet_day_cam_gb + fleet_day_lowdim_gb
assert fleet_day_total_gb == 10_373.76

annual_gb = fleet_day_total_gb * 365
annual_pb = annual_gb / 1e6
assert annual_gb == 3_786_422.4
assert round(annual_pb, 2) == 3.79
assert round(annual_pb * 10, 1) == 37.9                      # ten times the fleet: follow-ups, below

print("all requirements-and-scale numbers check out")


# ---- Wilson score interval ----
def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Inverts the test statistic (phat - p) / sqrt(p(1-p)/n) directly, rather than substituting phat
    for p in the variance as the Wald interval does -- gives closer-to-nominal coverage for small n or
    phat near 0 or 1, both common when real-robot evaluation trials are expensive."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / trials
    denom = 1 + z ** 2 / trials
    center = (phat + z ** 2 / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / trials + z ** 2 / (4 * trials ** 2))
    return center - half, center + half


lo, hi = wilson_interval(42, 60)
assert round(lo, 3) == 0.575 and round(hi, 3) == 0.801

# the interval's center, from the same derivation: shrunk toward 0.5 relative to phat=0.700 itself
z95 = stats.norm.ppf(0.975)
assert round(z95, 3) == 1.960                                # as quoted: z approx 1.960 at 95% confidence
phat = 42 / 60
wilson_center = (phat + z95 ** 2 / 120) / (1 + z95 ** 2 / 60)
assert round(wilson_center, 3) == 0.688

# the Wald interval, for comparison: centred exactly on phat, unlike Wilson's shrinkage toward 0.5
wald_half = z95 * math.sqrt(phat * (1 - phat) / 60)
wald_lo, wald_hi = phat - wald_half, phat + wald_half
assert round(wald_lo, 3) == 0.584 and round(wald_hi, 3) == 0.816
assert wald_lo > lo and wald_hi > hi                         # NOTE: a common mistake is assuming Wald sits
                                                              # inside Wilson, as a "correction" would -- it
                                                              # doesn't in general: Wald is centred on phat,
                                                              # Wilson is shrunk toward 0.5, so the intervals
                                                              # shift relative to each other rather than nest
assert (hi - lo) < (wald_hi - wald_lo)                       # and, here, Wilson is also the narrower interval


# ---- trials per arm to tell 60% from 70% apart, alpha=0.05 two-sided, 80% power ----
def required_n_per_arm(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Two independent samples of size n each; H0: p1 == p2, tested with the pooled-variance z-test.
    Solves P(reject | H1: true rates p1, p2) = power for n (the standard normal-approximation
    sample-size formula for a two-proportion test)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / (p1 - p2) ** 2)


n_per_arm = required_n_per_arm(0.60, 0.70)
assert n_per_arm == 356

# an independent recomputation of the same formula, kept unrounded, to check the numbers quoted around it
z_check, zpow_check = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
assert round(z_check, 3) == 1.960                             # as quoted: z_alpha/2 approx 1.960
assert round(zpow_check, 3) == 0.842                          # as quoted: z_power approx 0.842
pbar_6070 = (0.60 + 0.70) / 2
raw_n_per_arm = (z_check * math.sqrt(2 * pbar_6070 * (1 - pbar_6070))
                 + zpow_check * math.sqrt(0.60 * 0.40 + 0.70 * 0.30)) ** 2 / (0.60 - 0.70) ** 2
assert round(raw_n_per_arm, 1) == 355.9                       # as quoted: n before rounding up to 356

one_task_trials = n_per_arm * 2
assert one_task_trials == 712                                 # as quoted: 712 real episodes, one task
one_task_hours = one_task_trials * episode_s / 3_600
assert round(one_task_hours, 2) == 11.87
assert one_task_hours < 12                                    # as quoted: "just under 12 hours"

# ---- Monte Carlo check of that power at n = 356 per arm ----
rng = np.random.default_rng(0)
reps = 20_000
p1, p2, alpha = 0.60, 0.70, 0.05
z_alpha = stats.norm.ppf(1 - alpha / 2)

x1 = rng.binomial(n_per_arm, p1, size=reps)
x2 = rng.binomial(n_per_arm, p2, size=reps)
phat1, phat2 = x1 / n_per_arm, x2 / n_per_arm
pbar_hat = (x1 + x2) / (2 * n_per_arm)
se = np.sqrt(pbar_hat * (1 - pbar_hat) * (2 / n_per_arm))
z_stat = np.divide(phat1 - phat2, se, out=np.zeros(reps), where=se > 0)
simulated_power = float(np.mean(np.abs(z_stat) > z_alpha))
mc_se = math.sqrt(simulated_power * (1 - simulated_power) / reps)
assert mc_se < 0.005                                          # simulation is precise enough to trust the check
assert abs(simulated_power - 0.80) < 3 * mc_se                 # within simulation noise of the target power
assert simulated_power == 0.7999                               # as quoted: this simulation's exact rejection rate

# ---- why not just run every task to this standard on real hardware ----
n_eval_tasks = 50
full_program_trials = n_eval_tasks * n_per_arm * 2
assert full_program_trials == 35_600
full_program_hours = full_program_trials * episode_s / 3_600
assert round(full_program_hours, 1) == 593.3
assert round(full_program_hours / 24, 1) == 24.7

print("evaluation-statistics numbers check out")


# ---- action-chunk latency budget ----
control_hz = 10
tick_ms = 1_000 / control_hz
assert tick_ms == 100

chunk_deadline_ms = 100                                        # given: at most 100 ms per chunk, any H
                                                                 # NOTE: it's the call *rate* that falls as H
                                                                 # grows, not this per-call ceiling -- a common
                                                                 # mix-up is assuming the ceiling itself shrinks

onboard_forward_ms = 70
onboard_margin_ms = chunk_deadline_ms - onboard_forward_ms
assert onboard_margin_ms == 30

net_one_way_ms, serialize_ms, server_forward_ms = 4, 5, 30
server_total_ms = 2 * net_one_way_ms + 2 * serialize_ms + server_forward_ms
assert server_total_ms == 48
server_margin_ms = chunk_deadline_ms - server_total_ms
assert server_margin_ms == 52

H = 8
buffer_runway_ms = H * tick_ms
assert buffer_runway_ms == 800
call_rate_hz = control_hz / H
assert call_rate_hz == 1.25

# a single slow call: absorbed with H=8 buffering, but would stall the control loop with H=1
slow_call_ms = 180
assert slow_call_ms > chunk_deadline_ms                        # it is genuinely a deadline miss
remaining_runway_ms = (H - 1) * tick_ms                         # buffer already holds H-1 unconsumed actions
assert remaining_runway_ms == 700
assert slow_call_ms < remaining_runway_ms                       # H=8: fully absorbed, no stall
assert slow_call_ms > tick_ms                                   # H=1: would have stalled the very next tick

print("action-chunk latency budget checks out")


# ---- staged rollout slice sizes (deep dive c) ----
for pct, expected_count in [(0.01, 2), (0.10, 20), (0.50, 100), (1.00, 200)]:
    assert round(pct * n_robots) == expected_count             # as quoted: 1%/10%/50%/100% of the fleet

print("staged-rollout slice sizes check out")


# ---- closing the loop: priority-weighted collection versus uniform, over 50 tasks ----
N_TASKS, ROUNDS, BUDGET, EPS, LEARN_RATE = 50, 20, 1_000, 0.02, 0.0015

init_rng = np.random.default_rng(7)
initial_success = init_rng.uniform(0.2, 0.7, size=N_TASKS)
initial_gap = 1 - initial_success
hardest_task = int(np.argmax(initial_gap))
easiest_task = int(np.argmin(initial_gap))


def run_collection(weighted: bool, seed: int = 123):
    """Each round allocates BUDGET newly collected episodes across N_TASKS tasks -- uniformly, or
    weighted toward whichever tasks currently fail most often, with a floor EPS so a task that has
    reached near-mastery is still sampled occasionally rather than dropped to zero. A task's gap to
    mastery shrinks with the episodes it receives, diminishing returns per episode (a fixed fraction
    LEARN_RATE of what remains)."""
    gap = initial_gap.copy()
    r = np.random.default_rng(seed)
    total_alloc = np.zeros(N_TASKS)
    min_prob = 1.0
    for _ in range(ROUNDS):
        if weighted:
            weight = gap + EPS                                  # NOTE: without this floor, a near-mastered
                                                                  # task (gap approx 0) gets weight approx 0 and
                                                                  # stops being sampled -- silent starvation
            probs = weight / weight.sum()
        else:
            probs = np.full(N_TASKS, 1.0 / N_TASKS)
        min_prob = min(min_prob, probs.min())
        alloc = r.multinomial(BUDGET, probs)
        total_alloc += alloc
        gap = gap * (1 - LEARN_RATE) ** alloc
    return gap, total_alloc, min_prob


gap_uniform, alloc_uniform, _ = run_collection(weighted=False)
gap_weighted, alloc_weighted, min_prob_weighted = run_collection(weighted=True)

# no starvation: even the easiest task keeps receiving some episodes under the weighted scheme
assert alloc_weighted.min() > 0
assert min_prob_weighted > 0

# the mechanism does what it is meant to: more of the budget goes to the task that started hardest,
# less to the one that started easiest, compared with uniform allocation
assert alloc_weighted[hardest_task] > alloc_uniform[hardest_task]
assert alloc_weighted[easiest_task] < alloc_uniform[easiest_task]

# and the outcome that mechanism is for: the worst-performing task ends up in better shape under
# weighted collection than under uniform, even though uniform spent the same total budget
assert gap_weighted.max() < gap_uniform.max()

print("closing-the-loop simulation confirms no starvation and improved worst-task outcomes")
print("all checks passed")
```

</details>

</details>
