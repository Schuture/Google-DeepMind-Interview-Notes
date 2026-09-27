# 为单卡放不下的模型设计训练方案

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 机器学习系统设计 | ★★★☆☆ | 困难 | RE · MLE · RS | data-parallelism, tensor-parallelism, pipeline-parallelism, sharded-optimiser, checkpointing, fault-tolerance, loss-spikes | 45–60 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

*加速器*（accelerator）是一台拥有自己独立高带宽显存的单个计算设备——一块 GPU 或 TPU 芯片。设想训练一个仅解码器（decoder-only）的 Transformer：80 个 Transformer 层，模型宽度（隐藏维度）8,192，序列长度 4,096——总共 700 亿个可训练参数——从随机初始化开始，在一份包含 $1.4 \times 10^{12}$ 个 token 的训练语料上完整过一遍。训练使用 Adam 优化器，最小化按 token 计算的下一个 token 预测损失；Adam 为每个参数额外维护一个一阶矩（first moment，梯度的指数滑动平均）和一个二阶矩（second moment，梯度平方的指数滑动平均），与参数本身一起保存。

这个模型的参数、梯度与优化器状态放不进单块加速器的显存，要在合理时间内训练完成，需要多块加速器同时参与计算。为下面的规模设计：

- 2,048 块加速器，每块拥有 96 GB 高带宽显存，bf16（一种 8 位指数、16 位宽的浮点格式，训练中标准的降精度格式）下的峰值吞吐为 400 TFLOP/s（$4 \times 10^{14}$ FLOP/s）。
- 每台主机 8 块加速器，通过更快的机内互连（intra-host interconnect）连接，每块加速器 600 GB/s；主机之间、以及主机与存储之间，通过更慢的网络连接，每块加速器 50 GB/s。
- 混合精度训练：前向和反向传播中使用的权重与梯度以 bf16 保存（各 2 字节），Adam 另外为数值稳定的更新保留一份 fp32 的主权重副本和两个矩，均为 fp32（各 4 字节）。
- 每次优化器更新累积一个 4,000,000 token 的全局批次（global batch），由全部加速器共同贡献。
- 检查点存储在整个集群范围内的聚合写入带宽为 20 GB/s。
- 每块加速器独立发生故障，单块加速器两次故障之间的时间间隔服从均值为 50,000 小时的指数分布（即它的平均无故障时间，mean time between failures，MTBF）。
- 目标模型 FLOPs 利用率（model FLOPs utilisation，MFU）为 40%——即训练循环实际达到的吞吐量占集群峰值 FLOP/s 的比例，已经计入通信（计算无法掩盖的部分）、流水线气泡（pipeline bubble）、写检查点、重计算等一切空闲时间之后的结果。

范围内：计算量与显存预算；一套能放下模型、并在这个预算内完成训练的并行布局，以及布局中每一部分各自带来的通信开销；一套为每块加速器提供确定性、可续传 token 流的数据流水线；一套按集群故障率来确定间隔的检查点方案；掉队者（straggler）的检测与容忍；训练不稳定情况的诊断与恢复。范围外：模型架构本身在上述给定尺寸之外的部分（架构已固定，不是需要搜索的对象）；训练数据的内容或筛选；损失函数在“按 token 计算、适合梯度下降”这一点之外的具体数学形式；这一阶段之后的任何环节（微调、对齐、发布前评估）；以及对训练产出的检查点做推理服务。

要产出：

1. 一份计算量与时间估算：用 $C \approx 6ND$ 得到训练总计算量，以及在目标 MFU 下这意味着多长的实际用时。
2. 一份按加速器计的显存方案：bf16 权重、bf16 梯度、fp32 主权重和两个 fp32 Adam 矩各自每个参数占用的字节数；为什么在这个规模上，把这部分状态拆分到多块加速器上是不可避免的；以及激活值显存和激活值重计算（activation checkpointing）如何填进剩下的预算里。
3. 一套并行布局：张量并行、流水线并行、数据并行各自跑在哪条链路上、为什么这样安排，以及每一维各自增加了多少通信。
4. 一份训练运行状态的数据模型：数据流水线的分片顺序与可续传的位置，以及检查点本身。
5. 一张架构图，以及对一个训练步——从一批 token id 到一组更新后的参数——的完整走查。
6. 深入话题：(a) 检查点与故障处理——检查点大小与写入耗时、集群的故障率、能让检查点开销与重做损失之和最小的检查点间隔，以及如何替换一块失效的加速器；(b) 同步训练中掉队者的检测与容忍；(c) 第 300,000 步出现损失骤增（loss spike）时如何诊断与恢复；(d) 训练进行中如何验证这次运行是健康的。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手设计之前值得先确认：加速器预算是固定在 2,048 块（这样设计目标是在这个机群规模内尽量达到更高的 MFU），还是截止时间固定、加速器数量可以调整（这样就值得问一句更多加速器能不能缩短用时——在 MFU 固定的前提下，用时与加速器数量成反比，下面的计算量估算直接就能回答这个问题）；以及架构——80 层、宽度 8,192——本身是固定的，还是可以调整。这份设计假设加速器预算固定为 2,048 块、架构也固定不变，把并行布局当作唯一可以自由选择的变量。

### 需求与规模

**计算量与时间。** 一个稠密 Transformer 的训练计算量可以很好地用 $C \approx 6ND$ 来近似：一次前向传播每个 token 大约花费 $2N$ 次 FLOPs（token 的激活值流经的 $N$ 个参数，每个都对应一次乘法和一次加法），一次反向传播的开销大约是前向的两倍，每个 token $4N$ 次，因为它既要对一层的输入求梯度，也要对这一层的权重求梯度。取 $N = 70 \times 10^9$、$D = 1.4 \times 10^{12}$：

$$
C = 6ND = 6 \times 70 \times 10^9 \times 1.4 \times 10^{12} \approx 5.88 \times 10^{23} \text{ FLOPs。}
$$

集群的峰值吞吐量为 $2{,}048 \times 400 \times 10^{12} \approx 8.19 \times 10^{17}$ FLOP/s；在目标 40% 的 MFU 下，实际交付约 $3.28 \times 10^{17}$ FLOP/s，于是

$$
T = C / (0.40 \times 8.19 \times 10^{17}) \approx 1.79 \times 10^6 \text{ s} \approx 20.8 \text{ 天。}
$$

4,000,000 token 的全局批次意味着 $D / 4 \times 10^6 = 350{,}000$ 次优化器更新（step）；每一步的计算量为 $6N \times 4 \times 10^6 \approx 1.68 \times 10^{18}$ FLOPs，按实际交付的吞吐量算，耗时约 5.13 秒——与 $T / 350{,}000$ 吻合。

作为对给定参数量的一次核对，稠密 Transformer 标准的分块级近似公式 $N \approx 12 L d_{\text{model}}^2$（每层 $4d^2$ 来自四个注意力投影矩阵，加上 $8d^2$ 来自一个 4 倍宽度的 MLP），对这个架构给出 $12 \times 80 \times 8{,}192^2 \approx 64.4 \times 10^9$——与给定的 $70 \times 10^9$ 已经很接近，只低了约 8%，差额来自这个只算块内部分的近似公式没有计入的词嵌入（token embedding）和输出投影层。

**每块加速器的显存。** Adam 为每个参数保留：一个 bf16 权重和一个 bf16 梯度（各 2 字节，供前向和反向传播使用），以及供更新本身使用的一份 fp32 主权重和两个 fp32 矩（各 4 字节）——合计每个参数 16 字节。对整个模型而言就是 $16 \times 70 \times 10^9 \approx 1.12 \times 10^{12}$ 字节，1.12 TB——比任何一块加速器的 96 GB 多出一个数量级还不止，所以把这部分状态拆到多块加速器上不是一种优化，而是唯一能放得下的办法：哪怕把每一块加速器的每一个字节都腾出来只做这件事，只存一份也至少需要 $\lceil 1.12 \times 10^{12} / 96 \times 10^9 \rceil = 12$ 块加速器。

这里用到三种彼此大体独立的方式，把模型的参数与计算摊到多块加速器上。*张量并行*（tensor parallelism）把单个权重矩阵拆到多块加速器上——每块加速器持有每一层权重的一个切片，在同一批 token 上计算，并在层内交换各自的部分结果。*流水线并行*（pipeline parallelism）把模型的各层切成连续的若干阶段（stage），每个阶段完整地放在一块加速器（或一组加速器）上，各阶段之间前向传递激活值、反向传递梯度。*数据并行*（data parallelism）则相反，复制整份模型，把一批 token 拆开：每个副本各自在自己那部分数据上跑完整的前向与反向传播，各副本在每次更新前对梯度取平均。*分片优化器状态*（sharded optimiser state，即 ZeRO，或全分片数据并行，fully sharded data parallel，FSDP）叠加在数据并行之上：不再让每个数据并行副本各自为它持有的参数保留一份完整的优化器状态（更激进的形式下，连权重和梯度也一起分片），而是把这部分状态拆到各个副本上，每个副本只保留自己负责更新的那一片。

12 块加速器只是“只存一份”这个下界，并不是这份设计实际采用的拆分方式。单靠张量并行、只跨一台主机的 8 块加速器，每块加速器需要承担 $70 \times 10^9 / 8 \approx 8.75 \times 10^9$ 个参数份额的状态：按每参数 16 字节算，约 140 GB——还没存下一份激活值，就已经超出了 96 GB 的预算。再叠加流水线并行来扩大拆分——8 个流水线阶段，每台主机对应一个阶段，每阶段 10 层——把合计的拆分幅度提高到 $8 \times 8 = 64$ 路，每块加速器 $70 \times 10^9 / 64 \approx 1.09 \times 10^9$ 个参数，约 17.5 GB：舒服地落在预算之内。

这样一来，还剩下 $2{,}048 / 64 = 32$ 个这份 64 路分片模型的数据并行副本。如果就此打住，17.5 GB 的状态会被白白复制 32 份——一旦梯度做过平均，每个副本对自己那一片参数算出的 Adam 更新都完全相同，没有理由让每个副本各自存一份、各自算一遍这个更新。转而把优化器状态——主权重加两个矩，每参数 12 字节，完整保留的话每块加速器约 13.1 GB——分片到这 32 路数据并行组上，每块加速器只保留自己名下的 $1/32$，能把它压到约 0.41 GB；再加上每块加速器为了自己的前向和反向传播、仍然需要完整保留的 4.4 GB bf16 权重与梯度（把这部分也一起分片，就是全分片数据并行的做法，能进一步省显存，代价是每次使用前都要多一次权重的全收集，all-gather——这里用不上），权重、梯度与优化器状态加在一起，每块加速器不到 5 GB，留出接近 90 GB 的余量。

反向传播需要的激活值——每一层的输入，以及注意力和 MLP 内部的中间结果——其大小随一块加速器要同时为多少层保留激活值、微批次（microbatch）大小、以及序列长度而增长。在这份设计 4,096 的序列长度下，仅仅一个微批次在单个层边界处的激活值，以 bf16 计，就是 $4{,}096 \times 8{,}192 \times 2 \approx 67 \times 10^6$ 字节，64 MiB；一个持有 10 层的流水线阶段，如果同时为全部 10 层都保留每一个中间张量，这个数字还要再乘上好几倍。*激活值重计算*（activation checkpointing）在前向传播时只保留每一层的输入、丢弃其余全部中间激活值，在反向传播时再从这份输入重新算出被丢弃的部分——用多付出一次前向传播（大约多三分之一的 FLOPs，因为前向与反向的 FLOPs 之比约为 1:2）为代价，换来同时占用的显存不再随一个阶段跑多少层而增长，而是保持恒定。由于 $C = 6ND$ 只计入前向和反向传播、不计入这次重计算，激活值重计算带来的这一次额外前向，正是 40% 的 MFU 目标本就必须吸收的那类开销之一，和通信、流水线空闲时间一样。在权重、梯度与分片后的优化器状态之外，每块加速器还留有接近 90 GB 的余量，这份设计并不需要靠重计算来避免显存耗尽；采用它仍然值得，因为省下来的显存能撑起更深的在途微批次流水线，而不是被闲置。

### 数据模型与 API

**RunState（运行状态）**（整个任务一行）—— `run_id`、`step`（训练进度唯一的权威来源）、`status`（`running | checkpointing | recovering | paused`）、`last_checkpoint_step`、`created_at`。

**DataCursor（数据游标）**（每个数据并行副本一行，共 32 行）—— `run_id`、`dp_rank`、`shard_order_index`（在语料库已分词分片（tokenised shard）的一个按固定种子生成的确定性排列中的位置——同一个种子总能得到同一个顺序，所以只凭种子和这个位置索引就能重新推出来）、`offset_in_shard`（在当前分片内的 token 偏移量）、`epoch`。要精确重建一个副本已经消费过什么、还没消费什么，只需要这四个字段加上那个固定的种子。

**CheckpointManifest（检查点清单）**（每次写检查点的那一步一行）—— `run_id`、`step`、`created_at`、`optimiser_shard_files`（每个 `(tp_rank, pp_stage, dp_rank)` 三元组一个文件，因为分片后的优化器状态按数据并行 rank 各不相同）、`weight_shard_files`（每个 `(tp_rank, pp_stage)` 一个文件，因为梯度同步之后各数据并行副本的权重完全相同——是 64 个文件，不是 2,048 个）、`data_cursors`（截至这一步、每个副本各自的 `DataCursor`，这样恢复时能把数据位置和权重一起精确还原）。

控制面 API：

1. `POST /runs/{run_id}/checkpoint` —— 在常规间隔之外触发一次检查点，例如在一次计划内的维护窗口开始前。
2. `GET /runs/{run_id}/status` —— `{step, status, mfu_ewma, loss_ewma, grad_norm_ewma, flagged_hosts}`，供健康看板或告警规则轮询读取。
3. `POST /runs/{run_id}/resume` —— `{from_step}`（默认取 `last_checkpoint_step`）—— 让每个 worker 都从指定的检查点（重新）启动；既用于普通的故障恢复，也用于深入话题 (c) 中的回退。
4. `POST /runs/{run_id}/hosts/{host_id}/drain` —— 驱逐一台主机而不中止整个运行，背后依靠的是深入话题 (b) 中那套在线的对等节点替换机制。

面向 worker 的 API，由每块加速器上的训练进程使用：

1. `POST /runs/{run_id}/workers/{worker_id}/heartbeat` —— `{step, step_time_ms, loss, grad_norm}`，每一步都发送；同时供深入话题 (b) 的掉队者检测和深入话题 (d) 的健康监控使用。
2. `GET /runs/{run_id}/workers/{worker_id}/assignment` —— `{tp_rank, pp_stage, dp_rank, shard_seed}` —— 一个刚启动或重启的 worker 用它来获知自己在并行布局中的位置、以及自己在数据顺序里分到的那一份；替换掉一台失效主机的备用主机，问到的、拿到的也正是被它替换的那台主机原来持有的这份 assignment。

### 架构

```text
已分词的分片（对象存储）
  |
  |  32 个数据并行副本各自按自己确定性的分片顺序读取（DataCursor）
  v
流水线阶段 0 -> 阶段 1 -> ... -> 阶段 7          （一个数据并行副本；这一行 x32）
[主机 0，8 块加速器]  [主机 1，8 块加速器]      [主机 7，8 块加速器]
第 1-10 层           第 11-20 层                第 71-80 层
张量并行，8 路，机内互连 @ 600 GB/s；阶段之间前向传递激活值／反向传递梯度，
跨主机 @ 每块加速器 50 GB/s
  |
  |  算出损失，再反向经过全部 8 个阶段；此时每块加速器都持有一份 bf16 梯度，
  |  对应它的 (tp_rank, pp_stage) 位置所负责的 1/64 模型分片
  v
在 32 路数据并行组内做梯度归约－分散（reduce-scatter）（跨主机 @ 50 GB/s）
  |
  v
分片优化器更新 —— 每块加速器只更新自己名下那 1/32 的优化器状态切片，随后在 32 路组内
把更新后的权重做全收集（all-gather），让每个副本在下一步开始前拿到完全相同的 bf16 权重
  |
  +--> 检查点写入（每 T_opt 步一次）--> 检查点存储（聚合写入 20 GB/s）
  |
  v
任务控制器／健康监控 —— MFU、损失、梯度范数与各主机每步耗时遥测
```

一步从 32 个数据并行副本各自的第一个流水线阶段开始：从各自 `DataCursor` 当前指向的分片里取出下一个微批次的 token id。张量并行把一层里的每个权重矩阵都按 8 路拆到一台主机的各块加速器上；每块加速器在完整的微批次上计算，但只算每个矩阵里自己那一份切片，8 块加速器在注意力模块结束时做一次全归约（all-reduce）、在 MLP 模块结束时再做一次——前向传播每层 2 次全归约，反向传播按对称性再来 2 次，每次搬运一个大小为微批次 × 序列长度 × 隐藏维度的张量，走主机内 600 GB/s 的互连。这个过程在一个阶段的 10 层里、每一层、每一个微批次都要发生一次——是整套布局里迄今为止最频繁的通信，这正是张量并行被限制在可用的最快链路上的原因。

流水线并行则跨主机进行：阶段 0 为一个微批次算出的输出激活值被送到阶段 1，阶段 1 接着往下做前向传播，如此一路经过全部 8 个阶段，直到算出损失；反向传播把对应的梯度沿相反方向送回去，从阶段 7 送到阶段 0。每个阶段边界、每个微批次只有 2 次传输——远不如张量并行按层计的全归约那么频繁——这正是它能够容忍更慢的、50 GB/s 的跨主机链路的原因；8 个阶段之间同时有多个微批次在途（这正是前面提到的、激活值重计算省下的显存派上用场的地方），所以不会有哪个阶段闲坐着等一个微批次的往返。

等这一步 4,000,000 token 全局批次所需的每一个微批次都走完之后，每块加速器手上都有一份 bf16 梯度，对应它 `(tp_rank, pp_stage)` 位置所负责的 1/64 模型分片。数据并行的同步在这里只发生一次，是三者中最不频繁的一个：在 32 路数据并行组内做一次归约－分散，让每块加速器最终只留下它那份分片优化器状态所负责的那一份已归约梯度切片——对这样一份切片而言，$70 \times 10^9 / 64 \times 2 \approx 2.19 \times 10^9$ 字节，一次环形的归约－分散加全收集，会让每块加速器收发约 $2 \times \frac{31}{32} \times 2.19 \times 10^9 \approx 4.24$ GB，按 50 GB/s 算耗时约 85 毫秒——不到这份设计计算量估算所对应的约 5.13 秒一步的 2%。随后每块加速器对自己名下的那一份切片做 Adam 更新，再做一次全收集，让全部 32 个副本在下一步前向传播开始前重新拿到完全相同的 bf16 权重。这样的顺序安排——通信最频繁、扇入最高的张量并行放在最快的链路和最小的组里；次频繁的流水线并行跨主机、但每次只涉及相邻两个阶段；最不频繁、但单次传输量最大的数据并行同样走较慢的链路，却每一步只付一次——正是让三者合在一起仍然落在 40% MFU 目标所假设的通信预算之内的原因。

### 深入话题

**(a) 检查点与故障处理。** 一个检查点必须保存的是 fp32 主权重和两个 fp32 Adam 矩——bf16 权重可以由 fp32 主权重降精度得到，所以并非检查点本身严格必需，不过很多设计仍然会把它们也一起写入，以换取一次跳过降精度步骤、更快的重启。按每参数 12 字节算，必需部分的检查点是 $12 \times 70 \times 10^9 = 8.4 \times 10^{11}$ 字节，840 GB；把 bf16 权重也算进去，再加 $2 \times 70 \times 10^9 = 140$ GB，合计 980 GB。按集群 20 GB/s 的聚合检查点带宽，写完必需的 840 GB 需要 42 秒；980 GB 那个版本需要 49 秒。

由于张量并行把一台主机的 8 块加速器绑成了一个同步整体——每一层的全归约都需要全部 8 块参与——只要有一块加速器失效，这台主机为其所服务的那个流水线阶段就整个失去作用。这让一台主机的实际故障率变成单块加速器的 8 倍：每台主机的平均无故障时间为 $50{,}000 / 8 = 6{,}250$ 小时，而在全部 256 台主机范围内，整个集群的平均无故障时间约为 $6{,}250 / 256 \approx 24.4$ 小时（直接按加速器数量算也得到同一个数字，$50{,}000 / 2{,}048 \approx 24.4$ 小时——两种算法必然一致，因为二者算的都是同样这 2,048 块加速器合在一起的故障率，只是分组方式不同）。

检查点策略要在两种代价之间权衡：一种是把时间花在写那些其实从未派上用场的检查点上，另一种是故障发生时，把时间花在重做上一个检查点以来的工作上。设检查点写入耗时为 $t_{\text{ckpt}}$、平均无故障时间为 $\mu$（单位相同），能让二者之和最小的检查点间隔——这是通常归功于 Young 和 Daly 的一个标准结果——是 $T_{\text{opt}} = \sqrt{2 \, t_{\text{ckpt}} \, \mu}$。取 $t_{\text{ckpt}} = 42$ 秒、$\mu \approx 87{,}891$ 秒（24.4 小时），得到 $T_{\text{opt}} \approx 2{,}717$ 秒，约 45 分钟——大致每 45 分钟做一次检查点。假设故障在一个间隔内任意时刻发生的概率相等，每次故障平均重做的工作量就是半个间隔，$T_{\text{opt}}/2 \approx 1{,}359$ 秒，约 23 分钟；在这个最优点上，写检查点的开销比例（$t_{\text{ckpt}}/T_{\text{opt}}$）与预期重做的开销比例（$(T_{\text{opt}}/2)/\mu$）恰好相等，各约 1.5%，合计约占总用时的 3.1%——下面的估算核对会确认，换成其他间隔，开销只会更高。

在大约 498 小时的整个运行期间，预计总共会发生约 $2{,}048 \times 498 / 50{,}000 \approx 20$ 次加速器故障（等价地，$498 / 24.4$）。由于每块加速器都要参与每一步的集合通信，任何一处发生故障都会让整个 2,048 块加速器的任务停下来，而不只是影响它自己那一片分片——恢复意味着整个运行回退到最近一次检查点，全部加速器从那里重新开始，代价正是上面算出的那份重做工作量。保留一小批已经就绪、随时待命的备用主机——按这个故障率，少数几台就够了，用来防两次故障凑巧靠得很近这种情况，是一笔便宜的保险——能让一台失效主机腾出的位置立刻被补上，而不必等新硬件到位；替补的主机会向控制面 API 请求被它替换的那台主机原来持有的 assignment（`tp_rank`、`pp_stage`、`dp_rank`、`shard_seed`），并加载最近一次检查点里对应位置的那一份分片。如果一台主机只是被判定为持续掉队，而不是彻底失效，则完全可以跳过重新加载检查点这一步，转而直接从持有同一份分片的对等节点那里拷贝当前权重（深入话题 (b)）。

**(b) 掉队者的检测与容忍。** *掉队者*（straggler）指的是一台仍然有响应、但比其他同伴慢的主机——常见原因有过热降频、一块跑不到额定带宽的网卡，或者共享基础设施上一个吵闹的邻居（noisy neighbour）。由于这份设计里的每一种集合通信（张量并行的全归约、流水线并行的阶段间传输、数据并行的归约－分散）都要等所有参与者应答才会继续，一个掉队者会把整个 2,048 块加速器的任务拖慢到它自己的速度，而且每一步都是如此——比一个高度并行、彼此独立的工作负载里出现掉队者要昂贵得多，那种场景下掉队者只会拖慢它自己那一份工作。

检测直接用 worker 心跳（上面的数据模型）本来就在上报的按步计时：如果一台主机最近若干个连续步的耗时都明显高于全机群的中位数，而不只是某一步偶然偏高，就会被标记出来——之所以要看连续多步的平均情况，是为了把真正、持续的掉队者，和一次偶发的单步波动（比如一次垃圾回收暂停，或一次瞬时的网络重试）区分开来，这两种都不值得为之采取行动。

一旦被标记，有两种应对方式：

- *冗余计算*：为同一份分片，在那块慢加速器之外再配一块备用的一起算，取先算完的那个结果。这个办法在松耦合或异步的工作负载里效果不错，但放在这里意味着要为一个也许会自己恢复的位置，把整个张量并行组的计算（8 块加速器，而不是 1 块）都重复一遍，而且任务的集合通信仍然要等两者中较慢的那个算完——硬件代价很大，收益却并不确定。
- *把确诊的掉队者当作故障处理、直接替换*（选定）：驱逐被标记的主机，换上一台备用主机，走的是和硬故障相同的控制面 `drain` 路径。与硬故障的区别在于，掉队者本身仍然存活：与其重新加载磁盘上最近一次检查点、损失从那之后的工作，不如让替补主机在任务暂停于某个步边界时，直接通过网络从持有相同分片的对等节点——同一个 `(tp_rank, pp_stage)` 位置上另外 31 个数据并行副本中的任意一个——拷贝当前的权重。这只需要付出暂停加拷贝的时间（一份分片量的 bf16 权重，几个 GB，按 50 GB/s 算是几十毫秒），而不是一整个检查点间隔那么多的重做工作量，因为实际上什么都没有丢失。

检测阈值要在误判和反应迟缓之间权衡：驱逐一台只是短暂变慢的主机，代价是几十毫秒加一次不必要的替换；而放任一个真正的掉队者更久，代价是在它被发现之前的每一步，都让整个机群跟着承受同样的拖慢——在 2,048 块加速器的规模上，哪怕一个掉队者只让每一步慢了 10%，浪费掉的也是整个运行 10% 的加速器工时，远远超过反应稍微过于敏感的代价。因此阈值应当偏向更快出手。

**(c) 第 300,000 步出现损失骤增时的诊断与恢复。** 训练损失在某个特定步骤突然大幅升高，而不是像其余时候那样逐步、带噪声地下降，常见原因有四种，可以根据当时记录下来的按步遥测数据（worker 心跳，以及深入话题 (d) 保留的数据与梯度历史）分辨开来：

- *数据批次*：那一步具体的 token 本身有问题——比如分片里被破坏或错位的一段，或者一个异常重复、信息熵很低的批次。`DataCursor` 历史精确记录了每一步消费的是哪个分片、哪个偏移量，所以可疑的批次可以被重新检查，或者单独取出来，拿骤增发生前那一刻的模型重放一遍，不需要靠猜。
- *学习率*：那一步调度表给出的学习率，对训练当前所处的阶段而言偏高了——比如随着训练推进，梯度噪声尺度变大，一个早先还稳定的学习率可能不再稳定。把记录下来的调度表数值和步数对照检查，可以直接发现这一点。
- *数值问题*：某个激活值或梯度超出了 bf16 的表示范围，或者对于一个二阶矩在长时间小梯度之后已经衰减到接近零的参数，Adam 分母里的 epsilon 相对而言变得太小。这两种情况都会在按层记录的梯度范数历史（深入话题 (d)）里留下痕迹：某一层的范数远比其余层跳得高，把问题定位到某个具体运算，而不是整个模型。
- *优化器状态*：和数值问题密切相关——一部分参数（常见的例子是低频 token 对应的嵌入行）的 Adam 二阶矩已经衰减到接近零，会让这部分参数的有效步长（与 $1/\sqrt{v}$ 成正比）在它下一次收到一个不算小的梯度时被放得极大。

如果训练过程中没有记录这些信息，事后就无从诊断——这正是为什么深入话题 (d) 里按步记录数据位置、学习率、损失和梯度范数历史这件事，必须在任何一次骤增发生之前就已经在运行，而不是等出了问题才去补上。

恢复沿用深入话题 (a) 已经提供的检查点回退机制：用一次 `resume` 控制面调用，从骤增发生之前最近的一次检查点恢复（平均而言最多损失 $T_{\text{opt}}/2$ 那么多的工作量）。不同之处在于，继续训练之前要具体改动什么：

- 孤立的坏批次：恢复后专门跳过那个出问题步骤对应的数据位置，其余一切保持不变——确定性、可续传的数据流水线让这一步做得精确，而不是大致估摸。
- 系统性原因（学习率本身有问题，或者数值问题只是被某个特定批次触发、而非由它造成）：恢复时调低学习率，或收紧梯度裁剪，或加入一个 z-loss 项——一个惩罚 log-softmax 归一化项的小型辅助损失，能防止最后一层的 logits 无限增长，这是大型 Transformer 里一个已知的不稳定来源——因为如果根源还在，只跳过那一个批次并不能防止问题复发。

并不是每一次骤增都需要回退：梯度裁剪本身已经限制了单单一步的问题能对参数造成多大伤害，而且骤增有时候几步之内就会自己恢复。一个合理的策略是只有在损失连续若干步都没能恢复时才回退，而不是每次瞬时的骤增都回退——只在真正需要的时候，才去付深入话题 (a) 里那份重做工作的代价。

**(d) 验证训练是否健康。** 持续跟踪、并随每一步遥测一起记录下来的信号有四类，各自能捕捉到不同的问题。已达到的 MFU——这份设计自己的 $C_{\text{step}} / (\text{峰值 FLOP/s} \times \text{单步耗时})$，用每一步实测的墙钟时间实时算出来——如果在模型和数据都没有变化的情况下跌破大约 40% 的基线，是一个与模型本身无关的信号，说明基础设施出了问题：掉队者、互连链路退化，或者过热降频，它背后的遥测数据也正是深入话题 (b) 掉队者检测直接用到的那一份。训练损失本身，既包括带噪声的单步原始值（深入话题 (c) 里那种骤增最先在这里出现），也包括一份用来跟踪整体缓慢下降趋势的平滑滑动平均值（用来发现单看某一步数值发现不了的平台期或缓慢漂移），是最主要的进度信号。梯度范数，既包括裁剪之前的全局范数（这里出现跳变是骤增的先兆），也包括按层的范数（深入话题 (c) 用它来定位问题），能在问题还没必然反映到损失上之前就捕捉到不稳定。在一个固定的、与任何训练分片都不重叠的留出集（held-out set）上按固定的步数节奏做周期性评估，能捕捉到单靠训练损失捕捉不到的问题——最重要的一种，是数据流水线的 bug 让留出集数据泄漏进了训练集，这会让训练损失看起来一切正常，而真实的泛化能力却没有提升；同一次评估，也正是直接核对数据流水线自身不变量的地方——把每个分片记录下来的消费次数，和步数与全局批次大小推算出的应有次数做对比，而不是仅仅相信确定性顺序一直成立。

### 追问

- **混合专家（MoE）与专家并行。** 把每个 token 路由到许多专家 MLP 模块中的一小部分，而不是过一个统一的稠密 MLP，会引入第四个并行维度——专家并行（expert parallelism）：它把*不同*的专家拆到不同加速器上，而不是像张量并行那样拆分同一份稠密计算，并且需要一次全对全（all-to-all）交换——而不是全归约——把每个 token 的激活值路由到它被分到的那个专家所在的加速器，再送回来，这份通信量依数据而定，各专家之间可能并不均衡。
- **序列并行或上下文并行（sequence / context parallelism）。** 当序列长度长到激活值里注意力那一项（随序列长度的平方增长）、乃至它的线性项本身，比参数量本身更能左右显存预算时，把序列维度本身拆到多块加速器上——每块各持有一段连续的位置——就需要为注意力做一次环形的 key、value 分块交换，因为一个查询位置需要用到它之前所有位置的 key 和 value，这份交换随着环转动与注意力计算重叠进行。
- **算力变化时的弹性训练。** 数据并行是适合用来伸缩的维度，因为它的各个副本彼此可以互换，不像张量并行或流水线并行那样绑定在模型的某种固定切分方式上；一次伸缩需要重新计算全局批次大小（或者每个副本的微批次数量）、需要一次基于检查点的重启，重用和普通故障恢复相同的路径，并且要留意——如果允许全局批次随算力变化，而不是固定不变，学习率与批次大小之间的关系不能悄悄跟着漂移。
- **FP8 训练。** 在计算量最大的矩阵乘法里，把输入换成 8 位浮点，在原生支持它的硬件上，能让可达到的峰值 FLOP/s 相对 bf16 大致翻倍，直接抬高了 40% 这个 MFU 目标所占的那个分母的上限——但它不会改变每参数 12 字节的优化器状态这一项，因为主权重和两个矩仍然需要更高的精度，才能不发生下溢地累积微小的更新，而且它还会引入按张量或按块的缩放因子需要跟踪，这是一个和深入话题 (c) 直接相关的新的数值稳定性隐患。
- **一个大十倍的模型。** 显存方面的论证是线性缩放的——7,000 亿参数在每参数 16 字节下的状态总量约为 11.2 TB，需要成比例更多的加速器才能放下，而且很可能需要更深的张量乘流水线并行度，因为单台主机的 $8 \times 96$ GB，在同样的张量并行宽度下，已经放不下哪怕一份不大的切片了。在 token 数不变的前提下，计算用时随参数量线性增长；如果像计算最优缩放（compute-optimal scaling）所建议的那样，一个大十倍的模型也要配上成比例更多的训练 token，用时还会进一步叠加。

<details>
<summary>估算核对（可运行）</summary>

```python
import math
import random
import statistics as stats

# ---- setup ----
n_layers, d_model, seq_len = 80, 8192, 4096
N = 70e9                       # stated parameter count
D = 1.4e12                     # training tokens
n_accel, accel_mem, accel_peak = 2048, 96e9, 400e12
accel_per_host = 8
n_hosts = n_accel // accel_per_host
assert n_hosts == 256
inter_host_bw = 50e9           # bytes/s per accelerator
target_mfu = 0.40
global_batch = 4_000_000       # tokens/step
ckpt_bw = 20e9                 # bytes/s, aggregate
mtbf_accel_hours = 50_000

# ---- compute and time ----
C = 6 * N * D
assert C == 5.88e23

peak_cluster = n_accel * accel_peak
assert peak_cluster == 8.192e17
achieved = target_mfu * peak_cluster
assert math.isclose(achieved, 3.2768e17)

T_run = C / achieved
assert round(T_run / 86400, 1) == 20.8

steps = D / global_batch
assert steps == 350_000

C_step = 6 * N * global_batch
t_step = C_step / achieved
assert math.isclose(t_step, T_run / steps)
assert round(t_step, 2) == 5.13

# parameter-count check: the standard 12Ld^2 block approximation against the stated N
params_formula = 12 * n_layers * d_model ** 2
assert params_formula == 64_424_509_440
assert 0.85 < params_formula / N < 0.95            # within ~8%, the gap being embeddings + output head

# ---- memory per accelerator ----
bytes_bf16_weight, bytes_bf16_grad = 2, 2
bytes_fp32_master, bytes_fp32_m1, bytes_fp32_m2 = 4, 4, 4
bytes_per_param = bytes_bf16_weight + bytes_bf16_grad + bytes_fp32_master + bytes_fp32_m1 + bytes_fp32_m2
assert bytes_per_param == 16
opt_bytes_per_param = bytes_fp32_master + bytes_fp32_m1 + bytes_fp32_m2
assert opt_bytes_per_param == 12
wg_bytes_per_param = bytes_bf16_weight + bytes_bf16_grad
assert wg_bytes_per_param == 4

total_state = bytes_per_param * N
assert total_state == 1.12e12                       # 1.12 TB

floor_accelerators = math.ceil(total_state / accel_mem)
assert floor_accelerators == 12

TP = 8                                               # within a host, matches the fast intra-host link
mem_tp_only = (N / TP) * bytes_per_param
assert mem_tp_only == 140e9                          # exceeds one accelerator's 96 GB

PP = 8                                               # across hosts: one host per stage
assert n_layers % PP == 0 and n_layers // PP == 10
model_parallel = TP * PP
assert model_parallel == 64
params_per_shard = N / model_parallel
assert params_per_shard == 1_093_750_000

mem_unsharded = params_per_shard * bytes_per_param
assert mem_unsharded == 17.5e9                       # fits comfortably in 96 GB

DP = n_accel // model_parallel
assert DP == 32

opt_state_full = params_per_shard * opt_bytes_per_param
opt_state_sharded = opt_state_full / DP
wg_mem = params_per_shard * wg_bytes_per_param
mem_sharded = wg_mem + opt_state_sharded
assert round(opt_state_full / 1e9, 3) == 13.125
assert round(opt_state_sharded / 1e9, 3) == 0.41
assert round(wg_mem / 1e9, 3) == 4.375
assert mem_sharded < 5e9                             # "under 5 GB"
assert accel_mem - mem_sharded > 90e9                # "close to 90 GB of headroom"

activation_tensor_bytes = seq_len * d_model * 2      # one microbatch=1 sequence, one layer boundary, bf16
assert activation_tensor_bytes == 67_108_864
assert activation_tensor_bytes / 2**20 == 64.0        # 64 MiB

# ---- ring communication: data-parallel gradient reduce-scatter + all-gather ----
grad_shard_bytes = params_per_shard * bytes_bf16_grad  # NOTE: this accelerator's (tp,pp) shard, not N
assert grad_shard_bytes == 2_187_500_000
ring_volume = 2 * (DP - 1) / DP * grad_shard_bytes
assert round(ring_volume / 1e9, 2) == 4.24
t_allreduce = ring_volume / inter_host_bw
assert round(t_allreduce * 1000) == 85               # ms
assert t_allreduce / t_step < 0.02                    # under 2% of one step

print("all requirements-and-scale numbers check out")

# ---- checkpointing and cluster failure rate ----
ckpt_essential = opt_bytes_per_param * N
ckpt_with_weights = ckpt_essential + bytes_bf16_weight * N
assert ckpt_essential == 840e9
assert ckpt_with_weights == 980e9

t_ckpt = ckpt_essential / ckpt_bw
t_ckpt_full = ckpt_with_weights / ckpt_bw
assert t_ckpt == 42.0
assert t_ckpt_full == 49.0

host_mtbf_hours = mtbf_accel_hours / accel_per_host
cluster_mtbf_hours = host_mtbf_hours / n_hosts
assert host_mtbf_hours == 6_250.0
assert round(cluster_mtbf_hours, 1) == 24.4
assert math.isclose(cluster_mtbf_hours, mtbf_accel_hours / n_accel)   # same rate, counted either way

mtbf_seconds = cluster_mtbf_hours * 3600
T_opt = math.sqrt(2 * t_ckpt * mtbf_seconds)
assert round(T_opt) == 2717
assert round(T_opt / 60) == 45

expected_redo = T_opt / 2
assert round(expected_redo / 60) == 23

overhead_frac = math.sqrt(2 * t_ckpt / mtbf_seconds)
cross_check = t_ckpt / T_opt + expected_redo / mtbf_seconds
assert math.isclose(overhead_frac, cross_check)
assert round(overhead_frac * 100, 1) == 3.1

expected_failures = n_accel * (T_run / 3600) / mtbf_accel_hours
assert round(expected_failures) == 20
assert math.isclose(expected_failures, T_run / mtbf_seconds)

print("all checkpointing and failure-rate numbers check out")


# ---- tiny simulation: the Young-Daly interval approximately minimises total overhead ----
def simulate_overhead(interval: float, t_ckpt: float, mtbf: float, target_useful: float,
                       rng: random.Random) -> float:
    """Simulate one run needing `target_useful` seconds of useful compute, blocked by a
    checkpoint costing `t_ckpt` seconds every `interval` useful seconds; failures arrive as a
    Poisson process at rate 1/mtbf in wall-clock time and discard all progress since the last
    completed checkpoint. Returns wall-clock time minus target_useful: the pure overhead."""
    useful_done = progress = wall = 0.0
    next_failure = rng.expovariate(1.0 / mtbf)
    while useful_done < target_useful:
        remaining_to_ckpt = interval - progress
        time_to_failure = next_failure - wall
        step = min(remaining_to_ckpt, time_to_failure)
        wall += step
        progress += step
        if time_to_failure <= remaining_to_ckpt:        # a failure hits first: lose this segment
            progress = 0.0
            next_failure = wall + rng.expovariate(1.0 / mtbf)
            continue
        useful_done += progress                          # reached a checkpoint boundary
        progress = 0.0
        if useful_done >= target_useful:
            break
        wall += t_ckpt
    return wall - target_useful


factors = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0]
mean_overhead = {}
for f in factors:
    trials = [simulate_overhead(T_opt * f, t_ckpt, mtbf_seconds, T_run, random.Random(1000 + s))
              for s in range(300)]
    mean_overhead[f] = stats.mean(trials)

assert mean_overhead[1.0] == min(mean_overhead.values())         # T_opt is the best of the sweep
assert mean_overhead[0.15] > 2 * mean_overhead[1.0]               # far too short: overhead dominates
assert mean_overhead[4.0] > 2 * mean_overhead[1.0]                 # far too long: redone work dominates

print("checkpoint-interval simulation confirms the Young-Daly interval near-minimises total overhead")
print("all checks passed")
```

</details>

</details>
