# 大模型基础：注意力、Transformer 与训练流程

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 知识问答 · 口述，含推导 | ★★★★★ | 困难 | RS · RE · MLE · Applied AI · Intern | attention, transformer, rope, kv-cache, scaling-laws, perplexity, tokenisation, post-training, sampling, mixture-of-experts | 12 个问题 / 45–60 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

以下统一记号：模型是一个仅解码器（decoder-only）Transformer，共 $L$ 层，残差流（即“宽度”）维度为 $d$，$h$ 个注意力头，每个头的维度 $d_h = d / h$，序列长度为 $T$。自注意力始终是因果的（causal）：位置 $i$ 只能关注位置 $j \le i$。共十二个问题，按主题分组；每题都要口头作答，多数还需要一段简短推导。

### 注意力机制

**Q1.** 设 $q, k \in \mathbb{R}^{d_h}$，并假设 $q$ 与 $k$ 的全部 $2d_h$ 个分量相互独立，每个分量均值为 $0$、方差为 $1$。推导 $\mathrm{Var}(q^\top k)$。然后精确说明缩放点积注意力（scaled dot-product attention）中的 $1/\sqrt{d_h}$ 缩放具体防止了什么。

**Q2.** 对一层因果（causal）多头自注意力，分别给出它的时间开销，以及实际生成的分数矩阵与权重矩阵所占的内存，都写成 $T$ 与 $d$ 的函数——并明确说明你对头宽 $d_h$ 随 $d$ 增长时的行为做了什么假设。然后精确说明 FlashAttention 改变了这两项开销中的哪一项，以及为什么。

**Q3.** 为什么自回归解码（autoregressive decoding）要缓存键（key）和值（value），而不是每一步都重新计算？把 KV 缓存的字节数推导为层数 $L$、键/值头数 $h_{kv}$、头宽 $d_h$、序列长度 $T$、批大小 $b$，以及每个存储数值所占字节数的函数。对 $L = 32$、$h_{kv} = 8$、$d_h = 128$、$T = 32{,}768$、$b = 1$、bf16（2 字节）求出具体数值。然后说明多查询注意力（multi-query attention，MQA）与分组查询注意力（grouped-query attention，GQA）如何改变 $h_{kv}$，并通过它改变缓存大小。

**Q4.** 旋转位置编码（rotary position embeddings，RoPE）在做点积之前，把位置 $m$ 处的查询和位置 $n$ 处的键分别旋转一个与其位置成正比的角度。先在单个二维坐标对上、取旋转频率为 $\omega$，证明 $q_m^\top k_n$ 只通过 $m - n$ 依赖于 $m$ 和 $n$。再论证这一结论如何推广到一个真实查询/键向量的全部 $d$ 个坐标。将其与加在输入上、位于查询/键投影之前的可学习或正弦绝对位置编码作对比。

### 架构与开销

**Q5.** 计算一个解码器块（decoder block）的参数量 $N$，残差宽度为 $d$，MLP 隐藏层大小为 $4d$，不含偏置（bias），且忽略归一化层的参数。然后给出一个 $L$ 层、词表大小为 $V$、输入/输出嵌入绑定（tied，即输入和输出共用同一张 $V \times d$ 矩阵）的模型的总参数量。只用块内的非嵌入参数量 $N$，推导为什么训练每个 token 大约需要 $6N$ 次浮点运算（floating-point operations，FLOPs），推理大约需要 $2N$ 次，并明确说明这一近似舍弃了计算中的哪些部分。

**Q6.** 对比前置归一化（pre-norm）与后置归一化（post-norm）残差块：分别说明归一化相对残差相加处于什么位置。用残差流的一个简化线性模型——每个子层（包括其归一化）对残差流的整体效果用一个标量因子概括——推导为什么前置归一化在深层网络中训练更稳定。然后精确说明 RMSNorm 相对 LayerNorm 舍弃了什么，以及两者何时重合。

### 目标函数与数据

**Q7.** 定义逐 token 的交叉熵损失（cross-entropy）与困惑度（perplexity），并说明困惑度与“每 token 比特数”的关系。某模型为一个长度为 4 的序列的四个真实下一个 token（各自以之前的真实 token 为条件）分别给出概率 $0.5$、$0.2$、$0.8$、$0.25$。求以奈特（nat）为单位的交叉熵，以及困惑度。

**Q8.** 字节对编码（byte-pair encoding，BPE）从单个字符出发，反复合并出现频率最高的相邻符号对，逐步构建子词词表。从词频 `low: 5, lower: 2, newest: 6, widest: 3` 出发（每个词先拆分为字符，末尾再加一个词尾标记 `_`），手动执行前三次合并：写出每一步被合并的符号对及其计数，若出现并列，取字典序较小的符号对（普通字母按 a–z 比较，词尾标记排在所有字母之后）。然后说明合并次数在词表大小与序列长度之间造成的权衡。

**Q9.** 计算最优缩放律（compute-optimal scaling）用 $C \approx 6ND$ 表示预训练所需的浮点运算量（参数量 $N$，训练 token 数 $D$），并使用经验法则 $D \approx 20N$。对计算预算 $C = 10^{23}$ FLOPs，推导 $N$ 与 $D$。然后，不必重新计算数值，说明如果还要把模型部署生命周期内的总推理开销一并最小化（而不只是训练算力），$N$ 的选择会发生什么变化。

### 后训练与推理

**Q10.** 预训练、监督微调（supervised fine-tuning，SFT）与偏好优化（基于人类反馈的强化学习，reinforcement learning from human feedback，RLHF，配合一个学习到的奖励模型，或直接偏好优化，direct preference optimisation，DPO）是典型大模型训练流程的三个阶段。说明每个阶段的损失分别优化什么。解释为什么 SFT 的损失只施加在回复（response）token 上。从一个带 KL 正则的奖励最大化目标出发推导 DPO 的损失，并解释为什么后训练要保留对参考模型的 KL 惩罚。

**Q11.** 精确定义贪心解码（greedy decoding）、温度采样（temperature sampling）、top-$k$ 采样与核采样（nucleus / top-$p$ sampling），把每一种都写成对下一个 token 概率分布的一次变换，并说明每一种对采样分布形状的影响。某模型给出分布 `{a: 0.5, b: 0.2, c: 0.15, d: 0.1, e: 0.05}`（五个 token）。给出 $p = 0.7$ 的 top-$p$ 采样所保留的准确 token 集合。

**Q12.** 在一个混合专家（mixture-of-experts，MoE）层中，路由器（router）把每个 token 发送给 $E$ 个专家中的 $k$ 个，网络的其余部分（注意力、嵌入、路由器本身）则由所有 token 共享。定义总参数量与每个 token 的激活参数量，并对 $E = 8$ 个专家（每个 $2 \times 10^{9}$ 参数）、$5 \times 10^{8}$ 个共享参数、$k = 2$ 的配置分别求出两者。解释为什么需要一个辅助负载均衡损失，以及容量因子（capacity factor）的作用是什么。

## 参考解答

<details>
<summary>展开参考解答</summary>

回答之前先确认架构族：以下全部内容都假设一个标准的前置归一化、多头（或分组查询）仅解码器 Transformer，使用 RoPE 并绑定嵌入——当前大多数开源大模型都属于这一族——并且“参数”始终只指可训练权重，不包括 KV 缓存或优化器状态。凡问题要求给出数值处，先说明它是某个公式给出的精确值，还是一个被广泛使用的经验法则。

### 注意力机制

**Q1.** $\mathrm{Var}(q^\top k) = d_h$，而除以 $\sqrt{d_h}$ 恰好把方差重新变回 $1$，且对任意 $d_h$ 都成立。

写 $q^\top k = \sum_{i=1}^{d_h} q_ik_i$。由 $\mathrm{E}[q_i] = \mathrm{E}[k_i] = 0$ 且 $q_i \perp k_i$，得 $\mathrm{E}[q_ik_i] = 0$，于是 $\mathrm{E}[q^\top k] = 0$，且

$$\mathrm{Var}(q^\top k) = \mathrm{E}\bigl[(q^\top k)^2\bigr] = \sum_{i=1}^{d_h}\sum_{j=1}^{d_h} \mathrm{E}[q_ik_iq_jk_j].$$

当 $i \ne j$ 时，$q_i, k_i, q_j, k_j$ 是四个相互独立、均值为零的变量，故 $\mathrm{E}[q_ik_iq_jk_j] = \mathrm{E}[q_i]\,\mathrm{E}[k_i]\,\mathrm{E}[q_j]\,\mathrm{E}[k_j] = 0$；当 $i = j$ 时，由 $q_i$ 与 $k_i$ 独立得 $\mathrm{E}[q_i^2k_i^2] = \mathrm{E}[q_i^2]\,\mathrm{E}[k_i^2] = 1 \cdot 1 = 1$（用到 $\mathrm{E}[q_i^2] = \mathrm{Var}(q_i) = 1$，因为 $\mathrm{E}[q_i]=0$）。把这 $d_h$ 个留下来的对角项相加，得 $\mathrm{Var}(q^\top k) = d_h$。

因此未缩放的点积标准差为 $\sqrt{d_h}$，随头宽增长而增长。随着 $d_h$ 增大，softmax 之前的分数在绝对值上会散得更开；softmax 随即趋于饱和，输出接近 one-hot 向量（最大的分数主导整个和），其 Jacobian $\partial p_i/\partial s_j = p_i(\delta_{ij}-p_j)$ 在分布如此尖锐时每一项都趋于 $0$——对分数、进而对 $Q$、$K$ 的梯度消失，训练停滞。把 $q^\top k$ 除以 $\sqrt{d_h}$ 会把它的方差重新缩放到恰好 $1$，与 $d_h$ 无关，从而让 softmax 始终处在条件良好的区间，不受头宽影响。

**Q2.** 时间：四个 $d \times d$ 投影（$Q,K,V,O$）共 $\Theta(Td^2)$，加上分数与加权求和这两次收缩共 $\Theta(T^2d)$，合计 $\Theta(Td^2 + T^2d)$。$O(Td)$ 激活之外的内存：把头宽 $d_h$ 当作固定常数时，实际生成的分数/权重矩阵占用 $\Theta(dT^2)$ 个元素。FlashAttention 恰好去掉了这一项内存开销，而不改变 FLOPs 总量。

设 $Q,K,V \in \mathbb{R}^{T \times d}$ 各由一次 $d \times d$ 投影得到，一次投影耗费 $2Td^2$ FLOPs（每次乘加算 2 次），共四次投影（含输出投影 $W_o$）：$8Td^2$。拆成 $h$ 个宽度为 $d_h = d/h$ 的头后，单个头的分数收缩 $Q_hK_h^\top$ 耗费 $2Td_hT$ FLOPs；对 $h$ 个头求和得 $h \cdot 2T^2d_h = 2T^2(hd_h) = 2T^2d$——与拆成多少个头无关，因为无论 $h$ 取何值，$hd_h = d$ 都固定不变。对 $V$ 的加权求和是形状相同的收缩，同样是 $2T^2d$。合计：$8Td^2 + 4T^2d$ FLOPs，即 $\Theta(Td^2 + T^2d)$，在 $T \approx 2d$ 处交叉。

相比之下，分数与权重数组是 $h$ 个各自独立的 $T \times T$ 矩阵——每个头一个，因为每个头的 softmax 是各自独立归一化的——因此共占用 $hT^2 = (d/d_h)\,T^2$ 个数，这一项*并不*独立于头的拆分方式：把 $h$ 翻倍（$d_h$ 减半）会让这部分内存翻倍，尽管上面的 FLOPs 总量丝毫不变。把 $d_h$ 当作固定常数（64–128，基本不随模型规模变化）后，这部分内存是 $\Theta(dT^2)$：对 $d$ 线性、对 $T$ 二次，一旦 $T$ 较大就会主导 $O(Td)$ 的激活内存。

FlashAttention 计算的是同样的 $8Td^2 + 4T^2d$ FLOPs（反向传播中因为要重新计算分块、而不是读回保存的分数矩阵，会略多一些），但从不生成完整的 $T \times T$ 分数/权重矩阵。它把 $K, V$ 切成小块，并维护一个运行中的、未归一化的 softmax——运行最大值 $m$、运行求和 $\ell$、运行输出累加器 $\mathrm{acc}$——每处理一个块就更新一次。把新块的局部统计量 $(m_2, \ell_2, \mathrm{acc}_2)$ 合并进已有的 $(m_1, \ell_1, \mathrm{acc}_1)$ 时，对任一块 $i$ 都有 $e^{s - m} = e^{s - m_i}\,e^{m_i - m}$，于是取 $m = \max(m_1, m_2)$ 后，

$$\ell = \ell_1e^{m_1 - m} + \ell_2e^{m_2 - m}, \qquad \mathrm{acc} = \mathrm{acc}_1e^{m_1 - m} + \mathrm{acc}_2e^{m_2 - m},$$

最终输出为 $\mathrm{acc}/\ell$——这正是对所有已见过的键做 softmax 加权求和的精确结果，而计算过程中每次只需保留一个块的分数矩阵（$O(\text{block}\times d)$）。峰值内存从 $\Theta(dT^2)$ 降到 $O(Td)$，与该层其余部分同阶。

**Q3.** 缓存避免了在每一个新的解码步骤都为所有更早的位置重新计算 $K, V$。缓存大小 $= 2 \cdot L \cdot h_{kv} \cdot d_h \cdot T \cdot b \cdot \text{bytes}$，代入给定数值恰好是 $4$ GiB。

在因果掩码（causal mask）下，位置 $j$ 的键、值向量永远不依赖任何 $j$ 之后的位置；一旦算出，在生成过程剩余的时间里就保持不变。因此解码第 $T{+}1$ 个 token 只需要这一个新 token 的 $Q, K, V$：把新的 $K, V$ 追加到缓存了所有更早位置的缓存里即可，无需为整个前缀重新计算 $K, V$——把每 token $O(T)$、总共 $O(T^2)$ 的开销降为每 token $O(1)$。需要存储的是：对 $L$ 层中的每一层，$K$ 和 $V$ 都要存（系数 $2$），形状均为 $(b, h_{kv}, T, d_h)$——这里专门用 $h_{kv}$ 个键/值头，因为在 MQA/GQA 下 $K, V$ 保留的头数会少于 $Q$ 使用的头数（见下）——再乘以每个存储数值所占字节数。把每个因子相乘：$2 \cdot L \cdot h_{kv} \cdot d_h \cdot T \cdot b \cdot \text{bytes}$。

代入 $L = 2^5$、$h_{kv} = 2^3$、$d_h = 2^7$、$T = 2^{15}$、$b = 2^0$、bf16 $= 2^1$ 字节，每个因子都恰好是 2 的幂，最前面的系数 2 本身也是 $2^1$：把指数相加，$1+5+3+7+15+0+1 = 32$，缓存恰好是 $2^{32}$ 字节 $= 4$ GiB。

MQA 令 $h_{kv} = 1$：所有查询头共用一个键/值头，相对本例再额外缩小 $h_{kv}$ 倍（此处降到 $0.5$ GiB）。GQA 令 $h_{kv} = g$，其中 $1 < g < h$，若干查询头分组共用 $g$ 个键/值头之一——本例中的 $h_{kv} = 8$，只要模型的查询头数超过 8，本身就已经是一种 GQA 配置（比如 $h = 32$，即 $d = h d_h = 4096$）：完全多头注意力（$h_{kv} = h = 32$）需要本例缓存的 $4$ 倍（$16$ GiB），而 MQA 只需要本例的 $1/8$（$0.5$ GiB）——GQA 用一部分 MQA 式的内存节省，换取保留不止一个共享的键/值子空间，通常掉点更少。

**Q4.** $q_m^\top k_n$ 只通过 $m - n$ 依赖于 $m, n$，因为把两个向量都按与位置成线性关系的角度旋转后，点积就变成了角度*之差*的函数，而这个差恰好是 $(m-n)\omega$。

在单个二维坐标对上，记旋转矩阵

$$R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix},$$

它是正交矩阵，满足 $R(\theta)^\top = R(-\theta)$ 以及 $R(\theta_1)R(\theta_2) = R(\theta_1+\theta_2)$（两次旋转复合等于角度相加）。RoPE 把位置 $m$ 处的查询旋转角度 $m\omega$，位置 $n$ 处的键旋转角度 $n\omega$，其中 $\omega$ 是固定频率：$q_m = R(m\omega)\,q$，$k_n = R(n\omega)\,k$。于是

$$q_m^\top k_n = \bigl(R(m\omega)q\bigr)^\top\bigl(R(n\omega)k\bigr) = q^\top R(m\omega)^\top R(n\omega)\,k = q^\top R(n\omega - m\omega)\,k = q^\top R\bigl(-(m-n)\omega\bigr)k,$$

这里先用了 $R(m\omega)^\top = R(-m\omega)$，再用了 $R(-m\omega)R(n\omega) = R(n\omega - m\omega)$。等式右边是 $q$、$k$、$\omega$ 以及单一标量 $m-n$ 的固定函数：任意两对 $(m,n)$ 与 $(m+c, n+c)$ 都给出相同的角度 $-(m-n)\omega$，从而给出相同的值，与 $m,n$ 各自取什么无关。

对一个完整的 $d$ 维查询/键向量，RoPE 把 $d$ 个坐标拆成 $d/2$ 对，对第 $i$ 对使用它自己的频率 $\omega_i = \Theta^{-2i/d}$（一个几何级数，$\Theta$ 通常取 $10^4$–$10^6$）独立地做上述旋转，各对互不影响。完整点积是这 $d/2$ 个成对点积之和，$q_m^\top k_n = \sum_{i=0}^{d/2-1} q^{(i)\top}R\bigl(-(m-n)\omega_i\bigr)k^{(i)}$，其中每一项都只通过 $m-n$ 依赖于 $m,n$——因此整个和也是如此。

对比：可学习或正弦绝对位置编码 $p_m$ 是在查询/键投影*之前*加到 token 嵌入上的，所以 $q_m = W_q(x_m + p_m) = q + W_qp_m$（记 $q = W_qx_m$），同理 $k_n = k + W_kp_n$。于是

$$q_m^\top k_n = q^\top k + q^\top W_kp_n + p_m^\top W_q^\top k + p_m^\top W_q^\top W_kp_n,$$

其中四项里有三项直接涉及 $p_m$ 或 $p_n$，而不是二者之差——这种参数化方式没有任何机制强迫整个和坍缩成只依赖 $m-n$ 的函数。相对位置的行为至多是从训练数据里近似学到的，并非代数上必然成立，也没有理由在比训练时更长的序列长度上依然表现良好，这与 RoPE 精确的、只依赖相对位置的性质不同。

### 架构与开销

**Q5.** 一个块有 $12d^2$ 个参数；整个模型有 $12Ld^2 + Vd$ 个；训练每个 token 大约需要 $6N$ 次 FLOPs，推理大约需要 $2N$ 次，其中 $N = 12Ld^2$——因为线性层里的每个参数，在前向传播中恰好参与一次 token 的乘加，在反向传播中还会参与两次同样大小的矩阵乘法。

注意力有四个 $d \times d$ 矩阵（$W_q, W_k, W_v, W_o$，不含偏置）：$4d^2$。MLP 有 $W_1 \in \mathbb{R}^{d \times 4d}$ 与 $W_2 \in \mathbb{R}^{4d \times d}$：$2 \cdot 4d^2 = 8d^2$。一个块：$4d^2 + 8d^2 = 12d^2$。堆叠 $L$ 个块，再加上一张 $V \times d$ 的嵌入表——由于绑定，它只算一次，而不是为输入查找表和输出反嵌入（unembedding）各算一次：$N_{\text{total}} = 12Ld^2 + Vd$。

关于 FLOPs：一个线性层 $W \in \mathbb{R}^{a \times b}$ 作用在单个 token 的激活向量上，需要做 $ab$ 次乘加 $= 2ab$ 次 FLOPs——恰好是该层每个参数 2 次 FLOPs，因为对单个 token 而言，每个权重都恰好连接一个输入特征和一个输出特征。把 $L$ 个块内部所有线性层（共 $N = 12Ld^2$ 个参数）加起来，前向传播每个 token 大约需要 $2N$ 次 FLOPs。这一近似舍弃了两部分：嵌入查找是一次免费的 gather 操作，不是矩阵乘法；而注意力的分数/加权求和收缩每个 token 还要额外花费 $O(Td)$（见 Q2）——在 $T \ll d$ 这个此近似所针对的通常情形下，相对 $O(d^2)$ 可以忽略。反向传播经过一个线性层时，要再算两次与前向同样大小的矩阵乘法：对该层输入的梯度，以及对该层权重的梯度（一个同样形状的外积）；所以反向传播大约是前向的两倍，约 $4N$。前向加反向合计约 $2N + 4N = 6N$ 次 FLOPs，每个训练 token。用 KV 缓存为一个新 token 提供服务——不重新计算更早的位置——只跑前向：约 $2N$。

**Q6.** 后置归一化在残差相加*之后*才归一化；前置归一化只归一化送入子层的那条分支，残差相加本身不做归一化。正是这个未被归一化的相加，让前置归一化的梯度不会随深度增加而消失或爆炸。RMSNorm 相对 LayerNorm 舍弃的是均值中心化和加性偏置。

后置归一化：$x_{l+1} = \mathrm{Norm}\bigl(x_l + f_l(x_l)\bigr)$。前置归一化：$x_{l+1} = x_l + f_l\bigl(\mathrm{Norm}(x_l)\bigr)$，其中 $f_l$ 是注意力或 MLP 子层。

把每个子层对残差流的净效果线性化为一个标量 $a_l$（这是本论证中一个粗略但标准的简化）。在前置归一化中，$x_{l+1} = x_l + a_l\,g(x_l)$，其中 $g$ 有界——分支内部的归一化从不触碰 $x_l$ 本身——所以 $\partial x_{l+1}/\partial x_l = 1 + O(a_l)$，由链式法则，$\partial x_L/\partial x_0 = \prod_{l=1}^{L}\bigl(1 + O(a_l)\bigr)$：每一层都留下一个恰好为 $1$ 的*加性*恒等项，因此这个乘积不可能像一堆全都小于 $1$ 的因子相乘那样坍缩到 $0$。在后置归一化中，残差之和本身在每一层都被重新归一化，$x_{l+1} = \mathrm{Norm}\bigl(x_l + f_l(x_l)\bigr) \approx a_lx_l$（在某个线性化点附近）——恒等路径与分支被折叠进同一个乘法因子，因为归一化是把整个和一起重新缩放的——所以 $\partial x_L/\partial x_0 \approx \prod_{l=1}^{L} a_l$：一个 $L$ 项的普通乘积，除非每个 $a_l$ 都极其接近 $1$，否则会随 $L$ 指数级地消失或爆炸。这正是深层后置归一化 Transformer 历史上需要小心的学习率预热（warm-up）才能撑过训练早期的原因，而前置归一化在深得多的网络里也能开箱即用地稳定训练。

$$\mathrm{LN}(v)_i = \frac{v_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma_i + \beta_i, \quad \mu = \tfrac1d\textstyle\sum_i v_i,\ \sigma^2 = \tfrac1d\textstyle\sum_i (v_i-\mu)^2; \qquad \mathrm{RMSNorm}(v)_i = \frac{v_i}{\mathrm{RMS}(v)}\,\gamma_i,\quad \mathrm{RMS}(v)=\sqrt{\tfrac1d\textstyle\sum_i v_i^2 + \epsilon}.$$

RMSNorm 舍弃了减均值 $\mu$ 这一步（不做重新居中，只做缩放），也舍弃了加性偏置 $\beta$，只保留可学习的逐特征缩放 $\gamma$。当 $v$ 本身均值为零时，$\sigma^2 = \frac1d\sum_iv_i^2$ 与 $\mathrm{RMS}(v)^2$ 重合（相差仅在 $\epsilon$ 内），此时两者的缩放因子完全相同，区别只在于是否先减去了 $\mu$——RMSNorm 只是省去了这一次归约，每次调用少做一遍对 $d$ 个特征的遍历，并且少了 $\beta$ 的 $d$ 个额外参数。

### 目标函数与数据

**Q7.** 交叉熵约为 $0.978$ 奈特，困惑度 $= \sqrt[4]{50} \approx 2.659$。

对一个长度为 $T$ 的序列，模型在每一步给真实下一个 token 分配的概率记为 $p_t = \pi_\theta(y_t \mid y_{<t})$，逐 token 交叉熵（以奈特为单位）为 $\mathcal{H} = -\frac1T\sum_{t=1}^T \ln p_t$，困惑度为 $\mathrm{PPL} = e^{\mathcal{H}}$——平均而言，模型的不确定程度就如同每一步都在 $\mathrm{PPL}$ 个选项之间做均匀选择：均匀地从 $k$ 个选项中选一个，意外程度为 $-\ln(1/k) = \ln k$ 奈特，当 $\mathrm{PPL}=k$ 时正好等于 $\mathcal{H} = \ln k$。以 2 为底时，$\mathrm{PPL} = 2^{\mathcal{H}/\ln 2}$，所以 $\mathcal{H}/\ln 2$ 就是“每 token 比特数”。

代入 $p_1,\dots,p_4 = 0.5, 0.2, 0.8, 0.25$：$\prod_t p_t = 0.5 \times 0.2 \times 0.8 \times 0.25 = 0.02 = 1/50$，所以 $\mathrm{PPL} = \bigl(\prod_t p_t\bigr)^{-1/T} = 50^{1/4} = \sqrt[4]{50} \approx 2.659$，且 $\mathcal{H} = \ln(\mathrm{PPL}) = \tfrac14\ln 50 \approx 0.978$ 奈特 $\approx 1.411$ 比特每 token。

**Q8.** 前三次合并依次是 $(e,s)$、然后 $(es,t)$、然后 $(est,\_)$，计数都是 $9$。

先拆成符号：`low` → `l o w _`（×5）；`lower` → `l o w e r _`（×2）；`newest` → `n e w e s t _`（×6）；`widest` → `w i d e s t _`（×3）。把所有相邻符号对的计数按词频加权、在四个词上求和：$(l,o){=}7$、$(o,w){=}7$、$(w,\_){=}5$、$(w,e){=}8$、$(e,r){=}2$、$(r,\_){=}2$、$(n,e){=}6$、$(e,w){=}6$、$(e,s){=}9$、$(s,t){=}9$、$(t,\_){=}9$、$(w,i){=}3$、$(i,d){=}3$、$(d,e){=}3$——最大值 $9$ 出现三次并列，分别是 $(e,s)$、$(s,t)$、$(t,\_)$（`newest` 与 `widest` 共享完全相同的后缀 `est_`，其中每一个相邻对都得到同样的组合计数 $6 + 3$）。按字典序打破并列，优先选中 $(e,s)$。合并后 `newest` 变成 `n e w es t _`，`widest` 变成 `w i d es t _`；重新计数，$(es,t)$ 与 $(t,\_)$ 并列为 $9$，而 $(es,t)$ 按字典序更靠前（`es` < `t`），于是第二次合并选中它，得到 `n e w est _` 与 `w i d est _`。再次重新计数，此时只剩 $(est,\_) = 9$ 位居最高且不再并列，第三次合并选中它：`n e w est_` 与 `w i d est_`。`low` 与 `lower` 因为不含这三次合并所需的字母组合，始终未被触及。

合并次数越多，越能把词表里的冗余压缩掉——常见的多字符片段（如 `est_`，合并足够多次后甚至是完整的短词）会变成单个 token——这会缩短给定文本对应的 token 序列长度，而对于注意力开销随 $T$ 二次增长的 Transformer（见 Q2），更短的 $T$ 更便宜。但合并得越多，词表本身也越大（更多互不相同的合并符号需要嵌入、也需要在其上产生 logits，直接加到 Q5 里的 $Vd$ 上），而更罕见的合并 token 在训练中各自被见到的次数也更少，其嵌入是从更少的数据里学出来的；合并太少会让序列偏长、开销偏高，合并太多则会让词表的长尾训练不足，也难以在遇到词表外的拼写时优雅地退回到字符级表示。

**Q9.** $N \approx 2.9 \times 10^{10}$ 个参数，$D \approx 5.8 \times 10^{11}$ 个 token。

把经验法则 $D \approx 20N$ 代入 $C \approx 6ND$：$C \approx 6N(20N) = 120N^2$，于是

$$N \approx \sqrt{\frac{C}{120}}, \qquad D \approx 20N.$$

对 $C = 10^{23}$：$N \approx \sqrt{10^{23}/120} \approx 2.9\times10^{10}$，$D \approx 20 \times 2.9\times10^{10} \approx 5.8\times10^{11}$。

这个 $N$ 是在固定的*训练*算力预算下让训练损失最小，它没有考虑推理开销——推理开销大致随 $N$ 按每个生成 token 线性增长（Q5），并且模型每被服务一次就要付出一次这个代价，而一个已部署模型一生中被服务的 token 总数往往远远超过它训练时用过的 token 数。如果要最小化的是总开销（训练加上服务），最优点就会偏向*更小*的 $N$、在比 $D \approx 20N$ *更多*的 token 上训练：故意多花一些训练算力（偏离训练算力最优的前沿），换来一个更小、服务更便宜的模型，其损失比同等规模下按训练最优比例训练所能达到的更低。

### 后训练与推理

**Q10.** 预训练最大化原始文本上的下一个 token 似然；SFT 最大化同样的下一个 token 似然，但只在给定提示（prompt）下、针对示范回复的 token 上计算；偏好优化在保持与参考策略的 KL 散度足够小的约束下，最大化期望奖励（人类偏好，或一个学习到的奖励模型）——而 DPO 正是同一个带 KL 正则目标的闭式解，不需要单独的奖励模型或强化学习式的 rollout。

预训练：$\mathcal{L}_{\text{pre}} = -\frac1T\sum_t \ln \pi_\theta(x_t \mid x_{<t})$，作用在原始文本 $x$ 上。SFT，给定提示 $x$ 和示范回复 $y$：$\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|}\sum_{t \in y} \ln \pi_\theta(y_t \mid x, y_{<t})$——提示 token 上的损失被精确地掩蔽为 $0$（实践中通常给它们一个被忽略的标签）。提示是模型必须依赖的固定上下文，而不是应该被训练去*生成*的东西：提示的风格千差万别，往往来自与期望回复风格不同的来源（用户、模板），在提示 token 上训练会把梯度信号浪费在模仿提示措辞上，而不是提升回复质量，何况对于一段逐字交给模型的提示，本来也没有什么“正确”的生成方式。

对 DPO，带 KL 正则的目标是

$$\max_\theta\ \mathbb{E}_{x\sim\mathcal D,\ y\sim\pi_\theta(\cdot|x)}\bigl[r(x,y)\bigr] - \beta\,\mathrm{KL}\bigl(\pi_\theta(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x)\bigr).$$

对固定的 $x$，在约束 $\sum_y \pi(y) = 1$ 下对分布 $\pi(\cdot|x)$ 最大化 $\sum_y \pi(y)r(y) - \beta\sum_y\pi(y)\ln\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}$，得到闭式最优解

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\,\exp\!\Bigl(\frac{r(x,y)}{\beta}\Bigr), \qquad Z(x) = \sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\Bigl(\frac{r(x,y)}{\beta}\Bigr).$$

反解出奖励：

$$r(x,y) = \beta\ln\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\ln Z(x).$$

代入 Bradley–Terry 偏好模型 $P(y_w \succ y_l \mid x) = \sigma\bigl(r(x,y_w) - r(x,y_l)\bigr)$，$\beta\ln Z(x)$ 这一项对 $y_w$ 与 $y_l$ 是公共的（只依赖于 $x$），在作差时相消：

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta\ln\frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\ln\frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).$$

在这个模型下，用极大似然把 $\pi_\theta$ 拟合到观测到的偏好对上，正是 DPO 的损失，$\mathcal{L}_{\mathrm{DPO}} = -\ln\sigma\Bigl(\beta\ln\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\ln\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Bigr)$：不需要奖励模型，也不需要强化学习式的 rollout，因为 $\pi^*$ 与 $r$ 之间的闭式关系，把奖励直接代换成了 $\pi_\theta$ 自身上的一个有监督分类损失。

无论走哪条路，都要保留 KL 项，因为奖励信号（一个学习到的奖励模型，或者 DPO 偏好数据里隐含的奖励）只在它被训练或采集时所在的数据分布附近才准确：一个可以任意偏离 $\pi_{\mathrm{ref}}$ 的策略，能找到在不完美的奖励下得分很高、实际却并不好的输出，即所谓“奖励黑客”（reward hacking），而偏离 $\pi_{\mathrm{ref}}$ 太远还有可能丢掉预训练/SFT 阶段学到的语言能力。上面的闭式解把这一点讲得很明白：当 $\beta \to \infty$ 时，无论 $r$ 是什么，$\pi^* \to \pi_{\mathrm{ref}}$；当 $\beta \to 0$ 时，$\pi^*$ 会完全坍缩到最大化奖励的输出上，不再有任何正则化。

**Q11.** 记 $z$ 为下一个 token 的 logits，$p = \mathrm{softmax}(z)$。

*贪心解码*每一步都选 $\arg\max_i p_i$：确定性、零熵，是温度采样在 $\tau \to 0^+$ 时的极限。*温度采样*在 softmax 之前缩放 logits，$p_i(\tau) = e^{z_i/\tau} / \sum_je^{z_j/\tau}$：$\tau < 1$ 把分布往 arg-max 方向收紧，$\tau > 1$ 把分布往均匀方向拉平，$\tau = 1$ 保持不变——在任意有限的 $\tau$ 下，每个 token 都仍可能被采到，只是权重被重新分配。*Top-$k$ 采样*只保留概率最高的 $k$ 个 token，重新归一化使其和为 $1$，再从这个截断后的分布中采样（其余位置概率为 $0$）：它固定的是保留集合的*大小*，与概率质量的分布形态无关。*核（top-$p$）采样*把 token 按概率从大到小排序，保留累积概率首次达到 $p$ 的最短前缀，重新归一化后从中采样：它固定的是保留的*概率质量*，保留集合的大小则随之自适应——分布越尖锐（模型越自信）保留得越少，分布越平坦保留得越多。

已按降序排列的算例——`a:0.5, b:0.2, c:0.15, d:0.1, e:0.05`——取 $p = 0.7$：累加到 `a` 时是 $0.5 < 0.7$；累加到 `a, b` 时恰好是 $0.7 \ge 0.7$，于是保留集合是 $\{a, b\}$。

**Q12.** 总参数量 $= 1.65 \times 10^{10}$；每个 token 的激活参数量 $= 4.5 \times 10^{9}$（约为总量的 $27\%$）。

总参数量是共享参数加上每一个专家的参数，无论某个 token 是否用到它：$\text{shared} + E \cdot P_{\text{expert}}$。每个 token 的激活参数量则是共享参数加上该 token 的路由器实际选中的 $k$ 个专家的参数：$\text{shared} + k \cdot P_{\text{expert}}$（路由器本身增加的参数量相对很小，通常不单独计入）。代入 $E=8$、$P_{\text{expert}} = 2\times10^9$、共享参数 $=5\times10^8$、$k=2$：$\text{total} = 5\times10^8 + 8 \times 2\times10^9 = 1.65\times10^{10}$，$\text{active} = 5\times10^8 + 2 \times 2\times10^9 = 4.5\times10^9$——这个模型每个 token 的推理算力相当于一个 $4.5$B 参数的稠密模型，但容量、以及内存占用，相当于一个 $16.5$B 参数的模型。

路由器只按任务损失训练，这本身并不会对“token 在各专家间分布均匀”施加任何压力——放任不管，它确实会（经验上也确实会）坍缩成把大多数 token 都发给一小撮受偏爱的专家，尤其是在训练早期。被冷落的专家因此收到的梯度更新更少，训练更不充分，这又让路由器更不愿意选它们（一个赢家通吃的反馈回路），而且在一个高效的并行实现里，每个专家每一步的计算/内存预算是固定的，负载不均要么浪费容量（专家空闲），要么迫使 token 被丢弃（专家超载）——同时损害质量与硬件利用率。辅助损失加入一个显式惩罚项，典型形式是 $E \cdot \sum_e f_e P_e$（专家数乘以“实际路由到该专家的 token 比例 $f_e$”与“路由器给该专家的平均 softmax 概率 $P_e$”的点积），当路由均匀时这个惩罚恰好最小，从而把路由器推离坍缩，同时又不需要直接监督哪个 token 该由哪个专家处理。

*容量因子* $c$（通常取 $1$–$2$）把每个专家每个 batch 的最大 token 数设为 $\mathrm{capacity} = c \cdot N_{\text{tokens}}/E$——比完全均匀路由本应送来的 $N_{\text{tokens}}/E$ 多留出一些余量（因为实际路由从来不会完全均匀），同时仍限定了最坏情形。被发往一个已满容量专家的 token 会被丢弃（跳过，例如通过残差连接直接传过去）或改发给排名靠后的专家，从而让专家并行训练中每个专家的计算和内存在各设备间保持一致且有界，代价是当 $c$ 较小时，一部分 token 拿不到它们本该优先选择的专家。

### 追问

- **长上下文外推。** RoPE 的相对位置性质（Q4）在代数上对任意位置都成立，但训练出来的权重实际响应的*注意力模式*并不会免费外推：如果推理时的位置远超训练长度，模型从未在训练中见过的旋转角度会让效果变差，除非重新调整频率方案（例如增大 $\Theta$，或对位置做插值），把推理时出现的角度重新映到训练时见过的范围内。
- **为什么要拆成多个头。** 把 $d$ 拆成 $h$ 个宽度为 $d_h$ 的头，其 FLOPs 与参数量都和单个宽度为 $d$ 的头一样（Q2、Q5），但能让每个头专注于序列上不同的模式，并各自用自己的 softmax 独立归一化——某个头饱和了，也不会把另一个头本来很尖锐的模式冲淡，而单一个覆盖全部 $d$ 维的 softmax 就会有这个问题。
- **投机解码（speculative decoding）。** 用一个小而快的草稿模型提出若干个 token，再让大的目标模型在一次前向传播里把它们全部验证一遍——因为验证一段已经定好的序列只需要一次前向传播，而逐个生成它却需要每个 token 一次；被接受的 token 是“免费”的，第一个被拒绝的 token 之后，生成会从目标模型自己的分布重新继续——把原本好几次串行的大模型步骤，压缩到接近一次。
- **解绑嵌入。** 绑定嵌入能省下 $Vd$ 个参数（Q5），并把“读”一个 token 与“写”一个 token 所用的几何联系在一起，但也就没法让二者拥有不同的容量；一旦 $Vd$ 只占 $N$ 的一小部分，解绑所多花的容量相对很小，因此有时反而更受青睐。

<details>
<summary>验证代码（可运行）</summary>

```python
import math
from collections import Counter

import numpy as np
from scipy import stats

rng = np.random.default_rng(2026)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ============================================================
# Q1 -- Var(q . k) = d_h, and what 1/sqrt(d_h) scaling does to softmax
# ============================================================
for d_h in (8, 16, 64, 256):
    q = rng.standard_normal((200_000, d_h))
    k = rng.standard_normal((200_000, d_h))
    dot = np.sum(q * k, axis=1)
    assert abs(dot.var() - d_h) < 0.05 * d_h                          # Var(q . k) ~= d_h, unscaled
    assert abs((dot / math.sqrt(d_h)).var() - 1.0) < 0.05              # scaled variance ~= 1, for every d_h

raw = rng.standard_normal(6)                                           # NOTE: larger pre-softmax spread saturates
entropies = [stats.entropy(softmax(raw * s)) for s in (0.25, 1.0, 4.0, 16.0)]
assert all(a > b for a, b in zip(entropies, entropies[1:]))            # softmax entropy strictly falls as spread grows
grad_mag = [np.sum(softmax(raw * s) * (1 - softmax(raw * s))) for s in (0.25, 1.0, 4.0, 16.0)]
assert all(a > b for a, b in zip(grad_mag, grad_mag[1:]))               # and the softmax Jacobian's scale shrinks too

# ============================================================
# Q2 -- attention FLOP and memory cost, independent vs. dependent on the head split
# ============================================================
def count_matmul_flops(a_shape, b_shape):
    """2 FLOPs (multiply + add) per scalar multiply-accumulate of an (..., m, k) @ (..., k, n) contraction."""
    *_, m, k = a_shape
    *_, k2, n = b_shape
    assert k == k2
    return 2 * m * k * n


def naive_causal_attention(X, Wq, Wk, Wv, Wo, n_heads, flop_counter=None):
    """One layer of causal multi-head self-attention, straight from the formula. If flop_counter is a list,
    every matmul performed appends its FLOPs (an independent, shape-based FLOP count)."""
    T, d = X.shape
    d_head = d // n_heads

    def mm(a, b):
        if flop_counter is not None:
            flop_counter.append(count_matmul_flops(a.shape, b.shape))
        return a @ b

    Q, K, V = mm(X, Wq), mm(X, Wk), mm(X, Wv)
    Qh = Q.reshape(T, n_heads, d_head).transpose(1, 0, 2)               # (h, T, d_head)
    Kh = K.reshape(T, n_heads, d_head).transpose(1, 0, 2)
    Vh = V.reshape(T, n_heads, d_head).transpose(1, 0, 2)

    visible = np.tril(np.ones((T, T), dtype=bool))
    out_heads = np.zeros((n_heads, T, d_head))
    score_matrix_elements = 0                                          # explicit element count of the materialised
    for h in range(n_heads):                                           # T x T weight matrix, one per head
        scores = mm(Qh[h], Kh[h].T) / math.sqrt(d_head)
        scores = np.where(visible, scores, -np.inf)
        m = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - m)
        weights /= weights.sum(axis=-1, keepdims=True)
        score_matrix_elements += weights.size
        out_heads[h] = mm(weights, Vh[h])
    merged = out_heads.transpose(1, 0, 2).reshape(T, d)
    out = mm(merged, Wo)
    return out, score_matrix_elements


T, d = 24, 16
X = rng.normal(size=(T, d))
for n_heads in (1, 2, 4, 8):                                            # d = 16 is divisible by each of these
    Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.3 for _ in range(4))
    flops = []
    _, score_elements = naive_causal_attention(X, Wq, Wk, Wv, Wo, n_heads, flop_counter=flops)
    closed_form_flops = 8 * T * d ** 2 + 4 * T ** 2 * d
    assert sum(flops) == closed_form_flops                              # FLOPs: exactly independent of the head split
    assert score_elements == n_heads * T ** 2                           # memory: grows linearly with the head count
_, mem_1 = naive_causal_attention(X, *(rng.normal(size=(d, d)) * 0.3 for _ in range(4)), 1)
_, mem_8 = naive_causal_attention(X, *(rng.normal(size=(d, d)) * 0.3 for _ in range(4)), 8)
assert mem_8 == 8 * mem_1                                                # unlike FLOPs, memory scales with h


def naive_head_attention(Q, K, V):
    """Reference single-head causal attention, no tiling: the full T x T matrix is built at once."""
    T, d_head = Q.shape
    scores = (Q @ K.T) / math.sqrt(d_head)
    visible = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(visible, scores, -np.inf)
    m = scores.max(axis=-1, keepdims=True)
    w = np.exp(scores - m)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ V


def flash_attention_head(Q, K, V, block_q=8, block_k=8, track_block_sizes=None):
    """FlashAttention-style tiling: an online (running) softmax, never materialising the full T x T matrix."""
    T, d_head = Q.shape
    out = np.zeros((T, d_head))
    for qs in range(0, T, block_q):
        qe = min(qs + block_q, T)
        q_blk = Q[qs:qe]
        m = np.full((qe - qs, 1), -np.inf)
        l = np.zeros((qe - qs, 1))
        acc = np.zeros((qe - qs, d_head))
        for ks in range(0, qe, block_k):                                # causal: keys beyond qe are never visible
            ke = min(ks + block_k, qe)
            k_blk, v_blk = K[ks:ke], V[ks:ke]
            s = (q_blk @ k_blk.T) / math.sqrt(d_head)
            if track_block_sizes is not None:
                track_block_sizes.append(s.size)
            causal = np.arange(qs, qe)[:, None] >= np.arange(ks, ke)[None, :]
            s = np.where(causal, s, -np.inf)
            m_new = np.maximum(m, s.max(axis=1, keepdims=True))
            correction = np.exp(m - m_new)                              # NOTE: rescale running stats to the new max,
            l = l * correction + np.exp(s - m_new).sum(axis=1, keepdims=True)   # rather than recomputing from scratch
            acc = acc * correction + np.exp(s - m_new) @ v_blk
            m = m_new
        out[qs:qe] = acc / l
    return out


for T_try, d_head_try in ((10, 6), (37, 8), (63, 5)):
    Qh, Kh, Vh = (rng.normal(size=(T_try, d_head_try)) for _ in range(3))
    ref = naive_head_attention(Qh, Kh, Vh)
    tiled = flash_attention_head(Qh, Kh, Vh, block_q=4, block_k=4)
    assert np.allclose(ref, tiled, atol=1e-10)                          # exact, not approximate: same softmax, tiled

block_sizes = []
Qh, Kh, Vh = (rng.normal(size=(256, 8)) for _ in range(3))
flash_attention_head(Qh, Kh, Vh, block_q=8, block_k=8, track_block_sizes=block_sizes)
assert max(block_sizes) == 8 * 8                                        # flash: bounded by block_q * block_k
naive_matrix_size_at_256 = 256 ** 2
naive_matrix_size_at_16 = 16 ** 2
assert (naive_matrix_size_at_256 / (8 * 8)) > 100 * (naive_matrix_size_at_16 / (8 * 8))  # the gap widens like T^2

# ============================================================
# Q3 -- KV-cache size, and the MQA / GQA reductions
# ============================================================
def kv_cache_bytes(L, h_kv, d_head, T, b, bytes_per_element):
    return 2 * L * h_kv * d_head * T * b * bytes_per_element


L, h_kv, d_head, T_ctx, b, bf16_bytes = 32, 8, 128, 32_768, 1, 2
cache = kv_cache_bytes(L, h_kv, d_head, T_ctx, b, bf16_bytes)
assert cache == 2 ** 32 == 4 * 1024 ** 3                                # exactly 4 GiB

full_mha_heads = 32                                                     # a plausible query-head count for d_head=128
mha_cache = kv_cache_bytes(L, full_mha_heads, d_head, T_ctx, b, bf16_bytes)
mqa_cache = kv_cache_bytes(L, 1, d_head, T_ctx, b, bf16_bytes)
assert mha_cache == cache * (full_mha_heads // h_kv)                    # h_kv = 8 is already a 4x GQA reduction from 32
assert mqa_cache == cache // h_kv                                       # MQA shrinks the h_kv = 8 example 8x further
assert mha_cache == mqa_cache * full_mha_heads                          # and is 32x the MQA cache

# ============================================================
# Q4 -- RoPE: q_m^T k_n depends only on m - n
# ============================================================
def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


rope_rng = np.random.default_rng(7)
omega = 0.7
q2, k2 = rope_rng.normal(size=2), rope_rng.normal(size=2)
diffs = [float((rot2d(m * omega) @ q2) @ (rot2d(n * omega) @ k2)) for m, n in [(0, -2), (5, 3), (50, 48), (1000, 998)]]
assert max(diffs) - min(diffs) < 1e-8                                    # depends only on m - n = 2, not m, n themselves
assert abs(diffs[0] - float(q2 @ (rot2d(-2 * omega) @ k2))) < 1e-8        # matches q^T R(-(m-n) omega) k exactly


def rope_rotate(x, position, freqs):
    """x: (..., d) with d = 2 * len(freqs). Rotates each 2-D coordinate pair i by angle position * freqs[i]."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    angles = position * np.asarray(freqs)
    cos, sin = np.cos(angles), np.sin(angles)
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


d_model, n_pairs = 12, 6
freqs = 10000.0 ** (-2 * np.arange(n_pairs) / d_model)                   # the usual RoPE frequency schedule
for _ in range(30):
    q, k = rope_rng.normal(size=d_model), rope_rng.normal(size=d_model)
    delta = int(rope_rng.integers(-50, 50))
    base_m = int(rope_rng.integers(0, 500))
    dots = []
    for shift in (0, 17, 250):                                          # several (m, n) pairs sharing m - n = delta
        m, n = base_m + shift, base_m + shift - delta
        dots.append(float(rope_rotate(q, m, freqs) @ rope_rotate(k, n, freqs)))
    assert max(dots) - min(dots) < 1e-8 * (1 + abs(max(dots)))

q, k = rope_rng.normal(size=d_model), rope_rng.normal(size=d_model)      # a different delta really gives a different
distinct = [float(rope_rotate(q, 300, freqs) @ rope_rotate(k, 300 - delta, freqs)) for delta in (0, 5, -5, 20)]
assert len({round(v, 6) for v in distinct}) == len(distinct)             # value -- the invariance isn't a degenerate constant

# ============================================================
# Q5 -- parameter count, and the 2N / 6N FLOPs-per-token approximations
# ============================================================
def block_shapes(d):
    return [(d, d), (d, d), (d, d), (d, d), (d, 4 * d), (4 * d, d)]      # Wq, Wk, Wv, Wo, W1, W2


for d_try, L_try, V_try in ((8, 3, 37), (16, 5, 101), (32, 2, 61)):
    shapes = block_shapes(d_try) * L_try
    explicit_block_params = sum(a * b for a, b in shapes)                # explicit sum of every matrix's size
    formula_block_params = 12 * L_try * d_try ** 2
    assert explicit_block_params == formula_block_params

    total_explicit = explicit_block_params + V_try * d_try               # tied embedding: the V x d table once
    total_formula = formula_block_params + V_try * d_try
    assert total_explicit == total_formula

    N = formula_block_params
    forward_flops = sum(count_matmul_flops((1, a), (a, b)) for a, b in shapes)     # one token through every linear
    assert forward_flops == 2 * N                                                   # layer, excluding embedding/attention

    backward_flops = 0
    for a, b in shapes:
        backward_flops += count_matmul_flops((1, b), (b, a))             # d(loss)/d(input), same shape as forward
        backward_flops += count_matmul_flops((a, 1), (1, b))             # d(loss)/d(W), an outer product
    assert backward_flops == 4 * N
    assert forward_flops + backward_flops == 6 * N

# ============================================================
# Q6 -- pre-norm vs. post-norm depth scaling (toy linear model); RMSNorm vs. LayerNorm
# ============================================================
def toy_residual_stack(depth, style, a):
    """The exact chain rule of the toy 1-D linear model in the text: an identity term survives every layer
    in pre-norm, but not in post-norm."""
    grad = 1.0
    for _ in range(depth):
        grad *= (1.0 + a) if style == "pre" else a
    return grad


for a in (0.7, 0.9, 1.1):
    for depth in (10, 40, 100):
        assert math.isclose(toy_residual_stack(depth, "pre", a), (1.0 + a) ** depth, rel_tol=1e-9)
        assert math.isclose(toy_residual_stack(depth, "post", a), a ** depth, rel_tol=1e-9)
assert toy_residual_stack(200, "post", 0.9) < 1e-8                       # post-norm: 0.9^200 vanishes
assert toy_residual_stack(200, "pre", 0.9) > 1e30                        # pre-norm: (1.9)^200 grows instead


def layer_norm_np(v, gamma, beta, eps=1e-5):
    mu = v.mean()
    var = ((v - mu) ** 2).mean()
    return (v - mu) / math.sqrt(var + eps) * gamma + beta


def rms_norm_np(v, gamma, eps=1e-5):
    rms = math.sqrt((v ** 2).mean() + eps)
    return v / rms * gamma


d_dim = 10
gamma, beta_zero = rng.normal(size=d_dim), np.zeros(d_dim)
v = rng.normal(size=d_dim)
v_centered = v - v.mean()
assert np.allclose(layer_norm_np(v_centered, gamma, beta_zero), rms_norm_np(v_centered, gamma), atol=1e-6)
assert not np.allclose(layer_norm_np(v, gamma, beta_zero), rms_norm_np(v, gamma), atol=1e-3)   # differ when not centred

# ============================================================
# Q7 -- cross-entropy and perplexity from given next-token probabilities
# ============================================================
probs = [0.5, 0.2, 0.8, 0.25]
cross_entropy = sum(-math.log(p) for p in probs) / len(probs)
perplexity = math.exp(cross_entropy)
assert math.isclose(math.prod(probs), 0.02, rel_tol=1e-9)
assert math.isclose(perplexity, 50 ** 0.25, rel_tol=1e-9)                # closed form: PPL = (prod 1/p_t)^(1/T)
assert math.isclose(cross_entropy, math.log(50) / 4, rel_tol=1e-12)
bits_per_token = cross_entropy / math.log(2)
assert math.isclose(bits_per_token, math.log2(perplexity), rel_tol=1e-9)

# ============================================================
# Q8 -- BPE: the first three merges, by an independent brute-force pair counter
# ============================================================
END = "_"
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
symbol_seqs = {word: list(word) + [END] for word in corpus}


def count_pairs(seqs, freqs):
    """Independent brute-force pair counter: does not call apply_merge or reuse any merge-loop state."""
    counts = Counter()
    for word, seq in seqs.items():
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += freqs[word]
    return counts


def apply_merge(seqs, pair):
    merged = "".join(pair)
    out = {}
    for word, seq in seqs.items():
        new_seq, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        out[word] = new_seq
    return out


def symbol_rank(sym):
    return (1, "") if sym == END else (0, sym)


def best_pair(counts):
    """Most frequent pair; ties broken by the lexicographically smaller pair (letters a-z, then '_')."""
    best_count = max(counts.values())
    candidates = [pr for pr, c in counts.items() if c == best_count]
    return min(candidates, key=lambda pr: (symbol_rank(pr[0]), symbol_rank(pr[1])))


initial_counts = count_pairs(symbol_seqs, corpus)
assert sum(initial_counts.values()) == 15 + 10 + 36 + 18 == 79           # (len(word)+1-1) pairs, weighted by frequency
assert (initial_counts[("e", "s")], initial_counts[("s", "t")], initial_counts[("t", "_")]) == (9, 9, 9)
assert (initial_counts[("l", "o")], initial_counts[("w", "e")]) == (7, 8)

seqs, merges = symbol_seqs, []
for _ in range(3):
    counts = count_pairs(seqs, corpus)
    pair = best_pair(counts)
    merges.append((pair, counts[pair]))
    seqs = apply_merge(seqs, pair)

assert merges == [(("e", "s"), 9), (("es", "t"), 9), (("est", "_"), 9)]
assert seqs["newest"] == ["n", "e", "w", "est_"]
assert seqs["widest"] == ["w", "i", "d", "est_"]
assert seqs["low"] == ["l", "o", "w", "_"]                                # untouched: no e/s/t/_ run to merge
assert seqs["lower"] == ["l", "o", "w", "e", "r", "_"]

# ============================================================
# Q9 -- compute-optimal scaling: C = 6ND, D = 20N
# ============================================================
C = 1e23
N = math.sqrt(C / 120)
D = 20 * N
assert math.isclose(N, 2.9e10, rel_tol=0.01)
assert math.isclose(D, 5.8e11, rel_tol=0.01)
assert math.isclose(6 * N * D, C, rel_tol=1e-9)                           # recovers the compute budget exactly
assert math.isclose(D / N, 20.0, rel_tol=1e-9)

# ============================================================
# Q10 -- masked SFT loss, and the DPO reward substitution
# ============================================================
prompt_logp = [-0.2, -0.5, -0.1]
response_logp = [-0.3, -0.05, -0.9, -0.2]


def sft_loss(response_logp):
    return -sum(response_logp) / len(response_logp)                       # NOTE: prompt tokens contribute 0, not their own logp


masked = sft_loss(response_logp)
all_tokens_loss = -sum(prompt_logp + response_logp) / (len(prompt_logp) + len(response_logp))
assert masked != all_tokens_loss
assert math.isclose(masked, 0.3625, rel_tol=1e-9)
mask = [0, 0, 0, 1, 1, 1, 1]                                                # an equivalent view: per-token weights
weighted = -sum(w * lp for w, lp in zip(mask, prompt_logp + response_logp)) / sum(mask)
assert math.isclose(weighted, masked, rel_tol=1e-12)

dpo_rng = np.random.default_rng(11)
for _ in range(200):
    n_outcomes = int(dpo_rng.integers(3, 8))
    logits_ref = dpo_rng.normal(size=n_outcomes)
    pi_ref = np.exp(logits_ref) / np.exp(logits_ref).sum()
    r = dpo_rng.normal(size=n_outcomes) * 2.0                              # a ground-truth reward
    beta = float(dpo_rng.uniform(0.2, 3.0))

    unnorm = pi_ref * np.exp(r / beta)
    Z = unnorm.sum()
    pi_star = unnorm / Z                                                   # the closed-form KL-regularised optimum

    recovered_r = beta * np.log(pi_star / pi_ref) + beta * np.log(Z)       # invert the closed form for r
    assert np.allclose(recovered_r, r, atol=1e-8)                          # exact recovery, every outcome

    w, l = int(dpo_rng.integers(0, n_outcomes)), int(dpo_rng.integers(0, n_outcomes))
    if w == l:
        continue
    true_pref = 1.0 / (1.0 + math.exp(-(r[w] - r[l])))                     # Bradley-Terry with the true reward
    dpo_logits = beta * math.log(pi_star[w] / pi_ref[w]) - beta * math.log(pi_star[l] / pi_ref[l])
    dpo_pref = 1.0 / (1.0 + math.exp(-dpo_logits))                          # the same probability from pi*/pi_ref alone
    assert math.isclose(true_pref, dpo_pref, rel_tol=1e-6)                  # log Z(x) cancelled, no reward model needed

# ============================================================
# Q11 -- decoding: greedy, temperature, top-k, top-p (nucleus)
# ============================================================
def softmax_from_logits(z, tau=1.0):
    z = np.asarray(z, dtype=float) / tau
    e = np.exp(z - z.max())
    return e / e.sum()


logits = rng.normal(size=8)
assert np.argmax(softmax_from_logits(logits, tau=1e-6)) == np.argmax(logits)       # tau -> 0 recovers greedy
ent = [stats.entropy(softmax_from_logits(logits, tau=t)) for t in (0.3, 1.0, 3.0)]
assert ent[0] < ent[1] < ent[2]                                                     # lower tau -> sharper -> lower entropy


def rank_order(p):
    """Descending by probability; ties broken by ascending original index (the stated tie rule)."""
    return sorted(range(len(p)), key=lambda i: (-p[i], i))


def top_k_filter(probs, k):
    idx = rank_order(probs)[:k]
    out = np.zeros_like(probs)
    out[idx] = probs[idx]
    return out / out.sum()


def top_p_filter(probs, p):
    order = rank_order(probs)
    cum = np.cumsum(probs[order])
    # NOTE: side="left" finds the first index whose cumulative mass already reaches p (rather than the last
    #       index still short of it), so +1 turns that 0-based index into the correct prefix length.
    cutoff = int(np.searchsorted(cum, p, side="left")) + 1
    kept = order[:cutoff]
    out = np.zeros_like(probs)
    out[kept] = probs[kept]
    return out / out.sum()


def brute_force_top_p(probs, p):
    """Independent definition: walk the sorted list and stop at the first prefix reaching p."""
    order, total, kept = rank_order(probs), 0.0, []
    for i in order:
        kept.append(i)
        total += probs[i]
        if total >= p - 1e-12:
            break
    out = np.zeros_like(probs)
    out[kept] = probs[kept]
    return out / out.sum()


names = ["a", "b", "c", "d", "e"]
example_probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
kept_mask = top_p_filter(example_probs, 0.7) > 0
assert set(np.array(names)[kept_mask]) == {"a", "b"}
assert math.isclose(example_probs[:2].sum(), 0.7, rel_tol=1e-9)
assert set(np.array(names)[top_k_filter(example_probs, 2) > 0]) == {"a", "b"}
assert set(np.array(names)[top_k_filter(example_probs, 3) > 0]) == {"a", "b", "c"}

tied_probs = np.array([0.3, 0.3, 0.2, 0.2])                                  # exact ties: (0, 1) and (2, 3)
assert rank_order(tied_probs) == [0, 1, 2, 3]                                # smaller original index wins a tie
assert list(np.nonzero(top_p_filter(tied_probs, 0.5) > 0)[0]) == [0, 1]      # reaches p = 0.5 exactly at index 1
assert list(np.nonzero(top_k_filter(tied_probs, 3) > 0)[0]) == [0, 1, 2]     # ties broken the same way for top-k

for _ in range(500):
    n = int(rng.integers(2, 12))
    raw_p = rng.exponential(size=n) + 1e-6
    probs_r = raw_p / raw_p.sum()
    p_r = float(rng.uniform(0.05, 0.999))
    assert np.allclose(top_p_filter(probs_r, p_r), brute_force_top_p(probs_r, p_r))

for _ in range(200):
    n = int(rng.integers(2, 12))
    probs_r = rng.dirichlet(np.ones(n))
    k = int(rng.integers(1, n + 1))
    filtered = top_k_filter(probs_r, k)
    kept, discarded = filtered > 0, filtered == 0
    assert np.count_nonzero(kept) == k
    assert math.isclose(filtered.sum(), 1.0, rel_tol=1e-9)
    # independent, definition-level property of top-k: every kept probability is >= every discarded one
    assert discarded.sum() == 0 or probs_r[kept].min() >= probs_r[discarded].max()

# ============================================================
# Q12 -- MoE: total vs. active parameters
# ============================================================
def moe_params(n_experts, params_per_expert, shared_params, k):
    return shared_params + n_experts * params_per_expert, shared_params + k * params_per_expert


total, active = moe_params(n_experts=8, params_per_expert=2e9, shared_params=5e8, k=2)
assert total == 1.65e10
assert active == 4.5e9
assert math.isclose(active / total, 4.5 / 16.5, rel_tol=1e-9)

moe_rng = np.random.default_rng(3)                                            # brute force: literal per-expert arrays
experts = [moe_rng.integers(1, 5, size=(3, 3)) for _ in range(8)]
shared_w = moe_rng.integers(1, 5, size=(2, 2))
total_bf = shared_w.size + sum(e.size for e in experts)
chosen = [0, 3]                                                                # the k = 2 experts one token routes to
active_bf = shared_w.size + sum(experts[i].size for i in chosen)
assert total_bf == shared_w.size + 8 * experts[0].size
assert active_bf == shared_w.size + 2 * experts[0].size

print("all checks passed")
```

</details>

</details>
