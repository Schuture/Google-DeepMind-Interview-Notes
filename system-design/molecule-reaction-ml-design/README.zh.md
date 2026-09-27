# 机器学习设计：预测分子对的反应因子

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 机器学习系统设计 | ★★★☆☆ | 中等 | MLE · RE · RS | eda, molecular-fingerprints, pairwise-models, symmetry, data-splitting, leakage, graph-neural-networks, active-learning | 45–60 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

一个化学实验室测量了分子之间的两两反应，为测试过的每一对分子都记录下一个正数，称为*反应因子*（reaction
factor）。原始数据是一张由若干行组成的表，每行是 `(molecule1_name, molecule2_name, reaction_factor)`，其中
`molecule1_name` 和 `molecule2_name` 是两个不同分子的名字，`reaction_factor` 是把它们组合起来测得的结果。另
有一张独立的查找表，把数据中出现过的每个分子名字映射到它的 *SMILES* 字符串（Simplified Molecular Input
Line Entry System，简化分子线性输入规范）——一种把分子的原子、化学键与环状结构编码成一串字符的紧凑文本表示法
（例如 `CCO` 表示乙醇）。同一个分子可以写成不止一种合法的 SMILES 字符串；*规范 SMILES*（canonical
SMILES）是一种固定的规范化算法对给定结构总是产出的那一个唯一字符串，与输入的是哪一个等价字符串无关。任务：给
定这些已测量的行和名字到 SMILES 的查找表，构建一个模型，输入两个分子——如果两者都在查找表里就按名字输入，如
果查找表里从未见过某个分子就直接按它的 SMILES 结构输入——并预测它们的反应因子。

*EDA*（exploratory data analysis，探索性数据分析）是在选择模型之前，先检查数据集的分布、计数、重复值与异常
值的做法，这样建模上的选择就是对数据实际样子的回应，而不是模型本来会默默吸收下去的一个假设。*分子指纹*
（molecular fingerprint）是概括一个分子子结构的定长比特向量；常见的 *ECFP*（Morgan）指纹把每一个以原子为中
心、半径不超过某个固定值的局部子图都散列到一个比特位上，所以共享子结构的分子会共享许多置位的比特，并且这个指
纹只需要一个 SMILES 字符串就能算出来，完全不需要训练。*骨架*（scaffold）是去掉侧链之后剩下的分子核心环系与连
接框架（Bemis–Murcko 骨架是其标准定义），用来把结构相关的分子归到一起。*泄漏*（leakage）是指评估被本不该在预
测时可得的信息污染——比如同一次物理测量、或者它的近似重复，同时出现在训练数据和测试数据里——这会让测出来的指
标比模型的真实表现更乐观。*分子不相交切分*（molecule-disjoint split）是一种训练/测试切分方式：按整个分子、
而不是按单独的行，把它们分配给训练集或测试集，这样训练时见过的分子就不会再出现在任何一个测试对里——这正是衡
量模型对一个它从未遇到过的分子的泛化能力、而不是对熟悉分子的一种未见过组合的泛化能力所需要的切分方式。

前提，所有数字均已给出：

- $200{,}000$ 条已测量的行，涉及 $20{,}000$ 个不同的分子。
- 查找表给出每个分子名字的 SMILES 结构。
- 反应因子严格为正，跨越大约四个数量级。
- 大约 $5\%$ 的不同无序分子对被测量过两次或更多次；同一对分子的重复测量之间，取值的离散程度约为 $0.1$ 个
  $\log_{10}$ 单位。
- 一行里两个名字的先后顺序不带有任何化学意义——实验室确认 $(a, b)$ 测出的因子和 $(b, a)$ 测出的因子相
  同——但原始文件里两种顺序都会出现：哪个分子被写成 `molecule1_name` 是任意的。
- 这个模型将用于两个用途：(i) 预测那些各自都被测量过、但这个具体组合尚未测量过的分子对的因子；(ii) 预测其中
  一个或两个分子从未被测量过的分子对——化学家提出的新候选分子——的因子，以便决定接下来该让实验室去测量哪些
  分子对。

范围内：从原始数据表到选出接下来要测量的分子对的整个建模工作。范围外：把训练好的模型投入生产所需要的服务基础
设施；产生这个反应因子的化学或量子力学机制；以及湿实验室的物流安排。

要产出：

1. 建模之前你会问的问题，以及你会做的 EDA，还有每一项检查可能揭示出什么。
2. 单个分子和一个分子对的表示方式，包括如何让预测结果与两个分子给出的先后顺序无关。
3. 针对上述两种用途各自的数据切分策略，以及对行做朴素随机切分会出什么问题。
4. 一套从简单基线到最强合理选择的模型序列，每个模型各自的训练损失，以及它能捕捉什么、不能捕捉什么。
5. 至少一种提高模型对测量历史很少或完全没有的分子的泛化能力的具体技巧。
6. 一份评估方案：指标、噪声上限，以及按一个分子被测量得有多充分来切片评估。
7. 一份闭环方案：选择接下来该让实验室测量哪些分子对。

面试官可能穿插提出的问题，在设计涉及到相应部分时提出：

- 解释单个 Transformer 层内部的自注意力（self-attention）计算，并说明它的计算开销如何随序列长度变化。
- 一个直接读取分子 SMILES 字符串的 Transformer 编码器，对同一个分子的两个不同合法 SMILES 字符串会给出不同的
  输出；解释原因，并给出两种修复办法。
- 一个图神经网络的读出（readout）步骤，把它最终得到的原子表示汇聚起来——比如对所有原子求和或取平均——变成整
  个分子的一个定长向量；解释这为什么会让得到的分子嵌入与原子的列出顺序无关。
- 说明对于这样一个重尾（heavy-tailed）的回归目标，默认应该用 $L_1$ 损失还是 $L_2$ 损失更好，并说明理由。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手设计之前值得先确认：下游消费者是只需要反应因子的相对排序和对数尺度上的校准，还是需要它在原始尺度上的精确
预测值（这里假设：只需要相对排序和对数尺度上的校准，因为下面深入话题里的主动学习闭环实际上也只消费这些）；以
及查找表里的 SMILES 字符串是否已经保持一致地规范化过（这里假设：没有——不同行里同一个分子可能带着不同、但化
学上等价的 SMILES 字符串，所以规范化是这个设计自己要承担的一个预处理步骤，而不是一个既定前提）。

### 需求与规模

**已测量的部分占多大比例。** $20{,}000$ 个分子的不同无序对总数是
$\binom{20{,}000}{2} = 199{,}990{,}000$，所以这 $200{,}000$ 条已测量的行覆盖了大约

$$\frac{200{,}000}{199{,}990{,}000} \approx 0.1\%$$

的全部可能分子对——已测量的数据只是分子对空间里一个极其稀疏的样本，而这个模型将来会被问到的绝大多数东西
（尤其是主动学习闭环考虑的每一个候选）都是尚未测量过的。

**每个分子平均被测量过的伙伴数。** 每一行贡献两个端点，所以 $200{,}000 \times 2 / 20{,}000 = 20$，即平均每
个分子有 $20$ 个已测量的伙伴。这只是一个平均值：下文的数据与 EDA 一节会去看这个分布的具体形状，这一点很重
要，因为一个按分子个体建模的模型，只有对伙伴很多的分子才能拟合得可靠。

**指纹存储。** ECFP/Morgan 指纹一个常见的固定长度是 $2{,}048$ 位。为整个分子库的每个分子都存一份，代价是
$20{,}000 \times 2{,}048\ \text{位} / 8 = 5{,}120{,}000$ 字节，正好 $5.12$ MB——小到可以把整张指纹表在训练和
推理时都放进内存里，这一点与大规模检索索引不同。

**噪声下限。** $\log_{10}$ 尺度上大约 $0.1$ 的重复测量离散程度（下文数据与 EDA 一节推导出它，评估一节给出它
准确的含义），为任何模型在这个目标上能达到的 RMSE 设定了一个大致的下限：这是标签本身已经带有的噪声的标准
差，不是任何模型自身的性质。

### 数据与 EDA

**目标分布。** 画出 $\log_{10}(\text{反应因子})$ 的分布，而不是原始因子的分布：因为跨越四个数量级，原始尺度
上的直方图会被最大的那些值主导，而对数尺度上的分布才能显示出适合建模的自然范围，并直接引出下文模型一节里的
选择——预测对数值，而不是原始值。

**每个分子的测量次数。** 对每个分子，数一下它出现在多少条已测量的行里。整个集合的均值正好是 $20$（见上文需
求与规模），但这个分布是重尾的：少数几个枢纽分子——常见的试剂或溶剂，与很多伙伴都测量过——占了不成比例的一
大部分行，而一条长尾里的很多分子只有一两次测量。这在任何模型被拟合之前，就已经预示了一个泛化上的缺口：一个依
赖按分子个体建模的模型，只能在这个分布的枢纽一端被信任。

**重复测量的噪声。** 按一个与顺序无关的分子对键（下文数据切分与泄漏一节给出具体构造方式）对行分组，看那大约
$5\%$ 被测量过两次或更多次的不同分子对；每一组内 $\log_{10}(\text{反应因子})$ 的标准差，就是对不可消除的测
量噪声的估计。这正是前提里那个 $0.1$ 的来源，也是评估一节里，每个模型的 RMSE 都要对照的那个噪声下限。

**异常值与单位错误。** 一个对本身并无特别之处的分子对而言、和邻近值相差好几个数量级的取值，或者整整一批行都
恰好偏移了 $10$ 倍或 $1{,}000$ 倍，是单位搞错或者上游转录出错的典型迹象，而不是一次真正的极端测量，应当在被
当作有信息量的极端值之前先核实清楚。

**顺序冲突。** 因为顺序不带有化学意义、但两种顺序都会出现在文件里，按那个与顺序无关的分子对键分组，就能找出
某个分子对一次被记成 $(a, b)$、又在那 $5\%$ 被重复测量的分子对里被记成 $(b, a)$ 的情况；同一个物理分子对的两
次记录之间如果差异很大，既是对上一条“重复测量噪声”这个问题的又一次、独立的验证，也是对实验室“顺序无关”这
个说法是否真的在数据里成立、而不只是口头断言的一次直接检查。

### 特征与表示

**单个分子的表示，** 按泛化能力从弱到强排列：(i) *身份嵌入*（identity embedding）——按分子 id 学出的一个向
量，和矩阵分解里的用户或内容嵌入完全一样——能捕捉到这个分子自己的训练行所揭示的一切，但对一个训练时没见过的
分子没有定义，所以它只能服务于用途 (i)；(ii) *分子指纹*（molecular fingerprint），最常见的是 ECFP/Morgan，
只要知道 SMILES 就能定义出来，不需要任何训练去计算它；(iii) 手工构造的*描述符*（descriptor）——直接从结构上
读出的标量理化性质（分子量、特定官能团的计数、计算出的 $\log P$，……）；(iv) *分子图*（molecular
graph）——原子作为节点，化学键作为边——交给图神经网络去消费，图神经网络会学出自己的子结构特征，而不是依赖指
纹那种固定的哈希。表示 (ii)–(iv) 对一个从未被测量过的分子，只需要一个 SMILES 字符串就都能算出来——这正是用
途 (ii) 需要的性质，也正是 (i) 所缺少的。

**表示一个分子对，且与顺序无关。** 记 $h(\cdot)$ 为选定的那种单分子表示。一个配对函数 $f(h(a), h(b))$ 必须
满足 $f(h(a), h(b)) = f(h(b), h(a))$ 对每一对都成立，因为实验室已经确认因子本身不依赖于顺序：

```text
   molecule a --> h(a) --\
                           >-- symmetric combiner --> predicted log10(reaction factor)
   molecule b --> h(b) --/     (sum, product, |diff|,
                                 or a symmetric bilinear form)
```

有四种构造方式：

- 逐元素**求和**，$h(a) + h(b)$——天然与顺序无关，代价便宜，但会把所有表示之和相同的分子对都折叠成同一个输
  入，丢掉了这具体是哪一对的信息。
- 逐元素**相乘**，$h(a) \odot h(b)$——与顺序无关，捕捉到另一种配对信息（两个分子的特定指纹位或描述符取值同
  时出现），最有用的用法是和求和一起使用，而不是取代它。
- 逐元素绝对**差**，$|h(a) - h(b)|$——与顺序无关，适合目标依赖于两个分子有多不相似、而不是依赖于它们联合量
  级的情形。
- **对称双线性型**，$h(a)^\top S\, h(b)$，其中学出的矩阵满足 $S = S^\top$——之所以与顺序无关，是因为恰好在
  $S$ 对称时 $h(a)^\top S h(b) = h(b)^\top S h(a)$（一个一般的、非对称的 $S$ 不具备这个性质），并且比逐元
  素相乘严格更具表达力，因为它能把 $h(a)$ 的不同坐标和 $h(b)$ 的不同坐标交叉混合。

第五种做法不是让*特征*对称，而是让*模型*对称：把与顺序有关的拼接 $[h(a), h(b)]$ 喂给任意模型，对每一对的两
种顺序都进行训练，推理时把两次预测取平均，$\tfrac12(\hat y(a,b) + \hat y(b,a))$——无论模型本身学到了什么，
推理时都精确地与顺序无关，代价是训练和推理的开销大约翻倍。实践中求和与相乘（或者一个双线性型）会一起使用，
因为二者各自能捕捉到对方漏掉的配对信息；下文的检查会用一个小型拟合模型验证：这种组合确实精确地与顺序无关，
而一个朴素的、按顺序拼接的模型则不是。

### 数据切分与泄漏

**先去重，再切分。** 把每一行都规范化成一个与顺序无关的分子对键——例如把两个分子 id 排序后得到
`(min(id_a, id_b), max(id_a, id_b))`——并把共享同一个键的行合并，把它们的 $\log_{10}(\text{反应因子})$ 取平
均得到一个标签；上文数据与 EDA 一节已经估计过这样做会平均掉多少同一对内部的离散程度，大约是 $0.1$ 个对数单
位，正是变成噪声下限的那个数字。这一步必须在选定任何切分方式*之前*完成：如果直接对未去重的行做随机切分，可能
把同一个物理分子对的一次重复测量分到训练集，另一次带着略微不同噪声值的重复测量分到测试集——模型于是能够部分
地记住一个它实际上已经等价见过的值，这正是题目里定义的那种*泄漏*的一个实例，测出来的指标也就会比实际生产表
现更好看。

**按行随机切分。** 即便已经去重，把这些行均匀随机切分仍然是这里错误的默认做法：由于平均每个分子有 $20$ 个
已测量的伙伴，一个被分进测试集的分子，绝大多数情况下也会同时出现在好几条训练行里——这个切分是*按行*不相交
的，但不是*按分子*不相交的。这能公平地衡量用途 (i)：预测两个分子模型都已经在别处见过、但这个具体组合尚未测
量过的因子；但对用途 (ii) 而言是过于乐观的：模型可以部分靠记住按分子的身份就在这里拿到好分数，而完全不需要学
到任何能泛化到一个它从未见过的分子上的东西。

**单侧未见切分。** 把分子、而不是行，分配给训练集和测试集；如果一行里恰好有一个分子被分进了测试集（另一个在
训练集里），这一行就归入这个测试集。这衡量的是用途 (ii) 里更常见的情形：候选分子对里只有一侧是真正全新的。

**双侧未见切分。** 用同一套分子分配，只有当一行里*两个*分子都被分进了测试集，这一行才归入这个更难的测试集；
一行里两个分子分别落在训练集和测试集两侧，就把这一行整个丢弃——既不用于训练集，也不用于这个测试集的计算——
因为它既不能测试对两个全新分子的外推能力，也不能作为一个分子不相交模型合法的训练样本。这是对用途 (ii) 最棘手
情形最直接的测试：预测两个模型完全没有测量过的分子之间的因子。

**骨架切分。** 把分子均匀随机地分配到各折，仍然可能让一个测试分子和许多训练分子是结构上的近亲——共享同一个
核心环系，只在侧链上有区别——而一个指纹模型或图模型几乎可以像它真的见过这个分子一样，轻松利用这种相似性。改
为按*骨架*把分子分配到各折，让每一个和某个训练分子共享骨架的分子都留在训练折里，是对外推到一个化学上真正陌
生的系列的一种严格更难、也更真实的测试；同一个模型在按分子随机切分与按骨架切分下的分数之差，本身就是一个有
用、可核查的量，用来衡量上面“分子不相交”的结果里，究竟有多少其实来自训练集里近乎重复的分子。

### 模型

下面每一个模型预测的都是 $y = \log_{10}(\text{反应因子})$，从不是原始因子：原始因子跨越四个数量级的范围，
意味着它的方差被最大的那些值主导，所以在原始尺度上算出的损失，几乎会把全部梯度都花在少数几个巨大的测量值
上；而在对数尺度上，重复测量的噪声是同方差的（标准差大致恒定在 $0.1$，与量级无关——对数单位上的加性噪声，
对应原始尺度上的乘性噪声），这恰好是平方误差损失所假设的情形。

1. **全局均值。** 对每一对都预测 $\hat y = \bar y$，完全忽略分子身份，用均方误差
   $\mathrm{MSE} = \frac1n \sum_i (y_i - \bar y)^2$ 拟合和评分。无法区分任何两个分子对；它存在的意义是钉住
   $R^2 = 0$ 这个下面每个模型都必须超越的下限。
2. **逐分子加性模型。** $\hat y(a, b) = \mu + b_a + b_b$，每个分子 id 一个标量偏置，用普通最小二乘拟合。捕
   捉的是“这个分子平均而言对它的各个伙伴有多活跃”，捕捉不到任何关于*这具体一对*的信息。只对训练时出现过的分
   子有定义（一个在所有训练行里都从未出现过的分子，其偏置按构造为 $0$，因为根本没有数据能拟合它），所以对含
   一个这种分子的分子对，这个模型会直接忽略新的那一侧；对两个都是这种分子的分子对，它会退化成模型 1 那个毫无
   用处的全局均值——下面的检查会直接测出这个失效。
3. **矩阵分解。** $\hat y(a, b) = \mu + b_a + b_b + u_a^\top u_b$，加上一个按分子学出的低维隐向量及其内积——
   用上一节的说法，就是 $S = I$ 的一个对称双线性型——用来捕捉超出两个分子各自平均值之外、这具体一对特有的相
   互作用。它仍然只从身份信息里训练出来，所以继承了加性模型对任何训练集之外分子的盲区，但在*已测量过*的分子
   之间（用途 (i)）严格更丰富，因为 $u_a^\top u_b$ 能表达某一具体分子对反应得异常好或异常差，超出仅凭每个分
   子自己的平均值所能预测的部分。
4. **对称指纹特征上的梯度提升树。** 直接从 SMILES 为每个分子算出一个指纹 $h(\cdot)$——不像模型 2–3 那样需要
   训练出按分子的参数——再把它的一个对称配对函数，例如 $[h(a) + h(b),\ h(a) \odot h(b)]$，喂给梯度提升树，
   用平方误差训练（或者用 Huber 损失，只要 EDA 标记出的异常值或单位错误还没有被彻底清理干净，Huber 损失就
   是更稳妥的默认选择，因为残差超过某个阈值之后，它的损失是线性增长而不是平方增长，梯度因此有界）。因为
   $h$ 完全来自结构，这个模型对任何有 SMILES 字符串的分子都有定义，不论它是否在训练集中出现过——这正是用途
   (ii) 需要、而模型 2–3 所缺少的性质；下面的检查用同样这组对称特征上的岭回归（ridge regression）确认：这样
   的模型在分子不相交切分上保留了大部分准确度，而模型 2 在那里退化成全局均值。
5. **图神经网络或 Transformer 配对编码器。** 把固定的指纹换成一个*学出来*的分子编码器——直接读取分子图的图
   神经网络，或者把 SMILES 字符串当作 token 序列读取的 Transformer——配合上文特征与表示一节里的对称配对头，
   端到端地用平方或 Huber 损失训练。比模型 4 严格更有能力，因为编码器学到的是哪些子结构对这个具体目标真正重
   要，而不是依赖一个固定、手工设计的哈希，代价是需要足够多的训练分子对才能拟合出这个编码器，还需要下文深入
   话题里讲到的额外注意（给 Transformer 用规范化或增强过的 SMILES；图神经网络则不需要额外处理）。

### 评估

**指标。** 报告 $y = \log_{10}(\text{反应因子})$ 的均方根误差——它在各个切分之间、以及与下文噪声下限之间都
可以直接比较，而在原始因子上算出的指标做不到——以及每个切分内预测值与真实 $y$ 之间的 Spearman 秩相关，这个
指标独立于 RMSE 也有用，因为下文深入话题里闭环那个用途主要需要候选分子对被正确排序，而不需要被精确预测到具
体数值。

**噪声下限。** 数据与 EDA 一节测出的重复测量标准差（$\log_{10}$ 尺度上约 $0.1$）是加在每一次测量上的噪声的
标准差，测试集里的每一条也不例外。对于一个按“真实值 $+$ 标准差为 $\sigma$ 的独立噪声”生成的测试标签，即便
是真实值的完美预测器，其期望均方误差也恰好是 $\sigma^2$——RMSE 不可能靠一个更好的模型被压到 $\sigma$ 以下，
只能靠一次更好的测量。下面的检查在合成数据上拟合了可能的最好模型——在生成标签所用的隐藏特征上做最小二乘——
它的测试 RMSE 落在 $\sigma$ 的 $5\%$ 以内。每一个 RMSE 数字，都应该对照这个下限来读，而不是对照零。

**切片。** 至少按切分类型（按行随机、单侧未见、双侧未见、按骨架）拆开报告每一个指标，这样一个混在一起的数字
就不会掩盖“模型看起来不错只是因为大多数测试行都是容易的、已经测量得很充分的那种”这个事实；再按分子对里*测量
较少*的那个分子有多少训练测量拆开报告——伙伴多还是伙伴少，对应数据与 EDA 里那条重尾——因为这正是下文主动学习
闭环试图推动一个分子往前移动的那根轴。

### 深入话题

**(a) 提高对测量历史很少或完全没有的分子的泛化能力。** 基于结构的特征已经是上面模型 4–5 的要点；在此之上，
一个在规模大得多、弱标注甚至无标注的分子语料上*预训练*、再在这个数据集的 $200{,}000$ 行上微调过的分子编码
器，会给一个域内测量很少甚至为零的分子提供一个比只从这个数据集本身学到的编码器更好的起点表示。正则化会把模
型从过拟合数据与 EDA 那条重尾里识别出的那些测量充分的枢纽分子上拉回来。集成——在自助重采样或不同随机种子上
训练若干个模型——把它们之间的分歧变成一个不确定性信号：如果集成对某个全新分子的意见分歧很大，说明模型其实
并没有真正学到关于它的任何可靠信息，这个信号直接喂给 (b) 里的主动学习闭环。

**(b) 闭合循环：选择接下来该测量什么。** 把模型用于用途 (ii)，是一个主动学习问题，而不是一个单纯的预测问
题：目标不是在任何地方都预测得准，而是尽可能快地选出能改进模型、或者能找到高价值分子对的测量——这一点很关
键，因为需求与规模一节已经指出，分子对空间里迄今为止只有大约 $0.1\%$ 被测量过。*置信上界*（upper
confidence bound）规则按 $\hat y + \kappa \hat\sigma$ 给候选排序，其中 $\hat y$ 是预测值，$\hat\sigma$ 是一个
校准过的不确定性估计（来自 (a) 的集成分歧，校准方式见下文最后一条追问），$\kappa$ 是一个常数，权
衡着利用那些已经被预测为高价值的分子对，还是探索那些模型仅仅是不确定的分子对。*期望改进*（expected
improvement）则按测量一个候选分子对预期能把当前已知最好的值提高多少来给它加权，当当前最好的值已经很高、几
乎没有候选能合理地超过它时，这种做法比置信上界更保守。两种规则都会造成同一个下游问题：下一轮测量被有意地偏
向这一轮模型认为有希望、或者拿不准的那些分子对——所以在合并后的数据上重新训练出来的模型，相对最初那些行，训
练用的是一个*发生了偏移*的分布——这正是上文数据切分部分警告过的
那种采样偏差风险，只不过这次是自己造成的；值得持续监控的做法是检查新标注的一批批数据，是不是还在指纹空间里
同一个区域反复出现（说明在利用一个可能狭窄、甚至虚假的高价值区域），还是在不断转向此前覆盖不足的分子（这正
是不确定性那一项本该起到的效果）。

**(c) 自注意力及其开销。** 一个 Transformer 层把长度为 $L$ 的一串 token 嵌入投影成查询、键、值，
$Q = XW_Q$，$K = XW_K$，$V = XW_V$，并计算
$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(QK^\top / \sqrt{d_k}\right) V$：每个位置都要通过这个
$L \times L$ 的分数矩阵 $QK^\top$ 去关注其他每一个位置。构造并对这个矩阵做 softmax 归一化的开销是时间
$O(L^2 d_k)$、内存 $O(L^2)$，随序列长度呈平方增长——对一个 SMILES 字符串而言，$L$ 就是这个字符串（或
token）的长度，所以一个长的 SMILES 字符串用这种方式编码的开销会不成比例地更高，不像一个图神经网络那样，其
每层开销随化学键的数目增长，而不是随原子数目的平方增长。

**(d) 为什么一个 SMILES Transformer 对分子的写法很敏感，以及两种修复办法。** 自注意力没有任何内置机制去知
道一个字符串里的某个 token，和另一种写法的字符串里的某个 token，代表的是同一个原子：同一个分子的两个句法上
不同、但化学上完全相同的 SMILES 字符串（从不同的原子开始遍历，或者以不同的顺序写出分支）会产生完全不同的
token 序列，架构里没有任何东西强迫这两者得到相同的输出嵌入——不像图神经网络，它的输入就是图本身，而不是图的
某一种任意线性化写法。两种修复办法：编码之前总是先把 SMILES 字符串规范化，让每个唯一的分子恰好对应一个确定
的字符串，从根源上去掉这种歧义，而不是留给模型自己去学着忽略它；或者*随机化 SMILES 增强*——训练时对每个分子
都随机重新采样、使用许多不同的合法非规范字符串——让模型通过数据学会对这种选择近似不变，而不是依靠预处理上的
保证。规范化更便宜也更精确；增强需要更多训练时间，但也起到一种通用正则化的作用，即便规范化本身有 bug、或者
规范化算法在不同库版本之间发生了变化，增强仍然有用。

**(e) 为什么图神经网络的读出与原子顺序无关。** 图神经网络根据图的连接关系为每个原子（节点）算出一个表示，
然后用一个*读出*（readout）步骤把它们全部汇聚——求和或取平均——成整个分子的一个定长向量。每一层消息传递
（message passing）都用原子自身的向量，加上对其邻居向量的求和、平均或最大值，来更新这个原子，并且所有原子共
用同一组权重，所以对原子重新编号只会打乱逐原子向量列表的顺序，而不会改变其中任何一个向量（这些层是置换等变
的，permutation-equivariant）。求和与平均都是对其输入对称的函数：加法满足交换律和结合律，所以重新排列被求和
或取平均的各项，结果都不会变。不管图里的原子作为输入恰好按什么顺序编号或列出，这个把它们全部汇聚起来的读出步
骤，产出的都是完全相同的嵌入——这里的不变性是架构上就保证的，由置换等变的各层与所选的汇聚函数共同决定，不像
(d) 里那样需要靠训练数据去教给 Transformer。

**(f) 重尾目标上的 $L_1$ 对比 $L_2$ 损失。** 平方误差关于预测值的梯度随残差线性增长，所以一个特别大的误差
就能主导总梯度，把拟合方向拉去优先减小那一个点的误差，牺牲的是大多数普通点——这在一个重尾的原始目标上是真实
的风险。绝对误差的梯度大小恒定，与残差大小无关，所以一个异常值对拟合的拉力并不比一个典型点更大，代价是这个
损失在残差为零处不光滑，并且它拟合的是条件中位数，而不是条件均值。对
$\log_{10}(\text{反应因子})$ 建模，本身就已经把最极端的原始尺度取值拉了回来（需求与规模一节把原始尺度上四
个数量级的范围，变成了对数尺度上一个紧凑的范围），所以在对数尺度上用平方误差，通常已经是一个够用的默认选
择；EDA 没能发现的那些真正的单位错误异常值，比起把两种损失都推向各自的极端，更适合用 Huber 损失来处理——残
差小时是平方误差，超过某个阈值之后变成线性。

### 追问

- **不对称反应。** 如果后来的实验发现顺序其实是有影响的——比如其中一个分子是加入到底物里的催化剂，先加哪个
  会改变结果——上面每一个与顺序无关的组件都需要被替换或者补充：配对函数不再强制
  $f(h(a), h(b)) = f(h(b), h(a))$，数据切分与泄漏一节里那个分子对键的规范化，必须按“谁是催化剂”来规范化，
  而不是把两种顺序直接合并；而重复测量噪声的估计，也必须先确认哪些重复测量是真正意义上的重复（顺序相同），
  哪些其实是方向不同、结果本就该不同的另一种测量。
- **预测三个分子。** 一个为分子对设计的模型不会自动推广到三元组：对称组合函数的自然推广，是一个对全部三个
  输入都对称的函数——比如把每一对之间的双线性项都加起来，$\sum_{i<j} h_i^\top S h_j$，再加上每个参与者各自
  的单分子项——但对一个固定分子集合而言，可能的组合数量会随着组数以组合数的方式增长，所以在分子对这一级已
  经出现的稀疏性（大约只有 $0.1\%$ 的分子对空间被测量过）到了三元组会严重得多，模型也就更加依赖深入话题
  (a) 里那种基于结构的泛化，而不是任何依赖身份的手段。
- **多任务学习。** 如果实验室在同样这些分子上还做了另一项相关的检测——比如溶解度或稳定性——而且做得比这个
  两两反应频繁得多，那么在这个辅助任务和当前任务之间共享分子编码器、只用各自独立的输出头联合训练，能让样本
  量更大的辅助任务教会这个共享编码器一个更好的通用分子表示，直接惠及那些反应数据本身服务得不够好的、测量不
  充分的分子。
- **检测外推。** 除了集成本身的分歧（深入话题 (a)）之外，还有一个直接的检查：候选分子对里的分子，到训练集
  中实际见过的最近分子之间，在指纹或嵌入空间里的距离。一个最近训练邻居仍然很远的候选，是在被外推、而不是被
  内插；同一个距离本身也是 (b) 里主动学习闭环所消费的不确定性估计的一个有用的附加特征。
- **校准不确定性。** 集成的原始分歧（深入话题 (a)）并不会自动就是一个校准好的区间，而 (b) 里的置信上界规则
  恰恰需要它是：留出另一份已经测量过、且从未用来拟合或挑选集成成员的数据切片，检查它的名义预测区间是否确实
  以大致名义的比例包含真实值，如果系统性地不符，就重新缩放这个区间。

<details>
<summary>估算核对（可运行）</summary>

```python
import math
from math import comb

import numpy as np
from sklearn.linear_model import Ridge

# ---- requirements-and-scale numbers ----
n_molecules_real = 20_000
n_rows_real = 200_000

n_possible_pairs = comb(n_molecules_real, 2)
assert n_possible_pairs == 199_990_000
frac_measured_pct = 100 * n_rows_real / n_possible_pairs
assert round(frac_measured_pct, 2) == 0.10

avg_partners = 2 * n_rows_real / n_molecules_real   # each row has 2 endpoints
assert avg_partners == 20.0

fp_bits = 2_048
fp_store_bytes = n_molecules_real * fp_bits // 8
assert fp_store_bytes == 5_120_000
fp_store_mb = fp_store_bytes / 1e6
assert fp_store_mb == 5.12

replicate_noise_log10 = 0.1   # given: the spread across repeated measurements of the same pair

print("all requirements-and-scale numbers check out")


# ---- synthetic experiment: hidden molecular features, observed only through noisy binary fingerprints ----
# NOTE: 300 molecules / 4,000 pairs is a small stand-in for the real 20,000 / 200,000 scale -- enough to
# make the qualitative effects below stable, not a re-creation of the real dataset's own numbers.
rng = np.random.default_rng(0)

n_molecules, dim = 300, 6
x = rng.normal(size=(n_molecules, dim))                    # each molecule's hidden, never-observed features

g_coef = rng.uniform(0.5, 1.5, size=dim)                    # the single-molecule term g(x) = x . g_coef
w_diag = rng.uniform(0.3, 0.9, size=dim)
off = rng.normal(scale=0.15, size=(dim, dim))
off = (off + off.T) / 2                                     # NOTE: symmetrise before zeroing the diagonal,
np.fill_diagonal(off, 0.0)                                  # so W stays exactly symmetric (W == W.T) below
W = np.diag(w_diag) + off
assert np.allclose(W, W.T)                                  # the interaction matrix the text calls symmetric W

n_bits_per_dim, fp_noise = 4, 0.5
noise_bits = rng.normal(scale=fp_noise, size=(n_bits_per_dim, n_molecules, dim))
bits = (x[None, :, :] + noise_bits > 0).astype(np.float64)   # a noisy binary reading of each hidden coordinate
fingerprint = bits.transpose(1, 0, 2).reshape(n_molecules, n_bits_per_dim * dim)   # the observed "fingerprint"

all_a, all_b = np.triu_indices(n_molecules, k=1)             # every possible unordered pair, in a fixed,
n_pairs = 4_000                                              # deterministic order -- never a Python set, so
chosen = rng.permutation(len(all_a))[:n_pairs]               # results cannot depend on hash seed or iteration
mol_lo, mol_hi = all_a[chosen], all_b[chosen]                # order

swap = rng.integers(0, 2, size=n_pairs).astype(bool)         # which molecule the raw file happens to write
mol1 = np.where(swap, mol_hi, mol_lo)                        # first is arbitrary per row, exactly as the
mol2 = np.where(swap, mol_lo, mol_hi)                        # premise states


def pairwise_interaction(xa, xb):
    return np.einsum("ij,jk,ik->i", xa, W, xb)


true_log10 = x[mol1] @ g_coef + x[mol2] @ g_coef + pairwise_interaction(x[mol1], x[mol2])
# NOTE: true_log10 depends only on the unordered {mol1, mol2} pair -- swapping which one is "first" above
# leaves it unchanged, matching the premise that the factor itself does not depend on which name is written
# first.

sigma_noise = replicate_noise_log10
y = true_log10 + rng.normal(scale=sigma_noise, size=n_pairs)

noise_floor_rmse = math.sqrt(np.mean((y - true_log10) ** 2))   # the realised noise: the best any model could do
assert abs(noise_floor_rmse - sigma_noise) < 0.02               # matches the sigma used to generate the data

fp1, fp2 = fingerprint[mol1], fingerprint[mol2]
X_sym = np.concatenate([fp1 + fp2, fp1 * fp2], axis=1)        # order-invariant: unchanged by swapping fp1, fp2
X_ord = np.concatenate([fp1, fp2], axis=1)                    # order-dependent: swapping moves values across halves

X_additive = np.zeros((n_pairs, n_molecules))
X_additive[np.arange(n_pairs), mol1] += 1.0
X_additive[np.arange(n_pairs), mol2] += 1.0                    # row @ b = b_mol1 + b_mol2; mu is added separately


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def fit_additive_least_squares(train_mask, test_mask):
    """mu + b_a + b_b: mu is the training mean and the b's are fit to the residuals by ordinary least
    squares. A molecule absent from every training row has an all-zero column there, so lstsq's
    minimum-norm solution leaves its b at 0: a pair of two such molecules is predicted as mu, the global
    mean of model 1."""
    mu = y[train_mask].mean()
    coef, *_ = np.linalg.lstsq(X_additive[train_mask], y[train_mask] - mu, rcond=None)
    pred = mu + X_additive[test_mask] @ coef
    return r2_score(y[test_mask], pred)


def fit_ridge(features, train_mask, test_mask, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(features[train_mask], y[train_mask])
    pred = model.predict(features[test_mask])
    return r2_score(y[test_mask], pred), model


# random row split: molecule-disjointness is not enforced, so most test molecules also appear in training
row_order = rng.permutation(n_pairs)
row_test = np.zeros(n_pairs, dtype=bool)
row_test[row_order[: int(0.2 * n_pairs)]] = True
row_train = ~row_test

# molecule-disjoint split: assign MOLECULES, not rows, to train/test; drop rows straddling both folds
mol_order = rng.permutation(n_molecules)
mol_is_test = np.zeros(n_molecules, dtype=bool)
mol_is_test[mol_order[: int(0.3 * n_molecules)]] = True
row1_test, row2_test = mol_is_test[mol1], mol_is_test[mol2]
disjoint_train = (~row1_test) & (~row2_test)     # both molecules in the training fold
disjoint_test = row1_test & row2_test            # both molecules in the test fold (the "both-unseen" case);
                                                  # a row with one molecule on each side lands in neither mask

add_random_r2 = fit_additive_least_squares(row_train, row_test)
add_disjoint_r2 = fit_additive_least_squares(disjoint_train, disjoint_test)
sym_random_r2, sym_model = fit_ridge(X_sym, row_train, row_test)
sym_disjoint_r2, _ = fit_ridge(X_sym, disjoint_train, disjoint_test)
_, ord_model = fit_ridge(X_ord, row_train, row_test)

print(f"additive model:     random-split R^2={add_random_r2:.2f}, molecule-disjoint R^2={add_disjoint_r2:.2f}")
print(f"symmetric-fp model: random-split R^2={sym_random_r2:.2f}, molecule-disjoint R^2={sym_disjoint_r2:.2f}")

assert add_random_r2 > 0.5                     # (a) the additive model does well on the random row split...
assert add_disjoint_r2 < 0.05                  # ... but no better than predicting the mean once split by molecule
assert sym_disjoint_r2 > 0.4                   # (b) the symmetric-fingerprint model keeps real accuracy there...
assert sym_disjoint_r2 > 0.6 * sym_random_r2   # ... at least 60% of its own random-split R^2...
assert sym_disjoint_r2 - add_disjoint_r2 > 0.3   # ... clearly ahead of the additive model on the hard split

# (c) order sensitivity: predict both orders of every held-out pair and compare
pred_ord_fwd = ord_model.predict(X_ord[row_test])
pred_ord_bwd = ord_model.predict(np.concatenate([fp2[row_test], fp1[row_test]], axis=1))
mean_abs_diff_ord = np.mean(np.abs(pred_ord_fwd - pred_ord_bwd))

pred_sym_fwd = sym_model.predict(X_sym[row_test])
X_sym_swapped = np.concatenate([fp2[row_test] + fp1[row_test], fp2[row_test] * fp1[row_test]], axis=1)
pred_sym_bwd = sym_model.predict(X_sym_swapped)
mean_abs_diff_sym = np.mean(np.abs(pred_sym_fwd - pred_sym_bwd))

print(f"mean |pred(a,b) - pred(b,a)|: ordered features={mean_abs_diff_ord:.3f}, "
      f"symmetric features={mean_abs_diff_sym:.3f}")
assert mean_abs_diff_ord > 0.05    # a real, order-dependent shift, in log10 units
assert mean_abs_diff_sym < 1e-9    # exactly zero (to floating point) once features are order-invariant

# (d) the noise floor: even the best model possible -- least squares on the hidden features, in the exact form
# of the true function (x_a + x_b, plus every symmetric product x_a[j] x_b[k] + x_b[j] x_a[k]) -- only gets
# down to sigma on held-out rows, while the fingerprint model stays far above it
xa, xb = x[mol1], x[mol2]
j_idx, k_idx = np.triu_indices(dim)
sym_products = xa[:, j_idx] * xb[:, k_idx] + xb[:, j_idx] * xa[:, k_idx]
X_oracle = np.column_stack([np.ones(n_pairs), xa + xb, sym_products])
beta, *_ = np.linalg.lstsq(X_oracle[row_train], y[row_train], rcond=None)
oracle_rmse = math.sqrt(np.mean((y[row_test] - X_oracle[row_test] @ beta) ** 2))
sym_random_rmse = math.sqrt(np.mean((y[row_test] - pred_sym_fwd) ** 2))
print(f"noise floor RMSE={noise_floor_rmse:.3f} (sigma used to generate the data={sigma_noise}); "
      f"best possible model's test RMSE={oracle_rmse:.3f}; "
      f"symmetric-fingerprint model's test RMSE={sym_random_rmse:.3f}")
assert 0.95 * sigma_noise < oracle_rmse < 1.05 * sigma_noise   # the best model possible lands on the floor...
assert sym_random_rmse > oracle_rmse                          # ... and the fingerprint model sits above it

print("all checks passed")
```

</details>

</details>
