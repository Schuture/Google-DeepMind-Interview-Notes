<div align="center">

# Google DeepMind 面试题笔记

**按 Google DeepMind 的编程、机器学习知识问答、机器学习设计、研究方向和行为面整理的练习题，外加一份招聘与面试流程指南。<br>
完整的题面、详细的参考解答，以及每次推送都由 CI 自动运行的代码。**

[English](README.md) · 中文

[![Check](https://github.com/Schuture/Google-DeepMind-Interview-Notes/actions/workflows/check.yml/badge.svg)](https://github.com/Schuture/Google-DeepMind-Interview-Notes/actions/workflows/check.yml)
![Problems](https://img.shields.io/badge/problems-37-blue)
![Languages](https://img.shields.io/badge/languages-English%20%7C%20%E4%B8%AD%E6%96%87-blue)
[![Text: CC BY-NC 4.0](https://img.shields.io/badge/text-CC%20BY--NC%204.0-lightgrey)](LICENSE)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE-CODE)
[![GitHub stars](https://img.shields.io/github/stars/Schuture/Google-DeepMind-Interview-Notes?style=social)](https://github.com/Schuture/Google-DeepMind-Interview-Notes/stargazers)

</div>

> [!NOTE]
> **非官方项目。** 本项目与 Google DeepMind 或 Google 没有任何关联，未获其认可或赞助。题目主要根据大约最近一年的二手面试经历重建，
> 另有少量至今仍反复出现的较早题型，题面、示例、解答和代码全部重新撰写。公开的 Google DeepMind 面试细节比一些公司少得多，很多经历只提到题目的方向
> （“BFS 的困难变种”“某个大模型核心概念背后的数学”），这类页面是围绕该方向原创的题目。请把每一页当作对同类题目的练习，
> 而不是原题。如果你认为某些内容不应公开，请[提一个 issue](https://github.com/Schuture/Google-DeepMind-Interview-Notes/issues)，
> 相关内容会被下架。

## 内容概览

| 分类 | 页数 | 覆盖内容 |
| --- | ---: | --- |
| [招聘与面试流程](PROCESS.zh.md) | 1 | 进入 Google DeepMind 的两条路径，从 recruiter 电话到 offer 的每一轮，旧式 quiz 的变化，时间线，公开的薪资区间，Student Researcher 项目，以及面试中使用 AI 工具的规则 |
| [编程](#编程-17) | 17 | 多种形式的图搜索（基于边列表的深度优先遍历热身题、带最短路计数的状态空间 BFS、图上的博弈、单位换算图），Python 生成器、单元测试与加权采样，流式频率统计，双调数组与单调函数上的二分查找，位打包，快照，k-d 树，代码评审轮，以及机器学习实现：带 KV cache 的注意力、反向传播、EM、focal loss、调试训练循环 |
| [知识问答](#知识问答-7) | 7 | 口述的知识问答轮：机器学习基础、大模型基础、大模型强化学习，两页数学（概率、统计与线性代数；矩阵计算、统计检验与信息论），计算机基础，以及讲清一段代码在计算什么 |
| [系统设计](#系统设计-8) | 8 | Gemini App、大模型助手的评测系统、预测分子对的反应因子、检索增强智能体、推荐信息流、单卡放不下的模型的训练方案、机器人学习的数据飞轮，以及预测数据中心哪些机器需要更换 |
| [行为面](#行为面-5) | 5 | HR 初筛、研究深挖、研究报告与论文答辩、Googleyness 与文化、团队负责人与用人经理面，以及产品经理面 |

这些页面的特点：

- **完整的题面。** 每道题都写全了定义、函数签名和逐步演算的示例，拿来就能做，不用猜题意。大多数题目分成三个逐步递进的部分，
  和真实面试的节奏一致。
- **可以验证的解答。** 每道编程题、知识问答题和系统设计题的解答末尾都有一个折叠的、可运行的检查块：针对示例的断言，
  只依据题面独立写出的暴力解（能写的都写了），对每个推导的数值验证；系统设计题是用代码重算的估算。每次推送，CI 都会运行全部页面的代码。
- **易错点写在出错的地方。** 不单列一张易错清单，而是在最容易写错的那一行加 `# NOTE:` 注释。
- **把口述的知识问答写下来。** [知识问答](#知识问答-7)页按提问的方式给出每个问题，按口头回答的方式给出答案，
  再附上推导和代码检查。
- **不只有题目，还有流程。** [招聘与面试流程](PROCESS.zh.md)逐轮区分了 Google DeepMind 官方公开的信息（附链接）和候选人的反馈。
- **学习路线。** [学习路线](ROADMAP.zh.md)按研究方向（RS / RE）、软件工程、应用 AI 与机器学习工程、Student Researcher 与实习、
  产品经理五条路线给页面排好了顺序。
- **中英双语。** 每一页都有中英文两个版本，代码完全一致。

## 怎么用

### 1. 先读流程指南

从[招聘与面试流程](PROCESS.zh.md)开始：你从哪条路径进入（Google DeepMind 自己的岗位，还是 Google 的通用流程再 team match），
这意味着哪几轮，要花多长时间。“为什么选 DeepMind”几乎每一轮都会被问到，所以行为面的几页值得尽早开始准备。

### 2. 选路线，定节奏

从[学习路线](ROADMAP.zh.md)里和你岗位对应的那条往下做。★ 表示题目出现的频率，从 ★★★★★（反复出现）到 ★☆☆☆☆（少见），
每条路线都按收益排好了序，越靠前越值得先做。

| 可用时间 | [研究方向（RS / RE）](ROADMAP.zh.md#研究方向rs--re) | [SWE](ROADMAP.zh.md#软件工程swe) | [Applied AI / MLE](ROADMAP.zh.md#应用-ai-与机器学习工程applied-ai--mle) | [Student Researcher](ROADMAP.zh.md#student-researcher-与实习) |
| --- | --- | --- | --- | --- |
| 一周左右 | 第 1–3 阶段 | 第 1–2 阶段 | 第 1–3 阶段 | 第 1 阶段 |
| 两到四周 | 第 1–6 阶段 | 整条路线 | 整条路线 | 整条路线 |
| 更长 | 加上第 7 阶段，以及目标团队方向的一道设计题 | 加上机器学习知识问答页 | 加上研究路线的机器学习编程题 | 加上研究路线的知识问答页 |

### 3. 练一道题

**编程题。** 先只读**题目**一节。动手写代码之前，记下你会向面试官确认的问题（边界情况、并列时怎么处理、输入规模），
再和参考解答开头的几行对照，那里列出了值得确认的点。给自己设一个时限，按顺序做各个部分，把每个新部分当作一次需求变更：
在已有代码上扩展，而不是推倒重来。先在示例上跑通，再展开参考解答，对照思路、复杂度和 `# NOTE:` 注释。
检查代码调用的是题面里给出的函数名，所以只要保持相同的函数名和签名，一般就能拿它来测你自己的实现。

**知识问答。** 每道题给自己三分钟左右，大声作答，题目要求推导的就在纸上写出来。然后和参考答案对照：
答案先给出口头会说的内容，再给出推导过程；检查代码会把每个数值重算一遍。

**系统设计题。** 给自己 45 分钟左右，出声讲或在文档里写：需求、估算、API 与数据模型、架构，然后选两三个点深入。
最后和参考解答对照；估算都在检查块里用代码算过，可以改一个输入再跑一遍。

**行为面。** 每页列出这一轮问什么、每个问题考察什么、好的回答怎样组织，末尾有一份提纲，用你自己的经历填写。
大声练几遍，填好的版本放在 `my/` 目录里（git 会忽略它）。

### 4. 一页的结构

每页开头是一张表（题型、优先级、难度、岗位、考点，已知的话还有形式和轮次），正文只有两节：

| 章节 | 编程题 | 知识问答 | 系统设计题 | 行为面 |
| --- | --- | --- | --- | --- |
| **题目** | 定义，然后每个部分一块：任务、函数签名、示例 | 按主题分组、依次编号的问题 | 要设计的系统、用户、给定的数字、范围 | 这一轮的形式和问题 |
| **参考解答**（折叠） | 按部分写思路、推导、代码，易错点是代码里的 `# NOTE:` 注释；最后是追问和检查代码 | 逐题给出口头回答，再给推导；最后是检查代码 | 需求、数据模型与 API、架构、深入话题、追问、估算核对 | 考察点、回答的结构、待填写的提纲 |

岗位缩写：**RS** 研究科学家 · **RE** 研究工程师 · **SWE** 软件工程师 · **MLE** 机器学习工程师 ·
**Applied AI** 应用 AI 工程师 · **Intern** 实习与 Student Researcher 岗位 · **PM** 产品经理 · **TPM** 技术项目经理。

### 5. 运行代码

```bash
git clone https://github.com/Schuture/Google-DeepMind-Interview-Notes.git
cd Google-DeepMind-Interview-Notes
pip install -r requirements.txt                              # NumPy、SciPy、scikit-learn
python scripts/run_snippets.py coding/state-space-bfs        # 一页
python scripts/run_snippets.py --all                         # 全部页面
```

一页里的 ```` ```python ```` 代码块按顺序作为一个脚本运行；```` ```py ```` 代码块只作示意（单独的函数签名、含有预埋 bug 的代码），
不会执行。CI 使用 Python 3.11。

## 题目

每个分类内按优先级排序。难度一栏为 — 表示未评定。

<!-- index:begin -->
### 编程 (17)

| # | 题目 | 优先级 | 难度 | 岗位 | 考点 |
| ---: | --- | --- | --- | --- | --- |
| 1 | [穿过上锁的门：最短路径与最短路计数](coding/state-space-bfs/README.zh.md) | ★★★★★ | 困难 | SWE · RE · MLE · Intern | bfs, state-space-search, bitmask, path-counting, dijkstra, grid |
| 2 | [事件流中出现最频繁的事件](coding/event-frequency-stream/README.zh.md) | ★★★★☆ | 中等 | SWE · MLE · Applied AI · Intern | hash-map, sliding-window, frequency-counting, streaming, space-saving, heavy-hitters |
| 3 | [移动棋子游戏中的必胜态](coding/token-game-outcomes/README.zh.md) | ★★★★☆ | 困难 | SWE · RE · MLE · Intern | game-theory, dfs, retrograde-bfs, topological-order, sprague-grundy, graphs |
| 4 | [带 KV cache 与在线 softmax 的注意力](coding/attention-kv-cache/README.zh.md) | ★★★★☆ | 困难 | RS · RE · MLE | attention, kv-cache, online-softmax, flash-attention, grouped-query-attention, numerical-stability |
| 5 | [调试一个学不会的分类器](coding/training-loop-debug/README.zh.md) | ★★★★☆ | 中等 | RE · RS · MLE · Applied AI | debugging, softmax, data-shuffling, gradient-scaling, momentum, dropout, broadcasting, sanity-checks |
| 6 | [热身题：数组扫描与图的深度优先遍历](coding/graph-dfs-warmup/README.zh.md) | ★★★★☆ | 简单 | SWE · MLE · Intern | arrays, binary-search, graph-construction, dfs, iterative-dfs, connected-components, cycle-detection |
| 7 | [Python 生成器、单元测试与加权采样](coding/weighted-sampling-generator/README.zh.md) | ★★★☆☆ | 中等 | MLE · SWE · RE · Intern | generators, unit-testing, weighted-sampling, prefix-sums, binary-search, alias-method, chi-square-test |
| 8 | [代码评审：给一次 checkpoint 改动中的缺陷排序](coding/code-review-round/README.zh.md) | ★★★☆☆ | 中等 | RE · SWE · MLE | code-review, checkpointing, atomic-writes, reproducibility, data-sharding, testing |
| 9 | [把单位换算建成带权图](coding/unit-conversion-graph/README.zh.md) | ★★★☆☆ | 中等 | SWE · MLE · RE · Intern | graph, dfs, bfs, weighted-union-find, consistency-check, floating-point |
| 10 | [用最少的笔画刷完栅栏](coding/fence-painting-strokes/README.zh.md) | ★★★☆☆ | 中等 | SWE · MLE · Intern | greedy, divide-and-conquer, arrays, range-minimum, proof-of-optimality |
| 11 | [位打包编码器与解码器](coding/bit-packing-codec/README.zh.md) | ★★★☆☆ | 中等 | SWE · RE · Intern | bit-manipulation, varint, zigzag-encoding, frame-of-reference, serialisation |
| 12 | [快照数组：历史、压缩与差异](coding/snapshot-array/README.zh.md) | ★★★☆☆ | 中等 | SWE · RE · MLE · Intern | hash-map, binary-search, versioning, memory-trade-offs, journaling |
| 13 | [k-d 树：最近邻与范围查询](coding/kd-tree-nearest-neighbour/README.zh.md) | ★★★☆☆ | 困难 | RS · RE · SWE · MLE | kd-tree, nearest-neighbour, pruning, heap, range-search, curse-of-dimensionality |
| 14 | [从零实现反向传播](coding/mlp-backprop-from-scratch/README.zh.md) | ★★★☆☆ | 中等 | RS · RE · MLE · Intern | backpropagation, softmax-cross-entropy, gradient-check, initialisation, sgd-momentum |
| 15 | [高斯混合模型的 EM 算法：推导与实现](coding/em-gaussian-mixture/README.zh.md) | ★★☆☆☆ | 困难 | RS · RE · MLE | expectation-maximisation, gaussian-mixture, log-sum-exp, jensen-inequality, k-means, bic |
| 16 | [Focal loss 与交叉熵](coding/focal-loss/README.zh.md) | ★★☆☆☆ | 中等 | MLE · RS · RE · Applied AI | focal-loss, cross-entropy, class-imbalance, numerical-stability, initialisation, gradients |
| 17 | [双调数组与单调函数上的二分查找](coding/binary-search-monotone/README.zh.md) | ★★☆☆☆ | 中等 | RE · RS · SWE · MLE | binary-search, bitonic-array, exponential-search, monotone-functions, lower-bounds, bisection |

### 知识问答 (7)

| # | 题目 | 优先级 | 难度 | 岗位 | 考点 |
| ---: | --- | --- | --- | --- | --- |
| 1 | [机器学习基础：评估指标、损失函数、优化器与正则化](quiz/ml-fundamentals/README.zh.md) | ★★★★★ | 中等 | RS · RE · MLE · Applied AI · Intern | precision-recall, roc-auc, cross-entropy, logistic-regression, adam, weight-decay, bias-variance, normalisation, huber-loss, gan, distribution-shift |
| 2 | [大模型基础：注意力、Transformer 与训练流程](quiz/llm-fundamentals/README.zh.md) | ★★★★★ | 困难 | RS · RE · MLE · Applied AI · Intern | attention, transformer, rope, kv-cache, scaling-laws, perplexity, tokenisation, post-training, sampling, mixture-of-experts |
| 3 | [大模型强化学习：策略梯度、PPO、GRPO 与 DPO](quiz/rl-for-llms/README.zh.md) | ★★★☆☆ | 困难 | RS · RE · MLE | policy-gradient, ppo, grpo, dpo, kl-regularisation, rlhf, reward-hacking, off-policy, importance-sampling |
| 4 | [数学问答：概率、统计与线性代数](quiz/maths-probability-linear-algebra/README.zh.md) | ★★★☆☆ | 中等 | RS · RE · MLE · Intern | bayes-theorem, expectation, markov-chains, maximum-likelihood, map-estimation, kl-divergence, svd, matrix-calculus, conditioning |
| 5 | [数学问答：矩阵计算、统计检验与信息论](quiz/maths-stats-information-theory/README.zh.md) | ★★★☆☆ | 中等 | RS · RE · MLE · Intern | matrix-multiplication, rank, matrix-inverse, pseudo-inverse, moments, central-limit-theorem, hypothesis-testing, chi-square-test, entropy, mutual-information, integration |
| 6 | [计算机基础问答：内存、并发与浮点数](quiz/cs-fundamentals/README.zh.md) | ★★☆☆☆ | 中等 | SWE · Intern · RE | oop, memory-management, garbage-collection, race-conditions, deadlock, gil, cache-locality, floating-point, amortised-analysis |
| 7 | [代码理解：这段代码在计算什么？](quiz/code-comprehension/README.zh.md) | ★★☆☆☆ | 中等 | RS · RE · Intern · SWE | code-reading, convolution, padding, numerical-stability, streaming-statistics, attention-masks, reservoir-sampling |

### 系统设计 (8)

| # | 题目 | 优先级 | 难度 | 岗位 | 考点 |
| ---: | --- | --- | --- | --- | --- |
| 1 | [设计 Gemini App](system-design/gemini-app/README.zh.md) | ★★★★☆ | 困难 | SWE · MLE · Applied AI | streaming, conversation-storage, context-management, model-routing, safety-filtering, multimodal-uploads, capacity-planning |
| 2 | [为大模型助手设计评测系统](system-design/llm-evaluation-system/README.zh.md) | ★★★★☆ | 困难 | RE · RS · MLE · Applied AI | evaluation, autoraters, side-by-side, bradley-terry, statistical-power, contamination, release-gating |
| 3 | [机器学习设计：预测分子对的反应因子](system-design/molecule-reaction-ml-design/README.zh.md) | ★★★☆☆ | 中等 | MLE · RE · RS | eda, molecular-fingerprints, pairwise-models, symmetry, data-splitting, leakage, graph-neural-networks, active-learning |
| 4 | [设计基于企业文档的检索增强智能体](system-design/rag-agent-platform/README.zh.md) | ★★★☆☆ | 困难 | Applied AI · MLE · SWE | rag, vector-index, hybrid-search, access-control, agents, prompt-injection, evaluation |
| 5 | [为内容信息流设计推荐系统](system-design/recommendation-system/README.zh.md) | ★★★☆☆ | 中等 | MLE · SWE · Applied AI | candidate-generation, two-tower, ranking, multi-task-learning, feedback-loops, cold-start, ndcg |
| 6 | [为单卡放不下的模型设计训练方案](system-design/distributed-training/README.zh.md) | ★★★☆☆ | 困难 | RE · MLE · RS | data-parallelism, tensor-parallelism, pipeline-parallelism, sharded-optimiser, checkpointing, fault-tolerance, loss-spikes |
| 7 | [机器学习系统设计：机器人操作策略的数据飞轮](system-design/robot-learning-data-flywheel/README.zh.md) | ★★☆☆☆ | 困难 | RE · RS · MLE | robot-learning, data-collection, dataset-curation, vision-language-action, evaluation-statistics, deployment-safety |
| 8 | [机器学习设计：预测数据中心哪些机器需要更换](system-design/datacentre-failure-ml-design/README.zh.md) | ★★☆☆☆ | 中等 | MLE · RE · SWE | predictive-maintenance, label-construction, censoring, class-imbalance, categorical-embeddings, survival-analysis, precision-at-k, feedback-loops |

### 行为面 (5)

| # | 题目 | 优先级 | 难度 | 岗位 | 考点 |
| ---: | --- | --- | --- | --- | --- |
| 1 | [HR 初筛：动机、研究兴趣与流程事项](behavioral/recruiter-screen/README.zh.md) | ★★★★★ | — | 全部 | why-gdm, motivation, research-interests, background, visa, logistics, compensation |
| 2 | [研究深挖：论文讨论与研究报告](behavioral/research-deep-dive/README.zh.md) | ★★★★★ | — | RS · RE · Intern | paper-deep-dive, research-talk, experimental-design, ablations, limitations, research-taste |
| 3 | [Googleyness、领导力与文化：胜任力问题](behavioral/googleyness-and-culture/README.zh.md) | ★★★★★ | — | 全部 | star, why-gdm, collaboration, conflict, ownership, ambiguity, leadership, failure, responsibility |
| 4 | [团队负责人与用人经理面试：经历、研究品味与匹配度](behavioral/team-lead-rounds/README.zh.md) | ★★★★☆ | — | RS · RE · SWE · MLE | research-taste, ml-experimentation, ramp-up, team-fit, open-ended-problems |
| 5 | [产品经理面：AI 产品感与 AI 技术深挖](behavioral/pm-ai-product-sense/README.zh.md) | ★★★☆☆ | — | PM | product-sense, ai-product-strategy, agent-metrics, launch-risk, user-insights, ux-for-ai |
<!-- index:end -->

## 参与贡献

欢迎指正错误、补充变体和翻译，也欢迎用你自己的话描述最近的面试经历。发现了错误的答案、遗漏的边界情况、流程指南里过时的说法，
或者读不通的句子？[提一个 issue](https://github.com/Schuture/Google-DeepMind-Interview-Notes/issues)。提交 pull request 前请先看
[CONTRIBUTING.zh.md](CONTRIBUTING.zh.md)。

## 许可证

正文（文字、表格和图示）采用 [CC BY-NC 4.0](LICENSE) 许可；代码（包括页面中的代码和 `scripts/` 下的脚本）采用
[MIT 许可证](LICENSE-CODE)。
