# Google DeepMind 招聘与面试流程

[English](PROCESS.md) · 中文

> [!NOTE]
> **非官方整理，核实于 2026-09-23。** 本指南综合了两类信息。带来源链接的陈述来自 Google DeepMind
> （GDM）或 Google 官方页面——招聘官网、官方面试指南、职位发布与政策文件——核实于上述日期。标题为
> **候选人反馈**的段落，是对面试过 GDM 的人所写公开文章的总结，这些文章主要来自 2024 年底到 2026 年
> 9 月之间，能反映出某一轮面试如何演变的更早报道也会一并采用；这些内容并非官方信息，也不附来源链接，
> 只出现过一次的说法会特别标出。GDM 通过两条不同的渠道招人（见[在哪里申请、如何申请](#在哪里申请如何申请)），
> 各团队的实际流程也不完全一致：以你的招聘联系人的说明和职位发布页面为准，它们的优先级始终高于本页。

## 概览

| 阶段 | 内容 | 时长 | 适用对象 |
| --- | --- | --- | --- |
| [申请](#获得面试机会) | GDM 自己的职位发布，或者 Google 的通用流程再加 team match | — | 所有人 |
| [初步面试](#初步面试) | 一通关于你背景经历的 recruiter 电话，有时会有团队负责人一起参加 | 30 分钟（官方数据） | 所有人 |
| [技能面（Skills）](#技能面skills) | 编程、一场口头的机器学习知识问答，部分岗位还有机器学习编程或调试，以及一场因团队而异的机器学习设计轮 | 两到三通电话（官方数据），每场约一小时 | 大多数流程 |
| [Take-home](#take-home-作业) | 一次限时练习或一次准备好的报告展示 | 几个小时 | 少数路线 |
| [终面](#终面) | 团队负责人与领导层；研究岗位有研究报告或论文讨论；还有行为面 | 好几通电话 | 走到这一步的所有人 |
| [决定与 offer](#决定与-offer) | 招聘团队评审，有时会有团队内部讨论或 team match，然后是 offer | 几天到几周 | 终面之后 |

官方给出的全流程时间线是**4–10 周**
（[面试指南](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)）。
**候选人反馈**这个时间经常更长，轮次之间会沉默好几周，还有好几份报告里整个流程走了三到六个月
（详见[时间线](#时间线)）。

## 获得面试机会

### Google DeepMind 官方看重什么

- “We're searching for people who share our drive to build the next generation of breakthrough AI systems,
  safely and responsibly.”（我们在寻找与我们一样、渴望安全且负责任地构建下一代突破性 AI 系统的人。）
  （[招聘页](https://deepmind.google/about/careers/)）
- 招聘官网列出的岗位序列（role family）：Research Engineer、Software Engineer、Research Scientist、
  Product Manager、Program Manager、Technical Program Manager、Operations and Responsibility。
  （[招聘页](https://deepmind.google/about/careers/)）
- 官方面试指南要求给出具体、简洁的例子，逐步展示推理过程，遇到不会的问题要诚实承认，并建议在面试前
  阅读 GDM 的博客、使用它的产品。
  （[面试指南](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)）
- 无障碍支持：“If you have a disability, need assistive technology, or other extra support during the
  interview process, please let us know.”（如果你有残障情况、需要辅助技术，或者在面试过程中需要其他
  额外支持，请告诉我们。）（[招聘页](https://deepmind.google/about/careers/)）

### 在哪里申请、如何申请

招聘官网上的在招职位列表会跳转到 Google 招聘门户里筛选出的 DeepMind 职位；许多 GDM 职位也会同时发布在
GDM 自己的 [Greenhouse 职位板](https://job-boards.greenhouse.io/deepmind) 上，[职级与薪酬](#职级与薪酬)
一节里符合薪资透明法规的薪资区间也是从这里来的。（[招聘页](https://deepmind.google/about/careers/)）

进入 GDM 有两扇门，它们通向不同的面试：

1. **GDM 自己的职位发布与流程。** 一个为 GDM 团队打的职位广告，走的是 GDM 官方的四个阶段——初步
   面试、技能面（Skills）、终面、决定与 offer——具体内容由团队自己决定。研究科学家和研究工程师岗位，
   以及许多软件岗位，都是走这扇门。
2. **Google 的通用流程，再加 team match。** GDM 团队里的部分产品和基础设施工作（反复出现的例子是
   Gemini app），以及部分实习岗位，是通过 Google 的通用招聘流程招人的，走完流程后再把候选人匹配
   （match）给一个 GDM 团队。这扇门走的面试是 Google 的通用面试。

**候选人反馈：**

- 一位通过 Google 常规 team-match 招聘池的软件工程实习生，同时拿到了一个 GDM 团队（负责 Gemini app
  的 iOS 前端）和一个无关 Google 团队的匹配 offer。
- 一位同时面试了 Mountain View 一个 Google 团队和伦敦一个 DeepMind 团队的候选人反馈，Google 的流程
  是两轮编程、两轮系统设计和一轮行为面，而 DeepMind 的流程明显更难，形式也很不一样。
- 好几条独立的讨论帖都提到，**从 Google 内部岗位转到 GDM，意味着要重新申请 GDM 的职位、重新面试
  一遍**；讨论转岗的员工也说不清这样会不会让绿卡劳工证（labour certification）流程重新开始。同样在
  这些讨论帖里，门槛究竟是“GDM 已经认识的人，或者来自其他前沿实验室的人”存在争议：也有人说见过没有
  博士学位的应届生和背景各异的研究者拿到 offer。
- 据反馈，GDM 的招聘和 Google 其他部分是分开运作的，team match 和薪酬待遇两边并不互通（二手消息，
  出自 2025 年 1 月的一条讨论 offer 的帖子）。
- 裸投网申、招聘联系人主动联系和内部推荐都有人靠它们拿到面试。对研究岗位来说，同一条讨论帖认为，
  认识用人经理帮助很大，而其他推荐渠道帮助不大——这更接近学术界的求职市场，而不是内部推荐机制
  （只是一条讨论帖的说法）。
- 一位拥有博士学位、在 2025 年被拒的研究工程师候选人把门槛总结为：候选人的研究方向要和团队的方向
  高度匹配，并且要在相关领域的顶级会议上发表过七八篇论文——这只是一个人的看法。

### 地点与混合办公

- 招聘官网列出了十个据点：伦敦、湾区、班加罗尔、剑桥（美国）、蒙特利尔、纽约、巴黎、东京、多伦多
  和苏黎世。（[招聘页](https://deepmind.google/about/careers/)）
- “Most colleagues follow our hybrid model – working from the office Tuesday-Thursday and from home, or
  somewhere nearby, on Monday and Friday.”（大多数同事采用的是混合办公模式——周二到周四在办公室，
  周一和周五在家或附近的地方办公。）
  （[面试指南](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)）
- 面试大多是远程进行的：“the majority of our interviews are virtual”（我们的面试大多是线上进行
  的），会“with calendar invites with Google Hangout links for video call interviews”（发送带
  Google Hangout 视频通话链接的日历邀请）。
  （[面试指南](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)）
- 为写这份指南核实过的页面里，没有一个提到签证担保政策；如果这对你很重要，请直接问你的招聘联系人。

### 实习、Student Researcher 与 fellowship 项目

**Student Researcher Program**
（[student researcher program](https://deepmind.google/student-researcher-program/)）：

- “You must be enrolled in a Bachelor's, Master's, or PhD program.”（你必须在读本科、硕士或博士
  项目。）
- “Between 12 and 24 weeks, with a minimum time commitment of four days a week.”（时长 12 到 24
  周，每周至少投入四天。）
- “In-person, at a Google office so you can work directly with your host team.”（需要到 Google
  办公室现场工作，这样才能和接收团队直接共事。）
- 这个岗位是有薪的；页面上没有公布具体津贴数额。
- 申请人会被纳入 Google 各 AI 团队——Google DeepMind、Google Research 或其他 Google 团队——的相关
  岗位一并考虑，所以申请并不保证能进入 GDM。

**候选人反馈**这一轮流程有两种形式。大多数人描述的是两场纯聊研究、不涉及编程或系统设计的对话：先是
大约 45 分钟，和团队里的一位研究者聊候选人自己的研究，然后大约 30 分钟，和可能的接收导师（host）聊
一个可能的项目方向。少数人反馈的是一轮基础知识问答——一份报告里考的是线性回归、正则化、优化方法和
深度学习基础，另一份报告里考的是 PPO 和 GRPO 的区别——随后再进行一次 team-matching 对话。更早期的
研究实习流程（2020–2021 年）是一次 HR 初筛、一小时的技术面试（考数学、机器学习、强化学习和读
代码），然后是几场决定团队是否想要这名候选人的研究面试，最后是一轮文化面试；有些年份，实习岗位的
headcount（招聘名额）在申请开放后几周内就被占满，所以越早申请越好。

**Google PhD Fellowship** 是 Google 全公司范围的项目，并非 GDM 专属。2026–2027 周期的申请于 2026
年 3 月 5 日开放，4 月 30 日截止，8 月 31 日前公布结果；现任 Google 员工和往届获得者都没有申请资格，
津贴金额因地区而异。
（[PhD Fellowship](https://research.google/programs-and-events/phd-fellowship/)）

GDM 还资助自己的学术项目（[教育页](https://deepmind.google/education/)）：**博士后 fellowship**，
由合作院校接收（包括伦敦玛丽女王大学、剑桥、UCL、伯明翰、帝国理工学院、爱丁堡和牛津——这几所是早年
的合作院校，2025/26 年度的合作方是帝国理工的 Fleming Initiative 和 Wellcome Sanger Institute）；
**研究生 AI 奖学金**，分别由英国的 Martingale Foundation、国际范围内的 Institute of International
Education，以及非洲的 AI for Science Masters 负责管理；以及 **Research Ready**，一个 2023 年启动的
英国本科生暑期实习项目，2026 年暑期在 11 所大学开展。

### 招聘诈骗

没有找到 DeepMind 专门的诈骗预警页面。适用的是 Google 2025 年 11 月发布的招聘诈骗提醒：“A
legitimate company will never require upfront payments or training fees to secure a job.”（正规
公司绝不会要求你预付费用或培训费才能拿到工作。）它还描述了伪造的招聘页面与招聘人员资料、以“注册费”
或“手续费”名义索要费用，以及伪装成面试软件的恶意程序。
（[Google 欺诈与诈骗提醒](https://blog.google/innovation-and-ai/technology/safety-security/fraud-and-scams-advisory-november-2025/)）

## 流程各阶段

### 初步面试

“A 30-minute introductory call with your Recruiter, to cover your background and experiences.”（一通
30 分钟的入门通话，由你的招聘联系人主持，聊你的背景和经历。）
（[招聘页](https://deepmind.google/about/careers/)） GDM 官方表示“许多面试采用基于胜任力
（competency-based）的方式”，并建议用情境、任务、行动、结果（STAR）的结构组织你的答案。
（[面试指南](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)）
面试前你需要电子签署一份标准的保密协议（NDA）。

**候选人反馈**：这通电话有时会和团队负责人一起进行，内容不涉及技术：背景经历、（学生候选人的）毕业
时间和签证状态、候选人对 GDM 工作的了解和看法，此外在很多独立报告和岗位里都会被问到某种形式的“为
什么想加入 DeepMind”——一位研究工程师说他在同一轮流程里被问了整整五次。有些流程的第一场对话换成
了用人经理主持，会详细询问候选人的背景，还有一道简短的案例题。

### 技能面（Skills）

“Over two or three further calls, we'll evaluate you against the competencies and skills required for
success.”（接下来两到三通电话，我们会对照这个岗位所需要的胜任力和技能来评估你。）
（[招聘页](https://deepmind.google/about/careers/)） **候选人反馈**：这个阶段会混合以下几项内容，
具体顺序因团队而异：

- **编程。** 题目难度从 LeetCode 简单到 LeetCode 困难不等，代码写在共享编辑器里运行，有时还要通过
  面试官准备的测试用例（2025 年有一位候选人四个用例里通过了三个，自己仍然判断这一轮因为沟通
  太少而没过）。图的遍历是被提到最多的主题（一道困难的 BFS 变体；一道用 BFS/DFS 解决的博弈策略题；根据
  边构建一张无向图，返回它的深度优先遍历顺序），也有好几份报告描述了一道 LeetCode 上找不到的题
  目——2025 年的一道题是写一个针对列表的 Python 生成器，要写单元测试并处理像 `None` 这样的边界
  情况，然后是一个按给定概率对列表元素做加权采样的生成器。其他报告提到的还有一道带权衡取舍追问的
  哈希表题、“比较冷门的数据结构和树相关问题”、一道贪心的刷栅栏问题、带编码解码的位运算题，以及
  一道标准题加上一道基于候选人自己实现继续出的组合数学追问。更早期的研究工程师筛选轮问过如何在
  允许预处理的前提下，找出一批三维点里离查询点最近的邻居（即 k-d 树）、一个先升后降的数组的最大
  值，以及按权重生成随机数的方法。
- **口头的机器学习知识问答。** 最近的报告里提到了精确率（precision）和召回率（recall）、优化器与
  损失函数；“某个核心大模型概念背后的数学”；PPO 和 GRPO 的区别；Transformer 是怎么训练的（预
  训练、有监督微调、强化学习）；以及 focal loss 和交叉熵的对比。2025 年，一位研究工程师的两轮
  机器学习面试都把候选人自己的研究讨论和更难的机器学习问题、部分系统设计问题结合在一起，还默认
  候选人熟悉面试团队发表过的工作；一位机器学习工程师的“应用机器学习基础”轮则是深挖候选人自己的
  项目。参见[quiz 的演变](#quiz-的演变)。
- **机器学习编程或调试**，部分研究和应用岗位会有：在时间压力下写模型代码，并在训练代码里找 bug。
  一位拿到 offer 的研究科学家是靠从零实现注意力机制和反向传播来准备的；还有一位应用 AI 候选人的
  招聘联系人把调试轮的 bug 形容成“蠢，但不难”（这是对即将到来的一轮面试的预告，而不是已经完成
  一轮的反馈）。
- **因团队而异的机器学习系统设计轮。** 一位 Gemini Robotics 候选人特别强调，这一轮“不是标准的
  推荐系统问题”，标准的机器学习流水线式答案并不适用；一位应用 AI 候选人被告知要准备大规模生成式
  AI 系统（检索增强生成、智能体、效率）；2025 年伦敦的一位候选人被要求对形如（分子、分子、反应
  因子）的记录建模，预测其中的反应因子，内容从探索性分析、特征工程、数据切分到模型选择、泛化
  能力一路展开，面试官还会针对候选人提出的方案追问机器学习问题（这位候选人当时提的是
  Transformer）；还有一次 Gemini app 相关的流程直接让候选人设计 Gemini app 本身，形式是由
  面试官把控节奏的快速问答。2025 年的一次机器学习工程师流程里，仍然出现过推荐系统设计题，更早的
  一次研究工程师流程则要求做一个端到端的设计，预测数据中心里哪些机器需要更换。

### quiz 的演变

2022 年左右之前，第一个技术环节是一种很有特色的 **quiz**：两位面试官、两个小时，从大量题目中
抽出大约五十道简答题，难度大多相当于本科阶段的考试，公式要写在纸上举给摄像头看。在 2022
年的一次研究科学家面试流程里，第一位面试官考的是线性代数、微积分、计算机基础和信息论，第二位
面试官考的是概率、统计（包括统计检验）、优化方法、机器学习和强化学习。研究方向的实习筛选轮在
2020 年被压缩成一个小时，内容抽样自数学、机器学习、强化学习和读代码——比如解释一段给定代码在
算什么，像是一维卷积，然后是它的带 padding 版本。

从 2022 年起，报告里描述 quiz 被拆分成几场各一小时的独立面试：数学、机器学习，以及带编程的计算
机基础；题量变少，追问变多，数学题也会和它们在机器学习里的应用联系起来。2023 年的一次研究科学家
流程里，同样是这三场各一小时的面试。2025 年，一位研究工程师候选人被招聘联系人告知：“there is no
maths / stats quiz anymore. But there is ML / AI quiz”（现在已经没有数学/统计 quiz 了，但还有
机器学习/AI quiz）；另一位候选人反馈说自己经历了两轮机器学习面试，内容混合了自己的研究、更难的
机器学习问题和设计题；2026 年的一次研究工程师流程里，依然有单独的一场数学轮，部分软件工程和实习
流程里也依然会出现计算机和数学题。你应该预期会有机器学习知识问答轮，并直接问你的招聘联系人，数学
是否会被单独考察。

### Take-home 作业

**候选人反馈**：只有少数路线会有 take-home 作业：一个大约十轮的研究科学家流程里包含“a take home
with a timer (3 hours)”（一个带计时器的 take-home，3 小时），还有一位战略与运营
（strategy-and-operations）候选人准备并展示了一次关于 DeepMind 近期成果的报告。这两条都只是各自
单独的一份报告。一个被描述为“Gemini app prototype for DeepMind interview demo”（为 DeepMind
面试演示准备的 Gemini app 原型）的公开代码仓库说明，至少有一个流程要求候选人做一个能跑起来的小
应用。

### 终面

“During the final round, you'll meet Team Leads and leadership – including your potential manager.”
（在终面阶段，你会见到团队负责人和领导层——包括你未来可能的经理。）
（[招聘页](https://deepmind.google/about/careers/)）

**候选人反馈**：终面阶段有好几场各自独立的对话：

- **研究报告或论文讨论**，研究岗位会有：展示或深入讲解候选人自己的研究工作，有时之后还会和团队里
  的每个成员分别聊 30 分钟。一位研究科学家候选人形容两场研究面试里的其中一场“相当有对抗性，但很
  尊重人”；面试官们问了候选人想在 GDM 做什么方向的研究。2026 年的一次论文答辩轮没有编程环节，节奏
  很快，依次问到了这个问题为什么值得做、有哪些备选方案、如果某个核心假设不成立会怎样，以及下一步的
  研究方向；对于偏重评测或基准测试的多模态工作，这一轮还深入考察了候选人在训练和后训练上的经验有多深，
  以及这项贡献是否只是搭建了一个评测。
- **团队负责人与用人经理对话**，聊候选人的经历和匹配度。2026 年的一条讨论帖把团队负责人这一轮
  描述成一次纯聊匹配度、没有正式技术问题的对话；2025 年一次 Gemini 研究科学家的用人经理轮是开放
  式的，但紧扣团队所在的领域——当前多语言模型的局限性、候选人最引以为豪的项目、候选人想在团队里
  做什么方向、怎么做，以及现有理论能不能解释这个方向；2025 年一位机器学习工程师的用人经理面谈则把
  行为面问题、候选人自己提出的问题，以及一道商业案例题结合在了一起。
- **行为面**：再问一遍“为什么是 DeepMind”，还有冲突、失败、协作与优先级排序，形式是一场
  people-and-culture 面试，或者候选人称之为“Googleyness”的一轮。2020 年的一次实习流程里，行为
  面问到了 DeepMind 的使命是什么、AGI 是否可能实现，以及候选人自己的长期规划，候选人后来还听说
  有一位申请人因为答不出使命宣言而被拒；GDM 官方给出的表述是“to build AI responsibly to
  benefit humanity”（负责任地构建 AI，以造福人类）（[关于页](https://deepmind.google/about/)）。
- **代码评审轮**，一位研究工程师候选人在 2026 年反馈说，这是编程、数学和机器学习几轮之后一个
  没有预料到的终面轮次。

### 决定与 offer

“The hiring team will review your application against our criteria. If you're the best candidate for
the role, your Recruiter will share the exciting news.”（招聘团队会对照我们的标准评审你的申请。
如果你是这个岗位最合适的人选，你的招聘联系人会把这个好消息告诉你。）
（[招聘页](https://deepmind.google/about/careers/)）

**候选人反馈**：就算通过了所有技术轮，也不保证一定能拿到 offer——终面之后团队内部的讨论仍然可能
做出不利于候选人的决定；走 Google 那扇门的话，offer 还要看 team match 能不能成功。

## 时间线

官方说法：“the interview process takes between 4-10 weeks.”（整个面试流程需要 4 到 10 周。）
（[面试指南](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)）

**候选人反馈**的范围要宽得多：

| 阶段 | 报告中的时间范围 |
| --- | --- |
| 两轮之间的间隔 | 几天到大约四周 |
| 全流程走完 | 最快的报告里大约一个月；其他报告里是三个月、五个月和六个月 |

- 一位研究工程师反馈说，轮次之间出现过“4 周的沉默”，全程超过三个月。
- 一位机器学习候选人在大约三周里，三次收到不到半天前发出的面试取消通知。
- 也会有不声不响的拒绝：一份报告里，招聘联系人提议安排的一通电话始终没有被排上，两周后收到了一封
  自动生成的拒信，里面提到了重新申请前有一个“冷静期（cool down period）”。
- 一位 2026 年的 AI for Science 候选人 4 月投递，6 月到 8 月之间经历了五轮面试，9 月初被告知
  决定正在最终确定中，9 月底被拒，对方表示等岗位重新开放时会再联系——前后大约六个月。
- 进展快的一例：一位 2025 年的机器学习工程师，从 recruiter 电话到拿到 offer 只用了不到四周。
- 更早的报告里还出现过候选人已经通过好几轮面试，headcount（招聘名额）却被收回的情况。

## 路线

### 研究工程师与研究科学家

研究工程师和研究科学家是两条不同的序列（ladder），而不是初级/高级的区分：研究科学家主要做研究和
发论文，研究工程师则把研究、实现和产品化整合在一起（候选人反馈）。[流程各阶段](#流程各阶段)里
描述的流程在这里最为适用：一到两轮编程，一轮口头的机器学习知识问答，机器学习编程或设计，然后是
终面阶段的研究报告或论文讨论、团队负责人对话和行为面。2025 年的一条讨论 offer 的帖子把研究工程师
的流程描述成两轮编程、两轮机器学习和两轮团队负责人面试，研究科学家的流程则是先做一次特邀报告
（invited talk），再由团队挑选三到五场编程或机器学习面试；一位 2022 年的研究科学家反馈说自己的
流程里没有系统设计轮。好几条独立的讨论都认为，研究工程师这条路线的流程是比较难的之一，既需要吃透
当前的模型架构，也需要扎实的数学功底。

### 软件工程与 Gemini app 相关岗位

**候选人反馈**：这条路线的流程更偏重编程和系统设计：LeetCode 中等难度的编程轮（一份回答里说是
“动态规划和回溯”，而不是冷门数据结构），一轮系统设计有时会从候选人自己过去的某个项目讲起，还有
一场团队负责人对话和好几轮非技术面试。2025 年一次 Gemini app 移动端的流程里，用人经理轮一开始是
两道热身题：数出数组中超过某个阈值的元素并标出它们的下标，以及根据边构建一张无向图、返回它的
深度优先遍历顺序。走 Google 那扇门的话，有一个 Gemini 应用团队的机器学习
软件工程师流程考了在一批带时间戳的事件里找出现最频繁的事件，然后要求在最近 N 个事件的滑动窗口上
维护这个答案；它的用人经理轮问的是如何跨领域评估一个多任务系统，以及在没有额外数据的情况下怎么
改进模型。

### 应用 AI 与机器学习工程

**候选人反馈**：编程、机器学习基础、一轮机器学习调试，以及一轮围绕生成式 AI 产品（检索增强生成、
智能体框架、效率、评测）的系统设计。一位应用 AI 候选人被告知这个岗位不要求博士学位。2025 年的
一次机器学习工程师流程是：先是一通 recruiter 电话，然后是一轮关于推荐系统的系统设计、一轮编程
（一道 LeetCode 上找不到的博弈策略题）、一轮“应用机器学习基础”深挖候选人自己的项目，以及一次
用人经理面谈，最后给出的 offer 比申请的职级低一级，也没有给出理由。

### Student Researcher

参见[实习、Student Researcher 与 fellowship 项目](#实习student-researcher-与-fellowship-项目)。

### 产品与项目管理

**候选人反馈**：产品经理的流程包括一次 HR 初筛、一到两次 30 分钟的用人经理对话，以及终面阶段四到
五场面试——产品洞察或策略、用户体验、执行力、和工程师一起做的 AI 技术深挖，有时还会有一场和总监的
对话——再加一场 people-and-culture 对话。大多数轮次都是 AI 产品案例：一个会替用户执行操作的助手该
用什么指标、在模型仍然会犯错的情况下要不要上线有风险的操作、以及怎么处理一个反馈两极分化的功能。
技术项目经理（Technical Program Manager）的流程据反馈是一次简短的 HR 初筛，接着是一轮考察技术
深度和干系人（stakeholder）管理能力的用人经理面；关于项目管理类岗位的证据还比较少。

## 职级与薪酬

### 职位名称与职级

职位发布用的是职能类头衔——Research Scientist、Research Engineer、Software Engineer、Technical
Program Manager——再加上 Google 的资深程度修饰词，比如 Staff 和 Senior Staff。GDM 的页面不公开
职级对照表。

**候选人反馈**：GDM 定出来的职级，可能比预期更低：一次 team match 的讨论里，同一位候选人拿到的
GDM offer 是 L3，而 Google Search 给出的 offer 是 L4；还有一位 2025 年的机器学习工程师，面试的
是 L5，却在没有任何解释的情况下被给了 L4 的 offer。有一份 2024 年 L5 级别的研究工程师 offer，
据反馈年度总薪酬（total compensation）大约是 $550k（仅此一个数据点）。这类级差的具体大小，
只能当作传闻来看。

### 公开薪资区间

美国的职位发布会按薪资透明法规写出年度基本工资区间；核实过的非美国职位发布均未公开这一数字。以下
数据来自 GDM 的 Greenhouse 职位板，核实于 2026-09-23：

| 职位 | 地点 | 年薪（USD） |
| --- | --- | --- |
| [Research Scientist, Multimodal Alignment, Safety, and Fairness](https://job-boards.greenhouse.io/deepmind/jobs/7680885) | 柯克兰、山景城、纽约 | 147,000 – 211,000 |
| [Technical Program Manager, AI Safety](https://job-boards.greenhouse.io/deepmind/jobs/7409418) | 山景城 | 156,000 – 229,000 |
| [Staff Applied AI Engineer](https://job-boards.greenhouse.io/deepmind/jobs/7561938) | 山景城 | 197,000 – 291,000 |
| [Software Engineer, Gemini Personal Intelligence](https://job-boards.greenhouse.io/deepmind/jobs/7397693) | 山景城 | 248,000 – 349,000 |
| [Senior Staff Software Engineer, Gemini App](https://job-boards.greenhouse.io/deepmind/jobs/7053012) | 山景城 | 248,000 – 349,000 |

每条职位发布都在基本工资区间之外加了一句“+ bonus + equity + benefits”（另加奖金、股权和福利）。
职位发布会不断上下架；想看最新数字，还是要直接打开当前在招的那条。

### 股权

**候选人反馈**：好几条独立的薪酬讨论帖都提到，股权是按**33% / 33% / 22% / 12%** 的前重后轻
（front-loaded）方式，分四年归属（vesting）；一份研究工程师的 offer 里给出的比例是 38% / 32% /
20% / 10%。

## 面试规则

| 主题 | 具体规则 |
| --- | --- |
| AI 工具 | “You're free to use AI to help you prepare for interviews. But – unless told otherwise – please don't use any AI tools during live interviews, or during interview tasks.”（你可以自由使用 AI 帮你准备面试。但是——除非另有说明——请不要在实时面试或面试任务中使用任何 AI 工具。）（[面试指南](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)） |
| AI 工具试点 | 2025 年 12 月，GDM 的人才招聘负责人（head of talent acquisition）谈到一个编程测评方面的试点项目：候选人被要求使用一个自己选择的 AI 系统，考察的是他们怎么使用它。为写这份指南核实过的候选人文章里，目前还没人提到过这个试点。（[Arctic Shores 采访](https://www.arcticshores.com/insights/how-google-deepmind-uses-ai-in-recruitment-with-becky-pradal-rogers)） |
| Google 更大范围的变化 | 2025 年 8 月，媒体报道 Google 考虑增加线下面试，但没有公布正式政策（[Business Standard](https://www.business-standard.com/companies/news/google-ai-cheating-job-interviews-in-person-hiring-shift-sundar-pichai-125082600492_1.html)）。2026 年 5 月，媒体报道了一个新的代码理解面试环节：候选人使用一个 Google 提供、基于 Gemini 的助手，目前在 Google 内部部分团队试点，计划在 2026 年下半年推广（[新浪财经](https://finance.sina.com.cn/stock/t/2026-05-08/doc-inhxefhi2265587.shtml)）。这两篇报道都没有点名 Google DeepMind。 |
| 视频 | 大多是线上进行，通过 Google 的视频通话链接（[面试指南](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)） |
| 编程环境 | 候选人反馈是一个能运行代码的共享编辑器，比如 CoderPad |
| 保密 | “We'll also send you a standard non-disclosure agreement. You'll need to electronically sign this through a third-party tool, Ironclad.”（我们还会给你发一份标准的保密协议，你需要通过第三方工具 Ironclad 电子签署。）（[面试指南](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)） |

## 决定之后

没有找到 GDM 关于反馈或重新申请冷静期的页面。**候选人反馈：**

- 通过了所有技术轮之后，仍然会在终面团队内部讨论阶段被拒——2026 年的一份报告里，原因是团队
  “needed someone who could ramp up faster for that specific role”（需要一个能在这个具体岗位上
  更快上手的人）。
- 也有提到重新申请前有冷静期的自动拒信。
- 书面反馈不是必然的，但确实会发生：一位被拒的候选人收到的反馈提到了“strong command of
  Python”（扎实的 Python 功底）和“clean and readable coding”（干净易读的代码）。

## 如何用这个仓库准备面试

| 阶段 | 页面 |
| --- | --- |
| 初步面试 | [HR 初筛](behavioral/recruiter-screen/README.zh.md) |
| 编程 | [数组扫描与图的深度优先遍历](coding/graph-dfs-warmup/README.zh.md) · [穿过上锁的门的最短路径](coding/state-space-bfs/README.zh.md) · [事件流中出现最频繁的事件](coding/event-frequency-stream/README.zh.md) · [移动棋子游戏](coding/token-game-outcomes/README.zh.md) · [生成器、单元测试与加权采样](coding/weighted-sampling-generator/README.zh.md) · [把单位换算建成带权图](coding/unit-conversion-graph/README.zh.md) · [刷栅栏](coding/fence-painting-strokes/README.zh.md) · [位打包](coding/bit-packing-codec/README.zh.md) · [快照数组](coding/snapshot-array/README.zh.md) · [k-d 树](coding/kd-tree-nearest-neighbour/README.zh.md) · [双调数组与单调函数上的二分查找](coding/binary-search-monotone/README.zh.md) |
| 机器学习知识与数学问答轮 | [机器学习基础](quiz/ml-fundamentals/README.zh.md) · [大模型基础](quiz/llm-fundamentals/README.zh.md) · [大模型强化学习](quiz/rl-for-llms/README.zh.md) · [数学：概率、统计与线性代数](quiz/maths-probability-linear-algebra/README.zh.md) · [数学：矩阵计算、统计检验与信息论](quiz/maths-stats-information-theory/README.zh.md) · [计算机基础](quiz/cs-fundamentals/README.zh.md) · [代码理解](quiz/code-comprehension/README.zh.md) |
| 机器学习编程与调试 | [带 KV cache 的注意力](coding/attention-kv-cache/README.zh.md) · [调试分类器](coding/training-loop-debug/README.zh.md) · [反向传播](coding/mlp-backprop-from-scratch/README.zh.md) · [高斯混合模型的 EM](coding/em-gaussian-mixture/README.zh.md) · [Focal loss](coding/focal-loss/README.zh.md) |
| 机器学习与系统设计 | [Gemini app](system-design/gemini-app/README.zh.md) · [大模型助手的评测系统](system-design/llm-evaluation-system/README.zh.md) · [分子对的反应因子](system-design/molecule-reaction-ml-design/README.zh.md) · [检索增强智能体](system-design/rag-agent-platform/README.zh.md) · [推荐信息流](system-design/recommendation-system/README.zh.md) · [单卡放不下的模型的训练方案](system-design/distributed-training/README.zh.md) · [机器人学习的数据飞轮](system-design/robot-learning-data-flywheel/README.zh.md) · [数据中心机器更换](system-design/datacentre-failure-ml-design/README.zh.md) |
| 终面 | [研究深挖与论文答辩](behavioral/research-deep-dive/README.zh.md) · [团队负责人与用人经理面](behavioral/team-lead-rounds/README.zh.md) · [Googleyness 与文化](behavioral/googleyness-and-culture/README.zh.md) · [代码评审](coding/code-review-round/README.zh.md) |
| 产品管理 | [AI 产品感与 AI 技术深挖](behavioral/pm-ai-product-sense/README.zh.md) |

[学习路线](ROADMAP.zh.md)按路线和优先级给每个页面排了序。

## 官方信息来源

- 招聘信息、岗位序列、地点与面试阶段：<https://deepmind.google/about/careers/>
- 官方面试指南（阶段、时间线、STAR、AI 工具规则、NDA、混合办公）：
  <https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf>
- Student Researcher Program：<https://deepmind.google/student-researcher-program/>
- GDM 的 fellowship 与奖学金项目：<https://deepmind.google/education/>
- Google PhD Fellowship：<https://research.google/programs-and-events/phd-fellowship/>
- GDM 职位发布与美国薪资区间：<https://job-boards.greenhouse.io/deepmind>
- Google 2025 年 11 月发布的欺诈与诈骗提醒：
  <https://blog.google/innovation-and-ai/technology/safety-security/fraud-and-scams-advisory-november-2025/>
- GDM 编程测评中的 AI 工具（对 GDM 人才招聘负责人的采访，2025 年 12 月）：
  <https://www.arcticshores.com/insights/how-google-deepmind-uses-ai-in-recruitment-with-becky-pradal-rogers>
- Google 考虑增加线下面试（Business Standard，2025 年 8 月 26 日）：
  <https://www.business-standard.com/companies/news/google-ai-cheating-job-interviews-in-person-hiring-shift-sundar-pichai-125082600492_1.html>
- Google 内部一个由 Gemini 辅助的代码理解面试环节（新浪财经，2026 年 5 月 8 日）：
  <https://finance.sina.com.cn/stock/t/2026-05-08/doc-inhxefhi2265587.shtml>
