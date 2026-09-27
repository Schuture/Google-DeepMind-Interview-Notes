# The Google DeepMind Hiring Process

English · [中文](PROCESS.zh.md)

> [!NOTE]
> **Unofficial, last checked 2026-09-23.** This guide combines two kinds of information. Statements with a
> source link come from Google DeepMind's (GDM's) or Google's own pages — the careers site, the official
> interview guide, job postings and policies — checked on the date above. Paragraphs headed **Candidates
> report** summarise public write-ups by people who interviewed at GDM, mostly between late 2024 and September
> 2026, with older accounts used where they show how a round has changed; they are not official, they are not
> linked, and a pattern seen only once is marked as such. GDM fills roles through two different pipelines (see
> [Where and how to apply](#where-and-how-to-apply)) and teams run their loops differently: your recruiter's
> instructions and the job posting always take precedence over this page.

## At a glance

| Stage | What happens | Length | Who gets it |
| --- | --- | --- | --- |
| [Application](#getting-an-interview) | GDM's own posting, or Google's general pipeline followed by team match | — | everyone |
| [Initial interviews](#initial-interviews) | A recruiter call about your background and experience, sometimes with a team lead | 30 min (official) | everyone |
| [Skills interviews](#skills-interviews) | Coding, an oral ML-knowledge round, ML coding or debugging for some roles, a team-specific ML design round | two or three calls (official), about an hour each | most loops |
| [Take-home](#take-home-assignments) | A timed exercise or a prepared presentation | a few hours | a few tracks |
| [Final interviews](#final-interviews) | Team leads and leadership; a research talk or paper discussion for research roles; behavioural rounds | several calls | everyone who gets this far |
| [Decision and offer](#decision-and-offer) | Hiring-team review, sometimes a team debrief or team match, then the offer | days to weeks | after the final interviews |

Official end-to-end timeline: **4–10 weeks**
([interview guide](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)).
Candidates report it often running longer, with weeks of silence between rounds and, in several accounts,
three to six months end to end (see [Timeline](#timeline)).

## Getting an interview

### What Google DeepMind says it looks for

- "We're searching for people who share our drive to build the next generation of breakthrough AI systems,
  safely and responsibly." ([careers](https://deepmind.google/about/careers/))
- Role families named on the careers page: Research Engineer, Software Engineer, Research Scientist, Product
  Manager, Program Manager, Technical Program Manager, Operations and Responsibility.
  ([careers](https://deepmind.google/about/careers/))
- The interview guide asks for concrete, concise examples, reasoning shown step by step, and honesty when you do
  not know an answer, and suggests reading GDM's blog and using its products before interviewing.
  ([interview guide](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf))
- Accessibility: "If you have a disability, need assistive technology, or other extra support during the
  interview process, please let us know." ([careers](https://deepmind.google/about/careers/))

### Where and how to apply

The careers page's list of open roles leads to Google's careers portal filtered to DeepMind; many GDM roles are
also posted on GDM's own [Greenhouse board](https://job-boards.greenhouse.io/deepmind), which is where the
pay-transparency ranges in [Levels and compensation](#levels-and-compensation) come from.
([careers](https://deepmind.google/about/careers/))

There are two doors into GDM, and they lead to different interviews:

1. **GDM's own posting and process.** A role advertised for a GDM team runs GDM's four official stages —
   initial interviews, skills interviews, final interviews, decision and offer — with content chosen by the
   team. Research scientist and research engineer roles, and many software roles, come through this door.
2. **Google's general pipeline, then team match.** Some product and infrastructure work on GDM teams (the
   Gemini app is the example that recurs), and some internships, are filled through Google's general hiring
   loop, after which the candidate is matched to a GDM team. The interviews on this door are Google's general
   ones.

**Candidates report:**

- A software-engineering intern who went through Google's ordinary team-match pool was offered a match with a
  GDM team building the Gemini app's iOS front end, alongside an unrelated Google team.
- A candidate who interviewed both for a Google team in Mountain View and for a DeepMind team in London
  described the Google loop as two coding rounds, two system-design rounds and one behavioural round, and the
  DeepMind loop as markedly harder and differently shaped.
- **Moving from a Google role into GDM means applying to the GDM role and interviewing again**, according to
  several independent threads; employees discussing the move could not say whether it restarts a green-card
  labour certification. Whether the bar is "people GDM already knows, or people from other frontier labs" is
  disputed in the same threads: others describe new graduates without a PhD and researchers from varied
  backgrounds getting in.
- GDM's hiring is described as run separately from the rest of Google, with team match and compensation that do
  not carry over between the two (second-hand, in one offer thread from January 2025).
- Cold online applications, recruiter outreach and referrals all lead to interviews. For research roles, the same
  offer thread held that knowing the hiring manager helps a great deal and that other referrals add little —
  closer to the academic job market than to a referral programme (a single thread).
- A research-engineer candidate with a PhD, rejected in 2025, summed up the bar as a close match between the
  candidate's research direction and the team's, plus seven or eight papers at top venues in the area — one
  person's view.

### Locations and hybrid work

- Ten hubs are named on the careers page: London, the Bay Area, Bangalore, Cambridge (US), Montreal, New York
  City, Paris, Tokyo, Toronto and Zurich. ([careers](https://deepmind.google/about/careers/))
- "Most colleagues follow our hybrid model – working from the office Tuesday-Thursday and from home, or
  somewhere nearby, on Monday and Friday."
  ([interview guide](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf))
- Interviews are mostly remote: "the majority of our interviews are virtual," with "calendar invites with Google
  Hangout links for video call interviews."
  ([interview guide](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf))
- None of the pages checked for this guide state a visa-sponsorship policy; ask your recruiter if it matters for
  you.

### Internships, Student Researchers and fellowships

The **Student Researcher Program**
([student researcher program](https://deepmind.google/student-researcher-program/)):

- "You must be enrolled in a Bachelor's, Master's, or PhD program."
- "Between 12 and 24 weeks, with a minimum time commitment of four days a week."
- "In-person, at a Google office so you can work directly with your host team."
- The positions are paid; no stipend is published on the page.
- Applicants are considered for relevant positions across Google's AI teams — Google DeepMind, Google Research or
  other Google teams — so applying does not guarantee a GDM placement.

**Candidates report** two shapes for this loop. Most describe two research-only conversations with no coding or
system design: about 45 minutes on the candidate's own research with a researcher from the team, then about
30 minutes with the prospective host on a possible project. A minority describe a fundamentals round instead —
linear regression, regularisation, optimisation and basic deep learning in one report, the difference between
PPO and GRPO in another — followed by a team-matching conversation. Older research-internship loops (2020–2021)
ran a recruiter screen, a one-hour technical interview sampling maths, ML, reinforcement learning and code
reading, several research interviews that decided whether a team wanted the candidate, and a culture round; in
some years the internship headcount was gone within weeks of applications opening, so applying early matters.

The **Google PhD Fellowship** is Google-wide rather than GDM-specific. For the 2026–2027 cycle, applications
opened on 5 March 2026 and closed on 30 April 2026, with decisions by 31 August 2026; current Google employees and
past recipients are not eligible, and the stipend depends on the region.
([PhD Fellowship](https://research.google/programs-and-events/phd-fellowship/))

GDM funds its own academic programmes ([education](https://deepmind.google/education/)): **postdoctoral
fellowships** hosted at partner institutions (among them Queen Mary University of London, Cambridge, UCL,
Birmingham, Imperial College London, Edinburgh and Oxford in earlier years, and the Fleming Initiative at
Imperial and the Wellcome Sanger Institute for 2025/26); **postgraduate AI scholarships**, administered through
the Martingale Foundation in the UK, the Institute of International Education internationally and AI for Science
Masters in Africa; and **Research Ready**, a UK undergraduate summer placement scheme that started in 2023 and
runs at 11 universities in summer 2026.

### Recruiting scams

No DeepMind-specific warning page was found. Google's November 2025 advisory on job scams applies: "A legitimate
company will never require upfront payments or training fees to secure a job." It describes fake career pages
and recruiter profiles, requests for "registration" or "processing" fees, and malware disguised as interview
software.
([Google fraud and scams advisory](https://blog.google/innovation-and-ai/technology/safety-security/fraud-and-scams-advisory-november-2025/))

## The stages

### Initial interviews

"A 30-minute introductory call with your Recruiter, to cover your background and experiences."
([careers](https://deepmind.google/about/careers/)) GDM describes "a competency-based approach in many of our
interviews" and recommends structuring answers with the STAR method (Situation, Task, Action, Result).
([interview guide](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf))
A standard non-disclosure agreement is signed electronically before the interviews.

**Candidates report** this call, sometimes run jointly with a team lead, as non-technical: background,
graduation date and visa status for students, what the candidate knows and thinks about GDM's work, and, across
many independent reports and roles, some form of "why do you want to work at DeepMind" — one research engineer
counted it five times across a single loop. In some loops the first conversation is with the hiring manager
instead, with detailed questions about the candidate's background and a short case question.

### Skills interviews

"Over two or three further calls, we'll evaluate you against the competencies and skills required for success."
([careers](https://deepmind.google/about/careers/)) **Candidates report** this stage mixing some of the following,
in an order that varies by team:

- **Coding.** Problems from LeetCode-easy to LeetCode-hard difficulty, written so that the code runs in a shared
  editor, sometimes against the interviewer's test cases (one 2025 candidate passed three of four and still
  judged the round lost for too little communication). Graph traversal is the topic mentioned most (a hard BFS
  variant; a game-strategy problem solved with BFS/DFS; build an undirected graph from edges and return its DFS
  order), and several reports describe a problem that does not exist on LeetCode — in 2025, a Python generator
  over a list with unit tests and corner cases such as `None` entries, followed by a generator that samples the
  list's values with given probabilities. Other reports name a hashmap question with trade-off follow-ups,
  "obscure data structure and trees questions", a greedy fence-painting problem, bit manipulation with encoding
  and decoding, and a standard problem followed by a combinatorics question built on the candidate's own
  implementation. Older research-engineer screens asked for the nearest neighbour of a query among 3-D points with
  preprocessing allowed (a k-d tree), the maximum of an array that rises then falls, and weighted random number
  generation.
- **An oral ML-knowledge round.** Recent reports mention precision and recall, optimisers and loss functions; "the
  maths of some core LLM concept"; the difference between PPO and GRPO; how transformers are trained (pretraining,
  supervised fine-tuning, reinforcement learning); and focal loss compared with cross-entropy. In 2025 one
  research engineer's two ML rounds each combined a discussion of the candidate's own research with harder ML
  questions and some system-design questions, and assumed familiarity with the interviewing team's published
  work; a machine-learning engineer's "applied ML fundamentals" round was a deep dive into the candidate's own
  projects. See [How the quiz changed](#how-the-quiz-changed).
- **ML coding or debugging**, for some research and applied roles: writing model code under time pressure and
  finding bugs in training code. A research scientist who received an offer prepared by implementing attention and
  backpropagation from scratch, and one applied-AI candidate's recruiter described the debugging round's bugs as
  "stupid, not hard" (a preview of an upcoming loop, not a completed round).
- **A team-specific ML system-design round.** A Gemini Robotics candidate stressed that it was "not the standard
  recommendation system questions" and that standard ML-pipeline answers did not fit; an applied-AI candidate was
  told to prepare large-scale generative-AI systems (retrieval-augmented generation, agents, efficiency); a 2025
  London candidate was asked to model a reaction factor from records of the form (molecule, molecule, factor),
  from exploratory analysis through features, data splits and model choice to generalisation, with ML quiz
  questions on whatever the candidate proposed (a transformer, in that case); one Gemini-app loop asked the
  candidate to design the Gemini app itself, as a fast question-and-answer exchange paced by the interviewer. A
  recommendation-system design was still reported in a 2025 machine-learning-engineer loop, and an older
  research-engineer loop asked for an end-to-end design predicting which data-centre machines need replacing.

### How the quiz changed

Until about 2022 the first technical stage was a distinctive **quiz**: two hours with two interviewers and some
fifty short questions drawn from a large bank, most at the level of an undergraduate exam, with formulas written on
paper and held up to the camera. In one 2022 research-scientist loop the first interviewer covered linear algebra,
calculus, computer science and information theory, the second probability, statistics (including statistical
tests), optimisation, ML and reinforcement learning. Research-internship screens were cut in 2020 to a single hour
sampling maths, ML, reinforcement learning and code reading — explaining what a given snippet, such as a 1-D
convolution and then its padded version, computes.

From 2022 reports describe the quiz split into separate one-hour interviews for maths, ML, and computer science
with coding: fewer questions, many follow-ups, and maths questions tied to their use in ML. A 2023
research-scientist loop had the same three one-hour interviews. In 2025 one research-engineer candidate was told
by the recruiter that "there is no maths / stats quiz anymore. But there is ML / AI quiz", and another described
two ML rounds mixing their own research, harder ML questions and design; a 2026 research-engineer loop still had
a separate maths round, and computer-science and maths questions still appear in some software-engineering and
intern loops. Expect the ML-knowledge round, and ask your recruiter whether maths is examined separately.

### Take-home assignments

**Candidates report** take-homes in a few tracks: one research-scientist loop of about ten rounds included "a
take home with a timer (3 hours)", and a strategy-and-operations candidate prepared and presented a talk on a
recent DeepMind achievement. Each is a single report. A public code repository described as a "Gemini app
prototype for DeepMind interview demo" suggests that at least one loop asked for a small working application.

### Final interviews

"During the final round, you'll meet Team Leads and leadership – including your potential manager."
([careers](https://deepmind.google/about/careers/))

**Candidates report** several separate conversations:

- **A research talk or paper discussion**, for research roles: a presentation or a deep dive on the candidate's
  own work, sometimes followed by 30-minute conversations with each member of the team. One research-scientist
  candidate described one of two research interviews as "quite confrontational, but respectful"; the
  interviewers asked what the candidate would like to work on at GDM. A 2026 paper-defence round had no coding
  and moved quickly through why the problem was worth pursuing, the alternative designs, what happens if a
  central assumption fails, and the next research step; for evaluation- or benchmark-heavy multimodal work it
  probed how deep the candidate's training and post-training experience went and whether the contribution was
  more than assembling an evaluation.
- **Team-lead and hiring-manager conversations** about the candidate's experience and fit. One 2026 thread
  describes the team-lead round as a fit conversation without formal technical questions; a 2025 Gemini
  research-scientist hiring-manager round was open-ended but specific to the team's domain — the limitations of
  current multilingual models, the candidate's proudest project, what they would work on in the team and how,
  and whether current theory explains it; a 2025 machine-learning engineer's hiring-manager chat combined
  behavioural questions, the candidate's own questions and a business case study.
- **Behavioural rounds**: "why DeepMind" again, conflict, failure, collaboration and prioritisation, in a
  people-and-culture interview or a round candidates call "Googleyness". A 2020 internship loop's behavioural
  round asked what DeepMind's mission is, whether AGI is possible and the candidate's long-term plan, and the
  candidate later heard of an applicant rejected for not knowing the mission; GDM states it as "to build AI
  responsibly to benefit humanity" ([about](https://deepmind.google/about/)).
- **A code-review round**, reported in 2026 by a research-engineer candidate as an unexpected final round after
  the coding, maths and ML rounds.

### Decision and offer

"The hiring team will review your application against our criteria. If you're the best candidate for the role,
your Recruiter will share the exciting news." ([careers](https://deepmind.google/about/careers/))

**Candidates report** that passing every technical round does not guarantee an offer: a final team debrief can
still decide against a candidate, and on Google's door the offer depends on a successful team match.

## Timeline

Official: "the interview process takes between 4-10 weeks."
([interview guide](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf))

**Candidates report** a wider range:

| Step | Reported range |
| --- | --- |
| Gap between rounds | a few days to about four weeks |
| End to end | about one month in the fastest reports; three, five and six months in others |

- One research engineer reported "4 weeks of radio silence" between rounds and more than three months in total.
- One machine-learning candidate had rounds cancelled at less than half a day's notice three times over about
  three weeks.
- Silent rejections occur: in one report a call that the recruiter had proposed was never scheduled, and two weeks
  later an automated rejection mentioned a "cool down period" before reapplying.
- A 2026 AI-for-science candidate applied in April, had five rounds between June and August, was told in early
  September that the decision was being finalised, and was rejected at the end of September with an offer to be
  contacted when a role reopens — about six months in all.
- At the fast end, a 2025 machine-learning engineer went from the recruiter call to an offer in under four weeks.
- Older reports describe headcount being withdrawn from candidates who had already passed several rounds.

## Tracks

### Research Engineer and Research Scientist

Research engineer and research scientist are different ladders rather than a junior/senior split: research
scientists mainly do research and publish, research engineers mix research, implementation and integration into
products (candidates report). The loop in [The stages](#the-stages) applies most directly here: one or two coding
rounds, an oral ML-knowledge round, ML coding or design, then final interviews with a research talk or paper
discussion, team leads and a behavioural round. One 2025 offer thread describes the research-engineer loop as two
coding, two ML and two team-lead interviews, and the research-scientist loop as an invited talk followed by three
to five coding or ML interviews chosen by the team; a 2022 research scientist reported no system-design round.
Several independent threads describe the research-engineer loop as one of the harder ones, needing depth on
current model architectures as well as mathematics.

### Software engineering and Gemini-app roles

**Candidates report** a loop weighted toward coding and system design: coding rounds at LeetCode-medium level
("dynamic programming and backtracking" rather than exotic data structures, in one answer), a system-design round
that sometimes starts from one of the candidate's own past projects, a team-lead conversation and several
non-technical rounds. A 2025 Gemini-app mobile loop opened its hiring-manager round with two warm-up problems:
count and index the elements of an array above a threshold, and build an undirected graph from edges and return
its depth-first order. On Google's door, one machine-learning software-engineer loop for a Gemini applications team
asked for the most frequent event in a list of timestamped events and then for maintaining it over a sliding
window of the last N events; its hiring-manager round asked how to evaluate a multi-task system across domains and
how to improve a model with no additional data.

### Applied AI and ML engineering

**Candidates report** coding, ML fundamentals, an ML-debugging round and a system design centred on generative-AI
products (retrieval-augmented generation, agent frameworks, efficiency, evaluation). One applied-AI candidate was
told that no PhD was required. A 2025 machine-learning-engineer loop ran a recruiter call, a system design on
recommendation, a coding round with a game-strategy problem not found on LeetCode, an "applied ML fundamentals"
deep dive into the candidate's projects and a hiring-manager chat, then an offer one level below the level
applied for, with no reason given.

### Student Researcher

See [Internships, Student Researchers and fellowships](#internships-student-researchers-and-fellowships).

### Product and program management

**Candidates report** product-manager loops of a recruiter screen, one or two 30-minute hiring-manager
conversations, and a final loop of four or five interviews — product insight or strategy, user experience,
execution, an AI technical deep dive with an engineer, sometimes a director — plus a people-and-culture
conversation. Most rounds are AI-product cases: metrics for an assistant that takes actions for users, launching
risky actions while the model still makes mistakes, and fixing a feature with polarised feedback. Technical
program manager loops are reported as a short recruiter screen followed by a hiring-manager round on technical
depth and stakeholder management; the evidence for program roles is thin.

## Levels and compensation

### Titles and levels

Postings use functional titles — Research Scientist, Research Engineer, Software Engineer, Technical Program
Manager — with Google's seniority qualifiers such as Staff and Senior Staff. GDM's pages do not publish a level
chart.

**Candidates report** that GDM's levelling can come out lower than expected: one team-match discussion compared a
GDM offer at L3 with a Google Search offer at L4 for the same candidate, and a 2025 machine-learning engineer who
interviewed for L5 was offered L4 without explanation. One 2024 research-engineer offer at L5 was reported at
about $550k in total annual compensation (a single data point). Treat the size of any gap as hearsay.

### Posted salary ranges

US postings state an annual base-salary range under pay-transparency law; the non-US postings checked disclosed
none. From GDM's Greenhouse board, checked 2026-09-23:

| Role | Location | Annual base salary (USD) |
| --- | --- | --- |
| [Research Scientist, Multimodal Alignment, Safety, and Fairness](https://job-boards.greenhouse.io/deepmind/jobs/7680885) | Kirkland, Mountain View, New York | 147,000 – 211,000 |
| [Technical Program Manager, AI Safety](https://job-boards.greenhouse.io/deepmind/jobs/7409418) | Mountain View | 156,000 – 229,000 |
| [Staff Applied AI Engineer](https://job-boards.greenhouse.io/deepmind/jobs/7561938) | Mountain View | 197,000 – 291,000 |
| [Software Engineer, Gemini Personal Intelligence](https://job-boards.greenhouse.io/deepmind/jobs/7397693) | Mountain View | 248,000 – 349,000 |
| [Senior Staff Software Engineer, Gemini App](https://job-boards.greenhouse.io/deepmind/jobs/7053012) | Mountain View | 248,000 – 349,000 |

Each posting adds "+ bonus + equity + benefits" to the base range. Postings come and go; open the current ones for
up-to-date numbers.

### Equity

**Candidates report** front-loaded vesting of **33% / 33% / 22% / 12%** over four years in several independent
compensation threads; one research-engineer offer reported 38% / 32% / 20% / 10%.

## Rules during interviews

| Topic | What applies |
| --- | --- |
| AI tools | "You're free to use AI to help you prepare for interviews. But – unless told otherwise – please don't use any AI tools during live interviews, or during interview tasks." ([interview guide](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)) |
| A pilot with AI tools | In December 2025 GDM's head of talent acquisition described a pilot in coding assessments where candidates are told to use an AI system of their choice and are assessed on how they use it. No candidate write-up checked for this guide mentions it yet. ([Arctic Shores interview](https://www.arcticshores.com/insights/how-google-deepmind-uses-ai-in-recruitment-with-becky-pradal-rogers)) |
| Google's wider changes | In August 2025 the press reported Google considering more in-person interviews, with no formal policy announced ([Business Standard](https://www.business-standard.com/companies/news/google-ai-cheating-job-interviews-in-person-hiring-shift-sundar-pichai-125082600492_1.html)). In May 2026 the press reported a new code-comprehension interview segment in which candidates use a Google-provided Gemini-based assistant, piloted in parts of Google and planned to widen in the second half of 2026 ([Sina Finance](https://finance.sina.com.cn/stock/t/2026-05-08/doc-inhxefhi2265587.shtml)). Neither report names Google DeepMind. |
| Video | Mostly virtual, over Google video-call links ([interview guide](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)) |
| Coding environment | Candidates report a shared editor that runs code, such as CoderPad. |
| Confidentiality | "We'll also send you a standard non-disclosure agreement. You'll need to electronically sign this through a third-party tool, Ironclad." ([interview guide](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)) |

## After a decision

No GDM page on feedback or on a reapplication cool-down was found. **Candidates report:**

- Rejections after passing every technical round, at a final team debrief — in one 2026 report because the team
  "needed someone who could ramp up faster for that specific role".
- Automated rejections that mention a cool-down period before reapplying.
- Written feedback is not guaranteed but happens: one rejected candidate was told of a "strong command of Python"
  and "clean and readable coding".

## How to prepare with this repository

| Stage | Pages |
| --- | --- |
| Initial interviews | [Recruiter screen](behavioral/recruiter-screen/README.md) |
| Coding | [Array scan and depth-first traversal](coding/graph-dfs-warmup/README.md) · [Shortest paths through locked doors](coding/state-space-bfs/README.md) · [Most frequent events in a stream](coding/event-frequency-stream/README.md) · [Token-moving game](coding/token-game-outcomes/README.md) · [Generators, unit tests and weighted sampling](coding/weighted-sampling-generator/README.md) · [Unit conversions as a weighted graph](coding/unit-conversion-graph/README.md) · [Fence painting](coding/fence-painting-strokes/README.md) · [Bit packing](coding/bit-packing-codec/README.md) · [Snapshot array](coding/snapshot-array/README.md) · [k-d tree](coding/kd-tree-nearest-neighbour/README.md) · [Binary search on bitonic arrays and monotone functions](coding/binary-search-monotone/README.md) |
| The ML-knowledge and maths rounds | [ML fundamentals](quiz/ml-fundamentals/README.md) · [LLM fundamentals](quiz/llm-fundamentals/README.md) · [RL for language models](quiz/rl-for-llms/README.md) · [Maths: probability, statistics and linear algebra](quiz/maths-probability-linear-algebra/README.md) · [Maths: matrix computation, statistical tests and information theory](quiz/maths-stats-information-theory/README.md) · [CS fundamentals](quiz/cs-fundamentals/README.md) · [Code comprehension](quiz/code-comprehension/README.md) |
| ML coding and debugging | [Attention with a KV cache](coding/attention-kv-cache/README.md) · [Debugging a classifier](coding/training-loop-debug/README.md) · [Backpropagation](coding/mlp-backprop-from-scratch/README.md) · [EM for a Gaussian mixture](coding/em-gaussian-mixture/README.md) · [Focal loss](coding/focal-loss/README.md) |
| ML and system design | [Gemini app](system-design/gemini-app/README.md) · [LLM evaluation system](system-design/llm-evaluation-system/README.md) · [Reaction factor for pairs of molecules](system-design/molecule-reaction-ml-design/README.md) · [Retrieval-augmented agent](system-design/rag-agent-platform/README.md) · [Recommendation feed](system-design/recommendation-system/README.md) · [Distributed training](system-design/distributed-training/README.md) · [Robot-learning data flywheel](system-design/robot-learning-data-flywheel/README.md) · [Data-centre machine replacement](system-design/datacentre-failure-ml-design/README.md) |
| Final interviews | [Research deep dive and paper defence](behavioral/research-deep-dive/README.md) · [Team-lead and hiring-manager interviews](behavioral/team-lead-rounds/README.md) · [Googleyness and culture](behavioral/googleyness-and-culture/README.md) · [Code review](coding/code-review-round/README.md) |
| Product management | [AI product sense and the AI deep dive](behavioral/pm-ai-product-sense/README.md) |

The [roadmap](ROADMAP.md) orders every page by track and priority.

## Official sources

- Careers, role families, locations and interview stages: <https://deepmind.google/about/careers/>
- Official interview guide (stages, timeline, STAR, AI-tool rule, NDA, hybrid work):
  <https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf>
- Student Researcher Program: <https://deepmind.google/student-researcher-program/>
- GDM fellowships and scholarships: <https://deepmind.google/education/>
- Google PhD Fellowship: <https://research.google/programs-and-events/phd-fellowship/>
- GDM job postings and US salary ranges: <https://job-boards.greenhouse.io/deepmind>
- Google fraud and scams advisory, November 2025:
  <https://blog.google/innovation-and-ai/technology/safety-security/fraud-and-scams-advisory-november-2025/>
- AI tools in GDM coding assessments (interview with GDM's head of talent acquisition, December 2025):
  <https://www.arcticshores.com/insights/how-google-deepmind-uses-ai-in-recruitment-with-becky-pradal-rogers>
- Google considering in-person interviews (Business Standard, 26 August 2025):
  <https://www.business-standard.com/companies/news/google-ai-cheating-job-interviews-in-person-hiring-shift-sundar-pichai-125082600492_1.html>
- A Gemini-assisted code-comprehension interview segment at Google (Sina Finance, 8 May 2026):
  <https://finance.sina.com.cn/stock/t/2026-05-08/doc-inhxefhi2265587.shtml>
