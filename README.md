<div align="center">

# Google DeepMind Interview Notes

**Practice problems modelled on Google DeepMind's coding, ML-knowledge, ML-design, research and behavioural
interviews, plus a guide to the hiring process.<br>
Full problem statements, worked solutions, and code that CI runs on every push.**

English · [中文](README.zh.md)

[![Check](https://github.com/Schuture/Google-DeepMind-Interview-Notes/actions/workflows/check.yml/badge.svg)](https://github.com/Schuture/Google-DeepMind-Interview-Notes/actions/workflows/check.yml)
![Problems](https://img.shields.io/badge/problems-37-blue)
![Languages](https://img.shields.io/badge/languages-English%20%7C%20%E4%B8%AD%E6%96%87-blue)
[![Text: CC BY-NC 4.0](https://img.shields.io/badge/text-CC%20BY--NC%204.0-lightgrey)](LICENSE)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE-CODE)
[![GitHub stars](https://img.shields.io/github/stars/Schuture/Google-DeepMind-Interview-Notes?style=social)](https://github.com/Schuture/Google-DeepMind-Interview-Notes/stargazers)

</div>

> [!NOTE]
> **Unofficial.** This project is not affiliated with, endorsed by or sponsored by Google DeepMind or Google. The
> problems are reconstructed from second-hand accounts of interviews, mostly from roughly the past year, plus a
> few older question types that still recur, and written up from scratch: statements, examples, solutions and code
> are all original. Far fewer detailed accounts of Google
> DeepMind interviews are public than for some other companies, and many give only the topic of a question ("a
> hard BFS variant", "the maths of a core LLM concept"); such pages are original problems built on that topic.
> Treat every page as practice on the kind of problem you may meet, not as the exact question. If you believe
> something here should not be public, [open an issue](https://github.com/Schuture/Google-DeepMind-Interview-Notes/issues)
> and it will be taken down.

## What is in it

| Section | Pages | What they cover |
| --- | ---: | --- |
| [Hiring process guide](PROCESS.md) | 1 | The two ways into Google DeepMind, every stage from the recruiter call to the offer, how the old quiz changed, timelines, posted salary ranges, the Student Researcher Program, and the rules on AI tools during interviews |
| [Coding](#coding-17) | 17 | Graph search in several forms (a depth-first warm-up on an edge list, state-space BFS with path counting, games on graphs, a unit-conversion graph), Python generators with unit tests and weighted sampling, streaming frequency counting, binary search on bitonic arrays and monotone functions, bit packing, snapshots, k-d trees, a code-review round, and ML implementation: attention with a KV cache, backpropagation, EM, focal loss, debugging a training loop |
| [Quiz](#quiz-7) | 7 | The oral knowledge round: ML fundamentals, LLM fundamentals, reinforcement learning for language models, two maths pages (probability, statistics and linear algebra; matrix computation, statistical tests and information theory), computer-science fundamentals, and explaining what a piece of code computes |
| [System design](#system-design-8) | 8 | The Gemini app, an evaluation system for an LLM assistant, predicting a reaction factor for pairs of molecules, a retrieval-augmented agent, a recommendation feed, training a model that does not fit on one accelerator, a robot-learning data flywheel, and predicting which data-centre machines need replacing |
| [Behavioral](#behavioral-5) | 5 | Recruiter screen, research deep dive, research talk and paper defence, Googleyness and culture, team-lead and hiring-manager interviews, and the product-manager loop |

What sets the pages apart:

- **Complete statements.** Each problem is written out in full, with definitions, function signatures and worked
  examples, so you can attempt it without guessing what was meant. Most problems come in three parts that build
  on each other, the way the round does.
- **Solutions you can check.** Every coding, quiz and system-design solution ends with a collapsed block of
  runnable checks: asserts on the examples, a brute force built independently from the statement wherever one can
  be written, numerical verification of every derivation, and, for system design, the estimates recomputed in
  code. CI runs the code of all pages on every push.
- **Pitfalls where they happen.** Instead of a separate list of mistakes, `# NOTE:` comments sit on the line the
  mistake would be made.
- **The oral knowledge round, written down.** The [quiz pages](#quiz-7) give each question the way it is asked and
  the answer the way you would say it, followed by the derivation and a check in code.
- **The process, not only the questions.** The [hiring process guide](PROCESS.md) separates what Google DeepMind
  publishes (linked) from what candidates report, stage by stage.
- **Study tracks.** The [roadmap](ROADMAP.md) orders the pages for research (RS / RE), software engineering,
  applied AI and ML engineering, student researchers and interns, and product managers.
- **English and Chinese.** Every page exists in both languages with identical code.

## How to use it

### 1. Read the process guide

Start with the [hiring process guide](PROCESS.md): which door you are entering by (Google DeepMind's own posting,
or Google's general loop followed by team match), which rounds that means, and how long it takes. "Why DeepMind"
comes up in nearly every round, so the behavioural pages are worth starting early.

### 2. Pick a track and a pace

Work down the track for your role in the [roadmap](ROADMAP.md). ★ shows how often a question comes up, from
★★★★★ (again and again) to ★☆☆☆☆ (rarely), and each track is ordered so that the top of the list pays off first.

| Time you have | [Research (RS / RE)](ROADMAP.md#research-rs--re) | [SWE](ROADMAP.md#software-engineering-swe) | [Applied AI / MLE](ROADMAP.md#applied-ai-and-ml-engineering-applied-ai--mle) | [Student Researcher](ROADMAP.md#student-researcher-and-internships) |
| --- | --- | --- | --- | --- |
| About a week | stages 1–3 | stages 1–2 | stages 1–3 | stage 1 |
| Two to four weeks | stages 1–6 | the whole track | the whole track | the whole track |
| More than that | add stage 7 and a design in your target team's area | add the ML quiz pages | add the research track's ML coding | add the research track's quiz pages |

### 3. Practise one problem

**Coding.** Read only the **Problem** section. Before writing code, note what you would ask the interviewer
(edge cases, tie-breaking, input sizes), then compare with the first lines of the reference solution, which list
the points worth confirming. Give yourself a time limit and take the parts in order, treating each new part as a
change of requirements: extend your code instead of starting over. Run it on the examples, then open the reference
solution and compare approach, complexity and the `# NOTE:` comments. The checks call the functions named in the
statement, so if you keep the same names and signatures you can usually run them against your own code.

**Quiz.** Set a timer of about three minutes per question and answer aloud, then write any derivation the
question asks for on paper. Compare with the reference answer, which leads with what you would say and then
shows the working; the checks recompute every number.

**System design.** Give yourself about 45 minutes and work through the prompt out loud or in a document:
requirements, estimates, API and data model, architecture, then two or three deep dives. Then compare with the
reference solution; the estimates are computed in the checks block, so you can change an input and rerun them.

**Behavioral.** Each page lists what the round asks, what each question probes and how a strong answer is
structured, and ends with an outline to fill in with your own stories. Rehearse them aloud, and keep your
filled-in version in `my/`, which git ignores.

### 4. Read a page

Every page starts with a table (type, priority, difficulty, roles, topics and, where known, format and round)
and has exactly two sections:

| Section | Coding | Quiz | System design | Behavioral |
| --- | --- | --- | --- | --- |
| **Problem** | definitions, then one block per part: task, signature, example | numbered questions under topic headings | the system, its users, the numbers given, the scope | the round and its questions |
| **Reference solution** (collapsed) | per part: idea, derivation, code, with pitfalls as `# NOTE:` comments; then follow-ups and the checks | per question: the spoken answer, then the derivation; then the checks | requirements, data model and API, architecture, deep dives, follow-ups, estimate check | what is probed, how to structure the answer, an outline to fill in |

Roles: **RS** research scientist · **RE** research engineer · **SWE** software engineer · **MLE** machine-learning
engineer · **Applied AI** applied AI engineer · **Intern** internships and Student Researcher positions ·
**PM** product manager · **TPM** technical program manager.

### 5. Run the code

```bash
git clone https://github.com/Schuture/Google-DeepMind-Interview-Notes.git
cd Google-DeepMind-Interview-Notes
pip install -r requirements.txt                              # NumPy, SciPy, scikit-learn
python scripts/run_snippets.py coding/state-space-bfs        # one page
python scripts/run_snippets.py --all                         # every page
```

A page's ```` ```python ```` blocks run top to bottom as one script; ```` ```py ```` blocks are illustrative
(bare signatures, code that contains planted bugs) and are not executed. CI uses Python 3.11.

## Problems

Sorted by priority within each section. A dash under Difficulty means it has not been rated.

<!-- index:begin -->
### Coding (17)

| # | Problem | Priority | Difficulty | Roles | Topics |
| ---: | --- | --- | --- | --- | --- |
| 1 | [Shortest Paths Through Locked Doors](coding/state-space-bfs/README.md) | ★★★★★ | Hard | SWE · RE · MLE · Intern | bfs, state-space-search, bitmask, path-counting, dijkstra, grid |
| 2 | [Most Frequent Events in a Stream](coding/event-frequency-stream/README.md) | ★★★★☆ | Medium | SWE · MLE · Applied AI · Intern | hash-map, sliding-window, frequency-counting, streaming, space-saving, heavy-hitters |
| 3 | [Winning Positions in a Token-Moving Game](coding/token-game-outcomes/README.md) | ★★★★☆ | Hard | SWE · RE · MLE · Intern | game-theory, dfs, retrograde-bfs, topological-order, sprague-grundy, graphs |
| 4 | [Attention with a KV Cache and an Online Softmax](coding/attention-kv-cache/README.md) | ★★★★☆ | Hard | RS · RE · MLE | attention, kv-cache, online-softmax, flash-attention, grouped-query-attention, numerical-stability |
| 5 | [Debugging a Classifier That Does Not Learn](coding/training-loop-debug/README.md) | ★★★★☆ | Medium | RE · RS · MLE · Applied AI | debugging, softmax, data-shuffling, gradient-scaling, momentum, dropout, broadcasting, sanity-checks |
| 6 | [Warm-Up: Scanning an Array and Traversing a Graph Depth-First](coding/graph-dfs-warmup/README.md) | ★★★★☆ | Easy | SWE · MLE · Intern | arrays, binary-search, graph-construction, dfs, iterative-dfs, connected-components, cycle-detection |
| 7 | [Python Generators, Unit Tests and Weighted Sampling](coding/weighted-sampling-generator/README.md) | ★★★☆☆ | Medium | MLE · SWE · RE · Intern | generators, unit-testing, weighted-sampling, prefix-sums, binary-search, alias-method, chi-square-test |
| 8 | [Code Review: Ranking the Defects in a Checkpointing Change](coding/code-review-round/README.md) | ★★★☆☆ | Medium | RE · SWE · MLE | code-review, checkpointing, atomic-writes, reproducibility, data-sharding, testing |
| 9 | [Unit Conversions as a Weighted Graph](coding/unit-conversion-graph/README.md) | ★★★☆☆ | Medium | SWE · MLE · RE · Intern | graph, dfs, bfs, weighted-union-find, consistency-check, floating-point |
| 10 | [Painting a Fence with the Fewest Strokes](coding/fence-painting-strokes/README.md) | ★★★☆☆ | Medium | SWE · MLE · Intern | greedy, divide-and-conquer, arrays, range-minimum, proof-of-optimality |
| 11 | [Bit-Packing Encoders and Decoders](coding/bit-packing-codec/README.md) | ★★★☆☆ | Medium | SWE · RE · Intern | bit-manipulation, varint, zigzag-encoding, frame-of-reference, serialisation |
| 12 | [Snapshot Array: History, Compaction and Diffs](coding/snapshot-array/README.md) | ★★★☆☆ | Medium | SWE · RE · MLE · Intern | hash-map, binary-search, versioning, memory-trade-offs, journaling |
| 13 | [k-d Tree: Nearest Neighbours and Range Queries](coding/kd-tree-nearest-neighbour/README.md) | ★★★☆☆ | Hard | RS · RE · SWE · MLE | kd-tree, nearest-neighbour, pruning, heap, range-search, curse-of-dimensionality |
| 14 | [Backpropagation from Scratch](coding/mlp-backprop-from-scratch/README.md) | ★★★☆☆ | Medium | RS · RE · MLE · Intern | backpropagation, softmax-cross-entropy, gradient-check, initialisation, sgd-momentum |
| 15 | [EM for a Gaussian Mixture: Derive and Implement](coding/em-gaussian-mixture/README.md) | ★★☆☆☆ | Hard | RS · RE · MLE | expectation-maximisation, gaussian-mixture, log-sum-exp, jensen-inequality, k-means, bic |
| 16 | [Focal Loss versus Cross-Entropy](coding/focal-loss/README.md) | ★★☆☆☆ | Medium | MLE · RS · RE · Applied AI | focal-loss, cross-entropy, class-imbalance, numerical-stability, initialisation, gradients |
| 17 | [Binary Search on Bitonic Arrays and Monotone Functions](coding/binary-search-monotone/README.md) | ★★☆☆☆ | Medium | RE · RS · SWE · MLE | binary-search, bitonic-array, exponential-search, monotone-functions, lower-bounds, bisection |

### Quiz (7)

| # | Problem | Priority | Difficulty | Roles | Topics |
| ---: | --- | --- | --- | --- | --- |
| 1 | [ML Fundamentals: Metrics, Losses, Optimisers and Regularisation](quiz/ml-fundamentals/README.md) | ★★★★★ | Medium | RS · RE · MLE · Applied AI · Intern | precision-recall, roc-auc, cross-entropy, logistic-regression, adam, weight-decay, bias-variance, normalisation, huber-loss, gan, distribution-shift |
| 2 | [LLM Fundamentals: Attention, Transformers and the Training Pipeline](quiz/llm-fundamentals/README.md) | ★★★★★ | Hard | RS · RE · MLE · Applied AI · Intern | attention, transformer, rope, kv-cache, scaling-laws, perplexity, tokenisation, post-training, sampling, mixture-of-experts |
| 3 | [RL for Language Models: Policy Gradients, PPO, GRPO and DPO](quiz/rl-for-llms/README.md) | ★★★☆☆ | Hard | RS · RE · MLE | policy-gradient, ppo, grpo, dpo, kl-regularisation, rlhf, reward-hacking, off-policy, importance-sampling |
| 4 | [Maths Quiz: Probability, Statistics and Linear Algebra](quiz/maths-probability-linear-algebra/README.md) | ★★★☆☆ | Medium | RS · RE · MLE · Intern | bayes-theorem, expectation, markov-chains, maximum-likelihood, map-estimation, kl-divergence, svd, matrix-calculus, conditioning |
| 5 | [Maths Quiz: Matrix Computation, Statistical Tests and Information Theory](quiz/maths-stats-information-theory/README.md) | ★★★☆☆ | Medium | RS · RE · MLE · Intern | matrix-multiplication, rank, matrix-inverse, pseudo-inverse, moments, central-limit-theorem, hypothesis-testing, chi-square-test, entropy, mutual-information, integration |
| 6 | [CS Fundamentals Quiz: Memory, Concurrency and Floating Point](quiz/cs-fundamentals/README.md) | ★★☆☆☆ | Medium | SWE · Intern · RE | oop, memory-management, garbage-collection, race-conditions, deadlock, gil, cache-locality, floating-point, amortised-analysis |
| 7 | [Code Comprehension: What Does This Code Compute?](quiz/code-comprehension/README.md) | ★★☆☆☆ | Medium | RS · RE · Intern · SWE | code-reading, convolution, padding, numerical-stability, streaming-statistics, attention-masks, reservoir-sampling |

### System design (8)

| # | Problem | Priority | Difficulty | Roles | Topics |
| ---: | --- | --- | --- | --- | --- |
| 1 | [Design the Gemini App](system-design/gemini-app/README.md) | ★★★★☆ | Hard | SWE · MLE · Applied AI | streaming, conversation-storage, context-management, model-routing, safety-filtering, multimodal-uploads, capacity-planning |
| 2 | [Design an Evaluation System for an LLM Assistant](system-design/llm-evaluation-system/README.md) | ★★★★☆ | Hard | RE · RS · MLE · Applied AI | evaluation, autoraters, side-by-side, bradley-terry, statistical-power, contamination, release-gating |
| 3 | [ML Design: Predicting a Reaction Factor for Pairs of Molecules](system-design/molecule-reaction-ml-design/README.md) | ★★★☆☆ | Medium | MLE · RE · RS | eda, molecular-fingerprints, pairwise-models, symmetry, data-splitting, leakage, graph-neural-networks, active-learning |
| 4 | [Design a Retrieval-Augmented Agent over Company Documents](system-design/rag-agent-platform/README.md) | ★★★☆☆ | Hard | Applied AI · MLE · SWE | rag, vector-index, hybrid-search, access-control, agents, prompt-injection, evaluation |
| 5 | [Design a Recommendation System for a Content Feed](system-design/recommendation-system/README.md) | ★★★☆☆ | Medium | MLE · SWE · Applied AI | candidate-generation, two-tower, ranking, multi-task-learning, feedback-loops, cold-start, ndcg |
| 6 | [Training a Model That Does Not Fit on One Accelerator](system-design/distributed-training/README.md) | ★★★☆☆ | Hard | RE · MLE · RS | data-parallelism, tensor-parallelism, pipeline-parallelism, sharded-optimiser, checkpointing, fault-tolerance, loss-spikes |
| 7 | [ML Design: A Data Flywheel for a Robot Manipulation Policy](system-design/robot-learning-data-flywheel/README.md) | ★★☆☆☆ | Hard | RE · RS · MLE | robot-learning, data-collection, dataset-curation, vision-language-action, evaluation-statistics, deployment-safety |
| 8 | [ML Design: Predicting Which Data-Centre Machines Need Replacing](system-design/datacentre-failure-ml-design/README.md) | ★★☆☆☆ | Medium | MLE · RE · SWE | predictive-maintenance, label-construction, censoring, class-imbalance, categorical-embeddings, survival-analysis, precision-at-k, feedback-loops |

### Behavioral (5)

| # | Problem | Priority | Difficulty | Roles | Topics |
| ---: | --- | --- | --- | --- | --- |
| 1 | [Recruiter Screen: Motivation, Research Interests and Logistics](behavioral/recruiter-screen/README.md) | ★★★★★ | — | All | why-gdm, motivation, research-interests, background, visa, logistics, compensation |
| 2 | [Research Deep Dive: Paper Discussion and Research Talk](behavioral/research-deep-dive/README.md) | ★★★★★ | — | RS · RE · Intern | paper-deep-dive, research-talk, experimental-design, ablations, limitations, research-taste |
| 3 | [Googleyness, Leadership and Culture: Competency Questions](behavioral/googleyness-and-culture/README.md) | ★★★★★ | — | All | star, why-gdm, collaboration, conflict, ownership, ambiguity, leadership, failure, responsibility |
| 4 | [Team-Lead and Hiring-Manager Interviews: Experience, Research Taste and Fit](behavioral/team-lead-rounds/README.md) | ★★★★☆ | — | RS · RE · SWE · MLE | research-taste, ml-experimentation, ramp-up, team-fit, open-ended-problems |
| 5 | [Product Manager Loop: AI Product Sense and the AI Deep Dive](behavioral/pm-ai-product-sense/README.md) | ★★★☆☆ | — | PM | product-sense, ai-product-strategy, agent-metrics, launch-risk, user-insights, ux-for-ai |
<!-- index:end -->

## Contributing

Corrections, new variants and translations are welcome, and so are accounts of recent interviews described in
your own words. Found a wrong answer, a missing edge case, an outdated statement in the process guide or a
sentence that does not make sense? [Open an issue](https://github.com/Schuture/Google-DeepMind-Interview-Notes/issues).
For pull requests, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

The text (prose, tables and diagrams) is licensed under [CC BY-NC 4.0](LICENSE); the code, both in the pages and
under `scripts/`, under the [MIT License](LICENSE-CODE).
