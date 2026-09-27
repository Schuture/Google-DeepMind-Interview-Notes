# Design an Evaluation System for an LLM Assistant

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| ML system design | ★★★★☆ | Hard | RE · RS · MLE · Applied AI | evaluation, autoraters, side-by-side, bradley-terry, statistical-power, contamination, release-gating | 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

An assistant product ships a new candidate model every week. Before a candidate reaches production, it
must clear an evaluation system that decides whether it may replace the current production model — the
model presently serving the product's traffic. The product is evaluated across 12 task domains: coding,
mathematics, summarisation, multilingual chat, tool use, long-form writing, instruction following,
multi-step reasoning, factual question answering, multi-turn dialogue, agentic task completion, and code
review. Domains differ in how much labelled data exists for them, and the system must keep producing a
trustworthy decision as domains are added and as the product's own behaviour shifts underneath it week
over week.

For each domain, a *golden set* is a fixed collection of prompts held out from training and used only for
evaluation, each with enough attached information — a reference answer, a rubric, or a programmatically
checkable outcome — to judge a response to it. An *autorater* is a model, separate from the candidate and
the production model, prompted to grade or compare responses; it is also called an LLM-as-a-judge. A
*side-by-side (SxS) evaluation* shows a judge — human or autorater — the same prompt answered by two
models and asks which response is better, or whether they tie. Running an SxS evaluation over a set of
prompts for a pair of models produces, for that pair, a *win rate*: the fraction of paired comparisons one
model wins, with a tie counted as half a win for each side. *Contamination* is the presence of golden-set
prompts, or their reference answers, in a model's training data, verbatim or near-verbatim, which inflates
a model's measured performance on the golden set relative to its performance on genuinely unseen inputs.

Design the evaluation system: how a golden set and an autorater turn into a per-domain win rate, how the
resulting 12 win rates turn into a single release decision, and how the design keeps that decision
trustworthy as the product and its domains evolve.

Scale this design for:

- 12 task domains.
- Each domain has a golden set of 2,000 prompts.
- An autorater call costs 1/10 of a candidate-model call and takes 3 seconds.
- The human rating budget is 5,000 ratings per week.
- A release decision is due 48 hours after a candidate arrives.
- The target: detect whether a domain's true SxS win rate differs from 50% by at least 2 percentage
  points, at 95% confidence (two-sided) and 80% power.
- Some domains have no additional labelled data beyond their prompts: no reference answers, no rubrics,
  nothing to compare a response against except what the evaluation system itself produces.

In scope: offline evaluation (golden sets, autoraters, SxS evaluation with humans and with autoraters),
the statistical decision rule that turns 12 domains' results into a release verdict, slice analysis,
versioning of datasets, prompts and judges, contamination checks, and the online evaluation that continues
after a candidate launches (A/B tests, user feedback). Out of scope: training the candidate model itself.

Produce:

1. A requirements and scale estimate: the per-domain sample size the stated confidence and power require,
   how that compares with each domain's golden-set size and what the comparison implies; the compute and
   wall-clock time one candidate's evaluation costs; how the weekly human rating budget is allocated.
2. A data model — for prompts, responses, ratings, judge versions and results — and the API and workflow
   of one evaluation run, from a candidate arriving to a release verdict.
3. An architecture diagram and a walk-through of one evaluation run along it.
4. Deep dives into: (a) calibrating an autorater against human ratings — agreement metrics such as Cohen's
   kappa, position bias in SxS comparisons, and cancelling it by swapping response order; (b) aggregating
   many pairwise comparisons among several models into a single ranking with Bradley–Terry (or Elo), with
   a bootstrap confidence interval; (c) the gating rule across 12 domains — the multiple-comparisons
   problem this creates, and non-inferiority margins; (d) a domain with no additional labelled data —
   synthetic prompt generation, rubric-based autorating, and agentic evaluation that runs a task end to
   end against a checkable outcome; (e) contamination — detecting and preventing leakage of a golden set
   into training data.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Worth confirming before designing: what "replace" means — assumed here to mean the candidate is
non-inferior to production in every domain, within a small margin, and superior in at least one, rather
than strictly better everywhere (a bar a genuinely improved model can still miss in a domain or two by
chance) or better only on some blended average across domains (which can hide a real regression in one
domain behind gains in others) — and how a tie between two responses counts toward the win rate — assumed
here to count as half a win for each side, so a domain's win rate stays a single proportion in $[0,1]$ and
one comparison contributes one Bernoulli-like outcome to it, which is what the sample-size derivation below
needs to hold.

### Requirements and scale

**Per-domain sample size.** Treat one domain's SxS comparisons as a sequence of independent outcomes, each
contributing $0$, $0.5$, or $1$ to that domain's win rate, and test whether the true win probability differs
from $p_0=0.5$ (no real difference from production) by the target 2 percentage points, i.e. $p_1=0.52$ (by
symmetry, the same derivation covers $p_1=0.48$). Under $H_0$, the normal approximation to the binomial gives
$\hat p \sim \mathcal N(p_0, p_0(1-p_0)/n)$, so a two-sided level-$\alpha$ test rejects when $|\hat
p-p_0|>z_{\alpha/2}\sqrt{p_0(1-p_0)/n}$. Under $H_1$, $\hat p\sim\mathcal N(p_1,p_1(1-p_1)/n)$, and a test
with power $1-\beta$ needs the rejection boundary to sit $z_\beta$ standard deviations, measured under $H_1$,
below $p_1$. Equating that boundary with the one $H_0$ already fixes and solving for $n$ gives

$$n = \left(\frac{z_{\alpha/2}\sqrt{p_0(1-p_0)} + z_\beta\sqrt{p_1(1-p_1)}}{p_1-p_0}\right)^2.$$

At $\alpha=0.05$ two-sided ($z_{\alpha/2}\approx1.960$) and power $0.80$ ($z_\beta\approx0.842$), this comes
to $n\approx4{,}903$ comparisons, rounded up to $4{,}904$ (computed exactly below).

**Golden set vs. the statistical target.** Each domain's golden set holds 2,000 prompts, so one comparison
per prompt supplies only 2,000 of the 4,904 the target needs — 40.8%. Solving the same relationship for
the $p_1$ that a sample of exactly 2,000 can resolve, instead of for $n$, puts the golden set's own minimum
detectable effect near 3.1 percentage points, not 2. A single pass over the golden set cannot resolve a
genuine 2-point shift at the stated confidence and power; the gate built in (c) below is sized to roughly
what a domain's golden set can honestly resolve, and treats the nominal 2-point target as a number to work
toward by adding independent comparisons — extra sampled generations, or a non-golden prompt pool an
autorater can grade without needing any reference label — rather than one a single pass already provides.

**Compute and time per candidate.** One evaluation run needs a fresh candidate response for every one of
the $12\times2{,}000=24{,}000$ golden prompts; the paired production response is generated once, when a
domain's golden set is created or refreshed, and reused by every later candidate rather than regenerated
each week. Cancelling an autorater's position bias — its tendency to favour whichever response sits in a given
slot, derived in (a) — by running both response orders doubles
the grading load to $48{,}000$ autorater calls, so one candidate's evaluation costs
$24{,}000 + 48{,}000\times0.1 = 28{,}800$ candidate-call-equivalents — only 20% more than the generations
alone, since an autorater call is a tenth of the price. Run serially at the autorater's stated 3-second call
latency (assumed the same order for a generation call, since nothing in the premise says otherwise), the
$72{,}000$ total calls take $216{,}000$ s, 60 hours — already over the 48-hour SLA before a single human
rating is collected, which is why the pipeline runs on a worker pool rather than one call at a time:
fitting it inside 3 of the 48 hours needs $216{,}000/(3\times3{,}600)=20$ parallel workers, leaving the
remaining 45 hours for the human-rating loop, slice analysis and the release decision itself. The workload
is latency-insensitive, high-throughput batch work — closer in shape to an offline inference pool than to a
real-time serving path — and is a natural fit for whatever spare or off-peak capacity the product's own
serving fleet has between traffic peaks, rather than needing dedicated hardware.

**Human rating budget.** Humans calibrate the autorater and adjudicate its least certain verdicts; they do
not replace its coverage, because the week's entire budget of 5,000 ratings is barely enough to cover one
domain's own 4,904-comparison target on its own ($5{,}000/4{,}904\approx1.02\times$), let alone all 12. A
fixed floor of 30 ratings per domain (360 total) keeps every domain's calibration kappa current every week;
the remaining 4,640 ratings adjudicate whichever comparisons the two swapped autorater orders disagreed on,
split across domains in proportion to how many disagreements each domain actually produces rather than
evenly — a domain with a more subjective rubric (long-form writing, say) generates more disagreements than
one with a mostly mechanical check (coding against unit tests) and draws correspondingly more of the
budget, with an even split ($4{,}640/12\approx386.7$ per domain) only as the floor every domain is
guaranteed regardless of its own disagreement rate.

### Data model and API

**Prompt** — `prompt_id`, `domain`, `text`, `source` (`golden | synthetic | production_sample`),
`reference` (a reference answer, rubric, or programmatic checker, where the domain has one; null
otherwise), `golden_set_version`, `created_at`, `contamination_status` (`clean | flagged | retired`).

**Response** — `response_id`, `prompt_id`, `model_role` (`candidate | production`), `model_version`,
`text`, `generation_params`, `generated_at`. A production response is generated once per
`golden_set_version` and reused by every later candidate; a candidate response is generated fresh for its
own evaluation run.

**Comparison** — one pairwise SxS item: `comparison_id`, `run_id`, `domain`, `prompt_id`,
`response_a_id`, `response_b_id`, `candidate_position` (`a | b`, which side the candidate sits on for this
item), `pair_key` (shared by the two items that are the same prompt shown in opposite order, so they can be
recombined at aggregation time).

**Rating** — one verdict on one `Comparison`: `rating_id`, `comparison_id`, `judge_type`
(`autorater | human`), `judge_version`, `verdict` (`a_win | tie | b_win`), `confidence`, `rationale`,
`rated_at`.

**JudgeVersion** — `judge_version_id`, `domain`, `prompt_template_version`, `rubric_version`,
`effective_from`, `rolling_kappa` (against human ratings, updated as calibration ratings arrive).

**EvaluationRun** — `run_id`, `candidate_model`, `candidate_version`, `started_at`,
`status` (`running | complete`), `decision` (`release | block | pending`), `decided_at`, `decided_by` (a
person, set only on a manual override).

**DomainResult** — `run_id`, `domain`, `n_comparisons`, `win_rate`, `non_inferiority_p`, `superiority_p`,
`holm_adjusted`, `verdict` (`non_inferior | regressed`).

Core API:

1. `POST /v1/eval-runs` — `{candidate_model, candidate_version}` → `202 {run_id, status: "running"}`.
   Starts the full 12-domain evaluation for a candidate.
2. `GET /v1/eval-runs/{run_id}` — `{status, domains: [{domain, n_comparisons, win_rate, verdict}],
   decision}`. The overall status and the per-domain rollup.
3. `GET /v1/eval-runs/{run_id}/domains/{domain}/comparisons?filter=` — a domain's individual comparisons,
   filterable to `disagreement` (the ones routed for human adjudication) or `calibration`.
4. `POST /v1/comparisons/{comparison_id}/ratings` — `{judge_type, judge_version, verdict, confidence,
   rationale}`. Called by the autorater service and by the human rating tool alike, so every verdict
   writes through the same path regardless of who produced it.
5. `POST /v1/eval-runs/{run_id}/override` — `{decision, justification}`. A release engineer's manual
   override; `justification` is required and the call is audit-logged.
6. `GET /v1/judges/{domain}` — `{judge_version, rolling_kappa, last_calibrated_at}`. The current judge and
   its calibration health, read by the drift check in the follow-ups below.

### Architecture

```text
  candidate model
  arrives weekly
         |
         v
  +------------------------+          +----------------------+
  | Generation service     |<---------| Golden-set store     |
  | candidate responses;   |          | (prompts, by domain) |
  | production responses   |          +----------------------+
  | are cached, not redone |            |
  +------------------------+            | periodic scan
         |                            +--------------------------+
         v                            | Contamination checker    |
  +-----------------------+           | (n-gram overlap, canary) |
  | SxS scheduler         |           +--------------------------+
  | pairs candidate with  |
  | cached production,    |
  | both orders, routes a |
  | slice to humans       |
  +-----------------------+
            |                        |
            v                        v
  +-------------------+      +--------------+
  | Autorater service |      | Human rating |
  +-------------------+      | queue / tool |
                             +--------------+
            |                        |
            v                        v
  +---------------+
  | Ratings store |
  +---------------+
          |
          v
  +------------------------------------+
  | Aggregation and stats engine:      |
  | win rate, kappa, Bradley-Terry,    |
  | Holm-adjusted non-inferiority gate |
  +------------------------------------+
                     |
                     v
  +-----------------------+
  | Release-gate decision |
  +-----------------------+
             |                            |
        pass                                fail
             v                            v
  +--------------------+      +----------------------+
  | Production rollout |      | Blocked; report back |
  | + online A/B hook  |      | to the team          |
  +--------------------+      +----------------------+
```

A candidate's arrival creates an `EvaluationRun`. For each of the 12 domains, the generation service
produces one response per golden-set prompt — 24,000 calls in total — while the SxS scheduler reads that
domain's cached production responses rather than calling the production model again. The scheduler writes
two `Comparison` rows per prompt, one with the candidate in each position, and sends both to the autorater
service; a 20-worker pool (sized in requirements and scale above) grades every one and writes a `Rating`
row through the same endpoint the human tool uses, so aggregation never has to special-case which judge
produced a verdict. Alongside the full autorater pass, the scheduler routes a fixed calibration slice and
every comparison whose two order-swapped verdicts disagree to the human rating queue; a human verdict on a
routed comparison replaces the autorater's verdict in that comparison's tally, and the calibration slice
separately updates the domain's `JudgeVersion.rolling_kappa`. Once every domain's comparisons are rated,
the aggregation engine computes each domain's win rate, runs the non-inferiority and superiority tests from
(c), and applies the Holm adjustment across all 12 domains at once; the release-gate decision records
`release` only if every domain clears non-inferiority and at least one clears superiority, and `block`
otherwise, before notifying the team and, on a release, hopping to the online A/B hook. A separate,
periodic job — not on any single candidate's critical path — runs the contamination checker from (e)
against the golden-set store and flags prompts for rotation before they can affect a future decision.

### Deep dives

**(a) Calibrating an autorater against human ratings.** Calibration asks two separate questions: whether
the autorater's verdicts agree with a human's on the same comparisons, and whether the autorater carries a
systematic bias unrelated to quality.

Agreement is measured with Cohen's kappa rather than raw agreement, because raw agreement rewards a judge
that defaults to the majority class often for reasons that have nothing to do with tracking the human
verdict: with three outcomes (`production_win`, `tie`, `candidate_win`) and a domain where genuine ties are
common, two judges that both lean toward "tie" most of the time agree often by chance alone. Cohen's kappa
corrects for this by comparing the observed agreement rate $p_o$ against the rate two independent judges
with the same marginal distributions would produce by chance, $p_e$:

$$\kappa = \frac{p_o - p_e}{1 - p_e}.$$

On the 200-comparison calibration confusion matrix checked below, $p_o=0.80$ but $p_e\approx0.34$, giving
$\kappa\approx0.70$ — conventionally "substantial" agreement, not "almost perfect": enough to trust the
autorater for full-coverage grading, not enough to stop sampling human verdicts altogether, which is why
the calibration slice of the human budget keeps running every week rather than once.

Position bias is a different failure: a shift in the verdict that depends on which side of the pair a
response is shown on, independent of which response is actually better. A judge — human or autorater —
that leans toward whichever answer appears first reports a higher win rate for the candidate when the
candidate is always shown first than its true quality warrants, and a lower one if it is always shown
second. Detecting it needs nothing more than running the same pair in both orders and checking whether the
verdict flips when nothing about the two responses did. Cancelling it follows from the same idea: if
showing the candidate first adds a bias $b$ to its apparent score and showing it second subtracts the same
$b$ (production now occupies the position the bias favours), then averaging the two orders' scores for one
prompt cancels $b$ exactly, leaving only the two judgments' independent noise:

$$\mathbb E\left[\frac{s_{\text{first}}+s_{\text{second}}}{2}\right] = \frac{(\text{true}+b) + (\text{true}-b)}{2} = \text{true}.$$

A single order, or an order picked at random per item, only cancels $b$ in aggregate over many prompts, not
on any one comparison, so it has the same expectation as the doubled design but a noisier estimate at the
item level; running both orders for every comparison — affordable here, since an autorater call is a tenth
of a candidate-model call — is what the compute budget above already assumes. The checks below simulate a
judge with an 8-point position bias against a candidate with a true 55% win rate: a single, always-first
ordering measures a win rate close to 63%, and the order-swapped estimate lands back close to 55%.

**(b) Aggregating many pairwise comparisons into one ranking.** A single week compares exactly two models,
but the same golden-set infrastructure also runs whenever several candidates are compared at once —
competing fine-tunes, or a rolling leaderboard of past candidates — and pairwise SxS verdicts among more
than two models do not automatically produce one ranking: model A can beat B, B can beat C, and A vs. C
might never have been run directly. The Bradley–Terry model turns a set of pairwise win/loss counts into
one ranking by giving every model $i$ a latent strength $\pi_i>0$ (equivalently $\theta_i=\log\pi_i$) and
modelling

$$P(i \text{ beats } j) = \frac{\pi_i}{\pi_i+\pi_j} = \frac{1}{1+e^{-(\theta_i-\theta_j)}},$$

a logistic function of the strength gap, ties handled by counting as half a win to each side, exactly as
the win rate elsewhere on this page already does. Maximising the resulting log-likelihood over all observed
comparisons has no closed form, but setting its derivative with respect to each $\theta_i$ to zero yields a
simple fixed-point characterisation — the Zermelo/Hunter minorisation–maximisation update — that converges
monotonically from any positive starting point:

$$\pi_i \leftarrow \frac{W_i}{\displaystyle\sum_{j\neq i} \dfrac{n_{ij}}{\pi_i+\pi_j}},$$

where $W_i$ is model $i$'s total win count across all opponents and $n_{ij}$ the number of times $i$ and
$j$ were compared. Iterating this update for every model and renormalising — the strengths are identified
only up to a common scale, since multiplying every $\pi_i$ by the same constant leaves every pairwise
probability unchanged — converges to the maximum-likelihood strengths. The checks below fit four simulated
models this way and recover their true strength order exactly.

A point estimate alone hides how much of a ranking is noise from a finite number of comparisons. A
bootstrap confidence interval resamples, independently for every pair, that pair's comparisons with
replacement, refits Bradley–Terry on each resample, and reports the spread of the resulting pairwise win
probability across resamples — no assumption about the sampling distribution's shape is needed, only that
the resampling repeats whatever randomness actually generated the observed comparisons. The checks below
bootstrap a 95% interval for one pair's win probability and confirm it covers the true simulated value.

**(c) The gating rule across 12 domains.** Testing all 12 domains independently at $\alpha=0.05$ does not
keep the release process's overall false-positive rate at 5%. If a candidate is genuinely no different from
production in every domain, the chance that at least one of the 12 independent tests rejects its null
anyway is

$$1-(1-\alpha)^{12} \approx 0.46$$

— not a 5% chance of a wrong call, but closer to a coin flip (checked below). The standard fix is to test
each domain at a stricter level. Bonferroni's is the simplest: test every domain at $\alpha/12$, which
controls the family-wise error rate because a union bound over 12 events each below $\alpha/12$ cannot
exceed $\alpha$. Holm's step-down procedure controls the same quantity while rejecting at least as much:
sort the 12 p-values ascending and compare the $k$-th smallest against $\alpha/(12-k+1)$, stopping at the
first comparison that fails — everything before it is rejected, everything from it on is not. The first
comparison, against $\alpha/12$, is exactly Bonferroni's; every comparison after it is against a larger
threshold, so Holm can only reject a hypothesis Bonferroni would have missed, never the reverse (checked
below on a small example where Holm rejects three hypotheses to Bonferroni's one).

What each domain's test should actually ask is not "is the candidate better", because the opening question
of what "replace" means already ruled that out as too strict on its own, and because a test sized for 80%
power at exactly a 2-point difference will, by construction, fail to clear a domain that truly sits right
at that difference one time in five — not a margin a production gate should fail a genuinely tied domain
against. Each domain instead runs a one-sided non-inferiority test: $H_0$, the candidate's true win rate is
at or below $0.5-\delta$ (a real regression of at least $\delta$), against $H_1$, that it is above that.
$\delta$ is set to 3 percentage points — wider than the nominal 2-point target and close to the golden
set's own 3.1-point resolution limit found above, wide enough that a domain genuinely tied with production
clears the Holm-adjusted threshold with room to spare, which the nominal 2-point target does not once all
12 thresholds have to be cleared at once (checked below). A candidate passes the domain gate only when
every one of the 12 domains is non-inferior at the Holm-adjusted level; it earns a release, rather than a
shrug, only when at least one domain is also significantly superior at $0.5$ under the same correction —
the second half of the answer to what "replace" means: no worse anywhere by more than $\delta$, and better
somewhere.

**(d) A domain with no additional labelled data.** Agentic task completion — running a multi-step,
tool-using task end to end — is the domain on this list likeliest to have no additional labelled data at
all: no reference answer to compare against, sometimes not even an established prompt set, because the
domain itself is newer than the golden-set infrastructure built for the other eleven. Building it from
nothing runs in three stages.

Synthetic prompt generation produces the prompt pool that has to exist before anything else can happen: a
generator model, prompted with the task family's shape ("book an itinerary meeting these constraints",
"find and fix a failing test"), produces candidate prompts across whatever combination of parameters —
destinations, constraints, starting repository state — gives the coverage the domain needs, followed by
deduplication against near-identical outputs and a review pass, human or a stronger checking model, before
any prompt is trusted enough to affect a release decision. This produces volume cheaply but not
correctness: a generated prompt can be ambiguous, impossible, or already answerable without using any tool,
and the review pass exists to catch exactly that.

Rubric-based autorating replaces the SxS comparison used everywhere else on this page with a checklist
graded against a single response, because there is no reference answer to compare it to and, for many
agentic prompts, no single correct transcript either — only properties a correct one must have. A rubric
for one prompt is a short list of binary criteria ("called the search tool before booking", "never invents
a flight number", "total cost is within the stated budget"); the autorater grades each criterion
independently against the transcript, and the prompt's score is the fraction satisfied. This needs no
golden answer, only a rubric — cheaper to author than a full reference solution, though not free — and it
inherits an ordinary autorater's own calibration needs from (a): a rubric grader is measured against human
graders on the same rubric exactly as an SxS autorater is measured against human SxS verdicts.

Agentic evaluation goes one step further where the task allows it: instead of grading a transcript at all,
run the candidate's tool calls to completion inside a sandboxed copy of whatever environment the task
needs, and check the resulting end state against a programmatic oracle — did the test suite the agent was
fixing pass, does the itinerary's total cost fall under the stated budget, does the calendar hold the
meeting at the requested time. This sidesteps rubric authoring and any grading judgment call entirely for
the part of a task that has a checkable outcome, at the cost of only applying to domains where "correct"
has an executable definition — it has nothing to offer a domain like long-form writing, where rubric-based
autorating remains the fallback. A domain built this way carries a contamination risk the curated golden
sets do not carry in the same way: the prompt generator, and its reviewer, are themselves models, and if
one of them or a close relative ever appears in a candidate's training data, the generated prompts — and a
rubric or oracle written to match their exact phrasing — can leak into training alongside them; (e) covers
detecting this.

**(e) Contamination.** Contamination is a property of the golden set and the training data, not of any one
candidate, so detecting it is a standing check against the golden-set store rather than a step inside a
single evaluation run. The cheapest detector is exact overlap: break both the golden text (a prompt and,
where one exists, its reference answer) and any document under suspicion — a training-data shard, a web
crawl the training pipeline draws on — into overlapping word $n$-grams, and measure what fraction of the
golden text's $n$-grams also occur in the document. A high fraction, especially from one long unbroken run,
is the signature of the golden text sitting inside that document close to verbatim; published contamination
studies typically use $n$ around 8–13 words, long enough that a match is not just two documents sharing an
ordinary short phrase by chance. The checks below implement this at a smaller $n$ against short toy strings
and confirm it catches a verbatim copy and, just as tellingly, misses a paraphrase of the same sentence
entirely — an $n$-gram detector only ever proves textual overlap, never semantic leakage, so a paraphrasing
pass between a golden prompt and a training document defeats it completely; closing that gap needs an
embedding-similarity pass over the same candidate documents, at a much higher compute cost per comparison,
run only where exact-match screening cannot already rule contamination out on its own.

A second, cheap technique is a canary: a unique string, generated once and never used for anything else,
inserted into a copy of a golden-set prompt that is otherwise never published or logged anywhere a training
pipeline could plausibly collect it. If a canary ever comes back out of a candidate — verbatim, in a
completion, under a prompt that never contained it — the golden set leaked through some channel the
exact-match scan does not otherwise cover, without needing access to the training corpus itself to find
out.

Detection only closes half the loop; the other half is not needing to detect a leak in the first place. A
golden set's prompts are excluded from anything that could plausibly feed a future training corpus — public
releases, logged eval traffic sent back through the product's own data pipeline — and versioned, so a
specific prompt's provenance and every candidate it has ever been shown to are both recoverable. Since a
prompt's exposure only grows the longer it survives, a fraction of each domain's golden set is retired and
replaced on a fixed schedule, capping how long any single prompt's contamination risk can accumulate, and a
further held-out slice is never included in anything published even internally, so its exact contents
cannot leak through a report that quotes example prompts.

### Follow-ups

- **Safety evaluated separately, with its own gate.** A jailbreak or a harmful-content regression is not
  something a 2-point win-rate shift is built to catch — a candidate can win 51% of ordinary quality
  comparisons while failing a request it should have refused — so safety runs its own golden set, its own
  autorater tuned to a harm rubric rather than a quality comparison, and a fixed pass/fail bar with no
  non-inferiority margin: any regression on the safety benchmark blocks the release regardless of what the
  12 quality domains decided.
- **Long-context and multimodal evals change cost, not the statistics.** A 100,000-token context or an
  image input multiplies the generation and grading latency, and the autorater's own context budget, well
  past the 3-second figure used above, which pushes the compute-and-time estimate up but leaves the
  sample-size derivation, the calibration method and the gating rule unchanged — all three operate on a win
  rate, indifferent to how long the underlying prompt was.
- **Cost control: cascade the judge.** The 20% autorater overhead computed above assumes every comparison
  gets a full autorater grading in both orders; a cheaper policy runs a fast, low-cost autorater pass first
  and escalates only disagreements — between the two swapped orders, or between the autorater and its own
  repeated judgment on the same pair — to a slower, more capable autorater or to the human queue, trading a
  small loss of full-coverage precision for a proportional cut in the autorater budget on whichever
  fraction of comparisons are unambiguous enough not to need it.
- **Detecting autorater drift.** An autorater is itself a model, and updating it — a new version, a new
  judge prompt — changes what a given response scores without any change to the candidate being judged,
  which would masquerade as a shift in every domain's win rate at once if left unaccounted for. Freezing
  `JudgeVersion` per domain and re-running the calibration slice against a new judge version before it
  replaces the old one in production catches this before it reaches a release decision — the same
  mechanism (a) already runs continuously, applied to a judge change rather than a candidate change.
- **Online-offline correlation.** The offline gate is only useful if it predicts what an A/B test shows
  after launch; tracking, release over release, the correlation between a domain's offline SxS win rate and
  its online metric movement (task success, thumbs-up rate, retention) is what would catch the gate
  quietly drifting away from what users actually experience — a golden set that stops matching production
  traffic's real distribution, or an autorater that has learned to reward something users do not actually
  prefer.

<details>
<summary>Estimate check (runnable)</summary>

```python
import math
import re
from itertools import combinations

import numpy as np
from scipy import stats

# ---- per-domain sample size for the stated power ----
def required_n(p1: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> float:
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(p1 * (1 - p1))) ** 2 / (p1 - p0) ** 2

n_exact = required_n(p1=0.52)
assert round(n_exact, 1) == 4903.2
n_domain = math.ceil(n_exact)
assert n_domain == 4904

domains = 12
golden_set_size = 2_000
total_prompts = domains * golden_set_size
assert total_prompts == 24_000

coverage = golden_set_size / n_domain
assert round(coverage, 3) == 0.408

def mde_at_n(n_target: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> float:
    """Invert required_n by bisection: the p1 whose required sample size is exactly n_target."""
    lo, hi = p0 + 1e-6, p0 + 0.3
    for _ in range(100):
        mid = (lo + hi) / 2
        if required_n(mid, p0, alpha, power) < n_target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

mde_pp = (mde_at_n(golden_set_size) - 0.5) * 100
assert round(mde_pp, 1) == 3.1

# ---- compute and time per candidate ----
gen_calls = total_prompts                         # one candidate response per golden prompt
autorater_calls = total_prompts * 2                # both orders, to cancel position bias -- see (a)
assert gen_calls == 24_000 and autorater_calls == 48_000

autorater_cost_ratio = 0.1                         # of one candidate-model call
compute_units = gen_calls * 1 + autorater_calls * autorater_cost_ratio
assert compute_units == 28_800
assert round(compute_units / gen_calls, 2) == 1.20

call_latency_s = 3                                 # given for an autorater call; assumed same order for generation
serial_seconds = (gen_calls + autorater_calls) * call_latency_s
assert serial_seconds == 216_000
assert serial_seconds / 3600 == 60.0

sla_hours, target_pipeline_hours = 48, 3
workers_needed = math.ceil(serial_seconds / (target_pipeline_hours * 3600))
assert workers_needed == 20
assert sla_hours - target_pipeline_hours == 45

# ---- human rating budget ----
human_budget = 5_000
calibration_per_domain = 30
calibration_total = calibration_per_domain * domains
assert calibration_total == 360
adjudication_budget = human_budget - calibration_total
assert adjudication_budget == 4_640
adjudication_per_domain = adjudication_budget / domains
assert round(adjudication_per_domain, 1) == 386.7
assert round(human_budget / n_domain, 2) == 1.02        # the whole week's human budget barely covers one domain

print("all requirements-and-scale numbers check out")

# ---- (a) Cohen's kappa and position-bias cancellation ----
def cohens_kappa(confusion: list[list[int]]) -> float:
    conf = np.array(confusion, dtype=float)
    total = conf.sum()
    p_o = np.trace(conf) / total
    row_marg, col_marg = conf.sum(axis=1) / total, conf.sum(axis=0) / total
    p_e = float(np.dot(row_marg, col_marg))
    return (p_o - p_e) / (1 - p_e)

# a calibration confusion matrix: rows = human verdict, columns = autorater verdict, over
# {production_win, tie, candidate_win}, 200 calibration comparisons
confusion = [[70, 8, 2], [10, 40, 10], [3, 7, 50]]
assert sum(sum(row) for row in confusion) == 200
kappa = cohens_kappa(confusion)
assert round(kappa, 3) == 0.696

def simulate_position_bias(seed: int, n_prompts: int, true_p: float, bias: float):
    """true_p is the candidate's true win probability under an unbiased judge; bias is the
    autorater's own boost to whichever response it sees in the first position. 'naive' always
    shows the candidate first; 'swapped' runs both orders and averages the two verdicts."""
    rng = np.random.default_rng(seed)
    p_first, p_second = min(true_p + bias, 1.0), max(true_p - bias, 0.0)
    naive = (rng.random(n_prompts) < p_first).mean()
    score_first = (rng.random(n_prompts) < p_first).astype(float)
    score_second = (rng.random(n_prompts) < p_second).astype(float)
    swapped = ((score_first + score_second) / 2).mean()
    return naive, swapped

naive_rate, swapped_rate = simulate_position_bias(seed=7, n_prompts=3_000, true_p=0.55, bias=0.08)
assert abs(naive_rate - 0.63) < 0.02 and abs(naive_rate - 0.55) > 0.05     # naive: biased by ~the position boost
assert abs(swapped_rate - 0.55) < 0.02                                    # swap-and-average: bias cancels

print("kappa and position-bias checks confirm the calibration and de-biasing claims")

# ---- (b) Bradley-Terry ranking (MM fit) and a bootstrap confidence interval ----
def simulate_pairwise(theta_true: np.ndarray, n_per_pair: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = len(theta_true)
    wins = np.zeros((m, m))
    for i, j in combinations(range(m), 2):
        p_i_beats_j = 1.0 / (1.0 + math.exp(-(theta_true[i] - theta_true[j])))
        i_wins = int((rng.random(n_per_pair) < p_i_beats_j).sum())
        wins[i, j], wins[j, i] = i_wins, n_per_pair - i_wins
    return wins

def fit_bradley_terry(wins: np.ndarray, n_iter: int = 200, eps: float = 1e-9) -> np.ndarray:
    """Zermelo/Hunter MM fixed point for the Bradley-Terry log-likelihood: pi_i is updated to
    its total wins divided by the sum, over opponents j, of n_ij / (pi_i + pi_j); iterating this
    update converges to the maximum-likelihood strengths, unique up to a common scale factor."""
    m = wins.shape[0]
    n_ij = wins + wins.T
    total_wins = wins.sum(axis=1)
    pi = np.ones(m)
    for _ in range(n_iter):
        denom = np.array([sum(n_ij[i, j] / (pi[i] + pi[j]) for j in range(m) if j != i) for i in range(m)])
        pi = total_wins / np.maximum(denom, eps)
        pi = pi / pi[0]                          # anchor model 0's strength at 1: BT strengths need a scale
    return pi

theta_true = np.array([0.0, 0.7, -0.5, 1.5])       # log-strengths of 4 models, e.g. 4 candidates in one week
wins = simulate_pairwise(theta_true, n_per_pair=200, seed=3)
theta_fit = np.log(fit_bradley_terry(wins))
assert list(np.argsort(theta_true)) == list(np.argsort(theta_fit))        # recovers the true ranking

def bootstrap_pair_ci(wins: np.ndarray, pair: tuple[int, int], n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    m = wins.shape[0]
    n_ij = wins + wins.T
    i, j = pair
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        boot_wins = wins.copy()
        for a, c in combinations(range(m), 2):                # paired bootstrap: resample within each pair
            n_ac = int(n_ij[a, c])
            resampled = rng.binomial(n_ac, wins[a, c] / n_ac)
            boot_wins[a, c], boot_wins[c, a] = resampled, n_ac - resampled
        pi_boot = fit_bradley_terry(boot_wins, n_iter=150)
        estimates[b] = pi_boot[i] / (pi_boot[i] + pi_boot[j])
    return np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])

true_p_1_beats_0 = 1.0 / (1.0 + math.exp(-(theta_true[1] - theta_true[0])))
ci_lo, ci_hi = bootstrap_pair_ci(wins, pair=(1, 0), n_boot=300, seed=11)
assert ci_lo < true_p_1_beats_0 < ci_hi
assert 0.02 < (ci_hi - ci_lo) < 0.25

print("Bradley-Terry MM fit recovers the true ranking; the bootstrap CI covers the true win probability")

# ---- (c) multiple comparisons across 12 domains and a non-inferiority gate ----
def holm_reject(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])          # ascending p-value order
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break                                              # step-down: stop at the first non-rejection
    return reject

fwer_unadjusted = 1 - (1 - 0.05) ** domains
assert round(fwer_unadjusted, 2) == 0.46

demo_p = [0.005, 0.009, 0.011, 0.02, 0.03, 0.5]
bonferroni_rejections = sum(p <= 0.05 / len(demo_p) for p in demo_p)
holm_rejections = sum(holm_reject(demo_p, 0.05))
assert bonferroni_rejections == 1 and holm_rejections == 3       # Holm is uniformly more powerful here

def one_sided_p(p_hat: float, p0: float, n: int) -> tuple[float, float]:
    z = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)
    return 1 - stats.norm.cdf(z), z

def evaluate_release(true_win_rates: list[float], seed: int, margin: float = 0.03, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    p_hats = rng.binomial(n_domain, true_win_rates) / n_domain
    ni_p = [one_sided_p(p, 0.5 - margin, n_domain)[0] for p in p_hats]     # H0: worse than production by > margin
    sup_p = [one_sided_p(p, 0.5, n_domain)[0] for p in p_hats]            # H0: no better than production
    non_inferior = holm_reject(ni_p, alpha)
    superior = holm_reject(sup_p, alpha)
    return all(non_inferior) and any(superior), non_inferior, superior

# scenario A: eleven domains genuinely tied with production, one genuinely 6 points better -> release
gate_a, ni_a, sup_a = evaluate_release([0.50] * 11 + [0.56], seed=101)
assert gate_a is True and all(ni_a) and sup_a[-1] and sum(sup_a) == 1

# scenario B: the same eleven tied domains, but one domain has genuinely regressed -> block
gate_b, ni_b, sup_b = evaluate_release([0.50] * 11 + [0.40], seed=202)
assert gate_b is False and ni_b[-1] is False and sum(ni_b) == 11

print("the multiple-comparisons gate releases a genuine improvement and blocks a genuine regression")

# ---- (e) n-gram overlap contamination detector ----
def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

def contamination_overlap(golden: str, document: str, n: int = 5) -> float:
    """Fraction of golden's n-grams that also occur in document -- 1.0 means golden's text sits
    inside document somewhere, word-for-word, in an unbroken run of at least n words."""
    g = word_ngrams(golden, n)
    return len(g & word_ngrams(document, n)) / len(g) if g else 0.0

golden_prompt = "the quick brown fox jumps over the lazy dog near the river bank"
contaminated_doc = "some preamble text before it. " + golden_prompt + ". and some text after."
unrelated_doc = "interest rates rose sharply after the central bank meeting concluded yesterday afternoon"
paraphrased_doc = "a fast auburn fox leaps above a sleepy dog beside the riverbank"

assert contamination_overlap(golden_prompt, contaminated_doc) == 1.0
assert contamination_overlap(golden_prompt, unrelated_doc) == 0.0
assert contamination_overlap(golden_prompt, paraphrased_doc) == 0.0      # exact n-grams miss paraphrase entirely

print("n-gram contamination detector flags the verbatim copy and misses the paraphrase")
print("all checks passed")
```

</details>

</details>
