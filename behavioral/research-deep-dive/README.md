# Research Deep Dive: Paper Discussion and Research Talk

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Behavioral · research discussion | ★★★★★ | — | RS · RE · Intern | paper-deep-dive, research-talk, experimental-design, ablations, limitations, research-taste | 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

This round runs in one of three formats, all led by researchers from the team you would join, at the skills
or final interview stage. The paper or project deep dive is about 45 minutes: the interviewer opens with a
request to walk through one piece of your own research — for instance, "tell me about your paper X" — and
takes it as deep as the remaining time allows, at points arguing the opposite position on purpose to see
whether you defend the original claim or update it. The research talk is also about 45 minutes: you present
your own work to the room, questions follow, and some loops add short one-to-one conversations with team
members straight after. Paper presentation and defence is likewise about 45 minutes: you present one of your
own papers, then face a fast sequence of defence questions that moves across motivation, framing, design
choices, assumptions, alternatives, limitations and the next step; motivation and judgement weigh more here
than reciting method detail. A student-researcher loop replaces any of the three with a 45-minute research
deep dive plus a separate 30-minute conversation with the prospective host about a possible project; none of
these formats includes coding. Across all of them, the round tests whether your account of the work holds up
once someone who works in the area pushes on it, not only whether the headline result is real. The questions
fall into six groups.

### The problem and why it matters

- State the exact question the work answers, in one sentence a researcher outside your sub-field can still
  follow.
- Why was this question still open — what did the closest prior approach get wrong, not attempt, or get
  right only under an assumption that does not hold in general?
- If the central claim holds, who changes what they do because of it — a specific kind of practitioner, a
  specific downstream system, or a specific follow-on research direction?
- State the one-sentence contribution, in words that add information beyond the paper's title.
- What is the simplest setting where the effect already shows up, and the setting where it stops mattering?
- Why was the problem worth pursuing when you started, and why was the framing you chose — the specific way
  you turned it into a question with a measurable answer — the right one rather than a different formulation
  of the same underlying issue?

### Method and design choices

- Walk through the method on the smallest example where every step already does something non-trivial.
- Why this approach rather than the most obvious alternative a competent researcher would try first?
- Name the one assumption the method depends on, and what happens to the result when that assumption is
  violated.
- What did you try before arriving at this design that did not work, and what told you it had failed rather
  than that you had implemented it wrong?
- If you removed the one component you are proudest of, what would you predict changes, and did you
  actually run that experiment?
- Which alternative design could still take the place of the one you chose, and what would change in the
  result if it did?
- Why was each important design decision defensible with the information you had at the time, even where a
  later result made a different choice look better in hindsight?

### Experiments

- What are the baselines, and what makes the comparison fair — the same data, the same compute budget, and
  the same tuning effort on every side?
- Which ablation — a controlled run that changes exactly one component of the method — isolates the
  contribution of the specific idea from the contribution of extra engineering effort spent only on your
  method?
- How many seeds — independent reruns that differ only in random initialisation and data order — support
  the headline number, and what is the confidence interval or spread across them?
- What was the total compute budget, and how was it split between the final run and the search that found
  it?
- What result, if you had observed it, would have told you the central claim was wrong?

### Limitations and what next

- Name a regime — an input distribution, a scale, a domain — where the method measurably fails, not one
  where it simply has not been tried.
- With ten times the compute or ten times the data, what is the first experiment you would run, and what
  result would make a second one not worth running?
- Does the mechanism the method relies on still hold at the scale of a frontier model (the largest, most
  capable models being trained at a given time), or does a different effect start to dominate once scale
  changes the picture?
- Concretely, what would this feed into, or need from, the team's current line of work?

### Your contribution

- What did you personally design, run, or decide, separate from what your co-authors did?
- Name a decision you made that turned out to be wrong, and how you found out.
- Which specific figure, experiment, or part of the method would not exist without your share of the
  project?
- Where a co-author disagreed with a choice you made, what was the disagreement, and how was it resolved?

### Pressure and proposals

- The interviewer proposes that the headline result is explained by a confound in the data (a variable
  other than the one you changed that could explain the result just as well) or by tuning effort spent
  unevenly between your method and the baseline — respond.
- Critique a paper you did not write: name one concrete limitation and what a next version would change
  about it — for example, name one limitation of the FlashAttention paper and how a next version would
  address it.
- What would you work on in your first six months here, and why that question rather than another?
- For a student-researcher loop: sketch a 12–24 week project you would run with the prospective host, from
  question to first result.
- The interviewer suggests a simpler baseline "should" have matched your result — is the suggestion right,
  and how would you check?
- For work whose main contribution is an evaluation or a benchmark — a multimodal benchmark, for instance —
  what makes it more than assembling data, and how deep does your own understanding go of how the evaluated
  models were trained and post-trained: the data mixture, supervised fine-tuning, preference or RL training,
  and contamination? This is where such work gets probed hardest.
- What is the most credible next research direction to come out of this work, and which single experiment
  would falsify it first?

## Reference solution

<details>
<summary>Show the reference solution</summary>

Confirm which of the three formats your loop uses before you prepare, since it changes what you build: a deep
dive rewards a spoken story you can compress or expand on demand, a talk rewards a slide deck with its own
time budget, and presentation and defence rewards the same spoken story as the deep dive, compressed further
to survive a faster round of questions. Across any of them, the same discipline pays off — know the two-minute and the ten-minute
version of your story before you walk in, know precisely which numbers you can defend and which you cannot,
and answer each question in one clear sentence before elaborating, rather than pre-empting the follow-up
that would have come next.

### Choosing the paper or project

Pick the piece of work where you can trace every major decision back to a reason, not the one that sounds
best in one sentence. Depth matters more than polish: a project where you can answer two or three follow-ups
deep on each of its biggest decisions holds up better than one where only the headline result is rehearsed.
A project inside the team's own line of work is convenient — it lets the "how does this connect to what we
do" questions answer themselves — but is not a requirement; a project you can defend in real depth in an
adjacent area holds up better than a shallow account of something that only sounds relevant. A co-authored
project is fine as long as you can state your specific share precisely (see Your contribution below); what
is not fine is picking the more impressive-sounding project and hoping the co-author question does not come
up.

### The two-minute and ten-minute story

Prepare both regardless of which format your loop uses: a deep dive can open with nothing more than "tell me
about your paper X" and hand you the whole two minutes to shape, while a talk's opening slide compresses the
same material into about one minute. Either way, the interviewer may say nothing and simply wait once you
finish, which is the cue to continue into the ten-minute version rather than stop.

Two-minute version, about 120 seconds: 15 seconds on the question and why it was open; 30 seconds on the
method's core idea, stated so a competent researcher outside your sub-field follows it; 40 seconds on the
headline result and what it is measured against; 25 seconds on the sharpest limitation, volunteered rather
than waited for; 10 seconds on the one next step.

Ten-minute version adds, in order: the one design decision you would defend most strongly and the
alternative it beat, with the reason, specific to your setting, that the alternative did not work as well;
the full experimental setup — baselines, the ablation that isolates your contribution, and the seed count
behind the headline number; a second limitation together with how you would find out if it mattered; and how
the result connects to an open question in the area beyond your own next step.

### Slide outline

A slide-by-slide structure for the talk format, with time budgets that sum to about 31 of the 45 minutes,
leaving the rest for questions asked as you go or held to the end.

- **Slide 1 (1 min): title and one-line summary** — the question and the headline finding. Not: your
  background or the team's org chart.
- **Slide 2 (3 min): the problem** — who has it, why it has been hard, and what the closest prior approach
  still gets wrong. Not: a full literature review.
- **Slide 3 (4 min): the idea** — the method's core mechanism, in language a competent researcher outside
  your sub-field follows. Not: architecture detail that no later slide refers back to.
- **Slides 4–5 (4 min each): two key design choices, one slide each** — the alternative you did not take and
  the assumption that ruled it out. Not: a list of hyperparameters with no reasoning attached.
- **Slide 6 (2 min): experimental setup** — data, baselines, and compute budget, precise enough that a
  listener could reproduce the comparison. Not: every dataset statistic.
- **Slides 7–8 (3 min each): headline results** — the comparison against baselines, with the seed count and
  the spread. Not: a table with no comparison drawn out loud.
- **Slide 9 (3 min): ablations** — the one or two that isolate the contribution. Not: every ablation you ran.
- **Slide 10 (2 min): limitations** — the one you would volunteer even if not asked. Not: a single vague
  "more work is needed."
- **Slide 11 (2 min): next steps** — how this connects to open questions in the area. Not: a roadmap with no
  first step.
- **Slide 12 (backup, on standby): extended results** — further ablations, or the derivation cut for time.
  Shown only if asked.

### The problem and why it matters

What it probes: whether the motivation is a real question that predates the work, not a justification
assembled afterward for something you had already decided to build.

Skeleton, under 60 seconds: name the question in one sentence; name what the closest prior approach achieves
and the one specific thing it does not — a case it fails on, an assumption it needs, a cost it pays; name,
concretely, who would use a correct answer and what changes for them; end with the one-sentence contribution
stated as a claim about the world, not as a description of the method. Where the interviewer asks directly
why the problem was worth pursuing or why the framing was the right one, answer with the same material rather
than new material: the gap in the prior approach is the reason it was worth pursuing, and the framing was
right if it turned that specific gap into something measurable — name the alternative framing you did not
choose and why it would have measured the wrong thing.

Most common way to lose points: opening with the method — "we propose a new way to..." — before the question
has been named, so the motivation reads as invented to fit work that already existed. Fix: rehearse the
first minute so the question, and why it was still open, comes before any mention of what you built.

### Method and design choices

What it probes: whether the design was actually chosen against real alternatives, or is the only version you
ever tried, reported after the fact as though it were the obvious choice.

Skeleton, 60–90 seconds for the first pass, with a concrete example ready if pressed: state the one
assumption the method leans on; name the most obvious alternative and the reason, specific to your setting,
that it does not work as well; trace the method once on the smallest example where every step already does
something non-trivial; keep the failed first attempt ready as one sentence — what you tried, and the
specific evidence, not a hunch, that told you it had failed. For a question about which alternative design
could still replace yours, name a credible candidate — the rejected alternative from your alternative-design
table below, not a strawman — and say plainly what result would change if it did and what evidence would
make you switch, rather than defending the current design as unbeatable; for a
question about whether a decision was defensible at the time, cite the specific evidence available before you
ran the experiment, separate from evidence that only arrived afterwards.

Most common way to lose points: presenting the final design as though it were the only one considered, with
no failed attempt and no real alternative named. Fix: before the round, write down the version that did not
work and the one number or observation that ruled it out, so it is a specific memory rather than an
improvised admission.

### Experiments

What it probes: whether the comparison is fair and the effect is distinguishable from noise, not whether the
headline number is large.

Skeleton, under 90 seconds, then depth on demand: name the baselines and, in one clause each, why the
comparison is fair — the same data, the same compute budget, the same tuning effort on every side; name the
ablation that isolates your specific idea from the engineering effort spent implementing it; state the
number of seeds behind the headline number and the spread across them; name the one result that would have
falsified the claim and say plainly whether you looked for it.

Most common way to lose points: a single run reported as the result, with a baseline that was not tuned as
carefully as the proposed method. Fix: before the round, have the exact seed count, the compute spent tuning
the baseline specifically, and one sentence on the ablation that isolates the core idea, ready without having
to reconstruct them live.

### Limitations and what next

What it probes: whether you volunteer a real failure mode unprompted, and whether "what's next" names a
genuinely different experiment rather than a bigger version of the current one.

Skeleton, 30–45 seconds per sub-question: name one regime where the method measurably fails, stated as a
fact about the method — "it fails at X" is a different claim from "it has not been tested at X," and only
the first one is a limitation; for ten times the compute or data, name the first experiment and the specific
result that would make a second one not worth running; for frontier scale, name the one thing that would
have to still be true for the method's mechanism to still apply, and say plainly whether you believe it is;
name concretely what this connects to in the team's own line of work.

Most common way to lose points: defending every design choice instead of conceding a real limitation, or
answering "what's next" with a larger version of the same experiment. Fix: before the round, choose the one
limitation you would volunteer even if not asked, and check that your "what's next" answer would produce a
different kind of evidence, not just a bigger number from the same kind of run.

### Your contribution

What it probes: whether "we" stands in for a division of labour you can state precisely, in both directions
— what was yours, and just as clearly, what was not.

Skeleton, 30 seconds: one sentence tying your specific share to something checkable — "I designed and ran
the ablation in Section 5; a co-author built the training infrastructure I ran it on; the final method
choice was a joint call" — followed by one decision that was yours and turned out wrong, with how you found
out.

Most common way to lose points: claiming credit for a co-author's work, or the opposite — using "we" as
cover when asked for your specific share. Fix: before the round, write the one sentence above, naming your
part, a named co-author's part, and what was a joint call.

### Pressure and proposals

What it probes: whether you update on real evidence without collapsing under any pushback, and whether you
have a genuine next research question rather than a rehearsed one.

For a challenge to the result — a claimed confound or uneven tuning effort — restate the interviewer's claim
in your own words before answering it, about 10 seconds; state plainly what your evidence does and does not
rule out, about 15 seconds; name the specific experiment that would distinguish your explanation from
theirs, about 15 seconds; and if the point is right, say so in the first sentence, not the last. Use the same
four moves for a proposed simpler baseline: restate what it would predict, say what your results already
tell you about that prediction, and if you have not checked it, say so and name the run that would.

For a request to critique a paper you did not write, restate its central claim in one sentence, about 10
seconds; name one concrete limitation with the mechanism behind it, about 20 seconds; and say specifically
what a next version would change, about 20 seconds — one real limitation, argued precisely, beats a list of
surface objections. Worked example: FlashAttention computes exact softmax attention by tiling the sequence
into blocks that fit in a GPU's fast on-chip memory, rescaling a running softmax as each block is processed
so the full attention matrix is never written out to the GPU's slower off-chip memory — but the original
kernel parallelises only over the batch dimension and the number of attention heads. One concrete
limitation: for a long sequence combined with a small batch size and few heads, that leaves too few
independent blocks of work to keep a modern GPU's compute units busy, so throughput drops exactly in the
long-context regime the method targets. A next version could additionally split the query blocks along the
sequence dimension across those units and rebalance the work each one does, so less time goes to
coordination between units and more to the matrix multiplications themselves — which is the change
FlashAttention-2 made, and it measurably raised throughput on the same hardware.

For work whose main contribution is an evaluation or a benchmark, expect the round to press hardest here
rather than on a method: state in one sentence what makes the benchmark more than assembled data — a specific
failure mode it isolates that existing benchmarks do not, or a specific axis of difficulty it controls for —
then show depth on the models it evaluates: what is published about each evaluated model's data mixture and
its supervised fine-tuning and preference- or RL-training stages, and the specific contamination check you
ran between the benchmark and common pretraining corpora. A benchmark defended only as "more data, more coverage" reads
as assembly rather than research.

For the six-month question, the most credible next research direction, and for a student-researcher loop's
host-project sketch, use the template below.

Most common way to lose points: conceding every challenge regardless of whether it is right, which reads as
having had no real basis for the original claim; or answering "what would you work on here" with a
restatement of your current project instead of a genuinely new question. Fix: before the round, write out
the one confound most likely to be raised against your own result and check whether it actually survives
your existing evidence, and prepare a first-six-months question that is not simply "more of my current
project."

### The alternative-design table

Build this table before the round for the project you are presenting, and reach for one row whenever a
design-choice question calls for it: name the decision, the credible alternative you rejected — not a
strawman nobody would propose — the trade-off behind rejecting it, and the evidence that would make you
reverse it today, in that order, rather than defending the decision as though no alternative had ever been
considered.

| Decision | Credible rejected alternative | Trade-off behind the rejection | Evidence that would reverse it |
| --- | --- | --- | --- |
| Frozen vision encoder | Fine-tune the vision encoder jointly with the language model | Keeps training cheap and avoids catastrophic forgetting of visual features the pretraining already learned well | A held-out probe showing the frozen features, not the language model, are the bottleneck on a task the fine-tuned encoder would fix |
| Synthetic captions | Human-written captions | Scales to orders of magnitude more images at a fraction of the cost, at some loss of natural variation | A quality audit showing synthetic captions systematically miss a category of detail the downstream task needs |
| Leading with the more widely used benchmark as the headline claim | Leading with a newer, harder benchmark that is less contaminated but less recognised | Makes the result legible against prior published numbers, at the cost of the contamination and saturation risk that comes with a heavily reused test set | An audit showing the widely used benchmark is contaminated or saturated enough that it no longer separates competing methods |

### A template for what's next

The six-month question, the most credible next research direction, and, for a student-researcher loop, the
project sketch with the prospective host, all answer to the same five-part structure. State each part in one
or two sentences; the whole answer runs to about two minutes; and close on the experiment that would prove
the direction wrong, not one chosen because it is likely to succeed — an answer that only confirms is not yet
falsifiable.

- Question: one sentence, phrased as something you do not yet know the answer to, not a foregone conclusion
  dressed up as a question.
- Why now: what changed — a capability, a dataset, a result from your own or someone else's recent work —
  that makes this answerable now rather than two years ago.
- Success metric: the one number or comparison that would tell you, without further argument, that the
  direction is working.
- Risks: the way the plan most probably fails, and what you would do if it does.
- First experiment: the smallest experiment that would falsify the direction if it is wrong, not merely one
  that would look good if it is right; state what a positive and a negative result would each mean.

Fictional example, for a proposal about evaluation contamination:

| Element | Example |
| --- | --- |
| Question | Does contamination between a pretraining corpus and a widely used benchmark inflate its scores enough to change which of two methods looks better? |
| Why now | Two recent evaluations on the same benchmark disagreed with each other, and contamination is one explanation nobody has ruled out |
| Success metric | The ranking between the two methods on the clean split matches, or reverses, the published ranking |
| Risks | The clean split is too small to be conclusive either way; the extra week turns out better spent elsewhere |
| First experiment | Re-score both methods on a held-out split built after the pretraining cutoff, small enough to run in a week |

For a student-researcher loop, size the sketch to 12–24 weeks rather than six months: a first two-to-four
week experiment that tells you early whether the direction is viable, a mid-project checkpoint result you
would show the host, and a final deliverable — one clearly answered question, not a research programme —
sized to the weeks remaining.

### Saying "I don't know" well

Separate a genuine unknown from something you can estimate on the spot. For a number you have not
memorised — a compute budget, an exact seed count, a co-author's precise contribution to a part you did not
build — say plainly that you do not have it exactly, then give the order of magnitude you are confident of,
or say how you would go and find it. Never state a number you do not have as if you do: a stated gap costs
you one question, while an invented number that later turns out wrong undermines every other number you
gave in the round. Between a number you can estimate and a question fully outside your knowledge sits a third
case: a question you cannot answer conclusively but do have a real basis for reasoning about. There, do not
bluff a definite answer — lay out a defensible line of reasoning instead: what you would expect and why, and
what result or observation would distinguish the possibilities. This reads as considered judgement rather
than either false certainty or a shrug. For a question genuinely outside what you have thought about — a
connection to a sub-field you do not follow, an experiment nobody on the project ran — say so directly and, where you can,
name the specific experiment or reading that would let you answer it; a precise account of what you do not
know reads as more informed than a vague attempt to cover it.

### Prep outline

Copy this into `my/` and fill it in with your own paper or project.

```text
Paper or project chosen:
Format confirmed (deep dive / talk / student-researcher / presentation-and-defence):

Two-minute story (about 120s):
  Question and why it was open (15s):
  Method's core idea (30s):
  Headline result and what it's measured against (40s):
  Sharpest limitation, volunteered (25s):
  One next step (10s):

Ten-minute additions:
  Key design decision defended, and the alternative it beat:
  Full experimental setup (baselines / ablation / seed count):
  Second limitation, and how you'd know if it mattered:
  Connection to an open question in the area:

Slide budget (minutes), if the round is a talk:
  1. Title and headline finding:
  2. Problem:
  3. Idea:
  4-5. Two key design choices:
  6. Experimental setup:
  7-8. Headline results:
  9. Ablations:
  10. Limitations:
  11. Next steps:
  12. Backup, shown only if asked:

Why the problem was worth pursuing, and why your framing of it was the right one:

The one assumption the method leans on, and what breaks without it:
The alternative you tried first, and the evidence that killed it:
An alternative design that could still replace yours, and what would change if it did:
Why each important design decision was defensible with the information you had at the time:

Experiments:
  Baselines and why the comparison is fair:
  The ablation that isolates your specific idea:
  Seed count and spread on the headline number:
  The result that would have falsified the claim:

Limitations and what's next:
  One regime where the method measurably fails:
  First experiment with 10x compute/data, and what a negative result would mean:
  What must still hold for the mechanism to apply at frontier scale:
  Connection to the team's own line of work:

Your contribution, one sentence (yours / a named co-author's / joint):
A decision that was yours and turned out wrong, and how you found out:

Likely challenges to the result:
  The confound most likely to be raised, and whether it survives your evidence:
  A simpler baseline someone might propose, and what it would predict:
  If the contribution is an evaluation or a benchmark: what makes it more than assembling data, and how deep
  your understanding goes of how the evaluated models were trained and post-trained:

A paper you did not write, ready to critique:
  Its central claim, in one sentence:
  One concrete limitation, with the mechanism:
  What a next version would change:

Concise narrative (about two minutes): motivation, hypothesis, the decisive design choices, the evidence,
the limitations:

Alternative-design table (decision / credible rejected alternative / trade-off / evidence that would reverse
it), one row per major design choice:
  1.
  2.
  3.

Adversarial rehearsal: every assumption challenged, and your response ready for each:

Six-month proposal / most credible next research direction (or 12-24 week project, for a student-researcher
loop):
  Question:
  Why now:
  Success metric:
  Risks:
  First experiment (chosen to falsify the direction, not just confirm it):

Three numbers you have not memorised exactly, and the order of magnitude you'd give instead:
  1.
  2.
  3.

Ten likely follow-ups, one line each, with a 60-second answer ready:
  1.
  2.
  3.
  4.
  5.
  6.
  7.
  8.
  9.
  10.
```

</details>
