# Team-Lead and Hiring-Manager Interviews: Experience, Research Taste and Fit

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Behavioral | ★★★★☆ | — | RS · RE · SWE · MLE | research-taste, ml-experimentation, ramp-up, team-fit, open-ended-problems | 45 min | Hiring-manager screen · Final interview · Team match |
<!-- meta:end -->

## Problem

This round consists of one or two 45-minute conversations, each with a team lead — the person who leads the
specific team a role would join, sometimes a senior team lead or a director instead — or with the hiring
manager of that team (both are called the team lead below), plus shorter conversations with prospective
teammates on the same team. It can come at the final stage of the process, after the technical rounds, or as
the first conversation of the process, before any of them are scheduled; either way, the questions are
open-ended but specific to the team's own domain rather than a generic script. Unlike the coding, research
and system-design rounds elsewhere in the loop, this conversation is mostly about you: your experience in
depth, how you actually work day to day, and what you would want to work on if you joined. An open-ended
problem drawn from the team's own area can come up inside it, posed the way a real research, engineering or
product question would be, not as a scored exercise. What these conversations settle is separate from the
technical rounds: the team holds its own debrief afterwards on whether it wants you specifically and at what
scope, so you can pass every technical round and still not be matched to a team here. The questions fall into
five groups.

### Experience in depth

- Walk through the project you are proudest of: the problem it addressed, and precisely what your own part
  in it was, separate from what the rest of the team did.
- Describe how you design an experiment to test a specific hypothesis: the baseline you compare against, the
  ablation — a controlled run that changes exactly one part of the setup — that would isolate the effect you
  care about, and how many seeds — independent reruns that differ only in random initialisation and data
  order — you want before you trust the headline number.
- Describe the largest-scale system or training run you have personally worked on, in a concrete number —
  parameters, tokens, GPU-hours, or requests per second — and the specific part of it you owned.
- Describe a time a result surprised you: how you decided whether it was a real effect rather than noise, a
  bug, or an artifact of how it was measured.
- What is the most complex piece of infrastructure or tooling you have built or maintained, and who else
  depended on it?
- If you had to defend one technical decision from that project against someone who disagreed with it, which
  one, and why?

### Research taste and direction

- What would you want to work on in your first six months here, and why that direction rather than
  another — and how exactly would you execute it: the first experiment, the data, and the milestone that
  would show it is working?
- Can current theory fully explain the effect you would build on? If not, what is missing, and does it
  matter for your plan?
- What do you think of current systems in the team's area? For a team working on multilingual models, for
  instance, what are the main limitations of today's multilingual models?
- Name one direction in your area that you think is currently overrated, and one you think is underrated —
  and the reason for each.
- Describe how you decide to stop a project rather than keep pushing on it.
- Describe a specific result — your own or someone else's — that changed your mind about something you had
  believed.
- When you have several plausible next experiments and cannot run all of them, how do you decide which one
  goes first?

### Working style on a research team

- Describe how you would divide work between a research scientist and a research engineer on the same
  project, and where the line between the two roles gets blurry in practice.
- How do you decide how much to invest in a piece of research code — tests, documentation, a clean interface
  — versus how fast you can get a result, as a deadline approaches?
- Describe working on a piece of infrastructure shared by more than one team: what changed about how you
  wrote and shipped code, and how you avoided breaking someone else's work.
- Describe a time you had to tell your team or your manager about a negative or null result: how you framed
  it, and what happened next.
- Describe a technical disagreement with a teammate: what it was about, and how it was resolved.

### An open-ended problem from the team's area

- Your evaluation metric has improved over the last several training runs, but early users report that
  outputs feel worse — how do you investigate the gap?
- You have a small team and one month to make measurable progress on an open problem in the team's area —
  how do you spend the first week, and what would count as progress by the end of the month?
- Two versions of a model agree on the large majority of outputs but disagree sharply on one identifiable
  slice of inputs — how would you characterise the slice and decide which version is actually better on it?
- A metric the team uses to compare candidate models is too expensive to run more than a few times a week —
  how would you build a cheaper proxy for it, and how would you check that the proxy is not misleading you?
- After a change to the training data or the training recipe, a downstream metric regresses unexpectedly —
  walk through how you would narrow down the cause.
- A product team asks whether a model-powered feature is worth building: how would you estimate its value
  and its cost, and which metric would tell you after launch whether it worked?

### Fit, ramp-up and logistics

- What do you personally need, in your first few weeks on a new team, to become productive?
- Describe how you ramp up in an area that is new to you: what you actually do in the first days, and what
  tells you that you have moved from learning to contributing.
- Describe the kind of team or working environment you would not enjoy, even if the research area itself
  interests you.
- How do you decide how much to work out for yourself versus how much to ask, in your first weeks on a new
  team?
- The round sets aside time, near the end, for you to ask the team lead your own questions about the team,
  its current work, or how it operates.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Before the round, confirm with your recruiter whether you are meeting the team's lead, its hiring manager, or
both, which team it is, whether that is the team the role would place you on, and whether this conversation
falls before or after the technical rounds; if the team is not yet decided, ask when it will be. The team
lead moves between the five groups below in whatever order your background and the conversation
naturally suggest, not a fixed script, and the round can end once its ground is covered, sometimes before the
full 45 minutes. Experience in depth and research taste and direction take up the largest share of the time,
since they are what the team's own debrief turns on; working style takes a smaller share, and fit, ramp-up
and logistics close the conversation, with the last few minutes set aside for your own questions. When an
open-ended technical problem comes up, it is asked in place of some of this time rather than in addition to
it, cutting into the experience-in-depth or research-taste share; treat it as a genuine technical
conversation rather than a scored puzzle, and do not compress it to protect material you rehearsed elsewhere.
Bring, for each of the first three groups, an example with your own specific action and a result you can
state as a number or an observable change; where the loop also includes a separate research or project deep
dive, keep the project, the numbers and the decisions you describe consistent between the two rounds.

### Experience in depth

What it probes: whether you can name your own specific contribution to a result, separate from what a team
produced together; whether you have a real, repeatable process for designing and reading an experiment
rather than an intuition you cannot articulate; and whether the scale you claim is one you can back with a
concrete number and a specific part you owned, rather than a number that belonged to the team around you.

Skeleton for the project you are proudest of, about 90 seconds: name the project in one sentence — what it
was, who or what it served — then state the result as a number you can defend or an observable change,
before describing the work itself; only then describe your own actions with "I", naming the specific
decisions or code that were yours even where the project was a team effort ("the team built X; the piece I
owned was Y").

Skeleton for how you design and read an experiment, about 60–90 seconds: name the hypothesis in one
sentence, the baseline the comparison is measured against and why it is fair, the ablation that isolates the
effect from everything else that changed at the same time, and the seed count you want before trusting the
headline number; close with how you would tell a real effect from noise, for example a check across seeds or
a significance test, rather than reading a single run.

Skeleton for the largest-scale system or run, about 60 seconds: one concrete number for its scale, the
specific part you owned distinct from what the rest of the team operated, and the one thing that actually
broke or nearly broke at that scale — a scale question is really asking whether you have felt a system's
real limits, not only read about them.

Most common way to lose points: describing what "the team" or "we" did throughout a story, with no sentence
that isolates your own decisions, or treating the experiment-design question as an abstract methodology
answer with no named baseline, no ablation, and no seed count. Fix: rehearse each story with the result or
the concrete number in the first thirty seconds, and write out, ahead of time, the one sentence that names
the part only you did.

### Research taste and direction

What it probes: whether you have an actual, defensible opinion about where the team's area should go next,
backed by a specific question and a first experiment, rather than a list of directions that sound current;
and whether you can point to a real update — a specific result that changed a specific belief — rather than
a claim that you are generally open-minded.

Skeleton for the overrated/underrated question, about 60 seconds each: state the direction in one sentence,
then the reason, tied to a specific result or a specific gap you have noticed rather than to how active the
direction currently looks or how much attention it gets.

**The first-six-months answer** follows the same five-part question / why-now / success-signal / risks /
first-experiment structure as the six-month proposal template in
[Research Deep Dive](../research-deep-dive/README.md), compressed to about ninety seconds in total rather
than two minutes. When the question adds how exactly you would execute the plan, answer inside the same
structure rather than adding a sixth part: the first experiment, the data it needs, and the milestone that
would show it is working are the template's "first experiment" and "success signal" parts, made concrete
rather than left abstract.

For "can current theory fully explain the effect you would build on", separate two different kinds of claim
before answering: what a model or theory already in the literature predicts from first principles, and what
has only been observed empirically with no settled explanation. State plainly which side of that line your
effect sits on; if theory only partly explains it, name the specific missing piece and say whether your plan
depends on that piece being resolved or would work regardless — overclaiming a theoretical account you do not
have is a distinct way to lose credibility on this question.

Skeleton for "what do you think of current systems in the team's area", about 60–90 seconds: state one clear
limitation first, then the reasoning behind it, then what you would do about it — a critique with no proposed
first experiment reads as a complaint rather than taste. For a team working on multilingual models, a worked
set of limitations to draw on: data imbalance, where most training text comes from a handful of high-resource
languages and low-resource ones are left thin; tokenisers that split some scripts into many more tokens per
word than English, which raises serving cost and shrinks the effective context available in that language;
evaluation that leans on benchmarks translated from English and so misses cultural context a native benchmark
would catch; interference between languages sharing a fixed-capacity model, where adding more languages can
quietly cost the strongest ones some quality; and safety behaviour that is not uniform across languages,
weakest exactly where evaluation coverage is thinnest. Turn whichever one you name into a proposed first
experiment: for tokeniser inefficiency, for example, measure tokens-per-word on a fixed multilingual corpus
across a candidate vocabulary change, and check whether the shorter tokenisation it gives the affected scripts
holds quality on existing benchmarks — a concrete, checkable next step, not just a diagnosis.

For "how do you decide to stop a project", name the actual signal you look for — a result that had to appear
by a stated point and did not, or a cheaper alternative that closed most of the gap — rather than a general
feeling of diminishing returns. For "a result that changed your mind", name the specific prior belief, the
specific result, and what you did differently afterwards: a change of mind with no consequence attached
reads as a stated value rather than a real update.

Most common way to lose points: a six-month answer that lists trendy topics rather than a single question
with a first experiment behind it, which reads as enthusiasm rather than direction. Fix: before the round,
write the template above out in full for one real direction, so the first experiment is a specific memory
rather than something invented live.

### Working style on a research team

What it probes: whether you have a real, applied answer for where a research scientist's work ends and a
research engineer's begins, rather than a claim that the two roles are interchangeable; whether your
trade-off between research-code quality and speed actually shifts with the situation rather than defaulting
to one extreme; and, for the negative-results question specifically, whether you actually surfaced a result
you did not like rather than sat on it.

Skeleton for the research-scientist/research-engineer question, about 60 seconds: describe one concrete
instance rather than a general theory — who owned the research question, who owned the implementation that
ran it at scale, and the one piece of work that sat between the two and needed both people to get right.

Skeleton for code quality versus speed, about 60 seconds: name the rule you actually use — for example, an
exploratory script stays disposable until a result looks real, at which point the parts other people will
build on get tests and a stable interface while the rest stays as it was — and one concrete instance where
you drew that line.

Skeleton for shared infrastructure, about 60 seconds: name one specific habit that changed once your code
had users other than yourself — a compatibility check before a change, a warning before a breaking one, a
way to roll a change back — and one time that habit is what actually prevented an incident rather than
merely sounding responsible.

Skeleton for communicating a negative result, about 60–90 seconds: state the result plainly in the first
sentence, before any explanation of why it happened; name who you told and how soon after you had it; and
name the one thing that changed afterwards — a project redirected, an assumption dropped, a decision made
with the negative result as evidence — since a negative result that changed nothing suggests it was not
actually communicated as one.

Most common way to lose points: a negative-results story framed so gently that it reads as a success with
caveats, which is indistinguishable from having quietly minimised it at the time — the same instinct that
makes naming a real gap in the fit-and-ramp-up group below feel risky. Fix: before the round,
write the one sentence that states the negative result as plainly as you told it at the time, and name
specifically what it changed.

### An open-ended problem from the team's area

What it probes: whether you reach for a structured process under ambiguity rather than a first guess dressed
up as an answer, and whether you can turn a vague signal — "users say it got worse" — into a specific,
checkable question.

**A structure for the open-ended problem**, about eight to ten minutes once the interviewer starts pushing on
it:

1. **Clarify the metric.** State exactly what the metric measures, how it is computed, and what it could
   miss, before proposing any explanation. A metric and the thing people actually care about are rarely
   identical; naming the gap between them is the first real move, not a delay tactic.
2. **Form hypotheses.** List the specific, distinct explanations that would each produce the symptom you
   were given, rather than one favourite explanation elaborated at length. For a metric that improved while
   users report the opposite: the metric could reward something users do not value, the evaluation set could
   no longer represent what users actually do, the comparison could have shifted underneath the metric — a
   change to a default setting at the same time, for instance — or the reports themselves could come from a
   small, unrepresentative, or vocal minority.
3. **Find the cheapest discriminating check.** For each hypothesis, name the fastest, cheapest test that
   would move it up or down, and run the cheapest one first rather than the most thorough one. Reading
   twenty transcripts side by side to compare the metric's judgement with your own separates several
   hypotheses at once, before any new evaluation set or user study is built.
4. **State the decision.** Say what result would make you act, and what the action actually is — ship with
   monitoring, hold and patch the evaluation, or roll back — rather than stopping at "it depends" once you
   have gathered evidence.

Applied to the evaluation-metric example: state that the metric and user satisfaction are two different
measurements of the same system, before guessing which one is wrong; list the four hypotheses above rather
than jumping straight to "the evaluation set is stale"; propose reading a sample of recent outputs against
the metric's own judgement as the cheapest first check, since it needs no new data collection; and close with
what each outcome of that check would make you do next.

**A second skeleton, for a business-case question** such as the product-feature example, about five minutes
before the interviewer pushes on any one estimate: name the users and
the specific decision the feature would change for them, before any numbers; estimate its value as one
number — extra usage, retention, or revenue the changed decision would plausibly produce, stated with the
assumption behind it; estimate its cost as one number covering both what it takes to build and what it costs
to serve at the traffic you assume; name the main risk that could make the estimate wrong in either direction;
and close with the one launch metric you would watch, paired with a guardrail metric that would tell you to
roll back even if the launch metric looks good. Applied to the feature example: name who decides what today
without the feature, and what specifically changes once it exists; give a value estimate as a rough
multiplication — affected users times a plausible effect size — rather than a single unexplained number; give
a cost estimate that separates a one-off build cost from an ongoing serving cost per request; and name a
guardrail alongside the launch metric, for instance latency or a quality score, so an improvement on the
launch metric bought by a regression elsewhere does not read as a clean win.

Most common way to lose points: proposing a single explanation and a fix for it without naming the other
plausible explanations you did not check, which reads as a guess rather than a diagnosis. Fix: rehearse the
four-step structure on one worked example — the one above, or one from your own experience — until naming
several hypotheses before picking a check is the default move rather than an afterthought.

### Fit, ramp-up and logistics

What it probes: whether you have a genuine, specific answer for what makes you productive rather than a
generic "good documentation and a helpful team", and whether "ramping up quickly" is backed by an actual
plan and an honest account of the gap it closes, rather than an assurance.

Skeleton for what you need to be productive, about 45 seconds: name one or two things concretely — access to
a specific kind of compute or data, an introduction to whoever owns a system you will depend on, a week of
reading before you are expected to ship — rather than a general "good onboarding".

**A concrete ramp-up plan**, about 60–90 seconds, in place of an abstract answer: name what the first 30 days
produce — you can explain the team's main system to someone else, and have shipped one small, real change to
it; what the next 30 days add — you own a small piece of work end to end, with review rather than close
supervision; and what the final 30 days add — you are trusted with a piece of work whose scope you helped
define. Name one thing you would do in week one specifically to compress this timeline, such as pairing on a
real task before working solo, rather than reading documentation alone.

For the team-you-would-not-enjoy question, name a specific property of a working environment, not a research
area, and be honest rather than diplomatic, since a vague non-answer here just becomes a mismatch discovered
later — for example, a team where results are shared only once they are fully polished, if what you actually
want is fast feedback on rough work.

Most common way to lose points: claiming you would need no real ramp-up in an area that is genuinely new to
you, which reads as an inability to name your own gaps rather than as confidence; or answering the ramp-up
question with reassurance instead of a plan. Fix: name the actual gap plainly — the specific thing you do
not yet know — in the same breath as the plan for closing it, and write out the 30/60/90-day plan above with
your own concrete milestones before the round.

### Questions worth asking the team lead

- What is the team working on right now, and why is that the current priority over the alternatives?
- What would the role actually own in its first few months, concretely enough to compare against your own
  experience?
- Beyond the technical rounds already completed, what does the team's debrief after this round weigh most
  heavily?
- Where does the team currently fall short of where it wants to be, in the team lead's own account rather
  than a public description?
- What has changed about the team's direction or priorities over the last year?

Most common way to lose points: no questions at all, or a question a public source already answers, both of
which read as not having prepared for this specific conversation. Fix: write two or three of the above down
before the round, each naming something only this team lead can tell you.

### Prep outline

Copy this into `my/` and fill it in.

```text
Round confirmed: team lead or hiring manager (or both), which team, whether this comes before or after the
technical rounds, and whether the open-ended problem is likely:

Experience in depth:
  Project you are proudest of (result stated first, then your own action):
  How you design an experiment (hypothesis / baseline / ablation / seed count):
  How you tell a real effect from noise:
  Largest-scale system or run (the concrete number, and the part you owned):
  What actually broke, or nearly broke, at that scale:

Research taste and direction:
  First-six-months answer:
    Question:
    Why now:
    Success signal:
    Risks:
    First experiment (with the data and the milestone that shows it's working):
  Whether current theory fully explains the effect you'd build on, what's missing if not, and whether it
  matters for your plan:
  One clear limitation in the team's current systems, the reasoning, and a first experiment (multilingual
  example: data imbalance / tokeniser inefficiency / translated-benchmark evaluation / cross-lingual
  interference / uneven safety behaviour):
  One direction you think is overrated, and why:
  One direction you think is underrated, and why:
  How you decide to stop a project:
  A result that changed your mind, and what you did differently afterwards:

Working style on a research team:
  Research-scientist / research-engineer collaboration, one concrete instance:
  Your rule for research-code quality versus speed, and one instance you applied it:
  One habit that changed once your code had other users:
  A negative result you communicated: what you said, to whom, how soon, and what changed:

Open-ended problem (structure to reuse under pressure):
  Clarify the metric:
  Hypotheses (list more than one):
  Cheapest discriminating check:
  Decision rule (what result leads to what action):

Business-case question (second structure to reuse):
  Users and the decision the feature changes:
  Value estimate, with the assumption behind it:
  Cost estimate (build versus serving):
  Main risk to the estimate:
  Launch metric and its guardrail:

Fit, ramp-up and logistics:
  What you need to be productive, concretely:
  30/60/90-day ramp-up plan:
    30 days:
    60 days:
    90 days:
  One thing you'd do in week one to compress the timeline:
  The specific kind of team or environment you would not enjoy:

Three to five questions for the team lead:
  1.
  2.
  3.
```

</details>
