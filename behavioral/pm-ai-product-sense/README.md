# Product Manager Loop: AI Product Sense and the AI Deep Dive

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Behavioral · product case | ★★★☆☆ | — | PM | product-sense, ai-product-strategy, agent-metrics, launch-risk, user-insights, ux-for-ai | 45 min | Hiring-manager screen · Final interview |
<!-- meta:end -->

## Problem

The case rounds of the product-manager loop follow the initial recruiter screen: the
hiring-manager screen, one or two roughly 30-minute conversations with the manager for the team you would
join, and the final loop's four to six 45-minute interviews — product insight and strategy with a product
director, product design and user experience with a UX lead, craft and execution, an AI technical deep dive
with an engineer or a tech lead, and sometimes a further conversation with a director — closing with a
people-and-culture conversation on motivation and fit. The combined recommendation from these rounds decides
the offer. Almost every round opens with a case about a hypothetical AI product rather than a question about
your résumé, and the interviewer follows your first pass through the case with a harder version of the same
question, or a constraint that removes the easy answer. Every case is original and hypothetical: none
states a fact about any real product. The questions fall into five groups.

### Product sense and strategy

- You own the proactive features of a general-purpose AI assistant — the features that act before the user
  asks, such as drafting a reply before it is requested or reordering a household item before it runs out.
  Set the strategy for this area for the next year: what do you build first, and why?
- A rival chatbot is taking market share from the assistant you work on. Diagnose why, and lay out how the
  assistant wins share back.
- Founder case: you are starting a company built around an AI career coach. What do you build, who is the
  first customer, and why this wedge rather than another?
- A capable language model can run on a pair of smart glasses. What capability turns glasses into a distinct
  product category rather than a smaller phone screen, and what will slow people from adopting them?
- Engineering can ship only one of five candidate proactive features this quarter. Walk through how you
  decide which one.

### AI technical deep dive

- What would you measure to know that the assistant is correctly executing the actions a user actually asked
  for?
- The model still fails occasionally. For a high-stakes action such as sending an email or completing a
  purchase on the user's behalf, what do you launch alongside the feature to make failure safe?
- Once the assistant is live, how do you learn how it actually behaves in the real world — its use of
  external tools, its rate of hallucination, the times it responds inappropriately?
- Where do you draw the line between the assistant acting on its own and pausing to ask the user for
  confirmation, and how does that line move as the model's measured reliability improves over time?
- A launch candidate raises task success by five points and also doubles the rate of actions taken that the
  user did not want and could not undo. Do you ship it, and under what condition?

### Insights and execution

- A tutoring feature you shipped gets polarised feedback: under 10% of the users who try it love it, and most
  of the rest call it useless. Walk through what you do next.
- Walk through how you decide what to build next and what to cut from a roadmap, using a real prioritisation
  you have made.
- Describe a time you personally unblocked a roadblock that was stalling a launch: what was in the way, and
  what did you actually do?
- Describe a time you shipped something at unusual speed. What did you cut to move that fast, and what did
  you protect?
- A feature you pushed hard to ship underperformed against its target after launch. What did you do next?

### Design for AI

- The model your product depends on takes several seconds to respond. Design the product experience around
  that latency rather than around an instant response.
- Design a product that takes both text and images as input and can answer in either text or a generated
  image. Walk through the interaction model.
- A user's request is ambiguous — "clean up my photos" could mean delete duplicates, improve quality, or sort
  into albums. Design how the product asks for clarification without breaking a single conversational turn.
- The model is sometimes uncertain of its own output, for instance a generated summary it cannot verify is
  accurate. Design how the product communicates that uncertainty without undermining trust in every other,
  confident answer it gives.
- Walk through a 0-to-1 AI product you built: its success metric, the number before and after launch, and
  what you learned.

### Motivation

- Why this team, specifically, rather than another AI product team here or elsewhere?
- Which AI tools or agents have you built, configured, or used seriously to make yourself more productive,
  and what did that teach you about where today's models are still weak?
- Name one AI product you did not build that you use regularly and admire, and say what you would change
  about it if you owned it.
- Building AI products sometimes means shipping a feature before anyone is fully certain how people will use
  it. How comfortable are you with that, and what would tell you that you are not comfortable enough to ship?

## Reference solution

<details>
<summary>Show the reference solution</summary>

State the framework you intend to use in the first sentence of every case answer — for instance, "I'll cover
the user and their job, where the model creates new value, the risk, the sequencing, and the metric" — then
fill it in part by part. A follow-up lands on whichever part of the framework you passed over fastest, so a
visibly structured answer costs less follow-up time than one the interviewer has to reorganise for you before
pushing on it. Where it is not already obvious, confirm which
persona is asking before you answer: a product director presses hardest on sequencing and the market logic, a
UX lead on the concrete interaction, an engineer on the metric and the failure mode, so the same case answered
for two different interviewers earns two different follow-ups.

### Product sense and strategy

What it probes: whether you can turn a broad mandate — "own proactive features" — into a strategy grounded in
a specific user and a specific model capability, rather than a list of feature ideas in search of a
justification.

Skeleton, about eight minutes: state the structure in one sentence, then work through five parts in order —
the user and their job-to-be-done, stated as the task they are hiring the feature to do rather than a
demographic label; where the model's capability, not the product wrapper around it, creates value the user
could not get another way; the risk this specific capability introduces, named concretely rather than as a
generic "AI can go wrong"; the sequencing — what ships first, second and third, and the reason for that
order; and the metric that would tell you, without further argument, that the strategy is working.

Worked mini-answer, for the proactive-assistant case: users and jobs — commuters and parents managing a
household, whose job is "keep small, recurring tasks off my plate without having to remember to ask", not
"talk to an AI". Capability — a model that reads a user's calendar, email and past confirmations closely
enough to predict a narrow set of next actions, such as reordering a household item that is running low or
drafting a reply to a scheduling email, at a precision where acting first saves more time than it costs to
correct. Risk — an unwanted action costs more trust than a missed one, so the feature has to fail toward
silence, not toward a wrong action. Sequencing — launch with the single highest-confidence, most reversible
action first, a drafted-not-sent email reply, before anything that spends money or cannot be undone; widen
the action set only once measured precision on the current one clears a bar. Metric — track the intervention
rate, the share of proactive actions the user edits, cancels or complains about, falling as the action set
widens, rather than the raw count of actions taken, which rewards acting more without rewarding acting
correctly.

For the market-share and founder-case variants, apply the same five-part shape at speed rather than switching
frameworks. For market share, name the specific job the rival product wins on that the assistant does not,
before proposing any counter-feature. For the founder case, treat the AI career coach as the capability and
choose the wedge — the single job-to-be-done narrow enough to be excellent at from day one, such as
mock-interview feedback, over a broader promise no small team can support yet. For the smart-glasses case,
name the one capability a language model adds that a phone cannot, for instance understanding what the wearer
is currently looking at without them opening an app or typing a query, as the reason glasses are a distinct
category, and name the adoption blocker with the least to do with the model itself, such as battery life or
the social discomfort of a camera worn on someone's face, since the case is testing whether the strategy
separates the model's contribution from the hardware's constraints. For prioritising across five candidate
features with one slot, score each on the same two axes named above — reach and reversibility of not doing it
— and defend the trade-off explicitly: name the feature with the strongest individual case that still lost,
and the specific reason another feature beat it.

Most common way to lose points: opening with a list of feature ideas — "we could add proactive calendar
suggestions, proactive reminders, proactive reordering" — before naming the user or the job any of them
serves, so the strategy reads as a brainstorm rather than a strategy. Fix: say the user segment and its
job-to-be-done in the first sentence, before naming a single feature, and refer back to it explicitly when you
justify the sequencing.

### AI technical deep dive

What it probes: whether you reason about model failure in specific, measurable terms rather than in the
abstract, and whether your launch judgement for a risky action goes beyond a single confirmation dialog.

For what to measure, define each term precisely rather than only naming it: task success rate is the share of
requests where the assistant's final state matches what the user actually wanted, judged against the request
rather than against whether any action was taken at all; action precision is the share of actions the
assistant took that the user actually wanted, and recall is the share of wanted actions it actually took, and
the two only mean something read together, since a model that never acts scores perfect precision and zero
recall; harmful or irreversible action rate is tracked on its own because its acceptable value is close to
zero regardless of how the other numbers look; user intervention and undo rate — the share of the assistant's
actions that a person edits, cancels or reverses — is a faster-arriving, cheaper proxy for precision than
waiting on an explicit satisfaction signal; latency is tracked separately because a slow correct action and a
fast wrong one fail the user for different reasons and need different fixes.

For the high-stakes-action question, about five minutes: pair every safeguard with the specific failure it
closes rather than listing safeguards in the abstract. Worked mini-answer, for sending an email and completing
a purchase: for the email, require a confirmation step that shows the actual drafted content rather than a
generic "send this email?" prompt, so confirming is an informed choice and not a reflex click. For the
purchase, cap the first release to a user-set spending limit and an allow-list of merchants the user has
bought from before, so the blast radius of a wrong action stays small even when the confirmation is skimmed.
Stage the rollout by the action's own reversibility: draft-only actions ship broadly first; anything that
spends money or leaves the app ships to a small cohort behind a kill switch, with a person reviewing a random
sample of completed actions each day until the observed error rate holds steady. Undo matters as much as
confirmation: a sent email cannot be unsent, which is itself a reason to gate it behind a tighter allow-list
than a purchase that can still be refunded.

For learning how the assistant behaves in the real world, name three separate instruments rather than one: a
held-out sample of live sessions read by a human rater against a fixed rubric for tool-use correctness,
hallucination and tone; an automated check that flags a tool call whose result the model's next message
contradicts, which catches one common failure mode — acting on a tool result and then describing a different
one — without a human in the loop for every session; and a standing channel for users to flag a bad action
directly from the action itself, since the intervention-rate metric above catches only what a user bothers to
undo, not what they silently distrust and stop using.

For where to draw the autonomy line, set it by the action's stakes and reversibility, not by the model's
average confidence: a low-stakes, reversible action can run autonomously even at moderate confidence, while a
high-stakes, irreversible action stays behind confirmation even at high confidence; the line moves only when
the measured error rate for that specific action type, not the model's overall accuracy, clears a bar over a
sustained window. For the ship-versus-hold trade-off between a five-point gain in task success and a doubled
rate of unwanted, unrecoverable actions, put both numbers on the same scale before deciding: a doubled
irreversible-action rate against a near-zero acceptable baseline dominates a five-point gain in an averaged
success metric, so the answer is to hold the full launch and ship the success-rate improvement only once it
is decoupled from the regression, for instance behind the same tighter allow-list already used for other
irreversible actions rather than to the full population.

Most common way to lose points: treating "add a confirmation dialog" as a complete answer, with no mention of
what happens once a user learns to click through it, or of limits and staged rollout as a backstop for when
they do. Fix: name at least one safeguard the user cannot skip by clicking through — a hard spending limit, an
allow-list, a rollout gate — alongside the confirmation step, and pair each one with the specific failure it
is meant to close.

### Insights and execution

What it probes: whether you diagnose a product problem before proposing a fix, and whether your account of a
past decision, roadblock or fast launch is specific enough to check.

For the polarised-feedback case, about six minutes across four steps: segment the under-10% who love the
feature from the rest by what they actually asked for in their first session, not by demographics; compare
each segment's expectation against what the model can actually do; read session traces from the unhappy
segment specifically, since a fast abandon points at a positioning problem and a long fight to get the tutor
to do something else points at a capability gap; then decide among narrowing the audience, changing the
promise, or improving the model, in that order of cost.

Worked mini-answer: the likely split is students who wanted a worked, step-by-step explanation of one problem
against students who wanted the tutor to produce a finished answer outright. If session traces show the
second group abandoning after one turn rather than fighting for several, the model is not broken for them, it
is answering a question they did not ask; the fix is to narrow the entry screen's promise to "explains a
problem step by step" rather than a generic "AI tutor", which turns confused churn into an accurate no-thanks
from the wrong-fit segment and protects satisfaction among the users the feature is actually for. Only invest
in extending the model to the second job once the narrowed product is validated on the first.

For what to build and what to cut, name the axis you actually prioritise on — reach, reversibility of not
doing it, or dependency on something else already committed — rather than "impact versus effort" with no
further content, and give one real trade-off you made on that axis: the option you cut, and the specific
reason it lost, not only the option you kept. For the roadblock and speed stories, answer in Situation, Task,
Action, Result (STAR) shape, about three minutes each: 30 seconds on the situation and the task, 90 seconds on
the specific action you personally took, 30 seconds on the measurable result. For a launch that missed its
target, the same shape applies, with the result naming the actual gap between the target and what shipped,
and the action naming what changed afterward, not only what you felt about it.

Most common way to lose points: answering the polarised-feedback case with a single-track "retrain the model"
plan before checking whether the two groups wanted the same product in the first place. Fix: name the
segmentation step out loud, before proposing any change to the product or the model.

### Design for AI

What it probes: whether your design choices account for what a language model actually does badly — respond
slowly, misjudge intent, misjudge its own confidence — rather than treating the model as an instant,
always-right backend.

For the latency case, about five minutes: make the wait legible before making it shorter, by streaming output
as it generates and, where a request triggers multiple steps such as searching then reasoning then answering,
showing which step is running rather than one undifferentiated spinner, since a labelled four-second wait
reads as faster than an unlabelled two-second one. Worked mini-answer: make partial output useful as well as
visible — if the model drafts a long document, let the user start reading and editing the first section while
later ones are still generating, rather than gating all output behind full completion; and set a fast default
by routing short, common requests to a smaller or cached path with a tighter latency budget, so the product's
typical wait is set by the common case rather than the worst one, reserving the full model for requests that
actually need it.

For multimodal input and output, choose the default output modality by the task's cognitive load rather than
by what the model can technically produce — a comparison of five products is read faster as a table than
heard as a paragraph — and let the user switch modality on demand rather than committing to one. For the
ambiguous-request case, detect ambiguity as the condition where two plausible interpretations lead to two
different actions, not merely two different phrasings, and resolve it with one targeted clarifying question
asked inline in the same turn rather than a separate form; where you must act without asking, default to the
more reversible interpretation. For the model's-own-uncertainty case, match the confidence signal to the
actual error mode — flag the specific claim in a summary the model cannot verify against its source, rather
than hedging the entire response uniformly, since uniform hedging trains users to stop reading any confidence
signal at all.

For the 0-to-1 walkthrough, about six minutes: the user and the problem that existed before the product; the
specific model capability that made solving it newly possible, not merely convenient; the metric chosen and
the number before and after launch; and the one thing you learned that changed how you build the next
feature, stated as a specific change in practice rather than a general lesson.

Most common way to lose points: describing a latency fix purely as a backend change — "we would cache more
aggressively" — with no change to what the user actually sees while waiting. Fix: name the specific
interaction or UI change the user experiences, in addition to any infrastructure change behind it.

### Motivation

What it probes: whether your reason for wanting this specific team is tied to something you actually have an
opinion on, and whether you have a genuine, examined relationship with the AI tools you would be building.

Two-minute skeleton for "why this team": name the one specific thing about the team's stated focus — a
product surface, a capability, a stated problem — you actually have an opinion on; connect it to a concrete
piece of your own history that shows the interest predates this loop; state what you would actually do here
because of it.

For the tools question, name a specific tool or agent you built or configured yourself, not only one you used
out of the box, and the specific limitation you ran into that told you something about where current models
are weak, rather than a generic "AI makes me more productive". For the admired-product question, name the one
thing you would change and the trade-off that change would cost, not only what you like about it. For the
shipping-under-uncertainty question, give a genuine, specific line — a class of harm you would want ruled out
first, or a reversibility bar below which you would not ship — rather than a general "I'm comfortable with
ambiguity".

Most common way to lose points: a "why this team" answer built entirely from the team's own public
description of itself, with nothing that shows it is about you specifically. Fix: write down, before the
round, the one piece of the team's actual work you have an opinion on and the one piece of your own history
that backs it up.

### Questions worth asking back

- How does the team decide a proactive or high-stakes feature is safe enough to launch, and who has the
  authority to say no?
- What does this team's PM own end to end, versus share with the tech lead or the director, and where does
  that line actually sit day to day?
- What is a recent launch on this team that did not go the way it was expected to, and what changed
  afterward because of it?
- How is success defined for the team's current top priority, in a number more specific than engagement?
- What does the split between 0-to-1 exploration and hardening an existing feature look like on this team
  right now?

### Prep outline

Copy this into `my/` and fill it in.

```text
Product sense and strategy:
  User and job-to-be-done:
  Where the model's capability creates new value:
  Risk this capability introduces:
  Sequencing (what ships first, second, third, and why):
  Metric that shows it is working:
  Market-share / founder-case / smart-glasses variants, one line each:

AI technical deep dive:
  Task success rate, action precision and recall, in your own words:
  Harmful/irreversible action rate, your acceptable bar:
  User intervention and undo rate:
  Safeguards for one high-stakes action (confirmation / limits / allow-list / staged rollout / human review):
  Where you'd draw the autonomy-vs-confirmation line, and what moves it:
  How you'd learn real-world behaviour (tool use, hallucination, inappropriate responses):

Insights and execution:
  A polarised-feedback case, walked through (segment / compare / traces / decide):
  A real build-vs-cut trade-off, and the axis you used:
  Roadblock you unblocked (Situation / Task / Action / Result):
  Shipped at speed (Situation / Task / Action / Result):
  A launch that missed its target, and what changed afterward:

Design for AI:
  Latency: how you'd make the wait legible and useful:
  Multimodal input/output: your default-modality rule:
  Ambiguous request: your inline clarifying-question design:
  Model uncertainty: how you'd surface it without over-hedging:
  0-to-1 product: user / capability / metric before-after / what you learned:

Motivation:
  Why this team (the one thing you have an opinion on + your own history):
  An AI tool or agent you built or configured, and its specific limitation:
  An AI product you admire, and the one thing you would change:
  Your bar for shipping under uncertainty:

Five questions to ask back:
  1.
  2.
  3.
  4.
  5.
```

</details>
