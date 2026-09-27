# ML Design: A Data Flywheel for a Robot Manipulation Policy

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| ML system design | ★★☆☆☆ | Hard | RE · RS · MLE | robot-learning, data-collection, dataset-curation, vision-language-action, evaluation-statistics, deployment-safety | 45–60 min | Final interview |
<!-- meta:end -->

## Problem

A *vision-language-action* (VLA) policy is a model that takes one control-loop time step's camera images, the
robot's current joint state, and a natural-language task instruction, and outputs the robot arm's next action.
Design the system around it: a fleet of robot arms uses the policy to perform household manipulation tasks —
picking up an object, opening a container, placing an item — each given as a short instruction such as "put
the mug in the sink". The system to design collects data from the fleet, curates it, trains new versions of
the policy, evaluates a candidate version before it is trusted, and deploys it back to the fleet.

An *episode* is one continuous attempt at a single task, from a fixed instruction and an initial scene layout
to an end condition — the task is completed, a time limit is reached, or an operator stops it — after which
the robot resets for the next attempt; it is the unit of data collection, lasting about 60 s. An episode is
either a *teleoperated demonstration*, in which a human operator supplies every action in real time through a
teleoperation interface (the recorded actions are the operator's own commands), or an *autonomous rollout*, in
which the actions come from a policy running closed-loop with no human directly in control — though a human
safety operator continues to watch and may stop or override it at any point. A *success detector* is an
automated classifier — here, a vision-language model prompted with an episode's final frames and its task
instruction — that predicts whether the task was completed; it is what makes it possible to label the volume
of autonomous rollouts without a human reviewing every one. An *intervention* is the safety operator taking
manual control away from an autonomous rollout mid-episode, because the robot is about to violate a safety
limit or is visibly stuck or failing; the *intervention segment* is the part of the episode from the moment of
takeover to the moment control is handed back, or the episode's end.

The policy does not have to produce one action at a time: a single forward pass may output an *action chunk*,
a short sequence of consecutive future actions, which the robot's low-level control loop then consumes one at
a time while the next chunk is computed.

Scale this design for:

- 200 robots, each operating 8 hours a day.
- Each robot has 3 cameras at 640×480 resolution and 15 frames/s, stored as JPEG frames of about 40 KB each.
- Each robot also logs joint state and commanded action at 50 Hz; together these low-dimensional streams total
  about 1 KB/s.
- 30% of robot time runs teleoperated demonstrations, 70% runs autonomous rollouts.
- An episode lasts about 60 s.
- The evaluation suite is a fixed set of 50 held-out tasks.
- The policy runs on the robot's control loop at 10 Hz, one action consumed every 100 ms, and computing one
  action chunk — from the triggering camera frame to the chunk being ready, regardless of how many actions it
  contains — must take at most 100 ms.
- A human operator can stop any robot at any time, independent of anything the policy or the rest of the
  system is doing.

In scope: the data flywheel end to end — ingestion, labelling, curation, the training data mixture, training,
evaluation, and staged deployment back to the fleet — for a single, fixed robot embodiment (one robot model,
identical across the fleet); the statistics behind deciding whether a candidate policy actually improved; turning
interventions into training data; and
the safety and latency constraints on serving the policy. Out of scope: the policy's model architecture and
training objective beyond the input/output contract already stated; the low-level motor controller that
executes one action once issued; and the teleoperation hardware itself.

Produce:

1. A requirements and scale estimate: the camera data generated per robot-day and per fleet-day, the
   low-dimensional data volume (computed, not merely asserted small), episodes produced per day, and a year of
   storage.
2. The episode data model and its storage layout: what is stored, where, and what is indexed to support a
   curation job's queries.
3. The pipeline the data and the policy move through — ingestion, labelling, curation, the training data
   mixture, training, evaluation and canary deployment — as a diagram, with a walk-through of one episode's
   path along it.
4. Deep dives into: (a) evaluation statistics — a confidence interval for a measured success rate, how many
   evaluation trials are needed to tell a 60% success rate from a 70% one, controlling that comparison for
   scene resets and start order, and simulation versus real evaluation; (b) closing the loop — turning an
   intervention into training data and deciding what to collect next; (c) safety and rollout — action limits,
   fallback behaviour, and staged deployment across the fleet; (d) latency — on-robot versus server inference,
   and what action chunking buys.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points worth confirming before designing: what counts as success — assumed here to be binary per episode,
judged against a short written rubric fixed per task, with no partial credit for an almost-complete attempt —
and whether the fleet is homogeneous. This design assumes a single robot model across all 200 units, sharing
one arm, one camera rig and one joint configuration, so one policy — one set of weights — serves the whole
fleet with no per-robot branching; a fleet of different robot types is a follow-up.

### Requirements and scale

**Camera data.** Each robot's three cameras together produce

$$3 \times 15 \times 40\text{ KB} = 1{,}800\text{ KB/s} = 1.8\text{ MB/s.}$$

Over an 8-hour operating day ($8\times3{,}600=28{,}800$ s):

$$1.8\text{ MB/s} \times 28{,}800\text{ s} = 51{,}840\text{ MB} \approx 51.84\text{ GB/robot-day,}$$

and across the fleet, $51.84\times200=10{,}368$ GB, about **10.4 TB/fleet-day**.

**Low-dimensional data.** Joint state and commanded action together log at about 1 KB/s (given), so one
robot-day produces $1\text{ KB/s}\times28{,}800\text{ s}=28{,}800$ KB $\approx28.8$ MB, and the fleet
$28.8\times200=5{,}760$ MB $=5.76$ GB/day — exactly $1{,}800/1=1{,}800$ times smaller than the camera stream,
confirming "negligible" by computing it rather than assuming it.

**Episodes per day.** At about 60 s per episode and 28,800 s of operating time, one robot completes
$28{,}800/60=480$ episodes a day; the fleet completes $480\times200=96{,}000$ episodes a day, split 30%/70%
into $28{,}800$ teleoperated demonstrations and $67{,}200$ autonomous rollouts.

**A year of storage.** Fleet-day total is $10{,}368+5.76=10{,}373.76$ GB; over 365 days,

$$10{,}373.76\text{ GB}\times365=3{,}786{,}422.4\text{ GB}\approx3.79\text{ PB/year,}$$

camera frames accounting for essentially all of it. That figure is raw sensor data kept forever, which is not
actually the right default: curation (below) keeps every episode's metadata and the outcome of every labelling
decision permanently, but the case for keeping every raw frame past the window in which it might still be
re-labelled or audited is weaker — a rolling hot tier of, say, 90 days of raw video, with older raw frames
moved to cheaper cold storage or dropped once the episode they belong to has been through curation, is a
reasonable default that this design does not commit to a specific number for.

### Data model and storage

**Task** — `task_id`, `instruction_template` (the instruction the policy is given, e.g. "put the mug in the
sink"), `is_eval_task` (true for the 50 tasks held out as the evaluation suite), `success_rubric` (the written
criteria both the success detector's prompt and a human spot-checker are shown).

**Episode** — `episode_id`, `robot_id`, `task_id`, `episode_type` (`teleop | autonomous`), `policy_version`
(set for `autonomous`, null for `teleop`), `scene_config_id` (which randomised initial object layout this
attempt used — needed to control for scene resets in evaluation, deep dive (a)), `started_at`, `duration_s`,
`outcome` (`success | failure | unknown`), `outcome_source` (`detector | human`), `detector_confidence`,
`had_intervention`, `frame_ref`, `state_ref`. The last two are pointers into blob storage below, never the
frames or the joint-state trace themselves. Indexed on `(task_id, episode_type, outcome, policy_version,
had_intervention, started_at)` — exactly the combination a curation job filters on: every failed autonomous
episode of task X under policy version Y, every episode with an intervention in the last week, and so on,
answered directly by a row-oriented metadata store without touching blob storage at all.

**Intervention** — one row per intervention inside an episode: `episode_id`, `operator_id`, `start_t`, `end_t`
(offsets within the episode), `reason_code` (`near_limit | stuck | task_error`), `used_for_training` (set once
curation, deep dive (b), decides whether the correction belongs in the training set).

Camera frames are written to object storage keyed `{episode_id}/{camera_id}/{frame_index}.jpg`; the
joint/action trace is one file per episode, `{episode_id}/state.bin`, at its native 50 Hz. A robot buffers an
episode locally while it runs and, once the episode ends, uploads the blobs and only then writes the `Episode`
row — blobs first, metadata row second — so a crash between the two leaves at worst an orphaned blob nothing
points to yet, never a metadata row pointing at data that was never written.

The ingestion and curation paths are each a narrow interface over this model: a robot calls `POST
/v1/episodes` once an episode ends and its blobs are durably written, `{robot_id, task_id, episode_type,
scene_config_id, frame_ref, state_ref, ...}` → `{episode_id}`; a curation job calls `POST
/v1/episodes:query` with a filter over the indexed fields above and pages through the matching `episode_id`s;
and a labelling worker calls `PATCH /v1/episodes/{episode_id}` to attach `outcome`, `outcome_source` and any
`Intervention` rows once the detector — and, where sampled, a human — has scored it.

### Pipeline

```text
    +-------------------------------------------+
    |    Fleet: 200 robots running policy vN    |
    +-------------------------------------------+
                          |
                          v
      episodes: 30% teleoperated, 70% autonomous
                          |
                          v
    +-------------------------------------------+
    |                 Ingestion                 |
    |     write blobs, then one metadata row    |
    +-------------------------------------------+
                          |
                          v
    +-------------------------------------------+
    |                 Labelling                 |
    |  VLM success detector + human spot check  |
    |       + intervention-segment tagging      |
    +-------------------------------------------+
                          |
                          v
    +-------------------------------------------+
    |                  Curation                 |
    |   filter -> de-duplicate -> task-balance  |
    +-------------------------------------------+
                          |
                          v
          += web-scale vision-language data
                          |
                          v
    +-------------------------------------------+
    |           Training data mixture           |
    +-------------------------------------------+
                          |
                          v
    +-------------------------------------------+
    |    Training  -->  candidate policy vN+1   |
    +-------------------------------------------+
                          |
                          v
    +-------------------------------------------+
    |                 Evaluation                |
    | simulation sweep, then real-robot battery |
    +-------------------------------------------+
                          |
                          v
    +-------------------------------------------+
    |     Canary deployment: staged rollout     |
    +-------------------------------------------+
                          |
                          v
       vN+1 becomes the fleet's policy; its own
       episodes flow back into Ingestion, above
```

A teleoperated demonstration takes the short path: ingested, scored by the success detector mostly for
consistency (a human operator's own demonstration is taken as successful by construction unless the
teleoperation interface itself recorded an abort, which sets `outcome = failure` directly), and passed to
curation.

An autonomous rollout that draws an intervention takes the fuller one. It is ingested like any other episode,
and the success detector scores its final frames against the task's instruction; because it contains an
intervention, curation treats its two halves differently regardless of what the detector says. The segment
before the takeover — in which the policy was about to do the wrong thing — is excluded from the positive
training set, since it demonstrates the mistake rather than a corrected trajectory, but is kept and tagged for
the failure analysis in deep dive (b). The segment from the takeover onward is the operator's own correction,
and — once a human spot check confirms its `reason_code` and sets `used_for_training` — is added to the
training set exactly as a short teleoperated demonstration would be, because that is what it is: a human
supplying the right action from the exact state the current policy actually reached. Both kept segments then
go through the rest of curation identically to any other episode.

The success detector, not a human, scores the large majority of episodes — necessary at 67,200 autonomous
rollouts a day — with a fixed *percentage* of them (weighted toward low detector confidence and toward
episodes with an intervention, where getting the label right matters most) sampled for a human spot check,
rather than a fixed count, so the human-review budget scales with the fleet instead of falling further behind
it.

Curation acts on the labelled stream in three passes. Filtering drops autonomous rollouts the detector marks
failed from the positive training set — like a pre-intervention segment, a failed rollout is retained for
analysis rather than deleted, just excluded from what supervises the policy to succeed — and drops teleoperated
demonstrations the interface flagged as aborted. De-duplication clusters episodes by an embedding of their
trajectory and camera view within each `(task_id, scene_config_id)` bucket and caps how many near-identical
repeats of the same approach are kept; this has to be approximate similarity, not exact-match hashing, since
real sensor noise makes two literally identical episodes essentially impossible. Task balancing then resamples
across the 50 tasks — and, layered with deep dive (b)'s priority weighting, across how well the current policy
already does each one — so a training epoch does not simply mirror how often each task happened to be
attempted that week.

Training draws each batch from two sources: the curated robot episodes, and a fixed proportion of web-scale
image-text data carrying no robot action at all. Each robot example supervises the policy to reproduce the
recorded action — the operator's, for a teleoperated example, or the operator's correction, for a kept
intervention — given that time step's images, joint state and instruction; a web-scale example supervises only
the shared vision-language components. A mixture skewed too far toward robot data overfits to this fleet's
exact 50 tasks, cameras and lighting; skewed too far toward web data, the policy keeps broad visual-language
competence but under-weights the one signal that actually teaches it to act.

Evaluation runs a newly trained candidate against the 50-task suite before it is trusted with the fleet at
all — deep dive (a) sizes what that takes. Canary deployment then stages a version that passes evaluation
across the fleet in increasing slices, gated on live metrics rather than the offline evaluation score
alone — deep dive (c).

### Deep dives

**(a) Evaluation statistics.** A single evaluation run of $n$ trials with $k$ successes gives the point
estimate $\hat p=k/n$, but the point estimate alone hides how little $n$ real-robot trials actually pin down.
The familiar Wald interval, $\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$, comes from substituting $\hat p$ for the
unknown $p$ inside the variance of a normal approximation to $\hat p$; the Wilson score interval instead
inverts the test statistic $(\hat p-p)/\sqrt{p(1-p)/n}$ directly, without that substitution, by solving
$(\hat p-p)^2=z^2\,p(1-p)/n$ for $p$ — a quadratic in $p$ whose roots are

$$p_{\pm}=\frac{\hat p+\dfrac{z^2}{2n}\pm z\sqrt{\dfrac{\hat p(1-\hat p)}{n}+\dfrac{z^2}{4n^2}}}{1+\dfrac{z^2}{n}}.$$

For $n=60$ trials with $k=42$ successes ($\hat p=0.700$) at 95% confidence ($z\approx1.960$), this gives
$[0.575,\ 0.801]$ (checked below), centred at $0.688$ — shrunk toward $0.5$ relative to $\hat p$ itself, which
is exactly what keeps its coverage close to nominal as $\hat p$ moves toward 0 or 1 or $n$ gets small, both
routine for real-robot evaluation where every trial costs a robot's time. The Wald interval for the same data,
$[0.584,\ 0.816]$, sits shifted toward higher values and, here, is also the wider of the two.

Comparing two policy versions head to head needs a second question answered: how many trials per version are
enough to tell a true difference from noise? Fix the smallest difference worth reliably detecting at 60%
versus 70% — a smaller true difference would need more trials still; this is this design's stated target
resolution, not a universal constant. Under $H_0: p_1=p_2$, the pooled estimate $\bar p=(p_1+p_2)/2$ gives the
test statistic its variance and the test rejects at level $\alpha$ when $|z|>z_{\alpha/2}$; under $H_1$, with
the true rates $p_1,p_2$ apart, the same statistic has a different variance, built from $p_1,p_2$ individually.
Setting $P(\text{reject}\mid H_1)=\text{power}$ and solving for the common per-arm size $n$ gives the standard
two-proportion sample-size formula:

$$n=\frac{\left(z_{\alpha/2}\sqrt{2\bar p(1-\bar p)}+z_{\text{power}}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\right)^2}{(p_1-p_2)^2}.$$

At $\alpha=0.05$ two-sided ($z_{\alpha/2}\approx1.960$) and 80% power ($z_{\text{power}}\approx0.842$),
$p_1=0.60$, $p_2=0.70$: $n\approx355.9$, so **356 trials per arm** — 712 real episodes at about 60 s each, just
under 12 hours of continuous robot time, for one head-to-head comparison on one task. A Monte Carlo simulation
at exactly this $n$ (checked below) recovers a rejection rate of 0.7999 under $H_1$, matching the formula's 80%
target within the simulation's own margin.

That comparison is only valid if nothing besides the policy version differs between the trials being compared.
Run it as blocked, paired trials: for each of many independent scene resets, evaluate both versions back to
back before resetting again, alternating which version goes first. This removes scene-to-scene and
time-of-day nuisance variance — lighting drift, an object's accumulated wear, a camera that has drifted a few
millimetres — from the comparison rather than letting it inflate an apparent difference or mask a real one, and
alternating which version starts each pair cancels any first-mover advantage (a fresher object, a warmer
motor) from ever being credited to whichever version happened to go first. A paired design needs no more
trials than the unpaired figure above for the same power, and generally needs fewer, since it removes a source
of variance the unpaired test still has to pay for — by how much depends on how strongly scene identity
actually affects the outcome, not quantified further here.

Running the full paired comparison above on every one of the suite's 50 tasks would need
$50\times356\times2=35{,}600$ real trials, about 593 hours — close to 25 days of continuous robot time — even
before accounting for resets between trials: clearly not a routine cost to pay for every candidate. Simulation
is what actually absorbs that cost: scenes reset instantly, environments run in parallel, and a candidate can
be screened against all 50 tasks cheaply enough to filter out clear regressions before any real robot is
involved — provided the simulator's own gap from real hardware (rendering fidelity, contact and friction
dynamics, sensor noise) is itself periodically checked against a batch of real trials rather than trusted
blindly. The 356-per-arm real-trial budget is then spent on the small number of tasks a simulated screen flags
as close calls or as candidates for fleet-wide promotion, not on all 50 as a matter of routine.

**(b) Closing the loop.** Ordinary behaviour cloning trains on states an expert chose to visit — the
teleoperated demonstrations — but the deployed policy will visit its own states, including ones no
demonstration ever covered; a small early error can walk it into a state the training data never showed it how
to leave, and the error compounds instead of correcting. An intervention's corrective segment is, by
construction, a label at exactly the state the current policy actually reached and was about to mishandle —
the idea behind DAgger (Dataset Aggregation; Ross, Gordon & Bagnell, 2011): aggregate the expert's corrections
at the learner's own rollout states into the training set, rather than training only on the expert's original
demonstration distribution, so each round narrows the gap between the states the policy visits and the states
it has actually been shown how to handle. In this design that aggregation is not a separate offline step; it
is exactly what curation's `used_for_training` decision on an intervention segment (Pipeline, above) performs,
continuously, at fleet scale.

Not every task needs equal attention. A task the policy already handles well gains little from another
demonstration; one it fails often is exactly where the next round's marginal episode helps most. Weight how
often a unit of collection — a teleoperated demonstration, or which task an autonomous rollout is sent to
attempt — goes to task $i$ by $w_i\propto(1-\hat p_i)+\varepsilon$, its estimated failure rate plus a small
floor $\varepsilon$. The floor matters as much as the weighting: without it, a task the policy has driven to
near-100% success would stop being sampled altogether, and a later regression on exactly that task — caused by
something else entirely, such as a mixture change that quietly hurt an unrelated skill — would go undetected
until it turned up in evaluation instead of being caught by routine collection. A small simulation of this
mechanism, checked below, allocates more of a fixed collection budget to a task that starts out harder and
less to one that starts out easier, compared with allocating uniformly across tasks, while never letting any
task's allocation reach zero; run for enough rounds, it also leaves the worst-performing task in better shape
than uniform allocation does, for the same total number of collected episodes.

Directing more attempts at a task precisely because it is failing often does mean that task is also the one
most likely to need an intervention soon — not a contradiction, since that is how it gets fixed, but a reason
the weighting in this deep dive and the safety layer in the next one are a single design, not two independent
ones: prioritised collection decides where the fleet spends its autonomous attempts, and action limits and
fallbacks (deep dive c) bound how badly any single one of those attempts can go before a human or the robot's
own limits step in.

**(c) Safety and rollout.** Action limits sit as a hard, policy-independent layer between the policy's raw
output and the robot's motor controller: every commanded joint velocity, torque and position is clamped to a
fixed physical bound, the end-effector is held inside a fixed Cartesian workspace, and the change in a command
from one tick to the next is rate-limited. None of it is learned or something the policy can override, so a
policy bug or an out-of-distribution input degrades to clipped, jerky motion rather than an unbounded one.

Fallbacks cover the policy failing to deliver, not only delivering something wrong: a chunk that misses its
100 ms deadline (deep dive d), one that fails a basic validity check (a `NaN`, or a value outside the clamp by
far more than a legitimate edge case would produce), and — the premise's own requirement, not a hypothetical —
an operator stopping the robot outright. All three fall back the same way, freezing the last commanded
position rather than continuing to execute a stale or degenerate action; the operator's stop signal in
particular reaches the motor controller on a channel that never passes through the policy-serving path at all,
so a slow or crashed inference server, on-robot or remote, can never delay it.

A newly evaluated version is rolled out to the fleet in increasing slices rather than all at once: for example
1% (2 robots), 10% (20), 50% (100), then 100% (200), each stage running long enough to accumulate a meaningful
number of trials on its tasks — deep dive (a)'s trial-count reasoning, now applied to live fleet data instead
of a dedicated evaluation batch — before the next stage begins, and gated on the intervention rate and any
safety-limit trigger rate staying within the previous version's baseline, not only on the offline evaluation
score the version already passed. A regression the evaluation suite's 50 fixed scenes did not happen to cover
can still show up against the fleet's wider variety of real rooms and lighting, and a staged rollout catches it
against 2 robots' worth of exposure instead of 200. A stage that regresses rolls back on that slice alone,
rather than pausing the whole fleet while it is investigated.

**(d) Latency: on-robot versus server inference, and action chunking.** The control loop consumes one action
every 100 ms and must never run dry. Computing one action chunk — whatever its length $H$ — is budgeted at the
full 100 ms regardless of $H$, per the problem statement; what changes with $H$ is not that per-call ceiling
but how often a call is needed at all, since a chunk of $H$ actions gives the control loop $H\times100$ ms of
buffered runway before the next one is required — cutting the call rate from 10 Hz to $10/H$ Hz, 1.25 Hz at
$H=8$.

On-robot inference spends that budget on compute alone: an embedded accelerator's forward pass, slower than a
data-centre GPU's but paying no network cost, is the entire latency — say 70 ms, leaving 30 ms of margin.
Server inference spends the same budget on network plus compute instead: a request leaves the robot, is
serialised, runs on a faster remote GPU, is serialised back, and returns — say $2\times4$ ms of local-network
transit, $2\times5$ ms of serialisation, and 30 ms of server-side compute, 48 ms in total, leaving 52 ms of
margin against the same ceiling. Both fit comfortably on an average call; the difference shows up on a slow
one. On-robot compute time is fairly stable, since it depends only on the model and the accelerator and
nothing shared; a server round trip inherits whatever the network happens to be doing, and occasionally
exceeds 100 ms even when its median sits well under it.

Chunking is what turns that occasional overrun from a stall into nothing. With $H=1$, the buffer holds nothing
once the current action is consumed, so a call that overruns the deadline stalls the control loop until it
returns — exactly the freeze fallback from deep dive (c). With $H=8$, the buffer already holds $7\times100=700$
ms of runway once the newest chunk arrives, so a single call that spikes to, say, 180 ms — still a miss by the
letter of the 100 ms budget — is fully absorbed: the control loop is still consuming the previous chunk when
the late one arrives, and nothing downstream ever sees the overrun. Chunking never raises the 100 ms ceiling
any individual call is held to; it lowers how often a call has to be made and, as a direct consequence, how
much of a one-off overrun the system can absorb before it becomes visible at all.

### Follow-ups

- **Cross-embodiment data.** A fleet of more than one robot type breaks the "one set of weights, no branching"
  assumption above, since different arms have different degrees of freedom and different action spaces; the
  usual fix is a shared vision-language backbone with either an embodiment-specific action head or a
  canonical, embodiment-agnostic action representation (end-effector delta pose and gripper state, rather than
  raw joint commands) that demonstrations from different arms can all be expressed in. The data model would
  need an explicit `embodiment_id`, and task balancing (Pipeline, above) would need to balance across
  embodiments as well as tasks, not treat the fleet as one pool.
- **Learning from videos of humans.** A video of a person performing a task carries no robot action labels, so
  it cannot directly supervise the action-prediction loss the way a teleoperated demonstration does; it either
  pretrains the shared vision-language components (feeding the web-scale side of the training mixture, not the
  labelled-action side) or needs its own inverse-dynamics or hand-pose-to-action retargeting model to produce a
  pseudo-label, which then needs its own quality filtering separate from the robot-episode curation pipeline
  above.
- **Data privacy in homes.** Moving from a controlled fleet environment into people's homes puts bystanders and
  private spaces in the camera stream; a home-collected episode needs on-device redaction of faces before any
  frame leaves the robot, a stricter retention limit than the fleet's own default, and a path for a resident to
  have their episodes deleted on request — and, in the data model, its own tier that the ordinary curation and
  training jobs do not read from by default, rather than being pooled with the controlled-environment fleet's
  data.
- **Ten times more robots.** Storage and ingestion scale roughly linearly — about 37.9 PB/year of raw sensor
  data instead of 3.79 PB — which pushes the metadata index from a single instance to one sharded by
  `robot_id` or `task_id`; labelling load scales the same way, pushing human spot checks further from a fixed
  count toward a fixed percentage of what the detector scores, which is already this design's default rather
  than a change forced by scale; de-duplication, optional headroom at 200 robots, becomes load-bearing, since
  near-duplicate demonstrations grow faster than genuinely new coverage once enough robots attempt the same
  task; and staged deployment can use smaller percentage steps for the same statistical confidence per stage,
  since 1% of a larger fleet is still more robots' worth of trials than 1% of this one.

<details>
<summary>Estimate check (runnable)</summary>

```python
import math
import numpy as np
from scipy import stats

# ---- camera data ----
frame_bytes = 40_000
cams, fps = 3, 15
cam_bytes_per_s = cams * fps * frame_bytes
assert cam_bytes_per_s == 1_800_000

day_s = 8 * 3_600
assert day_s == 28_800

robot_day_cam_bytes = cam_bytes_per_s * day_s
robot_day_cam_gb = robot_day_cam_bytes / 1e9
assert robot_day_cam_bytes == 51_840_000_000
assert robot_day_cam_gb == 51.84

n_robots = 200
fleet_day_cam_gb = robot_day_cam_gb * n_robots
assert fleet_day_cam_gb == 10_368.0
assert round(fleet_day_cam_gb / 1000, 1) == 10.4          # TB/fleet-day, as quoted

# ---- low-dimensional data (negligible -- computed, not assumed) ----
lowdim_bytes_per_s = 1_000
robot_day_lowdim_bytes = lowdim_bytes_per_s * day_s
robot_day_lowdim_mb = robot_day_lowdim_bytes / 1e6
assert robot_day_lowdim_bytes == 28_800_000
assert robot_day_lowdim_mb == 28.8

fleet_day_lowdim_gb = robot_day_lowdim_mb * n_robots / 1000
assert fleet_day_lowdim_gb == 5.76

assert cam_bytes_per_s / lowdim_bytes_per_s == 1_800        # camera stream is 1,800x the low-dim one
assert round(fleet_day_cam_gb / fleet_day_lowdim_gb) == 1_800

# ---- episodes per day ----
episode_s = 60
episodes_per_robot_day = day_s / episode_s
assert episodes_per_robot_day == 480

episodes_fleet_day = episodes_per_robot_day * n_robots
assert episodes_fleet_day == 96_000

teleop_frac, autonomous_frac = 0.30, 0.70
teleop_episodes = episodes_fleet_day * teleop_frac
autonomous_episodes = episodes_fleet_day * autonomous_frac
assert teleop_episodes == 28_800 and autonomous_episodes == 67_200
assert teleop_episodes + autonomous_episodes == episodes_fleet_day

# ---- a year of storage ----
fleet_day_total_gb = fleet_day_cam_gb + fleet_day_lowdim_gb
assert fleet_day_total_gb == 10_373.76

annual_gb = fleet_day_total_gb * 365
annual_pb = annual_gb / 1e6
assert annual_gb == 3_786_422.4
assert round(annual_pb, 2) == 3.79
assert round(annual_pb * 10, 1) == 37.9                      # ten times the fleet: follow-ups, below

print("all requirements-and-scale numbers check out")


# ---- Wilson score interval ----
def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Inverts the test statistic (phat - p) / sqrt(p(1-p)/n) directly, rather than substituting phat
    for p in the variance as the Wald interval does -- gives closer-to-nominal coverage for small n or
    phat near 0 or 1, both common when real-robot evaluation trials are expensive."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / trials
    denom = 1 + z ** 2 / trials
    center = (phat + z ** 2 / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / trials + z ** 2 / (4 * trials ** 2))
    return center - half, center + half


lo, hi = wilson_interval(42, 60)
assert round(lo, 3) == 0.575 and round(hi, 3) == 0.801

# the interval's center, from the same derivation: shrunk toward 0.5 relative to phat=0.700 itself
z95 = stats.norm.ppf(0.975)
assert round(z95, 3) == 1.960                                # as quoted: z approx 1.960 at 95% confidence
phat = 42 / 60
wilson_center = (phat + z95 ** 2 / 120) / (1 + z95 ** 2 / 60)
assert round(wilson_center, 3) == 0.688

# the Wald interval, for comparison: centred exactly on phat, unlike Wilson's shrinkage toward 0.5
wald_half = z95 * math.sqrt(phat * (1 - phat) / 60)
wald_lo, wald_hi = phat - wald_half, phat + wald_half
assert round(wald_lo, 3) == 0.584 and round(wald_hi, 3) == 0.816
assert wald_lo > lo and wald_hi > hi                         # NOTE: a common mistake is assuming Wald sits
                                                              # inside Wilson, as a "correction" would -- it
                                                              # doesn't in general: Wald is centred on phat,
                                                              # Wilson is shrunk toward 0.5, so the intervals
                                                              # shift relative to each other rather than nest
assert (hi - lo) < (wald_hi - wald_lo)                       # and, here, Wilson is also the narrower interval


# ---- trials per arm to tell 60% from 70% apart, alpha=0.05 two-sided, 80% power ----
def required_n_per_arm(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Two independent samples of size n each; H0: p1 == p2, tested with the pooled-variance z-test.
    Solves P(reject | H1: true rates p1, p2) = power for n (the standard normal-approximation
    sample-size formula for a two-proportion test)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / (p1 - p2) ** 2)


n_per_arm = required_n_per_arm(0.60, 0.70)
assert n_per_arm == 356

# an independent recomputation of the same formula, kept unrounded, to check the numbers quoted around it
z_check, zpow_check = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
assert round(z_check, 3) == 1.960                             # as quoted: z_alpha/2 approx 1.960
assert round(zpow_check, 3) == 0.842                          # as quoted: z_power approx 0.842
pbar_6070 = (0.60 + 0.70) / 2
raw_n_per_arm = (z_check * math.sqrt(2 * pbar_6070 * (1 - pbar_6070))
                 + zpow_check * math.sqrt(0.60 * 0.40 + 0.70 * 0.30)) ** 2 / (0.60 - 0.70) ** 2
assert round(raw_n_per_arm, 1) == 355.9                       # as quoted: n before rounding up to 356

one_task_trials = n_per_arm * 2
assert one_task_trials == 712                                 # as quoted: 712 real episodes, one task
one_task_hours = one_task_trials * episode_s / 3_600
assert round(one_task_hours, 2) == 11.87
assert one_task_hours < 12                                    # as quoted: "just under 12 hours"

# ---- Monte Carlo check of that power at n = 356 per arm ----
rng = np.random.default_rng(0)
reps = 20_000
p1, p2, alpha = 0.60, 0.70, 0.05
z_alpha = stats.norm.ppf(1 - alpha / 2)

x1 = rng.binomial(n_per_arm, p1, size=reps)
x2 = rng.binomial(n_per_arm, p2, size=reps)
phat1, phat2 = x1 / n_per_arm, x2 / n_per_arm
pbar_hat = (x1 + x2) / (2 * n_per_arm)
se = np.sqrt(pbar_hat * (1 - pbar_hat) * (2 / n_per_arm))
z_stat = np.divide(phat1 - phat2, se, out=np.zeros(reps), where=se > 0)
simulated_power = float(np.mean(np.abs(z_stat) > z_alpha))
mc_se = math.sqrt(simulated_power * (1 - simulated_power) / reps)
assert mc_se < 0.005                                          # simulation is precise enough to trust the check
assert abs(simulated_power - 0.80) < 3 * mc_se                 # within simulation noise of the target power
assert simulated_power == 0.7999                               # as quoted: this simulation's exact rejection rate

# ---- why not just run every task to this standard on real hardware ----
n_eval_tasks = 50
full_program_trials = n_eval_tasks * n_per_arm * 2
assert full_program_trials == 35_600
full_program_hours = full_program_trials * episode_s / 3_600
assert round(full_program_hours, 1) == 593.3
assert round(full_program_hours / 24, 1) == 24.7

print("evaluation-statistics numbers check out")


# ---- action-chunk latency budget ----
control_hz = 10
tick_ms = 1_000 / control_hz
assert tick_ms == 100

chunk_deadline_ms = 100                                        # given: at most 100 ms per chunk, any H
                                                                 # NOTE: it's the call *rate* that falls as H
                                                                 # grows, not this per-call ceiling -- a common
                                                                 # mix-up is assuming the ceiling itself shrinks

onboard_forward_ms = 70
onboard_margin_ms = chunk_deadline_ms - onboard_forward_ms
assert onboard_margin_ms == 30

net_one_way_ms, serialize_ms, server_forward_ms = 4, 5, 30
server_total_ms = 2 * net_one_way_ms + 2 * serialize_ms + server_forward_ms
assert server_total_ms == 48
server_margin_ms = chunk_deadline_ms - server_total_ms
assert server_margin_ms == 52

H = 8
buffer_runway_ms = H * tick_ms
assert buffer_runway_ms == 800
call_rate_hz = control_hz / H
assert call_rate_hz == 1.25

# a single slow call: absorbed with H=8 buffering, but would stall the control loop with H=1
slow_call_ms = 180
assert slow_call_ms > chunk_deadline_ms                        # it is genuinely a deadline miss
remaining_runway_ms = (H - 1) * tick_ms                         # buffer already holds H-1 unconsumed actions
assert remaining_runway_ms == 700
assert slow_call_ms < remaining_runway_ms                       # H=8: fully absorbed, no stall
assert slow_call_ms > tick_ms                                   # H=1: would have stalled the very next tick

print("action-chunk latency budget checks out")


# ---- staged rollout slice sizes (deep dive c) ----
for pct, expected_count in [(0.01, 2), (0.10, 20), (0.50, 100), (1.00, 200)]:
    assert round(pct * n_robots) == expected_count             # as quoted: 1%/10%/50%/100% of the fleet

print("staged-rollout slice sizes check out")


# ---- closing the loop: priority-weighted collection versus uniform, over 50 tasks ----
N_TASKS, ROUNDS, BUDGET, EPS, LEARN_RATE = 50, 20, 1_000, 0.02, 0.0015

init_rng = np.random.default_rng(7)
initial_success = init_rng.uniform(0.2, 0.7, size=N_TASKS)
initial_gap = 1 - initial_success
hardest_task = int(np.argmax(initial_gap))
easiest_task = int(np.argmin(initial_gap))


def run_collection(weighted: bool, seed: int = 123):
    """Each round allocates BUDGET newly collected episodes across N_TASKS tasks -- uniformly, or
    weighted toward whichever tasks currently fail most often, with a floor EPS so a task that has
    reached near-mastery is still sampled occasionally rather than dropped to zero. A task's gap to
    mastery shrinks with the episodes it receives, diminishing returns per episode (a fixed fraction
    LEARN_RATE of what remains)."""
    gap = initial_gap.copy()
    r = np.random.default_rng(seed)
    total_alloc = np.zeros(N_TASKS)
    min_prob = 1.0
    for _ in range(ROUNDS):
        if weighted:
            weight = gap + EPS                                  # NOTE: without this floor, a near-mastered
                                                                  # task (gap approx 0) gets weight approx 0 and
                                                                  # stops being sampled -- silent starvation
            probs = weight / weight.sum()
        else:
            probs = np.full(N_TASKS, 1.0 / N_TASKS)
        min_prob = min(min_prob, probs.min())
        alloc = r.multinomial(BUDGET, probs)
        total_alloc += alloc
        gap = gap * (1 - LEARN_RATE) ** alloc
    return gap, total_alloc, min_prob


gap_uniform, alloc_uniform, _ = run_collection(weighted=False)
gap_weighted, alloc_weighted, min_prob_weighted = run_collection(weighted=True)

# no starvation: even the easiest task keeps receiving some episodes under the weighted scheme
assert alloc_weighted.min() > 0
assert min_prob_weighted > 0

# the mechanism does what it is meant to: more of the budget goes to the task that started hardest,
# less to the one that started easiest, compared with uniform allocation
assert alloc_weighted[hardest_task] > alloc_uniform[hardest_task]
assert alloc_weighted[easiest_task] < alloc_uniform[easiest_task]

# and the outcome that mechanism is for: the worst-performing task ends up in better shape under
# weighted collection than under uniform, even though uniform spent the same total budget
assert gap_weighted.max() < gap_uniform.max()

print("closing-the-loop simulation confirms no starvation and improved worst-task outcomes")
print("all checks passed")
```

</details>

</details>
