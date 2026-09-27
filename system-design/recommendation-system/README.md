# Design a Recommendation System for a Content Feed

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| ML system design | ★★★☆☆ | Medium | MLE · SWE · Applied AI | candidate-generation, two-tower, ranking, multi-task-learning, feedback-loops, cold-start, ndcg | 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

Design the backend that fills a personalised home feed of short videos: a stream of items a user
scrolls through one at a time, each one selected specifically for that user out of a catalogue shared
by everyone. A *feed request* is one call a client makes when a user opens or refreshes the feed; it
returns an ordered list of items to show next. Placing one item in front of one user, through any feed
request, is one *impression* — logged whether or not the user does anything with it.

Producing the list for one feed request happens in three stages, in order. *Candidate generation*
narrows the full catalogue down to a much smaller set of items likely to interest this particular user,
cheaply enough to run for every request. The chosen mechanism for it is a *two-tower model*: two neural
networks trained together, one encoding a user (with their recent activity and context) into a
fixed-length embedding vector, the other encoding an item into an embedding vector of the same length,
trained so that the dot product of the two vectors is large for pairs the user is likely to engage
with. At serving time this turns candidate generation into a search of a precomputed index of item
embeddings for the ones closest to the user's — an *approximate nearest-neighbour (ANN) index*, which
trades a small amount of recall for speed against exhaustively scoring every item in the catalogue. A
*ranking model* then scores each surviving candidate on how likely the user is to engage with it in
several distinct ways, combining those predictions into a single number. *Re-ranking* takes the
top-scored candidates and reorders them under constraints the ranking score alone does not capture —
diversity between adjacent items, freshness for recently uploaded videos, policy rules — before the
system returns the top of this re-ranked list.

Two problems cut across all three stages. A *cold start* is the case of producing recommendations for
a user or an item with little or no interaction history yet — a new signup with no watch history, or a
video uploaded seconds ago with no engagement recorded against it. *Exposure bias* is the distortion
that comes from training only on logged interactions: the log only ever records feedback on items some
earlier version of the system chose to show, so a model trained on it naively learns to prefer whatever
that earlier system already preferred, never discovering a good item it never showed in the first
place.

Scale this design for:

- $200{,}000{,}000$ daily active users.
- Each active user opens the feed $10$ times a day; each open is one feed request and returns $20$
  items.
- Peak traffic runs at $3\times$ the daily average.
- A catalogue of $500{,}000{,}000$ videos, growing by $5{,}000{,}000$ new uploads a day.
- Target latency for a feed request: 99th percentile (p99) under $200$ ms.
- Interaction events — impression, watch time, like, share, skip — must be usable by the models within
  $1$ hour of occurring.
- The business goal is long-term user satisfaction, not raw click or watch counts.

In scope: candidate generation, including the two-tower model and other candidate sources; the ranking
model and how its predictions combine into one score; re-ranking for diversity, freshness and policy;
the serving architecture and interaction logging; the training pipeline — labelling, negative sampling,
delayed feedback, retraining cadence; cold start for new users and new videos; feedback loops and
exposure bias; and offline and online evaluation. Out of scope: recommendation surfaces other than the
home feed (search results, notifications); the storage, transcoding and delivery of the video files
themselves; the trust-and-safety classifiers that decide whether a video is allowed on the platform at
all (assume every candidate item already carries a policy-eligibility flag by the time it reaches this
system); and authentication.

Produce:

1. A requirements and scale estimate: the average and peak feed-request rate; the number of items
   scored per second by the ranking model (state your assumption for how many candidates it scores per
   request); the retrieval index's memory footprint for $500{,}000{,}000$ items with $128$-dimensional
   `int8` embeddings; and the interaction-log volume per day (state your assumption for the size of one
   logged event).
2. The data model for users, items and interactions, stating which features must be available within
   the $1$-hour freshness bound and which can update on a slower cycle.
3. An architecture diagram and a walk-through of one feed request, covering candidate generation (the
   two-tower retrieval index plus at least one other candidate source), ranking, re-ranking, and how
   interactions are logged for training.
4. The training pipeline: how labels are derived from interactions, negative sampling for the two-tower
   model, how delayed feedback is handled, and the retraining cadence for each model.
5. Deep dives into: (a) choosing the ranking objective and combining several predicted signals into one
   score; (b) cold start for new users and new videos; (c) feedback loops and exposure bias, including
   exploration and logging propensities; (d) offline evaluation (recall@k for candidate generation,
   NDCG@k for ranking) and online evaluation (A/B test metrics and guardrails).

## Reference solution

<details>
<summary>Show the reference solution</summary>

Worth confirming before designing: what "long-term satisfaction" is measured by in practice (assumed
here: a blend of return-visit rate and total time spent on completed, not skipped, items, validated
periodically against slower online reads rather than trusted as a single proxy forever — see the
objective and evaluation deep dives), and what already constrains candidate content before it reaches
this system (assumed here: every candidate arrives with a policy-eligibility flag this design treats as
a given input, not something it decides).

### Requirements and scale

**Feed-request rate.** $200{,}000{,}000$ daily active users each open the feed $10$ times a day:

$$200{,}000{,}000 \times 10 = 2{,}000{,}000{,}000 \text{ requests/day,} \qquad
\frac{2{,}000{,}000{,}000}{86{,}400} \approx 23{,}148 \text{ requests/s average.}$$

At $3\times$ the daily average, peak throughput is about $69{,}444$ requests/s.

**Items returned versus items scored.** Each request returns $20$ items, so the response path — and
its logging — must sustain about $23{,}148 \times 20 \approx 462{,}963$ items/s on average and about
$1{,}388{,}889$ items/s at peak. The ranking model sees a larger number: candidate generation is set to
surface $C = 500$ candidates per request, $25\times$ what is finally shown — wide enough that
re-ranking has real choices for diversity and freshness, narrow enough to keep the ranking model's
per-request cost bounded — so the ranking model scores about $23{,}148 \times 500 \approx 11{,}574{,}074$
items/s on average and about $34{,}722{,}222$ items/s at peak.

**Ranking fleet.** Assume one ranking replica scores one request's $500$ candidates in a single batched
forward pass taking about $20$ ms, for a throughput of $500 / 0.02 = 25{,}000$ items/s. Meeting peak
demand needs $34{,}722{,}222 / 25{,}000 \approx 1{,}389$ replicas bare; with $20\%$ headroom for uneven
load and rolling deploys, $1{,}667$ replicas. (Batching several requests' candidates into one forward
pass, instead of one request per call, would raise this per-replica throughput and shrink the fleet;
the arithmetic here keeps the simpler one-request-per-call model.)

**Candidate index memory.** The retrieval index holds one $128$-dimensional embedding per catalogue
item, quantised to `int8` — one byte per dimension:
$500{,}000{,}000 \times 128\text{ B} = 64{,}000{,}000{,}000$ B, exactly $64$ GB. Sized into shards of
about $8$ GB each — comfortably inside one machine's memory alongside serving overhead — that is $8$
shards; replicated $3\times$ for availability and to spread query fan-out, $24$ machines serve the
index. The catalogue grows by $5{,}000{,}000$ items a day, $1\%$ of its size, adding only
$5{,}000{,}000 \times 128\text{ B} \approx 0.64$ GB of new embeddings a day — the index's memory
footprint is set by the existing catalogue, not by daily ingestion.

**Event volume.** Every impression is logged as one row: an event id, the request id grouping it with
the other $19$ items shown alongside it, the user id, the item id and a server timestamp ($8$ B each,
$40$ B total), plus its outcome — position in the feed ($1$ B), watch time in milliseconds ($4$ B), a
bitmask of engagement flags such as liked or shared ($1$ B), and the ranking score it was shown with,
kept for later analysis ($4$ B as a float32) — $50$ B/event. At
$200{,}000{,}000 \times 10 \times 20 = 40{,}000{,}000{,}000$ impressions/day, that is
$40{,}000{,}000{,}000 \times 50\text{ B} = 2{,}000{,}000{,}000{,}000$ B, exactly $2$ TB/day. A $30$-day
hot window kept for training and offline evaluation is therefore about $60$ TB.

### Data and features

**User** — a stable profile (`user_id`, signup date, declared locale, onboarding topic preferences)
updated on a daily batch cycle, plus a rolling summary of recent activity (session count, last $N$
engaged items and their outcomes) updated by the streaming pipeline below, plus session-local state
(items already shown or skipped in the session in progress) held in a low-latency cache and read
synchronously inside the request itself, since it must reflect the last few seconds, not the last hour.
The user tower of the two-tower model consumes the profile and the rolling summary and produces a
fixed-length *user embedding*, recomputed whenever a request needs it — cheap, since it is one small
forward pass, not a lookup into a precomputed table.

**Item** — metadata fixed at upload (`item_id`, `creator_id`, duration, language, topic tags, upload
time) plus rolling engagement-rate aggregates (trailing 24-hour and 7-day watch-through rate, like
rate, share rate) updated by the streaming pipeline. The item tower consumes both and produces an
*item embedding*; unlike the user embedding, it is precomputed for the whole catalogue and refreshed on
the training cadence below, since candidate generation must search a fixed, indexed set of vectors
rather than embed every item at request time.

**Interaction** — one row per impression, written by the serving path itself: `event_id`,
`request_id`, `user_id`, `item_id`, `position`, `served_at`, `watch_time_ms`, `engagement_flags`
(liked, shared), and `ranking_score`. Collapsing what could be five separate events — impression, watch
time, like, share, skip — into one row keyed by the impression avoids writing, and later joining, five
rows for every item shown; a *skip* is simply a row whose `watch_time_ms` is near zero, not a separate
event type.

**Freshness.** Three tiers, matched to how quickly each feature can change and how expensive it is to
keep current: session-local state is read synchronously inside the request and is current to the
second; the rolling user and item aggregates are updated by a streaming job consuming the interaction
log and are current within the requirement's $1$-hour bound; user and item embeddings, and everything
else derived by training, follow the retraining cadence in the next section and are current to a day. A
feed request only ever blocks on the first tier; the other two are always read from whatever the
background pipelines most recently wrote.

### Architecture

```text
client
  |
  | feed request
  v
+----------------------------------------------------------------------+
| Feed API  (reads user + session features; logs impressions)          |
+----------------------------------------------------------------------+
  |
  | user, context
  v
+----------------------------------------------------------------------+
| Candidate generation                                                 |
|                                                                      |
|   * Two-tower model       -> search the ANN retrieval index          |
|   * Fresh-item pool       -> recent uploads, by upload time          |
|   * Followed-creator pool -> unseen uploads, followed creators       |
|                                                                      |
|   merge, de-duplicate, drop already-shown and ineligible items       |
+----------------------------------------------------------------------+
  |
  | ~500 candidates
  v
+----------------------------------------------------------------------+
| Ranking model                                                        |
| multi-task: predicts several engagement signals per candidate        |
| and combines them into one score                                     |
+----------------------------------------------------------------------+
  |
  | scored candidates
  v
+----------------------------------------------------------------------+
| Re-ranking                                                           |
| diversity, freshness, and policy constraints                         |
+----------------------------------------------------------------------+
  |
  | top 20 -> Feed API -> client
  v
(response returned; every shown item is also logged, below)


+----------------------------------------------------------------------+
| Event log  (one row per impression)                                  |
+----------------------------------------------------------------------+
        |
        +--------------------------------+
        |                                |
        v                                v
+-------------------------+   +------------------------------+
| Streaming feature       |   | Offline training pipeline    |
| aggregation             |   | (labels, negative sampling,  |
| (updates within 1 hour) |   | retraining -- see Training)  |
+-------------------------+   +------------------------------+
        |                                   |
        v                                   +----------------------+
   Feature store                            |                      |
   (read by candidate                       v                      v
    generation, ranking)             new item embeddings    new model weights
                                      -> retrieval index     -> ranking model
                                         (above)                (above)
```

A feed request reaches the Feed API, which reads the requesting user's profile, rolling summary and
session state from the feature store — the only feature read on the critical path; item features are
already folded into the index and the ranking model's inputs by the background pipelines. Three
candidate sources run in parallel: the two-tower retrieval service embeds the user and searches the ANN
index for its nearest items by embedding dot product; the fresh-item pool holds recently uploaded items
ranked by upload time within each topic, for items too new to have earned a place in the embedding
index yet; the followed-creator pool returns unseen recent uploads from creators the user follows
directly, bypassing the relevance model entirely for that relationship. Candidate merge de-duplicates
across the three sources, drops anything already shown this session or failing the policy-eligibility
flag, and passes on roughly $500$ surviving candidates.

The ranking model scores every surviving candidate in one batched forward pass, producing several
predicted engagement probabilities per candidate and combining them into one score (the objective deep
dive below). Re-ranking walks the scored list once, from highest score down, filling $20$ output slots
subject to its constraints — no more than a small fixed number of consecutive items from the same
creator, a minimum share of slots going to items uploaded in the last day, and any remaining policy
caps — skipping a candidate that would violate a constraint in favour of the next-best one that does
not. The Feed API returns the resulting $20$ items and, independently of the response, logs one
interaction row per item to the event log. The event log feeds two consumers: the streaming aggregation
job that keeps rolling features current within the hour, and the offline training pipeline, which
writes back new item embeddings to the retrieval index and new model weights to the ranking service on
the cadence the training section sets out.

### Training

**Labels.** Every training row starts from one interaction row (Data and features, above):
$P(\text{watch} \ge 50\%)$'s label is $1$ when `watch_time_ms` reaches half the item's duration,
$P(\text{like})$ and $P(\text{share})$ read directly off `engagement_flags`, and
$P(\text{skip within 2 s})$'s label is $1$ when `watch_time_ms` is under $2{,}000$. All four are
computed from the same logged row, so one interaction produces one training example carrying all four
labels at once, for the shared multi-task model.

**Negative sampling for the two-tower model.** The ranking model above is trained on rows that already
carry both a positive and an implicit negative signal (a low score just means low predicted
engagement), but the two-tower retrieval model needs an explicit contrast: a positive pair is a (user,
item) row with $P(\text{watch} \ge 50\%)=1$, and its negatives are drawn *in-batch* — every other row's
item in the same training batch serves as a negative for this row's user, at no extra data-loading
cost, using the loss

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log
\frac{\exp(u_i \cdot v_i / \tau)}{\sum_{j=1}^{B} \exp(u_i \cdot v_j / \tau)},$$

for a batch of $B$ (user, positive item) pairs with user embeddings $u_i$ and item embeddings $v_i$ and
a temperature $\tau$ (the estimate check below computes this loss on a toy batch and checks it against
the same formula evaluated directly, term by term). In-batch negatives are popular items more often
than a uniform sample over the catalogue would be, simply because a random batch already
over-represents them — a useful property, not a flaw, since it manufactures exactly the hard negatives
(plausible, popular items the user did not engage with) retrieval will actually have to out-score at
serving time; the standard correction for the resulting sampling bias subtracts each negative's log
sampling probability from its logit before the softmax, so an item is not penalised purely for being
sampled more often.

**Delayed feedback.** `like` and `share` can arrive minutes after the impression that produced them, so
a label join run too soon undercounts positives for the most recent interactions. This design joins
labels a fixed attribution window (for example $6$ hours) after the impression, re-reading the
interaction row at join time rather than at write time, and treats an item with no positive signal by
the end of that window as a permanent negative — a small, bounded amount of noise from the rare
later-than-the-window action, traded for a join that never blocks on an unbounded wait.

**Retraining cadence.** The ranking model is retrained every few hours by fine-tuning on the
interactions joined since its last update, so it tracks fast-moving trends and the freshly aggregated
features the data section's streaming tier produces. The two-tower model is retrained from scratch once
a day and its retrieval index rebuilt from the resulting item embeddings, since a full retrain gives
every item's embedding a consistent, comparable geometry that an incremental update to only the day's
active items would not; the fresh-item pool and the cold-start smoothing below are what cover a video
in the gap between its upload and the next index rebuild.

### Deep dives

**(a) Choosing the objective and combining predicted signals.** The ranking model is multi-task: shared
lower layers consume the merged candidate's features, and separate heads predict several engagement
probabilities for it — $P(\text{watch} \ge 50\%)$, $P(\text{like})$, $P(\text{share})$, and
$P(\text{skip within 2 s})$ as an explicit negative signal — trained jointly from the same interaction
rows (Training, above), which lets the rarer labels (like, share) benefit from representations learned
mostly from the much more common watch-time label, rather than each being trained from a separate,
smaller model. Combining the four predictions into the one score re-ranking consumes is a weighted sum,

$$\text{score} = w_1 P(\text{watch} \ge 50\%) + w_2 P(\text{like}) + w_3 P(\text{share})
- w_4 P(\text{skip within 2 s}),$$

with weights set by experimentation against the online guardrails and primary metric in the evaluation
deep dive below, not learned end-to-end inside the model itself — chosen over training a single model
directly against one composite label, because the label a single model would need is exactly the
long-horizon satisfaction signal the business goal cares about, and that signal is both sparse (most
sessions do not resolve to an explicit satisfaction outcome) and slow (a return-rate label needs days to
resolve, long after the ranking decision that needs to learn from it). Optimising raw engagement alone
— collapsing the weighted sum to just $P(\text{watch})$ or a raw click signal — is rejected outright: it
rewards anything that holds attention regardless of the user's later regret, which the multi-task
predictions individually still capture (a share is a stronger signal of genuine value than a
watched-to-the-end video that is never revisited) but a single engagement number cannot distinguish. The
weights themselves are the tuning surface: raising $w_4$ trades some raw watch time for fewer quick
abandons, and the online guardrails, not an offline metric, are what actually validate that the trade is
worth it.

**(b) Cold start for new users and new videos.** A newly uploaded video has no engagement history, so
its rolling engagement-rate features (the item entity's 24-hour and 7-day aggregates) start undefined;
rather than leaving them undefined or zero — either of which the ranking model would misread as *known
to be unpopular* — they are initialised to a smoothed prior and shrink toward the item's own observed
rate as evidence accumulates: writing $n$ for the observed impression count and $\hat r$ for the item's
own observed rate so far, the feature used is $\frac{n\hat r + \kappa \bar r}{n + \kappa}$ for the
category average rate $\bar r$ and a constant $\kappa$ setting how many impressions of real evidence it
takes to outweigh the prior — equal to the prior at $n=0$ and converging to $\hat r$ as $n$ grows. This
still leaves the question of exposure: the item embedding used for retrieval is produced by the item
tower from upload-time metadata alone, so a brand-new video gets a real position in the ANN index and
can be retrieved by ordinary two-tower similarity from the moment it is uploaded, without waiting on
engagement data the two-tower model was never given as an input in the first place. The fresh-item pool
in the architecture above is a second, independent path for the same problem: a small quota of slots
reserved for recent uploads regardless of what retrieval or ranking would otherwise have chosen, which
matters because a video's content-only embedding, however good, is still competing on the ranking
model's engagement predictions against items with real engagement history behind them, and those
predictions are — correctly — pessimistic without any evidence yet.

A new user's session-local and rolling-activity features are similarly empty at signup; onboarding-time
signals (declared topic preferences, locale, device) seed the user tower's input until enough activity
accumulates, and personalisation itself is phased in rather than switched on immediately: candidate
generation and re-ranking blend a population-level popularity signal with the user embedding's own
similarity scores, in a proportion that shifts toward the personalised signal as the account's
interaction count grows, so a new account's first few sessions draw on what works broadly rather than on
a user embedding built from too little evidence to be reliable yet.

**(c) Feedback loops and exposure bias.** The interaction log this system trains on is generated by this
system's own earlier decisions: an item never shown earns no impression, so it can never earn a positive
label either, however good it might have been — training on the log alone therefore only ever reinforces
what an earlier model already preferred, the *exposure bias* defined in the problem statement. Two
mechanisms address it. First, *logging propensity*: every impression is logged with the probability the
serving policy assigned to placing that item where it did, not only whether it was shown; this turns
training into a form of importance-weighted learning — weighting an example by the inverse of its logged
propensity — so an item that was rarely shown but did well when it was counts more than its raw
impression count alone would suggest, correcting the training sample back toward what an unbiased log
would look like. Second, *exploration*: a small, fixed fraction of impressions (for example $2\%$) is
drawn uniformly at random from the merged candidate pool rather than by predicted score, deliberately
trading a little short-term engagement for coverage of items the current model would not otherwise have
shown at all — an item with a middling predicted score but genuine appeal only gets the chance to prove
it through this path, since a purely score-ranked slot never surfaces it often enough to find out. A
uniformly random slot's propensity is exact and trivial to log (one over the number of eligible
candidates); a purely score-ranked slot's propensity is not, since it depends on every other candidate's
score in that request — which is why some exploration traffic is kept even though a model with slightly
lower short-term engagement is, on paper, the cost of running it.

**(d) Evaluation offline and online.** Two offline metrics, one per stage, since a stage can only be
evaluated against what it is actually responsible for. *Recall@k* evaluates candidate generation alone:
over a held-out set of (user, item) pairs the user is later observed to engage with, the fraction of
those pairs whose item appears anywhere in the top $k$ candidates the retrieval stage would have
surfaced for that user — a candidate generator that never proposes an item gives ranking nothing to
recover, so recall@k is checked before ranking is blamed for a miss. *NDCG@k* (normalised discounted
cumulative gain) evaluates ranking's ordering of whatever candidates it was given, against a graded
relevance label per candidate — for example $0$ for a skip, $1$ for a partial watch, $2$ for a completed
watch, $3$ for a like, $4$ for a share. Writing $\mathrm{rel}_i$ for the true relevance of the item the
model placed at rank $i$ (rank $1$ first):

$$\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i+1)}, \qquad
\mathrm{NDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k},$$

where $\mathrm{IDCG@}k$ is the same sum computed on the ideal ordering — the candidates sorted by true
relevance, descending — so a perfect ranking scores $1$, and $\mathrm{NDCG@}k$ is defined as $0$ when
every candidate has relevance $0$ (then $\mathrm{IDCG@}k = 0$ too). For four candidates with true
relevances $[3, 2, 3, 0]$ in the model's chosen order, $\mathrm{DCG@4} \approx 12.393$ and
$\mathrm{IDCG@4}$ — the same four re-sorted, $[3,3,2,0]$ — $\approx 12.917$, giving
$\mathrm{NDCG@4} \approx 0.9595$: close to $1$ because the model's order is nearly, not perfectly, the
ideal one (the two rank-$3$ items are swapped).

Online, a launch is judged by a randomised A/B test, not by the offline metrics alone, since neither
recall@k nor NDCG@k can see the long-term satisfaction the business goal actually cares about. The
primary metric is a short-horizon proxy correlated with it — for example the fraction of the treatment
group that returns within $7$ days, and total time spent on completed, not skipped, items — checked
periodically against a slower, harder-to-run long-horizon read rather than trusted blindly, since
waiting for the long-horizon number on every candidate change would make iteration too slow. Guardrails
run alongside the primary metric and can block a launch even when it wins: p99 feed-request latency must
stay under the $200$ ms target; a complaint or hide-content rate must not regress; and the share of
impressions and watch time going to creators outside some popularity threshold must not shrink — the
online counterpart of the fairness follow-up below.

### Follow-ups

- **Item understanding with a multimodal encoder.** Replacing the item tower's hand-built metadata
  features with embeddings from a pretrained encoder over the video's frames, audio and any on-screen or
  caption text would give a real notion of content similarity alongside engagement-based similarity,
  improving both new-item cold start (a genuinely content-based embedding from the moment of upload, not
  just metadata) and re-ranking diversity (two items can be recognised as similar with no shared engagement
  pattern yet); the cost is a heavier per-upload embedding pipeline that must keep up with
  $5{,}000{,}000$ uploads a day rather than a handful of metadata lookups.
- **Real-time features at scale.** Not every feature can wait for the streaming tier's $1$-hour bound:
  this session's already-shown items must be checked inside the request itself, so they live in a
  low-latency store keyed by session, separate from the slower streaming aggregation pipeline —
  conflating the two would either slow every request down to the streaming tier's latency or stale the
  session-local check by up to an hour, silently reshowing items.
- **Fairness to new creators.** The training log and the ranking objective both compound against a small
  creator — less engagement data yields worse predictions, which yields less exposure, which yields even
  less data — so protecting new creators needs an explicit choice, not a side effect of the existing
  mechanisms: a re-ranking exposure quota for creators below a follower or view threshold, distinct from
  the exploration slots in the feedback-loop deep dive (those exist to improve the model, not to
  guarantee any creator segment's exposure), tracked as its own guardrail metric in every online test
  rather than checked only when someone happens to notice a problem.
- **Privacy constraints on features.** Some plausible features are excluded on principle rather than for
  lack of predictive value: one user's individual watch history never becomes a feature of another
  user's request, only aggregated, anonymised statistics do; and the interaction log itself carries no
  free-text or precise-location fields, so a compromised or over-broadly-accessed copy of it exposes
  aggregated behaviour patterns, never a specific person's specific viewing session.

<details>
<summary>Estimate check (runnable)</summary>

```python
import math
import random

import numpy as np

# ---- feed-request rate ----
dau = 200_000_000
opens_per_day = 10
items_per_request = 20
peak_factor = 3

requests_per_day = dau * opens_per_day
assert requests_per_day == 2_000_000_000

avg_qps = requests_per_day / 86_400
assert round(avg_qps) == 23_148

peak_qps = avg_qps * peak_factor
assert round(peak_qps) == 69_444

# ---- items returned vs. items scored ----
items_returned_avg = avg_qps * items_per_request
items_returned_peak = peak_qps * items_per_request
assert round(items_returned_avg) == 462_963
assert round(items_returned_peak) == 1_388_889

candidates_per_request = 500
assert candidates_per_request == 25 * items_per_request

items_scored_avg = avg_qps * candidates_per_request
items_scored_peak = peak_qps * candidates_per_request
assert round(items_scored_avg) == 11_574_074
assert round(items_scored_peak) == 34_722_222

# ---- ranking fleet ----
batch_time_s = 0.020
throughput_per_replica = candidates_per_request / batch_time_s
assert throughput_per_replica == 25_000.0

raw_replicas = items_scored_peak / throughput_per_replica
assert round(raw_replicas) == 1_389
headroom = 0.2
n_rank_replicas = math.ceil(raw_replicas * (1 + headroom))
assert n_rank_replicas == 1_667

# ---- candidate index memory ----
catalogue = 500_000_000
emb_dim = 128
bytes_per_dim = 1  # int8
index_bytes = catalogue * emb_dim * bytes_per_dim
index_gb = index_bytes / 1e9
assert index_gb == 64.0

shard_gb = 8
n_shards = index_gb / shard_gb
assert n_shards == 8.0
replication = 3
n_index_machines = int(n_shards * replication)
assert n_index_machines == 24

new_items_per_day = 5_000_000
assert new_items_per_day / catalogue == 0.01
new_emb_gb_per_day = new_items_per_day * emb_dim * bytes_per_dim / 1e9
assert round(new_emb_gb_per_day, 2) == 0.64

# ---- event-log volume ----
ids_bytes = 8 * 5              # event_id, request_id, user_id, item_id, served_at
outcome_bytes = 1 + 4 + 1 + 4   # position, watch_time_ms, engagement_flags, ranking_score
row_bytes = ids_bytes + outcome_bytes
assert row_bytes == 50

impressions_per_day = dau * opens_per_day * items_per_request
assert impressions_per_day == 40_000_000_000

daily_event_bytes = impressions_per_day * row_bytes
daily_event_tb = daily_event_bytes / 1e12
assert daily_event_tb == 2.0

hot_days = 30
hot_tb = daily_event_tb * hot_days
assert hot_tb == 60.0

print("all requirements-and-scale numbers check out")


# ---- NDCG@k, checked against an independent brute-force computation from the definition ----
def ndcg_at_k(relevance, k):
    """relevance: true relevance of each candidate, already in the model's ranked order (best first)."""
    relevance = np.asarray(relevance[:k], dtype=float)
    discounts = np.log2(np.arange(2, len(relevance) + 2))   # NOTE: rank 1 -> log2(2), never log2(1) == 0
    dcg = float(np.sum((2 ** relevance - 1) / discounts))
    ideal = np.sort(relevance)[::-1]
    idcg = float(np.sum((2 ** ideal - 1) / discounts))
    return dcg / idcg if idcg > 0 else 0.0


example_rel = [3, 2, 3, 0]                 # the worked example from the text
example_ndcg = ndcg_at_k(example_rel, 4)
assert round(example_ndcg, 4) == 0.9595


def brute_ndcg(relevance, k):
    """The same definition, recomputed independently in plain Python: no NumPy, no shared helper."""
    relevance = list(relevance[:k])
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


rng_py = random.Random(0)
for _ in range(2_000):
    n = rng_py.randint(1, 12)
    k = rng_py.randint(1, n)
    rel = [rng_py.randint(0, 4) for _ in range(n)]
    assert abs(ndcg_at_k(rel, k) - brute_ndcg(rel, k)) < 1e-9

assert ndcg_at_k([0, 0, 0], 3) == 0.0      # every candidate irrelevant: IDCG@k == 0 by definition

print("NDCG@k matches an independent brute-force computation on 2,000 random relevance lists")


# ---- recall@k of a toy two-tower retrieval index against exhaustive search ----
def build_ivf_index(item_vectors, n_clusters, seed, n_iters=5):
    """A minimal inverted-file index: a few iterations of Lloyd's algorithm assign each item to the
    nearest of n_clusters centroids, used to prune the search at query time."""
    rng = np.random.default_rng(seed)
    n_items = item_vectors.shape[0]
    centroids = item_vectors[rng.choice(n_items, size=n_clusters, replace=False)].copy()
    assignment = np.zeros(n_items, dtype=int)
    for _ in range(n_iters):
        dists = ((item_vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        assignment = dists.argmin(axis=1)
        for c in range(n_clusters):
            members = item_vectors[assignment == c]
            if len(members) > 0:            # NOTE: an empty cluster keeps its old centroid, never NaNs out
                centroids[c] = members.mean(axis=0)
    dists = ((item_vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return centroids, dists.argmin(axis=1)


def ivf_topk(query, item_vectors, centroids, assignment, nprobe, k):
    """Approximate top-k by dot product, searching only the nprobe clusters whose centroid is closest
    to the query. Clustering uses squared distance; retrieval scores by dot product, matching the
    two-tower model's own training objective -- the two need not be the same metric."""
    probe = np.argsort(-(centroids @ query))[:nprobe]
    idx = np.nonzero(np.isin(assignment, probe))[0]
    order = np.argsort(-(item_vectors[idx] @ query))[:k]
    return idx[order]


rng = np.random.default_rng(0)
n_items, dim, n_clusters, k = 2_000, 32, 25, 10
item_vectors = rng.normal(size=(n_items, dim))
centroids, assignment = build_ivf_index(item_vectors, n_clusters, seed=1)
assert len(assignment) == n_items and set(assignment.tolist()) <= set(range(n_clusters))

n_queries = 60
queries = rng.normal(size=(n_queries, dim))


def recall_at(nprobe):
    total = 0.0
    for q in queries:
        exact = set(np.argsort(-(item_vectors @ q))[:k].tolist())    # brute force: never calls ivf_topk
        approx = set(ivf_topk(q, item_vectors, centroids, assignment, nprobe, k).tolist())
        total += len(exact & approx) / k
    return total / n_queries


recall_1 = recall_at(1)
recall_5 = recall_at(5)
recall_full = recall_at(n_clusters)

assert recall_full == 1.0                       # every cluster searched == exhaustive search, exactly
assert 0.10 < recall_1 < 0.30
assert 0.40 < recall_5 < 0.70
assert recall_1 < recall_5 < recall_full         # probing more clusters never recovers fewer true neighbours

print(f"recall@{k}: nprobe=1 -> {recall_1:.3f}, nprobe=5 -> {recall_5:.3f}, nprobe=full -> {recall_full:.3f}")
print("toy retrieval index recall matches exhaustive search once every cluster is probed")


# ---- in-batch softmax loss for the two-tower model, checked against a direct formula ----
def in_batch_softmax_loss(user_emb, item_emb, temperature):
    """In-batch negatives: for row i, item_emb[i] is the positive and every item_emb[j] in the same
    batch is a negative for user_emb[i]."""
    logits = (user_emb @ item_emb.T) / temperature             # logits[i, j] = u_i . v_j / tau
    # NOTE: subtract the row max before exponentiating -- without it a large logit overflows exp()
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(np.diag(log_probs)))


def direct_in_batch_loss(user_emb, item_emb, temperature):
    """The same formula, evaluated term by term with no matrix operations and no shared helper."""
    B = user_emb.shape[0]
    total = 0.0
    for i in range(B):
        num = math.exp(float(np.dot(user_emb[i], item_emb[i])) / temperature)
        den = sum(math.exp(float(np.dot(user_emb[i], item_emb[j])) / temperature) for j in range(B))
        total += -math.log(num / den)
    return total / B


rng = np.random.default_rng(3)
for _ in range(200):
    B, dim = int(rng.integers(2, 9)), int(rng.integers(2, 6))
    user_emb = rng.normal(size=(B, dim))
    item_emb = rng.normal(size=(B, dim))
    temperature = float(rng.uniform(0.05, 1.0))
    assert abs(in_batch_softmax_loss(user_emb, item_emb, temperature)
               - direct_in_batch_loss(user_emb, item_emb, temperature)) < 1e-6

print("in-batch softmax loss matches a direct, term-by-term evaluation of the same formula on 200 random batches")

print("all checks passed")
```

</details>

</details>
