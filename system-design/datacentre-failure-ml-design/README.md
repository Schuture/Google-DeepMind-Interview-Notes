# ML Design: Predicting Which Data-Centre Machines Need Replacing

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| ML system design | ★★☆☆☆ | Medium | MLE · RE · SWE | predictive-maintenance, label-construction, censoring, class-imbalance, categorical-embeddings, survival-analysis, precision-at-k, feedback-loops | 45–60 min | Skills interview |
<!-- meta:end -->

## Problem

Design a system that predicts, ahead of time, which machines in a large data-centre fleet are likely to
need a hardware replacement, so the operations team can service them before they fail in production. A
*machine* is one physical server in the fleet; its *machine type* is the hardware model it was built from
(CPU generation, motherboard, chassis) — several hundred distinct types are in service at once, and new
types are introduced periodically as procurement refreshes the catalogue. A *repair ticket* records that a
technician replaced one or more hardware components on a machine, which components, and when. A
*decommission record* says when a machine left the fleet and why (a planned refresh, a relocation, or a
failure severe enough to retire the unit outright, among other reasons).

Scale this design for:

- $1{,}000{,}000$ machines across $20$ data centres, drawn from about $300$ machine types, with new types
  introduced roughly every quarter.
- Each machine reports about $50$ telemetry metrics once a minute: component temperatures, fan speeds,
  corrected memory-error counts, disk health counters, cumulative reboot counts, CPU throttling events, and
  power draw.
- Repair tickets record which component was replaced and when; decommission records say when and why a
  machine left the fleet.
- About $0.5\%$ of machines need a hardware replacement in any given $30$-day window.
- The operations team can drain and pre-emptively service at most $1{,}000$ machines a week.
- An unplanned failure in production costs about $20\times$ what a pre-emptive replacement costs.

Goal: every week, produce a ranked list of the machines most likely to need a hardware replacement within
the next $30$ days, so operations can spend its fixed weekly capacity where it matters most.

In scope: the whole prediction system, from the raw logs to the weekly ranked list and the feedback loop it
creates. Out of scope: the telemetry-collection agent itself, and the physical logistics of a repair (parts
inventory, technician scheduling).

Produce:

1. The questions to ask before designing, and a precise label definition: what counts as "needs replacing"
   and over what horizon.
2. How to build the training set from the logs above: snapshots, feature windows, label windows, leakage,
   censoring, and correlated rows.
3. Features and feature engineering, and how the transforms differ for a linear model, a tree ensemble and
   a neural network.
4. Embeddings for machine type and other high-cardinality categorical attributes, including a type
   introduced after training with no examples of its own.
5. Three candidate models and the loss function of each.
6. How the class imbalance is handled.
7. Evaluation: which metrics, what each one means, why it is the one to report, and the backtest design.
8. Deployment: how the predictions are used, monitoring, and the feedback loop the model's own actions
   create.

Questions the interviewer may interleave, once the relevant part of the design comes up:

- Why is the label imbalanced, and how do the chosen models and evaluation metrics account for it rather
  than being misled by it?
- How do you produce a usable embedding for a machine type with zero training examples?
- What does each evaluation metric actually measure, and why is it — rather than an alternative — the number
  to report to the operations team?

## Reference solution

<details>
<summary>Show the reference solution</summary>

Worth confirming before designing: whether a hardware replacement made as part of a *planned* fleet refresh
should count toward the label at all (assumed here: no — the target is unplanned, failure-driven
replacement, and a machine already inside its scheduled refresh window is excluded from both the positive
and the negative pool for that reason), and whether a machine type introduced this quarter must be scored
from day one (assumed yes, which is what motivates building its embedding from its own descriptive
attributes rather than waiting on data it will not have in time — Features, below).

### Requirements and scale

**Raw telemetry volume.** Each of $1{,}000{,}000$ machines reports $50$ metrics once a minute, $1{,}440$
times a day:

$$1{,}000{,}000 \times 50 \times 1{,}440 = 7.2\times10^{10} \text{ readings/day.}$$

Stored as an $8$-byte floating-point value each, that is $7.2\times10^{10}\times8=5.76\times10^{11}$ bytes,
$576$ GB/day — large enough that the design keeps only a rolling window of raw readings (a few weeks, for
audits and re-aggregation) and computes everything the models actually consume as hourly or daily
aggregates, never training directly off the per-minute stream.

**Training rows.** One snapshot per machine per week gives $1{,}000{,}000\times52=52$ million snapshot rows
a year — several orders of magnitude below the raw telemetry it is built from, a size an ordinary tabular
pipeline handles comfortably.

**Expected weekly positives versus weekly capacity.** At a stationary $0.5\%$ $30$-day replacement rate, a
given week's snapshot is expected to carry $1{,}000{,}000\times0.005=5{,}000$ positive labels — machines
that will need a replacement within the next $30$ days. Operations can service only $1{,}000$ of them a
week. Since a ranked list can put at most $1{,}000$ machines into that capacity no matter how good the
ranking, recall@$1{,}000$ — the fraction of the week's $5{,}000$ true positives actually caught — is bounded
above by $1{,}000/5{,}000=20\%$; a model that reaches that ceiling has filled the entire weekly capacity
with true positives (precision@$1{,}000=100\%$), and no ranking can do better than the ceiling at this
capacity. With the cutoff and the week's positives both fixed, recall@$1{,}000$ is just
precision@$1{,}000\times1{,}000/5{,}000$, so the two carry the same information about a week's ranking, and
precision@$1{,}000$ is the one reported (Evaluation, below). The ceiling is not the share of failures the
design lets happen: a $30$-day window spans about $30/7\approx4.3$ weekly snapshots, so consecutive
snapshots share most of their positives, and new replacements arrive at only about
$5{,}000\times7/30\approx1{,}170$ a week — a flow that $1{,}000$ pre-emptive services a week can mostly
cover, provided the ranking flags each machine in one of the weeks before it fails.

### Labels and training data

**Label.** For a machine observed at snapshot time $t$, the label is $1$ if a repair ticket records a
**hardware** component replaced in $(t, t+30\text{ days}]$, and $0$ otherwise; a ticket recording a
software-only fix (a firmware flash, a reseated cable with no part replaced) does not count. A **planned
refresh** — a replacement scheduled as part of the fleet's normal hardware-refresh cycle, independent of any
observed degradation — is also excluded: not just from the positive label, but from the negative pool too
for a machine inside its scheduled refresh window, since neither answer to "did it get a hardware
replacement" speaks to the question this system exists to answer, a machine's own failure risk. A planned
refresh is handled by the ordinary refresh process, not by this ranking.

**Snapshots.** Every machine gets one row a week, at a fixed weekly snapshot time $t$; a feature used in
that row is computed only from data available at or before $t$ — never from anything logged afterward,
including the very ticket the label is trying to predict. Three ways this rule is easy to break by
accident:

- **Diagnostics triggered by the failure itself.** A repair ticket often comes with a diagnostic flag or an
  error code the technician recorded *while* investigating the fault; that flag is a downstream product of
  the same event the label marks, not a precursor to it, and using it as a feature lets the model see the
  answer written into its own input.
- **A "days since last ticket" feature computed with future tickets.** This feature must count only tickets
  strictly before $t$; computing it once over a machine's whole history and reusing the same value for every
  one of its snapshots silently pulls in tickets that happen after $t$, including the one being predicted.
- **Fleet-level statistics computed over the whole period.** A feature such as "this machine type's average
  error rate this quarter" is safe only if "this quarter" means data up to $t$; computed once over the full
  data pull, which extends past $t$ for most snapshots, it leaks the outcome of weeks the model has not
  reached yet into every earlier row of the same machine type.

**Censoring.** A machine can be decommissioned for a reason unrelated to the failure being predicted (a
relocation, an unrelated planned refresh) before $t+30$ days, or the data pull can simply end before
$t+30$ days is reached; either way, that snapshot's true label is unknown, not negative — the machine may or
may not have needed a replacement had it stayed in the fleet. Silently treating an unresolved snapshot as a
negative example understates the true replacement rate, since the rows being mislabelled are
disproportionately the ones nearing the end of their observed life. Dropping these snapshots from a plain
classification training set instead biases the rate the other way: a machine that fails early in its window
resolves that snapshot as a positive even if it would have left the data later in the window, while one that
survives until it leaves has its snapshot dropped, so the kept rows over-represent failures (the estimate
check below measures both biases against the true rate of a synthetic fleet). A survival formulation
(Models and losses, below) avoids both by using a censored machine correctly — contributing exactly what it
actually tells us, that it survived up to the point it left the data, and nothing it does not.

**Splitting.** Two snapshots of the same machine one week apart look at almost the same telemetry and, if a
failure is on the way, are usually the same handful of weeks away from the same ticket, so their labels are
highly correlated. A random row split lets one land in training and the other in a test set, and a model
expressive enough to key off the machine itself — rather than the pattern in its telemetry — reports a
validation score with little bearing on how it performs on a machine it has genuinely never seen. The fix is
to split by machine and by time together: train on earlier calendar months, test on later ones, using
machines that appear on only one side of the split — never a random split by row. The estimate check below
fits the same simple model both ways and shows the gap between the two.

### Features

**Feature families.** Windowed aggregates of the raw telemetry — mean, max and slope over trailing $1$,
$7$ and $30$-day windows — turn a per-minute stream into a fixed-size weekly feature vector; counts and
rates of corrected memory errors and similar event counters are log-transformed, since they are heavy-tailed
(most machines log a handful, a few log thousands); deltas capture short-term rate of change on top of the
windowed level; static attributes (machine type, age, data centre, rack position, firmware version) change
rarely and are simply looked up as of $t$; fleet-relative features compare a machine's own reading against
other machines of the *same type* — a $z$-score against that type's own mean and standard deviation, not the
whole fleet's — since machine types run at genuinely different baseline temperatures and error rates, and
"$4$°C hotter than this type usually runs" is informative in a way "$4$°C above the fleet average" is not;
repair history (ticket count, time since the last ticket, computed only from tickets before $t$) captures
that a machine already in worse shape tends to stay that way.

**Transforms differ by model family.** A linear or logistic model needs its inputs scaled — else the fitted
weights, and any regularisation penalty, are dominated by whichever feature happens to have the largest raw
units — the skewed counts log-transformed before scaling, categorical attributes one-hot or target-encoded,
and any interaction the model should use written in explicitly, since a linear model cannot discover one on
its own. A tree ensemble needs none of this: a split threshold is invariant to any monotone transform of the
feature it splits on, so a log transform changes nothing a tree can learn; missing values are routed to
whichever branch the training data prefers, with no imputation step; and a categorical attribute can be used
as a plain integer or native category with no one-hot expansion. A neural network needs normalised inputs
for stable gradient-based training and, for a high-cardinality categorical, a learned embedding rather than
one-hot, which would be both extremely wide and unable to share statistical strength between similar
categories.

**Embeddings for machine type and other categoricals.** Data centre ($20$ values) is low-cardinality enough
for a plain one-hot encoding; machine type (about $300$ values, growing every quarter) and firmware version
are not, and get a learned embedding table in the neural network, trained jointly with the rest of the
model. A type introduced this quarter has no rows to learn its own embedding from yet, so its embedding
cannot be a bare lookup by ID; the fix is to build it from the type's own descriptive attributes instead —
vendor, CPU generation, disk model, memory configuration — through a small sub-network shared across all
types, so a brand-new type gets a reasonable embedding the day it is introduced, from attributes known
immediately, rather than waiting on failure history the model was never going to have in time. A coarser
fallback — one shared out-of-vocabulary row, or hashing the type identifier into a fixed-size table — costs
less to build but throws away exactly the attributes that make a new type's embedding meaningful before it
has its own history; falling back to the parent hardware family's embedding is a reasonable middle ground
when the full attribute set is not yet in the catalogue. Whichever mechanism is used, it is exercised at
serving time as soon as a type is introduced, since the fleet starts reporting telemetry for it immediately —
well before $52$ weeks of its own snapshots exist.

### Models and losses

Three models, in increasing order of how directly they use time:

**(i) Logistic regression** predicts $p(x)=\sigma(w\cdot x+b)$, fit by weighted binary cross-entropy,

$$\mathcal{L}=-\frac{1}{N}\sum_i \big[c_1\, y_i\log p_i + c_0\,(1-y_i)\log(1-p_i)\big],$$

with $c_1>c_0$ upweighting the rare positive class (Imbalance, below). Fast to train and to serve, and its
coefficients are directly readable, which matters for explaining a specific machine's score to a technician
(Deployment, below); its accuracy is capped by the features actually being linear, after the transforms
above, in the log-odds of replacement.

**(ii) Gradient-boosted trees**, fit stage-wise: at stage $m$, the current model $F_{m-1}$ has log loss
$-[y\log\sigma(F)+(1-y)\log(1-\sigma(F))]$, whose negative gradient with respect to $F$ is
$y-\sigma(F_{m-1}(x))$ — the observed label minus the currently predicted probability — so the next tree
$h_m$ is fit to approximate this residual and the model is updated $F_m=F_{m-1}+\eta\,h_m$. Because a tree
ensemble already handles the transforms above natively (Features), this is usually the strongest single
tabular model here without extra feature engineering, at the cost of a score that takes an explicit
attribution step (Deployment) rather than a coefficient to read directly.

**(iii) A discrete-time hazard model** treats each week a machine is in the fleet as one trial:
$h_k(x)=\sigma(w\cdot x_k+b)$ is the probability it needs replacing in week $k$, given it has needed none
through week $k-1$. Over the weeks a machine is actually observed, the likelihood is exactly a logistic
regression's, fit on one row per machine-week at risk rather than one row per machine:

$$\text{NLL}=-\sum_i\sum_{k=1}^{K_i}\big[y_{i,k}\log h_k(x_{i,k})+(1-y_{i,k})\log(1-h_k(x_{i,k}))\big],$$

where $K_i$ is the last week machine $i$ is observed and $y_{i,k}=1$ only at $k=K_i$ if that week is a
genuine replacement — so a censored machine (Labels, above) contributes only the survival terms
$\log(1-h_k)$ for every week it was actually observed, and never a fabricated event or non-event at the
point it left the data. That is exactly the information a censored row does carry, and the only information
it carries. Since the hazards are conditional probabilities — "replaced in week $k$, given it survived to
week $k$" — the chance of surviving four consecutive weeks is the product of surviving each one,
$\prod_{k=1}^4(1-h_k)$ — four weekly snapshots is this design's discretisation of the $30$-day horizon used
everywhere else on this page — so the risk this design ranks on is

$$P(\text{replaced within 30 days})=1-\prod_{k=1}^{4}(1-h_k).$$

An optional sequence encoder over the raw weekly telemetry window, in place of the hand-aggregated $x_k$,
can let this model learn its own temporal features, at the cost of needing more data and compute than the
hand-engineered version.

All three train and evaluate on the same weekly snapshot rows; (iii) additionally needs the expanded
per-machine-week table the estimate check builds below, and is the only one of the three that uses a
censored machine's partial history rather than discarding or mislabelling it.

### Imbalance

**Class weights.** Setting $c_1/c_0$ in the loss above — for instance inversely proportional to each class's
frequency — makes the optimiser pay as much total attention to the rare positive class as to the abundant
negative one; every row is still used, but training compute is unchanged, since most of it is still spent on
the $99.5\%$ of rows carrying a negative label.

**Down-sampling, and correcting for it.** Keeping every positive row but only a random keep-rate $r<1$ of the
negative rows cuts training compute to roughly a fraction $r$ of the original, but changes what the fitted
model estimates. Write $p(x)$ for the true probability of replacement and $p'(x)$ for what a model trained on
the down-sampled data estimates. A positive row is always kept and a negative row only with probability $r$,
so by Bayes' rule $p'(x)=P(y=1\mid x,\text{kept})=p(x)/\big(p(x)+r\,(1-p(x))\big)$: the *odds* of the
label the model actually sees are the true odds divided by $r$:

$$\frac{p'(x)}{1-p'(x)}=\frac{1}{r}\cdot\frac{p(x)}{1-p(x)}.$$

Solving for $p(x)$ gives the correction applied to every prediction at serving time:

$$p(x)=\frac{r\,p'(x)}{r\,p'(x)+1-p'(x)}.$$

The estimate check below fits a logistic model on down-sampled data and confirms both halves of this: the
uncorrected $p'$ is inflated well above the true rate, and applying the formula restores it to within a
fraction of a percentage point.

**Focal loss** is an alternative to either: it reweights the *loss*, not the rows, using
$-(1-p_t)^\gamma\log(p_t)$ where $p_t$ is the model's predicted probability of the true class, so an
already-easy, correctly-scored negative ($1-p_t$ small, hence $(1-p_t)^\gamma$ smaller still) contributes
almost nothing to the gradient, concentrating training on the rare positives and on whichever negatives the
model is currently getting wrong. It uses every row, unlike down-sampling, and needs no probability
correction of down-sampling's specific kind, but introduces its own hyperparameter $\gamma$ to tune and,
like any reweighting of the loss, still needs its own calibration check before its output is trusted as a
probability rather than just a ranking score.

### Evaluation

**The decision the metrics must match.** Operations acts on the top $1{,}000$ machines a week, so the two
numbers that matter operationally are **precision@$1{,}000$** (of the $1{,}000$ serviced, what fraction
actually needed it) and **recall@$1{,}000$** (of the week's roughly $5{,}000$ true positives, what fraction
were caught) — and, as derived in Requirements and scale, recall@$1{,}000$ tops out at $20\%$ regardless of
model quality and within a week is just precision@$1{,}000\times1{,}000/5{,}000$, so precision@$1{,}000$ —
the share of service calls that were actually needed, on the full $0$–$100\%$ scale — is the headline number.
**PR-AUC** (the area under precision versus recall as the cutoff sweeps over every possible rank, not just
$1{,}000$) is tracked alongside it as a threshold-free summary for comparing models and for
regression-testing a retrain before it ever reaches the top-$1{,}000$ cutoff.

**Why not just ROC-AUC.** ROC-AUC is the probability a random true positive is scored above a random true
negative, and its $x$-axis, the false-positive rate, is a fraction of the *negative* population — almost the
entire fleet, at $0.5\%$ prevalence. A model can misrank most of the top of its list — filling $1{,}000$
slots with only a couple of hundred genuine positives — and still show a tiny false-positive rate, because
those several hundred false positives are a minuscule share of the roughly $995{,}000$ machines that were never
going to need replacing. The estimate check below builds exactly this case: an ROC-AUC of about $0.82$
alongside a precision@k under $20\%$, with the false-positive rate at that same cutoff under a tenth of a
percent. ROC-AUC is not wrong here — it is answering a different question (ranking quality across the whole
population) from the one operations is asking (how many of a fixed $1{,}000$ picks were worth it).

**Calibration.** Precision and recall depend only on the *ordering* the model produces, not the absolute
probabilities; but the cost calculation below, and the down-sampling correction above, both consume the
probabilities themselves, so a reliability check (predicted probability against observed replacement rate in
each score bucket) belongs alongside the ranking metrics whenever a probability, not just a rank, feeds a
downstream decision.

**Expected cost saved.** With a pre-emptive replacement costing $1$ unit and an unplanned failure costing the
stated $20\times$ that, servicing the top $1{,}000$ each week costs $1{,}000$ regardless of who among them
truly needed it, and each true positive among them is a failure that no longer happens unplanned, avoiding
$20$ units. Against not acting on the week's list at all, the weekly saving is

$$\text{saved}=20\times\text{TP}-1{,}000,$$

where TP is the number of true positives among the top $1{,}000$. A perfectly precise week ($\text{TP}=1{,}000$)
saves $19{,}000$ units; a ranking no better than picking machines at random
($\text{TP}\approx1{,}000\times5{,}000/1{,}000{,}000=5$) *loses* about $900$ units against doing nothing at
all, since capacity is spent almost entirely on machines that did not need it while the real failures still
happen unplanned. Precision@$1{,}000$, not the mere existence of a model, is what turns this program from a
net cost into a net saving.

**Backtesting.** A single train/test split understates how performance drifts over calendar time, so the
design backtests on a rolling origin: train on the snapshots whose $30$-day label windows have closed by
week $W$ — exactly the labels production would have at $W$ — evaluate on the weeks just after $W$, then
advance $W$ and repeat, reporting the distribution of precision@$1{,}000$, recall@$1{,}000$ and PR-AUC across
folds rather than one point estimate — the same machine-and-time discipline as the training split (Labels,
above), extended to give a read on trend rather than a single snapshot.

### Deployment and feedback

```text
Telemetry, repair tickets, decommission records
                |
                v
   Weekly snapshot + label construction     (Labels and training data)
                |
                v
   Train: logistic regression / gradient boosting / hazard model     (Models and losses)
                |
                v
   Weekly batch scoring of the whole fleet
                |
                v
   Ranked list + top contributing features per machine
                |
                v
   Ops: pre-empt top 1,000 (weekly capacity)  ----->  a small random holdout: never pre-empted
                |                                              |
                v                                              v
   outcome: replaced before it could fail          outcome: fails, or doesn't, on its own
                |                                              |
                '--------------------> feeds back as new repair tickets / inspection results, above
```

**Serving and threshold.** Every week, the current model scores all $1{,}000{,}000$ machines and produces a
ranked list; the operative threshold is *rank*, not a fixed probability cutoff, since the constraint the
design is built around is a fixed count — $1{,}000$ machines a week — not a fixed confidence level, and a
probability cutoff would hand operations a different number of candidates every week depending on how bad
that week happens to be. A probability floor underneath the rank cutoff — skip a top-$1{,}000$ machine whose
own expected cost saved (Evaluation, above) has turned negative — guards against a quiet week where even the
$1{,}000$th-ranked machine is not actually worth a service call: servicing a machine with calibrated $30$-day
risk $p$ saves $20p-1$ units in expectation, so the floor sits at $p=1/20=5\%$ and binds only when the
$1{,}000$th-ranked machine's risk falls below it. Each machine on the list ships with its top
contributing features — a coefficient-times-value breakdown for logistic regression, a comparable per-feature
attribution for the tree ensemble — so a technician has a stated reason to check, not just a bare score.

**Monitoring for drift.** A new machine type entering the fleet each quarter needs its content-based
embedding (Features, above) to actually be exercised, not silently defaulted; a firmware update pushed
across a whole cohort of machines shifts their absolute telemetry overnight, in a way the fleet-relative
$z$-score features are designed to absorb only if the reference distribution they are computed against is
itself kept current; and ordinary seasonal temperature swings shift absolute thermal readings across the
whole fleet at once, which the same fleet-relative design choice also protects against, provided the
reference window rolls forward with the season rather than freezing at training time. Because the outcome
label lags $30$ days behind the snapshot, precision@$1{,}000$ and PR-AUC from the rolling backtest are a
trailing signal; tracking each feature's own distribution week to week catches a shift before it shows up in
outcomes.

**The feedback loop.** A machine the model flags and operations replaces pre-emptively never gets the chance
to fail on its own, so the one label that would confirm or refute the prediction — would it actually have
failed? — is never produced; every machine the model successfully acts on becomes permanently unobservable
to its own future training data, a form of censoring the *model's own decisions* create, not a
data-collection accident. Left unaddressed, the positive examples remaining in the log skew toward the
failures the model missed and the ones the ranking never reached, understating how good the model already is
and quietly retraining it against an increasingly unrepresentative sample of what it is actually being asked
to predict. Two mitigations, used together: a small, fixed random slice of the fleet is *never* pre-empted
regardless of what the model says, so its true outcomes are always eventually observed and give an unbiased
read on precision and recall, at the cost of accepting a few genuine unplanned failures on that slice as the
price of an honest evaluation signal; and where a machine is serviced pre-emptively, the removed component is
inspected on the bench, substituting a direct measurement of whether it was actually failing for the field
outcome the design deliberately chose not to observe.

### Follow-ups

- **Component-level models instead of one machine-level model.** Predicting separately for disk, memory and
  fan — each with its own label (a ticket that replaced that specific component) and its own feature subset —
  gives operations a reason that is already a specific action (which part to bring, which test to run first)
  rather than a single "this machine is at risk" score it must still diagnose; the cost is $k$ separate
  models to maintain and evaluate instead of one, and a fleet-level ranking now needs a rule for combining
  $k$ separate risk scores into the one list operations actually works from.
- **Online updating.** Retraining weekly on a rolling window keeps the model current with firmware changes
  and new machine types without waiting for a full offline cycle, at the cost of a noisier week-to-week model
  that needs the same rolling backtest (Evaluation, above) to confirm it has not regressed before it is
  trusted with the next week's list.
- **Shared inspection capacity.** When the $1{,}000$-machine weekly quota is itself shared with other
  maintenance work (a firmware rollout, a scheduled audit), the ranking this design produces becomes one
  input to a further capacity allocation rather than the allocation itself — the expected-cost-saved figure
  (Evaluation, above) is exactly the number that lets that allocation be made on a common basis with the
  other work competing for the same technicians.
- **Explaining predictions to technicians.** A bare score invites the same follow-up question every time —
  why this machine — so the per-feature attribution already sent with the ranked list (Deployment, above)
  should be phrased in the units a technician checks against (a temperature in degrees, an error count, not a
  raw feature weight), and tracked over time: a machine whose top reason keeps changing week to week is a
  sign the model's explanation, not just its score, deserves a second look.

<details>
<summary>Estimate check (runnable)</summary>

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

# ---- requirements and scale ----
n_machines = 1_000_000
n_metrics = 50
minutes_per_day = 24 * 60
readings_per_day = n_machines * n_metrics * minutes_per_day
assert readings_per_day == 72_000_000_000

bytes_per_reading = 8
gb_per_day = readings_per_day * bytes_per_reading / 1e9
assert gb_per_day == 576.0

weeks_per_year = 52
snapshot_rows_per_year = n_machines * weeks_per_year
assert snapshot_rows_per_year == 52_000_000

monthly_replace_rate = 0.005
expected_positives_per_week = n_machines * monthly_replace_rate
assert expected_positives_per_week == 5_000

negative_label_fraction = 1 - monthly_replace_rate
assert negative_label_fraction == 0.995

never_needs_replacing = n_machines - expected_positives_per_week
assert never_needs_replacing == 995_000

weekly_capacity = 1_000
recall_ceiling = weekly_capacity / expected_positives_per_week
assert recall_ceiling == 0.2

snapshots_per_window = 30 / 7                  # a 30-day window spans this many weekly snapshots, so
assert round(snapshots_per_window, 1) == 4.3    # consecutive snapshots share most of their positives
new_replacements_per_week = expected_positives_per_week / snapshots_per_window
assert round(new_replacements_per_week, -1) == 1_170

print("all requirements-and-scale numbers check out")

# ---- expected cost saved, at the stated 20:1 cost ratio ----
c_pre, c_fail = 1.0, 20.0
K, P = weekly_capacity, expected_positives_per_week

break_even_risk = c_pre / c_fail   # servicing a machine of 30-day risk p saves c_fail * p - c_pre in expectation
assert break_even_risk == 0.05


def cost_saved(true_positives):
    baseline = P * c_fail                                    # do nothing: every positive fails unplanned
    policy_cost = K * c_pre + (P - true_positives) * c_fail  # service K regardless; miss the rest
    return baseline - policy_cost


saved_perfect = cost_saved(K)                # every one of the 1,000 picks is a true positive
assert saved_perfect == 19_000

tp_random = K * (P / n_machines)             # a ranking no better than picking at random
assert tp_random == 5.0
saved_random = cost_saved(tp_random)
assert saved_random == -900.0

print("expected-cost-saved figures check out (perfect precision saves 19,000; a random pick loses 900; "
      "break-even risk 5%)")


# ---- precision@k / recall@k: sorting versus argpartition, and the recall = precision * (K/P) identity ----
def topk_by_sort(scores, k):
    return np.argsort(-scores)[:k]


def topk_by_partition(scores, k):
    return np.argpartition(-scores, k - 1)[:k]  # NOTE: only the SET of the k largest is guaranteed, not their order


rng = np.random.default_rng(1)
for _ in range(200):
    n = int(rng.integers(50, 500))
    k = int(rng.integers(1, n))
    y_true = rng.binomial(1, 0.1, size=n)
    scores = rng.normal(size=n)          # continuous scores: no ties, so the top-k SET is unambiguous
    set_sort = set(topk_by_sort(scores, k).tolist())
    set_part = set(topk_by_partition(scores, k).tolist())
    assert set_sort == set_part

print("precision/recall @k: sort- and argpartition-based top-k sets agree on 200 random trials")

n_toy, P_toy, K_toy = 50_000, 500, 100
y_true = np.zeros(n_toy, dtype=int)
pos_positions = rng.choice(n_toy, size=P_toy, replace=False)
y_true[pos_positions] = 1
non_pos = np.setdiff1d(np.arange(n_toy), pos_positions, assume_unique=True)
for true_positives in (100, 70, 20, 1):
    scores = rng.normal(size=n_toy)
    chosen_pos = rng.choice(pos_positions, size=true_positives, replace=False)
    chosen_neg = rng.choice(non_pos, size=K_toy - true_positives, replace=False)
    scores[chosen_pos] = 100 + rng.random(true_positives)
    scores[chosen_neg] = 100 + rng.random(K_toy - true_positives)
    idx = topk_by_sort(scores, K_toy)
    precision = y_true[idx].sum() / K_toy
    recall = y_true[idx].sum() / P_toy
    assert y_true[idx].sum() == true_positives
    assert abs(recall - precision * (K_toy / P_toy)) < 1e-12

print("recall@K == precision@K * (K / P) confirmed whenever K and P are fixed")


# ---- down-sampling negatives inflates predicted probability; the odds-derived correction restores it ----
rng = np.random.default_rng(2)
n_total, n_features = 200_000, 4
true_w, true_b = np.array([1.2, -0.8, 0.5, 0.3]), -5.0

X = rng.normal(size=(n_total, n_features))
p_true = 1.0 / (1.0 + np.exp(-(X @ true_w + true_b)))
y = rng.binomial(1, p_true)

n_train = 150_000
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]          # held out, never down-sampled: it must reflect the real world

lr_full = LogisticRegression(C=1e6, max_iter=1000).fit(X_train, y_train)

r = 0.1  # keep-rate for negatives
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]
keep_neg = neg_idx[rng.random(len(neg_idx)) < r]
ds_idx = np.concatenate([pos_idx, keep_neg])
lr_ds = LogisticRegression(C=1e6, max_iter=1000).fit(X_train[ds_idx], y_train[ds_idx])

p_prime = lr_ds.predict_proba(X_test)[:, 1]                        # uncorrected, on the untouched test set
p_corrected = r * p_prime / (r * p_prime + (1 - p_prime))          # p = r p' / (r p' + 1 - p'), derived above
true_rate = y_test.mean()

inflation = p_prime.mean() - true_rate
residual = abs(p_corrected.mean() - true_rate)
print(f"true rate {true_rate:.4f}, mean uncorrected p' {p_prime.mean():.4f}, mean corrected {p_corrected.mean():.4f}")
assert inflation > 0.05          # clearly inflated
assert residual < 0.01           # correction restores it to within a fraction of a percentage point

# the intercept shift the derivation predicts: log(1/r), slope unchanged
assert np.allclose(lr_ds.coef_, lr_full.coef_, atol=0.05)
assert abs((lr_ds.intercept_ - lr_full.intercept_)[0] - np.log(1 / r)) < 0.05
print("down-sampling correction confirmed: inflated probability restored to within tolerance of the true rate")


# ---- ROC-AUC well above 0.5 can coexist with low precision@k, at low prevalence ----
rng = np.random.default_rng(3)
n_ranking = 200_000
prevalence = 0.005
n_pos = int(n_ranking * prevalence)
y_rank = np.zeros(n_ranking, dtype=int)
y_rank[:n_pos] = 1
rng.shuffle(y_rank)

signal = rng.normal(size=n_ranking)
score = np.where(y_rank == 1, signal + 2.0, signal) + rng.normal(scale=1.2, size=n_ranking)


def auc_from_ranks(y_true, s):
    """AUC = P(random positive scores above random negative), via the Mann-Whitney rank-sum identity --
    an independent definition-level computation, not a call into sklearn's implementation."""
    ranks = stats.rankdata(s)
    n_p, n_n = y_true.sum(), len(y_true) - y_true.sum()
    return (ranks[y_true == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)


auc_hand = auc_from_ranks(y_rank, score)
auc_lib = roc_auc_score(y_rank, score)
assert abs(auc_hand - auc_lib) < 1e-9
assert round(auc_hand, 2) == 0.82

k_rank = 200  # 0.1% of n_ranking, the same ratio as 1,000 of 1,000,000 machines
idx = np.argsort(-score)[:k_rank]
precision_k = y_rank[idx].sum() / k_rank
fpr_k = (k_rank - y_rank[idx].sum()) / (n_ranking - n_pos)
print(f"ROC-AUC {auc_hand:.3f}, precision@{k_rank} {precision_k:.3f}, "
      f"false-positive rate at the same cutoff {fpr_k:.4f}")
assert precision_k < 0.2
assert fpr_k < 0.001         # the same false positives are a tiny share of the (huge) negative population
print("ROC-AUC-versus-precision@k dichotomy at low prevalence confirmed")


# ---- a small seeded synthetic fleet: hidden degradation before failure, some unrelated decommissions ----
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate_fleet(seed, n, n_weeks, w0, w1, base_sd, slope_hi, admin_censor_p):
    rng_f = np.random.default_rng(seed)
    base = rng_f.normal(0, base_sd, size=n)
    slope = rng_f.uniform(0.0, slope_hi, size=n)
    outcome_week = np.full(n, n_weeks)
    outcome_type = np.array(["alive"] * n, dtype=object)      # alive (to end) | failed | censored
    for i in range(n):
        for k in range(1, n_weeks + 1):
            x = base[i] + slope[i] * k
            if rng_f.random() < sigmoid(w1 * x + w0):
                outcome_week[i], outcome_type[i] = k, "failed"
                break
            if rng_f.random() < admin_censor_p:
                outcome_week[i], outcome_type[i] = k, "censored"
                break
    return base, slope, outcome_week, outcome_type


SEED, N_FLEET, N_WEEKS, HORIZON = 7, 2_500, 30, 4
W0_TRUE, W1_TRUE = -4.2, 1.0
base, slope, outcome_week, outcome_type = simulate_fleet(
    SEED, N_FLEET, N_WEEKS, W0_TRUE, W1_TRUE, base_sd=0.15, slope_hi=0.3, admin_censor_p=0.025)

n_failed = int((outcome_type == "failed").sum())
n_censored = int((outcome_type == "censored").sum())
assert n_failed > 300 and n_censored > 100

# -- expanded machine-week-at-risk table: the discrete-time hazard model IS logistic regression on this table --
exp_i, exp_x, exp_event = [], [], []
for i in range(N_FLEET):
    w = outcome_week[i]
    for t in range(1, w + 1):
        exp_i.append(i)
        exp_x.append(base[i] + slope[i] * t)
        exp_event.append(1 if (t == w and outcome_type[i] == "failed") else 0)
exp_i = np.array(exp_i)
exp_x = np.array(exp_x).reshape(-1, 1)
exp_event = np.array(exp_event)

never_failed = np.where(outcome_type != "failed")[0]        # censored + administratively-ended machines
assert exp_event[np.isin(exp_i, never_failed)].sum() == 0    # they contribute ONLY survival terms, never an event

lr_hazard = LogisticRegression(C=1e6, max_iter=2000).fit(exp_x, exp_event)
w_hat, b_hat = lr_hazard.coef_[0, 0], lr_hazard.intercept_[0]
assert abs(w_hat - W1_TRUE) < 0.1 and abs(b_hat - W0_TRUE) < 0.2   # true hazard recovered, censored machines in

# the hazard NLL written per machine from its definition -- survive weeks 1..K_i-1, then fail or survive week
# K_i -- without the expanded table, equals the logistic log loss on the expanded table
nll_machine = 0.0
for i in range(N_FLEET):
    h = sigmoid(w_hat * (base[i] + slope[i] * np.arange(1, outcome_week[i] + 1)) + b_hat)
    nll_machine -= np.log(1 - h[:-1]).sum()
    nll_machine -= np.log(h[-1]) if outcome_type[i] == "failed" else np.log(1 - h[-1])
nll_table = log_loss(exp_event, lr_hazard.predict_proba(exp_x)[:, 1], normalize=False)
assert abs(nll_machine - nll_table) < 1e-6 * nll_table
print(f"fitted hazard slope {w_hat:.3f} (true {W1_TRUE}), intercept {b_hat:.3f} (true {W0_TRUE}); "
      f"per-machine NLL {nll_machine:.2f} matches the expanded table's log loss {nll_table:.2f}")

sample_x = np.array([[0.3], [0.5], [0.7], [0.9]])          # one machine's next 4 weekly feature values
h4 = lr_hazard.predict_proba(sample_x)[:, 1]
closed_form_risk = 1 - np.prod(1 - h4)
rng_mc = np.random.default_rng(99)
n_mc = 300_000
mc_risk = (rng_mc.random((n_mc, 4)) < h4[None, :]).any(axis=1).mean()
se = np.sqrt(mc_risk * (1 - mc_risk) / n_mc)
print(f"closed-form 30-day risk {closed_form_risk:.4f} vs. Monte Carlo {mc_risk:.4f} (s.e. {se:.4f})")
assert abs(mc_risk - closed_form_risk) < 6 * se
print("discrete-time hazard NLL and the 30-day risk product formula both confirmed")

# -- snapshot / classification table, with censoring-aware labels (drop what is genuinely unresolved) --
snap_i, snap_t, snap_x, snap_label = [], [], [], []
n_candidates = 0
for i in range(N_FLEET):
    w, typ = outcome_week[i], outcome_type[i]
    for t in range(1, w):
        window_end = t + HORIZON
        n_candidates += 1
        is_event_in_window = typ == "failed" and w <= window_end
        if is_event_in_window:
            snap_i.append(i); snap_t.append(t); snap_x.append(base[i] + slope[i] * t); snap_label.append(1)
        elif window_end <= w:
            snap_i.append(i); snap_t.append(t); snap_x.append(base[i] + slope[i] * t); snap_label.append(0)
        # else: the machine is censored or administratively ended strictly inside the window -- unresolved, dropped

snap_i, snap_t = np.array(snap_i), np.array(snap_t)
snap_x, snap_label = np.array(snap_x), np.array(snap_label)
print(f"kept {len(snap_label)} of {n_candidates} candidate snapshots")


# The two label-rate biases are about one percentage point each, comparable to one small fleet's sampling noise,
# so they are measured on a fleet eight times larger (same generator, another seed).
def label_rates(seed, n):
    b_, s_, w_, typ_ = simulate_fleet(seed, n, N_WEEKS, W0_TRUE, W1_TRUE, base_sd=0.15, slope_hi=0.3,
                                      admin_censor_p=0.025)
    naive_pos, kept_pos, kept, total, true_sum = 0, 0, 0, 0, 0.0
    for i in range(n):
        for t in range(1, w_[i]):
            window_end = t + HORIZON
            ahead = np.arange(t + 1, window_end + 1)   # true P(fail in the window | alive at t), uncensored
            true_sum += 1 - np.prod(1 - sigmoid(W1_TRUE * (b_[i] + s_[i] * ahead) + W0_TRUE))
            event = typ_[i] == "failed" and w_[i] <= window_end
            total += 1
            naive_pos += event
            if event or window_end <= w_[i]:
                kept += 1
                kept_pos += event
    return naive_pos / total, kept_pos / kept, true_sum / total   # unresolved as negative, dropped, true


naive_rate, dropped_rate, true_rate = label_rates(seed=8, n=8 * N_FLEET)
print(f"true rate {true_rate:.4f}, unresolved-as-negative {naive_rate:.4f}, unresolved dropped {dropped_rate:.4f}")
assert naive_rate < true_rate - 0.004    # counting unresolved rows as negatives understates the rate...
assert dropped_rate > true_rate + 0.002  # ... and dropping them overstates it: an early failure resolves its row,
                                         # while a machine that survives until it leaves the data is dropped

# -- random row split vs. split by machine AND time, for the SAME (deliberately simple, memorization-capable) model --
ID_SCALE = 1_000.0  # dominates Euclidean distance unless machine_id matches exactly
features = np.column_stack([snap_i * ID_SCALE, snap_x])
n_snap = len(snap_label)
rng_split = np.random.default_rng(11)

perm = rng_split.permutation(n_snap)
cut = int(0.7 * n_snap)
train_idx, test_idx = perm[:cut], perm[cut:]
knn_random = KNeighborsClassifier(n_neighbors=1).fit(features[train_idx], snap_label[train_idx])
auc_random = roc_auc_score(snap_label[test_idx], knn_random.predict_proba(features[test_idx])[:, 1])
nn = knn_random.kneighbors(features[test_idx], n_neighbors=1, return_distance=False).ravel()
same_machine_random = (snap_i[train_idx][nn] == snap_i[test_idx]).mean()

test_machines = rng_split.choice(N_FLEET, size=int(0.3 * N_FLEET), replace=False)
is_test_machine = np.isin(snap_i, test_machines)
time_cutoff = 18
train_mask = (~is_test_machine) & (snap_t < time_cutoff)
test_mask = is_test_machine & (snap_t >= time_cutoff)
knn_grouped = KNeighborsClassifier(n_neighbors=1).fit(features[train_mask], snap_label[train_mask])
auc_grouped = roc_auc_score(snap_label[test_mask], knn_grouped.predict_proba(features[test_mask])[:, 1])
nn2 = knn_grouped.kneighbors(features[test_mask], n_neighbors=1, return_distance=False).ravel()
same_machine_grouped = (snap_i[train_mask][nn2] == snap_i[test_mask]).mean()

print(f"random-row-split AUC {auc_random:.3f} (nearest neighbour is the same machine "
      f"{same_machine_random:.1%} of the time) vs. machine-and-time split AUC {auc_grouped:.3f} "
      f"({same_machine_grouped:.0%} of the time)")
assert same_machine_random > 0.95
assert same_machine_grouped == 0.0
assert auc_random - auc_grouped > 0.2

print("all checks passed")
```

</details>

</details>
