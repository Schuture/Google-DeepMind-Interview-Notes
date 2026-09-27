# Training a Model That Does Not Fit on One Accelerator

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| ML system design | ★★★☆☆ | Hard | RE · MLE · RS | data-parallelism, tensor-parallelism, pipeline-parallelism, sharded-optimiser, checkpointing, fault-tolerance, loss-spikes | 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

An *accelerator* is a single compute device with its own high-bandwidth memory — a GPU or TPU chip.
Consider training a decoder-only transformer with 80 transformer layers, model width (hidden size)
8,192 and sequence length 4,096 — 70 billion trainable parameters in total — from a random
initialisation through one pass over a training corpus of $1.4 \times 10^{12}$ tokens. Training
minimises a per-token next-token-prediction loss with the Adam optimiser, which keeps, for every
parameter, a running first moment (an exponential moving average of the gradient) and second moment
(an exponential moving average of the squared gradient) alongside the parameter itself.

The model's parameters, gradients and optimiser state do not fit in one accelerator's memory, and
training it in a reasonable amount of time needs many accelerators computing at once. Scale this
design for:

- 2,048 accelerators, each with 96 GB of high-bandwidth memory and a peak throughput of 400 TFLOP/s
  ($4 \times 10^{14}$ FLOP/s) in bf16 (a 16-bit floating-point format with an 8-bit exponent, the
  standard reduced-precision format for training).
- 8 accelerators per host, joined by a fast intra-host interconnect at 600 GB/s per accelerator;
  hosts are joined to each other, and to storage, by a slower network at 50 GB/s per accelerator.
- Mixed-precision training: the weights and gradients used in the forward and backward pass are held
  in bf16 (2 bytes each), while Adam keeps a master copy of the weights and both moments in fp32
  (4 bytes each) for numerically stable updates.
- A global batch of 4,000,000 tokens per optimiser step, accumulated across every accelerator before
  each update.
- A checkpoint store with an aggregate write bandwidth of 20 GB/s across the whole cluster.
- Each accelerator fails independently, with the time between one accelerator's failures
  exponentially distributed with a mean of 50,000 hours (its mean time between failures, MTBF).
- A target model FLOPs utilisation (MFU) of 40% — the fraction of the cluster's peak FLOP/s that the
  training loop actually delivers, once every source of idle time (communication the compute cannot
  hide behind, pipeline bubbles, checkpointing, recomputation) is accounted for.

In scope: the compute and memory budget; a parallelism layout that fits the model and trains it
inside that budget, and the communication each part of the layout adds; a data pipeline that feeds
every accelerator a deterministic, resumable stream of tokens; a checkpointing scheme sized against
the cluster's failure rate; detecting and tolerating a straggler; diagnosing and recovering from a
training instability. Out of scope: the model's architecture beyond the sizes given above (fixed, not
something to search over); the training data's content or curation; the loss function's mathematical
form beyond it being a per-token loss suitable for gradient descent; any stage after this one
(fine-tuning, alignment, evaluation for release); and serving the resulting checkpoint.

Produce:

1. A compute and time estimate: total training compute from $C \approx 6ND$, and the wall-clock time
   this implies at the target MFU.
2. A memory plan per accelerator: bytes per parameter for the bf16 weight, the bf16 gradient, the
   fp32 master weight and the two fp32 Adam moments; why splitting this state across accelerators is
   unavoidable at this size; and how activation memory and activation checkpointing fit into what is
   left.
3. A parallelism layout: which of tensor, pipeline and data parallelism runs over which link and why,
   and the communication each one adds.
4. A data model for the run's state: the data pipeline's shard ordering and resumable position, and
   the checkpoint.
5. An architecture diagram and a walk-through of one training step, from a batch of token ids to an
   updated set of parameters.
6. Deep dives into: (a) checkpointing and failure handling — checkpoint size and write time, the
   cluster's failure rate, the checkpoint interval that minimises time lost to checkpointing and to
   redone work, and how a failed accelerator is replaced; (b) detecting and tolerating a straggler in
   synchronous training; (c) diagnosing and recovering from a loss spike at step 300,000; (d) how you
   would verify, while the run is in progress, that it is healthy.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Worth confirming before designing: whether the accelerator budget is fixed at 2,048 (so the design
should reach the best MFU it can within that fleet) or the deadline is fixed instead (so it is worth
asking whether more accelerators would shorten the run, which the compute estimate below answers
directly, since time scales as $1/\text{accelerator count}$ at fixed MFU); and whether the
architecture — 80 layers, width 8,192 — is itself fixed or open to change. This design assumes a
fixed budget of 2,048 accelerators and a fixed architecture, and treats the parallelism layout as the
free variable.

### Requirements and scale

**Compute and time.** Training compute for a dense transformer is well approximated by
$C \approx 6ND$: a forward pass costs about $2N$ FLOPs per token (one multiply and one add for each
of the $N$ parameters a token's activations pass through), and a backward pass costs about twice
that, $4N$ per token, since it computes a gradient with respect to both a layer's input and its
weights. With $N = 70 \times 10^9$ and $D = 1.4 \times 10^{12}$:

$$
C = 6ND = 6 \times 70 \times 10^9 \times 1.4 \times 10^{12} \approx 5.88 \times 10^{23} \text{ FLOPs.}
$$

The cluster's peak throughput is $2{,}048 \times 400 \times 10^{12} \approx 8.19 \times 10^{17}$
FLOP/s; at the target 40% MFU it delivers about $3.28 \times 10^{17}$ FLOP/s, so

$$
T = C / (0.40 \times 8.19 \times 10^{17}) \approx 1.79 \times 10^6 \text{ s} \approx 20.8 \text{ days.}
$$

A global batch of 4,000,000 tokens means $D / 4 \times 10^6 = 350{,}000$ optimiser steps, each costing
$6N \times 4 \times 10^6 \approx 1.68 \times 10^{18}$ FLOPs and, at the achieved throughput, about
5.13 s — matching $T / 350{,}000$.

As a check on the stated parameter count, the standard block-level approximation for a dense
transformer, $N \approx 12 L d_{\text{model}}^2$ (4$d^2$ for the four attention projections plus
$8d^2$ for a 4×-widening MLP, per layer), gives $12 \times 80 \times 8{,}192^2 \approx 64.4 \times
10^9$ for this architecture — close to, though about 8% under, the stated $70 \times 10^9$, the
remainder made up by the token embedding and output projection this block-only approximation leaves
out.

**Memory per accelerator.** Adam keeps, per parameter: a bf16 weight and a bf16 gradient (2 bytes
each, used in the forward and backward pass), and, for the update itself, an fp32 master weight and
two fp32 moments (4 bytes each) — 16 bytes/parameter in total. For the whole model that is
$16 \times 70 \times 10^9 \approx 1.12 \times 10^{12}$ bytes, 1.12 TB — over an order of magnitude
more than any one accelerator's 96 GB, so splitting this state across accelerators is not an
optimisation, it is the only way it fits: even granting every byte of every accelerator's memory to
nothing else, storing it once needs at least $\lceil 1.12 \times 10^{12} / 96 \times 10^9 \rceil = 12$
accelerators.

Three, largely independent, ways to spread a model's parameters and compute across accelerators are
in play here. *Tensor parallelism* splits an individual weight matrix across accelerators — each
accelerator holds a slice of every layer's weights and computes on the same batch of tokens,
exchanging partial results within the layer. *Pipeline parallelism* splits the model's layers into
consecutive stages, each held whole by a different accelerator (or group of accelerators), passing
activations forward and gradients backward between stages. *Data parallelism* instead replicates the
whole model and splits the batch of tokens: each replica runs a full forward and backward pass on its
own slice of the batch, and the replicas average their gradients before every update. *Sharded
optimiser state* — ZeRO, or fully sharded data parallel (FSDP) — layers on top of data parallelism:
instead of every data-parallel replica keeping a full copy of the optimiser state (and, in its more
aggressive forms, the weights and gradients too) for the parameters it holds, that state is split
across the replicas, each keeping only the slice it is responsible for updating.

Twelve accelerators is a bound on storing the state once, not on how this design actually spreads it.
Tensor parallelism alone, spanning the 8 accelerators of one host, gives each accelerator
$70 \times 10^9 / 8 \approx 8.75 \times 10^9$ parameters' worth of state: at 16 bytes/parameter,
$\approx 140$ GB — over the 96 GB budget before a single activation is stored. Extending the split
with pipeline parallelism too — 8 pipeline stages, one host per stage, 10 layers each — brings the
combined split to $8 \times 8 = 64$-way, $70 \times 10^9 / 64 \approx 1.09 \times 10^9$
parameters/accelerator, $\approx 17.5$ GB: comfortably inside the budget.

That leaves $2{,}048 / 64 = 32$ data-parallel replicas of this 64-way-sharded model. Left as is, this
replicates the 17.5 GB of state 32 times over for no benefit — every replica computes the identical
Adam update for its shard of the parameters once gradients are averaged, so there is no reason for
each to separately store, or separately compute, that update. Sharding the optimiser state — the
12 bytes/parameter of master weight and two moments, $\approx 13.1$ GB per accelerator if held in
full — across the 32-way data-parallel group instead, each accelerator keeping only the $1/32$ slice
it owns, cuts that to about 0.41 GB; combined with the 4.4 GB of bf16 weight and gradient every
accelerator still needs in full for its own forward and backward pass (sharding these too, as fully
sharded data parallel does, trades further memory for an extra all-gather of weights before every
use — not needed here), weights, gradients and optimiser state together come to under 5 GB per
accelerator, leaving close to 90 GB of headroom.

The activations a backward pass needs — each layer's input, and the intermediate results inside its
attention and MLP blocks — scale with the number of layers an accelerator holds activations for at
once, the microbatch size, and the sequence length. At this design's sequence length of 4,096, one
microbatch's activation at a single layer boundary alone, in bf16, is
$4{,}096 \times 8{,}192 \times 2 \approx 67 \times 10^6$ bytes, 64 MiB; a pipeline stage holding 10
layers that kept every intermediate tensor for all 10 at once would multiply that many times over.
*Activation checkpointing* discards every intermediate activation except each layer's input during
the forward pass, and recomputes the discarded ones from that input during the backward pass —
trading one extra forward pass (roughly a third more FLOPs, since forward:backward FLOPs run about
1:2) for making the memory held at once independent of how many layers a stage runs, rather than
growing with it. Because $C = 6ND$ counts only the forward and backward pass, not this recomputation,
the extra pass activation checkpointing adds is exactly the kind of overhead the 40% MFU target must
already absorb, alongside communication and pipeline idle time. With close to 90 GB of headroom per
accelerator after weights, gradients and sharded optimiser state, this design does not need
checkpointing to avoid running out of memory; it uses it anyway, since the freed memory buys a deeper
pipeline of in-flight microbatches rather than sitting unused.

### Data model and API

**RunState** (one row for the whole job) — `run_id`, `step` (the single source of truth for training
progress), `status` (`running | checkpointing | recovering | paused`), `last_checkpoint_step`,
`created_at`.

**DataCursor** (one per data-parallel replica, 32 rows) — `run_id`, `dp_rank`, `shard_order_index` (a
position in a seeded, deterministic permutation of the corpus's tokenised shards — the same seed
always yields the same order, so it can be re-derived from nothing but the seed and the index),
`offset_in_shard` (a token offset within the current shard), `epoch`. Reconstructing exactly what a
replica has and has not yet consumed needs only these four fields and the fixed seed.

**CheckpointManifest** (one per checkpointed step) — `run_id`, `step`, `created_at`,
`optimiser_shard_files` (one file per `(tp_rank, pp_stage, dp_rank)` triple, since sharded optimiser
state differs by data-parallel rank), `weight_shard_files` (one per `(tp_rank, pp_stage)`, since
weights are identical across data-parallel replicas once gradients are synchronised — 64 files, not
2,048), `data_cursors` (every replica's `DataCursor` as of this step, so resuming restores the exact
data position alongside the weights).

Control-plane API:

1. `POST /runs/{run_id}/checkpoint` — trigger a checkpoint outside the regular interval, for example
   immediately before a planned maintenance window.
2. `GET /runs/{run_id}/status` — `{step, status, mfu_ewma, loss_ewma, grad_norm_ewma,
   flagged_hosts}`, the read a health dashboard or an alerting rule polls.
3. `POST /runs/{run_id}/resume` — `{from_step}` (default: `last_checkpoint_step`) — (re)starts every
   worker from a specific checkpoint; used both for ordinary recovery and for the rewind in deep dive
   (c).
4. `POST /runs/{run_id}/hosts/{host_id}/drain` — evict one host without stopping the run, backed by
   the live peer-to-peer replacement in deep dive (b).

Worker-facing API, used by each accelerator's training process:

1. `POST /runs/{run_id}/workers/{worker_id}/heartbeat` — `{step, step_time_ms, loss, grad_norm}`,
   sent every step; feeds straggler detection (deep dive (b)) and health monitoring (deep dive (d)).
2. `GET /runs/{run_id}/workers/{worker_id}/assignment` — `{tp_rank, pp_stage, dp_rank, shard_seed}` —
   what a starting or restarted worker asks for to learn its position in the parallel layout and its
   slice of the data order; a spare host swapped in for a failed one asks for, and receives, the
   assignment the host it replaces held.

### Architecture

```text
tokenised shards (object storage)
  |
  |  each of 32 data-parallel replicas reads its own deterministic shard order (DataCursor)
  v
pipeline stage 0 -> stage 1 -> ... -> stage 7        (one data-parallel replica; this row x32)
[host 0, 8 acc.]    [host 1, 8 acc.]     [host 7, 8 acc.]
layers 1-10         layers 11-20         layers 71-80
tensor-parallel, 8-way, intra-host @ 600 GB/s each; activations forward / gradients backward
between stages, inter-host @ 50 GB/s per accelerator
  |
  |  loss, then backward through all 8 stages; every accelerator now holds a bf16 gradient for
  |  the 1/64 shard of the model its (tp_rank, pp_stage) position owns
  v
gradient reduce-scatter across the 32-way data-parallel group (inter-host @ 50 GB/s)
  |
  v
sharded optimiser step -- each accelerator updates only the 1/32 slice of its shard's optimiser
state that it owns, then updated weights are all-gathered back across the 32-way group so every
replica starts the next step with identical bf16 weights
  |
  +--> checkpoint writer, every T_opt steps --> checkpoint store (20 GB/s aggregate write)
  |
  v
job controller / health monitor -- MFU, loss, gradient-norm and per-host step-timing telemetry
```

A step begins with each of the 32 data-parallel replicas' first pipeline stage pulling its next
microbatch of token ids from the shard its `DataCursor` currently points to. Tensor parallelism
shards every weight matrix in a layer 8 ways across a host's accelerators; each accelerator computes
on the full microbatch but only its slice of every matrix, and the 8 accelerators exchange partial
results with an all-reduce at the end of the attention block and again at the end of the MLP block —
2 all-reduces per layer in the forward pass, and, by symmetry, 2 more in the backward pass, each
moving a tensor of size microbatch × sequence length × hidden width over the host's 600 GB/s
intra-host link. This happens at every one of a stage's 10 layers, for every microbatch — by far the
most frequent communication in the whole layout, which is exactly why tensor parallelism is confined
to the fastest link available.

Pipeline parallelism crosses hosts instead: stage 0's output activation for a microbatch is sent to
stage 1, which continues the forward pass, and so on through all 8 stages to the loss; the backward
pass sends the corresponding gradient the other way, stage 7 to stage 0. This is only 2 transfers per
microbatch per stage boundary — far less frequent than tensor parallelism's per-layer all-reduces —
which is what makes it tolerable on the slower, 50 GB/s inter-host link; several microbatches are
kept in flight across the 8 stages at once (the reason activation checkpointing's freed memory is
useful, as noted above), so no stage sits idle waiting on a single microbatch's round trip.

Once every microbatch needed for the step's 4,000,000-token global batch has passed through, each
accelerator holds a bf16 gradient for the 1/64 shard of the model its `(tp_rank, pp_stage)` position
owns. Data parallelism's synchronisation happens exactly once here, the least frequent of the three:
a reduce-scatter across the 32-way data-parallel group leaves each accelerator holding only the
reduced gradient slice its sharded optimiser state is responsible for — for one such slice,
$70 \times 10^9 / 64 \times 2 \approx 2.19 \times 10^9$ bytes, a ring reduce-scatter-plus-all-gather
moves about $2 \times \frac{31}{32} \times 2.19 \times 10^9 \approx 4.24$ GB into and out of each
accelerator, which at 50 GB/s takes about 85 ms — under 2% of the roughly 5.13 s step this design's
compute estimate implies. Each accelerator then applies the Adam update to the slice it owns, and a
final all-gather restores identical bf16 weights across all 32 replicas before the next step's
forward pass. This ordering — the most frequent, highest-fan-in communication (tensor parallelism) on
the fastest link and smallest group; the next most frequent (pipeline parallelism) crossing hosts but
touching only two neighbours at a time; the least frequent but largest single transfer (data
parallelism) also on the slower link, but paid only once per step — is what keeps the union of all
three within the communication budget the 40% MFU target assumes.

### Deep dives

**(a) Checkpointing and failure handling.** What a checkpoint must hold is the fp32 master weight and
both fp32 Adam moments — the bf16 weights can be recomputed by casting down from the fp32 master, so
are not strictly needed in the checkpoint itself, though many designs write them too for a faster
restart that skips the cast. At 12 bytes/parameter, the essential checkpoint is
$12 \times 70 \times 10^9 = 8.4 \times 10^{11}$ bytes, 840 GB; including the bf16 weights adds
$2 \times 70 \times 10^9 = 140$ GB, for 980 GB. At the cluster's 20 GB/s aggregate checkpoint
bandwidth, writing the essential 840 GB takes 42 s; the 980 GB variant, 49 s.

Because tensor parallelism ties a host's 8 accelerators into one synchronous unit — every layer's
all-reduce needs all 8 — a single accelerator failing takes its whole host out of action for the
pipeline stage it was serving. That makes a host's effective failure rate 8 times an individual
accelerator's: a mean time between failures of $50{,}000 / 8 = 6{,}250$ hours per host, and, across
all 256 hosts, a cluster-wide mean time between failures of $6{,}250 / 256 \approx 24.4$ hours (the
same figure a direct per-accelerator count gives, $50{,}000 / 2{,}048 \approx 24.4$ hours — the two
must agree, since both count the same 2,048 accelerators' combined failure rate, only grouped
differently).

Checkpointing trades two costs against each other: time spent writing checkpoints nobody needed, and,
on a failure, time spent redoing work since the last one. With checkpoint write time $t_{\text{ckpt}}$
and mean time between failures $\mu$ in the same units, the interval that minimises their sum — a
standard result usually attributed to Young and Daly — is
$T_{\text{opt}} = \sqrt{2 \, t_{\text{ckpt}} \, \mu}$. With $t_{\text{ckpt}} = 42$ s and
$\mu \approx 87{,}891$ s (24.4 hours), $T_{\text{opt}} \approx 2{,}717$ s, about 45 minutes —
checkpoint roughly every 45 minutes. Assuming a failure is equally likely at any point within an
interval, the expected work redone per failure is half the interval, $T_{\text{opt}}/2 \approx
1{,}359$ s, about 23 minutes; at the optimum, the checkpoint-writing overhead rate
($t_{\text{ckpt}}/T_{\text{opt}}$) and the expected-redo rate ($(T_{\text{opt}}/2)/\mu$) are equal,
each about 1.5%, for a combined overhead of about 3.1% of total wall-clock time — the estimate check
below confirms a sweep of other intervals costs more.

Over the roughly 498-hour run, expect about $2{,}048 \times 498 / 50{,}000 \approx 20$ accelerator
failures in total (equivalently, $498 / 24.4$). Because every accelerator participates in every
step's collectives, one failure anywhere stalls the entire 2,048-accelerator job, not just its own
shard — recovery means the whole run rolling back to the last checkpoint and every accelerator
resuming from there, exactly the redone-work cost above. A small pool of spare, already-provisioned
hosts — a handful is enough at this failure rate, cheap insurance against two failures landing close
together — lets a dead host's position be refilled immediately rather than waiting on new hardware to
be provisioned; the replacement asks the control-plane API for the assignment (`tp_rank`, `pp_stage`,
`dp_rank`, `shard_seed`) the host it replaces held, and loads that position's slice of the last
checkpoint. A host merely detected as a persistent straggler, rather than dead, can skip the
checkpoint reload entirely and instead copy current weights directly from a peer holding the same
shard (deep dive (b)).

**(b) Detecting and tolerating a straggler.** A straggler is a host that is still responding but
running slower than its peers — thermal throttling, a degraded NIC link running under its rated
bandwidth, or a noisy neighbour on shared infrastructure are typical causes. Because every collective
in this design (tensor parallelism's all-reduces, pipeline parallelism's stage-to-stage transfers,
data parallelism's reduce-scatter) blocks until every participant responds, one straggler slows the
entire 2,048-accelerator job to its pace, every single step — far more costly than a straggler in an
embarrassingly parallel workload, where it only slows its own share of the work.

Detection uses the same per-step timing the worker heartbeat already reports (the data model above):
a host whose recent step times sit well above the fleet's median for several consecutive steps,
rather than just one, is flagged — averaging over several steps is what separates a genuine,
persistent straggler from an ordinary single-step blip such as a garbage-collection pause or a
transient network retry, neither worth reacting to.

Two ways to respond, once flagged:

- *Redundant computation*: run a spare accelerator alongside the slow one for the same shard and take
  whichever finishes first. This works well for loosely coupled or asynchronous work, but here it
  would mean duplicating an entire tensor-parallel group's compute (8 accelerators, not 1) for a
  position that might recover on its own, and the job's collectives would still have to wait for
  whichever of the two finishes — a large hardware cost for an uncertain benefit.
- *Treat a confirmed straggler as a failure and replace it* (chosen): evict the flagged host and swap
  in a spare, using the same control-plane `drain` path a hard failure uses. The difference from a
  hard failure is that a straggler is still alive: rather than reloading the last on-disk checkpoint
  and losing the work since then, the replacement copies current bf16 weights directly from a peer
  that holds the identical shard — any of the other 31 data-parallel replicas at the same
  `(tp_rank, pp_stage)` position — over the network, while the job is paused at a step boundary. This
  costs only the pause-and-copy time (one shard's worth of bf16 weights, a few gigabytes, at
  50 GB/s: tens of milliseconds) rather than the checkpoint interval's worth of redone work, since
  nothing was actually lost.

The detection threshold trades false positives against slow reaction: evicting a host that was only
briefly slow costs tens of milliseconds and one unnecessary swap, while tolerating a genuine
straggler for longer costs the entire fleet that same slowdown on every subsequent step, for as long
as it goes undetected — at 2,048 accelerators, even a straggler adding 10% to every step's time
wastes 10% of the whole run's accelerator-hours, dwarfing the cost of reacting a little too eagerly.
The threshold should therefore favour fast action.

**(c) Diagnosing and recovering from a loss spike at step 300,000.** A sudden, large increase in the
training loss at a specific step, rather than the gradual, noisy decrease training otherwise shows,
has four common causes, distinguished by what the logged per-step telemetry (worker heartbeats, and
the data and gradient history deep dive (d) keeps) shows at that step:

- *The data batch*: the step's specific tokens are anomalous — a corrupted or misaligned shard
  region, or a batch that is unusually repetitive or low-entropy. The `DataCursor` history pins down
  exactly which shard and offset every step consumed, so the implicated batch can be re-examined, or
  replayed in isolation against the model from just before the spike, without guessing.
- *The learning rate*: the schedule's value at that step is too high for where training currently is
  — for instance, if the gradient noise scale has grown as training has progressed, a rate that was
  stable earlier can stop being so. Checking the logged schedule value against the step number
  catches this directly.
- *Numerics*: an activation or gradient overflowed bf16's range, or the Adam denominator's epsilon
  became too small relative to a parameter whose second moment had decayed to near zero after a long
  stretch of small gradients — both leave a signature in per-layer gradient-norm history (deep dive
  (d)): one layer's norm spikes far more than the rest, localising the fault to a specific operation
  rather than the whole model.
- *Optimiser state*: closely related to the numerics case — a parameter subset (rare-token embedding
  rows are a common example) whose Adam second moment has decayed close to zero makes that
  parameter's effective step size, proportional to $1/\sqrt{v}$, spike enormously the next time it
  receives a sizeable gradient.

None of this is diagnosable after the fact without having logged it during — which is exactly why the
per-step data position, learning rate, loss and gradient-norm history in deep dive (d) needs to
already be running before any spike happens, not added in response to one.

Recovery reuses the checkpoint-rewind machinery deep dive (a) already provides: resume from the last
checkpoint before the spike (at most $T_{\text{opt}}/2$ of work old, on average) with a `resume`
control-plane call. What differs is what changes before continuing:

- An isolated bad batch: resume and skip past the offending step's data position specifically,
  leaving everything else unchanged — the deterministic, resumable data pipeline makes this precise
  rather than approximate.
- A systemic cause (the learning rate, or numerics that a specific batch merely triggered rather than
  caused): resume with the learning rate lowered, or with gradient clipping tightened, or with a
  z-loss term added — a small auxiliary loss penalising the log-softmax normaliser, which keeps the
  final layer's logits from growing without bound, a known instability source in large transformers —
  since skipping one batch would not prevent recurrence if the underlying cause is still there.

Not every spike needs a rewind: gradient clipping already bounds how much damage a single bad step
can do to the parameters, and a spike sometimes recovers within a handful of steps on its own. A
reasonable policy rewinds only once the loss fails to recover within some bounded number of steps,
rather than on every transient spike — reusing the redone-work cost from deep dive (a) only when it
is actually needed.

**(d) Verifying the run is healthy.** Four signals, tracked continuously and logged with each step's
telemetry, catch different failure modes. Achieved MFU — this design's own
$C_{\text{step}} / (\text{peak FLOP/s} \times \text{step time})$, computed live from each step's
measured wall-clock time — falling below the roughly 40% baseline with no change to the model or data
is a model-agnostic signal of an infrastructure problem: a straggler, a degraded interconnect link,
or thermal throttling, and shares its underlying telemetry directly with the straggler detection in
deep dive (b). The training loss itself, both the raw per-step value (noisy, but where a spike like
deep dive (c)'s first shows up) and a smoothed running average (for tracking the slower, expected
decline, and catching a plateau or a gradual drift that no single step's value would reveal), is the
primary progress signal. Gradient norms, both the global norm before clipping (a jump here is the
spike's leading indicator) and the per-layer norms (which localise it, as deep dive (c) uses), catch
instability before it necessarily shows up in the loss. Periodic evaluation on a fixed held-out set,
distinct from any training shard, at a regular step cadence, catches what training loss alone cannot
— most importantly, a data-pipeline bug that lets held-out data leak into training, which would make
training loss look fine while true generalisation does not improve; the same evaluation pass is also
where the data pipeline's own invariant is checked directly, by comparing each shard's recorded
consumption count against what the step number and global batch size imply it should be, rather than
only trusting that the deterministic ordering has held.

### Follow-ups

- **Mixture-of-experts and expert parallelism.** Routing each token to a small subset of many expert
  MLP blocks instead of one dense MLP adds a fourth parallelism dimension, expert parallelism, that
  shards *different* experts across accelerators rather than sharding the same dense computation as
  tensor parallelism does, and needs an all-to-all exchange — not an all-reduce — to route each
  token's activation to its assigned expert's accelerator and back, with a data-dependent volume that
  can imbalance across experts.
- **Sequence or context parallelism.** At a sequence length long enough that activation memory's
  attention term (which scales with the square of sequence length) or even its linear term dominates
  the memory budget rather than parameter count, sharding the sequence dimension itself across
  accelerators — each holding a contiguous slice of positions — needs a ring-style exchange of key and
  value blocks for attention, since a query position needs the keys and values of every preceding
  position, overlapped with attention compute as the ring turns.
- **Elastic training when capacity changes.** Data parallelism is the dimension to resize, since its
  replicas are interchangeable, unlike tensor or pipeline parallelism, which are tied to a fixed
  sharding of the model; a resize needs the global batch size (or per-replica microbatch count)
  recomputed, a checkpoint-based restart at the new accelerator count reusing the same path as
  ordinary failure recovery, and care that the learning-rate/batch-size relationship does not silently
  drift if the global batch is allowed to change with capacity rather than held fixed.
- **FP8 training.** Using 8-bit floating point for the matrix-multiply inputs in the compute-heavy
  matmuls roughly doubles achievable peak FLOP/s over bf16 on hardware with native support, directly
  raising the ceiling the 40% MFU target is a fraction of — but it does not change the
  12-bytes/parameter optimiser-state term, since master weights and moments still need higher
  precision to accumulate small updates without underflow, and it adds per-tensor or per-block scaling
  factors to track, a new numerical-stability surface directly relevant to deep dive (c).
- **A 10× larger model.** The memory argument scales linearly — state at 16 bytes/parameter for
  700 billion parameters is about 11.2 TB, needing proportionally more accelerators just to fit, and
  likely a deeper tensor-times-pipeline degree, since a single host's $8 \times 96$ GB can no longer
  hold even a modest slice comfortably at the same tensor-parallel width. Compute time scales linearly
  in parameters for the same token count, and compounds further if, as compute-optimal scaling
  suggests, a 10× larger model is also trained on proportionally more tokens.

<details>
<summary>Estimate check (runnable)</summary>

```python
import math
import random
import statistics as stats

# ---- setup ----
n_layers, d_model, seq_len = 80, 8192, 4096
N = 70e9                       # stated parameter count
D = 1.4e12                     # training tokens
n_accel, accel_mem, accel_peak = 2048, 96e9, 400e12
accel_per_host = 8
n_hosts = n_accel // accel_per_host
assert n_hosts == 256
inter_host_bw = 50e9           # bytes/s per accelerator
target_mfu = 0.40
global_batch = 4_000_000       # tokens/step
ckpt_bw = 20e9                 # bytes/s, aggregate
mtbf_accel_hours = 50_000

# ---- compute and time ----
C = 6 * N * D
assert C == 5.88e23

peak_cluster = n_accel * accel_peak
assert peak_cluster == 8.192e17
achieved = target_mfu * peak_cluster
assert math.isclose(achieved, 3.2768e17)

T_run = C / achieved
assert round(T_run / 86400, 1) == 20.8

steps = D / global_batch
assert steps == 350_000

C_step = 6 * N * global_batch
t_step = C_step / achieved
assert math.isclose(t_step, T_run / steps)
assert round(t_step, 2) == 5.13

# parameter-count check: the standard 12Ld^2 block approximation against the stated N
params_formula = 12 * n_layers * d_model ** 2
assert params_formula == 64_424_509_440
assert 0.85 < params_formula / N < 0.95            # within ~8%, the gap being embeddings + output head

# ---- memory per accelerator ----
bytes_bf16_weight, bytes_bf16_grad = 2, 2
bytes_fp32_master, bytes_fp32_m1, bytes_fp32_m2 = 4, 4, 4
bytes_per_param = bytes_bf16_weight + bytes_bf16_grad + bytes_fp32_master + bytes_fp32_m1 + bytes_fp32_m2
assert bytes_per_param == 16
opt_bytes_per_param = bytes_fp32_master + bytes_fp32_m1 + bytes_fp32_m2
assert opt_bytes_per_param == 12
wg_bytes_per_param = bytes_bf16_weight + bytes_bf16_grad
assert wg_bytes_per_param == 4

total_state = bytes_per_param * N
assert total_state == 1.12e12                       # 1.12 TB

floor_accelerators = math.ceil(total_state / accel_mem)
assert floor_accelerators == 12

TP = 8                                               # within a host, matches the fast intra-host link
mem_tp_only = (N / TP) * bytes_per_param
assert mem_tp_only == 140e9                          # exceeds one accelerator's 96 GB

PP = 8                                               # across hosts: one host per stage
assert n_layers % PP == 0 and n_layers // PP == 10
model_parallel = TP * PP
assert model_parallel == 64
params_per_shard = N / model_parallel
assert params_per_shard == 1_093_750_000

mem_unsharded = params_per_shard * bytes_per_param
assert mem_unsharded == 17.5e9                       # fits comfortably in 96 GB

DP = n_accel // model_parallel
assert DP == 32

opt_state_full = params_per_shard * opt_bytes_per_param
opt_state_sharded = opt_state_full / DP
wg_mem = params_per_shard * wg_bytes_per_param
mem_sharded = wg_mem + opt_state_sharded
assert round(opt_state_full / 1e9, 3) == 13.125
assert round(opt_state_sharded / 1e9, 3) == 0.41
assert round(wg_mem / 1e9, 3) == 4.375
assert mem_sharded < 5e9                             # "under 5 GB"
assert accel_mem - mem_sharded > 90e9                # "close to 90 GB of headroom"

activation_tensor_bytes = seq_len * d_model * 2      # one microbatch=1 sequence, one layer boundary, bf16
assert activation_tensor_bytes == 67_108_864
assert activation_tensor_bytes / 2**20 == 64.0        # 64 MiB

# ---- ring communication: data-parallel gradient reduce-scatter + all-gather ----
grad_shard_bytes = params_per_shard * bytes_bf16_grad  # NOTE: this accelerator's (tp,pp) shard, not N
assert grad_shard_bytes == 2_187_500_000
ring_volume = 2 * (DP - 1) / DP * grad_shard_bytes
assert round(ring_volume / 1e9, 2) == 4.24
t_allreduce = ring_volume / inter_host_bw
assert round(t_allreduce * 1000) == 85               # ms
assert t_allreduce / t_step < 0.02                    # under 2% of one step

print("all requirements-and-scale numbers check out")

# ---- checkpointing and cluster failure rate ----
ckpt_essential = opt_bytes_per_param * N
ckpt_with_weights = ckpt_essential + bytes_bf16_weight * N
assert ckpt_essential == 840e9
assert ckpt_with_weights == 980e9

t_ckpt = ckpt_essential / ckpt_bw
t_ckpt_full = ckpt_with_weights / ckpt_bw
assert t_ckpt == 42.0
assert t_ckpt_full == 49.0

host_mtbf_hours = mtbf_accel_hours / accel_per_host
cluster_mtbf_hours = host_mtbf_hours / n_hosts
assert host_mtbf_hours == 6_250.0
assert round(cluster_mtbf_hours, 1) == 24.4
assert math.isclose(cluster_mtbf_hours, mtbf_accel_hours / n_accel)   # same rate, counted either way

mtbf_seconds = cluster_mtbf_hours * 3600
T_opt = math.sqrt(2 * t_ckpt * mtbf_seconds)
assert round(T_opt) == 2717
assert round(T_opt / 60) == 45

expected_redo = T_opt / 2
assert round(expected_redo / 60) == 23

overhead_frac = math.sqrt(2 * t_ckpt / mtbf_seconds)
cross_check = t_ckpt / T_opt + expected_redo / mtbf_seconds
assert math.isclose(overhead_frac, cross_check)
assert round(overhead_frac * 100, 1) == 3.1

expected_failures = n_accel * (T_run / 3600) / mtbf_accel_hours
assert round(expected_failures) == 20
assert math.isclose(expected_failures, T_run / mtbf_seconds)

print("all checkpointing and failure-rate numbers check out")


# ---- tiny simulation: the Young-Daly interval approximately minimises total overhead ----
def simulate_overhead(interval: float, t_ckpt: float, mtbf: float, target_useful: float,
                       rng: random.Random) -> float:
    """Simulate one run needing `target_useful` seconds of useful compute, blocked by a
    checkpoint costing `t_ckpt` seconds every `interval` useful seconds; failures arrive as a
    Poisson process at rate 1/mtbf in wall-clock time and discard all progress since the last
    completed checkpoint. Returns wall-clock time minus target_useful: the pure overhead."""
    useful_done = progress = wall = 0.0
    next_failure = rng.expovariate(1.0 / mtbf)
    while useful_done < target_useful:
        remaining_to_ckpt = interval - progress
        time_to_failure = next_failure - wall
        step = min(remaining_to_ckpt, time_to_failure)
        wall += step
        progress += step
        if time_to_failure <= remaining_to_ckpt:        # a failure hits first: lose this segment
            progress = 0.0
            next_failure = wall + rng.expovariate(1.0 / mtbf)
            continue
        useful_done += progress                          # reached a checkpoint boundary
        progress = 0.0
        if useful_done >= target_useful:
            break
        wall += t_ckpt
    return wall - target_useful


factors = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0]
mean_overhead = {}
for f in factors:
    trials = [simulate_overhead(T_opt * f, t_ckpt, mtbf_seconds, T_run, random.Random(1000 + s))
              for s in range(300)]
    mean_overhead[f] = stats.mean(trials)

assert mean_overhead[1.0] == min(mean_overhead.values())         # T_opt is the best of the sweep
assert mean_overhead[0.15] > 2 * mean_overhead[1.0]               # far too short: overhead dominates
assert mean_overhead[4.0] > 2 * mean_overhead[1.0]                 # far too long: redone work dominates

print("checkpoint-interval simulation confirms the Young-Daly interval near-minimises total overhead")
print("all checks passed")
```

</details>

</details>
