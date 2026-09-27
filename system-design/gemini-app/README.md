# Design the Gemini App

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| System design | ★★★★☆ | Hard | SWE · MLE · Applied AI | streaming, conversation-storage, context-management, model-routing, safety-filtering, multimodal-uploads, capacity-planning | 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

Design the backend of a consumer AI assistant app, with web and mobile clients, where a signed-in user
holds multi-turn conversations with a large language model, can attach images to a message, and watches
the answer stream in token by token.

A *turn* is one exchange between the user and the assistant: the message the user sends and the model's
reply to it, taken together. A *conversation* is an ordered sequence of turns belonging to one signed-in
user, identified by a conversation id; a user may hold many conversations, and a conversation is never
shared between users. *Time to first token* (TTFT) is the time from the moment the client sends a turn to
the moment the first token of the model's reply reaches that client. *Prefill* is the model's processing of
everything already known when a turn begins — the system prompt, the assembled conversation history, and
the new message — into a form (a key/value cache) the model can generate from; its cost scales with the
number of tokens it processes. *Decode* is the step that follows: generating the reply one output token at
a time, each token attending over the prefilled context and every token generated before it; its cost
scales with the number of tokens it generates.

Scale this design for:

- 60,000,000 daily active users, 12 turns per user per day on average; peak traffic is 3 times the daily
  average.
- An average turn sends 2,000 input tokens to the model (system prompt + conversation history + the new
  message) and generates 400 output tokens.
- 10% of turns include one image of about 1 MB.
- Latency targets: 95th-percentile TTFT under 1.5 s for text-only turns; streaming at least 30 output
  tokens/s per response once it has started.
- One serving host of the default model sustains 50,000 prefill tokens/s or 5,000 decode tokens/s — treat
  prefill and decode as two separate pools of hosts for this estimate, neither able to do the other's
  work. A larger model is also available; it costs 4 times as much per token, on either pool, as the
  default model.
- Conversation history is kept until the user deletes it. Availability target: 99.9%.

In scope: sending a turn and streaming the response; listing a user's conversations and paging through a
conversation's history; uploading an image and attaching it to a turn; assembling the model's context from
the conversation's history; choosing between the default and the larger model for a turn; safety filtering
of both the input a user sends and the output the model streams back; per-user rate limits and quotas;
recording a user's feedback (thumbs up or down) on a reply, for later evaluation. Out of scope: training
the model; voice input or output; sharing a conversation publicly; billing.

Produce:

1. Requirements and a scale estimate: average and peak turns/s; prefill and decode tokens/s at peak and
   the hosts each pool needs; the effect on that host count of routing a share of turns to the larger
   model; daily storage growth for conversation text and for images (state your bytes-per-token and
   per-message assumptions).
2. A data model and an API: the endpoints for managing conversations and attachments, and the protocol
   used to stream a turn's response.
3. An architecture diagram and a walk-through of one turn that includes an image, from the client's
   request to the client receiving the first and the last streamed token.
4. Deep dives into: (a) streaming and reconnection — how a client that drops mid-response resumes without
   the model regenerating anything; (b) context assembly — how history that has grown too large for the
   token budget is truncated or summarised, and how the system prompt and conversation history are
   prefix-cached; (c) safety — where input and output classifiers sit, and how a response already being
   streamed to the client is moderated; (d) model routing and graceful degradation when the system is
   overloaded; (e) quotas and abuse prevention.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points worth confirming before designing: whether this serves a single region or is global from the
start, since that changes whether conversation storage and the serving pools need to be replicated across
regions from day one; and whether an in-flight reply belongs to one connected client at a time or must also
reach a second device that opens the same conversation mid-stream. This design assumes a single region to
start (multi-region is a follow-up) and assumes no cross-device sync of an in-flight reply is required —
revisited in the streaming deep dive, where it turns out to fall out of the design almost for free.

### Requirements and scale

**Turn rate.** $60{,}000{,}000$ daily actives at 12 turns/day each:

$$60{,}000{,}000 \times 12 = 720{,}000{,}000 \text{ turns/day,}$$

$$\text{avg turns/s} = \frac{720{,}000{,}000}{86{,}400} \approx 8{,}333.$$

At $3\times$ peak-to-average, peak throughput is about $25{,}000$ turns/s.

**Prefill and decode tokens/s at peak.** Each turn sends 2,000 input tokens and generates 400 output
tokens, so at $25{,}000$ turns/s:

$$\text{peak prefill} = 25{,}000 \times 2{,}000 = 50{,}000{,}000 \text{ tokens/s} \approx 50\text{M/s,}$$

$$\text{peak decode} = 25{,}000 \times 400 = 10{,}000{,}000 \text{ tokens/s} \approx 10\text{M/s.}$$

One default-model host sustains 50,000 prefill tokens/s or 5,000 decode tokens/s, and the two pools are
separate, so the bare floor is $50{,}000{,}000/50{,}000=1{,}000$ prefill hosts and
$10{,}000{,}000/5{,}000=2{,}000$ decode hosts. Neither floor leaves any margin for a host failing a health
check or a rolling deploy; with 20% headroom, the provisioned fleet is $1{,}200$ prefill hosts and
$2{,}400$ decode hosts. A turn's own prefill, run alone with nothing queued ahead of it, takes
$2{,}000/50{,}000=40$ ms — a small fraction of the 1.5 s TTFT target, leaving most of the budget for
queueing under load, the classifiers in the safety deep dive, and network transit.

**Routing to the larger model.** Reading "4× per token" as 4× the host-time per token, routing a share $s$
of turns to it multiplies the token-processing load's host requirement by $(1-s) + 4s = 1 + 3s$: the $s$
fraction of turns now costs 4 times as many host-seconds per token as before, while the rest is unchanged.
At $s=10\%$, the floor rises to $1{,}000\times1.3=1{,}300$ prefill hosts and $2{,}000\times1.3=2{,}600$
decode hosts; with the same 20% headroom, $1{,}560$ and $3{,}120$. The relationship is linear in $s$, so
this prices any routing share the same way rather than needing a separate estimate per share.

**Storage.** A stored message needs `message_id` (16 bytes), `conversation_id` (16 bytes), `turn_seq` (8
bytes), `created_at` (8 bytes) and `role` (2 bytes) — 50 bytes of fixed fields — plus its text, at an
assumed 4 bytes/token of UTF-8 (a standard rule of thumb for English text). A turn stores two rows: the
user's new message, assumed to average 40 tokens — far shorter than the 2,000-token context, which is
mostly reconstructed history and the system prompt rather than newly written text — and the assistant's
reply, exactly the given 400 output tokens. That is $50+40\times4=210$ bytes for the user row and
$50+400\times4=1{,}650$ bytes for the assistant row, $1{,}860$ bytes/turn. At $720{,}000{,}000$ turns/day,
text grows by about $720{,}000{,}000\times1{,}860\approx1.34$ TB/day.

10% of turns carry one image of about 1 MB: $720{,}000{,}000\times0.10=72{,}000{,}000$ images/day,
$72{,}000{,}000\times1\text{ MB}=72$ TB/day. Image bytes outweigh text by about $54\times$, which is why the
object store, not the row store above, is what has to be built for scale, and why it is the one that needs
a lifecycle or cold-tier policy as retention accumulates indefinitely.

### Data model and API

Every table below is partitioned by `user_id`: a conversation belongs to exactly one user, so this keeps
one user's conversations, messages, attachments and feedback on the same shard, which is what makes "kept
until the user deletes it" a bounded, single-partition operation instead of a scatter-gather across the
store.

**User** — `user_id`, `created_at` (authentication and profile fields are out of scope).

**Conversation** — `user_id`, `conversation_id`, `title` (generated from the first turn), `last_turn_seq`,
`created_at`, `updated_at`. Partition key: `user_id`.

**Message** — `user_id`, `conversation_id`, `message_id`, `turn_seq` (the turn this message belongs to),
`role` (`user` | `assistant`), `text`, `attachment_ids` (possibly empty), `token_count`, `model` (which
model generated it; null for a user's own message), `status` (`streaming` | `complete` | `blocked`,
meaningful only for an assistant message), `created_at`. Partition key: `user_id`, clustered by
`(conversation_id, turn_seq)`.

**Attachment** — `user_id`, `attachment_id`, `message_id`, `object_key` (its location in the object
store), `mime_type`, `byte_size`, `moderation_status`, `created_at`. Partition key: `user_id`. The bytes
themselves live in the object store; this row is only metadata and a pointer.

**Feedback** — `user_id`, `feedback_id`, `message_id`, `conversation_id`, `rating` (`up` | `down`),
`reason` (optional), `created_at`. Partition key: `user_id`.

**TurnBuffer** (in a fast external store, not the row store above — short-lived, not part of "kept until
the user deletes it") — `turn_id` → the tokens generated so far for an in-flight turn, plus a `status`
(`streaming` | `done` | `blocked`). This is the resumable stream's backing store; the streaming deep dive
covers it. Key: `turn_id`, TTL a few minutes past completion — long enough to outlast a reconnect, short
because once a turn is `complete` its tokens already sit in the durable `Message.text` above, and a client
that gives up on the stream and re-reads the conversation gets the same answer that way.

REST, for conversation and attachment management:

1. `GET /v1/conversations?before=&limit=` — the caller's conversations, newest first, each with a
   preview — for the conversation list.
2. `POST /v1/conversations` — creates an empty conversation, returns `{conversation_id}`.
3. `GET /v1/conversations/{conversation_id}/messages?before_seq=&limit=` — paginated history, newest
   first.
4. `DELETE /v1/conversations/{conversation_id}` — deletes the conversation and everything under it
   (messages, attachments, feedback on those messages).
5. `POST /v1/attachments` — `{conversation_id, mime_type, byte_size}` → `{attachment_id, upload_url}`, a
   short-lived pre-signed URL the client uploads the image bytes to directly, so a 1 MB image never has to
   be proxied through the API tier.
6. `POST /v1/messages/{message_id}/feedback` — `{rating, reason?}` → `200`.

Turn submission and streaming — Server-Sent Events (SSE), chosen over a WebSocket because the stream is
one-directional (the client never needs to send anything back over the same connection once a turn is
submitted):

7. `POST /v1/conversations/{conversation_id}/turns` — `{idempotency_key, text, attachment_ids?}` →
   `202 {turn_id}`. The server starts assembling context and generating immediately; it does not wait for
   the client to open a stream. A retried call carrying an `idempotency_key` already seen on this
   conversation returns the existing `turn_id` instead of starting a second turn.
8. `GET /v1/turns/{turn_id}/stream?after_seq=` — opens an SSE connection and replays everything generated
   from `after_seq` onward (0 on first connect), then continues with live tokens as they are generated;
   the standard SSE `Last-Event-ID` header, with each event's `id` set to its `seq`, does the same job
   without the query parameter for a client that supports it. Event types:
   - `token {seq, text}` — one generated token (or a short run of them, batched for transport
     efficiency).
   - `done {seq, message_id}` — generation finished and persisted as `message_id`; `seq` is the total
     token count.
   - `blocked {seq, reason}` — the output safety classifier cut the response short.
   - `error {reason}` — generation failed; the client may retry the whole turn with the same
     `idempotency_key`.

### Architecture

```text
Client (web / mobile)
  |
  |  1. POST /v1/attachments -> pre-signed upload URL; client PUTs image bytes directly
  |  2. POST /v1/conversations/{id}/turns  {idempotency_key, text, attachment_ids}
  v
API gateway -- authenticates the caller; per-user rate limit and quota (deep dive e)
  |
  v
Turn orchestrator -- owns one turn end to end
  |  |
  |  +--> Conversation store (User, Conversation, Message, Attachment, Feedback; by user_id)
  |  +--> Object store (image bytes, referenced by Attachment.object_key)
  |  +--> Feedback log (append-only export of Feedback, for the evaluation pipeline)
  v
Context builder -- system prompt + (truncated/summarised) history + new message;
  |                 looks up the prefix cache for this conversation (deep dive b)
  v
Input safety classifier -- a violation short-circuits here, before any model host runs
  v
Model router -- default vs. larger model; load-aware, degrades toward default (deep dive d)
  v
Prefill pool (default | larger model hosts) -- computes the key/value cache for the context
  v
Decode pool (default | larger model hosts) -- generates output tokens from the handed-off cache
  v
Output safety classifier -- scores each newly generated span before release (deep dive c)
  v
Turn orchestrator -- appends each released token to the TurnBuffer and to the open stream
  v
Client (web / mobile) -- GET /v1/turns/{turn_id}/stream, resumable from any offset (deep dive a)
```

A user attaches a photo to a new message. The client first calls `POST /v1/attachments` with the image's
size and MIME type and gets back a pre-signed `upload_url`; it uploads the image bytes straight to the
object store over that URL, not through the API tier, and receives an `attachment_id`. It then calls
`POST /v1/conversations/{id}/turns` with the message text, that `attachment_id`, and a fresh
`idempotency_key`, and opens `GET /v1/turns/{turn_id}/stream` to receive the reply.

The API gateway authenticates the request and checks the user's rate budget (deep dive (e)) before
anything else runs. The turn orchestrator writes a `Message` row for the user's half of the turn —
referencing the `Attachment` row already written when the upload completed — and reads back enough of the
conversation's history for the context builder. The context builder assembles the system prompt, the
(possibly truncated or summarised, deep dive (b)) history, and the new message; because this turn carries
an image, it also includes the image's encoded representation, which sits outside the 2,000-input-token,
1.5 s TTFT profile the requirements describe for a text-only turn — an image-bearing turn is out of scope
for that tight a bound precisely because of this extra encoding step. The assembled context passes the
input safety classifier; a turn that fails here never reaches a model host at all.

The model router picks the default or the larger model for this turn (deep dive (d)) and hands the
assembled context to a host in that model's prefill pool, which computes the key/value cache for the whole
context and hands it off to a host in the same model's decode pool. Decode begins generating output
tokens; each is checked by the output safety classifier (deep dive (c)) and, once released, is both
appended to the turn's `TurnBuffer` and pushed down the client's open SSE connection as a `token` event —
the first of these events reaching the client is the point the TTFT target is measured against. Streaming
continues, at least 30 tokens/s per the target, until decode finishes; the orchestrator then writes the
completed reply as the assistant's `Message` row, marks the `TurnBuffer` `done`, and sends the `done` event
immediately after the client has received the last streamed token — the point this walk-through ends. If
the client's connection had dropped at any point, reconnecting to the same
`GET /v1/turns/{turn_id}/stream` with the offset of the last token it applied resumes exactly where it left
off, live tokens and all, without the model regenerating anything (deep dive (a)).

### Deep dives

**(a) Streaming and reconnection.** Every token the decode pool releases — after the output safety
classifier clears it — is appended to that turn's `TurnBuffer` before it is pushed to whichever client
happens to be connected; generation itself runs to completion (or to a `blocked` cut-off) regardless of
whether a client is connected to watch it, so the buffer is always caught up to everything actually
generated. A client's `GET /v1/turns/{turn_id}/stream?after_seq=` — on first connect with `after_seq=0`,
or again after a drop with the offset of the last token it applied — replays `buffer[after_seq:]` and then
keeps streaming live; nothing about this asks a model host to produce anything a second time, which is the
property the brief asks for by name.

*Where the buffer lives.* Pinning a reconnecting client to the one orchestrator instance that happened to
accept the original turn — a connection-registry lookup, the same shape as routing a message to whichever
gateway holds a specific user's connection in a chat system — would work, but only by adding a lookup and
a piece of state (which instance is handling `turn_id`) that a shared buffer makes unnecessary: any
orchestrator instance can serve the resume once the buffer itself lives in a fast external store keyed by
`turn_id`, not in one instance's memory. The cost is a small write per released token to that store instead
of to local memory; at $10{,}000{,}000$ decode tokens/s peak this is exactly the rate the decode pool is
already producing tokens at, so it adds a store write on the existing hot path rather than a new
bottleneck, and it is what lets any instance in a horizontally scaled orchestrator fleet serve any
reconnect — the same property that lets the same mechanism serve a second device attaching to the same
`turn_id`, if that assumption from the opening changes later.

*Why not just regenerate.* Resubmitting the whole turn on reconnect would need no buffer at all, but
decoding is stochastic, so a regenerated reply need not match what the client already rendered — the two
could visibly diverge mid-sentence — and it spends a full second decode pass on tokens already paid for
once. Resending the buffer's full contents from offset 0 on every reconnect, rather than from the client's
own offset, avoids that divergence but not the wasted bandwidth of re-sending a mostly-already-seen reply;
tracking the offset costs nothing beyond what the client was already going to send as `Last-Event-ID`.

The runnable check below models a turn's token-by-token generation against a client that drops and
reconnects at random points, including cases where several tokens accumulate in the buffer before a
reconnect drains them all at once, and confirms the client's reassembled transcript is exactly the
generated one — no gap, no repeat — for every trial.

**(b) Context assembly.** The context builder concatenates the system prompt, the conversation's history,
and the new message; when that would exceed the token budget the context builder is prepared to send, the
oldest turns are not simply dropped but folded into a running per-conversation summary, so a long-running
conversation still carries forward what it needs without regenerating the summary on every turn — only
when an eviction is about to happen and the current summary is stale is a small model call asked to fold
the newly evicted turns into it, keeping summarisation cost off the common turn's path entirely.

*Prefix caching.* Turn $n$'s context is turn $n-1$'s context with turn $n-1$'s reply appended and a new
message added at the end — an append-only prefix, not an arbitrary new prompt each time — which makes
multi-turn chat close to the best case for reusing a previously computed key/value cache: if the cache
built while serving turn $n-1$ is still resident on a decode host when turn $n$ arrives, only the newly
appended tokens need to be prefilled, not the whole context again. Using the numbers above, a cache hit
needs the assumed 40-token new message prefilled rather than the full 2,000-token context:

$$\text{effective tokens} = h \times 40 + (1-h) \times 2{,}000,$$

which at an assumed 80% hit rate is $0.8\times40+0.2\times2{,}000=432$ tokens, about a 78% cut in the
prefill load that turn would otherwise cost. This is an optimisation on top of the provisioned floor above,
not a reason to provision below it: a cold start, a fleet rebalance, or simply a burst of first turns in
new conversations all miss the cache and fall back to the full context, correctly, just at the un-cached
cost — the floor has to be sized for that case, the same way the inference pools above are, since nothing
about correctness depends on the cache being warm.

Getting anywhere near that hit rate needs the model router to prefer sending a conversation's next turn
back to a decode host that recently handled its previous one, within a bounded warm window, rather than
routing purely on load; a host that has not seen a conversation's traffic in a while evicts its cache
under memory pressure from more recently active ones, the same least-recently-used policy a shared cache
always needs, and a miss is never a correctness problem, only a slower turn.

**(c) Safety.** The input classifier scores the assembled context — the new message together with the
attached image, once the context builder has produced both a text form and whatever representation the
model consumes for the image — before the model router is even reached; a turn it rejects never reaches a
model host, so the input side spends no prefill or decode budget on something that will be blocked anyway.

The output side has a harder constraint: the reply is streaming out as it is generated, and a token, once
shown to the user, cannot be unshown. Classifying only the complete reply before releasing anything would
remove that risk entirely, but means holding back every token until generation finishes — exactly what the
$\geq 30$ tokens/s streaming target rules out. Classifying every single token as it is produced avoids that,
but gives the classifier almost nothing to work with per call — one token rarely carries enough context to
score — while adding a classifier call to every one of the $10{,}000{,}000$ decode tokens/s at peak. This
design instead classifies a short trailing span of newly generated text at a time — for example every 10
tokens, a third of a second of generation at the 30 tokens/s floor — and only releases that span to
the client once it clears; a span that fails stops generation, releases nothing further from it, and the
orchestrator sends `blocked` instead of `done`. The cost is a small, constant holding delay on the stream,
not one that grows with the length of the reply, and a call to the classifier once per span rather than
once per token; the benefit is that nothing the classifier would have rejected is ever actually rendered
to the user, unlike a design that only stops *further* tokens once a violation is caught after the fact.

**(d) Model routing and graceful degradation.** The router's baseline decision — default or larger model
for a turn — rests on a lightweight classifier over the new message, together with any explicit hint the
client sends (a user picking a "more capable" mode, say), run before a prefill pool is chosen; that
classification alone, ignoring load entirely, is what a turn would get with unlimited capacity, but it is
not what this design actually does with it.

*Static routing, ignoring load* — sending every turn the classifier flags to the larger pool regardless of
that pool's queue depth — was rejected: the larger pool is provisioned for its expected share (the
routing-share arithmetic above), and any burst above that share turns directly into queueing delay on
exactly the requests the classifier decided most needed a careful answer, with nothing to absorb it.
*Load-aware routing* (chosen) has the router read the larger pool's current queue depth alongside the
classifier's recommendation: once it crosses a threshold, turns the classifier only weakly preferred for
the larger model are sent to the default pool instead — a quality trade a user is unlikely to notice on a
borderline turn — while turns the classifier is confident need the larger model still get it as long as
that pool has any room. This spends capacity on the turns that need it most instead of spreading queueing
delay evenly, and keeps the degradation invisible (a marginally less capable answer) rather than visible (a
blown latency target) for as long as the overload is moderate.

Past the point where even the default pool cannot keep queueing delay inside the TTFT budget, the
remaining lever is the one that already showed up in the routing-share arithmetic: admission control caps
how many turns can be in flight (queued or executing) per pool at a small multiple of that pool's
steady-state throughput, and a turn arriving past the cap is told to retry rather than queued
indefinitely — a `503` with a short `Retry-After` computed from the pool's current depth, rather than a
fixed delay. An unbounded queue was rejected for the same reason it is in a pure batching design: past the
cap, the extra wait buys a turn that is already going to miss its own latency target, at the cost of still
occupying a host slot once it is finally served.

**(e) Quotas and abuse.** Each user holds a token bucket refilled continuously against a sustained rate
and a burst allowance, checked at `POST /v1/conversations/{id}/turns`; a request that would drain it below
zero gets `429` with a `Retry-After` computed from the refill rate. The bucket is metered in tokens
actually consumed — input and output tokens charged at 1 unit each, both multiplied by 4 for a turn routed
to the larger model — rather than in raw turn count: a turn count alone caps how often a user can ask, but
not how much each ask costs, and a user who always requests the largest reply the larger model will
produce is spending up to $4\times$ the host-time of an ordinary default-model turn of the same input
size, exactly the multiplier the routing-share arithmetic above already prices; metering what the
multiplier actually costs is what keeps one user's demand on GPU time bounded, not just their request
count.

A per-user limit alone is not enough against an attacker willing to create many accounts to multiply it:
the same admission path also tracks a coarser signal — requests per device fingerprint or IP range — and
tightens further, or requires additional verification, when that signal is far outside the range a single
person's usage would plausibly produce, independent of how many distinct `user_id`s sit behind it.
Exhausting the ordinary per-user quota fails the turn outright, but exhausting only the larger-model share
of it does not: the router treats it the same as an overload signal (deep dive (d)) and silently continues
the turn on the default model rather than failing it, since a slightly less capable answer is a better
outcome for both the user and the system than a hard stop — the same fallback path overload already
exercises.

### Follow-ups

- **Multi-region and data residency.** Partition conversation storage by a region tied to where the user
  signed up, and keep that user's data resident there behind a single global API domain; the prefill and
  decode pools then need a region-aware router too, and a travelling user's own history read from a
  different region simply costs a cross-region read, not a correctness problem.
- **Tool use and grounding with search.** Add a step to a turn's lifecycle where the model emits a
  structured tool call instead of a token; the orchestrator, not a model host, executes it against a search
  backend and folds the result back into context for a further prefill-and-decode pass on the same turn —
  the TTFT target then has to be restated against the first genuinely generated token, not the tool round
  trip.
- **Per-user memory.** Facts about a user that persist beyond one conversation belong in their own small
  store keyed by `user_id`, curated the same way history summarisation is, and folded into the
  system-prompt portion of context assembly — exactly the part already treated as the most cacheable
  prefix, so memory does not by itself change the prefix-caching arithmetic above as long as it changes
  slowly compared to one session.
- **Cost per turn.** Follows directly from the numbers above: a default-model turn's host-time is
  proportional to its 2,000 prefill and 400 decode tokens at the default per-host rates, and a
  larger-model turn 4 times that; multiplying by a fleet's cost per host-hour and dividing by the
  tokens-per-second-per-host figures turns into a cost per turn, and the routing-share formula turns that
  into a cost per day at any chosen routing split.
- **Privacy controls for feedback used in training.** A thumbs-up or thumbs-down is stored against the
  message and conversation it refers to, so using it in training means the export feeding that pipeline has
  to carry a reference back to the source message rather than a detached copy, and honour the same
  deletion that removes the `Message` row it was collected from, rather than surviving it as an orphaned
  record.

<details>
<summary>Estimate check (runnable)</summary>

```python
import math
import random

# ---- turn rate ----
dau = 60_000_000
turns_per_user_per_day = 12
total_turns_per_day = dau * turns_per_user_per_day
assert total_turns_per_day == 720_000_000

avg_turns_per_s = total_turns_per_day / 86_400
assert round(avg_turns_per_s) == 8_333

peak_factor = 3
peak_turns_per_s = avg_turns_per_s * peak_factor
assert round(peak_turns_per_s) == 25_000

# ---- prefill and decode tokens/s at peak, and the hosts each pool needs ----
input_tokens_per_turn = 2_000
output_tokens_per_turn = 400

peak_prefill_tokens_per_s = peak_turns_per_s * input_tokens_per_turn
assert round(peak_prefill_tokens_per_s / 1e6, 1) == 50.0

peak_decode_tokens_per_s = peak_turns_per_s * output_tokens_per_turn
assert round(peak_decode_tokens_per_s / 1e6, 1) == 10.0

prefill_tokens_per_s_per_host = 50_000
decode_tokens_per_s_per_host = 5_000

prefill_hosts_floor = math.ceil(peak_prefill_tokens_per_s / prefill_tokens_per_s_per_host)
assert prefill_hosts_floor == 1_000

decode_hosts_floor = math.ceil(peak_decode_tokens_per_s / decode_tokens_per_s_per_host)
assert decode_hosts_floor == 2_000

headroom = 0.2
prefill_hosts = math.ceil(prefill_hosts_floor * (1 + headroom))
assert prefill_hosts == 1_200

decode_hosts = math.ceil(decode_hosts_floor * (1 + headroom))
assert decode_hosts == 2_400

# a turn's own prefill, run alone with nothing queued ahead of it -- the compute floor under TTFT
prefill_floor_s = input_tokens_per_turn / prefill_tokens_per_s_per_host
assert round(prefill_floor_s * 1000) == 40

# ---- routing a share of turns to the larger (4x host-time per token) model ----
def hosts_with_routing(base_hosts_floor: float, large_share: float, cost_multiplier: float = 4.0) -> float:
    """base_hosts_floor hosts serve every turn at the default model's cost; routing a `large_share`
    fraction of turns to a model that costs `cost_multiplier` times as much host-time per token
    multiplies the floor by (1 - large_share) + large_share * cost_multiplier."""
    return base_hosts_floor * ((1 - large_share) + large_share * cost_multiplier)

assert hosts_with_routing(1_000, 0.0) == 1_000
for share in (0.05, 0.1, 0.2, 0.5, 1.0):
    assert math.isclose(hosts_with_routing(1_000, share), 1_000 * (1 + 3 * share))

large_share_example = 0.10
prefill_hosts_floor_at_share = hosts_with_routing(prefill_hosts_floor, large_share_example)
assert prefill_hosts_floor_at_share == 1_300

decode_hosts_floor_at_share = hosts_with_routing(decode_hosts_floor, large_share_example)
assert decode_hosts_floor_at_share == 2_600

prefill_hosts_at_share = math.ceil(prefill_hosts_floor_at_share * (1 + headroom))
assert prefill_hosts_at_share == 1_560

decode_hosts_at_share = math.ceil(decode_hosts_floor_at_share * (1 + headroom))
assert decode_hosts_at_share == 3_120

# ---- storage per day: conversation text and images ----
bytes_per_token = 4                     # assumption: ~4 bytes of UTF-8 text per token
row_overhead_bytes = 50                 # assumption: message_id16 + conversation_id16 + turn_seq8 + created_at8 + role2
avg_new_message_tokens = 40             # assumption: the user's freshly typed message, not the reconstructed context
avg_reply_tokens = output_tokens_per_turn

user_row_bytes = row_overhead_bytes + avg_new_message_tokens * bytes_per_token
assistant_row_bytes = row_overhead_bytes + avg_reply_tokens * bytes_per_token
assert user_row_bytes == 210
assert assistant_row_bytes == 1_650

text_bytes_per_turn = user_row_bytes + assistant_row_bytes
assert text_bytes_per_turn == 1_860

daily_text_bytes = total_turns_per_day * text_bytes_per_turn
daily_text_tb = daily_text_bytes / 1e12
assert round(daily_text_tb, 2) == 1.34

image_turn_fraction = 0.10
avg_image_bytes = 1_000_000             # 1 MB
daily_image_turns = total_turns_per_day * image_turn_fraction
assert daily_image_turns == 72_000_000

daily_image_bytes = daily_image_turns * avg_image_bytes
daily_image_tb = daily_image_bytes / 1e12
assert daily_image_tb == 72.0

assert round(daily_image_tb / daily_text_tb) == 54

print("all requirements-and-scale numbers check out")


# ---- prefix-cache savings in context assembly ----
def effective_prefill_tokens(hit_rate: float, new_message_tokens: int = avg_new_message_tokens,
                              full_context_tokens: int = input_tokens_per_turn) -> float:
    """On a cache hit only the newly appended tokens need prefilling, since the rest of the context's
    key/value state carries over from the previous turn; on a miss the whole context is prefilled cold."""
    return hit_rate * new_message_tokens + (1 - hit_rate) * full_context_tokens

assert effective_prefill_tokens(0.0) == input_tokens_per_turn
assert effective_prefill_tokens(1.0) == avg_new_message_tokens

example_hit_rate = 0.8
effective = effective_prefill_tokens(example_hit_rate)
assert round(effective) == 432  # NOTE: round(), not == 432 -- (1 - 0.8) is not exact in float64
reduction = 1 - effective / input_tokens_per_turn
assert round(reduction, 2) == 0.78

# ---- output-classifier holdback window, as a fraction of a second at the streaming floor ----
classifier_span_tokens = 10
streaming_floor_tokens_per_s = 30
holdback_s = classifier_span_tokens / streaming_floor_tokens_per_s
assert round(holdback_s, 2) == 0.33

print("context-assembly and safety-holdback numbers check out")


# ---- resume-by-offset streaming: a dropping, reconnecting client sees every token exactly once ----
class TurnBuffer:
    """Server-side per-turn token buffer backing the resumable stream: append-only for the life of the
    turn, readable from any offset by any orchestrator instance."""

    def __init__(self) -> None:
        self.tokens: list[str] = []

    def append(self, token: str) -> None:
        self.tokens.append(token)

    def read_from(self, offset: int) -> list[str]:
        return self.tokens[offset:]


def simulate_turn(seed: int, n_tokens: int = 50, p_drop: float = 0.15, p_reconnect: float = 0.5):
    """Generates n_tokens one at a time into a TurnBuffer -- generation never pauses for the client's
    connection state -- while a client drops and reconnects at random points, resuming from its own
    offset; returns the client's reassembled transcript plus counts confirming both the live-delivery
    and the catch-up-on-reconnect paths were actually exercised."""
    rng = random.Random(seed)
    buf = TurnBuffer()
    received: list[str] = []
    offset = 0
    connected = True
    live_delivered = 0
    catchup_lengths: list[int] = []

    for i in range(n_tokens):
        token = f"tok{i}"
        buf.append(token)  # NOTE: append happens before delivery, so a mid-turn crash never drops a token
        if connected:
            if rng.random() < p_drop:
                connected = False                 # drops right as this token would have been sent
            else:
                received.append(token)
                offset += 1
                live_delivered += 1
        elif rng.random() < p_reconnect:
            connected = True
            catch_up = buf.read_from(offset)      # never asks for regeneration, only a buffer read
            if catch_up:
                catchup_lengths.append(len(catch_up))
            received.extend(catch_up)
            offset += len(catch_up)

    if offset < len(buf.tokens):                  # the reconnect every real client eventually makes
        catch_up = buf.read_from(offset)
        if catch_up:
            catchup_lengths.append(len(catch_up))
        received.extend(catch_up)
        offset += len(catch_up)

    return received, buf.tokens, live_delivered, catchup_lengths


total_live, total_catchups, max_catchup = 0, 0, 0
for seed in range(40):
    received, generated, live, catchups = simulate_turn(seed)
    assert received == generated, seed            # every token exactly once, in order, despite drops
    total_live += live
    total_catchups += len(catchups)
    max_catchup = max(max_catchup, max(catchups, default=0))

assert total_live > 0                              # some tokens were genuinely delivered live
assert total_catchups > 0                          # some reconnects genuinely had to catch up
assert max_catchup > 1                             # at least one catch-up spanned more than one token

print("resume-by-offset simulation confirms exactly-once, in-order delivery across reconnects")
print("all checks passed")
```

</details>

</details>
