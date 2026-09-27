# Most Frequent Events in a Stream

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · hash maps, sliding windows, streaming | ★★★★☆ | Medium | SWE · MLE · Applied AI · Intern | hash-map, sliding-window, frequency-counting, streaming, space-saving, heavy-hitters | 3 parts / 45 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

An *event* is a pair `(message: str, timestamp: int)`; timestamps are non-decreasing in the input
order, as in a log file or a real event stream.

### Part 1 — The most frequent message and its timestamps

```py
def most_frequent(events: list[tuple[str, int]]) -> tuple[str, list[int]]: ...
```

Given a fixed, non-empty list of events, return the message that occurs most often, together with
the timestamps of its occurrences, in the order they appear in `events`. If two or more messages
tie for the highest count, return the one whose first occurrence in `events` comes earliest.
`events` is never empty when other parts of this page call `most_frequent`; called directly on `[]`
it raises `ValueError`.

```text
events = [("login", 1), ("click", 2), ("login", 4), ("click", 7), ("purchase", 9)]

most_frequent(events) == ("login", [1, 4])
# "login" and "click" both occur twice, the maximum; "login"'s first occurrence (timestamp 1) comes
# before "click"'s (timestamp 2), so "login" wins the tie. "purchase" occurs once and does not matter.
```

### Part 2 — The same question over a sliding window

Events now arrive one at a time. `WindowTopMessage` holds only the last `n` events it has seen:
when a new event arrives and `n` events are already held, the oldest held event — the one that
arrived first among those still held — is evicted before the new one is inserted. `n` is a positive
integer, fixed for the object's lifetime. The *count* of a message is the number of currently held
events carrying it.

```py
class WindowTopMessage:
    def __init__(self, n: int): ...
    def add(self, message: str, timestamp: int) -> tuple[str, int]: ...   # (message, count) after the arrival
```

After every `add`, return the message with the highest count and that count. Ties are broken by
recency of change, made precise as follows: number every individual count change across the whole
run — one decrement for each eviction, one increment for each arrival's insertion — in the order
they happen (an arrival's own eviction, if any, is numbered before that same arrival's insertion).
Among the messages sharing the maximum count, return the one whose most recent change has the
highest number. Because an eviction is always numbered before its arrival's insertion, an arriving
message that ties for the maximum is always the one `add` returns. Every operation must be $O(1)$
amortised, independent of `n` and of the number of distinct messages ever seen.

Traced with `n = 3`:

```text
add("login", 1)      -> ("login", 1)      # held: login
add("click", 2)      -> ("click", 1)      # held: login, click             (tie at 1; click changed last)
add("login", 3)      -> ("login", 2)      # held: login, click, login
add("purchase", 4)   -> ("purchase", 1)   # evicts login@1; held: click, login, purchase
                                           # login drops to count 1, tying click and purchase at 1;
                                           # the maximum falls from 2 to 1, and purchase -- the
                                           # arrival -- wins the tie
add("purchase", 6)   -> ("purchase", 2)   # evicts click@2; held: login, purchase, purchase
```

### Part 3 — Unbounded stream, bounded memory

The stream is now unbounded, arriving as one message at a time with no timestamp, and memory holds
counters for at most `k` distinct messages at once — the Space-Saving algorithm. `SpaceSaving`
starts with nothing monitored; whenever an unmonitored message arrives and `k` messages are already
monitored, the monitored message with the lowest counter is evicted to make room — ties broken in
favour of whichever tied message has held that counter value the longest — and the arriving
message's counter is seeded from the evicted one's count rather than starting at zero (the exact
seeding rule is derived, not asserted, in the solution). `k` is a positive integer, fixed for the
object's lifetime.

```py
class SpaceSaving:
    def __init__(self, k: int): ...
    def add(self, message: str) -> None: ...
    def top(self, m: int) -> list[tuple[str, int]]: ...   # monitored messages by reported count, descending; ties by message
```

Write $f(x)$ for the true number of times message $x$ has occurred among the $N$ messages `add` has
processed so far, and $c(x)$ for the counter `SpaceSaving` reports for $x$ while it is monitored.
The implementation must guarantee, at every point in the stream:

- *Heavy-hitter coverage.* Every message with $f(x) > N / k$ — a *heavy hitter* — is among the
  monitored messages.
- *Bounded overestimation.* For every monitored message $x$, $f(x) \le c(x) \le f(x) + N / k$.

`top(m)` returns the `m` monitored messages with the highest reported count, in descending order;
messages tied on count are ordered ascending by the message string. If fewer than `m` messages are
monitored, every monitored message is returned.

Traced with `k = 3`, messages arriving in the order `a, b, c, d, b, a, e`:

```text
arrives   evicted    monitored afterwards (reported count)
a         --         a: 1
b         --         a: 1, b: 1
c         --         a: 1, b: 1, c: 1                      # table now full
d         a (@1)     b: 1, c: 1, d: 2                      # d inherits a's count of 1, plus this occurrence
b         --         b: 2, c: 1, d: 2                      # b already monitored -- plain increment
a         c (@1)     b: 2, d: 2, a: 2                      # a inherits c's count of 1, plus this occurrence
e         d (@2)     b: 2, a: 2, e: 3                      # e inherits d's count of 2, plus this occurrence

top(3) == [("e", 3), ("a", 2), ("b", 2)]
# true counts over these 7 arrivals: a: 2, b: 2, c: 1, d: 1, e: 1 (N = 7, N / k = 2.33)
# every monitored message satisfies f <= c <= f + N / k; "e" is the tightest: 1 <= 3 <= 3.33
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points are worth confirming before coding: Part 1's tie-break is by earliest first occurrence,
not by the message string or by insertion order of the tie itself; and Part 2's "most frequent" is
recomputed exactly after every single arrival and eviction, not a smoothed or decayed rate.

### Part 1

One pass accumulates, for every message, its running count, the list of timestamps it has occurred
at (already in order, since `events` is scanned left to right), and the index of its first
occurrence — recorded once via `dict.setdefault`, which only inserts a value for a key that is not
already present, so a later occurrence never overwrites it. The winner is the message with the
largest `(count, -first_index)` pair: count first, and, among ties, the smallest first-occurrence
index, negated so that `max` still finds it. No two messages can ever share both coordinates — each
occupies a distinct position in `events`, hence a distinct first index — so this comparison never
needs a further tie-break.

```python
def most_frequent(events: list[tuple[str, int]]) -> tuple[str, list[int]]:
    if not events:
        raise ValueError("events must be non-empty")
    counts: dict[str, int] = {}
    timestamps: dict[str, list[int]] = {}
    first_index: dict[str, int] = {}
    for i, (message, ts) in enumerate(events):
        counts[message] = counts.get(message, 0) + 1
        timestamps.setdefault(message, []).append(ts)
        first_index.setdefault(message, i)   # NOTE: setdefault -- keeps the first occurrence, never overwrites it
    best = max(counts, key=lambda m: (counts[m], -first_index[m]))
    return best, timestamps[best]
```

The loop is $O(N)$ for $N$ events, and the three dictionaries together hold $O(N)$ entries; the
final `max` scans at most $N$ distinct messages, which does not change that bound.

### Part 2

Recomputing the maximum by scanning every held message after each arrival would cost $O(n)$ per
`add`. Reaching $O(1)$ needs two more pieces of state, kept incrementally: `_count`, the current
count of every held message, and `_bucket`, a *count of counts* — `_bucket[c]` is the set of
messages currently at count `c`, held as a `dict` used purely as an ordered set (`dict[str, None]`),
so that inserting a key appends it after every key already there. `_max_count` caches the current
maximum so it is never found by scanning.

An arrival always increments exactly one message's count by one (`_increment`): remove it from
`_bucket[old]` (dropping that bucket if it is now empty), add it to `_bucket[new]`, and raise
`_max_count` if `new` exceeds it. `_max_count` can only go up here, never down.

An eviction always decrements exactly one message's count by one (`_decrement`), which is the
direction that needs an argument. If the evicted message's old count was below `_max_count`, or
tied at `_max_count` with a survivor still there, the maximum is unaffected. The only case where
`_max_count` must change is when the evicted message was the sole occupant of
`_bucket[_max_count]`, emptying it — and even then, the new maximum can only be `_max_count - 1`:
the decremented message itself lands, in that same operation, in `_bucket[_max_count - 1]`, unless
its new count is `0`, which only happens when `_max_count` was `1` and the window has just become
empty of tracked messages, a state the arrival's own increment repairs a few lines later in `add`.
So `_max_count` always drops by exactly one here, never by scanning for the true new maximum.

The tie-break — report the message whose count changed most recently — is why each bucket is a
`dict` and not a plain `set`: every insertion into a bucket happens at its end (`dict[message] =
None` on a key not already present appends it), so the most-recently-changed message among a tied
group is `next(reversed(_bucket[_max_count]))`, an $O(1)$ read of a dict's last key that does not
scan the bucket. A superficially similar but different rule — tie-break by whichever message's most
recent *occurrence timestamp* is the latest, rather than by whichever count changed most recently —
would break this: an evicted message's occurrence timestamp does not change when it is decremented
(it did not just occur; it is leaving), so it would have to be reinserted into `_bucket[old - 1]` at
the position matching that old timestamp among whatever is already there — the middle of the
bucket's order, in general, not the end — and finding that position is no longer $O(1)$.

`add` evicts, if the window is full, before it inserts, matching the statement's tie rule directly:
the arriving message's increment is always the most recent change at the moment `add` returns, so
it wins any tie for the maximum it takes part in.

```python
from collections import deque


class WindowTopMessage:
    """Reports the most frequent message among the last n held events after every arrival."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._window: deque[tuple[str, int]] = deque()   # held events, oldest first
        self._count: dict[str, int] = {}
        self._bucket: dict[int, dict[str, None]] = {}    # count -> ordered set of messages at that count
        self._max_count = 0

    def _increment(self, message: str) -> None:
        old = self._count.get(message, 0)
        new = old + 1
        if old:
            del self._bucket[old][message]
            if not self._bucket[old]:
                del self._bucket[old]
        self._count[message] = new
        self._bucket.setdefault(new, {})[message] = None   # NOTE: appended last -- most recently changed
        if new > self._max_count:
            self._max_count = new

    def _decrement(self, message: str) -> None:
        old = self._count[message]
        new = old - 1
        del self._bucket[old][message]
        emptied = not self._bucket[old]
        if emptied:
            del self._bucket[old]        # NOTE: drop the empty bucket, or a later self._max_count could
                                          # point at a bucket that no longer holds any message
        if new:
            self._count[message] = new
            self._bucket.setdefault(new, {})[message] = None
        else:
            del self._count[message]
        if old == self._max_count and emptied:
            self._max_count -= 1         # NOTE: derived in the text -- the new maximum is exactly old - 1

    def add(self, message: str, timestamp: int) -> tuple[str, int]:
        if len(self._window) == self._n:
            evicted_message, _ = self._window.popleft()   # NOTE: eviction is numbered before insertion
            self._decrement(evicted_message)
        self._window.append((message, timestamp))
        self._increment(message)
        top = next(reversed(self._bucket[self._max_count]))
        return top, self._max_count
```

Every operation `add` performs — a handful of dict lookups, insertions and deletions, one `deque`
push and at most one pop, and one `next(reversed(...))` — is $O(1)$ amortised: a dict's get, set and
delete are $O(1)$ amortised by construction, and reading its most-recently-inserted key costs no
more, since it is read directly off the end of the dict's own entry order rather than found by
scanning. None of this depends on `n` or on how many distinct messages have ever been seen.

### Part 3

`SpaceSaving` reuses Part 2's count-of-counts bucket technique to track a minimum instead of a
maximum, but an eviction here does not decrement a survivor — it replaces one monitored message
with a different one, seeding the newcomer's counter from the evicted one.

Both guarantees follow from two facts about the counters.

First, every `add` increases the sum of all counters by exactly $1$, whichever branch it takes:
incrementing an existing counter adds $1$; starting a fresh one at $1$ adds $1$; and evicting
removes a counter of value $m$ (the minimum) and adds one of $m + 1$, a net $+1$ again. So after $N$
calls, the (at most $k$) counters sum to exactly $N$, and once the table is full, the minimum of $k$
numbers summing to $N$ is at most $N / k$, by the pigeonhole principle.

Second, a monitored counter never falls below its message's true count so far, and an unmonitored
message's true count so far never exceeds the current minimum monitored counter. Both hold at the
start (nothing monitored, nothing occurred) and survive every update: incrementing a monitored
message keeps pace with its own new occurrence one for one; a fresh counter starts at $1$ against a
true prior count of exactly $0$, since a message can only be unmonitored with no room left once
some eviction has already happened, and none happens while room remains; and evicting a message $z$
at counter value $m$ — by the same claim applied one step earlier, already at least $z$'s true
count — to seed $x$ at $m + 1$ leaves $x$'s counter one ahead of its own fresh occurrence, while
$z$'s true count, still at most $m$, is now measured against a minimum no lower than $m$.

Both guarantees follow directly. If some message's true count exceeded $N / k$ while it stayed
unmonitored, its true count would have to be at most the minimum monitored counter, at most
$N / k$ — a contradiction; so every message with true count above $N / k$ is monitored. For a
monitored message $x$: let $t_0$ be the step at which $x$ was most recently inserted, and
$m \le N / k$ the table's minimum there. From $t_0$ onward, `add` increments $x$'s counter once for
every true occurrence of $x$, including the one at $t_0$ itself, so $x$'s final counter is $m$ plus
that count of occurrences; $x$'s true total count is at least that same count (any occurrences
before $t_0$ only add to it), so the counter exceeds the truth by at most $m \le N / k$ — while
$f(x) \le c(x)$ is already the first fact above.

```python
class SpaceSaving:
    """Space-Saving with a stream-summary structure: buckets keyed by counter value, each an ordered
    set of the messages currently holding that value, plus O(1) access to the minimum bucket."""

    def __init__(self, k: int) -> None:
        self._k = k
        self._count: dict[str, int] = {}
        self._bucket: dict[int, dict[str, None]] = {}
        self._min_count = 0

    def _reinsert(self, message: str, new_count: int) -> None:
        old_count = self._count.get(message)
        if old_count is not None:
            del self._bucket[old_count][message]
            if not self._bucket[old_count]:
                del self._bucket[old_count]
        self._count[message] = new_count
        self._bucket.setdefault(new_count, {})[message] = None

    def add(self, message: str) -> None:
        if message in self._count:
            old = self._count[message]
            self._reinsert(message, old + 1)
            if old == self._min_count and not self._bucket.get(old):
                self._min_count = old + 1   # NOTE: this message just left the sole occupant of the minimum bucket
            return
        if len(self._count) < self._k:
            self._reinsert(message, 1)
            self._min_count = 1             # NOTE: a fresh counter starts at 1, the least any counter can be
            return
        m = self._min_count
        evicted = next(iter(self._bucket[m]))   # NOTE: longest-held at the minimum -- the statement's tie-break
        del self._count[evicted]
        del self._bucket[m][evicted]
        if not self._bucket[m]:
            del self._bucket[m]
            self._min_count = m + 1         # NOTE: derived in the text -- the arriving message witnesses m + 1
        self._reinsert(message, m + 1)

    def top(self, m: int) -> list[tuple[str, int]]:
        ranked = sorted(self._count.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:m]
```

Every `add` does a handful of dict operations plus, on the eviction path, one `next(iter(...))`
pulling an arbitrary element out of the minimum bucket — $O(1)$ amortised, exactly as in Part 2, and
for the same reason. `top(m)` sorts the at-most-$k$ monitored messages, $O(k \log k)$, independent
of how long the stream has been running.

A heap of `(count, message)` pairs with lazy deletion — push a fresh pair on every change, skip
stale ones on pop — offers $O(\log k)$ per `add` more simply, at the cost of growing past $k$
entries between clean-ups; the bucket structure above avoids that, never holding more than $k$
counters, matching the statement's bounded memory exactly.

### Follow-ups

- **Misra–Gries.** Keeping $k - 1$ counters and, on a full table, decrementing every one of them by
  one (discarding any that reach zero) instead of promoting the evicted counter's value to the
  newcomer, makes Misra–Gries the under-counting dual of Space-Saving: every monitored counter
  satisfies $c(x) \le f(x)$, and, symmetrically, every message with true count above $N / k$ is
  again guaranteed to be monitored.
- **Count-Min sketch.** Dropping message identities from the sketch entirely — $d$ independent hash
  functions each map every message into one of $w$ counters, incremented on every occurrence, and
  the estimate for a message is the minimum of its $d$ counters, since a hash collision can only
  inflate a count, never deflate it — trades the exact top-$k$ list for a point-query estimator:
  with $w = \lceil e / \varepsilon \rceil$ and $d = \lceil \ln(1 / \delta) \rceil$, the estimate
  exceeds the truth by more than $\varepsilon N$ with probability at most $\delta$. Recovering
  *which* messages are frequent needs a separate structure, such as Space-Saving, fed by the same
  stream.
- **Merging summaries from several machines.** Combining two machines' Space-Saving tables is not
  just summing the counters they share: a message monitored on only one machine may have occurred,
  unrecorded, on the other, so each machine's contribution is bumped up by that machine's own error
  bound ($N_{\text{machine}} / k$) before the counters are added — a tighter per-machine bound
  traded for a looser combined one.
- **Time-based windows.** Replacing the fixed count `n` with a fixed duration — the last five
  minutes rather than the last `n` events — still evicts from the front of a deque ordered by
  timestamp, but a single arrival can now evict zero, one, or several events at once (everything
  that has aged out), instead of exactly one. `_decrement`'s bucket bookkeeping is unchanged and
  simply runs once per evicted event, so the whole structure is still $O(1)$ amortised per event,
  just not per `add` call.
- **Detecting a sudden spike.** Running two windows side by side, a short one (the last minute) and
  a long one (the last hour), and comparing a message's short-window rate against its long-window
  rate catches a message that is suddenly trending, independent of its overall popularity — a plain
  single-window count cannot distinguish a spike from a message that is simply always near the top.

<details>
<summary>Checks (runnable)</summary>

```python
import random
from collections import Counter

# --- Part 1: the worked example ---
events_ex = [("login", 1), ("click", 2), ("login", 4), ("click", 7), ("purchase", 9)]
assert most_frequent(events_ex) == ("login", [1, 4])

try:
    most_frequent([])
    assert False, "expected ValueError"
except ValueError:
    pass


def _brute_force_most_frequent(events):
    """Independent restatement of Part 1's rule: count with collections.Counter, break ties by the
    smallest index of first occurrence, then collect that message's timestamps in input order."""
    if not events:
        raise ValueError("events must be non-empty")
    counts = Counter(m for m, _ in events)
    first_index = {}
    for i, (m, _) in enumerate(events):
        first_index.setdefault(m, i)
    best_count = max(counts.values())
    candidates = [m for m, c in counts.items() if c == best_count]
    winner = min(candidates, key=lambda m: first_index[m])
    return winner, [ts for msg, ts in events if msg == winner]


for seed in range(400):
    rng = random.Random(seed)
    alphabet = [f"m{i}" for i in range(rng.randint(1, 6))]
    ts = 0
    evs = []
    for _ in range(rng.randint(1, 30)):
        ts += rng.randint(0, 3)
        evs.append((rng.choice(alphabet), ts))
    assert most_frequent(evs) == _brute_force_most_frequent(evs), (seed, evs)

# --- Part 2: the worked example, traced step by step (n = 3) ---
wtm = WindowTopMessage(3)
trace = [
    ("login", 1, ("login", 1)),
    ("click", 2, ("click", 1)),
    ("login", 3, ("login", 2)),
    ("purchase", 4, ("purchase", 1)),   # evicts login@1 -- the maximum falls from 2 to 1
    ("purchase", 6, ("purchase", 2)),   # evicts click@2
]
for message, timestamp, expected in trace:
    assert wtm.add(message, timestamp) == expected, (message, timestamp)


def _brute_force_window(events, n):
    """Independent restatement of Part 2's rule: after every arrival, recompute counts from scratch
    over the currently held events, and track each message's step of last change -- incremented on
    every arrival and every eviction that involves it -- completely separately from WindowTopMessage."""
    window = deque()
    last_change_step = {}
    step = 0
    results = []
    for message, timestamp in events:
        if len(window) == n:
            evicted, _ = window.popleft()
            step += 1
            last_change_step[evicted] = step
        window.append((message, timestamp))
        step += 1
        last_change_step[message] = step
        counts = Counter(m for m, _ in window)
        best_count = max(counts.values())
        candidates = [m for m, c in counts.items() if c == best_count]
        results.append((max(candidates, key=lambda m: last_change_step[m]), best_count))
    return results


total_steps = 0
for seed in range(80):
    rng = random.Random(1000 + seed)
    alphabet = [f"m{i}" for i in range(rng.randint(1, 5))]
    n = rng.choice([1, 2, 3, 5, 8, 13])
    ts = 0
    evs = []
    for _ in range(rng.randint(50, 150)):
        ts += rng.randint(0, 2)
        evs.append((rng.choice(alphabet), ts))
    expected = _brute_force_window(evs, n)
    wtm2 = WindowTopMessage(n)
    got = [wtm2.add(m, t) for m, t in evs]
    assert got == expected, (seed, n)
    total_steps += len(evs)
assert total_steps > 2000

# --- Part 3: the worked example, traced state by state (k = 3) ---
ss = SpaceSaving(3)
trace3 = [
    ("a", {"a": 1}),
    ("b", {"a": 1, "b": 1}),
    ("c", {"a": 1, "b": 1, "c": 1}),          # table now full
    ("d", {"b": 1, "c": 1, "d": 2}),          # evicts a@1 -- d inherits count 1, plus this occurrence
    ("b", {"b": 2, "c": 1, "d": 2}),          # b already monitored -- plain increment
    ("a", {"b": 2, "d": 2, "a": 2}),          # evicts c@1 -- a inherits count 1, plus this occurrence
    ("e", {"b": 2, "a": 2, "e": 3}),          # evicts d@2 -- e inherits count 2, plus this occurrence
]
for message, expected_state in trace3:
    ss.add(message)
    assert dict(ss.top(3)) == expected_state, (message, dict(ss.top(3)), expected_state)
assert ss.top(3) == [("e", 3), ("a", 2), ("b", 2)]

true_counts_ex = Counter(m for m, _ in trace3)
N_ex, k_ex = len(trace3), 3
assert sum(dict(ss.top(3)).values()) == N_ex   # the sum-of-counters invariant, derived in the text
for message, c in ss.top(3):
    f = true_counts_ex[message]
    assert f <= c <= f + N_ex / k_ex

# a table that never fills (k at least the number of distinct messages) counts exactly, no error at all
rng = random.Random(7)
alphabet_small = [f"m{i}" for i in range(12)]
stream_small = [rng.choice(alphabet_small) for _ in range(500)]
ss_exact = SpaceSaving(len(alphabet_small))
for message in stream_small:
    ss_exact.add(message)
assert dict(ss_exact.top(len(alphabet_small))) == dict(Counter(stream_small))
assert len(ss_exact.top(10_000)) == len(set(stream_small))   # top(m) past the monitored count returns all of it


def _zipf_like_stream(rng, n_events, n_messages):
    """A synthetic stream over n_messages distinct names, weighted 1 / rank, so a handful of messages
    dominate -- unlike a uniform stream, and like most real event logs."""
    messages = [f"m{i}" for i in range(n_messages)]
    weights = [1.0 / (i + 1) for i in range(n_messages)]
    return rng.choices(messages, weights=weights, k=n_events)


rng = random.Random(2024)
N = 20_000
stream = _zipf_like_stream(rng, N, n_messages=300)
true_counts = Counter(stream)

for k in (10, 30, 100):
    ss_big = SpaceSaving(k)
    for message in stream:
        ss_big.add(message)
    monitored = ss_big.top(k)               # <= k monitored messages, so top(k) returns all of them
    assert len(monitored) <= k
    for (msg_a, count_a), (msg_b, count_b) in zip(monitored, monitored[1:]):
        assert count_a >= count_b and (count_a > count_b or msg_a < msg_b)   # top() ordering

    threshold = N / k
    monitored_counts = dict(monitored)
    assert sum(monitored_counts.values()) == N              # the sum-of-counters invariant, derived in the text
    assert min(monitored_counts.values()) <= threshold + 1e-9   # the minimum-counter bound, derived in the text
    heavy_hitters = [m for m, f in true_counts.items() if f > threshold]
    assert heavy_hitters   # the check must actually exercise the guarantee, not pass it vacuously
    for message in heavy_hitters:
        assert message in monitored_counts, (k, message, true_counts[message], threshold)
    for message, c in monitored_counts.items():
        f = true_counts.get(message, 0)
        assert f <= c <= f + threshold + 1e-9, (k, message, f, c, threshold)

print("all checks passed")
```

</details>

</details>
