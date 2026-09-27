# Snapshot Array: History, Compaction and Diffs

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · hash maps, binary search, space-time trade-offs | ★★★☆☆ | Medium | SWE · RE · MLE · Intern | hash-map, binary-search, versioning, memory-trade-offs, journaling | 3 parts / 45 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

A *snapshot array* of length `length` holds `length` integers, indexed `0` to `length - 1`, every one of
them `0` at the start. A *snapshot* is a read-only record of the array's entire contents, taken by an
explicit call and identified by a *snapshot id*: the first snapshot taken has id `0`, the second `1`, and
so on, in the order snapshots are actually taken. Between two snapshots, any number of writes may happen to
any index; reading a given snapshot id later must return exactly the value its index held at the moment
that snapshot was taken, regardless of what has been written since.

### Part 1 — Snapshots

```py
class SnapshotArray:
    def __init__(self, length: int) -> None: ...           # every index starts at 0
    def set(self, index: int, val: int) -> None: ...
    def snap(self) -> int: ...                              # returns the id just taken: 0, 1, 2, ...
    def get(self, index: int, snap_id: int) -> int: ...     # value at index when snap_id was taken
```

`set` and `get` raise `IndexError` for an `index` outside `[0, length)`; `get` also raises `IndexError` for
a `snap_id` that `snap` has not yet returned, including a negative one. `get` never returns a value that
has been `set` but not yet covered by a `snap` — such a write is invisible to every `get` until the next
`snap` call seals it into a new snapshot. Required costs: `snap` is $O(1)$; `get` is $O(\log s)$, where $s$
is the number of times the queried index has been written, never the number of snapshots taken and never
`length`; memory is $O(1)$ for every index that is never written, plus $O(1)$ for every call to `set`.

Traced, with `length = 4`:

```text
arr = SnapshotArray(4)                # [0, 0, 0, 0]
arr.set(0, 5)
arr.set(0, 6)                         # overwrites the pending write of 5 -- see below
s0 = arr.snap()                       # -> 0   snapshot 0 = [6, 0, 0, 0]
arr.set(1, 9)
arr.set(0, 1)
s1 = arr.snap()                       # -> 1   snapshot 1 = [1, 9, 0, 0]
arr.set(0, 4)
arr.set(2, 2)
s2 = arr.snap()                       # -> 2   snapshot 2 = [4, 9, 2, 0]
s3 = arr.snap()                       # -> 3   snapshot 3 = [4, 9, 2, 0]   (nothing set since snapshot 2)
s4 = arr.snap()                       # -> 4   snapshot 4 = [4, 9, 2, 0]   (nothing set since snapshot 2)
arr.set(0, 7)
s5 = arr.snap()                       # -> 5   snapshot 5 = [7, 9, 2, 0]

arr.get(0, 0)   # -> 6   only the later of the two writes before snapshot 0 is ever visible
arr.get(0, 1)   # -> 1
arr.get(0, 4)   # -> 4   index 0 did not change between snapshot 2 and snapshot 4
arr.get(0, 5)   # -> 7
arr.get(1, 0)   # -> 0   index 1 is not written until after snapshot 0
arr.get(3, 5)   # -> 0   index 3 is never written at all
```

### Part 2 — Releasing old snapshots

```py
    def release(self, snap_id: int) -> None: ...
    def stored_entries(self) -> int: ...
```

`release(snap_id)` records a promise from the caller: neither this call nor any later one will ever again
name a snapshot id `<= snap_id` in a call to `get` (Part 3's `diff`, added below, inherits the same
promise). `snap_id` must already have been returned by `snap`, exactly as for `get`, else `IndexError`.
Calling `release` again later only ever raises this boundary; a `snap_id` at or below one already released
changes nothing. A later `get` for a released snapshot id raises `ValueError`, not `IndexError` — the id
was valid, it has simply become unreadable.

Releasing memory is the entire point: from some finite point after a `release` call, every internal history
must hold, per index, only the entries still needed to answer a `get` for a live (not released) snapshot id
or to know the current value — never entries that answered only released ids. `stored_entries()` returns
the total number of `(snap_id, value)` entries currently held across every index, so that claim can be
measured directly; it costs $O(\text{length})$ to compute and exists only for that measurement.

Traced, continuing the Part 1 example (`arr` after `s5 = 5`, history for index `0` covering ids `0, 1, 2,
5`):

```text
arr.release(2)          # ids 0, 1 and 2 become released
arr.get(0, 1)            # -> ValueError   (1 <= 2)
arr.get(0, 2)            # -> ValueError   (2 <= 2, release is inclusive of its own argument)
arr.get(0, 4)            # -> 4             (4 > 2, still live, answer unchanged)
arr.release(0)            # no-op: 0 is already below the released boundary of 2
```

### Part 3 — Diffs between snapshots

```py
    def diff(self, a: int, b: int) -> dict[int, tuple[int, int]]: ...
```

`a` and `b` must both be live snapshot ids in the sense of Part 2 (`IndexError`/`ValueError` exactly as for
`get`), and `a < b`, else `ValueError`. `diff` returns every index whose value at snapshot `b` differs from
its value at snapshot `a`, mapped to `(value_at_a, value_at_b)`; an index that was written one or more
times between the two snapshots but ended up with the same value at both is absent from the result. This
must run in time proportional to the number of `set` calls recorded strictly after snapshot `a` and up to
snapshot `b`, plus the size of the output — never proportional to `length`.

Traced, continuing the same example:

```text
arr.diff(3, 5)   # -> {0: (4, 7)}   only index 0 changed between snapshots 3 and 5
arr.diff(2, 4)   # -> {}             nothing changed between snapshots 2 and 4
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points are worth confirming before coding: whether `get` may ever expose a write that has not yet been
covered by a `snap` (it may not — a pending write is invisible until the next `snap` seals it); and what
`snap` actually returns and how many snapshots may be taken (ids `0, 1, 2, ...` in call order, with no
upper bound other than memory — `snap`'s return value *is* the id, not a count or a separate token).

### Part 1

Every write is recorded per index, not per snapshot. `_history[index]` is a list of `(snap_id, value)`
pairs in increasing order of `snap_id` — but the `snap_id` a `set` call writes under is not necessarily one
`snap` has returned yet: it is `_next_id`, the id whichever call to `snap` comes *next* will return, i.e.
the snapshot currently being built. A `set` either appends a new pair for `_next_id`, or, if the index's
last recorded pair already carries that same id (an earlier `set` on the same index since the last `snap`),
overwrites it in place: no `get` can ever be asked for anything between two `snap` calls, so only the final
value written before the next one matters, and keeping the earlier ones would only waste memory. `snap`
itself touches no per-index state at all — it only reads and increments `_next_id` — which is exactly why
it costs $O(1)$ regardless of `length` or how many indices have ever been written. `get` locates the entry
that applied at `snap_id` with `bisect_right`, keyed by each pair's id: the insertion point for `snap_id`
among the stored ids sits one past every id `<= snap_id`, so subtracting `1` lands on the *last* (highest)
id that is still `<= snap_id` — the value recorded there is exactly what was current at `snap_id`. An
insertion point of `0` means no recorded id is `<= snap_id`, so the index was never written before that
snapshot and its value is still the initial `0`.

`_history` is a `dict`, populated lazily with `setdefault`, rather than a list preallocated to `length`: an
index that is never written is then simply never a key, costing nothing at all, which is a strictly smaller
footprint than a preallocated list of `length` empty containers (itself already an acceptable, if slightly
larger, constant-per-index cost).

```python
from bisect import bisect_right


class SnapshotArray:
    """array[i] is 0 until set. get(i, s) reads what array[i] held when snapshot s was taken; a write
    made after s, or not yet covered by any snap at all, is never visible to it."""

    def __init__(self, length: int) -> None:
        self._length = length
        self._next_id = 0                                     # id whichever snap() call is next will return
        self._history: dict[int, list[tuple[int, int]]] = {}  # index -> [(snap_id, value), ...], lazy

    def _check_index(self, index: int) -> None:
        if not (0 <= index < self._length):
            raise IndexError(f"index {index} out of range [0, {self._length})")

    def _check_taken(self, snap_id: int) -> None:
        if not (0 <= snap_id < self._next_id):
            raise IndexError(f"snapshot {snap_id} was never taken")

    def set(self, index: int, val: int) -> None:
        self._check_index(index)
        h = self._history.setdefault(index, [])
        if h and h[-1][0] == self._next_id:
            h[-1] = (self._next_id, val)    # NOTE: overwrite -- an earlier set in this same pending
        else:                               #      interval can never be read by any snapshot
            h.append((self._next_id, val))

    def snap(self) -> int:
        snap_id = self._next_id
        self._next_id += 1
        return snap_id

    def get(self, index: int, snap_id: int) -> int:
        self._check_index(index)
        self._check_taken(snap_id)
        h = self._history.get(index)
        if not h:
            return 0
        pos = bisect_right(h, snap_id, key=lambda e: e[0]) - 1  # NOTE: -1 -- bisect_right lands one past
        return h[pos][1] if pos >= 0 else 0                     #      every id <= snap_id; back up one
```

`set` is $O(1)$: it inspects and, at most, changes only the last element of one list, whether that list
lives in the dict already or is created on the spot. `snap` is $O(1)$: two attribute operations, entirely
independent of `length`. `get` is $O(\log s)$ for $s = \text{len}(h)$: one `bisect_right` over a list keyed
by its own stored ids. Across every index, `_history` holds at most one entry per call to `set` — a call
either appends one new entry or overwrites the existing last one, never both — so total memory is $O(1)$
per index actually written plus $O(1)$ per `set` call, meeting the stated bound with room to spare.

Two other shapes are worth comparing, to see why organising history *per index* earns those costs. Copying
the whole array into a fresh list on every `snap` makes `get` trivial — index straight into the chosen copy,
$O(1)$ — but `snap` becomes $O(\text{length})$, since every element is copied whether or not it changed,
and total memory is $O(\text{length} \times \text{snaps taken})$: one full copy per snapshot, almost all of
it identical to the copy before it. Recording, per snapshot, only the *dict of indices that changed* since
the previous one — a diff, not a full copy — fixes both of those, but `get` must now walk backward from
`snap_id` through however many snapshots precede it, checking each one's diff dict for the queried index,
until it finds one or runs out — $O(k)$, where $k$ is the number of snapshots since that index last
changed, a quantity the index itself has no control over. Keying history by index instead of by snapshot,
as this solution does, gives every index its own short, independently searchable list, so a `get` never has
to look through changes that happened to *other* indices at all.

| Approach | `set` | `snap` | `get` | Memory |
| --- | --- | --- | --- | --- |
| Full copy every `snap` | $O(1)$ | $O(\text{length})$ | $O(1)$ | $O(\text{length} \times \text{snaps})$ |
| One diff dict per snapshot, `get` walks back | $O(1)$ | $O(1)$ | $O(k)$, $k$ = snapshots since that index last changed | $O(\text{length} + \text{sets})$ |
| Per-index history, binary search (this solution) | $O(1)$ | $O(1)$ | $O(\log s)$, $s$ = changes to that index | $O(\text{length} + \text{sets})$ |

### Part 2

Once `release(r)` has taken effect (`r` the largest value ever passed to it), every snapshot id `<= r` is
gone for good — no future `get`, and no future `diff`, may name one. For a single index's history
$[(id_1, v_1), \dots, (id_m, v_m)]$, sorted by id, `bisect_right` would answer every one of the now-released
ids with the *same* entry: the last one with $id_i \le r$, since ids only increase along the list. Call
that the *floor* entry. Every entry strictly before the floor has an id $\le r$ too (ids are increasing),
so it could only ever have answered a query that is now released — it is provably dead. The floor entry
itself, and everything after it, remain exactly as reachable as before `release` was called: the floor
answers every live id up to whichever id comes next in the list (in particular, the smallest possible live
id, $r + 1$, whenever nothing was written at exactly $r + 1$), and every later entry answers exactly the
same ids it always did. So the invariant is: **keep the last entry with id $\le$ released, plus every entry
after it; drop everything before it.**

```python
class SnapshotArray(SnapshotArray):
    """Adds release(): the caller promises never to read a snapshot <= released again."""

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self._released = -1                                   # -1: nothing released yet

    def _compact(self, index: int) -> None:
        """Drops every entry of index's history that release() has made permanently unreachable."""
        h = self._history.get(index)
        if not h or self._released < 0:
            return
        pos = bisect_right(h, self._released, key=lambda e: e[0]) - 1
        if pos > 0:          # NOTE: pos itself is the floor entry to KEEP -- only what precedes it is dead
            del h[:pos]

    def _check_live(self, snap_id: int) -> None:
        self._check_taken(snap_id)
        if snap_id <= self._released:
            raise ValueError(f"snapshot {snap_id} was released (released up to {self._released})")

    def get(self, index: int, snap_id: int) -> int:
        self._check_index(index)
        self._check_live(snap_id)
        return super().get(index, snap_id)

    def stored_entries(self) -> int:
        return sum(len(h) for h in self._history.values())
```

Two ways to apply `_compact` trade `release`'s own cost against how soon `stored_entries()` reaches the
invariant's minimum. **Eager** compaction inspects every index the instant `release` is called:
$\Theta(\text{length})$ every single call, since nothing records in advance which indices might have
anything to trim, but `stored_entries()` is at the invariant's minimum the moment the call returns.
**Lazy** compaction instead does a small, fixed amount of *extra* work per call — compacting whichever
index a `set` call happens to touch anyway, plus a few more indices from a slowly advancing round-robin
pointer on every `release` call — and leaves the rest stale for a while. This is only cheap in total
because compaction only ever *removes* entries, never adds them, and a given entry can be removed at most
once in its entire lifetime (it does not come back). So, summed over the object's whole lifetime, the total
work `_compact` ever does — across every `release` call's fixed-size sweep step and every `set` call's
on-touch step — is at most the total number of entries ever created, i.e. at most the total number of
`set` calls: an amortised $O(1)$ *per call*, on top of each call's own $O(1)$ base cost, by the same
aggregate argument that makes popping from a queue built out of two stacks, or discarding stale entries
from a lazily-cleaned heap, amortised $O(1)$ even though any single pop can cost more. The price is that
`stored_entries()` does not necessarily reach the minimum the instant one `release` call returns — an index
that is neither set again nor yet reached by the sweep keeps its dead entries a while longer, bounded, but
not immediately.

```python
class SnapshotArrayEagerRelease(SnapshotArray):
    """release() compacts every index immediately."""

    def release(self, snap_id: int) -> None:
        self._check_taken(snap_id)
        if snap_id > self._released:
            self._released = snap_id
        for index in range(self._length):
            self._compact(index)


class SnapshotArrayLazyRelease(SnapshotArray):
    """release() only raises the boundary and nudges a small, fixed-size batch of a round-robin sweep
    forward; the rest of an index's compaction happens the next time that index is set."""

    _SWEEP_BATCH = 4          # arbitrary and small -- illustrates the mechanism, not a tuned constant

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self._sweep_pos = 0

    def set(self, index: int, val: int) -> None:
        self._check_index(index)
        self._compact(index)     # NOTE: on-touch -- the cheapest moment to drop this index's dead entries
        super().set(index, val)

    def release(self, snap_id: int) -> None:
        self._check_taken(snap_id)
        if snap_id > self._released:
            self._released = snap_id
        batch = min(self._SWEEP_BATCH, self._length)   # NOTE: min() -- also keeps this at 0 when length == 0
        for _ in range(batch):
            self._compact(self._sweep_pos)
            self._sweep_pos = (self._sweep_pos + 1) % self._length


SnapshotArray = SnapshotArrayLazyRelease   # the fuller answer: cheap release(), memory still bounded
```

| Approach | `release`'s own cost | extra cost per `set` | `stored_entries()` right after `release` returns |
| --- | --- | --- | --- |
| Eager | $\Theta(\text{length})$ | none | exactly the invariant's minimum |
| Lazy | $O(1)$ (fixed batch) | $O(1)$ amortised | may still hold released-only entries, bounded, not immediate |

### Part 3

Answering `diff(a, b)` without ever scanning `length` needs to know cheaply *which* indices could possibly
have changed, so the class also keeps a journal. `_pending_touched` collects the indices `set` since the
last `snap` — the ones that matter to whichever snapshot is sealed next — and `snap` files it away as
`_journal[snap_id]` before deferring to the base class for the id itself; after any `snap` call,
`len(_journal) == _next_id`, so `_journal[s]` is defined for exactly the snapshot ids that have actually
been taken. An index changed more than once inside one interval still contributes only one entry to that
interval's set (sets do not hold duplicates), and an index untouched in an interval contributes nothing to
it at all.

`diff(a, b)` unions `_journal[a + 1]` through `_journal[b]` — every index touched anywhere strictly after
`a` and up to `b` — then checks each candidate's actual value at `a` and at `b` with `get`, keeping only the
ones that really differ: being touched is necessary but not sufficient, since an index can be written back
to its old value, or written more than once and end up unchanged overall. Building the union costs $O(D)$,
where $D$ is the total number of journal entries visited — at most the number of `set` calls recorded in
snapshots $a + 1$ through $b$, however large `length` is; checking the (at most $D$) candidates costs
$O(D \log S)$ more, for $S$ the largest per-index history length among them; building the result costs
$O(\text{output size})$. None of it depends on `length`.

```python
class SnapshotArray(SnapshotArray):
    """Adds diff(): every index whose value changed between two live snapshots."""

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self._journal: list[set[int]] = []       # journal[s]: indices set during the interval sealed as s
        self._pending_touched: set[int] = set()

    def set(self, index: int, val: int) -> None:
        super().set(index, val)             # validates index, applies the inherited on-touch compaction
        self._pending_touched.add(index)     # NOTE: only after a successful set -- a rejected one changed nothing

    def snap(self) -> int:
        self._journal.append(self._pending_touched)
        self._pending_touched = set()
        return super().snap()

    def diff(self, a: int, b: int) -> dict[int, tuple[int, int]]:
        self._check_live(a)
        self._check_live(b)
        if not a < b:
            raise ValueError(f"diff requires a < b, got a={a}, b={b}")
        touched: set[int] = set()
        for s in range(a + 1, b + 1):
            touched |= self._journal[s]
        result: dict[int, tuple[int, int]] = {}
        for index in touched:
            va, vb = self.get(index, a), self.get(index, b)
            if va != vb:
                result[index] = (va, vb)
        return result
```

`snap` and `set` each do exactly one more $O(1)$ operation than their Part 2 versions — sealing or growing a
set — so both remain $O(1)$ (amortised, for `set`, exactly as before).

### Follow-ups

- **Persistence.** An append-only log of every `set` and `snap` call, flushed before the call returns, lets
  the array be rebuilt after a crash by replaying it from the start; once the log is long, replaying all of
  it becomes slow, so a periodic checkpoint — a full array copy tagged with the id of the last snapshot it
  reflects — lets recovery start there and replay only the log entries written afterwards.
- **Concurrent readers of old snapshots.** Once a snapshot is taken, the entries that answer it never
  change again: `set` only ever appends a new entry or overwrites the *current pending* one, never an entry
  belonging to an already-taken snapshot. So any number of threads may call `get` on already-taken
  snapshots with no locking at all. `set` and `snap` racing on the same object still need a lock, though,
  and it cannot be scoped to one index at a time: `snap` advances the single shared `_next_id` that every
  `set` call reads to decide whether to append or overwrite, so the two operations cannot safely run at
  once on any index while the other is in flight.
- **Snapshots of a 2-D array.** Flattening `(row, col)` to `row * n_cols + col` and using this exact same
  per-index history and journal reduces the two-dimensional case to the one already solved — `length`
  becomes `n_rows * n_cols`, and `diff` still returns flat indices, or `(row, col)` pairs if unflattened at
  the boundary.
- **Copy-on-write pages.** Splitting the array into fixed-size pages, addressed through a small persistent
  tree of page pointers rather than a flat array of them (so updating one pointer copies only the
  $O(\log P)$ nodes on its path, never the whole pointer table), and copying a whole page — never the whole
  array — the first time any of its elements is written after a snapshot, makes `snap` $O(1)$ (keep a
  reference to the current tree root; nothing to copy yet) at the cost of one page-sized copy, plus that
  $O(\log P)$ path, on a page's *first* write per interval, rather than one small entry per write. This is
  coarser than per-index history — one written element pays for its whole page — but it is the trade-off
  real copy-on-write filesystems and persistent data structures make, in exchange for far fewer things to
  track per snapshot.

<details>
<summary>Checks (runnable)</summary>

```python
import random

# --- Part 1: the worked example, traced step by step ---
arr = SnapshotArray(4)
arr.set(0, 5)
arr.set(0, 6)
assert arr.snap() == 0
arr.set(1, 9)
arr.set(0, 1)
assert arr.snap() == 1
arr.set(0, 4)
arr.set(2, 2)
assert arr.snap() == 2
assert arr.snap() == 3
assert arr.snap() == 4
arr.set(0, 7)
assert arr.snap() == 5

assert arr.get(0, 0) == 6
assert arr.get(0, 1) == 1
assert arr.get(0, 2) == 4
assert arr.get(0, 3) == 4
assert arr.get(0, 4) == 4
assert arr.get(0, 5) == 7
assert arr.get(1, 0) == 0
assert arr.get(1, 1) == 9
assert arr.get(2, 1) == 0
assert arr.get(2, 2) == 2
assert arr.get(3, 5) == 0

# --- Part 3's worked example, same object, before anything is released ---
assert arr.diff(3, 5) == {0: (4, 7)}
assert arr.diff(2, 4) == {}

# --- Part 2's worked example, same object ---
arr.release(2)
for snap_id in (1, 2):
    try:
        arr.get(0, snap_id)
        assert False, "expected ValueError"
    except ValueError:
        pass
assert arr.get(0, 4) == 4
arr.release(0)                 # no-op: 0 is already below the released boundary of 2
assert arr._released == 2

# --- IndexError / ValueError boundary cases ---
try:
    arr.set(4, 1)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.get(0, 100)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.get(0, -1)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.release(100)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.diff(3, 100)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.diff(5, 3)
    assert False, "expected ValueError"
except ValueError:
    pass
try:
    arr.diff(3, 3)
    assert False, "expected ValueError"
except ValueError:
    pass

# --- length == 0: every index is out of range, and release()'s sweep must not divide by zero ---
empty = SnapshotArray(0)
assert empty.snap() == 0
assert empty.snap() == 1
try:
    empty.set(0, 1)
    assert False, "expected IndexError"
except IndexError:
    pass
assert empty.diff(0, 1) == {}
empty.release(0)          # must not raise ZeroDivisionError -- see the NOTE on the sweep's min()
assert empty.stored_entries() == 0


# --- an independent brute force: a full array copy at every snapshot, nothing else ---
class _BruteForceSnapshotArray:
    def __init__(self, length: int) -> None:
        self._length = length
        self._current = [0] * length
        self._snapshots: list[list[int]] = []
        self._released = -1

    def set(self, index: int, val: int) -> None:
        if not (0 <= index < self._length):
            raise IndexError
        self._current[index] = val

    def snap(self) -> int:
        self._snapshots.append(list(self._current))
        return len(self._snapshots) - 1

    def get(self, index: int, snap_id: int) -> int:
        if not (0 <= index < self._length):
            raise IndexError
        if not (0 <= snap_id < len(self._snapshots)):
            raise IndexError
        if snap_id <= self._released:
            raise ValueError
        return self._snapshots[snap_id][index]

    def release(self, snap_id: int) -> None:
        if not (0 <= snap_id < len(self._snapshots)):
            raise IndexError
        self._released = max(self._released, snap_id)

    def diff(self, a: int, b: int) -> dict[int, tuple[int, int]]:
        for s in (a, b):
            if not (0 <= s < len(self._snapshots)):
                raise IndexError
            if s <= self._released:
                raise ValueError
        if not a < b:
            raise ValueError
        sa, sb = self._snapshots[a], self._snapshots[b]
        return {i: (sa[i], sb[i]) for i in range(self._length) if sa[i] != sb[i]}


def _apply(obj, op):
    try:
        if op[0] == "set":
            return "ok", obj.set(op[1], op[2])
        if op[0] == "snap":
            return "ok", obj.snap()
        if op[0] == "get":
            return "ok", obj.get(op[1], op[2])
        if op[0] == "release":
            return "ok", obj.release(op[1])
        if op[0] == "diff":
            return "ok", obj.diff(op[1], op[2])
    except (IndexError, ValueError) as e:
        return "error", type(e)
    raise AssertionError(f"unknown op {op[0]}")


def _random_ops(rng, length, n_ops):
    ops = []
    taken = 0
    for _ in range(n_ops):
        choice = rng.random()
        if choice < 0.30:
            ops.append(("set", rng.randrange(-1, length + 1), rng.randint(-9, 9)))
        elif choice < 0.50:
            ops.append(("snap",))
            taken += 1
        elif choice < 0.72:
            hi = max(taken, 1)
            ops.append(("get", rng.randrange(-1, length + 1), rng.randrange(-2, hi + 1)))
        elif choice < 0.85:
            hi = max(taken, 1)
            ops.append(("release", rng.randrange(-2, hi + 1)))
        else:
            hi = max(taken, 1)
            ops.append(("diff", rng.randrange(-2, hi + 1), rng.randrange(-2, hi + 1)))
    return ops


# --- randomised cross-check against the brute force, over many small arrays ---
counts = {"bad_index": 0, "bad_snap_id": 0, "released": 0, "bad_order": 0, "ok_diff": 0, "ok_nonempty_diff": 0}
for seed in range(400):
    rng = random.Random(seed)
    length = rng.randint(1, 6)
    sol = SnapshotArray(length)
    brute = _BruteForceSnapshotArray(length)
    for op in _random_ops(rng, length, 60):
        taken_before, released_before = len(brute._snapshots), brute._released
        got, want = _apply(sol, op), _apply(brute, op)
        assert got == want, (seed, op, got, want)
        if want[0] == "error":
            kind = op[0]
            if kind in ("set", "get") and not (0 <= op[1] < length):
                counts["bad_index"] += 1
            elif kind == "get" and not (0 <= op[2] < taken_before):
                counts["bad_snap_id"] += 1
            elif kind == "release" and not (0 <= op[1] < taken_before):
                counts["bad_snap_id"] += 1
            elif kind == "diff" and (not (0 <= op[1] < taken_before) or not (0 <= op[2] < taken_before)):
                counts["bad_snap_id"] += 1
            elif kind == "get" and op[2] <= released_before:
                counts["released"] += 1
            elif kind == "diff" and (op[1] <= released_before or op[2] <= released_before):
                counts["released"] += 1
            elif kind == "diff":
                counts["bad_order"] += 1
        elif op[0] == "diff":
            counts["ok_diff"] += 1
            if want[1]:
                counts["ok_nonempty_diff"] += 1
assert min(counts.values()) > 5, counts   # every interesting case actually fired, repeatedly, not just once

# --- Part 2: stored_entries() matches an independently computed bound, and release() shrinks it ---
# a workload with many overwritten snapshots: release() should collapse most of it away
probe = SnapshotArrayEagerRelease(1)
for s in range(30):
    for i in range(5):
        probe.set(0, s * 10 + i)     # 5 overwrites per interval -- only the last (s * 10 + 4) ever matters
    probe.snap()
assert probe.stored_entries() == 30   # 150 set() calls collapse to 30 entries, one per snapshot interval
assert probe.get(0, 0) == 4
probe.release(25)
assert probe.stored_entries() == 5    # release(25) then collapses ids 0..24 into their single floor entry


def _reference_entry_count(touched_by_index, released):
    """The number of entries the stated invariant allows to survive, computed only from touched_by_index
    (the snapshot ids at which each index was actually set, tracked directly from the operations applied --
    never read from SnapshotArray's own _history)."""
    total = 0
    for ids in touched_by_index:
        if not ids:
            continue
        if released < 0:
            total += len(ids)
            continue
        total += (1 if any(i <= released for i in ids) else 0) + sum(1 for i in ids if i > released)
    return total


rng = random.Random(2024)
length = 40
sol_eager = SnapshotArrayEagerRelease(length)
sol_lazy = SnapshotArrayLazyRelease(length)
touched_by_index = [[] for _ in range(length)]
pending, taken, release_calls = set(), 0, 0
for _ in range(800):
    action = rng.random()
    if action < 0.55:
        index, val = rng.randrange(length), rng.randint(-99, 99)
        sol_eager.set(index, val)
        sol_lazy.set(index, val)
        pending.add(index)
    elif action < 0.75:
        for index in pending:
            touched_by_index[index].append(taken)
        pending = set()
        assert sol_eager.snap() == taken
        assert sol_lazy.snap() == taken
        taken += 1
    elif taken:
        snap_id = rng.randrange(taken)
        sol_eager.release(snap_id)
        sol_lazy.release(snap_id)
        release_calls += 1
# seal any dangling, not-yet-snapped writes with one final snap, so touched_by_index fully accounts
# for every stored entry -- otherwise a write with no snap after it would be invisible to this bookkeeping
# but still present, correctly, in _history
for index in pending:
    touched_by_index[index].append(taken)
assert sol_eager.snap() == taken
assert sol_lazy.snap() == taken
taken += 1

released_level = sol_eager._released
bound = _reference_entry_count(touched_by_index, released_level)
uncompacted_total = sum(len(ids) for ids in touched_by_index)
assert release_calls > 20 and released_level >= 0          # the scenario actually exercises release()
assert sol_eager.stored_entries() == bound, (sol_eager.stored_entries(), bound)
assert sol_lazy.stored_entries() >= bound
assert sol_lazy.stored_entries() <= uncompacted_total
assert sol_eager.stored_entries() < uncompacted_total        # eager: release() really did shrink things

# --- Part 2: eager's release() touches every index; lazy's touches only a small, fixed batch ---
class _CountingEager(SnapshotArrayEagerRelease):
    def __init__(self, length):
        super().__init__(length)
        self.compact_calls = 0

    def _compact(self, index):
        self.compact_calls += 1
        super()._compact(index)


class _CountingLazy(SnapshotArrayLazyRelease):
    def __init__(self, length):
        super().__init__(length)
        self.compact_calls = 0

    def _compact(self, index):
        self.compact_calls += 1
        super()._compact(index)


for probe_length in (10, 5_000):
    eager_probe = _CountingEager(probe_length)
    eager_probe.snap()
    eager_probe.compact_calls = 0
    eager_probe.release(0)
    assert eager_probe.compact_calls == probe_length

    lazy_probe = _CountingLazy(probe_length)
    lazy_probe.snap()
    lazy_probe.compact_calls = 0
    lazy_probe.release(0)
    assert lazy_probe.compact_calls == min(SnapshotArrayLazyRelease._SWEEP_BATCH, probe_length)

# --- Part 3: diff() cost does not grow with length ---
def _diff_journal_scan_size(a_arr, a, b) -> int:
    """Independently recomputes, from a_arr._journal alone, how many (interval, index) entries diff(a, b)
    must visit -- the same quantity its own union loop touches, measured from the outside."""
    return sum(len(a_arr._journal[s]) for s in range(a + 1, b + 1))


for probe_length in (5, 50_000):
    diff_probe = SnapshotArray(probe_length)
    diff_probe.set(1 % probe_length, -1)
    diff_probe.snap()                    # snapshot 0 -- before the diffed range, must not be scanned
    diff_probe.set(2 % probe_length, -1)
    diff_probe.snap()                    # snapshot 1
    a = diff_probe.snap()                # snapshot 2, a = 2 -- nothing set since snapshot 1
    diff_probe.set(0, 1)
    diff_probe.set(3 % probe_length, 9)
    b = diff_probe.snap()                # snapshot 3, b = 3
    diff_probe.snap()                    # snapshot 4 -- after the diffed range, must not be scanned either
    assert _diff_journal_scan_size(diff_probe, a, b) == 2
    assert diff_probe.diff(a, b) == {0: (0, 1), 3 % probe_length: (0, 9)}

# --- Part 3: diff() against the brute force's full-array comparison, over many random workloads ---
diff_checks = 0
for seed in range(200):
    rng = random.Random(10_000 + seed)
    length = rng.randint(2, 8)
    sol = SnapshotArray(length)
    brute = _BruteForceSnapshotArray(length)
    for _ in range(rng.randint(10, 40)):
        if rng.random() < 0.7:
            index, val = rng.randrange(length), rng.randint(-9, 9)
            sol.set(index, val)
            brute.set(index, val)
        else:
            assert sol.snap() == brute.snap()
    taken = len(brute._snapshots)
    if taken >= 2:
        for _ in range(5):
            a, b = sorted(rng.sample(range(taken), 2))
            assert sol.diff(a, b) == brute.diff(a, b), (seed, a, b)
            diff_checks += 1
assert diff_checks > 500

print("all checks passed")
```

</details>

</details>
