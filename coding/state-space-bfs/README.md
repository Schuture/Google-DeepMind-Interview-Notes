# Shortest Paths Through Locked Doors

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · BFS over a state space | ★★★★★ | Hard | SWE · RE · MLE · Intern | bfs, state-space-search, bitmask, path-counting, dijkstra, grid | 3 parts / 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

A robot moves on a rectangular grid given as `grid: list[str]`, a list of equal-length strings. Each
character is one of six kinds of cell: `#` a wall, `.` floor, `S` the robot's start (exactly one cell in
the grid), `T` the target (exactly one cell), a lowercase letter `a`–`f` a key, or the matching uppercase
letter `A`–`F` a door. `S` and `T` are themselves floor for movement purposes. A grid uses at most six
distinct key letters, so the set of keys held at any moment fits in a 6-bit mask.

A *move* takes the robot from its current cell to one of the (at most four) edge-adjacent cells — one row
or one column away, never diagonal — that lies inside the grid and is not a wall. Moving onto a door is a
move like any other, except that it is legal only if the robot already holds the key of the same letter,
lower-cased; moving onto a key cell is always legal and picks that key up automatically and permanently
the instant the robot arrives — a key already held is picked up again with no effect, and no cell ever
consumes a key. The robot may revisit any cell, including one it already holds the key for or has already
crossed, any number of times.

A *path* is a sequence of cells $c_0, c_1, \ldots, c_n$ with $c_0 = S$, each $c_i$ to $c_{i+1}$ a legal
move given the keys held after $c_0, \ldots, c_i$ have been visited, and $c_n = T$ the first time any
$c_i$ equals $T$ — reaching $T$ ends the walk, so no path continues past it. Two paths are the same
exactly when they are the same sequence of cells; two paths that reach $T$ in the same number of moves
but differ in even one cell are different paths.

### Part 1 — Fewest moves

```py
def shortest_path(grid: list[str]) -> int: ...   # -1 if T is unreachable
```

Return the number of moves on a shortest path from `S` to `T`, or `-1` if no path reaches `T` at all.

```text
grid = [
    "S..AT",
    "##.##",
    "..a..",
]
```

Row 0 reads `S . . A T` and is blocked at once: the door at `(0, 3)` needs key `a`, which sits at
`(2, 2)`, two rows down and reachable only through `(1, 2)` — the single gap in row 1's wall. The robot
must detour: right, right, down, down (key `a` picked up at `(2, 2)`), then undo the detour — up, up —
before the door will let it through, then right, right, into `T`:

```text
(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)* -> (1,2) -> (0,2) -> (0,3) -> (0,4)
                                     ^ key a picked up here
```

Eight moves; `(0, 2)` and `(1, 2)` are each visited twice, and this is the only shortest path:
`shortest_path(grid) == 8`.

A second grid has no shortest path at all:

```text
grid = [
    "S.AT",
    ".#.#",
    ".#a#",
]
```

Key `a` sits at `(2, 2)`, whose only neighbour is `(1, 2)`, whose only other neighbour is the door
`(0, 2)` itself — the one cell that needs the key to be entered in the first place. Nothing ever reaches
`a`, the door never opens, and `T` is unreachable: `shortest_path(grid) == -1`.

### Part 2 — How many shortest paths

```py
def count_shortest_paths(grid: list[str], mod: int = 1_000_000_007) -> int: ...   # 0 if unreachable
```

Return the number of distinct shortest paths from `S` to `T`, modulo `mod` (two paths differ, per the
definition of *path* above, whenever they differ in any position — a different route to the same cell,
or the same route in a different order, is a different path even when both take the same number of
moves); `0` if `T` is unreachable.

Re-open one wall of Part 1's grid, at `(1, 0)`:

```text
grid = [
    "S..AT",
    ".#.##",
    "..a..",
]
```

The shortest length is still 8 — Part 1's path above still works — but there is now a second way to
reach the key, down the newly opened left side instead of detouring from the top, that also takes exactly
eight moves in total:

```text
(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)* -> (1,2) -> (0,2) -> (0,3) -> (0,4)   # via the top, as in Part 1
(0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2)* -> (1,2) -> (0,2) -> (0,3) -> (0,4)   # via the newly opened left side
```

Both reach the key on move 4 and both take the same four moves back out, through `(1, 2)`, `(0, 2)` and
the now-unlocked door; they differ in cells 1–3, so they count as two distinct paths:
`count_shortest_paths(grid) == 2`.

### Part 3 — Mud

Cells marked `~` are mud: moving onto one costs 2 rather than 1 (moving off it, onto an ordinary cell,
costs only that cell's own price, never more). Implement both functions again for this cost model, where
the quantity to minimise is total cost rather than move count:

```py
def shortest_path_weighted(grid: list[str]) -> int: ...
def count_shortest_paths_weighted(grid: list[str], mod: int = 1_000_000_007) -> int: ...
```

`shortest_path_weighted` returns the minimum total cost from `S` to `T` (`-1` if unreachable);
`count_shortest_paths_weighted` returns how many distinct minimum-cost paths achieve it, modulo `mod`.
Keys, doors and walls behave exactly as before — only the price of entering a cell changes. On a grid
with no `~` at all, both functions must agree with Parts 1 and 2 exactly, since every move then costs 1
and total cost equals move count.

```text
grid = [
    "S~~T",
    "....",
]
```

Going straight through the mud costs $2 + 2 + 1 = 5$ over three moves; going the long way around, through
four ordinary floor cells, costs $1+1+1+1+1 = 5$ over five moves — the same total cost by a different
number of moves, so neither is more optimal on cost than the other:

```text
(0,0) -> (0,1)~ -> (0,2)~ -> (0,3)                    # cost 2 + 2 + 1 = 5, three moves
(0,0) -> (1,0) -> (1,1) -> (1,2) -> (1,3) -> (0,3)     # cost 1x5 = 5, five moves
```

`shortest_path_weighted(grid) == 5` and `count_shortest_paths_weighted(grid) == 2`.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Three points are worth confirming before coding: whether the robot may revisit a cell, including one it
already holds the key for (yes — nothing in the statement forbids it, and Part 1's own example needs it);
whether a key is consumed on use (no — a door only checks that its bit is set, never clears it); and
whether `S` and `T` themselves count as ordinary floor for movement (yes — the robot walks across either
one freely, and `T` merely ends the walk the moment it is reached).

### Part 1

The obstacle is that legality at a cell depends on more than the cell itself: whether the robot may step
onto a door depends on which keys it currently holds, which depends on the route it took to get there.
Plain BFS over $(row, col)$ has no room for that — its visited set marks a cell done forever the first
time it is reached, so if the shortest way to reach some chokepoint cell happens to arrive without a key
already, that arrival is what gets recorded, and the search can never try the same cell again later with
a key in hand. Extending the state to a triple $(row, col, mask)$, where bit $i$ of `mask` records whether
the key of the $i$-th letter is held, and running BFS on states instead of cells fixes this: two states
are the same only when both position and key set match, so a cell that mattered once without a key and
again later with one is visited twice, correctly, as two different states.

On the first grid of Part 1, a plain cell-BFS reaches `(0, 2)` at distance 2, via `(0, 1)`, without the
key — and marks it done. Every legal way through the door at `(0, 3)` needs to reach `(0, 2)` again with
the key already held, which the plain BFS has already ruled out by marking `(0, 2)` visited; it reports
`T` unreachable, `-1`, when the true answer is 8 moves (checked below). The state-space BFS avoids this
because `(0, 2, 0)` (no key) and `(0, 2, 1)` (key `a` held) are different states, visited on different
iterations.

With $R$ rows, $C$ columns and $k \le 6$ distinct key letters in the grid, there are at most
$R \cdot C \cdot 2^k$ states, each with at most four outgoing transitions checked in $O(1)$, so BFS over
the state graph runs in $O(R \cdot C \cdot 2^k)$ time and space — the visited set (`dist` below) is keyed
by state, not by cell. `T` is terminal: once a state's cell is `T`, the walk that reached it has already
ended, so that state is never expanded, matching the definition of *path* in the statement (nothing
continues past the first arrival at `T`).

```python
from collections import deque

def _parse(grid: list[str]) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    R, C = len(grid), len(grid[0])
    start = target = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'T':
                target = (r, c)
    return R, C, start, target


def _transitions(grid, R, C, r, c, mask):
    """Yields (nr, nc, nmask) for every legal one-cell move out of state (r, c, mask): edge-adjacent,
    inside the grid, not a wall, and not a locked door (an uppercase letter whose key bit is not yet set
    in mask). Stepping onto a key cell sets its bit in nmask; every other cell leaves mask unchanged."""
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if not (0 <= nr < R and 0 <= nc < C):
            continue
        ch = grid[nr][nc]
        if ch == '#':
            continue
        if ch.isupper() and ch not in ('S', 'T'):
            if not (mask >> (ord(ch) - ord('A'))) & 1:
                continue        # NOTE: door's key bit not held yet -- this move is not legal
            nmask = mask
        elif ch.islower():
            nmask = mask | (1 << (ord(ch) - ord('a')))
        else:
            nmask = mask
        yield nr, nc, nmask


def _bfs(grid):
    """BFS over states (row, col, mask of keys held). Returns (target_cell, dist, ways): dist maps every
    reached state to its distance from S, and ways maps it to the number of distinct minimal-length
    state-sequences reaching it (Part 2). T is terminal -- once a state's cell is T it is never expanded."""
    R, C, start, target = _parse(grid)
    s0 = (start[0], start[1], 0)
    dist = {s0: 0}
    ways = {s0: 1}
    q = deque([s0])
    while q:
        r, c, mask = q.popleft()
        if (r, c) == target:
            continue            # NOTE: T is terminal -- reaching it ends the walk, so it is never expanded
        d = dist[(r, c, mask)]
        for nr, nc, nmask in _transitions(grid, R, C, r, c, mask):
            ns = (nr, nc, nmask)
            if ns not in dist:
                dist[ns] = d + 1
                ways[ns] = ways[(r, c, mask)]
                q.append(ns)
            elif dist[ns] == d + 1:
                ways[ns] += ways[(r, c, mask)]     # NOTE: another shortest way into ns, from a different state
    return target, dist, ways


def shortest_path(grid: list[str]) -> int:
    target, dist, _ways = _bfs(grid)
    lengths = [d for (r, c, _m), d in dist.items() if (r, c) == target]
    return min(lengths) if lengths else -1
```

`shortest_path` reads off the minimum over every state whose cell is the target — the robot can reach `T`
holding different key sets along different routes, and the shortest overall is the smallest of those,
regardless of which keys happen to still be held on arrival.

### Part 2

`shortest_path` already visits every state exactly once, in order of non-decreasing distance (the
standard BFS layer property, since every transition costs 1); `count_shortest_paths` reuses that same
traversal and, alongside `dist`, tracks `ways`, the number of distinct shortest *state*-sequences reaching
each state. A state is first discovered by exactly one predecessor, which sets `ways` to that
predecessor's own count; a second predecessor reaching the same state at the same distance —
`dist[ns] == d + 1` on a state already in `dist` — is a genuinely different tied route, so its count is
added in rather than replacing anything. Because BFS discovers every state at distance $d$ before any
state at distance $d + 1$, every predecessor that could contribute to `ways[ns]` has already been
processed by the time `ns` itself is dequeued, so `ways[ns]` is complete before it is ever read.

This counts state-sequences, but the problem defines a path as a sequence of *cells*, so the two need to
be shown to agree. Given any valid cell-sequence $c_0, \ldots, c_n$, the mask held after $c_i$ is
completely determined by the prefix $c_0, \ldots, c_i$ alone — it is the OR of the key bits of every key
cell among them, since pickup is automatic, permanent and depends on nothing else about the path. So the
cell-sequence determines exactly one state-sequence $(c_0, m_0), \ldots, (c_n, m_n)$, and that
state-sequence is itself a legal walk in the state graph, by construction of `_transitions`. Conversely,
dropping the mask from any legal state-sequence gives back a legal cell-sequence in the original problem.
Two different cell-sequences already differ in their cell component, hence in their state-sequence too;
so this correspondence is a length-preserving bijection between the two, and in particular it matches
shortest cell-sequences to shortest state-sequences one for one — counting one counts the other.

`T` is terminal in the state graph, so a state-sequence that reaches a target state cannot continue past
it, matching "ends at the first arrival at `T`" automatically; but the target can be reached with several
different masks, at possibly different distances (some routes hold more keys than others on arrival), so
the answer sums `ways` only over target states at the *overall* minimum distance, discarding any target
state reached later than that.

```python
def count_shortest_paths(grid: list[str], mod: int = 1_000_000_007) -> int:
    target, dist, ways = _bfs(grid)
    at_target = [(d, ways[s]) for s, d in dist.items() if s[0] == target[0] and s[1] == target[1]]
    if not at_target:
        return 0
    best = min(d for d, _ in at_target)
    return sum(w for d, w in at_target if d == best) % mod    # NOTE: only the states at the minimum distance
```

Same $O(R \cdot C \cdot 2^k)$ time and space as Part 1: `ways` adds $O(1)$ work per transition on top of
`_bfs`, no new asymptotic cost.

### Part 3

Moves no longer cost a uniform 1, so plain BFS layers no longer correspond to shortest cost — a
three-move path through two mud cells and a five-move path around it can tie, as the worked example
shows, so which one is "shorter" depends on cost, not on move count. Dijkstra's algorithm generalises
Part 1's BFS to this: pop states from a min-heap keyed by tentative cost, and once a state is popped it is
*settled* — its final, minimum cost is now known, because every edge weighs at least 1 and the heap
always pops the globally smallest tentative cost next, so nothing settled later could ever produce a
smaller cost for it.

Counting shortest paths under Dijkstra needs one more fact: since every edge costs at least 1, every
predecessor of a state $v$ has strictly smaller final cost than $v$ itself, so — by the same
popping-order argument — every predecessor of $v$ is already settled by the time $v$ is popped. That
makes it safe to accumulate `ways[v]` from any predecessor that relaxes $v$ to a tied cost at any point
before $v$ is popped, and to trust `ways[v]` as final only from the moment $v$ is settled, exactly
mirroring Part 2's BFS-layer argument with "settled" in place of "already at distance $d$". Three
mistakes follow directly from this and are marked `# NOTE:` at the line they would be made: crediting a
`ways` update before the state it updates is settled; adding a count across an edge that does not
actually tie the current best cost (a *slack*, not *tight*, edge); and forgetting that `T`, once popped,
is terminal and must not be expanded, exactly as in Parts 1–2.

```python
import heapq

def _step_cost(ch: str) -> int:
    return 2 if ch == '~' else 1


def _dijkstra(grid):
    R, C, start, target = _parse(grid)
    s0 = (start[0], start[1], 0)
    dist = {s0: 0}
    ways = {s0: 1}
    settled = set()
    heap = [(0, s0)]
    while heap:
        d, state = heapq.heappop(heap)
        if state in settled:
            continue            # NOTE: a stale heap entry -- this state's cost was already finalised
        settled.add(state)      # NOTE: ways[state] is only trustworthy from here, once popped at its least cost
        r, c, mask = state
        if (r, c) == target:
            continue            # NOTE: T is terminal -- never expanded, even once popped
        for nr, nc, nmask in _transitions(grid, R, C, r, c, mask):
            step = _step_cost(grid[nr][nc])
            ns = (nr, nc, nmask)
            nd = d + step
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                ways[ns] = ways[state]
                heapq.heappush(heap, (nd, ns))
            elif nd == dist[ns]:
                ways[ns] += ways[state]    # NOTE: only a tight edge (nd == the current best) may add to ways
    return target, dist, ways


def shortest_path_weighted(grid: list[str]) -> int:
    target, dist, _ways = _dijkstra(grid)
    lengths = [d for (r, c, _m), d in dist.items() if (r, c) == target]
    return min(lengths) if lengths else -1


def count_shortest_paths_weighted(grid: list[str], mod: int = 1_000_000_007) -> int:
    target, dist, ways = _dijkstra(grid)
    at_target = [(d, ways[s]) for s, d in dist.items() if s[0] == target[0] and s[1] == target[1]]
    if not at_target:
        return 0
    best = min(d for d, _ in at_target)
    return sum(w for d, w in at_target if d == best) % mod
```

With a binary heap, each of the $O(R \cdot C \cdot 2^k)$ states is pushed at most once per relaxation
that improves its tentative cost and popped at most once as settled, over at most 4 outgoing edges each,
for $O(R \cdot C \cdot 2^k \log(R \cdot C \cdot 2^k))$ time. Since every cost here is 1 or 2, a *bucket
queue* — an array of $2 \cdot R \cdot C \cdot 2^k$ buckets indexed by total cost, each holding the states
currently tentative at that cost, advanced one index at a time, a generalisation of 0-1 BFS to a
two-valued weight — settles states in the same order without a heap's $\log$ factor, for
$O(R \cdot C \cdot 2^k)$ time; Dijkstra is simpler to get right and is what the checks below run. On a
grid with no `~`, `_step_cost` always returns 1, every edge ties, and this reduces to exactly Parts 1–2's
BFS, visiting states in a different order but computing the same `dist` and `ways`.

### Follow-ups

- **A-star with a Manhattan-distance heuristic.** Adding $h(r, c) = |r - r_T| + |c - c_T|$ to a state's
  tentative cost before ordering the heap is admissible — it never overestimates, since every move
  changes row or column by exactly 1 at a cost of at least 1, so $h$ can only under-count the moves still
  needed — and it still finds the shortest path first, exactly as Dijkstra does; it typically settles far
  fewer states than plain Dijkstra once `T` is roughly in one direction from most of the grid, though the
  heuristic depends only on position, not on the mask, so it gives no head start on the part of the search
  that is really about routing to keys.
- **Bidirectional search.** Meeting a forward search from `S` with a backward one from `T` roughly halves
  the effective radius each side has to cover, but the backward side has to start without knowing which
  keys the robot holds on arrival at `T` — every mask is a plausible ending state — so it must fan out
  from all $2^k$ of them at once, closing the gap only once a forward state and a backward state agree on
  both position and mask; the saving from meeting in the middle is real only once $k$ is small enough that
  this is cheaper than searching the full radius from `S` alone.
- **Grids far larger than $R \cdot C \cdot 2^6$ can afford.** Most of the theoretical $2^k$ masks are
  never actually reachable once keys are gated behind other keys or behind long corridors, so building
  `dist` and `ways` as plain dicts keyed by the states BFS actually discovers, as above, rather than
  pre-allocating an array over every mask, already keeps the state space to the number of *reachable*
  states, not the worst-case bound.
- **The $k$ shortest paths, not just the shortest.** Counting ties at the minimum distance is not the
  same problem as listing the second-shortest, third-shortest and so on, some of which are not ties at
  all; that needs Eppstein's algorithm or a $k$-shortest-paths variant of Dijkstra run directly on the
  state graph above, replacing "keep the first settlement of a state" with "keep its best $k$".
- **The modulus.** `mod = 1_000_000_007` bounds the returned count to a fixed-width integer, the standard
  convention carried over from languages where an unreduced count would overflow; Python integers never
  overflow, but the reduction is kept for the same reason a competitive-programming answer is taken
  modulo a prime near $2^{30}$ — the number of tied paths can grow exponentially in the number of tied
  branches along the way, and reducing early keeps every intermediate sum cheap to add.

<details>
<summary>Checks (runnable)</summary>

```python
import random
from collections import deque


def _replay(grid, cells, weighted=False):
    """Independently replays `cells` against the statement's rules: starts at S, each step an
    edge-adjacent move onto a non-wall cell, a door only once its key bit is already held, T only as the
    very last cell. Returns the path's total cost (move count when weighted is False)."""
    R, C = len(grid), len(grid[0])
    assert grid[cells[0][0]][cells[0][1]] == 'S'
    assert all(grid[r][c] != 'T' for r, c in cells[:-1])   # T ends the walk -- never an interior cell
    assert grid[cells[-1][0]][cells[-1][1]] == 'T'
    mask, cost = 0, 0
    for (r, c), (nr, nc) in zip(cells, cells[1:]):
        assert abs(nr - r) + abs(nc - c) == 1 and 0 <= nr < R and 0 <= nc < C
        ch = grid[nr][nc]
        assert ch != '#'
        if ch.isupper() and ch not in ('S', 'T'):
            assert (mask >> (ord(ch) - ord('A'))) & 1, "door entered without its key"
        elif ch.islower():
            mask |= 1 << (ord(ch) - ord('a'))
        cost += 2 if (weighted and ch == '~') else 1
    return cost


# --- Part 1: the worked example, its trace, and the -1 example ---
grid_a = ["S..AT", "##.##", "..a.."]
path_a = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (1, 2), (0, 2), (0, 3), (0, 4)]
assert _replay(grid_a, path_a) == 8
assert shortest_path(grid_a) == 8
assert count_shortest_paths(grid_a) == 1        # this grid's detour is the only shortest route

grid_locked = ["S.AT", ".#.#", ".#a#"]           # key locked behind the very door it would open
assert shortest_path(grid_locked) == -1
assert count_shortest_paths(grid_locked) == 0


# a plain (row, col)-only BFS is wrong on grid_a: it reports -1, the true answer is 8 (fresh code,
# independent of _transitions/_bfs, isolating the visited-set bug described in Part 1)
def _naive_wrong_shortest_path(grid):
    R, C = len(grid), len(grid[0])
    start = target = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'T':
                target = (r, c)
    dist = {start: 0}
    q = deque([(start[0], start[1], 0)])
    while q:
        r, c, mask = q.popleft()
        if (r, c) == target:
            return dist[(r, c)]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C) or grid[nr][nc] == '#':
                continue
            ch = grid[nr][nc]
            if ch.isupper() and ch not in ('S', 'T') and not (mask >> (ord(ch) - ord('A'))) & 1:
                continue
            nmask = mask | (1 << (ord(ch) - ord('a'))) if ch.islower() else mask
            if (nr, nc) not in dist:                    # BUG: keyed by cell only, ignores mask
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc, nmask))
    return -1


assert _naive_wrong_shortest_path(grid_a) == -1 and shortest_path(grid_a) == 8

# --- Part 2: two distinct shortest paths, listed explicitly ---
grid_c = ["S..AT", ".#.##", "..a.."]
path_c1 = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (1, 2), (0, 2), (0, 3), (0, 4)]
path_c2 = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 3), (0, 4)]
assert _replay(grid_c, path_c1) == 8 and _replay(grid_c, path_c2) == 8 and path_c1 != path_c2
assert shortest_path(grid_c) == 8
assert count_shortest_paths(grid_c) == 2

# --- Part 3: the mud example, two paths of different length but equal cost ---
grid_d = ["S~~T", "...."]
path_d1 = [(0, 0), (0, 1), (0, 2), (0, 3)]
path_d2 = [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (0, 3)]
assert len(path_d1) - 1 == 3 and len(path_d2) - 1 == 5     # three moves, and five moves, as stated
assert _replay(grid_d, path_d1, weighted=True) == 5
assert _replay(grid_d, path_d2, weighted=True) == 5 and len(path_d1) != len(path_d2)
assert shortest_path_weighted(grid_d) == 5
assert count_shortest_paths_weighted(grid_d) == 2

# no mud anywhere -> the weighted functions agree with Parts 1-2 exactly
for g in (grid_a, grid_c, grid_locked):
    assert shortest_path_weighted(g) == shortest_path(g)
    assert count_shortest_paths_weighted(g) == count_shortest_paths(g)


# --- an independent brute force: counts raw move sequences by exact length, one length at a time,
# stopping at the first length that reaches T (a walk that reaches T is never extended further). Never
# calls _parse, _transitions, _bfs, _dijkstra or any other solution helper. ---
def _bf_unweighted(grid, max_len):
    R, C = len(grid), len(grid[0])
    start = target = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 'S':
                start = (r, c)
            if grid[r][c] == 'T':
                target = (r, c)

    def moves(r, c, mask):
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            cell = grid[nr][nc]
            if cell == '#':
                continue
            if cell.isupper() and cell not in ('S', 'T'):
                if not (mask >> (ord(cell) - ord('A'))) & 1:
                    continue
                out.append((nr, nc, mask))
            elif cell.islower():
                out.append((nr, nc, mask | (1 << (ord(cell) - ord('a')))))
            else:
                out.append((nr, nc, mask))
        return out

    counts = {(start[0], start[1], 0): 1}   # number of raw sequences of the current length at each state
    for length in range(1, max_len + 1):
        nxt = {}
        for (r, c, mask), n in counts.items():
            for nr, nc, nmask in moves(r, c, mask):
                key = (nr, nc, nmask)
                nxt[key] = nxt.get(key, 0) + n
        counts = nxt
        hit = sum(n for (r, c, _m), n in counts.items() if (r, c) == target)
        if hit:
            return length, hit
    return None, 0


def _bf_weighted(grid, max_cost):
    R, C = len(grid), len(grid[0])
    start = target = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 'S':
                start = (r, c)
            if grid[r][c] == 'T':
                target = (r, c)

    def moves(r, c, mask):
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            cell = grid[nr][nc]
            if cell == '#':
                continue
            step = 2 if cell == '~' else 1
            if cell.isupper() and cell not in ('S', 'T'):
                if not (mask >> (ord(cell) - ord('A'))) & 1:
                    continue
                out.append((nr, nc, mask, step))
            elif cell.islower():
                out.append((nr, nc, mask | (1 << (ord(cell) - ord('a'))), step))
            else:
                out.append((nr, nc, mask, step))
        return out

    buckets = [dict() for _ in range(max_cost + 3)]
    buckets[0][(start[0], start[1], 0)] = 1
    for cst in range(max_cost + 1):
        if not buckets[cst]:
            continue
        hit = sum(n for (r, c, _m), n in buckets[cst].items() if (r, c) == target)
        if hit:
            return cst, hit
        for (r, c, mask), n in buckets[cst].items():
            for nr, nc, nmask, step in moves(r, c, mask):
                buckets[cst + step][(nr, nc, nmask)] = buckets[cst + step].get((nr, nc, nmask), 0) + n
    return None, 0


def _random_grid(rng, R, C, wall_prob, mud_prob=0.0):
    """A small grid with one key/door pair, filtered to connected (ignoring locks and mud) so that
    almost every instance is genuinely solvable within the brute force's move/cost bound below."""
    while True:
        rows = [['.' for _ in range(C)] for _ in range(R)]
        for r in range(R):
            for c in range(C):
                if rng.random() < wall_prob:
                    rows[r][c] = '#'
        free = [(r, c) for r in range(R) for c in range(C)]
        rng.shuffle(free)
        if len(free) < 4:
            continue
        (sr, sc), (tr, tc), (kr, kc), (dr_, dc_) = free[:4]
        rows[sr][sc], rows[tr][tc], rows[kr][kc], rows[dr_][dc_] = 'S', 'T', 'a', 'A'
        if mud_prob:
            for r in range(R):
                for c in range(C):
                    if rows[r][c] == '.' and rng.random() < mud_prob:
                        rows[r][c] = '~'
        grid = ["".join(row) for row in rows]
        seen, stack = {(sr, sc)}, [(sr, sc)]
        while stack:                                  # ignore-locks connectivity: a cheap pre-filter only
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#' and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        if (tr, tc) in seen:                           # necessary, not sufficient, for real reachability
            return grid


MAX_LEN = 14
saw_count_gt_1 = False
for seed in range(300):
    rng = random.Random(seed)
    R, C = rng.choice([(3, 3), (3, 4)])
    grid = _random_grid(rng, R, C, wall_prob=0.2)
    bf_len, bf_cnt = _bf_unweighted(grid, MAX_LEN)
    sol_len, sol_cnt = shortest_path(grid), count_shortest_paths(grid)
    if bf_len is None:
        # inconclusive within MAX_LEN moves -- still enough to rule out a wrong SMALL answer
        assert sol_len == -1 or sol_len > MAX_LEN, (seed, grid, sol_len)
        continue
    assert sol_len == bf_len and sol_cnt == bf_cnt, (seed, grid, sol_len, sol_cnt, bf_len, bf_cnt)
    saw_count_gt_1 = saw_count_gt_1 or bf_cnt > 1
assert saw_count_gt_1     # the count > 1 rule was actually exercised by the fuzzing, not just stated

MAX_COST = 18
for seed in range(300):
    rng = random.Random(seed + 10_000)
    R, C = rng.choice([(3, 3), (3, 4)])
    mud_prob = 0.0 if seed % 3 == 0 else 0.3
    grid = _random_grid(rng, R, C, wall_prob=0.2, mud_prob=mud_prob)
    bf_cost, bf_cnt = _bf_weighted(grid, MAX_COST)
    sol_cost, sol_cnt = shortest_path_weighted(grid), count_shortest_paths_weighted(grid)
    if bf_cost is None:
        assert sol_cost == -1 or sol_cost > MAX_COST, (seed, grid, sol_cost)
        continue
    assert sol_cost == bf_cost and sol_cnt == bf_cnt, (seed, grid, sol_cost, sol_cnt, bf_cost, bf_cnt)
    if mud_prob == 0.0:                     # no mud on this seed -> must agree with Parts 1-2 exactly
        assert sol_cost == shortest_path(grid) and sol_cnt == count_shortest_paths(grid), seed

print("all checks passed")
```

</details>

</details>
