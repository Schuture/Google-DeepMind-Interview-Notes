# Painting a Fence with the Fewest Strokes

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · greedy and divide and conquer | ★★★☆☆ | Medium | SWE · MLE · Intern | greedy, divide-and-conquer, arrays, range-minimum, proof-of-optimality | 3 parts / 45 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

A fence has `n` planks of width 1 standing side by side, indexed `0` to `n - 1`; plank `i` has an
integer height `h[i] >= 0`. Model the fence as unit cells: cell `(i, y)` exists if and only if
`0 <= y < h[i]`, so plank `i` contributes exactly `h[i]` cells, stacked from `y = 0` (ground level) up
to `y = h[i] - 1`. A *horizontal stroke* paints the cells `(l, y), (l + 1, y), ..., (r, y)` of one row
`y`, for some `l <= r`; it is allowed only when every one of those cells exists, i.e. `h[i] > y` for
every `i` from `l` to `r` — a stroke can never cross a plank too short to reach row `y`. A *vertical
stroke* paints every cell of one plank, i.e. all of `(i, 0), (i, 1), ..., (i, h[i] - 1)`. A cell may be
painted more than once, since strokes are free to overlap, and every part below asks for the fewest
strokes — horizontal and vertical counted together — that paint every cell of the fence at least once.

### Part 1 — Horizontal strokes only

```py
def min_horizontal_strokes(h: list[int]) -> int: ...
```

Using only horizontal strokes, return the minimum number needed to paint every cell of the fence. `h`
may be empty (no planks, nothing to paint) and may contain `0`s (a plank of height `0` contributes no
cells and needs none).

```text
h = [2, 1, 3, 3, 1, 2]

plank:        0  1  2  3  4  5
y=2 (top)     .  .  #  #  .  .    run {2,3}             1 stroke
y=1           #  .  #  #  .  #    runs {0}, {2,3}, {5}  3 strokes
y=0 (ground)  #  #  #  #  #  #    run {0,1,2,3,4,5}     1 stroke

min_horizontal_strokes(h) == 1 + 3 + 1 == 5
```

### Part 2 — Planks change height

```py
class Fence:
    def __init__(self, h: list[int]): ...
    def set_height(self, i: int, new_height: int) -> None: ...   # O(1)
    def strokes(self) -> int: ...                                  # O(1): Part 1's answer for the current heights
```

`Fence(h)` starts from a height list exactly like Part 1's; the number of planks `n = len(h)` never
changes afterwards. `set_height(i, new_height)` requires `0 <= i < n` and `new_height >= 0`, and
replaces `h[i]` with `new_height` in `O(1)` time — it must not re-scan the other planks. `strokes()`
returns, in `O(1)` time, exactly what `min_horizontal_strokes` would return for the fence's current
heights: the same answer Part 1 computes from scratch, kept up to date instead of recomputed.

```text
f = Fence([1, 1, 1])
f.strokes()                # -> 1: one stroke across all three planks, all at height 1

f.set_height(1, 3)         # h becomes [1, 3, 1]
f.strokes()                # -> 3: plank 1 now sticks up above both of its neighbours

f.set_height(0, 3)         # h becomes [3, 3, 1]
f.strokes()                # -> 3: unchanged in total, though the internal boundary terms shift

f.set_height(1, 0)         # h becomes [3, 0, 1]
f.strokes()                # -> 4: plank 1 dropping to height 0 splits row 0 into two separate runs
```

### Part 3 — Vertical strokes allowed

```py
def min_strokes(h: list[int]) -> int: ...   # horizontal and vertical strokes, together
```

Return the minimum number of strokes — horizontal and vertical, in any combination — to paint every
cell of the fence, under the same rules as above.

```text
min_strokes([1, 5, 1]) == 2
# one horizontal stroke across all three planks paints row 0, the only row planks 0 and 2 reach;
# one vertical stroke on plank 1 then finishes its rows 1-4 -- far better than the 5 strokes
# min_horizontal_strokes([1, 5, 1]) needs (rows 1, 2, 3, 4 each isolated to plank 1 alone)

min_strokes([2, 1, 3, 3, 1, 2]) == 5
# equal to min_horizontal_strokes of the same heights (Part 1's example) -- for this shape, no use
# of vertical strokes beats painting purely row by row
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points are worth confirming before coding: a cell may be painted more than once, so strokes are
free to overlap with no penalty for it; and every stroke, horizontal or vertical, must stay entirely on
cells that exist — a horizontal stroke can never cross a plank too short to reach its row, and a
height-`0` plank never takes part in any stroke.

### Part 1

In row `y`, the existing cells are exactly the planks with `h[i] > y`, and they split into maximal runs
of consecutive such planks; one horizontal stroke per run is both necessary (nothing paints the gap
between two different runs) and sufficient (a single stroke spans an entire run), so the answer is the
total number of runs, summed over every row from `0` up to `max(h) - 1`.

Count runs a different way: a run starts at plank `i` in row `y` exactly when plank `i` has a cell at
row `y` but its left neighbour does not — `h[i] > y` and (`i == 0`, or `h[i - 1] <= y`), treating a
virtual `h[-1] = 0` so the `i == 0` case needs no separate rule. Fixing `i`, the rows for which this
holds are `h[i - 1] <= y < h[i]`, an interval of length `max(0, h[i] - h[i - 1])`; since every run-start
belongs to exactly one plank and one row, summing that length over every plank counts every run exactly
once:
$$\sum_i \max(0,\, h_i - h_{i-1}), \qquad h_{-1} = 0.$$

```python
def min_horizontal_strokes(h: list[int]) -> int:
    total = 0
    previous = 0
    for height in h:
        total += max(0, height - previous)   # NOTE: never negative -- a shrink starts no new run
        previous = height
    return total
```

One pass over `h`: `O(n)` time, `O(1)` extra space beyond the input.

### Part 2

`min_horizontal_strokes` is a sum of one term per plank, `term[i] = max(0, h[i] - h[i - 1])` (with
`h[-1] = 0`), and changing a single height only changes the two terms that mention it: `term[i]` itself,
which reads `h[i]` and `h[i - 1]`, and `term[i + 1]`, which reads `h[i + 1]` and `h[i]` — every other
term reads neither the old nor the new value of `h[i]` and is left untouched. Keeping a running `total`
alongside the array and the per-plank terms, and updating only these at-most-two terms on every call,
turns `strokes()` into a single lookup.

```python
class Fence:
    def __init__(self, h: list[int]) -> None:
        self._h = list(h)
        self._term = [0] * len(self._h)
        self._total = 0
        previous = 0
        for i, height in enumerate(self._h):
            self._term[i] = max(0, height - previous)
            self._total += self._term[i]
            previous = height

    def set_height(self, i: int, new_height: int) -> None:
        n = len(self._h)
        left = self._h[i - 1] if i > 0 else 0        # NOTE: i == 0 means the virtual h[-1] == 0
        new_term_i = max(0, new_height - left)
        self._total += new_term_i - self._term[i]
        self._term[i] = new_term_i
        if i + 1 < n:                                 # NOTE: no term i + 1 to fix when i is the last plank
            new_term_next = max(0, self._h[i + 1] - new_height)
            self._total += new_term_next - self._term[i + 1]
            self._term[i + 1] = new_term_next
        self._h[i] = new_height

    def strokes(self) -> int:
        return self._total
```

`__init__` costs `O(n)`, the same one pass as Part 1. `set_height` reads and writes a constant number of
array entries regardless of `n`, so it runs in `O(1)`, and so does `strokes()`, a single attribute
lookup.

### Part 3

For a contiguous range of planks `[l, r]` that is *isolated* at some base height `b` — every plank in it
has `h[i] > b`, and the range is bounded on both sides, by the array's ends or by a plank of height
`<= b` — any stroke that paints one of its cells at a row `y >= b` is either a vertical stroke on one of
its own planks, or a horizontal stroke lying entirely inside `[l, r]`: a horizontal stroke at a row
`y >= b` cannot cross the boundary, since the bounding plank has no cell there. So the minimum number of
strokes to paint `[l, r]`'s cells at rows `>= b`, call it `T(l, r, b)`, is a self-contained subproblem,
and the full answer is the sum of `T` over the maximal runs of planks with `h[i] > 0` (a height-`0`
plank never takes part in a stroke and always breaks a run, at every row).

Within one such range, let `m = min(h[l..r])`. Two strategies are available: paint every plank
vertically, costing `r - l + 1`; or paint each of the `m - b` rows from `b` to `m - 1` with one stroke
spanning all of `[l, r]` — every plank has a cell there, since `h[i] >= m` throughout the range — and
recurse with base `m` on the maximal sub-ranges where `h[i] > m`, which are isolated at base `m` by the
same argument as above (the planks left out, at exactly height `m`, are already fully painted and block
any row-`m`-or-above stroke from crossing them). `T(l, r, b)` is at most the smaller of these two costs,
since both are valid constructions; it remains to show it is never smaller. Take an optimal solution `S`
for this range and split it into `S_low`, its horizontal strokes at some row in `[b, m)`, and `S_rest`,
everything else — vertical strokes, and horizontal strokes at rows `>= m`. No stroke of `S_low` reaches
a row `>= m`, so every cell at a row `>= m` is covered by some stroke of `S_rest`, and each such stroke
is confined to one of the sub-ranges above `m` — a horizontal one cannot cross a plank at exactly height
`m`, which has no cell there, and a vertical one only reaches a row `>= m` if its own plank's height
exceeds `m` — so those sub-ranges are covered using only some of `S_rest`'s strokes, giving `S_rest` at
least the sum of `T` over them. For `S_low`: if some plank `p` in `[l, r]` has no vertical stroke of its own, then
cell `(p, y)` — which exists for every row `y` from `b` to `m - 1`, since `h[p] >= m` — can only be
covered by a horizontal stroke at that row, a member of `S_low`, and different rows need different
strokes, since one horizontal stroke lies in a single row; so `S_low` has at least `m - b` strokes, and
`S` has at least `m - b` plus the sum of `T` over the sub-ranges at base `m`. Otherwise every plank in
`[l, r]` has its own vertical stroke, giving `S` at least `r - l + 1` strokes directly, one distinct
stroke per plank. Either way, the size of `S` is at least one of the two constructions' costs, hence at
least their minimum. Writing `(l_k, r_k)` for the maximal sub-ranges of `[l, r]` with `h_i > m`:

$$T(l, r, b) = \min\left(r - l + 1,\ \ (m - b) + \sum_k T(l_k, r_k, m)\right).$$

```python
def _runs_above(h: list[int], lo: int, hi: int, floor: int) -> list[tuple[int, int]]:
    """Maximal contiguous sub-ranges of [lo, hi] where every plank's height exceeds floor."""
    runs = []
    i = lo
    while i <= hi:
        if h[i] <= floor:
            i += 1
            continue
        j = i
        while j <= hi and h[j] > floor:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _solve_range(h: list[int], l: int, r: int, base: int) -> int:
    """Minimum strokes to paint cells (i, y) with l <= i <= r and base <= y < h[i]."""
    answer: dict[tuple[int, int], int] = {}
    info: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]] = {}
    stack = [(l, r, base, False)]   # NOTE: explicit stack -- a sorted array needs depth O(n), past Python's limit
    while stack:
        lo, hi, b, expanded = stack.pop()
        if not expanded:
            m = min(h[lo:hi + 1])
            children = _runs_above(h, lo, hi, m)
            info[(lo, hi)] = (m, children)
            stack.append((lo, hi, b, True))
            for cl, cr in children:
                stack.append((cl, cr, m, False))
        else:
            m, children = info[(lo, hi)]
            horizontal = (m - b) + sum(answer[child] for child in children)
            vertical = hi - lo + 1
            answer[(lo, hi)] = min(vertical, horizontal)   # NOTE: (lo, hi) is a safe key alone -- ranges never collide
    return answer[(l, r)]


def min_strokes(h: list[int]) -> int:
    n = len(h)
    return sum(_solve_range(h, lo, hi, 0) for lo, hi in _runs_above(h, 0, n - 1, 0))
```

Each visit to a range costs time proportional to its length, to find its minimum and its sub-ranges; the
recursion tree has at most `n` nodes in total, since every visit permanently removes at least one plank
(the ones tied for the minimum, which never recur), though the ranges different nodes touch can still
overlap heavily from one level to the next. A strictly sorted array (heights increasing or decreasing
throughout) is the worst case: the minimum sits at one end, so each level peels off exactly one plank,
giving `n` levels of sizes `n, n - 1, ..., 1` and `O(n^2)` total work — and recursion depth `O(n)` to
match, well past Python's default recursion limit of `1000` for a sorted array of a little over a
thousand planks. That is exactly why `_solve_range` simulates the recursion with an explicit stack
rather than calling itself: every range is pushed once, to compute its minimum and its children, and
popped again only once those children's answers are already in `answer`, at which point its own
`min(vertical, horizontal)` can be taken.

### Follow-ups

- **Recovering the strokes, not just their count.** `_solve_range` already decides, at every range,
  between painting it all vertically or painting `m - b` full-width rows and recursing; recording that
  choice (and, in the vertical case, which planks) during the same post-order pass, then walking the
  chosen branches top-down, reconstructs an explicit list of strokes achieving the optimum, without
  changing the count.
- **`O(n log n)` worst case, with a range-minimum structure.** Splitting on every plank tied for the
  current minimum is not required for correctness — splitting at any single occurrence of it, found by an
  `O(1)`-query sparse table built in `O(n log n)`, gives the same optimal value, since a plank left
  exactly at the new base immediately becomes a zero-cost range of its own, while every range still
  shrinks by at least one plank; the recursion tree then has `O(n)` nodes and the whole computation costs
  `O(n log n)`, independent of how the heights happen to be ordered.
- **The same formula, a different problem.** Building an array from all zeros using operations that add
  `1` to every element of a chosen contiguous range, minimising the number of operations needed to reach a
  target `h`, has exactly Part 1's closed form as its answer: one "layer" of range-increments at height
  `y` corresponds to one horizontal stroke at row `y`, by the identical run-counting argument.
- **Painting a 2-D grid.** With a 2-D grid of cells and strokes that are full rows or full columns of
  existing cells, there is no longer a single height profile to recurse on: which cells a row-stroke and a
  column-stroke each cover interacts globally across the grid, and the problem becomes an instance of
  minimum set cover restricted to rows and columns, for which no efficient exact algorithm of this kind is
  known.

<details>
<summary>Checks (runnable)</summary>

```python
import random

# --- the worked examples of the statement ---
h = [2, 1, 3, 3, 1, 2]
assert min_horizontal_strokes(h) == 5


def _row_runs(h: list[int], y: int) -> int:
    """Number of maximal contiguous runs of planks with a cell at row y -- straight from the cell definition."""
    n = len(h)
    runs = 0
    i = 0
    while i < n:
        if h[i] <= y:
            i += 1
            continue
        runs += 1
        while i < n and h[i] > y:
            i += 1
    return runs


assert (_row_runs(h, 2), _row_runs(h, 1), _row_runs(h, 0)) == (1, 3, 1)   # the rows traced in the statement

f = Fence([1, 1, 1])
assert f.strokes() == 1
f.set_height(1, 3)
assert f.strokes() == 3
f.set_height(0, 3)
assert f.strokes() == 3
f.set_height(1, 0)
assert f.strokes() == 4

assert min_strokes([1, 5, 1]) == 2
assert min_strokes([2, 1, 3, 3, 1, 2]) == 5 == min_horizontal_strokes([2, 1, 3, 3, 1, 2])

# --- edge cases: no planks, and planks of height 0 ---
assert min_horizontal_strokes([]) == 0
assert min_strokes([]) == 0
assert min_horizontal_strokes([0, 0, 0]) == 0
assert min_strokes([0, 0, 0]) == 0
assert Fence([]).strokes() == 0


# --- Part 1: independent brute force, straight from the cell definition, row by row ---
def _brute_horizontal(h: list[int]) -> int:
    n = len(h)
    total = 0
    for y in range(max(h, default=0)):
        i = 0
        while i < n:
            if h[i] <= y:
                i += 1
                continue
            total += 1
            while i < n and h[i] > y:
                i += 1
    return total


rng = random.Random(0)
for _ in range(500):
    n = rng.randint(0, 40)
    heights = [rng.randint(0, 12) for _ in range(n)]
    assert min_horizontal_strokes(heights) == _brute_horizontal(heights), heights

# --- Part 2: Fence kept in sync against fresh recomputation, after random updates ---
rng = random.Random(1)
for _ in range(80):
    n = rng.randint(1, 25)
    initial = [rng.randint(0, 12) for _ in range(n)]
    fence = Fence(initial)
    mirror = list(initial)
    assert fence.strokes() == min_horizontal_strokes(mirror)
    for _ in range(40):
        i = rng.randrange(n)
        new_height = rng.randint(0, 12)
        fence.set_height(i, new_height)
        mirror[i] = new_height
        assert fence.strokes() == min_horizontal_strokes(mirror), (initial, i, new_height, mirror)


# --- Part 3: exhaustive brute force over every subset of vertically-painted planks (small n) ---
def _brute_min_strokes(h: list[int]) -> int:
    n = len(h)
    if n == 0:
        return 0
    row_runs = []                     # row_runs[y]: maximal existing-cell runs at row y, as (start, end)
    for y in range(max(h)):
        runs = []
        i = 0
        while i < n:
            if h[i] <= y:
                i += 1
                continue
            j = i
            while j < n and h[j] > y:
                j += 1
            runs.append((i, j))
            i = j
        row_runs.append(runs)
    best = None
    for mask in range(1 << n):        # mask's set bits: the planks painted vertically
        cost = bin(mask).count("1")
        for runs in row_runs:
            for start, end in runs:
                span = (1 << end) - (1 << start)      # bit positions [start, end)
                if span & ~mask:                        # some plank of this run is not in the mask
                    cost += 1
        best = cost if best is None else min(best, cost)
    return best


rng = random.Random(2)
for _ in range(300):
    n = rng.randint(0, 9)
    heights = [rng.randint(0, 4) for _ in range(n)]
    expected = _brute_min_strokes(heights)
    got = min_strokes(heights)
    assert got == expected, (heights, got, expected)
    assert got <= min_horizontal_strokes(heights)     # vertical strokes can only help, never hurt
    assert got <= n                                     # painting every plank vertically is always valid

print("all checks passed")
```

</details>

</details>
