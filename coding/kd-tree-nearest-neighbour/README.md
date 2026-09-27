# k-d Tree: Nearest Neighbours and Range Queries

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · trees and spatial search | ★★★☆☆ | Hard | RS · RE · SWE · MLE | kd-tree, nearest-neighbour, pruning, heap, range-search, curse-of-dimensionality | 3 parts / 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

Points are the rows of a NumPy array `P` of shape `(n, d)`, `n >= 1`, with floating-point coordinates; a
point is identified by its row index into `P`. The distance between two points $p, q \in \mathbb{R}^d$ is
Euclidean, $\lVert p - q \rVert_2 = \sqrt{\sum_{i=0}^{d-1} (p_i - q_i)^2}$. A *k-d tree* over `P` is a binary
tree in which every node stores one point's index, a *split axis* — an integer in $0, \dots, d-1$ — and,
implicitly, that point's coordinate on the axis as the *split value*; the root splits on axis $0$, and each
level cycles to the next axis, so a node at depth $t$ (the root is depth $0$) splits on axis $t \bmod d$.

### Part 1 — Build a balanced tree

```py
class KDTree:
    def __init__(self, P: np.ndarray) -> None: ...
```

`P` has shape `(n, d)`; the constructor raises `ValueError` if not, including `n == 0`. Build the tree
recursively: a node receiving $m \ge 1$ points at depth $t$ (the root receives all $n$, at depth $0$) splits
on axis $a = t \bmod d$. Order those $m$ points by `(coordinate on axis a, index)` ascending; the point at
position `m // 2` becomes this node. The `m // 2` points before it, in that same order, recursively become
the left subtree at depth $t+1$; the points after it recursively become the right subtree at depth $t+1$. A
side with no points assigned to it has no child there.

For every `P`, the resulting tree's height — the number of nodes on its longest root-to-leaf path — is at
most $\lceil \log_2(n+1) \rceil$, and the constructor must run in $O(n \log^2 n)$ time or better.

```text
P = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]     # index 0 .. 5, d = 2

                    5 (7,2)              axis x, depth 0
                   /        \
             1 (5,4)          2 (9,6)    axis y, depth 1
             /     \          /
       0 (2,3)   3 (4,7)  4 (8,1)        axis x, depth 2
```

Point $5$ ($x=7$) is the median of all six points on axis $x$. Among the left three, $\{0,1,3\}$ ($y=3,4,7$),
sorted by $y$ they are $0,1,3$, so point $1$ ($y=4$, the middle one) becomes that subtree's node, with $0$ and
$3$ as its two leaves. Among the right two, $\{2,4\}$ ($y=6,1$), sorted by $y$ they are $4,2$, so point $2$
($y=6$) becomes that subtree's node, with $4$ as its single left leaf. The height is $3$, matching
$\lceil \log_2(6+1) \rceil = 3$ exactly.

### Part 2 — Nearest neighbour

```py
    def nearest(self, q: np.ndarray) -> int: ...   # index of the nearest point; ties -> smallest index
```

`q` has shape `(d,)`; raises `ValueError` if not. Return the index of the point in `P` closest to `q` by
Euclidean distance; if two or more points tie for closest, return the smallest such index. `nearest` must
search the tree rather than scan every point: at each node, recurse into the child on the side of the split
containing `q` first; recurse into the other child only when it could still hold a point at least as close as
the best one found anywhere in the search so far — otherwise skip that child, and everything under it,
without visiting it.

```text
q = (4, 2)

distances² to every point: 0 -> 5   1 -> 5   2 -> 41   3 -> 25   4 -> 17   5 -> 9
                                 ^ points 0 and 1 tie for closest

visit root, point 5 (7,2), d² = 9                          best so far: 5 (d² = 9)
  split axis x at 7; q_x = 4 is on the near (left) side -> search that child first
    visit point 1 (5,4), d² = 5                             improves: best so far: 1 (d² = 5)
    split axis y at 4; q_y = 2 is on the near (left) side -> search that child first
      visit point 0 (2,3), d² = 5                            ties point 1; index 0 < 1 -> best so far: 0
      other child (point 3): (2 - 4)² = 4 <= 5 -> could still tie or improve -> must visit
      visit point 3 (4,7), d² = 25                            no improvement
  other child (points 2, 4): (4 - 7)² = 9 > 5 -> cannot tie or improve -> skipped entirely

nearest(np.array([4.0, 2.0])) == 0
```

### Part 3 — k nearest neighbours and box counting

```py
    def knn(self, q: np.ndarray, k: int) -> list[int]: ...   # k indices sorted by (distance, index); k <= n
    def count_in_box(self, lo: np.ndarray, hi: np.ndarray) -> int: ...   # points with lo <= p <= hi coordinate-wise
```

`knn` returns the `k` indices with the smallest `(distance², index)` pairs, ascending — the same ordering
`nearest` uses to break ties, extended to a list of length `k`. `1 <= k <= n`; raises `ValueError` otherwise.
`count_in_box` returns the number of points `p` with `lo[i] <= p[i] <= hi[i]` for every axis `i` (a closed
box, so a point exactly on an edge counts); `lo` and `hi` have shape `(d,)`, raising `ValueError` if not. If
`lo[i] > hi[i]` for some axis `i`, the box is empty and the count is `0`.

Continuing `q = (4, 2)` from Part 2, whose squared distances to points $0,\dots,5$ were $5,5,41,25,17,9$:
sorted by `(distance², index)` the order is $0,1,5,4,3,2$, so `knn(q, 3) == [0, 1, 5]`.

With `lo = (4, 1)` and `hi = (8, 5)`: point $1$ $(5,4)$, point $4$ $(8,1)$ and point $5$ $(7,2)$ satisfy
$4 \le x \le 8$ and $1 \le y \le 5$ (point $4$ sits exactly on two edges of the box and still counts); point
$0$ has $x=2<4$, point $2$ has $x=9>8$, point $3$ has $y=7>5$.
`count_in_box(np.array([4.0, 1.0]), np.array([8.0, 5.0])) == 3`.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Three points are worth confirming before coding: whether $d$ is assumed low (the pruning argument in Part 2
and the range-query bound in Part 3 both assume it; brute force overtakes the tree once $d$ grows past a
couple of dozen, discussed in the Follow-ups); whether `P` may contain duplicate points, two rows with
identical coordinates (yes — nothing forbids it, and the build's index tie-break together with the search's
`(distance, index)` ordering handle it with no special case); and the tie rule itself, smallest index wins,
which is what makes the build order, `nearest` and `knn` well-defined single answers rather than "any point
that happens to qualify".

### Part 1

The build is a direct recursion on the rule stated above; the only real design decision is how to store the
result. Rather than a chain of Python node objects, the tree lives in four flat NumPy arrays indexed by node
id — `point`, `axis`, `left`, `right` — so every later search step is a handful of array indexing operations,
not attribute lookups through a chain of objects. The constructor assigns node ids in the order nodes are
created (the root gets id $0$), which is convenient but otherwise arbitrary; nothing later relies on the
numbering beyond $-1$ meaning "no child".

Recursion depth tracks the tree's own height, bounded below, so plain recursion never risks Python's
recursion limit — no explicit stack is needed here, unlike a build that could produce an arbitrarily
unbalanced tree.

**Height.** Of the $m$ points at a node, the left child receives exactly $\lfloor m/2 \rfloor$ of them and the
right child receives the rest, $m - \lfloor m/2 \rfloor - 1$, which is at most $\lfloor m/2 \rfloor$ too (one
point becomes the node itself) — so both children hold at most $\lfloor m/2 \rfloor$ points, a strict halving.
Applying this once per level, a node $t$ levels below the root holds at most $\lfloor n/2^t \rfloor$ points,
which first reaches $0$ once $2^t > n$ — and the least such $t$ is exactly $n$'s bit length,
$\lceil \log_2(n+1) \rceil$. So no leaf sits deeper than that, regardless of the coordinates or the order
points arrive in.

**Build time.** Sorting the $m$ points at one node by `(coordinate, index)` costs $O(m \log m)$. The nodes at
any one depth partition all $n$ points disjointly, so their sorts cost at most $O(n \log n)$ together at that
depth alone; with at most $\lceil \log_2(n+1) \rceil$ depths (the height bound just derived), the total is
$O(n \log^2 n)$.

```python
import numpy as np


class KDTree:
    """A k-d tree over the rows of P (shape (n, d)). Node t stores one point's row index in
    self.point[t], its split axis in self.axis[t], and its children's node ids in self.left[t] /
    self.right[t] (-1 when absent) -- flat arrays indexed by node id, not a chain of Python objects."""

    def __init__(self, P: np.ndarray) -> None:
        P = np.asarray(P, dtype=np.float64)
        if P.ndim != 2 or P.shape[0] < 1:
            raise ValueError("P must have shape (n, d) with n >= 1")
        self.P = P
        self.n, self.d = P.shape
        self.point = np.empty(self.n, dtype=np.int64)
        self.axis = np.empty(self.n, dtype=np.int64)
        self.left = np.full(self.n, -1, dtype=np.int64)
        self.right = np.full(self.n, -1, dtype=np.int64)
        self._next_node = 0
        self.root = self._build(list(range(self.n)), depth=0)

    def _build(self, indices: list[int], depth: int) -> int:
        if not indices:
            return -1
        axis = depth % self.d
        indices.sort(key=lambda i: (self.P[i, axis], i))    # NOTE: index breaks ties -- otherwise which
                                                              #       duplicate becomes this node is unspecified
        mid = len(indices) // 2
        node = self._next_node
        self._next_node += 1
        self.point[node] = indices[mid]
        self.axis[node] = axis
        self.left[node] = self._build(indices[:mid], depth + 1)
        self.right[node] = self._build(indices[mid + 1:], depth + 1)
        return node
```

### Part 2

Searching the near side first is what makes pruning possible at all: it tends to find a close point early,
giving later pruning decisions a small radius to test against. The test itself needs a lower bound on how
close *any* point on the far side of a split could possibly be to `q`. Part 1's build guarantees that, at a
node splitting on axis $a$ at value $s$, every point in the left subtree has coordinate $\le s$ on axis $a$
and every point in the right subtree has coordinate $\ge s$ (both sides are sorted around that same split
position). So if `q`'s own coordinate $q_a$ is on the left ($q_a \le s$), every point $p$ on the right
satisfies $p_a - q_a \ge s - q_a \ge 0$, hence

$$\lVert p-q \rVert_2^2 = \sum_i (p_i-q_i)^2 \ge (p_a-q_a)^2 \ge (s-q_a)^2,$$

since every summand is non-negative and $(s-q_a)^2$ is one of them (the symmetric argument bounds the left
side when $q_a \ge s$). Write $d_{\text{best}}^2$ for the best squared distance found anywhere in the search
so far. If $(q_a-s)^2 > d_{\text{best}}^2$, no point on the far side can even match $d_{\text{best}}$, and the
whole subtree is skipped. But the tie rule compares `(distance², index)`, not distance alone: a point on the
far side that only *ties* $d_{\text{best}}$ still wins if its index is smaller, so the far side must still be
visited whenever it could tie, not only when it could strictly improve — exactly the boundary case
$(q_a-s)^2 = d_{\text{best}}^2$. So the pruning test is `<=`, not `<`:

$$\text{visit the far side} \iff (q_a-s)^2 \le d_{\text{best}}^2.$$

Comparing squared distances throughout avoids a `sqrt` on every node visited, since it changes no
comparison's outcome (every quantity compared above is non-negative).

```python
def nearest(self, q: np.ndarray) -> int:
    return self._nearest_visited(q)[0]


def _nearest_visited(self, q: np.ndarray) -> tuple[int, int]:
    """Same answer as nearest(q), plus the number of nodes visited -- nearest() uses this directly,
    and the checks below measure the count to quantify how much pruning actually saves."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (self.d,):
        raise ValueError(f"q must have shape ({self.d},)")
    best_d2 = np.inf
    best_idx = self.n          # NOTE: sentinel larger than every real index 0..n-1, so any point beats it once
    visited = 0

    def recurse(node: int) -> None:
        nonlocal best_d2, best_idx, visited
        if node == -1:
            return
        visited += 1
        pi = self.point[node]
        diff_vec = self.P[pi] - q
        d2 = float(diff_vec @ diff_vec)             # NOTE: squared distance -- sqrt is monotone, so wasted work
        if (d2, pi) < (best_d2, best_idx):           # NOTE: tuple compare enforces "ties -> smallest index"
            best_d2, best_idx = d2, pi
        a = self.axis[node]
        diff = q[a] - self.P[pi, a]
        near, far = (self.left[node], self.right[node]) if diff <= 0 else (self.right[node], self.left[node])
        recurse(near)
        if diff * diff <= best_d2:                  # NOTE: <= not < -- the far side can still hold a tie at
            recurse(far)                             #       the exact same distance with a smaller index

    recurse(self.root)
    return int(best_idx), visited


KDTree.nearest = nearest
KDTree._nearest_visited = _nearest_visited
```

Every visited node does $O(1)$ work beyond its two recursive calls, so total time is $O(\text{nodes
visited})$. Descending straight to the leaf nearest `q` — always following the near side — costs exactly the
tree's height, $O(\log n)$; how many *extra* nodes get visited depends on how often the far side's pruning
test passes when it need not. For $n$ points spread roughly uniformly over a region of volume $V$ in $d$
dimensions, each point accounts for about $V/n$ of that volume on average, so the typical distance from a
point to its nearest neighbour is of order $(V/n)^{1/d}$ — the radius of a ball of that volume. Once
$d_{\text{best}}$ has shrunk to about this size, a sibling subtree is skipped unless its region comes within
that same small radius of `q`, true of only a small fraction of the siblings met while descending for
well-spread data and a low, fixed $d$ — so the total visited stays close to the guaranteed $O(\log n)$
descent itself, which is what the checks below measure directly, rather than assert. In the worst case
pruning can fail completely: if every point in `P` sits at the same location, every split's value equals
every point's coordinate on that axis, so $(q_a-s)^2$ is one fixed number at every node — and, once any point
has been compared, that same number is already one term of the sum making up $d_{\text{best}}^2$, so it can
never exceed $d_{\text{best}}^2$. The far side is then always visited, and `nearest` visits all $n$ nodes.

### Part 3

`knn` generalises the same idea from one best candidate to the $k$ best. A bounded max-heap holds up to $k$
candidates as `(-distance², -index)` pairs, so the *worst* of the current $k$ — the one a new candidate must
beat to earn a place — sits at `heap[0]`: negating both fields turns "smallest `(distance², index)`" into
"largest negated pair", which is what a min-heap (Python's `heapq`) keeps at the top when read as the
smallest of the negated values. A node's point is pushed outright while the heap holds fewer than $k$
entries, and afterwards replaces `heap[0]` only when it is strictly better — a smaller `(distance², index)`
pair, i.e. a larger negated one. The pruning test generalises Part 2's directly, with $d_{\text{best}}^2$
replaced by the worst of the current top $k$: nothing may be pruned before $k$ candidates exist (any node
could still be needed just to reach $k$ at all), and afterwards the far side is visited exactly when
$(q_a-s)^2 \le$ that worst distance, by the identical lower-bound argument as Part 2.

```python
import heapq


def knn(self, q: np.ndarray, k: int) -> list[int]:
    return self._knn_visited(q, k)[0]


def _knn_visited(self, q: np.ndarray, k: int) -> tuple[list[int], int]:
    """Same answer as knn(q, k), plus the number of nodes visited -- knn() uses this directly, and the
    checks below use the count to confirm the worst case visits every node, exactly as Part 2's does."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (self.d,):
        raise ValueError(f"q must have shape ({self.d},)")
    if not (1 <= k <= self.n):
        raise ValueError(f"k must satisfy 1 <= k <= {self.n}")
    heap: list[tuple[float, int]] = []        # max-heap via negation; heap[0] is the current worst of the top k
    visited = 0

    def recurse(node: int) -> None:
        nonlocal visited
        if node == -1:
            return
        visited += 1
        pi = self.point[node]
        diff_vec = self.P[pi] - q
        d2 = float(diff_vec @ diff_vec)
        key = (-d2, -int(pi))
        if len(heap) < k:
            heapq.heappush(heap, key)
        elif key > heap[0]:                    # NOTE: strictly better than the current worst -- the same
            heapq.heapreplace(heap, key)        #       (distance, index) ordering as nearest's tie rule
        a = self.axis[node]
        diff = q[a] - self.P[pi, a]
        near, far = (self.left[node], self.right[node]) if diff <= 0 else (self.right[node], self.left[node])
        recurse(near)
        worst_d2 = -heap[0][0] if len(heap) == k else np.inf   # NOTE: nothing may be pruned before k are found
        if diff * diff <= worst_d2:
            recurse(far)

    recurse(self.root)
    ordered = sorted((-neg_d2, -neg_idx) for neg_d2, neg_idx in heap)
    return [idx for _d2, idx in ordered], visited


KDTree.knn = knn
KDTree._knn_visited = _knn_visited
```

Each visited node does $O(\log k)$ work for its heap push or replace, against $O(1)$ for `nearest`'s single
running best; the same descent-plus-pruning argument as Part 2 applies with $k$ candidates to fill instead of
$1$, so for well-spread low-dimensional data the nodes visited stay close to $O(k + \log n)$, for
$O((k+\log n)\log k)$ time overall, and the identical all-points-coincide construction as Part 2 forces the
worst case $O(n \log k)$.

`count_in_box` cannot afford to check every point individually once `P` is large: the same recursion instead
carries the query box's intersection with each node's own region — an implicit bounding box, narrowed on the
way down exactly as the split values imply — and short-circuits the moment that region is resolved. Each node
also needs the size of its own subtree, computed once, after Part 1's build, by a plain post-order pass over
the already-built `left`/`right` arrays (no need to touch `_build` itself), and cached for reuse: recomputing
it on every call would cost $O(n)$ every time, drowning out the bound derived below.

```python
def _fill_sizes(self, node: int, size: np.ndarray) -> int:
    if node == -1:
        return 0
    total = 1 + self._fill_sizes(self.left[node], size) + self._fill_sizes(self.right[node], size)
    size[node] = total
    return total


def _sizes(self) -> np.ndarray:
    cached = getattr(self, "_size_cache", None)
    if cached is None:
        cached = np.zeros(self.n, dtype=np.int64)
        self._fill_sizes(self.root, cached)
        self._size_cache = cached           # NOTE: cached once -- reused by every later count_in_box call
    return cached


def count_in_box(self, lo: np.ndarray, hi: np.ndarray) -> int:
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != (self.d,) or hi.shape != (self.d,):
        raise ValueError(f"lo and hi must have shape ({self.d},)")
    if np.any(lo > hi):
        return 0                             # NOTE: an empty box holds no points -- not an error
    cell_lo = np.full(self.d, -np.inf)
    cell_hi = np.full(self.d, np.inf)
    return self._count(self.root, cell_lo, cell_hi, lo, hi, self._sizes())


def _count(self, node: int, cell_lo: np.ndarray, cell_hi: np.ndarray,
           lo: np.ndarray, hi: np.ndarray, size: np.ndarray) -> int:
    if node == -1:
        return 0
    if np.any(cell_hi < lo) or np.any(cell_lo > hi):
        return 0                                            # NOTE: this node's whole region misses the box
    if np.all(cell_lo >= lo) and np.all(cell_hi <= hi):
        return int(size[node])                                # NOTE: whole region inside -- O(1) via the size
    pi = self.point[node]
    p = self.P[pi]
    total = 1 if np.all((lo <= p) & (p <= hi)) else 0
    a = self.axis[node]
    s = p[a]
    left_hi = cell_hi.copy()
    left_hi[a] = min(left_hi[a], s)
    right_lo = cell_lo.copy()
    right_lo[a] = max(right_lo[a], s)
    total += self._count(self.left[node], cell_lo, left_hi, lo, hi, size)
    total += self._count(self.right[node], right_lo, cell_hi, lo, hi, size)
    return total


KDTree._fill_sizes = _fill_sizes
KDTree._sizes = _sizes
KDTree.count_in_box = count_in_box
KDTree._count = _count
```

For a fixed, low $d$, a node's region either lies entirely inside the box (resolved in $O(1)$ above), entirely
outside ($O(1)$ again), or is crossed by one of the box's $2d$ bounding hyperplanes — one for each of `lo` and
`hi` on each axis — and must be examined further. Follow one such hyperplane, constant on axis $j$: a split on
axis $j$ sends it into at most one child (the split value falls on one side or the other), but a split on any
other axis sends it into both children, since that split says nothing about coordinate $j$. Axes cycle every
$d$ levels, so across each block of $d$ consecutive levels the number of nodes this one hyperplane still
crosses is multiplied by at most $2^{d-1}$ (doubling on the $d-1$ levels that split some other axis, unchanged
on the one level that splits axis $j$), while the points held per surviving node shrink by $2^d$. After $t$
such blocks, at most $2^{t(d-1)}$ nodes are still crossed, holding about $n/2^{td}$ points each; this stops
being useful once that count is $O(1)$, at $t \approx (\log_2 n)/d$, giving $2^{t(d-1)} = n^{1-1/d}$ crossed
nodes for this one hyperplane, and $O(n^{1-1/d})$ (constant $d$) summed over all $2d$ of them. A query that has
to list every matching point pays an extra $O(m)$ walking the point-by-point contents of the fully-inside
subtrees it finds; `count_in_box` never does that, since it adds a stored size in $O(1)$ instead — so it costs
$O(n^{1-1/d})$ with no $+m$ term at all.

### Follow-ups

- **A uniform grid, searched ring by ring.** Choose a cell side $s$ and map each point to integer cell
  coordinates $\lfloor p_i / s \rfloor$ per axis $i$, keeping a dictionary from cell to the points inside it;
  for a query $q$ in cell $c$, scan cells in rings $r = 0, 1, 2, \dots$, where ring $r$ holds every cell whose
  Chebyshev distance from $c$ — the largest per-axis index difference — equals $r$. Every point not yet
  scanned after ring $r$ sits in some cell at Chebyshev distance at least $r+1$ from $c$, so on whichever axis
  realises that gap its cell lies at least $r+1$ steps from $c$'s own, and since $q$ itself lies inside cell
  $c$, that already puts the point more than $r \cdot s$ from $q$ along that one axis alone — hence more than
  $r \cdot s$ away in full Euclidean distance too — so once the best distance found after ring $r$ is at most
  $r \cdot s$, nothing unscanned can improve on it and the search may stop. In three dimensions, with
  $s \approx (V/n)^{1/3}$ — about one point per cell, for $n$ points spread uniformly over a volume $V$ — a
  query scans $O(1)$ cells in expectation, but on clustered data most cells sit empty or overfull and the grid
  degrades, which is where the k-d tree, or an octree, wins: an octree splits a cube into eight equal
  children until a leaf holds at most a few points, searches with the same branch-and-bound as Part 2 but
  with cubes in place of half-spaces, and adapts to density like the k-d tree, though it splits at cube
  centres rather than medians, so its depth is not bounded by $O(\log n)$ for arbitrary inputs.
- **One query only.** With a single query and no preprocessing allowed, an $O(n)$ scan is optimal — every
  point must be examined, since any point left unexamined could be the nearest — and vectorised with NumPy it
  is fast in practice, with `np.argpartition` giving the $k$ nearest in $O(n)$ time without a full sort.
  Building the tree costs $O(n \log^2 n)$ (Part 1), which only pays off once enough queries amortise it.
- **Why the pruning stops helping as $d$ grows.** Axes cycle every $d$ levels, so across the tree's
  $O(\log n)$ height each axis is split only about $(\log_2 n)/d$ times — for fixed $n$, a split's margin
  shrinks toward the size of the whole domain as $d$ grows, while the current-best radius still has to reach
  the typical inter-point spacing, $(V/n)^{1/d}$, which grows the same way. Once the two are comparable, the
  pruning test rarely fails, the search ball intersects nearly every cell it meets, and most of the tree ends
  up visited anyway; a single vectorised pass computing all $n$ distances at once typically overtakes the
  tree somewhere around $d \approx 10\text{–}20$.
- **Approximate nearest neighbours.** Stopping the search once a fixed number of leaves $t$ has been visited,
  or relaxing the pruning test to $(q_a-s)^2 \le (1+\epsilon)^2 d_{\text{best}}^2$ for some $\epsilon>0$, both
  trade a small, controllable chance of missing the true nearest point (though never by more than a factor
  $1+\epsilon$, or beyond the visited budget) for pruning far more aggressively.
- **Ball trees and graph-based indexes.** A ball tree replaces axis-aligned splits with nested hyperspheres,
  keeping the same $O(\log n)$-height, prune-by-a-lower-bound structure while degrading more gracefully in
  higher dimensions, since a ball privileges no single coordinate axis; graph-based indexes such as HNSW
  instead build a navigable proximity graph with no tree at all and answer a query with a greedy best-first
  walk across it, and are what production embedding-search systems use once $d$ reaches the hundreds or
  thousands typical of learned representations.
- **Dynamic insertions.** Splicing a new point straight into an existing leaf can push the tree's height past
  the bound derived in Part 1, since nothing about a single insertion preserves the median-split invariant;
  the standard fix rebuilds only the smallest subtree that both contains the right place for the new point and
  stays balanced once it is added, or, more simply, batches insertions and rebuilds the whole tree once enough
  have accumulated to amortise the cost.

<details>
<summary>Checks (runnable)</summary>

```python
import random

# --- the worked example: exact tree structure and height ---
P_ex = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)], dtype=np.float64)
tree_ex = KDTree(P_ex)


def _tree_repr(tree, node):
    """(point index, axis, left subtree or None, right subtree or None) -- exact structural comparison."""
    if node == -1:
        return None
    return (int(tree.point[node]), int(tree.axis[node]),
            _tree_repr(tree, tree.left[node]), _tree_repr(tree, tree.right[node]))


def _height(tree, node):
    if node == -1:
        return 0
    return 1 + max(_height(tree, tree.left[node]), _height(tree, tree.right[node]))


expected_tree = (5, 0, (1, 1, (0, 0, None, None), (3, 0, None, None)), (2, 1, (4, 0, None, None), None))
assert _tree_repr(tree_ex, tree_ex.root) == expected_tree
assert _height(tree_ex, tree_ex.root) == 3 == (6).bit_length()   # the bound is tight on this example

# --- the worked example: nearest, matching the trace above exactly ---
q_ex = np.array([4.0, 2.0])
d2_ex = ((P_ex - q_ex) ** 2).sum(axis=1)
assert list(d2_ex) == [5, 5, 41, 25, 17, 9]
idx_ex, visited_ex = tree_ex._nearest_visited(q_ex)
assert idx_ex == 0
assert visited_ex == 4        # root, 1, 0, 3 -- points 2 and 4 are pruned, exactly as traced above
assert tree_ex.nearest(q_ex) == 0

# --- the worked example: knn and count_in_box ---
assert tree_ex.knn(q_ex, 3) == [0, 1, 5]
assert tree_ex.knn(q_ex, 6) == [0, 1, 5, 4, 3, 2]     # k == n: every point, fully ordered
lo_ex, hi_ex = np.array([4.0, 1.0]), np.array([8.0, 5.0])
assert tree_ex.count_in_box(lo_ex, hi_ex) == 3
assert tree_ex.count_in_box(np.array([100.0, 100.0]), np.array([200.0, 200.0])) == 0   # nothing in the box
assert tree_ex.count_in_box(np.array([5.0, 5.0]), np.array([1.0, 1.0])) == 0            # lo > hi: empty, no error

# --- errors ---
for bad_P in (np.zeros((0, 2)), np.zeros((3, 2, 2))):
    try:
        KDTree(bad_P)
        assert False, "expected ValueError"
    except ValueError:
        pass
try:
    tree_ex.nearest(np.array([1.0, 2.0, 3.0]))
    assert False, "expected ValueError"
except ValueError:
    pass
for bad_k in (0, 7):
    try:
        tree_ex.knn(q_ex, bad_k)
        assert False, "expected ValueError"
    except ValueError:
        pass

# --- n == 1: no children, every query answered from the single point ---
single = KDTree(np.array([[3.0, 4.0]]))
assert single.nearest(np.array([0.0, 0.0])) == 0
assert single.knn(np.array([0.0, 0.0]), 1) == [0]
assert single.count_in_box(np.array([0.0, 0.0]), np.array([10.0, 10.0])) == 1
assert single.count_in_box(np.array([10.0, 10.0]), np.array([20.0, 20.0])) == 0

# --- boundary case for the <= pruning test: two duplicate points, so the near side (visited
# unconditionally) and the far side (visited only if the pruning test passes) tie exactly, and the
# smaller index sits on the far side -- this catches the <= in Part 2 being weakened to plain < ---
P_tie = np.array([[0.0], [0.0]])
tree_tie = KDTree(P_tie)
q_tie = np.array([3.0])
idx_tie, visited_tie = tree_tie._nearest_visited(q_tie)
assert idx_tie == 0, idx_tie            # index 0 has the smaller index and must win the tie
assert visited_tie == 2, visited_tie    # both nodes visited -- the far side was not wrongly pruned
assert tree_tie.knn(q_tie, 1) == [0]


# --- independent brute force, from the statement alone; never touches KDTree's own helpers ---
def _brute_nearest(P, q):
    d2 = ((P - q) ** 2).sum(axis=1)
    best = d2.min()
    return int(np.nonzero(d2 == best)[0].min())


def _brute_knn(P, q, k):
    d2 = ((P - q) ** 2).sum(axis=1)
    order = sorted(range(len(P)), key=lambda i: (d2[i], i))
    return order[:k]


def _brute_count_in_box(P, lo, hi):
    if np.any(lo > hi):
        return 0
    return int(np.all((P >= lo) & (P <= hi), axis=1).sum())


# --- height bound across many random (n, d) ---
for trial in range(300):
    rng = random.Random(trial)
    n = rng.randint(1, 300)
    d = rng.choice([1, 2, 3, 5])
    P = np.array([[rng.uniform(-10, 10) for _ in range(d)] for _ in range(n)])
    tree = KDTree(P)
    assert _height(tree, tree.root) <= n.bit_length(), (n, d)

# --- cross-checked against the brute force: plain random points, an integer grid with heavy ties and
# duplicate coordinates, and explicit duplicate rows -- for d in {1, 2, 3, 5} ---
for d in (1, 2, 3, 5):
    for trial in range(60):
        rng = random.Random(d * 10_000 + trial)
        n = rng.randint(1, 80)
        kind = trial % 3
        if kind == 0:
            P = np.array([[rng.uniform(-20, 20) for _ in range(d)] for _ in range(n)])
        elif kind == 1:
            P = np.array([[float(rng.randint(-3, 3)) for _ in range(d)] for _ in range(n)])   # many ties
        else:
            base = [[rng.uniform(-5, 5) for _ in range(d)] for _ in range(max(1, n // 3))]
            P = np.array([rng.choice(base) for _ in range(n)])                                 # duplicate rows
        tree = KDTree(P)
        for _ in range(5):
            if rng.random() < 0.5:
                q = P[rng.randrange(n)].copy()             # sometimes query exactly at a stored point
            else:
                q = np.array([rng.uniform(-20, 20) for _ in range(d)])
            assert tree.nearest(q) == _brute_nearest(P, q), (d, trial, kind, q)
            k = rng.randint(1, n)
            assert tree.knn(q, k) == _brute_knn(P, q, k), (d, trial, kind, q, k)
            lo = np.array([rng.uniform(-20, 5) for _ in range(d)])
            hi = lo + np.array([rng.uniform(0, 20) for _ in range(d)])
            assert tree.count_in_box(lo, hi) == _brute_count_in_box(P, lo, hi), (d, trial, kind, lo, hi)

# --- worst case: every point at the same location -- nearest and knn must visit all n nodes, as derived ---
for d in (1, 2, 3):
    rng = random.Random(2000 + d)
    n = 50
    z = np.array([rng.uniform(-5, 5) for _ in range(d)])
    P = np.tile(z, (n, 1))
    tree = KDTree(P)
    q = np.array([rng.uniform(-5, 5) for _ in range(d)])
    _, visited = tree._nearest_visited(q)
    assert visited == n, (d, visited, n)
    _, knn_visited = tree._knn_visited(q, k=5)
    assert knn_visited == n, (d, knn_visited, n)

# --- pruning in practice: 2-D, 4,000 uniform points, average nodes visited per query well under n ---
rng_np = np.random.default_rng(0)
n_big = 4000
P_big = rng_np.uniform(0.0, 1.0, size=(n_big, 2))
tree_big = KDTree(P_big)
queries = rng_np.uniform(0.0, 1.0, size=(300, 2))
visited_counts = [tree_big._nearest_visited(q)[1] for q in queries]
average_visited = sum(visited_counts) / len(visited_counts)
assert average_visited < 0.05 * n_big, average_visited   # measured well under 1%; 5% leaves a generous margin

# --- Follow-up check: a uniform grid searched ring by ring, independent of KDTree ---
import itertools


def _grid_cells(P, s):
    """Bucket every point's row index by its integer cell coordinates floor(p / s), per axis."""
    cells: dict[tuple[int, ...], list[int]] = {}
    for i in range(len(P)):
        cell = tuple(int(np.floor(P[i, j] / s)) for j in range(P.shape[1]))
        cells.setdefault(cell, []).append(i)
    return cells


def _ring_offsets(r, d):
    """Integer cell offsets at Chebyshev distance exactly r from the origin, in d dimensions."""
    if r == 0:
        return [tuple(0 for _ in range(d))]
    return [off for off in itertools.product(range(-r, r + 1), repeat=d) if max(abs(o) for o in off) == r]


def _grid_nearest(P, q, s):
    """Ring-by-ring uniform-grid nearest neighbour (Follow-up 1); never touches KDTree. Stops once
    the best distance found is at most r * s -- the derived bound -- never merely at the first
    non-empty ring (see the naive counterexample below)."""
    d = P.shape[1]
    cells = _grid_cells(P, s)
    q_cell = tuple(int(np.floor(q[j] / s)) for j in range(d))
    best_d2, best_idx = np.inf, len(P)   # NOTE: sentinel index -- matches the page's (distance, index) tie rule
    r = 0
    while True:
        for off in _ring_offsets(r, d):
            for i in cells.get(tuple(q_cell[j] + off[j] for j in range(d)), ()):
                diff = P[i] - q
                d2 = float(diff @ diff)
                if (d2, i) < (best_d2, best_idx):
                    best_d2, best_idx = d2, i
        if best_d2 <= (r * s) ** 2:          # NOTE: the derived stopping rule -- squared, so no sqrt needed
            break
        r += 1
    return best_idx, best_d2


def _grid_nearest_first_nonempty_ring(P, q, s):
    """The naive, wrong stopping rule: return the best point in the first ring that contains any
    point at all, regardless of how far that point actually is -- used only to show the rule is wrong."""
    d = P.shape[1]
    cells = _grid_cells(P, s)
    q_cell = tuple(int(np.floor(q[j] / s)) for j in range(d))
    r = 0
    while True:
        best_d2, best_idx, found = np.inf, len(P), False
        for off in _ring_offsets(r, d):
            for i in cells.get(tuple(q_cell[j] + off[j] for j in range(d)), ()):
                found = True
                diff = P[i] - q
                d2 = float(diff @ diff)
                if (d2, i) < (best_d2, best_idx):
                    best_d2, best_idx = d2, i
        if found:
            return best_idx, best_d2
        r += 1


# --- cross-checked against the same independent brute force as above: uniform, strongly clustered
# (four tight Gaussian blobs), and heavily-tied 3-D data (only 64 distinct integer positions) ---
rng_grid = np.random.default_rng(1)
for grid_kind in ("uniform", "clustered", "ties"):
    if grid_kind == "uniform":
        n_grid = 200
        P_grid = rng_grid.uniform(0.0, 1.0, size=(n_grid, 3))
    elif grid_kind == "clustered":
        centers = np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.2], [0.2, 0.8, 0.9], [0.8, 0.9, 0.8]])
        P_grid = np.concatenate([c + rng_grid.normal(0.0, 0.01, size=(40, 3)) for c in centers])
        n_grid = len(P_grid)
    else:
        P_grid = rng_grid.integers(0, 4, size=(200, 3)).astype(np.float64)   # only 64 distinct
        n_grid = len(P_grid)                                                  # positions -- heavy ties
    s_grid = (1.0 / n_grid) ** (1 / 3)          # s ~ (V / n)^(1/3) with V = 1; any s > 0 is correct
    for trial in range(20):
        if rng_grid.random() < 0.3:
            q_grid = P_grid[rng_grid.integers(n_grid)].copy()             # sometimes exactly at a point
        else:
            q_grid = rng_grid.uniform(P_grid.min(axis=0), P_grid.max(axis=0))
        expected_idx = _brute_nearest(P_grid, q_grid)     # same tie rule: ties broken by smallest index
        got_idx, _ = _grid_nearest(P_grid, q_grid, s_grid)
        assert got_idx == expected_idx, (grid_kind, trial, q_grid)

# --- queries far outside the data's bounding box must still terminate and answer correctly ---
P_far = rng_grid.uniform(0.0, 1.0, size=(120, 3))
s_far = (1.0 / 120) ** (1 / 3)
for far_q in (np.array([2.0, 2.0, 2.0]), np.array([-1.5, 2.5, -1.0]), np.array([0.5, 0.5, 3.0])):
    expected_idx = _brute_nearest(P_far, far_q)
    got_idx, _ = _grid_nearest(P_far, far_q, s_far)
    assert got_idx == expected_idx, far_q

# --- the naive "stop at the first non-empty ring" rule is wrong: q sits near its cell's boundary,
# one point in the far corner of q's own cell (ring 0, distance ~1.20), a much closer point just
# across that boundary (ring 1, distance 0.02) -- the naive rule stops at ring 0 and misses it ---
s_naive = 1.0
q_naive = np.array([0.99, 0.5, 0.5])
P_naive = np.array([[0.01, 0.01, 0.01],
                     [1.01, 0.5, 0.5]])
correct_idx, _ = _grid_nearest(P_naive, q_naive, s_naive)
naive_idx, _ = _grid_nearest_first_nonempty_ring(P_naive, q_naive, s_naive)
assert correct_idx == 1, correct_idx
assert naive_idx == 0, naive_idx
assert naive_idx != correct_idx

print("all checks passed")
```

</details>

</details>
