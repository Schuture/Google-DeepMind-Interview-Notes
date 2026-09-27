# Unit Conversions as a Weighted Graph

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · graph construction and traversal | ★★★☆☆ | Medium | SWE · MLE · RE · Intern | graph, dfs, bfs, weighted-union-find, consistency-check, floating-point | 3 parts / 45 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

A *fact* is a triple `(a, b, r)`: two distinct unit names `a` and `b` (arbitrary, non-empty strings) and
a real number `r > 0`, stating that one unit of `a` equals `r` units of `b` — for example,
`("km", "m", 1000.0)` states that 1 km equals 1000 m. A list of `facts` defines a directed, weighted
graph: each fact contributes an edge `a -> b` of weight `r` and the inverse edge `b -> a` of weight
`1 / r` (if 1 `a` is `r` `b`, then 1 `b` is `1 / r` `a`). The graph's nodes — its *units* — are every
distinct name that appears in at least one fact, as `a` or as `b`; two units are *connected* when some
path of edges joins them, and the *conversion factor* from one to the other along a path is the product
of that path's edge weights. Parts 1 and 3 assume `facts` is internally consistent: every path between
the same two units carries the same conversion factor, so it never matters which path a search happens
to find — Part 2 is about the case where that assumption fails. A returned factor is a product, or in
Part 3 also a ratio, of floating-point numbers, so it can carry ordinary rounding error; every comparison
below, in the checks and inside `rel_tol`, uses a tolerance, never `==`.

### Part 1 — Answer conversion queries

```py
def convert(facts: list[tuple[str, str, float]], value: float, src: str, dst: str) -> float | None: ...
```

Return `value` expressed in `dst`: build the graph from `facts` once, then multiply `value` by the
conversion factor along any path from `src` to `dst`. `convert` never raises for a name it does not
recognise; instead it returns `None` when `src` or `dst` names a unit that appears in no fact at all, and
also when both are known but no path connects them. If `src == dst`, return `value` unchanged, provided
that unit appears in at least one fact — an unmentioned unit used as both `src` and `dst` still returns
`None`, exactly as any other unknown unit.

```text
facts = [
    ("km", "m", 1000.0),
    ("m", "cm", 100.0),
    ("inch", "cm", 2.54),
    ("foot", "inch", 12.0),
    ("mile", "foot", 5280.0),
    ("byte", "bit", 8.0),
]

convert(facts, 2.0, "mile", "cm")
# path mile -> foot -> inch -> cm, three edges, weights 5280.0, 12.0, 2.54
# factor = 5280.0 * 12.0 * 2.54 = 160934.4
# -> 2.0 * 160934.4 = 321868.8

convert(facts, 5.0, "km", "km")     # src == dst, and "km" appears in a fact -> value unchanged
# -> 5.0

convert(facts, 1.0, "m", "parsec")  # "parsec" appears in no fact at all
# -> None

convert(facts, 1.0, "km", "byte")   # both appear, but "byte"/"bit" is a separate, unconnected component
# -> None
```

### Part 2 — Detect contradictions

```py
def first_contradiction(facts: list[tuple[str, str, float]], rel_tol: float = 1e-9) -> int: ...
```

Process `facts` in order, building up the graph one fact at a time. A fact `(a, b, r)` *contradicts* the
facts before it when `a` and `b` are already connected by them and the conversion factor those earlier
facts imply between `a` and `b` is not close to `r`: writing `f` for that implied factor, precisely when
`not math.isclose(f, r, rel_tol=rel_tol)` (equivalently, `abs(f - r)` exceeds `rel_tol` times the larger
of `abs(f)` and `abs(r)`). A fact naming a unit not seen before, or bridging two components not yet
connected to each other, never contradicts anything, since there is nothing yet to compare it with.
Return the index of the first fact that contradicts the facts before it, or `-1` if no fact does.

```text
facts = [
    ("a", "b", 2.0),     # 0
    ("c", "d", 5.0),     # 1
    ("b", "c", 3.0),     # 2
    ("a", "d", 100.0),   # 3
]

first_contradiction(facts)
# index 0: "a", "b" unseen before -> add; 1 a = 2 b
# index 1: "c", "d" unseen before -> add; 1 c = 5 d (a separate component, so far)
# index 2: "b" and "c" are in different components -> bridges them, nothing to contradict yet;
#          1 b = 3 c, so the merged component now implies 1 a = 2 * 3 = 6 c = 6 * 5 = 30 d
# index 3: "a" and "d" ARE already connected; the implied factor is 30.0, but this fact states 100.0:
#          |30.0 - 100.0| / max(30.0, 100.0) = 0.7, far above rel_tol -> contradicts
# -> 3
```

### Part 3 — Many facts and queries, interleaved

`Converter` supports an *online* sequence of `add_fact` and `query` calls, in any order, each in
near-constant amortised time — no call may re-scan every fact seen before it. `add_fact(a, b, r)` records
one more fact, again with `a != b` and `r > 0`: if `a` and `b` are already connected and the implied
factor conflicts with `r` (the same `rel_tol = 1e-9` rule as Part 2), it changes nothing and returns
`False`; otherwise it incorporates the fact — adding a new unit, or merging two components, as needed —
and returns `True`. `query(src, dst)` returns the factor such that 1 `src` equals that many `dst`, never
raising, under the same connectivity rules as Part 1's `convert`: `None` if either unit was never accepted
by `add_fact`, or if the two are not connected by the facts accepted so far.

```py
class Converter:
    def add_fact(self, a: str, b: str, r: float) -> bool: ...        # False (and no change) if it contradicts
    def query(self, src: str, dst: str) -> float | None: ...          # factor such that 1 src = factor dst
```

`Converter()` takes no arguments and starts out with no known units.

```text
c = Converter()
c.add_fact("km", "m", 1000.0)      # -> True
c.add_fact("m", "cm", 100.0)       # -> True
c.query("km", "cm")                # -> approximately 100000.0 (ordinary rounding error; see above)
c.add_fact("inch", "cm", 2.54)     # -> True
c.query("km", "inch")              # -> approximately 39370.0787
c.add_fact("km", "inch", 39370.0)  # -> False: the true factor is ~39370.0787, not 39370.0 -- rejected
c.query("km", "inch")              # -> approximately 39370.0787, exactly as before the rejected call
c.add_fact("mile", "foot", 5280.0) # -> True, a new component so far
c.query("mile", "cm")              # -> None: not yet connected to the km/m/cm/inch component
c.add_fact("foot", "inch", 12.0)   # -> True, bridges the two components
c.query("mile", "cm")              # -> approximately 160934.4, matching Part 1's worked example above
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points are worth confirming before coding: the direction convention for `r` — a fact `(a, b, r)`
gives the `a -> b` edge weight `r` and `b -> a` weight `1 / r`, never the other way round — and what every
part returns for a disconnected or unknown unit: `None`, always, never an exception.

### Part 1

From `src`, an iterative depth-first search — an explicit list used as a stack, so a long chain of units
never risks Python's recursion limit — accumulates the product of edge weights on the way, stopping the
instant `dst` is reached.

```python
from collections import defaultdict


def _add_edges(graph: dict[str, list[tuple[str, float]]], a: str, b: str, r: float) -> None:
    """One fact contributes two directed edges: a -> b weighted r, and its inverse."""
    graph[a].append((b, r))
    graph[b].append((a, 1.0 / r))    # NOTE: the reverse edge divides -- it is 1 / r, never r again


def _build_graph(facts: list[tuple[str, str, float]]) -> dict[str, list[tuple[str, float]]]:
    graph: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for a, b, r in facts:
        _add_edges(graph, a, b, r)
    return graph


def _path_factor(graph: dict[str, list[tuple[str, float]]], src: str, dst: str) -> float | None:
    """Product of edge weights along some path from src to dst (None if none exists). Assumes graph
    was built from consistent facts, so it never matters which path a search happens to find."""
    if src not in graph or dst not in graph:
        return None
    if src == dst:
        return 1.0
    visited = {src}
    stack = [(src, 1.0)]             # NOTE: an explicit stack -- iterative, so no recursion-depth limit
    while stack:
        unit, acc = stack.pop()
        for neighbour, weight in graph[unit]:
            product = acc * weight   # NOTE: repeated multiplication -- floating-point error can accumulate
            if neighbour == dst:
                return product
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append((neighbour, product))
    return None


def convert(facts: list[tuple[str, str, float]], value: float, src: str, dst: str) -> float | None:
    factor = _path_factor(_build_graph(facts), src, dst)
    return None if factor is None else value * factor
```

Building the graph once costs $O(F)$ for $F = $ `len(facts)`, two directed edges per fact. Every
reachable node is pushed onto the stack at most once (`visited` prevents a second push), and every edge
out of a popped node is examined once, so the search itself costs $O(V + E)$, where $V$ is the number of
distinct units and $E = 2F$; a call to `convert`, including the graph build, therefore costs $O(V + E)$
overall — linear in the size of the input, however far apart `src` and `dst` turn out to be.

### Part 2

`first_contradiction` grows the same kind of graph one fact at a time and, before adding fact `i`, asks
`_path_factor` — unchanged from Part 1 — for the factor `a -> b` implied by the facts already added. That
call returns `None` when `a` or `b` is new, or when the two sit in different components; either way there
is nothing yet to contradict, so the fact is simply added. When it returns an actual factor, `a` and `b`
are already connected, and the fact is checked against it with `math.isclose`; the first fact to fail that
check is returned at once, without looking at anything after it.

```python
import math


def first_contradiction(facts: list[tuple[str, str, float]], rel_tol: float = 1e-9) -> int:
    graph: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for i, (a, b, r) in enumerate(facts):
        implied = _path_factor(graph, a, b)
        if implied is not None and not math.isclose(implied, r, rel_tol=rel_tol):
            return i                          # NOTE: only an already-connected pair can have an implied factor
        _add_edges(graph, a, b, r)
    return -1
```

Every one of the $F$ facts triggers at most one $O(V + E)$ search over the graph built from the facts so
far, for $O(F \cdot (V + E))$ overall: each contradiction check costs no more than one of Part 1's
queries. That is fine for a batch of facts processed once, but Part 3 needs every `add_fact` and `query`
call to cost near-constant time regardless of how many facts came before it, which rules out a fresh
search per call — it needs an incremental data structure instead: a *weighted union-find*.

### Part 3

A weighted union-find keeps every unit in a rooted tree, one tree per connected component, and represents
a fact through the trees' shapes rather than through explicit edges. Give every unit $u$ a parent pointer
and a weight $w(u)$ such that

$$1\,u = w(u) \cdot \mathrm{parent}(u),$$

with a root its own parent and $w(\mathrm{root}) = 1$; composing weights while walking parent pointers up
to the root — the same product-along-a-path rule as Part 1 — gives, for any $u$, the factor from $u$ to
its root. `_find(x)` walks from `x` to its root recursively and, unwinding, *compresses* the path: once
the recursive call has established that `x`'s current parent `p` has factor `w` to the root, `x`'s own
factor to the root is $w(x) \cdot w$ — its factor to `p`, times `p`'s factor to the root — so `x` is
repointed directly at the root with that composed weight; a lookup starting from `x` again later then
costs one hop, not the whole original chain.

`add_fact(a, b, r)` first makes sure `a` and `b` exist (a new unit starts as its own root, weight `1.0`,
size `1`), then finds both roots. Write $(R_a, w_a) = \mathrm{find}(a)$ and $(R_b, w_b) = \mathrm{find}(b)$,
so $1\,a = w_a R_a$ and $1\,b = w_b R_b$. Two cases:

- If $R_a = R_b$, both are already expressed over the same root, so the new fact can be checked directly:
  chaining $a \to R_a\ (=R_b) \to b$ gives an implied factor $a \to b$ of $w_a \cdot (1 / w_b)$ — the
  first leg is $a \to R_a$, namely $w_a$; the second is $R_b \to b$, the inverse of $b \to R_b = w_b$ —
  and the fact is accepted exactly when $w_a / w_b$ is close to $r$; otherwise `add_fact` changes nothing
  and returns `False` (the only thing that could already have happened is `_find`'s own path compression,
  which only flattens future lookups and never changes what any `query` returns, so it is harmless either
  way).
- If $R_a \neq R_b$, the two trees merge: attach whichever root has the smaller `_size` under the other
  (*union by size*), so no tree's depth needs to grow on most merges. Arrange the roles of `a` and `b`
  (swapping them, and replacing `r` with `1 / r`, when it is the other way round) so that $R_b$ is always
  the one attached under $R_a$; its new weight must satisfy $1\,R_b = w(R_b)\,R_a$, found by chaining
  $R_b \to b \to a \to R_a$ — the first leg inverts $b \to R_b = w_b$, the middle leg is the new fact read
  backwards ($b \to a = 1 / r$), and the last leg is $a \to R_a = w_a$:

  $$w(R_b) \;=\; \frac{1}{w_b} \cdot \frac{1}{r} \cdot w_a \;=\; \frac{w_a}{r\,w_b}.$$

`query(src, dst)` reuses exactly this $w_x / w_y$ reasoning for two units sharing a root — including when
`src == dst`, where $w_{\mathrm{src}} = w_{\mathrm{dst}}$ makes the ratio `1.0` automatically, no special
case needed, provided the unit is known at all.

```python
REL_TOL = 1e-9    # NOTE: shared with first_contradiction's default, so Part 2 and Part 3 agree on every fact


class Converter:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._weight: dict[str, float] = {}    # 1 unit = weight[unit] * parent[unit]
        self._size: dict[str, int] = {}

    def _find(self, x: str) -> tuple[str, float]:
        """Path-compressing find. Returns (root, w) with 1 x = w * root."""
        if self._parent[x] == x:
            return x, 1.0
        root, w = self._find(self._parent[x])
        self._weight[x] *= w             # NOTE: compose with the OLD weight[x] first, then overwrite it
        self._parent[x] = root           # path compression -- x now points directly at the root
        return root, self._weight[x]

    def add_fact(self, a: str, b: str, r: float) -> bool:
        for u in (a, b):
            if u not in self._parent:
                self._parent[u] = u
                self._weight[u] = 1.0
                self._size[u] = 1
        root_a, w_a = self._find(a)               # 1 a = w_a * root_a
        root_b, w_b = self._find(b)                # 1 b = w_b * root_b
        if root_a == root_b:
            return math.isclose(w_a / w_b, r, rel_tol=REL_TOL)   # NOTE: no mutation on either outcome here
        if self._size[root_a] < self._size[root_b]:
            root_a, root_b = root_b, root_a
            w_a, w_b = w_b, w_a
            r = 1.0 / r                            # NOTE: swapping which side is "a" inverts the stated factor
        self._parent[root_b] = root_a
        self._weight[root_b] = w_a / (r * w_b)     # derived above: 1 root_b = (w_a / (r * w_b)) * root_a
        self._size[root_a] += self._size[root_b]
        return True

    def query(self, src: str, dst: str) -> float | None:
        if src not in self._parent or dst not in self._parent:
            return None
        root_src, w_src = self._find(src)
        root_dst, w_dst = self._find(dst)
        if root_src != root_dst:
            return None
        return w_src / w_dst                       # same w_x / w_y reasoning as the case above
```

Union by size bounds every tree's depth by $O(\log n)$ for $n$ units: whenever a node's depth increases
by one, it is in the smaller of the two trees just merged, and the merged tree's size is the sum of both,
at least twice the smaller one's — a size cannot double more than $\log_2 n$ times before reaching $n$. So
`_find` costs $O(\log n)$ even with no path compression at all; path compression on top of union by size
is the classical combination (Tarjan) whose amortised cost per operation is $O(\alpha(n))$, the inverse
Ackermann function, under $5$ for any $n$ this page — or any real input — could reach, effectively
constant. `add_fact` and `query` each do two `_find` calls and $O(1)$ other work, so both run in
near-constant amortised time, and neither ever scans a fact it was not directly given.

### Follow-ups

- **Working in log space.** Every factor above is a product, or a ratio of products, of individual `r`
  values; summing $\log r$ along a path and exponentiating once at the end accumulates one rounding error
  instead of one per edge, and turns Part 3's union step into addition and subtraction of logs. The cost
  is that `rel_tol` no longer means quite the same thing: a fixed absolute tolerance on
  $|\log f - \log r|$ is close to a fixed *relative* tolerance on $f$ against $r$ only for small
  tolerances, so a value tuned in the linear domain has to be re-derived, not reused as-is.
- **Affine units.** A purely multiplicative fact cannot express Celsius versus Fahrenheit, where
  $F = 1.8\,C + 32$; the general case is $x_b = r\,x_a + s$ for a pair $(r, s)$, and composing two affine
  maps is itself affine, $(r_2, s_2) \circ (r_1, s_1) = (r_1 r_2,\ r_2 s_1 + s_2)$, so the union-find of
  Part 3 still works with every node storing an $(r, s)$ pair to its root instead of a single factor, and
  `_find`'s path compression composes pairs by that rule instead of multiplying two numbers.
- **Exact arithmetic.** Replacing every `float` above with `fractions.Fraction`, built from each `r`'s
  exact decimal text rather than from an already-rounded `float`, makes every comparison in Part 2 and
  Part 3 exact, so `rel_tol` disappears and two facts either agree exactly or do not; the cost is that a
  `Fraction`'s numerator and denominator can grow without bound as more facts are composed, so a long
  chain gets slower and more memory-hungry per operation, unlike a `float`'s fixed size.
- **Explaining a contradiction.** `first_contradiction` reports only the index of the first offending
  fact, not why it is wrong. Recovering the cycle it closes means having `_path_factor` return the
  sequence of earlier facts it walked from `a` to `b`, alongside the factor; the contradicting fact
  together with that path is exactly the minimal set of facts a caller would need to look at to fix the
  input.

<details>
<summary>Checks (runnable)</summary>

```python
import random

# --- the worked examples of the statement ---
facts = [
    ("km", "m", 1000.0),
    ("m", "cm", 100.0),
    ("inch", "cm", 2.54),
    ("foot", "inch", 12.0),
    ("mile", "foot", 5280.0),
    ("byte", "bit", 8.0),
]
assert convert(facts, 2.0, "mile", "cm") == 321868.8
assert convert(facts, 5.0, "km", "km") == 5.0
assert convert(facts, 1.0, "m", "parsec") is None
assert convert(facts, 1.0, "km", "byte") is None
assert convert(facts, 1.0, "gram", "gram") is None   # unmentioned unit, even as both src and dst
assert convert([], 1.0, "x", "x") is None             # empty facts: no unit is ever "known"

facts2 = [
    ("a", "b", 2.0),
    ("c", "d", 5.0),
    ("b", "c", 3.0),
    ("a", "d", 100.0),
]
assert first_contradiction(facts2) == 3
assert first_contradiction(facts2[:3]) == -1
assert first_contradiction([]) == -1

c = Converter()
assert c.query("km", "m") is None                      # nothing added yet
assert c.add_fact("km", "m", 1000.0) is True
assert c.add_fact("m", "cm", 100.0) is True
assert math.isclose(c.query("km", "cm"), 100000.0, rel_tol=1e-9)
assert c.add_fact("inch", "cm", 2.54) is True
assert math.isclose(c.query("km", "inch"), 1000.0 * 100.0 / 2.54, rel_tol=1e-9)
before = c.query("km", "inch")
assert c.add_fact("km", "inch", 39370.0) is False       # true factor is ~39370.0787..., not 39370.0
assert c.query("km", "inch") == before                  # rejected fact left the converter unchanged
assert c.add_fact("mile", "foot", 5280.0) is True
assert c.query("mile", "cm") is None                     # not yet connected to the km/m/cm/inch component
assert c.add_fact("foot", "inch", 12.0) is True
assert math.isclose(c.query("mile", "cm"), 160934.4, rel_tol=1e-9)
assert math.isclose(c.query("mile", "cm"), convert(facts, 1.0, "mile", "cm"), rel_tol=1e-9)

# --- Part 2 / Part 3 agree on the exact rel_tol = 1e-9 boundary, not only on grossly wrong facts ---
close_r, far_r = 2.0 * (1 + 4e-10), 2.0 * (1 + 2e-8)
assert first_contradiction([("p", "q", 2.0), ("p", "q", close_r)]) == -1   # inside rel_tol -> accepted
assert first_contradiction([("p", "q", 2.0), ("p", "q", far_r)]) == 1     # outside rel_tol -> rejected
c_tol = Converter()
assert c_tol.add_fact("p", "q", 2.0) is True
assert c_tol.add_fact("p", "q", close_r) is True
assert c_tol.add_fact("p", "q", far_r) is False

# --- the exact intermediate figures traced in the Part 2 worked example ---
assert convert(facts2[:3], 1.0, "a", "d") == 30.0    # the implied factor at the point fact 3 is checked
assert abs(30.0 - 100.0) / max(30.0, 100.0) == 0.7    # the relative-difference figure quoted in the trace

# --- the affine-composition formula from the "Affine units" follow-up, checked against direct composition ---
def _affine_compose(r1, s1, r2, s2):
    return r1 * r2, r2 * s1 + s2


rng_affine = random.Random(99)
for _ in range(20):
    r1, s1, r2, s2 = (rng_affine.uniform(-5.0, 5.0) for _ in range(4))
    x = rng_affine.uniform(-10.0, 10.0)
    direct = r2 * (r1 * x + s1) + s2                  # apply map1 then map2, independently of the formula
    r_c, s_c = _affine_compose(r1, s1, r2, s2)
    assert math.isclose(direct, r_c * x + s_c, rel_tol=1e-9, abs_tol=1e-9)


# --- shared random-generation helpers, independent of the solution's own code ---
def _random_units_and_hidden(rng, n_units, val_range=(0.2, 5.0)):
    units = [f"u{i}" for i in range(n_units)]
    hidden = {u: rng.uniform(*val_range) for u in units}
    return units, hidden


def _spanning_tree_facts(rng, units, hidden):
    """One consistent fact per unit after the first, each linking it to an already-placed unit, so
    together they connect every unit into a single component -- with the true ratio computed straight
    from the hidden values, never from anything convert() or Converter computes."""
    order = units[:]
    rng.shuffle(order)
    facts = []
    for i in range(1, len(order)):
        a = order[i]
        b = order[rng.randrange(i)]
        facts.append((a, b, hidden[a] / hidden[b]))
    return facts


def _dsu_find(parent, x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _dsu_union(parent, x, y):
    rx, ry = _dsu_find(parent, x), _dsu_find(parent, y)
    if rx != ry:
        parent[rx] = ry


# --- Part 1: independent brute force, a Floyd-Warshall-style closure over products ---
def _brute_all_factors(facts):
    units = sorted({u for a, b, _ in facts for u in (a, b)})   # NOTE: sorted -- order must not depend on hashing
    dist = {u: {v: None for v in units} for u in units}
    for u in units:
        dist[u][u] = 1.0
    for a, b, r in facts:
        dist[a][b] = r
        dist[b][a] = 1.0 / r
    for k in units:
        for i in units:
            if dist[i][k] is None:
                continue
            for j in units:
                if dist[k][j] is not None and dist[i][j] is None:
                    dist[i][j] = dist[i][k] * dist[k][j]
    return dist


for seed in range(200):
    rng = random.Random(seed)
    units, hidden = _random_units_and_hidden(rng, rng.randint(2, 9))
    facts_r = _spanning_tree_facts(rng, units, hidden)
    for _ in range(rng.randint(0, 4)):                # a few redundant but consistent extra edges
        a, b = rng.sample(units, 2)
        facts_r.append((a, b, hidden[a] / hidden[b]))
    rng.shuffle(facts_r)
    dist = _brute_all_factors(facts_r)
    for _ in range(6):
        src = rng.choice(units + ["ghost1", "ghost2"])   # sometimes an unmentioned unit
        dst = rng.choice(units + ["ghost1", "ghost2"])
        value = rng.uniform(-5.0, 5.0)
        got = convert(facts_r, value, src, dst)
        expected = dist.get(src, {}).get(dst)
        if expected is None:
            assert got is None, (seed, src, dst, got)
        else:
            assert got is not None and math.isclose(got, value * expected, rel_tol=1e-9), \
                (seed, src, dst, got, value * expected)

for seed in range(30):   # two separately-consistent components that never connect to each other
    rng = random.Random(1000 + seed)
    units_a, hidden_a = _random_units_and_hidden(rng, rng.randint(2, 4))
    units_b, hidden_b = _random_units_and_hidden(rng, rng.randint(2, 4))
    facts_a = _spanning_tree_facts(rng, units_a, hidden_a)
    facts_b = [(a + "_b", b + "_b", r) for a, b, r in _spanning_tree_facts(rng, units_b, hidden_b)]
    combined = facts_a + facts_b
    rng.shuffle(combined)
    assert convert(combined, 1.0, rng.choice(units_a), rng.choice(units_b) + "_b") is None

# --- Part 2: one perturbed fact inserted where it is provably checkable, and pure-consistent sets ---
for seed in range(200):
    rng = random.Random(2000 + seed)
    units, hidden = _random_units_and_hidden(rng, rng.randint(3, 9))
    tree_facts = _spanning_tree_facts(rng, units, hidden)

    i = rng.randint(1, len(tree_facts))          # a prefix of length >= 1 already connects >= 2 units
    parent: dict[str, str] = {}
    for a, b, _ in tree_facts[:i]:
        _dsu_union(parent, a, b)
    by_root = defaultdict(list)
    for u in units:
        if u in parent:
            by_root[_dsu_find(parent, u)].append(u)
    component = max(by_root.values(), key=len)
    a, b = rng.sample(component, 2)
    true_r = hidden[a] / hidden[b]
    wrong_r = true_r * rng.choice([2.0, 0.5, 3.0, 5.0, 10.0])
    perturbed = tree_facts[:i] + [(a, b, wrong_r)] + tree_facts[i:]
    assert first_contradiction(perturbed) == i, (seed, i, a, b, true_r, wrong_r)

    extra = [(x, y, hidden[x] / hidden[y]) for x, y in (rng.sample(units, 2) for _ in range(rng.randint(0, 3)))]
    consistent = tree_facts + extra
    rng.shuffle(consistent)
    assert first_contradiction(consistent) == -1, seed

# --- Part 3: random interleavings, cross-checked against Parts 1-2 and against the hidden ground truth ---
for seed in range(150):
    rng = random.Random(3000 + seed)
    units, hidden = _random_units_and_hidden(rng, rng.randint(3, 8))

    conv = Converter()
    accepted: list[tuple[str, str, float]] = []      # facts actually accepted, in order
    dsu_parent: dict[str, str] = {}                   # independent connectivity oracle over accepted facts

    for _ in range(40):
        if rng.random() < 0.6:
            a, b = rng.sample(units, 2)
            connected_now = (a in dsu_parent and b in dsu_parent
                              and _dsu_find(dsu_parent, a) == _dsu_find(dsu_parent, b))
            if connected_now and rng.random() < 0.5:
                r = (hidden[a] / hidden[b]) * rng.choice([2.0, 0.5, 4.0])   # a deliberate contradiction
            else:
                r = hidden[a] / hidden[b]

            expected_accept = first_contradiction(accepted + [(a, b, r)]) == -1
            got = conv.add_fact(a, b, r)
            assert got == expected_accept, (seed, a, b, r, got, expected_accept)
            if got:
                accepted.append((a, b, r))
                _dsu_union(dsu_parent, a, b)
        else:
            src, dst = rng.choice(units), rng.choice(units)
            connected = (src in dsu_parent and dst in dsu_parent
                         and _dsu_find(dsu_parent, src) == _dsu_find(dsu_parent, dst))
            got = conv.query(src, dst)
            cross = convert(accepted, 1.0, src, dst)
            if connected:
                expected = hidden[src] / hidden[dst]
                assert got is not None and math.isclose(got, expected, rel_tol=1e-9)
                assert cross is not None and math.isclose(cross, got, rel_tol=1e-9)
            else:
                assert got is None and cross is None

    assert first_contradiction(accepted) == -1    # everything Converter accepted is, as a whole, consistent

print("all checks passed")
```

</details>

</details>
