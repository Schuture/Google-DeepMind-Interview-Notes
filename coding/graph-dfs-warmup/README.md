# Warm-Up: Scanning an Array and Traversing a Graph Depth-First

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · arrays and graph traversal | ★★★★☆ | Easy | SWE · MLE · Intern | arrays, binary-search, graph-construction, dfs, iterative-dfs, connected-components, cycle-detection | 3 parts / 30–45 min | Hiring-manager screen · Skills interview |
<!-- meta:end -->

## Problem

### Part 1 — Elements above a threshold

```py
def above_threshold(arr: list[int], threshold: int) -> tuple[int, list[int]]: ...
```

Return the number of elements of `arr` that are strictly greater than `threshold`, together with their
indices in `arr` in increasing order. An element equal to `threshold` does not count. `arr` may be empty, in
which case the result is `(0, [])`.

Example: in `above_threshold([1, 5, 7, 2, 8, 3], 4)`, the elements `1`, `2` and `3` sit at or below the
threshold; `5`, `7` and `8` are above it, at indices `1`, `2` and `4`:

```text
above_threshold([1, 5, 7, 2, 8, 3], 4) == (3, [1, 2, 4])
```

Then a follow-up on the same array: many thresholds, asked one after another.

```py
def count_above_many(arr: list[int], thresholds: list[int]) -> list[int]: ...
```

For each threshold in `thresholds`, return only the count of elements of `arr` strictly greater than it
(the same rule as `above_threshold`), as a list in the same order as `thresholds`. Writing $n$ for
`len(arr)` and $q$ for `len(thresholds)`, this must run in $O((n + q) \log n)$ total, never $O(nq)$.

### Part 2 — Build the graph from edges and return the DFS order

```py
def build_graph(edges: list[tuple[str, str]]) -> dict[str, list[str]]: ...
def dfs_order(edges: list[tuple[str, str]], start: str) -> list[str]: ...
```

`edges` is a list of pairs of node names; together they define an undirected graph. An edge `(u, v)` with
`u != v` connects `u` and `v` both ways. An edge that repeats one already given — in the same order or
reversed — adds nothing new. An edge `(u, u)` (a *self-loop*) adds `u` as a node of the graph, with no
neighbour coming from that edge. `build_graph` returns, for every node that appears in at least one edge,
its neighbours as a list sorted in increasing order (node names compare as ordinary strings).

Depth-first search (DFS) from a node explores as far as possible along one neighbour before backtracking to
try the next. `dfs_order` starts at `start`, explores a node's unvisited neighbours in increasing order —
this is what makes the visiting order unique — and returns every node reachable from `start` exactly once,
in the order it is first visited (its *preorder*). If `start` is not a node of the graph at all (it appears
in no edge), the result is `[start]`.

```text
edges = [("A","B"), ("A","C"), ("B","D"), ("B","E"), ("C","F"), ("E","F")]

build_graph(edges) == {
    "A": ["B", "C"], "B": ["A", "D", "E"], "C": ["A", "F"],
    "D": ["B"],      "E": ["B", "F"],      "F": ["C", "E"],
}
```

`dfs_order(edges, "A")` always steps to the smallest unvisited neighbour, and backtracks only once none
is left:

```text
A -> B -> D              # D's only neighbour, B, is visited -- backtrack to B
       -> E -> F -> C    # C's neighbours, A and F, are both visited -- backtrack all the way to A, done

dfs_order(edges, "A") == ["A", "B", "D", "E", "F", "C"]
```

Implement `dfs_order` two ways: first a straightforward recursive version, then an iterative one that
returns exactly the same order but never fails on a path of 100,000 nodes, where the recursive version
does.

### Part 3 — Components and cycles

```py
def components(edges: list[tuple[str, str]]) -> list[list[str]]: ...
def has_cycle(edges: list[tuple[str, str]]) -> bool: ...
```

Two nodes of the graph built by `build_graph` are in the same *connected component* when some path of
edges joins them; every node is in exactly one component, including a node with no edges of its own beyond
a self-loop. `components` returns every component as its own DFS order (the rule of Part 2, started from
the component's smallest node), and returns the components
themselves ordered by that same smallest node.

```text
edges = [("B", "A"), ("A", "C"), ("E", "D"), ("G", "G")]

components(edges) == [["A", "B", "C"], ["D", "E"], ["G"]]
# {A, B, C} connect through A; {D, E} connect through the D-E edge; G's only edge is a self-loop, so
# G is a node with no neighbour at all -- its own component of size one
```

A *cycle* is a sequence of three or more distinct nodes $v_0, v_1, \ldots, v_{k-1}$ with each consecutive
pair, and also $v_{k-1}$ and $v_0$, joined by an edge. `has_cycle` returns whether the graph — after the
same normalisation as Part 2: a repeated edge merged away, a self-loop contributing no neighbour — contains
one anywhere.

```text
edges = [("A","B"), ("A","C"), ("B","D"), ("B","E"), ("C","F"), ("E","F")]   # Part 2's graph again

has_cycle(edges) == True          # A -> B -> E -> F -> C -> A is a cycle; D hangs off B, outside it
has_cycle(edges[:-1]) == False    # dropping the last edge, ("E","F"), leaves a tree: no cycle
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Worth confirming before coding: for Part 1, that the comparison is strict and the indices come back
ascending. For Part 2, that the graph is undirected, that neighbours are visited in increasing order —
what makes the DFS order unique — and that a `start` with no edges still returns `[start]`. Python's
default recursion limit (1,000 frames) matters once `dfs_order` runs on a long chain.

### Part 1

One pass with `enumerate` collects the qualifying indices directly, so nothing is computed twice.

```python
def above_threshold(arr: list[int], threshold: int) -> tuple[int, list[int]]:
    indices = [i for i, x in enumerate(arr) if x > threshold]
    return len(indices), indices
```

This is $O(n)$ for a single threshold: one pass over `arr`, one comparison per element.

Asking the same question for many thresholds at once is where sorting pays off. `bisect_right(sorted_arr,
t)` returns the number of elements of `sorted_arr` that are `<= t` (its insertion point keeps every element
equal to `t` to its left), so subtracting that count from $n$ leaves exactly the elements strictly greater
than `t`.

```python
from bisect import bisect_right


def count_above_many(arr: list[int], thresholds: list[int]) -> list[int]:
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    # NOTE: bisect_right, not bisect_left -- bisect_left's insertion point sits before every element
    # equal to t, so n - bisect_left(sorted_arr, t) would also count those equal elements as "above"
    return [n - bisect_right(sorted_arr, t) for t in thresholds]
```

Sorting costs $O(n \log n)$ once; each of the $q$ thresholds costs $O(\log n)$ for its own binary search,
for $O((n + q) \log n)$ total — against $O(nq)$ for calling `above_threshold` once per threshold.

### Part 2

`build_graph` keeps neighbours in a `set` while reading `edges`, so a repeated edge — either orientation —
lands on a set that already has it, and a self-loop `(u, u)` registers `u` (via `setdefault`) without
adding `u` to its own neighbours; sorting each set once at the end is cheaper and simpler than maintaining
a sorted structure throughout.

```python
def build_graph(edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    neighbours: dict[str, set[str]] = {}
    for u, v in edges:
        neighbours.setdefault(u, set())
        neighbours.setdefault(v, set())
        if u != v:
            neighbours[u].add(v)
            neighbours[v].add(u)
        # NOTE: u == v (a self-loop) falls through to here -- u is registered as a node above, but
        # gets no neighbour; a repeated edge adds nothing new, since neighbours are kept in a set
    return {node: sorted(adj) for node, adj in neighbours.items()}
```

A direct recursive DFS matches the definition of preorder most closely: visit a node, then recurse into
each of its unvisited neighbours in increasing order.

```python
def dfs_order_recursive(edges: list[tuple[str, str]], start: str) -> list[str]:
    graph = build_graph(edges)
    visited = {start}
    order: list[str] = []

    def visit(node: str) -> None:
        order.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                visit(neighbour)   # NOTE: one Python call frame stays open per node on this branch

    visit(start)
    return order
```

Each call frame of `visit` stays open until every node below it in the DFS tree is fully explored, so a
long chain of nodes needs just as long a chain of open frames; past Python's default limit of 1,000,
`dfs_order_recursive` raises `RecursionError` (checked below) — the fix is to stop using the call stack,
not to raise the limit.

An explicit stack — an ordinary Python list — replaces the call stack, so its size is bounded only by
memory. Pushing a node's neighbours in *reverse* sorted order, and marking a node visited only when it is
popped, reproduces the recursive order exactly: the smallest neighbour, pushed last, sits on top and is
explored first.

```python
def dfs_order(edges: list[tuple[str, str]], start: str) -> list[str]:
    graph = build_graph(edges)
    visited: set[str] = set()
    order: list[str] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue           # NOTE: a node is pushed once by each neighbour expanded before it is
                               # visited; every pop of it after the first is skipped here
        visited.add(node)      # NOTE: mark at pop time, not push time. Marking at push time turns edges
                               # A-B, A-C, B-D, B-E, C-D from A into A, B, D, E, C (C, marked when A
                               # pushed it, is skipped from D), not the DFS preorder A, B, D, C, E.
        order.append(node)
        for neighbour in reversed(graph.get(node, [])):
            if neighbour not in visited:
                stack.append(neighbour)
    return order
```

Building the graph sorts every node's neighbour list once, $O(E \log E)$ for $E$ = `len(edges)`; after that
each of the $V$ reachable nodes is visited once and has its neighbour list scanned once, so each edge is
examined at most twice (once from each end), each examination pushes at most one entry, and each pop
either visits a node or is skipped: $O(V + E)$ in all — the same bound and order whether `dfs_order` or
`dfs_order_recursive` computes it.

### Part 3

`components` visits nodes in increasing order and starts a fresh DFS (Part 2's iterative rule) from the
first one no earlier component reached; since nodes are tried smallest first, that start is always the
smallest node of an unreached component, so the components already come out of the loop in the required
order.

```python
def components(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph = build_graph(edges)
    visited: set[str] = set()
    result: list[list[str]] = []
    for node in sorted(graph):
        if node in visited:
            continue            # NOTE: sorted() -- the first unvisited node here is always the
                                # smallest node of a component nothing has reached yet
        order: list[str] = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            order.append(cur)
            for neighbour in reversed(graph.get(cur, [])):
                if neighbour not in visited:
                    stack.append(neighbour)
        result.append(order)
    return result
```

Same $O(V + E)$ traversal work as `dfs_order`, plus $O(V \log V)$ to sort the nodes: every node is still
visited, and its neighbour list scanned, exactly once overall, just split across however many components
there are.

Cycle detection carries each node's *parent* (the node that discovered it) alongside it on the stack, and
marks a neighbour visited the moment it is discovered, at push time rather than pop time, unlike
`dfs_order` above. Every node other than a search's start is then discovered along exactly one edge, and
these discovery edges form a spanning tree of its component. When a node is popped and its neighbours
scanned, its only tree edges lead to its parent and to the neighbours it discovers during this very scan;
so a neighbour that is already visited and is not its parent is joined to it by an edge outside the tree,
which together with the tree path between the two closes a cycle. Conversely, if no scan ever finds such
a neighbour, every edge is a tree edge, so every component is a tree and there is no cycle.

```python
def has_cycle(edges: list[tuple[str, str]]) -> bool:
    graph = build_graph(edges)
    visited: set[str] = set()
    for node in graph:
        if node in visited:
            continue
        visited.add(node)
        stack = [(node, None)]              # (current node, the node we arrived from)
        while stack:
            cur, parent = stack.pop()
            for neighbour in graph[cur]:
                if neighbour == parent:
                    continue                # NOTE: skip only the edge back to where we came from --
                                            # build_graph already merged any repeat of it away
                if neighbour in visited:
                    return True             # a visited neighbour that is not the parent closes a cycle
                visited.add(neighbour)      # NOTE: marked at discovery (push) time -- see above
                stack.append((neighbour, cur))
    return False
```

A counting test answers the same question from the numbers of nodes, edges and components alone. Write $V$ for
the nodes, $E$ for the (normalised, undirected) edges and $C$ for the components. A tree on $v$ nodes has
exactly $v - 1$ edges — true for $v = 1$, and preserved each time a tree grows by one node and one edge — so a
*forest* (every component a tree) with $C$ components over $V$ nodes has exactly $\sum_i (v_i - 1) = V - C$
edges. Any component needs at least $v_i - 1$ edges to stay connected, so $E \ge V - C$ always, with equality
exactly when every component is a tree; a cycle in some component pushes its edge count, and hence the total,
strictly above that (removing one edge of the cycle leaves the component connected, so at least $v_i - 1$
other edges remain). So the graph has a cycle exactly when $E > V - C$ — the fact the checks below use as an
independent test of `has_cycle`.

### Follow-ups

- **Directed graphs.** Treating each `(u, v)` as one-way turns `build_graph` into a directed adjacency list
  (`neighbours[u].add(v)` alone), and cycle detection changes with it: the parent-tracking rule breaks,
  because an edge to an already-visited node need not close a directed cycle (with A→B, B→C and A→C, C is
  reached twice and there is no cycle), so a directed cycle is instead found by
  three-colouring nodes white/grey/black during DFS and reporting one the moment an edge reaches a grey
  node. The same DFS gives a topological order by prepending each node the instant it turns black.
- **Shortest paths, unweighted.** `dfs_order` visits every reachable node, but not by increasing distance
  from `start`; when the question is fewest edges to some node rather than mere reachability,
  breadth-first search — a queue instead of a stack, so every node at distance $d$ is dequeued before any
  at distance $d + 1$ — is the right tool, and needs no parent-tracking trick to avoid cycles, since a node
  already queued or dequeued is never enqueued again.
- **Adjacency matrix.** A $V \times V$ boolean array (after numbering the nodes $0$ to $V - 1$) with
  `matrix[i][j]` true exactly when nodes $i$ and $j$ are joined trades `build_graph`'s $O(V + E)$ space for
  $O(V^2)$, worthwhile once the graph is dense enough that $E$ approaches $V^2$; it also answers "are $u$ and
  $v$ adjacent" in $O(1)$ instead of scanning a neighbour list, but the adjacency list above stays better for
  sparse graphs, with few edges per node.
- **Recovering the DFS tree.** Pushing `(neighbour, node)` pairs instead of bare nodes, and recording the
  pair when a node is popped and visited for the first time, gives the edge that actually discovered each
  node, and with it the tree DFS walked, not merely the visiting order. For a node pushed more than once,
  the push that counts is the one popped first (in the NOTE's example, C comes from D, not A); and in an
  undirected graph every edge outside this tree joins a node to one of its ancestors, so a component is
  acyclic exactly when its tree uses all of its edges.

<details>
<summary>Checks (runnable)</summary>

```python
import random
from collections import defaultdict

# --- Part 1: the worked example, then a brute force on random arrays ---
assert above_threshold([1, 5, 7, 2, 8, 3], 4) == (3, [1, 2, 4])
assert above_threshold([], 0) == (0, [])

for seed in range(300):
    rng = random.Random(seed)
    arr = [rng.randint(-10, 10) for _ in range(rng.randint(0, 12))]
    threshold = rng.choice(arr) if arr and rng.random() < 0.5 else rng.randint(-10, 10)
    expected_indices = [i for i, x in enumerate(arr) if x > threshold]   # independent, no bisect
    assert above_threshold(arr, threshold) == (len(expected_indices), expected_indices), (seed, arr, threshold)

    thresholds = [rng.randint(-10, 10) for _ in range(5)]
    expected_counts = [sum(1 for x in arr if x > t) for t in thresholds]  # direct counting, no bisect
    assert count_above_many(arr, thresholds) == expected_counts, (seed, arr, thresholds)

# --- Part 2: the worked examples ---
edges_ex = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("E", "F")]
assert build_graph(edges_ex) == {
    "A": ["B", "C"], "B": ["A", "D", "E"], "C": ["A", "F"],
    "D": ["B"], "E": ["B", "F"], "F": ["C", "E"],
}
assert dfs_order(edges_ex, "A") == ["A", "B", "D", "E", "F", "C"]
assert dfs_order_recursive(edges_ex, "A") == ["A", "B", "D", "E", "F", "C"]
assert dfs_order([("X", "Y")], "Z") == ["Z"]              # start appears in no edge at all
assert dfs_order_recursive([("X", "Y")], "Z") == ["Z"]


def _reference_graph(edges):
    """Independent adjacency construction, restating the Problem section's rule directly: neighbours
    as sorted lists, a self-loop registers its node with no neighbour, a repeated edge collapses."""
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set())
        adj.setdefault(v, set())
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return {node: sorted(ns) for node, ns in adj.items()}


def _reference_dfs_preorder(edges, start):
    """Independent restatement of the DFS preorder rule: recursive, over its own adjacency, never
    calling build_graph/dfs_order/dfs_order_recursive."""
    adj = _reference_graph(edges)
    visited = {start}
    order = []

    def go(node):
        order.append(node)
        for nxt in adj.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                go(nxt)

    go(start)
    return order


def _dfs_order_mark_on_push(edges, start):
    """A common but WRONG iterative variant: marks a node visited at push time rather than pop time.
    Kept only to contrast with dfs_order, never used as ground truth."""
    graph = build_graph(edges)
    visited = {start}
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbour in reversed(graph.get(node, [])):
            if neighbour not in visited:
                visited.add(neighbour)   # BUG: marks at push time, the mistake the NOTE in dfs_order names
                stack.append(neighbour)
    return order


# the mark-on-push variant happens to agree with dfs_order on the worked example above ...
assert _dfs_order_mark_on_push(edges_ex, "A") == ["A", "B", "D", "E", "F", "C"]

# ... but disagrees on the five-edge graph from the NOTE in dfs_order, with exactly the orders it states
edges_diff = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "D")]
assert dfs_order(edges_diff, "A") == ["A", "B", "D", "C", "E"]
assert dfs_order_recursive(edges_diff, "A") == ["A", "B", "D", "C", "E"]
assert _dfs_order_mark_on_push(edges_diff, "A") == ["A", "B", "D", "E", "C"]
assert dfs_order(edges_diff, "A") != _dfs_order_mark_on_push(edges_diff, "A")


# --- Part 2: recursive and iterative agree with the independent preorder on many random graphs,
# whose edge lists include duplicates, reversed duplicates and self-loops ---
def _random_edges(rng, n_nodes=8, n_edges=14):
    nodes = [chr(ord("A") + i) for i in range(n_nodes)]
    edges = []
    for _ in range(n_edges):
        u, v = rng.choice(nodes), rng.choice(nodes)
        edges.append((u, v))
        if rng.random() < 0.3:
            edges.append((v, u))     # a reversed duplicate of the same edge
        if rng.random() < 0.15:
            edges.append((u, v))     # an exact duplicate
    return nodes, edges


saw_self_loop = False
for seed in range(300):
    rng = random.Random(seed + 1_000)
    nodes, edges = _random_edges(rng)
    start = rng.choice(nodes)
    expected = _reference_dfs_preorder(edges, start)
    assert dfs_order(edges, start) == expected, (seed, edges, start)
    assert dfs_order_recursive(edges, start) == expected, (seed, edges, start)
    saw_self_loop = saw_self_loop or any(u == v for u, v in edges)
assert saw_self_loop   # the self-loop rule was actually exercised by the fuzzing, not just stated

# --- Part 2: a 100,000-node path -- iterative succeeds, recursive hits Python's call-stack limit ---
long_edges = [(str(i), str(i + 1)) for i in range(100_000 - 1)]
long_result = dfs_order(long_edges, "0")
assert len(long_result) == 100_000 and long_result == [str(i) for i in range(100_000)]
try:
    dfs_order_recursive(long_edges, "0")
except RecursionError:
    pass
else:
    raise AssertionError("dfs_order_recursive should raise RecursionError on a 100,000-node path")

# --- Part 3: the worked examples ---
edges_components = [("B", "A"), ("A", "C"), ("E", "D"), ("G", "G")]
assert components(edges_components) == [["A", "B", "C"], ["D", "E"], ["G"]]
assert has_cycle(edges_ex) is True
assert has_cycle(edges_ex[:-1]) is False


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


def _independent_components(edges):
    """Union-find over the raw edge list, independent of build_graph/components."""
    parent: dict[str, str] = {}
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
        _dsu_find(parent, u)
        _dsu_find(parent, v)
        if u != v:
            _dsu_union(parent, u, v)
    groups = defaultdict(list)
    for node in nodes:
        groups[_dsu_find(parent, node)].append(node)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def _independent_has_cycle(edges):
    """E > V - C, computed straight from the definitions, independent of build_graph/has_cycle."""
    adj = defaultdict(set)
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    edge_count = sum(len(ns) for ns in adj.values()) // 2   # each undirected edge counted from both ends
    return edge_count > len(nodes) - len(_independent_components(edges))


def _random_tree_edges(rng, n_nodes):
    """A tree by construction: node i (i >= 1) attached to a uniformly random earlier node."""
    nodes = [chr(ord("A") + i) for i in range(n_nodes)]
    return [(nodes[i], nodes[rng.randrange(i)]) for i in range(1, n_nodes)]


# --- components: grouping and per-component order, against the union-find and the preorder rule ---
for seed in range(200):
    rng = random.Random(seed + 2_000)
    _nodes, edges = _random_edges(rng, n_nodes=10, n_edges=16)
    got = components(edges)
    assert [c[0] for c in got] == sorted(c[0] for c in got)             # components ordered by their start
    assert sorted(sorted(c) for c in got) == _independent_components(edges)   # same grouping
    for group in got:
        assert group[0] == min(group)                                   # each starts at its smallest node
        assert group == _reference_dfs_preorder(edges, group[0])         # and is that node's DFS preorder

# --- has_cycle: against E > V - C, on random graphs and on random trees (always acyclic) ---
for seed in range(200):
    rng = random.Random(seed + 3_000)
    _nodes, edges = _random_edges(rng, n_nodes=7, n_edges=11)
    assert has_cycle(edges) == _independent_has_cycle(edges), (seed, edges)

for seed in range(100):
    rng = random.Random(seed + 4_000)
    tree_edges = _random_tree_edges(rng, rng.randint(1, 9))
    assert has_cycle(tree_edges) is False
    assert _independent_has_cycle(tree_edges) is False

# --- has_cycle: a tree plus one genuinely new edge always closes exactly one cycle ---
saw_cycle = False
for seed in range(100):
    rng = random.Random(seed + 5_000)
    n = rng.randint(3, 9)
    tree_edges = _random_tree_edges(rng, n)
    tree_pairs = {frozenset(e) for e in tree_edges}
    nodes = [chr(ord("A") + i) for i in range(n)]
    while True:
        extra_u, extra_v = rng.sample(nodes, 2)
        if frozenset((extra_u, extra_v)) not in tree_pairs:
            break
    cyclic_edges = tree_edges + [(extra_u, extra_v)]
    assert has_cycle(cyclic_edges) is True
    assert _independent_has_cycle(cyclic_edges) is True
    saw_cycle = True
assert saw_cycle

print("all checks passed")
```

</details>

</details>
