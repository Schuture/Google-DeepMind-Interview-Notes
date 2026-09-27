# Winning Positions in a Token-Moving Game

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · game search on a graph, BFS/DFS | ★★★★☆ | Hard | SWE · RE · MLE · Intern | game-theory, dfs, retrograde-bfs, topological-order, sprague-grundy, graphs | 3 parts / 45–60 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

A directed graph has nodes $0, 1, \ldots, n-1$ and a list of distinct edges `edges: list[tuple[int, int]]`,
where each edge $(u, v)$ with $u \neq v$ is one directed connection from $u$ to $v$. A *token* sits on
exactly one node at a time. Two players alternate turns; a *move* takes the token from its current node $u$
along one edge $(u, v)$ to $v$ — the player about to move picks which outgoing edge of $u$ to take, when
more than one exists. A player who cannot move — the token sits on a node with no outgoing edge — loses
immediately, and the game ends there.

A *position* is identified by the token's current node (the two players are otherwise interchangeable, so
nothing else distinguishes a position). The *outcome* of a position is `"W"` if the player about to move
there can force a win no matter how the opponent replies, `"L"` if the opponent can force a win no matter
how the player about to move replies, and, once the graph may contain cycles (Part 2 only), `"D"` if
neither player can force a win, so that with both playing to avoid ever losing, play continues forever.

### Part 1 — Acyclic graphs

```py
def outcomes_dag(n: int, edges: list[tuple[int, int]]) -> list[str]: ...
def winning_move(n: int, edges: list[tuple[int, int]], v: int) -> int | None: ...
```

The graph is guaranteed acyclic. `outcomes_dag` returns a list of length `n` whose entry `i` is the
outcome of the position with the token on node `i`, `"W"` or `"L"`. `n` may be as large as $10^5$, so a
solution whose call-stack depth grows with the length of a path through the graph is not acceptable:
computing every outcome must use an explicit order over the nodes rather than recursion that follows one
edge at a time.

`winning_move(n, edges, v)` returns the smallest-numbered node `w` such that the edge `(v, w)` exists and
moving there is a winning move — the outcome of `w` is `"L"` — or `None` if node `v`'s own outcome is
`"L"`, meaning no such move exists: either `v` has no outgoing edge at all, or every one of its successors
has outcome `"W"`.

```text
0 -> 1, 2, 3
1 -> 2, 5
2 -> 3
3 -> 4, 5
4 -> (no outgoing edge)
5 -> 6
6 -> (no outgoing edge)
```

Tracing from the sinks (the nodes with no outgoing edge) upward: nodes 4 and 6 have no outgoing edge, so
the mover there loses at once — both are `"L"`. Node 5's only edge goes to 6 (`"L"`), so the mover at 5
can move there and hand the opponent an immediate loss: `"W"`. Node 3 has edges to 4 (`"L"`) and 5
(`"W"`); the edge to 4 alone already makes node 3 `"W"`. Node 2's only edge goes to 3, whose outcome is
`"W"` — node 2 has no edge to an `"L"` node, so every move it has hands the opponent a win, and node 2 is
`"L"` itself. Node 1 has edges to 2 (`"L"`) and 5 (`"W"`); the edge to 2 makes node 1 `"W"`. Node 0 has
edges to 1 (`"W"`), 2 (`"L"`) and 3 (`"W"`); the edge to 2 makes node 0 `"W"` as well:

```text
outcomes_dag(7, edges) == ["W", "W", "L", "W", "L", "W", "L"]   # nodes 0..6, in order
```

`winning_move(7, edges, 0)` returns `2`: node 0's edges lead to 1, 2 and 3 in that order, and 2 is the
smallest-numbered one whose outcome is `"L"`. `winning_move(7, edges, 2)` returns `None`, since node 2's
own outcome is already `"L"`; so does `winning_move(7, edges, 6)`, for the same reason.

### Part 2 — Graphs with cycles

```py
def outcomes(n: int, edges: list[tuple[int, int]]) -> list[str]: ...
```

The graph may now contain cycles (still no self-loops, since every edge still has $u \neq v$). Return a
list of length `n` whose entry `i` is `"W"`, `"L"` or `"D"`, the outcome, as defined above, of the
position with the token on node `i`.

```text
0 -> (no outgoing edge)
1 -> 0
2 -> 3
3 -> 2
4 -> 2, 0
5 -> 2, 3
6 -> 1
```

Node 0 has no outgoing edge, so it is `"L"`. Node 1's only edge goes to 0 (`"L"`), so node 1 is `"W"`.
Nodes 2 and 3 have edges only to each other, `2 -> 3` and `3 -> 2`, and neither can ever reach node 0 or
any other decided node; with both players unable to do anything but keep moving between the two of them,
play continues forever, so both are `"D"`. Node 4 has edges to 2 (`"D"`) and 0 (`"L"`); the edge to 0
alone makes node 4 `"W"`, regardless of what the edge to 2 would lead to. Node 5 has edges only to 2 and 3,
both `"D"` — no edge to an `"L"` node, and no way to ever run out of non-`"W"` replies either, since 2 and
3 are never `"W"` — so node 5 is `"D"` as well. Node 6's only edge goes to 1 (`"W"`); since every edge
node 6 has (there is only the one) leads to a `"W"` node, node 6 is `"L"`, whichever way its mover plays.

```text
outcomes(7, edges) == ["L", "W", "D", "D", "W", "D", "L"]   # nodes 0..6, in order
```

### Part 3 — Several tokens

```py
def first_player_wins(n: int, edges: list[tuple[int, int]], tokens: list[int]) -> bool: ...
```

The graph is acyclic again, as in Part 1. `k` tokens sit on nodes of the graph, given as `tokens`, a list
of length `k` of node numbers; several tokens may share the same node. A move now picks any one token and
moves it along one outgoing edge of the node it currently occupies, exactly as before; the other `k - 1`
tokens stay where they are. A player who cannot move any token at all — every token occupies a node with
no outgoing edge — loses. Return `True` if the first player to move can force a win, `False` otherwise.

```text
0 -> (no outgoing edge)
1 -> 0
2 -> 0
```

A lone token on node 1 is a win for whoever moves first: the only move is to 0, which leaves the opponent
facing a token on a node with no outgoing edge, so the opponent cannot move and loses:
`first_player_wins(3, edges, [1]) == True`. By the same reasoning a lone token on node 2 is also a
first-player win: `first_player_wins(3, edges, [2]) == True`. With both tokens in play together,
`tokens = [1, 2]`, the first player must move one of them — say the one on node 1, to node 0 — and then
the second player moves the other one, the one still on node 2, to node 0, leaving the first player facing
two tokens, both on node 0, with no move available: the first player loses. Moving the other token first
is symmetric and loses the same way, so every move open to the first player loses:
`first_player_wins(3, edges, [1, 2]) == False`.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points are worth confirming before coding: what happens at a node with no outgoing edge — the mover
there loses immediately, by the statement's own rule, so it is always `"L"` — and whether the graph can
contain cycles, which it cannot in Parts 1 and 3 (both promise an acyclic graph) but can in Part 2.

### Part 1

`outcome(v)` is `"W"` exactly when some edge out of `v` leads to a node whose outcome is `"L"`, and `"L"`
exactly when every edge out of `v` (there may be none) leads to a node whose outcome is `"W"` — this is the
statement's own definition of *forcing a win*, unwound one move at a time: to force a win the mover needs
at least one reply that leaves the opponent facing exactly this same kind of loss, and to be forced into a
loss every reply must leave the opponent able to force a win in turn. Because the graph is acyclic, this
recursive definition has nothing circular to resolve: the outcome of `v` only ever depends on the outcome
of nodes reachable from `v`, and following an edge from any node reaches a strictly later node in a
topological order, so one pass over the nodes in *reverse* topological order — sinks first, `v` only once
every node reachable from it is already done — computes every outcome directly, touching each node exactly
once.

`_topo_order` produces a topological order with Kahn's algorithm: repeatedly take a node all of whose
incoming edges have already been removed, and remove its own outgoing edges in turn. On an acyclic graph
this empties the whole node set — the assertion inside `_topo_order` would fire on a graph with a cycle,
since some node's in-degree would then never reach zero — and the algorithm is iterative, a queue rather
than a call stack, so its depth is unrelated to the length of any path in the graph, which is exactly what
the $n$ up to $10^5$ requirement needs.

```python
from collections import deque


def _topo_order(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Kahn's algorithm: returns the nodes of an acyclic graph in a topological order (every edge u -> v
    has u appear before v). Iterative, so the call stack never grows with the size of the graph."""
    adj: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(u for u in range(n) if indeg[u] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    assert len(order) == n     # NOTE: fewer than n nodes emitted would mean a cycle; ruled out by the statement
    return order


def outcomes_dag(n: int, edges: list[tuple[int, int]]) -> list[str]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    outcome = [""] * n
    # NOTE: reverse topological order -- sinks first -- so every successor of v is already filled in by
    # the time v itself is evaluated; an explicit order rather than recursion keeps the call stack at O(1)
    for v in reversed(_topo_order(n, edges)):
        outcome[v] = "W" if any(outcome[w] == "L" for w in adj[v]) else "L"
    return outcome


def winning_move(n: int, edges: list[tuple[int, int]], v: int) -> int | None:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, w in edges:
        adj[u].append(w)
    outcome = outcomes_dag(n, edges)    # NOTE: this is O(n + m) on its own -- call it once and reuse the
    for w in sorted(adj[v]):            # array when checking many nodes, instead of recomputing it per call
        if outcome[w] == "L":
            return w
    return None
```

With $n$ nodes and $m$ edges, `_topo_order` is $O(n + m)$: building the adjacency list and in-degree array
is $O(n + m)$, and Kahn's algorithm removes each node once and inspects each of its outgoing edges exactly
once. The reversed pass in `outcomes_dag` is another $O(n + m)$, one look at `adj[v]` per node.
`winning_move` adds an $O(\deg(v) \log \deg(v))$ sort on top of that same $O(n + m)$ call.

### Part 2

On a graph with cycles, Part 1's recursion can be circular — `outcome(v)` might depend on `outcome(w)`,
which depends on `outcome(v)` — so no topological order exists, and evaluating nodes in a fixed order no
longer works. *Retrograde analysis* solves this by working outward from the known base cases (the sinks)
instead of committing to any per-node order in advance: label a node the moment its outcome is forced by
labels already assigned, and propagate along edges reversed, exactly as reachability is computed backward
from a goal. Every sink is immediately `"L"`. Whenever a node `x` is labelled, look at every unlabelled
predecessor `p` of `x` (every edge `p -> x`): if `x` is `"L"`, `p` becomes `"W"` at once, since `p` has a
move — to `x` — that hands the opponent a loss; if `x` is `"W"`, `p` moves one step closer to being forced
into `"L"` — keep, for each unlabelled `p`, a count of how many of its successors are not yet known to be
`"W"`, and label `p` as `"L"` the moment that count reaches zero, since every one of `p`'s replies has then
been shown to hand the opponent a win. Whatever is never labelled once no more labels can be produced is
`"D"`.

Three claims need proving. *Every node labelled `"L"` is truly a loss:* by induction on the order labels
are assigned, `p` is labelled `"L"` only once every one of its successors is already labelled `"W"`, and by
the induction hypothesis each of those `"W"` labels is already correct, so every move from `p` (there is at
least one, or `p` was already a sink) hands the opponent a true win. *Every node labelled `"W"` is truly a
win:* `p` is labelled `"W"` at the moment some successor `x` of it is found already labelled `"L"`; by the
induction hypothesis (`x` was labelled at a strictly earlier step) that `"L"` is already correct, so `p`
has a genuine move to a true loss for the opponent. *Every node never labelled is a draw:* such a node `v`
can never have a successor labelled `"L"` — the instant any successor of `v` is labelled `"L"`, `v` itself
would be labelled `"W"` in that same step, since predecessors are scanned immediately once a node is
labelled — and `v` always has at least one successor that is also never labelled, since `v` is never
labelled `"L"` itself, which by the counting rule above means its count of not-yet-`"W"` successors never
reaches zero, so at least one successor of `v` is never labelled `"W"` either, and by the first point it
cannot be `"L"`, so it too is never labelled (`v` does have at least one successor to begin with: a node
with none is a sink, labelled at once). So both players, from any never-labelled node, can always move to
another never-labelled node: doing so never hands the opponent an immediate win, since no move goes to an
`"L"` node, and never runs out of options, since a mover with a successor always has one to play — so play
started at a never-labelled node can continue forever with neither side ever forcing a win, exactly `"D"`.

```python
def outcomes(n: int, edges: list[tuple[int, int]]) -> list[str]:
    adj: list[list[int]] = [[] for _ in range(n)]
    pred: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        pred[v].append(u)
    unresolved = [len(succs) for succs in adj]   # successors of v not yet known to be "W"
    label = [None] * n
    q = deque(v for v in range(n) if not adj[v])    # sinks: the mover there has no move at all
    for v in q:
        label[v] = "L"
    while q:
        x = q.popleft()
        for p in pred[x]:
            if label[p] is not None:
                continue        # NOTE: a label, once assigned, never changes -- skip an already-decided p
            if label[x] == "L":
                label[p] = "W"       # p can move to x and hand the opponent a loss
                q.append(p)
            else:
                unresolved[p] -= 1
                if unresolved[p] == 0:
                    label[p] = "L"   # every successor of p now leads to a win for whoever moves there
                    q.append(p)
    return [label[v] if label[v] is not None else "D" for v in range(n)]
```

Every edge `p -> x` is inspected exactly once over the whole run — the moment `x` is dequeued — so
`outcomes` is $O(n + m)$ time and space, the same bound as Part 1, despite allowing cycles. On an acyclic
graph, every node is eventually labelled (a DAG has no cycle to get stuck circling inside), so `outcomes`
never returns `"D"` there, and reduces to exactly the same partition into `"W"` and `"L"` as `outcomes_dag`
— checked below.

### Part 3

With one token, the game at a node is an *impartial game* under the normal-play convention already fixed
by the statement (a player unable to move loses) — both players have exactly the same moves available from
any given node, which is why *position* only ever needed a node, never a player label. With several
tokens, a move picks one of `k` independent single-token games and advances it; this is the standard *sum
of games*, and the Sprague–Grundy theorem gives it a single number rather than requiring a fresh W/L/D
analysis of the whole product state space. Assign every node `v` a *Grundy number*
$g(v) = \operatorname{mex}\{g(w) : w \text{ a successor of } v\}$, the *minimum excludant*: the smallest
non-negative integer that is not the Grundy number of any successor of `v` (a sink has no successors, so
$g(\text{sink}) = \operatorname{mex}(\emptyset) = 0$). The claim to derive: with tokens on nodes
$t_1, \ldots, t_k$, the first player can force a win if and only if
$g(t_1) \oplus \cdots \oplus g(t_k) \neq 0$, where $\oplus$ is bitwise XOR — Bouton's rule for Nim, with
$g(t_i)$ playing the role of the size of the $i$-th pile.

The proof rests on two facts that follow directly from the definition of mex: $g(v)$ itself is never the
Grundy number of any successor of `v` (that is what "excluded" means), and every non-negative integer
smaller than $g(v)$ *is* the Grundy number of some successor of `v` (otherwise that smaller value would
have been excluded instead, and would be the mex). By strong induction on the number of moves remaining —
finite, since the graph is acyclic and every move advances one token strictly forward in a topological
order — a position with XOR $X = 0$ is a loss for the mover, and a position with $X \neq 0$ is a win. If
$X = 0$: any move changes exactly one token's contribution from $g(v)$ to $g(w)$ for some successor $w$,
and since $g(w) \neq g(v)$ (the first fact), the new XOR is $X \oplus g(v) \oplus g(w) = g(v) \oplus g(w)
\neq 0$; by the induction hypothesis a nonzero XOR is a win for whoever moves next, so every move from an
$X = 0$ position hands the opponent a win, making it a loss. If $X \neq 0$: let $i$ be a token whose Grundy
number $g(t_i)$ has the highest set bit of $X$ set (some token must, or that bit of $X$ could not be set);
then $g(t_i) \oplus X < g(t_i)$, since XOR-ing in $X$ clears that highest bit of $g(t_i)$ and only lower
bits are affected. By the second fact, some successor $w$ of $t_i$ has $g(w) = g(t_i) \oplus X$ exactly,
since it is smaller than $g(t_i)$; moving token $i$ there changes the XOR to
$X \oplus g(t_i) \oplus g(w) = X \oplus g(t_i) \oplus (g(t_i) \oplus X) = 0$, a position the induction
hypothesis makes a loss for the opponent. So an $X \neq 0$ position always has a move to a loss for the
opponent, making it a win.

```python
def first_player_wins(n: int, edges: list[tuple[int, int]], tokens: list[int]) -> bool:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    grundy = [0] * n
    for v in reversed(_topo_order(n, edges)):
        seen = {grundy[w] for w in adj[v]}
        g = 0
        while g in seen:        # NOTE: mex(seen) takes at most len(seen) + 1 steps -- by pigeonhole, len(seen)
            g += 1               # distinct non-negative integers cannot cover every value in 0 .. len(seen)
        grundy[v] = g
    xor_sum = 0
    for t in tokens:
        xor_sum ^= grundy[t]
    return xor_sum != 0
```

Reuses `_topo_order` from Part 1, so nodes are visited in the same reverse-topological order that made
`outcomes_dag` correct, for the same reason: `grundy[v]` only ever needs the Grundy numbers of nodes
reachable from `v`, all already computed. The `while` loop touches at most $\deg(v) + 1$ values, so
building `seen` and finding the mex together cost $O(\deg(v))$; summed over every node, `first_player_wins`
is $O(n + m)$ time and space, matching Part 1. On a single token this agrees with Part 1 exactly:
$g(v) \neq 0$ if and only if `outcomes_dag` labels `v` as `"W"`, by the same induction over reverse
topological order — a sink has $g = 0$ and outcome `"L"`, and inductively $g(v) \neq 0$ iff some successor
$w$ has $g(w) = 0$ iff (by the induction hypothesis) some successor has outcome `"L"` iff `v` is `"W"` — a
fact the checks below confirm directly.

### Follow-ups

- **Tokens on a graph with cycles.** Once a single-token game can end in a draw, summing several no longer
  reduces to XOR-ing one number per token: two drawn components do not automatically sum to a draw, since
  one side may have a faster forced win elsewhere once combined with a component that is not drawn. C. A.
  B. Smith's 1966 extension of Sprague–Grundy theory to graphs handles this by attaching each node a richer
  value than a nimber — one that records both its normal-play behaviour and its capacity to stall forever —
  rather than the single integer that suffices once every component is guaranteed to terminate.
- **Misère play.** Flipping the rule so that the player unable to move *wins* instead breaks the XOR rule
  for a general sum of games — Part 3's reduction is a normal-play phenomenon. Misère Nim itself still has
  a simple closed form (play the normal-play strategy until every pile would drop to size 1, then leave an
  odd number of size-1 piles), but that trick is specific to Nim's pile structure and does not extend to an
  arbitrary sum of impartial games; the general theory needs *misère quotients*, a substantially larger
  algebraic structure than one nimber per game.
- **The length of the game under optimal play.** Retrograde analysis already computes this with one extra
  field: labelling `p` as `"W"` via a successor `x` also sets `dist[p] = 1 + dist[x]`, keeping the smallest
  such value over every qualifying `x` — the winner wants the fastest forced win; labelling `p` as `"L"`
  sets `dist[p] = 1 + max(dist[x] for x in adj[p])` over *all* of `p`'s now-all-`"W"` successors — the
  loser, forced to move, picks whichever reply delays the loss the longest. This is exactly the
  "distance to mate" computation behind endgame tablebases.
- **Tokens that interact.** Sprague–Grundy applies only because the tokens' games are genuinely
  independent — no rule lets one token's move affect what another token can do. A rule such as "a token
  landing on another token removes it" (cat-and-mouse-style capture) breaks that independence, and the XOR
  shortcut no longer applies; the fix is to fall back to Part 2's retrograde analysis run directly on the
  product state space, one state per tuple of token positions, which is strictly more general but loses
  the $O(n + m)$ bound, since the product graph has up to $n^k$ states for $k$ tokens.

<details>
<summary>Checks (runnable)</summary>

```python
import random


# --- worked examples from the statement ---
edges1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 5), (2, 3), (3, 4), (3, 5), (5, 6)]
assert outcomes_dag(7, edges1) == ["W", "W", "L", "W", "L", "W", "L"]
assert winning_move(7, edges1, 0) == 2
assert winning_move(7, edges1, 2) is None
assert winning_move(7, edges1, 6) is None

edges2 = [(1, 0), (2, 3), (3, 2), (4, 2), (4, 0), (5, 2), (5, 3), (6, 1)]
assert outcomes(7, edges2) == ["L", "W", "D", "D", "W", "D", "L"]

edges3 = [(1, 0), (2, 0)]
assert first_player_wins(3, edges3, [1]) is True
assert first_player_wins(3, edges3, [2]) is True
assert first_player_wins(3, edges3, [1, 2]) is False


# --- Part 1: an independent recursive minimax, memoised, on tiny random DAGs ---
def _random_dag(rng: random.Random, n: int, p: float) -> list[tuple[int, int]]:
    """Every edge goes from a lower-numbered node to a higher-numbered one, which rules out cycles by
    construction -- independent of _topo_order, the solution helper this graph is used to test."""
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _bf_dag_outcome(edges: list[tuple[int, int]], v: int, memo: dict) -> str:
    if v in memo:
        return memo[v]
    succs = [b for a, b in edges if a == v]
    result = "L"
    for w in succs:
        if _bf_dag_outcome(edges, w, memo) == "L":
            result = "W"
            break
    memo[v] = result
    return result


saw_w = saw_l = False
for seed in range(300):
    rng = random.Random(seed)
    n = rng.randint(4, 9)
    edges = _random_dag(rng, n, p=0.35)
    got = outcomes_dag(n, edges)
    memo: dict = {}
    want = [_bf_dag_outcome(edges, v, memo) for v in range(n)]
    assert got == want, (seed, n, edges, got, want)
    for v in range(n):
        wm = winning_move(n, edges, v)
        succs = [b for a, b in edges if a == v]
        if want[v] == "W":
            assert wm is not None and wm in succs and want[wm] == "L", (seed, v, wm)
            assert wm == min(w for w in succs if want[w] == "L")
            saw_w = True
        else:
            assert wm is None, (seed, v, wm)
            saw_l = True
assert saw_w and saw_l     # both branches of winning_move were actually exercised by the fuzzing


# --- Part 2: an independent depth-limited minimax, on hundreds of small random digraphs with cycles ---
def _random_digraph(rng: random.Random, n: int, p: float) -> list[tuple[int, int]]:
    return [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < p]


def _bf_cyclic_outcome(edges: list[tuple[int, int]], n: int, plies_bound: int) -> list[str]:
    """Independent depth-limited minimax: a node is "W" if the mover can force reaching a node with no
    outgoing edge within `plies_bound` plies, "L" if the opponent can force that against the mover within
    the bound, otherwise "D".

    plies_bound = 2 * n is a generous bound: `outcomes` only ever labels a node "W" or "L" through a chain
    of other, already-labelled nodes (a fresh "W" needs one already-"L" successor; a fresh "L" needs every
    successor already "W"), no node is labelled twice, and a graph of n nodes has no chain of more than
    n - 1 distinct nodes -- so every node `outcomes` actually decides is forced within n - 1 plies of real
    play, and 2 * n is comfortable headroom over that bound."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    memo: dict[tuple[int, int], str] = {}

    def solve(v: int, plies: int) -> str:
        if not adj[v]:
            return "L"                      # NOTE: no move at all -- decided in 0 plies, any bound sees it
        if (v, plies) in memo:
            return memo[(v, plies)]
        if plies == 0:
            memo[(v, plies)] = "U"
            return "U"
        results = [solve(w, plies - 1) for w in adj[v]]
        if "L" in results:
            out = "W"
        elif all(r == "W" for r in results):
            out = "L"
        else:
            out = "U"
        memo[(v, plies)] = out
        return out

    raw = [solve(v, plies_bound) for v in range(n)]
    return ["D" if r == "U" else r for r in raw]


saw_d = False
for seed in range(300):
    rng = random.Random(seed + 500)
    n = rng.randint(4, 6)
    edges = _random_digraph(rng, n, p=0.35)
    got = outcomes(n, edges)
    want = _bf_cyclic_outcome(edges, n, plies_bound=2 * n)
    assert got == want, (seed, n, edges, got, want)
    saw_d = saw_d or "D" in want
assert saw_d     # the fuzzing actually produced a draw at least once, not just decided positions

# --- Part 2 agrees with Part 1 on acyclic graphs ---
for seed in range(150):
    rng = random.Random(seed + 900)
    n = rng.randint(4, 8)
    edges = _random_dag(rng, n, p=0.35)
    assert outcomes(n, edges) == outcomes_dag(n, edges), (seed, n, edges)
    assert "D" not in outcomes(n, edges)


# --- Part 3: an independent minimax over the tuple of token positions, on tiny DAGs with 2-3 tokens ---
def _bf_multitoken_win(edges: list[tuple[int, int]], positions: tuple[int, ...], memo: dict) -> bool:
    if positions in memo:
        return memo[positions]
    for i, v in enumerate(positions):
        for w in [b for a, b in edges if a == v]:
            nxt = positions[:i] + (w,) + positions[i + 1:]
            if not _bf_multitoken_win(edges, nxt, memo):
                memo[positions] = True
                return True
    memo[positions] = False
    return False


saw_true = saw_false = False
for seed in range(200):
    rng = random.Random(seed + 1300)
    n = rng.randint(3, 5)
    edges = _random_dag(rng, n, p=0.45)
    k = rng.choice([2, 3])
    tokens = [rng.randrange(n) for _ in range(k)]
    got = first_player_wins(n, edges, tokens)
    want = _bf_multitoken_win(edges, tuple(tokens), {})
    assert got == want, (seed, n, edges, tokens, got, want)
    saw_true = saw_true or got
    saw_false = saw_false or not got
assert saw_true and saw_false

# a single token must agree with Part 1's outcomes_dag exactly
for seed in range(50):
    rng = random.Random(seed + 1700)
    n = rng.randint(3, 6)
    edges = _random_dag(rng, n, p=0.4)
    single = outcomes_dag(n, edges)
    for v in range(n):
        assert first_player_wins(n, edges, [v]) == (single[v] == "W"), (seed, n, edges, v)

print("all checks passed")
```

</details>

</details>
