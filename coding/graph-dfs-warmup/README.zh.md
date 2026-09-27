# 热身题：数组扫描与图的深度优先遍历

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 数组与图遍历 | ★★★★☆ | 简单 | SWE · MLE · Intern | arrays, binary-search, graph-construction, dfs, iterative-dfs, connected-components, cycle-detection | 3 个部分 / 30–45 分钟 | 用人经理初筛 · 技能面（Skills） |
<!-- meta:end -->

## 题目

### Part 1 —— 高于阈值的元素

```py
def above_threshold(arr: list[int], threshold: int) -> tuple[int, list[int]]: ...
```

返回 `arr` 中严格大于 `threshold` 的元素个数，以及它们在 `arr` 中的下标，下标按递增顺序排列。等于
`threshold` 的元素不计入。`arr` 可以为空，此时结果是 `(0, [])`。

例如，在 `above_threshold([1, 5, 7, 2, 8, 3], 4)` 中，元素 `1`、`2`、`3` 都不超过阈值；`5`、`7`、`8`
都高于阈值，分别位于下标 `1`、`2`、`4`：

```text
above_threshold([1, 5, 7, 2, 8, 3], 4) == (3, [1, 2, 4])
```

接下来是同一个数组上的追问：连续问多个阈值。

```py
def count_above_many(arr: list[int], thresholds: list[int]) -> list[int]: ...
```

对 `thresholds` 中的每一个阈值，只返回 `arr` 中严格大于它的元素个数（规则与 `above_threshold` 相同），
按 `thresholds` 的顺序排成一个列表。记 $n$ 为 `len(arr)`、$q$ 为 `len(thresholds)`，这必须在
$O((n + q) \log n)$ 总时间内完成，绝不能是 $O(nq)$。

### Part 2 —— 从边构建图，并返回 DFS 顺序

```py
def build_graph(edges: list[tuple[str, str]]) -> dict[str, list[str]]: ...
def dfs_order(edges: list[tuple[str, str]], start: str) -> list[str]: ...
```

`edges` 是一个由节点名字组成的二元组列表，它们共同定义了一张无向图。一条边 `(u, v)`，只要
`u != v`，就把 `u` 和 `v` 双向连接起来。一条重复出现的边——不论方向是否和之前一致——不会带来任何
新信息。一条边 `(u, u)`（*自环*（self-loop））会把 `u` 添加为图中的一个节点，但这条边本身不带来任何
邻居。对每一个至少出现在一条边中的节点，`build_graph` 都返回它按递增顺序排好序的邻居列表（节点名字按
普通字符串比较）。

深度优先搜索（depth-first search，DFS）从一个节点出发，沿着一个邻居尽可能地往前走，直到无路可走才
回溯，去尝试下一个。`dfs_order` 从 `start` 出发，按递增顺序探索一个节点尚未访问过的邻居——这正是让
访问顺序唯一确定的原因——并返回从 `start` 可达的每一个节点，每个恰好一次，按它第一次被访问的顺序
排列（也就是它的*前序*（preorder））。如果 `start` 根本不是图中的节点（它不出现在任何一条边中），
结果就是 `[start]`。

```text
edges = [("A","B"), ("A","C"), ("B","D"), ("B","E"), ("C","F"), ("E","F")]

build_graph(edges) == {
    "A": ["B", "C"], "B": ["A", "D", "E"], "C": ["A", "F"],
    "D": ["B"],      "E": ["B", "F"],      "F": ["C", "E"],
}
```

`dfs_order(edges, "A")` 每次都走向尚未访问的邻居中最小的那个，只有无路可走时才回溯：

```text
A -> B -> D              # D 唯一的邻居 B 已经访问过 -- 回溯到 B
       -> E -> F -> C    # C 的邻居 A 和 F 都已经访问过 -- 一路回溯回 A，结束

dfs_order(edges, "A") == ["A", "B", "D", "E", "F", "C"]
```

把 `dfs_order` 实现两遍：先写一个直截了当的递归版本，再写一个迭代版本，它返回完全相同的顺序，却不会
像递归版本那样在一条 100,000 个节点的链上失败。

### Part 3 —— 连通分量与环

```py
def components(edges: list[tuple[str, str]]) -> list[list[str]]: ...
def has_cycle(edges: list[tuple[str, str]]) -> bool: ...
```

`build_graph` 构建出的图中，两个节点只要被某条由边组成的路径连接起来，就说它们属于同一个*连通
分量*（connected component）；每个节点都恰好属于一个连通分量，哪怕它除了一个自环之外没有任何别的
边。`components` 把每个连通分量都作为它自己的 DFS 顺序返回（沿用 Part 2 的规则，从这个分量里最小
的节点出发），并按这个最小节点本身的大小，排列这些分量。

```text
edges = [("B", "A"), ("A", "C"), ("E", "D"), ("G", "G")]

components(edges) == [["A", "B", "C"], ["D", "E"], ["G"]]
# {A, B, C} 通过 A 相连；{D, E} 通过 D-E 这条边相连；G 唯一的边是一个自环，所以 G 是一个完全没有
# 邻居的节点——自己单独构成一个大小为一的分量
```

*环*（cycle）是三个或更多互不相同的节点 $v_0, v_1, \ldots, v_{k-1}$ 组成的序列，其中每一对相邻的
节点，以及 $v_{k-1}$ 和 $v_0$，都由一条边相连。`has_cycle` 返回这张图——在经过和 Part 2 相同的规范化
之后：重复的边合并掉，自环不贡献任何邻居——是否在任何地方包含这样一个环。

```text
edges = [("A","B"), ("A","C"), ("B","D"), ("B","E"), ("C","F"), ("E","F")]   # 还是 Part 2 的那张图

has_cycle(edges) == True          # A -> B -> E -> F -> C -> A 是一个环；D 挂在 B 上，不在环内
has_cycle(edges[:-1]) == False    # 去掉最后一条边 ("E","F")，剩下的就是一棵树：没有环
```

## 参考解答

<details>
<summary>展开参考解答</summary>

开始之前值得先确认：对 Part 1 来说，比较是严格的，返回的下标按升序排列。对 Part 2 来说，图是无向
的，邻居按递增顺序访问——这正是让 DFS 顺序唯一确定的原因——而且 `start` 即使没有边，也仍然返回
`[start]`。Python 默认的递归深度上限（1,000 层）在 `dfs_order` 跑在很长的一条链上时很重要。

### Part 1

用 `enumerate` 扫一遍就能直接收集满足条件的下标，不需要重复计算。

```python
def above_threshold(arr: list[int], threshold: int) -> tuple[int, list[int]]:
    indices = [i for i, x in enumerate(arr) if x > threshold]
    return len(indices), indices
```

对单个阈值来说这是 $O(n)$：对 `arr` 扫一遍，每个元素比较一次。

一次问多个阈值，排序就开始划算了。`bisect_right(sorted_arr, t)` 返回 `sorted_arr` 中 `<= t` 的元素
个数（它的插入点把每一个等于 `t` 的元素都留在自己左边），所以用 $n$ 减去这个值，剩下的恰好就是严格
大于 `t` 的元素个数。

```python
from bisect import bisect_right


def count_above_many(arr: list[int], thresholds: list[int]) -> list[int]:
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    # NOTE: bisect_right, not bisect_left -- bisect_left's insertion point sits before every element
    # equal to t, so n - bisect_left(sorted_arr, t) would also count those equal elements as "above"
    return [n - bisect_right(sorted_arr, t) for t in thresholds]
```

排序一次是 $O(n \log n)$；$q$ 个阈值里的每一个各自做一次二分查找，代价 $O(\log n)$，加起来总共是
$O((n + q) \log n)$——而对每个阈值都调用一次 `above_threshold` 则是 $O(nq)$。

### Part 2

`build_graph` 在读取 `edges` 的过程中把邻居保存在一个 `set` 里，所以一条重复的边——不论方向——落到
一个已经有这个元素的集合里，不会有任何效果；一条自环 `(u, u)`（通过 `setdefault`）只会登记 `u` 这个
节点，却从不把 `u` 加进它自己的邻居集合。只在最后才把每个集合排序一次，比全程维护一个有序结构更省
事、也更简单。

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

直接写一个递归的 DFS 最贴近前序的定义：访问一个节点，然后按递增顺序递归地访问它每一个尚未访问的
邻居。

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

`visit` 的每一层调用帧都要一直留着，直到 DFS 树里它下面的所有节点都被完全探索完毕，所以一条很长的
节点链会留下同样长的一串未返回的调用帧；一旦超过 Python 默认的递归深度上限 1,000，
`dfs_order_recursive` 就会抛出 `RecursionError`（下面的验证代码会检查这一点）——修复办法是干脆不再
使用调用栈，而不是去调高这个上限。

用一个显式的栈——一个普通的 Python 列表——取代调用栈，它的大小就只受内存限制。把一个节点的邻居按
*逆序*排好序后依次压栈，并且只在真正出栈时才把节点标记为已访问，这样得到的顺序和递归版本完全一
致：最小的那个邻居最后压入，留在栈顶，最先被探索。

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

构建图会把每个节点的邻居列表各排序一次，记 $E$ = `len(edges)`，是 $O(E \log E)$；在此之后，$V$ 个
可达节点里的每一个都只被访问一次、只扫描一遍自己的邻居列表，所以每条边至多被检查两次（两个端点各一
次），每次检查至多压栈一个条目，而每次出栈要么访问一个节点、要么被跳过：合起来是 $O(V + E)$——不论
是 `dfs_order` 还是 `dfs_order_recursive` 来算，复杂度和顺序都完全一样。

### Part 3

`components` 按递增顺序遍历节点，每当遇到一个还没有被更早的分量触达的节点，就从它开始一次新的 DFS
（沿用 Part 2 的迭代规则）；因为节点是从小到大依次尝试的，这个起点必然是某个还没被触达的分量里最小
的节点，于是分量们从循环里出来时已经天然满足要求的顺序。

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

遍历部分和 `dfs_order` 一样是 $O(V + E)$，另加 $O(V \log V)$ 给节点排序：每个节点依然只被访问一
次、只扫描一遍邻居列表，只是分摊到了不同的分量里而已。

环检测在栈里把每个节点的*父节点*（parent，即发现它的那个节点）也一并带上，并且在一个邻居被发现的
那一刻、也就是压栈时——而不是像上面 `dfs_order` 那样在出栈时——就把它标记为已访问。这样一来，除了
每次搜索的起点之外，每个节点都恰好沿着一条边被发现，这些发现边构成了它所在分量的一棵生成树
（spanning tree）。当一个节点出栈、扫描它的邻居时，它的树边只有两种：连向它父节点的那条，以及连向
它在这次扫描中刚发现的那些邻居的边；所以一个已经访问过、又不是它父节点的邻居，与它之间的那条边一定
不在树中，这条边加上树中连接两者的路径就闭合成一个环。反过来，如果没有任何一次扫描找到这样的邻居，
那么每条边都是树边，每个分量都是一棵树，也就没有环。

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

还有一种计数判据，只凭节点数、边数和分量数就能回答同一个问题。记 $V$ 为节点数，$E$ 为（规范化之后的
无向）边数，$C$ 为分量数。一棵有 $v$ 个节点的树恰好有 $v - 1$ 条边——$v = 1$ 时显然成立，而且每当一棵树
多长出一个节点和一条边，这个性质都保持不变——所以一个共 $C$ 个分量、$V$ 个节点的*森林*（forest，即每
个分量都是一棵树）恰好有 $\sum_i (v_i - 1) = V - C$ 条边。任何一个分量都至少需要 $v_i - 1$ 条边才能保
持连通，所以总是 $E \ge V - C$，等号成立当且仅当每个分量都是一棵树；某个分量里的一个环，会把它的边数、
从而把总数，严格推高到这个下限之上（去掉环上的一条边后，这个分量仍然连通，所以除它之外还至少剩下
$v_i - 1$ 条边）。所以这张图存在环，当且仅当 $E > V - C$——下面的验证代码正是用这个事实作为 `has_cycle`
的独立检验。

### 追问

- **有向图。** 把每一条 `(u, v)` 都当作单向边，会把 `build_graph` 变成一个有向邻接表（只保留
  `neighbours[u].add(v)`），环检测也要跟着变：基于父节点的做法会失效，因为一条指向已访问节点的边未必闭合
  出有向环（在 A→B、B→C、A→C 中，C 被到达了两次，却没有环），所以有向图里的环要靠 DFS 过程中给节点做
  白/灰/黑三色标记来找——一条边指向一个灰色节点，就说明找到了环。同一次 DFS，只要在一个节点刚变黑的那一
  刻把它加到结果最前面，就能给出拓扑序。
- **无权图的最短路径。** `dfs_order` 会访问每一个可达节点，但不是按到 `start` 的距离递增的顺序；当
  问题是最少经过多少条边、而不只是能不能到达时，广度优先搜索——用队列代替栈，让距离为 $d$ 的节点都
  在距离 $d + 1$ 的节点之前出队——才是合适的工具，而且完全不需要父节点这种技巧来避免环，因为一个已
  经入队、或者已经出队的节点根本不会被第二次入队。
- **邻接矩阵。** 一个 $V \times V$ 的布尔数组（先把节点编号为 $0$ 到 $V - 1$），`matrix[i][j]` 为真当且仅
  当节点 $i$ 和 $j$ 相连，用 $O(V^2)$ 的空间换掉了 `build_graph` 的 $O(V + E)$；一旦图足够稠密、$E$ 接近
  $V^2$，这笔交易就划算了；它还能在 $O(1)$ 时间内回答“$u$ 和 $v$ 是否相邻”，不必扫描邻居列表；但对于稀疏
  图（每个节点只有少数几条边），上面的邻接表仍然是更好的选择。
- **还原 DFS 树。** 往栈里压入 `(neighbour, node)` 二元组而不是单独的节点，并在一个节点第一次出
  栈、被访问时记下这个二元组，就得到了真正发现每个节点的那条边，进而还原出 DFS 实际走过的那棵树，而
  不只是访问顺序。一个节点如果被压栈了不止一次，算数的是最先出栈的那一次（在 NOTE 的例子里，C 来自
  D，而不是 A）；而在无向图里，树之外的每条边都把一个节点连到它的某个祖先，所以一个分量无环，当且仅
  当它的树用上了它所有的边。

<details>
<summary>验证代码（可运行）</summary>

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
