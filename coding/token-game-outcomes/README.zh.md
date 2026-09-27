# 移动棋子游戏中的必胜态

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 图上的博弈搜索，BFS/DFS | ★★★★☆ | 困难 | SWE · RE · MLE · Intern | game-theory, dfs, retrograde-bfs, topological-order, sprague-grundy, graphs | 3 个部分 / 45–60 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

一个有向图的节点是 $0, 1, \ldots, n-1$，边由一个互不相同的边列表 `edges: list[tuple[int, int]]` 给出，
其中每条边 $(u, v)$ 满足 $u \neq v$，表示存在一条从 $u$ 指向 $v$ 的有向连接。一枚*棋子*（token）在任意
时刻都恰好停在一个节点上。两名玩家轮流行动；一次*移动*（move）把棋子从当前节点 $u$ 沿一条边 $(u, v)$
带到 $v$——如果 $u$ 的出边不止一条，由该轮行动的玩家选择走哪一条。如果一名玩家无法移动——即棋子所在
节点没有任何出边——他立刻输掉，游戏随之结束。

一个*局面*（position）只由棋子当前所在的节点确定（两名玩家在其余方面完全对称，因此不需要别的信息来区
分局面）。一个局面的*结果*（outcome）是 `"W"`，如果该局面下即将行动的玩家无论对手如何应对都能确保获
胜；是 `"L"`，如果无论即将行动的玩家如何应对，对手都能确保获胜；而一旦图中可能出现环（仅限 Part 2），
结果还可能是 `"D"`，表示双方都无法确保获胜，于是在双方都尽力避免输掉的最优对局下，游戏将永远进行下去。

### Part 1 —— 无环图

```py
def outcomes_dag(n: int, edges: list[tuple[int, int]]) -> list[str]: ...
def winning_move(n: int, edges: list[tuple[int, int]], v: int) -> int | None: ...
```

题目保证图是无环的（DAG）。`outcomes_dag` 返回一个长度为 `n` 的列表，其第 `i` 项是棋子在节点 `i` 上时
这个局面的结果，`"W"` 或 `"L"`。`n` 最多可达 $10^5$，所以调用栈深度随图中路径长度增长的解法是不可接受
的：计算每个结果必须使用节点上的某种显式顺序，而不是沿着边一条一条递归下去。

`winning_move(n, edges, v)` 返回编号最小的节点 `w`，满足边 `(v, w)` 存在且移动到那里是一步获胜的走
法——即 `w` 的结果是 `"L"`；如果节点 `v` 自身的结果就是 `"L"`，则返回 `None`，这意味着不存在这样的走
法：要么 `v` 根本没有出边，要么它的每一个后继节点的结果都是 `"W"`。

```text
0 -> 1, 2, 3
1 -> 2, 5
2 -> 3
3 -> 4, 5
4 -> (没有出边)
5 -> 6
6 -> (没有出边)
```

从汇点（sink，没有出边的节点）开始往上推：节点 4 和节点 6 都没有出边，所以落在这两点上的玩家立刻输
掉，两者都是 `"L"`。节点 5 唯一的一条边指向 6（`"L"`），所以在 5 上行动的玩家可以移动到那里，让对手
立刻陷入必败：`"W"`。节点 3 有边指向 4（`"L"`）和 5（`"W"`）；仅凭指向 4 的这条边，节点 3 就已经是
`"W"`。节点 2 唯一的边指向 3，而 3 的结果是 `"W"`——节点 2 没有任何一条边指向 `"L"` 节点，它的每一步
都会把胜利拱手让给对手，所以节点 2 自己是 `"L"`。节点 1 有边指向 2（`"L"`）和 5（`"W"`）；指向 2 的这
条边使节点 1 成为 `"W"`。节点 0 有边指向 1（`"W"`）、2（`"L"`）和 3（`"W"`）；指向 2 的边同样使节点
0 成为 `"W"`：

```text
outcomes_dag(7, edges) == ["W", "W", "L", "W", "L", "W", "L"]   # nodes 0..6, in order
```

`winning_move(7, edges, 0)` 返回 `2`：节点 0 的边依次指向 1、2、3，其中 2 是编号最小、结果为 `"L"`
的那一个。`winning_move(7, edges, 2)` 返回 `None`，因为节点 2 自身的结果已经是 `"L"`；
`winning_move(7, edges, 6)` 出于同样的原因，也返回 `None`。

### Part 2 —— 含环的图

```py
def outcomes(n: int, edges: list[tuple[int, int]]) -> list[str]: ...
```

现在图中可能出现环（仍然没有自环，因为每条边依旧满足 $u \neq v$）。返回一个长度为 `n` 的列表，其第
`i` 项是 `"W"`、`"L"` 或 `"D"`，即上面定义的、棋子在节点 `i` 上时这个局面的结果。

```text
0 -> (没有出边)
1 -> 0
2 -> 3
3 -> 2
4 -> 2, 0
5 -> 2, 3
6 -> 1
```

节点 0 没有出边，所以是 `"L"`。节点 1 唯一的边指向 0（`"L"`），所以节点 1 是 `"W"`。节点 2 和节点 3
只有彼此指向对方的边——`2 -> 3` 和 `3 -> 2`——而且两者都永远到不了节点 0 或者任何其他已确定的节点；
既然双方除了在这两个节点之间来回移动之外别无选择，对局就会永远进行下去，所以两者都是 `"D"`。节点 4
有边指向 2（`"D"`）和 0（`"L"`）；仅凭指向 0 的这条边，节点 4 就是 `"W"`，无论指向 2 的那条边会通向
什么结果。节点 5 只有指向 2 和 3 的边，二者都是 `"D"`——既没有边指向 `"L"` 节点，也不可能耗尽所有非
`"W"` 的选择，因为 2 和 3 永远不会变成 `"W"`——所以节点 5 同样是 `"D"`。节点 6 唯一的边指向 1
（`"W"`）；由于节点 6 的每一条边（这里只有这一条）都通向一个 `"W"` 节点，节点 6 是 `"L"`，无论它的
行动方怎么走都一样。

```text
outcomes(7, edges) == ["L", "W", "D", "D", "W", "D", "L"]   # nodes 0..6, in order
```

### Part 3 —— 多枚棋子

```py
def first_player_wins(n: int, edges: list[tuple[int, int]], tokens: list[int]) -> bool: ...
```

图再次是无环的，和 Part 1 一样。`k` 枚棋子分别停在图的若干节点上，由 `tokens` 给出，这是一个长度为
`k` 的节点编号列表；多枚棋子可以停在同一个节点上。现在一次移动可以选择任意一枚棋子，把它沿着它当前所
在节点的一条出边移动，规则和之前完全一样；其余 `k - 1` 枚棋子留在原地不动。如果一名玩家无法移动任何
一枚棋子——也就是说每一枚棋子所在的节点都没有出边——他就输了。如果先手玩家能确保获胜，返回 `True`，
否则返回 `False`。

```text
0 -> (没有出边)
1 -> 0
2 -> 0
```

单独一枚棋子停在节点 1 上时，无论谁先手都能获胜：唯一的走法是移动到 0，这会让对手面对一枚停在没有出
边的节点上的棋子，对手无法移动，只能认输：`first_player_wins(3, edges, [1]) == True`。同理，单独一
枚棋子停在节点 2 上时，先手同样必胜：`first_player_wins(3, edges, [2]) == True`。但当两枚棋子同时在
场，`tokens = [1, 2]` 时，先手必须移动其中一枚——比如把节点 1 上的那枚移到 0——接着后手把另一枚、还
留在节点 2 上的那枚也移到 0，这样先手就面对两枚都停在节点 0 上、谁都动不了的棋子：先手输了。先移动另
一枚棋子是对称的情形，结果同样是输，所以先手能选的每一步都会输：
`first_player_wins(3, edges, [1, 2]) == False`。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有两点值得先确认：在没有出边的节点上会发生什么——按照题目本身的规则，落在那里的玩家立刻输
掉，所以它的结果永远是 `"L"`——以及图中是否可能出现环：Part 1 和 Part 3 都不会（两者都保证图是无环
的），但 Part 2 会。

### Part 1

`outcome(v)` 是 `"W"`，当且仅当从 `v` 出发的某条边通向一个结果为 `"L"` 的节点；是 `"L"`，当且仅当从
`v` 出发的每一条边（也可能一条都没有）都通向一个结果为 `"W"` 的节点——这正是把题目里“确保获胜”这个
说法逐步展开一步之后的样子：要确保获胜，行动方需要至少一种应对方式，让对手落入与此完全相同的必败境
地；而要被逼入必败，则每一种应对方式都必须让对手能够反过来确保获胜。由于图是无环的，这个递归定义没有
任何循环需要解开：`v` 的结果只取决于从 `v` 可达的节点的结果，而从任意节点沿一条边走一步，都会到达拓
扑序（topological order）中更靠后的节点，所以只需按*逆*拓扑序——先处理汇点，只有当从 `v` 可达的每个
节点都已经处理完，才轮到 `v` 自己——遍历一遍节点，就能直接算出每个结果，而且每个节点恰好只被处理一次。

`_topo_order` 用 Kahn 算法（Kahn's algorithm）求出一个拓扑序：反复取出一个入边已经全部被移除的节点，
再依次移除它自己的出边。在无环图上，这会把整个节点集合清空——`_topo_order` 内部的断言会在图含有环时
触发，因为这时会有某个节点的入度永远降不到零——而且这个算法是迭代式的，用的是队列而不是调用栈，所以
它的深度和图中任何路径的长度都无关，这正好满足 $n$ 最多可达 $10^5$ 这一要求。

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

以 $n$ 个节点、$m$ 条边计，`_topo_order` 是 $O(n + m)$ 的：构建邻接表和入度数组是 $O(n + m)$，Kahn
算法对每个节点恰好移除一次，并且恰好检查它的每一条出边一次。`outcomes_dag` 里对 `reversed(...)` 的
遍历是另一个 $O(n + m)$，对每个节点只看一次 `adj[v]`。`winning_move` 在同一个 $O(n + m)$ 的调用之上，
额外加上一次 $O(\deg(v) \log \deg(v))$ 的排序。

### Part 2

当图中可能出现环时，Part 1 的递归可能出现循环——`outcome(v)` 可能依赖 `outcome(w)`，而 `outcome(w)`
又依赖 `outcome(v)`——因此不存在拓扑序，按固定顺序逐个求值的办法也就不再管用。逆向分析
（retrograde analysis）从已知的基础情形（汇点）出发向外扩展，而不是预先为每个节点定好处理顺序，从
而解决了这个问题：一旦某个节点的结果被已经确定的标签所决定，就立刻给它打上标签，并沿着反向边继续传
播——这和从目标出发反向计算可达性是同一个思路。每个汇点立刻就是 `"L"`。每当一个节点 `x` 被打上标签，
就去看 `x` 的每一个尚未打标签的前驱 `p`（也就是每一条边 `p -> x`）：如果 `x` 是 `"L"`，`p` 立刻变成
`"W"`，因为 `p` 有一步——走到 `x`——能把失败甩给对手；如果 `x` 是 `"W"`，`p` 就朝着被逼入 `"L"` 又近
了一步——为每个尚未打标签的 `p` 维护一个计数，记录它还有多少个后继节点尚未确定为 `"W"`，一旦这个计数
降到零，就把 `p` 标记为 `"L"`，因为这时 `p` 的每一种应对都已经证明会把胜利让给对手。直到再也产生不出
新标签为止，始终没有被标记的节点就是 `"D"`。

有三个命题需要证明。*每个标记为 `"L"` 的节点确实是必败的*：对标签被赋予的顺序做归纳，`p` 只有在它的
每一个后继节点都已经标记为 `"W"` 之后才会被标记为 `"L"`，而根据归纳假设，那些 `"W"` 标签本身都已经是
正确的，所以 `p` 的每一步（`p` 至少有一步可走，否则它一开始就是汇点）都会把真正的胜利交给对手。*每个
标记为 `"W"` 的节点确实是必胜的*：`p` 被标记为 `"W"` 的那一刻，恰好是发现它的某个后继节点 `x` 已经被
标记为 `"L"` 的时候；根据归纳假设（`x` 在严格更早的一步就已经被标记），那个 `"L"` 标签本身已经是正确
的，所以 `p` 确实有一步能把对手带入真正的必败。*每个始终没有被标记的节点都是平局*：这样的节点 `v`
永远不会有一个后继节点被标记为 `"L"`——一旦 `v` 的任何一个后继被标记为 `"L"`，`v` 自己会在同一步就被
标记为 `"W"`，因为一个节点被标记后会立刻扫描它的前驱——而且 `v` 总会至少有一个后继节点也始终没有被
标记：`v` 自己从未被标记为 `"L"`，按上面的计数规则，这意味着它那些尚未确定为 `"W"` 的后继数量永远降
不到零，所以至少有一个后继节点也始终没有被标记为 `"W"`，而根据前一点它也不可能是 `"L"`，于是它同样
始终未被标记（`v` 一开始就至少有一个后继：没有后继的节点是汇点，会立刻被标记）。于是，从任何一个始
终未被标记的节点出发，无论哪一方都总能移动到另一个始终未被标记的节点：这样走永远不会把即时的胜利拱
手让给对手，因为没有一步会走到 `"L"` 节点；也永远不会无路可走，因为只要一个行动方还有后继节点，就总
有一步可走——所以从一个始终未被标记的节点开始，对局可以永远进行下去，双方都无法逼出胜负，这正是
`"D"`。

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

每条边 `p -> x` 在整个运行过程中恰好被检查一次——也就是 `x` 出队的那一刻——所以 `outcomes` 的时间和
空间复杂度都是 $O(n + m)$，和 Part 1 一样，尽管这里允许出现环。在无环图上，每个节点最终都会被标记
（DAG 没有环可以在里面绕圈子），所以 `outcomes` 在无环图上永远不会返回 `"D"`，而是退化成和
`outcomes_dag` 完全相同的 `"W"`/`"L"` 划分——下面的验证代码会直接确认这一点。

### Part 3

只有一枚棋子时，某个节点上的博弈是题目本身早已固定的正常游戏约定（normal play convention，无法移动
的一方判负）下的一个*公平博弈*（impartial game）——在任意给定节点，双方能走的步法完全相同，这正是为
什么*局面*从头到尾只需要一个节点，从不需要标明是哪一方在动。当有多枚棋子时，一次移动就是从 `k` 个相
互独立的单棋子博弈里选一个来推进——这正是标准的*博弈之和*（sum of games），而 Sprague-Grundy 定理只
用一个数字就能刻画它，不必对整个乘积状态空间重新做一遍 W/L/D 分析。给每个节点 `v` 赋一个 *Grundy 数*
（Grundy number）$g(v) = \operatorname{mex}\{g(w) : w \text{ 是 } v \text{ 的后继}\}$，即*最小排除
数*（minimum excludant，mex）：不是 `v` 任何一个后继节点的 Grundy 数的、最小的非负整数（汇点没有后
继，所以 $g(\text{汇点}) = \operatorname{mex}(\emptyset) = 0$）。要推导的结论是：棋子分别停在节点
$t_1, \ldots, t_k$ 上时，先手能确保获胜，当且仅当 $g(t_1) \oplus \cdots \oplus g(t_k) \neq 0$，其中
$\oplus$ 是按位异或（XOR）——这正是 Nim 游戏里的 Bouton 法则，$g(t_i)$ 在其中扮演的正是第 $i$ 堆石
子数量的角色。

证明依赖 mex 的定义直接给出的两个事实：$g(v)$ 自身永远不会是 `v` 任何一个后继节点的 Grundy 数（根据
mex 的定义，$g(v)$ 本身正是被排除在外的那个值），而且每一个小于 $g(v)$ 的非负整数都*确实*是 `v` 某个
后继节点的 Grundy 数（否则那个更小的值就会被排除掉，成为 mex 本身）。对剩余步数做强归纳——步数是有限
的，因为图是无环的，而且每一步都会把某一枚棋子沿拓扑序严格向前推进——可以证明：异或值 $X = 0$ 的局面
对行动方是必败的，$X \neq 0$ 的局面是必胜的。若 $X = 0$：任何一步都会把恰好一枚棋子的贡献从 $g(v)$
变成某个后继 $w$ 的 $g(w)$，而由第一个事实 $g(w) \neq g(v)$，新的异或值是
$X \oplus g(v) \oplus g(w) = g(v) \oplus g(w) \neq 0$；根据归纳假设，非零异或值对下一个行动方是必胜
的，所以从 $X = 0$ 的局面出发的每一步都会把胜利交给对手，这就是必败。若 $X \neq 0$：设 $i$ 是某枚
Grundy 数 $g(t_i)$ 在 $X$ 的最高置位那一位上也置位的棋子（必然存在这样的棋子，否则 $X$ 的那一位就不
可能是 1）；那么 $g(t_i) \oplus X < g(t_i)$，因为异或上 $X$ 会清除 $g(t_i)$ 的那个最高位，只影响更低
的位。由第二个事实，$t_i$ 的某个后继 $w$ 恰好有 $g(w) = g(t_i) \oplus X$，因为它小于 $g(t_i)$；把第
$i$ 枚棋子移到那里，异或值就变成
$X \oplus g(t_i) \oplus g(w) = X \oplus g(t_i) \oplus (g(t_i) \oplus X) = 0$，根据归纳假设，这是一个
对手必败的局面。所以 $X \neq 0$ 的局面总有一步能把对手带入必败，这就是必胜。

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

沿用 Part 1 的 `_topo_order`，节点的访问顺序和让 `outcomes_dag` 得以成立的那个逆拓扑序完全一样，原
因也相同：`grundy[v]` 只需要用到从 `v` 可达的节点的 Grundy 数，而这些都已经算好了。`while` 循环最多
访问 $\deg(v) + 1$ 个值，所以构建 `seen` 和求 mex 加起来是 $O(\deg(v))$；对所有节点求和，
`first_player_wins` 是 $O(n + m)$ 的时间和空间，和 Part 1 一致。只有一枚棋子时，这和 Part 1 完全一
致：$g(v) \neq 0$ 当且仅当 `outcomes_dag` 把 `v` 标记为 `"W"`，理由是同一个按逆拓扑序展开的归纳——
汇点的 $g = 0$、结果是 `"L"`，归纳地，$g(v) \neq 0$ 当且仅当某个后继 $w$ 满足 $g(w) = 0$，当且仅当
（根据归纳假设）某个后继的结果是 `"L"`，当且仅当 `v` 是 `"W"`——下面的验证代码会直接确认这一事实。

### 追问

- **含环图上的多枚棋子。** 一旦单枚棋子的博弈可能以平局收场，把几个博弈相加就不再能简化成对每枚棋子
  各自的一个数字做异或：两个平局分量相加，结果未必还是平局，因为一旦和一个不是平局的分量组合在一起，
  某一方完全可能在别处找到一条更快的必胜之路。C. A. B. Smith 在 1966 年把 Sprague-Grundy 理论推广到
  含环的图上，办法是给每个节点赋一个比 Nim 值（nimber）更丰富的值——既记录它在正常游戏约定下的行为，
  又记录它能不能永远拖下去——而不是像每个分量都保证终止时那样，一个整数就够了。
- **反常游戏（misère play）。** 把规则反过来，让无法移动的一方反而*获胜*，会让异或法则在一般的博弈
  之和上彻底失效——Part 3 的化简是正常游戏约定下才有的现象。反常版 Nim 本身仍有一个简单的封闭解法
  （按正常游戏的策略走，直到每一堆都要降到 1 颗为止，然后改为留下奇数堆大小为 1 的石子），但这个技
  巧是 Nim 的堆结构特有的，推广不到任意的公平博弈之和；一般理论需要*反常商*（misère quotient），这
  是一个比每个博弈一个 Nim 值复杂得多的代数结构。
- **最优对局下的对局长度。** 逆向分析只需多维护一个字段就能算出这个量：通过后继 `x` 把 `p` 标记为
  `"W"` 的同时，也令 `dist[p] = 1 + dist[x]`，并在所有满足条件的 `x` 里取这个值最小的一个——胜者想
  要最快地逼出胜利；把 `p` 标记为 `"L"` 时，则令 `dist[p] = 1 + max(dist[x] for x in adj[p])`，取遍
  `p` 现在全部为 `"W"` 的*所有*后继——败者既然非走不可，就挑那种能把失败拖得最久的应对。这正是残局
  库（tablebase）背后计算杀棋步数（distance to mate）时使用的方法。
- **会相互影响的棋子。** Sprague-Grundy 之所以适用，仅仅因为各枚棋子的博弈是真正独立的——没有任何
  规则允许一枚棋子的移动影响另一枚棋子能做什么。像一枚棋子落到另一枚棋子所在节点上就将其吃掉这样的
  规则（类似猫捉老鼠式的捕获）会打破这种独立性，异或这条捷径也就不再适用；这时只能退回 Part 2 的逆
  向分析，直接在乘积状态空间上运行——每个状态对应一个棋子位置的元组——这样做更通用，但会失去
  $O(n + m)$ 的复杂度，因为对 `k` 枚棋子而言，乘积图最多有 $n^k$ 个状态。

<details>
<summary>验证代码（可运行）</summary>

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
