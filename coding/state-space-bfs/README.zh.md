# 穿过上锁的门：最短路径与最短路计数

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 状态空间上的 BFS | ★★★★★ | 困难 | SWE · RE · MLE · Intern | bfs, state-space-search, bitmask, path-counting, dijkstra, grid | 3 个部分 / 45–60 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

一个机器人在一个矩形网格上移动，网格以 `grid: list[str]` 的形式给出，是一个长度相等的字符串列表。每个
字符是六种格子之一：`#` 表示墙（wall）；`.` 表示地板（floor）；`S` 表示机器人的起点（整个网格里恰好
一个）；`T` 表示目标（整个网格里恰好一个）；小写字母 `a`–`f` 表示一把钥匙（key）；对应的大写字母
`A`–`F` 表示一扇门（door）。就移动而言，`S` 和 `T` 本身也算作地板。一个网格里最多出现六种不同的钥匙
字母，所以任意时刻持有的钥匙集合都能放进一个 6 位的掩码（mask）里。

一次*移动*（move）把机器人从当前格子带到（最多四个）边相邻的格子之一——即相差一行或一列，绝不是对角
线方向——前提是该格子在网格内且不是墙。移动到一扇门上和其他移动一样，只是仅当机器人已经持有同一个
字母（小写形式）对应的钥匙时才合法；移动到一个钥匙格子上总是合法的，而且机器人一到达该格子就会自动、
永久地拾取这把钥匙——已经持有的钥匙再拾取一次没有任何效果，也没有任何格子会消耗钥匙。机器人可以任意
次数地重新经过任何格子，包括已经拿到钥匙的格子，或者已经走过的格子。

一条*路径*（path）是一个格子序列 $c_0, c_1, \ldots, c_n$，满足 $c_0 = S$；每一步从 $c_i$ 到
$c_{i+1}$，在已经经过 $c_0, \ldots, c_i$ 之后所持有的钥匙下都是一次合法移动；并且 $c_n = T$ 是序列中
第一次出现 $c_i$ 等于 $T$ 的位置——一旦到达 $T$，这次行走就结束了，路径不会在 $T$ 之后继续。两条路径
相同，当且仅当它们是同一个格子序列；两条路径即使用相同的步数到达 $T$，只要有任何一个位置的格子不同，
就是不同的路径。

### Part 1 —— 最少步数

```py
def shortest_path(grid: list[str]) -> int: ...   # -1 if T is unreachable
```

返回从 `S` 到 `T` 的最短路径的步数；如果没有任何路径能到达 `T`，返回 `-1`。

```text
grid = [
    "S..AT",
    "##.##",
    "..a..",
]
```

第 0 行是 `S . . A T`，一上来就被挡住了：`(0, 3)` 处的门需要钥匙 `a`，而钥匙在 `(2, 2)`，往下两行，
并且只能经由 `(1, 2)`——第 1 行墙上唯一的缺口——才能到达。机器人必须绕路：向右、向右、向下、向下
（在 `(2, 2)` 拾取钥匙 `a`），再把这段绕路走回去——向上、向上——门才会放行，最后向右、向右，进入
`T`：

```text
(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)* -> (1,2) -> (0,2) -> (0,3) -> (0,4)
                                     ^ 在这里拾取钥匙 a
```

一共八步；`(0, 2)` 和 `(1, 2)` 各经过了两次，这是唯一一条最短路径：`shortest_path(grid) == 8`。

第二个网格根本没有任何路径能到达终点：

```text
grid = [
    "S.AT",
    ".#.#",
    ".#a#",
]
```

钥匙 `a` 在 `(2, 2)`，它唯一的邻居是 `(1, 2)`，而 `(1, 2)` 唯一的另一个邻居正是门 `(0, 2)` 本
身——恰恰是需要这把钥匙才能进入的那个格子。谁都到不了 `a`，门永远打不开，`T` 也就无法到达：
`shortest_path(grid) == -1`。

### Part 2 —— 最短路径有多少条

```py
def count_shortest_paths(grid: list[str], mod: int = 1_000_000_007) -> int: ...   # 0 if unreachable
```

返回从 `S` 到 `T` 的不同最短路径的数量，对 `mod` 取模（按照上面*路径*的定义，两条路径只要在任何一个
位置上不同就算不同——到达同一个格子的不同路线，或者顺序不同的同一条路线，即使步数相同也算不同的路
径）；如果 `T` 不可达，返回 `0`。

把 Part 1 网格里 `(1, 0)` 处的墙重新打开：

```text
grid = [
    "S..AT",
    ".#.##",
    "..a..",
]
```

最短长度仍然是 8——Part 1 里的那条路径依然有效——但现在多了第二种到达钥匙的方式，沿着刚打开的左侧
往下走，而不是从上方绕路，总步数同样恰好是八步：

```text
(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)* -> (1,2) -> (0,2) -> (0,3) -> (0,4)   # 走上方，和 Part 1 一样
(0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2)* -> (1,2) -> (0,2) -> (0,3) -> (0,4)   # 走刚打开的左侧
```

两条路径都在第 4 步到达钥匙，之后都经过同样的四步走出来，穿过 `(1, 2)`、`(0, 2)` 和已经解锁的门；它
们在第 1 到 3 个位置上不同，所以算作两条不同的路径：`count_shortest_paths(grid) == 2`。

### Part 3 —— 泥地

标记为 `~` 的格子是泥地（mud）：移动到泥地上代价是 2 而不是 1（离开泥地、移动到一个普通格子上时，代
价只是那个格子本身的价格，绝不会更多）。针对这种代价模型再实现一遍这两个函数，这次要最小化的是总代价
而不是步数：

```py
def shortest_path_weighted(grid: list[str]) -> int: ...
def count_shortest_paths_weighted(grid: list[str], mod: int = 1_000_000_007) -> int: ...
```

`shortest_path_weighted` 返回从 `S` 到 `T` 的最小总代价（不可达时返回 `-1`）；`count_shortest_paths_weighted`
返回达到这个最小代价的不同路径有多少条，对 `mod` 取模。钥匙、门和墙的行为和之前完全一样——变化的只是
进入一个格子的价格。如果网格里完全没有 `~`，这两个函数必须和 Part 1、Part 2 的结果完全一致，因为这时
每一步的代价都是 1，总代价就等于步数。

```text
grid = [
    "S~~T",
    "....",
]
```

直接穿过泥地的代价是 $2 + 2 + 1 = 5$，用三步；绕远路、经过四个普通地板格子的代价是
$1+1+1+1+1 = 5$，用五步——步数不同，总代价却相同，所以论代价，两者不分优劣：

```text
(0,0) -> (0,1)~ -> (0,2)~ -> (0,3)                    # 代价 2 + 2 + 1 = 5，三步
(0,0) -> (1,0) -> (1,1) -> (1,2) -> (1,3) -> (0,3)     # 代价 1x5 = 5，五步
```

`shortest_path_weighted(grid) == 5`，`count_shortest_paths_weighted(grid) == 2`。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有三点值得先确认：机器人能不能重新经过一个格子，包括已经拿到钥匙的格子（能——题目里没有任何
地方禁止这样做，而且 Part 1 自己的例子就需要这样做）；钥匙用一次是否会被消耗（不会——门只检查对应的
位是否已经置位，从不清除它）；`S` 和 `T` 本身是否算作普通地板（算——机器人可以自由地跨过其中任何一
个，`T` 只是在被到达的那一刻结束这次行走）。

### Part 1

困难之处在于，一个格子是否可以进入，不只取决于这个格子本身：机器人能不能踏上一扇门，取决于它当前持有
哪些钥匙，而这又取决于它是怎么走到这里的。单纯在 $(row, col)$ 上做 BFS 没有办法表达这一点——它的已
访问集合会在一个格子第一次被到达时就把它永久标记为已完成，所以如果到达某个关键格子最快的方式恰好是在
还没拿到某把钥匙的时候到达的，那么记录下来的就是这次到达，之后哪怕手里已经有了钥匙，搜索也再也不会重
新尝试这同一个格子。把状态扩展成三元组 $(row, col, mask)$，其中 `mask` 的第 $i$ 位记录是否持有第 $i$
个字母对应的钥匙，然后在状态而不是格子上做 BFS，就解决了这个问题：两个状态相同，当且仅当位置和钥匙集
合都相同，所以一个格子曾经在没有钥匙时起作用、后来在有钥匙时又起作用，就会被正确地当作两个不同的状态
各访问一次。

在 Part 1 的第一个网格上，单纯的按格子做 BFS 会在距离 2 处、经过 `(0, 1)`、在还没有钥匙的情况下到达
`(0, 2)`，并把它标记为完成。而任何一条经过 `(0, 3)` 处的门的合法路线，都需要在已经持有钥匙的情况下
再次到达 `(0, 2)`，但这已经被单纯的 BFS 排除了，因为它已经把 `(0, 2)` 标记为访问过了；于是它会报告
`T` 不可达，也就是 `-1`，而真正的答案是 8 步（下面的验证代码会检查这一点）。状态空间上的 BFS 不会有
这个问题，因为 `(0, 2, 0)`（没有钥匙）和 `(0, 2, 1)`（持有钥匙 `a`）是两个不同的状态，会在不同的轮
次里分别被访问。

设网格有 $R$ 行、$C$ 列，其中出现 $k \le 6$ 种不同的钥匙字母，那么状态最多有 $R \cdot C \cdot 2^k$
个，每个状态至多有四条要检查的出边，每条检查花费 $O(1)$，所以在状态图上做 BFS 的时间和空间都是
$O(R \cdot C \cdot 2^k)$——下面的已访问集合（`dist`）是按状态而不是按格子来索引的。`T` 是终止节点：
一旦某个状态的格子是 `T`，到达它的这次行走就已经结束了，所以这个状态永远不会被展开，这和题目里*路
径*的定义（第一次到达 `T` 之后不再继续）是一致的。

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

`shortest_path` 取的是所有格子为目标的状态里的最小值——机器人可能沿着不同的路线、带着不同的钥匙集合
到达 `T`，总体的最短距离就是这些距离里最小的那个，与到达时手里究竟还持有哪些钥匙无关。

### Part 2

`shortest_path` 已经按距离不递减的顺序，把每个状态恰好访问了一次（这是 BFS 分层的标准性质，因为每次
转移的代价都是 1）；`count_shortest_paths` 复用同一次遍历，在 `dist` 之外再维护 `ways`，记录到达每
个状态的不同最短*状态*序列有多少条。一个状态第一次被发现时，是由恰好一个前驱状态发现的，`ways` 就被
设成这个前驱自己的计数；如果又有第二个前驱在相同的距离上到达同一个状态——也就是某个已经在 `dist` 里
的状态满足 `dist[ns] == d + 1`——那这是一条确实不同、但并列最短的路线，它的计数应该被累加进去，而不
是替换掉已有的值。由于 BFS 会在处理任何距离为 $d + 1$ 的状态之前，先处理完所有距离为 $d$ 的状态，所
以等到 `ns` 真正出队的时候，所有可能贡献给 `ways[ns]` 的前驱都已经处理完毕，`ways[ns]` 在被读取之前
必然已经是完整的。

这里数的是状态序列，但题目里*路径*的定义是*格子*序列，所以需要说明两者是一致的。对任意一个合法的格
子序列 $c_0, \ldots, c_n$，经过 $c_i$ 之后持有的钥匙集合完全由前缀 $c_0, \ldots, c_i$ 本身决定——它
就是这些格子里所有钥匙格子对应位的或（OR），因为拾取是自动的、永久的，不依赖路径的其他任何信息。所以
这个格子序列恰好确定了一个状态序列 $(c_0, m_0), \ldots, (c_n, m_n)$，而根据 `_transitions` 的构造，
这个状态序列本身就是状态图里的一次合法行走。反过来，从任意一个合法的状态序列里去掉掩码部分，也能还原
出原问题里的一个合法格子序列。两个不同的格子序列，其格子部分本来就不同，因此对应的状态序列也不同；所
以这个对应关系是两者之间保长度的双射（bijection），特别是它把最短的格子序列和最短的状态序列一一对应
起来——数其中一个就是在数另一个。

`T` 在状态图里是终止节点，所以一个到达目标状态的状态序列不能继续往下走，这自动就和“在第一次到达 `T`
时结束”对上了；但目标可能在好几个不同的掩码下被到达，对应的距离也可能不同（有的路线到达时持有的钥匙
比别的路线多），所以答案只对处于*整体*最小距离上的那些目标状态求和，把在那之后才到达的目标状态都舍
弃掉。

```python
def count_shortest_paths(grid: list[str], mod: int = 1_000_000_007) -> int:
    target, dist, ways = _bfs(grid)
    at_target = [(d, ways[s]) for s, d in dist.items() if s[0] == target[0] and s[1] == target[1]]
    if not at_target:
        return 0
    best = min(d for d, _ in at_target)
    return sum(w for d, w in at_target if d == best) % mod    # NOTE: only the states at the minimum distance
```

时间和空间复杂度和 Part 1 一样，都是 $O(R \cdot C \cdot 2^k)$：`ways` 只是在 `_bfs` 的基础上给每次
转移增加了 $O(1)$ 的工作量，没有带来新的渐近开销。

### Part 3

现在每一步的代价不再统一是 1，所以单纯 BFS 的分层不再对应最小代价——就像上面的例子那样，一条经过两
格泥地、只走三步的路径，和一条绕开泥地、走五步的路径，代价可以打平，所以哪一条更“短”取决于代价，而
不是步数。Dijkstra 算法把 Part 1 的 BFS 推广到这种情形：从一个按当前最优代价排序的最小堆里弹出状
态，一个状态一旦被弹出就*确定*（settled）了——它最终的最小代价已经知道了，因为每条边的代价至少是
1，而堆每次弹出的都是全局最小的当前代价，所以之后确定的任何状态都不可能再让它的代价变得更小。

在 Dijkstra 上数最短路径的条数还需要多一个事实：因为每条边的代价至少是 1，一个状态 $v$ 的任何前驱的
最终代价都严格小于 $v$ 自己的代价，所以——用同样关于弹出顺序的论证——$v$ 的每一个前驱在 $v$ 被弹出
之前必然已经确定了。这意味着，在 $v$ 被弹出之前的任何时刻，只要某个前驱把 $v$ 松弛到一个打平的代
价，就可以放心地把这个前驱的计数累加进 `ways[v]`；而 `ways[v]` 只有从 $v$ 被确定的那一刻起才是可信
的最终值——这正好是把 Part 2 里“已经处于距离 $d$”换成“已经确定”之后的同一套论证。由此直接引出三
个经典错误，分别在会犯错的那一行用 `# NOTE:` 标出：在一个状态还没确定之前就采信它对 `ways` 的贡献；
把一条并没有打平当前最优代价的边（*松弛*边，而不是*紧*边）也算进计数；以及忘记 `T` 一旦被弹出就是终
止节点、不应该再被展开，这和 Part 1、Part 2 是一样的。

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

用二叉堆实现时，$O(R \cdot C \cdot 2^k)$ 个状态里的每一个，每次让它的当前最优代价变得更好时最多被压
入一次堆，作为已确定状态最多被弹出一次，每个状态至多对应 4 条出边，所以总时间是
$O(R \cdot C \cdot 2^k \log(R \cdot C \cdot 2^k))$。由于这里的代价只有 1 或 2 两种取值，用一个*桶队
列*（bucket queue）——一个按总代价索引、大小为 $2 \cdot R \cdot C \cdot 2^k$ 的桶数组，每个桶存放当
前处于该代价的状态，按下标依次向前推进；这是把 0-1 BFS 推广到两种取值的权重——可以按同样的顺序确定
状态，且不需要堆带来的 $\log$ 因子，时间是 $O(R \cdot C \cdot 2^k)$；Dijkstra 更容易写对，下面的验
证代码跑的就是它。如果网格里没有 `~`，`_step_cost` 永远返回 1，每条边都打平，这就恰好退化成 Part
1、Part 2 的 BFS，只是访问状态的顺序不同，算出的 `dist` 和 `ways` 完全一样。

### 追问

- **用曼哈顿距离（Manhattan distance）作启发函数的 A-star。** 在给状态的堆排序之前，给它的当前代价加上
  $h(r, c) = |r - r_T| + |c - c_T|$ 是可采纳的（admissible）——它绝不会高估，因为每次移动都恰好让行
  或列变化 1，代价至少是 1，所以 $h$ 只可能低估剩下还需要的步数——并且它仍然和 Dijkstra 一样，最先
  找到的就是最短路径；一旦 `T` 大致位于网格中大多数状态的某一个方向上，它通常比单纯的 Dijkstra 确定
  更少的状态，不过这个启发函数只依赖于位置、不依赖于掩码，所以对搜索里真正关于怎么绕路去拿钥匙的那
  部分帮不上忙。
- **双向搜索（bidirectional search）。** 让从 `S` 出发的正向搜索和从 `T` 出发的反向搜索相向而行，
  大致能把每一侧要覆盖的半径减半，但反向的一侧一开始并不知道机器人到达 `T` 时手里握着哪些钥匙——每
  一种掩码都是一个说得通的终止状态——所以它必须同时从全部 $2^k$ 种掩码出发展开搜索，只有当正向的某
  个状态和反向的某个状态在位置和掩码上都一致时，两边才算接上；只有当 $k$ 足够小、这样做比直接从
  `S` 搜索整个半径更便宜时，相向而行省下的开销才是真的。
- **网格规模远超 $R \cdot C \cdot 2^6$ 能承受的范围。** 一旦钥匙被其他钥匙或者长长的走廊挡住，理论
  上 $2^k$ 种掩码里的大多数其实根本到不了，所以像上面那样，把 `dist` 和 `ways` 建成以 BFS 实际发现
  的状态为键的普通字典，而不是为每一种掩码都预先分配一个数组位置，这样状态空间自然就只有*可达*状态
  那么大，而不是最坏情形的理论上界。
- **不只是最短路径，而是前 $k$ 短路径。** 数出在最小距离上打平的路径条数，和列出第二短、第三
  短……并不是同一个问题，其中有些甚至根本不是并列的最短路径；这需要 Eppstein 算法，或者直接在上面
  的状态图上运行一个 $k$ 短路径版本的 Dijkstra，把只保留一个状态第一次被确定的那次，换成保留它最好
  的 $k$ 次。
- **取模的作用。** `mod = 1_000_000_007` 把返回的计数限制在一个固定宽度的整数范围内，这是从那些不
  取模就会溢出的语言里沿用下来的惯例；Python 的整数不会溢出，但保留这次取模的理由和竞赛题的答案总
  是要对一个接近 $2^{30}$ 的质数取模是一样的——沿途打平的分支会让路径数量呈指数增长，提前取模能让
  每一次中间求和都保持廉价。

<details>
<summary>验证代码（可运行）</summary>

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
