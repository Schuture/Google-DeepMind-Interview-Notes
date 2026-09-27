# 把单位换算建成带权图

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 建图与图遍历 | ★★★☆☆ | 中等 | SWE · MLE · RE · Intern | graph, dfs, bfs, weighted-union-find, consistency-check, floating-point | 3 个部分 / 45 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

一条*事实*（fact）是一个三元组 `(a, b, r)`：两个不同的单位名 `a` 和 `b`（任意非空字符串），以及一个正
实数 `r > 0`，表示 1 个 `a` 等于 `r` 个 `b`——例如 `("km", "m", 1000.0)` 表示 1 km 等于 1000 m。
`facts` 列表定义了一张有向带权图：每条事实贡献一条权重为 `r` 的边 `a -> b`，以及一条权重为 `1 / r`
的反向边 `b -> a`（如果 1 个 `a` 是 `r` 个 `b`，那么 1 个 `b` 就是 `1 / r` 个 `a`）。这张图的节点——
也就是*单位*（unit）——是出现在至少一条事实中的每一个不同名字，不论是作为 `a` 还是作为 `b`；当一系列
边组成的路径把两个单位连接起来时，它们就是*连通*（connected）的，沿一条路径从其中一个单位到另一个
单位的*换算系数*（conversion factor），就是这条路径上所有边权重的乘积。Part 1 和 Part 3 假设 `facts`
是内部一致的：同一对单位之间的任意路径都给出相同的换算系数，所以搜索恰好找到哪一条路径并不重要——
Part 2 讨论的正是这个假设不成立的情况。下面返回的系数是若干浮点数的乘积（Part 3 中还会出现除法），
因此可能带有普通的浮点舍入误差；下文中的每一次比较——包括验证代码里的比较，以及 `rel_tol` 内部的
比较——用的都是容差，从不使用 `==`。

### Part 1 —— 回答换算查询

```py
def convert(facts: list[tuple[str, str, float]], value: float, src: str, dst: str) -> float | None: ...
```

返回用 `dst` 表示的 `value`：先用 `facts` 建一次图，再把 `value` 乘以从 `src` 到 `dst` 的任意一条
路径上的换算系数。对于认不出的名字，`convert` 从不抛出异常，而是返回 `None`——当 `src` 或 `dst` 是
任何事实里都没出现过的单位名时如此，当两者都是已知单位、但没有路径把它们连接起来时也是如此。如果
`src == dst`，只要这个单位在某条事实中出现过，就原样返回 `value`；如果这个单位从未出现过，即便
`src` 和 `dst` 相同，也仍然返回 `None`，和其他任何未知单位一样。

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
# 路径 mile -> foot -> inch -> cm，三条边，权重依次为 5280.0、12.0、2.54
# 系数 = 5280.0 * 12.0 * 2.54 = 160934.4
# -> 2.0 * 160934.4 = 321868.8

convert(facts, 5.0, "km", "km")     # src == dst，且 "km" 在某条事实中出现过 -> value 原样返回
# -> 5.0

convert(facts, 1.0, "m", "parsec")  # "parsec" 未出现在任何事实中
# -> None

convert(facts, 1.0, "km", "byte")   # 两者都出现过，但 "byte"/"bit" 是另一个不相连的连通分量
# -> None
```

### Part 2 —— 检测矛盾

```py
def first_contradiction(facts: list[tuple[str, str, float]], rel_tol: float = 1e-9) -> int: ...
```

按顺序处理 `facts`，每次加入一条事实地建图。当 `a` 和 `b` 已经被之前的事实连接起来、且这些事实所
隐含的 `a` 到 `b` 的换算系数与 `r` 不接近时，就说这条事实 `(a, b, r)` 与之前的事实*矛盾*（contradicts）：
记这个隐含系数为 `f`，精确的条件是 `not math.isclose(f, r, rel_tol=rel_tol)`
（等价于 `abs(f - r)` 超过 `rel_tol` 乘以 `abs(f)` 和 `abs(r)` 中较大的那一个）。如果一条事实提到
的是此前从未出现过的单位，或者是把两个此前互不相连的连通分量连接起来，它就永远不会与任何东西矛盾，
因为此时根本没有可比较的对象。返回第一条与之前的事实矛盾的事实的下标；如果没有任何一条事实矛盾，
就返回 `-1`。

```text
facts = [
    ("a", "b", 2.0),     # 0
    ("c", "d", 5.0),     # 1
    ("b", "c", 3.0),     # 2
    ("a", "d", 100.0),   # 3
]

first_contradiction(facts)
# 下标 0："a"、"b" 此前未出现过 -> 直接加入；1 a = 2 b
# 下标 1："c"、"d" 此前未出现过 -> 直接加入；1 c = 5 d（目前是另一个连通分量）
# 下标 2："b" 和 "c" 分属不同的连通分量 -> 把它们连接起来，此时无可比较对象；
#         1 b = 3 c，于是合并后的分量隐含 1 a = 2 * 3 = 6 c = 6 * 5 = 30 d
# 下标 3："a" 和 "d" 此时已经连通；隐含系数是 30.0，而这条事实给出的是 100.0：
#         |30.0 - 100.0| / max(30.0, 100.0) = 0.7，远超 rel_tol -> 矛盾
# -> 3
```

### Part 3 —— 大量事实与查询交替进行

`Converter` 支持以任意顺序*在线*（online）交替调用 `add_fact` 和 `query`，每次调用都是近乎常数的
均摊时间——任何一次调用都不能把此前见过的所有事实重新扫描一遍。`add_fact(a, b, r)` 记录多一条事实，
同样要求 `a != b` 且 `r > 0`：如果 `a` 和 `b` 已经连通，而隐含的系数与 `r` 冲突（判定规则与 Part 2
相同，`rel_tol = 1e-9`），就什么都不改变，返回 `False`；否则就把这条事实并入结构——按需要新增一个
单位，或者合并两个连通分量——并返回 `True`。`query(src, dst)` 返回一个系数，使得 1 个 `src` 等于
这么多个 `dst`，且从不抛出异常，判定连通性的规则与 Part 1 的 `convert` 相同：如果两个单位中有任何
一个从未被 `add_fact` 接受过，或者它们没有被目前已接受的事实连接起来，就返回 `None`。

```py
class Converter:
    def add_fact(self, a: str, b: str, r: float) -> bool: ...        # False (and no change) if it contradicts
    def query(self, src: str, dst: str) -> float | None: ...          # factor such that 1 src = factor dst
```

`Converter()` 不接受任何参数，构造出来时还没有任何已知单位。

```text
c = Converter()
c.add_fact("km", "m", 1000.0)      # -> True
c.add_fact("m", "cm", 100.0)       # -> True
c.query("km", "cm")                # -> 约为 100000.0（有普通的浮点舍入误差，见上文）
c.add_fact("inch", "cm", 2.54)     # -> True
c.query("km", "inch")              # -> 约为 39370.0787
c.add_fact("km", "inch", 39370.0)  # -> False：真实系数约为 39370.0787，不是 39370.0 -- 被拒绝
c.query("km", "inch")              # -> 约为 39370.0787，与被拒绝的那次调用之前完全一样
c.add_fact("mile", "foot", 5280.0) # -> True，目前是新的一个连通分量
c.query("mile", "cm")              # -> None：还没有和 km/m/cm/inch 所在的分量连接起来
c.add_fact("foot", "inch", 12.0)   # -> True，把两个分量连接了起来
c.query("mile", "cm")              # -> 约为 160934.4，与上面 Part 1 的例子一致
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有两点值得先确认：`r` 的方向约定——事实 `(a, b, r)` 把边 `a -> b` 的权重定为 `r`、
`b -> a` 的权重定为 `1 / r`，绝不会反过来——以及每个部分对不连通或未知的单位返回什么：始终是
`None`，绝不抛出异常。

### Part 1

从 `src` 出发，用一次迭代式的深度优先搜索——用一个显式的列表当栈，这样再长的单位链也不会有触及
Python 递归深度上限的风险——沿途累乘边的权重，一旦到达 `dst` 就立刻停止。

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

建一次图的开销是 $O(F)$，其中 $F = $ `len(facts)`，因为每条事实贡献两条有向边。每个可达节点至多
入栈一次（`visited` 防止重复入栈），每个出栈节点的每条边也只会被检查一次，所以搜索本身的开销是
$O(V + E)$，其中 $V$ 是不同单位的数目，$E = 2F$；算上建图，`convert` 一次调用的开销就是
$O(V + E)$——不论 `src` 和 `dst` 相距多远，开销都随输入规模线性增长。

### Part 2

`first_contradiction` 逐条事实地生长同一种图，在加入第 `i` 条事实之前，会调用
`_path_factor`——和 Part 1 中完全一样——去查询已经加入的事实所隐含的 `a -> b` 系数。当 `a` 或 `b`
是新出现的单位，或者两者分属不同的连通分量时，这次调用会返回 `None`；不管是哪种情况，此时都没有
可比较的对象，于是直接加入这条事实。当它返回一个实际的系数时，说明 `a` 和 `b` 已经连通，于是用
`math.isclose` 把这条事实和该系数做比较；第一条没通过检查的事实会被立即返回，之后的事实完全不会
再看。

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

这 $F$ 条事实中的每一条至多触发一次在目前为止已建好的图上进行的 $O(V + E)$ 搜索，总共
$O(F \cdot (V + E))$：每一次矛盾检测的开销都不超过 Part 1 里一次查询的开销。对于只处理一遍的一批
事实来说，这已经够用了，但 Part 3 需要每一次 `add_fact` 和 `query` 调用都是近乎常数时间，不管之前
已经有多少条事实，这就排除了每次调用都重新搜索一遍的做法——需要的是一种增量式的数据结构：*带权
并查集*（weighted union-find）。

### Part 3

带权并查集把每个单位放进一棵有根树里，每个连通分量对应一棵树，用树的形状而不是显式的边来表示一条
事实。给每个单位 $u$ 一个父指针和一个权重 $w(u)$，满足

$$1\,u = w(u) \cdot \mathrm{parent}(u),$$

根节点的父指针指向自己，$w(\mathrm{root}) = 1$；沿父指针一路走到根、并把途中的权重连乘起来——和
Part 1 里“沿路径累乘”是同一条规则——就得到任意 $u$ 到其根节点的系数。`_find(x)` 递归地从 `x` 走到
它的根节点，在递归返回的过程中*压缩*（compress）路径：一旦递归调用确定了 `x` 目前的父节点 `p` 到
根节点的系数是 `w`，`x` 自己到根节点的系数就是 $w(x) \cdot w$——它到 `p` 的系数，乘以 `p` 到根
节点的系数——于是把 `x` 直接指向根节点，权重设为这个合成后的值；之后再从 `x` 出发查询，就只需要
一步，不必再走一遍原来的整条链。

`add_fact(a, b, r)` 首先确保 `a` 和 `b` 都存在（一个新单位一开始是自己的根，权重 `1.0`，大小
`1`），然后查找两者的根节点。记 $(R_a, w_a) = \mathrm{find}(a)$、$(R_b, w_b) = \mathrm{find}(b)$，
于是 $1\,a = w_a R_a$，$1\,b = w_b R_b$。分两种情况：

- 若 $R_a = R_b$，两者已经用同一个根表示过了，于是可以直接检验这条新事实：把
  $a \to R_a\ (=R_b) \to b$ 这段路径串起来，隐含的 $a \to b$ 系数是 $w_a \cdot (1 / w_b)$——第一段
  是 $a \to R_a$，也就是 $w_a$；第二段是 $R_b \to b$，是 $b \to R_b = w_b$ 的倒数——当
  $w_a / w_b$ 与 $r$ 接近时就接受这条事实；否则 `add_fact` 什么都不改变，返回 `False`（此时唯一
  可能已经发生的事情是 `_find` 自身的路径压缩，它只会让以后的查找更快，从不改变任何 `query` 的
  返回值，所以不论哪种结果都是无害的）。
- 若 $R_a \neq R_b$，两棵树需要合并：把 `_size` 较小的那个根，接到另一个根下面，即*按大小合并*
  （union by size），这样绝大多数合并都不会增加树的深度。先调整 `a`、`b` 的角色（如果方向反了，
  就交换两者并把 `r` 换成 `1 / r`），使得被接到 $R_a$ 下面的始终是 $R_b$；它的新权重必须满足
  $1\,R_b = w(R_b)\,R_a$，这可以通过把 $R_b \to b \to a \to R_a$ 这段路径串起来求得——第一段是
  $b \to R_b = w_b$ 的倒数，中间一段是把这条新事实反过来读（$b \to a = 1 / r$），最后一段是
  $a \to R_a = w_a$：

  $$w(R_b) \;=\; \frac{1}{w_b} \cdot \frac{1}{r} \cdot w_a \;=\; \frac{w_a}{r\,w_b}.$$

`query(src, dst)` 对两个共享同一个根的单位，用的正是同一套 $w_x / w_y$ 推理——包括 `src == dst`
的情形：只要这个单位是已知的，$w_{\mathrm{src}} = w_{\mathrm{dst}}$ 就会让比值自动等于 `1.0`，
完全不需要特判。

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

按大小合并能把每棵树的深度控制在 $O(\log n)$（$n$ 为单位数）以内：每当一个节点的深度增加一，它
一定属于刚才合并的两棵树中较小的那一棵，而合并后的树的大小是两者之和，至少是较小那棵的两倍——一个
大小的值在达到 $n$ 之前，翻倍的次数不会超过 $\log_2 n$ 次。所以即使完全不做路径压缩，`_find` 的
开销也是 $O(\log n)$；在按大小合并之上再加上路径压缩，是经典的组合（Tarjan 的结果），每次操作的
均摊开销是 $O(\alpha(n))$，也就是反阿克曼函数（inverse Ackermann function）——对于这个页面、乃至
任何实际输入可能达到的 $n$，它都小于 $5$，实质上就是常数。`add_fact` 和 `query` 各自只做两次
`_find` 调用和 $O(1)$ 的其他工作，所以两者都是近乎常数的均摊时间，而且都不会去扫描一条没有被
直接给出的事实。

### 追问

- **在对数空间中计算。** 上面每一个系数都是若干 `r` 值的乘积，或者乘积之比；沿路径把 $\log r$
  累加起来、最后只在末尾取一次指数，只会积累一次舍入误差，而不是每条边都积累一次，还能把 Part 3
  里的合并操作变成对数的加减法。代价是 `rel_tol` 的含义不再完全相同：对 $|\log f - \log r|$ 设
  一个固定的绝对容差，只有在容差很小时才接近于对 $f$ 相对 $r$ 设一个固定的*相对*容差，所以线性
  空间里调好的容差值需要重新推导，不能直接照搬。
- **仿射单位（affine units）。** 单纯的乘法关系表达不了摄氏度和华氏度之间的换算，
  $F = 1.8\,C + 32$；更一般的情形是 $x_b = r\,x_a + s$，用一对 $(r, s)$ 表示，而两个仿射映射
  复合之后仍然是仿射的，$(r_2, s_2) \circ (r_1, s_1) = (r_1 r_2,\ r_2 s_1 + s_2)$，所以 Part 3
  的并查集依然适用，只需要让每个节点保存到其根节点的一对 $(r, s)$，而不是单个系数，`_find` 的
  路径压缩也就按这条规则复合两个数对，而不是相乘两个数。
- **精确算术。** 把上面所有的 `float` 换成 `fractions.Fraction`——从每个 `r` 精确的十进制文本
  构造，而不是从已经四舍五入过的 `float` 构造——能让 Part 2 和 Part 3 里的每一次比较都变得精确，
  `rel_tol` 也就不再需要，两条事实要么完全一致，要么就不一致；代价是随着越来越多的事实被复合
  进来，`Fraction` 的分子和分母可以无限增长，链条越长，每次操作就越慢、占用的内存也越多，不像
  `float` 的大小是固定的。
- **解释一次矛盾。** `first_contradiction` 只报告第一条出问题的事实的下标，不说明它为什么错。
  要还原它所闭合的那个环，需要让 `_path_factor` 在返回系数的同时，也返回它从 `a` 走到 `b` 时
  经过的那些更早的事实；这条矛盾的事实，加上这条路径，正是调用者要修复输入时唯一需要查看的最小
  事实集合。

<details>
<summary>验证代码（可运行）</summary>

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
