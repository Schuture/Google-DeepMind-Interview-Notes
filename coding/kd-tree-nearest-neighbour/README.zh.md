# k-d 树：最近邻与范围查询

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 树与空间搜索 | ★★★☆☆ | 困难 | RS · RE · SWE · MLE | kd-tree, nearest-neighbour, pruning, heap, range-search, curse-of-dimensionality | 3 个部分 / 45–60 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

点是一个 NumPy 数组 `P`（形状为 `(n, d)`，`n >= 1`，浮点坐标）的各行；每个点由它在 `P` 中的行索引唯一标识。
两点 $p, q \in \mathbb{R}^d$ 之间的距离是欧几里得距离（Euclidean distance），
$\lVert p - q \rVert_2 = \sqrt{\sum_{i=0}^{d-1} (p_i - q_i)^2}$。`P` 上的一棵*k-d 树*（k-d tree）是一棵二叉
树，其中每个节点存储一个点的索引、一个*切分轴*（split axis，取值为 $0, \dots, d-1$ 的整数），以及隐含的、
该点在这个轴上的坐标——作为*切分值*（split value）；根节点在轴 $0$ 上切分，此后每往下一层就换到下一个
轴，因此深度为 $t$ 的节点（根节点深度为 $0$）在轴 $t \bmod d$ 上切分。

### Part 1 —— 构建一棵平衡树

```py
class KDTree:
    def __init__(self, P: np.ndarray) -> None: ...
```

`P` 的形状为 `(n, d)`；否则构造函数抛出 `ValueError`，`n == 0` 也算在内。递归地构建这棵树：在深度 $t$
收到 $m \ge 1$ 个点的节点（根节点在深度 $0$ 收到全部 $n$ 个点）在轴 $a = t \bmod d$ 上切分。把这 $m$ 个点
按 `(该点在轴 a 上的坐标, 索引)` 升序排列；排在位置 `m // 2` 的点成为这个节点。排在它之前的 `m // 2` 个
点，保持原有顺序，递归地成为深度 $t+1$ 的左子树；排在它之后的点递归地成为深度 $t+1$ 的右子树。哪一侧
没有分到点，哪一侧就没有子节点。

对任意 `P`，构建出的树的高度——最长的根到叶节点路径上的节点数——至多为 $\lceil \log_2(n+1) \rceil$，并且
构造函数必须在 $O(n \log^2 n)$ 时间或更优的时间内完成。

```text
P = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]     # 索引 0 .. 5，d = 2

                    5 (7,2)              轴 x，深度 0
                   /        \
             1 (5,4)          2 (9,6)    轴 y，深度 1
             /     \          /
       0 (2,3)   3 (4,7)  4 (8,1)        轴 x，深度 2
```

点 $5$（$x=7$）是全部六个点在轴 $x$ 上的中位数。在左边三个点 $\{0,1,3\}$（$y=3,4,7$）中，按 $y$ 排序是
$0,1,3$，所以点 $1$（$y=4$，居中的那个）成为这棵子树的节点，$0$ 和 $3$ 是它的两个叶节点。在右边两个点
$\{2,4\}$（$y=6,1$）中，按 $y$ 排序是 $4,2$，所以点 $2$（$y=6$）成为这棵子树的节点，$4$ 是它唯一的左叶
节点。高度为 $3$，恰好等于 $\lceil \log_2(6+1) \rceil = 3$。

### Part 2 —— 最近邻

```py
    def nearest(self, q: np.ndarray) -> int: ...   # index of the nearest point; ties -> smallest index
```

`q` 的形状为 `(d,)`；否则抛出 `ValueError`。返回 `P` 中按欧几里得距离离 `q` 最近的点的索引；如果有两个或
更多点并列最近，返回其中索引最小的那个。`nearest` 必须通过搜索这棵树来完成，而不是逐一扫描每个点：在每
个节点上，先递归进入 `q` 所在切分一侧的子节点；只有当另一侧仍可能存有一个点、其距离不差于目前搜索中已
找到的最优点时，才递归进入另一侧——否则就跳过那一侧的子节点及其下的一切，完全不去访问它。

```text
q = (4, 2)

到每个点的距离²：0 -> 5   1 -> 5   2 -> 41   3 -> 25   4 -> 17   5 -> 9
                              ^ 点 0 和点 1 并列最近

访问根节点，点 5 (7,2)，d² = 9                          目前最优：点 5（d² = 9）
  在轴 x 上以 7 切分；q_x = 4 在较近（左）一侧 -> 先搜索这一侧
    访问点 1 (5,4)，d² = 5                              更优：目前最优：点 1（d² = 5）
    在轴 y 上以 4 切分；q_y = 2 在较近（左）一侧 -> 先搜索这一侧
      访问点 0 (2,3)，d² = 5                             与点 1 并列；索引 0 < 1 -> 目前最优：点 0
      另一侧（点 3）：(2 - 4)² = 4 <= 5 -> 仍可能并列或更优 -> 必须访问
      访问点 3 (4,7)，d² = 25                             没有改善
  另一侧（点 2、点 4）：(4 - 7)² = 9 > 5 -> 不可能并列或更优 -> 整体跳过

nearest(np.array([4.0, 2.0])) == 0
```

### Part 3 —— k 近邻与矩形计数

```py
    def knn(self, q: np.ndarray, k: int) -> list[int]: ...   # k indices sorted by (distance, index); k <= n
    def count_in_box(self, lo: np.ndarray, hi: np.ndarray) -> int: ...   # points with lo <= p <= hi coordinate-wise
```

`knn` 按升序返回 `(距离², 索引)` 这一序对最小的 `k` 个索引——与 `nearest` 打破并列时用的顺序相同，只是
扩展成了一个长度为 `k` 的列表。`1 <= k <= n`；否则抛出 `ValueError`。`count_in_box` 返回满足以下条件的点
`p` 的数量：对每个轴 `i` 都有 `lo[i] <= p[i] <= hi[i]`（这是一个闭矩形，恰好落在边界上的点也算在内）；
`lo` 和 `hi` 的形状均为 `(d,)`，否则抛出 `ValueError`。如果在某个轴 `i` 上 `lo[i] > hi[i]`，这个矩形为
空，计数为 `0`。

延续 Part 2 中的 `q = (4, 2)`，它到点 $0,\dots,5$ 的平方距离依次是 $5,5,41,25,17,9$：按 `(距离², 索引)`
排序，顺序是 $0,1,5,4,3,2$，所以 `knn(q, 3) == [0, 1, 5]`。

取 `lo = (4, 1)`、`hi = (8, 5)`：点 $1$ $(5,4)$、点 $4$ $(8,1)$ 和点 $5$ $(7,2)$ 满足 $4 \le x \le 8$ 且
$1 \le y \le 5$（点 $4$ 恰好落在这个矩形的两条边上，仍然计入）；点 $0$ 的 $x=2<4$，点 $2$ 的 $x=9>8$，点
$3$ 的 $y=7>5$。`count_in_box(np.array([4.0, 1.0]), np.array([8.0, 5.0])) == 3`。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手写代码之前，有三点值得先确认：是否假设 $d$ 较低（Part 2 的剪枝论证和 Part 3 的范围查询上界都依赖这
一点；一旦 $d$ 超过几十，暴力法就会反超这棵树，详见追问）；`P` 中是否可能存在重复
的点，即两行坐标完全相同（会——没有什么禁止这一点，构建时按索引打破并列，加上搜索时 `(距离, 索引)` 的
排序方式，两者都无需任何特殊处理就能应对）；以及并列规则本身，索引更小者胜出，正是这一点让构建顺序、
`nearest` 和 `knn` 都成为定义明确的唯一答案，而不是“恰好满足条件的任意一点”。

### Part 1

构建过程是对上面规则的直接递归；真正需要设计的只有如何存储结果。这里没有用一条由 Python 节点对象组成的
链，而是把整棵树存放在四个按节点编号索引的扁平 NumPy 数组里——`point`、`axis`、`left`、`right`——这样之
后搜索时的每一步都只是几次数组索引操作，而不是沿着一串对象做属性查找。构造函数按节点创建的先后顺序分
配编号（根节点得到编号 $0$），这只是为了方便，编号本身没有别的含义；后面的代码都不依赖具体编号，只依赖
$-1$ 表示“没有子节点”这一点。

递归深度与树本身的高度一致，而高度是有界的（下面推导），所以普通递归不会有触及 Python 递归深度上限的
风险——这里不需要显式栈，不同于一个可能构建出任意不平衡的树的过程。

**高度。** 在一个节点的 $m$ 个点中，左子树恰好分到 $\lfloor m/2 \rfloor$ 个，右子树分到剩下的
$m - \lfloor m/2 \rfloor - 1$ 个，这个数目也至多是 $\lfloor m/2 \rfloor$（因为还有一个点成为了节点本
身）——所以两个子树都至多持有 $\lfloor m/2 \rfloor$ 个点，是一次严格的减半。把这一结论逐层应用，根节点
往下第 $t$ 层的一个节点至多持有 $\lfloor n/2^t \rfloor$ 个点，一旦 $2^t > n$ 这个数目就首次降到
$0$——满足这一条件的最小 $t$ 恰好就是 $n$ 的位长度（bit length），$\lceil \log_2(n+1) \rceil$。所以无论
坐标如何、点以什么顺序到来，没有叶节点会比这更深。

**构建时间。** 对一个节点的 $m$ 个点按 `(坐标, 索引)` 排序，代价是 $O(m \log m)$。同一深度上的各个节点
把全部 $n$ 个点互不重叠地分完，所以仅这一层的排序总代价至多是 $O(n \log n)$；而深度至多有
$\lceil \log_2(n+1) \rceil$ 层（就是刚推出的高度上界），总代价便是 $O(n \log^2 n)$。

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

先搜索较近一侧，是剪枝（pruning）得以成立的前提：这样往往能较早找到一个足够近的点，为之后的剪枝判断提供
一个较小的搜索半径。判断本身需要一个下界，用来衡量切分另一侧*任意*一点离 `q` 可能有多近。Part 1 的构建
方式保证了：在轴 $a$、以值 $s$ 切分的一个节点上，左子树里的每个点在轴 $a$ 上的坐标都 $\le s$，右子树里
的每个点坐标都 $\ge s$（两侧都是围绕同一个切分位置排序的）。所以，如果 `q` 自己在这个轴上的坐标 $q_a$
位于左侧（$q_a \le s$），右侧的每个点 $p$ 都满足 $p_a - q_a \ge s - q_a \ge 0$，于是

$$\lVert p-q \rVert_2^2 = \sum_i (p_i-q_i)^2 \ge (p_a-q_a)^2 \ge (s-q_a)^2,$$

因为每一项都非负，而 $(s-q_a)^2$ 正是其中一项（$q_a \ge s$ 时是对称的论证，约束的是左侧）。记
$d_{\text{best}}^2$ 为目前搜索中在任何位置找到的最优平方距离。如果 $(q_a-s)^2 > d_{\text{best}}^2$，另一
侧的任何点甚至都不可能与 $d_{\text{best}}$ 打平，整棵子树都可以跳过。但并列规则比较的是
`(距离², 索引)`，而不仅仅是距离：另一侧一个仅仅*打平* $d_{\text{best}}$ 的点，只要索引更小就仍然获胜，
所以只要另一侧仍可能打平——而不只是严格更优——就仍须访问，这恰好就是边界情形
$(q_a-s)^2 = d_{\text{best}}^2$。因此剪枝判断用的是 `<=` 而不是 `<`：

$$\text{访问另一侧} \iff (q_a-s)^2 \le d_{\text{best}}^2.$$

全程比较平方距离，避免了每访问一个节点就要算一次 `sqrt`，因为这不会改变任何一次比较的结果（上面比较的
每个量都非负）。

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

每个被访问的节点除两次递归调用外只做 $O(1)$ 的工作，所以总时间是 $O(\text{访问的节点数})$。始终沿较近
一侧下探、直接走到离 `q` 最近的叶节点，代价恰好是树的高度，$O(\log n)$；此外还会*额外*访问多少节点，
取决于另一侧的剪枝判断有多频繁地在不必要时也通过了。对于在 $d$ 维中、体积为 $V$ 的区域内大致均匀分布的
$n$ 个点，平均每个点占据大约 $V/n$ 的体积，所以一个点到其最近邻的典型距离量级是 $(V/n)^{1/d}$——也就是
这样体积的一个球的半径。一旦 $d_{\text{best}}$ 已经缩小到这个量级，一个兄弟子树只有当它的区域落在离
`q` 同样小的半径之内时才会被访问，而对于分布均匀、$d$ 较低且固定的数据，下探过程中遇到的兄弟子树里只有
一小部分满足这一点——所以总的访问节点数会接近有保证的 $O(\log n)$ 下探本身，这一点由下面的验证代码直接
测量，而不是断言。最坏情况下剪枝可能完全失效：如果 `P` 中所有点都在同一个位置，那么每个节点的切分值都
等于每个点在该轴上的坐标，于是 $(q_a-s)^2$ 在每个节点上都是同一个固定的数——并且，一旦有任何一个点被
比较过，这个数已经是构成 $d_{\text{best}}^2$ 那个和式中的一项，因此永远不可能超过 $d_{\text{best}}^2$。
这样另一侧就总会被访问，`nearest` 会访问全部 $n$ 个节点。

### Part 3

`knn` 把同样的思路从一个最优候选推广到 $k$ 个最优候选。一个大小上限为 $k$ 的最大堆（max-heap）以
`(-距离², -索引)` 序对的形式保存候选，这样目前 $k$ 个候选中*最差*的那个——也就是新的候选想要占有一席之
地就必须击败的那个——正好在 `heap[0]`：把两个字段都取负，就把“`(距离², 索引)` 最小”变成了“取负后的
序对最大”，而这正是一个最小堆（Python 的 `heapq`）在把它读成取负后数值最小的那个时会一直放在堆顶的东
西。当堆里还不满 $k$ 个候选时，一个节点的点直接压入堆中；此后则只有严格更优——`(距离², 索引)` 更小，
也就是取负后的序对更大——才会替换 `heap[0]`。剪枝判断是 Part 2 的直接推广，只是把 $d_{\text{best}}^2$
换成了目前堆顶（最差）候选的距离：在凑满 $k$ 个候选之前什么都不能剪掉（任何一个节点都可能是凑够 $k$ 个
所必需的），凑满之后，另一侧当且仅当 $(q_a-s)^2 \le$ 这个最差距离时才被访问，用的是和 Part 2 完全相同的
下界论证。

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

每个被访问的节点为其堆的压入或替换操作花费 $O(\log k)$，相比之下 `nearest` 维护单个当前最优只需
$O(1)$；同样的下探加剪枝论证在这里换成了要凑够 $k$ 个候选而不是 $1$ 个，所以对分布均匀的低维数据而言，
访问的节点数仍接近 $O(k + \log n)$，总时间 $O((k+\log n)\log k)$，而与 Part 2 相同的“所有点重合”构
造，会把最坏情况逼到 $O(n \log k)$。

`count_in_box` 一旦 `P` 很大就负担不起逐点检查：同一套递归转而携带查询矩形与每个节点自身区域的交
集——一个隐式的边界框，随着下探按切分值不断收窄——并在该区域已经可以确定归属的那一刻就提前返回。每个
节点还需要知道自己子树的大小，这在 Part 1 的构建完成之后只算一次即可，做法是对已经建好的 `left`/
`right` 数组做一趟朴素的后序遍历（post-order traversal，不需要改动 `_build` 本身），并缓存下来复用：
如果每次调用都重新计算，每次都要花 $O(n)$，会把下面推出的界完全抵消掉。

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

对固定的低维 $d$，一个节点的区域要么整个落在矩形内部（在上面用 $O(1)$ 解决），要么整个落在外部（同样
$O(1)$），要么被矩形的 $2d$ 个边界超平面（bounding hyperplane）之一——`lo` 和 `hi` 在每个轴上各贡献一
个——穿过，需要进一步检查。跟踪其中一个超平面，它在轴 $j$ 上取常数：在轴 $j$ 上切分，最多只会把它送入
一个子节点（切分值落在这个超平面的某一侧）；但在其他任何轴上切分，都会把它同时送入两个子节点，因为那
次切分根本不涉及坐标 $j$。轴每 $d$ 层循环一轮，所以在每一个由 $d$ 个连续层组成的区块里，这一个超平面
仍然穿过的节点数最多被乘以 $2^{d-1}$（在切分其他轴的 $d-1$ 层里翻倍，在切分轴 $j$ 的那一层不变），而
每个存活节点所持有的点数则缩小 $2^d$ 倍。经过 $t$ 个这样的区块后，仍被穿过的节点至多有 $2^{t(d-1)}$
个，各自持有大约 $n/2^{td}$ 个点；一旦这个数目降到 $O(1)$ 就不再有用了，这发生在 $t \approx (\log_2
n)/d$ 处，对这一个超平面给出 $2^{t(d-1)} = n^{1-1/d}$ 个被穿过的节点，对全部 $2d$ 个超平面求和（$d$
视为常数）就是 $O(n^{1-1/d})$。一次需要列出每个匹配点的查询，还要为遍历它找到的、完全落在矩形内的子
树逐点付出额外的 $O(m)$；`count_in_box` 从不这样做，它是用 $O(1)$ 直接加上一个已经存好的大小——所以
它的代价是 $O(n^{1-1/d})$，完全没有 $+m$ 这一项。

### 追问

- **均匀网格，按环搜索。** 选定格子边长 $s$，把每个点映射到整数格坐标 $\lfloor p_i / s \rfloor$（每个轴 $i$
  分别取整），并用一个字典记录每个格子里有哪些点；对于落在格子 $c$ 中的查询 $q$，按环 $r = 0, 1, 2, \dots$
  依次扫描，其中环 $r$ 是与 $c$ 的切比雪夫距离（Chebyshev distance，即各轴上索引差的最大值）恰好等于 $r$
  的全部格子。扫描完环 $r$ 之后，任何尚未被扫描到的点必定落在与 $c$ 的切比雪夫距离至少为 $r+1$ 的某个格
  子里，于是在体现这一差距的那个轴上，它的格子与 $c$ 的格子至少相差 $r+1$ 步；又因为 $q$ 本身落在格子 $c$
  内，这就使得该点在这一个轴上到 $q$ 的距离已经超过 $r \cdot s$——从而完整的欧几里得距离也必然超过
  $r \cdot s$——所以只要扫描完环 $r$ 之后已经找到的最优距离不超过 $r \cdot s$，尚未扫描的部分就不可能比
  它更优，搜索即可停止。在三维空间中，取 $s \approx (V/n)^{1/3}$（即平均每个格子约有一个点，适用于 $n$
  个点均匀分布在体积为 $V$ 的区域中的情形），一次查询平均只需扫描 $O(1)$ 个格子；但在聚集分布的数据上，
  大多数格子要么空着要么过于拥挤，网格方法便会退化，这正是 k-d 树或者八叉树（octree）取胜之处：八叉树
  把一个立方体切分成八个大小相等的子立方体，直到每个叶子至多容纳几个点为止，其最近邻搜索与 Part 2 中同
  样的分支限界（branch-and-bound）方法完全一致，只是用立方体取代了半空间（half-space），它像 k-d 树一样
  能适应数据疏密的变化，但切分点取在立方体中心而非中位数上，因此对任意输入而言，它的深度并不像 k-d 树
  那样有 $O(\log n)$ 的保证。
- **只有一次查询。** 如果只有一次查询，且不允许做任何预处理，那么 $O(n)$ 的线性扫描就是最优的——每个点都
  必须被检查一遍，因为任何未被检查的点都可能就是最近的——用 NumPy 向量化后这一扫描在实践中很快，
  `np.argpartition` 还能在 $O(n)$ 时间内直接给出最近的 $k$ 个点，而不需要完整排序。构建这棵树的代价是
  $O(n \log^2 n)$（Part 1），只有当足够多的查询分摊这个代价之后才划算。
- **为什么剪枝的效果随 $d$ 增大而减弱。** 轴每 $d$ 层循环一轮，所以在树的整个 $O(\log n)$ 高度里，每
  个轴只被切分大约 $(\log_2 n)/d$ 次——在 $n$ 固定的情况下，随着 $d$ 增大，一次切分留下的余量会趋向
  整个定义域的大小，而当前最优半径却仍然要缩小到典型的点间距 $(V/n)^{1/d}$，它也按同样的方式增长。
  一旦二者可以相提并论，剪枝判断就很少能失败，搜索球会和它遇到的几乎每一个格子相交，大部分树最终还是
  会被访问到；一次性算出全部 $n$ 个距离的向量化遍历，通常在 $d \approx 10\text{–}20$ 左右就会反超这
  棵树。
- **近似最近邻。** 在访问过固定数量 $t$ 个叶节点后就停止搜索，或者把剪枝判断放宽为
  $(q_a-s)^2 \le (1+\epsilon)^2 d_{\text{best}}^2$（某个 $\epsilon>0$），两者都是用一个小的、可控的
  “漏掉真正最近点”的概率（即便漏掉，也不会偏离超过 $1+\epsilon$ 倍，或者不会超出访问预算），换取远为
  激进的剪枝。
- **球树与基于图的索引。** 球树（ball tree）用嵌套的超球代替坐标轴对齐的切分，保留了同样的
  $O(\log n)$ 高度、按下界剪枝的结构，同时在更高维度下退化得更缓和，因为球本身不偏向任何一个坐标轴；
  像 HNSW 这样基于图的索引则完全抛开树结构，构建一张可供导航的近邻图，用贪心的最优优先游走来回答查
  询，这正是生产环境中的向量检索系统在 $d$ 达到学习表示（learned representation）常见的成百上千维时
  所采用的方式。
- **动态插入。** 把一个新点直接拼接进已有的某个叶节点，可能把树的高度推过 Part 1 推出的上界，因为单
  次插入并不会维持中位数切分这一不变量；标准的解决办法是只重建同时包含新点应处位置、且加入新点后仍保
  持平衡的最小子树，或者更简单地，把插入操作攒成一批，累积到一定数量后再整体重建这棵树，用一点数据陈
  旧换取避免每次插入都重建一次。

<details>
<summary>验证代码（可运行）</summary>

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
