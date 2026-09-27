# 用最少的笔画刷完栅栏

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 贪心与分治 | ★★★☆☆ | 中等 | SWE · MLE · Intern | greedy, divide-and-conquer, arrays, range-minimum, proof-of-optimality | 3 个部分 / 45 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

一条栅栏由 `n` 块宽度为 1 的木板并排组成，下标从 `0` 到 `n - 1`；第 `i` 块木板的高度是一个整数
`h[i] >= 0`。把栅栏看成一个个单位格子：格子 `(i, y)` 存在，当且仅当 `0 <= y < h[i]`，因此第 `i`
块木板贡献恰好 `h[i]` 个格子，从地面 `y = 0` 一直堆到 `y = h[i] - 1`。一次*水平笔画*（horizontal
stroke）刷的是某一行 `y` 上的格子 `(l, y), (l + 1, y), ..., (r, y)`（对某个 `l <= r`）；只有当这些
格子全部存在——也就是对从 `l` 到 `r` 的每一个 `i` 都有 `h[i] > y`——才允许这么刷，换句话说，水平
笔画永远不能跨过一块矮到够不着第 `y` 行的木板。一次*竖直笔画*（vertical stroke）会刷完一整块木板
的所有格子，即 `(i, 0), (i, 1), ..., (i, h[i] - 1)` 全部。同一个格子可以被刷不止一次——笔画之间
允许重叠——下面每个部分要求的都是把栅栏的每一个格子至少刷一遍所需要的最少笔画数，水平笔画和竖直
笔画合并计数。

### Part 1 —— 只用水平笔画

```py
def min_horizontal_strokes(h: list[int]) -> int: ...
```

只使用水平笔画，返回刷完栅栏每一个格子所需要的最少笔画数。`h` 可以为空（没有木板，无需刷任何
东西），也可以包含 `0`（高度为 `0` 的木板不贡献任何格子，也不需要刷）。

```text
h = [2, 1, 3, 3, 1, 2]

木板：          0  1  2  3  4  5
y=2（顶部）      .  .  #  #  .  .    一段 {2,3}              1 笔
y=1             #  .  #  #  .  #    三段 {0}、{2,3}、{5}     3 笔
y=0（地面）      #  #  #  #  #  #    一段 {0,1,2,3,4,5}       1 笔

min_horizontal_strokes(h) == 1 + 3 + 1 == 5
```

### Part 2 —— 木板高度可以改变

```py
class Fence:
    def __init__(self, h: list[int]): ...
    def set_height(self, i: int, new_height: int) -> None: ...   # O(1)
    def strokes(self) -> int: ...                                  # O(1): Part 1's answer for the current heights
```

`Fence(h)` 用和 Part 1 完全相同的高度列表构造；此后木板数 `n = len(h)` 永远不变。
`set_height(i, new_height)` 要求 `0 <= i < n` 且 `new_height >= 0`，把 `h[i]` 替换成
`new_height`，耗时 `O(1)`——不能重新扫描其余木板。`strokes()` 以 `O(1)` 时间返回
`min_horizontal_strokes` 对栅栏当前高度会给出的结果：和 Part 1 从头计算出的答案完全一样，只是
一直保持更新，而不是每次重新计算。

```text
f = Fence([1, 1, 1])
f.strokes()                # -> 1：三块木板高度都是 1，一笔横扫全部

f.set_height(1, 3)         # h 变为 [1, 3, 1]
f.strokes()                # -> 3：木板 1 如今比两侧邻居都高出一截

f.set_height(0, 3)         # h 变为 [3, 3, 1]
f.strokes()                # -> 3：总数不变，但内部的两个边界项都变了

f.set_height(1, 0)         # h 变为 [3, 0, 1]
f.strokes()                # -> 4：木板 1 降到高度 0，把第 0 行拆成了两段
```

### Part 3 —— 允许竖直笔画

```py
def min_strokes(h: list[int]) -> int: ...   # horizontal and vertical strokes, together
```

返回把栅栏每一个格子刷完所需要的最少笔画数——水平笔画和竖直笔画可以任意组合，规则与上文相同。

```text
min_strokes([1, 5, 1]) == 2
# 一次水平笔画横扫三块木板，刷完第 0 行——这是木板 0 和木板 2 唯一够得着的一行；
# 再在木板 1 上来一次竖直笔画，刷完它的第 1-4 行——比 min_horizontal_strokes([1, 5, 1])
# 需要的 5 笔好得多（第 1、2、3、4 行都各自只有木板 1 单独一段）

min_strokes([2, 1, 3, 3, 1, 2]) == 5
# 和相同高度下 min_horizontal_strokes 的结果一致（即 Part 1 的例子）——对这种形状来说，
# 不论怎么用竖直笔画，都不会比逐行刷更好
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有两点值得先确认：同一个格子可以被刷不止一次，笔画之间可以自由重叠，不会因此受罚；以及
每一笔——不论水平还是竖直——都必须完全落在存在的格子上——水平笔画永远不能跨过矮到够不着这一行的
木板，高度为 `0` 的木板永远不会参与任何一笔。

### Part 1

在第 `y` 行，存在的格子恰好是满足 `h[i] > y` 的那些木板，它们会分成若干个由连续木板组成的极大段；
每一段配一次水平笔画，既是必要的（不同段之间的空隙没有任何东西可以覆盖），也是充分的（一次笔画
就能横扫整段），所以答案就是从第 `0` 行到第 `max(h) - 1` 行、每一行段数之和。

换一种方式数段数：木板 `i` 在第 `y` 行开启新的一段，当且仅当木板 `i` 在第 `y` 行有格子、而它左边
的邻居没有——`h[i] > y` 且（`i == 0`，或者 `h[i - 1] <= y`），约定一个虚拟的 `h[-1] = 0`，这样
`i == 0` 的情形就不必单独处理。固定 `i`，满足这个条件的行是 `h[i - 1] <= y < h[i]`，是一个长度为
`max(0, h[i] - h[i - 1])` 的区间；由于每一段的开头恰好属于唯一的一块木板、唯一的一行，把这个长度
对每块木板求和，就恰好把每一段数了一次：
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

只需遍历 `h` 一遍：`O(n)` 时间，除了输入之外只用 `O(1)` 的额外空间。

### Part 2

`min_horizontal_strokes` 是每块木板各贡献一项的和，`term[i] = max(0, h[i] - h[i - 1])`（约定
`h[-1] = 0`）；改变单块木板的高度，只会影响提到它的两项：`term[i]` 本身（读取 `h[i]` 和
`h[i - 1]`），以及 `term[i + 1]`（读取 `h[i + 1]` 和 `h[i]`）——其余每一项既不读取 `h[i]` 的旧值，
也不读取新值，完全不受影响。在数组之外维护一个滚动的 `total`，以及逐块木板的这些项，每次调用只
更新至多这两项，就能把 `strokes()` 变成一次简单的查表。

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

`__init__` 花费 `O(n)`，和 Part 1 是同一次遍历。`set_height` 只读写固定数目的数组元素，与 `n`
无关，因此是 `O(1)`；`strokes()` 同样是 `O(1)`，只是一次属性读取。

### Part 3

一段连续木板 `[l, r]`，如果它在某个基准高度 `b` 上*孤立*（isolated）——段内每块木板都满足
`h[i] > b`，并且这一段两侧都有边界，边界或者是数组的端点，或者是一块高度 `<= b` 的木板——那么任何
刷到它在第 `y >= b` 行某个格子的笔画，要么是它自己某块木板上的竖直笔画，要么是完全落在 `[l, r]`
内部的水平笔画：第 `y >= b` 行的水平笔画不可能跨过这个边界，因为边界那块木板在这一行没有格子。
因此，把 `[l, r]` 在第 `>= b` 行的所有格子刷完所需要的最少笔画数——记作 `T(l, r, b)`——是一个自
成一体的子问题，而整体答案就是把 `T` 对满足 `h[i] > 0` 的每一个极大段求和（高度为 `0` 的木板永远
不参与任何笔画，并且在每一行都会截断一段）。

在这样一段区间之内，令 `m = min(h[l..r])`。有两种可用的策略：把每块木板都竖直刷一遍，花费
`r - l + 1`；或者把第 `b` 行到第 `m - 1` 行这 `m - b` 行，每一行都用一笔横扫整个 `[l, r]`——区间
内每块木板在这些行都有格子，因为 `h[i] >= m` 对整个区间成立——然后以基准 `m` 递归处理满足
`h[i] > m` 的极大子区间，这些子区间在基准 `m` 上同样是孤立的，理由与上面完全一致（被排除在外、
高度恰好为 `m` 的那些木板已经被完全刷完，会挡住任何第 `m` 行及以上的笔画穿过它们）。`T(l, r, b)`
不会超过这两种策略中较小的一个，因为二者都是合法的构造；剩下要证明的是它不会比这更小。取这一段
区间的一个最优解 `S`，把它拆成 `S_low`（在 `[b, m)` 中某一行上的水平笔画）和 `S_rest`（其余部分：
竖直笔画，以及第 `>= m` 行上的水平笔画）。`S_low` 里没有任何一笔碰得到第 `>= m` 行，所以第
`>= m` 行的每一个格子都要靠 `S_rest` 中的某一笔覆盖，而这样的每一笔都被限定在基准 `m` 之上的某个
子区间内——一次水平笔画不可能跨过一块高度恰好为 `m` 的木板（那里没有格子），一次竖直笔画只有在
它所在木板的高度超过 `m` 时才会碰到第 `>= m` 行——于是这些子区间只靠 `S_rest` 中的一部分笔画就
被覆盖了，因此 `S_rest` 至少有这些子区间的 `T` 值之和那么多笔。再看 `S_low`：如果 `[l, r]` 中
某块木板 `p` 没有属于自己的竖直笔画，那么格子 `(p, y)`——对第 `b` 到第 `m - 1` 行的每一行都存在，
因为 `h[p] >= m`——就只能靠该行的一次水平笔画覆盖，也就是 `S_low` 的一个成员，而不同的行需要
不同的笔画，因为一次水平笔画只属于单独一行；于是 `S_low` 至少有 `m - b` 笔，`S` 也就至少有
`m - b` 笔，加上基准 `m` 上各子区间的 `T` 值之和。否则，`[l, r]` 中每块木板都各有一次属于自己的
竖直笔画，直接给出 `S` 至少有 `r - l + 1` 笔——每块木板恰好对应一笔各不相同的笔画。不论哪种情形，
`S` 的笔画数都至少达到两种构造之一的花费，因而至少达到二者中较小的那个。记 `(l_k, r_k)` 为
`[l, r]` 中满足 `h_i > m` 的极大子区间：

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

访问一段区间所花的时间与它的长度成正比，用来求它的最小值和它的子区间；整个递归树总共至多有 `n`
个节点，因为每次访问都会永久移除至少一块木板（并列取得最小值的那些木板，它们不会再出现），不过
不同节点所涉及的区间范围，从这一层到下一层仍然可能大量重叠。一个严格有序的数组（高度一路递增或
者一路递减）是最坏情形：最小值总在某一端，于是每一层恰好削掉一块木板，得到大小依次为
`n, n - 1, ..., 1` 的 `n` 层，总工作量 `O(n^2)`——递归深度同样是 `O(n)`，对于一个稍微超过一千
块木板的有序数组，这早已超出 Python 默认的递归深度上限 `1000`。这正是为什么 `_solve_range` 用一个
显式的栈来模拟递归，而不是直接调用自身：每一段区间只会被压栈一次，用来算出它的最小值和子区间，
等到这些子区间的答案都已经写进 `answer` 之后，它才会被再次弹出，到那时才能取它自己的
`min(vertical, horizontal)`。

### 追问

- **不仅算出笔画数，还原出笔画本身。** `_solve_range` 在每一段区间上，本来就要在“整段竖直刷”与
  “刷 `m - b` 行整宽横条再递归”之间做出选择；在同一次后序遍历中把这个选择（以及竖直情形下具体是
  哪些木板）记录下来，再沿着被选中的分支自顶向下走一遍，就能还原出一份达到最优值的具体笔画列表，
  且不改变笔画数本身。
- **最坏情形 `O(n log n)`，借助区间最小值结构。** 把并列取得当前最小值的每一块木板一次性全部排除
  出去，并不是正确性所必需的——只在其中任意一个位置上切分，用一个 `O(n log n)` 预处理、`O(1)`
  查询的稀疏表（sparse table）找到这个位置，得到的最优值完全相同：恰好落在新基准上的木板会立刻
  变成一个代价为零的单独区间，同时每个区间至少还是会缩小一块木板。这样递归树就只有 `O(n)` 个
  节点，总计算量是 `O(n log n)`，与高度的排列顺序无关。
- **同一个公式，换一个问题。** 从全零数组出发，每次操作把某个连续区间里的所有元素都加 `1`，要用
  最少的操作次数得到目标数组 `h`，答案恰好就是 Part 1 那个封闭形式：在高度 `y` 上的一“层”区间
  加一，正对应第 `y` 行的一次水平笔画，用的是完全相同的数段论证。
- **刷一个二维网格。** 如果换成一个二维格子的网格，笔画是覆盖“存在”格子的整行或整列，就不再有
  单一的高度剖面可以递归了：一次行笔画和一次列笔画各自覆盖哪些格子，在整个网格范围内相互牵连，
  这时问题就变成了限定在行、列上的最小集合覆盖（minimum set cover）的一个实例，这一类问题目前
  没有已知的高效精确算法。

<details>
<summary>验证代码（可运行）</summary>

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
