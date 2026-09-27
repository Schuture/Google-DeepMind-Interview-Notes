# 双调数组与单调函数上的二分查找

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 二分查找 | ★★☆☆☆ | 中等 | RE · RS · SWE · MLE | binary-search, bitonic-array, exponential-search, monotone-functions, lower-bounds, bisection | 3 个部分 / 45 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

如果数组 `a` 的 `n >= 1` 个整数之间存在一个峰值下标 `p`（`0 <= p <= n - 1`），使得
`a[0] < a[1] < ... < a[p]` 且 `a[p] > a[p + 1] > ... > a[n - 1]`，就称这个数组是*双调的*（bitonic）；峰值
两侧都可以为空，所以严格递增的数组（`p = n - 1`）和严格递减的数组（`p = 0`）都算双调数组。

### Part 1 —— 双调列表的最小值与最大值

```py
def bitonic_min_max(a: list[int]) -> tuple[int, int]: ...
```

`a` 是双调的，定义如上。返回 `(min(a), max(a))`：最小值用 O(1) 时间求出，最大值——也就是峰值处的元
素——需要对 `a` 中的元素做 O(log n) 次比较。

```text
a = [1, 3, 8, 12, 9, 5, 2]
bitonic_min_max(a) == (1, 12)
```

### Part 2 —— 对正整数上的严格递增函数求逆

`f` 是一个从正整数到整数的*严格递增*（strictly increasing）函数：对任意正整数 `x, x'`，`x < x'` 都能推出
`f(x) < f(x')`。这里不假设自变量有任何上界，而且 `f` 的求值可能代价高昂，所以算
法的代价按它调用 `f` 的次数来衡量，而不是按结果之间的算术运算或比较次数。

```py
from collections.abc import Callable

def inverse(f: Callable[[int], int], y: int) -> int | None: ...
def inverse_floor(f: Callable[[int], int], y: int) -> int | None: ...
```

`inverse(f, y)` 返回满足 `f(x) == y` 的正整数 `x`；如果没有任何正整数映射到 `y`，则返回 `None`。
`inverse_floor(f, y)` 返回满足 `f(x) <= y` 的最大正整数 `x`；如果连 `f(1) > y` 都成立（也就是根本没有正
整数满足这个不等式），则返回 `None`。

```text
inverse(lambda x: x * x, 49) == 7
inverse(lambda x: x * x, 50) is None
inverse(lambda x: x + 5, 5) is None      # x 得等于 0，但 0 不是正整数
inverse(lambda x: x + 5, 6) == 1
inverse(lambda x: x - 100, 5) == 105
inverse_floor(lambda x: x * x, 50) == 7
```

### Part 3 —— 平台区间与实数域上的求根

**(a)** `a` 仍然有 `n >= 1` 个元素，但现在只满足*非严格*双调：存在峰值下标 `p`（`0 <= p <= n - 1`），使得
`a[0] <= a[1] <= ... <= a[p] >= a[p + 1] >= ... >= a[n - 1]`，也就是说两侧的斜坡上都可能出现相邻元素相
等的情况——一段*平台*（plateau）。

```py
def bitonic_max_with_plateaus(a: list[int]) -> int: ...
```

最小值仍然可以在 O(1) 时间内求出。证明：在最坏情况下，任何算法都不可能只读取少于 `n` 个元素
就找到最大值；并给出一个能做到 O(n) 的算法。

```text
a = [0, 0, 0, 1, 0, 0, 0, 0]
```

这个数组是非严格双调的，峰值下标为 3（`a[3] = 1`），所以 `bitonic_max_with_plateaus(a) == 1`；其余元素
都是 0。

**(b)** `g` 是实数域上的一个函数，它是*连续的*（continuous）——非正式地说，它的图像没有跳跃也没有空
隙，因此它会取到自己任意两个值之间的每一个值——并且在 `[lo, hi]` 上*严格递增*，满足
`g(lo) <= y <= g(hi)`。连续性保证至少存在一个根 `x*` 使得 `g(x*) == y`；严格单调性则保证它是唯一的。

```py
def bisect_root(g: Callable[[float], float], y: float, lo: float, hi: float, eps: float) -> float: ...
```

返回一个满足 `|x - x*| <= eps` 的值 `x`。推导这需要对 `g` 求值多少次（作为 `hi - lo` 与 `eps` 的函
数），并让这个过程即使在 `eps = 0` 时，在浮点运算下也能终止。由于 `x*` 未必是一个浮点数，`eps = 0` 一
般无法被精确满足；这时这个过程必须返回一个在 `g` 的浮点求值所能分辨的范围内尽可能接近 `x*` 的 `x`。

```text
g(x) = x ** 3, y = 30, lo = 0, hi = 4, eps = 0.001
# 真正的根是 x* = 30 ** (1 / 3) ≈ 3.10723
bisect_root(g, 30, 0, 4, 0.001) ≈ 3.10723   # 与 x* 相差不超过 0.001
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有两点值得先确认：眼前的数组或函数到底是严格单调，还是允许出现相等的情况——Part 1 和 Part 2 都
假定严格单调，Part 3(a) 则对数组放宽了这一点——以及，对 Part 2 而言，答案的量级是否
预先有界；答案是没有，所以像 `[0, y]` 这样固定范围的搜索并不总是安全，下面的推导会说明这一点。

### Part 1

每一轮迭代都用一次相邻元素之间的比较，把 `[lo, hi]`——保证包含峰值的下标范围——替换成它的其中一半。当
`lo < hi` 且 `mid = (lo + hi) // 2` 时，整数除法保证 `mid <= hi - 1`，所以 `a[mid + 1]` 一定是合法的读
取。`a[mid] < a[mid + 1]` 说明数组在 `mid` 处仍在上升，峰值就在 `mid + 1` 或更靠右的位置：
`lo = mid + 1`。否则 `a[mid] > a[mid + 1]`——不会出现相等，因为双调数组的每一侧都是*严格*单调
的——峰值就在 `mid` 处或其左侧：`hi = mid`。两个分支都能让峰值始终留在 `[lo, hi]` 内，并把 `hi - lo`
缩小到原来的大约一半，所以循环最多经过 $\lceil \log_2 n \rceil$ 次迭代（每次一次对 `a` 的比较）就会到
达 `lo == hi`。最小值完全不需要搜索：递增一侧的每个元素都大于 `a[0]`，递减一侧的每个元素都大于
`a[-1]`，所以整体的最小值就是两端中较小的那一个。

```python
def bitonic_min_max(a: list[int]) -> tuple[int, int]:
    n = len(a)
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < a[mid + 1]:   # NOTE: floor-rounded mid < hi, so a[mid + 1] exists; comparing backward
            lo = mid + 1          #       (a[mid - 1], lo = mid) can loop forever once hi == lo + 1
        else:
            hi = mid
    return min(a[0], a[-1]), a[lo]
```

把 `a[mid]` 和 `a[mid + 1]` 正向比较，正是这里安全的关键。如果保留同样向下取整的 `mid`，却改成反向比
较——`a[mid] > a[mid - 1]`，命中时把 `lo` 移到 `mid`，否则把 `hi` 移到 `mid - 1`——一旦 `hi == lo + 1`
就可能永远循环下去：这时整数除法会让 `mid == lo`，于是只要比较成立，把 `lo` 设为 `mid` 的分支就让 `lo`
原地不动，循环永远到不了 `lo == hi`。下面的验证代码会在一个具体的数组上演示这一点。

在示例 `[1, 3, 8, 12, 9, 5, 2]` 上，搜索过程如下：

```text
lo=0, hi=6: mid=3, a[3]=12 > a[4]=9  -> hi=3   （峰值在下标 3 处或其左侧）
lo=0, hi=3: mid=1, a[1]=3  < a[2]=8  -> lo=2   （峰值在下标 1 的右侧）
lo=2, hi=3: mid=2, a[2]=8  < a[3]=12 -> lo=3   （峰值在下标 2 的右侧）
lo=3, hi=3: lo == hi，停止           -> 峰值下标 3，a[3] = 12
```

### Part 2

让候选值 `hi` 从 `1` 开始不断翻倍——`hi = 1, 2, 4, ...`——只要 `f(hi) < y` 就继续翻倍，这样能找到一个必
定包含答案的区间，即使答案很大，用到的求值次数也很少——这就是*指数搜索*（exponential search，也叫
*倍增搜索*，galloping search）。记 $m$ 为 `inverse_floor` 必须返回的值（满足 $f(m) \le y$ 的最大正整
数；前面 `f(1) > y` 的判断已经排除了不存在这种整数的情形）。假设翻倍循环在停下之前一共翻倍了
$k \ge 1$ 次，也就是在 $1, 2, 4, \ldots, 2^k$ 处对 `f` 求了值，并因为 $f(2^k) \ge y$ 而停止——上一次检
查时 $f(2^{k-1}) < y$。由于 `f` 在整数上严格递增：$f(2^{k-1}) < y$ 意味着 $2^{k-1} \le m$（一个 `f` 值
小于 `y` 的整数，不可能超过 `f` 值不超过 `y` 的*最大*整数）；而每个 $x > 2^k$ 都有
$f(x) > f(2^k) \ge y$，所以 $m \le 2^k$。于是 $lo = 2^{k-1}$、$hi = 2^k$ 从两侧把 $m$ 夹在中间。差值
`hi - lo` 一开始是 $2^{k-1}$，此后每一轮都恰好减半——只要这个差值本身是 2 的幂，`mid - lo` 和
`hi - mid` 就都恰好等于差值的一半，而它也确实会一直保持是 2 的幂——再经过 $k - 1$ 次求值就会缩小到
$1$；这时 `lo` 和 `hi` 相邻，而 `f(hi)` 早已缓存下来的值（来自最后一次给它赋值的那次求值）不需要再调
用一次 `f` 就能在两者之间做出判断。翻倍阶段花费 $k + 1$ 次求值，收缩阶段再花费 $k - 1$ 次，总共 $2k$
次——而在被排除在外的 $k = 0$ 情形中（`f(1)` 本身就等于 `y`，翻倍循环根本不会执行），恰好是 $1$ 次。

```python
from collections.abc import Callable


def _locate(f: Callable[[int], int], y: int) -> tuple[int, int] | None:
    """Returns (m, f(m)) for m = the largest positive integer with f(m) <= y, or None if even
    f(1) > y. f is strictly increasing. Every value of f computed here is cached and reused, so
    no argument is ever evaluated twice."""
    lo, f_lo = 1, f(1)
    if f_lo > y:
        return None
    hi, f_hi = lo, f_lo
    while f_hi < y:                 # NOTE: doubles with no assumed bound on m -- unlike a [0, y] search,
        lo, f_lo = hi, f_hi         #       this never assumes f(x) >= x (see the discussion below)
        hi *= 2                     # NOTE: a fixed-width language must guard hi and f(hi) against
        f_hi = f(hi)                #       overflow here; Python integers grow as needed
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f_mid = f(mid)
        if f_mid <= y:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return (hi, f_hi) if f_hi <= y else (lo, f_lo)   # NOTE: f_hi is already cached -- no new call to f


def inverse_floor(f: Callable[[int], int], y: int) -> int | None:
    located = _locate(f, y)
    return located[0] if located is not None else None


def inverse(f: Callable[[int], int], y: int) -> int | None:
    located = _locate(f, y)
    if located is None:
        return None
    x, f_x = located
    return x if f_x == y else None
```

记 $b$ 为 `m.bit_length()`（$m$ 的二进制表示所用的位数，满足 $2^{b - 1} \le m < 2^b$），就能把上面的结
果变成一个只依赖于 $m$、不依赖未知量 $k$ 的界：$2^{k - 1} \le m < 2^b$ 给出 $k - 1 < b$，也就是
$k \le b$（两者都是整数），于是 $2k \le 2b$——两个函数都至多需要 $2b$ 次求值，而 $k = 0$ 的情形
（$1 \le 2b$，因为只要 $m \ge 1$ 就有 $b \ge 1$）同样满足这个界。`inverse` 除了 `inverse_floor` 本身的
搜索之外不需要再花任何代价：`_locate` 已经在它返回的那个值上计算过 `f`，这个结果会被直接复用而不是重
新计算，所以多出来的那一步只是一次比较，不是再调用一次 `f`。

一个看起来更简单的做法——跳过翻倍阶段，直接在固定范围 `[0, y]` 上做二分查找——只有在答案保证落在这个范
围内时才站得住脚，而这需要对任意 `x` 都有 `f(x) >= x`。一个严格递增、取整数值的 `f` 满足
`f(x) >= f(1) + (x - 1)`，因为每向上走一步至少增加 1，所以只要 `f(1) >= 1` 就有 `f(x) >= x`——上面的
`x * x` 和 `x + 5` 都满足这一点，但 `f(x) = x - 100` 不满足：`f(1) = -99 < 1`，
`inverse(lambda x: x - 100, 5)` 需要 `x = 105`，远远超出了 `[0, 5]`。指数搜索不需要任何这样的假设，因
为它会不断扩大自己的搜索区间，直到 `f` 自己报告这个区间已经够大。

### Part 3

**(a)** Part 1 中寻找峰值的比较依赖严格不等式：`a[mid] < a[mid + 1]` 和 `a[mid] > a[mid + 1]` 是仅有的
两种可能。允许平台之后，`a[mid] == a[mid + 1]` 现在也可能出现，而且它不提供任何信息——两个相邻元素都
处于*某个*合法的非严格双调数组的同一段平坦区域上，所以两侧都不能被排除。这不只是某一个具体算法的局
限：在最坏情况下，任何算法——不论是什么形状，是否自适应——都不可能在漏读 `a` 中哪怕一个元素的情况下找到
最大值。假设某个算法总能在最多读取 `n` 个元素中的 `n - 1` 个之后停下。让它面对这样一个对手：只要还能拖
下去，每次读取都回答 `0`；当算法停下时，必定有某个下标 `q` 从未被读取过。全零数组是（平凡地）非严格双
调的：`0 <= 0 <= ... <= 0 >= 0 >= ... >= 0`，最大值为 `0`；把同一个数组在 `q` 处改成 `1`，同样是非严格
双调的（在 `q` 之前非递减，在 `q` 之后非递增，因为改动一个元素不会破坏任何一条链），但它的最大值是
`1`。这个算法在这两个数组上的读取序列和最终答案完全相同——它从未读过 `q`，而其他每个下标在两个数组上读
到的都是 `0`——所以不管它返回什么值，都会在其中一个数组上出错。因此读完全部 `n` 个元素是必要的，而这也
显然足够：`max(a)` 就是一次 O(n) 的遍历。

```python
def bitonic_max_with_plateaus(a: list[int]) -> int:
    return max(a)   # NOTE: no shortcut is possible in the worst case -- see the adversary argument above
```

**(b)** 二分法不断把已知包含根的区间 `[lo, hi]` 减半，用 `g(mid) - y` 的符号来判断哪一半仍然包含
根：`g` 严格递增意味着 `g(mid) < y` 时根严格地在 `mid` 右侧，`g(mid) >= y` 时根在 `mid` 处或其左侧。经
过 `k` 次减半后，区间宽度变为 `(hi - lo) / 2 ** k`，所以
$k = \lceil \log_2((\mathrm{hi} - \mathrm{lo}) / \mathrm{eps}) \rceil$ 次求值就足以把它缩小到不超过
`eps`，此时两个端点中的任意一个都与真正的根相差不超过 `eps`。在浮点运算中，无休止地减半永远无法把区间
宽度缩小到恰好 `0`——两个相邻浮点数的中点会被舍入到其中一个——所以这个循环真正的终止条件不是比较宽度，
而是区间彻底无法再缩小；而仅仅因为任意 `lo` 与 `hi` 之间只有有限多个浮点数，每一种输入都会在有限步内到
达这一点。

```python
def bisect_root(g: Callable[[float], float], y: float, lo: float, hi: float, eps: float) -> float:
    while hi - lo > eps:
        mid = lo + (hi - lo) / 2
        if mid == lo or mid == hi:   # NOTE: the bracket cannot shrink further in floating point -- this
            break                    #       is what makes the loop terminate even when eps == 0
        if g(mid) < y:
            lo = mid
        else:
            hi = mid
    return lo
```

### 追问

- **在答案上做二分查找。** 只要某个真假性质随着候选值增大而恰好翻转一次——“这个 batch size 是否能放进
  内存”、“这个学习率下损失是否仍然有限”、“这个阈值下召回率（recall）是否仍达到目标”——直接对候选值本身做二
  分查找，就把对一个单调性质的搜索变成了 Part 2 的那种搜索（还没有已知上界时先倍增，再减半），用对该
  性质的一次求值（一次训练步、一次内存分配）代替对 `f` 的一次调用。
- **三分查找需要严格性。** 三分查找通过比较两个内部点、丢弃两者都认为方向不对的那一侧，来定位一个*单
  峰*（unimodal）函数的最大值；如果低于峰值的地方存在一段平坦区域，这两个点就可能打平，无法提供关于真正最大值在
  哪一侧的任何信息——这正是 Part 3(a) 证明过的、无法避免的同一种失败：三分查找和 Part 1 的峰值搜索一
  样，都需要两侧严格单调才能保证正确。
- **牛顿法与二分法的对比。** 二分法每一步都把区间减半，与 `g` 具体是什么无关，因此是线性收敛；牛顿法
  则跳到当前猜测点处切线与零的交点，只要猜测已经足够接近、且 `g` 在那里光滑、导数不为零，每一步大致能
  让正确的有效数字位数翻倍（二次收敛）——但当 `g` 在猜测点附近远非线性，或者导数接近零时，牛顿法可能完
  全跳出区间，或者反复循环而不收敛，这正是二分法“保证区间不断缩小”这一点更安全的地方。

<details>
<summary>验证代码（可运行）</summary>

```python
import math
import random

from scipy.optimize import brentq

# ================= Part 1 =================

a_ex = [1, 3, 8, 12, 9, 5, 2]
assert bitonic_min_max(a_ex) == (1, 12)

# small, explicit edge cases: n = 1, n = 2 both orders, peak at either end, even and odd lengths
assert bitonic_min_max([5]) == (5, 5)
assert bitonic_min_max([2, 9]) == (2, 9)          # n = 2, increasing (peak at the right end)
assert bitonic_min_max([9, 2]) == (2, 9)          # n = 2, decreasing (peak at the left end)
assert bitonic_min_max([1, 5, 8, 3]) == (1, 8)    # even length, peak in the middle
assert bitonic_min_max([1, 4, 9, 6, 2]) == (1, 9)  # odd length, peak in the middle


class _CountingList:
    """Wraps a list so element reads can be counted from outside, without changing
    bitonic_min_max's own signature or code."""

    def __init__(self, data):
        self._data = data
        self.reads = 0

    def __getitem__(self, i):
        self.reads += 1
        return self._data[i]

    def __len__(self):
        return len(self._data)


def _random_bitonic(rng, n):
    """A strictly bitonic list of length n with a uniformly random peak position (including
    both ends): built by walking down from the peak on each side with strictly positive steps,
    then reversing the increasing side."""
    p = rng.randrange(n)
    peak = rng.randint(0, 50)
    left, v = [], peak
    for _ in range(p):
        v -= rng.randint(1, 5)
        left.append(v)
    left.reverse()
    right, v = [], peak
    for _ in range(n - 1 - p):
        v -= rng.randint(1, 5)
        right.append(v)
    return left + [peak] + right


for seed in range(4000):
    rng = random.Random(seed)
    n = rng.randint(1, 50)
    a = _random_bitonic(rng, n)
    wrapped = _CountingList(a)
    got = bitonic_min_max(wrapped)
    assert got == (min(a), max(a)), (seed, a, got)
    bound = 2 * (math.ceil(math.log2(n)) if n > 1 else 0) + 3   # loop reads + the 3 fixed reads outside it
    assert wrapped.reads <= bound, (seed, a, wrapped.reads, bound)

# the unsafe mid - 1 variant named in the NOTE: looping-forever, guarded with an iteration cap


def _buggy_peak_index(a: list[int], max_iters: int = 1000) -> int:
    """Same floor-rounded mid as the solution, but compares a[mid] with a[mid - 1] instead of
    a[mid + 1]. Raises RuntimeError instead of hanging forever when it fails to converge."""
    lo, hi = 0, len(a) - 1
    for _ in range(max_iters):
        if lo == hi:
            return lo
        mid = (lo + hi) // 2
        if a[mid] > a[mid - 1]:
            lo = mid
        else:
            hi = mid - 1
    raise RuntimeError("did not converge")


try:
    _buggy_peak_index([1, 5, 3])
    raise AssertionError("expected the unsafe variant to fail to converge")
except RuntimeError:
    pass

# ================= Part 2 =================

assert inverse(lambda x: x * x, 49) == 7
assert inverse(lambda x: x * x, 50) is None
assert inverse(lambda x: x + 5, 5) is None
assert inverse(lambda x: x + 5, 6) == 1
assert inverse(lambda x: x - 100, 5) == 105
assert inverse_floor(lambda x: x * x, 50) == 7
assert inverse_floor(lambda x: x + 1000, 5) is None      # f(1) = 1001 > 5: no positive x qualifies
assert inverse(lambda x: x + 1000, 5) is None


class _CallCounter:
    def __init__(self, f):
        self._f = f
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self._f(x)


# --- inverse_floor(x * x, y) against math.isqrt, with the evaluation-count bound checked exactly ---
for y in range(1, 20_000):
    counted = _CallCounter(lambda x: x * x)
    got = inverse_floor(counted, y)
    want = math.isqrt(y)
    assert got == want, (y, got, want)
    assert counted.calls <= 2 * want.bit_length(), (y, counted.calls, want.bit_length())

# --- against an independent linear scan, on random strictly increasing functions built from random
# positive increments and a random offset that may be negative (like f(x) = x - 100) ---
def _random_increasing(rng):
    """f(1) is a random offset; each f(x + 1) - f(x) is a random step in 1..max_step, drawn from a
    private generator the first time any call needs it and fixed from then on."""
    max_step = rng.randint(1, 20)
    values = [rng.randint(-200, 200)]
    steps = random.Random(rng.randrange(2**32))

    def f(x):
        while len(values) < x:
            values.append(values[-1] + steps.randint(1, max_step))
        return values[x - 1]

    return f


rng = random.Random(0)
for _ in range(2_000):
    f = _random_increasing(rng)
    y = rng.randint(-300, 3_000)

    if f(1) > y:
        expected_m = None
    else:
        x = 1
        while f(x + 1) <= y:            # independent linear scan -- never calls _locate or its helpers
            x += 1
        expected_m = x

    counted = _CallCounter(f)
    got_floor = inverse_floor(counted, y)
    assert got_floor == expected_m, (y, got_floor, expected_m)
    if expected_m is None:
        assert counted.calls == 1        # only the initial f(1) > y guard
    else:
        assert counted.calls <= 2 * expected_m.bit_length(), (y, expected_m, counted.calls)

    expected_inverse = expected_m if (expected_m is not None and f(expected_m) == y) else None
    assert inverse(f, y) == expected_inverse, (y, expected_inverse)

# --- the [0, y]-bounded search is wrong once f can fall below its own argument ---


def _naive_inverse_floor_bounded(f, y):
    """Wrong beyond the case f(1) >= 1: assumes the answer lies in [0, y]."""
    if f(1) > y:
        return None
    lo, hi = 0, y
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if f(mid) <= y:
            lo = mid
        else:
            hi = mid - 1
    return lo if lo >= 1 else None


assert inverse_floor(lambda x: x - 100, 5) == 105
assert _naive_inverse_floor_bounded(lambda x: x - 100, 5) == 5    # wrong: the true answer, 105, is outside [0, 5]

# ================= Part 3 =================

# --- (a): the worked example, and Part 1's algorithm fooled by the very plateau it cannot see past ---
plateau = [0, 0, 0, 1, 0, 0, 0, 0]
assert bitonic_max_with_plateaus(plateau) == 1
_, wrong_max = bitonic_min_max(plateau)     # Part 1's peak search, misapplied to non-strict data
assert wrong_max != 1, "Part 1's algorithm was expected to be fooled by this plateau"


def _random_plateau_bitonic(rng, n):
    """A non-strictly bitonic list (adjacent ties allowed) with a known true peak value, built the
    same way as _random_bitonic but with steps that may be 0."""
    p = rng.randrange(n)
    peak = rng.randint(0, 20)
    left, v = [], peak
    for _ in range(p):
        v -= rng.randint(0, 3)
        left.append(v)
    left.reverse()
    right, v = [], peak
    for _ in range(n - 1 - p):
        v -= rng.randint(0, 3)
        right.append(v)
    return left + [peak] + right, peak


saw_plateau_failure = False
for seed in range(2_000):
    rng = random.Random(seed)
    n = rng.randint(1, 30)
    a, peak = _random_plateau_bitonic(rng, n)
    assert max(a) == peak                                  # sanity: the construction's own invariant
    assert bitonic_max_with_plateaus(a) == peak
    assert bitonic_min_max(a)[0] == min(a)                 # the minimum is unaffected by plateaus
    if bitonic_min_max(a)[1] != peak:
        saw_plateau_failure = True
assert saw_plateau_failure   # the failure is not an artefact of one hand-picked array

# --- (b): the worked example, the evaluation count against the derived formula, and eps = 0 ---
assert abs(bisect_root(lambda x: x ** 3, 30, 0.0, 4.0, 0.001) - 30 ** (1 / 3)) <= 0.001

true_root_500 = brentq(lambda x: x ** 3 - 500, 0, 10, xtol=1e-300, rtol=1e-15, maxiter=200)
for eps in (1.0, 0.1, 0.01, 0.001, 1e-6):
    counted = _CallCounter(lambda x: x ** 3)
    got = bisect_root(counted, 500, 0.0, 10.0, eps)
    expected_calls = math.ceil(math.log2(10.0 / eps))
    assert counted.calls == expected_calls, (eps, counted.calls, expected_calls)
    assert abs(got - true_root_500) <= eps, (eps, got, true_root_500)

for y in (1, 10, 100, 500, 999):
    got = bisect_root(lambda x: x ** 3, y, 0.0, 10.0, 0.0)     # eps = 0 -- must still terminate
    true_root = brentq(lambda x: x ** 3 - y, 0, 10, xtol=1e-300, rtol=1e-15, maxiter=200)
    tolerance = 2 * math.ulp(true_root)
    assert abs(got - true_root) <= tolerance, (y, got, true_root, tolerance)

print("all checks passed")
```

</details>

</details>
