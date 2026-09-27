# Python 生成器、单元测试与加权采样

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · Python 生成器、测试与随机采样 | ★★★☆☆ | 中等 | MLE · SWE · RE · Intern | generators, unit-testing, weighted-sampling, prefix-sums, binary-search, alias-method, chi-square-test | 3 个部分 / 45 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

### Part 1 —— 覆盖边界情况的惰性生成器及其测试

```py
from collections.abc import Iterable, Iterator

def iter_ints(items: Iterable[object] | None) -> Iterator[int]: ...
```

编写 `iter_ints`，一个按顺序产出 `items` 中所有整数的生成器函数（generator function）。以下规则完全
决定了它的输出：

- 它*惰性*（lazily）地消费 `items`：每产出 `k` 个值之后，它对 `items` 的读取只会推进到产出第 `k`
  个值的那个元素为止，加上在此之前遇到的任意 `None` 元素，绝不会更远。因此它必须能在任意可迭代对象
  上工作，包括像 `itertools.count()` 这样永不结束的无穷迭代器。
- `items is None` 的行为等同于空输入：不产出任何值，且 `items` 完全不会被访问。
- `None` 元素会被*跳过*（skipped）：它不贡献任何输出，也不算作错误。其余每个元素都必须是
  `int` 的实例（instance）；`bool` 会被拒绝，即使 `bool` 在 Python 中是 `int` 的子类，像 `3.0` 这样的浮点数、`"3"`
  这样的字符串也会被拒绝，哪怕它们表示的是同一个数。遇到第一个这样的非法元素时，`iter_ints` 会抛出
  `TypeError`，其消息会指出该元素的位置——它在 `items` 中从 0 开始计数的下标——抛出的时机正是生成
  器到达该元素的那一刻：这之前每一个合法的值都已经被产出过了，之后位置上的任何元素都完全不会被
  看到。
- `items` 本身永远不会被修改。

带追踪的示例：

```text
list(iter_ints([3, None, 0, -2]))   # -> [3, 0, -2]      (None 被跳过，0 和 -2 都是普通的值)

g = iter_ints([1, 2, True])
next(g)   # -> 1
next(g)   # -> 2
next(g)   # 抛出 TypeError，消息中提到下标 2
```

接着写一个 `unittest.TestCase` 来测试 `iter_ints`，至少覆盖：空列表和 `items is None`；`None` 元素
分别出现在开头、中间、末尾，以及输入完全由 `None` 组成的情形；`0` 和负整数——它们是普通的值，而不是
缺失的标记；生成器是惰性的——对一个*有限*、且被改造得能记录自己已经被拉取过多少个元素的迭代器调用
一次 `next()`，只会从中拉取恰好一个元素（测试中不能出现无穷迭代器：一个非惰性的错误实现在那种输入
上只会卡住，而不是干净地失败）；一个非法元素出现在一个或多个合法元素之后，验证合法的那些会先被产
出，随后抛出的 `TypeError` 指出了正确的下标；`bool`、`float`、`str` 各自作为元素类型都被拒绝；
`iter_ints` 被完整消费之后，传入的列表本身没有变化；以及对同一个列表分别调用两次 `iter_ints`，会得
到两个互不影响的生成器。

### Part 2 —— 作为无穷生成器的加权采样

```py
import random
from collections.abc import Sequence

def weighted_sampler(values: Sequence[int], weights: Sequence[float], rng: random.Random) -> Iterator[int]: ...
```

`values` 和 `weights` 是两个等长的序列，`weights[i]` 给出 `values[i]` 的相对权重。`weighted_sampler`
返回一个无穷生成器：每次对它调用 `next()`，都会以概率 `weights[i] / sum(weights)` 返回
`values[i]`，且与之前的每一次抽样相互独立，其间用到的每一个随机选择都只使用 `rng`——不使用任何其他
随机性来源。

`values` 和 `weights` 只会被校验一次，而且校验得很精确：`weighted_sampler(values, weights, rng)` 会
*在被调用的那一刻*、在产出任何值之前抛出 `ValueError`，条件是下面任意一条不成立——`values` 和
`weights` 长度相等，且这个长度不为零；每个权重都是有限的（finite）且 `>= 0`；权重的浮点数之和是有
限的，且 `> 0`。权重恰好为 `0` 的值，此后任何一次 `next()` 调用都不会产出它。

示例：

```text
values = [1, 2, 3], weights = [0.1, 0.1, 0.8]
g = weighted_sampler(values, weights, random.Random(0))
[next(g) for _ in range(10)]   # -> 十个取自 {1, 2, 3} 的值
# 抽样次数足够多时：1 约占 10%，2 约占 10%，3 约占 80%

weighted_sampler([1, 2], [1.0], random.Random(0))    # 立即抛出 ValueError（长度不等）
weighted_sampler([1, 2], [2.0, 6.0], random.Random(0))  # 合法：权重之和不必为 1（25% / 75%）
```

再用一两句话说明：一个返回值本身是随机的函数，要如何为它编写单元测试。

### Part 3 —— 用 alias method 做到每次抽样 O(1)

实现 `AliasSampler`：只在 `__init__` 里付出一次 `O(n)` 的预处理开销，使得此后每次抽样的开销都是
`O(1)`，与 `n = len(values)` 无关：

```py
class AliasSampler:
    def __init__(self, values: Sequence[int], weights: Sequence[float], rng: random.Random) -> None: ...
    def sample(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
```

`AliasSampler(values, weights, rng)` 对 `values` 和 `weights` 的校验规则与 `weighted_sampler` 完全
一致，在构造时——在任何预处理之前——抛出同样的 `ValueError`。`sample()` 返回的一次抽样，其分布、
以及各次抽样之间的独立性，都与对 `weighted_sampler(values, weights, rng)` 调用一次 `next()` 完全
相同；权重恰好为 `0` 的 `values[i]`，同样永远不会被返回。`__iter__` 返回一个等价于反复调用
`sample()` 的无穷生成器。

```text
AliasSampler([1, 2], [0.0, 0.0], random.Random(0))
# 抛出 ValueError —— 和上面 weighted_sampler([1, 2], [0.0, 0.0], ...) 的规则相同

sampler = AliasSampler([1, 2, 3], [0.1, 0.1, 0.8], random.Random(0))
sampler.sample()      # -> 1、2 或 3 中的一个；多次调用后，比例趋向 10% / 10% / 80%
next(iter(sampler))   # -> 同样是 1、2 或 3 中的一个，经由 __iter__ 而不是 sample() 得到
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有几点值得先确认：权重是否已经归一化，使其和为 `1`（不需要——除以它们的和可以同等地处理这
两种情况）；`items` 中一个非法元素应该被跳过还是应该报错（这里的答案是报错，而且是在它之前每一个合
法的值都已经被产出之后）；以及调用方是否会传入随机数生成器，而不是由函数自己创建并设置种子（这里
的答案是调用方会传入，这正是下面的测试之所以可复现的原因）。

### Part 1

一个生成器函数的函数体，在对它调用第一次 `next()` 之前完全不会执行；每个 `yield` 都会把函数恰好挂
起在那个位置，所以推动 `enumerate(items)` 前进这件事，只会在这个函数被要求再产出一个值时才发生。
这正是 `iter_ints` 免费获得惰性的原因，即使输入永不结束也是如此：下面的代码完全不需要显式地写“产
出 `k` 个值之后停下”，因为它从来不会跑在 `next()` 要求它跑到的地方之前。

```python
from collections.abc import Iterable, Iterator


def iter_ints(items: Iterable[object] | None) -> Iterator[int]:
    """Yields the int elements of items, in order, skipping None. Raises TypeError, naming the
    0-based index, on the first element that is neither None nor an int (bool excluded)."""
    if items is None:
        return
    # NOTE: iterate items directly and yield as each value is ready -- calling list(items) first
    # would force even an infinite items to run forever before this function could produce anything.
    for index, x in enumerate(items):
        if x is None:           # NOTE: must be `is None`, not `if not x` -- the latter would also
            continue            #       skip the valid value 0, and would skip False instead of rejecting it
        # NOTE: isinstance(x, bool) must be checked and rejected separately -- bool is a subclass of
        # int in Python, so isinstance(x, int) alone would silently accept True and False as 1 and 0.
        if not isinstance(x, int) or isinstance(x, bool):
            # NOTE: raised right here, the moment this element is reached -- validating every element
            # of items before yielding any of them would raise before the valid values at earlier
            # positions are yielded, and would never finish on an input that runs forever.
            raise TypeError(f"element at index {index} is not an int: {x!r}")
        yield x
```

一个 `unittest.TestCase` 逐条验证上面的规则，其结构使得同一套测试之后可以原封不动地套用到一个存
心写错的实现上：每个测试方法调用的都是 `self.FUNC`，而不是直接调用 `iter_ints`，`FUNC` 是一个子
类可以覆盖的类属性。

```python
import unittest


class _CountingIterator:
    """Wraps a fixed, finite sequence and records how many of its elements __next__ has actually
    produced -- this is how the laziness test below observes pulls without needing an infinite
    iterator, so an eager bug shows up as a clean assertion failure rather than a hang."""

    def __init__(self, values: list[int]) -> None:
        self._values = list(values)
        self._position = 0
        self.pulled = 0

    def __iter__(self) -> "_CountingIterator":
        return self

    def __next__(self) -> int:
        if self._position >= len(self._values):
            raise StopIteration
        value = self._values[self._position]
        self._position += 1
        self.pulled += 1
        return value


class IterIntsTests(unittest.TestCase):
    FUNC = staticmethod(iter_ints)

    def test_empty_and_none_input(self) -> None:
        self.assertEqual(list(self.FUNC([])), [])
        self.assertEqual(list(self.FUNC(None)), [])

    def test_none_at_start_middle_end_and_all_none(self) -> None:
        self.assertEqual(list(self.FUNC([None, 1, 2])), [1, 2])
        self.assertEqual(list(self.FUNC([1, None, 2])), [1, 2])
        self.assertEqual(list(self.FUNC([1, 2, None])), [1, 2])
        self.assertEqual(list(self.FUNC([None, None, None])), [])

    def test_zero_and_negative_are_ordinary_values(self) -> None:
        self.assertEqual(list(self.FUNC([0, -5, 3, 0])), [0, -5, 3, 0])

    def test_lazy_one_pull_per_next_call(self) -> None:
        source = _CountingIterator([10, 20, 30])
        gen = self.FUNC(source)
        self.assertEqual(source.pulled, 0)              # nothing pulled before the first next()
        self.assertEqual(next(gen), 10)
        self.assertEqual(source.pulled, 1)
        self.assertEqual(next(gen), 20)
        self.assertEqual(source.pulled, 2)

    def test_invalid_element_after_valid_ones(self) -> None:
        gen = self.FUNC([1, 2, True])
        self.assertEqual(next(gen), 1)
        self.assertEqual(next(gen), 2)
        with self.assertRaises(TypeError) as ctx:
            next(gen)
        self.assertIn("index 2", str(ctx.exception))

    def test_bool_float_str_rejected(self) -> None:
        for bad in (True, False, 3.0, "3"):
            with self.assertRaises(TypeError):
                list(self.FUNC([bad]))

    def test_input_list_is_unchanged(self) -> None:
        items = [3, None, 0, -2]
        before = list(items)
        list(self.FUNC(items))
        self.assertEqual(items, before)

    def test_two_calls_are_independent(self) -> None:
        items = [1, 2, 3]
        first, second = self.FUNC(items), self.FUNC(items)
        self.assertEqual(next(first), 1)
        self.assertEqual(next(second), 1)               # unaffected by first's progress
        self.assertEqual(next(first), 2)
        self.assertEqual(list(second), [2, 3])
```

用 `python -m unittest` 运行它，和运行其他任何 `unittest.TestCase` 一样；下面的验证代码则改为以
编程方式运行 `IterIntsTests`——把它装进一个 `TestSuite`，交给一个把输出写进一次性用完的流的
`TextTestRunner`——这样整个页面仍然是一个自成一体的脚本，不需要另外调用测试运行器。

### Part 2

这里的关键点和 Part 1 是同一个，只是方向相反：生成器函数的函数体在第一次调用 `next()` 之前不会开
始执行，所以直接写在生成器函数内部的校验代码，也就不会在那之前运行——因而也不会抛出异常。让
`weighted_sampler` 本身是一个普通函数，先做校验，再返回一个内嵌的生成器，就能同时得到这两样东
西：校验是即时的，抽样是惰性的。

```python
import math
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from itertools import accumulate

import scipy.stats


def weighted_sampler(values: Sequence[int], weights: Sequence[float], rng: random.Random) -> Iterator[int]:
    """Infinite generator: each next() returns values[i] with probability weights[i] / sum(weights),
    independently of every earlier draw, using only rng for randomness."""
    # NOTE: weighted_sampler is an ordinary function, not itself a generator -- see above. Validating
    # here and returning a nested generator raises eagerly while the draws it returns stay lazy.
    if len(values) != len(weights) or len(values) == 0:
        raise ValueError("values and weights must have equal, non-zero length")
    for w in weights:
        if not (math.isfinite(w) and w >= 0):
            raise ValueError(f"weight {w!r} is not finite and non-negative")
    cumulative = list(accumulate(weights))
    total = cumulative[-1]
    # NOTE: finite weights can still add up to inf (1e308 + 1e308); rng.random() * inf would then lie
    # past every prefix sum, so the total must be checked for finiteness as well as positivity.
    if not 0 < total < math.inf:
        raise ValueError("weights must have a finite, positive sum")

    def _draws() -> Iterator[int]:
        while True:
            # NOTE: scale by `total`, i.e. by cumulative[-1] -- not by a separately recomputed
            # sum(weights). Floating-point addition is not perfectly associative, so the two sums can
            # differ in their last bit; scaling by a freshly recomputed sum could then push u a hair
            # past cumulative[-1], and bisect_right would return an index one past the end of values.
            u = rng.random() * total
            yield values[bisect_right(cumulative, u)]

    return _draws()
```

建好 `cumulative` 的开销是 $O(n)$，每次调用 `weighted_sampler` 只需要付出一次；此后每次抽样都在它
上面做一次 `bisect_right`，$O(\log n)$，外加 $O(1)$ 的其他工作。`bisect_right(cumulative, u)` 要
返回下标 $i$，$u$ 必须落在 $[\,\mathrm{cumulative}[i-1],\ \mathrm{cumulative}[i])$ 之内，其中
$i = 0$ 时区间的下端是 $0$，`u >= 0` 满足这一点。当 $\mathrm{weights}[i] = 0$ 时，
$\mathrm{cumulative}[i] = \mathrm{cumulative}[i - 1]$，这个区间为空，所以一个零权重的下标绝不可能是
`bisect_right` 的答案——下标 $0$ 也是同样的道理：它需要
$u < \mathrm{cumulative}[0]$，而一旦 $\mathrm{weights}[0] = 0$ 使得 $\mathrm{cumulative}[0] = 0$，
再加上 $u \geq 0$，这就不可能成立。

在示例上，前缀和（prefix sum，即运行总和 `weights[0]`、`weights[0] + weights[1]`……）是
`[0.1, 0.2, 1.0]`，`[0, 1.0)` 上的一次均匀抽样 `u` 按下面的规则映射到一个值：

```text
u 落在 [0,   0.1) -> 1        例如 u = 0.05 -> 1
u 落在 [0.1, 0.2) -> 2        例如 u = 0.15 -> 2
u 落在 [0.2, 1.0) -> 3        例如 u = 0.93 -> 3
```

一个返回值本身是随机的函数，没办法用单一的固定期望输出来测试，但四种互补的手段合在一起，可以把它
的行为钉住：固定 `rng` 的种子，让任意一次运行都可以复现；用一个*桩*（stub）随机数生成器代替
`rng`，让它的 `random()` 返回一个提前选定好的序列，这样就能确定性地、而不是靠统计去验证从 `u` 到
某个值的映射，边界值也不例外；多次抽样后检查几条简单的性质（抽出的值只会是 `values` 里的值，权重
为零的值永远不会出现），可以低成本地抓住明显的错误；最后是一次统计检验——这里用卡方拟合优度检验
（chi-square goodness-of-fit test），比较观测到的抽样计数与 `weights` 所隐含的计数——用来抓住比
例上有细微偏差的采样器，这是前面三种手段都发现不了的。

```python
class _StubRandom:
    """A random.Random look-alike whose random() returns a fixed, pre-chosen sequence of floats, so a
    test can pin down exactly which u each draw uses -- rather than which value happens to come out."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._position = 0

    def random(self) -> float:
        value = self._values[self._position]
        self._position += 1
        return value


def _chi_square_p_value(draws: list[int], values: Sequence[int], weights: Sequence[float]) -> float:
    """Independent statistical check: compares observed counts of each value in draws against the
    counts expected under weights / sum(weights), via scipy's chi-square goodness-of-fit test. Assumes
    every weight is strictly positive, so no expected count is ever zero."""
    total = math.fsum(weights)
    counts = Counter(draws)
    observed = [counts[v] for v in values]
    expected = [w / total * len(draws) for w in weights]
    return scipy.stats.chisquare(observed, expected).pvalue


class WeightedSamplerTests(unittest.TestCase):
    def test_validation_is_eager_not_deferred_to_first_next(self) -> None:
        with self.assertRaises(ValueError):
            weighted_sampler([1, 2], [1.0], random.Random(0))           # length mismatch
        with self.assertRaises(ValueError):
            weighted_sampler([], [], random.Random(0))                  # zero length
        with self.assertRaises(ValueError):
            weighted_sampler([1], [-1.0], random.Random(0))             # negative weight
        with self.assertRaises(ValueError):
            weighted_sampler([1], [math.inf], random.Random(0))         # non-finite weight
        with self.assertRaises(ValueError):
            weighted_sampler([1, 2], [0.0, 0.0], random.Random(0))      # sums to zero
        with self.assertRaises(ValueError):
            weighted_sampler([1, 2], [1e308, 1e308], random.Random(0))  # sum overflows to inf

    def test_stub_rng_maps_u_to_value_at_the_boundaries(self) -> None:
        values, weights = [1, 2, 3], [0.1, 0.1, 0.8]     # weights sum to exactly 1.0 here
        boundary_us = [0.0, 0.05, 0.1, 0.15, 0.2, 0.93]
        stub = _StubRandom(boundary_us)
        gen = weighted_sampler(values, weights, stub)
        self.assertEqual([next(gen) for _ in boundary_us], [1, 1, 2, 2, 3, 3])

    def test_only_listed_values_and_zero_weight_never_appears(self) -> None:
        values, weights = [10, 20, 30, 40], [0.0, 2.0, 0.0, 1.0]
        gen = weighted_sampler(values, weights, random.Random(1))
        draws = [next(gen) for _ in range(2_000)]
        self.assertTrue(set(draws) <= {10, 20, 30, 40})
        self.assertNotIn(10, draws)
        self.assertNotIn(30, draws)
        self.assertIn(20, draws)             # sanity: not everything got filtered out by mistake
        self.assertIn(40, draws)

    def test_chi_square_goodness_of_fit(self) -> None:
        values, weights = [1, 2, 3], [0.1, 0.1, 0.8]
        gen = weighted_sampler(values, weights, random.Random(12345))
        draws = [next(gen) for _ in range(150_000)]
        p_value = _chi_square_p_value(draws, values, weights)
        # NOTE: 1e-6 is a low bar on purpose -- a correctly distributed sampler clears it on virtually
        # any fixed seed, while a sampler biased by even a few percentage points fails it overwhelmingly
        # (see the deliberately biased samplers in the checks below).
        self.assertGreater(p_value, 1e-6)
```

用 `python -m unittest` 运行它，和运行 `IterIntsTests` 一样；下面的验证代码还会在 `unittest` 之外
直接复用 `_chi_square_p_value`，套用到两个存心写偏的采样器上，确认它确实会拒绝它们。

### Part 3

Vose 的 alias method 把 `values` 和 `weights` 预处理成两个长度为 $n$ 的数组，`prob` 和 `alias`，使
得每次抽样都变成一次均匀整数外加一次抛硬币。这张表（table）的第 $i$ 列永远恰好携带总概率质量的
$1/n$，在两个值之间分配：`values[i]` 自己占 $\mathrm{prob}[i] / n$，`values[alias[i]]` 占
$(1 - \mathrm{prob}[i]) / n$。`sample()` 先用 `rng.randrange(n)` 均匀选出一列 $i$，再用一次新的
`rng.random()` 与 $\mathrm{prob}[i]$ 比较，把这一列固定的 $1/n$ 份额分给这两个值中的一个——所以每
次抽样恰好只花一次 `randrange` 调用和一次 `random` 调用，与 $n$ 无关。

构建这张表，先把每个权重缩放到*平均*缩放权重恰好为 $1$：
$\mathrm{scaled}[i] = n \cdot \mathrm{weights}[i] / \mathrm{total}$。一个 $\mathrm{scaled}[i] < 1$
的下标（*轻*列，light）持有的份额不到自己应得的份额，需要别处的援助；一个
$\mathrm{scaled}[i] \geq 1$ 的下标（*重*列，heavy）持有的份额不少于自己应得的份额，还能拿出多余的
部分援助别人。这个方法反复把一个轻下标 $s$ 和一个重下标 $l$ 配成一对：$s$ 自己在表中的条目当场就
此敲定，$\mathrm{prob}[s] = \mathrm{scaled}[s]$，$\mathrm{alias}[s] = l$，而 $l$ 恰好援助 $s$ 所缺
的 $1 - \mathrm{scaled}[s]$，$\mathrm{scaled}[l]\ {-}{=}\ 1 - \mathrm{scaled}[s]$。如果这次援助让
$l$ 跌破 $1$，$l$ 自己就变成了轻列，重新进入待处理的池子，等着之后再和别的重列配对；否则它仍然是
重列，继续对外援助。

```python
class AliasSampler:
    """Vose's alias method: O(n) preprocessing, then O(1) time and at most two random numbers per
    draw, regardless of n. Same validation and semantics -- including which values ever appear -- as
    weighted_sampler."""

    def __init__(self, values: Sequence[int], weights: Sequence[float], rng: random.Random) -> None:
        n = len(values)
        if n != len(weights) or n == 0:
            raise ValueError("values and weights must have equal, non-zero length")
        for w in weights:
            if not (math.isfinite(w) and w >= 0):
                raise ValueError(f"weight {w!r} is not finite and non-negative")
        total = sum(weights)
        if not 0 < total < math.inf:
            raise ValueError("weights must have a finite, positive sum")

        self._values = list(values)
        self._rng = rng
        self._prob = [0.0] * n
        self._alias = [0] * n

        scaled = [w / total * n for w in weights]   # NOTE: n * w first could overflow to inf
        light = [i for i, s in enumerate(scaled) if s < 1.0]
        heavy = [i for i, s in enumerate(scaled) if s >= 1.0]

        while light and heavy:
            s = light.pop()
            l = heavy.pop()
            self._prob[s] = scaled[s]
            self._alias[s] = l
            scaled[l] -= 1.0 - scaled[s]      # NOTE: l donates exactly what s was short of 1.0
            if scaled[l] < 1.0:
                light.append(l)
            else:
                heavy.append(l)

        # NOTE: in exact arithmetic the loop ends with light empty and every column left in heavy at
        # scaled weight exactly 1 (see the invariant in the text below); rounding can also strand a
        # column in light just below 1. Every leftover column's true scaled weight is 1, so it keeps
        # its whole share: prob = 1, and alias is never consulted for it.
        for i in light:
            self._prob[i] = 1.0
        for i in heavy:
            self._prob[i] = 1.0

    def sample(self) -> int:
        i = self._rng.randrange(len(self._values))
        if self._rng.random() < self._prob[i]:
            return self._values[i]
        return self._values[self._alias[i]]

    def __iter__(self) -> Iterator[int]:
        while True:
            yield self.sample()
```

（当 $n = 1$ 时，唯一的那个下标有 $\mathrm{scaled}[0] = 1 \cdot w_0 / \mathrm{total} = 1$，不多不
少，所以它一开始就在 `heavy` 里，`light` 永远不会有成员，`while` 循环体一次都不会执行，这一列就直
接落入最后兜底的那一步，$\mathrm{prob}[0] = 1$——那里唯一的值总会被抽到。）

这个构造依赖一条不变式，在每一次配对中都保持成立：在任意时刻，尚未敲定的那些列，它们的缩放权重之
和恰好等于这些列的数目。一开始，
$\sum_i \mathrm{scaled}[i] = \sum_i n \cdot w_i / \mathrm{total} = n$，恰好对应全部 $n$ 列，一列都
还没敲定。每一次配对都把 $s$ 从未敲定集合中拿掉，把
$\mathrm{scaled}[s]$ 从这个和里减掉，并把 $l$ 的缩放权重减少 $1 - \mathrm{scaled}[s]$；于是未敲定
列的这个和恰好减少了 $\mathrm{scaled}[s] + (1 - \mathrm{scaled}[s]) = 1$，正好对应未敲定列的数目
从 $k$ 变成 $k - 1$。所以只要还剩 $k$ 列未敲定，它们的缩放权重之和就恰好是 $k$。因此循环不可能在
`light` 非空时因 `heavy` 变空而停下——$k \geq 1$ 个全都小于 $1$ 的数，和不可能等于 $k$——所以它停下
时 `light` 为空，而留在 `heavy` 里的 $k$ 列，每列都至少是 $1$、总和是 $k$，于是每列都*必然*恰好等
于 $1$：这正是最后 $\mathrm{prob} = 1$ 那一步所处理的情形（例如权重全部相等时，每一列都会留在那
里）。浮点舍入误差可能让某个剩下的列只是在 $1$ 附近的舍入误差范围内，而不是恰好等于 $1$，这样的列
在两个列表里都可能出现；同一处兜底逻辑同样覆盖它。

第 $s$ 列把 $\mathrm{prob}[s] / n$ 给 $\mathrm{values}[s]$，把 $(1 - \mathrm{prob}[s]) / n$ 给
$\mathrm{values}[\mathrm{alias}[s]]$，而 $1 - \mathrm{prob}[s]$ 恰好就是 $s$ 被敲定时从
$l = \mathrm{alias}[s]$ 的当前权重里减掉的量。任意下标 $v$ 的当前权重从 $n \cdot w_v / \mathrm{total}$
开始，恰好减去 $v$ 援助给其他列的量，剩下的部分在 $v$ 自己那一列被敲定时成为 $\mathrm{prob}[v]$
（剩下未配对的列则是 $1$，它不再对外援助）。把所有列加总，$v$ 得到的恰好是

$$
\frac{1}{n}\Bigl(\mathrm{prob}[v] + \sum_{s:\ \mathrm{alias}[s] = v} \bigl(1 - \mathrm{prob}[s]\bigr)\Bigr)
= \frac{1}{n} \cdot \frac{n \, w_v}{\mathrm{total}} = \frac{w_v}{\mathrm{total}},
$$

这正是下面的验证代码直接从 `prob` 和 `alias` 里验证的那个精确分布，完全不需要真的抽样。

构建 `light`、`heavy` 和 `scaled` 的开销是 $O(n)$；`while` 循环至多执行 $n - 1$ 次，因为每一次迭
代都会彻底敲定一列，每次迭代的开销是 $O(1)$——所以 `__init__` 总的开销是 $O(n)$。`sample()` 只做
一次 `randrange` 调用、一次 `random` 调用和一次比较：$O(1)$，与 $n$ 无关。

### 追问

- **两次抽样之间权重会变化。** 在权重上建一棵 Fenwick 树（树状数组，binary indexed tree），可以让
  更新一个权重和做一次加权抽样都是 $O(\log n)$：一次抽样从根节点往下走，每个节点上，如果一个随下降
  过程不断更新的均匀值落在它存的那个子树和以内，就走向左子树（否则减去这个和，转向右子树）——这正是
  `bisect_right` 在一个前缀和数组上做的同一种下降过程，只是不需要在每次写入之后重建这个数组。
- **带权重地采样 $k$ 个不放回的元素。** Efraimidis–Spirakis 采样给每个元素 $i$ 一个键
  $u_i^{1/w_i}$，其中 $u_i$ 从 $(0, 1)$ 中均匀抽取，然后保留键最大的 $k$ 个元素；只需要一趟遍历、
  每个元素一个键，不需要移除后再重新归一化的步骤。由于 $P(u_i^{1/w_i} \leq t) = t^{w_i}$，元素 $i$
  持有最大键的概率是 $\int_0^1 w_i t^{w_i - 1} \prod_{j \neq i} t^{w_j}\,dt = w_i / \sum_j w_j$（即
  Part 2、Part 3 的一次抽样），而其余的键除以最大键之后，仍然是同一形式的键；所以这 $k$ 个元素按键
  从大到小排列，其分布恰好等于连续 $k$ 次抽样：每次在尚未选中的元素中，以正比于权重的概率选出一个。
- **数据流太长、存不下的加权水库抽样（reservoir sampling）。** 把同样的键配上一个大小为 $k$ 的小
  顶堆，就得到了一个流式版本：保留目前为止见过的、键最大的 $k$ 个，每当新来的键胜过堆里最小的那个
  就替换掉它；数据流结束时堆里剩下的就是一份有效的加权不放回样本，全程至多存储 $k$ 个元素，每个元
  素只保留一个键。
- **连续型分布。** 离散情形下的想法可以直接推广：对于一个 CDF 为 $F$ 的连续分布，逆变换采样
  （inverse-transform sampling）先从 $[0, 1)$ 中均匀抽取 $u$，再返回 $F^{-1}(u)$；当 $F$ 没有闭式
  的反函数、但仍然单调时，`bisect_right` 在上面离散前缀和上做的那种二分查找，可以直接搬到 $F$ 上
  进行。
- **库里的实现。** `random.choices` 和 NumPy 的 `Generator.choice(p=...)` 用的都是和 Part 2 一样的
  前缀和加二分查找的思路，而不是 Part 3 的 alias method——权重经常变化时，前者重新构建的代价更
  低；但如果要从同一个固定分布里抽很多次，每次抽样的代价就比用一张 alias 表更高，这正是
  Part 3 要解决的权衡。

<details>
<summary>验证代码（可运行）</summary>

```python
import io
import itertools

# --- Part 1: the worked examples in the Problem section ---
assert list(iter_ints([3, None, 0, -2])) == [3, 0, -2]
g = iter_ints([1, 2, True])
assert next(g) == 1 and next(g) == 2
try:
    next(g)
    assert False, "expected TypeError"
except TypeError as exc:
    assert "index 2" in str(exc)

# iter_ints works on an infinite iterator, producing values lazily, one at a time
first_five = list(itertools.islice(iter_ints(itertools.count()), 5))
assert first_five == [0, 1, 2, 3, 4]


def _run_suite(test_case_cls):
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=0)
    result = runner.run(unittest.TestLoader().loadTestsFromTestCase(test_case_cls))
    return result


# --- Part 1: the page's own unittest suite passes on the correct implementation ---
assert _run_suite(IterIntsTests).wasSuccessful()


# --- Part 1: an independent brute force, restating the rule from the statement alone ---
def _reference_iter_ints(items):
    if items is None:
        return
    for i, x in enumerate(items):
        if x is None:
            continue
        if not isinstance(x, int) or isinstance(x, bool):
            raise TypeError(f"index {i}")
        yield x


def _drain(gen):
    """Consumes gen, returning (values yielded before any error, the error raised, or None)."""
    values = []
    try:
        for v in gen:
            values.append(v)
    except TypeError as exc:
        return values, exc
    return values, None


def _error_index(exc) -> int:
    tokens = str(exc).replace(":", " ").split()
    return int(tokens[tokens.index("index") + 1])


rng = random.Random(0)
for _ in range(300):
    items = []
    for _ in range(rng.randint(0, 12)):
        choice = rng.random()
        if choice < 0.2:
            items.append(None)
        elif choice < 0.9:
            items.append(rng.randint(-5, 5))
        else:
            items.append(rng.choice([True, False, 3.0, "3"]))
    before = list(items)
    got_values, got_error = _drain(iter_ints(items))
    want_values, want_error = _drain(_reference_iter_ints(items))
    assert got_values == want_values, (items, got_values, want_values)
    assert (got_error is None) == (want_error is None), (items, got_error, want_error)
    if want_error is not None:
        assert _error_index(got_error) == _error_index(want_error), (items, got_error, want_error)
    assert items == before   # NOTE: independently confirms items was never modified


# --- Part 1: mutation check -- four deliberately buggy variants, each failing IterIntsTests ---
def _buggy_drops_zero(items):
    if items is None:
        return
    for i, x in enumerate(items):
        if not x:                                       # BUG: also skips 0 (and False), not just None
            continue
        if not isinstance(x, int) or isinstance(x, bool):
            raise TypeError(f"index {i}")
        yield x


def _buggy_eager_list(items):
    if items is None:
        return
    materialised = list(items)                            # BUG: forces a possibly-infinite items up front
    for i, x in enumerate(materialised):
        if x is None:
            continue
        if not isinstance(x, int) or isinstance(x, bool):
            raise TypeError(f"index {i}")
        yield x


def _buggy_accepts_bool(items):
    if items is None:
        return
    for i, x in enumerate(items):
        if x is None:
            continue
        if not isinstance(x, int):                          # BUG: bool slips through unrejected
            raise TypeError(f"index {i}")
        yield x


def _buggy_validates_upfront(items):
    if items is None:
        return
    materialised = list(items)
    for i, x in enumerate(materialised):                     # BUG: raised before anything is yielded
        if x is not None and (not isinstance(x, int) or isinstance(x, bool)):
            raise TypeError(f"index {i}")
    for x in materialised:
        if x is not None:
            yield x


for buggy in (_buggy_drops_zero, _buggy_eager_list, _buggy_accepts_bool, _buggy_validates_upfront):
    buggy_cls = type("_BuggyIterIntsTests", (IterIntsTests,), {"FUNC": staticmethod(buggy)})
    assert not _run_suite(buggy_cls).wasSuccessful(), buggy.__name__


# --- Part 2: the page's own unittest suite passes on the correct implementation ---
assert _run_suite(WeightedSamplerTests).wasSuccessful()


# --- Part 2: deliberately biased samplers, rejected by the same chi-square test ---
def _biased_forgets_total_scaling(values, weights, rng):
    """BUG: scales u by 1.0 (uses rng.random() directly) instead of by cumulative[-1] -- invisible
    when weights already happen to sum to 1, catastrophic otherwise."""
    cumulative = list(accumulate(weights))
    while True:
        u = rng.random()                            # BUG: should be rng.random() * cumulative[-1]
        yield values[bisect_right(cumulative, u)]


def _biased_perturbed(values, weights, rng, bump=0.03):
    """BUG: silently inflates weights[0] by `bump` times the total -- a few percentage points of
    probability -- before building the prefix sums; everything else is the correct algorithm."""
    weights = list(weights)
    weights[0] += bump * math.fsum(weights)
    cumulative = list(accumulate(weights))
    total = cumulative[-1]
    while True:
        u = rng.random() * total
        yield values[bisect_right(cumulative, u)]


# Not used as a "biased" example: a bisect_left variant only disagrees with bisect_right when u lands
# exactly on a prefix sum, an event of probability 0 that no statistical test can ever observe.

values, weights = [1, 2, 3], [0.1, 0.1, 0.8]

no_scaling_gen = _biased_forgets_total_scaling([1, 2, 3], [1.0, 1.0, 8.0], random.Random(1))
no_scaling_draws = [next(no_scaling_gen) for _ in range(20_000)]
p_no_scaling = _chi_square_p_value(no_scaling_draws, [1, 2, 3], [1.0, 1.0, 8.0])
assert p_no_scaling < 1e-6, p_no_scaling              # catastrophically biased -- caught easily

perturbed_gen = _biased_perturbed(values, weights, random.Random(2))
perturbed_draws = [next(perturbed_gen) for _ in range(150_000)]
p_perturbed = _chi_square_p_value(perturbed_draws, values, weights)
assert p_perturbed < 1e-6, p_perturbed                # subtly biased -- still caught at this sample size

# the same seed reproduces the same sequence of successive draws
gen_a = weighted_sampler(values, weights, random.Random(555))
gen_b = weighted_sampler(values, weights, random.Random(555))
rep_a = [next(gen_a) for _ in range(20)]
rep_b = [next(gen_b) for _ in range(20)]
assert rep_a == rep_b and len(set(rep_a)) > 1


# --- Part 3: AliasSampler validates exactly like weighted_sampler, including the Problem's example ---
for bad_values, bad_weights in (
    ([1, 2], [1.0]),          # length mismatch
    ([], []),                  # zero length
    ([1], [-1.0]),             # negative weight
    ([1], [math.inf]),         # non-finite weight
    ([1, 2], [0.0, 0.0]),      # sums to zero -- the exact case traced in the Problem section
    ([1, 2], [1e308, 1e308]),  # finite weights whose sum overflows to inf
):
    for ctor in (weighted_sampler, AliasSampler):
        try:
            ctor(bad_values, bad_weights, random.Random(0))
            assert False, (ctor.__name__, bad_values, bad_weights)
        except ValueError:
            pass

# --- Part 3: the exact distribution implied by prob/alias matches w / total to 1e-12 ---
def _alias_table_distribution(sampler: AliasSampler) -> list[float]:
    """Recovers the exact probability AliasSampler assigns to each original index, from prob and alias
    alone -- no sampling involved, so this is exact rather than statistical."""
    n = len(sampler._values)
    dist = [0.0] * n
    for i in range(n):
        dist[i] += sampler._prob[i] / n
        dist[sampler._alias[i]] += (1.0 - sampler._prob[i]) / n
    return dist


def _weight_scenarios(rng, trials):
    """Random weight vectors covering zeros, one dominant weight, n = 1 and equal weights, alongside
    plain random ones."""
    for _ in range(trials):
        kind = rng.choice(["n1", "equal", "zeros", "dominant", "random"])
        if kind == "n1":
            ws = [rng.uniform(0.1, 5.0)]
        elif kind == "equal":
            ws = [1.0] * rng.randint(2, 6)
        elif kind == "zeros":
            n = rng.randint(2, 6)
            ws = [0.0 if rng.random() < 0.4 else rng.uniform(0.1, 5.0) for _ in range(n)]
            if not any(w > 0 for w in ws):
                ws[0] = 1.0
        elif kind == "dominant":
            n = rng.randint(2, 6)
            ws = [rng.uniform(0.001, 0.01) for _ in range(n)]
            ws[rng.randrange(n)] = 1_000.0
        else:
            ws = [rng.uniform(0.0, 5.0) for _ in range(rng.randint(1, 8))]
            if math.fsum(ws) == 0.0:
                ws[0] = 1.0
        yield list(range(100, 100 + len(ws))), ws


for scenario_values, scenario_weights in _weight_scenarios(random.Random(7), 300):
    scenario_total = math.fsum(scenario_weights)
    sampler = AliasSampler(scenario_values, scenario_weights, random.Random(0))
    dist = _alias_table_distribution(sampler)
    for w, p in zip(scenario_weights, dist):
        assert abs(p - w / scenario_total) < 1e-12, (scenario_weights, dist)

# huge weights: the total is finite, but n * w alone would overflow
huge_weights = [6e307, 6e307, 1.0]
huge_dist = _alias_table_distribution(AliasSampler([1, 2, 3], huge_weights, random.Random(0)))
assert all(abs(p - w / math.fsum(huge_weights)) < 1e-12 for w, p in zip(huge_weights, huge_dist))

# --- Part 3: sampling itself passes the same chi-square test as Part 2 ---
alias_sampler = AliasSampler(values, weights, random.Random(999))
alias_draws = [alias_sampler.sample() for _ in range(150_000)]
p_alias = _chi_square_p_value(alias_draws, values, weights)
assert p_alias > 1e-6, p_alias

# also exercised through __iter__, not just sample() directly
iter_draws = list(itertools.islice(iter(AliasSampler(values, weights, random.Random(1000))), 5_000))
assert set(iter_draws) == {1, 2, 3}


# --- Parts 2 and 3 agree in distribution: a two-sample chi-square test of homogeneity ---
# NOTE: both count vectors are random, so this is a 2 x n contingency-table test; plugging one sample
# into chisquare() as if it were the exact expected counts would double the statistic's variance.
gen2 = weighted_sampler(values, weights, random.Random(111))
sampler3 = AliasSampler(values, weights, random.Random(222))
counts2 = Counter(next(gen2) for _ in range(80_000))
counts3 = Counter(sampler3.sample() for _ in range(80_000))
table = [[counts2[v] for v in values], [counts3[v] for v in values]]
p_agree = scipy.stats.chi2_contingency(table).pvalue
assert p_agree > 1e-6, p_agree

print("all checks passed")
```

</details>

</details>
