# Python Generators, Unit Tests and Weighted Sampling

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · Python generators, testing and sampling | ★★★☆☆ | Medium | MLE · SWE · RE · Intern | generators, unit-testing, weighted-sampling, prefix-sums, binary-search, alias-method, chi-square-test | 3 parts / 45 min | Skills interview |
<!-- meta:end -->

## Problem

### Part 1 — A lazy generator over corner cases, and its tests

```py
from collections.abc import Iterable, Iterator

def iter_ints(items: Iterable[object] | None) -> Iterator[int]: ...
```

Write `iter_ints`, a generator function that yields the integers of `items`, in order. The following
rules decide its output completely:

- It consumes `items` *lazily*: after it has produced `k` values, it has pulled from `items` only as far
  as the element that produced the `k`-th value, plus any `None` elements before that point, never
  further. It must therefore work on any iterable, including an infinite one such as `itertools.count()`.
- `items is None` behaves like an empty input: nothing is yielded, and `items` is never touched.
- A `None` element is *skipped*: it contributes nothing to the output and does not count as an error.
  Every other element must be an instance of `int`; `bool` is rejected even though `bool` is a subclass
  of `int` in Python, and so are a float such as `3.0` and a string such as `"3"`, even where they represent the
  same number. On the first such invalid element, `iter_ints` raises `TypeError` with a message that
  names the element's position — its 0-based index in `items` — at the exact moment its generator
  reaches that element: every valid value at an earlier position has already been yielded by then, and
  nothing at a later position is ever looked at.
- `items` itself is never modified.

Worked example:

```text
list(iter_ints([3, None, 0, -2]))   # -> [3, 0, -2]      (None skipped, 0 and -2 are ordinary values)

g = iter_ints([1, 2, True])
next(g)   # -> 1
next(g)   # -> 2
next(g)   # raises TypeError, message mentions index 2
```

Then write a `unittest.TestCase` that exercises `iter_ints`, covering at least: an empty list and
`items is None`; a `None` element at the start, in the middle, at the end, and an input made entirely of
`None`; `0` and negative integers, which are ordinary values rather than missing ones; that the generator
is lazy — one call to `next()` pulls exactly one element from a *finite* iterator instrumented to record
how many elements have been pulled (an infinite iterator must not appear in a test: an eager
implementation would then simply hang instead of failing cleanly); an invalid element that comes after
one or more valid ones, checking that the valid ones are yielded first and that the `TypeError` raised
afterwards names the right index; each of `bool`, `float` and `str` rejected as an element type; that the
list passed in is unchanged after `iter_ints` has been fully consumed; and that two separate calls to
`iter_ints` on the same list produce independent generators.

### Part 2 — Weighted sampling as an infinite generator

```py
import random
from collections.abc import Sequence

def weighted_sampler(values: Sequence[int], weights: Sequence[float], rng: random.Random) -> Iterator[int]: ...
```

`values` and `weights` are equal-length sequences, `weights[i]` giving the relative weight of
`values[i]`. `weighted_sampler` returns an infinite generator: each call to `next()` on it returns
`values[i]` with probability `weights[i] / sum(weights)`, independently of every draw before it, using
`rng` — and no other source of randomness — for every random choice it makes.

`values` and `weights` are validated once, and precisely: `weighted_sampler(values, weights, rng)` raises
`ValueError` *at the moment it is called*, before any value has ever been produced, exactly when any of
the following fails — `values` and `weights` have the same length, and that length is not zero; every
weight is finite and `>= 0`; the floating-point sum of the weights is finite and `> 0`. A value whose
weight is exactly `0` is never produced by any later call to `next()`.

Worked example:

```text
values = [1, 2, 3], weights = [0.1, 0.1, 0.8]
g = weighted_sampler(values, weights, random.Random(0))
[next(g) for _ in range(10)]   # -> ten values from {1, 2, 3}
# over many draws: 1 about 10% of the time, 2 about 10%, 3 about 80%

weighted_sampler([1, 2], [1.0], random.Random(0))    # raises ValueError at once (lengths differ)
weighted_sampler([1, 2], [2.0, 6.0], random.Random(0))  # fine: weights need not sum to 1 (25% / 75%)
```

Also state, in a sentence or two, how a function whose return value is random can be unit-tested at all.

### Part 3 — O(1) per draw with the alias method

Implement `AliasSampler`, which pays for an `O(n)` preprocessing step once, in `__init__`, so that every
draw afterwards costs `O(1)` time regardless of `n = len(values)`:

```py
class AliasSampler:
    def __init__(self, values: Sequence[int], weights: Sequence[float], rng: random.Random) -> None: ...
    def sample(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
```

`AliasSampler(values, weights, rng)` validates `values` and `weights` exactly as `weighted_sampler` does,
raising the same `ValueError` at construction time, before any preprocessing happens. `sample()` returns
one draw with the same distribution, and the same independence between draws, that a call to `next()` on
`weighted_sampler(values, weights, rng)` would; `values[i]` with weight exactly `0` is, again, never
returned. `__iter__` returns an infinite generator equivalent to calling `sample()` forever.

```text
AliasSampler([1, 2], [0.0, 0.0], random.Random(0))
# raises ValueError -- same rule as weighted_sampler([1, 2], [0.0, 0.0], ...) above

sampler = AliasSampler([1, 2, 3], [0.1, 0.1, 0.8], random.Random(0))
sampler.sample()      # -> one of 1, 2 or 3; over many calls, in proportion 10% / 10% / 80%
next(iter(sampler))   # -> also one of 1, 2 or 3, drawn via __iter__ instead of sample()
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

A few points are worth confirming before coding: whether the weights are already normalised to sum to
`1` (they need not be — dividing by their sum handles both cases identically); whether an invalid
element of `items` should be skipped or should raise (here it raises, after every valid element before it
has already been produced); and whether the caller supplies the random number generator rather than the
function seeding its own (here the caller does, which is exactly what makes the tests below
reproducible).

### Part 1

A generator function's body does not execute at all until the first call to `next()` on it; each
`yield` then suspends the function exactly where it is, so driving `enumerate(items)` forward only ever
happens when this function is asked to produce another value. That is what gives `iter_ints` its
laziness for free, including on an input that never ends: nothing about the code below needs to say
"stop after `k` values" explicitly, because nothing runs ahead of what `next()` asks for.

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

A `unittest.TestCase` exercises every rule above, structured so the exact same suite can later be run
against a deliberately buggy implementation without being rewritten: each test method calls `self.FUNC`
rather than `iter_ints` directly, and `FUNC` is a class attribute a subclass can override.

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

Run with `python -m unittest` from the command line, exactly as any other `unittest.TestCase`; the
checks below instead run `IterIntsTests` programmatically — loading it into a `TestSuite` and handing
that to a `TextTestRunner` writing to a throwaway stream — so that the whole page stays one
self-contained script with no separate test runner invocation.

### Part 2

The key point here is the same one from Part 1, applied in the opposite direction: a generator
function's body does not start running until the first call to `next()`, so validation written
directly inside a generator function would not run — and therefore would not raise — until then. Making
`weighted_sampler` itself an ordinary function that validates first and only then returns a nested
generator gets both properties at once: eager validation, lazy draws.

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

Building `cumulative` costs $O(n)$, done once per call to `weighted_sampler`; each draw afterwards does one
`bisect_right` over it, $O(\log n)$, plus $O(1)$ other work. For `bisect_right(cumulative, u)` to return index
$i$, $u$ must fall in $[\,\mathrm{cumulative}[i-1],\ \mathrm{cumulative}[i])$, where for $i = 0$ the lower end
is $0$, which `u >= 0` satisfies. When $\mathrm{weights}[i] = 0$, that interval is empty because
$\mathrm{cumulative}[i] = \mathrm{cumulative}[i - 1]$, so a zero-weight index can never be `bisect_right`'s
answer — including index $0$, which needs $u < \mathrm{cumulative}[0]$, impossible once
$\mathrm{weights}[0] = 0$ makes $\mathrm{cumulative}[0] = 0$ and $u \geq 0$.

On the worked example, the prefix sums (running totals `weights[0]`, `weights[0] + weights[1]`, ...) are
`[0.1, 0.2, 1.0]`, and a uniform draw `u` from `[0, 1.0)` maps to a value as follows:

```text
u in [0,   0.1) -> 1        e.g. u = 0.05 -> 1
u in [0.1, 0.2) -> 2        e.g. u = 0.15 -> 2
u in [0.2, 1.0) -> 3        e.g. u = 0.93 -> 3
```

A function whose return value is random cannot be tested with a single fixed expected output, but four
complementary techniques together pin its behaviour down: fixing `rng`'s seed makes any one
run reproducible; a *stub* random number generator, standing in for `rng` and returning a chosen,
pre-decided sequence from `random()`, tests the exact mapping from `u` to a value, including boundary
values, deterministically rather than statistically; drawing many times and checking simple properties
(only values in `values` ever appear, a zero-weight value never does) catches gross errors cheaply; and,
finally, a statistical test — here, a chi-square goodness-of-fit test comparing observed draw counts
against the counts `weights` implies — catches a sampler that is subtly wrong in its proportions, which
none of the other three would ever notice.

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

Run with `python -m unittest`, exactly like `IterIntsTests`; the checks below also reuse
`_chi_square_p_value` directly, outside `unittest`, against a couple of deliberately biased samplers to
confirm it actually rejects them.

### Part 3

Vose's alias method preprocesses `values` and `weights` into two arrays of length $n$, `prob` and
`alias`, that turn every draw into one uniform integer plus one coin flip. Column $i$ of the table always
carries exactly $1/n$ of the total probability mass, split between two values: `values[i]` itself, with
share $\mathrm{prob}[i] / n$, and `values[alias[i]]`, with share $(1 - \mathrm{prob}[i]) / n$. `sample()`
picks a column $i$ uniformly with `rng.randrange(n)`, then splits that column's fixed $1/n$ share between
the two values it names by comparing a fresh `rng.random()` against $\mathrm{prob}[i]$ — so every draw
costs exactly one call to `randrange` and one call to `random`, regardless of $n$.

Building the table starts by scaling each weight so the *average* scaled weight is exactly $1$:
$\mathrm{scaled}[i] = n \cdot \mathrm{weights}[i] / \mathrm{total}$. An index with $\mathrm{scaled}[i] <
1$ (a *light* column) holds less than its fair share and needs a donation from somewhere; an index with
$\mathrm{scaled}[i] \geq 1$ (a *heavy* column) holds at least its fair share and can afford to donate.
The method repeatedly pairs one light index $s$ with one heavy index $l$: $s$'s own table entry is
finalised on the spot, $\mathrm{prob}[s] = \mathrm{scaled}[s]$ and $\mathrm{alias}[s] = l$, and $l$
donates exactly the $1 - \mathrm{scaled}[s]$ that $s$ was short, $\mathrm{scaled}[l]\ {-}{=}\ 1 -
\mathrm{scaled}[s]$. If that donation drops $l$ below $1$, $l$ becomes light itself and re-enters the
pool to be paired with some other heavy index later; otherwise it stays heavy and keeps donating.

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

(When $n = 1$, the single index has $\mathrm{scaled}[0] = 1 \cdot w_0 / \mathrm{total} = 1$ exactly, so
it starts in `heavy`, `light` never gains a member, the `while` loop body never runs, and the column
falls straight through to the leftover fallback with $\mathrm{prob}[0] = 1$ — the only value there is
always drawn.)

The construction relies on one invariant, maintained across every pairing: at every point, the columns
not yet finalised have scaled weights that sum to exactly the number of such columns. Initially,
$\sum_i \mathrm{scaled}[i] = \sum_i n \cdot w_i / \mathrm{total} = n$, matching all $n$ columns, none yet
finalised. Each pairing removes $s$ from the unfinished set, taking $\mathrm{scaled}[s]$ out of that sum,
and reduces $l$'s scaled weight by $1 - \mathrm{scaled}[s]$; the sum over still-unfinished columns
therefore drops by $\mathrm{scaled}[s] + (1 - \mathrm{scaled}[s]) = 1$ exactly, matching the drop from $k$
unfinished columns to $k - 1$. So whenever $k$ columns remain unfinished, their scaled weights sum to
exactly $k$. The loop therefore cannot stop with `heavy` empty while `light` is not — $k \geq 1$ columns
all below $1$ cannot sum to $k$ — so it stops with `light` empty, and the $k$ columns left in `heavy`,
each at least $1$ and summing to $k$, are each *forced* to equal exactly $1$: the case the final
$\mathrm{prob} = 1$ assignment handles (equal weights, for example, leave every column there).
Floating-point rounding can leave a leftover column within rounding error of $1$ rather than exactly $1$,
in either list; the same fallback covers it.

Column $s$ gives $\mathrm{prob}[s] / n$ to $\mathrm{values}[s]$ and $(1 - \mathrm{prob}[s]) / n$ to
$\mathrm{values}[\mathrm{alias}[s]]$, and $1 - \mathrm{prob}[s]$ is exactly what was subtracted from the
running weight of $l = \mathrm{alias}[s]$ when $s$ was finalised. The running weight of any index $v$
starts at $n \cdot w_v / \mathrm{total}$, loses exactly what $v$ donates to other columns, and the rest
becomes $\mathrm{prob}[v]$ when $v$'s own column is finalised ($1$ for a leftover column, which donates
nothing more). Summing over all columns, $v$ therefore receives exactly

$$
\frac{1}{n}\Bigl(\mathrm{prob}[v] + \sum_{s:\ \mathrm{alias}[s] = v} \bigl(1 - \mathrm{prob}[s]\bigr)\Bigr)
= \frac{1}{n} \cdot \frac{n \, w_v}{\mathrm{total}} = \frac{w_v}{\mathrm{total}},
$$

the exact distribution the checks below verify directly from `prob` and `alias`, with no sampling
involved.

Building `light`, `heavy` and `scaled` costs $O(n)$; the `while` loop runs at most $n - 1$ times, since
each iteration finalises one column for good, and does $O(1)$ work per iteration — so `__init__` costs
$O(n)$ altogether. `sample()` does one `randrange` call, one `random` call and one comparison: $O(1)$,
independent of $n$.

### Follow-ups

- **Weights that change between draws.** A Fenwick tree (binary indexed tree) built over the weights
  supports both updating one weight and taking a weighted draw in $O(\log n)$: a draw descends from the
  root, at each node taking the left subtree if a running uniform value falls under its stored sum
  (subtracting that sum and continuing right otherwise) — the same descent `bisect_right` performs over
  a flat array of prefix sums, but without rebuilding that array after every write.
- **Sampling $k$ items without replacement, with weights.** Efraimidis–Spirakis sampling assigns every
  item $i$ a key $u_i^{1/w_i}$ for $u_i$ drawn uniformly from $(0, 1)$, and keeps the $k$ items with the
  largest keys, in one pass, with one key per item and no removal-and-renormalise step. Since
  $P(u_i^{1/w_i} \leq t) = t^{w_i}$, item $i$ holds the largest key with probability
  $\int_0^1 w_i t^{w_i - 1} \prod_{j \neq i} t^{w_j}\,dt = w_i / \sum_j w_j$ (one draw of Parts 2 and 3),
  and the remaining keys divided by the largest are again keys of the same form; so the $k$ items, in
  decreasing key order, are distributed exactly as $k$ successive draws that each pick one of the items
  not yet chosen with probability proportional to its weight.
- **Weighted reservoir sampling over a stream too long to store.** Pairing the same keys with a
  size-$k$ min-heap gives a streaming version: keep the $k$ largest keys seen so far, replacing the
  smallest whenever a new key beats it, and whatever remains once the stream ends is a valid weighted
  sample without replacement, having stored at most $k$ items and one key per item at any time.
- **A continuous distribution.** The discrete idea generalises directly: for a continuous distribution
  with CDF $F$, inverse-transform sampling draws $u$ uniformly from $[0, 1)$ and returns $F^{-1}(u)$;
  when $F$ has no closed form to invert but is monotone, the same binary search that `bisect_right`
  performs over the discrete prefix sums above runs instead directly over $F$.
- **Library implementations.** `random.choices` and NumPy's `Generator.choice(p=...)` are both built on
  the same prefix-sum-and-binary-search idea as Part 2, rather than the alias method of Part 3 — cheaper
  to build fresh when weights change often, but costlier per draw than an alias table once many draws
  are wanted from one fixed distribution, exactly the trade-off Part 3 makes.

<details>
<summary>Checks (runnable)</summary>

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
