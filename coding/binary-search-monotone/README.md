# Binary Search on Bitonic Arrays and Monotone Functions

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · binary search | ★★☆☆☆ | Medium | RE · RS · SWE · MLE | binary-search, bitonic-array, exponential-search, monotone-functions, lower-bounds, bisection | 3 parts / 45 min | Skills interview |
<!-- meta:end -->

## Problem

An array `a` of `n >= 1` integers is *bitonic* if there is a peak index `p` (`0 <= p <= n - 1`) with
`a[0] < a[1] < ... < a[p]` and `a[p] > a[p + 1] > ... > a[n - 1]`; either side of the peak may be empty, so
a strictly increasing array (`p = n - 1`) and a strictly decreasing one (`p = 0`) both count as bitonic.

### Part 1 — Minimum and maximum of a bitonic list

```py
def bitonic_min_max(a: list[int]) -> tuple[int, int]: ...
```

`a` is bitonic, as defined above. Return `(min(a), max(a))`: the minimum in O(1) time, the maximum — the
value at the peak — in O(log n) comparisons between elements of `a`.

```text
a = [1, 3, 8, 12, 9, 5, 2]
bitonic_min_max(a) == (1, 12)
```

### Part 2 — Invert a strictly increasing function on the positive integers

`f` is a *strictly increasing* function from the positive integers to the integers: `x < x'` implies
`f(x) < f(x')` for all positive integers `x, x'`. No upper bound on the argument is assumed, and
`f` may be expensive to evaluate, so an algorithm's cost is measured in the number of calls it makes to
`f`, not in arithmetic or comparisons between the results.

```py
from collections.abc import Callable

def inverse(f: Callable[[int], int], y: int) -> int | None: ...
def inverse_floor(f: Callable[[int], int], y: int) -> int | None: ...
```

`inverse(f, y)` returns the positive integer `x` with `f(x) == y`, or `None` if no positive integer maps
to `y`. `inverse_floor(f, y)` returns the largest positive integer `x` with `f(x) <= y`, or `None` if even
`f(1) > y` (no positive integer satisfies the inequality at all).

```text
inverse(lambda x: x * x, 49) == 7
inverse(lambda x: x * x, 50) is None
inverse(lambda x: x + 5, 5) is None      # x would have to be 0, which is not a positive integer
inverse(lambda x: x + 5, 6) == 1
inverse(lambda x: x - 100, 5) == 105
inverse_floor(lambda x: x * x, 50) == 7
```

### Part 3 — Plateaus, and root-finding over the reals

**(a)** `a` again has `n >= 1` elements, but is now only *non-strictly* bitonic: a peak index `p`
(`0 <= p <= n - 1`) with `a[0] <= a[1] <= ... <= a[p] >= a[p + 1] >= ... >= a[n - 1]`, so two adjacent equal
elements — a *plateau* — are now allowed on either slope.

```py
def bitonic_max_with_plateaus(a: list[int]) -> int: ...
```

The minimum can still be found in O(1) time. Show that no algorithm can find the maximum while
reading fewer than `n` elements of `a` in the worst case, and give an O(n) algorithm that does.

```text
a = [0, 0, 0, 1, 0, 0, 0, 0]
```

This is non-strictly bitonic with peak index 3 (`a[3] = 1`), so `bitonic_max_with_plateaus(a) == 1`; every
other element is 0.

**(b)** `g` is a function on the reals that is *continuous* — informally, its graph has no jumps or gaps,
so it takes every value between any two of its values — and *strictly increasing* on `[lo, hi]`, with
`g(lo) <= y <= g(hi)`. Continuity guarantees at least one root `x*` with `g(x*) == y`; strict monotonicity
makes it unique.

```py
def bisect_root(g: Callable[[float], float], y: float, lo: float, hi: float, eps: float) -> float: ...
```

Return a value `x` with `|x - x*| <= eps`. Derive the number of evaluations of `g` this requires, as a
function of `hi - lo` and `eps`, and make the procedure terminate in floating-point arithmetic even when
`eps = 0`. Since `x*` need not be a floating-point number, `eps = 0` cannot be met exactly in general; the
procedure must then return an `x` as close to `x*` as the floating-point evaluation of `g` can resolve.

```text
g(x) = x ** 3, y = 30, lo = 0, hi = 4, eps = 0.001
# the true root is x* = 30 ** (1 / 3) ≈ 3.10723
bisect_root(g, 30, 0, 4, 0.001) ≈ 3.10723   # within 0.001 of x*
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points are worth confirming before coding: whether the array or function in front of you is strictly
monotone or only allows ties — Parts 1 and 2 assume strict monotonicity, and Part 3(a) drops it for the
array — and, for Part 2, whether the answer's magnitude is bounded in advance; it is
not, so a search over a fixed range such as `[0, y]` is not safe in general, as the derivation below shows.

### Part 1

Each iteration replaces `[lo, hi]` — the range of indices guaranteed to contain the peak — with one of its
two halves, using a single comparison between neighbouring elements. With `lo < hi` and
`mid = (lo + hi) // 2`, integer division gives `mid <= hi - 1`, so `a[mid + 1]` is always a valid read.
`a[mid] < a[mid + 1]` means the array is still rising at `mid`, so the peak lies at `mid + 1` or beyond:
`lo = mid + 1`. Otherwise `a[mid] > a[mid + 1]` — equality never happens, since each side of a bitonic
array is *strictly* monotone — so the peak is at `mid` or to its left: `hi = mid`. Both branches keep the
peak inside `[lo, hi]` and shrink `hi - lo` to about half its previous value, so the loop reaches `lo ==
hi` after at most $\lceil \log_2 n \rceil$ iterations, one comparison of `a` each. The minimum needs no
search at all: every element on the increasing side exceeds `a[0]`, and every element on the decreasing
side exceeds `a[-1]`, so the overall minimum is whichever end is smaller.

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

Comparing forward, `a[mid]` against `a[mid + 1]`, is what makes this safe. Keeping the same floor-rounded
`mid` but comparing backward instead — `a[mid] > a[mid - 1]`, moving `lo` to `mid` on a hit and `hi` to
`mid - 1` otherwise — can loop forever once `hi == lo + 1`: floor division then makes `mid == lo`, so
whenever the comparison succeeds, the branch that sets `lo = mid` leaves `lo` exactly where it was, and the
loop never reaches `lo == hi`. The
checks below exercise this on a concrete array.

On the worked example `[1, 3, 8, 12, 9, 5, 2]`, the search runs:

```text
lo=0, hi=6: mid=3, a[3]=12 > a[4]=9  -> hi=3   (the peak is at index 3 or to its left)
lo=0, hi=3: mid=1, a[1]=3  < a[2]=8  -> lo=2   (the peak is to the right of index 1)
lo=2, hi=3: mid=2, a[2]=8  < a[3]=12 -> lo=3   (the peak is to the right of index 2)
lo=3, hi=3: lo == hi, stop           -> peak index 3, a[3] = 12
```

### Part 2

Doubling a candidate `hi = 1, 2, 4, ...` while `f(hi) < y` finds a bracket that must contain the answer, using
few evaluations regardless of how large it turns out to be — *exponential* (or *galloping*) search. Write $m$
for the value `inverse_floor` must return (the largest positive integer with $f(m) \le y$; the guard
`f(1) > y` above rules out the case where no such integer exists). Suppose the doubling loop performs
$k \ge 1$ doublings before stopping, so it evaluates `f` at $1, 2, 4, \ldots, 2^k$ and then stops because
$f(2^k) \ge y$, having found $f(2^{k-1}) < y$ on the previous check. Since `f` is strictly increasing on the
integers: $f(2^{k-1}) < y$ forces $2^{k-1} \le m$ (an integer with `f` below `y` cannot exceed the *largest*
integer with `f` at most `y`); and every $x > 2^k$ has $f(x) > f(2^k) \ge y$, so $m \le 2^k$. Hence
$lo = 2^{k-1}$, $hi = 2^k$ bracket $m$ on both sides. The gap `hi - lo` starts at $2^{k-1}$ and exactly halves
on every later iteration — both `mid - lo` and `hi - mid` equal half the gap whenever the gap is itself a
power of two, which it stays throughout — reaching $1$ after $k - 1$ further evaluations; at that point `lo`
and `hi` are adjacent, and the already-cached value of `f(hi)` (from whichever evaluation last set it) decides
between them with no new call to `f`. Doubling cost $k + 1$ evaluations and narrowing cost $k - 1$ more, for
$2k$ in total — and, in the excluded case $k = 0$ (`f(1)` already equal to `y`, so the doubling loop never
runs at all), exactly $1$.

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

Writing $b$ for `m.bit_length()` (the number of bits of $m$'s binary representation, so
$2^{b - 1} \le m < 2^b$) turns this into a bound that depends only on $m$, not on the unknown $k$:
$2^{k - 1} \le m < 2^b$ gives $k - 1 < b$, i.e. $k \le b$ (both are integers), so $2k \le 2b$ — at most
$2b$ evaluations for either function, and the $k = 0$ case ($1 \le 2b$, since $b \ge 1$ whenever $m \ge 1$)
fits the same bound. `inverse` spends nothing beyond `inverse_floor`'s own search: `_locate` already
computed `f` at the value it returns, and that result is reused rather than recomputed, so the one extra
step is a comparison, not another call to `f`.

A simpler-looking alternative — binary search directly on the fixed range `[0, y]`, skipping the doubling
phase — is only sound when the answer is guaranteed to lie in that range, which needs `f(x) >= x` for
every `x`. A strictly increasing, integer-valued `f` satisfies `f(x) >= f(1) + (x - 1)`, since each unit
step up increases the value by at least 1, so `f(x) >= x` whenever `f(1) >= 1` — true of `x * x` and
`x + 5` above, but not of `f(x) = x - 100`: `f(1) = -99 < 1`, and `inverse(lambda x: x - 100, 5)` needs
`x = 105`, far outside `[0, 5]`. Exponential search assumes no such thing, since it grows its own bracket
until `f` itself reports that the bracket is large enough.

### Part 3

**(a)** The peak-finding comparison from Part 1 relies on strict inequality: `a[mid] < a[mid + 1]` and
`a[mid] > a[mid + 1]` were the only two possibilities. With plateaus, `a[mid] == a[mid + 1]` is now
possible, and it is uninformative — both neighbours are on the same, flat part of *some* valid non-strictly
bitonic array, so neither half of the range can be discarded. This is not merely a limitation of one
algorithm: no algorithm — of any shape, adaptive or not — can find the maximum while leaving even one
element of `a` unread, in the worst case. Suppose some algorithm always stops after reading at most
`n - 1` of the `n` elements. Run it against an adversary that answers every read with `0` for as long as
possible; when the algorithm stops, some index `q` was never read. The all-zero array is non-strictly
bitonic (trivially: `0 <= 0 <= ... <= 0 >= 0 >= ... >= 0`) with maximum `0`, and so is the same array with a
single `1` written at `q` instead (non-decreasing up to `q`, non-increasing from `q`, since one element
does not break either chain), but its maximum is `1`. The algorithm's sequence of reads and its final
answer are identical on both arrays — it never read `q`, and every other index reads `0` on both — so
whichever value it returns is wrong for one of the two. Reading all `n` elements is therefore necessary,
and trivially sufficient: `max(a)` is a single O(n) pass.

```python
def bitonic_max_with_plateaus(a: list[int]) -> int:
    return max(a)   # NOTE: no shortcut is possible in the worst case -- see the adversary argument above
```

**(b)** Bisection halves a bracket `[lo, hi]` known to contain the root, using the sign of `g(mid) - y` to
decide which half still does: `g` strictly increasing means `g(mid) < y` puts the root strictly above `mid`,
and `g(mid) >= y` puts it at `mid` or below. After `k` halvings the bracket has width `(hi - lo) / 2 ** k`, so
$k = \lceil \log_2((\mathrm{hi} - \mathrm{lo}) / \mathrm{eps}) \rceil$ evaluations already shrink it to at
most `eps`, at which point either endpoint is within `eps` of the true root. Halving forever would never reach
a bracket of width exactly `0` in floating-point arithmetic — the midpoint of two adjacent floating-point
numbers rounds to one of them — so the loop's real termination condition is not a width comparison but the
bracket failing to shrink at all, which every input reaches in finitely many steps purely because there are
only finitely many floating-point numbers between any `lo` and `hi`.

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

### Follow-ups

- **Binary search on the answer.** Whenever a yes/no property flips exactly once as a candidate value
  increases — "does this batch size fit in memory", "is the loss still finite at this learning rate", "does
  this threshold still keep recall at the target" — binary-searching the candidate value itself turns a
  search over a monotone property into Part 2's search (galloping while no upper bound is known, then
  halving), with one evaluation of the property (a training step, a memory allocation) standing in for one
  call to `f`.
- **Ternary search needs strictness.** Ternary search locates the maximum of a *unimodal* function by
  comparing two interior points and discarding the side both agree is going the wrong way; a flat stretch
  below the peak can make the two points tie with no information about which side the true maximum is on, the
  same failure Part 3(a) proves is unavoidable — ternary search, like Part 1's peak search, needs strict
  monotonicity on each side to be correct.
- **Newton's method versus bisection.** Bisection halves the bracket every step regardless of `g`, giving
  linear convergence; Newton's method steps to where the tangent line at the current guess crosses zero,
  which roughly doubles the number of correct digits every step (quadratic convergence) whenever the guess
  is already close and `g` is smooth with a nonzero derivative there — but it can overshoot the bracket
  entirely or cycle without converging when `g` is far from linear near the guess or its derivative is
  near zero, exactly the cases where bisection's guaranteed-to-shrink bracket is the safer choice.

<details>
<summary>Checks (runnable)</summary>

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
