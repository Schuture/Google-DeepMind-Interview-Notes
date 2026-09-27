# CS Fundamentals Quiz: Memory, Concurrency and Floating Point

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Quiz · oral, with small demonstrations | ★★☆☆☆ | Medium | SWE · Intern · RE | oop, memory-management, garbage-collection, race-conditions, deadlock, gil, cache-locality, floating-point, amortised-analysis | 10 questions / 45 min | Skills interview |
<!-- meta:end -->

## Problem

Throughout, complexity bounds are worst case unless a question says "average", and any specific runtime
behaviour described (the GIL, CPython's collector, list growth) is CPython 3.11's, not a general property of
the Python language.

### Programming and memory

**Q1.** What is object-oriented programming (OOP)? Define *encapsulation*, *inheritance*, *polymorphism* and
*abstraction* in one sentence each. Then contrast these two designs for a LIFO (last-in, first-out) stack of integers:

```py
class StackInherit(list):
    """Reuses list by inheritance: is-a list."""
    def push(self, x: int) -> None: ...


class StackCompose:
    """Reuses list by composition: has-a list."""
    def push(self, x: int) -> None: ...
    def pop(self) -> int: ...
    def peek(self) -> int: ...
```

State which design is preferable, and why, in terms of the four ideas above.

**Q2.** Define *stack* and *heap* as regions of memory, and contrast how each is allocated and freed. In a
garbage-collected language, what is a *memory leak*, given that the collector never leaves unreachable memory
allocated? Give two concrete causes. In C++, what ownership does each of `std::unique_ptr`, `std::shared_ptr`
and `std::weak_ptr` express, and what specific problem does `weak_ptr` solve that `shared_ptr` alone cannot?

**Q3.** Contrast *reference counting* and *tracing garbage collection* as strategies for reclaiming memory,
and state the one situation where an object's reference count never reaches zero even though the object
itself has become unreachable from the rest of the program. Explain how CPython combines the two — what
triggers its generational cycle collector, and which objects it has to consider. Demonstrate an object that
becomes part of such a cycle: show that deleting every name that points into it does not free it, and that an
explicit collection pass does.

### Concurrency

**Q4.** Two threads run concurrently on a shared variable `x`, initially `0`, each executing three steps —
*load* (read `x` into a private register), *add* (increment the register by 1), *store* (write the register
back to `x`):

```text
thread 1:  L1: r1 = x        A1: r1 = r1 + 1        S1: x = r1
thread 2:  L2: r2 = x        A2: r2 = r2 + 1        S2: x = r2
```

Each thread's own steps keep their order (`L` before `A` before `S`), but the six steps may otherwise
interleave in any order. List every possible final value of `x` over all such interleavings, and identify
exactly which interleavings produce each value. Then explain how protecting the three steps with a mutex, or
replacing them with one atomic *fetch-and-add* instruction, fixes it.

**Q5.** Define the Python *global interpreter lock* (GIL) — what single thing does it serialise? Given a
program with several threads, explain why adding threads speeds up wall-clock time for an I/O-bound workload
(e.g. several threads each waiting on a network socket) but not for a CPU-bound workload written in pure
Python (e.g. each thread summing a large range in a loop). Name two things that let CPU-bound work actually
use multiple cores from Python.

**Q6.** State the four Coffman conditions that must all hold for a deadlock to exist. Define the *wait-for
graph* of a set of threads and resources — one node per thread, an edge $T_i \to T_j$ whenever $T_i$ is
blocked waiting for a resource currently held by $T_j$ — and state the condition on this graph that is
equivalent to deadlock. Explain, in terms of which Coffman condition it removes, why having every thread
acquire locks in one fixed global order prevents deadlock.

### Performance and numbers

**Q7.** A two-dimensional array `A` with `R` rows and `C` columns of 8-byte elements is stored in row-major
order, so `A[i][j]` sits at byte offset `(i * C + j) * 8` from the start. The machine's cache is organised
into 64-byte *lines*: accessing any byte not currently cached is a *cache-line load* that fetches the whole
64-byte line containing it and evicts the least-recently-used line if the cache — here, only 4 lines — is
full; accessing a byte inside an already-cached line is free. For `R = 8`, `C = 16`, give the exact number of
cache-line loads made by traversing `A` row by row (`for i: for j: visit(i, j)`) and by traversing it column
by column (`for j: for i: visit(i, j)`), and explain the difference in terms of *stride* (the byte distance
between consecutive accesses).

**Q8.** In IEEE 754 double precision, explain why `0.1 + 0.2 != 0.3`. Define *machine epsilon* and give its
value for doubles. Define *catastrophic cancellation*, and explain why computing $1 - \cos x$ directly loses
accuracy for small $x$ while the algebraically equal $2\sin^2(x/2)$ does not. Give the tolerance-based test
that should replace `a == b` when comparing two floating-point results, and explain why it needs both a
relative and an absolute term.

**Q9.** For fp32, fp16 and bf16 (each a sign bit plus an exponent field, which sets the representable range,
and a mantissa/fraction field, which sets the precision), give the number of exponent and mantissa bits and,
from them, the largest finite value each format can represent. Give *machine epsilon* for each of the three
formats — the gap between 1 and the next representable value — as a function of its mantissa bits. Explain
why bf16 is generally preferred over fp16 for training neural networks, and what specific numerical failure
*loss scaling* is designed to prevent when fp16 is used instead.

**Q10.** Define the *load factor* of a hash table, and state its average-case and worst-case time complexity
for lookup, each in terms of $n$ (the number of stored keys). Explain what causes the worst case, and why
resizing — doubling the capacity and rehashing every key — keeps the average case at $O(1)$ as $n$ grows.
Derive that, over $n$ insertions into a table that starts at capacity 1 and doubles whenever it is full, the
total number of element copies performed by all the resizes combined is less than $2n$ — the same doubling
argument that gives Python's `list.append` its amortised $O(1)$ cost.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two things worth pinning down aloud before answering: complexity bounds below are worst case unless a
question says "average", and every claim about the GIL, reference counting or list growth targets CPython
specifically — other implementations (PyPy, with its own JIT and collector; Jython; GraalPy) are free to
behave differently, and do.

### Programming and memory

**Q1.** Object-oriented programming (OOP) organises a program as a set of interacting objects, each bundling
state together with the code that operates on it, and builds code reuse and polymorphic behaviour around
classes of such objects. *Encapsulation* is bundling an object's data with the operations that act on it and
hiding that data behind those operations, so it can change only through a controlled interface.
*Inheritance* is a subclass acquiring the fields and methods of a superclass, modelling an "is-a"
relationship. *Polymorphism* is calling the same method name on objects of different types and having each
one run its own type-specific implementation, chosen at run time. *Abstraction* is exposing an object's
essential behaviour through a simplified interface while hiding how that behaviour is implemented.
`StackCompose` is the better design: `StackInherit(list)` is-a list, so it inherits every list method,
including `insert`, `__setitem__` and `sort`, none of which respect the stack's only invariant (elements
enter and leave from one end) — encapsulation is broken the moment a caller reaches for a method that was
never meant to be part of the stack's interface, as the checks below demonstrate concretely. `StackCompose`
has-a list but exposes only `push`, `pop` and `peek`, so the invariant cannot be violated through its
interface at all. The general rule: prefer composition when a class needs another class's *implementation*
without wanting to commit to (and expose) its full *interface*; reach for inheritance only when there is a
genuine is-a relationship and a subclass instance is meant to be usable anywhere the superclass is expected.

**Q2.** The *stack* is the memory region holding call frames — parameters, local variables and the return
address for each active function call — pushed on a call and popped on return, in strict last-in-first-out
order; allocation and deallocation are just moving a stack pointer, so both are essentially free, and a
value's lifetime is tied exactly to the scope that created it. The *heap* is memory for objects whose size or
lifetime cannot be tied to a single call frame — allocated explicitly (`malloc`/`new`) or by a garbage
collector, and freed explicitly, by a collector, or (with C++ smart pointers) when an owning value goes out
of scope; bookkeeping makes both allocation and deallocation more expensive than on the stack. In a
garbage-collected language the collector only ever frees memory that has become *unreachable*, so a leak
there is not unreachable memory going unfreed — it is memory that stays reachable, and hence correctly kept
alive, along some reference chain the program no longer actually needs. Two concrete causes: an unbounded
cache that stores every result it has ever computed with no eviction policy, so its size grows for as long as
the process runs; and a listener registered on a long-lived publisher (e.g. a bound method passed to
`subscribe`) that is never unsubscribed, so the publisher's listener list keeps the subscriber — and
everything its closure captures — alive long after the code that created it is done with it, as the checks
below demonstrate for both. In C++, `unique_ptr` expresses *exclusive* ownership: it cannot be copied, only
moved, and its destructor frees the pointee exactly once, when the single owning `unique_ptr` goes out of
scope. `shared_ptr` expresses *shared* ownership through reference counting (a control block with an atomic
strong count): copying one increments the count, destroying one decrements it, and the pointee is freed when
the count reaches zero. `weak_ptr` is a *non-owning* observer of a `shared_ptr`'s pointee: holding one does
not affect the strong count, and using it safely means calling `.lock()`, which returns an empty `shared_ptr`
if the object has already been freed. It exists because `shared_ptr` alone uses plain reference counting with
no cycle collector: two objects holding `shared_ptr`s to each other (a parent and child that each need a live
pointer to the other) form a cycle whose strong count never reaches zero, even after every external reference
is gone, leaking both forever — the same failure mode reference counting has in Q3, but here with no tracing
collector as a backstop, so the cycle must be broken by hand: the child holds a `weak_ptr` back to its parent
instead of a `shared_ptr`.

**Q3.** *Reference counting* attaches to every object a count of how many references point to it, incremented
on each new reference and decremented when one goes away; the object is freed the instant its count reaches
zero. It reclaims memory immediately and deterministically, but it has exactly one blind spot: a *cycle* of
objects that reference each other, directly or through a chain, but that nothing outside the cycle points to
— every object in it still has a positive count, contributed by another member of the same cycle, so no count
ever reaches zero and the whole cycle leaks even though it is unreachable from the rest of the program.
*Tracing garbage collection* instead starts from a set of roots (global variables, the call stack) and walks
the reachability graph, marking every object it can reach; anything left unmarked is garbage regardless of
how its internal references are wired, so cycles are reclaimed as easily as anything else, at the cost of a
scan over live memory rather than a per-object decrement. CPython uses reference counting as its primary
mechanism — every object carries a count, and the ordinary case of a value going out of scope frees it
immediately — and layers a generational tracing *cycle collector* (the `gc` module) on top, purely as a
backstop for what reference counting cannot do. That collector only has to consider "container" objects
capable of participating in a cycle (lists, dicts, class instances, closures and the like — anything that can
hold a reference to another object); it groups them into three generations by how many collections they have
survived, and only scans a generation once enough container allocations have accumulated in it since the last
scan, so most objects (numbers, strings, non-cyclic instances) never cost it anything. The checks below build
two objects that reference each other, delete every name pointing into the cycle, and show a weak reference
to one of them is still alive — reference counting could not free it — until an explicit `gc.collect()` runs,
after which it is gone; a third, non-cyclic object is freed the instant its last name is deleted, with no
collection needed at all.

### Concurrency

**Q4.** The six steps interleave in $\binom{6}{3} = 20$ ways that respect each thread's own order (choosing
which 3 of the 6 slots belong to thread 1 fixes the rest). Only two final values of `x` are possible: **1**
or **2** — never $0$, since `x` only ever increases, and never above $2$, since only one `add` per thread ever
executes. The value is **2** exactly when the schedule is fully serial — one thread's load, add and store all
complete before the other thread's load runs — because then the second thread's load reads the first
thread's already-stored result and adds 1 to it; there are exactly 2 such schedules (thread 1 entirely then
thread 2, or the reverse) out of the 20. Every other schedule lets *both* loads read `x` while it is still
$0$ (the second thread's load slips in before the first thread's store), so both threads compute $0 + 1 = 1$
locally and whichever store lands last simply overwrites the other with the same value — a classic *lost
update*, where one of the two increments has no effect at all. Protecting the three steps with a mutex forces
mutual exclusion between the two threads' critical sections, which collapses the schedule space down to
exactly the two fully-serial orderings, both of which give the correct $2$. A single atomic fetch-and-add
instruction achieves the same thing at the hardware level: load, add and store happen as one indivisible step
that no other core can see partway through, so again only the two (now instruction-level) serial orderings
are possible.

**Q5.** The GIL is a single mutex inside the CPython interpreter that only one OS thread may hold while
executing Python bytecode; it exists because CPython's internal bookkeeping — most visibly the reference
counts from Q3 — is not safe to update from multiple threads at once without it. Its effect is that two
Python threads never execute Python bytecode at the same instant, even on a machine with many cores: at most
one thread is ever "inside" the interpreter. A thread blocked on I/O (a socket, a disk read, `time.sleep`)
releases the GIL for the duration of the underlying blocking call and reacquires it only once that call
returns, so while it waits, another thread is free to run — several I/O-bound threads therefore overlap their
*waiting* time, and wall-clock throughput improves even though execution itself is never simultaneous. A
thread doing pure-Python computation never blocks voluntarily, so it holds the GIL continuously except for
periodic forced hand-offs (CPython interrupts the running thread and offers the GIL to another waiting one on
a fixed switch interval, by default every 5 ms); this only *time-slices* a single core across the threads
rather than running them on separate cores, so total throughput on a CPU-bound pure-Python workload does not
improve with more threads, and typically gets slightly worse once lock hand-off is accounted for. Two ways to
actually use multiple cores from Python: run separate OS *processes* (`multiprocessing`) — each has its own
interpreter and its own GIL, so they run truly in parallel, at the cost of separate memory spaces and
explicit inter-process communication; or call into *native code that releases the GIL around its
computational core*, such as NumPy's C loops, which wrap the actual number-crunching in calls that drop the
GIL — so several Python threads that are each waiting on a NumPy call can have their underlying C work
genuinely overlap on separate cores.

**Q6.** The four Coffman conditions, all of which must hold simultaneously for deadlock to be possible: (1)
*mutual exclusion* — at least one resource is held in a mode that only one thread can hold at a time; (2)
*hold and wait* — a thread holds at least one resource while it waits to acquire another; (3) *no
preemption* — a resource cannot be forcibly taken from the thread holding it, only released voluntarily; (4)
*circular wait* — there is a cycle of threads $T_1, \dots, T_k$ where each $T_i$ is waiting for a resource
held by $T_{(i \bmod k) + 1}$. In the wait-for graph, the system is deadlocked exactly when the graph
contains a cycle — a set of threads each waiting, directly or transitively, on one of the others in the same
set, so none of them can ever be the one to make progress and release what the next one needs. Requiring
every thread to acquire its locks in one fixed global order (say, by an ID assigned once to each lock)
removes the *circular-wait* condition specifically, and by Coffman that alone is enough to make deadlock
impossible: if a thread holds lock $a$ and is waiting on lock $b$, the global order forces $a < b$ (it must
have acquired the smaller one first), so every edge in the wait-for graph goes from a smaller lock ID to a
larger one; a cycle would need the IDs along it to strictly increase all the way around back to where it
started, which is impossible under a fixed total order. The checks below confirm this both on hand-built
examples (a 2-cycle, a 3-cycle, and two acyclic graphs) and by comparing a cycle detector against an
independent reachability computation on hundreds of random graphs, and separately confirm that graphs whose
edges are constrained to respect one fixed node order never contain a cycle.

### Performance and numbers

**Q7.** Row-major traversal makes exactly **16** cache-line loads; column-major traversal makes exactly
**128** — eight times as many, essentially one for every single visit. With 8-byte elements and 64-byte
lines, one line holds 8 consecutive elements of a row, i.e. 8 consecutive values of `j` for fixed `i`.
Row-major order (`for i: for j:`) has stride 8 bytes between consecutive accesses — it walks straight along
the 8 elements of a line before the next access falls in a new one, so it loads each line exactly once and
never has to revisit an old one; the total is one load per 8 elements visited, $8 \times 16 / 8 = 16$, and
this holds even with a cache as small as a single line, since nothing is ever reused. Column-major order
(`for j: for i:`) has stride $C \times 8 = 128$ bytes — each step jumps to the *next row*, a different line,
specifically line $2i + \lfloor j/8 \rfloor$ for row `i`:

```text
column-major, line = (i * 16 + j) // 8, first 6 visits:
  (i=0,j=0) line 0   MISS
  (i=1,j=0) line 2   MISS
  (i=2,j=0) line 4   MISS
  (i=3,j=0) line 6   MISS    <- cache (capacity 4) now holds {0, 2, 4, 6}
  (i=4,j=0) line 8   MISS    <- a 5th distinct line: line 0 (least recently used) is evicted
  (i=5,j=0) line 10  MISS
```

For a fixed `j`, the 8 rows already touch 8 distinct lines, and the cache holds only 4, so by the time row 4's
access is made, row 0's line has already been evicted by rows 1–3 — and it stays evicted, because moving to
`j + 1` (while `j < 8`) revisits those exact same 8 lines in the exact same order, always more than the 4 the
cache can hold. With no reuse ever surviving to be exploited, every visit is a miss, giving one load per
element rather than one per 8: $8 \times 16 = 128$.

**Q8.** `0.1 + 0.2 != 0.3` because none of the three decimal literals is exactly representable in IEEE 754
double precision — $0.1 = 1/10$ has no finite binary expansion, just as $1/3$ has no finite decimal one — so
each is rounded to its nearest representable double before any arithmetic happens, and the double closest to
$0.1$ plus the double closest to $0.2$, itself rounded to the nearest double, does not land exactly on the
double closest to $0.3$; the checks below confirm this with exact rational arithmetic, which has no rounding
of its own to blur the comparison. *Machine epsilon* $\varepsilon$ is the gap between $1.0$ and the next
larger representable value — equivalently, the smallest positive $\varepsilon$ with $1 + \varepsilon \ne 1$ in
floating-point arithmetic; for doubles (52 explicit mantissa bits) $\varepsilon = 2^{-52}$, and representable
numbers near any value $x$ are spaced about $\varepsilon$ apart *relative to* $x$, twice as sparse every time
$|x|$ doubles. *Catastrophic cancellation* is what happens when two floating-point numbers that are nearly
equal are subtracted: almost all of their significant digits cancel, so the rounding error each one already
carried (up to about $\varepsilon$ relative to its own size) is left standing relative to the much smaller
true result — the number of correct digits remaining falls by roughly how many digits cancelled. Computing
$1 - \cos x$ literally does exactly this for small $x$: $\cos x$ is first rounded to a double extremely close
to $1$ (its expansion is $1 - x^2/2 + \dots$, and for small $x$ that tiny $x^2/2$ term can be completely
absorbed by rounding), and subtracting two near-equal doubles then leaves that fixed absolute rounding error
standing relative to a true answer of order $x^2$ — by $x = 10^{-8}$, `cos(x)` has already rounded to exactly
`1.0`, so `1 - cos(x)` is exactly `0.0`: total cancellation, not one correct digit left. The identity
$1 - \cos x = 2\sin^2(x/2)$ computes the same quantity without ever subtracting two close numbers —
$\sin(x/2)$ is computed directly, accurate to the ordinary $O(\varepsilon)$ relative error, then squared and
doubled, with no cancellation at any step — so it stays accurate down to values of $x$ far smaller than where
the direct formula has already lost every digit. Comparing two floating-point results should use a combined
tolerance, `abs(a - b) <= atol + rtol * abs(b)` (as `math.isclose`/`numpy.isclose` do), rather than `a == b` or
a purely relative test: the relative term matters when the values are large (a fixed number of correct
digits, not a fixed absolute error, is all floating point promises), but it becomes meaningless as `b`
approaches $0$ — a purely relative test can reject two numbers that are both negligibly close to zero simply
because they differ by a large *factor* — so the absolute term is needed to say what "close to zero" itself
means.

**Q9.** fp32 has 8 exponent bits and 23 mantissa bits, giving a largest finite value of
$(2 - 2^{-23}) \times 2^{127} \approx 3.4028 \times 10^{38}$. fp16 has 5 exponent bits and 10 mantissa bits,
giving $(2 - 2^{-10}) \times 2^{15} = 65504$ exactly. bf16 ("brain float16") has 8 exponent bits — the same as
fp32 — and only 7 mantissa bits, giving $(2 - 2^{-7}) \times 2^{127} \approx 3.3895 \times 10^{38}$:
essentially fp32's range with its mantissa truncated to 7 bits, which is exactly how bf16 is produced in
practice (and in the checks below) — keep the top 16 bits of a value's fp32 bit pattern and drop the rest.
Machine epsilon is $2^{-23}$ for fp32, $2^{-10}$ for fp16, and $2^{-7}$ for bf16 — in each case, one unit in
the last mantissa bit. bf16 is generally preferred for training because it keeps fp32's full exponent range:
any activation or gradient magnitude that fits in fp32 fits in bf16 too, with no risk of overflowing to
infinity or underflowing to zero purely from the cast, while the precision it sacrifices is well tolerated by
gradient-based training, where minibatch sampling noise already dominates any single value's rounding error;
casting fp32 to bf16 is consequently just dropping the bottom 16 mantissa bits, with the exponent untouched.
fp16's exponent is only 5 bits, so its range is far narrower — its smallest normal value is about
$6.1 \times 10^{-5}$, and subnormals reach down only to about $6 \times 10^{-8}$ — and gradients smaller than
that, which routinely occur deep in a network or after a loss with a small scale, are flushed to exactly
zero the moment they are stored in fp16, silently destroying that part of the learning signal. *Loss scaling*
prevents exactly this: multiplying the loss, and hence by the chain rule every gradient, by a large constant
power of two before the backward pass shifts the whole distribution of gradient magnitudes up out of the
underflow region — the checks below use a gradient of $10^{-8}$, which fp16 stores as exactly zero directly
but recovers correctly once scaled by $2^{16}$ before casting and divided back out afterwards — and
multiplying by a power of two is exact in floating point (it only shifts the exponent field), so this costs
no additional rounding error of its own.

**Q10.** The load factor $\alpha = n/m$ is the number of stored keys $n$ divided by the table's capacity $m$
(its number of buckets). Average-case lookup is $O(1)$: with a hash function that spreads keys roughly
uniformly and a load factor kept below some fixed bound, each bucket holds $O(1)$ keys on average, so finding
a key costs an expected constant number of comparisons regardless of $n$. Worst-case lookup is $O(n)$: if the
hash function sends many, or all, of the $n$ keys to the same bucket — whether by bad luck or a deliberately
bad hash function — that bucket degenerates into a plain list that has to be scanned linearly, and resizing
does not fix this, since a bad hash function keeps sending everything to the same place however large the
table grows, as confirmed below by forcing every key through a constant hash function. Resizing — allocating
a new, larger array of buckets and rehashing every existing key into it once $\alpha$ crosses a fixed
threshold — is what keeps the *average* case at $O(1)$ as $n$ grows without bound: without it, $\alpha$ would
grow linearly with $n$, and so would the average bucket length. For the cost of resizing itself: start an
empty table at capacity 1, and double its capacity — copying every element it currently holds into the new
array — exactly when an insertion would exceed capacity. Over $n$ insertions, let $m$ be the smallest power
of two with $m \ge n$; a resize happens exactly when the table's size reaches each of $1, 2, 4, \dots, m/2$ in
turn (the resize at size $m/2$ grows the capacity to $m$, comfortably covering the rest of the $n$ insertions
with no further resize), and each one copies exactly that many elements, so the total number of copies is the
geometric sum

$$1 + 2 + 4 + \cdots + \frac{m}{2} = m - 1.$$

Because $m$ is the *smallest* power of two at least $n$, the previous power of two, $m/2$, must be smaller
than $n$ — otherwise $m$ would not have been smallest — so $m < 2n$, and the total is $m - 1 < 2n$: fewer
than two copies per insertion, however large $n$ gets, confirmed below both by this exact formula and by
simulation. This is precisely the argument behind Python's own `list.append`: CPython over-allocates on every
growth rather than growing by a fixed amount, so the same geometric-series reasoning applies — its actual
growth factor is smaller (asymptotically about $1.125\times$, not $2\times$), which still gives amortised
$O(1)$, just with a larger constant ($9n$ rather than $2n$ copies overall, confirmed below), since a smaller
growth factor means more frequent, and hence more total, resizes for the same $n$.

<details>
<summary>Checks (runnable)</summary>

```python
import gc
import itertools
import math
import sys
import weakref
from collections import Counter, OrderedDict
from fractions import Fraction

import numpy as np

rng = np.random.default_rng(2026)

# ---- Q1: composition vs inheritance -- StackInherit exposes list's whole interface, StackCompose does not
class StackInherit(list):
    def push(self, x: int) -> None:
        self.append(x)


class StackCompose:
    def __init__(self) -> None:
        self._items: list[int] = []

    def push(self, x: int) -> None:
        self._items.append(x)

    def pop(self) -> int:
        return self._items.pop()

    def peek(self) -> int:
        return self._items[-1]


si = StackInherit()
si.push(1)
si.push(2)
assert hasattr(si, "insert")              # inherited from list: breaks the LIFO invariant
si.insert(0, 99)                          # NOTE: nothing in StackInherit's own code stops this
assert list(si) == [99, 1, 2]
sc = StackCompose()
sc.push(1)
sc.push(2)
assert not hasattr(sc, "insert") and not hasattr(sc, "append")   # only push/pop/peek are reachable
assert sc.pop() == 2 and sc.peek() == 1

# ---- Q2: two leak patterns -- an unbounded cache, and a listener that is never unsubscribed
cache: dict[int, int] = {}


def memo(n: int) -> int:
    if n not in cache:
        cache[n] = n * n
    return cache[n]


for i in range(2000):
    memo(i)
assert len(cache) == 2000                 # every distinct call grows it forever: no eviction policy


class Publisher:
    def __init__(self) -> None:
        self._listeners: list = []

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def unsubscribe(self, fn) -> None:
        self._listeners.remove(fn)


class Subscriber:
    def on_event(self) -> None:
        pass


pub = Publisher()
sub = Subscriber()
sub_ref = weakref.ref(sub)
pub.subscribe(sub.on_event)               # a bound method keeps its underlying object alive
del sub                                   # the caller's own name is gone, but the publisher still holds a reference
assert sub_ref() is not None              # NOTE: leaked -- unreachable from the caller, but not from the publisher
pub.unsubscribe(sub_ref().on_event)
assert sub_ref() is None                  # freed the instant the last reference (the publisher's) is dropped

# ---- Q3: a reference cycle needs the tracing collector; a non-cyclic object does not
class Node:
    def __init__(self) -> None:
        self.other: "Node | None" = None


gc.disable()                              # NOTE: otherwise an unrelated threshold-triggered collection could
try:                                      #       run between `del` and the assert below, by coincidence
    a, b = Node(), Node()
    a.other, b.other = b, a               # a -> b -> a: a cycle, unreachable from any name once both are deleted
    ref_a = weakref.ref(a)
    del a, b
    assert ref_a() is not None            # reference counting alone cannot free a cycle
    collected = gc.collect()
    assert ref_a() is None                # the tracing collector finds it and frees it
    assert collected >= 2

    c = Node()                            # no cycle: an ordinary object
    ref_c = weakref.ref(c)
    del c
    assert ref_c() is None                # freed the instant its refcount hits zero -- no gc.collect() needed
finally:
    gc.enable()

# ---- Q4: exhaustive enumeration of the load/add/store interleavings of two threads
def interleavings(len_a: int, len_b: int) -> list[list[tuple[str, int]]]:
    """Every way to merge an A-sequence and a B-sequence of the given lengths, preserving each one's own order."""
    out = []
    for a_positions in itertools.combinations(range(len_a + len_b), len_a):
        a_set = set(a_positions)
        seq, ai, bi = [], 0, 0
        for i in range(len_a + len_b):
            if i in a_set:
                seq.append(("A", ai)); ai += 1
            else:
                seq.append(("B", bi)); bi += 1
        out.append(seq)
    return out


def run_schedule(schedule: list[tuple[str, int]]) -> int:
    """Steps 0/1/2 are load/add/store; a thread's register is private, x is the one shared variable."""
    x = 0
    reg = {"A": None, "B": None}
    for thread, step in schedule:
        if step == 0:
            reg[thread] = x
        elif step == 1:
            reg[thread] = reg[thread] + 1
        else:
            x = reg[thread]
    return x


schedules = interleavings(3, 3)
assert len(schedules) == math.comb(6, 3) == 20
outcomes = {run_schedule(s) for s in schedules}
assert outcomes == {1, 2}
counts = Counter(run_schedule(s) for s in schedules)
serial = [("A", 0), ("A", 1), ("A", 2), ("B", 0), ("B", 1), ("B", 2)]
serial_rev = [("B", 0), ("B", 1), ("B", 2), ("A", 0), ("A", 1), ("A", 2)]
assert run_schedule(serial) == 2 and run_schedule(serial_rev) == 2
fully_serial = [s for s in schedules if s == serial or s == serial_rev]
assert len(fully_serial) == 2 and counts[2] == 2 and counts[1] == 18   # only the 2 serial schedules avoid the lost update

# a mutex (or one atomic fetch-and-add) restricts execution to exactly the fully-serial schedules
assert {run_schedule(s) for s in fully_serial} == {2}

# ---- Q5: the GIL's default forced switch interval
assert math.isclose(sys.getswitchinterval(), 0.005)      # 5 ms, the value quoted in the answer

# ---- Q6: deadlock as a cycle in the wait-for graph, cross-checked against an independent reachability test
def wait_for_cycle_dfs(graph: dict[str, list[str]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def visit(node: str) -> bool:
        color[node] = GRAY
        for nxt in graph.get(node, []):
            if color.get(nxt, WHITE) == GRAY:
                return True
            if color.get(nxt, WHITE) == WHITE and visit(nxt):
                return True
        color[node] = BLACK
        return False

    return any(color[n] == WHITE and visit(n) for n in graph)


def wait_for_cycle_brute(graph: dict[str, list[str]]) -> bool:
    """Independent check: a full reachability closure (Floyd-Warshall); a cycle exists iff some node reaches itself."""
    nodes = sorted(graph)                 # NOTE: sorted -- a stable, order-independent enumeration of the nodes
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    reach = [[False] * n for _ in range(n)]
    for u, targets in graph.items():
        for v in targets:
            reach[idx[u]][idx[v]] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    reach[i][j] = reach[i][j] or reach[k][j]
    return any(reach[i][i] for i in range(n))


G1 = {"T1": ["T2"], "T2": ["T1"]}                               # 2-cycle: deadlocked
G2 = {"T1": ["T2"], "T2": ["T3"], "T3": ["T1"]}                 # 3-cycle: deadlocked
G3 = {"T1": ["T2"], "T2": ["T3"], "T3": []}                     # a chain: not deadlocked
G4 = {"T1": ["T2", "T3"], "T2": [], "T3": ["T2"]}                # a DAG: not deadlocked
for g, expected in ((G1, True), (G2, True), (G3, False), (G4, False)):
    assert wait_for_cycle_dfs(g) == wait_for_cycle_brute(g) == expected

for _ in range(500):                      # random graphs: the two detectors always agree
    n_nodes = int(rng.integers(2, 8))
    nodes = [f"T{i}" for i in range(n_nodes)]
    graph = {u: [] for u in nodes}
    for u in nodes:
        for v in nodes:
            if u != v and rng.random() < 0.25:
                graph[u].append(v)
    assert wait_for_cycle_dfs(graph) == wait_for_cycle_brute(graph)

for _ in range(500):                      # a global lock order (edges only go to a higher-numbered node) never cycles
    n_nodes = int(rng.integers(2, 8))
    nodes = [f"T{i}" for i in range(n_nodes)]
    ordered_graph = {nodes[i]: [nodes[j] for j in range(n_nodes) if j > i and rng.random() < 0.4]
                     for i in range(n_nodes)}
    assert wait_for_cycle_dfs(ordered_graph) is False

# ---- Q7: cache-line loads for row-major versus column-major traversal, under a tiny LRU model
def line_loads(addresses: list[int], line_size: int, capacity: int) -> int:
    """A trace-driven LRU cache of `capacity` lines; returns the number of misses (cache-line loads)."""
    order: list[int] = []                 # front = most recently used
    misses = 0
    for addr in addresses:
        line = addr // line_size
        if line in order:
            order.remove(line)
            order.insert(0, line)
        else:
            misses += 1
            order.insert(0, line)
            if len(order) > capacity:
                order.pop()
    return misses


def line_loads_ordereddict(addresses: list[int], line_size: int, capacity: int) -> int:
    """Independent re-implementation of the same LRU policy, via OrderedDict instead of a plain list."""
    od: OrderedDict = OrderedDict()
    misses = 0
    for addr in addresses:
        line = addr // line_size
        if line in od:
            od.move_to_end(line)
        else:
            misses += 1
            od[line] = True
            if len(od) > capacity:
                od.popitem(last=False)
    return misses


ROWS, COLS, ELEM_SIZE, LINE_SIZE, CACHE_LINES = 8, 16, 8, 64, 4
row_major_addrs = [(i * COLS + j) * ELEM_SIZE for i in range(ROWS) for j in range(COLS)]
col_major_addrs = [(i * COLS + j) * ELEM_SIZE for j in range(COLS) for i in range(ROWS)]
row_major_misses = line_loads(row_major_addrs, LINE_SIZE, CACHE_LINES)
col_major_misses = line_loads(col_major_addrs, LINE_SIZE, CACHE_LINES)
assert row_major_misses == 16                                # one load per 8 elements: (8*16)/8
assert col_major_misses == 128                                # one load per element: every access misses
assert col_major_misses == 8 * row_major_misses               # exactly the 8 elements-per-line ratio
assert line_loads_ordereddict(row_major_addrs, LINE_SIZE, CACHE_LINES) == row_major_misses
assert line_loads_ordereddict(col_major_addrs, LINE_SIZE, CACHE_LINES) == col_major_misses
# the traced prefix from the statement: the first column exhausts the 4-line cache after 4 distinct rows
first_col_lines = [addr // LINE_SIZE for addr in col_major_addrs[:8]]
assert first_col_lines == [0, 2, 4, 6, 8, 10, 12, 14]

# ---- Q8: the float facts -- 0.1 + 0.2, machine epsilon, catastrophic cancellation, tolerance comparison
assert 0.1 + 0.2 != 0.3
exact_gap = Fraction(0.1) + Fraction(0.2) - Fraction(0.3)     # exact rational arithmetic: no rounding of its own
assert exact_gap != 0                                          # proves the inequality independently of float `!=`
assert 0 < abs(exact_gap) < Fraction(1, 10 ** 16)
assert (0.1 + 0.2) - 0.3 == 2.0 ** -54                          # NOTE: the float *expression* rounds once more than
                                                                 #       exact_gap does, and is not equal to it

eps64 = np.finfo(np.float64).eps
assert eps64 == 2.0 ** -52
assert 1.0 + 2 ** -53 == 1.0                                    # rounds away: below half the spacing at 1.0
assert 1.0 + 2 ** -52 != 1.0                                    # the definition of eps: smallest such that 1+eps != 1


def taylor_one_minus_cos(x: float) -> float:
    """1 - cos(x) via its Taylor series: shrinking, non-cancelling terms, accurate for small x."""
    return x ** 2 / 2 - x ** 4 / 24 + x ** 6 / 720 - x ** 8 / 40320


for x in (1e-4, 1e-6):
    ref = taylor_one_minus_cos(x)
    naive = 1 - math.cos(x)
    stable = 2 * math.sin(x / 2) ** 2
    stable_rel_err = abs(stable - ref) / ref
    naive_rel_err = abs(naive - ref) / ref
    assert stable_rel_err < 1e-9                                # stays at machine-precision accuracy
    assert naive_rel_err > 1000 * stable_rel_err                # cancellation costs several digits, and grows as x shrinks
assert math.cos(1e-8) == 1.0 and 1 - math.cos(1e-8) == 0.0       # total cancellation: not one correct digit left
assert 2 * math.sin(5e-9) ** 2 > 0.0                             # the stable formula has no such floor

assert 0.1 * 3 != 0.3 and math.isclose(0.1 * 3, 0.3)             # a relative tolerance catches ordinary rounding
assert not math.isclose(1e-300, 2e-300, rel_tol=1e-9, abs_tol=0.0)   # purely relative: still "different" near 0
assert math.isclose(1e-300, 2e-300, rel_tol=1e-9, abs_tol=1e-200)    # an absolute term is needed there instead

# ---- Q9: fp32 / fp16 / bf16 -- bit layout, largest value, epsilon; bf16 emulated by truncating fp32's bits
fi32, fi16 = np.finfo(np.float32), np.finfo(np.float16)
assert (fi32.nexp, fi32.nmant) == (8, 23) and (fi16.nexp, fi16.nmant) == (5, 10)
assert math.isclose(float(fi32.max), (2 - 2.0 ** -23) * 2.0 ** 127, rel_tol=1e-6)
assert fi16.max == 65504.0 == (2 - 2.0 ** -10) * 2.0 ** 15
assert fi32.eps == np.float32(2.0 ** -23) and fi16.eps == np.float16(2.0 ** -10)
assert float(fi16.tiny) == 2.0 ** -14                           # fp16's smallest normal value: ~6.1e-5 in the text
assert float(fi16.smallest_subnormal) == 2.0 ** -24              # fp16's smallest subnormal value: ~6e-8 in the text


def to_bf16(x: np.float32) -> np.float32:
    """bf16 has fp32's 8 exponent bits and only 7 mantissa bits: keep the top 16 bits of the fp32 pattern."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


bf16_max = to_bf16(np.float32(fi32.max))                        # truncating never overflows the exponent field
assert math.isclose(float(bf16_max), (2 - 2.0 ** -7) * 2.0 ** 127, rel_tol=1e-6)
one_bits = np.float32(1.0).view(np.uint32)
bf16_next = (one_bits + np.uint32(1 << 16)).view(np.float32)    # the smallest step the 7-bit mantissa can take
assert float(bf16_next) == 1.0 + 2.0 ** -7

tiny_grad = 1e-8                                                 # smaller than fp16's smallest subnormal (~6e-8)
assert np.float16(tiny_grad) == 0.0                              # fp16: flushed to exactly zero
scaled = np.float16(tiny_grad * 2.0 ** 16)                       # loss scaling by a power of two: an exact shift
assert scaled != 0.0
recovered = float(scaled) / 2.0 ** 16
assert math.isclose(recovered, tiny_grad, rel_tol=2 ** -9)       # within fp16's own relative precision
bf16_tiny = to_bf16(np.float32(tiny_grad))                       # bf16: fp32's exponent range, so no underflow here
assert bf16_tiny != 0.0
assert math.isclose(float(bf16_tiny), tiny_grad, rel_tol=2 ** -6)

# ---- Q10: doubling gives amortised O(1) insertion (total copies < 2n); average- vs worst-case hash-table lookup
def doubling_copies(n: int, growth: float = 2.0) -> int:
    """Simulate n insertions into an array that starts at capacity 1 and grows by `growth` whenever full.
    Returns the total number of element copies performed across every resize."""
    capacity, size, total_copies = 1, 0, 0
    for _ in range(n):
        if size == capacity:
            total_copies += size          # every existing element is copied into the new array
            capacity = max(capacity + 1, math.ceil(capacity * growth))
        size += 1
    return total_copies


for n in (1, 7, 100, 1_000, 100_000):
    m = 1 << (n - 1).bit_length() if n > 0 else 1     # smallest power of two >= n
    assert doubling_copies(n, growth=2.0) == m - 1     # the exact closed form derived in the text
    assert m - 1 < 2 * n                               # ... and it is always under 2n

# the same geometric argument at Python list's actual (smaller) growth factor: still O(1) amortised, bigger constant
for n in (1_000, 100_000):
    assert doubling_copies(n, growth=1.125) < 9 * n


class SimpleHashTable:
    """Separate chaining; doubles capacity whenever inserting would push the load factor above 0.75."""

    def __init__(self, capacity: int = 8, hash_fn=hash) -> None:
        self._capacity = capacity
        self._hash_fn = hash_fn
        self._buckets: list[list[tuple]] = [[] for _ in range(capacity)]
        self._size = 0
        self.resize_copies = 0

    def _grow(self) -> None:
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        for bucket in old_buckets:
            for key, value in bucket:
                self._buckets[self._hash_fn(key) % self._capacity].append((key, value))
                self.resize_copies += 1

    def insert(self, key, value) -> None:
        if (self._size + 1) / self._capacity > 0.75:
            self._grow()
        bucket = self._buckets[self._hash_fn(key) % self._capacity]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1

    def get(self, key):
        """Returns (value, comparisons) -- comparisons is how many keys were examined: a counting-model stand-in
        for lookup cost, used instead of timing it."""
        bucket = self._buckets[self._hash_fn(key) % self._capacity]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                return v, i + 1
        return None, len(bucket)


for n in (500, 5_000, 50_000):                                   # good hash: average comparisons stay flat in n
    keys = [int(k) for k in np.unique(rng.integers(0, 10 ** 12, size=2 * n))[:n]]
    table = SimpleHashTable()
    for k in keys:
        table.insert(k, k * 2)
    total_comparisons = sum(table.get(k)[1] for k in keys)
    assert total_comparisons / len(keys) < 3.0                   # O(1) average: no growth with n
    assert table.resize_copies < 3 * n                            # its own resizes are linear in n too

bad_n = 2_000                                                     # worst case: every key hashes to the same bucket
bad_keys = [int(k) for k in np.unique(rng.integers(0, 10 ** 12, size=2 * bad_n))[:bad_n]]
bad_table = SimpleHashTable(hash_fn=lambda k: 0)
for k in bad_keys:
    bad_table.insert(k, k)
assert len(bad_table._buckets[0]) == bad_n                        # resizing cannot help: every key still collides
_, comparisons = bad_table.get(bad_keys[-1])
assert comparisons == bad_n                                       # a full linear scan: worst-case O(n)

print("all checks passed")
```

</details>

</details>
