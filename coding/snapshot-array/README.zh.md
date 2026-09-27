# 快照数组：历史、压缩与差异

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 哈希表、二分查找、时空权衡 | ★★★☆☆ | 中等 | SWE · RE · MLE · Intern | hash-map, binary-search, versioning, memory-trade-offs, journaling | 3 个部分 / 45 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

一个长度为 `length` 的*快照数组*（snapshot array）保存 `length` 个整数，下标从 `0` 到 `length - 1`，初始时每
个位置都是 `0`。*快照*（snapshot）是数组全部内容的一份只读记录，由一次显式调用产生，并用一个*快照 id*
（snapshot id）标识：第一个快照的 id 是 `0`，第二个是 `1`，依此类推，按快照实际产生的顺序编号。两次快照之
间，任意下标都可以被写入任意次；之后按某个快照 id 读取时，必须返回该下标在这个快照产生那一刻的值，与此后
又写入过什么无关。

### Part 1 —— 快照

```py
class SnapshotArray:
    def __init__(self, length: int) -> None: ...           # every index starts at 0
    def set(self, index: int, val: int) -> None: ...
    def snap(self) -> int: ...                              # returns the id just taken: 0, 1, 2, ...
    def get(self, index: int, snap_id: int) -> int: ...     # value at index when snap_id was taken
```

`set` 和 `get` 在 `index` 超出 `[0, length)` 时都会抛出 `IndexError`；`get` 在 `snap_id` 不是 `snap` 已经返
回过的值时（包括负数）同样抛出 `IndexError`。`get` 绝不会返回某次 `set` 之后、还没有被 `snap` 覆盖到的
值——这样的写入在下一次 `snap` 把它封存进新快照之前，对任何 `get` 都不可见。所需代价：`snap` 是 $O(1)$；
`get` 是 $O(\log s)$，其中 $s$ 是被查询下标被写入过的次数，既不是已取快照的数量，也不是 `length`；内存方
面，从未被写入的下标只占 $O(1)$，每调用一次 `set` 再额外占 $O(1)$。

以 `length = 4` 为例进行追踪：

```text
arr = SnapshotArray(4)                # [0, 0, 0, 0]
arr.set(0, 5)
arr.set(0, 6)                         # 覆盖了还未封存的写入 5 —— 见下文
s0 = arr.snap()                       # -> 0   快照 0 = [6, 0, 0, 0]
arr.set(1, 9)
arr.set(0, 1)
s1 = arr.snap()                       # -> 1   快照 1 = [1, 9, 0, 0]
arr.set(0, 4)
arr.set(2, 2)
s2 = arr.snap()                       # -> 2   快照 2 = [4, 9, 2, 0]
s3 = arr.snap()                       # -> 3   快照 3 = [4, 9, 2, 0]   （快照 2 之后没有 set 过）
s4 = arr.snap()                       # -> 4   快照 4 = [4, 9, 2, 0]   （快照 2 之后没有 set 过）
arr.set(0, 7)
s5 = arr.snap()                       # -> 5   快照 5 = [7, 9, 2, 0]

arr.get(0, 0)   # -> 6   两次写入里只有较晚的一次在快照 0 中可见
arr.get(0, 1)   # -> 1
arr.get(0, 4)   # -> 4   下标 0 在快照 2 和快照 4 之间没有变化
arr.get(0, 5)   # -> 7
arr.get(1, 0)   # -> 0   下标 1 直到快照 0 之后才被写入
arr.get(3, 5)   # -> 0   下标 3 从未被写入过
```

### Part 2 —— 释放旧快照

```py
    def release(self, snap_id: int) -> None: ...
    def stored_entries(self) -> int: ...
```

`release(snap_id)` 记录调用方的一个承诺：无论是这次调用还是以后的调用，都不会再对 `get` 传入任何
`<= snap_id` 的快照 id（下面 Part 3 加入的 `diff` 也继承同样的承诺）。`snap_id` 必须是 `snap` 已经返回过的
值，规则与 `get` 完全相同，否则抛出 `IndexError`。之后再次调用 `release` 只会把这条边界继续抬高：传入一个
不高于已释放边界的 `snap_id` 不会有任何效果。之后对一个已释放的快照 id 调用 `get` 会抛出
`ValueError`，而不是 `IndexError`——这个 id 本身是合法的，只是已经变得不可读。

释放内存正是这个操作的意义所在：在一次 `release` 调用之后的有限步之内，每个下标内部保存的历史都必须只剩下
回答存活（未释放）快照的 `get`、或者得知当前值所必需的条目——绝不能再保留那些只能回答已释放 id 的条目。
`stored_entries()` 返回当前所有下标合计保存的 `(snap_id, value)` 条目总数，用来直接衡量这条承诺；它本身计
算代价是 $O(\text{length})$，存在的唯一目的就是这项测量。

接续 Part 1 例子的追踪（此时 `arr` 已经历 `s5 = 5`，下标 `0` 的历史覆盖 id `0, 1, 2, 5`）：

```text
arr.release(2)          # id 0、1、2 都变为已释放
arr.get(0, 1)            # -> ValueError   （1 <= 2）
arr.get(0, 2)            # -> ValueError   （2 <= 2，release 本身传入的 id 也算在内）
arr.get(0, 4)            # -> 4             （4 > 2，仍然存活，结果不变）
arr.release(0)            # 空操作：0 已经低于已释放边界 2
```

### Part 3 —— 快照之间的差异

```py
    def diff(self, a: int, b: int) -> dict[int, tuple[int, int]]: ...
```

`a` 和 `b` 都必须是 Part 2 意义下存活的快照 id（`IndexError`/`ValueError` 的规则与 `get` 完全相同），并且要
满足 `a < b`，否则抛出 `ValueError`。`diff` 返回每一个在快照 `b` 的值与快照 `a` 的值不同的下标，映射到
`(value_at_a, value_at_b)`；一个下标即使在这两个快照之间被写入过一次或多次，只要最终在两处的值相同，就不
会出现在结果里。这个操作的运行时间必须只与快照 `a` 之后、直到快照 `b` 为止记录的 `set` 调用次数、再加上输
出规模成正比——绝不能与 `length` 成正比。

接续同一个例子的追踪：

```text
arr.diff(3, 5)   # -> {0: (4, 7)}   只有下标 0 在快照 3 和快照 5 之间发生了变化
arr.diff(2, 4)   # -> {}             快照 2 和快照 4 之间什么都没变
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有两点值得先确认：`get` 是否可能暴露一次还没有被 `snap` 覆盖到的写入（不会——一次尚未封存的写
入，在下一次 `snap` 把它封存之前始终不可见）；以及 `snap` 到底返回什么、总共能取多少次快照（按调用顺序返
回 id `0, 1, 2, ...`，除了内存没有其他上限——`snap` 的返回值*就是*这个 id 本身，不是计数，也不是另外的令
牌）。

### Part 1

每次写入都是按下标记录的，而不是按快照记录的。`_history[index]` 是一个按 `snap_id` 递增排列的
`(snap_id, value)` 列表——但 `set` 调用写入时用的这个 `snap_id`，未必是 `snap` 已经返回过的值：它是
`_next_id`，也就是*下一次* `snap` 调用将会返回的 id，换句话说，是当前正在构建、尚未封存的那个快照。一次
`set` 要么为 `_next_id` 追加一个新的条目，要么——如果该下标最后记录的条目已经带有同一个 id（说明自上次
`snap` 以来，同一个下标已经被 `set` 过）——原地覆盖它：因为任何 `get` 都不可能查询两次 `snap` 调用之间的
某个中间状态，所以只有下一次 `snap` 之前写入的最后一个值才有意义，保留更早的那些只会白白浪费内存。`snap`
本身完全不涉及任何按下标的状态——它只是读取并自增 `_next_id`——这正是它无论 `length` 有多大、无论已经写
过多少个下标都能做到 $O(1)$ 的原因。`get` 用 `bisect_right`、以每个条目自己的 id 为键，定位在 `snap_id`
时生效的那个条目：`snap_id` 在已存储 id 中的插入位置，恰好落在所有 `<= snap_id` 的 id 之后一位，所以减去
`1` 就落在仍然 `<= snap_id` 的*最后一个*（也是最大的）id 上——那里记录的值正是 `snap_id` 时刻的当前值。如
果插入位置是 `0`，说明没有任何已记录的 id `<= snap_id`，也就是说该下标在这个快照之前从未被写入过，它的值
仍然是初始的 `0`。

`_history` 是一个用 `setdefault` 惰性填充的 `dict`，而不是一个预先按 `length` 分配好的列表：这样一来，一
个从未被写入的下标根本不会成为一个键，不占用任何额外内存，这比预先分配 `length` 个空容器的列表还要更
省——尽管后者（每个下标一份固定的常数开销，稍大一些）本身也是可以接受的。

```python
from bisect import bisect_right


class SnapshotArray:
    """array[i] is 0 until set. get(i, s) reads what array[i] held when snapshot s was taken; a write
    made after s, or not yet covered by any snap at all, is never visible to it."""

    def __init__(self, length: int) -> None:
        self._length = length
        self._next_id = 0                                     # id whichever snap() call is next will return
        self._history: dict[int, list[tuple[int, int]]] = {}  # index -> [(snap_id, value), ...], lazy

    def _check_index(self, index: int) -> None:
        if not (0 <= index < self._length):
            raise IndexError(f"index {index} out of range [0, {self._length})")

    def _check_taken(self, snap_id: int) -> None:
        if not (0 <= snap_id < self._next_id):
            raise IndexError(f"snapshot {snap_id} was never taken")

    def set(self, index: int, val: int) -> None:
        self._check_index(index)
        h = self._history.setdefault(index, [])
        if h and h[-1][0] == self._next_id:
            h[-1] = (self._next_id, val)    # NOTE: overwrite -- an earlier set in this same pending
        else:                               #      interval can never be read by any snapshot
            h.append((self._next_id, val))

    def snap(self) -> int:
        snap_id = self._next_id
        self._next_id += 1
        return snap_id

    def get(self, index: int, snap_id: int) -> int:
        self._check_index(index)
        self._check_taken(snap_id)
        h = self._history.get(index)
        if not h:
            return 0
        pos = bisect_right(h, snap_id, key=lambda e: e[0]) - 1  # NOTE: -1 -- bisect_right lands one past
        return h[pos][1] if pos >= 0 else 0                     #      every id <= snap_id; back up one
```

`set` 是 $O(1)$：它最多只检查并修改一个列表的最后一个元素，不管这个列表是字典里已有的，还是刚刚现场创建
的。`snap` 是 $O(1)$：两次属性操作，完全独立于 `length`。`get` 是 $O(\log s)$，$s = \text{len}(h)$：对一
个以自身存储的 id 为键的列表做一次 `bisect_right`。综合所有下标来看，`_history` 里每调用一次 `set` 至多
增加一个条目——一次调用要么追加一个新条目，要么覆盖已有的最后一个，两者不会同时发生——所以总内存是每个
真正被写入过的下标占 $O(1)$，再加上每次 `set` 调用占 $O(1)$，比题目要求的上界还要更宽裕。

值得再比较另外两种设计，看看为什么按*下标*组织历史能换来上面这些代价。每次 `snap` 都把整个数组复制进一个
新列表，会让 `get` 变得毫无难度——直接在选中的那份副本里按下标取值，$O(1)$——但 `snap` 会变成
$O(\text{length})$，因为不管某个元素变没变，都要把它复制一遍，总内存则是
$O(\text{length} \times \text{已取快照数})$：每个快照一份完整副本，其中绝大部分都和前一份一模一样。改成
每个快照只记录*相对上一个快照发生变化的下标组成的字典*——一份差异，而不是完整副本——能同时解决这两个问
题，但 `get` 现在必须从 `snap_id` 往回走，逐个检查它前面每一个快照的差异字典里有没有查询的下标，直到找到
或者走到头为止——$O(k)$，$k$ 是该下标上一次变化以来经过的快照数，而这个数字并不受下标自身控制。像本方案
这样按下标而不是按快照来组织历史，能让每个下标拥有自己独立、简短、可单独查找的列表，`get` 就完全不需要翻
看发生在*其他*下标上的变化了。

| 方案 | `set` | `snap` | `get` | 内存 |
| --- | --- | --- | --- | --- |
| 每次 `snap` 全量复制 | $O(1)$ | $O(\text{length})$ | $O(1)$ | $O(\text{length} \times \text{快照数})$ |
| 每个快照一份差异字典，`get` 往回走 | $O(1)$ | $O(1)$ | $O(k)$，$k$ = 该下标上次变化以来经过的快照数 | $O(\text{length} + \text{set 次数})$ |
| 按下标记录历史，二分查找（本方案） | $O(1)$ | $O(1)$ | $O(\log s)$，$s$ = 该下标的变化次数 | $O(\text{length} + \text{set 次数})$ |

### Part 2

一旦 `release(r)` 生效（`r` 是历次传入的最大值），所有 `<= r` 的快照 id 就永久消失了——之后无论是 `get`
还是 `diff`，都不能再提到其中任何一个。对某个下标的历史 $[(id_1, v_1), \dots, (id_m, v_m)]$（按 id 排序）
来说，`bisect_right` 会用*同一个*条目去回答所有这些已经释放的 id：因为列表里的 id 只会递增，所以答案永远
是最后一个满足 $id_i \le r$ 的条目。把它称为*底值*（floor）条目。底值之前的每一个条目，其 id 也都
$\le r$（id 递增），所以它们能回答的查询只可能是现在已经释放的那些——可以证明它们已经彻底失效。底值条目
本身、以及它之后的所有条目，可达性和 `release` 调用之前完全一样：底值条目回答列表中下一个 id 之前的所有
存活 id（尤其是可能存在的最小存活 id，即 $r + 1$——只要 $r + 1$ 这个位置本身没有写入过），后面的每个条目
回答的 id 集合也和以前完全相同。所以不变式是：**保留 id $\le$ 已释放边界的最后一个条目，以及它之后的所有
条目；丢弃它之前的一切。**

```python
class SnapshotArray(SnapshotArray):
    """Adds release(): the caller promises never to read a snapshot <= released again."""

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self._released = -1                                   # -1: nothing released yet

    def _compact(self, index: int) -> None:
        """Drops every entry of index's history that release() has made permanently unreachable."""
        h = self._history.get(index)
        if not h or self._released < 0:
            return
        pos = bisect_right(h, self._released, key=lambda e: e[0]) - 1
        if pos > 0:          # NOTE: pos itself is the floor entry to KEEP -- only what precedes it is dead
            del h[:pos]

    def _check_live(self, snap_id: int) -> None:
        self._check_taken(snap_id)
        if snap_id <= self._released:
            raise ValueError(f"snapshot {snap_id} was released (released up to {self._released})")

    def get(self, index: int, snap_id: int) -> int:
        self._check_index(index)
        self._check_live(snap_id)
        return super().get(index, snap_id)

    def stored_entries(self) -> int:
        return sum(len(h) for h in self._history.values())
```

应用 `_compact` 有两种方式，二者在 `release` 自身的开销和 `stored_entries()` 多快能降到不变式允许的最小值
之间做取舍。**立即**（eager）压缩在 `release` 被调用的那一刻就检查每一个下标：每次调用都是
$\Theta(\text{length})$，因为没有任何机制能提前知道哪些下标可能需要修剪，但 `release` 一返回，
`stored_entries()` 就已经处在不变式的最小值上。**惰性**（lazy）压缩则每次调用只做一小份固定的*额外*工
作——顺便压缩某次 `set` 恰好触及的下标，再加上每次 `release` 调用时，让一个缓慢前进的轮询指针多压缩几个
下标——其余的暂时保持陈旧状态。这样做总体上便宜，原因在于压缩只会*删除*条目、从不新增，而一个条目在它一
生中最多只能被删除一次（不会再回来）。因此，在对象的整个生命周期里累计起来，`_compact` 做过的全部工
作——包括每次 `release` 调用里固定大小的轮询步骤，以及每次 `set` 调用里的顺手压缩——至多等于历史上曾经
创建过的条目总数，也就是至多等于 `set` 调用的总次数：这正是*每次调用*均摊 $O(1)$，叠加在每次调用本身
$O(1)$ 的基础开销之上——用的是和“两个栈实现的队列弹出元素”或者“惰性清理的堆丢弃陈旧条目”均摊 $O(1)$ 完
全相同的聚合论证，即便单次操作本身可能更贵。代价是：`stored_entries()` 未必会在某一次 `release` 调用返回
的瞬间就降到最小值——一个既没有被重新 `set`、又还没被轮询扫到的下标，会把它的失效条目多留一段时间，这段
时间是有界的，但不是立刻清零。

```python
class SnapshotArrayEagerRelease(SnapshotArray):
    """release() compacts every index immediately."""

    def release(self, snap_id: int) -> None:
        self._check_taken(snap_id)
        if snap_id > self._released:
            self._released = snap_id
        for index in range(self._length):
            self._compact(index)


class SnapshotArrayLazyRelease(SnapshotArray):
    """release() only raises the boundary and nudges a small, fixed-size batch of a round-robin sweep
    forward; the rest of an index's compaction happens the next time that index is set."""

    _SWEEP_BATCH = 4          # arbitrary and small -- illustrates the mechanism, not a tuned constant

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self._sweep_pos = 0

    def set(self, index: int, val: int) -> None:
        self._check_index(index)
        self._compact(index)     # NOTE: on-touch -- the cheapest moment to drop this index's dead entries
        super().set(index, val)

    def release(self, snap_id: int) -> None:
        self._check_taken(snap_id)
        if snap_id > self._released:
            self._released = snap_id
        batch = min(self._SWEEP_BATCH, self._length)   # NOTE: min() -- also keeps this at 0 when length == 0
        for _ in range(batch):
            self._compact(self._sweep_pos)
            self._sweep_pos = (self._sweep_pos + 1) % self._length


SnapshotArray = SnapshotArrayLazyRelease   # the fuller answer: cheap release(), memory still bounded
```

| 方案 | `release` 自身的开销 | 每次 `set` 多付出的代价 | `release` 返回后 `stored_entries()` |
| --- | --- | --- | --- |
| 立即（eager） | $\Theta(\text{length})$ | 无 | 恰好等于不变式允许的最小值 |
| 惰性（lazy） | $O(1)$（固定批量） | 均摊 $O(1)$ | 可能仍保留一些只对应已释放 id 的条目，有界，但不会立刻清零 |

### Part 3

要在不扫描 `length` 的前提下回答 `diff(a, b)`，需要能低成本地知道*哪些*下标可能发生过变化，所以这个类还
维护了一份日志。`_pending_touched` 收集自上次 `snap` 以来被 `set` 过的下标——也就是接下来要被封存的那个
快照会用到的下标——`snap` 会先把它存档为 `_journal[snap_id]`，再把 id 本身的生成交给基类处理；任何一次
`snap` 调用之后都有 `len(_journal) == _next_id`，所以 `_journal[s]` 对每一个真正取过的快照 id 都有定义。
一个下标即使在同一个区间里变化了不止一次，也只会给这个区间的集合贡献一个条目（集合不会有重复元素）；一个
在某个区间里完全没被碰过的下标，则不会给它贡献任何东西。

`diff(a, b)` 把 `_journal[a + 1]` 到 `_journal[b]` 取并集——也就是所有在 `a` 之后、直到 `b` 为止被碰过的
下标——然后用 `get` 逐一核对每个候选下标在 `a` 和 `b` 处的实际值，只保留真正不同的那些：被碰过是必要条
件，但不是充分条件，因为一个下标完全可能被写回原来的值，或者被写了不止一次但最终结果不变。构建并集的代价
是 $O(D)$，$D$ 是被访问过的日志条目总数——不管 `length` 有多大，它最多等于快照 $a + 1$ 到 $b$ 之间记录的
`set` 调用次数；核对（至多 $D$ 个）候选下标再花费 $O(D \log S)$，$S$ 是其中最大的单下标历史长度；构建结
果的代价是 $O(\text{输出规模})$。这里没有任何一步依赖 `length`。

```python
class SnapshotArray(SnapshotArray):
    """Adds diff(): every index whose value changed between two live snapshots."""

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self._journal: list[set[int]] = []       # journal[s]: indices set during the interval sealed as s
        self._pending_touched: set[int] = set()

    def set(self, index: int, val: int) -> None:
        super().set(index, val)             # validates index, applies the inherited on-touch compaction
        self._pending_touched.add(index)     # NOTE: only after a successful set -- a rejected one changed nothing

    def snap(self) -> int:
        self._journal.append(self._pending_touched)
        self._pending_touched = set()
        return super().snap()

    def diff(self, a: int, b: int) -> dict[int, tuple[int, int]]:
        self._check_live(a)
        self._check_live(b)
        if not a < b:
            raise ValueError(f"diff requires a < b, got a={a}, b={b}")
        touched: set[int] = set()
        for s in range(a + 1, b + 1):
            touched |= self._journal[s]
        result: dict[int, tuple[int, int]] = {}
        for index in touched:
            va, vb = self.get(index, a), self.get(index, b)
            if va != vb:
                result[index] = (va, vb)
        return result
```

`snap` 和 `set` 各自都只比 Part 2 的版本多做一次 $O(1)$ 操作——封存或者扩充一个集合——所以两者都仍然是
$O(1)$（对 `set` 而言，和之前一样是均摊意义下的）。

### 追问

- **持久化。** 把每一次 `set` 和 `snap` 调用都写进一份只追加的日志，在调用返回之前落盘，就能在崩溃之后从
  头重放日志来重建数组；一旦日志变长，把它整个重放一遍就会变慢，所以需要定期做检查点（checkpoint）——一
  份完整的数组副本，标注上它对应的最后一个快照的 id——这样恢复时可以直接从检查点开始，只需要重放它之后写
  入的日志条目。
- **旧快照的并发读取。** 一个快照一旦生成，能够回答它的那些条目就再也不会变化：`set` 要么追加一个新条
  目，要么覆盖*当前尚未封存*的那一个，绝不会动到属于某个已经生成的快照的条目。所以任意多个线程都可以在完
  全不加锁的情况下对已生成的快照调用 `get`。但在同一个对象上并发调用 `set` 和 `snap` 仍然需要加锁，而且
  这把锁不能只锁住某一个下标：`snap` 推进的是唯一共享的 `_next_id`，每次 `set` 都要读取它来判断该追加还
  是覆盖，所以只要另一个操作还没完成，这两者就不能在任何下标上安全地同时运行。
- **二维数组的快照。** 把 `(row, col)` 压平成 `row * n_cols + col`，再原样套用同一套按下标记录历史和日志
  的方案，就能把二维的情形归约到已经解决的一维情形——`length` 变成 `n_rows * n_cols`，`diff` 依然返回压
  平后的下标，或者在边界处再还原成 `(row, col)` 二元组。
- **写时复制（copy-on-write）页。** 把数组切分成固定大小的页（page），用一棵小型的持久化树来存放页指
  针，而不是一个扁平的指针数组（这样更新一个指针只需要复制它所在路径上的 $O(\log P)$ 个节点，不用动整张
  指针表）；在某一页的任何一个元素在某次快照之后第一次被写入时，复制这*一整页*、而不是整个数组。这样
  `snap` 就是 $O(1)$（只需要保留当前这棵树的根节点引用，暂时什么都不用复制），代价是每个区间里，一页只
  在*第一次*被写入时，付出一次整页大小的复制，外加那条 $O(\log P)$ 路径上的复制，而不是每次写入都记一个
  小条目。这比按下标记录历史要粗——写一个元素就要为它所在的整页付费——但这正是真实的写时复制文件系统和
  持久化数据结构所做的权衡：用它换来每个快照需要追踪的东西少得多。

<details>
<summary>验证代码（可运行）</summary>

```python
import random

# --- Part 1: the worked example, traced step by step ---
arr = SnapshotArray(4)
arr.set(0, 5)
arr.set(0, 6)
assert arr.snap() == 0
arr.set(1, 9)
arr.set(0, 1)
assert arr.snap() == 1
arr.set(0, 4)
arr.set(2, 2)
assert arr.snap() == 2
assert arr.snap() == 3
assert arr.snap() == 4
arr.set(0, 7)
assert arr.snap() == 5

assert arr.get(0, 0) == 6
assert arr.get(0, 1) == 1
assert arr.get(0, 2) == 4
assert arr.get(0, 3) == 4
assert arr.get(0, 4) == 4
assert arr.get(0, 5) == 7
assert arr.get(1, 0) == 0
assert arr.get(1, 1) == 9
assert arr.get(2, 1) == 0
assert arr.get(2, 2) == 2
assert arr.get(3, 5) == 0

# --- Part 3's worked example, same object, before anything is released ---
assert arr.diff(3, 5) == {0: (4, 7)}
assert arr.diff(2, 4) == {}

# --- Part 2's worked example, same object ---
arr.release(2)
for snap_id in (1, 2):
    try:
        arr.get(0, snap_id)
        assert False, "expected ValueError"
    except ValueError:
        pass
assert arr.get(0, 4) == 4
arr.release(0)                 # no-op: 0 is already below the released boundary of 2
assert arr._released == 2

# --- IndexError / ValueError boundary cases ---
try:
    arr.set(4, 1)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.get(0, 100)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.get(0, -1)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.release(100)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.diff(3, 100)
    assert False, "expected IndexError"
except IndexError:
    pass
try:
    arr.diff(5, 3)
    assert False, "expected ValueError"
except ValueError:
    pass
try:
    arr.diff(3, 3)
    assert False, "expected ValueError"
except ValueError:
    pass

# --- length == 0: every index is out of range, and release()'s sweep must not divide by zero ---
empty = SnapshotArray(0)
assert empty.snap() == 0
assert empty.snap() == 1
try:
    empty.set(0, 1)
    assert False, "expected IndexError"
except IndexError:
    pass
assert empty.diff(0, 1) == {}
empty.release(0)          # must not raise ZeroDivisionError -- see the NOTE on the sweep's min()
assert empty.stored_entries() == 0


# --- an independent brute force: a full array copy at every snapshot, nothing else ---
class _BruteForceSnapshotArray:
    def __init__(self, length: int) -> None:
        self._length = length
        self._current = [0] * length
        self._snapshots: list[list[int]] = []
        self._released = -1

    def set(self, index: int, val: int) -> None:
        if not (0 <= index < self._length):
            raise IndexError
        self._current[index] = val

    def snap(self) -> int:
        self._snapshots.append(list(self._current))
        return len(self._snapshots) - 1

    def get(self, index: int, snap_id: int) -> int:
        if not (0 <= index < self._length):
            raise IndexError
        if not (0 <= snap_id < len(self._snapshots)):
            raise IndexError
        if snap_id <= self._released:
            raise ValueError
        return self._snapshots[snap_id][index]

    def release(self, snap_id: int) -> None:
        if not (0 <= snap_id < len(self._snapshots)):
            raise IndexError
        self._released = max(self._released, snap_id)

    def diff(self, a: int, b: int) -> dict[int, tuple[int, int]]:
        for s in (a, b):
            if not (0 <= s < len(self._snapshots)):
                raise IndexError
            if s <= self._released:
                raise ValueError
        if not a < b:
            raise ValueError
        sa, sb = self._snapshots[a], self._snapshots[b]
        return {i: (sa[i], sb[i]) for i in range(self._length) if sa[i] != sb[i]}


def _apply(obj, op):
    try:
        if op[0] == "set":
            return "ok", obj.set(op[1], op[2])
        if op[0] == "snap":
            return "ok", obj.snap()
        if op[0] == "get":
            return "ok", obj.get(op[1], op[2])
        if op[0] == "release":
            return "ok", obj.release(op[1])
        if op[0] == "diff":
            return "ok", obj.diff(op[1], op[2])
    except (IndexError, ValueError) as e:
        return "error", type(e)
    raise AssertionError(f"unknown op {op[0]}")


def _random_ops(rng, length, n_ops):
    ops = []
    taken = 0
    for _ in range(n_ops):
        choice = rng.random()
        if choice < 0.30:
            ops.append(("set", rng.randrange(-1, length + 1), rng.randint(-9, 9)))
        elif choice < 0.50:
            ops.append(("snap",))
            taken += 1
        elif choice < 0.72:
            hi = max(taken, 1)
            ops.append(("get", rng.randrange(-1, length + 1), rng.randrange(-2, hi + 1)))
        elif choice < 0.85:
            hi = max(taken, 1)
            ops.append(("release", rng.randrange(-2, hi + 1)))
        else:
            hi = max(taken, 1)
            ops.append(("diff", rng.randrange(-2, hi + 1), rng.randrange(-2, hi + 1)))
    return ops


# --- randomised cross-check against the brute force, over many small arrays ---
counts = {"bad_index": 0, "bad_snap_id": 0, "released": 0, "bad_order": 0, "ok_diff": 0, "ok_nonempty_diff": 0}
for seed in range(400):
    rng = random.Random(seed)
    length = rng.randint(1, 6)
    sol = SnapshotArray(length)
    brute = _BruteForceSnapshotArray(length)
    for op in _random_ops(rng, length, 60):
        taken_before, released_before = len(brute._snapshots), brute._released
        got, want = _apply(sol, op), _apply(brute, op)
        assert got == want, (seed, op, got, want)
        if want[0] == "error":
            kind = op[0]
            if kind in ("set", "get") and not (0 <= op[1] < length):
                counts["bad_index"] += 1
            elif kind == "get" and not (0 <= op[2] < taken_before):
                counts["bad_snap_id"] += 1
            elif kind == "release" and not (0 <= op[1] < taken_before):
                counts["bad_snap_id"] += 1
            elif kind == "diff" and (not (0 <= op[1] < taken_before) or not (0 <= op[2] < taken_before)):
                counts["bad_snap_id"] += 1
            elif kind == "get" and op[2] <= released_before:
                counts["released"] += 1
            elif kind == "diff" and (op[1] <= released_before or op[2] <= released_before):
                counts["released"] += 1
            elif kind == "diff":
                counts["bad_order"] += 1
        elif op[0] == "diff":
            counts["ok_diff"] += 1
            if want[1]:
                counts["ok_nonempty_diff"] += 1
assert min(counts.values()) > 5, counts   # every interesting case actually fired, repeatedly, not just once

# --- Part 2: stored_entries() matches an independently computed bound, and release() shrinks it ---
# a workload with many overwritten snapshots: release() should collapse most of it away
probe = SnapshotArrayEagerRelease(1)
for s in range(30):
    for i in range(5):
        probe.set(0, s * 10 + i)     # 5 overwrites per interval -- only the last (s * 10 + 4) ever matters
    probe.snap()
assert probe.stored_entries() == 30   # 150 set() calls collapse to 30 entries, one per snapshot interval
assert probe.get(0, 0) == 4
probe.release(25)
assert probe.stored_entries() == 5    # release(25) then collapses ids 0..24 into their single floor entry


def _reference_entry_count(touched_by_index, released):
    """The number of entries the stated invariant allows to survive, computed only from touched_by_index
    (the snapshot ids at which each index was actually set, tracked directly from the operations applied --
    never read from SnapshotArray's own _history)."""
    total = 0
    for ids in touched_by_index:
        if not ids:
            continue
        if released < 0:
            total += len(ids)
            continue
        total += (1 if any(i <= released for i in ids) else 0) + sum(1 for i in ids if i > released)
    return total


rng = random.Random(2024)
length = 40
sol_eager = SnapshotArrayEagerRelease(length)
sol_lazy = SnapshotArrayLazyRelease(length)
touched_by_index = [[] for _ in range(length)]
pending, taken, release_calls = set(), 0, 0
for _ in range(800):
    action = rng.random()
    if action < 0.55:
        index, val = rng.randrange(length), rng.randint(-99, 99)
        sol_eager.set(index, val)
        sol_lazy.set(index, val)
        pending.add(index)
    elif action < 0.75:
        for index in pending:
            touched_by_index[index].append(taken)
        pending = set()
        assert sol_eager.snap() == taken
        assert sol_lazy.snap() == taken
        taken += 1
    elif taken:
        snap_id = rng.randrange(taken)
        sol_eager.release(snap_id)
        sol_lazy.release(snap_id)
        release_calls += 1
# seal any dangling, not-yet-snapped writes with one final snap, so touched_by_index fully accounts
# for every stored entry -- otherwise a write with no snap after it would be invisible to this bookkeeping
# but still present, correctly, in _history
for index in pending:
    touched_by_index[index].append(taken)
assert sol_eager.snap() == taken
assert sol_lazy.snap() == taken
taken += 1

released_level = sol_eager._released
bound = _reference_entry_count(touched_by_index, released_level)
uncompacted_total = sum(len(ids) for ids in touched_by_index)
assert release_calls > 20 and released_level >= 0          # the scenario actually exercises release()
assert sol_eager.stored_entries() == bound, (sol_eager.stored_entries(), bound)
assert sol_lazy.stored_entries() >= bound
assert sol_lazy.stored_entries() <= uncompacted_total
assert sol_eager.stored_entries() < uncompacted_total        # eager: release() really did shrink things

# --- Part 2: eager's release() touches every index; lazy's touches only a small, fixed batch ---
class _CountingEager(SnapshotArrayEagerRelease):
    def __init__(self, length):
        super().__init__(length)
        self.compact_calls = 0

    def _compact(self, index):
        self.compact_calls += 1
        super()._compact(index)


class _CountingLazy(SnapshotArrayLazyRelease):
    def __init__(self, length):
        super().__init__(length)
        self.compact_calls = 0

    def _compact(self, index):
        self.compact_calls += 1
        super()._compact(index)


for probe_length in (10, 5_000):
    eager_probe = _CountingEager(probe_length)
    eager_probe.snap()
    eager_probe.compact_calls = 0
    eager_probe.release(0)
    assert eager_probe.compact_calls == probe_length

    lazy_probe = _CountingLazy(probe_length)
    lazy_probe.snap()
    lazy_probe.compact_calls = 0
    lazy_probe.release(0)
    assert lazy_probe.compact_calls == min(SnapshotArrayLazyRelease._SWEEP_BATCH, probe_length)

# --- Part 3: diff() cost does not grow with length ---
def _diff_journal_scan_size(a_arr, a, b) -> int:
    """Independently recomputes, from a_arr._journal alone, how many (interval, index) entries diff(a, b)
    must visit -- the same quantity its own union loop touches, measured from the outside."""
    return sum(len(a_arr._journal[s]) for s in range(a + 1, b + 1))


for probe_length in (5, 50_000):
    diff_probe = SnapshotArray(probe_length)
    diff_probe.set(1 % probe_length, -1)
    diff_probe.snap()                    # snapshot 0 -- before the diffed range, must not be scanned
    diff_probe.set(2 % probe_length, -1)
    diff_probe.snap()                    # snapshot 1
    a = diff_probe.snap()                # snapshot 2, a = 2 -- nothing set since snapshot 1
    diff_probe.set(0, 1)
    diff_probe.set(3 % probe_length, 9)
    b = diff_probe.snap()                # snapshot 3, b = 3
    diff_probe.snap()                    # snapshot 4 -- after the diffed range, must not be scanned either
    assert _diff_journal_scan_size(diff_probe, a, b) == 2
    assert diff_probe.diff(a, b) == {0: (0, 1), 3 % probe_length: (0, 9)}

# --- Part 3: diff() against the brute force's full-array comparison, over many random workloads ---
diff_checks = 0
for seed in range(200):
    rng = random.Random(10_000 + seed)
    length = rng.randint(2, 8)
    sol = SnapshotArray(length)
    brute = _BruteForceSnapshotArray(length)
    for _ in range(rng.randint(10, 40)):
        if rng.random() < 0.7:
            index, val = rng.randrange(length), rng.randint(-9, 9)
            sol.set(index, val)
            brute.set(index, val)
        else:
            assert sol.snap() == brute.snap()
    taken = len(brute._snapshots)
    if taken >= 2:
        for _ in range(5):
            a, b = sorted(rng.sample(range(taken), 2))
            assert sol.diff(a, b) == brute.diff(a, b), (seed, a, b)
            diff_checks += 1
assert diff_checks > 500

print("all checks passed")
```

</details>

</details>
