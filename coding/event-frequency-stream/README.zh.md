# 事件流中出现最频繁的事件

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 哈希表、滑动窗口、流式算法 | ★★★★☆ | 中等 | SWE · MLE · Applied AI · Intern | hash-map, sliding-window, frequency-counting, streaming, space-saving, heavy-hitters | 3 个部分 / 45 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

*事件*（event）是一个 `(message: str, timestamp: int)` 二元组；时间戳在输入顺序中是非递减的，就像日志文件或真实的事件流一样。

### Part 1 —— 最频繁的消息及其时间戳

```py
def most_frequent(events: list[tuple[str, int]]) -> tuple[str, list[int]]: ...
```

给定一个固定、非空的事件列表，返回出现次数最多的消息，以及它每次出现的时间戳——按它们在 `events` 中出现的顺序排列。如果有两个或更多消息在出现次数上并列最高，返回其中在 `events` 里第一次出现得最早的那一个。本页其他部分调用 `most_frequent` 时，`events` 永远不会为空；如果直接对 `[]` 调用它，会抛出 `ValueError`。

```text
events = [("login", 1), ("click", 2), ("login", 4), ("click", 7), ("purchase", 9)]

most_frequent(events) == ("login", [1, 4])
# "login" 和 "click" 都出现了两次，是最高的出现次数；"login" 的第一次出现（时间戳 1）早于 "click"
# 的第一次出现（时间戳 2），所以 "login" 赢得了这次并列。"purchase" 只出现一次，与结果无关。
```

### Part 2 —— 同样的问题，换成滑动窗口

现在事件是逐个到达的。`WindowTopMessage` 只保留它见过的最后 `n` 个事件：当一个新事件到达、且当前已经持有 `n` 个事件时，最先到达的那个持有事件——也就是当前所有持有事件里最早到达的一个——会先被淘汰，然后新事件才被插入。`n` 是一个正整数，在对象的整个生命周期内固定不变。一条消息的*计数*（count）是当前持有的事件中携带这条消息的事件个数。

```py
class WindowTopMessage:
    def __init__(self, n: int): ...
    def add(self, message: str, timestamp: int) -> tuple[str, int]: ...   # (message, count) after the arrival
```

每次 `add` 之后，返回计数最高的消息以及这个计数。并列的打破规则由“变化的先后”决定，精确定义如下：给整个运行过程中每一次单独的计数变化编号——每次淘汰对应一次减一，每次到达的插入对应一次加一——按它们发生的先后顺序编号（如果一次到达自带一次淘汰，那次淘汰的编号排在这次到达的插入之前）。在计数并列最高的消息里，返回其最近一次变化编号最大的那一个。由于一次淘汰的编号总是排在同一次到达的插入之前，如果到达的消息本身也并列最高，那么 `add` 返回的永远是它。每次操作都必须是均摊 $O(1)$，与 `n` 以及历史上出现过的不同消息数量都无关。

用 `n = 3` 逐步演算：

```text
add("login", 1)      -> ("login", 1)      # 持有：login
add("click", 2)      -> ("click", 1)      # 持有：login、click             （在 1 处并列；click 最近变化）
add("login", 3)      -> ("login", 2)      # 持有：login、click、login
add("purchase", 4)   -> ("purchase", 1)   # 淘汰 login@1；持有：click、login、purchase
                                           # login 的计数降到 1，与 click、purchase 一起并列在 1；
                                           # 最大值从 2 降到 1，而 purchase——这次到达的消息——
                                           # 赢得了这次并列
add("purchase", 6)   -> ("purchase", 2)   # 淘汰 click@2；持有：login、purchase、purchase
```

### Part 3 —— 无界的流，有限的内存

现在流是无界的，消息逐个到达，不再带时间戳，而内存最多只能为 `k` 条不同的消息保留计数器——这就是 Space-Saving 算法。`SpaceSaving` 一开始什么消息都不监控；每当一条未被监控的消息到达、且已经有 `k` 条消息受监控时，就淘汰当前计数器最低的那条受监控消息来腾出位置——如果多条消息并列最低，就淘汰其中在这个计数值上停留时间最长的一条——到达的这条消息的计数器不是从零开始，而是从被淘汰消息的计数器值继承而来（精确的赋值规则在参考解答中推导，而不是断言）。`k` 是一个正整数，在对象的整个生命周期内固定不变。

```py
class SpaceSaving:
    def __init__(self, k: int): ...
    def add(self, message: str) -> None: ...
    def top(self, m: int) -> list[tuple[str, int]]: ...   # monitored messages by reported count, descending; ties by message
```

记 $f(x)$ 为 `add` 目前为止处理过的 $N$ 条消息中，消息 $x$ 真实出现的次数；记 $c(x)$ 为消息 $x$ 受监控期间 `SpaceSaving` 为它汇报的计数器值。实现必须在流中的任意时刻保证：

- *高频项覆盖。*（Heavy-hitter coverage.）任何满足 $f(x) > N / k$ 的消息——称为*高频项*（heavy hitter）——都在受监控的消息之中。
- *高估有界。*（Bounded overestimation.）对每条受监控的消息 $x$，都有 $f(x) \le c(x) \le f(x) + N / k$。

`top(m)` 按汇报的计数降序返回 `m` 条受监控的消息；计数并列的消息按消息字符串升序排列。如果受监控的消息不足 `m` 条，就返回全部受监控的消息。

用 `k = 3`、消息按 `a, b, c, d, b, a, e` 的顺序到达来演算：

```text
到达        被淘汰       之后受监控的消息（汇报的计数）
a          --          a: 1
b          --          a: 1, b: 1
c          --          a: 1, b: 1, c: 1                   # 表现在满了
d          a（@1）      b: 1, c: 1, d: 2                   # d 继承 a 的计数 1，再加上这次出现
b          --          b: 2, c: 1, d: 2                   # b 已经受监控——直接加一
a          c（@1）      b: 2, d: 2, a: 2                   # a 继承 c 的计数 1，再加上这次出现
e          d（@2）      b: 2, a: 2, e: 3                   # e 继承 d 的计数 2，再加上这次出现

top(3) == [("e", 3), ("a", 2), ("b", 2)]
# 这 7 次到达的真实计数：a: 2, b: 2, c: 1, d: 1, e: 1（N = 7，N / k = 2.33）
# 每条受监控的消息都满足 f <= c <= f + N / k；"e" 是最紧的一个：1 <= 3 <= 3.33
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有两点值得先确认：Part 1 的并列规则是按最早的首次出现打破，而不是按消息字符串本身，也不是按并列本身的插入顺序；Part 2 的“最频繁”是在每一次到达和每一次淘汰之后都精确重新计算的，不是平滑或衰减过的速率。

### Part 1

一次遍历里就为每条消息同时积累三样东西：它的累计计数、它每次出现的时间戳列表（因为 `events` 是从左到右扫描的，天然就是有序的），以及它首次出现的下标——用 `dict.setdefault` 只记录一次，因为 `setdefault` 只在某个键还不存在时才写入，所以后面再出现同一条消息也不会覆盖它。胜出者是 `(count, -first_index)` 这个二元组最大的消息：先比较计数，并列时再比较首次出现下标——取负数是为了让 `max` 依然能找到它。任何两条消息都不可能在这两个坐标上同时相同——它们在 `events` 里各自占据不同的位置，因而首次出现的下标也必然不同——所以这个比较永远不需要再加一层并列规则。

```python
def most_frequent(events: list[tuple[str, int]]) -> tuple[str, list[int]]:
    if not events:
        raise ValueError("events must be non-empty")
    counts: dict[str, int] = {}
    timestamps: dict[str, list[int]] = {}
    first_index: dict[str, int] = {}
    for i, (message, ts) in enumerate(events):
        counts[message] = counts.get(message, 0) + 1
        timestamps.setdefault(message, []).append(ts)
        first_index.setdefault(message, i)   # NOTE: setdefault -- keeps the first occurrence, never overwrites it
    best = max(counts, key=lambda m: (counts[m], -first_index[m]))
    return best, timestamps[best]
```

对 $N$ 个事件来说，这个循环是 $O(N)$；三个字典加起来一共保存 $O(N)$ 个条目；最后的 `max` 最多扫描 $N$ 个不同的消息，不会改变这个量级。

### Part 2

如果每次到达后都靠扫描全部持有消息来重新算最大值，每次 `add` 就要花 $O(n)$。要做到 $O(1)$，还需要增量维护两样状态：`_count`，记录每条持有消息当前的计数；以及 `_bucket`，一种*计数的计数*（count of counts）——`_bucket[c]` 是当前计数恰好为 `c` 的消息集合，用一个纯粹当作有序集合来用的 `dict`（`dict[str, None]`）保存，这样插入一个键就会把它追加到已有的所有键之后。`_max_count` 缓存当前的最大值，这样它就永远不需要靠扫描来找。

一次到达总是让恰好一条消息的计数加一（`_increment`）：把它从 `_bucket[old]` 里删掉（如果这个桶因此变空就一并删掉这个桶），把它加进 `_bucket[new]`，如果 `new` 超过了 `_max_count` 就更新它。在这里 `_max_count` 只会变大，不会变小。

一次淘汰总是让恰好一条消息的计数减一（`_decrement`），这个方向才是需要证明的。如果被淘汰消息原来的计数低于 `_max_count`，或者恰好在 `_max_count` 处还有别的幸存者与它并列，最大值都不受影响。唯一需要改变 `_max_count` 的情形，是被淘汰消息恰好是 `_bucket[_max_count]` 里唯一的成员，把这个桶清空了——即便如此，新的最大值也只可能是 `_max_count - 1`：被减一的这条消息本身，就在这同一次操作里落进了 `_bucket[_max_count - 1]`，除非它的新计数是 `0`，而这只会发生在 `_max_count` 原本是 `1` 的时候，也就是窗口刚好变得空无一物，这个状态会被 `add` 里紧接着的、这次到达自己的加一操作修复。所以这里 `_max_count` 总是恰好降低 1，而不是靠扫描去找真正的新最大值。

打破并列的规则——汇报计数变化最近的那条消息——正是每个桶要用 `dict` 而不是普通 `set` 的原因：每次插入一个桶都发生在它的末尾（对一个还不存在的键执行 `dict[message] = None` 会把它追加到最后），所以并列的一组消息里变化最近的那一个，就是 `next(reversed(_bucket[_max_count]))`——这是对 dict 最后一个键的 $O(1)$ 读取，不需要扫描这个桶。有一条表面相似、实则不同的规则——按哪条消息*最近一次出现的时间戳*最晚来打破并列，而不是按哪条消息的计数变化得最近——会破坏这一点：一条消息被减一时，它的出现时间戳并不会跟着变（它不是刚刚出现，而是正要离开），所以它得被重新插入到 `_bucket[old - 1]` 里、按它那个旧时间戳在已有成员中该在的位置——一般来说是桶的中间，而不是末尾——而找到这个位置就不再是 $O(1)$ 了。

`add` 在窗口已满时，总是先淘汰、再插入，这直接对应了题目里的并列规则：到达消息的这次加一，在 `add` 返回的那一刻永远是最近的一次变化，所以只要它参与了最大值的并列，赢的总是它。

```python
from collections import deque


class WindowTopMessage:
    """Reports the most frequent message among the last n held events after every arrival."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._window: deque[tuple[str, int]] = deque()   # held events, oldest first
        self._count: dict[str, int] = {}
        self._bucket: dict[int, dict[str, None]] = {}    # count -> ordered set of messages at that count
        self._max_count = 0

    def _increment(self, message: str) -> None:
        old = self._count.get(message, 0)
        new = old + 1
        if old:
            del self._bucket[old][message]
            if not self._bucket[old]:
                del self._bucket[old]
        self._count[message] = new
        self._bucket.setdefault(new, {})[message] = None   # NOTE: appended last -- most recently changed
        if new > self._max_count:
            self._max_count = new

    def _decrement(self, message: str) -> None:
        old = self._count[message]
        new = old - 1
        del self._bucket[old][message]
        emptied = not self._bucket[old]
        if emptied:
            del self._bucket[old]        # NOTE: drop the empty bucket, or a later self._max_count could
                                          # point at a bucket that no longer holds any message
        if new:
            self._count[message] = new
            self._bucket.setdefault(new, {})[message] = None
        else:
            del self._count[message]
        if old == self._max_count and emptied:
            self._max_count -= 1         # NOTE: derived in the text -- the new maximum is exactly old - 1

    def add(self, message: str, timestamp: int) -> tuple[str, int]:
        if len(self._window) == self._n:
            evicted_message, _ = self._window.popleft()   # NOTE: eviction is numbered before insertion
            self._decrement(evicted_message)
        self._window.append((message, timestamp))
        self._increment(message)
        top = next(reversed(self._bucket[self._max_count]))
        return top, self._max_count
```

`add` 所做的每一件事——若干次字典查找、插入和删除，一次 `deque` 的入队和至多一次出队，以及一次 `next(reversed(...))`——都是均摊 $O(1)$：dict 的取、存、删本身就是均摊 $O(1)$，而读取它最近插入的键代价也不会更高，因为这是直接从 dict 自身的插入顺序末尾读出来的，不是靠扫描找到的。这一切都与 `n`、以及历史上出现过多少种不同的消息无关。

### Part 3

`SpaceSaving` 复用了 Part 2 那套“计数的计数”分桶技术，只是现在追踪的是最小值而不是最大值；但这里的淘汰不是给某个幸存者减一，而是把一条受监控的消息整个换成另一条不同的消息。

题目要求的两条保证，都由关于计数器的两个事实推出。

第一，无论走哪个分支，每次 `add` 都恰好让所有计数器的总和加 $1$：给一个已有的计数器加一，总和加 $1$；新建一个从 $1$ 开始的计数器，总和也加 $1$；淘汰时则是去掉一个值为 $m$（最小值）的计数器、加上一个值为 $m + 1$ 的计数器，净变化还是 $+1$。所以调用 $N$ 次之后，（至多 $k$ 个）计数器的总和恰好是 $N$；而一旦表被填满，$k$ 个数之和为 $N$，由鸽笼原理，其中的最小值至多是 $N / k$。

第二，一条受监控消息的计数器永远不会低于它目前为止的真实出现次数；一条未受监控消息目前为止的真实出现次数，也永远不会超过当前受监控计数器里的最小值。这两条在一开始都成立（什么都还没发生，什么都没被监控），并且在每一种更新下都能延续下去：给一条受监控消息加一，恰好跟上它自己新增的那一次出现；一个新计数器从 $1$ 开始，对应的真实历史计数恰好是 $0$，因为只有在已经发生过淘汰、位置已经不够的情况下，一条消息才可能“未受监控”，而只要还有空位，就不会发生淘汰；淘汰一条计数器值为 $m$ 的消息 $z$（由同一条结论用在 $z$ 身上、早一步成立，它的真实计数本来就至多是 $m$）去让新消息 $x$ 从 $m + 1$ 开始，让 $x$ 的计数器比它自己刚刚这一次出现领先一步，而 $z$ 的真实计数依然至多是 $m$，此时要拿去比较的最小值也不可能低于 $m$。

题目的两条保证由此直接得到。如果某条消息的真实计数超过了 $N / k$、却一直未受监控，那么它的真实计数就必须至多等于受监控计数器里的最小值，而这个最小值至多是 $N / k$——矛盾；所以真实计数超过 $N / k$ 的消息一定受监控。再看一条受监控的消息 $x$：设 $t_0$ 为 $x$ 最近一次被插入的那一步， $m \le N / k$ 为那一刻表里的最小值。从 $t_0$ 开始，`add` 每碰到一次 $x$ 真实出现（包括 $t_0$ 这一次本身）都会让它的计数器加一，所以 $x$ 最终的计数器是 $m$ 加上这个出现次数；而 $x$ 的真实总次数至少也是这同一个出现次数（$t_0$ 之前如果还有过出现，只会让它更多），所以计数器超出真实计数的部分至多是 $m \le N / k$——而 $f(x) \le c(x)$ 这一半，上面第一个事实已经给出。

```python
class SpaceSaving:
    """Space-Saving with a stream-summary structure: buckets keyed by counter value, each an ordered
    set of the messages currently holding that value, plus O(1) access to the minimum bucket."""

    def __init__(self, k: int) -> None:
        self._k = k
        self._count: dict[str, int] = {}
        self._bucket: dict[int, dict[str, None]] = {}
        self._min_count = 0

    def _reinsert(self, message: str, new_count: int) -> None:
        old_count = self._count.get(message)
        if old_count is not None:
            del self._bucket[old_count][message]
            if not self._bucket[old_count]:
                del self._bucket[old_count]
        self._count[message] = new_count
        self._bucket.setdefault(new_count, {})[message] = None

    def add(self, message: str) -> None:
        if message in self._count:
            old = self._count[message]
            self._reinsert(message, old + 1)
            if old == self._min_count and not self._bucket.get(old):
                self._min_count = old + 1   # NOTE: this message just left the sole occupant of the minimum bucket
            return
        if len(self._count) < self._k:
            self._reinsert(message, 1)
            self._min_count = 1             # NOTE: a fresh counter starts at 1, the least any counter can be
            return
        m = self._min_count
        evicted = next(iter(self._bucket[m]))   # NOTE: longest-held at the minimum -- the statement's tie-break
        del self._count[evicted]
        del self._bucket[m][evicted]
        if not self._bucket[m]:
            del self._bucket[m]
            self._min_count = m + 1         # NOTE: derived in the text -- the arriving message witnesses m + 1
        self._reinsert(message, m + 1)

    def top(self, m: int) -> list[tuple[str, int]]:
        ranked = sorted(self._count.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:m]
```

每次 `add` 只做若干次字典操作，再加上淘汰这条路径上的一次 `next(iter(...))`，从最小值的桶里任取一个元素——均摊 $O(1)$，和 Part 2 一样，原因也一样。`top(m)` 要对至多 $k$ 条受监控的消息排序， $O(k \log k)$，与流已经运行了多久无关。

用一个 `(count, message)` 的堆、配合惰性删除——每次计数器变化就压入一个新的二元组，弹出时跳过已经过期的旧条目——是一种更简单的替代方案，每次 `add` 只需要 $O(\log k)$，代价是堆本身会在两次清理之间膨胀到超过 $k$ 个条目。上面的分桶结构完全避免了这一点：它任何时候持有的计数器都不会超过 $k$ 个，恰好符合题目“有限内存”的要求。

### 追问

- **Misra–Gries。** 只保留 $k - 1$ 个计数器，表满时不是把被淘汰计数器的值提升给新来者，而是把每一个计数器都减一（减到零的直接丢弃），这就让 Misra–Gries 成了 Space-Saving 低估版本的对偶：每个受监控的计数器都满足 $c(x) \le f(x)$，而且对称地，真实计数超过 $N / k$ 的消息同样保证会被监控。
- **Count-Min sketch。** 干脆放弃追踪消息的身份——用 $d$ 个相互独立的哈希函数，把每条消息都映射到 $w$ 个计数器中的一个，每次出现就把对应计数器加一，某条消息的估计值取它 $d$ 个计数器里的最小值，因为哈希碰撞只会把计数抬高，不会拉低——这是用精确的 top-$k$ 列表换来一个点查询估计器：取 $w = \lceil e / \varepsilon \rceil$、$d = \lceil \ln(1 / \delta) \rceil$，估计值超出真实值 $\varepsilon N$ 以上的概率至多是 $\delta$。要知道*到底是哪些*消息频繁出现，还需要一个独立的结构，比如用同一条流再喂给一个 Space-Saving。
- **合并来自多台机器的摘要。** 合并两台机器各自的 Space-Saving 表，并不只是把它们共有的计数器加起来：一条消息可能只在一台机器上受监控，却在另一台机器上真实出现过、只是没被记下来，所以每台机器的贡献都要先加上这台机器自己的误差上界（$N_{\text{machine}} / k$）再相加——用更紧的单机误差换来更松的合并后误差。
- **按时间开窗口。** 把固定的个数 `n` 换成固定的时长——比如最近五分钟，而不是最近 `n` 个事件——仍然是从一个按时间戳排序的 deque 前端淘汰，但现在一次到达可能一次淘汰零个、一个或好几个事件（所有已经过期的），而不再是恰好一个。`_decrement` 里分桶的记账逻辑完全不变，只是要为每个被淘汰的事件各跑一次，所以整个结构仍然是每个事件均摊 $O(1)$，只是不再是每次 `add` 调用均摊 $O(1)$。
- **检测突发的峰值。** 同时跑两个窗口，一个短的（最近一分钟）、一个长的（最近一小时），把一条消息在短窗口里的速率和它在长窗口里的速率相比较，就能捕捉到一条正在突然走红的消息，与它整体是否热门无关——单一窗口的原始计数是没法把“突发峰值”和“本来就一直很热门”区分开的。

<details>
<summary>验证代码（可运行）</summary>

```python
import random
from collections import Counter

# --- Part 1: the worked example ---
events_ex = [("login", 1), ("click", 2), ("login", 4), ("click", 7), ("purchase", 9)]
assert most_frequent(events_ex) == ("login", [1, 4])

try:
    most_frequent([])
    assert False, "expected ValueError"
except ValueError:
    pass


def _brute_force_most_frequent(events):
    """Independent restatement of Part 1's rule: count with collections.Counter, break ties by the
    smallest index of first occurrence, then collect that message's timestamps in input order."""
    if not events:
        raise ValueError("events must be non-empty")
    counts = Counter(m for m, _ in events)
    first_index = {}
    for i, (m, _) in enumerate(events):
        first_index.setdefault(m, i)
    best_count = max(counts.values())
    candidates = [m for m, c in counts.items() if c == best_count]
    winner = min(candidates, key=lambda m: first_index[m])
    return winner, [ts for msg, ts in events if msg == winner]


for seed in range(400):
    rng = random.Random(seed)
    alphabet = [f"m{i}" for i in range(rng.randint(1, 6))]
    ts = 0
    evs = []
    for _ in range(rng.randint(1, 30)):
        ts += rng.randint(0, 3)
        evs.append((rng.choice(alphabet), ts))
    assert most_frequent(evs) == _brute_force_most_frequent(evs), (seed, evs)

# --- Part 2: the worked example, traced step by step (n = 3) ---
wtm = WindowTopMessage(3)
trace = [
    ("login", 1, ("login", 1)),
    ("click", 2, ("click", 1)),
    ("login", 3, ("login", 2)),
    ("purchase", 4, ("purchase", 1)),   # evicts login@1 -- the maximum falls from 2 to 1
    ("purchase", 6, ("purchase", 2)),   # evicts click@2
]
for message, timestamp, expected in trace:
    assert wtm.add(message, timestamp) == expected, (message, timestamp)


def _brute_force_window(events, n):
    """Independent restatement of Part 2's rule: after every arrival, recompute counts from scratch
    over the currently held events, and track each message's step of last change -- incremented on
    every arrival and every eviction that involves it -- completely separately from WindowTopMessage."""
    window = deque()
    last_change_step = {}
    step = 0
    results = []
    for message, timestamp in events:
        if len(window) == n:
            evicted, _ = window.popleft()
            step += 1
            last_change_step[evicted] = step
        window.append((message, timestamp))
        step += 1
        last_change_step[message] = step
        counts = Counter(m for m, _ in window)
        best_count = max(counts.values())
        candidates = [m for m, c in counts.items() if c == best_count]
        results.append((max(candidates, key=lambda m: last_change_step[m]), best_count))
    return results


total_steps = 0
for seed in range(80):
    rng = random.Random(1000 + seed)
    alphabet = [f"m{i}" for i in range(rng.randint(1, 5))]
    n = rng.choice([1, 2, 3, 5, 8, 13])
    ts = 0
    evs = []
    for _ in range(rng.randint(50, 150)):
        ts += rng.randint(0, 2)
        evs.append((rng.choice(alphabet), ts))
    expected = _brute_force_window(evs, n)
    wtm2 = WindowTopMessage(n)
    got = [wtm2.add(m, t) for m, t in evs]
    assert got == expected, (seed, n)
    total_steps += len(evs)
assert total_steps > 2000

# --- Part 3: the worked example, traced state by state (k = 3) ---
ss = SpaceSaving(3)
trace3 = [
    ("a", {"a": 1}),
    ("b", {"a": 1, "b": 1}),
    ("c", {"a": 1, "b": 1, "c": 1}),          # table now full
    ("d", {"b": 1, "c": 1, "d": 2}),          # evicts a@1 -- d inherits count 1, plus this occurrence
    ("b", {"b": 2, "c": 1, "d": 2}),          # b already monitored -- plain increment
    ("a", {"b": 2, "d": 2, "a": 2}),          # evicts c@1 -- a inherits count 1, plus this occurrence
    ("e", {"b": 2, "a": 2, "e": 3}),          # evicts d@2 -- e inherits count 2, plus this occurrence
]
for message, expected_state in trace3:
    ss.add(message)
    assert dict(ss.top(3)) == expected_state, (message, dict(ss.top(3)), expected_state)
assert ss.top(3) == [("e", 3), ("a", 2), ("b", 2)]

true_counts_ex = Counter(m for m, _ in trace3)
N_ex, k_ex = len(trace3), 3
assert sum(dict(ss.top(3)).values()) == N_ex   # the sum-of-counters invariant, derived in the text
for message, c in ss.top(3):
    f = true_counts_ex[message]
    assert f <= c <= f + N_ex / k_ex

# a table that never fills (k at least the number of distinct messages) counts exactly, no error at all
rng = random.Random(7)
alphabet_small = [f"m{i}" for i in range(12)]
stream_small = [rng.choice(alphabet_small) for _ in range(500)]
ss_exact = SpaceSaving(len(alphabet_small))
for message in stream_small:
    ss_exact.add(message)
assert dict(ss_exact.top(len(alphabet_small))) == dict(Counter(stream_small))
assert len(ss_exact.top(10_000)) == len(set(stream_small))   # top(m) past the monitored count returns all of it


def _zipf_like_stream(rng, n_events, n_messages):
    """A synthetic stream over n_messages distinct names, weighted 1 / rank, so a handful of messages
    dominate -- unlike a uniform stream, and like most real event logs."""
    messages = [f"m{i}" for i in range(n_messages)]
    weights = [1.0 / (i + 1) for i in range(n_messages)]
    return rng.choices(messages, weights=weights, k=n_events)


rng = random.Random(2024)
N = 20_000
stream = _zipf_like_stream(rng, N, n_messages=300)
true_counts = Counter(stream)

for k in (10, 30, 100):
    ss_big = SpaceSaving(k)
    for message in stream:
        ss_big.add(message)
    monitored = ss_big.top(k)               # <= k monitored messages, so top(k) returns all of them
    assert len(monitored) <= k
    for (msg_a, count_a), (msg_b, count_b) in zip(monitored, monitored[1:]):
        assert count_a >= count_b and (count_a > count_b or msg_a < msg_b)   # top() ordering

    threshold = N / k
    monitored_counts = dict(monitored)
    assert sum(monitored_counts.values()) == N              # the sum-of-counters invariant, derived in the text
    assert min(monitored_counts.values()) <= threshold + 1e-9   # the minimum-counter bound, derived in the text
    heavy_hitters = [m for m, f in true_counts.items() if f > threshold]
    assert heavy_hitters   # the check must actually exercise the guarantee, not pass it vacuously
    for message in heavy_hitters:
        assert message in monitored_counts, (k, message, true_counts[message], threshold)
    for message, c in monitored_counts.items():
        f = true_counts.get(message, 0)
        assert f <= c <= f + threshold + 1e-9, (k, message, f, c, threshold)

print("all checks passed")
```

</details>

</details>
