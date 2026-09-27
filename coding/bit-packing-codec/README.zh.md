# 位打包编码器与解码器

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 位运算与编码长度公式 | ★★★☆☆ | 中等 | SWE · RE · Intern | bit-manipulation, varint, zigzag-encoding, frame-of-reference, serialisation | 3 个部分 / 45 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

下面每一部分都实现一种二进制编解码器（codec）：一对函数，把一串 Python 整数变成一个 `bytes` 对象，再变
回来，使得对编码器产生的字节解码，能精确还原出原始序列。`bytes` 是 Python 中不可变的 8 位值序列，每个
值介于 `0` 到 `255` 之间；在一个字节内，*最高有效位*（most significant bit，MSB）是权重为 $2^7 = 128$ 的
那一位，*最低有效位*（least significant bit，LSB）是权重为 $2^0 = 1$ 的那一位。对非负整数 $v$，
`v.bit_length()`——下文记作 $\mathrm{bitlen}(v)$——是它二进制表示中不含前导零的位数：$\mathrm{bitlen}(0) = 0$，
$\mathrm{bitlen}(7) = 3$，因为 $7$ 的二进制是 `111`。

### Part 1 —— 定宽打包

```py
def pack(values: list[int], k: int) -> bytes: ...
def unpack(data: bytes, k: int, count: int) -> list[int]: ...
```

字段宽度 $k$ 满足 $1 \le k \le 32$，`values` 中的每个值都必须满足 $0 \le v < 2^k$；如果 $k$ 超出这个范围，
或者有值放不进 $k$ 个比特，`pack` 就抛出 `ValueError`。它把每个值写成一个 $k$ 比特的字段，最高位在前，
各字段之间没有间隙：字段 $0$ 的最高位就是字节 $0$ 的最高位，字段 $1$ 紧接在字段 $0$ 结束的下一个比特位置
开始——只要 $k$ 不能整除当前的比特偏移量，就会跨进下一个字节——如此继续，直到最后一个字段。这
$n = \mathrm{len(values)}$ 个字段一共用去 $kn$ 个比特；`pack` 接着把最后一个字节里剩下的低位补上 $0$，
直到凑满一个整字节，所以它的返回值总是恰好 $\lceil kn / 8 \rceil$ 个字节（`values` 为空时是 $0$ 个字节）。
`unpack` 是它精确的逆过程：给定同样的 `k` 和 `count = len(values)`，它按 `pack` 写入的顺序，从 `data`
里依次读出 `count` 个连续的 $k$ 比特字段。如果 `k` 超出范围，或者 `len(data)` 不恰好等于
$\lceil k \cdot \mathrm{count} / 8 \rceil$——也就是 `pack` 对这一组 `k`、`count` 唯一可能产生的长度——它
就抛出 `ValueError`，因为其他任何长度都不可能是 `pack` 的输出。

```text
values = [5, 1, 7], k = 3
5 = 101，1 = 001，7 = 111（每个值都写成一个 3 比特字段，最高位在前）

拼接后的字段：      101 001 111             （共 9 个比特）
补齐到 2 个字节：    10100111 10000000       （末尾补上 7 个 0 比特）
对应字节：           0xa7       0x80

pack([5, 1, 7], 3) == b"\xa7\x80"
unpack(b"\xa7\x80", 3, 3) == [5, 1, 7]
```

### Part 2 —— 变长整数

无符号 LEB128（*varint*）把一个非负整数 $v$ 编码成一个或多个字节，最低有效的 $7$ 位一组最先写出：只要
$v$ 还有非零的比特没写出，就取它当前的低 $7$ 位作为一个字节的低 $7$ 位，如果后面还有更多组，就把这个
字节的*延续位*（continuation bit）——第 $7$ 位，掩码 `0x80`——置为 $1$，如果这是最后（也是最高有效）
一组所在的字节，就置为 $0$，然后把这 $7$ 位从 $v$ 中右移丢弃；$v = 0$ 时仍然会写出一个字节 `0x00`，
因为这个过程总会至少写一组。

```py
def encode_uvarints(values: list[int]) -> bytes: ...
def decode_uvarints(data: bytes) -> list[int]: ...   # ValueError if the data ends inside a number
def zigzag(n: int) -> int: ...                       # 0, -1, 1, -2, 2, ... -> 0, 1, 2, 3, 4, ...
def unzigzag(z: int) -> int: ...
```

`encode_uvarints` 按顺序把 `values` 里每个值的 varint 编码拼接起来；如果有值是负数，就抛出
`ValueError`。`decode_uvarints` 是它在整个 `data` 上的逆过程：一个接一个地读出 varint，直到字节读完
为止；如果 `data` 在某个数中间就结束了——最后读到的字节延续位仍然是 $1$，却没有更多字节可以延续——就
抛出 `ValueError`。`zigzag` 把一个有符号整数映射成一个非负整数，使得不论正负、只要量级小，映射的结果
也小：它把 $0, -1, 1, -2, 2, \ldots$ 映射成 $0, 1, 2, 3, 4, \ldots$，结果每增大 $1$，输入就在非负和负数
之间交替；`unzigzag` 是它的逆映射，对每个非负整数都有定义。

```text
v = 300 = 0b1_0010_1100   （9 个比特）

300 的低 7 位：0101100 = 44；300 >> 7 = 2（非零，还有更多字节）        -> 字节 0 = 0b1_0101100 = 0xac
2 的低 7 位：  0000010 =  2；  2 >> 7 = 0（没有剩余，是最后一个字节）  -> 字节 1 = 0b0_0000010 = 0x02

encode_uvarints([300]) == b"\xac\x02"
decode_uvarints(b"\xac\x02") == [300]

zigzag：   0  -1   1  -2   2  ...
        -> 0   1   2   3   4  ...
```

### Part 3 —— frame-of-reference 编码块与最优选择

一个*块*（block）是一组要一起编码的非空有符号整数列表；对某个块记 $m = \min(\text{values})$、
$M = \max(\text{values})$。它的 *frame-of-reference* 编码由一个头部和一个负载拼接而成。头部依次存放：
$m$ 本身，编成一个 zigzag varint（所以可以是负数）；*比特宽度*（bit width）$w = \mathrm{bitlen}(M - m)$，
存成单独一个无符号字节（块内所有值都等于 $m$ 时，恰好 $w = 0$）；以及 `len(values)`，编成一个普通
（不经过 zigzag）的 varint，因为个数永远不会是负的。负载只在 $w > 0$ 时才存在，是
`pack([v - m for v in values], w)`——把每个值重新表示成它相对 $m$ 的非负偏移量，根据 $M$ 和 $w$ 的
定义，这个偏移量必然能放进 $w$ 个比特。`encode_for` 在 `values` 为空时抛出 `ValueError`，因为这时 $m$
没有意义；如果 $w$ 会超过 $32$——`pack` 本身对字段宽度的限制——也会抛出 `ValueError`。

```py
def encode_for(values: list[int]) -> bytes: ...
def decode_for(data: bytes) -> list[int]: ...
def for_size(values: list[int]) -> int: ...          # exact size in bytes, computed by formula, without encoding
def best_encoding(values: list[int]) -> str: ...     # "for" or "varint" (zigzag varints, one per value, plus a count varint); ties -> "for"
```

`decode_for` 是 `encode_for` 精确的逆过程：它读取头部，还原出 $m$、$w$ 和个数，然后在 $w > 0$ 时解出
这么多个 $w$ 比特的字段，再给每一个都加回 $m$（$w = 0$ 时，每个值就都是 $m$）。如果 `data` 太短，
装不下头部或者头部所描述的负载，就会抛出 `ValueError`；如果按这个头部本该由 `encode_for` 写出的全部
内容之后，`data` 里还剩下多余的字节——包括 $w = 0$ 的情形，因为这时头部之后按定义完全没有负载，不该
再出现任何字节——也会抛出 `ValueError`。`for_size` 只根据头部
和负载各自的大小公式，就能得到 `encode_for` 会产生的精确字节数，完全不用构造任何字节。本页把
frame-of-reference 拿来对比的替代方案，是把同一个块编码成一串普通的 zigzag varint，不共享任何偏移
量：先是 `len(values)` 编成一个 varint，然后每个值的 `zigzag(v)` 各编成一个 varint，依次排列。
`best_encoding` 返回 `"for"` 和 `"varint"` 中对 `values` 产生更短字节串的那一个——恰好打平时优先选
`"for"`——并且和 `encode_for` 一样，在 `values` 为空时抛出 `ValueError`。

```text
values = [998, 1000, 1005, 1000]
m = min(values) = 998，max(values) = 1005，w = bitlen(1005 - 998) = bitlen(7) = 3
deltas = [v - 998 for v in values] = [0, 2, 7, 2]

头部：   zigzag(998) = 1996，编成 varint：2 个字节
         w = 3，编成一个字节：            1 个字节
         len(values) = 4，编成 varint：   1 个字节
         -> 头部共 2 + 1 + 1 = 4 个字节
负载：   pack([0, 2, 7, 2], 3) -> ceil(3 * 4 / 8) = 2 个字节

for_size(values) == 4 + 2 == 6
decode_for(encode_for(values)) == [998, 1000, 1005, 1000]
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有两点值得先确认：定宽字段的比特顺序——这里是最高位在前，但也有不少实际格式反而是最低位在
前（见后面的追问）——以及下面每种方案怎么知道该在哪里停下。`unpack` 是从外部拿到 `count` 的，因为
一段打包的比特流本身不带任何标记；varint 流则依靠每个 varint 自己的延续位，一直读到 `data` 用完为
止；`decode_for` 则是从 `encode_for` 写入的头部里还原出自己的个数，所以从来不需要另外传入。

### Part 1

`pack` 维护一个比特累加器 `acc`，以及 `nbits`——它当前低位中有效比特的个数。追加一个值时，把 `acc`
左移 $k$ 位，再把这个值 `OR` 进腾出来的低位——这和把 $k$ 个新比特写到一段不断增长的比特串右端是同一
回事——之后只要还有 $8$ 位或更多比特待处理，就把这 `nbits` 位里最靠前的一个字节（`nbits` 先减去 $8$
之后的 `acc >> nbits`）剥离出来追加到输出里；每次取出一个字节之后，都把 `acc` 掩码到只剩下 `nbits`
位，这样它就不会随着打包的值越来越多而无限增长，因为已经写出去的字节不必再被用到。所有值都处理完
之后，还会剩下 $0$ 到 $7$ 个待处理的比特；把它们左移到再多一个字节的高位，也就是 `acc << (8 - nbits)`，
得到的正好就是题目要求的零填充，因为一个整数在自己的数值之上不会有任何比特被置位。

```python
def _packed_size(k: int, count: int) -> int:
    """Bytes pack(values, k) uses for count values: ceil(k * count / 8), via integer arithmetic."""
    return (k * count + 7) // 8


def pack(values: list[int], k: int) -> bytes:
    if not (1 <= k <= 32):
        raise ValueError(f"k must satisfy 1 <= k <= 32, got {k}")
    out = bytearray()
    acc = 0                    # unflushed bits, right-aligned in the low nbits positions
    nbits = 0
    limit = 1 << k
    for v in values:
        if not (0 <= v < limit):
            raise ValueError(f"value {v} does not fit in {k} unsigned bits")   # NOTE: checked before packing
        acc = (acc << k) | v
        nbits += k
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1     # NOTE: drop the bits already flushed, so acc stays a handful of bits
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)    # NOTE: zero-pad the final byte's low bits
    return bytes(out)
```

`unpack` 用同样的思路反过来做：不断吞入整字节，直到待处理的比特数至少有 `k` 位，从累加器里读出最靠前
的 `k` 位，再像 `pack` 一样把剩下的部分掩码保留，因此两边共享同一个不变量——`acc` 低位的 `nbits` 位，
在这一侧永远恰好是尚未被消费的比特，在另一侧则永远恰好是尚未被写出的比特。

```python
def unpack(data: bytes, k: int, count: int) -> list[int]:
    if not (1 <= k <= 32):
        raise ValueError(f"k must satisfy 1 <= k <= 32, got {k}")
    expected = _packed_size(k, count)
    if len(data) != expected:
        raise ValueError(f"expected {expected} bytes for k={k}, count={count}, got {len(data)}")
    values = []
    acc = 0
    nbits = 0
    idx = 0
    mask = (1 << k) - 1
    for _ in range(count):
        while nbits < k:               # NOTE: pull in whole bytes until there are enough bits for one field
            acc = (acc << 8) | data[idx]
            idx += 1
            nbits += 8
        nbits -= k
        values.append((acc >> nbits) & mask)
        acc &= (1 << nbits) - 1
    return values
```

在读取任何内容之前先检查 `data` 的长度很重要：如果不检查，过短的缓冲区会让 `idx` 在处理最后一个字段
的过程中越过 `data` 末尾，而过长的缓冲区则会悄悄地成功返回，忽略掉那些不可能来自 `pack` 的多余尾部
字节。输出的每个字节都恰好由内层 `while` 循环的一次迭代产生（在 `pack` 里）或消费（在 `unpack` 里），
每个值又恰好贡献外层循环的一趟，所以两个函数都运行在 $O(kn / 8) = O(kn)$ 时间——线性于它们自己输出的
大小，而这个大小在 $k$ 固定时又线性于 $n$。这个输出大小可以直接从比特计数得到：$n$ 个 $k$ 比特的字段
一共用去 $kn$ 个比特，补齐到整字节最多再加 $7$ 位，恰好给出 $\lceil kn/8 \rceil$ 个字节——这正是
`_packed_size`，`unpack` 用它在信任 `data` 之前先验证长度，Part 3 也复用它，在不实际打包的情况下就
能算出一个 frame-of-reference 块负载的大小。

### Part 2

`_encode_uvarint` 就是 LEB128 定义本身的直接翻译：`v & 0x7F` 取出低 $7$ 位，`v >>= 7` 把它们丢弃，
循环在第一次不再剩下非零比特时停止——如果 $v = 0$，第一轮迭代就会停止，因为循环体总会至少执行一次，
这就给出了那唯一必需的字节 `0x00`。`_uvarint_size` 则换一种方式，用公式直接算出对应的字节数：对
$v > 0$，上面的循环会写出的组数，是满足 $v \gg 7g = 0$ 的最小 $g$，也就是 $7g \ge \mathrm{bitlen}(v)$，
即 $g = \lceil \mathrm{bitlen}(v)/7 \rceil$；$v = 0$ 则需要这个公式本身预测不出来的那一组，因为
$\mathrm{bitlen}(0) = 0$ 会给出 $\lceil 0/7 \rceil = 0$——这就是 `max(1, ...)` 的由来。
（`-(-v.bit_length() // 7)` 只用地板除法就算出了这个上取整，靠的是标准恒等式
$\lceil a/b \rceil = -\lfloor -a/b \rfloor$。）

```python
def _uvarint_size(v: int) -> int:
    """Bytes _encode_uvarint(v) uses, from the size formula alone: max(1, ceil(bitlen(v) / 7))."""
    return max(1, -(-v.bit_length() // 7))


def _encode_uvarint(v: int) -> bytes:
    if v < 0:
        raise ValueError(f"unsigned varint expects v >= 0, got {v}")
    out = bytearray()
    while True:
        group = v & 0x7F
        v >>= 7
        if v:
            out.append(group | 0x80)       # NOTE: high bit set -- at least one more byte follows
        else:
            out.append(group)              # last group: high bit stays 0
            return bytes(out)


def encode_uvarints(values: list[int]) -> bytes:
    return b"".join(_encode_uvarint(v) for v in values)
```

解码沿相反方向累积，最低有效的一组最先处理：`_read_uvarint` 把每个字节的 $7$ 位负载比特，按一个不断
增长的 `shift` `OR` 进结果里，一旦遇到延续位为 $0$ 的字节就停止。在延续位仍然为 $1$ 的时候 `data` 就
用完了——这个检查在读那个字节*之前*，而不是之后——这正是 `decode_uvarints` 对“数据在某个数中间结束”
的定义。

```python
def _read_uvarint(data: bytes, pos: int) -> tuple[int, int]:
    """Reads one unsigned varint starting at data[pos]. Returns (value, position right after it)."""
    v = 0
    shift = 0
    n = len(data)
    while True:
        if pos >= n:
            raise ValueError("truncated varint data")   # NOTE: ran out of bytes with a continuation pending
        byte = data[pos]
        pos += 1
        v |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return v, pos
        shift += 7


def decode_uvarints(data: bytes) -> list[int]:
    values = []
    pos = 0
    n = len(data)
    while pos < n:
        v, pos = _read_uvarint(data, pos)
        values.append(v)
    return values


def zigzag(n: int) -> int:
    return 2 * n if n >= 0 else -2 * n - 1


def unzigzag(z: int) -> int:
    return z // 2 if z % 2 == 0 else -(z + 1) // 2
```

两个方向的开销都是 $O(\lceil \mathrm{bitlen}(v)/7 \rceil)$（每产生或消费一个字节，循环就执行一轮）；
因此 `encode_uvarints` 和 `decode_uvarints` 的运行时间，都线性于大小公式为 `values` 中所有值预测出的
总字节数，不会更多。

直接用 `_encode_uvarint` 编码一个负的 Python 整数是没有意义的，因为对一个负数不断右移永远不会到达
$0$；常见的退路是把它重新解释成一个定宽的无符号值，而这恰恰是小负数变得昂贵的原因。作为一个 64 位
二进制补码值，$-1$ 就是 $2^{64}-1$，它的 $\mathrm{bitlen}$ 是 $64$，因此一个量级只有 $1$ 的值要花费
$\lceil 64/7 \rceil = 10$ 个 varint 字节。`zigzag` 把小的负数映射成小的非负编码，而不是巨大的编码，
从而避开了这个问题——$\mathrm{zigzag}(-1) = 1$，只占一个字节——输出每增大 $1$，输入就在非负和负数
之间交替，正是上文所说的 $0, -1, 1, -2, 2, \ldots \to 0, 1, 2, 3, 4, \ldots$ 对应关系。`unzigzag`
直接反转这两种情况：偶数编码 $z = 2n$ 来自非负的 $n = z / 2$，奇数编码 $z = -2n - 1$（于是
$z + 1 = -2n$）来自负的 $n = -(z+1)/2$。

本页用的分支写法 `2 * n if n >= 0 else -2 * n - 1`，和真实 varint 库在定宽寄存器上使用的位运算形式
是一致的：对每个 64 位二进制补码的 $n$，都有
$\mathrm{zigzag}(n) = \big((n \ll 1) \oplus (n \gg 63)\big) \bmod 2^{64}$。其中 $n \gg 63$ 是一次算术
（保留符号）右移，$n \ge 0$ 时结果是 $0$，$n < 0$ 时结果对 $2^{64}$ 取模后是全 $1$。和 $0$ 异或不改变
任何东西，所以非负的情形直接给出 $n \ll 1 = 2n$；和全 $1$ 异或则是按位取反，所以负数的情形给出
$\overline{2n} = -2n - 1$（Python 里 `~x` 本身对任意整数 $x$ 就定义成 $-x - 1$）——正是 `zigzag`
自己所走的这两个分支。

### Part 3

`_for_header_width` 把 `encode_for` 和 `for_size` 必须达成一致的那部分计算抽取了出来：$m$，以及让
每个偏移量 $v - m$ 都符合 `pack` 约定的比特宽度 $w$。由于 $v - m$ 的取值范围是从 $0$（在等于 $m$ 的
那个值处）到 $M - m$（在等于 $M$ 的那个值处），根据定义，$w = \mathrm{bitlen}(M - m)$ 正是能容纳这个
最大值的最小字段宽度。它还负责抛出两个函数都必须抛出的那个 `ValueError`：块为空时（此时 $m$ 没有
意义），以及 $w$ 会超过 $32$ 时——这时块的跨度超出了 `pack` 能表示的范围，这种负载编码方式本身就无法
处理。

```python
def _for_header_width(values: list[int]) -> tuple[int, int]:
    """(m, w) shared by encode_for and for_size, after validating the block is non-empty and its
    range still fits pack's own k <= 32 limit."""
    if not values:
        raise ValueError("a frame-of-reference block needs at least one value")
    m = min(values)
    w = (max(values) - m).bit_length()
    if w > 32:
        raise ValueError(f"block range needs {w} bits, more than pack's 32-bit limit")   # NOTE: propagate early
    return m, w


def encode_for(values: list[int]) -> bytes:
    m, w = _for_header_width(values)
    header = _encode_uvarint(zigzag(m)) + bytes([w]) + _encode_uvarint(len(values))
    payload = pack([v - m for v in values], w) if w else b""   # NOTE: w == 0 -- every delta is 0, skip pack
    return header + payload


def decode_for(data: bytes) -> list[int]:
    zm, pos = _read_uvarint(data, 0)
    m = unzigzag(zm)
    if pos >= len(data):
        raise ValueError("truncated frame-of-reference header")
    w = data[pos]
    pos += 1
    count, pos = _read_uvarint(data, pos)
    if w == 0:
        if len(data) != pos:      # NOTE: w == 0 carries no payload -- nothing may follow the header
            raise ValueError("trailing bytes after an all-equal frame-of-reference block")
        return [m] * count
    return [m + d for d in unpack(data[pos:], w, count)]


def for_size(values: list[int]) -> int:
    m, w = _for_header_width(values)
    header = _uvarint_size(zigzag(m)) + 1 + _uvarint_size(len(values))
    payload = _packed_size(w, len(values)) if w else 0
    return header + payload
```

`decode_for` 按相同顺序读出同样的三个头部字段，一旦知道了 $w$，要么把 $m$ 重复 `count` 次直接返回
（同时检查头部之后不再有任何内容，因为一个格式良好的全相等块根本没有负载），要么解出 `count` 个各
$w$ 比特的字段，再给每一个都加回 $m$，恰好撤销 `encode_for` 所做的减法。`for_size` 逐字段地镜像了
`encode_for` 的结构，只是用 `_uvarint_size` 和 `_packed_size` 取代了真正的编码器，所以它完全不构造
任何字节；`for_size(values) == len(encode_for(values))` 成立的原因，和 Part 1 里 `_packed_size` 之于
`pack` 的原因完全一样，只是现在同时用在了头部的两个 varint 和负载上。`_for_header_width` 本身就是
$O(n)$，开销来自 `min`/`max` 扫描；负载按 Part 1 的界是 $O(wn)$，由于 $w \le 32$ 是个不随 $n$
变化的有界值，这也就是 $O(n)$。所以对一个有 $n$ 个值的块，`encode_for` 和 `decode_for` 整体都是
$O(n)$——和 `pack`、`unpack` 本身的界一样——头部那两个 varint 只是在此之上再多花几个字节的功夫。

```python
def _varint_block_size(values: list[int]) -> int:
    """Bytes the alternative scheme uses: a count varint, then one zigzag varint per value."""
    return _uvarint_size(len(values)) + sum(_uvarint_size(zigzag(v)) for v in values)


def best_encoding(values: list[int]) -> str:
    # NOTE: for_size raises on an empty block, so best_encoding does too -- no separate check needed
    return "for" if for_size(values) <= _varint_block_size(values) else "varint"
```

frame-of-reference 只为块的偏移量付出一次代价，也就是 $m$，之后每个值都只需固定的 $w$ 个比特，与这个
值自身的大小无关；普通 varint 不需要为共享的偏移量付出任何代价，但要为每个值自己的量级单独付费。这个
权衡决定了谁会获胜。取值聚集在一个大的公共偏移量附近、范围很窄的一批值——$[100000, 100001, 100000,
100003, 100002]$，提出 $m = 100000$ 之后 $w = \mathrm{bitlen}(3) = 2$——`for_size` 只需要 $7$ 个字节，
而五个独立的 varint 则需要 $16$ 个字节（每个值自身的量级都在 $10^5$ 左右，不论邻居离得多近，都要花
$\lceil 18/7 \rceil = 3$ 个字节，再加上 $1$ 个字节存个数）；`best_encoding` 会选择 `"for"`。一个重尾
（heavy-tailed）的块，$[1, 2, 1, 3, 1000000000]$，情况正好相反：那一个离群值迫使
$w = \mathrm{bitlen}(999999999) = 30$ 个比特成为每一个值的宽度，包括那四个很小的值在内，所以
`for_size` 要花 $22$ 个字节；而 varint 让每个小值只为自己微小的量级付出一个字节，只有那个离群值才
需要为自己的量级（$31$ 个比特，$5$ 个字节）单独付费，一共 $4 + 5 + 1 = 10$ 个字节；`best_encoding`
会选择 `"varint"`。

### 追问

- **对有序数据做差分编码。** 近乎单调的数据——时间戳、累计计数器、有序 ID——并不适合这里给出的
  frame-of-reference，因为即便相邻元素彼此很接近，一个比特宽度也必须覆盖最小值和最大值之间的整个
  跨度；改为编码第一个值再加上每一步的差值（`values[i] - values[i-1]`），每个差值就只受限于*相邻*
  元素之间能漂移多远，这通常比整个块的跨度小得多。只是近似有序的数据仍可能偶尔出现负的差值，所以这些
  差值在变成 varint 之前都要先经过 `zigzag`，和其他任何有符号值一样；如果序列保证严格递增，则可以
  直接对这些差值使用 `encode_uvarints`，因为它们永远不可能是负的。
- **对 SIMD 友好的布局。** `pack` 在整次调用中只按一个宽度 $k$，写出一条不设边界、最高位在前的流。
  真实的位打包格式（Parquet 的 bit-packing、Lucene 的 `PackedInts`、FastPFOR）则会把值切分成固定的
  块——常见的是 $128$ 个——每块各自有自己的宽度 $w$，并且每块都按最低位在前打包。$128$ 并不是一个
  随意选定的整数：对从 $1$ 到 $32$ 的每一个整数 $w$，$128w$ 都同时是 $32$ 和 $64$ 的倍数（因为
  $128 = 4 \times 32 = 2 \times 64$），所以一个 $128$ 个值的块，总是恰好占据整数个 $32$ 位或 $64$
  位的机器字，没有任何一个值的比特会被切在块的边界两侧——这样向量化的解包代码就可以按整字加载一个
  块，把同一小组预先算好的、按通道（lane）使用的移位与掩码常量，一次性应用到 SIMD 寄存器的每个通道
  上，而不必像 `pack` 的累加器那样逐个值顺序遍历。
- **修补离群值（PFOR）。** 就像上面重尾的例子一样，一个离群值就会把整个 frame-of-reference 块的 $w$
  拉高；patched frame-of-reference（PFOR）则改为选择一个能覆盖大多数值的 $w$——通常取块内某个固定的
  分位数——不论是否真的放得下，都按这个更窄的宽度打包每一个值（离群值放不下的高位会被直接丢弃），再
  另外存一份 `(index, true value)` *补丁*（patch）列表，记录那些宽度容纳不下的值，在解包之后用来
  覆盖被截断的结果。代价是少量的异常记录开销——每个离群值一个下标加一个值的 varint——用来换取本该由
  每个非离群值多付出的宽度。
- **错误检测。** 上面所有格式都检测不出数据损坏：`pack` 编码的流中一个比特翻转，会悄无声息地把它
  之后的每个字段都错位；varint 中一个延续位翻转，则会让之后的每个值都失去同步。在负载之后追加一份
  对块字节计算出的校验和——哪怕只是一个廉价的累加和，或者更稳妥地用 CRC-32 这样的标准算法——并让
  解码器在信任其余内容之前先重新计算并比对这份校验和，就能把这种悄无声息的数据损坏，变成解码时立刻
  抛出的 `ValueError`，代价只是每个块多付出几个固定的额外字节。

<details>
<summary>验证代码（可运行）</summary>

```python
import random


def _raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False


# --- Part 1: the worked example, traced bit by bit ---
assert pack([5, 1, 7], 3) == b"\xa7\x80"
assert unpack(b"\xa7\x80", 3, 3) == [5, 1, 7]
assert pack([], 5) == b""
assert unpack(b"", 5, 0) == []


def _brute_pack(values: list[int], k: int) -> bytes:
    """Independent restatement of Part 1's rule: build one big string of '0'/'1' characters and
    slice it into bytes, never touching pack's bit-accumulator code."""
    bits = "".join(format(v, f"0{k}b") for v in values)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def _brute_unpack(data: bytes, k: int, count: int) -> list[int]:
    bits = "".join(format(b, "08b") for b in data)
    return [int(bits[i:i + k], 2) for i in range(0, k * count, k)]


rng = random.Random(0)
for _ in range(2000):
    k = rng.randint(1, 32)
    n = rng.randint(0, 8)
    values = [rng.randint(0, (1 << k) - 1) for _ in range(n)]
    got = pack(values, k)
    assert got == _brute_pack(values, k), (k, values)
    assert len(got) == _packed_size(k, n)
    assert unpack(got, k, n) == values
    assert _brute_unpack(got, k, n) == values

for k in range(1, 33):                          # every field width from 1 to 32 bits, explicitly
    values = [rng.randint(0, (1 << k) - 1) for _ in range(5)]
    assert unpack(pack(values, k), k, len(values)) == values

assert _raises(ValueError, pack, [1], 0)                    # k too small
assert _raises(ValueError, pack, [1], 33)                   # k too large
assert _raises(ValueError, pack, [-1], 3)                   # value negative
assert _raises(ValueError, pack, [8], 3)                    # value == 2**k, out of range
assert _raises(ValueError, unpack, b"\x00", 0, 1)
assert _raises(ValueError, unpack, b"\x00", 33, 1)
assert _raises(ValueError, unpack, b"", 3, 1)                # too short
assert _raises(ValueError, unpack, b"\xa7\x80\x00", 3, 3)    # too long

# --- Part 2: the worked example ---
assert encode_uvarints([300]) == b"\xac\x02"
assert decode_uvarints(b"\xac\x02") == [300]
assert _encode_uvarint(0) == b"\x00"
assert _uvarint_size(300) == 2
assert [zigzag(n) for n in (0, -1, 1, -2, 2)] == [0, 1, 2, 3, 4]
assert [unzigzag(z) for z in (0, 1, 2, 3, 4)] == [0, -1, 1, -2, 2]

# why zigzag: a negative number encoded naively as an "unsigned" 64-bit varint costs 10 bytes,
# against 1 byte for its zigzag code, even though its magnitude is tiny
assert _uvarint_size((-1) % (1 << 64)) == 10
assert _uvarint_size(zigzag(-1)) == 1


def _brute_uvarint(v: int) -> bytes:
    """Independent LEB128 encoder built from repeated divmod by 128, not the solution's shift/mask loop."""
    if v < 0:
        raise ValueError(f"unsigned varint expects v >= 0, got {v}")
    groups = []
    while True:
        v, rem = divmod(v, 128)
        groups.append(rem)
        if v == 0:
            break
    return bytes(g | 0x80 if i < len(groups) - 1 else g for i, g in enumerate(groups))


def _brute_zigzag(n: int) -> int:
    return 2 * n if n >= 0 else -2 * n - 1


def _brute_decode_uvarint(data: bytes, pos: int) -> tuple[int, int]:
    """Independent decoder: collect the 7-bit groups first, then rebuild the value from the most
    significant group down by repeated multiplication, rather than the solution's running shift."""
    groups = []
    while True:
        if pos >= len(data):
            raise ValueError("truncated varint data")
        byte = data[pos]
        pos += 1
        groups.append(byte & 0x7F)
        if not (byte & 0x80):
            break
    v = 0
    for g in reversed(groups):
        v = v * 128 + g
    return v, pos


rng = random.Random(1)
for _ in range(3000):
    v = rng.choice([0, 1, 127, 128, 300]) if rng.random() < 0.1 else rng.randrange(2 ** 90)
    enc = _encode_uvarint(v)
    assert enc == _brute_uvarint(v), v
    assert len(enc) == _uvarint_size(v), v
    assert _read_uvarint(enc, 0) == (v, len(enc))
    assert _brute_decode_uvarint(enc, 0) == (v, len(enc))

assert _raises(ValueError, _encode_uvarint, -1)
assert _raises(ValueError, encode_uvarints, [-1])

rng = random.Random(2)
for _ in range(500):
    n = rng.randint(0, 150)
    values = [rng.choice([0, 2 ** 63, rng.randrange(2 ** 80)]) for _ in range(n)]
    data = encode_uvarints(values)
    assert data == b"".join(_brute_uvarint(v) for v in values)
    assert decode_uvarints(data) == values

assert decode_uvarints(b"") == []
assert _raises(ValueError, decode_uvarints, b"\xac")     # continuation bit set, then nothing
assert _raises(ValueError, decode_uvarints, b"\x80")

MASK64 = (1 << 64) - 1
rng = random.Random(3)
for _ in range(20000):
    n = rng.randint(-(2 ** 63), 2 ** 63 - 1)
    assert zigzag(n) == ((n << 1) ^ (n >> 63)) & MASK64, n
    assert unzigzag(zigzag(n)) == n
for n in (0, -1, 1, -(2 ** 63), 2 ** 63 - 1, 2 ** 62, -(2 ** 62)):
    assert zigzag(n) == ((n << 1) ^ (n >> 63)) & MASK64, n

# --- Part 3: the worked example, m/w/deltas/header traced by hand and checked exactly ---
block = [998, 1000, 1005, 1000]
m, w = _for_header_width(block)
assert (m, w) == (998, 3)
assert [v - m for v in block] == [0, 2, 7, 2]
assert zigzag(m) == 1996 and _uvarint_size(zigzag(m)) == 2      # m as a zigzag varint: 2 bytes
header_size = _uvarint_size(zigzag(m)) + 1 + _uvarint_size(len(block))
assert header_size == 4                                          # 2 (m) + 1 (w) + 1 (count)
payload_size = _packed_size(w, len(block))
assert payload_size == 2                                         # ceil(3 * 4 / 8)
assert for_size(block) == header_size + payload_size == 6
assert len(encode_for(block)) == 6
assert decode_for(encode_for(block)) == block
assert _varint_block_size(block) == 9
assert best_encoding(block) == "for"                              # 6 bytes beats the alternative's 9

assert decode_for(encode_for([7, 7, 7, 7])) == [7, 7, 7, 7]       # w == 0: all four values equal
assert for_size([7, 7, 7, 7]) == 3

# small range around a large offset -- frame-of-reference wins
clustered = [100_000, 100_001, 100_000, 100_003, 100_002]
m_c, w_c = _for_header_width(clustered)
assert w_c == 2                                                   # bitlen(100_003 - 100_000) = bitlen(3)
assert _uvarint_size(zigzag(m_c)) == 3                            # m ~= 10**5 needs 3 varint bytes
assert for_size(clustered) == 7
assert _varint_block_size(clustered) == 16                        # 5 values * 3 bytes each, + 1 for the count
assert best_encoding(clustered) == "for"

# heavy-tailed values (one huge outlier) -- the per-value varint alternative wins
heavy_tailed = [1, 2, 1, 3, 1_000_000_000]
m_h, w_h = _for_header_width(heavy_tailed)
assert w_h == 30                                                  # bitlen(1_000_000_000 - 1)
assert for_size(heavy_tailed) == 22
assert _uvarint_size(zigzag(1_000_000_000)) == 5                  # the outlier alone, as a varint
assert _varint_block_size(heavy_tailed) == 4 * 1 + 5 + 1           # four tiny values + the outlier + the count
assert _varint_block_size(heavy_tailed) == 10
assert best_encoding(heavy_tailed) == "varint"


def _brute_encode_for(values: list[int]) -> bytes:
    """Independent restatement of Part 3's layout, from the statement alone: the divmod-based varint
    encoder, the literal zigzag formula, and Part 1's 0/1-string bit packer -- never encode_for itself."""
    m = min(values)
    w = (max(values) - m).bit_length()
    header = _brute_uvarint(_brute_zigzag(m)) + bytes([w]) + _brute_uvarint(len(values))
    if w == 0:
        return header
    return header + _brute_pack([v - m for v in values], w)


def _brute_varint_block(values: list[int]) -> bytes:
    """The alternative scheme, built the same independent way: a count varint, then one zigzag
    varint per value."""
    return _brute_uvarint(len(values)) + b"".join(_brute_uvarint(_brute_zigzag(v)) for v in values)


def _random_block(rng: random.Random) -> list[int]:
    kind = rng.choice(["clustered", "heavy_tailed", "uniform", "equal"])
    n = rng.randint(1, 12)
    if kind == "clustered":
        base = rng.randrange(2 ** 40)
        return [base + rng.randint(0, 40) for _ in range(n)]
    if kind == "heavy_tailed":
        vals = [rng.randint(-5, 5) for _ in range(n)]
        vals[rng.randrange(n)] = rng.choice([1, -1]) * rng.randint(2 ** 20, 2 ** 30)
        return vals
    if kind == "uniform":
        return [rng.randint(-2 ** 20, 2 ** 20) for _ in range(n)]
    return [rng.randint(-1000, 1000)] * n           # every value equal -> w == 0


rng = random.Random(4)
for _ in range(800):
    values = _random_block(rng)
    expected = _brute_encode_for(values)
    got = encode_for(values)
    assert got == expected, values
    assert len(got) == for_size(values) == len(expected)
    assert decode_for(got) == values

    for_len = len(got)
    varint_len = len(_brute_varint_block(values))
    assert varint_len == _varint_block_size(values)
    expected_choice = "for" if for_len <= varint_len else "varint"
    assert best_encoding(values) == expected_choice, values

# edge cases: an empty block, and a block whose range needs more than 32 bits
assert _raises(ValueError, encode_for, [])
assert _raises(ValueError, for_size, [])
assert _raises(ValueError, best_encoding, [])

wide = [0, 2 ** 40]                                  # range needs 41 bits, more than pack's 32-bit limit
assert _raises(ValueError, encode_for, wide)
assert _raises(ValueError, for_size, wide)
assert _raises(ValueError, best_encoding, wide)

assert _raises(ValueError, decode_for, b"")                              # nothing to read at all
good = encode_for([998, 1000, 1005, 1000])
assert _raises(ValueError, decode_for, good + b"\x00")                   # trailing garbage after payload
equal_block = encode_for([7, 7, 7, 7])
assert _raises(ValueError, decode_for, equal_block + b"\x00")            # trailing garbage, w == 0 case

# the SIMD-friendly-layout follow-up: 128 values divide evenly into whole 32-bit and 64-bit words
assert 128 == 4 * 32 == 2 * 64

print("all checks passed")
```

</details>

</details>
