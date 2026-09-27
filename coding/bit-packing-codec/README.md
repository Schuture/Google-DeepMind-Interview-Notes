# Bit-Packing Encoders and Decoders

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · bit manipulation and size formulas | ★★★☆☆ | Medium | SWE · RE · Intern | bit-manipulation, varint, zigzag-encoding, frame-of-reference, serialisation | 3 parts / 45 min | Skills interview · Final interview |
<!-- meta:end -->

## Problem

Every part below implements one binary codec: a pair of functions that turn a sequence of Python integers
into a `bytes` object and back, so that decoding the bytes an encoder produced reproduces the original
sequence exactly. `bytes` is Python's immutable sequence of 8-bit values, each from `0` to `255`; within one
byte, its *most significant bit* (MSB) is the one worth $2^7 = 128$, and its *least significant bit* (LSB) is
the one worth $2^0 = 1$. For a non-negative integer $v$, `v.bit_length()` — written $\mathrm{bitlen}(v)$ below
— is the number of bits in its binary representation with no leading zero: $\mathrm{bitlen}(0) = 0$, and
$\mathrm{bitlen}(7) = 3$, since $7$ is `111` in binary.

### Part 1 — Fixed-width packing

```py
def pack(values: list[int], k: int) -> bytes: ...
def unpack(data: bytes, k: int, count: int) -> list[int]: ...
```

A field width $k$ satisfies $1 \le k \le 32$, and every value in `values` must satisfy $0 \le v < 2^k$; `pack`
raises `ValueError` if $k$ is outside that range, or if any value does not fit in $k$ bits. It writes each
value as a $k$-bit field, most significant bit first, with no gaps between fields: field $0$'s most
significant bit is the most significant bit of byte $0$, field $1$ begins in the very next bit position after
field $0$ ends — crossing into the next byte whenever $k$ does not divide that position evenly — and so on
through the last field. The $n = \mathrm{len(values)}$ fields together use exactly $kn$ bits; `pack` then pads
the least-significant positions of its final byte with zero bits up to the next whole byte, so it always
returns exactly $\lceil kn / 8 \rceil$ bytes ($0$ bytes when `values` is empty). `unpack` is the exact inverse:
given the same `k` and `count = len(values)`, it reads back `count` consecutive $k$-bit fields from `data`, in
the same order `pack` wrote them. It raises `ValueError` if `k` is out of range, and also if `len(data)` is
not exactly $\lceil k \cdot \mathrm{count} / 8 \rceil$ — the one length `pack` could have produced for that
`k` and `count` — since any other length could not be `pack`'s output.

```text
values = [5, 1, 7], k = 3
5 = 101, 1 = 001, 7 = 111   (each value as a 3-bit field, most significant bit first)

fields concatenated:  101 001 111             (9 bits)
padded to 2 bytes:    10100111 10000000       (7 zero bits appended at the end)
as bytes:              0xa7      0x80

pack([5, 1, 7], 3) == b"\xa7\x80"
unpack(b"\xa7\x80", 3, 3) == [5, 1, 7]
```

### Part 2 — Variable-length integers

Unsigned LEB128 (*varint*) encodes a non-negative integer $v$ as one or more bytes, its least significant
$7$-bit group first: while any nonzero bits of $v$ remain to be written, take its current low $7$ bits as one
byte's low $7$ bits, set that byte's *continuation bit* (bit $7$, mask `0x80`) to $1$ when more groups still
follow and to $0$ on the byte holding the last (most significant) group, then discard those $7$ bits by
shifting $v$ right; $v = 0$ still emits one byte, `0x00`, since the process always writes at least one group.

```py
def encode_uvarints(values: list[int]) -> bytes: ...
def decode_uvarints(data: bytes) -> list[int]: ...   # ValueError if the data ends inside a number
def zigzag(n: int) -> int: ...                       # 0, -1, 1, -2, 2, ... -> 0, 1, 2, 3, 4, ...
def unzigzag(z: int) -> int: ...
```

`encode_uvarints` concatenates the varint encoding of every value of `values` in order; every value must be
non-negative, or it raises `ValueError`. `decode_uvarints` is its inverse over the whole of `data`: it reads
varints back to back until no bytes remain, and raises `ValueError` if `data` ends in the middle of one — the
last byte read still had its continuation bit set, with no further byte to continue into. `zigzag` maps a
signed integer to a non-negative one so that small magnitudes, positive or negative, map to small results: it
sends $0, -1, 1, -2, 2, \ldots$ to $0, 1, 2, 3, 4, \ldots$, alternating between a non-negative and a negative
input as the result grows by one each step; `unzigzag` is its inverse, defined over every non-negative
integer.

```text
v = 300 = 0b1_0010_1100   (9 bits)

low 7 bits of 300: 0101100 = 44; 300 >> 7 = 2 (nonzero, more bytes follow) -> byte 0 = 0b1_0101100 = 0xac
low 7 bits of 2:   0000010 =  2;   2 >> 7 = 0 (nothing left, last byte)    -> byte 1 = 0b0_0000010 = 0x02

encode_uvarints([300]) == b"\xac\x02"
decode_uvarints(b"\xac\x02") == [300]

zigzag:    0  -1   1  -2   2  ...
        -> 0   1   2   3   4  ...
```

### Part 3 — Frame-of-reference blocks and the best encoding

A *block* is a non-empty list of signed integers to be encoded together; write $m = \min(\text{values})$ and
$M = \max(\text{values})$ for one. Its *frame-of-reference* encoding concatenates a header and a payload. The
header holds, in order: $m$ itself, as a zigzag varint (so it may be negative); the *bit width*
$w = \mathrm{bitlen}(M - m)$, as a single unsigned byte ($w = 0$ exactly when every value in the block equals
$m$); and `len(values)`, as a plain (non-zigzag) varint, since a count is never negative. The payload, present
only when $w > 0$, is `pack([v - m for v in values], w)` — every value re-expressed as its non-negative offset
from $m$, which by the definitions of $M$ and $w$ always fits in $w$ bits. `encode_for` raises `ValueError` on
an empty `values`, since $m$ is undefined for one, and also if $w$ would exceed $32$, the field-width limit
`pack` itself imposes.

```py
def encode_for(values: list[int]) -> bytes: ...
def decode_for(data: bytes) -> list[int]: ...
def for_size(values: list[int]) -> int: ...          # exact size in bytes, computed by formula, without encoding
def best_encoding(values: list[int]) -> str: ...     # "for" or "varint" (zigzag varints, one per value, plus a count varint); ties -> "for"
```

`decode_for` is the exact inverse of `encode_for`: it reads the header to recover $m$, $w$ and the count,
then, if $w > 0$, unpacks that many $w$-bit fields and adds $m$ back to each (if $w = 0$, every value is
simply $m$). It raises `ValueError` if `data` is too short to hold the header or the payload the header
describes, and also if bytes remain after everything `encode_for` would have written for that header —
including when $w = 0$, whose header is by definition followed by no payload at all, so no byte may come
after it. `for_size` returns the exact byte length `encode_for` would produce, from the header and payload
size formulas alone, without constructing any bytes. The alternative this page compares frame-of-reference
against is encoding the same block as plain zigzag varints, with no shared offset: `len(values)` as a varint,
then `zigzag(v)` as a varint for every value, back to back. `best_encoding` returns whichever of `"for"` or
`"varint"` produces the shorter byte string for `values` — preferring `"for"` on an exact tie — and, like
`encode_for`, raises `ValueError` on an empty `values`.

```text
values = [998, 1000, 1005, 1000]
m = min(values) = 998, max(values) = 1005, w = bitlen(1005 - 998) = bitlen(7) = 3
deltas = [v - 998 for v in values] = [0, 2, 7, 2]

header:  zigzag(998) = 1996, as a varint: 2 bytes
         w = 3, as one byte:              1 byte
         len(values) = 4, as a varint:    1 byte
         -> header = 2 + 1 + 1 = 4 bytes
payload: pack([0, 2, 7, 2], 3) -> ceil(3 * 4 / 8) = 2 bytes

for_size(values) == 4 + 2 == 6
decode_for(encode_for(values)) == [998, 1000, 1005, 1000]
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points are worth confirming before coding: the bit order for fixed-width fields — most significant bit
first here, though several real formats instead write least significant bit first (see the follow-ups) — and
how each scheme below knows where to stop. `unpack` is handed `count` from outside, since a run of packed
bits carries no marker of its own; the varint stream instead relies on each varint's own continuation bit and
simply reads until `data` runs out; and `decode_for` recovers its own count from the header `encode_for`
wrote, so it is never passed one separately.

### Part 1

`pack` keeps a bit accumulator `acc` together with `nbits`, the number of its low-order bits that are
currently meaningful. Appending a value left-shifts `acc` by $k$ and OR's the value into the vacated low bits
— writing $k$ more bits onto the right end of a growing bit string — after which, while $8$ or more bits are
pending, the top byte of those `nbits` bits (`acc >> nbits`, once `nbits` has already been reduced by $8$) is
peeled off and appended to the output; masking `acc` down to its remaining `nbits` bits after every flush
keeps it from growing without bound as more values are packed, since a byte already written never needs to be
looked at again. Once every value is processed, between $0$ and $7$ bits are still pending; left-shifting them
to the high end of one more byte, `acc << (8 - nbits)`, supplies exactly the zero padding the statement
requires, because an integer has no bits set above its own value.

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

`unpack` runs the same idea in reverse: it pulls in whole bytes until at least `k` bits are pending, reads the
top `k` of them off the accumulator, and masks the rest away exactly as `pack` does, so the two sides share
one invariant — `acc`'s low `nbits` bits are always exactly the bits not yet consumed here, or not yet flushed
on the other side.

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

Checking `data`'s length before reading anything matters because a short buffer would otherwise run `idx`
past the end of `data` partway through the last field, while a long one would silently succeed and ignore
extra trailing bytes that could not have come from `pack`. Every byte of the output is produced (in `pack`)
or consumed (in `unpack`) by exactly one iteration of the inner `while` loop, and every value contributes
exactly one pass of the outer loop, so both functions run in $O(kn / 8) = O(kn)$ time — linear in the size of
their own output, which is itself linear in $n$ for a fixed $k$. That output size follows directly from
counting bits: $n$ fields of $k$ bits use $kn$ bits, and padding up to a whole byte adds at most $7$ more,
giving exactly $\lceil kn/8 \rceil$ bytes — `_packed_size`, which `unpack` uses to validate `data` before
trusting it, and which Part 3 reuses to size a frame-of-reference block's payload without packing it.

### Part 2

`_encode_uvarint` is LEB128's own definition, transcribed directly: `v & 0x7F` reads off the low $7$ bits,
`v >>= 7` discards them, and the loop stops the first time nothing nonzero remains — which happens on the very
first iteration when $v = 0$, since the loop body always runs at least once, giving the one required byte
`0x00`. `_uvarint_size` computes the resulting byte count from a formula instead: the number of groups the
loop above writes for $v > 0$ is the smallest $g$ with $v \gg 7g = 0$, i.e. $7g \ge \mathrm{bitlen}(v)$, i.e.
$g = \lceil \mathrm{bitlen}(v)/7 \rceil$; $v = 0$ needs the one group that formula alone would not predict,
since $\mathrm{bitlen}(0) = 0$ gives $\lceil 0/7 \rceil = 0$ — hence the `max(1, ...)`.
(`-(-v.bit_length() // 7)` computes that ceiling using only floor division, by the standard identity
$\lceil a/b \rceil = -\lfloor -a/b \rfloor$.)

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

Decoding accumulates in the opposite direction, least significant group first: `_read_uvarint` OR's each
byte's $7$ payload bits in at an ever-growing `shift`, and stops as soon as it sees a byte whose continuation
bit is clear. Running out of `data` while a continuation bit is still pending — checked before that byte is
read, not after — is exactly `decode_uvarints`'s definition of data that ends inside a number.

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

Both directions cost $O(\lceil \mathrm{bitlen}(v)/7 \rceil)$ per value — one loop iteration per byte produced
or consumed — so `encode_uvarints` and `decode_uvarints` each run in time linear in the total number of bytes
the size formula predicts for all of `values`, never more.

Encoding a negative Python integer with `_encode_uvarint` directly makes no sense, since shifting a negative
number right never reaches $0$; the usual fallback, reinterpreting it as a fixed-width unsigned value, is
exactly what makes small negatives expensive. As a 64-bit two's-complement value, $-1$ is $2^{64}-1$, whose
$\mathrm{bitlen}$ is $64$, so it would cost $\lceil 64/7 \rceil = 10$ varint bytes for a value of magnitude
$1$. `zigzag` avoids that by mapping small negatives to small non-negative codes instead of huge ones —
$\mathrm{zigzag}(-1) = 1$, a single byte — alternating between a non-negative and a negative input as its
output grows by one each step, exactly the $0, -1, 1, -2, 2, \ldots \to 0, 1, 2, 3, 4, \ldots$ correspondence
stated above. `unzigzag` inverts the two cases directly: an even code $z = 2n$ came from a non-negative
$n = z / 2$, and an odd code $z = -2n - 1$ (so $z + 1 = -2n$) came from a negative $n = -(z+1)/2$.

The page's branch, `2 * n if n >= 0 else -2 * n - 1`, agrees with the bitwise form real varint libraries use
on fixed-width registers, $\mathrm{zigzag}(n) = \big((n \ll 1) \oplus (n \gg 63)\big) \bmod 2^{64}$ for every
64-bit two's-complement $n$: $n \gg 63$, an arithmetic (sign-extending) shift, is $0$ for $n \ge 0$ and,
reduced mod $2^{64}$, all-ones for $n < 0$. XOR-ing with $0$ is a no-op, so the non-negative case gives
$n \ll 1 = 2n$ directly; XOR-ing with all-ones is a bitwise complement, so the negative case gives
$\overline{2n} = -2n - 1$ (Python's own `~x` is defined as $-x - 1$ for every integer $x$) — the same two
branches `zigzag` itself takes.

### Part 3

`_for_header_width` factors out the one computation `encode_for` and `for_size` must agree on: $m$, and the
bit width $w$ that makes every offset $v - m$ fit `pack`'s contract. Since $v - m$ ranges from $0$ (at the
value equal to $m$) up to $M - m$ (at the value equal to $M$), $w = \mathrm{bitlen}(M - m)$ is by definition
the smallest field width that top value fits in. It also raises the one `ValueError` both functions must
raise: on an empty block, where $m$ has no meaning, and when $w$ would exceed $32$ — a block spanning more
than `pack` can represent, which this particular payload encoding simply cannot handle.

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

`decode_for` reads the same three header fields in the same order and, once it knows $w$, either returns $m$
repeated `count` times (checking that nothing follows the header, since a well-formed all-equal block has no
payload at all) or unpacks `count` fields of $w$ bits each and adds $m$ back onto every one, undoing exactly
the subtraction `encode_for` performed. `for_size` mirrors `encode_for`'s structure field for field, with
`_uvarint_size` and `_packed_size` in place of the encoders themselves, so it never builds a single byte;
`for_size(values) == len(encode_for(values))` for the same reason `_packed_size` already gave `pack` in
Part 1, applied now to the header's two varints as well as the payload. `_for_header_width` costs $O(n)$ on
its own, from the `min`/`max` scan; the payload costs $O(wn)$ by Part 1's bound, which is $O(n)$ since
$w \le 32$ is bounded regardless of $n$. So `encode_for` and `decode_for` both cost $O(n)$ overall for a
block of $n$ values — the same linear bound `pack` and `unpack` already have — with the header's own two
varints adding only a few more bytes of work on top.

```python
def _varint_block_size(values: list[int]) -> int:
    """Bytes the alternative scheme uses: a count varint, then one zigzag varint per value."""
    return _uvarint_size(len(values)) + sum(_uvarint_size(zigzag(v)) for v in values)


def best_encoding(values: list[int]) -> str:
    # NOTE: for_size raises on an empty block, so best_encoding does too -- no separate check needed
    return "for" if for_size(values) <= _varint_block_size(values) else "varint"
```

Frame-of-reference pays for the block's offset once, in $m$, then a fixed $w$ bits per value regardless of
that value's own size; plain varints pay nothing for a shared offset but charge every value for its own
magnitude. That trade-off decides which wins. Values clustered in a narrow range around a large common
offset — $[100000, 100001, 100000, 100003, 100002]$, where $w = \mathrm{bitlen}(3) = 2$ once $m = 100000$ is
factored out — cost `for_size` only $7$ bytes, against $16$ bytes for five independent varints (each value's
own magnitude, around $10^5$, needs $\lceil 18/7 \rceil = 3$ bytes whether or not its neighbours are nearby,
plus one byte for the count); `best_encoding` picks `"for"`. A heavy-tailed block,
$[1, 2, 1, 3, 1000000000]$, is the opposite: the one outlier forces $w = \mathrm{bitlen}(999999999) = 30$ bits
for every value, including the four tiny ones, so `for_size` costs $22$ bytes, while varints let each small
value pay a single byte for its own tiny magnitude and only the outlier pay for its own ($5$ bytes, for $31$
bits), $4 + 5 + 1 = 10$ bytes total; `best_encoding` picks `"varint"`.

### Follow-ups

- **Delta encoding for sorted data.** Nearly monotonic data — timestamps, cumulative counters, sorted IDs —
  is a poor fit for frame-of-reference as given here, since one bit width must cover the gap between the
  smallest and largest element even when consecutive elements sit close together; encoding the first value
  plus each successive difference (`values[i] - values[i-1]`) instead bounds every difference by how far
  *consecutive* elements can drift, normally far smaller than the block's whole range. Data that is only
  nearly sorted can still produce an occasional negative difference, so those differences go through `zigzag`
  before becoming varints, exactly like any other signed values; a sequence guaranteed strictly increasing
  could use `encode_uvarints` on the plain differences instead, since they can then never be negative.
- **SIMD-friendly layouts.** `pack` writes one unbounded, most-significant-bit-first stream at one width $k$
  for the whole call. Real bit-packed formats (Parquet's bit-packing, Lucene's `PackedInts`, FastPFOR) instead
  split values into fixed blocks — commonly $128$ — each with its own width $w$, and pack each block least
  significant bit first. $128$ is not an arbitrary round number: for every integer $w$ from $1$ to $32$,
  $128w$ is a multiple of both $32$ and $64$ (since $128 = 4 \times 32 = 2 \times 64$), so a block of $128$
  values always occupies a whole number of $32$-bit or $64$-bit machine words, with no value's bits ever split
  across a block boundary — letting a vectorised unpacker load a block as whole words and apply the same
  small, precomputed set of per-lane shift-and-mask constants to every lane of a SIMD register at once, rather
  than walking the stream sequentially the way `pack`'s accumulator does.
- **Patching outliers (PFOR).** A single outlier forces $w$ up for an entire frame-of-reference block, exactly
  as in the heavy-tailed example above; patched frame-of-reference instead picks $w$ to cover most values —
  often a fixed percentile of the block — packs every value at that narrower width regardless of whether it
  actually fits (an outlier's high bits are simply discarded), and separately stores a list of
  `(index, true value)` *patches* for the values that width was too narrow for, applied after unpacking to
  overwrite the truncated results. The cost is a small amount of exception bookkeeping — an index and a value
  varint per outlier — against the width every non-outlier value would otherwise pay for.
- **Error detection.** None of the formats above detect corruption: a single bit flipped inside a `pack`ed
  stream silently shifts every field after it, and a flipped continuation bit inside a varint desynchronises
  every value that follows. Appending a checksum of the block's bytes — even a cheap running sum, or more
  robustly a standard one such as CRC-32 — after the payload, and having the decoder recompute and compare it
  before trusting the rest, turns that silent corruption into an immediate `ValueError` at decode time, at
  the cost of a handful of fixed extra bytes on every block.

<details>
<summary>Checks (runnable)</summary>

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
