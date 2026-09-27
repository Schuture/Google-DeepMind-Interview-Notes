# 代码评审：给一次 checkpoint 改动中的缺陷排序

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 代码评审，再修复与测试 | ★★★☆☆ | 中等 | RE · SWE · MLE | code-review, checkpointing, atomic-writes, reproducibility, data-sharding, testing | 3 个部分 / 45–60 分钟 | 终面（Final） |
<!-- meta:end -->

## 题目

一个 pull request 给训练代码库加入了下面这个 `checkpointing.py` 模块，它的描述是：“可续训（resumable
training）：每 N 步保存一次，只保留最近 3 个，从最新的一个续训”。这个模块提供
`save_checkpoint(directory, step, state, metadata=...)`——把 `state`（一个不透明的、由调用方定义的
dict，实际中是模型和优化器的数组）写入 `directory` 下的文件 `ckpt_<step>.pkl`，并附带调用方提供的
`metadata` dict；`load_latest(directory)`——返回 `directory` 中最新 checkpoint 的 `state`，如果
`directory` 里一个都没有则返回 `None`；`apply_retention(directory, keep=3)`——删除 `directory` 中除最新
`keep` 个之外的所有 checkpoint；以及 `ShardedSampler(num_examples, rank, world_size, seed, epoch)`，
这个类的 `indices()` 方法返回数据并行（data-parallel）worker `rank`（`world_size` 个 worker 之一，编号从
`0` 到 `world_size - 1`）在 `epoch` 这个 epoch 里要处理的、`range(num_examples)` 中的下标列表，取自用
`seed` 和 `epoch` 算出的同一次 `range(num_examples)` 洗牌结果，再按 worker 切分。

```py
"""checkpointing.py -- resumable training: save every N steps, keep the last 3, resume from the latest.

Usage:
    save_checkpoint(run_dir, step, state)          # called every N training steps
    apply_retention(run_dir, keep=3)                # called right after, to bound disk usage
    state = load_latest(run_dir)                    # called once, at process start, to resume
"""
import glob
import os
import pickle
import random

CHECKPOINT_GLOB = "ckpt_*.pkl"
DEFAULT_KEEP = 3


def _checkpoint_path(directory: str, step: int) -> str:
    return os.path.join(directory, f"ckpt_{step}.pkl")


def _checkpoint_files(directory: str) -> list:
    return glob.glob(os.path.join(directory, CHECKPOINT_GLOB))


def save_checkpoint(directory: str, step: int, state: dict, metadata: dict = {}) -> None:
    """Writes `state` (model/optimizer arrays, kept opaque here) to ckpt_<step>.pkl in `directory`,
    tagged with `metadata` -- small caller-supplied bookkeeping, e.g. which steps have been saved
    so far, for debugging retention."""
    os.makedirs(directory, exist_ok=True)
    metadata["step"] = step
    metadata.setdefault("history", []).append(step)
    payload = {"state": state, "metadata": metadata}
    with open(_checkpoint_path(directory, step), "wb") as f:
        pickle.dump(payload, f)


def load_latest(directory: str):
    """Returns the state of the newest checkpoint in `directory`, or None if there is none."""
    files = sorted(_checkpoint_files(directory))
    if not files:
        return None
    try:
        with open(files[-1], "rb") as f:
            payload = pickle.load(f)
        return payload["state"]
    except Exception:
        return None


def apply_retention(directory: str, keep: int = DEFAULT_KEEP) -> None:
    """Deletes all but the newest `keep` checkpoints in `directory`."""
    files = sorted(_checkpoint_files(directory))
    for path in files[:-keep]:
        os.remove(path)


class ShardedSampler:
    """Shuffles range(num_examples) with `seed` and `epoch`, then splits it across `world_size`
    data-parallel workers. indices() returns the list of example indices worker `rank` processes
    this epoch."""

    def __init__(self, num_examples: int, rank: int, world_size: int, seed: int, epoch: int):
        self.num_examples = num_examples
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch

    def indices(self) -> list:
        rng = random.Random(self.seed + self.epoch)
        perm = list(range(self.num_examples))
        rng.shuffle(perm)
        per_worker = self.num_examples // self.world_size
        start = self.rank * per_worker
        return perm[start:start + per_worker]
```

假设同一时间只有一个进程会写同一个 checkpoint 目录：对同一个 `directory`，不会有两次 `save_checkpoint`
调用同时进行，也不会有一次 `apply_retention` 和一次 `save_checkpoint` 同时进行。`step` 是调用方指定的正
整数，在同一次训练中随调用严格递增。`apply_retention` 的 `keep` 是一个正整数，不超过目前为止保存过的
checkpoint 数。一次训练里的每个 worker 调用 `ShardedSampler` 时，`num_examples`、`world_size`、`seed`、
`epoch` 都相同，只有 `rank` 不同。举例：

```text
save_checkpoint(run_dir, step=1, state={"loss": 0.9})
save_checkpoint(run_dir, step=2, state={"loss": 0.7})
load_latest(run_dir)                # -> {"loss": 0.7}，即 step 2 保存的 state
apply_retention(run_dir, keep=1)    # -> run_dir 下只剩 ckpt_2.pkl
```

### Part 1 —— 评审

列出你在上面这个模块中找到的每一个缺陷，按严重程度排序。对每一个缺陷给出：它的位置（引用具体的代码行或
表达式）；具体会出什么问题、在什么情况下触发；以及它排在第几位。排序依据是*影响面*（blast
radius）——正确性或已完成的工作损失了多少，乘以在一个长期运行的训练任务生命周期里，触发条件出现的可能性
有多大——而不是修起来要改几行代码。先说明你排序所依据的原则，再据此排序。交付物是这份有序列表，不是补丁。

### Part 2 —— 修复

写出修正后的 `checkpointing.py`。至少要做到：让 `save_checkpoint` 具有原子性（atomic），使得进程在调用
过程中的任意时刻被杀死，`directory` 要么和调用前完全一样，要么新的 checkpoint 已经完整存在，绝不会停在
中间某个状态；让 `load_latest` 和 `apply_retention` 按 step 排序 checkpoint，而不是按文件名字符串；让
`load_latest` 不再把“现有的每一个 checkpoint 都读不出来”和“这里从来没保存过 checkpoint”混为一谈——响亮地
失败（loud failure）和一个有文档说明的回退（fallback）都可以接受，但不能悄无声息；去掉可变默认参数
（mutable default argument）；为 `state` 和 `metadata` 选择并论证一种序列化（serialisation）方式，说明它
对“谁能写入 `directory`”这件事做了什么信任假设；修复 `ShardedSampler`，使 `world_size` 个 worker 的
`indices()` 合起来，对同一个 epoch 恰好覆盖 `range(num_examples)` 一次，并明确说明剩下的
`num_examples % world_size` 个下标具体是怎么分配的。保留每一个函数名和类名；只有当上面某个缺陷确实要求
时，才可以改动某个签名。

### Part 3 —— 测试

写出一些测试，使它们在上面给出的原始模块上失败，在你 Part 2 的修复上通过。至少要包含：

- 模拟一个进程在 `save_checkpoint` 调用进行到一半时被杀死——做法是让底层的写入在已经有一部分字节到达
  操作系统之后再抛出异常，而不是手工写一个格式错误的文件——并确认在它之前保存的那个 checkpoint，之后
  `load_latest` 返回的仍然正是它。
- 保存一串 step 号，其中既有一位数、也有两位数，确认 `load_latest` 和 `apply_retention` 都把两位数的那个
  当作最新的。
- 对多组 `(num_examples, world_size)`，确认所有 worker 在一个 epoch 里 `indices()` 的并集恰好是
  `range(num_examples)`，既不缺也不重。
- 确认一次在 epoch 中途被打断、之后又续训的运行，产生的样本序列——以及这次运行消耗的其它任何逐步
  （per-step）随机性——和不曾被打断的运行完全一致。

所有操作都在一个 `tempfile.TemporaryDirectory()` 内部完成。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手评审之前，有两点值得先确认：“续训”（resume）到底要保证什么——是要和不曾被打断的那次运行逐位
（bit-for-bit）一致，还是只要求不丢失已完成的工作——因为这两种保证的实现代价相差很大；以及到底谁、或者
什么东西，有可能会往 checkpoint 目录里写文件，因为这决定了序列化格式的选择有多重要。

### Part 1

排序原则：影响面是正确性或已完成工作被破坏的程度，乘以在一次长期运行中触发条件出现的可能性。两个同样
容易触发的缺陷里，*悄无声息地*失败——给出一个看似合理、实则错误的结果，而不是报错——的那个，排名要高于
*响亮地*失败的那个，因为响亮的失败会被立刻发现并重跑，而一个悄无声息的错误结果可能在整个项目剩下的时间
里都没人发现。又因为大型训练任务经常被抢占（preempt），任何“进程恰好在不凑巧的时刻被杀死”就会触发的
缺陷，都应该被当作在一次长期运行里近乎必然发生，而不是小概率的边界情况。

1. **`load_latest` 里的 `except Exception: return None`。** 任何读取最新 checkpoint 文件失败的情况——
   被崩溃截断、磁盘上损坏，或者别的什么原因——都被当成和“这个目录从来没保存过 checkpoint”完全一样。训练
   悄悄地从初始化重新开始，之前的每一步进展全部作废，既不抛异常，也没有日志能把这次运行的输出和一次全新
   的运行区分开。更糟的是，它连*更早的*、完全有效的 checkpoint 也一并丢弃了：`load_latest` 从不尝试除
   最新那个文件以外的任何东西。影响面最高，因为它把任何来源的任何损坏，都变成了彻底的静默丢失。
2. **`save_checkpoint` 里对最终路径的直接写入。** `open(_checkpoint_path(directory, step), "wb")` 在
   `state` 的任何一个字节被写入之前，就已经截断或创建了目标文件；进程在调用过程中的任意位置被杀
   死——这也是一个 checkpoint 步骤唯一会被打断的方式——都会在 `load_latest` 接下来要找的那个路径上，
   留下一个不完整、解析不了的 `ckpt_<step>.pkl`。排在第二而不是第一，只是因为单看它自己、配上一个称职的
   `load_latest`，它本该表现为一次响亮的解析错误；它能排到前面，是因为它是触发上面第 1 条缺陷最常见的
   方式，而大型训练任务被抢占的频率之高，让“进程在保存过程中死掉”更接近每周都会发生的常态，而不是边界
   情况。
3. **`load_latest` 和 `apply_retention` 里对文件名的字符串排序。** `sorted(_checkpoint_files(directory))`
   逐字符比较 `"ckpt_10.pkl"` 和 `"ckpt_9.pkl"`，而 `"1" < "9"` 让第十步的 checkpoint 排在了第九步的
   *前面*。任何一次训练一旦保存了第十个 checkpoint，`load_latest` 就会从第 9 步开始续训——悄悄地把已经
   完成过的几步重新训一遍，输出上不会有任何看起来不对的地方，除非有人拿它和不曾续训的运行做对比。
   `apply_retention` 读到的是同一个错误的顺序：当磁盘上有第 1 到第 10 步、`keep=3` 时，它保留的是*按
   字典序*最大的三个名字（`ckpt_7`、`ckpt_8`、`ckpt_9`），删掉的是 `ckpt_10`——唯一真正新的那个
   checkpoint——就在第十个 checkpoint 出现的那一刻。排在第三位：和第 1、2 条缺陷不同，它完全不需要崩溃
   或损坏，只需要一次跑得足够久、能存到第十个 checkpoint 的普通训练，这让它比前两条更加确定无疑地发生，
   但它造成的破坏——一个过旧的续训点，几次过于激进的删除——比第 1、2 条彻底的静默重启还是要轻一档。
4. **要续训所需的 RNG 与 sampler 位置，都不在被保存的内容之内。** 这里没有任何地方——无论是这个模块本身，
   还是 PR 描述里提到的任何内容——把一个 worker 在当前 epoch 里已经消费到了哪个位置、或者其它任何运行期
   随机性的状态，作为 `state` 的一部分记录下来。一次续训必然要么把当前 epoch 从头开始，要么直接跳过它
   剩下的部分，还会从重新推出的某个种子里抽取全新的随机数，因此不会复现不曾被打断的那次运行本该产生的
   样本序列和逐步随机性。这是一次悄悄*续错*的训练，而不是丢失了训练：训练仍在推进，损失曲线看起来也依然
   合理，这正是它排在第 1–3 条之下（没有任何东西被破坏）、却又排在第 5–7 条之上的原因（它会破坏每一次
   续训，而不只是一小部分样本，也不是一个有条件才成立的攻击面）。
5. **`ShardedSampler` 漏掉了余数。** `per_worker = self.num_examples // self.world_size`，再加上
   `perm[start:start + per_worker]`，给每个 worker 分配的是相等的*向下取整*份额，从未发出去的是余下的
   `num_examples % world_size` 个下标——每一个 epoch 都有多达 `world_size - 1` 个样本，静静地从未被任何人
   训练过。排在第 1–4 条之下，因为损失的比例小而且有界（至多是 `num_examples` 中的 `world_size - 1`
   个），也不会损失任何已经完成的工作，只是少用了一些数据。
6. **`save_checkpoint` 里可变默认参数 `metadata={}`。** 每一次省略 `metadata` 的调用，都会共享并修改
   *同一个* dict 对象，于是本该只描述一个 checkpoint 的记录信息，就渗进了下一次调用里：
   `metadata.setdefault("history", []).append(step)` 的结果是，第 5 个 checkpoint 保存下来的 metadata 里
   `history` 是 `[1, 2, 3, 4, 5]`，而不是 `[5]`。这个问题局限在 `metadata` 这条旁路里——真正训练用的
   `state` 完好无损——所以它污染的是调试信息，不是模型本身。
7. **对 checkpoint 目录里的文件调用 `pickle.load`。** 反序列化（unpickling）不是“解析不可信的字节”：
   一个精心构造的文件，其 `__reduce__` 会在 `load_latest` 读取它的那一刻就执行任意代码，早于任何应用层
   逻辑看到结果的哪怕一个字节。它究竟是这里排名最高的缺陷，还是几乎不值一提，完全取决于前面那个还没确认
   的问题——一个只有这个训练进程会写入的目录，和一个共享基础设施、一次恢复的备份、或者另一个租户也可能
   写入文件的目录，风险完全不是一回事。这里之所以只是标记出来、而不是直接排到最前面，是因为在没有证据
   表明目录是共享的情况下，它的发生概率无从确立；不论答案是什么，Part 2 都会消除这个暴露面，因为这样做
   的代价很低。

还有两点很小的问题，值得顺手改一行，但不值得排进上面的名次：`apply_retention` 删除 checkpoint 时没有任何
日志，所以没有任何记录说明删掉了什么、什么时候删的，让“我的 checkpoint 去哪了”变成一场靠文件系统时间戳
猜测的游戏；`load_latest` 的文档字符串从未说明“latest”指的是最新的*文件*，还是最新的*有效*
checkpoint——这恰好正是上面第 1 条缺陷在代码里始终没有回答的问题。

### Part 2

每一个函数名、类名，以及每个调用点的形状都保持不变；唯一的签名改动是 `metadata` 的默认值，从可变的
`{}` 改成了 `None`，这是修复本身所要求的。`save_checkpoint` 现在先写入同一目录下的一个临时文件，
`fsync` 它，然后才用 `os.replace` 把它换到最终路径：`os.replace` 在 POSIX 文件系统上是原子的，所以
读者永远不会看到一个既不是旧 checkpoint、也不是完整新 checkpoint 的文件；在换入之前的任意一步失败，
留下的只是待清理的临时文件，最终路径则纹丝不动。checkpoint 文件名先被解析、再按整数 step 排序，绝不
按字符串排序。`load_latest` 从最新到最旧依次尝试 checkpoint，对每一个解析失败的都记一条警告日志再试
下一个，只有当*每一个*现存的 checkpoint 都失败之后，才会抛出异常——而不是返回 `None`——因为“这个目录
一个 checkpoint 都没有”和“这个目录唯一的 checkpoint 已经损坏”是调用方绝不能混淆的两种情形。

`state` 和 `metadata` 被拆成两部分：NumPy 数组用 `numpy.savez` 写入；其余的一切——普通的 Python
标量、字符串、list、小 dict，包括 RNG 状态和一个 sampler 位置——编码成 JSON，塞进同一个 `.npz`
归档里，作为又一个字节数组，因此整个 checkpoint 仍然只是一个文件，靠一次原子的改名完成写入。这里假设
`state` 和 `metadata` 里只有数组和 JSON 安全（JSON-safe）的值，这正是放弃 `pickle` 所要付出的代价：
读取时调用 `numpy.load` 会带上 `allow_pickle=False`，所以一个不是这段代码自己刚刚写出来的文
件——损坏的、被截断的，或者存心搞破坏的——都不可能借着被加载而执行任何东西，不论“谁能写入这个目录”
这个问题最终的答案是什么。

`ShardedSampler.indices()` 仍然返回一个 worker 在这个 epoch 里的完整份额，但现在按 rank 排在前面的
`num_examples % world_size` 个 worker，各多分到一个下标，于是 `world_size` 个 worker 的份额恰好
拼出 `range(num_examples)` 的一个划分（partition），大小之间最多相差一。因为这个排列只是
`(seed, epoch)` 的一个纯函数，调用方想从 epoch 中途续训时，并不需要 `ShardedSampler` 记住任何东西：
对被 checkpoint 下来的任意整数 `position`，`indices()[position:]` 就恰好是剩下的部分——这也是为什么
续训*对 sampler 本身而言*根本不需要额外的 RNG 状态；只有 `position`、`epoch`，以及训练循环另外用到的
任何其它随机性的状态（下面 Part 3 会用 Python 自己的 `random.getstate()`/`setstate()` 演示这一点），
才需要放进 `state` 里跟着走。

```python
import glob
import json
import logging
import os
import random
import re
import tempfile

import numpy as np

CHECKPOINT_GLOB = "ckpt_*.npz"
DEFAULT_KEEP = 3
_CKPT_RE = re.compile(r"^ckpt_(\d+)\.npz$")
_PAYLOAD_KEY = "__payload_json__"

logger = logging.getLogger(__name__)


def _checkpoint_path(directory: str, step: int) -> str:
    return os.path.join(directory, f"ckpt_{step}.npz")


def _step_of(path: str) -> int:
    m = _CKPT_RE.match(os.path.basename(path))
    if not m:
        raise ValueError(f"not a checkpoint file: {path}")
    return int(m.group(1))


def _checkpoint_files_by_step(directory: str) -> list:
    """Every checkpoint file in `directory`, oldest first -- sorted on the parsed integer step,
    not the filename string (ckpt_9.npz must sort before ckpt_10.npz)."""
    return sorted(glob.glob(os.path.join(directory, CHECKPOINT_GLOB)), key=_step_of)


def save_checkpoint(directory: str, step: int, state: dict, metadata: dict | None = None) -> None:
    """Writes `state` (a dict of numpy arrays and/or JSON-safe scalars) to ckpt_<step>.npz in
    `directory`, tagged with `metadata`. Atomic: a crash at any point leaves either the previous
    checkpoint or nothing at the final path, never a truncated ckpt_<step>.npz."""
    os.makedirs(directory, exist_ok=True)
    metadata = dict(metadata) if metadata is not None else {}   # NOTE: a fresh dict every call --
    metadata["step"] = step                                     # never mutate a shared default
    arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
    plain = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
    payload = json.dumps({"state": plain, "metadata": metadata}).encode("utf-8")
    final_path = _checkpoint_path(directory, step)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=f".tmp-ckpt_{step}-", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as f:
            # NOTE: allow_pickle stays off on load below, so only plain arrays belong here --
            # everything else (dicts, RNG state, strings) travels through the JSON side channel.
            np.savez(f, **arrays, **{_PAYLOAD_KEY: np.frombuffer(payload, dtype=np.uint8)})
            f.flush()
            os.fsync(f.fileno())          # NOTE: bytes are durably on disk before the rename below
        os.replace(tmp_path, final_path)  # NOTE: atomic on a POSIX filesystem -- a reader sees either
                                           #       the old file or the complete new one, never a partial write
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)           # NOTE: never leave a half-written temp file behind
        raise


def _load_one(path: str) -> dict:
    with np.load(path, allow_pickle=False) as npz:  # NOTE: allow_pickle=False -- a corrupt or hostile
        raw = bytes(npz[_PAYLOAD_KEY]).decode("utf-8")  # file cannot execute code by being loaded
        payload = json.loads(raw)
        arrays = {k: npz[k] for k in npz.files if k != _PAYLOAD_KEY}
    return {**arrays, **payload["state"]}


def load_latest(directory: str):
    """Returns the state of the newest readable checkpoint in `directory`, or None if there is
    none at all. A checkpoint that fails to load (partial write, disk corruption) is skipped with
    a logged warning and the next-newest one is tried, rather than silently treated the same as
    "no checkpoint"; if every checkpoint present fails to load, the last error is raised instead
    of returning None, since that is a disk problem the caller must not mistake for a fresh start."""
    files = _checkpoint_files_by_step(directory)
    last_error = None
    for path in reversed(files):
        try:
            return _load_one(path)
        except Exception as e:
            logger.warning("checkpoint %s failed to load (%s); falling back to an older one", path, e)
            last_error = e
    if files:
        raise RuntimeError(f"no checkpoint in {directory!r} could be read") from last_error
    return None


def apply_retention(directory: str, keep: int = DEFAULT_KEEP) -> None:
    """Deletes all but the newest `keep` checkpoints in `directory`."""
    for path in _checkpoint_files_by_step(directory)[:-keep]:
        os.remove(path)
        logger.info("removed old checkpoint %s", path)


class ShardedSampler:
    """Shuffles range(num_examples) with `seed` and `epoch`, then splits it across `world_size`
    data-parallel workers so every index is used exactly once. The first `num_examples %
    world_size` workers by rank get one extra example each, so split sizes differ by at most one.
    indices() returns worker `rank`'s full list for the epoch; to resume partway through an
    epoch, slice it -- indices()[position:] -- since the permutation is a pure function of
    (seed, epoch), so an integer position is all a caller needs to save to resume exactly."""

    def __init__(self, num_examples: int, rank: int, world_size: int, seed: int, epoch: int):
        self.num_examples = num_examples
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch

    def indices(self) -> list:
        rng = random.Random(self.seed + self.epoch)
        perm = list(range(self.num_examples))
        rng.shuffle(perm)
        base, remainder = divmod(self.num_examples, self.world_size)
        # NOTE: ranks [0, remainder) get one extra example each, so sizes sum to num_examples exactly
        start = self.rank * base + min(self.rank, remainder)
        size = base + (1 if self.rank < remainder else 0)
        return perm[start:start + size]
```

### Part 3

四个测试，对应上面的四条要求。第一个测试模拟的是一次真正的中断，而不是手工改出来的坏文件：它给
序列化调用打了个猴子补丁（monkeypatch），让它只生效一次——先往拿到的文件对象里写入几个真实的字节，
再抛出异常——这正是磁盘写满或者被 `SIGKILL` 时，从写入者的视角看到的样子：操作系统已经接受了一部分
字节，写入却还没完成。第二个测试就是 Part 1 里那个手算过的例子，只是换成对真正的函数调用一遍。第三、
第四个测试检验的是 Part 2 为 `ShardedSampler` 建立的两条性质：一个 epoch 被完整覆盖，以及一次续训精确
复现出续训所抽取的 `(index, noise)` 序列，其中 `noise` 代表训练循环另外消耗的任何逐步随机性——这里用
一个普通的 `random.Random` 来代表它，用 `random.getstate()`/`setstate()` 做 checkpoint，方式和 `state`
与 `position` 完全一样。

```python
import contextlib


def test_crash_mid_write_preserves_previous_checkpoint(save_checkpoint, load_latest, crashing_dump):
    """A crash while writing a NEW checkpoint must never destroy a previously valid one."""
    with tempfile.TemporaryDirectory() as d:
        save_checkpoint(d, 1, {"w": 1.0})
        with contextlib.suppress(OSError):
            with crashing_dump():
                save_checkpoint(d, 2, {"w": 2.0})
        state = load_latest(d)
        assert state is not None, "a crash while saving step 2 must not lose the valid step-1 checkpoint"
        assert state["w"] == 1.0, f"expected step 1's state, got {state}"


def test_numeric_step_ordering(save_checkpoint, load_latest, apply_retention):
    """Once step numbers reach two digits, the newest checkpoint is still resumed from and kept."""
    with tempfile.TemporaryDirectory() as d:
        for step in range(1, 11):
            save_checkpoint(d, step, {"w": float(step)})
        state = load_latest(d)
        assert state is not None and state["w"] == 10.0, f"load_latest must resume from step 10, got {state}"
        apply_retention(d, keep=3)
        kept = load_latest(d)
        assert kept is not None and kept["w"] == 10.0, "step 10 must survive retention"


def test_sharded_sampler_covers_every_example_once(sampler_cls):
    """The union of every worker's indices() for one epoch must be exactly range(num_examples)."""
    seed, epoch = 0, 0
    for num_examples in (17, 23, 40, 101):
        for world_size in (2, 3, 4, 7):
            union = []
            for rank in range(world_size):
                union.extend(sampler_cls(num_examples, rank, world_size, seed, epoch).indices())
            assert sorted(union) == list(range(num_examples)), (num_examples, world_size, len(union))


def _run_epochs(sampler_cls, num_examples, world_size, rank, seed, epochs, rng, start_epoch=0, start_position=0):
    """The sequence of (index, noise) pairs from start_epoch (skipping its first start_position
    indices) through epochs - 1, drawing one rng.random() noise value per index."""
    seq = []
    for epoch in range(start_epoch, epochs):
        idxs = sampler_cls(num_examples, rank, world_size, seed, epoch).indices()
        position = start_position if epoch == start_epoch else 0
        for idx in idxs[position:]:
            seq.append((idx, rng.random()))
    return seq


def test_resume_reproduces_uninterrupted_sequence(sampler_cls, save_checkpoint, load_latest, use_position):
    """A run interrupted mid-epoch and resumed must reproduce the uninterrupted sample sequence."""
    num_examples, world_size, rank, seed, noise_seed, epochs = 23, 4, 1, 0, 123, 2
    full = _run_epochs(sampler_cls, num_examples, world_size, rank, seed, epochs, random.Random(noise_seed))
    crash_after = len(full) // 2 + 2

    with tempfile.TemporaryDirectory() as d:
        rng = random.Random(noise_seed)
        seq, step, stop = [], 0, None
        for epoch in range(epochs):
            idxs = sampler_cls(num_examples, rank, world_size, seed, epoch).indices()
            for pos, idx in enumerate(idxs):
                seq.append((idx, rng.random()))
                step += 1
                if step == crash_after:
                    version, ints, gauss = rng.getstate()
                    state = {"epoch": epoch, "position": pos + 1, "rng_state": [version, list(ints), gauss]}
                    save_checkpoint(d, step, state)
                    stop = (epoch, pos + 1)
                    break
            if stop:
                break

        saved = load_latest(d)
        if use_position:
            version, ints, gauss = saved["rng_state"]
            rng2 = random.Random()
            rng2.setstate((version, tuple(ints), gauss))
            seq += _run_epochs(sampler_cls, num_examples, world_size, rank, seed, epochs, rng2,
                                start_epoch=saved["epoch"], start_position=saved["position"])
        else:
            # NOTE: nothing in the original module's contract says a caller must track position or
            # rng state, so the most natural resume from just "which epoch was I on" restarts that
            # epoch's noise from a fresh draw -- exactly the divergence the fix eliminates
            rng2 = random.Random(noise_seed)
            seq += _run_epochs(sampler_cls, num_examples, world_size, rank, seed, epochs, rng2,
                                start_epoch=saved["epoch"], start_position=0)

    assert seq == full, "resumed run must reproduce the uninterrupted sample sequence exactly"
```

### 追问

- **来自多台主机的分片模型状态。** 没有哪一台主机单独持有完整模型，所以每台主机只 checkpoint 自己的
  那一分片（shard），存进自己的文件名下；上面的原子写入模式，保护的是单个分片不被一次崩溃中的写入损坏，
  但这时候“一个有效的 checkpoint”意味着“同一个 step、每一台主机的分片，全部都在”——这不是任何单个文件
  自己能表达的性质。标准做法是另加一个标记文件（比如 `ckpt_<step>.commit`），它本身也原子地写入，而且
  只有在确认这个 step 的每一个分片都已落盘之后才会写；`load_latest`（这时要在整个任务的层面重新定义，
  而不是每台主机各自定义）只寻找带有 commit 标记的最新 step，从不直接看单个分片文件，这样一来，某些
  主机已经存完、有一台先崩溃的那个 step，就绝不会看起来是可以加载的。
- **异步 checkpoint。** 对一个大模型来说，保存 `state` 并 `fsync` 它，耗时可能比一个训练 step 本身还长，
  而上面的 `save_checkpoint` 是在调用方自己的线程里同步运行的，所以每一次 checkpoint 都要把这个耗时
  完整地付出一遍。标准做法是把数组从训练 step 自己的内存里拷贝出来（这一步拷贝是阻塞的，但比整个写入
  廉价得多），再把这份拷贝交给一个后台线程或进程，让训练在写入进行的同时继续往前走；这里的原子改名之所以
  能让这样做安全，正是因为 `load_latest` 能观察到新文件的唯一时刻，就是它已经完整写完之后，无论是哪个
  线程写的都一样。
- **校验和（checksum）验证。** `fsync` 加原子改名，排除的是*不完整*的写入抵达最终路径的可能，排除不了
  一个已经完整的文件，在磁盘故障或者不稳定的网络文件系统上悄悄发生的位级（bit-level）损坏。把数组字节的
  校验和存下来（放进同一条 JSON 旁路里），加载时再校验一遍，能把这类失效变成和 `load_latest` 已经用于
  “完全解析不出来的文件”那条一样的响亮失败或回退路径，而不是一个加载时不报错、内部数字却悄悄错了的
  checkpoint。
- **在时间紧张时评审。** 时间有限时，优先扫一遍数据丢失和悄悄给出错误答案的路径：任何写向一个可能已经
  持有有效数据的路径、又不明显是原子的写入；任何吞掉错误、又不区分“出了问题”和“本来就无事可做”的宽泛
  `except`；以及任何随机或并行的操作，它的覆盖范围或顺序只在文档字符串里断言过，代码里却没有能看见的
  保证。命名、日志和小的测试缺口值得指出，但晚一点修也花不了多少代价；一个悄悄的、看起来还算合理的错误
  答案，要是上线几个月后才被发现，代价要大得多，所以这才是一次限时评审应该最先找到的东西。

<details>
<summary>验证代码（可运行）</summary>

```python
# --- worked examples from the statement, run against the real functions ---
with tempfile.TemporaryDirectory() as d:
    save_checkpoint(d, 1, {"loss": 0.9})
    save_checkpoint(d, 2, {"loss": 0.7})
    assert load_latest(d) == {"loss": 0.7}
    apply_retention(d, keep=1)
    assert [os.path.basename(p) for p in _checkpoint_files_by_step(d)] == ["ckpt_2.npz"]
print("worked example OK")


# --- the original module, reconstructed under orig_* names exactly as given in the Problem section,
#     independent of the fix above except for the standard library it also uses ---
import pickle


def orig_checkpoint_path(directory, step):
    return os.path.join(directory, f"ckpt_{step}.pkl")


def orig_checkpoint_files(directory):
    return glob.glob(os.path.join(directory, "ckpt_*.pkl"))


def orig_save_checkpoint(directory, step, state, metadata={}):
    os.makedirs(directory, exist_ok=True)
    metadata["step"] = step
    metadata.setdefault("history", []).append(step)
    payload = {"state": state, "metadata": metadata}
    with open(orig_checkpoint_path(directory, step), "wb") as f:
        pickle.dump(payload, f)


def orig_load_latest(directory):
    files = sorted(orig_checkpoint_files(directory))
    if not files:
        return None
    try:
        with open(files[-1], "rb") as f:
            payload = pickle.load(f)
        return payload["state"]
    except Exception:
        return None


def orig_apply_retention(directory, keep=3):
    files = sorted(orig_checkpoint_files(directory))
    for path in files[:-keep]:
        os.remove(path)


class OriginalShardedSampler:
    def __init__(self, num_examples, rank, world_size, seed, epoch):
        self.num_examples = num_examples
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch

    def indices(self):
        rng = random.Random(self.seed + self.epoch)
        perm = list(range(self.num_examples))
        rng.shuffle(perm)
        per_worker = self.num_examples // self.world_size
        start = self.rank * per_worker
        return perm[start:start + per_worker]


def expect_assertion_failure(fn, *args, **kwargs):
    """Calls fn(*args, **kwargs) and confirms it raises AssertionError -- i.e. that the behaviour
    fn checks for is genuinely absent, not merely that fn is broken some other way."""
    try:
        fn(*args, **kwargs)
    except AssertionError:
        return
    raise AssertionError(f"expected {fn.__name__} to fail against this implementation, but it passed")


@contextlib.contextmanager
def crashing_pickle_dump(garbage=b"\x80\x04CRASHED-MID-WRITE-GARBAGE"):
    """While active, pickle.dump(obj, f) writes `garbage` straight to f, flushes it to the OS, then
    raises -- simulating a process killed partway through a checkpoint write, after some bytes are
    already on disk but before the write finished."""
    original = pickle.dump

    def crashing(obj, f, *a, **kw):
        f.write(garbage)
        f.flush()
        raise OSError("simulated crash mid-write")

    pickle.dump = crashing
    try:
        yield
    finally:
        pickle.dump = original


@contextlib.contextmanager
def crashing_np_savez(garbage=b"PK\x03\x04CRASHED-MID-WRITE-GARBAGE"):
    """The same idea for numpy.savez(f, ...): writes a few garbage bytes to f, flushes, then raises."""
    original = np.savez

    def crashing(f, *a, **kw):
        f.write(garbage)
        f.flush()
        raise OSError("simulated crash mid-write")

    np.savez = crashing
    try:
        yield
    finally:
        np.savez = original


# --- Part 3's four tests: each fails against the original module, and passes against the fix ---
expect_assertion_failure(test_crash_mid_write_preserves_previous_checkpoint,
                          orig_save_checkpoint, orig_load_latest, crashing_pickle_dump)
test_crash_mid_write_preserves_previous_checkpoint(save_checkpoint, load_latest, crashing_np_savez)

expect_assertion_failure(test_numeric_step_ordering, orig_save_checkpoint, orig_load_latest, orig_apply_retention)
test_numeric_step_ordering(save_checkpoint, load_latest, apply_retention)

expect_assertion_failure(test_sharded_sampler_covers_every_example_once, OriginalShardedSampler)
test_sharded_sampler_covers_every_example_once(ShardedSampler)

expect_assertion_failure(test_resume_reproduces_uninterrupted_sequence,
                          OriginalShardedSampler, orig_save_checkpoint, orig_load_latest, False)
test_resume_reproduces_uninterrupted_sequence(ShardedSampler, save_checkpoint, load_latest, True)
print("Part 3: all four tests fail against the original module and pass against the fix")


# --- defect 6 (mutable default): the leak on the original, its absence on the fix ---
# NOTE: earlier calls above already used orig_save_checkpoint, and its shared default is the very
# thing under test, so it is already polluted; reset it so this demonstration starts from the same
# clean slate a freshly started training process would.
orig_save_checkpoint.__defaults__ = ({},)
with tempfile.TemporaryDirectory() as d:
    for step in range(1, 6):
        orig_save_checkpoint(d, step, {"w": step})
    with open(orig_checkpoint_path(d, 1), "rb") as f:
        history_1 = pickle.load(f)["metadata"]["history"]
    with open(orig_checkpoint_path(d, 5), "rb") as f:
        history_5 = pickle.load(f)["metadata"]["history"]
assert history_1 == [1]
assert history_5 == [1, 2, 3, 4, 5], history_5  # every earlier call's step leaked into this one

with tempfile.TemporaryDirectory() as d:
    for step in range(1, 6):
        save_checkpoint(d, step, {"w": float(step)})
    assert load_latest(d) == {"w": 5.0}  # no shared state to leak in the first place
print("defect 6 (mutable default): confirmed the leak on the original, absent on the fix")


# --- defect 7 (pickle RCE): the original executes a crafted file's payload; the fix refuses to ---
_rce_marker = []


def _record(tag):        # a module-level function so pickle can reference it by name, like any real one
    _rce_marker.append(tag)
    return {"w": 0.0}     # orig_load_latest then returns this in place of the "real" state


class MaliciousState:
    """What an attacker-controlled "saved state" file can contain: __reduce__ makes unpickling it
    call an arbitrary function with arbitrary arguments, before any application code runs."""

    def __reduce__(self):
        return (_record, ("pwned",))


with tempfile.TemporaryDirectory() as d:
    with open(orig_checkpoint_path(d, 1), "wb") as f:
        pickle.dump({"state": MaliciousState(), "metadata": {}}, f)
    orig_load_latest(d)                                       # merely loading the file runs _record
assert _rce_marker == ["pwned"], "unpickling a crafted checkpoint should have executed the payload"

with tempfile.TemporaryDirectory() as d:
    path = _checkpoint_path(d, 1)
    np.savez(path, w=np.array([MaliciousState()], dtype=object))
    try:
        np.load(path, allow_pickle=False)["w"]
        raised = None
    except ValueError as e:
        raised = e
assert raised is not None, "allow_pickle=False must refuse an object array outright"
assert _rce_marker == ["pwned"]  # unchanged -- the fixed format never touched the payload
print(f"defect 7 (pickle RCE): the original executed the payload; the fix's allow_pickle=False raised "
      f"{type(raised).__name__} instead")

print("all checks passed")
```

</details>

</details>
