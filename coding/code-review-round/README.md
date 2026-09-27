# Code Review: Ranking the Defects in a Checkpointing Change

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Coding · code review, then fix and test | ★★★☆☆ | Medium | RE · SWE · MLE | code-review, checkpointing, atomic-writes, reproducibility, data-sharding, testing | 3 parts / 45–60 min | Final interview |
<!-- meta:end -->

## Problem

A pull request adds the module `checkpointing.py` below to a training codebase. Its description: "resumable
training: save every N steps, keep the last 3, resume from the latest." The module provides
`save_checkpoint(directory, step, state, metadata=...)`, which writes `state` — an opaque, caller-defined
dict, model and optimizer arrays in practice — to a file `ckpt_<step>.pkl` under `directory`, tagged with a
caller-supplied `metadata` dict; `load_latest(directory)`, which returns the `state` of the newest checkpoint
in `directory`, or `None` if `directory` holds none; `apply_retention(directory, keep=3)`, which deletes
every checkpoint in `directory` except the newest `keep`; and `ShardedSampler(num_examples, rank, world_size,
seed, epoch)`, a class whose `indices()` method returns the list of indices into `range(num_examples)` that
data-parallel worker `rank` — one of `world_size` workers, numbered `0` to `world_size - 1` — processes
during epoch `epoch`, drawn from one shuffle of `range(num_examples)` computed from `seed` and `epoch` and
then split across workers.

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

A single process writes to a given checkpoint directory at a time: no two calls to `save_checkpoint`, and no
call to `apply_retention` overlapping one to `save_checkpoint`, ever run concurrently on the same
`directory`. Steps are positive integers assigned by the caller, strictly increasing across calls for a
given run. `apply_retention`'s `keep` is a positive integer no larger than the number of checkpoints ever
saved. Every worker in a run calls `ShardedSampler` with the same `num_examples`, `world_size`, `seed` and
`epoch`, differing only in `rank`. For example:

```text
save_checkpoint(run_dir, step=1, state={"loss": 0.9})
save_checkpoint(run_dir, step=2, state={"loss": 0.7})
load_latest(run_dir)                # -> {"loss": 0.7}, the state saved at step 2
apply_retention(run_dir, keep=1)    # -> only ckpt_2.pkl remains under run_dir
```

### Part 1 — Review

List every defect you find in the module above, ordered by severity. For each: its location (quote the
exact line or expression); what goes wrong, concretely, and under what circumstance it is triggered; and
where it ranks. Rank by *blast radius* — how much correctness or completed work is lost, multiplied by how
likely the trigger is to occur over the life of a long-running training job — not by how many lines the fix
takes. State the principle behind your ranking before applying it. The deliverable is the ordered list, not
a patch.

### Part 2 — Fix

Write a corrected `checkpointing.py`. At minimum, it must: make `save_checkpoint` atomic, so that a process
killed at any point during a call leaves `directory` exactly as it was before the call, or with the new
checkpoint fully present, never anything in between; make `load_latest` and `apply_retention` order
checkpoints by step, not by filename text; make `load_latest` never equate "every checkpoint present is
unreadable" with "no checkpoint has ever been saved" — a loud failure and a documented fallback are both
acceptable, but silence is not; remove the mutable default argument; choose and justify a serialisation for
`state` and `metadata`, stating what trust assumption it makes about who can write to `directory`; and fix
`ShardedSampler` so that the `world_size` workers' `indices()` together cover `range(num_examples)` exactly
once each for a shared epoch, stating precisely how the `num_examples % world_size` leftover indices are
distributed. Keep every function and class name; a signature may only change where a defect above forces it
to.

### Part 3 — Tests

Write tests that fail against the module as given above and pass against your fix from Part 2. At minimum:

- Simulate a process killed partway through a `save_checkpoint` call — by making the underlying write raise
  after some of its bytes have already reached the OS, not by hand-writing a malformed file — and confirm
  the checkpoint saved just before it is still exactly what `load_latest` returns afterwards.
- Save checkpoints for a run of steps that passes through both a one-digit and a two-digit step number, and
  confirm both `load_latest` and `apply_retention` treat the two-digit one as the newest.
- For several `(num_examples, world_size)` pairs, confirm that the union of every worker's `indices()` for
  one epoch is exactly `range(num_examples)`, with nothing missing and nothing duplicated.
- Confirm that a run interrupted partway through an epoch and then resumed reproduces exactly the sequence
  of examples — and any other per-step randomness the run consumes — that an uninterrupted run would have
  produced.

Run everything inside a `tempfile.TemporaryDirectory()`.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two points are worth confirming before reviewing: what "resume" must guarantee — bit-for-bit reproducibility
of the run that would have happened without the interruption, or merely that no already-completed work is
lost — since the two guarantees cost very different amounts to provide; and who, or what, can ever write a
file into the checkpoint directory, since that decides how much the choice of serialisation format matters.

### Part 1

Ranking principle: blast radius is how much correctness or completed work is destroyed, multiplied by how
likely the trigger is over the life of a long run. Between two defects that are equally likely, the one that
fails silently — producing a plausible, wrong result instead of an error — ranks above the one that fails
loudly, because a loud failure is noticed at once and re-run, while a silent wrong result can stand unnoticed
for the rest of a project. And because large training jobs are preempted routinely, any defect whose trigger
is "the process was killed at an inconvenient moment" is treated as near-certain to fire over a long run, not
as a rare edge case.

1. **`except Exception: return None` in `load_latest`.** Any failure to read the newest checkpoint file —
   truncated by a crash mid-write, corrupted on disk, or anything else — is treated exactly like "this
   directory has never had a checkpoint saved to it." Training silently restarts from initialisation, every
   prior step of progress discarded, with no exception and no log line to distinguish this run's output from
   a fresh one's. Worse, it discards every *older*, perfectly valid checkpoint too: `load_latest` never tries
   anything but the single newest file. Highest blast radius, because it turns any corruption, from any
   source, into total silent loss.
2. **Direct write to the final path in `save_checkpoint`.** `open(_checkpoint_path(directory, step), "wb")`
   truncates or creates the target file before a single byte of `state` has been written; a process killed
   anywhere during the call — the only way a checkpoint step is ever interrupted — leaves a partial,
   unparseable `ckpt_<step>.pkl` sitting exactly where `load_latest` will look next. Ranked second, not
   first, only because on its own, paired with a competent `load_latest`, it would announce itself as a loud
   parse error; it earns its place near the top because it is the single most common way to trigger defect 1
   above, and large training jobs are preempted routinely enough that "the process dies mid-save" is closer
   to a weekly occurrence than an edge case.
3. **String sort of filenames in `load_latest` and `apply_retention`.** `sorted(_checkpoint_files(directory))`
   compares `"ckpt_10.pkl"` and `"ckpt_9.pkl"` character by character, and `"1" < "9"` puts the ten-step
   checkpoint *before* the nine-step one. The moment any run saves a tenth checkpoint, `load_latest` starts
   resuming from step 9 — silently re-training steps already completed, with nothing wrong-looking in the
   output unless someone compares it against the un-resumed run. `apply_retention` reads the same wrong
   order: with `keep=3` and steps `1..10` on disk, it keeps the three *lexicographically* largest names
   (`ckpt_7`, `ckpt_8`, `ckpt_9`) and deletes `ckpt_10` — the only genuinely new checkpoint — the moment a
   tenth one exists. Ranked third: unlike defects 1–2, it needs no crash or corruption at all, only an
   ordinary run long enough to save ten checkpoints, which makes it certain rather than merely likely, but
   its damage — an old resume point, some over-eager deletions — is a notch short of defects 1–2's complete
   silent restart.
4. **RNG and sampler position absent from what gets checkpointed.** Nothing here — not this module, and
   nothing the PR description mentions — captures which examples within the current epoch a worker has
   already consumed, or the state of any other run-time randomness, as part of `state`. A resumed run
   necessarily starts its current epoch over (or skips its remainder outright) and draws fresh randomness
   from whatever it re-derives, so it does not reproduce the sequence of examples and per-step randomness the
   uninterrupted run would have produced. A silent *wrong* resume, not a lost one: training keeps moving and
   the loss curve still looks plausible, which is exactly why it ranks below defects 1–3 (nothing is
   destroyed) yet above defects 5–7 (it corrupts every single resumed run, not a bounded fraction of examples
   or a conditional attack surface).
5. **`ShardedSampler` drops the remainder.** `per_worker = self.num_examples // self.world_size` followed by
   `perm[start:start + per_worker]` gives every worker an equal *floor* share and never hands out the
   `num_examples % world_size` indices left over — up to `world_size - 1` examples every single epoch,
   silently never trained on by anybody. Ranked below defects 1–4 because the fraction lost is small and
   bounded (at most `world_size - 1` out of `num_examples`) and costs no already-completed work, only some
   under-used data.
6. **Mutable default `metadata={}` in `save_checkpoint`.** Every call that omits `metadata` shares and
   mutates the *same* dict object, so bookkeeping meant to describe one checkpoint bleeds into the next:
   `metadata.setdefault("history", []).append(step)` means checkpoint 5's saved metadata lists `[1, 2, 3, 4,
   5]`, not `[5]`. Confined to the `metadata` side-channel — the trained `state` itself is untouched — so it
   corrupts debugging information, not the model.
7. **`pickle.load` on files in the checkpoint directory.** Unpickling is not "parse untrusted bytes": a
   crafted file's `__reduce__` runs arbitrary code the moment `load_latest` reads it, before any application
   logic sees a single byte of the result. Whether this is the top-ranked defect here or barely a concern at
   all turns entirely on the still-unconfirmed question above — a directory only this training process ever
   writes to is a very different risk from one that shared infrastructure, a restored backup, or another
   tenant could also place a file into. Flagged rather than ranked to the top because, absent evidence the
   directory is shared, its likelihood is unestablished; Part 2 removes the exposure regardless, since the
   cost of doing so is low.

Two minor points are worth a one-line fix but not worth ranking among the above: `apply_retention` deletes
checkpoints with no log line, so nothing records what was removed or when, turning "where did my checkpoint
go" into a filesystem-timestamp guessing game; and `load_latest`'s docstring never says whether "latest"
means the newest *file* or the newest *valid* checkpoint — precisely the question defect 1 above leaves
unanswered in code.

### Part 2

Every function and class name, and the shape of every call site, stays the same; the only signature change
is `metadata`'s default, from the mutable `{}` to `None`, which the fix requires. `save_checkpoint` now
writes to a temporary file in the same directory, `fsync`s it, and only then `os.replace`s it onto the final
path: `os.replace` is atomic on a POSIX filesystem, so a reader never observes a file that is neither the old
checkpoint nor the complete new one, and a failure at any point before the replace leaves the temporary file
to clean up and the final path untouched. Checkpoint filenames are parsed and sorted on the integer step,
never the string. `load_latest` walks checkpoints from newest to oldest, logging a warning and trying the
next one for each that fails to parse, and raises — rather than returning `None` — only once every single
checkpoint present has failed, since a directory with zero checkpoints and a directory whose only checkpoint
is corrupt are different situations a caller must not confuse.

`state` and `metadata` are split into NumPy arrays, written with `numpy.savez`, and everything else — plain
Python scalars, strings, lists, small dicts, including RNG state and a sampler position — encoded as JSON and
folded into that same `.npz` archive as one more array of bytes, so the whole checkpoint is still exactly one
file, renamed atomically. This assumes `state` and `metadata` hold only arrays and JSON-safe values, which is
the trade-off for dropping `pickle`: on load, `numpy.load` is called with `allow_pickle=False`, so a file
this code did not itself just write — corrupt, truncated, or actively hostile — cannot make it execute
anything, regardless of what turns out to be true about who can write into the directory.

`ShardedSampler.indices()` still returns one worker's full share for the epoch, but the first `num_examples %
world_size` workers by rank now get one extra index each, so the `world_size` workers' shares partition
`range(num_examples)` exactly, with sizes differing by at most one. Because the permutation is a pure
function of `(seed, epoch)` alone, a caller that wants to resume partway through an epoch does not need
`ShardedSampler` to remember anything: `indices()[position:]`, for whatever integer `position` was
checkpointed, is exactly the remainder — which is also why resuming needs no RNG state *for the sampler
itself*; only `position`, `epoch`, and the state of any other randomness the training loop separately
consumes (drawn out below, in Part 3, via Python's own `random.getstate()`/`setstate()`) need to travel in
`state`.

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

Four tests, one per requirement above. The first simulates a real interruption rather than a hand-edited
file: it monkey-patches the serialisation call so that, once, it writes a few genuine bytes to the file
object it is given and then raises — exactly what an `OSError` from a full disk, or a `SIGKILL`, looks like
from the writer's point of view, after the OS has already accepted some bytes but before the write is
complete. The second is the worked example from Part 1, run against the real functions instead of by hand.
The third and fourth check the two `ShardedSampler` properties Part 2 establishes: full coverage of one
epoch, and an exact reproduction of the sequence of `(index, noise)` pairs a resumed run draws, where `noise`
stands in for any other per-step randomness a training loop separately consumes — drawn here from a plain
`random.Random`, checkpointed via `random.getstate()`/`setstate()` exactly as `state` and `position` are.

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

### Follow-ups

- **Sharded model state across many hosts.** No single host holds the full model, so each host checkpoints
  only its own shard, under its own filename; the atomic-write pattern above keeps any one shard safe from a
  crash mid-write, but a valid checkpoint now means "every host's shard, from the same step, all present" —
  a property no single file states. The standard fix is a separate marker file (`ckpt_<step>.commit`), itself
  written atomically and only after every shard for that step is confirmed on disk; `load_latest`, redefined
  at the job level, looks for the newest step with a commit marker, never at per-shard files directly, so a
  step where some hosts finished and one crashed first never looks loadable.
- **Asynchronous checkpointing.** Saving `state` and `fsync`ing it can take longer than a training step for a
  large model, and `save_checkpoint` above runs synchronously in the caller's own thread, so every
  checkpointing step pays that cost in full. The standard fix copies the arrays out of the training step's
  own memory (a blocking copy, but a far cheaper one than the write) and hands the copy to a background
  thread or process, letting training continue while the write happens off the critical path; the atomic
  rename here is exactly what makes that safe, since the only moment `load_latest` can observe the new file
  is after it is completely written, regardless of which thread produced it.
- **Checksum verification.** `fsync` and an atomic rename rule out a *partial* write reaching the final path,
  but not silent bit-level corruption of an already-complete file, from a failing disk or a flaky network
  filesystem. Storing a checksum of the array bytes alongside them (in the same JSON side-channel) and
  verifying it on load turns that failure mode into the same loud-or-fallback path `load_latest` already
  takes for a file it cannot parse at all, rather than a checkpoint that loads without error yet holds
  silently wrong numbers.
- **Reviewing under time pressure.** Skim for data-loss and silent-wrong-answer paths first: every write to a
  path that might already hold valid data and is not obviously atomic; every broad `except` that swallows an
  error without distinguishing it from "there was nothing to do"; and every random or parallel operation whose
  coverage or ordering is only asserted in a docstring, never visibly enforced in the code. Naming, logging
  and small test gaps are worth flagging but cost little to fix later; a silent, plausible-looking wrong
  answer found only after months in production costs far more, so it is what a time-boxed review should find
  first.

<details>
<summary>Checks (runnable)</summary>

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
