"""CPU tests of the CUDA spill tier's core (``tensorfold.cuda.spill``, patch 0088): file encoding for every dtype, the
CRC-32 check, the size cap and the free-disk floor, writes and read-back across a restart, the configuration, and the
index two ranks agree at start. CUDA is hidden (``torch.cuda.is_available`` is False), so the writer and reader threads
copy through host memory; the staging buffers are shrunk so small tensors cross several buffers and CRC blocks."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import spill  # noqa: E402
from tensorfold.cuda.spill import SpillConfig, SpillStore, ids_key, key_name  # noqa: E402

pytestmark = pytest.mark.torch


@dataclass
class Snap:
    ids: list
    tensors: dict
    arr: np.ndarray
    pair: tuple
    nested: dict
    label: str
    n: int
    scale: float
    flag: bool
    nothing: None = None
    extra: int = 7                 # skipped at save: decode gives the default back
    later: list = field(default_factory=lambda: [1, 2])

    transient = ("cache_view",)


CLASSES = [f"{Snap.__module__}:{Snap.__qualname__}"]


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(spill, "CHUNK", 256 << 10)
    monkeypatch.setattr(spill, "BLOCK", 64 << 10)
    monkeypatch.setattr(spill, "_PINNED", {})


def _cfg(root, **kw) -> SpillConfig:
    base = dict(root=str(root), gib=1.0, min_tokens=4, min_free_gib=0.0, writers=1, direct=False)
    base.update(kw)
    return SpillConfig(**base)


def _store(root, signature: str = "sig", **kw) -> SpillStore:
    return SpillStore(_cfg(root, **kw), device="cpu", signature=signature, weights="w", classes=CLASSES, quiet=True)


def _ids(start: int, n: int) -> list[int]:
    return list(range(start, start + n))


def _all_dtypes() -> dict:
    g = torch.Generator().manual_seed(0)
    out = {}
    for dt, code in spill._DT.items():
        if dt is torch.bool:
            out[code] = torch.randint(0, 2, (3, 5), generator=g).bool()
            continue
        size = torch.empty(0, dtype=dt).element_size()
        raw = torch.randint(0, 256, (3, 5 * size), dtype=torch.uint8, generator=g)
        out[code] = raw.view(dt)                     # any bit pattern (NaNs too): compared as bytes
    return out


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and list(a.shape) == list(b.shape) and torch.equal(_bytes(a), _bytes(b))


def _snap(ids) -> Snap:
    s = Snap(ids=list(ids), tensors=_all_dtypes(), arr=np.arange(12, dtype=np.int32).reshape(3, 4),
             pair=(1, "two", torch.arange(4, dtype=torch.int64)), nested={"a": [1.5, None], "b": {"c": True}},
             label="x", n=len(ids), scale=0.25, flag=True, extra=99)
    s.cache_view = torch.zeros(2)                    # transient: not stored
    s._private = 5                                   # underscore: not stored
    return s


def _data_at(path: str) -> int:
    with open(path, "rb") as f:
        return spill._read_header_file(f)[1]


def _path(store: SpillStore, ids) -> str:
    return os.path.join(store.dir, key_name(ids_key(ids), len(ids)) + ".safetensors")


# -- encoding ----------------------------------------------------------------------------------------------------------
def test_dtype_table_has_fp8_and_uint8():
    assert spill._DT[torch.uint8] == "U8"
    assert spill._DT[torch.float8_e4m3fn] == "F8_E4M3"
    assert spill._DT[torch.float8_e5m2] == "F8_E5M2"
    assert spill._TD["F8_E4M3"] is torch.float8_e4m3fn


def test_encode_decode_in_memory():
    s = _snap(_ids(0, 8))
    items, layer = spill.encode(s, skip=("extra",))
    tensors = dict(items)
    assert not any(k.startswith("0.cache_view") or k.startswith("0._private") for k in tensors)
    got = spill.decode(layer, tensors, CLASSES)
    assert isinstance(got, Snap) and got.extra == 7 and got.later == [1, 2]
    assert not hasattr(got, "cache_view") and not hasattr(got, "_private")
    assert got.pair[:2] == (1, "two") and isinstance(got.pair, tuple)
    assert got.nested == {"a": [1.5, None], "b": {"c": True}}
    for code, t in s.tensors.items():
        assert _same(got.tensors[code], t), code
    with pytest.raises(ValueError):
        spill.decode(layer, tensors, ["os:path"])          # only the named classes are built
    with pytest.raises(TypeError):
        spill.encode(SimpleNamespace(x=object()))


def test_materialize_runs_first():
    seen = []
    spill.encode(SimpleNamespace(a=1), materialize=lambda o: seen.append(o.a))
    assert seen == [1]


@pytest.mark.parametrize("direct", [False, True])
def test_round_trip_every_dtype_with_rows(tmp_path, direct):
    st = _store(tmp_path, direct=direct)
    ids = _ids(100, 16)
    s = _snap(ids)
    fp8 = torch.randint(0, 256, (70, 1024), dtype=torch.uint8).view(torch.float8_e4m3fn)   # > a CHUNK: several buffers
    kv = torch.randint(0, 256, (33, 4096), dtype=torch.uint8)                               # FP8 KV rows as uint8
    job = st.save(ids, s, [("p0", kv), ("p1", fp8)], skip=("extra",), head=True)
    assert job is not None
    assert st.drain(30)
    assert job.ok and os.path.exists(_path(st, ids))
    crcs = spill._read_header_file(open(_path(st, ids), "rb"))[0]["__metadata__"]["crc32"].split(",")
    assert len(crcs) > 1                                    # several CRC blocks

    dst = torch.zeros_like(kv)
    got, rows, meta = st.load(ids_key(ids), lambda name, shape, dt: dst if name == "rows.p0" else None)
    assert rows["p0"] is dst and torch.equal(dst, kv)
    assert _same(rows["p1"], fp8)
    assert isinstance(got, Snap) and got.extra == 7 and got.label == "x" and got.n == 16
    assert np.array_equal(got.arr, s.arr) and got.arr.dtype == np.int32
    for code, t in s.tensors.items():
        assert _same(got.tensors[code], t), code
    assert meta["head"] == "1" and int(meta["n"]) == 16
    assert st.stats.loaded == 1 and st.stats.loaded_tokens == 16 and st.stats.written == 1


def test_load_destination_must_match(tmp_path):
    st = _store(tmp_path)
    ids = _ids(0, 8)
    st.save(ids, None, [("p0", torch.zeros(4, 8, dtype=torch.uint8))])
    assert st.drain(30)
    with pytest.raises(ValueError):
        st.load(ids_key(ids), lambda *a: torch.zeros(4, 9, dtype=torch.uint8))
    with pytest.raises(ValueError):
        st.load(ids_key(ids), lambda *a: torch.zeros(4, 8, dtype=torch.int8))
    with pytest.raises(FileNotFoundError):
        st.load(ids_key(_ids(50, 8)))


def test_read_while_still_writing(tmp_path):
    st = _store(tmp_path)
    ids = _ids(0, 8)
    kv = torch.randint(0, 256, (200, 4096), dtype=torch.uint8)
    st.save(ids, None, [("p0", kv)])
    _, rows, _ = st.load(ids_key(ids))                    # may meet the write in flight: it waits for it
    assert torch.equal(rows["p0"], kv)


# -- integrity ---------------------------------------------------------------------------------------------------------
def test_crc_failure_then_forget_drops_the_file(tmp_path):
    st = _store(tmp_path)
    ids = _ids(0, 8)
    kv = torch.randint(0, 256, (64, 4096), dtype=torch.uint8)
    st.save(ids, None, [("p0", kv)])
    assert st.drain(30)
    path = _path(st, ids)
    size = os.path.getsize(path)
    with open(path, "r+b") as f:                          # one byte in the third CRC block, size unchanged
        at = _data_at(path) + 150_000
        f.seek(at)
        b = f.read(1)
        f.seek(at)
        f.write(bytes([b[0] ^ 0xFF]))
    assert os.path.getsize(path) == size
    with pytest.raises(ValueError, match="block 2 does not match its CRC-32"):
        st.load(ids_key(ids))
    assert st.stats.crc_failed == 1
    st.forget(ids_key(ids))                               # what the engine hook does on a failed read
    assert not os.path.exists(path) and not st.has(ids)


def test_restart_scan_drops_partial_cut_and_foreign_files(tmp_path):
    st = _store(tmp_path)
    good, cut = _ids(0, 8), _ids(10, 8)
    for ids in (good, cut):
        st.save(ids, None, [("p0", torch.ones(16, 1024, dtype=torch.uint8))])
    assert st.drain(30)
    with open(_path(st, cut), "r+b") as f:
        f.truncate(os.path.getsize(_path(st, cut)) - 1)
    partial = os.path.join(st.dir, key_name(ids_key(_ids(20, 8)), 8) + ".safetensors.3.partial")
    junk = os.path.join(st.dir, "junk.safetensors")
    for p in (partial, junk):
        with open(p, "wb") as f:
            f.write(b"\0" * 64)

    again = _store(tmp_path)
    assert again.compat == st.compat
    assert again.has(good) and not again.has(cut)
    for p in (partial, junk, _path(st, cut)):
        assert not os.path.exists(p)
    _, rows, _ = again.load(ids_key(good))
    assert torch.equal(rows["p0"], torch.ones(16, 1024, dtype=torch.uint8))


def test_failed_write_marks_entry_bad_but_keeps_it_indexed(tmp_path):
    st = _store(tmp_path)
    ids = _ids(0, 8)
    os.chmod(st.dir, 0o500)
    try:
        job = st.save(ids, None, [("p0", torch.zeros(8, dtype=torch.uint8))])
        assert job is not None
        job.done.wait(30)
    finally:
        os.chmod(st.dir, 0o700)
    assert not job.ok and st.stats.write_failed == 1
    assert st.has(ids)                                    # every rank's index alike: a read decides, then forgets
    assert st.find(ids + [1]) is None                     # a bad entry is never offered
    with pytest.raises(FileNotFoundError):
        st.load_async(ids_key(ids), lambda *a: None)


# -- budget ------------------------------------------------------------------------------------------------------------
def _rows(i: int) -> list:
    return [("p0", torch.full((64, 1024), i, dtype=torch.uint8))]


def _one_size(tmp_path) -> int:
    probe = _store(tmp_path / "probe")
    probe.save(_ids(0, 8), None, _rows(0))
    probe.drain(30)
    return probe.index[ids_key(_ids(0, 8))].size


def test_cap_evicts_least_recently_used(tmp_path):
    size = _one_size(tmp_path)
    st = _store(tmp_path / "s", gib=2.5 * size / (1 << 30))
    a, b, c, d = (_ids(1000 * i, 8) for i in range(1, 5))
    for i, ids in enumerate((a, b)):
        assert st.save(ids, None, _rows(i)) is not None
    st.touch_key(ids_key(a))                              # a is now newer than b
    assert st.save(c, None, _rows(2)) is not None         # b goes
    assert st.drain(30)
    assert st.has(a) and not st.has(b) and st.has(c)
    assert not os.path.exists(_path(st, b))
    assert st.stats.pruned == 1
    assert st.save(d, None, _rows(3), keep=frozenset({ids_key(a)})) is not None   # a kept: c goes
    assert st.drain(30)
    assert st.has(a) and not st.has(c) and st.has(d)
    assert st.info()["entries"] == 2 and st.info()["stored_bytes"] == 2 * size


def test_skips_full_short_and_no_space(tmp_path):
    size = _one_size(tmp_path)
    st = _store(tmp_path / "s", gib=0.5 * size / (1 << 30))
    assert st.save(_ids(0, 8), None, _rows(0)) is None and st.stats.full == 1      # larger than the cap
    assert st.save(_ids(0, 3), None, _rows(0)) is None                              # under min_tokens
    st2 = _store(tmp_path / "t")
    ids = _ids(0, 8)
    assert st2.save(ids, None, _rows(0)) is not None
    assert st2.save(ids, None, _rows(0)) is None and st2.stats.skipped == 1         # stored already
    st3 = _store(tmp_path / "u", min_free_gib=1e9)
    assert st3.save(ids, None, _rows(0)) is None and st3.stats.no_space == 1
    assert st3.info()["entries"] == 0


def test_find_longest_prefix(tmp_path):
    st = _store(tmp_path)
    short, long_ = _ids(0, 8), _ids(0, 12)
    st.save(short, None, _rows(0))
    st.save(long_, None, _rows(1), head=False)
    assert st.drain(30)
    prompt = _ids(0, 20)
    assert st.find(prompt).n == 12
    assert st.find(prompt, grid=8).n == 8                 # 12 is off a grid of 8
    assert st.find(prompt, usable=lambda e: e.n < 10).n == 8
    assert st.find(long_).n == 8                          # the whole prompt needs its head row stored
    assert st.find(long_, whole_needs_head=False).n == 12
    assert st.find(_ids(1, 20)) is None


# -- builds and restart ------------------------------------------------------------------------------------------------
def test_other_builds_pruned(tmp_path):
    for name, age in (("spill-old", 1000), ("spill-new", 10)):
        os.makedirs(tmp_path / name / "rank0")
        t = os.path.getmtime(tmp_path) - age
        os.utime(tmp_path / name, (t, t))
    st = _store(tmp_path, keep_builds=1)
    assert not (tmp_path / "spill-old").exists() and (tmp_path / "spill-new").exists()
    assert os.path.isdir(st.dir)
    _store(tmp_path, keep_builds=0)
    assert not (tmp_path / "spill-new").exists()
    other = _store(tmp_path, signature="another build", keep_builds=1)
    assert other.compat != st.compat and os.path.isdir(st.top)


def test_flush_then_restart_reads_back(tmp_path):
    st = _store(tmp_path)
    prompts = [_ids(1000 * i, 8 + i) for i in range(3)]
    for i, ids in enumerate(prompts):
        st.save(ids, _snap(ids), _rows(i), skip=("extra",))
    assert st.drain(30) and st.pending_jobs() == 0
    again = _store(tmp_path)
    assert again.info()["entries"] == 3
    for i, ids in enumerate(prompts):
        e = again.find(ids + [7])
        assert e is not None and e.n == len(ids) and e.fields["label"] == "x"
        snap, rows, _ = again.load(e.key)
        assert snap.ids == ids and torch.equal(rows["p0"], _rows(i)[0][1])


# -- configuration -----------------------------------------------------------------------------------------------------
def test_config_checks():
    for bad in (dict(highwater=0.0), dict(highwater=1.5), dict(gib=0.0), dict(writers=0)):
        with pytest.raises(ValueError):
            _cfg("/x", **bad)
    assert _cfg("/x", highwater=1.0).highwater == 1.0


def test_config_from_args_and_env(monkeypatch):
    for k in ("TF_SPILL_WRITERS", "TF_SPILL_DIRECT", "TF_SPILL_FLUSH_S"):
        monkeypatch.delenv(k, raising=False)
    assert SpillConfig.from_args(SimpleNamespace(spill_gib=0)) is None
    assert SpillConfig.from_args(SimpleNamespace()) is None
    for root in (None, "", "none", "None"):
        with pytest.raises(ValueError):
            SpillConfig.from_args(SimpleNamespace(spill_gib=4, snapshot_dir=root))
    c = SpillConfig.from_args(SimpleNamespace(spill_gib=4, snapshot_dir="~/snaps"))
    assert c.root == os.path.expanduser("~/snaps") and c.gib == 4.0
    assert (c.highwater, c.min_tokens, c.min_free_gib, c.keep_builds) == (0.70, 8192, 50.0, 1)
    assert (c.writers, c.direct, c.flush_s) == (2, True, 60.0)
    monkeypatch.setenv("TF_SPILL_WRITERS", "3")
    monkeypatch.setenv("TF_SPILL_DIRECT", "0")
    monkeypatch.setenv("TF_SPILL_FLUSH_S", "5")
    c = SpillConfig.from_args(SimpleNamespace(spill_gib=2, snapshot_dir="/s", spill_highwater=0.5,
                                              spill_min_tokens=100, spill_min_free_gib=10, spill_keep_builds=2))
    assert (c.highwater, c.min_tokens, c.min_free_gib, c.keep_builds) == (0.5, 100, 10.0, 2)
    assert (c.writers, c.direct, c.flush_s) == (3, False, 5.0)
    monkeypatch.setenv("TF_SPILL_WRITERS", "0")
    with pytest.raises(ValueError):
        SpillConfig.from_args(SimpleNamespace(spill_gib=2, snapshot_dir="/s"))


def test_preallocate():
    spill.preallocate(None)
    assert spill._PINNED == {}
    spill.preallocate(_cfg("/x", writers=3))
    assert len(spill._PINNED["write"]) == 6 and len(spill._PINNED["read"]) == 2
    assert all(b.numel() == spill.CHUNK and not b.is_pinned() for b in spill._PINNED["write"])
    first = spill._PINNED["write"][0]
    spill.preallocate(_cfg("/x", writers=1))              # once only
    assert spill._PINNED["write"][0] is first


def test_weights_id_follows_files(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    a = spill.weights_id(tmp_path, None)
    (tmp_path / "notes.txt").write_bytes(b"xy")           # not a weight file
    assert spill.weights_id(tmp_path) == a
    (tmp_path / "model.safetensors").write_bytes(b"xy")
    assert spill.weights_id(tmp_path) != a


# -- two ranks ---------------------------------------------------------------------------------------------------------
class _Comm:
    """Host collectives for two rank threads: the k-th call on one rank meets the k-th on the other."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=60)
        self.slot: list = [None, None]

    def gather(self, rank: int, vals) -> list:
        self.slot[rank] = list(vals)
        self.barrier.wait()
        out = [list(s) for s in self.slot]
        self.barrier.wait()
        return out

    def share(self, rank: int, flat) -> list:
        if rank == 0:
            self.slot[0] = list(flat)
        self.barrier.wait()
        out = list(self.slot[0])
        self.barrier.wait()
        return out


def _two_ranks(fn) -> list:
    comm, out, errs = _Comm(), [None, None], []

    def run(r: int) -> None:
        try:
            out[r] = fn(r, comm)
        except BaseException as exc:  # noqa: BLE001
            errs.append(exc)
            comm.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    if errs:
        raise errs[0]
    return out


def _rank_store(root, r: int, comm: _Comm) -> SpillStore:
    return SpillStore(_cfg(root), rank=r, world=2, device="cpu", signature="sig", weights=f"w{r}", classes=CLASSES,
                      quiet=True, share=lambda flat: comm.share(r, flat), gather=lambda v: comm.gather(r, v))


def test_two_ranks_reconcile_index(tmp_path):
    a, b, c = _ids(0, 8), _ids(100, 8), _ids(200, 8)

    def first(r, comm):
        st = _rank_store(tmp_path, r, comm)
        for i, ids in enumerate((a, b, c)):
            assert st.save(ids, None, [("p0", torch.full((4 + r, 1024), i, dtype=torch.uint8))]) is not None
        assert st.drain(30)
        return st

    s0, s1 = _two_ranks(first)
    assert s0.compat == s1.compat and s0.top == s1.top and s0.dir != s1.dir
    assert {e.size for e in s0.index.values()} == {e.size for e in s1.index.values()} == {1 << 20}   # agreed

    os.remove(_path(s0, b))                               # rank 0 lost b
    t = os.path.getmtime(_path(s0, a))
    for ids, age in ((a, 30), (c, 20)):                   # rank 0: a then c; rank 1: the other way round
        os.utime(_path(s0, ids), (t - age, t - age))
    for ids, age in ((a, 10), (b, 20), (c, 30)):
        os.utime(_path(s1, ids), (t - age, t - age))

    r0, r1 = _two_ranks(lambda r, comm: _rank_store(tmp_path, r, comm))
    for st in (r0, r1):
        assert st.has(a) and st.has(c) and not st.has(b)
        assert st.index[ids_key(a)].last < st.index[ids_key(c)].last      # rank 0's order on both
    assert not os.path.exists(_path(r1, b))               # rank 1 dropped what rank 0 lacks
    assert [e.size for e in r0.index.values()] == [e.size for e in r1.index.values()]

    def read(r, comm):
        st = (r0, r1)[r]
        _, rows, _ = st.load(ids_key(c))
        return rows["p0"]

    got = _two_ranks(read)
    assert got[0].shape[0] == 4 and got[1].shape[0] == 5 and int(got[1][0, 0]) == 2


def test_a_picture_state_is_found_by_its_keyed_ids_only(tmp_path):
    """GLM --parallel (0054 + 0088): a kept state with pictures is stored under its keyed ids (``MultiDecoder._ident``:
    each picture's rows hold its content key), so the same placeholder ids with another picture never find it."""

    from types import SimpleNamespace

    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder, keyed_ids

    ids, rows = _ids(0, 12), [4, 5, 6, 7]                 # one picture's four rows at positions 4 .. 7
    mine, other = keyed_ids(ids, rows, [4], [111]), keyed_ids(ids, rows, [4], [222])
    c = SimpleNamespace(ids=ids, key=mine, images=True)
    st = _store(tmp_path)
    st.save(MultiDecoder._ident(c), None, _rows(0), tokens=c.ids)
    assert st.drain(30)
    assert st.find(mine + [1]).n == 12
    assert st.find(other + [1]) is None and st.find(ids + [1]) is None
    assert _store(tmp_path).find(mine + [1]).n == 12      # after a restart too (the file keeps the keyed ids)
