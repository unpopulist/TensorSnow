"""The RoCE all-gather (``tensorfold/cuda/roce.cu``, ``roce.py``) on one GPU.

Single host, no NIC traffic: the proxy library is built only for its layout; the peer is played by the host, which
writes the peer's shard into the receive slot and that slot's per-HCA flag before the launch (what the remote
proxy's RDMA writes do). That checks the kernel's layout, rank-order output, slot alternation, epoch, CUDA graph replay
and the timeout path, plus setup failures being reported to every rank instead of leaving one in NCCL.

Two-rank tests over the real link are at the bottom and skipped unless TF_ROCE_TEST_RANK is set (see their docstring).
"""

from __future__ import annotations

import os
import types

import pytest
import torch

from tensorfold.cuda import roce

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def _fake_gather(rank: int = 0, world: int = 2, n_hca: int = 1, max_bytes: int = 64 << 10, spin: int = 2_000_000):
    """A RoceGather with its pinned region and counters but no proxy (ctx None): the host stands in for the peer."""

    lib = roce.proxy_library()
    g = object.__new__(roce.RoceGather)
    g.lib, g.ctx, g.rank, g.world, g.max_bytes = lib, None, rank, world, max_bytes
    g.slot_bytes = -(-max_bytes // roce.SLOT_ALIGN) * roce.SLOT_ALIGN
    import ctypes

    out = (ctypes.c_uint64 * 7)()
    assert lib.roce_layout(world, g.slot_bytes, out) == 0
    recv_off, flag_off, send_off, ctrl_off, total, g.flag_stride, g.slots = (int(v) for v in out)
    g.region = torch.zeros(total, dtype=torch.uint8, pin_memory=True)
    base = g.region.data_ptr()
    g.recv, g.flag, g.send, g.ctrl = base + recv_off, base + flag_off, base + send_off, base + ctrl_off
    g.ctrl_words = g.region[ctrl_off:ctrl_off + 28].view(torch.int32).numpy()
    g.classes = roce.BLOCKS.bit_length()
    g.counters = torch.zeros(2 + 2 * g.classes, dtype=torch.int32, device="cuda")
    g.hcas = [f"hca{h}" for h in range(n_hca)]
    g.spin = spin
    g.offs = types.SimpleNamespace(recv=recv_off, flag=flag_off, send=send_off, ctrl=ctrl_off)
    return g


def _play_peer(g, seq: int, shards: dict[int, torch.Tensor]) -> None:
    """Every other rank's shard into recv[src][seq & 1] and its flags to ``seq``, as the peers' proxies would."""

    slot = seq & 1
    for src, shard in shards.items():
        raw = shard.contiguous().view(torch.uint8).cpu()
        at = g.offs.recv + (src * g.slots + slot) * g.slot_bytes
        g.region[at:at + raw.numel()].copy_(raw)
        for h in range(len(g.hcas)):
            f = g.offs.flag + ((src * g.slots + slot) * len(g.hcas) + h) * g.flag_stride
            g.region[f:f + 4].view(torch.int32)[0] = seq


@pytest.mark.parametrize("rank,world,n_hca", [(0, 2, 1), (1, 2, 1), (0, 2, 2), (2, 3, 1)])
@pytest.mark.parametrize("numel", [4, 1024, 4096 * 4, 16 << 10])
def test_rank_order_output_and_epoch(rank, world, n_hca, numel):
    g = _fake_gather(rank, world, n_hca)
    for seq in (1, 2, 3):                                       # both slots, twice slot 1
        shards = {r: torch.randn(numel, device="cuda") + 100 * r + seq for r in range(world)}
        _play_peer(g, seq, {r: s for r, s in shards.items() if r != rank})
        recv = torch.full((world * numel,), float("nan"), device="cuda")
        g.all_gather(shards[rank], recv)
        torch.cuda.synchronize()
        assert torch.equal(recv, torch.cat([shards[r] for r in range(world)])), seq
        assert int(g.counters[0]) == seq and int(g.ctrl_words[0]) == seq     # epoch and doorbell
        assert int(g.ctrl_words[4 + (seq & 1)]) == numel * 4                 # the slot's nbytes for the proxy
        staged = g.region[g.offs.send + (seq & 1) * g.slot_bytes:][:numel * 4].view(torch.float32)
        assert torch.equal(staged, shards[rank].cpu())                      # what the proxy sends


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int32, torch.uint8])
def test_any_dtype_moves_bits(dtype):
    g = _fake_gather()
    mine = torch.randint(0, 120, (4096,), device="cuda").to(dtype)
    peer = torch.randint(0, 120, (4096,), device="cuda").to(dtype)
    _play_peer(g, 1, {1: peer})
    recv = torch.empty(8192, dtype=dtype, device="cuda")
    g.all_gather(mine, recv)
    torch.cuda.synchronize()
    assert torch.equal(recv, torch.cat([mine, peer]))


def test_graph_replay_advances_the_epoch():
    g = _fake_gather()
    mine = torch.zeros(2048, device="cuda")
    recv = torch.zeros(4096, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):                                  # one eager run (seq 1) warms up the extension
        _play_peer(g, 1, {1: torch.full((2048,), -1.0)})
        g.all_gather(mine, recv)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        g.all_gather(mine, recv)
    for seq in (2, 3, 4):
        mine.fill_(seq)
        peer = torch.arange(2048, dtype=torch.float32) * seq
        _play_peer(g, seq, {1: peer})
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(recv[:2048], mine) and torch.equal(recv[2048:].cpu(), peer), seq
        assert int(g.counters[0]) == seq
    g.check()


def test_a_missing_peer_poisons_and_check_raises():
    g = _fake_gather(spin=20_000)                               # ~tens of ms of polls
    mine, recv = torch.ones(1024, device="cuda"), torch.zeros(2048, device="cuda")
    g.all_gather(mine, recv)        # the launch's own check runs before the kernel ends: it cannot see this one yet
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match=r"RoCE all-gather failed \(sequence 1, peer 1, hca 0\).*no proxy"):
        g.check()                   # what comm.check(comm) after a synchronizing exchange does (the engine hook)
    assert int(g.counters[-1]) == 1 and int(g.counters[0]) == 0  # poisoned; the epoch does not advance
    _play_peer(g, 1, {1: torch.zeros(1024)})
    recv.fill_(7)
    with pytest.raises(RuntimeError, match="RoCE all-gather failed"):
        g.all_gather(mine, recv)                                # later launches do nothing
    assert bool((recv == 7).all())


class OneRankComm:
    """A two-rank all-gather where the other rank's rows are given: what the peer would contribute."""

    def __init__(self, peer_rows):
        self.peer_rows, self.calls = list(peer_rows), 0

    def all_gather(self, send, recv):
        peer = self.peer_rows[self.calls]
        self.calls += 1
        recv.copy_(torch.cat([send, peer.to(send.device, send.dtype)]))


def test_a_local_setup_failure_still_meets_the_peer(monkeypatch):
    """No proxy here (an unwritable cache, say): the verdict gather runs anyway and both ranks raise."""

    def broken():
        raise RuntimeError("the RoCE proxy's cache directory /opt/x is not writable")

    monkeypatch.setattr(roce, "proxy_library", broken)
    c = OneRankComm([torch.tensor([0, 1, 1, 128, 192], dtype=torch.int32)])
    with pytest.raises(RuntimeError, match="setup failed on rank 0: the RoCE proxy's cache directory"):
        roce.RoceGather(c, 0, 2)
    assert c.calls == 1


class FakeLib:
    """The proxy library's setup calls, succeeding without opening a device (the real layout)."""

    def __init__(self):
        self.real, self.destroyed = roce.proxy_library(), 0

    def roce_layout(self, *a):
        return self.real.roce_layout(*a)

    def roce_blob_bytes(self):
        return 192

    def roce_create(self, *a):
        return 1

    def roce_local_blob(self, ctx, buf, n):
        return 0

    def roce_stop(self, ctx):
        pass

    def roce_destroy(self, ctx):
        self.destroyed += 1


def test_a_peer_setup_failure_is_named(monkeypatch):
    lib = FakeLib()
    monkeypatch.setattr(roce, "proxy_library", lambda: lib)
    monkeypatch.setenv("TF_ROCE_HCA", "rocep1s0f0")
    c = OneRankComm([torch.tensor([1, 0, 0, 0, 0], dtype=torch.int32)])
    with pytest.raises(RuntimeError, match="setup failed on rank 1: see that rank's log"):
        roce.RoceGather(c, 0, 2)
    assert c.calls == 1 and lib.destroyed == 1


def test_differing_geometry_is_refused(monkeypatch):
    lib = FakeLib()
    monkeypatch.setattr(roce, "proxy_library", lambda: lib)
    monkeypatch.setenv("TF_ROCE_HCA", "rocep1s0f0")
    c = OneRankComm([torch.tensor([0, 2, 64, 128, 192], dtype=torch.int32)])     # the peer has two devices
    with pytest.raises(RuntimeError, match="geometry differs"):
        roce.RoceGather(c, 0, 2)


def test_a_peer_connect_failure_is_named(monkeypatch):
    lib = FakeLib()
    lib.roce_connect = lambda ctx, blobs, n: 0
    lib.roce_start = lambda ctx: 0
    monkeypatch.setattr(roce, "proxy_library", lambda: lib)
    monkeypatch.setenv("TF_ROCE_HCA", "rocep1s0f0")
    monkeypatch.delenv("TF_ROCE_MAX_KB", raising=False)
    ok_row = torch.tensor([0, 1, 64, 128, 192], dtype=torch.int32)
    c = OneRankComm([ok_row, torch.zeros(48, dtype=torch.int32), torch.tensor([1], dtype=torch.int32)])
    with pytest.raises(RuntimeError, match="could not connect on rank 1: see that rank's log"):
        roce.RoceGather(c, 0, 2)
    assert c.calls == 3


# ------------------------------------------------------------------------------------------------ two ranks (skipped)

TWO_RANK = os.environ.get("TF_ROCE_TEST_RANK")


@pytest.mark.skipif(TWO_RANK is None, reason="two-rank RoCE test: set TF_ROCE_TEST_RANK (0 here, 1 on the worker), "
                    "TF_ROCE_TEST_MASTER and NCCL_IB_HCA / NCCL_IB_GID_INDEX on both Sparks")
def test_two_ranks_equal_nccl_bit_for_bit_and_time_it():
    """Run at once on both Sparks (nothing else on the link's GPUs), e.g. on the head:
        TF_ROCE_TEST_RANK=0 TF_ROCE_TEST_MASTER=<rank0-roce-ip> NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \\
        NCCL_IB_GID_INDEX=3 python -m pytest -q -s tests/cuda/test_roce_kernel.py -k two_ranks
    and on the worker the same with TF_ROCE_TEST_RANK=1 and that node's RoCE v2 GID index."""

    import time

    from tensorfold.cuda.comm import open_comm

    rank = int(TWO_RANK)
    master = os.environ["TF_ROCE_TEST_MASTER"]
    port = int(os.environ.get("TF_ROCE_TEST_PORT", "29561"))
    torch.cuda.set_device(0)
    c = open_comm(rank, 2, master, port, backend="roce")
    c.barrier()
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    sizes = [4, 12, 16, 1000, 4096, 16 << 10 >> 2, 64 << 10, (256 << 10) >> 2, ((256 << 10) >> 2) + 4, 300_000]
    for n in sizes:                                             # fp32 elements a rank, both transports
        send = torch.randn(n, device="cuda", generator=gen)
        got, ref = torch.empty(2 * n, device="cuda"), torch.empty(2 * n, device="cuda")
        c.all_gather(send, got)
        c.nccl.all_gather(send, ref)
        torch.cuda.synchronize()
        assert torch.equal(got, ref), n
    c.settle()
    off = torch.empty(4097, device="cuda")                      # an unaligned send: same transport on both ranks
    got = torch.empty(8192, device="cuda")
    ref = torch.empty(8192, device="cuda")
    off[1:].normal_(generator=gen)
    c.all_gather(off[1:], got)
    c.nccl.all_gather(off[1:].contiguous(), ref)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)
    send, recv = torch.randn(4096, device="cuda", generator=gen), torch.empty(8192, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        c.all_gather(send, recv)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        c.all_gather(send, recv)
    ref = torch.empty_like(recv)
    for i in range(20):
        send.normal_(generator=gen)
        graph.replay()
        c.nccl.all_gather(send, ref)
        torch.cuda.synchronize()
        assert torch.equal(recv, ref), i
    c.check()

    def timed(fn, reps=2000):
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps * 1e6

    roce_us = timed(lambda: c.all_gather(send, recv))
    nccl_us = timed(lambda: c.nccl.all_gather(send, recv))
    print(f"\n[rank {rank}] 16 KiB all-gather: RoCE {roce_us:.1f} us, NCCL {nccl_us:.1f} us "
          f"(Mia measured 11 vs 45)", flush=True)
    c.barrier()
    c.close()
