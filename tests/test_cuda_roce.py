"""The RoCE transport's host side without a NIC or a peer: its settings, the TF_COMM_BACKEND=roce registration, and
RoceComm's choice of transport (by byte count alone, the same on every rank), aligned copies and startup barriers.
The kernel is tested on one GPU in tests/cuda/test_roce_kernel.py; the two-rank exchange there too (skipped)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import comm, roce  # noqa: E402


def test_wait_limit_from_env(monkeypatch):
    monkeypatch.delenv("TF_ROCE_WAIT_S", raising=False)
    assert roce.spin_limit() == roce.SPIN_LIMIT
    monkeypatch.setenv("TF_ROCE_WAIT_S", "300")
    assert roce.spin_limit() == 300_000_000
    for bad in ("0.5", "3601", "soon"):
        monkeypatch.setenv("TF_ROCE_WAIT_S", bad)
        with pytest.raises(ValueError, match="TF_ROCE_WAIT_S"):
            roce.spin_limit()


def test_max_bytes_from_env(monkeypatch):
    monkeypatch.delenv("TF_ROCE_MAX_KB", raising=False)
    assert roce.max_bytes_from_env() == 256 << 10
    monkeypatch.setenv("TF_ROCE_MAX_KB", "512")
    assert roce.max_bytes_from_env() == 512 << 10
    for bad in ("0", "-4", "big", "70000"):
        monkeypatch.setenv("TF_ROCE_MAX_KB", bad)
        with pytest.raises(ValueError, match="TF_ROCE_MAX_KB"):
            roce.max_bytes_from_env()


def _sysfs(tmp_path: Path, states: dict[str, str]) -> Path:
    for name, state in states.items():
        d = tmp_path / name / "ports" / "1"
        d.mkdir(parents=True)
        (d / "state").write_text(state + "\n")
    return tmp_path


def test_devices_from_env_then_nccl_then_sysfs(monkeypatch, tmp_path):
    root = _sysfs(tmp_path, {"rocep1s0f0": "4: ACTIVE", "rocep1s0f1": "1: DOWN", "roceP2p1s0f0": "4: ACTIVE",
                             "roceP2p1s0f1": "1: DOWN"})
    monkeypatch.delenv("TF_ROCE_HCA", raising=False)
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    # sorted() puts the capital P first: with neither variable set, both twins of the port are used
    assert roce.hcas_from_env(root) == ["roceP2p1s0f0", "rocep1s0f0"]
    monkeypatch.setenv("NCCL_IB_HCA", "=rocep1s0f0:1")
    assert roce.hcas_from_env(root) == ["rocep1s0f0"]
    monkeypatch.setenv("TF_ROCE_HCA", "rocep1s0f0,roceP2p1s0f0,extra")
    assert roce.hcas_from_env(root) == ["rocep1s0f0", "roceP2p1s0f0"]


def test_gid_index_from_env(monkeypatch):
    monkeypatch.delenv("TF_ROCE_GID_INDEX", raising=False)
    monkeypatch.delenv("NCCL_IB_GID_INDEX", raising=False)
    assert roce.gid_index_from_env() == 3
    monkeypatch.setenv("NCCL_IB_GID_INDEX", "4")
    assert roce.gid_index_from_env() == 4
    monkeypatch.setenv("TF_ROCE_GID_INDEX", "5")
    assert roce.gid_index_from_env() == 5


def test_roce_is_a_registered_backend(monkeypatch):
    assert "roce" in comm.BACKENDS
    made = []
    monkeypatch.setattr(roce, "RoceComm", lambda base: made.append(base) or ("roce", base))
    monkeypatch.setattr(comm, "NCCL", lambda *a, **k: "nccl")
    monkeypatch.setenv(comm.BACKEND_ENV, "roce")
    assert comm.open_comm(0, 2, "192.0.2.1", 29551) == ("roce", "nccl") and made == ["nccl"]


def test_settle_is_optional():
    comm.settle(object())
    seen = []

    class T:
        def settle(self):
            seen.append(1)

    comm.settle(T())
    assert seen == [1]


class FakeNccl:
    def __init__(self, rank=0, world=2):
        self.rank, self.world, self.calls, self.store = rank, world, [], "store"

    def all_gather(self, send, recv):
        self.calls.append(("gather", send.numel() * send.element_size()))
        recv.copy_(torch.cat([send] * self.world))

    def barrier(self):
        self.calls.append(("barrier", 0))

    def ready(self, label):
        return label


class FakeGather:
    """RoceGather's interface: the size rule and a recorded gather (the peer's shard equals this one's)."""

    hcas = ["rocep1s0f0"]
    spin = roce.SPIN_LIMIT

    def __init__(self, base, rank, world, max_bytes):
        self.world, self.max_bytes, self.calls, self.checks = world, max_bytes, [], 0

    fits_bytes = roce.RoceGather.fits_bytes

    def all_gather(self, send, recv):
        assert send.is_contiguous() and recv.is_contiguous()
        assert send.data_ptr() % roce.PACK == 0 and recv.data_ptr() % roce.PACK == 0
        self.calls.append(send.numel() * send.element_size())
        recv.copy_(torch.cat([send] * self.world))

    def check(self):
        self.checks += 1

    def close(self):
        pass


@pytest.fixture
def pair(monkeypatch):
    monkeypatch.setattr(roce, "RoceGather", FakeGather)
    monkeypatch.delenv("TF_ROCE_MAX_KB", raising=False)
    base = FakeNccl()
    return base, roce.RoceComm(base)


def test_rank0_says_it_is_live(monkeypatch, capsys):
    monkeypatch.setattr(roce, "RoceGather", FakeGather)
    roce.RoceComm(FakeNccl(rank=1))
    assert capsys.readouterr().out == ""
    roce.RoceComm(FakeNccl(rank=0))
    assert "RoCE all-gather is live: gathers up to 256 KiB a rank over rocep1s0f0" in capsys.readouterr().out


def test_transport_by_bytes(pair):
    base, c = pair
    for n, over_roce in ((4, True), (256 << 10 >> 2, True), ((256 << 10 >> 2) + 4, False), (3, False), (0, False)):
        send, recv = torch.arange(n, dtype=torch.float32), torch.empty(2 * n)
        before = len(c.roce.calls)
        c.settle()
        c.all_gather(send, recv)
        assert (len(c.roce.calls) > before) == over_roce, n
        assert torch.equal(recv, torch.cat([send, send]))
    assert c.small == 2 and c.large == 3
    assert all(kind == "gather" for kind, _ in base.calls)          # NCCL only for the big / odd ones


def test_unaligned_buffers_copy_instead_of_changing_transport(pair):
    base, c = pair
    c.settle()
    big = torch.arange(65, dtype=torch.float32)
    send = big[1:]                                                  # 4 bytes past a 16-byte boundary
    out = torch.zeros(129)
    recv = out[1:]
    c.all_gather(send, recv)
    assert c.roce.calls == [256] and base.calls == []
    assert torch.equal(recv, torch.cat([send, send]))
    strided = torch.arange(128, dtype=torch.float32)[::2]           # not contiguous: copied, still RoCE
    r2 = torch.empty(128)
    c.all_gather(strided, r2)
    assert c.roce.calls == [256, 256] and torch.equal(r2, torch.cat([strided, strided]))


def test_barrier_first_until_settled(pair):
    base, c = pair
    send, recv = torch.zeros(4), torch.empty(8)
    c.all_gather(send, recv)
    assert base.calls == [("barrier", 0)]
    c.settle()
    c.all_gather(send, recv)
    assert base.calls == [("barrier", 0)]


def test_shape_check_and_passthrough(pair):
    base, c = pair
    with pytest.raises(ValueError, match="world x send"):
        c.all_gather(torch.zeros(4), torch.zeros(4))
    assert c.store == "store" and c.ready("loading") == "loading" and (c.rank, c.world) == (0, 2)
    assert c.all_gather_fast.__func__ is roce.RoceComm.all_gather
    c.barrier()
    assert base.calls[-1] == ("barrier", 0)
    c.check()
    assert c.roce.checks == 1


def test_one_rank_refused(monkeypatch):
    monkeypatch.setattr(roce, "RoceGather", FakeGather)
    with pytest.raises(ValueError, match="two or more ranks"):
        roce.RoceComm(FakeNccl(world=1))
