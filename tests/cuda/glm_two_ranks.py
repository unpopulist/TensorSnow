"""Two GLM ranks in one process on one GPU (tests): each rank a thread with its own weights half, engine and
communicator; the communicators pair their collectives through host barriers, so rank r's k-th all-gather or
exchange meets the peer's k-th. Eager only (no CUDA graphs: a capture in one thread would see the other's launches).

Process-wide scratch the loader shares (exl3_dense.SCRATCH) is given to each rank separately (``_own_scratch``).
The arithmetic is the two-rank model's (real partials of both halves, summed rank 0 first), slowly: every collective
synchronizes the stream it runs on."""

from __future__ import annotations

import threading

import torch


class Hub:
    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=600)
        self.slot: list = [None, None]

    def _meet(self, rank: int, item):
        self.slot[rank] = item
        self.barrier.wait()
        return self.slot[1 - rank]

    def _done(self) -> None:
        torch.cuda.current_stream().synchronize()
        self.barrier.wait()                       # the peer has read this rank's tensors


class PairComm:
    def __init__(self, rank: int, hub: Hub) -> None:
        self.rank, self.world, self.hub = rank, 2, hub
        self.calls = 0

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        torch.cuda.current_stream().synchronize()
        n = send.numel()
        mine = send.reshape(-1).clone()          # in place gathers (send inside recv) stay correct
        theirs = self.hub._meet(self.rank, mine)
        flat = recv.view(-1)
        flat[self.rank * n:(self.rank + 1) * n].copy_(mine)
        flat[(1 - self.rank) * n:(2 - self.rank) * n].copy_(theirs)
        self.calls += 1
        self.hub._done()

    def exchange(self, sends, recvs, peer: int) -> None:
        if peer != 1 - self.rank:
            raise ValueError("exchange: the peer is the other rank")
        torch.cuda.current_stream().synchronize()
        mine = [s.contiguous() for s in sends]
        theirs = self.hub._meet(self.rank, mine)
        for r, t in zip(recvs, theirs):
            r.view(-1).copy_(t.reshape(-1))
        self.calls += 1
        self.hub._done()

    def barrier(self) -> None:
        torch.cuda.synchronize()
        self.hub.barrier.wait()

    def ready(self, label: str, **kwargs) -> None:
        pass


def both(hub: Hub, fn0, fn1):
    """fn0() and fn1() on two threads (rank 0, rank 1); their results, re-raising the first failure (which breaks
    the hub's barrier, so the other rank does not wait for it)."""

    if hub.barrier.broken:
        hub.barrier.reset()
    out: list = [None, None]
    err: list = [None, None]

    def run(i, fn):
        try:
            torch.cuda.set_device(0)
            out[i] = fn()
        except BaseException as e:            # noqa: BLE001 - re-raised below
            err[i] = e
            hub.barrier.abort()

    threads = [threading.Thread(target=run, args=(i, f)) for i, f in enumerate((fn0, fn1))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in err:
        if e is not None:
            raise e
    return out


def load_pair(path, *, layers=None, mtp: bool = False):
    """Both ranks' weights of a checkpoint, each with its PairComm (one Hub)."""

    from tensorfold.families.glm5_next.cuda import weights

    hub = Hub()
    ws = []
    for rank in (0, 1):
        kw = {} if layers is None else {"layers": layers}
        w = weights.load(path, rank=rank, device="cuda", mtp=mtp, world=2, **kw)
        w.comm = PairComm(rank, hub)
        _own_scratch(w)
        ws.append(w)
    return ws, hub


def _own_scratch(w) -> None:
    """The rank's EXL3 dense matrices on a scratch of their own: the loader gives every X3 / Stack the module's one
    SCRATCH (one rank a process), which two rank threads would race on."""

    from tensorfold.families.glm5_next.cuda.exl3_dense import Scratch, Stack
    from tensorfold.families.glm5_next.cuda.weights import exl3_users

    users = exl3_users(w)
    if not users:
        return
    sc = Scratch()
    for u in users:
        u.sc = sc
        for p in (u.parts if isinstance(u, Stack) else []):
            if hasattr(p, "sc"):
                p.sc = sc
    sc.reserve(users, w.device)
