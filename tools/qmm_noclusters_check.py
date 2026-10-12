#!/usr/bin/env python3
"""tools/qmm_noclusters_check.py [--time]: Mia's patch 0103 (by BadAd84) as ported to this tree. Decode windows' 4-bit
matmuls (tensorfold/cuda/kernels/qmm) reduce their K slices through the partials buffer and reduce_kernel instead of a
thread-block cluster when the window has at most 64 rows and the weight holds fewer than 28 Mi values.

Here, unlike Mia's tree, sm_12x runs groups of 64 through the grouped kernel (qmm_group.cu), which always reduces in a
cluster; ``qmm.matmul`` now asks qmm.cu's rule (``clusters``) first and takes qmm.cu's buffer path where it says so.
In RED-SNOW only the DFlash2 drafter's matmuls are 4-bit, so the shapes are its rank-0 shapes at TP=2.

Run on one sm_12x GPU (GB10), with the server stopped, from the tree:
    PYTHONPATH=src python tools/qmm_noclusters_check.py [--time]

NEW = ``glm qmm.matmul`` (what the drafter calls); OLD = the grouped kernel called directly (``qmm_group``, the path
before the port, built in the same extension). At the drafter's shapes and 1-128 rows:
1. bits: NEW == OLD bit for bit, outputs pre-filled with NaN;
2. mechanism: NEW launches reduce_kernel (and no group_kernel) exactly when sk > 1, rows <= 64 and n * k < 28 Mi
   (torch.profiler's kernel names), else group_kernel; with TF_GLM_QMM_CLUSTERS=1 (read once a process: a child
   process) NEW runs group_kernel where it would take the buffer;
3. graphs: matmuls on three side streams captured in one CUDA graph, replayed with new inputs == OLD eager (the
   buffer path's partials are allocated per call, stream-ordered);
4. the extension loads as tensorfold_qmm_v6.
One PASS / FAIL line a check; exits 1 on any FAIL. --time: CUDA-graph us a call, OLD -> NEW, cold L2, rows 1-64."""
import os
import subprocess
import sys

import torch

from tensorfold.cuda.kernels import qmm as shared
from tensorfold.cuda.kernels.qmm_tiles import group_tile
from tensorfold.families.glm5_next.cuda import qmm

ROWS = (1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 65, 96, 128)
D, HD, H, KV, INTER, TAPS, CONV_GS = 4096, 128, 32 // 2, 8 // 2, 12288 // 2, 5, 16   # the drafter at TP=2, rank 0
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += not ok


def shapes():
    """(label, n, k, fp32 out) of the DFlash2 drafter's 4-bit matrices on rank 0 at TP=2 (dflash2.py), plus two shapes
    either side of the 28 Mi rule."""
    return [("taps fc", D, TAPS * D, False), ("attention_conv kp", 2 * 2 * D // CONV_GS, D, False),
            ("qkv", (H + 2 * KV) * HD, D, False), ("ctx kv", 2 * KV * HD, D, False), ("o", D, H * HD, True),
            ("gate/up", 2 * INTER, D, False), ("down", D, INTER, True),
            ("28 Mi values exactly", 7168, 4096, False), ("just under 28 Mi", 7040, 4096, False)]


def main():
    dev = torch.device("cuda")
    src = open(os.path.join(os.path.dirname(shared.__file__), "qmm.py")).read()
    check('name="tensorfold_qmm_v6"' in src and 'name="tensorfold_qmm_v5"' not in src,
          "the extension loads as tensorfold_qmm_v6 (not v5's build from a shared kernel folder)")
    check(torch.cuda.get_device_capability()[0] == 12, "an sm_12x GPU (the grouped kernel's path)")
    ext = shared._ext()
    g = torch.Generator(device=dev).manual_seed(86)

    def weights(n, k):
        words = torch.randint(-2 ** 31, 2 ** 31 - 1, (n, k // 8), dtype=torch.int32, device=dev, generator=g)
        sc = (torch.rand((n, k // 64), device=dev, generator=g) * 0.02 + 0.001).to(torch.bfloat16)
        bi = ((torch.rand((n, k // 64), device=dev, generator=g) - 0.5) * 0.02).to(torch.bfloat16)
        return qmm.make_q4(words, sc, bi)

    def empty(m, q, f32):
        return torch.full((m, q.n), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)

    def new(x, q, f32, out=None):
        out = empty(x.shape[0], q, f32) if out is None else out
        return qmm.matmul(x, q, out=out, f32=f32)

    def old(x, q, f32, out=None):
        out = empty(x.shape[0], q, f32) if out is None else out
        tile = group_tile(x.shape[0], *shared._chip(torch.cuda.current_device()))
        ext.qmm_group(x, qmm.group_sums(x), [q.weight], [q.scales], [q.biases], [out], [q.n],
                      [qmm.split_k(q.n, q.k)], f32, tile, -1)
        return out

    def bits(t):
        return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)

    def kernels(fn, *a):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*a)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        return any("group_kernel" in nm for nm in names), any("reduce_kernel" in nm for nm in names)

    if "--clusters-child" in sys.argv:              # the opt-out check's child: TF_GLM_QMM_CLUSTERS=1 in its env
        q = weights((H + 2 * KV) * HD, D)
        x = (torch.randn((8, D), device=dev, generator=g) * 0.5).to(torch.bfloat16)
        grp, red = kernels(new, x, q, False)
        print(f"CHILD {'cluster' if grp and not red else 'buffer'}", flush=True)
        sys.exit(0)

    for label, n, k, f32 in shapes():
        label = f"{label} ({n} x {k})"
        q = weights(n, k)
        sk = qmm.split_k(n, k)
        differ, wrong = [], []
        for m in ROWS:
            x = (torch.randn((m, k), device=dev, generator=g) * 0.5).to(torch.bfloat16)
            try:
                ref = old(x, q, f32)
                if torch.isnan(ref.float()).any():
                    differ.append(f"m={m}: OLD left a slot unwritten")
                if not torch.equal(bits(new(x, q, f32)), bits(ref)):
                    differ.append(f"m={m}")
                want = sk > 1 and m <= 64 and n * k < (28 << 20)
                grp, red = kernels(new, x, q, f32)
                if red != want or grp == want:
                    wrong.append(f"m={m}: NEW {'buffer' if red else 'cluster'}, want {'buffer' if want else 'cluster'}")
            except RuntimeError as exc:
                differ.append(f"m={m}: raised {str(exc).splitlines()[0][:100]}")
        check(not differ, f"{label}: NEW == OLD bit for bit at rows {ROWS[0]}-{ROWS[-1]} ({differ[:3] or 'all equal'})")
        check(not wrong, f"{label}: sk {sk}, the buffer path exactly at rows <= 64 below 28 Mi values "
                         f"({wrong[:3] or 'as wanted'})")
        del q
    child = subprocess.run([sys.executable, os.path.abspath(__file__), "--clusters-child"], capture_output=True,
                           text=True, env={**os.environ, "TF_GLM_QMM_CLUSTERS": "1"})
    check("CHILD cluster" in child.stdout,
          "TF_GLM_QMM_CLUSTERS=1: an 8-row drafter qkv runs the grouped kernel's cluster, the rule before the port "
          f"({(child.stdout.strip().splitlines() or [child.stderr.strip()[-200:]])[-1]})")

    picks = [s for s in shapes() if s[0] in ("attention_conv kp", "qkv", "o", "down")]
    qs = [weights(n, k) for _, n, k, _ in picks]
    m = 16
    xin = [torch.zeros((m, k), dtype=torch.bfloat16, device=dev) for _, _, k, _ in picks]
    outs = [torch.zeros((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev) for _, n, _, f32 in picks]
    streams = [torch.cuda.Stream() for _ in range(3)]

    def body():
        main_st = torch.cuda.current_stream()
        for i, (_, n, k, f32) in enumerate(picks):
            st = streams[i % 3]
            st.wait_stream(main_st)
            with torch.cuda.stream(st):
                for _ in range(3):          # repeated calls: partials buffers recycled within the stream
                    new(xin[i], qs[i], f32, outs[i])
            main_st.wait_stream(st)

    bad = []
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        for rep in range(2):
            for i, (_, n, k, f32) in enumerate(picks):
                xin[i].copy_((torch.randn((m, k), device=dev, generator=g) * (0.5 + rep)).to(torch.bfloat16))
                outs[i].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for i, (label, n, k, f32) in enumerate(picks):
                if not torch.equal(bits(outs[i]), bits(old(xin[i], qs[i], f32))):
                    bad.append(f"replay {rep}: {label}")
    except RuntimeError as exc:
        bad.append(f"raised {str(exc).splitlines()[0][:100]}")
    check(not bad, f"a CUDA graph of matmuls on three side streams, replayed twice with new inputs == OLD eager "
                   f"({bad or 'all equal'})")

    if "--time" in sys.argv:
        def timed(fn, x, qcopies, f32, reps=20):
            out = empty(x.shape[0], qcopies[0], f32)

            def body2():
                for q in qcopies:
                    fn(x, q, f32, out)
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                body2()
            torch.cuda.current_stream().wait_stream(st)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                body2()
            gr.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps):
                gr.replay()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) * 1e3 / reps / len(qcopies)

        for label, n, k, f32 in shapes()[:7]:
            q0 = weights(n, k)
            copies = max(2, min(64, -(-(256 << 20) // q0.nbytes())))
            qc = [q0] + [qmm.Q4(q0.weight.clone(), q0.scales.clone(), q0.biases.clone(), n, k)
                         for _ in range(copies - 1)]
            line = []
            for m in (1, 4, 8, 16, 32, 64):
                x = (torch.randn((m, k), device=dev, generator=g) * 0.5).to(torch.bfloat16)
                line.append(f"{m}: {timed(old, x, qc, f32):.1f} -> {timed(new, x, qc, f32):.1f}")
            print(f"TIME {label} ({n} x {k}, sk {qmm.split_k(n, k)}) us a call, rows " + ", ".join(line), flush=True)
            del qc, q0
            torch.cuda.empty_cache()
    print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:   # noqa: BLE001 - an error in the extension under test (e.g. a CUDA fault) is a failure
        print(f"FAIL the check stopped on {type(exc).__name__}: {str(exc).strip().splitlines()[0][:150]}", flush=True)
        sys.exit(1)
