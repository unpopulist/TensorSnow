"""GLM DFlash2 draft dump and offline simulator (Mia's 0011), CPU only.

A fake target (logits a function of the position alone) and a fake DFlash2 drafter (the real ``Drafter.chain`` and
``Drafter.propose`` over synthetic candidates) run ``draft_dump.dump_decode`` and the real ``decode.dflash_decode``; the
written ``.npz`` loads back, passes ``draft_sim.check``, and the simulator's replay of the production rule equals the
live drafted run round for round.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling, choose_rows
from tensorfold.families.glm5_next.cuda import decode, dflash2, draft_dump, draft_sim
from tensorfold.families.glm5_next.cuda.decode import DepthPolicy, dflash_decode

V = 96              # vocabulary
RANK = 4            # selector rank
BLOCK = 6           # drafter block: 5 drafts a round at most
TOP_K = 8           # selector candidates a depth
PROMPT = 17         # prompt length: the reply's first sampled token sits at position PROMPT


def _target_row(pos: int) -> np.ndarray:
    """The fake target's logits for the token at ``pos`` (independent of the context)."""

    return np.random.default_rng(1000 + pos).normal(0.0, 2.0, V).astype(np.float32)


def _greedy(pos: int) -> int:
    row = _target_row(pos).astype(np.float64)
    return int(np.lexsort((np.arange(V), -row))[0])


class FakeEngine:
    def __init__(self, eos: int = -1) -> None:
        self.w = SimpleNamespace(cfg=SimpleNamespace(eos=[eos]), comm=None, world=1, vocab_offset=0)
        self.st = SimpleNamespace(pos=PROMPT)
        self.buf = None

    def verify_window(self, tokens):
        return list(tokens)

    def forward(self, tokens):
        return torch.from_numpy(np.stack([_target_row(self.st.pos + 1 + r) for r in range(len(tokens))]))

    def sample(self, logits, positions, sampling):
        x = logits.numpy().astype(np.float64)
        ids = np.broadcast_to(np.arange(V, dtype=np.int64), x.shape).copy()
        if sampling is None or sampling.temperature <= 0:
            return [int(i[np.lexsort((i, -r))[0]]) for r, i in zip(x, ids)]
        return choose_rows(x, ids, positions, sampling)

    def follow(self, tokens):
        pass

    def tap_rows(self, n):
        return torch.zeros((n, 1))


class FakeDrafter:
    """dflash2.Drafter's host side: candidates from a seeded table, the real chain/propose."""

    chain = dflash2.Drafter.chain
    propose = dflash2.Drafter.propose

    def __init__(self, hit: float = 0.7) -> None:
        g = np.random.default_rng(5)
        self.block, self.top_k = BLOCK, TOP_K
        self.succ = g.normal(0, 1, (V, RANK)).astype(np.float32)
        self.pred = g.normal(0, 1, (V, RANK)).astype(np.float32)
        self.context_end = PROMPT
        self.hit = hit

    def candidates(self, pending: int, depth: int):
        first = self.context_end + 1
        g = np.random.default_rng(7919 * first + pending)
        proj = g.normal(0, 0.3, (BLOCK - 1, RANK))[:depth]      # a depth's rows never depend on how many were asked
        tokens = np.zeros((depth, TOP_K), dtype=np.int64)
        values = np.zeros((depth, TOP_K), dtype=np.float64)
        for d in range(depth):
            ids = g.permutation(V)[:TOP_K]
            want = _greedy(first + d)
            if g.random() < self.hit and want not in ids:
                ids[g.integers(TOP_K)] = want          # the target's own choice is among the candidates
            vals = g.normal(0, 2, TOP_K).astype(np.float32).astype(np.float64)
            if want in ids and g.random() < self.hit:
                vals[list(ids).index(want)] = vals.max() + 1.0
            order = np.lexsort((ids, -vals))
            tokens[d], values[d] = ids[order], vals[order]
        return tokens, values, proj

    def add_taps(self, taps):
        self.context_end += int(taps.shape[0])


@pytest.fixture
def cpu(monkeypatch):
    def commit(w, st, b, R, keep):
        assert 1 <= keep <= R
        st.pos += keep

    # the fake engines do no GPU work: present a no-GPU environment so decode's _Clock times with the wall clock
    # instead of CUDA events. With CUDA present it took the event branch, but nothing here waits on the GPU, so
    # elapsed_time() raced the (empty) event's completion under load ("Both events must be completed", intermittent).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(draft_dump, "commit", commit)
    monkeypatch.setattr(decode, "commit", commit)


SAMPLED = Sampling(seed=1234, temperature=0.8, top_k=20, top_p=0.95)


def _dump(tmp_path, sampling, count=60, eos=-1, stop_eos=False, hit=0.7):
    e, dr = FakeEngine(eos), FakeDrafter(hit)
    first = e.sample(e.forward([0])[:1], [PROMPT], sampling)[0]
    e.st.pos, dr.context_end = PROMPT, PROMPT           # the prompt ends at PROMPT - 1; ``first`` sits at PROMPT
    res = draft_dump.dump_decode(e, dr, first, count, sampling, out_dir=tmp_path, stop_eos=stop_eos,
                                 meta={"note": "synthetic"})
    return first, res


def test_constants_match_the_drafter_and_engine():
    from tensorfold.families.glm5_next.cuda.engine import DFLASH_POLICY

    assert (draft_sim.EDGE, draft_sim.NOISE) == (dflash2.EDGE, dflash2.NOISE)
    assert f"fc{draft_dump.PROD_MOST}:{draft_dump.PROD_CONFIDENCE}" == DFLASH_POLICY
    assert draft_dump.TARGET_K >= 20 + draft_sim.MARGIN


def test_dump_dir_env():
    assert draft_dump.dump_dir({}) is None
    assert draft_dump.dump_dir({"TF_GLM_DRAFT_DUMP": ""}) is None
    assert str(draft_dump.dump_dir({"TF_GLM_DRAFT_DUMP": "/x/y"})) == "/x/y"


@pytest.mark.parametrize("sampling", [None, SAMPLED], ids=["greedy", "sampled"])
def test_dump_round_trip_and_check(cpu, tmp_path, sampling):
    first, res = _dump(tmp_path, sampling)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1 and str(files[0]) == res.dump_path
    assert not list(tmp_path.glob(".*.tmp.npz"))
    d = draft_sim.load_dump(res.dump_path)
    assert d.n == len(res.tokens) - 1 == 59 and d.depth == BLOCK - 1
    assert d.tokens.tolist() == res.tokens and d.tokens[0] == first
    assert d.first.tolist() == list(range(PROMPT + 1, PROMPT + 1 + d.n))
    assert d.cand_ids.shape == (d.n, BLOCK - 1, TOP_K) and d.edges.shape == (d.n, BLOCK - 2, TOP_K, TOP_K)
    assert d.tgt_ids.shape == (d.n, draft_dump.TARGET_K)
    assert d.sampled == (sampling is not None)
    assert d.meta["note"] == "synthetic" and d.meta["block"] == BLOCK and d.meta["version"] == draft_dump.VERSION
    # the reply is serial's: the target's own choice at every position
    e = FakeEngine()
    want = [e.sample(torch.from_numpy(_target_row(PROMPT + 1 + i)[None]), [PROMPT + 1 + i], sampling)[0]
            for i in range(d.n)]
    assert d.tokens[1:].tolist() == want
    c = draft_sim.check(d)
    assert c["prod_chain_ok"] == 1.0
    assert c["target_ok"] == 1.0
    if sampling is not None:
        assert c["noise_ok"]


@pytest.mark.parametrize("sampling", [None, SAMPLED], ids=["greedy", "sampled"])
def test_simulated_production_rule_equals_the_live_drafted_run(cpu, tmp_path, sampling):
    first, res = _dump(tmp_path, sampling)
    d = draft_sim.load_dump(res.dump_path)
    sim = draft_sim.simulate(d, draft_sim.parse_policy("fc5:0.3"))

    e, dr = FakeEngine(), FakeDrafter()
    live = dflash_decode(e, dr, first, 60, sampling, policy=DepthPolicy(5, fixed=True, confidence=0.3))
    assert live.tokens == res.tokens                       # drafting never changes the reply
    assert sim.keeps == live.keeps and sim.depths == live.depths
    assert (sim.rounds, sim.drafted, sim.accepted) == (live.rounds, live.drafted, live.accepted)
    assert sim.tokens == len(live.tokens) - 1
    assert 0 < sim.accepted < sim.drafted                  # the synthetic drafter is neither perfect nor useless


def test_eos_reply_room_and_acceptance(cpu, tmp_path):
    eos = _greedy(PROMPT + 1 + 9)                          # greedy reaches EOS at its 10th step (or earlier)
    first, res = _dump(tmp_path, None, count=60, eos=eos, stop_eos=True)
    d = draft_sim.load_dump(res.dump_path)
    assert d.tokens[-1] == eos and d.n <= 10 and d.eos == frozenset({eos})
    assert d.room(0) == 60 - 1                             # an EOS reply may draft up to max_tokens
    sim = draft_sim.simulate(d, draft_sim.parse_policy("fc5:0.3"))
    e, dr = FakeEngine(eos), FakeDrafter()
    live = dflash_decode(e, dr, first, 60, None, policy=DepthPolicy(5, fixed=True, confidence=0.3), stop_eos=True)
    assert live.tokens == res.tokens
    assert (sim.keeps, sim.depths) == (live.keeps, live.depths)


def _perfect_dump(n: int, depth: int, k: int = 2, miss: dict | None = None) -> draft_sim.Dump:
    """A greedy dump whose top candidate at every depth is the reply's next token, except ``miss`` {step: depth}."""

    tokens = np.arange(100, 101 + n, dtype=np.int64)
    cand = np.zeros((n, depth, k), dtype=np.int64)
    vals = np.zeros((n, depth, k), dtype=np.float64)
    for i in range(n):
        for dd in range(depth):
            t = int(tokens[i + 1 + dd]) if i + 1 + dd <= n else 1
            if miss and miss.get(i) == dd:
                t = 2
            cand[i, dd] = [t, 3]
            vals[i, dd] = [5.0, 0.0]
    return draft_sim.Dump(
        meta={"temperature": 0.0, "eos": [], "stop_eos": False}, tokens=tokens,
        first=np.arange(1, n + 1, dtype=np.int64), cand_ids=cand, cand_vals=vals,
        edge0=np.zeros((n, k)), edges=np.zeros((n, max(depth - 1, 0), k, k)), noise=None,
        tgt_ids=np.zeros((n, 1), dtype=np.int64), tgt_vals=np.zeros((n, 1)), tgt_lse=np.zeros(n),
        prod_chain=np.full((n, depth), -1, dtype=np.int64))


def test_acceptance_length_on_a_perfect_drafter():
    d = _perfect_dump(10, 3)
    r = draft_sim.simulate(d, draft_sim.parse_policy("f3"))
    assert r.keeps == [4, 4, 3] and r.depths == [3, 3, 2]      # the last round's room is the reply's 2 tokens
    assert r.tokens == 10 and r.rounds == 3 and r.accepted == r.drafted == 8
    one = draft_sim.simulate(d, draft_sim.parse_policy("f1"))
    assert one.keeps == [2] * 5 and one.tokens == 10
    none = draft_sim.simulate(d, draft_sim.parse_policy("f0"))
    assert none.keeps == [1] * 10 and none.drafted == 0


def test_a_miss_cuts_the_round():
    d = _perfect_dump(10, 3, miss={0: 1})                     # step 0: the second draft is wrong
    r = draft_sim.simulate(d, draft_sim.parse_policy("f3"))
    assert r.keeps[0] == 2 and r.depths[0] == 3                # one draft kept, plus the sampled token
    assert r.tokens == 10 and sum(r.keeps) >= 10


def test_costs_and_oracle_bound():
    d = _perfect_dump(30, 5, miss={i: i % 5 for i in range(0, 30, 3)})
    costs = draft_sim.Costs()
    sims = {s: draft_sim.run_policy([d], s, costs).summary() for s in ("f1", "f2", "f3", "f5", "fc5:0.3", "fa")}
    best = draft_sim.run_policy([d], "oracle5", costs).summary()
    assert all(best["tok_per_s"] >= s["tok_per_s"] - 1e-9 for s in sims.values())
    f1 = draft_sim.simulate(d, draft_sim.parse_policy("f1"), costs)
    assert f1.ms == pytest.approx(sum(costs.round_ms(dep, k, True) for dep, k in zip(f1.depths, f1.keeps)))


def test_best_depth_is_monotone():
    p = np.full(7, 0.8)
    flat = draft_sim.Costs(verify=(10.0,) * 8, block=0.0)
    assert draft_sim.best_depth(np.ones(7), flat) == 7         # free rows, certain drafts: draft everything
    assert draft_sim.best_depth(np.zeros(7), draft_sim.COSTS) == 0
    depths = []
    for slope in (0.0, 1.0, 3.0, 6.0, 12.0, 30.0):            # dearer rows: never deeper
        c = draft_sim.Costs(verify=tuple(30.0 + slope * r for r in range(8)), block=3.0)
        depths.append(draft_sim.best_depth(p, c))
    assert depths == sorted(depths, reverse=True) and depths[0] > depths[-1]
    depths = [draft_sim.best_depth(np.full(7, q), draft_sim.COSTS) for q in (0.1, 0.4, 0.6, 0.8, 0.9, 0.99)]
    assert depths == sorted(depths)                            # likelier drafts: never shallower


def test_parse_policy():
    assert isinstance(draft_sim.parse_policy("fc5:0.3"), draft_sim.ProductPolicy)
    assert isinstance(draft_sim.parse_policy("nc7:0.3"), draft_sim.NoisyProductPolicy)
    assert isinstance(draft_sim.parse_policy("cost7:noisy:2"), draft_sim.CostPolicy)
    assert isinstance(draft_sim.parse_policy("fa:0.5:0.9"), draft_sim.AdaptivePolicy)
    assert draft_sim.parse_policy("oracle7") == "oracle7"
    p = draft_sim.parse_policy("fc5:0.3@noise=1.0,trunc=1")
    assert p.rule == draft_sim.PickRule(edge=0.6, noise=1.0, trunc=True)
    for bad in ("x3", "fc5", "cost7:nope", "f3@bogus=1"):
        with pytest.raises(ValueError):
            draft_sim.parse_policy(bad)


def test_command_line(cpu, tmp_path, capsys):
    _dump(tmp_path, SAMPLED, count=40)
    _dump(tmp_path, None, count=40)
    assert len(list(tmp_path.glob("*.npz"))) == 2
    assert draft_sim.main([str(tmp_path), "--json", "--check", "--policy", "fc5:0.3", "--policy", "cost5:logit",
                           "--policy", "oracle5"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    checks = [x for x in lines if "prod_chain_ok" in x]
    assert len(checks) == 2 and all(c["prod_chain_ok"] == 1.0 and c["target_ok"] == 1.0 for c in checks)
    runs = {x["policy"]: x for x in lines if "policy" in x}
    assert set(runs) == {"fc5:0.3", "cost5:logit", "oracle5"}
    assert runs["oracle5"]["tok_per_s"] >= runs["fc5:0.3"]["tok_per_s"]
