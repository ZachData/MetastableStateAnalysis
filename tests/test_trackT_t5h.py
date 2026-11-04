"""T5 U4 (`trackT_timaeus/t5h.py`, design-T5.md "U4, the causal arm, designed"): the schedule, the
stream's rows, the rule on planted series and the interaction's planted null (the loop against a
hand-written AdamW: `test_trackT_t5h_smoke.py`)."""
import math

import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports, transformers
from trackT_timaeus import t2, t5h

NULL_SEED = 20261103            # fixed before the test was first run


def test_the_recipe_constants():
    assert t5h.TOKENS_STEP == 524_288 and t5h.N_MAIN == 4369 * 256 and t5h.N_TAIL == 1601 * 256
    assert len(t5h.CKPT_STEPS) == 49 and t5h.CKPT_STEPS[0] == 0 and t5h.CKPT_STEPS[-1] == 4369
    assert t5h.CONT_STEPS == (5108, 5970)
    assert set(t5h.L_STEPS) <= set(t5h.CKPT_STEPS) and set(t5h.LP_STEPS) <= set(t5h.CONT_STEPS)
    assert t5h.TROUGH_BY in t5h.CKPT_STEPS and min(s for s in t2.GRID if s >= t5h.T_FROM) == 103
    assert (t5h.THETA_ALL, t5h.THETA_ARMS) == (51_911, 39_768)        # B11, rounded
    assert t5h.ARMS["A0pp"] == ("F", 2, False) and t5h.REPLICATES == ("A0", "A0p", "A0pp")
    assert (t5h.BAR_MIN, t5h.FLOOR_MAX) == (0.2, 0.2)                 # Blocked 133 (i) (d)
    assert t5h.BAND == {"F": (0.787, 0.881), "P": (0.415, 0.544)}


def test_the_schedule():
    P = t5h.PEAK
    assert t5h.lr_at(1) == pytest.approx(P / 1000) and t5h.lr_at(1000) == pytest.approx(P)
    assert t5h.lr_at(4369) / P == pytest.approx(0.6627, abs=1e-3)    # B3, floor 0
    assert t5h.lr_at(9536) == pytest.approx(0.0, abs=1e-15)
    cum = sum(t5h.lr_at(s) for s in range(1, 4370)) / P
    assert cum == pytest.approx(3469.7, abs=0.1)                      # B6
    with pytest.raises(t5h.T5hRefused):
        t5h.lr_at(0)


def test_the_rows():
    assert t5h.batch_rows(1) == ("main", 0, 256)
    assert t5h.batch_rows(4369) == ("main", t5h.N_MAIN - 256, t5h.N_MAIN)
    assert t5h.batch_rows(4370) == ("tail", 0, 256) and t5h.batch_rows(5970)[2] == t5h.N_TAIL
    for bad in (0, 5971):
        with pytest.raises(t5h.T5hRefused):
            t5h.batch_rows(bad)


def test_an_empty_segment_and_the_named_resume_crossing():
    # Parked 2 on #310: --until at or below the resume step trains nothing (and writes no segment line)
    assert t5h.nothing_to_train(3198, 103) and t5h.nothing_to_train(4369, 4369)
    assert not t5h.nothing_to_train(4369, 5970) and not t5h.nothing_to_train(0, 103)
    # A0's 4369 state was written at 66af413c31fe; U4b's first change names that crossing
    assert t5h.resume_allowed("66af413c31fe", "new") and t5h.resume_allowed("abc", "abc")
    assert not t5h.resume_allowed("abc", "new") and t5h.resume_allowed("abc", "new", allow_any=True)


def _dodeca_like(noise=0.0, seed=0):
    """A ρ_mid series on the checkpoint grid shaped as `-32m`'s (B4): 0.8 at 0, 1.0 by 40, the trough
    0.111 at 917, a rise to 0.763 at 3738 and 0.812 at 4369, on to 0.872 at 5970."""
    r = np.random.default_rng(seed)
    pts = {0: 0.8, 40: 1.0, 917: 0.111, 3738: 0.763, 4369: 0.812, 5108: 0.85, 5970: 0.872}
    ks = sorted(pts)
    out = {}
    for s in t5h.CKPT_STEPS + t5h.CONT_STEPS:
        v = float(np.interp(math.log1p(s), [math.log1p(k) for k in ks], [pts[k] for k in ks]))
        out[s] = v + (noise * r.standard_normal() if noise else 0.0)
    return out


def test_the_rule_on_planted_series():
    a0 = _dodeca_like()
    L0, (T0, ts) = t5h.late_level(a0), t5h.trough(a0)
    assert L0 == pytest.approx(0.7875) and (T0, ts) == (pytest.approx(0.111), 917)
    R0 = L0 - T0
    a0p = {s: v + 0.01 for s, v in a0.items()}                       # A0′: the seed noise, 0.01
    a0pp = {s: v - 0.005 for s, v in a0.items()}                     # A0″
    tau, bar = t5h.tau_bar(L0, t5h.late_level(a0p), t5h.late_level(a0pp))
    assert (tau, bar) == (pytest.approx(0.02), pytest.approx(0.2)) and t5h.readable(R0, bar)
    # τ is the largest of the three pairs: A0′ − A0″ here, not either pair with A0
    assert t5h.tau_bar(0.80, 0.85, 0.72) == (pytest.approx(0.13), pytest.approx(0.26))
    with pytest.raises(t5h.T5hRefused):
        t5h.tau_bar(0.80, 0.85)
    def arm(shift, cont_shift=None):
        k = dict(a0)
        for s in t5h.L_STEPS:
            k[s] -= shift
        for s in t5h.LP_STEPS:
            k[s] -= shift if cont_shift is None else cont_shift
        return k
    lab = lambda k: t5h.arm_label(L0 - t5h.late_level(k), R0, bar)
    assert lab(arm(R0 / 2)) == "lowers"                               # the design's planted "removes"
    assert lab(a0p) == "does not"                                     # A0′'s series
    assert lab(arm(0.25)) == "partly" and lab(arm(-0.25)) == "raises" and lab(arm(0.15)) == "does not"
    # the delay check
    dl = R0 / 2
    hold = arm(dl)
    assert t5h.delay_label(dl, t5h.late_level_cont(a0) - t5h.late_level_cont(hold), bar) == "holds"
    closes = arm(dl, cont_shift=dl / 3)
    assert t5h.delay_label(dl, t5h.late_level_cont(a0) - t5h.late_level_cont(closes), bar) == "delays"
    gone = arm(dl, cont_shift=0.05)
    assert t5h.delay_label(dl, t5h.late_level_cont(a0) - t5h.late_level_cont(gone), bar) == "delays"
    # the hypothesis labels
    assert t5h.hyp_label("lowers", "holds") == "supported" and t5h.hyp_label("partly", "holds") == "partly"
    assert t5h.hyp_label("lowers", "delays").startswith("not told apart")
    assert t5h.hyp_label("does not", None) == "not supported" == t5h.hyp_label("raises", "holds")
    assert "never" not in t5h.hyp_label("lowers", None) and "not told apart" in t5h.hyp_label("lowers", None)
    # the seed-noise gate: bar ≥ R/2 is unreadable
    assert not t5h.readable(0.15, 0.1)


def test_the_interaction():
    bar = t5h.BAR_MIN               # 0.2, the smallest bar the rule can give (Blocked 133 (i) (d)); /challenge-pr on #310, finding 4
    L = {"A0": 0.8, "A1": 0.8, "A2": 0.8, "A3": 0.8}
    assert t5h.interaction(L, bar) == (0.0, "no interaction seen at this noise")
    i, lab = t5h.interaction({**L, "A3": 0.8 - math.sqrt(2) * bar - 1e-9}, bar)
    assert lab == "together lower it more than the sum" and i == pytest.approx(math.sqrt(2) * bar)
    assert t5h.interaction({**L, "A1": 0.5}, bar)[1] == "less than the sum"
    assert t5h.interaction({**L, "A3": 0.8 - 0.28}, bar)[1] == "no interaction seen at this noise"   # < √2·0.2


@pytest.mark.parametrize("low, want", [(0.205, 0), (0.19, 3)])
def test_the_floor_gate_reads_the_pair_difference(tmp_path, monkeypatch, low, want):
    """The floor gate's second half (Blocked 133 (i) (d); /challenge-pr on #310, finding 4): nine seeds at
    0.40 and one at `low` keep 2·SD ≤ 0.2 (0.13 at 0.19), so only the largest pair difference decides
    (0.195 passes, 0.21 refuses). Records are planted where `floor` caches them; no model is loaded."""
    import json
    import types
    from trackC_writeup import c2_aniso
    monkeypatch.setattr(c2_aniso, "_torch_setup", lambda: (None, "cpu"))
    monkeypatch.setattr(t2, "p_tokenizer", lambda: None)
    monkeypatch.setattr(t5h, "prompts_hash", lambda: t5h.t5.PROMPTS_HASH)   # conftest stubs PROMPTS (no V1 keys)
    monkeypatch.setenv("METS_DATA", str(tmp_path / "data"))
    u2 = tmp_path / "data" / t5h.U2_DIR / "pythia-410m"
    u2.mkdir(parents=True)
    (tmp_path / "floor").mkdir()
    for s in t5h.FLOOR_STEPS:
        (u2 / f"step{s}.json").write_text(json.dumps({"rho_mid": 0.40}))
        for k in t5h.FLOOR_SEEDS:
            v = low if (k == 9 and s == 143000) else 0.40
            (tmp_path / "floor" / f"seed{k}_step{s}.json").write_text(json.dumps({"rho_mid": v}))
    assert t5h.floor(types.SimpleNamespace(root=str(tmp_path), hf_roots=None)) == want
    res = json.loads((tmp_path / "floor" / "floor.json").read_text())
    assert res["two_sd"]["143000"] <= 0.2 and (res["max_pair_diff"]["143000"] > 0.2) == bool(want)


NULL_DRAWS = 100_000            # /challenge-pr on #309, finding 2: at the design's 1,000 the test passed ~76 % of seeds


def interaction_null(sigma: float, rng, draws: int = NULL_DRAWS):
    """Six arms' L i.i.d. N(0.79, σ); τ from the largest of the A0, A0′, A0″ pairs (t5h.tau_bar's bar,
    vectorised).
    Returns (rate of 'interaction': |I| ≥ √2·bar; rate of one main-effect pair (A0, A1) reading
    'partly' or more either way: |ΔL| ≥ bar)."""
    L0, L0p, L0pp, L1, L2, L3 = rng.normal(0.79, sigma, (6, draws))
    reps = np.stack([L0, L0p, L0pp])
    bar = np.maximum(t5h.BAR_MIN, 2 * np.maximum(t5h.TAU_MIN, reps.max(0) - reps.min(0)))
    i = L1 + L2 - L0 - L3
    return float(np.mean(np.abs(i) >= math.sqrt(2) * bar)), float(np.mean(np.abs(L0 - L1) >= bar))


def test_the_vectorised_null_matches_the_rule_functions():
    """The vectorised rates equal `tau_bar`, `interaction` and `arm_label` applied draw by draw."""
    rng = np.random.default_rng(7)
    L = rng.normal(0.79, 0.12, (500, 6))      # σ large enough that bar's floor 0.2 does not decide every draw
    inter = main = 0
    for L0, L0p, L0pp, L1, L2, L3 in L:
        _, bar = t5h.tau_bar(L0, L0p, L0pp)
        inter += t5h.interaction({"A0": L0, "A1": L1, "A2": L2, "A3": L3}, bar)[1] != "no interaction seen at this noise"
        main += t5h.arm_label(L0 - L1, 10.0, bar) != "does not"
    L0, L0p, L0pp, L1, L2, L3 = L.T
    reps = np.stack([L0, L0p, L0pp])
    bar = np.maximum(0.2, 2 * np.maximum(0.02, reps.max(0) - reps.min(0)))
    assert (float(np.mean(np.abs(L1 + L2 - L0 - L3) >= math.sqrt(2) * bar)), float(np.mean(np.abs(L0 - L1) >= bar))) \
        == (inter / 500, main / 500)


@pytest.mark.parametrize("sigma", [0.03, 0.05])
def test_the_interactions_planted_null(sigma):
    """design-T5.md (finding 3 on #307): 'interaction' at most as often as a main-effect pair reads
    'partly' or more, plus 0.02. Read two-sided (I's label is two-sided; 'raises' is the pair's
    other side), (t11); 1e5 draws so the test reads the rates, not one draw of them."""
    inter, main = interaction_null(sigma, np.random.default_rng(NULL_SEED + int(sigma * 1000)))
    assert inter <= main + 0.02, (sigma, inter, main)
