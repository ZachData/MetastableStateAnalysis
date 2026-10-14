"""core/nudge.py's pure parts and the B0 gate's verdict (`trackB_circuit/design-B.md` "B0")."""
import numpy as np
import pytest

from core import nudge as nd

pytestmark = pytest.mark.pure


def test_seeds_and_directions_are_reproducible_and_unit():
    assert nd.seed_for("a", 5) == nd.seed_for("a", 5) != nd.seed_for("a", 6)
    R = nd.random_directions(64, 8, nd.seed_for("p", "resid_pre", 5))
    assert R.shape == (8, 64) and np.allclose(np.linalg.norm(R, axis=1), 1)
    assert np.array_equal(R, nd.random_directions(64, 8, nd.seed_for("p", "resid_pre", 5)))


def test_positions_respect_b1s_bounds():
    J = nd.draw_positions("homer_iliad", 242)
    assert len(J) == 32 == len(set(J.tolist())) and J.min() >= 8 and J.max() + 5 <= 241
    assert np.array_equal(J, nd.draw_positions("homer_iliad", 242))
    with pytest.raises(nd.NudgeError):
        nd.draw_positions("x", 40)


def test_rows_and_the_odd_even_split():
    idx, sign = nd.signed_rows(np.zeros((3, 4)))
    assert idx.tolist() == [-1, 0, 0, 1, 1, 2, 2] and sign.tolist() == [0, 1, -1, 1, -1, 1, -1]
    # f(x) = a·x + b·x² + x³ along a line: A = aε + ε³, S = bε²
    a, b, eps = 1.5, -0.7, 0.1
    f = lambda t: a * t + b * t ** 2 + t ** 3   # noqa: E731
    out = np.array([f(0.0), f(eps), f(-eps)])
    base, plus, minus = nd.split_rows(out)
    A, S = nd.anti_sym(plus, minus, base)
    assert np.isclose(A[0], a * eps + eps ** 3) and np.isclose(S[0], b * eps ** 2)


def test_rank_and_turn_on():
    assert nd.exact_rank(5.0, [1, 2, 3]) == 1
    assert nd.exact_rank(2.0, [1, 2, 3]) == 2                # a tie does not count against it
    assert nd.turn_on([0.01, 0.03, 0.1], [1e-9, 1e-3, 1e-1], [1e-6] * 3) == 0.03
    assert nd.turn_on([0.01], [1e-6], [1e-6]) is None
    assert nd.rms_norm(np.array([[3.0, 4.0], [0.0, 0.0]])) == pytest.approx(np.sqrt(12.5))


def _gpu(rate_hits: int, n: int = 224, g3_bad: int = 0, lin_off: int = 0, lin_miss: int = 0):
    from trackB_circuit import b0_gate as bg
    ranks = [1] * rate_hits + [5] * (n - rate_hits)
    lin = [2 if i < lin_off else k for i, k in enumerate(ranks)]   # first lin_off rank-1 hits unpredicted
    lin = [1 if rate_hits <= i < rate_hits + lin_miss else q for i, q in enumerate(lin)]   # predicted, not hit
    arms = [{"c": {str(c): {"rank": k} for c in nd.C_GRID}, "lin_rank": q} for k, q in zip(ranks, lin)]
    per = len(arms) // 7
    P = {}
    for a, key in enumerate(bg.V1_SCORED):
        g3 = {"grad_norm": 2.0, "c": {str(c): {"A0_over_eps": 2.0, "rank": 1} for c in nd.C_GRID}}
        if a < g3_bad:
            g3["c"]["0.1"]["rank"] = 2
        P[key] = {"G0_max_abs": 0.0, "G1": {"pass": True}, "G4a": {"pass": True}, "G3": g3,
                  "G4b": {"arms": arms[a * per:(a + 1) * per]}}
    cpu = {"prompts": {"k": {"positions": [{"c": {"0.01": {"n_tokens": 10, "n_ok": 9}}}]}}}
    return bg.verdict_step({"prompts": P}, cpu)


def test_gate_verdict_rules():
    v = _gpu(rate_hits=6)
    assert v["pass"] and v["G3_prompts"] == 7 and v["G2"]
    assert v["G4b"]["0.01"]["bound"] == pytest.approx(1 / 33 + 2 * np.sqrt((1 / 33) * (32 / 33) / 224))
    v20 = _gpu(rate_hits=20)                                  # 20/224 = 0.089 > 0.053: reported only
    assert v20["pass"] and v20["G4b"]["0.01"]["over_bound"] and v20["G4b_lin"]["rank1_nudged"] == 20
    v1 = _gpu(rate_hits=6, lin_off=1)                         # one rank-1 hit the linear response misses
    assert not v1["pass"] and v1["G4b_lin"]["rank1_mismatch"] == 1 and v1["G4b_lin"]["rank_within_1"] == 224
    v2 = _gpu(rate_hits=6, lin_miss=1)                        # step 128's way round: linear 1, nudged 5
    assert not v2["pass"] and v2["G4b_lin"]["rank1_mismatch"] == 1 and v2["G4b_lin"]["rank1_linear"] == 7
    assert _gpu(rate_hits=6, g3_bad=1)["pass"]                # 6 of 7
    assert not _gpu(rate_hits=6, g3_bad=2)["pass"]


def test_gate_prompts_are_b_pres():
    import ast
    from pathlib import Path
    from trackB_circuit import b0_gate as bg
    src = (Path(__file__).resolve().parents[1] / "tools/run/b_pre_check.py").read_text()
    node = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "V1_SCORED")
    assert tuple(ast.literal_eval(node.value)) == bg.V1_SCORED
