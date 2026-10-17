"""
`trackA_force/a2_ablate.py` — A2 unit 5's pure parts: types and bands, size deciles, the
size-matched null (Blocked 59 (b): MLPs in any band), the arms, the step rule and the reading.
The hooked ablation on a tiny GPT-NeoX: `test_trackA_a2_ablate_smoke.py`.
"""
import numpy as np
import pytest

from trackA_force import a2_ablate as ab

pytestmark = pytest.mark.deps  # torch
K = ab.K


def test_types_bands_and_deciles():
    assert ab.ctype(0) == "head" and ab.ctype(K - 2) == "bias" and ab.ctype(K - 1) == "mlp"
    assert ab.cband(0) == 0 and ab.cband(8 * K) == 1 and ab.cband(23 * K + 5) == 2
    d = ab.deciles(np.arange(432.0))
    assert d.min() == 0 and d.max() == 9 and np.bincount(d).min() >= 43


def _sizes():
    s = np.ones(24 * K)
    s[K - 1::K] = 100.0 + np.arange(24)          # every MLP in the top decile
    s[:16] = 50.0                                # block 0's heads big too
    return s


def test_null_matches_type_and_decile_mlps_in_any_band():
    dec = ab.deciles(_sizes())
    arm = [1 * K + K - 1, 2 * K + K - 1, 3 * K + K - 1, 5 * K + K - 1, 7 * K + K - 1]   # 5 early MLPs
    r = ab.null_sets(arm, dec, np.random.default_rng(0))
    assert len(r["sets"]) == ab.N_NULL and len({frozenset(s) for s in r["sets"]}) == ab.N_NULL
    for s in r["sets"]:
        assert len(s) == len(arm) and not set(s) & set(arm)
        assert all(ab.ctype(j) == "mlp" and dec[j] == dec[arm[0]] for j in s)
    assert any(b != 0 for bs in r["bands"] for b in bs)        # bands not matched for MLPs
    assert r["widened"] == 0


def test_null_keeps_the_band_for_heads_and_refuses_when_too_few():
    dec = ab.deciles(_sizes())
    P = ab.pool(3, dec, {3}, band=True)
    assert P and all(ab.cband(j) == 0 and ab.ctype(j) == "head" and dec[j] == dec[3] for j in P)
    dec2 = np.zeros(24 * K, dtype=int)                       # one decile: 383 heads to draw from
    r = ab.null_sets([3], dec2, np.random.default_rng(1))
    assert not r["heads_any_band"] and all(ab.cband(x[0]) == 0 and ab.ctype(x[0]) == "head" for x in r["sets"])
    dec3 = np.zeros(24 * K, dtype=int)
    dec3[[j for j in range(24 * K) if ab.ctype(j) == "head" and ab.cband(j) == 0][10:]] = 1
    r = ab.null_sets([3], dec3, np.random.default_rng(1))    # 9 band-matched heads: the band dropped
    assert r["heads_any_band"] and all(ab.ctype(x[0]) == "head" and dec3[x[0]] == 0 for x in r["sets"])
    with pytest.raises(SystemExit):
        ab.null_sets(list(range(K - 1, 24 * K, K))[:20], dec, np.random.default_rng(2))   # 20 of 24 MLPs


def test_arms():
    proj = np.zeros(24 * K)
    proj[[3, 40, 77]] = [-5.0, -4.0, -1.0]
    proj[[5, 6, 7, 8]] = [3.0, 2.0, 1.0, 0.5]
    off = np.zeros(24 * K)
    off[[10, 11]] = [0.6, 0.4]
    a = ab.arms_from(proj, off)
    assert a["S_minus"] == [3] and a["S_plus"] == [5] and a["S_zero"] == [10]
    proj[[3, 40]] = [-2.0, -2.0]
    assert ab.arms_from(proj, off)["S_plus"] == [5, 6]


def _rows(cos, rise):
    return {"prompts": [{"cos": cos, "rise": [rise] * 24}]}


def test_step_rule_and_reading():
    res = {"none": _rows(0.30, False), "S_minus": _rows(0.50, True), "S_plus": _rows(0.20, False),
           "S_minus_null": [_rows(0.40, False) for _ in range(20)]}
    r = ab.step_rule(res)
    assert r["cos_pass"] and r["rise_pass"] and r["both"]
    res["S_minus_null"][7] = _rows(0.55, False)                # one null above on (i)
    r2 = ab.step_rule(res)
    assert not r2["cos_pass"] and r2["rise_pass"]
    assert ab.reading({0: r, 1: r, 2: r2})["verdict"] == "supported"
    assert ab.reading({0: r, 1: r2, 2: r2})["verdict"] == "mixed: (iii) monotone energy"
    no = dict(r2, rise_pass=False, both=False)
    assert ab.reading({0: r, 1: no, 2: no})["verdict"] == "refused"


def test_energy_rises_with_the_gram_for_either_sign():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float64"):
        pytest.skip("stub torch")
    rng = np.random.default_rng(3)
    U = torch.tensor(rng.standard_normal((30, 8)))
    U = U / U.norm(dim=-1, keepdim=True)
    V = U + 0.5 * U.mean(0)
    V = V / V.norm(dim=-1, keepdim=True)                        # pulled together
    for beta in (-2.0, 0.0, 0.7, 3.5):
        assert ab.energy(V, beta) > ab.energy(U, beta)
