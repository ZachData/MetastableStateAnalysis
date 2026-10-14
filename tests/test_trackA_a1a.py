"""A1a's pure parts: the step lifts, the frame, the bracket, the totals, the OV row (`trackA_force/design-A.md` "A1a")."""

import json
import math

import numpy as np
import pytest

from p1c_frames.integration_time import unit_steps
from trackA_force import a1a
from trackA_force.a1a_gate import start

pytestmark = pytest.mark.deps  # p1c_frames (scipy)


def _tangent(U, rng):
    Z = rng.standard_normal(U.shape)
    return Z - np.sum(Z * U, axis=1, keepdims=True) * U


def _setup(seed=0, n=5, d=64, h=0.3):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    X = _tangent(U, rng) * 0.2
    Z = _tangent(U, rng)
    P = X / np.linalg.norm(X, axis=1, keepdims=True)
    xi = Z - np.sum(Z * P, axis=1, keepdims=True) * P
    xi *= 0.05 / np.linalg.norm(xi, axis=1, keepdims=True)
    return U, X, xi, h


def test_log_lift_exact_on_a_geodesic():
    U, X, xi, h = _setup()
    V = h * X + xi
    th = np.linalg.norm(V, axis=1, keepdims=True)
    U1 = np.cos(th) * U + np.sin(th) * V / th
    s = unit_steps(U, U1, X, lift="log")
    xn = np.linalg.norm(X, axis=1)
    assert np.allclose(s["proj"], h, rtol=1e-10)
    assert np.allclose(s["cal"], np.sqrt(h ** 2 + 0.05 ** 2 / xn ** 2), rtol=1e-10)


def test_tangent_lift_on_a_retraction_is_v_over_sqrt():
    U, X, xi, h = _setup(1)
    V = h * X + xi
    U1 = (U + V) / np.linalg.norm(U + V, axis=1, keepdims=True)
    s = unit_steps(U, U1, X)
    assert np.allclose(s["proj"], h / np.sqrt(1 + np.sum(V * V, axis=1)), rtol=1e-10)


def test_zero_field_is_nan_and_unknown_lift_refused():
    U, X, _, _ = _setup(2)
    X[0] = 0.0
    s = unit_steps(U, U, X)
    assert math.isnan(s["proj"][0]) and math.isnan(s["cal"][0]) and s["disp"][0] == 0.0
    with pytest.raises(ValueError):
        unit_steps(U, U, X, lift="chord")


def test_start_has_the_planted_cosine():
    for g0 in (0.0, 0.3):
        U = start(40, g0, np.random.default_rng(0))
        assert np.allclose(np.linalg.norm(U, axis=1), 1.0)
        assert a1a.mean_cos(U) == pytest.approx(g0, abs=1e-12)


def test_affine_free_frame_is_the_centred_unit_row():
    X = np.random.default_rng(3).standard_normal((4, 16)) + 2.0
    U = a1a.frame(X, np.ones(16), np.zeros(16), 0.0)
    C = X - X.mean(axis=1, keepdims=True)
    assert np.allclose(U, C / np.linalg.norm(C, axis=1, keepdims=True))


def test_bracket_reads_blocks_0_to_22_and_refuses_nan():
    lb = np.arange(24, dtype=float)
    lb[23] = 1e6                                    # block 23 is beside, never in the bracket
    br = a1a.bracket(lb)
    assert br["p90"] == pytest.approx(np.percentile(np.arange(23), 90))
    lb[4] = np.nan
    with pytest.raises(ValueError):
        a1a.bracket(lb)


def test_totals_leave_block_23_out():
    rows = [{"block": L, **{k: 1.0 for k in a1a.KINDS}} for L in range(24)]
    assert a1a.totals(rows)["T_proj"] == 23.0


def test_ov_share(tmp_path):
    per = {f"{L},{h}": {"attractive_energy_fraction_core": [0.0 if h < 2 else 0.5, 0.25]}
           for L in range(2) for h in range(4)}
    p = tmp_path / "ov.json"
    p.write_text(json.dumps({"steps": [0, 1000], "n_heads": 4, "n_blocks": 2, "per_head": per}))
    o = a1a.ov_share(p)
    assert o[0]["median"] == [0.25, 0.25] and o[0]["frac_zero"] == [0.5, 0.5]
    assert o[1000]["median"] == [0.25, 0.25] and o[1000]["frac_zero"] == [0.0, 0.0]


def test_batch_refuses_a_failing_or_foreign_gate(tmp_path):
    p = tmp_path / "g.json"
    from trackA_force.a1a_gate import gate_sha
    ok = {"code": "c", "gate_code": gate_sha(), "lift": a1a.LIFT, "passes": True}
    for bad in ({"passes": False}, {"code": "other"}, {"lift": "chord"}, {"gate_code": "older"}):
        p.write_text(json.dumps({**ok, **bad}))
        with pytest.raises(SystemExit):
            a1a.check_gate(p, "c")
    p.write_text(json.dumps(ok))
    a1a.check_gate(p, "c")


def test_gnomonic_lift_inverts_one_add_and_renormalise_step():
    U, X, xi, h = _setup()
    V = h * X + xi
    U1 = (U + V) / np.linalg.norm(U + V, axis=1, keepdims=True)
    s = unit_steps(U, U1, X, lift="gnomonic")
    assert np.allclose(s["disp"], np.linalg.norm(V, axis=1), rtol=1e-12)
    assert np.allclose(s["proj"], h, rtol=1e-12)


def test_ln_plant_in_an_affine_free_frame_is_g1():
    """G4's plant with w = 1, b = 0 reads exactly what G1 reads from the centred start."""
    from trackA_force.a1a_gate import D, ln_plant, planted_h, read, trajectory
    rng = np.random.default_rng(3)
    U0 = start(40, 0.0, rng)
    h = planted_h(1)[:5]
    g4 = ln_plant(U0, h, 3.5, {"w": np.ones(D), "b": np.zeros(D), "eps": 0.0})
    C = U0 - U0.mean(axis=1, keepdims=True)
    g1 = read(trajectory(C / np.linalg.norm(C, axis=1, keepdims=True), h, 3.5, "field", rng), 3.5)
    for l in ("log", "tangent", "gnomonic"):
        assert np.allclose(g4[l]["proj"], g1[l]["proj"], rtol=1e-9)
    assert np.allclose(g4["gnomonic"]["proj"], h, rtol=1e-9)


def test_block_steps_recovers_a_planted_step_through_the_real_path():
    """`block_steps` (the batch's path): position 0 dropped, each block's own β, the LN frame."""
    from p1c_frames.integration_time import sa_field
    rng = np.random.default_rng(5)
    n, d, h, beta = 7, 32, 0.01, 2.0
    hs = rng.standard_normal((a1a.N_BLOCKS + 1, n, d))
    hs -= hs.mean(axis=2, keepdims=True)              # centred rows: unit LN with w=1, b=0 is unit(x)
    hs *= 5.0 / np.linalg.norm(hs, axis=2, keepdims=True)
    moves = {}
    for L in range(a1a.N_BLOCKS):
        U = hs[L][1:] / 5.0
        V = np.zeros_like(hs[L])
        V[1:] = 5.0 * h * sa_field(U, beta, causal=True)
        moves[L] = {"block": V, "attn": 0.5 * V}
    ln = {"w": np.ones((a1a.N_BLOCKS, d)), "b": np.zeros((a1a.N_BLOCKS, d)), "eps": 0.0}
    rows = a1a.block_steps(hs, moves, ln, np.full(a1a.N_BLOCKS, beta))
    for r in rows:
        assert r["proj"] == pytest.approx(h, rel=1e-3) and r["cal"] == pytest.approx(h, rel=1e-3)
        assert r["attn_proj"] == pytest.approx(h / 2, rel=1e-3)
    assert a1a.totals(rows)["T_proj"] == pytest.approx(23 * h, rel=1e-3)
