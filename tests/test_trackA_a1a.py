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


def test_drift_free_slope_is_exact_for_a_field_step_and_blind_to_a_constant_drift():
    """Gnomonic inverts add-and-renormalise of a tangent V; V = hX + c with c ⊥ every row is tangent."""
    from p1c_frames.integration_time import drift_free_slope
    U, X, xi, h = _setup()
    Q, _ = np.linalg.qr(U.T)
    c = np.random.default_rng(1).standard_normal(U.shape[1])
    c -= Q @ (Q.T @ c)                                   # ⊥ span(U): tangent at every row
    c *= 0.2 / np.linalg.norm(c)
    for V, want in ((h * X, h), (h * X + c, h), (np.tile(c, (U.shape[0], 1)), 0.0)):
        U1 = (U + V) / np.linalg.norm(U + V, axis=1, keepdims=True)
        assert drift_free_slope(U, U1, X, lift="gnomonic") == pytest.approx(want, abs=1e-9)
    X0 = X.copy()
    X0[0] = 0.0                                          # a row without field is left out, as in unit_steps
    U1 = (U + h * X0) / np.linalg.norm(U + h * X0, axis=1, keepdims=True)
    assert drift_free_slope(U, U1, X0, lift="log") == pytest.approx(
        drift_free_slope(U[1:], U1[1:], X0[1:], lift="log"), abs=1e-15)


def test_ln_chain_torch_matches_numpy_and_affine_free_chain_recovers_h():
    from trackA_force.a1a_gate import D, ln_chain, ln_chain_np, planted_h
    rng = np.random.default_rng(7)
    U0 = start(30, 0.0, rng)
    h = planted_h(2, 4)
    w = 1.0 + 0.3 * rng.standard_normal((4, D))
    b = 0.1 * rng.standard_normal((4, D))
    ref = ln_chain_np(U0, h, 3.5, w, b, 1e-5)
    got = ln_chain(U0, h, 3.5, np.stack([w, w]), np.stack([b, b]), 1e-5, "cpu")
    for g in got:
        for l in ("log", "tangent", "gnomonic"):
            for r in ("proj", "proj_c"):
                assert np.allclose(g[l][r], ref[l][r], rtol=0, atol=1e-9)
        assert np.allclose(g["frame_g"], ref["frame_g"], atol=1e-12)
    af = ln_chain_np(U0, h, 3.5, np.ones((4, D)), np.zeros((4, D)), 0.0)
    assert np.allclose(af["gnomonic"]["proj"], h, rtol=1e-9)
    assert np.allclose(af["gnomonic"]["proj_c"], h, rtol=1e-9)


def test_block_steps_records_the_drift_free_read():
    from p1c_frames.integration_time import sa_field
    rng = np.random.default_rng(5)
    n, d, h, beta = 9, 32, 0.01, 2.0
    hs = rng.standard_normal((a1a.N_BLOCKS + 1, n, d))
    hs -= hs.mean(axis=2, keepdims=True)
    hs *= 5.0 / np.linalg.norm(hs, axis=2, keepdims=True)
    moves = {}
    for L in range(a1a.N_BLOCKS):
        V = np.zeros_like(hs[L])
        V[1:] = 5.0 * h * sa_field(hs[L][1:] / 5.0, beta, causal=True)
        moves[L] = {"block": V, "attn": 0.5 * V}
    rows = a1a.block_steps(hs, moves, {"w": np.ones((a1a.N_BLOCKS, d)), "b": np.zeros((a1a.N_BLOCKS, d)),
                                       "eps": 0.0}, np.full(a1a.N_BLOCKS, beta))
    for r in rows:
        assert r["proj_c"] == pytest.approx(h, rel=1e-3) and r["attn_proj_c"] == pytest.approx(h / 2, rel=1e-3)
    assert "T_proj_c" in a1a.totals(rows)


def test_batch_refuses_a_gate_for_other_ln_weights(tmp_path):
    from trackA_force.a1a_gate import gate_sha
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"code": "c", "gate_code": gate_sha(), "lift": a1a.LIFT, "passes": True,
                             "ln_sha": "aaa"}))
    with pytest.raises(SystemExit):
        a1a.check_gate(p, "c", "bbb")
    a1a.check_gate(p, "c", "aaa")
    w = np.ones((len(a1a.STEPS), a1a.N_BLOCKS, 4))
    cache = {"w": w, "b": 0 * w, "eps": 1e-5}
    a1a.check_ln({"w": w[0], "b": 0 * w[0], "eps": 1e-5}, cache, 0)
    with pytest.raises(SystemExit):
        a1a.check_ln({"w": 2 * w[0], "b": 0 * w[0], "eps": 1e-5}, cache, 0)


def test_ln_from_reads_the_bare_model_and_the_lm_alike():
    """`core.models.load_model` returns the bare GPTNeoXModel; the gate's cache the LM (#205's crash)."""
    torch = pytest.importorskip("torch")

    class LN(torch.nn.LayerNorm):
        pass

    class Layer(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.input_layernorm = LN(4, eps=1e-5)
            torch.nn.init.constant_(self.input_layernorm.weight, k)

    bare = torch.nn.Module()
    bare.layers = torch.nn.ModuleList([Layer(1.0), Layer(2.0)])
    lm = torch.nn.Module()
    lm.gpt_neox = bare
    for m in (bare, lm):
        ln = a1a.ln_from(m)
        assert ln["w"].shape == (2, 4) and ln["w"][1, 0] == 2.0 and ln["eps"] == 1e-5
