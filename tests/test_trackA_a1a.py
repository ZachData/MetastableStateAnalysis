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
    for g in ({"code": "c", "lift": a1a.LIFT, "passes": False},
              {"code": "other", "lift": a1a.LIFT, "passes": True},
              {"code": "c", "lift": "chord", "passes": True}):
        p.write_text(json.dumps(g))
        with pytest.raises(SystemExit):
            a1a.check_gate(p, "c")
    p.write_text(json.dumps({"code": "c", "lift": a1a.LIFT, "passes": True}))
    a1a.check_gate(p, "c")
