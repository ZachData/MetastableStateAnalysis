"""
tests/test_phase1d_position_null.py — the position-keeping null
(`p1d_cluster_ensemble/position_null.py`, Blocked 11‴): the synthetic opening
that rejected the committed form (`status-1d.md` "Blocked 11″ decided"), now
a test of the revised one, plus the null's algebra and the driver's raw rows.

The synthetic is #123's mechanism with nothing else in it:
``Y_t = E_t + 6 mean(V_0..V_t)``, n 150, d 300, ``E``, ``V`` iid N(0, I).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble import position_null as pn
from p1d_cluster_ensemble.admit import _job, admit_record
from p1d_cluster_ensemble.gaussian_null import _unit_rows, frame_vectors, gaussian_draw

N, D, AMP, SEEDS, DRAWS = 150, 300, 6.0, range(10), 39


def _opening(seed, amp=AMP):
    rng = np.random.default_rng(seed)
    V, E = rng.standard_normal((N, D)), rng.standard_normal((N, D))
    return E + amp * np.cumsum(V, axis=0) / np.arange(1, N + 1)[:, None]


def _admits(Y, frame, null, seed, holding_0=True):
    rec = admit_record(Y, frame, DRAWS, seed, (2,), null_kind=null, positions=np.arange(N))
    return any(g["admitted_excess"] and (0 in g["members"] or not holding_0)
               for g in rec["arms"]["2"]["groups"])


class TestSyntheticOpening:
    """Records (of 10 seeds) admitting a group that holds position 0."""

    def test_the_gaussian_admits_the_opening(self):
        # #122's null (unit rows): the failure this null exists to remove.
        # Observed 7 / 10 raw, 10 / 10 centred; this is what makes the next
        # test able to fail.
        assert sum(_admits(_opening(s), "raw", "gaussian", s) for s in SEEDS) >= 5

    @pytest.mark.parametrize("frame", ["raw", "centred"])
    def test_smooth_null_does_not(self, frame):
        # Fitted and drawn before normalisation (11‴). Observed 0 / 10 in both
        # frames; on unit rows even the true mean admitted 7 / 10.
        assert sum(_admits(_opening(s), frame, "smooth", s) for s in SEEDS) <= 1

    @pytest.mark.parametrize("null", ["smooth", "prefix"])
    def test_pure_noise_admits_nothing(self, null):
        assert sum(_admits(_opening(s, 0.0), "raw", null, s, holding_0=False)
                   for s in SEEDS) <= 1

    def test_prefix_under_fits(self):
        # The arm: one scalar c, fitted mostly where there is no opening.
        # Recorded, not required: observed 2 / 10 raw, 3 / 10 centred.
        _, _, info = pn.fit(_opening(0), np.arange(N), "prefix")
        assert 0.0 < info["c"] < 1.0


class TestAlgebra:
    def test_gaussian_kind_is_gaussian_draw_unnormalised(self):
        Y = _opening(1)[:40]
        mu, R, _ = pn.fit(Y, np.arange(40), "gaussian")
        a = pn.draw(mu, R, np.random.default_rng(5))
        b = gaussian_draw(Y, np.random.default_rng(5), renorm=False)
        np.testing.assert_allclose(a, b, atol=1e-12)

    @pytest.mark.parametrize("frame", ["raw", "centred"])
    def test_apply_frame_is_frame_vectors(self, frame):
        X = _opening(2)[:50] + 3.0
        np.testing.assert_allclose(pn.apply_frame(X, frame), frame_vectors(X, frame)[0],
                                   atol=1e-12)

    @pytest.mark.parametrize("null", ["smooth", "prefix"])
    def test_residual_centred_and_rows_kept(self, null):
        Y = _opening(3)
        mu, R, info = pn.fit(Y, np.arange(N), null)
        np.testing.assert_allclose(R.mean(axis=0), 0.0, atol=1e-10)
        assert mu.shape == Y.shape and 0.0 <= info["resid_top_share"] <= 1.0

    def test_smooth_picks_no_position_on_noise(self):
        # CV's grid ends at "no position"; on pure noise a wide kernel or
        # none at all wins.
        _, info = pn.fit_smooth(_opening(4, 0.0), np.arange(N))
        assert info["h"] >= 1.0

    def test_smooth_finds_the_opening(self):
        _, info = pn.fit_smooth(_opening(4), np.arange(N))
        assert info["h"] < 1.0

    def test_one_massive_row_owns_the_noise(self):
        # The trained-layer confound: token 0's massive activation becomes
        # most of every draw's noise; `resid_top_share` reports it.
        Y = _opening(5, 0.0)
        Y[0] *= 40.0
        _, _, info = pn.fit(Y, np.arange(N), "smooth")
        assert info["resid_top_row"] == 0 and info["resid_top_share"] > 0.5

    def test_refuses_unordered_positions(self):
        with pytest.raises(ValueError):
            pn.fit(_opening(0)[:5], [0, 2, 1, 3, 4], "smooth")

    def test_calibration_refits_to_its_own_draw(self):
        Y = _opening(6)
        kw = dict(min_cluster_sizes=(2,), null_kind="smooth", positions=np.arange(N))
        a = admit_record(Y, "raw", 3, 0, calibrate=True, **kw)
        b = admit_record(Y, "raw", 3, 0, calibrate=False, **kw)
        assert a["info"]["calibrate"] and "calibrate" not in b["info"]
        assert a["info"]["position_null"]["r2"] != b["info"]["position_null"]["r2"]
        assert a == admit_record(Y, "raw", 3, 0, calibrate=True, **kw)


def test_driver_fits_on_raw_rows(tmp_path):
    # Stored activations are unit rows; raw = norms * activations (`p1_io`).
    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step0_wiki_paragraph"
    d.mkdir(parents=True)
    Y = _opening(8)[:60]
    norms = np.linalg.norm(Y, axis=1)
    np.savez(d / "activations.npz", activations=_unit_rows(Y)[None].repeat(2, 0).astype(np.float32),
             norms=norms[None].repeat(2, 0).astype(np.float32))
    tokens = [f"t{i}" for i in range(60)]
    tokens[7] = tokens[2]
    (d / "geometry.json").write_text(json.dumps({"tokens": tokens}))
    rec = _job((str(d), 1, "raw", 3, 0, False, 0, "smooth"))
    keep = np.array(rec["keep"])
    raw = (_unit_rows(Y).astype(np.float32).astype(np.float64)
           * norms.astype(np.float32).astype(np.float64)[:, None])[keep]
    want = admit_record(raw, "raw", 3, 0, null_kind="smooth", positions=keep)
    assert 7 not in rec["keep"]
    assert rec["info"]["position_null"] == want["info"]["position_null"]
    assert rec["arms"]["2"]["null"]["excess"] == want["arms"]["2"]["null"]["excess"]
