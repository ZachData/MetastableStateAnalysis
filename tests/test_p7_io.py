"""
tests/test_p7_io.py — p7_motifs/p7_io.py's reading of Phase 2 / 2b
projectors, and its artifact writers.

The producer/consumer seam is where this project's recurring bug class
lives (Phase 5's blockers 2 and 3 were both instances), so these tests are
written against artifacts built the way Phase 2 actually builds them —
`P = Z @ Z.T`, keyed `schur_attract_{layer}` — rather than against a
convenient stand-in.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.interactions import projection_fractions
from p7_motifs.p7_io import (
    SIGN_CHANNEL_CHOICES,
    load_sign_channel,
    rotational_channel_from_blocks,
    write_formation_curve,
    write_motif_counts,
)


def _write_phase2_projectors(tmp_path, model="pythia-1.4b", d=8, layers=("layer_0", "layer_1")):
    """Build ov_projectors_{stem}.npz exactly as p2_eigenspectra/weights.py
    does: orthogonal projectors P = Z @ Z.T, per-layer keys."""
    rng = np.random.default_rng(0)
    arrays = {}
    for i, name in enumerate(layers):
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Z_a, Z_r = Q[:, :3], Q[:, 3:]
        arrays[f"schur_attract_{name}"] = Z_a @ Z_a.T
        arrays[f"schur_repulse_{name}"] = Z_r @ Z_r.T
        arrays[f"sym_attract_{name}"] = Z_a @ Z_a.T
        arrays[f"sym_repulse_{name}"] = Z_r @ Z_r.T
    path = tmp_path / f"ov_projectors_{model.replace('/', '_')}.npz"
    np.savez_compressed(path, **arrays)
    return tmp_path, model


class TestLoadSignChannel:

    def test_loads_a_named_layer(self, tmp_path):
        d, model = 8, "pythia-1.4b"
        wd, _ = _write_phase2_projectors(tmp_path, model, d=d)
        res = load_sign_channel(wd, model, sign_channel="schur", layer_name="layer_0")
        assert res["U_pos"].shape == (d, d)
        assert res["provenance"]["layer_name"] == "layer_0"

    def test_attractive_and_repulsive_fractions_sum_to_one(self, tmp_path):
        d, model = 8, "pythia-1.4b"
        wd, _ = _write_phase2_projectors(tmp_path, model, d=d)
        res = load_sign_channel(wd, model, sign_channel="schur", layer_name="layer_0")
        f = np.random.default_rng(1).standard_normal((20, d))
        a = projection_fractions(f, res["U_pos"])
        r = projection_fractions(f, res["U_neg"])
        assert np.allclose(a + r, 1.0)
        assert 0.05 < a.mean() < 0.95      # a genuine split, not 0/1

    def test_sign_channel_has_no_default(self, tmp_path):
        """schur and sym answer different questions — Phase 2b's finding is
        that the symmetric part carries all violation causality — so the
        choice must be made explicitly and cannot be omitted."""
        wd, model = _write_phase2_projectors(tmp_path)
        with pytest.raises(TypeError):
            load_sign_channel(wd, model, layer_name="layer_0")

    @pytest.mark.parametrize("bad", ["Schur", "eig", "", None])
    def test_unknown_sign_channel_is_refused(self, tmp_path, bad):
        wd, model = _write_phase2_projectors(tmp_path)
        with pytest.raises(ValueError, match="sign_channel must be one of"):
            load_sign_channel(wd, model, sign_channel=bad, layer_name="layer_0")

    @pytest.mark.parametrize("channel", SIGN_CHANNEL_CHOICES)
    def test_both_channels_load_and_stamp_their_choice(self, tmp_path, channel):
        wd, model = _write_phase2_projectors(tmp_path)
        res = load_sign_channel(wd, model, sign_channel=channel, layer_name="layer_0")
        assert res["provenance"]["sign_channel"] == channel

    def test_missing_layer_refuses_rather_than_falling_back_to_shared(self, tmp_path):
        """The dangerous fallback: applying one layer's geometry to
        another's activations produces numbers that look fine."""
        wd, model = _write_phase2_projectors(tmp_path)
        with pytest.raises(KeyError, match="Refusing to substitute"):
            load_sign_channel(wd, model, sign_channel="schur", layer_name="layer_99")

    def test_missing_artifact_names_the_producer(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run Phase 2"):
            load_sign_channel(tmp_path, "no-such-model",
                              sign_channel="schur", layer_name="layer_0")

    def test_shared_key_selected_when_no_layer_given(self, tmp_path):
        d = 6
        rng = np.random.default_rng(2)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        np.savez_compressed(
            tmp_path / "ov_projectors_m.npz",
            schur_attract_shared=Q[:, :2] @ Q[:, :2].T,
            schur_repulse_shared=Q[:, 2:] @ Q[:, 2:].T,
        )
        res = load_sign_channel(tmp_path, "m", sign_channel="schur")
        assert res["provenance"]["layer_name"] == "shared"


class TestRotationalChannel:
    """p2b returns a LIST of (d, 2) plane bases, never a (d, d) projector.
    The adapter must preserve that — materializing the projector is the
    ~7 GB mistake top_rotation_planes was written to avoid."""

    def _blocks(self, d=8, n_complex=2):
        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        return {
            "blocks_2x2": [
                {"plane": Q[:, 2 * i:2 * i + 2], "rho": 1.0 - 0.1 * i,
                 "theta": 0.3, "sign": 1, "idx": i}
                for i in range(n_complex)
            ],
            "n_complex": n_complex,
            "n_real": d - 2 * n_complex,
            "d": d,
        }

    def test_returns_a_list_of_planes_not_a_projector(self):
        res = rotational_channel_from_blocks(self._blocks())
        assert isinstance(res["U_A"], list)
        assert all(b.shape[1] == 2 for b in res["U_A"])

    def test_planes_feed_projection_fractions_directly(self):
        res = rotational_channel_from_blocks(self._blocks())
        f = np.random.default_rng(4).standard_normal((10, 8))
        frac = projection_fractions(f, res["U_A"])
        assert np.all((frac >= 0) & (frac <= 1))

    def test_untruncated_basis_is_not_flagged_as_a_lower_bound(self):
        res = rotational_channel_from_blocks(self._blocks(n_complex=2), top_k=32)
        assert res["provenance"]["truncated"] is False
        assert res["provenance"]["imag_frac_is_lower_bound"] is False

    def test_truncation_is_recorded_as_making_imag_frac_a_lower_bound(self):
        """Silently truncating would make imag_frac look like the full
        rotational fraction when it is only part of it."""
        res = rotational_channel_from_blocks(self._blocks(n_complex=3), top_k=1)
        assert res["provenance"]["truncated"] is True
        assert res["provenance"]["imag_frac_is_lower_bound"] is True
        assert res["provenance"]["n_planes_used"] == 1
        assert res["provenance"]["n_planes_available"] == 3

    def test_real_fraction_is_the_complement_of_the_untruncated_imaginary(self):
        """The identity the adapter relies on instead of storing a second
        projector: rotation planes and real Schur directions are orthogonal
        by construction, so real_frac = 1 - imag_frac exactly."""
        blocks = self._blocks(d=8, n_complex=2)
        res = rotational_channel_from_blocks(blocks, top_k=32)
        f = np.random.default_rng(5).standard_normal((12, 8))
        imag = projection_fractions(f, res["U_A"])
        # the real complement, built here only to check the identity
        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
        real = projection_fractions(f, Q[:, 4:])
        assert np.allclose(imag + real, 1.0)


class TestWriters:

    def _counts(self):
        from p7_motifs.motif_stats import motif_counts_payload
        return motif_counts_payload({}, {}, {}, [], None)

    def test_motif_counts_roundtrips_as_json(self, tmp_path):
        p = write_motif_counts(self._counts(), tmp_path)
        assert json.loads(p.read_text())["motif_alphabet_version"]

    def test_motif_counts_refuses_an_incomplete_payload(self, tmp_path):
        with pytest.raises(ValueError, match="missing required keys"):
            write_motif_counts({"counts": {}}, tmp_path)

    def test_formation_curve_requires_independence_source(self, tmp_path):
        """A curve of motif strength against behavioural score that cannot
        say what makes them independent has plotted one thing twice."""
        payload = {
            "checkpoint_steps": [0, 1000],
            "motif_strength": [0.0, 0.4],
            "behavioral_induction_score": [0.0, 0.5],
        }
        with pytest.raises(ValueError, match="independence_source"):
            write_formation_curve(payload, tmp_path)

    def test_formation_curve_writes_when_complete(self, tmp_path):
        payload = {
            "checkpoint_steps": [0, 1000],
            "motif_strength": [0.0, 0.4],
            "behavioral_induction_score": [0.0, 0.5],
            "independence_source": "two_stage",
        }
        p = write_formation_curve(payload, tmp_path)
        assert json.loads(p.read_text())["independence_source"] == "two_stage"

    def test_numpy_scalars_serialize(self, tmp_path):
        """Every number in this pipeline arrives as a numpy scalar; a
        writer that chokes on them fails only at the end of a long run."""
        payload = {
            "checkpoint_steps": np.array([0, 1000]),
            "motif_strength": [np.float64(0.4)],
            "behavioral_induction_score": [np.float32(0.5)],
            "independence_source": "force_channel",
        }
        p = write_formation_curve(payload, tmp_path)
        assert json.loads(p.read_text())["checkpoint_steps"] == [0, 1000]
