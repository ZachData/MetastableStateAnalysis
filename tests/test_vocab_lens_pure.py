"""
tests/test_vocab_lens_pure.py — pure-numpy tests for
p2_eigenspectra/vocab_projection.py's projection math and
p2_eigenspectra/lens_band.py's band statistics.

Oracles are constructed cases with known answers: a planted unembedding
row must rank first under its own direction; a peaked logit row must
have high excess kurtosis and a Gaussian row ~0; a constant top-1
sequence must show positive null-corrected autocorrelation and an iid
one ~0; a synthetic three-regime depth profile must yield the planted
onset/motor layers.

No torch anywhere on these paths (project smoke convention:
torch-touching work lives in a separate SMOKE_REAL_DEPS file).
"""

import json
import math

import numpy as np
import pytest

from p2_eigenspectra.vocab_projection import (
    project_directions, decode_scores, _select_sym_directions,
    label_ov_directions, save_unembedding, load_unembedding,
)
from p2_eigenspectra.lens_band import (
    numpy_layernorm, excess_kurtosis_rows, top1_autocorr,
    compute_lens_band, detect_band, LENS_BAND_FILENAME,
    _s_band_onset_frac,
)


def _unemb(vocab=32, d=8, seed=0, with_gain=False):
    rng = np.random.default_rng(seed)
    W_U = rng.standard_normal((vocab, d)).astype(np.float32) * 0.1
    return {
        "W_U": W_U,
        "ln_gamma": (rng.uniform(0.5, 2.0, d).astype(np.float32)
                     if with_gain else None),
        "ln_beta": None,
        "head_attr": "embed_out",
        "vocab_size": vocab,
        "d_model": d,
    }


# ---------------------------------------------------------------------------
# vocab_projection — projection orientation and sign poles
# ---------------------------------------------------------------------------

class TestProjection:
    def test_planted_row_ranks_first(self):
        unemb = _unemb()
        v = np.zeros(8, dtype=np.float32)
        # make token 7's row exactly aligned with v
        unemb["W_U"][7] = 0.0
        unemb["W_U"][7, 3] = 5.0
        v[3] = 1.0
        scores = project_directions(v, unemb, apply_ln_gain=False)
        assert scores.shape == (32,)
        assert int(np.argmax(scores)) == 7

    def test_negated_direction_swaps_poles(self):
        unemb = _unemb(seed=1)
        v = np.random.default_rng(2).standard_normal(8).astype(np.float32)
        pos = decode_scores(project_directions(v, unemb, apply_ln_gain=False))
        neg = decode_scores(project_directions(-v, unemb, apply_ln_gain=False))
        assert [e["id"] for e in pos["pole_pos"]] == [e["id"] for e in neg["pole_neg"]]
        assert [e["id"] for e in pos["pole_neg"]] == [e["id"] for e in neg["pole_pos"]]

    def test_ln_gain_changes_ranking(self):
        unemb = _unemb(seed=3, with_gain=True)
        # gain concentrated on coordinate 0 flips which row wins
        unemb["W_U"][:] = 0.0
        unemb["W_U"][0, 0] = 1.0   # token 0 reads coord 0
        unemb["W_U"][1, 1] = 1.0   # token 1 reads coord 1
        unemb["ln_gamma"] = np.array([10.0, 1.0] + [1.0] * 6, dtype=np.float32)
        v = np.zeros(8, dtype=np.float32)
        v[0], v[1] = 1.0, 2.0      # raw: token 1 wins; with gain: token 0
        raw = project_directions(v, unemb, apply_ln_gain=False)
        gained = project_directions(v, unemb, apply_ln_gain=True)
        assert int(np.argmax(raw)) == 1
        assert int(np.argmax(gained)) == 0

    def test_unit_normalisation_scale_invariance(self):
        unemb = _unemb(seed=4)
        v = np.random.default_rng(5).standard_normal(8).astype(np.float32)
        s1 = project_directions(v, unemb, apply_ln_gain=False)
        s2 = project_directions(100.0 * v, unemb, apply_ln_gain=False)
        np.testing.assert_allclose(s1, s2, rtol=1e-5, atol=1e-5)

    def test_dim_mismatch_raises(self):
        unemb = _unemb()
        with pytest.raises(ValueError):
            project_directions(np.zeros(9, dtype=np.float32), unemb)

    def test_decode_without_tokenizer_gives_ids(self):
        out = decode_scores(np.arange(10, dtype=np.float64), top_k=3)
        assert [e["id"] for e in out["pole_pos"]] == [9, 8, 7]
        assert [e["id"] for e in out["pole_neg"]] == [0, 1, 2]
        assert all(e["token"] is None for e in out["pole_pos"])


class TestSymSelection:
    def _decomp(self, vals):
        vals = np.asarray(sorted(vals), dtype=np.float64)  # eigh: ascending
        d = len(vals)
        return {"sym_eigenvalues": vals, "sym_eigenvectors": np.eye(d, dtype=np.float32)}

    def test_picks_signed_extremes(self):
        picked = _select_sym_directions(self._decomp([-3, -1, 0.0, 2, 5]), 2)
        kinds_vals = [(k, v) for (k, _, v, _) in picked]
        assert ("repulsive", -3.0) == kinds_vals[0]
        assert ("repulsive", -1.0) == kinds_vals[1]
        assert ("attractive", 5.0) == kinds_vals[2]
        assert ("attractive", 2.0) == kinds_vals[3]

    def test_zero_eigenvalues_excluded(self):
        picked = _select_sym_directions(self._decomp([0.0, 0.0, 0.0]), 2)
        assert picked == []

    def test_label_ov_directions_shared_layout(self):
        unemb = _unemb()
        d = unemb["d_model"]
        vals = np.linspace(-1, 1, d)
        loaded = {
            "summary": {"is_per_layer": False, "layers": {"shared": {}}},
            "decomp": {"sym_eigenvalues": vals,
                       "sym_eigenvectors": np.eye(d, dtype=np.float32)},
        }
        result = label_ov_directions(loaded, unemb, n_directions=2, top_k=3,
                                     apply_ln_gain=False)
        assert set(result["layers"].keys()) == {"shared"}
        entries = result["layers"]["shared"]
        assert len(entries) == 4
        assert {e["kind"] for e in entries} == {"repulsive", "attractive"}
        assert all("pole_pos" in e and "pole_neg" in e for e in entries)

    def test_label_ov_directions_d_mismatch_raises(self):
        unemb = _unemb()
        loaded = {
            "summary": {"is_per_layer": False, "layers": {"shared": {}}},
            "decomp": {"sym_eigenvalues": np.array([-1.0, 1.0]),
                       "sym_eigenvectors": np.eye(2, dtype=np.float32)},
        }
        with pytest.raises(ValueError):
            label_ov_directions(loaded, unemb)


class TestUnembeddingRoundtrip:
    def test_npz_roundtrip(self, tmp_path):
        unemb = _unemb(with_gain=True)
        p = tmp_path / "u.npz"
        save_unembedding(unemb, p)
        back = load_unembedding(p)
        np.testing.assert_array_equal(back["W_U"], unemb["W_U"])
        np.testing.assert_array_equal(back["ln_gamma"], unemb["ln_gamma"])
        assert back["ln_beta"] is None
        assert back["head_attr"] == "embed_out"


# ---------------------------------------------------------------------------
# lens_band — primitives
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_layernorm_zero_mean_unit_var(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, 16)) * 3 + 7
        y = numpy_layernorm(x, None, None)
        np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-5)
        np.testing.assert_allclose(y.std(axis=-1), 1.0, atol=1e-3)

    def test_kurtosis_peaked_vs_gaussian(self):
        rng = np.random.default_rng(1)
        gauss = rng.standard_normal((1, 5000))
        peaked = np.zeros((1, 5000))
        peaked[0, :3] = 50.0
        k_g = excess_kurtosis_rows(gauss)[0]
        k_p = excess_kurtosis_rows(peaked)[0]
        assert abs(k_g) < 0.5
        assert k_p > 100.0

    def test_autocorr_constant_sequence(self):
        ids = np.zeros(100, dtype=int)
        match, null = top1_autocorr(ids, 1)
        assert match == 1.0 and null == 1.0    # excess = 0: null saturates too

    def test_autocorr_alternating_vs_iid(self):
        # perfectly persistent blocks vs iid uniform
        blocks = np.repeat(np.arange(10), 10)
        m_b, n_b = top1_autocorr(blocks, 1)
        assert m_b - n_b > 0.5
        rng = np.random.default_rng(2)
        iid = rng.integers(0, 50, 2000)
        m_i, n_i = top1_autocorr(iid, 1)
        assert abs(m_i - n_i) < 0.05

    def test_autocorr_short_sequence_nan(self):
        m, n = top1_autocorr(np.array([1, 2]), 4)
        assert math.isnan(m) and math.isnan(n)


# ---------------------------------------------------------------------------
# lens_band — end-to-end on synthetic three-regime activations
# ---------------------------------------------------------------------------

def _synthetic_run(n_layers=12, n_tokens=60, d=16, vocab=64,
                   onset=4, motor=10, seed=0):
    """
    Layers < onset: isotropic noise (no readout structure).
    onset <= L < motor: activations aligned with a persistent per-block
      token direction (high kurtosis, high autocorrelation).
    L >= motor: aligned with the final layer's per-position prediction.
    """
    rng = np.random.default_rng(seed)
    W_U = rng.standard_normal((vocab, d)).astype(np.float32)
    W_U /= np.linalg.norm(W_U, axis=1, keepdims=True)
    unemb = {"W_U": W_U, "ln_gamma": None, "ln_beta": None,
             "head_attr": "lm_head", "vocab_size": vocab, "d_model": d}

    # final prediction: token id varies per position
    pred_ids = rng.integers(0, vocab, n_tokens)
    # persistent mid-band concept: one id per 20-token block
    concept_ids = np.repeat(rng.integers(0, vocab, n_tokens // 20 + 1),
                            20)[:n_tokens]

    acts = np.zeros((n_layers, n_tokens, d), dtype=np.float32)
    for L in range(n_layers):
        noise = rng.standard_normal((n_tokens, d)).astype(np.float32) * 0.05
        if L < onset:
            acts[L] = noise
        elif L < motor:
            acts[L] = 8.0 * W_U[concept_ids] + noise
        else:
            acts[L] = 8.0 * W_U[pred_ids] + noise
    return acts, unemb, onset, motor


class TestBandDetection:
    def test_planted_onsets_recovered(self):
        acts, unemb, onset, motor = _synthetic_run()
        result = compute_lens_band(acts, unemb, apply_ln=False)
        band = detect_band(result)
        assert band["band_onset_layer"] == onset
        assert band["motor_onset_layer"] == motor
        assert band["band_width"] == motor - onset

    def test_final_layer_agrees_with_itself(self):
        acts, unemb, _, _ = _synthetic_run()
        result = compute_lens_band(acts, unemb, apply_ln=False)
        assert result["per_layer"][-1]["agree_top1"] == 1.0

    def test_midband_autocorr_positive(self):
        acts, unemb, _, _ = _synthetic_run()
        result = compute_lens_band(acts, unemb, apply_ln=False)
        band = detect_band(result)
        assert band["midband_autocorr_d4"] > 0.3

    def test_no_band_gives_nan_not_crash(self):
        rng = np.random.default_rng(3)
        acts = rng.standard_normal((6, 40, 8)).astype(np.float32) * 0.01
        unemb = _unemb(vocab=32, d=8)
        result = compute_lens_band(acts, unemb, apply_ln=False)
        band = detect_band(result, kurt_threshold=50.0, agree_threshold=1.01)
        assert math.isnan(band["band_onset_layer"])
        assert math.isnan(band["band_width_frac"])

    def test_shape_and_dim_errors(self):
        unemb = _unemb(vocab=32, d=8)
        with pytest.raises(ValueError):
            compute_lens_band(np.zeros((4, 8)), unemb)
        with pytest.raises(ValueError):
            compute_lens_band(np.zeros((2, 4, 9), dtype=np.float32), unemb)


class TestScalarExtractors:
    def test_extractor_reads_written_json(self, tmp_path):
        acts, unemb, onset, _ = _synthetic_run()
        result = compute_lens_band(acts, unemb, apply_ln=False)
        band = detect_band(result)
        with open(tmp_path / LENS_BAND_FILENAME, "w") as f:
            json.dump({"series": result, "band": band}, f)
        val = _s_band_onset_frac(tmp_path)
        assert val == pytest.approx(onset / (result["n_layers"] - 1))

    def test_extractor_missing_file_nan(self, tmp_path):
        assert math.isnan(_s_band_onset_frac(tmp_path))
