"""
archive/tests/test_p5_tuned_lens_math.py — pure-numpy tests for
archive/p5_single_mstate_analysis/train_tuned_lens.py's fitting logic
(item 4, Group E fix).

Split out of tests/test_head_ablation_math.py when Phase 5 was archived:
that file tested two unrelated modules, one of which (p2_eigenspectra/
head_ablation.py) is still live. The p2 half stayed under tests/; this is
the p5 half, archived with the code it exercises.

No torch anywhere on these paths.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# train_tuned_lens — fitting logic
# ---------------------------------------------------------------------------

class TestFitAffineTranslator:
    def test_recovers_planted_affine_map(self):
        """Oracle: generate H_final = A* H + b* exactly; the fit must
        recover (A*, b*) to numerical precision when n >> d."""
        from p5_single_mstate_analysis.train_tuned_lens import fit_affine_translator
        rng = np.random.default_rng(0)
        d, n = 6, 400
        A_star = rng.standard_normal((d, d))
        b_star = rng.standard_normal(d)
        H = rng.standard_normal((n, d))
        H_final = H @ A_star.T + b_star
        A, b = fit_affine_translator(H, H_final, ridge=1e-8)
        np.testing.assert_allclose(A, A_star, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(b, b_star, rtol=1e-3, atol=1e-3)

    def test_identity_at_final_layer(self):
        from p5_single_mstate_analysis.train_tuned_lens import (
            fit_affine_translator, identity_deviation,
        )
        rng = np.random.default_rng(1)
        H = rng.standard_normal((300, 5))
        A, b = fit_affine_translator(H, H, ridge=1e-8)
        assert identity_deviation(A, b) < 1e-3

    def test_shape_mismatch_raises(self):
        from p5_single_mstate_analysis.train_tuned_lens import fit_affine_translator
        with pytest.raises(ValueError):
            fit_affine_translator(np.zeros((10, 4)), np.zeros((10, 5)))


class TestLensStackAndIO:
    def test_stack_fit_and_npz_roundtrip(self, tmp_path):
        """save_lens must write the exact A_L{i}/b_L{i} format
        tuned_lens_cluster.load_tuned_lens reads — the producer/consumer
        contract this whole fix exists for."""
        from p5_single_mstate_analysis.train_tuned_lens import (
            fit_lens_from_activation_stack, save_lens,
        )
        from p5_single_mstate_analysis.tuned_lens_cluster import (
            load_tuned_lens, apply_tuned_lens,
        )
        rng = np.random.default_rng(2)
        n_layers, n_tokens, d = 4, 200, 5
        acts = rng.standard_normal((n_layers, n_tokens, d)).astype(np.float32)
        lens = fit_lens_from_activation_stack(acts, ridge=1e-6)
        assert set(lens.keys()) == set(range(n_layers))
        # near-identity at final layer
        assert lens[n_layers - 1]["identity_deviation"] < 1e-2

        out = save_lens(lens, tmp_path / "tuned_lens_test.npz",
                        meta={"model": "test"})
        loaded = load_tuned_lens(out)
        assert loaded is not None
        assert set(loaded.keys()) == set(range(n_layers))
        for L in range(n_layers):
            np.testing.assert_allclose(loaded[L]["A"], lens[L]["A"])
            np.testing.assert_allclose(loaded[L]["b"], lens[L]["b"])

        # apply_tuned_lens consumes the loaded dict without error
        v = rng.standard_normal(d).astype(np.float32)
        out_v = apply_tuned_lens(v, 1, loaded)
        np.testing.assert_allclose(
            out_v, loaded[1]["A"] @ v + loaded[1]["b"], rtol=1e-5)
        # sidecar written
        assert (tmp_path / "tuned_lens_test.json").exists()

    def test_translated_vector_closer_to_final_than_raw(self):
        """The functional claim behind the fix: on structured data (an
        affine drift per layer), the translated layer-0 vector must land
        closer to the final state than the untranslated one — otherwise
        the lens adds nothing over the frozen-head fallback."""
        from p5_single_mstate_analysis.train_tuned_lens import (
            fit_lens_from_activation_stack,
        )
        rng = np.random.default_rng(3)
        n_tokens, d = 300, 6
        base = rng.standard_normal((n_tokens, d))
        M = np.eye(d) + 0.3 * rng.standard_normal((d, d))
        shift = rng.standard_normal(d)
        acts = np.stack([base, base @ M.T + shift,
                         (base @ M.T + shift) @ M.T + shift], axis=0)
        lens = fit_lens_from_activation_stack(acts, ridge=1e-6)
        A0, b0 = lens[0]["A"], lens[0]["b"]
        translated = acts[0] @ A0.T + b0
        err_translated = np.linalg.norm(translated - acts[-1])
        err_raw = np.linalg.norm(acts[0] - acts[-1])
        assert err_translated < 0.1 * err_raw
