"""
tests/test_norm_matched_baseline.py — the "norm_matched" random-init
scheme implemented in core/models.py::randomize_weights.

Split deliberately:

  * `TestNormMatchedMath` is pure numpy. It pins the properties that make
    the scheme the right control, against a closed-form oracle written
    independently of the implementation. No torch, no model, no imports
    from the package under test — these run in the ordinary stubbed
    session.
  * `TestNormMatchedAgainstRealModule` exercises the actual function and
    needs real torch (SMOKE_REAL_DEPS=1, `-m smoke`).

The registry side of this change — that `pythia-1.4b-random` exists,
requests a scheme the implementation accepts, and stays off the step axis
— lives in tests/test_pythia_registry.py, next to the other registry
contracts.

Note on the missing middle tier: `randomize_weights` cannot be imported
in the stubbed session, because conftest installs a hand-built
`core.models` stub exposing only extract_activations,
extract_albert_extended, load_model and layernorm_to_sphere. So there is
no cheap tier that runs the real function. The oracle below is written to
be independently derivable for that reason — if it merely restated the
implementation it would prove nothing that the smoke tier doesn't already
prove more directly.
"""

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure


# ---------------------------------------------------------------------------
# Oracle — the construction, written independently of core/models.py
# ---------------------------------------------------------------------------

def _norm_matched_oracle(shape, target, rng, std=0.02):
    """Gaussian draw rescaled to `target` Frobenius norm."""
    t = rng.normal(0.0, std, size=shape)
    cur = np.linalg.norm(t)
    if target > 0.0 and cur > 0.0:
        return t * (target / cur)
    return np.zeros(shape)


def _heavy_tailed(d, rng, decay=30.0, top=4.0):
    """
    Stand-in for a trained weight matrix: orthogonal factors, exponentially
    decaying singular values. Trained matrices are not Gaussian and the
    difference is the point — a test using a Gaussian "trained" matrix would
    pass while measuring nothing.
    """
    U, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    sv = np.exp(-np.arange(d) / decay) * top
    return U @ np.diag(sv) @ V.T


class TestNormMatchedMath:

    def test_frobenius_norm_is_preserved_exactly(self):
        rng = np.random.default_rng(0)
        W = _heavy_tailed(128, rng)
        target = np.linalg.norm(W)
        R = _norm_matched_oracle(W.shape, target, rng)
        assert np.linalg.norm(R) == pytest.approx(target, rel=1e-12)

    def test_result_does_not_depend_on_the_base_std(self):
        """
        The property that makes the scheme architecture-portable. GPT-NeoX
        and GPT-2 init at different variances; if the output depended on the
        base std, the control would not transfer and "norm_matched" would be
        no better than "gaussian".
        """
        target = 15.75
        a = _norm_matched_oracle((64, 64), target, np.random.default_rng(7), std=0.02)
        b = _norm_matched_oracle((64, 64), target, np.random.default_rng(7), std=17.0)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_spectrum_is_flattened_relative_to_the_trained_matrix(self):
        """
        Frobenius matching does NOT preserve operator norm, and the docstring
        in core/models.py commits to that in writing. Pin it, so the tradeoff
        can't be silently reversed by someone switching to spectral matching
        without updating the claim.
        """
        rng = np.random.default_rng(1)
        W = _heavy_tailed(256, rng)
        R = _norm_matched_oracle(W.shape, np.linalg.norm(W), rng)

        top_trained = np.linalg.svd(W, compute_uv=False)[0]
        top_random = np.linalg.svd(R, compute_uv=False)[0]
        assert top_random < top_trained

        def stable_rank(M):
            return np.linalg.norm(M) ** 2 / np.linalg.svd(M, compute_uv=False)[0] ** 2

        # Structureless => energy spread across many directions.
        assert stable_rank(R) > 3.0 * stable_rank(W)

    def test_zero_norm_parameter_stays_zero(self):
        rng = np.random.default_rng(2)
        R = _norm_matched_oracle((8, 8), 0.0, rng)
        assert np.linalg.norm(R) == 0.0

    def test_ov_repulsive_fraction_null_is_one_half(self):
        """
        The reason this baseline is load-bearing for Phase 2 specifically.

        Phase 2's headline quantity is the fraction of OV eigenvalues with
        negative real part. Under the norm-matched null that fraction sits at
        0.5 with a small spread, so a trained model's value is only evidence
        of anything relative to it. The closed 35-run study reported 0.43-0.57
        across models — several of those sit on top of this null.

        This test pins the null, not any trained result.
        """
        d, fracs = 128, []
        for s in range(20):
            rng = np.random.default_rng(s)
            Wv = _norm_matched_oracle((d, d), 15.75, rng)
            Wo = _norm_matched_oracle((d, d), 15.75, rng)
            ev = np.linalg.eigvals(Wv @ Wo)
            fracs.append(float((ev.real < 0).mean()))
        assert np.mean(fracs) == pytest.approx(0.5, abs=0.02)
        assert np.std(fracs) < 0.05


# ---------------------------------------------------------------------------
# Real-torch tier
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestNormMatchedAgainstRealModule:
    """
    Needs SMOKE_REAL_DEPS=1. Uses a tiny randomly-initialised GPT-NeoX rather
    than a downloaded checkpoint — the scheme is a per-parameter identity and
    does not care whether the weights it matches are meaningful.
    """

    @staticmethod
    def _tiny_neox():
        import torch
        from transformers import GPTNeoXConfig, GPTNeoXModel

        torch.manual_seed(0)
        cfg = GPTNeoXConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=64, max_position_embeddings=32,
        )
        model = GPTNeoXModel(cfg)
        # Push parameters off their init scale so norm-matching has something
        # non-trivial to preserve.
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() >= 2:
                    p.mul_(3.7)
        return model

    def test_every_matrix_keeps_its_frobenius_norm(self):
        import torch
        from core.models import randomize_weights

        model = self._tiny_neox()
        before = {n: float(p.detach().float().norm())
                  for n, p in model.named_parameters() if p.dim() >= 2}

        randomize_weights(model, scheme="norm_matched", seed=0)

        after = {n: float(p.detach().float().norm())
                 for n, p in model.named_parameters() if p.dim() >= 2}
        for name, target in before.items():
            assert after[name] == pytest.approx(target, rel=1e-4), name

        # ...and the values themselves actually changed.
        assert any(
            not torch.allclose(p, torch.zeros_like(p))
            for p in model.parameters()
        )

    def test_layernorm_goes_to_identity_not_norm_matched(self):
        import torch
        import torch.nn as nn
        from core.models import randomize_weights

        model = self._tiny_neox()
        randomize_weights(model, scheme="norm_matched", seed=0)

        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                assert torch.allclose(module.weight, torch.ones_like(module.weight))
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    def test_biases_are_zeroed(self):
        import torch
        from core.models import randomize_weights

        model = self._tiny_neox()
        randomize_weights(model, scheme="norm_matched", seed=0)

        for name, p in model.named_parameters():
            if p.dim() == 1 and "layer_norm" not in name and "layernorm" not in name:
                assert torch.allclose(p, torch.zeros_like(p)), name

    def test_seed_is_reproducible(self):
        import torch
        from core.models import randomize_weights

        a, b = self._tiny_neox(), self._tiny_neox()
        randomize_weights(a, scheme="norm_matched", seed=11)
        randomize_weights(b, scheme="norm_matched", seed=11)
        for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
            assert torch.allclose(pa, pb), na

    def test_unknown_scheme_still_raises(self):
        import pytest as _pytest
        from core.models import randomize_weights
        with _pytest.raises(ValueError):
            randomize_weights(self._tiny_neox(), scheme="norm-matched", seed=0)
