"""
tests/test_core_ln_frame.py — Oracle tests for core/ln_frame.py (frames item 2).

Pure tier, torch-free. The extraction functions are duck-typed on module
attribute structure, so they are tested here against SimpleNamespace
fakes shaped like a GPT-NeoX model — no transformers import needed. The
finite-difference test cross-checks ln_transform's forward map against
p2b_imaginary/layernorm_jacobian.py's analytic Jacobian so the two
modules cannot drift apart silently.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from core.ln_frame import (
    ln_transform,
    ln_frame_gram,
    get_ln_params,
    get_final_ln_params,
    n_blocks,
    frame_for_hidden_state,
    DEFAULT_LN_EPS,
)

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

_rng = np.random.default_rng(77)


# ---------------------------------------------------------------------------
# Pure transform — exact oracles
# ---------------------------------------------------------------------------

class TestLnTransform:
    def test_hand_computed_example(self):
        """3-vector worked by hand: x = [1, 2, 6], eps = 0.
        mu = 3, biased var = (4 + 1 + 9)/3 = 14/3,
        xhat = [-2, -1, 3] / sqrt(14/3)."""
        x = np.array([1.0, 2.0, 6.0])
        expected = np.array([-2.0, -1.0, 3.0]) / math.sqrt(14.0 / 3.0)
        out = ln_transform(x, eps=0.0)[0]
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_plain_ln_zero_mean_unit_var(self):
        """gamma=1, beta=0 -> every row zero-mean, biased variance
        var/(var+eps) (i.e. ~1 for eps -> 0)."""
        X = _rng.normal(size=(10, 64)) * 5.0 + 2.0
        Y = ln_transform(X, eps=0.0)
        np.testing.assert_allclose(Y.mean(axis=-1), 0.0, atol=1e-10)
        np.testing.assert_allclose(((Y - 0.0) ** 2).mean(axis=-1), 1.0, atol=1e-10)

    def test_planted_affine_recovered(self):
        """Output must equal gamma * plain_LN(x) + beta exactly."""
        X = _rng.normal(size=(6, 16))
        gamma = _rng.normal(size=16)
        beta = _rng.normal(size=16)
        plain = ln_transform(X)
        np.testing.assert_allclose(
            ln_transform(X, gamma=gamma, beta=beta),
            plain * gamma + beta, rtol=1e-12,
        )

    def test_eps_inside_sqrt_torch_convention(self):
        """torch.nn.LayerNorm adds eps to var INSIDE the sqrt. A constant
        row (var = 0) must map to exactly 0, not nan/inf."""
        x = np.full((1, 8), 3.7)
        out = ln_transform(x, eps=1e-5)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_scale_invariance_at_eps_zero(self):
        """Plain LN with eps=0 is invariant to positive rescaling and to
        additive constants — the projection property the module docstring
        claims."""
        x = _rng.normal(size=(1, 32))
        base = ln_transform(x, eps=0.0)
        np.testing.assert_allclose(ln_transform(7.3 * x, eps=0.0), base, rtol=1e-9)
        np.testing.assert_allclose(ln_transform(x + 11.0, eps=0.0), base, atol=1e-8)

    def test_finite_difference_matches_layernorm_jacobian(self):
        """Cross-module oracle: ln_transform (gamma=1, beta=0) must be the
        forward map whose derivative p2b_imaginary/layernorm_jacobian.py
        computes. Directional finite differences vs J @ v."""
        try:
            from p2b_imaginary.layernorm_jacobian import layernorm_jacobian
        except ImportError:
            pytest.skip("p2b_imaginary not importable in this session")
        d = 24
        x = _rng.normal(size=d) * 2.0
        eps_ln = 1e-8
        J = layernorm_jacobian(x, eps=eps_ln)
        h = 1e-6
        for _ in range(5):
            v = _rng.normal(size=d)
            v /= np.linalg.norm(v)
            fd = (ln_transform(x + h * v, eps=eps_ln)[0]
                  - ln_transform(x - h * v, eps=eps_ln)[0]) / (2 * h)
            np.testing.assert_allclose(fd, J @ v, rtol=1e-4, atol=1e-6)


class TestLnFrameGram:
    def test_gram_properties(self):
        X = _rng.normal(size=(9, 32))
        gamma = np.abs(_rng.normal(size=32)) + 0.1
        G = ln_frame_gram(X, gamma=gamma)
        np.testing.assert_allclose(np.diag(G), 1.0, atol=1e-10)
        np.testing.assert_allclose(G, G.T, atol=1e-12)
        assert np.abs(G).max() <= 1.0 + 1e-10

    def test_frame_changes_geometry(self):
        """A strongly anisotropic gamma must change pairwise cosines vs
        the plain L2 frame — otherwise the second frame is vacuous."""
        from core.metrics import gram_matrix, pairwise_upper
        X = _rng.normal(size=(12, 32))
        gamma = np.ones(32)
        gamma[:4] = 40.0                     # crush geometry onto 4 channels
        G_ln = ln_frame_gram(X, gamma=gamma)
        G_l2 = gram_matrix(X)
        diff = np.abs(pairwise_upper(G_ln) - pairwise_upper(G_l2)).max()
        assert diff > 0.05

    def test_beta_included_in_frame(self):
        """The learned bias shifts all rows by one vector, which changes
        directions — the frame must include it (module docstring
        contract), so beta != 0 must move the Gram."""
        X = _rng.normal(size=(8, 16))
        beta = np.zeros(16)
        beta[0] = 25.0
        G0 = ln_frame_gram(X)
        G1 = ln_frame_gram(X, beta=beta)
        assert np.abs(G0 - G1).max() > 0.05


# ---------------------------------------------------------------------------
# Extraction — duck-typed against a GPT-NeoX-shaped fake
# ---------------------------------------------------------------------------

def _fake_ln(d, seed, with_bias=True, eps=1e-5):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        weight=rng.normal(size=d),
        bias=rng.normal(size=d) if with_bias else None,
        eps=eps,
    )


def _fake_neox(n_layers=4, d=16, wrapped=False):
    layers = [
        SimpleNamespace(
            input_layernorm=_fake_ln(d, seed=10 + i),
            post_attention_layernorm=_fake_ln(d, seed=100 + i),
        )
        for i in range(n_layers)
    ]
    inner = SimpleNamespace(layers=layers, final_layer_norm=_fake_ln(d, seed=999))
    if wrapped:
        return SimpleNamespace(gpt_neox=inner)          # ForCausalLM shape
    return inner                                        # bare GPTNeoXModel shape


class TestGetLnParams:
    def test_attn_vs_mlp_distinct(self):
        m = _fake_neox()
        pa = get_ln_params(m, 2, which="attn")
        pm = get_ln_params(m, 2, which="mlp")
        assert not np.allclose(pa["gamma"], pm["gamma"])
        np.testing.assert_allclose(
            pa["gamma"], m.layers[2].input_layernorm.weight
        )
        np.testing.assert_allclose(
            pm["gamma"], m.layers[2].post_attention_layernorm.weight
        )

    def test_wrapped_and_bare_agree(self):
        bare, wrapped = _fake_neox(), None
        wrapped = SimpleNamespace(gpt_neox=bare)
        np.testing.assert_allclose(
            get_ln_params(bare, 1)["gamma"],
            get_ln_params(wrapped, 1)["gamma"],
        )

    def test_bad_which_raises(self):
        with pytest.raises(ValueError):
            get_ln_params(_fake_neox(), 0, which="ffn")

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            get_ln_params(_fake_neox(n_layers=4), 4)

    def test_final_ln(self):
        m = _fake_neox()
        np.testing.assert_allclose(
            get_final_ln_params(m)["gamma"], m.final_layer_norm.weight
        )

    def test_missing_bias_gives_none(self):
        m = _fake_neox()
        m.layers[0].input_layernorm = _fake_ln(16, seed=1, with_bias=False)
        assert get_ln_params(m, 0)["beta"] is None

    def test_eps_read_from_module(self):
        m = _fake_neox()
        m.layers[0].input_layernorm.eps = 3e-4
        assert get_ln_params(m, 0)["eps"] == pytest.approx(3e-4)


class TestFrameForHiddenState:
    """The off-by-one contract, exhaustively. n_blocks=4 fake;
    embedding-stripped convention -> 4 hidden states (block outputs 0-3)."""

    def test_interior_reads_next_block(self):
        m = _fake_neox(n_layers=4)
        f = frame_for_hidden_state(m, 1, n_hidden_states=4)
        assert f["frame"] == "block" and f["block_idx"] == 2
        np.testing.assert_allclose(
            f["params"]["gamma"], m.layers[2].input_layernorm.weight
        )

    def test_last_reads_final_ln(self):
        m = _fake_neox(n_layers=4)
        f = frame_for_hidden_state(m, 3, n_hidden_states=4)
        assert f["frame"] == "final"
        np.testing.assert_allclose(
            f["params"]["gamma"], m.final_layer_norm.weight
        )

    def test_last_post_final_ln_is_identity(self):
        """core/models.py standard path records the final entry post-ln_f
        (status-5.md) — applying final LN again would be wrong, so the
        frame must be the identity."""
        m = _fake_neox(n_layers=4)
        f = frame_for_hidden_state(m, 3, n_hidden_states=4,
                                   last_is_post_final_ln=True)
        assert f["frame"] == "identity" and f["params"] is None

    def test_unstripped_convention(self):
        """embedding_stripped=False: index 0 is the raw embedding, read
        by block 0."""
        m = _fake_neox(n_layers=4)
        f = frame_for_hidden_state(m, 0, n_hidden_states=5,
                                   embedding_stripped=False)
        assert f["frame"] == "block" and f["block_idx"] == 0

    def test_mlp_frame_selectable(self):
        m = _fake_neox(n_layers=4)
        f = frame_for_hidden_state(m, 0, n_hidden_states=4, which="mlp")
        np.testing.assert_allclose(
            f["params"]["gamma"], m.layers[1].post_attention_layernorm.weight
        )

    def test_inconsistent_conventions_refuse_to_guess(self):
        """5 hidden states, embedding_stripped=True, 4 blocks: index 3
        would resolve to reader block 4 — out of range, must raise, not
        silently pick a frame."""
        m = _fake_neox(n_layers=4)
        with pytest.raises(IndexError):
            frame_for_hidden_state(m, 3, n_hidden_states=5)

    def test_index_out_of_range_raises(self):
        with pytest.raises(IndexError):
            frame_for_hidden_state(_fake_neox(), 4, n_hidden_states=4)

    def test_params_splat_into_transform(self):
        """End-to-end shape: resolved params feed ln_frame_gram directly."""
        m = _fake_neox(n_layers=4, d=16)
        f = frame_for_hidden_state(m, 0, n_hidden_states=4)
        X = _rng.normal(size=(6, 16))
        G = ln_frame_gram(X, **f["params"])
        assert G.shape == (6, 6)
        np.testing.assert_allclose(np.diag(G), 1.0, atol=1e-10)
