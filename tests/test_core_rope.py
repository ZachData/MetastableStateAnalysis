"""
tests/test_core_rope.py — oracle tests for core/rope.py.

Every test here checks an exact mathematical property or an independent
reimplementation, not a pipeline behaviour. Pure numpy; runs under the
stubbed session with no torch, no transformers, no weights.

The property that matters most is `test_relative_offset_identity`: it is
the one that licenses treating the weight-space bilinear as a function of
Δ = j - i, which is what the whole rotary-aware Phase 6 rewrite rests on.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.rope import (
    DEFAULT_ROPE_BASE,
    apply_rope,
    model_uses_rope,
    qk_logits_with_rope,
    qk_matrix_at_offset,
    qk_sa_fractions_at_offset,
    rope_angles,
    rope_config_from_model,
    rope_energy_fraction,
    rope_frequencies,
    rope_rotation,
    rope_sa_fractions,
)


HEAD_SIZE = 16
N_ROT = 8            # rotary_pct = 0.5 here; Pythia's 0.25 is exercised separately
D_MODEL = 24
BASE = DEFAULT_ROPE_BASE


def _rng():
    return np.random.default_rng(20260721)


# ---------------------------------------------------------------------------
# Frequencies
# ---------------------------------------------------------------------------

class TestFrequencies:

    def test_matches_hf_formula(self):
        """inv_freq = 1 / base ** (arange(0, dim, 2) / dim), independently written."""
        got = rope_frequencies(N_ROT, BASE)
        want = 1.0 / (BASE ** (np.arange(0, N_ROT, 2) / N_ROT))
        assert np.allclose(got, want, atol=1e-15)
        assert got.shape == (N_ROT // 2,)

    def test_first_frequency_is_one(self):
        assert rope_frequencies(N_ROT, BASE)[0] == pytest.approx(1.0)

    def test_monotone_decreasing(self):
        f = rope_frequencies(64, BASE)
        assert np.all(np.diff(f) < 0)

    def test_odd_width_rejected(self):
        with pytest.raises(ValueError):
            rope_frequencies(7, BASE)

    def test_zero_width_is_empty(self):
        assert rope_frequencies(0, BASE).shape == (0,)

    def test_angles_shape_and_scaling(self):
        pos = np.array([0.0, 1.0, 5.0])
        th = rope_angles(pos, N_ROT, BASE)
        assert th.shape == (3, N_ROT // 2)
        assert np.allclose(th[0], 0.0)
        assert np.allclose(th[2], 5.0 * th[1])


# ---------------------------------------------------------------------------
# apply_rope — independent reimplementation of HF's rotate_half path
# ---------------------------------------------------------------------------

def _hf_style_rope(x, positions, n_rot, base):
    """
    Independent reimplementation: y = x*cos + rotate_half(x)*sin on the
    rotary block, with cos/sin built from cat(freqs, freqs) exactly as HF
    does. Written from the HF formulation rather than from rope.py, so
    agreement is evidence and not a tautology.
    """
    h = n_rot // 2
    inv = 1.0 / (base ** (np.arange(0, n_rot, 2) / n_rot))
    freqs = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)          # (n, n_rot)
    cos, sin = np.cos(emb), np.sin(emb)

    x_rot = x[:, :n_rot]
    x_pass = x[:, n_rot:]
    rotate_half = np.concatenate([-x_rot[:, h:], x_rot[:, :h]], axis=-1)
    y_rot = x_rot * cos + rotate_half * sin
    return np.concatenate([y_rot, x_pass], axis=-1)


class TestApplyRope:

    def test_matches_hf_reimplementation(self):
        rng = _rng()
        x = rng.normal(size=(7, HEAD_SIZE))
        pos = np.arange(7)
        assert np.allclose(
            apply_rope(x, pos, N_ROT, BASE),
            _hf_style_rope(x, pos, N_ROT, BASE),
            atol=1e-12,
        )

    def test_pythia_ratio_quarter(self):
        """Pythia's actual geometry: head_size 128, rotary_pct 0.25."""
        rng = _rng()
        x = rng.normal(size=(5, 128))
        pos = np.arange(5)
        assert np.allclose(
            apply_rope(x, pos, 32, BASE),
            _hf_style_rope(x, pos, 32, BASE),
            atol=1e-12,
        )

    def test_pass_through_dims_untouched(self):
        rng = _rng()
        x = rng.normal(size=(6, HEAD_SIZE))
        y = apply_rope(x, np.arange(6), N_ROT, BASE)
        assert np.allclose(y[:, N_ROT:], x[:, N_ROT:], atol=1e-15)

    def test_position_zero_is_identity(self):
        rng = _rng()
        x = rng.normal(size=(4, HEAD_SIZE))
        y = apply_rope(x, np.zeros(4), N_ROT, BASE)
        assert np.allclose(y, x, atol=1e-15)

    def test_norm_preserved(self):
        rng = _rng()
        x = rng.normal(size=(9, HEAD_SIZE))
        y = apply_rope(x, np.arange(9), N_ROT, BASE)
        assert np.allclose(
            np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1), atol=1e-12
        )

    def test_zero_rotary_is_noop(self):
        rng = _rng()
        x = rng.normal(size=(4, HEAD_SIZE))
        assert np.allclose(apply_rope(x, np.arange(4), 0, BASE), x)

    def test_position_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            apply_rope(np.zeros((4, HEAD_SIZE)), np.arange(3), N_ROT, BASE)

    def test_rotary_wider_than_head_raises(self):
        with pytest.raises(ValueError):
            apply_rope(np.zeros((2, 8)), np.arange(2), 16, BASE)


# ---------------------------------------------------------------------------
# rope_rotation and the relative-offset identity
# ---------------------------------------------------------------------------

class TestRotationMatrix:

    def test_orthogonal(self):
        R = rope_rotation(3.0, HEAD_SIZE, N_ROT, BASE)
        assert np.allclose(R @ R.T, np.eye(HEAD_SIZE), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_agrees_with_apply_rope(self):
        """R(m) @ q must equal apply_rope(q, m)."""
        rng = _rng()
        q = rng.normal(size=HEAD_SIZE)
        for m in (0.0, 1.0, 4.0, 17.0):
            R = rope_rotation(m, HEAD_SIZE, N_ROT, BASE)
            assert np.allclose(
                R @ q, apply_rope(q[None, :], [m], N_ROT, BASE)[0], atol=1e-12
            )

    def test_relative_offset_identity(self):
        """
        ⟨rope(q, m), rope(k, n)⟩ == q^T R(n - m) k.

        This is the licence for treating the bilinear as a function of
        Δ alone. If it fails, every offset-resolved quantity downstream
        is meaningless.
        """
        rng = _rng()
        q = rng.normal(size=HEAD_SIZE)
        k = rng.normal(size=HEAD_SIZE)
        for m, n in [(0, 0), (3, 1), (1, 3), (12, 5), (7, 7)]:
            lhs = float(
                apply_rope(q[None, :], [m], N_ROT, BASE)[0]
                @ apply_rope(k[None, :], [n], N_ROT, BASE)[0]
            )
            rhs = float(q @ rope_rotation(n - m, HEAD_SIZE, N_ROT, BASE) @ k)
            assert lhs == pytest.approx(rhs, abs=1e-10)

    def test_zero_offset_is_identity(self):
        R = rope_rotation(0.0, HEAD_SIZE, N_ROT, BASE)
        assert np.allclose(R, np.eye(HEAD_SIZE), atol=1e-15)

    def test_composition(self):
        a = rope_rotation(2.0, HEAD_SIZE, N_ROT, BASE)
        b = rope_rotation(5.0, HEAD_SIZE, N_ROT, BASE)
        assert np.allclose(a @ b, rope_rotation(7.0, HEAD_SIZE, N_ROT, BASE), atol=1e-12)

    def test_transpose_is_negative_offset(self):
        R = rope_rotation(4.0, HEAD_SIZE, N_ROT, BASE)
        assert np.allclose(R.T, rope_rotation(-4.0, HEAD_SIZE, N_ROT, BASE), atol=1e-12)


# ---------------------------------------------------------------------------
# S/A structure
# ---------------------------------------------------------------------------

class TestRopeSAFractions:

    def test_closed_form_matches_brute_force(self):
        for delta in (0.0, 1.0, 2.0, 13.0, -4.0):
            R = rope_rotation(delta, HEAD_SIZE, N_ROT, BASE)
            S = (R + R.T) / 2.0
            A = (R - R.T) / 2.0
            got = rope_sa_fractions(delta, HEAD_SIZE, N_ROT, BASE)
            n2 = np.linalg.norm(R, "fro") ** 2
            assert got["norm2"] == pytest.approx(n2, abs=1e-10)
            assert got["s_frac"] == pytest.approx(
                np.linalg.norm(S, "fro") ** 2 / n2, abs=1e-10
            )
            assert got["a_frac"] == pytest.approx(
                np.linalg.norm(A, "fro") ** 2 / n2, abs=1e-10
            )

    def test_fractions_sum_to_one(self):
        for delta in (0.0, 1.0, 9.0):
            f = rope_sa_fractions(delta, HEAD_SIZE, N_ROT, BASE)
            assert f["s_frac"] + f["a_frac"] == pytest.approx(1.0, abs=1e-12)

    def test_zero_offset_has_no_antisymmetry(self):
        assert rope_sa_fractions(0.0, HEAD_SIZE, N_ROT, BASE)["a_frac"] == 0.0

    def test_antisymmetry_appears_at_nonzero_offset(self):
        """The null model P6-I2 must be measured against."""
        assert rope_sa_fractions(1.0, HEAD_SIZE, N_ROT, BASE)["a_frac"] > 0.0

    def test_even_in_offset_sign(self):
        a_pos = rope_sa_fractions(6.0, HEAD_SIZE, N_ROT, BASE)["a_frac"]
        a_neg = rope_sa_fractions(-6.0, HEAD_SIZE, N_ROT, BASE)["a_frac"]
        assert a_pos == pytest.approx(a_neg, abs=1e-14)

    def test_bounded_by_rotary_share(self):
        """a_frac can never exceed the fraction of dims rotary touches."""
        share = rope_energy_fraction(HEAD_SIZE, N_ROT)
        for delta in np.linspace(-50, 50, 41):
            assert rope_sa_fractions(delta, HEAD_SIZE, N_ROT, BASE)["a_frac"] <= share + 1e-12


class TestQKSAFractions:

    def test_trace_identities_match_brute_force(self):
        """
        The whole point of the closed form: it must equal the explicit
        d_model-space S/A split of M(Δ) = W_Q R(Δ) W_K^T.
        """
        rng = _rng()
        WQ = rng.normal(size=(D_MODEL, HEAD_SIZE))
        WK = rng.normal(size=(D_MODEL, HEAD_SIZE))
        for delta in (0.0, 1.0, -3.0, 11.0):
            R = rope_rotation(delta, HEAD_SIZE, N_ROT, BASE)
            M = qk_matrix_at_offset(WQ, WK, R)
            S = (M + M.T) / 2.0
            A = (M - M.T) / 2.0
            n2 = np.linalg.norm(M, "fro") ** 2

            got = qk_sa_fractions_at_offset(WQ, WK, R)
            assert got["norm2"] == pytest.approx(n2, rel=1e-10)
            assert got["s_frac"] == pytest.approx(
                np.linalg.norm(S, "fro") ** 2 / n2, rel=1e-9
            )
            assert got["a_frac"] == pytest.approx(
                np.linalg.norm(A, "fro") ** 2 / n2, rel=1e-9
            )

    def test_fractions_sum_to_one(self):
        rng = _rng()
        WQ = rng.normal(size=(D_MODEL, HEAD_SIZE))
        WK = rng.normal(size=(D_MODEL, HEAD_SIZE))
        f = qk_sa_fractions_at_offset(WQ, WK, rope_rotation(5.0, HEAD_SIZE, N_ROT, BASE))
        assert f["s_frac"] + f["a_frac"] == pytest.approx(1.0, abs=1e-10)

    def test_identity_R_reduces_to_plain_qk(self):
        """
        With R = None this must reproduce qk_decompose.decompose_qk_matrix's
        fractions exactly — the GPT-2 path is unchanged.
        """
        rng = _rng()
        WQ = rng.normal(size=(D_MODEL, HEAD_SIZE))
        WK = rng.normal(size=(D_MODEL, HEAD_SIZE))
        M = WQ @ WK.T
        n2 = np.linalg.norm(M, "fro") ** 2
        got = qk_sa_fractions_at_offset(WQ, WK, None)
        assert got["a_frac"] == pytest.approx(
            np.linalg.norm((M - M.T) / 2.0, "fro") ** 2 / n2, rel=1e-9
        )

    def test_content_bilinear_already_antisymmetric_at_zero_offset(self):
        """
        M(0) = W_Q W_K^T is not symmetric for generic weights, so a_frac > 0
        at Δ=0 is a content fact, not a positional one. Stated as a test so
        the distinction cannot be quietly lost.
        """
        rng = _rng()
        WQ = rng.normal(size=(D_MODEL, HEAD_SIZE))
        WK = rng.normal(size=(D_MODEL, HEAD_SIZE))
        assert qk_sa_fractions_at_offset(WQ, WK, None)["a_frac"] > 0.01

    def test_shape_guards(self):
        with pytest.raises(ValueError):
            qk_sa_fractions_at_offset(np.zeros((4, 8)), np.zeros((4, 8)))   # d_model < d_head
        with pytest.raises(ValueError):
            qk_sa_fractions_at_offset(np.zeros((D_MODEL, 8)), np.zeros((D_MODEL, 4)))
        with pytest.raises(ValueError):
            qk_sa_fractions_at_offset(
                np.zeros((D_MODEL, HEAD_SIZE)), np.zeros((D_MODEL, HEAD_SIZE)),
                R=np.eye(3),
            )


# ---------------------------------------------------------------------------
# Logit path
# ---------------------------------------------------------------------------

class TestLogits:

    def test_matches_offset_bilinear(self):
        """logits[i, j] == q_i^T R(j - i) k_j, the definition."""
        rng = _rng()
        X = rng.normal(size=(6, D_MODEL))
        WQ = rng.normal(size=(D_MODEL, HEAD_SIZE))
        WK = rng.normal(size=(D_MODEL, HEAD_SIZE))
        L = qk_logits_with_rope(X, WQ, WK, N_ROT, BASE)

        Q = X @ WQ
        K = X @ WK
        for i in range(6):
            for j in range(6):
                want = Q[i] @ rope_rotation(j - i, HEAD_SIZE, N_ROT, BASE) @ K[j]
                assert L[i, j] == pytest.approx(want, abs=1e-9)

    def test_zero_rotary_reduces_to_plain_bilinear(self):
        rng = _rng()
        X = rng.normal(size=(5, D_MODEL))
        WQ = rng.normal(size=(D_MODEL, HEAD_SIZE))
        WK = rng.normal(size=(D_MODEL, HEAD_SIZE))
        assert np.allclose(
            qk_logits_with_rope(X, WQ, WK, 0, BASE),
            X @ (WQ @ WK.T) @ X.T,
            atol=1e-9,
        )

    def test_rotary_actually_changes_logits(self):
        """
        Guard against a silently-disabled rotary branch: the whole point is
        that this differs from the plain bilinear.
        """
        rng = _rng()
        X = rng.normal(size=(6, D_MODEL))
        WQ = rng.normal(size=(D_MODEL, HEAD_SIZE))
        WK = rng.normal(size=(D_MODEL, HEAD_SIZE))
        plain = X @ (WQ @ WK.T) @ X.T
        roped = qk_logits_with_rope(X, WQ, WK, N_ROT, BASE)
        assert not np.allclose(plain, roped, atol=1e-6)
        assert np.allclose(np.diag(plain), np.diag(roped), atol=1e-9)  # Δ=0 unchanged

    def test_scale_applied(self):
        rng = _rng()
        X = rng.normal(size=(4, D_MODEL))
        WQ = rng.normal(size=(D_MODEL, HEAD_SIZE))
        WK = rng.normal(size=(D_MODEL, HEAD_SIZE))
        s = 1.0 / np.sqrt(HEAD_SIZE)
        assert np.allclose(
            qk_logits_with_rope(X, WQ, WK, N_ROT, BASE, scale=s),
            qk_logits_with_rope(X, WQ, WK, N_ROT, BASE) * s,
            atol=1e-12,
        )

    def test_custom_positions_respected(self):
        rng = _rng()
        X = rng.normal(size=(4, D_MODEL))
        WQ = rng.normal(size=(D_MODEL, HEAD_SIZE))
        WK = rng.normal(size=(D_MODEL, HEAD_SIZE))
        all_zero = qk_logits_with_rope(X, WQ, WK, N_ROT, BASE, positions=np.zeros(4))
        assert np.allclose(all_zero, X @ (WQ @ WK.T) @ X.T, atol=1e-9)


# ---------------------------------------------------------------------------
# Config extraction (duck-typed, no torch)
# ---------------------------------------------------------------------------

class TestConfigExtraction:

    def _fake(self, **kw):
        cfg = SimpleNamespace(
            hidden_size=2048, num_attention_heads=16,
            rotary_pct=0.25, rotary_emb_base=10000,
        )
        for k, v in kw.items():
            setattr(cfg, k, v)
        return SimpleNamespace(config=cfg)

    def test_pythia_1_4b_geometry(self):
        c = rope_config_from_model(self._fake())
        assert c["head_size"] == 128
        assert c["rotary_ndims"] == 32          # 0.25 * 128 — NOT 128
        assert c["base"] == 10000.0
        assert c["scale"] == pytest.approx(1.0 / np.sqrt(128))

    def test_rotary_pct_default_is_full(self):
        cfg = SimpleNamespace(hidden_size=64, num_attention_heads=4)
        c = rope_config_from_model(SimpleNamespace(config=cfg))
        assert c["rotary_ndims"] == c["head_size"]

    def test_odd_rotary_ndims_rejected(self):
        with pytest.raises(ValueError):
            rope_config_from_model(self._fake(hidden_size=48, num_attention_heads=8,
                                              rotary_pct=0.25))

    def test_energy_fraction(self):
        assert rope_energy_fraction(128, 32) == pytest.approx(0.25)
        assert rope_energy_fraction(64, 0) == 0.0

    def test_model_uses_rope_by_name(self):
        assert model_uses_rope("EleutherAI/pythia-1.4b")
        assert model_uses_rope("pythia-1.4b-step1000")
        assert not model_uses_rope("gpt2-large")
        assert not model_uses_rope("albert-base-v2")

    def test_model_uses_rope_by_config(self):
        assert model_uses_rope(self._fake())
        plain = SimpleNamespace(config=SimpleNamespace(_name_or_path="gpt2-large"))
        assert not model_uses_rope(plain)
