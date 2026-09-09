"""
tests/test_phase2_weights_gptneox.py — GPT-NeoX branch of
p2_eigenspectra/weights.py (transition plan v2, item 5).

Oracle-style, torch-free: builds a fake GPT-NeoX model whose weights are
plain numpy arrays wrapped to answer .weight.detach().cpu().float().numpy(),
then checks the extracted per-head OV / QK matrices against a brute-force
numpy simulation of the actual NeoX forward math:

    qkv   = x @ W_qkv.T            (fused Linear)
    per-head interleaved [Q_h|K_h|V_h] slicing (what GPTNeoXAttention's
    .view(..., num_heads, 3*head_size) does)
    out   = concat_h(v_h) @ W_dense.T

If Σ_h x @ OV_h from the extraction equals the simulated value-pathway
output, the head slicing, the fused-layout handling, and the Linear
orientation are all right at once. This is the same "proven special case"
logic as tests/test_core_oracle.py, applied to the extraction layer.

Relies on conftest.py's stub session for core.config (MODEL_CONFIGS);
core.pythia_weights is imported lazily inside the functions under test,
and its only torch use on ndarray input is torch.is_tensor — covered by
the stub torch these tests install if none is present.
"""
import sys
import types

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure


# ---------------------------------------------------------------------------
# Minimal torch stub (only if real torch is absent) — core.pythia_weights
# calls torch.is_tensor on ndarray input; nothing else on this path.
# ---------------------------------------------------------------------------

def _ensure_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch")
        stub.is_tensor = lambda x: False
        sys.modules.setdefault("torch", stub)


_ensure_torch()


# ---------------------------------------------------------------------------
# Fake-model scaffolding
# ---------------------------------------------------------------------------

class _W:
    """Mimic a torch Parameter far enough for .detach().cpu().float().numpy()
    and numpy fallthrough (np.asarray) both to work."""
    def __init__(self, arr):
        self._a = np.asarray(arr, dtype=np.float32)
        self.shape = self._a.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._a

    def __array__(self, dtype=None):
        return self._a if dtype is None else self._a.astype(dtype)


class _Module:
    def __init__(self, weight):
        self.weight = _W(weight)


class _Attn:
    def __init__(self, w_qkv, w_dense, n_heads, head_size):
        self.query_key_value = _Module(w_qkv)
        self.dense = _Module(w_dense)
        self.num_attention_heads = n_heads
        self.head_size = head_size


class _Layer:
    def __init__(self, attn):
        self.attention = attn


class _FakeNeoX:
    """Bare GPTNeoXModel shape: blocks at model.layers."""
    def __init__(self, layers):
        self.layers = layers


class _FakeNeoXWrapped:
    """GPTNeoXForCausalLM shape: blocks at model.gpt_neox.layers."""
    def __init__(self, layers):
        self.gpt_neox = _FakeNeoX(layers)


def _build_fused_qkv(rng, n_heads, head_size, d_model):
    """Fused weight with the real interleaved-per-head layout, plus the
    ground-truth per-head Q/K/V blocks it was built from."""
    Q = rng.standard_normal((n_heads, head_size, d_model)).astype(np.float32)
    K = rng.standard_normal((n_heads, head_size, d_model)).astype(np.float32)
    V = rng.standard_normal((n_heads, head_size, d_model)).astype(np.float32)
    # [Q_h | K_h | V_h] contiguous per head
    fused = np.concatenate(
        [np.concatenate([Q[h], K[h], V[h]], axis=0) for h in range(n_heads)],
        axis=0,
    )  # (n_heads*3*head_size, d_model)
    return fused, Q, K, V


def _make_model(seed=0, n_layers=2, n_heads=3, head_size=4, wrapped=False):
    rng = np.random.default_rng(seed)
    d_model = n_heads * head_size
    layers, truth = [], []
    for _ in range(n_layers):
        fused, Q, K, V = _build_fused_qkv(rng, n_heads, head_size, d_model)
        w_dense = rng.standard_normal((d_model, d_model)).astype(np.float32)
        layers.append(_Layer(_Attn(fused, w_dense, n_heads, head_size)))
        truth.append({"Q": Q, "K": K, "V": V, "dense": w_dense})
    model = _FakeNeoXWrapped(layers) if wrapped else _FakeNeoX(layers)
    return model, truth, d_model, n_heads, head_size


def _value_pathway_bruteforce(x, layer_truth):
    """out = concat_h(x @ V_h.T) @ W_dense.T — the real NeoX value path."""
    V, W_dense = layer_truth["V"], layer_truth["dense"]
    v_cat = np.concatenate([x @ V[h].T for h in range(V.shape[0])], axis=1)
    return v_cat @ W_dense.T


# ---------------------------------------------------------------------------
# _detect_model_type
# ---------------------------------------------------------------------------

class TestDetectModelType:
    def test_bare_gptneox_detected(self):
        from p2_eigenspectra.weights import _detect_model_type
        model, *_ = _make_model()
        assert _detect_model_type(model) == "gptneox"

    def test_wrapped_gptneox_detected(self):
        from p2_eigenspectra.weights import _detect_model_type
        model, *_ = _make_model(wrapped=True)
        assert _detect_model_type(model) == "gptneox"

    def test_unrecognised_still_raises(self):
        from p2_eigenspectra.weights import _detect_model_type
        with pytest.raises(ValueError):
            _detect_model_type(object())


# ---------------------------------------------------------------------------
# OV extraction — exactness oracle
# ---------------------------------------------------------------------------

class TestGptneoxOV:
    def _extract(self, model):
        from p2_eigenspectra.weights import _extract_gptneox_ov
        return _extract_gptneox_ov(model, "pythia-test")

    def test_shapes_and_metadata(self):
        model, _, d_model, n_heads, head_size = _make_model(n_layers=2)
        ov = self._extract(model)
        assert ov["is_per_layer"] is True
        assert ov["n_heads"] == n_heads
        assert ov["d_head"] == head_size
        assert ov["d_model"] == d_model
        assert ov["layer_names"] == ["layer_0", "layer_1"]
        assert len(ov["ov_per_head"]) == 2
        assert len(ov["ov_per_head"][0]) == n_heads
        assert ov["ov_per_head"][0][0].shape == (d_model, d_model)

    def test_ov_sum_matches_bruteforce_value_pathway(self):
        """Σ_h x @ OV_h == the simulated NeoX value-pathway output —
        head slicing, interleave handling, and Linear orientation all
        verified at once."""
        model, truth, d_model, n_heads, _ = _make_model(seed=7)
        ov = self._extract(model)
        rng = np.random.default_rng(1)
        x = rng.standard_normal((5, d_model)).astype(np.float32)
        for L, layer_truth in enumerate(truth):
            expected = _value_pathway_bruteforce(x, layer_truth)
            got = sum(x @ ov["ov_per_head"][L][h] for h in range(n_heads))
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_per_head_ov_matches_single_head_bruteforce(self):
        """Each OV_h individually equals that head's isolated contribution
        (all other heads' V rows zeroed) — not just the sum."""
        model, truth, d_model, n_heads, head_size = _make_model(seed=3, n_layers=1)
        ov = self._extract(model)
        rng = np.random.default_rng(2)
        x = rng.standard_normal((4, d_model)).astype(np.float32)
        lt = truth[0]
        for h in range(n_heads):
            v_h = x @ lt["V"][h].T                       # (n, head_size)
            s, e = h * head_size, (h + 1) * head_size
            expected = v_h @ lt["dense"][:, s:e].T        # (n, d_model)
            got = x @ ov["ov_per_head"][0][h]
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_ov_total_is_sum_of_heads(self):
        model, _, _, n_heads, _ = _make_model(seed=5, n_layers=1)
        ov = self._extract(model)
        np.testing.assert_allclose(
            ov["ov_total"][0],
            sum(ov["ov_per_head"][0][h] for h in range(n_heads)),
            rtol=1e-5,
        )

    def test_wrapped_and_bare_agree(self):
        bare, _, _, _, _ = _make_model(seed=11, wrapped=False)
        wrap, _, _, _, _ = _make_model(seed=11, wrapped=True)
        ov_b = self._extract(bare)
        ov_w = self._extract(wrap)
        np.testing.assert_allclose(ov_b["ov_total"][0], ov_w["ov_total"][0])


# ---------------------------------------------------------------------------
# QK extraction
# ---------------------------------------------------------------------------

class TestGptneoxQK:
    def test_qk_spectral_norms_match_truth(self):
        from p2_eigenspectra.weights import _qk_gptneox
        from scipy.linalg import svdvals
        model, truth, _, n_heads, _ = _make_model(seed=9, n_layers=2)
        out = _qk_gptneox(model)
        assert out["layer_names"] == ["layer_0", "layer_1"]
        for L, lt in enumerate(truth):
            for h in range(n_heads):
                expected = float(svdvals(lt["Q"][h] @ lt["K"][h].T)[0])
                assert out["qk_spectral_norms"][L][h] == pytest.approx(
                    expected, rel=1e-4)

    def test_extract_qk_per_head_orientation_and_bilinear(self):
        """Canonical (d_model, d_head) orientation: x_i^T (WQ_h WK_h^T) x_j
        must equal q_i · k_j from the simulated forward — the exact
        contract extract_qk_per_head's docstring states."""
        from p2_eigenspectra.weights import extract_qk_per_head
        model, truth, d_model, n_heads, head_size = _make_model(seed=13, n_layers=1)
        out = extract_qk_per_head(model, model_type="gptneox")
        assert out["is_per_layer"] is True
        rng = np.random.default_rng(4)
        xi = rng.standard_normal(d_model).astype(np.float32)
        xj = rng.standard_normal(d_model).astype(np.float32)
        lt = truth[0]
        for h in range(n_heads):
            WQ_h = out["wq_per_head"][0][h]   # (d_model, d_head)
            WK_h = out["wk_per_head"][0][h]
            assert WQ_h.shape == (d_model, head_size)
            bilinear = xi @ (WQ_h @ WK_h.T) @ xj
            q_i = lt["Q"][h] @ xi
            k_j = lt["K"][h] @ xj
            assert bilinear == pytest.approx(float(q_i @ k_j), rel=1e-3)


# ---------------------------------------------------------------------------
# Name-based dispatch guard
# ---------------------------------------------------------------------------

class TestNameDispatch:
    @pytest.mark.parametrize("name", [
        "pythia-410m-step512", "pythia-1.4b-step143000", "pythia-1.4b-random",
        "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    ])
    def test_registry_and_smoke_names_route_to_gptneox(self, name):
        from p2_eigenspectra.weights import _is_gptneox_name
        assert _is_gptneox_name(name)

    @pytest.mark.parametrize("name", [
        "gpt2-large", "bert-base-uncased", "albert-xlarge-v2",
    ])
    def test_existing_names_do_not_route_to_gptneox(self, name):
        from p2_eigenspectra.weights import _is_gptneox_name
        assert not _is_gptneox_name(name)


# ---------------------------------------------------------------------------
# The precision the projectors are decomposed at, which Phase 7 depends on
# ---------------------------------------------------------------------------

class TestDecompositionPrecision:
    """
    Phase 2's projectors are consumed by `core.interactions._as_basis`,
    which refuses a square matrix that is not a symmetric idempotent
    projector to within PROJECTOR_TOL. Single-precision Schur does not
    reach that: measured on pythia-410m layer 9, float32 gives Schur
    vectors orthogonal only to 1.5e-05 and a projector at ||P@P-P|| ~ 6e-06
    against a 1e-6 tolerance.

    It did not fail uniformly, which is why it survived. `np.allclose`
    carries rtol=1e-5 beside the atol, so whether a ~6e-06 residual passed
    depended on where in the matrix it landed — Phase 7 refused step 2 of
    the registered sweep and accepted steps 1 and 4 on projectors that were
    equally non-idempotent.
    """

    def _projector(self, dtype):
        from scipy.linalg import schur
        rng = np.random.default_rng(0)
        d = 128
        A = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(dtype)
        _, Z = schur(A, output="real")
        Zs = Z[:, : d // 2]
        return Zs @ Zs.T

    def test_float32_schur_does_not_reach_the_projector_tolerance(self):
        """The defect itself. If this ever passes, single precision became
        good enough and the promotion below is no longer load-bearing —
        which is worth being told about explicitly."""
        from core.interactions import PROJECTOR_TOL
        P = self._projector(np.float32)
        assert np.abs(P @ P - P).max() > PROJECTOR_TOL

    def test_float64_schur_clears_it_by_orders_of_magnitude(self):
        from core.interactions import PROJECTOR_TOL
        P = self._projector(np.float64)
        assert np.abs(P @ P - P).max() < PROJECTOR_TOL / 1000

    def test_extraction_promotes_every_float_array(self):
        """The promotion happens at the one point every model type passes
        through, so a new `_extract_*` branch cannot miss it."""
        from p2_eigenspectra.weights import _as_float64
        out = _as_float64({
            "ov_total": [np.zeros((4, 4), dtype=np.float32)],
            "ov_per_head": [[np.zeros((4, 4), dtype=np.float32)]],
            "ov_head_core": np.zeros((2, 2, 2), dtype=np.float32),
            "n_heads": 2,
            "layer_names": ["layer_0"],
        })
        assert out["ov_total"][0].dtype == np.float64
        assert out["ov_per_head"][0][0].dtype == np.float64
        assert out["ov_head_core"].dtype == np.float64
        assert out["n_heads"] == 2 and out["layer_names"] == ["layer_0"]

    def test_a_float64_projector_is_accepted_by_the_phase_7_validator(self):
        """The end-to-end contract, stated as the consumer sees it."""
        from core.interactions import _as_basis
        P = self._projector(np.float64)
        assert _as_basis(P, P.shape[0]) is P

    def test_float32_leaves_acceptance_to_rtol_rather_than_guaranteeing_it(self):
        """Why the failure was arbitrary, stated as the property it is.

        `_as_basis` tests `np.allclose(P @ P, P, atol=PROJECTOR_TOL)`, and
        allclose carries rtol=1e-5 as well — so a residual ABOVE atol is
        accepted or refused depending on where it sits relative to |P|.
        float32 lands in exactly that band and float64 does not, which is
        the whole difference: one is decided by placement, the other by
        margin. Asserting a flat refusal for float32 would be wrong — at
        d=128 it is accepted, at d=1024 on real weights step 2 was not.
        """
        from core.interactions import PROJECTOR_TOL
        f32 = np.abs(np.subtract(*(lambda P: (P @ P, P))(self._projector(np.float32)))).max()
        f64 = np.abs(np.subtract(*(lambda P: (P @ P, P))(self._projector(np.float64)))).max()
        assert f32 > PROJECTOR_TOL          # in the band rtol decides
        assert f64 < PROJECTOR_TOL / 1000   # decided by margin, not placement
        assert f32 / f64 > 1e6

