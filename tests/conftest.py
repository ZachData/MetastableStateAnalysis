"""
conftest.py — session-wide stubs and shared fixtures.

Installs lightweight stubs for torch, transformers, and core.* into
sys.modules before any test module is collected, so pure numpy/scipy
computation logic can run without a GPU, model weights, or a full
transformers installation.

Also exposes session-scoped geometric and tracking fixtures used across
phase-1 and phase-4 test suites.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock
import torch

import numpy as np
import pytest

from tests.config import D, N_LAYERS, N_TOKENS

# ===========================================================================
# 1. Heavy-dependency stubs
#    Must run before any project import so sys.modules is populated first.
# ===========================================================================

def _install_stubs() -> None:
    """Install torch, transformers, and core.* stubs exactly once."""

    # --- torch ---
    _torch = MagicMock()
    _torch.cuda.is_available.return_value = False
    sys.modules.setdefault("torch", _torch)

    # --- transformers ---
    for _mod in ("transformers", "transformers.models", "transformers.models.albert"):
        sys.modules.setdefault(_mod, MagicMock())

    # --- core.config ---
    _core = types.ModuleType("core")
    _cfg = types.ModuleType("core.config")

    _cfg.BETA_VALUES         = [0.1, 1.0, 2.0, 5.0]
    _cfg.MODEL_CONFIGS       = {}
    _cfg.BASE_RESULTS_DIR    = None
    _cfg.DEVICE              = "cpu"
    _cfg.ALBERT_MAX_ITERATIONS = 48
    _cfg.ALBERT_SNAPSHOTS    = [12, 24, 36, 48]
    _cfg.SINKHORN_MAX_ITER   = 100
    _cfg.SINKHORN_TOL        = 1e-6
    _cfg.SPECTRAL_MAX_K      = 15
    _cfg.DISTANCE_THRESHOLDS = np.linspace(0.05, 0.6, 12)
    _cfg.K_RANGE             = range(2, 10)
    _cfg.PROMPTS = {
        "short_heterogeneous": (
            "Quantum mechanics governs the behavior of subatomic particles. "
            "Meanwhile, the stock market closed higher on Friday."
        ),
        "wiki_paragraph": (
            "Charlotte Nicholls (née Brontë; 21 April 1816 – 31 March 1855), "
            "commonly known by her maiden name Charlotte Brontë, was an English "
            "novelist and poet."
        ),
    }

    # --- core.models ---
    _models = types.ModuleType("core.models")
    _models.extract_activations       = MagicMock()
    _models.extract_albert_extended   = MagicMock()
    _models.load_model                = MagicMock()

    def _real_layernorm_to_sphere(activation):
        import torch.nn.functional as F  # resolves to real torch at call-time
        return F.normalize(activation.float(), p=2, dim=-1)

    _models.layernorm_to_sphere = _real_layernorm_to_sphere

    sys.modules.setdefault("core",         _core)
    sys.modules.setdefault("core.config",  _cfg)
    sys.modules.setdefault("core.models",  _models)


# ===========================================================================
# 2. p4_mstate_features package bootstrap
# ===========================================================================

_P4_SRC = Path(__file__).parent.parent  # project root


def _ensure_p4_package() -> None:
    """Register p4_mstate_features as a package in sys.modules if absent."""
    if "p4_mstate_features" in sys.modules:
        return
    pkg = types.ModuleType("p4_mstate_features")
    pkg.__path__    = [str(_P4_SRC)]
    pkg.__package__ = "p4_mstate_features"
    sys.modules["p4_mstate_features"] = pkg


def _load_p4_submodule(filename: str) -> types.ModuleType:
    """Load a file from the project root as p4_mstate_features.<stem>."""
    stem      = Path(filename).stem
    full_name = f"p4_mstate_features.{stem}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    filepath = _P4_SRC / filename
    spec     = importlib.util.spec_from_file_location(full_name, filepath)
    mod      = importlib.util.module_from_spec(spec)
    mod.__package__       = "p4_mstate_features"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Run at collection time
# ---------------------------------------------------------------------------

_install_stubs()
_ensure_p4_package()

# Pre-load p4 modules; order matters — chorus imports from activation_trajectories.
_load_p4_submodule("p4_mstate_features/activation_trajectories.py")
_load_p4_submodule("p4_mstate_features/chorus.py")
_load_p4_submodule("p4_mstate_features/geometric.py")


# ===========================================================================
# 3. Shared fixtures
# ===========================================================================

_rng = np.random.default_rng(42)  # fixed seed → deterministic across runs


# ---------------------------------------------------------------------------
# Helpers (not fixtures)
# ---------------------------------------------------------------------------

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / norms).astype(np.float32)


def _make_results(label_list_per_layer: list) -> dict:
    """
    Build a minimal results dict in the shape that track_clusters reads:
        results["layers"][i]["clustering"]["hdbscan"]["labels"]
    """
    return {
        "layers": [
            {"clustering": {"hdbscan": {"labels": list(labels)}}}
            for labels in label_list_per_layer
        ]
    }


# ---------------------------------------------------------------------------
# Activation geometry fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def antipodal_normed() -> np.ndarray:
    """
    Two tight antipodal clusters on S^{d-1}.

    Half the tokens near +e₁, half near −e₁, with σ=0.05 isotropic noise
    before normalisation.  Within-cluster inner products ≈ +1;
    between-cluster ≈ −1; effective rank ≈ 2.
    """
    half = N_TOKENS // 2
    X    = np.zeros((N_TOKENS, D), dtype=np.float32)
    X[:half, 0] =  1.0
    X[half:, 0] = -1.0
    noise = _rng.standard_normal((N_TOKENS, D)).astype(np.float32) * 0.05
    return _l2_normalize(X + noise)


@pytest.fixture(scope="session")
def uniform_normed() -> np.ndarray:
    """
    Uniform spread on S^{d-1}: i.i.d. Gaussian vectors, L2-normalised.

    In the d → ∞ limit, ⟨xᵢ, xⱼ⟩ ≈ 0 for i ≠ j; effective rank ≈ d.
    """
    X = _rng.standard_normal((N_TOKENS, D)).astype(np.float32)
    return _l2_normalize(X)


@pytest.fixture(scope="session")
def collapsed_normed() -> np.ndarray:
    """Single tight cluster: all tokens concentrated near +e₁."""
    X     = np.zeros((N_TOKENS, D), dtype=np.float32)
    X[:, 0] = 1.0
    noise = _rng.standard_normal((N_TOKENS, D)).astype(np.float32) * 0.05
    return _l2_normalize(X + noise)


import torch  # add to conftest imports if not already present

# ---------------------------------------------------------------------------
# Gram matrix fixtures
# (add after the "Activation geometry fixtures" block)
#
# Use float64 so Laplacian eigenvalues stay above -1e-10.
# float32 gram matrices produce numerical noise at ~-7e-8, which fails the
# test_eigenvalues_non_negative tolerance.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def antipodal_gram(antipodal_normed: np.ndarray) -> np.ndarray:
    X = antipodal_normed.astype(np.float64)
    return (X @ X.T)


@pytest.fixture(scope="session")
def uniform_gram(uniform_normed: np.ndarray) -> np.ndarray:
    X = uniform_normed.astype(np.float64)
    return (X @ X.T)


@pytest.fixture(scope="session")
def collapsed_gram(collapsed_normed: np.ndarray) -> np.ndarray:
    X = collapsed_normed.astype(np.float64)
    return (X @ X.T)


# ---------------------------------------------------------------------------
# Effective-rank tensor fixtures
# (used by TestEffectiveRank in test_phase1_metrics.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rank1_tensor() -> "torch.Tensor":
    """
    (N_TOKENS, D) tensor where every row is the same unit vector.
    Rank = 1 → only one non-zero singular value → effective_rank = 1.
    """
    v = torch.zeros(D, dtype=torch.float32)
    v[0] = 1.0
    return v.unsqueeze(0).expand(N_TOKENS, -1).clone()


@pytest.fixture(scope="session")
def uniform_sv_tensor() -> "torch.Tensor":
    """
    (N_TOKENS, D) tensor with D equal singular values (all = 1).
    QR decomposition of a random matrix gives an orthonormal column basis.
    Entropy = log(D) → effective_rank = D.
    """
    rng = np.random.default_rng(99)
    X   = rng.standard_normal((N_TOKENS, D)).astype(np.float32)
    Q, _ = np.linalg.qr(X)   # reduced QR: Q is (N_TOKENS, D), orthonormal cols
    return torch.from_numpy(Q)


# ---------------------------------------------------------------------------
# Attention matrix fixtures
# (used by TestAttentionEntropy and TestAnalyzeAttentionSinkhorn)
# ---------------------------------------------------------------------------

_N_HEADS = 4  # matches the head count assumed by the attention tests


@pytest.fixture(scope="session")
def uniform_attention() -> "torch.Tensor":
    """
    (n_heads, N_TOKENS, N_TOKENS) torch.Tensor.
    Every entry = 1/N_TOKENS → each row sums to 1, doubly stochastic.
    Shannon entropy per row = log(N_TOKENS).

    analyze_attention_sinkhorn and attention_entropy both call .numpy() on
    their input, so the fixture must be a torch.Tensor, not a numpy array.
    """
    arr = np.full((_N_HEADS, N_TOKENS, N_TOKENS), 1.0 / N_TOKENS, dtype=np.float32)
    return torch.from_numpy(arr)


@pytest.fixture(scope="session")
def identity_attention() -> "torch.Tensor":
    """
    (n_heads, N_TOKENS, N_TOKENS) torch.Tensor.
    Each head is the identity matrix → each token attends only to itself.
    Shannon entropy per row = 0.
    """
    eye = np.eye(N_TOKENS, dtype=np.float32)
    return torch.from_numpy(np.stack([eye] * _N_HEADS))

# ---------------------------------------------------------------------------
# Gram matrix fixtures
# (add after the existing "Activation geometry fixtures" block in conftest.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def antipodal_gram(antipodal_normed: np.ndarray) -> np.ndarray:
    X = antipodal_normed.astype(np.float64)
    return (X @ X.T)


@pytest.fixture(scope="session")
def uniform_gram(uniform_normed: np.ndarray) -> np.ndarray:
    X = uniform_normed.astype(np.float64)
    return (X @ X.T)


@pytest.fixture(scope="session")
def collapsed_gram(collapsed_normed: np.ndarray) -> np.ndarray:
    X = collapsed_normed.astype(np.float64)
    return (X @ X.T)

# ---------------------------------------------------------------------------
# Cluster-tracking fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def stable_tracking_results() -> dict:
    """
    Six identical layers: 20 tokens in cluster 0, 20 in cluster 1.
    Expected: no births, deaths, or merges across any transition.
    """
    labels = [0] * (N_TOKENS // 2) + [1] * (N_TOKENS // 2)
    return _make_results([labels] * N_LAYERS)


@pytest.fixture(scope="session")
def one_merge_tracking_results() -> dict:
    """
    Layers 0–2: two clusters.  Layer 3+: single cluster.

    At the layer-2 → layer-3 transition the Hungarian algorithm matches
    prev-cluster-0 → curr-cluster-0; the merge-detection loop then finds
    prev-cluster-1 also overlapping curr-cluster-0 and records one merge.

    Expected: summary["total_merges"] == 1.
    """
    two_clusters = [0] * (N_TOKENS // 2) + [1] * (N_TOKENS // 2)
    one_cluster  = [0] * N_TOKENS
    return _make_results([two_clusters] * 3 + [one_cluster] * 3)