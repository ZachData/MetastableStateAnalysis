"""
conftest.py — Shared synthetic fixtures for phase-1 pure-computation tests.

No model loading.  All fixtures produce deterministic numpy/torch data that
satisfy known analytical properties, documented inline.

Geometry
--------
All token vectors live on S^{d-1} (unit sphere in R^d).  Fixtures are built
as L2-normalised float32 ndarrays; torch.Tensor wrappers are added where
required by function signatures.

Shape constants used throughout: n_layers=6, n_tokens=40, d=16.
"""
from __future__ import annotations

import numpy as np
import torch
import pytest

from tests.config import N_LAYERS, N_TOKENS, D

_rng = np.random.default_rng(42)   # fixed seed → deterministic across runs



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Activation geometry fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def antipodal_normed() -> np.ndarray:
    """
    Two tight antipodal clusters on S^{d-1}.

    Construction
    ------------
    Half the tokens are concentrated near the north pole +e₁; the other half
    near the south pole −e₁.  Small isotropic Gaussian noise (σ=0.05) is
    added before normalisation.

    Analytical properties (noise → 0 limit)
    ----------------------------------------
    Within-cluster inner products ⟨xᵢ, xⱼ⟩  ≈ +1
    Between-cluster inner products            ≈ −1
    Effective rank                            ≈  2  (two dominant directions)
    Interaction energy E_β                    > E_β(uniform)  for any β>0
      (cosh(β)/(2β) vs 1/(2β), since cosh(β) > 1)
    """
    half  = N_TOKENS // 2
    X     = np.zeros((N_TOKENS, D), dtype=np.float32)
    X[:half, 0]  =  1.0
    X[half:, 0]  = -1.0
    noise = _rng.standard_normal((N_TOKENS, D)).astype(np.float32) * 0.05
    return _l2_normalize(X + noise)


@pytest.fixture(scope="session")
def uniform_normed() -> np.ndarray:
    """
    Uniform spread on S^{d-1}: i.i.d. Gaussian vectors, L2-normalised.

    Analytical properties (d → ∞ limit)
    -------------------------------------
    ⟨xᵢ, xⱼ⟩  ≈ 0  for i ≠ j
    Effective rank ≈ d   (spectrum is flat)
    Interaction energy E_β ≈ 1/(2β)  (off-diagonal exp terms average to 1)
    """
    X = _rng.standard_normal((N_TOKENS, D)).astype(np.float32)
    return _l2_normalize(X)


@pytest.fixture(scope="session")
def collapsed_normed() -> np.ndarray:
    """
    Single tight cluster: all tokens concentrated near +e₁.

    Analytical properties (noise → 0 limit)
    ----------------------------------------
    All inner products ≈ +1
    Effective rank ≈ 1
    Interaction energy E_β = exp(β)/(2β)  — highest of the three geometries
      because exp(β) > cosh(β) > 1.
    """
    X     = np.zeros((N_TOKENS, D), dtype=np.float32)
    X[:, 0] = 1.0
    noise = _rng.standard_normal((N_TOKENS, D)).astype(np.float32) * 0.001
    return _l2_normalize(X + noise)


# ---------------------------------------------------------------------------
# Gram matrices (pre-computed once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def antipodal_gram(antipodal_normed) -> np.ndarray:
    """(n_tokens, n_tokens) float64 Gram matrix for antipodal activations."""
    return (antipodal_normed.astype(np.float64) @
            antipodal_normed.astype(np.float64).T)


@pytest.fixture(scope="session")
def uniform_gram(uniform_normed) -> np.ndarray:
    return (uniform_normed.astype(np.float64) @
            uniform_normed.astype(np.float64).T)


@pytest.fixture(scope="session")
def collapsed_gram(collapsed_normed) -> np.ndarray:
    return (collapsed_normed.astype(np.float64) @
            collapsed_normed.astype(np.float64).T)


# ---------------------------------------------------------------------------
# Synthetic activation tensors for effective-rank tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rank1_tensor() -> torch.Tensor:
    """
    (N_TOKENS, D) float32 tensor of rank exactly 1.

    All rows are identical (the all-ones vector), so only one singular value
    is non-zero.  effective_rank_from_raw must return 1.0.
    """
    v = torch.ones(N_TOKENS, 1, dtype=torch.float32)
    w = torch.ones(1, D,       dtype=torch.float32)
    return v @ w   # (40, 16), rank 1


@pytest.fixture(scope="session")
def uniform_sv_tensor() -> torch.Tensor:
    """
    (N_TOKENS, D) float32 tensor whose singular values are all equal to 1.

    Construction: draw a random (N_TOKENS, D) Gaussian matrix, compute its
    compact SVD, replace the singular-value diagonal with ones, reconstruct.
    By construction svdvals = [1]*D, so effective_rank = exp(H([1/D]*D)) = D.
    """
    rng_fixed = np.random.default_rng(0)
    A = rng_fixed.standard_normal((N_TOKENS, D)).astype(np.float64)
    U, _, Vh = np.linalg.svd(A, full_matrices=False)  # U: (40,16), Vh: (16,16)
    # Replace singular values with ones: X = U @ I @ Vh
    X = (U @ Vh).astype(np.float32)
    return torch.tensor(X)


# ---------------------------------------------------------------------------
# Attention tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def uniform_attention() -> torch.Tensor:
    """
    (n_heads=4, N_TOKENS, N_TOKENS) doubly stochastic attention:
    every entry = 1/N_TOKENS.

    Row sums = col sums = 1  (doubly stochastic).
    Shannon entropy per row = log(N_TOKENS)  — maximum possible.
    row_col_balance (std of column sums) = 0.
    """
    n_heads = 4
    val = 1.0 / N_TOKENS
    return torch.full((n_heads, N_TOKENS, N_TOKENS), val, dtype=torch.float32)


@pytest.fixture(scope="session")
def identity_attention() -> torch.Tensor:
    """
    (n_heads=4, N_TOKENS, N_TOKENS) attention: each token attends only to itself.

    Row sums = col sums = 1  (doubly stochastic).
    Shannon entropy per row = 0  — minimum possible.
    """
    n_heads = 4
    attn    = torch.zeros(n_heads, N_TOKENS, N_TOKENS, dtype=torch.float32)
    idx     = torch.arange(N_TOKENS)
    attn[:, idx, idx] = 1.0
    return attn


# ---------------------------------------------------------------------------
# Cluster-tracking results dicts
# ---------------------------------------------------------------------------

def _make_results(label_list_per_layer):
    """
    Build the results dict expected by track_clusters from a plain list of
    per-layer label arrays.

    track_clusters reads: results["layers"][i]["clustering"]["hdbscan"]["labels"]
    """
    return {
        "layers": [
            {"clustering": {"hdbscan": {"labels": list(labels)}}}
            for labels in label_list_per_layer
        ]
    }


@pytest.fixture(scope="session")
def stable_tracking_results():
    """
    Six layers with identical cluster assignments (20 tokens in cluster 0,
    20 in cluster 1).

    Expected: no births, no deaths, no merges across any transition.
    """
    labels = [0] * (N_TOKENS // 2) + [1] * (N_TOKENS // 2)
    return _make_results([labels] * N_LAYERS)


@pytest.fixture(scope="session")
def one_merge_tracking_results():
    """
    Layers 0-2: two clusters (0→tokens 0-19, 1→tokens 20-39).
    Layer  3:   single cluster (0→all 40 tokens).  ← merge event here.
    Layers 4-5: single cluster persists.

    At the layer-2 → layer-3 transition, match_layer_pair sees:
      overlap(prev=0, curr=0) = 20/40 = 0.5
      overlap(prev=1, curr=0) = 20/40 = 0.5
    The Hungarian algorithm matches prev-cluster-0 to curr-cluster-0; then
    the merge-detection loop finds prev-cluster-1 also overlaps curr-cluster-0
    and records exactly ONE merge event.

    Expected: summary["total_merges"] == 1.
    """
    two_clusters = [0] * (N_TOKENS // 2) + [1] * (N_TOKENS // 2)
    one_cluster  = [0] * N_TOKENS
    layers = (
        [two_clusters] * 3
        + [one_cluster] * 3
    )
    return _make_results(layers)



"""
conftest.py — stub heavy deps so pytest collects phase1h tests
without a GPU, torch installation, or transformers.

Place this file at tests/phase1h/conftest.py (or phase1h/conftest.py
if running tests from inside the phase1h source directory).
"""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. torch stub
# ---------------------------------------------------------------------------
_torch = MagicMock()
_torch.cuda.is_available.return_value = False
sys.modules.setdefault("torch", _torch)

# ---------------------------------------------------------------------------
# 2. transformers stub
# ---------------------------------------------------------------------------
for _mod in [
    "transformers",
    "transformers.models",
    "transformers.models.albert",
]:
    sys.modules.setdefault(_mod, MagicMock())

# ---------------------------------------------------------------------------
# 3. Minimal core.config stub
#    Avoids the torch.cuda call and model-class imports in the real module.
# ---------------------------------------------------------------------------
_core = types.ModuleType("core")
_core_config = types.ModuleType("core.config")
_core_config.BETA_VALUES = [0.1, 1.0, 2.0, 5.0]
_core_config.MODEL_CONFIGS = {}
_core_config.BASE_RESULTS_DIR = None
_core_config.DEVICE = "cpu"
_core_config.ALBERT_MAX_ITERATIONS = 48
_core_config.ALBERT_SNAPSHOTS = [12, 24, 36, 48]
_core_config.PROMPTS = {}
_core_config.SINKHORN_MAX_ITER = 100
_core_config.SINKHORN_TOL = 1e-6
_core_config.SPECTRAL_MAX_K = 15

#fix added later for import consts:
_core_config.ALBERT_MAX_ITERATIONS = 48
_core_config.ALBERT_SNAPSHOTS     = [12, 24, 36, 48]
_core_config.PROMPTS               = {}
_core_config.SINKHORN_MAX_ITER     = 100
_core_config.SINKHORN_TOL          = 1e-6
_core_config.SPECTRAL_MAX_K        = 15
_core_config.DISTANCE_THRESHOLDS   = np.linspace(0.05, 0.6, 12)
_core_config.K_RANGE               = range(2, 10)

_core_models = types.ModuleType("core.models")
_core_models.extract_activations = MagicMock()
_core_models.extract_albert_extended = MagicMock()
_core_models.layernorm_to_sphere = MagicMock()
_core_models.load_model = MagicMock()

sys.modules.setdefault("core", _core)
sys.modules.setdefault("core.config", _core_config)
sys.modules.setdefault("core.models", _core_models)


"""
conftest.py — pytest session-scoped import stubs.

eigendecompose / build_subspace_projectors / rescale_matrix and the
trajectory helpers are pure numpy/scipy.  They don't call torch or
transformers at test-time, but those packages are imported at the
module level in core/config.py, which is itself imported at the top
of weights.py and trajectory.py.

Stub every heavy dep in sys.modules *before* the first project import
so pytest can collect and run tests without a GPU or model weights.
"""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Stub torch so core/config.py doesn't need a real installation
# ---------------------------------------------------------------------------
_torch = MagicMock()
_torch.cuda.is_available.return_value = False
sys.modules.setdefault("torch", _torch)

# ---------------------------------------------------------------------------
# 2. Stub transformers (model classes imported at top of config.py)
# ---------------------------------------------------------------------------
for _mod in [
    "transformers",
    "transformers.models",
    "transformers.models.albert",
]:
    sys.modules.setdefault(_mod, MagicMock())

# ---------------------------------------------------------------------------
# 3. Inject a minimal core.config so the real module never runs
#    (avoids the torch.cuda call and model-class imports)
# ---------------------------------------------------------------------------
_core = types.ModuleType("core")
_core_config = types.ModuleType("core.config")
_core_config.BETA_VALUES = [0.1, 1.0, 2.0, 5.0]
_core_config.MODEL_CONFIGS = {}
_core_config.BASE_RESULTS_DIR = None
_core_config.DEVICE = "cpu"
#fix added later for import consts:
_core_config.ALBERT_MAX_ITERATIONS = 48
_core_config.ALBERT_SNAPSHOTS     = [12, 24, 36, 48]
_core_config.PROMPTS               = {}
_core_config.SINKHORN_MAX_ITER     = 100
_core_config.SINKHORN_TOL          = 1e-6
_core_config.SPECTRAL_MAX_K        = 15
_core_config.DISTANCE_THRESHOLDS   = np.linspace(0.05, 0.6, 12)
_core_config.K_RANGE               = range(2, 10)


sys.modules["core"] = _core
sys.modules["core.config"] = _core_config


"""
conftest.py — bootstrap stubs required by p4_mstate_features modules.

activation_trajectories.py and chorus.py both import torch (unavailable in the
test environment) and phase3.{crosscoder,data} (require a live model checkpoint).
This file installs minimal stubs into sys.modules before any test module is
collected, so the pure-numpy/scipy computation logic can be imported and tested
without those dependencies.

The stub must satisfy three constraints:
  1. torch.Tensor must be a real class — scipy.stats uses issubclass() on it.
  2. torch.no_grad() must behave as a context manager (used in
     extract_activation_trajectories, which we never call in tests).
  3. chorus.py uses a relative import  `from .activation_trajectories import …`
     which requires both modules to share the p4_mstate_features package
     identity in sys.modules.
"""
import sys
import types
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1.  torch stub
# ---------------------------------------------------------------------------

def _make_torch_stub() -> types.ModuleType:
    stub = types.ModuleType("torch")

    class Tensor:
        """Minimal real class so scipy's issubclass checks don't raise."""

    _cm = MagicMock()
    _cm.__enter__ = MagicMock(return_value=None)
    _cm.__exit__ = MagicMock(return_value=False)

    stub.Tensor = Tensor
    stub.no_grad = MagicMock(return_value=_cm)
    stub.nn = MagicMock()
    return stub


def _install_stubs() -> None:
    if "torch" in sys.modules:
        return  # already done (e.g. pytest re-uses the process)

    torch_stub = _make_torch_stub()
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_stub.nn


# ---------------------------------------------------------------------------
# 2.  p4_mstate_features package + submodule loader
# ---------------------------------------------------------------------------

_P4_SRC = Path(__file__).parent.parent  # project root (where *.py files live)


def _ensure_p4_package() -> None:
    """Register a fake p4_mstate_features package pointing at the project root."""
    if "p4_mstate_features" in sys.modules:
        return

    pkg = types.ModuleType("p4_mstate_features")
    pkg.__path__ = [str(_P4_SRC)]
    pkg.__package__ = "p4_mstate_features"
    sys.modules["p4_mstate_features"] = pkg


def _load_p4_submodule(filename: str) -> types.ModuleType:
    """Load a file from the project root as p4_mstate_features.<stem>."""
    stem = Path(filename).stem
    full_name = f"p4_mstate_features.{stem}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    filepath = _P4_SRC / filename
    spec = importlib.util.spec_from_file_location(full_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "p4_mstate_features"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod



# ---------------------------------------------------------------------------
# 3.  Run at collection time
# ---------------------------------------------------------------------------

_install_stubs()
_ensure_p4_package()

# Pre-load both p4 modules so their symbols are available to test files that
# import them at module level.  Order matters: chorus imports from
# activation_trajectories via a relative import.
_load_p4_submodule("p4_mstate_features/activation_trajectories.py")
_load_p4_submodule("p4_mstate_features/chorus.py")
_load_p4_submodule("p4_mstate_features/geometric.py")
