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
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock
import torch

import numpy as np
import pytest

from tests.config import D, N_LAYERS, N_TOKENS

# Smoke tests (tests/test_*_smoke.py) need the *real* torch/transformers to
# do an actual forward pass on a tiny model. Everything else in this suite
# runs against the MagicMock stubs installed below. Run smoke tests with:
#     SMOKE_REAL_DEPS=1 pytest -m smoke
# which skips stub installation for that session so the real imports stand.
_STUB_HEAVY_DEPS = os.environ.get("SMOKE_REAL_DEPS") != "1"

# ===========================================================================
# 1. Heavy-dependency stubs
#    Must run before any project import so sys.modules is populated first.
# ===========================================================================

_PROJECT_ROOT = Path(__file__).parent.parent


def _install_stubs() -> None:
    """Install torch, transformers, and core.* stubs exactly once."""

    # --- torch ---
    _torch = MagicMock()
    _torch.cuda.is_available.return_value = False
    sys.modules.setdefault("torch", _torch)

    # --- transformers ---
    for _mod in ("transformers", "transformers.models", "transformers.models.albert"):
        sys.modules.setdefault(_mod, MagicMock())

    # --- core (package) ---
    # __path__ must be set before any submodule is loaded; without it Python
    # refuses "from core.io import ..." with "core is not a package".
    _core = types.ModuleType("core")
    _core.__path__    = [str(_PROJECT_ROOT / "core")]
    _core.__package__ = "core"
    sys.modules.setdefault("core", _core)

    # --- core.config ---
    _cfg = types.ModuleType("core.config")

    _cfg.DEGENERATE_RANK_THRESHOLD = 2
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

    # --- core.io (real implementation — only depends on stdlib + numpy) ---
    _core_io_path = _PROJECT_ROOT / "core" / "io.py"
    if _core_io_path.exists():
        _spec   = importlib.util.spec_from_file_location("core.io", _core_io_path)
        _core_io = importlib.util.module_from_spec(_spec)
        _core_io.__package__ = "core"
        sys.modules.setdefault("core.io", _core_io)
        _spec.loader.exec_module(_core_io)
    else:
        sys.modules.setdefault("core.io", types.ModuleType("core.io"))

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


# ===========================================================================
# 3. p5b_manifold package bootstrap
# ===========================================================================

def _ensure_p5b_package() -> None:
    """Register p5b_manifold as a package in sys.modules if absent."""
    if "p5b_manifold" in sys.modules:
        return
    pkg = types.ModuleType("p5b_manifold")
    pkg.__path__    = [str(_P4_SRC)]
    pkg.__package__ = "p5b_manifold"
    sys.modules["p5b_manifold"] = pkg


def _load_p5b_submodule(filename: str) -> types.ModuleType:
    """Load a file from the project root as p5b_manifold.<stem>."""
    stem      = Path(filename).stem
    full_name = f"p5b_manifold.{stem}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    filepath = _P4_SRC / filename
    spec     = importlib.util.spec_from_file_location(full_name, filepath)
    mod      = importlib.util.module_from_spec(spec)
    mod.__package__       = "p5b_manifold"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Run at collection time
# ---------------------------------------------------------------------------

if _STUB_HEAVY_DEPS:
    _install_stubs()
_ensure_p4_package()
_ensure_p5b_package()

# Pre-load p4 modules; order matters — chorus imports from activation_trajectories.
_load_p4_submodule("p4_mstate_features/activation_trajectories.py")
_load_p4_submodule("p4_mstate_features/chorus.py")
_load_p4_submodule("p4_mstate_features/geometric.py")
_load_p4_submodule("p4_mstate_features/analysis.py")


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



"""
Additions to tests/conftest.py for Phase 5b.
"""


# Shared constants for p5b fixtures
P5B_N_C   = 7     # clusters (one per "weekday")
P5B_D     = 64    # activation dimension
P5B_K_PCA = 16    # PCA dimension for fixture (small for speed)
P5B_VOCAB = 256   # vocabulary size for behavior manifold


# ---------------------------------------------------------------------------
# Centroid geometry: ring (cyclic) and line (sequential)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ring_centroids_raw() -> np.ndarray:
    """
    (P5B_N_C, P5B_D) unit-norm centroids arranged on a ring in the first
    two PCA dimensions with small noise in the remaining dimensions.

    These have known geodesic structure: arc-lengths are proportional to
    angle differences on the circle.
    """
    rng    = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, P5B_N_C, endpoint=False)
    c      = np.zeros((P5B_N_C, P5B_D))
    c[:, 0] = np.cos(angles)
    c[:, 1] = np.sin(angles)
    c[:, 2:] = rng.standard_normal((P5B_N_C, P5B_D - 2)) * 0.05
    norms  = np.linalg.norm(c, axis=1, keepdims=True)
    return (c / norms).astype(np.float32)


@pytest.fixture(scope="session")
def ring_pca(ring_centroids_raw) -> tuple:
    """(scores, basis, evr) from pca_reduce on ring centroids."""
    from p5b_manifold.manifold_fit import pca_reduce
    return pca_reduce(ring_centroids_raw, P5B_K_PCA)


@pytest.fixture(scope="session")
def ring_mh(ring_pca) -> dict:
    """Activation manifold fit to ring centroids (periodic)."""
    from p5b_manifold.manifold_fit import arc_length_params, fit_activation_manifold
    scores, _, _ = ring_pca
    u = arc_length_params(scores, periodic=True)
    return fit_activation_manifold(scores, u, periodic=True)


@pytest.fixture(scope="session")
def ring_dists_peaked() -> np.ndarray:
    """
    (P5B_N_C, P5B_VOCAB) peaked distributions: cluster i concentrates mass
    on token i with neighbour spillover.  Designed to recapitulate ring
    structure in behavior space.
    """
    rng = np.random.default_rng(1)
    p   = np.ones((P5B_N_C, P5B_VOCAB)) * 1e-6
    for i in range(P5B_N_C):
        p[i, i % P5B_VOCAB]           = 0.80
        p[i, (i + 1) % P5B_VOCAB]     = 0.10
        p[i, (i - 1) % P5B_VOCAB]     = 0.10
    p /= p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


@pytest.fixture(scope="session")
def ring_my(ring_dists_peaked, ring_pca) -> dict:
    """Behavior manifold fit to peaked ring distributions (periodic)."""
    from p5b_manifold.manifold_fit import arc_length_params, fit_behavior_manifold
    scores, _, _ = ring_pca
    u = arc_length_params(scores, periodic=True)   # same u as ring_mh
    return fit_behavior_manifold(ring_dists_peaked, u, periodic=True)


@pytest.fixture(scope="session")
def ring_pairwise(ring_mh, ring_my, ring_pca, ring_centroids_raw) -> dict:
    """Precomputed pairwise distances for the ring geometry."""
    from p5b_manifold.isometry_test import pairwise_distances
    scores, _, _ = ring_pca
    angles = np.linspace(0, 2 * np.pi, P5B_N_C, endpoint=False)
    u      = angles / (2 * np.pi)   # normalise to [0, 1)
    return pairwise_distances(ring_mh, ring_my, u, scores, n_pts=50)

# ===========================================================================
# 4. Smoke-test fixtures (Tier 1 — real end-to-end pipeline checks)
#
# Only meaningful under SMOKE_REAL_DEPS=1 (see _STUB_HEAVY_DEPS above), since
# they need the real torch/transformers, not the MagicMock stand-ins used by
# the rest of this file. Defining them unconditionally is harmless — they
# just never get requested unless a test marked `smoke` pulls them in.
#
# Model choice: hf-internal-testing/tiny-random-gpt2 and its GPT-NeoX
# counterpart hf-internal-testing/tiny-random-GPTNeoXForCausalLM — both
# real, publicly hosted, randomly-initialised checkpoints (a few hundred KB
# each), confirmed to exist on the Hub. Requires network once; cached by
# huggingface_hub after that.
#
# Confirmed by an actual SMOKE_REAL_DEPS=1 run (not just inferred): load_model
# does resolve both registry entries correctly — the GPT-2 and GPT-NeoX
# fixtures both get as far as tokenizing and calling the model. That run also
# surfaced a real, pre-existing, non-Pythia-specific bug in core/models.py's
# extract_activations: on a CUDA-visible machine, core.config.DEVICE resolves
# to "cuda", load_model puts the model there, but extract_activations
# tokenizes the prompt and calls the model without moving the resulting
# `inputs` onto that device — "Expected all tensors to be on the same device
# ... index is on cuda:0, different from other tensors on cpu". Every
# model_to_run entry then fails silently inside run_1.run_all's per-model
# try/except, so `results` comes back empty and both tiny_phase1_dir fixtures
# (GPT-2 and GPT-NeoX) fail their own assert with the same message.
#
# Fix (not made here — belongs in core/models.py, not in test code):
#     inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
#     inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
# mirroring what run_1.py's own _run_sublayer_analysis already does
# correctly. On a CPU-only machine this bug doesn't surface (model and
# inputs are both already on "cpu"), which is presumably why it went
# unnoticed until now.
# ===========================================================================

SMOKE_TINY_GPT2    = "hf-internal-testing/tiny-random-gpt2"
SMOKE_TINY_GPTNEOX = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
SMOKE_PROMPT       = "short_heterogeneous"


@pytest.fixture(scope="session", autouse=True)
def _register_smoke_models():
    """Add the tiny smoke-test checkpoints to the real model registry.

    autouse + session-scoped: cheap (dict inserts), and every smoke test
    needs at least one of these present regardless of which phase it's
    checking, so there's no value in making each test request it explicitly.
    No-ops when running under the stubbed (non-smoke) session, since
    core.config there is the MagicMock/stub version and MODEL_CONFIGS = {}
    is thrown away at session end either way.
    """
    import core.config as cfg

    cfg.MODEL_CONFIGS.setdefault(SMOKE_TINY_GPT2, {
        "model_class":     __import__("transformers").GPT2Model,
        "tokenizer_class": __import__("transformers").GPT2Tokenizer,
        "is_albert":       False,
        "random_init":     False,
    })
    cfg.MODEL_CONFIGS.setdefault(SMOKE_TINY_GPTNEOX, {
        "model_class":     __import__("transformers").GPTNeoXModel,
        "tokenizer_class": __import__("transformers").AutoTokenizer,
        "is_albert":       False,
        "random_init":     False,
    })


@pytest.fixture(scope="session")
def tiny_phase1_dir(tmp_path_factory):
    """
    Run the real phase 1 pipeline on the tiny GPT-2 checkpoint and one
    short prompt. Returns the phase 1 output root (the directory phase 2's
    _find_run_dir expects as `phase1_dir`), so downstream phases can be
    smoke-tested against a real on-disk artifact set instead of a mock —
    which is the point: this is what actually exercises the artifact
    contract (producer/consumer naming) rather than assuming it holds.
    """
    from p1_mstate_tracking import run_1

    out_root = tmp_path_factory.mktemp("phase1_smoke")
    # run_1 imported BASE_RESULTS_DIR by name at module load time, so the
    # module-level attribute has to be patched directly — patching
    # core.config.BASE_RESULTS_DIR alone would not reach run_1's copy.
    run_1.BASE_RESULTS_DIR = out_root

    results = run_1.run_all(
        models_to_run=[SMOKE_TINY_GPT2],
        prompts_to_run=[SMOKE_PROMPT],
        run_extended=False,
    )
    assert results, (
        "phase 1 smoke run produced no results — check that "
        f"{SMOKE_TINY_GPT2} loads via core.models.load_model, and if it "
        "does, check for the CUDA/CPU device-mismatch bug documented above "
        "(extract_activations needs to move `inputs` onto model.device)"
    )
    return run_1.OUTPUT_DIR


@pytest.fixture(scope="session")
def tiny_phase1_gptneox_dir(tmp_path_factory):
    """
    Run the real phase 1 pipeline on the tiny GPT-NeoX checkpoint and one
    short prompt — the GPT-NeoX-family counterpart to tiny_phase1_dir.

    Two things this specifically exercises that the GPT-2 smoke fixture
    can't: (1) _is_causal_model recognizing a non-"gpt2"-prefixed causal
    model, and (2) analyze_value_eigenspectrum's fused-query_key_value
    branch, since hf-internal-testing/tiny-random-GPTNeoXForCausalLM has
    real (if tiny) query_key_value weights to split.

    Same session-scoped shape as tiny_phase1_dir, and subject to the same
    device-mismatch bug documented in the "Smoke-test fixtures" header
    above — that bug lives in core/models.py, not here, and affects this
    fixture identically.
    """
    from p1_mstate_tracking import run_1

    out_root = tmp_path_factory.mktemp("phase1_smoke_gptneox")
    run_1.BASE_RESULTS_DIR = out_root

    results = run_1.run_all(
        models_to_run=[SMOKE_TINY_GPTNEOX],
        prompts_to_run=[SMOKE_PROMPT],
        run_extended=False,
    )
    assert results, (
        "phase 1 smoke run produced no results for the GPT-NeoX branch — "
        f"check that {SMOKE_TINY_GPTNEOX} loads via core.models.load_model, "
        "and if it does, check for the CUDA/CPU device-mismatch bug "
        "documented above (extract_activations needs to move `inputs` onto "
        "model.device)"
    )
    return run_1.OUTPUT_DIR


@pytest.fixture(scope="session")
def tiny_phase2_dir(tiny_phase1_dir, tmp_path_factory):
    """
    Run the real phase 2 pipeline against the real phase 1 output above.
    Deliberately uses run_full (loads its own model + calls analyze_weights)
    rather than run_offline, so this also re-exercises model loading for
    the same tiny checkpoint under phase 2's code path, not just phase 1's.
    """
    from p2_eigenspectra import run_2

    out_root = tmp_path_factory.mktemp("phase2_smoke")
    run_2.BASE_RESULTS_DIR = out_root

    verdicts = run_2.run_full(
        models_to_run=[SMOKE_TINY_GPT2],
        prompts_to_run=[SMOKE_PROMPT],
        phase1_dir=tiny_phase1_dir,
    )
    assert verdicts, (
        "phase 2 smoke run produced no verdicts — most likely _find_run_dir "
        "didn't match phase 1's stem naming for this model_name; that "
        "mismatch is exactly the artifact-contract bug class this is "
        "meant to catch"
    )
    return out_root
