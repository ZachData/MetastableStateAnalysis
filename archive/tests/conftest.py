"""
archive/tests/conftest.py — fixtures for the archived Phase 5b tests.

Moved out of tests/conftest.py when Phases 3-6 were archived. The live
conftest carried a p5b_manifold package bootstrap and the ring/line
manifold fixtures below, all of which existed only for tests that now
live in this directory.

These tests are NOT collected by default (see pytest.ini: norecursedirs)
and are NOT maintained. This file is not sufficient to run them on its
own either: they also depend on shared fixtures and the heavy-dependency
stub installer that stayed in tests/conftest.py. Reconnecting them is
part of a deliberate reintroduction, not something the archive offers
ready-made. See archive/README.md.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


# ===========================================================================
# p5b_manifold package bootstrap
#
# The archived Phase 5b modules import each other as `p5b_manifold.<stem>`
# rather than by their directory name. _P5B_SRC pointed at the project root
# when p5b_manifold_steering/ lived there; it points into the archive now.
# ===========================================================================

_P5B_SRC = Path(__file__).parent.parent / "p5b_manifold_steering"


def _ensure_p5b_package() -> None:
    """Register p5b_manifold as a package in sys.modules if absent."""
    if "p5b_manifold" in sys.modules:
        return
    pkg = types.ModuleType("p5b_manifold")
    pkg.__path__    = [str(_P5B_SRC)]
    pkg.__package__ = "p5b_manifold"
    sys.modules["p5b_manifold"] = pkg


def _load_p5b_submodule(filename: str) -> types.ModuleType:
    """Load a file from the archived p5b directory as p5b_manifold.<stem>."""
    stem      = Path(filename).stem
    full_name = f"p5b_manifold.{stem}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    filepath = _P5B_SRC / filename
    spec     = importlib.util.spec_from_file_location(full_name, filepath)
    mod      = importlib.util.module_from_spec(spec)
    mod.__package__       = "p5b_manifold"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_p5b_package()




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
