"""
archive/tests/test_p6_oracle_degeneracy.py — oracle-tier test for
archive/p6_subspace/eigenspace_degeneracy.py.

Split out of tests/test_core_oracle.py when Phase 6 was archived. The other
three oracle cases in that file exercise live code (core.metrics,
p1_mstate_tracking.clustering, p2b_imaginary.rotational_schur) and stayed
there; this fourth case is the only one that reached into p6.

NOT collected by default (pytest.ini: norecursedirs = archive) and not
maintained. See archive/README.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from p6_subspace.eigenspace_degeneracy import (
    degeneracy_ratio,
    project_to_subspace,
    run_eigenspace_degeneracy,
)


# ============================================================================
# Oracle 4 — planted-subspace degeneracy ratio (eigenspace_degeneracy.py)
#
# Doubles as the regression/coverage test status-6.md names for known bug 1
# ("eigenspace_degeneracy.py — NameError, `d` undefined"). That specific
# error does not reproduce against the current source (traced: `d` in
# degeneracy_sweep is a local assigned from X.shape[1] before any use; no
# bare `d` appears anywhere else in the module) — most likely already fixed
# in the same pass that produced this file's two documented fixes, without
# the fix being added to that list or to status-6.md. Either way, neither
# degeneracy_sweep nor run_eigenspace_degeneracy had any test coverage
# before this (test_phase6.py only covers project_to_subspace,
# degeneracy_ratio, lda_direction, subspace_alignment), so this closes that
# gap regardless of whether the original bug still exists. status-6.md
# itself flags this exact function as "a candidate for the oracle-tier
# suite since a degeneracy-ratio computation has a known-correct answer on
# planted synthetic clusters" — this is that test.
# ============================================================================

class TestPlantedSubspaceDegeneracyRatio:

    def _planted_data(self, d: int = 40, k_planted: int = 4,
                       n_clusters: int = 3, n_per: int = 25,
                       center_scale: float = 5.0, noise: float = 0.2,
                       seed: int = 0):
        """
        Cluster centroids live entirely inside a planted k_planted-dim
        subspace (U_pos); every other direction (U_neg, U_A) carries only
        i.i.d. noise with no cluster-dependent structure. Basis is a random
        orthogonal Q so the planted subspace is not axis-aligned.
        """
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        U_pos = Q[:, :k_planted]
        U_neg = Q[:, k_planted:2 * k_planted]
        U_A   = Q[:, 2 * k_planted:]

        centers = rng.standard_normal((n_clusters, k_planted)) * center_scale
        X, labels = [], []
        for ci in range(n_clusters):
            coeffs     = centers[ci] + rng.standard_normal((n_per, k_planted)) * noise
            noise_rest = rng.standard_normal((n_per, d - k_planted)) * noise
            pts = coeffs @ U_pos.T + noise_rest @ np.hstack([U_neg, U_A]).T
            X.append(pts)
            labels += [ci] * n_per
        X = np.vstack(X).astype(np.float32)
        return X, np.array(labels), U_pos, U_neg, U_A

    def test_ratio_high_in_planted_subspace_low_elsewhere(self):
        X, labels, U_pos, U_neg, _ = self._planted_data()
        ratio_pos = degeneracy_ratio(project_to_subspace(X, U_pos), labels)["ratio"]
        ratio_neg = degeneracy_ratio(project_to_subspace(X, U_neg), labels)["ratio"]

        assert ratio_pos >= 5.0, (
            f"expected R>=5 in the subspace clusters actually live in "
            f"(P6-R1's own threshold), got {ratio_pos:.2f}"
        )
        assert ratio_neg < 2.0, (
            f"expected near-null R in a same-dimension subspace with no "
            f"planted cluster structure, got {ratio_neg:.4f}"
        )
        assert ratio_pos > 50 * ratio_neg

    def test_full_pipeline_runs_without_error_and_p6r1_passes(self):
        """
        Guards the previously-reported NameError directly: this calls
        run_eigenspace_degeneracy end-to-end (degeneracy_sweep +
        run_eigenspace_degeneracy's LDA-alignment path together), which had
        no prior test at all.
        """
        X, labels, U_pos, U_neg, U_A = self._planted_data()
        ctx = {
            "activations_per_layer": [X, X],
            "labels_per_layer":      [labels, labels],
            "layer_type_labels":     ["plateau", "plateau"],
            "projectors": {
                "per_layer": [{"U_pos": U_pos, "U_neg": U_neg, "U_A": U_A}] * 2,
                "d_model": U_pos.shape[0],
            },
            "layer_names": ["L0", "L1"],
        }
        result = run_eigenspace_degeneracy(ctx)   # must not raise
        assert result.applicable is True
        assert result.payload["n_p6r1_pass"] == result.payload["n_plateau_layers"] == 2
