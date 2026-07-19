"""
tests/test_dual_reading.py — core/dual_reading.py.

geometric_reading and the numpy/plain-Python pieces of semantic_reading
(LDA projection, probe prediction) are pure numpy — no torch anywhere in
that path — and ARE runtime-verified here. The frozen-head-decode piece
of semantic_reading needs a real model and is NOT (see
core/dual_reading.py's own docstring) — covered separately by
tests/test_dual_reading_smoke.py, unverified for the same reason as
core/intervention.py's smoke test.
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt

from core.dual_reading import (
    geometric_reading,
    semantic_reading,
    dual_reading,
    effective_rank_contribution,
    to_particle_row_fields,
)


def _orth_basis(d, r, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, max(r, 1))))
    return Q[:, :r]


# ---------------------------------------------------------------------------
# effective_rank_contribution
# ---------------------------------------------------------------------------

class TestEffectiveRankContribution:

    def test_none_mask_gives_nan(self):
        rng = np.random.default_rng(0)
        pop = rng.standard_normal((20, 8))
        result = effective_rank_contribution(pop, None)
        assert np.isnan(result)

    def test_empty_mask_gives_zero(self):
        rng = np.random.default_rng(0)
        pop = rng.standard_normal((20, 8))
        mask = np.zeros(20, dtype=bool)
        result = effective_rank_contribution(pop, mask)
        assert result == 0.0

    def test_removing_a_redundant_point_costs_little(self):
        """A point that's an exact duplicate of another point in the
        population contributes ~nothing to effective rank when removed."""
        rng = np.random.default_rng(1)
        base = rng.standard_normal((10, 8))
        duplicate_row = base[0:1].copy()
        pop = np.concatenate([base, duplicate_row], axis=0)  # 11 rows, last is a dup
        mask = np.zeros(11, dtype=bool)
        mask[-1] = True  # remove the duplicate

        contribution = effective_rank_contribution(pop, mask)
        assert abs(contribution) < 0.05

    def test_removing_a_uniquely_individuated_point_costs_more(self):
        """A point occupying a direction no other point represents should
        cost more effective rank to remove than a point that's redundant
        with several others along the same direction.

        Magnitudes are matched across both groups deliberately: effective
        rank in mode="raw" is scale-sensitive, so a naive "big-norm
        outlier vs small-norm cluster" setup confounds magnitude with
        directional individuation (a large-norm single point can dominate
        the raw spectrum and make the *whole population* look more
        collapsed, not less -- the opposite of what "individuated" is
        meant to capture here). Same norm throughout isolates direction
        as the only variable.
        """
        rng = np.random.default_rng(2)
        d = 8
        e1 = np.zeros(d); e1[0] = 1.0
        e2 = np.zeros(d); e2[1] = 1.0

        # 5 points clustered tightly around e1 (mutually redundant, same
        # direction, small perturbations) + 1 point along e2 (the only
        # representative of that direction). All unit-norm-scale.
        redundant_group = e1[None, :] + rng.standard_normal((5, d)) * 0.01
        unique_point = e2[None, :].copy()
        pop = np.concatenate([redundant_group, unique_point], axis=0)  # 6 rows

        mask_redundant = np.zeros(6, dtype=bool); mask_redundant[0] = True
        mask_unique = np.zeros(6, dtype=bool); mask_unique[-1] = True

        contrib_redundant = effective_rank_contribution(pop, mask_redundant)
        contrib_unique = effective_rank_contribution(pop, mask_unique)
        assert contrib_unique > contrib_redundant

    def test_removing_entire_population_gives_nan(self):
        rng = np.random.default_rng(0)
        pop = rng.standard_normal((5, 8))
        mask = np.ones(5, dtype=bool)
        result = effective_rank_contribution(pop, mask)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# geometric_reading
# ---------------------------------------------------------------------------

class TestGeometricReading:

    def test_all_fields_present(self):
        d = 8
        rng = np.random.default_rng(0)
        vector = rng.standard_normal(d)
        population = rng.standard_normal((15, d))
        projectors = {
            "U_pos": _orth_basis(d, 2, seed=1),
            "U_neg": _orth_basis(d, 2, seed=2),
            "U_S":   _orth_basis(d, 3, seed=3),
            "U_A":   _orth_basis(d, 3, seed=4),
        }
        mask = np.zeros(15, dtype=bool); mask[0] = True

        result = geometric_reading(vector, population, projectors, mask)
        for key in ("v_attractive_frac", "v_repulsive_frac", "real_frac",
                    "imag_frac", "effective_rank_contribution"):
            assert key in result

    def test_missing_projector_key_gives_none(self):
        d = 8
        rng = np.random.default_rng(0)
        vector = rng.standard_normal(d)
        population = rng.standard_normal((15, d))
        projectors = {"U_pos": _orth_basis(d, 2)}  # no U_neg, U_S, U_A

        result = geometric_reading(vector, population, projectors)
        assert result["v_attractive_frac"] is not None
        assert result["v_repulsive_frac"] is None
        assert result["real_frac"] is None
        assert result["imag_frac"] is None

    def test_projection_onto_own_subspace_is_one(self):
        """A vector lying entirely inside U_pos's column space should have
        v_attractive_frac == 1.0 exactly."""
        d = 8
        U_pos = _orth_basis(d, 3, seed=1)
        vector = U_pos @ np.array([1.0, 2.0, -1.5])  # entirely in span(U_pos)
        projectors = {"U_pos": U_pos}

        result = geometric_reading(vector, np.zeros((1, d)), projectors)
        npt.assert_allclose(result["v_attractive_frac"], 1.0, atol=1e-10)

    def test_projection_onto_orthogonal_subspace_is_zero(self):
        d = 8
        rng = np.random.default_rng(5)
        full_basis, _ = np.linalg.qr(rng.standard_normal((d, d)))
        U_pos = full_basis[:, :3]
        U_orth = full_basis[:, 3:6]
        vector = U_orth @ np.array([1.0, 1.0, 1.0])  # orthogonal to U_pos

        result = geometric_reading(vector, np.zeros((1, d)), {"U_pos": U_pos})
        npt.assert_allclose(result["v_attractive_frac"], 0.0, atol=1e-10)

    def test_zero_vector_gives_none_not_crash(self):
        d = 8
        projectors = {"U_pos": _orth_basis(d, 2)}
        result = geometric_reading(np.zeros(d), np.zeros((1, d)), projectors)
        assert result["v_attractive_frac"] is None


# ---------------------------------------------------------------------------
# semantic_reading (numpy-only pieces: lda_projection, probe_predicted_label)
# ---------------------------------------------------------------------------

class _FakeProbe:
    """Minimal sklearn-like classifier for testing without a real fit."""
    def predict(self, X):
        return np.array([7])


class TestSemanticReadingNumpyPieces:

    def test_all_none_when_nothing_supplied(self):
        vector = np.zeros(8)
        result = semantic_reading(vector)
        for key in result:
            assert result[key] is None

    def test_lda_projection_computed_when_direction_given(self):
        vector = np.array([1.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        result = semantic_reading(vector, lda_direction=direction)
        npt.assert_allclose(result["lda_projection"], 1.0)

    def test_lda_projection_zero_when_orthogonal(self):
        vector = np.array([1.0, 0.0, 0.0])
        direction = np.array([0.0, 1.0, 0.0])
        result = semantic_reading(vector, lda_direction=direction)
        npt.assert_allclose(result["lda_projection"], 0.0, atol=1e-10)

    def test_probe_prediction_used_when_probe_given(self):
        vector = np.zeros(8)
        result = semantic_reading(vector, probe=_FakeProbe())
        assert result["probe_predicted_label"] == 7

    def test_decode_fields_none_without_model(self):
        vector = np.zeros(8)
        result = semantic_reading(vector, lda_direction=np.ones(8), probe=_FakeProbe())
        assert result["decode_entropy"] is None
        assert result["decode_top1_id"] is None
        assert result["decode_top_k"] is None
        # non-decode fields still populated independently
        assert result["lda_projection"] is not None
        assert result["probe_predicted_label"] is not None


# ---------------------------------------------------------------------------
# dual_reading (combined) + to_particle_row_fields
# ---------------------------------------------------------------------------

class TestDualReadingCombined:

    def test_returns_geometric_and_semantic_keys(self):
        d = 8
        rng = np.random.default_rng(0)
        vector = rng.standard_normal(d)
        population = rng.standard_normal((10, d))
        projectors = {"U_pos": _orth_basis(d, 2)}

        result = dual_reading(vector, population, projectors)
        assert set(result.keys()) == {"geometric", "semantic"}

    def test_no_optional_inputs_still_returns_full_schema(self):
        d = 8
        vector = np.zeros(d)
        population = np.zeros((1, d))
        result = dual_reading(vector, population, {})
        assert "v_attractive_frac" in result["geometric"]
        assert "lda_projection" in result["semantic"]


class TestParticleRowProjection:

    def test_produces_expected_keys(self):
        d = 8
        rng = np.random.default_rng(0)
        vector = rng.standard_normal(d)
        population = rng.standard_normal((10, d))
        projectors = {"U_pos": _orth_basis(d, 2), "U_neg": _orth_basis(d, 2, seed=9)}
        reading = dual_reading(vector, population, projectors, lda_direction=np.ones(d))

        row = to_particle_row_fields(reading)
        assert "v_attractive_proj" in row
        assert "v_repulsive_proj" in row
        assert "extra__real_frac" in row
        assert "extra__eff_rank_contribution" in row
        assert "extra__lda_projection" in row
        # explicitly excluded (non-scalar) fields must NOT appear
        assert "decode_top_k" not in row
        assert "decode_top1_token" not in row
        assert not any("token" in k for k in row)

    def test_matches_source_reading_values(self):
        d = 8
        rng = np.random.default_rng(3)
        vector = rng.standard_normal(d)
        population = rng.standard_normal((10, d))
        projectors = {"U_pos": _orth_basis(d, 2)}
        reading = dual_reading(vector, population, projectors)

        row = to_particle_row_fields(reading)
        assert row["v_attractive_proj"] == reading["geometric"]["v_attractive_frac"]
        assert row["extra__imag_frac"] == reading["geometric"]["imag_frac"]
