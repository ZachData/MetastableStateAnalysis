"""
tests/test_core_interactions.py — core/interactions.py's typed-edge table.

Pure numpy. Covers the three things that can silently corrupt every
downstream motif count:

  1. projection_fractions' NaN-vs-zero distinction. "No projector was
     supplied" and "this force has no component there" must not collapse
     into the same number (standing rule 4).
  2. The non-symmetric anchor arm. UPDATE_PLAN.md 5.6 is the cautionary
     case: a wrong contraction that agreed with the truth at M = I and at
     every symmetric M, passing its anchor while being wrong for every
     real head. So every projector check here has an arm whose answer
     differs between a subspace and its complement.
  3. The retention contract. An absent edge is not a zero-force edge, and
     concat must refuse to merge tables thinned differently.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.interactions import (
    HEAD_AGNOSTIC,
    InteractionTable,
    PAIR_TYPES,
    classify_pair_types,
    projection_fractions,
)


# ---------------------------------------------------------------------------
# projection_fractions
# ---------------------------------------------------------------------------

class TestProjectionFractions:

    def test_absent_projector_is_nan_not_zero(self):
        """The distinction the whole 'refuse rather than degrade' rule is
        about: a missing U_A must not read as 'no imaginary component'."""
        f = np.array([[1.0, 2.0, 3.0]])
        out = projection_fractions(f, None)
        assert np.isnan(out).all()

    def test_force_entirely_inside_subspace_reads_one(self):
        U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])   # span(e0, e1)
        f = np.array([[3.0, 4.0, 0.0]])
        assert projection_fractions(f, U)[0] == pytest.approx(1.0)

    def test_force_entirely_outside_subspace_reads_zero(self):
        U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        f = np.array([[0.0, 0.0, 5.0]])
        assert projection_fractions(f, U)[0] == pytest.approx(0.0)

    def test_complementary_subspaces_sum_to_one(self):
        """A generic force split across a subspace and its orthogonal
        complement: the two fractions must sum to exactly 1. This is the
        non-symmetric arm — the answer differs between the two projectors,
        so a bug that ignored which projector it was handed would fail
        here and pass every 'all the mass is in one place' test above."""
        rng = np.random.default_rng(0)
        d = 8
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        U_a, U_b = Q[:, :3], Q[:, 3:]
        f = rng.standard_normal((25, d))
        a = projection_fractions(f, U_a)
        b = projection_fractions(f, U_b)
        assert np.allclose(a + b, 1.0)
        # and the split is genuinely non-trivial, not 0/1 by accident
        assert 0.05 < a.mean() < 0.95

    def test_rotated_basis_gives_same_answer_as_axis_aligned(self):
        """Invariance under change of basis: projecting onto span(Q[:, :2])
        a force built as Q @ v must match projecting v onto span(e0, e1).
        An implementation that transposed U would pass axis-aligned tests
        and fail this one."""
        rng = np.random.default_rng(7)
        d = 6
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        v = rng.standard_normal((10, d))
        axis = projection_fractions(v, np.eye(d)[:, :2])
        rotated = projection_fractions(v @ Q.T, Q[:, :2])
        assert np.allclose(axis, rotated)

    def test_zero_force_is_zero_not_nan(self):
        """An edge that moves nothing has a defined answer."""
        U = np.eye(3)[:, :1]
        assert projection_fractions(np.zeros((1, 3)), U)[0] == 0.0

    def test_dimension_mismatch_raises(self):
        """A projector from another layer's d_model is a frame error and
        must not broadcast into a plausible-looking number."""
        with pytest.raises(ValueError, match="frame mismatch"):
            projection_fractions(np.ones((2, 5)), np.eye(8)[:, :2])

    def test_accepts_single_vector(self):
        assert projection_fractions(np.array([1.0, 0.0]), np.eye(2)[:, :1]).shape == (1,)


# ---------------------------------------------------------------------------
# classify_pair_types
# ---------------------------------------------------------------------------

class TestClassifyPairTypes:

    def test_precedence_induction_over_strict_over_same_content(self):
        t = [5, 6, 7, 8]
        s = [1, 2, 3, 4]
        out = classify_pair_types(
            t, s,
            induction_pairs=[(5, 1)],
            strict_pairs=[(5, 1), (6, 2)],       # (5,1) is in both -> induction wins
            same_content_pairs=[(7, 3)],
        )
        assert out.tolist() == ["induction", "strict", "same_content", "neither"]

    def test_every_label_is_a_known_pair_type(self):
        out = classify_pair_types([1, 2], [0, 0], induction_pairs=[(1, 0)])
        assert set(out.tolist()) <= set(PAIR_TYPES)

    def test_no_pair_sets_gives_all_neither(self):
        out = classify_pair_types([3, 4], [1, 2])
        assert out.tolist() == ["neither", "neither"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            classify_pair_types([1, 2, 3], [0, 1])


# ---------------------------------------------------------------------------
# InteractionTable
# ---------------------------------------------------------------------------

def _table(n=6, d=5, seed=0, **kw):
    rng = np.random.default_rng(seed)
    targets = np.arange(1, n + 1)
    sources = np.zeros(n, dtype=int)
    defaults = dict(
        model="pythia-1.4b-step1000", prompt_key="p", layer=3, head=2,
        targets=targets, sources=sources,
        weight=rng.random(n),
        force=rng.standard_normal((n, d)),
        checkpoint_step=1000,
    )
    defaults.update(kw)
    return InteractionTable.from_head(**defaults)


class TestInteractionTableConstruction:

    def test_offset_is_target_minus_source(self):
        t = _table()
        assert np.array_equal(
            t.columns["offset"],
            t.columns["target"] - t.columns["source"],
        )

    def test_force_magnitude_matches_norm(self):
        rng = np.random.default_rng(3)
        force = rng.standard_normal((6, 5))
        t = _table(force=force)
        assert np.allclose(t.columns["force_magnitude"], np.linalg.norm(force, axis=1))

    def test_missing_2b_projectors_leave_rotational_channel_nan(self):
        """U_S / U_A are optional; their absence must be visible in the
        artifact rather than defaulting to zero."""
        t = _table()
        assert np.isnan(t.columns["real_frac"]).all()
        assert np.isnan(t.columns["imag_frac"]).all()

    def test_sign_channel_populated_when_projectors_given(self):
        rng = np.random.default_rng(11)
        d = 5
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        t = _table(d=d, U_pos=Q[:, :2], U_neg=Q[:, 2:])
        both = t.columns["attractive_frac"] + t.columns["repulsive_frac"]
        assert np.allclose(both, 1.0)

    def test_checkpoint_sentinel_for_uncheckpointed_model(self):
        t = _table(checkpoint_step=None)
        assert (t.columns["checkpoint_step"] == -1).all()

    def test_head_agnostic_sentinel_allowed(self):
        t = _table(head=HEAD_AGNOSTIC)
        assert (t.columns["head"] == -1).all()

    def test_unknown_pair_type_raises(self):
        with pytest.raises(ValueError, match="unknown pair_type"):
            _table(n=2, pair_type=["induction", "bogus"])

    def test_force_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="force must be"):
            _table(n=4, force=np.zeros((3, 5)))

    def test_object_dtype_extra_column_refused(self):
        with pytest.raises(ValueError, match="dtype=object"):
            _table(n=2, extra={"bad": np.array([{"a": 1}, {"b": 2}], dtype=object)})


class TestRetentionContract:

    def test_concat_refuses_mismatched_retention(self):
        """Two tables thinned by different cutoffs cannot be counted
        together: an absent edge means different things in each."""
        a = _table(retention={"mode": "top_k_by_force", "k": 32})
        b = _table(retention={"mode": "top_k_by_force", "k": 8})
        with pytest.raises(ValueError, match="different retention"):
            InteractionTable.concat([a, b])

    def test_concat_preserves_shared_retention(self):
        r = {"mode": "top_k_by_force", "k": 32}
        out = InteractionTable.concat([_table(retention=r), _table(retention=r, seed=1)])
        assert out.retention == r
        assert len(out) == 12

    def test_concat_of_untimmed_tables_is_fine(self):
        out = InteractionTable.concat([_table(), _table(seed=2)])
        assert out.retention is None

    def test_concat_empty_gives_empty_table(self):
        assert len(InteractionTable.concat([])) == 0


class TestSelection:

    def test_filter_by_pair_type(self):
        t = _table(n=4, pair_type=["induction", "neither", "induction", "strict"])
        assert len(t.filter(pair_type="induction")) == 2

    def test_filter_unknown_column_raises(self):
        with pytest.raises(KeyError):
            _table().filter(nonexistent=1)

    def test_mask_preserves_retention(self):
        r = {"mode": "top_k_by_force", "k": 4}
        t = _table(n=4, retention=r)
        assert t.mask(np.array([True, False, True, False])).retention == r

    def test_mask_wrong_length_raises(self):
        with pytest.raises(ValueError):
            _table(n=4).mask(np.array([True, False]))


class TestPersistence:

    def test_npz_roundtrip_without_pickle(self, tmp_path):
        """load() uses allow_pickle=False; anything save() writes must be
        readable under it."""
        r = {"mode": "top_k_by_force", "k": 16}
        t = _table(n=5, pair_type=["induction"] * 5, retention=r,
                   extra={"stage": np.arange(5)})
        p = tmp_path / "it.npz"
        t.save(p)
        back = InteractionTable.load(p)

        assert len(back) == len(t)
        assert set(back.columns) == set(t.columns)
        for col, before in t.columns.items():
            after = back.columns[col]
            assert after.dtype.kind == before.dtype.kind, col
            if before.dtype.kind == "f":
                # equal_nan matters: real_frac/imag_frac are all-NaN here
                # (no 2b projectors), and a roundtrip that turned NaN into
                # 0.0 would be exactly the degradation this schema forbids.
                assert np.allclose(after, before, equal_nan=True), col
            else:
                assert np.array_equal(after, before), col
        assert np.array_equal(back.extra["stage"], np.arange(5))
        assert back.retention == r

    def test_retention_survives_the_file(self, tmp_path):
        """A thinned table must say so about itself — the reader cannot be
        relied on to remember what cutoff produced it."""
        p = tmp_path / "thin.npz"
        _table(retention={"mode": "top_k_by_force", "k": 3}).save(p)
        assert InteractionTable.load(p).retention["k"] == 3

    def test_untrimmed_table_roundtrips_with_no_retention(self, tmp_path):
        p = tmp_path / "full.npz"
        _table().save(p)
        assert InteractionTable.load(p).retention is None


class TestArtifactContract:

    def test_registered_required_keys_are_all_produced(self):
        """The contract in core/artifacts.py was written before this
        producer existed. This is the test that keeps the two from
        drifting — the bug class Phase 5's blockers 2 and 3 were."""
        from core.artifacts import get_spec
        spec = get_spec("phase7", "interaction_table")
        produced = set(_table().columns)
        missing = set(spec.required_keys) - produced
        assert not missing, f"declared but not produced: {sorted(missing)}"
