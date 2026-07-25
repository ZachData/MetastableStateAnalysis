"""
tests/test_core_frames.py — oracle tests for core/frames.py.

The ledger's whole value is that it *refuses*. Most of these tests are
therefore negative: they assert that a comparison which would previously
have succeeded silently now raises. Pure numpy; no torch, no weights.
"""

import numpy as np
import pytest

from core.frames import (
    CONVENTION_FIELDS,
    FINAL_LN_BLOCK,
    FRAME_KINDS,
    FrameMismatch,
    FrameSpec,
    UNKNOWN_REV,
    apply_frame,
    apply_pos0_policy,
    attach_frame,
    frame_gram,
    frame_of,
    frame_summary_lines,
    pos0_mask,
    verify_all_same_frame,
    verify_same_frame,
    verify_same_revision,
)
from core.ln_frame import ln_transform
from core.metrics import l2_normalize


D = 12
N = 7


def _X(seed=7):
    return np.random.default_rng(seed).normal(size=(N, D)) * 3.0 + 1.0


def _ln_params(seed=11):
    rng = np.random.default_rng(seed)
    return {"gamma": rng.normal(size=D), "beta": rng.normal(size=D), "eps": 1e-5}


def _ln_spec(**kw):
    base = dict(kind="ln_attn", layer_idx=3, reader_block=4, model_rev="step143000")
    base.update(kw)
    return FrameSpec(**base)


# ---------------------------------------------------------------------------
# Construction and invariants
# ---------------------------------------------------------------------------

class TestFrameSpec:

    def test_all_kinds_constructible(self):
        for kind in FRAME_KINDS:
            extra = {"reader_block": 2} if kind in ("ln_attn", "ln_mlp") else {}
            assert FrameSpec(kind=kind, **extra).kind == kind

    def test_bad_kind_rejected(self):
        with pytest.raises(ValueError):
            FrameSpec(kind="sphere")          # near-miss of l2_sphere

    def test_bad_pos0_policy_rejected(self):
        with pytest.raises(ValueError):
            FrameSpec(kind="raw", pos0_policy="dropped")

    def test_ln_kind_requires_reader_block(self):
        """The off-by-one must be pinned down, not deferred."""
        with pytest.raises(ValueError):
            FrameSpec(kind="ln_attn", layer_idx=3)
        with pytest.raises(ValueError):
            FrameSpec(kind="ln_mlp", layer_idx=3)

    def test_non_ln_kinds_do_not_require_reader_block(self):
        assert FrameSpec(kind="l2_sphere").reader_block is None

    def test_frozen(self):
        s = _ln_spec()
        with pytest.raises(Exception):
            s.kind = "raw"

    def test_hashable(self):
        assert len({_ln_spec(), _ln_spec(), _ln_spec(layer_idx=9)}) == 2

    def test_with_returns_new_object(self):
        s = _ln_spec()
        t = s.with_(pos0_policy="excluded")
        assert s.pos0_policy == "included" and t.pos0_policy == "excluded"
        assert t.reader_block == s.reader_block

    def test_defaults_are_conservative(self):
        """rope_applied defaults False and model_rev unknown — both must be
        set deliberately, so an unset value is visible rather than assumed."""
        s = FrameSpec(kind="raw")
        assert s.rope_applied is False
        assert s.model_rev == UNKNOWN_REV
        assert s.pos0_policy == "included"


class TestSerialisation:

    def test_round_trip(self):
        s = _ln_spec(rope_applied=True, pos0_policy="excluded")
        assert FrameSpec.from_dict(s.to_dict()) == s

    def test_round_trip_with_extras(self):
        s = FrameSpec(kind="raw", extras=(("note", "sink probe"),))
        assert FrameSpec.from_dict(s.to_dict()) == s

    def test_to_dict_is_json_safe(self):
        import json
        json.dumps(_ln_spec(rope_applied=True).to_dict())

    def test_from_dict_tolerates_missing_optionals(self):
        s = FrameSpec.from_dict({"kind": "l2_sphere"})
        assert s.model_rev == UNKNOWN_REV and s.rope_applied is False


class TestFromLnResolution:

    def test_block_frame(self):
        res = {"frame": "block", "block_idx": 5, "params": _ln_params()}
        s = FrameSpec.from_ln_resolution(res, layer_idx=4, model_rev="r1", which="attn")
        assert s.kind == "ln_attn" and s.reader_block == 5 and s.layer_idx == 4

    def test_mlp_which(self):
        res = {"frame": "block", "block_idx": 5, "params": _ln_params()}
        assert FrameSpec.from_ln_resolution(res, 4, which="mlp").kind == "ln_mlp"

    def test_identity_frame(self):
        s = FrameSpec.from_ln_resolution({"frame": "identity", "params": None}, 11)
        assert s.kind == "identity" and s.reader_block is None

    def test_final_frame_uses_sentinel(self):
        s = FrameSpec.from_ln_resolution({"frame": "final", "params": _ln_params()}, 11)
        assert s.reader_block == FINAL_LN_BLOCK
        assert dict(s.extras)["ln_source"] == "final_layer_norm"

    def test_unknown_frame_rejected(self):
        with pytest.raises(ValueError):
            FrameSpec.from_ln_resolution({"frame": "post_attn"}, 1)


# ---------------------------------------------------------------------------
# apply_frame — the single transform site
# ---------------------------------------------------------------------------

class TestApplyFrame:

    def test_ln_matches_ln_transform(self):
        X, p = _X(), _ln_params()
        got = apply_frame(X, _ln_spec(), ln_params=p)
        assert np.allclose(got, ln_transform(X, p["gamma"], p["beta"], p["eps"]))

    def test_l2_matches_l2_normalize(self):
        X = _X()
        assert np.allclose(apply_frame(X, FrameSpec.l2_sphere()), l2_normalize(X))

    def test_raw_is_identity_but_a_copy(self):
        X = _X()
        got = apply_frame(X, FrameSpec.raw())
        assert np.allclose(got, X)
        got[0, 0] = 999.0
        assert X[0, 0] != 999.0

    def test_identity_kind_passes_through(self):
        X = _X()
        assert np.allclose(apply_frame(X, FrameSpec(kind="identity")), X)

    def test_ln_without_params_raises(self):
        """
        Silently defaulting to plain LN would drop gamma — the exact shape of
        the bug this module exists to prevent.
        """
        with pytest.raises(ValueError):
            apply_frame(_X(), _ln_spec())

    def test_ln_and_l2_actually_differ(self):
        """Guard against a frame switch that does nothing."""
        X, p = _X(), _ln_params()
        assert not np.allclose(
            apply_frame(X, _ln_spec(), ln_params=p),
            apply_frame(X, FrameSpec.l2_sphere()),
            atol=1e-6,
        )

    def test_gamma_actually_used(self):
        X, p = _X(), _ln_params()
        p2 = dict(p, gamma=p["gamma"] * 2.0)
        assert not np.allclose(
            apply_frame(X, _ln_spec(), p), apply_frame(X, _ln_spec(), p2), atol=1e-6
        )

    def test_1d_input_promoted(self):
        assert apply_frame(np.arange(D, dtype=float), FrameSpec.raw()).shape == (1, D)

    def test_frame_gram_is_gram_of_frame(self):
        X, p = _X(), _ln_params()
        Xf = apply_frame(X, _ln_spec(), p)
        assert np.allclose(frame_gram(X, _ln_spec(), p), Xf @ Xf.T)

    def test_l2_gram_diagonal_is_one(self):
        assert np.allclose(np.diag(frame_gram(_X(), FrameSpec.l2_sphere())), 1.0)


# ---------------------------------------------------------------------------
# Position-0 policy
# ---------------------------------------------------------------------------

class TestPos0Policy:

    def test_included_keeps_everything(self):
        assert pos0_mask(5, "included").all()

    def test_excluded_drops_only_first(self):
        m = pos0_mask(5, "excluded")
        assert not m[0] and m[1:].all()

    def test_empty_sequence_safe(self):
        assert pos0_mask(0, "excluded").shape == (0,)

    def test_bad_policy_raises(self):
        with pytest.raises(ValueError):
            pos0_mask(5, "drop")

    def test_aligned_multi_array_filtering(self):
        X = _X()
        ids = np.arange(N)
        A = np.random.default_rng(1).random((N, N))
        Xf, idf, Af = apply_pos0_policy([X, ids, A], "excluded",
                                        axes=[(0,), (0,), (0, 1)])
        assert Xf.shape == (N - 1, D)
        assert idf.shape == (N - 1,) and idf[0] == 1
        assert Af.shape == (N - 1, N - 1)
        assert np.allclose(Af, A[1:, 1:])

    def test_included_is_a_noop(self):
        X = _X()
        (Xf,) = apply_pos0_policy([X], "included")
        assert np.allclose(Xf, X)

    def test_misaligned_arrays_raise(self):
        """A mask applied to activations but not token_ids is silent corruption."""
        with pytest.raises(ValueError):
            apply_pos0_policy([_X(), np.arange(N + 3)], "excluded")

    def test_axes_length_checked(self):
        with pytest.raises(ValueError):
            apply_pos0_policy([_X(), np.arange(N)], "excluded", axes=[(0,)])


# ---------------------------------------------------------------------------
# The guard — the point of the module
# ---------------------------------------------------------------------------

class TestGuards:

    def test_attach_and_read_back(self):
        rec = {"value": 1.0}
        attach_frame(rec, _ln_spec())
        assert frame_of(rec) == _ln_spec()

    def test_record_without_frame_raises(self):
        with pytest.raises(FrameMismatch):
            frame_of({"value": 1.0})

    def test_record_without_frame_non_strict_returns_none(self):
        assert frame_of({"value": 1.0}, strict=False) is None

    def test_same_frame_passes(self):
        a, b = {}, {}
        attach_frame(a, _ln_spec(layer_idx=1))
        attach_frame(b, _ln_spec(layer_idx=9))     # layer differs: data, not convention
        verify_same_frame(a, b)

    def test_kind_mismatch_raises(self):
        with pytest.raises(FrameMismatch):
            verify_same_frame(_ln_spec(), FrameSpec.l2_sphere())

    def test_rope_mismatch_raises(self):
        """The rotary omission, caught structurally."""
        with pytest.raises(FrameMismatch):
            verify_same_frame(_ln_spec(rope_applied=True), _ln_spec(rope_applied=False))

    def test_pos0_mismatch_raises(self):
        """The sink confound, caught structurally."""
        with pytest.raises(FrameMismatch):
            verify_same_frame(_ln_spec(pos0_policy="excluded"), _ln_spec())

    def test_message_names_the_field(self):
        try:
            verify_same_frame(_ln_spec(rope_applied=True), _ln_spec(),
                              context="P6-I2 induction")
        except FrameMismatch as e:
            msg = str(e)
            assert "rope_applied" in msg and "P6-I2 induction" in msg
        else:
            raise AssertionError("expected FrameMismatch")

    def test_reader_block_is_not_a_convention(self):
        assert "reader_block" not in CONVENTION_FIELDS
        verify_same_frame(_ln_spec(reader_block=4), _ln_spec(reader_block=8))

    def test_revision_check_separate_from_frame_check(self):
        a, b = _ln_spec(model_rev="step1000"), _ln_spec(model_rev="step143000")
        verify_same_frame(a, b)                      # allowed: sweep compares revisions
        with pytest.raises(FrameMismatch):
            verify_same_revision(a, b)

    def test_unknown_revision_cannot_be_verified(self):
        a, b = FrameSpec(kind="raw"), FrameSpec(kind="raw")
        with pytest.raises(FrameMismatch):
            verify_same_revision(a, b)

    def test_deduped_and_non_deduped_are_different_revisions(self):
        """Policy item P4, enforced rather than documented."""
        a = _ln_spec(model_rev="pythia-1.4b@step1000")
        b = _ln_spec(model_rev="pythia-1.4b-deduped@step1000")
        with pytest.raises(FrameMismatch):
            verify_same_revision(a, b)

    def test_verify_all_returns_common_spec(self):
        recs = [attach_frame({}, _ln_spec(layer_idx=i)) for i in range(4)]
        assert verify_all_same_frame(recs).kind == "ln_attn"

    def test_verify_all_catches_one_bad_record(self):
        recs = [attach_frame({}, _ln_spec()) for _ in range(4)]
        attach_frame(recs[2], FrameSpec.l2_sphere())
        with pytest.raises(FrameMismatch):
            verify_all_same_frame(recs)

    def test_verify_all_empty_raises(self):
        with pytest.raises(FrameMismatch):
            verify_all_same_frame([])

    def test_accepts_specs_or_records(self):
        rec = attach_frame({}, _ln_spec())
        verify_same_frame(_ln_spec(), rec)
        verify_same_frame(rec, _ln_spec())


class TestReporting:

    def test_describe_mentions_rope_and_pos0(self):
        d = _ln_spec(rope_applied=True, pos0_policy="excluded").describe()
        assert "rope" in d and "pos0=excluded" in d

    def test_no_rope_is_visible(self):
        assert "no-rope" in _ln_spec().describe()

    def test_summary_flags_omitted_rotary_loudly(self):
        lines = "\n".join(frame_summary_lines(_ln_spec(rope_applied=False)))
        assert "OMITTED" in lines
        assert "OMITTED" not in "\n".join(frame_summary_lines(_ln_spec(rope_applied=True)))

    def test_is_ln(self):
        assert _ln_spec().is_ln()
        assert not FrameSpec.l2_sphere().is_ln()
