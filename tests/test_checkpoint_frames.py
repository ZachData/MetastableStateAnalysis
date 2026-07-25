"""
tests/test_checkpoint_frames.py — oracle tests for checkpoint_frames.py.

Two things under test: that a cache refuses to serve the wrong checkpoint's
LayerNorm parameters, and that transition detection in log-step disagrees
with index-based detection on Pythia's actual release schedule.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.checkpoint_frames import (
    CHEAP_TIER_METRICS,
    interval_rates,
    CheckpointFrameCache,
    assert_family_is_homogeneous,
    checkpoint_revision,
    detect_transitions,
    index_derivative,
    log_step_derivative,
    parse_checkpoint,
    revision_key,
    spacing_change_steps,
    step_x,
    transition_summary_lines,
)
from core.frame_card import FrameCardError, build_frame_card

D_MODEL = 16
N_HEADS = 2
N_BLOCKS = 3

#: Pythia's actual schedule: log-spaced to 512, then every 1000.
PYTHIA_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                1000, 2000, 3000, 4000, 5000, 6000]


def _ln(seed):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(weight=rng.normal(size=D_MODEL),
                           bias=rng.normal(size=D_MODEL), eps=1e-5)


def _model(seed=0):
    layers = [SimpleNamespace(input_layernorm=_ln(seed + i),
                              post_attention_layernorm=_ln(seed + 50 + i))
              for i in range(N_BLOCKS)]
    cfg = SimpleNamespace(hidden_size=D_MODEL, num_attention_heads=N_HEADS,
                          rotary_pct=0.25, rotary_emb_base=10000,
                          layer_norm_eps=1e-5, vocab_size=64,
                          use_parallel_residual=True,
                          _name_or_path="EleutherAI/pythia-fake")
    return SimpleNamespace(config=cfg, gpt_neox=SimpleNamespace(
        layers=layers, final_layer_norm=_ln(seed + 900)))


def _card(model_name):
    p = parse_checkpoint(model_name)
    return build_frame_card(_model(p["step"] or 0), p["base"],
                            checkpoint_revision(model_name))


class TestRevisionIdentity:

    def test_parse_plain_checkpoint(self):
        p = parse_checkpoint("pythia-1.4b-step1000")
        assert p == {"base": "pythia-1.4b", "step": 1000, "is_checkpoint": True}

    def test_deduped_base_is_distinct(self):
        """
        Policy P4, checked: the regex keeps deduped separate, so families
        cannot silently merge.
        """
        a = parse_checkpoint("pythia-1.4b-step1000")["base"]
        b = parse_checkpoint("pythia-1.4b-deduped-step1000")["base"]
        assert a != b

    def test_non_checkpoint_name(self):
        p = parse_checkpoint("pythia-1.4b")
        assert p["is_checkpoint"] is False and p["step"] is None

    def test_revision_strings(self):
        assert checkpoint_revision("pythia-1.4b-step0") == "step0"
        assert checkpoint_revision("pythia-1.4b") == "main"

    def test_revision_key_carries_the_base(self):
        assert revision_key("pythia-1.4b-deduped-step1000") == \
            "pythia-1.4b-deduped@step1000"
        assert revision_key("pythia-1.4b-step1000") != \
            revision_key("pythia-1.4b-deduped-step1000")


class TestCache:

    def test_put_and_get(self):
        c = CheckpointFrameCache()
        card, store = _card("pythia-1.4b-step1000")
        c.put("pythia-1.4b-step1000", card, store)
        got, _ = c.get("pythia-1.4b-step1000")
        assert got.revision == "step1000"

    def test_miss_raises_rather_than_falling_back(self):
        c = CheckpointFrameCache()
        card, store = _card("pythia-1.4b-step1000")
        c.put("pythia-1.4b-step1000", card, store)
        with pytest.raises(FrameCardError):
            c.get("pythia-1.4b-step143000")

    def test_final_model_card_cannot_be_stored_under_a_checkpoint(self):
        """
        The failure item 11 exists to prevent: the final model's LayerNorm
        parameters applied to an early checkpoint.
        """
        c = CheckpointFrameCache()
        final_card, store = _card("pythia-1.4b")          # revision 'main'
        with pytest.raises(FrameCardError):
            c.put("pythia-1.4b-step1000", final_card, store)

    def test_base_mismatch_rejected(self):
        c = CheckpointFrameCache()
        card, store = _card("pythia-1.4b-step1000")
        with pytest.raises(FrameCardError):
            c.put("pythia-410m-step1000", card, store)

    def test_deduped_does_not_satisfy_non_deduped(self):
        c = CheckpointFrameCache()
        card, store = _card("pythia-1.4b-deduped-step1000")
        c.put("pythia-1.4b-deduped-step1000", card, store)
        assert c.has("pythia-1.4b-deduped-step1000")
        assert not c.has("pythia-1.4b-step1000")

    def test_separate_entries_per_checkpoint(self):
        c = CheckpointFrameCache()
        for n in ("pythia-1.4b-step0", "pythia-1.4b-step1000"):
            card, store = _card(n)
            c.put(n, card, store)
        assert len(c) == 2
        a, _ = c.get("pythia-1.4b-step0")
        b, _ = c.get("pythia-1.4b-step1000")
        assert a.revision != b.revision

    def test_ln_params_actually_differ_between_checkpoints(self):
        """A cache that served the wrong entry would be substituting these."""
        _, s0 = _card("pythia-1.4b-step0")
        _, s1 = _card("pythia-1.4b-step1000")
        assert not np.allclose(s0.arrays["ln_gamma_attn"],
                               s1.arrays["ln_gamma_attn"])


class TestFamilyHomogeneity:

    def test_single_family_passes(self):
        assert assert_family_is_homogeneous(
            ["pythia-1.4b-step0", "pythia-1.4b-step1000"]) == "pythia-1.4b"

    def test_mixed_deduped_rejected(self):
        with pytest.raises(ValueError):
            assert_family_is_homogeneous(
                ["pythia-1.4b-step0", "pythia-1.4b-deduped-step0"])

    def test_mixed_sizes_rejected(self):
        with pytest.raises(ValueError):
            assert_family_is_homogeneous(
                ["pythia-1.4b-step0", "pythia-410m-step0"])


class TestLogStepDerivative:

    def test_step_zero_maps_to_zero(self):
        assert step_x([0])[0] == 0.0

    def test_linear_in_log_step_has_constant_derivative(self):
        s = np.array(PYTHIA_STEPS)
        v = 3.0 * step_x(s)
        d = log_step_derivative(s, v)["derivative"]
        assert np.allclose(d, 3.0, atol=1e-8)

    def test_unsorted_input_handled(self):
        s = [1000, 0, 8, 64]
        v = 2.0 * step_x(s)
        d = log_step_derivative(s, v)["derivative"]
        assert np.allclose(d, 2.0, atol=1e-8)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            log_step_derivative([0, 1, 2], [0.0, 1.0])

    def test_too_few_points(self):
        d = log_step_derivative([0, 1], [1.0, 2.0])["derivative"]
        assert np.all(np.isnan(d))


class TestSpacingArtifacts:

    def test_spacing_change_detected(self):
        """Pythia switches from log-spacing to every-1000 after step 512."""
        assert 1000 in spacing_change_steps(PYTHIA_STEPS)

    def test_uniform_spacing_has_none(self):
        assert spacing_change_steps([0, 100, 200, 300]) == []

    def test_the_two_axes_disagree_on_where_change_is_fastest(self):
        """
        A quantity linear in REAL training step. Per unit training, the rate
        is constant; per checkpoint index it is not, because the intervals
        between checkpoints span wildly different amounts of training (1 step
        early, 1000 steps late). The two axes therefore nominate different
        regions as fastest-changing, and only one of them is about training.
        """
        s = np.array(PYTHIA_STEPS, dtype=float)
        v = s / 1000.0                      # linear in training step
        out = detect_transitions(s, v, n_top=1)
        assert out["intervals"] != out["index_intervals"]

    def test_index_rate_ignores_how_much_training_separates_samples(self):
        s = np.array(PYTHIA_STEPS, dtype=float)
        v = s / 1000.0
        # Consecutive raw differences are largest between the widely spaced
        # late checkpoints, which span 1000 training steps each.
        out = detect_transitions(s, v, n_top=1)
        assert out["index_intervals"][0][0] >= 512

    def test_log_step_derivative_is_flat_on_the_same_data(self):
        s = np.array(PYTHIA_STEPS)
        d = np.abs(log_step_derivative(s, step_x(s))["derivative"])
        assert d.max() - d.min() < 1e-8

    def test_detector_localises_a_jump_to_an_interval(self):
        """
        A sweep samples training, so the strongest statement available is
        "between step A and step B". Central differences would smear the jump
        across both neighbours; forward differences pin the interval.
        """
        s = np.array(PYTHIA_STEPS, dtype=float)
        v = step_x(s).copy()
        v[6:] += 2.0                       # a level shift entering step 32
        out = detect_transitions(s, v, n_top=1)
        assert out["intervals"] == [(16, 32)]

    def test_intervals_are_ordered_pairs(self):
        s = np.array(PYTHIA_STEPS, dtype=float)
        out = detect_transitions(s, np.arange(s.size, dtype=float), n_top=3)
        assert all(lo < hi for lo, hi in out["intervals"])

    def test_summary_notes_disagreement_between_axes(self):
        s = np.array(PYTHIA_STEPS, dtype=float)
        v = s / 1000.0
        out = detect_transitions(s, v, n_top=1)
        text = "\n".join(transition_summary_lines(out, s))
        assert "load-bearing" in text

    def test_summary_flags_an_interval_spanning_a_spacing_change(self):
        s = np.array(PYTHIA_STEPS, dtype=float)
        v = np.zeros(s.size)
        v[11:] = 5.0                       # shift entering step 1000
        out = detect_transitions(s, v, n_top=1)
        text = "\n".join(transition_summary_lines(out, s))
        assert "spacing change" in text


class TestCheapTier:

    def test_polar_is_in_the_cheap_tier(self):
        """
        Norm outliers and attention sinks emerge during training, so sphere
        validity is a per-checkpoint question rather than a property of the
        final model.
        """
        assert "polar_layer_record" in CHEAP_TIER_METRICS
        assert "sphere_gap" in CHEAP_TIER_METRICS
