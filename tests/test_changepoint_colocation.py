"""
tests/test_changepoint_colocation.py — the changepoint co-location
construction, and CLAIM-B's gate.

One construction, two registry entries: `CLAIM-B` under H-EMERGE and `P-I1`
under H-BRIDGE. `claims/EVALUABILITY.md` named them as sharing it and said they
should be built together rather than each inventing one. P-I1's thin half is
pinned in `tests/test_p7_formation_gate.py`.

Same standard the other constructed nulls were held to: exactness, validity
under H0, power against a planted effect, p = 1 with the effect reversed, and
every refusal -- on synthetic inputs with known answers, because no checkpoint
sweep exists in this repository.

Four of these are worth reading rather than counting.

`test_smallest_spacing_interval_wins_on_equal_increments` is why the estimator
is NOT `checkpoint_frames.detect_transitions`. With the SAME increment in every
interval, the rate-based estimator picks the interval with the tightest
log-step spacing every time -- so under the null its argmax concentrates there,
and a binary "the two top intervals coincide" statistic is floored by that
concentration, measured at 0.447 on a 25-checkpoint Pythia sweep with the value
series permuted against the fixed step grid. The floor
argument that retired the obvious estimator, in one deterministic assertion.

`test_splitting_an_interval_moves_the_rate_centroid_and_not_the_mass_centroid`
is why the profile is change MASS and not change RATE. Adding one checkpoint
inside an existing interval is a change to the sampling grid and not to
training, and the two weightings disagree about that.

`test_exhaustive_pairing_p_matches_an_independent_rank` is a second
implementation of the null's arithmetic, checked against the module's, the same
way `TestFastPathMatchesTheGate` pins CLAIM-C's curve against its gate.

`test_reversed_pairing_fires_the_reciprocal_branch` is the "can it fail" check.
`POPPER_PLAN.md` §6h found an audit arm reporting PASS because it was incapable
of failing; a verdict branch nothing can trigger is the same defect, and
RE-ANCHORS is the branch CLAIM-B's falsifier exists for.
"""

from __future__ import annotations

import json
import math
import pathlib
import tempfile
from itertools import permutations

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from core import changepoint_colocation as C
from core.checkpoint_frames import interval_rates, step_x

# A plausible cheap-tier Pythia sweep: log-spaced releases to 512, then the
# every-1000 releases thinned. Same schedule the committed calibration uses.
SWEEP = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000,
         8000, 13000, 23000, 33000, 43000, 63000, 83000, 103000, 123000, 143000]

SMALL = [0, 1, 10, 100, 1000, 10000]

# The grid the ANCHOR arm's tests run on, and it is not `SWEEP`. A change
# location is a weighted mean of interval midpoints, so a series with no
# located change lands on the grid's own midpoint -- and `SWEEP`'s midpoint is
# step 955, inside CLAIM-B's 512-2000 window. On that grid the anchor arm
# cannot tell "the change is at the anchor" from "there is no change", and
# `anchor_arm` now refuses rather than returning a verdict it cannot support
# (`TestTheDiffuseReferenceRefusal` below, `POPPER_PLAN.md` 6o). Pythia's full
# every-1000 release schedule puts the midpoint at step 31496, well outside,
# while keeping intervals 10 and 11 -- 512->1000 and 1000->2000 -- inside it.
ANCHOR_SWEEP = sorted(set([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                          + list(range(1000, 144000, 1000))))


def _step_curve(steps, jump_at_index, direction=+1.0):
    """A series that changes in exactly one interval and nowhere else."""
    v = np.zeros(len(steps), dtype=float)
    v[jump_at_index + 1:] = direction
    return v


def _logistic(steps, mid_step, rng=None, width=0.35, noise=0.0):
    x = step_x(steps)
    v = 1.0 / (1.0 + np.exp(-(x - math.log10(mid_step + 1.0)) / width))
    if noise and rng is not None:
        v = v + noise * rng.standard_normal(v.size)
    return v


# ---------------------------------------------------------------------------
# The estimator, and why it is not detect_transitions
# ---------------------------------------------------------------------------

class TestWhyNotTheRateEstimator:

    def test_smallest_spacing_interval_wins_on_equal_increments(self):
        """
        The geometry bias that floors a binary co-location statistic.

        Every interval carries the identical increment, so nothing about
        training distinguishes them -- and the rate estimator still picks one,
        the one whose log-step spacing is tightest. That is the concentration
        that makes the argmax of a permuted value series land there 44.7% of
        the time.
        """
        v = np.arange(len(SWEEP), dtype=float)          # increment 1 everywhere
        rate = interval_rates(SWEEP, v)["rate"]
        dx = np.diff(step_x(SWEEP))
        assert int(np.argmax(rate)) == int(np.argmin(dx))
        # and the spread is large, which is what makes the bias bite
        assert dx.max() / dx.min() > 4.0

    def test_change_mass_profile_is_flat_on_equal_increments(self):
        """The same input, under the estimator actually used: no interval is
        preferred, because no interval carries more of the change."""
        v = np.arange(len(SWEEP), dtype=float)
        w = C.change_profile(SWEEP, v, "rise")["weights"]
        assert np.allclose(w, w[0])

    def test_splitting_an_interval_moves_the_rate_centroid_and_not_the_mass_centroid(self):
        """
        Adding a checkpoint inside an interval changes the sampling grid, not
        training. The change-mass centroid barely notices; the rate-weighted
        centroid moves an order of magnitude further.
        """
        steps = [0, 1, 10, 100, 1000, 10000, 100000]
        v = _logistic(steps, 300.0)

        finer = [0, 1, 10, 30, 100, 1000, 10000, 100000]
        vf = _logistic(finer, 300.0)

        mass = C.change_profile(steps, v, "rise")["centroid_log_step"]
        mass_f = C.change_profile(finer, vf, "rise")["centroid_log_step"]

        def rate_centroid(s, y):
            r = np.clip(interval_rates(s, y)["rate"], 0.0, None)
            x = C.interval_midpoints(s)
            return float(np.sum(r * x) / r.sum())

        rate = rate_centroid(steps, v)
        rate_f = rate_centroid(finer, vf)
        # Measured on this fixture: the mass centroid moves 0.023 in log10-step
        # and the rate centroid 0.108, a factor of 4.6. Asserted at 3x so the
        # test states the effect rather than its exact size.
        assert abs(mass_f - mass) < abs(rate_f - rate) / 3.0


class TestChangeProfile:

    def test_a_single_interval_change_puts_the_centroid_in_that_interval(self):
        for j in (0, 2, len(SWEEP) - 2):
            p = C.change_profile(SWEEP, _step_curve(SWEEP, j), "rise")
            assert p["weights"][j] == pytest.approx(1.0)
            assert p["centroid_log_step"] == pytest.approx(
                C.interval_midpoints(SWEEP)[j])
            assert p["dispersion_log_step"] == pytest.approx(0.0)
            assert p["concentration"] == pytest.approx(1.0)

    def test_drop_on_a_falling_series_matches_rise_on_its_negation(self):
        v = _logistic(SWEEP, 700.0)
        a = C.change_profile(SWEEP, -v, "drop")
        b = C.change_profile(SWEEP, v, "rise")
        assert a["centroid_log_step"] == pytest.approx(b["centroid_log_step"])

    def test_centroid_step_is_the_log_axis_inverted(self):
        p = C.change_profile(SWEEP, _step_curve(SWEEP, 11), "rise")
        assert p["centroid_step"] == pytest.approx(
            10.0 ** p["centroid_log_step"] - 1.0)

    def test_dispersion_separates_one_change_from_two(self):
        one = _step_curve(SWEEP, 11)
        two = _step_curve(SWEEP, 2) + _step_curve(SWEEP, 20)
        assert (C.change_profile(SWEEP, two, "rise")["dispersion_log_step"]
                > C.change_profile(SWEEP, one, "rise")["dispersion_log_step"] + 1.0)

    @pytest.mark.parametrize("bad", ["up", "", "RISE", None])
    def test_refuses_an_unregistered_direction(self, bad):
        with pytest.raises(C.ColocationRefused, match="change direction"):
            C.change_profile(SWEEP, _step_curve(SWEEP, 3), bad)

    def test_refuses_a_series_with_no_change_in_the_registered_direction(self):
        """A uniform profile here would report 'spread evenly over training'
        for 'absent', which is a different claim."""
        with pytest.raises(C.ColocationRefused, match="no drop anywhere"):
            C.change_profile(SWEEP, _step_curve(SWEEP, 4), "drop")

    def test_refuses_non_finite_values(self):
        v = _step_curve(SWEEP, 4)
        v[7] = np.nan
        with pytest.raises(C.ColocationRefused, match="non-finite"):
            C.change_profile(SWEEP, v, "rise")

    def test_refuses_unsorted_steps_rather_than_sorting_them(self):
        s = list(SWEEP)
        s[3], s[4] = s[4], s[3]
        with pytest.raises(C.ColocationRefused, match="strictly increasing"):
            C.change_profile(s, np.arange(len(s), dtype=float), "rise")

    def test_refuses_a_sweep_too_short_to_locate_anything(self):
        with pytest.raises(C.ColocationRefused, match="at least 3 checkpoints"):
            C.change_profile([0, 100], [0.0, 1.0], "rise")

    def test_refuses_a_shape_mismatch(self):
        with pytest.raises(C.ColocationRefused, match="index the same checkpoints"):
            C.change_profile(SWEEP, np.zeros(len(SWEEP) - 1), "rise")


class TestAnchorStatistic:

    def test_zero_inside_the_window_and_negative_outside(self):
        inside = C.change_profile(SWEEP, _step_curve(SWEEP, 11), "rise")   # 1000->2000
        assert C.anchor_statistic(inside, C.CLAIM_B_ANCHOR_WINDOW) == 0.0
        outside = C.change_profile(SWEEP, _step_curve(SWEEP, 0), "rise")   # 0->1
        assert C.anchor_statistic(outside, C.CLAIM_B_ANCHOR_WINDOW) < 0.0

    def test_the_window_comes_from_the_registered_statement(self):
        assert C.CLAIM_B_ANCHOR_WINDOW == (512.0, 2000.0)

    def test_refuses_a_malformed_window(self):
        p = C.change_profile(SWEEP, _step_curve(SWEEP, 4), "rise")
        for bad in ((2000.0, 512.0), (-1.0, 5.0), (float("nan"), 5.0)):
            with pytest.raises(C.ColocationRefused, match="anchor window"):
                C.anchor_statistic(p, bad)


# ---------------------------------------------------------------------------
# The pairing null
# ---------------------------------------------------------------------------

def _paired(a_jumps, b_jumps, steps=None, alpha=0.05):
    s = SWEEP if steps is None else steps
    return C.paired_colocation_arm(
        s, [_step_curve(s, j) for j in a_jumps], "rise",
        [_step_curve(s, j) for j in b_jumps], "rise",
        alpha=alpha, unit_name="layer", arm_name="mutual")


class TestPairingNull:

    def test_exhaustive_pairing_p_matches_an_independent_rank(self):
        """
        A second implementation of the null's arithmetic, computed here from
        the centroids and compared cell for cell.
        """
        a_j, b_j = [1, 5, 11, 17, 21], [2, 6, 10, 18, 20]
        res = _paired(a_j, b_j)
        assert res["null_exhaustive"] is True
        assert res["n_pairings"] == math.factorial(5)

        x = C.interval_midpoints(SWEEP)
        ca, cb = x[np.array(a_j)], x[np.array(b_j)]
        stats = [-np.mean(np.abs(ca - cb[list(p)])) for p in permutations(range(5))]
        obs = -np.mean(np.abs(ca - cb))
        assert res["observed"] == pytest.approx(obs)
        assert res["p_value"] == pytest.approx(
            sum(s >= obs - 1e-15 for s in stats) / len(stats), abs=1e-12)
        assert res["p_reciprocal"] == pytest.approx(
            sum(s <= obs + 1e-15 for s in stats) / len(stats), abs=1e-12)

    def test_identity_pairing_is_in_the_null_so_p_is_never_zero(self):
        res = _paired([1, 5, 11, 17, 21], [1, 5, 11, 17, 21])
        assert res["p_value"] >= 1.0 / res["n_pairings"] - 1e-15
        assert res["p_value"] == pytest.approx(1.0 / math.factorial(5))

    def test_perfect_co_location_reaches_the_floor(self):
        j = [1, 5, 11, 17, 21, 23, 3, 9]
        res = _paired(j, j)
        assert res["p_value"] == pytest.approx(res["attainable_floor"])
        assert C.gate_verdict(res["p_value"], res["p_reciprocal"],
                              alpha=0.05)["verdict"] == "CO-LOCATES"

    def test_reversed_pairing_fires_the_reciprocal_branch(self):
        """
        The "can it fail" check. RE-ANCHORS is the branch CLAIM-B's falsifier
        exists for, so a design in which nothing can trigger it would be
        unfalsifiable in exactly the way the apparatus is meant to prevent.
        """
        j = [1, 4, 8, 12, 16, 20, 23]
        res = _paired(j, list(reversed(j)))
        assert res["p_value"] >= 0.99
        assert res["p_reciprocal"] <= 0.05
        v = C.gate_verdict(res["p_value"], res["p_reciprocal"], alpha=0.05)
        assert v["verdict"] == "RE-ANCHORS"
        assert v["falsified"] is True

    def test_sampled_regime_above_the_enumeration_limit(self):
        j = list(range(1, 9))          # 8 units -> 40320 pairings > 5040
        res = _paired(j, j)
        assert res["null_exhaustive"] is False
        assert res["n_pairings"] == C.N_PAIRING_PERMUTATIONS + 1
        assert res["attainable_floor"] == pytest.approx(
            1.0 / (C.N_PAIRING_PERMUTATIONS + 1))

    def test_refuses_when_too_few_units_to_reach_alpha(self):
        """
        Three units give six pairings and a floor of 0.167. A test that cannot
        reject on a perfect result reports 'not significant' about its own
        design, which on CLAIM-B would read as evidence against emergence.
        """
        with pytest.raises(C.ColocationRefused, match="attainable floor"):
            _paired([1, 5, 11], [1, 5, 11], alpha=0.05)

    def test_refuses_a_single_unit(self):
        with pytest.raises(C.ColocationRefused, match="at least two units"):
            _paired([5], [5])

    def test_refuses_when_every_pairing_gives_the_same_statistic(self):
        """The units then contribute one observation; permuting them is the
        wrong null and not a conservative one."""
        j = [7, 7, 7, 7, 7, 7, 7]
        with pytest.raises(C.ColocationRefused, match="identical statistic"):
            _paired(j, j)

    def test_refuses_mismatched_unit_counts(self):
        with pytest.raises(C.ColocationRefused, match="same units on both"):
            C.paired_colocation_arm(
                SWEEP, [_step_curve(SWEEP, j) for j in (1, 5, 9, 13, 17)], "rise",
                [_step_curve(SWEEP, j) for j in (1, 5, 9, 13)], "rise",
                alpha=0.05, unit_name="layer", arm_name="mutual")

    def test_the_diagnostic_separates_a_monotone_unit_factor(self):
        """
        The construction's measured limitation, made visible. A confound that
        orders both series by unit index shows up here; one that does not, does
        not, and the docstring says so rather than the code pretending
        otherwise.
        """
        asc = [1, 4, 8, 12, 16, 20, 23]
        d = _paired(asc, asc)["shared_unit_factor_diagnostic"]
        assert d["centroid_rank_corr_a_vs_unit_index"] == pytest.approx(1.0)
        assert d["centroid_rank_corr_b_vs_unit_index"] == pytest.approx(1.0)

        shuffled = [12, 1, 20, 4, 23, 8, 16]
        d2 = _paired(shuffled, shuffled)["shared_unit_factor_diagnostic"]
        assert abs(d2["centroid_rank_corr_a_vs_unit_index"]) < 0.6
        assert d2["centroid_rank_corr_a_vs_b"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The anchor arm, whose floor is the one likely to bind
# ---------------------------------------------------------------------------

def _controls(n, jumps=None, steps=None):
    s = ANCHOR_SWEEP if steps is None else steps
    # Default control jumps avoid intervals 10 and 11 -- the only two whose
    # midpoints lie inside the anchor window -- so the control population is a
    # null for "the change is in the window" rather than a sample of it.
    _outside = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23]
    js = jumps if jumps is not None else [_outside[i % len(_outside)]
                                          for i in range(n)]
    return ({f"ctrl{i}": [_step_curve(s, js[i])] for i in range(n)},
            {f"ctrl{i}": "rise" for i in range(n)})


class TestAnchorArm:

    def test_refuses_below_twenty_controls_at_alpha_point_oh_five(self):
        """
        The floor computed BEFORE the pilot runs: a cheap-tier sweep measuring
        six metrics has six controls, and eighteen still cannot express a p at
        0.05. This is a requirement on what the sweep must measure, not a
        result.
        """
        ctrl, dirs = _controls(18)
        with pytest.raises(C.ColocationRefused, match="attainable floor"):
            C.anchor_arm(ANCHOR_SWEEP, [_step_curve(ANCHOR_SWEEP, 11)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, ctrl, dirs,
                         alpha=0.05, unit_name="layer", arm_name="anchor")

    def test_nineteen_controls_is_the_first_workable_anchor_arm(self):
        ctrl, dirs = _controls(19)
        r = C.anchor_arm(ANCHOR_SWEEP, [_step_curve(ANCHOR_SWEEP, 11)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, ctrl, dirs,
                         alpha=0.05, unit_name="layer", arm_name="anchor")
        assert r["attainable_floor"]["control_null_floor"] == pytest.approx(1 / 20)
        assert r["p_value"] <= 0.05        # no control is inside the window
        assert r["n_controls"] == 19

    def test_a_series_far_from_the_window_does_not_clear(self):
        ctrl, dirs = _controls(19)
        r = C.anchor_arm(ANCHOR_SWEEP, [_step_curve(ANCHOR_SWEEP, 0)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, ctrl, dirs,
                         alpha=0.05, unit_name="layer", arm_name="anchor")
        assert r["p_value"] > 0.05

    def test_refuses_an_empty_control_set(self):
        with pytest.raises(C.ColocationRefused, match="no control series"):
            C.anchor_arm(ANCHOR_SWEEP, [_step_curve(ANCHOR_SWEEP, 11)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, {}, {},
                         alpha=0.05, unit_name="layer", arm_name="anchor")

    def test_refuses_a_control_with_no_registered_direction(self):
        ctrl, dirs = _controls(19)
        dirs.pop("ctrl3")
        with pytest.raises(C.ColocationRefused, match="registered change direction"):
            C.anchor_arm(ANCHOR_SWEEP, [_step_curve(ANCHOR_SWEEP, 11)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, ctrl, dirs,
                         alpha=0.05, unit_name="layer", arm_name="anchor")

    def test_refuses_a_control_covering_different_units(self):
        ctrl, dirs = _controls(19)
        ctrl["ctrl5"] = [_step_curve(ANCHOR_SWEEP, 4), _step_curve(ANCHOR_SWEEP, 6)]
        with pytest.raises(C.ColocationRefused, match="not matched"):
            C.anchor_arm(ANCHOR_SWEEP, [_step_curve(ANCHOR_SWEEP, 11)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, ctrl, dirs,
                         alpha=0.05, unit_name="layer", arm_name="anchor")

    def test_refuses_when_every_control_gives_the_same_statistic(self):
        ctrl, dirs = _controls(19, jumps=[11] * 19)
        with pytest.raises(C.ColocationRefused, match="identical statistic"):
            C.anchor_arm(ANCHOR_SWEEP, [_step_curve(ANCHOR_SWEEP, 11)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, ctrl, dirs,
                         alpha=0.05, unit_name="layer", arm_name="anchor")


class TestTheDiffuseReferenceRefusal:
    """
    The defect the dry run found, and the refusal that answers it.

    A change location is a weighted mean of interval midpoints, so a series
    whose change mass is spread over the sweep lands on the grid's own midpoint
    exactly. `SWEEP` -- CLAIM-B's REGISTERED 25-checkpoint cheap-tier sweep --
    puts that midpoint at step 955, inside the 512-2000 window, so a series
    that changes nowhere receives the anchor arm's maximum statistic.
    `POPPER_PLAN.md` 6o; `claims/audits/claim_b_p_i1_dry_run.json`.
    """

    def test_a_series_with_no_change_lands_on_the_grid_midpoint(self):
        """Exactly, not approximately: it is the definition of the centroid."""
        ref = C.diffuse_reference_profile(SWEEP)
        x = C.interval_midpoints(SWEEP)
        assert ref["centroid_log_step"] == pytest.approx(float(np.mean(x)))
        assert C.anchor_statistic(ref, C.CLAIM_B_ANCHOR_WINDOW) == 0.0

    def test_the_registered_cheap_sweep_puts_that_midpoint_in_the_window(self):
        r = C.grid_reference_report(SWEEP)
        lo, hi = C.CLAIM_B_ANCHOR_WINDOW
        assert lo < r["uniform_profile_centroid_step"] < hi

    def test_the_arm_refuses_on_the_registered_cheap_sweep(self):
        ctrl, dirs = _controls(19, steps=SWEEP)
        with pytest.raises(C.ColocationRefused, match="NO located change"):
            C.anchor_arm(SWEEP, [_step_curve(SWEEP, 11)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, ctrl, dirs,
                         alpha=0.05, unit_name="layer", arm_name="anchor")

    def test_and_the_whole_gate_refuses_with_it(self):
        """
        Unanimity, so the anchor arm takes CLAIM-B down with it -- on the sweep
        the registry names as this prediction's instrument, and on an input
        whose mutual arm is perfect.
        """
        n = 8
        ej = [11 + (i % 2) for i in range(n)]
        e = [_step_curve(SWEEP, j) for j in ej]
        f = [-_step_curve(SWEEP, j) for j in ej]
        ctrl, dirs = _controls(19, steps=SWEEP)
        ctrl = {k: v * n for k, v in ctrl.items()}
        r = C.p_value_claim_b(SWEEP, e, f, ctrl, dirs,
                              control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                              alpha=0.05)
        assert r["p_value"] is None
        assert "NO located change" in r["reason"]
        assert r["verdict"] == "INSUFFICIENT"

    def test_it_does_not_fire_where_the_arm_can_discriminate(self):
        """
        The other direction, which is the one 6h's audit arm failed: a refusal
        that fires everywhere is not a check. On a grid whose midpoint is
        outside the window the reference ranks last and the arm emits.
        """
        ctrl, dirs = _controls(19)
        r = C.anchor_arm(ANCHOR_SWEEP, [_step_curve(ANCHOR_SWEEP, 11)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, ctrl, dirs,
                         alpha=0.05, unit_name="layer", arm_name="anchor")
        assert r["p_value"] <= 0.05
        assert r["diffuse_reference"]["p_value"] > 0.05
        assert r["diffuse_reference"]["inside_window"] is False

    def test_the_condition_is_the_grid_and_not_the_control_family(self):
        """
        Measured rather than argued, and it is the correction this pass had to
        make to its own first attempt. The reference is a NOISELESS uniform
        profile and a realised change-free series is a noisy one, so the
        reference outranks even the change-free members of a family and its
        rank pegs at the floor whichever family it is handed. A refusal built
        on that rank cannot see the composition. What the composition decides
        is a RATE -- 1/(k+1) in the change-free controls, in
        `claims/audits/claim_b_p_i1_dry_run.json` -- so the rank is reported
        and the ceiling is what is refused.
        """
        ctrl, dirs = _controls(19, steps=SWEEP)
        for i in range(6):                # six controls that change everywhere
            ctrl[f"ctrl{i}"] = [np.arange(len(SWEEP), dtype=float)]
        with pytest.raises(C.ColocationRefused, match="NO located change"):
            C.anchor_arm(SWEEP, [_step_curve(SWEEP, 11)], "rise",
                         C.CLAIM_B_ANCHOR_WINDOW, ctrl, dirs,
                         alpha=0.05, unit_name="layer", arm_name="anchor")

    def test_the_reference_outscores_every_realised_change_free_series(self):
        """
        Which is why its RANK could not have carried the refusal. The reference
        is noiseless, so it sits at the ceiling exactly; a realised change-free
        series scatters below it. A rank-based condition therefore pegs at the
        floor whether or not the control family contains change-free members,
        and cannot see the composition that actually sets the rate.
        """
        rng = np.random.default_rng(0)
        ref = C.anchor_statistic(C.diffuse_reference_profile(SWEEP),
                                 C.CLAIM_B_ANCHOR_WINDOW)
        realised = [
            C.anchor_statistic(
                C.change_profile(SWEEP, 0.02 * rng.standard_normal(len(SWEEP)),
                                 "rise"),
                C.CLAIM_B_ANCHOR_WINDOW)
            for _ in range(200)]
        assert ref == 0.0
        assert max(realised) <= ref
        assert np.mean(np.asarray(realised) < ref) > 0.2

    def test_the_noise_share_diagnostic_is_zero_on_a_monotone_series(self):
        p = C.change_profile(SWEEP, _step_curve(SWEEP, 11), "rise")
        assert p["reverse_change_mass"] == 0.0
        assert p["noise_mass_share_estimate"] == 0.0

    def test_and_rises_with_per_checkpoint_noise(self):
        rng = np.random.default_rng(0)
        clean = C.change_profile(SWEEP, _logistic(SWEEP, 1000), "rise")
        noisy = C.change_profile(
            SWEEP, _logistic(SWEEP, 1000, rng=rng, noise=0.05), "rise")
        assert noisy["noise_mass_share_estimate"] > clean["noise_mass_share_estimate"]


class TestAttainableFloorReport:

    def test_the_minimum_control_count_is_derived_from_alpha(self):
        """
        19, not 20: the floor is 1/(n + 1) and it must not EXCEED alpha, so
        1/20 = 0.05 at alpha = 0.05 is admissible and rejects exactly when the
        observed statistic beats every control -- probability 0.05 under H0,
        which is nominal rather than lucky.
        """
        assert C.attainable_floor_report(18, 0.05)["sufficient"] is False
        assert C.attainable_floor_report(19, 0.05)["sufficient"] is True
        assert C.attainable_floor_report(19, 0.05)["min_controls_for_alpha"] == 19
        # and it moves when alpha does, which is what marks a derived cut
        assert C.attainable_floor_report(19, 0.01)["sufficient"] is False
        assert C.attainable_floor_report(99, 0.01)["sufficient"] is True


# ---------------------------------------------------------------------------
# Combining arms, and the three-way verdict
# ---------------------------------------------------------------------------

class TestCombineArms:

    def _arm(self, name, pg, pl):
        return {"arm": name, "p_value": pg, "p_reciprocal": pl}

    def test_intersection_union_takes_the_max_in_both_directions(self):
        out = C.combine_arms([self._arm("a", 0.01, 0.9),
                              self._arm("b", 0.30, 0.5),
                              self._arm("c", 0.02, 0.7)])
        assert out["p_value"] == pytest.approx(0.30)
        assert out["p_reciprocal"] == pytest.approx(0.9)
        assert out["binding_arm"] == "b"

    def test_refuses_when_any_arm_carries_no_p(self):
        """A max over a set with an undefined member is undefined, and
        reporting the rest would silently drop the hardest arm."""
        with pytest.raises(C.ColocationRefused, match="undefined"):
            C.combine_arms([self._arm("a", 0.01, 0.9), self._arm("b", None, None)])

    def test_refuses_an_empty_arm_set(self):
        with pytest.raises(C.ColocationRefused, match="no arms"):
            C.combine_arms([])


class TestVerdict:

    def test_three_branches(self):
        assert C.gate_verdict(0.01, 0.99, 0.05)["verdict"] == "CO-LOCATES"
        assert C.gate_verdict(0.99, 0.01, 0.05)["verdict"] == "RE-ANCHORS"
        assert C.gate_verdict(0.40, 0.60, 0.05)["verdict"] == "INSUFFICIENT"
        assert C.gate_verdict(None, None, 0.05)["verdict"] == "INSUFFICIENT"

    def test_only_re_anchors_is_a_falsification(self):
        assert C.gate_verdict(0.99, 0.01, 0.05)["falsified"] is True
        for p in ((0.01, 0.99), (0.40, 0.60), (None, None)):
            assert C.gate_verdict(*p, alpha=0.05)["falsified"] is False

    def test_co_location_wins_a_tie_so_one_statistic_cannot_be_both(self):
        assert C.gate_verdict(0.01, 0.01, 0.05)["verdict"] == "CO-LOCATES"


# ---------------------------------------------------------------------------
# CLAIM-B's gate
# ---------------------------------------------------------------------------

def _claim_b_inputs(n_layers=8, energy_jumps=None, fiedler_jumps=None,
                    n_controls=19, steps=None):
    steps = ANCHOR_SWEEP if steps is None else list(steps)
    n_int = len(steps) - 1
    ej = energy_jumps or [11 + (i % 2) for i in range(n_layers)]
    fj = fiedler_jumps or list(ej)
    ej = [j % n_int for j in ej]
    fj = [j % n_int for j in fj]
    energy = [_step_curve(steps, j) for j in ej]
    fiedler = [-_step_curve(steps, j) for j in fj]          # a DROP
    _outside = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23]
    js = [_outside[i % len(_outside)] % n_int for i in range(n_controls)]
    ctrl = {f"ctrl{i}": [_step_curve(steps, js[i])] * n_layers
            for i in range(n_controls)}
    dirs = {f"ctrl{i}": "rise" for i in range(n_controls)}
    return energy, fiedler, ctrl, dirs


#: CLAIM-B's registered sweep. The adjudication tests below have to run on it,
#: because `adjudicate_claim_b` turns away a result computed on any other grid
#: -- which is the point of registering one.
REGISTERED_SWEEP = list(C.REGISTERED_CLAIM_B_SWEEP)


class TestClaimBGate:

    def test_refuses_an_unregistered_control_family(self):
        """
        Validity IS the claim that the controls are exchangeable with the
        series under test under H0, so which population they come from is a
        pre-registered decision and not a per-run argument. Same refusal shape
        as P6-R2's on a caller-supplied exchangeable unit.
        """
        e, f, ctrl, dirs = _claim_b_inputs()
        r = C.p_value_claim_b(ANCHOR_SWEEP, e, f, ctrl, dirs,
                              control_family="whatever was lying around",
                              alpha=0.05)
        assert r["p_value"] is None
        assert "registered one" in r["reason"]
        assert r["verdict"] == "INSUFFICIENT"

    def test_end_to_end_co_location(self):
        e, f, ctrl, dirs = _claim_b_inputs()
        r = C.p_value_claim_b(ANCHOR_SWEEP, e, f, ctrl, dirs,
                              control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                              alpha=0.05)
        assert r["reason"] is None
        assert r["n_arms"] == 3
        assert {a["arm"] for a in r["arms"]} == {
            "mutual", "anchor:energy_break", "anchor:fiedler_drop"}
        assert r["p_value"] == max(a["p_value"] for a in r["arms"])
        assert r["verdict"] == "CO-LOCATES"

    def test_unanimity_one_failing_arm_binds_the_whole_gate(self):
        """The mutual arm can be perfect and the gate still not clear, because
        the anchor arms are a separate requirement the statement also makes."""
        e, f, ctrl, dirs = _claim_b_inputs(energy_jumps=[0, 1] * 4,
                                           fiedler_jumps=[0, 1] * 4)
        r = C.p_value_claim_b(ANCHOR_SWEEP, e, f, ctrl, dirs,
                              control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                              alpha=0.05)
        assert r["arms"][0]["p_value"] <= 0.05            # mutual arm is perfect
        assert r["verdict"] != "CO-LOCATES"
        assert r["binding_arm"].startswith("anchor:")

    def test_the_anchor_arms_refuse_on_a_six_metric_sweep(self):
        e, f, ctrl, dirs = _claim_b_inputs(n_controls=6)
        r = C.p_value_claim_b(ANCHOR_SWEEP, e, f, ctrl, dirs,
                              control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                              alpha=0.05)
        assert r["p_value"] is None
        assert "attainable floor" in r["reason"]

    def test_the_record_carries_the_spacing_report_it_claims_not_to_need(self):
        e, f, ctrl, dirs = _claim_b_inputs()
        r = C.p_value_claim_b(ANCHOR_SWEEP, e, f, ctrl, dirs,
                              control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                              alpha=0.05)
        sp = r["spacing"]
        assert sp["log_step_spacing_ratio"] > 4.0
        assert sp["spacing_change_steps"]

    def test_registered_series_and_fields(self):
        assert C.CLAIM_B_SERIES["energy_break"]["direction"] == "rise"
        assert C.CLAIM_B_SERIES["fiedler_drop"]["direction"] == "drop"
        assert "sum_severity" in C.CLAIM_B_SERIES["energy_break"]["field"]
        assert "n_violations" not in C.CLAIM_B_SERIES["energy_break"]["field"]
        assert C.CLAIM_B_UNIT == "layer"


class TestClaimBAdjudication:
    """
    Every test here that asks to adjudicate passes an isolated
    `adjudications_dir`, and every one runs on the REGISTERED sweep, because
    those are now the same requirement: `adjudicate_claim_b` refuses any other
    grid. `POPPER_PLAN.md` 6l and 6q both record what registering a decision
    removes -- there, a refusal that had been doubling as the thing keeping a
    synthetic p-value out of a ledger slot -- so the isolation is asserted
    rather than assumed.
    """

    def _ok(self):
        return _claim_b_inputs(steps=REGISTERED_SWEEP)

    def test_the_real_ledger_directory_is_never_touched(self):
        """
        The invariant the isolation is for. `core.adjudication` refuses to
        overwrite a record once written, so one fixture run reaching the real
        directory would occupy CLAIM-B's slot with a synthetic p-value
        permanently. 6l and 6q both record this and 6q found a dead opt-in flag
        behind it, so it is asserted rather than left to the call sites.
        """
        e, f, ctrl, dirs = self._ok()
        C.adjudicate_claim_b(REGISTERED_SWEEP, e, f, ctrl, dirs,
                             control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                             alpha=0.05, adjudicate=True,
                             adjudications_dir=pathlib.Path(
                                 tempfile.mkdtemp()))
        real = pathlib.Path(__file__).resolve().parents[1] / "claims" / "adjudications"
        assert not real.exists(), (
            f"{real} exists after an adjudicating test; every call here must "
            f"pass an isolated adjudications_dir")

    def test_a_result_computed_on_another_grid_is_refused(self):
        """
        Which checkpoints the sweep samples decides what the anchor arms can
        express before any data exists, so it is a pre-registered decision of
        CLAIM-C's criterion's class. `p_value_claim_b` computes on any grid;
        only the registered one may enter an e-process, the same division
        `p7_motifs/patching_gate.py` makes between what `unit=` computes and
        what may be adjudicated.
        """
        e, f, ctrl, dirs = _claim_b_inputs()
        with pytest.raises(C.ColocationRefused, match="registered"):
            C.adjudicate_claim_b(ANCHOR_SWEEP, e, f, ctrl, dirs,
                                 control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                                 alpha=0.05)

    def test_the_registered_sweep_is_one_the_arithmetic_admits(self):
        """Not a number anyone typed: it comes from the computed set, and the
        two conditions that need no series properties are re-checked here."""
        f = C.grid_feasibility(REGISTERED_SWEEP, C.CLAIM_B_ANCHOR_WINDOW)
        assert f["reference_outside_window"]
        assert C.anchor_statistic(
            C.diffuse_reference_profile(REGISTERED_SWEEP),
            C.CLAIM_B_ANCHOR_WINDOW) != 0.0

    def test_a_result_says_on_its_face_whether_it_is_on_the_registered_sweep(self):
        """The refusal only fires when someone asks to adjudicate. A reader of
        a p-value that never asked needs the same fact on the record."""
        e, f, ctrl, dirs = _claim_b_inputs()
        off = C.p_value_claim_b(ANCHOR_SWEEP, e, f, ctrl, dirs,
                                control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                                alpha=0.05)
        assert off["on_the_registered_sweep"] is False
        assert off["registered_sweep"] == list(C.REGISTERED_CLAIM_B_SWEEP)
        e, f, ctrl, dirs = self._ok()
        on = C.p_value_claim_b(REGISTERED_SWEEP, e, f, ctrl, dirs,
                               control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                               alpha=0.05)
        assert on["on_the_registered_sweep"] is True

    def test_every_registered_step_is_a_published_checkpoint(self):
        published = set([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                        + list(range(1000, 143001, 1000)))
        assert set(REGISTERED_SWEEP) <= published

    def test_opt_in_writes_nothing_by_default(self, tmp_path):
        e, f, ctrl, dirs = self._ok()
        r = C.adjudicate_claim_b(REGISTERED_SWEEP, e, f, ctrl, dirs,
                                 control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                                 alpha=0.05, adjudications_dir=tmp_path)
        assert r["adjudication"] is None
        assert not list(tmp_path.glob("*.json"))

    def test_emits_into_the_ledger_when_asked(self, tmp_path):
        e, f, ctrl, dirs = self._ok()
        r = C.adjudicate_claim_b(REGISTERED_SWEEP, e, f, ctrl, dirs,
                                 control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                                 alpha=0.05, adjudicate=True,
                                 adjudications_dir=tmp_path)
        adj = r["adjudication"]
        assert adj is not None and adj["prediction_id"] == "CLAIM-B"
        assert adj["claim"] == "H-EMERGE"
        assert "intersection-union" in adj["test_name"]
        assert (tmp_path / "CLAIM-B.json").exists()

    def test_the_record_names_the_shared_estimator_dependence(self, tmp_path):
        """P-I1 runs the same estimator under a different claim. A reader of
        the ledger must not take their product for two independent factors."""
        e, f, ctrl, dirs = self._ok()
        r = C.adjudicate_claim_b(REGISTERED_SWEEP, e, f, ctrl, dirs,
                                 control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                                 alpha=0.05, adjudicate=True,
                                 adjudications_dir=tmp_path)
        notes = r["adjudication"]["notes"]
        assert "P-I1" in notes and "independent factors" in notes
        assert "p_reciprocal" in notes and "NOT" in notes

    def test_a_refused_gate_writes_nothing_even_when_asked(self, tmp_path):
        e, f, ctrl, dirs = _claim_b_inputs(n_controls=6, steps=REGISTERED_SWEEP)
        r = C.adjudicate_claim_b(REGISTERED_SWEEP, e, f, ctrl, dirs,
                                 control_family=C.CLAIM_B_ANCHOR_CONTROL_FAMILY,
                                 alpha=0.05, adjudicate=True,
                                 adjudications_dir=tmp_path)
        assert r["adjudication"] is None
        assert not list(tmp_path.glob("*.json"))


# ---------------------------------------------------------------------------
# The committed calibration
# ---------------------------------------------------------------------------

class TestCommittedCalibration:
    """
    The measured rates, pinned. Recomputing them takes ~100 seconds, which the
    ten-second gating tier does not have -- the same division of labour
    `claims/calibration/claim_c_homogeneity.json` and
    `claims/audits/p6_projector_labels.json` use. What must not happen is the
    record drifting silently, so it is asserted rather than read.
    """

    def _doc(self):
        from tools.calibrate_changepoint_colocation import OUT_PATH, SCHEMA_VERSION
        doc = json.loads(OUT_PATH.read_text())
        assert doc["schema_version"] == SCHEMA_VERSION
        return doc

    def _rows(self, family):
        return [r for r in self._doc()["rows"] if r["family"] == family]

    def test_the_artifact_describes_the_design_it_measured(self):
        doc = self._doc()
        assert doc["alpha"] == 0.05
        assert 20 <= doc["n_checkpoints"] <= 30      # CLAIM-B's registered sweep
        assert doc["replicates"] >= 300
        assert len(doc["rows"]) == 15

    def test_valid_under_the_plain_h0(self):
        for r in self._rows("independent"):
            assert r["rejection_rate"] <= 0.09       # 300 draws, nominal 0.05
            assert 0.45 <= r["mean_p"] <= 0.60

    def test_valid_under_the_common_trend_that_defeats_every_order_permutation(self):
        """
        The row that decided the null. A permutation over checkpoint order
        rejects at 0.32-0.45 here and an enumerated circular shift at 0.103,
        because both assert a change could equally have been anywhere. The
        pairing null holds each series' real locations fixed on both sides.
        """
        for r in self._rows("common-trend"):
            assert r["rejection_rate"] <= 0.09

    def test_the_shared_unit_factor_limitation_is_severe_and_recorded(self):
        """
        Measured, not described. A common per-unit factor unrelated to the
        claim produces near-certain rejection, and no null over the pairing can
        separate it from the claim, because a confound present at every unit is
        present under every permutation.
        """
        for r in self._rows("shared-unit-factor"):
            assert r["rejection_rate"] >= 0.99

    def test_the_reciprocal_branch_can_actually_fire(self):
        for r in self._rows("reversed"):
            assert r["rejection_rate"] == 0.0
            assert r["reciprocal_rejection_rate"] >= 0.98
            assert r["mean_p"] >= 0.99

    def test_power_against_a_planted_co_location(self):
        for r in self._rows("power"):
            assert r["rejection_rate"] >= 0.99
