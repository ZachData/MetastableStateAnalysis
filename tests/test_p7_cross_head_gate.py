"""
tests/test_p7_cross_head_gate.py — P-I3's gate (the cross-head entry, as a
motif-rate contrast against control heads matched on the behavioural induction
score).

Same standard the other constructed nulls were held to: exactness, validity
under H0, power against a planted effect, the reciprocal branch firing with the
effect reversed, and every refusal -- on synthetic head tables with known
answers, because no motif sweep exists in this repository.

Five of these are worth reading rather than counting.

`test_a_threshold_classification_floors_the_design_at_one` is the finding, and
it needs no simulation. An induction head is one whose behavioural induction
score clears a cutoff, so when the classification IS that cutoff no induction
head has a control head above it in score, no matched set can be straddled, and
the design floor is 1.000. `PREDICTIONS.md`'s Phase 7 adjudication constraint 2
-- the tautology risk this phase calls its central methodological danger --
becomes arithmetic that runs before a single edge is counted, where `P-I1`'s
gate could only refuse the degenerate case and leave the rest to the analyst.

`test_the_registered_null_draws_impossible_classifications` is why "permutation
over the head classification" is not the null this gate uses. Of the C(384, 8)
label assignments it draws from, exactly one is a threshold on the score.

`test_selection_attenuates_the_correlation_and_not_the_slope` is the degeneracy
the construction is built around, and it is the reason the statistic P-I3's
wording names reaches no ledger: on ONE population with ONE relation the two
arms' correlations differ by more than the effect any real result would carry.

`test_a_one_sided_match_reads_curvature_as_an_effect` pins the discarded
matching. Taking the nearest control heads by score is what "matched control"
first suggests, and it leaks, because an induction head's nearest controls are
almost all below it.

`test_the_layer_key_removes_the_band_confound_and_costs_power` pins that the
registered decision is a trade with both sides measured, not a free
improvement.
"""

from __future__ import annotations

import json
from math import comb

import numpy as np
import pytest

from p7_motifs.cross_head_gate import (
    CONTROL_MATCHING_KEYS,
    INDEPENDENCE_SOURCES,
    N_CONTROLS_PER_INDUCTION_HEAD,
    P_I3_ALTERNATIVE,
    P_I3_RECIPROCAL_ALTERNATIVE,
    REGISTERED_CONTROL_MATCHING_KEY,
    REGISTERED_EXCHANGEABLE_UNIT,
    CrossHeadRefused,
    adjudicate_p_i3,
    attainable_floor_report,
    correlation_contrast_report,
    exact_rank_arm,
    gate_verdict,
    layer_concentration_diagnostic,
    matched_sets,
    matching_report,
    p_value_p_i3,
    registered_null_invariance_report,
    separation_report,
)

pytestmark = pytest.mark.pure

ALPHA = 0.05
N_LAYERS, N_HEADS_PER_LAYER = 24, 16
N_HEADS = N_LAYERS * N_HEADS_PER_LAYER
K_INDUCTION = 8


# ---------------------------------------------------------------------------
# One checkpoint's head table, with a planted answer
# ---------------------------------------------------------------------------

def heads(seed=0, *, rho_c=0.8, k=K_INDUCTION, main=1.0, curve=0.0,
          effect=0.0, band_elevation=0.0, confine_to_band=False):
    r = np.random.default_rng(seed)
    keys = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS_PER_LAYER)]
    layer = np.array([kk[0] for kk in keys])
    b = r.standard_normal(N_HEADS)
    c = rho_c * b + np.sqrt(max(0.0, 1.0 - rho_c ** 2)) * r.standard_normal(N_HEADS)
    if confine_to_band:
        c = np.where((layer >= 6) & (layer <= 11), c, c - 3.0)
    lab = np.zeros(N_HEADS, dtype=bool)
    lab[np.argsort(c, kind="mergesort")[-k:]] = True
    elev = np.where((layer >= 6) & (layer <= 11), band_elevation, 0.0)
    x = (main * (b + curve * (b ** 2 - 1.0)) + effect * lab + elev
         + r.standard_normal(N_HEADS))
    return keys, x, b, lab, layer


def dicts(seed=0, **kw):
    keys, x, b, lab, _ = heads(seed, **kw)
    return ({k: float(x[i]) for i, k in enumerate(keys)},
            {k: float(b[i]) for i, k in enumerate(keys)},
            {k: bool(lab[i]) for i, k in enumerate(keys)})


def rate(fn, n=120, **kw):
    hits = seen = 0
    for s in range(n):
        got = fn(s, **kw)
        if got is None:
            continue
        seen += 1
        hits += int(got)
    return hits / max(1, seen), seen


# ---------------------------------------------------------------------------
class TestTheFloorAndTheRegisteredNull:

    def test_a_threshold_classification_floors_the_design_at_one(self):
        """
        The finding, and it needs no draws: with the classification a cutoff on
        the behavioural score, nothing is above an induction head to straddle
        it with, so no set survives and no input whatever could reject.
        """
        _, _, b, lab, layer = heads(1, rho_c=1.0)
        ms = matched_sets(b, lab, layer, key="score")
        assert ms["n_sets"] == 0
        assert ms["n_dropped"] == K_INDUCTION
        floor = attainable_floor_report(0, 0, N_CONTROLS_PER_INDUCTION_HEAD, ALPHA)
        assert floor["attainable_floor"] == 1.0
        assert floor["sufficient"] is False

    def test_the_gate_refuses_it_and_says_which_constraint_it_is(self):
        res = p_value_p_i3(*dicts(2, rho_c=1.0), "two_stage", alpha=ALPHA)
        assert res["p_value"] is None
        assert res["verdict"] == "INSUFFICIENT"
        assert "constraint 2" in res["reason"]
        assert res["separation"]["perfectly_separated"] is True

    def test_the_registered_null_draws_impossible_classifications(self):
        rep = registered_null_invariance_report(N_HEADS, K_INDUCTION)
        assert rep["group_size"] == comb(N_HEADS, K_INDUCTION)
        assert rep["assignments_that_are_a_threshold_on_the_score"] == 1
        assert rep["nominal_floor"] == 1.0 / comb(N_HEADS, K_INDUCTION)

    def test_the_floor_is_the_design_s_and_carries_no_draw_count(self):
        """
        The null enumerates, so 6p's max-of-two rule applies with one term
        absent -- and the report says that rather than omitting it, because an
        omitted sampling floor and a forgotten one look identical.
        """
        f = attainable_floor_report(6, 6, 4, ALPHA)
        assert f["design_floor"] == pytest.approx(5.0 ** -6)
        assert f["sampling_floor"] is None
        assert f["binds"] == "design"
        assert "ENUMERATES" in f["_note"]

    def test_the_minimum_design_is_arithmetic_on_the_control_count(self):
        for m, expected in ((2, 3), (4, 2), (8, 2)):
            got = attainable_floor_report(0, 0, m, ALPHA)[
                "min_informative_sets_for_alpha"]
            assert got == expected, (m, got)
            assert (m + 1.0) ** -got <= ALPHA
            assert (m + 1.0) ** -(got - 1) > ALPHA

    def test_an_all_tied_set_is_enumerated_out_rather_than_counted_in(self):
        """
        CLAIM-C's informative rows (6l) and P-AB1's even ablation grids (6q),
        reached by a third group: a set whose motif rates are all equal adds
        the same number to the observation and to every draw.
        """
        sets = [(0, [1, 2]), (3, [4, 5])]
        values = np.array([1.0, 0.0, 0.0, 7.0, 7.0, 7.0])
        arm = exact_rank_arm(values, sets)
        assert arm["n_sets"] == 2
        assert arm["n_informative_sets"] == 1
        assert arm["design_floor"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
class TestTheExactArm:

    def test_the_null_is_exact_by_enumeration(self):
        """
        Every assignment of the induction slot within every set, enumerated:
        the p-value of the largest attainable statistic is exactly the floor.
        """
        sets = [(0, [1, 2, 3, 4]), (5, [6, 7, 8, 9])]
        v = np.zeros(10)
        v[0] = v[5] = 1.0
        arm = exact_rank_arm(v, sets)
        assert arm["p_greater"] == pytest.approx(5.0 ** -2)
        assert arm["design_floor"] == pytest.approx(5.0 ** -2)

    def test_the_two_tails_are_a_partition_up_to_the_observed_atom(self):
        sets = [(0, [1, 2, 3, 4]), (5, [6, 7, 8, 9])]
        v = np.arange(10, dtype=float)
        arm = exact_rank_arm(v, sets)
        assert arm["p_greater"] + arm["p_less"] >= 1.0
        assert 0 < arm["p_greater"] <= 1.0
        assert 0 < arm["p_less"] <= 1.0

    def test_ties_stay_exact_rather_than_becoming_approximate(self):
        """
        Mid-ranks, so a set with tied motif rates contributes a distribution
        with fewer atoms rather than a wrong one. Most heads carry no relay at
        all, so ties are the expected case and not an edge case.
        """
        sets = [(0, [1, 2])]
        arm = exact_rank_arm(np.array([1.0, 1.0, 0.0]), sets)
        # the induction head ties the larger control: mid-rank 2.5 of 1,2.5,2.5
        assert arm["mean_rank"] == pytest.approx(2.5)
        assert arm["p_greater"] == pytest.approx(2.0 / 3.0)

    def test_a_non_finite_rate_is_refused_rather_than_imputed(self):
        with pytest.raises(CrossHeadRefused, match="undefined rate is not a zero"):
            exact_rank_arm(np.array([np.nan, 0.0, 1.0]), [(0, [1, 2])])

    def test_valid_under_the_plain_h0(self):
        def one(s):
            res = p_value_p_i3(*dicts(500 + s, main=0.0), "two_stage", alpha=ALPHA)
            return None if res["p_value"] is None else res["p_value"] <= ALPHA
        r, seen = rate(one)
        assert seen >= 100
        assert r <= 0.12                      # 120 draws, nominal 0.05

    def test_valid_when_the_motif_tracks_the_score_with_no_effect(self):
        """
        The tautology's non-degenerate form: the motif tracks the behavioural
        score, and nothing is carried by the classification itself. The
        matching is what has to remove it.
        """
        def one(s):
            res = p_value_p_i3(*dicts(700 + s, main=1.0), "two_stage", alpha=ALPHA)
            return None if res["p_value"] is None else res["p_value"] <= ALPHA
        r, seen = rate(one)
        assert seen >= 100
        assert r <= 0.12

    def test_power_against_a_planted_effect(self):
        def one(s):
            res = p_value_p_i3(*dicts(900 + s, effect=1.5), "two_stage", alpha=ALPHA)
            return None if res["p_value"] is None else res["p_value"] <= ALPHA
        r, _ = rate(one)
        assert r >= 0.5

    def test_the_falsification_branch_is_one_that_can_fire(self):
        """
        6i's requirement. A branch nothing can trigger is not a branch.
        """
        def one(s):
            res = p_value_p_i3(*dicts(1100 + s, effect=-1.5), "two_stage",
                               alpha=ALPHA)
            return (None if res["p_value"] is None
                    else res["verdict"] == "ACTIVATION_PROPERTY")
        r, _ = rate(one)
        assert r >= 0.5


# ---------------------------------------------------------------------------
class TestWhatTheStatisticDegeneratesOn:

    def test_selection_attenuates_the_correlation_and_not_the_slope(self):
        """
        ONE population, ONE relation, no interaction anywhere. The registered
        wording's statistic still reads a large negative contrast, in the
        falsifier's direction, on nothing but where the classification cut.
        """
        ri, rc, si, sc = [], [], [], []
        for s in range(60):
            _, x, b, lab, _ = heads(1300 + s, rho_c=1.0, main=1.0)
            rep = correlation_contrast_report(x, b, lab)
            ri.append(rep["spearman_induction_heads"])
            rc.append(rep["spearman_control_heads"])
            si.append(rep["slope_induction_heads"])
            sc.append(rep["slope_control_heads"])
        assert np.nanmean(ri) < np.nanmean(rc) - 0.2      # attenuated
        # the slope is unbiased -- against the induction arm's own spread,
        # which is the point of the pair of assertions
        se = np.nanstd(si) / np.sqrt(len(si))
        assert abs(np.nanmean(si) - np.nanmean(sc)) <= 3 * se
        assert np.nanstd(si) > 5 * np.nanstd(sc)

    def test_the_correlation_contrast_is_reported_and_never_adjudicated(self):
        res = p_value_p_i3(*dicts(1500), "two_stage", alpha=ALPHA)
        cc = res["correlation_contrast"]
        assert cc["adjudicated"] is False
        assert "degenerates on" in cc["_why_not"]
        assert cc["induction_score_spread_ratio"] < 1.0

    def test_the_separation_report_reads_two_vectors_and_nothing_else(self):
        """
        Decidable before a checkpoint's edges are counted, which is where a
        requirement on a pilot belongs -- 6o's refusal has the same posture.
        """
        _, _, b, lab, _ = heads(1700, rho_c=1.0)
        rep = separation_report(b, lab)
        assert rep["perfectly_separated"] is True
        assert rep["control_heads_above_it"] == 0
        assert rep["induction_heads_with_a_control_above_them"] == 0
        _, _, b2, lab2, _ = heads(1700, rho_c=0.6)
        assert separation_report(b2, lab2)["perfectly_separated"] is False

    def test_an_empty_arm_is_refused_rather_than_scored(self):
        b = np.arange(10, dtype=float)
        with pytest.raises(CrossHeadRefused, match="mandatory rather than optional"):
            separation_report(b, np.zeros(10, dtype=bool))


# ---------------------------------------------------------------------------
class TestTheMatching:

    def test_a_one_sided_match_reads_curvature_as_an_effect(self):
        """
        The discarded matching, pinned. An induction head's NEAREST control
        heads are almost all below it in score, so the residual gap is
        one-signed and curvature is read as an effect.
        """
        def nearest(b, lab, m):
            ind = np.flatnonzero(lab)
            avail = list(np.flatnonzero(~lab))
            out = []
            for i in ind[np.argsort(b[ind], kind="mergesort")]:
                d = np.abs(b[np.array(avail)] - b[i])
                pick = [avail[t] for t in np.argsort(d, kind="mergesort")[:m]]
                for j in pick:
                    avail.remove(j)
                out.append((int(i), [int(j) for j in pick]))
            return out

        near = strad = seen = 0
        for s in range(120):
            _, x, b, lab, layer = heads(1900 + s, main=1.0, curve=0.8)
            ns = nearest(b, lab, N_CONTROLS_PER_INDUCTION_HEAD)
            ss = matched_sets(b, lab, layer, key="score")["sets"]
            if not ns or not ss:
                continue                       # nothing to compare on this draw
            seen += 1
            near += int(exact_rank_arm(x, ns)["p_greater"] <= ALPHA)
            strad += int(exact_rank_arm(x, ss)["p_greater"] <= ALPHA)
        assert seen >= 100
        assert near / seen > 0.10              # leaks
        assert strad / seen <= 0.10            # does not

    def test_every_retained_set_straddles_its_induction_head(self):
        _, _, b, lab, layer = heads(2100)
        ms = matched_sets(b, lab, layer, key="score")
        half = N_CONTROLS_PER_INDUCTION_HEAD // 2
        for i, ctrl in ms["sets"]:
            c = b[np.array(ctrl)]
            assert (c <= b[i]).sum() >= half
            assert (c > b[i]).sum() >= half
        assert matching_report(b, ms["sets"],
                               N_CONTROLS_PER_INDUCTION_HEAD)["all_sets_straddle"]

    def test_the_dropped_heads_are_the_top_of_the_ranking_and_are_named(self):
        """
        Dropping is not neutral: what cannot be straddled is the highest-scoring
        induction heads, which are the ones the prediction is most about. The
        record names them rather than reporting a count.
        """
        _, _, b, lab, layer = heads(2300)
        ms = matched_sets(b, lab, layer, key="score")
        assert ms["n_dropped"] >= 1
        assert len(ms["dropped_score_ranks"]) == ms["n_dropped"]
        assert min(ms["dropped_score_ranks"]) > max(ms["retained_score_ranks"])

    def test_a_control_head_is_never_used_twice(self):
        _, _, b, lab, layer = heads(2500)
        used = [j for _, ctrl in matched_sets(b, lab, layer, key="score")["sets"]
                for j in ctrl]
        assert len(used) == len(set(used))

    def test_the_matching_is_deterministic(self):
        _, _, b, lab, layer = heads(2700)
        a = matched_sets(b, lab, layer, key="score")["sets"]
        c = matched_sets(b, lab, layer, key="score")["sets"]
        assert a == c

    def test_an_odd_control_count_is_refused_because_the_straddle_needs_halves(self):
        _, _, b, lab, layer = heads(2900)
        with pytest.raises(CrossHeadRefused, match="even count"):
            matched_sets(b, lab, layer, key="score", n_controls=3)

    def test_the_layer_key_needs_layers(self):
        _, _, b, lab, _ = heads(3100)
        with pytest.raises(CrossHeadRefused, match="needs a layer"):
            matched_sets(b, lab, None, key="score_and_layer")

    def test_the_layer_key_draws_every_control_from_the_head_s_own_layer(self):
        _, _, b, lab, layer = heads(3300)
        for i, ctrl in matched_sets(b, lab, layer, key="score_and_layer")["sets"]:
            assert all(layer[j] == layer[i] for j in ctrl)


# ---------------------------------------------------------------------------
class TestTheGate:

    def test_the_three_branches_and_only_one_is_a_falsification(self):
        assert gate_verdict(0.01, 1.0, ALPHA)["verdict"] == "TRACKS_CLASSIFICATION"
        assert gate_verdict(1.0, 0.01, ALPHA)["verdict"] == "ACTIVATION_PROPERTY"
        assert gate_verdict(1.0, 0.01, ALPHA)["falsified"] is True
        assert gate_verdict(0.4, 0.4, ALPHA)["verdict"] == "INSUFFICIENT"
        assert gate_verdict(None, None, ALPHA)["verdict"] == "INSUFFICIENT"
        assert gate_verdict(0.4, 0.4, ALPHA)["falsified"] is False

    def test_the_same_rate_reading_is_the_null_and_reaches_insufficient(self):
        """
        6k's rule, a fourth time: the registered falsifier ("non-induction
        heads carry the motif at the SAME rate") describes the null, and an
        e-process records insufficient evidence rather than a null accepted.
        """
        v = gate_verdict(0.4, 0.4, ALPHA)
        assert "null" in v["reading"]
        assert v["falsified"] is False

    def test_the_independence_source_is_required_and_not_defaulted(self):
        res = p_value_p_i3(*dicts(3500), "none", alpha=ALPHA)
        assert res["p_value"] is None
        assert "constraint 2" in res["reason"]
        for src in INDEPENDENCE_SOURCES:
            assert p_value_p_i3(*dicts(3500), src, alpha=ALPHA)["p_value"] is not None

    def test_a_refused_record_still_says_how_much_of_the_design_survived(self):
        res = p_value_p_i3(*dicts(3700, rho_c=1.0), "two_stage", alpha=ALPHA)
        assert res["p_value"] is None
        assert res["separation"] is not None
        assert res["matched_sets"] is not None
        assert res["floor"]["attainable_floor"] == 1.0
        assert res["correlation_contrast"] is not None

    def test_a_design_that_cannot_clear_alpha_is_refused_before_it_is_scored(self):
        res = p_value_p_i3(*dicts(3900, k=2, rho_c=0.5), "two_stage", alpha=ALPHA)
        if res["p_value"] is None and "cannot express" in (res["reason"] or ""):
            assert res["floor"]["attainable_floor"] > ALPHA
        else:                                   # enough sets survived
            assert res["floor"]["sufficient"] is True

    def test_undefined_rates_are_dropped_and_counted_rather_than_imputed(self):
        m, b, lab = dicts(4100)
        for key in list(m)[:5]:
            m[key] = float("nan")
        res = p_value_p_i3(m, b, lab, "two_stage", alpha=ALPHA)
        assert res["n_undefined_dropped"] == 5

    def test_the_alternative_and_its_reciprocal_are_fixed_in_advance(self):
        assert P_I3_ALTERNATIVE == "greater"
        assert P_I3_RECIPROCAL_ALTERNATIVE == "less"
        assert REGISTERED_EXCHANGEABLE_UNIT == "head"

    def test_the_layer_concentration_is_reported_on_every_record(self):
        res = p_value_p_i3(*dicts(4300, confine_to_band=True), "two_stage",
                           alpha=ALPHA)
        lc = res["layer_concentration"]
        assert lc["n_layers_represented"] <= 6
        assert lc["layer_span"][0] >= 6 and lc["layer_span"][1] <= 11


# ---------------------------------------------------------------------------
class TestTheLimitationTheScoreKeyDoesNotRemove:

    def test_the_layer_key_removes_the_band_confound_and_costs_power(self):
        """
        The registered decision, with both sides measured. A shared elevation
        across the layers the induction heads occupy is invisible to a control
        matched on score alone; matching within the layer removes it and costs
        informative sets and power.
        """
        def confound(key):
            hits = seen = 0
            for s in range(80):
                res = p_value_p_i3(*dicts(4500 + s, main=1.0,
                                          band_elevation=1.0,
                                          confine_to_band=True),
                                   "two_stage", key=key, alpha=ALPHA)
                if res["p_value"] is None:
                    continue
                seen += 1
                hits += int(res["verdict"] == "TRACKS_CLASSIFICATION")
            return hits / max(1, seen), seen

        score_rate, _ = confound("score")
        layer_rate, layer_seen = confound("score_and_layer")
        assert score_rate > 0.15                      # confounded
        assert layer_rate <= 0.15                     # not
        assert layer_seen < 80                        # and it costs emissions

    def test_the_concentration_diagnostic_sees_clustering_and_not_elevation(self):
        """
        Pinned so it cannot later be mistaken for coverage -- P-AB1's
        shared-prompt-factor diagnostic has the same test for the same reason.
        """
        _, _, _, lab, layer = heads(4700, confine_to_band=True)
        flat = layer_concentration_diagnostic(layer, lab)
        _, _, _, lab2, layer2 = heads(4700, confine_to_band=True,
                                      band_elevation=1.0)
        elevated = layer_concentration_diagnostic(layer2, lab2)
        assert flat == elevated                       # blind to the elevation
        assert flat["n_layers_represented"] <= 6      # sees the clustering


# ---------------------------------------------------------------------------
class TestTheRegisteredMatchingKey:
    """
    EVERY test here that asks to adjudicate passes an isolated
    `adjudications_dir`. While no key is registered the refusal is also the
    safety catch keeping a synthetic p-value out of P-I3's ledger slot, and
    `core.adjudication` refuses to overwrite a record once written, so a
    fixture run that reached the real directory would occupy the slot
    permanently. 6l recorded the same consequence for `P6-R2` and 6q found a
    defect behind it.
    """

    def _res(self, key="score", seed=4900):
        """
        The first seed at or after `seed` on which this key emits a p-value.
        Deterministic, and a search rather than a fixed seed because the
        registered key refuses a large minority of draws by design -- that
        refusal rate is the price the author weighed, and a fixture that
        pretended otherwise would be testing a different construction.
        """
        for s in range(seed, seed + 60):
            res = p_value_p_i3(*dicts(s, effect=1.5), "two_stage", key=key,
                               alpha=ALPHA)
            if res["p_value"] is not None:
                return res
        raise AssertionError(f"key={key!r} emitted on no seed in [{seed}, {seed+60})")

    def test_the_registered_key_is_the_one_the_measurement_pointed_at(self):
        assert REGISTERED_CONTROL_MATCHING_KEY == "score_and_layer"
        assert REGISTERED_CONTROL_MATCHING_KEY in CONTROL_MATCHING_KEYS

    def test_the_key_argument_does_not_route_around_the_constant(self):
        """
        `key=` selects what to COMPUTE. The module constant decides what may
        enter an e-process -- 6h's construction, and its reason.
        """
        other = [k for k in CONTROL_MATCHING_KEYS
                 if k != REGISTERED_CONTROL_MATCHING_KEY][0]
        res = self._res(other)
        assert res["p_value"] is not None
        with pytest.raises(CrossHeadRefused, match="registered key is"):
            adjudicate_p_i3(res, adjudicate=True)

    @pytest.mark.skipif(REGISTERED_CONTROL_MATCHING_KEY is None,
                        reason="no matching key is registered yet")
    def test_a_result_under_the_registered_key_reaches_the_ledger_path(self, tmp_path):
        res = self._res(REGISTERED_CONTROL_MATCHING_KEY)
        assert res["p_value"] is not None
        out = adjudicate_p_i3(res, adjudicate=True, adjudications_dir=tmp_path)
        assert out["adjudication"] is not None
        assert list(tmp_path.iterdir())            # written HERE and nowhere else

    @pytest.mark.skipif(REGISTERED_CONTROL_MATCHING_KEY is None,
                        reason="no matching key is registered yet")
    def test_nothing_is_adjudicated_when_the_gate_refused(self, tmp_path):
        res = p_value_p_i3(*dicts(5100, rho_c=1.0), "two_stage",
                           key=REGISTERED_CONTROL_MATCHING_KEY, alpha=ALPHA)
        assert res["p_value"] is None
        out = adjudicate_p_i3(res, adjudicate=True, adjudications_dir=tmp_path)
        assert out["adjudication"] is None
        assert not list(tmp_path.iterdir())

    @pytest.mark.skipif(REGISTERED_CONTROL_MATCHING_KEY is None,
                        reason="no matching key is registered yet")
    def test_the_ledger_is_not_touched_without_the_flag(self, tmp_path):
        out = adjudicate_p_i3(self._res(REGISTERED_CONTROL_MATCHING_KEY),
                              adjudications_dir=tmp_path)
        assert out["adjudication"] is None
        assert not list(tmp_path.iterdir())

    @pytest.mark.skipif(REGISTERED_CONTROL_MATCHING_KEY is None,
                        reason="no matching key is registered yet")
    def test_the_ledger_note_names_what_the_p_value_is_not(self, tmp_path):
        out = adjudicate_p_i3(self._res(REGISTERED_CONTROL_MATCHING_KEY),
                              adjudicate=True, adjudications_dir=tmp_path)
        notes = out["adjudication"]["notes"]
        assert "is NOT what this p-value tests" in notes
        assert "dropped as unstraddleable" in notes

    def test_the_real_adjudications_directory_is_not_created(self):
        """
        Asserted here rather than left to the call sites -- 6r added the same
        test for CLAIM-B and this is the second construction to carry it.
        """
        from tools.check_registry import ADJUDICATIONS
        assert not ADJUDICATIONS.exists()


# ---------------------------------------------------------------------------
class TestCommittedCalibration:
    """
    The measured rates, pinned. Recomputing them is minutes, which the
    ten-second gating tier does not have -- the same division of labour the
    five calibrations before this one use. The record stores its own
    `elapsed_seconds`, so the cost is a field rather than a claim in prose.
    """

    def _doc(self):
        from tools.calibrate_cross_head_association import OUT_PATH, SCHEMA_VERSION
        doc = json.loads(OUT_PATH.read_text())
        assert doc["schema_version"] == SCHEMA_VERSION
        return doc

    def test_the_record_still_supports_the_section_it_is_evidence_for(self):
        from tools.calibrate_cross_head_association import check_record
        assert check_record(self._doc()) == []

    def test_the_record_matches_the_construction_it_measured(self):
        from tools.calibrate_cross_head_association import (
            CONSTRUCTION_PATH, _sha256)
        assert self._doc()["construction_sha256"] == _sha256(CONSTRUCTION_PATH)

    def test_the_artifact_describes_the_design_it_measured(self):
        doc = self._doc()
        assert doc["alpha"] == ALPHA
        assert doc["design"]["n_heads"] == N_HEADS
        assert doc["design"]["n_induction_heads"] == K_INDUCTION
        assert (doc["design"]["n_controls_per_induction_head"]
                == N_CONTROLS_PER_INDUCTION_HEAD)
        assert doc["replicates"] >= 300
        assert doc["self_check"]["ok"] is True
        assert doc["elapsed_seconds"] > 0

    def test_the_registered_null_discriminates_nothing(self):
        rows = {r["statistic"]: r
                for r in self._doc()["registered_null"]["rejection_rates"]}
        assert set(rows) == {"correlation_contrast", "slope_contrast"}
        for r in rows.values():
            assert abs(r["discrimination"]) <= 0.10
        assert rows["slope_contrast"]["h0"] >= 0.10      # and is not a null

    def test_the_selection_attenuation_is_in_the_falsifier_s_direction(self):
        row = {r["n_induction_heads"]: r for r in
               self._doc()["degeneracy"]["selection_attenuation"]}[K_INDUCTION]
        assert row["spearman_contrast"] <= -0.20
        assert row["slope_sd_ratio"] >= 5.0

    def test_the_tautology_family_is_refused_on_every_draw(self):
        rows = {(r["family"], r["matching_key"]): r for r in self._doc()["validity"]}
        for key in CONTROL_MATCHING_KEYS:
            r = rows[("tautology", key)]
            assert r["refusal_rate"] == 1.0
            assert r["counterfactual_rate_nearest_matching"] > 0.20

    def test_the_tautology_leak_grows_with_the_score_to_motif_relation(self):
        """
        The danger constraint 2 names is not a literal identity -- that is the
        degenerate case P-I1's gate already refuses -- but the motif tracking
        the score at all. As a curve rather than as a caution.
        """
        leak = {r["score_to_motif_relation"]: r
                for r in self._doc()["grid"]["tautology_leak"]}
        assert leak[0.0]["nearest_matching_rejection"] <= 0.15
        assert max(leak)  # the strongest relation
        assert leak[max(leak)]["nearest_matching_rejection"] > 0.5
        assert all(r["straddled_matched_sets"] == 0 for r in leak.values())

    def test_no_h0_family_confirms(self):
        for r in self._doc()["validity"]:
            if r["correct_verdict"] == "INSUFFICIENT":
                assert r["tracks_classification_rate"] <= 0.15, r["family"]

    def test_both_branches_fire_on_the_inputs_built_for_them(self):
        rows = {(r["family"], r["matching_key"]): r for r in self._doc()["validity"]}
        assert rows[("effect-1.5", "score")]["tracks_classification_rate"] >= 0.5
        assert rows[("reciprocal-1.5", "score")]["activation_property_rate"] >= 0.5

    def test_the_control_count_was_fixed_where_power_stops_rising(self):
        front = {r["n_controls"]: r
                 for r in self._doc()["grid"]["n_controls_frontier"]}
        chosen = front[N_CONTROLS_PER_INDUCTION_HEAD]["power_at_effect_0.8"]
        assert all(r["power_at_effect_0.8"] <= chosen + 0.05
                   for m, r in front.items() if m > N_CONTROLS_PER_INDUCTION_HEAD)
        ranks = [front[m]["mean_score_rank_of_retained_induction_heads"]
                 for m in sorted(front)]
        assert ranks[0] > ranks[-1]

    def test_the_overlap_frontier_starts_at_nothing(self):
        ovl = {r["classification_correlation_with_the_score"]: r
               for r in self._doc()["grid"]["overlap_frontier"]}
        assert ovl[1.0]["mean_matched_sets"] == 0.0
        assert ovl[1.0]["share_of_draws_whose_floor_clears_alpha"] == 0.0
        assert ovl[0.8]["mean_matched_sets"] > 3.0

    def test_the_layer_band_confound_and_its_price_are_both_in_the_record(self):
        band = {(r["shared_elevation_on_the_induction_band_sd"],
                 r["matching_key"]): r
                for r in self._doc()["limitation"]["layer_band"]}
        assert band[(1.0, "score")]["tracks_classification_rate"] > 0.15
        assert band[(1.0, "score_and_layer")]["tracks_classification_rate"] <= 0.15
        pw = {(r["planted_effect"], r["matching_key"]): r
              for r in self._doc()["limitation"]["layer_band_power"]}
        assert pw[(1.5, "score")]["power"] > pw[(1.5, "score_and_layer")]["power"]
        assert pw[(1.5, "score_and_layer")]["emitted_rate"] < 1.0
