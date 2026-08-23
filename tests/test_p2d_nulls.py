"""
tests/test_p2d_nulls.py — P-T1 and P-M1 null constructions (POPPER_PLAN.md B6).

The last two predictions that are both `e-value` and `active`. With P-S1 these
are everything this project can currently adjudicate, so between them these
three tests cover the whole live falsification surface.

Both are permutation tests, and both permute the thing the prediction's own
falsifier names — P-T1 breaks the association between classification and
modality ("trimodality is a property of the activations, not of the
classification"); P-M1 breaks the association between regime distance and
violations ("violations are not explained by leaving the gradient-flow
regime"). That is not a coincidence: a falsifier stated well enough to be
falsifiable usually names its own null.
"""

from __future__ import annotations

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable.
pytestmark = pytest.mark.pure

from p2d_operator_activation.gradient_flow_condition import (
    P_M1_ALTERNATIVE,
    P_M1_PRIMARY_AGGREGATE,
    adjudicate_p_m1_from_regimes,
    p_value_p_m1,
)
from p2d_operator_activation.table1_predictions import (
    P_T1_ALTERNATIVE,
    P_T1_TARGET_MODES,
    adjudicate_p_t1_from_results,
    p_value_p_t1,
)


def _head(candidate: bool, stable_modes, spaced: bool = True) -> dict:
    return {
        "row2_candidate": candidate,
        "stability": {"stable_n_modes": stable_modes},
        "modality": {"trimodal": stable_modes == 3, "equally_spaced": spaced},
    }


def _regimes(distances, threshold: float = 0.1) -> list:
    return [{"layer": l, "regime_distance": float(x),
             "in_gradient_flow_regime": bool(x < threshold)}
            for l, x in enumerate(distances)]


# ---------------------------------------------------------------------------
# P-T1
# ---------------------------------------------------------------------------

class TestPT1:

    def test_adjudicates_on_stable_n_modes_not_the_single_bandwidth_flag(self):
        """
        The amendment's central instruction. Here every head's single-bandwidth
        `modality["trimodal"]` says True while the bandwidth scan says the mode
        count is 1 — the exact case the addendum was written for ("any
        distribution can be made multimodal by under-smoothing"). The p-value
        must follow the scan.
        """
        results = [_head(True, 1) for _ in range(8)] + [_head(False, 1) for _ in range(8)]
        for r in results:
            r["modality"]["trimodal"] = True          # the misleading flag
        out = p_value_p_t1(results, n_perm=500)
        assert out["adjudicated_on"] == "stable_n_modes"
        assert out["trimodal_rate_candidates"] == 0.0
        assert out["p_value"] > 0.5

    def test_undetermined_modality_is_excluded_not_resolved(self):
        """
        `stable_n_modes is None` means the data does not determine the head's
        modality. Filling it in either direction would invent a measurement, so
        it is dropped from both arms and counted.
        """
        results = ([_head(True, None) for _ in range(5)]
                   + [_head(True, 3) for _ in range(4)]
                   + [_head(False, 1) for _ in range(4)])
        out = p_value_p_t1(results, n_perm=500)
        assert out["n_undetermined"] == 5
        assert out["n_candidates"] == 4        # the None heads are not counted
        assert out["n_controls"] == 4

    def test_detects_a_real_association(self):
        results = [_head(True, 3) for _ in range(8)] + [_head(False, 1) for _ in range(12)]
        out = p_value_p_t1(results, n_perm=2000)
        assert out["p_value"] < 0.01
        assert out["trimodal_rate_candidates"] == 1.0
        assert out["trimodal_rate_controls"] == 0.0

    def test_no_association_gives_an_unremarkable_p(self):
        rng = np.random.default_rng(0)
        results = [_head(bool(i % 2), int(rng.choice([1, 3]))) for i in range(30)]
        assert p_value_p_t1(results, n_perm=2000)["p_value"] > 0.05

    def test_calibrated_under_the_null(self):
        """
        Labels assigned at random => H0 exactly. The rejection rate at nominal
        must not exceed it. A permutation test is exact here by construction,
        so this is a wiring check as much as a calibration one.
        """
        rng = np.random.default_rng(7)
        ps = []
        for _ in range(120):
            tri = rng.random(24) < 0.4
            lab = rng.permutation([True] * 10 + [False] * 14)
            results = [_head(bool(c), 3 if t else 1) for c, t in zip(lab, tri)]
            ps.append(p_value_p_t1(results, n_perm=200)["p_value"])
        ps = np.array(ps)
        assert (ps <= 0.05).mean() <= 0.12, f"rejection rate {(ps <= 0.05).mean():.3f}"
        assert 0.35 < ps.mean() < 0.65

    def test_one_sided_ignores_controls_beating_candidates(self):
        results = [_head(True, 1) for _ in range(8)] + [_head(False, 3) for _ in range(8)]
        assert p_value_p_t1(results, n_perm=1000)["p_value"] > 0.9

    def test_missing_arm_is_a_measurement_not_a_test(self):
        out = p_value_p_t1([_head(True, 3) for _ in range(6)], n_perm=200)
        assert out["p_value"] is None
        assert "No candidates at all is itself a result" in out["reason"] \
            or "need both arms" in out["reason"]

    def test_declared_constants(self):
        assert P_T1_TARGET_MODES == 3
        assert P_T1_ALTERNATIVE == "greater"

    def test_spacing_reported_but_not_adjudicated(self):
        results = [_head(True, 3, spaced=False) for _ in range(6)] + \
                  [_head(False, 1) for _ in range(6)]
        out = p_value_p_t1(results, n_perm=500)
        assert out["equally_spaced_rate"] == 0.0     # reported
        assert out["p_value"] < 0.05                 # and not used in the p-value
        assert "NOT adjudicated" in out["spacing_note"]


# ---------------------------------------------------------------------------
# P-M1
# ---------------------------------------------------------------------------

class TestPM1:

    def test_detects_concentration_in_out_of_regime_layers(self):
        d = np.linspace(0, 1, 16)
        out = p_value_p_m1(_regimes(d), (d > 0.5).astype(float), n_perm=2000)
        assert out["p_value"] < 0.01
        assert out["observed"] > 0.5

    def test_shuffled_violations_give_an_unremarkable_p(self):
        rng = np.random.default_rng(3)
        d = np.linspace(0, 1, 16)
        v = rng.permutation((d > 0.5).astype(float))
        assert p_value_p_m1(_regimes(d), v, n_perm=2000)["p_value"] > 0.05

    def test_one_sided_ignores_the_wrong_direction(self):
        """
        Violations concentrating in layers CLOSE to the gradient-flow condition
        is P-M1 being wrong, not P-M1 being right.
        """
        d = np.linspace(0, 1, 16)
        out = p_value_p_m1(_regimes(d), 1.0 - (d > 0.5).astype(float), n_perm=1000)
        assert out["p_value"] > 0.9

    def test_calibrated_under_the_null(self):
        rng = np.random.default_rng(11)
        d = np.linspace(0, 1, 16)
        regs = _regimes(d)
        ps = [p_value_p_m1(regs, rng.permutation((d > 0.5).astype(float)),
                           n_perm=200)["p_value"] for _ in range(80)]
        ps = np.array(ps)
        assert (ps <= 0.05).mean() <= 0.12, f"rejection rate {(ps <= 0.05).mean():.3f}"

    def test_refuses_when_aggregates_disagree_in_sign(self):
        """
        `adjudicate_p_m1` already establishes that sign disagreement across the
        mean/min/max head-to-layer aggregates means per-layer energies cannot
        resolve a per-head claim. Emitting a p-value for one chosen aggregate
        would convert that resolution limit into a result.
        """
        # Two heads per layer, arranged so mean rises with the violation series
        # while min falls -- the aggregates then disagree in sign.
        regs = []
        for l in range(12):
            hi = 0.1 + 0.08 * l
            lo = 0.9 - 0.07 * l
            for x in (hi, lo):
                regs.append({"layer": l, "regime_distance": float(x),
                             "in_gradient_flow_regime": bool(x < 0.1)})
        v = np.arange(12, dtype=float)
        out = p_value_p_m1(regs, v, n_perm=200)
        if out["p_value"] is None:
            assert "disagree in SIGN" in out["reason"]
        else:
            signs = {np.sign(a["corr"]) for a in out["aggregates"].values()
                     if np.isfinite(a["corr"])}
            assert len(signs) == 1, "refusal should have triggered"

    def test_records_the_series_convention(self):
        """
        UPDATE_PLAN.md 5.9: a violation is a per-boundary INDICATOR, not a
        per-layer count. The record says which convention it got rather than
        leaving it to be inferred.
        """
        d = np.linspace(0, 1, 12)
        binary = p_value_p_m1(_regimes(d), (d > 0.5).astype(float), n_perm=200)
        counts = p_value_p_m1(_regimes(d), np.arange(12, dtype=float), n_perm=200)
        assert binary["violation_series_is_binary"] is True
        assert counts["violation_series_is_binary"] is False
        assert binary["violation_series_len"] == 12

    def test_degenerate_input_returns_no_p_value(self):
        d = np.linspace(0, 1, 12)
        out = p_value_p_m1(_regimes(d), np.zeros(12), n_perm=200)
        assert out["p_value"] is None
        assert "degenerate" in out["reason"] or "no usable" in out["reason"]

    def test_declared_constants(self):
        assert P_M1_PRIMARY_AGGREGATE == "mean"
        assert P_M1_ALTERNATIVE == "greater"


# ---------------------------------------------------------------------------
# Ledger wiring
# ---------------------------------------------------------------------------

class TestWiring:

    def test_p_t1_opt_in(self, tmp_path):
        results = [_head(True, 3)] * 6 + [_head(False, 1)] * 6
        assert adjudicate_p_t1_from_results(
            results, n_perm=200, adjudications_dir=tmp_path)["adjudication"] is None
        assert not list(tmp_path.glob("*.json"))

    def test_p_t1_emits(self, tmp_path):
        results = [_head(True, 3)] * 6 + [_head(False, 1)] * 6
        adj = adjudicate_p_t1_from_results(
            results, n_perm=500, adjudicate=True,
            adjudications_dir=tmp_path)["adjudication"]
        assert adj["prediction_id"] == "P-T1" and adj["claim"] == "H-OPERATOR"
        assert "stable_n_modes == 3" in adj["test_name"]
        assert "n_undetermined=0" in adj["notes"]

    def test_p_m1_emits(self, tmp_path):
        d = np.linspace(0, 1, 16)
        adj = adjudicate_p_m1_from_regimes(
            _regimes(d), (d > 0.5).astype(float), n_perm=500, adjudicate=True,
            adjudications_dir=tmp_path)["adjudication"]
        assert adj["prediction_id"] == "P-M1" and adj["claim"] == "H-OPERATOR"
        assert "permutation over layers" in adj["test_name"]

    def test_both_accumulate_on_one_claim(self, tmp_path):
        """
        P-T1 and P-M1 are both H-OPERATOR, so their e-values multiply into one
        e-process — which is the whole point of the claim layer.
        """
        results = [_head(True, 3)] * 8 + [_head(False, 1)] * 12
        a = adjudicate_p_t1_from_results(results, n_perm=1000, adjudicate=True,
                                         adjudications_dir=tmp_path)["adjudication"]
        d = np.linspace(0, 1, 16)
        b = adjudicate_p_m1_from_regimes(_regimes(d), (d > 0.5).astype(float),
                                         n_perm=1000, adjudicate=True,
                                         adjudications_dir=tmp_path)["adjudication"]
        assert a["claim_sequence_index"] == 1
        assert b["claim_sequence_index"] == 2
        assert b["claim_E_after"] > a["claim_E_after"]

        from core.adjudication import verify_ledger
        assert verify_ledger(tmp_path) == []

    def test_real_ledger_untouched(self, tmp_path):
        from core.adjudication import load_adjudications
        before = {r["prediction_id"] for r in load_adjudications()}
        results = [_head(True, 3)] * 6 + [_head(False, 1)] * 6
        adjudicate_p_t1_from_results(results, n_perm=200, adjudicate=True,
                                     adjudications_dir=tmp_path)
        assert {r["prediction_id"] for r in load_adjudications()} == before
