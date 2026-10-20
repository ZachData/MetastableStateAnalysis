"""
tests/test_tools_design_preflight.py — the preflight refuses the designs on record that could
not have rejected, and passes one that can.

Each refusal is tested on a reconstruction of the instance in `LESSONS.md` 6 that motivated it.
Those are the tool's known answers: a preflight that let any of them through would not have
saved the run it was built to save. The clean design is the planted effect: the tool must not
refuse what can reject.

Tier: pure (numpy, scipy).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.pure

from tools import design_preflight as dp


def _clean(**over) -> dp.Inputs:
    """A design that can reject: N(0, 1) nulls of 2000, doses +1.5, +3, +4.5 σ."""
    rng = np.random.default_rng(0)
    d = dict(statistic="shift", direction="greater",
             nulls={"step0": rng.normal(0, 1, 2000).tolist(), "random": rng.normal(0, 1, 2000).tolist()},
             trained=rng.normal(0.5, 1, 400).tolist(), trained_step="512",
             planted={"0": rng.normal(0, 1, 400).tolist(),
                      **{str(m): rng.normal(m, 1, 400).tolist() for m in (1.5, 3, 4.5)}},
             alpha=0.05, kappa=0.5)
    d.update(over)
    return dp.Inputs(**d)


def _codes(report) -> set:
    return {r["code"] for r in report["refusals"]}


class TestCleanDesign:

    def test_a_design_that_can_reject_passes(self):
        r = dp.preflight(_clean())
        assert r["verdict"] == "pass", r["refusals"]
        assert r["floor"]["max_attainable_E"] >= 20

    def test_the_detection_curve_rises_with_dose(self):
        curve = dp.preflight(_clean())["detection_curve"]
        rates = [curve[d]["detected"]["step0"] for d in ("1.5", "3", "4.5")]
        assert rates == sorted(rates) and rates[-1] > 0.95

    def test_a_less_side_design_reads_the_lower_tail(self):
        flip = lambda xs: [-x for x in xs]
        c = _clean()
        r = dp.preflight(dp.Inputs(**{**c.__dict__, "direction": "less",
                                      "nulls": {k: flip(v) for k, v in c.nulls.items()},
                                      "trained": flip(c.trained),
                                      "planted": {k: flip(v) for k, v in c.planted.items()}}))
        assert r["verdict"] == "pass", r["refusals"]


class TestTheFloor:
    """`P-S1`: 500 draws put the best rank p at 1/501, E 11.2 < 20."""

    def test_p_s1_s_500_draws_are_refused(self):
        r = dp.preflight(_clean(n_null_draws=500))
        assert "floor" in _codes(r)
        assert r["floor"]["max_attainable_E"] == pytest.approx(11.19, abs=0.01)
        assert r["floor"]["min_null_draws"] == 1599

    def test_1599_draws_reach_20_and_1598_do_not(self):
        assert "floor" not in _codes(dp.preflight(_clean(n_null_draws=1599)))
        assert "floor" in _codes(dp.preflight(_clean(n_null_draws=1598)))

    def test_an_explicit_p_floor_is_used(self):
        assert "floor" in _codes(dp.preflight(_clean(p_floor=0.01)))


class TestTheBand:
    """B2's design: step 0's band 0.4963–0.5037; 99.7 % of trained heads outside it."""

    def _b2(self, **over):
        rng = np.random.default_rng(1)
        null = rng.normal(0.5, 0.0019, 2000).tolist()
        trained = rng.uniform(0.2, 0.8, 384).tolist()
        return _clean(nulls={"step0": null}, trained=trained,
                      planted={"0": rng.normal(0.5, 0.0019, 400).tolist(),
                               **{str(m): rng.normal(0.5 + m * 0.002, 0.0019, 400).tolist()
                                  for m in (2, 4, 6)}}, **over)

    def test_b2_s_band_is_refused(self):
        r = dp.preflight(self._b2())
        assert "band:step0" in _codes(r)
        assert r["nulls"]["step0"]["trained_outside_band"] > 0.95

    def test_declared_as_a_moved_mark_it_is_not_a_null(self):
        r = dp.preflight(self._b2(moved_marks=["step0"]))
        assert "band:step0" not in _codes(r)
        assert "band" in _codes(r)   # and then nothing is left to test against


class TestTheCeiling:
    """A2 unit 5: the share the plant had to raise was already 1.000 unablated."""

    def test_a2_s_control_is_refused(self):
        rng = np.random.default_rng(2)
        r = dp.preflight(_clean(bounds=[0, 1],
                                nulls={"step0": rng.uniform(0, 1, 2000).tolist()},
                                trained=rng.uniform(0, 1, 400).tolist(),
                                planted={"0": [1.0] * 50, "1": [1.0] * 50, "2": [1.0] * 50, "3": [1.0] * 50}))
        assert "ceiling" in _codes(r)
        assert r["control"]["headroom"] == 0

    def test_no_bounds_is_stated_not_passed_silently(self):
        r = dp.preflight(_clean())
        assert r["control"]["headroom"] is None and "not checked" in r["control"]["headroom_note"]

    def test_a_control_that_reads_an_effect_unplanted_is_refused(self):
        rng = np.random.default_rng(3)
        c = _clean()
        r = dp.preflight(dp.Inputs(**{**c.__dict__,
                                      "planted": {**c.planted, "0": rng.normal(2, 1, 400).tolist()}}))
        assert "size" in _codes(r)


class TestTheDose:
    """B2's re-gate: |p − ½| has null mean ¼; a set held at the 70th percentile scores 0.20."""

    def test_b2_s_70th_percentile_dose_is_refused(self):
        rng = np.random.default_rng(4)
        stat = lambda p: abs(p - 0.5)
        null = [stat(p) for p in rng.uniform(0, 1, 2000)]
        r = dp.preflight(_clean(nulls={"random": null},
                                trained=[stat(p) for p in rng.uniform(0, 1, 400)],
                                planted={"0": [stat(p) for p in rng.uniform(0, 1, 400)],
                                         "70": [0.20] * 100, "95": [0.45] * 100, "99": [0.49] * 100}))
        assert "dose:70" in _codes(r)
        assert r["nulls"]["random"]["mean"] == pytest.approx(0.25, abs=0.01)
        assert "dose:95" not in _codes(r)


class TestPower:
    """A3 unit 2: the effect arm could not place its own planted move."""

    def test_a_largest_dose_seen_rarely_is_refused(self):
        rng = np.random.default_rng(5)
        r = dp.preflight(_clean(planted={"0": rng.normal(0, 1, 400).tolist(),
                                         **{str(m): rng.normal(m, 1, 400).tolist() for m in (0.2, 0.4, 0.6)}}))
        assert "power" in _codes(r)


class TestShape:

    def test_fewer_than_three_doses_is_refused(self):
        c = _clean()
        r = dp.preflight(dp.Inputs(**{**c.__dict__, "planted": {k: c.planted[k] for k in ("0", "3")}}))
        assert "doses" in _codes(r)

    def test_no_dose_zero_is_refused(self):
        c = _clean()
        r = dp.preflight(dp.Inputs(**{**c.__dict__,
                                      "planted": {k: v for k, v in c.planted.items() if k != "0"}}))
        assert "doses" in _codes(r)

    def test_an_empty_or_nan_population_is_refused(self):
        assert "empty" in _codes(dp.preflight(_clean(trained=[])))
        assert "empty" in _codes(dp.preflight(_clean(trained=[0.1, float("nan")])))

    def test_an_unknown_moved_mark_is_refused(self):
        assert "moved_marks" in _codes(dp.preflight(_clean(moved_marks=["nope"])))

    def test_a_bad_direction_raises(self):
        with pytest.raises(ValueError):
            dp.preflight(_clean(direction="two-sided"))


class TestCli:

    def test_the_cli_writes_a_citable_report_and_exits_by_verdict(self, tmp_path):
        c = _clean()
        good = tmp_path / "good.json"
        good.write_text(json.dumps({**c.__dict__, "extra": None} | {"note": "kept"}))
        out = tmp_path / "r.json"
        assert dp.main([str(good), "--out", str(out)]) == 0
        rep = json.loads(out.read_text())
        assert rep["verdict"] == "pass" and len(rep["input_sha256"]) == 64
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({**c.__dict__, "extra": None, "n_null_draws": 500}))
        assert dp.main([str(bad)]) == 1

    def test_evaluate_applies_a_statistic_to_each_population(self):
        assert dp.evaluate(len, {"a": [[1, 2], [3]], "b": []}) == {"a": [2.0, 1.0], "b": []}
