"""
`tools/run/p10_partition_function.py` — F12, parked versus pinned.

The claims this row makes are all of the form "quantity X is mostly position,
and after the correction it is less so", so the tests are built around two
constructions whose answers are known exactly. `_idealised` is the regime
`math-10.md` §2 states its closed form in, where every inner product including
the self term equals a common gamma and `Z_i = (i+1) e^{beta gamma}` exactly.
`_realisable` is what sphere-projected activations can actually be — unit
diagonal — where `Z_i = e^beta + i e^{beta gamma}` is affine rather than
proportional, so the `(i+1)` correction leaves a residual trend. The second is
not a flaw in the construction; it is why the real sweep's corrected R^2 does
not go to zero, and a test says so.
"""
import json
import math

import numpy as np
import pytest

from tools.run.p10_partition_function import (
    BETAS,
    N_PERMUTATIONS,
    aggregate,
    measure_directory,
    measure_layer,
    percentile_of,
    position_r2,
    standardised_difference,
)

# Tier: numpy and scipy only -- no torch, transformers, sklearn or
# matplotlib -- so this runs in `scripts/check.sh pure`. Declared, not
# assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure



def _rng():
    return np.random.default_rng(0)


def _from_gram(G):
    w, V = np.linalg.eigh(G)
    return V @ np.diag(np.sqrt(np.clip(w, 0, None)))


def _idealised(n, gamma=0.9):
    """The IDEALISED concentration regime: EVERY inner product equals `gamma`,
    the self term included. This is the model `math-10.md` §2's closed form is
    stated in, and it gives exactly ``Z_i = (i+1) e^{beta gamma}``.

    It is not realisable by unit vectors -- their diagonal is 1, not gamma --
    which is the point of `_realisable` below."""
    return _from_gram(np.full((n, n), float(gamma)))


def _realisable(n, gamma=0.9):
    """The same, but with a unit diagonal, which is what sphere-projected
    activations actually have. Then

        Z_i = e^{beta} + i e^{beta gamma}

    which is affine in `i` rather than proportional to `(i+1)`, so dividing by
    `(i+1)` leaves a residual trend. This is why the real sweep's
    `position_r2_corrected` does not go to zero, and the tests say so rather
    than treating it as noise."""
    G = np.full((n, n), float(gamma))
    np.fill_diagonal(G, 1.0)
    return _from_gram(G)


# --- the pieces ------------------------------------------------------------

def test_position_r2_is_one_on_the_exact_masked_baseline():
    """`log Z = log(i+1) + const` in the concentration regime, so the fit is
    exact and R^2 must be 1."""
    n = 60
    v = np.log(np.arange(1, n + 1, dtype=float)) + 3.7
    assert position_r2(v) == pytest.approx(1.0)


def test_position_r2_is_near_zero_on_position_free_noise():
    v = np.random.default_rng(1).normal(size=500)
    assert position_r2(v) < 0.05


def test_position_r2_is_nan_on_a_constant_input_not_zero_or_one():
    assert math.isnan(position_r2(np.full(50, 2.0)))
    assert math.isnan(position_r2(np.array([1.0, 2.0])))


def test_percentile_of_puts_the_minimum_at_zero_and_the_maximum_at_one():
    v = np.arange(10.0)
    assert percentile_of(v, 0) == pytest.approx(0.05)     # min, half its own mass
    assert percentile_of(v, 9) == pytest.approx(0.95)


def test_percentile_of_handles_ties_by_splitting_them():
    v = np.array([1.0, 1.0, 1.0, 1.0])
    assert percentile_of(v, 0) == pytest.approx(0.5)


def test_standardised_difference_sign_and_scale():
    v = np.array([0.0, 0.0, 2.0, 2.0])
    lab = np.array([-1, -1, 0, 0])
    # clustered mean 2, noise mean 0, std 1 -> +2
    assert standardised_difference(v, lab) == pytest.approx(2.0)
    assert standardised_difference(v, np.array([0, 0, -1, -1])) == pytest.approx(-2.0)


def test_standardised_difference_is_a_difference_not_a_ratio():
    """log Z is signed, and a ratio on a quantity whose mean can pass through
    zero is meaningless. Shifting everything by a constant must not change a
    standardised difference; it would wreck a ratio."""
    v = np.array([-3.0, -1.0, 1.0, 3.0])
    lab = np.array([-1, -1, 0, 0])
    a = standardised_difference(v, lab)
    b = standardised_difference(v + 1000.0, lab)
    assert a == pytest.approx(b)


def test_standardised_difference_is_nan_when_a_population_is_empty():
    v = np.arange(4.0)
    assert math.isnan(standardised_difference(v, np.full(4, -1)))
    assert math.isnan(standardised_difference(v, np.zeros(4, dtype=int)))


def test_standardised_difference_is_nan_on_zero_spread():
    assert math.isnan(standardised_difference(np.full(4, 7.0),
                                              np.array([-1, -1, 0, 0])))


# --- the closed-form case --------------------------------------------------

def test_the_idealised_regime_reproduces_math_10_section_2_exactly():
    """Three claims in one construction: raw log Z is ENTIRELY position, the
    sink at position 0 is the MINIMUM of the raw distribution, and the
    correction removes the position dependence completely."""
    n = 80
    lab = np.full(n, -1)
    lab[::2] = 0
    rec = measure_layer(_idealised(n), lab, 2.0, _rng())
    assert rec["position_r2_raw"] == pytest.approx(1.0, abs=1e-3)
    assert rec["sink_percentile_raw"] == pytest.approx(0.5 / n, abs=1e-3)
    assert math.isnan(rec["position_r2_corrected"]) or \
        rec["position_r2_corrected"] < 0.05


def test_a_unit_diagonal_leaves_a_residual_trend_the_correction_cannot_remove():
    """And this is the honest reading of the sweep's `position_r2_corrected`.

    Sphere-projected activations have `<x_i, x_i> = 1`, so in the
    concentration regime `Z_i = e^beta + i e^{beta gamma}` -- AFFINE in `i`,
    not proportional to `(i+1)`. Dividing by `(i+1)` therefore leaves
    structure, and a corrected R^2 well above 0 on real data is that residual
    rather than content. The correction is still the right one: it removes the
    dominant term, and what remains is a known functional form rather than an
    unexplained trend."""
    n = 80
    lab = np.full(n, -1)
    lab[::2] = 0
    rec = measure_layer(_realisable(n), lab, 2.0, _rng())
    assert rec["position_r2_raw"] > 0.95
    assert 0.1 < rec["position_r2_corrected"] < 0.95


def test_the_sink_is_the_minimum_of_raw_z_under_a_mask():
    """`math-10.md` §2's headline, and the reason §1A.6's identification is
    labelled an unmasked-model statement: position 0 has the SMALLEST Z."""
    n = 50
    lab = np.full(n, -1)
    lab[::2] = 0
    logZ_rank = measure_layer(_realisable(n), lab, 1.0, _rng())["sink_percentile_raw"]
    assert logZ_rank < 0.05


def test_the_correction_is_applied_to_the_statistic_not_just_reported():
    """The population comparison must read the CORRECTED quantity. If it read
    the raw one, a partition with any position structure would show a huge
    difference driven by (i+1) alone."""
    n = 120
    lab = np.full(n, -1)
    lab[60:] = 0                       # clustered strictly late
    rec = measure_layer(_idealised(n), lab, 1.0, _rng())
    # In the IDEALISED regime the corrected quantity is exactly constant, so
    # there is nothing for the statistic to find however the labels sit.
    assert rec["clustered_minus_noise"] is None or rec["degenerate"] is True


def test_a_real_population_difference_is_detected():
    """Power. Give the clustered tokens genuinely larger inner products with
    everything, independent of position."""
    rng = np.random.default_rng(3)
    n = 160
    lab = np.full(n, -1)
    clustered = rng.choice(n, size=80, replace=False)
    lab[clustered] = rng.integers(0, 10, size=80)
    X = rng.normal(size=(n, 24))
    X[clustered] += 3.0                # a shared direction -> larger Z
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    rec = measure_layer(X, lab, 4.0, rng)
    assert rec["clustered_minus_noise"] > 0.5
    assert rec["clustered_minus_noise_p"] <= 1.0 / (N_PERMUTATIONS + 1) + 1e-12


def test_the_direction_is_two_sided_so_both_hypotheses_can_show():
    """H-PARK predicts clustered particles are cheap to move and H-CAT the
    opposite, so a one-sided test would be able to confirm only one of them."""
    rng = np.random.default_rng(4)
    n = 160
    lab = np.full(n, -1)
    clustered = rng.choice(n, size=80, replace=False)
    lab[clustered] = rng.integers(0, 10, size=80)
    X = rng.normal(size=(n, 24))
    X[np.setdiff1d(np.arange(n), clustered)] += 3.0      # NOISE gets the boost
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    rec = measure_layer(X, lab, 4.0, rng)
    assert rec["clustered_minus_noise"] < -0.5
    assert rec["clustered_minus_noise_p"] <= 1.0 / (N_PERMUTATIONS + 1) + 1e-12


def test_large_beta_does_not_overflow_the_row():
    """The reason `log_partition_function` is the primary form."""
    n = 40
    lab = np.full(n, -1)
    lab[::2] = 0
    rec = measure_layer(_realisable(n), lab, 400.0, _rng())
    assert np.isfinite(rec["position_r2_raw"])
    assert np.isfinite(rec["sink_percentile_raw"])


def test_mismatched_labels_are_refused():
    with pytest.raises(ValueError, match="disagree"):
        measure_layer(_realisable(20), np.full(10, -1), 1.0, _rng())


def test_a_tiny_layer_is_dropped():
    assert measure_layer(_realisable(3), np.array([-1, 0, 0]), 1.0, _rng()) is None


# --- the directory walk and the aggregate ----------------------------------

def test_missing_inputs_are_skipped_with_a_reason(tmp_path):
    import json

    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step0_wiki"
    d.mkdir(parents=True)
    assert measure_directory(d, _rng())["skipped"] == "no HDBSCAN partition"
    (d / "hdbscan_labels.json").write_text(json.dumps({"0": [0, -1, 0, -1]}))
    assert measure_directory(d, _rng())["skipped"] == "no activations.npz"


def test_every_beta_is_measured_on_every_layer(tmp_path):
    import json

    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step9_wiki"
    d.mkdir(parents=True)
    n = 30
    lab = np.full(n, -1)
    lab[::2] = 0
    np.savez(d / "activations.npz",
             activations=np.stack([_realisable(n) for _ in range(3)]))
    (d / "hdbscan_labels.json").write_text(
        json.dumps({str(k): lab.tolist() for k in range(3)}))
    rows = measure_directory(d, _rng())["rows"]
    assert len(rows) == 3 * len(BETAS)
    assert sorted({r["beta"] for r in rows}) == sorted(BETAS)
    assert sorted({r["layer"] for r in rows}) == [0, 1, 2]


def _row(beta=1.0, p=0.5, diff=0.2):
    return {"beta": beta, "layer": 0, "n_tokens": 100,
            "position_r2_raw": 0.97, "position_r2_corrected": 0.4,
            "sink_percentile_raw": 0.002, "sink_percentile_corrected": 0.99,
            "clustered_minus_noise": diff, "clustered_minus_noise_p": p,
            "degenerate": False}


def test_aggregate_of_nothing_says_so():
    assert aggregate([{"skipped": "x"}]) == {"n_units": 0}


def test_aggregate_splits_by_beta_because_beta_is_undecided():
    """`docs/AXES.md` §4: the unit convention is an open decision worth a
    factor of 8, so a single merged number across betas would hide it."""
    dirs = [{"checkpoint": 1, "rows": [_row(beta=1.0, p=0.9),
                                       _row(beta=4.0, p=0.001)]}]
    got = aggregate(dirs)
    assert set(got["by_beta"]) == {"1.0", "4.0"}
    assert got["by_beta"]["4.0"]["clustered_minus_noise"]["E"] > \
        got["by_beta"]["1.0"]["clustered_minus_noise"]["E"]


def test_aggregate_splits_by_checkpoint_too():
    dirs = [{"checkpoint": 0, "rows": [_row(p=0.9)]},
            {"checkpoint": 143000, "rows": [_row(p=0.001)]}]
    got = aggregate(dirs)
    assert list(got["by_checkpoint"]) == ["0", "143000"]


def test_aggregate_merges_with_the_mean_not_the_product():
    from core.evalues import calibrate

    got = aggregate([{"checkpoint": 1, "rows": [_row(p=0.19) for _ in range(25)]}])
    assert got["clustered_minus_noise"]["E"] == pytest.approx(calibrate(0.19), abs=1e-3)
    assert got["clustered_minus_noise"]["reject"] is False


def test_none_differences_are_dropped_not_scored_as_zero():
    rows = [_row(p=0.5), {**_row(), "clustered_minus_noise": None,
                          "clustered_minus_noise_p": None}]
    got = aggregate([{"checkpoint": 1, "rows": rows}])
    assert got["n_units"] == 2
    assert got["clustered_minus_noise"]["n"] == 1
    assert got["mean_clustered_minus_noise"] == pytest.approx(0.2)


# --- the re-read (R2): one label-source column -------------------------------

from tools.run.p10_partition_function import members_rest, reread_by_step, reread_run, unit_rng  # noqa: E402
from core.parking import log_partition_function, log_position_corrected_partition_function  # noqa: E402

O = -2   # OUTSIDE


def test_members_rest_reads_the_domain_only():
    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0, -100.0])
    lab = np.array([0, 0, -1, -1, O, O])
    r = members_rest(v, lab, _rng())
    assert r["stat"] == round(standardised_difference(v[:4], lab[:4]), 4)
    assert (r["n_members"], r["n_rest"]) == (2, 2)
    # an OUTSIDE position is not "the rest": with no -1 the statistic is undefined
    assert members_rest(v, np.array([0, 0, O, O, O, O]), _rng()) is None


def test_unit_rng_is_keyed_by_the_unit_not_by_run_order():
    a = unit_rng(0, 512, "wiki", 3, 1.0).random(3)
    assert np.array_equal(a, unit_rng(0, 512, "wiki", 3, 1.0).random(3))
    assert not np.array_equal(a, unit_rng(0, 512, "wiki", 4, 1.0).random(3))


def test_reread_takes_z_over_every_position_and_the_statistic_on_the_domain(tmp_path):
    n = 24
    X = np.random.default_rng(3).normal(size=(2, n, 6))
    X /= np.linalg.norm(X, axis=-1, keepdims=True)
    np.savez(tmp_path / "activations.npz", activations=X)
    lab = np.full(n, -1)
    lab[2:8] = 0
    lab[[0, 1]] = O          # position 0 and one more off the domain
    rows = reread_run(tmp_path, {1: lab}, 512, "wiki", 0)
    assert len(rows) == len(BETAS) and {r["layer"] for r in rows} == {1}
    for r in rows:
        corr = log_position_corrected_partition_function(log_partition_function(X[1], r["beta"], causal=True))
        dom = lab != O
        assert r["stat"] == round(standardised_difference(corr[dom], lab[dom]), 4)
    assert rows == reread_run(tmp_path, {1: lab}, 512, "wiki", 0)   # same unit, same p


def test_reread_by_step_pools_prompts_and_betas():
    runs = {"0|a": [{"beta": 1.0, "stat": 0.2, "p": 0.01}, {"beta": 2.0, "stat": 0.4, "p": 0.5}],
            "0|b": [{"beta": 1.0, "stat": 0.6, "p": 0.02}], "64|a": []}
    by = reread_by_step(runs)
    assert by[0]["n"] == 3 and by[0]["mean"] == pytest.approx(0.4) and by[0]["median_p"] == 0.02
    assert reread_by_step(runs, by_beta=True)[0]["1.0"]["n"] == 2


# --- R2m: the matched control (design-10.md "R2m") ---------------------------

from tools.run.p10_partition_function import matched_runs  # noqa: E402


def _step_file(src, step, runs):
    (src / f"step{step}.json").write_text(json.dumps({"prompts": {p: {"stage0_run": str(r)} for p, r in runs.items()}}))


def test_matched_runs_maps_every_step_to_the_same_prompts_step_0_run(tmp_path):
    src = tmp_path / "labels"
    src.mkdir()
    r0, r512, other = (tmp_path / n for n in ("s0_wiki", "s512_wiki", "s0_iliad"))
    for r, toks in ((r0, "a\nb\n"), (r512, "a\nb\n"), (other, "c\n")):
        r.mkdir()
        (r / "tokens.txt").write_text(toks)
    _step_file(src, 0, {"wiki": r0, "iliad": other})
    got = matched_runs(src, {(512, "wiki"): r512, (0, "wiki"): r0}, 0)
    assert got == {(512, "wiki"): r0, (0, "wiki"): r0}


def test_matched_runs_refuses_other_tokens_or_a_missing_prompt(tmp_path):
    src = tmp_path / "labels"
    src.mkdir()
    r0, r512 = tmp_path / "s0", tmp_path / "s512"
    for r, toks in ((r0, "a\nb\n"), (r512, "a\nX\n")):
        r.mkdir()
        (r / "tokens.txt").write_text(toks)
    _step_file(src, 0, {"wiki": r0})
    with pytest.raises(SystemExit, match="different tokens"):
        matched_runs(src, {(512, "wiki"): r512}, 0)
    with pytest.raises(SystemExit, match="no step-0 run"):
        matched_runs(src, {(512, "iliad"): r512}, 0)


def test_reread_scores_step_s_labels_on_step_0s_run_when_matched(tmp_path, monkeypatch):
    """/challenge-pr on #149, finding 4: the step-0 Δ = 0 check cannot see a swap, so pin the pairing."""
    import sys
    import types
    from argparse import Namespace
    from tools.run.p10_partition_function import reread
    runs = {}
    for s in (0, 512):
        r = tmp_path / f"s{s}"
        r.mkdir()
        (r / "tokens.txt").write_text("a\nb\n")
        runs[(s, "wiki")] = r
    src = tmp_path / "labels"
    src.mkdir()
    _step_file(src, 0, {"wiki": runs[(0, "wiki")]})
    lab = {1: np.array([0, -1])}
    fake = types.ModuleType("tools.run.p10_label_source")
    fake.SPLITS, fake.LEARNED_STEP, fake.MODELS = {}, "step143000", ("step0", "step512")
    fake.reader_input = lambda labels, column: {
        "runs": runs, "labels": {k: lab for k in runs}, "records": {0: [1, 1], 512: [1, 1]}, "meta": {}}
    monkeypatch.setitem(sys.modules, "tools.run.p10_label_source", fake)
    seen = {}

    def measure(run_dir, labels, step, prompt, seed):
        seen[step] = run_dir
        return []
    args = Namespace(labels=src, column="c3", jobs=1, seed=0)
    rec = reread(args, measure, "test", acts_step=0)
    assert seen == {0: runs[(0, "wiki")], 512: runs[(0, "wiki")]}
    assert rec["activations_step"] == 0 and dict(rec["inputs"])["512|wiki"] == str(runs[(512, "wiki")])
    reread(args, measure, "test")
    assert seen[512] == runs[(512, "wiki")]


@pytest.mark.parametrize("column", ["c3_learned", "c3x_learned", "c3x_unlearned"])
def test_reread_reads_every_learned_split_at_143000_only(tmp_path, monkeypatch, column):
    """R9 / R2: the step check took only c3's split as 143000-only, so c3x's split refused."""
    import sys
    import types
    from argparse import Namespace
    from tools.run import p10_label_source as ls
    from tools.run.p10_partition_function import reread
    r = tmp_path / "s143000"
    r.mkdir()
    (r / "tokens.txt").write_text("a\nb\n")
    fake = types.ModuleType("tools.run.p10_label_source")
    fake.SPLITS, fake.LEARNED_STEP, fake.MODELS = ls.SPLITS, ls.LEARNED_STEP, ls.MODELS
    fake.reader_input = lambda labels, column: {
        "runs": {(143000, "wiki"): r}, "labels": {(143000, "wiki"): {1: np.array([0, -1])}},
        "records": {143000: [1, 1]}, "meta": {}}
    monkeypatch.setitem(sys.modules, "tools.run.p10_label_source", fake)
    seen = []
    reread(Namespace(labels=tmp_path, column=column, jobs=1, seed=0),
           lambda run_dir, labels, step, prompt, seed: seen.append(step) or [], "test")
    assert seen == [143000]
