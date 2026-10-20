"""A5's statistic and D10's tie-breaker (`trackA_force/a5_seeds.py`, design-A5.md): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force import a3, a5_seeds as a5
from trackA_force.a1a import STEPS

SEED0_V = np.array([0, 0, 0, 0, 0, 0, 0, 0, 19, 56, 56, 61, 57, 59, 64, 76, 72, 70], float)


def test_seed0_reads_512_inside():
    r = a3.d4a(SEED0_V)
    assert r["step"] == 512 and r["plateau_median"] == 57.0 and a5.inside(r)


def test_planted_effect_one_step_later_is_outside():
    r = a3.d4a(a5.shift(SEED0_V, 1))
    assert r["step"] == 1000 and not a5.inside(r)


def test_shift_pads_with_end_values():
    v = np.arange(len(STEPS), dtype=float)
    assert a5.shift(v, 2)[:3].tolist() == [0, 0, 0] and a5.shift(v, -2)[-3:].tolist() == [17, 17, 17]


def test_zero_plateau_counts_outside():
    assert not a5.inside(a3.d4a(np.zeros(len(STEPS))))


def test_inside_iff_planted_in_window():
    i0 = STEPS.index(512)
    for s in a5.NULL_STEPS:
        r = a3.d4a(a5.shift(SEED0_V, STEPS.index(s) - i0))
        assert a5.inside(r) == (s in a5.WINDOW), s


def test_k_counts_failed_seeds_outside():
    per = {1: a5.break_reading(SEED0_V), 2: a5.break_reading(a5.shift(SEED0_V, 1)),
           3: {"reading": "failed to load", "inside": False}}
    assert a5.k_of(per) == 1


def test_seed_records_from_another_run_refuse():
    recs = {(s, "x"): {"seed": 2, "repo": a5.repo_for(2), "revision": f"step{s}", "code": "c"} for s in STEPS}
    assert a5.check_seed_records(recs, 2) == ["c"]
    with pytest.raises(a5.A5Refused):
        a5.check_seed_records(recs, 1)


def test_gate_tail_and_e():
    g = a5.p_value_p_a5(7)
    assert abs(g["p"] - 9.387066250063505e-05) < 1e-15 and round(g["e"]) == 52
    assert a5.p_value_p_a5(0)["p"] == pytest.approx(1.0)


def test_pick_ties_to_lowest_index():
    s = np.zeros((24, 16)); s[3, 4] = s[7, 1] = 1.0
    p = a5.pick(s)
    assert p["head"] == "L3H4" and p["runner_up"] == "L7H1"
    assert a5.rank_of(s, 7, 1) == 1 and a5.rank_of(s, 0, 0) == 3


def _row(rid, series, ref=None, plateau=None):
    x = np.asarray(series, float)
    r = a3.d4a(x) if rid == "break" else a3.place(x, "up", sigma=None)
    r.update(id=rid, series=x.tolist(), censored=False)
    return r


def test_tie_break_shares_at_step_before():
    i = STEPS.index(1000)
    a = np.zeros(len(STEPS)); a[i - 1] = 0.4; a[i:] = 1.0
    b = np.zeros(len(STEPS)); b[i - 1] = 0.1; b[i:] = 1.0
    t = a5.tie_break(_row("prev", a), _row("ind", b))
    assert t["tie"] and t["before"] == 512 and t["further"] == "prev"
    assert t["shares"] == pytest.approx([0.4, 0.1]) and t["lr_share"] == pytest.approx(512 / 1430)


def test_tie_break_untied_reports_dates_only():
    t = a5.tie_break(_row("break", SEED0_V), _row("ind", np.r_[np.zeros(10), np.ones(8)]))
    assert t["tie"] is False and t["steps"] == [512, 1000] and "shares" not in t


def test_break_share_is_against_plateau_median():
    br = _row("break", SEED0_V)
    assert a5.share(br, STEPS.index(256)) == pytest.approx(19 / 57)


def test_record_margins_read_v_at_the_window_edges():
    from trackA_force import a5_record
    score = {"per_seed": {"1": {"plateau_median": 57.0}}, "notes": {"1": {"V": SEED0_V.tolist()}}}
    assert a5_record.margins(score) == {"1": {"half": 28.5, "V128": 0.0, "V256": 19.0, "V512": 56.0}}


def test_record_refuses_an_incomplete_seed(tmp_path):
    from trackA_force import a5_record
    (tmp_path / "records").mkdir()
    (tmp_path / "records" / "step0_x.json").write_text("{}")
    with pytest.raises(SystemExit, match="refusing"):
        a5_record.seed_hashes(tmp_path)
