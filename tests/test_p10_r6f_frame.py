"""
`tools/run/p10_r6f_frame.py` — R6f's split of R6w's within into members against frame, as
`design-10.md` "R6f" fixed it.
"""
import pytest

from tools.run import p10_r6f_frame as f
from tools.run import p10_r6w_drift as w

# Tier: the module's top level imports numpy and R6w (numpy only); readers load lazily inside main.
pytestmark = pytest.mark.pure


def test_members_and_frame_sum_to_the_own_term():
    own = {"total": 0.3, "within": 0.25, **{k: 0.01 for k in w.TERMS}}
    fixed = {"total": 0.1, "within": 0.05, **{k: -0.02 for k in w.TERMS}}
    for k, v in f.split(own, fixed).items():
        assert v["members"] + v["frame"] == pytest.approx(v["own"], abs=1e-15)


def test_a_missing_reference_stays_missing():
    s = f.split({"total": 0.2, "within": None}, {"total": 0.1, "within": None})
    assert s["within"] == {"own": None, "members": None, "frame": None}


def test_identical_members_under_a_moving_frame_is_all_frame():
    # the same group at s and t; its own-frame value moves with layer 0, its fixed-frame value cannot
    own = w.decompose([(1, 0.10, "stable")], [(1, 0.30, "stable")])
    fixed = w.decompose([(1, 0.20, "stable")], [(1, 0.20, "stable")])
    share, lab = f.share_label(own["within"], fixed["within"])
    assert share == pytest.approx(0.0) and lab == "frame"


def test_members_moving_in_a_still_frame_is_members():
    own = w.decompose([(1, 0.10, "stable")], [(1, 0.30, "stable")])
    share, lab = f.share_label(own["within"], own["within"])
    assert share == pytest.approx(1.0) and lab == "members"


@pytest.mark.parametrize("within_fixed, want", [(0.07, "members"), (0.05, "both"), (0.04, "both"),
                                                (0.02, "frame"), (-0.05, "frame"), (0.2, "members")])
def test_share_bars(within_fixed, want):
    assert f.share_label(0.09, within_fixed)[1] == want


def test_below_the_floor_is_no_drift_and_unshared():
    assert f.share_label(0.04, 0.04) == (None, "no drift")


def test_frames_that_disagree_are_named():
    assert f.row_label(["members", "members"]) == "members"
    assert f.row_label(["members", "frame"]) == "frame-dependent"


def test_a_frame_with_other_tokens_is_refused(monkeypatch):
    # added after /challenge-pr on #153: the frame's run must hold the same tokens as the step's
    import numpy as np
    from tools.run import p10_ext_sem_threshold as ext
    toks = {"step": np.array(["a", "b", "c"]), "frame": np.array(["a", "b", "d"])}
    monkeypatch.setattr(ext, "read_tokens", lambda d: toks[d.name])
    monkeypatch.setattr(ext, "layer0_gram", lambda d: np.eye(3))
    monkeypatch.setattr(w, "focal_rows", lambda *a, **k: {})
    with pytest.raises(f.FrameError, match="differ from frame 512"):
        f._job((1000, "p", "/x/step", {"c3": {}}, {}, set(), {512: "/x/frame"}))
    toks["frame"] = toks["step"]
    (_, out) = f._job((1000, "p", "/x/step", {"c3": {}}, {}, set(), {512: "/x/frame"}))
    assert set(out) == {"own", 512}


# ---------------------------------------------------------------- R9 / R6f (design-10.md "R9 / R6f")

def _entry(row="frame", f512="frame", f143="frame", prompts=None):
    e = {"row_label": row, "label": {"512": f512, "143000": f143},
         "prompt_values": {p: {"label_mean": v} for p, v in (prompts or {}).items()}}
    return {s: {st: e for st in f.EMB} for s, _ in f.RECORD_SETS}


def test_cells_hold_the_row_each_frame_and_each_prompt():
    c = f.cells(_entry(prompts={"a": "members"}))
    assert len(c) == 2 * 3 * 4 and c["fixed_set|emb_pct_own|prompt a"] == "members"
    assert sum(f.is_headline(k) for k in c) == 6


def test_compare_counts_a_cell_held_on_one_side_only_and_skips_one_held_on_neither():
    ref = {"x|row": "frame", "y|frame 512": None, "z|prompt p": None, "w|prompt q": "both"}
    col = {"x|row": "members", "y|frame 512": None, "z|prompt p": "frame", "w|prompt q": "both"}
    cmp = f.compare_cells(ref, col)
    assert cmp["changed"] == ["x|row", "z|prompt p"] and cmp["n_compared"] == 3
    assert cmp["headline_changed"] == ["x|row"] and cmp["not_compared"] == ["y|frame 512"]


def test_a_draw_drops_c3xs_count_per_record_and_only_c3_groups():
    import numpy as np
    dom = {(512, "p", 1): {"c3": np.array([0, 0, 1, 1, 2, 2, -1]), "c3x": np.array([0, 0, -1, -1, -1, -1, -1])},
           (512, "p", 2): {"c3": np.array([3, 3, -1]), "c3x": np.array([3, 3, -1])}}
    labs, dropped = f.drop_labels(dom, np.random.default_rng(0))
    assert len(dropped[(512, "p", 1)]) == 2 and dropped[(512, "p", 2)] == set()
    kept = set(labs[(512, "p", 1)][labs[(512, "p", 1)] >= 0])
    assert len(kept) == 1 and kept <= {0, 1, 2}
    assert f.dropped_sizes(dom, dropped) == [2, 2]
    again, _ = f.drop_labels(dom, np.random.default_rng(0))       # a seed fixes the draw
    assert all((again[k] == labs[k]).all() for k in labs)
    none, nd = f.drop_labels(dom)                                 # no rng: c3 itself
    assert all((none[k] == dom[k]["c3"]).all() for k in dom) and not any(nd.values())


def test_reference_reading_is_within_at_the_95th_percentile_and_beyond_above():
    counts = list(range(100))                                     # 'higher' quantile: 95
    assert f.reference_reading(95, counts)["reading"] == "within random drops"
    r = f.reference_reading(96, counts)
    assert r["reading"] == "beyond random drops" and r["rank_p"] == pytest.approx(5 / 101)


def test_reproduce_names_each_differing_key_and_a_missing_column():
    other = {"meta": {"columns": ["c3", "c0"]}, "columns": {"c3": {"n_records": 51, "steps": [1]}, "c0": {"n_records": 2}}}
    data = {"columns": {"c3": {"n_records": 50, "steps": [1]}}}
    assert f.reproduce(data, other) == ["c3/n_records", "c0: missing"]
