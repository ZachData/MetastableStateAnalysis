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
