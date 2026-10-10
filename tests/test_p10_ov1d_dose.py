"""OV1d's arms, distance and matching, the parts without a model (`p10_cluster_function/design-10.md`
"OV1d"; `tools/run/p10_ov1d_dose.py`): ``w`` writes ``t·S₊`` (no pairs at t = 0); the hook's ``Δ`` is
``t·S₊ − K``; ``dc`` is 0 for a common shift; the matching reads the first bracketing segment and
leaves the rest unmatched; the labels and readings on planted records. The model tests are
`test_p10_ov1d_dose_smoke.py`.
"""
import numpy as np
import pytest

from tools.run import p10_ov1_cut as ov
from tools.run import p10_ov1d_dose as dd
from tests.test_p10_ov1s_sign import head, parts

pytestmark = pytest.mark.deps                  # merge_tree / the label source, as OV1's tests


def test_names():
    assert dd.name("w", 1.0) == "w+1" and dd.name("w", -0.5) == "w-0.5" and dd.name("z", 0.0) == "z+0"
    assert set(dd.Z_T) <= set(dd.W_T) and 0.0 in dd.W_T and 0.0 in dd.Z_T
    assert dd.OV1S == {"w+1": "att", "w-1": "neg"} and all(a in dd.ARMS for a in dd.OV1S)


@pytest.mark.parametrize("t", [1.0, 0.25, -0.5, -3.0])
def test_pick_is_t_s_plus(t):
    hd, _, _ = head(1)
    Sp, _ = parts(hd)
    U, lam = dd.pick(hd, t)
    assert np.allclose(U @ np.diag(lam) @ U.T, t * Sp, atol=1e-12)
    assert (np.sign(lam) == np.sign(t)).all()


def test_pick_zero_writes_no_pairs():
    hd, _, _ = head(2)
    U, lam = dd.pick(hd, 0.0)
    assert lam.size == 0 and U.shape[1] == 0


@pytest.mark.parametrize("t", [1.0, 0.0, -2.0])
def test_delta_is_t_s_plus_less_k(t):
    hd, WO, WVg = head(3)
    K = WO @ WVg
    Sp, _ = parts(hd)
    L, R = dd.delta(hd, t, WO, WVg)
    assert np.allclose(K + L @ R.T, t * Sp, atol=1e-12)


def test_dc_ignores_a_common_shift_and_scales():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(30, 8))
    assert dd.dc(B + rng.normal(size=8)[None, :], B) < 1e-12
    assert np.isclose(dd.dc(2 * B, B), 1.0)


def test_matching_first_segment_and_unmatched():
    # positive branch d: 1 (t=0) → 2 → 4; negative: 1 → 1.5 → 3 → 6 (beyond the positive's 4: unmatched)
    pts = [(0.0, 1.0, 0.1), (0.5, 2.0, 0.5), (1.0, 4.0, 0.9),
           (-1.0, 1.5, 0.0), (-2.0, 3.0, 0.2), (-3.0, 6.0, 0.4)]
    got = {t: (mp, mn) for t, mp, mn in dd.matched(pts)}
    assert set(got) == {0.5, 1.0, -1.0, -2.0}                      # -3 at d 6 is past the positive branch
    assert np.allclose(got[-1.0], (0.1 + 0.4 * 0.5, 0.0))          # d 1.5 on [0 → 0.5]
    assert np.allclose(got[-2.0], (0.5 + 0.4 * 0.5, 0.2))          # d 3 on [0.5 → 1]
    assert np.allclose(got[0.5], (0.5, 0.0 + 0.2 * (0.5 / 1.5)))   # d 2 on [−1 → −2]
    assert np.allclose(got[1.0], (0.9, 0.2 + 0.2 * (1 / 3)))       # d 4 on [−2 → −3]


def test_matching_takes_the_first_crossing():
    # a non-monotone positive branch: d 1 → 3 → 2; d 2.5 is bracketed by both segments, the first wins
    pts = [(0.0, 1.0, 0.0), (0.5, 3.0, 0.4), (1.0, 2.0, 1.0), (-1.0, 2.5, 0.0)]
    got = {t: (mp, mn) for t, mp, mn in dd.matched(pts)}
    assert np.allclose(got[-1.0], (0.3, 0.0))


def test_matching_needs_t_zero():
    with pytest.raises(ov.OV1Error):
        dd.matched([(1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)])


def rec(kind_of, dist_of, nx=4):
    """One layer per band (L9, L17); c3x ids 0..nx-1; ``kind_of(arm)``, ``dist_of(arm)`` = dc."""
    x = list(range(nx))
    arms = {a: {"rel": dist_of(a), "dc": dist_of(a), "dz": 0.0, "kind": {str(g): kind_of(a) for g in x}}
            for a in dd.ARMS}
    return {"layers": {L: {"c3x": x, "groups": x, "arms": arms} for L in ("9", "17")},
            "refused_layers": {}}


def planted(merge_pos: bool, z_too: bool):
    """Distance |t| + 1 on both signs; t > 0 merges (and z's only if ``z_too``), t ≤ 0 stays."""
    def kind(a):
        if a == "base" or dd.T[a] <= 0:
            return "stable"
        if a.startswith("z") and not z_too:
            return "stable"
        return "merge" if merge_pos else "stable"
    return {(16000, p): rec(kind, lambda a: 0.0 if a == "base" else abs(dd.T[a]) + 1.0) for p in dd.PASSAGES}


def table_of(recs):
    return {(16000, b, lab): dd.cell(recs, 16000, b, lab) for b in dd.BANDS for lab in dd.LABELS}


def test_labels_and_readings_merge_through_the_sink():
    t = table_of(planted(True, False))
    assert t[(16000, "L9-16", "M")]["label"] == "merges"
    assert t[(16000, "L9-16", "M0")]["label"] == "mixed"           # every D is 0
    r = dd.reading(t, 16000, "L9-16")
    assert r[0] == "at matched centred distance S₊ merges more than −S₊: S1 is not the distance moved"
    assert r[1].startswith("not shown without position 0's channel")
    assert dd.reading(t, 16000, "L1-8") == []                      # not one of S1's windows


def test_labels_and_readings_not_the_sink():
    t = table_of(planted(True, True))
    r = dd.reading(t, 16000, "L17-24")
    assert t[(16000, "L17-24", "M0")]["label"] == "merges"
    assert "not one shift through the sink" in r[1]


def test_no_matched_distance():
    # every t ≠ 0 point far from the other branch: positive at d 10+, negative at d 1–2 below t=0's 5
    recs = {(16000, p): rec(lambda a: "stable",
                            lambda a: 0.0 if a == "base" else (5.0 if dd.T[a] == 0 else
                                                                (10 + dd.T[a] if dd.T[a] > 0 else 1.0)))
            for p in dd.PASSAGES}
    t = table_of(recs)
    assert t[(16000, "L9-16", "M")]["label"] == "too few"
    assert dd.reading(t, 16000, "L9-16")[0] == "no matched distance (Blocked 33's fallback, (c))"


def test_dissolves_beside():
    recs = {(16000, p): rec(lambda a: "death" if a.startswith("w-") else "stable",
                            lambda a: 0.0 if a == "base" else abs(dd.T[a]) + 1.0) for p in dd.PASSAGES}
    t = table_of(recs)
    assert set(t[(16000, "L9-16", "M")]["dissolves"]) == {"w-3", "w-2", "w-1", "w-0.5"}
    assert any(s.startswith("M: dissolves") for s in dd.reading(t, 16000, "L9-16"))


def test_distance_only_null_blocks_a_merge():
    # (/challenge-pr on #184, finding 1) m = (dc)² on asymmetric distances: positive arms far, negative
    # near, so linear interpolation on the positive branch overstates m; the label merges, the null too
    def dist(a):
        if a == "base":
            return 0.0
        t = dd.T[a]
        return 1.0 + (2.0 * t if t > 0 else 0.1 * abs(t))
    recs = {}
    for p in dd.PASSAGES:
        r = rec(lambda a: "stable", dist, nx=900)             # fine shares: m tracks (dc)² closely
        for c in r["layers"].values():
            for a, arm in c["arms"].items():
                share = min(1.0, dist(a) ** 2 / 9.0)
                n_m = int(round(share * len(c["c3x"])))
                arm["kind"] = {str(g): ("merge" if i < n_m else "stable") for i, g in enumerate(c["c3x"])}
        recs[(16000, p)] = r
    t = table_of(recs)
    c = t[(16000, "L9-16", "M")]
    assert c["label"] == "merges" and "d²" in dd.null_merges(c)
    r = dd.reading(t, 16000, "L9-16")
    assert r[0].startswith("M merges, but so does a merged share of distance alone")
    assert not any("not one shift through the sink" in s for s in r)


def test_m0_not_read_where_m_is_not():
    # (/challenge-pr on #184, finding 2) z merges, w does not: M0's sentence is beside, not a reading
    def kind(a):
        return "merge" if a.startswith("z") and dd.T[a] > 0 else "stable"
    recs = {(16000, p): rec(kind, lambda a: 0.0 if a == "base" else abs(dd.T[a]) + 1.0) for p in dd.PASSAGES}
    t = table_of(recs)
    assert t[(16000, "L9-16", "M")]["label"] == "mixed" and t[(16000, "L9-16", "M0")]["label"] == "merges"
    r = dd.reading(t, 16000, "L9-16")
    assert r[0] == "S1 not shown beyond the distance moved"
    assert not any("still merges" in s for s in r)
    assert any(s.startswith("M0 merges where M is not read") for s in r)
