"""OV1s's arms and labels, the parts without a model (`p10_cluster_function/design-10.md` "OV1s";
`tools/run/p10_ov1s_sign.py`): ``neg`` is ``att`` with the sign reversed; the hook arms' ``Δ`` gives
``K + S₋``, ``K − S₊`` and (the check) ``S₊``; their forms on the plane are ``S₊`` / ``−S₋``; the
labels read c3x against its partner arm and against the rest. The model tests are
`test_p10_ov1s_sign_smoke.py`.
"""
import numpy as np
import pytest

from tools.run import p10_ov1_cut as ov
from tools.run import p10_ov1s_sign as sg

pytestmark = pytest.mark.deps                  # merge_tree / the label source, as OV1's tests


def head(seed=0, d=20, k=3):
    rng = np.random.default_rng(seed)
    W_O, W_V, g = rng.normal(size=(d, k)), rng.normal(size=(k, d)), 0.5 + rng.random(d)
    lam, U, kn = ov.head_eig(W_O, W_V, g)
    return {"lam": lam, "U": U, "K": kn}, W_O, W_V * g[None, :]


def parts(hd):
    lam, U = hd["lam"], hd["U"]
    Sp = U[:, lam > 0] @ np.diag(lam[lam > 0]) @ U[:, lam > 0].T
    Sm = -U[:, lam < 0] @ np.diag(lam[lam < 0]) @ U[:, lam < 0].T
    return Sp, Sm


def test_neg_is_att_reversed():
    hd, _, _ = head()
    Ua, la = sg.pick(hd, "att")
    Un, ln = sg.pick(hd, "neg")
    assert np.array_equal(Ua, Un) and np.allclose(ln, -la) and (la > 0).all()


@pytest.mark.parametrize("arm", ["norep", "noatt", "att_hook"])
def test_hook_delta(arm):
    hd, WO, WVg = head(3)
    K = WO @ WVg
    Sp, Sm = parts(hd)
    L, R = sg.delta(hd, arm, WO, WVg)
    want = {"norep": K + Sm, "noatt": K - Sp, "att_hook": Sp}[arm]
    assert np.allclose(K + L @ R.T, want, atol=1e-12)
    d = K.shape[0]
    Pi = np.eye(d) - 1.0 / d
    M = Pi @ (K + L @ R.T) @ Pi
    form = {"norep": Sp, "noatt": -Sm, "att_hook": Sp}[arm]
    assert np.allclose((M + M.T) / 2, form, atol=1e-12)      # the rule's "form on the plane"


def rec(kinds_x, kinds_rest, arm_kinds):
    """One layer (L9), c3x ids 0.., rest ids after; ``arm_kinds[arm] = (c3x kinds, rest kinds)``."""
    nx, nr = len(kinds_x), len(kinds_rest)
    x, rest = list(range(nx)), list(range(nx, nx + nr))
    arms = {a: {"rel": 1.0, "kind": {str(g): k for g, k in zip(x + rest, kx + kr)}}
            for a, (kx, kr) in arm_kinds.items()}
    return {"layers": {"9": {"c3x": x, "groups": x + rest, "arms": arms}}}


def test_labels_read_partner_and_rest():
    m, s = "merge", "stable"
    recs = {}
    for i, p in enumerate(sg.PASSAGES):
        recs[(16000, p)] = rec([0] * 4, [0] * 4, {
            "att": ([m] * 4, [m, s, s, s]), "neg": ([s] * 4, [s] * 4),
            "norep": ([m, m, s, s], [m, m, s, s]), "noatt": ([s] * 4, [s] * 4)})
    c = {lab: sg.cell(recs, 16000, "L9-16", lab) for lab in sg.LABELS}
    assert c["S1"]["label"] == "merges" and np.allclose(c["S1"]["values"], 1.0)
    assert c["S2"]["label"] == "merges"
    assert c["S3a"]["label"] == "c3x more" and np.allclose(c["S3a"]["values"], 0.75)
    assert c["S3b"]["label"] == "mixed" and np.allclose(c["S3b"]["values"], 0.0)
    table = {(16000, "L9-16", lab): v for lab, v in c.items()}
    r = sg.reading(table, 16000, "L9-16")
    assert "the sign does it, at fixed eigenvectors and size" in r
    assert "S3a: specific to c3x's groups" in r
    assert any(x.startswith("S3b: the stream") for x in r)
    assert "neg moves the stream as far as att: OV1's 'moved further' caveat does not apply" in r


def test_ov1_window_not_sign():
    s = "stable"
    recs = {(16000, p): rec([s] * 4, [s] * 4, {a: ([s] * 4, [s] * 4) for a in sg.ARMS[1:]})
            for p in sg.PASSAGES}
    table = {(16000, "L9-16", lab): sg.cell(recs, 16000, "L9-16", lab) for lab in sg.LABELS}
    assert table[(16000, "L9-16", "S1")]["label"] == "mixed"
    assert sg.reading(table, 16000, "L9-16") == ["OV1's reading is not the sign at fixed size"]


def test_too_few_rest():
    m = "merge"
    recs = {(16000, p): rec([m] * 4, [m] * 2, {a: ([m] * 4, [m] * 2) for a in sg.ARMS[1:]})
            for p in sg.PASSAGES}
    assert sg.cell(recs, 16000, "L9-16", "S3a")["label"] == "too few"
