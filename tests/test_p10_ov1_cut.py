"""OV1's cut and readout, the parts without a model (`p10_cluster_function/design-10.md` "OV1";
`tools/run/p10_ov1_cut.py`): the low-rank eigenpairs match the dense ``sym(Π K Π)``; the write-back
gives the target; the controls have ``att``'s (``rep``'s) count and norm; a stored pair joined in
one arm group links as merged, an unchanged partition as stable. The model tests are
`test_p10_ov1_cut_smoke.py`.
"""
import numpy as np
import pytest

from tools.run import p10_ov1_cut as ov

pytestmark = pytest.mark.deps                  # merge_tree / the label source, as E1's tests


def dense_S(W_O, W_V, g):
    d = W_O.shape[0]
    Pi = np.eye(d) - 1.0 / d
    K = Pi @ W_O @ W_V @ np.diag(g) @ Pi
    return (K + K.T) / 2


def test_head_eig_matches_dense():
    rng = np.random.default_rng(1)
    W_O, W_V, g = rng.normal(size=(20, 3)), rng.normal(size=(3, 20)), 0.5 + rng.random(20)
    lam, U, kn = ov.head_eig(W_O, W_V, g)
    S = dense_S(W_O, W_V, g)
    assert np.allclose(U @ np.diag(lam) @ U.T, S, atol=1e-12)
    assert np.allclose(U.T @ U, np.eye(lam.size), atol=1e-12)
    assert np.allclose(U.sum(axis=0), 0, atol=1e-12)            # in the plane
    assert np.isclose(kn, np.linalg.norm(W_O @ W_V @ np.diag(g)))
    assert (lam > 0).sum() <= 3 and (lam < 0).sum() <= 3


def test_write_back_is_the_target():
    rng = np.random.default_rng(2)
    W_O, W_V = rng.normal(size=(20, 3)), rng.normal(size=(3, 20))
    g, b = 0.5 + rng.random(20), rng.normal(size=20)
    lam, U, _ = ov.head_eig(W_O, W_V, g)
    for arm in ("att", "rep", "ctl+1", "ctl-2"):
        idx, sel = ov.choose(lam, arm, f"t|{arm}")
        WV, bV, WO = ov.write_back(U[:, idx], sel, g, b, 3)
        assert np.allclose(WO @ WV @ np.diag(g), U[:, idx] @ np.diag(sel) @ U[:, idx].T, atol=1e-12)
        assert np.allclose(WO @ (WV @ b + bV), 0, atol=1e-12)
        assert ov.readback_err(WV, WO, g, U[:, idx], sel) < 1e-6   # Gram route: ~sqrt(eps) floor


def test_controls_match_count_and_norm():
    rng = np.random.default_rng(3)
    lam = np.concatenate([rng.random(5) + 0.1, -(rng.random(7) + 0.1)])
    for arm, side in (("ctl+0", lam[lam > 0]), ("ctl-4", lam[lam < 0])):
        idx, sel = ov.choose(lam, arm, f"x|{arm}")
        assert idx.size == side.size == np.unique(idx).size
        assert np.isclose(np.linalg.norm(sel), np.linalg.norm(side))
    a1, _ = ov.choose(lam, "ctl+0", "same")
    a2, _ = ov.choose(lam, "ctl+0", "same")
    assert np.array_equal(a1, a2)


def test_planted_links():
    stored = np.array([0, 0, 0, 1, 1, 1, -1, 2, 2])
    joined = np.array([0, 0, 0, 0, 0, 0, -1, 1, 1])
    k = ov.kinds(stored, joined, [0, 1, 2])
    assert k[0][0] == k[1][0] == "merge" and k[2][0] == "stable" and k[2][1] == 1.0
    k = ov.kinds(stored, stored.copy(), [0, 1, 2])
    assert all(v == ("stable", 1.0) for v in k.values())
    gone = np.array([-1, -1, -1, 1, 1, 1, -1, 2, 2])
    assert ov.kinds(stored, gone, [0])[0][0] == "death"


def test_sign_label():
    assert ov.sign_label([0.1] * 7, "merges", "separates") == "merges"
    assert ov.sign_label([0.1] * 6 + [-0.1], "merges", "separates") == "leans merges"
    assert ov.sign_label([0.1] * 5, "merges", "separates") == "too few"
    assert ov.sign_label([0.1] * 4 + [-0.1] * 3, "merges", "separates") == "mixed"
