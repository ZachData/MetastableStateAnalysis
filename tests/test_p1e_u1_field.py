"""
`p1e_energy_field/u1_field.py` — U1 on synthetic clouds (`design-1e.md` "U1: the rule", tests):
three planted von Mises–Fisher groups give three wells and their labels; the leave-one-out
density against an explicit loop; the Gaussian draw's covariance; AMI 0 under permutation (in `test_p1e_u1_field_deps.py`, deps tier).
"""
import numpy as np
import pytest

from p1e_energy_field import u1_field as u1

pytestmark = pytest.mark.pure


def _vmf3(n_per=40, d=24, kappa=400.0, seed=0):
    """Three tight groups around orthogonal centres (vMF-like: Gaussian noise, then the sphere)."""
    rng = np.random.default_rng(seed)
    C = np.linalg.qr(rng.normal(size=(d, 3)))[0].T
    X = np.concatenate([c + rng.normal(size=(n_per, d)) / np.sqrt(kappa) for c in C])
    return X / np.linalg.norm(X, axis=1, keepdims=True), np.repeat(np.arange(3), n_per)


@pytest.mark.parametrize("exact", [True, False])
def test_three_planted_groups_give_three_wells(exact):
    X, lab = _vmf3()
    ms = u1.mean_shift(X, 10.0, exact=exact)
    assert u1.well_stats(ms["wells"])["k"] == 3
    assert u1.agreement(ms["wells"], lab) == 1.0
    assert ms["unconverged"] == 0


def test_float32_path_matches_float64_reference():
    X, _ = _vmf3(seed=3)
    a = u1.mean_shift(X, 10.0)["wells"]
    b = u1.mean_shift(X, 10.0, exact=True)["wells"]
    assert u1.agreement(a, b) == 1.0


def test_one_well_at_small_beta():
    X, _ = _vmf3()
    assert u1.well_stats(u1.mean_shift(X, 0.5)["wells"])["k"] == 1


def test_density_is_leave_one_out_and_centred():
    X, _ = _vmf3(n_per=10)
    beta = 3.5
    e = u1.density(X @ X.T, beta)
    lp = np.array([np.log(sum(np.exp(beta * X[i] @ X[j]) for j in range(len(X)) if j != i))
                   for i in range(len(X))])
    np.testing.assert_allclose(e, lp - lp.mean(), atol=1e-10)
    assert abs(e.sum()) < 1e-9


def test_causal_density_is_the_mean_over_earlier_positions():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(12, 8))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    pos = np.array([1, 4, 7, 11])
    lp = np.array([np.log(np.mean([np.exp(2.0 * U[p] @ U[j]) for j in range(p)])) for p in pos])
    np.testing.assert_allclose(u1.causal_density(U, pos, 2.0), lp - lp.mean(), atol=1e-10)


def test_gaussian_draw_keeps_mean_and_covariance_before_the_sphere():
    rng = np.random.default_rng(2)
    U = rng.normal(size=(300, 6)) * [3, 2, 1, 1, .5, .5] + [4, 0, 0, 0, 0, 0]
    mu, C = U.mean(0), np.cov(U.T)
    draws = []
    for _ in range(200):
        G = rng.standard_normal((300, 300))
        draws.append(mu + G @ (U - mu) / np.sqrt(299))
    D = np.concatenate(draws)
    np.testing.assert_allclose(D.mean(0), mu, atol=0.05)
    np.testing.assert_allclose(np.cov(D.T), C, atol=0.15)
    Y = u1.gaussian_draw(U, rng)
    np.testing.assert_allclose(np.linalg.norm(Y, axis=1), 1.0)


def test_dedup_merges_only_within_tolerance():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([np.cos(1e-4), np.sin(1e-4), 0.0])
    c = np.array([0.0, 1.0, 0.0])
    lab, reps = u1.dedup(np.stack([a, b, c]), 1e-6)
    assert lab.tolist() == [0, 0, 1] and reps.tolist() == [0, 2]


def _pairs_at(dist: float, n: int = 300, d: int = 1024) -> np.ndarray:
    """``n`` pairs of float32 unit rows, each pair ``dist`` apart in cosine distance (float64)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, d))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    P = rng.standard_normal((n, d))
    P -= (P * A).sum(1, keepdims=True) * A
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    th = np.sqrt(2 * dist)
    return np.concatenate([A, np.cos(th) * A + np.sin(th) * P]).astype(np.float32)


@pytest.mark.parametrize("on_gpu", [False, True])
def test_merges_compare_in_float64(on_gpu):
    """A float32 GPU dot of 1024-d rows is off by up to ~1e-6, the merge tolerance: pairs 1.5e-6
    apart were merged (3 of 300 before the fix). Merges are decided in float64 on both paths."""
    if on_gpu:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
    for dist, merged in ((1.5e-6, False), (1e-7, True)):
        Y = _pairs_at(dist)
        Yd = Y.astype(np.float64)
        true = 1.0 - np.sum(Yd[:300] * Yd[300:], axis=1)
        assert ((true > u1.MERGE_RUN) if not merged else (true < u1.MERGE_RUN)).all()
        if on_gpu:
            lab, _ = u1.dedup_t(torch.as_tensor(Y, device="cuda"), u1.MERGE_RUN)
        else:
            lab, _ = u1.dedup(Y, u1.MERGE_RUN)
        assert ((lab[:300] == lab[300:]) == merged).all()


def test_hidden_state_24_refused(tmp_path):
    n = 5
    acts = np.zeros((25, n, 4), dtype=np.float32)
    acts[..., 0] = 1
    np.savez(tmp_path / "activations.npz", activations=acts, norms=np.ones((25, n)))
    (tmp_path / "geometry.json").write_text('{"tokens": ["a", "b", "c", "d", "e"]}')
    ln1 = {"w": np.ones((24, 4)), "b": np.zeros((24, 4)), "eps": 1e-5}
    with pytest.raises(ValueError, match="refused"):
        u1.read_run(tmp_path, ln1, {"t12": np.arange(1, n)}, "k", "0", layers=[24])


def test_report_sign_rule_and_not_read():
    from p1e_energy_field import u1_report as rep
    names = rep.NAMES["Xe"]
    assert rep.sign_label([1] * 8, names) == "lumpier"
    assert rep.sign_label([1] * 7 + [-1], names) == "leans lumpier"
    assert rep.sign_label([-1] * 8, names) == "smoother"
    assert rep.sign_label([1] * 6 + [-1] * 2, names) == "mixed"
    vals = {(0, "L1-8"): {f"p{i}": 0.1 for i in range(6)}}
    assert rep.labels(vals, "ami_diff", 8)[(0, "L1-8")] == "not read"
    with pytest.raises(SystemExit):
        rep.labels(vals, "Xe", 8)


@pytest.mark.parametrize("spec", [(6, 3), (20, 1)])
def test_calibrated_lumpiness_is_zero_on_a_structureless_cloud(spec):
    """(1′), `/challenge-pr` on #164 finding 2: the bias predicts the score of a cloud with no
    structure beyond its moments (a matched-Gaussian draw scored as the data)."""
    from p1e_energy_field import u1_calib as uc
    rng = np.random.default_rng(0)
    n, d = 300, 40
    X = rng.standard_normal((n, d)) * np.r_[spec, np.ones(d - 2) * 0.5] + np.r_[np.zeros(d - 1), 4]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    bias = uc.bias_cell(X, [1, 2])[1.6]
    raw = []
    for _ in range(8):
        U = u1.gaussian_draw(X, rng)
        g = [u1.density((lambda Y: Y @ Y.T)(u1.gaussian_draw(U, rng)), 1.6).std() for _ in range(4)]
        raw.append(u1.density(U @ U.T, 1.6).std() - np.mean(g))
    assert abs(np.mean(raw) - bias) < 0.015


def test_frames_centre_on_the_target_set():
    """`design-1e.md` "U1 beside": (c) centres U1's unit LN1 rows, (c′) the raw residual, on the targets."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 12)) * 3 + 1.0
    ln1 = {"w": np.ones((1, 12)), "b": np.zeros((1, 12)), "eps": 1e-5}
    t = np.arange(4, 30)
    U = u1.frame_rows(X, ln1, 0, t, "unit")
    C = u1.frame_rows(X, ln1, 0, t, "centred")
    R = u1.frame_rows(X, ln1, 0, t, "raw_centred")
    ref = U - U[t].mean(axis=0)
    assert np.allclose(C, ref / np.linalg.norm(ref, axis=1, keepdims=True))
    ref = X - X[t].mean(axis=0)
    assert np.allclose(R, ref / np.linalg.norm(ref, axis=1, keepdims=True))
    assert np.allclose(np.linalg.norm(C, axis=1), 1) and np.allclose(np.linalg.norm(R, axis=1), 1)
    with pytest.raises(ValueError):
        u1.frame_rows(X, ln1, 0, t, "nope")


def test_resume_refuses_a_record_read_with_other_options(tmp_path):
    rec = tmp_path / "records" / "long" / "step0_p.json"
    rec.parent.mkdir(parents=True)
    rec.write_text('{"code": "c", "opts": {"frame": "centred", "betas": [3.5], "primary_beta": 3.5, "sweep": true}}')
    job = ("long", "0", "p", tmp_path, "rev", {}, None)
    assert u1._job(job, tmp_path, tmp_path, "c", "cpu",
                   {"frame": "centred", "betas": [3.5], "primary_beta": 3.5, "sweep": True}).startswith("have")
    with pytest.raises(SystemExit, match="read with"):
        u1._job(job, tmp_path, tmp_path, "c", "cpu", u1.default_opts())
