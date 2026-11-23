"""C4′ (e) (`trackC_writeup/c4peg_read.py`, design-C4peg.md "(e) as U1 runs it"): the pure parts
on planted inputs."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackC_writeup import c4peg_read as e


def _ln(raw, g, b, eps=1e-5):
    mu = raw.mean(1, keepdims=True)
    return g * (raw - mu) / np.sqrt(((raw - mu) ** 2).mean(1, keepdims=True) + eps) + b


def test_cells_match_the_design():
    assert len(e.STEPS) == 29 and 3500 in e.STEPS
    assert e.window_steps("A0", "collapse") == [24, 32, 48, 64]
    assert e.window_steps("F", "collapse") == [1, 2, 4, 8, 12, 16]
    assert e.window_steps("Fp", "V") == [32] and e.window_steps("A0p", "V") == [320]
    assert all(s in e.STEPS for pair in e.EVENTS for s in pair)


def test_split_is_exact_without_ln():
    rng = np.random.default_rng(0)
    h, a, f = rng.normal(size=(3, 20, 16)) + np.array([1.0] + [0] * 15)
    sp = e.split_transition(h, a, f)
    assert sp["resid"] < 1e-12 and sp["sum_resid"] < 1e-12
    m0, m1 = e.unit_mean(h), e.unit_mean(h + a + f)
    assert np.isclose(sp["delta"], m1 @ m1 - m0 @ m0)
    nh = np.sum((h + a + f) ** 2, 1) - np.sum(h ** 2, 1)
    assert np.isclose(sp["ga"] + sp["gf"], nh.mean())


def test_split_is_exact_through_the_ln():
    rng = np.random.default_rng(1)
    h, a, f = rng.normal(size=(3, 12, 16)) + 0.5
    g, b = rng.uniform(0.5, 2, 16), rng.normal(size=16) * 0.3
    s = _ln(h + a + f, g, b)
    sp = e.split_transition(h, a, f, ln={"gamma": g, "beta": b, "eps": 1e-5}, s_next=s)
    assert sp["resid"] < 1e-12 and sp["sum_resid"] < 1e-12 and "B" in sp


def test_the_reviews_toy_reads_dilution_by_attention():
    """#366 finding 2: attention writes large vectors off the mean; the fall is dilution, attention."""
    rng = np.random.default_rng(2)
    n, d = 64, 32
    h = np.tile(np.eye(d)[0], (n, 1)) + 0.1 * rng.normal(size=(n, d))
    a = 3.0 * rng.normal(size=(n, d)); a[:, 0] = 0
    f = 0.01 * rng.normal(size=(n, d))
    sp = e.split_transition(h, a, f)
    assert sp["delta"] < 0 and sp["R"] < min(sp["A"], sp["M"])
    assert e.fall_label(sp) == "dilution, attention"


def test_removal_by_the_mlp():
    rng = np.random.default_rng(3)
    n, d = 64, 32
    h = np.tile(np.eye(d)[0], (n, 1)) + 0.1 * rng.normal(size=(n, d))
    f = np.tile(-0.6 * np.eye(d)[0], (n, 1))
    a = 0.01 * rng.normal(size=(n, d))
    sp = e.split_transition(h, a, f)
    assert e.fall_label(sp) == "removal, MLP"
    assert e.fall_label({**sp, "delta": 0.1}) is None


def test_component_means_sum_to_the_state_means():
    rng = np.random.default_rng(4)
    n, d = 10, 16
    emb = rng.normal(size=(n, d)) + 1
    attn = [rng.normal(size=(n, d)) for _ in range(6)]
    mlp = [rng.normal(size=(n, d)) for _ in range(6)]
    hs = [emb]
    for l in range(5):
        hs.append(hs[-1] + attn[l] + mlp[l])
    g, b = rng.uniform(0.5, 2, d), rng.normal(size=d) * 0.3
    hs.append(_ln(hs[-1] + attn[5] + mlp[5], g, b))
    cm = e.component_means(emb, attn, mlp, hs, {"gamma": g, "beta": b, "eps": 1e-5})
    for l in range(7):
        assert np.allclose(sum(cm[l].values()), e.unit_mean(hs[l]), atol=1e-12)
        assert np.isclose(sum(e.shares(cm[l]).values()), 1.0)
    assert set(cm[6]) == {"embed", "ln_bias"} | {f"a{k}" for k in range(6)} | {f"f{k}" for k in range(6)}
    assert set(cm[3]) == {"embed", "a0", "a1", "a2", "f0", "f1", "f2"}


def test_ls_direction_recovers_a_planted_one():
    rng = np.random.default_rng(5)
    M = rng.normal(size=(2000, 8))
    u = rng.normal(size=8)
    ell = M @ u + 3.0
    v, r2 = e.ls_direction(M, ell)
    assert np.allclose(v, u) and np.isclose(r2, 1.0)
    assert np.isclose(e.spearman(M @ v, ell), 1.0)


def test_spearman_handles_ties():
    ell = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    assert np.isclose(e.spearman(ell, ell), 1.0)
    assert np.isclose(e.spearman(-ell, ell), -1.0)


def test_nkl_bounds():
    rng = np.random.default_rng(6)
    z = rng.normal(size=500)
    p = np.exp(2 * z); p /= p.sum()
    nk, b = e.nkl(z, p)
    assert nk < 1e-6 and np.isclose(b, 2.0, atol=1e-3)
    nk0, _ = e.nkl(rng.normal(size=500), p)
    assert 0.5 < nk0 <= 1.0 + 1e-9


def test_labels_and_windows():
    assert e.label_e1(0.3, 0.1, True) == "aligned" and e.label_e1(-0.3, 0.1, True) == "anti-aligned"
    assert e.label_e1(0.05, 0.1, True) == "not aligned" and e.label_e1(0.9, 0.1, False) == "unreadable"
    assert e.gate_e1(0.5, 0.1, 0.2) and not e.gate_e1(0.15, 0.1, 0.2)
    assert e.label_e3(0.9, 0.8) == "shared" and e.label_e3(0.75, 0.8) == "between"
    assert e.label_e3(0.6, 0.8) == "schedule-specific"
    assert e.window_reads(["aligned", "aligned", "not aligned"], "aligned") is True
    assert e.window_reads(["aligned", "not aligned", "not aligned"], "aligned") is False
    assert e.window_reads(["aligned", "unreadable"], "aligned") is None
