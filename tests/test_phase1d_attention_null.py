"""
tests/test_phase1d_attention_null.py — item (3)'s attention communities
(`p1d_cluster_ensemble/attention_graph.py`, `neox_block.py`,
`attention_null.py`): the graph construction, the three nulls' defining
properties, and β recovery. The block map against transformers' own layer
is `tests/test_phase1d_neox_block_smoke.py` (real deps).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble.attention_graph import (
    diagonal_shuffle, drop_sink, fit_betas, graph_stats, kernel_attention, lambda2,
    modularity, rollout, symmetrise,
)


def _causal_uniform(m):
    A = np.tril(np.ones((m, m)))
    return A / A.sum(axis=1, keepdims=True)


def _planted(m=40, blocks=4, w_in=1.0, w_out=0.02, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(blocks), m // blocks)
    S = np.where(lab[:, None] == lab[None, :], w_in, w_out) * (0.5 + rng.random((m, m)))
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)
    return S, lab


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def test_drop_sink_rows_are_stochastic_and_refuse_sink():
    A = _causal_uniform(10)[None].repeat(3, axis=0)
    D = drop_sink(A, np.arange(1, 10))
    assert D.shape == (3, 9, 9)
    np.testing.assert_allclose(D.sum(axis=-1), 1.0)
    assert np.allclose(np.triu(D[0], 1), 0.0)
    with pytest.raises(ValueError):
        drop_sink(A, np.arange(0, 10))


def test_rollout_order_and_identity():
    M1, M2 = _causal_uniform(5), np.eye(5)
    np.testing.assert_allclose(rollout([np.eye(5)]), np.eye(5))
    R = rollout([M1, M2])
    step = lambda M: 0.5 * np.eye(5) + 0.5 * M
    np.testing.assert_allclose(R, step(M2) @ step(M1))
    np.testing.assert_allclose(R.sum(axis=1), 1.0)


def test_symmetrise_is_symmetric_zero_diagonal():
    R = _causal_uniform(6)
    for how in ("mutual", "coattn"):
        S = symmetrise(R, how)
        np.testing.assert_allclose(S, S.T)
        assert np.all(np.diag(S) == 0)
    with pytest.raises(ValueError):
        symmetrise(R, "directed")


def test_communities_recover_planted_blocks():
    pytest.importorskip("igraph")
    from sklearn.metrics import adjusted_rand_score
    from p1d_cluster_ensemble.attention_graph import communities
    S, lab = _planted()
    got = communities(S, seed=0)
    assert adjusted_rand_score(lab, got) > 0.95
    assert modularity(S, got) > 0.5
    np.testing.assert_array_equal(got, communities(S, seed=0))


def test_modularity_matches_igraph():
    ig = pytest.importorskip("igraph")
    S, lab = _planted(seed=1)
    g = ig.Graph.Weighted_Adjacency(S, mode="undirected", attr="weight", loops=False)
    assert modularity(S, lab) == pytest.approx(g.modularity(lab.tolist(), weights="weight"), abs=1e-9)


def test_lambda2_small_for_blocks_large_for_complete():
    S, _ = _planted(w_out=1e-4)
    K = np.ones((40, 40)) - np.eye(40)
    assert lambda2(S) < 0.05 < lambda2(K)


def test_graph_stats_contiguity_and_local():
    S, lab = _planted()
    st = graph_stats(S, lab, np.arange(40))
    assert st["k"] == 4 and st["k4"] == 4
    assert st["contig"] == pytest.approx(36 / 39)
    assert 0 < st["local"] < 1


# ---------------------------------------------------------------------------
# Null B
# ---------------------------------------------------------------------------

def test_diagonal_shuffle_fixes_uniform_attention():
    """The step-0 failure: raw-weight shuffling made uniform attention modular."""
    U = _causal_uniform(30)
    np.testing.assert_allclose(diagonal_shuffle(U, np.random.default_rng(0)), U)


def test_diagonal_shuffle_keeps_offset_multisets_and_causality():
    rng = np.random.default_rng(3)
    M = np.tril(rng.random((25, 25)))
    M /= M.sum(axis=1, keepdims=True)
    out = diagonal_shuffle(M, np.random.default_rng(1))
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert np.allclose(np.triu(out, 1), 0.0)
    assert not np.allclose(out, M)
    # before renormalisation the relative weights at each offset are a permutation
    scale = np.arange(1, 26)[:, None]
    E, F = M * scale, np.zeros_like(M)
    rng2 = np.random.default_rng(1)
    for d in range(25):
        rows = np.arange(d, 25)
        vals = E[rows, rows - d]
        F[rows, rows - d] = vals[rng2.permutation(vals.size)]
        assert sorted(F[rows, rows - d]) == pytest.approx(sorted(vals))


# ---------------------------------------------------------------------------
# Null C and β
# ---------------------------------------------------------------------------

def test_kernel_beta_zero_is_uniform_causal():
    rng = np.random.default_rng(0)
    U = rng.standard_normal((12, 8))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    K = kernel_attention(U, [0.0, np.nan])
    np.testing.assert_allclose(K[0], _causal_uniform(12))
    np.testing.assert_allclose(K[1], _causal_uniform(12))


def test_fit_betas_recovers_kernel_beta_with_a_sink():
    """β fitted on unit rows reproduces the kernel's β: no unit convention enters."""
    rng = np.random.default_rng(0)
    n, d, beta = 60, 16, 4.0
    U = rng.standard_normal((n, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    logits = beta * (U @ U.T)
    logits[:, 0] += 3.0                               # a sink the fit must not see
    logits = np.where(np.triu(np.ones((n, n), bool), 1), -np.inf, logits)
    A = np.exp(logits - logits.max(axis=1, keepdims=True))
    A /= A.sum(axis=1, keepdims=True)
    out = fit_betas(A[None], U, np.arange(1, n), np.arange(n))
    assert out[0]["beta"] == pytest.approx(beta, rel=1e-6)
    assert out[0]["r2"] == pytest.approx(1.0, abs=1e-9)


def test_fit_betas_offsets_use_positions_not_indices():
    """Deduped rows are not consecutive positions; the offset regressor must know."""
    rng = np.random.default_rng(2)
    n, beta, slope = 40, 2.0, -0.05
    pos = np.sort(rng.choice(200, n, replace=False))
    pos[0] = 0
    U = rng.standard_normal((n, 8))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    off = pos[None, :] - pos[:, None]
    logits = beta * (U @ U.T) + slope * off
    logits = np.where(np.triu(np.ones((n, n), bool), 1), -np.inf, logits)
    A = np.exp(logits - logits.max(axis=1, keepdims=True))
    A /= A.sum(axis=1, keepdims=True)
    out = fit_betas(A[None], U, np.arange(1, n), pos)
    assert out[0]["beta"] == pytest.approx(beta, rel=1e-6)
    assert out[0]["offset_coeff"] == pytest.approx(slope, rel=1e-6)


# ---------------------------------------------------------------------------
# Driver pieces
# ---------------------------------------------------------------------------

def test_summarise_counts_the_stronger_tail():
    from p1d_cluster_ensemble.attention_null import summarise
    rec = lambda layer, p: {"step": "s", "layer": layer, "stats": {"A": {"w1_mutual": {
        "Q": {"obs": 0.5, "null_mean": 0.4, "null_sd": 0.01, "p_upper": p, "p_lower": 1.0},
        "k": {"obs": 3, "null_mean": 3, "null_sd": 0, "p_upper": 0.5, "p_lower": 0.5}}}}}
    rows = summarise([rec(3, 0.01), rec(4, 0.5), {"skipped": "x", "layer": 1}])
    assert len(rows) == 1
    r = rows[0]
    assert (r["null"], r["graph"], r["band"], r["stat"], r["n"], r["tail"]) == \
        ("A", "w1_mutual", "L1-8", "Q", 2, 1)
    assert r["median_z"] == pytest.approx(10.0)
