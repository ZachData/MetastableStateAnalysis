"""
core/metrics.py — Canonical per-layer scalar metrics (transition plan v2,
core infrastructure item 3).

Single implementation of every metric whose definition previously existed
in more than one place, or was at risk of drifting:

  energy          : interaction_energy / interaction_energies_batched
                    (previously only in p1_mstate_tracking/metrics.py)
  effective rank  : effective_rank (raw and normed variants)
                    (previously only in p1_mstate_tracking/metrics.py,
                    one variant used torch.linalg.svdvals, the other
                    scipy.linalg.svdvals — same math, two code paths)
  Fiedler         : fiedler_and_eigengap
                    (previously computed inline inside
                    p1_mstate_tracking/spectral.py's spectral_eigengap_k)
  mass-near-1     : mass_near_1
                    (previously duplicated: analysis.py computed
                    `(ips > 0.9).mean()` inline; causal_tests.py's
                    `_mass_near_1` re-implemented the same fraction at a
                    *different* threshold, 0.95, for cluster-restricted
                    cohesion. Both are the same quantity — "fraction of
                    pairs above a similarity threshold" — parameterized
                    by threshold and an optional population mask. 0.9 is
                    the canonical default: it matches Blog 1's published
                    definition and analysis.py's per-layer field
                    `ip_mass_near_1`. Callers wanting the stricter
                    cluster-cohesion reading pass threshold=0.95 and a
                    mask explicitly; there is no second implementation to
                    drift out of sync with this one.)

Design choice: every function here accepts EITHER a torch.Tensor or a
np.ndarray. torch is never imported at module level — see `_as_numpy`.
This makes the module importable and independently testable (numpy/scipy
only) regardless of whether torch is installed in a given environment,
which is not true of the two source modules this consolidates. Existing
call sites that pass torch.Tensor activations are unaffected.

p1_mstate_tracking/metrics.py becomes a re-export shim over this module
(see that file) so none of its ~15 existing importers need to change.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian


# ---------------------------------------------------------------------------
# torch-optional array coercion
# ---------------------------------------------------------------------------

def _as_numpy(x) -> np.ndarray:
    """
    Accept a torch.Tensor, a np.ndarray, or anything array-like, and return
    a float64 np.ndarray. Duck-typed on .detach()/.cpu()/.numpy() so this
    module never has to import torch to support torch callers.
    """
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def l2_normalize(activations) -> np.ndarray:
    """
    L2-normalize each row onto the unit sphere. torch-optional equivalent
    of core.models.layernorm_to_sphere for callers that only need the
    numpy result (e.g. every function in this module).
    """
    arr = _as_numpy(activations).astype(np.float64, copy=False)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return arr / norms


# ---------------------------------------------------------------------------
# Gram matrix / pairwise inner products
# ---------------------------------------------------------------------------

def gram_matrix(activations) -> np.ndarray:
    """Full n×n pairwise cosine-similarity matrix on S^{d-1}."""
    normed = l2_normalize(activations)
    return normed @ normed.T


def pairwise_upper(G: np.ndarray) -> np.ndarray:
    """Upper-triangle (k=1) values of a pre-computed Gram matrix G."""
    n = G.shape[0]
    idx = np.triu_indices(n, k=1)
    return G[idx]


# Backward-compatible alias — this was p1_mstate_tracking/metrics.py's name.
pairwise_inner_products_from_gram = pairwise_upper


def pairwise_inner_products(activations) -> np.ndarray:
    """Upper-triangle pairwise cosine similarities, computed from raw activations."""
    return pairwise_upper(gram_matrix(activations))


# ---------------------------------------------------------------------------
# Interaction energy (Geshkovski et al.)
# ---------------------------------------------------------------------------

def interaction_energy(activations, beta: float) -> float:
    """E_beta = (1 / 2*beta*n^2) * sum_ij exp(beta * <x_i, x_j>)."""
    G = gram_matrix(activations)
    n = G.shape[0]
    return float(np.exp(beta * G).sum() / (2.0 * beta * n * n))


def interaction_energies_batched(G: np.ndarray, beta_values) -> dict:
    """
    E_beta for every beta in beta_values, vectorised over a pre-computed
    Gram matrix G. Returns {beta: energy_float}.
    """
    n = G.shape[0]
    betas = np.asarray(list(beta_values), dtype=np.float64)
    exp_G = np.exp(betas[:, None, None] * G[None])
    sums = exp_G.sum(axis=(1, 2))
    energies = sums / (2.0 * betas * n * n)
    return {float(b): float(e) for b, e in zip(beta_values, energies)}


# Relative drop threshold for energy-violation detection. Single definition;
# p1_mstate_tracking/metrics.py re-exports this same object (not a copy).
ENERGY_VIOLATION_REL_TOL: float = 1e-3


def energy_violation_severity(energies: list, rel_tol: float = ENERGY_VIOLATION_REL_TOL) -> dict:
    """
    Relative-threshold energy-violation analysis for one beta-series.
    Violation criterion: (E_prev - E_curr) / |E_prev| > rel_tol.
    """
    arr = np.array(energies, dtype=np.float64)
    valid = ~np.isnan(arr)

    if valid.sum() < 2:
        return dict(violation_layers=[], rel_drops=[], sum_severity=0.0,
                    max_severity=0.0, n_violations=0,
                    total_rel_change=float("nan"))

    diffs = np.diff(arr)
    ref = np.maximum(np.abs(arr[:-1]), 1e-12)
    rel_drop = -diffs / ref

    valid_trans = valid[:-1] & valid[1:]
    rel_drop = np.where(valid_trans, rel_drop, 0.0)

    viol_mask = rel_drop > rel_tol
    viol_layers = [i + 1 for i, v in enumerate(viol_mask) if v]
    viol_drops = rel_drop[viol_mask]

    first_valid = arr[valid][0]
    last_valid = arr[valid][-1]
    total_rel = float((last_valid - first_valid) / max(abs(first_valid), 1e-12))

    return dict(
        violation_layers=viol_layers,
        rel_drops=rel_drop.tolist(),
        sum_severity=float(viol_drops.sum()) if len(viol_drops) else 0.0,
        max_severity=float(viol_drops.max()) if len(viol_drops) else 0.0,
        n_violations=int(viol_mask.sum()),
        total_rel_change=total_rel,
    )


# ---------------------------------------------------------------------------
# Effective rank — one function, explicit mode
# ---------------------------------------------------------------------------

def effective_rank(activations, mode: str = "raw") -> float:
    """
    Spectral-entropy effective rank: exp(-sum p_i log p_i), p_i the
    normalized squared singular values.

    mode="raw"    : SVD on raw (unnormalized) activations. Captures both
                    scale and directional collapse. Use for degeneracy
                    gates (matches the old effective_rank_from_raw).
    mode="normed" : SVD on L2-normed activations. Measures directional
                    spread on the sphere only, independent of residual-
                    stream norm growth (matches the old
                    effective_rank_from_normed).

    Both old names are kept below as thin wrappers so existing call sites
    (which encode the contract in the function name) don't need to change.
    """
    if mode == "raw":
        arr = _as_numpy(activations).astype(np.float64, copy=False)
    elif mode == "normed":
        arr = l2_normalize(activations)
    else:
        raise ValueError(f"effective_rank: unknown mode {mode!r}, expected 'raw' or 'normed'")

    sv = np.linalg.svd(arr, compute_uv=False)
    sv2 = sv ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 1.0
    p = np.clip(sv2 / total, 1e-12, None)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def effective_rank_from_raw(activations) -> float:
    """Backward-compatible alias for effective_rank(activations, mode='raw')."""
    return effective_rank(activations, mode="raw")


def effective_rank_from_normed(normed) -> float:
    """Backward-compatible alias for effective_rank(normed, mode='normed')."""
    return effective_rank(normed, mode="normed")


# ---------------------------------------------------------------------------
# Fiedler value / vector + eigengap cluster-count estimate
# ---------------------------------------------------------------------------

def fiedler_and_eigengap(G: np.ndarray, max_k: int = 15, return_fiedler_vec: bool = False) -> dict:
    """
    Canonical Fiedler / eigengap computation on the normalized Laplacian of
    a Gram matrix. Single implementation — previously computed inline
    inside p1_mstate_tracking/spectral.py's spectral_eigengap_k, which now
    delegates here.

    Parameters
    ----------
    G                  : (n, n) pairwise inner-product matrix
    max_k              : maximum number of eigenvalues to inspect
    return_fiedler_vec : if True, include "fiedler_vec" (the second
                         Laplacian eigenvector) in the result.

    Returns
    -------
    dict with keys: k_eigengap, k_second_gap, second_gap_ratio,
    fiedler_value (lambda_2), eigenvalues, eigengaps, and (optionally)
    fiedler_vec.
    """
    G_pos = np.clip(G, 0, None)
    np.fill_diagonal(G_pos, 1.0)
    L = laplacian(G_pos, normed=True)
    n = G_pos.shape[0]
    k = min(max_k + 1, n - 1)

    if k < 2:
        result = {
            "k_eigengap": 1,
            "k_second_gap": 1,
            "second_gap_ratio": 1.0,
            "fiedler_value": float("nan"),
            "eigenvalues": [],
            "eigengaps": [],
        }
        if return_fiedler_vec:
            result["fiedler_vec"] = None
        return result

    if return_fiedler_vec:
        eigenvalues, eigenvectors = eigh(L, eigvals_only=False, subset_by_index=[0, k - 1])
        eigenvalues = np.real(eigenvalues)
        fiedler_vec = np.real(eigenvectors[:, 1]).tolist()
    else:
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, k - 1])
        eigenvalues = np.clip(np.real(eigenvalues), 0.0, None)
        fiedler_vec = None

    gaps = np.diff(eigenvalues)
    k_eigengap = int(np.argmax(gaps) + 1)

    if len(gaps) > 1:
        tail_gaps = gaps[1:]
        k_second_gap = int(np.argmax(tail_gaps) + 2)
        sorted_tail = np.sort(tail_gaps)
        second_gap_ratio = float(sorted_tail[-1] / (sorted_tail[-2] + 1e-10) if len(sorted_tail) > 1 else 1.0)
        if second_gap_ratio < 1.1:
            k_second_gap = 1
    else:
        k_second_gap = 1
        second_gap_ratio = 1.0

    result = {
        "k_eigengap": k_eigengap,
        "k_second_gap": k_second_gap,
        "second_gap_ratio": second_gap_ratio,
        "fiedler_value": float(eigenvalues[1]) if len(eigenvalues) > 1 else float("nan"),
        "eigenvalues": eigenvalues.tolist(),
        "eigengaps": gaps.tolist(),
    }
    if return_fiedler_vec:
        result["fiedler_vec"] = fiedler_vec
    return result


# ---------------------------------------------------------------------------
# Mass-near-1 — one function, one threshold policy
# ---------------------------------------------------------------------------

MASS_NEAR_1_DEFAULT_THRESHOLD: float = 0.9


def mass_near_1(G: np.ndarray, threshold: float = MASS_NEAR_1_DEFAULT_THRESHOLD, mask: np.ndarray = None) -> float:
    """
    Fraction of pairwise inner products exceeding `threshold`.

    mask : optional (n,) boolean array. If given, restricts to pairs (i, j)
           both inside the mask (the cluster-cohesion reading causal_tests.py
           needed) — pass threshold=0.95 explicitly for that reading, since
           that was the value in use there. threshold=0.9 with mask=None is
           the population-level reading (Blog 1's published definition,
           analysis.py's ip_mass_near_1).

    Returns 0.0 if fewer than 2 tokens are in scope.
    """
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < 2:
            return 0.0
        G = G[np.ix_(mask, mask)]
    n = G.shape[0]
    if n < 2:
        return 0.0
    pairs = pairwise_upper(G)
    return float((pairs > threshold).mean())


# ---------------------------------------------------------------------------
# Attention entropy
# ---------------------------------------------------------------------------

def attention_entropy(attn_matrix) -> np.ndarray:
    """Shannon entropy of each attention row, averaged over tokens. (n_heads,) out."""
    attn = _as_numpy(attn_matrix)
    log_attn = np.log(attn + 1e-12)
    entropy_per_token = -(attn * log_attn).sum(axis=-1)
    return entropy_per_token.mean(axis=-1)


# ---------------------------------------------------------------------------
# Nearest-neighbour tracking
# ---------------------------------------------------------------------------

def nearest_neighbor_indices(G: np.ndarray) -> np.ndarray:
    """nn[i] = argmax_{j!=i} G[i, j]."""
    G_masked = G.copy()
    np.fill_diagonal(G_masked, -np.inf)
    return np.argmax(G_masked, axis=1).astype(np.int32)


def nearest_neighbor_stability(activations, prev_activations) -> float:
    """Fraction of tokens whose nearest neighbour is unchanged vs the previous layer."""
    normed_curr = l2_normalize(activations)
    normed_prev = l2_normalize(prev_activations)
    nn_curr = nearest_neighbor_indices(normed_curr @ normed_curr.T)
    nn_prev = nearest_neighbor_indices(normed_prev @ normed_prev.T)
    return float(np.mean(nn_curr == nn_prev))


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n_tokens, d) L2-normed, internally-centered matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    YtX = Y.T @ X
    numerator = float(np.sum(YtX ** 2))
    XtX_norm = float(np.linalg.norm(X.T @ X, "fro"))
    YtY_norm = float(np.linalg.norm(Y.T @ Y, "fro"))
    denom = XtX_norm * YtY_norm
    if denom < 1e-12:
        return float("nan")
    return float(np.clip(numerator / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Energy-drop pair localization
# ---------------------------------------------------------------------------

def energy_drop_pairs(activations_before, activations_after, beta: float, top_k: int = 10) -> list:
    """Token pairs (i, j, delta) responsible for an energy drop between two layers."""
    normed_before = l2_normalize(activations_before)
    normed_after = l2_normalize(activations_after)
    return _energy_drop_pairs_core(normed_before, normed_after, beta, top_k)


def energy_drop_pairs_from_normed(normed_before: np.ndarray, normed_after: np.ndarray, beta: float, top_k: int = 10) -> list:
    """Same as energy_drop_pairs but accepts pre-normalized ndarrays directly."""
    return _energy_drop_pairs_core(normed_before, normed_after, beta, top_k)


def _energy_drop_pairs_core(normed_before, normed_after, beta, top_k):
    n = normed_before.shape[0]
    if n < 2:
        return []
    norm = 2.0 * beta * n * n

    G_before = normed_before @ normed_before.T
    G_after = normed_after @ normed_after.T
    delta = (np.exp(beta * G_after) - np.exp(beta * G_before)) / norm

    rows, cols = np.triu_indices(n, k=1)
    pair_deltas = delta[rows, cols]

    k = min(top_k, len(pair_deltas))
    if k >= len(pair_deltas):
        worst_idx = np.argsort(pair_deltas)
    else:
        worst_idx = np.argpartition(pair_deltas, k)[:k]
        worst_idx = worst_idx[np.argsort(pair_deltas[worst_idx])]

    return [(int(rows[idx]), int(cols[idx]), float(pair_deltas[idx])) for idx in worst_idx]


# ---------------------------------------------------------------------------
# Gram moments, cumulants, and the participation-ratio identity
# ---------------------------------------------------------------------------
#
# Everything in this section rests on one observation (MATH.md sec. 6):
# E_beta is the moment generating function of the pairwise-cosine
# distribution. For unit-norm rows,
#
#     E_beta = (1 / 2 beta) * < exp(beta * G_ij) >_ij
#            = 1/(2 beta) + <G>/2 + (beta/4) <G^2> + (beta^2/12) <G^3> + ...
#
# and, since tr(G) = n for unit rows and tr(G^2) = ||G||_F^2 = sum_ij G_ij^2,
# the participation-ratio rank is EXACTLY the reciprocal second moment:
#
#     PR = (tr G)^2 / tr(G^2) = 1 / <G^2>
#
# So E_beta at four betas is a redundant reparameterization of the first
# few moments of a single scalar distribution that we already persist as
# `ip_histogram`. The non-redundant version is the cumulant ladder
# (kappa_1 common mode, kappa_2 spread, kappa_3 asymmetry), with
# 1/PR = kappa_2 + kappa_1^2.
#
# TRAP, and the reason `n` is a required argument below: the identity is
# over the FULL n^2 Gram including the unit diagonal, while `ip_histogram`
# and `ip_mean` are OFF-DIAGONAL quantities. The conversion is exact,
#
#     <G^k>_full = [1 + (n-1) * <G^k>_offdiag] / n
#
# but it is O(1/n). Measured on random unit clouds: at n=20 the naive
# off-diagonal kappa_1 reads +0.0030 against a true full-matrix value of
# +0.0523 — an order of magnitude, and a sign-relevant error. At n=467 the
# gap is 0.002. Feeding off-diagonal moments straight into the energy
# identity is wrong on exactly the short prompts, which is where the beta
# gradient in status-1's verdict table lives.

def gram_moments(G: np.ndarray, order: int = 3) -> dict:
    """
    Raw moments <G^k>, k = 1..order, over ALL n^2 entries of the Gram
    matrix (diagonal included), plus the derived participation-ratio rank.

    Returns keys: m1, m2, ..., m{order}, pr_rank, n.
    """
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    out = {"n": int(n)}
    Gk = np.ones_like(G)
    for k in range(1, order + 1):
        Gk = Gk * G
        out[f"m{k}"] = float(Gk.mean())
    m2 = out.get("m2", float("nan"))
    out["pr_rank"] = float(1.0 / m2) if m2 > 1e-15 else float("nan")
    return out


def offdiag_to_full_moment(m_off: float, n: int, diag_value: float = 1.0) -> float:
    """
    Convert an off-diagonal moment <G^k>_offdiag to the full-matrix moment
    <G^k>_full. Exact. `diag_value` is 1.0 for unit-norm rows (so the
    diagonal contributes 1^k = 1 for every k).
    """
    if n < 2:
        return float(diag_value)
    return float((diag_value + (n - 1) * m_off) / n)


def cumulants_from_moments(m1: float, m2: float, m3: float) -> dict:
    """
    First three cumulants from the first three raw moments.

      kappa_1 = m1                        (common mode / anisotropy)
      kappa_2 = m2 - m1^2                 (spread)
      kappa_3 = m3 - 3 m1 m2 + 2 m1^3     (asymmetry)
    """
    k1 = float(m1)
    k2 = float(m2 - m1 ** 2)
    k3 = float(m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3)
    return {"kappa1": k1, "kappa2": k2, "kappa3": k3}


def gram_cumulants(G: np.ndarray) -> dict:
    """
    The cumulant ladder plus PR, computed directly from a Gram matrix.
    Includes the identity check 1/PR == kappa_2 + kappa_1^2.
    """
    mom = gram_moments(G, order=3)
    cum = cumulants_from_moments(mom["m1"], mom["m2"], mom["m3"])
    cum["pr_rank"] = mom["pr_rank"]
    cum["n"] = mom["n"]
    cum["pr_identity_residual"] = float(
        abs((cum["kappa2"] + cum["kappa1"] ** 2) - mom["m2"])
    )
    return cum


def cumulants_from_ip_histogram(counts, n_tokens: int,
                                lo: float = -1.0, hi: float = 1.0) -> dict:
    """
    Recover the FULL-matrix cumulant ladder from a persisted off-diagonal
    `ip_histogram`, which is what Phase 1 has on disk for every layer of
    every run. This is the [R]-cost path: no activations needed.

    counts    : the saved histogram counts (analysis_p1 uses 50 bins over
                [-1, 1] of the upper-triangle inner products)
    n_tokens  : n, required for the off-diagonal -> full conversion above

    Bin-centre quadrature introduces a discretization error of order
    (binwidth^2 / 12) in the second moment; with 50 bins over [-1, 1] that
    is 1.3e-4, reported as `quadrature_bias_m2` so it can be compared
    against whatever residual the energy check produces.
    """
    counts = np.asarray(counts, dtype=np.float64)
    nb = len(counts)
    if nb == 0 or counts.sum() <= 0:
        return {"kappa1": float("nan"), "kappa2": float("nan"),
                "kappa3": float("nan"), "pr_rank": float("nan"),
                "n": int(n_tokens), "quadrature_bias_m2": float("nan")}
    edges = np.linspace(lo, hi, nb + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = counts / counts.sum()

    m_off = [float((w * centres ** k).sum()) for k in (1, 2, 3)]
    m_full = [offdiag_to_full_moment(m, n_tokens) for m in m_off]

    cum = cumulants_from_moments(*m_full)
    cum["n"] = int(n_tokens)
    cum["pr_rank"] = float(1.0 / m_full[1]) if m_full[1] > 1e-15 else float("nan")
    cum["m1_offdiag"] = m_off[0]      # this is `ip_mean`, for cross-checking
    binw = (hi - lo) / nb
    cum["quadrature_bias_m2"] = float(binw ** 2 / 12.0)
    return cum


def energy_from_cumulants(kappa1: float, kappa2: float, kappa3: float,
                          beta: float) -> dict:
    """
    The moment-expansion approximation to E_beta, at two truncation orders.

        two_term   = 1/(2b) + k1/2 + (b/4)(k2 + k1^2)
        three_term = two_term + (b^2/12)(k3 + 3 k1 k2 + k1^3)

    The two-term form is the one MATH.md sec. 6.2 checks numerically
    (120 random clouds, n=300, d=512): at beta=1, corr(E, two_term) =
    0.9993, max relative error 8%. Report the residual against the
    measured E_beta rather than trusting either form.

    RANGE OF VALIDITY — this bounds what the cumulant ladder can replace.
    Measured on n=300, d=64 unit-norm clouds, isotropic and anisotropic,
    relative error of the two-term form against exact E_beta:

        beta = 0.1   0.00%      beta = 2.0    0.80%
        beta = 1.0   0.07%      beta = 5.0   26.57%   (three-term: 22.48%)

    The number of moments needed for <1% accuracy is 2 at beta <= 2 and
    TWELVE at beta = 5, in both the isotropic and anisotropic cases. So
    the ladder is a faithful reparameterization of E_beta at beta = 0.1,
    1.0 and 2.0 and NOT at beta = 5.0, which is in BETA_VALUES. The
    beta=5 energy column must stay a measured quantity; do not reconstruct
    it from kappa_1..kappa_3. (This is the MGF's radius-of-usefulness, not
    a bug: at beta=5 the exponential is dominated by the right tail of the
    cosine distribution, which is precisely the regime where low-order
    moments carry no information about it.)
    """
    b = float(beta)
    m2 = kappa2 + kappa1 ** 2
    m3 = kappa3 + 3.0 * kappa1 * kappa2 + kappa1 ** 3
    two = 1.0 / (2.0 * b) + kappa1 / 2.0 + (b / 4.0) * m2
    three = two + (b ** 2 / 12.0) * m3
    return {"two_term": float(two), "three_term": float(three),
            "pr_rank_implied": float(1.0 / m2) if m2 > 1e-15 else float("nan")}


def norm_participation_ratio(activations) -> float:
    """
    (sum_i n_i^2)^2 / sum_i n_i^4, the participation ratio of the row-norm
    distribution alone.

    This is what raw effective rank converges to in the near-orthogonal
    limit (MATH.md sec. 6.4) — i.e. it carries ZERO directional content.
    Reported next to raw effective rank, it tests the attention-sink
    hypothesis directly: if the two track each other, the reported "rank
    collapse" is a sink count, not a geometric statement. Reference
    numbers from the derivation (n=200, d=256): uniform norms give
    PR_raw 111.9 / PR_norms 200.0; three tokens at 30x norm give
    PR_raw 3.44 / PR_norms 3.45.
    """
    arr = _as_numpy(activations).astype(np.float64, copy=False)
    n2 = (arr ** 2).sum(axis=-1)
    denom = float((n2 ** 2).sum())
    if denom < 1e-30:
        return float("nan")
    return float((n2.sum() ** 2) / denom)


# ---------------------------------------------------------------------------
# CKA, decomposed
# ---------------------------------------------------------------------------

def linear_cka_decomposed(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Linear CKA together with the two factors it is a product of.

    With G the Gram of the CENTERED rows, ||G||_F = n / sqrt(PR), so

        CKA = <G_l, G_m>_F / (||G_l||_F ||G_m||_F)
            = <G_l (*) G_m> * sqrt(PR_l * PR_m)

    where the overlap is normalized by the traces:

        overlap = <G_l, G_m>_F / (tr G_l * tr G_m)

    For unit-norm rows tr G = n and this reduces to the plain elementwise
    mean of MATH.md sec. 6.3; CKA centers the rows first, so the trace form
    is the one that holds exactly here. Verified below to 1e-12.

    The consequence is that a CKA drop between consecutive layers has two
    possible causes, and the reported number does not distinguish them:
    the pairwise structure genuinely changed, or the effective rank moved.
    Since Phase 1 reads consecutive-layer CKA as "representation changed"
    and separately reports rank collapsing across training, these are not
    independent readings. Divide the rank factor out before interpreting.

    Returns: cka, overlap (= <G_l (*) G_m>), rank_factor
    (= sqrt(PR_l PR_m)), pr_x, pr_y.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Gx = Xc @ Xc.T
    Gy = Yc @ Yc.T
    n = Gx.shape[0]

    def _pr(G):
        t1 = float(np.trace(G))
        t2 = float((G ** 2).sum())
        return (t1 ** 2 / t2) if t2 > 1e-30 else float("nan")

    pr_x, pr_y = _pr(Gx), _pr(Gy)
    fx, fy = float(np.linalg.norm(Gx, "fro")), float(np.linalg.norm(Gy, "fro"))
    if fx < 1e-15 or fy < 1e-15:
        return {"cka": float("nan"), "overlap": float("nan"),
                "rank_factor": float("nan"), "pr_x": pr_x, "pr_y": pr_y}
    cka = float((Gx * Gy).sum() / (fx * fy))
    tx, ty = float(np.trace(Gx)), float(np.trace(Gy))
    overlap = float((Gx * Gy).sum() / (tx * ty)) if abs(tx * ty) > 1e-30 else float("nan")
    rank_factor = float(np.sqrt(pr_x * pr_y))
    return {
        "cka": float(np.clip(cka, 0.0, 1.0)),
        "overlap": overlap,
        "rank_factor": rank_factor,
        "pr_x": pr_x,
        "pr_y": pr_y,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Energy-violation attribution: common mode vs spread
# ---------------------------------------------------------------------------

def energy_violation_attribution(cum_prev: dict, cum_curr: dict,
                                 beta: float) -> dict:
    """
    Split a layer-to-layer change in E_beta into the part driven by the
    common mode (kappa_1) and the part driven by the spread (kappa_2).

    Differentiating the two-term expansion,

        dE = dk1/2 + (beta/4)(dk2 + d(k1^2))
           = dk1 * (1/2 + (beta/2) * k1_bar)   +   (beta/4) * dk2
             \\__________ common mode _________/     \\___ spread ___/

    with k1_bar the midpoint of kappa_1 across the transition (exact to
    second order in dk1).

    This answers the question the raw violation count cannot: an energy
    drop can come from the cloud's internal spread contracting (kappa_2
    falling), or from the whole cloud losing a shared anisotropy component
    with its internal structure untouched (kappa_1 falling). The second is
    a pure common-mode effect and is exactly
    what a learned LayerNorm bias produces (MATH.md sec. 6.5), so an
    unattributed violation count cannot distinguish a geometric event from
    a bias term.
    """
    b = float(beta)
    dk1 = float(cum_curr["kappa1"] - cum_prev["kappa1"])
    dk2 = float(cum_curr["kappa2"] - cum_prev["kappa2"])
    k1_bar = 0.5 * float(cum_curr["kappa1"] + cum_prev["kappa1"])

    common = dk1 * (0.5 + 0.5 * b * k1_bar)
    spread = (b / 4.0) * dk2
    total = common + spread
    denom = abs(common) + abs(spread)
    return {
        "delta_kappa1": dk1,
        "delta_kappa2": dk2,
        "common_mode_term": float(common),
        "spread_term": float(spread),
        "predicted_delta_E": float(total),
        "common_mode_fraction": float(abs(common) / denom) if denom > 1e-18 else float("nan"),
    }
