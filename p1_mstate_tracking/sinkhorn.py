"""
sinkhorn.py — Sinkhorn-Knopp doubly stochastic normalization + Fiedler analysis.

Motivated by Sander et al. (Sinkformers) and Section 3.3 of Geshkovski et al.
A doubly stochastic attention matrix is the gradient-flow object; the gap
between raw attention and doubly stochastic form measures deviation from
idealized dynamics.

Fix 3 (causal-mask confound):
    GPT-2 attention is lower-triangular.  Sinkhorn + symmetrize (P+Pᵀ)/2 on
    a lower-triangular matrix forces a low-connectivity graph structure
    independent of content, likely manufacturing "100% STABLE-CLUSTER" across
    prompts.  The fix has two parts:

    1.  Baseline subtraction — build a content-free attention (uniform within
        the causal triangle), compute its per-head Fiedler, and report each
        head's actual Fiedler *minus* this mask-only baseline.  Classify
        CLUSTER/MIXED/MIXING on the *deviation*, not the raw value.

    2.  Causal-mask control — apply an artificial causal mask to the input
        (zero upper triangle + row-renorm) before Sinkhorn.  Running this on
        BERT tests whether masking alone collapses heads to "STABLE-CLUSTER".
        If it does, the GPT-2/BERT split is mask-driven, not weight-driven,
        and the routing claim is withdrawn.

Functions
---------
sinkhorn_normalize              : iterative row/col normalisation (single head)
sinkhorn_normalize_batched      : vectorised normalisation across all heads
fiedler_value                   : λ₂ of the normalised Laplacian
sinkhorn_cluster_count          : eigenvalues near 1 ≈ cluster count
_uniform_causal_attention       : content-free lower-triangular baseline
causal_fiedler_baseline         : Fiedler of the mask-only baseline
analyze_attention_sinkhorn      : per-head summary dict for one attention layer
"""

import numpy as np
import torch

from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian

from core.config import SINKHORN_MAX_ITER, SINKHORN_TOL


# ---------------------------------------------------------------------------
# Core Sinkhorn normalisation
# ---------------------------------------------------------------------------

def sinkhorn_normalize(
    A: np.ndarray,
    max_iter: int = SINKHORN_MAX_ITER,
    tol: float = SINKHORN_TOL,
) -> np.ndarray:
    """
    Iteratively row- and column-normalise *A* until it is doubly stochastic.

    Convergence is declared when the max elementwise change < *tol*.
    Single-head version — kept for external use and the Fiedler/cluster
    functions that operate on one matrix at a time.
    """
    P, _info = sinkhorn_normalize_with_info(A, max_iter=max_iter, tol=tol)
    return P


def sinkhorn_normalize_with_info(
    A: np.ndarray,
    max_iter: int = SINKHORN_MAX_ITER,
    tol: float = SINKHORN_TOL,
):
    """
    As sinkhorn_normalize, but returns (P, info) where info records whether
    the iteration actually converged.

    status-1 defect D9: the cap was hit silently. The n=20 causal baseline
    needs 232 iterations to reach tol=1e-6, and the old cap was 100, so
    every short-prompt run was returning a matrix with residual ~4.7e-4
    while reporting nothing. The lambda_2 error is negligible for the
    uniform baseline (0.108894 vs 0.108889) but real attention is more
    peaked and converges more slowly, and no per-layer residual existed
    anywhere in the artifact to check that against.

    info keys: converged (bool), n_iter (int), residual (float, the final
    max elementwise change), max_iter, tol.
    """
    P = np.clip(np.asarray(A).copy().astype(np.float64), 1e-12, None)
    residual = float("inf")
    n_iter = 0
    converged = False
    for it in range(max_iter):
        P_prev = P.copy()
        P      = P / P.sum(axis=1, keepdims=True)
        P      = P / P.sum(axis=0, keepdims=True)
        residual = float(np.abs(P - P_prev).max())
        n_iter = it + 1
        if residual < tol:
            converged = True
            break
    return P, {
        "converged": bool(converged),
        "n_iter": int(n_iter),
        "residual": residual,
        "max_iter": int(max_iter),
        "tol": float(tol),
    }


def sinkhorn_normalize_batched(
    A: np.ndarray,
    max_iter: int = SINKHORN_MAX_ITER,
    tol: float = SINKHORN_TOL,
) -> np.ndarray:
    """
    Vectorised Sinkhorn-Knopp across all attention heads simultaneously.

    Parameters
    ----------
    A : (n_heads, n_tokens, n_tokens)  raw attention weights

    Returns
    -------
    P : (n_heads, n_tokens, n_tokens)  doubly stochastic matrices
    """
    P, _info = sinkhorn_normalize_batched_with_info(A, max_iter=max_iter, tol=tol)
    return P


def sinkhorn_normalize_batched_with_info(
    A: np.ndarray,
    max_iter: int = SINKHORN_MAX_ITER,
    tol: float = SINKHORN_TOL,
):
    """
    As sinkhorn_normalize_batched, but returns (P, info).

    The batched loop breaks on the max residual ACROSS ALL HEADS, so a
    single slow head holds every head in the iteration — which is the
    correct behaviour, but it means a per-head residual is the only way to
    know which head was responsible. info["residual_per_head"] carries it.
    """
    P = np.clip(np.asarray(A).astype(np.float64), 1e-12, None)
    residual = float("inf")
    per_head = None
    n_iter = 0
    converged = False
    for it in range(max_iter):
        P_prev = P.copy()
        P     /= P.sum(axis=2, keepdims=True)   # row-normalise all heads
        P     /= P.sum(axis=1, keepdims=True)   # col-normalise all heads
        delta = np.abs(P - P_prev)
        per_head = delta.reshape(delta.shape[0], -1).max(axis=1)
        residual = float(per_head.max())
        n_iter = it + 1
        if residual < tol:
            converged = True
            break
    return P, {
        "converged": bool(converged),
        "n_iter": int(n_iter),
        "residual": residual,
        "residual_per_head": (per_head.tolist() if per_head is not None else []),
        "max_iter": int(max_iter),
        "tol": float(tol),
    }


# ---------------------------------------------------------------------------
# Fiedler / cluster-count helpers
# ---------------------------------------------------------------------------

def fiedler_value(P: np.ndarray) -> float:
    """
    Second-smallest eigenvalue (λ₂) of the normalised Laplacian of P.

    Interpretation:
      λ₂ ≈ 0  → near-disconnected components → strong cluster separation
      λ₂ large → well-connected → tokens mixing freely

    A low Fiedler value at a given layer indicates attention routing
    consistent with a metastable state.
    """
    P_sym       = (P + P.T) / 2
    L           = laplacian(P_sym, normed=True)
    n           = L.shape[0]
    k           = min(3, n - 1)
    eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, k - 1])
    return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


def sinkhorn_cluster_count(P: np.ndarray, min_gap_ratio: float = 0.1) -> int:
    """
    Estimate cluster count from the eigengap of the doubly stochastic matrix P.

    Fix 8: replaces the hard > 0.5 threshold with an eigengap method.

    For a k-cluster structure the k largest eigenvalues sit near 1 with a
    drop below them.  The largest gap in the descending eigenvalue sequence
    identifies k without a hard threshold on the eigenvalue magnitude.

    Falls back to the hard > 0.5 count when no clear gap exists, i.e. when
    the largest gap is less than min_gap_ratio * (λ_max − λ_min).  This
    keeps the function robust on near-uniform matrices (post-collapse layers)
    where there is no genuine structure to detect.

    Parameters
    ----------
    P             : doubly stochastic matrix (n_tokens, n_tokens)
    min_gap_ratio : gap must be at least this fraction of the total eigenvalue
                    range to count as a genuine cluster boundary.  Default 0.1;
                    include in the Fix 7 tolerance sweep.

    Returns
    -------
    k : int ≥ 1

    See sinkhorn_cluster_count_traced for the branch-recording version.
    status-1's design note applies here: on a model where no clear eigengap
    EVER exists, the "fallback" is the whole metric, and nothing in the
    output distinguished the two branches.
    """
    return sinkhorn_cluster_count_traced(P, min_gap_ratio)[0]


def sinkhorn_cluster_count_traced(P: np.ndarray, min_gap_ratio: float = 0.1):
    """
    As sinkhorn_cluster_count, but returns (k, branch) where branch is one
    of: "degenerate_n" (n < 2), "uniform_spectrum" (eigenvalue range below
    1e-6), "hard_threshold_fallback" (no gap exceeded min_gap_ratio, so the
    pre-Fix-8 >0.5 count was used), or "eigengap".

    The design principle this implements: any filter, gate, or fallback
    whose behaviour depends on the data must record what it did in the
    artifact, because the alternative is a curve that looks like a finding
    and is partly a filter.
    """
    eigs = np.real(np.linalg.eigvals(P))
    eigs = np.sort(eigs)[::-1]          # descending
    n    = len(eigs)
    if n < 2:
        return 1, "degenerate_n"

    gap_sizes  = np.abs(np.diff(eigs))  # |λ_i − λ_{i+1}|
    eig_range  = float(eigs[0] - eigs[-1])

    if eig_range < 1e-6:
        return 1, "uniform_spectrum"     # no structure

    largest_gap_pos = int(np.argmax(gap_sizes))
    if gap_sizes[largest_gap_pos] / eig_range < min_gap_ratio:
        # No clear gap — fall back to hard threshold
        return max(1, int((eigs > 0.5).sum())), "hard_threshold_fallback"

    return max(1, largest_gap_pos + 1), "eigengap"


# ---------------------------------------------------------------------------
# Fix 3 — causal-mask baseline
# ---------------------------------------------------------------------------

def _uniform_causal_attention(n: int) -> np.ndarray:
    """
    Content-free causal attention: token i attends uniformly to tokens 0..i.

    This is the mask-only baseline — what a causal (GPT-2 style) model
    produces when all QK logits are identical (zero before softmax).

      A[i, j] = 1 / (i + 1)   for j <= i
      A[i, j] = 0              for j >  i

    Parameters
    ----------
    n : sequence length

    Returns
    -------
    A : (n, n) float64 lower-triangular attention matrix, rows sum to 1
    """
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, :i + 1] = 1.0 / (i + 1)
    return A


def causal_fiedler_baseline(n: int) -> float:
    """
    Fiedler value of the content-free causal attention (mask-only baseline).

    Used to separate content-driven connectivity from the structural effect of
    the causal mask.  Classification on the deviation (actual − baseline)
    answers: "does this head route into clusters *beyond* what the mask forces?"

    Parameters
    ----------
    n : sequence length

    Returns
    -------
    float — λ₂ of the Sinkhorn-normalised uniform-causal attention Laplacian
    """
    A_base = _uniform_causal_attention(n)
    P_base = sinkhorn_normalize(A_base)
    return fiedler_value(P_base)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_attention_sinkhorn(
    attn_matrix: torch.Tensor,
    is_causal: bool = False,
    apply_causal_control: bool = False,
) -> dict:
    """
    Run per-head Sinkhorn analysis for one attention layer.

    Parameters
    ----------
    attn_matrix          : (n_heads, n_tokens, n_tokens) float tensor
    is_causal            : True for decoder-only (GPT-2 style) models.
                           Computes a per-sequence-length mask-only Fiedler
                           baseline and reports per-head deviations
                           (actual − baseline).  The deviation is the correct
                           quantity for CLUSTER/MIXED/MIXING classification in
                           causal models; raw values are retained for reference.
    apply_causal_control : When True, applies an artificial causal mask to the
                           input (zero upper triangle + row-renorm) before
                           Sinkhorn and reports Fiedler on the masked version.
                           Use on BERT to test whether masking alone produces
                           low Fiedler.  If BERT heads also collapse to
                           STABLE-CLUSTER under this control, the GPT-2/BERT
                           split is mask-driven, not weight-driven, and the
                           "weight-level routing" claim is withdrawn.

    Returns
    -------
    dict with keys:
      fiedler_mean                     float  — mean λ₂ across heads (raw)
      fiedler_per_head                 list   — raw λ₂ for each head
      fiedler_per_head_deviation       list   — [causal] actual − baseline per head
      fiedler_baseline                 float  — [causal] mask-only baseline Fiedler
      fiedler_causal_control_per_head  list   — [control] Fiedler after artificial mask
      sinkhorn_cluster_count_mean      float  — mean cluster count across heads
      sinkhorn_cluster_counts          list   — count per head
      row_col_balance_mean             float  — mean std of raw attention column sums
                                                (0 = already doubly stochastic)
    """
    attn    = attn_matrix.numpy()    # (n_heads, n, n)
    n_heads = attn.shape[0]
    n       = attn.shape[-1]

    # Row/col balance on raw attention (content-independent diagnostic)
    col_sums        = attn.sum(axis=1)                   # (n_heads, n)
    row_col_balance = np.std(col_sums, axis=1).tolist()  # (n_heads,)

    # All heads normalised in one batched call. D9: the convergence info is
    # now captured rather than discarded, so a run that hit the iteration
    # cap is visible in the artifact instead of silently returning a matrix
    # that is not doubly stochastic.
    P_all, sk_info = sinkhorn_normalize_batched_with_info(attn)   # (n_heads, n, n)

    fiedler_vals = [fiedler_value(P_all[h]) for h in range(n_heads)]
    counts_and_branches = [sinkhorn_cluster_count_traced(P_all[h]) for h in range(n_heads)]
    cluster_counts   = [c for c, _ in counts_and_branches]
    cluster_branches = [b for _, b in counts_and_branches]

    result = {
        "fiedler_mean":                float(np.mean(fiedler_vals)),
        "fiedler_per_head":            fiedler_vals,
        "sinkhorn_cluster_count_mean": float(np.mean(cluster_counts)),
        "sinkhorn_cluster_counts":     cluster_counts,
        # Which branch produced each head's count. On a model where no
        # clear eigengap ever exists, "hard_threshold_fallback" IS the
        # metric, and that has to be readable from the artifact.
        "sinkhorn_cluster_count_branches": cluster_branches,
        "sinkhorn_fallback_fraction": float(
            sum(1 for b in cluster_branches if b == "hard_threshold_fallback") / max(n_heads, 1)
        ),
        "row_col_balance_mean":        float(np.mean(row_col_balance)),
        # D9 — convergence diagnostics, per layer.
        "sinkhorn_converged":     sk_info["converged"],
        "sinkhorn_n_iter":        sk_info["n_iter"],
        "sinkhorn_residual":      sk_info["residual"],
        "sinkhorn_residual_max_head": (
            int(np.argmax(sk_info["residual_per_head"]))
            if sk_info["residual_per_head"] else -1
        ),
    }

    # ------------------------------------------------------------------
    # Fix 3a — causal-mask baseline subtraction
    # ------------------------------------------------------------------
    if is_causal:
        baseline = causal_fiedler_baseline(n)
        result["fiedler_baseline"]           = baseline
        result["fiedler_per_head_deviation"] = [
            round(f - baseline, 6) for f in fiedler_vals
        ]

    # ------------------------------------------------------------------
    # Fix 3b — causal-mask control (run on BERT to check mask confound)
    # ------------------------------------------------------------------
    if apply_causal_control:
        causal_mask = np.tril(np.ones((n, n), dtype=np.float64))
        attn_masked = attn * causal_mask[None, :, :]        # zero upper tri
        row_sums    = attn_masked.sum(axis=2, keepdims=True)
        row_sums    = np.where(row_sums < 1e-12, 1.0, row_sums)
        attn_masked = attn_masked / row_sums                # re-normalise rows
        P_ctrl, _   = sinkhorn_normalize_batched_with_info(attn_masked)
        result["fiedler_causal_control_per_head"] = [
            fiedler_value(P_ctrl[h]) for h in range(n_heads)
        ]

    return result