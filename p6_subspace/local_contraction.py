"""
local_contraction.py — Track B: Per-cluster local linear maps.

During a plateau, each cluster is near an attractor of S. Locally, the
dynamics should be contracting in the real subspace (S eigenvalues < 1)
and neutrally rotating in the imaginary subspace (A eigenvalues on unit circle).

At merge events, S's local contraction should destabilise: at least one
S-eigenvalue near or outside the unit disk.

This is the per-cluster, layer-resolved version of the Phase 1 global
spectral radius finding (ALBERT-xlarge's V > 1 predicts collapse).

Falsifiable predictions tested
-------------------------------
P6-R5 : Plateau layers: W_C^S has spectral radius < 1 (local contraction).
         Merge layers:   W_C^S has spectral radius ≥ 1 (attractor destabilising).
         Throughout:     W_C^A has spectral radius ≈ 1 (rotation, norm-preserving).

Functions
---------
fit_local_map          : least-squares W_C from x^{L} → x^{L+1}
decompose_local_map    : W_C → W_C^S + W_C^A (symmetric + antisymmetric)
spectral_radius        : max |eigenvalue| of a square matrix
local_map_profile      : all-layers profile for one cluster
run_local_contraction  : full pipeline → SubResult
"""

import numpy as np
from scipy.linalg import eigvals

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Local linear map
# ---------------------------------------------------------------------------

def fit_local_map(
    X_curr: np.ndarray,
    X_next: np.ndarray,
) -> np.ndarray | None:
    """
    Fit a linear map W such that X_next ≈ X_curr @ W.

    Solves: W = argmin ||X_curr @ W - X_next||_F²
    via least squares (normal equations).

    np.linalg.lstsq(X_curr, X_next) where both are (n, d) returns W of
    shape (d, d) directly — no transposition needed.

    Parameters
    ----------
    X_curr : (n, d) — token activations at layer L
    X_next : (n, d) — token activations at layer L+1

    Returns
    -------
    W : (d, d) or None if under-determined (too few tokens or low-rank fit)
    """
    n, d = X_curr.shape
    if n < 4:
        return None

    # lstsq solves X_curr @ W ≈ X_next; both inputs are (n, d) → W is (d, d)
    W, _, rank, _ = np.linalg.lstsq(X_curr, X_next, rcond=None)

    if rank < min(n, d) * 0.5:
        # Very low rank fit — unreliable
        return None

    # FIX (Bug 6): removed incorrect conditional "return W.T if W.shape == (n, d) else W".
    # lstsq(A, B) where A is (n, d) and B is (n, d) always returns W of shape (d, d).
    # The transpose condition W.shape == (n, d) is only True when n == d (square
    # coincidence), at which point it would incorrectly transpose an already-correct
    # result.  The unconditional return below is always right.
    return W

def decompose_local_map(W: np.ndarray) -> dict:
    """
    Decompose local map W into symmetric S and antisymmetric A parts.

    W_S = (W + W^T) / 2
    W_A = (W - W^T) / 2

    Returns
    -------
    dict with W_S, W_A, rho_S (spectral radius of W_S), rho_A, rho_W,
    contracting_S (rho_S < 1), neutral_A (|rho_A - 1| < 0.15)
    """
    W_S = (W + W.T) / 2.0
    W_A = (W - W.T) / 2.0

    rho_W = spectral_radius(W)
    rho_S = spectral_radius(W_S)
    rho_A = spectral_radius(W_A)

    return {
        "W_S":          W_S,
        "W_A":          W_A,
        "rho_W":        rho_W,
        "rho_S":        rho_S,
        "rho_A":        rho_A,
        "contracting_S": bool(rho_S < 1.0),
        "neutral_A":     bool(abs(rho_A - 1.0) < 0.15),
    }


def spectral_radius(M: np.ndarray) -> float:
    """Max absolute eigenvalue of square matrix M."""
    try:
        ev = eigvals(M)
        return float(np.max(np.abs(ev)))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Per-cluster, per-layer profile
# ---------------------------------------------------------------------------

def local_map_profile(
    activations_per_layer: list[np.ndarray],
    labels_per_layer:      list[np.ndarray],
    layer_types:           list[str],
    layer_names:           list[str],
    cluster_id:            int,
    min_tokens:            int = 4,
) -> list[dict]:
    """
    Fit and decompose the local linear map for one cluster at each layer transition.

    Parameters
    ----------
    min_tokens : skip transitions where cluster has fewer than this many tokens

    Returns
    -------
    list of dicts, one per transition where fit succeeds
    """
    results = []
    n_layers = len(activations_per_layer)

    for L in range(n_layers - 1):
        labels_cur = labels_per_layer[L]
        labels_nxt = labels_per_layer[L + 1]

        mask_cur = labels_cur == cluster_id
        mask_nxt = labels_nxt == cluster_id

        if mask_cur.sum() < min_tokens:
            continue

        X_cur = activations_per_layer[L][mask_cur]

        # For X_next: use same token positions (not cluster membership at L+1)
        # This tracks the same tokens across the transition
        token_indices = np.where(mask_cur)[0]
        X_nxt = activations_per_layer[L + 1][token_indices]

        W = fit_local_map(X_cur, X_nxt)
        if W is None:
            continue

        decomp = decompose_local_map(W)

        results.append({
            "layer_from":   layer_names[L],
            "layer_to":     layer_names[L + 1],
            "layer_type_L": layer_types[L],
            "n_tokens":     int(mask_cur.sum()),
            "cluster_id":   cluster_id,
            "rho_W":        decomp["rho_W"],
            "rho_S":        decomp["rho_S"],
            "rho_A":        decomp["rho_A"],
            "contracting_S": decomp["rho_S"] < 1.0,
            "neutral_A":     abs(decomp["rho_A"] - 1.0) < 0.15,
        })

    return results


# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_local_contraction(ctx: dict) -> SubResult:
    """
    Track B sub-experiment: per-cluster local contraction analysis.

    Required ctx keys
    -----------------
    activations_per_layer : list of (n, d)
    labels_per_layer      : list of (n,)
    layer_type_labels     : list of str
    layer_names           : list of str

    Optional ctx keys
    -----------------
    tracked_cluster_ids   : list[int] (default: all unique non-noise)
    min_tokens_for_fit    : int (default 4)
    """
    acts        = ctx["activations_per_layer"]
    labels      = ctx["labels_per_layer"]
    layer_types = ctx["layer_type_labels"]
    layer_names = ctx["layer_names"]
    min_tokens  = ctx.get("min_tokens_for_fit", 4)

    all_labels = np.unique(np.concatenate([l[l >= 0] for l in labels if (l >= 0).any()]))
    tracked = ctx.get("tracked_cluster_ids", all_labels.tolist())

    all_steps: list[dict] = []
    for cid in tracked:
        steps = local_map_profile(
            acts, labels, layer_types, layer_names, int(cid), min_tokens
        )
        all_steps.extend(steps)

    if not all_steps:
        return SubResult(
            name="local_contraction",
            applicable=False,
            payload={},
            summary_lines=["local_contraction: no valid fits (too few tokens per cluster)"],
            verdict_contribution={},
        )

    # Aggregate by layer type
    def _agg(steps, ltype, key):
        vals = [s[key] for s in steps if s["layer_type_L"] == ltype and s[key] is not None]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)

    mu_rho_S_plat, std_rho_S_plat, n_plat = _agg(all_steps, "plateau", "rho_S")
    mu_rho_S_merg, std_rho_S_merg, n_merg = _agg(all_steps, "merge",   "rho_S")
    mu_rho_A_plat, std_rho_A_plat, _      = _agg(all_steps, "plateau", "rho_A")
    mu_rho_A_merg, std_rho_A_merg, _      = _agg(all_steps, "merge",   "rho_A")

    # P6-R5 checks
    n_contracting_S_plat = sum(
        1 for s in all_steps
        if s["layer_type_L"] == "plateau" and s["contracting_S"]
    )
    n_neutral_A_plat = sum(
        1 for s in all_steps
        if s["layer_type_L"] == "plateau" and s["neutral_A"]
    )
    n_destab_S_merg = sum(
        1 for s in all_steps
        if s["layer_type_L"] == "merge" and not s["contracting_S"]
    )

    p6_r5_contraction = (
        n_contracting_S_plat > n_plat * 0.7 if n_plat > 0 else None
    )
    p6_r5_neutral_A = (
        n_neutral_A_plat > n_plat * 0.7 if n_plat > 0 else None
    )
    p6_r5_destab = (
        n_destab_S_merg > n_merg * 0.5 if n_merg > 0 else None
    )
    p6_r5_satisfied = bool(p6_r5_contraction and p6_r5_neutral_A)

    payload = {
        "n_steps_total":         len(all_steps),
        "n_steps_plateau":       n_plat,
        "n_steps_merge":         n_merg,
        "mean_rho_S_plateau":    mu_rho_S_plat,
        "std_rho_S_plateau":     std_rho_S_plat,
        "mean_rho_S_merge":      mu_rho_S_merg,
        "std_rho_S_merge":       std_rho_S_merg,
        "mean_rho_A_plateau":    mu_rho_A_plat,
        "std_rho_A_plateau":     std_rho_A_plat,
        "n_contracting_S_plat":  n_contracting_S_plat,
        "n_neutral_A_plat":      n_neutral_A_plat,
        "n_destab_S_merge":      n_destab_S_merg,
        "p6_r5_satisfied":       p6_r5_satisfied,
    }

    # --- Summary lines ---
    lines = [
        SEP_THICK,
        "LOCAL CONTRACTION ANALYSIS  [Track B]",
        SEP_THICK,
        f"Total transition fits:       {len(all_steps)}",
        f"  plateau transitions:       {n_plat}",
        f"  merge transitions:         {n_merg}",
        "",
        "Spectral radius of per-cluster local linear map, symmetric part W_C^S:",
        _bullet("mean ρ_S at plateau layers", mu_rho_S_plat),
        _bullet("std  ρ_S at plateau layers", std_rho_S_plat),
        _bullet("mean ρ_S at merge layers",   mu_rho_S_merg),
        _bullet("std  ρ_S at merge layers",   std_rho_S_merg),
        "",
        "Spectral radius of antisymmetric part W_C^A (should be ≈ 1.0 throughout):",
        _bullet("mean ρ_A at plateau layers", mu_rho_A_plat),
        _bullet("std  ρ_A at plateau layers", std_rho_A_plat),
        _bullet("mean ρ_A at merge layers",   mu_rho_A_merg),
        _bullet("std  ρ_A at merge layers",   std_rho_A_merg),
        "",
        "P6-R5 component checks:",
        _bullet("plateau steps with ρ_S < 1 (contracting)", n_contracting_S_plat),
        _bullet("plateau steps with |ρ_A - 1| < 0.15 (neutral)", n_neutral_A_plat),
        _bullet("merge steps with ρ_S ≥ 1 (destabilising)", n_destab_S_merg),
        "",
        "Prediction P6-R5: W_C^S contracts at plateau, destabilises at merge;",
        "                  W_C^A has spectral radius ≈ 1 throughout.",
        _verdict_line(
            "P6-R5 (contraction at plateau)",
            p6_r5_contraction,
            f"{n_contracting_S_plat}/{n_plat} plateau steps with ρ_S < 1"
            f" (mean ρ_S={_fmt(mu_rho_S_plat)})",
        ),
        _verdict_line(
            "P6-R5 (neutral rotation at plateau)",
            p6_r5_neutral_A,
            f"{n_neutral_A_plat}/{n_plat} plateau steps with |ρ_A−1|<0.15"
            f" (mean ρ_A={_fmt(mu_rho_A_plat)})",
        ),
        _verdict_line(
            "P6-R5 (destabilisation at merge)",
            p6_r5_destab,
            f"{n_destab_S_merg}/{n_merg} merge steps with ρ_S ≥ 1"
            f" (mean ρ_S_merge={_fmt(mu_rho_S_merg)})",
        ),
    ]

    vc = {
        "lc_mean_rho_S_plateau":   mu_rho_S_plat,
        "lc_mean_rho_S_merge":     mu_rho_S_merg,
        "lc_mean_rho_A_plateau":   mu_rho_A_plat,
        "lc_n_contracting_plateau": n_contracting_S_plat,
        "lc_n_neutral_A_plateau":  n_neutral_A_plat,
        "lc_n_destab_merge":       n_destab_S_merg,
        "lc_p6_r5_satisfied":      p6_r5_satisfied,
    }

    return SubResult(
        name="local_contraction",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )
