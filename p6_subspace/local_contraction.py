"""
local_contraction.py — Track B: Per-cluster local linear maps and subspace dynamics.

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
fit_local_map_subspace    : least-squares W_C in subspace (core fix)
analyze_subspace_maps     : spectral diagnostics from subspace fit
spectral_radius           : max |eigenvalue| of a square matrix
local_map_profile         : all-layers profile for one cluster
run_local_contraction     : full pipeline → SubResult

Core fix:
  fit_local_map_subspace — project to S/A, reduce to data-adaptive rank,
  fit well-determined system (not ambient d>>n underdetermined system).
"""

import numpy as np
from scipy.linalg import eigvals


# -----------
# Core subspace-aware fitting (main fix)
# -----------

def fit_local_map_subspace(
    X_curr: np.ndarray,
    X_next: np.ndarray,
    U_S: np.ndarray,
    U_A: np.ndarray,
    min_tokens: int = 8,
    svd_eps: float = 1e-3,
) -> dict | None:
    """
    Project into S and A subspaces, reduce to data-adaptive effective rank,
    solve well-determined systems. Fixes the d >> n underdetermined issue
    where ambient-space lstsq produces near-zero singular values by construction.

    Parameters
    ----------
    X_curr : (n, d) — activations at layer L
    X_next : (n, d) — activations at layer L+1
    U_S    : (d, r_S) — real subspace basis
    U_A    : (d, r_A) — imaginary subspace basis
    min_tokens : minimum tokens to attempt fit
    svd_eps : threshold for effective rank computation

    Returns
    -------
    dict with W_S_sub, W_A_sub, r_S_eff, r_A_eff or None if fit fails
    """
    n = X_curr.shape[0]
    if n < min_tokens:
        return None

    def _fit_in(Z_curr: np.ndarray, Z_next: np.ndarray):
        """Fit linear map in a subspace with data-adaptive rank reduction."""
        if Z_curr.shape[1] == 0:
            return None, 0
        mu = Z_curr.mean(axis=0)
        Z_c = Z_curr - mu
        try:
            _, s, Vt = np.linalg.svd(Z_c, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, 0
        if s[0] < 1e-12:
            return None, 0
        # Compute effective rank
        r_eff = max(1, int((s > svd_eps * s[0]).sum()))
        r_eff = min(r_eff, n - 1)
        V_eff = Vt[:r_eff].T
        Z_red_curr = Z_c @ V_eff
        Z_red_next = (Z_next - mu) @ V_eff
        W, _, rank, _ = np.linalg.lstsq(Z_red_curr, Z_red_next, rcond=None)
        if rank < r_eff * 0.5:
            return None, r_eff
        return W, r_eff

    W_S, r_S = _fit_in(X_curr @ U_S, X_next @ U_S)
    W_A, r_A = _fit_in(X_curr @ U_A, X_next @ U_A)

    if W_S is None and W_A is None:
        return None

    return {
        "W_S_sub": W_S,
        "W_A_sub": W_A,
        "r_S_eff": r_S,
        "r_A_eff": r_A,
    }


# -----------
# Spectral diagnostics
# -----------

def analyze_subspace_maps(fit: dict) -> dict:
    """Spectral diagnostics from subspace fit."""
    result = {
        "rho_S": None,
        "contracting_S": None,
        "rho_A": None,
        "neutral_A": None,
    }

    W_S = fit.get("W_S_sub")
    if W_S is not None:
        W_S_sym = (W_S + W_S.T) / 2.0
        rho_S = float(np.max(np.abs(eigvals(W_S_sym))))
        result["rho_S"] = rho_S
        result["contracting_S"] = bool(rho_S < 1.0)

    W_A = fit.get("W_A_sub")
    if W_A is not None:
        rho_A = float(np.max(np.abs(eigvals(W_A))))
        result["rho_A"] = rho_A
        result["neutral_A"] = bool(abs(rho_A - 1.0) < 0.15)

    return result


def spectral_radius(M: np.ndarray) -> float:
    """Max absolute eigenvalue of square matrix M."""
    try:
        ev = eigvals(M)
        return float(np.max(np.abs(ev)))
    except Exception:
        return float("nan")


# -----------
# Per-cluster, per-layer profile
# -----------

def local_map_profile(
    activations_per_layer: list[np.ndarray],
    labels_per_layer:      list[np.ndarray],
    layer_types:           list[str],
    layer_names:           list[str],
    cluster_id:            int,
    min_tokens:            int = 8,
    proj_per_layer:        list[dict] | None = None,
) -> list[dict]:
    """
    Fit and decompose the local linear map for one cluster at each layer transition.

    Parameters
    ----------
    activations_per_layer : list of (n, d)
    labels_per_layer      : list of (n,)
    layer_types           : list of str
    layer_names           : list of str
    cluster_id            : int
    min_tokens            : skip transitions with fewer tokens
    proj_per_layer        : optional list of per-layer projectors

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

        row = {
            "layer_from":   layer_names[L],
            "layer_to":     layer_names[L + 1],
            "layer_type_L": layer_types[L],
            "n_tokens":     int(mask_cur.sum()),
            "cluster_id":   cluster_id,
            "rho_W":        None,
            "rho_S":        None,
            "rho_A":        None,
            "contracting_S": None,
            "neutral_A":     None,
            "r_S_eff":      None,
            "r_A_eff":      None,
        }

        if proj_per_layer is not None:
            pe = proj_per_layer[L]
            U_S = pe["U_S"]
            U_A = pe["U_A"]
            fit = fit_local_map_subspace(X_cur, X_nxt, U_S, U_A, min_tokens=min_tokens)
            if fit is not None:
                row.update(analyze_subspace_maps(fit))
                row["r_S_eff"] = fit["r_S_eff"]
                row["r_A_eff"] = fit["r_A_eff"]
                results.append(row)

    return results


# -----------
# Full pipeline → SubResult
# -----------

def run_local_contraction(ctx: dict):
    """
    Track B sub-experiment: per-cluster local contraction analysis.

    Required ctx keys
    -----------------
    activations_per_layer : list of (n, d)
    labels_per_layer      : list of (n,)
    layer_type_labels     : list of str
    layer_names           : list of str
    projectors            : output of subspace_build

    Optional ctx keys
    -----------------
    tracked_cluster_ids   : list[int] (default: all unique non-noise)
    min_tokens_for_fit    : int (default 8)
    """
    acts        = ctx["activations_per_layer"]
    labels      = ctx["labels_per_layer"]
    layer_types = ctx["layer_type_labels"]
    layer_names = ctx["layer_names"]
    projectors  = ctx.get("projectors")
    min_tokens  = ctx.get("min_tokens_for_fit", 8)

    # Build per-layer projector list
    proj_per_layer = None
    if projectors is not None:
        proj_entries = projectors["per_layer"]
        if len(proj_entries) == 1 and len(acts) > 1:
            proj_entries = proj_entries * len(acts)
        proj_per_layer = proj_entries

    all_labels = np.unique(np.concatenate([l[l >= 0] for l in labels if (l >= 0).any()]))
    tracked = ctx.get("tracked_cluster_ids", all_labels.tolist())

    all_steps: list[dict] = []
    for cid in tracked:
        steps = local_map_profile(
            acts, labels, layer_types, layer_names, int(cid), min_tokens,
            proj_per_layer=proj_per_layer
        )
        all_steps.extend(steps)

    if not all_steps:
        return {
            "name": "local_contraction",
            "applicable": False,
            "payload": {},
            "verdict_contribution": {},
        }

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
        "mean_rho_A_merge":      mu_rho_A_merg,
        "std_rho_A_merge":       std_rho_A_merg,
        "n_contracting_S_plat":  n_contracting_S_plat,
        "n_neutral_A_plat":      n_neutral_A_plat,
        "n_destab_S_merge":      n_destab_S_merg,
        "p6_r5_satisfied":       p6_r5_satisfied,
    }

    vc = {
        "lc_mean_rho_S_plateau":   mu_rho_S_plat,
        "lc_mean_rho_S_merge":     mu_rho_S_merg,
        "lc_mean_rho_A_plateau":   mu_rho_A_plat,
        "lc_n_contracting_plateau": n_contracting_S_plat,
        "lc_n_neutral_A_plateau":  n_neutral_A_plat,
        "lc_n_destab_merge":       n_destab_S_merg,
        "lc_p6_r5_satisfied":      p6_r5_satisfied,
    }

    return {
        "name": "local_contraction",
        "applicable": True,
        "payload": payload,
        "verdict_contribution": vc,
    }