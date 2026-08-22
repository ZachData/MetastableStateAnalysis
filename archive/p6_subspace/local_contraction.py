"""
local_contraction.py — Track B: Per-cluster local linear maps.

During a plateau, each cluster is near an attractor of S. Locally, the
dynamics should be contracting in the real subspace (S eigenvalues < 1)
and neutrally rotating in the imaginary subspace (A eigenvalues on unit circle).

At merge events, S's local contraction should destabilise: at least one
S-eigenvalue near or outside the unit disk.

Falsifiable predictions tested
-------------------------------
P6-R5 : Plateau layers: W_C^S has spectral radius < 1 (local contraction).
         Merge layers:   W_C^S has spectral radius ≥ 1 (attractor destabilising).
         Throughout:     W_C^A has spectral radius ≈ 1 (rotation, norm-preserving).

Implementation note — subspace-aware fitting (Bug fix)
------------------------------------------------------
The model dimension d (768–2048) is far larger than tokens per cluster n (5–50).
Fitting a (d×d) local map from n token pairs is massively underdetermined:
the minimum-norm lstsq solution has near-zero singular values by construction,
making rho_S trivially small (P6-R5 contraction always passes) and rho_A
trivially near zero (neutral_A always fails). Neither diagnostic is informative.

Fix: project tokens into the S and A subspaces first, then reduce to the
data-adaptive effective rank via SVD before fitting. The resulting maps are
(r_S × r_S) and (r_A × r_A) with r_S, r_A ≤ min(n−1, dim_S/A). Because r_eff
is bounded by n−1, the system is always well-determined, and the spectral
diagnostics reflect actual subspace dynamics rather than min-norm artefacts.

Functions
---------
fit_local_map              : legacy ambient-space fit (kept for unit tests)
fit_local_map_subspace     : subspace-aware fit — primary production path
analyze_subspace_maps      : spectral diagnostics from a subspace fit result
decompose_local_map        : legacy symmetric decomposition (kept for unit tests)
spectral_radius            : max |eigenvalue| of square matrix
local_map_profile          : all-layers profile for one cluster
run_local_contraction      : full pipeline → SubResult
"""

import numpy as np
from scipy.linalg import eigvals

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Legacy ambient-space fit — kept for backward compatibility / unit tests
# ---------------------------------------------------------------------------

def fit_local_map(
    X_curr: np.ndarray,
    X_next: np.ndarray,
) -> np.ndarray | None:
    """
    Fit a linear map W such that X_next ≈ X_curr @ W  (least squares).

    Returns (d, d) or None.

    NOTE: Retained for unit tests only.  For d >> n this is underdetermined;
    use fit_local_map_subspace for production runs.
    """
    n, d = X_curr.shape
    if n < 4:
        return None

    W, _, rank, _ = np.linalg.lstsq(X_curr, X_next, rcond=None)

    if rank < min(n, d) * 0.5:
        return None

    # FIX (Bug 6): removed incorrect conditional transpose; lstsq always returns (d, d).
    return W


# ---------------------------------------------------------------------------
# Subspace-aware fit — primary production path
# ---------------------------------------------------------------------------

def fit_local_map_subspace(
    X_curr:     np.ndarray,
    X_next:     np.ndarray,
    U_S:        np.ndarray,
    U_A:        np.ndarray,
    min_tokens: int   = 8,
    svd_eps:    float = 1e-3,
) -> dict | None:
    """
    Fit local linear maps within the S and A subspaces separately.

    Rather than fitting a (d×d) ambient map, projects X_curr / X_next into
    each subspace, reduces to the data-adaptive effective rank via SVD, then
    solves the well-determined system:

        Z_S_next ≈ Z_S_curr @ W_S_sub      W_S_sub ∈ ℝ^{r_S × r_S}
        Z_A_next ≈ Z_A_curr @ W_A_sub      W_A_sub ∈ ℝ^{r_A × r_A}

    r_S (r_A) = number of singular values of Z_S_curr (Z_A_curr) exceeding
    svd_eps × s_max, bounded above by min(n−1, dim_S/A).  The n−1 cap
    guarantees the reduced system is always determined.

    Parameters
    ----------
    X_curr     : (n, d) token activations at layer L
    X_next     : (n, d) token activations at layer L+1, same token positions
    U_S        : (d, dim_S) full real-subspace basis  (span of U_pos ∪ U_neg)
    U_A        : (d, dim_A) imaginary-subspace basis
    min_tokens : return None if n < min_tokens
    svd_eps    : singular-value threshold relative to s_max for effective rank

    Returns
    -------
    dict with keys  W_S_sub, W_A_sub, r_S_eff, r_A_eff
    None if n < min_tokens or both subspace fits fail.
    """
    n = X_curr.shape[0]
    if n < min_tokens:
        return None

    def _fit_in(Z_curr: np.ndarray, Z_next: np.ndarray):
        """
        Mean-centre Z_curr, SVD-reduce to effective rank, fit linear map.
        Both inputs are projected into the same effective basis.
        Returns (W_red, r_eff) or (None, 0).
        """
        if Z_curr.shape[1] == 0:
            return None, 0

        mu   = Z_curr.mean(axis=0)
        Z_c  = Z_curr - mu

        try:
            _, s, Vt = np.linalg.svd(Z_c, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, 0

        if s[0] < 1e-12:        # all tokens identical in this subspace
            return None, 0

        r_eff = max(1, int((s > svd_eps * s[0]).sum()))
        r_eff = min(r_eff, n - 1)   # cap: r_eff < n keeps system determined

        V_eff      = Vt[:r_eff].T                   # (dim, r_eff)
        Z_red_curr = Z_c @ V_eff                    # (n, r_eff)
        Z_red_next = (Z_next - mu) @ V_eff          # project next into same basis

        W, _, rank, _ = np.linalg.lstsq(Z_red_curr, Z_red_next, rcond=None)

        if rank < r_eff * 0.5:
            return None, r_eff

        return W, r_eff     # W is (r_eff, r_eff)

    W_S, r_S = _fit_in(X_curr @ U_S, X_next @ U_S)
    W_A, r_A = _fit_in(X_curr @ U_A, X_next @ U_A)

    if W_S is None and W_A is None:
        return None

    return {
        "W_S_sub": W_S,   # (r_S, r_S) or None
        "W_A_sub": W_A,   # (r_A, r_A) or None
        "r_S_eff": r_S,
        "r_A_eff": r_A,
    }


def analyze_subspace_maps(fit: dict) -> dict:
    """
    Derive spectral diagnostics from the output of fit_local_map_subspace.

    S-subspace map (W_S_sub):
      Decompose into symmetric + antisymmetric parts.
      rho_S = spectral radius of the *symmetric* part (contracting / expanding).
      Prediction P6-R5a: rho_S < 1 at plateau, rho_S ≥ 1 at merge.

    A-subspace map (W_A_sub):
      rho_A = spectral radius of the *full* map (should ≈ 1 for rotation).
      Prediction P6-R5b: rho_A ≈ 1 throughout, independent of layer type.

    Returns
    -------
    dict with rho_S, rho_A, contracting_S, neutral_A (any may be None if
    the corresponding subspace fit was unavailable).
    """
    result: dict = {
        "rho_S":         None,
        "contracting_S": None,
        "rho_A":         None,
        "neutral_A":     None,
    }

    W_S = fit.get("W_S_sub")
    if W_S is not None:
        W_S_sym = (W_S + W_S.T) / 2.0
        rho_S   = spectral_radius(W_S_sym)
        result["rho_S"]         = rho_S
        result["contracting_S"] = bool(rho_S < 1.0)

    W_A = fit.get("W_A_sub")
    if W_A is not None:
        rho_A = spectral_radius(W_A)
        result["rho_A"]    = rho_A
        result["neutral_A"] = bool(abs(rho_A - 1.0) < 0.15)

    return result


# ---------------------------------------------------------------------------
# Legacy decomposition — kept for backward compatibility / unit tests
# ---------------------------------------------------------------------------

def decompose_local_map(W: np.ndarray) -> dict:
    """
    Decompose ambient-space map W into symmetric S and antisymmetric A parts.

    NOTE: Retained for unit tests that call this directly.
    Production path uses analyze_subspace_maps.
    """
    W_S = (W + W.T) / 2.0
    W_A = (W - W.T) / 2.0

    rho_W = spectral_radius(W)
    rho_S = spectral_radius(W_S)
    rho_A = spectral_radius(W_A)

    return {
        "W_S":           W_S,
        "W_A":           W_A,
        "rho_W":         rho_W,
        "rho_S":         rho_S,
        "rho_A":         rho_A,
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
    min_tokens:            int = 8,
    proj_per_layer:        list[dict] | None = None,
) -> list[dict]:
    """
    Fit and analyze the local linear map for one cluster at each layer transition.

    Parameters
    ----------
    proj_per_layer : list of subspace_build projector dicts, one per layer,
                     each containing at minimum "U_S" and "U_A".
                     When provided → fit_local_map_subspace (recommended).
                     When None    → legacy fit_local_map in ambient space.
    min_tokens     : skip transitions with fewer cluster tokens.
                     Default 8 (raised from 4 to improve fit reliability).
    """
    results  = []
    n_layers = len(activations_per_layer)

    for L in range(n_layers - 1):
        mask_cur = labels_per_layer[L] == cluster_id

        if mask_cur.sum() < min_tokens:
            continue

        token_indices = np.where(mask_cur)[0]
        X_cur = activations_per_layer[L][mask_cur]
        X_nxt = activations_per_layer[L + 1][token_indices]

        row: dict = {
            "layer_from":    layer_names[L],
            "layer_to":      layer_names[L + 1],
            "layer_type_L":  layer_types[L],
            "n_tokens":      int(mask_cur.sum()),
            "cluster_id":    cluster_id,
            "rho_S":         None,
            "rho_A":         None,
            "contracting_S": None,
            "neutral_A":     None,
            "r_S_eff":       None,
            "r_A_eff":       None,
        }

        if proj_per_layer is not None:
            pe  = proj_per_layer[L]
            U_S = pe["U_S"]   # full real basis (U_pos ∪ U_neg)
            U_A = pe["U_A"]
            fit = fit_local_map_subspace(X_cur, X_nxt, U_S, U_A, min_tokens=min_tokens)
            if fit is not None:
                row.update(analyze_subspace_maps(fit))
                row["r_S_eff"] = fit["r_S_eff"]
                row["r_A_eff"] = fit["r_A_eff"]
                results.append(row)
        else:
            # Legacy path — ambient-space fit (underdetermined when n << d)
            W = fit_local_map(X_cur, X_nxt)
            if W is not None:
                decomp = decompose_local_map(W)
                row["rho_S"]         = decomp["rho_S"]
                row["rho_A"]         = decomp["rho_A"]
                row["contracting_S"] = decomp["contracting_S"]
                row["neutral_A"]     = decomp["neutral_A"]
                results.append(row)

    return results


# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_local_contraction(ctx: dict) -> SubResult:
    """
    Track B sub-experiment: per-cluster local contraction analysis.

    Sources projectors from ctx and passes them to local_map_profile for
    subspace-aware fitting.  Falls back to legacy ambient path if projectors
    are absent, and flags this prominently in the summary.

    Required ctx keys
    -----------------
    activations_per_layer : list of (n, d)
    labels_per_layer      : list of (n,)
    layer_type_labels     : list of str
    layer_names           : list of str

    Optional ctx keys
    -----------------
    projectors            : output of subspace_build  [strongly recommended]
    tracked_cluster_ids   : list[int] (default: all unique non-noise)
    min_tokens_for_fit    : int (default 8)
    """
    acts        = ctx["activations_per_layer"]
    labels      = ctx["labels_per_layer"]
    layer_types = ctx["layer_type_labels"]
    layer_names = ctx["layer_names"]
    min_tokens  = ctx.get("min_tokens_for_fit", 8)

    # Build per-layer projector list
    projectors         = ctx.get("projectors")
    proj_per_layer     = None
    using_subspace_fit = False

    if projectors is not None:
        proj_entries = projectors["per_layer"]
        if len(proj_entries) == 1 and len(acts) > 1:
            proj_entries = proj_entries * len(acts)
        proj_per_layer     = proj_entries
        using_subspace_fit = True

    all_labels = np.unique(np.concatenate([l[l >= 0] for l in labels if (l >= 0).any()]))
    tracked    = ctx.get("tracked_cluster_ids", all_labels.tolist())

    all_steps: list[dict] = []
    for cid in tracked:
        steps = local_map_profile(
            acts, labels, layer_types, layer_names, int(cid),
            min_tokens=min_tokens,
            proj_per_layer=proj_per_layer,
        )
        all_steps.extend(steps)

    if not all_steps:
        return SubResult(
            name="local_contraction",
            applicable=False,
            payload={},
            summary_lines=[
                "local_contraction: no valid fits",
                f"  min_tokens={min_tokens}  using_subspace_fit={using_subspace_fit}",
            ],
            verdict_contribution={},
        )

    def _agg(steps, ltype, key):
        vals = [s[key] for s in steps if s["layer_type_L"] == ltype and s[key] is not None]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)

    plateau_steps = [s for s in all_steps if s["layer_type_L"] == "plateau"]
    merge_steps   = [s for s in all_steps if s["layer_type_L"] == "merge"]

    mu_rho_S_plat, std_rho_S_plat, n_plat = _agg(all_steps, "plateau", "rho_S")
    mu_rho_S_merg, std_rho_S_merg, n_merg = _agg(all_steps, "merge",   "rho_S")
    mu_rho_A_plat, std_rho_A_plat, _      = _agg(all_steps, "plateau", "rho_A")
    mu_rho_A_merg, std_rho_A_merg, _      = _agg(all_steps, "merge",   "rho_A")

    n_contracting_S_plat = sum(1 for s in plateau_steps if s["contracting_S"] is True)
    n_neutral_A_plat     = sum(1 for s in plateau_steps if s["neutral_A"]     is True)
    n_destab_S_merg      = sum(1 for s in merge_steps
                               if s["contracting_S"] is False and s["rho_S"] is not None)

    p6_r5_contraction = (n_contracting_S_plat > n_plat // 2) if n_plat > 0 else None
    p6_r5_neutral_A   = (n_neutral_A_plat     > n_plat // 2) if n_plat > 0 else None
    p6_r5_destab      = (n_destab_S_merg      > n_merg // 2) if n_merg > 0 else None
    p6_r5_satisfied   = bool(p6_r5_contraction and p6_r5_destab)

    r_S_vals = [s["r_S_eff"] for s in all_steps if s["r_S_eff"] is not None]
    r_A_vals = [s["r_A_eff"] for s in all_steps if s["r_A_eff"] is not None]
    mean_r_S = float(np.mean(r_S_vals)) if r_S_vals else None
    mean_r_A = float(np.mean(r_A_vals)) if r_A_vals else None

    payload = {
        "using_subspace_fit":    using_subspace_fit,
        "mean_r_S_eff":          mean_r_S,
        "mean_r_A_eff":          mean_r_A,
        "n_transitions":         len(all_steps),
        "n_plateau":             n_plat,
        "n_merge":               n_merg,
        "mu_rho_S_plateau":      mu_rho_S_plat,
        "std_rho_S_plateau":     std_rho_S_plat,
        "mu_rho_S_merge":        mu_rho_S_merg,
        "std_rho_S_merge":       std_rho_S_merg,
        "mu_rho_A_plateau":      mu_rho_A_plat,
        "std_rho_A_plateau":     std_rho_A_plat,
        "mu_rho_A_merge":        mu_rho_A_merg,
        "std_rho_A_merge":       std_rho_A_merg,
        "n_contracting_S_plat":  n_contracting_S_plat,
        "n_neutral_A_plat":      n_neutral_A_plat,
        "n_destab_S_merge":      n_destab_S_merg,
        "p6_r5_satisfied":       p6_r5_satisfied,
    }

    fit_mode = (
        "subspace-aware (U_S / U_A projected)"
        if using_subspace_fit
        else "WARN: legacy ambient-space — projectors missing from ctx"
    )
    lines = [
        SEP_THICK,
        "LOCAL CONTRACTION ANALYSIS  [Track B]",
        SEP_THICK,
        f"Fit mode: {fit_mode}",
    ]
    if using_subspace_fit:
        lines += [
            _bullet("mean effective rank used in S", mean_r_S),
            _bullet("mean effective rank used in A", mean_r_A),
        ]
    lines += [
        f"Total transition fits:       {len(all_steps)}",
        f"  plateau transitions:       {n_plat}",
        f"  merge transitions:         {n_merg}",
        "",
        "Spectral radius of symmetric part of S-subspace map (W_S_sub_sym):",
        _bullet("mean ρ_S at plateau layers", mu_rho_S_plat),
        _bullet("std  ρ_S at plateau layers", std_rho_S_plat),
        _bullet("mean ρ_S at merge layers",   mu_rho_S_merg),
        _bullet("std  ρ_S at merge layers",   std_rho_S_merg),
        "",
        "Spectral radius of full A-subspace map W_A_sub (should be ≈ 1.0):",
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
        "lc_mean_rho_S_plateau":    mu_rho_S_plat,
        "lc_mean_rho_S_merge":      mu_rho_S_merg,
        "lc_mean_rho_A_plateau":    mu_rho_A_plat,
        "lc_n_contracting_plateau": n_contracting_S_plat,
        "lc_n_neutral_A_plateau":   n_neutral_A_plat,
        "lc_n_destab_merge":        n_destab_S_merg,
        "lc_p6_r5_satisfied":       p6_r5_satisfied,
    }

    return SubResult(
        name="local_contraction",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )