"""
subspace_build.py — Global S/A residual-stream projectors for Phase 6.

Fixes applied
-------------
Bug #3  : Kernel filtering — relative tolerance eig_rel_tol * max|λ|
Bug #4  : Exclusive projectors P_S_excl, P_A_excl (orthogonal partition)
Bug #5  : Cross-head orthogonalization — U_A ⊥ U_S
Bug #6  : U_neg orthogonalization — U_neg ⊥ U_pos (fixes contraction metrics)
Bug #7  : Principal angles & overlap diagnostics
Bug #8  : _orthonormal_basis raises on empty input (no silent shape propagation)

Design
------
Real Schur decomposition of per-head OV matrices partitions d_model into:
  - 1×1 blocks with λ > cutoff  → attractive (real positive)
  - 1×1 blocks with λ < -cutoff → repulsive (real negative)
  - 1×1 blocks with |λ| ≤ cutoff → kernel (excluded from S)
  - 2×2 blocks with |λ| > cutoff → rotation (imaginary)

Union of all heads' subspaces forms the global residual-stream channels.
Orthogonalization via SVD produces ranked bases U_S, U_A.

For Bug #4, we also compute exclusive projectors P_S_excl = U_S_excl U_S_excl^T
where U_S_excl spans span(U_S) ∩ span(U_A)^⊥, giving a clean partition.
"""

import json
import numpy as np
from pathlib import Path
from scipy.linalg import schur


# ---------------------------------------------------------------------------
# Internal: Orthonormal basis (Bug #8)
# ---------------------------------------------------------------------------

def _orthonormal_basis(vecs: list[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    """
    Orthonormal basis for span of column vectors via thin SVD.

    FIX Bug #8: Raises ValueError on empty input (not silent shape propagation).
    All call sites must guard: U = _orthonormal_basis(v, tol) if v else np.zeros((d, 0))
    """
    if not vecs:
        raise ValueError(
            "_orthonormal_basis requires at least one vector. "
            "Guard the call site with: U = _orthonormal_basis(v, tol) if v else np.zeros((d, 0))"
        )
    V = np.column_stack(vecs).astype(np.float64)
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    r = int(np.sum(s > tol * s[0]))
    return U[:, :r]


# ---------------------------------------------------------------------------
# Internal: Subspace projection helpers (Bug #5, #6)
# ---------------------------------------------------------------------------

def _project_out(
    U_target: np.ndarray,
    U_remove: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    FIX Bug #5, #6: Orthonormal basis for span(U_target) ∩ span(U_remove)^⊥.

    Projects U_target onto the orthogonal complement of U_remove, then
    re-orthonormalises. Both inputs expected to have orthonormal columns.

    Used to build: U_A := U_A_raw ⊥ U_S (Bug #5)
                   U_neg := U_neg_raw ⊥ U_pos (Bug #6)
    """
    d = U_target.shape[0] if U_target.ndim == 2 else U_remove.shape[0]
    if U_target.shape[1] == 0:
        return np.zeros((d, 0))
    if U_remove.shape[1] == 0:
        return U_target

    coords = U_remove.T @ U_target  # (r_remove, r_target)
    residual = U_target - U_remove @ coords  # (d, r_target)

    cols = [residual[:, i] for i in range(residual.shape[1])]
    if not cols:
        return np.zeros((d, 0))
    return _orthonormal_basis(cols, tol)


def _principal_angles_svd(
    U_S: np.ndarray,
    U_A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FIX Bug #7: Principal-angle SVD between two orthonormal-column matrices.

    Singular values of M = U_S^T U_A are cosines of principal angles between
    the subspaces. σ_i = 1 means a shared direction; σ_i = 0 means orthogonal.

    Returns
    -------
    sigma : (k,) array — descending principal angle cosines
    L_S   : (d, k) array — left principal vectors in span(U_S)
    L_A   : (d, k) array — right principal vectors in span(U_A)
    """
    if U_S.shape[1] == 0 or U_A.shape[1] == 0:
        d = U_S.shape[0] if U_S.shape[0] > 0 else U_A.shape[0]
        return np.zeros((0,)), np.zeros((d, 0)), np.zeros((d, 0))
    M = U_S.T @ U_A
    X, sigma, Yt = np.linalg.svd(M, full_matrices=False)
    return sigma, U_S @ X, U_A @ Yt.T


def _orthogonalize_against(
    U: np.ndarray,
    V: np.ndarray,
    svd_tol: float,
) -> np.ndarray:
    """
    FIX Bug #4: Orthonormal basis for span(U) ∩ span(V)^⊥.

    Projects U onto orthogonal complement of V, drops residuals below svd_tol,
    then SVD-orthonormalises.  Used to build U_S_excl, U_A_excl.
    """
    if U.shape[1] == 0:
        return U.copy()
    if V.shape[1] == 0:
        return U.copy()
    U_perp = U - V @ (V.T @ U)
    cols = [
        U_perp[:, k]
        for k in range(U_perp.shape[1])
        if np.linalg.norm(U_perp[:, k]) > svd_tol
    ]
    if not cols:
        return np.zeros((U.shape[0], 0))
    return _orthonormal_basis(cols, svd_tol)


# ---------------------------------------------------------------------------
# Schur extraction with kernel filtering (Bug #3)
# ---------------------------------------------------------------------------

def _extract_schur_subspaces(
    OV: np.ndarray,
    eig_rel_tol: float = 1e-8,
) -> dict:
    """
    FIX Bug #3: Real Schur with kernel filtering using relative tolerance.

    OV = W_V @ W_O has rank ≤ d_head, so carries d_model - d_head zero
    eigenvalues. Previous code routed λ=0 to real_pos, saturating U_pos.
    Now: filter using eig_rel_tol * max_eigenvalue_magnitude.

    For 1×1 blocks: magnitude is |λ|.
    For 2×2 blocks: magnitude is sqrt(det) = |λ|_1 = |λ|_2 for conj pair.

    Returns dict with real_pos_vecs, real_neg_vecs, rot_vecs (all surviving),
    plus diagnostics: n_kernel_real, n_kernel_rot, max_eig_mag.
    """
    d = OV.shape[0]
    T, Z = schur(OV, output="real")

    # Pass 1: parse structure and gather magnitudes
    blocks: list[tuple[str, int, float]] = []  # (kind, start_idx, signed_val_or_mag)
    eig_mags: list[float] = []

    i = 0
    while i < d:
        if i < d - 1 and abs(T[i + 1, i]) > 1e-10:
            # 2×2 block
            a = float(T[i, i])
            b = float(T[i, i + 1])
            c = float(T[i + 1, i])
            dd = float(T[i + 1, i + 1])
            det = a * dd - b * c
            mag = float(np.sqrt(max(det, 0.0)))
            blocks.append(("rot", i, mag))
            eig_mags.append(mag)
            i += 2
        else:
            val = float(T[i, i])
            blocks.append(("real", i, val))
            eig_mags.append(abs(val))
            i += 1

    max_mag = max(eig_mags) if eig_mags else 0.0
    cutoff = eig_rel_tol * max_mag

    # Pass 2: bucket survivors and count drops
    real_pos: list[np.ndarray] = []
    real_neg: list[np.ndarray] = []
    rot: list[np.ndarray] = []
    n_kernel_real = 0
    n_kernel_rot = 0

    for kind, start, info in blocks:
        if kind == "rot":
            mag = info
            if mag <= cutoff:
                n_kernel_rot += 1
                continue
            rot.append(Z[:, start].copy())
            rot.append(Z[:, start + 1].copy())
        else:  # "real"
            val = info
            if abs(val) <= cutoff:
                n_kernel_real += 1
                continue
            if val > 0:
                real_pos.append(Z[:, start].copy())
            else:
                real_neg.append(Z[:, start].copy())

    return {
        "real_pos_vecs": real_pos,
        "real_neg_vecs": real_neg,
        "rot_vecs": rot,
        "n_kernel_real": n_kernel_real,
        "n_kernel_rot": n_kernel_rot,
        "max_eig_mag": max_mag,
    }


# ---------------------------------------------------------------------------
# Layer aggregation (Bug #5, #6, #7)
# ---------------------------------------------------------------------------

def _build_for_layer(
    head_ovs: list,
    d: int,
    svd_tol: float,
    eig_rel_tol: float = 1e-8,
    cos_overlap_tol: float = 0.99,
) -> dict:
    """
    FIX Bugs #3–#7: Aggregate Schur subspaces across all heads for one layer.

    Pipeline:
      1. Per-head Schur extraction → real_pos / real_neg / rot vectors
      2. Orthonormalise each union → U_pos_raw, U_neg_raw, U_A_raw
      3. Compute principal angles (U_S vs U_A)
      4. Resolve cross-head disagreement: U_A := U_A ⊥ U_S (Bug #5)
      5. Resolve U_pos / U_neg overlap: U_neg := U_neg ⊥ U_pos (Bug #6)
      6. Build P_S, P_A and exclusive variants P_S_excl, P_A_excl (Bug #4)

    Returns dict with standard projectors + diagnostics.
    """
    all_real_pos, all_real_neg, all_rot = [], [], []
    total_kernel_real = 0
    total_kernel_rot = 0
    max_eig_mag = 0.0

    for OV in head_ovs:
        sub = _extract_schur_subspaces(OV, eig_rel_tol=eig_rel_tol)
        all_real_pos.extend(sub["real_pos_vecs"])
        all_real_neg.extend(sub["real_neg_vecs"])
        all_rot.extend(sub["rot_vecs"])
        total_kernel_real += sub["n_kernel_real"]
        total_kernel_rot += sub["n_kernel_rot"]
        max_eig_mag = max(max_eig_mag, sub["max_eig_mag"])

    # Step 2: orthonormalise (guarded against empty lists)
    U_pos_raw = (
        _orthonormal_basis(all_real_pos, svd_tol)
        if all_real_pos
        else np.zeros((d, 0))
    )
    U_neg_raw = (
        _orthonormal_basis(all_real_neg, svd_tol)
        if all_real_neg
        else np.zeros((d, 0))
    )
    U_A_raw = _orthonormal_basis(all_rot, svd_tol) if all_rot else np.zeros((d, 0))

    # Pre-resolution U_S for diagnostic
    real_vecs = (
        list(U_pos_raw.T) + list(U_neg_raw.T)
        if (U_pos_raw.shape[1] + U_neg_raw.shape[1]) > 0
        else []
    )
    U_S_raw = _orthonormal_basis(real_vecs, svd_tol) if real_vecs else np.zeros((d, 0))

    # Step 3: principal angles diagnostic
    sigma, L_S, L_A = _principal_angles_svd(U_S_raw, U_A_raw)
    dim_overlap = int(np.sum(sigma > cos_overlap_tol)) if sigma.size > 0 else 0

    # Step 4: A := A ⊥ S (S wins; A is residual)
    U_A = _project_out(U_A_raw, U_S_raw, svd_tol)

    # Step 5: U_neg := U_neg ⊥ U_pos, then U_S = [U_pos | U_neg]
    U_pos = U_pos_raw
    U_neg = _project_out(U_neg_raw, U_pos, svd_tol)

    if U_pos.shape[1] + U_neg.shape[1] > 0:
        U_S = np.column_stack([U_pos, U_neg])
    else:
        U_S = np.zeros((d, 0))

    # Step 6: projectors and exclusive variants
    P_S = U_S @ U_S.T if U_S.shape[1] > 0 else np.zeros((d, d))
    P_A = U_A @ U_A.T if U_A.shape[1] > 0 else np.zeros((d, d))

    U_S_excl = _orthogonalize_against(U_S, U_A, svd_tol)
    U_A_excl = _orthogonalize_against(U_A, U_S, svd_tol)

    P_S_excl = (
        U_S_excl @ U_S_excl.T if U_S_excl.shape[1] > 0 else np.zeros((d, d))
    )
    P_A_excl = (
        U_A_excl @ U_A_excl.T if U_A_excl.shape[1] > 0 else np.zeros((d, d))
    )

    # Orthogonality metric
    if U_S.shape[1] > 0 and U_A.shape[1] > 0:
        PS_PA_fro = float(np.linalg.norm(P_S @ P_A, ord="fro"))
    else:
        PS_PA_fro = 0.0

    dim_S = U_S.shape[1]
    dim_A = U_A.shape[1]

    return {
        # Original contract
        "P_S": P_S,
        "P_A": P_A,
        "U_pos": U_pos,
        "U_neg": U_neg,
        "U_S": U_S,
        "U_A": U_A,
        "dim_S": dim_S,
        "dim_A": dim_A,
        "frac_S": float(dim_S / d),
        "frac_A": float(dim_A / d),
        # Bug #3 diagnostics
        "n_kernel_real_dropped": total_kernel_real,
        "n_kernel_rot_dropped": total_kernel_rot,
        "max_eig_mag": float(max_eig_mag),
        "eig_rel_tol": float(eig_rel_tol),
        # Bug #4: exclusive projectors
        "P_S_excl": P_S_excl,
        "P_A_excl": P_A_excl,
        "U_S_excl": U_S_excl,
        "U_A_excl": U_A_excl,
        "dim_S_excl": U_S_excl.shape[1],
        "dim_A_excl": U_A_excl.shape[1],
        # Bug #7: principal angles
        "principal_angles": sigma.tolist() if sigma.size > 0 else [],
        "L_S_principal": L_S,
        "L_A_principal": L_A,
        "dim_overlap": dim_overlap,
        "frac_overlap": float(dim_overlap / d),
        "PS_PA_fro_norm": PS_PA_fro,
        "cos_overlap_tol": float(cos_overlap_tol),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_global_projectors(
    ov_data: dict,
    svd_tol: float = 1e-8,
    eig_rel_tol: float = 1e-8,
    cos_overlap_tol: float = 0.99,
) -> dict:
    """
    Build global residual-stream S/A projectors from per-head OV matrices.

    FIX Bugs #3–#7: Thread eig_rel_tol, cos_overlap_tol through to _build_for_layer.

    Parameters
    ----------
    ov_data         : output of weights.extract_ov_circuit.
                      Keys: is_per_layer, ov_per_head, n_heads, d_model, layer_names
    svd_tol         : SVD truncation tolerance for orthonormalisation
    eig_rel_tol     : Relative tolerance for eigenvalue kernel filtering
                      (Bug #3). Eigenvalues with |λ| ≤ eig_rel_tol * max|λ| dropped.
    cos_overlap_tol : Principal angle cutoff for dim_overlap diagnostic (Bug #7)

    Returns
    -------
    dict with is_per_layer, layer_names, d_model, per_layer (list of projector dicts).
    Each projector dict includes P_S, P_A, U_S, U_A, plus diagnostics and
    exclusive projectors (Bugs #3, #4, #7).
    """
    is_per_layer = ov_data["is_per_layer"]
    d = ov_data["d_model"]
    layer_names = ov_data["layer_names"]

    results = []

    if is_per_layer:
        for layer_idx, lname in enumerate(layer_names):
            head_ovs = ov_data["ov_per_head"][layer_idx]
            entry = _build_for_layer(
                head_ovs, d, svd_tol, eig_rel_tol, cos_overlap_tol
            )
            results.append(entry)
    else:
        head_ovs = ov_data["ov_per_head"]
        entry = _build_for_layer(head_ovs, d, svd_tol, eig_rel_tol, cos_overlap_tol)
        results.append(entry)

    return {
        "is_per_layer": is_per_layer,
        "layer_names": layer_names,
        "d_model": d,
        "per_layer": results,
    }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_projectors(proj: dict, path: Path) -> None:
    """
    Persist projector dict to .npz + .json sidecar.

    Stores new diagnostics (Bug #3, #4, #7) in JSON for later inspection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    meta = {
        "is_per_layer": proj["is_per_layer"],
        "layer_names": proj["layer_names"],
        "d_model": proj["d_model"],
        "per_layer_meta": [],
    }

    for idx, (lname, entry) in enumerate(
        zip(proj["layer_names"], proj["per_layer"])
    ):
        prefix = f"layer{idx}_"
        for key in ("P_S", "P_A", "U_S", "U_A", "U_pos", "U_neg", "P_S_excl",
                    "P_A_excl", "U_S_excl", "U_A_excl", "L_S_principal", "L_A_principal"):
            if key in entry and entry[key] is not None:
                arr = entry[key].astype(np.float32)
                arrays[prefix + key] = arr

        meta["per_layer_meta"].append({
            "layer_name": lname,
            "dim_S": entry["dim_S"],
            "dim_A": entry["dim_A"],
            "frac_S": entry["frac_S"],
            "frac_A": entry["frac_A"],
            # Bug #3 diagnostics
            "n_kernel_real_dropped": entry.get("n_kernel_real_dropped", 0),
            "n_kernel_rot_dropped": entry.get("n_kernel_rot_dropped", 0),
            "max_eig_mag": entry.get("max_eig_mag", 0.0),
            "eig_rel_tol": entry.get("eig_rel_tol", 1e-8),
            # Bug #4: exclusive projectors
            "dim_S_excl": entry.get("dim_S_excl", 0),
            "dim_A_excl": entry.get("dim_A_excl", 0),
            "PS_PA_fro_norm": entry.get("PS_PA_fro_norm", 0.0),
            # Bug #7: principal angles
            "principal_angles": entry.get("principal_angles", []),
            "dim_overlap": entry.get("dim_overlap", 0),
            "frac_overlap": entry.get("frac_overlap", 0.0),
            "cos_overlap_tol": entry.get("cos_overlap_tol", 0.99),
        })

    np.savez(path, **arrays)

    json_path = path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_projectors(path: Path) -> dict:
    """Load projector dict from .npz + .json sidecar written by save_projectors."""
    path = Path(path)
    json_path = path.with_suffix(".json")

    with open(json_path) as f:
        meta = json.load(f)

    npz_path = str(path) if path.suffix == ".npz" else str(path) + ".npz"
    data = np.load(npz_path)

    per_layer = []
    for idx, lmeta in enumerate(meta["per_layer_meta"]):
        prefix = f"layer{idx}_"
        entry = {
            "P_S": data[prefix + "P_S"].astype(np.float64),
            "P_A": data[prefix + "P_A"].astype(np.float64),
            "U_S": data[prefix + "U_S"].astype(np.float64),
            "U_A": data[prefix + "U_A"].astype(np.float64),
            "U_pos": data[prefix + "U_pos"].astype(np.float64),
            "U_neg": data[prefix + "U_neg"].astype(np.float64),
            "dim_S": lmeta["dim_S"],
            "dim_A": lmeta["dim_A"],
            "frac_S": lmeta["frac_S"],
            "frac_A": lmeta["frac_A"],
            # Bug #3
            "n_kernel_real_dropped": lmeta.get("n_kernel_real_dropped", 0),
            "n_kernel_rot_dropped": lmeta.get("n_kernel_rot_dropped", 0),
            "max_eig_mag": lmeta.get("max_eig_mag", 0.0),
            "eig_rel_tol": lmeta.get("eig_rel_tol", 1e-8),
            # Bug #4
            "P_S_excl": (
                data[prefix + "P_S_excl"].astype(np.float64)
                if prefix + "P_S_excl" in data
                else np.zeros((lmeta.get("d_model", 768), lmeta.get("d_model", 768)))
            ),
            "P_A_excl": (
                data[prefix + "P_A_excl"].astype(np.float64)
                if prefix + "P_A_excl" in data
                else np.zeros((lmeta.get("d_model", 768), lmeta.get("d_model", 768)))
            ),
            "U_S_excl": (
                data[prefix + "U_S_excl"].astype(np.float64)
                if prefix + "U_S_excl" in data
                else np.zeros((lmeta.get("d_model", 768), 0))
            ),
            "U_A_excl": (
                data[prefix + "U_A_excl"].astype(np.float64)
                if prefix + "U_A_excl" in data
                else np.zeros((lmeta.get("d_model", 768), 0))
            ),
            "dim_S_excl": lmeta.get("dim_S_excl", 0),
            "dim_A_excl": lmeta.get("dim_A_excl", 0),
            "PS_PA_fro_norm": lmeta.get("PS_PA_fro_norm", 0.0),
            # Bug #7
            "principal_angles": lmeta.get("principal_angles", []),
            "L_S_principal": (
                data[prefix + "L_S_principal"].astype(np.float64)
                if prefix + "L_S_principal" in data
                else np.zeros((lmeta.get("d_model", 768), 0))
            ),
            "L_A_principal": (
                data[prefix + "L_A_principal"].astype(np.float64)
                if prefix + "L_A_principal" in data
                else np.zeros((lmeta.get("d_model", 768), 0))
            ),
            "dim_overlap": lmeta.get("dim_overlap", 0),
            "frac_overlap": lmeta.get("frac_overlap", 0.0),
            "cos_overlap_tol": lmeta.get("cos_overlap_tol", 0.99),
        }
        per_layer.append(entry)

    return {
        "is_per_layer": meta["is_per_layer"],
        "layer_names": meta["layer_names"],
        "d_model": meta["d_model"],
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def projector_summary(proj: dict) -> list[dict]:
    """
    Return per-layer scalar diagnostics including Bug #3, #4, #7 metrics.
    """
    d = proj["d_model"]
    rows = []
    for lname, entry in zip(proj["layer_names"], proj["per_layer"]):
        rows.append({
            "layer_name": lname,
            "dim_S": entry["dim_S"],
            "dim_A": entry["dim_A"],
            "dim_S_excl": entry.get("dim_S_excl", 0),
            "dim_A_excl": entry.get("dim_A_excl", 0),
            "frac_S": entry["frac_S"],
            "frac_A": entry["frac_A"],
            "frac_S_excl": float(entry.get("dim_S_excl", 0) / d),
            "frac_A_excl": float(entry.get("dim_A_excl", 0) / d),
            "n_kernel_real": entry.get("n_kernel_real_dropped", 0),
            "n_kernel_rot": entry.get("n_kernel_rot_dropped", 0),
            "max_eig_mag": entry.get("max_eig_mag", 0.0),
            "fro_norm_PS_PA": entry.get("PS_PA_fro_norm", 0.0),
            "dim_overlap": entry.get("dim_overlap", 0),
            "frac_overlap": entry.get("frac_overlap", 0.0),
        })
    return rows


def print_projector_summary(proj: dict) -> None:
    """Print projector diagnostics table."""
    rows = projector_summary(proj)
    header = (
        f"{'layer':<20} {'dim_S':>6} {'dim_A':>6} {'dim_S_exc':>8} "
        f"{'n_kernel':>9} {'fro':>8} {'overlap':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        n_kernel = r["n_kernel_real"] + r["n_kernel_rot"]
        print(
            f"{r['layer_name']:<20} "
            f"{r['dim_S']:>6d} "
            f"{r['dim_A']:>6d} "
            f"{r['dim_S_excl']:>8d} "
            f"{n_kernel:>9d} "
            f"{r['fro_norm_PS_PA']:>8.4f} "
            f"{r['frac_overlap']:>8.4f}"
        )