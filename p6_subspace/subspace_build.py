"""
subspace_build.py — Global S/A residual-stream projectors.

Fixes applied
-------------
#3  Kernel filtering: relative tolerance eig_rel_tol × ‖T‖_F per head.
    λ = 0 modes routed to real_zero bucket and excluded from U_S.
#4  Exclusive projectors P_S_excl / P_A_excl (S∖A, A∖S orthogonal partition).
#5  Cross-head orthogonalisation: U_A := U_A ∩ span(U_S)^⊥.
#6  U_neg orthogonalisation: U_neg := U_neg ∩ span(U_pos)^⊥.
#7  Principal angles and overlap diagnostics stored per layer.
#8  _orthonormal_basis raises on empty input; all call sites guarded.

Design
------
Real Schur decomposition of per-head OV = W_V @ W_O partitions d_model into:

  1×1 block, λ >  eig_tol  → attractive (real_pos)   → U_pos → U_S
  1×1 block, λ < -eig_tol  → repulsive  (real_neg)   → U_neg → U_S
  1×1 block, |λ| ≤ eig_tol → kernel    (real_zero)  → excluded from S
  2×2 block                 → rotation               → U_A

Both eig_tol and block_tol are relative to ‖T‖_F (not absolute), which
handles OV matrices whose operator norm differs substantially from 1.

After per-head extraction, cross-head unions are orthonormalised.
Resolution order: S wins over A (#5); U_neg cleaned of U_pos overlap (#6).
Exclusive projectors expose the clean S∖A and A∖S partition (#4).
"""

import json
import numpy as np
from pathlib import Path
from scipy.linalg import schur


# ---------------------------------------------------------------------------
# Orthonormal-basis helper  (fix #8)
# ---------------------------------------------------------------------------

def _orthonormal_basis(vecs: list[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    """
    Orthonormal basis for span(vecs) via thin SVD.

    Fix #8: raises ValueError on empty input so missing-data bugs surface at
    the call site.  All internal callers are guarded:
        U = _orthonormal_basis(v, tol) if v else np.zeros((d, 0))
    """
    if not vecs:
        raise ValueError(
            "_orthonormal_basis requires at least one vector.  "
            "Guard with: U = _orthonormal_basis(v, tol) if v else np.zeros((d, 0))"
        )
    V = np.column_stack(vecs).astype(np.float64)
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    r = int(np.sum(s > tol * s[0]))
    return U[:, :r]


# ---------------------------------------------------------------------------
# Subspace helpers  (fixes #4, #5, #6, #7)
# ---------------------------------------------------------------------------

def _project_out(
    U_target: np.ndarray,
    U_remove: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Orthonormal basis for span(U_target) ∩ span(U_remove)^⊥  (fixes #5, #6).

    Both inputs must have orthonormal columns (callers ensure this).
    Computes the residual of U_target after projecting onto span(U_remove),
    then re-orthonormalises via SVD with rank truncation at ``tol``.

    Returns (d, r), r ≤ U_target.shape[1]; returns (d, 0) when U_target ⊆ U_remove.
    """
    d = U_target.shape[0]
    if U_target.shape[1] == 0:
        return np.zeros((d, 0))
    if U_remove.shape[1] == 0:
        return U_target.copy()

    residual = U_target - U_remove @ (U_remove.T @ U_target)
    cols = [residual[:, i] for i in range(residual.shape[1])]
    return _orthonormal_basis(cols, tol) if cols else np.zeros((d, 0))


def _orthogonalize_against(
    U:   np.ndarray,
    V:   np.ndarray,
    tol: float,
) -> np.ndarray:
    """
    Orthonormal basis for span(U) ∩ span(V)^⊥  (fix #4 — exclusive projectors).

    Like _project_out but drops residual columns with norm ≤ tol before
    re-orthonormalising; slightly more conservative for near-zero residuals.
    Used exclusively to build U_S_excl and U_A_excl.
    """
    if U.shape[1] == 0 or V.shape[1] == 0:
        return U.copy()
    U_perp = U - V @ (V.T @ U)
    cols = [
        U_perp[:, k]
        for k in range(U_perp.shape[1])
        if np.linalg.norm(U_perp[:, k]) > tol
    ]
    return _orthonormal_basis(cols, tol) if cols else np.zeros((U.shape[0], 0))


def _principal_angles_svd(
    U_S: np.ndarray,
    U_A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Principal angles between two subspaces via SVD  (fix #7).

    Singular values of M = U_S^T U_A are cos(θ_i).
    σ_i = 1 → shared direction; σ_i = 0 → orthogonal pair.

    Returns
    -------
    sigma : (k,) ndarray, descending cosines, k = min(m, n)
    L_S   : (d, k) ndarray — left  principal vectors in span(U_S)
    L_A   : (d, k) ndarray — right principal vectors in span(U_A)
    """
    if U_S.shape[1] == 0 or U_A.shape[1] == 0:
        d = U_S.shape[0]
        return np.zeros((0,)), np.zeros((d, 0)), np.zeros((d, 0))
    M = U_S.T @ U_A
    X, sigma, Yt = np.linalg.svd(M, full_matrices=False)
    return sigma, U_S @ X, U_A @ Yt.T


def _subspace_overlap(U_a: np.ndarray, U_b: np.ndarray) -> dict:
    """
    Lightweight overlap diagnostic between two orthonormal bases.

    Returns dict with:
        norm_proj      ∈ [0,1]: ‖σ‖₂ / √(dim_a · dim_b)
                       0 = fully orthogonal, 1 = identical span
        min_angle_deg  : smallest principal angle (degrees)
        max_angle_deg  : largest  principal angle (degrees)
        n_shared_dirs  : count of cosines > 0.99
        n_principal    : number of principal angles computed
    """
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return {
            "norm_proj":     0.0,
            "min_angle_deg": 90.0,
            "max_angle_deg": 90.0,
            "n_shared_dirs": 0,
            "n_principal":   0,
        }
    s = np.linalg.svd(U_a.T @ U_b, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "norm_proj":     float(np.linalg.norm(s) / np.sqrt(U_a.shape[1] * U_b.shape[1])),
        "min_angle_deg": float(angles.min()),
        "max_angle_deg": float(angles.max()),
        "n_shared_dirs": int(np.sum(s > 0.99)),
        "n_principal":   int(len(s)),
    }


# ---------------------------------------------------------------------------
# Schur extraction with relative-tolerance kernel filtering  (fix #3)
# ---------------------------------------------------------------------------

def _extract_schur_subspaces(
    OV:            np.ndarray,
    eig_rel_tol:   float       = 1e-8,
    block_rel_tol: float | None = None,
) -> dict:
    """
    Real Schur decomposition of one OV matrix, partitioned by block type.

    Both thresholds are relative to ‖T‖_F  (fixes #3 + Doc-13 #8).
    The floor of machine-epsilon × 10 × ‖T‖_F ensures the cutoff is never
    tighter than numerical noise:

        eig_tol   = max(‖T‖_F × eig_rel_tol,   ‖T‖_F × 10ε,  1e-12)
        block_tol = max(‖T‖_F × block_rel_tol,  ‖T‖_F × 10ε,  1e-12)
                    (block_rel_tol defaults to eig_rel_tol)

    This replaces the previous mix of absolute 1e-10 (for block detection)
    and per-head relative (for eigenvalue cutoff), which failed silently for
    OV matrices with ‖OV‖ ≪ 1 or ≫ 1.

    Block classification
    --------------------
    2×2 block, |T[i+1,i]| > block_tol    → rot_vecs
    1×1 block, λ >  eig_tol              → real_pos_vecs
    1×1 block, λ < -eig_tol              → real_neg_vecs
    1×1 block, |λ| ≤ eig_tol (kernel)   → real_zero_vecs   (excluded from S)
    2×2 block, |det|^½ ≤ eig_tol        → kernel (n_kernel_rot)

    Returns
    -------
    dict with real_pos_vecs, real_neg_vecs, real_zero_vecs, rot_vecs (lists of
    (d,) arrays), plus diagnostics: n_kernel_real, n_kernel_rot, max_eig_mag,
    eig_tol_used.
    """
    d = OV.shape[0]
    T, Z = schur(OV, output="real")

    nrm = float(np.linalg.norm(T, "fro"))
    eps = float(np.finfo(T.dtype).eps)
    _brel = block_rel_tol if block_rel_tol is not None else eig_rel_tol

    eig_tol   = max(nrm * eig_rel_tol, nrm * eps * 10.0, 1e-12)
    block_tol = max(nrm * _brel,        nrm * eps * 10.0, 1e-12)

    real_pos:  list[np.ndarray] = []
    real_neg:  list[np.ndarray] = []
    real_zero: list[np.ndarray] = []
    rot:       list[np.ndarray] = []
    n_kernel_real = 0
    n_kernel_rot  = 0
    eig_mags: list[float] = []

    i = 0
    while i < d:
        if i < d - 1 and abs(T[i + 1, i]) > block_tol:
            # 2×2 rotation block — eigenvalues are a complex conjugate pair
            a,  b  = float(T[i,     i]),     float(T[i,     i + 1])
            c,  dd = float(T[i + 1, i]),     float(T[i + 1, i + 1])
            mag = float(np.sqrt(max(a * dd - b * c, 0.0)))
            eig_mags.append(mag)
            if mag <= eig_tol:
                n_kernel_rot += 1
            else:
                rot.append(Z[:, i    ].copy())
                rot.append(Z[:, i + 1].copy())
            i += 2
        else:
            val = float(T[i, i])
            eig_mags.append(abs(val))
            if val > eig_tol:
                real_pos.append(Z[:, i].copy())
            elif val < -eig_tol:
                real_neg.append(Z[:, i].copy())
            else:
                real_zero.append(Z[:, i].copy())
                n_kernel_real += 1
            i += 1

    return {
        "real_pos_vecs":  real_pos,
        "real_neg_vecs":  real_neg,
        "real_zero_vecs": real_zero,
        "rot_vecs":       rot,
        "n_kernel_real":  n_kernel_real,
        "n_kernel_rot":   n_kernel_rot,
        "max_eig_mag":    max(eig_mags) if eig_mags else 0.0,
        "eig_tol_used":   eig_tol,
    }


# ---------------------------------------------------------------------------
# Layer aggregation  (fixes #3–#7)
# ---------------------------------------------------------------------------

def _build_for_layer(
    head_ovs:        list[np.ndarray],
    d:               int,
    svd_tol:         float,
    eig_rel_tol:     float = 1e-8,
    cos_overlap_tol: float = 0.99,
) -> dict:
    """
    Aggregate Schur subspaces across all attention heads for one layer.

    Pipeline
    --------
    1. Per-head Schur extraction → real_pos / real_neg / real_zero / rot.
    2. Orthonormalise each cross-head union: U_pos_raw, U_neg_raw, U_A_raw.
       Build U_S_raw = orth([U_pos_raw | U_neg_raw]) for diagnostics.
    3. Pre-resolution overlap diagnostic: U_S_raw ↔ U_A_raw.
    4. Fix #5: U_A  := U_A_raw  ∩ span(U_S_raw)^⊥   (S wins).
    5. Fix #6: U_neg := U_neg_raw ∩ span(U_pos)^⊥;
               U_S   := [U_pos | U_neg].
    6. Build P_S = U_S U_S^T,  P_A = U_A U_A^T.
    7. Fix #4: exclusive projectors P_S_excl (S∖A),  P_A_excl (A∖S).
    8. Fix #7: principal angles and overlap diagnostics.

    Note on step 4: span(U_S_raw) = span(U_S) because step 5 only changes
    the basis of U_S, not its span.  Projecting A out of U_S_raw is therefore
    equivalent to projecting out of the final U_S.
    """
    all_real_pos:  list[np.ndarray] = []
    all_real_neg:  list[np.ndarray] = []
    all_real_zero: list[np.ndarray] = []
    all_rot:       list[np.ndarray] = []
    total_kernel_real = 0
    total_kernel_rot  = 0
    max_eig_mag       = 0.0

    for OV in head_ovs:
        sub = _extract_schur_subspaces(OV, eig_rel_tol=eig_rel_tol)
        all_real_pos .extend(sub["real_pos_vecs"])
        all_real_neg .extend(sub["real_neg_vecs"])
        all_real_zero.extend(sub["real_zero_vecs"])
        all_rot      .extend(sub["rot_vecs"])
        total_kernel_real += sub["n_kernel_real"]
        total_kernel_rot  += sub["n_kernel_rot"]
        max_eig_mag        = max(max_eig_mag, sub["max_eig_mag"])

    # Step 2: orthonormalise cross-head unions (fix #8 guards)
    U_pos_raw = (
        _orthonormal_basis(all_real_pos, svd_tol)
        if all_real_pos else np.zeros((d, 0))
    )
    U_neg_raw = (
        _orthonormal_basis(all_real_neg, svd_tol)
        if all_real_neg else np.zeros((d, 0))
    )
    U_A_raw = (
        _orthonormal_basis(all_rot, svd_tol)
        if all_rot else np.zeros((d, 0))
    )

    real_vecs = list(U_pos_raw.T) + list(U_neg_raw.T)
    U_S_raw = (
        _orthonormal_basis(real_vecs, svd_tol)
        if real_vecs else np.zeros((d, 0))
    )

    # Step 3: pre-resolution overlap diagnostic
    sa_overlap_pre = _subspace_overlap(U_S_raw, U_A_raw)

    # Step 4 (fix #5): A must be orthogonal to S; S wins
    U_A = _project_out(U_A_raw, U_S_raw, svd_tol)

    # Step 5 (fix #6): U_neg must be orthogonal to U_pos
    U_pos = U_pos_raw
    U_neg = _project_out(U_neg_raw, U_pos, svd_tol)

    U_S = (
        np.column_stack([U_pos, U_neg])
        if U_pos.shape[1] + U_neg.shape[1] > 0
        else np.zeros((d, 0))
    )

    # Step 6: projectors
    P_S = U_S @ U_S.T if U_S.shape[1] > 0 else np.zeros((d, d))
    P_A = U_A @ U_A.T if U_A.shape[1] > 0 else np.zeros((d, d))

    # Step 7 (fix #4): exclusive projectors for S∖A and A∖S
    U_S_excl = _orthogonalize_against(U_S, U_A, svd_tol)
    U_A_excl = _orthogonalize_against(U_A, U_S, svd_tol)
    P_S_excl = (
        U_S_excl @ U_S_excl.T if U_S_excl.shape[1] > 0 else np.zeros((d, d))
    )
    P_A_excl = (
        U_A_excl @ U_A_excl.T if U_A_excl.shape[1] > 0 else np.zeros((d, d))
    )

    # Step 8 (fix #7): principal angles between final U_S and U_A
    sigma, L_S, L_A = _principal_angles_svd(U_S, U_A)
    dim_overlap = int(np.sum(sigma > cos_overlap_tol)) if sigma.size > 0 else 0
    PS_PA_fro = (
        float(np.linalg.norm(P_S @ P_A, ord="fro"))
        if U_S.shape[1] > 0 and U_A.shape[1] > 0 else 0.0
    )

    # Kernel dimension (rank of union of real_zero vectors)
    dim_kernel = 0
    if all_real_zero:
        try:
            dim_kernel = int(_orthonormal_basis(all_real_zero, svd_tol).shape[1])
        except Exception:
            dim_kernel = 0

    dim_S = int(U_S.shape[1])
    dim_A = int(U_A.shape[1])

    return {
        # ── Core projectors (unchanged contract) ─────────────────────────
        "P_S":    P_S,
        "P_A":    P_A,
        "U_S":    U_S,
        "U_A":    U_A,
        "U_pos":  U_pos,
        "U_neg":  U_neg,
        "dim_S":  dim_S,
        "dim_A":  dim_A,
        "frac_S": float(dim_S / d) if d else 0.0,
        "frac_A": float(dim_A / d) if d else 0.0,
        # ── Fix #3 diagnostics ───────────────────────────────────────────
        "dim_kernel":            dim_kernel,
        "dim_pos_pre":           int(U_pos_raw.shape[1]),
        "dim_neg_pre":           int(U_neg_raw.shape[1]),
        "dim_A_pre":             int(U_A_raw.shape[1]),
        "n_kernel_real_dropped": total_kernel_real,
        "n_kernel_rot_dropped":  total_kernel_rot,
        "max_eig_mag":           float(max_eig_mag),
        "eig_rel_tol":           float(eig_rel_tol),
        # ── Fix #5/#6 pre-resolution overlap ─────────────────────────────
        "sa_overlap_pre":        sa_overlap_pre,
        # ── Fix #4: exclusive projectors ─────────────────────────────────
        "P_S_excl":  P_S_excl,
        "P_A_excl":  P_A_excl,
        "U_S_excl":  U_S_excl,
        "U_A_excl":  U_A_excl,
        "dim_S_excl": int(U_S_excl.shape[1]),
        "dim_A_excl": int(U_A_excl.shape[1]),
        # ── Fix #7: principal angles ──────────────────────────────────────
        "principal_angles": sigma.tolist() if sigma.size > 0 else [],
        "L_S_principal":    L_S,
        "L_A_principal":    L_A,
        "dim_overlap":      dim_overlap,
        "frac_overlap":     float(dim_overlap / d) if d else 0.0,
        "PS_PA_fro_norm":   PS_PA_fro,
        "cos_overlap_tol":  float(cos_overlap_tol),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_global_projectors(
    ov_data:         dict,
    svd_tol:         float = 1e-8,
    eig_rel_tol:     float = 1e-12, # was 1e-8
    cos_overlap_tol: float = 0.99,
    ) -> dict:
    """
    Build global residual-stream S/A projectors from per-head OV matrices.

    Parameters
    ----------
    ov_data         : output of weights.extract_ov_circuit.
                      Required keys: is_per_layer, ov_per_head, n_heads,
                      d_model, layer_names.
    svd_tol         : SVD truncation tolerance for _orthonormal_basis calls.
    eig_rel_tol     : Relative eigenvalue cutoff per head: eigenvalues with
                      |λ| ≤ eig_rel_tol × ‖T‖_F are treated as kernel and
                      excluded from U_S.  (Fix #3.)
    cos_overlap_tol : Principal-angle cosine threshold for dim_overlap count.
                      (Fix #7.)

    Returns
    -------
    dict with is_per_layer, layer_names, d_model, per_layer (one projector
    dict per layer), each carrying P_S, P_A, U_S, U_A, exclusive projectors,
    and all diagnostic scalars.
    """
    is_per_layer = ov_data["is_per_layer"]
    d            = ov_data["d_model"]
    layer_names  = ov_data["layer_names"]

    results = []
    if is_per_layer:
        for layer_idx in range(len(layer_names)):
            results.append(
                _build_for_layer(
                    ov_data["ov_per_head"][layer_idx],
                    d, svd_tol, eig_rel_tol, cos_overlap_tol,
                )
            )
    else:
        results.append(
            _build_for_layer(
                ov_data["ov_per_head"],
                d, svd_tol, eig_rel_tol, cos_overlap_tol,
            )
        )

    return {
        "is_per_layer": is_per_layer,
        "layer_names":  layer_names,
        "d_model":      d,
        "per_layer":    results,
    }


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

# Arrays stored in the NPZ.  Optional arrays (exclusive projectors, principal
# vectors) have size 0 when empty and are skipped in that case.
_ARRAY_KEYS = (
    "P_S", "P_A", "U_S", "U_A", "U_pos", "U_neg",
    "P_S_excl", "P_A_excl", "U_S_excl", "U_A_excl",
    "L_S_principal", "L_A_principal",
)

# Scalar / list fields persisted in the JSON sidecar.
_SCALAR_META_KEYS = (
    "dim_S", "dim_A", "frac_S", "frac_A",
    "dim_kernel", "dim_pos_pre", "dim_neg_pre", "dim_A_pre",
    "n_kernel_real_dropped", "n_kernel_rot_dropped",
    "max_eig_mag", "eig_rel_tol",
    "sa_overlap_pre",
    "dim_S_excl", "dim_A_excl",
    "dim_overlap", "frac_overlap", "PS_PA_fro_norm", "cos_overlap_tol",
    "principal_angles",
)

_SCALAR_DEFAULTS: dict = {
    "dim_kernel": 0, "dim_pos_pre": 0, "dim_neg_pre": 0, "dim_A_pre": 0,
    "n_kernel_real_dropped": 0, "n_kernel_rot_dropped": 0,
    "max_eig_mag": 0.0, "eig_rel_tol": 1e-8,
    "sa_overlap_pre": {},
    "dim_S_excl": 0, "dim_A_excl": 0,
    "dim_overlap": 0, "frac_overlap": 0.0,
    "PS_PA_fro_norm": 0.0, "cos_overlap_tol": 0.99,
    "principal_angles": [],
}


def save_projectors(proj: dict, path: Path) -> None:
    """
    Persist projector dict to ``<path>.npz`` + ``<path>.json`` sidecar.

    Arrays are stored at float32 in the NPZ; all scalars and diagnostics go
    in the JSON sidecar.  Backward-compatible: callers that only use P_S / P_A
    see no change.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    meta: dict = {
        "is_per_layer":   proj["is_per_layer"],
        "layer_names":    proj["layer_names"],
        "d_model":        proj["d_model"],
        "per_layer_meta": [],
    }

    for idx, (lname, entry) in enumerate(zip(proj["layer_names"], proj["per_layer"])):
        prefix = f"layer{idx}_"
        for key in _ARRAY_KEYS:
            arr = entry.get(key)
            if arr is not None and np.asarray(arr).size > 0:
                arrays[prefix + key] = np.asarray(arr, dtype=np.float32)

        lmeta: dict = {"layer_name": lname}
        for key in _SCALAR_META_KEYS:
            lmeta[key] = entry.get(key, _SCALAR_DEFAULTS.get(key))
        meta["per_layer_meta"].append(lmeta)

    np.savez(path, **arrays)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_projectors(path: Path) -> dict:
    """
    Load projector dict from ``<path>.npz`` + ``<path>.json`` sidecar.

    Drop-in replacement for callers that only use P_S / P_A.
    Optional arrays fall back to zero arrays of the correct shape when
    absent from the NPZ (e.g. files saved before Bug #4 was added).

    Note: d_model is read from the top-level JSON key, not from per-layer
    metadata (which does not carry it).
    """
    path = Path(path)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)

    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    data = np.load(str(npz_path))

    d = meta["d_model"]   # single authoritative source; not lmeta
    per_layer = []

    for idx, lmeta in enumerate(meta["per_layer_meta"]):
        prefix = f"layer{idx}_"
        entry: dict = {}

        # Required arrays — KeyError here means a corrupt save file
        for key in ("P_S", "P_A", "U_S", "U_A", "U_pos", "U_neg"):
            entry[key] = data[prefix + key].astype(np.float64)

        # Optional (d, d) projector arrays — zero matrix if absent
        for key in ("P_S_excl", "P_A_excl"):
            npz_key = prefix + key
            entry[key] = (
                data[npz_key].astype(np.float64)
                if npz_key in data else np.zeros((d, d))
            )

        # Optional (d, r) basis arrays — zero (d, 0) if absent
        for key in ("U_S_excl", "U_A_excl", "L_S_principal", "L_A_principal"):
            npz_key = prefix + key
            entry[key] = (
                data[npz_key].astype(np.float64)
                if npz_key in data else np.zeros((d, 0))
            )

        # Scalar metadata
        for key in _SCALAR_META_KEYS:
            entry[key] = lmeta.get(key, _SCALAR_DEFAULTS.get(key))

        per_layer.append(entry)

    return {
        "is_per_layer": meta["is_per_layer"],
        "layer_names":  meta["layer_names"],
        "d_model":      d,
        "per_layer":    per_layer,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def projector_summary(proj: dict) -> list[dict]:
    """Per-layer scalar diagnostics as a list of dicts."""
    d = proj["d_model"]
    rows = []
    for lname, entry in zip(proj["layer_names"], proj["per_layer"]):
        n_kernel = (
            entry.get("n_kernel_real_dropped", 0)
            + entry.get("n_kernel_rot_dropped", 0)
        )
        rows.append({
            "layer_name":   lname,
            "dim_S":        entry["dim_S"],
            "dim_A":        entry["dim_A"],
            "dim_S_excl":   entry.get("dim_S_excl",   0),
            "dim_A_excl":   entry.get("dim_A_excl",   0),
            "frac_S":       entry["frac_S"],
            "frac_A":       entry["frac_A"],
            "frac_S_excl":  float(entry.get("dim_S_excl", 0) / d) if d else 0.0,
            "frac_A_excl":  float(entry.get("dim_A_excl", 0) / d) if d else 0.0,
            "n_kernel":     n_kernel,
            "dim_kernel":   entry.get("dim_kernel",   0),
            "max_eig_mag":  entry.get("max_eig_mag",  0.0),
            "PS_PA_fro":    entry.get("PS_PA_fro_norm", 0.0),
            "dim_overlap":  entry.get("dim_overlap",  0),
            "frac_overlap": entry.get("frac_overlap", 0.0),
        })
    return rows


def print_projector_summary(proj: dict) -> None:
    """Print a compact diagnostics table to stdout."""
    rows = projector_summary(proj)
    header = (
        f"{'layer':<24} {'dim_S':>6} {'dim_A':>6} "
        f"{'S_excl':>7} {'A_excl':>7} {'n_kern':>7} "
        f"{'fro(PS·PA)':>11} {'frac_ovlp':>10}"
    )
    sep = "─" * len(header)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['layer_name']:<24} "
            f"{r['dim_S']:>6d} "
            f"{r['dim_A']:>6d} "
            f"{r['dim_S_excl']:>7d} "
            f"{r['dim_A_excl']:>7d} "
            f"{r['n_kernel']:>7d} "
            f"{r['PS_PA_fro']:>11.5f} "
            f"{r['frac_overlap']:>10.5f}"
        )