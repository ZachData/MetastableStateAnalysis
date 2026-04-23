"""
subspace_build.py — Global S/A residual-stream projectors for Phase 6.

Phase 6 needs projectors onto the *real* (S) and *imaginary* (A) subspaces
of the residual stream as a whole, not just per-head OV subspaces.

Strategy
--------
For each head h at each layer l, the OV circuit's real Schur decomposition
partitions d_model dimensions into:
  - 1×1 blocks: real eigenvectors  → contribute to P_S
  - 2×2 blocks: rotation planes    → contribute to P_A

Across all heads, the union of real Schur vectors spans a (possibly
overcomplete) basis for the real channel; the union of rotation plane
vectors spans the imaginary channel.

We orthonormalise each union via SVD (thin) and build rank-d projectors.
For ALBERT (shared weights, one OV per head), one decomposition suffices.
For per-layer models, we build one pair of projectors per layer.

The output is a dict of projector matrices cached to disk.  Every Phase 6
module loads from this cache rather than re-running the Schur decomposition.

Key design decisions
--------------------
- We use the UNION of all-head subspaces, not an intersection.  Rationale:
  the residual stream receives writes from all heads; the channel the stream
  occupies is the union of what every head can write.
- Orthonormalisation via SVD (not QR) to get a clean ranked basis.  The
  singular values reveal how much of the target channel is actually spanned.
- Attractive/repulsive split retained within P_S for downstream use (B.2/B.3
  need U_+ and U_- separately).
- For per-layer models the projectors are indexed by layer name.

Functions
---------
build_global_projectors     : main entry — builds and returns projector dict
save_projectors             : persist to disk (.npz)
load_projectors             : load from disk
projector_summary           : scalar diagnostics (dim, condition, etc.)
"""

import json
import numpy as np
from pathlib import Path
from scipy.linalg import schur


# ---------------------------------------------------------------------------
# Internal: orthonormal basis from a set of column vectors
# ---------------------------------------------------------------------------

def _orthonormal_basis(vecs: list[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    """
    Given a list of column vectors (each shape (d,)), return an orthonormal
    basis for their span via thin SVD.

    Callers must NOT pass an empty list — guard with:
        U = _orthonormal_basis(vecs, tol) if vecs else np.zeros((d, 0))
    Passing an empty list raises ValueError to make the contract explicit.

    Returns
    -------
    U : (d, r) ndarray, columns orthonormal, r = effective rank
    """
    # FIX (Bug 8): raise instead of silently returning a (0, 0) array that
    # would propagate incorrect shapes into every downstream projector.
    # All call sites in this module already guard with "if vecs else np.zeros((d,0))".
    if not vecs:
        raise ValueError(
            "_orthonormal_basis requires at least one vector. "
            "Guard the call site with: U = _orthonormal_basis(v, tol) if v else np.zeros((d, 0))"
        )
    V = np.column_stack(vecs).astype(np.float64)   # (d, n_vecs)
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    r = int(np.sum(s > tol * s[0]))
    return U[:, :r]


# ---------------------------------------------------------------------------
# Internal: single-OV Schur extraction
# ---------------------------------------------------------------------------

def _extract_schur_subspaces(OV: np.ndarray) -> dict:
    """
    Real Schur decomposition of one OV matrix.

    Returns
    -------
    dict with:
      real_pos_vecs  : list of (d,) arrays — 1×1 blocks with positive eigenvalue
      real_neg_vecs  : list of (d,) arrays — 1×1 blocks with negative eigenvalue
      rot_vecs       : list of (d,) arrays — paired vectors from 2×2 blocks
                       (both columns of each rotation plane, interleaved)
    """
    d = OV.shape[0]
    T, Z = schur(OV, output='real')

    real_pos, real_neg, rot = [], [], []

    i = 0
    while i < d:
        if i < d - 1 and abs(T[i + 1, i]) > 1e-10:
            # 2×2 rotation block
            rot.append(Z[:, i].copy())
            rot.append(Z[:, i + 1].copy())
            i += 2
        else:
            # 1×1 real block
            val = float(T[i, i])
            vec = Z[:, i].copy()
            if val >= 0:
                real_pos.append(vec)
            else:
                real_neg.append(vec)
            i += 1

    return {"real_pos_vecs": real_pos, "real_neg_vecs": real_neg, "rot_vecs": rot}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_global_projectors(
    ov_data: dict,
    svd_tol: float = 1e-8,
) -> dict:
    """
    Build global residual-stream S/A projectors from per-head OV matrices.

    Parameters
    ----------
    ov_data   : output of weights.extract_ov_circuit (or loaded from disk).
                Expected keys:
                  is_per_layer : bool
                  ov_per_head  : for ALBERT — list of (d, d) per-head OV
                               : for per-layer — list of lists, indexed
                                 [layer_idx][head_idx]
                  n_heads      : int
                  d_model      : int
                  layer_names  : list of str

    Returns
    -------
    dict with:
      is_per_layer : bool
      layer_names  : list of str (length 1 for ALBERT, L for per-layer)
      d_model      : int
      per_layer    : list of dicts, one per entry in layer_names, each with:
          P_S          : (d, d) projector onto real subspace (union of all heads)
          P_A          : (d, d) projector onto imaginary subspace (union of all heads)
          U_pos        : (d, r_pos) basis for attractive 1×1 subspace (λ > 0)
          U_neg        : (d, r_neg) basis for repulsive 1×1 subspace (λ < 0)
          U_S          : (d, r_S)  full real subspace basis (U_pos | U_neg)
          U_A          : (d, r_A)  imaginary subspace basis
          dim_S        : int — effective rank of real subspace
          dim_A        : int — effective rank of imaginary subspace
          frac_S       : float — dim_S / d_model
          frac_A       : float — dim_A / d_model
    """
    is_per_layer = ov_data["is_per_layer"]
    d = ov_data["d_model"]
    n_heads = ov_data["n_heads"]
    layer_names = ov_data["layer_names"]

    results = []

    if is_per_layer:
        # ov_data["ov_per_head"] is a list (layers) of lists (heads)
        for layer_idx, lname in enumerate(layer_names):
            head_ovs = ov_data["ov_per_head"][layer_idx]   # list of (d,d)
            entry = _build_for_layer(head_ovs, d, svd_tol)
            results.append(entry)
    else:
        # ALBERT: single shared set of per-head OVs
        head_ovs = ov_data["ov_per_head"]   # list of (d,d), length n_heads
        entry = _build_for_layer(head_ovs, d, svd_tol)
        results.append(entry)

    return {
        "is_per_layer": is_per_layer,
        "layer_names":  layer_names,
        "d_model":      d,
        "per_layer":    results,
    }


def _build_for_layer(head_ovs: list, d: int, svd_tol: float) -> dict:
    """Aggregate Schur subspaces across all heads for one layer."""
    all_real_pos, all_real_neg, all_rot = [], [], []

    for OV in head_ovs:
        sub = _extract_schur_subspaces(OV)
        all_real_pos.extend(sub["real_pos_vecs"])
        all_real_neg.extend(sub["real_neg_vecs"])
        all_rot.extend(sub["rot_vecs"])

    # Orthonormal bases — guard against empty lists before calling _orthonormal_basis
    U_pos = _orthonormal_basis(all_real_pos, svd_tol) if all_real_pos else np.zeros((d, 0))
    U_neg = _orthonormal_basis(all_real_neg, svd_tol) if all_real_neg else np.zeros((d, 0))
    U_A   = _orthonormal_basis(all_rot,      svd_tol) if all_rot      else np.zeros((d, 0))

    # Full real basis = span(U_pos ∪ U_neg), re-orthonormalised
    real_vecs = (
        list(U_pos.T) + list(U_neg.T)
        if (U_pos.shape[1] + U_neg.shape[1]) > 0
        else []
    )
    U_S = _orthonormal_basis(real_vecs, svd_tol) if real_vecs else np.zeros((d, 0))

    # Projectors
    P_S = U_S @ U_S.T if U_S.shape[1] > 0 else np.zeros((d, d))
    P_A = U_A @ U_A.T if U_A.shape[1] > 0 else np.zeros((d, d))

    dim_S = U_S.shape[1]
    dim_A = U_A.shape[1]

    return {
        "P_S":    P_S,
        "P_A":    P_A,
        "U_pos":  U_pos,
        "U_neg":  U_neg,
        "U_S":    U_S,
        "U_A":    U_A,
        "dim_S":  dim_S,
        "dim_A":  dim_A,
        "frac_S": float(dim_S / d),
        "frac_A": float(dim_A / d),
    }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_projectors(proj: dict, path: Path) -> None:
    """
    Persist projector dict to an .npz archive.

    Large arrays (P_S, P_A, U_S, U_A, U_pos, U_neg) are stored as float32
    to keep file sizes manageable.  Scalar metadata is stored as a JSON
    sidecar with the same stem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    meta = {
        "is_per_layer": proj["is_per_layer"],
        "layer_names":  proj["layer_names"],
        "d_model":      proj["d_model"],
        "per_layer_meta": [],
    }

    for idx, (lname, entry) in enumerate(zip(proj["layer_names"], proj["per_layer"])):
        prefix = f"layer{idx}_"
        for key in ("P_S", "P_A", "U_S", "U_A", "U_pos", "U_neg"):
            arr = entry[key].astype(np.float32)
            arrays[prefix + key] = arr
        meta["per_layer_meta"].append({
            "layer_name": lname,
            "dim_S":  entry["dim_S"],
            "dim_A":  entry["dim_A"],
            "frac_S": entry["frac_S"],
            "frac_A": entry["frac_A"],
        })

    np.savez(path, **arrays)

    json_path = path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_projectors(path: Path) -> dict:
    """
    Load projector dict from .npz + .json sidecar written by save_projectors.
    """
    path = Path(path)
    json_path = path.with_suffix(".json")

    with open(json_path) as f:
        meta = json.load(f)

    data = np.load(str(path) if path.suffix == ".npz" else str(path) + ".npz")

    per_layer = []
    for idx, lmeta in enumerate(meta["per_layer_meta"]):
        prefix = f"layer{idx}_"
        entry = {
            "P_S":    data[prefix + "P_S"].astype(np.float64),
            "P_A":    data[prefix + "P_A"].astype(np.float64),
            "U_S":    data[prefix + "U_S"].astype(np.float64),
            "U_A":    data[prefix + "U_A"].astype(np.float64),
            "U_pos":  data[prefix + "U_pos"].astype(np.float64),
            "U_neg":  data[prefix + "U_neg"].astype(np.float64),
            "dim_S":  lmeta["dim_S"],
            "dim_A":  lmeta["dim_A"],
            "frac_S": lmeta["frac_S"],
            "frac_A": lmeta["frac_A"],
        }
        per_layer.append(entry)

    return {
        "is_per_layer": meta["is_per_layer"],
        "layer_names":  meta["layer_names"],
        "d_model":      meta["d_model"],
        "per_layer":    per_layer,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def projector_summary(proj: dict) -> list[dict]:
    """
    Return a list of per-layer scalar diagnostics.

    For each layer:
      layer_name    : str
      dim_S, dim_A  : int — effective rank of each channel
      frac_S, frac_A: float — fraction of d_model in each channel
      overlap       : float — ||P_S @ P_A||_F / sqrt(dim_S * dim_A)
                      Near 0 = channels orthogonal; near 1 = channels overlap
      coverage      : float — dim_S + dim_A as fraction of d_model
                      > 1.0 means the channels overlap
    """
    d = proj["d_model"]
    rows = []
    for lname, entry in zip(proj["layer_names"], proj["per_layer"]):
        overlap_mat = entry["P_S"] @ entry["P_A"]
        denom = max(np.sqrt(entry["dim_S"] * entry["dim_A"]), 1.0)
        overlap = float(np.linalg.norm(overlap_mat, "fro") / denom)
        rows.append({
            "layer_name": lname,
            "dim_S":      entry["dim_S"],
            "dim_A":      entry["dim_A"],
            "frac_S":     entry["frac_S"],
            "frac_A":     entry["frac_A"],
            "overlap":    overlap,
            "coverage":   float((entry["dim_S"] + entry["dim_A"]) / d),
        })
    return rows


def print_projector_summary(proj: dict) -> None:
    rows = projector_summary(proj)
    header = f"{'layer':<20} {'dim_S':>6} {'dim_A':>6} {'frac_S':>7} {'frac_A':>7} {'overlap':>8} {'coverage':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['layer_name']:<20} "
            f"{r['dim_S']:>6d} "
            f"{r['dim_A']:>6d} "
            f"{r['frac_S']:>7.3f} "
            f"{r['frac_A']:>7.3f} "
            f"{r['overlap']:>8.4f} "
            f"{r['coverage']:>9.3f}"
        )
