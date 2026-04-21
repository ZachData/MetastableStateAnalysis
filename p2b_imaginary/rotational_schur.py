"""
rotational_schur.py — Characterize V_eff's rotational spectrum.

The Schur decomposition V = Z T Z^T produces a quasi-triangular T with:
  - 1×1 blocks: real eigenvalues (the ~3% Phase 2 analyzed)
  - 2×2 blocks: complex conjugate pairs encoding rotation + scaling

Each 2×2 block [[a, b], [c, a]] encodes:
  - Rotation angle θ = arctan2(sqrt(-bc), a)  (for bc < 0)
  - Spectral radius ρ = sqrt(a² - bc)
  - Real part sign: sgn(a) → attractive (a>0) or repulsive (a<0)
  - The rotation plane: the pair of Schur vectors spanning the invariant 2D subspace

This script extracts per-block statistics and computes the Henrici
departure from normality (sum of squared off-diagonal entries of T),
which measures how much the rotational blocks interact.

Functions
---------
extract_schur_blocks       : parse T into 1×1 and 2×2 blocks with statistics
rotation_energy_fractions  : fraction of spectral energy in rotational vs signed modes
henrici_nonnormality       : departure from normality measure
rotation_depth_profile     : per-layer rotation angle distribution (GPT-2)
analyze_rotational_spectrum: full pipeline for one model
"""

import numpy as np
from scipy.linalg import schur, norm as la_norm


# ---------------------------------------------------------------------------
# Core block extraction
# ---------------------------------------------------------------------------

def extract_schur_blocks(OV: np.ndarray) -> dict:
    """
    Schur-decompose OV and parse into 1×1 and 2×2 blocks.

    Parameters
    ----------
    OV : (d, d) ndarray — composed OV matrix (row-vector convention)

    Returns
    -------
    dict with:
      schur_T         : (d, d) quasi-triangular Schur form
      schur_Z         : (d, d) orthogonal Schur vectors
      blocks_1x1      : list of dicts with keys:
                          idx     — diagonal index
                          value   — the real eigenvalue
                          sign    — +1 or -1
                          schur_vec — the corresponding column of Z
      blocks_2x2      : list of dicts with keys:
                          idx     — starting diagonal index
                          theta   — rotation angle in radians [0, π]
                          rho     — spectral radius of the block
                          a       — real part (diagonal entry)
                          bc      — product of off-diagonal entries (negative for rotation)
                          sign    — sgn(a), the radial direction
                          plane   — (d, 2) ndarray, the two Schur vectors spanning
                                    the invariant 2D subspace
      d               : int — dimension
      n_real          : int — number of 1×1 blocks
      n_complex       : int — number of 2×2 blocks
    """
    d = OV.shape[0]

    # Real Schur: T quasi-triangular, Z orthogonal, V = Z T Z^T
    T, Z = schur(OV, output='real')

    blocks_1x1 = []
    blocks_2x2 = []

    i = 0
    while i < d:
        if i == d - 1:
            # Last entry, must be 1×1
            blocks_1x1.append({
                "idx":       i,
                "value":     float(T[i, i]),
                "sign":      1 if T[i, i] >= 0 else -1,
                "schur_vec": Z[:, i].copy(),
            })
            i += 1
        elif abs(T[i + 1, i]) > 1e-10:
            # 2×2 block: T[i:i+2, i:i+2] = [[a, b], [c, a']]
            # For a proper Schur block, a ≈ a' (both are the real part)
            a = float(T[i, i])
            b = float(T[i, i + 1])
            c = float(T[i + 1, i])

            # bc < 0 for genuine rotation (conjugate pair)
            bc = b * c
            if bc < 0:
                theta = float(np.arctan2(np.sqrt(-bc), abs(a)))
            else:
                # Degenerate: two real eigenvalues packed as 2×2
                theta = 0.0

            rho = float(np.sqrt(a * a - bc)) if (a * a - bc) >= 0 else float(abs(a))

            blocks_2x2.append({
                "idx":   i,
                "theta": theta,
                "rho":   rho,
                "a":     a,
                "bc":    bc,
                "sign":  1 if a >= 0 else -1,
                "plane": Z[:, i:i + 2].copy(),   # (d, 2)
            })
            i += 2
        else:
            # 1×1 block
            blocks_1x1.append({
                "idx":       i,
                "value":     float(T[i, i]),
                "sign":      1 if T[i, i] >= 0 else -1,
                "schur_vec": Z[:, i].copy(),
            })
            i += 1

    return {
        "schur_T":    T,
        "schur_Z":    Z,
        "blocks_1x1": blocks_1x1,
        "blocks_2x2": blocks_2x2,
        "d":          d,
        "n_real":     len(blocks_1x1),
        "n_complex":  len(blocks_2x2),
    }


# ---------------------------------------------------------------------------
# Energy fractions
# ---------------------------------------------------------------------------

def rotation_energy_fractions(block_data: dict) -> dict:
    """
    Fraction of V_eff's spectral energy in rotational (2×2) vs signed (1×1) modes.

    Spectral energy of a 1×1 block: |λ|²
    Spectral energy of a 2×2 block: ρ² (= a² - bc = |λ|² for the conjugate pair)
    Total spectral energy: sum over all blocks.

    Returns
    -------
    dict with:
      rotational_energy  : float — sum of ρ² over 2×2 blocks
      signed_energy      : float — sum of λ² over 1×1 blocks
      total_energy       : float
      rotational_fraction: float — rotational / total
      signed_fraction    : float — signed / total
      n_real             : int
      n_complex          : int
      frac_real          : float — n_real / d
      frac_complex_dims  : float — 2 * n_complex / d (fraction of dimensions in rotation)
    """
    signed_e = sum(b["value"] ** 2 for b in block_data["blocks_1x1"])
    rotation_e = sum(b["rho"] ** 2 for b in block_data["blocks_2x2"])
    total = signed_e + rotation_e

    d = block_data["d"]
    n_r = block_data["n_real"]
    n_c = block_data["n_complex"]

    return {
        "rotational_energy":   float(rotation_e),
        "signed_energy":       float(signed_e),
        "total_energy":        float(total),
        "rotational_fraction": float(rotation_e / max(total, 1e-12)),
        "signed_fraction":     float(signed_e / max(total, 1e-12)),
        "n_real":              n_r,
        "n_complex":           n_c,
        "frac_real":           float(n_r / d),
        "frac_complex_dims":   float(2 * n_c / d),
    }


# ---------------------------------------------------------------------------
# Rotation angle statistics
# ---------------------------------------------------------------------------

def rotation_angle_stats(block_data: dict) -> dict:
    """
    Summary statistics of the rotation angle distribution.

    Returns
    -------
    dict with:
      thetas        : list of float — all rotation angles
      rhos          : list of float — all spectral radii
      signs         : list of int — radial direction per block
      theta_mean    : float
      theta_std     : float
      theta_median  : float
      rho_mean      : float
      rho_std       : float
      frac_expanding: float — fraction with ρ > 1 (spiral outward)
      frac_contracting: float — fraction with ρ < 1
      frac_attractive_rot: float — fraction of 2×2 blocks with a > 0
      frac_repulsive_rot : float — fraction with a < 0
    """
    blocks = block_data["blocks_2x2"]
    if not blocks:
        return {
            "thetas": [], "rhos": [], "signs": [],
            "theta_mean": 0.0, "theta_std": 0.0, "theta_median": 0.0,
            "rho_mean": 0.0, "rho_std": 0.0,
            "frac_expanding": 0.0, "frac_contracting": 0.0,
            "frac_attractive_rot": 0.0, "frac_repulsive_rot": 0.0,
        }

    thetas = [b["theta"] for b in blocks]
    rhos = [b["rho"] for b in blocks]
    signs = [b["sign"] for b in blocks]
    n = len(blocks)

    return {
        "thetas":              thetas,
        "rhos":                rhos,
        "signs":               signs,
        "theta_mean":          float(np.mean(thetas)),
        "theta_std":           float(np.std(thetas)),
        "theta_median":        float(np.median(thetas)),
        "rho_mean":            float(np.mean(rhos)),
        "rho_std":             float(np.std(rhos)),
        "frac_expanding":      float(sum(1 for r in rhos if r > 1.0) / n),
        "frac_contracting":    float(sum(1 for r in rhos if r < 1.0) / n),
        "frac_attractive_rot": float(sum(1 for s in signs if s > 0) / n),
        "frac_repulsive_rot":  float(sum(1 for s in signs if s < 0) / n),
    }


# ---------------------------------------------------------------------------
# Henrici non-normality
# ---------------------------------------------------------------------------

def henrici_nonnormality(block_data: dict) -> dict:
    """
    Henrici departure from normality: ||T||_F² - Σ|λ_i|²

    For a normal matrix this is zero (T is diagonal in the complex Schur form).
    For a non-normal matrix, this equals the sum of squared off-diagonal
    entries of T, measuring how much the Schur blocks interact.

    Also computes the relative departure: Henrici / ||T||_F².

    Returns
    -------
    dict with:
      henrici_absolute  : float — ||T||_F² - Σ|λ_i|²
      henrici_relative  : float — henrici_absolute / ||T||_F²
      T_frob_sq         : float — ||T||_F²
      eigenvalue_energy  : float — Σ|λ_i|²
    """
    T = block_data["schur_T"]
    T_frob_sq = float(np.sum(T ** 2))

    # Eigenvalue energy: sum of |λ|² over all eigenvalues
    # For 1×1 blocks: λ² = value²
    # For 2×2 blocks: |λ|² = ρ² (each conjugate pair contributes 2 × ρ²)
    eig_energy = sum(b["value"] ** 2 for b in block_data["blocks_1x1"])
    eig_energy += 2.0 * sum(b["rho"] ** 2 for b in block_data["blocks_2x2"])

    henrici = T_frob_sq - eig_energy

    return {
        "henrici_absolute": float(max(henrici, 0.0)),  # clamp numerical noise
        "henrici_relative": float(max(henrici, 0.0) / max(T_frob_sq, 1e-12)),
        "T_frob_sq":        T_frob_sq,
        "eigenvalue_energy": float(eig_energy),
    }


# ---------------------------------------------------------------------------
# Rotation planes as projectors
# ---------------------------------------------------------------------------

def build_rotation_plane_projectors(block_data: dict, top_k: int = 32) -> dict:
    """
    Build projectors onto the top-k rotation planes (by spectral radius).

    Each rotation plane is spanned by two Schur vectors. The projector
    onto plane j is P_j = v1 @ v1^T + v2 @ v2^T where v1, v2 are the
    columns of block["plane"].

    Also builds a combined projector onto ALL rotation planes and
    onto the real (non-rotating) subspace.

    Parameters
    ----------
    block_data : output of extract_schur_blocks
    top_k      : number of dominant rotation planes to return individually

    Returns
    -------
    dict with:
      top_k_planes       : list of (d, 2) ndarrays — the plane basis vectors
      top_k_projectors   : list of (d, d) ndarrays — projectors onto each plane
      top_k_rhos         : list of float — spectral radii (sorted descending)
      top_k_thetas       : list of float — rotation angles
      combined_rotation  : (d, d) ndarray — projector onto ALL rotation planes
      real_subspace      : (d, d) ndarray — projector onto 1×1 (non-rotating) subspace
      dim_rotation       : int — dimension of combined rotation subspace (2 * n_complex)
      dim_real           : int — dimension of real subspace
    """
    d = block_data["d"]

    # Sort 2×2 blocks by spectral radius, descending
    blocks_sorted = sorted(block_data["blocks_2x2"],
                           key=lambda b: b["rho"], reverse=True)
    k = min(top_k, len(blocks_sorted))

    top_k_planes = []
    top_k_projectors = []
    top_k_rhos = []
    top_k_thetas = []

    for j in range(k):
        plane = blocks_sorted[j]["plane"]   # (d, 2)
        P = plane @ plane.T                  # (d, d)
        top_k_planes.append(plane)
        top_k_projectors.append(P)
        top_k_rhos.append(blocks_sorted[j]["rho"])
        top_k_thetas.append(blocks_sorted[j]["theta"])

    # Combined projectors
    all_rot_vecs = []
    for b in block_data["blocks_2x2"]:
        all_rot_vecs.append(b["plane"][:, 0])
        all_rot_vecs.append(b["plane"][:, 1])

    if all_rot_vecs:
        V_rot = np.column_stack(all_rot_vecs)   # (d, 2*n_complex)
        P_rot = V_rot @ V_rot.T
    else:
        P_rot = np.zeros((d, d))

    all_real_vecs = [b["schur_vec"] for b in block_data["blocks_1x1"]]
    if all_real_vecs:
        V_real = np.column_stack(all_real_vecs)  # (d, n_real)
        P_real = V_real @ V_real.T
    else:
        P_real = np.zeros((d, d))

    return {
        "top_k_planes":      top_k_planes,
        "top_k_projectors":  top_k_projectors,
        "top_k_rhos":        top_k_rhos,
        "top_k_thetas":      top_k_thetas,
        "combined_rotation": P_rot,
        "real_subspace":     P_real,
        "dim_rotation":      2 * block_data["n_complex"],
        "dim_real":          block_data["n_real"],
    }


# ---------------------------------------------------------------------------
# Depth profile (per-layer models)
# ---------------------------------------------------------------------------

def rotation_depth_profile(ov_list: list, layer_names: list) -> dict:
    """
    Per-layer rotation angle and spectral radius distributions for GPT-2.

    For each layer's V_eff, extracts Schur blocks and computes summary
    statistics. Enables testing whether rotation angle correlates with
    the repulsive fraction depth gradient from Phase 2.

    Parameters
    ----------
    ov_list     : list of (d, d) ndarrays — per-layer V_eff matrices
    layer_names : list of str — layer identifiers

    Returns
    -------
    dict with:
      per_layer : list of dicts, each containing:
                    layer_name, n_real, n_complex,
                    theta_mean, theta_std, rho_mean, rho_std,
                    frac_expanding, frac_contracting,
                    rotational_fraction, henrici_relative
      summary   : dict with cross-layer statistics
    """
    per_layer = []

    for OV, name in zip(ov_list, layer_names):
        blocks = extract_schur_blocks(OV)
        angles = rotation_angle_stats(blocks)
        energy = rotation_energy_fractions(blocks)
        henrici = henrici_nonnormality(blocks)

        per_layer.append({
            "layer_name":          name,
            "n_real":              blocks["n_real"],
            "n_complex":           blocks["n_complex"],
            "theta_mean":          angles["theta_mean"],
            "theta_std":           angles["theta_std"],
            "rho_mean":            angles["rho_mean"],
            "rho_std":             angles["rho_std"],
            "frac_expanding":      angles["frac_expanding"],
            "frac_contracting":    angles["frac_contracting"],
            "rotational_fraction": energy["rotational_fraction"],
            "henrici_relative":    henrici["henrici_relative"],
        })

    # Cross-layer summary
    if per_layer:
        theta_means = [p["theta_mean"] for p in per_layer]
        rho_means = [p["rho_mean"] for p in per_layer]
        henrici_vals = [p["henrici_relative"] for p in per_layer]
        summary = {
            "theta_mean_across_layers": float(np.mean(theta_means)),
            "theta_std_across_layers":  float(np.std(theta_means)),
            "rho_mean_across_layers":   float(np.mean(rho_means)),
            "henrici_mean":             float(np.mean(henrici_vals)),
            "henrici_max":              float(np.max(henrici_vals)),
            "henrici_max_layer":        layer_names[int(np.argmax(henrici_vals))],
        }
    else:
        summary = {}

    return {"per_layer": per_layer, "summary": summary}


# ---------------------------------------------------------------------------
# Full pipeline for one model
# ---------------------------------------------------------------------------

def analyze_rotational_spectrum(
    ov_data: dict,
    top_k_planes: int = 32,
) -> dict:
    """
    Full Block 1a analysis for one model.

    Parameters
    ----------
    ov_data      : output of weights.extract_ov_circuit (or loaded from disk)
    top_k_planes : number of dominant rotation planes to extract

    Returns
    -------
    dict with per-layer (or single) block analysis, energy fractions,
    angle statistics, Henrici measure, rotation plane projectors,
    and (for per-layer models) the depth profile.
    """
    is_per_layer = ov_data["is_per_layer"]

    if is_per_layer:
        ov_list = ov_data["ov_total"]
        layer_names = ov_data["layer_names"]

        all_blocks = []
        all_angles = []
        all_energy = []
        all_henrici = []
        all_planes = []

        for OV in ov_list:
            blocks = extract_schur_blocks(OV)
            all_blocks.append(blocks)
            all_angles.append(rotation_angle_stats(blocks))
            all_energy.append(rotation_energy_fractions(blocks))
            all_henrici.append(henrici_nonnormality(blocks))
            all_planes.append(build_rotation_plane_projectors(blocks, top_k_planes))

        depth = rotation_depth_profile(ov_list, layer_names)

        return {
            "is_per_layer":   True,
            "layer_names":    layer_names,
            "blocks":         all_blocks,
            "angle_stats":    all_angles,
            "energy_fractions": all_energy,
            "henrici":        all_henrici,
            "plane_projectors": all_planes,
            "depth_profile":  depth,
        }
    else:
        OV = ov_data["ov_total"]
        blocks = extract_schur_blocks(OV)
        angles = rotation_angle_stats(blocks)
        energy = rotation_energy_fractions(blocks)
        henrici = henrici_nonnormality(blocks)
        planes = build_rotation_plane_projectors(blocks, top_k_planes)

        return {
            "is_per_layer":    False,
            "layer_names":     ["shared"],
            "blocks":          blocks,
            "angle_stats":     angles,
            "energy_fractions": energy,
            "henrici":         henrici,
            "plane_projectors": planes,
        }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def summary_to_json(result: dict) -> dict:
    """
    Extract JSON-serializable summary from analyze_rotational_spectrum output.
    Drops large arrays (schur_T, schur_Z, projector matrices, plane vectors).
    """
    if result["is_per_layer"]:
        layers = []
        for i, name in enumerate(result["layer_names"]):
            layers.append({
                "layer": name,
                "n_real": result["blocks"][i]["n_real"],
                "n_complex": result["blocks"][i]["n_complex"],
                **{k: v for k, v in result["angle_stats"][i].items()
                   if k not in ("thetas", "rhos", "signs")},
                **{k: v for k, v in result["energy_fractions"][i].items()},
                **{k: v for k, v in result["henrici"][i].items()},
            })
        return {
            "is_per_layer": True,
            "per_layer": layers,
            "depth_summary": result["depth_profile"]["summary"],
        }
    else:
        return {
            "is_per_layer": False,
            "n_real": result["blocks"]["n_real"],
            "n_complex": result["blocks"]["n_complex"],
            **{k: v for k, v in result["angle_stats"].items()
               if k not in ("thetas", "rhos", "signs")},
            **{k: v for k, v in result["energy_fractions"].items()},
            **{k: v for k, v in result["henrici"].items()},
        }
