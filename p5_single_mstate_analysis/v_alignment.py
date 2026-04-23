"""
v_alignment.py — Group B: paper-theoretical alignment.

Tests predictions of Geshkovski et al. at the level of one cluster.

What gets measured:
  - Effective β from attention softmax (regress logits on inner products)
  - Intra-cluster mass-near-1 trajectory vs Theorem 6.3 prediction
  - Energy E_β restricted to cluster pairs across layers
  - Centroid decomposition: attractive vs repulsive V-subspace
  - Cluster-mean displacement Δx̄ = x̄(L+1) − x̄(L): V-subspace projection
  - S/A local test: fraction of updates in rotational (antisymmetric) subspace
  - Merge-event geometry: fusion-direction alignment with V's subspaces
  - Rotational blocks intersecting the cluster direction
"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _project_energy(v: np.ndarray, basis: np.ndarray) -> float:
    """
    ||P_basis v||^2 where P_basis = U U^T. Works for basis shape (d, k).
    """
    if basis is None or basis.size == 0:
        return 0.0
    coef = basis.T @ v
    return float(coef @ coef)


def _split_vector(v: np.ndarray, U_att: np.ndarray, U_rep: np.ndarray) -> dict:
    """
    Decompose v into attractive + repulsive + orthogonal components.

    Returns
    -------
    dict with keys: attr_sq, rep_sq, orth_sq, total_sq,
                    attr_frac, rep_frac, orth_frac
    """
    total_sq = float(v @ v)
    attr_sq = _project_energy(v, U_att)
    rep_sq  = _project_energy(v, U_rep)
    orth_sq = max(0.0, total_sq - attr_sq - rep_sq)

    def frac(x):
        return float(x / total_sq) if total_sq > 0 else 0.0

    return {
        "attr_sq":  attr_sq,  "rep_sq":  rep_sq,  "orth_sq":  orth_sq,
        "total_sq": total_sq,
        "attr_frac": frac(attr_sq),
        "rep_frac":  frac(rep_sq),
        "orth_frac": frac(orth_sq),
    }


# ---------------------------------------------------------------------------
# Effective β estimate
# ---------------------------------------------------------------------------

def estimate_effective_beta(
    attentions_layer: np.ndarray,  # (n_heads, n, n)
    activations_layer: np.ndarray, # (n, d), unit vectors
    cluster_indices: np.ndarray,
) -> dict:
    """
    For the softmax in each head, attention weight A_ij ∝ exp(β x_i^T x_j)
    assuming x already absorbed Q/K and is L2-normed. Regress log A_ij on
    <x_i, x_j> restricted to cluster rows. Slope ≈ β_effective.

    Returns
    -------
    dict with per-head beta + cluster-mean
    """
    if len(cluster_indices) < 3:
        return {"per_head_beta": [], "cluster_mean_beta": float("nan"),
                "note": "cluster too small (<3) for regression"}

    X = activations_layer[cluster_indices]
    # Pairwise inner products among cluster members
    G = X @ X.T
    iu = np.triu_indices(len(cluster_indices), k=1)
    ips = G[iu]

    per_head = []
    for h in range(attentions_layer.shape[0]):
        A = attentions_layer[h, cluster_indices][:, cluster_indices]
        # Avoid log(0): clip
        A_safe = np.clip(A, 1e-12, None)
        log_A = np.log(A_safe)
        log_vals = log_A[iu]
        # Regress log_vals on ips (intercept absorbs softmax denominator)
        if np.std(ips) < 1e-6:
            per_head.append(float("nan"))
            continue
        slope = float(np.polyfit(ips, log_vals, 1)[0])
        per_head.append(slope)

    per_head_arr = np.array(per_head, dtype=np.float64)
    valid = per_head_arr[~np.isnan(per_head_arr)]
    return {
        "per_head_beta":     [round(b, 3) for b in per_head],
        "cluster_mean_beta": float(valid.mean()) if valid.size else float("nan"),
        "cluster_median_beta": float(np.median(valid)) if valid.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Energy restricted to cluster pairs
# ---------------------------------------------------------------------------

def cluster_energy_trajectory(
    activations: np.ndarray,  # (n_layers, n, d)
    hdb_labels: list,
    chain: list,
    beta: float = 1.0,
) -> list:
    """
    E_β over cluster-internal pairs, per layer.

    E_β(cluster) = (1/|C|^2) Σ_{i,j in C} exp(β <x_i, x_j>)
    """
    out = []
    for layer, cid in chain:
        if layer >= activations.shape[0]:
            break
        mask = hdb_labels[layer] == cid
        if mask.sum() < 2:
            out.append(float("nan"))
            continue
        X = activations[layer][mask]
        G = X @ X.T
        # Include diagonal (=1) as in the paper
        E = float(np.exp(beta * G).mean())
        out.append(E)
    return out


# ---------------------------------------------------------------------------
# Theorem 6.3 prediction (intra-cluster mass-near-1)
# ---------------------------------------------------------------------------

def theorem_6_3_prediction(n_cluster: int, d_model: int) -> dict:
    """
    Paper's Theorem 6.3 predicts the mass-near-1 proportion for n tokens in
    dimension d under the attention dynamics approaches a theoretical limit.
    Returns the limiting value + a note if n / d is outside the paper's regime.

    Approximation: for the simplified model (large β, attention-only),
    predicted limit is ≈ 1 - O(log n / d). We return a rough bound, not an
    exact value — Theorem 6.3's constants depend on the specific distribution
    of initial tokens.
    """
    if n_cluster < 2 or d_model < 2:
        return {"prediction": float("nan"), "note": "n or d too small"}

    # Rough predicted asymptotic mass-near-1: 1 - c * log(n)/d for some c.
    # We use c=1 as a reference; the interesting quantity is whether the
    # empirical trajectory approaches any stable value, not whether it hits
    # an exact predicted number.
    pred = 1.0 - np.log(n_cluster) / d_model
    return {
        "prediction":  round(float(pred), 4),
        "note":        "rough upper bound; exact constant is distribution-dependent",
    }


# ---------------------------------------------------------------------------
# Centroid and displacement V-eigenspace decomposition
# ---------------------------------------------------------------------------

def centroid_subspace_trajectory(
    centroid_coords: np.ndarray,    # (n_alive, d)
    U_att: np.ndarray,
    U_rep: np.ndarray,
) -> list:
    """
    For each centroid in the trajectory, decompose into attractive / repulsive
    components of V.
    """
    out = []
    for k, c in enumerate(centroid_coords):
        decomp = _split_vector(c, U_att, U_rep)
        decomp["step"] = int(k)
        out.append(decomp)
    return out


def displacement_subspace_trajectory(
    centroid_coords: np.ndarray,
    U_att: np.ndarray,
    U_rep: np.ndarray,
) -> list:
    """
    Decompose Δx̄ = x̄(k+1) - x̄(k) (on the ambient space, not the sphere)
    into attractive / repulsive components. Captures where the update is
    pushing the cluster.
    """
    out = []
    for k in range(1, centroid_coords.shape[0]):
        dx = centroid_coords[k] - centroid_coords[k - 1]
        decomp = _split_vector(dx, U_att, U_rep)
        decomp["step"] = int(k - 1)
        out.append(decomp)
    return out


# ---------------------------------------------------------------------------
# S/A (rotational neutrality) local test
# ---------------------------------------------------------------------------

def rotational_local_test(
    centroid_coords: np.ndarray,
    V_sym: np.ndarray,
    V_asym: np.ndarray,
) -> dict:
    """
    Project the cluster centroids and their displacement through V_sym and
    V_asym separately. Compare the magnitudes: if ||V_asym · x̄|| is small
    relative to ||V_sym · x̄||, the cluster remains rotation-neutral locally.

    Returns per-step dict with sym_norm, asym_norm, ratio.
    """
    if V_sym is None or V_asym is None:
        return {"available": False, "per_step": []}

    per_step = []
    for k in range(centroid_coords.shape[0]):
        x = centroid_coords[k]
        s_norm = float(np.linalg.norm(V_sym @ x))
        a_norm = float(np.linalg.norm(V_asym @ x))
        total = s_norm + a_norm
        per_step.append({
            "step":       int(k),
            "sym_norm":   s_norm,
            "asym_norm":  a_norm,
            "asym_frac":  float(a_norm / total) if total > 0 else 0.0,
        })

    asym_fracs = [p["asym_frac"] for p in per_step]
    verdict = (
        "locally_rotation_neutral" if np.mean(asym_fracs) < 0.1
        else "locally_rotational"
    )
    return {
        "available":       True,
        "per_step":        per_step,
        "mean_asym_frac":  float(np.mean(asym_fracs)) if asym_fracs else 0.0,
        "verdict":         verdict,
    }


# ---------------------------------------------------------------------------
# Merge-event geometry
# ---------------------------------------------------------------------------

def merge_event_geometry(
    centroid_primary: np.ndarray,     # (d,) pre-merge centroid of the primary
    centroid_sibling: np.ndarray,     # (d,) pre-merge centroid of the sibling
    centroid_fused: np.ndarray,       # (d,) post-merge centroid (at layer_to)
    U_att: np.ndarray,
    U_rep: np.ndarray,
) -> dict:
    """
    At the merge layer, compute:
      - angle between pre-merge centroids
      - fusion direction = unit((c_fused - (c_primary + c_sibling) / 2))
        (i.e. the direction the fused centroid differs from the midpoint)
      - cosine of fusion direction against attractive and repulsive
        V-subspaces (via projection energy normalized by ||fusion_dir||^2)
    """
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    c_p = _unit(centroid_primary)
    c_s = _unit(centroid_sibling)
    c_f = _unit(centroid_fused)

    dot_ps = float(np.clip(c_p @ c_s, -1.0, 1.0))
    angle_ps = float(np.arccos(dot_ps))

    midpoint = _unit((c_p + c_s) / 2)
    fusion_dir_raw = c_f - midpoint
    fusion_dir_norm = float(np.linalg.norm(fusion_dir_raw))
    fusion_dir = (fusion_dir_raw / fusion_dir_norm
                  if fusion_dir_norm > 1e-12 else fusion_dir_raw)

    # Cosine-style alignment: ||P_subspace · fusion_dir||
    attr_sq = _project_energy(fusion_dir, U_att)
    rep_sq  = _project_energy(fusion_dir, U_rep)

    if attr_sq > rep_sq:
        verdict = "fusion_attractive_dominant"
    elif rep_sq > attr_sq:
        verdict = "fusion_repulsive_dominant"
    else:
        verdict = "fusion_orthogonal_or_mixed"

    return {
        "pre_merge_angle_rad":    round(angle_ps, 4),
        "pre_merge_cosine":       round(dot_ps, 4),
        "fusion_dir_magnitude":   round(fusion_dir_norm, 4),
        "fusion_attr_alignment":  round(float(np.sqrt(attr_sq)), 4),
        "fusion_rep_alignment":   round(float(np.sqrt(rep_sq)), 4),
        "verdict":                verdict,
    }


# ---------------------------------------------------------------------------
# Schur block overlap
# ---------------------------------------------------------------------------

def schur_block_overlap(
    centroid_direction: np.ndarray,   # (d,) representative cluster direction
    schur_Z: np.ndarray,              # (d, d) Schur orthogonal basis
    schur_T: np.ndarray,              # (d, d) upper-triangular-plus-2x2 blocks
    lda_direction: np.ndarray = None,
    top_k: int = 8,
) -> list:
    """
    Identify 2×2 rotational Schur blocks with highest overlap with the
    cluster-centroid direction and with the LDA direction.

    Schur structure: T is upper-triangular with 2×2 blocks on the diagonal
    corresponding to complex eigenvalue pairs. A block at rows [i, i+1] spans
    an invariant 2D plane defined by Z[:, i], Z[:, i+1].
    """
    if schur_Z is None or schur_T is None:
        return []

    d = schur_T.shape[0]
    blocks = []
    i = 0
    while i < d:
        # 2x2 block if subdiagonal element is nonzero
        if i + 1 < d and abs(schur_T[i + 1, i]) > 1e-8:
            z1, z2 = schur_Z[:, i], schur_Z[:, i + 1]
            # Overlap = projection of centroid_direction onto span(z1, z2)
            a = float(z1 @ centroid_direction)
            b = float(z2 @ centroid_direction)
            overlap_c = float(np.sqrt(a * a + b * b))
            if lda_direction is not None:
                la = float(z1 @ lda_direction)
                lb = float(z2 @ lda_direction)
                overlap_l = float(np.sqrt(la * la + lb * lb))
            else:
                overlap_l = None
            # Real/imag parts of the block eigenvalue
            block = schur_T[i:i + 2, i:i + 2]
            tr = float(np.trace(block))
            det = float(np.linalg.det(block))
            disc = tr * tr - 4 * det
            if disc < 0:
                eig_real = tr / 2
                eig_imag = float(np.sqrt(-disc) / 2)
            else:
                eig_real, eig_imag = tr / 2, 0.0
            blocks.append({
                "start_idx":      i,
                "overlap_c":      round(overlap_c, 4),
                "overlap_lda":    round(overlap_l, 4) if overlap_l is not None else None,
                "eig_real":       round(eig_real, 4),
                "eig_imag":       round(eig_imag, 4),
            })
            i += 2
        else:
            i += 1

    blocks.sort(key=lambda b: b["overlap_c"], reverse=True)
    return blocks[:top_k]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_v_alignment(
    activations: np.ndarray,
    attentions: np.ndarray,
    hdb_labels: list,
    trajectory: dict,
    sibling_trajectory: dict,
    centroid_coords: np.ndarray,
    sibling_centroid_coords: np.ndarray,
    v_projectors: dict,
    phase2i: dict = None,
    d_model: int = None,
    beta: float = 1.0,
) -> dict:
    """
    Full Group B analysis for one trajectory.

    Parameters
    ----------
    v_projectors : output of io.load_phase2_projectors
    phase2i      : output of io.load_phase2i (or empty dict)
    """
    U_att = v_projectors.get("U_att")
    U_rep = v_projectors.get("U_rep")
    if d_model is None:
        d_model = activations.shape[-1]

    chain = trajectory["chain"]

    # Per-layer beta estimate
    beta_per_layer = []
    for layer, cid in chain:
        if layer >= attentions.shape[0] or layer >= activations.shape[0]:
            continue
        idx = np.where(hdb_labels[layer] == cid)[0]
        beta_est = estimate_effective_beta(
            attentions[layer], activations[layer], idx,
        )
        beta_per_layer.append({
            "layer":  int(layer),
            "mean":   round(beta_est["cluster_mean_beta"], 3)
                      if not np.isnan(beta_est["cluster_mean_beta"]) else None,
            "median": round(beta_est["cluster_median_beta"], 3)
                      if not np.isnan(beta_est["cluster_median_beta"]) else None,
        })

    # Mass-near-1 trajectory (this is restricted to cluster pairs; separate
    # from the global mass_near_1 in the layer metrics)
    mass_trajectory = []
    for layer, cid in chain:
        if layer >= activations.shape[0]:
            break
        mask = hdb_labels[layer] == cid
        n = int(mask.sum())
        if n < 2:
            mass_trajectory.append({"layer": int(layer), "n": n,
                                     "mass_near_1": None})
            continue
        X = activations[layer][mask]
        G = X @ X.T
        iu = np.triu_indices(n, k=1)
        mass = float((G[iu] > 0.95).mean())
        mass_trajectory.append({
            "layer": int(layer), "n": n,
            "mass_near_1": round(mass, 4),
            "theorem_6_3_pred": theorem_6_3_prediction(n, d_model)["prediction"],
        })

    # Energy trajectory
    energy_traj = cluster_energy_trajectory(
        activations, hdb_labels, chain, beta=beta,
    )

    # Centroid subspace decomposition
    centroid_decomp = centroid_subspace_trajectory(centroid_coords, U_att, U_rep)
    # Displacement subspace decomposition
    disp_decomp = displacement_subspace_trajectory(centroid_coords, U_att, U_rep)

    # S/A local test
    sa_test = rotational_local_test(
        centroid_coords,
        phase2i.get("V_sym") if phase2i else None,
        phase2i.get("V_asym") if phase2i else None,
    )

    # Schur block overlap: representative direction = mean centroid
    rep_direction = centroid_coords.mean(axis=0)
    rep_direction = rep_direction / max(float(np.linalg.norm(rep_direction)), 1e-12)
    schur_blocks = schur_block_overlap(
        rep_direction,
        phase2i.get("schur_Z") if phase2i else None,
        phase2i.get("schur_T") if phase2i else None,
    )

    # Merge-event geometry
    merge_geom = None
    merge_ev = trajectory.get("merge_event")
    if (merge_ev is not None and sibling_centroid_coords is not None
            and sibling_centroid_coords.size > 0):
        lf = merge_ev["layer_from"]
        lt = merge_ev["layer_to"]
        # Build layer → index maps that mirror _centroid_coords: only layers
        # where the cluster was non-empty produce a centroid row, so we must
        # skip the same layers here rather than using chain position directly.
        valid_primary = []
        for layer, cid in chain:
            if layer >= activations.shape[0]:
                break
            if (hdb_labels[layer] == cid).sum() >= 1:
                valid_primary.append(layer)
        chain_dict = {l: k for k, l in enumerate(valid_primary)}

        sib_chain = sibling_trajectory["chain"]
        valid_sibling = []
        for layer, cid in sib_chain:
            if layer >= activations.shape[0]:
                break
            if (hdb_labels[layer] == cid).sum() >= 1:
                valid_sibling.append(layer)
        sib_chain_dict = {l: k for k, l in enumerate(valid_sibling)}

        if (lf in chain_dict and lf in sib_chain_dict
                and lt in chain_dict):
            merge_geom = merge_event_geometry(
                centroid_coords[chain_dict[lf]],
                sibling_centroid_coords[sib_chain_dict[lf]],
                centroid_coords[chain_dict[lt]],
                U_att, U_rep,
            )

    # Summary verdict
    attr_fracs = [d["attr_frac"] for d in centroid_decomp]
    rep_fracs  = [d["rep_frac"]  for d in centroid_decomp]
    summary = {
        "mean_centroid_attr_frac": round(float(np.mean(attr_fracs)), 4) if attr_fracs else None,
        "mean_centroid_rep_frac":  round(float(np.mean(rep_fracs)),  4) if rep_fracs else None,
        "mean_beta_estimate":      round(float(np.nanmean(
            [b["mean"] for b in beta_per_layer if b["mean"] is not None]
        )), 3) if any(b["mean"] is not None for b in beta_per_layer) else None,
        "mean_cluster_energy":     round(float(np.nanmean(energy_traj)), 4)
                                   if energy_traj else None,
        "rotational_neutral_local": sa_test.get("verdict"),
    }

    return {
        "trajectory_id":       int(trajectory["id"]),
        "beta_per_layer":      beta_per_layer,
        "mass_trajectory":     mass_trajectory,
        "energy_trajectory":   [round(e, 4) if not np.isnan(e) else None
                                for e in energy_traj],
        "centroid_decomp":     centroid_decomp,
        "displacement_decomp": disp_decomp,
        "sa_local_test":       sa_test,
        "schur_blocks":        schur_blocks,
        "merge_geometry":      merge_geom,
        "summary":              summary,
    }


def save_v_alignment(result: dict, out_dir: Path, tag: str = "primary") -> None:
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"group_B_v_alignment_{tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")
