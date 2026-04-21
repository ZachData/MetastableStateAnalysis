"""
analysis.py — Cross-track comparison and alignment tests.

Pulls results from Tracks 1–3 and Phase 3's existing cross-phase
analyses into a unified picture. Determines which track(s) found
cluster–feature alignment, quantifies agreement between tracks,
and produces the final verdict for Phase 4.

Also handles saving outputs structured for Phase 5/6 consumption.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Cross-track agreement
# ---------------------------------------------------------------------------

def cross_track_agreement(
    track1_results: dict,
    track2_results: dict,
    track3_results: Optional[dict] = None,
) -> dict:
    """
    Compare findings across tracks. The key comparisons:

    - Track 1 (crosscoder activations) vs Track 2 (direct geometry):
      Do the layers where crosscoder features track clusters match
      the layers where LDA probes achieve high accuracy?

    - Track 2 (probes) vs Track 3 (low-rank AE):
      Do probe directions and AE bottleneck directions span the
      same subspace?

    - Track 1 chorus ARI vs Track 2 probe accuracy:
      Correlation across layers. High correlation = crosscoder is
      capturing the same structure as direct geometry.

    Parameters
    ----------
    track1_results : from activation_trajectories + chorus
    track2_results : from geometric
    track3_results : from low_rank_ae (optional)

    Returns
    -------
    dict with agreement metrics and interpretation
    """
    agreement = {}

    # --- Track 1 vs Track 2: per-layer comparison ---
    t1_per_layer = track1_results.get("chorus_per_layer", {})
    t2_per_layer = track2_results.get("probe_per_layer", {})

    common_layers = sorted(
        set(t1_per_layer.keys()) & set(t2_per_layer.keys()),
        key=lambda x: int(x) if isinstance(x, (int, str)) else 0,
    )

    if common_layers:
        t1_scores = []
        t2_scores = []
        for layer in common_layers:
            t1_ari = _extract_ari(t1_per_layer[layer])
            t2_acc = _extract_accuracy(t2_per_layer[layer])
            if t1_ari is not None and t2_acc is not None:
                t1_scores.append(t1_ari)
                t2_scores.append(t2_acc)

        if len(t1_scores) >= 3:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(t1_scores, t2_scores)
            agreement["t1_t2_correlation"] = {
                "spearman_rho": float(rho),
                "pval": float(pval),
                "n_layers": len(t1_scores),
                "interpretation": (
                    "strong_agreement" if rho > 0.7 and pval < 0.05
                    else "moderate_agreement" if rho > 0.4 and pval < 0.1
                    else "weak_or_no_agreement"
                ),
            }
        else:
            agreement["t1_t2_correlation"] = {
                "error": "fewer than 3 common layers with both scores"
            }
    else:
        agreement["t1_t2_correlation"] = {
            "error": "no common layers between tracks"
        }

    # --- Track 2 vs Track 3: subspace overlap ---
    if track3_results:
        t2_dirs = track2_results.get("probe_directions", {})
        t3_dirs = track3_results.get("bottleneck_directions")

        if t2_dirs and t3_dirs is not None:
            # Measure principal angle between probe subspace and AE subspace
            overlap = _subspace_overlap(t2_dirs, t3_dirs)
            agreement["t2_t3_subspace"] = overlap
        else:
            agreement["t2_t3_subspace"] = {"error": "missing directions"}

    # --- V-alignment comparison across tracks ---
    t2_v = track2_results.get("v_alignment", {})
    t3_v = track3_results.get("v_alignment", {}) if track3_results else {}

    if t2_v and t3_v:
        agreement["v_alignment_comparison"] = {
            "track2_mean_repulsive": t2_v.get("mean_repulsive", 0),
            "track2_mean_attractive": t2_v.get("mean_attractive", 0),
            "track3_mean_repulsive": t3_v.get("mean_repulsive", 0),
            "track3_mean_attractive": t3_v.get("mean_attractive", 0),
        }

    return agreement


def _extract_ari(layer_result: dict) -> Optional[float]:
    """Pull ARI from a chorus layer result."""
    if isinstance(layer_result, dict):
        ari = layer_result.get("ari")
        if isinstance(ari, dict):
            return ari.get("ari")
        return ari
    return None


def _extract_accuracy(layer_result: dict) -> Optional[float]:
    """Pull accuracy from a probe layer result."""
    if isinstance(layer_result, dict):
        return layer_result.get("accuracy")
    return None


def _subspace_overlap(
    probe_dirs: dict,
    ae_dirs: np.ndarray,
) -> dict:
    """
    Compute principal angles between the probe direction subspace
    and the AE bottleneck subspace.

    Uses the first layer where probes exist as representative.
    """
    # Get a representative probe direction matrix
    for layer, W in sorted(probe_dirs.items()):
        if isinstance(W, np.ndarray) and W.ndim == 2:
            break
    else:
        return {"error": "no valid probe directions"}

    # W is (n_classes, d), ae_dirs is (r, d_total) or (r, L*d)
    # They may live in different spaces. If so, just compare within
    # the per-layer slice.
    d_probe = W.shape[1]
    d_ae = ae_dirs.shape[1]

    if d_probe != d_ae:
        # ae_dirs is in L*d space; take first d_model slice
        ae_slice = ae_dirs[:, :d_probe]
    else:
        ae_slice = ae_dirs

    # Orthonormalize both
    try:
        Q1, _ = np.linalg.qr(W.T)      # (d, k1)
        Q2, _ = np.linalg.qr(ae_slice.T)  # (d, k2)
    except np.linalg.LinAlgError:
        return {"error": "QR decomposition failed"}

    # SVD of Q1^T Q2 gives cosines of principal angles
    M = Q1.T @ Q2
    svd_vals = np.linalg.svd(M, compute_uv=False)

    # Clip to [0, 1]
    svd_vals = np.clip(svd_vals, 0, 1)
    angles_deg = np.degrees(np.arccos(svd_vals))

    return {
        "principal_angles_deg": angles_deg.tolist(),
        "mean_angle_deg": float(np.mean(angles_deg)),
        "min_angle_deg": float(np.min(angles_deg)),
        "grassmann_distance": float(np.sqrt(np.sum(np.arccos(svd_vals) ** 2))),
        "subspace_overlap_score": float(np.mean(svd_vals ** 2)),
    }


# ---------------------------------------------------------------------------
# 2. Phase 4 verdict
# ---------------------------------------------------------------------------

def build_phase4_verdict(
    track1_results: dict,
    track2_results: dict,
    track3_results: Optional[dict],
    agreement: dict,
    plateau_alignment: dict,
) -> dict:
    """
    Synthesize all Phase 4 results into a verdict.

    The falsification criterion: feature plateaus don't align with
    cluster count plateaus → features aren't tracking metastable
    configurations. Applies to all three tracks independently.

    Returns
    -------
    dict with per-track verdicts and overall assessment
    """
    verdict = {"tracks": {}}

    # --- Track 1: crosscoder activation patterns ---
    t1 = {}
    chorus = track1_results.get("chorus_summary", {})
    t1["chorus_mean_ari"] = chorus.get("mean_ari", 0.0)
    t1["chorus_mean_purity"] = chorus.get("mean_purity", 0.0)

    mi = track1_results.get("mi_summary", {})
    t1["max_nmi"] = mi.get("max_nmi", 0.0)
    t1["mean_nmi"] = mi.get("mean_nmi", 0.0)

    pa = plateau_alignment
    t1["plateau_alignment_rate"] = pa.get("alignment_rate", 0.0)
    t1["plateau_falsification"] = pa.get("falsification", "untestable")

    # Interpret
    if t1["chorus_mean_ari"] > 0.2 or t1["max_nmi"] > 0.3:
        t1["verdict"] = "signal_detected"
    elif t1["chorus_mean_ari"] > 0.05 or t1["max_nmi"] > 0.1:
        t1["verdict"] = "weak_signal"
    else:
        t1["verdict"] = "null"

    verdict["tracks"]["track1_crosscoder"] = t1

    # --- Track 2: direct geometry ---
    t2 = {}
    probes = track2_results.get("probe_summary", {})
    t2["mean_probe_accuracy"] = probes.get("mean_accuracy", 0.0)
    t2["max_probe_accuracy"] = probes.get("max_accuracy", 0.0)

    lda = track2_results.get("lda_summary", {})
    t2["lda_mean_cosine_stability"] = lda.get("mean_cosine", 0.0)

    deltas = track2_results.get("delta_pca_summary", {})
    t2["mean_update_variance"] = deltas.get("mean_total_variance", 0.0)

    if t2["mean_probe_accuracy"] > 0.7:
        t2["verdict"] = "strong_linear_encoding"
    elif t2["mean_probe_accuracy"] > 0.4:
        t2["verdict"] = "moderate_linear_encoding"
    else:
        t2["verdict"] = "weak_or_no_linear_encoding"

    verdict["tracks"]["track2_geometric"] = t2

    # --- Track 3: low-rank AE ---
    if track3_results:
        t3 = {}
        v_align = track3_results.get("v_alignment", {})
        t3["mean_repulsive"] = v_align.get("mean_repulsive", 0.0)
        t3["mean_attractive"] = v_align.get("mean_attractive", 0.0)
        t3["n_repulsive_dominant"] = v_align.get("n_repulsive_dominant", 0)
        t3["n_attractive_dominant"] = v_align.get("n_attractive_dominant", 0)

        recon = track3_results.get("reconstruction", {})
        t3["mean_recon_ratio"] = float(np.mean([
            v.get("ratio", 1.0) for v in recon.values()
        ])) if recon else 1.0

        # Does removing sparsity recover V-alignment?
        rep = t3["mean_repulsive"]
        att = t3["mean_attractive"]
        if max(rep, att) > 0.1 and abs(rep - att) > 0.05:
            t3["verdict"] = "v_alignment_recovered"
        else:
            t3["verdict"] = "v_alignment_still_null"

        verdict["tracks"]["track3_low_rank_ae"] = t3

    # --- Cross-track agreement ---
    verdict["agreement"] = agreement

    # --- Overall ---
    track_verdicts = [
        v.get("verdict", "null")
        for v in verdict["tracks"].values()
    ]

    if any("signal" in v or "strong" in v or "recovered" in v
           for v in track_verdicts):
        verdict["overall"] = "metastable_features_detected"
    elif any("moderate" in v or "weak_signal" in v for v in track_verdicts):
        verdict["overall"] = "partial_signal"
    else:
        verdict["overall"] = "cross_track_null"
        verdict["interpretation"] = (
            "Dynamical structure from Phases 1-2 is real but doesn't "
            "organize the representation at a level accessible to "
            "dictionary learning. Metastability is a property of the "
            "bulk geometry that doesn't decompose into feature-level units."
        )

    return verdict


# ---------------------------------------------------------------------------
# 3. Save outputs for Phase 5/6
# ---------------------------------------------------------------------------

def save_phase4_outputs(
    out_dir: Path,
    track1_results: dict,
    track2_results: dict,
    track3_results: Optional[dict],
    verdict: dict,
    activations_per_layer: Optional[dict] = None,
    hdbscan_labels: Optional[dict] = None,
):
    """
    Save Phase 4 outputs structured for downstream consumption.

    Phase 5 needs:
      - cluster identity feature sets per plateau layer
      - LDA directions per plateau layer

    Phase 6 needs:
      - cluster centroids in residual stream space per (prompt, layer, cluster)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Verdict ---
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=_json_default)

    # --- Track 1 results ---
    t1_dir = out_dir / "track1"
    t1_dir.mkdir(exist_ok=True)

    # MI results (cluster identity features per layer)
    mi = track1_results.get("mi_results", {})
    if mi:
        with open(t1_dir / "feature_cluster_mi.json", "w") as f:
            json.dump(mi, f, indent=2, default=_json_default)

    # Chorus results
    chorus = track1_results.get("chorus_results", {})
    if chorus:
        with open(t1_dir / "chorus.json", "w") as f:
            json.dump(chorus, f, indent=2, default=_json_default)

    # Feature plateaus
    fp = track1_results.get("feature_plateaus", {})
    if fp:
        with open(t1_dir / "feature_plateaus.json", "w") as f:
            json.dump(fp, f, indent=2, default=_json_default)

    # Merge dynamics
    md = track1_results.get("merge_dynamics", {})
    if md:
        with open(t1_dir / "merge_dynamics.json", "w") as f:
            json.dump(md, f, indent=2, default=_json_default)

    # Plateau alignment (falsification test)
    pa = track1_results.get("plateau_alignment", {})
    if pa:
        with open(t1_dir / "plateau_alignment.json", "w") as f:
            json.dump(pa, f, indent=2, default=_json_default)

    # --- Track 2 results ---
    t2_dir = out_dir / "track2"
    t2_dir.mkdir(exist_ok=True)

    # LDA directions (npz for Phase 5)
    lda = track2_results.get("lda_results", {})
    lda_dirs_to_save = {}
    lda_meta = {}
    for layer_key, layer_data in lda.get("per_layer", {}).items():
        direction = layer_data.get("direction")
        if direction is not None:
            lda_dirs_to_save[f"layer_{layer_key}"] = direction
            lda_meta[str(layer_key)] = {
                "accuracy": layer_data.get("accuracy"),
                "n_classes": layer_data.get("n_classes"),
            }
    if lda_dirs_to_save:
        np.savez(t2_dir / "lda_directions.npz", **lda_dirs_to_save)
        with open(t2_dir / "lda_meta.json", "w") as f:
            json.dump(lda_meta, f, indent=2)

    # Probe results
    probes = track2_results.get("probe_results", {})
    probe_dirs = track2_results.get("probe_directions", {})
    if probes:
        with open(t2_dir / "probe_accuracy.json", "w") as f:
            json.dump(
                {str(k): v for k, v in probes.get("per_layer", {}).items()},
                f, indent=2, default=_json_default,
            )
    if probe_dirs:
        dirs_to_save = {
            f"layer_{k}": v for k, v in probe_dirs.items()
            if isinstance(v, np.ndarray)
        }
        if dirs_to_save:
            np.savez(t2_dir / "probe_directions.npz", **dirs_to_save)

    # Delta PCA
    delta_pca = track2_results.get("delta_pca_results", {})
    if delta_pca:
        with open(t2_dir / "delta_pca.json", "w") as f:
            json.dump(delta_pca, f, indent=2, default=_json_default)

    # --- Track 3 results ---
    if track3_results:
        t3_dir = out_dir / "track3"
        t3_dir.mkdir(exist_ok=True)

        v_align = track3_results.get("v_alignment", {})
        if v_align:
            with open(t3_dir / "v_alignment.json", "w") as f:
                json.dump(v_align, f, indent=2, default=_json_default)

        recon = track3_results.get("reconstruction", {})
        if recon:
            with open(t3_dir / "reconstruction.json", "w") as f:
                json.dump(recon, f, indent=2, default=_json_default)

        ae_dirs = track3_results.get("bottleneck_directions")
        if ae_dirs is not None:
            np.savez(t3_dir / "bottleneck_directions.npz",
                     directions=ae_dirs)

    # --- Phase 6: cluster centroids ---
    if activations_per_layer and hdbscan_labels:
        centroids_dir = out_dir / "centroids"
        centroids_dir.mkdir(exist_ok=True)
        _save_cluster_centroids(
            centroids_dir, activations_per_layer, hdbscan_labels
        )

    print(f"  [phase4] Results saved to {out_dir}")


def _save_cluster_centroids(
    out_dir: Path,
    activations_per_layer: dict,
    hdbscan_labels: dict,
):
    """
    Compute and save cluster centroids in residual stream space
    at each layer, for Phase 6 tuned lens consumption.

    Saved as npz keyed by prompt_layer_clusterN.
    """
    centroids = {}
    meta = {}

    for prompt_key, prompt_labels in hdbscan_labels.items():
        for layer_key, labels in prompt_labels.items():
            try:
                layer = int(layer_key.replace("layer_", ""))
            except (ValueError, AttributeError):
                continue

            # Find activations for this prompt/layer
            acts = activations_per_layer.get(prompt_key, {}).get(layer)
            if acts is None:
                continue

            labels_arr = np.array(labels)
            if len(labels_arr) != acts.shape[0]:
                continue

            clusters = set(labels_arr[labels_arr >= 0].tolist())
            for c in clusters:
                mask = labels_arr == c
                centroid = acts[mask].mean(axis=0)
                key = f"{prompt_key}_layer{layer}_cluster{c}"
                centroids[key] = centroid
                meta[key] = {
                    "prompt": prompt_key,
                    "layer": layer,
                    "cluster_id": int(c),
                    "n_tokens": int(mask.sum()),
                }

    if centroids:
        np.savez(out_dir / "cluster_centroids.npz", **centroids)
        with open(out_dir / "centroid_meta.json", "w") as f:
            json.dump(meta, f, indent=2)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
