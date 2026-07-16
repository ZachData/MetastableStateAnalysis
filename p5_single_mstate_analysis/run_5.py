"""
run_5.py — Phase 5 case study CLI.

Loads artifacts from Phases 1–4, selects a primary trajectory and sibling,
runs the requested analysis groups (A through G), writes per-group JSON
fragments to a run directory, then calls report.py to emit a flat text
report.

Usage
-----
  python -m phase5_case.run --phase1-dir --phase2-dir --phase2i-dir --phase4-dir

  python -m phase5_case.run \\
      --model albert-xlarge-v2 \\
      --groups A B C1 C2 D G \\
      --out results/phase5/albert_xlarge_v2_sullivan

Directory layout inferred from core.config if flags are omitted:
  phase1-dir : results/phase1
  phase2-dir : results/phase2
  phase2i-dir: results/phase2i   (kept for CLI compat; no NPZ files used)
  phase3-dir : checkpoints/<model>/final
  phase3-cache: activation_cache/<model>
  phase4-dir : results/phase4

--force-prompt and --force-trajectory-id override the default rank-0 pick.
--runner-up-rank changes which ranked trajectory is reserved for replication.

Group F (causal) and Group E (tuned-lens) load the actual model and are
substantially slower; they're opt-in via the --groups flag.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from p6_subspace.subspace_build import build_global_projectors
from p5_single_mstate_analysis.v_alignment import (
    compute_v_alignment,
    estimate_effective_beta,
    theorem_6_3_prediction,
    rotational_local_test,
    merge_event_geometry,
    schur_block_overlap,
    save_v_alignment,
)

from . import constants as C
from . import io_p5 as p5io
from .select_cluster     import select_primary_and_sibling, save_selection
from .cluster_profile    import compute_profile, save_profile
from .v_alignment        import compute_v_alignment, save_v_alignment
from .head_contributions import analyze_heads, save_head_contributions
from .ffn_contributions  import analyze_ffn, save_ffn_contributions, load_ffn_deltas
from .feature_signature  import analyze_features, save_feature_signature
from .sibling_contrast   import run_sibling_contrast, save_sibling_contrast
from .report             import write_report

ALL_GROUPS = ["A", "B", "C1", "C2", "D", "E", "F", "G"]


# ---------------------------------------------------------------------------
# Trajectory lookup helpers
# ---------------------------------------------------------------------------

def _traj_by_id(trajs: list, tid: int) -> dict:
    for t in trajs:
        if int(t["id"]) == int(tid):
            return t
    raise RuntimeError(f"Trajectory id={tid} not found in phase1 output")


def _centroid_coords(trajectory: dict, centroid_trajs: dict,
                     activations: np.ndarray, hdb_labels: list) -> np.ndarray:
    """
    Prefer the pre-computed centroid trajectory from Phase 1; reconstruct
    from activations if not stored.
    """
    tid = int(trajectory["id"])
    if tid in centroid_trajs:
        return centroid_trajs[tid].astype(np.float32)
    coords = []
    for layer, cid in trajectory["chain"]:
        if layer >= activations.shape[0]:
            break
        mask = hdb_labels[layer] == cid
        if mask.sum() < 1:
            continue
        c = activations[layer][mask].mean(axis=0)
        n = float(np.linalg.norm(c))
        coords.append(c / max(n, 1e-12))
    return np.stack(coords).astype(np.float32) if coords else np.zeros((0, 0))


# ---------------------------------------------------------------------------
# Feature activations on demand
# ---------------------------------------------------------------------------

def _compute_feature_activations(phase3: dict, prompt_key: str,
                                   layers_needed: list) -> dict:
    """
    Crosscoder feature activations for one prompt, broadcast to every
    requested layer.

    The crosscoder (crosscoder.py) has no per-layer *encoder* — it encodes
    one joint (n_tokens, n_features) feature matrix per prompt from the
    concatenation of all its sampled layers ("input is the concatenation of
    residual stream activations across L_sampled layers"). Only the
    *decoder* is per-layer. So there is nothing to fetch "for layer L" on
    the encoder side: get the prompt's full stack the same way every other
    Phase 3/4 consumer does (get_stacked_tensor — see activation_trajectories.py,
    geometric.py, low_rank_ae.py), run the crosscoder once, and hand the
    resulting matrix to analyze_features under every layer key it asks for.
    analyze_features then correlates that one matrix against each layer's
    own HDBSCAN cluster mask to see where in depth each feature tracks
    cluster identity.
    """
    store = phase3.get("prompt_store")
    cc    = phase3.get("crosscoder")
    if store is None or cc is None:
        return {}
    try:
        import torch
    except ImportError:
        return {}

    try:
        x = store.get_stacked_tensor(prompt_key)   # (n_tokens, n_layers, d_model)
    except Exception as e:
        print(f"  [D] prompt_store.get_stacked_tensor failed for "
              f"'{prompt_key}': {e}")
        return {}
    if x is None:
        return {}

    device = next(cc.parameters()).device
    try:
        with torch.no_grad():
            x = x.to(device=device, dtype=torch.float32)
            if hasattr(cc, "forward"):
                out = cc(x)
                z = out["z"] if isinstance(out, dict) and "z" in out else out
            elif hasattr(cc, "encode"):
                z = cc.encode(x.reshape(x.shape[0], -1))
            else:
                return {}
    except Exception as e:
        print(f"  [D] crosscoder forward failed for '{prompt_key}': {e}")
        return {}

    feats = z.cpu().numpy()
    return {int(L): feats for L in layers_needed}


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _extract_centroid(activations_per_layer: list, labels_per_layer: list,
                       layer_idx: int, cluster_id: int) -> Optional[np.ndarray]:
    if layer_idx < 0 or layer_idx >= len(activations_per_layer):
        return None
    X      = activations_per_layer[layer_idx]
    labels = labels_per_layer[layer_idx]
    mask   = labels == cluster_id
    if mask.sum() < 1:
        return None
    return X[mask].mean(axis=0)


def _centroid_sequence(activations_per_layer: list, labels_per_layer: list,
                        trajectory: dict) -> tuple[np.ndarray, list[int]]:
    """
    Extract the centroid at each layer where the trajectory is alive.
    Silently skips layers where the cluster has no members.
    Returns (centroids (n_alive, d), layer_idxs).
    """
    centroids, idxs = [], []
    for layer_idx, cluster_id in trajectory["chain"]:
        c = _extract_centroid(activations_per_layer, labels_per_layer,
                               layer_idx, cluster_id)
        if c is not None:
            centroids.append(c)
            idxs.append(layer_idx)
    if not centroids:
        return np.empty((0, 0)), []
    return np.array(centroids), idxs


# ---------------------------------------------------------------------------
# OV matrix loading  (shared by Groups B and C1)
#
# Key format written by save_weight_decomposition in weights.py:
#   shared (ALBERT):   ov_head{h}_shared,    ov_total_shared
#   per-layer (GPT-2): ov_head{h}_layer_{i}, ov_total_layer_{i}
# ---------------------------------------------------------------------------

def _load_ov_head_matrices(data: np.lib.npyio.NpzFile
                            ) -> tuple[list, bool, list]:
    """
    Extract per-head OV matrices from an ov_weights NPZ.

    Returns
    -------
    head_ovs     : shared → list[(d,d)]; per-layer → list of list[(d,d)]
    is_per_layer : bool
    layer_names  : list of str
    """
    keys         = list(data.keys())
    is_per_layer = any(k.startswith("ov_total_layer_") for k in keys)

    if not is_per_layer:
        head_keys = sorted(
            [k for k in keys
             if k.startswith("ov_head") and k.endswith("_shared")],
            key=lambda k: int(k[len("ov_head"):k.index("_shared")]),
        )
        if not head_keys:
            return [], False, []
        return [data[k].astype(np.float64) for k in head_keys], False, ["shared"]

    layer_indices = sorted({
        int(k.split("_layer_")[1])
        for k in keys
        if k.startswith("ov_head") and "_layer_" in k
    })
    layer_names      = [f"layer_{i}" for i in layer_indices]
    per_layer_heads  = []
    for i in layer_indices:
        hkeys = sorted(
            [k for k in keys
             if k.startswith("ov_head") and k.endswith(f"_layer_{i}")],
            key=lambda k: int(k[len("ov_head"):k.index(f"_layer_{i}")]),
        )
        per_layer_heads.append([data[k].astype(np.float64) for k in hkeys])
    return per_layer_heads, True, layer_names


# ---------------------------------------------------------------------------
# _build_v_projectors_from_ov
#
# Phase 2i NPZ files do not exist on disk; all subspace data must be built
# on the fly from Phase 2 artifacts.
#
# Priority:
#   1. ov_decomp_{stem}.npz  — sym_evecs/evals already computed; also has
#      schur_Z/T for schur_block_overlap.  V_sym/V_asym reconstructed from
#      ov_total in the weights NPZ if available alongside.
#   2. ov_weights_{stem}.npz — build V_sym on-the-fly from ov_total, then
#      run build_global_projectors for Schur-based U_pos/U_neg.
# ---------------------------------------------------------------------------

def _build_v_projectors_from_ov(p2_dir: Path, stem: str) -> Optional[dict]:
    """
    Return a bundle containing:
      U_attractive, U_repulsive : (d, k) float32 basis matrices
      V_sym, V_asym             : (d, d) float32 or None
      schur_Z, schur_T          : (d, d) float32 or None
      _source                   : str  — path that succeeded
      _ov_data                  : dict or None  — for C1 reuse

    Returns None if no usable Phase 2 artifact is found under p2_dir.
    """
    stem_h = stem.replace("_", "-")

    def _cands(prefix: str) -> list[Path]:
        return [p2_dir / f"{prefix}_{stem_h}.npz",
                p2_dir / f"{prefix}_{stem}.npz"]

    # ------------------------------------------------------------------
    # Path 1 — ov_decomp NPZ (sym_evecs / sym_evals already computed)
    # ------------------------------------------------------------------
    for decomp_path in _cands("ov_decomp"):
        if not decomp_path.exists():
            continue
        try:
            dec  = np.load(decomp_path)
            keys = list(dec.keys())

            if "sym_evecs_shared" in keys and "sym_evals_shared" in keys:
                sym_evecs = dec["sym_evecs_shared"].astype(np.float64)
                sym_evals = dec["sym_evals_shared"].astype(np.float64)
            else:
                evec_keys = sorted(
                    [k for k in keys if k.startswith("sym_evecs_layer_")],
                    key=lambda k: int(k.split("_layer_")[1]),
                )
                eval_keys = sorted(
                    [k for k in keys if k.startswith("sym_evals_layer_")],
                    key=lambda k: int(k.split("_layer_")[1]),
                )
                if not evec_keys:
                    continue
                sym_evals = np.mean(
                    [dec[k].astype(np.float64) for k in eval_keys], axis=0
                )
                sym_evecs = dec[evec_keys[len(evec_keys) // 2]].astype(np.float64)

            U_pos = sym_evecs[:, sym_evals > 0].astype(np.float32)
            U_neg = sym_evecs[:, sym_evals < 0].astype(np.float32)

            if U_pos.shape[1] == 0:
                warnings.warn(
                    f"[Group B] U_pos empty from {decomp_path.name} — "
                    "V_sym has no positive eigenvalues."
                )

            schur_Z = dec.get("schur_Z_shared")
            schur_T = dec.get("schur_T_shared")

            # Reconstruct V_sym / V_asym from ov_total if weights file is present
            V_sym = V_asym = None
            for wt_path in _cands("ov_weights"):
                if not wt_path.exists():
                    continue
                try:
                    wt = np.load(wt_path)
                    if "ov_total_shared" in wt:
                        V = wt["ov_total_shared"].astype(np.float64)
                    else:
                        tkeys = sorted(
                            [k for k in wt if k.startswith("ov_total_layer_")],
                            key=lambda k: int(k.split("_layer_")[1]),
                        )
                        V = (np.mean([wt[k].astype(np.float64) for k in tkeys],
                                     axis=0)
                             if tkeys else None)
                    if V is not None:
                        V_sym  = ((V + V.T) / 2).astype(np.float32)
                        V_asym = ((V - V.T) / 2).astype(np.float32)
                except Exception:
                    pass
                break

            return {
                "U_attractive": U_pos,
                "U_repulsive":  U_neg,
                "V_sym":   V_sym,
                "V_asym":  V_asym,
                "schur_Z": schur_Z,
                "schur_T": schur_T,
                "_ov_data": None,
                "_source":  str(decomp_path),
            }
        except Exception as e:
            warnings.warn(f"[Group B] Failed to parse {decomp_path.name}: {e}")

    # ------------------------------------------------------------------
    # Path 2 — ov_weights NPZ (build V_sym on-the-fly)
    # ------------------------------------------------------------------
    for wt_path in _cands("ov_weights"):
        if not wt_path.exists():
            continue
        try:
            wt = np.load(wt_path)
            head_ovs, is_per_layer, layer_names = _load_ov_head_matrices(wt)

            if not head_ovs:
                warnings.warn(
                    f"[Group B] No ov_head* keys found in {wt_path.name}."
                )
                continue

            flat_heads = (
                [ov for layer in head_ovs for ov in layer]
                if is_per_layer else head_ovs
            )
            d = flat_heads[0].shape[0]

            if is_per_layer:
                V_eff = np.mean(
                    [np.sum(layer, axis=0) for layer in head_ovs], axis=0
                )
            else:
                V_eff = np.sum(flat_heads, axis=0)

            V_sym  = (V_eff + V_eff.T) / 2
            V_asym = (V_eff - V_eff.T) / 2

            evals, evecs = np.linalg.eigh(V_sym)
            U_pos = evecs[:, evals > 0].astype(np.float32)
            U_neg = evecs[:, evals < 0].astype(np.float32)

            if U_pos.shape[1] == 0:
                warnings.warn(
                    f"[Group B] U_pos empty after eigh on V_sym from "
                    f"{wt_path.name} — check OV sign convention."
                )

            ov_data_struct = {
                "is_per_layer": False,
                "ov_per_head":  flat_heads,
                "n_heads":      len(flat_heads),
                "d_model":      d,
                "layer_names":  ["shared"],
            }
            try:
                projectors  = build_global_projectors(ov_data_struct)
                layer_proj  = projectors["per_layer"][0]
                # Prefer Schur-based bases; fall back to eigh if empty
                if layer_proj["U_pos"].shape[1] > 0:
                    U_pos = layer_proj["U_pos"]
                if layer_proj["U_neg"].shape[1] > 0:
                    U_neg = layer_proj["U_neg"]
            except Exception as e:
                warnings.warn(
                    f"[Group B] build_global_projectors failed ({e}); "
                    "using eigh result."
                )

            return {
                "U_attractive": U_pos,
                "U_repulsive":  U_neg,
                "V_sym":   V_sym.astype(np.float32),
                "V_asym":  V_asym.astype(np.float32),
                "schur_Z": None,
                "schur_T": None,
                "_ov_data": {
                    "is_per_layer": is_per_layer,
                    "ov_per_head":  head_ovs,
                    "n_heads":      (len(head_ovs[0]) if is_per_layer
                                     else len(head_ovs)),
                    "d_model":      d,
                    "layer_names":  layer_names,
                },
                "_source": str(wt_path),
            }
        except Exception as e:
            warnings.warn(
                f"[Group B] Failed to build projectors from {wt_path.name}: {e}"
            )

    warnings.warn(
        f"[Group B] No usable Phase 2 artifact found for stem '{stem}' "
        f"in {p2_dir}. Tried ov_decomp and ov_weights with both stem forms."
    )
    return None


# ---------------------------------------------------------------------------
# Individual group runners
# ---------------------------------------------------------------------------

def _run_group_A(run, primary_raw, sibling_raw, out_dir) -> dict:
    profile = compute_profile(
        run["activations"], run["hdbscan_labels"],
        primary_raw, sibling_raw,
        run["tokens"], run.get("metrics", {"layers": []}),
    )
    save_profile(profile, out_dir, tag="primary")
    return profile


def _run_group_B(
    trajectory:             dict,
    sibling_trajectory:     Optional[dict],
    activations_per_layer:  list,
    labels_per_layer:       list,
    attentions_per_layer:   Optional[list],
    merge_events:           list,
    p2_dir:                 Path,
    p2i_dir:                Path,   # kept for caller compatibility; not used
    stem:                   str,
    out_dir:                Path,
    tag:                    str = "primary",
    ) -> dict:
    """
    Run Group B: paper-theoretical alignment.

    Phase 2i NPZ files do not exist; all projector data is built on-the-fly
    from Phase 2 artifacts via _build_v_projectors_from_ov.
    p2i_dir is accepted but intentionally unused.
    """
    result: dict = {"trajectory_id": int(trajectory["id"])}

    # ------------------------------------------------------------------
    # 1. Build projectors — always on-the-fly from Phase 2 artifacts
    # ------------------------------------------------------------------
    bundle = _build_v_projectors_from_ov(p2_dir, stem)
    if bundle is None:
        result["available"] = False
        result["skipped_reason"] = "no_projectors"
        save_v_alignment(result, out_dir, tag=tag)
        return result

    v_projectors = {
        "U_attractive": bundle["U_attractive"],
        "U_repulsive":  bundle["U_repulsive"],
    }
    V_sym  = bundle.get("V_sym")
    V_asym = bundle.get("V_asym")
    result["projector_source"] = bundle.get("_source", "unknown")

    # ------------------------------------------------------------------
    # 2. Centroid trajectory
    # ------------------------------------------------------------------
    centroid_coords, live_layer_idxs = _centroid_sequence(
        activations_per_layer, labels_per_layer, trajectory
    )
    if centroid_coords.shape[0] < 2:
        result["available"] = False
        result["skipped_reason"] = "insufficient_centroid_layers"
        save_v_alignment(result, out_dir, tag=tag)
        return result

    # ------------------------------------------------------------------
    # 3. Core V-alignment decomposition + summary scalars
    #    report.py reads result["summary"]["mean_centroid_attr_frac"] etc.
    #    compute_v_alignment returns per-step data in "centroid_trajectory";
    #    we compute the means here so the saved JSON has the right shape.
    # ------------------------------------------------------------------
    va = compute_v_alignment(centroid_coords, v_projectors, trajectory)
    result["v_alignment"] = va

    ct = va.get("centroid_trajectory", [])
    result["summary"] = {
        "mean_centroid_attr_frac": (
            float(np.mean([s["attr_frac"] for s in ct])) if ct else None
        ),
        "mean_centroid_rep_frac": (
            float(np.mean([s["rep_frac"] for s in ct])) if ct else None
        ),
        # rotational_neutral_local is filled in after step 6
    }

    # ------------------------------------------------------------------
    # 4. Theorem 6.3 prediction
    # ------------------------------------------------------------------
    n_cluster = int(np.mean([
        (labels_per_layer[li] == trajectory["chain"][i][1]).sum()
        for i, li in enumerate(live_layer_idxs)
        if li < len(labels_per_layer)
    ]))
    result["theorem_6_3"] = theorem_6_3_prediction(
        n_cluster, centroid_coords.shape[1]
    )

    # ------------------------------------------------------------------
    # 5. Effective β estimate
    # ------------------------------------------------------------------
    if attentions_per_layer is not None and live_layer_idxs:
        beta_results = []
        for i, li in enumerate(live_layer_idxs):
            if li >= len(attentions_per_layer):
                continue
            attn = attentions_per_layer[li]
            acts = activations_per_layer[li]
            _, cluster_id = trajectory["chain"][i]
            cluster_indices = np.where(labels_per_layer[li] == cluster_id)[0]
            if cluster_indices.size < 2:
                continue
            br = estimate_effective_beta(attn, acts, cluster_indices)
            br["layer_idx"] = li
            beta_results.append(br)
        result["effective_beta"] = beta_results
    else:
        result["effective_beta"] = {"available": False}

    # ------------------------------------------------------------------
    # 6. Rotational local test — uses V_sym / V_asym from bundle.
    #    Saved as "sa_local_test" to match the key report.py reads.
    #    Also writes the scalar verdict into summary["rotational_neutral_local"].
    # ------------------------------------------------------------------
    if V_sym is not None and V_asym is not None:
        rot = rotational_local_test(centroid_coords, V_sym, V_asym)
    else:
        rot = {"available": False, "reason": "V_sym_unavailable_decomp_path_only"}

    result["sa_local_test"] = rot
    result["summary"]["rotational_neutral_local"] = (
        rot.get("verdict") if rot.get("available") else None
    )

    # ------------------------------------------------------------------
    # 7. Merge-event geometry.
    #    Saved as "merge_geometry" to match the key report.py reads.
    # ------------------------------------------------------------------
    merge_result: dict = {"available": False}
    if sibling_trajectory is not None and merge_events:
        traj_id  = trajectory["id"]
        relevant = [
            ev for ev in merge_events
            if traj_id in ev.get("prev_ids", [])
        ]
        if relevant:
            ev               = relevant[0]
            merge_layer_idx  = ev.get("layer_from")
            if merge_layer_idx is not None:
                pre_idx  = merge_layer_idx - 1
                _, cid_p = (trajectory["chain"][-1]
                            if trajectory["chain"] else (None, None))
                _, cid_s = (sibling_trajectory["chain"][-1]
                            if sibling_trajectory["chain"] else (None, None))
                c_primary = (
                    _extract_centroid(activations_per_layer, labels_per_layer,
                                      pre_idx, cid_p)
                    if cid_p is not None else None
                )
                c_sibling = (
                    _extract_centroid(activations_per_layer, labels_per_layer,
                                      pre_idx, cid_s)
                    if cid_s is not None else None
                )
                c_fused = _extract_centroid(
                    activations_per_layer, labels_per_layer,
                    merge_layer_idx, ev.get("curr_id", -1)
                )
                if all(x is not None for x in [c_primary, c_sibling, c_fused]):
                    merge_result = merge_event_geometry(
                        c_primary, c_sibling, c_fused,
                        v_projectors["U_attractive"],
                        v_projectors["U_repulsive"],
                    )
                    merge_result["available"] = True
                else:
                    merge_result["reason"] = "centroid_extraction_failed"
    result["merge_geometry"] = merge_result

    # ------------------------------------------------------------------
    # 8. Schur block overlap.
    #    Saved as "schur_blocks" (list) to match the key report.py reads.
    # ------------------------------------------------------------------
    schur_Z = bundle.get("schur_Z")
    schur_T = bundle.get("schur_T")
    if schur_Z is not None and schur_T is not None:
        centroid_dir = centroid_coords.mean(axis=0)
        centroid_dir = centroid_dir / (np.linalg.norm(centroid_dir) + 1e-12)
        sob = schur_block_overlap(centroid_dir, schur_Z, schur_T)
        # report.py iterates result["schur_blocks"] as a list of block dicts
        result["schur_blocks"] = sob.get("blocks", []) if isinstance(sob, dict) else []
        result["schur_block_overlap"] = sob
    else:
        result["schur_blocks"] = []
        result["schur_block_overlap"] = {
            "available": False,
            "reason": "schur_Z_T_unavailable",
        }

    save_v_alignment(result, out_dir, tag=tag)
    return result

def _ov_top_eigval_from_composed(OV_h: np.ndarray) -> Optional[float]:
    """
    Extract the top eigenvalue by |eigval| from a pre-composed OV matrix
    (OV = W_V @ W_O).  Uses the symmetric part for stability, matching
    the convention in head_ov_cluster_alignment.
    """
    try:
        OV_sym  = 0.5 * (OV_h + OV_h.T)
        eigvals = np.linalg.eigh(OV_sym)[0]
        idx     = int(np.argmax(np.abs(eigvals)))
        return float(eigvals[idx])
    except Exception:
        return None


def _ov_eigval_from_per_layer(result: dict, h: int) -> Optional[float]:
    """
    Mean top eigval for head h across trajectory layers, read from the
    ov_alignment entries that analyze_heads writes into per_layer when
    W_V / W_O are available as separate matrices.
    Returns None when those entries are absent (composed-only or no-OV run).
    """
    vals = []
    for ld in result.get("per_layer", []):
        for ph in ld.get("per_head", []):
            if ph.get("head") != h:
                continue
            ov = ph.get("ov_alignment")
            if not isinstance(ov, dict):
                continue
            top_eigs = ov.get("top_eigenvectors", [])
            if top_eigs:
                vals.append(top_eigs[0]["eigval"])
    if not vals:
        return None
    return round(float(np.mean(vals)), 4)


def _ov_eigval_from_composed_weights(
    weights:      dict,
    h:            int,
    n_layers_hit: int = None,
    ) -> Optional[float]:
    """
    Mean top eigval for head h computed directly from composed OV matrices
    stored in weights["ov_per_head"].  Used when the NPZ fallback loaded
    composed matrices (W_V @ W_O already multiplied) rather than the
    separate W_V / W_O that analyze_heads expects.

    For per-layer weights, averages over all layers.
    For shared weights, returns the single-layer value.
    """
    ov_per_head = weights.get("ov_per_head")
    if not ov_per_head:
        return None

    vals = []
    if weights.get("is_per_layer"):
        for layer_heads in ov_per_head:
            if h < len(layer_heads):
                ev = _ov_top_eigval_from_composed(layer_heads[h])
                if ev is not None:
                    vals.append(ev)
    else:
        if h < len(ov_per_head):
            ev = _ov_top_eigval_from_composed(ov_per_head[h])
            if ev is not None:
                vals.append(ev)

    if not vals:
        return None
    return round(float(np.mean(vals)), 4)

def _ov_profile_from_composed_weights(weights: dict, h: int) -> dict:
    """
    Compute a mean OV spectral profile for head h from pre-composed matrices
    stored in weights["ov_per_head"].

    Used when the NPZ fallback loaded OV = W_V @ W_O (already composed),
    which analyze_heads cannot use for cohesion but which we can still
    eigendecompose directly.

    centroid is unavailable in this path, so cluster_overlap_att/rep are
    omitted (ov_spectral_profile returns None for those fields).
    """
    ov_per_head = weights.get("ov_per_head")
    if not ov_per_head:
        return {}

    profiles = []
    if weights.get("is_per_layer"):
        for layer_heads in ov_per_head:
            if h < len(layer_heads):
                try:
                    profiles.append(ov_spectral_profile(layer_heads[h]))
                except Exception:
                    pass
    else:
        if h < len(ov_per_head):
            try:
                profiles.append(ov_spectral_profile(ov_per_head[h]))
            except Exception:
                pass

    if not profiles:
        return {}

    def _mean(key):
        vals = [p[key] for p in profiles if p.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    top_k_max = max(len(p["top_eigvals"]) for p in profiles)
    top_evs   = [
        round(float(np.mean([p["top_eigvals"][i]
                              for p in profiles if i < len(p["top_eigvals"])])), 4)
        for i in range(top_k_max)
    ]

    return {
        "ov_frac_attractive":     _mean("frac_attractive"),
        "ov_participation_ratio": round(float(np.mean(
                                      [p["participation_ratio"] for p in profiles])), 3),
        "ov_cluster_overlap_att": None,   # centroid not available in composed path
        "ov_cluster_overlap_rep": None,
        "ov_top_eigvals":         top_evs,
    }


def _run_group_C1(
    run:         dict,
    primary_raw: dict,
    weights:     dict,
    out_dir:     Path,
    p2_dir:      Optional[Path] = None,
    stem:        Optional[str]  = None,
    ) -> dict:
    """
    Run Group C.1: per-head attention contributions.

    Weight-loading priority
    -----------------------
    1. weights arg actually contains factored W_V / W_O — analyze_heads
       computes the cohesion scalar and full OV spectral profile (including
       cluster_overlap_att/rep) per head per layer directly from them.
    2. ov_weights_{stem}.npz (composed OV = W_V @ W_O, the schema the Phase 2
       NPZ actually persists) — fed into analyze_heads as `composed_ov`. Since
       OV = W_V @ W_O, head_cohesion_scalar_composed gives the same cohesion
       scalar as path 1 (matrix-mult associativity), and ov_spectral_profile
       is computed with the real per-layer centroid, so cluster_overlap_att/
       rep are populated too — not just frac_attractive/participation_ratio.
    3. Neither available — inward_mass cohesion only; all OV fields None.

    `weights` being a non-empty dict does NOT imply path 1 is usable: when
    `load_phase2_weights` resolves to the same ov_weights_*.npz used for
    path 2, the dict is non-empty but holds ov_head{h}_shared / _layer_{i}
    keys, not W_V / W_O. So the guard below checks for the factored keys
    specifically, not dict truthiness.
    """
    if run.get("attentions") is None:
        print("  [C1] skipped: attentions.npz missing")
        return {"_error": "attentions unavailable"}

    has_factored = bool(weights) and weights.get("W_V") is not None \
        and weights.get("W_O") is not None

    # ------------------------------------------------------------------
    # Attempt to load composed OV matrices from Phase 2 NPZ whenever the
    # factored path isn't usable — independent of whether `weights` is an
    # empty dict or a non-empty dict of differently-keyed (composed) arrays.
    # Kept separate from `weights` so analyze_heads always receives a clean
    # schema for each path.
    # ------------------------------------------------------------------
    composed_weights = {}

    if not has_factored and p2_dir is not None and stem is not None:
        stem_h = stem.replace("_", "-")
        for cand in (p2_dir / f"ov_weights_{stem_h}.npz",
                     p2_dir / f"ov_weights_{stem}.npz"):
            if not cand.exists():
                continue
            try:
                wt = np.load(cand)
                head_ovs, is_per_layer, layer_names = _load_ov_head_matrices(wt)
                if head_ovs:
                    n_h = len(head_ovs[0]) if is_per_layer else len(head_ovs)
                    d   = head_ovs[0][0].shape[0] if is_per_layer else head_ovs[0].shape[0]
                    layer_index_map = None
                    if is_per_layer:
                        # layer_names are "layer_{i}" with i the absolute layer
                        # index; map it to the position in head_ovs so analyze_heads
                        # can look up the right per-head OV list by chain layer.
                        layer_index_map = {
                            int(name.split("_")[1]): idx
                            for idx, name in enumerate(layer_names)
                        }
                    composed_weights = {
                        "is_per_layer":    is_per_layer,
                        "ov_per_head":     head_ovs,
                        "n_heads":         n_h,
                        "d_model":         d,
                        "layer_names":     layer_names,
                        "layer_index_map": layer_index_map,
                    }
                    print(f"  [C1] loaded composed OV matrices from {cand.name} "
                          f"({'per-layer' if is_per_layer else 'shared'}, "
                          f"{n_h} heads, cohesion via composed OV)")
                    break
            except Exception as e:
                print(f"  [C1] OV fallback load failed ({cand.name}): {e}")

        if not composed_weights:
            print("  [C1] no OV weights available — attention-pattern cohesion only")

    result = analyze_heads(
        run["activations"], run["attentions"], run["hdbscan_labels"],
        primary_raw, run["tokens"], weights=weights,
        composed_ov=composed_weights or None,
    )

    # ------------------------------------------------------------------
    # inward_mass fallback: fires only when neither the factored nor the
    # composed OV path produced a signal (analyze_heads left
    # cohesion_source None and cumulative_cohesion all-zero).
    # ------------------------------------------------------------------
    cc = result.get("cumulative_cohesion", [])
    if cc and not any(x != 0.0 for x in cc) and result.get("cohesion_source") is None:
        per_layer = result.get("per_layer", [])
        n_heads   = len(cc)
        fallback  = np.zeros(n_heads)
        for ld in per_layer:
            for ph in ld.get("per_head", []):
                h = ph.get("head", 0)
                if h < n_heads:
                    fallback[h] += ph.get("inward_mass", 0.0)

        result["cumulative_cohesion"] = [round(float(x), 4) for x in fallback]
        result["cohesion_source"]     = "inward_mass_fallback"

        if fallback.any():
            order = np.argsort(-fallback)
            result["top_attractor_heads"] = [
                {
                    "head":     int(h),
                    "cohesion": round(float(fallback[h]), 4),
                }
                for h in order[:8]
                if fallback[h] > 0.0
            ]

    save_head_contributions(result, out_dir, tag="primary")
    return result

def _run_group_C2(run, primary_raw, sibling_raw, phase2_dir, out_dir,
                   model_stem=None) -> dict:
    deltas      = load_ffn_deltas(Path(phase2_dir), run["prompt_key"],
                                   model_stem=model_stem)
    ffn_deltas  = deltas["ffn"]  if deltas else None
    attn_deltas = deltas["attn"] if deltas else None
    if ffn_deltas is None:
        print(f"  [C2] no phase2 ffn deltas for prompt {run['prompt_key']} "
              "— LDA + centroid directions only, no projection metrics")
    result = analyze_ffn(
        run["activations"], run["hdbscan_labels"],
        primary_raw, sibling_raw,
        ffn_deltas=ffn_deltas, attn_deltas=attn_deltas,
    )
    save_ffn_contributions(result, out_dir, tag="primary")
    return result


def _run_group_D(run, primary_raw, sibling_raw, phase3, v_proj, phase4,
                  centroid_coords, out_dir) -> dict:
    layers_needed = sorted({int(l) for l, _ in primary_raw["chain"]})
    feature_acts  = _compute_feature_activations(
        phase3, run["prompt_key"], layers_needed,
    )
    if not feature_acts:
        print("  [D] skipped: feature activations unavailable "
              "(phase3 crosscoder or prompt store missing)")
        result = {"_error": "feature activations unavailable"}
        save_feature_signature(result, out_dir, tag="primary")
        return result

    decoder_dirs = None
    cc = phase3.get("crosscoder")
    if cc is not None:
        for attr in ("decoder_weight", "W_dec", "decoder"):
            if hasattr(cc, attr):
                d = getattr(cc, attr)
                if hasattr(d, "weight"):
                    d = d.weight
                decoder_dirs = (d.detach().cpu().numpy()
                                if hasattr(d, "detach")
                                else np.asarray(d))
                break

    lda_npz_path = out_dir / "group_C2_lda_directions_primary.npz"
    lda_dirs = ({k: v for k, v in np.load(lda_npz_path).items()}
                if lda_npz_path.exists() else None)

    bn = None
    bn_blob = phase4.get("t3_bottleneck_directions")
    if bn_blob:
        for k, v in bn_blob.items():
            bn = v
            break

    result = analyze_features(
        feature_acts, decoder_dirs,
        run["hdbscan_labels"], primary_raw, sibling_raw,
        lda_directions_per_layer=lda_dirs,
        v_projectors=v_proj,
        bottleneck_directions=bn,
        cluster_centroid=(
            centroid_coords.mean(axis=0)
            if centroid_coords is not None
               and centroid_coords.ndim == 2
               and centroid_coords.shape[0] > 0
            else None
        ),
    )
    save_feature_signature(result, out_dir, tag="primary")
    return result


def _run_group_E(run, primary_raw, sibling_raw, model, tokenizer, out_dir,
                  tuned_lens_path) -> dict:
    from .tuned_lens_cluster import (
        decode_cluster_trajectory, save_tuned_lens_result,
        kl_sibling_contrast, load_tuned_lens,
    )
    tuned_lens     = (load_tuned_lens(Path(tuned_lens_path))
                      if tuned_lens_path else None)
    primary_result = decode_cluster_trajectory(
        run["activations"], run["hdbscan_labels"],
        primary_raw, run["tokens"],
        model, tokenizer, tuned_lens=tuned_lens,
    )
    if sibling_raw is not None and sibling_raw.get("chain"):
        sibling_result = decode_cluster_trajectory(
            run["activations"], run["hdbscan_labels"],
            sibling_raw, run["tokens"],
            model, tokenizer, tuned_lens=tuned_lens,
            decode_members=False,
        )
        merge       = primary_raw.get("merge_event")
        merge_layer = merge["layer_from"] if merge else None
        kl = kl_sibling_contrast(
            primary_result, sibling_result,
            primary_result.get("_distributions", {}),
            sibling_result.get("_distributions", {}),
            primary_raw["chain"], sibling_raw["chain"],
            merge_layer=merge_layer,
        )
        primary_result["kl_contrast"] = kl
        save_tuned_lens_result(sibling_result, out_dir, tag="sibling")
    save_tuned_lens_result(primary_result, out_dir, tag="primary")
    return primary_result


def _run_group_F(run, primary_raw, sibling_raw, model, tokenizer,
                  c1_result, c2_result, max_iterations, steering_alpha,
                  out_dir) -> dict:
    from .causal_tests import run_causal_tests, save_causal

    top_heads = (c1_result.get("top_attractor_heads", [])
                 if c1_result else [])

    lda_dir   = None
    c2_npz    = out_dir / "group_C2_lda_directions_primary.npz"
    if c2_npz.exists():
        data     = dict(np.load(c2_npz))
        merge    = primary_raw.get("merge_event")
        target_L = (merge["layer_from"] if merge else
                    primary_raw["chain"][len(primary_raw["chain"]) // 2][0])
        for offset in (0, -1, -2, 1):
            key = f"lda_L{target_L + offset}"
            if key in data:
                lda_dir = data[key]
                break

    chain           = primary_raw["chain"]
    mid_layer, mid_cid = chain[len(chain) // 2]
    mask            = run["hdbscan_labels"][mid_layer] == mid_cid
    centroid        = run["activations"][mid_layer][mask].mean(axis=0)
    centroid        = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

    prompt_text = tokenizer.convert_tokens_to_string(run["tokens"])

    result = run_causal_tests(
        model, tokenizer, prompt_text, max_iterations,
        run["activations"], run["hdbscan_labels"],
        primary_raw, sibling_raw,
        top_heads=top_heads,
        lda_direction=lda_dir,
        centroid_direction=centroid,
        steering_alpha=steering_alpha,
    )
    save_causal(result, out_dir)
    return result


def _run_group_G(run, primary_raw, sibling_raw, weights, out_dir) -> dict:
    result = run_sibling_contrast(
        run["activations"], run.get("attentions"),
        run["hdbscan_labels"],
        primary_raw, sibling_raw,
        run["tokens"], run.get("metrics", {"layers": []}),
        weights=weights,
    )
    save_sibling_contrast(result, out_dir)
    return result


# ---------------------------------------------------------------------------
# Model loading for Groups E and F
# ---------------------------------------------------------------------------

def _load_model(model_name: str, device: str = "cpu"):
    from transformers import (
        AutoModel, AutoTokenizer,
        AutoModelForMaskedLM, AutoModelForCausalLM,
    )
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = None
    for loader in (AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel):
        try:
            model = loader.from_pretrained(model_name).to(device)
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(
            f"Could not load model '{model_name}' with any AutoModel class"
        )
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Default model list
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "albert-base-v2",
    "albert-xlarge-v2",
    "bert-base-uncased",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 5 case study orchestrator (multi-model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, metavar="MODEL")
    p.add_argument("--model", default=None,
                   help="Single HF model id (legacy; use --models instead).")
    p.add_argument("--model-stem", default=None)

    p.add_argument("--phase1-dir",   default="results/phase1")
    p.add_argument("--phase2-dir",   default="results/phase2")
    p.add_argument("--phase2i-dir",  default="results/phase2i")
    p.add_argument("--phase3-ckpt",  default=None)
    p.add_argument("--phase3-cache", default=None)
    p.add_argument("--phase4-dir",   default="results/phase4")
    p.add_argument("--tuned-lens",   default=None)

    p.add_argument("--out", default=None)
    p.add_argument("--groups", nargs="+", default=ALL_GROUPS,
                   choices=ALL_GROUPS)

    p.add_argument("--force-prompt",        default=None)
    p.add_argument("--force-trajectory-id", type=int, default=None)
    p.add_argument("--runner-up-rank",      type=int, default=1)
    p.add_argument("--max-iterations",      type=int, default=64)
    p.add_argument("--steering-alpha",      type=float, default=2.0)
    p.add_argument("--device",              default="cpu")
    p.add_argument("--dry-run",             action="store_true")
    return p


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------

def _run_one_model(args, model_name: str, model_stem: str,
                   out_dir: Path) -> int:
    phase3_ckpt  = Path(args.phase3_ckpt  or f"checkpoints/{model_name}/final")
    phase3_cache = Path(args.phase3_cache or f"activation_cache/{model_name}")

    with open(out_dir / "run_config.json", "w") as f:
        cfg = vars(args).copy()
        cfg.update({"_model": model_name, "_model_stem": model_stem})
        json.dump(cfg, f, indent=2, default=str)

    # --- Phase 1: discover + load ---
    print(f"[phase5] discovering phase1 runs in {args.phase1_dir} "
          f"matching stem '{model_stem}'")
    run_paths = p5io.find_phase1_runs(Path(args.phase1_dir), model_stem)
    if not run_paths:
        print("  no phase1 runs found — skipping")
        return 1
    print(f"  found: {list(run_paths.keys())}")

    phase1_runs = {}
    for prompt_key, run_path in run_paths.items():
        try:
            phase1_runs[prompt_key] = p5io.load_phase1_run(run_path)
        except Exception as e:
            print(f"  [skip] {prompt_key}: {e}")

    # --- Trajectory selection ---
    print("[phase5] ranking trajectories")
    selection = select_primary_and_sibling(
        phase1_runs,
        force_prompt=args.force_prompt,
        force_trajectory_id=args.force_trajectory_id,
        runner_up_rank=args.runner_up_rank,
    )
    save_selection(selection, out_dir / "cluster_metadata.json")

    primary = selection["primary"]
    sibling = selection["sibling"]
    print(f"  primary: prompt={primary['prompt_key']} id={primary['id']} "
          f"score={primary['total_score']:.3f}")
    if sibling:
        print(f"  sibling: id={sibling['id']} "
              f"{'(below gates)' if 'note' in sibling else ''}")
    if selection.get("runner_up"):
        ru = selection["runner_up"]
        print(f"  runner_up: prompt={ru['prompt_key']} id={ru['id']} "
              f"score={ru['total_score']:.3f}")

    if args.dry_run:
        print("[phase5] dry run complete")
        return 0

    # --- Resolve run + trajectory objects ---
    run         = phase1_runs[primary["prompt_key"]]
    primary_raw = _traj_by_id(run["trajectories"], primary["id"])
    primary_raw["merge_event"] = primary.get("merge_event")

    sibling_raw = None
    if sibling and sibling.get("id") is not None:
        try:
            sibling_raw = _traj_by_id(run["trajectories"], sibling["id"])
        except Exception as e:
            print(f"  [warn] sibling raw lookup failed: {e}")

    centroid_coords = _centroid_coords(
        primary_raw, run.get("centroid_trajs", {}),
        run["activations"], run["hdbscan_labels"],
    )
    sibling_centroid_coords = (
        _centroid_coords(
            sibling_raw, run.get("centroid_trajs", {}),
            run["activations"], run["hdbscan_labels"],
        ) if sibling_raw is not None else None
    )

    # --- Load cross-phase artifacts ---
    v_proj  = {}
    weights = {}
    phase3  = {}
    phase4  = {}

    if any(g in args.groups for g in ("B", "D")):
        print("[phase5] loading phase2 projectors")
        v_proj = p5io.load_phase2_projectors(
            Path(args.phase2_dir), model_stem, k_top=C.V_PROJECTOR_K_TOP,
        )
        if v_proj.get("path") is None:
            print(f"  [warn] no phase2 projectors found for stem {model_stem}")

    if any(g in args.groups for g in ("C1", "G")):
        print("[phase5] loading phase2 weights")
        weights = p5io.load_phase2_weights(Path(args.phase2_dir), model_stem)
        if not (weights.get("W_V") is not None and weights.get("W_O") is not None):
            print(f"  [note] no factored W_V/W_O in phase2 weights — "
                  "C1 will attempt the composed-OV load from ov_weights NPZ")

    if "D" in args.groups:
        print("[phase5] loading phase3 crosscoder + prompt store")
        phase3 = p5io.load_phase3(phase3_ckpt, phase3_cache, device=args.device)
        print("[phase5] loading phase4 artifacts")
        phase4 = p5io.load_phase4(Path(args.phase4_dir), model_stem)

    # --- Run groups ---
    c1_result = None
    c2_result = None

    if "A" in args.groups:
        print("[phase5] GROUP A: structural profile")
        _run_group_A(run, primary_raw, sibling_raw, out_dir)

    if "B" in args.groups:
        print("[phase5] GROUP B: v-alignment")
        _run_group_B(
            trajectory            = primary_raw,
            sibling_trajectory    = sibling_raw,
            activations_per_layer = run["activations"],
            labels_per_layer      = run["hdbscan_labels"],
            attentions_per_layer  = run.get("attentions"),
            merge_events          = run.get("events", []),
            p2_dir                = Path(args.phase2_dir),
            p2i_dir               = Path(args.phase2i_dir),
            stem                  = model_stem,
            out_dir               = out_dir,
            tag                   = "primary",
        )

    if "C1" in args.groups:
        print("[phase5] GROUP C.1: per-head attention")
        c1_result = _run_group_C1(
            run, primary_raw, weights, out_dir,
            p2_dir=Path(args.phase2_dir),
            stem=model_stem,
        )

    if "C2" in args.groups:
        print("[phase5] GROUP C.2: ffn contributions")
        c2_result = _run_group_C2(
            run, primary_raw, sibling_raw,
            args.phase2_dir, out_dir, model_stem=model_stem,
        )

    if "D" in args.groups:
        print("[phase5] GROUP D: feature signatures")
        _run_group_D(
            run, primary_raw, sibling_raw, phase3, v_proj, phase4,
            centroid_coords, out_dir,
        )

    if "E" in args.groups:
        print(f"[phase5] GROUP E: tuned-lens (loading {model_name})")
        try:
            model_obj, tokenizer = _load_model(model_name, device=args.device)
            _run_group_E(
                run, primary_raw, sibling_raw,
                model_obj, tokenizer, out_dir, args.tuned_lens,
            )
        except Exception as e:
            print(f"  [E] failed: {e}")

    if "F" in args.groups:
        print(f"[phase5] GROUP F: causal tests (loading {model_name})")
        if c1_result is None:
            print("  [warn] C1 not run — ablation targets unknown")
        try:
            model_obj, tokenizer = _load_model(model_name, device=args.device)
            _run_group_F(
                run, primary_raw, sibling_raw, model_obj, tokenizer,
                c1_result, c2_result, args.max_iterations,
                args.steering_alpha, out_dir,
            )
        except Exception as e:
            print(f"  [F] failed: {e}")

    if "G" in args.groups:
        print("[phase5] GROUP G: sibling + random control")
        _run_group_G(run, primary_raw, sibling_raw, weights, out_dir)

    # --- Persist shared per-layer arrays ---
    shared = {}
    for i, arr in enumerate(run["hdbscan_labels"]):
        shared[f"hdb_L{i}"] = np.asarray(arr, dtype=np.int32)
    shared["primary_centroids"] = centroid_coords
    if sibling_centroid_coords is not None:
        shared["sibling_centroids"] = sibling_centroid_coords
    np.savez_compressed(out_dir / "per_layer_arrays.npz", **shared)

    # --- Report ---
    print("[phase5] writing report")
    report_path = write_report(out_dir, model=model_name, tag="primary")
    print(f"  wrote {report_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import traceback

    args   = build_argparser().parse_args(argv)
    models = [args.model] if args.model else args.models

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.out) if args.out else Path("results/phase5")
    results: dict[str, str] = {}

    for model_name in models:
        model_stem = (
            args.model_stem
            if args.model_stem and len(models) == 1
            else model_name.replace("-", "_").replace("/", "_")
        )
        out_dir = base_out / f"{model_stem}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*64}")
        print(f"[phase5] model     : {model_name}")
        print(f"[phase5] stem      : {model_stem}")
        print(f"[phase5] output dir: {out_dir}")
        print(f"{'='*64}")

        try:
            rc = _run_one_model(args, model_name, model_stem, out_dir)
            results[model_name] = ("skipped (no phase1 runs)" if rc == 1
                                   else "ok")
        except Exception as exc:
            print(f"  [ERROR] {model_name} failed: {exc}")
            traceback.print_exc()
            results[model_name] = f"FAILED: {exc}"

    print(f"\n{'='*64}")
    print("[phase5] run summary")
    for m, status in results.items():
        icon = "✓" if status == "ok" else ("~" if "skipped" in status else "✗")
        print(f"  {icon}  {m:30s}  {status}")
    print(f"{'='*64}")

    n_failed = sum(1 for s in results.values() if s.startswith("FAILED"))
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())