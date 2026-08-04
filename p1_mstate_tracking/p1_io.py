"""
p1_mstate_tracking/p1_io.py — Save and load Phase 1 run artifacts.

THE canonical Phase 1 reader and writer. Every other phase reads Phase 1
output through load_phase1_run / find_phase1_run_dir in this module —
p5_single_mstate_analysis/p5_io.py and p5b_manifold_steering/p5b_io.py both
delegate here and add only their own phase-specific derived values on top.
Do not add another load_phase1_run somewhere else; extend this one.

This file was io_utils.py. Renamed to p1_io.py for the project-wide
one-io-module-per-phase convention (p1_io.py, p5_io.py, p5b_io.py,
p6_io.py) — the bare basename io.py is reserved for core/io.py.

load_phase1_run and find_phase1_run_dir were absorbed from core/io.py in
the same pass: there were three independent Phase 1 readers (this file's
own load_run/load_activations/etc., core/io.py's load_phase1_run, and
p5-local variants) that silently disagreed on key names and source files.
core/io.py's version also had a real bug — it read `trajectories` from the
top level of trajectory.json, but _save_trajectory below writes them
nested under cluster_tracking, so it had been returning [] on every real
run. Fixed as part of the merge; see load_phase1_run's docstring.

Layout (v2): one JSON per experiment type + large arrays to .npz.
Layout (v1): single metrics.json — detected automatically by load_run.

Per-prompt directory (v2)
--------------------------
JSON (small, LLM + downstream code accessible):
  geometry.json         ip stats, CKA, effective rank, NN stability per layer
  energies.json         interaction energies and energy-drop events per layer
  clustering.json       kmeans / HDBSCAN / nesting / pair-agreement summaries
  spectral.json         spectral eigengap k, eigenvalues, Fiedler partition
  sinkhorn.json         attention entropy and Sinkhorn statistics per layer
  trajectory.json       cluster-tracking events/summary, plateau layers
  hdbscan_labels.json   {layer_idx: [labels]}            — Phase 3 bridge
  events.json           merge_layers, energy_violations  — Phase 3 bridge
  layer_metrics.json    flat per-layer scalars            — Phase 3 plateau detection

NPZ (large arrays):
  activations.npz           L2-normed hidden states (n_layers, n_tokens, d)
  attentions.npz            attention weights (n_layers, n_heads, n, n)
  clusters.npz              kmeans / HDBSCAN labels + kmeans centroids + agglom mid-labels
  centroid_trajectories.npz HDBSCAN centroid paths across layers
  plateau_attentions.npz    attention at plateau layers
  pca_trajectories.npz      PCA projections, layer_{i} -> (n_tokens, 3)
  fiedler_vecs.npz          Fiedler eigenvectors, fiedler_L{i} -> (n_tokens,)

Text:
  tokens.txt        token list with indices
  layer_metrics.csv flat CSV of key per-layer scalars

Global at phase1_dir root (written by aggregate_global_artifacts):
  pair_agreement.json  aggregated pair-agreement summary across all prompt runs
"""

from __future__ import annotations

import re
import csv
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """Handles numpy scalars, arrays, and numpy-typed dict keys."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def iterencode(self, obj, _one_shot=False):
        obj = self._coerce_keys(obj)
        return super().iterencode(obj, _one_shot=_one_shot)

    @staticmethod
    def _coerce_keys(obj):
        if isinstance(obj, dict):
            return {
                (int(k) if isinstance(k, np.integer) else
                 float(k) if isinstance(k, np.floating) else k):
                NumpyEncoder._coerce_keys(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [NumpyEncoder._coerce_keys(v) for v in obj]
        return obj


def _jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# Saving -- per-experiment helpers
# ---------------------------------------------------------------------------

# Provenance fields recorded by analyze_trajectory from
# core.models.describe_extraction. Written verbatim into geometry.json.
#
# These were computed and carried in the in-memory results dict but never
# reached disk, which made every saved run indistinguishable from every
# other except by its directory name — the exact failure describe_extraction
# was written to prevent (checkpoints.py otherwise recovers the training
# step by parsing the directory name back out, so a rename silently
# re-labels the x-axis of every developmental plot).
#
# `sublayer_semantics` in particular is the documented discriminator between
# a pre-fix GPT-2 @attn/@ffn run (which stored sublayer *deltas* under a
# stream label) and a post-fix one (true residual streams). Runs predating
# this change lack the field entirely, and that absence is itself the signal.
_PROVENANCE_FIELDS = (
    "n_layers_total",
    "n_layers_analyzed",
    "lm_head_excluded",
    "hidden_state_0_is_embedding",
    "final_hidden_state_is_post_ln",
    "model_family",
    "weight_dtype",
    "autocast",
    "hf_repo",
    "revision",
    "checkpoint_step",
    "random_init",
    "sublayer_semantics",
    "parallel_residual",
)


def _save_geometry(results, run_dir):
    """ip stats, CKA, effective rank, NN stability per layer, plus the
    extraction provenance that makes the run self-describing."""
    layers_out = []
    for lr in results["layers"]:
        layers_out.append({
            "layer":                    lr["layer"],
            "ip_mean":                  lr["ip_mean"],
            "ip_std":                   lr["ip_std"],
            "ip_mass_near_1":           lr["ip_mass_near_1"],
            "ip_histogram":             lr.get("ip_histogram", []),
            "effective_rank":           lr["effective_rank"],
            # The frame-correct companion to effective_rank. Kept as a
            # separate key rather than replacing it: the two are not
            # interchangeable (see core/metrics.py effective_rank modes).
            "effective_rank_normed":    lr.get("effective_rank_normed"),
            "cka_prev":                 lr.get("cka_prev"),
            "nn_stability":             lr.get("nn_stability"),
            "nn_indices":               lr.get("nn_indices", []),
            "pca_explained_variance":   lr.get("pca_explained_variance", []),
        })

    payload = {
        "model":    results["model"],
        "prompt":   results["prompt"],
        "n_layers": results["n_layers"],
        "n_tokens": results["n_tokens"],
        "d_model":  results["d_model"],
        "tokens":   results["tokens"],
        "layers":   layers_out,
    }
    # .get() rather than [] so a caller that built `results` without
    # extraction_meta (legacy, or a test fixture) still saves rather than
    # raising — the field lands as null, which reads as "not recorded".
    for key in _PROVENANCE_FIELDS:
        payload[key] = results.get(key)

    _jdump(payload, run_dir / "geometry.json")


def _save_energies(results, run_dir):
    """Interaction energies and energy-drop events per layer."""
    layers_out = []
    for lr in results["layers"]:
        edp = {str(b): v for b, v in lr.get("energy_drop_pairs", {}).items()}
        layers_out.append({
            "layer":             lr["layer"],
            "energies":          {str(b): v for b, v in lr.get("energies", {}).items()},
            "energy_drop_pairs": edp,
        })
    _jdump({"model": results["model"], "prompt": results["prompt"],
            "n_layers": results["n_layers"], "layers": layers_out},
           run_dir / "energies.json")


def _save_clustering(results, run_dir):
    """
    Clustering summaries without label arrays (those go to clusters.npz).
    Covers kmeans best-k/silhouette, HDBSCAN count, nesting, pair_agreement.
    """
    layers_out = []
    for lr in results["layers"]:
        cl  = lr.get("clustering", {})
        km  = cl.get("kmeans", {})
        hdb = cl.get("hdbscan", {})
        agg = cl.get("agglomerative", {})

        agg_counts = {k: v for k, v in agg.items() if k != "mid_labels"}

        hdb_labels = hdb.get("labels", [])
        n_noise    = sum(1 for x in hdb_labels if x == -1)
        n_tok      = len(hdb_labels) if hdb_labels else results["n_tokens"]

        layers_out.append({
            "layer": lr["layer"],
            "clustering": {
                "agglomerative": agg_counts,
                "kmeans": {
                    "best_k":          km.get("best_k"),
                    "best_silhouette": km.get("best_silhouette"),
                },
                "hdbscan": {
                    "n_clusters":     hdb.get("n_clusters"),
                    "noise_count":    n_noise,
                    "noise_fraction": round(n_noise / n_tok, 4) if n_tok else None,
                },
            },
            "nesting":        lr.get("nesting", {}),
            "pair_agreement": lr.get("pair_agreement", {}),
        })
    _jdump({"model": results["model"], "prompt": results["prompt"],
            "n_layers": results["n_layers"], "layers": layers_out},
           run_dir / "clustering.json")


def _save_spectral(results, run_dir):
    """
    Spectral eigengap k, eigenvalues, Fiedler bipartition per layer.
    Fiedler eigenvectors go to fiedler_vecs.npz.
    """
    layers_out = []
    for lr in results["layers"]:
        sp = lr.get("spectral", {})
        layers_out.append({
            "layer":               lr["layer"],
            "k_eigengap":          sp.get("k_eigengap"),
            "eigenvalues":         sp.get("eigenvalues", []),
            "eigengaps":           sp.get("eigengaps", []),
            "fiedler_bipartition": lr.get("fiedler_bipartition"),
        })
    _jdump({"model": results["model"], "prompt": results["prompt"],
            "n_layers": results["n_layers"], "layers": layers_out},
           run_dir / "spectral.json")


def _save_sinkhorn(results, run_dir):
    """Attention entropy and Sinkhorn statistics per layer."""
    layers_out = []
    for lr in results["layers"]:
        sk = lr.get("sinkhorn", {})
        layers_out.append({
            "layer":                       lr["layer"],
            "fiedler_mean":                sk.get("fiedler_mean"),
            "sinkhorn_cluster_count_mean": sk.get("sinkhorn_cluster_count_mean"),
            "row_col_balance_mean":        sk.get("row_col_balance_mean"),
            "attention_entropy_mean":      lr.get("attention_entropy_mean"),
            "attention_entropy_per_head":  lr.get("attention_entropy_per_head", []),
        })
    _jdump({"model": results["model"], "prompt": results["prompt"],
            "n_layers": results["n_layers"], "layers": layers_out},
           run_dir / "sinkhorn.json")


def _save_trajectory(results, run_dir):
    """Cluster tracking events/summary and plateau layers."""
    _jdump({
        "model":            results["model"],
        "prompt":           results["prompt"],
        "plateau_layers":   results.get("plateau_layers", []),
        "cluster_tracking": results.get("cluster_tracking", {}),
    }, run_dir / "trajectory.json")


def _save_bridge_files(results, run_dir):
    """
    Write the three JSON files consumed by Phase 3 _load_artifacts:
      hdbscan_labels.json  {str(layer_idx): [int, ...]}
      events.json          merge_layers, energy_violations, energy_drop_pairs
      layer_metrics.json   flat list for _detect_plateau_windows
    """
    from core.config import BETA_VALUES

    # hdbscan_labels.json
    hdb_labels = {}
    for lr in results["layers"]:
        hdb = lr.get("clustering", {}).get("hdbscan", {})
        if "labels" in hdb:
            hdb_labels[str(lr["layer"])] = hdb["labels"]
    _jdump(hdb_labels, run_dir / "hdbscan_labels.json")

    # events.json
    tracking = results.get("cluster_tracking", {})
    events   = tracking.get("events", [])

    # Events from track_clusters have no "type"/"layer" fields; the correct
    # fields are "layer_from" and "n_merges".  The old predicate was always
    # False, so merge_layers was always [].
    merge_layers = sorted({
        e["layer_from"] for e in events if e.get("n_merges", 0) > 0
    })

    energy_violations = {}
    for beta in BETA_VALUES:
        beta_str = str(beta)
        viol = []
        for lr in results["layers"]:
            edp = lr.get("energy_drop_pairs", {})
            pairs = edp.get(beta, edp.get(beta_str, []))
            if pairs:
                viol.append(lr["layer"])
        energy_violations[beta_str] = viol

    edp_by_layer = {}
    for lr in results["layers"]:
        edp = lr.get("energy_drop_pairs", {})
        if any(len(v) > 0 for v in edp.values()):
            edp_by_layer[str(lr["layer"])] = {str(b): v for b, v in edp.items()}

    _jdump({
        "merge_layers":      merge_layers,
        "energy_violations": energy_violations,
        "energy_drop_pairs": edp_by_layer,
    }, run_dir / "events.json")

    # layer_metrics.json — flat list for _detect_plateau_windows
    lm_rows = []
    for lr in results["layers"]:
        hdb = lr.get("clustering", {}).get("hdbscan", {})
        sp  = lr.get("spectral", {})
        lm_rows.append({
            "layer":        lr["layer"],
            "cka":          lr.get("cka_prev"),
            "nn_stability": lr.get("nn_stability"),
            "hdbscan_k":    hdb.get("n_clusters"),
            "spectral_k":   sp.get("k_eigengap"),
        })
    _jdump(lm_rows, run_dir / "layer_metrics.json")


def _save_layer_metrics_csv(results, run_dir):
    """Flat CSV of key per-layer scalars for human inspection."""
    from core.config import BETA_VALUES

    rows = []
    for lr in results["layers"]:
        row = {
            "layer":             lr["layer"],
            "ip_mean":           lr["ip_mean"],
            "ip_std":            lr["ip_std"],
            "ip_mass_near_1":    lr["ip_mass_near_1"],
            "effective_rank":    lr["effective_rank"],
            "spectral_k":        lr.get("spectral", {}).get("k_eigengap", ""),
            "hdbscan_k":         lr.get("clustering", {}).get("hdbscan", {}).get("n_clusters", ""),
            "kmeans_k":          lr.get("clustering", {}).get("kmeans", {}).get("best_k", ""),
            "kmeans_silhouette": lr.get("clustering", {}).get("kmeans", {}).get("best_silhouette", ""),
            "nn_stability":      lr.get("nn_stability", ""),
            "cka":               lr.get("cka_prev", ""),
        }
        for beta in BETA_VALUES:
            row[f"energy_beta{beta}"] = lr.get("energies", {}).get(beta, "")
        sk = lr.get("sinkhorn", {})
        if sk:
            row["fiedler_mean"]      = sk.get("fiedler_mean", "")
            row["sinkhorn_k_mean"]   = sk.get("sinkhorn_cluster_count_mean", "")
            row["attn_entropy_mean"] = lr.get("attention_entropy_mean", "")
            row["row_col_balance"]   = sk.get("row_col_balance_mean", "")
        rows.append(row)

    if rows:
        with open(run_dir / "layer_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def _save_tokens(results, run_dir):
    with open(run_dir / "tokens.txt", "w") as f:
        for i, tok in enumerate(results["tokens"]):
            f.write(f"{i:3d}  {tok}\n")


def _save_activations(hidden_states, run_dir):
    """
    activations.npz: sphere-projected activations, plus the per-token norms
    that projection discards.

    `activations` is unit-norm, which is what every Phase 1 metric operates
    on and what Phases 2/5b/6 expect — that stays unchanged. But writing
    only the unit vectors threw away the radius at save time, and the radius
    is not recoverable afterwards. That made core/polar.py's analyses
    (attention-sink norm outliers, cluster/norm coupling, sphere_gap) and
    the ParticleTable norm column impossible to compute from any saved run,
    which is a re-extraction of the whole model to recover something that
    was in memory at write time.

    `norms` is (n_layers, n_tokens) and reconstructs the raw activations
    exactly: raw = norms[..., None] * activations. Adding a key to the npz
    is backward-compatible — every existing reader indexes `activations`
    by name — and older run directories simply have no `norms` entry.
    """
    if not hidden_states:
        return
    import torch
    from core.models import layernorm_to_sphere

    stacked   = torch.stack(hidden_states)
    act_stack = layernorm_to_sphere(stacked).numpy()

    raw   = np.asarray(stacked.numpy() if hasattr(stacked, "numpy") else stacked)
    norms = np.linalg.norm(raw, axis=-1).astype(np.float32)

    np.savez_compressed(
        run_dir / "activations.npz",
        activations=act_stack,
        norms=norms,
    )


def _save_attentions(attentions, run_dir):
    if not attentions:
        return
    attn_stack = np.stack([a.numpy() for a in attentions])
    np.savez_compressed(run_dir / "attentions.npz", attentions=attn_stack)


def _save_clusters(results, run_dir):
    """
    clusters.npz: kmeans labels+centroids, HDBSCAN labels,
    agglomerative mid-threshold labels.
    """
    arrays = {}
    for lr in results["layers"]:
        i   = lr["layer"]
        cl  = lr.get("clustering", {})
        km  = cl.get("kmeans", {})
        hdb = cl.get("hdbscan", {})
        agg = cl.get("agglomerative", {})

        if "labels" in km:
            arrays[f"kmeans_labels_L{i}"] = np.array(km["labels"], dtype=np.int32)
        if "cluster_centroids_kmeans" in lr:
            arrays[f"kmeans_centroids_L{i}"] = np.array(
                lr["cluster_centroids_kmeans"], dtype=np.float32
            )
        if "labels" in hdb:
            arrays[f"hdbscan_labels_L{i}"] = np.array(hdb["labels"], dtype=np.int32)
        if "mid_labels" in agg:
            arrays[f"agglom_mid_labels_L{i}"] = np.array(agg["mid_labels"], dtype=np.int32)

    if arrays:
        np.savez_compressed(run_dir / "clusters.npz", **arrays)


def _save_pca_trajectories(results, run_dir):
    """PCA projections evicted from JSON -- (n_tokens, 3) per layer."""
    arrays = {}
    for i, proj in enumerate(results.get("pca_trajectories", [])):
        arrays[f"layer_{i}"] = np.array(proj, dtype=np.float32)
    if arrays:
        np.savez_compressed(run_dir / "pca_trajectories.npz", **arrays)


def _save_fiedler_vecs(results, run_dir):
    """Fiedler eigenvectors evicted from JSON -- (n_tokens,) per layer."""
    arrays = {}
    for lr in results["layers"]:
        fvec = lr.get("spectral", {}).get("fiedler_vec")
        if fvec is not None:
            arrays[f"fiedler_L{lr['layer']}"] = np.array(fvec, dtype=np.float32)
    if arrays:
        np.savez_compressed(run_dir / "fiedler_vecs.npz", **arrays)


def _save_centroid_trajectories(results, hidden_states, run_dir):
    tracking = results.get("cluster_tracking", {})
    if not tracking.get("trajectories"):
        return
    from .cluster_tracking import compute_centroid_trajectories

    label_arrays = []
    for lr in results["layers"]:
        hdb = lr.get("clustering", {}).get("hdbscan", {})
        if "labels" in hdb:
            label_arrays.append(np.array(hdb["labels"], dtype=np.int32))
        else:
            label_arrays.append(np.zeros(results["n_tokens"], dtype=np.int32))

    centroid_trajs = compute_centroid_trajectories(tracking, hidden_states, label_arrays)
    if centroid_trajs:
        arrays = {f"traj_{tid}": coords for tid, coords in centroid_trajs.items()}
        np.savez_compressed(run_dir / "centroid_trajectories.npz", **arrays)


def _save_plateau_attentions(results, attentions, run_dir):
    plateau_layers = results.get("plateau_layers", [])
    if not plateau_layers or not attentions:
        return
    arrays = {}
    for li in plateau_layers:
        if li < len(attentions):
            a = attentions[li]
            arrays[f"attn_L{li}"] = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    if arrays:
        np.savez_compressed(run_dir / "plateau_attentions.npz", **arrays)


# ---------------------------------------------------------------------------
# Saving -- main entry point
# ---------------------------------------------------------------------------

def save_run(results, hidden_states, attentions, run_dir):
    """
    Persist everything needed to reproduce plots and reports.
    One JSON per experiment type; large arrays to .npz.
    See module docstring for the full file list.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_tokens(results, run_dir)
    _save_geometry(results, run_dir)
    _save_energies(results, run_dir)
    _save_clustering(results, run_dir)
    _save_spectral(results, run_dir)
    _save_sinkhorn(results, run_dir)
    _save_trajectory(results, run_dir)
    _save_bridge_files(results, run_dir)
    _save_layer_metrics_csv(results, run_dir)
    _save_activations(hidden_states, run_dir)
    _save_attentions(attentions, run_dir)
    _save_clusters(results, run_dir)
    _save_centroid_trajectories(results, hidden_states, run_dir)
    _save_plateau_attentions(results, attentions, run_dir)
    _save_pca_trajectories(results, run_dir)
    _save_fiedler_vecs(results, run_dir)


# ---------------------------------------------------------------------------
# Global aggregation (called by run_1 after all prompt runs finish)
# ---------------------------------------------------------------------------

def aggregate_global_artifacts(all_results, phase1_dir):
    """
    Write aggregated artifacts to the phase1_dir root.

    Produces:
      pair_agreement.json          per-prompt pair-agreement summary
      centroid_trajectories.npz    all per-prompt centroid trajectories merged,
                                   keyed as {prompt_slug}__traj_{tid} so
                                   analyses that load phase1_dir/centroid_trajectories.npz
                                   find a real file without per-prompt path logic
    """
    phase1_dir = Path(phase1_dir)

    # --- pair_agreement.json ---
    pa_out = {}
    for r in all_results:
        prompt_key = r.get("prompt", "unknown")
        layers     = r.get("layers", [])
        n_semantic = sum(lr.get("pair_agreement", {}).get("n_semantic", 0) for lr in layers)
        n_artifact = sum(lr.get("pair_agreement", {}).get("n_artifact", 0) for lr in layers)
        n_noise    = sum(lr.get("pair_agreement", {}).get("n_noise",    0) for lr in layers)
        total      = n_semantic + n_artifact + n_noise
        pa_out[prompt_key] = {
            "n_semantic":       n_semantic,
            "n_artifact":       n_artifact,
            "n_noise":          n_noise,
            "artifact_fraction": round(n_artifact / total, 4) if total else 0.0,
            "plateau_layers":   r.get("plateau_layers", []),
            "per_layer": [
                {
                    "layer":             lr["layer"],
                    "n_semantic":        lr.get("pair_agreement", {}).get("n_semantic", 0),
                    "n_artifact":        lr.get("pair_agreement", {}).get("n_artifact", 0),
                    "artifact_fraction": lr.get("pair_agreement", {}).get("artifact_fraction", 0.0),
                }
                for lr in layers
            ],
        }
    _jdump(pa_out, phase1_dir / "pair_agreement.json")
    print(f"  Wrote pair_agreement.json ({len(pa_out)} prompts)")

    # --- centroid_trajectories.npz at root ---
    # Merge all per-prompt centroid_trajectories.npz files into one, keyed as
    # {prompt_slug}__traj_{tid}.  Analysis functions that build the path as
    # phase1_dir/"centroid_trajectories.npz" will find a real file; those that
    # know about prompt namespacing can iterate the keys.
    # Scan per-prompt subdirectories; we don't have the raw hidden_states here.
    ct_merged = {}
    first_model = all_results[0].get("model", "") if all_results else ""
    model_slug  = first_model.replace("/", "_").replace("-", "_")
    for sub in sorted(phase1_dir.iterdir()):
        if not sub.is_dir():
            continue
        if model_slug and model_slug not in sub.name:
            continue
        ct_path = sub / "centroid_trajectories.npz"
        if not ct_path.exists():
            continue
        # Infer prompt slug from directory name by stripping model prefix
        prompt_slug = sub.name.replace(model_slug + "_", "", 1)
        data = np.load(ct_path)
        for k in data.files:
            tid = k.split("_")[1]   # "traj_N" -> "N"
            ct_merged[f"{prompt_slug}__traj_{tid}"] = data[k]

    if ct_merged:
        np.savez_compressed(phase1_dir / "centroid_trajectories.npz", **ct_merged)
        print(f"  Wrote centroid_trajectories.npz ({len(ct_merged)} trajectories)")


# ---------------------------------------------------------------------------
# Loading -- auto-detect format
# ---------------------------------------------------------------------------

def load_run(run_dir):
    """
    Restore a results dict from a saved run directory.
    Auto-detects v1 (metrics.json) vs v2 (split JSON files).
    """
    run_dir = Path(run_dir)
    if (run_dir / "metrics.json").exists():
        return _load_run_legacy(run_dir)
    return _load_run_v2(run_dir)


def _load_run_legacy(run_dir):
    """Load v1 format (single metrics.json) with backward-compat patches."""
    with open(run_dir / "metrics.json") as f:
        results = json.load(f)

    if "pca_trajectories" not in results:
        results["pca_trajectories"] = []
    if "v_spectrum" not in results:
        results["v_spectrum"] = {}

    for layer in results.get("layers", []):
        if "cka_prev" not in layer:
            layer["cka_prev"] = float("nan")
        if "energy_drop_pairs" not in layer:
            layer["energy_drop_pairs"] = {}
        elif isinstance(layer["energy_drop_pairs"], list):
            layer["energy_drop_pairs"] = {1.0: layer["energy_drop_pairs"]}
        elif isinstance(layer["energy_drop_pairs"], dict):
            layer["energy_drop_pairs"] = {
                float(k): v for k, v in layer["energy_drop_pairs"].items()
            }
        if "energies" in layer:
            layer["energies"] = {float(k): v for k, v in layer["energies"].items()}
        if "nesting" not in layer:
            layer["nesting"] = {
                "global_spectral_k": layer.get("spectral", {}).get("k_eigengap", 1),
                "per_cluster": {}, "has_nesting": False,
                "nesting_summary": "not computed (old run)",
                "n_clusters_with_substructure": 0,
            }
        if "pair_agreement" not in layer:
            layer["pair_agreement"] = {
                "mutual_pairs": [], "n_semantic": 0, "n_artifact": 0,
                "n_noise": 0, "artifact_fraction": 0.0,
            }

    if "cluster_tracking" not in results:
        results["cluster_tracking"] = {
            "events": [], "trajectories": [],
            "summary": {"total_births": 0, "total_deaths": 0, "total_merges": 0,
                        "max_alive": 0, "n_trajectories": 0,
                        "mean_lifespan": 0.0, "max_lifespan": 0},
        }
    if "plateau_layers" not in results:
        results["plateau_layers"] = []

    print(f"Loaded (v1): {results['model']} | {results['prompt']}")
    return results


def _load_run_v2(run_dir):
    """
    Load v2 format (split files) and reassemble the canonical results dict
    expected by plots.py / reporting.py / Phase 3+.
    """
    with open(run_dir / "geometry.json") as f:
        geo = json.load(f)
    with open(run_dir / "trajectory.json") as f:
        traj = json.load(f)

    n_layers  = geo["n_layers"]
    layer_map = {lr["layer"]: dict(lr) for lr in geo["layers"]}

    # Merge each optional experiment file into the layer map
    def _merge(fname):
        path = run_dir / fname
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        rows = data.get("layers", [])
        for row in rows:
            li = row.get("layer")
            if li is not None and li in layer_map:
                layer_map[li].update(row)

    for fname in ("energies.json", "clustering.json", "spectral.json", "sinkhorn.json"):
        _merge(fname)

    # Rehydrate float keys that JSON stringified
    for lr in layer_map.values():
        if "energies" in lr:
            lr["energies"] = {float(k): v for k, v in lr["energies"].items()}
        if "energy_drop_pairs" in lr:
            lr["energy_drop_pairs"] = {float(k): v for k, v in lr["energy_drop_pairs"].items()}

        # Reconstruct spectral sub-dict
        if "k_eigengap" in lr and "spectral" not in lr:
            lr["spectral"] = {
                "k_eigengap": lr.pop("k_eigengap"),
                "eigenvalues": lr.pop("eigenvalues", []),
                "eigengaps":   lr.pop("eigengaps", []),
            }
        # Reconstruct sinkhorn sub-dict
        if "fiedler_mean" in lr and "sinkhorn" not in lr:
            lr["sinkhorn"] = {
                "fiedler_mean":               lr.pop("fiedler_mean", None),
                "sinkhorn_cluster_count_mean": lr.pop("sinkhorn_cluster_count_mean", None),
                "row_col_balance_mean":        lr.pop("row_col_balance_mean", None),
            }

    # Inject large arrays from npz files
    clusters_path = run_dir / "clusters.npz"
    if clusters_path.exists():
        cdata = np.load(clusters_path)
        for li, lr in layer_map.items():
            cl  = lr.setdefault("clustering", {})
            km  = cl.setdefault("kmeans", {})
            hdb = cl.setdefault("hdbscan", {})
            agg = cl.setdefault("agglomerative", {})
            if f"kmeans_labels_L{li}" in cdata.files:
                km["labels"] = cdata[f"kmeans_labels_L{li}"].tolist()
            if f"kmeans_centroids_L{li}" in cdata.files:
                lr["cluster_centroids_kmeans"] = cdata[f"kmeans_centroids_L{li}"].tolist()
            if f"hdbscan_labels_L{li}" in cdata.files:
                hdb["labels"] = cdata[f"hdbscan_labels_L{li}"].tolist()
            if f"agglom_mid_labels_L{li}" in cdata.files:
                agg["mid_labels"] = cdata[f"agglom_mid_labels_L{li}"].tolist()

    fvec_path = run_dir / "fiedler_vecs.npz"
    if fvec_path.exists():
        fdata = np.load(fvec_path)
        for li, lr in layer_map.items():
            key = f"fiedler_L{li}"
            if key in fdata.files:
                lr.setdefault("spectral", {})["fiedler_vec"] = fdata[key].tolist()

    pca_trajs = []
    pca_path  = run_dir / "pca_trajectories.npz"
    if pca_path.exists():
        pdata    = np.load(pca_path)
        pca_trajs = [
            pdata[f"layer_{i}"].tolist()
            for i in range(n_layers)
            if f"layer_{i}" in pdata.files
        ]

    layers = [layer_map[i] for i in range(n_layers) if i in layer_map]

    for lr in layers:
        if "cka_prev" not in lr:
            lr["cka_prev"] = float("nan")
        if "nesting" not in lr:
            lr["nesting"] = {
                "global_spectral_k": lr.get("spectral", {}).get("k_eigengap", 1),
                "per_cluster": {}, "has_nesting": False,
                "nesting_summary": "not computed", "n_clusters_with_substructure": 0,
            }
        if "pair_agreement" not in lr:
            lr["pair_agreement"] = {
                "mutual_pairs": [], "n_semantic": 0, "n_artifact": 0,
                "n_noise": 0, "artifact_fraction": 0.0,
            }

    results = {
        "model":            geo["model"],
        "prompt":           geo["prompt"],
        "tokens":           geo["tokens"],
        "n_layers":         n_layers,
        "n_tokens":         geo["n_tokens"],
        "d_model":          geo["d_model"],
        "layers":           layers,
        "pca_trajectories": pca_trajs,
        "cluster_tracking": traj.get("cluster_tracking", {
            "events": [], "trajectories": [],
            "summary": {"total_births": 0, "total_deaths": 0, "total_merges": 0,
                        "max_alive": 0, "n_trajectories": 0,
                        "mean_lifespan": 0.0, "max_lifespan": 0},
        }),
        "plateau_layers":   traj.get("plateau_layers", []),
        "v_spectrum":       {},
    }

    print(f"Loaded (v2): {results['model']} | {results['prompt']}")
    return results


# ---------------------------------------------------------------------------
# Loaders for individual artifact files
# ---------------------------------------------------------------------------

def load_activations(run_dir):
    """Returns (n_layers, n_tokens, d_model) float32."""
    data = np.load(Path(run_dir) / "activations.npz")
    return data["activations"]


def load_attentions(run_dir):
    """Returns (n_layers, n_heads, n_tokens, n_tokens) float32."""
    data = np.load(Path(run_dir) / "attentions.npz")
    return data["attentions"]


def load_clusters(run_dir):
    """
    Returns dict with keys:
      kmeans_labels, kmeans_centroids, hdbscan_labels
    Each is a list of per-layer arrays, ordered by layer index.
    """
    path = Path(run_dir) / "clusters.npz"
    data = np.load(path)
    layer_indices = sorted(
        int(k.split("_L")[1])
        for k in data.files
        if k.startswith("kmeans_labels_L")
    )
    return {
        "kmeans_labels":    [data[f"kmeans_labels_L{i}"]    for i in layer_indices],
        "kmeans_centroids": [data[f"kmeans_centroids_L{i}"] for i in layer_indices
                             if f"kmeans_centroids_L{i}" in data.files],
        "hdbscan_labels":   [data[f"hdbscan_labels_L{i}"]   for i in layer_indices
                             if f"hdbscan_labels_L{i}" in data.files],
    }


def load_centroid_trajectories(run_dir):
    """Returns {trajectory_id (int): (lifespan, d) float32}."""
    path = Path(run_dir) / "centroid_trajectories.npz"
    data = np.load(path)
    return {int(k.split("_")[1]): data[k] for k in data.files}


def load_plateau_attentions(run_dir):
    """Returns {layer_index (int): (n_heads, n_tokens, n_tokens) float32}."""
    path = Path(run_dir) / "plateau_attentions.npz"
    data = np.load(path)
    return {int(k.split("_L")[1]): data[k] for k in data.files}


def load_pca_trajectories(run_dir):
    """Returns list of (n_tokens, 3) float32 arrays, one per layer."""
    path = Path(run_dir) / "pca_trajectories.npz"
    data = np.load(path)
    n = max(int(k.split("_")[1]) for k in data.files) + 1
    return [data[f"layer_{i}"] for i in range(n) if f"layer_{i}" in data.files]


def load_fiedler_vecs(run_dir):
    """Returns {layer_index (int): (n_tokens,) float32}."""
    path = Path(run_dir) / "fiedler_vecs.npz"
    data = np.load(path)
    return {int(k.split("_L")[1]): data[k] for k in data.files}


# ===========================================================================
# Cross-phase artifact loader — absorbed from core/io.py.
#
# find_phase1_run_dir resolution order
# -------------------------------------
# Phase 1 writes flat dirs like:
#   phase1_dir/albert-xlarge-v2_48iter_wiki_paragraph/
#
# This resolves:
#   1. phase1_dir / stem / *prompt_key*      (legacy nested layout)
#   2. phase1_dir / *{model_any_form}*{prompt}*   (flat layout)
#   3. Any dir containing either model fragment or prompt fragment
#   4. None if nothing matches
# ===========================================================================
# Variant tags run_1.py appends between model and prompt: ALBERT snapshot
# depth ("48iter") and sublayer streams ("attn"/"ffn"). Anything else in
# that position is a different model, not a variant of this one.
_VARIANT_TAG_RE = re.compile(r"^(?:\d+iter|attn|ffn)$")


def _norm(s: str) -> str:
    return s.replace("-", "_").replace("/", "_").replace("@", "_").lower()


def _stem_matches(dirname: str, stem: str) -> bool:
    """True if dirname starts with stem, permissively.

    Used only by the fallback tiers (3 and 4) of find_phase1_run_dir, after
    _exact_run_match (tier 2) has already failed to find an exact directory
    for (stem, prompt_key). Tier 2 is where exactness is enforced — it
    rejects "pythia_410m_step1000..." for a wanted stem of
    "pythia_410m_step1" regardless of what this function does. Anchoring
    this predicate at a segment boundary as well made it reject the same
    neighbour in the fallback tiers too, which turned "return the nearest
    neighbour with a warning" into "return None silently" whenever no exact
    directory existed — the one failure mode find_phase1_run_dir's docstring
    says the fallback tiers must not produce. Every caller of this function
    is inside _resolve_inexact, which always warns, so permissiveness here
    is safe: it can only ever produce an audible neighbour, never a silent
    exact-looking match.
    """
    d, s = _norm(dirname), _norm(stem)
    if not s:
        return True                      # preserved: empty stem matches all
    return d.startswith(s)


def _exact_run_match(dirname: str, stem: str, prompt_key: str) -> bool:
    """True only for the name run_1 would have written for (stem, prompt)."""
    d, s, p = _norm(dirname), _norm(stem), _norm(prompt_key)
    suffix = "_" + p
    if not d.endswith(suffix):
        return False
    head = d[: -len(suffix)]
    if head == s:
        return True
    if head.startswith(s + "_"):
        return bool(_VARIANT_TAG_RE.match(head[len(s) + 1:]))
    return False


def _newest(dirs):
    return sorted(dirs, key=lambda p: p.stat().st_mtime)[-1]


def _resolve_inexact(candidates, stem, prompt_key, model_name):
    """Rank inexact matches by leftover name length, then recency, and warn.

    An inexact match on a checkpoint-suffixed registry is exactly how the
    wrong Pythia step gets loaded without anything failing, so this path is
    audible rather than silent.
    """
    n = len(_norm(stem))
    ranked = sorted(candidates,
                    key=lambda d: (len(_norm(d.name)) - n, -d.stat().st_mtime))
    chosen = ranked[0]
    warnings.warn(
        f"find_phase1_run_dir: no exact run directory for "
        f"({model_name!r}, {prompt_key!r}); using {chosen.name!r} from "
        f"{[d.name for d in ranked]}. Confirm this is the intended "
        f"checkpoint before trusting downstream results.",
        stacklevel=3,
    )
    return chosen


def find_phase1_run_dir(
    phase1_dir:  Path,
    model_name:  str,
    prompt_key:  str,
) -> Optional[Path]:
    """
    Locate the Phase 1 run directory for (model_name, prompt_key).

    Resolution order, tightest first:
      1. legacy nested layout   phase1_dir/{stem}/*{prompt}*
      2. exact flat name        phase1_dir/{stem}[_{tag}]_{prompt}
      3. prefix + prompt substring          (warns)
      4. prefix only, prompt absent         (warns)

    Steps 3 and 4 exist for pre-v2 directory names and are the only paths
    that can return a directory for a different model; both warn.
    """
    phase1_dir = Path(phase1_dir)
    if not phase1_dir.exists():
        return None

    stem_under  = model_name.replace("-", "_").replace("/", "_")
    stem_hyphen = model_name.replace("_", "-").replace("/", "-")

    nested = phase1_dir / stem_under
    if nested.is_dir():
        candidates = sorted(nested.glob(f"*{prompt_key}*"),
                            key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
        all_sub = [d for d in nested.iterdir() if d.is_dir()]
        if all_sub:
            return _newest(all_sub)

    subdirs = [d for d in phase1_dir.iterdir() if d.is_dir()]

    exact = [d for d in subdirs if _exact_run_match(d.name, stem_under, prompt_key)]
    if exact:
        return _newest(exact)

    for stem_form in (stem_hyphen, stem_under):
        loose = [d for d in subdirs
                 if _stem_matches(d.name, stem_form) and prompt_key in d.name]
        if loose:
            return _resolve_inexact(loose, stem_form, prompt_key, model_name)

    for stem_form in (stem_hyphen, stem_under):
        loose = [d for d in subdirs if _stem_matches(d.name, stem_form)]
        if loose:
            return _resolve_inexact(loose, stem_form, prompt_key, model_name)

    return None


def load_phase1_run(run_dir: Path) -> dict:
    """
    Load Phase 1 on-disk artifacts for a single (model, prompt) run and
    return a flat dict conforming to the cross-phase contract used by
    Phases 5, 5b, and 6.

    THE canonical Phase 1 reader — p5_single_mstate_analysis/p5_io.py and
    p5b_manifold_steering/p5b_io.py both delegate to this function and add
    only their own phase-specific derived values on top.

    Parameters
    ----------
    run_dir : path to the per-prompt run directory written by save_run
              (this module).

    Returns
    -------
    dict with keys:
      activations     : ndarray (n_layers, n_tokens, d) or None
      attentions      : ndarray (n_layers, n_heads, n_tokens, n_tokens) or None
      tokens          : list[str] of length n_tokens
      hdbscan_labels  : list[ndarray(n_tokens,)] or None — a LIST, never a
                        string-keyed dict, indexed by layer position
                        0..n_layers-1, gaps filled with all-(-1) noise arrays
      events          : list[dict] — normalised merge/violation event records
      merge_layers    : list[int] — raw merge layer indices from events.json.
                        Distinct from the normalised `events` records above;
                        p5b_io needs the raw list, run_6/build_context needs
                        the normalised records, so both are provided.
      trajectories    : list[dict] — cluster chain trajectories. Read from
                        trajectory.json's cluster_tracking.trajectories (the
                        real on-disk shape written by _save_trajectory
                        above), falling back to a top-level "trajectories"
                        key for older/synthetic fixtures that don't nest it.
                        The prior version of this function (core/io.py) only
                        checked the top-level key, which real Phase 1 output
                        never has — it had been silently returning [] on
                        every real run.
      plateau_layers  : list[int]
      centroid_trajs  : {int trajectory_id: ndarray (lifespan, d_model)},
                        read from centroid_trajectories.npz (keys "traj_{id}")
      n_layers        : int
      n_tokens        : int
      d_model         : int
      prompt          : str
      model           : str
      run_dir         : str

    Never raises on missing or malformed files — absent artifacts become
    None / [] / {} / 0 / "" so callers can degrade gracefully.

    Key invariant: hdbscan_labels is always a list (never a string-keyed
    dict) so that ctx["labels_per_layer"][L] works with integer L in every
    Track B/D sub-experiment.
    """
    run_dir = Path(run_dir)
    out: dict = {}

    # ------------------------------------------------------------------
    # activations.npz  →  (n_layers, n_tokens, d)
    # ------------------------------------------------------------------
    acts_path = run_dir / "activations.npz"
    if acts_path.exists():
        data = np.load(acts_path)
        key  = "activations" if "activations" in data else list(data.keys())[0]
        out["activations"] = data[key]
    else:
        out["activations"] = None

    n_layers = out["activations"].shape[0] if out["activations"] is not None else 0
    n_tokens = out["activations"].shape[1] if out["activations"] is not None else 0
    d_model  = out["activations"].shape[2] if out["activations"] is not None else 0

    # ------------------------------------------------------------------
    # geometry.json  →  model, prompt, and n_layers/n_tokens/d_model
    # fallback for when activations.npz is absent. Tolerant of a missing or
    # corrupt file — every other artifact here already is, and this one
    # previously wasn't in the p5b-local direct loader this replaces.
    # ------------------------------------------------------------------
    geo_path = run_dir / "geometry.json"
    geo: dict = {}
    if geo_path.exists():
        try:
            with open(geo_path) as f:
                geo = json.load(f)
        except Exception:
            geo = {}

    out["prompt"]   = geo.get("prompt", "")
    out["model"]    = geo.get("model", "")
    out["n_layers"] = n_layers or geo.get("n_layers", 0)
    out["n_tokens"] = n_tokens or geo.get("n_tokens", 0)
    out["d_model"]  = d_model  or geo.get("d_model", 0)

    # ------------------------------------------------------------------
    # tokens.txt  →  list[str]
    # ------------------------------------------------------------------
    out["tokens"] = _load_tokens(run_dir, out["n_tokens"])

    # ------------------------------------------------------------------
    # hdbscan_labels.json  →  list[ndarray]  (NOT dict)
    #
    # Phase 1 writes: {"0": [0,1,0,...], "2": [...], ...}
    # We must convert to a list indexed by layer position so that
    # ctx["labels_per_layer"][L] works with integer L.
    # ------------------------------------------------------------------
    out["hdbscan_labels"] = _load_hdbscan_labels(
        run_dir / "hdbscan_labels.json", out["n_layers"], out["n_tokens"]
    )

    # ------------------------------------------------------------------
    # events.json  →  normalised list[dict] + raw merge_layers list
    # ------------------------------------------------------------------
    events_path = run_dir / "events.json"
    out["events"] = _load_events(events_path)

    if events_path.exists():
        try:
            with open(events_path) as f:
                ev_raw = json.load(f)
            out["merge_layers"] = [int(l) for l in ev_raw.get("merge_layers", [])]
        except Exception:
            out["merge_layers"] = []
    else:
        out["merge_layers"] = []

    # ------------------------------------------------------------------
    # trajectory.json  →  trajectories list + plateau_layers
    #
    # See docstring above re: cluster_tracking nesting fix.
    # ------------------------------------------------------------------
    traj_path = run_dir / "trajectory.json"
    if traj_path.exists():
        try:
            with open(traj_path) as f:
                tj = json.load(f)
            out["trajectories"] = (
                tj.get("cluster_tracking", {}).get("trajectories")
                or tj.get("trajectories", [])
            )
            out["plateau_layers"] = tj.get("plateau_layers", [])
        except Exception:
            out["trajectories"]   = []
            out["plateau_layers"] = []
    else:
        out["trajectories"]   = []
        out["plateau_layers"] = []

    # ------------------------------------------------------------------
    # centroid_trajectories.npz  →  {int trajectory_id: (lifespan, d)}
    # ------------------------------------------------------------------
    ct_path = run_dir / "centroid_trajectories.npz"
    out["centroid_trajs"] = {}
    if ct_path.exists():
        try:
            ct_data = np.load(ct_path)
            for k in ct_data.files:
                if k.startswith("traj_"):
                    try:
                        tid = int(k[len("traj_"):])
                        out["centroid_trajs"][tid] = ct_data[k].astype(np.float32)
                    except (ValueError, Exception):
                        pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # attentions.npz  →  (n_layers, n_heads, n_tokens, n_tokens) or None
    # ------------------------------------------------------------------
    attn_path = run_dir / "attentions.npz"
    if attn_path.exists():
        try:
            adata = np.load(attn_path)
            akey  = "attentions" if "attentions" in adata else list(adata.keys())[0]
            out["attentions"] = adata[akey]
        except Exception:
            out["attentions"] = None
    else:
        out["attentions"] = None

    out["run_dir"] = str(run_dir)

    return out


def _load_tokens(run_dir: Path, n_tokens: int) -> list[str]:
    """
    Load tokens from tokens.txt (tab-separated index\\ttoken per line).
    Falls back to synthetic names if file absent or malformed.
    """
    txt_path = run_dir / "tokens.txt"
    if txt_path.exists():
        try:
            toks = []
            with open(txt_path) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if "\t" in line:
                        _, tok = line.split("\t", 1)
                    else:
                        tok = line
                    toks.append(tok)
            if toks:
                return toks
        except Exception:
            pass
    # Fallback: try tokens.json (alternate format)
    json_path = run_dir / "tokens.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                toks = json.load(f)
            if isinstance(toks, list):
                return [str(t) for t in toks]
        except Exception:
            pass
    return [f"<tok_{i}>" for i in range(n_tokens)]


def _load_hdbscan_labels(
    path:     Path,
    n_layers: int,
    n_tokens: int,
) -> list[np.ndarray] | None:
    """
    Convert hdbscan_labels.json from string-keyed dict to list[ndarray].

    Input (on disk): {"0": [0, 1, 0, ...], "1": [...], ...}
    Output:          [ndarray([0,1,0,...]), ndarray([...]), ...]
                      indexed by layer position 0..n_layers-1

    Gaps (layers absent from the JSON) are filled with noise-label arrays
    (all -1) so downstream code never receives None in the list.
    """
    if not path.exists():
        return None

    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception as e:
        warnings.warn(f"load_phase1_run: could not parse {path}: {e}")
        return None

    if not isinstance(raw, dict):
        warnings.warn(f"load_phase1_run: expected dict in {path}, got {type(raw)}")
        return None

    # Infer n_layers from the dict if activations weren't loaded
    effective_n = n_layers if n_layers > 0 else (
        max(int(k) for k in raw) + 1 if raw else 0
    )
    if effective_n == 0:
        return None

    noise = np.full(n_tokens, -1, dtype=np.int32)
    result: list[np.ndarray] = []

    for L in range(effective_n):
        if str(L) in raw:
            result.append(np.array(raw[str(L)], dtype=np.int32))
        else:
            result.append(noise.copy())

    return result


def _load_events(path: Path) -> list[dict]:
    """
    Load events.json and return a flat list of event dicts.

    Phase 1 writes events.json as:
      {"merge_layers": [2, 5], "energy_violations": {"1.0": [3, 4]}}

    We normalise this into a list of {"type": ..., "layer": ...} dicts
    matching what build_context and _classify_layer_types expect.
    """
    if not path.exists():
        return []
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return []

    events: list[dict] = []

    for layer in raw.get("merge_layers", []):
        events.append({"type": "merge", "layer_name": str(layer), "layer_from": str(layer)})

    for beta_str, layers in raw.get("energy_violations", {}).items():
        for layer in layers:
            events.append({"type": "energy_violation", "layer": layer, "beta": float(beta_str)})

    return events


# ---------------------------------------------------------------------------
# Replot from saved run
# ---------------------------------------------------------------------------

def replot_all(run_dir, out_dir=None):
    """
    Recreate every plot from a saved run directory.
    No model loading required.
    """
    from .plots import (
        plot_trajectory, plot_ip_histograms, plot_pca_panels,
        plot_sinkhorn_detail, plot_spectral_eigengap,
        plot_eigenvalue_spectra, plot_cka_trajectory,
    )
    from .reporting_p1 import print_summary

    run_dir = Path(run_dir)
    out_dir = out_dir or run_dir
    results = load_run(run_dir)

    print("Regenerating plots...")
    plot_trajectory(results, out_dir)
    plot_ip_histograms(results, out_dir)
    plot_pca_panels(results, out_dir)
    plot_sinkhorn_detail(results, out_dir)
    plot_spectral_eigengap(results, out_dir)
    plot_eigenvalue_spectra(results, out_dir)
    plot_cka_trajectory(results, out_dir)
    print_summary(results)
    print(f"Done. Plots written to {out_dir}/")