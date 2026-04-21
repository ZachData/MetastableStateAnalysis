"""
io_utils.py — Save and load run artifacts.

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

import csv
import json
import numpy as np
import torch
from pathlib import Path

from core.config import BETA_VALUES
from core.models import layernorm_to_sphere


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

def _save_geometry(results, run_dir):
    """ip stats, CKA, effective rank, NN stability per layer."""
    layers_out = []
    for lr in results["layers"]:
        layers_out.append({
            "layer":                    lr["layer"],
            "ip_mean":                  lr["ip_mean"],
            "ip_std":                   lr["ip_std"],
            "ip_mass_near_1":           lr["ip_mass_near_1"],
            "ip_histogram":             lr.get("ip_histogram", []),
            "effective_rank":           lr["effective_rank"],
            "cka_prev":                 lr.get("cka_prev"),
            "nn_stability":             lr.get("nn_stability"),
            "nn_indices":               lr.get("nn_indices", []),
            "pca_explained_variance":   lr.get("pca_explained_variance", []),
        })
    _jdump({
        "model":    results["model"],
        "prompt":   results["prompt"],
        "n_layers": results["n_layers"],
        "n_tokens": results["n_tokens"],
        "d_model":  results["d_model"],
        "tokens":   results["tokens"],
        "layers":   layers_out,
    }, run_dir / "geometry.json")


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

    merge_layers = sorted({
        e["layer"] for e in events if e.get("type") == "merge"
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
    if not hidden_states:
        return
    stacked   = torch.stack(hidden_states)
    act_stack = layernorm_to_sphere(stacked).numpy()
    np.savez_compressed(run_dir / "activations.npz", activations=act_stack)


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
    from .reporting import print_summary

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
