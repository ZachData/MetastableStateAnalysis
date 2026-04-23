"""
p4_mstate_features/analysis.py — Cross-track comparison, verdicts, and I/O.

Responsibilities
----------------
1. cross_track_agreement   — correlate Track 1 / 2 / 3 layer-level signals
2. build_phase4_verdict    — structured verdict dict consumed by Phase 5/6
3. Per-track save helpers  — each called immediately after its track finishes
     save_track1_outputs(out_dir, t1_results)
     save_track2_outputs(out_dir, t2_results)
     save_track3_outputs(out_dir, t3_results)
4. write_llm_summary       — concise human/LLM-readable summary.txt
5. save_phase4_outputs     — thin wrapper that calls per-track helpers +
                             verdict + summary (backward-compatible)

The design intent is that each track saves its own files *immediately* after
it runs so partial results are always on disk, even if later tracks crash.
The final summary.txt is written only after all tracks complete and the
verdict is assembled.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _jdump(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# 1. Cross-track agreement
# ---------------------------------------------------------------------------

def cross_track_agreement(
    t1_results: dict,
    t2_results: dict,
    t3_results: Optional[dict],
) -> dict:
    """
    Correlate per-layer signals across tracks.

    Track 1: chorus ARI per layer  (proxy: cluster-feature correspondence)
    Track 2: probe accuracy per layer (proxy: linear separability)
    Track 3: (optional) reconstruction ratio per prompt

    Returns
    -------
    {
      "t1_t2_correlation": {"spearman_rho": float, "pval": float,
                            "n_layers": int, "interpretation": str},
      "t1_available": bool,
      "t2_available": bool,
      "t3_available": bool,
      "cross_track_n": int,   # number of prompts/layers both tracks covered
    }
    """
    from scipy.stats import spearmanr

    agreement = {
        "t1_available": bool(t1_results),
        "t2_available": bool(t2_results),
        "t3_available": bool(t3_results),
    }

    # Build parallel layer vectors for T1 and T2
    t1_per_layer = t1_results.get("chorus_per_layer", {})
    t2_per_layer = t2_results.get("probe_per_layer", {})

    shared_layers = sorted(set(t1_per_layer) & set(t2_per_layer))
    agreement["cross_track_n"] = len(shared_layers)

    if len(shared_layers) >= 4:
        t1_vals = np.array([
            t1_per_layer[lk].get("ari", {}).get("ari", float("nan"))
            for lk in shared_layers
        ])
        t2_vals = np.array([
            t2_per_layer[lk].get("accuracy", float("nan"))
            for lk in shared_layers
        ])

        mask = np.isfinite(t1_vals) & np.isfinite(t2_vals)
        if mask.sum() >= 4:
            rho, pval = spearmanr(t1_vals[mask], t2_vals[mask])
            agreement["t1_t2_correlation"] = {
                "spearman_rho": float(rho),
                "pval": float(pval),
                "n_layers": int(mask.sum()),
                "interpretation": (
                    "converging" if abs(rho) > 0.5 and pval < 0.05
                    else "weak" if abs(rho) > 0.3
                    else "null"
                ),
            }
        else:
            agreement["t1_t2_correlation"] = {
                "spearman_rho": float("nan"),
                "pval": float("nan"),
                "n_layers": int(mask.sum()),
                "interpretation": "insufficient_data",
            }
    else:
        agreement["t1_t2_correlation"] = {
            "spearman_rho": float("nan"),
            "n_layers": len(shared_layers),
            "interpretation": "insufficient_data",
        }

    # T3: note whether reconstruction ratio favours LRAE
    if t3_results:
        recon = t3_results.get("reconstruction", {})
        ratios = [v.get("ratio", 1.0) for v in recon.values() if isinstance(v, dict)]
        agreement["t3_mean_recon_ratio"] = float(np.mean(ratios)) if ratios else None

    return agreement


# ---------------------------------------------------------------------------
# 2. Phase 4 verdict
# ---------------------------------------------------------------------------

def build_phase4_verdict(
    t1_results: dict,
    t2_results: dict,
    t3_results: Optional[dict],
    agreement: dict,
    plateau_alignment: dict,
) -> dict:
    """
    Assemble a structured verdict dict.

    verdict["tracks"]  — per-track sub-verdicts
    verdict["overall"] — one of:
        "metastable_features_detected" | "partial_signal" | "cross_track_null"
    verdict["interpretation"] — plain-text conclusion (set on null)
    """
    verdict = {"tracks": {}}

    # --- Track 1 ---
    t1 = {}
    cs = t1_results.get("chorus_summary", {})
    ms = t1_results.get("mi_summary", {})
    pa = plateau_alignment or {}

    t1["max_chorus_ari"]   = cs.get("max_ari", 0.0)
    t1["mean_chorus_ari"]  = cs.get("mean_ari", 0.0)
    t1["max_nmi"]          = ms.get("max_nmi", 0.0)
    t1["mean_nmi"]         = ms.get("mean_nmi", 0.0)
    t1["plateau_falsification"] = pa.get("falsification", "untestable")

    if t1["max_chorus_ari"] > 0.5 or t1["max_nmi"] > 0.3:
        t1["verdict"] = "crosscoder_tracks_clusters"
    elif t1["max_chorus_ari"] > 0.2 or t1["max_nmi"] > 0.1:
        t1["verdict"] = "weak_crosscoder_signal"
    else:
        t1["verdict"] = "crosscoder_null"

    verdict["tracks"]["track1_crosscoder"] = t1

    # --- Track 2 ---
    t2 = {}
    ps = t2_results.get("probe_summary", {})
    ls = t2_results.get("lda_summary", {})
    dp = t2_results.get("delta_pca_summary", {})

    t2["mean_probe_accuracy"] = ps.get("mean_accuracy", 0.0)
    t2["max_probe_accuracy"]  = ps.get("max_accuracy", 0.0)
    t2["mean_lda_cosine"]     = ls.get("mean_cosine", 0.0)
    t2["mean_update_var"]     = dp.get("mean_total_variance", 0.0)

    if t2["max_probe_accuracy"] > 0.8:
        t2["verdict"] = "strong_linear_separability"
    elif t2["max_probe_accuracy"] > 0.6:
        t2["verdict"] = "moderate_linear_separability"
    else:
        t2["verdict"] = "geometric_null"

    verdict["tracks"]["track2_geometric"] = t2

    # --- Track 3 (optional) ---
    if t3_results:
        t3 = {}
        va = t3_results.get("v_alignment", {})
        t3["mean_repulsive"] = va.get("mean_repulsive", 0.0)
        t3["mean_attractive"] = va.get("mean_attractive", 0.0)
        t3["n_repulsive_dominant"] = va.get("n_repulsive_dominant", 0)
        t3["n_attractive_dominant"] = va.get("n_attractive_dominant", 0)

        if max(t3["mean_repulsive"], t3["mean_attractive"]) > 0.1 and \
                abs(t3["mean_repulsive"] - t3["mean_attractive"]) > 0.05:
            t3["verdict"] = "v_alignment_recovered"
        else:
            t3["verdict"] = "v_alignment_still_null"

        verdict["tracks"]["track3_low_rank_ae"] = t3

    verdict["agreement"] = agreement

    # --- Overall ---
    track_verdicts = [
        v.get("verdict", "null")
        for v in verdict["tracks"].values()
    ]

    if any(
        "strong" in v or "crosscoder_tracks" in v or "recovered" in v
        for v in track_verdicts
    ):
        verdict["overall"] = "metastable_features_detected"
    elif any("moderate" in v or "weak" in v for v in track_verdicts):
        verdict["overall"] = "partial_signal"
    else:
        verdict["overall"] = "cross_track_null"
        verdict["interpretation"] = (
            "Dynamical structure from Phases 1-2 is real but doesn't organize "
            "the representation at a level accessible to dictionary learning. "
            "Metastability is a property of the bulk geometry that doesn't "
            "decompose into feature-level units."
        )

    return verdict


# ---------------------------------------------------------------------------
# 3. Per-track save helpers
#    Called immediately after each track completes in run_4.py.
# ---------------------------------------------------------------------------

def save_track1_outputs(out_dir: Path, t1_results: dict) -> None:
    """
    Save Track 1 (crosscoder activation patterns) results to out_dir/track1/.

    Files written
    -------------
    track1/
      feature_plateaus.json     — per-prompt plateau detection
      feature_cluster_mi.json   — NMI of features vs cluster labels per layer
      chorus.json               — co-activation clique ARI / purity per layer
      merge_dynamics.json       — feature birth/death at merge events
      plateau_alignment.json    — falsification test result
      track1_summary.json       — flattened numeric summaries for aggregation
    """
    t1_dir = Path(out_dir) / "track1"
    t1_dir.mkdir(parents=True, exist_ok=True)

    _optional_dump(t1_results.get("feature_plateaus"),  t1_dir / "feature_plateaus.json")
    _optional_dump(t1_results.get("mi_results"),        t1_dir / "feature_cluster_mi.json")
    _optional_dump(t1_results.get("chorus_results"),    t1_dir / "chorus.json")
    _optional_dump(t1_results.get("merge_dynamics"),    t1_dir / "merge_dynamics.json")
    _optional_dump(t1_results.get("plateau_alignment"), t1_dir / "plateau_alignment.json")

    summary = {
        "chorus_summary":   t1_results.get("chorus_summary", {}),
        "mi_summary":       t1_results.get("mi_summary", {}),
        "plateau_falsification": (
            t1_results.get("plateau_alignment", {}).get("falsification", "untestable")
        ),
        "n_prompts": t1_results.get("n_prompts", 0),
    }
    _jdump(summary, t1_dir / "track1_summary.json")


def save_track2_outputs(out_dir: Path, t2_results: dict) -> None:
    """
    Save Track 2 (direct geometric methods) results to out_dir/track2/.

    Files written
    -------------
    track2/
      lda_stability.json        — LDA direction cosine trajectory per layer
      probe_accuracy.json       — linear probe accuracy per layer
      delta_pca.json            — PCA on layer-to-layer deltas
      probe_directions.npz      — probe weight vectors (cluster identity directions)
      v_alignment.json          — probe V-subspace alignment (if phase2 available)
      track2_summary.json       — flattened numeric summaries
    """
    t2_dir = Path(out_dir) / "track2"
    t2_dir.mkdir(parents=True, exist_ok=True)

    _optional_dump(t2_results.get("lda_results"),      t2_dir / "lda_stability.json")
    _optional_dump(t2_results.get("probe_results"),    t2_dir / "probe_accuracy.json")
    _optional_dump(t2_results.get("delta_pca_results"),t2_dir / "delta_pca.json")
    _optional_dump(t2_results.get("v_alignment"),      t2_dir / "v_alignment.json")

    # Probe weight vectors as npz (needed by Phase 5)
    probe_dirs = t2_results.get("probe_directions")
    if probe_dirs:
        np.savez(
            t2_dir / "probe_directions.npz",
            **{
                str(k): np.array(v) if not isinstance(v, np.ndarray) else v
                for k, v in probe_dirs.items()
            },
        )

    summary = {
        "probe_summary":    t2_results.get("probe_summary", {}),
        "lda_summary":      t2_results.get("lda_summary", {}),
        "delta_pca_summary":t2_results.get("delta_pca_summary", {}),
    }
    _jdump(summary, t2_dir / "track2_summary.json")


def save_track3_outputs(out_dir: Path, t3_results: dict) -> None:
    """
    Save Track 3 (low-rank AE) results to out_dir/track3/.

    Files written
    -------------
    track3/
      bottleneck_directions.npz — low-rank AE bottleneck vectors
      v_alignment.json          — bottleneck V-subspace alignment
      reconstruction.json       — LRAE vs crosscoder MSE per prompt
      track3_summary.json       — flattened numeric summaries
    """
    if not t3_results:
        return

    t3_dir = Path(out_dir) / "track3"
    t3_dir.mkdir(parents=True, exist_ok=True)

    bd = t3_results.get("bottleneck_directions")
    if bd is not None:
        arr = np.array(bd) if not isinstance(bd, np.ndarray) else bd
        np.savez(t3_dir / "bottleneck_directions.npz", directions=arr)

    _optional_dump(t3_results.get("v_alignment"),   t3_dir / "v_alignment.json")
    _optional_dump(t3_results.get("reconstruction"),t3_dir / "reconstruction.json")

    va = t3_results.get("v_alignment", {})
    recon = t3_results.get("reconstruction", {})
    ratios = [v.get("ratio", 1.0) for v in recon.values() if isinstance(v, dict)]
    summary = {
        "v_alignment_summary": {
            "mean_repulsive":        va.get("mean_repulsive", 0.0),
            "mean_attractive":       va.get("mean_attractive", 0.0),
            "n_repulsive_dominant":  va.get("n_repulsive_dominant", 0),
            "n_attractive_dominant": va.get("n_attractive_dominant", 0),
        },
        "reconstruction_summary": {
            "mean_ratio": float(np.mean(ratios)) if ratios else None,
            "n_prompts":  len(ratios),
        },
    }
    _jdump(summary, t3_dir / "track3_summary.json")


def _optional_dump(obj, path: Path) -> None:
    """Write obj to path as JSON only if obj is non-empty."""
    if obj is not None and obj != {} and obj != []:
        _jdump(obj, path)


# ---------------------------------------------------------------------------
# 4. LLM-friendly summary
# ---------------------------------------------------------------------------

def write_llm_summary(
    out_dir: Path,
    model_name: str,
    t1_results: dict,
    t2_results: dict,
    t3_results: Optional[dict],
    verdict: dict,
    agreement: dict,
) -> None:
    """
    Write summary.txt — a concise, prose-first file readable by an LLM
    or human reviewer without any JSON parsing.

    Structure
    ---------
    HEADER: model, timestamp, output dir
    TRACK 1: key stats + sub-verdict
    TRACK 2: key stats + sub-verdict
    TRACK 3: key stats + sub-verdict (if run)
    CROSS-TRACK: T1/T2 correlation
    VERDICT: per-track + overall + interpretation
    FILES: index of all output files
    """
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def h(title):
        lines.append("")
        lines.append("=" * 60)
        lines.append(title)
        lines.append("=" * 60)

    def kv(key, val, indent=0):
        prefix = "  " * indent
        lines.append(f"{prefix}{key}: {val}")

    # --- Header ---
    h("PHASE 4: IDENTIFYING METASTABLE FEATURES")
    kv("Model",     model_name)
    kv("Timestamp", now)
    kv("Output",    str(out_dir))

    # --- Track 1 ---
    h("TRACK 1: Crosscoder Activation Patterns")
    cs = t1_results.get("chorus_summary", {})
    ms = t1_results.get("mi_summary", {})
    pa = t1_results.get("plateau_alignment", {})

    kv("Feature–cluster MI (max NMI)",   f"{ms.get('max_nmi', 0.0):.4f}")
    kv("Feature–cluster MI (mean NMI)",  f"{ms.get('mean_nmi', 0.0):.4f}")
    kv("Chorus ARI (max)",               f"{cs.get('max_ari', 0.0):.4f}")
    kv("Chorus ARI (mean)",              f"{cs.get('mean_ari', 0.0):.4f}")
    kv("Chorus purity (mean)",           f"{cs.get('mean_purity', 0.0):.4f}")
    kv("Plateau alignment rate",         f"{pa.get('alignment_rate', 'N/A')}")
    kv("Falsification result",           pa.get("falsification", "untestable"))
    kv("Prompts analyzed",               t1_results.get("n_prompts", 0))

    t1v = verdict.get("tracks", {}).get("track1_crosscoder", {})
    kv("Sub-verdict", t1v.get("verdict", "N/A"))

    # Per-prompt plateau summary
    all_plateaus = t1_results.get("feature_plateaus", {})
    if all_plateaus:
        lines.append("")
        lines.append("  Feature plateau summary (per prompt):")
        for pk, fp in all_plateaus.items():
            s = fp.get("summary", {})
            lines.append(
                f"    {pk}: {s.get('n_features_with_plateaus', 0)}/"
                f"{s.get('n_features_total', 0)} features have plateaus "
                f"(mean length {s.get('mean_plateau_length', 0):.1f} layers)"
            )

    # --- Track 2 ---
    h("TRACK 2: Direct Geometric Methods")
    ps = t2_results.get("probe_summary", {})
    ls = t2_results.get("lda_summary", {})
    dp = t2_results.get("delta_pca_summary", {})

    kv("Linear probe accuracy (mean)", f"{ps.get('mean_accuracy', 0.0):.4f}")
    kv("Linear probe accuracy (max)",  f"{ps.get('max_accuracy', 0.0):.4f}")
    kv("LDA cosine stability (mean)",  f"{ls.get('mean_cosine', 0.0):.4f}")
    kv("Update variance (mean)",       f"{dp.get('mean_total_variance', 0.0):.6f}")
    kv("Top-1 PC explained (mean)",    f"{dp.get('mean_top1_explained', 0.0):.4f}")

    va2 = t2_results.get("v_alignment")
    if va2:
        lines.append("")
        lines.append("  Probe V-subspace alignment (per layer):")
        for layer_key, la in (va2.items() if isinstance(va2, dict) else []):
            if isinstance(la, dict) and "mean_repulsive" in la:
                lines.append(
                    f"    {layer_key}: repulsive={la['mean_repulsive']:.4f}  "
                    f"attractive={la['mean_attractive']:.4f}"
                )

    t2v = verdict.get("tracks", {}).get("track2_geometric", {})
    kv("Sub-verdict", t2v.get("verdict", "N/A"))

    # --- Track 3 ---
    if t3_results:
        h("TRACK 3: Low-Rank Autoencoder")
        va3 = t3_results.get("v_alignment", {})
        recon = t3_results.get("reconstruction", {})
        ratios = [v.get("ratio", 1.0) for v in recon.values() if isinstance(v, dict)]

        kv("Bottleneck→V repulsive (mean)",    f"{va3.get('mean_repulsive', 0.0):.4f}")
        kv("Bottleneck→V attractive (mean)",   f"{va3.get('mean_attractive', 0.0):.4f}")
        kv("Repulsive-dominant directions",    va3.get("n_repulsive_dominant", 0))
        kv("Attractive-dominant directions",   va3.get("n_attractive_dominant", 0))
        kv("LRAE/crosscoder MSE ratio (mean)", f"{np.mean(ratios):.4f}" if ratios else "N/A")

        lines.append("")
        lines.append("  Reconstruction comparison (per prompt):")
        for pk, r in recon.items():
            if isinstance(r, dict):
                lines.append(
                    f"    {pk}: LRAE={r['lrae_mse']:.6f}  "
                    f"CC={r['crosscoder_mse']:.6f}  ratio={r['ratio']:.3f}"
                )

        t3v = verdict.get("tracks", {}).get("track3_low_rank_ae", {})
        kv("Sub-verdict", t3v.get("verdict", "N/A"))

    # --- Cross-track ---
    h("CROSS-TRACK COMPARISON")
    t1t2 = agreement.get("t1_t2_correlation", {})
    kv("T1/T2 Spearman ρ",   f"{t1t2.get('spearman_rho', float('nan')):.4f}")
    kv("T1/T2 p-value",      f"{t1t2.get('pval', float('nan')):.4f}")
    kv("T1/T2 n_layers",     t1t2.get("n_layers", 0))
    kv("T1/T2 interpretation", t1t2.get("interpretation", "N/A"))
    if "t3_mean_recon_ratio" in agreement and agreement["t3_mean_recon_ratio"] is not None:
        kv("T3 LRAE/CC ratio",   f"{agreement['t3_mean_recon_ratio']:.4f}")

    # --- Verdict ---
    h("VERDICT")
    for track_name, tv in verdict.get("tracks", {}).items():
        kv(track_name, tv.get("verdict", "N/A"))
    lines.append("")
    kv("Overall", verdict.get("overall", "N/A"))
    if "interpretation" in verdict:
        lines.append("")
        lines.append("Interpretation:")
        lines.append(f"  {verdict['interpretation']}")

    # --- File index ---
    h("OUTPUT FILES")
    out_dir = Path(out_dir)
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            lines.append(f"  {p.relative_to(out_dir)}  ({size_kb:.1f} KB)")

    # --- Write ---
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Summary written → {summary_path}")


# ---------------------------------------------------------------------------
# 5. Backward-compatible combined save (thin wrapper)
# ---------------------------------------------------------------------------

def save_phase4_outputs(
    out_dir: Path,
    track1_results: dict,
    track2_results: dict,
    track3_results: Optional[dict],
    verdict: dict,
    model_name: str = "unknown",
    activations_per_layer: Optional[dict] = None,
    hdbscan_labels: Optional[dict] = None,
) -> None:
    """
    Save all Phase 4 outputs.  Prefer calling the per-track helpers directly
    from run_4.py so files land on disk immediately; this wrapper exists for
    scripts that call save_phase4_outputs at the end.

    Extras beyond the per-track helpers
    ------------------------------------
    verdict.json              — structured cross-track verdict
    centroids.npz             — cluster centroids per (prompt, layer) for Phase 6
    hdbscan_labels.json       — Phase 1 labels forwarded for Phase 5
    summary.txt               — LLM-friendly plain-text summary
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-track (idempotent — no-ops if already called inline)
    save_track1_outputs(out_dir, track1_results)
    save_track2_outputs(out_dir, track2_results)
    save_track3_outputs(out_dir, track3_results)

    # Verdict
    _jdump(verdict, out_dir / "verdict.json")

    # Cluster centroids for Phase 6 (per (prompt, layer, cluster))
    if activations_per_layer:
        centroid_dict = {}
        for pk, layers in activations_per_layer.items():
            pk_labels = (hdbscan_labels or {}).get(pk, {})
            for layer_key, acts in (layers.items() if isinstance(layers, dict) else []):
                labels = pk_labels.get(layer_key)
                if labels is None:
                    continue
                labels_arr = np.array(labels)
                acts_arr   = np.array(acts) if not isinstance(acts, np.ndarray) else acts
                for c in set(labels_arr[labels_arr >= 0].tolist()):
                    mask = labels_arr == c
                    centroid = acts_arr[mask].mean(axis=0)
                    key = f"{pk}__{layer_key}__c{int(c)}"
                    centroid_dict[key] = centroid
        if centroid_dict:
            np.savez(out_dir / "centroids.npz", **centroid_dict)

    if hdbscan_labels:
        _jdump(hdbscan_labels, out_dir / "hdbscan_labels.json")

    # Build agreement for summary
    agreement = verdict.get("agreement") or cross_track_agreement(
        track1_results, track2_results, track3_results
    )

    write_llm_summary(
        out_dir, model_name,
        track1_results, track2_results, track3_results,
        verdict, agreement,
    )
