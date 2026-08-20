"""
p1d_cluster_ensemble/p1d_io.py — artifact loading and persistence for
Phase 1d.

Phase 1d reads Phase 1's outputs and writes its own. The same two rules
p1c_io.py carries apply, for the same reasons:

1. IF A QUANTITY APPEARS IN A REPORT, IT IS PERSISTED. This phase reports
   per-particle confidences, per-family gate decisions, and the whole
   hyperparameter surface — none of which can be recomputed cheaply (the
   sweep is the expensive part) and all of which are written.

2. EVERY DATA-DEPENDENT FALLBACK RECORDS THE BRANCH IT TOOK. Which
   HDBSCAN backend was used, whether the shipped labels were found,
   whether scipy was available for the consensus cut, which families
   abstained at which layer.

What this phase refuses to run without
--------------------------------------
`activations.npz`. Phase 1d re-clusters; it cannot re-analyse persisted
label arrays into a tuned partition. A run directory without activations
raises rather than silently degrading to "compare the four partitions
already on disk", which is the analysis Phase 1's visualization package
already does and which this phase exists because of.

`hdbscan_labels.json` is not required but its absence is consequential
and recorded: without the shipped partition, P-C2 and P-C3 have no
reference and are skipped, and the driver says so per run rather than
reporting a comparison against a re-run stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.particles import ParticleTable

from .constants import KMEANS_RANK_MIN, KMEANS_SIL_MIN

#: Written into every artifact. Bump when a change makes an older
#: p1d_results.json non-comparable with a newer one, so that a
#: cross-checkpoint aggregation cannot silently mix two definitions.
P1D_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> Dict:
    """
    Everything Phase 1d needs from one Phase 1 run directory.

    Returns a dict with keys: run_dir, geometry, activations
    (n_layers, n_tokens, d — already L2-normed, as Phase 1 saves them),
    norms (or None), shipped_hdbscan ({layer: (n,) int array}), tokens,
    available (set), provenance.

    Raises FileNotFoundError when activations.npz is missing or unreadable
    — see the module docstring.
    """
    run_dir = Path(run_dir)
    out: Dict[str, object] = {"run_dir": str(run_dir), "available": set()}

    geo_p = run_dir / "geometry.json"
    geometry = None
    if geo_p.exists():
        with open(geo_p) as f:
            geometry = json.load(f)
        out["available"].add("geometry")
    out["geometry"] = geometry

    act_p = run_dir / "activations.npz"
    if not act_p.exists():
        raise FileNotFoundError(
            f"{act_p} not found. Phase 1d re-clusters the activations; it "
            "cannot tune a method against persisted label arrays. A run "
            "without activations cannot contribute to this phase."
        )
    data = np.load(act_p)
    key = "activations" if "activations" in data.files else data.files[0]
    out["activations"] = data[key]
    out["norms"] = data["norms"] if "norms" in data.files else None
    out["available"].add("activations")
    if out["norms"] is not None:
        out["available"].add("norms")

    hdb_p = run_dir / "hdbscan_labels.json"
    shipped: Dict[int, np.ndarray] = {}
    if hdb_p.exists():
        with open(hdb_p) as f:
            raw = json.load(f)
        shipped = {int(k): np.asarray(v, dtype=np.int32) for k, v in raw.items()}
        out["available"].add("shipped_hdbscan")
    out["shipped_hdbscan"] = shipped

    tokens: List[str] = []
    if geometry and geometry.get("tokens"):
        tokens = [str(t) for t in geometry["tokens"]]
        out["available"].add("tokens")
    out["tokens"] = tokens

    out["provenance"] = {
        p.name: {"exists": p.exists(),
                 "mtime": (p.stat().st_mtime if p.exists() else None),
                 "bytes": (p.stat().st_size if p.exists() else None)}
        for p in (geo_p, act_p, hdb_p)
    }
    return out


def run_identity(run: Dict) -> Dict:
    """
    The (model, checkpoint, prompt) triple that names a run, straight out
    of geometry.json's provenance block.

    `checkpoint_step` is left as None when the artifact does not record
    one, and ParticleTable's own sentinel handles that — a checkpoint step
    invented here would be indistinguishable downstream from a real one.
    """
    geo = run.get("geometry") or {}
    return {
        "model": geo.get("model", Path(run["run_dir"]).name),
        # geometry.json's "prompt" is whatever Phase 1 was given: a key
        # into core.config.PROMPTS on current runs, free text on older
        # ones. It is passed through unchanged rather than normalized,
        # because a normalization here would not match the one Phase 1's
        # own aggregations use.
        "prompt_key": geo.get("prompt", ""),
        "checkpoint_step": geo.get("checkpoint_step"),
        "revision": geo.get("revision"),
        "n_tokens": geo.get("n_tokens"),
        "n_layers": geo.get("n_layers"),
        "random_init": geo.get("random_init"),
    }


def layer_activations(run: Dict, layer: int) -> np.ndarray:
    """(n_tokens, d) L2-normed activations for one layer."""
    acts = np.asarray(run["activations"])
    if layer < 0 or layer >= acts.shape[0]:
        raise IndexError(f"layer {layer} outside 0..{acts.shape[0] - 1}")
    return acts[layer]


# ---------------------------------------------------------------------------
# Phase 1's own agreement layers — P-C1's scope
# ---------------------------------------------------------------------------

def phase1_agreement_layers(run_dir: Path) -> Dict:
    """
    The layers at which Phase 1 already reports the methods agreeing:
    every trusted cluster-count estimate within +/-1 of every other.

    P-C1 is registered about exactly these layers ("at layers where Phase
    1 already reports method agreement"), so adjudicating it over all
    layers instead would be answering a different, weaker question. That
    is why this reader exists rather than the driver passing None.

    The criterion is `cluster_count_table`'s in
    p1_visualization/cluster_methods.py, down to the KMeans trust gate,
    whose two constants are read out of that module by
    p1d_cluster_ensemble.constants rather than copied. The two
    implementations are asserted equivalent in
    tests/test_phase1d_ensemble.py wherever the visualization package can
    be imported; this one exists because that package's __init__ pulls in
    the whole figure pipeline and this phase must stay importable without
    it.

    Returns {layers: [int], per_layer: {layer: {counts, trusted, agree}},
    available: bool, reason: str}. `available=False` with a reason is the
    honest outcome for a run that never wrote clustering.json — P-C1 is
    then adjudicated over all layers and says so.
    """
    run_dir = Path(run_dir)
    clustering = _read_json(run_dir / "clustering.json")
    if not clustering.get("layers"):
        return {"layers": [], "per_layer": {}, "available": False,
                "reason": "clustering.json absent or empty — Phase 1's own "
                          "agreement layers cannot be reconstructed"}

    geometry = _read_json(run_dir / "geometry.json")
    spectral = {int(lr["layer"]): lr for lr in _read_json(run_dir / "spectral.json").get("layers", [])}
    sinkhorn = {int(lr["layer"]): lr for lr in _read_json(run_dir / "sinkhorn.json").get("layers", [])}
    ranks = {int(lr["layer"]): lr.get("effective_rank")
             for lr in geometry.get("layers", [])}

    thresholds = _agglom_thresholds(clustering)
    mid = thresholds[len(thresholds) // 2] if thresholds else None

    per_layer: Dict[int, Dict] = {}
    agreement: List[int] = []
    for lr in clustering["layers"]:
        li = int(lr["layer"])
        cl = lr.get("clustering", {})
        counts: Dict[str, float] = {}

        hdb_k = cl.get("hdbscan", {}).get("n_clusters")
        if hdb_k is not None:
            counts["hdbscan"] = float(hdb_k)

        agg = cl.get("agglomerative", {})
        if mid is not None:
            for key, val in agg.items():
                if key == "mid_labels":
                    continue
                try:
                    if np.isclose(float(key), mid):
                        counts["agglomerative"] = float(val)
                        break
                except (TypeError, ValueError):
                    continue

        sp = spectral.get(li, {}).get("k_eigengap")
        if sp is not None:
            counts["spectral_k"] = float(sp)
        sk = sinkhorn.get(li, {}).get("sinkhorn_cluster_count_mean")
        if sk is not None:
            counts["sinkhorn_k"] = float(round(sk))

        sil = cl.get("kmeans", {}).get("best_silhouette")
        rank = ranks.get(li)
        trusted = bool(sil is not None and rank is not None
                       and sil >= KMEANS_SIL_MIN and rank >= KMEANS_RANK_MIN)
        km_k = cl.get("kmeans", {}).get("best_k")
        if trusted and km_k is not None:
            counts["kmeans"] = float(km_k)

        vals = [v for v in counts.values() if np.isfinite(v) and v > 0]
        agree = bool(vals) and (max(vals) - min(vals)) <= 1
        per_layer[li] = {"counts": counts, "kmeans_trusted": trusted, "agree": agree}
        if agree:
            agreement.append(li)

    return {"layers": agreement, "per_layer": per_layer, "available": True,
            "reason": "", "mid_threshold": mid}


def _agglom_thresholds(clustering: Dict) -> List[float]:
    """
    Ascending agglomerative sweep thresholds present in clustering.json.
    JSON stringifies the float keys, so they are parsed rather than looked
    up — "0.30000000000000004" does not compare equal to 0.3 as a string.
    """
    found = set()
    for lr in clustering.get("layers", []):
        for key in lr.get("clustering", {}).get("agglomerative", {}):
            if key == "mid_labels":
                continue
            try:
                found.add(float(key))
            except (TypeError, ValueError):
                continue
    return sorted(found)


def _read_json(path: Path) -> Dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_p1d(
    out_dir: Path,
    results: Dict,
    per_layer_arrays: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    save_matrices: bool = True,
) -> Dict[str, str]:
    """
    Write this phase's three artifacts and return {name: path}.

    p1d_results.json   every scalar, verdict, gate decision and
                        hyperparameter surface. The readable record.
    p1d_ensemble.npz    per-layer arrays: co-association matrices,
                        consensus labels, confidences, recalls, refusal
                        fractions. Matrices are stored float32 — they are
                        the bulk of the file (n_tokens^2 per layer) and
                        no downstream reading of a co-association value
                        needs more than 7 digits.
    particle_table.npz  the ParticleTable export, written by
                        core.particles so its schema cannot drift from
                        the canonical one.

    `save_matrices=False` drops the co-association matrices only. Every
    per-particle array is still written, because those are what a
    downstream phase reads; the matrices are what a figure reads.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    payload = _sanitize({"schema_version": P1D_SCHEMA_VERSION, **results})
    res_p = out_dir / "p1d_results.json"
    with open(res_p, "w") as f:
        # allow_nan=False so anything _sanitize missed raises here rather
        # than writing a NaN literal, which is not JSON and which this
        # phase would otherwise emit for some values and `null` for others
        # depending only on whether the number happened to be a numpy
        # scalar or a Python float.
        json.dump(payload, f, indent=2, allow_nan=False)
    written["p1d_results"] = str(res_p)

    if per_layer_arrays:
        arrays: Dict[str, np.ndarray] = {}
        for layer, block in per_layer_arrays.items():
            for name, arr in block.items():
                if name == "co_association" and not save_matrices:
                    continue
                arr = np.asarray(arr)
                if name == "co_association":
                    arr = arr.astype(np.float32)
                if arr.dtype == object:
                    arr = arr.astype(str)
                arrays[f"{name}_L{int(layer)}"] = arr
        ens_p = out_dir / "p1d_ensemble.npz"
        np.savez_compressed(ens_p, **arrays)
        written["p1d_ensemble"] = str(ens_p)

    return written


def build_particle_table(
    identity: Dict,
    per_layer: Dict[int, Dict[str, np.ndarray]],
    tokens: Optional[List[str]] = None,
) -> ParticleTable:
    """
    The per-particle export: one row per (model, checkpoint, prompt,
    layer, token), carrying the consensus label as `cluster_label` and the
    trichotomy tag as `population`.

    This is the point of the phase for everything downstream. Phase 5c's
    object of study is the unclustered population; with this table its
    selector stops being `cluster_label < 0` and becomes
    `population="contested"` — or a threshold on the `confidence` extra
    column, which is a continuous quantity the categorical label cannot
    express at all.

    `cluster_label` is the CONSENSUS label, not HDBSCAN's. HDBSCAN's is
    carried alongside as `extra__hdbscan_label` so nothing is lost and the
    two are never confused: a consumer that wants the shipped partition
    must ask for it by name.
    """
    tables = []
    for layer in sorted(per_layer):
        block = per_layer[layer]
        labels = np.asarray(block["consensus_labels"], dtype=np.int64)
        n = labels.size
        extra = {
            name: np.asarray(block[name])
            for name in ("confidence", "mean_recall", "min_recall",
                         "refusal_fraction", "hdbscan_label", "n_families")
            if name in block
        }
        tables.append(ParticleTable.from_layer(
            model=str(identity.get("model", "")),
            prompt_key=str(identity.get("prompt_key", "")),
            layer=int(layer),
            cluster_labels=labels,
            checkpoint_step=identity.get("checkpoint_step"),
            population=[str(p) for p in block["population"]],
            token_str=(tokens[:n] if tokens and len(tokens) >= n else None),
            extra=extra,
        ))
    return ParticleTable.concat(tables)


def _sanitize(obj):
    """
    Recursively convert a results dict into strictly-JSON-able values.

    Non-finite floats become `null`, everywhere and regardless of dtype.
    NaN in this phase means "not measurable" — a separation with fewer
    than two clusters, a stability with no usable repeat — and that is
    exactly what a JSON null says. The alternative, Python's non-standard
    `NaN` literal, is what the rest of this project writes; the departure
    is deliberate and local to Phase 1d, because half these values arrive
    as numpy scalars and half as Python floats, and a writer that spells
    the same missing value two ways depending on incidental type is worse
    than either convention.
    """
    if isinstance(obj, dict):
        return {(_sanitize_key(k)): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_sanitize(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _sanitize_key(key):
    """numpy-typed and non-string dict keys into what JSON permits."""
    if isinstance(key, (np.integer,)):
        return str(int(key))
    if isinstance(key, (np.floating,)):
        return str(float(key))
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    return str(key)
