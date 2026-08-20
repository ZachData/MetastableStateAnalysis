"""
p1d_cluster_ensemble/run_1d.py — Phase 1d driver.

    python -m p1d_cluster_ensemble.run_1d --results results/phase1 \
        --out results_p1d/ --subexp A B C D E

Sub-experiments
---------------
    A  tune every family at every layer            selection.py
    B  build the ensemble and calibrate confidence ensemble.py
    C  the shipped partition and its refusals      comparison.py  (P-C2, P-C3)
    D  persistence prediction                      comparison.py  (P-C4)
    E  export the particle table                   p1d_io.py

B needs A, C and D need B, E needs B. Prerequisites are pulled in
automatically and the expansion is recorded in the artifact, so a result
never silently depends on a step the command line did not name.

WHAT THIS DELIBERATELY DOES NOT DO

It does not adjudicate P-C1..P-C4 inside the loop that computes their
inputs. The adjudicators are separate functions in comparison.py, called
once at the end over already-computed per-layer results, so the compute
step can be rerun without re-deciding anything — the same separation
run_1c.py keeps, and for the same reason: a verdict must not be
reachable from a code path that also chose its inputs.

Cost
----
The sweep is the expensive part and it is quadratic in the wrong places:
n_grid settings x (1 + 2*n_repeats) fits per family per layer, plus
top_m x n_null x (1 + 2*n_null_repeats) more for the gates. On a 24-layer
run with a few hundred tokens the full grid is hours, not minutes. The
knobs that actually move it, in order: --layer-stride, --grid quick,
--n-null, --top-m. Every one of them is written into the artifact,
because a selection made under --grid quick is a different claim from one
made over the full grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import comparison, ensemble
from .constants import SHIPPED_HDBSCAN_PARAMS
from .methods import LayerData, available_families, fit, hdbscan_backend
from .p1d_io import (
    build_particle_table, layer_activations, load_run, phase1_agreement_layers,
    run_identity, save_p1d,
)
from .selection import (
    NULL_ALPHA, select_all_families, selected_labels, selection_weights,
)

SUBEXPERIMENTS = ("A", "B", "C", "D", "E")
_REQUIRES = {"A": set(), "B": {"A"}, "C": {"A", "B"}, "D": {"A", "B"}, "E": {"A", "B"}}


def expand_subexperiments(requested: Sequence[str]) -> List[str]:
    """Requested sub-experiments plus everything they need, in order."""
    wanted = set(requested)
    for name in list(wanted):
        wanted |= _REQUIRES.get(name, set())
    return [s for s in SUBEXPERIMENTS if s in wanted]


# ---------------------------------------------------------------------------
# Per-layer work
# ---------------------------------------------------------------------------

def null_confidences(
    data: LayerData,
    selection: Dict[str, Dict],
    n_draws: int = 10,
    noise_policy: str = "exclude",
    seed: int = 0,
) -> List[np.ndarray]:
    """
    Per-particle confidences from the whole ensemble re-run on
    shuffled-dimension null draws, using each family's already-selected
    setting.

    This is what puts a scale on a confidence number. It re-runs the
    ensemble rather than the selection: the question is "what agreement
    do these methods, at these settings, manufacture out of structureless
    data with these marginals", not "what would they have selected there"
    — the second is the gate's question and was already asked.
    """
    families = [f for f, res in selection.items() if res.get("selected") is not None]
    if not families:
        return []
    params = {f: selection[f]["selected"]["params"] for f in families}
    weights = selection_weights(selection)

    rng = np.random.default_rng(seed + 5081)
    X = np.asarray(data.normed, dtype=np.float64)
    n_tokens, d = X.shape
    out: List[np.ndarray] = []
    for _ in range(max(0, int(n_draws))):
        shuffled = np.empty_like(X)
        for col in range(d):
            shuffled[:, col] = X[rng.permutation(n_tokens), col]
        norms = np.linalg.norm(shuffled, axis=1, keepdims=True)
        null_data = LayerData.from_normed(shuffled / np.maximum(norms, 1e-12))
        labels = {f: fit(f, params[f], null_data, seed=seed) for f in families}
        built = ensemble.build(labels, weights=weights, noise_policy=noise_policy)
        out.append(np.asarray(built["confidence"], dtype=float))
    return out


def process_layer(
    data: LayerData,
    families: Sequence[str],
    shipped: Optional[np.ndarray],
    args: argparse.Namespace,
    stages: Sequence[str],
) -> Dict:
    """One layer: tune (A), ensemble (B), shipped comparison + rescue (C)."""
    result: Dict[str, object] = {"n_tokens": data.n}

    selection = select_all_families(
        data, list(families), grid=args.grid, n_repeats=args.n_repeats,
        n_null=args.n_null, n_null_repeats=args.n_null_repeats,
        top_m=args.top_m, alpha=args.alpha, seed=args.seed,
    )
    result["selection"] = {
        f: {k: v for k, v in res.items()
            if k != "candidates" or args.save_surface}
        for f, res in selection.items()
    }
    result["abstained"] = sorted(f for f, res in selection.items()
                                 if res.get("selected") is None)

    if "B" not in stages:
        return result

    labels_by_family = selected_labels(selection)
    weights = selection_weights(selection)
    if not labels_by_family:
        result["ensemble"] = None
        result["ensemble_skipped"] = "every family abstained at this layer"
        return result

    built = ensemble.build(labels_by_family, weights=weights,
                           noise_policy=args.noise_policy)
    nulls = null_confidences(data, selection, n_draws=args.n_null_confidence,
                             noise_policy=args.noise_policy, seed=args.seed)
    thresholds = ensemble.confidence_thresholds(nulls)
    population = ensemble.trichotomy(built["confidence"], thresholds)

    result["ensemble"] = {
        "n_families": built["n_families"],
        "families": built["families"],
        "weights": built["weights"],
        "consensus_strength": built["consensus_strength"],
        "n_clusters": built["consensus"]["n_clusters"],
        "mirkin_objective": built["consensus"]["objective"],
        "cut_height": built["consensus"]["cut_height"],
        "consensus_branch": built["consensus"]["branch"],
        "thresholds": thresholds,
        "population_counts": {
            tag: int((population == tag).sum())
            for tag in ("core", "halo", "contested", "uncalibrated")
        },
        "mean_confidence": float(np.nanmean(built["confidence"]))
        if built["confidence"].size else float("nan"),
    }
    result["_arrays"] = {
        "co_association": built["co_association"]["C"],
        "consensus_labels": built["consensus"]["labels"],
        "confidence": built["confidence"],
        "mean_recall": built["mean_recall"],
        "min_recall": built["min_recall"],
        "refusal_fraction": built["refusal_fraction"],
        "population": population,
        "n_families": np.full(data.n, built["n_families"], dtype=np.int32),
    }

    if "C" in stages and shipped is not None and shipped.size == data.n:
        result["_arrays"]["hdbscan_label"] = np.asarray(shipped, dtype=np.int64)
        tuned = selection.get("hdbscan", {}).get("selected")
        if tuned is not None:
            result["shipped_comparison"] = comparison.shipped_comparison(
                np.asarray(selection["hdbscan"]["selected_labels"]),
                shipped, tuned["params"],
            )
        else:
            result["shipped_comparison"] = {
                "skipped": "hdbscan abstained at this layer; nothing to compare"
            }
        result["noise_rescue"] = comparison.noise_rescue(
            shipped, built["confidence"], built["consensus"]["labels"], thresholds,
        )
    elif "C" in stages:
        result["shipped_comparison"] = {"skipped": "no shipped labels for this layer"}
        result["noise_rescue"] = {"skipped": "no shipped labels for this layer"}

    return result


# ---------------------------------------------------------------------------
# Per-run
# ---------------------------------------------------------------------------

def run_one(run_dir: Path, args: argparse.Namespace) -> Dict:
    """Tune, ensemble, compare and export one Phase 1 run directory."""
    stages = expand_subexperiments(args.subexp)
    run = load_run(run_dir)
    identity = run_identity(run)
    acts = np.asarray(run["activations"])
    n_layers = int(acts.shape[0])

    layers = (sorted(int(l) for l in args.layers) if args.layers
              else list(range(0, n_layers, max(1, args.layer_stride))))
    layers = [l for l in layers if 0 <= l < n_layers]

    families = [f for f in (args.families or available_families())
                if f in available_families()]

    out: Dict[str, object] = {
        "run_dir": str(run_dir),
        "identity": identity,
        "stages": stages,
        "requested_subexp": list(args.subexp),
        "layers": layers,
        "families": families,
        "hdbscan_backend": hdbscan_backend(),
        "shipped_hdbscan_params": dict(SHIPPED_HDBSCAN_PARAMS),
        "available": sorted(run["available"]),
        "provenance": run["provenance"],
        "settings": {
            "grid": args.grid, "n_repeats": args.n_repeats,
            "n_null": args.n_null, "n_null_repeats": args.n_null_repeats,
            "n_null_confidence": args.n_null_confidence, "top_m": args.top_m,
            "alpha": args.alpha, "noise_policy": args.noise_policy,
            "seed": args.seed, "layer_stride": args.layer_stride,
            "save_surface": args.save_surface,
        },
        "per_layer": {},
        "skipped": {},
    }
    if not run["shipped_hdbscan"]:
        out["skipped"]["shipped_hdbscan"] = (
            "hdbscan_labels.json absent — P-C2 and P-C3 have no reference "
            "partition and are not adjudicated for this run"
        )

    per_layer_arrays: Dict[int, Dict[str, np.ndarray]] = {}
    for layer in layers:
        data = LayerData.from_normed(layer_activations(run, layer))
        shipped = run["shipped_hdbscan"].get(layer)
        res = process_layer(data, families, shipped, args, stages)
        arrays = res.pop("_arrays", None)
        if arrays is not None:
            per_layer_arrays[layer] = arrays
        out["per_layer"][str(layer)] = res
        if args.verbose:
            _print_layer(layer, res)

    out["phase1_agreement"] = phase1_agreement_layers(run_dir)
    out["verdicts"] = _adjudicate(out, per_layer_arrays, stages, args)
    return {"results": out, "arrays": per_layer_arrays,
            "identity": identity, "tokens": run["tokens"]}


def _adjudicate(out: Dict, arrays: Dict[int, Dict[str, np.ndarray]],
                stages: Sequence[str], args: argparse.Namespace) -> Dict:
    """
    Every registered prediction, decided once, from already-computed
    per-layer results. Nothing here recomputes an input.
    """
    verdicts: Dict[str, object] = {}
    per_layer = out["per_layer"]

    if "B" in stages:
        strength = {
            int(l): float(res["ensemble"]["consensus_strength"])
            for l, res in per_layer.items()
            if res.get("ensemble") and np.isfinite(res["ensemble"]["consensus_strength"])
        }
        agreement = out.get("phase1_agreement", {})
        # P-C1 is registered about the layers Phase 1 itself calls
        # agreeing. When clustering.json is absent that set cannot be
        # reconstructed, and the fallback to every layer is recorded in
        # the verdict string rather than left to be inferred.
        scope = agreement.get("layers") if agreement.get("available") else None
        verdicts["P-C1"] = comparison.adjudicate_p_c1(strength, agreement_layers=scope)

    if "C" in stages:
        shipped_cmp = {
            int(l): res["shipped_comparison"] for l, res in per_layer.items()
            if isinstance(res.get("shipped_comparison"), dict)
            and "skipped" not in res["shipped_comparison"]
        }
        verdicts["P-C2"] = (comparison.adjudicate_p_c2(shipped_cmp) if shipped_cmp
                            else {"verdict": "UNDECIDED — no layer had both a tuned "
                                             "HDBSCAN setting and shipped labels"})
        rescue = {
            int(l): res["noise_rescue"] for l, res in per_layer.items()
            if isinstance(res.get("noise_rescue"), dict)
            and "skipped" not in res["noise_rescue"]
        }
        verdicts["P-C3"] = (comparison.adjudicate_p_c3(rescue) if rescue
                            else {"verdict": "UNDECIDED — no layer had refused particles "
                                             "and a calibrated threshold"})

    if "D" in stages:
        per_boundary: Dict[int, Dict] = {}
        ordered = sorted(arrays)
        for here, nxt in zip(ordered, ordered[1:]):
            a, b = arrays[here], arrays[nxt]
            if a["consensus_labels"].size != b["consensus_labels"].size:
                continue
            target = comparison.persistence_target(
                a["consensus_labels"], b["consensus_labels"])
            binary = (np.asarray(a["hdbscan_label"]) >= 0).astype(float) \
                if "hdbscan_label" in a else None
            if binary is None:
                continue
            per_boundary[here] = comparison.delta_auc_report(
                a["confidence"], binary, target,
                n_permutations=args.n_permutations,
                n_bootstrap=args.n_bootstrap, seed=args.seed,
            )
        out["per_boundary"] = {str(k): v for k, v in per_boundary.items()}
        verdicts["P-C4"] = (comparison.adjudicate_p_c4(per_boundary) if per_boundary
                            else {"verdict": "UNDECIDED — no adjacent layer pair had "
                                             "both an ensemble and shipped labels"})
    return verdicts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_layer(layer: int, res: Dict) -> None:
    ens = res.get("ensemble")
    if not ens:
        print(f"  L{layer:<3d} every family abstained")
        return
    print(f"  L{layer:<3d} n={res['n_tokens']:<4d} "
          f"families={ens['n_families']} "
          f"k={ens['n_clusters']:<3d} "
          f"strength={ens['consensus_strength']:.2f} "
          f"core/halo/contested="
          f"{ens['population_counts']['core']}/"
          f"{ens['population_counts']['halo']}/"
          f"{ens['population_counts']['contested']}"
          + (f"  abstained: {','.join(res['abstained'])}" if res["abstained"] else ""))


def summary_text(results: Dict) -> str:
    """A short human-readable record: settings, coverage, and verdicts."""
    lines: List[str] = []
    ident = results["identity"]
    lines.append("Phase 1d — cluster-method ensemble")
    lines.append(f"run          : {results['run_dir']}")
    lines.append(f"model        : {ident['model']}  prompt: {ident['prompt_key']}  "
                 f"checkpoint: {ident['checkpoint_step']}")
    lines.append(f"backend      : {results['hdbscan_backend']}")
    lines.append(f"families     : {', '.join(results['families'])}")
    lines.append(f"layers       : {results['layers']}")
    lines.append(f"settings     : {results['settings']}")
    if results.get("skipped"):
        for key, why in results["skipped"].items():
            lines.append(f"skipped      : {key} — {why}")

    lines.append("")
    lines.append("Per-layer abstentions")
    for layer, res in sorted(results["per_layer"].items(), key=lambda kv: int(kv[0])):
        if res.get("abstained"):
            lines.append(f"  L{layer}: {', '.join(res['abstained'])}")
    agreement = results.get("phase1_agreement", {})
    if agreement:
        lines.append("")
        lines.append(
            f"Phase 1 agreement layers: {agreement.get('layers')}"
            if agreement.get("available")
            else f"Phase 1 agreement layers: unavailable — {agreement.get('reason')}"
        )
    lines.append("")
    lines.append("Verdicts")
    for name, verdict in results.get("verdicts", {}).items():
        lines.append(f"  {name}: {verdict.get('verdict', '(none)')}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1d — cluster-method ensemble")
    p.add_argument("--results", type=Path, required=True,
                   help="Phase 1 results directory (scanned for run subdirectories) "
                        "or a single run directory")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--subexp", nargs="+", default=list(SUBEXPERIMENTS),
                   choices=list(SUBEXPERIMENTS))
    p.add_argument("--families", nargs="+", default=None,
                   help="subset of method families (default: every available one)")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="explicit layer indices (overrides --layer-stride)")
    p.add_argument("--layer-stride", type=int, default=1)
    p.add_argument("--grid", choices=("full", "quick"), default="full")
    p.add_argument("--n-repeats", type=int, default=5,
                   help="subsample repeats per stability estimate")
    p.add_argument("--n-null", type=int, default=20,
                   help="matched-null draws per gated candidate; must satisfy "
                        "1/(n_null+1) <= alpha or the gate cannot pass")
    p.add_argument("--n-null-repeats", type=int, default=3,
                   help="subsample repeats per null draw")
    p.add_argument("--n-null-confidence", type=int, default=10,
                   help="null draws used to calibrate the confidence thresholds")
    p.add_argument("--top-m", type=int, default=3,
                   help="candidates per family the null gate is computed for")
    p.add_argument("--alpha", type=float, default=NULL_ALPHA)
    p.add_argument("--noise-policy", choices=("exclude", "singleton"), default="exclude")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-surface", action="store_true",
                   help="write every grid point's numbers, not just the gated ones "
                        "(large: n_grid entries per family per layer)")
    p.add_argument("--no-save-matrices", action="store_true",
                   help="omit the co-association matrices from p1d_ensemble.npz")
    p.add_argument("--verbose", action="store_true")
    return p


def discover_runs(results_dir: Path) -> List[Path]:
    """
    Run directories under `results_dir`, or the directory itself when it
    is one. A "run directory" is one holding activations.npz — the file
    this phase cannot work without — so a directory that would raise on
    load is never returned as a candidate.
    """
    results_dir = Path(results_dir)
    if (results_dir / "activations.npz").exists():
        return [results_dir]
    return sorted(d for d in results_dir.iterdir()
                  if d.is_dir() and (d / "activations.npz").exists())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runs = discover_runs(args.results)
    if not runs:
        print(f"no run directory under {args.results} contains activations.npz; "
              f"Phase 1d has nothing to re-cluster", file=sys.stderr)
        return 1

    for run_dir in runs:
        print(f"[p1d] {run_dir}")
        bundle = run_one(run_dir, args)
        out_dir = Path(args.out) / run_dir.name
        written = save_p1d(out_dir, bundle["results"], bundle["arrays"],
                           save_matrices=not args.no_save_matrices)

        if "E" in bundle["results"]["stages"] and bundle["arrays"]:
            table = build_particle_table(bundle["identity"], bundle["arrays"],
                                         tokens=bundle["tokens"])
            table_path = out_dir / "particle_table.npz"
            table.save(table_path)
            written["particle_table"] = str(table_path)

        summary = summary_text(bundle["results"])
        (out_dir / "p1d_summary.txt").write_text(summary)
        print(summary)
        for name, path in written.items():
            print(f"  wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
