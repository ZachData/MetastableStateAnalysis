"""
p1d_cluster_ensemble/vote_rules.py — how should 1d's families vote?

The weighting decision `status-1d.md` "A0" left open: the ensemble's
core / halo / contested grading is not a usable readout until two things
are decided (`/challenge-pr` on #100, finding 1).

- **What a token outside a family's substantial structure does on a pair.**
  Now an HDBSCAN ``-1`` abstains, while another family's singleton (or any
  cluster below ``SUBSTANTIAL_CLUSTER_SIZE``) votes "apart" on every pair it
  touches. ``refusal_fraction`` already calls those the same refusal ("the
  same refusal in a family that cannot spell it"); ``co_association`` does
  not.
- **How much each family's vote weighs.** Now it is raw subsample
  stability, highest for fine partitions and for k = 2 (the scale problem,
  `lit-1d.md` §1).

This re-tunes each family once per layer (run_1d's stage A, unchanged),
fits every selected setting once on each shuffled-dimension null draw
(run_1d's ``null_confidences`` draws, same seed), and then builds the
ensemble under every rule below from those same labels, thresholds
recomputed under that rule. Nothing about selection changes.

Rules (``NOISE_RULES`` x ``WEIGHT_RULES``):

- noise ``current``: run_1d's ``exclude`` (``-1`` abstains, small clusters vote).
- noise ``abstain_small``: every token in a cluster below
  ``SUBSTANTIAL_CLUSTER_SIZE`` becomes ``-1`` first, then ``exclude``: the
  reading ``refusal_fraction`` already uses.
- noise ``singleton``: run_1d's ``singleton`` (``-1`` becomes its own
  cluster and votes "apart").
- weights ``stability``: ``selection_weights``, raw subsample ARI.
- weights ``uniform``: 1 per admitted family; admission is the gate's job.
- weights ``kappa``: stability above its own null,
  ``(s - s_null) / (1 - s_null)`` clipped at 0 (Cohen's form), with
  ``s_null`` the selected setting's stability null mean.

**Reading, fixed before any record was looked at** (2026-09-30):

1. Primary: **single-family sway.** For each rule and record, drop one
   family at a time (nulls rebuilt without it too) and take the worst case
   over families: the Jaccard of the core set with and without it, and the
   ARI of the consensus partition. The design says what an ensemble buys
   is that "no single method's bias can be blamed for a structure that
   survives it" (`design-1d.md` "What the co-association matrix is not");
   a rule under which one family's removal replaces the core set is a
   veto, not a vote. Median over trained (step143000) records.
2. Guard: **dominance**, the largest ARI between the consensus and any
   one family's partition. A rule can be sway-proof by being one family's
   partition (five k = 2 families at L12 already are most of the vote);
   dominance >= 0.95 is read as "the consensus is that family".
   *Added after the one-record smoke (``wiki_paragraph`` L12, the A0
   record, already seen in #100), before the batch:* core-set Jaccard is
   all-or-nothing there (the whole core crosses the threshold together),
   and two empty core sets score 1. So each rule also reports the worst
   Spearman of per-token confidence with and without a family (continuous,
   no threshold), and the records whose core is empty.
3. Reported, not scored: core share, consensus k, share of pairs no
   family had an opinion about (``zero_support``; ``consensus_partition``
   and ``confidence`` read their 0 as disagreement), step 0 beside
   step143000.

The thresholds come from the shuffled-dimension null run_1d uses. That
null is weak (#106, #108 beat it on position and token identity alone),
so core counts here compare rules with each other, not tokens with
structurelessness. Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from core.holdout import add_holdout_args, refuse_held_out

from . import ensemble
from .constants import SUBSTANTIAL_CLUSTER_SIZE
from .gaussian_null import _step_prompt, band_of, input_fingerprint
from .methods import LayerData, available_families, fit
from .p1d_io import layer_activations, load_run
from .selection import select_all_families, selected_labels, selection_weights

NOISE_RULES = ("current", "abstain_small", "singleton")
WEIGHT_RULES = ("stability", "uniform", "kappa")
#: run_1d's own rule: the reference every other rule's consensus is compared to.
REFERENCE = ("current", "stability")
#: The ``null_confidences`` seed offset, so these draws are run_1d's draws.
NULL_SEED_OFFSET = 5081
#: PLACED: "the consensus is that family's partition" (reading 2).
DOMINANCE = 0.95


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def abstain_small(labels: np.ndarray, substantial: int = SUBSTANTIAL_CLUSTER_SIZE) -> np.ndarray:
    """Every token in a cluster smaller than ``substantial`` becomes ``-1``."""
    lab = np.asarray(labels, dtype=np.int64).copy()
    ids, sizes = np.unique(lab[lab >= 0], return_counts=True)
    small = ids[sizes < substantial]
    lab[np.isin(lab, small)] = -1
    return lab


def rule_labels(labels_by_family: Dict[str, np.ndarray], noise: str) -> Tuple[Dict[str, np.ndarray], str]:
    """``(labels, noise_policy)`` to hand ``ensemble.build`` under a noise rule."""
    if noise == "current":
        return dict(labels_by_family), "exclude"
    if noise == "abstain_small":
        return {f: abstain_small(l) for f, l in labels_by_family.items()}, "exclude"
    if noise == "singleton":
        return dict(labels_by_family), "singleton"
    raise ValueError(f"unknown noise rule {noise!r}; use one of {NOISE_RULES}")


def rule_weights(selection: Dict[str, Dict], rule: str) -> Dict[str, float]:
    """``{family: weight}`` under a weight rule, for the families that selected something."""
    raw = selection_weights(selection)
    if rule == "stability":
        return raw
    if rule == "uniform":
        return {f: 1.0 for f in raw}
    if rule == "kappa":
        out = {}
        for f in raw:
            s = selection[f]["selected"]["stability"]["mean_ari"]
            s0 = selection[f]["selected"]["null"]["stability"]["null_mean"]
            out[f] = float(max(0.0, (s - s0) / (1.0 - s0))) if s0 < 1.0 else 0.0
        return out
    raise ValueError(f"unknown weight rule {rule!r}; use one of {WEIGHT_RULES}")


# ---------------------------------------------------------------------------
# One rule on one layer
# ---------------------------------------------------------------------------

def _ari(a: np.ndarray, b: np.ndarray) -> float:
    with warnings.catch_warnings():
        # sklearn warns when most labels are distinct; that is a partition
        # with many singletons, which ARI handles.
        warnings.simplefilter("ignore", UserWarning)
        return float(adjusted_rand_score(a, b))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank agreement of two confidence arrays; 1 when both are constant
    and equal, 0 when only one is constant (no ranking to agree with)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    ca, cb = np.ptp(a) == 0, np.ptp(b) == 0
    if ca or cb:
        return 1.0 if (ca and cb and np.allclose(a, b)) else 0.0
    return float(spearmanr(a, b).statistic)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def graded(labels: Dict[str, np.ndarray], null_labels: List[Dict[str, np.ndarray]],
           weights: Dict[str, float], noise: str) -> Dict[str, object]:
    """Build the ensemble on the tokens and on each null draw under one rule;
    grade the tokens against thresholds from those draws."""
    lab, policy = rule_labels(labels, noise)
    built = ensemble.build(lab, weights=weights, noise_policy=policy)
    nulls = []
    for draw in null_labels:
        nl, _ = rule_labels({f: draw[f] for f in labels}, noise)
        nulls.append(ensemble.build(nl, weights=weights, noise_policy=policy)["confidence"])
    thr = ensemble.confidence_thresholds(nulls)
    pop = ensemble.trichotomy(built["confidence"], thr)
    support = built["co_association"]["support"]
    off = ~np.eye(support.shape[0], dtype=bool)
    return {"built": built, "thresholds": thr, "population": pop,
            "zero_support": float((support[off] <= 0).mean()) if off.any() else 0.0}


def rule_record(labels: Dict[str, np.ndarray], null_labels: List[Dict[str, np.ndarray]],
                selection: Dict[str, Dict], noise: str, weight: str,
                reference: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Everything the reading needs for one rule at one layer."""
    weights = rule_weights(selection, weight)
    voting = {f: l for f, l in labels.items() if weights.get(f, 0.0) > 0}
    if not voting:
        return {"noise": noise, "weight": weight, "skipped": "no family has a positive weight"}
    full = graded(voting, null_labels, weights, noise)
    cons = full["built"]["consensus"]["labels"]
    core = full["population"] == "core"
    n = cons.size

    sway = {}
    for f in voting:
        rest = {g: l for g, l in voting.items() if g != f}
        if not rest:
            continue
        loo = graded(rest, null_labels, weights, noise)
        sway[f] = {"core_jaccard": _jaccard(core, loo["population"] == "core"),
                   "consensus_ari": _ari(cons, loo["built"]["consensus"]["labels"]),
                   "confidence_spearman": _spearman(full["built"]["confidence"],
                                                    loo["built"]["confidence"])}
    dominance = {f: _ari(cons, ensemble.noise_as_singletons(l)) for f, l in voting.items()}
    worst_j = min(sway, key=lambda f: sway[f]["core_jaccard"]) if sway else None
    worst_a = min(sway, key=lambda f: sway[f]["consensus_ari"]) if sway else None
    worst_r = min(sway, key=lambda f: sway[f]["confidence_spearman"]) if sway else None
    top = max(dominance, key=dominance.get)
    counts = {t: int((full["population"] == t).sum()) for t in ("core", "halo", "contested", "uncalibrated")}
    return {
        "noise": noise, "weight": weight,
        "weights": {f: round(w, 4) for f, w in weights.items()},
        "n_families": len(voting),
        "n_clusters": int(full["built"]["consensus"]["n_clusters"]),
        "consensus_strength": float(full["built"]["consensus_strength"]),
        "thresholds": {k: full["thresholds"][k] for k in ("core", "contested")},
        "counts": counts, "core_share": counts["core"] / n,
        "zero_support": full["zero_support"],
        "ari_to_reference": None if reference is None else _ari(reference, cons),
        "sway": sway,
        "worst_core_jaccard": sway[worst_j]["core_jaccard"] if worst_j else None,
        "worst_core_family": worst_j,
        "worst_consensus_ari": sway[worst_a]["consensus_ari"] if worst_a else None,
        "worst_consensus_family": worst_a,
        "worst_confidence_spearman": sway[worst_r]["confidence_spearman"] if worst_r else None,
        "worst_confidence_family": worst_r,
        "dominance": dominance, "dominant_family": top, "dominance_max": dominance[top],
        "_consensus": cons,
    }


# ---------------------------------------------------------------------------
# One layer-record
# ---------------------------------------------------------------------------

def null_draw_labels(data: LayerData, params: Dict[str, Dict], n_draws: int,
                     seed: int) -> List[Dict[str, np.ndarray]]:
    """Each selected setting fitted on run_1d's shuffled-dimension null draws
    (``run_1d.null_confidences``: same generator, same seed, same order)."""
    rng = np.random.default_rng(seed + NULL_SEED_OFFSET)
    X = np.asarray(data.normed, dtype=np.float64)
    n_tokens, d = X.shape
    out = []
    for _ in range(max(0, int(n_draws))):
        shuffled = np.empty_like(X)
        for col in range(d):
            shuffled[:, col] = X[rng.permutation(n_tokens), col]
        norms = np.linalg.norm(shuffled, axis=1, keepdims=True)
        null_data = LayerData.from_normed(shuffled / np.maximum(norms, 1e-12))
        out.append({f: np.asarray(fit(f, p, null_data, seed=seed)) for f, p in params.items()})
    return out


def layer_record(data: LayerData, settings: Dict) -> Dict[str, object]:
    """Tune once, fit the nulls once, then every rule on those labels."""
    selection = select_all_families(
        data, list(settings["families"]), grid=settings["grid"],
        n_repeats=settings["n_repeats"], n_null=settings["n_null"],
        n_null_repeats=settings["n_null_repeats"], top_m=settings["top_m"],
        alpha=settings["alpha"], seed=settings["seed"])
    labels = selected_labels(selection)
    if not labels:
        return {"skipped": "every family abstained at this layer"}
    params = {f: selection[f]["selected"]["params"] for f in labels}
    nulls = null_draw_labels(data, params, settings["n_null_confidence"], settings["seed"])

    ref = rule_record(labels, nulls, selection, *REFERENCE)
    reference = ref.get("_consensus")
    rules = []
    for noise in NOISE_RULES:
        for weight in WEIGHT_RULES:
            rec = ref if (noise, weight) == REFERENCE else rule_record(
                labels, nulls, selection, noise, weight, reference=reference)
            if (noise, weight) == REFERENCE and reference is not None:
                rec["ari_to_reference"] = 1.0
            rules.append({k: v for k, v in rec.items() if k != "_consensus"})
    families = {}
    for f, lab in labels.items():
        sel = selection[f]["selected"]
        families[f] = {"params": sel["params"], "k": sel["shape"]["k"],
                       "k_substantial": sel["shape"]["k_substantial"],
                       "refused": int((lab < 0).sum()),
                       "in_small": int((abstain_small(lab) < 0).sum() - (lab < 0).sum()),
                       "stability": sel["stability"]["mean_ari"],
                       "stability_null_mean": sel["null"]["stability"]["null_mean"]}
    return {"families": families, "abstained": sorted(set(selection) - set(labels)),
            "rules": rules}


def _job(args: Tuple) -> Dict:
    run_dir, layer, settings = args
    run = load_run(Path(run_dir))
    step, prompt = _step_prompt(Path(run_dir))
    t0 = time.time()
    data = LayerData.from_normed(layer_activations(run, layer))
    try:
        rec = layer_record(data, settings)
    except Exception as exc:
        # A pool's traceback does not say which job raised; this does.
        raise RuntimeError(f"vote_rules failed on {run_dir} L{layer}: {exc!r}") from exc
    rec.update({"run_dir": str(run_dir), "step": step, "prompt": prompt,
                "layer": int(layer), "band": band_of(int(layer)), "n_tokens": data.n,
                "seconds": round(time.time() - t0, 1)})
    return rec


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _median(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def summarise(records: List[Dict]) -> List[Dict]:
    """Per (step, noise, weight): medians of the readouts over layer-records."""
    rows: Dict[Tuple, List[Dict]] = {}
    for r in records:
        if "skipped" in r:
            continue
        for rule in r["rules"]:
            if "skipped" in rule:
                continue
            rows.setdefault((r["step"], rule["noise"], rule["weight"]), []).append(rule)
    out = []
    for (step, noise, weight), rs in sorted(rows.items()):
        worst_fam = [x["worst_core_family"] for x in rs if x["worst_core_family"]]
        out.append({
            "step": step, "noise": noise, "weight": weight, "n": len(rs),
            "worst_core_jaccard": _median([x["worst_core_jaccard"] for x in rs]),
            "veto_records": sum(1 for x in rs if x["worst_core_jaccard"] is not None
                                and x["worst_core_jaccard"] < 0.5),
            "empty_core_records": sum(1 for x in rs if x["counts"]["core"] == 0),
            "worst_consensus_ari": _median([x["worst_consensus_ari"] for x in rs]),
            "worst_confidence_spearman": _median([x["worst_confidence_spearman"] for x in rs]),
            "dominance_max": _median([x["dominance_max"] for x in rs]),
            "dominated_records": sum(1 for x in rs if x["dominance_max"] >= DOMINANCE),
            "core_share": _median([x["core_share"] for x in rs]),
            "n_clusters": _median([x["n_clusters"] for x in rs]),
            "zero_support": _median([x["zero_support"] for x in rs]),
            "ari_to_reference": _median([x["ari_to_reference"] for x in rs]),
            "most_swaying_family": max(set(worst_fam), key=worst_fam.count) if worst_fam else None,
        })
    return out


def summary_text(out: Dict) -> str:
    head = ("step        noise          weight     n  worstJ  veto  empty  worstARI  worstRho"
            "  dom   dom>=.95  core   k    zero  ARIref  swayer")
    lines = [f"vote rules: {len(out['records'])} layer-records, inputs {len(out['inputs'])} runs, "
             f"settings {out['settings']}", head]
    f = lambda v, p=3: "  -  " if v is None else f"{v:.{p}f}"
    for r in out["summary"]:
        lines.append(f"{r['step']:<11} {r['noise']:<14} {r['weight']:<9} {r['n']:>3}  "
                     f"{f(r['worst_core_jaccard'])}  {r['veto_records']:>4}  {r['empty_core_records']:>5}  "
                     f"{f(r['worst_consensus_ari'])}     {f(r['worst_confidence_spearman'])}"
                     f"     {f(r['dominance_max'], 2)}  {r['dominated_records']:>6}   "
                     f"{f(r['core_share'], 2)}  {f(r['n_clusters'], 0):>3}  {f(r['zero_support'], 2)}  "
                     f"{f(r['ari_to_reference'], 2)}  {r['most_swaying_family']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="Phase 1 prompt directories (each with activations.npz)")
    ap.add_argument("--out", type=Path, required=True, help="JSON to write (summary beside it)")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18])
    ap.add_argument("--grid", choices=("full", "quick"), default="quick")
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--n-null-repeats", type=int, default=3)
    ap.add_argument("--n-null-confidence", type=int, default=10)
    ap.add_argument("--top-m", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    add_holdout_args(ap)
    args = ap.parse_args(argv)

    runs, record = refuse_held_out(list(args.runs), allow=args.allow_holdout,
                                   drop=args.v1_only, context="vote_rules")
    missing = [r for r in runs if not (r / "activations.npz").exists()]
    if missing:
        print(f"refusing: no activations.npz in {missing}", file=sys.stderr)
        return 1
    if not runs:
        print("no runs", file=sys.stderr)
        return 1
    settings = {"families": available_families(), "grid": args.grid,
                "n_repeats": args.n_repeats, "n_null": args.n_null,
                "n_null_repeats": args.n_null_repeats,
                "n_null_confidence": args.n_null_confidence, "top_m": args.top_m,
                "alpha": args.alpha, "seed": args.seed,
                "substantial_cluster_size": SUBSTANTIAL_CLUSTER_SIZE}
    jobs = [(str(r), int(L), settings) for r in runs for L in args.layers]

    # Resumable, as gaussian_null: a part is reused only if its settings and
    # its run's input files match this call's.
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    fingerprints = {str(r): input_fingerprint(r, ("activations.npz",)) for r in runs}

    def part_of(job) -> Path:
        return parts / f"{Path(job[0]).parent.name}__{Path(job[0]).name}__L{job[1]}.json"

    done, todo = [], []
    for j in jobs:
        p = part_of(j)
        if p.exists():
            rec = json.loads(p.read_text())
            if rec.get("_settings") == {**settings, "input": fingerprints[j[0]]}:
                done.append(rec)
                continue
        todo.append(j)
    print(f"  {len(done)} of {len(jobs)} records already done; running {len(todo)}", flush=True)

    t0 = time.time()

    def keep(job, rec):
        rec["_settings"] = {**settings, "input": fingerprints[job[0]]}
        tmp = part_of(job).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(part_of(job))
        done.append(rec)
        print(f"  {len(done)}/{len(jobs)} done ({time.time() - t0:.0f} s this call)", flush=True)

    if args.workers > 1 and todo:
        from multiprocessing import get_context
        by_key = {(j[0], j[1]): j for j in todo}
        with get_context("spawn").Pool(args.workers) as pool:
            for rec in pool.imap_unordered(_job, todo, chunksize=1):
                keep(by_key[(rec["run_dir"], rec["layer"])], rec)
    else:
        for job in todo:
            keep(job, _job(job))

    order = {(j[0], j[1]): i for i, j in enumerate(jobs)}
    records = sorted(done, key=lambda r: order[(r["run_dir"], r["layer"])])
    for r in records:
        r.pop("_settings", None)
    out = {"settings": settings, "holdout": record, "inputs": [str(r) for r in runs],
           "layers": args.layers, "seconds": round(time.time() - t0, 1), "records": records}
    out["summary"] = summarise(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    text = summary_text(out)
    args.out.with_suffix(".txt").write_text(text + "\n")
    print(text)
    print(f"  wrote {args.out} ({out['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
