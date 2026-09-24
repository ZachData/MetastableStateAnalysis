"""F0 of `p10_cluster_function/notes-10.md` §8: the anchor test.

Are cluster nuclei early tokens, as the parking reading says? Per-token,
position-indexed, against the HDBSCAN partition and nothing else — **no beta,
no unit convention, no forward pass and no reading of the paper.**

WHY THIS IS F0 AND NOT THE SCALING LAW
---------------------------------------
`lit-1.md` §4 described `2411.04990`'s parking result as a cluster count as a
function of `n`, with the Renyi density constant 0.7476, and rated checking it
the project's best cheap experiment. `lit-10.md` §5's second scan found that
description wrong on both halves: the published scaling is
``Theta(beta^((d-1)/2))`` — a law in **beta and dimension, not in `n`** — and
0.7476 does not appear in it. At `d = 1024` the exponent is 511.5 and the
prediction is unusable.

What the paper does supply is a **mechanism**: early tokens act as nuclei for
cluster formation. That is a per-token, position-indexed prediction, it needs
none of the machinery the scaling law needs, and it is what this runner tests.

THE STATISTIC, AND THE NULL IT IS NOTHING WITHOUT
--------------------------------------------------
``mean_nucleus_position`` is the mean, over clusters, of the normalised
position of each cluster's **earliest member** — the only reading of "nucleus"
a causal mask permits, since a token at position p cannot have been drawn
toward anything later than p.

Its raw value means nothing. The minimum of a size-s subset of n positions
falls at about ``n/(s+1)``, so the partition's SIZE PROFILE moves the raw
number before any position coupling does. Permuting labels preserves the
cluster count and every cluster's size exactly, so that is held fixed and what
is left is the coupling.

**Two nulls, because one of them answers the wrong question.** The ordinary
permutation is free to move clusters between the early and late halves of the
sequence. If the clustered population as a whole sits later than the
unclustered one — and on this sweep it does, `clustered_position_bias` is
positive — then its clusters' earliest members are late for that reason alone,
and the ordinary null reports that as a finding about nuclei. The restricted
null (`core.nulls.label_permutation_null_within`) holds the clustered/noise
split fixed and shuffles only which clustered token carries which cluster id,
so it asks what F0 means: **given which tokens are clustered, are the seeds
early?** Both are reported; disagreement between them is informative.

`clustered_position_bias` runs beside them: the mean normalised position of
clustered minus noise tokens, which is the confound itself. `math-10.md` §1
shows a partition with that property reproduces the attention flip with no
learned behaviour, so this row and row A0 read the same confound from two
directions.

DIRECTIONS, FIXED HERE BEFORE ANY SWEEP IS READ
------------------------------------------------
Nucleus position: **"less"**. The parking reading says clusters are anchored
early, so small is the prediction. It is written in
`core.parking.mean_nucleus_position`'s docstring as well as here.
Position bias: **two-sided**; both signs are informative and neither was
predicted.

TIER 1, EXPLORATORY, NOT REGISTERED
------------------------------------
`notes-10.md` §11 calls F0 the phase's tier-2/3 candidate and the best one in
the project. It is being run exploratory first, by decision, which caps it at
tier 1: the answer will have been seen before any wording is frozen. Nothing
here touches `claims/registry.json`, and no result from this runner may be
quoted as an adjudication.

The e-values are reported because a number with a null behind it should be
calibrated as one. The merger is the **arithmetic mean**, not the product: the
units share a model, a text and a forward pass. See
`tools/run/p10_attention_baseline.py`'s docstring and
`tests/test_core_evalues_average.py`.

Run:
    python tools/run/p10_anchor.py --out data/analysis/p10_f0_anchor.json
    python tools/run/p10_anchor.py --root /run/media/system/HDD_1TB/Mets_archive
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))  # this checkout, not a hard-coded main tree
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.evalues import (
    DEFAULT_ALPHA,
    DEFAULT_KAPPA,
    average_p,
    calibrate,
    max_attainable_average_E,
)
from core.nulls import (
    label_permutation_null,
    label_permutation_null_within,
    p_from_null_tolerant,
)
from core.holdout import add_holdout_args, refuse_held_out
from core.parking import (
    cluster_nuclei,
    clustered_position_bias,
    mean_nucleus_position,
)
from tools.run.backfill_hdbscan import labels_provenance, read_labels

N_PERMUTATIONS = 2000

_STEP = re.compile(r"step(\d+)")


def checkpoint_of(run_dir_name: str):
    """The training step a directory's name encodes, or None."""
    m = _STEP.search(run_dir_name)
    return int(m.group(1)) if m else None


def measure_layer(labels, rng) -> dict:
    """The two statistics and their p-values for one layer, or ``None`` when
    the layer carries no usable partition.

    A layer with no clusters, or with no noise, is dropped rather than scored:
    both statistics are undefined there and a dropped layer must not become a
    measured zero.
    """
    lab = np.asarray(labels)
    n = lab.size
    if n < 4:
        return None
    n_clusters = len(set(lab.tolist()) - {-1})
    if n_clusters < 2:
        return None

    positions = np.arange(n, dtype=np.float64)
    nucleus = mean_nucleus_position(positions, lab)
    if not np.isfinite(nucleus):
        return None

    # TWO NULLS, because they answer different questions and the first one
    # alone cannot tell nucleation from the population split.
    #
    #  wide    permutes labels among ALL tokens. "Is this labelling special
    #          among relabellings of the same sizes?" If clustered tokens sit
    #          later than unclustered ones — and on this sweep they do — then
    #          their clusters' earliest members are late for that reason
    #          alone, and this null reports it as a finding about nuclei.
    #  within  holds the clustered/noise split FIXED and shuffles only which
    #          clustered token carries which cluster id. "GIVEN which tokens
    #          are clustered, are the seeds early?" That is what F0 means.
    nuc_draws = label_permutation_null(positions, lab, mean_nucleus_position,
                                       n_permutations=N_PERMUTATIONS, rng=rng)
    nuc = p_from_null_tolerant(nucleus, nuc_draws, alternative="less")
    within_draws = label_permutation_null_within(
        positions, lab, mean_nucleus_position,
        n_permutations=N_PERMUTATIONS, rng=rng)
    within = p_from_null_tolerant(nucleus, within_draws, alternative="less")

    out = {
        "n_tokens": int(n),
        "n_clusters": int(n_clusters),
        "noise_fraction": round(float((lab == -1).mean()), 4),
        "mean_nucleus_position": round(float(nucleus), 4),
        "nucleus_null_mean": round(float(np.mean(nuc_draws)), 4),
        "nucleus_p": float(nuc["p_value"]),
        "nucleus_degenerate": bool(nuc["degenerate_null"]),
        "nucleus_within_null_mean": round(float(np.mean(within_draws)), 4),
        "nucleus_within_p": float(within["p_value"]),
        "nucleus_within_degenerate": bool(within["degenerate_null"]),
        "nucleus_sigma": round(float(nuc.get("n_sigma", float("nan"))), 3)
        if np.isfinite(nuc.get("n_sigma", float("nan"))) else None,
    }

    bias = clustered_position_bias(positions, lab)
    if np.isfinite(bias):
        bias_draws = label_permutation_null(positions, lab, clustered_position_bias,
                                            n_permutations=N_PERMUTATIONS, rng=rng)
        b = p_from_null_tolerant(bias, bias_draws, alternative="two-sided")
        out["position_bias"] = round(float(bias), 4)
        out["position_bias_p"] = float(b["p_value"])
    else:
        out["position_bias"] = None
        out["position_bias_p"] = None

    # Descriptive, not tested: where the earliest members actually sit.
    nuclei = cluster_nuclei(lab)
    firsts = np.array([v["first"] for v in nuclei.values()], dtype=np.float64)
    out["nucleus_first_decile_fraction"] = round(
        float(np.mean(firsts <= 0.1 * (n - 1))), 4)
    return out


def measure_directory(run_dir: Path, rng) -> dict:
    labels_by_layer = read_labels(run_dir)
    if not labels_by_layer:
        return {"run_dir": run_dir.name, "skipped": "no HDBSCAN partition"}
    layers = []
    for layer in sorted(labels_by_layer):
        rec = measure_layer(labels_by_layer[layer], rng)
        if rec is not None:
            rec["layer"] = int(layer)
            layers.append(rec)
    return {
        "run_dir": run_dir.name,
        "timestamp": run_dir.parent.name,
        "checkpoint": checkpoint_of(run_dir.name),
        "labels_from": labels_provenance(run_dir),
        "layers": layers,
    }


def _merge(ps: list) -> dict:
    ps = [p for p in ps if p is not None and np.isfinite(p)]
    if not ps:
        return {"n": 0}
    E, reject = average_p(ps)
    return {
        "n": len(ps),
        "E": round(float(E), 4),
        "reject": bool(reject),
        "median_p": round(float(np.median(ps)), 4),
        "frac_below_05": round(float(np.mean(np.array(ps) < 0.05)), 4),
    }


def aggregate(dirs: list) -> dict:
    rows = [(d, l) for d in dirs for l in d.get("layers", [])]
    if not rows:
        return {"n_units": 0}
    layers = [l for _, l in rows]

    def m(key):
        v = np.array([l[key] for l in layers
                      if l.get(key) is not None and np.isfinite(l[key])])
        return round(float(v.mean()), 4) if v.size else None

    out = {
        "n_units": len(layers),
        "n_directories": sum(1 for d in dirs if d.get("layers")),
        "mean_nucleus_position": m("mean_nucleus_position"),
        "mean_nucleus_null": m("nucleus_null_mean"),
        "mean_nucleus_within_null": m("nucleus_within_null_mean"),
        "mean_position_bias": m("position_bias"),
        "mean_n_clusters": m("n_clusters"),
        "mean_noise_fraction": m("noise_fraction"),
        "nucleus": _merge([l["nucleus_p"] for l in layers]),
        "nucleus_within": _merge([l["nucleus_within_p"] for l in layers]),
        "position_bias": _merge([l["position_bias_p"] for l in layers]),
    }

    # Per checkpoint, because "when" is the axis this project exists for, and a
    # developmental effect averaged over training reads as no effect.
    by_ckpt = defaultdict(list)
    for d, l in rows:
        if d.get("checkpoint") is not None:
            by_ckpt[d["checkpoint"]].append(l)
    out["by_checkpoint"] = {
        str(step): {
            "n_units": len(ls),
            "mean_nucleus_position": round(float(np.mean(
                [l["mean_nucleus_position"] for l in ls])), 4),
            "nucleus": _merge([l["nucleus_p"] for l in ls]),
            "nucleus_within": _merge([l["nucleus_within_p"] for l in ls]),
        }
        for step, ls in sorted(by_ckpt.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(DATA / "phase12"))
    ap.add_argument("--pattern", default="pythia-410m-*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_f0_anchor.json"))
    add_holdout_args(ap)
    args = ap.parse_args()

    root = Path(args.root)
    candidates, holdout = refuse_held_out(
        (d for ts in sorted(root.glob("*")) if ts.is_dir()
         for d in sorted(ts.glob(args.pattern)) if d.is_dir()),
        allow=args.allow_holdout, drop=args.v1_only, context="p10_anchor")
    targets = sorted(
        d for d in candidates if read_labels(d)
    )
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)} directories carry a partition under {root}")

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    dirs = []
    for i, d in enumerate(targets, 1):
        dirs.append(measure_directory(d, rng))
        if i % 25 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] {d.name}  ({time.time() - t0:.0f}s)", flush=True)

    summary = aggregate(dirs)
    record = {
        "schema": "p10_f0_anchor/1",
        "row": "F0 — the anchor test: are cluster nuclei early tokens?",
        "tier": "1 (exploratory, unregistered — run before design-10.md freezes "
                "anything, by decision; not quotable as an adjudication)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "holdout": holdout,
        "n_permutations": N_PERMUTATIONS,
        "seed": args.seed,
        "kappa": DEFAULT_KAPPA,
        "alpha": DEFAULT_ALPHA,
        "merger": "arithmetic mean (core.evalues.average)",
        "resolution_floor_e": round(calibrate(1.0 / (N_PERMUTATIONS + 1)), 3),
        # "Could this design have rejected at all?" -- a different question
        # from the resolution floor, and one a permutation null merged by the
        # mean can answer exactly. See `core.evalues.max_attainable_average_E`.
        "max_attainable_E": round(max_attainable_average_E(N_PERMUTATIONS)[0], 3),
        "design_can_reject": bool(max_attainable_average_E(N_PERMUTATIONS)[1]),
        "alternatives": {"nucleus_p": "less", "nucleus_within_p": "less",
                         "position_bias_p": "two-sided"},
        "nulls": {
            "nucleus_p": "label permutation over ALL tokens — confounded by the "
                         "clustered/noise position split",
            "nucleus_within_p": "label permutation restricted to clustered "
                                "tokens — the nucleation question proper",
        },
        "summary": summary,
        "directories": dirs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out}")
    for k, v in summary.items():
        if k != "by_checkpoint":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
