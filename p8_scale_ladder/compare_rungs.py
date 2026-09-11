"""Cross-rung comparison of the catalogue sweep, on threshold-free statistics.

WHY THIS EXISTS
---------------
`design-8.md` is explicit that **no absolute threshold transfers** between
rungs -- "not the +0.05 membership bar, not `r* = 12`, not the noise floor".
The first 70m write-up broke that rule anyway, comparing "19 of 48 heads clear
+0.05" against 410m's handful of 384. That comparison is not interpretable:
the bar was calibrated on 410m's baseline NLL (0.585) and 70m's is an order of
magnitude higher (5.725), so the same nats mean different things.

Invariant 1's actual claim is about **shape** -- "a heavy tail with an
identifiable end -- not the count". So the statistics here are the ones that
express shape without naming a cut:

  participation ratio   `(sum d^2)^2 / sum d^4` over |dNLL|. The effective
                        NUMBER OF HEADS carrying the effect: 1.0 if one head
                        does everything, n if all n are equal. Directly the
                        "how concentrated is the tail" question, and the same
                        functional that `member_subspace_geometry` already uses
                        for spectra, so it is not a new convention.
  top-k share           fraction of total |dNLL| held by the k largest heads.
                        A Lorenz reading; `gini` is its threshold-free summary.
  null-relative count   heads clearing the rung's OWN control distribution
                        rather than an imported bar -- the only "count" that
                        survives the no-threshold rule. Defined as
                        |dNLL| > q-th percentile of the per-rung median bulk.

THE ABLATION AXIS. Every row is reported per ablation mode, because the mode is
not a detail: zero-ablation takes the residual stream off-distribution, and one
head is 1/8 of a pythia-70m layer against 1/16 at 410m, so the off-distribution
component of dNLL does not cancel in a cross-rung comparison. `mean` is the
control. See `induction_rank_sweep.ablate_heads`.

NO p-value; nothing registered. Both rungs are exploration-tier under the
`design-8.md` rung policy.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np


def participation_ratio(v):
    """Effective number of contributing heads. Threshold-free."""
    e = np.asarray(v, dtype=np.float64) ** 2
    s = e.sum()
    return float(s ** 2 / (e ** 2).sum()) if s > 0 else float("nan")


def gini(v):
    """Concentration of |dNLL| over heads: 0 = all equal, ->1 = one head."""
    x = np.sort(np.abs(np.asarray(v, dtype=np.float64)))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def load(path):
    d = json.load(open(path))
    rows = d["heads"]
    return d, np.array([r["dnll"] for r in rows], dtype=np.float64), rows


def summarise(d, vals, rows, topk):
    a = np.abs(vals)
    order = np.argsort(-a)
    tot = a.sum()
    # The rung's own bulk: the central half, which no induction head is in.
    bulk = a[(a >= np.percentile(a, 25)) & (a <= np.percentile(a, 75))]
    bar = float(np.percentile(a, 75) + 3 * (bulk.std() or 1e-12))
    return {
        "model": d.get("model") or "pythia-410m",
        "ablation": d.get("ablation", "ov"),
        "step": d.get("step"),
        "n_heads": len(vals),
        "baseline_nll": d.get("baseline_nll"),
        "median_dnll": float(np.median(vals)),
        "median_abs": float(np.median(a)),
        "max": float(vals.max()),
        "min": float(vals.min()),
        # shape, threshold-free
        "participation_ratio": participation_ratio(a),
        "pr_frac_of_heads": participation_ratio(a) / len(vals),
        "gini": gini(a),
        "top1_share": float(a[order[0]] / tot) if tot else float("nan"),
        "topk_share": float(a[order[:topk]].sum() / tot) if tot else float("nan"),
        # the only defensible count: against this rung's own bulk
        "n_above_own_bulk": int((a > bar).sum()),
        "own_bulk_bar": bar,
        "top_heads": [rows[j]["head"] for j in order[:topk]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="redundancy_catalog*.json, one per (rung, ablation)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rowsum = [summarise(*load(f), args.topk) for f in args.files]

    hdr = (f"{'model':>13} {'abl':>5} {'base':>7} {'med':>9} {'PR':>6} "
           f"{'PR/n':>6} {'gini':>6} {'top1':>6} {'top%d' % args.topk:>6} "
           f"{'>bulk':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rowsum:
        print(f"{r['model']:>13} {r['ablation']:>5} {r['baseline_nll']:>7.3f} "
              f"{r['median_dnll']:>+9.5f} {r['participation_ratio']:>6.2f} "
              f"{r['pr_frac_of_heads']:>6.3f} {r['gini']:>6.3f} "
              f"{r['top1_share']:>6.3f} {r['topk_share']:>6.3f} "
              f"{r['n_above_own_bulk']:>6d}")
    print()
    for r in rowsum:
        print(f"{r['model']} / {r['ablation']}: top{args.topk} "
              f"{', '.join(r['top_heads'])}")

    # The cross-rung ratios the phase actually cares about, per ablation mode,
    # so a mode-driven artifact cannot hide inside a scale claim.
    print()
    by_mode = {}
    for r in rowsum:
        by_mode.setdefault(r["ablation"], {})[r["model"]] = r
    for mode, d in sorted(by_mode.items()):
        if len(d) < 2:
            continue
        small = min(d.values(), key=lambda r: r["n_heads"])
        large = max(d.values(), key=lambda r: r["n_heads"])
        print(f"[{mode}] {small['model']} vs {large['model']}: "
              f"median x{small['median_abs'] / large['median_abs']:.1f}, "
              f"PR/n {small['pr_frac_of_heads']:.3f} vs "
              f"{large['pr_frac_of_heads']:.3f}, "
              f"gini {small['gini']:.3f} vs {large['gini']:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"_what_this_is": __doc__, "rungs": rowsum},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
