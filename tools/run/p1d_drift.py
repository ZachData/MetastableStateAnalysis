"""Does Phase 1d's tuning reduce the float-noise drift? (tier 1, no null)

Reads two Phase 1d output roots written by ``tools/run/p1d_drift_batch.sh``
on the same (step, prompt) runs from two sweeps whose activations differ only
by float noise (``p10_cluster_function/status-10.md`` §3, §1.11), and reports
per layer-record the ARI between the sweeps for:

- ``shipped``: the stored HDBSCAN labels (``min_cluster_size=2``), the baseline;
- each tuned family's selected partition (``selected_labels``), and whether
  the selected parameters are the same on both sides;
- the consensus partition and the core/halo/contested grading.

A family that abstains on one side and not the other counts as a pick change,
with no ARI. Noise (-1) is kept as a label, as the baseline in §3 does.

    python -m tools.run.p1d_drift <root_a> <root_b> [--json out.json]
    python -m tools.run.p1d_drift <root_a> <root_b> --self   # same input twice
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari


def _load(run: Path) -> tuple[dict, dict]:
    res = json.loads((run / "p1d_results.json").read_text())
    arr = np.load(run / "p1d_ensemble.npz")
    return res, {k: arr[k] for k in arr.files}


def _layer(res: dict, arr: dict, L: str) -> dict:
    pl = res["per_layer"][L]
    fam = {}
    for f, s in pl["selection"].items():
        sel = s.get("selected")
        fam[f] = None if sel is None else (sel["params"], np.asarray(s["selected_labels"]))
    return {
        "fam": fam,
        "consensus": arr.get(f"consensus_labels_L{L}"),
        "population": arr.get(f"population_L{L}"),
        "shipped": arr.get(f"hdbscan_label_L{L}"),
    }


def compare(a: Path, b: Path) -> list[dict]:
    rows = []
    for run in sorted(p for p in a.iterdir() if (p / "p1d_results.json").exists()):
        other = b / run.name
        if not (other / "p1d_results.json").exists():
            continue
        ra, aa = _load(run)
        rb, ab = _load(other)
        step = int(run.name.split("step")[1].split("_")[0])
        key = run.name.split(f"step{step}_")[1]
        for L in sorted(set(ra["per_layer"]) & set(rb["per_layer"]), key=int):
            x, y = _layer(ra, aa, L), _layer(rb, ab, L)
            row = {"step": step, "prompt": key, "layer": int(L),
                   "n": int(len(x["consensus"])) if x["consensus"] is not None else None}
            for name in ("shipped", "consensus"):
                u, v = x[name], y[name]
                row[name] = None if u is None or v is None else float(ari(u, v))
            if x["population"] is not None and y["population"] is not None:
                row["grade_agree"] = float(np.mean(x["population"] == y["population"]))
            for f in sorted(set(x["fam"]) | set(y["fam"])):
                p, q = x["fam"].get(f), y["fam"].get(f)
                if p is None or q is None:
                    row[f] = {"same_pick": p is None and q is None, "ari": None,
                              "abstain": [p is None, q is None]}
                else:
                    row[f] = {"same_pick": p[0] == q[0], "ari": float(ari(p[1], q[1])),
                              "k": [int(len(set(p[1].tolist()) - {-1})),
                                    int(len(set(q[1].tolist()) - {-1}))]}
            rows.append(row)
    return rows


def summarise(rows: list[dict]) -> None:
    fams = sorted({k for r in rows for k, v in r.items() if isinstance(v, dict)})
    strata = {"all": rows,
              "shipped moved": [r for r in rows if (r["shipped"] or 1) < 1],
              "shipped identical": [r for r in rows if r["shipped"] == 1]}
    for name, rs in strata.items():
        if not rs:
            continue
        print(f"\n== {name}: {len(rs)} layer-records")
        print(f"{'partition':18s} {'n':>4s} {'identical':>9s} {'mean':>6s} {'p5':>6s} {'min':>6s} {'same pick':>9s}")

        def line(label, vals, picks=None):
            v = np.array([x for x in vals if x is not None], dtype=float)
            if v.size == 0:
                print(f"{label:18s} {0:4d}"); return
            pk = "" if picks is None else f"{sum(picks)}/{len(picks)}"
            print(f"{label:18s} {v.size:4d} {int((v >= 1 - 1e-12).sum()):9d} "
                  f"{v.mean():6.3f} {np.percentile(v, 5):6.3f} {v.min():6.3f} {pk:>9s}")
        line("shipped HDBSCAN", [r["shipped"] for r in rs])
        for f in fams:
            line(f, [r[f]["ari"] for r in rs if f in r], [r[f]["same_pick"] for r in rs if f in r])
        line("consensus", [r["consensus"] for r in rs])
        line("grade agreement", [r.get("grade_agree") for r in rs])
    by = defaultdict(list)
    for r in rows:
        by[r["step"]].append(r)
    print("\nper step: shipped moved / consensus moved / any family pick changed")
    for s, rs in sorted(by.items()):
        print(f"  {s:>6d}: {sum((r['shipped'] or 1) < 1 for r in rs)}/{len(rs)}  "
              f"{sum((r['consensus'] or 1) < 1 for r in rs)}/{len(rs)}  "
              f"{sum(any(not r[f]['same_pick'] for f in fams if f in r) for r in rs)}/{len(rs)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--self", action="store_true",
                   help="a and b are the same input; any difference is 1d's own nondeterminism")
    args = p.parse_args()
    rows = compare(args.a, args.b)
    if not rows:
        raise SystemExit("no (run, layer) present in both roots")
    summarise(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
