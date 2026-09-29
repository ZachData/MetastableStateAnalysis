"""
p1d_cluster_ensemble/attention_null_report.py — the tables `status-1d.md`
"Attention communities" quotes, from `attention_null.py`'s output.

Per (step, null, graph, statistic, band): records in the stronger tail, real
against a calibration **with the same dedupe setting** (refuses otherwise,
#106's rule), plus median observed and null values. Excludes
``repeated_tokens`` (skipped when deduped, so leaving it in would change the
prompt set between columns). Also the β table: per (step, band), β and R²
over heads × prompts at the window's first layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .attention_null import TAIL, summarise
from .gaussian_null import band_of

EXCLUDE = ("repeated_tokens",)
BANDS = ("L0", "L1-8", "L9-16", "L17-24", "L1-24")


def _records(d: Dict) -> List[Dict]:
    """Records outside ``EXCLUDE``, plus a pooled ``L1-24`` copy of every layer >= 1."""
    out = []
    for r in d["records"]:
        if "skipped" in r or r["prompt"] in EXCLUDE:
            continue
        out.append(r)
        if r["layer"] >= 1:
            out.append({**r, "band": "L1-24"})
    return out


def _summ(d: Dict) -> Dict:
    return {(r["step"], r["null"], r["graph"], r["stat"], r["band"]): r
            for r in summarise(_records(d))}


def table(real: Dict, cal: Dict) -> List[Dict]:
    if bool(real.get("dedupe_strings")) != bool(cal.get("dedupe_strings")):
        raise ValueError("refusing: the calibration's dedupe setting differs from the real run's")
    R, C = _summ(real), _summ(cal)
    out = []
    for k, r in sorted(R.items()):
        c = C.get(k)
        out.append({"step": k[0], "null": k[1], "graph": k[2], "stat": k[3], "band": k[4],
                    "n": r["n"], "tail": r["tail"], "median_obs": r["median_obs"],
                    "median_null": r["median_null"], "median_z": r["median_z"],
                    "cal_n": c["n"] if c else 0, "cal_tail": c["tail"] if c else None,
                    "cal_median_z": c["median_z"] if c else None})
    return out


def beta_table(real: Dict) -> List[Dict]:
    by: Dict = {}
    for r in _records(real):
        band = r.get("band") or band_of(r["layer"])
        for h in r["betas"][0]:
            if h["beta"] is None or not np.isfinite(h["beta"]):
                continue
            by.setdefault((r["step"], band), {"beta": [], "r2": []})
            by[(r["step"], band)]["beta"].append(h["beta"])
            by[(r["step"], band)]["r2"].append(h["r2"])
    out = []
    for (step, band), v in sorted(by.items()):
        b, r2 = np.array(v["beta"]), np.array(v["r2"], dtype=float)
        out.append({"step": step, "band": band, "n": int(b.size),
                    "beta_median": float(np.median(b)),
                    "beta_iqr": [float(np.percentile(b, 25)), float(np.percentile(b, 75))],
                    "r2_median": float(np.nanmedian(r2))})
    return out


def text(rows: List[Dict], betas: List[Dict], stats: Sequence[str]) -> str:
    lines = ["tail = records in the stronger 2.5 % tail (" +
             ", ".join(f"{k} {v}" for k, v in TAIL.items() if k in stats) + "); real / calibration"]
    for s in stats:
        lines.append(f"\n[{s}] step       null graph        band     n  tail/cal   med obs  med null  med z [cal]")
        for r in rows:
            if r["stat"] != s:
                continue
            cal_t = "" if r["cal_tail"] is None else str(r["cal_tail"])
            cal_z = "" if r["cal_median_z"] is None else f"{r['cal_median_z']:.2f}"
            lines.append(f"      {r['step']:<10} {r['null']:<4} {r['graph']:<12} {r['band']:<7} {r['n']:>3}  "
                         f"{r['tail']:>4}/{cal_t:<4} "
                         f"{r['median_obs']:>8.3f}  {r['median_null']:>8.3f}  {r['median_z']:>6.2f} "
                         f"[{cal_z}]")
    lines.append("\n[beta] step       band     n   median   IQR              R2 median")
    for b in betas:
        lines.append(f"       {b['step']:<10} {b['band']:<7} {b['n']:>4}  {b['beta_median']:>6.2f}   "
                     f"[{b['beta_iqr'][0]:.2f}, {b['beta_iqr'][1]:.2f}]   {b['r2_median']:.3f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--null", type=Path, required=True)
    ap.add_argument("--calibrate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats", nargs="*", default=list(TAIL))
    args = ap.parse_args(argv)
    real, cal = json.loads(args.null.read_text()), json.loads(args.calibrate.read_text())
    try:
        rows = table(real, cal)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    betas = beta_table(real)
    args.out.write_text(json.dumps({"null": str(args.null), "calibrate": str(args.calibrate),
                                    "dedupe_strings": bool(real.get("dedupe_strings")),
                                    "table": rows, "betas": betas}))
    t = text(rows, betas, args.stats)
    args.out.with_suffix(".txt").write_text(t + "\n")
    print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
