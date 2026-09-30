"""
beta_long_compare.py — β on the long prompts against #110's β on their v1
prefixes (Phase 1d, `status-1d.md` "Long prompts").

Reads two `beta_refit` outputs: the long runs' and #110's
(`data/p1d/beta_refit_2026-09-29/beta_refit.json`), keeps #110's heads for
the prompts the long run has (``<key>_long`` → ``<key>``), and reports per
(dedupe, step, band) and variant (``linear``, ``fe_full``):

- the median [IQR] of each side over its own finite heads;
- the paired difference long − v1 over heads finite on both sides (same
  step, prompt, layer, head): median [IQR] and how many went up.

The paired row is the length effect on one head; two medians also move
when different heads fall out as NaN. No fits here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .beta_refit import EXCLUDE, _band

COMPARED = ("linear", "fe_full")
BANDS = ("L1-8", "L9-16", "L17-23", "L1-23")


def _key(r: Dict, prompt: str):
    return (r["dedupe"], r["step"], prompt, r["layer"], r["head"])


def _q(x: np.ndarray):
    return ([float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75)),
             int(x.size)] if x.size else None)


def compare(long_heads: List[Dict], v1_heads: List[Dict]) -> List[Dict]:
    longs = {_key(r, r["prompt"][: -len("_long")]): r
             for r in long_heads if r["prompt"].endswith("_long")}
    prompts = {k[2] for k in longs}
    v1 = {_key(r, r["prompt"]): r for r in v1_heads
          if r["prompt"] in prompts and r["prompt"] not in EXCLUDE}
    groups: Dict = {}
    for k in set(longs) | set(v1):
        dedupe, step, _, layer, _ = k
        if layer < 1:
            continue
        for b in (_band(layer), "L1-23"):
            groups.setdefault((dedupe, step, b), []).append(k)
    rows = []
    for (dedupe, step, band), keys in sorted(groups.items()):
        row = {"dedupe": dedupe, "step": step, "band": band}
        for v in COMPARED:
            lo = np.array([longs[k][v] for k in keys if k in longs], dtype=float)
            vo = np.array([v1[k][v] for k in keys if k in v1], dtype=float)
            both = [k for k in keys if k in longs and k in v1]
            d = np.array([longs[k][v] - v1[k][v] for k in both], dtype=float)
            d = d[np.isfinite(d)]
            row[v] = {"long": _q(lo[np.isfinite(lo)]), "v1": _q(vo[np.isfinite(vo)]),
                      "paired_diff": _q(d), "n_up": int((d > 0).sum())}
        rows.append(row)
    return rows


def text(rows: List[Dict], prompts: Sequence[str]) -> str:
    def f(q):
        return "—" if q is None else f"{q[0]:.2f} [{q[1]:.2f}, {q[2]:.2f}] n={q[3]}"
    lines = [f"β long vs v1 prefix, same prompts ({', '.join(sorted(prompts))}); "
             "median [IQR] n; paired = long − v1 per head", ""]
    for r in rows:
        for v in COMPARED:
            c = r[v]
            n = c["paired_diff"][3] if c["paired_diff"] else 0
            lines.append(f"{'dd' if r['dedupe'] else 'all'} {r['step']:<10} {r['band']:<6} "
                         f"{v:<7} | v1 {f(c['v1'])} | long {f(c['long'])} | "
                         f"paired {f(c['paired_diff'])}, up {c['n_up']}/{n}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--long", type=Path, required=True, help="beta_refit output on the long runs")
    ap.add_argument("--v1", type=Path, required=True, help="#110's beta_refit output")
    ap.add_argument("--out", type=Path, required=True, help="JSON to write (text beside it)")
    args = ap.parse_args(argv)
    long_heads = json.loads(args.long.read_text())["heads"]
    v1_heads = json.loads(args.v1.read_text())["heads"]
    rows = compare(long_heads, v1_heads)
    prompts = sorted({r["prompt"][: -len("_long")] for r in long_heads
                      if r["prompt"].endswith("_long")})
    if not rows or not any(r["fe_full"]["paired_diff"] for r in rows):
        print("refusing: no head is on both sides", flush=True)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"long": str(args.long), "v1": str(args.v1),
                                    "prompts": prompts, "rows": rows}))
    args.out.with_suffix(".txt").write_text(text(rows, prompts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
