"""
p1e_energy_field/page.py — the data behind 1e's page (`viz/index.html`, the template;
published at https://claude.ai/artifact/8ZqeVhbPrtTBxkDWGm4Hts): U2's labels by
training stage and U1's one well, read from the reports already written.

No computation of its own: each grid is a label family's (step, band) labels from
a `report.json`, the per-head counts from `u2_heads`, the sweep's median `k_eff`
and the calibrated lumpiness from `u1_field`'s `labels.json`. Refuses if a
(step, band) cell is missing, or if a grid's label counts over the trained bands
differ from the count its report printed (the page shows what the report says).

    python -m p1e_energy_field.page --data <main>/data/p1e --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict

TEMPLATE = Path(__file__).parent / "viz" / "index.html"
TRAINED = ("L1-8", "L9-16", "L17-22")

# page key -> (output dir, report section, label family); long passages, T1 + T2
GRIDS = {
    "block_whole": ("u2_block_2026-10-07", ("long",), "t12|causal|3.5"),
    "block_shared": ("u2_block_2026-10-07", ("shared_long",), "t12|causal:resid|3.5"),
    "block_tok": ("u2_block_2026-10-07", ("shared_long",), "t12|causal:r1out|3.5"),
    "mlp_tok": ("u2_attn_2026-10-07", ("long", "labels"), "t12|causal:mlp:r1out|3.5"),
    "attn_tok": ("u2_attn_2026-10-07", ("long", "labels"), "t12|causal:keys:r1out|3.5"),
    "kern_tok": ("u2_heads_2026-10-07", ("long", "labels"), "t12|kernns:keys:r1out|0.0"),
}
HEADS = ("u2_heads_2026-10-07", "kernns_h:keys_h:r1out")
U1 = "u1_field_2026-10-08"


def _section(data: Path, d: str, path) -> Dict:
    x = json.loads((data / d / "report.json").read_text())
    for p in path:
        x = x[p]
    return x


def build(data: Path) -> Dict:
    out: Dict = {}
    for key, (d, path, fam) in GRIDS.items():
        sec = _section(data, d, path)
        g = {}
        for k, v in sec["rows"].items():
            p = k.split("|")
            if "|".join(p[:3]) == fam:
                if "mean_X" not in v:
                    sys.exit(f"refuse: {key} {k} has no mean_X")
                g[p[3] + "|" + p[4]] = [v["label"], round(v["mean_X"], 4)]
        out[key] = g
        trained = collections.Counter(v[0] for k, v in g.items() if k.split("|")[1] in TRAINED)
        printed = {k: n for k, n in sec["counts"][fam].items()}
        if dict(trained) != printed:
            sys.exit(f"refuse: {key} ({fam}) counts {dict(trained)} != report's {printed}")
    steps = sorted({int(k.split("|")[0]) for k in out["block_whole"]})
    if len(steps) != 18:
        sys.exit(f"refuse: expected 18 steps, got {steps}")
    for key, g in out.items():
        miss = [f"{s}|{b}" for s in steps for b in TRAINED if f"{s}|{b}" not in g]
        if miss:
            sys.exit(f"refuse: {key} missing cells {miss[:4]}")
    heads = _section(data, HEADS[0], ("long", "heads"))[HEADS[1]]
    out["heads"] = {}
    for s in steps:
        for b in TRAINED:
            v = heads[f"{s}|{b}"]
            out["heads"][f"{s}|{b}"] = {
                "units": v["units"],
                "desc": v.get("descends", 0) + v.get("leans descends", 0),
                "asc": v.get("ascends", 0) + v.get("leans ascends", 0),
                "desc_full": v.get("descends", 0), "asc_full": v.get("ascends", 0)}
    u1 = json.loads((data / U1 / "labels.json").read_text())["long"]
    out["sweep"] = {k: round(v, 3) for k, v in u1["sweep_k_eff"].items()}
    # R² of e_i on <u_i, ū> (median over passages), β 3.5; the page's text quotes its range
    out["r2"] = {}
    for s in steps:
        for b in ("L1-8", "L9-16", "L17-23"):
            cell = u1["beside"].get(f"{s}|{b}|3.5", {})
            if "r2_mean" not in cell:
                sys.exit(f"refuse: r2_mean missing {s}|{b}")
            out["r2"][f"{s}|{b}"] = round(cell["r2_mean"], 3)
    low = sorted(k for k, v in out["r2"].items() if v < 0.645)
    if low != ["128|L1-8", "256|L1-8", "64|L1-8"]:
        sys.exit(f"refuse: the page says R² ≥ 0.65 except L1–8 at 64–256; below 0.645: {low}")
    calib = json.loads((data / U1 / "calib" / "labels.json").read_text())["long"]["3.5"]["table"]
    out["lump"] = {"3.5": {k: [v["label"], round(v["Xe_cal"], 4), round(v["Xe"], 4), round(v["bias"], 4)]
                           for k, v in calib.items()}}
    for s in steps:
        for b in ("L1-8", "L9-16", "L17-23"):
            if f"{s}|{b}" not in out["lump"]["3.5"]:
                sys.exit(f"refuse: lumpiness missing {s}|{b}")
            for L in ("L4", "L12", "L20"):
                for be in ("0.5", "1", "1.6", "2.5", "3.5", "5.6", "10", "20", "50", "100"):
                    if f"{s}|{L}|{be}" not in out["sweep"]:
                        sys.exit(f"refuse: sweep missing {s}|{L}|{be}")
    out["steps"] = steps
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", type=Path, required=True, help="the main tree's data/p1e")
    ap.add_argument("--out", type=Path, required=True, help="directory for index.html")
    a = ap.parse_args(argv)
    d = build(a.data)
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(d, separators=(",", ":")))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "index.html").write_text(html)
    print(f"wrote {a.out / 'index.html'} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
