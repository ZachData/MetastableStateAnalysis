"""U1 beside: the numbers `status-1e.md` "Blocked 30 (b) and (c)" quotes outside the labels.

Reads one U1 output (after its audit and report): the wells table (median over passages of the
band mean of `k` / `k_eff` / largest, long passages, primary targets and β), AMI against token
class and position, and the c3x purity on v1 (trained cells with c3x groups and `k2 >= 2`).
With ``--against`` it also compares the two runs' well counts cell by cell. Reads only.

    python -m p1e_energy_field.u1_beside_read --data <main>/data/p1e u1_rawcentred_2026-10-08 \
        --against u1_centred_2026-10-08
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .u1_report import BANDS, load

STEPS = (0, 128, 1000, 8000, 143000)
SHOWN = ("L1-8", "L9-16", "L17-23", "L0")


def cells(out: Path, kind: str, targets: str, beta: float) -> Dict[Tuple, Dict]:
    return {(c["step"], c["passage"], c["layer"]): c for c in load(out, kind)
            if c["targets"] == targets and c["beta"] == beta and not c.get("sweep")}


def read(out: Path) -> None:
    lab = out / "labels.json"
    if not (out / "audit.json").exists() or not lab.exists():
        raise SystemExit(f"refusing: {out} lacks audit.json or labels.json (audit, then report)")
    labels = json.loads(lab.read_text())
    beta = labels["long"]["opts"]["primary_beta"]
    b = labels["long"]["beside"]
    print(f"{out.name}: frame {labels['long']['opts'].get('frame')}, β {beta:g}")
    print("wells (long, t12; median over passages of the band mean): k / k_eff / largest")
    for s in STEPS:
        row = [b.get(f"{s}|{band}|{beta:g}", {}) for band in SHOWN]
        print(f"{s:>7}  " + "  |  ".join(
            f"{d.get('k', np.nan):.0f} / {d.get('k_eff', np.nan):.1f} / {d.get('largest', np.nan):.2f}"
            for d in row))
    print("AMI class / position (long, t12):")
    for s in STEPS:
        row = [b.get(f"{s}|{band}|{beta:g}", {}) for band in SHOWN[:3]]
        print(f"{s:>7}  " + "  |  ".join(
            f"{d.get('ami_cls', np.nan):.2f} / {d.get('ami_pos', np.nan):.2f}" for d in row))
    v1 = [c for c in cells(out, "v1", "r0", beta).values()
          if c["step"] > 0 and c.get("purity") is not None and c.get("k2", 0) >= 2]
    print(f"c3x purity (v1, r0, trained, c3x groups, k2 >= 2): {len(v1)} cells")
    for band, layers in BANDS.items():
        cs = [c for c in v1 if c["layer"] in layers]
        if cs:
            ex = [c["purity"] - c["purity_null"] for c in cs]
            print(f"  {band}: n {len(cs)}, median excess {np.median(ex):+.2f}, "
                  f"p <= 0.05 in {np.mean([c['purity_p'] <= 0.05 for c in cs]):.0%}")


def compare(a_dir: Path, b_dir: Path) -> None:
    beta = json.loads((a_dir / "labels.json").read_text())["long"]["opts"]["primary_beta"]
    a, b = cells(a_dir, "long", "t12", beta), cells(b_dir, "long", "t12", beta)
    if set(a) != set(b):
        raise SystemExit(f"refusing: the runs' cells differ ({len(a)} against {len(b)})")
    print(f"{a_dir.name} against {b_dir.name} (long, t12, β {beta:g}):")
    for name, keep in (("all", lambda k: True), ("step 0", lambda k: k[0] == 0),
                       ("trained, L1-23", lambda k: k[0] > 0 and k[2] > 0)):
        ks = [k for k in a if keep(k)]
        dk = np.array([abs(a[k]["k"] - b[k]["k"]) for k in ks])
        de = np.median([abs(a[k]["k_eff"] - b[k]["k_eff"]) for k in ks])
        print(f"  {name}: {len(ks)} cells, same k {np.mean(dk == 0):.0%}, "
              f"within 1 {np.mean(dk <= 1):.0%}, median |dk_eff| {de:.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="the data/p1e directory")
    ap.add_argument("run")
    ap.add_argument("--against", default=None)
    a = ap.parse_args(argv)
    read(a.data / a.run)
    if a.against:
        compare(a.data / a.run, a.data / a.against)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
