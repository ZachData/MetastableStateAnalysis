"""U1 beside: the numbers `status-1e.md` "Blocked 30 (b) and (c)" quotes outside the labels.

Reads one U1 output (after its audit and report): the wells table (median over passages of the
band mean of `k` / `k_eff` / largest, long passages, primary targets and β), AMI against token
class and position, and the c3x purity on v1 (trained cells with c3x groups and `k2 >= 2`).
With ``--against`` it also compares the two runs cell by cell: the well count against a chance
pairing (each cell with another passage's at the same step and layer) and the token partitions
(AMI). Both runs must pass their audit (`u1_audit.audit_passes`); a missing value refuses. Reads
only.

    python -m p1e_energy_field.u1_beside_read --data <main>/data/p1e u1_rawcentred_2026-10-08 \
        --against u1_centred_2026-10-08
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from sklearn.metrics import adjusted_mutual_info_score as ami

from .u1_audit import audit_passes
from .u1_report import BANDS, load

STEPS = (0, 128, 1000, 8000, 143000)
SHOWN = ("L1-8", "L9-16", "L17-23", "L0")
# the 2026-10-06 probe's cells (`tools/run/p10_phi_wells_probe.py` defaults)
PROBE_PASSAGES, PROBE_LAYERS = ("homer_iliad", "wiki_paragraph"), (4, 12, 20)


def cells(out: Path, kind: str, targets: str, beta: float) -> Dict[Tuple, Dict]:
    return {(c["step"], c["passage"], c["layer"]): c for c in load(out, kind)
            if c["targets"] == targets and c["beta"] == beta and not c.get("sweep")}


def need(d: Dict, key: str, where: str) -> float:
    if d.get(key) is None:
        raise SystemExit(f"refusing: {where} lacks {key}")
    return d[key]


def read(out: Path) -> None:
    audit_passes(out)
    lab = out / "labels.json"
    if not lab.exists():
        raise SystemExit(f"refusing: {out} has no labels.json (run the report first)")
    labels = json.loads(lab.read_text())
    beta = labels["long"]["opts"]["primary_beta"]
    b = labels["long"]["beside"]
    print(f"{out.name}: frame {labels['long']['opts'].get('frame')}, β {beta:g}")
    print("wells (long, t12; median over passages of the band mean): k / k_eff / largest")
    for s in STEPS:
        row = [(f"{s}|{band}|{beta:g}", b.get(f"{s}|{band}|{beta:g}", {})) for band in SHOWN]
        print(f"{s:>7}  " + "  |  ".join(
            f"{need(d, 'k', w):.0f} / {need(d, 'k_eff', w):.1f} / {need(d, 'largest', w):.2f}"
            for w, d in row))
    print("AMI class / position (long, t12):")
    for s in STEPS:
        row = [(f"{s}|{band}|{beta:g}", b.get(f"{s}|{band}|{beta:g}", {})) for band in SHOWN[:3]]
        print(f"{s:>7}  " + "  |  ".join(
            f"{need(d, 'ami_cls', w):.2f} / {need(d, 'ami_pos', w):.2f}" for w, d in row))
    v1c = cells(out, "v1", "r0", beta)
    hi = max(json.loads(lab.read_text())["v1"]["opts"]["betas"])
    v1hi = cells(out, "v1", "r0", hi)
    print(f"the 2026-10-06 probe's cells (v1, r0): k (k_eff) at β {beta:g} / k at β {hi:g}, of n")
    for s in (0, 143000):
        for p in PROBE_PASSAGES:
            print(f"  {s:>6} {p:<15} " + "  ".join(
                f"L{L}: {need(v1c.get((s, p, L), {}), 'k', f'{s} {p} L{L}')} "
                f"({v1c[(s, p, L)]['k_eff']:.1f}) / {need(v1hi.get((s, p, L), {}), 'k', f'{s} {p} L{L} β {hi:g}')}"
                f" of {v1c[(s, p, L)]['n']}" for L in PROBE_LAYERS))
    allv1 = v1c.values()
    v1 = [c for c in allv1 if c["step"] > 0 and c.get("purity") is not None and c.get("k2", 0) >= 2]
    floor = [c for c in allv1 if c["step"] == 0 and c.get("purity") is not None and c.get("k2", 0) >= 2]
    print(f"c3x purity (v1, r0, trained, c3x groups, k2 >= 2): {len(v1)} cells "
          f"(step 0: {len(floor)} such cells, so no floor)" if not floor else
          f"c3x purity (v1, r0, trained, c3x groups, k2 >= 2): {len(v1)} cells; step 0 {len(floor)}")
    for band, layers in BANDS.items():
        cs = [c for c in v1 if c["layer"] in layers]
        if cs:
            ex = [c["purity"] - c["purity_null"] for c in cs]
            print(f"  {band}: n {len(cs)}, median excess {np.median(ex):+.2f}, "
                  f"p <= 0.05 in {np.mean([c['purity_p'] <= 0.05 for c in cs]):.0%}")


def partitions(out: Path, beta: float) -> Dict[Tuple, Tuple[np.ndarray, np.ndarray]]:
    """``{(step, passage, layer): (positions, well labels)}`` from the long records' ``.npz``."""
    res = {}
    for p in sorted((out / "records" / "long").glob("*.npz")):
        step, passage = p.stem[len("step"):].split("_", 1)
        z = np.load(p)
        w = z[f"t12/wells_{beta:g}"]
        for layer in range(w.shape[0]):
            res[(int(step), passage, layer)] = (z["t12/positions"], w[layer])
    return res


def compare(a_dir: Path, b_dir: Path) -> None:
    audit_passes(b_dir)
    beta = json.loads((a_dir / "labels.json").read_text())["long"]["opts"]["primary_beta"]
    a, b = cells(a_dir, "long", "t12", beta), cells(b_dir, "long", "t12", beta)
    if set(a) != set(b):
        raise SystemExit(f"refusing: the runs' cells differ ({len(a)} against {len(b)})")
    pa, pb = partitions(a_dir, beta), partitions(b_dir, beta)
    passages = sorted({k[1] for k in a})
    print(f"{a_dir.name} against {b_dir.name} (long, t12, β {beta:g}):")
    for name, keep in (("all", lambda k: True), ("step 0", lambda k: k[0] == 0),
                       ("trained, L1-23", lambda k: k[0] > 0 and k[2] > 0)):
        ks = [k for k in a if keep(k)]
        dk = np.array([abs(a[k]["k"] - b[k]["k"]) for k in ks])
        de = np.median([abs(a[k]["k_eff"] - b[k]["k_eff"]) for k in ks])
        # chance: the same cell paired with the next passage's (same step and layer)
        nxt = {p: passages[(i + 1) % len(passages)] for i, p in enumerate(passages)}
        ck = np.array([abs(a[k]["k"] - b[(k[0], nxt[k[1]], k[2])]["k"]) for k in ks])
        am = []
        for k in ks:
            (qa, wa), (qb, wb) = pa[k], pb[k]
            if not np.array_equal(qa, qb):
                raise SystemExit(f"refusing: {k} has different target positions in the two runs")
            am.append(ami(wa, wb) if len(set(wa)) > 1 or len(set(wb)) > 1 else 1.0)
        am = np.array(am)
        print(f"  {name}: {len(ks)} cells, same k {np.mean(dk == 0):.0%} (chance pairing "
              f"{np.mean(ck == 0):.0%}), within 1 {np.mean(dk <= 1):.0%} ({np.mean(ck <= 1):.0%}), "
              f"median |dk_eff| {de:.2f}; partition AMI median {np.median(am):.2f}, "
              f"quartiles {np.quantile(am, 0.25):.2f} / {np.quantile(am, 0.75):.2f}, "
              f"below 0.4 {np.mean(am < 0.4):.0%}")


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
