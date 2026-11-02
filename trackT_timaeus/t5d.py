"""
trackT_timaeus/t5d.py — T5 U2b: T2's planted gates in Dodecahedron's trough
(`trackT_timaeus/design-T5.md` "U2b", fixed 2026-11-01 before any U2b output; Blocked 126 (a)).

The records are `t2.py preflight --model 32m --trained-step <s>` at s ∈ TROUGH and at 9536 (the
known answer against T2 unit 1's `preflight_plants.json`). This module only reads them: the gate
first (refuses the reading unless 9536 reproduces unit 1), then per step and arm T2's full gate
and its trained part P = (A share ≤ 0.1 and B share ≥ 0.4), the label, and the uncentred plant-A
share at 1e-3 beside.

Tier 1, description (nothing here is registered).
    METS_DATA=<main>/data python -m trackT_timaeus.t5d read --pre <dir> --ref <unit1 preflight dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from trackC_writeup import c2_seeds as cs

TROUGH = (491, 671, 917)
SCALE_STEP = 9536
KNOWN_TOL = 1e-12          # placed: the known answer is the same code path, so exact up to float noise
UNC_PYTHIA = 0.79      # the lowest uncentred plant-A share at 1e-3 of 70m, 160m, GPT-2 small (T2 unit 1)
UNC_AS_9536 = 0.10


class U2bRefused(RuntimeError):
    pass


def _numbers(x, path=""):
    """Every number in a nested record, by path (booleans as 0/1)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _numbers(v, f"{path}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _numbers(v, f"{path}/{i}")
    elif isinstance(x, (bool, int, float)):
        yield path, float(x)
    else:
        yield path, x


def known_answer(new: dict, ref: dict, tol: float = KNOWN_TOL) -> dict:
    """``derived``, ``arms`` and ``gate`` of a re-run at 9536 against unit 1's record."""
    worst, mism = 0.0, []
    for key in ("derived", "arms", "gate"):
        a, b = dict(_numbers(new[key])), dict(_numbers(ref[key]))
        if a.keys() != b.keys():
            mism.append(f"{key}: keys differ")
            continue
        for p in a:
            if isinstance(a[p], float) and isinstance(b[p], float):
                worst = max(worst, abs(a[p] - b[p]))
            elif a[p] != b[p]:
                mism.append(f"{key}{p}: {a[p]!r} != {b[p]!r}")
    return {"max_abs_diff": worst, "mismatches": mism, "pass": worst <= tol and not mism}


def trained_part(arm: dict) -> bool:
    return arm["A_share"] <= cs.A_LEAK and arm["B_share"] >= cs.B_SEEN


def step_reading(rec: dict) -> dict:
    arms = {k: {"null_rate": v["null_rate"], "B_share": v["B_share"], "A_share": v["A_share"],
                "full_gate": bool(v["pass"]), "trained_part": trained_part(v)}
            for k, v in rec["gate"]["arms"].items()}
    a_u = rec["uncentred_at_1e-3"]["A_share"]
    unc = "as on Pythia" if a_u >= UNC_PYTHIA else "as at 9536" if a_u <= UNC_AS_9536 else "between"
    return {"b_dose": rec["gate"]["b_dose"], "derived": rec["derived"], "arms": arms,
            "any_trained_part": any(a["trained_part"] for a in arms.values()),
            "any_full_gate": any(a["full_gate"] for a in arms.values()),
            "uncentred_beside": {"A_share": a_u, "null_rate": rec["uncentred_at_1e-3"]["null_rate"],
                                 "base_rate": rec["uncentred_at_1e-3"]["base_rate"], "reads": unc},
            "loss_mean_beside": rec["checks"][f"step{rec['trained_step']}"]["loss_mean"]}


def label(per_step: Dict[int, dict]) -> str:
    hits = [per_step[s]["any_trained_part"] for s in sorted(per_step)]
    if all(hits):
        return "passes in the trough"
    if not any(hits):
        return "fails in the trough"
    return "mixed"


def load(pre: Path, s: int) -> dict:
    d = pre / ("32m" if s == SCALE_STEP else f"32m-step{s}")
    rec = json.loads((d / "preflight_plants.json").read_text())
    if int(rec.get("trained_step", SCALE_STEP)) != s:
        raise U2bRefused(f"{d}: trained_step {rec.get('trained_step')} is not {s}")
    return rec


def read(a) -> int:
    pre, ref_dir = Path(a.pre), Path(a.ref)
    ref = json.loads((ref_dir / "32m" / "preflight_plants.json").read_text())
    ka = known_answer(load(pre, SCALE_STEP), ref)
    res: Dict[str, object] = {"design": "trackT_timaeus/design-T5.md U2b", "gate": {"known_answer": ka}}
    if not ka["pass"]:
        res["refused"] = "the known answer at 9536 does not reproduce T2 unit 1"
    else:
        per = {s: step_reading(load(pre, s)) for s in TROUGH}
        codes = {s: load(pre, s)["code"] for s in (*TROUGH, SCALE_STEP)}
        res.update({"record_codes": codes, "per_step": per, "label": label(per),
                    "full_gate_passes_anywhere": any(p["any_full_gate"] for p in per.values())})
    (pre / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if ka["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("read")
    sp.add_argument("--pre", required=True)
    sp.add_argument("--ref", required=True)
    a = ap.parse_args(argv)
    return {"read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
