"""
trackT_timaeus/t5e.py — T5 U3's first read: T2's planted gates along Dodecahedron's late rise
(`trackT_timaeus/design-T5.md` "U3 revised toward the late rise", fixed 2026-11-01 before any U3
output; Blocked 130 (a)).

The records are `t2.py preflight --model 32m --trained-step <s>` at the 14 grid steps in
(917, 9536) and at the two known answers, 9536 (against T2 unit 1's `preflight_plants.json`) and
917 (against U2b's). This module only reads them: the gate first (refuses the reading unless both
reproduce), then per step (i) T2's four centred arms as derived at s, the full gate and the trained
part P (as U2b, `t5d.step_reading`), (ii) the fixed-τ arms (τ × B dose, `t5d`'s sweep code, now
fixed), (iii) the uncentred plant-A share at 1e-3, binned as U2b; ρ_mid, ρ_mid⁻⁰, position 0's
norm (U2's record), the mean loss and the LR phase beside. Then where each count stops, a
description with no label, and the one-sided reading of each stop.

Tier 1, description (nothing here is registered).
    METS_DATA=<main>/data python -m trackT_timaeus.t5e read --pre <dir> \
        --ref <main>/data/trackT/t2u1/2026-10-29-preflight --ref-917 <main>/data/trackT/t5u2b/2026-11-01 \
        --u2 <main>/data/trackT/t5u2/2026-10-30/reading.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from trackT_timaeus import t5d

#: the 14 grid steps in (917, 9536), with the two known answers at the ends
INNER = (1072, 1253, 1465, 1713, 2002, 2340, 2735, 3198, 3738, 4369, 5108, 5970, 6979, 8158)
RISE = (917, *INNER, t5d.SCALE_STEP)
#: (ii)'s arms, fixed now (post hoc in U2b): `t5d`'s sweep values
TAUS = t5d.SWEEP_TAUS
DOSES = t5d.SWEEP_DOSES
#: (ii)'s stop reads one cell: U2b's passing band
STOP_TAU, STOP_DOSE = "0.003", "0.5"
WARMUP = 1000          # Dodecahedron's linear warmup (T1); cosine decay after, its floor unknown


class U3Refused(RuntimeError):
    pass


def lr_phase(s: int) -> str:
    if s < WARMUP:
        return f"warmup, {s / WARMUP:.2f}"
    return "end of decay" if s == t5d.SCALE_STEP else "decay"


def stop(series: Sequence[Tuple[int, bool]]) -> dict:
    """Where a pass/fail series stops passing: the last passing step, and the first step from
    which it fails at that step and every later one (None if the last step passes). A fail before
    that stop (the series crossing back) makes it "not monotone", with those steps."""
    steps = [s for s, _ in series]
    ok = [bool(p) for _, p in series]
    first = None
    for i in range(len(ok) - 1, -1, -1):
        if ok[i]:
            break
        first = steps[i]
    last_pass = max((s for s, p in zip(steps, ok) if p), default=None)
    early = [s for s, p in zip(steps, ok) if not p and (first is None or s < first)]
    return {"last_pass": last_pass, "stop": first, "monotone": not early,
            "fails_before_stop": early}


def one_sided(stop_step: Optional[int]) -> str:
    """design-T5.md "the one-sided reading": what a stop at each step can say."""
    if stop_step is None:
        return "does not stop"
    if stop_step <= 917:
        return "stops in the trough (outside the rule)"
    if stop_step == 1072:
        return "stops before ρ rises"
    if stop_step == 1253:
        return "at the rise's start"
    return "with the rise"


def series_beside(u2: dict) -> Dict[int, dict]:
    ser = u2["reading"]["dodecahedron-32m"]["series"]
    idx = {s: i for i, s in enumerate(ser["step"])}
    return {s: {"rho_mid": ser["rho_mid"][idx[s]], "rho_mid_no0": ser["rho_mid_no0"][idx[s]],
                "n0_mid": ser["n0_mid"][idx[s]], "lr_phase": lr_phase(s)} for s in RISE}


def fixed_tau(units: list) -> dict:
    from trackC_writeup import c2_dose as d
    return {bd: {k: {"A_share": v["A_share"], "B_share": v["B_share"], "null_rate": v["null_rate"],
                     "pass": bool(v["pass"])}
                 for k, v in d.gate_arms(units, TAUS, bd)["arms"].items()} for bd in DOSES}


def uncentred_a(units: dict) -> float:
    """(iii) from the units alone, as `t2.py preflight` computes ``uncentred_at_1e-3.A_share``
    (it does not need B's dose, so it is read at a step whose B derivation refused)."""
    tr = units["trained"]
    return t5d.cs.share(t5d.cs.rate(tr["plants"]["A_u"][t5d.cs.A_DOSE], t5d.cs.REL_TOL),
                        t5d.cs.rate(tr["base"]["u"], t5d.cs.REL_TOL))


def pass_series(per: Dict[int, dict]) -> Dict[str, List[Tuple[int, Optional[bool]]]]:
    """(i) some centred arm has P (None where T2's derivation refused at s); (ii) plant A ≤ 0.10 at
    τ 3e-3, B dose 0.5; (iii) the uncentred plant-A share ≥ 0.79 ("as on Pythia")."""
    s = sorted(per)
    return {
        "i_centred_any_P": [(k, None if "refused" in per[k]["centred"] else per[k]["centred"]["any_trained_part"])
                            for k in s],
        "ii_fixed_tau": [(k, per[k]["fixed_tau"][STOP_DOSE][STOP_TAU]["A_share"] <= t5d.cs.A_LEAK) for k in s],
        "iii_uncentred": [(k, per[k]["uncentred_A_share"] >= t5d.UNC_PYTHIA) for k in s],
    }


def stops(per: Dict[int, dict]) -> dict:
    """Where each count stops. A step whose derivation refused (only (i) can have one) is read two
    ways, fixed after the batch and before the reading (post hoc: the frozen rule did not foresee a
    refusal): counted as no arm with P, and left out; ``refused_agree`` says whether they agree."""
    out = {}
    for k, ser in pass_series(per).items():
        refused = [s for s, p in ser if p is None]
        as_fail = stop([(s, bool(p)) for s, p in ser])
        out[k] = {**as_fail, "passes": [s for s, p in ser if p], "refused": refused}
        if refused:
            left_out = stop([(s, p) for s, p in ser if p is not None])
            out[k]["refused_left_out"] = left_out
            out[k]["refused_agree"] = left_out["stop"] == as_fail["stop"]
        out[k]["reads"] = one_sided(out[k]["stop"])
    out["ii_fixed_tau"]["cell"] = {"tau": STOP_TAU, "b_dose": STOP_DOSE}
    return out


def stated_before_output(per: Dict[int, dict]) -> Dict[str, bool]:
    """917 passes and 9536 fails on each of (i)–(iii) (the records), so each crossing lies in
    (917, 9536] unless a series is not monotone."""
    return {k: bool(ser[0][1]) and not ser[-1][1] for k, ser in pass_series(per).items()}


def refusal(pre: Path, s: int) -> str:
    """The refusal a step's preflight raised (its log's last ``Refused`` line); a step with no
    plants record and no refusal is not a reading but a failed run."""
    lines = [ln for ln in (pre / f"log_{s}.txt").read_text().splitlines() if "Refused:" in ln]
    if not lines:
        raise U3Refused(f"step {s}: no preflight_plants.json and no refusal in its log")
    return lines[-1].strip()


def read(a) -> int:
    pre = Path(a.pre)
    ref = json.loads((Path(a.ref) / "32m" / "preflight_plants.json").read_text())
    ref917 = json.loads((Path(a.ref_917) / "32m-step917" / "preflight_plants.json").read_text())
    ka = {"9536": t5d.known_answer(t5d.load(pre, t5d.SCALE_STEP), ref),
          "917": t5d.known_answer(t5d.load(pre, 917), ref917)}
    res: Dict[str, object] = {"design": "trackT_timaeus/design-T5.md U3 revised toward the late rise",
                              "gate": {"known_answer": ka}}
    if not all(v["pass"] for v in ka.values()):
        res["refused"] = "a known answer (9536 against T2 unit 1, 917 against U2b) does not reproduce"
    else:
        beside = series_beside(json.loads(Path(a.u2).read_text()))
        per: Dict[int, dict] = {}
        codes, unc_check = {}, {}
        for s in RISE:
            units = json.loads((t5d._dir(pre, s) / "preflight_units.json").read_text())
            if (t5d._dir(pre, s) / "preflight_plants.json").exists():
                rec = t5d.load(pre, s)
                codes[s] = rec["code"]
                centred = t5d.step_reading(rec)
                unc_check[s] = abs(uncentred_a(units) - rec["uncentred_at_1e-3"]["A_share"])
            elif s in (917, t5d.SCALE_STEP):
                raise U3Refused(f"step {s}: no preflight_plants.json")
            else:
                centred = {"refused": refusal(pre, s)}
            per[s] = {"centred": centred, "fixed_tau": fixed_tau(units), "uncentred_A_share": uncentred_a(units),
                      "beside": beside[s]}
        worst = max(unc_check.values())
        res["gate"]["uncentred_from_units"] = {"max_abs_diff": worst, "steps": sorted(unc_check),
                                               "pass": worst <= t5d.KNOWN_TOL}
        if worst > t5d.KNOWN_TOL:
            raise U3Refused(f"(iii) from the units differs from the records by {worst}")
        st = stops(per)
        res.update({"record_codes": codes, "per_step": per, "where_each_count_stops": st,
                    "stated_before_output_held": stated_before_output(per)})
    (pre / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    _print(res)
    return 0 if "refused" not in res else 1


def _print(res: dict) -> None:
    print(json.dumps(res["gate"], indent=1))
    if "refused" in res:
        print("REFUSED:", res["refused"])
        return
    for s, p in res["per_step"].items():
        c, b = p["centred"], p["beside"]
        ft = p["fixed_tau"][STOP_DOSE][STOP_TAU]
        if "refused" in c:
            print(f"{s:>5} rho {b['rho_mid']:.3f} n0 {b['n0_mid']:5.1f} | (i) REFUSED: {c['refused']} "
                  f"| tau3e-3: A {ft['A_share']:.2f} | unc A {p['uncentred_A_share']:.2f}")
            continue
        arms = " ".join(f"{k}:{'P' if v['trained_part'] else '-'}{'G' if v['full_gate'] else ''}"
                        f"(A{v['A_share']:.2f},B{v['B_share']:.2f})" for k, v in c["arms"].items())
        print(f"{s:>5} rho {b['rho_mid']:.3f} n0 {b['n0_mid']:5.1f} loss {c['loss_mean_beside']:.2f} "
              f"| {arms} | tau3e-3: A {ft['A_share']:.2f} | unc A {p['uncentred_A_share']:.2f}")
    print(json.dumps(res["where_each_count_stops"], indent=1))
    print(json.dumps(res["stated_before_output_held"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("read")
    sp.add_argument("--pre", required=True)
    sp.add_argument("--ref", required=True, help="T2 unit 1's preflight dir (holds 32m/)")
    sp.add_argument("--ref-917", required=True, help="U2b's records dir (holds 32m-step917/)")
    sp.add_argument("--u2", required=True, help="U2's reading.json")
    a = ap.parse_args(argv)
    return {"read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
