"""C9 unit 3: the 160m seeds' two dates (`design-C9.md` "U3 as it runs"; D5 (a): tier 1,
description, nothing registered).

Order, each step refusing before the next:
(1) `known`: seed 0's rerun (`c9_u3_pass`) against C2 3b's 160m records: ids equal, e1, e1c, e1r,
    cos within `c2_aniso.ENERGY_TOL`; V_u and every arm's V_c equal to 3b's; dates E₁ 256 and
    centred 1000 on all four arms; 3b's gate passed every arm.
(2) the gate on seed 0's series (A5's, moved to 160m's date): the planted-null table (inside A5's
    window exactly when planted at 256 or 512), the planted effect (moved to 1000: reads 1000, not
    inside), and each arm's V_c moved one step earlier (reads 512, readable).
(3) `read`: per seed, E₁'s break and the centred dates (`c2_dose.size_reading` at 3b's arms and
    gates); U1's statistics on the ten seeds pooled, labelled by D1 (a) against U1's recorded bar.
    Beside, never read: A5's window, C2 3s's events, 3b's schedule sets, the tally against 410m.

Run (CPU):
    METS_DATA=<main>/data python -m trackC_writeup.c9_u3 known --out <dir>
    METS_DATA=<main>/data python -m trackC_writeup.c9_u3 read --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from trackA_force import a3
from trackA_force import a5_seeds as a5
from trackA_force.a1a import DATA, REPO, STEPS, V1
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_dose as cd
from trackC_writeup import c2_seeds as cs
from trackC_writeup import c9_u1 as u1
from trackC_writeup.c9_u3_pass import REF, SEEDS, rec_path

TC = DATA / "trackC"
B3_READING = TC / "c2u3b" / "2026-10-22-run" / "reading.json"
U1_RECORD = TC / "c9" / "2026-11-25-u1" / "u1.json"
SIZE = "160m"
ARMS = cd.TAU_ARMS[SIZE]
LR = cd.SIZES[SIZE][1]
WANT = {"u": 256, "c": 1000}                    # 3b's seed-0 dates (reading.json)
KEYS = ("e1", "e1c", "e1r", "cos")


class Refused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c9_u3.py", "trackC_writeup/c9_u3_pass.py", "trackC_writeup/c2_dose.py",
              "trackC_writeup/c2_seeds.py", "trackC_writeup/c2_aniso.py", "trackC_writeup/c9_u1.py",
              "trackA_force/a5_seeds.py", "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def load_dir(d: Path, seed: Optional[int] = None) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in STEPS:
        for p in V1:
            f = d / f"step{s}_{p}.json"
            if not f.exists():
                raise Refused(f"{f}: missing")
            r = json.loads(f.read_text())
            if seed is not None and r.get("seed") != seed:
                raise Refused(f"{f}: seed {r.get('seed')}, not {seed}")
            if len(r["e1"]) != 13 or any(len(r[k]) != 13 for k in KEYS):
                raise Refused(f"{f}: not 13 states")
            recs[(s, p)] = r
    return recs


def arm_key(t: float) -> str:
    return f"{t:.3g}"


def series(recs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    return c.series(recs, "e1"), {arm_key(t): c.series(recs, "e1c", t) for t in ARMS}


def b3_gates() -> Dict[str, bool]:
    arms = json.loads(B3_READING.read_text())["per_size"][SIZE]["centred"]["arms"]
    g = {k: bool(v["gated"]) for k, v in arms.items()}
    if sorted(g) != sorted(arm_key(t) for t in ARMS):
        raise Refused(f"3b's arms {sorted(g)} are not TAU_ARMS {ARMS}")
    return g


# ---------------------------------------------------------------- (1) the known answer

def known(out: Path) -> dict:
    ours, ref = load_dir(rec_path(out, 0, 0, V1[0]).parent, 0), load_dir(REF)
    ids_bad = [f"{s}|{p}" for (s, p), r in ours.items() if r["ids_sha"] != ref[(s, p)]["ids_sha"]]
    dev = {k: max(float(np.max(np.abs(np.subtract(r[k], ref[sp][k])) / np.maximum(np.abs(ref[sp][k]), 1e-12)))
                  for sp, r in ours.items()) for k in KEYS}
    vu, vcs = series(ours)
    vu_ref, vcs_ref = series(ref)
    gates = b3_gates()
    rd = cd.size_reading(vu, vcs, gates, LR)
    checks = {"ids_equal": not ids_bad, "within_tol": all(v <= c.ENERGY_TOL for v in dev.values()),
              "V_u_equal": bool(np.array_equal(vu, vu_ref)),
              "V_c_equal": all(np.array_equal(vcs[k], vcs_ref[k]) for k in vcs),
              "u_date": rd["uncentred"]["step"] == WANT["u"],
              "c_dates": [a["step"] for a in rd["centred"]["arms"].values()] == [WANT["c"]] * len(ARMS),
              "b3_gates_pass": all(gates.values())}
    return {"pass": all(checks.values()), "checks": checks, "ids_mismatch": ids_bad, "max_rel_dev": dev,
            "tol": c.ENERGY_TOL, "V_u": vu.tolist(), "V_c": {k: v.tolist() for k, v in vcs.items()},
            "reading": rd}


# ---------------------------------------------------------------- (2) the gate on seed 0's series

def gate(vu: np.ndarray, vcs: Dict[str, np.ndarray]) -> dict:
    i0 = a3.d4a(vu)["index"]
    table = []
    for s in a5.NULL_STEPS:
        r = a5.break_reading(a5.shift(vu, STEPS.index(s) - i0))
        table.append({"planted": s, "reads": r["step"], "inside": r["inside"]})
    null_ok = all(t["inside"] == (t["planted"] in a5.WINDOW) for t in table)
    eff = a5.break_reading(a5.shift(vu, STEPS.index(1000) - i0))
    eff_ok = eff["step"] == 1000 and not eff["inside"]
    moved = {}
    for k, v in vcs.items():                    # one step earlier only (design: "later" dropped)
        e = cd.date_reading(a5.shift(v, -1))
        moved[k] = {"earlier": e["step"], "pass": e["step"] == 512 and not e["unreadable"]}
    ok = null_ok and eff_ok and all(m["pass"] for m in moved.values())
    return {"pass": bool(ok), "planted_null": {"pass": null_ok, "table": table},
            "planted_effect": {"pass": bool(eff_ok), "reading": eff}, "centred_moved": moved}


# ---------------------------------------------------------------- (3) the reading

def u1_stats(out: Path) -> dict:
    rec = json.loads(U1_RECORD.read_text())
    m410, bar = rec["known_answer"]["M410"], rec["bar"]
    if abs((m410 - u1.MARGIN) - bar) > 1e-15:
        raise Refused("U1's bar is not its M410 − margin")
    runs = {f"seed{s}": u1.load_run(rec_path(out, s, 0, V1[0]).parent) for s in SEEDS}
    if {tuple(st) for st, _ in runs.values()} != {tuple(STEPS)}:
        raise Refused("a 160m seed is not on Pythia's 18 steps")
    plants = {k: u1.read_model(runs, k)["pooled"] for k in ("mean", "R")}
    why = []
    if plants["mean"]["M"] != 1.0 or plants["mean"]["S_any"] != 1.0:
        why.append(f"planted mean reads M {plants['mean']['M']}, S_any {plants['mean']['S_any']}")
    if plants["R"]["M"] != 0.0:
        why.append(f"planted R reads M {plants['R']['M']}")
    real = u1.read_model(runs)
    if real["rest_min"] < -1e-9:
        why.append(f"R < 0 ({real['rest_min']:.2e})")
    p = real["pooled"]
    return {"M410": m410, "bar": bar, "u1_record": str(U1_RECORD.relative_to(DATA)), "plants": plants,
            "refused": why, "M": p["M"], "S_any": p["S_any"],
            "label": "refused" if why else u1.label(p["M"], m410),
            "beside": {"pooled": p, "per_run": real["per_run"], "rest_min": real["rest_min"],
                       "u1_seed0": rec["models"]["pythia-160m"]["M"]}}


def read(out: Path) -> int:
    ka = known(out)
    (out / "known.json").write_text(json.dumps({"code": code_sha(), **ka}, indent=1) + "\n")
    if not ka["pass"]:
        raise Refused(f"known answer fails: {ka['checks']}")
    g = gate(np.asarray(ka["V_u"]), {k: np.asarray(v) for k, v in ka["V_c"].items()})
    (out / "gate.json").write_text(json.dumps({"code": code_sha(), **g}, indent=1) + "\n")
    if not g["pass"]:
        raise Refused("the gate on seed 0's series fails")
    gates = b3_gates()
    per, codes = {}, {}
    for s in SEEDS:
        recs = load_dir(rec_path(out, s, 0, V1[0]).parent, s)
        codes[s] = sorted({r["code"] for r in recs.values()})
        vu, vcs = series(recs)
        rd = cd.size_reading(vu, vcs, gates, LR)
        try:
            ev = cs.seed_reading(vu, vcs)
            ev = {k: ev[k] for k in ("event1", "event2", "unreadable")}
        except cs.SeedsRefused as e:
            ev = {"refused": str(e)}
        brk = a3.d4a(vu)
        per[s] = {"E1": rd["uncentred"]["step"], "E1c": rd["centred"]["dates"],
                  "E1c_agree": rd["centred"]["agree"], "reading": rd,
                  "beside": {"inside_A5_window": a5.inside(brk), "c2_3s_events": ev,
                             "V_u": vu.tolist(), "V_c": {k: v.tolist() for k, v in vcs.items()}}}
    if len({tuple(v) for v in codes.values()}) != 1:
        raise Refused(f"records from more than one producer version: {codes}")
    u1s = u1_stats(out)
    b410 = u1.breaks()["pythia-410m"]
    res = {"code": code_sha(), "record_code": codes[0], "design": 'trackC_writeup/design-C9.md "U3 as it runs"',
           "decided": "D5 (a), the user's, 2026-11-26: tier 1, nothing registered",
           "tau_arms": list(ARMS), "b3_gates": gates, "per_seed": per, "u1": u1s,
           "beside": {"tally_E1": dict(sorted(Counter(str(r["E1"]) for r in per.values()).items())),
                      "tally_E1c": dict(sorted(Counter(str(r["E1c"]) for r in per.values()).items())),
                      "tally_410m_E1": dict(sorted(Counter(str(v) for v in b410.values()).items())),
                      "inside_A5_window": sum(bool(r["beside"]["inside_A5_window"]) for r in per.values()),
                      "label": "none (D5 (a))"}}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    for s, r in per.items():
        print(f"seed {s}: E1 {r['E1']}  E1c {r['E1c']}  agree {r['E1c_agree']}  "
              f"inside {r['beside']['inside_A5_window']}  events {r['beside']['c2_3s_events']}")
    print(f"U1 on 160m x10: M {u1s['M']:.4f} ({u1s['label']}, bar {u1s['bar']:.4f}); S_any {u1s['S_any']:.4f}")
    print(json.dumps(res["beside"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("known", "read"):
        sub.add_parser(n).add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "known":
        ka = known(out)
        (out / "known.json").write_text(json.dumps({"code": code_sha(), **ka}, indent=1) + "\n")
        g = gate(np.asarray(ka["V_u"]), {k: np.asarray(v) for k, v in ka["V_c"].items()})
        print(json.dumps({"known": ka["checks"], "max_rel_dev": ka["max_rel_dev"], "gate": g["pass"],
                          "null": g["planted_null"]["pass"], "effect": g["planted_effect"]["pass"],
                          "moved": {k: v["pass"] for k, v in g["centred_moved"].items()}}))
        return 0 if ka["pass"] and g["pass"] else 1
    return read(out)


if __name__ == "__main__":
    raise SystemExit(main())
