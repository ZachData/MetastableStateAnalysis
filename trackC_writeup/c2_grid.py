"""The layer × step small multiples' record (the user's, 2026-10-29; figure 13 in `numbers.py
paper-figures`). Description only: nothing registered, no threshold, no date, no forward pass.

Four quantities on seed 0's 18 steps, each at every depth, from the Phase 1 and Phase 2 runs in
`data/phase12/` (`a1a.RUNS`, `P2_RUNS`):

- **effective rank** of the unit tokens (`geometry.json` `effective_rank_normed`, paper E5's
  quantity), at the 25 states, mean over the 7 prompts;
- **mass near 1**, the fraction of token pairs with cosine above 0.9 (`ip_mass_near_1`), at the 25
  states, mean over prompts; at state 0 duplicate tokens pin it to a prompt's floor
  (`design-1.md` "Mass near 1: the floor problem");
- **energy change**, (E₁(t) − E₁(t−1)) / E₁(t−1) at the 24 transitions (`energies.json`, β = 1),
  mean over prompts; A5's drop rule is this below −REL_TOL;
- **Schur ratio**, log₁₀ ‖T₊₊‖ / ‖T₋₋‖ of each block's OV map (`ov_summary` `schur_cond`): the
  weights alone, no prompt. A rough degeneracy flag, not a condition number (`math-2.md` §3.2).

**First, the check that these runs are A5's** (refuses on any failure, before any grid value
is read): at every step and prompt, `tokens.txt` equals A5's seed-0 record's tokens; E₁ at each
of the 25 states equals the record's `e1` to A5's ENERGY_TOL (relative); `layer_metrics.csv`'s
`energy_beta1.0` equals `energies.json`'s exactly; and V counted from these energies by A5's rule
equals A5's gate known answer at every step. Beside, also refusing: the prompt mean of each
prompt's minimum effective rank over states reproduces A3's E5 series (rtol 5e-4, as A3's own).

Run: METS_DATA=<main>/data python -m trackC_writeup.c2_grid
"""
from __future__ import annotations

import csv
import hashlib
import json
from typing import Dict, List

import numpy as np

from tools.run.behavioural import _tokens_txt
from trackA_force.a1a import DATA, REPO, RUNS, STEPS, V1, run_dir
from trackA_force.a5_seeds import ENERGY_TOL
from trackC_writeup.c2_readings import drops

RECS = DATA / "trackA" / "a5" / "2026-10-20-u1" / "seed0" / "records"
GATE0 = DATA / "trackA" / "a5" / "2026-10-20-u1" / "seed0" / "gate.json"
A3_ROWS = DATA / "trackA" / "a3" / "2026-10-20-A3u5" / "a3_rows.json"
OUT = DATA / "trackC" / "c2grid" / "2026-10-29"
#: The one Phase 2 directory holding each step's `ov_summary` (checked: one per step).
P2_RUNS = {0: "2026-09-01_14-51-27", 2: "2026-08-31_11-56-26", 4: "2026-08-31_12-31-11",
           8: "2026-08-31_13-06-28", 16: "2026-08-31_13-41-44", 32: "2026-08-31_14-17-04",
           64: "2026-08-31_14-52-05", 128: "2026-08-31_15-27-59", 256: "2026-08-31_16-05-07",
           512: "2026-08-31_16-43-54", 1000: "2026-08-31_17-24-43", 2000: "2026-09-01_15-15-54",
           4000: "2026-09-01_15-37-37", 8000: "2026-09-01_16-26-34", 16000: "2026-09-01_17-16-09",
           32000: "2026-09-01_18-03-40", 54000: "2026-08-31_18-14-44", 143000: "2026-09-01_18-53-22"}
STATES, BLOCKS = 25, 24
E5_RTOL = 5e-4                                  # placed: A3's own (`a3.rows_e1_e5`): the reports print 2 decimals


class GridRefused(SystemExit):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c2_grid.py", "trackC_writeup/c2_readings.py", "trackA_force/a1a.py",
              "trackA_force/a5_seeds.py", "tools/run/behavioural.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def p2_summary(step: int):
    return DATA / "phase12" / f"p2_eigenspectra_{P2_RUNS[step]}" / f"ov_summary_pythia-410m-step{step}.json"


def inputs_sha() -> str:
    """sha256 over every file read, in a fixed order (A5's gate and A3's rows, A5's records, then each
    run's files)."""
    h = hashlib.sha256()
    for f in [GATE0, A3_ROWS] + sorted(RECS.glob("step*_*.json")):
        h.update(f.name.encode() + f.read_bytes())
    for s in STEPS:
        for p in V1:
            for n in ("tokens.txt", "energies.json", "layer_metrics.csv", "geometry.json"):
                h.update((run_dir(s, p) / n).read_bytes())
        h.update(p2_summary(s).read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- the check against A5

def check_a5() -> dict:
    v_known = json.loads(GATE0.read_text())["known_answer"]["V"]
    tok_bad, csv_bad, e_rel, v = [], [], 0.0, []
    for s in STEPS:
        n = 0
        for p in V1:
            d, rec = run_dir(s, p), json.loads((RECS / f"step{s}_{p}.json").read_text())
            if rec["tokens"] != _tokens_txt(d):
                tok_bad.append(f"{s}|{p}")
            e = np.array([l["energies"]["1.0"] for l in json.loads((d / "energies.json").read_text())["layers"]])
            lm = np.array([float(r["energy_beta1.0"]) for r in csv.DictReader((d / "layer_metrics.csv").open())])
            if len(e) != STATES or not np.array_equal(e, lm):
                csv_bad.append(f"{s}|{p}")
            e_rel = max(e_rel, float(np.max(np.abs(e - np.asarray(rec["e1"])) / np.abs(e))))
            n += int(drops(e).sum())
        v.append(n)
    out = {"tokens_mismatch": tok_bad, "csv_mismatch": csv_bad, "energy_max_rel_diff": e_rel,
           "energy_tol": ENERGY_TOL, "V": v, "V_known": list(v_known), "V_equal": v == list(v_known)}
    out["pass"] = not tok_bad and not csv_bad and e_rel <= ENERGY_TOL and out["V_equal"]
    if not out["pass"]:
        raise GridRefused(f"refusing: data/phase12 does not reproduce A5's seed-0 energies: {out}")
    return out


# ---------------------------------------------------------------- the grid

def grid() -> dict:
    er, mass, de, schur, minrank = {}, {}, {}, {}, []
    for s in STEPS:
        ers, ms, des = [], [], []
        for p in V1:
            d = run_dir(s, p)
            g = json.loads((d / "geometry.json").read_text())["layers"]
            e = np.array([l["energies"]["1.0"] for l in json.loads((d / "energies.json").read_text())["layers"]])
            if len(g) != STATES:
                raise GridRefused(f"refusing: {d}/geometry.json has {len(g)} states, not {STATES}")
            ers.append([float(l["effective_rank_normed"]) for l in g])
            ms.append([float(l["ip_mass_near_1"]) for l in g])
            des.append((np.diff(e) / e[:-1]).tolist())
        for x in (ers, ms, des):
            if not np.all(np.isfinite(np.asarray(x, float))):
                raise GridRefused(f"refusing: a non-finite value at step {s}")
        minrank.append(float(np.mean([min(r) for r in ers])))
        er[str(s)] = np.mean(ers, axis=0).tolist()
        mass[str(s)] = np.mean(ms, axis=0).tolist()
        de[str(s)] = np.mean(des, axis=0).tolist()
        lay = json.loads(p2_summary(s).read_text())["layers"]
        c = np.array([lay[f"layer_{b}"]["schur_cond"] for b in range(BLOCKS)], float)
        if len(lay) != BLOCKS or not np.all(np.isfinite(c) & (c > 0)):
            raise GridRefused(f"refusing: step {s}'s schur_cond is not 24 finite positive values")
        schur[str(s)] = np.log10(c).tolist()
    e5 = json.loads(A3_ROWS.read_text())["rows"]["E5"]["series"]
    if not np.allclose(minrank, e5, rtol=E5_RTOL, atol=0):
        raise GridRefused(f"refusing: the minimum effective rank {minrank} does not reproduce A3's E5 {e5}")
    return {"eff_rank": er, "mass_near_1": mass, "energy_change": de, "log10_schur": schur,
            "e5_check": {"min_rank_prompt_mean": minrank, "a3_E5": e5, "rtol": E5_RTOL,
                         "max_rel_diff": float(np.max(np.abs(np.subtract(minrank, e5)) / np.abs(e5)))}}


def main() -> int:
    a5 = check_a5()
    print(f"A5 check passes: tokens equal, max relative dE {a5['energy_max_rel_diff']:.1e} "
          f"(tol {ENERGY_TOL:g}), V equal at all {len(STEPS)} steps", flush=True)
    g = grid()
    print(f"E5 check passes: max relative diff {g['e5_check']['max_rel_diff']:.1e}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"code": code_sha(), "inputs_sha256": inputs_sha(), "steps": list(STEPS), "prompts": list(V1),
           "phase1_runs": {str(s): r for s, r in RUNS.items()},
           "phase2_runs": {str(s): r for s, r in P2_RUNS.items()}, "a5_check": a5, **g}
    (OUT / "grid.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {OUT / 'grid.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
