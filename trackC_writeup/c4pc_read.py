"""
trackC_writeup/c4pc_read.py — C4′ (c): the drivers around the five rules (`c4pc_rules.py`, whose
hash the entries register): the gates for L7–L9h (U1), and, **added in U3**, the records' and
heads' reads on the new arms, L10's K wiring, the readings M1–M4 and the scoring. This file may
grow without moving the registered rules' hash (`/challenge-pr` on #362, finding 1).

**Gates** (`gate`, U1, no forward pass): for L7–L9h, the planted effect is C4's pair (F, A0) on
the dense grid (must read "holds"), the planted null is (released `pythia-70m`, A0) on Pythia's
13 steps ≤ 4000 (must be readable and not hold), and the seed null is each `pythia-70m-seed1..9`
against the released run on the same 13 steps (≤ 1 of 9 may hold). The gated τ arms for every
gate pair are C4's (`data/trackC/c4/2026-11-12-u2/gate.json`: 9.61e-5, 1e-4, 3e-4); V is read.
An entry whose gate fails is reported as description and scores nothing. L10 (b)'s gate
(`c4pc_rules.l10b_gate`) runs in U3, since it needs A0′. Beside (`beside.c4_pair_coarse`): C4's
own pair re-read on the nulls' 13 steps, to show the rules can fire at that resolution.

**Refusal** (`require_scorable`, to be called by U3's scoring): no entry is scored before
`c4pc_train.registration()` has every id registered or declined, and before `gate.json` exists
with the rules' hash.

    METS_DATA=<main>/data python -m trackC_writeup.c4pc_read gate --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, V1
from trackC_writeup import c4_close as cc
from trackC_writeup import c4_read as cr
from trackC_writeup import c4_train as ct
from trackC_writeup import c4pc_rules as rl
from trackC_writeup import c4pc_train as pt
from trackC_writeup import c6_warmup as cw

GRID = tuple(ct.GRID)
PYTHIA = tuple(cr.PYTHIA)
# the gates' inputs (all main-tree records, read only)
C4_U2 = DATA / "trackC/c4/2026-11-12-u2"            # A0 and F records, gate.json
C4_CLOSE = DATA / "trackC/c4/2026-11-14-close"      # heads_{A0,F,released}.json
SEED_RECS = DATA / "trackC/c6/2026-10-31-u0-seeds"  # seed0 (pythia-70m) … seed9 records
SEED_HEADS = DATA / "trackC/c4p/2026-11-15-u1"      # heads_seed{0..9}.json
SEEDS = tuple(f"seed{k}" for k in range(1, 10))


class C4pcReadRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4pc_read.py",):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- a pair's readings from records

def pair_inputs(recs: Dict[Tuple[int, str], dict], heads: Dict[str, list], grid: Sequence[int],
                arms: Sequence[str]) -> dict:
    """One side's matrices, V's per-transition counts, dates and induction from records and heads."""
    mats = {k: cc.per_transition(recs, k, grid) for k in arms}
    u = cc.per_transition(recs, "u", grid)
    cen = {k: cr.date_on(cr.series(recs, k, grid), grid) for k in arms}
    top = [float(np.max(heads[str(s)])) for s in grid]
    return {"mats": mats, "u": u, "cen": cen, "ind": cr.date_on(top, grid),
            "sc4000": np.asarray(heads[str(ct.HORIZON)], float)}


def read_pair(f: dict, a: dict, grid: Sequence[int], arms: Sequence[str]) -> dict:
    return {"L7": rl.l7(f["mats"], a["mats"], grid, arms), "L8": rl.l8(f["u"], a["u"], grid),
            "L9": rl.l9(f["ind"], a["ind"], f["cen"], a["cen"], arms), "L9h": rl.l9h(f["sc4000"], a["sc4000"])}


def _load_recs(path_of, steps) -> Dict[Tuple[int, str], dict]:
    out = {}
    for s in steps:
        for p in V1:
            f = path_of(s, p)
            if not f.exists():
                raise C4pcReadRefused(f"{f}: missing")
            out[(s, p)] = json.loads(f.read_text())
    return out


def _heads(f: Path) -> Dict[str, list]:
    return json.loads(f.read_text())["per_step"]


def c4_arms() -> Tuple[str, ...]:
    g = json.loads((C4_U2 / "gate.json").read_text())
    if not g["u_read"]:
        raise C4pcReadRefused("C4's gate did not read V")
    return tuple(k for k in cr.KEYS[1:] if g["arms_gated"][k])


def _c4_side(arm: str, grid, arms) -> dict:
    return pair_inputs(_load_recs(lambda s, p: cr.rec_path(C4_U2, arm, s, p), grid),
                       _heads(C4_CLOSE / f"heads_{arm}.json"), grid, arms)


def gate(a) -> int:
    out = Path(a.out)
    arms = c4_arms()
    side = {arm: _c4_side(arm, GRID, arms) for arm in ("A0", "F")}
    coarse = {arm: _c4_side(arm, PYTHIA, arms) for arm in ("A0", "F")}
    seed = {}
    for nm in ("seed0",) + SEEDS:
        seed[nm] = pair_inputs(_load_recs(lambda s, p, nm=nm: cw.rec_path(SEED_RECS, nm, s, p), PYTHIA),
                               _heads(SEED_HEADS / f"heads_{nm}.json"), PYTHIA, arms)
    rel_close = pair_inputs(_load_recs(lambda s, p: cw.rec_path(SEED_RECS, "seed0", s, p), PYTHIA),
                            _heads(C4_CLOSE / "heads_released.json"), PYTHIA, arms)
    effect = read_pair(side["F"], side["A0"], GRID, arms)
    null = read_pair(seed["seed0"], coarse["A0"], PYTHIA, arms)
    seeds = {nm: read_pair(seed[nm], seed["seed0"], PYTHIA, arms) for nm in SEEDS}
    on_coarse = read_pair(coarse["F"], coarse["A0"], PYTHIA, arms)
    beside = {"null_with_c4_close_released_heads": {k: v["holds"] for k, v in
                                                    read_pair(rel_close, coarse["A0"], PYTHIA, arms).items()},
              "c4_pair_coarse": {k: v["holds"] for k, v in on_coarse.items()},
              "c4_pair_coarse_L9": {k: on_coarse["L9"].get(k) for k in ("t_F", "t_A0", "bar_i")}}
    res = {"rules": rl.code_sha(), "code": code_sha(), "arms": list(arms),
           "grids": {"effect": list(GRID), "nulls": list(PYTHIA)},
           "inputs": {"c4_u2": str(C4_U2), "c4_close": str(C4_CLOSE), "seed_recs": str(SEED_RECS),
                      "seed_heads": str(SEED_HEADS)},
           "gate": {e: rl.gate_entry(effect[e], null[e], {k: v[e] for k, v in seeds.items()}) for e in effect},
           "effect": effect, "null": null, "seeds": seeds, "beside": beside}
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(res, indent=1, default=_np) + "\n")
    for e, g in res["gate"].items():
        print(e.ljust(4), json.dumps(g))
    print(json.dumps(beside))
    return 0


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def require_scorable(out: Path) -> dict:
    """No score before every entry is registered or declined and the gate exists with the rules' hash."""
    st = pt.require_decided()
    g = out / "gate.json"
    if not g.exists():
        raise C4pcReadRefused(f"{g}: the gate first")
    gj = json.loads(g.read_text())
    if gj["rules"] != rl.code_sha():
        raise C4pcReadRefused(f"gate rules {gj['rules']} ≠ {rl.code_sha()}")
    return {"registration": st, "gate": gj["gate"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("gate")
    q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"gate": gate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
