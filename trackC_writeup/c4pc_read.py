"""
trackC_writeup/c4pc_read.py — C4′ (c): the five registered leans' rules (L7, L8, L9, L9h, L10),
their gates, and the measured readings M1–M4 (`design-C4pc.md` "The readings"; frozen on Blocked
166 all (a), 2026-11-18). A new module, so `c4_read` (`220161b9159b`), `c4_close`, `c4p_read`
and `c4p_path`'s hashes stay.

Every rule takes a **pair** (pseudo-F′, pseudo-A0′) on a grid ending at 4000 and states the lean
against the pair's own A0′ side. Indices are 0-based transitions (T1 is state 1 → 2, as
`c4p_read.a3` prints them) and 0-based layers (L2H6 is layer 2).

- **L7** (`l7`): at every gated centred τ arm, F′ has a transient (a transition majority-violated
  at some grid step ≤ 1000, not in the 4000 majority set), A0′ none, and both 4000 sets are
  {T1, T2}. No gated arm: unreadable.
- **L8** (`l8`): V's majority sets at 1000, 2000 and 4000 all non-empty on both arms, and
  min(F′'s) = min(A0′'s) − 1 at each.
- **L9** (`l9`): induction's half-rise (`c4_read.date_on` on the top head's score) on both arms;
  (i) t_F′ < t_A0′^{3/2} / √(2W), and (ii) d_induction < d_centred at every gated τ arm, with
  d_X = |ln(t_F′ / m_cum(t_A0′))|, m_cum(t) = t² / 2W. An undated event, or an A0′ date past W
  (no flat map), is unreadable.
- **L9h** (`l9h`): the top head at 4000 by induction score: F′'s in layer 2, A0′'s in 3 or 4.
- **L10** (`l10`): (a) `c4p_path.read_pseudo`'s early window reads "another path" (F′ against
  A0′'s path); (b) r_F(s) > r_A0′(s) at ≥ ⅔ of F's 12 late steps (`l10b`).

**Gates** (`gate`, U1, no forward pass): for L7–L9h, the planted effect is C4's pair (F, A0) on
the dense grid (must read "holds"), the planted null is (released `pythia-70m`, A0) on Pythia's
13 steps ≤ 4000 (must be readable and not hold), and the seed null is each `pythia-70m-seed1..9`
against the released run on the same 13 steps (≤ 1 of 9 may hold). The gated τ arms for every
gate pair are C4's (`data/trackC/c4/2026-11-12-u2/gate.json`: 9.61e-5, 1e-4, 3e-4); V is read.
An entry whose gate fails is reported as description and scores nothing. L10 (b)'s gate
(`l10b_gate`) runs in U3, since it needs A0′.

**Refusal** (`score`): no entry is scored before `c4pc_train.registration()` has every id
registered or declined, and before `gate.json` exists with this code.

    METS_DATA=<main>/data python -m trackC_writeup.c4pc_read gate --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, V1
from trackC_writeup import c4_close as cc
from trackC_writeup import c4_read as cr
from trackC_writeup import c4_train as ct
from trackC_writeup import c4pc_train as pt
from trackC_writeup import c6_warmup as cw

W = ct.W
GRID = tuple(ct.GRID)
PYTHIA = tuple(cr.PYTHIA)
END_SET = (1, 2)                     # {T1, T2}
TRANSIENT_MAX = 1000
V_STEPS = (1000, 2000, 4000)
L9H_F, L9H_A = (2,), (3, 4)
LATE_FRAC = 2 / 3
SEED_NULL_MAX = 1                    # of 9 seeds may read "holds"
L10B_SEEDS = 8                       # of 9 at ≥ ⅔ on the effect side
ENTRIES = ("L7", "L8", "L9", "L9h", "L10")
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
    for f in ("trackC_writeup/c4pc_read.py", "trackC_writeup/c4_read.py", "trackC_writeup/c4_close.py",
              "trackC_writeup/c4p_path.py", "trackC_writeup/c4pc_train.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the rules

def end_and_transients(mat: np.ndarray, grid: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    g = list(grid)
    if g[-1] != ct.HORIZON:
        raise C4pcReadRefused(f"grid ends at {g[-1]}, not {ct.HORIZON}")
    end = cc.majority(mat[-1])
    seen = {t for i, s in enumerate(g) if s <= TRANSIENT_MAX for t in cc.majority(mat[i])}
    return tuple(end), tuple(sorted(seen - set(end)))


def l7(f_mats: Dict[str, np.ndarray], a_mats: Dict[str, np.ndarray], grid: Sequence[int],
       arms: Sequence[str]) -> dict:
    if not arms:
        return {"holds": None, "status": "unreadable (no gated τ arm)"}
    per = {}
    for k in arms:
        ef, tf = end_and_transients(f_mats[k], grid)
        ea, ta = end_and_transients(a_mats[k], grid)
        per[k] = {"F_end": list(ef), "F_transient": list(tf), "A0_end": list(ea), "A0_transient": list(ta),
                  "holds": bool(tf and not ta and ef == END_SET and ea == END_SET)}
    return {"holds": all(v["holds"] for v in per.values()), "status": "read", "per_arm": per}


def l8(f_u: np.ndarray, a_u: np.ndarray, grid: Sequence[int]) -> dict:
    g = list(grid)
    per = {}
    for s in V_STEPS:
        sf, sa = cc.majority(f_u[g.index(s)]), cc.majority(a_u[g.index(s)])
        per[str(s)] = {"F": list(sf), "A0": list(sa),
                       "holds": bool(sf and sa and min(sf) == min(sa) - 1)}
    return {"holds": all(v["holds"] for v in per.values()), "status": "read", "per_step": per}


def m_cum(t: float) -> float:
    return cr.m("cumulative", t)


def nearer_bar(t_a0: float) -> float:
    """(i)'s bar: t_F′ is nearer m_cum(t_A0′) than t_A0′ in log iff t_F′ < this (for t_F′ < t_A0′)."""
    return t_a0 ** 1.5 / math.sqrt(2 * W)


def d_cum(t_f: float, t_a0: float) -> float:
    return abs(math.log(t_f / m_cum(t_a0)))


def l9(ind_f: dict, ind_a: dict, cen_f: Dict[str, dict], cen_a: Dict[str, dict], arms: Sequence[str]) -> dict:
    """``ind_*`` and ``cen_*[k]`` are `c4_read.date_on` results."""
    tf, ta = ind_f["step"], ind_a["step"]
    if tf is None or ta is None:
        return {"holds": None, "status": f"unreadable (induction F {ind_f['status']}, A0 {ind_a['status']})"}
    if ta > W:
        return {"holds": None, "status": f"unreadable (A0's induction date {ta} past W)", "t_F": tf, "t_A0": ta}
    if not arms:
        return {"holds": None, "status": "unreadable (no gated τ arm)", "t_F": tf, "t_A0": ta}
    i_ok = tf < nearer_bar(ta)
    d_ind = d_cum(tf, ta)
    per = {}
    for k in arms:
        cf, ca = cen_f[k]["step"], cen_a[k]["step"]
        if cf is None or ca is None or ca > W:
            return {"holds": None, "status": f"unreadable (centred {k}: F {cf}, A0 {ca})", "t_F": tf, "t_A0": ta}
        per[k] = {"t_F": cf, "t_A0": ca, "d": d_cum(cf, ca), "holds": d_ind < d_cum(cf, ca)}
    ii_ok = all(v["holds"] for v in per.values())
    return {"holds": bool(i_ok and ii_ok), "status": "read", "t_F": tf, "t_A0": ta, "bar_i": nearer_bar(ta),
            "i": bool(i_ok), "d_induction": d_ind, "ii": bool(ii_ok), "per_arm": per}


def top_head(sc: np.ndarray) -> Tuple[int, int]:
    li, hi = np.unravel_index(int(np.argmax(sc)), sc.shape)
    return int(li), int(hi)


def l9h(sc_f_4000: np.ndarray, sc_a_4000: np.ndarray) -> dict:
    hf, ha = top_head(np.asarray(sc_f_4000)), top_head(np.asarray(sc_a_4000))
    return {"holds": bool(hf[0] in L9H_F and ha[0] in L9H_A), "status": "read",
            "F_top": list(hf), "A0_top": list(ha)}


def l10b(r_f: Sequence[float], r_a0p: Sequence[float]) -> dict:
    """r_F(s) > r_A0′(s) at ≥ ⅔ of the late steps."""
    if len(r_f) != len(r_a0p) or not r_f:
        raise C4pcReadRefused(f"late steps: {len(r_f)} against {len(r_a0p)}")
    beyond = [bool(x > y) for x, y in zip(r_f, r_a0p)]
    n = sum(beyond)
    return {"beyond": beyond, "n_beyond": n, "n": len(beyond), "holds": bool(n >= LATE_FRAC * len(beyond))}


def l10(early_reading: Optional[str], b: Optional[dict]) -> dict:
    """(a) and (b); (b) None when its gate refused it: then L10 scores on (a) alone."""
    if early_reading is None:
        return {"holds": None, "status": "unreadable (early window refused)"}
    a_ok = early_reading == "another path"
    if b is None:
        return {"holds": bool(a_ok), "status": "read on (a) alone ((b)'s gate refused)", "a": early_reading}
    return {"holds": bool(a_ok and b["holds"]), "status": "read", "a": early_reading, "b": b}


def l10b_gate(null: dict, seeds: Dict[str, dict]) -> dict:
    """``null`` and each seed are `l10b` results: the null at < ⅔, ≥ L10B_SEEDS seeds at ≥ ⅔."""
    n_seeds = sum(bool(s["holds"]) for s in seeds.values())
    ok = (not null["holds"]) and n_seeds >= L10B_SEEDS
    return {"null_n_beyond": f"{null['n_beyond']} of {null['n']}", "seeds_holding": n_seeds, "pass": bool(ok)}


def gate_entry(effect: dict, null: dict, seeds: Dict[str, dict]) -> dict:
    """Effect reads holds; the null is readable and does not hold; ≤ SEED_NULL_MAX seeds hold."""
    n_hold = sum(s["holds"] is True for s in seeds.values())
    ok = effect["holds"] is True and null["holds"] is False and n_hold <= SEED_NULL_MAX
    return {"effect": effect["holds"], "null": null["holds"], "null_status": null["status"],
            "seeds_holding": n_hold, "seed_results": {k: s["holds"] for k, s in seeds.items()},
            "pass": bool(ok)}


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
    return {"L7": l7(f["mats"], a["mats"], grid, arms), "L8": l8(f["u"], a["u"], grid),
            "L9": l9(f["ind"], a["ind"], f["cen"], a["cen"], arms), "L9h": l9h(f["sc4000"], a["sc4000"])}


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


def gate(a) -> int:
    out = Path(a.out)
    arms = c4_arms()
    side = {}
    for arm in ("A0", "F"):
        side[arm] = pair_inputs(_load_recs(lambda s, p, arm=arm: cr.rec_path(C4_U2, arm, s, p), GRID),
                                _heads(C4_CLOSE / f"heads_{arm}.json"), GRID, arms)
    a0_py = pair_inputs(_load_recs(lambda s, p: cr.rec_path(C4_U2, "A0", s, p), PYTHIA),
                        _heads(C4_CLOSE / "heads_A0.json"), PYTHIA, arms)
    seed = {}
    for nm in ("seed0",) + SEEDS:
        seed[nm] = pair_inputs(_load_recs(lambda s, p, nm=nm: cw.rec_path(SEED_RECS, nm, s, p), PYTHIA),
                               _heads(SEED_HEADS / f"heads_{nm}.json"), PYTHIA, arms)
    rel_close = pair_inputs(_load_recs(lambda s, p: cw.rec_path(SEED_RECS, "seed0", s, p), PYTHIA),
                            _heads(C4_CLOSE / "heads_released.json"), PYTHIA, arms)
    effect = read_pair(side["F"], side["A0"], GRID, arms)
    null = read_pair(seed["seed0"], a0_py, PYTHIA, arms)
    seeds = {nm: read_pair(seed[nm], seed["seed0"], PYTHIA, arms) for nm in SEEDS}
    beside = {"null_with_c4_close_released_heads": {k: v["holds"] for k, v in
                                                    read_pair(rel_close, a0_py, PYTHIA, arms).items()}}
    res = {"code": code_sha(), "arms": list(arms), "grids": {"effect": list(GRID), "nulls": list(PYTHIA)},
           "inputs": {"c4_u2": str(C4_U2), "c4_close": str(C4_CLOSE), "seed_recs": str(SEED_RECS),
                      "seed_heads": str(SEED_HEADS)},
           "gate": {e: gate_entry(effect[e], null[e], {k: v[e] for k, v in seeds.items()}) for e in effect},
           "effect": effect, "null": null, "seeds": seeds, "beside": beside}
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(res, indent=1, default=_np) + "\n")
    for e, g in res["gate"].items():
        print(e.ljust(4), json.dumps(g))
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
    """No score before every entry is registered or declined and the gate exists with this code."""
    st = pt.require_decided()
    g = out / "gate.json"
    if not g.exists():
        raise C4pcReadRefused(f"{g}: the gate first")
    gj = json.loads(g.read_text())
    if gj["code"] != code_sha():
        raise C4pcReadRefused(f"gate code {gj['code']} ≠ {code_sha()}")
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
