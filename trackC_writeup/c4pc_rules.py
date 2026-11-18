"""
trackC_writeup/c4pc_rules.py — C4′ (c): the five registered leans' rules, pure (`design-C4pc.md`
"The registered family", "The gates"; frozen on Blocked 166 all (a), 2026-11-18).

**This file's `code_sha()` is the hash the five entries register** (`/challenge-pr` on #362,
finding 1). It hashes this file and the frozen instruments the rules call (`c4_read.py`
`220161b9159b`'s `date_on` / `m` / `KEYS`, `c4_close.py`'s `per_transition` / `majority`, and
`c4p_path.py`'s `read_pseudo` for L10), none of which C4′ (c) edits. The I/O, the gates' drivers,
the readings M1–M4 and U3's K wiring live in `c4pc_read.py`, which may grow without moving this
hash.

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
  A0′'s path); (b) r_F(s) > r_A0′(s) at ≥ ⅔ of F's 12 late steps (`l10b`); (b)'s gate `l10b_gate`.
- **A gate** (`gate_entry`): the effect reads "holds", the null is readable and does not hold,
  and ≤ `SEED_NULL_MAX` of the seed pseudo-pairs hold.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import REPO
from trackC_writeup import c4_close as cc
from trackC_writeup import c4_read as cr
from trackC_writeup import c4_train as ct

W = ct.W
# every constant below is placed (design-C4pc.md "The registered family", "The gates")
END_SET = (1, 2)                     # {T1, T2}, C4's end set (placed)
TRANSIENT_MAX = 1000                 # placed (design)
V_STEPS = (1000, 2000, 4000)
L9H_F, L9H_A = (2,), (3, 4)
LATE_FRAC = 2 / 3                    # placed (design)
SEED_NULL_MAX = 1                    # of 9 seeds may read "holds" (placed, design)
L10B_SEEDS = 8                       # of 9 at ≥ ⅔ on the effect side (placed, design)
ENTRIES = ("L7", "L8", "L9", "L9h", "L10")


class C4pcRuleRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4pc_rules.py", "trackC_writeup/c4_read.py", "trackC_writeup/c4_close.py",
              "trackC_writeup/c4p_path.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def end_and_transients(mat: np.ndarray, grid: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    g = list(grid)
    if g[-1] != ct.HORIZON:
        raise C4pcRuleRefused(f"grid ends at {g[-1]}, not {ct.HORIZON}")
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
        raise C4pcRuleRefused(f"late steps: {len(r_f)} against {len(r_a0p)}")
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
