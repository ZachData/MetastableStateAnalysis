"""
trackC_writeup/c4pdf_rules.py — C4′ (d)+(f): the rules, pure (`design-C4pdf.md` "The readings";
frozen on Blocked 172, 2026-11-23, before any new arm trained).

**This file's `code_sha()` is the hash a D3 (b) entry registers.** It hashes this file and the
frozen instrument it calls (`c4_close.py`'s `majority`, via `c4_close.MAJORITY`). The I/O, the
gates' drivers and the reads live in `c4pdf_read.py`, which may grow without moving this hash.

Transitions are 0-based (T1 is state 1 → 2, as C4 prints them). A "series" is a list of saved
states (updates since the run began, or absolute steps for a branch: only differences matter).

- **T** (`transient`): a maximal run of consecutive states, each with some transition outside
  {T1, T2} majority violated (> 3.5 of 7), whose last state is ≥ `SPAN_MIN` updates after its
  first. Onset = the first such run's first state.
- **Bands** (`bands`): a reference event known to (t⁻, t]; at rate fraction r, step
  [t⁻/ρ₀, tρ₀], dose [t⁻/(rρ₀), tρ₀/r]; separable iff disjoint (r < t⁻/(tρ₀²)).
- **d1** (`label_onset`, `d1_joint`): per arm and per gated τ arm, then the joint table.
- **d2** (`bar_date`, `label_induction`): the first state with the top head's score ≥ b = ½ P̄.
- **f** (`f_label`): B_P and B_O's transients.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import REPO
from trackC_writeup import c4_close as cc

END_SET = (1, 2)                    # {T1, T2}, C4's end set (placed in C4)
SPAN_MIN = 16                       # placed (design, after #371 finding 4; after seeing A0′'s one-state blip)
RHO0 = 1.25                         # C4's RHO_MIN (placed there as the grid's; design "d1")
RATES = {"Fh": 0.5, "Ft": 0.1}
HORIZON = 1000                      # F½ and F⅒ (D1 (a))
IND_REF = (256, 320)                # F's induction date's interval on F's grid (design "d2")
#: grid D: every 4 updates to 128, every 16 to 256, every 32 to 992, and 1000 (design "The arms")
GRID_D = tuple(sorted(set(range(0, 129, 4)) | set(range(144, 257, 16)) | set(range(288, 993, 32)) | {1000}))
#: the branches' offsets from 256 (C4's early grid)
OFFSETS = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256)
BRANCH_FROM = 256


class C4pdfRuleRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4pdf_rules.py", "trackC_writeup/c4_close.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def outside(row: Sequence[int]) -> Tuple[int, ...]:
    """The majority-violated transitions outside {T1, T2} at one state."""
    return tuple(t for t in cc.majority(row) if t not in END_SET)


def runs(mat: np.ndarray, states: Sequence[int]) -> List[Tuple[int, int]]:
    """Maximal runs (first state, last state) of consecutive states with a majority outside {T1, T2}."""
    mat = np.asarray(mat)
    if len(mat) != len(states):
        raise C4pdfRuleRefused(f"{len(mat)} rows on {len(states)} states")
    out, start = [], None
    for i, s in enumerate(states):
        if outside(mat[i]):
            if start is None:
                start = s
            last = s
        elif start is not None:
            out.append((start, last))
            start = None
    if start is not None:
        out.append((start, last))
    return out


def transient(mat: np.ndarray, states: Sequence[int]) -> dict:
    """Rule T. ``size`` (beside, never read): prompts violating outside {T1, T2}, summed over the run."""
    rs = runs(mat, states)
    ok = [r for r in rs if r[1] - r[0] >= SPAN_MIN]
    if not ok:
        return {"present": False, "onset": None, "runs": rs}
    a, b = ok[0]
    st = list(states)
    i, j = st.index(a), st.index(b)
    mat = np.asarray(mat)
    size = int(sum(mat[k][t] for k in range(i, j + 1) for t in range(mat.shape[1]) if t not in END_SET))
    return {"present": True, "onset": int(a), "end": int(b), "runs": rs, "size": size}


def previous(states: Sequence[int], t: int) -> int:
    st = list(states)
    i = st.index(t)
    if i == 0:
        raise C4pdfRuleRefused(f"reference at the first state {t}")
    return int(st[i - 1])


def bands(t_prev: float, t: float, r: float, rho: float = RHO0) -> dict:
    step = (t_prev / rho, t * rho)
    dose = (t_prev / (r * rho), t * rho / r)
    return {"step": step, "dose": dose, "separable": bool(dose[0] > step[1])}


def _in(x: float, b: Tuple[float, float]) -> bool:
    return b[0] <= x <= b[1]


def label_onset(tr: dict, ref: Tuple[int, int], r: float, horizon: int) -> dict:
    """d1 at one arm and one τ arm. ``ref`` = (t⁻, t) from F¹."""
    b = bands(ref[0], ref[1], r)
    if not b["separable"]:
        return {"label": "refused", **b}
    if not tr["present"]:
        if horizon > b["dose"][1]:
            return {"label": "none", **b}
        return {"label": "unreadable (horizon inside the dose band)", **b}
    x = tr["onset"]
    lab = "step" if _in(x, b["step"]) else "dose" if _in(x, b["dose"]) else "neither"
    return {"label": lab, "onset": x, **b}


JOINT = {("dose", "dose"): "a dose clock, no threshold down to a tenth of the peak",
         ("dose", "none"): "a threshold between a tenth and a half of the peak",
         ("step", "none"): "a threshold between a tenth and a half, on a step clock above it",
         ("none", "none"): "a threshold above half the peak",
         ("step", "step"): "a step clock"}


def d1_joint(per_arm: Dict[str, Dict[str, dict]], arms: Sequence[str]) -> dict:
    """``per_arm[τ] = {"Fh": label_onset, "Ft": label_onset}``; the joint reading holds only if
    every gated τ arm gives the same table row."""
    rows = {k: JOINT.get((per_arm[k]["Fh"]["label"], per_arm[k]["Ft"]["label"]), "mixed") for k in arms}
    vals = set(rows.values())
    joint = vals.pop() if len(vals) == 1 else "mixed"
    return {"joint": joint, "per_arm": rows}


def bar_date(top: Sequence[float], states: Sequence[int], b: float) -> Optional[int]:
    for x, s in zip(top, states):
        if x >= b:
            return int(s)
    return None


def label_induction(date: Optional[int], r: float, horizon: int, ref: Tuple[int, int] = IND_REF) -> dict:
    """d2 at one arm."""
    b = bands(ref[0], ref[1], r)
    if not b["separable"]:
        return {"label": "refused", **b}
    if date is None:
        if horizon > b["dose"][1]:
            return {"label": "not formed (slower than the dose)", **b}
        if horizon > b["step"][1]:
            return {"label": "not step", **b}
        return {"label": "unreadable", **b}
    lab = "step" if _in(date, b["step"]) else "dose" if _in(date, b["dose"]) else "neither"
    return {"label": lab, "date": date, **b}


F_TABLE = {(True, False): "not the early state's", (False, False): "the early state's",
           (True, True): "the restart's", (False, True): "unreadable (the control has a transient)"}


def f_label(tr_p: dict, tr_o: dict) -> str:
    return F_TABLE[(bool(tr_p["present"]), bool(tr_o["present"]))]


def stretch(mat: np.ndarray, src_states: Sequence[int], dst_states: Sequence[int], r: float) -> np.ndarray:
    """A plant: the state at s takes the source's at the largest source state ≤ r s (G-d1, G-d2)."""
    src = list(src_states)
    mat = np.asarray(mat)
    out = []
    for s in dst_states:
        k = max(i for i, x in enumerate(src) if x <= r * s)
        out.append(mat[k])
    return np.asarray(out)
