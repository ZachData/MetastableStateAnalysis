"""
trackA_force/a1b.py — A1b: T_proj against the causal t* per (step, v1 prompt), on A1a's records
(`design-A.md` "A1b, the reading"; the family and the rule fixed there before any output).

- **The reading per cell**, on `T_proj` alone: *below* if `T_proj` < t* at the bracket's p10 β,
  *not below* if `T_proj` ≥ t* at its p90 β, else *straddles*; a t* that is infinite or refused
  makes the cell *not read*. *Far below*: `T_proj` < t*(p10) / 2.
- **Per step:** the reading of ≥ 6 of 7 prompts, else *mixed*.
- **P-gamma2's number:** the median ratio `T_proj / t*` per step (both bracket ends), its prompt
  range; the same at the registered comparator ((6.9) from g0 = 0) beside.
- **Beside, not in the reading:** `T_cal`, `T_disp`, `T_attn_cal`, `T_attn_proj`, the off-field
  share `1 − T_proj / T_cal`, `T_proj_c / T_proj` (the drift row, `status-A.md` "A1b, before
  output"), block 23's `proj`, the mean cosine at state 23 (≥ 0.9 is collapse), and at 143000 the
  share of `T_proj` carried by blocks 20–22 (Blocked 48 (a)).
- **H-WARP** (beside): per-block mean cosine against cumulative `T_proj`, a pooled isotonic fit
  over the other 17 steps predicts the left-out one; the warp *collapses the curves* if that
  error is ≤ ½ of the depth-axis error in ≥ 12 of 18 steps. Null: each step given another
  step's T profile (17 draws).
- **1e's question** (beside): the β at which (6.9)'s t* from g0 doubles its β = 0 value, at
  n = 467 and 2,041; no new run.

Tier 1: exploratory, unregistered.
    python -m trackA_force.a1b --records <a1a batch dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .a1a import STEPS, V1  # the family: 18 steps × 7 v1 prompts

CODE = "a08544e9b644"                # A1a's records this reading is fixed on (`status-A.md`)
PROMPTS = "976b62d677bc"
PER_STEP = 6                         # of 7 prompts
FAR = 0.5
WARP_RATIO = 0.5
WARP_STEPS = 12                      # of 18
LATE_BLOCKS = (20, 21, 22)
COLLAPSE = 0.9


def read_cell(T: float, t10: Optional[float], t90: Optional[float]) -> str:
    if t10 is None or t90 is None or not np.isfinite(t10) or not np.isfinite(t90):
        return "not read"
    if t10 > t90:
        raise ValueError(f"t* at p10 {t10} above t* at p90 {t90}: the rule's ordering fails")
    if T < t10:
        return "below"
    if T >= t90:
        return "not below"
    return "straddles"


def read_step(cells: List[str]) -> str:
    for r in ("below", "not below", "straddles", "not read"):
        if cells.count(r) >= PER_STEP:
            return r
    return "mixed"


def load(records: Path) -> List[Dict]:
    recs = [json.loads(f.read_text()) for f in sorted(Path(records).glob("step*_*.json"))]
    have = {(r["step"], r["prompt"]) for r in recs}
    missing = [(s, p) for s in STEPS for p in V1 if (s, p) not in have]
    if missing or len(recs) != len(STEPS) * len(V1):
        raise SystemExit(f"refusing: {len(recs)} records, missing {missing}")
    bad = {(r["code"], r["prompts_hash"]) for r in recs} - {(CODE, PROMPTS)}
    if bad:
        raise SystemExit(f"refusing: records from code/prompts {bad}, reading fixed on {CODE}/{PROMPTS}")
    return recs


def cell(r: Dict) -> Dict:
    T, ts = r["T"], r["tstar"]
    t10, t90 = ts["causal"].get("p10"), ts["causal"].get("p90")
    tp = T["T_proj"]
    reading = read_cell(tp, t10, t90)
    b = {x["block"]: x for x in r["blocks"]}
    z10, z90 = ts["eq69_zero"].get("p10"), ts["eq69_zero"].get("p90")
    ratio = lambda t: None if t is None else tp / t
    return {"step": r["step"], "prompt": r["prompt"], "reading": reading,
            "far_below": reading == "below" and tp < FAR * t10,
            "T_proj": tp, "t10": t10, "t90": t90, "r10": ratio(t10), "r90": ratio(t90),
            "reg_r10": ratio(z10), "reg_r90": ratio(z90),
            "T_cal": T["T_cal"], "T_disp": T["T_disp"], "T_attn_cal": T["T_attn_cal"],
            "T_attn_proj": T["T_attn_proj"], "T_proj_c": T["T_proj_c"],
            "off_field": 1.0 - tp / T["T_cal"], "drift_ratio": T["T_proj_c"] / tp if tp else None,
            "proj_23": b[23]["proj"], "cos_23": b[23]["mean_cos"], "collapse": b[23]["mean_cos"] >= COLLAPSE,
            "late_share": sum(b[k]["proj"] for k in LATE_BLOCKS) / tp if tp else None,
            "g0": ts["g0"], "beta": r["beta"]["bracket"]}


def _iso_loso(xs: Dict[int, np.ndarray], ys: Dict[int, np.ndarray]) -> Dict[int, float]:
    from sklearn.isotonic import IsotonicRegression
    err = {}
    for s in xs:
        X = np.concatenate([xs[o] for o in xs if o != s])
        Y = np.concatenate([ys[o] for o in ys if o != s])
        f = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(X, Y)
        err[s] = float(np.mean(np.abs(f.predict(xs[s]) - ys[s])))
    return err


def hwarp(recs: List[Dict]) -> Dict:
    """Per step, points (cumulative T_proj before block l, mean cosine at block l's input) over
    the 7 prompts and blocks 0–23; leave-one-step-out isotonic error on the T axis against the
    depth axis (x = l). Null: the T profile (per prompt) of another step, 17 draws per step."""
    prof, cos = {}, {}
    for r in recs:
        b = sorted(r["blocks"], key=lambda x: x["block"])
        prof[(r["step"], r["prompt"])] = np.concatenate([[0.0], np.cumsum([x["proj"] for x in b])[:-1]])
        cos[(r["step"], r["prompt"])] = np.array([x["mean_cos"] for x in b])
    depth = np.arange(len(prof[(STEPS[0], V1[0])]), dtype=float)
    ys = {s: np.concatenate([cos[(s, p)] for p in V1]) for s in STEPS}
    xt = {s: np.concatenate([prof[(s, p)] for p in V1]) for s in STEPS}
    xd = {s: np.tile(depth, len(V1)) for s in STEPS}
    eT, eD = _iso_loso(xt, ys), _iso_loso(xd, ys)
    ratio = {s: eT[s] / eD[s] for s in STEPS}
    passes = sum(ratio[s] <= WARP_RATIO for s in STEPS)
    # null: every step takes the T profile k steps along (k = 1…17), a derangement per draw
    null = []
    for k in range(1, len(STEPS)):
        xn = {s: xt[STEPS[(i + k) % len(STEPS)]] for i, s in enumerate(STEPS)}
        en = _iso_loso(xn, ys)
        null.append(sum(en[s] / eD[s] <= WARP_RATIO for s in STEPS))
    return {"err_T": eT, "err_depth": eD, "ratio": ratio, "steps_pass": int(passes),
            "collapses": passes >= WARP_STEPS, "null_steps_pass": null,
            "null_ge_observed": int(sum(n >= passes for n in null))}


def beta_doubling(n: int, g0: float) -> Optional[float]:
    """The β at which (6.9)'s t* from g0 is twice its β = 0 value (bisection on [0, 20])."""
    from .tstar_table import tstar
    base = tstar(n, 0.0, g0)
    f = lambda b: tstar(n, b, g0) - 2 * base
    lo, hi = 0.0, 20.0
    if f(hi) < 0:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def _fmt(x, f="{:.2f}"):
    return "—" if x is None else f.format(x)


def report(cells: List[Dict], warp: Dict, wells: Dict):
    L = [f"# A1b: T_proj against the causal t* (records code {CODE}, prompts {PROMPTS}; "
         f"{len(cells)} cells)", "",
         "## The reading", "",
         "step | reading (≥6 of 7) | below / straddles / not below / not read | far below | "
         "T_proj / t*(p10), median [range] | T_proj / t*(p90), median | registered: / (6.9, g0=0) p10, p90 | "
         "collapse at 23 (cos ≥ 0.9)",
         "---|---|---|---|---|---|---|---"]
    per_step = {}
    for s in STEPS:
        cs = [c for c in cells if c["step"] == s]
        rd = [c["reading"] for c in cs]
        per_step[s] = read_step(rd)
        r10 = [c["r10"] for c in cs if c["r10"] is not None]
        r90 = [c["r90"] for c in cs if c["r90"] is not None]
        g10 = [c["reg_r10"] for c in cs if c["reg_r10"] is not None]
        g90 = [c["reg_r90"] for c in cs if c["reg_r90"] is not None]
        L.append(f"{s} | {per_step[s]} | {rd.count('below')} / {rd.count('straddles')} / "
                 f"{rd.count('not below')} / {rd.count('not read')} | {sum(c['far_below'] for c in cs)} | "
                 f"{np.median(r10):.2f} [{min(r10):.2f}, {max(r10):.2f}] | {np.median(r90):.2f} | "
                 f"{np.median(g10):.2f}, {np.median(g90):.2f} | {sum(c['collapse'] for c in cs)} of 7")
    L += ["", "## Beside (medians over the 7 prompts)", "",
          "step | t*(p10), t*(p90) | T_proj | T_cal | T_disp | T_attn_cal | T_attn_proj | off-field 1−proj/cal | "
          "T_proj_c / T_proj | block 23 proj | cos at 23 | g0",
          "---|---|---|---|---|---|---|---|---|---|---|---"]
    med = lambda cs, k: float(np.median([c[k] for c in cs if c[k] is not None]))
    for s in STEPS:
        cs = [c for c in cells if c["step"] == s]
        L.append(f"{s} | {med(cs, 't10'):.2f}, {med(cs, 't90'):.2f} | {med(cs, 'T_proj'):.3f} | "
                 f"{med(cs, 'T_cal'):.2f} | {med(cs, 'T_disp'):.2f} | {med(cs, 'T_attn_cal'):.2f} | "
                 f"{med(cs, 'T_attn_proj'):.3f} | {med(cs, 'off_field'):.3f} | {med(cs, 'drift_ratio'):.2f} | "
                 f"{med(cs, 'proj_23'):.3f} | {med(cs, 'cos_23'):.3f} | {med(cs, 'g0'):.3f}")
    late = [c["late_share"] for c in cells if c["step"] == 143000]
    L += ["", f"143000: share of T_proj carried by blocks 20–22, per prompt: "
          f"{', '.join(_fmt(x) for x in late)} (median {np.median(late):.2f})",
          "", "## H-WARP (beside)", "",
          f"steps where LOSO error on the T axis ≤ {WARP_RATIO} × depth axis: {warp['steps_pass']} of 18 "
          f"(collapses the curves: {warp['collapses']}); null (17 shifted profiles): {warp['null_steps_pass']}, "
          f"≥ observed in {warp['null_ge_observed']} of 17", "",
          "step | err T axis | err depth axis | ratio", "---|---|---|---"]
    for s in STEPS:
        L.append(f"{s} | {warp['err_T'][s]:.4f} | {warp['err_depth'][s]:.4f} | {warp['ratio'][s]:.2f}")
    L += ["", "## 1e's question (beside): β where (6.9)'s t* doubles its β = 0 value", "",
          "n | g0 | β | t*(0)", "---|---|---|---"]
    for (n, g0), (b, t0) in wells.items():
        L.append(f"{n} | {g0:.3f} | {_fmt(b)} | {t0:.2f}")
    return "\n".join(L) + "\n", per_step


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .tstar_table import tstar
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--records", type=Path, required=True, help="A1a's batch dir (holds records/)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    recs = load(a.records / "records")
    cells = [cell(r) for r in recs]
    if not all(np.isfinite(c["T_proj"]) for c in cells):
        raise SystemExit("refusing: a non-finite T_proj")
    warp = hwarp(recs)
    g0_med = float(np.median([c["g0"] for c in cells if c["step"] == 143000]))
    wells = {(n, g): (beta_doubling(n, g), tstar(n, 0.0, g)) for n in (467, 2041) for g in (0.0, g0_med)}
    txt, per_step = report(cells, warp, wells)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "a1b_report.txt").write_text(txt)
    (a.out / "a1b.json").write_text(json.dumps(
        {"records": str(a.records), "code": CODE, "prompts_hash": PROMPTS, "per_step": per_step,
         "cells": cells, "hwarp": {k: (v if not isinstance(v, dict) else {str(s): x for s, x in v.items()})
                                   for k, v in warp.items()},
         "wells": [{"n": n, "g0": g, "beta": b, "t0": t0} for (n, g), (b, t0) in wells.items()]}, indent=1) + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
