"""
trackC_writeup/c8_rules.py — C8 U1, candidate (i): the loss-rise and grad-spike rule R_i, pure
(`design-C8.md` "U1, candidate (i)"; fixed there in full on 2026-11-27, before any log was opened).

numpy only, so it runs in the pure test tier. The I/O, the gates' drivers and the read live in
`c8_read.py`. **`code_sha()` hashes this file**; the reader writes it into every output.

A log is a dict {update s: value}; update s read batch s − 1 on every arm (`c4_train`'s loop), so
arms on one order read the same batch at the same s, and a twin is matched on s.

- **Window** (`window`): (t⁻ − 16, t], and on a branch not before the restart's first 8 updates.
- **Events** (`events`), at s ≥ 9 (s ≥ 265 on a branch): loss rise L_s − min(L_{s−8..s−1}) > δ_L;
  grad spike g_s > κ · median(g_{s−8..s−1}). **Counted** (`counted`): less the batch twin's events
  of the same kind at the same s.
- **Constants** (`constants`): on A0's updates 2001–4000, σ̂_L = 1.4826 MAD(ΔL)/√2,
  ĉ_g = 1.4826 MAD(Δ ln g)/√2; δ_L = max(0.02, 6σ̂_L), κ = max(1.5, exp(6ĉ_g)).
- **Background** (`background`): blocks of the window's length from the span's first update to
  min(end, 1000), less any within 16 of the window; b = share holding a counted event; < 4 blocks
  refuses the condition.
- **Label** (`label`): marked / marked at ≥ half the peak / silent / mixed over the five
  conditions, Π over the factors {F, F½} (larger b), F′, F⅒, B_P; `joint` across the gated τ.
- **Clip** (`clip_label`): share of window updates with g > 1.0, a description with no bar.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

LEAD = 16                 # rule T's SPAN_MIN (c4pdf_rules)
RESTART = 8               # a branch's first updates, excluded (the restart spike, seen)
GUARD = 16                # a background block within this many updates of the window is dropped
MIN_BLOCKS = 4            # fewer background blocks refuses the condition
TRAIL = 8                 # trailing updates an event is judged against
FIRST = TRAIL + 1         # first update with a full trailing set
BRANCH_FROM = 256
BG_END = 1000
CONST_SPAN = (2001, 4000)  # A0's late log (a control arm, smooth there)
MAD_K = 1.4826
NSIG = 6.0
DL_FLOOR = 0.02
K_FLOOR = 1.5
PI_BAR = 0.05
CLIP = 1.0
KINDS = ("loss", "grad")
CONDITIONS = ("F", "Fp", "Fh", "Ft", "BP")
FACTORS_4 = (("F", "Fh"), ("Fp",), ("Ft",), ("BP",))
FACTORS_3 = (("F", "Fh"), ("Fp",), ("BP",))
HALF_PEAK = ("F", "Fp", "Fh", "BP")
LABELS = ("marked", "marked at ≥ half the peak", "silent", "mixed")

Log = Mapping[int, float]
Events = Dict[str, Set[int]]


class C8RuleRefused(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def window(tm: int, t: int, branch: Optional[int] = None) -> Tuple[int, int]:
    """(lo, hi): updates s with lo < s ≤ hi."""
    lo = tm - LEAD
    if branch is not None:
        lo = max(lo, branch + RESTART)
    if lo >= t:
        raise C8RuleRefused(f"empty window for ({tm}, {t}] on branch {branch}")
    return lo, t


def first_update(branch: Optional[int] = None) -> int:
    return branch + RESTART + 1 if branch is not None else FIRST


def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def constants(loss: Log, gn: Log, span: Tuple[int, int] = CONST_SPAN) -> dict:
    a, b = span
    s = list(range(a, b + 1))
    missing = [k for k in s if k not in loss or k not in gn]
    if missing:
        raise C8RuleRefused(f"constants' span {span}: {len(missing)} updates missing (first {missing[0]})")
    L = np.array([loss[k] for k in s], dtype=float)
    g = np.array([gn[k] for k in s], dtype=float)
    if np.any(g <= 0):
        raise C8RuleRefused("non-positive grad norm in the constants' span")
    sl = MAD_K * _mad(np.diff(L)) / np.sqrt(2)
    cg = MAD_K * _mad(np.diff(np.log(g))) / np.sqrt(2)
    return {"span": [a, b], "sigma_L": float(sl), "c_g": float(cg),
            "delta_L": float(max(DL_FLOOR, NSIG * sl)), "kappa": float(max(K_FLOOR, np.exp(NSIG * cg))),
            "floors_bind": {"delta_L": bool(NSIG * sl < DL_FLOOR), "kappa": bool(np.exp(NSIG * cg) < K_FLOOR)}}


def events(loss: Log, gn: Log, dl: float, kappa: float, lo: int, hi: int, branch: Optional[int] = None) -> Events:
    """Events at updates s with lo < s ≤ hi (and s ≥ first_update)."""
    out: Events = {"loss": set(), "grad": set()}
    for s in range(max(lo + 1, first_update(branch)), hi + 1):
        tr = range(s - TRAIL, s)
        if s not in loss or any(k not in loss for k in tr):
            raise C8RuleRefused(f"update {s} or its trailing {TRAIL} missing")
        if loss[s] - min(loss[k] for k in tr) > dl:
            out["loss"].add(s)
        if gn[s] > kappa * float(np.median([gn[k] for k in tr])):
            out["grad"].add(s)
    return out


def counted(ev: Events, twin: Optional[Events]) -> Events:
    if twin is None:
        return {k: set(ev[k]) for k in KINDS}
    return {k: ev[k] - twin[k] for k in KINDS}


def union(ev: Events) -> Set[int]:
    return set().union(*(ev[k] for k in KINDS))


def blocks(lo: int, hi: int, span: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Non-overlapping blocks [s, s + n − 1] of the window's length n from span[0], kept if none is
    within GUARD of the window (`c8_thresholds.blocks`)."""
    n = hi - lo
    a, b = span
    out, s = [], a
    while s + n - 1 <= b:
        e = s + n - 1
        if e <= lo - GUARD or s > hi + GUARD:
            out.append((s, e))
        s += n
    return out


def background(cnt: Events, lo: int, hi: int, span: Tuple[int, int]) -> dict:
    bl = blocks(lo, hi, span)
    ev = union(cnt)
    hit = [any(a <= s <= e for s in ev) for a, e in bl]
    if len(bl) < MIN_BLOCKS:
        return {"n_blocks": len(bl), "n_hit": sum(hit), "b": None, "refused": True}
    return {"n_blocks": len(bl), "n_hit": int(sum(hit)), "b": float(np.mean(hit)), "refused": False}


def span_of(log_steps: Iterable[int], branch: Optional[int] = None) -> Tuple[int, int]:
    return first_update(branch), min(max(log_steps), BG_END)


def clip_label(gn: Log, lo: int, hi: int) -> dict:
    w = [gn[s] for s in range(lo + 1, hi + 1)]
    share = float(np.mean([g > CLIP for g in w]))
    lab = "clipped" if share >= 0.5 else ("unclipped" if share == 0 else "partly")
    return {"share": share, "label": lab}


def condition(loss: Log, gn: Log, twin: Optional[Tuple[Log, Log]], lo: int, hi: int, dl: float, kappa: float,
              branch: Optional[int] = None) -> dict:
    """One condition at one window (lo, hi] (from `window`, or shifted by the G-null): counted events
    in the window, and the background b over the arm's span."""
    sp = span_of(loss.keys(), branch)
    if lo + 1 < sp[0] or hi > sp[1]:
        raise C8RuleRefused(f"window ({lo}, {hi}] outside the span {sp}")
    a, b = sp
    ev = events(loss, gn, dl, kappa, a - 1, b, branch)
    tw = events(twin[0], twin[1], dl, kappa, a - 1, b, branch) if twin is not None else None
    cnt = counted(ev, tw)
    inw = {k: sorted(s for s in cnt[k] if lo < s <= hi) for k in KINDS}
    twin_in = {k: sorted(s for s in (ev[k] - cnt[k]) if lo < s <= hi) for k in KINDS}
    return {"window": [lo, hi], "span": [a, b], "events": inw, "twin_removed": twin_in,
            "has_event": bool(inw["loss"] or inw["grad"]), "background": background(cnt, lo, hi, sp),
            "leads": {k: [hi - s for s in inw[k]] for k in KINDS}}


def shift_range(wins: Mapping[str, Tuple[int, int]], spans: Mapping[str, Tuple[int, int]]) -> List[int]:
    """Every k with each window (lo + k n, hi + k n], n = hi − lo, inside its arm's span (the G-null's
    joint shifts; a span starting after a branch's restart keeps every shift clear of it)."""
    lo_k, hi_k = -10 ** 9, 10 ** 9
    for c, (lo, hi) in wins.items():
        n, (a, b) = hi - lo, spans[c]
        lo_k = max(lo_k, -((lo + 1 - a) // n))          # lo + 1 + k n ≥ a
        hi_k = min(hi_k, (b - hi) // n)                 # hi + k n ≤ b
    return list(range(lo_k, hi_k + 1))


def pi(conds: Mapping[str, dict], factors: Sequence[Sequence[str]]) -> Optional[float]:
    p = 1.0
    for f in factors:
        bs = [conds[c]["background"]["b"] for c in f]
        if any(b is None for b in bs):
            return None
        p *= max(bs)
    return float(p)


def label(conds: Mapping[str, dict], bk_clean: bool) -> dict:
    """R_i at one τ over the five conditions; ``bk_clean``: neither B_K nor BKO has a counted event
    in B_P's window."""
    if set(conds) != set(CONDITIONS):
        raise C8RuleRefused(f"conditions {sorted(conds)} are not {sorted(CONDITIONS)}")
    refused = [c for c in CONDITIONS if conds[c]["background"]["refused"]]
    has = {c: conds[c]["has_event"] for c in CONDITIONS}
    p4, p3 = pi(conds, FACTORS_4), pi(conds, FACTORS_3)
    if refused:
        lab = "refused"
    elif all(has.values()) and p4 <= PI_BAR and bk_clean:
        lab = "marked"
    elif all(has[c] for c in HALF_PEAK) and not has["Ft"] and p3 <= PI_BAR and bk_clean:
        lab = "marked at ≥ half the peak"
    elif not any(has.values()):
        lab = "silent"
    else:
        lab = "mixed"
    return {"label": lab, "has_event": has, "Pi4": p4, "Pi3": p3, "bk_clean": bool(bk_clean),
            "refused": refused}


def joint(per_tau: Mapping[str, dict]) -> str:
    labs = {v["label"] for v in per_tau.values()}
    return labs.pop() if len(labs) == 1 else "mixed across τ"


def parse_log(lines: Iterable[str]) -> Dict[int, dict]:
    """{update: row}; a repeated update (a resumed segment) keeps its last row; gaps refuse."""
    import json
    out: Dict[int, dict] = {}
    for ln in lines:
        if not ln.strip():
            continue
        r = json.loads(ln)
        if "step" in r:
            out[int(r["step"])] = r
    if not out:
        raise C8RuleRefused("no update rows")
    st = sorted(out)
    if st != list(range(st[0], st[-1] + 1)):
        raise C8RuleRefused(f"updates not contiguous between {st[0]} and {st[-1]}")
    return out
