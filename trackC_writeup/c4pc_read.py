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

**Refusal** (`require_scorable`, called by `score`): no entry is scored before
`c4pc_train.registration()` has every id registered or declined, before U1's `gate.json` exists
with the rules' hash, and before **C4's gate at 3500 on F′** (`gate3500`) is recorded and pinned:
`trackC_writeup/c4pc_gate3500.json`, tracked and committed, holds the record's sha256, so the
gated τ arms and V's readability are fixed in git before any cell (`/challenge-pr` on #363,
finding 4). `records`, `heads`, `kcache`, `kjob`, `l10b-gate` and `measure` refuse before the
pin too.

**U3's wiring** (C4's and C4′'s instruments unchanged, since `c4pc_rules`' registered hash covers
`c4_read.py`, `c4_close.py` and `c4p_path.py`): `gate3500` runs `c4_read.gate` on an alias root
(A0 → A0p, F → Fp); `records` and `heads` call `c4_read.records` and `c4_close.heads` with the new
arm names; `kcache` and `kjob` run `c4p_path.cache` and `k_job` inside `wired()`, which extends
its job list and loader to two jobs: `A0p` (A0's 29 states, linked from C4′ U2's cache, against
A0′'s 28; r_A0′ for L10 (b)) and `Fp` (A0′'s 29 against F′'s 28; L10 (a)). `l10b-gate` is L10
(b)'s gate, written before F′'s K (which `kjob` refuses until then). `score` reads the five
entries; `measure` M1–M4.

    METS_DATA=<main>/data python -m trackC_writeup.c4pc_read gate --out <dir>
    HF_HOME=… python -m trackC_writeup.c4pc_read gate3500 --out <u3>; … pin3500 --out <u3>; commit the pin
    HF_HOME=… python -m trackC_writeup.c4pc_read records --out <u3>; … heads --out <u3>
    HF_HOME=… python -m trackC_writeup.c4pc_read kcache --out <u3> --job A0p Fp; … kjob --out <u3> --job A0p
    python -m trackC_writeup.c4pc_read l10b-gate --out <u3>; HF_HOME=… … kjob --out <u3> --job Fp
    python -m trackC_writeup.c4pc_read score --out <u3>; … measure --out <u3>
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, V1
from trackC_writeup import c4_close as cc
from trackC_writeup import c4_read as cr
from trackC_writeup import c4_train as ct
from trackC_writeup import c4p_path as cp
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
# U3's inputs
U1_OUT = DATA / "trackC/c4pc/2026-11-18-u1"         # U1's gate.json (L7–L9h)
C4P_U2 = DATA / "trackC/c4p/2026-11-17-u2"          # C4′ U2: L10 (a)'s gate, (b)'s null and seeds, r_F, A0's cache
PIN = REPO / "trackC_writeup/c4pc_gate3500.json"    # the 3500 gate's pin: tracked and committed before any score


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


def _sha256(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


def gate3500_record(out: Path) -> Path:
    return out / "gate3500" / "gate.json"


def require_pin(out: Path, pin: Path = PIN, check_git: bool = True) -> dict:
    """The 3500 gate's record exists under ``out``, was made by C4's frozen `c4_read`, and is the
    one pinned in the tracked, committed ``pin`` (`/challenge-pr` on #363, finding 4)."""
    if not pin.exists():
        raise C4pcReadRefused(f"{pin}: the 3500 gate's pin first")
    if check_git and not pt.committed(pin):
        raise C4pcReadRefused(f"{pin}: the pin counts only when committed (tracked and clean)")
    pj = json.loads(pin.read_text())
    rec = gate3500_record(out)
    if not rec.exists():
        raise C4pcReadRefused(f"{rec}: the 3500 gate first")
    if _sha256(rec) != pj["sha256"]:
        raise C4pcReadRefused(f"{rec}: sha256 differs from the pin's (the gate was re-run after the pin)")
    g = json.loads(rec.read_text())
    if g["code"] != cr.code_sha():
        raise C4pcReadRefused(f"3500 gate code {g['code']} ≠ c4_read {cr.code_sha()}")
    return {"u_read": bool(g["u_read"]), "arms_gated": g["arms_gated"], "mode": g["mode"]}


def require_scorable(out: Path, u1: Path = U1_OUT, pin: Path = PIN, check_git: bool = True) -> dict:
    """No score before every entry is registered or declined, U1's gate exists with the rules'
    hash, and C4's gate at 3500 on F′ is recorded under ``out`` and pinned in git."""
    st = pt.require_decided()
    g = u1 / "gate.json"
    if not g.exists():
        raise C4pcReadRefused(f"{g}: the gate first")
    gj = json.loads(g.read_text())
    if gj["rules"] != rl.code_sha():
        raise C4pcReadRefused(f"gate rules {gj['rules']} ≠ {rl.code_sha()}")
    return {"registration": st, "gate": gj["gate"], "gate3500": require_pin(out, pin, check_git)}


# ---------------------------------------------------------------- U3: C4's gate at 3500 on F′

def alias_root(out: Path, root: Path = pt.ROOT) -> Path:
    """A root where C4's arm names point at the new arms (A0 → A0p, F → Fp), so C4's frozen
    `c4_read.gate` runs unchanged on F′."""
    al = out / "gate3500" / "root"
    al.mkdir(parents=True, exist_ok=True)
    for c4_arm, arm in (("A0", "A0p"), ("F", "Fp")):
        ln = al / c4_arm
        if ln.is_symlink():
            if ln.resolve() != (root / arm).resolve():
                raise C4pcReadRefused(f"{ln} → {ln.resolve()}, not {root / arm}")
        else:
            ln.symlink_to(root / arm, target_is_directory=True)
    return al


def gate3500(a) -> int:
    out = Path(a.out)
    rec = gate3500_record(out)
    if rec.exists():
        raise C4pcReadRefused(f"{rec} exists: the gate is run once")
    al = alias_root(out, Path(a.root))
    return cr.gate(SimpleNamespace(root=str(al), out=str(rec.parent)))


def pin3500(a) -> int:
    """Writes the tracked pin from the gate's record (committing it is what makes it count)."""
    out = Path(a.out)
    rec = gate3500_record(out)
    g = json.loads(rec.read_text())
    pin = {"record": str(rec), "sha256": _sha256(rec), "code_c4_read": g["code"],
           "arm": "Fp (F′), through the alias root A0 → A0p, F → Fp", "step": g["step"], "mode": g["mode"],
           "u_read": g["u_read"], "arms_gated": g["arms_gated"], "b_dose": g["gate"].get("b_dose"),
           "null_vs_3b": g["null_vs_3b"], "pinned": time.strftime("%Y-%m-%dT%H:%M:%S")}
    PIN.write_text(json.dumps(pin, indent=1) + "\n")
    print(json.dumps(pin, indent=1))
    return 0


# ---------------------------------------------------------------- U3: records and heads on the new arms

def records(a) -> int:
    """C4's `records` (`c4_read.records`, unchanged) on A0′ and F′; refused before the 3500 pin."""
    out = Path(a.out)
    require_pin(out)
    for arm in a.arm:
        cr.records(SimpleNamespace(arm=arm, root=a.root, out=str(out), upto=ct.HORIZON, steps=None))
    return 0


def heads(a) -> int:
    """C4's per-head induction scores (`c4_close.heads`, unchanged) on every saved step of an arm."""
    out = Path(a.out)
    require_pin(out)
    for arm in a.arm:
        if (out / f"heads_{arm}.json").exists():
            continue
        cc.heads(SimpleNamespace(model=arm, root=a.root, out=str(out)))
    return 0


# ---------------------------------------------------------------- U3: L10's K wiring (c4p_path unchanged)

def _pc_job_keys(orig):
    def job_keys(job: str) -> list:
        """"Fp": F′ against A0′'s path (L10 (a)); "A0p": A0′ against A0's path (r_A0′, L10 (b));
        any other job is `c4p_path`'s own."""
        if job == "Fp":
            return [cp.key("A0p", t) for t in cp.A0_STEPS] + [cp.key("Fp", s) for s in cp.F_STEPS]
        if job == "A0p":
            return [cp.key("A0", t) for t in cp.A0_STEPS] + [cp.key("A0p", s) for s in cp.F_STEPS]
        return orig(job)
    return job_keys


PC_JOBS = ("A0p", "Fp")


def _pc_load(orig):
    def load(k: str, device):
        kind, s = cp.parse(k)
        if kind not in pt.ARMS:
            return orig(k, device)
        import torch
        from transformers import GPTNeoXForCausalLM
        from core.models import from_pretrained_eager
        m = from_pretrained_eager(GPTNeoXForCausalLM, str(ct.ckpt_dir(pt.ROOT, kind, s)),
                                  torch_dtype=torch.float32).to(device).eval()
        if any(p.dtype != torch.float32 for p in m.parameters()):
            raise C4pcReadRefused(f"{k}: not float32")
        return m
    return load


@contextlib.contextmanager
def wired():
    """`c4p_path`'s cache and K loops, unchanged, with its job list and loader extended to the
    new arms' kinds (A0p, Fp from `c4pc_train.ROOT`). Restored on exit."""
    saved = (cp.job_keys, cp.load, cp.JOBS)
    cp.job_keys, cp.load, cp.JOBS = _pc_job_keys(cp.job_keys), _pc_load(cp.load), PC_JOBS
    try:
        yield
    finally:
        cp.job_keys, cp.load, cp.JOBS = saved


def link_a0_cache(out: Path, src: Path = C4P_U2) -> int:
    """A0's 29 cached states from C4′ U2 (same code, same rows), linked rather than recomputed."""
    cd = cp.cache_dir(out)
    cd.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in cp.A0_STEPS:
        k = cp.key("A0", t)
        s = cp.cache_dir(src) / k
        meta = json.loads((s / "meta.json").read_text())
        if meta["code"] != cp.code_sha():
            raise C4pcReadRefused(f"{s}: cache code {meta['code']} ≠ c4p_path {cp.code_sha()}")
        d = cd / k
        if not d.exists():
            d.symlink_to(s, target_is_directory=True)
            n += 1
    return n


def kcache(a) -> int:
    out = Path(a.out)
    require_pin(out)
    if "A0p" in a.job:
        link_a0_cache(out)
    with wired():
        return cp.cache(SimpleNamespace(out=str(out), job=list(a.job)))


def kjob(a) -> int:
    out = Path(a.out)
    require_pin(out)
    if "Fp" in a.job and not (out / "l10b_gate.json").exists():
        raise C4pcReadRefused("F′'s K is computed after L10 (b)'s gate: l10b_gate.json missing")
    with wired():
        return cp.k_job(SimpleNamespace(out=str(out), job=list(a.job)))


def _u2(name: str) -> dict:
    d = json.loads((C4P_U2 / name).read_text())
    if d["code"] != cp.code_sha():
        raise C4pcReadRefused(f"C4′ U2's {name}: code {d['code']} ≠ c4p_path {cp.code_sha()}")
    return d


def _late(rd: dict, steps: Sequence[int]) -> list:
    r = dict(zip(rd["steps"], rd["r"]))
    return [r[s] for s in steps]


def r_a0p(out: Path) -> dict:
    """A0′ against A0's path: `read_pseudo` with A0′ as pseudo-F; r_A0′(s) is its r."""
    with wired():
        K, keys, hs, _ = cp._job(out, "A0p")
    return cp.read_pseudo(K, keys, "A0p", cp.F_STEPS, "A0", hs)


L10B_LATE = tuple(s for s in cp.F_STEPS if s >= cp.LATE_MIN)          # F's 12 late steps
L10B_GATE_LATE = tuple(s for s in cp.SEED_STEPS if s >= cp.LATE_MIN)  # the null's and seeds' 4


def l10b_gate(a) -> int:
    """L10 (b)'s gate (design "The gates"): U2's null and seeds' r against r_A0′ on their 4 late
    steps, by `c4pc_rules.l10b_gate`. Written before F′'s K exists."""
    out = Path(a.out)
    require_pin(out)
    if (out / "K_Fp.npz").exists():
        raise C4pcReadRefused("K_Fp.npz exists: L10 (b)'s gate is read before F′")
    u2g = _u2("gate.json")
    a0p = r_a0p(out)
    ra = _late(a0p, L10B_GATE_LATE)
    null = rl.l10b(_late(u2g["null"], L10B_GATE_LATE), ra)
    seeds = {sd: rl.l10b(_late(u2g["seeds"][sd], L10B_GATE_LATE), ra) for sd in cp.SEEDS}
    res = {"code": code_sha(), "rules": rl.code_sha(), "c4p_path": cp.code_sha(), "steps": list(L10B_GATE_LATE),
           "r_A0p": dict(zip(map(str, L10B_GATE_LATE), ra)), "gate": rl.l10b_gate(null, seeds),
           "null": null, "seeds": seeds, "A0p_vs_A0": a0p}
    ct._jdump(out / "l10b_gate.json", json.loads(json.dumps(res, default=_np)))
    print(json.dumps(res["gate"]))
    print(json.dumps({"A0p_vs_A0": _brief_path(a0p)}))
    return 0


def _brief_path(rd: dict) -> dict:
    return {w: (v["reading"], f"{v['n_on']}/{v['n']} on", f"{v['n_identity']} identity") for w, v in rd["windows"].items()}


# ---------------------------------------------------------------- U3: the scores (no forward pass)

def _new_side(out: Path, arm: str, arms: Sequence[str]) -> dict:
    return pair_inputs(_load_recs(lambda s, p, arm=arm: cr.rec_path(out, arm, s, p), GRID),
                       _heads(out / f"heads_{arm}.json"), GRID, arms)


def score(a) -> int:
    out = Path(a.out)
    sc = require_scorable(out)
    g35 = sc["gate3500"]
    arms = tuple(k for k in cr.KEYS[1:] if g35["arms_gated"][k])
    f, a0 = _new_side(out, "Fp", arms), _new_side(out, "A0p", arms)
    cells = read_pair(f, a0, GRID, arms)
    if not g35["u_read"]:
        cells["L8"] = {"holds": None, "status": "unreadable (V refused by the 3500 gate)"}
    u2g, u2r = _u2("gate.json"), _u2("reading.json")
    early_ok = bool(u2g["gate"]["early"]["readable"])
    lg = json.loads((out / "l10b_gate.json").read_text())
    with wired():
        K, keys, hs, _ = cp._job(out, "Fp")
    fp = cp.read_pseudo(K, keys, "Fp", cp.F_STEPS, "A0p", hs)
    b_desc = rl.l10b(_late(u2r["F"], L10B_LATE), _late(lg["A0p_vs_A0"], L10B_LATE))
    cells["L10"] = rl.l10(fp["windows"]["early"]["reading"] if early_ok else None,
                          b_desc if lg["gate"]["pass"] else None)
    gate_pass = {e: bool(sc["gate"][e]["pass"]) for e in ("L7", "L8", "L9", "L9h")}
    gate_pass["L10"] = early_ok
    scores = {}
    for e in rl.ENTRIES:
        h = cells[e]["holds"]
        scores[e] = {"id": pt.ENTRY_OF[e], "gate_pass": gate_pass[e], "holds": h if gate_pass[e] else None,
                     "scored": bool(gate_pass[e] and h is not None), "status": cells[e]["status"]}
    res = {"code": code_sha(), "rules": rl.code_sha(), "registration": sc["registration"], "gate3500": g35,
           "arms": list(arms), "scores": scores, "cells": cells, "L10_Fp_vs_A0p": fp,
           "L10b": {"gate": lg["gate"], "as_read": b_desc, "scored": bool(lg["gate"]["pass"])}}
    ct._jdump(out / "scores.json", json.loads(json.dumps(res, default=_np)))
    for e, s in scores.items():
        print(e.ljust(4), json.dumps(s))
    return 0


# ---------------------------------------------------------------- U3: M1–M4 (measured, not registered)

def label_at(f: dict, a0: dict, rho: float) -> dict:
    """`c4_read.label` with ρ given (M2, M3: ρ = max(the grid's, the rerun ratio q))."""
    if a0["status"] != "dated":
        return {"label": f"A0 {a0['status']}", "consistent": None}
    if f["status"] == "not formed by 4000":
        return {"label": "not formed by 4000", "consistent": None}
    if f["status"] != "dated":
        return {"label": f"F {f['status']}", "consistent": None}
    if not cr.separable(a0["step"], rho):
        return {"label": "A0 past the separable range", "consistent": None, "rho": rho, "t_a0": a0["step"]}
    cons = cr.consistent(f["step"], a0["step"], rho)
    return {"label": cr.name(cons), "consistent": cons, "rho": rho, "t_a0": a0["step"], "t_f": f["step"],
            "bands": {h: cr.band(h, a0["step"], rho) for h in cr.HYPS}}


def ratio(x: Optional[int], y: Optional[int]) -> Optional[float]:
    return None if not x or not y else float(max(x / y, y / x))


def readings(dates_f: dict, dates_a: dict, q: Dict[str, Optional[float]], gated: Dict[str, bool]) -> dict:
    """Each count's label at ρ = max(`rho_at`(t_A0), q) (q None: the grid's alone), R2's centred
    form over the gated τ arms, R3 on V when V is gated; `rho_star` beside."""
    per = {}
    for k in cr.KEYS:
        t = dates_a[k]["step"]
        per[k] = cr.label(dates_f[k], dates_a[k]) if t is None else \
            label_at(dates_f[k], dates_a[k], max(cr.rho_at(t), q[k] or 0.0))
    return {"per_key": per, "R2_centred": cr.centred_label({k: per[k] for k in cr.KEYS[1:]}, gated),
            "R3_V": per["u"] if gated.get("u") else None,
            "rho_star": {k: ({h: cc.rho_star(dates_a[k]["step"], dates_f[k]["step"], h) for h in cr.HYPS}
                             if dates_a[k]["step"] and dates_f[k]["step"] else None) for k in cr.KEYS}}


def measure(a) -> int:
    out = Path(a.out)
    g35 = require_pin(out)
    c4 = json.loads((C4_U2 / "reading.json").read_text())
    c4g = json.loads((C4_U2 / "gate.json").read_text())
    new = {arm: cr.all_series(_load_recs(lambda s, p, arm=arm: cr.rec_path(out, arm, s, p), GRID), GRID)
           for arm in pt.ARMS}
    d = {arm: {k: cr.date_on(new[arm][k]) for k in cr.KEYS} for arm in pt.ARMS}
    da, dfc = c4["dates"]["A0"], c4["dates"]["F"]
    q = {k: ratio(d["A0p"][k]["step"], da[k]["step"]) for k in cr.KEYS}
    m1 = {k: {"t_A0p": d["A0p"][k]["step"], "status_A0p": d["A0p"][k]["status"], "t_A0": da[k]["step"], "q": q[k]}
          for k in cr.KEYS}
    m2 = readings(dfc, da, q, {"u": c4g["u_read"], **c4g["arms_gated"]})
    m3 = readings(d["Fp"], d["A0p"], q, {"u": g35["u_read"], **g35["arms_gated"]})
    m4 = {k: {"t_Fp": d["Fp"][k]["step"], "status_Fp": d["Fp"][k]["status"], "t_F": dfc[k]["step"],
              "q": ratio(d["Fp"][k]["step"], dfc[k]["step"])} for k in cr.KEYS}
    res = {"code": code_sha(), "c4_read": cr.code_sha(), "gate3500": g35, "M1": m1, "M2": m2, "M3": m3, "M4": m4,
           "dates": d, "series": {arm: {k: v.tolist() for k, v in s.items()} for arm, s in new.items()},
           "inputs": {"c4_reading": str(C4_U2 / "reading.json"), "c4_gate": str(C4_U2 / "gate.json")}}
    ct._jdump(out / "measured.json", json.loads(json.dumps(res, default=_np)))
    print(json.dumps({"M1": m1, "M4": m4}, indent=1))
    for nm, m in (("M2", m2), ("M3", m3)):
        print(nm, json.dumps({"R2": m["R2_centred"]["label"], "R3": m["R3_V"]["label"] if m["R3_V"] else None,
                              "per_key": {k: v["label"] for k, v in m["per_key"].items()}}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("gate")
    q.add_argument("--out", required=True)
    for cmd in ("gate3500", "pin3500", "records", "heads", "kcache", "kjob", "l10b-gate", "score", "measure"):
        q = sub.add_parser(cmd)
        q.add_argument("--out", required=True)
        q.add_argument("--root", default=str(pt.ROOT))
        if cmd in ("records", "heads"):
            q.add_argument("--arm", nargs="+", choices=pt.ARMS, default=list(pt.ARMS))
        if cmd in ("kcache", "kjob"):
            q.add_argument("--job", nargs="+", choices=PC_JOBS, required=True)
    a = p.parse_args(argv)
    return {"gate": gate, "gate3500": gate3500, "pin3500": pin3500, "records": records, "heads": heads,
            "kcache": kcache, "kjob": kjob, "l10b-gate": l10b_gate, "score": score, "measure": measure}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
