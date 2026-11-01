"""
trackC_writeup/c6_warmup.py — C6: V and the centred count on `EleutherAI/pythia-70m-warmup01`
(`design-C6.md`; ROADMAP §8 row C6; `lit-C2p-fulltext.md` M1, Parked 1–2).

The run is `pythia-70m` with its warmup ten times longer (14 300 steps, Belrose et al.), at
Pythia's peak, from the same initialisation and first batch (`c6_schedule.py`, U0). The reading is
C2 unit 3b's (`c2_dose.py`), with two changes the run forces:

- **the reference is 70m's date range, not one date**: `pythia-70m` (seed 0, C2 unit 3b's records,
  re-derived here) and PolyPythias' `pythia-70m-seed1..9`, each read as unit 3b read 70m (V at
  D2's 1e-3; the centred V at 70m's four frozen arms; `a3.d4a`'s half-rise on the 1000–4000
  plateau). A count's reference set (`REF70`) is the union of its readable dates over the ten
  seeds, per arm for the centred count (the conservative band, as 3b's `REF`);
- **each schedule maps the whole reading, its plateau window included**: under a schedule h the
  run's timeline is 70m's through h's map m_h (step: t; cumulative: Σ rate equal; instantaneous:
  the rate itself equal, on the run's rising warmup). The run's grid step s then holds 70m's
  latest grid state that h has reached by s (`held`), so the run is read under h on the grid
  steps holding a 70m state in 1000–4000 (h's plateau window) with h's early set (the steps
  holding a state ≤ 128) for 92 (i)'s unreadable check, and h is consistent with a count if every
  gated arm's date is in that arm's band under h: the grid steps whose interval (prev(s), s]
  meets some (m_h(prev g), m_h(g)], g in the arm's `REF70`. A window fixed at 1000–4000 would
  read a stretched run mid-rise and date it early, which favours "step" (`design-C6.md` "Why
  each schedule has its own window"). `plant` maps a 70m profile onto the run's grid under h; the
  reading was calibrated on the ten seeds' profiles planted under each h before it froze (`calibrate`).

Under step the reading is exactly unit 3b's (`date_on(v, "step") == c2_dose.date_reading(v)`).

The gate (U1, before any scored record on the run): 70m's four arms, not re-derived (the dates
are compared with 70m's at those arms); plant B's dose by 3b's rule 100 (a) and the thresholds of
unit 3s, read on the run's off-grid steps 7000 and 20 000 and the 3 re-inits of 70m's config; an
arm is gated if it passes at both steps; the uncentred count is read only if 100 (a) finds B's
dose at both. The re-inits' rates must equal C2 unit 3b's 70m preflight (the same config and seeds).

Tier 1, description (nothing here is registered).
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c6_warmup run --out <dir> --models seed0 seed1 …
    python -m trackC_writeup.c6_warmup seeds --out <dir>          # REF70 and the bands
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c6_warmup gate --out <dir>      # U1
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c6_warmup run --out <dir> --models run   # U1
    python -m trackC_writeup.c6_warmup read --out <dir>           # U1
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, STEPS, V1, prompts_hash
from trackA_force.a5_seeds import REL_TOL, TOKENIZER
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_dose as cd
from trackC_writeup import c2_seeds as cs
from trackC_writeup.c6_schedule import DECAY, MIN_LR, PEAK, W_BASE

MODELS = {"seed0": "EleutherAI/pythia-70m", "run": "EleutherAI/pythia-70m-warmup01",
          **{f"seed{k}": f"EleutherAI/pythia-70m-seed{k}" for k in range(1, 10)}}
SEEDS = tuple(f"seed{k}" for k in range(10))
W_RUN = 14300                                   # Belrose et al.; checked against the weights (c6_schedule)
ARMS = cd.TAU_ARMS["70m"]                       # 70m's four arms, C2 unit 3b's preflight
HYPS = cd.HYPS                                  # step, cumulative, instantaneous
WINDOW = (1000, 2000, 4000)                     # a3.d4a's plateau steps on 70m
EARLY_TOP = 128                                 # 92 (i): the early set is the steps ≤ 128 on 70m
GATE_STEPS = (7000, 20000)                      # the run's off-grid gate steps (U1)
#: the 70m dates over seeds 0–9 (union; the centred per arm), from `seeds`
#: (`data/trackC/c6/2026-10-31-u0-seeds/seeds.json`); frozen 2026-10-31 before any output on the run.
REF70: Optional[dict] = {"u": (256, 512),
                         "c": {"9.61e-05": (512, 1000), "0.0001": (512, 1000), "0.0003": (512, 1000),
                               "0.00478": (1000, 2000)}}
U3B_70M = DATA / "trackC/c2u3b/2026-10-22-run/70m/records"    # seed 0's known answer
U3B_PREFLIGHT = DATA / "trackC/c2u3b/2026-10-22-preflight/70m/preflight_plants.json"
ENERGY_TOL = 1e-4                               # placed: |Δe1| / e1 against unit 3b's records (seed 0 gives 0.0)
SEED = 20261031


class C6Refused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c6_warmup.py", "trackC_writeup/c6_schedule.py", "trackC_writeup/c2_dose.py",
              "trackC_writeup/c2_seeds.py", "trackC_writeup/c2_aniso.py", "trackA_force/a5_seeds.py",
              "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the schedules' maps

def lr(x: float, warmup: int, peak: float = PEAK) -> float:
    """NeoX's rate at (continuous) step x: linear warmup, then cosine to 0.1 × peak at 143 000."""
    mn = peak * MIN_LR / PEAK
    if x <= warmup:
        return peak * x / warmup
    return mn + (peak - mn) / 2 * (1 + math.cos(math.pi * (x - warmup) / (DECAY - warmup)))


def cum(x: float, warmup: int, peak: float = PEAK) -> float:
    """∫₀ˣ lr (closed form; `tools/math_checks/c6_bands.py`)."""
    mn = peak * MIN_LR / PEAK
    if x <= warmup:
        return peak * x * x / (2 * warmup)
    u, L = x - warmup, DECAY - warmup
    return peak * warmup / 2 + mn * u + (peak - mn) / 2 * (u + L / math.pi * math.sin(math.pi * u / L))


def m(hyp: str, t: float) -> float:
    """The run's step at which it matches 70m's step t under ``hyp``."""
    if hyp == "step":
        return float(t)
    if hyp == "cumulative":
        target, lo, hi = cum(t, W_BASE), 0.0, float(DECAY)
        if target > cum(hi, W_RUN):
            return math.inf
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cum(mid, W_RUN) < target else (lo, mid)
        return (lo + hi) / 2
    if hyp == "instantaneous":
        return W_RUN * lr(t, W_BASE) / PEAK              # the run's rising warmup reaches 70m's rate
    raise ValueError(hyp)


def prev(s: int) -> int:
    return cd.prev(s)


def grid_of(x: float) -> Optional[int]:
    """The grid step whose interval (prev(s), s] holds x (step 0 for x ≤ 0; None past 143 000)."""
    if x <= 0:
        return 0
    return next((s for s in STEPS[1:] if prev(s) < x <= s), None)


def band(ref: Iterable[int], hyp: str) -> FrozenSet[int]:
    """The run's grid steps a schedule predicts from 70m's dates ``ref``: m_h's range over
    [prev(g), g]. m_h is monotone on each side of 70m's warmup end (the instantaneous map turns
    there: 70m's rate falls after its peak), so the range is spanned by the ends and that point."""
    out = set()
    for g in ref:
        pts = [m(hyp, prev(g)), m(hyp, g)] + ([m(hyp, W_BASE)] if prev(g) < W_BASE < g else [])
        lo, hi = min(pts), max(pts)
        out |= {s for s in STEPS[1:] if s > lo and prev(s) < hi}
    return frozenset(out)


@functools.lru_cache(None)
def reached(g: int, hyp: str) -> float:
    """The run's step by which it has reached 70m's state at g under ``hyp`` (m_h's running max:
    the instantaneous map turns at 70m's warmup end, and a state is reached once, not undone)."""
    return max([m(hyp, g)] + ([m(hyp, W_BASE)] if g >= W_BASE else []))


@functools.lru_cache(None)
def held(s: int, hyp: str) -> int:
    """70m's latest grid state the run holds at its grid step s under ``hyp``."""
    return max(g for g in STEPS if reached(g, hyp) <= s * (1 + 1e-9) + 1e-6)


@functools.lru_cache(None)
def window(hyp: str) -> Tuple[int, ...]:
    """The run's grid steps holding a 70m state in 70m's window; if none (the instantaneous map
    jumps it), the first holding a later one."""
    w = tuple(s for s in STEPS if WINDOW[0] <= held(s, hyp) <= WINDOW[-1])
    return w or (next(s for s in STEPS if held(s, hyp) >= WINDOW[0]),)


@functools.lru_cache(None)
def early(hyp: str) -> Tuple[int, ...]:
    return tuple(s for s in STEPS if held(s, hyp) <= EARLY_TOP)


LATE = ("literal", "same_step", "cumulative")   # the instantaneous plant past 70m's peak


@functools.lru_cache(None)
def _held_late(s: int, late: str) -> int:
    """`held` under instantaneous with 70m's states past its peak (g > W_BASE) reached as ``late``
    says: "literal" all at once (m_h's turn, `reached`); "same_step" when the run reaches 70m's
    step g; "cumulative" when it reaches g's summed rate (`/challenge-pr` on #295, finding 1)."""
    def r(g):
        if g <= W_BASE or late == "literal":
            return reached(g, "instantaneous")
        top = m("instantaneous", W_BASE)
        return max(top, g if late == "same_step" else m("cumulative", g))
    return max(g for g in STEPS if r(g) <= s * (1 + 1e-9) + 1e-6)


def plant(v: Sequence[float], hyp: str, late: str = "literal") -> List[float]:
    """A 70m profile on the run's grid if the run follows ``hyp``: V at each run step is 70m's V
    at the state it holds (the calibration's planted effect); ``late`` only for instantaneous."""
    if hyp == "instantaneous":
        return [float(v[STEPS.index(_held_late(s, late))]) for s in STEPS]
    return [float(v[STEPS.index(held(s, hyp))]) for s in STEPS]


# ---------------------------------------------------------------- pure: the reading

def date_on(v: Sequence[float], hyp: str) -> dict:
    """A count's half-rise under ``hyp``: the first grid step with V ≥ half the median of V over
    h's window; unreadable if that plateau is 0 or a step in h's early set is ≥ half of it."""
    v = np.asarray(v, float)
    w, e = window(hyp), early(hyp)
    p = float(np.median([v[STEPS.index(s)] for s in w]))
    ev = {s: float(v[STEPS.index(s)]) for s in e}
    bad = p <= 0 or max(ev.values()) >= p / 2
    step = None if bad else STEPS[int(np.argmax(v >= p / 2))]
    return {"step": step, "plateau": p, "window": list(w), "early": ev, "unreadable": bad}


def seed_dates(vu: Sequence[float], vcs: Dict[str, Sequence[float]]) -> dict:
    """One 70m seed, read as unit 3b (the step reading): its uncentred date, its centred dates."""
    u = date_on(vu, "step")
    arms = {k: date_on(v, "step") for k, v in vcs.items()}
    return {"u": u, "c": arms, "u_date": u["step"],
            "c_dates": sorted({a["step"] for a in arms.values() if not a["unreadable"]}),
            "c_arm_dates": {k: a["step"] for k, a in arms.items() if not a["unreadable"]}}


def ref70(per_seed: Dict[str, dict]) -> dict:
    """{"u": dates, "c": {arm: dates}}: each count's (each arm's) readable dates over the seeds."""
    u = sorted({r["u_date"] for r in per_seed.values() if r["u_date"] is not None})
    arms = sorted({k for r in per_seed.values() for k in r["c_arm_dates"]})
    cc = {k: tuple(sorted({r["c_arm_dates"][k] for r in per_seed.values() if k in r["c_arm_dates"]}))
          for k in arms}
    if not u or not cc:
        raise C6Refused(f"a count has no readable date on any 70m seed (u {u}, c {cc})")
    return {"u": tuple(u), "c": cc}


def _arm_ref(ref_k, arm: str) -> Tuple[int, ...]:
    return tuple(ref_k.get(arm, ())) if isinstance(ref_k, dict) else tuple(ref_k)


def bands(ref: dict) -> dict:
    out = {"u": {h: sorted(band(ref["u"], h)) for h in HYPS}}
    rc = ref["c"]
    out["c"] = ({a: {h: sorted(band(d, h)) for h in HYPS} for a, d in rc.items()} if isinstance(rc, dict)
                else {h: sorted(band(rc, h)) for h in HYPS})
    return out


def count_reading(vs: Dict[str, Sequence[float]], gated: Dict[str, bool], kind: str,
                  ref: Dict[str, Sequence[int]]) -> dict:
    """One count on the run: per schedule h and gated arm, the date on h's window. h maps 70m's
    readable reading onto the run (low through h's early set, a plateau on h's window, the
    half-rise in h's band), so an arm whose run count fails any of the three contradicts h: a
    plateau of 0 there or an early count ≥ half of it is not "unreadable" under h, as 92 (i) has
    it for one model, but "inconsistent". An arm counts if it is gated and has a reference date
    (``ref[kind]`` is one set for every arm, or one per arm); h is consistent if every counted
    arm's date is in that arm's band, inconsistent otherwise; the count is unreadable only if no
    arm counts. (Any one arm sufficing let the centred count fit every schedule: `calibrate`.)"""
    out = {}
    for h in HYPS:
        per, ok, bnd = {}, [], {}
        for k, v in vs.items():
            r = _arm_ref(ref[kind], k)
            if not gated[k] or not r:
                continue
            b = band(r, h)
            per[k], bnd[k] = date_on(v, h), sorted(b)
            ok.append(not per[k]["unreadable"] and per[k]["step"] in b)
        dates = sorted({r_["step"] for r_ in per.values() if not r_["unreadable"]})
        status = "unreadable" if not per else "consistent" if all(ok) else "inconsistent"
        out[h] = {"dates": dates, "band": bnd, "status": status, "per_arm": per}
    return out


def calibrate(beside: Dict[str, dict], ref: dict, loo: bool = False, late: str = "literal") -> dict:
    """The planted effect: each 70m seed's own profiles planted under each schedule (`plant`),
    read with every arm gated against ``ref`` (or, with ``loo``, the other seeds' dates). The rule
    was revised on these plants, so the table is in-sample to it (#295, finding 2)."""
    rows = {}
    for name, b in beside.items():
        r = ref
        if loo:
            others = {n: seed_dates(x["V_u"], x["V_c"]) for n, x in beside.items() if n != name}
            r = ref70(others)
        for truth in HYPS:
            vu, vcs = plant(b["V_u"], truth, late), {k: plant(v, truth, late) for k, v in b["V_c"].items()}
            counts = {"u": count_reading({"1e-3": vu}, {"1e-3": True}, "u", r),
                      "c": count_reading(vcs, {k: True for k in vcs}, "c", r)}
            rows[f"{name}|{truth}"] = {"joint": joint(counts),
                                       "status": {k: {h: counts[k][h]["status"] for h in HYPS} for k in counts}}
    summary = {t: {"truth_survives": sum(t in rows[f"{n}|{t}"]["joint"]["survivors"] for n in beside),
                   "truth_alone": sum(rows[f"{n}|{t}"]["joint"]["survivors"] == [t] for n in beside),
                   "labels": sorted({rows[f"{n}|{t}"]["joint"]["label"] for n in beside})}
               for t in HYPS}
    return {"summary": summary, "rows": rows}


def joint(counts: Dict[str, dict]) -> dict:
    """The schedules consistent on every count where they are readable, and readable on one."""
    surv, unread = [], []
    for h in HYPS:
        st = [counts[k][h]["status"] for k in counts]
        if all(s == "unreadable" for s in st):
            unread.append(h)
        elif "inconsistent" not in st:
            surv.append(h)
    readable = [h for h in HYPS if h not in unread]
    if not readable:
        label = "unreadable"
    elif set(surv) == set(readable) and len(readable) == len(HYPS):
        label = "not told apart"
    elif not surv:
        label = "neither"
    else:
        label = "consistent with " + " and ".join(surv)
    return {"survivors": surv, "unreadable": unread, "label": label}


# ---------------------------------------------------------------- model passes

def rec_path(out: Path, model: str, step: int, prompt: str) -> Path:
    return out / model / "records" / f"step{step}_{prompt}.json"


def meta(torch) -> dict:
    return {"design": "trackC_writeup/design-C6.md", "code": code_sha(), "arms": list(ARMS),
            "models": MODELS, "ref70": REF70, "w_run": W_RUN, "prompts_hash": prompts_hash(),
            "prompts": list(V1), "tokenizer": list(TOKENIZER), "dtype": "float32",
            "device": torch.cuda.get_device_name(0), "tf32": False, "torch": torch.__version__,
            "hf_home": os.environ.get("HF_HOME")}


def _frozen() -> Dict[str, Tuple[int, ...]]:
    if REF70 is None:
        raise C6Refused("REF70 is not set: the bands are not frozen")
    return REF70


def run(a) -> int:
    out = Path(a.out)
    if "run" in a.models:
        _frozen()
        g = out / "gate.json"
        if not g.exists() or not json.loads(g.read_text()).get("readable"):
            raise C6Refused(f"{g}: the run's gate has not passed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer
    out.mkdir(parents=True, exist_ok=True)
    m_ = meta(torch)
    (out / f"plan_{'_'.join(a.models)}.json").write_text(json.dumps({**m_, "run_models": a.models}, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    first, t0 = True, time.time()
    for name in a.models:
        repo = MODELS[name]
        rec_path(out, name, 0, V1[0]).parent.mkdir(parents=True, exist_ok=True)
        for step in STEPS:
            todo = [p for p in V1 if not rec_path(out, name, step, p).exists()]
            if not todo:
                continue
            model = cs.load(repo, step, device)
            n_states = model.config.num_hidden_layers + 1
            for p in todo:
                hs, ids = c.states(model, tok, p, device)
                rec = cd.record(hs, ids, p, n_states)
                rec.update(model=name, repo=repo, step=step, revision=f"step{step}", code=m_["code"])
                rec_path(out, name, step, p).write_text(json.dumps(rec) + "\n")
                if first:
                    print(f"first record populated: {name} step {step} {p}, n {rec['n']}, states {n_states}, "
                          f"e1[0] {rec['e1'][0]:.6f}, e1c[0] {rec['e1c'][0]:.6f}, cos[-1] {rec['cos'][-1]:.4f}",
                          flush=True)
                    first = False
            del model
            torch.cuda.empty_cache()
        print(f"{name} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


def load_model(out: Path, name: str) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in STEPS:
        for p in V1:
            f = rec_path(out, name, s, p)
            if not f.exists():
                raise C6Refused(f"{f}: missing")
            r = json.loads(f.read_text())
            if r.get("model") != name:
                raise C6Refused(f"{f}: model {r.get('model')}, not {name}")
            recs[(s, p)] = r
    return recs


def known_answer(recs: Dict[Tuple[int, str], dict]) -> float:
    """Seed 0's e1 against C2 unit 3b's 70m records; refuse past ENERGY_TOL or on other tokens."""
    worst = 0.0
    for (s, p), r in recs.items():
        old = json.loads((U3B_70M / f"step{s}_{p}.json").read_text())
        if old["ids_sha"] != r["ids_sha"]:
            raise C6Refused(f"seed0 step {s} {p}: tokens differ from unit 3b's")
        d = max(abs(x - y) / abs(y) for x, y in zip(r["e1"], old["e1"]))
        worst = max(worst, d)
    if worst > ENERGY_TOL:
        raise C6Refused(f"seed0's e1 differs from unit 3b's by {worst:.3g} > {ENERGY_TOL}")
    return worst


def seeds(a) -> int:
    """The 70m seeds' dates, REF70 and the bands per schedule (before the run is read)."""
    out = Path(a.out)
    per, beside = {}, {}
    for name in SEEDS:
        recs = load_model(out, name)
        vu = c.series(recs, "e1")
        vcs = {f"{t:.3g}": c.series(recs, "e1c", t) for t in ARMS}
        per[name] = seed_dates(vu, vcs)
        beside[name] = {"V_u": vu.tolist(), "V_c": {k: v.tolist() for k, v in vcs.items()}}
    ka = known_answer(load_model(out, "seed0"))
    ref, ref0 = ref70(per), ref70({"seed0": per["seed0"]})
    cal, cal_loo = calibrate(beside, ref), calibrate(beside, ref, loo=True)
    cal_late = {lt: calibrate(beside, ref, late=lt) for lt in LATE[1:]}
    res = {"code": code_sha(), "seed0_known_answer_max_rel": ka, "per_seed": per, "ref70": ref,
           "ref70_frozen_equal": REF70 is not None and _jsonable(REF70) == _jsonable(ref),
           "ref_seed0": ref0, "bands": bands(ref), "bands_seed0": bands(ref0),
           "c_union": sorted({d for r in per.values() for d in r["c_dates"]}),
           "windows": {h: list(window(h)) for h in HYPS}, "early": {h: list(early(h)) for h in HYPS},
           "held": {h: {str(s): held(s, h) for s in STEPS} for h in HYPS},
           "maps": {h: {str(t): m(h, t) for t in (64, 128, 256, 512, 1000, 2000, 4000)} for h in HYPS},
           "calibration": cal, "calibration_loo": cal_loo, "calibration_late": cal_late, "beside": beside}
    (out / "seeds.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({n: {"u": r["u_date"], "c": r["c_arm_dates"]} for n, r in per.items()}))
    print(json.dumps({"ref70": ref, "frozen_equal": res["ref70_frozen_equal"], "bands": res["bands"],
                      "windows": res["windows"], "early_top": {h: early(h)[-1] for h in HYPS}, "known_answer": ka,
                      "calibration": cal["summary"], "calibration_loo": cal_loo["summary"],
                      "calibration_late": {lt: c_["summary"] for lt, c_ in cal_late.items()}}, indent=1))
    return 0


def _jsonable(x):
    return json.loads(json.dumps(x))


# ---------------------------------------------------------------- the run's gate (U1)

def _units(model, tok, device, rng, u) -> dict:
    per = {}
    for p in V1:
        hs, _ = c.states(model, tok, p, device)
        per[p] = {"base": cs._drops(hs), "plants": cd._plants(hs, u, rng)}
    return per


def _flat(pops: dict, names: Sequence[str], w: str, key: str = None, dose: str = None) -> List[float]:
    return [x for n in names for p in V1
            for x in (pops[n][p]["base"][w] if key is None else pops[n][p]["plants"][key][dose])]


def gate_one(pops: dict, reinits: Sequence[str], step: str) -> dict:
    units = {pop: {"base": {w: _flat(pops, ns, w) for w in ("u", "c")},
                   "plants": {k: {f"{d:g}": _flat(pops, ns, None, k, f"{d:g}")
                                  for d in (cs.DOSES_A if k[0] == "A" else cd.DOSES_B)}
                              for k in ("A_u", "A_c", "B_u", "B_c")}}
             for pop, ns in (("reinit", reinits), ("trained", [step]))}
    try:
        bd = cd.b_dose(units["trained"]["base"]["u"], units["trained"]["plants"]["B_u"])
    except cd.DoseRefused as e:
        return {"b_dose": None, "refused": str(e), "arms": {}, "pass": False}
    g = cd.gate_arms(units, ARMS, bd)
    g["uncentred_null_rate"] = cs.rate(units["reinit"]["base"]["u"], REL_TOL)
    return g


def gate(a) -> int:
    """U1's gate. Needs `run --models seed0` into the same --out first: seed 0's records there must
    equal unit 3b's (`known_answer`), else refuse (#295, finding 4)."""
    _frozen()
    ka = known_answer(load_model(Path(a.out), "seed0"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoConfig, AutoTokenizer
    from p1d_cluster_ensemble.arch_null import reinit_model
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    cfg = AutoConfig.from_pretrained(MODELS["seed0"], revision="step0")
    rng = np.random.default_rng(SEED)
    u = rng.standard_normal(cfg.hidden_size)
    u /= np.linalg.norm(u)
    pops, reinits = {}, [f"reinit:{s}" for s in range(c.N_REINIT)]
    for name in reinits:
        model = reinit_model(cfg, int(name.split(":")[1])).to(device)
        pops[name] = _units(model, tok, device, rng, u)
        del model
    for s in GATE_STEPS:
        model = cs.load(MODELS["run"], s, device)
        pops[f"step{s}"] = _units(model, tok, device, rng, u)
        del model
        torch.cuda.empty_cache()
    per_step = {f"step{s}": gate_one(pops, reinits, f"step{s}") for s in GATE_STEPS}
    pf = json.loads(U3B_PREFLIGHT.read_text())
    null_now = {k: v["null_rate"] for k, v in per_step[f"step{GATE_STEPS[0]}"]["arms"].items()}
    null_3b = {k: pf["gate"]["arms"][k]["null_rate"] for k in null_now}
    if any(abs(null_now[k] - null_3b[k]) > 0.005 for k in null_now):
        raise C6Refused(f"the re-inits' rates {null_now} differ from unit 3b's {null_3b}")
    arms = {f"{t:.3g}": all(g["arms"].get(f"{t:.3g}", {}).get("pass", False) for g in per_step.values())
            for t in ARMS}
    u_ok = all(g["b_dose"] is not None for g in per_step.values())
    res = {"code": code_sha(), "meta": meta(torch), "seed0_known_answer_max_rel": ka,
           "per_step": per_step, "arms_gated": arms,
           "uncentred_read": u_ok, "readable": u_ok or any(arms.values()), "null_vs_3b": [null_now, null_3b]}
    (out / "gate.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"arms_gated": arms, "uncentred_read": u_ok,
                      "b_dose": {k: g["b_dose"] for k, g in per_step.items()}}, indent=1))
    return 0 if res["readable"] else 1


def read(a) -> int:
    ref = _frozen()
    out = Path(a.out)
    g = json.loads((out / "gate.json").read_text())
    if not g.get("readable"):
        raise C6Refused("gate.json: the run's gate has not passed")
    recs = load_model(out, "run")
    codes = {r["code"] for r in recs.values()}
    if codes != {g["code"]} or codes != {code_sha()}:
        raise C6Refused(f"the run's records ({sorted(codes)}), the gate ({g['code']}) and this code "
                        f"({code_sha()}) must be one version")
    vu = c.series(recs, "e1")
    vcs = {f"{t:.3g}": c.series(recs, "e1c", t) for t in ARMS}
    counts = {"u": count_reading({"1e-3": vu}, {"1e-3": g["uncentred_read"]}, "u", ref),
              "c": count_reading(vcs, g["arms_gated"], "c", ref)}
    res = {"code": code_sha(), "record_codes": sorted({r["code"] for r in recs.values()}), "ref70": ref,
           "early_first": {k: {h: {a_: r["early"] for a_, r in counts[k][h]["per_arm"].items()} for h in HYPS}
                           for k in counts},
           "counts": counts, "joint": joint(counts),
           "beside": {"V_u": vu.tolist(), "V_c": {k: v.tolist() for k, v in vcs.items()},
                      "c2_window_dates": {"u": date_on(vu, "step")["step"],
                                          "c": {k: date_on(v, "step")["step"] for k, v in vcs.items()}}}}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: {h: (counts[k][h]["dates"], counts[k][h]["status"]) for h in HYPS} for k in counts}))
    print(json.dumps(res["joint"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "seeds", "gate", "read"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True)
        if cmd == "run":
            sp.add_argument("--models", nargs="+", choices=sorted(MODELS), required=True)
    a = ap.parse_args(argv)
    return {"run": run, "seeds": seeds, "gate": gate, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
