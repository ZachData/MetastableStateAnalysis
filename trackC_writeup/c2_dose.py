"""
trackC_writeup/c2_dose.py — C2 unit 3b: the LR-dose arm on `pythia-70m` and `pythia-160m`
(`design-C2.md` Blocked 92 (iii), Blocked 98 (c) and 99 (a); "Unit 3b: the LR-dose arm" there).

The three sizes share the warmup (1430 steps, linear), the batch (so tokens per step) and the
data order; only the peak LR differs (1e-3, 6e-4, 3e-4). On Pythia's grid the frozen 92 (iii)
labels overlapped: a break that tracks the cumulative LR moves 70m's half-rise from 410m's
(128, 512] to (70, 280], which the grid reads as 128, 256 or 512, the same steps "step-locked"
predicts. So each size's two dates (98 (c): E₁'s break, and the token-centred cloud's) are read
against three schedules' predictions on the grid (99 (a), the user's, before any output):

- **step**: the date is 410m's (the warmup's timing, or the tokens seen; not told apart);
- **cumulative**: the date is where the summed LR equals 410m's at its date (t · √(3e-4 / lr));
- **instantaneous**: the date is where the LR itself equals 410m's at its date (t · 3e-4 / lr).

410m's dates are its 10 seeds' (`REF`); a grid date g stands for a crossing in (prev(g), g], and
a schedule predicts every grid step whose interval meets the scaled one (`predicted`;
`tools/math_checks/c2_dose_grid.py`). The reading is the set of schedules consistent with every
readable date (`joint`). Size (width, depth) is confounded throughout: description.

The centred count uses unit 3s's procedure per size: arms null-matched and B-matched re-derived
on that size's 3 re-inits and off-grid step 7000, beside 1e-4 and 3e-4; each arm gated as unit
3s's. Records as unit 3's (e1, e1c, e1r, cos per state), with that size's number of states.

Tier 1, description (nothing here is registered).
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_dose preflight --size 70m --out <dir>
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_dose run --out <dir> [--sizes 70m 160m]
    python -m trackC_writeup.c2_dose read --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force import a3
from trackA_force.a1a import REPO, STEPS, V1, prompts_hash
from trackA_force.a5_seeds import REL_TOL, TOKENIZER
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_seeds as cs

SIZES = {"70m": ("EleutherAI/pythia-70m", 1e-3), "160m": ("EleutherAI/pythia-160m", 6e-4)}
REF_LR = 3e-4                                   # pythia-410m's peak LR (numbers.json cfg.lr_410m)
WARMUP = 1430                                   # every size's (cfg.warmup_70m, cfg.warmup_160m)
#: 410m's grid dates over its 10 seeds: E₁'s break (A5, seeds 0–9: 512 ×8, 256 ×2) and the
#: centred half-rise at any of unit 3s's four arms (1000; seed 6 at 3e-4 reads 512, a tie).
REF = {"u": (256, 512), "c": (512, 1000)}
HYPS = ("step", "cumulative", "instantaneous")
EARLY = c.EARLY                                 # 92 (i): reported first; the unreadable check
FIXED_ARMS = cs.FIXED_ARMS                      # 1e-4, 3e-4, as unit 3s
#: per size: null-matched, 1e-4, 3e-4, B-matched, from that size's preflight (3 s.f.), frozen
#: before any scored record. None: not frozen, `run` and `read` refuse.
TAU_ARMS: Dict[str, Optional[Tuple[float, ...]]] = {"70m": (9.61e-5, 1e-4, 3e-4, 4.78e-3),
                                                   "160m": (1.25e-4, 1e-4, 3e-4, 8.22e-4)}
SCALE_STEP = cs.SCALE_STEP                      # 7000, off the grid
DOSES_B = cs.DOSES_B + (0.5,)                   # unit 3s's, and 0.5 for a size whose layers move more
B_SEEN_U = 0.4                                  # 100 (a): B's gate dose is the smallest the uncentred count sees at this share
SEED = 20261024


class DoseRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c2_dose.py", "trackC_writeup/c2_seeds.py", "trackC_writeup/c2_aniso.py",
              "trackA_force/a5_seeds.py", "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the predictions

def factor(hyp: str, lr: float, ref_lr: float = REF_LR) -> float:
    """The step at which a size of peak LR ``lr`` matches 410m's step t is factor · t (t ≤ WARMUP)."""
    if hyp == "step":
        return 1.0
    if hyp == "cumulative":
        return math.sqrt(ref_lr / lr)               # p t² / 2W equal
    if hyp == "instantaneous":
        return ref_lr / lr                          # p t / W equal
    raise ValueError(hyp)


def prev(g: int) -> int:
    i = STEPS.index(g)
    if i == 0:
        raise DoseRefused("step 0 has no interval: a half-rise at 0 is unreadable")
    return STEPS[i - 1]


def predicted(ref: Iterable[int], f: float) -> FrozenSet[int]:
    """The grid steps a schedule predicts: each ref date g stands for a crossing in (prev(g), g];
    scaled by f it is (f·prev(g), f·g], and the grid reads s when (prev(s), s] meets it."""
    out = set()
    for g in ref:
        lo, hi = f * prev(g), f * g
        out |= {s for s in STEPS[1:] if s > lo and prev(s) < hi}
    return frozenset(out)


def consistent(dates: Iterable[int], kind: str, lr: float) -> FrozenSet[str]:
    """The schedules that predict at least one of ``dates`` (a size's date, or its arms' dates when
    they disagree) for that size's LR."""
    ds = set(dates)
    return frozenset(h for h in HYPS if ds & predicted(REF[kind], factor(h, lr)))


# ---------------------------------------------------------------- pure: the gate (Blocked 100 (a))

def b_dose(base_u: Sequence[float], planted_u: Dict[str, Sequence[float]]) -> str:
    """Plant B's gate dose on one size: the smallest dose at which the uncentred count at D2's 1e-3
    turns ≥ B_SEEN_U of the units not already violating (0.25 on 410m, unit 3s's B_DOSE)."""
    base = cs.rate(base_u, REL_TOL)
    for k in sorted((k for k in planted_u if float(k) > 0), key=float):
        if cs.share(cs.rate(planted_u[k], REL_TOL), base) >= B_SEEN_U:
            return k
    raise DoseRefused(f"no plant-B dose up to {max(map(float, planted_u))} is seen by the uncentred count")


def gate_arms(units: dict, arms: Sequence[float], bd: str) -> dict:
    """`c2_seeds.gate_arms` with plant B read at the size's dose ``bd``; thresholds unchanged."""
    tr, nu = units["trained"], units["reinit"]
    out = {}
    for t in arms:
        base = cs.rate(tr["base"]["c"], t)
        b = cs.share(cs.rate(tr["plants"]["B_c"][bd], t), base)
        a = cs.share(cs.rate(tr["plants"]["A_c"][cs.A_DOSE], t), base)
        n = cs.rate(nu["base"]["c"], t)
        out[f"{t:.3g}"] = {"null_rate": n, "B_share": b, "A_share": a,
                           "pass": n <= cs.NULL_RATE and b >= cs.B_SEEN and a <= cs.A_LEAK,
                           "beside": {"base_rate_c": base, "base_rate_u": cs.rate(tr["base"]["u"], REL_TOL),
                                      "A_share_dose1": cs.share(cs.rate(tr["plants"]["A_c"]["1"], t), base),
                                      "B_share_0.25": cs.share(cs.rate(tr["plants"]["B_c"]["0.25"], t), base)}}
    return {"arms": out, "b_dose": bd, "pass": all(v["pass"] for v in out.values())}


# ---------------------------------------------------------------- pure: the reading

def date_reading(v: np.ndarray) -> dict:
    """One count's half-rise on its own plateau; unreadable if the plateau is 0 or a step 0–128
    count is ≥ half of it (92 (i))."""
    v = np.asarray(v, float)
    r = a3.d4a(v)
    early = {s: float(v[STEPS.index(s)]) for s in EARLY}
    p = r["plateau_median"]
    bad = p <= 0 or max(early.values()) >= p / 2
    return {"step": None if bad else r["step"], "plateau": p, "early": early, "unreadable": bad}


def size_reading(vu: np.ndarray, vcs: Dict[str, np.ndarray], gate: Dict[str, bool], lr: float) -> dict:
    """One size. The uncentred date; the centred date on each arm that passed its gate (the dates
    of the readable arms; unreadable if no arm is both gated and readable). Each date's
    consistent schedules."""
    u = date_reading(vu)
    arms = {k: {**date_reading(v), "gated": gate[k]} for k, v in vcs.items()}
    cdates = sorted({a["step"] for a in arms.values() if a["gated"] and not a["unreadable"]})
    out = {"uncentred": {**u, "consistent": None}, "centred": {"dates": cdates, "arms": arms,
                                                               "agree": len(cdates) == 1,
                                                               "unreadable": not cdates,
                                                               "consistent": None}}
    if not u["unreadable"]:
        out["uncentred"]["consistent"] = sorted(consistent([u["step"]], "u", lr))
    if cdates:
        out["centred"]["consistent"] = sorted(consistent(cdates, "c", lr))
    return out


def joint(per_size: Dict[str, dict]) -> dict:
    """The schedules consistent with every readable date, over sizes and the two counts."""
    sets, unread = [], []
    for s, r in per_size.items():
        for kind in ("uncentred", "centred"):
            if r[kind]["consistent"] is None:
                unread.append(f"{s} {kind}")
            else:
                sets.append(frozenset(r[kind]["consistent"]))
    if not sets:
        return {"survivors": [], "label": "unreadable", "readable_dates": 0, "unreadable": unread}
    surv = frozenset(HYPS).intersection(*sets)
    label = ("not told apart" if surv == frozenset(HYPS) else "neither" if not surv
             else "consistent with " + " and ".join(h for h in HYPS if h in surv))
    return {"survivors": [h for h in HYPS if h in surv], "label": label,
            "readable_dates": len(sets), "unreadable": unread}


# ---------------------------------------------------------------- model passes

def load(repo: str, step: int, device):
    return cs.load(repo, step, device)


def check_populated(rec: dict, n_states: int) -> None:
    """Refuse rather than degrade: n_states finite values per key, energies positive."""
    bad = [k for k in ("e1", "e1c", "e1r", "cos")
           if len(rec[k]) != n_states or not np.all(np.isfinite(rec[k]))
           or (k != "cos" and min(rec[k]) <= 0)]
    if bad:
        raise DoseRefused(f"{rec['prompt']}: not populated: {bad}")


def record(hs: Sequence[np.ndarray], ids: List[int], prompt: str, n_states: int) -> dict:
    rows = [c.state_row(h) for h in hs]
    rec = {"prompt": prompt, "n": len(ids), "ids_sha": hashlib.sha256(str(ids).encode()).hexdigest()[:12],
           **{k: [r[k] for r in rows] for k in ("e1", "e1c", "e1r", "cos")}}
    check_populated(rec, n_states)
    return rec


def rec_path(out: Path, size: str, step: int, prompt: str) -> Path:
    return out / size / "records" / f"step{step}_{prompt}.json"


def meta(torch) -> dict:
    return {"design": "trackC_writeup/design-C2.md", "code": code_sha(), "tau_arms": TAU_ARMS,
            "sizes": {k: list(v) for k, v in SIZES.items()}, "ref": REF, "ref_lr": REF_LR,
            "prompts_hash": prompts_hash(), "prompts": list(V1), "tokenizer": list(TOKENIZER),
            "dtype": "float32", "device": torch.cuda.get_device_name(0), "tf32": False,
            "torch": torch.__version__, "hf_home": os.environ.get("HF_HOME")}


def _frozen(size: str) -> Tuple[float, ...]:
    if TAU_ARMS.get(size) is None:
        raise DoseRefused(f"TAU_ARMS[{size!r}] is not set: the rule is not frozen")
    return TAU_ARMS[size]


def run(a) -> int:
    for s in a.sizes:
        _frozen(s)
    os.environ["HF_HUB_OFFLINE"] = "1"            # fetched before; nothing is downloaded here
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = meta(torch)
    (out / "plan.json").write_text(json.dumps({**m, "run_sizes": list(a.sizes)}, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    first, t0 = True, time.time()
    for size in a.sizes:
        repo = SIZES[size][0]
        rec_path(out, size, 0, V1[0]).parent.mkdir(parents=True, exist_ok=True)
        for step in STEPS:
            todo = [p for p in V1 if not rec_path(out, size, step, p).exists()]
            if not todo:
                continue
            model = load(repo, step, device)
            n_states = model.config.num_hidden_layers + 1
            for p in todo:
                hs, ids = c.states(model, tok, p, device)
                rec = record(hs, ids, p, n_states)
                rec.update(size=size, repo=repo, step=step, revision=f"step{step}", code=m["code"])
                rec_path(out, size, step, p).write_text(json.dumps(rec) + "\n")
                if first:
                    print(f"first record populated: {size} step {step} {p}, n {rec['n']}, "
                          f"states {n_states}, e1[0] {rec['e1'][0]:.6f}, e1c[0] {rec['e1c'][0]:.6f}, "
                          f"cos[-1] {rec['cos'][-1]:.4f}", flush=True)
                    first = False
            del model
            torch.cuda.empty_cache()
        print(f"{size} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


# ---------------------------------------------------------------- reading

def load_size(d: Path, size: str) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in STEPS:
        for p in V1:
            f = d / f"step{s}_{p}.json"
            if not f.exists():
                raise DoseRefused(f"{f}: missing")
            r = json.loads(f.read_text())
            if r.get("size") != size:
                raise DoseRefused(f"{f}: size {r.get('size')}, not {size}")
            recs[(s, p)] = r
    return recs


def read(a) -> int:
    out = Path(a.out)
    per, beside, codes = {}, {}, {}
    for size, (repo, lr) in SIZES.items():
        arms = _frozen(size)
        pf = json.loads((Path(a.preflight) / size / "preflight_plants.json").read_text())
        if [float(f"{t:.3g}") for t in pf["arms"]] != list(arms):
            raise DoseRefused(f"{size}: the preflight's arms {pf['arms']} are not the frozen {arms}")
        gate = {f"{t:.3g}": pf["gate"]["arms"][f"{t:.3g}"]["pass"] for t in pf["arms"]}
        recs = load_size(out / size / "records", size)
        codes[size] = sorted({r["code"] for r in recs.values()})
        vu = c.series(recs, "e1")
        vcs = {f"{t:.3g}": c.series(recs, "e1c", t) for t in arms}
        per[size] = size_reading(vu, vcs, gate, lr)
        n_states = len(next(iter(recs.values()))["e1"])
        beside[size] = {"V_u": vu.tolist(), "V_c": {k: v.tolist() for k, v in vcs.items()},
                        "V_c_at_1e-3": c.series(recs, "e1c", 1e-3).tolist(),
                        "V_raw_centred": {k: c.series(recs, "e1r", t).tolist() for k, t in zip(vcs, arms)},
                        "units_per_step": len(V1) * (n_states - 1),
                        "mean_cos_layer_mean": [float(np.mean([recs[(st, p)]["cos"] for p in V1]))
                                                for st in STEPS],
                        "predicted": {kind: {h: sorted(predicted(REF[kind[0]], factor(h, lr))) for h in HYPS}
                                      for kind in ("uncentred", "centred")}}
    if len({tuple(v) for v in codes.values()}) != 1:
        raise DoseRefused(f"records from more than one code version: {codes}")
    res = {"code": code_sha(), "record_codes": codes, "tau_arms": TAU_ARMS, "ref": REF,
           "early_first": {s: {"uncentred": r["uncentred"]["early"],
                               "centred": {k: a["early"] for k, a in r["centred"]["arms"].items()}}
                           for s, r in per.items()},
           "per_size": per, "joint": joint(per), "beside": beside}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["early_first"]))
    print(json.dumps({s: {"u": r["uncentred"]["step"], "c": r["centred"]["dates"],
                          "u_cons": r["uncentred"]["consistent"], "c_cons": r["centred"]["consistent"]}
                      for s, r in per.items()}))
    print(json.dumps(res["joint"]))
    return 0


# ---------------------------------------------------------------- the preflight (before the freeze)

def _plants(hs: Sequence[np.ndarray], u: np.ndarray, rng: np.random.Generator) -> dict:
    """`c2_seeds._plants` over this unit's DOSES_B."""
    base = {w: [c._e(h, w) for h in hs] for w in ("u", "c")}
    out: Dict[str, Dict[str, List[float]]] = {k: {} for k in ("A_u", "A_c", "B_u", "B_c")}
    for d in cs.DOSES_A:
        for w in ("u", "c"):
            out[f"A_{w}"][f"{d:g}"] = []
        for t in range(len(hs) - 1):
            hp = c.plant_common(hs[t], d, u)
            for w in ("u", "c"):
                e0 = c._e(hp, w)
                out[f"A_{w}"][f"{d:g}"].append((e0 - base[w][t + 1]) / abs(e0))
    for d in DOSES_B:
        for w in ("u", "c"):
            out[f"B_{w}"][f"{d:g}"] = []
        for t in range(len(hs) - 1):
            hp = c.plant_spread(hs[t + 1], d, rng)
            for w in ("u", "c"):
                out[f"B_{w}"][f"{d:g}"].append((base[w][t] - c._e(hp, w)) / abs(base[w][t]))
    return out


def preflight(a) -> int:
    """As unit 3s's, on one size: 3 re-inits of its config and its off-grid step 7000; plants A and
    B through both counts; the null-matched and B-matched arms derived; each arm gated. Writes the
    design-preflight tool's input for the centred statistic and for the uncentred one (the
    uncentred count was never gated off 410m)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoConfig, AutoTokenizer
    from p1d_cluster_ensemble.arch_null import init_summary, reinit_model
    size = a.size
    repo = SIZES[size][0]
    out = Path(a.out) / size
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    cfg = AutoConfig.from_pretrained(repo, revision="step0")
    rng = np.random.default_rng(SEED)
    u = rng.standard_normal(cfg.hidden_size)
    u /= np.linalg.norm(u)
    pops: Dict[str, dict] = {}
    names = [f"reinit:{s}" for s in range(c.N_REINIT)] + [f"step{SCALE_STEP}"]
    for name in names:
        if name.startswith("reinit"):
            model = reinit_model(cfg, int(name.split(":")[1])).to(device)
            init = init_summary(model)
        else:
            model, init = load(repo, SCALE_STEP, device), None
        per = {}
        for p in V1:
            hs, _ = c.states(model, tok, p, device)
            per[p] = {"base": cs._drops(hs), "plants": _plants(hs, u, rng)}
        pops[name] = {"per_prompt": per, "init_summary": init}
        del model
        torch.cuda.empty_cache()
        print(f"{size} {name} done", flush=True)

    reinits, trained = names[:-1], names[-1:]

    def flat(ns, w, key=None, dose=None):
        return [x for n in ns for p in V1
                for x in (pops[n]["per_prompt"][p]["base"][w] if key is None
                          else pops[n]["per_prompt"][p]["plants"][key][dose])]

    units = {pop: {"base": {w: flat(ns, w) for w in ("u", "c")},
                   "plants": {k: {f"{d:g}": flat(ns, None, k, f"{d:g}")
                                  for d in (cs.DOSES_A if k[0] == "A" else DOSES_B)}
                              for k in ("A_u", "A_c", "B_u", "B_c")}}
             for pop, ns in (("reinit", reinits), ("trained", trained))}
    (out / "preflight_units.json").write_text(json.dumps(units) + "\n")
    tr, nu = units["trained"], units["reinit"]
    bd = b_dose(tr["base"]["u"], tr["plants"]["B_u"])
    b_target = cs.share(cs.rate(tr["plants"]["B_u"][bd], REL_TOL), cs.rate(tr["base"]["u"], REL_TOL))
    derived = {"b_dose": bd,
               "null_matched": cs.null_matched_tol(nu["base"]["u"], nu["base"]["c"]),
               "b_matched": cs.b_matched_tol(tr["base"]["c"], tr["plants"]["B_c"][bd], b_target)}
    again = (float(f"{derived['null_matched']:.3g}"), *FIXED_ARMS, float(f"{derived['b_matched']:.3g}"))
    frozen = TAU_ARMS.get(size)
    if frozen and tuple(frozen) != again:
        raise DoseRefused(f"{size}: the preflight derives {again}, not the frozen {frozen}")
    arms = frozen or again
    gate = gate_arms(units, arms, bd)
    gate_u = {"null_rate": cs.rate(nu["base"]["u"], REL_TOL), "base_rate": cs.rate(tr["base"]["u"], REL_TOL),
              "B_share": b_target, "b_dose": bd,
              "A_share": cs.share(cs.rate(tr["plants"]["A_u"][cs.A_DOSE], REL_TOL), cs.rate(tr["base"]["u"], REL_TOL))}
    for w, key in (("c", "B_c"), ("u", "B_u")):
        tool_in = {"statistic": f"{'centred' if w == 'c' else 'uncentred'} E_1 relative drop per (prompt, "
                                f"transition), (E_t − E_t+1) / |E_t|, pythia-{size} (c2_dose)",
                   "direction": "greater", "nulls": {"reinit": nu["base"][w]}, "moved_marks": [],
                   "trained": tr["base"][w], "trained_step": str(SCALE_STEP),
                   "scored_steps": [str(s) for s in STEPS],
                   "planted": {f"{d:g}": nu["plants"][key][f"{d:g}"] for d in DOSES_B},
                   "bounds": None, "decision_p": 0.05}
        (out / f"preflight_input_{w}.json").write_text(json.dumps(tool_in) + "\n")
    rec = {"code": code_sha(), "meta": meta(torch), "size": size, "repo": repo, "derived": derived,
           "b_target": b_target, "arms": list(arms), "frozen": frozen is not None, "gate": gate,
           "uncentred_at_1e-3": gate_u, "units_per_population": {p: len(units[p]["base"]["c"]) for p in units},
           "scales": {pop: {"s_u": cs.scale(units[pop]["base"]["u"]), "s_c": cs.scale(units[pop]["base"]["c"])}
                      for pop in units},
           "doses_A": list(cs.DOSES_A), "doses_B": list(DOSES_B), "u_seed": SEED,
           "init_summary": {n: pops[n]["init_summary"] for n in reinits}}
    (out / "preflight_plants.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"derived": derived, "arms": list(arms), "gate": gate, "uncentred": gate_u}, indent=1))
    return 0 if gate["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("preflight", "run", "read"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True)
        if cmd == "preflight":
            sp.add_argument("--size", choices=sorted(SIZES), required=True)
        if cmd == "run":
            sp.add_argument("--sizes", nargs="+", choices=sorted(SIZES), default=list(SIZES))
        if cmd == "read":
            sp.add_argument("--preflight", required=True)
    a = ap.parse_args(argv)
    return {"preflight": preflight, "run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
