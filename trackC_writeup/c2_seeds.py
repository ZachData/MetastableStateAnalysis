"""
trackC_writeup/c2_seeds.py — C2 unit 3s: the centred count on the 9 PolyPythias seeds
(`design-C2.md` Blocked 96 (b), "Unit 3s: the centred count on the seeds").

Unit 3 (`c2_aniso.py`) read seed 0 with the centred count at D2 (a)'s 1e-3, a tolerance about 30×
stricter on the centred E₁ than on the uncentred one (`status-C.md` "C2 unit 3", the scale row).
96 (b) asked for one centred tolerance on the centred energy's own scale, gated by a plant read
through both counts. The preflight refused it: no tolerance makes plant B (a change both counts
should see) read alike through both, because the centred count's dose response is steeper
(`median_matched_tol` is kept to show it). So the rule reads **dates only, each on its own
count's plateau, at four centred tolerances** (Blocked 97 (a), the user's, in session):

- **the arms** (`TAU_ARMS`): the null-matched τ (`null_matched_tol`: the re-inits' centred false
  positive rate equal to the uncentred one at 1e-3), 1e-4, 3e-4, and the B-matched τ
  (`b_matched_tol`: plant B at dose 0.25 seen equally by both counts at step 7000), from the
  preflight on seed 0's off-grid step 7000 and 3 re-inits, frozen before any seed record;
- **the gate** (`gate_arms`), per arm: the re-inits' centred rate ≤ 5 %; plant B at 0.25 turns
  ≥ 0.4 of the units not already violating; plant A (a common vector) at 0.5 turns ≤ 0.1;
- **per seed** (`seed_reading`): records as unit 3's (e1, e1c, e1r, cos per state), e1 checked
  against A5's records for that seed; event 1, the centred half-rise (`a3.d4a`) is later than the
  seed's uncentred break; event 2, it is at 1000. A seed carries an event only if it holds at all
  four arms; unreadable at any arm reads both False.

Tier 1, description (nothing here is registered).
    python -m trackC_writeup.c2_seeds preflight --out <dir>
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_seeds run --out <dir> [--seeds 1 2 ...]
    python -m trackC_writeup.c2_seeds read --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force import a3
from trackA_force.a1a import DATA, REPO, STEPS, V1, prompts_hash
from trackA_force.a5_seeds import REL_TOL, TOKENIZER, repo_for
from trackC_writeup import c2_aniso as c

SEEDS = tuple(range(1, 10))                     # the 9 PolyPythias seeds; seed 0 is read beside
EARLY = c.EARLY                                 # 92 (i): reported first; the unreadable check
DATE = 1000                                     # event 2: the centred half-rise on seed 0 (unit 3)
REPRODUCED, SEED0_ONLY = 7, 2                   # tally: ≥ 7 of 9 "reproduced", ≤ 2 "seed 0's only"
SCALE_STEP = 7000                               # off the grid; unit 3's preflight step
DOSES_A = (0.0, 0.25, 0.5, 1.0)                 # plant A, unit 3's doses (× median token norm)
DOSES_B = (0.0, 0.01, 0.02, 0.05, 0.1, 0.25)    # plant B, small enough to sit between 0 and 1
B_DOSE, A_DOSE = "0.25", "0.5"                  # the gate's doses (B-matching uses B_DOSE too)
B_SEEN, A_LEAK, NULL_RATE = 0.4, 0.1, 0.05      # the gate, per arm (set after the v0 preflight; 97 (a))
FIXED_ARMS = (1e-4, 3e-4)                       # unit 3's 1e-4, and one between it and B-matched
TAU_ARMS = (7.55e-5, 1e-4, 3e-4, 7.58e-4)       # null-matched, fixed, fixed, B-matched (preflight v1, 3 s.f.)
A5_SEEDS = "trackA/a5/2026-10-20-u2"            # A5's seed records and score.json (code 7ae0bff62ada)
C2U3 = "trackC/c2u3/2026-10-22-run"             # unit 3's seed-0 records (code d90bb06b7910)
SEED = 20261023


class SeedsRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c2_seeds.py", "trackC_writeup/c2_aniso.py",
              "trackA_force/a5_seeds.py", "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure

def scale(r: Sequence[float]) -> float:
    """The median |relative change| per unit: an energy's own layer-to-layer scale."""
    return float(np.median(np.abs(np.asarray(r, float))))


def median_matched_tol(r_u: Sequence[float], r_c: Sequence[float], tol_u: float = REL_TOL) -> float:
    """96 (b)'s τ: the uncentred tolerance as the same multiple of the centred energy's scale.
    Refused by the v0 preflight (plant B 2–5× stronger centred); kept beside, as the reason."""
    su, sc = scale(r_u), scale(r_c)
    if su <= 0 or sc <= 0:
        raise SeedsRefused(f"a zero scale (uncentred {su:.3g}, centred {sc:.3g})")
    return tol_u * sc / su


def rate(vals: Sequence[float], tol: float) -> float:
    return float(np.mean(np.asarray(vals, float) > tol))


def share(r: float, base: float) -> float:
    """The planted part of a violation rate: (rate − base) / (1 − base), the share of the
    units not already violating that the plant turned."""
    return (r - base) / (1 - base) if base < 1 else 0.0


def null_matched_tol(null_u: Sequence[float], null_c: Sequence[float], tol_u: float = REL_TOL) -> float:
    """τ with the null's centred violation rate equal to its uncentred rate at ``tol_u``."""
    return float(np.quantile(np.asarray(null_c, float), 1 - rate(null_u, tol_u)))


def b_matched_tol(base_c: Sequence[float], planted_c: Sequence[float], target: float,
                  lo: float = 1e-7, hi: float = 1e-2) -> float:
    """τ at which plant B's centred share (``planted_c`` against ``base_c``) equals ``target``,
    the uncentred share at 1e-3; bisection in log τ (the share falls as τ rises)."""
    def sh(t):
        return share(rate(planted_c, t), rate(base_c, t))
    if not sh(lo) >= target >= sh(hi):
        raise SeedsRefused(f"B's centred share does not cross {target:.3g} in [{lo:g}, {hi:g}]")
    for _ in range(80):
        mid = float(np.sqrt(lo * hi))
        lo, hi = (mid, hi) if sh(mid) > target else (lo, mid)
    return float(np.sqrt(lo * hi))


def gate_arms(units: dict, arms: Sequence[float]) -> dict:
    """Per arm, on the preflight's per-unit drops (``units[pop]['base'|'plants']``): the re-inits'
    centred rate ≤ NULL_RATE; at step 7000 plant B at B_DOSE turns ≥ B_SEEN of the units not
    already violating, plant A at A_DOSE ≤ A_LEAK. Plant A at dose 1 and B at 0.1 beside."""
    tr, nu = units["trained"], units["reinit"]
    out = {}
    for t in arms:
        base = rate(tr["base"]["c"], t)
        b = share(rate(tr["plants"]["B_c"][B_DOSE], t), base)
        a = share(rate(tr["plants"]["A_c"][A_DOSE], t), base)
        n = rate(nu["base"]["c"], t)
        out[f"{t:.3g}"] = {"null_rate": n, "B_share": b, "A_share": a,
                           "pass": n <= NULL_RATE and b >= B_SEEN and a <= A_LEAK,
                           "beside": {"base_rate_c": base, "base_rate_u": rate(tr["base"]["u"], REL_TOL),
                                      "A_share_dose1": share(rate(tr["plants"]["A_c"]["1"], t), base),
                                      "B_share_0.1": share(rate(tr["plants"]["B_c"]["0.1"], t), base)}}
    return {"arms": out, "pass": all(v["pass"] for v in out.values())}


def seed_reading(vu: np.ndarray, vcs: Dict[str, np.ndarray]) -> dict:
    """One seed. ``vu`` A5's V; ``vcs`` the centred V per arm. Per arm the centred half-rise on its
    own plateau; unreadable if that plateau is 0 or a step 0–128 count is ≥ half of it. Event 1:
    the centred half-rise later than the uncentred break; event 2: at DATE. A seed carries an event
    only if it holds at every arm; unreadable at any arm reads both False, flagged."""
    vu = np.asarray(vu, float)
    ru = a3.d4a(vu)
    if ru["step"] is None:
        raise SeedsRefused("the uncentred plateau is 0: these are not the records A5 read")
    b = ru["step"]
    arms = {}
    for k, vc in vcs.items():
        vc = np.asarray(vc, float)
        rc = a3.d4a(vc)
        early = {s: float(vc[STEPS.index(s)]) for s in EARLY}
        pc = rc["plateau_median"]
        bad = pc <= 0 or max(early.values()) >= pc / 2
        arms[k] = {"centred_step": rc["step"], "plateau_centred": pc, "early_centred": early,
                   "V_c_at_break": float(vc[STEPS.index(b)]), "unreadable": bad,
                   "event1": (not bad) and rc["step"] > b, "event2": (not bad) and rc["step"] == DATE}
    unreadable = any(a["unreadable"] for a in arms.values())
    return {"break_step": b, "V_u_at_break": float(vu[STEPS.index(b)]),
            "plateau_uncentred": ru["plateau_median"], "arms": arms, "unreadable": unreadable,
            "event1": all(a["event1"] for a in arms.values()),
            "event2": all(a["event2"] for a in arms.values())}


def tally_label(k: int) -> str:
    return "reproduced" if k >= REPRODUCED else "seed 0's only" if k <= SEED0_ONLY else "mixed"


def tally(per_seed: Dict[int, dict]) -> dict:
    k1 = int(sum(r["event1"] for r in per_seed.values()))
    k2 = int(sum(r["event2"] for r in per_seed.values()))
    return {"K1": k1, "K2": k2, "n": len(per_seed), "event1": tally_label(k1), "event2": tally_label(k2),
            "unreadable": sorted(s for s, r in per_seed.items() if r["unreadable"])}


# ---------------------------------------------------------------- model passes

def load(repo: str, step: int, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    m = from_pretrained_eager(GPTNeoXForCausalLM, repo, revision=f"step{step}",
                              torch_dtype=torch.float32).to(device).eval()
    if next(m.parameters()).dtype != torch.float32:
        raise SeedsRefused(f"{repo} step {step}: weights are not float32")
    return m


def rec_path(out: Path, seed: int, step: int, prompt: str) -> Path:
    return out / f"seed{seed}" / "records" / f"step{step}_{prompt}.json"


def meta(torch) -> dict:
    return {"design": "trackC_writeup/design-C2.md", "code": code_sha(), "tau_arms": TAU_ARMS,
            "prompts_hash": prompts_hash(), "prompts": list(V1), "tokenizer": list(TOKENIZER),
            "dtype": "float32", "device": torch.cuda.get_device_name(0), "tf32": False,
            "torch": torch.__version__, "hf_home": os.environ.get("HF_HOME")}


def _frozen() -> Tuple[float, ...]:
    if TAU_ARMS is None:
        raise SeedsRefused("TAU_ARMS is not set: the rule is not frozen")
    return TAU_ARMS


def run(a) -> int:
    _frozen()
    os.environ["HF_HUB_OFFLINE"] = "1"            # the seeds' 18 steps are cached; nothing is downloaded
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = meta(torch)
    (out / "plan.json").write_text(json.dumps({**m, "seeds": list(a.seeds)}, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    first, t0 = True, time.time()
    for seed in a.seeds:
        repo, ref_dir = repo_for(seed), DATA / A5_SEEDS / f"seed{seed}" / "records"
        rec_path(out, seed, 0, V1[0]).parent.mkdir(parents=True, exist_ok=True)
        for step in STEPS:
            todo = [p for p in V1 if not rec_path(out, seed, step, p).exists()]
            if not todo:
                continue
            model = load(repo, step, device)
            for p in todo:
                hs, ids = c.states(model, tok, p, device)
                rec = c.record(hs, ids, p)
                ref = json.loads((ref_dir / f"step{step}_{p}.json").read_text())
                if rec["ids_sha"] != ref["ids_sha"]:
                    raise SeedsRefused(f"seed {seed} step {step} {p}: tokens differ from A5's record")
                dev = float(np.max(np.abs(np.subtract(rec["e1"], ref["e1"])) / np.asarray(ref["e1"])))
                if dev > c.ENERGY_TOL:
                    raise SeedsRefused(f"seed {seed} step {step} {p}: e1 off A5's record by {dev:.2g}")
                rec.update(seed=seed, repo=repo, step=step, revision=f"step{step}", code=m["code"],
                           e1_dev_a5=dev)
                rec_path(out, seed, step, p).write_text(json.dumps(rec) + "\n")
                if first:
                    print(f"first record populated: seed {seed} step {step} {p}, n {rec['n']}, "
                          f"e1 dev {dev:.1e}, e1c[0] {rec['e1c'][0]:.6f}, cos[12] {rec['cos'][12]:.4f}",
                          flush=True)
                    first = False
            del model
            torch.cuda.empty_cache()
        print(f"seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


# ---------------------------------------------------------------- reading

def load_seed(d: Path, seed: Optional[int]) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in STEPS:
        for p in V1:
            f = d / f"step{s}_{p}.json"
            if not f.exists():
                raise SeedsRefused(f"{f}: missing")
            r = json.loads(f.read_text())
            if seed is not None and r.get("seed") != seed:
                raise SeedsRefused(f"{f}: seed {r.get('seed')}, not {seed}")
            recs[(s, p)] = r
    return recs


def read(a) -> int:
    arms = _frozen()
    out = Path(a.out)
    score = json.loads((DATA / A5_SEEDS / "score.json").read_text())
    sets = {s: load_seed(out / f"seed{s}" / "records", s) for s in SEEDS}
    sets[0] = load_seed(DATA / C2U3 / "records", None)
    codes = {s: sorted({r["code"] for r in recs.values()}) for s, recs in sets.items()}
    if len({tuple(v) for s, v in codes.items() if s != 0}) != 1:
        raise SeedsRefused(f"seed records from more than one code version: {codes}")
    per, rows_beside = {}, {}
    for s in (0,) + SEEDS:
        recs = sets[s]
        vu = c.series(recs, "e1")
        if s != 0 and vu.tolist() != score["notes"][str(s)]["V"]:
            raise SeedsRefused(f"seed {s}: V from e1 is not A5's ({vu.tolist()})")
        vcs = {f"{t:.3g}": c.series(recs, "e1c", t) for t in arms}
        r = seed_reading(vu, vcs)
        rows = score["notes"][str(s)]["d10"]["rows"] if s != 0 else {}
        rows_beside[s] = {"V_u": vu.tolist(), "V_c": {k: v.tolist() for k, v in vcs.items()},
                          "V_c_at_1e-3": c.series(recs, "e1c", 1e-3).tolist(),
                          "V_raw_centred": {k: c.series(recs, "e1r", t).tolist()
                                            for k, t in zip(vcs, arms)},
                          "mean_cos_layer_mean": [float(np.mean([recs[(st, p)]["cos"] for p in V1]))
                                                  for st in STEPS],
                          "prev_step": rows.get("prev", {}).get("step"),
                          "ind_step": rows.get("ind", {}).get("step")}
        per[s] = r
    seed0 = per.pop(0)                             # the development set: beside, not in the tally
    res = {"code": code_sha(), "record_codes": codes, "tau_arms": list(arms), "seeds": list(SEEDS),
           "per_seed": per, "tally": tally(per), "seed0_development": seed0, "beside": rows_beside}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({s: {"break": r["break_step"], "unreadable": r["unreadable"], "e1": r["event1"],
                          "e2": r["event2"], "dates": [a["centred_step"] for a in r["arms"].values()]}
                      for s, r in per.items()}))
    print(json.dumps(res["tally"]))
    return 0


# ---------------------------------------------------------------- the preflight (before the freeze)

def _drops(hs: Sequence[np.ndarray]) -> Dict[str, List[float]]:
    return {w: c.rel_drops([c._e(h, w) for h in hs]).tolist() for w in ("u", "c")}


def _plants(hs: Sequence[np.ndarray], u: np.ndarray, rng: np.random.Generator) -> dict:
    """Per dose, each transition planted on its own, its relative drop read through both energies:
    A (a common vector on the earlier state), B (token-independent spread on the later)."""
    base = {w: [c._e(h, w) for h in hs] for w in ("u", "c")}
    out: Dict[str, Dict[str, List[float]]] = {k: {} for k in ("A_u", "A_c", "B_u", "B_c")}
    for d in DOSES_A:
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
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoConfig, AutoTokenizer
    from p1d_cluster_ensemble.arch_null import init_summary, reinit_model
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    rng = np.random.default_rng(SEED)
    u = rng.standard_normal(1024)
    u /= np.linalg.norm(u)
    pops: Dict[str, dict] = {}
    names = [f"reinit:{s}" for s in range(c.N_REINIT)] + [f"step{SCALE_STEP}"]
    for name in names:
        if name.startswith("reinit"):
            cfg = AutoConfig.from_pretrained(c.MODEL, revision="step0")
            model = reinit_model(cfg, int(name.split(":")[1])).to(device)
            init = init_summary(model)
        else:
            model, init = load(c.MODEL, SCALE_STEP, device), None
        per = {}
        for p in V1:
            hs, _ = c.states(model, tok, p, device)
            per[p] = {"base": _drops(hs), "plants": _plants(hs, u, rng)}
        pops[name] = {"per_prompt": per, "init_summary": init}
        del model
        torch.cuda.empty_cache()
        print(f"{name} done", flush=True)

    reinits, trained = names[:-1], names[-1:]

    def flat(ns, w, key=None, dose=None):
        return [x for n in ns for p in V1
                for x in (pops[n]["per_prompt"][p]["base"][w] if key is None
                          else pops[n]["per_prompt"][p]["plants"][key][dose])]

    units = {pop: {"base": {w: flat(ns, w) for w in ("u", "c")},
                   "plants": {k: {f"{d:g}": flat(ns, None, k, f"{d:g}")
                                  for d in (DOSES_A if k[0] == "A" else DOSES_B)}
                              for k in ("A_u", "A_c", "B_u", "B_c")}}
             for pop, ns in (("reinit", reinits), ("trained", trained))}
    (out / "preflight_units.json").write_text(json.dumps(units) + "\n")
    tr, nu = units["trained"], units["reinit"]
    b_target = share(rate(tr["plants"]["B_u"][B_DOSE], REL_TOL), rate(tr["base"]["u"], REL_TOL))
    derived = {"null_matched": null_matched_tol(nu["base"]["u"], nu["base"]["c"]),
               "b_matched": b_matched_tol(tr["base"]["c"], tr["plants"]["B_c"][B_DOSE], b_target),
               "median_matched_refused": median_matched_tol(tr["base"]["u"], tr["base"]["c"])}
    arms = TAU_ARMS or (derived["null_matched"], *FIXED_ARMS, derived["b_matched"])
    again = (float(f"{derived['null_matched']:.3g}"), *FIXED_ARMS, float(f"{derived['b_matched']:.3g}"))
    if TAU_ARMS and tuple(TAU_ARMS) != again:
        raise SeedsRefused(f"the preflight derives {again}, not the frozen {TAU_ARMS}")
    gate = gate_arms(units, arms)
    tool_in = {"statistic": "centred E_1 relative drop per (prompt, transition), (E_t − E_t+1) / |E_t| "
                            "on the token-centred, re-normalised cloud (c2_aniso)",
               "direction": "greater",
               "nulls": {"reinit": nu["base"]["c"]},
               "moved_marks": [],
               "trained": tr["base"]["c"],
               "trained_step": str(SCALE_STEP),
               "scored_steps": [str(s) for s in STEPS],
               "planted": {f"{d:g}": nu["plants"]["B_c"][f"{d:g}"] for d in DOSES_B},
               "bounds": None,
               "decision_p": 0.05}
    (out / "preflight_input.json").write_text(json.dumps(tool_in) + "\n")
    rec = {"code": code_sha(), "meta": meta(torch), "derived": derived, "b_target": b_target,
           "arms": list(arms), "frozen": TAU_ARMS is not None, "gate": gate,
           "scales": {pop: {"s_u": scale(units[pop]["base"]["u"]), "s_c": scale(units[pop]["base"]["c"])}
                      for pop in units},
           "doses_A": list(DOSES_A), "doses_B": list(DOSES_B), "u_seed": SEED,
           "init_summary": {n: pops[n]["init_summary"] for n in reinits}}
    (out / "preflight_plants.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"derived": derived, "arms": list(arms), "gate": gate}, indent=1))
    return 0 if gate["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("preflight", "run", "read"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True)
        if cmd == "run":
            sp.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args(argv)
    return {"preflight": preflight, "run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
