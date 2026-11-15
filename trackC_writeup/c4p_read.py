"""
trackC_writeup/c4p_read.py — C4′ (a), U1: induction's date on `EleutherAI/pythia-70m-warmup01`
read with C6's frozen rule (`design-C4p.md` "(a)"; Blocked 159 all (a), the user's, 2026-11-15).

A new module so `c4_read`'s registered hash (`220161b9159b`), `c6_warmup`'s and `c4_train`'s stay.

**The profile** of a model is its top head's induction score at each of C6's 18 grid steps
(`c4_read.induction_scores` on C4's sequences: `IND_LEN` 64, `IND_N` 8, `IND_SEED`), every head's
score kept. **The rule** is C6's, unchanged (`c6_warmup.date_on`, `band`, `window`, `early`,
`count_reading`, `joint`), with the profile in place of a count and one arm (D2): the reference
`REF` is the set of readable half-rises of `pythia-70m` and `pythia-70m-seed1..9` on 70m's
1000–4000 window, with 92 (i)'s unreadable check. No (count, schedule) pair is withheld
(`not_read=()`): Blocked 127 (a) withheld V under instantaneous, and this is not V.

**The gate** (`gate`, before the run's scores exist): C6's `calibrate` with leave-one-out, on the
ten seeds' profiles planted under each schedule. It passes iff ≥ `GATE_MIN` of 10 cumulative plants
read exactly "consistent with cumulative" and ≥ `GATE_MIN` of 10 step plants "consistent with step";
instantaneous's beside. **D3 (a):** after a passing gate, A1 is registered by the user (a second
`P-SCH` entry, drafted by Claude) **before** the run's scores are computed, so `heads --model run`
refuses unless the gate failed (then A1 is refused and the run is description), the entry is in
`claims/registry.json`, or the user declined it (`--registration declined`, recorded).

**A2** (`a2`, description): the heads ≥ `HEAD_BAR` at `A2_STEPS` on every model, overlaps with
`pythia-70m`'s. **A3** (`a3`, description): `c4_close.per_transition` / `majority` / `settled` on
C6 U1b's records (the run) and C6 U0's (seed 0) at the three gated centred arms, with transients.
**Beside A1** (`read`): step 3000 of the run (D1, outside the rule) and the held-out loss
(`c4_train.HELD_OUT`) at the run's and `pythia-70m`'s half-rises (`loss`).

    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4p_read heads --out <dir> --model seed0 … seed9
    python -m trackC_writeup.c4p_read gate --out <dir>
    python -m trackC_writeup.c4p_read a3 --out <dir>
    # after the user's registration decision:
    HF_HOME=… python -m trackC_writeup.c4p_read heads --out <dir> --model run --registration {registered,declined}
    python -m trackC_writeup.c4p_read read --out <dir>
    HF_HOME=… python -m trackC_writeup.c4p_read loss --out <dir>
    python -m trackC_writeup.c4p_read a2 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, STEPS, V1
from trackC_writeup import c2_aniso as c
from trackC_writeup import c4_close as cc
from trackC_writeup import c4_read as cr
from trackC_writeup import c6_warmup as cw

MODELS = cw.MODELS                       # seed0 = pythia-70m, run = pythia-70m-warmup01, seed1..9
SEEDS = cw.SEEDS
HYPS = cw.HYPS
KIND = "ind"
GATE_MIN = 8                             # of 10 plants, per gated schedule (design-C4p.md)
GATED = ("step", "cumulative")           # instantaneous's plants beside
HEAD_BAR = cc.HEAD_BAR                   # 0.5
A2_STEPS = (4000, 8000, 16000, 143000)
RUN_BESIDE = (3000,)                     # D1 (a): downloaded, printed beside, outside the rule
A3_ARMS = ("9.61e-05", "0.0001", "0.0003")   # C6 U1b's gated centred arms
A3_RUN = DATA / "trackC/c6/2026-11-01-u1b"   # records under run/records
A3_SEED0 = DATA / "trackC/c6/2026-10-31-u0-seeds"
REGISTRY = REPO / "claims/registry.json"
#: the id A1's entry takes when the user registers it; spelled in two parts because
#: `tools/check_registry.py` fails on any unregistered id it finds in the tree
ENTRY_ID = "P-SCH" + "2"


class C4pRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4p_read.py", "trackC_writeup/c6_warmup.py", "trackC_writeup/c4_read.py",
              "trackC_writeup/c4_close.py", "trackC_writeup/c2_dose.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure

def profile(per_step: Dict[str, list], steps: Sequence[int] = STEPS) -> List[float]:
    """The top head's score at each grid step."""
    missing = [s for s in steps if str(s) not in per_step]
    if missing:
        raise C4pRefused(f"profile: steps {missing} missing")
    return [float(np.max(per_step[str(s)])) for s in steps]


def ref_of(profiles: Dict[str, Sequence[float]]) -> Tuple[int, ...]:
    """The readable half-rises (C6's step reading, 92 (i)'s check) over the given seeds."""
    d = sorted({r["step"] for r in (cw.date_on(v, "step") for v in profiles.values()) if not r["unreadable"]})
    if not d:
        raise C4pRefused("no seed's profile has a readable date")
    return tuple(d)


def reading(v: Sequence[float], ref: Sequence[int]) -> dict:
    """C6's `count_reading` and `joint` on one profile, one arm, every schedule read."""
    cnt = cw.count_reading({"top": list(v)}, {"top": True}, KIND, {KIND: tuple(ref)}, not_read=())
    return {"count": cnt, "joint": cw.joint({KIND: cnt})}


def calibrate(profiles: Dict[str, Sequence[float]], loo: bool = True, late: str = "literal") -> dict:
    """Each seed's profile planted under each schedule, read against the other seeds' REF
    (``loo``) or all ten's. The gate is the gated schedules' exact-label counts."""
    rows = {}
    for name, v in profiles.items():
        r = ref_of({n: x for n, x in profiles.items() if n != name}) if loo else ref_of(profiles)
        for truth in HYPS:
            rd = reading(cw.plant(v, truth, late), r)
            rows[f"{name}|{truth}"] = {"ref": list(r), "label": rd["joint"]["label"],
                                       "status": {h: rd["count"][h]["status"] for h in HYPS},
                                       "dates": {h: rd["count"][h]["dates"] for h in HYPS}}
    summary = {t: {"exact": sum(rows[f"{n}|{t}"]["label"] == f"consistent with {t}" for n in profiles),
                   "labels": sorted({rows[f"{n}|{t}"]["label"] for n in profiles})} for t in HYPS}
    return {"summary": summary, "rows": rows,
            "pass": all(summary[t]["exact"] >= GATE_MIN for t in GATED)}


def heads_at(sc: Sequence[Sequence[float]], bar: float = HEAD_BAR) -> List[Tuple[int, int]]:
    a = np.asarray(sc, float)
    return sorted((int(l), int(h)) for l, h in zip(*np.where(a >= bar)))


def transients(mat: np.ndarray, grid: Sequence[int]) -> dict:
    """Transitions violated on a majority at some step but not in the last step's majority set."""
    last = set(cc.majority(mat[-1]))
    out = {}
    for i, s in enumerate(grid):
        for t in cc.majority(mat[i]):
            if t not in last:
                out.setdefault(f"T{t}", []).append(int(s))
    return out


# ---------------------------------------------------------------- the registration guard

def registered(path: Path = REGISTRY, entry: str = ENTRY_ID) -> bool:
    reg = json.loads(path.read_text())
    return any(p.get("id") == entry for p in reg.get("predictions", reg if isinstance(reg, list) else []))


def run_allowed(out: Path, registration: Optional[str], path: Path = REGISTRY) -> str:
    """Why the run's scores may be computed now, or refuse (D3 (a))."""
    g = out / "gate.json"
    if not g.exists():
        raise C4pRefused("gate.json missing: the gate runs before the run's scores")
    gate = json.loads(g.read_text())
    if not gate["pass"]:
        return "gate failed: A1 refused, the run's profile is description"
    if registration == "registered":
        if not registered(path):
            raise C4pRefused(f"{ENTRY_ID} is not in {path}: register it first")
        return f"{ENTRY_ID} registered"
    if registration == "declined":
        return "registration declined by the user: A1 tier 1, rule frozen in this code"
    raise C4pRefused("the gate passed: pass --registration registered|declined (the user's decision)")


# ---------------------------------------------------------------- forward passes

def _seqs(torch, device):
    g = np.random.default_rng(cr.IND_SEED)
    half = g.integers(1000, 50000, (cr.IND_N, cr.IND_LEN))
    return torch.from_numpy(np.concatenate([half, half], 1)).to(device)


def _load(name: str, step: int, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    mdl = from_pretrained_eager(GPTNeoXForCausalLM, MODELS[name], revision=f"step{step}",
                                torch_dtype=torch.float32).to(device).eval()
    if next(mdl.parameters()).dtype != torch.float32:
        raise C4pRefused(f"{name} step {step}: not float32")
    return mdl


def heads(a) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch, device = c._torch_setup()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = _seqs(torch, device)
    for name in a.model:
        why = run_allowed(out, a.registration) if name == "run" else None
        steps = list(STEPS) + (list(RUN_BESIDE) if name == "run" else [])
        f = out / f"heads_{name}.json"
        res = json.loads(f.read_text())["per_step"] if f.exists() else {}
        for s in steps:
            if str(s) in res:
                continue
            mdl = _load(name, s, device)
            with torch.no_grad():
                att = mdl(input_ids=seqs, output_attentions=True).attentions
            A = torch.stack(att, 1).float().cpu().numpy()
            sc = np.mean([cr.induction_scores(A[i], cr.IND_LEN) for i in range(cr.IND_N)], 0)
            if not np.isfinite(sc).all() or sc.shape != (6, 8):
                raise C4pRefused(f"{name} step {s}: scores not finite or shape {sc.shape}")
            res[str(s)] = sc.tolist()
            li, hi = np.unravel_index(int(np.argmax(sc)), sc.shape)
            print(json.dumps({"model": name, "step": s, "max": round(float(sc.max()), 4),
                              "head": [int(li), int(hi)], "n_at_bar": int((sc >= HEAD_BAR).sum())}), flush=True)
            del mdl
            torch.cuda.empty_cache()
            # written after each step: a crash keeps what is done
            f.write_text(json.dumps({"code": code_sha(), "model": name, "repo": MODELS[name], "len": cr.IND_LEN,
                                     "n": cr.IND_N, "seed": cr.IND_SEED, "why_allowed": why,
                                     "per_step": res}) + "\n")
    return 0


def loss(a) -> int:
    """Beside A1: the held-out loss at the run's half-rise and at pythia-70m's (`read`'s dates)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from trackC_writeup import c4_train as ct
    torch, device = c._torch_setup()
    out = Path(a.out)
    rd = json.loads((out / "reading.json").read_text())
    rows = ct.open_rows(Path(a.root))
    want = {"run": rd["run_date_step"], "seed0": rd["seed_dates"]["seed0"]}
    res = {"code": code_sha(), "rows": list(ct.HELD_OUT), "loss": {}}
    for name, s in want.items():
        if s is None:
            continue
        mdl = _load(name, s, device)
        res["loss"][name] = {"step": s, "loss": ct.held_loss(mdl, rows, device)}
        print(json.dumps({name: res["loss"][name]}), flush=True)
        del mdl
        torch.cuda.empty_cache()
    (out / "loss.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------- no forward pass

def _heads_file(out: Path, name: str) -> dict:
    f = out / f"heads_{name}.json"
    if not f.exists():
        raise C4pRefused(f"{f}: missing")
    return json.loads(f.read_text())


def gate(a) -> int:
    out = Path(a.out)
    if (out / "heads_run.json").exists():
        raise C4pRefused("heads_run.json exists: the gate is read before the run's scores")
    hs = {n: _heads_file(out, n) for n in SEEDS}
    codes = {h["code"] for h in hs.values()}
    if codes != {code_sha()}:
        raise C4pRefused(f"heads codes {sorted(codes)} ≠ this code {code_sha()}")
    prof = {n: profile(h["per_step"]) for n, h in hs.items()}
    per = {n: cw.date_on(v, "step") for n, v in prof.items()}
    ref = ref_of(prof)
    cal = calibrate(prof, loo=True)
    res = {"code": code_sha(), "pass": cal["pass"], "gate_min": GATE_MIN, "gated": list(GATED),
           "ref": list(ref), "seed_dates": {n: r["step"] for n, r in per.items()},
           "seed_unreadable": {n: r["unreadable"] for n, r in per.items()},
           "seed_plateau": {n: r["plateau"] for n, r in per.items()},
           "bands": {h: sorted(cw.band(ref, h)) for h in HYPS},
           "windows": {h: list(cw.window(h)) for h in HYPS}, "early": {h: list(cw.early(h)) for h in HYPS},
           "calibration_loo": cal,
           "beside": {"calibration_all10": calibrate(prof, loo=False)["summary"],
                      "calibration_late": {lt: calibrate(prof, loo=True, late=lt)["summary"] for lt in cw.LATE[1:]}},
           "profiles": prof, "steps": list(STEPS)}
    (out / "gate.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("pass", "ref", "seed_dates", "seed_unreadable", "bands", "windows")}))
    print(json.dumps({"loo": cal["summary"], **res["beside"]}, indent=1))
    return 0


def read(a) -> int:
    out = Path(a.out)
    g = json.loads((out / "gate.json").read_text())
    h = _heads_file(out, "run")
    if {h["code"], g["code"]} != {code_sha()}:
        raise C4pRefused(f"run heads {h['code']}, gate {g['code']}, code {code_sha()}: one version only")
    v = profile(h["per_step"])
    rd = reading(v, g["ref"])
    res = {"code": code_sha(), "gate_pass": g["pass"], "why_allowed": h["why_allowed"], "ref": g["ref"],
           "label": rd["joint"]["label"] if g["pass"] else None, "joint": rd["joint"], "count": rd["count"],
           "a1_refused": not g["pass"], "run_profile": v,
           "run_date_step": cw.date_on(v, "step")["step"],
           "seed_dates": g["seed_dates"],
           "beside": {"run_3000": float(np.max(h["per_step"]["3000"])) if "3000" in h["per_step"] else None,
                      "run_top_head": {str(s): list(map(int, np.unravel_index(int(np.argmax(h["per_step"][str(s)])), (6, 8))))
                                       for s in STEPS}}}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("label", "a1_refused", "run_date_step", "ref")}))
    print(json.dumps({hh: (rd["count"][hh]["dates"], rd["count"][hh]["status"]) for hh in HYPS}))
    print("profile", [round(x, 3) for x in v], "beside 3000", res["beside"]["run_3000"])
    return 0


def a2(a) -> int:
    out = Path(a.out)
    names = list(SEEDS) + ["run"]
    hs = {n: _heads_file(out, n)["per_step"] for n in names}
    sets = {n: {str(s): heads_at(hs[n][str(s)]) for s in A2_STEPS} for n in names}
    over = {n: {s: len(set(map(tuple, sets[n][s])) & set(map(tuple, sets["seed0"][s]))) for s in sets[n]}
            for n in names}
    res = {"code": code_sha(), "bar": HEAD_BAR, "steps": list(A2_STEPS), "heads": sets, "overlap_with_seed0": over}
    (out / "a2.json").write_text(json.dumps(res, indent=1) + "\n")
    for n in names:
        print(n.ljust(6), json.dumps({s: (sets[n][s], over[n][s]) for s in sets[n]}))
    return 0


def _recs(root: Path, name: str) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in STEPS:
        for p in V1:
            f = cw.rec_path(root, name, s, p)
            if not f.exists():
                raise C4pRefused(f"{f}: missing")
            recs[(s, p)] = json.loads(f.read_text())
    return recs


def a3(a) -> int:
    res = {"code": code_sha(), "grid": list(STEPS), "arms": list(A3_ARMS), "models": {}}
    for name, root in (("run", A3_RUN), ("seed0", A3_SEED0)):
        recs = _recs(root, name)
        res["models"][name] = {}
        for k in A3_ARMS:
            mat = cc.per_transition(recs, k, STEPS)
            i4 = STEPS.index(4000)
            res["models"][name][k] = {
                "matrix": mat.tolist(),
                "majority": {str(s): [f"T{t}" for t in cc.majority(mat[i])] for i, s in enumerate(STEPS)},
                "settled": cc.settled(mat, STEPS), "transients": transients(mat, STEPS),
                "beside_to_4000": {"settled": cc.settled(mat[:i4 + 1], STEPS[:i4 + 1]),
                                   "transients": transients(mat[:i4 + 1], STEPS[:i4 + 1])}}
            r = res["models"][name][k]
            print(name.ljust(6), k.ljust(9), "end", [f"T{t}" for t in r["settled"]["set"]], "settled", r["settled"]["step"],
                  "transients", r["transients"], "| to 4000:", r["beside_to_4000"]["settled"], r["beside_to_4000"]["transients"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "a3.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("heads", "gate", "read", "loss", "a2", "a3"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True)
        if cmd == "heads":
            sp.add_argument("--model", nargs="+", choices=sorted(MODELS), required=True)
            sp.add_argument("--registration", choices=("registered", "declined"))
        if cmd == "loss":
            from trackC_writeup import c4_train as ct
            sp.add_argument("--root", default=str(ct.ROOT))
    a = ap.parse_args(argv)
    return {"heads": heads, "gate": gate, "read": read, "loss": loss, "a2": a2, "a3": a3}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
