"""
trackC_writeup/c4_close.py — C4's close-out readings (2026-11-14). **Post hoc, description only.**

Everything here was chosen after `reading.json` (R1–R3) and the description beside had been
seen, so nothing in it is a label, and it moves none of R1–R3 or `P-SCH1`. A module of its own
so `c4_read.py`'s registered hash (`220161b9159b`) and `c4_train.py`'s (the arms') are untouched.

**`heads`** (GPU, a minute an arm): the design's induction witness ("Readings and outcomes",
beside) on every saved checkpoint of an arm, or the released `pythia-70m` on Pythia's grid to
4000 (the control: does the harness arm grow the released run's heads?), keeping **every head's
score** (`c4_read induction` keeps the top head only, and U3 did not run it on F or past 1000 on
A0). Same sequences and score as `c4_read` (`IND_SEED`, `induction_scores`).

**`close`** (no forward pass) reads the two arms' records, `reading.json`, the description
beside, the `heads` files and the training logs, and writes `close.json`:
- `transitions`: per count (V, each centred arm) and step, how many of the 7 prompts violate at
  each layer transition (transition i is state i → i + 1; 0 is the embeddings → block 0's output).
  Each row sums to the count `c4_read` read (checked).
- `settled`: the first grid step from which the set of transitions violated on a majority of
  prompts equals the set at 4000 at every later grid step.
- `induction`: the top head's score by step, its half-rise on its own plateau (`c4_read.date_on`,
  the registered counts' rule), and the heads at or above `HEAD_BAR` at 4000 (the released run's
  beside).
- `events`: each dated event's step on A0 and on F, the F step each registered map gives from
  A0's (`c4_read.m`; none past W), and each arm's training loss there (mean over ± `LOSS_HALF`
  iterations). Numbers only, no consistency call.
- `margin`: for each count, the smallest tolerance ρ at which F's date would enter each map's
  band (`rho_star`). The registered ρ is the grid's (1.25–1.33), not a measured run-to-run spread.

    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_close heads --out <dir> --model {A0,F,released}
    python -m trackC_writeup.c4_close close --out <dir> --reading <U2's dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import REPO, V1
from trackA_force.a5_seeds import REL_TOL
from trackC_writeup import c2_aniso as c
from trackC_writeup import c4_read as cr
from trackC_writeup import c4_train as ct

HEAD_BAR = 0.5          # a head "at or above" this induction score at 4000 is listed
LOSS_HALF = 4           # the loss at a step is the mean over the iterations within ± this
MAJORITY = len(V1) / 2  # a transition is "violated" at a step when more than half the prompts violate


class C4CloseRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4_close.py", "trackC_writeup/c4_read.py", "core/metrics.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure

def per_transition(recs: Dict[Tuple[int, str], dict], key: str, steps: Sequence[int],
                   prompts: Sequence[str] = V1) -> np.ndarray:
    """(steps, transitions): the prompts violating at each transition, by `c4_read.series`'s rule."""
    from core.metrics import energy_violation_severity
    field, tol = ("e1", REL_TOL) if key == "u" else ("e1c", float(key))
    n = len(recs[(steps[0], prompts[0])][field]) - 1
    out = np.zeros((len(steps), n), int)
    for i, s in enumerate(steps):
        for p in prompts:
            for layer in energy_violation_severity(list(recs[(s, p)][field]), rel_tol=tol)["violation_layers"]:
                out[i, layer - 1] += 1
    return out


def majority(row: Sequence[int], bar: float = MAJORITY) -> Tuple[int, ...]:
    return tuple(i for i, x in enumerate(row) if x > bar)


def settled(mat: np.ndarray, grid: Sequence[int], bar: float = MAJORITY) -> dict:
    """The first grid step from which the majority set equals the last step's at every later step."""
    last = majority(mat[-1], bar)
    i = len(grid) - 1
    while i > 0 and majority(mat[i - 1], bar) == last:
        i -= 1
    return {"step": int(grid[i]), "set": list(last)}


def rho_star(t_a0: float, t_f: float, h: str) -> float:
    """The smallest ρ ≥ 1 with t_f inside `c4_read.band(h, t_a0, ρ)`, ignoring the cap at W
    (the upper edge is taken as m(h, t_a0 ρ)); 1.0 if t_f is the map's own value."""
    if t_a0 <= 0 or t_f <= 0:
        raise C4CloseRefused("dates must be positive")
    if h == "step":
        return float(max(t_a0 / t_f, t_f / t_a0))
    if h == "cumulative":
        r = t_a0 / math.sqrt(2.0 * ct.W * t_f)
        return float(max(r, 1.0 / r))
    raise C4CloseRefused(f"hypothesis {h}")


def window_loss(losses: Sequence[float], step: int, half: int = LOSS_HALF) -> Optional[float]:
    """Mean training loss over iterations step − half … step + half (iteration t's loss is losses[t − 1])."""
    lo, hi = max(1, step - half), min(len(losses), step + half)
    if lo > hi:
        return None
    return float(np.mean(losses[lo - 1:hi]))


def mapped(t: Optional[int]) -> dict:
    """A0's step under the two registered maps (`c4_read.m`); None past W."""
    if t is None:
        return {"step": None, "cumulative": None}
    return {"step": float(t), "cumulative": cr.m("cumulative", t) if t <= ct.W else None}


def event(name: str, a0: Optional[int], f: Optional[int], loss: Dict[str, Sequence[float]]) -> dict:
    return {"event": name, "A0": a0, "F": f, "ratio": None if not (a0 and f) else a0 / f,
            "map": mapped(a0),
            "loss": {"A0": None if a0 is None else window_loss(loss["A0"], a0),
                     "F": None if f is None else window_loss(loss["F"], f)}}


# ---------------------------------------------------------------- heads (GPU)

def heads(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    out, root = Path(a.out), Path(a.root)
    g = np.random.default_rng(cr.IND_SEED)
    half = g.integers(1000, 50000, (cr.IND_N, cr.IND_LEN))
    seqs = torch.from_numpy(np.concatenate([half, half], 1)).to(device)
    if a.model == "released":
        from transformers import GPTNeoXForCausalLM
        from core.models import from_pretrained_eager
        steps = list(cr.PYTHIA)
    else:
        steps = [s for s in ct.SAVE_STEPS if ct.ckpt_dir(root, a.model, s).exists()]
        if steps != list(ct.SAVE_STEPS):
            raise C4CloseRefused(f"{a.model}: checkpoints {sorted(set(ct.SAVE_STEPS) - set(steps))} missing")
    res = {}
    for s in steps:
        if a.model == "released":
            mdl = from_pretrained_eager(GPTNeoXForCausalLM, ct.BASE, revision=f"step{s}",
                                        torch_dtype=torch.float32).to(device).eval()
        else:
            mdl = cr.load_local(ct.ckpt_dir(root, a.model, s), device)
        with torch.no_grad():
            att = mdl(input_ids=seqs, output_attentions=True).attentions
        A = torch.stack(att, 1).float().cpu().numpy()
        sc = np.mean([cr.induction_scores(A[i], cr.IND_LEN) for i in range(cr.IND_N)], 0)
        if not np.isfinite(sc).all():
            raise C4CloseRefused(f"{a.model} step {s}: a score is not finite")
        res[str(s)] = sc.tolist()
        li, hi = np.unravel_index(int(np.argmax(sc)), sc.shape)
        print(json.dumps({"model": a.model, "step": s, "max": round(float(sc.max()), 4), "head": [int(li), int(hi)],
                          "n_at_bar": int((sc >= HEAD_BAR).sum())}), flush=True)
        del mdl
        torch.cuda.empty_cache()
    ct._jdump(out / f"heads_{a.model}.json", {"code": code_sha(), "len": cr.IND_LEN, "n": cr.IND_N,
                                             "seed": cr.IND_SEED, "model": a.model, "per_step": res})
    return 0


# ---------------------------------------------------------------- close (no forward pass)

def induction_of(per_step: Dict[str, list]) -> dict:
    grid = list(ct.GRID)
    sc = {s: np.asarray(per_step[str(s)], float) for s in grid}
    top = [float(sc[s].max()) for s in grid]
    d = cr.date_on(top, grid)
    end = sc[grid[-1]]
    at_bar = sorted(((int(l), int(h), float(end[l, h])) for l, h in zip(*np.where(end >= HEAD_BAR))),
                    key=lambda x: -x[2])
    return {"top": top, "top_head": [list(map(int, np.unravel_index(int(np.argmax(sc[s])), sc[s].shape))) for s in grid],
            "half_rise": d["step"], "plateau": d["plateau"], "status": d["status"],
            "heads_at_bar_4000": [{"layer": l, "head": h, "score": v} for l, h, v in at_bar]}


def close(a) -> int:
    out, rd, root = Path(a.out), Path(a.reading), Path(a.root)
    reading = json.loads((rd / "reading.json").read_text())
    desc = json.loads((rd / "beside" / "description.json").read_text())
    grid = list(ct.GRID)
    loss = {}
    for arm in ct.ARMS:
        with open(root / arm / "train_log.jsonl") as fh:
            loss[arm] = cr.train_losses(fh)
    res = {"code": code_sha(), "post_hoc": True,
           "reading_sha256": hashlib.sha256((rd / "reading.json").read_bytes()).hexdigest(),
           "description_sha256": hashlib.sha256((rd / "beside" / "description.json").read_bytes()).hexdigest(),
           "grid": grid, "head_bar": HEAD_BAR, "transitions": {}, "settled": {}, "induction": {}}
    for arm in ct.ARMS:
        recs = cr.load_arm(rd, arm, grid)
        res["transitions"][arm], res["settled"][arm] = {}, {}
        for k in cr.KEYS:
            mat = per_transition(recs, k, grid)
            if mat.sum(1).tolist() != [int(x) for x in reading["beside"][arm][k]]:
                raise C4CloseRefused(f"{arm} {k}: the transitions do not sum to the count read")
            res["transitions"][arm][k] = mat.tolist()
            res["settled"][arm][k] = settled(mat, grid)
        h = json.loads((out / f"heads_{arm}.json").read_text())
        res["induction"][arm] = induction_of(h["per_step"])
        res["induction"][arm]["heads_code"] = h["code"]
    rel = json.loads((out / "heads_released.json").read_text())
    end = np.asarray(rel["per_step"][str(ct.HORIZON)], float)
    res["induction"]["released"] = {
        "heads_code": rel["code"], "top": [float(np.max(rel["per_step"][str(s)])) for s in cr.PYTHIA],
        "heads_at_bar_4000": sorted(({"layer": int(l), "head": int(h), "score": float(end[l, h])}
                                     for l, h in zip(*np.where(end >= HEAD_BAR))), key=lambda x: -x["score"])}
    gated = [k for k in cr.KEYS[1:] if reading["gated"][k]]
    dates = reading["dates"]
    ev = [event("rho_mid peak", desc["A0"]["rho_mid"]["peak"]["step"], desc["F"]["rho_mid"]["peak"]["step"], loss),
          event("rho_mid half-fall", desc["A0"]["rho_mid"]["half"]["step"], desc["F"]["rho_mid"]["half"]["step"], loss),
          event("V date (R3)", dates["A0"]["u"]["step"], dates["F"]["u"]["step"], loss)]
    for k in gated:
        ev.append(event(f"centred date {k} (R2)", dates["A0"][k]["step"], dates["F"][k]["step"], loss))
    ev += [event("er_mid half-rise", desc["A0"]["er_mid"]["half"]["step"], desc["F"]["er_mid"]["half"]["step"], loss),
           event("induction half-rise", res["induction"]["A0"]["half_rise"], res["induction"]["F"]["half_rise"], loss)]
    for k in gated:
        ev.append(event(f"centred set settled {k}", res["settled"]["A0"][k]["step"], res["settled"]["F"][k]["step"], loss))
    res["events"] = ev
    res["margin"] = {k: {"t_a0": dates["A0"][k]["step"], "t_f": dates["F"][k]["step"], "rho": reading["per_key"][k]["rho"],
                         **{h: rho_star(dates["A0"][k]["step"], dates["F"][k]["step"], h) for h in cr.HYPS}}
                     for k in cr.KEYS if dates["A0"][k]["step"] and dates["F"][k]["step"]}
    res["loss_4000"] = {arm: window_loss(loss[arm], 4000) for arm in ct.ARMS}
    ct._jdump(out / "close.json", res)
    print(json.dumps({k: res[k] for k in ("settled", "events", "margin", "loss_4000")}, indent=1))
    print(json.dumps({arm: {k: res["induction"][arm][k] for k in ("half_rise", "plateau", "heads_at_bar_4000")}
                      for arm in ct.ARMS}, indent=1))
    print(json.dumps({"released": res["induction"]["released"]["heads_at_bar_4000"]}, indent=1))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("heads")
    q.add_argument("--out", required=True)
    q.add_argument("--root", default=str(ct.ROOT))
    q.add_argument("--model", required=True, choices=ct.ARMS + ("released",))
    q = sub.add_parser("close")
    q.add_argument("--out", required=True)
    q.add_argument("--reading", required=True)
    q.add_argument("--root", default=str(ct.ROOT))
    a = p.parse_args(argv)
    return {"heads": heads, "close": close}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
