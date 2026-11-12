"""
trackC_writeup/c4_beside.py — C4's description beside (`design-C4.md` "Beside: the cloud's
concentration and effective rank across the ramp", the user's, added 2026-11-10).

**Description only:** no label, no rule, not part of R1–R3 or `P-SCH1`, and it moves none of
them. Run in U3 only after `c4_read read` has written `reading.json` (the frozen rule applied to
both arms); `describe` refuses before that. A module of its own so `c4_read.py`'s registered
code hash (`220161b9159b`) and `c4_train.py`'s (the arms') are untouched.

**Records** (`records`): on every saved checkpoint of an arm (the dense grid and 3500), or the
released `pythia-70m` on Pythia's grid to 4000, on v1's 7 prompts, float32 eager:
ρ_ℓ, ρ_ℓ without position 0 and m_ℓ per state (`trackT_timaeus.t5.state_stats`, mean over
prompts), the sphere's effective rank per state (`core.metrics.effective_rank(h, "normed")`,
mean over prompts); then ρ_mid, ρ_1:5, ρ_mid⁻⁰ (`t5.rho_mid`, `t5.rho_15`; on 70m's 7 states
the median and mean over states 1–5) and er_mid (the median of the effective rank over states 1–5).

**Description** (`describe`, pure part `shape_of`), per series, fixed here before any value
was seen:
- ρ_mid: the peak (the first grid step of the maximum); the half-fall (the first step after the
  peak at or below the midpoint of the peak and the 4000 value); the minimum after the peak
  (censored if it is at 4000); beside, the minimum over the whole grid.
- er_mid: the mirror: the trough (first minimum), the half-rise (first step after it at or above
  the midpoint of the trough and the 4000 value), the maximum after the trough (censored at 4000);
  beside, the maximum over the whole grid.
- beside each of A0's steps, the F step the step map gives (t) and the one the summed-rate map
  gives: the continuous F step whose summed rate equals A0's to t (`c4_train.summed`), which is
  t²/2W on the warmup, as numbers, with no consistency call.

    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_beside records --out <dir> --model A0
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_beside records --out <dir> --model F
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_beside records --out <dir> --model released
    python -m trackC_writeup.c4_beside describe --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from trackA_force.a1a import REPO, V1
from trackA_force.a5_seeds import TOKENIZER
from trackC_writeup import c2_aniso as c
from trackC_writeup import c4_read as cr
from trackC_writeup import c4_train as ct

MODELS = ct.ARMS + ("released",)


class C4BesideRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4_beside.py", "trackT_timaeus/t5.py", "core/metrics.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the description (pure)

def shape_of(v: Sequence[float], grid: Sequence[int], kind: str) -> dict:
    """``kind`` "peak": ρ_mid's peak, half-fall, minimum after the peak; "trough": er_mid's mirror.
    The end value is the series' value at the last grid step (4000)."""
    x = np.asarray(v, np.float64)
    if len(x) != len(grid) or not np.all(np.isfinite(x)):
        raise C4BesideRefused("series not populated")
    s = 1.0 if kind == "peak" else -1.0 if kind == "trough" else None
    if s is None:
        raise C4BesideRefused(f"kind {kind!r}")
    y = s * x                                               # the peak of y is ρ's peak / er's trough
    i0 = int(np.argmax(y))
    mid = (y[i0] + y[-1]) / 2
    half = next((j for j in range(i0 + 1, len(y)) if y[j] <= mid), None) if i0 < len(y) - 1 else None
    j_ext = i0 + int(np.argmin(y[i0:]))                     # the far extreme after the first one
    j_all = int(np.argmin(y))
    first, far = ("peak", "min_after") if kind == "peak" else ("trough", "max_after")
    return {first: {"step": grid[i0], "value": float(x[i0])},
            "half": None if half is None else {"step": grid[half], "value": float(x[half])},
            far: {"step": grid[j_ext], "value": float(x[j_ext]), "censored": j_ext == len(y) - 1},
            ("min_all" if kind == "peak" else "max_all"): {"step": grid[j_all], "value": float(x[j_all])},
            "end": float(x[-1])}


def summed_map(t: int) -> float:
    """The continuous F step whose summed rate equals A0's summed rate to checkpoint t."""
    target = ct.summed("A0", t)
    acc, i = 0.0, 0
    while True:
        r = ct.lr_flat(i)
        if acc + r >= target:
            return i + (target - acc) / r
        acc, i = acc + r, i + 1


def maps(shape: dict) -> dict:
    """Beside each of A0's described steps: the step map's and the summed-rate map's F step."""
    out = {}
    for k, d in shape.items():
        if isinstance(d, dict) and "step" in d:
            out[k] = {"A0": d["step"], "step_map": d["step"], "summed_map": round(summed_map(d["step"]), 1)}
    return out


# ---------------------------------------------------------------- the records

def stats_of(hs: Sequence[np.ndarray]) -> dict:
    from core.metrics import effective_rank
    from trackT_timaeus import t5
    rows = [t5.state_stats(h) for h in hs]
    return {"rho": [r["rho"] for r in rows], "rho_no0": [r["rho_no0"] for r in rows],
            "m": [r["m"] for r in rows], "er": [effective_rank(h, mode="normed") for h in hs]}


def summarise(per_prompt: Sequence[dict], n_states: int) -> dict:
    from trackT_timaeus import t5
    keys = ("rho", "rho_no0", "m", "er")
    rec = {k: np.mean([p[k] for p in per_prompt], 0).tolist() for k in keys}
    if any(len(rec[k]) != n_states or not np.all(np.isfinite(rec[k])) for k in keys):
        raise C4BesideRefused(f"record not populated ({n_states} states)")
    rec["rho_mid"], rec["rho_15"] = t5.rho_mid(rec["rho"]), t5.rho_15(rec["rho"])
    rec["rho_mid_no0"] = t5.rho_mid(rec["rho_no0"])
    rec["er_mid"] = float(np.median(np.asarray(rec["er"])[1:n_states - 1]))
    rec["m_mid"] = float(np.median(np.asarray(rec["m"])[1:n_states - 1]))
    return rec


def steps_of(model: str, root: Path) -> list:
    if model == "released":
        return list(cr.PYTHIA)
    return [s for s in ct.SAVE_STEPS if ct.ckpt_dir(root, model, s).exists()]


def records(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    out, root = Path(a.out), Path(a.root)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    steps = steps_of(a.model, root)
    if a.model != "released" and steps != list(ct.SAVE_STEPS):
        raise C4BesideRefused(f"{a.model}: checkpoints {sorted(set(ct.SAVE_STEPS) - set(steps))} missing")
    f = out / "beside" / f"{a.model}.json"
    res = json.loads(f.read_text())["per_step"] if f.exists() else {}
    for s in steps:
        if str(s) in res:
            continue
        if a.model == "released":
            mdl = from_pretrained_eager(GPTNeoXForCausalLM, ct.BASE, revision=f"step{s}",
                                        torch_dtype=torch.float32).to(device).eval()
        else:
            mdl = cr.load_local(ct.ckpt_dir(root, a.model, s), device)
        n_states = mdl.config.num_hidden_layers + 1
        per = [stats_of(c.states(mdl, tok, p, device)[0]) for p in V1]
        res[str(s)] = summarise(per, n_states)
        print(json.dumps({"model": a.model, "step": s, **{k: round(res[str(s)][k], 5) for k in
                                                          ("rho_mid", "rho_15", "er_mid", "m_mid")}}), flush=True)
        ct._jdump(f, {"code": code_sha(), "model": a.model, "prompts": list(V1), "per_step": res})
        del mdl
        torch.cuda.empty_cache()
    return 0


def describe(a) -> int:
    out = Path(a.out)
    if not (out / "reading.json").exists():
        raise C4BesideRefused("reading.json absent: R1–R3 are applied first (design 'When')")
    res = {"code": code_sha(), "reading_sha256": hashlib.sha256((out / "reading.json").read_bytes()).hexdigest()}
    for model in MODELS:
        per = json.loads((out / "beside" / f"{model}.json").read_text())["per_step"]
        grid = list(cr.PYTHIA) if model == "released" else list(ct.GRID)   # 3500 kept in the file, off the grid
        res[model] = {"rho_mid": shape_of([per[str(s)]["rho_mid"] for s in grid], grid, "peak"),
                      "rho_15": shape_of([per[str(s)]["rho_15"] for s in grid], grid, "peak"),
                      "er_mid": shape_of([per[str(s)]["er_mid"] for s in grid], grid, "trough")}
    res["A0_mapped"] = {k: maps(res["A0"][k]) for k in ("rho_mid", "rho_15", "er_mid")}
    ct._jdump(out / "beside" / "description.json", res)
    print(json.dumps(res, indent=1))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("records")
    q.add_argument("--out", required=True)
    q.add_argument("--root", default=str(ct.ROOT))
    q.add_argument("--model", required=True, choices=MODELS)
    q = sub.add_parser("describe")
    q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"records": records, "describe": describe}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
