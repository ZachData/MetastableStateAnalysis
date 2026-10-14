"""
trackA_force/a1a.py — A1a: β per head over the 18 steps and the four step sizes per block
(`design-A.md` "A1a, the instrument"; frozen 2026-10-14, Blocked 45).

Per (step, v1 prompt), one record:

- **β**: `core.beta_eff.estimate_beta_offset_fe` (`fe_full`) per head on the stored
  `attentions.npz`, in the unit LN1 frame of that head's block, position 0 neither query nor key;
  the layer's β is the median over its 16 heads, the bracket the p10 / p50 / p90 over the layer βs
  of blocks 0–22 (the blocks whose steps enter T).
- **steps**: one hooked GPU pass (`p1e_energy_field.u2_attn.Hooked`, float32, eager, TF32 off)
  gives the residual and attention's output at every block. In block l's frame
  ``f_l(x) = unit(LN1_l(x))``, positions 1…n−1, the move ``f_l(x_l + Δ) − f_l(x_l)`` is read
  against the causal field at the layer's β on ``f_l(x_l)`` by
  `p1c_frames.integration_time.unit_steps`: ``disp``, ``cal``, ``proj`` for the block's whole
  move, ``attn_cal`` (the design's ``h_attn``) and ``attn_proj`` (beside) for attention's part.
  Each is the median over tokens; T is the sum over blocks 0–22, block 23 beside (state 24 is
  post-final-LN, so block 23's end is rebuilt as ``x_23 + attn + mlp``).
- **t\***: `sa_sim.tstar` from ``f_0(x_0)`` (the embedding in block 0's frame, position 0 out),
  causal, ``check_dt``, at the bracket's p10 and p90 β; beside, eq. (6.9) at the cloud's mean
  cosine and at 0, and g0 in the affine-free frame.

The OV attractive share row (Blocked 45's addition) is read at report time from
`data/analysis/ov_per_head_series.json`. A1a reports β (F0b) and the checks; the reading of T
against t* is A1b's.

Tier 1: exploratory, unregistered.
    python -m trackA_force.a1a run --gate <a1a_gate_<lift>.json> --out <dir> [--steps 143000 ...] [--first-only]
    python -m trackA_force.a1a report --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))

STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
V1 = ("camus_letranger", "hdbscan_code", "homer_iliad", "latex_monograph",
      "paper_excerpt", "sullivan_ballou", "wiki_paragraph")       # = trackB_circuit.b0_gate.V1_SCORED
#: The one `data/phase12` directory holding all 7 v1 prompts with attentions at each step (checked).
RUNS = {0: "2026-09-01_13-30-37", 2: "2026-08-31_11-42-56", 4: "2026-08-31_12-17-32",
        8: "2026-08-31_12-52-44", 16: "2026-08-31_13-27-45", 32: "2026-08-31_14-03-29",
        64: "2026-08-31_14-38-22", 128: "2026-08-31_15-13-34", 256: "2026-08-31_15-50-18",
        512: "2026-08-31_16-27-24", 1000: "2026-08-31_17-05-50", 2000: "2026-09-01_13-44-28",
        4000: "2026-09-01_14-07-09", 8000: "2026-09-01_15-59-26", 16000: "2026-09-01_16-48-38",
        32000: "2026-09-01_17-37-38", 54000: "2026-08-31_17-46-24", 143000: "2026-09-01_18-25-12"}
FIRST = 143000
N_BLOCKS = 24
T_BLOCKS = tuple(range(23))         # blocks whose steps enter T; 23 is beside
KINDS = ("disp", "cal", "proj", "attn_cal", "attn_proj")
#: The step's lift (`integration_time.unit_steps`): "tangent" as designed; the gate's G3 fails it
#: on its largest steps and passes "log" (STATE Blocked 46, open).
LIFT = "tangent"
TARGET = 0.9
T_MAX = 200.0                       # placed: sa_sim's horizon; beyond it t* is inf, the cell "not read"
BETA_REPRO_TOL = 1e-4               # placed: fe_full at 143000 against 1d's stored refit (same inputs)
OV_FILE = "analysis/ov_per_head_series.json"
BETA_REFIT = "p1d/beta_refit_2026-09-29/beta_refit.json"


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackA_force/a1a.py", "trackA_force/sa_sim.py", "trackA_force/tstar_table.py",
              "p1c_frames/integration_time.py", "core/beta_eff.py", "p1e_energy_field/u2_attn.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def prompts_hash() -> str:
    from core.config import PROMPTS
    h = hashlib.sha256()
    for k in V1:
        h.update(k.encode() + b"\0" + PROMPTS[k].encode() + b"\0")
    return h.hexdigest()[:12]


def run_dir(step: int, prompt: str) -> Path:
    return DATA / "phase12" / RUNS[step] / f"pythia-410m-step{step}_{prompt}"


# ---------------------------------------------------------------- pure

def frame(X: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """``unit(LN1(x))`` row-wise, float64."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    Y = (X - mu) / np.sqrt(((X - mu) ** 2).mean(axis=1, keepdims=True) + eps) * w + b
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def mean_cos(U: np.ndarray) -> float:
    n = U.shape[0]
    s = U.sum(axis=0)
    return float((s @ s - n) / (n * (n - 1)))


def head_betas(acts: np.ndarray, norms: np.ndarray, A_all: np.ndarray, ln: Dict) -> np.ndarray:
    """(24, 16) ``fe_full`` β, each head on its block's unit LN1 rows, position 0 out."""
    from core.beta_eff import estimate_beta_offset_fe
    n = acts.shape[1]
    idx = np.arange(1, n)
    out = np.full((A_all.shape[0], A_all.shape[1]), np.nan)
    for L in range(A_all.shape[0]):
        U = frame(acts[L].astype(np.float64) * norms[L][:, None], ln["w"][L], ln["b"][L], ln["eps"])
        G = U @ U.T
        for h in range(A_all.shape[1]):
            out[L, h] = estimate_beta_offset_fe(A_all[L, h], G, idx)["beta_raw"]
    return out


def bracket(layer_beta: np.ndarray) -> Dict[str, float]:
    """p10 / p50 / p90 over the layer βs of blocks 0–22; refuses a non-finite layer β."""
    lb = np.asarray(layer_beta, dtype=np.float64)[list(T_BLOCKS)]
    if not np.all(np.isfinite(lb)):
        raise ValueError(f"non-finite layer beta at blocks {np.flatnonzero(~np.isfinite(lb)).tolist()}")
    p10, p50, p90 = np.percentile(lb, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def block_steps(hs: np.ndarray, moves: Dict[int, Dict[str, np.ndarray]], ln: Dict,
                layer_beta: np.ndarray, lift: str = LIFT) -> List[Dict]:
    """
    One row per block: the median over positions 1…n−1 of each step kind, the median field
    magnitude and the mean cosine of ``f_l(x_l)``. ``moves[l]`` holds ``block`` and ``attn``
    (n, d) raw updates; the block's end is ``x_l + block`` (state 24 is post-final-LN).
    """
    from p1c_frames.integration_time import sa_field, unit_steps
    rows = []
    for L in range(N_BLOCKS):
        w, b, eps = ln["w"][L], ln["b"][L], ln["eps"]
        X = hs[L].astype(np.float64)
        U = frame(X, w, b, eps)[1:]
        F = sa_field(U, float(layer_beta[L]), causal=True)
        whole = unit_steps(U, frame(X + moves[L]["block"], w, b, eps)[1:], F, lift=lift)
        attn = unit_steps(U, frame(X + moves[L]["attn"], w, b, eps)[1:], F, lift=lift)
        vals = {"disp": whole["disp"], "cal": whole["cal"], "proj": whole["proj"],
                "attn_cal": attn["cal"], "attn_proj": attn["proj"]}
        rows.append({"block": L, "beta": float(layer_beta[L]),
                     **{k: float(np.nanmedian(v)) for k, v in vals.items()},
                     "field": float(np.median(np.linalg.norm(F, axis=1))),
                     "mean_cos": mean_cos(U),
                     "n_nan": int(np.isnan(whole["proj"]).sum())})
    return rows


def totals(rows: List[Dict]) -> Dict[str, float]:
    return {f"T_{k}": float(sum(r[k] for r in rows if r["block"] in T_BLOCKS)) for k in KINDS}


def start_and_tstar(hs0: np.ndarray, ln0: Dict, br: Dict[str, float]) -> Dict:
    """t* from the measured cloud at the bracket's ends; (6.9) and the affine-free g0 beside."""
    from . import sa_sim
    from .tstar_table import tstar as t69
    U0 = frame(hs0, ln0["w"], ln0["b"], ln0["eps"])[1:]
    U0af = frame(hs0, np.ones_like(ln0["w"]), np.zeros_like(ln0["b"]), ln0["eps"])[1:]
    n = U0.shape[0]
    g0 = mean_cos(U0)
    rec = {"n": n, "g0": g0, "g0_affine_free": mean_cos(U0af), "causal": {}, "eq69_g0": {},
           "eq69_zero": {}, "refused": {}}
    for end in ("p10", "p90"):
        beta = br[end]
        try:
            t = sa_sim.tstar(U0, beta, causal=True, check_dt=True, t_max=T_MAX)
            rec["causal"][end] = None if math.isinf(t) else t
        except (ValueError, RuntimeError) as e:
            rec["causal"][end], rec["refused"][f"causal_{end}"] = None, str(e)
        for key, g in (("eq69_g0", g0), ("eq69_zero", 0.0)):
            try:
                t = t69(n, beta, g)
                rec[key][end] = None if math.isinf(t) else t
            except ValueError as e:
                rec[key][end], rec["refused"][f"{key}_{end}"] = None, str(e)
    return rec


def ov_share(path: Path) -> Dict[int, Dict[str, List[float]]]:
    """Per step: each layer's median over heads of ``attractive_energy_fraction_core`` and the
    share of its heads at exactly 0 (`design-A.md` "Frozen 2026-10-14")."""
    d = json.loads(Path(path).read_text())
    steps, H = d["steps"], d["n_heads"]
    out = {}
    for i, s in enumerate(steps):
        med, zero = [], []
        for L in range(d["n_blocks"]):
            v = np.array([d["per_head"][f"{L},{h}"]["attractive_energy_fraction_core"][i] for h in range(H)])
            med.append(float(np.median(v)))
            zero.append(float(np.mean(v == 0.0)))
        out[int(s)] = {"median": med, "frac_zero": zero}
    return out


# ---------------------------------------------------------------- checks

def check_populated(rec: Dict) -> None:
    """The first record must be populated before the batch goes on (`CLAUDE.md` "Refuse")."""
    hb = np.array(rec["beta"]["per_head"], dtype=float)
    rows = rec["blocks"]
    bad = []
    if np.isfinite(hb).mean() < 0.95:
        bad.append(f"{np.isfinite(hb).mean():.2f} of head betas finite")
    if len(rows) != N_BLOCKS:
        bad.append(f"{len(rows)} block rows")
    for r in rows:
        if r["block"] in T_BLOCKS and not all(np.isfinite(r[k]) and r[k] != 0.0 for k in KINDS):
            bad.append(f"block {r['block']} has a zero or non-finite step")
            break
    if rec["tstar"]["causal"].get("p10") is None:
        bad.append("causal t* at p10 missing")
    if bad:
        raise SystemExit("refusing: first record not populated: " + "; ".join(bad))


def check_beta_repro(rec: Dict) -> Optional[float]:
    """At 143000, max |fe_full − 1d's stored fe_full| over this prompt's heads, if 1d fitted it."""
    p = DATA / BETA_REFIT
    if rec["step"] != 143000 or not p.exists():
        return None
    d = json.loads(p.read_text())
    ref = {(h["layer"], h["head"]): h["fe_full"] for h in d["heads"]
           if str(h.get("step")) in ("143000", "step143000") and h.get("prompt") == rec["prompt"]
           and not h.get("dedupe")}
    if not ref:
        return None
    hb = np.array(rec["beta"]["per_head"], dtype=float)
    mine = np.array([hb[L, h] for L, h in ref])
    theirs = np.array([np.nan if v is None else v for v in ref.values()], dtype=float)
    if np.any(np.isfinite(mine) != np.isfinite(theirs)):
        return math.inf                 # finite on one side only counts as a mismatch
    ok = np.isfinite(mine)
    return float(np.max(np.abs(mine[ok] - theirs[ok]))) if ok.any() else None


def check_gate(path: Path, code: str) -> None:
    """The batch runs only behind a passing gate record for this code and lift (`design-A.md`:
    "If any arm fails, A1a is not done")."""
    g = json.loads(Path(path).read_text())
    if g.get("code") != code or g.get("lift") != LIFT or not g.get("passes"):
        raise SystemExit(f"refusing: gate {path}: code {g.get('code')} (this {code}), lift "
                         f"{g.get('lift')} (this {LIFT}), passes {g.get('passes')}")


# ---------------------------------------------------------------- the batch

def _beta_job(args) -> List[List[float]]:
    rd, ln = args
    z = np.load(Path(rd) / "activations.npz")
    A = np.load(Path(rd) / "attentions.npz")["attentions"]
    return head_betas(z["activations"], z["norms"], A, ln).tolist()


def ln_from(model) -> Dict:
    lay = model.gpt_neox.layers
    return {"w": np.stack([l.input_layernorm.weight.detach().double().cpu().numpy() for l in lay]),
            "b": np.stack([l.input_layernorm.bias.detach().double().cpu().numpy() for l in lay]),
            "eps": float(lay[0].input_layernorm.eps)}


def run(a) -> int:
    import torch
    from transformers import AutoTokenizer
    from core.holdout import refuse_held_out
    from core.models import load_model
    from p1e_energy_field.u2_attn import Hooked, check_pass, token_ids
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (A1a's pass is the GPU's)")
    code, ph = code_sha(), prompts_hash()
    check_gate(Path(a.gate), code)
    steps = [int(s) for s in a.steps] if a.steps else [FIRST] + [s for s in STEPS if s != FIRST]
    if a.first_only:
        steps = steps[:1]
    refuse_held_out([run_dir(s, p) for s in steps for p in V1], context="a1a")
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"code": code, "prompts_hash": ph, "steps": steps,
                                               "runs": RUNS, "t_max": T_MAX, "target": TARGET}, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    first = True
    pool = ProcessPoolExecutor(max_workers=a.workers)
    for step in steps:
        todo = [p for p in V1 if not (out / "records" / f"step{step}_{p}.json").exists()]
        for p in V1:
            f = out / "records" / f"step{step}_{p}.json"
            if f.exists() and json.loads(f.read_text()).get("code") != code:
                raise SystemExit(f"refusing to resume: {f} was written by other code")
        if not todo:
            print(f"have step {step}", flush=True)
            first = False
            continue
        t0 = time.monotonic()
        model, _ = load_model(f"pythia-410m-step{step}")
        if next(model.parameters()).dtype != torch.float32:
            raise SystemExit("refusing: model is not float32")
        ln = ln_from(model)
        futs = {p: pool.submit(_beta_job, (str(run_dir(step, p)), ln)) for p in todo}
        hk = Hooked(model)
        dev = next(model.parameters()).device
        for p in todo:
            rd = run_dir(step, p)
            man = json.loads((rd / "manifest.json").read_text())
            if man.get("hf_revision") != f"step{step}":
                raise SystemExit(f"refusing: {rd} manifest revision {man.get('hf_revision')}")
            ids = torch.tensor([token_ids(tok, rd)], device=dev)
            hs, comps = hk.run(ids, range(N_BLOCKS))
            chk = check_pass(hs, {L: comps[L] for L in T_BLOCKS}, rd, "v1")
            moves = {L: {"block": comps[L]["block"].astype(np.float64),
                         "attn": comps[L]["attn"].astype(np.float64)} for L in range(N_BLOCKS)}
            hb = np.array(futs[p].result())
            layer_beta = np.nanmedian(hb, axis=1)
            br = bracket(layer_beta)
            rows = block_steps(hs, moves, ln, layer_beta)
            ts = start_and_tstar(hs[0].astype(np.float64), {k: (v[0] if k != "eps" else v) for k, v in ln.items()}, br)
            rec = {"step": step, "prompt": p, "run": str(rd), "code": code, "prompts_hash": ph,
                   "pass": "cuda:float32", "checks": chk,
                   "beta": {"per_head": hb.tolist(), "layer": layer_beta.tolist(), "bracket": br},
                   "blocks": rows, "T": totals(rows), "tstar": ts}
            rec["checks"]["beta_repro_143000"] = check_beta_repro(rec)
            if first:
                check_populated(rec)
                r = rec["checks"]["beta_repro_143000"]
                if r is not None and r > BETA_REPRO_TOL:
                    raise SystemExit(f"refusing: fe_full at 143000 differs from 1d's refit by {r:.1e}")
                print(f"first record populated; checks {rec['checks']}", flush=True)
                first = False
            f = out / "records" / f"step{step}_{p}.json"
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec) + "\n")
            tmp.rename(f)
            print(f"done step{step}_{p} match {chk['match_unit']:.1e}", flush=True)
        hk.close()
        del model, hk
        torch.cuda.empty_cache()
        print(f"step {step}: {time.monotonic() - t0:.0f}s", flush=True)
    pool.shutdown()
    return 0


def report(out: Path) -> int:
    """β over steps (F0b), the OV row, the checks. Not T against t* (A1b's reading)."""
    recs = [json.loads(f.read_text()) for f in sorted((Path(out) / "records").glob("step*_*.json"))]
    have = {(r["step"], r["prompt"]) for r in recs}
    missing = [(s, p) for s in STEPS for p in V1 if (s, p) not in have]
    ov = ov_share(DATA / OV_FILE)
    codes = {r["code"] for r in recs}
    lines = [f"# A1a records: {len(recs)} of {len(STEPS) * len(V1)}; code {sorted(codes)}; "
             f"prompts {sorted({r['prompts_hash'] for r in recs})}",
             f"missing: {missing or 'none'}", "",
             "step | layer beta p10 / p50 / p90 (median over prompts) | heads finite | "
             "max match | OV attractive, median over blocks 0-22 | heads at 0",
             "---|---|---|---|---|---"]
    for s in STEPS:
        rs = [r for r in recs if r["step"] == s]
        if not rs:
            continue
        br = {k: float(np.median([r["beta"]["bracket"][k] for r in rs])) for k in ("p10", "p50", "p90")}
        fin = float(np.mean([np.isfinite(np.array(r["beta"]["per_head"], dtype=float)).mean() for r in rs]))
        m = max(r["checks"]["match_unit"] for r in rs)
        o = ov[s]
        lines.append(f"{s} | {br['p10']:.2f} / {br['p50']:.2f} / {br['p90']:.2f} | {fin:.3f} | {m:.1e} | "
                     f"{np.median(o['median'][:23]):.3f} | {np.mean(o['frac_zero'][:23]):.3f}")
    repro = [r["checks"].get("beta_repro_143000") for r in recs if r["step"] == 143000]
    lines += ["", f"fe_full at 143000 against 1d's refit, max |Δ| per prompt: {repro}"]
    txt = "\n".join(lines) + "\n"
    (Path(out) / "a1a_report.txt").write_text(txt)
    print(txt)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", nargs="*", default=None)
    r.add_argument("--first-only", action="store_true")
    r.add_argument("--gate", type=Path, required=True, help="a1a_gate_<lift>.json; must pass")
    r.add_argument("--workers", type=int, default=7)
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return report(a.out) if a.cmd == "report" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
