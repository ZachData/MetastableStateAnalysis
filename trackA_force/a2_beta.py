"""
trackA_force/a2_beta.py — A2 unit 2: the β refit (`design-A2.md` "Unit 2"; frozen 2026-10-16,
Blocked 56 (a)).

Per (step, v1 prompt), on the inputs A1a read (its records, the stored ``attentions.npz``, the
unit LN1 rows ``a1a.frame``, position 0 neither query nor key):

- **the 2 × 2:** {M1 at full D (`trackB_circuit.b2a.fit`, the softmax's own cross-entropy, B2a's
  offset bins, rows renormalised over keys ≥ 1), A1a's ``fe_full``} × {all heads, heads with
  ``fe_full``'s ``partial_r2`` ≥ ``FLOOR``}. Each head is fitted on that prompt's rows alone. The
  layer's β is the median over its (kept) heads; a layer with fewer than ``MIN_KEPT`` kept heads
  has none. The bracket is p10 / p50 / p90 over the layer βs of blocks 0–22 (A1a's).
  **Primary: M1, all heads.**
- **checks first, or the unit refuses:** (a) B2a's own leave-one-prompt-out path, on B2a unit 2's
  inputs (its GPU pass, its recomputed attention), reproduces that record's ``beta_full`` on its
  14 heads to ``TOL_B2A``; (b) ``fe_full`` on A1a's inputs reproduces A1a's stored head βs to
  ``TOL_A1A``.
- **the rule:** A1's number *moves* if the primary's p50 lies outside A1a's stored [p10, p90] in
  ≥ ``MOVES_OF`` of 7 prompts.
- **A1b's cell, re-run** (whatever the rule says): A1a's hooked pass with the primary's layer βs
  (the field at block l at the refit layer β, t* at the refit bracket), read by `a1b.cell`;
  the same pass at A1a's stored βs must reproduce A1a's ``T_proj`` and t* first.
- **beside:** M1's β per band (heads × prompts pooled) against the convention's [1.6, 5.6]
  (Blocked 9); outside goes to the user, nothing is changed here.

Tier 1: exploratory, unregistered.
    python -m trackA_force.a2_beta run --out <dir> [--steps 143000 ...] [--no-a1b]
    python -m trackA_force.a2_beta report --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .a1a import DATA, N_BLOCKS, REPO, STEPS, T_BLOCKS, V1, bracket, frame, prompts_hash, run_dir

A1A_DIR = DATA / "trackA/a1a_batch_2026-10-15/records"
A1A_CODE = "a08544e9b644"
PROMPTS = "976b62d677bc"
B2A_REAL = DATA / "b2a/2026-10-16/real.json"       # B2a unit 2, step 143000
FIRST = 143000
FLOOR = 0.10                        # placed (design): fe_full's partial_r2 for "kept"
MIN_KEPT = 4                        # placed (design): fewer kept heads, no layer β
MOVES_OF = 4                        # placed (design): of 7 prompts
CONVENTION = (1.6, 5.6)             # Blocked 9: 3.5 [1.6, 5.6], fe_full, all heads, v1 length
BANDS = {"L1-8": range(1, 9), "L9-16": range(9, 17), "L17-23": range(17, 24), "L1-23": range(1, 24)}
TOL_A1A = 1e-9
TOL_B2A = 1e-6
TOL_TPROJ = 1e-6                    # placed here: A1a's own pass at its own βs, same GPU and code path
ESTIMATORS = ("m1", "fe")
SUBSETS = ("all", "kept")


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackA_force/a2_beta.py", "trackA_force/a1a.py", "trackA_force/a1b.py",
              "trackA_force/sa_sim.py", "trackB_circuit/b2a.py", "core/beta_eff.py",
              "p1c_frames/integration_time.py", "p1e_energy_field/u2_attn.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure

def rows_of(acts: np.ndarray, norms: np.ndarray, L: int, ln: Dict) -> np.ndarray:
    """Block L's unit LN1 rows, A1a's frame (position 0 kept; callers drop it)."""
    return frame(acts[L].astype(np.float64) * norms[L][:, None], ln["w"][L], ln["b"][L], ln["eps"])


def fe_heads(acts: np.ndarray, norms: np.ndarray, A_all: np.ndarray, ln: Dict) -> Dict[str, np.ndarray]:
    """(24, 16) ``fe_full`` β and its ``partial_r2``, as `a1a.head_betas` (same call, same rows)."""
    from core.beta_eff import estimate_beta_offset_fe
    n = acts.shape[1]
    idx = np.arange(1, n)
    beta = np.full(A_all.shape[:2], np.nan)
    pr2 = np.full(A_all.shape[:2], np.nan)
    for L in range(A_all.shape[0]):
        U = rows_of(acts, norms, L, ln)
        G = U @ U.T
        for h in range(A_all.shape[1]):
            r = estimate_beta_offset_fe(A_all[L, h], G, idx)
            beta[L, h], pr2[L, h] = r["beta_raw"], r["partial_r2"]
    return {"beta": beta, "partial_r2": pr2}


def m1_heads(acts: np.ndarray, norms: np.ndarray, A_all: np.ndarray, ln: Dict,
             device: str) -> Dict[str, object]:
    """(24, 16) M1 β at full D, each head on this prompt's rows alone (`b2a.fit`, one prompt);
    a fit that does not converge is NaN and listed."""
    import torch
    from trackB_circuit.b2a import B2aError, Seq, cos_feat, fit, renorm
    beta = np.full(A_all.shape[:2], np.nan)
    failed, iters = [], 0
    for L in range(A_all.shape[0]):
        u = torch.as_tensor(rows_of(acts, norms, L, ln)[1:], dtype=torch.float64, device=device)
        C = cos_feat(u, None)
        for h in range(A_all.shape[1]):
            s = Seq(renorm(np.asarray(A_all[L, h], dtype=np.float64)), device)
            try:
                th, info = fit([s.off(C)], [s])
            except B2aError as e:
                failed.append({"layer": L, "head": h, "why": str(e)})
                continue
            beta[L, h] = float(th[0, 0])
            iters = max(iters, info["iters"])
    return {"beta": beta, "failed": failed, "max_iters": iters}


def layer_betas(hb: np.ndarray, pr2: np.ndarray, subset: str) -> np.ndarray:
    """(24,) median over a layer's heads ("all") or its heads with partial_r2 ≥ FLOOR ("kept");
    NaN where fewer than MIN_KEPT kept heads have a finite β."""
    hb, pr2 = np.asarray(hb, dtype=float), np.asarray(pr2, dtype=float)
    out = np.full(hb.shape[0], np.nan)
    for L in range(hb.shape[0]):
        keep = np.isfinite(hb[L]) & ((pr2[L] >= FLOOR) if subset == "kept" else True)
        if subset == "all" and keep.any():
            out[L] = float(np.median(hb[L][keep]))
        elif subset == "kept" and keep.sum() >= MIN_KEPT:
            out[L] = float(np.median(hb[L][keep]))
    return out


def bracket_or_none(lb: np.ndarray) -> Dict:
    """`a1a.bracket` where every block 0–22 has a β; otherwise the percentiles over those that
    do, marked partial (beside only; the primary must be whole)."""
    lb = np.asarray(lb, dtype=float)
    fin = np.isfinite(lb[list(T_BLOCKS)])
    if fin.all():
        return {**bracket(lb), "n_layers": int(fin.sum()), "whole": True}
    if not fin.any():
        return {"p10": None, "p50": None, "p90": None, "n_layers": 0, "whole": False}
    p10, p50, p90 = np.percentile(lb[list(T_BLOCKS)][fin], [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90), "n_layers": int(fin.sum()),
            "whole": False}


def outside(p50: float, a1: Dict) -> bool:
    return bool(p50 < a1["p10"] or p50 > a1["p90"])


def rule(cells: Sequence[Dict]) -> Dict:
    """The fixed rule on the primary: moves if outside A1's bracket in ≥ MOVES_OF prompts."""
    out = [c["prompt"] for c in cells if outside(c["refit"]["m1/all"]["p50"], c["a1"])]
    return {"outside": out, "n_outside": len(out), "of": len(cells), "moves": len(out) >= MOVES_OF}


def band_betas(per_prompt: Sequence[np.ndarray]) -> Dict[str, Dict]:
    """Heads × prompts pooled per band: median [p10, p90]; flagged outside the convention."""
    out = {}
    for name, layers in BANDS.items():
        v = np.concatenate([np.asarray(hb)[list(layers)].ravel() for hb in per_prompt])
        v = v[np.isfinite(v)]
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        out[name] = {"n": int(v.size), "p10": float(p10), "p50": float(p50), "p90": float(p90),
                     "outside_convention": bool(p50 < CONVENTION[0] or p50 > CONVENTION[1])}
    return out


# ---------------------------------------------------------------- checks

def a1a_record(step: int, prompt: str) -> Dict:
    r = json.loads((A1A_DIR / f"step{step}_{prompt}.json").read_text())
    if r["code"] != A1A_CODE or r["prompts_hash"] != PROMPTS:
        raise SystemExit(f"refusing: A1a record step{step}_{prompt} is {r['code']}/{r['prompts_hash']}")
    return r


def check_fe(fe: np.ndarray, rec: Dict) -> float:
    """Max |Δ| of ``fe_full`` head βs against A1a's stored ones (finite on one side = inf)."""
    st = np.array(rec["beta"]["per_head"], dtype=float)
    if np.any(np.isfinite(fe) != np.isfinite(st)):
        return math.inf
    ok = np.isfinite(st)
    return float(np.max(np.abs(fe[ok] - st[ok])))


def check_b2a(device: str) -> Dict:
    """Check (a): B2a unit 2's ``beta_full`` from its own inputs through `b2a.loo` (GPU pass)."""
    import torch
    from p1e_energy_field.u2_attn import token_ids
    from trackB_circuit import b2a
    from trackB_circuit.b0_gate import load
    ref = json.loads(B2A_REAL.read_text())
    if ref["step"] != FIRST:
        raise SystemExit(f"refusing: {B2A_REAL} is step {ref['step']}")
    names = list(ref["heads"])
    model, tok = load(FIRST, device, torch.float32)
    lay = model.gpt_neox.layers
    eps = float(lay[0].input_layernorm.eps)
    need = sorted({b2a.parse_head(h)[0] for h in names})
    hs = {}
    for q in V1:
        ids = torch.tensor([token_ids(tok, run_dir(FIRST, q))], device=device)
        with torch.no_grad():
            o = model(ids, output_hidden_states=True)
        hs[q] = {l: o.hidden_states[l][0].float().cpu().numpy() for l in need}
    model.cpu()
    Ws = {h: b2a.head_weights(model, h) for h in names}
    ln = {l: (lay[l].input_layernorm.weight.detach().double().numpy(),
              lay[l].input_layernorm.bias.detach().double().numpy()) for l in need}
    del model
    torch.cuda.empty_cache()
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)  # noqa: E731
    diffs = {}
    for h in names:
        l, _ = b2a.parse_head(h)
        ys = [b2a.ln1(hs[q][l], *ln[l], eps)[0] for q in V1]
        As = [b2a.renorm(b2a.softmax_rows(b2a.head_logits(y, Ws[h]))) for y in ys]
        seqs = [b2a.Seq(A, device) for A in As]
        got = float(b2a.loo([T(b2a._unit(y)[1:]) for y in ys], seqs, None)["beta"][0])
        diffs[h] = {"got": got, "stored": ref["heads"][h]["beta_full"],
                    "abs": abs(got - ref["heads"][h]["beta_full"])}
    worst = max(diffs.values(), key=lambda d: d["abs"])["abs"]
    return {"heads": diffs, "max_abs": worst, "tol": TOL_B2A, "pass": worst <= TOL_B2A}


# ---------------------------------------------------------------- jobs

def _fe_job(args) -> Dict:
    rd, ln = args
    z = np.load(Path(rd) / "activations.npz")
    A = np.load(Path(rd) / "attentions.npz")["attentions"]
    r = fe_heads(z["activations"], z["norms"], A, ln)
    return {k: v.tolist() for k, v in r.items()}


def refit_step(step: int, ln: Dict, device: str, workers: int) -> Dict:
    """The 2 × 2 on every v1 prompt at ``step``; refuses on check (b)."""
    recs = {p: a1a_record(step, p) for p in V1}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fe = {p: pool.submit(_fe_job, (str(run_dir(step, p)), ln)) for p in V1}
        m1 = {}
        for p in V1:
            z = np.load(run_dir(step, p) / "activations.npz")
            A = np.load(run_dir(step, p) / "attentions.npz")["attentions"]
            t0 = time.monotonic()
            m1[p] = m1_heads(z["activations"], z["norms"], A, ln, device)
            print(f"step {step} {p}: M1 {time.monotonic() - t0:.0f}s, "
                  f"{len(m1[p]['failed'])} not converged", flush=True)
        fe = {p: {k: np.array(v) for k, v in f.result().items()} for p, f in fe.items()}
    cells = []
    for p in V1:
        d = check_fe(fe[p]["beta"], recs[p])
        if not d <= TOL_A1A:
            raise SystemExit(f"refusing: check (b): fe_full at step {step} {p} off A1a's by {d:.1e}")
        pr2 = fe[p]["partial_r2"]
        heads = {"m1": m1[p]["beta"], "fe": fe[p]["beta"]}
        refit = {}
        for e in ESTIMATORS:
            for s in SUBSETS:
                lb = layer_betas(heads[e], pr2, s)
                refit[f"{e}/{s}"] = {"layer": [None if not np.isfinite(x) else float(x) for x in lb],
                                     **bracket_or_none(lb)}
        if not refit["m1/all"]["whole"]:
            raise SystemExit(f"refusing: step {step} {p}: the primary has no β at some block 0–22")
        cells.append({"step": step, "prompt": p, "a1": recs[p]["beta"]["bracket"],
                      "check_fe": d, "refit": refit,
                      "m1_heads": m1[p]["beta"].tolist(), "m1_failed": m1[p]["failed"],
                      "m1_max_iters": m1[p]["max_iters"], "partial_r2": pr2.tolist(),
                      "kept_heads": int(np.sum(pr2[list(T_BLOCKS)] >= FLOOR))})
    return {"step": step, "cells": cells, "rule": rule(cells),
            "bands_m1": band_betas([np.array(c["m1_heads"], dtype=float) for c in cells]),
            "bands_fe": band_betas([fe[p]["beta"] for p in V1])}


def a1b_rerun(step: int, layer_beta: Dict[str, List[float]], device: str) -> List[Dict]:
    """A1a's pass at ``step``; per prompt, A1b's cell at A1a's βs (must reproduce its record) and
    at the refit layer βs."""
    import torch
    from transformers import AutoTokenizer
    from core.models import load_model
    from p1e_energy_field.u2_attn import Hooked, check_pass, token_ids
    from . import a1b
    from .a1a import block_steps, ln_from, start_and_tstar, totals
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    model, _ = load_model(f"pythia-410m-step{step}")
    if next(model.parameters()).dtype != torch.float32:
        raise SystemExit("refusing: model is not float32")
    ln = ln_from(model)
    ln0 = {k: (v[0] if k != "eps" else v) for k, v in ln.items()}
    hk = Hooked(model)
    dev = next(model.parameters()).device
    out = []
    for p in V1:
        rec = a1a_record(step, p)
        rd = run_dir(step, p)
        ids = torch.tensor([token_ids(tok, rd)], device=dev)
        hs, comps = hk.run(ids, range(N_BLOCKS))
        chk = check_pass(hs, {L: comps[L] for L in T_BLOCKS}, rd, "v1")
        moves = {L: {"block": comps[L]["block"].astype(np.float64),
                     "attn": comps[L]["attn"].astype(np.float64)} for L in range(N_BLOCKS)}
        row = {"prompt": p, "checks": chk}
        for tag, lb in (("a1a", np.array(rec["beta"]["layer"], dtype=float)),
                        ("refit", np.array(layer_beta[p], dtype=float))):
            br = bracket(lb)
            blocks = block_steps(hs, moves, ln, lb)
            new = {**rec, "beta": {**rec["beta"], "layer": lb.tolist(), "bracket": br},
                   "blocks": blocks, "T": totals(blocks),
                   "tstar": start_and_tstar(hs[0].astype(np.float64), ln0, br)}
            row[tag] = a1b.cell(new)
        d_t = abs(row["a1a"]["T_proj"] - rec["T"]["T_proj"])
        d_s = max(abs((row["a1a"][k] or 0.0) - (rec["tstar"]["causal"].get(e) or 0.0))
                  for k, e in (("t10", "p10"), ("t90", "p90")))
        row["repro"] = {"T_proj": d_t, "tstar": d_s}
        if d_t > TOL_TPROJ or d_s > TOL_TPROJ:
            raise SystemExit(f"refusing: A1a's own pass at its βs off its record at {p}: "
                             f"T_proj {d_t:.1e}, t* {d_s:.1e}")
        out.append(row)
        print(f"A1b {p}: {row['a1a']['reading']} -> {row['refit']['reading']} "
              f"(T_proj {row['a1a']['T_proj']:.3f} -> {row['refit']['T_proj']:.3f}, "
              f"t* {row['refit']['t10']}, {row['refit']['t90']})", flush=True)
    hk.close()
    del model, hk
    torch.cuda.empty_cache()
    return out


def run(a) -> int:
    import torch
    from core.holdout import refuse_held_out
    from .a1a_gate import ln_all
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if prompts_hash() != PROMPTS:
        raise SystemExit(f"refusing: v1 prompts hash {prompts_hash()}, A1a's {PROMPTS}")
    steps = [int(s) for s in a.steps] if a.steps else [FIRST]
    refuse_held_out([run_dir(s, p) for s in steps for p in V1], context="a2_beta")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    code = code_sha()
    lns = ln_all()
    if FIRST in steps:
        f = out / "check_b2a.json"
        if not f.exists():
            t0 = time.monotonic()
            c = check_b2a(device)
            f.write_text(json.dumps({"code": code, **c}, indent=1) + "\n")
            print(f"check (a): max |Δ| {c['max_abs']:.1e} ({time.monotonic() - t0:.0f}s)", flush=True)
        c = json.loads(f.read_text())
        if not c["pass"]:
            raise SystemExit(f"refusing: check (a): B2a's beta_full off by {c['max_abs']:.1e}")
    for step in steps:
        f = out / f"refit_step{step}.json"
        if f.exists():
            print(f"have step {step}", flush=True)
            continue
        i = STEPS.index(step)
        ln = {"w": lns["w"][i], "b": lns["b"][i], "eps": lns["eps"]}
        t0 = time.monotonic()
        r = refit_step(step, ln, device, a.workers)
        r.update(code=code, device=device, ln_sha=lns["sha"], a1a_code=A1A_CODE, prompts_hash=PROMPTS,
                 seconds=time.monotonic() - t0)
        f.write_text(json.dumps(r) + "\n")
        print(f"step {step}: rule {r['rule']} ({r['seconds']:.0f}s)", flush=True)
    if not a.no_a1b and FIRST in steps:
        f = out / "a1b_step143000.json"
        if not f.exists():
            r = json.loads((out / f"refit_step{FIRST}.json").read_text())
            lb = {c["prompt"]: c["refit"]["m1/all"]["layer"] for c in r["cells"]}
            rows = a1b_rerun(FIRST, lb, "cuda")
            from .a1b import read_step
            f.write_text(json.dumps({"code": code, "step": FIRST, "cells": rows,
                                     "step_a1a": read_step([x["a1a"]["reading"] for x in rows]),
                                     "step_refit": read_step([x["refit"]["reading"] for x in rows])},
                                    indent=1) + "\n")
    return report(out)


def _f(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}"


def report(out: Path) -> int:
    out = Path(out)
    lines = []
    c = out / "check_b2a.json"
    if c.exists():
        c = json.loads(c.read_text())
        lines += [f"check (a), B2a unit 2's beta_full through b2a.loo: max |Δ| {c['max_abs']:.1e} "
                  f"(tol {c['tol']:.0e}) {'pass' if c['pass'] else 'FAIL'}", ""]
    for f in sorted(out.glob("refit_step*.json"), key=lambda p: int(p.stem[10:])):
        r = json.loads(f.read_text())
        ru = r["rule"]
        lines += [f"# step {r['step']}: code {r['code']}, A1a {r['a1a_code']}, prompts {r['prompts_hash']}, "
                  f"{r['device']}, {r['seconds']:.0f}s",
                  f"check (b), fe_full against A1a's head βs: max |Δ| "
                  f"{max(x['check_fe'] for x in r['cells']):.1e} (tol {TOL_A1A:.0e})",
                  f"M1 fits not converged: {sum(len(x['m1_failed']) for x in r['cells'])} of "
                  f"{len(r['cells']) * N_BLOCKS * 16}; max Newton iterations "
                  f"{max(x['m1_max_iters'] for x in r['cells'])}",
                  f"**rule (primary M1/all p50 outside A1's [p10, p90] in ≥ {MOVES_OF} of 7): "
                  f"{ru['n_outside']} of {ru['of']} → {'MOVES' if ru['moves'] else 'does not move'}**"
                  f" (outside: {', '.join(ru['outside']) or 'none'})", "",
                  "prompt | A1 p10 / p50 / p90 | M1 all | M1 kept | fe all | fe kept | kept heads (0–22)",
                  "---|---|---|---|---|---|---"]
        for x in r["cells"]:
            a1 = x["a1"]
            cell = lambda k: (f"{_f(x['refit'][k]['p10'])} / {_f(x['refit'][k]['p50'])} / "  # noqa: E731
                              f"{_f(x['refit'][k]['p90'])}"
                              + ("" if x["refit"][k]["whole"] else f" ({x['refit'][k]['n_layers']} layers)"))
            lines.append(f"{x['prompt']} | {a1['p10']:.2f} / {a1['p50']:.2f} / {a1['p90']:.2f} | "
                         f"{cell('m1/all')} | {cell('m1/kept')} | {cell('fe/all')} | {cell('fe/kept')} | "
                         f"{x['kept_heads']} of {len(T_BLOCKS) * 16}")
        lines += ["", f"β per band, heads × prompts pooled, median [p10, p90] (convention "
                      f"{CONVENTION[0]}–{CONVENTION[1]}, fe_full):",
                  "band | M1 | fe_full | n | M1 outside convention", "---|---|---|---|---"]
        for b in BANDS:
            m, e = r["bands_m1"][b], r["bands_fe"][b]
            lines.append(f"{b} | {m['p50']:.2f} [{m['p10']:.2f}, {m['p90']:.2f}] | "
                         f"{e['p50']:.2f} [{e['p10']:.2f}, {e['p90']:.2f}] | {m['n']} | "
                         f"{'YES' if m['outside_convention'] else 'no'}")
        lines.append("")
    f = out / "a1b_step143000.json"
    if f.exists():
        r = json.loads(f.read_text())
        lines += [f"# A1b's cell at 143000 re-run at the primary's layer βs: step reading "
                  f"{r['step_a1a']} → {r['step_refit']}",
                  "prompt | reading A1a → refit | T_proj A1a → refit | t*(p10), t*(p90) A1a → refit | "
                  "T_proj / t*(p10) refit | repro T_proj, t*",
                  "---|---|---|---|---|---"]
        for x in r["cells"]:
            a, b = x["a1a"], x["refit"]
            lines.append(f"{x['prompt']} | {a['reading']} → {b['reading']} | {a['T_proj']:.3f} → "
                         f"{b['T_proj']:.3f} | {_f(a['t10'])}, {_f(a['t90'])} → {_f(b['t10'])}, "
                         f"{_f(b['t90'])} | {_f(b['r10'])} | {x['repro']['T_proj']:.0e}, {x['repro']['tstar']:.0e}")
    txt = "\n".join(lines) + "\n"
    (out / "a2_beta_report.txt").write_text(txt)
    print(txt)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", nargs="*", default=None)
    r.add_argument("--workers", type=int, default=7)
    r.add_argument("--no-a1b", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return report(a.out) if a.cmd == "report" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
