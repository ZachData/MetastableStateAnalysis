"""
trackA_force/a1a_gate.py — A1a's calibration gate, run before any real reading
(`design-A.md` "The calibration gate", re-planted by Blocked 46 (b); `CLAUDE.md` "Calibration gate").

Synthetic clouds at the real design points (n = each v1 prompt's length less position 0,
d = 1024), planted trajectories of 23 forward-Euler steps:

- G0, comparator: `sa_sim` with full attention from a Gaussian start against eq. (6.9) at
  β ∈ {0, 3.5, 5.6}, within 3 %; the causal arm's dt check converges.
- G1, planted effect: the identity-weight causal field at known h_l (log-uniform in
  [0.02, 0.5]), from an orthogonal start and from pairwise cosine 0.3, each step retracted to
  the sphere; ``proj`` and ``cal`` recover each h_l within 5 % and T within 3 %.
- G2, planted null: the same move norms per token, directions uniform in the tangent plane;
  |T_proj| ≤ 0.1 T (``cal`` beside: it cannot see direction). Beside, not gated, **G2s**: the
  same norms, every token moving toward one fixed random direction (a shared drift).
- G3, mixed: ``Δ = hX + ξ``, ξ tangent and ⊥ X with |ξ| = |hX|; ``proj`` within 5 % of h,
  ``cal`` within 5 % of √2 h.
- G4, frame (Blocked 46 (b)): the move a V = I head makes in the reading's frame. The raw rows
  ``x`` (start √d·U0) are read as ``U = unit(LN1(x))``; the field is evaluated on U and the move
  ``h·|Cx|·X(U)`` (C centres; the scale makes the affine-free frame exactly G1) is added to the
  raw residual, then read back through the same LN1. T_proj within 10 % of T on step 143000's
  block-5 LN1 at all 42 cells, **and** on every block 0–22 × step LN1 (the 18 steps) for one
  prompt's six (β, start) cells (block 23 beside). Beside: the earlier G4 (G1's raw trajectory
  read through block 5's LN1, "G4raw").

Every arm is read with three lifts (`integration_time.unit_steps`): the verdict is ``LIFT``
(log, as Blocked 46 (b) picked); tangent and gnomonic beside.

    python -m trackA_force.a1a_gate --out <dir> [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from p1c_frames.integration_time import sa_field, unit_steps

from .a1a import DATA, LIFT, N_BLOCKS, STEPS, T_BLOCKS, code_sha, frame, mean_cos

D = 1024
#: Each v1 prompt's token count less position 0 (activations.npz at 143000; all steps alike).
NS = {"camus_letranger": 464, "hdbscan_code": 241, "homer_iliad": 511, "latex_monograph": 445,
      "paper_excerpt": 285, "sullivan_ballou": 481, "wiki_paragraph": 466}
BETAS = (0.0, 3.5, 7.0)
G0_BETAS = (0.0, 3.5, 5.6)
STARTS = (0.0, 0.3)
N_STEPS = 23
H_LO, H_HI = 0.02, 0.5
SEED = 20261014
TOL = {"G0": 0.03, "G1_h": 0.05, "G1_T": 0.03, "G2_T": 0.10, "G3_h": 0.05, "G4_T": 0.10}
LIFTS = ("log", "tangent", "gnomonic")
#: G4's sweep over every LN1: the prompt whose cell read lowest in the first G4 (0.742, β 7, cosine 0).
SWEEP_PROMPT = "latex_monograph"
LN_FILE = "trackA/ln1_18steps.npz"


def gate_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def planted_h(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(math.log(H_LO), math.log(H_HI), N_STEPS))


def start(n: int, g0: float, rng) -> np.ndarray:
    """n unit rows with every pairwise cosine exactly g0 (orthonormal when g0 = 0)."""
    Q, _ = np.linalg.qr(rng.standard_normal((D, n + 1)))
    e, q = Q[:, 0], Q[:, 1:].T
    return math.sqrt(g0) * e[None, :] + math.sqrt(1.0 - g0) * q


def _retract(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    Y = U + V
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def _unit(Z: np.ndarray) -> np.ndarray:
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300)


def _tangent_noise(U: np.ndarray, norms: np.ndarray, rng, also_perp: Optional[np.ndarray] = None) -> np.ndarray:
    """Per row a uniform direction tangent at U (and ⊥ ``also_perp`` if given), scaled to ``norms``."""
    Z = rng.standard_normal(U.shape)
    Z -= np.sum(Z * U, axis=1, keepdims=True) * U
    if also_perp is not None:
        P = _unit(also_perp)
        Z -= np.sum(Z * P, axis=1, keepdims=True) * P
    return _unit(Z) * norms[:, None]


def trajectory(U0: np.ndarray, h: np.ndarray, beta: float, arm: str, rng) -> List[np.ndarray]:
    """N_STEPS + 1 configurations; ``arm`` is ``field``, ``null``, ``drift`` or ``mixed``."""
    out = [U0]
    U = U0
    e = _unit(rng.standard_normal((1, U0.shape[1]))) if arm == "drift" else None
    for hl in h:
        F = sa_field(U, beta, causal=True)
        hX = hl * F
        if arm == "field":
            V = hX
        elif arm == "null":
            V = _tangent_noise(U, np.linalg.norm(hX, axis=1), rng)
        elif arm == "drift":
            V = _unit(e - (U @ e.T) * U) * np.linalg.norm(hX, axis=1, keepdims=True)
        elif arm == "mixed":
            V = hX + _tangent_noise(U, np.linalg.norm(hX, axis=1), rng, also_perp=F)
        else:
            raise ValueError(arm)
        U = _retract(U, V)
        out.append(U)
    return out


def _acc() -> Dict[str, Dict[str, list]]:
    return {l: {"proj": [], "cal": []} for l in LIFTS}


def _add(acc, U, U1, F) -> None:
    for l in LIFTS:
        s = unit_steps(U, U1, F, lift=l)
        acc[l]["proj"].append(float(np.nanmedian(s["proj"])))
        acc[l]["cal"].append(float(np.nanmedian(s["cal"])))


def _close(acc) -> Dict:
    return {l: {**v, "T_proj": float(sum(v["proj"])), "T_cal": float(sum(v["cal"]))} for l, v in acc.items()}


def read(traj: Sequence[np.ndarray], beta: float, fr: Optional[Callable] = None) -> Dict:
    """Per lift: per-step medians of ``proj`` and ``cal`` and their sums, in frame ``fr`` (identity if None)."""
    fr = fr or (lambda Y: Y)
    acc = _acc()
    for a, b in zip(traj[:-1], traj[1:]):
        U = fr(a)
        _add(acc, U, fr(b), sa_field(U, beta, causal=True))
    return _close(acc)


def ln_plant(U0: np.ndarray, h: np.ndarray, beta: float, ln: Dict) -> Dict:
    """
    G4: a V = I head in the reading's frame. Raw rows x (start √d·U0); each step the field on
    ``U = unit(LN1(x))``, the raw move ``h·|Cx|·X(U)``, read back through the same LN1. With
    w = 1, b = 0 this is G1 exactly (``unit(Cx + h|Cx|X) = retract(U + hX)``). Returns the
    per-lift reading plus the frame's start and end mean cosine.
    """
    w, b, eps = ln["w"], ln["b"], ln["eps"]
    x = math.sqrt(D) * U0
    U = frame(x, w, b, eps)
    g_start = mean_cos(U)
    acc = _acc()
    for hl in h:
        F = sa_field(U, beta, causal=True)
        k = np.linalg.norm(x - x.mean(axis=1, keepdims=True), axis=1, keepdims=True)
        x = x + hl * k * F
        U1 = frame(x, w, b, eps)
        _add(acc, U, U1, F)
        U = U1
    return {**_close(acc), "frame_g_start": g_start, "frame_g_end": mean_cos(U)}


def _rel(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))))


def g0_arm() -> List[Dict]:
    from . import sa_sim
    from .tstar_table import tstar as t69
    rows = []
    for key, n in NS.items():
        U0 = np.random.default_rng(SEED).standard_normal((n, D))
        for beta in G0_BETAS:
            full = sa_sim.tstar(U0, beta, causal=False)
            ref = t69(n, beta, 0.0)
            try:
                causal, conv = sa_sim.tstar(U0, beta, causal=True, check_dt=True), True
            except RuntimeError:
                causal, conv = float("nan"), False
            rel = abs(full - ref) / ref
            rows.append({"prompt": key, "n": n, "beta": beta, "full": full, "eq69": ref, "rel": rel,
                         "causal": causal, "dt_converged": conv,
                         "pass": bool(rel <= TOL["G0"] and conv)})
    return rows


def ln_all() -> Dict:
    """LN1 weight and bias of every block at the 18 steps, cached (weights only, CPU)."""
    p = DATA / LN_FILE
    if not p.exists():
        import torch
        from trackB_circuit.b0_gate import load
        w, b = [], []
        for s in STEPS:
            m, _ = load(s, "cpu", torch.float32)
            lay = m.gpt_neox.layers
            w.append([l.input_layernorm.weight.detach().double().numpy() for l in lay])
            b.append([l.input_layernorm.bias.detach().double().numpy() for l in lay])
            eps = float(lay[0].input_layernorm.eps)
            del m
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, steps=np.array(STEPS), w=np.array(w), b=np.array(b), eps=eps)
    z = np.load(p)
    if tuple(z["steps"].tolist()) != STEPS or z["w"].shape[1] != N_BLOCKS:
        raise SystemExit(f"refusing: {p} does not hold the 18 steps × {N_BLOCKS} blocks")
    sha = hashlib.sha256(z["w"].tobytes() + z["b"].tobytes()).hexdigest()[:12]
    return {"w": z["w"], "b": z["b"], "eps": float(z["eps"]), "sha": sha}


def _ln(lns: Dict, step: int, block: int) -> Dict:
    i = STEPS.index(step)
    return {"w": lns["w"][i, block], "b": lns["b"][i, block], "eps": lns["eps"]}


def _cell_seed(key: str, beta: float, g0: float) -> int:
    return SEED + 100 * list(NS).index(key) + 10 * BETAS.index(beta) + STARTS.index(g0)


def planted_arms(ln5: Dict) -> List[Dict]:
    fr5 = lambda Y: frame(Y, ln5["w"], ln5["b"], ln5["eps"])     # noqa: E731
    rows = []
    for key, n in NS.items():
        for beta in BETAS:
            for g0 in STARTS:
                seed = _cell_seed(key, beta, g0)
                h = planted_h(seed)
                T = float(h.sum())
                U0 = start(n, g0, np.random.default_rng(seed))
                tf = trajectory(U0, h, beta, "field", np.random.default_rng(seed + 1))
                g1 = read(tf, beta)
                g2 = read(trajectory(U0, h, beta, "null", np.random.default_rng(seed + 2)), beta)
                g3 = read(trajectory(U0, h, beta, "mixed", np.random.default_rng(seed + 3)), beta)
                g2s = read(trajectory(U0, h, beta, "drift", np.random.default_rng(seed + 4)), beta)
                g4raw = read(tf, beta, fr5)
                g4 = ln_plant(U0, h, beta, ln5)
                r = {"prompt": key, "n": n, "beta": beta, "g0": g0, "T": T, "g0_end": mean_cos(tf[-1]),
                     "G4_frame_g": [g4["frame_g_start"], g4["frame_g_end"]], "by_lift": {}}
                for l in LIFTS:
                    m = {"G1": {"h_proj_rel": _rel(g1[l]["proj"], h), "h_cal_rel": _rel(g1[l]["cal"], h),
                                "T_proj_rel": abs(g1[l]["T_proj"] - T) / T, "T_cal_rel": abs(g1[l]["T_cal"] - T) / T},
                         "G2": {"T_proj_over_T": g2[l]["T_proj"] / T, "T_cal_over_G1": g2[l]["T_cal"] / g1[l]["T_cal"]},
                         "G2s": {"T_proj_over_T": g2s[l]["T_proj"] / T},
                         "G3": {"h_proj_rel": _rel(g3[l]["proj"], h),
                                "h_cal_rel_sqrt2": _rel(g3[l]["cal"], math.sqrt(2) * h)},
                         "G4": {"T_proj_over_T": g4[l]["T_proj"] / T, "h_proj_rel": _rel(g4[l]["proj"], h)},
                         "G4raw": {"T_proj_over_T": g4raw[l]["T_proj"] / T}}
                    m["pass"] = {
                        "G1": m["G1"]["h_proj_rel"] <= TOL["G1_h"] and m["G1"]["h_cal_rel"] <= TOL["G1_h"]
                        and m["G1"]["T_proj_rel"] <= TOL["G1_T"] and m["G1"]["T_cal_rel"] <= TOL["G1_T"],
                        "G2": abs(m["G2"]["T_proj_over_T"]) <= TOL["G2_T"],
                        "G3": m["G3"]["h_proj_rel"] <= TOL["G3_h"] and m["G3"]["h_cal_rel_sqrt2"] <= TOL["G3_h"],
                        "G4": abs(m["G4"]["T_proj_over_T"] - 1.0) <= TOL["G4_T"]}
                    r["by_lift"][l] = m
                rows.append(r)
                print(f"{key} n={n} beta={beta} g0={g0}: {r['by_lift'][LIFT]['pass']} "
                      f"G4 {r['by_lift'][LIFT]['G4']['T_proj_over_T']:.3f}", flush=True)
    return rows


def _sweep_job(args) -> Dict:
    step, block, w, b, eps = args
    key = SWEEP_PROMPT
    out = {"step": step, "block": block, "cells": []}
    for beta in BETAS:
        for g0 in STARTS:
            seed = _cell_seed(key, beta, g0)
            h = planted_h(seed)
            g4 = ln_plant(start(NS[key], g0, np.random.default_rng(seed)), h, beta, {"w": w, "b": b, "eps": eps})
            out["cells"].append({"beta": beta, "g0": g0, "frame_g": [g4["frame_g_start"], g4["frame_g_end"]],
                                 **{l: g4[l]["T_proj"] / float(h.sum()) for l in LIFTS}})
    return out


def _one_thread() -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _pool(workers: int) -> ProcessPoolExecutor:
    """One BLAS thread per worker, set inside it: 8 workers at numpy's default (16 threads each)
    put the load at ~110 and took the sweep 1h52m (2026-10-14, LESSONS 2). An environment
    variable set around the constructor does not reach them: the executor starts its workers at
    the first submit, after it is restored."""
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_one_thread)


def sweep(lns: Dict, workers: int) -> List[Dict]:
    """Each line prints both ends and the count outside G4's bar: a min alone hid every
    overshoot of the first re-gate (2026-10-14, LESSONS 2). The record keeps every cell."""
    jobs = [(s, l, lns["w"][i, l], lns["b"][i, l], lns["eps"]) for i, s in enumerate(STEPS) for l in range(N_BLOCKS)]
    rows = []
    with _pool(workers) as pool:
        for r in pool.map(_sweep_job, jobs):
            v = [c[LIFT] for c in r["cells"]]
            bad = sum(abs(x - 1.0) > TOL["G4_T"] for x in v)
            tag = "" if r["block"] in T_BLOCKS else " (beside)"
            print(f"sweep step {r['step']} block {r['block']}{tag}: T_proj/T {min(v):.3f}-{max(v):.3f}, "
                  f"{bad} of {len(v)} outside", flush=True)
            rows.append(r)
    out = [(r["step"], r["block"], c["beta"], c["g0"], round(c[LIFT], 3))
           for r in rows if r["block"] in T_BLOCKS for c in r["cells"] if abs(c[LIFT] - 1.0) > TOL["G4_T"]]
    print(f"sweep: {len(out)} of {sum(len(r['cells']) for r in rows if r['block'] in T_BLOCKS)} gated cells "
          f"outside (step, block, beta, g0, T_proj/T): {out}", flush=True)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    lns = ln_all()
    g0 = g0_arm()
    print(f"G0: {sum(r['pass'] for r in g0)} of {len(g0)}", flush=True)
    rows = planted_arms(_ln(lns, 143000, 5))
    sw = sweep(lns, a.workers)
    sweep_ok = {l: all(abs(c[l] - 1.0) <= TOL["G4_T"] for r in sw if r["block"] in T_BLOCKS for c in r["cells"])
                for l in LIFTS}
    verdicts = {}
    for l in LIFTS:
        v = {"G0": all(r["pass"] for r in g0),
             **{g: all(r["by_lift"][l]["pass"][g] for r in rows) for g in ("G1", "G2", "G3", "G4")}}
        v["G4"] = v["G4"] and sweep_ok[l]
        verdicts[l] = v
    verdict = verdicts[LIFT]
    rec = {"code": code_sha(), "gate_code": gate_sha(), "lift": LIFT, "lifts": LIFTS, "ln_sha": lns["sha"],
           "seed": SEED, "tol": TOL, "ns": NS, "betas": BETAS, "starts": STARTS, "sweep_prompt": SWEEP_PROMPT,
           "verdict": verdict, "verdicts_beside": verdicts, "passes": all(verdict.values()),
           "G0": g0, "planted": rows, "sweep": sw, "seconds": time.monotonic() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "a1a_gate.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(verdicts), "passes" if rec["passes"] else "FAILS")
    return 0 if rec["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
