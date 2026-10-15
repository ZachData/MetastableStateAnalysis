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
  raw residual, then read back through the same LN1. T within 10 % on step 143000's block-5 LN1
  at all 42 cells. Beside: the earlier G4 (G1's raw trajectory read through block 5's LN1, "G4raw").
- G4c, the chain (Blocked 47 (d)): per training step, the planted cloud passes blocks 0–23 in
  turn, one planted step per block in that block's LN1 frame, read as `a1a.block_steps` reads
  the real stream; T = Σ h over blocks 0–22 (block 23 beside) within 10 %, at all 7 prompts ×
  3 β × 2 starts × the 18 steps (756 totals). Torch float64 on the GPU, checked against the
  numpy path (`ln_chain_np`) on one cell before the sweep.

Every arm is read twice: ``proj`` (median over tokens) and ``proj_c`` (the slope on the field
with one shared drift removed, `integration_time.drift_free_slope`; Blocked 47), both gated at
the same bars. G2s, a shared drift, is beside on both reads with no bar (Blocked 48 (a): held 23
steps the drift collapses the cloud onto its target, where drift and field are the same move, so
it is not a null; its numbers are recorded, ``G2s_within_0.1`` beside the verdict). Three lifts (`integration_time.unit_steps`): the verdict is
``LIFT`` (log, Blocked 46 (b)); tangent and gnomonic beside.

    python -m trackA_force.a1a_gate --out <dir> [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from p1c_frames.integration_time import drift_free_slope, sa_field, unit_steps

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
READS = ("proj", "proj_c")
CHAIN_PROBE_TOL = 1e-9              # torch against numpy on one G4c cell, before the sweep
LN_FILE = "trackA/ln1_18steps.npz"


def gate_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def planted_h(seed: int, k: int = N_STEPS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(math.log(H_LO), math.log(H_HI), k))


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
    return {l: {"proj": [], "cal": [], "proj_c": []} for l in LIFTS}


def _add(acc, U, U1, F) -> None:
    for l in LIFTS:
        s = unit_steps(U, U1, F, lift=l)
        acc[l]["proj"].append(float(np.nanmedian(s["proj"])))
        acc[l]["cal"].append(float(np.nanmedian(s["cal"])))
        acc[l]["proj_c"].append(drift_free_slope(U, U1, F, lift=l))


def _close(acc) -> Dict:
    return {l: {**v, "T_proj": float(sum(v["proj"])), "T_cal": float(sum(v["cal"])),
                "T_proj_c": float(sum(v["proj_c"]))} for l, v in acc.items()}


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
                                "h_proj_c_rel": _rel(g1[l]["proj_c"], h),
                                "T_proj_rel": abs(g1[l]["T_proj"] - T) / T, "T_cal_rel": abs(g1[l]["T_cal"] - T) / T,
                                "T_proj_c_rel": abs(g1[l]["T_proj_c"] - T) / T},
                         "G2": {"T_proj_over_T": g2[l]["T_proj"] / T, "T_proj_c_over_T": g2[l]["T_proj_c"] / T,
                                "T_cal_over_G1": g2[l]["T_cal"] / g1[l]["T_cal"]},
                         "G2s": {"T_proj_over_T": g2s[l]["T_proj"] / T, "T_proj_c_over_T": g2s[l]["T_proj_c"] / T},
                         "G3": {"h_proj_rel": _rel(g3[l]["proj"], h), "h_proj_c_rel": _rel(g3[l]["proj_c"], h),
                                "h_cal_rel_sqrt2": _rel(g3[l]["cal"], math.sqrt(2) * h)},
                         "G4": {"T_proj_over_T": g4[l]["T_proj"] / T, "T_proj_c_over_T": g4[l]["T_proj_c"] / T,
                                "h_proj_rel": _rel(g4[l]["proj"], h)},
                         "G4raw": {"T_proj_over_T": g4raw[l]["T_proj"] / T}}
                    m["pass"] = {
                        "G1": m["G1"]["h_proj_rel"] <= TOL["G1_h"] and m["G1"]["h_cal_rel"] <= TOL["G1_h"]
                        and m["G1"]["h_proj_c_rel"] <= TOL["G1_h"]
                        and m["G1"]["T_proj_rel"] <= TOL["G1_T"] and m["G1"]["T_cal_rel"] <= TOL["G1_T"]
                        and m["G1"]["T_proj_c_rel"] <= TOL["G1_T"],
                        "G2": abs(m["G2"]["T_proj_over_T"]) <= TOL["G2_T"] and abs(m["G2"]["T_proj_c_over_T"]) <= TOL["G2_T"],
                        "G3": m["G3"]["h_proj_rel"] <= TOL["G3_h"] and m["G3"]["h_proj_c_rel"] <= TOL["G3_h"]
                        and m["G3"]["h_cal_rel_sqrt2"] <= TOL["G3_h"],
                        "G4": abs(m["G4"]["T_proj_over_T"] - 1.0) <= TOL["G4_T"]
                        and abs(m["G4"]["T_proj_c_over_T"] - 1.0) <= TOL["G4_T"]}
                    r["by_lift"][l] = m
                rows.append(r)
                m = r["by_lift"][LIFT]
                print(f"{key} n={n} beta={beta} g0={g0}: {m['pass']} G4 {m['G4']['T_proj_over_T']:.3f}/"
                      f"{m['G4']['T_proj_c_over_T']:.3f} G2s {m['G2s']['T_proj_over_T']:.3f}/"
                      f"{m['G2s']['T_proj_c_over_T']:.3f} (proj/proj_c)", flush=True)
    return rows


def chain_h(key: str, beta: float, g0: float) -> np.ndarray:
    """G4c's 24 planted steps for a cell, one per block (T sums blocks 0–22)."""
    return planted_h(_cell_seed(key, beta, g0) + 5, N_BLOCKS)


def ln_chain_np(U0: np.ndarray, h: np.ndarray, beta: float, w: np.ndarray, b: np.ndarray, eps: float) -> Dict:
    """
    G4c's reference path (numpy): raw rows x (start √d·U0) pass blocks 0…len(h)−1; in block l's
    frame ``U = unit(LN1_l(x))`` the field, the raw move ``h_l·|Cx|·X(U)``, read back through
    LN1_l, as `a1a.block_steps` reads one block. ``w``, ``b``: (blocks, d). Per lift and read,
    one value per block.
    """
    x = math.sqrt(D) * U0
    out = {l: {r: [] for r in READS} for l in LIFTS}
    g_start = mean_cos(frame(x, w[0], b[0], eps))
    for L, hl in enumerate(h):
        U = frame(x, w[L], b[L], eps)
        F = sa_field(U, beta, causal=True)
        k = np.linalg.norm(x - x.mean(axis=1, keepdims=True), axis=1, keepdims=True)
        x = x + hl * k * F
        U1 = frame(x, w[L], b[L], eps)
        for l in LIFTS:
            out[l]["proj"].append(float(np.nanmedian(unit_steps(U, U1, F, lift=l)["proj"])))
            out[l]["proj_c"].append(drift_free_slope(U, U1, F, lift=l))
    return {**out, "frame_g": [g_start, mean_cos(U1)]}


def _t_frame(x, w, b, eps):
    import torch
    mu = x.mean(dim=-1, keepdim=True)
    y = (x - mu) / torch.sqrt(((x - mu) ** 2).mean(dim=-1, keepdim=True) + eps) * w[:, None, :] + b[:, None, :]
    return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _t_field(U, beta: float):
    import torch
    n = U.shape[1]
    S = beta * (U @ U.transpose(1, 2))
    S = S.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool, device=U.device), 1), -math.inf)
    tgt = torch.softmax(S, dim=-1) @ U
    return tgt - (tgt * U).sum(-1, keepdim=True) * U


def _t_lifted(U, U1, lift: str):
    import torch
    if lift == "gnomonic":
        c = (U * U1).sum(-1, keepdim=True)
        Dm = U1 / torch.where(c > 0, c, torch.full_like(c, math.nan)) - U
    else:
        Dm = U1 - U
    Dm = Dm - (Dm * U).sum(-1, keepdim=True) * U
    if lift == "log":
        th = torch.arccos((U * U1).sum(-1, keepdim=True).clamp(-1.0, 1.0))
        Dm = Dm / Dm.norm(dim=-1, keepdim=True).clamp_min(1e-300) * th
    return Dm


def _t_reads(Dm, F):
    """Per batch row: median ``proj`` and the drift-free slope, as numpy's `unit_steps` / `drift_free_slope`."""
    import torch
    xx = (F * F).sum(-1)
    proj = (Dm * F).sum(-1) / torch.where(xx < 1e-18, torch.full_like(xx, math.nan), xx)
    med = torch.nanquantile(proj, 0.5, dim=1)
    ok = (torch.isfinite(Dm).all(-1) & (xx >= 1e-18)).unsqueeze(-1)
    cnt = ok.sum(1, keepdim=True)
    Dz = torch.where(ok, Dm, torch.zeros_like(Dm))
    Fz = torch.where(ok, F, torch.zeros_like(F))
    Dc = torch.where(ok, Dz - Dz.sum(1, keepdim=True) / cnt, torch.zeros_like(Dm))
    Fc = torch.where(ok, Fz - Fz.sum(1, keepdim=True) / cnt, torch.zeros_like(F))
    den = (Fc * Fc).sum((1, 2))
    slope = (Dc * Fc).sum((1, 2)) / torch.where(den > 1e-18, den, torch.full_like(den, math.nan))
    return med, slope


def ln_chain(U0: np.ndarray, h: np.ndarray, beta: float, w: np.ndarray, b: np.ndarray, eps: float,
             device: str) -> List[Dict]:
    """`ln_chain_np` in torch float64, batched over a leading axis of LN sets: ``w``, ``b`` are
    (S, blocks, d), one chain per S. Returns one `ln_chain_np`-shaped dict per S."""
    import torch
    dt = torch.float64
    W = torch.as_tensor(w, dtype=dt, device=device)
    Bb = torch.as_tensor(b, dtype=dt, device=device)
    S = W.shape[0]
    x = (math.sqrt(D) * torch.as_tensor(U0, dtype=dt, device=device)).expand(S, -1, -1).clone()
    res = {l: {r: [] for r in READS} for l in LIFTS}

    def mc(U):
        n = U.shape[1]
        s = U.sum(1)
        return ((s * s).sum(-1) - n) / (n * (n - 1))
    g_start = mc(_t_frame(x, W[:, 0], Bb[:, 0], eps))
    for L, hl in enumerate(h):
        U = _t_frame(x, W[:, L], Bb[:, L], eps)
        F = _t_field(U, beta)
        k = (x - x.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)
        x = x + float(hl) * k * F
        U1 = _t_frame(x, W[:, L], Bb[:, L], eps)
        for l in LIFTS:
            med, slope = _t_reads(_t_lifted(U, U1, l), F)
            res[l]["proj"].append(med.cpu().numpy())
            res[l]["proj_c"].append(slope.cpu().numpy())
    g_end = mc(U1).cpu().numpy()
    g_start = g_start.cpu().numpy()
    return [{**{l: {r: [float(v[i]) for v in res[l][r]] for r in READS} for l in LIFTS},
             "frame_g": [float(g_start[i]), float(g_end[i])]} for i in range(S)]


def _chain_T(c: Dict, h: np.ndarray) -> Dict:
    T = float(h[list(T_BLOCKS)].sum())
    return {l: {r: float(sum(c[l][r][L] for L in T_BLOCKS)) / T for r in READS} for l in LIFTS}


def chain_probe(lns: Dict, device: str) -> float:
    """Torch against numpy on one G4c cell (the first G4's lowest: β 7, start 0) at step 143000,
    before the sweep (`design-A.md` "Blocked 47 decided", Device)."""
    key, beta, g0 = "latex_monograph", 7.0, 0.0
    h = chain_h(key, beta, g0)
    U0 = start(NS[key], g0, np.random.default_rng(_cell_seed(key, beta, g0)))
    i = STEPS.index(143000)
    ref = ln_chain_np(U0, h, beta, lns["w"][i], lns["b"][i], lns["eps"])
    got = ln_chain(U0, h, beta, lns["w"][i:i + 1], lns["b"][i:i + 1], lns["eps"], device)[0]
    return max(abs(a - b) for l in LIFTS for r in READS for a, b in zip(ref[l][r], got[l][r]))


def chain(lns: Dict, device: str) -> List[Dict]:
    """G4c: every prompt × β × start, the 18 steps batched on ``device``. Each line prints both
    ends and the count outside the bar for both reads (LESSONS 2: a min alone hid overshoots)."""
    rows = []
    for key, n in NS.items():
        for beta in BETAS:
            for g0 in STARTS:
                h = chain_h(key, beta, g0)
                U0 = start(n, g0, np.random.default_rng(_cell_seed(key, beta, g0)))
                cs = ln_chain(U0, h, beta, lns["w"], lns["b"], lns["eps"], device)
                for s, c in zip(STEPS, cs):
                    rows.append({"prompt": key, "n": n, "beta": beta, "g0": g0, "step": s,
                                 "T": float(h[list(T_BLOCKS)].sum()), "h": h.tolist(),
                                 "frame_g": c["frame_g"], "per_block": {l: c[l] for l in LIFTS},
                                 "T_over_T": _chain_T(c, h)})
                v = {r: [x["T_over_T"][LIFT][r] for x in rows[-len(STEPS):]] for r in READS}
                print(f"G4c {key} beta={beta} g0={g0}: " + "; ".join(
                    f"{r} {min(v[r]):.3f}-{max(v[r]):.3f}, {sum(abs(x - 1) > TOL['G4_T'] for x in v[r])} "
                    f"of {len(v[r])} outside" for r in READS), flush=True)
    for r in READS:
        out = [(x["prompt"], x["step"], x["beta"], x["g0"], round(x["T_over_T"][LIFT][r], 3))
               for x in rows if abs(x["T_over_T"][LIFT][r] - 1.0) > TOL["G4_T"]]
        print(f"G4c {r}: {len(out)} of {len(rows)} outside (prompt, step, beta, g0, T/T): {out}", flush=True)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda", help="G4c's device (the GPU; cpu only for tests)")
    a = ap.parse_args(argv)
    import torch
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device for G4c (pass --device cpu to run it on the CPU, slowly)")
    t0 = time.monotonic()
    lns = ln_all()
    probe = chain_probe(lns, a.device)
    print(f"G4c probe, {a.device} against numpy on one cell: max |diff| {probe:.1e}", flush=True)
    if not probe <= CHAIN_PROBE_TOL:
        raise SystemExit(f"refusing: G4c's torch path differs from numpy by {probe:.1e} > {CHAIN_PROBE_TOL}")
    t_chain = time.monotonic()
    ch = chain(lns, a.device)
    t_chain = time.monotonic() - t_chain
    print(f"G4c: {t_chain:.0f}s on {a.device}", flush=True)
    g0 = g0_arm()
    print(f"G0: {sum(r['pass'] for r in g0)} of {len(g0)}", flush=True)
    rows = planted_arms(_ln(lns, 143000, 5))
    verdicts = {}
    for l in LIFTS:
        v = {"G0": all(r["pass"] for r in g0),
             **{g: all(r["by_lift"][l]["pass"][g] for r in rows) for g in ("G1", "G2", "G3", "G4")},
             "G4c": all(abs(x["T_over_T"][l][r] - 1.0) <= TOL["G4_T"] for x in ch for r in READS)}
        verdicts[l] = v
    g2s_beside = {l: sum(abs(r["by_lift"][l]["G2s"]["T_proj_c_over_T"]) <= TOL["G2_T"] for r in rows)
                  for l in LIFTS}
    verdict = verdicts[LIFT]
    rec = {"code": code_sha(), "gate_code": gate_sha(), "lift": LIFT, "lifts": LIFTS, "reads": READS,
           "ln_sha": lns["sha"], "seed": SEED, "tol": TOL, "ns": NS, "betas": BETAS, "starts": STARTS,
           "device": a.device, "chain_probe": probe, "chain_seconds": t_chain,
           "verdict": verdict, "verdicts_beside": verdicts, "passes": all(verdict.values()),
           "G2s_within_0.1": {"cells": len(rows), "proj_c_within": g2s_beside, "gated": False},
           "G0": g0, "planted": rows, "chain": ch, "seconds": time.monotonic() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "a1a_gate.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(verdicts), "passes" if rec["passes"] else "FAILS")
    return 0 if rec["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
