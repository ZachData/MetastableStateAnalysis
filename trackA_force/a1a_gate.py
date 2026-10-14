"""
trackA_force/a1a_gate.py — A1a's calibration gate, run before any real reading
(`design-A.md` "The calibration gate"; `CLAUDE.md` "Calibration gate").

Synthetic clouds at the real design points (n = each v1 prompt's length less position 0,
d = 1024), planted trajectories of 23 forward-Euler steps with retraction to the sphere:

- G0, comparator: `sa_sim` with full attention from a Gaussian start against eq. (6.9) at
  β ∈ {0, 3.5, 5.6}, within 3 %; the causal arm's dt check converges.
- G1, planted effect: the identity-weight causal field at known h_l (log-uniform in
  [0.02, 0.5]), from an orthogonal start and from pairwise cosine 0.3; ``proj`` and ``cal``
  recover each h_l within 5 % and T within 3 %.
- G2, planted null: the same move norms per token, directions uniform in the tangent plane;
  |T_proj| ≤ 0.1 T (``cal`` beside: it cannot see direction).
- G3, mixed: ``Δ = hX + ξ``, ξ tangent and ⊥ X with |ξ| = |hX|; ``proj`` within 5 % of h,
  ``cal`` within 5 % of √2 h.
- G4, frame: G1's trajectory read through step 143000's block-5 LN1 before unit-normalising;
  T_proj within 10 % of T.

    python -m trackA_force.a1a_gate --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from p1c_frames.integration_time import sa_field, unit_steps

from .a1a import code_sha, frame, mean_cos

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


def _tangent_noise(U: np.ndarray, norms: np.ndarray, rng, also_perp: Optional[np.ndarray] = None) -> np.ndarray:
    """Per row a uniform direction tangent at U (and ⊥ ``also_perp`` if given), scaled to ``norms``."""
    Z = rng.standard_normal(U.shape)
    Z -= np.sum(Z * U, axis=1, keepdims=True) * U
    if also_perp is not None:
        P = also_perp / np.maximum(np.linalg.norm(also_perp, axis=1, keepdims=True), 1e-300)
        Z -= np.sum(Z * P, axis=1, keepdims=True) * P
    return Z / np.linalg.norm(Z, axis=1, keepdims=True) * norms[:, None]


def trajectory(U0: np.ndarray, h: np.ndarray, beta: float, arm: str, rng) -> List[np.ndarray]:
    """N_STEPS + 1 configurations; ``arm`` is ``field``, ``null`` or ``mixed``."""
    out = [U0]
    U = U0
    for hl in h:
        F = sa_field(U, beta, causal=True)
        hX = hl * F
        if arm == "field":
            V = hX
        elif arm == "null":
            V = _tangent_noise(U, np.linalg.norm(hX, axis=1), rng)
        elif arm == "mixed":
            V = hX + _tangent_noise(U, np.linalg.norm(hX, axis=1), rng, also_perp=F)
        else:
            raise ValueError(arm)
        U = _retract(U, V)
        out.append(U)
    return out


def read(traj: Sequence[np.ndarray], beta: float, fr: Optional[Callable] = None,
         lift: str = "tangent") -> Dict:
    """Per-step medians of ``proj`` and ``cal`` and their sums, in frame ``fr`` (identity if None)."""
    fr = fr or (lambda Y: Y)
    proj, cal = [], []
    for a, b in zip(traj[:-1], traj[1:]):
        U = fr(a)
        s = unit_steps(U, fr(b), sa_field(U, beta, causal=True), lift=lift)
        proj.append(float(np.nanmedian(s["proj"])))
        cal.append(float(np.nanmedian(s["cal"])))
    return {"proj": proj, "cal": cal, "T_proj": float(sum(proj)), "T_cal": float(sum(cal))}


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


def ln_143000_block5() -> Dict:
    from trackB_circuit.b0_gate import load
    import torch
    m, _ = load(143000, "cpu", torch.float32)
    ln = m.gpt_neox.layers[5].input_layernorm
    return {"w": ln.weight.detach().double().numpy(), "b": ln.bias.detach().double().numpy(),
            "eps": float(ln.eps)}


def planted_arms(ln5: Dict, lift: str = "tangent") -> List[Dict]:
    fr5 = lambda Y: frame(Y, ln5["w"], ln5["b"], ln5["eps"])     # noqa: E731
    rows = []
    for i, (key, n) in enumerate(NS.items()):
        for j, beta in enumerate(BETAS):
            for k, g0 in enumerate(STARTS):
                seed = SEED + 100 * i + 10 * j + k
                h = planted_h(seed)
                T = float(h.sum())
                U0 = start(n, g0, np.random.default_rng(seed))
                tf = trajectory(U0, h, beta, "field", np.random.default_rng(seed + 1))
                g1 = read(tf, beta, lift=lift)
                g2 = read(trajectory(U0, h, beta, "null", np.random.default_rng(seed + 2)), beta, lift=lift)
                g3 = read(trajectory(U0, h, beta, "mixed", np.random.default_rng(seed + 3)), beta, lift=lift)
                g4 = read(tf, beta, fr5, lift=lift)
                r = {"prompt": key, "n": n, "beta": beta, "g0": g0, "T": T,
                     "g0_end": mean_cos(tf[-1]),
                     "G1": {"h_proj_rel": _rel(g1["proj"], h), "h_cal_rel": _rel(g1["cal"], h),
                            "T_proj_rel": abs(g1["T_proj"] - T) / T, "T_cal_rel": abs(g1["T_cal"] - T) / T},
                     "G2": {"T_proj_over_T": g2["T_proj"] / T, "T_cal_over_G1": g2["T_cal"] / g1["T_cal"]},
                     "G3": {"h_proj_rel": _rel(g3["proj"], h), "h_cal_rel_sqrt2": _rel(g3["cal"], math.sqrt(2) * h)},
                     "G4": {"T_proj_rel": abs(g4["T_proj"] - T) / T, "T_proj_over_T": g4["T_proj"] / T}}
                r["pass"] = {
                    "G1": r["G1"]["h_proj_rel"] <= TOL["G1_h"] and r["G1"]["h_cal_rel"] <= TOL["G1_h"]
                    and r["G1"]["T_proj_rel"] <= TOL["G1_T"] and r["G1"]["T_cal_rel"] <= TOL["G1_T"],
                    "G2": abs(r["G2"]["T_proj_over_T"]) <= TOL["G2_T"],
                    "G3": r["G3"]["h_proj_rel"] <= TOL["G3_h"] and r["G3"]["h_cal_rel_sqrt2"] <= TOL["G3_h"],
                    "G4": r["G4"]["T_proj_rel"] <= TOL["G4_T"]}
                rows.append(r)
                print(f"{key} n={n} beta={beta} g0={g0}: {r['pass']}", flush=True)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lift", choices=("tangent", "log"), default="tangent")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    g0 = g0_arm()
    print(f"G0: {sum(r['pass'] for r in g0)} of {len(g0)}", flush=True)
    rows = planted_arms(ln_143000_block5(), a.lift)
    verdict = {"G0": all(r["pass"] for r in g0),
               **{g: all(r["pass"][g] for r in rows) for g in ("G1", "G2", "G3", "G4")}}
    import hashlib
    gate_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    rec = {"code": code_sha(), "gate_code": gate_sha, "lift": a.lift, "seed": SEED, "tol": TOL, "ns": NS, "betas": BETAS, "starts": STARTS,
           "verdict": verdict, "passes": all(verdict.values()), "G0": g0, "planted": rows,
           "seconds": time.monotonic() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"a1a_gate_{a.lift}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(verdict), "passes" if rec["passes"] else "FAILS")
    return 0 if rec["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
