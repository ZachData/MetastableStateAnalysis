"""trackA_force/a3_weights.py — A3 unit 3, the weights-only rows' producer (`design-A3.md` §2
E7, E8; ΔW's stable rank beside).

Per checkpoint, layer and matrix type (QKV, O, MLP in, MLP out; the weight, not the bias):

- **E7** ``‖W_t − W_0‖_F / ‖W_0‖_F``;
- **E8** the ESD tail exponent α (HTSR, 1901.08276): the eigenvalues λ of W Wᵀ (the smaller
  Gram, 1024 × 1024 for every type), a continuous power law p(λ) ∝ λ^−α fitted to the tail
  λ ≥ λ_min by maximum likelihood, λ_min chosen by the least Kolmogorov–Smirnov distance
  (Clauset, Shalizi and Newman 2009, as WeightWatcher), over tails of at least ``MIN_TAIL``;
- *beside:* ΔW's stable rank ``‖ΔW‖_F² / ‖ΔW‖_2²`` (undefined at step 0, where ΔW = 0).

All 32 distinct steps on disk (`design-A3.md` §1) are computed; `a3.py` locates on the 18. CPU,
float64 Grams; the checkpoint is loaded float32 (`b0_gate.load`). Each step's record is written
as it finishes, so a stopped run resumes.

Tier 1: exploratory, unregistered. No forward pass.
    python -m trackA_force.a3_weights run --out <dir> [--steps 0 143000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .a1a import STEPS

EXTRA = (3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 40000, 60000, 80000,
         100000, 120000)
ALL_STEPS = tuple(sorted(set(STEPS) | set(EXTRA)))          # 32 distinct
TYPES = ("qkv", "o", "mlp_in", "mlp_out")
MIN_TAIL = 10            # placed: the least tail the α fit may choose


class WeightsRefused(RuntimeError):
    pass


def matrices(model, l: int) -> dict:
    b = model.gpt_neox.layers[l]
    return {"qkv": b.attention.query_key_value.weight, "o": b.attention.dense.weight,
            "mlp_in": b.mlp.dense_h_to_4h.weight, "mlp_out": b.mlp.dense_4h_to_h.weight}


def gram_eigs(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, np.float64)
    G = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
    return np.clip(np.linalg.eigvalsh(G), 0, None)


def powerlaw_alpha(lam: np.ndarray, min_tail: int = MIN_TAIL) -> dict:
    """Continuous power-law MLE on the tail, λ_min by least KS distance (Clauset et al. 2009)."""
    x = np.sort(np.asarray(lam, np.float64))
    x = x[x > 0]
    n = len(x)
    if n < min_tail + 1:
        raise WeightsRefused(f"{n} positive eigenvalues, fewer than {min_tail + 1}")
    logx = np.log(x)
    # suffix sums of log x: S[i] = Σ_{j ≥ i} log x_j
    S = np.cumsum(logx[::-1])[::-1]
    best = None
    for i in range(0, n - min_tail + 1):
        m = n - i
        a = 1 + m / (S[i] - m * logx[i])
        if not np.isfinite(a) or a <= 1:
            continue
        tail = x[i:]
        emp = np.arange(m) / m                                 # CDF just below each point
        fit = 1 - (tail / x[i]) ** (1 - a)
        D = float(max(np.max(np.abs(fit - emp)), np.max(np.abs(fit - (emp + 1 / m)))))
        if best is None or D < best["D"]:
            best = {"alpha": float(a), "xmin": float(x[i]), "n_tail": int(m), "D": D}
    if best is None:
        raise WeightsRefused("no tail with a finite α > 1")
    return best


def stable_rank(lam: np.ndarray) -> float:
    top = float(np.max(lam))
    return float(np.sum(lam) / top) if top > 0 else float("nan")


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def run(out: Path, steps) -> None:
    import torch
    from trackB_circuit.b0_gate import load
    out.mkdir(parents=True, exist_ok=True)
    code = code_sha()
    m0, _ = load(0, "cpu", torch.float32)
    L = len(m0.gpt_neox.layers)
    W0 = [{k: v.detach().double().numpy().copy() for k, v in matrices(m0, l).items()}
          for l in range(L)]
    del m0
    for s in steps:
        f = out / f"step{s}.json"
        if f.exists():
            if json.loads(f.read_text()).get("code") != code:
                raise WeightsRefused(f"refusing to resume: {f} was written by other code")
            continue
        t0 = time.time()
        m, _ = load(s, "cpu", torch.float32)
        if len(m.gpt_neox.layers) != L:
            raise WeightsRefused(f"step {s}: {len(m.gpt_neox.layers)} layers")
        rec = {"step": s, "code": code, "rel_change": {t: [] for t in TYPES},
               "alpha": {t: [] for t in TYPES}, "alpha_fit": {t: [] for t in TYPES},
               "dW_stable_rank": {t: [] for t in TYPES}}
        for l in range(L):
            for t, W in matrices(m, l).items():
                W = W.detach().double().numpy()
                dW = W - W0[l][t]
                rec["rel_change"][t].append(float(np.linalg.norm(dW) / np.linalg.norm(W0[l][t])))
                fit = powerlaw_alpha(gram_eigs(W))
                rec["alpha"][t].append(fit["alpha"])
                rec["alpha_fit"][t].append(fit)
                rec["dW_stable_rank"][t].append(stable_rank(gram_eigs(dW)) if s else None)
        del m
        f.write_text(json.dumps(rec) + "\n")
        med = {t: round(float(np.median(rec["alpha"][t])), 2) for t in TYPES}
        rc = {t: round(float(np.median(rec["rel_change"][t])), 4) for t in TYPES}
        print(f"step {s}: rel change {rc} alpha {med} ({time.time() - t0:.0f} s)", flush=True)
    print("done", flush=True)


def load_steps(out: Path, steps=ALL_STEPS) -> dict:
    recs = {}
    for s in steps:
        f = out / f"step{s}.json"
        if not f.exists():
            raise WeightsRefused(f"weights rows: no record {f}")
        recs[s] = json.loads(f.read_text())
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", nargs="*", type=int)
    a = ap.parse_args(argv)
    steps = tuple(a.steps) if a.steps else ALL_STEPS
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        raise SystemExit(f"refusing: steps {bad} are not on the 32-step axis")
    run(a.out, steps)


if __name__ == "__main__":
    main()
