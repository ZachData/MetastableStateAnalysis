"""B2a, the 2D spike, unit 1 (`trackB_circuit/design-B2a.md`, frozen 2026-10-15, Blocked 50 (a)):
V's ladder, the reduced attention M1 (primary) and the head's own QK on U (M2, beside), the
Haar nulls and the rule, run on planted heads (the gate) and on the weights of the 14 heads over
the 18 steps. No real token is read here; unit 2 reads them.

**V of head h** in block l's unit LN1 frame: ``V_h = diag(g_l) C W_O^h W_V^h`` (biases out), kept
as ``left = diag(g) C W_O^h`` (d, 64) and ``right = W_V^h`` (64, d). Its nonzero spectrum is that of
``right @ left`` (64 × 64); an eigenvector w there lifts to the invariant direction ``left @ w``.

**V's ladder**: the nonzero spectrum's blocks (a real eigenline, D += 1; a complex pair, D += 2,
kept whole) ordered by ``|λ|``, largest first (ties in ``|λ|`` by eig's order, so a degenerate
eigenspace is split at an arbitrary but invariant line); ``U_D`` the orthonormalised span of the
first blocks of total dimension D; rungs `LADDER` capped at the rank, a rung inside a pair moved
up by one. Each U's invariance residual ``‖(I − UUᵀ) V U‖ / ‖V U‖`` is recorded, refused above
`INV_TOL`.

**M1** (the paper's class, Q = K = I on U): ``log Â_ij = β · cos(Uᵀu_i, Uᵀu_j) + φ(i − j) − log Z_i``,
causal; φ piecewise constant on `n_bins` offset bins (0 … 15 one each, then [16, 32), [32, 64), …,
[1024, 2048)). The logit is linear in (β, φ), so the cross-entropy against the real rows is convex:
damped Newton, ridge `RIDGE`, fitted leave one prompt out. TV per row ``½ Σ_j |A_ij − Â_ij|``;
the head's statistic is the mean over prompts of the per-prompt mean over rows.

**The rule** (design "The rule", constants placed there): outside the class if TV(full) > 0.5,
or if neither M1's content term nor its offset profile adds anything and the head is not fitted
outright: TV(0) − TV(full) < 0.05, TV(uniform) − TV(0) < 0.05 and TV(full) > 0.05 (Blocked 51 (a),
before any real token: attention M1 cannot express otherwise reads D* = 0, "positional"; gate arm
G4w; amended by Blocked 52 (a′) after one real head, `L5H2`, read "outside" under (a); arm G2n,
a positional profile plus content noise, must read D* = 0); else D* = the smallest rung with TV(D) ≤ TV(full) + 0.05 and, for D ≥ 1, below all 320 Haar
subspaces at that D (each fitted the same way; ties count against). Nulls run in chunks and stop
at the first null that ties or beats V's rung (the verdict needs no more); ``--all-nulls`` runs
every one.

**Corrections to the gate's construction, made before any output** (`design-B2a.md` "Unit 1's
gate, as built"): G1–G1d's tokens are placed in the planted block (uniform on its unit sphere,
`G_OFF` of each token's norm² off it) instead of run through the paper's map, because that map
cannot give spread tokens near the plane with this V: at the design's ν = 2, γ = 0.3 the plane's
``|b| = |1 − γ − γν| = 0.1`` is below the kernel's 1 − γ = 0.7 and the tokens leave it (median
occupancy 1e-6 after 20 steps, scratch run); at ν = 8 they reach it but collapse to one cluster
flipping sign each step (``γ_eff = γν/(1 − γ + γν) = 0.77 > ½`` contracts deviations), whose
attention is near uniform, so D* = 0 was the right reading (first G1 run, TV(0) = 0.04). G3's isotropic
tokens make the paper's attention at β = 4 near uniform plus self (off-diagonal cosines ±1/32), so
D = 0 would be faithful and "none" could not be the right answer; G3's tokens lie in a random
8-dimensional subspace unrelated to V instead, so its attention reads content.

NUMERICS: float64 throughout; the gate's fits on the GPU (the memory rule; the design said CPU
because the gate is small, and float64 is kept), the weights on CPU. Checkpoints load in float32
(HF's float64 causal mask overflows in transformers 4.57) and their weights are cast to float64,
exactly; G5 (iii) compares with HF's float32 attention, so its statistic is the difference over
float32's eps times the head's largest |logit| (two reference heads reach |logit| ~ 3e3–8e3 on
random tokens, where one float32 ulp of the logit is ~5e-4 in attention), passing at `G5_HF_TOL`.

Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b2a gate    --out <dir> [--device cuda] [--arms G1 ...] [--all-nulls]
    python -m trackB_circuit.b2a weights --out <dir> [--steps 143000 ...]
    python -m trackB_circuit.b2a real    --out <dir> --gate <gate.json> [--g0-only] [--all-nulls]
    python -m trackB_circuit.b2a occupancy --out <dir> [--steps ...] [--unit2 <real.json>]
    python -m trackB_circuit.b2a report  --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))

STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
N_LAYERS, N_HEADS, HEAD_DIM, D_MODEL = 24, 16, 64, 1024
ROT_DIMS, ROT_BASE = 16, 10000.0          # Pythia: rotary_pct 0.25 of 64, base 10000

NAMED = ("L5H2", "L7H8", "L12H5", "L8H6", "L11H14", "L10H7")
REF_SEED, N_REF, REF_LAYERS = 20261015, 8, tuple(range(23))

LADDER = (0, 1, 2, 3, 4, 6, 8, 16, 32, 64)
TV_OUT = 0.5                              # placed (design): outside the class above it
TV_TOL = 0.05                             # placed (design): D's allowance over TV(full)
K_NULL, NULL_CHUNK = 320, 32
GAMMA_W = 0.1                             # placed (design): unit 1's γ for the |b| ordering
DEGEN_TOL = 0.05                          # placed here: two real λ within 5 % are "degenerate"
RANK_RTOL = 1e-9                          # |λ| below this × the top is the kernel
INV_TOL = 1e-8
RIDGE = 1e-6
MAX_IT = 60
G5_TOL = 1e-10
G5_HF_TOL = 10.0                          # placed here: (iii) in units of float32 eps × the head's max |logit|

# the gate (design "Unit 1's gate"; ν and G3's tokens as built, see the docstring)
G_D, G_N, G_PROMPTS = 1024, 500, 7
G_BETA, G_NU, G_PHI, G_REST, G_RANK = 4.0, 2.0, 2.5, 0.5, 64
G_OFF = 0.1                               # G1–G1d: share of each token's norm² off the block
G2_PEAK = 5.0                             # G2: φ*(1) = 5, every other offset 0
G3_SEEDS, G3_SUB, G3_PASS = 20, 8, 19
G4_SD = 3.0                               # G4: logit = 3 uᵢᵀ G uⱼ, G standard normal
PLANE_TOL = 1e-3                          # rad
G0_TOL = G5_HF_TOL                        # placed (design "Fixed for unit 2"): G5 (iii)'s bound
A1A_RECORDS = Path(os.environ.get("METS_DATA", str(REPO / "data"))) / "trackA/a1a_batch_2026-10-15/records"
G4W_SD = (0.5, 1.0)                       # placed, G4w (Blocked 51 (a)): weak random QK, must read outside;
                                          # G2n (Blocked 52 (a')): G2's profile plus it, must read D* = 0
ARMS = ("G1", "G1r", "G1d", "G2", "G2n", "G3", "G4", "G4w", "G5")


class B2aError(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackB_circuit/b2a.py").read_bytes()).hexdigest()[:12]


def seed_for(*parts) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def parse_head(name: str) -> Tuple[int, int]:
    l, h = name[1:].split("H")
    return int(l), int(h)


def reference_heads() -> Tuple[str, ...]:
    """8 heads from the other 362 of layers 0–22, ``numpy`` seed 20261015, in draw order."""
    pool = [f"L{l}H{h}" for l in REF_LAYERS for h in range(N_HEADS) if f"L{l}H{h}" not in NAMED]
    if len(pool) != 362:
        raise B2aError(f"reference pool has {len(pool)} heads, not 362")
    idx = np.random.default_rng(REF_SEED).choice(len(pool), N_REF, replace=False)
    return tuple(pool[i] for i in idx)


def heads() -> Tuple[str, ...]:
    return NAMED + reference_heads()


# ---------------------------------------------------------------- V's ladder (numpy, float64)

def blocks(left: np.ndarray, right: np.ndarray, rtol: float = RANK_RTOL) -> List[Dict]:
    """V = left @ right's nonzero invariant blocks, ordered by |λ| (largest first)."""
    lam, W = np.linalg.eig(right @ left)
    top = float(np.abs(lam).max())
    out = []
    for a in np.argsort(-np.abs(lam), kind="stable"):
        l = complex(lam[a])
        if abs(l) <= rtol * top:
            continue
        if abs(l.imag) <= 1e-10 * top:
            v = W[:, a]
            v = (v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))).real
            out.append({"lam": complex(l.real, 0.0), "dim": 1, "vecs": v[:, None]})
        elif l.imag > 0:
            out.append({"lam": l, "dim": 2, "vecs": np.stack([W[:, a].real, W[:, a].imag], 1)})
    return out


def invariance_residual(left: np.ndarray, right: np.ndarray, U: np.ndarray) -> float:
    VU = left @ (right @ U)
    return float(np.linalg.norm(VU - U @ (U.T @ VU)) / max(np.linalg.norm(VU), 1e-300))


def ladder(left: np.ndarray, right: np.ndarray, rungs: Sequence[int] = LADDER) -> List[Dict]:
    """[{D, U (d, D), n_blocks, residual}] ascending in D, rungs deduplicated after moving up."""
    bl = blocks(left, right)
    cum = np.cumsum([b["dim"] for b in bl])
    d = left.shape[0]
    out: Dict[int, Dict] = {}
    for D in rungs:
        if D == 0:
            out[0] = {"D": 0, "U": np.zeros((d, 0)), "n_blocks": 0, "residual": 0.0}
            continue
        k = int(np.searchsorted(cum, min(D, int(cum[-1]))))
        Dp = int(cum[k])
        if Dp in out:
            continue
        U, _ = np.linalg.qr(left @ np.concatenate([b["vecs"] for b in bl[:k + 1]], 1))
        res = invariance_residual(left, right, U)
        if res > INV_TOL:
            raise B2aError(f"U_{Dp} not invariant: residual {res:.2e}")
        out[Dp] = {"D": Dp, "U": U, "n_blocks": k + 1, "residual": res}
    return [out[k] for k in sorted(out)]


def u2_type(bl: List[Dict]) -> Dict:
    """The type of V's top two dimensions: a rotation pair, a degenerate real pair, or mixed."""
    b0 = bl[0]
    if b0["dim"] == 2:
        return {"type": "rotation", "phi": abs(math.atan2(b0["lam"].imag, b0["lam"].real))}
    if len(bl) < 2 or bl[1]["dim"] != 1:
        return {"type": "mixed", "phi": None}
    l1, l2 = b0["lam"].real, bl[1]["lam"].real
    if l1 * l2 > 0 and abs(l1 - l2) <= DEGEN_TOL * abs(l1):
        return {"type": "degenerate negative" if l1 < 0 else "degenerate positive", "phi": None}
    return {"type": "mixed", "phi": None}


def eigs_sorted(bl: List[Dict]) -> List[complex]:
    out = []
    for b in bl:
        out += [b["lam"]] if b["dim"] == 1 else [b["lam"], b["lam"].conjugate()]
    return out


def selection(bl: List[Dict], gamma: float, d: int) -> Dict:
    """The paper's aligned-cluster amplification ``|b_a| = |1 − γ + γλ_a|`` (2609.28448 §6.1)."""
    b = np.array([abs(1 - gamma + gamma * x["lam"]) for x in bl])
    rank = int(sum(x["dim"] for x in bl))
    kern = (1 - gamma) if rank < d else 0.0
    top = int(np.argmax(b))
    others = np.concatenate([np.delete(b, top), [kern]])
    re_top = int(np.argmax([x["lam"].real for x in bl]))
    return {"b_top": float(b[top]), "rho_perp": float(others.max() / b[top]),
            "kernel_b": kern, "kernel_beaten": bool(b[top] > kern),
            "sel_top_block": top, "sel_top_is_lam_top": top == 0,
            "re_top_block": re_top, "re_top_is_lam_top": re_top == 0}


def weights_row(left: np.ndarray, right: np.ndarray, gamma: float = GAMMA_W) -> Dict:
    bl = blocks(left, right)
    ev = eigs_sorted(bl)
    lad = ladder(left, right)
    t = u2_type(bl)
    return {"rank": int(sum(x["dim"] for x in bl)), "u2": t["type"], "u2_phi": t["phi"],
            "lam_top": [ev[0].real, ev[0].imag],
            "ratio_32": float(abs(ev[2]) / abs(ev[1])) if len(ev) >= 3 else None,
            **selection(bl, gamma, left.shape[0]),
            "ladder_D": [r["D"] for r in lad], "max_residual": max(r["residual"] for r in lad),
            "eigs": [[x.real, x.imag] for x in ev]}


def plane_angle(U: np.ndarray, P: np.ndarray) -> float:
    """The largest principal angle between span U and span P (same dimension)."""
    s = np.linalg.svd(U.T @ P, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))


# ---------------------------------------------------------------- M1 (torch, float64)

def n_bins(n_max: int = 2048) -> int:
    return bin_of(n_max - 1) + 1


def bin_of(delta: int) -> int:
    return delta if delta < 16 else 12 + int(math.floor(math.log2(delta)))


class Seq:
    """One prompt: the real rows in offset coordinates ``X[i, Δ] = X[i, i − Δ]``, Δ ≤ i."""

    def __init__(self, A: np.ndarray, device: str):
        import torch
        n = A.shape[0]
        if n > 2048:
            raise B2aError(f"n = {n} > 2048: no offset bin")
        self.n = n
        i = torch.arange(n, device=device)
        delta = i[None, :].expand(n, n)
        self.valid = delta <= i[:, None]
        self.idx = (i[:, None] - delta).clamp(min=0)
        b = torch.tensor([bin_of(k) for k in range(n)], device=device)
        self.OH = torch.nn.functional.one_hot(b, n_bins()).to(torch.float64)      # (n, NB)
        At = torch.as_tensor(A, dtype=torch.float64, device=device)
        self.A = self.off(At[None])[0]

    def off(self, X):
        """(B, n, n) in key coordinates → offset coordinates, invalid entries 0."""
        out = X.gather(-1, self.idx.expand(X.shape[0], -1, -1))
        return out * self.valid


def cos_feat(u, Us):
    """u (n, d) unit tokens; Us (B, d, D) or None for U = I; D = 0 gives zeros. (B, n, n)."""
    import torch
    if Us is None:
        return (u @ u.T)[None]
    if Us.shape[-1] == 0:
        return torch.zeros((Us.shape[0], u.shape[0], u.shape[0]), dtype=u.dtype, device=u.device)
    p = torch.einsum("nd,bdk->bnk", u, Us)
    p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-300)
    return p @ p.transpose(1, 2)


def _logp(theta, c, s: Seq):
    import torch
    phi = theta[:, 1:] @ s.OH.T                                         # (B, n) over Δ
    L = theta[:, :1, None] * c + phi[:, None, :]
    L = L.masked_fill(~s.valid, -torch.inf)
    lp = L - torch.logsumexp(L, dim=-1, keepdim=True)
    return lp.masked_fill(~s.valid, 0.0)


def _ce(theta, c, s: Seq):
    return -(s.A * _logp(theta, c, s)).sum((1, 2))


def _ghl(theta, c, s: Seq):
    import torch
    lp = _logp(theta, c, s)
    P = lp.exp() * s.valid
    R = P - s.A
    g = torch.cat([(R * c).sum((1, 2))[:, None], R.sum(1) @ s.OH], 1)
    Pc = P * c
    sc, scc = Pc.sum(-1), (Pc * c).sum(-1)                              # (B, n)
    Pb, Pcb = P @ s.OH, Pc @ s.OH                                       # (B, n, NB)
    nb = s.OH.shape[1]
    H = torch.zeros((theta.shape[0], nb + 1, nb + 1), dtype=theta.dtype, device=theta.device)
    H[:, 0, 0] = (scc - sc ** 2).sum(1)
    h0 = (Pcb - sc[..., None] * Pb).sum(1)
    H[:, 0, 1:], H[:, 1:, 0] = h0, h0
    H[:, 1:, 1:] = torch.diag_embed(Pb.sum(1)) - torch.einsum("bik,bil->bkl", Pb, Pb)
    return g, H, -(s.A * lp).sum((1, 2))


def fit(cs: Sequence, seqs: Sequence[Seq]):
    """Newton on the convex cross-entropy; θ = (β, φ_bins) per batch row. Returns θ, info."""
    import torch
    B, nb = cs[0].shape[0], seqs[0].OH.shape[1]
    dev = cs[0].device
    theta = torch.zeros((B, nb + 1), dtype=torch.float64, device=dev)
    eye = torch.eye(nb + 1, dtype=torch.float64, device=dev)
    rows = sum(s.n for s in seqs)
    for it in range(MAX_IT):
        g = H = ce = 0
        for c, s in zip(cs, seqs):
            gi, Hi, ci = _ghl(theta, c, s)
            g, H, ce = g + gi, H + Hi, ce + ci
        g = g + RIDGE * theta
        ce = ce + 0.5 * RIDGE * (theta ** 2).sum(1)
        delta = -torch.linalg.solve(H + RIDGE * eye, g[..., None])[..., 0]
        gd = (g * delta).sum(1)
        t = torch.ones(B, dtype=torch.float64, device=dev)
        for _ in range(40):
            th = theta + t[:, None] * delta
            cn = sum(_ce(th, c, s) for c, s in zip(cs, seqs)) + 0.5 * RIDGE * (th ** 2).sum(1)
            ok = cn <= ce + 1e-4 * t * gd + 1e-12 * ce.abs()
            if bool(ok.all()):
                break
            t = torch.where(ok, t, t / 2)
        theta = th
        if float((t[:, None] * delta).abs().max()) < 1e-10:
            break
    gn = float(g.abs().max()) / rows
    if gn > 1e-6:
        raise B2aError(f"M1 fit did not converge: max |grad| / rows = {gn:.2e} after {it + 1} it")
    return theta, {"iters": it + 1, "grad": gn}


def tv(theta, c, s: Seq):
    P = _logp(theta, c, s).exp() * s.valid
    return 0.5 * (P - s.A).abs().sum(-1).mean(-1)                       # (B,)


def tv_uniform(s: Seq) -> float:
    """TV of causal-uniform attention (keys j ≤ i) against the real rows, mean over rows."""
    U = s.valid.to(s.A.dtype)
    U = U / U.sum(-1, keepdim=True)
    return float(0.5 * (U - s.A).abs().sum(-1).mean())


def loo(us: Sequence, seqs: Sequence[Seq], Us) -> Dict:
    """Leave one prompt out: fit on the others, TV on the one left out. Us as in `cos_feat`."""
    import torch
    cs = [s.off(cos_feat(u, Us)) for u, s in zip(us, seqs)]
    tvs, betas, its = [], [], 0
    for k in range(len(seqs)):
        rest = [i for i in range(len(seqs)) if i != k]
        th, info = fit([cs[i] for i in rest], [seqs[i] for i in rest])
        tvs.append(tv(th, cs[k], seqs[k]))
        betas.append(th[:, 0])
        its = max(its, info["iters"])
    T, Bt = torch.stack(tvs), torch.stack(betas)
    return {"tv": T.mean(0).cpu().numpy(), "beta": Bt.median(0).values.cpu().numpy(), "iters": its}


def occupancy(us: Sequence[np.ndarray], U: np.ndarray) -> float:
    """Median over prompts of the median token share ``‖Uᵀu_i‖²`` (random: D / d)."""
    return float(np.median([np.median(np.sum((np.asarray(u) @ U) ** 2, 1)) for u in us]))


def haar(d: int, D: int, k: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    out = np.empty((k, d, D))
    for i in range(k):
        out[i] = np.linalg.qr(g.standard_normal((d, D)))[0]
    return out


def read_head(us: Sequence[np.ndarray], As: Sequence[np.ndarray], lad: List[Dict], tag: str,
              device: str, all_nulls: bool = False) -> Dict:
    """The rule on one head: TV(full), TV per rung of V's ladder, nulls where the TV clause
    passes, the verdict."""
    import torch
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)
    seqs = [Seq(A, device) for A in As]
    ut = [T(u) for u in us]
    d = us[0].shape[1]
    full = loo(ut, seqs, None)
    rungs = []
    for r in lad:
        o = loo(ut, seqs, T(r["U"][None]))
        rungs.append({"D": r["D"], "tv": float(o["tv"][0]), "beta": float(o["beta"][0]),
                      "occupancy": occupancy(us, r["U"])})
    rec = {"tv_full": float(full["tv"][0]), "beta_full": float(full["beta"][0]),
           "rungs": rungs, "nulls": {}}
    tv0 = next(r["tv"] for r in rungs if r["D"] == 0)
    tvu = float(np.mean([tv_uniform(s) for s in seqs]))
    rec["tv_uniform"] = tvu
    if rec["tv_full"] > TV_OUT:
        rec.update(verdict="outside", outside_why="TV(full) > TV_OUT", D_star=None)
        return rec
    if tv0 - rec["tv_full"] < TV_TOL and tvu - tv0 < TV_TOL and rec["tv_full"] > TV_TOL:
        rec.update(verdict="outside", outside_why="content M1 cannot express", D_star=None)
        return rec
    rec.update(verdict="none", D_star=None)
    for r in rungs:
        if r["tv"] > rec["tv_full"] + TV_TOL:
            continue
        if r["D"] == 0:
            if rec["D_star"] is None:
                rec.update(verdict="D*", D_star=0)
            if not all_nulls:
                break
            continue
        null_tv: List[float] = []
        null_occ: List[float] = []
        for ch in range(K_NULL // NULL_CHUNK):
            Us = haar(d, r["D"], NULL_CHUNK, seed_for("b2a-null", tag, r["D"], ch))
            null_tv += loo(ut, seqs, T(Us))["tv"].tolist()
            null_occ += [occupancy(us, U) for U in Us]
            if not all_nulls and min(null_tv) <= r["tv"]:
                break
        beat = int(sum(x <= r["tv"] for x in null_tv))
        rec["nulls"][str(r["D"])] = {"n": len(null_tv), "beat": beat, "min": min(null_tv),
                                     "median": float(np.median(null_tv)), "tv": null_tv,
                                     "occupancy": null_occ}
        if beat == 0 and len(null_tv) == K_NULL and rec["D_star"] is None:
            rec.update(verdict="D*", D_star=r["D"])
            if not all_nulls:
                break
    return rec


# ---------------------------------------------------------------- M2: the head's own QK

def rotary(n: int):
    """cos, sin (n, 16) as HF GPT-NeoX computes them (inv_freq and the angles in float32)."""
    import torch
    inv = 1.0 / (ROT_BASE ** (torch.arange(0, ROT_DIMS, 2, dtype=torch.int64).float() / ROT_DIMS))
    f = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([f, f], -1)
    return emb.cos().double().numpy(), emb.sin().double().numpy()


def _rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    r = x[:, :ROT_DIMS]
    half = ROT_DIMS // 2
    rot = np.concatenate([-r[:, half:], r[:, :half]], 1)
    return np.concatenate([r * cos + rot * sin, x[:, ROT_DIMS:]], 1)


def head_logits(y: np.ndarray, W: Dict) -> np.ndarray:
    """The head's causal attention logits on LN1 rows y (n, d); W = {Wq, bq, Wk, bk}."""
    n = y.shape[0]
    cos, sin = rotary(n)
    q = _rope(y @ W["Wq"].T + W["bq"], cos, sin)
    k = _rope(y @ W["Wk"].T + W["bk"], cos, sin)
    L = (q @ k.T) / math.sqrt(HEAD_DIM)
    L[np.triu_indices(n, 1)] = -np.inf
    return L


def softmax_rows(L: np.ndarray) -> np.ndarray:
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def m2_logits(y: np.ndarray, U: np.ndarray, W: Dict) -> np.ndarray:
    return head_logits(y @ U @ U.T, W)


def qk_space(W: Dict) -> np.ndarray:
    """Orthonormal basis of the row span of [W_Q; W_K] (rank ≤ 128)."""
    M = np.concatenate([W["Wq"], W["Wk"]], 0)
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[s > s.max() * RANK_RTOL].T


# ---------------------------------------------------------------- weights of a checkpoint

def head_weights(model, name: str) -> Dict:
    l, h = parse_head(name)
    blk = model.gpt_neox.layers[l]
    qkv = blk.attention.query_key_value
    W = qkv.weight.detach().double().numpy().reshape(N_HEADS, 3 * HEAD_DIM, -1)[h]
    b = qkv.bias.detach().double().numpy().reshape(N_HEADS, 3 * HEAD_DIM)[h]
    Wo = blk.attention.dense.weight.detach().double().numpy().reshape(-1, N_HEADS, HEAD_DIM)[:, h]
    g = blk.input_layernorm.weight.detach().double().numpy()
    d = g.shape[0]
    C = np.eye(d) - 1.0 / d
    return {"Wq": W[:HEAD_DIM], "Wk": W[HEAD_DIM:2 * HEAD_DIM], "Wv": W[2 * HEAD_DIM:],
            "bq": b[:HEAD_DIM], "bk": b[HEAD_DIM:2 * HEAD_DIM], "Wo": Wo, "g": g,
            "left": g[:, None] * (C @ Wo), "right": W[2 * HEAD_DIM:]}


def g5_real(model, names: Sequence[str]) -> Dict:
    """G5 on real weights: (ii) M2 on the QK read space equals the head's logit; (iii) the
    head's attention from these weights equals HF's on a seeded random-token pass."""
    import torch
    g = np.random.default_rng(seed_for("b2a-g5"))
    ids = torch.as_tensor(g.integers(0, model.config.vocab_size, (1, 64)))
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
    ii, iii, raw = 0.0, 0.0, 0.0
    eps32 = float(np.finfo(np.float32).eps)
    for name in names:
        l, h = parse_head(name)
        W = head_weights(model, name)
        ln = model.gpt_neox.layers[l].input_layernorm
        with torch.no_grad():
            y = ln(out.hidden_states[l][0]).double().numpy()
        L = head_logits(y, W)
        Lm = m2_logits(y, qk_space(W), W)
        fin = np.isfinite(L)
        ii = max(ii, float(np.abs(L[fin] - Lm[fin]).max()))
        dA = float(np.abs(softmax_rows(L) - out.attentions[l][0, h].double().numpy()).max())
        raw, iii = max(raw, dA), max(iii, dA / (eps32 * float(np.abs(L[fin]).max())))
    return {"ii_m2_full_rank": ii, "iii_hf_attention": iii, "iii_raw_max": raw,
            "pass": ii <= G5_TOL and iii <= G5_HF_TOL}


# ---------------------------------------------------------------- the gate's planted heads

def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _causal_softmax(L: np.ndarray) -> np.ndarray:
    L = L.copy()
    L[np.triu_indices(L.shape[0], 1)] = -np.inf
    return softmax_rows(L)


def paper_attention(u: np.ndarray, beta: float = G_BETA) -> np.ndarray:
    return _causal_softmax(beta * (u @ u.T))


def planted_V(kind: str, g) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """left (d, r), right (r, d) and the planted block's basis P (d, k)."""
    P, _ = np.linalg.qr(g.standard_normal((G_D, G_RANK)))
    S = np.diag(g.uniform(-G_REST, G_REST, G_RANK))
    if kind == "G1":
        k, S[:2, :2] = 2, -G_NU * np.eye(2)
    elif kind == "G1r":
        c, s = math.cos(G_PHI), math.sin(G_PHI)
        k, S[:2, :2] = 2, G_NU * np.array([[c, -s], [s, c]])
    elif kind == "G1d":
        k, S[:4, :4] = 4, -G_NU * np.eye(4)
    else:
        raise ValueError(kind)
    return P @ S, P.T, P[:, :k]


def block_tokens(P: np.ndarray, g) -> np.ndarray:
    """Uniform on the planted block's unit sphere, `G_OFF` of the norm² in a random direction
    off the block."""
    v = _unit(g.standard_normal((G_N, P.shape[1])))
    w = g.standard_normal((G_N, G_D))
    w = _unit(w - (w @ P) @ P.T)
    return math.sqrt(1 - G_OFF) * v @ P.T + math.sqrt(G_OFF) * w


def random_V(g) -> Tuple[np.ndarray, np.ndarray]:
    return g.standard_normal((G_D, G_RANK)) / math.sqrt(G_D), g.standard_normal((G_RANK, G_D))


def arm_data(arm: str, seed: int = 0, scale: Optional[float] = None) -> Dict:
    g = np.random.default_rng(seed_for("b2a-gate", arm, seed) if scale is None
                              else seed_for("b2a-gate", arm, seed, scale))
    P = None
    if arm in ("G1", "G1r", "G1d"):
        left, right, P = planted_V(arm, g)
        us = [block_tokens(P, g) for _ in range(G_PROMPTS)]
        As = [paper_attention(u) for u in us]
    else:
        left, right = random_V(g)
        if arm == "G3":
            S = np.linalg.qr(g.standard_normal((G_D, G3_SUB)))[0]
            us = [_unit(g.standard_normal((G_N, G3_SUB)) @ S.T) for _ in range(G_PROMPTS)]
        else:
            us = [_unit(g.standard_normal((G_N, G_D))) for _ in range(G_PROMPTS)]
        if arm in ("G2", "G2n"):
            phi = np.zeros(G_N)
            phi[1] = G2_PEAK
            i = np.arange(G_N)
            L = phi[np.clip(i[:, None] - i[None, :], 0, G_N - 1)]
            if arm == "G2":
                As = [_causal_softmax(L) for _ in us]
            else:
                G = g.standard_normal((G_D, G_D))
                As = [_causal_softmax(L + scale * (u @ G @ u.T)) for u in us]
        elif arm == "G3":
            As = [paper_attention(u) for u in us]
        elif arm in ("G4", "G4w"):
            G = g.standard_normal((G_D, G_D))
            sd = G4_SD if arm == "G4" else scale
            As = [_causal_softmax(sd * (u @ G @ u.T)) for u in us]
        else:
            raise ValueError(arm)
    return {"left": left, "right": right, "P": P, "us": us, "As": As}


def run_arm(arm: str, device: str, all_nulls: bool, seed: int = 0,
            scale: Optional[float] = None) -> Dict:
    t0 = time.monotonic()
    z = arm_data(arm, seed, scale)
    bl = blocks(z["left"], z["right"])
    lad = ladder(z["left"], z["right"])
    tag = f"{arm}-{seed}" if scale is None else f"{arm}-{seed}-{scale}"
    rec = read_head(z["us"], z["As"], lad, tag, device, all_nulls)
    rec["u2"] = u2_type(bl)["type"]
    if z["P"] is not None:
        k = z["P"].shape[1]
        rec["angle"] = plane_angle(next(r["U"] for r in lad if r["D"] == k), z["P"])
        rec["occupancy"] = float(np.median([np.median(np.sum((u @ z["P"]) ** 2, 1)) for u in z["us"]]))
    rec["seconds"] = time.monotonic() - t0
    return rec


def g5_planted(device: str) -> float:
    """(i) M1 at U = I with the generating θ equals the paper's model, through the offset code."""
    import torch
    z = arm_data("G1", 0)
    u, A = z["us"][0], z["As"][0]
    s = Seq(A, device)
    th = torch.zeros((1, n_bins() + 1), dtype=torch.float64, device=device)
    th[0, 0] = G_BETA
    ut = torch.as_tensor(u, dtype=torch.float64, device=device)
    P = (_logp(th, s.off(cos_feat(ut, None)), s).exp() * s.valid)[0]
    return float((P - s.A).abs().max())


def arm_pass(arm: str, rec) -> Tuple[bool, str]:
    if arm == "G1":
        ok = rec["D_star"] == 2 and rec["angle"] <= PLANE_TOL and rec["u2"] == "degenerate negative"
        return ok, f"D*={rec['D_star']}, angle {rec['angle']:.1e}, {rec['u2']}"
    if arm == "G1r":
        return rec["D_star"] == 2 and rec["u2"] == "rotation", f"D*={rec['D_star']}, {rec['u2']}"
    if arm == "G1d":
        return rec["D_star"] == 4, f"D*={rec['D_star']}"
    if arm == "G2":
        return rec["D_star"] == 0, f"D*={rec['D_star']}"
    if arm == "G2n":
        return (all(r["D_star"] == 0 for r in rec),
                "; ".join(f"scale {s}: D*={r['D_star']} ({r['verdict']}), TV(full) {r['tv_full']:.3f}, "
                          f"TV(uniform) {r['tv_uniform']:.3f}" for s, r in zip(G4W_SD, rec)))
    if arm == "G3":
        none = sum(r["verdict"] == "none" for r in rec)
        return none >= G3_PASS, f"none in {none} of {len(rec)}"
    if arm == "G4":
        return rec["verdict"] == "outside", f"{rec['verdict']}, TV(full) {rec['tv_full']:.3f}"
    if arm == "G4w":
        tv0 = [next(x["tv"] for x in r["rungs"] if x["D"] == 0) for r in rec]
        return (all(r["verdict"] == "outside" for r in rec),
                "; ".join(f"scale {s}: {r['verdict']} ({r.get('outside_why')}), TV(full) {r['tv_full']:.3f}, "
                          f"TV(0) {t:.3f}" for s, r, t in zip(G4W_SD, rec, tv0)))
    if arm == "G5":
        return rec["i_m1_identity"] <= G5_TOL, f"(i) {rec['i_m1_identity']:.1e}"
    raise ValueError(arm)


# ---------------------------------------------------------------- unit 2: real tokens

def ln1(X: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """LN1 rows y (n, d) and the scale σ (n,), float64 from the pass's float32 rows."""
    X = np.asarray(X, dtype=np.float64)
    C = X - X.mean(1, keepdims=True)
    sig = np.sqrt((C ** 2).mean(1) + eps)
    return C / sig[:, None] * w + b, sig


def renorm(A: np.ndarray) -> np.ndarray:
    """Position 0 out as key and query; the real rows renormalised over keys j ≥ 1."""
    R = A[1:, 1:]
    m = R.sum(1, keepdims=True)
    if not np.all(np.isfinite(m)) or m.min() <= 1e-300:
        raise B2aError(f"a row puts no mass off position 0 (min {m.min():.1e}): its TV would be NaN")
    return R / m


def mean_tv(P: np.ndarray, A: np.ndarray) -> float:
    return float(0.5 * np.abs(P - A).sum(1).mean())


def m1_attention(u: np.ndarray, U: Optional[np.ndarray], theta: np.ndarray) -> np.ndarray:
    """M1's Â on unit tokens u (n, d) at fitted θ = (β, φ_bins); U None is the full frame."""
    if U is None:
        c = u @ u.T
    elif U.shape[1] == 0:
        c = np.zeros((u.shape[0],) * 2)
    else:
        p = u @ U
        p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-300)
        c = p @ p.T
    n = u.shape[0]
    i = np.arange(n)
    bins = np.array([bin_of(k) for k in range(n)])
    L = theta[0] * c + theta[1:][bins[np.clip(i[:, None] - i[None, :], 0, n - 1)]]
    return _causal_softmax(L)


def fit_all(us: Sequence[np.ndarray], As: Sequence[np.ndarray], U: Optional[np.ndarray],
            device: str) -> np.ndarray:
    """θ fitted on every prompt (for the beside rows; the rule uses `loo`)."""
    import torch
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)
    seqs = [Seq(A, device) for A in As]
    Us = None if U is None else T(U[None])
    th, _ = fit([s.off(cos_feat(T(u), Us)) for u, s in zip(us, seqs)], seqs)
    return th[0].cpu().numpy()


def motion(u: np.ndarray, y: np.ndarray, Ar: np.ndarray, Ah: np.ndarray, U: Optional[np.ndarray],
           V: np.ndarray) -> np.ndarray:
    """Per token, the cosine of the tangent parts (off u_i) of the reduced update
    ``U W_D Σ_j Â_ij Uᵀu_j`` (W_D = Uᵀ V U; U None: V Σ_j Â_ij u_j) and the head's real output in
    the frame, ``V Σ_j A_ij y_j`` on the renormalised rows (biases out, as V)."""
    red = Ah @ u @ V.T if U is None else Ah @ u @ U @ (U.T @ V @ U).T @ U.T
    real = Ar @ y @ V.T
    tan = lambda m: m - np.sum(m * u, 1, keepdims=True) * u
    a, b = tan(red), tan(real)
    return np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)


def qk_ladder(W: Dict) -> List[Dict]:
    """The QK read space's ladder: top-D right singular vectors of [W_Q; W_K], D on `LADDER`
    capped at the rank, plus the full rank (M2 there is exact, G5 (ii))."""
    M = np.concatenate([W["Wq"], W["Wk"]], 0)
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = int((s > s.max() * RANK_RTOL).sum())
    return [{"D": D, "U": Vt[:D].T} for D in sorted({min(D, r) for D in LADDER} | {r})]


def a1a_beta(step: int, prompt: str, name: str) -> float:
    l, h = parse_head(name)
    f = A1A_RECORDS / f"step{step}_{prompt}.json"
    return float(json.loads(f.read_text())["beta"]["per_head"][l][h])


def check_gate_file(path: Path) -> Dict:
    g = json.loads(path.read_text())
    if g.get("code") != code_sha():
        raise SystemExit(f"refusing: gate {path} is from code {g.get('code')}, not {code_sha()}")
    bad = [a for a in ARMS if not g["arms"].get(a, {}).get("pass")]
    if bad:
        raise SystemExit(f"refusing: gate arms not passed: {bad}")
    return g


def cmd_real(a) -> int:
    import torch
    from core.holdout import refuse_held_out
    from p1e_energy_field.u2_attn import token_ids
    from trackA_force.a1a import V1, prompts_hash, run_dir
    from trackB_circuit.b0_gate import load
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (unit 2's pass is the GPU's, as A1a)")
    check_gate_file(Path(a.gate))
    step = a.step
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "real.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    if rec and rec.get("code") != code_sha():
        raise SystemExit(f"refusing: {p} is from code {rec.get('code')}, not {code_sha()}")
    rds = [run_dir(step, q) for q in V1]
    refuse_held_out(rds, context="b2a")
    rec.update(_env("cuda"), step=step, prompts=list(V1), prompts_hash=prompts_hash(),
               runs=[str(r) for r in rds], gate=str(a.gate), pass_="cuda:float32")
    rec.setdefault("heads", {})
    t0 = time.monotonic()
    model, tok = load(step, "cuda", torch.float32)
    lay = model.gpt_neox.layers
    eps = float(lay[0].input_layernorm.eps)
    need = sorted({parse_head(h)[0] for h in heads()})
    hs = {}
    for q, rd in zip(V1, rds):
        man = json.loads((rd / "manifest.json").read_text())
        if man.get("hf_revision") != f"step{step}":
            raise SystemExit(f"refusing: {rd} manifest revision {man.get('hf_revision')}")
        ids = torch.tensor([token_ids(tok, rd)], device="cuda")
        with torch.no_grad():
            o = model(ids, output_hidden_states=True)
        hs[q] = {l: o.hidden_states[l][0].float().cpu().numpy() for l in need}
    model.cpu()
    Ws = {h: head_weights(model, h) for h in heads()}
    ln = {l: (lay[l].input_layernorm.weight.detach().double().numpy(),
              lay[l].input_layernorm.bias.detach().double().numpy()) for l in need}
    del model
    torch.cuda.empty_cache()
    print(f"pass: {time.monotonic() - t0:.0f}s", flush=True)

    # G0 first, every head and prompt, before the rule reads anything
    eps32 = float(np.finfo(np.float32).eps)
    data: Dict[str, Dict] = {}
    g0 = {}
    for q, rd in zip(V1, rds):
        A_st = np.load(rd / "attentions.npz")["attentions"]
        for h in heads():
            l, hh = parse_head(h)
            y, sig = ln1(hs[q][l], *ln[l], eps)
            L = head_logits(y, Ws[h])
            A = softmax_rows(L)
            fin = np.isfinite(L)
            dA = float(np.abs(A - A_st[l, hh]).max())
            g0[f"{h}/{q}"] = {"raw": dA, "scaled": dA / (eps32 * float(np.abs(L[fin]).max()))}
            data.setdefault(h, {})[q] = {"y": y, "sig": sig, "A": A}
        del A_st
    worst = max(g0, key=lambda k: g0[k]["scaled"])
    rec["g0"] = {"tol": G0_TOL, "worst": worst, **g0[worst],
                 "pass": g0[worst]["scaled"] <= G0_TOL, "cells": g0}
    p.write_text(json.dumps(rec, indent=1))
    print(f"G0: worst {worst} raw {g0[worst]['raw']:.2e} scaled {g0[worst]['scaled']:.2f} "
          f"(tol {G0_TOL})", flush=True)
    if not rec["g0"]["pass"]:
        raise SystemExit("refusing: G0 failed; the rule is not read")
    if a.g0_only:
        return 0

    for h in (a.heads or heads()):
        if h in rec["heads"]:
            print(f"have {h}", flush=True)
            continue
        t1 = time.monotonic()
        W = Ws[h]
        V = W["left"] @ W["right"]
        bl = blocks(W["left"], W["right"])
        lad = ladder(W["left"], W["right"])
        D_ = data[h]
        ys = [D_[q]["y"] for q in V1]
        us = [_unit(y)[1:] for y in ys]
        Ars = [renorm(D_[q]["A"]) for q in V1]
        # γ: the head's own output (real attention, sink kept, biases out) mapped into the frame
        ratio = np.concatenate([
            np.linalg.norm((D_[q]["A"] @ D_[q]["y"] @ V.T) / D_[q]["sig"][:, None], axis=1)
            / np.linalg.norm(D_[q]["y"], axis=1) for q in V1])
        gr = float(np.median(ratio))
        gamma = gr / (1 + gr)
        r = read_head(us, Ars, lad, f"real-{step}-{h}", "cuda", a.all_nulls)
        # beside: M1 fitted on all 7 prompts per rung, its motion row; M2 on both ladders
        for rung in [{"D": None, "U": None}] + lad:
            if rung["D"] == 0:
                continue
            th = fit_all(us, Ars, rung["U"], "cuda")
            cs = np.concatenate([motion(u, y[1:], A, m1_attention(u, rung["U"], th), rung["U"], V)
                                 for u, y, A in zip(us, ys, Ars)])
            row = {"beta_all": float(th[0]), "motion": float(np.median(cs))}
            if rung["D"] is None:
                r["full_beside"] = row
            else:
                next(x for x in r["rungs"] if x["D"] == rung["D"]).update(row)
        m2 = {}
        for name, lad2 in (("v", lad), ("qk", qk_ladder(W))):
            m2[name] = [{"D": x["D"], "tv": float(np.mean([
                mean_tv(renorm(softmax_rows(m2_logits(y, x["U"], W))), A) for y, A in zip(ys, Ars)]))}
                for x in lad2]
        r.update({
            "gamma": gamma, "selection": selection(bl, gamma, V.shape[0]),
            "u2": u2_type(bl), "weights": weights_row(W["left"], W["right"], gamma),
            "sink": float(np.mean([D_[q]["A"][1:, 0].mean() for q in V1])),
            "sink_per_prompt": [float(D_[q]["A"][1:, 0].mean()) for q in V1],
            "a1a_beta": [a1a_beta(step, q, h) for q in V1],
            "m2": m2, "seconds": time.monotonic() - t1})
        r["weights"].pop("eigs", None)
        rec["heads"][h] = r
        p.write_text(json.dumps(rec, indent=1))
        print(f"{h}: {r['verdict']} D*={r['D_star']} TV(full) {r['tv_full']:.3f} "
              f"TV(0) {r['rungs'][0]['tv']:.3f} {r['seconds']:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------- unit 3: occupancy first

OCC_RUNGS = (2, 64)
OCC_RATIO = 2.0                           # placed (Blocked 53 (a), before any output): a step is read
                                          # when some head's U₂ share is above twice random, 4/1024


def rung_at(lad: List[Dict], D: int) -> Optional[Dict]:
    """The smallest rung of dimension ≥ D (D = 2 moves up to 3 when V's top block is a line and the
    next a pair)."""
    return next((x for x in lad if x["D"] >= D), None)


def occ_row(us: Sequence[np.ndarray], lad: List[Dict]) -> Dict:
    d = lad[0]["U"].shape[0]
    row: Dict[str, Optional[Dict]] = {}
    for D in OCC_RUNGS:
        x = rung_at(lad, D)
        if x is None:
            row[str(D)] = None
            continue
        s = occupancy(us, x["U"])
        row[str(D)] = {"D": x["D"], "share": s, "ratio": s / (x["D"] / d)}
    return row


def steps_to_read(rec: Dict) -> List[int]:
    """Blocked 53 (a): the steps where any head's U₂ share is above `OCC_RATIO` × random."""
    return sorted(int(s) for s, rows in rec["steps"].items()
                  if any(r["2"] is not None and r["2"]["ratio"] > OCC_RATIO for r in rows.values()))


def cmd_occupancy(a) -> int:
    import torch
    from core.holdout import refuse_held_out
    from p1e_energy_field.u2_attn import token_ids
    from trackA_force.a1a import V1, prompts_hash, run_dir
    from trackB_circuit.b0_gate import load
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (unit 3's pass is the GPU's, as unit 2)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "occupancy.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    if rec and rec.get("code") != code_sha():
        raise SystemExit(f"refusing: {p} is from code {rec.get('code')}, not {code_sha()}")
    rec.update(_env("cuda"), prompts=list(V1), prompts_hash=prompts_hash(), pass_="cuda:float32",
               heads=list(heads()), occ_ratio=OCC_RATIO)
    rec.setdefault("steps", {})
    need = sorted({parse_head(h)[0] for h in heads()})
    for step in a.steps:
        if str(step) in rec["steps"]:
            print(f"have {step}", flush=True)
            continue
        t0 = time.monotonic()
        rds = [run_dir(step, q) for q in V1]
        refuse_held_out(rds, context="b2a")
        model, tok = load(step, "cuda", torch.float32)
        lay = model.gpt_neox.layers
        eps = float(lay[0].input_layernorm.eps)
        hs = {}
        for q, rd in zip(V1, rds):
            man = json.loads((rd / "manifest.json").read_text())
            if man.get("hf_revision") != f"step{step}":
                raise SystemExit(f"refusing: {rd} manifest revision {man.get('hf_revision')}")
            ids = torch.tensor([token_ids(tok, rd)], device="cuda")
            with torch.no_grad():
                o = model(ids, output_hidden_states=True)
            hs[q] = {l: o.hidden_states[l][0].float().cpu().numpy() for l in need}
        model.cpu()
        rows = {}
        for h in heads():
            l, _ = parse_head(h)
            W = head_weights(model, h)
            ln = (lay[l].input_layernorm.weight.detach().double().numpy(),
                  lay[l].input_layernorm.bias.detach().double().numpy())
            us = [_unit(ln1(hs[q][l], *ln, eps)[0])[1:] for q in V1]
            rows[h] = occ_row(us, ladder(W["left"], W["right"]))
        del model
        torch.cuda.empty_cache()
        if step == 143000 and a.unit2:
            # the pass must reproduce unit 2's occupancy at its rungs (same code path, same device)
            u2 = json.loads(Path(a.unit2).read_text())["heads"]
            diff = max(abs(rows[h][k]["share"] - next(x["occupancy"] for x in u2[h]["rungs"]
                                                      if x["D"] == rows[h][k]["D"]))
                       for h in heads() for k in ("2", "64") if rows[h][k] is not None)
            rec["unit2_check"] = {"file": str(a.unit2), "max_abs_diff": diff}
            print(f"unit 2 check: max |Δ share| {diff:.2e}", flush=True)
            if diff > 1e-9:
                raise SystemExit("refusing: the pass does not reproduce unit 2's occupancy")
        rec["steps"][str(step)] = rows
        rec["read"] = steps_to_read(rec)
        p.write_text(json.dumps(rec, indent=1))
        top = max(rows, key=lambda h: rows[h]["2"]["ratio"] if rows[h]["2"] else -1)
        print(f"step {step}: max U2 ratio {rows[top]['2']['ratio']:.2f} ({top}) "
              f"{time.monotonic() - t0:.0f}s", flush=True)
    print(f"steps to read (U2 share > {OCC_RATIO} x random): {rec['read']}", flush=True)
    return 0


def report_occupancy(out: Path) -> None:
    x = json.loads((out / "occupancy.json").read_text())
    print(f"occupancy (code {x['code']}, prompts {x['prompts_hash']}): U2 share / random; "
          f"read where any > {x['occ_ratio']}: {x['read']}")
    steps = sorted(x["steps"], key=int)
    for k in OCC_RUNGS:
        print(f"U{k}  " + " ".join(f"{s:>6s}" for s in steps))
        for h in x["heads"]:
            cells = [x["steps"][s][h][str(k)] for s in steps]
            print(f"{h:8s} " + " ".join("     —" if c is None else f"{c['ratio']:6.2f}" for c in cells))


def constants() -> Dict:
    return {k: globals()[k] for k in (
        "LADDER", "TV_OUT", "TV_TOL", "K_NULL", "GAMMA_W", "DEGEN_TOL", "RANK_RTOL", "RIDGE",
        "G_D", "G_N", "G_PROMPTS", "G_BETA", "G_NU", "G_PHI", "G_REST", "G_OFF",
        "G_RANK", "G2_PEAK", "G3_SEEDS", "G3_SUB", "G3_PASS", "G4_SD", "G4W_SD", "PLANE_TOL", "G5_TOL",
        "G5_HF_TOL", "G0_TOL")}


def _env(device: str) -> Dict:
    import torch
    return {"code": code_sha(), "torch": torch.__version__, "device": device,
            "dtype": "float64", "constants": constants()}


def cmd_gate(a) -> int:
    import torch
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (pass --device cpu to run on the CPU)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "gate.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    if rec and rec.get("code") != code_sha():
        raise SystemExit(f"refusing: {p} is from code {rec.get('code')}, not {code_sha()}")
    rec.update(_env(a.device))
    rec.setdefault("arms", {})
    for arm in a.arms:
        t0 = time.monotonic()
        if arm == "G3":
            r = [run_arm("G3", a.device, a.all_nulls, s) for s in range(G3_SEEDS)]
        elif arm in ("G4w", "G2n"):
            r = [run_arm(arm, a.device, a.all_nulls, 0, s) for s in G4W_SD]
        elif arm == "G5":
            r = {"i_m1_identity": g5_planted(a.device)}
        else:
            r = run_arm(arm, a.device, a.all_nulls)
        ok, why = arm_pass(arm, r)
        rec["arms"][arm] = {"pass": ok, "why": why, "record": r}
        print(f"{arm}: {'PASS' if ok else 'FAIL'} ({why}) {time.monotonic() - t0:.0f}s", flush=True)
        p.write_text(json.dumps(rec, indent=1))
    return 0


def cmd_weights(a) -> int:
    import torch
    from trackB_circuit.b0_gate import load
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "weights.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    if rec and rec.get("code") != code_sha():
        raise SystemExit(f"refusing: {p} is from code {rec.get('code')}, not {code_sha()}")
    rec.update(_env("cpu"), heads=list(heads()), gamma=GAMMA_W)
    rec.setdefault("steps", {})
    for step in a.steps:
        t0 = time.monotonic()
        model, _ = load(step, "cpu", torch.float32)
        rows = {}
        for name in heads():
            W = head_weights(model, name)
            rows[name] = weights_row(W["left"], W["right"])
        rec["steps"][str(step)] = rows
        if step == 143000:
            rec["g5_real"] = g5_real(model, heads())
            print(f"G5 real: {rec['g5_real']}", flush=True)
        del model
        p.write_text(json.dumps(rec, indent=1))
        print(f"step {step}: {time.monotonic() - t0:.0f}s", flush=True)
    return 0


def report_real(out: Path) -> None:
    x = json.loads((out / "real.json").read_text())
    g0 = x["g0"]
    print(f"real (code {x['code']}, step {x['step']}, prompts {x['prompts_hash']}): "
          f"G0 {'PASS' if g0['pass'] else 'FAIL'} worst {g0['worst']} scaled {g0['scaled']:.2f}")
    print("head     verdict  D*   TVfull TV(0)  TV(D*) occ(D*) nullmin  beta_f a1a_b  gamma sink  "
          "mot_full mot2  M2qk(full) u2")
    for h, r in x["heads"].items():
        rg = {q["D"]: q for q in r["rungs"]}
        ds = r["D_star"]
        rs = rg.get(ds, {}) if ds is not None else {}
        nl = r["nulls"].get(str(ds), {}) if ds else {}
        print(f"{h:8s} {r['verdict']:8s} {str(ds):4s} {r['tv_full']:.3f}  {rg[0]['tv']:.3f}  "
              f"{rs.get('tv', float('nan')):.3f}  {rs.get('occupancy', float('nan')):.3f}   "
              f"{nl.get('min', float('nan')):.3f}   {r['beta_full']:6.2f} {np.median(r['a1a_beta']):6.2f} "
              f"{r['gamma']:.3f} {r['sink']:.2f}  {r['full_beside']['motion']:+.2f}   "
              f"{rg.get(2, {}).get('motion', float('nan')):+.2f}  {r['m2']['qk'][-1]['tv']:.3f}     "
              f"{r['u2']['type']}")


def cmd_report(a) -> int:
    out = Path(a.out)
    if (out / "occupancy.json").exists():
        report_occupancy(out)
    if (out / "real.json").exists():
        report_real(out)
    if not (out / "gate.json").exists():
        return 0
    g = json.loads((out / "gate.json").read_text())
    print(f"gate (code {g['code']}, {g['device']}):")
    for arm in ARMS:
        r = g["arms"].get(arm)
        print(f"  {arm:4s} {'—' if r is None else ('PASS' if r['pass'] else 'FAIL')}  {'' if r is None else r['why']}")
    w = out / "weights.json"
    if w.exists():
        w = json.loads(w.read_text())
        print(f"weights (code {w['code']}): G5 real {w.get('g5_real')}")
        print("head      step   rank u2                   |λ1|   ratio32 b_top  rho⊥  kern<  sel=λ  re=λ")
        for s in w["steps"]:
            for h, r in w["steps"][s].items():
                print(f"{h:8s} {s:>6s} {r['rank']:4d} {r['u2']:20s} {math.hypot(*r['lam_top']):6.2f} "
                      f"{r['ratio_32'] or float('nan'):6.2f} {r['b_top']:6.3f} {r['rho_perp']:5.2f} "
                      f"{str(r['kernel_beaten'])[0]:>5s} {str(r['sel_top_is_lam_top'])[0]:>5s} "
                      f"{str(r['re_top_is_lam_top'])[0]:>5s}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="b2a")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--out", required=True)
    g.add_argument("--device", default="cuda")
    g.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    g.add_argument("--all-nulls", action="store_true")
    w = sub.add_parser("weights")
    w.add_argument("--out", required=True)
    w.add_argument("--steps", nargs="+", type=int, default=list(STEPS))
    x = sub.add_parser("real")
    x.add_argument("--out", required=True)
    x.add_argument("--gate", required=True, help="gate.json from this code, every arm passed")
    x.add_argument("--step", type=int, default=143000)
    x.add_argument("--all-nulls", action="store_true")
    x.add_argument("--g0-only", action="store_true")
    x.add_argument("--heads", nargs="+", choices=heads(), help="a slice (default: all 14)")
    o = sub.add_parser("occupancy")
    o.add_argument("--out", required=True)
    o.add_argument("--steps", nargs="+", type=int, default=list(STEPS))
    o.add_argument("--unit2", help="unit 2's real.json; at 143000 the pass must reproduce its occupancy")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"gate": cmd_gate, "weights": cmd_weights, "real": cmd_real, "occupancy": cmd_occupancy,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
