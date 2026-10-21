"""S1, the shared-subspace atlas (`trackS_subspace/design-S1.md`; `docs/ROADMAP.md` §7a). Weights
only: no token is read.

**Subspaces per head** (raw residual frame, column convention ``x ↦ V x``, ``logit = xᵀ M y``;
V = W_O W_V, M_0 = W_Qᵀ W_K / √64 as B2's ``m_factors``). Each family gives a head an orthonormal
basis U_h (d × k_h):

- ``ov_write`` / ``ov_read``: V's top-r left / right singular vectors;
- ``qk_query`` / ``qk_key``: M_0's top-r left / right singular vectors (offset 0; rotary leaves
  M_0 as stored there);
- ``ov_neg`` / ``ov_pos`` / ``ov_cplx``: V's invariant subspaces among its top-r eigenvalues by
  modulus (a complex pair counts 2), split by type: real < 0, real > 0, complex (each pair's
  real and imaginary parts). LAPACK returns an exactly zero imaginary part for a real eigenvalue
  of a real matrix, so "real" needs no tolerance.

Every product is A Bᵀ with A, B of size d × 64, so its SVD and its nonzero eigenpairs come from
64 × 64 cores (B2's "exact low-rank spectra").

**The statistic.** For a pool of heads (one layer's 16, or all 384), the mean projector
P̄ = (1/n) Σ_h U_h U_hᵀ; its eigenvalues λ₁ ≥ λ₂ ≥ … lie in [0, 1], and λ is the share of the
pool's heads that hold the direction. Its nonzero spectrum is that of the Gram (1/n) BᵀB, B the
heads' bases side by side, whichever is smaller.

**The null.** A head enters P̄ only through its subspace, so the rotation null (each head's
subspace turned by its own Haar orthogonal map), a norm-matched Gaussian draw and Pythia's i.i.d.
normal init all give the same law: independent Haar subspaces of the heads' own dimensions
(`tools/math_checks/s1_projector_null.py`). One null per pool shape, drawn on the GPU.

Tier 1: exploratory, unregistered.
    python -m trackS_subspace.s1 explore --steps 0 8000 --out <dir>
    python -m trackS_subspace.s1 preflight --out <dir>
    python -m trackS_subspace.s1 r1qk --out <dir> --gate <regate dir>
    python -m trackS_subspace.s1 r3 --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackB_circuit.b2a import D_MODEL, HEAD_DIM, N_HEADS, N_LAYERS, REPO, seed_for

R_SUB = 16                                 # placed (design): heads' top-r directions
R_BESIDE = (8, 32)
FAMILIES = ("ov_write", "ov_read", "qk_query", "qk_key", "ov_neg", "ov_pos", "ov_cplx")
SINGULAR = FAMILIES[:4]


class S1Error(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackS_subspace/s1.py").read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- one head's subspaces

def _core(A: np.ndarray, B: np.ndarray):
    """A Bᵀ = Qa C Qbᵀ with C = Ra Rbᵀ (64 × 64)."""
    Qa, Ra = np.linalg.qr(A)
    Qb, Rb = np.linalg.qr(B)
    return Qa, Ra @ Rb.T, Qb


def singular_bases(A: np.ndarray, B: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top-r left and right singular vectors of A Bᵀ."""
    Qa, C, Qb = _core(A, B)
    U, _, Vt = np.linalg.svd(C)
    return Qa @ U[:, :r], Qb @ Vt[:r].T


def invariant_bases(A: np.ndarray, B: np.ndarray, r: int) -> Dict[str, np.ndarray]:
    """V = A Bᵀ's invariant subspaces among its top-r eigenvalues by modulus, by type. The nonzero
    eigenpairs of A Bᵀ are (λ, A v) for (λ, v) of Bᵀ A."""
    lam, v = np.linalg.eig(B.T @ A)
    order = np.argsort(-np.abs(lam), kind="stable")
    lam, v = lam[order], v[:, order]
    cols = {"ov_neg": [], "ov_pos": [], "ov_cplx": []}
    taken = 0
    for i in range(len(lam)):
        if taken >= r:
            break
        if lam[i].imag == 0:
            cols["ov_neg" if lam[i].real < 0 else "ov_pos"].append(A @ v[:, i].real)
            taken += 1
        elif lam[i].imag > 0:              # one of each conjugate pair
            w = A @ v[:, i]
            cols["ov_cplx"] += [w.real, w.imag]
            taken += 2
    out = {}
    for k, c in cols.items():
        out[k] = np.linalg.qr(np.stack(c, 1))[0] if c else np.zeros((A.shape[0], 0))
    return out


def head_bases(W: Dict, r: int = R_SUB) -> Dict[str, np.ndarray]:
    Vw, Vr = singular_bases(W["Wo"], W["Wv"].T, r)
    Qq, Qk = singular_bases(W["Wq"].T / np.sqrt(HEAD_DIM), W["Wk"].T, r)
    out = {"ov_write": Vw, "ov_read": Vr, "qk_query": Qq, "qk_key": Qk}
    out.update(invariant_bases(W["Wo"], W["Wv"].T, r))
    return out


# ---------------------------------------------------------------- the mean projector

def pools() -> Dict[str, List[int]]:
    out = {f"L{l}": list(range(l * N_HEADS, (l + 1) * N_HEADS)) for l in range(N_LAYERS)}
    out["all"] = list(range(N_LAYERS * N_HEADS))
    return out


def mean_projector_eigs(bases: Sequence[np.ndarray]) -> np.ndarray:
    """Eigenvalues of (1/n) Σ U Uᵀ, descending, length min(d, Σ k) (empty if every k is 0)."""
    kept = [b for b in bases if b.shape[1]]
    if not kept:
        return np.zeros(0)
    B = np.concatenate(kept, 1)
    n = len(bases)
    G = B.T @ B if B.shape[1] <= B.shape[0] else B @ B.T
    return np.sort(np.linalg.eigvalsh(G))[::-1] / n


def mean_projector(bases: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors (d × m) of P̄, descending."""
    B = np.concatenate([b for b in bases if b.shape[1]], 1)
    lam, U = np.linalg.eigh(B @ B.T / len(bases))
    return lam[::-1], U[:, ::-1]


# ---------------------------------------------------------------- the Haar null (GPU)

def haar_eigs(dims: Sequence[int], n_draw: int, seed: int, top: int = 32, d: int = D_MODEL,
              device: str = "cuda") -> np.ndarray:
    """(n_draw, top) leading eigenvalues of P̄ for independent Haar subspaces of the given dims
    (zeros past the nonzero spectrum). Heads of equal dimension are drawn in one batched QR."""
    import torch
    g = torch.Generator(device=device).manual_seed(seed % (2 ** 63))
    n, K = len(dims), int(sum(dims))
    groups = {k: sum(1 for x in dims if x == k) for k in sorted(set(dims)) if k}
    out = np.zeros((n_draw, top))
    batch = max(1, int(3e8 // (d * max(K, 1) * 8)))
    for s in range(0, n_draw, batch):
        b = min(batch, n_draw - s)
        Bs = []
        for k, c in groups.items():
            Z = torch.randn(b * c, d, k, generator=g, device=device, dtype=torch.float64)
            Q = torch.linalg.qr(Z)[0].reshape(b, c, d, k).permute(0, 2, 1, 3).reshape(b, d, c * k)
            Bs.append(Q)
        B = torch.cat(Bs, 2)
        G = B.transpose(1, 2) @ B if K <= d else B @ B.transpose(1, 2)
        ev = torch.linalg.eigvalsh(G).flip(-1)[:, :top] / n
        out[s:s + b, :ev.shape[1]] = ev.cpu().numpy()
    return out


def haar_bases(dims: Sequence[int], rng: np.random.Generator, d: int = D_MODEL) -> List[np.ndarray]:
    """Independent Haar bases (numpy; the planted control's carrier and the tests' null)."""
    return [np.linalg.qr(rng.standard_normal((d, k)))[0] if k else np.zeros((d, 0)) for k in dims]


def haar_orthogonal(b: int, d: int, g, device: str = "cuda"):
    """b Haar matrices on O(d) (torch, float64): QR of a Gaussian with R's diagonal signs fixed."""
    import torch
    Q, R = torch.linalg.qr(torch.randn(b, d, d, generator=g, device=device, dtype=torch.float64))
    return Q * torch.sign(torch.diagonal(R, dim1=1, dim2=2))[:, None, :]


def rotate_heads(hb: Sequence[Dict[str, np.ndarray]], seed: int, families: Sequence[str] = FAMILIES,
                 device: str = "cuda", chunk: int = 16) -> List[Dict[str, np.ndarray]]:
    """The rotation null: every family of head h turned by the same Haar map Q_h (the head's
    internal geometry kept), heads independent. By fact 4 its law is Haar of the same dims."""
    import torch
    g = torch.Generator(device=device).manual_seed(seed % (2 ** 63))
    out = []
    for s in range(0, len(hb), chunk):
        Q = haar_orthogonal(min(chunk, len(hb) - s), D_MODEL, g, device)
        for j, h in enumerate(hb[s:s + chunk]):
            out.append({f: (Q[j] @ torch.from_numpy(h[f]).to(device)).cpu().numpy() if h[f].shape[1]
                        else h[f] for f in families})
    return out


def gaussian_heads(n: int, norms: Dict[str, float], rng: np.random.Generator) -> List[Dict]:
    """Norm-matched random draw: i.i.d. normal W_Q, W_K, W_V (64 × d), W_O (d × 64), each scaled
    to the given Frobenius norm (the subspaces do not depend on the scale; kept for the record)."""
    out = []
    for _ in range(n):
        W = {"Wq": rng.standard_normal((HEAD_DIM, D_MODEL)), "Wk": rng.standard_normal((HEAD_DIM, D_MODEL)),
             "Wv": rng.standard_normal((HEAD_DIM, D_MODEL)), "Wo": rng.standard_normal((D_MODEL, HEAD_DIM))}
        out.append({k: v * norms[k] / np.linalg.norm(v) for k, v in W.items()})
    return out


def plant(bases: Sequence[np.ndarray], u: np.ndarray, heads: Sequence[int]) -> List[np.ndarray]:
    """Put the unit vector u into each listed head's subspace: its last basis column is replaced
    by u and the basis re-orthonormalised (dimension kept, u exactly inside)."""
    out = list(bases)
    for h in heads:
        b = out[h]
        out[h] = np.linalg.qr(np.concatenate([u[:, None], b[:, :-1]], 1))[0]
    return out


def count_above(ev: np.ndarray, threshold: float) -> int:
    """Eigenvalues above the null's threshold (the null's largest λ₁, so the count's false rate
    is the threshold's for every eigenvalue at once)."""
    return int(np.sum(ev > threshold))


def reading(bases: Sequence[np.ndarray], threshold: float) -> Dict:
    """R1's cell: N, the mass m (counted λ over the mean dimension), λ₁ and the top 200."""
    ev = mean_projector_eigs(bases)
    n = count_above(ev, threshold)
    mean_k = float(np.mean([b.shape[1] for b in bases]))
    return {"N": n, "m": float(ev[:n].sum() / mean_k) if mean_k else 0.0,
            "l1": float(ev[0]) if ev.size else 0.0, "eig": ev[:200].tolist(), "threshold": threshold}


def plant_many(bases: Sequence[np.ndarray], n_dir: int, k: int, rng: np.random.Generator,
               min_dim: int = 4) -> Optional[List[np.ndarray]]:
    """G3's plant: n_dir random unit directions, each put into k heads drawn among those with at
    least `min_dim` dimensions (so a head keeps its dimension and an earlier plant survives).
    None when fewer than k heads qualify (`ov_pos` at 8000: G3's first run crashed there)."""
    out = list(bases)
    eligible = [i for i, b in enumerate(bases) if b.shape[1] >= min_dim]
    if len(eligible) < k:
        return None
    for _ in range(n_dir):
        u = rng.standard_normal(bases[0].shape[0])
        out = plant(out, u / np.linalg.norm(u), rng.choice(eligible, k, replace=False))
    return out


# ---------------------------------------------------------------- a checkpoint

def step_bases(step: int, r: int = R_SUB) -> List[Dict[str, np.ndarray]]:
    import torch
    from trackB_circuit.b0_gate import load
    from trackB_circuit.b2 import layer_weights
    model, _ = load(step, "cpu", torch.float32)
    out = [head_bases(W, r) for l in range(N_LAYERS) for W in layer_weights(model, l)]
    del model
    return out


def explore(steps: Sequence[int], out: Path, n_draw: int = 2000) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rec = {"code_sha": code_sha(), "r": R_SUB, "steps": {}, "null": {}}
    P = pools()
    for step in steps:
        t = time.time()
        hb = step_bases(step)
        cell = {}
        for fam in FAMILIES:
            for name, idx in P.items():
                bs = [hb[i][fam] for i in idx]
                ev = mean_projector_eigs(bs)
                cell[f"{fam}|{name}"] = {"dims": [b.shape[1] for b in bs], "eig": ev[:32].tolist()}
        rec["steps"][str(step)] = cell
        print(f"step {step}: {time.time() - t:.0f}s", flush=True)
    for name, n in (("layer", N_HEADS), ("all", N_LAYERS * N_HEADS)):
        rec["null"][name] = haar_eigs([R_SUB] * n, n_draw, seed_for("s1-null", name, R_SUB)).tolist()
    (out / "explore.json").write_text(json.dumps(rec))


# ---------------------------------------------------------------- the design preflight

SCORED_STEPS = (0, 128, 256, 512, 1000, 2000, 4000, 143000)
PREFLIGHT_STEP = 8000                      # off the scored grid (`tools/design_preflight.py`)
PREFLIGHT_DOSES = {"haar": (0, 3, 4, 6), "shuffle": (0, 4, 8, 12)}   # heads of 16 holding u
N_NULL, N_PLANT = 2000, 200


def layer_l1(hb: Sequence[Dict], fam: str) -> List[float]:
    return [float(mean_projector_eigs([hb[i][fam] for i in range(l * N_HEADS, (l + 1) * N_HEADS)])[0])
            for l in range(N_LAYERS)]


def preflight_values(hb: Sequence[Dict], fam: str, null: str, haar_l1: np.ndarray) -> Dict:
    """`tools/design_preflight.py`'s input for λ₁ of a layer pool, against the Haar null or the
    step's layer-shuffle null (random 16-head sets of the same step's 384), planted control a
    random unit vector put into k of the pool's 16 heads, carried by a null pool."""
    rng = np.random.default_rng(seed_for("s1-preflight", fam, null))
    if null == "haar":
        null_vals = haar_l1
        carrier = lambda: haar_bases([R_SUB] * N_HEADS, rng)                       # noqa: E731
    else:
        carrier = lambda: [hb[i][fam] for i in rng.choice(len(hb), N_HEADS, replace=False)]  # noqa: E731
        null_vals = np.array([mean_projector_eigs(carrier())[0] for _ in range(N_NULL)])
    planted = {}
    for k in PREFLIGHT_DOSES[null]:
        vals = []
        for _ in range(N_PLANT):
            u = rng.standard_normal(D_MODEL)
            vals.append(float(mean_projector_eigs(plant(carrier(), u / np.linalg.norm(u), range(k)))[0]))
        planted[str(k)] = vals
    return {"statistic": f"S1: λ₁ of a layer pool's mean top-{R_SUB} projector (16 heads), family "
                         f"{fam}, raw frame; null {null}; planted dose = heads of 16 holding u",
            "direction": "greater", "nulls": {null: [float(x) for x in null_vals]}, "moved_marks": [],
            "trained": layer_l1(hb, fam), "trained_step": str(PREFLIGHT_STEP),
            "scored_steps": [str(s) for s in SCORED_STEPS], "planted": planted, "bounds": [0, 1],
            "n_null_draws": N_NULL}


def preflight(out: Path) -> None:
    from tools.design_preflight import main as run_preflight
    out.mkdir(parents=True, exist_ok=True)
    hb = step_bases(PREFLIGHT_STEP)
    hb0 = step_bases(0)
    haar_l1 = haar_eigs([R_SUB] * N_HEADS, N_NULL, seed_for("s1-null", "layer", R_SUB), top=1)[:, 0]
    summary = {"code_sha": code_sha(), "step": PREFLIGHT_STEP, "reports": {},
               "step0_layer_l1": {f: layer_l1(hb0, f) for f in SINGULAR},
               "haar_l1_band": np.quantile(haar_l1, [0.025, 0.975, 1.0]).tolist()}
    for fam in SINGULAR:
        for null in ("haar", "shuffle"):
            vals = out / f"values_{fam}_{null}.json"
            vals.write_text(json.dumps(preflight_values(hb, fam, null, haar_l1)))
            rep = out / f"report_{fam}_{null}.json"
            code = run_preflight([str(vals), "--out", str(rep)])
            summary["reports"][f"{fam}|{null}"] = {
                "exit": code, "sha256": hashlib.sha256(rep.read_bytes()).hexdigest(),
                "refusals": [r["code"] for r in json.loads(rep.read_text())["refusals"]]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- unit 1: the gate

G0_HEADS = (0, 82, 197, 383)               # L0H0, L5H2, L12H5, L23H15 at the preflight step
G0_TOL = {"singular": 1e-8, "invariant": 1e-6}   # placed (design G0): sin of the largest principal angle
G1_DRAWS = {"layer": 500, "all": 200}
G1_ROT_DRAWS = {"layer": 200, "all": 100}
KS_P = 1e-3
G2_MAX_CELLS = 1
G3_DOSES, G3_MORE = (12, 16, 24, 32), (48, 64)
G3_DIRS, G3_DRAWS, G3_RATE = 4, 50, 0.8
G3L_DOSES = {"ov": (4, 8, 12), "qk": (8, 12, 16)}
G3L_DRAWS = 200
POOL_SIZE = {"layer": N_HEADS, "all": N_LAYERS * N_HEADS}


def sin_max(U: np.ndarray, W: np.ndarray) -> float:
    """sin of the largest principal angle between two subspaces of equal dimension, as
    ‖(I − U Uᵀ) W‖₂ (exact to eps; √(1 − σ²) floors at √eps ≈ 1.5e-8, which G0's first run
    read as 5e-8)."""
    if U.shape[1] != W.shape[1]:
        return float("inf")
    if not U.shape[1]:
        return 0.0
    return float(np.linalg.norm(W - U @ (U.T @ W), 2))


def dense_bases(W: Dict, r: int = R_SUB) -> Dict[str, np.ndarray]:
    """G0's reference: the same families from dense d × d SVD and eig (real if |Im λ| ≤ 1e-9 |λ|)."""
    out = {}
    for (fl, fr), M in ((("ov_write", "ov_read"), W["Wo"] @ W["Wv"]),
                        (("qk_query", "qk_key"), W["Wq"].T @ W["Wk"] / np.sqrt(HEAD_DIM))):
        U, _, Vt = np.linalg.svd(M)
        out[fl], out[fr] = U[:, :r], Vt[:r].T
    lam, v = np.linalg.eig(W["Wo"] @ W["Wv"])
    o = np.argsort(-np.abs(lam), kind="stable")
    cols, taken = {"ov_neg": [], "ov_pos": [], "ov_cplx": []}, 0
    for i in o:
        if taken >= r:
            break
        if abs(lam[i].imag) <= 1e-9 * abs(lam[i]):
            cols["ov_neg" if lam[i].real < 0 else "ov_pos"].append(v[:, i].real)
            taken += 1
        elif lam[i].imag > 0:
            cols["ov_cplx"] += [v[:, i].real, v[:, i].imag]
            taken += 2
    for k, c in cols.items():
        out[k] = np.linalg.qr(np.stack(c, 1))[0] if c else np.zeros((D_MODEL, 0))
    return out


def ks(a, b) -> float:
    from scipy.stats import ks_2samp
    return float(ks_2samp(np.asarray(a), np.asarray(b)).pvalue)


def l1_of(bases: Sequence[np.ndarray]) -> float:
    ev = mean_projector_eigs(bases)
    return float(ev[0]) if ev.size else 0.0


def step_weights(step: int) -> List[Dict]:
    import torch
    from trackB_circuit.b0_gate import load
    from trackB_circuit.b2 import layer_weights
    model, _ = load(step, "cpu", torch.float32)
    out = [W for l in range(N_LAYERS) for W in layer_weights(model, l)]
    del model
    return out


def haar_max(dims: Sequence[int], tag) -> Tuple[float, np.ndarray]:
    """The cell's threshold: the largest λ₁ over N_NULL Haar draws of these dims (and the draws)."""
    x = haar_eigs(list(dims), N_NULL, seed_for("s1-null", *tag), top=1)[:, 0]
    return float(x.max()), x


def _save(out: Path, name: str, rec: Dict) -> None:
    """Every arm's record names the code that wrote it (`/challenge-pr` on #243, finding 4)."""
    rec = {"code_sha": code_sha(), **rec}
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"{name}: {'PASS' if rec.get('pass') else 'recorded' if rec.get('pass') is None else 'FAIL'}",
          flush=True)


def g0(Ws8000: Sequence[Dict], hb8000: Sequence[Dict], null_layer: np.ndarray, null_all: np.ndarray) -> Dict:
    rng = np.random.default_rng(seed_for("s1-g0"))
    worst = {f: 0.0 for f in FAMILIES}
    for W in gaussian_heads(20, {"Wq": 1, "Wk": 1, "Wv": 1, "Wo": 1}, rng) + [Ws8000[i] for i in G0_HEADS]:
        lo, de = head_bases(W), dense_bases(W)
        for f in FAMILIES:
            worst[f] = max(worst[f], sin_max(lo[f], de[f]))
    pool = [hb8000[i]["ov_write"] for i in range(5 * N_HEADS, 6 * N_HEADS)]
    B = np.concatenate(pool, 1)
    gram_err = float(np.abs(mean_projector_eigs(pool)
                            - np.sort(np.linalg.eigvalsh(B @ B.T / N_HEADS))[::-1][:B.shape[1]]).max())
    np_layer = [l1_of(haar_bases([R_SUB] * N_HEADS, rng)) for _ in range(G1_DRAWS["layer"])]
    np_all = [l1_of(haar_bases([R_SUB] * POOL_SIZE["all"], rng)) for _ in range(G1_DRAWS["all"])]
    rec = {"sin_max": worst, "gram_err": gram_err,
           "ks_gpu_numpy": {"layer": ks(np_layer, null_layer), "all": ks(np_all, null_all)}}
    rec["pass"] = bool(all(worst[f] < G0_TOL["singular" if f in SINGULAR else "invariant"] for f in FAMILIES)
                       and gram_err < 1e-10 and min(rec["ks_gpu_numpy"].values()) > KS_P)
    return rec


def g1(Ws8000: Sequence[Dict], hb8000: Sequence[Dict], nulls: Dict[str, np.ndarray]) -> Dict:
    """(i) Gaussian-weight heads and (ii) 8000's heads rotated, against Haar: singular families
    against the GPU draws, invariant families against paired numpy Haar draws of the same dims."""
    rng = np.random.default_rng(seed_for("s1-g1"))
    norms = {k: float(np.mean([np.linalg.norm(W[k]) for W in Ws8000])) for k in ("Wq", "Wk", "Wv", "Wo")}
    rec = {"norms": norms, "ks": {}}
    for arm in ("gaussian", "rotation"):
        for shape, n in POOL_SIZE.items():
            draws = (G1_DRAWS if arm == "gaussian" else G1_ROT_DRAWS)[shape]
            obs = {f: [] for f in FAMILIES}
            paired = {f: [] for f in FAMILIES if f not in SINGULAR}
            for j in range(draws):
                if arm == "gaussian":
                    pool = [head_bases(W) for W in gaussian_heads(n, norms, rng)]
                else:
                    idx = range(n) if shape == "all" else range((j % N_LAYERS) * N_HEADS, (j % N_LAYERS + 1) * N_HEADS)
                    pool = rotate_heads([hb8000[i] for i in idx], seed_for("s1-g1-rot", shape, j))
                for f in FAMILIES:
                    obs[f].append(l1_of([h[f] for h in pool]))
                    if f in paired:
                        paired[f].append(l1_of(haar_bases([h[f].shape[1] for h in pool], rng)))
            for f in FAMILIES:
                ref = nulls[shape] if f in SINGULAR else paired[f]
                rec["ks"][f"{arm}|{shape}|{f}"] = ks(obs[f], ref)
            print(f"  G1 {arm} {shape} done", flush=True)
    rec["min_p"] = min(rec["ks"].values())
    rec["pass"] = bool(rec["min_p"] > KS_P)
    return rec


def g3(hb8000: Sequence[Dict], thresholds: Dict[str, float]) -> Dict:
    rng = np.random.default_rng(seed_for("s1-g3"))
    rec = {"families": {}}
    for f in FAMILIES:
        real = [h[f] for h in hb8000]
        dims = [b.shape[1] for b in real]
        T = thresholds[f]
        n_real = count_above(mean_projector_eigs(real), T)
        rows, doses = {}, list(G3_DOSES)
        while doses:
            k = doses.pop(0)
            d_haar, d_real = [], []
            if plant_many(real, G3_DIRS, k, rng) is None:
                rows[k] = {"rate_haar": 0.0, "rate_real": 0.0, "not_plantable": True,
                           "eligible": int(sum(b.shape[1] >= 4 for b in real))}
                if k == G3_DOSES[-1] and not doses:
                    doses = list(G3_MORE)
                continue
            for _ in range(G3_DRAWS):
                carrier = haar_bases(dims, rng)
                n0 = count_above(mean_projector_eigs(carrier), T)
                d_haar.append(count_above(mean_projector_eigs(plant_many(carrier, G3_DIRS, k, rng)), T) - n0)
                d_real.append(count_above(mean_projector_eigs(plant_many(real, G3_DIRS, k, rng)), T) - n_real)
            rows[k] = {"rate_haar": float(np.mean(np.array(d_haar) >= G3_DIRS)),
                       "rate_real": float(np.mean(np.array(d_real) >= G3_DIRS)),
                       "dN_real_median": float(np.median(d_real)), "dN_haar_median": float(np.median(d_haar)),
                       "dN_real": [int(x) for x in d_real], "dN_haar": [int(x) for x in d_haar]}
            if k == G3_DOSES[-1] and min(rows[k]["rate_haar"], rows[k]["rate_real"]) < G3_RATE and not doses:
                doses = list(G3_MORE)                  # pre-authorised fallback (design G3)
        ok = [k for k, r in rows.items() if min(r["rate_haar"], r["rate_real"]) >= G3_RATE]
        rec["families"][f] = {"N_unplanted_8000": n_real, "threshold": T, "doses": rows,
                              "k_star": min((k for k, r in rows.items() if r["rate_real"] >= G3_RATE), default=None),
                              "pass_at": min(ok) if ok else None}
        print(f"  G3 {f}: k* {rec['families'][f]['k_star']}", flush=True)
    rec["pass"] = bool(all(v["doses"][G3_DOSES[-1]]["rate_haar"] >= G3_RATE and v["doses"][G3_DOSES[-1]]["rate_real"] >= G3_RATE
                           for v in rec["families"].values()))
    rec["refused_families"] = [f for f, v in rec["families"].items() if v["pass_at"] is None]
    return rec


def g3l(hb8000: Sequence[Dict]) -> Dict:
    """R2's planted effect on the layer-shuffle carrier (recorded, no pass)."""
    rng = np.random.default_rng(seed_for("s1-g3l"))
    rec = {}
    for f in FAMILIES:
        pool = [h[f] for h in hb8000]
        draw = lambda: [pool[i] for i in rng.choice(len(pool), N_HEADS, replace=False)]   # noqa: E731
        T = max(l1_of(draw()) for _ in range(N_NULL))
        rows = {}
        for k in G3L_DOSES["qk" if f.startswith("qk") else "ov"]:
            hits = []
            for _ in range(G3L_DRAWS):
                c = draw()
                elig = [i for i, b in enumerate(c) if b.shape[1] >= 1]
                if len(elig) < k:
                    continue
                u = rng.standard_normal(D_MODEL)
                hits.append(l1_of(plant(c, u / np.linalg.norm(u), rng.choice(elig, k, replace=False))) > T)
            rows[k] = {"rate": float(np.mean(hits)) if hits else None, "n": len(hits)}
        rec[f] = {"shuffle_max": T, "doses": rows}
    return {"families": rec, "pass": None}


def gate(out: Path, arms: Sequence[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Ws8000 = step_weights(PREFLIGHT_STEP)
    hb8000 = [head_bases(W) for W in Ws8000]
    if set(arms) <= {"G3", "G3l"} and (out / "nulls.json").exists():
        # resume after G3's crash (2026-10-20): the per-cell thresholds are the saved ones
        cells = json.loads((out / "nulls.json").read_text())["cells"]
        if "G3" in arms:
            _save(out, "G3", g3(hb8000, {f: cells[f"{PREFLIGHT_STEP}|{f}"]["threshold"] for f in FAMILIES}))
        if "G3l" in arms:
            _save(out, "G3l", g3l(hb8000))
        print(f"gate (resumed) done {time.time() - t0:.0f}s", flush=True)
        return
    null_layer = haar_eigs([R_SUB] * N_HEADS, N_NULL, seed_for("s1-null", "layer", R_SUB), top=1)[:, 0]
    T_sing, null_all = haar_max([R_SUB] * POOL_SIZE["all"], ("all", R_SUB))
    meta = {"code_sha": code_sha(), "r": R_SUB, "n_null": N_NULL, "T_singular_all": T_sing,
            "null_layer_max": float(null_layer.max())}
    print(f"nulls ready {time.time() - t0:.0f}s; T_singular_all {T_sing:.5f}", flush=True)
    if "G0" in arms:
        _save(out, "G0", g0(Ws8000, hb8000, null_layer, null_all))
    # the invariant families' per-cell thresholds, at every scored step and at 8000 (kept for unit 2)
    cells = {}
    hb_steps = {}
    for step in SCORED_STEPS + (PREFLIGHT_STEP,):
        hb = hb8000 if step == PREFLIGHT_STEP else [head_bases(W) for W in step_weights(step)]
        if step in SCORED_STEPS:
            hb_steps[step] = hb
        for f in FAMILIES:
            dims = [h[f].shape[1] for h in hb]
            T = T_sing if f in SINGULAR else haar_max(dims, ("all", f, step))[0]
            cells[f"{step}|{f}"] = {"threshold": T, "dims_sum": int(sum(dims)), "dims_min": int(min(dims))}
        print(f"  thresholds step {step} ({time.time() - t0:.0f}s)", flush=True)
    meta["cells"] = cells
    (out / "nulls.json").write_text(json.dumps(meta, indent=1))
    if "G2" in arms:
        rec = {"cells": {}}
        for step, hb in hb_steps.items():
            rot = rotate_heads(hb, seed_for("s1-g2", step))
            for f in FAMILIES:
                rec["cells"][f"{step}|{f}"] = count_above(mean_projector_eigs([h[f] for h in rot]),
                                                         cells[f"{step}|{f}"]["threshold"])
        rec["n_cells_nonzero"] = int(sum(v > 0 for v in rec["cells"].values()))
        rec["pass"] = rec["n_cells_nonzero"] <= G2_MAX_CELLS
        _save(out, "G2", rec)
    if "G1iii" in arms:
        rec = {"cells": {f: reading([h[f] for h in hb_steps[0]], cells[f"0|{f}"]["threshold"])["N"]
                         for f in FAMILIES}}
        rec["pass"] = all(v == 0 for v in rec["cells"].values())
        _save(out, "G1iii", rec)
    del hb_steps
    if "G1" in arms:
        _save(out, "G1", g1(Ws8000, hb8000, {"layer": null_layer, "all": null_all}))
    if "G3" in arms:
        _save(out, "G3", g3(hb8000, {f: cells[f"{PREFLIGHT_STEP}|{f}"]["threshold"] for f in FAMILIES}))
    if "G3l" in arms:
        _save(out, "G3l", g3l(hb8000))
    print(f"gate done {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- unit 2: R1′ and its re-gate

# placed (design "Blocked 82 decided (b)"): the additive plant's doses, draws and bars
G4_DOSES, G4_MORE = G3_DOSES, G3_MORE
G4_DIRS, G4_DRAWS, G4_RATE, G4_SIZE = 4, 50, 0.8, 0.75
G4T_DOSES, G4T_COS = (12, 16, 24, 32, 48, 64), 0.9
G2P_MIN_CELLS = 55
N_TOP_NULL = 500
G0P_FAMILIES = ("ov_write", "qk_query", "ov_neg", "ov_cplx")


def pr0(dims: Sequence[int], d: int = D_MODEL) -> float:
    """PR₀ = (tr P̄)² / E₀ tr P̄² for independent Haar subspaces of these dims (math check fact 5)."""
    k = np.asarray(dims, float)
    S = k.sum()
    return float(S ** 2 / (S + (S ** 2 - (k ** 2).sum()) / d))


def pr_of(bases: Sequence[np.ndarray]) -> Dict:
    """R1′'s cell: PR = (tr P̄)² / tr P̄² (tr P̄² by the Gram's Frobenius norm, fact 2), PR₀ at the
    pool's own dims, ρ = PR / PR₀ and the excess kernel χ = 1/PR − 1/PR₀."""
    B = np.concatenate([b for b in bases if b.shape[1]], 1)
    n = len(bases)
    G = B.T @ B if B.shape[1] <= B.shape[0] else B @ B.T
    tr, tr2 = B.shape[1] / n, float(np.sum(G * G)) / n ** 2
    p, p0 = tr ** 2 / tr2, pr0([b.shape[1] for b in bases], d=B.shape[0])
    return {"PR": p, "PR0": p0, "rho": p / p0, "chi": 1 / p - 1 / p0}


def haar_tr2(dims: Sequence[int], n_draw: int, seed: int, d: int = D_MODEL, l1: bool = False,
             device: str = "cuda") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """tr P̄² (and λ₁ if asked) per draw for independent Haar subspaces of these dims in R^d (GPU,
    float64; the draws as `haar_eigs`)."""
    import torch
    g = torch.Generator(device=device).manual_seed(seed % (2 ** 63))
    n, K = len(dims), int(sum(dims))
    groups = {k: sum(1 for x in dims if x == k) for k in sorted(set(dims)) if k}
    tr2, top = np.zeros(n_draw), np.zeros(n_draw)
    batch = max(1, int(3e8 // (d * max(K, 1) * 8)))
    for s in range(0, n_draw, batch):
        b = min(batch, n_draw - s)
        Bs = []
        for k, c in groups.items():
            Z = torch.randn(b * c, d, k, generator=g, device=device, dtype=torch.float64)
            Bs.append(torch.linalg.qr(Z)[0].reshape(b, c, d, k).permute(0, 2, 1, 3).reshape(b, d, c * k))
        B = torch.cat(Bs, 2)
        G = B.transpose(1, 2) @ B if K <= d else B @ B.transpose(1, 2)
        tr2[s:s + b] = ((G * G).sum((1, 2)) / n ** 2).cpu().numpy()
        if l1:
            top[s:s + b] = (torch.linalg.eigvalsh(G)[:, -1] / n).cpu().numpy()
    return tr2, (top if l1 else None)


def dims_key(dims: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(int(x) for x in dims)).encode()).hexdigest()[:12]


def pr_band(dims: Sequence[int]) -> Dict:
    """The cell's Haar band (N_NULL draws of its dims): PR's min / max / mean, χ's width, and the
    draws' mean tr P̄² against E₀'s closed form (G0′)."""
    tr2, _ = haar_tr2(list(dims), N_NULL, seed_for("s1-pr-band", dims_key(dims)))
    n, K = len(dims), float(sum(dims))
    p = (K / n) ** 2 / tr2
    p0 = pr0(dims)
    e2 = (K / n) ** 2 / p0
    return {"PR_min": float(p.min()), "PR_max": float(p.max()), "PR_mean": float(p.mean()), "PR0": p0,
            "chi_width": float((1 / p).max() - (1 / p).min()),
            "tr2_z": float((tr2.mean() - e2) / (tr2.std(ddof=1) / np.sqrt(len(tr2))))}


def top_null(dims: Sequence[int], m: int, tag) -> float:
    """The top's null: the largest λ₁ over N_TOP_NULL draws of Haar subspaces of these dims in R^m
    (Haar inside any fixed m-dim subspace), m = round(PR)."""
    return float(haar_tr2([k for k in dims], N_TOP_NULL, seed_for("s1-top", *tag), d=m, l1=True)[1].max())


def plant_add(bases: Sequence[np.ndarray], u: np.ndarray, heads: Sequence[int]) -> List[np.ndarray]:
    """G4's plant: each listed head's basis gains u's part orthogonal to it, normalised (u exactly
    inside, nothing removed, the head's dimension + 1)."""
    out = list(bases)
    for h in heads:
        q = u - out[h] @ (out[h].T @ u)
        out[h] = np.concatenate([out[h], (q / np.linalg.norm(q))[:, None]], 1)
    return out


def plant_add_many(bases: Sequence[np.ndarray], n_dir: int, k: int, rng: np.random.Generator
                   ) -> List[np.ndarray]:
    out = list(bases)
    for _ in range(n_dir):
        u = rng.standard_normal(bases[0].shape[0])
        out = plant_add(out, u / np.linalg.norm(u), rng.choice(len(bases), k, replace=False))
    return out


def g4(hb8000: Sequence[Dict], bands: Dict[str, Dict]) -> Dict:
    """G4 (additive plant, the pass) and G4r (unit 1's displacing plant, recorded) on one draw
    stream: both carriers, Δχ against the unplanted carrier."""
    rng = np.random.default_rng(seed_for("s1-g4"))
    rec = {"families": {}}
    for f in FAMILIES:
        real = [h[f] for h in hb8000]
        dims = [b.shape[1] for b in real]
        band = bands[f]
        chi_real = pr_of(real)["chi"]
        rows, doses = {}, list(G4_DOSES)
        while doses:
            k = doses.pop(0)
            r = {"dchi_haar": [], "below_band": [], "dchi_real": [], "r_dchi_haar": [], "r_dchi_real": []}
            for _ in range(G4_DRAWS):
                carrier = haar_bases(dims, rng)
                c0 = pr_of(carrier)
                pl = pr_of(plant_add_many(carrier, G4_DIRS, k, rng))
                r["dchi_haar"].append(pl["chi"] - c0["chi"])
                r["below_band"].append(pl["PR"] < band["PR_min"])
                r["dchi_real"].append(pr_of(plant_add_many(real, G4_DIRS, k, rng))["chi"] - chi_real)
                for name, car, base in (("r_dchi_haar", carrier, c0["chi"]), ("r_dchi_real", real, chi_real)):
                    disp = plant_many(car, G4_DIRS, k, rng)
                    if disp is not None:
                        r[name].append(pr_of(disp)["chi"] - base)
            med = float(np.median(r["dchi_haar"]))
            dr = np.array(r["dchi_real"])
            rows[k] = {"rate_detect_haar": float(np.mean(r["below_band"])), "dchi_haar_median": med,
                       "rate_size_real": float(np.mean(dr >= G4_SIZE * med)),
                       "rate_resolved_real": float(np.mean(dr > band["chi_width"])),
                       "ratio_real_median": float(np.median(dr) / med),
                       "G4r_dchi_haar_median": float(np.median(r["r_dchi_haar"])) if r["r_dchi_haar"] else None,
                       "G4r_dchi_real_median": float(np.median(r["r_dchi_real"])) if r["r_dchi_real"] else None,
                       **{key: [float(x) for x in v] for key, v in r.items() if key != "below_band"}}
            rows[k]["pass"] = rows[k]["rate_detect_haar"] >= G4_RATE and rows[k]["rate_size_real"] >= G4_RATE
            if k == G4_DOSES[-1] and not rows[k]["pass"] and not doses:
                doses = list(G4_MORE)                  # pre-authorised fallback (design)
        ok = [k for k, v in rows.items() if v["pass"]]
        rec["families"][f] = {"chi_unplanted": chi_real, "band": band, "doses": rows,
                              "usable": bool(ok),
                              "k_star": min((k for k, v in rows.items()
                                             if v["pass"] and v["rate_resolved_real"] >= G4_RATE), default=None)}
        print(f"  G4 {f}: pass at 32 {rows[G4_DOSES[-1]]['pass']}, k* {rec['families'][f]['k_star']}", flush=True)
    rec["pass"] = bool(all(v["doses"][G4_DOSES[-1]]["pass"] for v in rec["families"].values()))
    rec["spectrum_only"] = [f for f, v in rec["families"].items() if not v["usable"]]
    return rec


def g4t(hb8000: Sequence[Dict]) -> Dict:
    """The top's planted control (recorded): one direction added to k of 8000's heads; recovered if
    an eigenvector of P̄ has |cos| ≥ G4T_COS with it, above the m-dim null's largest λ₁."""
    rng = np.random.default_rng(seed_for("s1-g4t"))
    rec = {}
    for f in FAMILIES:
        real = [h[f] for h in hb8000]
        dims = [b.shape[1] for b in real]
        m = int(round(pr_of(real)["PR"]))
        T = top_null(dims, m, ("g4t", f))
        rows = {}
        for k in G4T_DOSES:
            hit, cos, ratio = [], [], []
            for _ in range(G4_DRAWS):
                u = rng.standard_normal(D_MODEL)
                u /= np.linalg.norm(u)
                lam, V = mean_projector(plant_add(real, u, rng.choice(len(real), k, replace=False)))
                c = np.abs(V.T @ u)
                j = int(np.argmax(c))
                hit.append(bool(c[j] >= G4T_COS and lam[j] > T))
                cos.append(float(c[j]))
                ratio.append(float(lam[j] / T))
            rows[k] = {"rate": float(np.mean(hit)), "cos_median": float(np.median(cos)),
                       "lam_over_null_median": float(np.median(ratio))}
        rec[f] = {"m": m, "null_max_l1": T, "doses": rows}
        print(f"  G4t {f}: m {m}, " + ", ".join(f"{k}:{v['rate']:.2f}" for k, v in rows.items()), flush=True)
    return {"families": rec, "pass": None}


def regate(out: Path) -> None:
    """Unit 2's re-gate (design "Blocked 82 decided (b)"): G0′, G1iii′, G2′, G4 (+ G4r), G4t. Reads
    no scored-step R1′ value except step 0's (G1iii′) and the rotation null's."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hb = {s: step_bases(s) for s in SCORED_STEPS + (PREFLIGHT_STEP,)}
    print(f"bases ready {time.time() - t0:.0f}s", flush=True)
    bands, cell_key = {}, {}
    for s, h in hb.items():
        for f in FAMILIES:
            dims = [x[f].shape[1] for x in h]
            key = dims_key(dims)
            if key not in bands:
                bands[key] = pr_band(dims)
            cell_key[f"{s}|{f}"] = key
        print(f"  bands step {s} ({time.time() - t0:.0f}s, {len(bands)} distinct)", flush=True)
    (out / "bands.json").write_text(json.dumps({"code_sha": code_sha(), "n_null": N_NULL, "bands": bands,
                                                "cells": cell_key}, indent=1))
    band = lambda s, f: bands[cell_key[f"{s}|{f}"]]   # noqa: E731
    inside = lambda p, b: b["PR_min"] <= p <= b["PR_max"]   # noqa: E731
    h8 = hb[PREFLIGHT_STEP]
    errs = {}
    for f in G0P_FAMILIES:
        ev = mean_projector_eigs([x[f] for x in h8])
        errs[f] = abs(ev.sum() ** 2 / np.sum(ev ** 2) / pr_of([x[f] for x in h8])["PR"] - 1)
    z = {k: b["tr2_z"] for k, b in bands.items()}
    _save(out, "G0p", {"pr_rel_err": errs, "tr2_z": z,
                       "pass": bool(max(errs.values()) < 1e-10 and max(abs(v) for v in z.values()) < 4)})
    rec = {"cells": {f: {**pr_of([x[f] for x in hb[0]]), "band": band(0, f)} for f in FAMILIES}}
    rec["pass"] = all(inside(v["PR"], v["band"]) for v in rec["cells"].values())
    _save(out, "G1iii_p", rec)
    rec = {"cells": {}}
    for s in SCORED_STEPS:
        rot = rotate_heads(hb[s], seed_for("s1-g2", s))
        for f in FAMILIES:
            p = pr_of([x[f] for x in rot])["PR"]
            rec["cells"][f"{s}|{f}"] = {"PR": p, "inside": inside(p, band(s, f))}
    rec["n_inside"] = int(sum(v["inside"] for v in rec["cells"].values()))
    rec["pass"] = rec["n_inside"] >= G2P_MIN_CELLS
    _save(out, "G2p", rec)
    for s in SCORED_STEPS[1:]:
        del hb[s]
    _save(out, "G4", g4(h8, {f: band(PREFLIGHT_STEP, f) for f in FAMILIES}))
    _save(out, "G4t", g4t(h8))
    print(f"regate done {time.time() - t0:.0f}s", flush=True)


def fold_ln1(W: Dict) -> Dict:
    """LN1's gain folded into the read side: W_Q, W_K, W_V right-multiplied by diag(g)."""
    return {**W, "Wq": W["Wq"] * W["g"], "Wk": W["Wk"] * W["g"], "Wv": W["Wv"] * W["g"]}


def r1(out: Path, gate_dir: Path) -> None:
    """R1′ at the 8 scored steps, after the re-gate (refuses unless G0′, G1iii′ and G2′ pass; a
    family G4 left unusable is read as its spectrum only)."""
    rec = {a: json.loads((gate_dir / f"{a}.json").read_text()) for a in ("G0p", "G1iii_p", "G2p", "G4")}
    failed = [a for a in ("G0p", "G1iii_p", "G2p") if not rec[a]["pass"]]
    if failed:
        raise S1Error(f"re-gate arms failed: {failed}; R1′ refused (design fallbacks)")
    bands = json.loads((gate_dir / "bands.json").read_text())
    spectrum_only = set(rec["G4"]["spectrum_only"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"code_sha": code_sha(), "gate_dir": str(gate_dir), "spectrum_only": sorted(spectrum_only),
           "cells": {}}
    path = out / "r1.json"
    if path.exists():                          # resume: written after every step (`LESSONS.md` 11)
        prev = json.loads(path.read_text())
        if prev["code_sha"] != res["code_sha"]:
            raise S1Error(f"{path} was written by code {prev['code_sha']}; not resuming")
        res = prev
    for s in SCORED_STEPS:
        if all(f"{s}|{f}" in res["cells"] for f in FAMILIES):
            continue
        Ws = step_weights(s)
        frames = {"r16": [head_bases(W) for W in Ws]}
        frames.update({f"r{r}": [head_bases(W, r) for W in Ws] for r in R_BESIDE})
        frames["folded"] = [head_bases(fold_ln1(W)) for W in Ws]
        del Ws
        for f in FAMILIES:
            bs = [x[f] for x in frames["r16"]]
            ev = mean_projector_eigs(bs)
            cell = {"eig": ev[:200].tolist(), "k_star": rec["G4"]["families"][f]["k_star"]}
            if f not in spectrum_only:
                cell.update(rho_cell(s, f, frames, bands, ev))
            res["cells"][f"{s}|{f}"] = cell
        path.write_text(json.dumps(res, indent=1, default=float))
        print(f"step {s} ({time.time() - t0:.0f}s)", flush=True)
    print(f"r1 done {time.time() - t0:.0f}s", flush=True)


def rho_cell(s: int, f: str, frames: Dict[str, List[Dict]], bands: Dict, ev: np.ndarray) -> Dict:
    """One R1′ cell: PR, PR₀, ρ, χ, the gate's band, the top against the m-dim null, ρ beside."""
    bs = [x[f] for x in frames["r16"]]
    key = bands["cells"].get(f"{s}|{f}")
    dims = [b.shape[1] for b in bs]
    if key != dims_key(dims):
        raise S1Error(f"{s}|{f}: the gate's band was drawn for other dims")
    cell = {**pr_of(bs), "band": bands["bands"][key]}
    m = int(round(cell["PR"]))
    T = top_null(dims, m, ("r1", s, f))
    cell["top"] = {"m": m, "null_max_l1": T, "l_over_null": (ev[:10] / T).tolist()}
    cell["beside_rho"] = {fr: pr_of([x[f] for x in frames[fr]])["rho"] for fr in ("r8", "r32", "folded")}
    return cell


# ---------------------------------------------------------------- unit 3 (design "Blocked 83–85 decided")

QK = ("qk_query", "qk_key")
S_WIDTH = 10                               # 84 (a): S = P̄'s top 10 eigenvectors
R3_HAAR = 100
R3_READS = ("share", "cplx", "arg")        # summarised against Haar's [min, max]
N_PC = 10


def r1_qk(out: Path, gate_dir: Path) -> None:
    """83 (a): QK's 16 R1′ cells, as the OV cells (`rho_cell`), to `r1_qk.json`."""
    rec = {a: json.loads((gate_dir / f"{a}.json").read_text()) for a in ("G0p", "G1iii_p", "G2p")}
    failed = [a for a, v in rec.items() if not v["pass"]]
    if failed:
        raise S1Error(f"re-gate arms failed: {failed}; R1′ refused (design fallbacks)")
    bands = json.loads((gate_dir / "bands.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"code_sha": code_sha(), "gate_dir": str(gate_dir), "cells": {}}
    for s in SCORED_STEPS:
        Ws = step_weights(s)
        frames = {"r16": [head_bases(W) for W in Ws]}
        frames.update({f"r{r}": [head_bases(W, r) for W in Ws] for r in R_BESIDE})
        frames["folded"] = [head_bases(fold_ln1(W)) for W in Ws]
        del Ws
        for f in QK:
            ev = mean_projector_eigs([x[f] for x in frames["r16"]])
            res["cells"][f"{s}|{f}"] = {"eig": ev[:200].tolist(), **rho_cell(s, f, frames, bands, ev)}
        (out / "r1_qk.json").write_text(json.dumps(res, indent=1, default=float))
        print(f"step {s} ({time.time() - t0:.0f}s)", flush=True)
    print(f"r1_qk done {time.time() - t0:.0f}s", flush=True)


def shared_space(bases: Sequence[np.ndarray], width: int = S_WIDTH) -> np.ndarray:
    """84 (a): U_S, P̄'s top `width` eigenvectors (d × width)."""
    return mean_projector(bases)[1][:, :width]


def layer_ov(Ws: Sequence[Dict], l: int) -> np.ndarray:
    """V_l = Σ_h W_O W_V over layer l's heads (raw frame)."""
    return sum(W["Wo"] @ W["Wv"] for W in Ws[l * N_HEADS:(l + 1) * N_HEADS])


def r3_reading(C: np.ndarray) -> Dict:
    """R3 on one compression C = U_Sᵀ V_l U_S: B2's `share`, `top_weight` and negative-pair flag on
    the symmetric part; the complex share of C's eigenvalues and their mean |arg λ| (None if none;
    LAPACK gives a real eigenvalue of a real matrix an exactly zero imaginary part)."""
    from trackB_circuit.b2 import share, top_weight
    ev = np.linalg.eigvalsh((C + C.T) / 2)
    tw, pair = top_weight(ev)
    lam = np.linalg.eigvals(C)
    cx = lam[lam.imag != 0]
    return {"share": share(ev), "top_weight": tw, "pair": pair, "cplx": cx.size / lam.size,
            "arg": float(np.mean(np.abs(np.angle(cx)))) if cx.size else None}


def r3_summary(obs: Sequence[Dict], haar: Sequence[Sequence[Dict]]) -> Dict:
    """Per family × step (design, "Summary (fixed now)"): each reading's median over the layers;
    for `share`, `cplx`, `arg` the layers outside Haar's [min, max] at that layer; the flag's layer
    count beside Haar's mean count. `obs[l]`, `haar[l][j]` are `r3_reading`s."""
    out = {}
    for k in ("share", "top_weight", "cplx", "arg"):
        v = [o[k] for o in obs if o[k] is not None]
        out[f"{k}_median"] = float(np.median(v)) if v else None
    for k in R3_READS:
        n = 0
        for o, hs in zip(obs, haar):
            hv = [h[k] for h in hs if h[k] is not None]
            if o[k] is not None and hv and not min(hv) <= o[k] <= max(hv):
                n += 1
        out[f"{k}_outside"] = n
    out["pair_layers"] = int(sum(o["pair"] for o in obs))
    out["pair_layers_haar_mean"] = float(np.mean([sum(hs[j]["pair"] for hs in haar)
                                                  for j in range(len(haar[0]))]))
    return out


def variance_share(U: np.ndarray, S: np.ndarray, n_pc: int = N_PC) -> Dict:
    """The residual check for one covariance S: q = tr(Uᵀ S U) / (S's top n_pc eigenvalues summed),
    the overlap o = ‖Uᵀ U_pc‖²_F / width, and a Haar U's expected q, (width / d) tr S / top sum."""
    lam, E = np.linalg.eigh(S)
    top = lam[::-1][:n_pc].sum()
    w, d = U.shape[1], U.shape[0]
    return {"q": float(np.trace(U.T @ S @ U) / top),
            "o": float(np.sum((U.T @ E[:, ::-1][:, :n_pc]) ** 2) / w),
            "q_haar": float(w / d * np.trace(S) / top)}


def residual_moments(model, tok) -> Dict[str, List[np.ndarray]]:
    """Block inputs l = 0 … 23 on the 7 scored v1 prompts, pooled: centred covariance with position 0
    dropped (primary) and kept, and the uncentred second moment (position 0 dropped)."""
    import torch
    from core.holdout import HELD_OUT_PROMPT_KEYS
    from trackB_circuit.b0_gate import V1_SCORED, token_ids
    if set(V1_SCORED) & HELD_OUT_PROMPT_KEYS:
        raise S1Error("a held-out prompt in V1_SCORED")
    dev = next(model.parameters()).device
    xs = {"drop0": [[] for _ in range(N_LAYERS)], "keep0": [[] for _ in range(N_LAYERS)]}
    with torch.no_grad():
        for key in V1_SCORED:
            ids = torch.tensor([token_ids(tok, key)], device=dev)
            hs = model(ids, output_hidden_states=True).hidden_states
            for l in range(N_LAYERS):
                x = hs[l][0].double().cpu().numpy()
                xs["keep0"][l].append(x)
                xs["drop0"][l].append(x[1:])
    out = {"cov": [], "cov_keep0": [], "second": []}
    for l in range(N_LAYERS):
        X, Xk = np.concatenate(xs["drop0"][l]), np.concatenate(xs["keep0"][l])
        out["cov"].append(np.cov(X, rowvar=False))
        out["cov_keep0"].append(np.cov(Xk, rowvar=False))
        out["second"].append(X.T @ X / len(X))
    return out


def r3(out: Path) -> None:
    """Unit 3: R3 on S and the residual check at the 8 scored steps, written after every step."""
    import torch
    from trackB_circuit.b0_gate import load
    from trackB_circuit.b2 import layer_weights
    out.mkdir(parents=True, exist_ok=True)
    path = out / "r3.json"
    res = {"code_sha": code_sha(), "s_width": S_WIDTH, "n_haar": R3_HAAR, "cells": {}}
    if path.exists():                          # resume (`LESSONS.md` 11)
        prev = json.loads(path.read_text())
        if prev["code_sha"] != res["code_sha"]:
            raise S1Error(f"{path} was written by code {prev['code_sha']}; not resuming")
        res = prev
    t0 = time.time()
    V0 = [layer_ov(step_weights(0), l) for l in range(N_LAYERS)]
    for s in SCORED_STEPS:
        if all(f"{s}|{f}" in res["cells"] for f in FAMILIES):
            continue
        model, tok = load(s, "cpu", torch.float32)
        Ws = [W for l in range(N_LAYERS) for W in layer_weights(model, l)]
        model = model.to("cuda")
        mom = residual_moments(model, tok)
        del model
        hb = [head_bases(W) for W in Ws]
        Vl = [layer_ov(Ws, l) for l in range(N_LAYERS)]
        del Ws
        for f in FAMILIES:
            U = shared_space([x[f] for x in hb])
            rng = np.random.default_rng(seed_for("s1-r3-haar", s, f))
            H = [np.linalg.qr(rng.standard_normal((D_MODEL, S_WIDTH)))[0] for _ in range(R3_HAAR)]
            obs = [r3_reading(U.T @ V @ U) for V in Vl]
            haar = [[r3_reading(h.T @ V @ h) for h in H] for V in Vl]
            rng_ = [{k: [min(v), max(v)] if (v := [h[k] for h in hs if h[k] is not None]) else None
                     for k in R3_READS} for hs in haar]
            step0 = [r3_reading(U.T @ V @ U) for V in V0]
            cell = {"r3": {"layers": obs, "step0": step0, "haar_range": rng_,
                           "summary": r3_summary(obs, haar), "summary_step0": r3_summary(step0, haar)},
                    "top_coords": [np.argsort(-np.abs(U[:, i]))[:3].tolist() for i in range(S_WIDTH)],
                    "resid": {}}
            for m, Ss in mom.items():
                rows = [variance_share(U, S) for S in Ss]
                cell["resid"][m] = {"layers": rows,
                                    **{f"{k}_{a}": float(fn([r[k] for r in rows]))
                                       for k in ("q", "o", "q_haar")
                                       for a, fn in (("median", np.median), ("max", np.max))}}
            cell["resid"]["top_variance"] = bool(cell["resid"]["cov"]["q_median"] >= 0.5)
            res["cells"][f"{s}|{f}"] = cell
        path.write_text(json.dumps(res, indent=1, default=float))
        print(f"step {s} ({time.time() - t0:.0f}s)", flush=True)
    print(f"r3 done {time.time() - t0:.0f}s", flush=True)


MASSIVE_STEP = 143000


def massive(out: Path) -> None:
    """`lit-S1.md` Parked 2, beside the residual check (description): 410m's massive-activation
    coordinates at 143000. Per block input l, the 3 coordinates with the largest max |x| over the
    7 v1 prompts' tokens (position 0 dropped, and position 0 alone), each with that max over the
    coordinates' median max; then each family's S-weight ‖U_Sᵀ e_i‖² on the coordinates found at
    l ≥ 1 (a Haar S: 10/1024 each)."""
    import torch
    from core.holdout import HELD_OUT_PROMPT_KEYS
    from trackB_circuit.b0_gate import V1_SCORED, load, token_ids
    from trackB_circuit.b2 import layer_weights
    if set(V1_SCORED) & HELD_OUT_PROMPT_KEYS:
        raise S1Error("a held-out prompt in V1_SCORED")
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load(MASSIVE_STEP, "cpu", torch.float32)
    Ws = [W for l in range(N_LAYERS) for W in layer_weights(model, l)]
    model = model.to("cuda")
    rest, first = [[] for _ in range(N_LAYERS)], [[] for _ in range(N_LAYERS)]
    with torch.no_grad():
        for key in V1_SCORED:
            ids = torch.tensor([token_ids(tok, key)], device="cuda")
            hs = model(ids, output_hidden_states=True).hidden_states
            for l in range(N_LAYERS):
                a = hs[l][0].abs().double().cpu().numpy()
                rest[l].append(a[1:].max(0))
                first[l].append(a[0])
    del model

    def top3(a: np.ndarray) -> List[Dict]:
        med = float(np.median(a))
        return [{"coord": int(i), "max_abs": float(a[i]), "over_median": float(a[i] / med)}
                for i in np.argsort(-a)[:3]]

    layers = [{"rest": top3(np.max(rest[l], 0)), "pos0": top3(np.max(first[l], 0))} for l in range(N_LAYERS)]
    coords = sorted({c["coord"] for L in layers[1:] for c in L["rest"] + L["pos0"]})
    hb = [head_bases(W) for W in Ws]
    weight = {}
    for f in FAMILIES:
        U = shared_space([x[f] for x in hb])
        weight[f] = {str(i): float(U[i] @ U[i]) for i in coords}
    rec = {"code_sha": code_sha(), "step": MASSIVE_STEP, "layers": layers, "coords": coords,
           "s_weight": weight, "haar_s_weight": S_WIDTH / D_MODEL}
    (out / "massive.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"massive: coords {coords}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("explore")
    e.add_argument("--steps", type=int, nargs="+", required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--n-draw", type=int, default=2000)
    p = sub.add_parser("preflight")
    p.add_argument("--out", type=Path, required=True)
    gt = sub.add_parser("gate")
    gt.add_argument("--out", type=Path, required=True)
    gt.add_argument("--arms", nargs="+", default=["G0", "G2", "G1iii", "G1", "G3", "G3l"])
    rg = sub.add_parser("regate")
    rg.add_argument("--out", type=Path, required=True)
    rr = sub.add_parser("r1")
    rr.add_argument("--out", type=Path, required=True)
    rr.add_argument("--gate", type=Path, required=True)
    rq = sub.add_parser("r1qk")
    rq.add_argument("--out", type=Path, required=True)
    rq.add_argument("--gate", type=Path, required=True)
    r3p = sub.add_parser("r3")
    r3p.add_argument("--out", type=Path, required=True)
    ms = sub.add_parser("massive")
    ms.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "regate":
        regate(a.out)
    elif a.cmd == "r1":
        r1(a.out, a.gate)
    elif a.cmd == "r1qk":
        r1_qk(a.out, a.gate)
    elif a.cmd == "r3":
        r3(a.out)
    elif a.cmd == "massive":
        massive(a.out)
    elif a.cmd == "explore":
        explore(a.steps, a.out, a.n_draw)
    elif a.cmd == "preflight":
        preflight(a.out)
    elif a.cmd == "gate":
        gate(a.out, a.arms)


if __name__ == "__main__":
    main()
