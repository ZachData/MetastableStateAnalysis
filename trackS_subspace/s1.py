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
    a = ap.parse_args(argv)
    if a.cmd == "explore":
        explore(a.steps, a.out, a.n_draw)
    elif a.cmd == "preflight":
        preflight(a.out)
    elif a.cmd == "gate":
        gate(a.out, a.arms)


if __name__ == "__main__":
    main()
