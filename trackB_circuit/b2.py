"""B2, spectra from the weights, unit 1 (`trackB_circuit/design-B2.md`, frozen 2026-10-17, Blocked
61 (a), with the LN1 frame's QK form corrected before any output, below): the columns C1–C6 per
head, the member statistic and its layer-matched null, C7's reader, and the gate G0–G7. No token is
read; unit 2 builds the table.

**Frames.** B2a's LN1 frame is the space of LN1's rows y (`b2a.ln1`, gain in): the head's logit is
``yᵀ W_Qᵀ W_K y′ / √64`` (`b2a.head_logits`) and V is ``diag(g) C W_O W_V``, mapping y into the frame.
So in that frame **M_0 = W_Qᵀ W_K / √64**, the same matrix as the raw frame's; the design's
``diag(g) C … C diag(g)`` applied the gain twice (the user's pick, 2026-10-17, before any output).
C3 is therefore frame-free, and C1, C2, C4 differ between frames through V only. The raw frame's V
is ``W_O W_V``. Convention: columns act on column vectors, ``x ↦ V x`` and ``logit = xᵀ M y``, so the
paper's condition is V = M (2d's `ov_qk_alignment`); the stored dense OV is ``(W_O W_V)ᵀ``
(`p2_eigenspectra/weights.py`, row convention), which G0 checks.

**Exact low-rank spectra.** Every matrix here is ``A Bᵀ`` with A, B of size d × k (k = 64, or 66
for G4's planted pair). Its symmetric part is ``Q (R J Rᵀ) Qᵀ`` with ``[A B] = Q R`` (QR, Q with
orthonormal columns) and ``J = ½[[0, I], [I, 0]]``, so its nonzero eigenvalues are those of the
2k × 2k ``R J Rᵀ`` (`tools/math_checks/b2_symmetric_inertia.py` fact 1). Frobenius products are
``⟨A₁B₁ᵀ, A₂B₂ᵀ⟩ = Σ (A₁ᵀA₂) ∘ (B₁ᵀB₂)``. G0 (iii) checks both against 2d's dense functions.

**Columns** (design "The columns"; one scalar each): C1 ``a(V)``; C2 the signed top weight of
sym V (the negative-pair flag beside); C3 ``a(M_0)`` (2d's asymmetry and the unrotated 48
dimensions' share beside); C4 2d's signed cosine of V and M_0 (2d's regime label beside, by
`regime_label`, which is `head_regime`'s rule on the same two numbers, checked in G0 (iii));
C6 the K-composition overlap ``‖W_Aᵀ K_B‖²_F / r`` (W_A the top-r left singular vectors of A's
raw ``W_O W_V``, K_B the top-r right singular vectors of B's ``W_K``), r = 16, 8 and 32 beside;
C7 the QK read rank ``r90`` from B2a's records.

**The rule** (design "The rule"): a cell's percentile p in its own step's 384 heads is
``(#below + ½ #tied) / 384``; ``T = mean over members and steps of |p − ½|``; the null is 10,000
layer-matched sets (one head per member's layer, drawn from that layer's heads other than the
member); rank = the share of null T strictly below the observed (ties count against).

**Gate as built (choices the design left open, made before any output):** G2's T and G3's and G5's
read one step each (the arm's step), not the rule's mean over steps ≥ 512; the fake members' null
pool is their layers' other 15 heads, as the real members' is. G3's dose adds ``t · A Aᵀ`` to V
(``W_V += t Aᵀ``, A = ``diag(g) C W_O``, so W_V moves along the folded ``W_Oᵀ``), t by bisection
to the target percentile of the step's unplanted C1 (t = 0 for a head already above it). G5 sets
B's top-k right singular vectors of ``W_K`` to the orthonormalised planted directions (A's top 4
write directions for every A below B), keeping W_K's singular values; its primary is one seeded
set, as the design reads, with 200 draws beside.

**The re-gate (Blocked 62 (a), the user's, 2026-10-18; `regate`):** G3m is G3 on the rule's own
statistic, T over the 9 steps ≥ 512. The same 200 fake sets as G3 (seed ``b2-g3``); at every step
each planted head is dosed, as G3 doses, to that step's own quantile of its unplanted C1, and each
step's percentiles are taken with the set's planted values in. Doses 70th–90th; the pass is G3's
(the 90th, ≥ 80 % of draws), the rest beside. A late-onset variant plants only at steps ≥ 8000 (5
of the 9), the unplanted values at 512–4000, beside. A one-step slice at 143000 with G3's own seeds
must return unit 1's G3 rates (the wiring check). G5's dose curve copies 1 and 2 of A's write
directions beside G5's 4, at 143000 as G5 reads. 62 (c) is frozen as written: p in the step's 384.

NUMERICS: float64 on the CPU. Checkpoints load in float32 (`b0_gate.load`) and their weights are
cast to float64, exactly; the stored OVs were extracted in float32, so G0 (i)'s tolerance 1e-5 is
float32's eps with room. CPU, not the GPU, because every operation is a QR or SVD of at most
1024 × 132 per head (seconds per step), and the population is never on a GPU-sized batch.

Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b2 gate --out <dir> [--arms G0 G1 ...]
    python -m trackB_circuit.b2 regate --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackB_circuit.b2a import (D_MODEL, HEAD_DIM, N_HEADS, N_LAYERS, NAMED, REPO, ROT_DIMS, STEPS,
                                parse_head, seed_for)


DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
SERIES = DATA / "analysis/ov_per_head_series.json"
B2A = DATA / "b2a"

MEMBERS = NAMED                           # L5H2, L7H8, L12H5, L8H6, L11H14, L10H7
CIRCUIT = NAMED[:5]                       # C6's five
N_POP = N_LAYERS * N_HEADS                # 384
R_SUB, R_BESIDE = 16, (8, 32)             # placed (design): C6's r
PAIR_RATIO = 0.8                          # placed (design): negative-pair flag |λ₂|/|λ₁|
STAT_FROM = 512                           # the statistic's steps (design rule 2)
STAT_STEPS = tuple(x for x in STEPS if x >= STAT_FROM)   # 9 steps
N_NULL, N_DRAW = 10_000, 200
RANK_PASS = 0.99                          # 0.05 / 5, one family
C7_FRAC = 0.1                             # placed (design, after 143000's M2 was seen)
G0_TOL, G0_FOLD_TOL, G0_DENSE_TOL = 1e-5, 1e-10, 1e-8
G1_BAND = (0.03, 0.08)
G2_MAX = 0.02
G3_DOSE, G3_DOSES, G3_PASS = 0.90, (0.75, 0.80, 0.90, 0.95, 0.99), 0.80
G4_NU = 2.0
G5_COPY = 4
G3M_DOSES, G3M_LATE = (0.70, 0.75, 0.80, 0.90), 8000   # Blocked 62 (a): 0.90 gated, the rest beside
G5_COPIES = (1, 2)                                      # G5's dose curve, beside
G6_ASYM, G6_ALIGN = 1e-6, 0.999
G7_TV = 1e-6
G7_CODES = {"2026-10-16": "9fe738d5a008", "2026-10-16-occ": "37d4f2487738",
            "2026-10-16-occ2": "939e114f7afb"}
G7_PROMPTS = "976b62d677bc"
COLS = ("C1", "C2", "C3", "C4")
ARMS = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7")


class B2Error(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackB_circuit/b2.py").read_bytes()).hexdigest()[:12]


def head_index(name: str) -> int:
    l, h = parse_head(name)
    return l * N_HEADS + h


def head_name(i: int) -> str:
    return f"L{i // N_HEADS}H{i % N_HEADS}"


# ---------------------------------------------------------------- exact low-rank algebra

def sym_eigs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Eigenvalues of ½(A Bᵀ + B Aᵀ) on the span of [A B] (the rest are zeros), ascending."""
    k = A.shape[1]
    Q, R = np.linalg.qr(np.concatenate([A, B], 1))
    J = np.zeros((2 * k, 2 * k))
    J[:k, k:] = J[k:, :k] = 0.5 * np.eye(k)
    return np.linalg.eigvalsh(R @ J @ R.T)


def frob_dot(A1, B1, A2, B2) -> float:
    return float(((A1.T @ A2) * (B1.T @ B2)).sum())


def share(ev: np.ndarray) -> float:
    """a = Σλ₊ / Σ|λ|."""
    return float(ev[ev > 0].sum() / np.abs(ev).sum())


def top_weight(ev: np.ndarray) -> Tuple[float, bool]:
    """C2: sign(λ₁)|λ₁| / Σ|λ| (λ₁ the largest |λ|), and the negative-pair flag."""
    o = np.argsort(-np.abs(ev))
    l1, l2 = ev[o[0]], ev[o[1]]
    return float(l1 / np.abs(ev).sum()), bool(l1 < 0 and l2 < 0 and abs(l2) / abs(l1) >= PAIR_RATIO)


def regime_label(asym: float, align: float) -> str:
    """`gradient_flow_condition.head_regime`'s rule on its two numbers (checked in G0 (iii))."""
    from p2d_operator_activation.gradient_flow_condition import ALIGN_TOL, SYMMETRY_TOL
    if not np.isfinite(align):
        return "outside"
    if align < -ALIGN_TOL:
        return "repulsive_aligned"
    if asym < SYMMETRY_TOL and align > ALIGN_TOL:
        return "gradient_flow"
    return "symmetric_only" if asym < SYMMETRY_TOL else "outside"


# ---------------------------------------------------------------- one head's columns

def v_factors(W: Dict, frame: str) -> Tuple[np.ndarray, np.ndarray]:
    """V = A Bᵀ: LN1 frame A = diag(g) C W_O, raw A = W_O; B = W_Vᵀ."""
    Wo = W["Wo"]
    A = W["g"][:, None] * (Wo - Wo.mean(0)) if frame == "ln" else Wo
    return A, W["Wv"].T


def m_factors(W: Dict, dims: slice = slice(None)) -> Tuple[np.ndarray, np.ndarray]:
    """M_0 = W_Qᵀ W_K / √64 (both frames), or its part on the head dimensions `dims`."""
    return W["Wq"][dims].T / np.sqrt(HEAD_DIM), W["Wk"][dims].T


def m_row(W: Dict) -> Dict:
    Aq, Bk = m_factors(W)
    ev = sym_eigs(Aq, Bk)
    fro2 = frob_dot(Aq, Bk, Aq, Bk)
    asym = float(np.sqrt(max(fro2 - (ev ** 2).sum(), 0.0) / fro2))
    return {"C3": share(ev), "asym": asym,
            "C3_unrot": share(sym_eigs(*m_factors(W, slice(ROT_DIMS, None)))),
            "_m": (Aq, Bk, fro2)}


def v_row(A: np.ndarray, B: np.ndarray, m: Dict) -> Dict:
    ev = sym_eigs(A, B)
    c2, pair = top_weight(ev)
    Aq, Bk, fro2 = m["_m"]
    align = frob_dot(A, B, Aq, Bk) / np.sqrt(frob_dot(A, B, A, B) * fro2)
    return {"C1": share(ev), "C2": c2, "pair": pair, "C4": float(align),
            "regime": regime_label(m["asym"], align)}


def head_row(W: Dict) -> Dict:
    """C1–C4 in both frames (C3 is frame-free) and the beside values."""
    m = m_row(W)
    out = {k: m[k] for k in ("C3", "asym", "C3_unrot")}
    for frame in ("ln", "raw"):
        for k, v in v_row(*v_factors(W, frame), m).items():
            out[k if frame == "ln" else f"{k}_raw"] = v
    return out


def bases(W: Dict, r: int = max(R_BESIDE)) -> Tuple[np.ndarray, np.ndarray]:
    """Raw frame: the top-r left singular vectors of W_O W_V (write) and right of W_K (key)."""
    Q, R = np.linalg.qr(W["Wo"])
    U, _, _ = np.linalg.svd(R @ W["Wv"], full_matrices=False)
    _, _, Vt = np.linalg.svd(W["Wk"], full_matrices=False)
    return Q @ U[:, :r], Vt[:r].T


def overlap(Wa: np.ndarray, Kb: np.ndarray, r: int = R_SUB) -> float:
    """tr(P_W P_K) / r on the top-r columns."""
    return float(((Wa[:, :r].T @ Kb[:, :r]) ** 2).sum() / r)


# ---------------------------------------------------------------- a checkpoint's weights

def layer_weights(model, l: int) -> List[Dict]:
    blk = model.gpt_neox.layers[l]
    W = blk.attention.query_key_value.weight.detach().double().numpy().reshape(N_HEADS, 3 * HEAD_DIM, -1)
    Wo = blk.attention.dense.weight.detach().double().numpy().reshape(-1, N_HEADS, HEAD_DIM)
    g = blk.input_layernorm.weight.detach().double().numpy()
    return [{"Wq": W[h, :HEAD_DIM], "Wk": W[h, HEAD_DIM:2 * HEAD_DIM], "Wv": W[h, 2 * HEAD_DIM:],
             "Wo": Wo[:, h], "g": g} for h in range(N_HEADS)]


def load_weights(step: int) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """All 384 heads' factors at `step`, float64, indexed by `head_index`; beside, B2a's own V_h
    (`b2a.head_weights`) for its 14 heads, for G0 (ii)."""
    import torch
    from trackB_circuit import b2a
    from trackB_circuit.b0_gate import load
    model, _ = load(step, "cpu", torch.float32)
    Ws = [w for l in range(N_LAYERS) for w in layer_weights(model, l)]
    refs = {}
    for h in b2a.heads():
        hw = b2a.head_weights(model, h)
        refs[h] = hw["left"] @ hw["right"]
    del model
    return Ws, refs


def population(Ws: Sequence[Dict]) -> Dict:
    """Every head's columns (arrays of 384) and raw bases."""
    rows = [head_row(W) for W in Ws]
    pop = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    b = [bases(W) for W in Ws]
    pop["W"] = np.stack([x[0] for x in b])
    pop["K"] = np.stack([x[1] for x in b])
    return pop


# ---------------------------------------------------------------- the rule

def percentiles(x: np.ndarray) -> np.ndarray:
    """(#below + ½ #tied) / n for every entry."""
    s = np.sort(x)
    lo = np.searchsorted(s, x, "left")
    hi = np.searchsorted(s, x, "right")
    return (lo + 0.5 * (hi - lo)) / len(x)


def layer_pools(heads: Sequence[int]) -> List[np.ndarray]:
    """For each head, its layer's other 15 heads."""
    out = []
    for i in heads:
        l = i // N_HEADS
        out.append(np.array([l * N_HEADS + h for h in range(N_HEADS) if l * N_HEADS + h != i]))
    return out


def draw_sets(pools: Sequence[np.ndarray], n: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    return np.stack([p[g.integers(0, len(p), n)] for p in pools], 1)


def dev(P: np.ndarray) -> np.ndarray:
    """|p − ½| averaged over the rows (steps) of a (steps, 384) percentile array."""
    return np.abs(np.atleast_2d(P) - 0.5).mean(0)


def rank_of(obs: float, null: np.ndarray) -> float:
    return float((null < obs).mean())


def member_rank(D: np.ndarray, heads: Sequence[int], seed: int, n: int = N_NULL) -> Dict:
    """T = mean of D over `heads`; ranked against layer-matched sets from the other heads."""
    T = float(D[list(heads)].mean())
    null = D[draw_sets(layer_pools(heads), n, seed)].mean(1)
    return {"T": T, "rank": rank_of(T, null), "null_q": np.quantile(null, [0.05, 0.5, 0.95, 0.99]).tolist()}


def ordered_pairs(heads: Sequence[int]) -> List[Tuple[int, int]]:
    """(a, b) positions in `heads` with a's layer below b's."""
    return [(a, b) for a in range(len(heads)) for b in range(len(heads))
            if heads[a] // N_HEADS < heads[b] // N_HEADS]


def comp_table(Wb: np.ndarray, Kb: np.ndarray, r: int = R_SUB) -> np.ndarray:
    """O[i, j] = ‖W_iᵀ K_j‖²_F / r, the K-composition overlap of every ordered head pair."""
    n, d = Wb.shape[:2]
    G = Wb[:, :, :r].transpose(0, 2, 1).reshape(n * r, d) @ Kb[:, :, :r].reshape(n, d, r).transpose(1, 0, 2).reshape(d, n * r)
    return (G.reshape(n, r, n, r) ** 2).sum((1, 3)) / r


def comp_mean(O: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """C6 per set (rows of head indices, the same layer in each column): mean overlap over the
    ordered pairs with the first head's layer below the second's."""
    pairs = ordered_pairs(list(sets[0]))
    return np.mean([O[sets[:, a], sets[:, b]] for a, b in pairs], 0)


def comp_rank(O: np.ndarray, heads: Sequence[int], seed: int, n: int = N_NULL) -> Dict:
    obs = float(comp_mean(O, np.array([heads]))[0])
    null = comp_mean(O, draw_sets(layer_pools(heads), n, seed))
    return {"C6": obs, "rank": rank_of(obs, null), "null_q": np.quantile(null, [0.05, 0.5, 0.95, 0.99]).tolist()}


def r90(curve: Sequence[Dict], frac: float = C7_FRAC) -> Optional[int]:
    """C7: the smallest D on the QK ladder with TV(D) ≤ frac · TV(0)."""
    tv0 = next(x["tv"] for x in curve if x["D"] == 0)
    hit = [x["D"] for x in curve if x["tv"] <= frac * tv0]
    return min(hit) if hit else None


# ---------------------------------------------------------------- the gate

def g0(Ws: Dict[int, List[Dict]], refs: Dict[int, Dict[str, np.ndarray]]) -> Dict:
    """(i) stored dense OV against W_O W_V, every head, steps 0 and 143000; (ii) the LN1 fold
    against B2a's V_h for its 14 heads at both steps; (iii) the low-rank columns against 2d's
    dense functions on those 14 heads at 143000."""
    from p2d_operator_activation.gradient_flow_condition import head_regime
    from trackB_circuit import b2a
    prov = json.loads(SERIES.read_text())["input_provenance"]
    out = {}
    for step, W in Ws.items():
        z = np.load(DATA / "phase12" / prov[f"ov_weights_step{step}"]
                    / f"ov_weights_pythia-410m-step{step}.npz")
        e_t, e_n = [], []
        for i, w in enumerate(W):
            l, h = divmod(i, N_HEADS)
            S = z[f"ov_head{h}_layer_{l}"]
            V = w["Wo"] @ w["Wv"]
            nv = np.linalg.norm(V)
            e_t.append(float(np.linalg.norm(S - V.T) / nv))
            e_n.append(float(np.linalg.norm(S - V) / nv))
        orient = "transposed" if np.median(e_t) < np.median(e_n) else "as_is"
        err = e_t if orient == "transposed" else e_n
        fold = 0.0
        for name in b2a.heads():
            A, B = v_factors(W[head_index(name)], "ln")
            ref = refs[step][name]
            fold = max(fold, float(np.linalg.norm(A @ B.T - ref) / np.linalg.norm(ref)))
        out[f"step{step}"] = {"orientation": orient, "max_rel": max(err), "worst": head_name(int(np.argmax(err))),
                              "other_orientation_min": min(e_n if orient == "transposed" else e_t),
                              "fold_max_rel": fold}
    dense = 0.0
    for name in b2a.heads():
        w = Ws[143000][head_index(name)]
        row = head_row(w)
        A, B = v_factors(w, "ln")
        V = A @ B.T
        hr = head_regime(w["Wq"].T, w["Wk"].T, V)
        S = 0.5 * (V + V.T)
        ev = np.linalg.eigvalsh(S)
        dense = max(dense, abs(row["C1"] - share(ev)), abs(row["C2"] - top_weight(ev)[0]),
                    abs(row["asym"] - hr["asymmetry"]), abs(row["C4"] - hr["align"]))
        if row["regime"] != hr["regime"]:
            dense = np.inf
    out["dense_max_abs"] = dense
    out["pass"] = (all(out[f"step{s}"]["max_rel"] <= G0_TOL and out[f"step{s}"]["fold_max_rel"] <= G0_FOLD_TOL
                       for s in Ws)
                   and len({out[f"step{s}"]["orientation"] for s in Ws}) == 1 and dense <= G0_DENSE_TOL)
    return out


def g1(pop0: Dict) -> Dict:
    """The step-0 band's calibration: halves 200 times, a half's rate outside the other's band."""
    g = np.random.default_rng(seed_for("b2-g1"))
    out = {}
    for c in COLS:
        x = pop0[c]
        rates = []
        for _ in range(N_DRAW):
            o = g.permutation(N_POP)
            a, b = x[o[:N_POP // 2]], x[o[N_POP // 2:]]
            lo, hi = np.quantile(a, [0.025, 0.975])
            rates.append(float(((b < lo) | (b > hi)).mean()))
        out[c] = {"rate": float(np.mean(rates)), "rate_q05_95": np.quantile(rates, [0.05, 0.95]).tolist(),
                  "band_step0": np.quantile(x, [0.025, 0.975]).tolist()}
        out[c]["pass"] = G1_BAND[0] <= out[c]["rate"] <= G1_BAND[1]
    out["C1_raw_band_step0"] = np.quantile(pop0["C1_raw"], [0.025, 0.975]).tolist()
    out["pass"] = all(out[c]["pass"] for c in COLS)
    return out


def fake_sets(layers: Sequence[int], n: int, seed: int) -> np.ndarray:
    """n sets of one non-member head from each of `layers`."""
    mem = {head_index(m) for m in MEMBERS}
    pools = [np.array([l * N_HEADS + h for h in range(N_HEADS) if l * N_HEADS + h not in mem])
             for l in layers]
    return draw_sets(pools, n, seed)


def g2(pops: Dict[int, Dict]) -> Dict:
    """The member statistic's false pass: 200 layer-matched non-member sets as members."""
    lay_m = [parse_head(m)[0] for m in MEMBERS]
    lay_c = [parse_head(m)[0] for m in CIRCUIT]
    out = {}
    for step, pop in pops.items():
        F = fake_sets(lay_m, N_DRAW, seed_for("b2-g2", step))
        Fc = fake_sets(lay_c, N_DRAW, seed_for("b2-g2c", step))
        row = {}
        for c in COLS:
            D = dev(percentiles(pop[c]))
            ranks = [member_rank(D, f, seed_for("b2-g2", step, c, k))["rank"] for k, f in enumerate(F)]
            row[c] = float(np.mean(np.array(ranks) >= RANK_PASS))
        O = comp_table(pop["W"], pop["K"])
        ranks = [comp_rank(O, list(f), seed_for("b2-g2c", step, k))["rank"]
                 for k, f in enumerate(Fc)]
        row["C6"] = float(np.mean(np.array(ranks) >= RANK_PASS))
        out[f"step{step}"] = row
    out["pass"] = all(v <= G2_MAX for s in pops for v in out[f"step{s}"].values())
    return out


def dosed(W: Dict, t: float) -> Dict:
    A, _ = v_factors(W, "ln")
    return {**W, "Wv": W["Wv"] + t * A.T}


def dose_to(W: Dict, target: float, m: Dict) -> Tuple[float, float]:
    """The smallest t (bisection) with a(V + t A Aᵀ) ≥ target; t = 0 if already there."""
    def a(t):
        return v_row(*v_factors(dosed(W, t), "ln"), m)["C1"]
    if a(0.0) >= target:
        return 0.0, a(0.0)
    A, B = v_factors(W, "ln")
    hi = np.linalg.norm(B) / np.linalg.norm(A)
    while a(hi) < target:
        hi *= 2
        if hi > 1e12:
            raise B2Error("G3: dose does not reach the target")
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if a(mid) < target else (lo, mid)
    return hi, a(hi)


def g3(W143: List[Dict], pop: Dict) -> Dict:
    """Planted attraction in trained weights at 143000, the dose curve beside."""
    lay_m = [parse_head(m)[0] for m in MEMBERS]
    F = fake_sets(lay_m, N_DRAW, seed_for("b2-g3"))
    heads = sorted(set(F.ravel().tolist()))
    ms = {i: m_row(W143[i]) for i in heads}
    out = {}
    for q in G3_DOSES:
        target = float(np.quantile(pop["C1"], q))
        planted = {i: dose_to(W143[i], target, ms[i])[1] for i in heads}
        ranks = []
        for k, f in enumerate(F):
            x = pop["C1"].copy()
            for i in f:
                x[i] = planted[i]
            ranks.append(member_rank(dev(percentiles(x)), f, seed_for("b2-g3", q, k))["rank"])
        out[f"q{q:.2f}"] = {"target": target, "pass_rate": float(np.mean(np.array(ranks) >= RANK_PASS)),
                            "rank_median": float(np.median(ranks))}
    out["pass"] = out[f"q{G3_DOSE:.2f}"]["pass_rate"] >= G3_PASS
    return out


def g3m_step(Ws: Sequence[Dict], F: np.ndarray, doses: Sequence[float] = G3M_DOSES) -> Dict:
    """One step's part of G3m: every head's unplanted C1 and, per dose, each fake-set head's C1
    dosed to this step's own quantile."""
    heads = sorted(set(F.ravel().tolist()))
    c1 = np.array([head_row(W)["C1"] for W in Ws])
    ms = {i: m_row(Ws[i]) for i in heads}
    planted = {q: np.array([dose_to(Ws[i], float(np.quantile(c1, q)), ms[i])[1] for i in heads])
               for q in doses}
    return {"C1": c1, "heads": np.array(heads), "planted": planted}


def g3m(parts: Dict[int, Dict], F: np.ndarray, doses: Sequence[float] = G3M_DOSES,
        late: int = G3M_LATE) -> Dict:
    """G3 on the rule's T (mean over `parts`' steps); the late-onset variant and the one-step
    slice at the last step (G3's seeds, so it must return unit 1's G3) beside."""
    steps = sorted(parts)
    pos = {int(h): k for k, h in enumerate(parts[steps[0]]["heads"])}
    variants = {"all": (steps, steps, "b2-g3m"), "late": (steps, [x for x in steps if x >= late], "b2-g3m-late"),
                f"one{steps[-1]}": ([steps[-1]], [steps[-1]], "b2-g3")}
    out = {"steps": steps, "late_from": late}
    for name, (read, on, tag) in variants.items():
        for q in doses:
            ranks = []
            for k, f in enumerate(F):
                P = []
                for st in read:
                    x = parts[st]["C1"].copy()
                    if st in on:
                        x[f] = parts[st]["planted"][q][[pos[int(i)] for i in f]]
                    P.append(percentiles(x))
                ranks.append(member_rank(dev(np.array(P)), f, seed_for(tag, q, k))["rank"])
            out[f"{name}_q{q:.2f}"] = {"pass_rate": float(np.mean(np.array(ranks) >= RANK_PASS)),
                                      "rank_median": float(np.median(ranks))}
    out["pass"] = out[f"all_q{G3_DOSE:.2f}"]["pass_rate"] >= G3_PASS
    return out


def g4(W143: List[Dict]) -> Dict:
    """A planted repulsive pair in one 143000 head's V: C2 < 0 and the flag set; ν = 0 returns
    the head's own values."""
    g = np.random.default_rng(seed_for("b2-g4"))
    mem = {head_index(m) for m in MEMBERS}
    i = int(g.choice([j for j in range(N_POP) if j not in mem]))
    W = W143[i]
    A, B = v_factors(W, "ln")
    m = m_row(W)
    ev = sym_eigs(A, B)
    Q, R = np.linalg.qr(A)
    u, _, _ = np.linalg.svd(R @ B.T, full_matrices=False)
    E = Q @ u[:, :2]                                    # V's top two write directions
    own = v_row(A, B, m)

    def planted(nu):
        return v_row(np.concatenate([A, -nu * E], 1), np.concatenate([B, E], 1), m)
    nu = G4_NU * float(np.abs(ev).max())
    p, z = planted(nu), planted(0.0)
    same = max(abs(z[k] - own[k]) for k in ("C1", "C2", "C4"))
    return {"head": head_name(i), "nu": nu, "C2": p["C2"], "pair": p["pair"], "nu0_max_abs": same,
            "pass": p["C2"] < 0 and p["pair"] and z["pair"] == own["pair"] and same <= 1e-12}


def plant_keys(W: Dict, dirs: np.ndarray) -> Dict:
    """W_K with its top-k right singular vectors set to the orthonormalised `dirs` (d × k)."""
    U, s, Vt = np.linalg.svd(W["Wk"], full_matrices=False)
    P, _ = np.linalg.qr(dirs)
    rest = Vt.T - P @ (P.T @ Vt.T)
    Qr, _ = np.linalg.qr(rest)
    basis = np.concatenate([P, Qr[:, :W["Wk"].shape[0] - P.shape[1]]], 1)
    return {**W, "Wk": U @ np.diag(s) @ basis.T}


def g5_one(W143: List[Dict], pop: Dict, O: np.ndarray, f: Sequence[int], seed, copy: int = G5_COPY) -> Dict:
    f = sorted(f, key=lambda i: i // N_HEADS)
    un = comp_rank(O, f, seed)
    O2 = O.copy()
    for b in range(len(f)):
        below = [a for a in range(len(f)) if f[a] // N_HEADS < f[b] // N_HEADS]
        if not below:
            continue
        dirs = np.concatenate([pop["W"][f[a]][:, :copy] for a in below], 1)
        Kb = bases(plant_keys(W143[f[b]], dirs))[1][:, :R_SUB]
        O2[:, f[b]] = ((np.einsum("ndi,dj->nij", pop["W"][:, :, :R_SUB], Kb)) ** 2).sum((1, 2)) / R_SUB
    pl = comp_rank(O2, f, seed)
    return {"heads": [head_name(i) for i in f], "planted_rank": pl["rank"], "planted_C6": pl["C6"],
            "unplanted_rank": un["rank"], "unplanted_C6": un["C6"]}


def g5(W143: List[Dict], pop: Dict, copy: int = G5_COPY) -> Dict:
    lay_c = [parse_head(m)[0] for m in CIRCUIT]
    F = fake_sets(lay_c, N_DRAW + 1, seed_for("b2-g5"))
    O = comp_table(pop["W"], pop["K"])
    prim = g5_one(W143, pop, O, list(F[0]), seed_for("b2-g5", 0), copy)
    besides = [g5_one(W143, pop, O, list(f), seed_for("b2-g5", k + 1), copy) for k, f in enumerate(F[1:])]
    return {"primary": prim,
            "draws_planted_pass": float(np.mean([b["planted_rank"] >= RANK_PASS for b in besides])),
            "draws_unplanted_central90": float(np.mean([0.05 <= b["unplanted_rank"] <= 0.95 for b in besides])),
            "pass": prim["planted_rank"] >= RANK_PASS and 0.05 <= prim["unplanted_rank"] <= 0.95}


def g6(W143: List[Dict]) -> Dict:
    """A planted gradient-flow head: W_K := W_Q, V := sym M_0."""
    from p2d_operator_activation.gradient_flow_condition import head_regime
    from p2d_operator_activation.table1_predictions import classify_ov_row
    g = np.random.default_rng(seed_for("b2-g6"))
    mem = {head_index(m) for m in MEMBERS}
    i = int(g.choice([j for j in range(N_POP) if j not in mem]))
    W = {**W143[i], "Wk": W143[i]["Wq"].copy()}
    m = m_row(W)
    Aq, Bk, _ = m["_m"]
    A, B = 0.5 * np.concatenate([Aq, Bk], 1), np.concatenate([Bk, Aq], 1)   # sym M_0 = A Bᵀ
    row = v_row(A, B, m)
    M = Aq @ Bk.T
    V = A @ B.T
    hr = head_regime(W["Wq"].T, W["Wk"].T, V)
    cr = classify_ov_row(V, M)
    ev = np.linalg.eigvalsh(0.5 * (V + V.T))[::-1]
    simple = (ev[0] - ev[1]) / ev[0] > 0.05
    expect = "row2_three_hyperplanes" if simple else "unclassified"   # λ₁ real > 0, φ₁ᵀMφ₁ = λ₁ > 0
    return {"head": head_name(i), "asym": m["asym"], "C4": row["C4"], "C1": row["C1"],
            "regime": hr["regime"], "row": cr["row"], "row_expected": expect,
            "pass": (m["asym"] <= G6_ASYM and row["C4"] >= G6_ALIGN and hr["regime"] == "gradient_flow"
                     and row["regime"] == "gradient_flow" and cr["row"] == expect)}


def c7_records() -> Dict[int, Path]:
    out = {143000: B2A / "2026-10-16/real.json"}
    for run in ("2026-10-16-occ", "2026-10-16-occ2"):
        for d in sorted((B2A / run).glob("step*")):
            if d.is_dir():
                out[int(d.name[4:])] = d / "real.json"
    return out


def g7() -> Dict:
    """C7's reader: the 14 heads' m2.qk at all 18 steps, inputs as named, TV at full rank exact."""
    from trackB_circuit import b2a
    recs = c7_records()
    bad, worst = [], 0.0
    if sorted(recs) != sorted(STEPS):
        bad.append(f"steps {sorted(recs)}")
    for step, p in recs.items():
        d = json.loads(p.read_text())
        run = p.parent.name if step == 143000 else p.parent.parent.name
        if d["code"] != G7_CODES[run] or d["prompts_hash"] != G7_PROMPTS or d["step"] != step:
            bad.append(f"{step}: code {d['code']} prompts {d['prompts_hash']}")
        for h in b2a.heads():
            qk = d["heads"].get(h, {}).get("m2", {}).get("qk")
            if not qk or qk[0]["D"] != 0:
                bad.append(f"{step}/{h}: no m2.qk")
                continue
            if qk[-1]["D"] != 2 * HEAD_DIM:
                bad.append(f"{step}/{h}: full rung D={qk[-1]['D']}")
            worst = max(worst, qk[-1]["tv"])
    return {"n_records": len(recs), "tv_full_max": worst, "problems": bad,
            "pass": not bad and worst <= G7_TV}


def constants() -> Dict:
    return {k: globals()[k] for k in ("R_SUB", "R_BESIDE", "PAIR_RATIO", "STAT_FROM", "N_NULL", "N_DRAW",
                                      "RANK_PASS", "C7_FRAC", "G0_TOL", "G1_BAND", "G2_MAX", "G3_DOSE",
                                      "G3_PASS", "G4_NU", "G5_COPY", "G6_ASYM", "G6_ALIGN", "G7_TV")}


def cmd_gate(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "gate.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    if rec and rec.get("code") != code_sha():
        raise SystemExit(f"refusing: {p} is from code {rec.get('code')}, not {code_sha()}")
    import torch
    rec.update(code=code_sha(), torch=torch.__version__, numpy=np.__version__, device="cpu",
               dtype="float64", constants=constants())
    rec.setdefault("arms", {})
    arms = a.arms or ARMS
    t0 = time.monotonic()
    need = {0} if set(arms) & {"G0", "G1"} else set()
    if set(arms) & {"G0", "G2", "G3", "G4", "G5", "G6"}:
        need |= {143000}
    if "G2" in arms:
        need |= {512}
    if "G0" in arms:
        need |= {0, 143000}
    loaded = {s: load_weights(s) for s in sorted(need)}
    Ws = {s: v[0] for s, v in loaded.items()}
    pops = {s: population(Ws[s]) for s in Ws if s in (0, 512, 143000)}
    print(f"weights and populations: {time.monotonic() - t0:.0f}s", flush=True)
    for s, pop in pops.items():
        np.savez(out / f"population_step{s}.npz", **{k: v for k, v in pop.items() if k not in ("W", "K")})
    for arm in arms:
        t1 = time.monotonic()
        if arm == "G0":
            r = g0({s: Ws[s] for s in (0, 143000)}, {s: loaded[s][1] for s in (0, 143000)})
        elif arm == "G1":
            r = g1(pops[0])
        elif arm == "G2":
            r = g2({s: pops[s] for s in (512, 143000)})
        elif arm == "G3":
            r = g3(Ws[143000], pops[143000])
        elif arm == "G4":
            r = g4(Ws[143000])
        elif arm == "G5":
            r = g5(Ws[143000], pops[143000])
        elif arm == "G6":
            r = g6(Ws[143000])
        else:
            r = g7()
        r["seconds"] = time.monotonic() - t1
        rec["arms"][arm] = r
        p.write_text(json.dumps(rec, indent=1, default=float))
        print(f"{arm}: {'pass' if r['pass'] else 'FAIL'} ({r['seconds']:.0f}s)", flush=True)
    rec["pass"] = all(rec["arms"].get(x, {}).get("pass") for x in ARMS)
    p.write_text(json.dumps(rec, indent=1, default=float))
    return 0 if rec["pass"] else 1


def cmd_regate(a) -> int:
    """Blocked 62 (a): G3m over the 9 steps ≥ 512, G5's dose curve at 143000."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "regate.json"
    if p.exists():
        raise SystemExit(f"refusing: {p} exists")
    import torch
    rec = {"code": code_sha(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
           "dtype": "float64", "constants": {**constants(), "G3M_DOSES": G3M_DOSES, "G3M_LATE": G3M_LATE,
                                              "G5_COPIES": G5_COPIES, "STAT_STEPS": STAT_STEPS}, "arms": {}}
    F = fake_sets([parse_head(m)[0] for m in MEMBERS], N_DRAW, seed_for("b2-g3"))
    parts = {}
    for st in STAT_STEPS:
        t0 = time.monotonic()
        Ws, _ = load_weights(st)
        parts[st] = g3m_step(Ws, F)
        np.savez(out / f"g3m_step{st}.npz", C1=parts[st]["C1"], heads=parts[st]["heads"],
                 **{f"planted_q{q:.2f}": v for q, v in parts[st]["planted"].items()})
        print(f"step {st}: {time.monotonic() - t0:.0f}s", flush=True)
        if st == 143000:
            t1 = time.monotonic()
            pop = population(Ws)
            rec["arms"]["G5c"] = {f"copy{c}": g5(Ws, pop, c) for c in G5_COPIES}
            rec["arms"]["G5c"]["seconds"] = time.monotonic() - t1
            p.write_text(json.dumps(rec, indent=1, default=float))
            print(f"G5c ({rec['arms']['G5c']['seconds']:.0f}s)", flush=True)
        del Ws
    t1 = time.monotonic()
    r = g3m(parts, F)
    r["seconds"] = time.monotonic() - t1
    rec["arms"]["G3m"] = r
    rec["pass"] = r["pass"]
    p.write_text(json.dumps(rec, indent=1, default=float))
    print(f"G3m: {'pass' if r['pass'] else 'FAIL'}", flush=True)
    return 0 if r["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trackB_circuit.b2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--out", required=True)
    g.add_argument("--arms", nargs="+", choices=ARMS)
    r = sub.add_parser("regate")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_gate(a) if a.cmd == "gate" else cmd_regate(a)


if __name__ == "__main__":
    raise SystemExit(main())
