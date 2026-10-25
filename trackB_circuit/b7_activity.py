"""B7, routing activity across depth, unit 1 (`trackB_circuit/design-B7.md`, frozen 2026-10-25,
Blocked 110 (a)): per-head partner persistence from stored attention against N1 (shuffled layer)
and N0 (offset profile), the cell descriptions, the beside columns, and the gate G0–G9.

**Rows** (design "Objects"): query i ≥ 16 of each stored `attentions.npz` (A1a's `RUNS`, the 7
scored v1 prompts), position 0 dropped as key, the row renormalised over keys 1..i; rows with
``w_i = 1 − A_i0 < 0.05`` are out (B3's ``W_MIN``). A row is condensed when its top share > 0.5;
its partner is that key and its offset i − π(i).

**P** for (l, h) against layer l′: the best-matched head h* of l′ (most agreeing rows) is chosen
leave-one-prompt-out, P is pooled over the 7 folds; a cell is the entering heads' agreeing rows
over their jointly condensed rows (the weighted mean). **N1** is that statistic between l and
every l′ with |l′ − l| ≥ 2 that is read, averaged. **N0** (unit 1b, Blocked 111 (a)) is the mean
over the read rows of c_i, the share of ordered pairs (j, j′) of *different* jointly condensed rows
of i's 32-position bin (from 16) over the 7 prompts, with j's offset in (l, h) and j′'s in
(l′, h*) both valid at i, whose offsets agree: (Σ_k p(k) q(k) − #rows agreeing with themselves) /
(C_p C_q − #rows with both offsets valid), sums over offsets ≤ i − 1. Unit 1's c_i (Σ p q over the
same rows, self-pairs included) rose with persistence by about P/m on m rows a bin (`/challenge-pr`
on #272, finding 1). A row with no such pair is left out of N0 and counted.

**Choices the design left open, made before any output:**
(1) The offsets valid at i are 0..i − 1 (keys 1..i). The design's "1..i − 1" counted from key 1 of
a sink-free row, where a self partner (offset 0) is a valid key; check 6 drew partners from
1..i − 2, so it is unchanged. (2) A head enters on the sample's own count (unit 1c: choice 18; until then ≥ 50 jointly
condensed rows with its h* over the 7 folds), and the entering set is fixed in every bootstrap draw; a
draw that cannot choose h* for some fold (all its prompts one prompt; P about 1e-5) is dropped
and counted. (3) In a draw, h* is chosen on the drawn copies of the other prompts (every copy of
the fold's own prompt left out); N0's profiles are the sample's in every draw (the design fixes
them "over all 7 prompts"), so its interval holds the read rows' and h*'s variation, not the
profiles'. (4) h* is argmax of the agreeing-row count, ties to the lowest index. (5) N1 averages
the distant layers whose own cell would be read (≥ 2 entering heads); with none, the cell is
"not read" and says why. (6) The two-cycle candidate reads `P_2 − N0` with N0 of the lag-2 pairs.
(7) Neighbour persistence and `D_ℓ` use state indices 0..23 (index l is what attention layer l
reads) for neighbours, and 0..24 for `D_1`, `D_2`; neighbours are read on rows i ≥ 16 among
tokens 1..i − 1; `D` pools every token from position 1 over the 7 prompts. (8) The pooled row
weights a head by w_i where w_i ≥ 0.05 and is out where no head is in. (9) G7's thinning keeps,
for every head pair of every layer pair, its 50 jointly condensed rows of lowest random priority
(one priority per row and layer); G7 counts a read cell described as anything but "rerouting",
and reports unread cells beside. (10) G8 records differences from the unplanted run only, never
its values.

**Unit 1b (Blocked 111 (a), the user's; `/challenge-pr` on #272, findings 1–6):** (11) N0 over
pairs of different rows, above. (12) A cell is "not read", with the reason, when its sample N0 or
the interval it is described by is not finite, or when more than NONFINITE_MAX of that interval's
bootstrap draws are not finite (each interval's non-finite draws are counted). (13) G1 and G3 also
run at the minimum entry (`thin` = MIN_JOINT); G3's band there is ± G3_Z standard errors of a
binomial r = 0.8 on the cell's jointly condensed rows (4: about 0.4 % that one of 69 cells fails
by chance), since ± 0.03 is about 2 standard errors at 800 rows. (14) G2 passes on the pooled
deviation |Σ P − Σ N0| / Σ N0 ≤ 0.25 over a sink step's cells (rows weighted); the per-cell
extreme is beside (finding 3). (15) G8 prints the other cells' largest N1 change and the layers whose
description changed, never the descriptions (choice 10; N1 reads layers 10–11, so it may move);
their |ΔP| = 0 is a code check (separable by construction), not evidence. (16) The gate records each input file's sha256, the torch and numpy
versions and the device; `read` refuses unless the gate's code, `prompts_hash`, `RUNS` and every
file hash it reads match. (17) NaN is written to JSON as null.

**Unit 1c (Blocked 112 (a), the user's; `/challenge-pr` on #273, findings 2 and 4):** (18) A head
enters on its jointly condensed rows with the h* of all 7 prompts (≥ MIN_JOINT), used for entry
only; P, N0, N1, the extras and every interval stay leave-one-out, so a head's weight in P is its
leave-one-out rows, which can be fewer than MIN_JOINT. Each head's leave-one-out rows and entry
rows and h* are printed, and a cell's minimum and median over its entering heads beside. Replaces
choice 2's count (its fixed entering set in every draw stands). (19) G3 prints the pooled r
(rows-weighted Σ(P − N0) / Σ(1 − N0) over a run's read cells) beside.

NUMERICS: stored attention (float32) cast to float64 on the GPU; offset histograms float32 (exact
integer counts), their ratios and every statistic float64. CPU allowed (argmax and counting).

Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b7_activity gate --out <dir>
    python -m trackB_circuit.b7_activity read --out <dir> --gate <dir>/gate.json [--steps 512,143000]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import RUNS, V1, prompts_hash, run_dir
from trackB_circuit.b2a import N_HEADS, N_LAYERS, REPO, STEPS, seed_for

DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
B3_ROWS = DATA / "b3/2026-10-18-unit1"
B3_ROWS_CODE = "2c13479d759c"
PROMPTS_HASH = "976b62d677bc"
ROW_FROM = 16
W_MIN = 0.05                               # B3's
TOP = 0.5                                  # placed (design): condensed
MIN_JOINT = 50                             # placed (design): a head enters
MIN_HEADS = 2                              # placed (design): a cell is read
POS_N0 = 0.5                               # placed (design): positional, not separable
BIN = 32                                   # placed (design): N0's position bins
N_BOOT = 2000                              # placed (design)
PLANT = 0.8                                # design (gate): a planted row's partner mass
SINK_STEPS = (512, 8000, 143000)           # design (gate): real sink masses
G0_TOL = 1e-5
NONFINITE_MAX = 0.05                       # placed (unit 1b): an interval's non-finite draws
G2_MAX = 0.25                              # design (gate): G2's relative band, pooled (unit 1b)
G3_R, G3_TOL, G3_Z = 0.8, 0.03, 4.0        # design (gate); G3_Z placed (unit 1b): the minimum entry
G7_SEEDS, G7_MAX = 20, 0.05
G8_STEP, G8_LAYERS, G8_HEADS = 143000, (10, 11), (0, 1, 2, 3)
DESCRIPTIONS = ("not read", "positional, not separable", "rerouting", "depth-local persistence",
                "frozen across depth", "persistent, less than at distant layers")
ARMS = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9")


class B7Error(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackB_circuit/b7_activity.py").read_bytes()).hexdigest()[:12]


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------- stage 1: one prompt

def partners(A, top: float = TOP, w_min: float = W_MIN, row_from: int = ROW_FROM):
    """Condensed rows of a stack ``A`` (…, n, n), torch float64: the partner key (1..i) where
    w = 1 − A_i0 ≥ `w_min` and the sink-free row's top share > `top`, else −1; and the in-mask."""
    R = A[..., row_from:, :]
    K = R[..., 1:]
    top_v, arg = K.max(-1)
    inn = (1 - R[..., 0]) >= w_min
    cond = inn & (top_v > top * K.sum(-1))
    return _torch().where(cond, arg + 1, -1), inn


def pooled_partners(A, top: float = TOP, w_min: float = W_MIN, row_from: int = ROW_FROM):
    """The layer's row (beside): the in-heads' Ã weighted by w_i, renormalised; (L, R) as `partners`."""
    torch = _torch()
    parts, ins = [], []
    for l in range(A.shape[0]):
        R = A[l, :, row_from:, :]
        K = R[..., 1:]
        S = K.sum(-1)
        w = 1 - R[..., 0]
        wt = torch.where(w >= w_min, w, 0.0)
        tot = wt.sum(0)
        row = (K * (wt / torch.where(S > 0, S, 1.0))[..., None]).sum(0) / torch.where(tot > 0, tot, 1.0)[:, None]
        top_v, arg = row.max(-1)
        inn = tot > 0
        cond = inn & (top_v > top)
        parts.append(torch.where(cond, arg + 1, -1))
        ins.append(inn)
    return torch.stack(parts), torch.stack(ins)


def soft_sums(A, part, row_from: int = ROW_FROM):
    """Σ over jointly condensed rows of 1 − ½‖Ã_{l,h,i} − Ã_{l+1,g,i}‖₁, (L − 1, H, H) float64."""
    torch = _torch()
    L, H = A.shape[:2]
    out = torch.zeros((L - 1, H, H), dtype=torch.float64, device=A.device)

    def norm(l):
        K = A[l, :, row_from:, 1:]
        S = K.sum(-1, keepdim=True)
        return K / torch.where(S > 0, S, 1.0)

    nxt = norm(0)
    for l in range(L - 1):
        cur, nxt = nxt, norm(l + 1)
        c1 = part[l + 1] >= 0
        for h in range(H):
            ov = 1 - 0.5 * (cur[h][None] - nxt).abs().sum(-1)
            m = (part[l, h] >= 0)[None] & c1
            out[l, h] = (ov * m).sum(-1)
    return out


def neighbours(X, row_from: int = ROW_FROM):
    """Each row i ≥ `row_from`'s nearest earlier token by cosine among 1..i − 1, per state index;
    ``X`` (Lx, n, d) torch float64. Returns (Lx, R) token indices."""
    torch = _torch()
    X = X / X.norm(dim=-1, keepdim=True)
    n = X.shape[1]
    i = torch.arange(row_from, n, device=X.device)
    c = torch.arange(n - 1, device=X.device)
    ok = c[None, :] <= (i - 2)[:, None]               # key c + 1 ≤ i − 1
    out = []
    for l in range(X.shape[0]):
        G = X[l, row_from:] @ X[l, 1:].T
        out.append(torch.where(ok, G, -torch.inf).argmax(-1) + 1)
    return torch.stack(out)


def d_sums(X) -> Dict[str, np.ndarray]:
    """Σ_i ‖u_i(l + k) − u_i(l)‖² over tokens 1..n − 1 for k = 1, 2, and the token count."""
    U = X[:, 1:]
    return {"s1": ((U[1:] - U[:-1]) ** 2).sum((-1, -2)).cpu().numpy(),
            "s2": ((U[2:] - U[:-2]) ** 2).sum((-1, -2)).cpu().numpy(), "n": int(U.shape[1])}


def d_of(ds: Sequence[Dict]) -> Dict[str, List[float]]:
    n = sum(d["n"] for d in ds)
    return {k: np.sqrt(sum(d[f"s{k[-1]}"] for d in ds) / n).tolist() for k in ("D1", "D2")}


def prompt_stage(A, X=None, soft: bool = False, row_from: int = ROW_FROM) -> Dict:
    """Stage 1 for one prompt: partners, in-masks, the pooled row's, optionally soft sums, and from
    the states (if given) neighbours and D sums."""
    part, inn = partners(A, row_from=row_from)
    out = {"n": int(A.shape[-1]), "part": part, "in": inn}
    out["ppart"], out["pin"] = pooled_partners(A, row_from=row_from)
    if soft:
        out["soft"] = soft_sums(A, part, row_from)
    if X is not None:
        out["nb"] = neighbours(X[: A.shape[0]], row_from)
        out["d"] = d_sums(X)
    return out


# ---------------------------------------------------------------- stage 2: every layer pair

def _cat(stages: Sequence[Dict], key: str, row_from: int = ROW_FROM):
    torch = _torch()
    dev = stages[0][key].device
    x = torch.cat([s[key] for s in stages], -1)
    pos = torch.cat([torch.arange(row_from, s["n"], device=dev) for s in stages])
    pid = torch.cat([torch.full((s["n"] - row_from,), q, device=dev) for q, s in enumerate(stages)])
    return x, pos, pid


def n0_sums(pa, pb, J, pos, pid, Q: int, bin_: int = BIN):
    """N0's row sums (design N0 as revised, Blocked 111 (a)): per prompt, Σ c_i over the jointly
    condensed rows where c_i is defined, and the count of those rows, each (Q, H, G) float64.
    c_i is over ordered pairs of different jointly condensed rows of i's bin, offsets ≤ i − 1.
    ``pa`` (H, R), ``pb`` (G, R) partner keys; ``J`` (H, G, R) the joint mask (possibly thinned)."""
    torch = _torch()
    H, G = pa.shape[0], pb.shape[0]
    out = torch.zeros((H, G, Q), dtype=torch.float64, device=pa.device)
    rows = torch.zeros_like(out)
    oa = torch.where(pa >= 0, pos - pa, 0)
    ob = torch.where(pb >= 0, pos - pb, 0)
    top = int(pos.max()) + 1
    for b0 in range(int(pos.min()), top, bin_):
        r = ((pos >= b0) & (pos < b0 + bin_)).nonzero().squeeze(-1)
        if r.numel() == 0:
            continue
        Jr = J[..., r]
        if not bool(Jr.any()):
            continue
        Jb = Jr.to(torch.float32)
        Kb = b0 + bin_ - 1
        OA = torch.zeros((H, r.numel(), Kb), dtype=torch.float32, device=pa.device)
        OB = torch.zeros((G, r.numel(), Kb), dtype=torch.float32, device=pa.device)
        OA.scatter_(-1, oa[:, r, None], 1.0)
        OB.scatter_(-1, ob[:, r, None], 1.0)
        p = torch.einsum("hgr,hrk->hgk", Jb, OA).to(torch.float64)
        q = torch.einsum("hgr,grk->hgk", Jb, OB).to(torch.float64)
        idx = (pos[r] - 1).expand(H, G, -1)                # offsets 0..i − 1 valid at i
        Cpq = torch.cumsum(p * q, -1).gather(-1, idx)
        Cp = torch.cumsum(p, -1).gather(-1, idx)
        Cq = torch.cumsum(q, -1).gather(-1, idx)
        # self-pairs: row j valid at i when both its offsets are ≤ i − 1 (V[h, g, i, j])
        mx = torch.maximum(oa[:, None, r], ob[None, :, r])                   # (H, G, r)
        V = (mx[..., None, :] <= (pos[r] - 1)[:, None]) & Jr[..., None, :]
        same = (oa[:, None, r] == ob[None, :, r])[..., None, :]
        num = Cpq - (V & same).sum(-1).to(torch.float64)
        den = Cp * Cq - V.sum(-1).to(torch.float64)
        ok = Jr & (den > 0)
        c = torch.where(ok, num / torch.where(den > 0, den, 1.0), 0.0)
        onehot = torch.nn.functional.one_hot(pid[r], Q).to(torch.float64)
        out += c @ onehot
        rows += ok.to(torch.float64) @ onehot
    return out.permute(2, 0, 1), rows.permute(2, 0, 1)


def pair_counts(stages: Sequence[Dict], lags: Sequence[int] = (1, 2), thin: Optional[int] = None,
                seed: int = 0, key: str = "part") -> Dict:
    """Per-prompt agreeing and jointly condensed row counts for every ordered layer pair,
    (L, L, Q, H, H), and N0's row sums and defined rows for the pairs (l, l + lag). With `thin`, each head pair
    of each layer pair keeps only its `thin` jointly condensed rows of lowest random priority."""
    torch = _torch()
    part, pos, pid = _cat(stages, key)
    L, H, R = part.shape
    Q = len(stages)
    dev = part.device
    onehot = torch.nn.functional.one_hot(pid, Q).to(torch.float64)
    cond = part >= 0
    agree = torch.zeros((L, L, Q, H, H), dtype=torch.float64, device=dev)
    joint = torch.zeros_like(agree)
    n0 = {}
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for l in range(L):
        J = cond[l][:, None, None, :] & cond[None, :, :, :]               # (H, L, H, R)
        if thin is not None:
            order = torch.randperm(R, generator=gen).to(dev)
            Js = J[..., order]
            keep = torch.cumsum(Js.to(torch.int32), -1) <= thin
            J = torch.zeros_like(J)
            J[..., order] = Js & keep
        E = J & (part[l][:, None, None, :] == part[None, :, :, :])
        agree[l] = torch.einsum("hlgr,rq->lqhg", E.to(torch.float64), onehot)
        joint[l] = torch.einsum("hlgr,rq->lqhg", J.to(torch.float64), onehot)
        for lag in lags:
            if l + lag < L:
                n0[(l, l + lag)] = n0_sums(part[l], part[l + lag], J[:, l + lag], pos, pid, Q)
    return {"agree": agree, "joint": joint, "n0": n0, "Q": Q, "L": L, "H": H}


def flat_counts(stages: Sequence[Dict], key: str) -> Dict:
    """Agreeing and jointly valid row counts for (L, R) partner arrays (pooled rows or
    neighbours): (L, L, Q), where a neighbour is always valid."""
    torch = _torch()
    x, pos, pid = _cat(stages, key)
    Q = len(stages)
    onehot = torch.nn.functional.one_hot(pid, Q).to(torch.float64)
    c = x >= 0
    J = c[:, None] & c[None]
    E = J & (x[:, None] == x[None])
    return {"agree": E.to(torch.float64) @ onehot, "joint": J.to(torch.float64) @ onehot}


# ---------------------------------------------------------------- stage 3: matching and cells

def draws(Q: int, n: int, seed: int, device) -> "torch.Tensor":
    """(n + 1, Q) prompt multiplicities; row 0 is the sample itself."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    M = np.vstack([np.ones((1, Q)), rng.multinomial(Q, np.full(Q, 1 / Q), size=n)])
    return torch.as_tensor(M, dtype=torch.float64, device=device)


def matched(agree, joint, M, extras: Optional[Dict] = None, min_joint: int = MIN_JOINT) -> Dict:
    """Leave-one-prompt-out best match of each head of one layer among the other's, for one
    ordered layer pair: ``agree``, ``joint``, ``extras[k]`` (Q, H, G); ``M`` (D, Q). Returns the
    cell's per-draw P and extras (sums over entering heads / their joint rows), the entering
    count, and per head the sample's P, joint rows and h* per fold, and the entry's h* and rows.
    A head enters on its joint rows with h* chosen on all Q prompts of the sample (choice 18);
    P, the extras and their draws stay leave-one-out."""
    torch = _torch()
    extras = extras or {}
    D, Q = M.shape
    S_all = torch.einsum("dq,qhg->dhg", M, agree)
    S_out = S_all[:, None] - M[:, :, None, None] * agree[None]              # (D, Q, H, G)
    hs = S_out.argmax(-1)                                                    # first max: lowest g
    others = M.sum(1, keepdim=True) - M                                      # (D, Q)
    bad = ((M > 0) & (others == 0)).any(1)
    idx = hs[..., None]

    def take(x):
        v = x[None].expand(D, -1, -1, -1).gather(-1, idx).squeeze(-1)        # (D, Q, H)
        return (M[:, :, None] * v).sum(1)                                    # (D, H)

    Ah, Jh = take(agree), take(joint)
    he = S_all[0].argmax(-1)                                                 # (H,) entry's h*
    Je = joint.sum(0).gather(-1, he[:, None]).squeeze(-1)                    # (H,) its rows
    enter = Je >= min_joint
    den = Jh[:, enter].sum(1)
    nan = torch.full_like(den, torch.nan)
    ok = (den > 0) & ~bad

    def cell(x):
        return torch.where(ok, x[:, enter].sum(1) / torch.where(den > 0, den, 1.0), nan).cpu().numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        headP = (Ah[0] / Jh[0]).cpu().numpy()
    return {"P": cell(Ah), **{k: cell(take(v)) for k, v in extras.items()},
            "n_enter": int(enter.sum()), "dropped": int(bad[1:].sum()), "enter": enter.cpu().numpy(),
            "head_P": headP, "head_joint": Jh[0].cpu().numpy(), "hstar": hs[0].cpu().numpy(),
            "head_joint_entry": Je.cpu().numpy(), "hstar_entry": he.cpu().numpy()}


def interval(x: np.ndarray) -> Tuple[float, float]:
    """Percentile 2.5–97.5 % over the finite draws (rows 1..)."""
    v = x[1:][np.isfinite(x[1:])]
    if v.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return (float(lo), float(hi))


def nonfinite(x: np.ndarray) -> float:
    """The share of bootstrap draws (rows 1..) that are not finite."""
    return float(1 - np.isfinite(x[1:]).mean())


def describe(n_enter: int, n0: float, d0: Tuple[float, float], d1: Tuple[float, float],
             n1_read: bool = True, bad0: float = 0.0, bad1: float = 0.0) -> Tuple[str, Optional[str]]:
    """The design's ordered partition: the first description that applies, and why a cell is not
    read (choice 12: a non-finite value or interval is never read as a description)."""
    if n_enter < MIN_HEADS:
        return DESCRIPTIONS[0], f"{n_enter} entering heads"
    if not np.isfinite(n0):
        return DESCRIPTIONS[0], "N0 not finite"
    if n0 >= POS_N0:
        return DESCRIPTIONS[1], None
    if not np.isfinite(d0).all() or bad0 > NONFINITE_MAX:
        return DESCRIPTIONS[0], f"P - N0: {bad0:.3f} of draws not finite"
    if not d0[0] > 0:
        return DESCRIPTIONS[2], None
    if not n1_read:
        return DESCRIPTIONS[0], "no distant layer read"
    if not np.isfinite(d1).all() or bad1 > NONFINITE_MAX:
        return DESCRIPTIONS[0], f"P - N1: {bad1:.3f} of draws not finite"
    if d1[0] > 0:
        return DESCRIPTIONS[3], None
    if d1[1] >= 0:
        return DESCRIPTIONS[4], None
    return DESCRIPTIONS[5], None


def ratio(agree, joint, M) -> np.ndarray:
    """Σ m agree / Σ m joint per draw for (Q,) sums (pooled rows, neighbours)."""
    a, j = M @ agree, M @ joint
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a / j).cpu().numpy()


def cells(cnt: Dict, M, flat: Optional[Dict] = None) -> List[Dict]:
    """The cell table for one step: per layer pair l → l + 1, P, N0, N1, intervals, description,
    the lag-2 columns and per-head values; `flat` adds the pooled and neighbour columns."""
    L = cnt["L"]
    ag, jt, n0 = cnt["agree"], cnt["joint"], cnt["n0"]

    def with_n0(a, b):
        m = matched(ag[a, b], jt[a, b], M, {"N0s": n0[(a, b)][0], "N0r": n0[(a, b)][1]})
        with np.errstate(invalid="ignore", divide="ignore"):
            m["N0"] = m["N0s"] / m["N0r"]                  # Σ c_i / defined rows, per draw
        return m

    out = []
    for l in range(L - 1):
        m1 = with_n0(l, l + 1)
        far = []
        for l2 in range(L):
            if abs(l2 - l) >= 2:
                m = matched(ag[l, l2], jt[l, l2], M)
                if m["n_enter"] >= MIN_HEADS:
                    far.append(m["P"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            N1 = np.nanmean(np.stack(far), 0) if far else np.full_like(m1["P"], np.nan)
        P, N0 = m1["P"], m1["N0"]
        d0, d1 = interval(P - N0), interval(P - N1)
        bad0, bad1 = nonfinite(P - N0), nonfinite(P - N1)
        desc, why = describe(m1["n_enter"], float(N0[0]), d0, d1, bool(far), bad0, bad1)
        joint_rows = float(m1["head_joint"][m1["enter"]].sum())
        loo = m1["head_joint"][m1["enter"]]
        c = {"l": l, "n_enter": m1["n_enter"], "n_far": len(far), "dropped": m1["dropped"],
             "joint_rows": joint_rows, "description": desc, "not_read_because": why,
             "loo_rows_min": float(loo.min()) if loo.size else None,          # beside (choice 18)
             "loo_rows_median": float(np.median(loo)) if loo.size else None,
             "nonfinite_draws": {"P_minus_N0": bad0, "P_minus_N1": bad1}}
        if m1["n_enter"] >= MIN_HEADS:
            c.update({"P": float(P[0]), "N0": float(N0[0]), "N1": float(N1[0]), "P_ci": interval(P),
                      "P_minus_N0_ci": d0, "P_minus_N1_ci": d1,
                      "n0_rows_undefined": joint_rows * (1 - float(m1["N0r"][0]))})
        c["heads"] = {"P": m1["head_P"].tolist(), "joint": m1["head_joint"].tolist(),
                      "enter": m1["enter"].tolist(), "hstar": m1["hstar"].tolist(),
                      "joint_entry": m1["head_joint_entry"].tolist(), "hstar_entry": m1["hstar_entry"].tolist()}
        if l + 2 < L:
            m2 = with_n0(l, l + 2)
            if m2["n_enter"] >= MIN_HEADS:
                x2 = m2["P"] - m2["N0"]
                d2 = interval(x2)
                c.update({"P2": float(m2["P"][0]), "N0_2": float(m2["N0"][0]), "P2_minus_N0_2_ci": d2,
                          "nonfinite_draws_2": nonfinite(x2)})
                c["two_cycle"] = bool(c["description"] == DESCRIPTIONS[2] and d2[0] > 0
                                      and nonfinite(x2) <= NONFINITE_MAX)
        if "soft" in cnt and m1["n_enter"] >= MIN_HEADS:
            s = matched(ag[l, l + 1], jt[l, l + 1], M[:1], {"soft": cnt["soft"][:, l]})
            c["soft"] = float(s["soft"][0])
        if flat is not None:
            for name, f in flat.items():
                v = ratio(f["agree"][l, l + 1], f["joint"][l, l + 1], M)
                c[f"{name}_P"], c[f"{name}_P_ci"] = float(v[0]), interval(v)
                if name == "nb":
                    fr = [ratio(f["agree"][l, l2], f["joint"][l, l2], M) for l2 in range(f["agree"].shape[0])
                          if abs(l2 - l) >= 2]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        n1 = np.nanmean(np.stack(fr), 0)
                    c["nb_N1"], c["nb_P_minus_N1_ci"] = float(n1[0]), interval(v - n1)
        out.append(c)
    return out


def condensed_share(stages: Sequence[Dict]) -> Dict:
    """Per head and per layer: condensed rows / rows in; the pooled row's per layer."""
    cond = sum((s["part"] >= 0).sum(-1).cpu().numpy() for s in stages)
    inn = sum(s["in"].sum(-1).cpu().numpy() for s in stages)
    pc = sum((s["ppart"] >= 0).sum(-1).cpu().numpy() for s in stages)
    pi = sum(s["pin"].sum(-1).cpu().numpy() for s in stages)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"head": (cond / inn).tolist(), "layer": (cond.sum(1) / inn.sum(1)).tolist(),
                "pooled": (pc / pi).tolist(), "head_rows_in": inn.tolist()}


def run_step(stages: Sequence[Dict], seed: int, n_boot: int = N_BOOT, thin: Optional[int] = None,
             beside: bool = False) -> Dict:
    """Stages 2 and 3 for one step's prompts."""
    cnt = pair_counts(stages, thin=thin, seed=seed)
    M = draws(cnt["Q"], n_boot, seed, cnt["agree"].device)
    flat = None
    if beside:
        torch = _torch()
        cnt["soft"] = torch.stack([s["soft"] for s in stages])          # (Q, L − 1, H, H)
        flat = {"pooled": flat_counts(stages, "ppart"), "nb": flat_counts(stages, "nb")}
    return {"cells": cells(cnt, M, flat), "share": condensed_share(stages)}


# ---------------------------------------------------------------- inputs

def _load(path: Path, key: str, device: str, hashes: Optional[Dict[str, str]] = None):
    """One stored array as torch float64, read once; its file's sha256 goes into `hashes`."""
    torch = _torch()
    b = path.read_bytes()
    if hashes is not None:
        key_ = str(path.relative_to(DATA)) if path.is_relative_to(DATA) else str(path)
        hashes[key_] = hashlib.sha256(b).hexdigest()
    with np.load(io.BytesIO(b)) as z:
        a = z[key]
    return torch.from_numpy(a).to(device=device, dtype=torch.float64)


def load_attn(path: Path, device: str, hashes: Optional[Dict[str, str]] = None):
    return _load(path, "attentions", device, hashes)


def load_states(path: Path, device: str, hashes: Optional[Dict[str, str]] = None):
    return _load(path, "activations", device, hashes)


def clean(x):
    """JSON-safe: NaN and ±inf become null (choice 17), numpy scalars and tuples plain."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def dump(path: Path, rec: Dict) -> None:
    path.write_text(json.dumps(clean(rec), indent=1, allow_nan=False) + "\n")


def alpha_beside(step: int) -> Dict:
    """B3's per-head α (weighted) from its stored rows and the layer median, code hash checked."""
    from trackB_circuit.b3_route import alpha_of, load_rows
    rows, rec = load_rows(B3_ROWS, step)
    if rec["code"] != B3_ROWS_CODE:
        raise B7Error(f"B3 rows step {step}: code {rec['code']}, want {B3_ROWS_CODE}")
    a = alpha_of(rows).reshape(N_LAYERS, N_HEADS)
    with np.errstate(invalid="ignore"):
        return {"head": a.tolist(), "layer_median": np.nanmedian(a, 1).tolist()}


# ---------------------------------------------------------------- the gate

def plant(sink, part, n: int, row_from: int = ROW_FROM):
    """A planted stack (L, H, n, n) float64: rows i ≥ `row_from` keep `sink` (L, H, R) as A_i0,
    put PLANT of the rest on `part` (key 1..i; −1 spreads it all) and the remainder uniformly over
    keys 1..i; rows below `row_from` uniform over keys 0..i."""
    torch = _torch()
    L, H, R = sink.shape
    dev = sink.device
    i = torch.arange(n, device=dev)
    tri = (i[None, :] <= i[:, None]).to(torch.float64)
    A = (tri / (i + 1)[:, None].to(torch.float64)).expand(L, H, n, n).clone()
    ii = torch.arange(row_from, n, device=dev).to(torch.float64)
    w = 1 - sink
    keys = (i[None, :] >= 1) & (i[None, :] <= i[row_from:, None])            # (R, n)
    has = part >= 0
    spread = torch.where(has, w * (1 - PLANT), w) / ii
    rows = spread[..., None] * keys.to(torch.float64)
    rows[..., 0] = sink
    rows.scatter_add_(-1, part.clamp(min=0)[..., None], torch.where(has, w * PLANT, 0.0)[..., None])
    A[:, :, row_from:, :] = rows
    return A


def draw_map(n: int, rng: np.random.Generator, row_from: int = ROW_FROM) -> np.ndarray:
    """Partner keys uniform on 1..i − 2 for rows i = row_from..n − 1 (never i − 1)."""
    i = np.arange(row_from, n)
    return rng.integers(1, i - 1)


def arm_maps(arm: str, n: int, L: int, H: int, rng: np.random.Generator, row_from: int = ROW_FROM):
    """(L, H, R) planted partner keys for arms G1–G6 (and G7, as G2)."""
    R = n - row_from
    if arm == "G1":
        return np.broadcast_to(draw_map(n, rng, row_from), (L, H, R)).copy()
    if arm in ("G2", "G7"):
        return np.stack([[draw_map(n, rng, row_from) for _ in range(H)] for _ in range(L)])
    if arm == "G3":
        out = np.empty((L, H, R), np.int64)
        for h in range(H):
            out[0, h] = draw_map(n, rng, row_from)
            for l in range(1, L):
                keep = rng.random(R) < 0.8
                out[l, h] = np.where(keep, out[l - 1, h], draw_map(n, rng, row_from))
        return out
    if arm == "G4":
        return np.broadcast_to(np.arange(row_from, n) - 1, (L, H, R)).copy()
    if arm == "G5":
        a, b = draw_map(n, rng, row_from), draw_map(n, rng, row_from)
        return np.stack([np.broadcast_to(a if l % 2 == 0 else b, (H, R)) for l in range(L)])
    if arm == "G6":
        return np.full((L, H, R), -1)
    raise B7Error(arm)


def planted_stages(arm: str, sinks: Sequence, ns: Sequence[int], seed: int, row_from: int = ROW_FROM):
    torch = _torch()
    rng = np.random.default_rng(seed)
    out = []
    for sink, n in zip(sinks, ns):
        L, H = sink.shape[:2]
        part = torch.as_tensor(arm_maps(arm, n, L, H, rng, row_from), device=sink.device)
        out.append(prompt_stage(plant(sink, part, n, row_from), row_from=row_from))
    return out


def arm_check(arm: str, res: Dict, min_entry: bool = False) -> Dict:
    """One arm's pass condition on one run's cells (design "Unit 1's gate"; G1 and G3 also at the
    minimum entry, G2 pooled, unit 1b)."""
    cs = res["cells"]
    desc = [c["description"] for c in cs]
    read = [c for c in cs if "P" in c]
    f = lambda k: [c[k] for c in read]
    rec: Dict = {"descriptions": {d: desc.count(d) for d in sorted(set(desc))}}
    if read:
        rec.update({"P_range": [min(f("P")), max(f("P"))], "N0_range": [min(f("N0")), max(f("N0"))],
                    "N1_range": [float(np.nanmin(f("N1"))), float(np.nanmax(f("N1")))],
                    "joint_rows_range": [min(f("joint_rows")), max(f("joint_rows"))],
                    "loo_rows_min": min(f("loo_rows_min")),                     # beside (choice 18)
                    "n0_rows_undefined_share": sum(f("n0_rows_undefined")) / sum(f("joint_rows"))})
    if arm == "G1":
        ok = (bool(read) and len(read) == len(cs) and min(f("P")) >= 0.99 and min(f("N1")) >= 0.99
              and min(p - q for p, q in zip(f("P"), f("N0"))) >= 0.9 and set(desc) == {DESCRIPTIONS[4]})
    elif arm == "G2":
        w = np.array(f("joint_rows"))
        rec["max_rel_dev"] = max(abs(c["P"] - c["N0"]) / c["N0"] for c in read) if read else None   # beside
        rec["pooled_rel_dev"] = float((w @ f("P") - w @ f("N0")) / (w @ f("N0"))) if read else None
        ok = (bool(read) and len(read) == len(cs) and abs(rec["pooled_rel_dev"]) <= G2_MAX
              and set(desc) == {DESCRIPTIONS[2]})
    elif arm == "G3":
        r = [(c["P"] - c["N0"]) / (1 - c["N0"]) for c in read]
        tol = [G3_Z * math.sqrt(G3_R * (1 - G3_R) / c["joint_rows"]) if min_entry else G3_TOL for c in read]
        rec["r_range"] = [min(r), max(r)] if r else None
        w = np.array(f("joint_rows"))
        rec["r_pooled"] = (float((w @ f("P") - w @ f("N0")) / (w @ (1 - np.array(f("N0")))))   # beside
                           if read else None)
        rec["band_min"] = min(tol) if tol else None
        ok = (bool(read) and len(read) == len(cs) and all(abs(x - G3_R) <= t for x, t in zip(r, tol))
              and set(desc) == {DESCRIPTIONS[3]})
    elif arm == "G4":
        ok = len(read) == len(cs) and min(f("N0")) >= 0.99 and set(desc) == {DESCRIPTIONS[1]}
    elif arm == "G5":
        lag2 = [c for c in cs if "P2" in c]
        rec["P2_min"] = min(c["P2"] for c in lag2) if lag2 else None
        ok = (set(desc) == {DESCRIPTIONS[2]} and len(lag2) == len(cs) - 1
              and all(c.get("two_cycle") for c in lag2) and rec["P2_min"] >= 0.99)
    elif arm == "G6":
        share = res["share"]["layer"]
        rec["max_layer_share"] = float(np.nanmax(share))
        ok = rec["max_layer_share"] == 0 and set(desc) == {DESCRIPTIONS[0]} and not read
    else:
        raise B7Error(arm)
    rec["pass"] = bool(ok)
    return rec


def g0(device: str) -> Tuple[Dict, Dict]:
    """Inputs: every attention and state file, shapes, row sums, unit norms, B3's rows and hash;
    keeps the real sink masses at SINK_STEPS for the planted arms."""
    gaps, worst = [], {"rowsum": 0.0, "norm": 0.0}
    sinks: Dict[int, List] = {s: [] for s in SINK_STEPS}
    ns: Dict[str, int] = {}
    hashes: Dict[str, str] = {}
    for step in STEPS:
        for p in V1:
            d = run_dir(step, p)
            fa, fx = d / "attentions.npz", d / "activations.npz"
            if not (fa.exists() and fx.exists()):
                gaps.append(f"{d}: missing")
                continue
            A = load_attn(fa, device, hashes)
            L, H, n, n2 = A.shape
            if (L, H) != (N_LAYERS, N_HEADS) or n != n2:
                gaps.append(f"{fa}: shape {tuple(A.shape)}")
            worst["rowsum"] = max(worst["rowsum"], float((A.sum(-1) - 1).abs().max()))
            if step in sinks:
                sinks[step].append(A[:, :, ROW_FROM:, 0].clone())
            del A
            X = load_states(fx, device, hashes)
            if tuple(X.shape) != (N_LAYERS + 1, n, 1024):
                gaps.append(f"{fx}: shape {tuple(X.shape)}")
            worst["norm"] = max(worst["norm"], float((X.norm(dim=-1) - 1).abs().max()))
            if ns.setdefault(p, n) != n:
                gaps.append(f"{p}: n {n} at {step}, {ns[p]} before")
        b3 = json.loads((B3_ROWS / "rows" / f"step{step}.json").read_text())
        if b3["code"] != B3_ROWS_CODE or not (B3_ROWS / "rows" / f"step{step}.npz").exists():
            gaps.append(f"B3 rows step {step}: code {b3['code']}")
    ph = prompts_hash()
    if ph != PROMPTS_HASH:
        gaps.append(f"prompts_hash {ph}, want {PROMPTS_HASH}")
    ok = not gaps and worst["rowsum"] <= G0_TOL and worst["norm"] <= G0_TOL
    return {"pass": bool(ok), "gaps": gaps, "max_rowsum_dev": worst["rowsum"], "max_norm_dev": worst["norm"],
            "n": ns, "files": len(hashes), "sha256": hashes}, {"sinks": sinks, "ns": [ns[p] for p in V1]}


def g8(device: str, seed: int) -> Dict:
    """Real attention at G8_STEP with heads G8_HEADS of layers G8_LAYERS overwritten by one shared
    planted map (each row's own sink kept); only differences from the unplanted run are kept."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    real, plan = [], []
    for p in V1:
        A = load_attn(run_dir(G8_STEP, p) / "attentions.npz", device)
        n = A.shape[-1]
        real.append(prompt_stage(A))
        mp = torch.as_tensor(draw_map(n, rng), device=device)
        for l in G8_LAYERS:
            for h in G8_HEADS:
                A[l, h] = plant(A[l, h, ROW_FROM:, 0][None, None], mp[None, None], n)[0, 0]
        plan.append(prompt_stage(A))
        del A
    r0, r1 = run_step(real, seed), run_step(plan, seed)
    lo, hi = min(G8_LAYERS), max(G8_LAYERS)
    touch = {lo - 1, lo, hi}
    others = [(a, b) for a, b in zip(r0["cells"], r1["cells"]) if a["l"] not in touch]
    dP = [abs(a.get("P", np.nan) - b.get("P", np.nan)) for a, b in others if "P" in a or "P" in b]
    dN1 = [abs(a["N1"] - b["N1"]) for a, b in others if "N1" in a and "N1" in b]
    same_read = all(("P" in a) == ("P" in b) for a, b in others)
    moved = [a["l"] for a, b in others if a["description"] != b["description"]]   # layers only (choice 10)
    c = r1["cells"][lo]
    hp = [c["heads"]["P"][h] for h in G8_HEADS]
    hs = [sorted(set(np.asarray(c["heads"]["hstar"])[:, h].tolist())) for h in G8_HEADS]
    found = all(set(x) <= set(G8_HEADS) for x in hs)
    ok = (c["description"] in DESCRIPTIONS[3:5] and min(hp) >= 0.95 and found and same_read
          and (max(dP) if dP else 0.0) <= 0.01)
    # |ΔP| = 0 on the other cells is a code check (cell l reads layers l and l + 1 only, same
    # seed); their N1 averages distant layers, 10–11 among them, so it may move (printed, choice 15)
    return {"pass": bool(ok), "cell": f"{lo} -> {hi}", "description": c["description"],
            "planted_heads_P": hp, "planted_heads_joint": [c["heads"]["joint"][h] for h in G8_HEADS],
            "planted_heads_hstar": hs, "other_cells_max_abs_dP_code_check": max(dP) if dP else 0.0,
            "other_cells_same_read": same_read, "other_cells": len(dP),
            "other_cells_max_abs_dN1": max(dN1) if dN1 else None, "other_cells_description_changed_at": moved}


def g9() -> Dict:
    torch = _torch()
    g = torch.Generator().manual_seed(0)
    u = torch.randn(1, 64, 1024, generator=g, dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True)
    same = d_of([d_sums(u.expand(25, -1, -1))])
    flip = d_of([d_sums(torch.cat([u * (-1) ** l for l in range(25)]))])
    err = max(max(abs(x) for x in same["D1"] + same["D2"]), max(abs(x - 2) for x in flip["D1"]),
              max(abs(x) for x in flip["D2"]))
    return {"pass": bool(err <= 1e-6), "max_err": float(err), "same_D1": same["D1"][0], "flip_D1": flip["D1"][0],
            "flip_D2": flip["D2"][0]}


def cmd_gate(a) -> int:
    torch = _torch()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = {"code": code_sha(), "prompts_hash": prompts_hash(), "runs": RUNS, "b3_rows": str(B3_ROWS),
            "device": a.device, "torch": torch.__version__, "numpy": np.__version__,
            "constants": {"ROW_FROM": ROW_FROM, "W_MIN": W_MIN, "TOP": TOP, "MIN_JOINT": MIN_JOINT,
                          "MIN_HEADS": MIN_HEADS, "POS_N0": POS_N0, "BIN": BIN, "N_BOOT": N_BOOT,
                          "PLANT": PLANT, "SINK_STEPS": SINK_STEPS, "NONFINITE_MAX": NONFINITE_MAX,
                          "G2_MAX": G2_MAX, "G3": [G3_R, G3_TOL, G3_Z], "G7_SEEDS": G7_SEEDS}}
    gate: Dict = {"meta": meta}
    gate["G0"], real = g0(a.device)
    print(f"G0 pass={gate['G0']['pass']} ({time.time() - t0:.0f}s)", flush=True)
    for arm in ("G1", "G2", "G3", "G4", "G5", "G6"):
        runs = {}
        for s in SINK_STEPS:
            st = planted_stages(arm, real["sinks"][s], real["ns"], seed_for("B7", arm, s))
            runs[str(s)] = arm_check(arm, run_step(st, seed_for("B7", arm, s, "boot")))
            if arm in ("G1", "G3"):                       # choice 13: also at the minimum entry
                runs[f"{s}_min_entry"] = arm_check(
                    arm, run_step(st, seed_for("B7", arm, s, "boot"), thin=MIN_JOINT), min_entry=True)
            del st
        gate[arm] = {"pass": all(r["pass"] for r in runs.values()), "by_sink_step": runs}
        print(f"{arm} pass={gate[arm]['pass']} ({time.time() - t0:.0f}s)", flush=True)
    g7 = {}
    for variant, thin in (("full", None), ("min_entry", MIN_JOINT)):
        false = read = unread = 0
        for s in SINK_STEPS:
            for k in range(G7_SEEDS):
                st = planted_stages("G7", real["sinks"][s], real["ns"], seed_for("B7", "G7", s, k))
                cs = run_step(st, seed_for("B7", "G7", s, k, "boot"), thin=thin)["cells"]
                del st
                for c in cs:
                    if c["description"] == DESCRIPTIONS[0]:
                        unread += 1
                    else:
                        read += 1
                        false += c["description"] != DESCRIPTIONS[2]
        g7[variant] = {"false": false, "read": read, "unread": unread, "rate": false / read if read else None,
                       "pass": bool(read and false / read <= G7_MAX)}
        print(f"G7 {variant}: {false}/{read} ({time.time() - t0:.0f}s)", flush=True)
    gate["G7"] = {"pass": all(v["pass"] for v in g7.values()), **g7}
    gate["G8"] = g8(a.device, seed_for("B7", "G8"))
    print(f"G8 pass={gate['G8']['pass']} ({time.time() - t0:.0f}s)", flush=True)
    gate["G9"] = g9()
    gate["pass"] = all(gate[k]["pass"] for k in ARMS)
    gate["seconds"] = round(time.time() - t0)
    dump(out / "gate.json", gate)
    print(json.dumps({k: gate[k]["pass"] for k in ARMS} | {"pass": gate["pass"]}))
    torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- the read (unit 2)

def check_gate(gate: Dict) -> None:
    """Refuse unless the gate passed on this code, these prompts and these runs (choice 16)."""
    m = gate.get("meta", {})
    want = {"code": code_sha(), "prompts_hash": prompts_hash(), "runs": json.loads(json.dumps(RUNS))}
    bad = [f"{k} {m.get(k)!r} vs {v!r}" for k, v in want.items() if m.get(k) != v]
    if not gate.get("pass") or bad or not gate.get("G0", {}).get("sha256"):
        raise B7Error(f"gate: pass={gate.get('pass')}; {'; '.join(bad) or 'no input hashes'}"
                      " (refuse: no real cell before a passing gate on this code and these inputs)")


def cmd_read(a) -> int:
    gate = json.loads(Path(a.gate).read_text())
    check_gate(gate)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in a.steps.split(",")] if a.steps else list(STEPS)
    for step in steps:
        t0 = time.time()
        st, ds, hashes = [], [], {}
        for p in V1:
            d = run_dir(step, p)
            s = prompt_stage(load_attn(d / "attentions.npz", a.device, hashes),
                             load_states(d / "activations.npz", a.device, hashes), soft=True)
            ds.append(s.pop("d"))
            st.append(s)
        moved = [k for k, v in hashes.items() if gate["G0"]["sha256"].get(k) != v]
        if moved:
            raise B7Error(f"step {step}: {len(moved)} input files differ from the gate's ({moved[0]}, ...)")
        res = run_step(st, seed_for("B7", "read", step), beside=True)
        rec = {"step": step, "code": code_sha(), "prompts_hash": prompts_hash(), "gate": str(a.gate),
               "runs": {p: str(run_dir(step, p)) for p in V1}, "sha256": hashes, "n": [s["n"] for s in st],
               **res, "D": d_of(ds), "alpha": alpha_beside(step)}
        dump(out / f"step{step}.json", rec)
        print(f"step {step}: {time.time() - t0:.0f}s", flush=True)
        del st
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("read")
    p.add_argument("--out", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--steps", default="")
    p.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    return {"gate": cmd_gate, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
