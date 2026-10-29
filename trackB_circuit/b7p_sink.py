"""B7′, sink-excluded routing activity, unit 1 (`trackB_circuit/design-B7p.md`, frozen 2026-10-25,
Blocked 114 (a)): each prompt's second-sink keys found per head (R1–R6) and dropped as position 0
is, B7's instrument (`b7_activity.py` at `B7_CODE`, imported, not edited) on the dropped rows with
N0 re-estimated per bootstrap draw, the beside columns, and the gate G0–G15.

**The drop** (R4) moves the mass of every key in S_q onto position 0, and puts all of a row whose
query is in S_q on position 0. B7's `partners`, `pooled_partners` and `soft_sums` then read the
dropped stack unchanged: w_i = 1 − A_i0 − Σ_{k ∈ S_q, k ≤ i} A_ik, keys renormalised over 1..i
outside S_q, a query in S_q out (w = 0). Passes apply cumulatively: dropping S_1 and then S_2 is
dropping S_1 ∪ S_2.

**N0 per draw** (design "Objects changed from B7"). For a draw with prompt multiplicities m_q:
c_i(m) = [Σ_{q,q′} m_q m_q′ X_qq′(i) − Σ_q m_q² S_q(i)] / [C_p(m) C_q(m) − Σ_q m_q² V_q(i)]. Here X_qq′(i) =
Σ_{k ≤ i−1} p_q(k) q_q′(k) over the bin's per-prompt histograms, C_p(m) = Σ_q m_q C_p,q(i), and V_q,
S_q count prompt q's rows of the bin that are valid at i, and valid and self-agreeing. Row 0 (m = 1)
is B7's `n0_sums`. A draw's N0 is Σ_q m_q Σ_{i ∈ q} c_i(m) over Σ_q m_q (rows defined), gathered at
each fold's h* as B7 gathers it.

**Choices the design left open, made before any output:**
(1) R2 takes every key at ≥ SINK_FRAC of a head's condensed rows (a head can name up to 4). Keys
count from 1, since position 0 is never a partner.
(2) R5: passes 1..SINK_PASSES each add the keys R2 finds on the current rows. After a pass 3 that
added a key, R2 runs once more; what it would add is recorded as `unconverged` and not dropped.
(3) A query in S_q (a key ≥ 16) is out at every layer and head.
(4) G1–G7 plant B7's stacks with B7's seeds (`seed_for("B7", …)`), so with S_q empty the rows are
B7's and only N0 changes. G7 and G13 plant each seed once and run both entries on it. The read
uses B7's bootstrap seeds, so B7 → B7′ transitions come from the rows and N0, not from the draws.
(5) The heads-differ maps (G1d, G3d, G11, G12): per prompt 16 content channels, and per layer one
head permutation σ_l, drawn per seed and shared by the 7 prompts; head h of layer l carries
channel σ_l(h). G3d's 0.8 chain runs per channel. G10's maps (G2's) are drawn per head, so they
differ already.
(6) The second-sink plant, drawn per seed in this order: the permutations, the heavy prompts A and
B, three distinct keys in KEY_RANGE (k_A, k_B, k_B′), then per prompt its maps, membership chain,
row types and shares. s_{q,h} is drawn per layer as well (the design's fence: no head identity
across layers). The chain u is one per prompt over the 24 layers. A light prompt's row is
condensed in layers 7–20 when u < LIGHT_COND (the same chain), on its content partner. Its own key
sits on rows drawn from the diffuse rest, independently per (layer, head, row), at the rate that
makes the key's share of the head's condensed rows s ~ U(LIGHT_S). So the light key does not
persist across layers or disturb the content rows; a persisting sub-threshold key is G15's.
Light-key rows are pure (0.8 on the key). No content partner in a planted prompt is a planted key
(a colliding draw is redrawn), so (d) measures the rule, not collisions.
(7) (d) counts the hidden rows that are in after the drop (w ≥ 0.05). A hidden row with A_i0 >
0.83 leaves 0.3 (1 − A_i0) < 0.05 and is out by R4, like any row; the out count is recorded.
"No pure row condensed" counts every pure row.
(8) G10 (c) and G14 (c): every read cell, not every cell, must be "rerouting", and the pooled band
is over the read cells; unread cells are counted.
(9) G10–G12, G14, G15 run at each of SINK_STEPS on one seed each, as B7's G1–G6 do; G11 and G12 also
at the minimum entry. G13 runs 20 seeds a sink step at both entries, as G7 does.
(10) N0k (beside 8) is computed in the read and in G10–G15, never in G1–G9 (it changes no
description). Its validity mirrors N0's: key k is valid at i when k ≤ i.
(11) Equal-weight P (beside 1) per draw is Σ_q m_q P_q / Σ_q m_q over the drawn prompts with joint
rows, P_q over the entering heads at that fold's h*.
(12) Neighbour persistence, D and α beside stay B7's (they come from the states and B3, not from
attention rows). The pooled row and the soft overlap are read on the dropped stack.
(13) The w quartiles (beside 7) are over the rows jointly condensed by an entering head and its
entry h*, over the 7 prompts, with w at layer l.
(14) R6's tokens are decoded with pythia-410m's tokenizer. A key is the prompt's first `.` when it
is the first position ≥ 1 whose decoded token, stripped, is "."; it is the first `\\n` when it is
the first position whose decoded token contains a newline.
(15) G8: S_q is computed on a copy before the plant, and each prompt's shared map is redrawn
where it hits S_q (keys outside S_q).
(16) G0 also requires B7's gate (`B7_GATE`) passed at `B7_CODE`, `b7_activity.py` still at
`B7_CODE`, the 252 file hashes equal to that gate's, and B7's 18 step records (`B7_READ`) present
at `B7_CODE` with the same hashes.

NUMERICS: stored attention (float32) cast to float64 on the GPU; histograms and N0's per-draw
numerators and denominators in float32 (integer counts below 2^24, TF32 off, so exact); c_i, its
sums and every statistic in float64. CPU allowed (argmax and counting).

Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b7p_sink gate --out <dir>
    python -m trackB_circuit.b7p_sink read --out <dir> --gate <dir>/gate.json [--steps 512,143000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import RUNS, V1, prompts_hash, run_dir
from trackB_circuit import b7_activity as b7
from trackB_circuit.b2a import REPO, STEPS, seed_for
from trackB_circuit.b7_activity import (BIN, DESCRIPTIONS, MIN_HEADS, MIN_JOINT, N_BOOT, PLANT, ROW_FROM,
                                        SINK_STEPS, W_MIN, B7Error, clean, describe, dump, interval,
                                        nonfinite)

DATA = b7.DATA
B7_CODE = "a1de85990c74"
B7_GATE = DATA / "b7/2026-10-25-unit1c-r2/gate.json"
B7_READ = DATA / "b7/2026-10-25-unit2"
SINK_FRAC = 0.25                           # design R2 (placed)
SINK_MIN = MIN_JOINT                       # design R2 (placed; B7's MIN_JOINT)
SINK_PASSES = 3                            # design R5 (placed)
NEAR_LO = 0.125                            # design beside 6: the near-miss band [1/8, 1/4)
FRAC_BESIDE = (0.125, 0.5)                 # design beside 3
MIDDLE = tuple(range(7, 20))               # cells 7 → 8 … 19 → 20
KEY_RANGE = (10, 54)                       # design (gate): B7's real sinks at 10–54
A_LAYERS, B_LAYERS, B2_LAYERS = tuple(range(7, 21)), tuple(range(9, 13)), tuple(range(14, 21))
LIGHT_LAYERS = A_LAYERS
SINK_S = {"full": (0.5, 0.95), "sparse": (0.3, 0.6), "leak": (0.10, 0.20)}   # design G10, G14, G15
SPARSE_HEADS = (3, 6)                      # design G14: 3–6 of 16 heads carry the sink
KEEP = 0.8                                 # design: G3's chain, on sink membership
HIDDEN = (0.7, 0.2)                        # design: a hidden row's mass on k, on its partner (0.1 uniform)
PURE = 0.8                                 # design: a pure row's mass on k (0.2 uniform)
LIGHT_COND, LIGHT_S = 0.1, (0.0, 0.04)     # design: light prompts in 7–20
G10_L = 0.75                               # design G10 (b)
G2_MAX, G7_SEEDS, G7_MAX = b7.G2_MAX, b7.G7_SEEDS, b7.G7_MAX
D_MIN = 0.99                               # design (d)
BESIDE5 = {"prompts": ("hdbscan_code", "latex_monograph"), "heads": ((6, 4), (15, 7))}
CONTENT = {"G10": "G2", "G11": "G1d", "G12": "G3d", "G13": "G2", "G14": "G2", "G15": "G2"}
VARIANT = {"G10": "full", "G11": "full", "G12": "full", "G13": "full", "G14": "sparse", "G15": "leak"}
ARMS = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G1d", "G3d", "G10", "G11", "G12",
        "G13", "G14")                      # G15 has no pass condition (recorded)


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackB_circuit/b7p_sink.py").read_bytes()).hexdigest()[:12]


def _torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False          # N0's float32 counts stay exact
    return torch


# ---------------------------------------------------------------- R1–R6: the rule and the drop

def drop_(A, keys: Sequence[int], row_from: int = ROW_FROM):
    """R4 in place on a stack (…, n, n): each key's mass moved onto position 0 and its column
    zeroed; a query in `keys` (≥ `row_from`) put wholly on position 0, so it is out (w = 0)."""
    torch = _torch()
    if not len(keys):
        return A
    k = torch.as_tensor(sorted(keys), device=A.device)
    A[..., 0] += A[..., k].sum(-1)
    A[..., k] = 0
    q = k[k >= row_from]
    if q.numel():
        A[..., q, :] = 0
        A[..., q, 0] = 1
    return A


def key_shares(part, n: int):
    """Each head's share of its condensed rows on each key, (…, n), and its condensed rows (…)."""
    torch = _torch()
    cond = part >= 0
    cnt = torch.zeros(part.shape[:-1] + (n,), dtype=torch.float64, device=part.device)
    cnt.scatter_add_(-1, part.clamp(min=0), cond.to(torch.float64))
    tot = cond.sum(-1)
    return cnt / tot.clamp(min=1)[..., None], tot


def _hits(part, n: int, frac: float, min_rows: int, have: Sequence[int]) -> Dict[int, List]:
    share, tot = key_shares(part, n)
    hit = (share >= frac) & (tot >= min_rows)[..., None]
    out: Dict[int, List] = {}
    for l, h, k in hit.nonzero().tolist():
        if k not in have:
            out.setdefault(k, []).append([l, h, round(float(share[l, h, k]), 4)])
    return out


def sink_rule(A, frac: float = SINK_FRAC, min_rows: int = SINK_MIN, passes: int = SINK_PASSES) -> Dict:
    """R1–R5 on one prompt's stack (L, H, n, n), dropping in place: the sink set, and per key the
    pass and the heads (l, h) where it qualified with its share there (R6, choice 2)."""
    n = A.shape[-1]
    S: List[int] = []
    found, unconv = [], []
    for p in range(1, passes + 2):
        part, _ = b7.partners(A)
        new = _hits(part, n, frac, min_rows, S)
        if not new:
            break
        recs = [{"key": k, "pass": p, "heads": v} for k, v in sorted(new.items())]
        if p > passes:
            unconv = recs
            break
        found += recs
        S += sorted(new)
        drop_(A, sorted(new))
    return {"keys": sorted(S), "found": found, "unconverged": unconv}


def sink_stage(A, X=None, soft: bool = False, frac: float = SINK_FRAC) -> Dict:
    """The rule, then B7's stage 1 on the dropped stack (``A`` is changed in place); adds each
    row's w after the drop and each head's near-miss share (beside 6)."""
    torch = _torch()
    rule = sink_rule(A, frac)
    st = b7.prompt_stage(A, X, soft=soft)
    st["w"] = 1 - A[:, :, ROW_FROM:, 0]
    share, tot = key_shares(st["part"], A.shape[-1])
    st["near"] = torch.where(tot >= SINK_MIN, share.max(-1).values, torch.nan)
    st["rule"] = rule
    return st


# ---------------------------------------------------------------- N0 per draw

def n0_draws(va, vb, J, pos, pid, M, bin_: int = BIN, row_from: int = ROW_FROM, chunk: int = 512):
    """N0's row sums per draw over pairs of different original rows: for each draw d, prompt q and
    head pair (h, g), Σ c_i(m_d) over q's rows where J holds and c_i is defined, and the count of
    those rows, each (D, Q, H, G) float64. ``va`` (H, R), ``vb`` (G, R) the rows' values in
    0..i − 1 (offsets i − π(i), or keys π(i) − 1 for N0k), meaningful where J (H, G, R); draws in
    chunks of `chunk` (memory only)."""
    torch = _torch()
    F = torch.nn.functional
    f32, f64 = torch.float32, torch.float64
    H, G, R = J.shape
    D, Q = M.shape
    dev = J.device
    K = int(pos.max())                                   # values 0..K − 1
    bi = (pos - row_from) // bin_
    nb = int(bi.max()) + 1
    ohc = F.one_hot(bi * Q + pid, nb * Q).to(f32)         # (R, nb Q): the row's bin and prompt
    seg = torch.bincount(pid, minlength=Q).tolist()       # rows are contiguous by prompt (`_cat`)
    if not bool((pid[1:] >= pid[:-1]).all()):
        raise B7Error("n0_draws: rows not contiguous by prompt")
    at = pos - 1
    Mf = M.to(f32)
    M2 = Mf ** 2
    MM = (Mf[:, :, None] * Mf[:, None, :]).reshape(D, Q * Q)
    OB = F.one_hot(vb, K).to(f32)                         # (G, R, K)
    sums = torch.zeros((D, Q, H, G), dtype=f64, device=dev)
    rows = torch.zeros_like(sums)

    def read(x):                                          # (G, nb Q, K) histogram → cumulative at i − 1
        c = torch.cumsum(x.view(G, nb, Q, K), -1).permute(0, 1, 3, 2)
        return c[:, bi, at]                               # (G, R, Q)

    for h in range(H):
        W = J[h].to(f32)[:, :, None] * ohc[None]          # (G, R, nb Q)
        p = torch.einsum("grc,rk->gck", W, F.one_hot(va[h], K).to(f32)).view(G, nb, Q, K)
        q = torch.einsum("grc,grk->gck", W, OB).view(G, nb, Q, K)
        X = torch.cumsum(p[:, :, :, None, :] * q[:, :, None, :, :], -1).permute(0, 1, 4, 2, 3)
        Xr = X[:, bi, at].reshape(G * R, Q * Q)           # (G R, Q Q)
        del X
        OM = F.one_hot(torch.maximum(va[h][None], vb), K).to(f32)
        Vq = read(torch.einsum("grc,grk->gck", W, OM)).reshape(G * R, Q)
        Sq = read(torch.einsum("grc,grk->gck", W * (va[h][None] == vb)[..., None].to(f32), OM)).reshape(G * R, Q)
        Cp = read(p.reshape(G, nb * Q, K)).reshape(G * R, Q)
        Cq = read(q.reshape(G, nb * Q, K)).reshape(G * R, Q)
        Jh = J[h].reshape(1, G * R)
        for d0 in range(0, D, chunk):
            d = slice(d0, d0 + chunk)
            num = MM[d] @ Xr.T - M2[d] @ Sq.T              # (chunk, G R), exact integers
            den = (Mf[d] @ Cp.T) * (Mf[d] @ Cq.T) - M2[d] @ Vq.T
            ok = Jh & (den > 0)
            c = torch.where(ok, num.to(f64) / torch.where(den > 0, den, 1.0).to(f64), 0.0)
            n_ = c.shape[0]
            for qq, (cs, os) in enumerate(zip(c.view(n_, G, R).split(seg, -1), ok.view(n_, G, R).split(seg, -1))):
                sums[d, qq, h] = cs.sum(-1)
                rows[d, qq, h] = os.sum(-1).to(f64)
            del num, den, ok, c
    return sums, rows


def pair_counts_p(stages: Sequence[Dict], M, lags: Sequence[int] = (1, 2), thin: Optional[int] = None,
                  seed: int = 0, n0k: bool = False) -> Dict:
    """B7's `pair_counts` (same thinning and generator) with N0's sums per draw, and N0k's for the
    pairs (l, l + 1) when `n0k`."""
    torch = _torch()
    part, pos, pid = b7._cat(stages, "part")
    L, H, R = part.shape
    Q = len(stages)
    dev = part.device
    onehot = torch.nn.functional.one_hot(pid, Q).to(torch.float64)
    cond = part >= 0
    off = torch.where(cond, pos - part, 0)
    key = torch.where(cond, part - 1, 0)
    agree = torch.zeros((L, L, Q, H, H), dtype=torch.float64, device=dev)
    joint = torch.zeros_like(agree)
    n0, n0k_ = {}, {}
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for l in range(L):
        J = cond[l][:, None, None, :] & cond[None, :, :, :]
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
                n0[(l, l + lag)] = n0_draws(off[l], off[l + lag], J[:, l + lag], pos, pid, M)
        if n0k and l + 1 < L:
            n0k_[(l, l + 1)] = n0_draws(key[l], key[l + 1], J[:, l + 1], pos, pid, M)
    out = {"agree": agree, "joint": joint, "n0": n0, "Q": Q, "L": L, "H": H}
    if n0k:
        out["n0k"] = n0k_
    return out


# ---------------------------------------------------------------- matching and cells

def matched_p(agree, joint, M, extras: Optional[Dict] = None, min_joint: int = MIN_JOINT,
              entry: str = "all") -> Dict:
    """B7's `matched` with extras that may vary by draw ((Q, H, G) or (D, Q, H, G)), the
    equal-weight P (beside 1, choice 11), and `entry` "loo" for unit 1b's entry rule (beside 2)."""
    torch = _torch()
    extras = extras or {}
    D, Q = M.shape
    S_all = torch.einsum("dq,qhg->dhg", M, agree)
    S_out = S_all[:, None] - M[:, :, None, None] * agree[None]
    hs = S_out.argmax(-1)
    others = M.sum(1, keepdim=True) - M
    bad = ((M > 0) & (others == 0)).any(1)
    idx = hs[..., None]

    def per_q(x):                                                            # (D, Q, H)
        x = x[None].expand(D, -1, -1, -1) if x.dim() == 3 else x
        return x.gather(-1, idx).squeeze(-1)

    def take(x):
        return (M[:, :, None] * per_q(x)).sum(1)                             # (D, H)

    Aq, Jq = per_q(agree), per_q(joint)
    Ah, Jh = (M[:, :, None] * Aq).sum(1), (M[:, :, None] * Jq).sum(1)
    he = S_all[0].argmax(-1)
    Je = joint.sum(0).gather(-1, he[:, None]).squeeze(-1)
    enter = (Je >= min_joint) & (Jh[0] > 0) if entry == "all" else Jh[0] >= min_joint
    den = Jh[:, enter].sum(1)
    nan = torch.full_like(den, torch.nan)
    ok = (den > 0) & ~bad

    def cell(x):
        return torch.where(ok, x[:, enter].sum(1) / torch.where(den > 0, den, 1.0), nan).cpu().numpy()

    aq, jq = Aq[:, :, enter].sum(-1), Jq[:, :, enter].sum(-1)                # (D, Q)
    has = ((jq > 0) & (M > 0)).to(torch.float64)
    pq = torch.where(jq > 0, aq / torch.where(jq > 0, jq, 1.0), 0.0)
    wq = (M * has).sum(1)
    peq = torch.where(~bad & (wq > 0), (M * has * pq).sum(1) / torch.where(wq > 0, wq, 1.0), nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        headP = (Ah[0] / Jh[0]).cpu().numpy()
    return {"P": cell(Ah), **{k: cell(take(v)) for k, v in extras.items()}, "P_eq": peq.cpu().numpy(),
            "n_enter": int(enter.sum()), "dropped": int(bad[1:].sum()), "enter": enter.cpu().numpy(),
            "head_P": headP, "head_joint": Jh[0].cpu().numpy(), "hstar": hs[0].cpu().numpy(),
            "head_joint_entry": Je.cpu().numpy(), "hstar_entry": he.cpu().numpy(),
            "n_enter_loo_rule": int((Jh[0] >= min_joint).sum())}


def _n0_extras(pair):
    return {"N0s": pair[0], "N0r": pair[1]}


def _ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / b


def _far(ag, jt, M, l: int, L: int, entry: str):
    far = []
    for l2 in range(L):
        if abs(l2 - l) >= 2:
            m = matched_p(ag[l, l2], jt[l, l2], M, entry=entry)
            if m["n_enter"] >= MIN_HEADS:
                far.append(m["P"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return far, (np.nanmean(np.stack(far), 0) if far else None)


def _describe(m, N0, N1, far) -> Tuple[str, Optional[str], Tuple, Tuple, float, float]:
    P = m["P"]
    N1 = N1 if N1 is not None else np.full_like(P, np.nan)
    d0, d1 = interval(P - N0), interval(P - N1)
    bad0, bad1 = nonfinite(P - N0), nonfinite(P - N1)
    desc, why = describe(m["n_enter"], float(N0[0]), d0, d1, bool(far), bad0, bad1)
    return desc, why, d0, d1, bad0, bad1


def cells_p(cnt: Dict, M, flat: Optional[Dict] = None) -> List[Dict]:
    """B7's cell table for one step with N0 per draw; beside, the equal-weight P, the description
    under unit 1b's entry rule and, where counted, N0k and its flag (design beside 1, 2, 8)."""
    L = cnt["L"]
    ag, jt, n0 = cnt["agree"], cnt["joint"], cnt["n0"]
    out = []
    for l in range(L - 1):
        m1 = matched_p(ag[l, l + 1], jt[l, l + 1], M, _n0_extras(n0[(l, l + 1)]))
        N0 = _ratio(m1["N0s"], m1["N0r"])
        far, N1 = _far(ag, jt, M, l, L, "all")
        desc, why, d0, d1, bad0, bad1 = _describe(m1, N0, N1, far)
        P = m1["P"]
        loo = m1["head_joint"][m1["enter"]]
        c = {"l": l, "n_enter": m1["n_enter"], "n_far": len(far), "dropped": m1["dropped"],
             "joint_rows": float(loo.sum()), "description": desc, "not_read_because": why,
             "loo_rows_min": float(loo.min()) if loo.size else None,
             "loo_rows_median": float(np.median(loo)) if loo.size else None,
             "n_enter_loo_rule": m1["n_enter_loo_rule"],
             "nonfinite_draws": {"P_minus_N0": bad0, "P_minus_N1": bad1}}
        if m1["n_enter"] >= MIN_HEADS:
            c.update({"P": float(P[0]), "N0": float(N0[0]), "N1": float(N1[0]), "P_ci": interval(P),
                      "N0_ci": interval(N0), "P_minus_N0_ci": d0, "P_minus_N1_ci": d1,
                      "n0_rows_undefined": c["joint_rows"] * (1 - float(m1["N0r"][0])),
                      "P_eq": float(m1["P_eq"][0]), "P_eq_ci": interval(m1["P_eq"])})
        # beside 2: unit 1b's entry (≥ MIN_JOINT leave-one-out rows), described in full
        mb = matched_p(ag[l, l + 1], jt[l, l + 1], M, _n0_extras(n0[(l, l + 1)]), entry="loo")
        farb, N1b = _far(ag, jt, M, l, L, "loo")
        c["loo_rule"] = {"n_enter": mb["n_enter"],
                         "description": _describe(mb, _ratio(mb["N0s"], mb["N0r"]), N1b, farb)[0]}
        if "n0k" in cnt and m1["n_enter"] >= MIN_HEADS:                       # beside 8
            mk = matched_p(ag[l, l + 1], jt[l, l + 1], M, _n0_extras(cnt["n0k"][(l, l + 1)]))
            N0k = _ratio(mk["N0s"], mk["N0r"])
            dk = interval(P - N0k)
            c.update({"N0k": float(N0k[0]), "P_minus_N0k_ci": dk,
                      "n0k_flag": bool(d0[0] > 0 and not dk[0] > 0)})
        c["heads"] = {"P": m1["head_P"].tolist(), "joint": m1["head_joint"].tolist(),
                      "enter": m1["enter"].tolist(), "hstar": m1["hstar"].tolist(),
                      "joint_entry": m1["head_joint_entry"].tolist(), "hstar_entry": m1["hstar_entry"].tolist()}
        if l + 2 < L:
            m2 = matched_p(ag[l, l + 2], jt[l, l + 2], M, _n0_extras(n0[(l, l + 2)]))
            if m2["n_enter"] >= MIN_HEADS:
                N0_2 = _ratio(m2["N0s"], m2["N0r"])
                x2 = m2["P"] - N0_2
                d2 = interval(x2)
                c.update({"P2": float(m2["P"][0]), "N0_2": float(N0_2[0]),
                          "P2_minus_N0_2_ci": d2, "nonfinite_draws_2": nonfinite(x2)})
                c["two_cycle"] = bool(c["description"] == DESCRIPTIONS[2] and d2[0] > 0
                                      and nonfinite(x2) <= b7.NONFINITE_MAX)
        if "soft" in cnt and m1["n_enter"] >= MIN_HEADS:
            s = b7.matched(ag[l, l + 1], jt[l, l + 1], M[:1], {"soft": cnt["soft"][:, l]})
            c["soft"] = float(s["soft"][0])
        if flat is not None:
            for name, f in flat.items():
                v = b7.ratio(f["agree"][l, l + 1], f["joint"][l, l + 1], M)
                c[f"{name}_P"], c[f"{name}_P_ci"] = float(v[0]), interval(v)
                if name == "nb":
                    fr = [b7.ratio(f["agree"][l, l2], f["joint"][l, l2], M) for l2 in range(f["agree"].shape[0])
                          if abs(l2 - l) >= 2]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        n1 = np.nanmean(np.stack(fr), 0)
                    c["nb_N1"], c["nb_P_minus_N1_ci"] = float(n1[0]), interval(v - n1)
        out.append(c)
    return out


def run_step_p(stages: Sequence[Dict], seed: int, n_boot: int = N_BOOT, thin: Optional[int] = None,
               beside: bool = False, n0k: bool = False) -> Dict:
    """B7's `run_step` with N0 per draw (the draws are B7's for the same seed)."""
    torch = _torch()
    M = b7.draws(len(stages), n_boot, seed, stages[0]["part"].device)
    cnt = pair_counts_p(stages, M, thin=thin, seed=seed, n0k=n0k)
    flat = None
    if beside:
        cnt["soft"] = torch.stack([s["soft"] for s in stages])
        flat = {"pooled": b7.flat_counts(stages, "ppart"), "nb": b7.flat_counts(stages, "nb")}
    return {"cells": cells_p(cnt, M, flat), "share": b7.condensed_share(stages)}


def w_quartiles(stages: Sequence[Dict], cell: Dict) -> Optional[List[float]]:
    """Beside 7 (choice 13): quartiles of w at layer l over rows jointly condensed by an entering
    head and its entry h*, over the prompts."""
    torch = _torch()
    l = cell["l"]
    vals = []
    for h, (e, g) in enumerate(zip(cell["heads"]["enter"], cell["heads"]["hstar_entry"])):
        if e:
            for s in stages:
                j = (s["part"][l, h] >= 0) & (s["part"][l + 1, g] >= 0)
                vals.append(s["w"][l, h][j])
    if not vals:
        return None
    v = torch.cat(vals)
    return torch.quantile(v, torch.tensor([0.25, 0.5, 0.75], dtype=v.dtype, device=v.device)).tolist() if v.numel() else None


# ---------------------------------------------------------------- the plants

def plant2(sink, part, pm, key, km, n: int, row_from: int = ROW_FROM):
    """A planted stack (L, H, n, n) float64: rows i ≥ `row_from` keep `sink` as A_i0; of w = 1 −
    sink, `pm` goes on `part`, `km` on `key` (−1: none) and the rest uniformly over keys 1..i;
    rows below `row_from` uniform over keys 0..i. With km = 0 and pm = PLANT it is B7's `plant`."""
    torch = _torch()
    L, H, R = sink.shape
    dev = sink.device
    i = torch.arange(n, device=dev)
    tri = (i[None, :] <= i[:, None]).to(torch.float64)
    A = (tri / (i + 1)[:, None].to(torch.float64)).expand(L, H, n, n).clone()
    ii = torch.arange(row_from, n, device=dev).to(torch.float64)
    w = 1 - sink
    pm = torch.where(part >= 0, pm, 0.0)
    km = torch.where(key >= 0, km, 0.0)
    keys = (i[None, :] >= 1) & (i[None, :] <= i[row_from:, None])
    rows = (w * (1 - pm - km) / ii)[..., None] * keys.to(torch.float64)
    rows[..., 0] = sink
    rows.scatter_add_(-1, part.clamp(min=0)[..., None], (w * pm)[..., None])
    rows.scatter_add_(-1, key.clamp(min=0)[..., None], (w * km)[..., None])
    A[:, :, row_from:, :] = rows
    return A


def draw_map_avoid(n: int, rng: np.random.Generator, avoid=(), row_from: int = ROW_FROM) -> np.ndarray:
    """B7's `draw_map` (uniform on 1..i − 2), redrawn where it hits `avoid` (choice 6)."""
    m = b7.draw_map(n, rng, row_from)
    if len(avoid):
        hi = np.arange(row_from, n) - 1
        bad = np.isin(m, list(avoid))
        while bad.any():
            m[bad] = rng.integers(1, hi[bad])
            bad = np.isin(m, list(avoid))
    return m


def content_maps(kind: str, n: int, L: int, H: int, rng: np.random.Generator, perms: np.ndarray,
                 avoid=(), row_from: int = ROW_FROM) -> np.ndarray:
    """(L, H, R) content partners: "G2" drawn per head, "G1d" one frozen channel per head through
    σ_l, "G3d" each channel a 0.8 chain over layers through σ_l (choice 5)."""
    R = n - row_from
    if kind == "G2":
        return np.stack([[draw_map_avoid(n, rng, avoid, row_from) for _ in range(H)] for _ in range(L)])
    if kind == "G1d":
        ch = np.stack([draw_map_avoid(n, rng, avoid, row_from) for _ in range(H)])
        return np.stack([ch[perms[l]] for l in range(L)])
    if kind == "G3d":
        ch = np.empty((L, H, R), np.int64)
        for c in range(H):
            ch[0, c] = draw_map_avoid(n, rng, avoid, row_from)
            for l in range(1, L):
                keep = rng.random(R) < KEEP
                ch[l, c] = np.where(keep, ch[l - 1, c], draw_map_avoid(n, rng, avoid, row_from))
        return np.stack([ch[l, perms[l]] for l in range(L)])
    raise B7Error(kind)


def _chain(L: int, R: int, rng: np.random.Generator) -> np.ndarray:
    u = np.empty((L, R))
    u[0] = rng.random(R)
    for l in range(1, L):
        u[l] = np.where(rng.random(R) < KEEP, u[l - 1], rng.random(R))
    return u


def _shares(variant: str, H: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = SINK_S[variant]
    if variant == "full":
        return rng.uniform(lo, hi, H)
    s = np.zeros(H)
    on = rng.choice(H, int(rng.integers(SPARSE_HEADS[0], SPARSE_HEADS[1] + 1)), replace=False)
    s[on] = rng.uniform(lo, hi, on.size)
    return s


def heads_differ_stages(kind: str, sinks, ns: Sequence[int], seed: int) -> Tuple[List[Dict], List[Dict]]:
    """G1d / G3d: B7's plant with maps that differ by prompt and head (choice 5), each through the
    rule; returns the stages and each prompt's rule record."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    L, H = sinks[0].shape[:2]
    perms = np.stack([rng.permutation(H) for _ in range(L)])
    st, rules = [], []
    for sink, n in zip(sinks, ns):
        part = torch.as_tensor(content_maps(kind, n, L, H, rng, perms), device=sink.device)
        s = sink_stage(b7.plant(sink, part, n))
        rules.append(s["rule"])
        st.append(s)
    return st, rules


def sink_plant(content: str, variant: str, sinks, ns: Sequence[int], seed: int,
               key_range: Tuple[int, int] = KEY_RANGE) -> Iterator[Tuple]:
    """The second-sink plant (design G10–G15; choice 6), one prompt at a time: yields the stack
    (L, H, n, n) and its truth (the prompt's planted keys, hidden and pure sink rows, partners)."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    Q = len(ns)
    L, H = sinks[0].shape[:2]
    perms = np.stack([rng.permutation(H) for _ in range(L)])
    qa, qb = (int(x) for x in rng.choice(Q, 2, replace=False))
    ka, kb, kb2 = (int(x) for x in rng.choice(np.arange(key_range[0], key_range[1] + 1), 3, replace=False))
    sched = {qa: {l: ka for l in A_LAYERS}, qb: {**{l: kb for l in B_LAYERS}, **{l: kb2 for l in B2_LAYERS}}}
    for q, (sink, n) in enumerate(zip(sinks, ns)):
        dev = sink.device
        R = n - ROW_FROM
        pos = np.arange(ROW_FROM, n)
        heavy = q in sched
        lkey = None if heavy else int(rng.integers(key_range[0], key_range[1] + 1))
        avoid = sorted(set(sched[q].values())) if heavy else [lkey]
        part = content_maps(content, n, L, H, rng, perms, avoid)
        u = _chain(L, R, rng)
        hidden = rng.random(R) < 0.5
        pm = np.where(part >= 0, PLANT, 0.0)
        km = np.zeros((L, H, R))
        key = np.full((L, H, R), -1)
        on = np.zeros((L, H, R), bool)
        eff = part.copy()
        if heavy:
            for l, k in sched[q].items():
                s = _shares(variant, H, rng)
                on[l] = (u[l][None] < s[:, None]) & (pos > k)[None]
                key[l][on[l]] = k
                km[l] = np.where(on[l], np.where(hidden, HIDDEN[0], PURE)[None], 0.0)
                pm[l] = np.where(on[l], np.where(hidden, HIDDEN[1], 0.0)[None], pm[l])
                eff[l] = np.where(on[l] & ~hidden[None], -1, part[l])
        else:
            for l in LIGHT_LAYERS:
                cond = u[l] < LIGHT_COND
                s = rng.uniform(*LIGHT_S, H)
                t = s / (1 - s) * LIGHT_COND / (1 - LIGHT_COND)
                onk = (~cond & (pos > lkey))[None] & (rng.random((H, R)) < t[:, None])
                eff[l][:, ~cond] = -1
                pm[l][:, ~cond] = 0.0
                key[l][onk] = lkey
                km[l][onk] = PURE
        T = lambda x: torch.as_tensor(x, device=dev)
        A = plant2(sink, T(eff), T(pm), T(key), T(km), n)
        truth = {"q": q, "heavy": heavy, "keys": avoid if heavy else [], "light_key": lkey,
                 "hidden_on": T(on & hidden[None]), "pure_on": T(on & ~hidden[None]), "partner": T(part)}
        yield A, truth


# ---------------------------------------------------------------- the gate

def _arm_rule(rules: Sequence[Dict]) -> Dict:
    return {"S_by_prompt": [r["keys"] for r in rules], "S_empty": all(not r["keys"] for r in rules),
            "unconverged": [bool(r["unconverged"]) for r in rules]}


def planted_stages_p(arm: str, sinks, ns: Sequence[int], seed: int) -> Tuple[List[Dict], List[Dict]]:
    """B7's `planted_stages` (same rng sequence) through the rule."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    st, rules = [], []
    for sink, n in zip(sinks, ns):
        L, H = sink.shape[:2]
        part = torch.as_tensor(b7.arm_maps(arm, n, L, H, rng), device=sink.device)
        s = sink_stage(b7.plant(sink, part, n))
        rules.append(s["rule"])
        st.append(s)
    return st, rules


def check_c(res: Dict) -> Dict:
    """G10 (c), G14 (c) (choice 8): every read cell "rerouting", G2's pooled band over them."""
    read = [c for c in res["cells"] if "P" in c]
    desc = [c["description"] for c in res["cells"]]
    rec = {"descriptions": {d: desc.count(d) for d in sorted(set(desc))}, "read": len(read)}
    if not read:
        return {**rec, "pass": False}
    w = np.array([c["joint_rows"] for c in read])
    P, N0 = np.array([c["P"] for c in read]), np.array([c["N0"] for c in read])
    rec["pooled_rel_dev"] = float((w @ P - w @ N0) / (w @ N0))
    rec["pass"] = bool(all(c["description"] == DESCRIPTIONS[2] for c in read) and abs(rec["pooled_rel_dev"]) <= G2_MAX)
    return rec


def mid_not_rerouting(res: Dict) -> Dict:
    cs = [c for c in res["cells"] if c["l"] in MIDDLE]
    read = [c for c in cs if c["description"] != DESCRIPTIONS[0]]
    other = [c for c in read if c["description"] != DESCRIPTIONS[2]]
    return {"cells": len(cs), "read": len(read), "not_rerouting": len(other),
            "share": len(other) / len(cs) if cs else None,
            "n0k_flags": sum(bool(c.get("n0k_flag")) for c in cs)}


def sink_arm(arm: str, sinks, ns: Sequence[int], seed: int, boot: int, thins: Sequence[Optional[int]],
             confound: bool = False, n0k: bool = False, key_range: Tuple[int, int] = KEY_RANGE,
             n_boot: int = N_BOOT) -> Dict:
    """One plant of G10–G15 at one sink step: (a) the sink sets, (d) the hidden and pure rows after
    the drop, (b) B7's instrument without the drop when `confound`, and the dropped run per `thin`."""
    st, b7st, truth_rec = [], [], []
    hid = hid_in = hid_ok = pure = pure_cond = 0
    a_ok = True
    for A, t in sink_plant(CONTENT[arm], VARIANT[arm], sinks, ns, seed, key_range):
        if confound:
            b7st.append(b7.prompt_stage(A))
        s = sink_stage(A)
        del A
        got = s["rule"]["keys"]
        a_ok &= got == t["keys"]
        truth_rec.append({"q": t["q"], "heavy": t["heavy"], "planted": t["keys"], "light_key": t["light_key"],
                          "S": got, "found": s["rule"]["found"], "unconverged": s["rule"]["unconverged"]})
        hi = t["hidden_on"] & s["in"]
        hid += int(t["hidden_on"].sum())
        hid_in += int(hi.sum())
        hid_ok += int((hi & (s["part"] == t["partner"])).sum())
        pure += int(t["pure_on"].sum())
        pure_cond += int((t["pure_on"] & (s["part"] >= 0)).sum())
        st.append(s)
    near = np.concatenate([s["near"].cpu().numpy().ravel() for s in st])
    rec: Dict = {"prompts": truth_rec, "a": bool(a_ok),
                 "d": {"hidden": hid, "hidden_in_after_drop": hid_in, "hidden_on_partner": hid_ok,
                       "share": hid_ok / hid_in if hid_in else None, "pure": pure, "pure_condensed": pure_cond,
                       "pass": bool(hid_in and hid_ok / hid_in >= D_MIN and pure_cond == 0)},
                 "near_miss_heads": int(((near >= NEAR_LO) & (near < SINK_FRAC)).sum()),
                 "near_miss_max": float(np.nanmax(near)) if np.isfinite(near).any() else None}
    if confound:
        cs = b7.run_step(b7st, boot, n_boot=n_boot)["cells"]
        mid = [c for c in cs if c["l"] in MIDDLE]
        nl = sum(c["description"] == DESCRIPTIONS[3] for c in mid)
        rec["b"] = {"depth_local": nl, "cells": len(mid),
                    "descriptions": {d: [c["description"] for c in mid].count(d) for d in DESCRIPTIONS},
                    "pass": bool(nl >= G10_L * len(mid))}
    rec["runs"] = {}
    for thin in thins:
        rec["runs"]["full" if thin is None else "min_entry"] = run_step_p(st, boot, n_boot=n_boot, thin=thin, n0k=n0k)
    return rec


def _strip(rec: Dict) -> Dict:
    """Drop the cell lists from a sink arm's runs once checked (the gate keeps counts)."""
    rec = dict(rec)
    rec["runs"] = {k: {"descriptions": {d: [c["description"] for c in r["cells"]].count(d)
                                        for d in DESCRIPTIONS}, "middle": mid_not_rerouting(r)}
                   for k, r in rec["runs"].items()}
    return rec


def g0p(device: str) -> Tuple[Dict, Dict]:
    """B7's G0 (inputs, shapes, row sums, norms, B3's rows), and B7's gate and records at their
    hashes (choice 16)."""
    rec, real = b7.g0(device)
    gaps = list(rec["gaps"])
    if b7.code_sha() != B7_CODE:
        gaps.append(f"b7_activity.py {b7.code_sha()}, want {B7_CODE}")
    g = json.loads(B7_GATE.read_text()) if B7_GATE.exists() else {}
    if not g.get("pass") or g.get("meta", {}).get("code") != B7_CODE:
        gaps.append(f"B7 gate {B7_GATE}: pass={g.get('pass')}, code {g.get('meta', {}).get('code')}")
    if g.get("G0", {}).get("sha256") != rec["sha256"]:
        gaps.append("input hashes differ from B7's gate")
    for step in STEPS:
        f = B7_READ / f"step{step}.json"
        r = json.loads(f.read_text()) if f.exists() else {}
        if r.get("code") != B7_CODE or any(rec["sha256"].get(k) != v for k, v in r.get("sha256", {}).items()) \
                or not r.get("sha256"):
            gaps.append(f"B7 record {f}: code {r.get('code')}")
    rec["gaps"] = gaps
    rec["b7_gate"], rec["b7_read"] = str(B7_GATE), str(B7_READ)
    rec["pass"] = bool(rec["pass"] and not gaps)
    return rec, real


def g8p(device: str, seed: int) -> Dict:
    """B7's G8 with R1–R5: real attention at G8_STEP, heads G8_HEADS of layers G8_LAYERS
    overwritten by one shared map per prompt with keys outside S_q (choice 15); S_q with and
    without the plant, and only differences from the unplanted sink-excluded run."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    real, plan, S0, S1 = [], [], [], []
    for p in V1:
        A = b7.load_attn(run_dir(b7.G8_STEP, p) / "attentions.npz", device)
        n = A.shape[-1]
        B = A.clone()
        s0 = sink_stage(A)
        del A
        mp = torch.as_tensor(draw_map_avoid(n, rng, s0["rule"]["keys"]), device=device)
        for l in b7.G8_LAYERS:
            for h in b7.G8_HEADS:
                B[l, h] = b7.plant(B[l, h, ROW_FROM:, 0][None, None], mp[None, None], n)[0, 0]
        s1 = sink_stage(B)
        del B
        real.append(s0)
        plan.append(s1)
        S0.append(s0["rule"]["keys"])
        S1.append(s1["rule"]["keys"])
    r0, r1 = run_step_p(real, seed), run_step_p(plan, seed)
    lo, hi = min(b7.G8_LAYERS), max(b7.G8_LAYERS)
    touch = {lo - 1, lo, hi}
    others = [(a, b) for a, b in zip(r0["cells"], r1["cells"]) if a["l"] not in touch]
    dP = [abs(a.get("P", np.nan) - b.get("P", np.nan)) for a, b in others if "P" in a or "P" in b]
    dN1 = [abs(a["N1"] - b["N1"]) for a, b in others if "N1" in a and "N1" in b]
    same_read = all(("P" in a) == ("P" in b) for a, b in others)
    moved = [a["l"] for a, b in others if a["description"] != b["description"]]
    c = r1["cells"][lo]
    hp = [c["heads"]["P"][h] for h in b7.G8_HEADS]
    hs = [sorted(set(np.asarray(c["heads"]["hstar"])[:, h].tolist())) for h in b7.G8_HEADS]
    found = all(set(x) <= set(b7.G8_HEADS) for x in hs)
    same_S = S0 == S1
    ok = (same_S and c["description"] in DESCRIPTIONS[3:5] and min(hp) >= 0.95 and found and same_read
          and (max(dP) if dP else 0.0) <= 0.01)
    return {"pass": bool(ok), "S_q": dict(zip(V1, S0)), "S_q_same_with_plant": same_S,
            "S_q_planted": dict(zip(V1, S1)) if not same_S else None,
            "found": {p: s["rule"]["found"] for p, s in zip(V1, real)},
            "unconverged": {p: s["rule"]["unconverged"] for p, s in zip(V1, real)},
            "cell": f"{lo} -> {hi}", "description": c["description"],
            "planted_heads_P": hp, "planted_heads_joint": [c["heads"]["joint"][h] for h in b7.G8_HEADS],
            "planted_heads_hstar": hs, "other_cells_max_abs_dP_code_check": max(dP) if dP else 0.0,
            "other_cells_same_read": same_read, "other_cells": len(dP),
            "other_cells_max_abs_dN1": max(dN1) if dN1 else None, "other_cells_description_changed_at": moved}


def cmd_gate(a) -> int:
    torch = _torch()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tick = lambda k, v: print(f"{k} pass={v} ({time.time() - t0:.0f}s)", flush=True)
    meta = {"code": code_sha(), "b7_code": b7.code_sha(), "prompts_hash": prompts_hash(), "runs": RUNS,
            "device": a.device, "torch": torch.__version__, "numpy": np.__version__,
            "constants": {"SINK_FRAC": SINK_FRAC, "SINK_MIN": SINK_MIN, "SINK_PASSES": SINK_PASSES,
                          "KEY_RANGE": KEY_RANGE, "SINK_S": SINK_S, "SPARSE_HEADS": SPARSE_HEADS, "KEEP": KEEP,
                          "HIDDEN": HIDDEN, "PURE": PURE, "LIGHT": [LIGHT_COND, LIGHT_S], "G10_L": G10_L,
                          "D_MIN": D_MIN, "N_BOOT": N_BOOT, "MIN_JOINT": MIN_JOINT, "SINK_STEPS": SINK_STEPS}}
    gate: Dict = {"meta": meta}
    gate["G0"], real = g0p(a.device)
    tick("G0", gate["G0"]["pass"])
    sinks, ns = real["sinks"], real["ns"]
    for arm in ("G1", "G2", "G3", "G4", "G5", "G6"):
        runs = {}
        for s in SINK_STEPS:
            st, rules = planted_stages_p(arm, sinks[s], ns, seed_for("B7", arm, s))
            boot = seed_for("B7", arm, s, "boot")
            variants = [(str(s), None)] + ([(f"{s}_min_entry", MIN_JOINT)] if arm in ("G1", "G3") else [])
            for name, thin in variants:
                r = b7.arm_check(arm, run_step_p(st, boot, thin=thin), min_entry=thin is not None)
                r.update(_arm_rule(rules))
                r["pass"] = bool(r["pass"] and r["S_empty"])
                runs[name] = r
            del st
        gate[arm] = {"pass": all(r["pass"] for r in runs.values()), "by_sink_step": runs}
        tick(arm, gate[arm]["pass"])
    g7 = {v: {"false": 0, "read": 0, "unread": 0, "loo_rows_min": math.inf} for v in ("full", "min_entry")}
    s_nonempty = 0
    for s in SINK_STEPS:
        for k in range(G7_SEEDS):
            st, rules = planted_stages_p("G7", sinks[s], ns, seed_for("B7", "G7", s, k))
            s_nonempty += sum(bool(r["keys"]) for r in rules)
            for v, thin in (("full", None), ("min_entry", MIN_JOINT)):
                for c in run_step_p(st, seed_for("B7", "G7", s, k, "boot"), thin=thin)["cells"]:
                    if c["description"] == DESCRIPTIONS[0]:
                        g7[v]["unread"] += 1
                    else:
                        g7[v]["read"] += 1
                        g7[v]["false"] += c["description"] != DESCRIPTIONS[2]
                        g7[v]["loo_rows_min"] = min(g7[v]["loo_rows_min"], c["loo_rows_min"])
            del st
    for v in g7.values():
        v["rate"] = v["false"] / v["read"] if v["read"] else None
        v["pass"] = bool(v["read"] and v["rate"] <= G7_MAX)
    gate["G7"] = {"pass": all(v["pass"] for v in g7.values()) and s_nonempty == 0,
                  "prompts_with_a_sink": s_nonempty, **g7}
    tick("G7", gate["G7"]["pass"])
    for arm, kind, chk in (("G1d", "G1d", "G1"), ("G3d", "G3d", "G3")):
        runs = {}
        for s in SINK_STEPS:
            st, rules = heads_differ_stages(kind, sinks[s], ns, seed_for("B7p", arm, s))
            boot = seed_for("B7p", arm, s, "boot")
            for name, thin in ((str(s), None), (f"{s}_min_entry", MIN_JOINT)):
                r = b7.arm_check(chk, run_step_p(st, boot, thin=thin), min_entry=thin is not None)
                r.update(_arm_rule(rules))
                r["pass"] = bool(r["pass"] and r["S_empty"])
                runs[name] = r
            del st
        gate[arm] = {"pass": all(r["pass"] for r in runs.values()), "by_sink_step": runs}
        tick(arm, gate[arm]["pass"])
    for arm in ("G10", "G11", "G12", "G14", "G15"):
        runs = {}
        for s in SINK_STEPS:
            thins = (None, MIN_JOINT) if arm in ("G11", "G12") else (None,)
            r = sink_arm(arm, sinks[s], ns, seed_for("B7p", arm, s), seed_for("B7p", arm, s, "boot"), thins,
                         confound=arm in ("G10", "G14"), n0k=True)
            if arm in ("G10", "G14"):
                r["c"] = check_c(r["runs"]["full"])
                r["pass"] = bool(r["a"] and r["c"]["pass"] and r["d"]["pass"] and (arm == "G14" or r["b"]["pass"]))
            elif arm in ("G11", "G12"):
                chk = "G1" if arm == "G11" else "G3"
                r["full"] = b7.arm_check(chk, r["runs"]["full"])
                r["min_entry"] = b7.arm_check(chk, r["runs"]["min_entry"], min_entry=True)
                r["pass"] = bool(r["a"] and r["d"]["pass"] and r["full"]["pass"] and r["min_entry"]["pass"])
            runs[str(s)] = _strip(r)
        gate[arm] = {"by_sink_step": runs}
        if arm != "G15":
            gate[arm]["pass"] = all(r["pass"] for r in runs.values())
        tick(arm, gate[arm].get("pass", "recorded"))
    g13 = {v: {"false": 0, "read": 0, "unread": 0} for v in ("full", "min_entry")}
    a_fail = 0
    for s in SINK_STEPS:
        for k in range(G7_SEEDS):
            r = sink_arm("G13", sinks[s], ns, seed_for("B7p", "G13", s, k), seed_for("B7p", "G13", s, k, "boot"),
                         (None, MIN_JOINT))
            a_fail += not r["a"]
            for v, res in r["runs"].items():
                for c in res["cells"]:
                    if c["description"] == DESCRIPTIONS[0]:
                        g13[v]["unread"] += 1
                    else:
                        g13[v]["read"] += 1
                        g13[v]["false"] += c["description"] != DESCRIPTIONS[2]
    for v in g13.values():
        v["rate"] = v["false"] / v["read"] if v["read"] else None
        v["pass"] = bool(v["read"] and v["rate"] <= G7_MAX)
    gate["G13"] = {"pass": all(v["pass"] for v in g13.values()), "plants_with_a_wrong_sink_set": a_fail, **g13}
    tick("G13", gate["G13"]["pass"])
    gate["G8"] = g8p(a.device, seed_for("B7p", "G8"))
    tick("G8", gate["G8"]["pass"])
    gate["G9"] = b7.g9()
    gate["pass"] = all(gate[k]["pass"] for k in ARMS)
    gate["seconds"] = round(time.time() - t0)
    dump(out / "gate.json", gate)
    print(json.dumps({k: gate[k]["pass"] for k in ARMS} | {"pass": gate["pass"]}))
    torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- the read (unit 2)

def check_gate_p(gate: Dict) -> None:
    """Refuse unless the gate passed on this code, B7's code, these prompts and these runs."""
    m = gate.get("meta", {})
    want = {"code": code_sha(), "b7_code": b7.code_sha(), "prompts_hash": prompts_hash(),
            "runs": json.loads(json.dumps(RUNS))}
    bad = [f"{k} {m.get(k)!r} vs {v!r}" for k, v in want.items() if m.get(k) != v]
    if not gate.get("pass") or bad or not gate.get("G0", {}).get("sha256"):
        raise B7Error(f"gate: pass={gate.get('pass')}; {'; '.join(bad) or 'no input hashes'}"
                      " (refuse: no real cell before a passing gate on this code and these inputs)")


def _token_flags(ids: Sequence[int], tok) -> Tuple[Optional[int], Optional[int]]:
    """Choice 14: the prompt's first `.` and first newline positions (≥ 1)."""
    dot = nl = None
    for k in range(1, len(ids)):
        t = tok.decode(ids[k])
        if dot is None and t.strip() == ".":
            dot = k
        if nl is None and "\n" in t:
            nl = k
    return dot, nl


def cmd_read(a) -> int:
    torch = _torch()
    from transformers import AutoTokenizer
    from core.config import PROMPTS
    gate = json.loads(Path(a.gate).read_text())
    check_gate_p(gate)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    ids = {p: tok(PROMPTS[p])["input_ids"] for p in V1}
    flags = {p: _token_flags(ids[p], tok) for p in V1}
    steps = [int(s) for s in a.steps.split(",")] if a.steps else list(STEPS)
    for step in steps:
        t0 = time.time()
        st, ds, hashes, sinks, b5 = [], [], {}, {}, {}
        alt = {f: [] for f in FRAC_BESIDE}
        for p in V1:
            d = run_dir(step, p)
            A = b7.load_attn(d / "attentions.npz", a.device, hashes)
            X = b7.load_states(d / "activations.npz", a.device, hashes)
            part0 = b7.partners(A)[0] if p in BESIDE5["prompts"] else None
            for f in FRAC_BESIDE:
                alt[f].append(sink_stage(A.clone(), frac=f))
            s = sink_stage(A, X, soft=True)
            del A, X
            ds.append(s.pop("d"))
            S = s["rule"]["keys"]
            dot, nl = flags[p]
            sinks[p] = {"keys": [{"key": k, "token": tok.decode(ids[p][k]) if k < len(ids[p]) else None,
                                  "first_dot": k == dot, "first_newline": k == nl} for k in S],
                        "found": s["rule"]["found"], "unconverged": s["rule"]["unconverged"],
                        "beside_frac": {str(f): alt[f][-1]["rule"]["keys"] for f in FRAC_BESIDE}}
            near = s["near"].cpu().numpy()
            sinks[p]["near_miss"] = {"heads_in_band": int(((near >= NEAR_LO) & (near < SINK_FRAC)).sum()),
                                     "share": near.tolist()}
            if part0 is not None:
                kk = torch.as_tensor(S, device=part0.device, dtype=part0.dtype)
                b5[p] = {}
                for l, h in BESIDE5["heads"]:
                    v = part0[l, h]
                    v = v[v >= 0]
                    b5[p][f"L{l}H{h}"] = float(torch.isin(v, kk).double().mean()) if v.numel() else None
            st.append(s)
        moved = [k for k, v in hashes.items() if gate["G0"]["sha256"].get(k) != v]
        if moved:
            raise B7Error(f"step {step}: {len(moved)} input files differ from the gate's ({moved[0]}, ...)")
        seed = seed_for("B7", "read", step)                     # choice 4: B7's draws
        res = run_step_p(st, seed, beside=True, n0k=True)
        changed = {}
        for f in FRAC_BESIDE:
            ra = run_step_p(alt[f], seed)
            changed[str(f)] = sum(x["description"] != y["description"] for x, y in zip(res["cells"], ra["cells"]))
        b7rec = json.loads((B7_READ / f"step{step}.json").read_text())
        if b7rec.get("code") != B7_CODE:
            raise B7Error(f"B7 record step {step}: code {b7rec.get('code')}")
        for c, c7 in zip(res["cells"], b7rec["cells"]):
            c["b7_description"], c["b7_joint_rows"] = c7["description"], c7["joint_rows"]
            c["rows_kept_of_b7"] = c["joint_rows"] / c7["joint_rows"] if c7["joint_rows"] else None
            c["w_quartiles"] = w_quartiles(st, c)
        rec = {"step": step, "code": code_sha(), "b7_code": b7.code_sha(), "prompts_hash": prompts_hash(),
               "gate": str(a.gate), "runs": {p: str(run_dir(step, p)) for p in V1}, "sha256": hashes,
               "n": [s["n"] for s in st], "sinks": sinks, **res, "D": b7.d_of(ds), "alpha": b7.alpha_beside(step),
               "beside_frac_changed": changed, "beside_L6H4_L15H7_on_S": b5}
        dump(out / f"step{step}.json", rec)
        print(f"step {step}: {time.time() - t0:.0f}s", flush=True)
        del st, alt
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
