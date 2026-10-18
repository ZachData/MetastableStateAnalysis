"""B3, routing, unit 1 (`trackB_circuit/design-B3.md`, frozen 2026-10-18, Blocked 65 (a)): α per
head over the 18 steps from stored attention, head types per step, the gate G0–G3, R1, R2, C3′,
A3's descriptive row and β beside.

**Rows** (design "Objects"): query i ≥ 16 of each stored `attentions.npz` (A1a's `RUNS`, the 7
scored v1 prompts), position 0 dropped as key and the row renormalised over keys 1..i;
``N_eff,i = S² / Σ_j A_ij²`` with ``S = Σ_{j≥1} A_ij`` (the renormalised row's inverse
participation, without forming it); ``w_i = 1 − A_i0``; rows with ``w_i < 0.05`` are out. α is the
weighted least-squares slope of log N_eff on log i, pooled over prompts. The logit spread is the
std of log Ã_ij over keys with Ã_ij > 0 (float32 underflow leaves exact zeros; their count is
kept beside).

**Head types** (one forward per step, GPU float32, 7d's probe: 16 sequences, ``N_REP`` = 96, the
`wide` arm, ``EVAL_SEED``): previous-token score as `prev_token_profile.prev_token_scores`, the
induction score at offset 0 as `fv_score.induction_scores` (query ``N_REP + j`` → key j) and at
+1 (→ key j + 1) beside; the first 8 sequences' offset-0 score for G0 against `fv_score.json`
(8 sequences: `batch` draws sequence by sequence, so they are the 16's first 8). α on the probe's
rows is kept beside.

**Choices the design left open, made before any output:** a head with fewer than ``MIN_ROWS``
in-rows at a step has no α there; its percentile is ½ and its count is reported (none expected).
R2's onset needs the next grid step, so 143000 cannot be an onset. C3′'s null draws one pool head
per (member, step) independently. G2's fake members for C3′ are drawn from each member's pool at
143000, and their own pools are the 5 non-members nearest the fake's type score (for `L10H7`'s
fake, layer 10's other non-members). G3 plants at the members' formation steps; its count must be
5, and its rank is read only if R2's attainable rank is ≥ 0.99 (otherwise R2 is descriptive by the
design and no rank can pass). A3 reads the attention of Phase 10's label source (its own Stage 0
runs, not A1a's: the two sets differ in run directory at every step ≥ 512), received attention
on rows i ≥ 1 renormalised over keys 1..i, divided by uniform routing's ``Σ_{i ≥ j} 1/i``,
log(member mean / rest mean) on the column's domain without position 0. M1's β is stored per
layer only (`a2_beta_2026-10-16/refit_step143000.json`), so β beside is `fe_full` per head and
M1 per layer.

NUMERICS: stored attention (float32) is cast to float64 on the GPU; head-type forwards float32 on
the GPU (as B0), scores accumulated in float64. Statistics on the CPU in float64.

Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b3_route rows  --out <dir> [--steps 0,512]
    python -m trackB_circuit.b3_route types --out <dir> [--steps ...]
    python -m trackB_circuit.b3_route read  --out <dir>
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

from trackA_force.a1a import RUNS, V1, prompts_hash, run_dir
from trackB_circuit.b2 import draw_sets, head_index, head_name, layer_pools, percentiles, rank_of
from trackB_circuit.b2a import N_HEADS, N_LAYERS, NAMED, REPO, STEPS, parse_head, seed_for

DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
LABELS = DATA / "p10/reread_r9_2026-10-08/labels"
B2_TABLE = DATA / "b2/2026-10-18-table"
B2_CODE = "038b04f5c7cd"
A1A_RECORDS = DATA / "trackA/a1a_batch_2026-10-15/records"
A2_BETA = DATA / "trackA/a2_beta_2026-10-16/refit_step143000.json"
PREV_FILE = DATA / "analysis/prev_token_profile.json"
FV_FILE = DATA / "analysis/fv_score.json"
CATALOG = DATA / "analysis/redundancy_catalog.json"

MEMBERS = NAMED                            # L5H2, L7H8, L12H5, L8H6, L11H14, L10H7
TYPED = {"L5H2": "prev", "L7H8": "ind", "L12H5": "ind", "L8H6": "ind", "L11H14": "ind"}
INDUCTION = ("L7H8", "L8H6", "L11H14", "L12H5")
N_POP = N_LAYERS * N_HEADS
ROW_FROM = 16
W_MIN = 0.05                               # placed (design)
MIN_ROWS = 10                              # placed (module docstring)
NEFF_BAND = (192, 320)
STAT_FROM = 512
STAT_STEPS = tuple(s for s in STEPS if s >= STAT_FROM)
N_NULL, N_DRAW = 10_000, 200
RANK_PASS = 0.99
ONSET_Q = 0.10                             # placed (design): R2's onset quantile
FORM_FRAC = 0.5                            # placed (design): formation at half the maximum
NEAR_K = 5                                 # placed (design): C3′'s pool size
PROSE = ("camus_letranger", "homer_iliad", "paper_excerpt", "sullivan_ballou", "wiki_paragraph")
N_PROBE, N_FV = 16, 8
G0_ROWSUM, G0_TYPES = 1e-5, 1e-4          # placed (design G0)
G0_TYPE_STEPS = (4000, 16000, 143000)
G1_BANDS = {"uniform": (0.97, 1.0), "two": (-0.02, 0.02), "gauss1": (0.85, None)}   # placed (design G1)
G2_MAX = 0.02                              # placed (design G2)
G3_BESIDE = 50                             # placed (module docstring): sets beside G3's primary
ARMS = ("G0", "G1", "G2", "G3")


class B3Error(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackB_circuit/b3_route.py").read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- pure: rows and slopes

def row_stats(A, row_from: int = ROW_FROM) -> Dict[str, np.ndarray]:
    """Per-row statistics of one head stack ``A`` (…, n, n), a float64 torch tensor or array:
    sink ``A_i0``, ``S`` (mass on keys 1..i), ``N_eff``, the log-weight spread and the count of
    exact zeros among keys 1..i, for rows i = row_from..n−1."""
    import torch
    A = torch.as_tensor(A, dtype=torch.float64)
    n = A.shape[-1]
    R = A[..., row_from:, :]
    K = R[..., 1:]
    S = K.sum(-1)
    neff = S ** 2 / (K ** 2).sum(-1)
    i = torch.arange(row_from, n, device=A.device)[:, None]
    j = torch.arange(1, n, device=A.device)[None, :]
    valid = j <= i
    pos = valid & (K > 0)
    lg = torch.where(pos, torch.log(torch.where(pos, K, 1.0) / S[..., None]), torch.zeros((), dtype=A.dtype,
                                                                                          device=A.device))
    cnt = pos.sum(-1).to(A.dtype)
    m = lg.sum(-1) / cnt
    var = (torch.where(pos, lg - m[..., None], 0.0) ** 2).sum(-1) / cnt     # two-pass
    return {"sink": R[..., 0].cpu().numpy(), "S": S.cpu().numpy(), "neff": neff.cpu().numpy(),
            "spread": torch.sqrt(torch.clamp(var, min=0)).cpu().numpy(),
            "zeros": (valid & ~(K > 0)).sum(-1).cpu().numpy().astype(np.int32)}


def wls_sums(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(…, 5) sums Σw, Σwx, Σwy, Σwx², Σwxy over the last axis (x broadcasts)."""
    x = np.broadcast_to(x, y.shape)
    return np.stack([w.sum(-1), (w * x).sum(-1), (w * y).sum(-1), (w * x * x).sum(-1),
                     (w * x * y).sum(-1)], -1)


def slope_of(s: np.ndarray, n_rows: Optional[np.ndarray] = None, min_rows: int = MIN_ROWS) -> np.ndarray:
    """The WLS slope from `wls_sums`' last axis; NaN where fewer than `min_rows` rows."""
    sw, sx, sy, sxx, sxy = np.moveaxis(s, -1, 0)
    den = sw * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (sw * sxy - sx * sy) / den
    b = np.where(den > 0, b, np.nan)
    if n_rows is not None:
        b = np.where(n_rows >= min_rows, b, np.nan)
    return b


def head_weights(sink: np.ndarray, w_min: float = W_MIN) -> np.ndarray:
    w = 1.0 - sink
    return np.where(w >= w_min, w, 0.0)


def alpha_of(rows: Dict[str, Dict[str, np.ndarray]], weighted: bool = True, key: str = "neff") -> np.ndarray:
    """α per head (flattened 384) pooled over the prompts in `rows` {prompt: row_stats + "n"}."""
    tot, cnt = 0.0, 0
    for r in rows.values():
        x = np.log(r["n"])
        y = np.log(r[key]) if key == "neff" else r[key]
        w = head_weights(r["sink"]) if weighted else np.ones_like(r["sink"])
        tot = tot + wls_sums(x, y, w)
        cnt = cnt + (w > 0).sum(-1)
    return slope_of(tot, cnt).reshape(-1)


def pctl_nan(x: np.ndarray) -> np.ndarray:
    """B2's percentile in the population; a head without a value gets ½."""
    p = np.full(x.shape, 0.5)
    ok = np.isfinite(x)
    p[ok] = percentiles(x[ok])
    return p


def onset_index(alpha: np.ndarray, q: np.ndarray) -> Optional[int]:
    """First grid index k ≥ 1 with α_k ≤ q_k and α_{k+1} ≤ q_{k+1} (R2's onset), else None."""
    hit = np.isfinite(alpha) & (alpha <= q)
    for k in range(1, len(alpha) - 1):
        if hit[k] and hit[k + 1]:
            return k
    return None


def formation_index(score: np.ndarray, frac: float = FORM_FRAC) -> int:
    return int(np.flatnonzero(score >= frac * score.max())[0])


def r2_hits(A: np.ndarray, heads: Sequence[int], forms: Sequence[int], q: np.ndarray) -> np.ndarray:
    """A (18, 384) α grid; whether each head's onset is within one grid step of its formation."""
    out = []
    for h, f in zip(heads, forms):
        k = onset_index(A[:, h], q)
        out.append(k is not None and abs(k - f) <= 1)
    return np.array(out, bool)


def r2_null(A: np.ndarray, pools: Sequence[np.ndarray], forms: Sequence[int], q: np.ndarray,
            n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """(null counts over n layer-matched sets, each pool's exact hit rate)."""
    hit = [r2_hits(A, p, [f] * len(p), q) for p, f in zip(pools, forms)]
    rates = np.array([h.mean() for h in hit])
    g = np.random.default_rng(seed)
    counts = sum(h[g.integers(0, len(h), n)].astype(int) for h in hit)
    return counts, rates


def nearest_pool(score: np.ndarray, target: float, exclude: Sequence[int], k: int = NEAR_K) -> np.ndarray:
    d = np.abs(score - target)
    d[list(exclude)] = np.inf
    return np.argsort(d, kind="stable")[:k]


def c3_null(P: np.ndarray, pools: Sequence[Sequence[np.ndarray]], n: int, seed: int) -> np.ndarray:
    """P (steps, 384) percentiles; pools[m][s] the pool of member m at step s. Null T's."""
    g = np.random.default_rng(seed)
    tot = np.zeros(n)
    cells = 0
    for pm in pools:
        for s, pool in enumerate(pm):
            tot += np.abs(P[s, pool[g.integers(0, len(pool), n)]] - 0.5)
            cells += 1
    return tot / cells


def received_ratio(A, lab: np.ndarray) -> np.ndarray:
    """A3: per head of one layer stack ``A`` (H, n, n), log(member mean / rest mean) of received
    attention over rows i ≥ 1 renormalised on keys 1..i, divided by uniform routing's Σ_{i≥j} 1/i;
    ``lab`` the c3x labels over every position (−2 outside the domain, −1 rest, ≥ 0 members)."""
    import torch
    A = torch.as_tensor(A, dtype=torch.float64)
    n = A.shape[-1]
    K = A[:, 1:, 1:]
    K = K / K.sum(-1, keepdim=True).clamp_min(1e-300)
    rec = K.sum(1).cpu().numpy()                                   # (H, n−1): keys 1..n−1
    base = np.cumsum((1.0 / np.arange(1, n))[::-1])[::-1]          # Σ_{i=j}^{n−1} 1/i
    c = rec / base
    lab = np.asarray(lab)[1:]
    mem, rest = lab >= 0, lab == -1
    if not mem.any() or not rest.any():
        return np.full(A.shape[0], np.nan)
    return np.log(c[:, mem].mean(1) / c[:, rest].mean(1))


def spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    return float(spearmanr(a[ok], b[ok]).statistic)


# ---------------------------------------------------------------- rows: stored attention

def _load_attn(path: Path, device: str):
    import torch
    with np.load(path) as z:
        a = z["attentions"]
    return torch.from_numpy(a).to(device=device, dtype=torch.float64)


def rows_step(step: int, out: Path, device: str) -> Dict:
    from tools.run.p10_label_source import LabelSourceError, load_column, readable
    rec: Dict = {"step": step, "code": code_sha(), "prompts": {}, "a3": {}, "entropy": {}}
    arrays = {}
    for p in V1:
        d = run_dir(step, p)
        A = _load_attn(d / "attentions.npz", device)
        L, H, n, n2 = A.shape
        if (L, H) != (N_LAYERS, N_HEADS) or n != n2:
            raise B3Error(f"{d}: attention shape {tuple(A.shape)}")
        rowsum = float((A.sum(-1) - 1).abs().max())
        parts = [row_stats(A[l]) for l in range(L)]
        for k in parts[0]:
            arrays[f"{p}/{k}"] = np.stack([q[k] for q in parts]).astype(np.float64 if k != "zeros" else np.int32)
        arrays[f"{p}/n"] = np.arange(ROW_FROM, n, dtype=np.float64)
        rec["prompts"][p] = {"run": str(d), "n": int(n), "rowsum_max_dev": rowsum}
        sk = json.loads((d / "sinkhorn.json").read_text())["layers"]
        rec["entropy"][p] = [sk[l]["attention_entropy_per_head"] for l in range(N_LAYERS)]
        del A
        if step >= STAT_FROM:
            lab_step = json.loads((LABELS / f"step{step}.json").read_text())
            src_run = Path(lab_step["prompts"][p]["stage0_run"])
            A = _load_attn(src_run / "attentions.npz", device)
            a3 = np.full((N_LAYERS, N_HEADS), np.nan)
            why = {}
            for l in range(1, N_LAYERS):
                try:
                    pos, lab = load_column(LABELS, f"step{step}", p, l, "c3x", {f"step{step}": lab_step})
                except LabelSourceError as e:
                    why[l] = str(e)
                    continue
                if not readable(lab):
                    why[l] = "unreadable"
                    continue
                full = np.full(A.shape[-1], -2, int)
                full[pos] = lab
                a3[l] = received_ratio(A[l], full)
            rec["a3"][p] = {"run": str(src_run), "value": a3.tolist(), "skipped": why}
            del A
    np.savez_compressed(out / "rows" / f"step{step}.npz", **arrays)
    (out / "rows" / f"step{step}.json").write_text(json.dumps(rec) + "\n")
    return rec


def load_rows(out: Path, step: int) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict]:
    rec = json.loads((out / "rows" / f"step{step}.json").read_text())
    with np.load(out / "rows" / f"step{step}.npz") as z:
        rows = {p: {k: z[f"{p}/{k}"] for k in ("sink", "S", "neff", "spread", "zeros", "n")} for p in V1}
    return rows, rec


# ---------------------------------------------------------------- types: one forward per step

def types_step(step: int, out: Path, device: str) -> Dict:
    import torch
    from p7d_redundancy.member_formation_curves import batch
    from tools.run.induction_rank_sweep import EVAL_SEED, N_REP, PROBE_ARMS
    from trackB_circuit.b0_gate import load
    model, _ = load(step, device, torch.float32)
    ids = batch(np.random.default_rng(EVAL_SEED), N_PROBE, *PROBE_ARMS["wide"]).to(device)
    T = ids.shape[1]
    q = torch.arange(N_REP, 2 * N_REP, device=device)
    prev = np.zeros(N_POP)
    ind0, ind1, ind0_fv = np.zeros(N_POP), np.zeros(N_POP), np.zeros(N_POP)
    sums = np.zeros((N_POP, 5))
    cnt = np.zeros(N_POP)
    sink = np.zeros(N_POP)
    i_all = torch.arange(1, T, device=device)
    with torch.no_grad():
        for c in range(0, N_PROBE, 4):
            o = model(ids[c:c + 4], output_attentions=True)
            for l, att in enumerate(o.attentions):
                a = att.double()                                    # (B, H, T, T)
                sl = slice(l * N_HEADS, (l + 1) * N_HEADS)
                prev[sl] += a[:, :, i_all, i_all - 1].sum((0, 2)).cpu().numpy()
                ind0[sl] += a[:, :, q, q - N_REP].sum((0, 2)).cpu().numpy()
                ind1[sl] += a[:, :, q, q - N_REP + 1].sum((0, 2)).cpu().numpy()
                sink[sl] += a[:, :, ROW_FROM:, 0].sum((0, 2)).cpu().numpy()
                if c < N_FV:
                    ind0_fv[sl] += a[:, :, q, q - N_REP].sum((0, 2)).cpu().numpy()
                r = row_stats(a.transpose(0, 1).reshape(N_HEADS, -1, T, T).flatten(0, 1))
                r = {k: v.reshape(N_HEADS, a.shape[0], -1) for k, v in r.items()}
                w = head_weights(r["sink"])
                x = np.log(np.arange(ROW_FROM, T, dtype=float))
                sums[sl] += wls_sums(x, np.log(r["neff"]), w).sum(1)
                cnt[sl] += (w > 0).sum((1, 2))
            del o
    rec = {"step": step, "code": code_sha(), "device": device, "torch": torch.__version__,
           "n_seqs": N_PROBE, "n_rep": N_REP, "seed": EVAL_SEED,
           "prev": (prev / (N_PROBE * (T - 1))).tolist(), "ind0": (ind0 / (N_PROBE * N_REP)).tolist(),
           "ind1": (ind1 / (N_PROBE * N_REP)).tolist(), "ind0_fv": (ind0_fv / (N_FV * N_REP)).tolist(),
           "alpha_probe": slope_of(sums, cnt).tolist(),
           "sink_probe": (sink / (N_PROBE * (T - ROW_FROM))).tolist()}   # mean A_i0, rows i ≥ 16 (beside)
    (out / "types" / f"step{step}.json").write_text(json.dumps(rec) + "\n")
    return rec


# ---------------------------------------------------------------- read: gate, then the readings

def load_all(out: Path) -> Dict:
    rows, recs, types = {}, {}, {}
    for s in STEPS:
        rows[s], recs[s] = load_rows(out, s)
        types[s] = json.loads((out / "types" / f"step{s}.json").read_text())
    return {"rows": rows, "recs": recs, "types": types}


def g0(D: Dict) -> Dict:
    dev = max(r["prompts"][p]["rowsum_max_dev"] for r in D["recs"].values() for p in V1)
    shapes = all(D["rows"][s][p]["neff"].shape[:2] == (N_LAYERS, N_HEADS) for s in STEPS for p in V1)
    stored = json.loads(PREV_FILE.read_text())["per_step"]
    prev_d, ind_d = [], []
    for s in G0_TYPE_STEPS:
        h = stored[str(s)]["heads"]
        t = D["types"][s]
        prev_d.append(max(abs(t["prev"][i] - h[head_name(i)]["prev_token"]) for i in range(N_POP)))
        ind_d.append(max(abs(t["ind0"][i] - h[head_name(i)]["induction"]) for i in range(N_POP)))
    fv = json.loads(FV_FILE.read_text())
    fv_d = []
    for s in fv["steps"]:
        for m in fv["members"]:
            fv_d.append(abs(D["types"][s]["ind0_fv"][head_index(m)] -
                            fv["per_step"][str(s)]["heads"][m]["induction_score"]))
    rec = {"rowsum_max_dev": dev, "shapes": shapes, "prev_max_dev": max(prev_d), "ind_max_dev": max(ind_d),
           "fv_ind_max_dev": max(fv_d), "prev_by_step": dict(zip(map(str, G0_TYPE_STEPS), prev_d)),
           "ind_by_step": dict(zip(map(str, G0_TYPE_STEPS), ind_d))}
    rec["pass"] = bool(shapes and dev <= G0_ROWSUM and max(prev_d) <= G0_TYPES and max(ind_d) <= G0_TYPES
                       and max(fv_d) <= G0_TYPES)
    return rec


def g1(D: Dict) -> Dict:
    """Planted α on one non-member head's real rows (its n_i and sink kept)."""
    mem = {head_index(m) for m in MEMBERS}
    g = np.random.default_rng(seed_for("b3-g1"))
    h = int(g.choice([i for i in range(N_POP) if i not in mem]))
    l, hh = divmod(h, N_HEADS)
    out = {"head": head_name(h)}
    for s in (512, 143000):
        rows = D["rows"][s]
        res = {}
        for kind in G1_BANDS:
            tot, cnt = 0.0, 0
            for p, r in rows.items():
                n = r["n"]
                w = head_weights(r["sink"][l, hh])
                if kind == "uniform":
                    y = np.log(n)
                elif kind == "two":
                    y = np.full_like(n, np.log(2.0))
                else:
                    gg = np.random.default_rng(seed_for("b3-g1", s, p))
                    y = np.array([np.log(np.exp(z).sum() ** 2 / np.exp(2 * z).sum())
                                  for z in (gg.standard_normal(int(i)) for i in n)])
                tot = tot + wls_sums(np.log(n), y, w)
                cnt = cnt + (w > 0).sum()
            a = float(slope_of(np.asarray(tot), np.asarray(cnt)))
            lo, hi = G1_BANDS[kind]
            res[kind] = {"alpha": a, "pass": bool(a >= lo and (hi is None or a <= hi))}
        out[str(s)] = res
    out["pass"] = all(v["pass"] for s in ("512", "143000") for v in out[s].values())
    return out


def alpha_grid(D: Dict, weighted: bool = True, key: str = "neff") -> np.ndarray:
    return np.stack([alpha_of(D["rows"][s], weighted, key) for s in STEPS])     # (18, 384)


def type_series(D: Dict, key: str) -> np.ndarray:
    return np.stack([np.asarray(D["types"][s][key]) for s in STEPS])             # (18, 384)


def member_offset(D: Dict) -> Dict:
    t = D["types"][143000]
    votes = {m: ("0" if t["ind0"][head_index(m)] >= t["ind1"][head_index(m)] else "+1") for m in INDUCTION}
    zero = sum(v == "0" for v in votes.values())
    return {"votes": votes, "offset": "0" if zero >= 2 else "+1",
            "ind0": {m: t["ind0"][head_index(m)] for m in INDUCTION},
            "ind1": {m: t["ind1"][head_index(m)] for m in INDUCTION}}


def type_scores(D: Dict, offset: str) -> Dict[str, np.ndarray]:
    return {"prev": type_series(D, "prev"), "ind": type_series(D, "ind0" if offset == "0" else "ind1")}


def formations(ts: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {m: formation_index(ts[TYPED[m]][:, head_index(m)]) for m in TYPED}


def r1_stat(A: np.ndarray, heads: Sequence[int]) -> Tuple[float, np.ndarray]:
    """T over `heads` and the 9 steps ≥ 512, and D (384) for the null."""
    idx = [STEPS.index(s) for s in STAT_STEPS]
    P = np.stack([pctl_nan(A[k]) for k in idx])
    Dv = np.abs(P - 0.5).mean(0)
    return float(Dv[list(heads)].mean()), Dv


def r1_rank(A: np.ndarray, heads: Sequence[int], seed: int, n: int = N_NULL) -> Dict:
    T, Dv = r1_stat(A, heads)
    null = Dv[draw_sets(layer_pools(heads), n, seed)].mean(1)
    return {"T": T, "rank": rank_of(T, null), "null_q": np.quantile(null, [0.05, 0.5, 0.95, 0.99]).tolist()}


def q10(A: np.ndarray) -> np.ndarray:
    return np.array([np.nanquantile(A[k], ONSET_Q) for k in range(A.shape[0])])


def r2_read(A: np.ndarray, heads: Sequence[int], forms: Sequence[int], seed: int, n: int = N_NULL) -> Dict:
    q = q10(A)
    pools = layer_pools(heads)
    counts, rates = r2_null(A, pools, forms, q, n, seed)
    attain = float(1 - np.prod(rates))
    hits = r2_hits(A, heads, forms, q)
    c = int(hits.sum())
    return {"count": c, "rank": rank_of(c, counts), "attainable": attain, "rates": rates.tolist(),
            "hits": hits.tolist(), "onsets": [onset_index(A[:, h], q) for h in heads]}


def type_pools(ts: Dict[str, np.ndarray], refs: Dict[str, int], exclude: Sequence[int]) -> List[List[np.ndarray]]:
    """Per member name (in `refs`, name → the head whose score centres the pool), per stat step."""
    out = []
    for m, h in refs.items():
        per = []
        for s in STAT_STEPS:
            k = STEPS.index(s)
            if m == "L10H7":
                per.append(np.array([10 * N_HEADS + j for j in range(N_HEADS)
                                     if 10 * N_HEADS + j not in exclude]))
            else:
                sc = ts[TYPED[m]][k]
                per.append(nearest_pool(sc, sc[h], exclude))
        out.append(per)
    return out


def c3_values() -> Tuple[np.ndarray, str]:
    P, codes = [], set()
    for s in STAT_STEPS:
        with np.load(B2_TABLE / f"step{s}.npz") as z:
            P.append(percentiles(z["C3"].astype(float)))
            codes.add(str(z["code"]))
    return np.stack(P), ",".join(sorted(codes))


def c3_rank(P: np.ndarray, heads: Sequence[int], pools, seed: int, n: int = N_NULL) -> Dict:
    T = float(np.abs(P[:, list(heads)] - 0.5).mean())
    null = c3_null(P, pools, n, seed)
    return {"T": T, "rank": rank_of(T, null), "null_q": np.quantile(null, [0.05, 0.5, 0.95, 0.99]).tolist()}


def fake_sets(layers: Sequence[int], n: int, seed: int) -> np.ndarray:
    mem = {head_index(m) for m in MEMBERS}
    pools = [np.array([l * N_HEADS + h for h in range(N_HEADS) if l * N_HEADS + h not in mem]) for l in layers]
    return draw_sets(pools, n, seed)


def g2(A: np.ndarray, ts: Dict[str, np.ndarray], forms: Dict[str, int], P3: np.ndarray) -> Dict:
    mem = [head_index(m) for m in MEMBERS]
    lay = [parse_head(m)[0] for m in MEMBERS]
    F = fake_sets(lay, N_DRAW, seed_for("b3-g2-r1"))
    r1 = [r1_rank(A, f, seed_for("b3-g2-r1", k), 2000)["rank"] for k, f in enumerate(F)]
    typed = list(TYPED)
    F2 = fake_sets([parse_head(m)[0] for m in typed], N_DRAW, seed_for("b3-g2-r2"))
    fs = [forms[m] for m in typed]
    r2 = [r2_read(A, f, fs, seed_for("b3-g2-r2", k), 2000)["rank"] for k, f in enumerate(F2)]
    g = np.random.default_rng(seed_for("b3-g2-c3"))
    real_pools = type_pools(ts, {m: head_index(m) for m in MEMBERS}, mem)
    k143 = STAT_STEPS.index(143000)
    c3 = []
    for k in range(N_DRAW):
        fake = {m: int(real_pools[i][k143][g.integers(0, len(real_pools[i][k143]))]) for i, m in enumerate(MEMBERS)}
        pools = type_pools(ts, fake, mem + list(fake.values()))
        c3.append(c3_rank(P3, list(fake.values()), pools, seed_for("b3-g2-c3", k), 2000)["rank"])
    rate = {k: float(np.mean(np.array(v) >= RANK_PASS)) for k, v in (("R1", r1), ("R2", r2), ("C3p", c3))}
    return {"rates": rate, "n_draws": N_DRAW, "null_per_draw": 2000,
            "pass": all(v <= G2_MAX for v in rate.values())}


def g3(D: Dict, A: np.ndarray, forms: Dict[str, int], attain: float) -> Dict:
    """Planted onset: 5 layer-matched non-members, step 0's α before the formation step, the
    two-partner rows' α (0, computed) from it on."""
    typed = list(TYPED)
    fs = [forms[m] for m in typed]
    lay = [parse_head(m)[0] for m in typed]
    two = {}
    for s in STEPS:                                               # α of the planted 2-partner rows
        tot, cnt = 0.0, 0
        for r in D["rows"][s].values():
            w = head_weights(r["sink"])
            tot = tot + wls_sums(np.log(r["n"]), np.full(r["neff"].shape, np.log(2.0)), w)
            cnt = cnt + (w > 0).sum(-1)
        two[s] = slope_of(tot, cnt).reshape(-1)

    def one(f: np.ndarray, seed: int) -> Dict:
        Ap = A.copy()
        for h, fk in zip(f, fs):
            Ap[:fk, h] = A[0, h]
            for k in range(fk, len(STEPS)):
                Ap[k, h] = two[STEPS[k]][h]
        return r2_read(Ap, list(f), fs, seed, 2000)

    F = fake_sets(lay, 1 + G3_BESIDE, seed_for("b3-g3"))
    prim = one(F[0], seed_for("b3-g3", 0))
    bes = [one(f, seed_for("b3-g3", k + 1)) for k, f in enumerate(F[1:])]
    rank_read = attain >= RANK_PASS
    ok = prim["count"] == len(typed) and (prim["rank"] >= RANK_PASS or not rank_read)
    return {"primary": {k: prim[k] for k in ("count", "rank", "attainable", "onsets")},
            "heads": [head_name(int(h)) for h in F[0]],
            "beside_count5_rate": float(np.mean([b["count"] == len(typed) for b in bes])),
            "rank_read": rank_read, "pass": bool(ok)}


def cmd_rows(a) -> int:
    out = Path(a.out)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    for s in _steps(a):
        if (out / "rows" / f"step{s}.json").exists():
            continue
        t = time.time()
        rows_step(s, out, a.device)
        print(f"rows step {s}: {time.time() - t:.0f}s", flush=True)
    return 0


def cmd_types(a) -> int:
    out = Path(a.out)
    (out / "types").mkdir(parents=True, exist_ok=True)
    for s in _steps(a):
        if (out / "types" / f"step{s}.json").exists():
            continue
        t = time.time()
        types_step(s, out, a.device)
        print(f"types step {s}: {time.time() - t:.0f}s", flush=True)
    return 0


def _steps(a) -> List[int]:
    return [int(x) for x in a.steps.split(",")] if a.steps else list(STEPS)


def _write(path: Path, rec: Dict) -> None:
    path.write_text(json.dumps(rec, indent=1, default=float) + "\n")


def cmd_read(a) -> int:
    out = Path(a.out)
    D = load_all(out)
    meta = {"code": code_sha(), "prompts_hash": prompts_hash(), "runs": RUNS, "labels": str(LABELS),
            "b2_table": str(B2_TABLE), "constants": {"W_MIN": W_MIN, "MIN_ROWS": MIN_ROWS, "ROW_FROM": ROW_FROM,
                                                    "ONSET_Q": ONSET_Q, "FORM_FRAC": FORM_FRAC, "NEAR_K": NEAR_K,
                                                    "N_NULL": N_NULL, "N_DRAW": N_DRAW, "RANK_PASS": RANK_PASS}}
    A = alpha_grid(D)
    off = member_offset(D)
    ts = type_scores(D, off["offset"])
    forms = formations(ts)
    P3, codes = c3_values()
    if codes != B2_CODE:
        raise B3Error(f"B2 table code {codes}, want {B2_CODE}")
    # ---- the gate, before any member value
    gate = {"meta": meta, "offset": off, "formation_steps": {m: STEPS[k] for m, k in forms.items()},
            "no_alpha": {str(s): int(np.isnan(A[k]).sum()) for k, s in enumerate(STEPS)},
            "G0": g0(D), "G1": g1(D), "G2": g2(A, ts, forms, P3)}
    typed = list(TYPED)
    pre = r2_read(A, [head_index(m) for m in typed], [forms[m] for m in typed], seed_for("b3-r2"))
    gate["R2_attainable"] = {"attainable": pre["attainable"], "rates": dict(zip(typed, pre["rates"])),
                             "descriptive": pre["attainable"] < RANK_PASS}
    gate["G3"] = g3(D, A, forms, pre["attainable"])
    gate["pass"] = all(gate[k]["pass"] for k in ARMS)
    _write(out / "gate.json", gate)
    print(json.dumps({k: gate[k]["pass"] for k in ARMS}), "attainable", round(pre["attainable"], 4), flush=True)
    if not gate["pass"]:
        print("gate fails: no member value read (gate.json)")
        return 2
    if a.gate_only:
        return 0
    # ---- the readings
    mem = [head_index(m) for m in MEMBERS]
    res = {"meta": meta, "gate": str(out / "gate.json")}
    res["R1"] = r1_rank(A, mem, seed_for("b3-r1"))
    res["R2"] = {**{k: pre[k] for k in ("count", "rank", "attainable", "hits", "onsets")},
                 "members": typed, "formation_steps": [STEPS[forms[m]] for m in typed],
                 "onset_steps": [None if k is None else STEPS[k] for k in pre["onsets"]],
                 "reads": "descriptive (attainable < 0.99)" if pre["attainable"] < RANK_PASS else
                 ("members differ" if pre["rank"] >= RANK_PASS else "no difference shown")}
    res["R1"]["reads"] = "members differ" if res["R1"]["rank"] >= RANK_PASS else "no difference shown"
    pools = type_pools(ts, {m: head_index(m) for m in MEMBERS}, mem)
    res["C3p"] = c3_rank(P3, mem, pools, seed_for("b3-c3p"))
    res["C3p"]["pools_143000"] = {m: [head_name(int(h)) for h in pools[i][-1]] for i, m in enumerate(MEMBERS)}
    res["C3p"]["reads"] = ("C3 sets the members apart beyond head type" if res["C3p"]["rank"] >= RANK_PASS
                           else "C3 marks head type, not shown to mark the circuit")
    res["C3p"]["b2_C3"] = json.loads((B2_TABLE / "table.json").read_text())["ranks"]["C3"]
    res["beside"] = beside(D, A, ts, mem)
    _write(out / "result.json", res)
    report(out, gate, res)
    return 0


def beside(D: Dict, A: np.ndarray, ts: Dict[str, np.ndarray], mem: Sequence[int]) -> Dict:
    Au = alpha_grid(D, weighted=False)
    Sp = alpha_grid(D, key="spread")
    Ap = type_series(D, "alpha_probe")
    lo, hi = np.nanquantile(A[0], [0.025, 0.975])
    out = {"step0_band": [float(lo), float(hi)], "population": {}, "members": {}}
    for k, s in enumerate(STEPS):
        out["population"][str(s)] = {"q": np.nanquantile(A[k], [0.1, 0.5, 0.9]).tolist(),
                                     "below_band": float(np.nanmean(A[k] < lo))}
    nb = NEFF_BAND
    for m, h in zip(MEMBERS, mem):
        l, hh = divmod(h, N_HEADS)
        rows = []
        for k, s in enumerate(STEPS):
            r = D["rows"][s]
            ne, sink, out_n, per_p = [], [], 0, {}
            for p, rr in r.items():
                w = 1 - rr["sink"][l, hh]
                sel = (rr["n"] >= nb[0]) & (rr["n"] <= nb[1]) & (w >= W_MIN)
                ne.extend(rr["neff"][l, hh][sel].tolist())
                sink.append(float(rr["sink"][l, hh].mean()))
                out_n += int((w < W_MIN).sum())
                ws = head_weights(rr["sink"][l, hh])
                per_p[p] = float(slope_of(wls_sums(np.log(rr["n"]), np.log(rr["neff"][l, hh]), ws),
                                          np.asarray((ws > 0).sum())))
            rows.append({"step": s, "alpha": A[k, h], "p": float(pctl_nan(A[k])[h]), "alpha_unweighted": Au[k, h],
                         "spread_slope": Sp[k, h], "neff_256": float(np.median(ne)) if ne else None,
                         "sink": float(np.mean(sink)), "rows_out": out_n, "alpha_probe": Ap[k, h],
                         "prev": ts["prev"][k, h], "ind": ts["ind"][k, h], "alpha_per_prompt": per_p})
        out["members"][m] = rows
    out["beta"] = beta_beside(A, mem)
    out["a3"] = a3_beside(D, A, mem)
    return out


def beta_beside(A: np.ndarray, mem: Sequence[int]) -> Dict:
    out = {"fe_full": {}, "m1_layer_143000": {}}
    for k, s in enumerate(STEPS):
        B = {p: np.asarray(json.loads((A1A_RECORDS / f"step{s}_{p}.json").read_text())["beta"]["per_head"]).reshape(-1)
             for p in V1}
        prose = np.median(np.stack([B[p] for p in PROSE]), 0)
        out["fe_full"][str(s)] = {"members_prose": {head_name(h): float(prose[h]) for h in mem},
                                  "members_code": {head_name(h): float(B["hdbscan_code"][h]) for h in mem},
                                  "members_latex": {head_name(h): float(B["latex_monograph"][h]) for h in mem},
                                  "pop_prose_q": np.nanquantile(prose, [0.1, 0.5, 0.9]).tolist(),
                                  "spearman_alpha_prose": spearman(A[k], prose)}
    a2 = json.loads(A2_BETA.read_text())
    for c in a2["cells"]:
        out["m1_layer_143000"][c["prompt"]] = {f"L{parse_head(m)[0]}": c["refit"]["m1/all"]["layer"][parse_head(m)[0]]
                                               for m in MEMBERS}
    return out


def a3_beside(D: Dict, A: np.ndarray, mem: Sequence[int]) -> Dict:
    cat = {h["head"]: h["dnll"] for h in json.loads(CATALOG.read_text())["heads"]}
    solo = np.array([cat.get(head_name(i), np.nan) for i in range(N_POP)], float)
    per_step, skipped = {}, {}
    for s in STAT_STEPS:
        rec = D["recs"][s]
        V = np.stack([np.asarray(rec["a3"][p]["value"], float).reshape(-1) for p in V1])
        per_step[s] = np.nanmedian(V, 0)
        skipped[str(s)] = {p: len(rec["a3"][p]["skipped"]) for p in V1}
    allv = np.nanmedian(np.stack(list(per_step.values())), 0)
    ent = {}
    for k, s in enumerate(STEPS):
        E = np.median(np.stack([np.asarray(D["recs"][s]["entropy"][p], float).reshape(-1) for p in V1]), 0)
        ent[str(s)] = spearman(A[k], E)
    rest = np.array([i not in mem for i in range(N_POP)])
    return {"members": {head_name(h): {"median": float(allv[h]), **{str(s): float(per_step[s][h]) for s in STAT_STEPS}}
                        for h in mem},
            "rest_q": np.nanquantile(allv[rest], [0.1, 0.5, 0.9]).tolist(),
            "spearman_solo_16000": spearman(per_step[16000], solo),
            "spearman_solo_pooled": spearman(allv, solo),
            "spearman_alpha_entropy": ent, "layers_skipped": skipped,
            "note": "layer 0 heads have no c3x layer (labels L1–24, attn[l] reads layer l); L24 has no attention"}


def report(out: Path, gate: Dict, res: Dict) -> None:
    L = ["# B3 unit 1: routing", "", f"code {res['meta']['code']}, prompts {res['meta']['prompts_hash']}", "",
         "## Gate", ""]
    for k in ARMS:
        L.append(f"- {k}: {'pass' if gate[k]['pass'] else 'FAIL'}")
    L += [f"- members' offset: {gate['offset']['offset']} ({gate['offset']['votes']})",
          f"- formation steps: {gate['formation_steps']}",
          f"- R2 attainable rank: {gate['R2_attainable']['attainable']:.4f}", "", "## Family", "",
          "| row | T / count | rank | reads |", "|---|---|---|---|",
          f"| R1 | {res['R1']['T']:.4f} | {res['R1']['rank']:.4f} | {res['R1']['reads']} |",
          f"| R2 | {res['R2']['count']} of 5 | {res['R2']['rank']:.4f} | {res['R2']['reads']} |",
          f"| C3′ | {res['C3p']['T']:.4f} | {res['C3p']['rank']:.4f} | {res['C3p']['reads']} |", "",
          f"B2's C3 (layer-matched) beside: T {res['C3p']['b2_C3']['T']:.4f}, rank {res['C3p']['b2_C3']['rank']}",
          "", "## Members, α per step (p in the 384)", "",
          "| step | " + " | ".join(MEMBERS) + " |", "|---" * (len(MEMBERS) + 1) + "|"]
    mb = res["beside"]["members"]
    for k, s in enumerate(STEPS):
        L.append(f"| {s} | " + " | ".join(f"{mb[m][k]['alpha']:.3f} ({mb[m][k]['p']:.2f})" for m in MEMBERS) + " |")
    (out / "report.txt").write_text("\n".join(L) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rows", "types"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--steps", default="")
        p.add_argument("--device", default="cuda")
    p = sub.add_parser("read")
    p.add_argument("--out", required=True)
    p.add_argument("--gate-only", action="store_true")
    a = ap.parse_args(argv)
    return {"rows": cmd_rows, "types": cmd_types, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
