#!/usr/bin/env python3
"""
F13, the centre scan, with the cross (A9) and the coverage curve (F15), as `design-10.md` "F13"
fixed it (`9f1c525`, before any centre was computed).

Per (step, passage): Stage 0's stored activations and attentions, no forward pass. At each layer,
in unit LN1 rows (primary) and the γ-free centred sphere (beside), geodesic distance: the strong
and ordinary Rényi centres (`2411.04990` §5.1) of the stored order and of 200 random arrival orders
(position 0 first), swept in δ = c·3.5^-1/2. Scores: F13's counts (`D_N`, against the exact order
null `Σ 1/(a_x+1)`) and index (`D_I`); A9's position-divided received attention of the centres and
of c3x's non-centre members, in matched position bins; F15's coverage by L_def's centres followed
through depth, against the order null's centres and against count-matched random sets.

    python tools/run/p10_f13_centres.py run --labels <R9 source> --out <dir> [--first-only] [--workers 12]
    python tools/run/p10_f13_centres.py report --out <dir>

Results: `status-10.md` §1.41.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
MEMBER_STEPS = tuple(s for s in STEPS if s >= 64)           # c3x's steps (the rule's membership labels)
PASSAGES = ("wiki_paragraph", "sullivan_ballou", "paper_excerpt", "homer_iliad", "hdbscan_code",
            "camus_letranger", "latex_monograph")
FRAMES = {"ln1": tuple(range(0, 24)), "sphere": tuple(range(0, 25))}   # layers each frame reads
PRIMARY_FRAME = "ln1"
BETA = 3.5
CS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
PRIMARY_C = 2.0
A9_CS = (1.0, 1.5, 2.0)
DELTAS = tuple(c / np.sqrt(BETA) for c in CS)
N_PERM = 200
N_BINS = 8
L_DEFS = ("first", 8, 16)                                # "first": the first layer ≥ 1 with N_s in range (amended)
PRIMARY_LDEF = "first"
A9_LAYERS = tuple(range(1, 24))                             # block L reads hs[L]; L24 has no block
BANDS = {"L1-8": tuple(range(1, 9)), "L9-16": tuple(range(9, 17)), "L17-23": tuple(range(17, 24))}
COLUMNS = ("c3x", "c2")
MIN_A, MIN_B = 3, 10                                        # A9: first set, second set (placed)
MIN_LAYERS, MIN_PASSAGES = 3, 6
NULL_SE, NULL_RATE, NULL_Q = 3.0, 0.0027, 0.999          # a 3-SE miss's two-sided rate; the bar's quantile
ATTN_ROW_TOL, ATTN_UPPER_TOL = 1e-3, 1e-6
FIRST = (512, "wiki_paragraph")
HOMER_MAX = 512


class F13Error(RuntimeError):
    pass


# ---------------------------------------------------------------- geometry (pure)
def geodesic(U: np.ndarray) -> np.ndarray:
    """Pairwise geodesic distance of unit rows, exactly symmetric (a BLAS Gram is not, in its last bits,
    and the greedy and the coverage read the matrix from opposite sides), zero diagonal."""
    G = U @ U.T
    D = np.arccos(np.clip((G + G.T) / 2, -1.0, 1.0))
    np.fill_diagonal(D, 0.0)
    return D


def sphere_rows(X: np.ndarray) -> np.ndarray:
    """The γ-free centred sphere: ``unit(x − mean over channels)``."""
    X = np.asarray(X, dtype=np.float64)
    Y = X - X.mean(axis=1, keepdims=True)
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def orders(step: int, passage: str, n: int, n_perm: int = N_PERM) -> np.ndarray:
    """(1 + n_perm, n): row 0 the stored order, then random orders with position 0 first."""
    rng = np.random.default_rng(zlib.crc32(f"f13|{step}|{passage}".encode()))
    out = np.empty((1 + n_perm, n), dtype=np.int64)
    out[0] = np.arange(n)
    for p in range(n_perm):
        out[1 + p, 0] = 0
        out[1 + p, 1:] = 1 + rng.permutation(n - 1)
    return out


def ranks_of(O: np.ndarray) -> np.ndarray:
    R = np.empty_like(O)
    rows = np.arange(O.shape[0])[:, None]
    R[rows, O] = np.arange(O.shape[1])[None, :]
    return R


def nearest_pred(D: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(P, n): each token's distance to its nearest predecessor in each order (inf for the first)."""
    out = np.empty(R.shape, dtype=np.float64)
    for p in range(R.shape[0]):
        r = R[p]
        out[p] = np.where(r[None, :] < r[:, None], D, np.inf).min(axis=1)
    return out


def strong_sets(m: np.ndarray, deltas: Sequence[float]) -> np.ndarray:
    """(C, P, n) bool: strong centres, ``m > δ``."""
    return m[None] > np.asarray(deltas)[:, None, None]


def ordinary_sets(D: np.ndarray, O: np.ndarray, deltas: Sequence[float]) -> np.ndarray:
    """(C, P, n) bool: the greedy ordinary centres of each order, every δ at once."""
    W = D[None] <= np.asarray(deltas)[:, None, None]            # (C, n, n)
    C, (P, n) = W.shape[0], O.shape
    blocked = np.zeros((C, P, n), dtype=bool)
    acc = np.zeros((C, P, n), dtype=bool)
    rows = np.arange(P)
    for t in range(n):
        j = O[:, t]
        a = ~blocked[:, rows, j]                                # (C, P)
        acc[:, rows, j] = a
        blocked |= a[..., None] & W[:, j, :]
    return acc


def null_bar(total: int) -> int:
    """The most 3-SE misses a correct null gives in ``total`` records: Binomial(total, 0.0027)'s 99.9th
    percentile (amended: a 1 % share refused ~5 of 252 correct cells by chance)."""
    from scipy.stats import binom
    return int(binom.ppf(NULL_Q, total, NULL_RATE))


def exact_null_strong(D: np.ndarray, kept: np.ndarray, delta: float) -> float:
    """The order null's mean strong count over ``kept`` (position 0 held first): renyi_f13.py M3."""
    near = D <= delta
    tot = 0.0
    for x in kept:
        if x == 0 or near[0, x]:
            continue
        a = int(near[x, 1:].sum()) - 1
        tot += 1.0 / (a + 1)
    return tot


# ---------------------------------------------------------------- attention (pure)
def log_rho(A: np.ndarray) -> np.ndarray:
    """Position-divided received attention: head mean of Σ_i A[i, j] over Σ_{i≥j} 1/(i+1)."""
    Am = A.mean(axis=0, dtype=np.float64) if A.ndim == 3 else np.asarray(A, dtype=np.float64)
    n = Am.shape[0]
    r = Am.sum(axis=0)
    inv = 1.0 / np.arange(1, n + 1)
    denom = np.cumsum(inv[::-1])[::-1]                          # Σ_{i=j}^{n-1} 1/(i+1)
    return np.log(np.maximum(r, 1e-300) / denom)


def bins_of(kept: np.ndarray, n_bins: int = N_BINS) -> List[np.ndarray]:
    return [b for b in np.array_split(np.sort(kept), n_bins) if b.size]


def binned_diff(v: np.ndarray, A: np.ndarray, B: np.ndarray, bins: List[np.ndarray],
                min_a: int = MIN_A, min_b: int = MIN_B) -> Optional[float]:
    """Bin-wise mean(v[A]) − mean(v[B]), weighted by A's count, over bins holding both."""
    if A.size < min_a or B.size < min_b:
        return None
    sa, sb = set(A.tolist()), set(B.tolist())
    num = den = 0.0
    for b in bins:
        a = [x for x in b if x in sa]
        c = [x for x in b if x in sb]
        if a and c:
            num += len(a) * (v[a].mean() - v[c].mean())
            den += len(a)
    return num / den if den else None


# ---------------------------------------------------------------- coverage (pure)
def coverage(Cm: np.ndarray, Win: np.ndarray, kept_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For centre masks (S, n): the share of kept non-centres within δ of a centre, and the mean
    number of them per centre."""
    Cf = Cm.astype(np.float32)
    hits = Cf @ Win.astype(np.float32)                          # (S, n): centres within δ of each token
    tgt = kept_mask[None, :] & ~Cm
    nt = tgt.sum(1)
    cov = np.where(nt > 0, ((hits > 0) & tgt).sum(1) / np.maximum(nt, 1), np.nan)   # no targets: vacuous
    per = (hits * tgt).sum(1) / np.maximum(Cm.sum(1), 1)        # Σ_c |targets within δ of c| / |C|
    return cov, per


def random_sets(seed_key: str, pool: np.ndarray, size: int, n: int, n_draws: int = N_PERM) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
    out = np.zeros((n_draws, n), dtype=bool)
    if size > 0:
        for k in range(n_draws):
            out[k, rng.choice(pool, size=min(size, pool.size), replace=False)] = True
    return out


# ---------------------------------------------------------------- one (step, passage)
def t2_positions(tokens: Sequence[str], kept: np.ndarray) -> List[int]:
    """Non-kept positions > 0 that are their token's first occurrence (the T2 tokens and the like)."""
    ks, seen, out = set(kept.tolist()), set(), []
    for i, t in enumerate(tokens):
        if i > 0 and i not in ks and t not in seen:
            out.append(i)
        seen.add(t)
    return out


def check_inputs(acts: np.ndarray, attn: np.ndarray, where: str) -> Dict:
    if acts.ndim != 3 or acts.shape[0] != 25 or acts.shape[2] != 1024 or not np.isfinite(acts).all():
        raise F13Error(f"{where}: activations {acts.shape} not (25, n, 1024) and finite")
    n = acts.shape[1]
    if attn.shape != (24, 16, n, n):
        raise F13Error(f"{where}: attentions {attn.shape} not (24, 16, {n}, {n})")
    rows = np.abs(attn.sum(-1) - 1).max()
    upper = attn[..., np.triu_indices(n, 1)[0], np.triu_indices(n, 1)[1]].max() if n > 1 else 0.0
    if rows > ATTN_ROW_TOL or upper > ATTN_UPPER_TOL:
        raise F13Error(f"{where}: attention rows off 1 by {rows:.1e} or upper triangle {upper:.1e}")
    return {"attn_row_dev": float(rows), "attn_upper": float(upper)}


def frame_layers(acts: np.ndarray, ln1: Dict, frame: str) -> Dict[int, np.ndarray]:
    from p1e_energy_field.u2_block import unit_rows
    if frame == "ln1":
        return {L: unit_rows(acts[L], ln1["w"][L], ln1["b"][L], ln1["eps"]) for L in FRAMES["ln1"]}
    return {L: sphere_rows(acts[L]) for L in FRAMES["sphere"]}


def read_passage(step: int, passage: str, acts: np.ndarray, attn: np.ndarray, ln1: Dict,
                 kept: np.ndarray, labels: Dict[str, Dict[int, np.ndarray]], tokens: Sequence[str]) -> Dict:
    from tools.run import p10_label_source as ls
    n = acts.shape[1]
    O = orders(step, passage, n)
    R = ranks_of(O)
    kept = np.asarray(sorted(int(k) for k in kept if k != 0))
    kmask = np.zeros(n, dtype=bool)
    kmask[kept] = True
    bins = bins_of(kept)
    t2 = t2_positions(tokens, kept)
    lr = {L: log_rho(attn[L]) for L in A9_LAYERS}
    ci = {c: CS.index(c) for c in A9_CS}
    pc = CS.index(PRIMARY_C)
    rec: Dict = {"n": n, "kept": int(kept.size), "t2": t2, "frames": {}, "null_check": {}}
    rec["log_rho"] = {"pos0": {L: float(lr[L][0]) for L in A9_LAYERS},
                      "t2": {str(t): {L: float(lr[L][t]) for L in A9_LAYERS} for t in t2}}
    for frame in FRAMES:
        U = frame_layers(acts, ln1, frame)
        F: Dict = {"layers": {}, "f15": {}}
        sets_at: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        Ds: Dict[int, np.ndarray] = {}
        outside = total = 0
        for L, UL in U.items():
            D = geodesic(UL)
            Ds[L] = D
            m = nearest_pred(D, R)
            S = strong_sets(m, DELTAS)                          # (C, P, n)
            Od = ordinary_sets(D, O, DELTAS)
            if not (S <= Od).all():
                raise F13Error(f"{step} {passage} {frame} L{L}: a strong centre is not ordinary")
            sets_at[L] = (S[pc], Od[pc])
            Sk, Ok = S[:, :, kept], Od[:, :, kept]
            Ns, No = Sk.sum(-1), Ok.sum(-1)                     # (C, P)
            idx = R[:, kept] / (n - 1)
            Ival = np.where(Sk.sum(-1) > 0, (Sk * idx[None]).sum(-1) / np.maximum(Sk.sum(-1), 1), np.nan)
            lay = {"Ns": Ns[:, 0].tolist(), "No": No[:, 0].tolist(), "exact_Ns": [], "null_Ns_mean": [],
                   "null_Ns_sd": [], "null_No_mean": [], "null_No_sd": [], "I": [], "I_null": []}
            for k, dl in enumerate(DELTAS):
                ex = exact_null_strong(D, kept, dl)
                mu, sd = Ns[k, 1:].mean(), Ns[k, 1:].std(ddof=1)
                total += 1
                if abs(mu - ex) > NULL_SE * max(sd, np.sqrt(ex)) / np.sqrt(N_PERM) + 1e-9:
                    outside += 1
                lay["exact_Ns"].append(ex)
                lay["null_Ns_mean"].append(float(mu))
                lay["null_Ns_sd"].append(float(sd))
                lay["null_No_mean"].append(float(No[k, 1:].mean()))
                lay["null_No_sd"].append(float(No[k, 1:].std(ddof=1)))
                lay["I"].append(float(Ival[k, 0]))
                lay["I_null"].append(float(np.nanmean(Ival[k, 1:])) if np.isfinite(Ival[k, 1:]).any() else None)
            lay["centres_strong"] = np.flatnonzero(S[pc, 0]).tolist()
            lay["centres_ordinary"] = np.flatnonzero(Od[pc, 0]).tolist()
            lay["t2_strong"] = {str(t): bool(S[pc, 0, t]) for t in t2}
            if L in A9_LAYERS:
                a9: Dict = {}
                for c, k in ci.items():
                    cen = np.intersect1d(np.flatnonzero(S[k, 0]), kept)
                    non = np.setdiff1d(kept, cen)
                    row = {"n_centres": int(cen.size), "S_c": binned_diff(lr[L], cen, non, bins)}
                    for col in COLUMNS:
                        pos, lab = ls.load_column(labels["src"], f"step{step}", passage, L, col, labels["cache"])
                        if not np.array_equal(np.asarray(pos), kept):
                            raise F13Error(f"{step} {passage} L{L}: {col} domain differs from the kept offsets")
                        if not ls.readable(lab):
                            row[col] = None
                            continue
                        mem = np.asarray(pos)[np.asarray(lab) >= 0]
                        memn = np.setdiff1d(mem, cen)
                        nonm = np.setdiff1d(np.setdiff1d(kept, mem), cen)
                        row[col] = {"members": int(mem.size), "member_centres": int(np.intersect1d(mem, cen).size),
                                    "S_b": binned_diff(lr[L], memn, nonm, bins),
                                    "S_cm": binned_diff(lr[L], cen, memn, bins)}
                    a9[str(c)] = row
                lay["a9"] = a9
            F["layers"][str(L)] = lay
        rec["null_check"][frame] = {"outside": outside, "total": total}
        if outside > null_bar(total):
            raise F13Error(f"{step} {passage} {frame}: the null's mean is off its exact value in "
                           f"{outside} of {total} records")
        # F15: L_def's centres followed through depth
        dl = DELTAS[pc]
        first = next((L for L in sorted(U) if L >= 1 and in_range(F["layers"][str(L)]["Ns"][pc], kept.size)), None)
        for key in L_DEFS:
            Ld = first if key == "first" else key
            for kind, j in (("strong", 0), ("ordinary", 1)):
                if Ld is None:
                    F["f15"][f"{key}:{kind}"] = {"n_centres": 0, "L": {}, "L_def": None}
                    continue
                M = sets_at[Ld][j] & kmask[None, :]             # (P, n), kept only
                n_obs = int(M[0].sum())
                rnd = random_sets(f"f13r|{step}|{passage}|{frame}|{key}|{kind}", kept, n_obs, n)
                cur = {"n_centres": n_obs, "L": {}, "L_def": int(Ld)}
                if n_obs >= MIN_A:
                    for L in U:
                        if L <= Ld:
                            continue
                        Win = Ds[L] <= dl
                        cv, per = coverage(M, Win, kmask)
                        cr, pr = coverage(rnd, Win, kmask)
                        cur["L"][str(L)] = {"cov": float(cv[0]), "null_mean": float(np.nanmean(cv[1:])),
                                            "null_sd": float(np.nanstd(cv[1:], ddof=1)), "rand_mean": float(np.nanmean(cr)),
                                            "rand_sd": float(np.nanstd(cr, ddof=1)), "per": float(per[0]),
                                            "per_null": float(per[1:].mean()), "per_rand": float(pr.mean())}
                    c0, _ = coverage(M[:1], Ds[Ld] <= dl, kmask)
                    cur["cov_def"] = float(c0[0])
                if kind == "ordinary":                          # the greedy's check: every centre, kept or not
                    full, _ = coverage(sets_at[Ld][1][:1], Ds[Ld] <= dl, kmask)
                    if np.isfinite(full[0]) and full[0] < 1.0 - 1e-12:
                        raise F13Error(f"{step} {passage} {frame} L{Ld}: ordinary centres leave a token uncovered")
                F["f15"][f"{key}:{kind}"] = cur
        rec["frames"][frame] = F
    return rec


# ---------------------------------------------------------------- run
def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def check_populated(rec: Dict) -> None:
    """The first record: at c = 2, N_s in range in ≥ half of L1–23 in the primary frame."""
    pc = CS.index(PRIMARY_C)
    lay = rec["frames"][PRIMARY_FRAME]["layers"]
    ok = sum(MIN_A <= lay[str(L)]["Ns"][pc] <= rec["kept"] / 2 for L in A9_LAYERS)
    if ok < len(A9_LAYERS) / 2:
        raise F13Error(f"first record: N_s in range at c = 2 in {ok} of {len(A9_LAYERS)} layers; amend the grid")


def _job(args) -> str:
    step, passage, run_dir, kept, src, ln1_dir, out, code = args
    from p1e_energy_field.u2_block import export_ln1, load_ln1
    t0 = time.monotonic()
    rd = Path(run_dir)
    acts = np.load(rd / "activations.npz")["activations"]
    attn = np.load(rd / "attentions.npz")["attentions"]
    tokens = json.loads((rd / "geometry.json").read_text())["tokens"]
    if passage == "homer_iliad":
        acts, attn, tokens = acts[:, :HOMER_MAX], attn[:, :, :HOMER_MAX, :HOMER_MAX], tokens[:HOMER_MAX]
    if len(tokens) != acts.shape[1]:
        raise F13Error(f"{step} {passage}: {len(tokens)} tokens against {acts.shape[1]} positions")
    chk = check_inputs(acts, attn, f"{step} {passage}")
    ln1 = load_ln1(export_ln1(f"step{step}", Path(ln1_dir)))
    if ln1["revision"] != f"step{step}":
        raise F13Error(f"{step}: LN1 file is {ln1['revision']}")
    rec = read_passage(step, passage, acts.astype(np.float64), attn, ln1,
                       np.asarray(kept), {"src": Path(src), "cache": {}}, tokens)
    rec.update({"step": step, "passage": passage, "run": str(rd), "code": code, "checks": chk,
                "ln1_snapshot": ln1["snapshot"], "seconds": round(time.monotonic() - t0, 1)})
    path = Path(out) / "records" / f"step{step}_{passage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec) + "\n")
    tmp.rename(path)
    return f"done step {step} {passage}: {rec['seconds']}s"


def run(a) -> int:
    from tools.run import p10_label_source as ls
    src = Path(a.labels)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    code = code_sha()
    meta = {"rule": "design-10.md F13 (9f1c525)", "git": git_head(), "code": code, "labels": str(src),
            "summary_sha256": hashlib.sha256((src / "summary.json").read_bytes()).hexdigest(),
            "deltas": list(DELTAS), "cs": list(CS), "beta": BETA, "n_perm": N_PERM}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    jobs = []
    for step in (a.steps or STEPS):
        d = ls.load_step(src, f"step{step}")
        for p in PASSAGES:
            if (out / "records" / f"step{step}_{p}.json").exists():
                continue
            pr = d["prompts"][p]
            jobs.append((step, p, pr["stage0_run"], pr["kept"], str(src), str(a.ln1), str(out), code))
    first = [j for j in jobs if (j[0], j[1]) == FIRST]
    if first:
        print(_job(first[0]), flush=True)
        rec = json.loads((out / "records" / f"step{FIRST[0]}_{FIRST[1]}.json").read_text())
        check_populated(rec)
        pc = CS.index(PRIMARY_C)
        print("first record populated: N_s at c = 2 by layer",
              [rec["frames"][PRIMARY_FRAME]["layers"][str(L)]["Ns"][pc] for L in FRAMES[PRIMARY_FRAME]],
              f"kept {rec['kept']}; null check {rec['null_check']}", flush=True)
        jobs = [j for j in jobs if (j[0], j[1]) != FIRST]
    if a.first_only:
        return 0
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(_job, jobs):
            print(msg, flush=True)
    return 0


# ---------------------------------------------------------------- labels (pure)
def sign_label(vals: Sequence[float], pos: str, neg: str) -> str:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    n = len(v)
    if n < MIN_PASSAGES:
        return "too few"
    p, q = sum(x > 0 for x in v), sum(x < 0 for x in v)
    if p == n:
        return pos
    if q == n:
        return neg
    if p == n - 1:
        return f"leans {pos}"
    if q == n - 1:
        return f"leans {neg}"
    return "mixed"


def band_mean(by_layer: Dict[int, Optional[float]], band: Sequence[int]) -> Optional[float]:
    v = [by_layer[L] for L in band if by_layer.get(L) is not None and np.isfinite(by_layer[L])]
    return float(np.mean(v)) if len(v) >= MIN_LAYERS else None


def mark_isolated(labels: Dict[int, str]) -> Dict[int, str]:
    """A lean not shared (same label) by an adjacent step is marked isolated."""
    out, steps = dict(labels), sorted(labels)
    for i, s in enumerate(steps):
        lab = labels[s]
        if lab.startswith("leans"):
            nb = [labels[steps[j]] for j in (i - 1, i + 1) if 0 <= j < len(steps)]
            if lab not in nb:
                out[s] = lab + " (isolated)"
    return out


def in_range(N: float, kept: int) -> bool:
    return MIN_A <= N <= kept / 2


# ---------------------------------------------------------------- report
def load_records(out: Path) -> Dict[Tuple[int, str], Dict]:
    recs = {}
    for f in sorted((out / "records").glob("step*_*.json")):
        r = json.loads(f.read_text())
        recs[(r["step"], r["passage"])] = r
    return recs


def f13_values(r: Dict, frame: str, c: float, stat: str) -> Dict[int, Optional[float]]:
    k = CS.index(c)
    out = {}
    for L, lay in r["frames"][frame]["layers"].items():
        L = int(L)
        if stat == "D_Ns":
            N = lay["Ns"][k]
            out[L] = float(np.log((N + 0.5) / (lay["exact_Ns"][k] + 0.5))) if in_range(N, r["kept"]) else None
        elif stat == "D_No":
            N = lay["No"][k]
            out[L] = float(np.log((N + 0.5) / (lay["null_No_mean"][k] + 0.5))) if in_range(N, r["kept"]) else None
        elif stat == "D_I":
            ok = in_range(lay["Ns"][k], r["kept"]) and lay["I_null"][k] is not None
            out[L] = lay["I"][k] - lay["I_null"][k] if ok else None
    return out


def a9_values(r: Dict, frame: str, c: float, stat: str, column: str = "c3x") -> Dict[int, Optional[float]]:
    out = {}
    for L, lay in r["frames"][frame]["layers"].items():
        if "a9" not in lay:
            continue
        row = lay["a9"][str(c)]
        if stat == "S_c":
            out[int(L)] = row["S_c"]
        else:
            col = row.get(column)
            out[int(L)] = None if col is None else col[stat]
    return out


def f15_values(r: Dict, frame: str, Ld, kind: str, null: str) -> Dict[int, Optional[float]]:
    cur = r["frames"][frame]["f15"][f"{Ld}:{kind}"]
    key = "null_mean" if null == "order" else "rand_mean"
    return {int(L): v["cov"] - v[key] for L, v in cur.get("L", {}).items()}


NAMES = {"D_Ns": ("more centres", "fewer centres"), "D_No": ("more centres", "fewer centres"),
         "D_I": ("later", "earlier"), "S_c": ("attended more", "attended less"),
         "S_b": ("attended more", "attended less"), "S_cm": ("attended more", "attended less"),
         "F15": ("captures more", "captures less")}


def label_grid(recs: Dict, steps: Sequence[int], fn) -> Dict[str, Dict[int, Tuple[str, List]]]:
    """{band: {step: (label, passage values)}} with fn(record, band) → value or None."""
    out = {}
    for band, layers in BANDS.items():
        row = {}
        for s in steps:
            vals = [fn(recs[(s, p)], layers) for p in PASSAGES if (s, p) in recs]
            row[s] = vals
        out[band] = row
    return out


def labelled(grid: Dict, pos: str, neg: str) -> Dict[str, Dict[int, str]]:
    res = {}
    for band, row in grid.items():
        labs = {s: sign_label(v, pos, neg) for s, v in row.items()}
        labs = mark_isolated(labs)
        if 0 in labs:
            base = labs[0]
            for s in labs:
                if s != 0 and labs[s] == base and base not in ("mixed", "too few"):
                    labs[s] += " (at init)"
        res[band] = labs
    return res


def f15_label(recs: Dict, steps, frame: str, Ld, kind: str) -> Dict[str, Dict[int, str]]:
    pos, neg = NAMES["F15"]
    a = labelled(label_grid(recs, steps, lambda r, b: band_mean(f15_values(r, frame, Ld, kind, "order"), b)), pos, neg)
    b = labelled(label_grid(recs, steps, lambda r, bb: band_mean(f15_values(r, frame, Ld, kind, "rand"), bb)), pos, neg)
    out = {}
    for band in a:
        out[band] = {}
        for s in a[band]:
            la, lb = a[band][s], b[band][s]
            core = lambda x: x.replace(" (isolated)", "").replace(" (at init)", "")
            if core(la) in ("too few",) or core(lb) in ("too few",):
                out[band][s] = "too few"
            elif core(la) == core(lb):
                out[band][s] = la
            else:
                out[band][s] = f"count-dependent ({core(la)} / {core(lb)})"
    return out


def a9_reading(sc: str, sb: str, scm: str) -> str:
    core = lambda x: x.replace(" (isolated)", "").replace(" (at init)", "")
    sc, sb, scm = core(sc), core(sb), core(scm)
    if "too few" in (sc,):
        return "not read"
    if sc == "attended more":
        return "anchor and ballast" if sb == "attended less" else "centres attended, members not ballast"
    if scm != "attended more":
        return "centres not sinks; H-ANCHOR/BALLAST does not hold"
    return "centres not sinks"


def table(title: str, labs: Dict[str, Dict[int, str]], steps) -> List[str]:
    lines = [f"\n{title}", "step    " + " | ".join(f"{b:<34}" for b in BANDS)]
    for s in steps:
        lines.append(f"{s:<7} " + " | ".join(f"{labs[b][s]:<34}" for b in BANDS))
    return lines


def report(a) -> int:
    out = Path(a.out)
    recs = load_records(out)
    steps = [s for s in STEPS if all((s, p) in recs for p in PASSAGES)]
    if len(recs) != len(STEPS) * len(PASSAGES):
        print(f"warning: {len(recs)} records of {len(STEPS) * len(PASSAGES)}")
    L: List[str] = [f"F13 / A9 / F15 report — {len(recs)} records, rule design-10.md F13; "
                    f"labels over {len(PASSAGES)} passages; primary frame {PRIMARY_FRAME}, c = {PRIMARY_C} "
                    f"(δ = {DELTAS[CS.index(PRIMARY_C)]:.3f})"]
    summary: Dict = {"steps": steps, "labels": {}}
    nc = [v for r in recs.values() for v in r["null_check"].values()]
    L.append(f"null check: {sum(x['outside'] for x in nc)} of {sum(x['total'] for x in nc)} records "
             f"outside 3 SE of the exact mean")
    for frame in FRAMES:
        for c in (PRIMARY_C, 1.0, 1.5):
            tag = f"{frame} c={c}"
            for stat in ("D_Ns", "D_No", "D_I"):
                labs = labelled(label_grid(recs, steps, lambda r, b, st=stat: band_mean(f13_values(r, frame, c, st), b)), *NAMES[stat])
                summary["labels"][f"F13 {stat} {tag}"] = labs
                if frame == PRIMARY_FRAME or c == PRIMARY_C:
                    L += table(f"F13 {stat} [{tag}] (strong unless D_No)", labs, steps)
            a9 = {}
            for stat in ("S_c", "S_b", "S_cm"):
                labs = labelled(label_grid(recs, steps, lambda r, b, st=stat: band_mean(a9_values(r, frame, c, st), b)), *NAMES[stat])
                a9[stat] = labs
                summary["labels"][f"A9 {stat} {tag}"] = labs
                if frame == PRIMARY_FRAME or c == PRIMARY_C:
                    L += table(f"A9 {stat} [{tag}]", labs, steps)
            reading = {b: {s: (a9_reading(a9["S_c"][b][s], a9["S_b"][b][s], a9["S_cm"][b][s])
                               if s in MEMBER_STEPS else
                               ("centres attended" if a9["S_c"][b][s].startswith("attended more") else "—"))
                           for s in steps} for b in BANDS}
            summary["labels"][f"A9 reading {tag}"] = reading
            if frame == PRIMARY_FRAME or c == PRIMARY_C:
                L += table(f"A9 READING [{tag}] (membership read at 64–143000)", reading, steps)
        for Ld in L_DEFS:
            for kind in ("strong", "ordinary"):
                labs = f15_label(recs, steps, frame, Ld, kind)
                summary["labels"][f"F15 {frame} Ldef={Ld} {kind}"] = labs
                if Ld == PRIMARY_LDEF or frame == PRIMARY_FRAME:
                    L += table(f"F15 [{frame}, L_def {Ld}, {kind}]", labs, steps)
    # beside: S_b on c2, position 0, T2
    labs = labelled(label_grid(recs, steps, lambda r, b: band_mean(a9_values(r, PRIMARY_FRAME, PRIMARY_C, "S_b", "c2"), b)), *NAMES["S_b"])
    L += table("A9 S_b on c2 (beside)", labs, steps)
    L.append("\nbeside: position 0's and the T2 tokens' log rho (band mean, passage median) and T2 strong-centre share")
    for s in steps:
        p0 = [band_mean({int(k): v for k, v in recs[(s, p)]["log_rho"]["pos0"].items()}, BANDS["L9-16"]) for p in PASSAGES]
        t2r, t2s = [], []
        for p in PASSAGES:
            r = recs[(s, p)]
            for t, by in r["log_rho"]["t2"].items():
                t2r.append(band_mean({int(k): v for k, v in by.items()}, BANDS["L9-16"]))
                t2s += [r["frames"][PRIMARY_FRAME]["layers"][str(Lq)]["t2_strong"][t] for Lq in BANDS["L9-16"]]
        L.append(f"  {s:<7} pos0 L9-16 {np.nanmedian([x for x in p0 if x is not None]):+.2f}   "
                 f"T2 L9-16 {np.nanmedian([x for x in t2r if x is not None]) if t2r else float('nan'):+.2f}   "
                 f"T2 strong share L9-16 {np.mean(t2s) if t2s else float('nan'):.2f}")
    (out / "report.txt").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(L))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--labels", type=Path, default=DATA / "p10" / "reread_r9_2026-10-08" / "labels")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--ln1", type=Path, default=DATA / "p1e" / "u2_block_2026-10-07" / "ln1")
    r.add_argument("--steps", type=int, nargs="*")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--first-only", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
