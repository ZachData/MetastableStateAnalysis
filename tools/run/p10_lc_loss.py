"""LC: loss coupling (`p10_cluster_function/design-10.md` "LC", fixed before any pass).

Is the next-token loss lower at c3x's members than at the other kept tokens at the same
positions, and from which step? Per (step, passage) one GPU pass gives, per position i,
``s[i] = −log p(x_{i+1} | x_≤i)`` (primary), ``s_own[i] = s[i−1]`` and the entropy ``H[i]``. Per
(step, passage, layer, column) ``D`` is F13's position-binned difference members − rest over R0's
kept offsets (i ≤ n − 2), in units of the domain's mean s; beside it a linear position fit
(``D~0``), the same with the next token's frequency and class (``D~nx``), ``D_own``, ``D_H``, and
the positive control ``P_ct`` (positions whose next token is a continuation against the rest).
``calibrate`` runs the whole reading on planted surprisal over the real design points (no pass).
Tier 1: exploratory, unregistered.
    python tools/run/p10_lc_loss.py calibrate --labels <R9 source> --out <dir>
    python tools/run/p10_lc_loss.py run --labels <R9 source> --out <dir> [--steps 512 ...]
    python tools/run/p10_lc_loss.py stats --labels <R9 source> --out <dir>
    python tools/run/p10_lc_loss.py report --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from tools.run import p10_f13_centres as f
from tools.run.p10_f13_rare import ols_centre

STEPS = f.STEPS
PRIMARY_STEPS = tuple(s for s in STEPS if s >= 64)
PASSAGES = f.PASSAGES
LAYERS = tuple(range(1, 25))
BANDS = {"L1-8": tuple(range(1, 9)), "L9-16": tuple(range(9, 17)), "L17-24": tuple(range(17, 25))}
COLUMNS = ("c3x", "c2")
N_BINS = 8
MIN_A, MIN_B = 3, 10
FLOOR = 0.05                                               # A0's floor, R3's relative units
N_BOOT = 1000
FIRST = (512, "wiki_paragraph")
HOMER_MAX = f.HOMER_MAX                       # placed: the v1 rows read homer_iliad at its first 512 tokens
MATCH_TOL = 1e-4                                           # placed: u2_attn.MATCH_TOL["v1"]
STATS = ("D", "A", "D8", "D~0", "D~nx", "D_own", "D_H")
NAMES = ("higher", "lower")                                # F13's sign_label(pos, neg): value < 0 is lower
CO_LOCATES = (1000, 2000, 4000)
# calibration (the rule's)
N_DRAWS = 200
DELTAS = (0.05, 0.1, 0.2, 0.3)
NULL_CELL_BAR, NULL_ONSET_BAR = 0.02, 0.90


class LCError(RuntimeError):
    pass


# ---------------------------------------------------------------- pure pieces
def surprisals(logp: np.ndarray, ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """From log-probabilities (n, V): ``s`` (n − 1; s[i] = −log p(x_{i+1} | x_≤i)) and ``H`` (n)."""
    ids = np.asarray(ids)
    s = -logp[np.arange(len(ids) - 1), ids[1:]]
    H = -(np.exp(logp) * logp).sum(1)
    return s, H


def own(s: np.ndarray) -> np.ndarray:
    """``s_own[i] = s[i − 1]`` (NaN at 0), on s's n − 1 positions plus the last."""
    out = np.full(s.size + 1, np.nan)
    out[1:] = s
    return out


def domain(kept: np.ndarray, n: int) -> np.ndarray:
    k = np.sort(np.asarray(kept, dtype=int))
    return k[(k >= 1) & (k <= n - 2)]


def rel_binned(v: np.ndarray, dom: np.ndarray, A: np.ndarray, B: np.ndarray) -> Optional[float]:
    """F13's binned A − B over N_BINS position bins of ``dom``, over the domain's mean of v."""
    d = f.binned_diff(v, A, B, f.bins_of(dom, N_BINS), MIN_A, MIN_B)
    scale = float(np.nanmean(v[dom]))
    return None if d is None or not scale > 0 else d / scale


def position_resid(v: np.ndarray, dom: np.ndarray) -> np.ndarray:
    """v less its membership-blind least-squares fit on a cubic in log(i + 1) over the domain (NaN off it)."""
    lp = np.log(dom + 1.0)
    X = np.column_stack([np.ones_like(lp), lp, lp ** 2, lp ** 3])
    y = v[dom]
    ok = np.isfinite(y)
    out = np.full(v.shape, np.nan)
    out[dom[ok]] = y[ok] - X[ok] @ np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return out


def rel_binned_resid(v: np.ndarray, dom: np.ndarray, A: np.ndarray, B: np.ndarray) -> Optional[float]:
    """``rel_binned`` on v's position residual, still over the domain's mean of v (amended D)."""
    d = f.binned_diff(position_resid(v, dom), A, B, f.bins_of(dom, N_BINS), MIN_A, MIN_B)
    scale = float(np.nanmean(v[dom]))
    return None if d is None or not scale > 0 else d / scale


def rank_gap(v: np.ndarray, dom: np.ndarray, A: np.ndarray, B: np.ndarray) -> Optional[float]:
    """``A``: per position bin, P(a member's position-residual v > a rest token's) − 0.5 (ties ½),
    weighted by m·r / n_bin. Its null is symmetric, so the sign rule is not tilted by v's skew."""
    if A.size < MIN_A or B.size < MIN_B:
        return None
    return rank_gap_resid(position_resid(v, dom), dom, A, B)


def rank_gap_resid(r: np.ndarray, dom: np.ndarray, A: np.ndarray, B: np.ndarray) -> Optional[float]:
    """``rank_gap`` on a residual already taken (LCo's covariate residual)."""
    if A.size < MIN_A or B.size < MIN_B:
        return None
    sa, sb = set(A.tolist()), set(B.tolist())
    num = den = 0.0
    for b in f.bins_of(dom, N_BINS):
        a = r[[x for x in b if x in sa]]
        c = r[[x for x in b if x in sb]]
        a, c = a[np.isfinite(a)], c[np.isfinite(c)]
        if a.size and c.size:
            u = ((a[:, None] > c[None, :]).mean() + 0.5 * (a[:, None] == c[None, :]).mean())
            w = a.size * c.size / b.size
            num += w * (u - 0.5)
            den += w
    return num / den if den else None


def rel_ols(v: np.ndarray, dom: np.ndarray, member: np.ndarray, covs: Sequence[np.ndarray]) -> Optional[float]:
    if member.sum() < MIN_A or (~member).sum() < MIN_B:
        return None
    c = ols_centre(v[dom], member, [x[dom] for x in covs], dom.astype(float))
    scale = float(np.nanmean(v[dom]))
    return None if c is None or not scale > 0 else c / scale


def layer_stats(s: np.ndarray, H: np.ndarray, dom: np.ndarray, lab_dom: np.ndarray,
                nx: Dict[str, np.ndarray]) -> Dict[str, Optional[float]]:
    """``lab_dom``: the column's labels at ``dom`` (≥ 0 member)."""
    mem = lab_dom >= 0
    A, B = dom[mem], dom[~mem]
    so = own(s)
    s_full = np.append(s, np.nan)                          # index by position; the last has no next token
    return {"D": rel_binned_resid(s_full, dom, A, B),
            "A": rank_gap(s_full, dom, A, B),
            "D8": rel_binned(s_full, dom, A, B),
            "D~0": rel_ols(s_full, dom, mem, []),
            "D~nx": rel_ols(s_full, dom, mem, [nx["lf"], nx["ws"], nx["ct"]]),
            "D_own": rel_binned_resid(so, dom[dom >= 1], A[A >= 1], B[B >= 1]),
            "D_H": rel_binned_resid(H, dom, A, B),
            "n_members": int(mem.sum()), "n_rest": int((~mem).sum())}


def next_features(tokens: Sequence[str], counts: np.ndarray, vocab: dict, added: set) -> Dict[str, np.ndarray]:
    """Per position i, the *next* token's lf and word_start / continuation indicators (NaN at the last)."""
    from tools.run import p10_token_composition as tc
    tf = tc.token_features(np.asarray(tokens), vocab, added)
    ids = np.array([t["id"] for t in tf])
    cls = np.array([t["cls"] for t in tf])
    lf = np.log10((counts[ids] + 0.5) / counts.sum())
    out = {}
    for k, v in (("lf", lf), ("ws", (cls == "word_start").astype(float)), ("ct", (cls == "continuation").astype(float))):
        out[k] = np.append(v[1:], np.nan)
    return out


def control(s: np.ndarray, dom: np.ndarray, nx: Dict[str, np.ndarray]) -> Optional[float]:
    """P_ct: positions whose next token is a continuation, against the rest of the domain."""
    s_full = np.append(s, np.nan)
    ct = nx["ct"][dom] == 1
    return rel_binned_resid(s_full, dom, dom[ct], dom[~ct])


def control_rank(s: np.ndarray, dom: np.ndarray, nx: Dict[str, np.ndarray]) -> Optional[float]:
    """P_ct's rank gap (A's statistic), for its sign (after `/challenge-pr` on #192, finding 5)."""
    s_full = np.append(s, np.nan)
    ct = nx["ct"][dom] == 1
    return rank_gap(s_full, dom, dom[ct], dom[~ct])


# ---------------------------------------------------------------- labels and readings (pure)
def label_grid(vals: Dict[Tuple[int, str], Dict], steps: Sequence[int], stat: str, column: str) -> Dict:
    """{band: {step: [passage band values]}} from per-record {column: {layer: stats}}."""
    grid: Dict = {b: {} for b in BANDS}
    for b, Ls in BANDS.items():
        for s in steps:
            row = []
            for p in PASSAGES:
                rec = vals.get((s, p))
                if rec is None:
                    continue
                by = {L: rec[column][str(L)][stat] if str(L) in rec[column] else None for L in LAYERS}
                row.append(f.band_mean(by, Ls))
            grid[b][s] = row
    return grid


def floor_label(vals: Sequence[Optional[float]], sizes: Optional[Sequence[Optional[float]]] = None) -> str:
    """Sign rule on ``vals``; the floor on the median of ``sizes`` (``vals`` when None). D's labels
    take their sign from A (amended) and their size from D."""
    lab = f.sign_label(vals, *NAMES)
    v = [x for x in (vals if sizes is None else sizes) if x is not None and np.isfinite(x)]
    if lab not in ("mixed", "too few") and abs(float(np.median(v))) < FLOOR:
        return f"{lab} (under the floor)"
    if lab not in ("mixed", "too few") and (float(np.median(v)) < 0) != lab.endswith("lower"):
        return f"{lab} (size disagrees)"                   # after `/challenge-pr` on #192, finding 5
    return lab


def labels(grid: Dict, sizes: Optional[Dict] = None) -> Dict[str, Dict[int, str]]:
    """F13's ``labelled`` (isolated leans, at-init marks) with the floor applied first."""
    out = {}
    for band, row in grid.items():
        labs = {s: floor_label(v, None if sizes is None else sizes[band][s]) for s, v in row.items()}
        labs = f.mark_isolated(labs)
        if 0 in labs:
            base = labs[0]
            for s in labs:
                if s != 0 and labs[s] == base and base not in ("mixed", "too few"):
                    labs[s] += " (at init)"
        out[band] = labs
    return out


def core(lab: str) -> str:
    for m in (" (isolated)", " (at init)"):
        lab = lab.replace(m, "")
    return lab


def onset(labs: Dict[int, str], steps: Sequence[int] = PRIMARY_STEPS) -> Tuple[Optional[int], str]:
    """The first step labelled lower (full, at the floor) whose next readable step is lower or leans
    lower; steps with too few readable passages are skipped (amended at the calibration gate)."""
    st = [s for s in steps if s in labs and core(labs[s]) != "too few"]
    for i, s in enumerate(st):
        if core(labs[s]) != "lower":
            continue
        if i == len(st) - 1 or core(labs[st[i + 1]]) in ("lower", "leans lower"):
            return s, ("precedes" if s <= 512 else "co-locates" if s in CO_LOCATES else "follows")
    return None, "no gap"


def boot_share(vals: Sequence[Optional[float]], seed: int, n: int = N_BOOT) -> Optional[float]:
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if v.size < f.MIN_PASSAGES:
        return None
    rng = np.random.default_rng(seed)
    med = np.median(v[rng.integers(0, v.size, (n, v.size))], axis=1)
    return float((med <= -FLOOR).mean())


# ---------------------------------------------------------------- the design points
def design(labels_src: Path, counts_dir: Optional[Path]) -> Dict[Tuple[int, str], Dict]:
    """Per (step, passage): n, the domain, each column's labels at the domain per layer, tokens."""
    from tools.run import p10_label_source as ls
    vocab = added = counts = None
    if counts_dir is not None:
        from tools.run import p10_token_composition as tc
        counts = np.load(Path(counts_dir) / "counts.npy")
        vocab, added = tc.load_vocab(tc.find_tokenizer(Path(os.environ.get("HF_HOME", str(DATA / "hf")))))
    out = {}
    for s in STEPS:
        d = ls.load_step(labels_src, f"step{s}")
        cache: Dict = {f"step{s}": d}
        for p in PASSAGES:
            pr = d["prompts"][p]
            n = int(pr["n_positions"])
            kept = np.asarray(pr["kept"], dtype=int)
            dom = domain(kept, n)
            where = np.searchsorted(kept, dom)
            cols = {}
            for c in COLUMNS:
                cols[c] = {}
                for L in LAYERS:
                    try:
                        pos, lab = ls.load_column(labels_src, f"step{s}", p, L, c, cache)
                    except ls.LabelSourceError:
                        continue
                    if not np.array_equal(pos, kept):
                        raise LCError(f"{s} {p} L{L} {c}: positions are not R0's kept offsets")
                    cols[c][L] = lab[where]
            tokens = json.loads((Path(pr["stage0_run"]) / "geometry.json").read_text())["tokens"][:n]
            rec = {"n": n, "dom": dom, "cols": cols, "run": pr["stage0_run"], "tokens": tokens}
            if counts is not None:
                rec["nx"] = next_features(tokens, counts, vocab, added)
            out[(s, p)] = rec
    return out


def record_stats(dp: Dict, s: np.ndarray, H: np.ndarray) -> Dict:
    out = {"P_ct": control(s, dp["dom"], dp["nx"]), "P_ct_A": control_rank(s, dp["dom"], dp["nx"]), "mean_s": float(np.mean(s[dp["dom"]])),
           "sd_over_mean": float(np.std(s[dp["dom"]]) / np.mean(s[dp["dom"]]))}
    for c in COLUMNS:
        out[c] = {str(L): layer_stats(s, H, dp["dom"], lab, dp["nx"]) for L, lab in dp["cols"][c].items()}
    return out


# ---------------------------------------------------------------- calibration (planted, no pass)
def _cal_job(args) -> Dict:
    """One planted (step, passage): s = f(i)·Exp(1) (N1: × 0.5 before a continuation); E / O scale
    each layer's members by (1 − δ) from ``start``."""
    key, dp, arm, delta, start, draw = args
    step, passage = key
    rng = np.random.default_rng([zlib.crc32(f"{arm}|{delta}|{start}|{step}|{passage}".encode()), draw])
    n = dp["n"]
    i = np.arange(n - 1)
    base = (3 + 5 * (1 + i) ** -0.4) * rng.exponential(1.0, n - 1)
    if arm == "N1":
        base = np.where(dp["nx"]["ct"][:n - 1] == 1, 0.5 * base, base)
    H = np.append(base, base[-1])                          # placeholder: D_H is not calibrated
    out = {"c3x": {}}
    dom = dp["dom"]
    for L, lab in dp["cols"]["c3x"].items():
        s = base.copy()
        if arm in ("E", "O") and step >= start:
            mem = dom[lab >= 0]
            s[mem] = s[mem] * (1 - delta)
        st = layer_stats(s, H, dom, lab, dp["nx"])
        out["c3x"][str(L)] = {k: st[k] for k in ("D", "A", "D8", "D~0", "D~nx")}
    return out


def calibrate(a) -> int:
    dps = design(a.labels, a.counts)
    arms = [("N0", 0.0, 0), ("N1", 0.0, 0)] + [("E", d, 0) for d in DELTAS] + [("O", 0.2, 2000), ("O", 0.2, 256), ("O", 0.3, 2000), ("O", 0.3, 256)]
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "calibration.json"
    res = json.loads(path.read_text()) if (a.only and path.exists()) else {}
    keys = [k for k in dps if k[0] in PRIMARY_STEPS]
    slim = {k: {"n": dps[k]["n"], "dom": dps[k]["dom"], "cols": {"c3x": dps[k]["cols"]["c3x"]}, "nx": dps[k]["nx"]}
            for k in keys}
    with Pool(a.workers) as pool:
        for arm, delta, start in arms:
            tag = f"{arm}" + (f"_d{delta}" if arm in ("E", "O") else "") + (f"_from{start}" if arm == "O" else "")
            if a.only and tag not in a.only:
                continue
            t0 = time.monotonic()
            cells, onsets, nx_cells, d8_cells = [], [], [], []
            for draw in range(a.draws):
                jobs = [(k, slim[k], arm, delta, start, draw) for k in keys]
                vals = dict(zip(keys, pool.map(_cal_job, jobs, chunksize=4)))
                labs = labels(label_grid(vals, PRIMARY_STEPS, "A", "c3x"), label_grid(vals, PRIMARY_STEPS, "D", "c3x"))
                labs_nx = labels(label_grid(vals, PRIMARY_STEPS, "D~nx", "c3x"))
                labs_8 = labels(label_grid(vals, PRIMARY_STEPS, "D8", "c3x"))
                for b in BANDS:
                    for s in PRIMARY_STEPS:
                        cells.append(core(labs[b][s]))
                        nx_cells.append(core(labs_nx[b][s]))
                        d8_cells.append(core(labs_8[b][s]))
                    onsets.append(onset(labs[b]))
            cells_a, nx_a = np.array(cells), np.array(nx_cells)
            res[tag] = {
                "cells": len(cells),
                "full_lower": float((cells_a == "lower").mean()), "full_higher": float((cells_a == "higher").mean()),
                "lower_or_lean": float(np.isin(cells_a, ["lower", "leans lower"]).mean()),
                "nx_full_lower": float((nx_a == "lower").mean()),
                "D8_full_lower": float((np.array(d8_cells) == "lower").mean()),
                "D8_full_higher": float((np.array(d8_cells) == "higher").mean()),
                "onset_categories": {c: sum(o[1] == c for o in onsets) / len(onsets)
                                     for c in ("precedes", "co-locates", "follows", "no gap")},
                "onset_steps": {str(s): sum(o[0] == s for o in onsets) / len(onsets) for s in PRIMARY_STEPS},
                "seconds": round(time.monotonic() - t0, 1)}
            print(tag, json.dumps(res[tag]), flush=True)
            path.write_text(json.dumps(res, indent=1) + "\n")
    n0 = res["N0"]
    ok = (n0["full_lower"] + n0["full_higher"] <= NULL_CELL_BAR and n0["onset_categories"]["no gap"] >= NULL_ONSET_BAR)
    print(f"N0 bar: {'passes' if ok else 'FAILS'}", flush=True)
    return 0 if ok else 2


# ---------------------------------------------------------------- the pass (GPU)
def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def check_hidden(hs: np.ndarray, run_dir: Path) -> float:
    z = np.load(run_dir / "activations.npz")
    stored = z["activations"] * z["norms"][..., None]
    if stored.shape != hs.shape:
        raise LCError(f"{run_dir}: stored {stored.shape}, pass {hs.shape}")
    unit = lambda A: A / np.linalg.norm(A, axis=-1, keepdims=True)    # noqa: E731
    match = float(np.abs(unit(hs) - unit(stored)).max())
    if match > MATCH_TOL:
        raise LCError(f"{run_dir}: pass against stored unit rows {match:.1e} > {MATCH_TOL}")
    return match


def run(a) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.lm_loading import resolve_lm_entry
    from core.models import from_pretrained_eager
    from p1e_energy_field import u2_attn as ua
    from tools.run import p10_label_source as ls
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (the rule's pass is the GPU's)")
    code = code_sha()
    rec_dir = a.out / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps({"rule": 'design-10.md "LC"', "code": code,
                                                 "labels": str(a.labels)}, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    steps = [FIRST[0]] + [s for s in (a.steps or STEPS) if s != FIRST[0]]
    first_done = (rec_dir / f"step{FIRST[0]}_{FIRST[1]}.npz").exists()
    for step in steps:
        d = ls.load_step(a.labels, f"step{step}")
        todo = [p for p in PASSAGES if not (rec_dir / f"step{step}_{p}.npz").exists()]
        if step == FIRST[0]:
            todo = sorted(todo, key=lambda p: p != FIRST[1])
        if not todo:
            continue
        # the causal variant (core.models.load_model is the bare body, no head), same repo and pinned
        # revision as the bare load, in float32: core.lm_loading.load_causal_lm would give bfloat16 on cuda
        entry = resolve_lm_entry(f"pythia-410m-step{step}")
        model = from_pretrained_eager(AutoModelForCausalLM, entry["repo_id"], revision=entry["revision"],
                                      torch_dtype=torch.float32).to("cuda").eval()
        if next(model.parameters()).dtype != torch.float32:
            raise SystemExit("refusing: model is not float32")
        dev = next(model.parameters()).device
        for p in todo:
            t0 = time.monotonic()
            pr = d["prompts"][p]
            rd = Path(pr["stage0_run"])
            ids = ua.token_ids(tok, rd)
            with torch.no_grad():
                o = model(torch.tensor([ids], device=dev), output_hidden_states=True)
            hs = torch.stack(o.hidden_states)[:, 0].double().cpu().numpy()
            match = check_hidden(hs, rd)
            logp = torch.log_softmax(o.logits[0].double(), dim=-1).cpu().numpy()
            n = int(pr["n_positions"])
            s, H = surprisals(logp[:n], ids[:n])
            if not (np.isfinite(s).all() and np.isfinite(H).all()):
                raise LCError(f"{step} {p}: non-finite surprisal")
            dom = domain(np.asarray(pr["kept"]), n)
            if (step, p) == FIRST and not first_done:
                m = float(s[dom].mean())
                if not (1 < m < 11) or np.ptp(s[dom]) == 0:
                    raise LCError(f"first record: mean s {m:.2f} over the domain, not in (1, 11) or constant")
                print(f"first record populated: mean s {m:.3f}, sd {s[dom].std():.3f}, match {match:.1e}", flush=True)
                first_done = True
            path = rec_dir / f"step{step}_{p}.npz"
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, s=s, H=H, ids=np.asarray(ids[:n]), match=match, code=code)
            tmp.rename(path)
            print(f"done step {step} {p}: n {n}, mean s {s[dom].mean():.3f}, {time.monotonic() - t0:.1f}s, "
                  f"match {match:.1e}", flush=True)
            if a.first_only:
                return 0
        del model
        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- stats and report (CPU)
def stats(a) -> int:
    dps = design(a.labels, a.counts)
    ran = json.loads((a.out / "plan.json").read_text())["code"]
    out = {}
    for (s, p), dp in dps.items():
        z = np.load(a.out / "records" / f"step{s}_{p}.npz")
        if len(z["s"]) != dp["n"] - 1 or str(z["code"]) != ran:
            raise LCError(f"{s} {p}: {len(z['s'])} surprisals for n {dp['n']}, code {z['code']} (plan {ran})")
        out[f"{s}|{p}"] = record_stats(dp, z["s"], z["H"])
    (a.out / "stats.json").write_text(json.dumps(out) + "\n")
    noise = {k: v["sd_over_mean"] for k, v in out.items()}
    print("sd/mean of s in the domain (planted Exp: 1): median %.2f, range %.2f-%.2f"
          % (np.median(list(noise.values())), min(noise.values()), max(noise.values())))
    return 0


def readings(rep: Dict) -> Dict:
    """The rule's readings (1)–(4) on c3x at the primary steps."""
    labs, nx, d0 = rep["labels"]["c3x|D"], rep["labels"]["c3x|D~nx"], rep["labels"]["c3x|D~0"]
    out: Dict = {"trained": {b: labs[b][143000] for b in BANDS}, "onset": {}, "carried_by_next": [],
                 "higher_carried_by_next": [], "position_flag": []}
    for b in BANDS:
        out["onset"][b] = onset(labs[b])
        for s in PRIMARY_STEPS:
            lab = core(labs[b][s])
            if lab in ("lower", "leans lower") and core(nx[b][s]) not in ("lower", "leans lower"):
                out["carried_by_next"].append([b, s, labs[b][s], nx[b][s]])
            # the same reading for "higher" (after `/challenge-pr` on #192, finding 1; after output)
            if lab in ("higher", "leans higher") and core(nx[b][s]) not in ("higher", "leans higher"):
                out["higher_carried_by_next"].append([b, s, labs[b][s], nx[b][s]])
            opp = {"lower": "higher", "higher": "lower"}.get(lab)
            if opp and core(d0[b][s]) == opp:
                out["position_flag"].append([b, s, labs[b][s], d0[b][s]])
    return out


def report(a) -> int:
    raw = json.loads((a.out / "stats.json").read_text())
    vals = {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in raw.items()}
    rep: Dict = {"labels": {}, "medians": {}, "onset": {}, "boot": {}}
    for c in COLUMNS:
        for st in STATS:
            g = label_grid(vals, STEPS, st, c)
            rep["labels"][f"{c}|{st}"] = labels(label_grid(vals, STEPS, "A", c), g) if st == "D" else labels(g)
            rep["medians"][f"{c}|{st}"] = {b: {str(s): (float(np.median([x for x in v if x is not None]))
                                                          if any(x is not None for x in v) else None)
                                               for s, v in row.items()} for b, row in g.items()}
            if st == "D":
                rep["boot"][c] = {b: {str(s): boot_share(v, zlib.crc32(f"{c}|{b}|{s}".encode()))
                                      for s, v in row.items()} for b, row in g.items()}
    for b in BANDS:
        rep["onset"][b] = onset(rep["labels"]["c3x|D"][b])
    rep["readings"] = readings(rep)
    pc = {s: [vals[(s, p)]["P_ct"] for p in PASSAGES if (s, p) in vals] for s in STEPS}
    pa = {s: [vals[(s, p)].get("P_ct_A") for p in PASSAGES if (s, p) in vals] for s in STEPS}
    rep["P_ct"] = {str(s): {"label": floor_label(v), "label_A_sign": floor_label(pa[s], v),
                            "median": float(np.median([x for x in v if x is not None]))} for s, v in pc.items()}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    lines = ["# LC report", "", "c3x, D (members − rest, binned, / mean s): label (median)", ""]
    hdr = "| band | " + " | ".join(str(s) for s in STEPS) + " |"
    lines += [hdr, "|" + "---|" * (len(STEPS) + 1)]
    for c in COLUMNS:
        for st in STATS:
            for b in BANDS:
                labs, meds = rep["labels"][f"{c}|{st}"][b], rep["medians"][f"{c}|{st}"][b]
                cells = [f"{labs[s]} ({meds[str(s)]:+.3f})" if meds[str(s)] is not None else labs[s] for s in STEPS]
                lines.append(f"| {c} {st} {b} | " + " | ".join(cells) + " |")
    lines += ["", "P_ct: " + "; ".join(f"{s} {v['label']} ({v['median']:+.3f})" for s, v in rep["P_ct"].items()),
              "", "onset (c3x D): " + "; ".join(f"{b} {o}" for b, o in rep["onset"].items()),
              "", "readings: " + json.dumps(rep["readings"])]
    (a.out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "run", "stats", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name != "report":
            p.add_argument("--labels", type=Path, required=True)
        if name in ("calibrate", "stats"):
            p.add_argument("--counts", type=Path, default=DATA / "p10" / "pile_unigram_2026-10-10")
        if name == "calibrate":
            p.add_argument("--draws", type=int, default=N_DRAWS)
            p.add_argument("--workers", type=int, default=12)
            p.add_argument("--only", nargs="*", help="arm tags to run, merged into an existing calibration.json")
        if name == "run":
            p.add_argument("--steps", type=int, nargs="*")
            p.add_argument("--first-only", action="store_true")
    a = ap.parse_args(argv)
    return {"calibrate": calibrate, "run": run, "stats": stats, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
