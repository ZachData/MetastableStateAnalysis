"""
p1d_cluster_ensemble/arch_null.py — unit 2 of the Blocked 11⁗ programme:
the architecture as the null (`design-1d.md` "Unit 2"; rules fixed there
before any run).

The same statistics on the same prompt through many initialisations of
pythia-410m's architecture:

- **real inits**: ``pythia-410m`` ``step0`` (seed 0) and PolyPythias
  ``pythia-410m-seed{1..9}`` ``step0`` (`lit-1d.md` §10 rows 1, 1a);
- **re-inits**: ``N_REINITS`` draws of the same config at Pythia's two σ
  (`lit-1d.md` §10 row 2a), `reinit_model`; never written to disk.
  transformers' ``init_weights()`` is not Pythia's init and is not used.

Per (model, prompt, layer L1–24, frame), on the prompt's kept tokens, one
cloud record (`cloud_record`): the observed ``ci2``, ``nn1`` (both as
`gaussian_null.lumpiness` defines them) and level-set HDBSCAN group counts
``hdb_k_2`` / ``hdb_k_4`` (`admit.level_set_hdbscan`, admission's route,
not the shipped call with its tie order); the same on ``N_DRAWS`` draws of
the cloud's own matched-covariance Gaussian (`gaussian_null.gaussian_draw`,
same frame, same n); ``z_G`` = (obs − draw mean) / draw SD; and per
level-set group ``s`` = ``excess`` ÷ the 95th percentile of the draws'
largest ``excess`` (admission's threshold, so ``s > 1`` is admission's
verdict). The draws use the same seed for the same (prompt, layer, frame)
in every model (common random numbers).

Token rules (`design-1d.md` "Token rules"), and how they are read here:

- T1: position 0 is in no cloud.
- T2: a position whose norm exceeds ``MASSIVE_RATIO`` x its layer's median
  (over all positions) at any of L2–20 in **any model of the comparison**
  is dropped from every model's cloud of that prompt (`token_sets`). The
  comparison is the ``step`` being run: the step-0 first check takes the
  union over the 10 real inits and the re-inits only, so it reads no
  trained run (as unit 1).
- T3: first occurrence of each string (one tokenizer, so one token set per
  prompt across models).

**First check** (`first_check`, before any trained cell is read): per
(statistic, band, frame), each real init's ``z_G`` is placed among the
re-inits' at every (prompt, layer) by its mid-rank fraction ``u``; per
(seed, prompt) the median ``u`` over the band's layers; the cell **passes**
if at most ``FIRST_CHECK_BOUND`` of those 70 medians fall in the outer
``OUTER`` of the re-inits (``u < OUTER/2`` or ``u > 1 − OUTER/2``). A cell
that fails falls back to the 10 real inits, where every rank-p rule refuses.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .admit import level_set_hdbscan
from .gaussian_null import cluster_index_2, frame_vectors, gaussian_draw, span_coordinates
from .move_text import (LAYERS, MASSIVE_LAYERS, MASSIVE_RATIO, V1_MAX_TOKENS, V1_PASSAGES,
                        band, forward, kept_offsets, massive_positions)

FRAMES = ("centred", "raw")
MIN_CLUSTER_SIZES = (2, 4)
BANDS = ("L1-8", "L9-16", "L17-24")
#: Per-cloud statistics; ``hdb_k_2`` is primary, ``hdb_k_4`` its arm.
STATS = ("hdb_k_2", "hdb_k_4", "nn1", "ci2")
#: Placed (`design-1d.md`): 40 re-inits, 100 Gaussian draws per cloud.
N_REINITS = 40
N_DRAWS = 100
#: Admission's level: a group's threshold is the draws' max ``excess`` q95.
ALPHA = 0.05
POLY_SEEDS = tuple(range(1, 10))
#: Seeds 3 and 4 are PolyPythias' 410m outliers (`lit-1d.md` §10 row 1): kept, flagged.
FLAGGED_SEEDS = (3, 4)
#: First check, placed (`design-1d.md`).
OUTER = 0.10
FIRST_CHECK_BOUND = 0.20
BASE_REPO = "EleutherAI/pythia-410m"

# Pythia's init (`lit-1d.md` §10 rows 2, 2a), by parameter name.
SMALL_INIT = ("gpt_neox.embed_in.weight", "attention.query_key_value.weight",
              "mlp.dense_h_to_4h.weight", "embed_out.weight")
WANG_INIT = ("attention.dense.weight", "mlp.dense_4h_to_h.weight")
#: A re-init tensor whose sample SD is off its σ by more than this share (or
#: 5 standard errors, 5 / √(2n), for a small tensor) refuses.
SIGMA_TOLERANCE = 0.01


def sigmas(d: int, n_layers: int) -> Dict[str, float]:
    """``small_init`` √(2 / 5d) and ``wang_init`` 2 / (L √d)."""
    return {"small": math.sqrt(2.0 / (5.0 * d)), "wang": 2.0 / (n_layers * math.sqrt(d))}


def init_class(name: str) -> str:
    """Which init a parameter takes: ``small``, ``wang``, ``zero`` (biases), ``one`` (LN weights)."""
    if name.endswith(".bias"):
        return "zero"
    if "layernorm" in name or "layer_norm" in name:
        return "one"
    if any(name.endswith(k) for k in WANG_INIT):
        return "wang"
    if any(name.endswith(k) for k in SMALL_INIT):
        return "small"
    raise ValueError(f"no Pythia init rule for parameter {name!r}; refusing to guess")


def reinit_model(config, seed: int):
    """
    A ``GPTNeoXForCausalLM`` of ``config`` with every parameter redrawn at
    Pythia's σ from ``torch.Generator(seed)`` (biases 0, LayerNorm (1, 0)),
    in `named_parameters` order; eval mode, eager attention. Refuses if any
    weight's sample SD is off its σ by more than ``SIGMA_TOLERANCE``
    (or 5 standard errors).
    """
    import torch
    from transformers import GPTNeoXForCausalLM
    config._attn_implementation = "eager"
    model = GPTNeoXForCausalLM(config).to(torch.float32).eval()
    sig = sigmas(config.hidden_size, config.num_hidden_layers)
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for name, p in model.named_parameters():
            c = init_class(name)
            if c == "zero":
                p.zero_()
            elif c == "one":
                p.fill_(1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32) * sig[c])
                sd = float(p.std())
                if abs(sd / sig[c] - 1) > max(SIGMA_TOLERANCE, 5 / math.sqrt(2 * p.numel())):
                    raise ValueError(f"{name}: SD {sd:.5g} vs σ {sig[c]:.5g}")
    return model


def init_summary(model) -> Dict[str, Dict[str, float]]:
    """Per init class: the min and max sample SD over its weights (and |max| for zero / one)."""
    out: Dict[str, Dict[str, float]] = {}
    for name, p in model.named_parameters():
        c = init_class(name)
        v = float(p.detach().std()) if c in ("small", "wang") else \
            float((p.detach() - (1.0 if c == "one" else 0.0)).abs().max())
        o = out.setdefault(c, {"min": v, "max": v})
        o["min"], o["max"] = min(o["min"], v), max(o["max"], v)
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def model_ids(which: str) -> List[str]:
    """``init:0..9`` (seed 0 = ``pythia-410m``) and ``reinit:0..N_REINITS-1``."""
    ids = []
    if which in ("init", "all"):
        ids += [f"init:{s}" for s in (0, *POLY_SEEDS)]
    if which in ("reinit", "all"):
        ids += [f"reinit:{s}" for s in range(N_REINITS)]
    if not ids:
        raise ValueError(which)
    return ids


def repo_of(seed: int) -> str:
    return BASE_REPO if seed == 0 else f"{BASE_REPO}-seed{seed}"


def load(mid: str, step: str):
    """The model for ``mid`` at ``step`` (re-inits exist at step 0 only)."""
    import torch
    from transformers import AutoConfig, GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    kind, s = mid.split(":")
    if kind == "reinit":
        if step != "step0":
            raise ValueError("re-inits are step 0 only")
        return reinit_model(AutoConfig.from_pretrained(BASE_REPO, revision="step0"), int(s))
    m = from_pretrained_eager(GPTNeoXForCausalLM, repo_of(int(s)), revision=step,
                              torch_dtype=torch.float32)
    return m.eval()


def check_config(model) -> None:
    """Every model must be pythia-410m's architecture."""
    c = model.config
    want = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16,
            "intermediate_size": 4096, "rotary_pct": 0.25, "use_parallel_residual": True,
            "vocab_size": 50304}
    bad = {k: getattr(c, k, None) for k, v in want.items() if getattr(c, k, None) != v}
    if bad:
        raise ValueError(f"not pythia-410m's config: {bad}")


# ---------------------------------------------------------------------------
# Prompts and token sets
# ---------------------------------------------------------------------------

def prompt_ids(tokenizer) -> Dict[str, List[int]]:
    """The 7 v1 passages, truncated to ``V1_MAX_TOKENS`` as every v1 run read them."""
    from .move_text import passage_inputs
    return {k: v["ids"] for k, v in passage_inputs("v1", tokenizer).items()}


def massive_table(norms: Dict[str, np.ndarray]) -> Dict[str, Dict[int, Tuple[float, int]]]:
    """Per prompt: `move_text.massive_positions` of one model's ``(25, n)`` norms."""
    return {k: massive_positions(v) for k, v in norms.items()}


def token_sets(tokens: Dict[str, List[str]], massive_by_model: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Per prompt: kept positions (T1, T2 union over every model given, T3)
    and the union's list (position, string, max ratio, layer, model).
    """
    out = {}
    for k, toks in tokens.items():
        union: Dict[int, Dict] = {}
        for mid, tab in massive_by_model.items():
            for p, (r, L) in tab.get(k, {}).items():
                p = int(p)
                if p not in union or r > union[p]["ratio"]:
                    union[p] = {"position": p, "token": toks[p], "ratio": float(r), "layer": int(L),
                                "model": mid}
        kept = kept_offsets(toks, list(union))
        out[k] = {"kept": kept.tolist(), "massive": sorted(union.values(), key=lambda m: m["position"])}
    return out


# ---------------------------------------------------------------------------
# One cloud
# ---------------------------------------------------------------------------

def nn1(Z: np.ndarray) -> float:
    """`gaussian_null.lumpiness`' ``nn1``: mean cosine distance to the nearest other row."""
    from .methods import LayerData
    D = LayerData.from_normed(Z).cos_dist.copy()
    np.fill_diagonal(D, np.inf)
    return float(D.min(axis=1).mean())


def cloud_stats(Z: np.ndarray, seed: int) -> Tuple[Dict[str, float], Dict[int, List[Dict]]]:
    """Every per-cloud statistic of unit rows ``Z``, and the level-set groups per size."""
    from .methods import LayerData
    D = LayerData.from_normed(Z).cos_dist
    st, groups = {"nn1": nn1(Z), "ci2": cluster_index_2(Z, seed)}, {}
    for m in MIN_CLUSTER_SIZES:
        _, rows = level_set_hdbscan(D, m)
        st[f"hdb_k_{m}"] = float(len(rows))
        groups[m] = rows
    return st, groups


def z_of(obs: float, null: np.ndarray) -> Optional[float]:
    sd = float(np.std(null, ddof=1)) if null.size > 1 else 0.0
    return None if sd <= 0 or not np.isfinite(obs) else float((obs - null.mean()) / sd)


def cloud_record(Y: np.ndarray, frame: str, n_draws: int, seed) -> Dict:
    """
    One (model, prompt, layer, frame): observed statistics, their ``z_G``
    against ``n_draws`` draws of the cloud's matched-covariance Gaussian, and
    every level-set group's ``excess`` and ``s`` per size (members are row
    indices into ``Y``). A statistic whose draws have SD 0 gets ``z`` None.
    """
    Z, info = frame_vectors(Y, frame)
    Zs = span_coordinates(Z)
    obs, groups = cloud_stats(Zs, 0)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in STATS}
    dmax = {m: [] for m in MIN_CLUSTER_SIZES}
    for _ in range(int(n_draws)):
        st, dg = cloud_stats(gaussian_draw(Zs, rng), 0)
        for k in STATS:
            draws[k].append(st[k])
        for m in MIN_CLUSTER_SIZES:
            # A draw with no group sets no bar (as `admit.admit_record`).
            dmax[m].append(max((r["excess"] for r in dg[m]), default=0.0))
    stats = {}
    for k in STATS:
        d = np.asarray(draws[k], dtype=np.float64)
        stats[k] = {"obs": obs[k], "null_mean": float(d.mean()),
                    "null_sd": float(d.std(ddof=1)) if d.size > 1 else 0.0, "z": z_of(obs[k], d)}
    arms = {}
    for m in MIN_CLUSTER_SIZES:
        q95 = float(np.quantile(dmax[m], 1 - ALPHA))
        arms[str(m)] = {"q95": q95, "groups": [
            {"members": [int(i) for i in r["members"]], "excess": float(r["excess"]),
             "s": float(r["excess"] / q95) if q95 > 0 else None} for r in groups[m]]}
    return {"info": {k: info[k] for k in ("mean_share", "eff_dim")}, "stats": stats, "arms": arms}


# ---------------------------------------------------------------------------
# One model: forward every prompt, then every (prompt, layer, frame)
# ---------------------------------------------------------------------------

#: Threading variables pinned to 1 in each worker. The workers are spawned,
#: not forked: forked after torch's forward pass, KMeans' OpenMP hung every
#: worker at load 0 (2026-10-02).
_ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _job(args: Tuple[str, int, str, np.ndarray, np.ndarray, int, int]) -> Tuple[str, Dict]:
    prompt, L, frame, Y, kept, seed, n_draws = args
    rec = cloud_record(Y, frame, n_draws, [seed, V1_PASSAGES.index(prompt), L, FRAMES.index(frame)])
    for a in rec["arms"].values():
        for g in a["groups"]:
            g["members"] = kept[g["members"]].tolist()
    return prompt, {"layer": L, "frame": frame, **rec}


def make_pool(workers: int):
    """A spawn pool whose workers each use one BLAS / OpenMP thread (None for one worker)."""
    if workers <= 1:
        return None
    import multiprocessing as mp
    old = {v: os.environ.get(v) for v in _ONE_THREAD}
    os.environ.update({v: "1" for v in _ONE_THREAD})
    try:
        return mp.get_context("spawn").Pool(workers)
    finally:
        for v, x in old.items():
            if x is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = x


def run_model(model, ids: Dict[str, List[int]], sets: Dict[str, Dict], keys: Sequence[str],
              pool, seed: int, n_draws: int) -> Dict[str, Dict]:
    hidden, t0 = {}, time.monotonic()
    for k in keys:
        hidden[k], _ = forward(model, ids[k])
    t_fwd = time.monotonic() - t0
    jobs = []
    for k in keys:
        kept = np.asarray(sets[k]["kept"])
        jobs += [(k, L, f, hidden[k][L][kept], kept, seed, n_draws) for L in LAYERS for f in FRAMES]
    res = pool.map(_job, jobs, chunksize=1) if pool is not None else [_job(j) for j in jobs]
    out = {k: {"layers": [], "seconds_forward_all": round(t_fwd, 1)} for k in keys}
    for k, r in res:
        out[k]["layers"].append(r)
    return out


# ---------------------------------------------------------------------------
# The first check
# ---------------------------------------------------------------------------

def mid_rank(obs: float, ref: np.ndarray) -> float:
    """Fraction of ``ref`` below ``obs``, ties counted half."""
    ref = np.asarray(ref, dtype=np.float64)
    return float((np.sum(ref < obs) + 0.5 * np.sum(ref == obs)) / ref.size)


def z_table(recs: Sequence[Dict]) -> Dict[Tuple, float]:
    """``{(model, prompt, layer, frame, stat): z}`` (None kept as None)."""
    out = {}
    for r in recs:
        for lay in r["layers"]:
            for s in STATS:
                out[(r["model"], r["prompt"], lay["layer"], lay["frame"], s)] = lay["stats"][s]["z"]
    return out


def first_check(recs: Sequence[Dict], reinits: Sequence[str], inits: Sequence[str],
                prompts: Sequence[str] = V1_PASSAGES) -> List[Dict]:
    """
    One row per (statistic, band, frame): the share of (init, prompt) median
    mid-ranks in the outer ``OUTER`` of the re-inits, and the verdict. A
    (prompt, layer) where any z is None is left out of the median and
    counted (``n_missing``); a cell with any missing value does not pass.
    """
    z = z_table(recs)
    rows = []
    for s in STATS:
        for frame in FRAMES:
            for b in BANDS:
                layers = [L for L in LAYERS if band(L) == b]
                med, n_missing, per_init = [], 0, {}
                for mid in inits:
                    for p in prompts:
                        us = []
                        for L in layers:
                            ref = [z.get((r, p, L, frame, s)) for r in reinits]
                            o = z.get((mid, p, L, frame, s))
                            if o is None or any(v is None for v in ref):
                                n_missing += 1
                                continue
                            us.append(mid_rank(o, np.asarray(ref)))
                        if us:
                            u = float(np.median(us))
                            med.append(u)
                            per_init.setdefault(mid, []).append(u)
                outer = [u < OUTER / 2 or u > 1 - OUTER / 2 for u in med]
                share = float(np.mean(outer)) if med else None
                ok = share is not None and n_missing == 0 and share <= FIRST_CHECK_BOUND
                rows.append({"stat": s, "frame": frame, "band": b, "n": len(med), "n_missing": n_missing,
                             "n_outer": int(sum(outer)), "share_outer": share,
                             "n_low": int(sum(u < OUTER / 2 for u in med)),
                             "n_high": int(sum(u > 1 - OUTER / 2 for u in med)),
                             "median_u": float(np.median(med)) if med else None,
                             "per_init_median_u": {m: round(float(np.median(v)), 3)
                                                   for m, v in per_init.items()},
                             "verdict": "pass" if ok else "fail (fall back to the 10 real inits)"})
    return rows


def raw_table(recs: Sequence[Dict], kinds=("init", "reinit")) -> List[Dict]:
    """Per (statistic, band, frame, kind): median observed value and median z, for reading beside the check."""
    cells: Dict[Tuple, Dict[str, List[float]]] = {}
    for r in recs:
        kind = r["model"].split(":")[0]
        for lay in r["layers"]:
            for s in STATS:
                c = cells.setdefault((s, lay["frame"], band(lay["layer"]), kind), {"obs": [], "z": []})
                c["obs"].append(lay["stats"][s]["obs"])
                if lay["stats"][s]["z"] is not None:
                    c["z"].append(lay["stats"][s]["z"])
    return [{"stat": s, "frame": f, "band": b, "kind": k,
             "median_obs": round(float(np.median(v["obs"])), 4),
             "median_z": round(float(np.median(v["z"])), 3) if v["z"] else None}
            for (s, f, b, k), v in sorted(cells.items()) if k in kinds]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_REPO)


def _mfile(mid: str) -> str:
    return mid.replace(":", "_")


def norms_cmd(argv: Optional[Sequence[str]] = None) -> int:
    """Forward pass per model and prompt; write each prompt's T2 table (no hidden states kept)."""
    ap = argparse.ArgumentParser(prog="arch_null norms")
    ap.add_argument("--step", default="step0")
    ap.add_argument("--models", choices=("init", "reinit", "all"), default="all")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    tok = _tok()
    ids = prompt_ids(tok)
    d = args.out / args.step / "norms"
    d.mkdir(parents=True, exist_ok=True)
    for mid in model_ids(args.models):
        f = d / f"{_mfile(mid)}.json"
        if f.exists():
            continue
        t0 = time.monotonic()
        model = load(mid, args.step)
        check_config(model)
        norms = {k: forward(model, ids[k])[1] for k in ids}
        tab = massive_table(norms)
        rec = {"model": mid, "step": args.step, "init": init_summary(model),
               "massive": {k: {str(p): list(v) for p, v in t.items()} for k, t in tab.items()},
               "position0_ratio_max": {k: float(max(n[L][0] / np.median(n[L]) for L in MASSIVE_LAYERS))
                                       for k, n in norms.items()},
               "nonpos0_ratio_max": {k: float(max(np.max(n[L][1:] / np.median(n[L])) for L in MASSIVE_LAYERS))
                                     for k, n in norms.items()},
               "git": _git_head()}
        f.write_text(json.dumps(rec) + "\n")
        print(f"norms {mid} {time.monotonic() - t0:.0f} s", flush=True)
        del model
    return 0


def _load_norms(out: Path, step: str, mids: Sequence[str]) -> Dict[str, Dict]:
    d = out / step / "norms"
    missing = [m for m in mids if not (d / f"{_mfile(m)}.json").exists()]
    if missing:
        raise SystemExit(f"refusing: T2 needs every model's norms; missing {missing}")
    return {m: json.loads((d / f"{_mfile(m)}.json").read_text()) for m in mids}


def comparison_models(step: str) -> List[str]:
    """The models whose T2 union fixes a step's token sets: step 0 reads no trained run."""
    if step != "step0":
        raise NotImplementedError("trained steps: after the first check (`design-1d.md` Unit 2)")
    return model_ids("all")


def run_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="arch_null run")
    ap.add_argument("--step", default="step0")
    ap.add_argument("--models", choices=("init", "reinit", "all"), default="all")
    ap.add_argument("--only", nargs="*", default=None, help="a subset of model ids")
    ap.add_argument("--keys", nargs="*", default=None, help="a subset of prompts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--torch-threads", type=int, default=None)
    args = ap.parse_args(argv)
    import torch
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    tok = _tok()
    ids = prompt_ids(tok)
    keys = args.keys or list(ids)
    comp = comparison_models(args.step)
    norms = _load_norms(args.out, args.step, comp)
    toks = {k: tok.convert_ids_to_tokens(ids[k]) for k in ids}
    sets = token_sets(toks, {m: {k: {int(p): tuple(v) for p, v in t.items()}
                                 for k, t in n["massive"].items()} for m, n in norms.items()})
    (args.out / args.step / "token_sets.json").write_text(json.dumps(
        {"comparison": comp, "sets": sets}, indent=1) + "\n")
    meta = {"git": _git_head(), "step": args.step, "seed": args.seed, "n_draws": args.n_draws,
            "settings": {"massive_ratio": MASSIVE_RATIO, "massive_layers": [MASSIVE_LAYERS[0], MASSIVE_LAYERS[-1]],
                         "frames": FRAMES, "min_cluster_sizes": MIN_CLUSTER_SIZES, "alpha": ALPHA,
                         "v1_max_tokens": V1_MAX_TOKENS, "comparison": comp}}
    mids = args.only or model_ids(args.models)
    pool = make_pool(args.workers)
    for mid in mids:
        d = args.out / args.step / _mfile(mid)
        todo = [k for k in keys if not (d / f"{k}.json").exists()]
        if not todo:
            print(f"already {d}", flush=True)
            continue
        t0 = time.monotonic()
        model = load(mid, args.step)
        check_config(model)
        res = run_model(model, ids, sets, todo, pool, args.seed, args.n_draws)
        del model
        d.mkdir(parents=True, exist_ok=True)
        for k, r in res.items():
            r.update(model=mid, prompt=k, step=args.step, n_kept=len(sets[k]["kept"]),
                     kept=sets[k]["kept"], massive=sets[k]["massive"], meta=meta)
            (d / f"{k}.json").write_text(json.dumps(r) + "\n")
        print(f"done {mid} {len(todo)} prompts {time.monotonic() - t0:.0f} s", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    return 0


def load_records(out: Path, step: str, mids: Sequence[str]) -> List[Dict]:
    recs = []
    for m in mids:
        for k in V1_PASSAGES:
            f = out / step / _mfile(m) / f"{k}.json"
            if f.exists():
                recs.append(json.loads(f.read_text()))
    return recs


def check_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="arch_null check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    inits, reinits = model_ids("init"), model_ids("reinit")
    recs = load_records(args.out, "step0", inits + reinits)
    want = len(V1_PASSAGES) * (len(inits) + len(reinits))
    if len(recs) != want:
        print(f"refusing: {len(recs)} of {want} step-0 records", file=sys.stderr)
        return 1
    rows = first_check(recs, reinits, inits)
    res = {"rows": rows, "raw": raw_table(recs), "git": sorted({r["meta"]["git"] for r in recs}),
           "n_draws": sorted({r["meta"]["n_draws"] for r in recs}),
           "bound": FIRST_CHECK_BOUND, "outer": OUTER, "flagged_seeds": FLAGGED_SEEDS}
    (args.out / "first_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"first check: share of (init, prompt) band-median ranks in the re-inits' outer "
          f"{OUTER:.0%}; pass <= {FIRST_CHECK_BOUND:.0%}")
    for r in rows:
        print(f"  {r['stat']:8s} {r['frame']:8s} {r['band']:7s} n={r['n']:2d} miss={r['n_missing']:2d} "
              f"outer={r['n_outer']:2d} (low {r['n_low']}, high {r['n_high']}) "
              f"share={r['share_outer'] if r['share_outer'] is None else round(r['share_outer'], 3)} "
              f"median_u={r['median_u'] if r['median_u'] is None else round(r['median_u'], 3)}  {r['verdict']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"norms": norms_cmd, "run": run_cmd, "check": check_cmd}
    if not argv or argv[0] not in cmds:
        print(f"usage: arch_null {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
