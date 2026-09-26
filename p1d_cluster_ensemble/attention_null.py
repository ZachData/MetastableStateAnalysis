"""
p1d_cluster_ensemble/attention_null.py — attention communities against
three nulls, per (run, layer). Item (3) of 1d; the graph and nulls B and C
are `attention_graph.py`'s, the model's block is `neox_block.py`'s.

Everything is in *sequence-local* form: index 0 is the sink (position 0),
then the kept tokens at their original positions (all of them, or each
token string's first occurrence with ``--dedupe-strings``). The sink is in
every softmax and dropped from every graph.

Per job:

- **real**: the stored attention (layers ``l .. l+w-1``), its graph set, and
  each head's β on the unit LN1 frame (`attention_graph.fit_betas`).
- **C-real**: the idealised head with those β on the real tokens: one
  deterministic "cosine-predicted attention". ``ari_kernel`` is how far the
  real communities are what cosine predicts.
- **null A**: tokens replaced by a Gaussian with their covariance
  (`gaussian_null.gaussian_draw`, fitted on the kept non-sink unit rows),
  each row scaled to its real token's norm, the real sink kept at position
  0, read by the checkpoint's own blocks. For ``w > 1`` the draw is carried
  through the real blocks, not redrawn, so a token stays one token.
- **null C**: the same propagated draws, attention replaced by the idealised
  head with the real β.
- **null B**: each real layer's chain diagonal-shuffled.

``--calibrate`` replaces the real run by one null-A draw (seed offset) and
refits everything to it, #106's convention: each real result is read
against a calibration on its own inputs.

``--verify`` recomputes a stored layer's attention and next residual from
the stored residual and refuses to run if they do not match.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.holdout import add_holdout_args, refuse_held_out

from .attention_graph import (SYMMETRISATIONS, WINDOWS, diagonal_shuffle, fit_betas,
                              graph_set, head_mean, kernel_attention, unit_ln_rows)
from .gaussian_null import (CALIBRATE_SEED_OFFSET, FRAMES, MIN_TOKENS, _unit_rows,
                            band_of, compare, first_occurrences, frame_vectors,
                            gaussian_draw, run_tokens)

NULLS = ("A", "B", "C")
#: Seed offsets per null, so A/C and B never share a stream.
NULL_SEED = {"A": 0, "B": 1_000}
#: PLACED. Tolerances for ``--verify`` (float32; measured 2026-09-26 on
#: wiki_paragraph: attention max abs <= 1.5e-3 at L18/L23, block rel <= 3.3e-5).
VERIFY_ATTN_ABS = 5e-3
VERIFY_BLOCK_REL = 1e-3


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def run_checkpoint(run_dir: Path) -> Tuple[str, str]:
    """(HF repo, revision) from the run's manifest."""
    man = json.loads((Path(run_dir) / "manifest.json").read_text())
    model = man["model"]
    base = model.rsplit("-step", 1)[0]
    if not base.startswith("pythia"):
        raise ValueError(f"{run_dir}: only Pythia is implemented, got {model!r}")
    return f"EleutherAI/{base}", str(man["hf_revision"])


def _step_prompt(run_dir: Path) -> Tuple[str, str]:
    model, _, prompt = Path(run_dir).name.partition("_")
    return model.rsplit("-", 1)[-1], prompt


def weights_dir(root: Path, revision: str) -> Path:
    return Path(root) / revision


def ensure_weights(root: Path, runs: Sequence[Path]) -> None:
    from .neox_block import export_blocks
    for repo, rev in sorted({run_checkpoint(r) for r in runs}):
        d = weights_dir(root, rev)
        if not (d / "source.txt").exists():
            n = export_blocks(repo, rev, d)
            print(f"  exported {n} blocks of {repo}@{rev} to {d}")


def blocks_for(root: Path, revision: str, layers: Sequence[int]) -> Dict[int, object]:
    from .neox_block import read_block
    d = weights_dir(root, revision)
    return {l: read_block(d / f"block_{l}.npz") for l in layers}


# ---------------------------------------------------------------------------
# One world (real, pseudo-real or a null draw), in sequence-local form
# ---------------------------------------------------------------------------

def model_world(rows_l: np.ndarray, positions: np.ndarray, blocks: Dict[int, object],
                layer: int, n_layers: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Attention (heads, m+1, m+1) and input residual at layers ``l .. l+n_layers-1``."""
    from .neox_block import attention_probs, block_forward
    X, atts, xs = np.asarray(rows_l, dtype=np.float32), [], []
    for k in range(n_layers):
        B = blocks[layer + k]
        atts.append(attention_probs(X, B, positions))
        xs.append(X)
        if k < n_layers - 1:
            X = block_forward(X, B, positions)
    return atts, xs


def draw_rows(Y_unit: np.ndarray, norms: np.ndarray, x0: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    """A null-A input: the real sink row, then Gaussian unit rows at their tokens' norms."""
    D = gaussian_draw(Y_unit, rng) * norms[:, None]
    return np.vstack([x0[None, :], D])


def frames_of(Y_unit: np.ndarray) -> Dict[str, np.ndarray]:
    return {f: frame_vectors(Y_unit, f)[0] for f in FRAMES}


def world_graphs(atts: Sequence[np.ndarray], positions: np.ndarray,
                 frames: Dict[str, np.ndarray], seed: int, keep_labels: bool = False
                 ) -> Tuple[Dict, List[np.ndarray]]:
    m1 = atts[0].shape[-1]
    local = np.arange(1, m1)
    mats = [head_mean(a, local) for a in atts]
    return graph_set(mats, positions[1:], frames, seed=seed, keep_labels=keep_labels), mats


def kernel_mats(xs: Sequence[np.ndarray], blocks: Dict[int, object], layer: int,
                betas: Sequence[Sequence[float]]) -> List[np.ndarray]:
    out = []
    for k, X in enumerate(xs):
        B = blocks[layer + k]
        U = unit_ln_rows(np.asarray(X)[1:], B.ln1_w, B.ln1_b, B.eps)
        out.append(kernel_attention(U, betas[k]).mean(axis=0))
    return out


def world_betas(atts: Sequence[np.ndarray], xs: Sequence[np.ndarray],
                blocks: Dict[int, object], layer: int, positions: np.ndarray) -> List[List[Dict]]:
    out = []
    for k, (A, X) in enumerate(zip(atts, xs)):
        B = blocks[layer + k]
        U = unit_ln_rows(X, B.ln1_w, B.ln1_b, B.eps)
        out.append(fit_betas(A, U, np.arange(1, A.shape[-1]), positions))
    return out


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

def _ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def _job(args: Tuple) -> Dict:
    (run_dir, layer, n_draws, seed, calibrate, dedupe, wroot, nulls) = args
    import torch
    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    step, prompt = _step_prompt(run_dir)
    z = np.load(run_dir / "activations.npz")
    acts, norms_all = z["activations"], z["norms"]
    n_blocks = acts.shape[0] - 1
    n = acts.shape[1]
    base = {"run_dir": str(run_dir), "step": step, "prompt": prompt, "layer": int(layer),
            "n_tokens": int(n)}
    keep = np.arange(1, n)
    if dedupe:
        tokens = run_tokens(run_dir)
        if len(tokens) != n:
            raise ValueError(f"{run_dir}: {len(tokens)} token strings for {n} rows; refusing to dedupe")
        keep = first_occurrences(tokens)
        keep = keep[keep != 0]
        base["n_kept"] = int(keep.size)
        if keep.size < MIN_TOKENS:
            return {**base, "skipped": f"{keep.size} distinct strings < {MIN_TOKENS}"}
    seq = np.concatenate([[0], keep])
    positions = seq.astype(int)
    n_layers = min(max(WINDOWS), n_blocks - layer)
    windows = [w for w in WINDOWS if w <= n_layers]
    _, rev = run_checkpoint(run_dir)
    blocks = blocks_for(wroot, rev, range(layer, layer + n_layers))
    t0 = time.time()

    # The real world, or its pseudo-real replacement.
    X_real = [acts[layer + k][seq].astype(np.float64) * norms_all[layer + k][seq][:, None]
              for k in range(n_layers)]
    if calibrate:
        Yr = _unit_rows(acts[layer][keep])
        rows = draw_rows(Yr, norms_all[layer][keep], X_real[0][0],
                         np.random.default_rng(seed + CALIBRATE_SEED_OFFSET))
        atts, xs = model_world(rows, positions, blocks, layer, n_layers)
        base["calibrate"] = True
    else:
        A_all = np.load(run_dir / "attentions.npz")["attentions"]
        atts = [A_all[layer + k][:, seq[:, None], seq[None, :]].astype(np.float64)
                for k in range(n_layers)]
        del A_all
        xs = X_real
    Y_unit = _unit_rows(np.asarray(xs[0])[1:])
    norms = np.linalg.norm(np.asarray(xs[0], dtype=np.float64)[1:], axis=1)
    x0 = np.asarray(xs[0], dtype=np.float64)[0]

    betas_full = world_betas(atts, xs, blocks, layer, positions)
    betas = [[h["beta"] for h in bl] for bl in betas_full]
    real, real_mats = world_graphs(atts, positions, frames_of(Y_unit), seed, keep_labels=True)
    kern = graph_set(kernel_mats(xs, blocks, layer, betas), positions[1:], None,
                     windows, seed, keep_labels=True)
    for g in real:
        real[g]["ari_kernel"] = _ari(real[g]["labels"], kern[g]["labels"])

    draws: Dict[str, Dict[str, Dict[str, List[float]]]] = {k: {} for k in nulls}

    def add(null, gs):
        for g, st in gs.items():
            for s, v in st.items():
                if s != "labels":
                    draws[null].setdefault(g, {}).setdefault(s, []).append(v)

    if "A" in nulls or "C" in nulls:
        rng = np.random.default_rng(seed + NULL_SEED["A"])
        for _ in range(int(n_draws)):
            rows = draw_rows(Y_unit, norms, x0, rng)
            a_atts, a_xs = model_world(rows, positions, blocks, layer, n_layers)
            fr = frames_of(_unit_rows(rows[1:]))
            if "A" in nulls:
                add("A", world_graphs(a_atts, positions, fr, seed)[0])
            if "C" in nulls:
                add("C", graph_set(kernel_mats(a_xs, blocks, layer, betas), positions[1:],
                                   fr, windows, seed))
    if "B" in nulls:
        rng = np.random.default_rng(seed + NULL_SEED["B"])
        real_frames = frames_of(Y_unit)
        for _ in range(int(n_draws)):
            mats = [diagonal_shuffle(M, rng) for M in real_mats]
            gs = graph_set(mats, positions[1:], real_frames, windows, seed, keep_labels=True)
            for g in gs:
                gs[g]["ari_real"] = _ari(gs[g].pop("labels"), real[g]["labels"])
            add("B", gs)

    stats = {}
    for null in nulls:
        for g, per in draws[null].items():
            for s, vals in per.items():
                if s == "ari_real":
                    stats.setdefault(null, {}).setdefault(g, {})[s] = {
                        "null_mean": float(np.nanmean(vals)), "q975": float(np.nanquantile(vals, 0.975))}
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size < 2 or not np.isfinite(real[g].get(s, np.nan)):
                    continue
                stats.setdefault(null, {}).setdefault(g, {})[s] = compare(real[g][s], arr)
    return {**base, "windows": windows, "n_draws": int(n_draws), "seed": int(seed),
            "betas": [[{k: (None if v is None else (round(v, 5) if isinstance(v, float) else v))
                        for k, v in h.items()} for h in bl] for bl in betas_full],
            "real": real, "kernel_real": {g: {k: v for k, v in st.items() if k != "labels"}
                                          for g, st in kern.items()},
            "stats": stats, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# Verify the map before trusting a null built on it
# ---------------------------------------------------------------------------

def verify(run_dir: Path, wroot: Path, layers: Sequence[int]) -> Dict:
    from .neox_block import attention_probs, block_forward
    z = np.load(Path(run_dir) / "activations.npz")
    X = z["activations"].astype(np.float64) * z["norms"][..., None]
    A = np.load(Path(run_dir) / "attentions.npz")["attentions"]
    _, rev = run_checkpoint(Path(run_dir))
    out = {"run_dir": str(run_dir), "layers": {}}
    blocks = blocks_for(wroot, rev, layers)
    for l in layers:
        P = attention_probs(X[l], blocks[l])
        rec = {"attn_max_abs": float(np.abs(P - A[l]).max())}
        if l < A.shape[0] - 1:
            H = block_forward(X[l], blocks[l])
            rel = np.linalg.norm(H - X[l + 1], axis=1) / np.linalg.norm(X[l + 1], axis=1)
            rec["block_rel_max"] = float(rel.max())
        out["layers"][int(l)] = rec
    out["ok"] = all(r["attn_max_abs"] <= VERIFY_ATTN_ABS
                    and r.get("block_rel_max", 0.0) <= VERIFY_BLOCK_REL
                    for r in out["layers"].values())
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

#: The tail read as "more community structure / agreement than the null".
TAIL = {"Q": "p_upper", "lam2": "p_lower", "ari_raw": "p_upper", "ari_centred": "p_upper",
        "ari_centred_norogue": "p_upper", "k4": "p_upper", "contig": "p_upper",
        "local": "p_upper"}


def summarise(records: List[Dict], alpha: float = 0.025) -> List[Dict]:
    rows: Dict[Tuple, Dict] = {}
    for r in records:
        if "skipped" in r:
            continue
        for null, per_g in r["stats"].items():
            for g, per_s in per_g.items():
                for s, c in per_s.items():
                    if s not in TAIL:
                        continue
                    key = (r["step"], null, g, r.get("band") or band_of(r["layer"]), s)
                    row = rows.setdefault(key, {"step": key[0], "null": null, "graph": g,
                                                "band": key[3], "stat": s, "n": 0,
                                                "tail": 0, "z": [], "obs": [], "null_mean": []})
                    row["n"] += 1
                    row["tail"] += int(c[TAIL[s]] <= alpha)
                    row["obs"].append(c["obs"])
                    row["null_mean"].append(c["null_mean"])
                    if c["null_sd"] > 0:
                        row["z"].append((c["obs"] - c["null_mean"]) / c["null_sd"])
    out = []
    for row in rows.values():
        z, o, nm = row.pop("z"), row.pop("obs"), row.pop("null_mean")
        row["median_z"] = float(np.median(z)) if z else float("nan")
        row["median_obs"] = float(np.median(o))
        row["median_null"] = float(np.median(nm))
        out.append(row)
    return sorted(out, key=lambda r: (r["stat"], r["null"], r["graph"], r["step"], r["band"]))


def summary_text(out: Dict) -> str:
    skipped = sum("skipped" in r for r in out["records"])
    head = ("CALIBRATION (tokens replaced by one null-A draw): " if out.get("calibrate") else "") \
        + ("DEDUPED: " if out.get("dedupe_strings") else "")
    lines = [head + f"Attention communities vs nulls {','.join(out['nulls'])}: "
             f"{out['n_draws']} draws, seed {out['seed']}, "
             f"{len(out['records']) - skipped} layer-records ({skipped} skipped)",
             "tail = records in the null's 2.5 % tail on the stronger side "
             + ", ".join(f"{k} {v}" for k, v in TAIL.items())]
    stat = None
    for r in out["summary"]:
        if r["stat"] != stat:
            stat = r["stat"]
            lines.append(f"\n[{stat}] null graph        step       band     n tail  med obs  med null  med z")
        lines.append(f"        {r['null']:<4} {r['graph']:<12} {r['step']:<10} {r['band']:<7} "
                     f"{r['n']:>3} {r['tail']:>4}  {r['median_obs']:>7.3f}  {r['median_null']:>8.3f}"
                     f"  {r['median_z']:>6.2f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="JSON to write (summary beside it)")
    ap.add_argument("--weights", type=Path, required=True,
                    help="directory for per-block weight files (under data/, never committed)")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="default: every block")
    ap.add_argument("--nulls", nargs="*", default=list(NULLS), choices=NULLS)
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--dedupe-strings", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="only check the block map against the first run and exit")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    add_holdout_args(ap)
    args = ap.parse_args(argv)

    runs, record = refuse_held_out(list(args.runs), allow=args.allow_holdout,
                                   drop=args.v1_only, context="attention_null")
    missing = [r for r in runs if not ((r / "activations.npz").exists()
                                       and (r / "attentions.npz").exists())]
    if missing or not runs:
        print(f"refusing: no runs, or missing activations/attentions in {missing}", file=sys.stderr)
        return 1
    ensure_weights(args.weights, runs)
    checks = [verify(r, args.weights, args.layers or [0, 6, 12, 18, 23])
              for r in {run_checkpoint(r)[1]: r for r in runs}.values()]
    for c in checks:
        print(f"  verify {Path(c['run_dir']).name}: ok={c['ok']} "
              + " ".join(f"L{l}:{v['attn_max_abs']:.1e}/{v.get('block_rel_max', 0):.1e}"
                         for l, v in c["layers"].items()))
    if not all(c["ok"] for c in checks):
        print("refusing: the block map does not reproduce the stored run", file=sys.stderr)
        return 1
    if args.verify:
        return 0

    jobs = []
    for r in runs:
        n_blocks = np.load(r / "activations.npz")["activations"].shape[0] - 1
        for L in (args.layers if args.layers else range(n_blocks)):
            jobs.append((str(r), int(L), args.n_draws, args.seed, args.calibrate,
                         args.dedupe_strings, str(args.weights), tuple(args.nulls)))
    # Each finished record is written at once, so a stopped run resumes where it
    # stopped (a 3 h run lost everything to a stop on 2026-09-26). A part is
    # reused only if its settings match this call's.
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    settings = {"n_draws": args.n_draws, "seed": args.seed, "calibrate": bool(args.calibrate),
                "dedupe_strings": bool(args.dedupe_strings), "nulls": list(args.nulls)}
    (parts / "settings.json").write_text(json.dumps(settings))

    def part_of(job) -> Path:
        return parts / f"{Path(job[0]).parent.name}__{Path(job[0]).name}__L{job[1]}.json"

    done, todo = [], []
    for j in jobs:
        p = part_of(j)
        if p.exists():
            rec = json.loads(p.read_text())
            if rec.get("_settings") == settings:
                done.append(rec)
                continue
        todo.append(j)
    print(f"  {len(done)} of {len(jobs)} layer-records already done; running {len(todo)}", flush=True)

    def keep(job, rec):
        rec["_settings"] = settings
        tmp = part_of(job).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(part_of(job))
        done.append(rec)
        if len(done) % 12 == 0 or len(done) == len(jobs):
            print(f"  {len(done)}/{len(jobs)} done ({time.time() - t0:.0f} s this call)", flush=True)

    t0 = time.time()
    if args.workers > 1 and todo:
        from multiprocessing import get_context
        by_key = {(j[0], j[1]): j for j in todo}
        with get_context("spawn").Pool(args.workers) as pool:
            for rec in pool.imap_unordered(_job, todo, chunksize=1):
                keep(by_key[(rec["run_dir"], rec["layer"])], rec)
    else:
        for job in todo:
            keep(job, _job(job))
    order = {(j[0], j[1]): i for i, j in enumerate(jobs)}
    records = sorted(done, key=lambda r: order[(r["run_dir"], r["layer"])])
    for r in records:
        r.pop("_settings", None)
    out = {"n_draws": args.n_draws, "seed": args.seed, "nulls": list(args.nulls),
           "windows": list(WINDOWS), "symmetrisations": list(SYMMETRISATIONS),
           "calibrate": bool(args.calibrate), "dedupe_strings": bool(args.dedupe_strings),
           "verify": checks, "holdout": record, "inputs": [str(r) for r in runs],
           "seconds": round(time.time() - t0, 1), "records": records}
    out["summary"] = summarise(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    text = summary_text(out)
    args.out.with_suffix(".txt").write_text(text + "\n")
    print(text)
    print(f"  wrote {args.out} ({out['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
