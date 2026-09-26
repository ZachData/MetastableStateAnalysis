"""
p1d_cluster_ensemble/viz_page.py — the data behind 1d's interactive page
(`p1d_cluster_ensemble/viz/index.html`): which frame, and which null, shows
the strongest structure.

Reads the Gaussian-null outputs (#106, `gaussian_null.py`) and, when present,
item (3)'s attention-null outputs (`attention_null.py`), and writes
``summary.json`` plus one ``tokens_<step>_<prompt>.json`` per run (2-D
projections, groups, attention communities), which the page fetches lazily.

z is oriented so that **positive = more structure than the null** (lumpier,
or more modular / more agreement); a tail flag marks p <= 0.025 on that side.

    python -m p1d_cluster_ensemble.viz_page \
      --gauss <main>/data/p1d/gaussian_null_2026-09-26 \
      --attn <main>/data/p1d/attention_null_2026-09-26 --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .gaussian_null import (FRAMES, STATISTICS, _unit_rows, first_occurrences,
                            frame_vectors, gaussian_draw, run_tokens)

ALPHA = 0.025
#: The lumpier side of each Gaussian-null statistic (`gaussian_null_report.TAIL`).
G_SIDE = {"ci2": "lower", "nn1": "lower", "hdb_noise": "lower", "mt_life": "upper",
          "hdb_k": "upper", "mt_k": "upper"}
#: The stronger side of each attention statistic (`attention_null.TAIL`).
A_SIDE = {"Q": "upper", "lam2": "lower", "ari_raw": "upper", "ari_centred": "upper",
          "ari_centred_norogue": "upper", "k4": "upper", "contig": "upper", "local": "upper"}
PROJ_LAYERS = (1, 4, 8, 12, 16, 20, 24)
GAUSS_FILES = {("all", "real"): "null", ("all", "cal"): "calibrate",
               ("dedupe", "real"): "null_dedupe", ("dedupe", "cal"): "calibrate_dedupe"}


def _oriented(c: Dict, side: str):
    if not c or not c.get("null_sd"):
        return None, 0
    z = (c["obs"] - c["null_mean"]) / c["null_sd"]
    p = c["p_upper"] if side == "upper" else c["p_lower"]
    return round(float(z if side == "upper" else -z), 2), int(p <= ALPHA)


def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------------------
# Gaussian null (frames)
# ---------------------------------------------------------------------------

def gauss_summary(gdir: Path) -> Dict:
    out: Dict = {"cells": {}, "geometry": {}, "prompts": set(), "steps": set()}
    for (tokset, kind), name in GAUSS_FILES.items():
        f = gdir / f"{name}.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text())["records"]:
            if "skipped" in r:
                continue
            frame = r["info"]["frame"]
            out["prompts"].add(r["prompt"]); out["steps"].add(r["step"])
            for s in STATISTICS:
                z, t = _oriented(r["stats"][s], G_SIDE[s])
                key = f"{tokset}|{kind}|{r['step']}|{frame}|{s}"
                out["cells"].setdefault(key, {}).setdefault(r["prompt"], {})[r["layer"]] = [z, t]
            if kind == "real":
                gk = f"{tokset}|{r['step']}|{frame}"
                out["geometry"].setdefault(gk, {}).setdefault(r["prompt"], {})[r["layer"]] = [
                    _r(r["info"]["mean_share"]), _r(r["info"]["eff_dim"], 1)]
    out["prompts"], out["steps"] = sorted(out["prompts"]), sorted(out["steps"], key=lambda s: int(s[4:]))
    return out


# ---------------------------------------------------------------------------
# Attention null
# ---------------------------------------------------------------------------

def attn_summary(adir: Path) -> Optional[Dict]:
    files = {("all", "real"): "null", ("all", "cal"): "calibrate",
             ("dedupe", "real"): "null_dedupe", ("dedupe", "cal"): "calibrate_dedupe"}
    if not any((adir / f"{n}.json").exists() for n in files.values()):
        return None
    out: Dict = {"cells": {}, "values": {}, "betas": {}, "graphs": set(), "meta": {}}
    for (tokset, kind), name in files.items():
        f = adir / f"{name}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        out["meta"][f"{tokset}|{kind}"] = {"n_draws": d["n_draws"], "seed": d["seed"],
                                           "verify": d.get("verify")}
        for r in d["records"]:
            if "skipped" in r:
                continue
            for null, per_g in r["stats"].items():
                for g, per_s in per_g.items():
                    out["graphs"].add(g)
                    for s, c in per_s.items():
                        if s not in A_SIDE:
                            continue
                        z, t = _oriented(c, A_SIDE[s])
                        key = f"{tokset}|{kind}|{r['step']}|{null}|{g}|{s}"
                        out["cells"].setdefault(key, {}).setdefault(r["prompt"], {})[r["layer"]] = [z, t]
            for g, st in r["real"].items():
                vk = f"{tokset}|{kind}|{r['step']}|{g}"
                row = {k: _r(v) for k, v in st.items() if k != "labels"}
                for null in ("A", "B", "C"):
                    for s in ("Q", "ari_raw", "ari_centred", "ari_centred_norogue"):
                        c = r["stats"].get(null, {}).get(g, {}).get(s)
                        if c:
                            row[f"{null}_{s}"] = _r(c["null_mean"])
                    b = r["stats"].get("B", {}).get(g, {}).get("ari_real")
                    if b and null == "B":
                        row["B_ari_real"] = _r(b["null_mean"])
                for k, v in r["kernel_real"].get(g, {}).items():
                    row[f"K_{k}"] = _r(v)
                out["values"].setdefault(vk, {}).setdefault(r["prompt"], {})[r["layer"]] = row
            bk = f"{tokset}|{kind}|{r['step']}"
            b0 = [h for h in r["betas"][0]]
            bs = np.array([np.nan if h["beta"] is None else h["beta"] for h in b0], float)
            r2 = np.array([np.nan if h["r2"] is None else h["r2"] for h in b0], float)
            out["betas"].setdefault(bk, {}).setdefault(r["prompt"], {})[r["layer"]] = {
                "beta": [_r(x, 2) for x in bs], "r2": [_r(x, 2) for x in r2]}
    out["graphs"] = sorted(out["graphs"])
    return out


def attn_labels(adir: Path) -> Dict:
    """Real community labels per (step, prompt, layer, graph), all tokens only."""
    f = adir / "null.json"
    if not f.exists():
        return {}
    out: Dict = {}
    for r in json.loads(f.read_text())["records"]:
        if "skipped" in r:
            continue
        for g, st in r["real"].items():
            out.setdefault((r["step"], r["prompt"]), {}).setdefault(str(r["layer"]), {})[g] = st["labels"]
    return out


# ---------------------------------------------------------------------------
# Per-run token files
# ---------------------------------------------------------------------------

def _pca2(Z: np.ndarray, *others: np.ndarray):
    mu = Z.mean(axis=0)
    _, _, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    P = Vt[:2].T
    return [np.round((X - mu) @ P, 3).tolist() for X in (Z,) + others]


def token_file(run_dir: Path, labels: Dict, seed: int = 0) -> Dict:
    from .methods import LayerData, _fit_hdbscan
    acts = np.load(run_dir / "activations.npz")["activations"]
    toks = run_tokens(run_dir)
    first = set(first_occurrences(toks).tolist())
    out = {"tokens": toks, "repeat": [0 if i in first else 1 for i in range(len(toks))],
           "proj": {}, "communities": labels}
    for L in PROJ_LAYERS:
        if L >= acts.shape[0]:
            continue
        for f in FRAMES:
            Z, _ = frame_vectors(acts[L], f)
            D = gaussian_draw(Z, np.random.default_rng(seed + L))
            hdb = _fit_hdbscan({"min_cluster_size": 2}, LayerData.from_normed(Z), seed)
            real, draw = _pca2(Z, D)
            out["proj"][f"{L}|{f}"] = {"real": real, "draw": draw, "hdb": hdb.tolist()}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gauss", type=Path, required=True)
    ap.add_argument("--attn", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-tokens", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    g = gauss_summary(args.gauss)
    a = attn_summary(args.attn) if args.attn else None
    summary = {"gauss": g, "attn": a, "frames": list(FRAMES), "g_stats": list(STATISTICS),
               "a_stats": list(A_SIDE), "proj_layers": list(PROJ_LAYERS)}
    (args.out / "summary.json").write_text(json.dumps(summary, separators=(",", ":")))
    if not args.skip_tokens:
        labels = attn_labels(args.attn) if args.attn else {}
        inputs = json.loads((args.gauss / "null.json").read_text())["inputs"]
        for rd in inputs:
            rd = Path(rd)
            step = rd.name.partition("_")[0].rsplit("-", 1)[-1]
            prompt = rd.name.partition("_")[2]
            tf = token_file(rd, labels.get((step, prompt), {}))
            (args.out / f"tokens_{step}_{prompt}.json").write_text(json.dumps(tf, separators=(",", ":")))
            print(f"  wrote tokens_{step}_{prompt}.json")
    print(f"  wrote {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
