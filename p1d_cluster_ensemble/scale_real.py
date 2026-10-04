"""
p1d_cluster_ensemble/scale_real.py — unit 3's real-input reader, step 1:
specificity on clouds with nothing learned (`design-1d.md` "The real-input
reader, step 1"; every rule fixed there before any code).

**Clouds**: step 0 of unit 2's 50 models (`arch_null.model_ids("all")`: the
10 real inits and the 40 float16 Pythia-σ re-inits), the 7 v1 prompts as
every v1 run read them, L1–24, the kept tokens of unit 2's **trained** T2
union (``token_sets.json`` of ``arch_null_trained_2026-10-02/step0``, read,
never recomputed), **centred frame only**.

**Per cloud** (`run`): `scale_spectrum.spectrum`, the reader the synthetic
checked (grid, (a) stability, 50 matched Gaussian draws, (iv) informative),
with the same seed for the same (prompt, layer) in every model (`cloud_seed`;
common random numbers, as unit 2). Stored per (model, prompt): every row's
stability ``z_G`` per (``r``, arm), and the cuts' labels (the continuity rule
needs them once (b) is known).

**(b) on real input** (`read`): at each (prompt, layer, ``r``, arm), the
cloud's ``z_G`` ranked (higher tail) among a reference set's (`rank_rows`):

- **real-init arm**: each real init among the 40 re-inits;
- **re-init arm**: each re-init among the other 39.

A reference whose cut has no cluster of the arm's size counts as below the
cloud; one whose ``z_G`` is undefined (draws' SD 0) is dropped and N is
reported; a point is admissible only with its own ``z_G`` defined, (iv)
informative and N ``>= MIN_REF`` (`ref_values`, `rank_rows`). The plateau is
`scale_spectrum.robust_plateaus` with that p; the main arm gates.

**Pass** (`verdicts`), per band L1–8 / 9–16 / 17–24:

- **rate**: the re-init arm's share of clouds with a main-arm plateau is
  ``<= MAX_RATE``; where it is not, ``MIN_RUN`` is raised one step at a time
  up to ``MAX_MIN_RUN``, re-init arm only; still failing, the band refuses;
- **route**: the real-init arm's share, at the band's ``MIN_RUN``, is ``<=``
  the 95th percentile of the share over ``N_SUBSETS`` random 10-of-40 subsets
  of the re-inits' own readings.

A band that fails either refuses the trained reading there. Beside, not the
pass: the same without (b), the size-2 arm, plateaus per prompt with their
``r`` ranges. The synthetic's sensitivity re-read at a raised ``MIN_RUN`` (the
design's post-hoc line) is not built here; `read` says when it is owed.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .arch_null import (BANDS, _git_head, _mfile, _tok, check_config, comparison_models, label,
                        load, make_pool, model_ids, prompt_ids)
from .move_text import LAYERS, V1_PASSAGES, band
from .scale_spectrum import (ALPHA, ARMS, GATING_ARM, GRID_HI, GRID_LO, GRID_N, MIN_RUN, N_DRAWS,
                             N_SUBSAMPLES, robust_plateaus, spectrum)

FRAME = "centred"
#: A point is admissible only with at least this many references left (design row "missing z").
MIN_REF = 30
#: Pass, rate (re-init arm), per band.
MAX_RATE = 0.05
#: If the rate fails in a band, MIN_RUN is raised one step at a time up to this.
MAX_MIN_RUN = 5
#: Pass, route: the real-init share against this quantile of random re-init subsets.
N_SUBSETS = 1000
ROUTE_QUANTILE = 0.95
SUBSET_SIZE = 10
#: Seed base for the per-(prompt, layer) streams; apart from the synthetic's seeds (0-51).
SEED_BASE = 10_000
#: The token sets: unit 2's trained union (design row "clouds").
DEFAULT_SETS = Path("p1d/arch_null_trained_2026-10-02/step0/token_sets.json")


def cloud_seed(prompt: str, layer: int) -> int:
    """One seed per (prompt, layer), the same in every model (common random numbers)."""
    return SEED_BASE + 100 * V1_PASSAGES.index(prompt) + int(layer)


# ---------------------------------------------------------------------------
# Run: one model's clouds
# ---------------------------------------------------------------------------

def read_cloud(Y: np.ndarray, seed: int) -> Tuple[Dict, Optional[np.ndarray]]:
    """
    One cloud's spectrum (centred), its rows per arm, and beside it the plateaus
    against its own Gaussian draws and without (b). Labels ``(GRID_N, n)``. A merge
    tree that refuses is recorded as ``error`` (`read` then refuses), not raised.
    """
    try:
        spec = spectrum(Y, FRAME, seed)
    except RuntimeError as e:
        return {"error": str(e)}, None
    labels = spec["_labels"]
    beside = {a: {"plateaus_gauss": robust_plateaus(spec["rows"][a], labels, min_size=s),
                  "plateaus_without_b": robust_plateaus(spec["rows"][a], labels, use_p=False, min_size=s)}
              for a, s in ARMS.items()}
    rec = {"n": spec["n"], "median": spec["median"], "info": spec["info"], "rows": spec["rows"],
           "beside": beside}
    return rec, np.stack(labels).astype(np.uint16)


def _job(args: Tuple[str, str, int, np.ndarray]) -> Tuple[str, str, int, Dict, Optional[np.ndarray]]:
    mid, prompt, L, Y = args
    rec, lab = read_cloud(Y, cloud_seed(prompt, L))
    return mid, prompt, L, rec, lab


def load_sets(path: Path) -> Dict[str, List[int]]:
    """The kept positions per prompt; refuses unless the file is the trained union's."""
    d = json.loads(path.read_text())
    want = [label(st, m) for st, m in comparison_models("trained")]
    if d.get("comparison") != want:
        raise SystemExit(f"refusing: {path} is not unit 2's trained T2 union")
    if set(d["sets"]) != set(V1_PASSAGES):
        raise SystemExit(f"refusing: {path} prompts {sorted(d['sets'])} are not the 7 v1 prompts")
    return {k: v["kept"] for k, v in d["sets"].items()}


def _paths(out: Path, mid: str, prompt: str) -> Tuple[Path, Path]:
    d = out / _mfile(mid)
    return d / f"{prompt}.json", d / f"{prompt}.labels.npz"


def _write(out: Path, mid: str, prompt: str, layers: Dict[int, Tuple[Dict, Optional[np.ndarray]]],
           meta: Dict) -> None:
    """Labels first, the JSON last: a JSON on disk means the pair is complete."""
    fj, fl = _paths(out, mid, prompt)
    fj.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fl, **{f"L{L}": lab for L, (_, lab) in layers.items() if lab is not None})
    recs = [{"layer": L, **layers[L][0]} for L in sorted(layers)]
    fj.write_text(json.dumps({"model": mid, "prompt": prompt, "frame": FRAME, "layers": recs,
                              "meta": meta}) + "\n")


def run_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="scale_real run")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", type=Path, default=None,
                    help=f"token_sets.json (default $METS_DATA/{DEFAULT_SETS})")
    ap.add_argument("--only", nargs="*", default=None, help="a subset of model ids")
    ap.add_argument("--keys", nargs="*", default=None, help="a subset of prompts")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args(argv)
    sets_path = args.sets or Path(os.environ["METS_DATA"]) / DEFAULT_SETS
    kept = load_sets(sets_path)
    import torch
    torch.set_num_threads(args.torch_threads)
    from .move_text import forward
    tok = _tok()
    ids = prompt_ids(tok)
    keys = args.keys or list(V1_PASSAGES)
    mids = args.only or model_ids("all")
    meta = {"git": _git_head(), "step": "step0", "frame": FRAME,
            "sets": str(sets_path), "sets_sha256": hashlib.sha256(sets_path.read_bytes()).hexdigest()[:16],
            "prompt_token_sha256": {k: hashlib.sha256(json.dumps(ids[k]).encode()).hexdigest()[:16]
                                    for k in keys},
            "constants": {"grid": [GRID_LO, GRID_HI, GRID_N], "n_subsamples": N_SUBSAMPLES,
                          "n_draws": N_DRAWS, "arms": ARMS, "seed_base": SEED_BASE}}
    args.out.mkdir(parents=True, exist_ok=True)
    pool = make_pool(args.workers)
    t0 = time.monotonic()
    pending: List[Tuple[str, List[str], list]] = []  # (model, prompts, async results)

    def collect(item) -> None:
        mid, todo, res = item
        got: Dict[str, Dict[int, Tuple[Dict, Optional[np.ndarray]]]] = {k: {} for k in todo}
        for r in res:
            m, k, L, rec, lab = r.get() if pool is not None else r
            got[k][L] = (rec, lab)
        for k in todo:
            _write(args.out, mid, k, got[k], meta)
        errs = sum("error" in v[0] for g in got.values() for v in g.values())
        print(f"done {mid} {len(todo)} prompts, {errs} refused trees, {time.monotonic() - t0:.0f} s",
              flush=True)

    for mid in mids:
        todo = [k for k in keys if not _paths(args.out, mid, k)[0].exists()]
        if not todo:
            print(f"already {mid}", flush=True)
            continue
        model = load(mid, "step0")
        check_config(model)
        jobs = []
        for k in todo:
            H, _ = forward(model, ids[k])
            kp = np.asarray(kept[k])
            jobs += [(mid, k, L, H[L][kp]) for L in LAYERS]
        del model
        res = [pool.apply_async(_job, (j,)) for j in jobs] if pool is not None else [_job(j) for j in jobs]
        pending.append((mid, todo, res))
        # At most two models in flight: the next forward pass runs while this one's clouds do.
        while len(pending) > 1:
            collect(pending.pop(0))
    while pending:
        collect(pending.pop(0))
    if pool is not None:
        pool.close()
        pool.join()
    return 0


# ---------------------------------------------------------------------------
# Read: (b) against a reference set, plateaus, the pass
# ---------------------------------------------------------------------------

def ref_values(rows: Sequence[Dict]) -> np.ndarray:
    """
    One reference cloud's value per grid point: its ``z_G``; ``-inf`` where its cut has
    no cluster of the arm's size (counts as below); NaN where ``z_G`` is undefined
    because its draws' SD is 0 (dropped).
    """
    out = np.empty(len(rows))
    for g, r in enumerate(rows):
        out[g] = -np.inf if r["stability"] is None else (np.nan if r["z"] is None else r["z"])
    return out


def rank_rows(rows: Sequence[Dict], refs: np.ndarray) -> List[Dict]:
    """
    ``rows`` with (b) re-read against ``refs`` (``(n_ref, GRID_N)``, `ref_values`):
    ``p = (1 + #{ref >= z}) / (N + 1)`` over the N references left at that point;
    ``informative`` true only where the cloud's own ``z_G`` is defined, its draws
    were informative and N ``>= MIN_REF`` (else ``p`` is 1). The Gaussian p is kept
    as ``p_gauss``.
    """
    out = []
    for g, r in enumerate(rows):
        col = refs[:, g]
        col = col[~np.isnan(col)]
        ok = r["z"] is not None and r["informative"] and col.size >= MIN_REF
        p = float((1 + np.sum(col >= r["z"])) / (col.size + 1)) if ok else 1.0
        out.append({**r, "p_gauss": r["p"], "p": p, "n_ref": int(col.size),
                    "informative_draws": r["informative"], "informative": bool(ok)})
    return out


def load_run(out: Path) -> Tuple[Dict, Dict]:
    """
    Every (model, prompt) record and its labels: ``recs[mid][prompt][L]`` and
    ``labs[mid][prompt][L]`` (``(GRID_N, n)``). Refuses on a missing pair, a refused
    tree, a mixed git, or token sets that differ between records.
    """
    recs: Dict = {}
    labs: Dict = {}
    missing, errors, metas = [], [], set()
    for mid in model_ids("all"):
        for k in V1_PASSAGES:
            fj, fl = _paths(out, mid, k)
            if not fj.exists():
                missing.append(f"{mid}/{k}")
                continue
            d = json.loads(fj.read_text())
            metas.add(json.dumps({x: d["meta"][x] for x in ("git", "sets_sha256")}, sort_keys=True))
            z = np.load(fl)
            for r in d["layers"]:
                if "error" in r:
                    errors.append(f"{mid}/{k}/L{r['layer']}: {r['error']}")
            recs.setdefault(mid, {})[k] = {r["layer"]: r for r in d["layers"]}
            labs.setdefault(mid, {})[k] = {int(n[1:]): z[n] for n in z.files}
    if missing:
        raise SystemExit(f"refusing: {len(missing)} of {50 * len(V1_PASSAGES)} records missing "
                         f"(first {missing[:3]})")
    if errors:
        raise SystemExit(f"refusing: {len(errors)} clouds with a refused tree (first {errors[:3]})")
    if len(metas) != 1:
        raise SystemExit(f"refusing: records from more than one (git, token sets): {sorted(metas)}")
    return recs, labs


def plateau_table(recs: Dict, labs: Dict, arm: str, min_runs: Sequence[int]) -> Dict:
    """
    Per (model, prompt, layer), the arm's plateaus under each reference arm and
    each ``min_run``: ``{"real_init" | "reinit": {min_run: {(mid, prompt, L): [plateaus]}}}``;
    and without (b) at ``MIN_RUN``. Real inits are ranked among the 40 re-inits,
    each re-init among the other 39.
    """
    inits, reinits = model_ids("init"), model_ids("reinit")
    size = ARMS[arm]
    out = {"real_init": {m: {} for m in min_runs}, "reinit": {m: {} for m in min_runs},
           "without_b": {}, "n_ref_min": {}}
    for k in V1_PASSAGES:
        for L in LAYERS:
            ref = {m: ref_values(recs[m][k][L]["rows"][arm]) for m in reinits}
            for mid in inits + reinits:
                refs = np.stack([ref[m] for m in reinits if m != mid])
                rows = rank_rows(recs[mid][k][L]["rows"][arm], refs)
                lab = list(labs[mid][k][L])
                kind = "real_init" if mid in inits else "reinit"
                for m in min_runs:
                    out[kind][m][(mid, k, L)] = robust_plateaus(rows, lab, min_size=size, min_run=m)
                out["without_b"][(mid, k, L)] = robust_plateaus(rows, lab, use_p=False, min_size=size)
                out["n_ref_min"][(mid, k, L)] = min(r["n_ref"] for r in rows)
    return out


def share(table: Dict[Tuple, list], mids: Sequence[str], bnd: str) -> Tuple[int, int]:
    """(clouds with >= 1 plateau, clouds) over ``mids`` x prompts x the band's layers."""
    keys = [key for key in table if key[0] in mids and band(key[2]) == bnd]
    return sum(bool(table[key]) for key in keys), len(keys)


def subset_quantile(table: Dict[Tuple, list], reinits: Sequence[str], bnd: str, seed: int = 0,
                    n_subsets: int = N_SUBSETS, size: int = SUBSET_SIZE, q: float = ROUTE_QUANTILE) -> float:
    """The ``q`` quantile of the plateau share over ``n_subsets`` random ``size``-of-re-inits subsets."""
    per = {m: share(table, [m], bnd) for m in reinits}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_subsets)):
        pick = rng.choice(len(reinits), size=size, replace=False)
        hit = sum(per[reinits[i]][0] for i in pick)
        tot = sum(per[reinits[i]][1] for i in pick)
        vals.append(hit / tot)
    return float(np.quantile(vals, q))


def verdicts(tab: Dict) -> List[Dict]:
    """The pass per band (module docstring): rate on the re-init arm, raising MIN_RUN; route on the real inits."""
    inits, reinits = model_ids("init"), model_ids("reinit")
    out = []
    for bnd in BANDS:
        rates = {}
        chosen = None
        for m in range(MIN_RUN, MAX_MIN_RUN + 1):
            h, n = share(tab["reinit"][m], reinits, bnd)
            rates[m] = {"hits": h, "n": n, "share": h / n}
            if h / n <= MAX_RATE:
                chosen = m
                break
        row = {"band": bnd, "reinit_rate_by_min_run": rates, "min_run": chosen,
               "rate_pass": chosen is not None}
        m = chosen if chosen is not None else MAX_MIN_RUN
        h, n = share(tab["real_init"][m], inits, bnd)
        q = subset_quantile(tab["reinit"][m], reinits, bnd)
        row.update({"real_init": {"min_run": m, "hits": h, "n": n, "share": h / n},
                    "real_init_at_min_run_3": dict(zip(("hits", "n"), share(tab["real_init"][MIN_RUN], inits, bnd))),
                    "route_q95": q, "route_pass": h / n <= q})
        row["pass"] = bool(row["rate_pass"] and row["route_pass"])
        out.append(row)
    return out


def _plist(tab: Dict[Tuple, list]) -> List[Dict]:
    return [{"model": mid, "prompt": k, "layer": L, "r_lo": p["r_lo"], "r_hi": p["r_hi"],
             "k_lo": p["k_lo"], "k_hi": p["k_hi"], "points": p["end"] - p["start"] + 1}
            for (mid, k, L), pls in sorted(tab.items()) for p in pls]


def beside(tab: Dict, min_run: int = MIN_RUN) -> Dict:
    """Shares without (b) and per prompt, both reference arms, at ``min_run``."""
    inits, reinits = model_ids("init"), model_ids("reinit")
    out = {}
    for bnd in BANDS:
        out[bnd] = {
            "without_b": {kind: dict(zip(("hits", "n"), share(tab["without_b"], mids, bnd)))
                          for kind, mids in (("real_init", inits), ("reinit", reinits))},
            "per_prompt": {k: {kind: sum(bool(v) for key, v in tab[kind][min_run].items()
                                         if key[1] == k and band(key[2]) == bnd)
                               for kind in ("real_init", "reinit")} for k in V1_PASSAGES}}
    return out


def read_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="scale_real read")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    recs, labs = load_run(args.out)
    min_runs = tuple(range(MIN_RUN, MAX_MIN_RUN + 1))
    tabs = {a: plateau_table(recs, labs, a, min_runs) for a in ARMS}
    main = tabs[GATING_ARM]
    rows = verdicts(main)
    raised = sorted({r["min_run"] for r in rows if r["min_run"] not in (None, MIN_RUN)})
    meta = json.loads(_paths(args.out, model_ids("all")[0], V1_PASSAGES[0])[0].read_text())["meta"]
    res = {"git": _git_head(), "records": meta, "alpha": ALPHA, "min_ref": MIN_REF, "max_rate": MAX_RATE,
           "n_subsets": N_SUBSETS, "route_quantile": ROUTE_QUANTILE,
           "bands": rows, "pass_all": all(r["pass"] for r in rows),
           "synthetic_reread_owed_at_min_run": raised,
           "n_ref_min": int(min(main["n_ref_min"].values())),
           "beside": {a: beside(t) for a, t in tabs.items()},
           "plateaus": {a: {kind: _plist(t[kind][MIN_RUN]) for kind in ("real_init", "reinit")}
                        for a, t in tabs.items()}}
    (args.out / "specificity.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"specificity ({GATING_ARM} arm, centred): share of clouds with >= 1 plateau; "
          f"rate pass <= {MAX_RATE:.0%} on the re-inits, route pass <= q{ROUTE_QUANTILE:.2f} of "
          f"{N_SUBSETS} 10-of-40 subsets")
    for r in rows:
        rr = " ".join(f"MIN_RUN {m}: {v['hits']}/{v['n']} = {v['share']:.3f}"
                      for m, v in r["reinit_rate_by_min_run"].items())
        ri = r["real_init"]
        print(f"  {r['band']:7s} re-inits {rr} | real inits (MIN_RUN {ri['min_run']}) {ri['hits']}/{ri['n']} "
              f"= {ri['share']:.3f} vs q95 {r['route_q95']:.3f} | {'PASS' if r['pass'] else 'REFUSE'}")
    for a, b in res["beside"].items():
        for bnd, v in b.items():
            wb = v["without_b"]
            print(f"  beside {a:5s} {bnd:7s} without (b): re-inits {wb['reinit']['hits']}/{wb['reinit']['n']}, "
                  f"real inits {wb['real_init']['hits']}/{wb['real_init']['n']}")
    if raised:
        print(f"owed: the synthetic's sensitivity re-read at MIN_RUN {raised} on seeds 13-22 (post hoc, beside)")
    return 0 if res["pass_all"] else 1


# ---------------------------------------------------------------------------
# Resolution: how many grid points can hold a plateau (trees only)
# ---------------------------------------------------------------------------

def resolution(Y: np.ndarray) -> Dict:
    """
    Trees only, centred: the grid points whose cut has ``>= MIN_K`` main-arm clusters
    (a plateau needs ``MIN_RUN`` of them in a row), and the ``r`` range where the cut
    is neither all singletons nor one cluster.
    """
    from .gaussian_null import frame_vectors, span_coordinates
    from .merge_tree import labels_at_delta
    from .methods import LayerData
    from .scale_spectrum import MIN_K, _median_distance, _tree, relative_grid, substantial_count
    Z, _ = frame_vectors(Y, FRAME)
    d = LayerData.from_normed(span_coordinates(Z))
    t, med, grid = _tree(d), _median_distance(d), relative_grid()
    labs = [labels_at_delta(t, d.n, r * med) for r in grid]
    multi = [g for g, lab in enumerate(labs) if substantial_count(lab, ARMS[GATING_ARM]) >= MIN_K]
    mid = [g for g, lab in enumerate(labs) if 1 < np.unique(lab).size < d.n]
    return {"n": int(d.n), "median": med, "n_points_k2": len(multi),
            "r_lo": float(grid[mid[0]]) if mid else None, "r_hi": float(grid[mid[-1]]) if mid else None}


def resolution_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="scale_real resolution")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["reinit:0@step0", "init:0@step0", "init:0@step143000"],
                    help="model@step pairs")
    args = ap.parse_args(argv)
    sets_path = Path(os.environ["METS_DATA"]) / DEFAULT_SETS
    kept = load_sets(sets_path)
    from .move_text import forward
    ids = prompt_ids(_tok())
    res = {"git": _git_head(), "sets_sha256": hashlib.sha256(sets_path.read_bytes()).hexdigest()[:16],
           "models": {}}
    for ms in args.models:
        mid, step = ms.split("@")
        model = load(mid, step)
        check_config(model)
        rows = []
        for k in V1_PASSAGES:
            H, _ = forward(model, ids[k])
            rows += [{"prompt": k, "layer": L, **resolution(H[L][np.asarray(kept[k])])} for L in LAYERS]
        del model
        res["models"][ms] = rows
        for bnd in BANDS:
            v = [r for r in rows if band(r["layer"]) == bnd]
            pts = [r["n_points_k2"] for r in v]
            print(f"{ms:20s} {bnd:7s} grid points with >= 2 main clusters: median {np.median(pts):.0f} "
                  f"max {max(pts)}; >= MIN_RUN in {sum(p >= MIN_RUN for p in pts)} of {len(v)} clouds; "
                  f"non-trivial r {np.median([r['r_lo'] for r in v]):.3f}-{np.median([r['r_hi'] for r in v]):.3f}; "
                  f"median distance {np.median([r['median'] for r in v]):.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "resolution.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"run": run_cmd, "read": read_cmd, "resolution": resolution_cmd}
    if not argv or argv[0] not in cmds:
        print(f"usage: scale_real {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
