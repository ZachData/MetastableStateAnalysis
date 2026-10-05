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

**(b) on real input** (option 4, Blocked 19): the **conjunction** of the
cloud's own Gaussian p (``p_gauss``, against its 50 draws) and its ``z_G``
ranked (higher tail) among a reference set's at the same (prompt, layer, ``r``,
arm) (`rank_rows`; ``p = max`` of the two):

- **real-init arm**: each real init among the 40 re-inits;
- **re-init arm**: each re-init among the other 39.

A reference whose cut has no cluster of the arm's size counts as below the
cloud; one whose ``z_G`` is undefined (draws' SD 0) takes its sign (`ref_values`);
a point is admissible only with its own ``z_G`` defined, (iv) informative and N
``>= MIN_REF``. The plateau is `scale_spectrum.robust_plateaus` with that p; the
main arm gates.

**Gate** (`gate`; `design-1d.md` "Step 1 under option 4"): one matched-covariance
Gaussian of each of ``GATE_N`` trained clouds (`gate_cells`), read through the
conjunction against the re-inits of a finished `run`; passes with ``<=
MAX_GAUSSIAN_PLATEAUS`` main-arm plateaus. A fail refuses the trained reading.

**Beside, not the gate** (`read`, `verdicts`), per band L1–8 / 9–16 / 17–24:

- **rate**: the re-init arm's share of clouds with a main-arm plateau against
  ``MAX_RATE``, ``MIN_RUN`` raised one step at a time up to ``MAX_MIN_RUN``;
- **route**: the real-init arm's share, at that ``MIN_RUN``, against the 95th
  percentile of the share over ``N_SUBSETS`` random 10-of-40 subsets of the
  re-inits' own readings.

Also beside: the same without (b), the size-2 arm, plateaus per prompt with
their ``r`` ranges.

**The trained reading** (`run --step step143000`, then `trained`; Blocked 20,
option (a)): the 10 inits at ``step143000`` read through the conjunction
against the step-0 batch's re-inits, each cloud with its **window** (`window`:
the points where (b) could pass whatever the cloud's value); a missing plateau
says nothing outside it. Per plateau, its partition (`partition`); beside,
replication across the inits (`replication`).

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

from .arch_null import (BANDS, TRAINED_STEP, _git_head, _mfile, _tok, check_config,
                        comparison_models, label, load, make_pool, model_ids, prompt_ids)
from .gaussian_null import frame_vectors, gaussian_draw, span_coordinates
from .move_text import LAYERS, V1_PASSAGES, band
from .scale_spectrum import (ALPHA, ARMS, GATING_ARM, GRID_HI, GRID_LO, GRID_N, MAX_GAUSSIAN_PLATEAUS,
                             MIN_RUN, N_DRAWS, N_SUBSAMPLES, robust_plateaus, spectrum)

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
#: The gate (option 4): Gaussians of this many trained clouds, per band L1-8 / 9-16 / 17-24.
GATE_PER_BAND = (17, 17, 16)
GATE_N = sum(GATE_PER_BAND)
#: Seed of the gate's cell draw; cell i's Gaussian is drawn from [GATE_SEED, i], read with seed GATE_SEED + 1 + i.
GATE_SEED = 20_000


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
    ap.add_argument("--step", choices=("step0", TRAINED_STEP), default="step0",
                    help=f"step0: all 50 models (the references); {TRAINED_STEP}: the 10 inits (the trained reading)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args(argv)
    allowed = model_ids("all" if args.step == "step0" else "init")
    if args.only and set(args.only) - set(allowed):
        raise SystemExit(f"refusing: {sorted(set(args.only) - set(allowed))} do not exist at {args.step}")
    sets_path = args.sets or Path(os.environ["METS_DATA"]) / DEFAULT_SETS
    kept = load_sets(sets_path)
    import torch
    torch.set_num_threads(args.torch_threads)
    from .move_text import forward
    tok = _tok()
    ids = prompt_ids(tok)
    keys = args.keys or list(V1_PASSAGES)
    mids = args.only or allowed
    meta = {"git": _git_head(), "step": args.step, "frame": FRAME,
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
        model = load(mid, args.step)
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
    no cluster of the arm's size (counts as below). Where ``z_G`` is undefined because
    its draws' SD is 0, the sign of (its stability − the draws' common value): ``+inf``
    above (typically every draw empty, scored 0, while the reference has a cluster: the
    reference furthest above its covariance), ``-inf`` below, NaN (dropped) only when
    equal (e.g. every cut one cluster at stability 1). *Changed after `/challenge-pr`
    on #139, finding 1:* the design row dropped every SD-0 reference, which drops the
    strongest references and makes the check easier.
    """
    out = np.empty(len(rows))
    for g, r in enumerate(rows):
        if r["stability"] is None:
            out[g] = -np.inf
        elif r["z"] is not None:
            out[g] = r["z"]
        else:
            d = r["stability"] - r["null_stab_mean"]
            out[g] = np.nan if abs(d) <= 1e-12 else np.inf if d > 0 else -np.inf
    return out


def rank_rows(rows: Sequence[Dict], refs: np.ndarray) -> List[Dict]:
    """
    ``rows`` with (b) re-read against ``refs`` (``(n_ref, GRID_N)``, `ref_values`):
    ``p_rank = (1 + #{ref >= z}) / (N + 1)`` over the N references left at that point,
    and (b) the conjunction ``p = max(p_gauss, p_rank)`` (option 4, Blocked 19; the
    Gaussian p is the row's own ``p``, kept as ``p_gauss``). ``informative`` true only
    where the cloud's own ``z_G`` is defined, its draws were informative and N
    ``>= MIN_REF`` (else ``p`` and ``p_rank`` are 1).
    """
    out = []
    for g, r in enumerate(rows):
        col = refs[:, g]
        col = col[~np.isnan(col)]
        ok = r["z"] is not None and r["informative"] and col.size >= MIN_REF
        p_rank = float((1 + np.sum(col >= r["z"])) / (col.size + 1)) if ok else 1.0
        out.append({**r, "p_gauss": r["p"], "p_rank": p_rank, "p": max(r["p"], p_rank) if ok else 1.0,
                    "n_ref": int(col.size), "informative_draws": r["informative"], "informative": bool(ok)})
    return out


def load_run(out: Path, mids: Optional[Sequence[str]] = None) -> Tuple[Dict, Dict]:
    """
    Every (model, prompt) record and its labels, over ``mids`` (default all 50):
    ``recs[mid][prompt][L]`` and ``labs[mid][prompt][L]`` (``(GRID_N, n)``). Refuses on
    a missing pair, a refused tree, a mixed git, or token sets that differ between records.
    """
    mids = list(mids or model_ids("all"))
    recs: Dict = {}
    labs: Dict = {}
    missing, errors, metas = [], [], set()
    for mid in mids:
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
        raise SystemExit(f"refusing: {len(missing)} of {len(mids) * len(V1_PASSAGES)} records missing "
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
           "n_subsets": N_SUBSETS, "route_quantile": ROUTE_QUANTILE, "b": "conjunction", "gated": False,
           "bands": rows, "beside_all_within": all(r["pass"] for r in rows),
           "synthetic_reread_owed_at_min_run": raised,
           "n_ref_min": int(min(main["n_ref_min"].values())),
           "beside": {a: beside(t) for a, t in tabs.items()},
           "plateaus": {a: {kind: _plist(t[kind][MIN_RUN]) for kind in ("real_init", "reinit")}
                        for a, t in tabs.items()}}
    (args.out / "specificity.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"step-0 specificity, beside, not gated (option 4; {GATING_ARM} arm, centred, (b) the conjunction): "
          f"share of clouds with >= 1 plateau; rate <= {MAX_RATE:.0%} on the re-inits, route <= "
          f"q{ROUTE_QUANTILE:.2f} of {N_SUBSETS} 10-of-40 subsets")
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
        print(f"beside: MIN_RUN would be raised to {raised} in some band (not applied under option 4)")
    return 0


# ---------------------------------------------------------------------------
# Gate (option 4): Gaussians of trained clouds through the conjunction
# ---------------------------------------------------------------------------

def gate_cells(seed: int = GATE_SEED) -> List[Tuple[str, str, int]]:
    """``GATE_N`` (model, prompt, layer) cells of the trained reading, ``GATE_PER_BAND`` per band, without replacement."""
    rng = np.random.default_rng(seed)
    cells: List[Tuple[str, str, int]] = []
    for bnd, n in zip(BANDS, GATE_PER_BAND):
        pool = [(m, k, L) for m in model_ids("init") for k in V1_PASSAGES for L in LAYERS if band(L) == bnd]
        cells += [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    return cells


def gate_cloud(Y: np.ndarray, i: int) -> Tuple[Dict, Optional[np.ndarray]]:
    """Cell ``i``'s matched-covariance Gaussian (centred span coordinates, as `read_gaussian`), read by `read_cloud`."""
    Z, _ = frame_vectors(Y, FRAME)
    G = gaussian_draw(span_coordinates(Z), np.random.default_rng([GATE_SEED, i]))
    return read_cloud(G, GATE_SEED + 1 + i)


def _gate_job(args: Tuple[int, np.ndarray]) -> Tuple[int, Dict, Optional[np.ndarray]]:
    i, Y = args
    return (i, *gate_cloud(Y, i))


def gate_plateaus(rec: Dict, lab: np.ndarray, refs_by_arm: Dict[str, np.ndarray]) -> Dict:
    """Per arm: plateaus under the conjunction (the gate), and beside each term alone and without (b)."""
    out = {}
    for a, size in ARMS.items():
        rows = rank_rows(rec["rows"][a], refs_by_arm[a])
        labels = list(lab)
        out[a] = {"conjunction": robust_plateaus(rows, labels, min_size=size),
                  "gauss_only": rec["beside"][a]["plateaus_gauss"],
                  "rank_only": robust_plateaus(rows, labels, p_key="p_rank", min_size=size),
                  "without_b": rec["beside"][a]["plateaus_without_b"],
                  "n_ref_min": min(r["n_ref"] for r in rows)}
    return out


def gate_verdict(cells: Sequence[Tuple[str, str, int]], plateaus: Sequence[Dict]) -> Dict:
    """Counts of Gaussians with >= 1 plateau, per reading and arm, pooled and per band; the gate on the main arm's conjunction."""
    counts = {a: {kind: {"all": 0, **{b: 0 for b in BANDS}}
                  for kind in ("conjunction", "gauss_only", "rank_only", "without_b")} for a in ARMS}
    for (_, _, L), pl in zip(cells, plateaus):
        for a in ARMS:
            for kind, c in counts[a].items():
                if pl[a][kind]:
                    c["all"] += 1
                    c[band(L)] += 1
    hits = counts[GATING_ARM]["conjunction"]["all"]
    return {"n": len(cells), "max": MAX_GAUSSIAN_PLATEAUS, "hits": hits,
            "pass": hits <= MAX_GAUSSIAN_PLATEAUS, "counts": counts}


def gate_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="scale_real gate")
    ap.add_argument("--out", type=Path, required=True, help="a finished `run` directory (the re-inits' z_G)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args(argv)
    recs, _ = load_run(args.out)
    meta = json.loads(_paths(args.out, model_ids("all")[0], V1_PASSAGES[0])[0].read_text())["meta"]
    sets_path = Path(meta["sets"])
    if hashlib.sha256(sets_path.read_bytes()).hexdigest()[:16] != meta["sets_sha256"]:
        raise SystemExit(f"refusing: {sets_path} is not the token sets the run read")
    kept = load_sets(sets_path)
    import torch
    torch.set_num_threads(args.torch_threads)
    from .move_text import forward
    ids = prompt_ids(_tok())
    cells = gate_cells()
    pool = make_pool(args.workers)
    t0 = time.monotonic()
    res = []
    for mid in sorted({c[0] for c in cells}, key=model_ids("init").index):
        model = load(mid, TRAINED_STEP)
        check_config(model)
        for k in sorted({c[1] for c in cells if c[0] == mid}):
            H, _ = forward(model, ids[k])
            for i, c in enumerate(cells):
                if c[0] == mid and c[1] == k:
                    job = (i, H[c[2]][np.asarray(kept[k])])
                    res.append(pool.apply_async(_gate_job, (job,)) if pool is not None else _gate_job(job))
        del model
    got = {}
    for r in res:
        i, rec, lab = r.get() if pool is not None else r
        got[i] = (rec, lab)
    if pool is not None:
        pool.close()
        pool.join()
    errs = [i for i, (rec, _) in got.items() if "error" in rec]
    if errs:
        raise SystemExit(f"refusing: {len(errs)} gate clouds with a refused tree (cells {errs[:3]})")
    reinits = model_ids("reinit")
    plateaus = []
    for i, (mid, k, L) in enumerate(cells):
        refs = {a: np.stack([ref_values(recs[m][k][L]["rows"][a]) for m in reinits]) for a in ARMS}
        plateaus.append(gate_plateaus(*got[i], refs))
    v = gate_verdict(cells, plateaus)
    res_json = {"git": _git_head(), "records": meta, "trained_step": TRAINED_STEP, "gate_seed": GATE_SEED,
                "per_band": dict(zip(BANDS, GATE_PER_BAND)), **v,
                "cells": [{"i": i, "model": m, "prompt": k, "layer": L, "band": band(L), "n": got[i][0]["n"],
                           "median": got[i][0]["median"], "plateaus": pl}
                          for i, ((m, k, L), pl) in enumerate(zip(cells, plateaus))]}
    (args.out / "gate.json").write_text(json.dumps(res_json, indent=1) + "\n")
    np.savez_compressed(args.out / "gate_labels.npz", **{f"c{i}": got[i][1] for i in sorted(got)})
    (args.out / "gate_records.json").write_text(json.dumps([got[i][0] for i in sorted(got)]) + "\n")
    print(f"gate (option 4): {v['hits']} of {v['n']} Gaussians of trained clouds with a {GATING_ARM}-arm plateau "
          f"under the conjunction (pass <= {MAX_GAUSSIAN_PLATEAUS}): {'PASS' if v['pass'] else 'REFUSE'}; "
          f"{time.monotonic() - t0:.0f} s")
    for a, by in v["counts"].items():
        print(f"  {a:5s} " + " | ".join(f"{kind} {c['all']}/{v['n']} (" + ", ".join(f"{b} {c[b]}" for b in BANDS) + ")"
                                        for kind, c in by.items()))
    return 0 if v["pass"] else 1


# ---------------------------------------------------------------------------
# The trained reading (Blocked 20, option (a)): the reader as gated, with its window
# ---------------------------------------------------------------------------

def window(rows: Sequence[Dict], refs: np.ndarray) -> Dict:
    """
    The grid points where (b) could pass whatever the cloud's own value (`design-1d.md`
    "The trained reading", row "window"): `rank_rows` informative (own ``z_G`` defined,
    (iv), N ``>= MIN_REF``) and ``(1 + #{ref = +inf}) / (N + 1) <= ALPHA``. Returns the
    points, their ``r`` range, the longest run of consecutive points, and ``readable``
    (that run ``>= MIN_RUN``).
    """
    ranked = rank_rows(rows, refs)
    pts = []
    for g, r in enumerate(ranked):
        col = refs[:, g]
        col = col[~np.isnan(col)]
        if r["informative"] and (1 + np.sum(np.isposinf(col))) / (col.size + 1) <= ALPHA:
            pts.append(g)
    longest, run = 0, 0
    for g in range(len(rows)):
        run = run + 1 if g in pts else 0
        longest = max(longest, run)
    return {"points": pts, "r_lo": rows[pts[0]]["r"] if pts else None,
            "r_hi": rows[pts[-1]]["r"] if pts else None, "longest_run": longest,
            "readable": longest >= MIN_RUN}


def partition(lab: np.ndarray, size: int, kept: Sequence[int], tokens: Sequence[str]) -> List[Dict]:
    """The clusters of ``>= size`` tokens of one cut, as kept positions and decoded tokens, largest first."""
    ids, counts = np.unique(lab, return_counts=True)
    out = []
    for c in ids[counts >= size]:
        idx = np.flatnonzero(lab == c)
        out.append({"positions": [int(kept[i]) for i in idx], "tokens": [tokens[kept[i]] for i in idx]})
    return sorted(out, key=lambda d: (-len(d["positions"]), d["positions"][0]))


def pair_ari(lab_a: np.ndarray, lab_b: np.ndarray, size: int) -> Optional[float]:
    """ARI between two cuts over the tokens in clusters of ``>= size`` in both; None under 2 such tokens."""
    from sklearn.metrics import adjusted_rand_score

    def clustered(lab):
        _, inv, counts = np.unique(lab, return_inverse=True, return_counts=True)
        return counts[inv] >= size
    m = clustered(lab_a) & clustered(lab_b)
    return float(adjusted_rand_score(lab_a[m], lab_b[m])) if m.sum() >= 2 else None


def replication(cells: Sequence[Dict], labs: Dict, arm: str) -> Dict:
    """
    Per band: (prompt, layer) cells by how many of the inits hold a conjunction plateau on
    ``arm``, and the median over pairs that both do of `pair_ari` between their first
    (lowest-``r``) plateaus' first cuts.
    """
    by: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
    for c in cells:
        pl = c["plateaus"][arm]["conjunction"]
        if pl:
            by.setdefault((c["prompt"], c["layer"]), []).append((c["model"], pl[0]["start"]))
    out = {}
    for bnd in BANDS:
        hist: Dict[int, int] = {}
        aris: List[float] = []
        for (k, L), hits in by.items():
            if band(L) != bnd:
                continue
            hist[len(hits)] = hist.get(len(hits), 0) + 1
            for i in range(len(hits)):
                for j in range(i + 1, len(hits)):
                    (ma, ga), (mb, gb) = hits[i], hits[j]
                    a = pair_ari(labs[ma][k][L][ga], labs[mb][k][L][gb], ARMS[arm])
                    if a is not None:
                        aris.append(a)
        out[bnd] = {"cells_by_n_inits": dict(sorted(hist.items())), "n_pairs": len(aris),
                    "median_pair_ari": float(np.median(aris)) if aris else None}
    return out


def trained_summary(cells: Sequence[Dict]) -> Dict:
    """Per arm and band: clouds, readable clouds, clouds with >= 1 plateau per reading, median readable r range."""
    out: Dict = {}
    for a in ARMS:
        out[a] = {}
        for bnd in BANDS:
            cs = [c for c in cells if band(c["layer"]) == bnd]
            rd = [c for c in cs if c["window"][a]["readable"]]
            row = {"clouds": len(cs), "readable": len(rd),
                   "median_r_lo": float(np.median([c["window"][a]["r_lo"] for c in rd])) if rd else None,
                   "median_r_hi": float(np.median([c["window"][a]["r_hi"] for c in rd])) if rd else None}
            for kind in ("conjunction", "gauss_only", "rank_only", "without_b"):
                row[kind] = sum(bool(c["plateaus"][a][kind]) for c in cs)
            out[a][bnd] = row
    return out


def point_failures(rows: Sequence[Dict], refs: np.ndarray) -> List[Dict]:
    """
    Per grid point of one cloud and arm: whether it is in the `window`, whether the rank
    term is **free** there (every reference without a cluster of the arm's size, so
    ``p_rank`` = 1/(N + 1) whatever the cloud), and each admissibility condition alone.
    """
    win = set(window(rows, refs)["points"])
    out = []
    for g, r in enumerate(rank_rows(rows, refs)):
        col = refs[:, g]
        col = col[~np.isnan(col)]
        out.append({"g": g, "r": r["r"], "window": g in win, "rank_free": bool(col.size and np.all(np.isneginf(col))),
                    "k2": r["k_sub"] >= 2, "stable": r["stability"] is not None and r["stability"] >= 0.75,
                    "gauss": r["p_gauss"] <= ALPHA, "rank": r["informative"] and r["p_rank"] <= ALPHA,
                    "admissible": (r["k_sub"] >= 2 and r["stability"] is not None and r["stability"] >= 0.75
                                   and r["informative"] and r["p"] <= ALPHA)})
    return out


def diagnose_cmd(argv: Optional[Sequence[str]] = None) -> int:
    """Beside the trained reading: per arm, band and grid point, how many window clouds meet each condition."""
    ap = argparse.ArgumentParser(prog="scale_real diagnose")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    args = ap.parse_args(argv)
    inits, reinits = model_ids("init"), model_ids("reinit")
    recs, _ = load_run(args.out, inits)
    refs_recs, _ = load_run(args.ref)
    keys = ("window", "rank_free", "k2", "stable", "gauss", "rank", "admissible")
    tab = {a: {b: {} for b in BANDS} for a in ARMS}
    for k in V1_PASSAGES:
        for L in LAYERS:
            refs = {a: np.stack([ref_values(refs_recs[m][k][L]["rows"][a]) for m in reinits]) for a in ARMS}
            for mid in inits:
                for a in ARMS:
                    for pt in point_failures(recs[mid][k][L]["rows"][a], refs[a]):
                        if not pt["window"]:
                            continue
                        row = tab[a][band(L)].setdefault(round(pt["r"], 3), {x: 0 for x in keys})
                        for x in keys:
                            row[x] += bool(pt[x])
    (args.out / "diagnose.json").write_text(json.dumps({"git": _git_head(), "table": tab}, indent=1) + "\n")
    print("per window point (clouds in the window at that r): " + ", ".join(keys[1:]))
    for a in ARMS:
        for b in BANDS:
            for r, row in sorted(tab[a][b].items()):
                print(f"  {a:5s} {b:7s} r {r:.3f}: window {row['window']:4d} | "
                      + " ".join(f"{x} {row[x]:4d}" for x in keys[1:]))
    return 0


def _meta(out: Path, mid: str) -> Dict:
    return json.loads(_paths(out, mid, V1_PASSAGES[0])[0].read_text())["meta"]


def trained_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="scale_real trained")
    ap.add_argument("--out", type=Path, required=True, help=f"a finished `run --step {TRAINED_STEP}` directory")
    ap.add_argument("--ref", type=Path, required=True, help="the finished step-0 batch (the re-inits' z_G)")
    args = ap.parse_args(argv)
    inits, reinits = model_ids("init"), model_ids("reinit")
    recs, labs = load_run(args.out, inits)
    refs_recs, _ = load_run(args.ref)
    meta, ref_meta = _meta(args.out, inits[0]), _meta(args.ref, inits[0])
    if meta["step"] != TRAINED_STEP or ref_meta["step"] != "step0":
        raise SystemExit(f"refusing: steps {meta['step']} / {ref_meta['step']}, want {TRAINED_STEP} / step0")
    if meta["sets_sha256"] != ref_meta["sets_sha256"]:
        raise SystemExit("refusing: the trained clouds and the references read different token sets")
    sets_path = Path(meta["sets"])
    if hashlib.sha256(sets_path.read_bytes()).hexdigest()[:16] != meta["sets_sha256"]:
        raise SystemExit(f"refusing: {sets_path} is not the token sets the run read")
    kept = load_sets(sets_path)
    tok = _tok()
    toks = {k: tok.convert_ids_to_tokens(v) for k, v in prompt_ids(tok).items()}
    cells = []
    for mid in inits:
        for k in V1_PASSAGES:
            for L in LAYERS:
                rec, lab = recs[mid][k][L], labs[mid][k][L]
                refs = {a: np.stack([ref_values(refs_recs[m][k][L]["rows"][a]) for m in reinits]) for a in ARMS}
                pl = gate_plateaus(rec, lab, refs)
                for a, size in ARMS.items():
                    for p in pl[a]["conjunction"]:
                        p["partition"] = partition(lab[p["start"]], size, kept[k], toks[k])
                cells.append({"model": mid, "prompt": k, "layer": L, "band": band(L), "n": rec["n"],
                              "median": rec["median"], "plateaus": pl,
                              "window": {a: window(rec["rows"][a], refs[a]) for a in ARMS}})
    summ = trained_summary(cells)
    rep = {a: replication(cells, labs, a) for a in ARMS}
    res = {"git": _git_head(), "records": meta, "ref_records": ref_meta, "ref": str(args.ref), "alpha": ALPHA,
           "min_run": MIN_RUN, "min_ref": MIN_REF, "b": "conjunction", "reading_arm": GATING_ARM,
           "summary": summ, "replication": rep, "cells": cells}
    (args.out / "trained.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"trained reading ({TRAINED_STEP}, centred, (b) the conjunction against {len(reinits)} step-0 re-inits; "
          f"the {GATING_ARM} arm is the reading): clouds with >= 1 plateau")
    for a in ARMS:
        for bnd, r in summ[a].items():
            w = (f"readable {r['readable']}/{r['clouds']}, median window r {r['median_r_lo']:.2f}-{r['median_r_hi']:.2f}"
                 if r["readable"] else f"readable 0/{r['clouds']}")
            rp = rep[a][bnd]
            ari = "-" if rp["median_pair_ari"] is None else f"{rp['median_pair_ari']:.2f}"
            print(f"  {a:5s} {bnd:7s} {r['conjunction']}/{r['clouds']} ({w}) | beside: gauss only "
                  f"{r['gauss_only']}, rank only {r['rank_only']}, without (b) {r['without_b']} | cells by inits "
                  f"{rp['cells_by_n_inits']}, pair ARI {ari} ({rp['n_pairs']} pairs)")
    return 0


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
    cmds = {"run": run_cmd, "read": read_cmd, "gate": gate_cmd, "trained": trained_cmd, "diagnose": diagnose_cmd,
            "resolution": resolution_cmd}
    if not argv or argv[0] not in cmds:
        print(f"usage: scale_real {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
