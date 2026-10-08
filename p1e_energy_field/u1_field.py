"""
p1e_energy_field/u1_field.py — U1, the field at the tokens (`design-1e.md` "U1: the rule",
fixed before any output).

Per (step, passage, layer ℓ = 0–23), in β's frame (layer ℓ's LN1 on hidden state ℓ, unit rows),
with the states held fixed:

- **density**: ``e_i = log φ_β^{−i}(u_i) − mean_k log φ_β^{−k}(u_k)``, the full field over the
  targets with the self term out; its spread, its relation to position (log offset) and to the
  β → 0 term ``⟨u_i, ū⟩``; the causal ``e_i`` (log of the mean over every stored j < i) beside;
- **wells**: mean shift on ``φ_β`` from every target (GPU float32 until rows move < 1e-6, rows
  merged as they coincide, then float64 on the CPU to < 1e-10); modes within 1e-3 are one well;
- **what the wells are**: AMI against position (8 equal-count bins) and token class; the
  opening's well; on v1 the purity of R0's c3x groups (Blocked 27 (a)) against permutations;
- **null**: 4 matched-Gaussian clouds per cell (the targets' mean and covariance, put on the
  sphere), read the same way at β 3.5. Step 0 beside, by the same rule (report).

Not computed (fences): inner products between well centres, any design statistic, ``|ū|`` or the
mean pairwise inner product (P-S1); step lengths (P-γ). Tier 1: exploratory, unregistered.
    python -m p1e_energy_field.u1_field run --runs <p1e_long8 dir> --r0 <R0 labels> \
        --r8x <R8x dir> --out <dir> [--device cuda|cpu] [--first-only]
    python -m p1e_energy_field.u1_field report --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .extract_long8 import STEPS
from .u2_block import code_sha, export_ln1, load_ln1, unit_rows

BETAS = (1.6, 3.5, 5.6)
PRIMARY_BETA = 3.5
SWEEP = (0.5, 1.0, 2.5, 10.0, 20.0, 50.0, 100.0)
SWEEP_LAYERS = (4, 12, 20)
#: Hidden state 24 is refused: it is after ``final_layer_norm``.
LAYERS = tuple(range(24))
BANDS = {"L0": (0,), "L1-8": tuple(range(1, 9)), "L9-16": tuple(range(9, 17)),
         "L17-23": tuple(range(17, 24))}
N_DRAW = 4
SEED = 20261008
#: Placed by the rule, not calibrated.
MOVE32, MOVE64, MERGE_RUN, MERGE = 1e-6, 1e-10, 1e-6, 1e-3
MERGE_BESIDE = (1e-4, 1e-2)
MAX32, MAX64 = 2000, 5000
POS_BINS, OPEN_K, N_PERM_PURITY = 8, 64, 200
UNIT_TOL, CONVERGED_MIN, AGREE_MIN = 1e-4, 0.99, 0.999
CHECK_LAYERS = (4, 12, 20)
FIRST = ("143000", "wiki_paragraph_long")


# ---------------------------------------------------------------- pure: density

def _lse(Z: np.ndarray) -> np.ndarray:
    m = Z.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True)))[:, 0]


def density(S: np.ndarray, beta: float) -> np.ndarray:
    """Leave-one-out ``log φ_β`` at each row (``S`` the Gram of unit rows), centred (M1)."""
    Z = beta * S.astype(np.float64)
    np.fill_diagonal(Z, -np.inf)
    lp = _lse(Z)
    return lp - lp.mean()


def causal_density(U_all: np.ndarray, pos: np.ndarray, beta: float) -> np.ndarray:
    """``log mean_{j < i} exp(β⟨u_i, u_j⟩)`` over every stored position j < i, centred over ``pos``."""
    Z = beta * (U_all[pos] @ U_all.T)
    mask = np.arange(U_all.shape[0])[None, :] < pos[:, None]
    Z = np.where(mask, Z, -np.inf)
    lp = _lse(Z) - np.log(pos.astype(np.float64))
    return lp - lp.mean()


def r2(y: np.ndarray, x: np.ndarray) -> tuple:
    """(R², residual sd) of ``y`` on ``[1, x]``."""
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    tot = np.sum((y - y.mean()) ** 2)
    return (float(1 - np.sum(res ** 2) / tot) if tot > 0 else float("nan")), float(res.std())


def gaussian_draw(U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """n rows with the targets' mean and covariance, ``ū + G (U − ū) / √(n − 1)``, on the sphere."""
    n = U.shape[0]
    mu = U.mean(axis=0)
    Y = mu + rng.standard_normal((n, n)) @ (U - mu) / np.sqrt(n - 1)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


# ---------------------------------------------------------------- pure: wells

def dedup(Y: np.ndarray, tol: float) -> tuple:
    """Greedy merge of rows within ``tol`` cosine distance: ``(label per row, representative rows)``."""
    close = (1.0 - Y @ Y.T) < tol
    lab = -np.ones(len(Y), dtype=int)
    reps = []
    for i in range(len(Y)):
        if lab[i] < 0:
            lab[(lab < 0) & close[i]] = len(reps)
            reps.append(i)
    return lab, np.asarray(reps, dtype=int)


def _step(Y, X, beta):
    Yn = np.exp(beta * (Y @ X.T - 1.0)) @ X
    return Yn / np.linalg.norm(Yn, axis=1, keepdims=True)


def _phase32_numpy(X: np.ndarray, beta: float) -> tuple:
    X32 = X.astype(np.float32)
    Y, owner = X32.copy(), np.arange(len(X))
    for t in range(MAX32):
        Yn = _step(Y, X32, np.float32(beta))
        mv = 1.0 - np.sum(Yn * Y, axis=1)
        Y = Yn
        if mv.max() < MOVE32:
            break
        if t % 5 == 4:
            lab, reps = dedup(Y, MERGE_RUN)
            owner, Y = lab[owner], Y[reps]
    lab, reps = dedup(Y, MERGE_RUN)
    return lab[owner], Y[reps]


def _phase32_torch(X: np.ndarray, beta: float) -> tuple:
    import torch
    dev = torch.device("cuda")
    Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
    Y, owner = Xt.clone(), np.arange(len(X))
    for t in range(MAX32):
        Yn = torch.exp(beta * (Y @ Xt.T - 1.0)) @ Xt
        Yn = Yn / Yn.norm(dim=1, keepdim=True)
        mv = float((1.0 - (Yn * Y).sum(dim=1)).max())
        Y = Yn
        if mv < MOVE32:
            break
        if t % 5 == 4:
            lab, reps = dedup_t(Y, MERGE_RUN)
            owner, Y = lab[owner], Y[torch.as_tensor(reps, device=dev)]
    lab, reps = dedup_t(Y, MERGE_RUN)
    return lab[owner], Y[torch.as_tensor(reps, device=dev)].double().cpu().numpy()


def dedup_t(Y, tol: float) -> tuple:
    close = ((1.0 - Y @ Y.T) < tol).cpu().numpy()
    lab = -np.ones(close.shape[0], dtype=int)
    reps = []
    for i in range(close.shape[0]):
        if lab[i] < 0:
            lab[(lab < 0) & close[i]] = len(reps)
            reps.append(i)
    return lab, np.asarray(reps, dtype=int)


def mean_shift(X: np.ndarray, beta: float, device: str = "cpu", exact: bool = False) -> Dict:
    """
    Basins of ``φ_β`` over unit rows ``X`` (float64), from every row. ``exact``: float64 from the
    start (the first check's reference). Returns each row's well (merged at ``MERGE``), the
    counts at ``MERGE_BESIDE``, and how many rows' trajectories did not converge.
    """
    X = np.asarray(X, dtype=np.float64)
    if exact:
        owner, Y = np.arange(len(X)), X.copy()
    elif device == "cuda":
        owner, Y = _phase32_torch(X, beta)
    else:
        owner, Y = _phase32_numpy(X, beta)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    conv = np.zeros(len(Y), dtype=bool)
    iters = 0
    for iters in range(1, MAX64 + 1):
        live = ~conv
        Yn = _step(Y[live], X, beta)
        mv = 1.0 - np.sum(Yn * Y[live], axis=1)
        Y[live] = Yn
        idx = np.flatnonzero(live)
        conv[idx[mv < MOVE64]] = True
        if conv.all():
            break
        if exact and iters % 5 == 0:               # merge coincident rows in the reference too
            lab, reps = dedup(Y, MERGE_RUN)
            owner, Y, conv = lab[owner], Y[reps], conv[reps]
    lab, _ = dedup(Y, MERGE)
    wells = lab[owner]
    beside = {f"k_{t:g}": int(len(dedup(Y, t)[1])) for t in MERGE_BESIDE}
    return {"wells": wells, "unconverged": int((~conv[owner]).sum()), "iters": iters, **beside}


def well_stats(w: np.ndarray) -> Dict:
    sizes = np.bincount(w)
    sizes = sizes[sizes > 0]
    p = sizes / sizes.sum()
    return {"k": int(sizes.size), "k2": int((sizes >= 2).sum()),
            "k_eff": float(np.exp(-np.sum(p * np.log(p)))), "largest": float(p.max())}


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Share of rows on which two partitions agree, by best overlap each way (the smaller)."""
    from scipy.sparse import coo_matrix
    C = coo_matrix((np.ones(a.size), (a, b))).toarray()
    return float(min(C.max(axis=1).sum(), C.max(axis=0).sum()) / a.size)


def position_bins(pos: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(pos))
    return (r * POS_BINS // len(pos)).astype(int)


def purity(w: np.ndarray, groups: List[np.ndarray]) -> float:
    """Mean over groups of the share of members in the group's modal well."""
    return float(np.mean([np.bincount(w[g]).max() / g.size for g in groups]))


def describe(w: np.ndarray, pos: np.ndarray, cls: np.ndarray, groups: Optional[List[np.ndarray]],
             rng: np.random.Generator) -> Dict:
    from sklearn.metrics import adjusted_mutual_info_score as ami
    out = {"ami_pos": float(ami(position_bins(pos), w)), "ami_cls": float(ami(cls, w))}
    first = np.argsort(pos)[:OPEN_K]
    w0 = w[np.argmin(pos)]
    out["open_share"] = float(np.mean(w[first] == w0))
    out["open_well_share"] = float(np.mean(w == w0))
    if groups:
        obs = purity(w, groups)
        null = np.array([purity(rng.permutation(w), groups) for _ in range(N_PERM_PURITY)])
        out.update(n_c3x=len(groups), purity=obs, purity_null=float(null.mean()),
                   purity_p=float((1 + np.sum(null >= obs)) / (N_PERM_PURITY + 1)))
    return out


def decile_mix(e: np.ndarray, cls: np.ndarray) -> Dict:
    k = max(1, e.size // 10)
    o = np.argsort(e)
    mix = lambda idx: {c: int(n) for c, n in zip(*np.unique(cls[idx], return_counts=True))}  # noqa: E731
    return {"void": mix(o[:k]), "dense": mix(o[-k:]), "all": mix(np.arange(e.size))}


# ---------------------------------------------------------------- one layer

def read_layer(U_all: np.ndarray, pos: np.ndarray, cls: np.ndarray, groups, beta_set, rng_key,
               device: str, sweep: bool, draws: bool) -> tuple:
    """Records and saved arrays for one (run, layer, target set)."""
    U = U_all[pos]
    S = U @ U.T
    lpos = np.log(pos.astype(np.float64))
    ubar = U.mean(axis=0)
    rng = np.random.default_rng(rng_key)
    recs, saved = [], {}
    gauss = [gaussian_draw(U, rng) for _ in range(N_DRAW)] if draws else []
    gram_g = [Y @ Y.T for Y in gauss]
    for beta in beta_set:
        e = density(S, beta)
        from scipy.stats import spearmanr
        r2p, sdr = r2(e, lpos)
        ms = mean_shift(U, beta, device)
        w = ms.pop("wells")
        rec = {"beta": beta, "n": int(pos.size), "sd_e": float(e.std()),
               "rho_pos": float(spearmanr(e, lpos).statistic), "r2_pos": r2p, "sd_resid_pos": sdr,
               "r2_mean": r2(e, U @ ubar)[0], **ms, **well_stats(w)}
        st = well_stats(w)
        if st["k2"] >= 2:
            rec.update(describe(w, pos, cls, groups, rng))
        if draws:              # the same draws at every β; their wells at β 3.5 only (the rule)
            g_sd = [float(density(S_g, beta).std()) for S_g in gram_g]
            rec.update(sd_e_G=g_sd, Xe=float(rec["sd_e"] - np.mean(g_sd)))
        if beta == PRIMARY_BETA:
            rec["mix"] = decile_mix(e, cls)
            ec = causal_density(U_all, pos, beta)
            rec["rho_pos_causal"] = float(spearmanr(ec, lpos).statistic)
            saved["e"], saved["e_causal"] = e, ec
            if draws:
                gs = [well_stats(mean_shift(Y, beta, device)["wells"]) for Y in gauss]
                g_keff = [g["k_eff"] for g in gs]
                rec.update(k_G=[g["k"] for g in gs], k_eff_G=g_keff,
                           Xw=float(np.log(rec["k_eff"]) - np.mean(np.log(g_keff))))
        saved[f"wells_{beta:g}"] = w
        recs.append(rec)
    if sweep:
        for beta in SWEEP:
            ms = mean_shift(U, beta, device)
            recs.append({"beta": beta, "sweep": True, "n": int(pos.size),
                         "unconverged": ms["unconverged"], **well_stats(ms["wells"])})
    return recs, saved


def token_classes(tokens: Sequence[str]) -> np.ndarray:
    from tools.run.p10_token_composition import decode, token_class
    texts = [decode(t) for t in tokens]
    return np.asarray([token_class(t, texts[p - 1] if p else None, p) for p, t in enumerate(texts)])


def read_run(run_dir: Path, ln1: Dict, targets: Dict[str, np.ndarray], key: str, step: str,
             c3x: Optional[Dict[int, List[np.ndarray]]] = None, device: str = "cpu",
             layers: Sequence[int] = LAYERS) -> tuple:
    z = np.load(run_dir / "activations.npz")
    acts, norms = z["activations"], z["norms"]
    if acts.shape[0] != 25:
        raise ValueError(f"{run_dir}: {acts.shape[0]} hidden states, expected 25")
    dev = float(np.abs(np.linalg.norm(acts, axis=2) - 1).max())
    if dev > UNIT_TOL:
        raise ValueError(f"{run_dir}: stored rows off unit by {dev:.1e} > {UNIT_TOL}; refusing")
    tokens = json.loads((run_dir / "geometry.json").read_text())["tokens"]
    n = acts.shape[1]
    if len(tokens) != n:
        raise ValueError(f"{run_dir}: {len(tokens)} tokens, {n} stored positions")
    for name, t in targets.items():
        if t.size == 0 or t.min() < 1 or t.max() >= n:
            raise ValueError(f"{run_dir}: target set {name} does not index the stored positions")
    cls_all = token_classes(tokens)
    out, saved = [], {}
    for L in layers:
        if L not in LAYERS:
            raise ValueError(f"layer {L} refused (hidden state 24 is after final LN)")
        U_all = unit_rows(acts[L] * norms[L][:, None], ln1["w"][L], ln1["b"][L], ln1["eps"])
        for name, t in targets.items():
            primary = name in ("t12", "r0")
            groups = None
            if c3x is not None and L in c3x:
                idx = {int(p): i for i, p in enumerate(t)}
                groups = [np.asarray([idx[int(p)] for p in g]) for g in c3x[L]]
            rng_key = [SEED, zlib.crc32(f"{key}|{step}|{L}|{name}".encode())]
            recs, sv = read_layer(U_all, t, cls_all[t], groups,
                                  BETAS if primary else (PRIMARY_BETA,), rng_key, device,
                                  sweep=primary and L in SWEEP_LAYERS, draws=True)
            out += [{"layer": L, "targets": name, **r} for r in recs]
            for k, v in sv.items():
                saved.setdefault(f"{name}/{k}", []).append(v)
    return out, {k: np.stack(v).astype(np.float32 if v[0].dtype.kind == "f" else np.int32)
                 for k, v in saved.items()}


# ---------------------------------------------------------------- inputs

def c3x_groups(r0: Path, r8x: Path, step: str, key: str, kept: List[int]) -> Dict[int, List[np.ndarray]]:
    """R0's groups that pass c3x at this (step, passage), as stored positions per layer."""
    lab = json.loads((r0 / f"step{step}.json").read_text())["prompts"][key]
    rows = json.loads((r8x / "rows" / f"step{step}" / f"{key}.json").read_text())["rows"]
    out: Dict[int, List[np.ndarray]] = {}
    kept_a = np.asarray(kept, dtype=int)
    for r in rows:
        if not r["c3x"] or r["layer"] not in LAYERS:
            continue
        col = np.asarray(lab["layers"][str(r["layer"])]["c2a"])
        members = kept_a[col == r["id"]]
        if members.size != r["size"]:
            raise SystemExit(f"refusing: R8x row {step}/{key}/L{r['layer']}#{r['id']} has size "
                             f"{r['size']}, R0's c2a holds {members.size}")
        out.setdefault(r["layer"], []).append(members)
    return out


def plan(runs: Path, r0: Optional[Path], r8x: Optional[Path]) -> tuple:
    from .long8_targets import target_positions
    from .long_prompts_1e import LONG8_HASH
    keys = sorted({d.name.split("_", 1)[1] for d in runs.glob("pythia-410m-step0_*")})
    if len(keys) != 8:
        raise SystemExit(f"refusing: {len(keys)} long passages in {runs}, expected 8")
    jobs, revs, meta = [], set(), {"long8_hash": LONG8_HASH, "targets": {}, "v1": {}}
    for key in keys:
        _, massive, t12, t123 = target_positions(runs, key)
        meta["targets"][key] = {"t12": int(t12.size), "t123": int(t123.size)}
        for s in STEPS:
            rd = runs / f"pythia-410m-step{s}_{key}"
            man = json.loads((rd / "manifest.json").read_text())
            if man.get("long8_hash") != LONG8_HASH or man.get("checkpoint_step") != s:
                raise SystemExit(f"refusing: {rd} manifest: hash {man.get('long8_hash')}, "
                                 f"step {man.get('checkpoint_step')}")
            revs.add(man["hf_revision"])
            jobs.append(("long", str(s), key, rd, man["hf_revision"],
                         {"t12": t12, "t123": t123}, None))
    if r0 is not None:
        for s in STEPS:
            lab = json.loads((r0 / f"step{s}.json").read_text())
            for key, p in sorted(lab["prompts"].items()):
                rd = Path(p["stage0_run"])
                man = json.loads((rd / "manifest.json").read_text())
                kept = list(p["kept"])
                n = len(json.loads((rd / "geometry.json").read_text())["tokens"])
                if man.get("checkpoint_step") != s or max(kept) >= n or min(kept) < 1:
                    raise SystemExit(f"refusing: R0's kept offsets for {key} at step {s} do not "
                                     f"index {rd}")
                c3x = c3x_groups(r0, r8x, str(s), key, kept) if r8x is not None else None
                meta["v1"][f"{s}|{key}"] = {"kept": len(kept),
                                            "c3x": sum(len(v) for v in (c3x or {}).values())}
                revs.add(man["hf_revision"])
                jobs.append(("v1", str(s), key, rd, man["hf_revision"],
                             {"r0": np.asarray(kept, dtype=int)}, c3x))
    return jobs, sorted(revs), meta


# ---------------------------------------------------------------- run

def _job(job, ln1_dir: Path, out: Path, code: str, device: str) -> str:
    kind, step, key, rd, rev, tg, c3x = job
    path = out / "records" / kind / f"step{step}_{key}.json"
    if path.exists():
        had = json.loads(path.read_text()).get("code")
        if had != code:
            raise SystemExit(f"refusing to resume: {path} was written by {had}, this is {code}")
        return f"have {path.name}"
    t0 = time.monotonic()
    recs, saved = read_run(rd, load_ln1(ln1_dir / f"{rev}.npz"), tg, key, step, c3x, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "records" / kind / f"step{step}_{key}.npz",
                        **saved, **{f"{k}/positions": v for k, v in tg.items()})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"kind": kind, "step": step, "passage": key, "run": str(rd),
                               "code": code, "device": device,
                               "targets": {k: int(len(v)) for k, v in tg.items()},
                               "seconds": round(time.monotonic() - t0, 1), "cells": recs}) + "\n")
    tmp.rename(path)
    return f"done {path.name} {time.monotonic() - t0:.0f}s"


def check_first(path: Path, job, ln1_dir: Path, device: str) -> Dict:
    """Refuse unless the first cell is populated and its GPU wells equal a CPU float64 run's."""
    rec = json.loads(path.read_text())
    bad = []
    for c in rec["cells"]:
        if c.get("sweep"):
            continue
        if not np.isfinite(c["sd_e"]) or c["unconverged"] > (1 - CONVERGED_MIN) * c["n"]:
            bad.append((c["layer"], c["targets"], c["beta"]))
    layers = {c["layer"] for c in rec["cells"]}
    z = np.load(path.with_suffix(".npz"))
    for k in z.files:
        if k.endswith("/e") and not np.isfinite(z[k]).all():
            bad.append(k)
        if "/wells_" in k and (z[k] < 0).any():
            bad.append(k)
    if bad or layers != set(LAYERS):
        raise SystemExit(f"refusing: first cell {path.name} not populated: {bad[:5]}")
    kind, step, key, rd, rev, tg, _ = job
    ln1 = load_ln1(ln1_dir / f"{rev}.npz")
    acts = np.load(rd / "activations.npz")
    A, N = acts["activations"], acts["norms"]
    agree = {}
    for L in CHECK_LAYERS:
        U = unit_rows(A[L] * N[L][:, None], ln1["w"][L], ln1["b"][L], ln1["eps"])[tg["t12"]]
        ref = mean_shift(U, PRIMARY_BETA, exact=True)["wells"]
        got = z[f"t12/wells_{PRIMARY_BETA:g}"][L]
        agree[L] = agreement(got, ref)
    print(f"first cell populated: {path.name}; {device} against CPU float64 wells: "
          + ", ".join(f"L{L} {v:.4f}" for L, v in agree.items()), flush=True)
    if min(agree.values()) < AGREE_MIN:
        raise SystemExit(f"refusing: GPU wells differ from the CPU float64 reference ({agree})")
    return {str(L): v for L, v in agree.items()}


def run(a) -> int:
    out, code = a.out, code_sha()
    jobs, revs, meta = plan(a.runs, a.r0, a.r8x)
    ln1_dir = out / "ln1"
    for r in revs:
        z = load_ln1(export_ln1(r, ln1_dir))
        if z["revision"] != r:
            raise SystemExit(f"refusing: LN1 for {r} holds {z['revision']}")
    meta.update(code=code, device=a.device, ln1={r: load_ln1(ln1_dir / f"{r}.npz")["snapshot"]
                                                 for r in revs},
                inputs={"runs": str(a.runs), "r0": str(a.r0), "r8x": str(a.r8x)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(meta, indent=1) + "\n")
    first = next(j for j in jobs if j[0] == "long" and (j[1], j[2]) == FIRST)
    print(_job(first, ln1_dir, out, code, a.device), flush=True)
    path = out / "records" / "long" / f"step{FIRST[0]}_{FIRST[1]}.json"
    chk = check_first(path, first, ln1_dir, a.device)
    (out / "first_check.json").write_text(json.dumps({"cell": path.name, "agreement": chk,
                                                      "min": AGREE_MIN}, indent=1) + "\n")
    if a.first_only:
        return 0
    for j in jobs:
        if j is not first and (not a.steps or j[1] in a.steps):
            print(_job(j, ln1_dir, out, code, a.device), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", type=Path, required=True)
    r.add_argument("--r0", type=Path, default=None, help="R0's labels dir (v1 beside)")
    r.add_argument("--r8x", type=Path, default=None, help="R8x's output dir (c3x groups on v1)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", nargs="*", default=None)
    r.add_argument("--first-only", action="store_true")
    r.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        from .u1_report import report
        return report(a.out)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
