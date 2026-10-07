"""
p1e_energy_field/u2_block.py — U2's block arm: does each block's update ascend the field
`φ_β` or descend it (`design-1e.md` "U2's block arm: the rule", fixed before any output).

Per (step, passage, block ℓ = 0–22), in β's frame (layer ℓ's LN1 applied to both residuals,
unit rows): the move ``d_i = P⊥_{u_i}(u'_i − u_i)`` against the force
``g_i = P⊥_{u_i} m_β(u_i)`` (M2), ``a_i = cos(d_i, g_i)``, ``A`` = mean over the targets.
Sources: ``causal`` (j ≤ i, primary), ``nosink`` (1 ≤ j ≤ i), ``full`` (every position) and
``local`` (causal ``m_β − m_0``, M4). β = 1.6, 3.5, 5.6. Targets: ``t12`` (T1 + T2, primary on
the long passages), ``t123`` (R0's rule, beside); on v1, R0's kept offsets (``r0``).

Null: 1,000 within-passage permutations of the moves among the targets (token i's force
against token k's move, projected onto i's tangent space); ``X = A − mean(A_null)`` and two
one-sided ranks. ``At`` / ``Xt``: the same with the mean move across targets subtracted
first. The permutations are drawn once per (passage, target count) and reused across steps,
blocks and fields, so nulls are correlated across cells; each cell's rank is still exact.
Not computed (fences): step lengths, pairwise ⟨·,·⟩ statistics of the cloud, per-head kernels.

Beside: each passage's mean next-token NLL per step, from the stored last hidden state
(after ``final_layer_norm``) through the checkpoint's ``embed_out``; no forward pass.

Tier 1: exploratory, unregistered. CPU, float64 (no forward pass here).
    python -m p1e_energy_field.u2_block run --runs <p1e_long8 dir> --r0 <R0 labels> --out <dir>
    python -m p1e_energy_field.u2_block report --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .extract_long8 import STEPS

BETAS = (1.6, 3.5, 5.6)
SOURCES = ("causal", "nosink", "full", "local")
#: Block 23 is refused: the stored last hidden state is after ``final_layer_norm``.
BLOCKS = tuple(range(23))
BANDS = {"L0": (0,), "L1-8": tuple(range(1, 9)), "L9-16": tuple(range(9, 17)),
         "L17-22": tuple(range(17, 23))}
N_PERM = 1000
SEED = 20261007
#: Both placed by the rule (`design-1e.md` "per token" and "first checks"), not calibrated.
TINY = 1e-9
UNIT_TOL = 1e-4
FIRST = ("143000", "wiki_paragraph_long")
REPO = "EleutherAI/pythia-410m"


# ---------------------------------------------------------------- pure

def unit_rows(X: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """``LN(x) / |LN(x)|`` row-wise, float64 (`attention_graph.unit_ln_rows`'s map)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    Y = (X - mu) / np.sqrt(((X - mu) ** 2).mean(axis=1, keepdims=True) + eps) * w + b
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def tangent(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Each row of ``V`` projected off its row of ``U`` (unit rows)."""
    return V - np.sum(V * U, axis=1, keepdims=True) * U


def forces(U: np.ndarray, beta: float, S: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """The tangential force ``g_i`` at every row of ``U`` under each source rule (M2, M4)."""
    n = U.shape[0]
    S = U @ U.T if S is None else S
    lower = np.tril(np.ones((n, n), dtype=bool))
    out = {}
    for src in ("causal", "nosink", "full"):
        mask = lower.copy() if src != "full" else np.ones((n, n), dtype=bool)
        if src == "nosink":
            mask[1:, 0] = False
        Z = np.where(mask, beta * S, -np.inf)
        W = np.exp(Z - Z.max(axis=1, keepdims=True))
        W /= W.sum(axis=1, keepdims=True)
        out[src] = W @ U
    m0 = np.cumsum(U, axis=0) / np.arange(1, n + 1)[:, None]
    out["local"] = out["causal"] - m0
    return {k: tangent(U, m) for k, m in out.items()}


def null_cos(Dg: np.ndarray, Du: np.ndarray, dn2: np.ndarray, perms: np.ndarray) -> tuple:
    """
    ``(a, A_null)``: per-target cosine and the permutation null of its mean. ``Dg[k, i] = d_k·ĝ_i``,
    ``Du[k, i] = d_k·u_i``, ``dn2[k] = |d_k|²``; under ``π`` token i reads ``d_{π(i)}`` projected
    onto its own tangent space, ``|P⊥_{u_i} d_k|² = |d_k|² − (d_k·u_i)²``.
    """
    ar = np.arange(Dg.shape[0])
    a = Dg[ar, ar] / np.sqrt(np.maximum(dn2 - Du[ar, ar] ** 2, 1e-300))
    num, du = Dg[perms, ar], Du[perms, ar]
    return a, (num / np.sqrt(np.maximum(dn2[perms] - du ** 2, 1e-300))).mean(axis=1)


def cell(d: np.ndarray, g: np.ndarray, U: np.ndarray, tgt: np.ndarray, perms_for) -> Dict:
    """One (block, field, target set): ``A``, its null, ``At`` (mean move out), position."""
    ok = (np.linalg.norm(d[tgt], axis=1) >= TINY) & (np.linalg.norm(g[tgt], axis=1) >= TINY)
    t = tgt[ok]
    D, Gh, Ut = d[t], g[t] / np.linalg.norm(g[t], axis=1, keepdims=True), U[t]
    P = perms_for(t.size)
    Dg, Du, dn2 = D @ Gh.T, D @ Ut.T, np.sum(D * D, axis=1)
    a, nul = null_cos(Dg, Du, dn2, P)
    dbar = D.mean(axis=0)
    Dgt = Dg - (Gh @ dbar)[None, :]
    Dut = Du - (Ut @ dbar)[None, :]
    dn2t = dn2 - 2 * (D @ dbar) + dbar @ dbar
    at, nult = null_cos(Dgt, Dut, dn2t, P)
    A, At = float(a.mean()), float(at.mean())
    from scipy.stats import spearmanr
    rho = spearmanr(a, np.log(t)).statistic if t.size > 2 and t.min() > 0 else float("nan")
    return {"n": int(t.size), "left_out": int((~ok).sum()), "A": A,
            "null_mean": float(nul.mean()), "null_sd": float(nul.std()), "X": A - float(nul.mean()),
            "p_hi": float((1 + np.sum(nul >= A)) / (len(nul) + 1)),
            "p_lo": float((1 + np.sum(nul <= A)) / (len(nul) + 1)),
            "At": At, "Xt": At - float(nult.mean()),
            "pt_hi": float((1 + np.sum(nult >= At)) / (len(nult) + 1)),
            "pt_lo": float((1 + np.sum(nult <= At)) / (len(nult) + 1)),
            "rho_logpos": float(rho)}


def make_perms(key: str):
    """Permutations per target count, one RNG per passage (``SEED``, crc32 of the key)."""
    cache = {}

    def get(n: int) -> np.ndarray:
        if n not in cache:
            rng = np.random.default_rng([SEED, zlib.crc32(key.encode()), n])
            cache[n] = np.stack([rng.permutation(n) for _ in range(N_PERM)])
        return cache[n]
    return get


def read_run(run_dir: Path, ln1: Dict, targets: Dict[str, np.ndarray], key: str,
             blocks: Sequence[int] = BLOCKS) -> List[Dict]:
    """Every block × β × source × target set of one stored run."""
    z = np.load(run_dir / "activations.npz")
    acts, norms = z["activations"], z["norms"]
    if acts.shape[0] != 25:
        raise ValueError(f"{run_dir}: {acts.shape[0]} hidden states, expected 25 (embedding + 24)")
    dev = float(np.abs(np.linalg.norm(acts, axis=2) - 1).max())
    if dev > UNIT_TOL:
        raise ValueError(f"{run_dir}: stored rows off unit by {dev:.1e} > {UNIT_TOL}; refusing")
    n = acts.shape[1]
    for name, t in targets.items():
        if t.size == 0 or t.min() < 1 or t.max() >= n:
            raise ValueError(f"{run_dir}: target set {name} does not index the stored positions")
    perms_for, out = make_perms(key), []
    for L in blocks:
        if L not in BLOCKS:
            raise ValueError(f"block {L} refused (`design-1e.md`: block 23's output is after final LN)")
        w, b, eps = ln1["w"][L], ln1["b"][L], ln1["eps"]
        U = unit_rows(acts[L] * norms[L][:, None], w, b, eps)
        U2 = unit_rows(acts[L + 1] * norms[L + 1][:, None], w, b, eps)
        d = tangent(U, U2 - U)
        S = U @ U.T
        for beta in BETAS:
            for src, g in forces(U, beta, S).items():
                for name, t in targets.items():
                    out.append({"block": L, "beta": beta, "source": src, "targets": name,
                                **cell(d, g, U, t, perms_for)})
    return out


# ---------------------------------------------------------------- weights and NLL (heavy, lazy)

def checkpoint_files(revision: str) -> tuple:
    from huggingface_hub import hf_hub_download
    st = Path(hf_hub_download(REPO, "model.safetensors", revision=revision))
    cfg = json.loads(Path(hf_hub_download(REPO, "config.json", revision=revision)).read_text())
    return st, cfg


def export_ln1(revision: str, out: Path) -> Path:
    """Layer norms' LN1 weight / bias (24, d) and eps of one checkpoint, from the cache."""
    from safetensors import safe_open
    path = out / f"{revision}.npz"
    if path.exists():
        return path
    st, cfg = checkpoint_files(revision)
    with safe_open(str(st), "np") as f:
        w = np.stack([f.get_tensor(f"gpt_neox.layers.{l}.input_layernorm.weight") for l in range(24)])
        b = np.stack([f.get_tensor(f"gpt_neox.layers.{l}.input_layernorm.bias") for l in range(24)])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(path, w=w, b=b, eps=float(cfg["layer_norm_eps"]), revision=revision,
             snapshot=st.parent.name)
    return path


def load_ln1(path: Path) -> Dict:
    z = np.load(path)
    return {"w": z["w"].astype(np.float64), "b": z["b"].astype(np.float64), "eps": float(z["eps"]),
            "revision": str(z["revision"]), "snapshot": str(z["snapshot"])}


def passage_nll(run_dirs: Dict[str, Path], revision: str, tok) -> Dict[str, float]:
    """Mean next-token NLL per passage at one checkpoint (stored final state × ``embed_out``)."""
    from safetensors import safe_open
    st, _ = checkpoint_files(revision)
    with safe_open(str(st), "np") as f:
        E = f.get_tensor("embed_out.weight").astype(np.float32)
    out = {}
    for key, rd in run_dirs.items():
        z = np.load(rd / "activations.npz")
        H = (z["activations"][-1] * z["norms"][-1][:, None]).astype(np.float32)
        tokens = json.loads((rd / "geometry.json").read_text())["tokens"]
        ids = np.asarray(tok.convert_tokens_to_ids(tokens))
        if tok.convert_ids_to_tokens(ids.tolist()) != tokens:
            raise ValueError(f"{rd}: stored tokens do not round-trip through the tokenizer")
        nll = []
        for a in range(0, len(ids) - 1, 256):
            lg = H[a:min(a + 256, len(ids) - 1)] @ E.T
            lg = lg.astype(np.float64)
            lse = lg.max(axis=1) + np.log(np.exp(lg - lg.max(axis=1, keepdims=True)).sum(axis=1))
            nxt = ids[a + 1:a + 1 + lg.shape[0]]
            nll.append(lse - lg[np.arange(lg.shape[0]), nxt])
        out[key] = float(np.concatenate(nll).mean())
    return out


# ---------------------------------------------------------------- one job

def _job(args) -> str:
    kind, step, key, run_dir, ln1_path, tg, out = args
    path = Path(out) / "records" / kind / f"step{step}_{key}.json"
    if path.exists():
        return f"have {path.name}"
    t0 = time.monotonic()
    targets = {k: np.asarray(v, dtype=int) for k, v in tg.items()}
    recs = read_run(Path(run_dir), load_ln1(Path(ln1_path)), targets, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"kind": kind, "step": step, "passage": key, "run": str(run_dir),
                               "targets": {k: len(v) for k, v in tg.items()}, "cells": recs}) + "\n")
    tmp.rename(path)
    return f"done {path.name} {time.monotonic() - t0:.0f}s"


def check_first(path: Path) -> None:
    """Refuse unless ≥ 90 % of targets have a finite ``a_i`` at every block (populated)."""
    rec = json.loads(path.read_text())
    bad = [c for c in rec["cells"]
           if c["n"] < 0.9 * rec["targets"][c["targets"]] or not np.isfinite(c["A"])
           or not np.isfinite(c["X"])]
    blocks = {c["block"] for c in rec["cells"]}
    if bad or blocks != set(BLOCKS):
        raise SystemExit(f"refusing: first cell {path.name} not populated "
                         f"({len(bad)} bad cells, blocks {sorted(blocks)[:3]}…)")
    print(f"first cell populated: {path.name}, {len(rec['cells'])} cells")


# ---------------------------------------------------------------- plan

def plan(runs: Path, r0: Optional[Path], out: Path) -> tuple:
    from .long8_targets import target_positions
    from .long_prompts_1e import LONG8_HASH
    keys = sorted({d.name.split("_", 1)[1] for d in runs.glob("pythia-410m-step0_*")})
    if len(keys) != 8:
        raise SystemExit(f"refusing: {len(keys)} long passages in {runs}, expected 8")
    jobs, revs, meta = [], set(), {"targets": {}, "v1": {}}
    for key in keys:
        _, massive, t12, t123 = target_positions(runs, key)
        meta["targets"][key] = {"t12": int(t12.size), "t123": int(t123.size),
                                "massive": sorted(int(p) for p in massive)}
        for s in STEPS:
            rd = runs / f"pythia-410m-step{s}_{key}"
            man = json.loads((rd / "manifest.json").read_text())
            if man.get("long8_hash") != LONG8_HASH or man.get("checkpoint_step") != s:
                raise SystemExit(f"refusing: {rd} manifest: hash {man.get('long8_hash')}, "
                                 f"step {man.get('checkpoint_step')}")
            revs.add(man["hf_revision"])
            jobs.append(["long", str(s), key, str(rd), man["hf_revision"],
                         {"t12": t12.tolist(), "t123": t123.tolist()}])
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
                                     f"index {rd} (n = {n}, step {man.get('checkpoint_step')})")
                meta["v1"][f"{s}|{key}"] = len(kept)
                revs.add(man["hf_revision"])
                jobs.append(["v1", str(s), key, str(rd), man["hf_revision"], {"r0": kept}])
    return jobs, sorted(revs), meta


def run(a) -> int:
    out = a.out
    jobs, revs, meta = plan(a.runs, a.r0, out)
    ln1_dir = out / "ln1"
    ln1 = {r: str(export_ln1(r, ln1_dir)) for r in revs}
    for r, p in ln1.items():
        z = load_ln1(Path(p))
        if z["revision"] != r:
            raise SystemExit(f"refusing: {p} holds {z['revision']}, not {r}")
    meta["ln1"] = {r: load_ln1(Path(p))["snapshot"] for r, p in ln1.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(meta, indent=1) + "\n")
    args = [(k, s, key, rd, ln1[rev], tg, str(out)) for k, s, key, rd, rev, tg in jobs]
    first = [x for x in args if x[0] == "long" and (x[1], x[2]) == FIRST]
    rest = [x for x in args if x not in first]
    if a.steps:
        rest = [x for x in rest if x[1] in a.steps]
    print(_job(first[0]), flush=True)
    check_first(out / "records" / "long" / f"step{FIRST[0]}_{FIRST[1]}.json")
    if a.first_only:
        return 0
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(_job, rest):
            print(msg, flush=True)
    return 0


def nll(a) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, revision="step143000")
    keys = sorted({d.name.split("_", 1)[1] for d in a.runs.glob("pythia-410m-step0_*")})
    rec = {}
    for s in STEPS:
        rec[str(s)] = r = passage_nll({k: a.runs / f"pythia-410m-step{s}_{k}" for k in keys},
                                      f"step{s}", tok)
        print(s, {k: round(v, 3) for k, v in r.items()}, flush=True)
    (a.out / "nll.json").write_text(json.dumps(rec, indent=1) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", type=Path, required=True)
    r.add_argument("--r0", type=Path, default=None, help="R0's labels dir (v1 beside)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=14)
    r.add_argument("--steps", nargs="*", default=None)
    r.add_argument("--first-only", action="store_true")
    n = sub.add_parser("nll")
    n.add_argument("--runs", type=Path, required=True)
    n.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        from .u2_report import report
        return report(a.out)
    return {"run": run, "nll": nll}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
