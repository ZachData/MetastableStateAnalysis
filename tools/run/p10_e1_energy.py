"""E1: energy against c3x's groups (`p10_cluster_function/design-10.md` "E1", fixed before any pass).

Does attention's token-specific update pull a c3x group's members towards each other, towards the
group's own members more than towards as many other tokens? Per (step, passage) one hooked GPU pass
(`p1e_energy_field.u2_attn.Hooked`) splits every block's update; in 1e U2's frame (block ℓ's LN1,
unit rows) each component's move ``d_i`` (``r1out``: less its component along the passage's shared
direction) is read against the field of a set S of kept offsets, ``g^S_i = P⊥_{u_i} Σ_{j∈S∖{i}}
softmax_j(β u_i·u_j) u_j``, full (not causal). Per group g at layer L, read at block L (block L−1
beside): ``X_g = A(g) − mean A(S′)`` over 200 size-matched draws S′ of kept non-members, the
members' own moves held fixed. Beside: ``X_knn`` (the |g| nearest non-members) and the members'
move coherence ``C(g)`` against the draws' own.

Every cosine is a ratio of scalars: ``d_i`` is tangent at ``u_i``, so ``d_i·g^S_i ∝ Σ_j w_j
d_i·u_j`` and ``|g^S_i|² ∝ wᵀ S_SS w − (Σ_j w_j u_i·u_j)²`` (``S`` the Gram of the kept rows), and
the draws cost no vectors. Tier 1: exploratory, unregistered.
    python tools/run/p10_e1_energy.py run --labels <R9 source> --out <dir> [--steps 512 ...]
    python tools/run/p10_e1_energy.py report --out <dir> --u2-attn <1e attention arm dir>
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
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from tools.run import p10_label_source as ls

STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
PRIMARY_STEPS = tuple(s for s in STEPS if s >= 64)       # the rule's, from the structure count
PASSAGES = ("wiki_paragraph", "sullivan_ballou", "paper_excerpt", "homer_iliad", "hdbscan_code",
            "camus_letranger", "latex_monograph")
LAYERS = tuple(range(1, 24))                             # L24 has no block
BANDS = {"L1-8": tuple(range(1, 9)), "L9-16": tuple(range(9, 17)), "L17-23": tuple(range(17, 24))}
#: 1e's v1 bands are blocks; its last is L17-22 (the rule sets E1's L17-23 against it).
BANDS_1E = {"L1-8": "L1-8", "L9-16": "L9-16", "L17-23": "L17-22"}
BETAS = (3.5, 1.6, 5.6, 0.0)
PRIMARY = ("dep", "attn:r1out", 3.5)
#: (component, reading): primary first.
READS = (("attn", "r1out"), ("keys", "r1out"), ("attn", "frozen"), ("mlpx", "r1out"), ("block", "r1out"))
BLOCKS = {"dep": 0, "arr": -1}                           # block L (departure), L−1 (arrival)
COLUMNS = ("c3x", "c2")
N_DRAWS = 200
TINY = 1e-9
MIN_RECORDS, MIN_PASSAGES = 3, 6                          # placed by the rule
FIRST = (512, "wiki_paragraph")
POPULATED_SHARE = 0.9


class E1Error(RuntimeError):
    pass


# ---------------------------------------------------------------- the scores (pure)

def set_cosines(S: np.ndarray, M: np.ndarray, dn: np.ndarray, rows: np.ndarray, sets: np.ndarray,
                beta: float) -> np.ndarray:
    """
    ``a[c, d, m] = cos(d^c_i, g^{S_d}_i)`` for member ``i = rows[m]`` and source set ``sets[d]``
    (each set shared by every member; a member inside a set is left out of its own field).
    ``S`` (n, n) the Gram of the unit rows, ``M`` (C, n, n) ``M[c, i, j] = d^c_i·u_j``, ``dn``
    (C, n) ``|d^c_i|``. NaN where ``|d_i|`` or ``|g_i|`` < ``TINY``.
    """
    rows, sets = np.asarray(rows), np.asarray(sets)
    Si = S[rows[None, :, None], sets[:, None, :]]                   # (D, m, k)
    keep = sets[:, None, :] != rows[None, :, None]
    Z = np.where(keep, beta * Si, -np.inf)
    w = np.exp(Z - Z.max(axis=-1, keepdims=True))
    w = np.where(keep, w, 0.0)
    wsum = w.sum(axis=-1)                                            # (D, m)
    Sjk = S[sets[:, :, None], sets[:, None, :]]                     # (D, k, k)
    g2 = np.einsum("dmj,djk,dmk->dm", w, Sjk, w) - np.einsum("dmk,dmk->dm", w, Si) ** 2
    gn = np.sqrt(np.maximum(g2, 0.0))
    num = np.einsum("dmk,cdmk->cdm", w, M[:, rows[None, :, None], sets[:, None, :]])
    dm = dn[:, rows][:, None, :]                                     # (C, 1, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = num / (dm * gn[None])
    bad = (dm < TINY) | ((gn / np.maximum(wsum, 1e-300)) < TINY)[None]
    return np.where(bad, np.nan, a)


def coherence(Q: np.ndarray, ok: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """``C[c, d]``: the mean pairwise cosine of the moves over ``sets[d]``'s valid rows.
    ``Q`` (C, n, n) the Gram of the unit moves (rows with ``ok`` False zeroed); ``ok`` (C, n)."""
    sets = np.asarray(sets)
    tot = Q[:, sets[:, :, None], sets[:, None, :]].sum(axis=(-1, -2))   # (C, D), diagonal = m'
    m = ok[:, sets].sum(axis=-1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m >= 2, (tot - m) / (m * (m - 1)), np.nan)


def draw_sets(seed_key: str, pool: np.ndarray, size: int, n_draws: int = N_DRAWS) -> np.ndarray:
    """``(n_draws, size)`` rows drawn without replacement from ``pool``; one generator per group."""
    if pool.size < size:
        raise E1Error(f"{seed_key}: {pool.size} non-members for a group of {size}")
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
    return np.stack([rng.choice(pool, size=size, replace=False) for _ in range(n_draws)])


def nearest_set(U: np.ndarray, rows: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """The ``len(rows)`` rows of ``pool`` nearest the members' mean unit row (cosine)."""
    c = U[rows].mean(axis=0)
    return pool[np.argsort(-(U[pool] @ c), kind="stable")[:rows.size]]


def rank_hi(obs: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    return float((1 + np.sum(null >= obs)) / (null.size + 1))


def rank_lo(obs: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    return float((1 + np.sum(null <= obs)) / (null.size + 1))


def score_group(blk: Dict, rows: np.ndarray, pool: np.ndarray, draws: np.ndarray) -> Dict:
    """One group at one block: per reading and β, A, the null, X, ranks, X_knn; per reading C(g)."""
    S, M, dn, Q, ok, U = blk["S"], blk["M"], blk["dn"], blk["Q"], blk["ok"], blk["U"]
    own = rows[None, :]
    knn = nearest_set(U, rows, pool)[None, :]
    out: Dict = {"reads": {}}
    for be in BETAS:
        a_own = set_cosines(S, M, dn, rows, own, be)[:, 0]            # (C, m)
        a_nul = set_cosines(S, M, dn, rows, draws, be)                # (C, D, m)
        a_knn = set_cosines(S, M, dn, rows, knn, be)[:, 0]
        for c, (comp, rd) in enumerate(READS):
            A = float(np.nanmean(a_own[c])) if np.isfinite(a_own[c]).any() else float("nan")
            nul = np.nanmean(a_nul[c], axis=-1)
            Ak = float(np.nanmean(a_knn[c])) if np.isfinite(a_knn[c]).any() else float("nan")
            out["reads"].setdefault(f"{comp}:{rd}", {})[str(be)] = {
                "A": A, "null_mean": float(np.nanmean(nul)), "null_sd": float(np.nanstd(nul)),
                "X": A - float(np.nanmean(nul)), "p_hi": rank_hi(A, nul), "p_lo": rank_lo(A, nul),
                "X_knn": A - Ak, "kept": int(np.isfinite(a_own[c]).sum())}
    c_own = coherence(Q, ok, own)[:, 0]
    c_nul = coherence(Q, ok, draws)
    for c, (comp, rd) in enumerate(READS):
        nul = c_nul[c]
        out["reads"][f"{comp}:{rd}"]["coherence"] = {
            "C": float(c_own[c]), "null_mean": float(np.nanmean(nul)), "X": float(c_own[c] - np.nanmean(nul)),
            "p_hi": rank_hi(float(c_own[c]), nul)}
    return out


# ---------------------------------------------------------------- one block's frame (pure)

def block_frame(x: np.ndarray, comps: Dict[str, np.ndarray], w: np.ndarray, b: np.ndarray, eps: float,
                t: np.ndarray) -> Dict:
    """Unit rows, Gram and moves at the kept rows ``t`` for one block (1e U2's frame and readings)."""
    from p1e_energy_field import u2_block as ub
    X = x[t].astype(np.float64)
    U = ub.unit_rows(X, w, b, eps)
    D = []
    for comp, rd in READS:
        c = comps[comp][t].astype(np.float64)
        if rd == "r1out":
            cbar = c.mean(axis=0)
            chat = cbar / np.linalg.norm(cbar)
            c = c - np.outer(c @ chat, chat)
        D.append(ub.tangent(U, ub.unit_rows(X + c, w, b, eps) - U))
    D = np.stack(D)                                                    # (C, n, d)
    dn = np.linalg.norm(D, axis=-1)
    ok = dn >= TINY
    Dh = np.where(ok[..., None], D / np.maximum(dn, 1e-300)[..., None], 0.0)
    return {"U": U, "S": U @ U.T, "M": np.einsum("cnd,md->cnm", D, U), "dn": dn, "ok": ok,
            "Q": np.einsum("cnd,cmd->cnm", Dh, Dh)}


def groups_at(src: Path, step: int, passage: str, L: int, cache: Dict) -> Dict:
    """``{gid: (rows into the kept offsets, in c3x)}`` for c2's groups at L; refuses unless c3x ⊆ c2
    with the same members, both on R0's kept offsets."""
    pos2, lab2 = ls.load_column(src, f"step{step}", passage, L, "c2", cache)
    posx, labx = ls.load_column(src, f"step{step}", passage, L, "c3x", cache)
    if not np.array_equal(pos2, posx):
        raise E1Error(f"step {step} {passage} L{L}: c2 and c3x domains differ")
    out = {}
    for g in np.unique(lab2[lab2 >= 0]):
        rows = np.flatnonzero(lab2 == g)
        inx = np.unique(labx[rows])
        if inx.size != 1 or (inx[0] >= 0 and not np.array_equal(np.flatnonzero(labx == inx[0]), rows)):
            raise E1Error(f"step {step} {passage} L{L}: c3x group differs from c2 group {g}")
        out[int(g)] = (rows, bool(inx[0] >= 0))
    if set(np.unique(labx[labx >= 0]).tolist()) - {int(labx[r[0]]) for r, x in out.values() if x}:
        raise E1Error(f"step {step} {passage} L{L}: a c3x group is not a c2 group")
    return out


def read_passage(hs: np.ndarray, comps: Dict, ln: Dict, src: Path, step: int, passage: str,
                 cache: Dict, kept: np.ndarray) -> List[Dict]:
    """Every c2 group (c3x flagged) at L1–23 of one pass, at block L and L−1."""
    frames: Dict[int, Dict] = {}

    def frame(bk: int) -> Dict:
        if bk not in frames:
            frames[bk] = block_frame(hs[bk], comps[bk], ln["w"][bk], ln["b"][bk], ln["eps"], kept)
        return frames[bk]

    recs = []
    for L in LAYERS:
        for g, (rows, inx) in groups_at(src, step, passage, L, cache).items():
            pool = np.setdiff1d(np.arange(kept.size), rows)
            draws = draw_sets(f"{step}|{passage}|{L}|{g}", pool, rows.size)
            rec = {"layer": L, "group": g, "c3x": inx, "size": int(rows.size), "blocks": {}}
            for kind, off in BLOCKS.items():
                rec["blocks"][kind] = score_group(frame(L + off), rows, pool, draws)
            recs.append(rec)
        frames.pop(L - 1, None)                     # block L−1 is not read again
    return recs


# ---------------------------------------------------------------- the batch

def code_sha() -> str:
    """This file's blob hash at HEAD; refuses on an uncommitted change (each record names its producer)."""
    here = Path(__file__).resolve()
    git = ["git", "-C", str(here.parent)]
    if subprocess.run(git + ["status", "--porcelain", "--", here.name], capture_output=True, text=True,
                      check=True).stdout.strip():
        raise SystemExit(f"refusing: {here.name} has uncommitted changes; commit first")
    return subprocess.run(git + ["rev-parse", f"HEAD:tools/run/{here.name}"], capture_output=True,
                          text=True, check=True).stdout.strip()


def check_last_block(hs: np.ndarray, C: Dict[str, np.ndarray], model) -> float:
    """Block 23 (*departure at the gate*): the stored ``hs[24]`` is after ``final_layer_norm``, so
    `u2_attn.check_pass` cannot compare ``x + attn + mlp`` with it; compare ``LN_f(x + attn + mlp)``
    instead, and the parts' sum with the block, both relative, at `u2_attn.SPLIT_TOL`."""
    from p1e_energy_field import u2_attn as ua
    fl = getattr(model, "gpt_neox", model).final_layer_norm
    w, b = (t.detach().double().cpu().numpy() for t in (fl.weight, fl.bias))
    y = hs[23].astype(np.float64) + C["block"].astype(np.float64)
    mu = y.mean(axis=1, keepdims=True)
    ln = (y - mu) / np.sqrt(((y - mu) ** 2).mean(axis=1, keepdims=True) + fl.eps) * w + b
    ref = hs[24].astype(np.float64)
    rel = float(np.abs(ln - ref).max() / np.abs(ref).max())
    d = np.linalg.norm(C["block"], axis=1).max()
    split = float(np.abs(sum(C[k] for k in ua.PARTS) - C["block"]).max() / d)
    if rel > ua.SPLIT_TOL or split > ua.SPLIT_TOL:
        raise SystemExit(f"refusing: block 23: LN_f(x + attn + mlp) off by {rel:.1e}, parts off by {split:.1e}")
    return rel


def check_populated(recs: List[Dict]) -> None:
    """The rule's first-record check: every c3x group has a finite X_g with ≥ 90 % of members kept."""
    px = [r for r in recs if r["c3x"]]
    if not px:
        raise SystemExit("refusing: first record has no c3x group")
    vals = []
    for r in px:
        cell = r["blocks"]["dep"]["reads"][PRIMARY[1]][str(PRIMARY[2])]
        if not np.isfinite(cell["X"]) or cell["kept"] < POPULATED_SHARE * r["size"]:
            raise SystemExit(f"refusing: first record, L{r['layer']} group {r['group']}: X {cell['X']}, "
                             f"{cell['kept']} of {r['size']} members kept")
        vals.append(cell["X"])
    if len(set(np.round(vals, 12))) < 2:
        raise SystemExit("refusing: first record's X_g are all equal")


def run(a) -> int:
    import torch
    from transformers import AutoTokenizer
    from core.models import load_model
    from p1e_energy_field import u2_attn as ua
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (the rule's pass is the GPU's)")
    code = code_sha()
    src = Path(a.labels)
    summary = src / "summary.json"
    meta = {"rule": "design-10.md \"E1\"", "code": code, "labels": str(src),
            "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest()[:16],
            "n_draws": N_DRAWS, "betas": BETAS, "reads": READS, "primary": PRIMARY}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps(meta, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    steps = [FIRST[0]] + [s for s in (a.steps or STEPS) if s != FIRST[0]]
    first_done = (a.out / "records" / f"step{FIRST[0]}_{FIRST[1]}.json").exists()
    for step in steps:
        cache: Dict = {}
        d = ls.load_step(src, f"step{step}")
        todo = [p for p in PASSAGES if not (a.out / "records" / f"step{step}_{p}.json").exists()]
        if step == FIRST[0]:
            todo = sorted(todo, key=lambda p: p != FIRST[1])
        if not todo:
            print(f"have step {step}", flush=True)
            continue
        model, _ = load_model(f"pythia-410m-step{step}")
        if next(model.parameters()).dtype != torch.float32:
            raise SystemExit("refusing: model is not float32")
        hk, ln = ua.Hooked(model), ua.ln_from(model)
        for p in todo:
            t0 = time.monotonic()
            pr = d["prompts"][p]
            rd = Path(pr["stage0_run"])
            ids = torch.tensor([ua.token_ids(tok, rd)], device=next(model.parameters()).device)
            hs, comps = hk.run(ids, tuple(range(24)))
            chk = ua.check_pass(hs, {L: comps[L] for L in range(23)}, rd, "v1")
            chk["last_block_rel"] = check_last_block(hs, comps[23], model)
            kept = np.asarray(pr["kept"], dtype=int)
            recs = read_passage(hs, comps, ln, src, step, p, cache, kept)
            rec = {"step": step, "passage": p, "run": str(rd), "code": code, "pass": "cuda:float32",
                   "kept": int(kept.size), "checks": chk, "groups": recs}
            if (step, p) == FIRST and not first_done:
                check_populated(recs)
                first_done = True
                print(f"first record populated: {sum(r['c3x'] for r in recs)} c3x, {len(recs)} c2 groups; "
                      f"checks {chk}", flush=True)
            path = a.out / "records" / f"step{step}_{p}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec) + "\n")
            tmp.rename(path)
            print(f"done step {step} {p}: {len(recs)} groups, {time.monotonic() - t0:.0f}s, "
                  f"match {chk['match_unit']:.1e}", flush=True)
            if a.first_only:
                hk.close()
                return 0
        hk.close()
        del model, hk
        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- the reading

def sign_label(vals: Sequence[float]) -> str:
    """The rule's sign rule over readable passages (≥ 6): all > 0, all but one, the same below 0."""
    v = [x for x in vals if np.isfinite(x)]
    n = len(v)
    if n < MIN_PASSAGES:
        return "too few"
    pos, neg = sum(x > 0 for x in v), sum(x < 0 for x in v)
    if pos == n:
        return "pulls together"
    if neg == n:
        return "pushes apart"
    if pos == n - 1:
        return "leans pulls"
    if neg == n - 1:
        return "leans pushes"
    return "mixed"


def chance(n: int) -> Dict[str, float]:
    return {"full_either": 2 / 2 ** n, "lean_either": 2 * n / 2 ** n}


def cell_values(recs: Dict, step: int, band: str, column: str, kind: str, read: str, beta: float,
                field: str = "X") -> Dict[str, float]:
    """Each passage's mean over its group-layer records in the band (NaN under ``MIN_RECORDS``)."""
    out = {}
    for p in PASSAGES:
        r = recs.get((step, p))
        if r is None:
            out[p] = float("nan")
            continue
        xs = []
        for g in r["groups"]:
            if g["layer"] not in BANDS[band] or (column == "c3x" and not g["c3x"]):
                continue
            c = g["blocks"][kind]["reads"][read]
            v = c["coherence"]["X"] if field == "coherence" else c[str(beta)][field]
            if np.isfinite(v):
                xs.append(v)
        out[p] = float(np.mean(xs)) if len(xs) >= MIN_RECORDS else float("nan")
    return out


def group_shares(recs: Dict, step: int, band: str, column: str, kind: str, read: str, beta: float) -> Dict:
    xs, ps, ks, cs = [], [], [], []
    for p in PASSAGES:
        for g in (recs.get((step, p)) or {"groups": []})["groups"]:
            if g["layer"] in BANDS[band] and (column == "c2" or g["c3x"]):
                c = g["blocks"][kind]["reads"][read]
                if np.isfinite(c[str(beta)]["X"]):
                    xs.append(c[str(beta)]["X"]); ps.append(c[str(beta)]["p_hi"])
                    ks.append(c[str(beta)]["X_knn"]); cs.append(c["coherence"]["X"])
    if not xs:
        return {"n": 0}
    xs, ps, ks, cs = map(np.asarray, (xs, ps, ks, cs))
    return {"n": int(xs.size), "share_pos": float((xs > 0).mean()), "share_p05": float((ps <= 0.05).mean()),
            "median_abs_X": float(np.median(np.abs(xs))), "median_X": float(np.median(xs)),
            "share_knn_pos": float(np.nanmean(ks > 0)), "share_coh_pos": float(np.nanmean(cs > 0))}


def load_records(out: Path) -> Dict:
    recs = {}
    for f in sorted((out / "records").glob("step*_*.json")):
        r = json.loads(f.read_text())
        recs[(int(r["step"]), r["passage"])] = r
    return recs


def labels_1e(u2_attn: Optional[Path]) -> Dict:
    if u2_attn is None:
        return {}
    rows = json.loads((u2_attn / "report.json").read_text())["v1"]["labels"]["rows"]
    return {(int(k.split("|")[3]), b): rows.get(f"r0|causal:attn:r1out|3.5|{k.split('|')[3]}|{b1}", {}).get("label")
            for k in rows for b, b1 in BANDS_1E.items() if k.startswith("r0|causal:attn:r1out|3.5|")}


def report(a) -> int:
    recs = load_records(a.out)
    codes = {r["code"] for r in recs.values()}
    if len(codes) != 1:
        raise SystemExit(f"refusing: records from {len(codes)} producers {sorted(codes)}")
    steps = sorted({s for s, _ in recs})
    l1e = labels_1e(a.u2_attn)
    res: Dict = {"code": codes.pop(), "steps": steps, "rows": {}, "shares": {}, "chance": {}}
    rows = [(k, f"{c}:{r}", b, f) for k in BLOCKS for c, r in READS for b in BETAS for f in ("X",)]
    rows += [(k, f"{c}:{r}", 3.5, f) for k in BLOCKS for c, r in READS for f in ("X_knn", "coherence")]
    for col in COLUMNS:
        for kind, read, beta, field in rows:
            key = f"{col}|{kind}|{read}|{beta}|{field}"
            res["rows"][key] = {}
            for s in steps:
                for band in BANDS:
                    v = cell_values(recs, s, band, col, kind, read, beta, field)
                    res["rows"][key][f"{s}|{band}"] = {"label": sign_label(list(v.values())), "values": v}
    # β-robust, step-0 baseline (c2's), isolated leans: primary rows only
    for col in COLUMNS:
        prim = res["rows"][f"{col}|dep|attn:r1out|3.5|X"]
        for s in steps:
            for band in BANDS:
                c = prim[f"{s}|{band}"]
                c["beta_robust"] = all(res["rows"][f"{col}|dep|attn:r1out|{b}|X"][f"{s}|{band}"]["label"]
                                       == c["label"] for b in (1.6, 5.6))
                c["step0_c2"] = res["rows"]["c2|dep|attn:r1out|3.5|X"].get(f"0|{band}", {}).get("label")
                c["learned"] = c["label"] not in ("mixed", "too few") and c["label"] != c["step0_c2"]
                if c["label"].startswith("leans"):
                    i = steps.index(s)
                    nb = [steps[j] for j in (i - 1, i + 1) if 0 <= j < len(steps)]
                    c["isolated"] = not any(prim[f"{t}|{band}"]["label"] == c["label"] for t in nb)
                c["u2_attn_1e"] = l1e.get((s, band))
                res["shares"][f"{col}|{s}|{band}"] = group_shares(recs, s, band, col, "dep", "attn:r1out", 3.5)
        n_read = [sum(np.isfinite(list(prim[f"{s}|{b}"]["values"].values()))) for s in PRIMARY_STEPS
                  if s in steps for b in BANDS]
        res["chance"][col] = {"cells_read": sum(n >= MIN_PASSAGES for n in n_read),
                              "expected_full_either": sum(chance(n)["full_either"] for n in n_read if n >= MIN_PASSAGES),
                              "expected_lean_either": sum(chance(n)["lean_either"] for n in n_read if n >= MIN_PASSAGES)}
    (a.out / "report.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = [f"E1 (design-10.md \"E1\"), producer {res['code'][:10]}, {len(recs)} records; primary dep "
             "attn:r1out β 3.5; marks: β = β-robust, L = learned (c2's step 0 differs), i = isolated lean"]
    for col in COLUMNS:
        prim = res["rows"][f"{col}|dep|attn:r1out|3.5|X"]
        lines.append(f"\n== {col}  (chance over primary cells: {res['chance'][col]})")
        lines.append(f"{'step':>7} | " + " | ".join(f"{b:<34}" for b in BANDS))
        for s in steps:
            cells = []
            for b in BANDS:
                c = prim[f"{s}|{b}"]
                v = [x for x in c["values"].values() if np.isfinite(x)]
                mk = ("β" if c.get("beta_robust") else "") + ("L" if c.get("learned") else "") + ("i" if c.get("isolated") else "")
                sh = res["shares"][f"{col}|{s}|{b}"]
                cells.append(f"{c['label']:<14}{mk:<3} {np.mean(v) if v else float('nan'):+.4f} "
                             f"1e:{(c.get('u2_attn_1e') or '-')[:9]:<9}" + (f" n{sh['n']}" if sh.get("n") else ""))
            lines.append(f"{s:>7} | " + " | ".join(f"{x:<34}" for x in cells))
    lines.append("\n== beside (c3x, dep unless named): label per row, steps 64-143000 x bands")
    for kind, read, beta, field in rows:
        key = f"c3x|{kind}|{read}|{beta}|{field}"
        labs = [res["rows"][key][f"{s}|{b}"]["label"] for s in steps if s in PRIMARY_STEPS for b in BANDS]
        from collections import Counter
        lines.append(f"{kind:>3} {read:<12} β{beta:<4} {field:<9} " + ", ".join(f"{k} {v}" for k, v in Counter(labs).most_common()))
    (a.out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--labels", type=Path, required=True, help="the R9 label source (c3x, c2)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", type=int, nargs="*", default=None)
    r.add_argument("--first-only", action="store_true", help="the first record only, then stop")
    q = sub.add_parser("report")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--u2-attn", type=Path, default=None, help="1e's attention arm (its v1 labels beside)")
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
