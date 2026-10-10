"""E1p: E1m on the moves less their group-blind part (`p10_cluster_function/design-10.md` "E1p").

E1m (`p10_e1m_match.py`) pairs each member's fellows with non-members of the same similarity to it and
asks whether attention's move ``d_i`` points at the fellows more; `/challenge-pr` on #177 showed that a
pull that ignores the group (towards neighbours, towards the cloud's mean) also scores > 0 there. The
parked pseudo-group placebo failed its own planted null (the rule says how), so here each move loses
its projection on the **group-blind basis** of the idealised field, per token and view: the field at
β 0 / 1.6 / 3.5 / 5.6 over the visible rows and the pull towards the 8 most similar of them. E1m's
pairing and score run on what is left (``X_p``); E1m's unprojected ``X`` is recomputed in the same pass
and must reproduce E1m's stored records.

Same frame, moves, groups, pairing and views as E1m. Tier 1: exploratory, unregistered.
    python tools/run/p10_e1p_project.py run --labels <R9 source> --out <dir> [--steps 512 ...]
    python tools/run/p10_e1p_project.py report --out <dir> --e1m <E1m dir> --u2-attn <1e attention arm dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from tools.run import p10_e1_energy as e1
from tools.run import p10_e1m_match as e1m
from tools.run import p10_label_source as ls

STEPS, PRIMARY_STEPS, PASSAGES = e1.STEPS, e1.PRIMARY_STEPS, e1.PASSAGES
LAYERS, BANDS, READS, TINY, BETAS = e1.LAYERS, e1.BANDS, e1.READS, e1.TINY, e1m.BETAS
VIS, PRIMARY, COLUMNS, EPS, FIRST = e1m.VIS, e1m.PRIMARY, e1m.COLUMNS, e1m.EPS, e1m.FIRST
BASIS_BETAS = (0.0, 1.6, 3.5, 5.6)                         # E1's βs; β 0 is the visible cloud's mean
NN = 8                                                     # E1m's planted neighbour pull
SV_TOL = 1e-8                                              # placed: relative to the largest singular value
REPRO_TOL = 1e-9                                           # placed: E1m amendment's reproduce bound
MIN_READABLE_SHARE = e1m.MIN_READABLE_SHARE


class E1pError(RuntimeError):
    pass


# ---------------------------------------------------------------- the frame, the basis, the residual (pure)

def frame_moves(x: np.ndarray, comps: Dict[str, np.ndarray], w: np.ndarray, b: np.ndarray, eps: float,
                t: np.ndarray) -> Dict:
    """`p10_e1_energy.block_frame`'s unit rows and moves, keeping the moves ``D`` (C, n, d)."""
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
    return {"U": U, "S": U @ U.T, "D": np.stack(D)}


def visibility_mask(vis: str, n: int) -> np.ndarray:
    """``mask[i, j]``: row j is in E1m's ``visible(vis, n, i)``."""
    if vis == "full":
        return ~np.eye(n, dtype=bool)
    if vis == "causal":
        return np.tril(np.ones((n, n), dtype=bool), k=-1)
    raise E1pError(f"unknown visibility {vis}")


def basis(U: np.ndarray, S: np.ndarray, vis: str) -> np.ndarray:
    """``Q`` (n, k, d): per row i an orthonormal basis (zero rows where dropped) of the tangent at ``u_i``
    of the group-blind pulls over the rows i sees: the field at each of ``BASIS_BETAS`` and the mean of
    the min(``NN``, |V_i|) most similar rows, each less ``u_i``. An empty V_i gives no basis."""
    n = U.shape[0]
    mask = visibility_mask(vis, n)
    has = mask.any(axis=1)
    B = []
    for be in BASIS_BETAS:
        Z = np.where(mask, be * S, -np.inf)
        Z = Z - np.where(has, Z.max(axis=1), 0.0)[:, None]
        W = np.where(mask, np.exp(Z), 0.0)
        W = W / np.where(has, W.sum(axis=1), 1.0)[:, None]
        B.append(W @ U - U)
    order = np.argsort(-np.where(mask, S, -np.inf), axis=1, kind="stable")[:, :NN]
    k = np.minimum(mask.sum(axis=1), NN)
    take = np.arange(min(NN, n))[None, :] < k[:, None]
    nb = (U[order[:, :take.shape[1]]] * take[..., None]).sum(axis=1) / np.maximum(k, 1)[:, None]
    B.append(nb - U)
    B = np.stack(B, axis=1)                                          # (n, k, d)
    B = B - np.einsum("nkd,nd->nk", B, U)[..., None] * U[:, None, :]
    B[~has] = 0.0
    _, s, Vh = np.linalg.svd(B, full_matrices=False)
    keep = (s > SV_TOL * s[:, :1]) & (s > 0)
    return Vh * keep[..., None]


def residual(D: np.ndarray, Q: np.ndarray) -> Dict:
    """Each move less its projection on its row's basis; the removed share |proj|² / |d|² per move."""
    coef = np.einsum("cnd,nkd->cnk", D, Q)
    R = D - np.einsum("cnk,nkd->cnd", coef, Q)
    d2 = np.sum(D ** 2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(d2 >= TINY ** 2, np.sum(coef ** 2, axis=-1) / d2, np.nan)
    return {"R": R, "share": share}


def blk_of(U: np.ndarray, S: np.ndarray, D: np.ndarray) -> Dict:
    """What E1m's ``score_group`` reads: the Gram, the moves against every row, their norms."""
    return {"S": S, "M": np.einsum("cnd,md->cnm", D, U), "dn": np.linalg.norm(D, axis=-1)}


def score_group(fr: Dict, res: Dict, rows: np.ndarray, vis: str, pos: Optional[np.ndarray] = None,
                eps: float = EPS) -> Dict:
    """E1m's score on the residual moves (``reads`` / ``cover``), plus E1m's unprojected ``X``
    (``raw``) and the members' mean removed share per reading (``removed``)."""
    out = e1m.score_group(res["blk"], rows, vis, eps, pos)
    raw = e1m.score_group(fr["blk"], rows, vis, eps, pos)
    out["raw"] = {k: {be: c["X"] for be, c in cells.items()} for k, cells in raw["reads"].items()}
    out["removed"] = {f"{comp}:{rd}": _mean(res["share"][c, rows]) for c, (comp, rd) in enumerate(READS)}
    return out


def _mean(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


# ---------------------------------------------------------------- the pass

def read_passage(hs: np.ndarray, comps: Dict, ln: Dict, src: Path, step: int, passage: str,
                 cache: Dict, kept: np.ndarray) -> List[Dict]:
    """Every c2 group (c3x flagged) at L1–23 of one pass, read at block L, both views."""
    if not np.all(np.diff(kept) > 0):
        raise E1pError("kept offsets are not increasing: the causal view would not be 'rows before'")
    recs = []
    for L in LAYERS:
        fr = frame_moves(hs[L], comps[L], ln["w"][L], ln["b"][L], ln["eps"], kept)
        fr["blk"] = blk_of(fr["U"], fr["S"], fr["D"])
        res = {}
        for v in VIS:
            r = residual(fr["D"], basis(fr["U"], fr["S"], v))
            res[v] = {"blk": blk_of(fr["U"], fr["S"], r["R"]), "share": r["share"]}
        for g, (rows, inx) in e1.groups_at(src, step, passage, L, cache).items():
            recs.append({"layer": L, "group": g, "c3x": inx, "size": int(rows.size),
                         "blocks": {v: score_group(fr, res[v], rows, v, kept) for v in VIS}})
    return recs


def code_sha() -> str:
    """This file's blob hash at HEAD; refuses on an uncommitted change (each record names its producer)."""
    here = Path(__file__).resolve()
    git = ["git", "-C", str(here.parent)]
    if subprocess.run(git + ["status", "--porcelain", "--", here.name], capture_output=True, text=True,
                      check=True).stdout.strip():
        raise SystemExit(f"refusing: {here.name} has uncommitted changes; commit first")
    return subprocess.run(git + ["rev-parse", f"HEAD:tools/run/{here.name}"], capture_output=True,
                          text=True, check=True).stdout.strip()


def raw_mismatch(rec: Dict, e1m_rec: Dict) -> float:
    """The largest |E1p's unprojected X − E1m's stored X| over groups, views, readings and βs; refuses
    on a different group list."""
    a, b = rec["groups"], e1m_rec["groups"]
    if [(g["layer"], g["group"], g["c3x"]) for g in a] != [(g["layer"], g["group"], g["c3x"]) for g in b]:
        raise SystemExit(f"refusing: step {rec['step']} {rec['passage']}: the group list differs from E1m's")
    worst = 0.0
    for x, y in zip(a, b):
        for v in VIS:
            for rd, cells in x["blocks"][v]["raw"].items():
                for be, val in cells.items():
                    old = y["blocks"][v]["reads"][rd][be]["X"]
                    if np.isnan(val) != np.isnan(old):
                        return float("inf")
                    if not np.isnan(val):
                        worst = max(worst, abs(val - old))
    return worst


def check_first(recs: List[Dict], rec: Dict, e1m_dir: Path) -> None:
    """The rule's first-record checks: E1m reproduced; ``X_p`` finite for enough c3x groups and varying;
    the primary component's removed share under 1."""
    old = json.loads((e1m_dir / "records" / f"step{FIRST[0]}_{FIRST[1]}.json").read_text())
    worst = raw_mismatch(rec, old)
    if worst > REPRO_TOL:
        raise SystemExit(f"refusing: first record, unprojected X differs from E1m's by {worst:.2e}")
    px = [r for r in recs if r["c3x"]]
    for v in VIS:
        ok = [c for c in (r["blocks"][v]["reads"][PRIMARY[1]][str(PRIMARY[2])]["X"] for r in px) if np.isfinite(c)]
        if len(ok) < MIN_READABLE_SHARE * len(px):
            raise SystemExit(f"refusing: first record, {v}: {len(ok)} of {len(px)} c3x groups have an X_p")
        if len(set(np.round(ok, 12))) < 2:
            raise SystemExit(f"refusing: first record, {v}: X_p all equal")
        sh = [r["blocks"][v]["removed"][PRIMARY[1]] for r in px]
        if not np.nanmax(sh) < 1.0:
            raise SystemExit(f"refusing: first record, {v}: removed share reaches 1")
    print(f"first record: unprojected X reproduces E1m to {worst:.1e}", flush=True)


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
    meta = {"rule": "design-10.md \"E1p\"", "code": code, "labels": str(src), "e1m": str(a.e1m),
            "summary_sha256": hashlib.sha256((src / "summary.json").read_bytes()).hexdigest()[:16],
            "eps": EPS, "basis_betas": BASIS_BETAS, "nn": NN, "sv_tol": SV_TOL, "betas": BETAS,
            "reads": READS, "primary": PRIMARY}
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
            chk["last_block_rel"] = e1.check_last_block(hs, comps[23], model)
            kept = np.asarray(pr["kept"], dtype=int)
            recs = read_passage(hs, comps, ln, src, step, p, cache, kept)
            rec = {"step": step, "passage": p, "run": str(rd), "code": code, "pass": "cuda:float32",
                   "kept": int(kept.size), "checks": chk, "groups": recs}
            if (step, p) == FIRST and not first_done:
                check_first(recs, rec, a.e1m)
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

def reading(lab: str, e1m_lab: Optional[str], l1e: Optional[str]) -> str:
    """The rule's reading of one primary cell (c3x, causal): see design-10.md "E1p"."""
    lean = "(leans) " if lab.startswith("leans") else ""
    if lab in ("pulls together", "leans pulls"):
        tail = {"descends": "; holds the group against the rest (1e descends)",
                "ascends": "; pulls towards everything, the members most (1e ascends)"}
        hit = next((v for k, v in tail.items() if l1e and l1e.startswith(k)), "; no reading added (1e not descends / ascends)")
        return lean + "membership beyond LN1-cosine nearness, within the pairing's reach" + hit
    if lab in ("pushes apart", "leans pushes"):
        return lean + "pushes the members off equally near non-members beyond the group-blind pulls"
    if lab == "mixed":
        return ("E1m's cell was the group-blind part" if e1m_lab in ("pulls together", "leans pulls")
                else "no membership effect beyond the group-blind pulls")
    return "not separable here (too few)"


def removed_median(recs: Dict, step: int, band: str, vis: str, read: str) -> float:
    v = [g["blocks"][vis]["removed"][read] for p in PASSAGES for g in (recs.get((step, p)) or {"groups": []})["groups"]
         if g["layer"] in BANDS[band] and g["c3x"]]
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def report(a) -> int:
    recs = e1.load_records(a.out)
    codes = {r["code"] for r in recs.values()}
    if len(codes) != 1:
        raise SystemExit(f"refusing: records from {len(codes)} producers {sorted(codes)}")
    steps = sorted({s for s, _ in recs})
    missing = [(s, p) for s in steps for p in PASSAGES if (s, p) not in recs]
    if missing:
        raise SystemExit(f"refusing: {len(missing)} (step, passage) records missing, e.g. {missing[:3]}")
    old = e1.load_records(a.e1m)
    worst = max(raw_mismatch(r, old[k]) for k, r in recs.items())
    if worst > REPRO_TOL:
        raise SystemExit(f"refusing: unprojected X differs from E1m's records by {worst:.2e}")
    e1m_rows = json.loads((a.e1m / "report.json").read_text())["rows"]
    l1e = e1.labels_1e(a.u2_attn)
    res: Dict = {"code": codes.pop(), "steps": steps, "e1m_reproduced": worst, "rows": {}, "shares": {},
                 "removed": {}, "chance": {}, "reading": {}}
    for col in COLUMNS:
        for vis in VIS:
            for comp, rd in READS:
                for be in BETAS:
                    key = f"{col}|{vis}|{comp}:{rd}|{be}"
                    res["rows"][key] = {}
                    for s in steps:
                        for band in BANDS:
                            v = e1.cell_values(recs, s, band, col, vis, f"{comp}:{rd}", be, "X")
                            res["rows"][key][f"{s}|{band}"] = {"label": e1.sign_label(list(v.values())), "values": v}
                key = f"{col}|{vis}|{comp}:{rd}|3.5|X1"
                res["rows"][key] = {}
                for s in steps:
                    for band in BANDS:
                        v = e1.cell_values(recs, s, band, col, vis, f"{comp}:{rd}", 3.5, "X1")
                        res["rows"][key][f"{s}|{band}"] = {"label": e1.sign_label(list(v.values())), "values": v}
    for col in COLUMNS:
        for vis in VIS:
            prim = res["rows"][f"{col}|{vis}|{PRIMARY[1]}|{PRIMARY[2]}"]
            for s in steps:
                for band in BANDS:
                    c = prim[f"{s}|{band}"]
                    c["beta_robust"] = all(res["rows"][f"{col}|{vis}|{PRIMARY[1]}|{b}"][f"{s}|{band}"]["label"]
                                           == c["label"] for b in (1.6, 5.6))
                    c["step0_c2"] = res["rows"][f"c2|{vis}|{PRIMARY[1]}|{PRIMARY[2]}"].get(f"0|{band}", {}).get("label")
                    c["learned"] = c["label"] not in ("mixed", "too few") and c["label"] != c["step0_c2"]
                    if c["label"].startswith("leans"):
                        i = steps.index(s)
                        nb = [steps[j] for j in (i - 1, i + 1) if 0 <= j < len(steps)]
                        c["isolated"] = not any(prim[f"{t}|{band}"]["label"] == c["label"] for t in nb)
                    res["shares"][f"{col}|{vis}|{s}|{band}"] = e1m.shares(recs, s, band, col, vis, PRIMARY[1], PRIMARY[2])
            n_read = [int(np.isfinite(list(prim[f"{s}|{b}"]["values"].values())).sum()) for s in PRIMARY_STEPS
                      if s in steps for b in BANDS]
            res["chance"][f"{col}|{vis}"] = {
                "cells_read": int(sum(n >= e1.MIN_PASSAGES for n in n_read)),
                "expected_full_either": sum(e1.chance(n)["full_either"] for n in n_read if n >= e1.MIN_PASSAGES),
                "expected_lean_either": sum(e1.chance(n)["lean_either"] for n in n_read if n >= e1.MIN_PASSAGES)}
    for vis in VIS:
        for comp, rd in READS:
            for s in steps:
                for band in BANDS:
                    res["removed"][f"{vis}|{comp}:{rd}|{s}|{band}"] = removed_median(recs, s, band, vis, f"{comp}:{rd}")
    diffs = []
    for s in steps:
        for band in BANDS:
            k = f"{s}|{band}"
            lab = res["rows"][f"c3x|causal|{PRIMARY[1]}|{PRIMARY[2]}"][k]["label"]
            m_lab = (e1m_rows.get(f"c3x|causal|{PRIMARY[1]}|{PRIMARY[2]}", {}).get(k) or {}).get("label")
            res["reading"][k] = {"e1p": lab, "e1m": m_lab, "1e": l1e.get((s, band)),
                                 "reading": reading(lab, m_lab, l1e.get((s, band)))}
            if s in PRIMARY_STEPS and lab != m_lab:
                diffs.append(k)
    res["differs_from_e1m"] = diffs
    (a.out / "report.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = [f"E1p (design-10.md \"E1p\"), producer {res['code'][:10]}, {len(recs)} records; unprojected X "
             f"reproduces E1m to {worst:.1e}; primary causal attn:r1out β 3.5 on the residual moves; "
             "marks: β = β-robust, L = learned (c2's step 0 differs), i = isolated lean"]
    for col in COLUMNS:
        for vis in VIS:
            prim = res["rows"][f"{col}|{vis}|{PRIMARY[1]}|{PRIMARY[2]}"]
            lines.append(f"\n== {col} {vis}  (chance over primary cells: {res['chance'][f'{col}|{vis}']})")
            lines.append(f"{'step':>7} | " + " | ".join(f"{b:<44}" for b in BANDS))
            for s in steps:
                cells = []
                for b in BANDS:
                    c = prim[f"{s}|{b}"]
                    v = [x for x in c["values"].values() if np.isfinite(x)]
                    mk = ("β" if c.get("beta_robust") else "") + ("L" if c.get("learned") else "") + ("i" if c.get("isolated") else "")
                    sh = res["shares"][f"{col}|{vis}|{s}|{b}"]
                    cells.append(f"{c['label']:<14}{mk:<3} {np.mean(v) if v else float('nan'):+.4f} "
                                 f"rd{sh['readable']}/{sh['groups']} |X|{sh['median_abs_X']:.3f}")
                lines.append(f"{s:>7} | " + " | ".join(f"{x:<44}" for x in cells))
    lines.append("\n== reading per primary cell (c3x causal, steps 64-143000): E1p | E1m | 1e -> reading")
    for s in steps:
        if s in PRIMARY_STEPS:
            for b in BANDS:
                r = res["reading"][f"{s}|{b}"]
                lines.append(f"{s:>7} {b:<6} {r['e1p']:<15}| {str(r['e1m']):<15}| {str(r['1e']):<10} -> {r['reading']}")
    lines.append(f"\n== primary cells where E1p differs from E1m: {len(diffs)} — " + ", ".join(diffs))
    lines.append("\n== removed share (median over c3x group-layer records; causal | full), attn:r1out and mlpx:r1out")
    for comp in (PRIMARY[1], "mlpx:r1out"):
        for b in BANDS:
            lines.append(f"{comp:<11} {b:<6} " + " ".join(
                f"{res['removed'][f'causal|{comp}|{s}|{b}']:.2f}/{res['removed'][f'full|{comp}|{s}|{b}']:.2f}"
                for s in steps if s in PRIMARY_STEPS))
    lines.append("\n== beside (c3x): label per row, steps 64-143000 x bands")
    for vis in VIS:
        for comp, rd in READS:
            for be in list(BETAS) + ["3.5|X1"]:
                labs = [res["rows"][f"c3x|{vis}|{comp}:{rd}|{be}"][f"{s}|{b}"]["label"] for s in steps
                        if s in PRIMARY_STEPS for b in BANDS]
                lines.append(f"{vis:6} {comp}:{rd:<10} β{be:<7} " + ", ".join(f"{k} {v}" for k, v in Counter(labs).most_common()))
    lines.append("\n== compact (c3x, steps 64-143000): P / p pulls / leans, X / x pushes / leans, . mixed, - too few")
    for vis in VIS:
        for comp in (PRIMARY[1], "mlpx:r1out"):
            for b in BANDS:
                labs = [res["rows"][f"c3x|{vis}|{comp}|{PRIMARY[2]}"][f"{s}|{b}"]["label"] for s in steps if s in PRIMARY_STEPS]
                old_l = [(e1m_rows.get(f"c3x|{vis}|{comp}|{PRIMARY[2]}", {}).get(f"{s}|{b}") or {}).get("label", "too few")
                         for s in steps if s in PRIMARY_STEPS]
                lines.append(f"{vis:6} {comp:<11} {b:<6} E1p {' '.join(e1m.L_ABBR[x] for x in labs)}   "
                             f"E1m {' '.join(e1m.L_ABBR[x] for x in old_l)}")
    (a.out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--labels", type=Path, required=True, help="the R9 label source (c3x, c2)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--e1m", type=Path, default=DATA / "p10" / "e1m_match_2026-10-09", help="E1m's output dir")
    r.add_argument("--steps", type=int, nargs="*", default=None)
    r.add_argument("--first-only", action="store_true", help="the first record only, then stop")
    q = sub.add_parser("report")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--e1m", type=Path, default=DATA / "p10" / "e1m_match_2026-10-09")
    q.add_argument("--u2-attn", type=Path, default=None, help="1e's attention arm (its v1 labels beside)")
    a = ap.parse_args(argv)
    return {"run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
