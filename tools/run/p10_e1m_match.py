"""E1m: E1's membership-against-nearness re-run (`p10_cluster_function/design-10.md` "E1m").

E1 (`p10_e1_energy.py`) asked whether attention's token-specific move ``d_i`` points at a c3x group's
members more than at as many random other tokens, and `/challenge-pr` on #176 showed members are
nearer each other than the nearest non-members in 95-98 % of groups, so that cannot separate
membership from nearness. Here, per member i, each fellow member f (a member of the group other than
i, visible to i) is paired with a **non-member of the same similarity to i** (``|u_i·u_n − u_i·u_f| ≤
EPS``, greedy, nearest first, without replacement). Attention's weight on a token is a function of
that similarity alone, so a pull towards near tokens gives the paired set the same force as the
members': ``X_g = mean_i [cos(d_i, field of i's matched fellows) − cos(d_i, field of the paired
non-members)]``. A fellow with no non-member of its similarity is unmatched and left out (counted:
where members are nearer than every non-member, nothing can be matched, and that is a result).
Two visibilities: ``full`` (E1's) and ``causal`` (rows before i: 1e U2's field, what a causal head can
see; primary).

Same frame, moves and fields as E1 (`block_frame`, unit LN1 rows, ``r1out``); every cosine a ratio of
scalars from the Gram. Tier 1: exploratory, unregistered.
    python tools/run/p10_e1m_match.py probe --labels <R9 source> --step 512 --passage wiki_paragraph
    python tools/run/p10_e1m_match.py run --labels <R9 source> --out <dir> [--steps 512 ...]
    python tools/run/p10_e1m_match.py report --out <dir> --e1 <E1 dir> --u2-attn <1e attention arm dir>
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from tools.run import p10_e1_energy as e1
from tools.run import p10_label_source as ls

STEPS, PRIMARY_STEPS, PASSAGES = e1.STEPS, e1.PRIMARY_STEPS, e1.PASSAGES
LAYERS, BANDS, BANDS_1E, READS, TINY = e1.LAYERS, e1.BANDS, e1.BANDS_1E, e1.READS, e1.TINY
BETAS = e1.BETAS                                           # 3.5 first (primary), then 1.6, 5.6, 0
VIS = ("causal", "full")
PRIMARY = ("causal", "attn:r1out", 3.5)
COLUMNS = ("c3x", "c2")
EPS = 0.02                                                 # similarity tolerance of a pair (rule)
MIN_MEMBERS = 3                                            # members with a pair for a group's X (rule)
MIN_RECORDS, MIN_PASSAGES = e1.MIN_RECORDS, e1.MIN_PASSAGES
FIRST = e1.FIRST


class E1mError(RuntimeError):
    pass


# ---------------------------------------------------------------- the pairing and the scores (pure)

def match_member(s_f: np.ndarray, s_n: np.ndarray, eps: float = EPS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pair fellows to non-members of the same similarity: fellows from the most similar down, each takes
    the unused non-member whose similarity is nearest its own, if within ``eps`` (ties to the lower
    index). Returns the paired fellow indices and their non-member indices (into ``s_f`` / ``s_n``).
    """
    order = np.argsort(-s_f, kind="stable")
    by = np.argsort(s_n, kind="stable")
    vals, ids = list(s_n[by]), list(by)
    fi, ni = [], []
    for f in order:
        if not vals:
            break
        p = bisect.bisect_left(vals, s_f[f])
        q = min((c for c in (p - 1, p) if 0 <= c < len(vals)), key=lambda c: (abs(vals[c] - s_f[f]), c))
        if abs(vals[q] - s_f[f]) <= eps:
            fi.append(int(f))
            ni.append(int(ids[q]))
            vals.pop(q)
            ids.pop(q)
    return np.asarray(fi, dtype=int), np.asarray(ni, dtype=int)


def pair_cosines(S: np.ndarray, M: np.ndarray, dn: np.ndarray, i: int, idx: np.ndarray,
                 betas: Sequence[float]) -> np.ndarray:
    """``a[c, b] = cos(d^c_i, P⊥_{u_i} Σ_{j∈idx} softmax_j(β_b u_i·u_j) u_j)``; NaN as E1's (``|d_i|`` or
    ``|g_i|`` under ``TINY``) and for an empty ``idx``."""
    C, B = M.shape[0], len(betas)
    if len(idx) == 0:
        return np.full((C, B), np.nan)
    s = S[i, idx]
    Z = np.asarray(betas, dtype=float)[:, None] * s[None, :]
    w = np.exp(Z - Z.max(axis=1, keepdims=True))
    wsum = w.sum(axis=1)
    g2 = np.einsum("bj,jk,bk->b", w, S[np.ix_(idx, idx)], w) - (w @ s) ** 2
    gn = np.sqrt(np.maximum(g2, 0.0))
    num = np.einsum("bk,ck->cb", w, M[:, i, idx])
    with np.errstate(invalid="ignore", divide="ignore"):
        a = num / (dn[:, i][:, None] * gn[None, :])
    bad = (dn[:, i] < TINY)[:, None] | ((gn / np.maximum(wsum, 1e-300)) < TINY)[None, :]
    return np.where(bad, np.nan, a)


def member_pairs(S: np.ndarray, i: int, fellows: np.ndarray, others: np.ndarray, eps: float = EPS):
    """The (fellow rows, non-member rows) paired for member ``i`` among the rows ``fellows`` / ``others``."""
    if len(fellows) == 0 or len(others) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    fi, ni = match_member(S[i, fellows], S[i, others], eps)
    return fellows[fi], others[ni]


def visible(vis: str, n: int, i: int) -> np.ndarray:
    """Rows member ``i`` sees: every other row (``full``) or the rows before it (``causal``)."""
    if vis == "full":
        return np.delete(np.arange(n), i)
    if vis == "causal":
        return np.arange(i)
    raise E1mError(f"unknown visibility {vis}")


def score_group(blk: Dict, rows: np.ndarray, vis: str, eps: float = EPS) -> Dict:
    """
    One group at one block, one visibility: per reading and β, ``A_all`` (all visible fellows),
    ``A_obs`` / ``A_cmp`` (the matched fellows / their paired non-members, over the members with
    both) and ``X = A_obs − A_cmp`` (NaN under ``MIN_MEMBERS`` members); plus the pairing's coverage.
    """
    S, M, dn = blk["S"], blk["M"], blk["dn"]
    n = S.shape[0]
    inside = np.zeros(n, dtype=bool)
    inside[rows] = True
    C, B = M.shape[0], len(BETAS)
    A_all, A_obs, A_cmp = [], [], []
    rel = paired_rel = n_paired_members = 0
    ds = []
    for i in rows:
        vis_rows = visible(vis, n, int(i))
        fell, oth = vis_rows[inside[vis_rows]], vis_rows[~inside[vis_rows]]
        rel += len(fell)
        if len(fell) == 0:
            continue
        pf, pn = member_pairs(S, int(i), fell, oth, eps)
        A_all.append(pair_cosines(S, M, dn, int(i), fell, BETAS))
        if len(pf):
            n_paired_members += 1
            paired_rel += len(pf)
            ds.extend(np.abs(S[i, pf] - S[i, pn]).tolist())
            A_obs.append(pair_cosines(S, M, dn, int(i), pf, BETAS))
            A_cmp.append(pair_cosines(S, M, dn, int(i), pn, BETAS))
    out: Dict = {"reads": {}, "cover": {
        "members": int(rows.size), "members_visible": int(len(A_all)), "members_paired": n_paired_members,
        "relations": int(rel), "paired": int(paired_rel),
        "mean_abs_ds": float(np.mean(ds)) if ds else float("nan")}}
    all_ = np.stack(A_all) if A_all else np.full((0, C, B), np.nan)
    obs = np.stack(A_obs) if A_obs else np.full((0, C, B), np.nan)
    cmp_ = np.stack(A_cmp) if A_cmp else np.full((0, C, B), np.nan)
    both = np.isfinite(obs) & np.isfinite(cmp_)
    for c, (comp, rd) in enumerate(READS):
        cell = out["reads"].setdefault(f"{comp}:{rd}", {})
        for b, be in enumerate(BETAS):
            k = int(both[:, c, b].sum())
            ao, ac = obs[:, c, b][both[:, c, b]], cmp_[:, c, b][both[:, c, b]]
            cell[str(be)] = {
                "A_all": _mean(all_[:, c, b]), "A_obs": _mean(ao), "A_cmp": _mean(ac), "members": k,
                "X": (float(ao.mean() - ac.mean()) if k >= MIN_MEMBERS else float("nan"))}
    return out


def _mean(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


# ---------------------------------------------------------------- the pass

def read_passage(hs: np.ndarray, comps: Dict, ln: Dict, src: Path, step: int, passage: str,
                 cache: Dict, kept: np.ndarray) -> List[Dict]:
    """Every c2 group (c3x flagged) at L1–23 of one pass, read at block L (departure), both visibilities."""
    if not np.all(np.diff(kept) > 0):
        raise E1mError("kept offsets are not increasing: the causal field would not be 'rows before'")
    recs = []
    for L in LAYERS:
        blk = e1.block_frame(hs[L], comps[L], ln["w"][L], ln["b"][L], ln["eps"], kept)
        for g, (rows, inx) in e1.groups_at(src, step, passage, L, cache).items():
            recs.append({"layer": L, "group": g, "c3x": inx, "size": int(rows.size),
                         "blocks": {v: score_group(blk, rows, v) for v in VIS}})
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


def check_populated(recs: List[Dict]) -> None:
    """The rule's first-record check: the pairing covers and X is finite and varies."""
    px = [r for r in recs if r["c3x"]]
    if not px:
        raise SystemExit("refusing: first record has no c3x group")
    for v in VIS:
        cells = [r["blocks"][v]["reads"][PRIMARY[1]][str(PRIMARY[2])] for r in px]
        ok = [c["X"] for c in cells if np.isfinite(c["X"])]
        if len(ok) < MIN_READABLE_SHARE * len(cells):
            raise SystemExit(f"refusing: first record, {v}: {len(ok)} of {len(cells)} c3x groups have an X "
                             f"(a matched pair for ≥ {MIN_MEMBERS} members); under {MIN_READABLE_SHARE:.0%}")
        if len(set(np.round(ok, 12))) < 2:
            raise SystemExit(f"refusing: first record, {v}: X all equal")


MIN_READABLE_SHARE = 0.25                                  # of c3x groups at the first record (rule)


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
    meta = {"rule": "design-10.md \"E1m\"", "code": code, "labels": str(src),
            "summary_sha256": hashlib.sha256((src / "summary.json").read_bytes()).hexdigest()[:16],
            "eps": EPS, "min_members": MIN_MEMBERS, "betas": BETAS, "reads": READS, "primary": PRIMARY}
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


# ---------------------------------------------------------------- the probe (geometry only: no X)

def probe(a) -> int:
    """How much of the fellow relations the pairing can match, per band and visibility, at several
    tolerances. Reads the geometry (similarities) only: no move, no cosine."""
    import torch
    from transformers import AutoTokenizer
    from core.models import load_model
    from p1e_energy_field import u2_attn as ua
    from p1e_energy_field import u2_block as ub
    src = Path(a.labels)
    d = ls.load_step(src, f"step{a.step}")
    pr = d["prompts"][a.passage]
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    model, _ = load_model(f"pythia-410m-step{a.step}")
    hk, ln = ua.Hooked(model), ua.ln_from(model)
    rd = Path(pr["stage0_run"])
    ids = torch.tensor([ua.token_ids(tok, rd)], device=next(model.parameters()).device)
    hs, _ = hk.run(ids, tuple(range(24)))
    kept = np.asarray(pr["kept"], dtype=int)
    cache: Dict = {}
    print(f"probe step {a.step} {a.passage}: {kept.size} kept rows; share of fellow relations matched / "
          "members with a pair / c3x groups with ≥ MIN_MEMBERS paired members (c3x groups only)")
    for vis in VIS:
        for band, layers in BANDS.items():
            line = [f"{vis:6} {band:6}"]
            for eps in a.eps:
                rel = pair = mem = mem_p = grp = grp_ok = 0
                for L in layers:
                    U = ub.unit_rows(hs[L][kept].astype(np.float64), ln["w"][L], ln["b"][L], ln["eps"])
                    S = U @ U.T
                    for g, (rows, inx) in e1.groups_at(src, a.step, a.passage, L, cache).items():
                        if not inx:
                            continue
                        inside = np.zeros(kept.size, dtype=bool)
                        inside[rows] = True
                        paired_members = 0
                        grp += 1
                        for i in rows:
                            v = visible(vis, kept.size, int(i))
                            fell, oth = v[inside[v]], v[~inside[v]]
                            rel += len(fell)
                            if len(fell):
                                mem += 1
                                pf, _ = member_pairs(S, int(i), fell, oth, eps)
                                pair += len(pf)
                                mem_p += bool(len(pf))
                                paired_members += bool(len(pf))
                        grp_ok += paired_members >= MIN_MEMBERS
                line.append(f"eps {eps}: rel {pair}/{rel} = {pair / max(rel, 1):.2f}, members "
                            f"{mem_p}/{mem} = {mem_p / max(mem, 1):.2f}, groups {grp_ok}/{grp}")
            print(" | ".join(line), flush=True)
    hk.close()
    return 0


# ---------------------------------------------------------------- the reading

def shares(recs: Dict, step: int, band: str, column: str, vis: str, read: str, beta: float) -> Dict:
    """Per (step, band): readable groups, the share with ``X`` > 0, the median ``|X|``, and the pairing's
    coverage (relations paired, members paired) over the column's groups."""
    xs, rel, pair, mem, memp = [], 0, 0, 0, 0
    groups = 0
    for p in PASSAGES:
        for g in (recs.get((step, p)) or {"groups": []})["groups"]:
            if g["layer"] in BANDS[band] and (column == "c2" or g["c3x"]):
                groups += 1
                b = g["blocks"][vis]
                cv = b["cover"]
                rel += cv["relations"]
                pair += cv["paired"]
                mem += cv["members_visible"]
                memp += cv["members_paired"]
                x = b["reads"][read][str(beta)]["X"]
                if np.isfinite(x):
                    xs.append(x)
    xs = np.asarray(xs)
    return {"groups": groups, "readable": int(xs.size), "share_pos": float((xs > 0).mean()) if xs.size else float("nan"),
            "median_abs_X": float(np.median(np.abs(xs))) if xs.size else float("nan"),
            "paired_relations": pair / rel if rel else float("nan"),
            "paired_members": memp / mem if mem else float("nan")}


def reading(c_lab: str, f_lab: str, e1_lab: Optional[str], l1e: Optional[str]) -> str:
    """The rule's reading of one primary cell (c3x): see design-10.md "E1m"."""
    if c_lab == "too few" and f_lab == "too few":
        return "not separable here (too few paired)"
    if c_lab in ("pulls together", "leans pulls"):
        tail = {"descends": "; holds the group against the rest (1e descends)",
                "ascends": "; pulls towards everything, the members most (1e ascends)"}
        hit = next((v for k, v in tail.items() if l1e and l1e.startswith(k)), "; 1e's window is mixed: no reading added")
        return "members beyond equally near non-members" + hit
    if c_lab in ("pushes apart", "leans pushes"):
        return "pushes the members apart beyond equally near non-members"
    if c_lab == "mixed":
        return ("E1's pull was nearness" if e1_lab in ("pulls together", "leans pulls") else "no membership effect")
    return "causal row unreadable" + ("" if f_lab == "too few" else f"; full reads {f_lab}")


def report(a) -> int:
    recs = e1.load_records(a.out)
    codes = {r["code"] for r in recs.values()}
    if len(codes) != 1:
        raise SystemExit(f"refusing: records from {len(codes)} producers {sorted(codes)}")
    steps = sorted({s for s, _ in recs})
    missing = [(s, p) for s in steps for p in PASSAGES if (s, p) not in recs]
    if missing:
        raise SystemExit(f"refusing: {len(missing)} (step, passage) records missing, e.g. {missing[:3]}; "
                         "a missing passage would be read as unreadable")
    l1e = e1.labels_1e(a.u2_attn)
    e1_rep = json.loads((a.e1 / "report.json").read_text())["rows"] if a.e1 else {}
    res: Dict = {"code": codes.pop(), "steps": steps, "rows": {}, "shares": {}, "chance": {}, "reading": {}}
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
                    res["shares"][f"{col}|{vis}|{s}|{band}"] = shares(recs, s, band, col, vis, PRIMARY[1], PRIMARY[2])
            n_read = [int(np.isfinite(list(prim[f"{s}|{b}"]["values"].values())).sum()) for s in PRIMARY_STEPS
                      if s in steps for b in BANDS]
            res["chance"][f"{col}|{vis}"] = {
                "cells_read": int(sum(n >= MIN_PASSAGES for n in n_read)),
                "expected_full_either": sum(e1.chance(n)["full_either"] for n in n_read if n >= MIN_PASSAGES),
                "expected_lean_either": sum(e1.chance(n)["lean_either"] for n in n_read if n >= MIN_PASSAGES)}
    for s in steps:
        for band in BANDS:
            k = f"{s}|{band}"
            lab = lambda vis: res["rows"][f"c3x|{vis}|{PRIMARY[1]}|{PRIMARY[2]}"][k]["label"]
            e1_lab = (e1_rep.get("c3x|dep|attn:r1out|3.5|X", {}).get(k) or {}).get("label")
            res["reading"][k] = {"causal": lab("causal"), "full": lab("full"), "e1": e1_lab,
                                 "1e": l1e.get((s, band)),
                                 "reading": reading(lab("causal"), lab("full"), e1_lab, l1e.get((s, band)))}
    (a.out / "report.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = [f"E1m (design-10.md \"E1m\"), producer {res['code'][:10]}, {len(recs)} records; primary causal "
             "attn:r1out β 3.5 (matched); marks: β = β-robust, L = learned (c2's step 0 differs), i = isolated lean"]
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
                                 f"rd{sh['readable']}/{sh['groups']} pr{sh['paired_relations']:.2f}")
                lines.append(f"{s:>7} | " + " | ".join(f"{x:<44}" for x in cells))
    lines.append("\n== reading per primary cell (c3x, steps 64-143000): causal | full | E1 | 1e -> reading")
    for s in steps:
        if s in PRIMARY_STEPS:
            for b in BANDS:
                r = res["reading"][f"{s}|{b}"]
                lines.append(f"{s:>7} {b:<6} {r['causal']:<15}| {r['full']:<15}| {str(r['e1']):<15}| "
                             f"{str(r['1e']):<10} -> {r['reading']}")
    lines.append("\n== beside (c3x): label per row, steps 64-143000 x bands")
    from collections import Counter
    for vis in VIS:
        for comp, rd in READS:
            for be in BETAS:
                labs = [res["rows"][f"c3x|{vis}|{comp}:{rd}|{be}"][f"{s}|{b}"]["label"] for s in steps
                        if s in PRIMARY_STEPS for b in BANDS]
                lines.append(f"{vis:6} {comp}:{rd:<10} β{be:<4} " + ", ".join(f"{k} {v}" for k, v in Counter(labs).most_common()))
    (a.out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--step", type=int, default=FIRST[0])
    p.add_argument("--passage", default=FIRST[1])
    p.add_argument("--eps", type=float, nargs="*", default=[0.01, 0.02, 0.05])
    r = sub.add_parser("run")
    r.add_argument("--labels", type=Path, required=True, help="the R9 label source (c3x, c2)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", type=int, nargs="*", default=None)
    r.add_argument("--first-only", action="store_true", help="the first record only, then stop")
    q = sub.add_parser("report")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--e1", type=Path, default=None, help="E1's output dir (its labels beside)")
    q.add_argument("--u2-attn", type=Path, default=None, help="1e's attention arm (its v1 labels beside)")
    a = ap.parse_args(argv)
    return {"probe": probe, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
