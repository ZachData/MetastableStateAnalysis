"""
p1d_cluster_ensemble/move_text.py — unit 1 of the Blocked 11⁗ programme:
move the passage and see whether its groups move with it
(`design-1d.md` "Unit 1: move the text"; rules fixed there before any run).

Per passage, step and condition, one forward pass of
``preamble[:P] + join + passage`` (P = 0: the passage alone, no join), hidden
states only. The preamble is the first P tokens of a long prompt's
continuation (`long_prompts.py`: the long text's tokens after its v1
prefix), never the passage's own. Then, per layer L1–24, frame (centred,
raw) and ``min_cluster_size`` (2, 4):

- **passage cloud**: the passage's kept tokens at their passage offsets (the
  same set at every P), level-set HDBSCAN (`admit.level_set_hdbscan`, Phase
  1's float64 cosine route); no null draws;
- **(a)** each P = 0 group's best-Jaccard group at every P > 0;
- **(b)** each kept token's cosine between its state at P and at P = 0, in
  the frame (per group: the median over its members);
- **(c)** the whole-sequence cloud (preamble + join + passage): the group
  holding the earliest kept position, and where its members sit;
- **noise floor**: each P = 0 group's best-match Jaccards over
  ``N_SUBSAMPLES`` subsamples of ``SUBSAMPLE_FRACTION`` of the kept tokens
  (Hennig 2007: the group restricted to the subsample against the
  subsample's groups, the frame recomputed on the subsample); ``J0`` is
  their ``FLOOR_PERCENTILE``-th percentile; median < ``STABLE_MEDIAN`` is
  unstable and not classified;
- **class** per stable P = 0 group (primary join): *moves* (best Jaccard >=
  J0 at every P > 0 under every preamble), *preamble-dependent* (under
  some), *opening-bound* (under none, holds passage offsets <
  ``OPENING_OFFSETS``, and (c) holds), *context-bound* (none, otherwise).
  The ``\\n\\n`` join's class is written beside it.

Token rules (`design-1d.md` "Token rules"), and how they are read here:

- T1: position 0 is in no cloud; the passage cloud also drops passage
  offset 0 at every P (at P = 0 it is position 0).
- T2: a passage offset whose norm exceeds ``MASSIVE_RATIO`` x its layer's
  median (over the run's positions) at any of L2–20 in **any condition of
  the same passage and step** is dropped from every condition of it. The
  union is not taken across steps, so the step-0 first check does not read
  a trained run; the tokens a cross-step union would add are written beside
  (``massive_other_step``, filled by `check`). The whole-sequence cloud
  drops its own run's massive positions and the passage's union.
- T3: the passage cloud keeps each string's first occurrence **within the
  passage**, so its token set does not change with the preamble; the
  whole-sequence cloud keeps first occurrences over the whole sequence.
- ``--kept-from`` (v1 only) replaces the passage's token set with another
  comparison's (`arch_null`'s ``token_sets.json``; `design-1d.md` "The
  candidates"), refusing if it keeps an offset this run's own rules drop;
  the whole-sequence cloud also drops that comparison's positions.
- ``--stage0-index`` (v1 only; the Phase 10 re-read, `design-10.md`) checks
  each P = 0 pass against that (step, passage)'s Stage 0 ``activations.npz``
  (`p0_match`, ``P0_MATCH_TOL``) right after it runs; a passage over the
  tolerance refuses and writes no record. The match goes into the record.

**Operational rules for the first checks** (placed 2026-10-02, before any
forward pass; `design-1d.md` states them in words):

- The *opening group* of a (passage, layer, frame, size) record is the P = 0
  group holding the earliest kept passage offset, if that group holds an
  offset < ``OPENING_OFFSETS``; otherwise the record has none.
- (c) *holds* in a record when, under every preamble, at every P > 0 with the
  primary join, the whole-sequence cloud's earliest kept position is in a
  group (not HDBSCAN noise).
- **Step-0 first check** (v1, centred, size 2): per passage, the opening
  group's modal class over L1–24 (records with no opening group, or an
  unstable one, are counted as such). **Pass**: opening-bound is the modal
  class in at least ``MOST`` (4) of the 7 passages. **Stop** ("the opening
  is not position"): *moves* is, in at least 4. Otherwise: neither, and
  the user reads the table before any trained cell.
- **Designed-content check**: `designed_prompts.py`'s rule, frozen with the
  texts.
- `run` refuses a trained v1 step unless both checks passed in the same
  output directory (``first_checks.json``), or ``--override-first-checks
  REASON`` is given, which is written into every record.

Activations are kept (``--act-out``) only for P in ``KEEP_P``, the first
preamble and the primary join. Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import designed_prompts as dp
from .admit import level_set_hdbscan
from .gaussian_null import first_occurrences, frame_vectors

#: The 7 v1 passages (not `repeated_tokens`, not `short_heterogeneous`).
V1_PASSAGES = ("wiki_paragraph", "sullivan_ballou", "paper_excerpt", "homer_iliad",
               "hdbscan_code", "camus_letranger", "latex_monograph")
#: Long prompts whose continuation has >= 1000 tokens (`sullivan_ballou`'s 550 does not).
PREAMBLE_SOURCES = ("wiki_paragraph", "hdbscan_code", "latex_monograph")
P_VALUES = (50, 300, 1000)
#: `core.models.extract_activations`' cap, which every v1 run read under.
V1_MAX_TOKENS = 512
JOINS = ("eod", "nl2")
PRIMARY_JOIN = "eod"
LAYERS = tuple(range(1, 25))
FRAMES = ("centred", "raw")
MIN_CLUSTER_SIZES = (2, 4)
#: T2, placed (`design-1d.md`).
MASSIVE_RATIO = 10.0
MASSIVE_LAYERS = tuple(range(2, 21))
#: Passage offsets that count as the opening.
OPENING_OFFSETS = 8
#: Noise floor, placed (`design-1d.md`).
N_SUBSAMPLES = 50
SUBSAMPLE_FRACTION = 0.8
FLOOR_PERCENTILE = 10
STABLE_MEDIAN = 0.5
#: Activations written to disk for these P only (first preamble, primary join).
KEEP_P = (0, 1000)
#: "Most of the 7 passages", placed.
MOST = 4
#: Unit 1 ran steps 0 and 143000; the Phase 10 re-read (`p10_cluster_function/design-10.md`
#: "Checkpoints: the cost") needs "moves" at every distinct Stage 0 step (step 1 = step 0's weights).
P10_STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
MODELS = {f"step{s}": f"pythia-410m-step{s}" for s in P10_STEPS}
#: The P = 0 cloud against Stage 0's stored `activations.npz` (`design-10.md` "The check that
#: joins the two passes"): max elementwise difference of the unit rows and max relative norm
#: difference, over L0-24 and every position; above it the (step, passage) refuses. Placed.
P0_MATCH_TOL = 1e-5
ACT_ROOT = Path("/run/media/system/HDD_1TB/mets_data")
CLASSES = ("moves", "preamble-dependent", "opening-bound", "context-bound")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def passage_inputs(which: str, tokenizer) -> Dict[str, Dict]:
    """``{key: {"ids", "labels"}}`` for ``v1`` or ``designed`` (labels None for v1)."""
    if which == "v1":
        from core.config import PROMPTS
        from core.holdout import V1_PROMPT_KEYS
        assert set(V1_PASSAGES) <= set(V1_PROMPT_KEYS)
        out = {}
        for k in V1_PASSAGES:
            # The v1 runs read the first V1_MAX_TOKENS (`extract_activations`
            # truncates); `homer_iliad` is longer. The passage is what they read.
            ids = tokenizer(PROMPTS[k])["input_ids"]
            out[k] = {"ids": list(ids[:V1_MAX_TOKENS]), "labels": None, "n_text_tokens": len(ids)}
        return out
    if which == "designed":
        return {k: dict(zip(("ids", "labels"), dp.token_labels(t, s, tokenizer)))
                for k, (t, s) in dp.prompts().items()}
    raise ValueError(which)


def continuations(tokenizer) -> Dict[str, List[int]]:
    """Each preamble source's continuation tokens: the long prompt after its v1 prefix."""
    from .long_prompts import load, load_provenance
    texts, prov = load(), load_provenance()["prompts"]
    out = {}
    for src in PREAMBLE_SOURCES:
        lk = f"{src}_long"
        ids = tokenizer(texts[lk])["input_ids"]
        n_v1 = prov[lk]["n_v1_tokens"]
        if len(ids) != prov[lk]["n_tokens"]:
            raise ValueError(f"{lk}: {len(ids)} tokens, provenance says {prov[lk]['n_tokens']}")
        out[src] = list(ids[n_v1:])
        if len(out[src]) < max(P_VALUES):
            raise ValueError(f"{src}: continuation {len(out[src])} < {max(P_VALUES)} tokens")
    return out


def join_ids(tokenizer) -> Dict[str, List[int]]:
    j = {"eod": [int(tokenizer.eos_token_id)], "nl2": list(tokenizer("\n\n")["input_ids"])}
    if len(j["nl2"]) != 1:
        raise ValueError(f"'\\n\\n' is {j['nl2']}, not one token")
    return j


def conditions(passage: str, ids: Sequence[int], conts: Dict[str, List[int]],
               joins: Dict[str, List[int]], p0_only: bool = False) -> List[Dict]:
    """Every (preamble, P, join) for one passage; the passage's own continuation is skipped."""
    out = [{"id": "P0", "preamble": None, "P": 0, "join": None, "ids": list(ids), "start": 0}]
    if p0_only:
        return out
    for src in PREAMBLE_SOURCES:
        if src == passage:
            continue
        for P in P_VALUES:
            for j in JOINS:
                pre = conts[src][:P] + joins[j]
                out.append({"id": f"{src}|{P}|{j}", "preamble": src, "P": P, "join": j,
                            "ids": pre + list(ids), "start": len(pre)})
    return out


# ---------------------------------------------------------------------------
# Token rules
# ---------------------------------------------------------------------------

def massive_positions(norms: np.ndarray, ratio: float = MASSIVE_RATIO,
                      layers: Sequence[int] = MASSIVE_LAYERS) -> Dict[int, Tuple[float, int]]:
    """``{position: (max ratio, its layer)}`` over ``layers`` of a ``(n_layers, n)`` norm array."""
    out = {}
    for L in layers:
        r = norms[L] / np.median(norms[L])
        for p in np.flatnonzero(r > ratio):
            if p not in out or r[p] > out[p][0]:
                out[int(p)] = (float(r[p]), int(L))
    return out


def kept_offsets(tokens: Sequence[str], massive: Sequence[int]) -> np.ndarray:
    """T1 (offset 0), T2 (``massive`` offsets) and T3 within the passage."""
    first = first_occurrences(tokens)
    bad = set(int(m) for m in massive) | {0}
    return np.asarray([o for o in first if o not in bad], dtype=int)


def forced_kept(own_kept: Sequence[int], given: Sequence[int], passage: str) -> np.ndarray:
    """
    ``--kept-from``: another comparison's token set (`design-1d.md` "The
    candidates"), refused if it keeps an offset this run's own T1–T3 drop.
    """
    extra = sorted(set(int(o) for o in given) - set(int(o) for o in own_kept))
    if extra:
        raise SystemExit(f"refusing: {passage}: the given token set keeps offsets {extra[:5]} "
                         f"that this run's own token rules drop")
    return np.asarray(sorted(int(o) for o in given), dtype=int)


def whole_kept(tokens: Sequence[str], start: int, own_massive: Sequence[int],
               passage_dropped: Sequence[int]) -> np.ndarray:
    """Whole-sequence positions after T1, T2 (own run + the passage's union) and T3."""
    first = first_occurrences(tokens)
    bad = set(int(m) for m in own_massive) | {0} | {start + int(o) for o in passage_dropped}
    return np.asarray([p for p in first if p not in bad], dtype=int)


def p0_match(H: np.ndarray, run_dir: Path, tol: float = P0_MATCH_TOL) -> Dict:
    """
    The P = 0 hidden states ``H`` (25, n, d) against a Stage 0 run's stored
    ``activations.npz`` (unit rows ``activations`` and ``norms``): the largest
    elementwise difference of the unit rows and the largest relative norm
    difference, and whether both are within ``tol``.
    """
    a = np.load(Path(run_dir) / "activations.npz")
    U, N = a["activations"], a["norms"]
    out = {"run_dir": str(run_dir), "tol": tol}
    if U.shape != H.shape:
        return out | {"ok": False, "shape": [list(H.shape), list(U.shape)]}
    n = np.linalg.norm(H.astype(np.float64), axis=-1)
    d = float(np.max(np.abs(H / n[..., None] - U)))
    r = float(np.max(np.abs(n / N - 1.0)))
    return out | {"direction_max": d, "rel_norm_max": r, "ok": bool(d <= tol and r <= tol)}


def stage0_runs(index: Path) -> Dict[Tuple[int, str], Path]:
    """``{(step, prompt): run_dir}`` from Stage 0's ``stage0_index.json``, the only selector."""
    runs = json.loads(Path(index).read_text())["runs"]
    return {(int(k.split("|")[0]), k.split("|")[1]): Path(v) for k, v in runs.items()}


class P0Mismatch(RuntimeError):
    """The P = 0 pass does not reproduce Stage 0's stored activations."""


# ---------------------------------------------------------------------------
# Groups and their readouts
# ---------------------------------------------------------------------------

def groups_of(Y: np.ndarray, frame: str, mcs: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Level-set HDBSCAN labels on ``Y``'s rows in ``frame``, and each group's row indices."""
    from .methods import LayerData
    Z, _ = frame_vectors(Y, frame)
    labels, rows = level_set_hdbscan(LayerData.from_normed(Z).cos_dist, mcs)
    return labels, [np.asarray(r["members"], dtype=int) for r in rows]


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def best_jaccard(group, others: Sequence) -> float:
    return max((jaccard(group, o) for o in others), default=0.0)


def subsample_floor(Y: np.ndarray, frame: str, mcs: int, groups: Sequence[np.ndarray],
                    rng: np.random.Generator, n_sub: int = N_SUBSAMPLES,
                    frac: float = SUBSAMPLE_FRACTION) -> List[Dict]:
    """Per group: Hennig's best-match Jaccards over ``n_sub`` subsamples, their median and J0."""
    n = Y.shape[0]
    m = int(round(frac * n))
    jac = np.zeros((len(groups), n_sub))
    for s in range(n_sub):
        sub = np.sort(rng.choice(n, size=m, replace=False))
        _, sg = groups_of(Y[sub], frame, mcs)
        sg = [set(sub[g].tolist()) for g in sg]
        for i, g in enumerate(groups):
            gs = set(g.tolist()) & set(sub.tolist())
            jac[i, s] = best_jaccard(gs, sg) if gs else 0.0
    return [{"median": float(np.median(j)), "J0": float(np.percentile(j, FLOOR_PERCENTILE)),
             "stable": bool(np.median(j) >= STABLE_MEDIAN)} for j in jac]


def content_label(member_labels: Sequence) -> Optional[str]:
    """`designed_prompts` rule 1: the label of a content group, or None."""
    c = Counter(lab for lab in member_labels if lab is not None)
    if not c:
        return None
    lab, k = c.most_common(1)[0]
    return lab if k >= dp.CONTENT_MIN and k / sum(c.values()) >= dp.CONTENT_PURITY else None


def classify(moves_by_preamble: Dict[str, bool], opening: bool, c_holds: bool) -> str:
    """Unit 1's class for one stable P = 0 group."""
    m = list(moves_by_preamble.values())
    if m and all(m):
        return "moves"
    if any(m):
        return "preamble-dependent"
    return "opening-bound" if (opening and c_holds) else "context-bound"


#: A stable group whose own floor is 0 cannot fail "best Jaccard >= J0", so it
#: is not classified against its floor (`/challenge-pr` on #130, finding 1:
#: 336 of step 0's 412 "moves" had J0 = 0).
FLOOR_ZERO = "floor 0"
#: The fixed bar reported beside the own-floor rule (finding 1), placed.
FIXED_BAR = 0.5


def group_classes(m: Dict, join: str = PRIMARY_JOIN, bar: Optional[float] = None) -> List[str]:
    """
    Each P = 0 group's class from a record's stored Jaccards: against its
    own floor ``J0`` (``bar`` None; ``J0 == 0`` gives `FLOOR_ZERO`) or against
    a fixed ``bar``. Unstable groups are ``"unstable"``. Recomputed here, not
    read from the record's ``class`` field, which runs before the fix wrote
    with zero floors passing.
    """
    out = []
    for i, g in enumerate(m["groups"]):
        if not g["stable"]:
            out.append("unstable")
            continue
        thr = g["J0"] if bar is None else bar
        if bar is None and thr <= 0:
            out.append(FLOOR_ZERO)
            continue
        mv: Dict[str, bool] = {}
        for cid, pc in m["conditions"].items():
            src, _, j = cid.split("|")
            if j == join:
                mv[src] = mv.get(src, True) and pc["best_jaccard"][i] >= thr
        out.append(classify(mv, g["opening"], m["c_holds"]) if mv else "unclassified")
    return out


# ---------------------------------------------------------------------------
# One (layer, frame) of one (step, passage): every condition, both sizes
# ---------------------------------------------------------------------------

_G: Dict = {}   # set before the pool forks: hidden states and per-condition indices


def _layer_job(args: Tuple[int, str]) -> Dict:
    L, frame = args
    g = _G
    conds, kept, labels = g["conds"], g["kept"], g["labels"]
    cos_rec, out = {}, {"layer": L, "frame": frame, "mcs": {}}
    Y0 = g["hidden"]["P0"][L][kept]
    Z0, _ = frame_vectors(Y0, frame)
    for c in conds[1:]:
        Zp, _ = frame_vectors(g["hidden"][c["id"]][L][c["start"] + kept], frame)
        cos_rec[c["id"]] = np.sum(Z0 * Zp, axis=1)
    for mcs in MIN_CLUSTER_SIZES:
        _, g0 = groups_of(Y0, frame, mcs)
        rng = np.random.default_rng([g["seed"], L, FRAMES.index(frame), g["passage_index"]])
        floor = subsample_floor(Y0, frame, mcs, g0, rng)
        offs = [kept[x] for x in g0]
        recs = []
        for i, (x, o) in enumerate(zip(g0, offs)):
            r = {"offsets": o.tolist(), "size": int(o.size), "opening": bool(o.min() < OPENING_OFFSETS),
                 **floor[i]}
            if labels is not None:
                ml = [labels[k] for k in o]
                r["label_counts"] = dict(Counter(lab for lab in ml if lab is not None))
                r["content"] = content_label(ml)
            recs.append(r)
        first = int(np.argmin(kept))
        open_i = next((i for i, x in enumerate(g0) if first in x.tolist()), None)
        if open_i is not None and not recs[open_i]["opening"]:
            open_i = None
        per_cond = {}
        for c in conds[1:]:
            Yp = g["hidden"][c["id"]][L][c["start"] + kept]
            _, gp = groups_of(Yp, frame, mcs)
            gp_off = [kept[x] for x in gp]
            cos = cos_rec[c["id"]]
            pc = {"k": len(gp), "best_jaccard": [best_jaccard(o, gp_off) for o in offs],
                  "group_cos": [float(np.median(cos[x])) for x in g0]}
            if c["join"] == PRIMARY_JOIN:
                wk = g["whole_kept"][c["id"]]
                Yw = g["hidden"][c["id"]][L][wk]
                lw, gw = groups_of(Yw, frame, mcs)
                e = int(np.argmin(wk))
                lab = int(lw[e])
                w = {"n_kept": int(wk.size), "k": len(gw), "earliest_pos": int(wk[e]),
                     "earliest_in_group": lab >= 0}
                if lab >= 0:
                    pos = wk[gw[lab]]
                    po = pos - c["start"]
                    w.update(size=int(pos.size), n_preamble=int(np.sum(pos < c["start"] - 1)),
                             has_join=bool(np.any(pos == c["start"] - 1)),
                             n_passage=int(np.sum(po >= 0)),
                             n_passage_opening=int(np.sum((po >= 0) & (po < OPENING_OFFSETS))))
                pc["whole"] = w
            per_cond[c["id"]] = pc
        c_holds = bool(per_cond) and all(
            pc["whole"]["earliest_in_group"] for cid, pc in per_cond.items() if "whole" in pc)
        m = {"groups": recs, "opening_group": open_i, "c_holds": c_holds, "conditions": per_cond}
        if per_cond:
            for j in JOINS:
                key = "class" if j == PRIMARY_JOIN else f"class_{j}"
                for r, cls in zip(recs, group_classes(m, j)):
                    if r["stable"]:
                        r[key] = cls
        out["mcs"][str(mcs)] = m
    out["cos"] = {cid: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                        "p90": float(np.percentile(v, 90))} for cid, v in cos_rec.items()}
    return out


# ---------------------------------------------------------------------------
# One (step, passage)
# ---------------------------------------------------------------------------

def forward(model, ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Raw hidden states ``(25, n, d)`` float32 and their norms ``(25, n)``."""
    import torch
    with torch.no_grad():
        out = model(input_ids=torch.tensor([list(ids)]), output_hidden_states=True,
                    output_attentions=False)
    H = torch.stack([h[0] for h in out.hidden_states]).to(torch.float32).numpy()
    return H, np.linalg.norm(H, axis=-1)


def run_passage(model, tokenizer, step: str, passage: str, pinfo: Dict, conds: List[Dict],
                workers: int, seed: int, passage_index: int, act_dir: Optional[Path],
                given: Optional[Dict] = None, stage0: Optional[Path] = None) -> Dict:
    hidden, norms, toks, t0 = {}, {}, {}, time.monotonic()
    match = None
    for c in conds:
        H, N = forward(model, c["ids"])
        if c["id"] == "P0" and stage0 is not None:
            match = p0_match(H, stage0)
            if not match["ok"]:
                raise P0Mismatch(f"{step}/{passage}: {json.dumps(match)}")
        hidden[c["id"]], norms[c["id"]] = H, N
        toks[c["id"]] = tokenizer.convert_ids_to_tokens(c["ids"])
        if act_dir is not None and c["P"] in KEEP_P and c["join"] in (None, PRIMARY_JOIN) and \
                c["preamble"] in (None, next(s for s in PREAMBLE_SOURCES if s != passage)):
            d = act_dir / step / passage
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{c['id'].replace('|', '_')}.npz", hidden=H[1:], ids=np.asarray(c["ids"]),
                     start=c["start"])
    t_fwd = time.monotonic() - t0
    n_pass = len(pinfo["ids"])
    ptoks = toks["P0"]
    massive: Dict[int, Dict] = {}
    own: Dict[str, Dict[int, Tuple[float, int]]] = {}
    for c in conds:
        own[c["id"]] = massive_positions(norms[c["id"]])
        for p, (r, L) in own[c["id"]].items():
            o = p - c["start"]
            if 0 <= o < n_pass and (o not in massive or r > massive[o]["ratio"]):
                massive[o] = {"offset": o, "token": ptoks[o], "ratio": r, "layer": L, "condition": c["id"]}
    kept = kept_offsets(ptoks, list(massive))
    dropped = list(massive)
    if given is not None:
        kept = forced_kept(kept, given["kept"], passage)
        dropped = sorted(set(dropped) | {int(m["position"]) for m in given["massive"]})
    whole = {c["id"]: whole_kept(toks[c["id"]], c["start"], list(own[c["id"]]), dropped)
             for c in conds[1:]}
    _G.clear()
    _G.update(conds=conds, kept=kept, labels=pinfo["labels"], hidden=hidden, whole_kept=whole,
              seed=seed, passage_index=passage_index)
    jobs = [(L, f) for L in LAYERS for f in FRAMES]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            layers = pool.map(_layer_job, jobs, chunksize=1)
    else:
        layers = [_layer_job(j) for j in jobs]
    _G.clear()
    return {"step": step, "passage": passage, "n_passage": n_pass, "n_kept": int(kept.size), "p0_match": match,
            "kept_offsets": kept.tolist(), "massive": sorted(massive.values(), key=lambda m: m["offset"]),
            "position0": {cid: {"ratio_max": float(max(norms[cid][L][0] / np.median(norms[cid][L])
                                                       for L in MASSIVE_LAYERS)),
                                "token": toks[cid][0]} for cid in norms},
            "conditions": [{k: c[k] for k in ("id", "preamble", "P", "join", "start")} | {"n": len(c["ids"])}
                           for c in conds],
            "seconds": {"forward": round(t_fwd, 1), "total": round(time.monotonic() - t0, 1)},
            "layers": layers}


# ---------------------------------------------------------------------------
# The first checks
# ---------------------------------------------------------------------------

def _modal(classes: List[str]) -> str:
    c = Counter(classes)
    return c.most_common(1)[0][0] if c else "none"


def opening_table(recs: Sequence[Dict], frame: str = "centred", mcs: int = 2,
                  bar: Optional[float] = None) -> Dict[str, Dict]:
    """Per passage: the opening group's class per layer (`group_classes`), its counts, the modal class."""
    out = {}
    for r in recs:
        per = []
        for lay in r["layers"]:
            if lay["frame"] != frame:
                continue
            m = lay["mcs"][str(mcs)]
            i = m["opening_group"]
            per.append("no opening group" if i is None else group_classes(m, bar=bar)[i])
        out[r["passage"]] = {"by_layer": per, "counts": dict(Counter(per)), "modal": _modal(per)}
    return out


def step0_verdict(table: Dict[str, Dict]) -> str:
    modal = Counter(v["modal"] for v in table.values())
    if modal["opening-bound"] >= MOST:
        return "pass"
    if modal["moves"] >= MOST:
        return "stop"
    return "neither"


def designed_table(recs: Sequence[Dict], frame: str = "centred", mcs: int = 2,
                   bar: Optional[float] = None) -> Dict[str, Dict]:
    """Per designed prompt: content groups, the stable ones by class (rule 2), and floor-0 ones apart."""
    out = {}
    for r in recs:
        cls, n_content, n_zero = Counter(), 0, 0
        for lay in r["layers"]:
            if lay["frame"] != frame:
                continue
            m = lay["mcs"][str(mcs)]
            for g, c in zip(m["groups"], group_classes(m, bar=bar)):
                if g.get("content") is None:
                    continue
                n_content += 1
                if c == FLOOR_ZERO:
                    n_zero += 1
                elif c in CLASSES:
                    cls[c] += 1
        out[r["passage"]] = {"content_groups": n_content, "classified": dict(cls), "floor_zero": n_zero,
                             "share_moves": cls["moves"] / sum(cls.values()) if cls else None}
    return out


def designed_verdict(table: Dict[str, Dict]) -> str:
    tot = Counter()
    for v in table.values():
        tot.update(v["classified"])
    n = sum(tot.values())
    if n == 0:
        return "fail (no classified content group)"
    return "pass" if tot["moves"] / n >= dp.PASS_SHARE else "fail"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def _out_file(out: Path, step: str, passage: str, p0_only: bool) -> Path:
    return out / step / f"{passage}{'_p0' if p0_only else ''}.json"


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="move_text run")
    ap.add_argument("--step", choices=sorted(MODELS), required=True)
    ap.add_argument("--passages", choices=("v1", "designed"), required=True)
    ap.add_argument("--keys", nargs="*", default=None, help="a subset of the passages")
    ap.add_argument("--p0-only", action="store_true", help="P = 0 only (designed at step 0)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--act-out", type=Path, default=None, help="where kept activations go")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override-first-checks", default=None, metavar="REASON")
    ap.add_argument("--kept-from", type=Path, default=None, metavar="TOKEN_SETS_JSON",
                    help="v1 only: take each passage's token set from `arch_null`'s token_sets.json")
    ap.add_argument("--stage0-index", type=Path, default=None, metavar="STAGE0_INDEX_JSON",
                    help="v1 only: check each P = 0 pass against that (step, passage)'s Stage 0 "
                         "activations; a passage over P0_MATCH_TOL refuses (no record)")
    args = ap.parse_args(argv)
    if args.stage0_index is not None and args.passages != "v1":
        print("refusing: --stage0-index is for the v1 passages", file=sys.stderr)
        return 1
    given, kept_from = None, None
    if args.kept_from is not None:
        if args.passages != "v1":
            print("refusing: --kept-from is for the v1 passages", file=sys.stderr)
            return 1
        import hashlib
        raw = args.kept_from.read_bytes()
        ts = json.loads(raw)
        given = ts["sets"]
        kept_from = {"path": str(args.kept_from), "sha256": hashlib.sha256(raw).hexdigest()[:16],
                     "comparison_size": len(ts["comparison"])}

    if args.passages == "v1" and args.step != "step0" and not args.override_first_checks:
        fc = args.out / "first_checks.json"
        ok = fc.exists() and all(v == "pass" for v in json.loads(fc.read_text())["verdicts"].values())
        if not ok:
            print(f"refusing: trained v1 cells need both first checks passed in {fc} "
                  "(run `check`), or --override-first-checks REASON", file=sys.stderr)
            return 1

    import torch
    from transformers import AutoTokenizer
    from core.models import load_model
    from .long_prompts import long_prompts_hash
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    pins = passage_inputs(args.passages, tok)
    keys = args.keys or list(pins)
    conts, joins = continuations(tok), join_ids(tok)
    meta = {"git": _git_head(), "model": MODELS[args.step], "designed_hash": dp.designed_hash(),
            "long_prompts_hash": long_prompts_hash(), "seed": args.seed, "p0_only": args.p0_only,
            "override_first_checks": args.override_first_checks, "kept_from": kept_from,
            "stage0_index": None if args.stage0_index is None else str(args.stage0_index),
            "settings": {"P": P_VALUES, "joins": JOINS, "preambles": PREAMBLE_SOURCES,
                         "massive_ratio": MASSIVE_RATIO, "massive_layers": [MASSIVE_LAYERS[0], MASSIVE_LAYERS[-1]],
                         "n_subsamples": N_SUBSAMPLES, "subsample_fraction": SUBSAMPLE_FRACTION,
                         "floor_percentile": FLOOR_PERCENTILE, "stable_median": STABLE_MEDIAN,
                         "opening_offsets": OPENING_OFFSETS, "min_cluster_sizes": MIN_CLUSTER_SIZES}}
    s0 = stage0_runs(args.stage0_index) if args.stage0_index is not None else None
    step_n = int(args.step.removeprefix("step"))
    model, refused = None, []
    for k in keys:
        f = _out_file(args.out, args.step, k, args.p0_only)
        if f.exists():
            print(f"already {f}", flush=True)
            continue
        if model is None:
            model, _ = load_model(MODELS[args.step])
        conds = conditions(k, pins[k]["ids"], conts, joins, p0_only=args.p0_only)
        stage0 = None
        if s0 is not None:
            stage0 = s0.get((step_n, k))
            if stage0 is None:
                print(f"refusing: {args.step}/{k} has no run in {args.stage0_index}", file=sys.stderr)
                refused.append(k)
                continue
        try:
            rec = run_passage(model, tok, args.step, k, pins[k], conds, args.workers, args.seed,
                              list(pins).index(k), args.act_out, None if given is None else given[k],
                              stage0)
        except P0Mismatch as e:
            print(f"refusing: {e}", file=sys.stderr)
            refused.append(k)
            continue
        rec["meta"] = meta
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec) + "\n")
        print(f"done {f} {rec['seconds']}", flush=True)
    if refused:
        print(f"refused (no record written): {refused}", file=sys.stderr)
        return 1
    return 0


def _load_dir(d: Path) -> List[Dict]:
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))] if d.exists() else []


def check(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="move_text check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    s0 = [r for r in _load_dir(args.out / "step0") if r["passage"] in V1_PASSAGES]
    des = [r for r in _load_dir(args.out / "step143000") if r["passage"] in dp.KEYS]
    des0 = [r for r in _load_dir(args.out / "step0") if r["passage"] in dp.KEYS]
    res: Dict = {"verdicts": {}, "inputs": {}}
    if len(s0) == len(V1_PASSAGES):
        t = opening_table(s0)
        res["step0"] = t
        res["verdicts"]["step0"] = step0_verdict(t)
        res["inputs"]["step0"] = sorted({r["meta"]["git"] for r in s0})
    if len(des) == len(dp.KEYS):
        t = designed_table(des)
        for r in des0:
            t[r["passage"]]["step0_content_groups"] = designed_table([r])[r["passage"]]["content_groups"]
        res["designed"] = t
        res["verdicts"]["designed"] = designed_verdict(t)
        res["inputs"]["designed"] = sorted({r["meta"]["git"] for r in des})
    (args.out / "first_checks.json").write_text(json.dumps(res, indent=1) + "\n")
    for k in ("step0", "designed"):
        if k in res:
            print(f"== {k}: {res['verdicts'][k]}")
            for p, v in res[k].items():
                print(f"  {p:24s} " + json.dumps({x: y for x, y in v.items() if x != "by_layer"}))
        else:
            print(f"== {k}: inputs incomplete")
    return 0


def band(layer: int) -> str:
    return "L1-8" if layer <= 8 else "L9-16" if layer <= 16 else "L17-24"


def class_table(recs: Sequence[Dict], join: str = PRIMARY_JOIN, bar: Optional[float] = None) -> List[Dict]:
    """
    Stable P = 0 groups by class (`group_classes`: own floor, or a fixed
    ``bar``), per (frame, size, band), pooled over passages and the band's
    layers (records are not independent), with floor-0 groups counted apart
    and the median of (b)'s per-token cosine at P = 1000, primary join.
    """
    cells: Dict[Tuple, Dict] = {}
    for r in recs:
        for lay in r["layers"]:
            lay_cos = [v["median"] for cid, v in lay["cos"].items()
                       if f"|{max(P_VALUES)}|{PRIMARY_JOIN}" in cid]
            for mcs, m in lay["mcs"].items():
                c = cells.setdefault((lay["frame"], int(mcs), band(lay["layer"])),
                                     {"classes": Counter(), "groups": 0, "cos": []})
                c["groups"] += len(m["groups"])
                c["classes"].update(group_classes(m, join, bar))
                c["cos"] += lay_cos
    out = []
    for (frame, mcs, b), c in sorted(cells.items()):
        n = sum(c["classes"][k] for k in CLASSES)
        out.append({"frame": frame, "mcs": mcs, "band": b, "groups": c["groups"],
                    "unstable": c["classes"]["unstable"], "floor_zero": c["classes"][FLOOR_ZERO],
                    "classified": n, **{k: c["classes"][k] for k in CLASSES},
                    "share_moves": round(c["classes"]["moves"] / n, 3) if n else None,
                    "cos_P1000_median": round(float(np.median(c["cos"])), 3) if c["cos"] else None})
    return out


def report(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="move_text report")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = {}
    for step in MODELS:
        recs = [r for r in _load_dir(args.out / step) if r["passage"] in V1_PASSAGES]
        if len(recs) != len(V1_PASSAGES):
            continue
        res[step] = {"classes": class_table(recs), "classes_nl2": class_table(recs, "nl2"),
                     "classes_fixed_bar": class_table(recs, bar=FIXED_BAR),
                     "classes_nl2_fixed_bar": class_table(recs, "nl2", FIXED_BAR),
                     "opening": {f"{f}/{m}": opening_table(recs, f, m)
                                 for f in FRAMES for m in MIN_CLUSTER_SIZES},
                     "massive": {r["passage"]: r["massive"] for r in recs},
                     "git": sorted({r["meta"]["git"] for r in recs})}
    (args.out / "report.json").write_text(json.dumps(res, indent=1) + "\n")
    cols = ("frame", "mcs", "band", "groups", "unstable", "floor_zero", "classified", *CLASSES,
            "share_moves", "cos_P1000_median")
    for step, v in res.items():
        print(f"== {step}  (primary join; stable groups by class)")
        print("  " + " ".join(f"{c[:12]:>12s}" for c in cols))
        for row in v["classes"]:
            print("  " + " ".join(f"{str(row[c])[:12]:>12s}" for c in cols))
        print(f"   fixed bar {FIXED_BAR}, share moves: " + json.dumps(
            {f"{x['frame']}/{x['mcs']}/{x['band']}": x["share_moves"] for x in v["classes_fixed_bar"]}))
        print(f"   opening group, centred / 2: " + json.dumps(
            {p: x["modal"] for p, x in v["opening"]["centred/2"].items()}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"run": run, "check": check, "report": report}
    if not argv or argv[0] not in cmds:
        print(f"usage: move_text {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
