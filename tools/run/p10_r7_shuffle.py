"""R7: the context-shuffle test (`p10_cluster_function/design-10.md` "R7"; `STATE.md` Blocked 26 (c)).

Does a definition group need its passage's context, or only its tokens? Per (step, passage),
the passage is run as read (`orig`, checked against Stage 0's stored activations), with its
offsets 1 … n−1 block-shuffled (b ∈ 64, 16, 4, 1; K = 5 permutations each, the same at every
step), and each kept token `alone` after the passage's offset 0. In each condition the same kept
offsets are clustered (centred, level-set HDBSCAN size 2, all groups) and each c2 group of R0's
labels (c3 a subset) gets its best-match Jaccard there. A group **survives** a level when the
median of its K Jaccards is ≥ its own unit 1 floor `J0`; `alone` when J ≥ `J0`. Label:
**token-borne** (survives `alone`), **bag-borne** (fails `alone`, survives b = 1), **order-borne**
(fails both). `run` writes one record per (step, passage); `read` labels and pools them.

First checks (refuse rather than degrade): `orig` matches Stage 0 (`move_text.p0_match`); `orig`'s
groups are R0's c2a member sets at every layer (else the layer refuses); every shuffled sequence
is a permutation with offset 0 fixed and each kept token at its mapped position; the `alone` state
of offset 1 equals `orig`'s position 1 within `ALONE_TOL` (a departure, below).
Tier 1: exploratory, unregistered, descriptive. Run (METS_DATA set; see `run_r7.sh` in the data dir):
    python tools/run/p10_r7_shuffle.py run --step step512 --labels <R0 labels> --unit1 <R0 unit1> \
        --index <stage0_index.json> --out <dir>
    python tools/run/p10_r7_shuffle.py read --src <dir> --labels <R0 labels> --out <r7.json>
R9 / R7 (`design-10.md` "R9 / R7"): c3x beside c3 on the stored records (filtered, exact), with the
random-drop reference matched per (step, passage) and per record:
    python tools/run/p10_r7_shuffle.py reread --src <dir> --labels <R9 labels> --out <file> --lead c3x \
        [--reproduce <r7.json> --reproduce-labels <R0 labels>] [--draws 100]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from p1d_cluster_ensemble import move_text as mt
from p1d_cluster_ensemble.gaussian_null import frame_vectors
from tools.run import p10_label_source as ls

BLOCKS = (64, 16, 4, 1)                 # design-10 "R7": graded, b = 1 a full token shuffle
K = 5                                   # permutations per level
FRAME, MCS = "centred", 2               # the definition's cloud
N_CHANCE, CHANCE_PCT = 20, 95           # chance level per group size and condition
FIXED_BAR = mt.FIXED_BAR                # 0.5, beside
TOKEN_HI, TOKEN_LO = 2 / 3, 1 / 3       # step label, placed (as R6f's)
PROMPT_MIN = 5                          # group-layer records for a per-prompt label
BANDS = {"L1-8": range(1, 9), "L9-16": range(9, 17), "L17-24": range(17, 25)}
LABELS = ("token-borne", "bag-borne", "order-borne")
#: Check (d)'s tolerance. *Departure, after the gate (2026-10-06):* the rule borrowed
#: `P0_MATCH_TOL` (1e-5, placed for a re-run of the same input); a 2-token pass against the
#: passage's pass is a different-length float32 computation, 1.5e-5 apart at step 143000 on
#: `latex_monograph` even unbatched. A wrong pairing or mask differs at O(1).
ALONE_TOL = 1e-4


class ShuffleError(RuntimeError):
    pass


# ---------------------------------------------------------------- conditions (pure)

def block_order(n: int, b: int, passage_index: int, k: int, seed: int = 0) -> np.ndarray:
    """Offsets in run order: 0 first, then offsets 1 … n−1 in blocks of b, blocks permuted (not identity)."""
    blocks = [np.arange(s, min(s + b, n)) for s in range(1, n, b)]
    if len(blocks) < 2:
        raise ShuffleError(f"n = {n}, b = {b}: one block, no shuffle")
    rng = np.random.default_rng([seed, passage_index, b, k])
    order = rng.permutation(len(blocks))
    while np.all(order == np.arange(len(blocks))):
        order = rng.permutation(len(blocks))
    return np.concatenate([[0]] + [blocks[i] for i in order]).astype(int)


def condition_ids(n: int, passage_index: int, seed: int = 0) -> dict:
    """{name: offsets in run order} for orig and every block level × permutation."""
    out = {"orig": np.arange(n)}
    for b in BLOCKS:
        for k in range(K):
            out[f"b{b}_k{k}"] = block_order(n, b, passage_index, k, seed)
    return out


def positions_of(order: np.ndarray) -> np.ndarray:
    """pos[o] = the position offset o runs at."""
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    return pos


def check_order(ids, order, kept) -> None:
    """Check (c): a permutation with 0 fixed, each kept offset's token at its mapped position."""
    n = len(ids)
    if order[0] != 0 or sorted(order.tolist()) != list(range(n)):
        raise ShuffleError("not a permutation with offset 0 first")
    seq, pos = np.asarray(ids)[order], positions_of(order)
    if not all(seq[pos[o]] == ids[o] for o in kept):
        raise ShuffleError("a kept token is not at its mapped position")


# ---------------------------------------------------------------- Jaccards on a labelling (pure)

def best_jaccard_lab(members: np.ndarray, lab: np.ndarray) -> float:
    """Best-match Jaccard of index set ``members`` against the disjoint groups of ``lab`` (−1 noise,
    −2 dropped). Dropped indices leave the group (it is restricted); noise does not."""
    m = members[lab[members] != -2]
    if m.size == 0:
        return 0.0
    lm = lab[m]
    lm = lm[lm >= 0]
    if lm.size == 0:
        return 0.0
    sizes = np.bincount(lab[lab >= 0])
    ids, inter = np.unique(lm, return_counts=True)
    return float(np.max(inter / (m.size + sizes[ids] - inter)))


def chance95(size: int, lab: np.ndarray, rng: np.random.Generator) -> float:
    """95th percentile of the best-match Jaccard of N_CHANCE random same-size sets of present indices."""
    pool = np.flatnonzero(lab != -2)
    if pool.size < size:
        return float("nan")
    js = [best_jaccard_lab(rng.choice(pool, size=size, replace=False), lab) for _ in range(N_CHANCE)]
    return float(np.percentile(js, CHANCE_PCT))


# ---------------------------------------------------------------- labels (pure)

def survives(js, bar: float) -> bool:
    return bool(np.median(js) >= bar)


def group_label(g: dict, bar=None) -> dict:
    """Survival per level and the label for one stored group (bar None = its own J0)."""
    thr = g["J0"] if bar is None else bar
    s = {f"b{b}": survives(g["J"][f"b{b}"], thr) for b in BLOCKS}
    s["alone"] = bool(g["J"]["alone"] >= thr)
    lab = "token-borne" if s["alone"] else ("bag-borne" if s["b1"] else "order-borne")
    return {"survives": s, "label": lab}


def step_label(token_share) -> str:
    if token_share is None:
        return "no records"
    return "token" if token_share >= TOKEN_HI else ("context" if token_share <= TOKEN_LO else "mixed")


def tally(labels) -> dict:
    c = Counter(labels)
    n = sum(c.values())
    return {"n": n, **{k: c.get(k, 0) for k in LABELS},
            "token_share": (c.get("token-borne", 0) / n) if n else None}


# ---------------------------------------------------------------- the run

_G: dict = {}


def _layer_job(L: int) -> dict:
    g = _G
    kept, r0 = g["kept"], g["r0_layers"].get(str(L))
    if r0 is None:
        return {"layer": L, "refused": "R0 refused this layer"}
    c2a = np.asarray(r0["c2a"])
    want = {frozenset(np.flatnonzero(c2a == i).tolist()) for i in set(c2a.tolist()) if i >= 0}
    lab0, g0 = mt.groups_of(g["H"]["orig"][L], FRAME, MCS)
    if {frozenset(x.tolist()) for x in g0} != want:
        return {"layer": L, "refused": "orig groups are not R0's c2a member sets"}
    c2, c3 = np.asarray(r0["c2"]), np.asarray(r0["c3"])
    u1 = g["u1_groups"][L]
    ids = sorted(i for i in set(c2.tolist()) if i >= 0)
    groups = []
    for i in ids:
        mem = np.flatnonzero(c2a == i)
        if sorted(kept[mem].tolist()) != sorted(u1[i]["offsets"]):
            raise ShuffleError(f"L{L} group {i}: R0's members are not unit 1's")
        cl = [g["classes"][j] for j in mem]
        pairs = [(a, b) for x, a in enumerate(cl) for b in cl[x + 1:]]
        groups.append({"id": int(i), "size": int(mem.size), "J0": float(u1[i]["J0"]),
                       "c3": bool(np.any(c3 == i)), "mem": mem,
                       "same_class": float(np.mean([a == b for a, b in pairs])) if pairs else None,
                       "J": {}, "chance95": {}})
    Z0, _ = frame_vectors(g["H"]["orig"][L], FRAME)
    c3m = c3 >= 0
    selfsim, dropped = {}, {}
    for ci, (cname, H) in enumerate(g["H"].items()):
        if cname == "orig":
            continue
        drop = g["massive"].get(cname, np.zeros(kept.size, bool))
        keep = ~drop
        lab = np.full(kept.size, -2, dtype=int)
        lab[keep] = mt.groups_of(H[L][keep], FRAME, MCS)[0]
        rng = np.random.default_rng([g["seed"], g["passage_index"], L, ci])
        cache = {}
        for gr in groups:
            gr["J"][cname] = best_jaccard_lab(gr["mem"], lab)
            if gr["size"] not in cache:
                cache[gr["size"]] = chance95(gr["size"], lab, rng)
            gr["chance95"][cname] = cache[gr["size"]]
        Zc, _ = frame_vectors(H[L][keep], FRAME)
        cos = np.sum(Z0[keep] * Zc, axis=1)
        mk = c3m[keep]
        selfsim[cname] = {"members": float(np.median(cos[mk])) if mk.any() else None,
                          "rest": float(np.median(cos[~mk])) if (~mk).any() else None}
        if drop.any():
            dropped[cname] = int(drop.sum())
    for gr in groups:
        del gr["mem"]
    return {"layer": L, "groups": groups, "selfsim": selfsim, "massive_dropped": dropped}


def _rel(H, U, tol: float = mt.P0_MATCH_TOL) -> dict:
    """Direction and relative-norm differences of two (..., d) arrays (as `move_text.p0_match`)."""
    nh, nu = np.linalg.norm(H.astype(np.float64), axis=-1), np.linalg.norm(U.astype(np.float64), axis=-1)
    d = float(np.max(np.abs(H / nh[..., None] - U / nu[..., None])))
    r = float(np.max(np.abs(nh / nu - 1.0)))
    return {"direction_max": d, "rel_norm_max": r, "tol": tol, "ok": bool(d <= tol and r <= tol)}


def run_passage(model, tok, step: str, passage: str, ids, r0p: dict, u1: dict, stage0: Path,
                workers: int, seed: int) -> dict:
    import torch
    t0 = time.monotonic()
    pi = mt.V1_PASSAGES.index(passage)
    kept = np.asarray(r0p["kept"], dtype=int)
    n = len(ids)
    if r0p["n_positions"] != n:
        raise ShuffleError(f"{step}/{passage}: R0 has {r0p['n_positions']} positions, the passage {n}")
    H, massive, checks = {}, {}, {}
    for cname, order in condition_ids(n, pi, seed).items():
        check_order(ids, order, kept)
        Hc, Nc = mt.forward(model, np.asarray(ids)[order].tolist())
        if cname == "orig":
            m = mt.p0_match(Hc, stage0)
            checks["orig_match"] = m
            if not m["ok"]:
                raise ShuffleError(f"{step}/{passage}: orig does not match Stage 0: {json.dumps(m)}")
            orig_pos1 = Hc[:, 1, :].copy()
        pos = positions_of(order)
        H[cname] = Hc[:, pos[kept], :]
        mp = mt.massive_positions(Nc)
        if cname != "orig":
            bad = {int(order[p]) for p in mp}
            massive[cname] = np.isin(kept, sorted(bad))
    # alone: [offset 0, t] for every kept token, plus offset 1 for check (d)
    toks = [ids[o] for o in kept] + [ids[1]]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[ids[0], t] for t in toks]), output_hidden_states=True)
    A = torch.stack([h[:, 1, :] for h in out.hidden_states]).to(torch.float32).numpy()
    checks["alone_offset1"] = _rel(A[:, -1, :], orig_pos1, ALONE_TOL)
    if not checks["alone_offset1"]["ok"]:
        raise ShuffleError(f"{step}/{passage}: alone offset 1 != orig position 1: {checks['alone_offset1']}")
    H["alone"] = A[:, :-1, :]
    An = np.linalg.norm(H["alone"], axis=-1)
    am = mt.massive_positions(An)        # T2 over the kept tokens' position-1 states
    massive["alone"] = np.isin(np.arange(kept.size), sorted(am))
    t_fwd = time.monotonic() - t0
    from tools.run.p10_token_composition import decode, token_class
    strs = tok.convert_ids_to_tokens(list(ids))
    text = [decode(s) for s in strs]
    classes = [token_class(text[o], text[o - 1] if o > 0 else None, int(o)) for o in kept]
    u1_groups = {}
    for lay in u1["layers"]:
        if lay["frame"] == FRAME:
            u1_groups[lay["layer"]] = lay["mcs"][str(MCS)]["groups"]
    _G.clear()
    _G.update(H=H, kept=kept, r0_layers=r0p["layers"], u1_groups=u1_groups, classes=classes,
              massive=massive, seed=seed, passage_index=pi)
    if workers > 1:
        import multiprocessing as mp_
        with mp_.get_context("fork").Pool(workers) as pool:
            layers = pool.map(_layer_job, mt.LAYERS, chunksize=1)
    else:
        layers = [_layer_job(L) for L in mt.LAYERS]
    _G.clear()
    return {"step": step, "passage": passage, "n": n, "n_kept": int(kept.size), "checks": checks,
            "massive_alone": int(massive["alone"].sum()),
            "seconds": {"forward": round(t_fwd, 1), "total": round(time.monotonic() - t0, 1)},
            "layers": layers}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def _git_head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="p10_r7_shuffle run")
    ap.add_argument("--step", choices=sorted(mt.MODELS), required=True)
    ap.add_argument("--labels", type=Path, required=True, help="R0's label source dir")
    ap.add_argument("--unit1", type=Path, required=True, help="R0's unit 1 records dir")
    ap.add_argument("--index", type=Path, required=True, help="stage0_index.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoTokenizer
    from core.models import load_model
    if a.torch_threads:
        torch.set_num_threads(a.torch_threads)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    pins = mt.passage_inputs("v1", tok)
    r0 = ls.load_step(a.labels, a.step)
    s0 = mt.stage0_runs(a.index)
    step_n = int(a.step.removeprefix("step"))
    meta = {"git": _git_head(), "model": mt.MODELS[a.step], "seed": a.seed, "blocks": BLOCKS, "K": K,
            "frame": FRAME, "mcs": MCS, "n_chance": N_CHANCE, "labels": str(a.labels),
            "labels_sha": _sha(Path(a.labels) / f"{a.step}.json"), "index_sha": _sha(a.index)}
    model, refused = None, {}
    for k in a.keys or mt.V1_PASSAGES:
        f = a.out / a.step / f"{k}.json"
        if f.exists():
            print(f"already {f}", flush=True)
            continue
        if k not in r0["prompts"]:
            refused[k] = f"R0 refused: {r0['refused'].get(k)}"
            continue
        u1f = a.unit1 / a.step / f"{k}.json"
        if (step_n, k) not in s0 or not u1f.exists():
            refused[k] = "no Stage 0 run or unit 1 record"
            continue
        if model is None:
            model, _ = load_model(mt.MODELS[a.step])
        try:
            rec = run_passage(model, tok, a.step, k, pins[k]["ids"], r0["prompts"][k],
                              json.loads(u1f.read_text()), s0[(step_n, k)], a.workers, a.seed)
        except ShuffleError as e:
            refused[k] = str(e)
            continue
        rec["meta"] = meta | {"unit1_sha": _sha(u1f)}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec) + "\n")
        print(f"done {f} {rec['seconds']}", flush=True)
    if refused:
        print(f"refused (no record): {json.dumps(refused)}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------- the reading

def _records(src: Path):
    for f in sorted(Path(src).glob("step*/*.json")):
        yield json.loads(f.read_text())


def _block_js(g: dict) -> dict:
    """{"b64": [K values], …, "alone": value} from a stored group's per-condition Jaccards."""
    out = {f"b{b}": [g["J"][f"b{b}_k{k}"] for k in range(K)] for b in BLOCKS}
    out["alone"] = g["J"]["alone"]
    return out


def primary_column(r0_step: dict) -> str:
    """c3, or c2 where fewer than half the step's records are readable on c3 (design-10 "Readable")."""
    recs = [lay["c3"] for p in r0_step["prompts"].values() for lay in p["layers"].values()]
    ok = sum(ls.readable(r) for r in recs)
    return "c3" if recs and ok / len(recs) >= 0.5 else "c2"


def step_rows(recs) -> list:
    """Every stored c2 group-layer record of one step, labelled (its id, size and c3 flag kept, R9 / R7)."""
    rows = []
    for r in recs:
        for lay in r["layers"]:
            for g in lay.get("groups", []):
                gg = {"J0": g["J0"], "J": _block_js(g)}
                own, fixed = group_label(gg), group_label(gg, FIXED_BAR)
                ch = [g["chance95"][c] for c in g["chance95"]]
                rows.append({"passage": r["passage"], "layer": lay["layer"], "id": g["id"], "size": g["size"],
                             "c3": g["c3"], "label": own["label"],
                             "label_fixed": fixed["label"], "surv": own["survives"],
                             "same_class": g["same_class"],
                             "J0_below_chance": bool(np.nanmedian(ch) > g["J0"]) if ch else None,
                             "token_fail_b1": own["label"] == "token-borne" and not own["survives"]["b1"]})
    return rows


def read_step(recs, column: str) -> dict:
    """Labels pooled over prompts and layers for one step and column (c3 or c2)."""
    return read_rows([r for r in step_rows(recs) if column != "c3" or r["c3"]])


def read_rows(rows) -> dict:
    """R7's reading of one step's rows: shares, step label, bands, prompts."""
    out = tally(r["label"] for r in rows)
    out["label"] = step_label(out["token_share"])
    out["fixed_bar"] = tally(r["label_fixed"] for r in rows)
    out["survival_share"] = {lv: (float(np.mean([r["surv"][lv] for r in rows])) if rows else None)
                             for lv in [f"b{b}" for b in BLOCKS] + ["alone"]}
    out["token_borne_failing_b1"] = sum(r["token_fail_b1"] for r in rows)
    out["J0_below_chance"] = sum(bool(r["J0_below_chance"]) for r in rows)
    # After `/challenge-pr` on #154 (finding 1), beside, not part of the rule: the labels split by
    # whether the group's own floor clears its chance level; token-borne among token-borne's b = 1.
    out["by_floor"] = {k: tally(r["label"] for r in rows if bool(r["J0_below_chance"]) == below)
                       for k, below in (("J0_at_or_above_chance", False), ("J0_below_chance", True))}
    nt = out["token-borne"]
    out["token_borne_failing_b1_share"] = out["token_borne_failing_b1"] / nt if nt else None
    out["bands"] = {}
    for name, rng in BANDS.items():
        t = tally(r["label"] for r in rows if r["layer"] in rng)
        out["bands"][name] = t | {"label": step_label(t["token_share"])}
    out["prompts"] = {}
    for p in mt.V1_PASSAGES:
        t = tally(r["label"] for r in rows if r["passage"] == p)
        out["prompts"][p] = t | {"label": step_label(t["token_share"]) if t["n"] >= PROMPT_MIN else "too few"}
    out["prompt_labels"] = dict(Counter(v["label"] for v in out["prompts"].values()))
    out["same_class"] = {lab: (float(np.mean(v)) if (v := [r["same_class"] for r in rows
                                                            if r["label"] == lab and r["same_class"] is not None])
                               else None) for lab in LABELS}
    return out


def read_selfsim(recs, column_c3=True) -> dict:
    """Median over (prompt, layer) of the members' and the rest's self-similarity, per level."""
    acc = {}
    for r in recs:
        for lay in r["layers"]:
            for c, v in lay.get("selfsim", {}).items():
                lv = c.split("_")[0]
                acc.setdefault(lv, {"members": [], "rest": []})
                for side in ("members", "rest"):
                    if v[side] is not None:
                        acc[lv][side].append(v[side])
    return {lv: {s: (float(np.median(x)) if x else None) for s, x in d.items()} for lv, d in acc.items()}


def _step_n(step: str) -> int:
    return int(step.removeprefix("step"))


def records_by_step(src: Path) -> dict:
    by_step = {}
    for r in _records(src):
        by_step.setdefault(r["step"], []).append(r)
    return {s: by_step[s] for s in sorted(by_step, key=_step_n)}


def read_steps(by_step: dict, labels: Path) -> dict:
    """`read`'s per-step entries (c3 and c2 on the label source ``labels``)."""
    out = {}
    for step, recs in by_step.items():
        out[step] = {"n_prompts": len(recs), "primary": primary_column(ls.load_step(labels, step)),
                     "c3": read_step(recs, "c3"), "c2": read_step(recs, "c2"),
                     "selfsim": read_selfsim(recs),
                     "refused_layers": sum("refused" in lay for r in recs for lay in r["layers"]),
                     "massive_dropped": sum(sum(lay.get("massive_dropped", {}).values())
                                            for r in recs for lay in r["layers"])}
    return out


# ---------------------------------------------------------------- R9 / R7: c3x beside c3, filtered (pure)
# `design-10.md` "R9 / R7". A group's stored label reads only its own members and each condition's
# clustering of all kept offsets, so a column that is a subset of c3 is c3's rows less the others.

N_DRAWS, QUANTILE = 100, 0.95
MATCHES = ("step_passage", "record")    # the rule's match, then R6f's beside


def domains(src: Path, steps) -> dict:
    """{(step, passage, layer): {"c3", "c3x"} domain labels} from the R9 source."""
    out = {}
    for s in steps:
        for p, pr in ls.load_step(src, s)["prompts"].items():
            for L, lay in pr["layers"].items():
                out[(s, p, int(L))] = {c: np.asarray(lay[c], dtype=int) for c in ("c3", "c3x")}
    return out


def ids_of(lab: np.ndarray) -> set:
    return set(map(int, np.unique(lab[lab >= 0])))


def check_structure(by_step: dict, src: Path) -> list:
    """The rule's structure check: stored ids are the source's c2, stored c3 flags its c3, c3x ⊆ c3
    with the same members. The mismatching (step, passage, layer)s."""
    bad = []
    for s, recs in by_step.items():
        d = ls.load_step(src, s)
        for r in recs:
            for lay in r["layers"]:
                if "refused" in lay:
                    continue
                src_lay = d["prompts"][r["passage"]]["layers"][str(lay["layer"])]
                L = {c: np.asarray(src_lay[c], dtype=int) for c in ("c2", "c3", "c3x")}
                c3, c3x = ids_of(L["c3"]), ids_of(L["c3x"])
                if ({g["id"] for g in lay["groups"]} != ids_of(L["c2"])
                        or {g["id"] for g in lay["groups"] if g["c3"]} != c3 or not c3x <= c3
                        or any(not np.array_equal(L["c3"] == i, L["c3x"] == i) for i in c3x)):
                    bad.append((s, r["passage"], lay["layer"]))
    return bad


def c3x_drops(dom: dict) -> dict:
    """{key: the c3 group ids c3x drops there}."""
    return {k: ids_of(d["c3"]) - ids_of(d["c3x"]) for k, d in dom.items()}


def draw_drops(dom: dict, x_drops: dict, match: str, rng: np.random.Generator) -> dict:
    """One draw: c3x's count of dropped c3 groups per record (``match`` "record") or pooled over the
    layers of each (step, passage) ("step_passage"), uniformly without replacement. Keys sorted, so
    the seed fixes the draw."""
    out = {k: set() for k in dom}
    if match == "record":
        for k in sorted(dom):
            if x_drops[k]:
                out[k] = set(map(int, rng.choice(sorted(ids_of(dom[k]["c3"])), len(x_drops[k]), replace=False)))
        return out
    if match != "step_passage":
        raise ValueError(match)
    by_sp = {}
    for k in sorted(dom):
        by_sp.setdefault(k[:2], []).append(k)
    for sp, keys in by_sp.items():
        n = sum(len(x_drops[k]) for k in keys)
        if n:
            pool = [(k, g) for k in keys for g in sorted(ids_of(dom[k]["c3"]))]
            for i in rng.choice(len(pool), n, replace=False):
                out[pool[i][0]].add(pool[i][1])
    return out


def drop_labels(dom: dict, dropped: dict) -> dict:
    """{key: c3 with the dropped groups' members → −1}."""
    return {k: np.where(np.isin(d["c3"], sorted(dropped.get(k, ()))), -1, d["c3"]) for k, d in dom.items()}


def read_column(rows: dict, labs: dict, keep, steps) -> dict:
    """{step: {"primary": "own" or "c2", "read": read_rows of the kept c3 rows}}: own where at least
    half the step's records are readable on ``labs`` (as `primary_column`)."""
    out = {}
    for s in steps:
        recs = [lab for (st, _, _), lab in labs.items() if st == s]
        ok = sum(ls.readable(lab) for lab in recs)
        out[s] = {"primary": "own" if recs and ok / len(recs) >= 0.5 else "c2",
                  "read": read_rows([r for r in rows[s] if r["c3"] and keep(s, r)])}
    return out


def cells(col: dict, steps) -> dict:
    """{"step|cell": label} at ``steps``: the step label on the own floor and the fixed bar (the
    headline), each band's and each prompt's. ``None``: the column reads on c2 there (not held)."""
    out = {}
    for s in steps:
        held, t = col[s]["primary"] == "own", col[s]["read"]
        out[f"{s}|step own"] = t["label"] if held else None
        out[f"{s}|step fixed"] = step_label(t["fixed_bar"]["token_share"]) if held else None
        for b in BANDS:
            out[f"{s}|band {b}"] = t["bands"][b]["label"] if held else None
        for p in mt.V1_PASSAGES:
            out[f"{s}|prompt {p}"] = t["prompts"][p]["label"] if held else None
    return out


def is_headline(key: str) -> bool:
    return "|step " in key


def compare_cells(ref: dict, col: dict) -> dict:
    """A cell is compared when either side holds a label, and changes when the two differ."""
    keys = sorted(set(ref) | set(col))
    compared = [k for k in keys if ref.get(k) is not None or col.get(k) is not None]
    changed = [k for k in compared if ref.get(k) != col.get(k)]
    return {"n_compared": len(compared), "n_changed": len(changed), "changed": changed,
            "headline_changed": [k for k in changed if is_headline(k)]}


def drop_profile(rows: dict, dropped: dict, steps) -> dict:
    """The dropped group-layer records at ``steps``: count, mean size, own-floor label shares, the
    share with `J0` below chance (what a count-matched draw does not match)."""
    rs = [r for s in steps for r in rows[s] if r["c3"] and r["id"] in dropped.get((s, r["passage"], r["layer"]), ())]
    n = len(rs)
    return {"n": n, "mean_size": float(np.mean([r["size"] for r in rs])) if n else None,
            "label_share": {lab: sum(r["label"] == lab for r in rs) / n if n else None for lab in LABELS},
            "J0_below_chance": sum(bool(r["J0_below_chance"]) for r in rs) / n if n else None}


def reference_reading(obs: int, counts: list) -> dict:
    """Within random drops if ``obs`` ≤ the draws' 95th percentile, else beyond; the rank p beside."""
    q = float(np.quantile(counts, QUANTILE, method="higher"))
    return {"obs": obs, "q95": q, "median": float(np.median(counts)),
            "reading": "within random drops" if obs <= q else "beyond random drops",
            "rank_p": (1 + sum(x >= obs for x in counts)) / (len(counts) + 1)}


def run_reference(rows, dom, x_drops, steps, compared, c3_cells, x_cmp, match, n_draws) -> dict:
    """The random-drop reference under one match: (r3) the first draw is read and checked populated
    before the rest; each count against c3x's."""
    draws = []
    for seed in range(n_draws):
        dropped = draw_drops(dom, x_drops, match, np.random.default_rng(seed))
        col = read_column(rows, drop_labels(dom, dropped),
                          lambda s, r, d=dropped: r["id"] not in d[(s, r["passage"], r["layer"])], steps)
        cmp = compare_cells(c3_cells, cells(col, compared))
        n_rec = sum(col[s]["read"]["n"] for s in compared)
        if seed == 0 and (not n_rec or not cmp["n_compared"]):
            raise ShuffleError(f"check (r3): the first draw has {n_rec} records, {cmp['n_compared']} "
                               "compared cells; refusing")
        draws.append({"seed": seed, "n_records": n_rec, "n_changed": cmp["n_changed"],
                      "n_headline_changed": len(cmp["headline_changed"]), "changed": cmp["changed"],
                      "profile": drop_profile(rows, dropped, compared)})
    counts, head = [d["n_changed"] for d in draws], [d["n_headline_changed"] for d in draws]
    every = sorted({k for d in draws for k in d["changed"]} | set(x_cmp["changed"]))
    always = sorted(k for k in every if all(k in d["changed"] for d in draws))
    prof = [d["profile"] for d in draws]
    q = lambda xs: {"median": float(np.median(xs)), "5_95": [float(np.quantile(xs, 0.05)), float(np.quantile(xs, 0.95))]}
    return {"match": match, "n_draws": n_draws, "seeds": [0, n_draws - 1], "quantile": QUANTILE,
            "quantile_method": "higher",
            "all": reference_reading(x_cmp["n_changed"], counts),
            "headline": reference_reading(len(x_cmp["headline_changed"]), head),
            "cell_share": {k: sum(k in d["changed"] for d in draws) / n_draws for k in every},
            "forced": {"cells_every_draw_changes": always,
                       "all_without_them": reference_reading(sum(k not in always for k in x_cmp["changed"]),
                                                             [sum(k not in always for k in d["changed"]) for d in draws])},
            "draws_profile": {"mean_size": q([p["mean_size"] for p in prof]),
                              "label_share": {lab: q([p["label_share"][lab] for p in prof]) for lab in LABELS},
                              "J0_below_chance": q([p["J0_below_chance"] for p in prof])},
            "draws": [{k: d[k] for k in ("seed", "n_records", "n_changed", "n_headline_changed")} for d in draws]}


def reproduce(data: dict, stored: dict) -> list:
    """The steps whose c3 / c2 entry read on the new source differs from the stored `r7.json`'s."""
    return [s for s in stored["steps"] if data["steps"].get(s) != stored["steps"][s]]


def reread(argv=None) -> int:
    """R9 / R7: c3x beside c3 on the stored records, and the random-drop reference."""
    from tools.run import p10_r9_lead as lead_args
    ap = argparse.ArgumentParser(prog="p10_r7_shuffle reread")
    ap.add_argument("--src", type=Path, required=True, help="R7's records dir")
    ap.add_argument("--labels", type=Path, required=True, help="the R9 label source (with c3x)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    lead_args.add_args(ap, "R7")
    a = ap.parse_args(argv)
    lead_args.reproduce_inputs(a)
    summary = a.labels / "summary.json"
    if not summary.exists():
        raise SystemExit(f"no {summary}: the input would be unnamed")
    by_step = records_by_step(a.src)
    steps = list(by_step)
    bad = check_structure(by_step, a.labels)
    if bad:
        raise ShuffleError(f"structure check: stored groups are not the source's at {bad[:3]}; refusing")
    print(f"structure check: {sum(len(r['layers']) for v in by_step.values() for r in v)} records match the source")
    mine = json.loads(json.dumps(read_steps(by_step, a.labels)))

    def load(rec: Path, labels: Path) -> dict:
        stored = json.loads(Path(rec).read_text())
        if stored["meta"]["labels"] != str(labels):
            raise lead_args.LadderError(f"{rec} read {stored['meta']['labels']}, not {labels}")
        return stored
    reproduces = lead_args.check_reproduces(a, {"steps": mine}, load, reproduce, "c3 and c2, every step")

    rows = {s: step_rows(recs) for s, recs in by_step.items()}
    dom = domains(a.labels, steps)
    c3 = read_column(rows, {k: d["c3"] for k, d in dom.items()}, lambda s, r: True, steps)
    compared = [s for s in steps if c3[s]["primary"] == "own"]
    if any(json.dumps(c3[s]["read"]) != json.dumps(mine[s]["c3"])
           or (mine[s]["primary"] == "c3") != (s in compared) for s in steps):
        raise ShuffleError("c3 through the column reader is not read_step's c3; refusing")
    out = {"checks": {"structure": True}, "compared_steps": compared,
           "columns": {"c3": json.loads(json.dumps(c3))}}
    prim = {_step_n(s): ("c3" if s in compared else "c2") for s in steps}
    if a.lead == "c3x":
        x = read_column(rows, {k: d["c3x"] for k, d in dom.items()},
                        lambda s, r: bool(np.any(dom[(s, r["passage"], r["layer"])]["c3x"] == r["id"])), steps)
        if not x[compared[0]]["read"]["n"]:
            raise ShuffleError(f"c3x has no records at {compared[0]}; refusing")
        print(f"c3x at {compared[0]}: {x[compared[0]]['read']['n']} group-layer records")
        prim = {_step_n(s): ("c3x" if x[s]["primary"] == "own" else "c2") for s in steps}
        x_drops = c3x_drops(dom)
        # check (r1): c3 less c3x's dropped groups is c3x, labels and cells (a valid draw under both
        # matches: it has their counts by construction)
        labs = drop_labels(dom, x_drops)
        if any(not np.array_equal(labs[k], dom[k]["c3x"]) for k in dom):
            raise ShuffleError("check (r1): c3 less c3x's drops is not c3x's labels; refusing")
        xd = read_column(rows, labs, lambda s, r: r["id"] not in x_drops[(s, r["passage"], r["layer"])], steps)
        if json.dumps(xd) != json.dumps(x):
            raise ShuffleError("check (r1): c3 less c3x's drops is not c3x's reading; refusing")
        none = read_column(rows, drop_labels(dom, {}), lambda s, r: True, steps)
        if json.dumps(none) != json.dumps(c3):    # check (r2)
            raise ShuffleError("check (r2): a draw dropping nothing is not c3; refusing")
        out["checks"].update(r1_c3_less_drops_is_c3x=True, r2_no_drop_is_c3=True)
        out["columns"]["c3x"] = json.loads(json.dumps(x))
        c3_cells = cells(c3, compared)
        x_cmp = compare_cells(c3_cells, cells(x, compared))
        out["c3x_vs_c3"] = {**x_cmp, "c3_cells": c3_cells, "c3x_cells": cells(x, compared)}
        sp = {}
        for k in dom:
            sp.setdefault(k[:2], []).append(k)
        out["c3x_drops"] = {
            "group_layer_records": sum(len(v) for v in x_drops.values()),
            "records_emptied": sum(1 for k, d in dom.items() if x_drops[k] and not (d["c3x"] >= 0).any()),
            "step_passages_emptied": sum(1 for ks in sp.values()
                                         if any(x_drops[k] for k in ks) and not any((dom[k]["c3x"] >= 0).any() for k in ks)),
            "profile_compared_steps": drop_profile(rows, x_drops, compared)}
        if a.draws > 0:
            out["reference"] = {m: run_reference(rows, dom, x_drops, steps, compared, c3_cells, x_cmp, m, a.draws)
                                for m in MATCHES}
    out["meta"] = {**lead_args.header(a.labels, hashlib.sha256(summary.read_bytes()).hexdigest(), a.lead,
                                      reproduces, prim, json.loads(summary.read_text())),
                   "src": str(a.src), "draws": a.draws if a.lead == "c3x" else 0, "matches": list(MATCHES),
                   "git": _git_head(), "python": sys.version.split()[0], "numpy": np.__version__,
                   "runner_dirty": bool(subprocess.run(
                       ["git", "-C", str(REPO), "status", "--porcelain", "--", "tools/run/p10_r7_shuffle.py",
                        "tools/run/p10_r9_lead.py"], capture_output=True, text=True).stdout.strip())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    msg = f"wrote {a.out}; compared steps {compared[0]}–{compared[-1]}"
    if a.lead == "c3x":
        v = out["c3x_vs_c3"]
        msg += f"; c3x vs c3: {v['n_changed']} of {v['n_compared']} cells change, headline {len(v['headline_changed'])}"
        for m, ref in out.get("reference", {}).items():
            msg += (f"; {m}: all {ref['all']['reading']} (q95 {ref['all']['q95']:g}, p {ref['all']['rank_p']:.2f}), "
                    f"headline {ref['headline']['reading']} (q95 {ref['headline']['q95']:g})")
    print(msg)
    return 0


def read(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="p10_r7_shuffle read")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = {"meta": {"git": _git_head(), "src": str(a.src), "labels": str(a.labels)},
           "steps": read_steps(records_by_step(a.src), a.labels)}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'step':>8} {'col':>3} {'n':>5} {'token':>6} {'bag':>5} {'order':>6} {'label':>8}  "
          + " ".join(f"{lv:>6}" for lv in [f"b{b}" for b in BLOCKS] + ["alone"]))
    for step, v in res["steps"].items():
        t = v[v["primary"]]
        print(f"{step:>8} {v['primary']:>3} {t['n']:>5} {t['token-borne']:>6} {t['bag-borne']:>5} "
              f"{t['order-borne']:>6} {t['label']:>8}  "
              + " ".join(f"{x:6.2f}" if x is not None else "     -" for x in t["survival_share"].values()))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"run": run, "read": read, "reread": reread}
    if not argv or argv[0] not in cmds:
        print("usage: p10_r7_shuffle.py {run,read,reread} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
