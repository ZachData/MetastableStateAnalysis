"""Phase 10 re-read, R0: the label source every re-read row takes
(`p10_cluster_function/design-10.md` "Inputs, fixed here", "The ladder").

Per (step, prompt, layer L1–24) of the 7 v1 passages at the 18 distinct Stage 0
steps, one labelling per ladder column. Inputs, one set for every row:

- Stage 0's stored ``activations.npz``, the run dir selected through
  ``stage0_index.json`` only;
- unit 1's records re-run at that step (`move_text run --kept-from
  <unit 2's token_sets.json> --stage0-index ...`): the kept tokens, each P = 0
  group's floor and class, and the P = 0 match against the same Stage 0 run;
- at step 143000, unit 2's ``learned`` per group, from `candidates read`'s
  ``candidate_rows.json`` (joined by member set).

THE COLUMNS (labels over the **kept** positions, in their order; −1 = kept,
in no group of the column; positions outside ``kept`` are outside the domain,
not −1):

====== ==================================================================
c0     the stored labels (`backfill_hdbscan.read_labels`), over **every**
       stored position: the old partition, for readers re-run unchanged.
       Stored from **float32** cosine distances (Phase 1's `clustering.py`)
c0f    the same shipped call, ``HDBSCAN(min_cluster_size=2)``, on **float64**
       distances (`core.metrics.cosine_distance_matrix`) over every stored
       position: c0 → c0f is the precision alone, c0f → c1 the token rules
       (`/challenge-pr` on #145, finding 1)
c1     shipped ``HDBSCAN(min_cluster_size=2)``, float64, raw (stored) frame
c1c    as c1, centred frame
c2a    level-set groups, centred, size 2 (`admit.layer_groups`: the same
       distances as c1c)
c2b    c2a minus bulk groups (>= ``BULK_SHARE`` of the kept tokens)
c2     c2b minus unstable groups (unit 1's floor: median < 0.5, or J0 = 0)
c3     c2 minus groups whose unit 1 class (own floor, EOD join) is not
       ``moves``: **the working definition** (`design-1d.md`)
c3_c4  the arm: c3's filters on level-set groups, centred, size 4
c3_r2  the arm: c3's filters on level-set groups, raw frame, size 2
====== ==================================================================

c2a–c3 keep c2a's group ids, so a group is the same id in every column it
survives to. ``learned`` (step 143000 only): for each c3 id, whether unit 2
calls it learned. At other steps it is not written: unit 2's bars are stored,
but each group's ``s`` needs the Gaussian draws at that step's cloud, which
nothing has run (`design-10.md` "learned" row: "otherwise 143000 only").

REFUSALS (refuse rather than degrade). A (step, prompt) is not written, and is
listed under ``refused`` with why, when: unit 1's record is missing, is not on
the given token set, or carries no passing ``p0_match``; its Stage 0 run is not
the one the match was taken against; or the stored activations' token count
differs. A (step, prompt, layer) is refused when the level-set groups
recomputed here from Stage 0's activations (centred 2, centred 4, raw 2) are
not the member sets of unit 1's P = 0 groups: the filters are unit 1's, so the
groups must be its groups. `load_column` refuses a refused or missing record,
an unknown column, and ``learned`` off step 143000.

``extend`` (R9, `design-10.md` "R9"; Blocked 27 (a)) writes a new source: R0's step files with one
more column per layer, **c3x** = c3 minus the groups R8x's rows (`p10_r8x_exact.py`) do not mark
``c3x`` (each EOD condition's J >= max(J0, its own chance level)), every other column unchanged;
``c3x_learned`` / ``c3x_unlearned`` at step 143000 as c3's split. It refuses unless R8x's rows
mark exactly c3's groups ``c3``, every ``c3x`` row is a c3 group, and each such group's member
count is the row's ``size``.

Run ``build`` per step as unit 1's records arrive (resumable: a step's file is
skipped if present), then ``summary`` (counts per column, the readable share,
and, given ``--definitions``, c3's group-layer records at steps 0 and 143000
against `candidates definitions`' "moves" counts).

The re-read's readers take one column by ``--labels <dir> --column <c>`` (`add_reader_args`)
and read it through `reader_input`: full-length labels with ``OUTSIDE`` off the domain,
readable records only, and two more columns at step 143000, ``c3_learned`` / ``c3_unlearned``
(c3 with the other groups' members moved to the rest).

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.holdout import HELD_OUT_PROMPT_KEYS
from p1d_cluster_ensemble.arch_null import BULK_SHARE
from p1d_cluster_ensemble.move_text import (FLOOR_ZERO, MODELS, V1_PASSAGES, band, group_classes,
                                            stage0_runs)

COLUMNS = ("c0", "c0f", "c1", "c1c", "c2a", "c2b", "c2", "c3", "c3_c4", "c3_r2")
#: Columns ``extend`` adds to a built source (R9); absent from R0's own.
EXTENDED = ("c3x",)
ALL_COLUMNS = COLUMNS + EXTENDED
LAYERS = tuple(range(1, 25))
LEARNED_STEP = "step143000"
#: Columns over every stored position; the rest are over the kept positions.
ALL_POSITIONS = ("c0", "c0f")
#: A record is readable with at least this many members and this many of the rest
#: (`design-10.md` "Readable"), placed there.
READABLE_MIN = 10
#: Unit 1's classes that are not "stable" under the definition.
NOT_STABLE = ("unstable", FLOOR_ZERO)
#: The arms: (column, frame, min_cluster_size) of unit 1's record.
GROUP_CELLS = (("c3", "centred", 2), ("c3_c4", "centred", 4), ("c3_r2", "raw", 2))


class LabelSourceError(RuntimeError):
    """A label-source read that would degrade instead of refusing."""


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------

def same_groups(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    """Whether two group lists hold the same member sets (order free)."""
    return Counter(frozenset(map(int, g)) for g in a) == Counter(frozenset(map(int, g)) for g in b)


def labels_of(n: int, groups: Sequence[Sequence[int]], keep: Optional[Sequence[bool]] = None) -> List[int]:
    """Group ``i``'s members get label ``i`` if ``keep[i]``; every other index is −1."""
    lab = np.full(n, -1, dtype=int)
    for i, g in enumerate(groups):
        if keep is None or keep[i]:
            if np.any(lab[np.asarray(g, dtype=int)] >= 0):
                raise ValueError("groups overlap")
            lab[np.asarray(g, dtype=int)] = i
    return lab.tolist()


def ladder_keeps(sizes: Sequence[int], classes: Sequence[str], n_kept: int) -> Dict[str, List[bool]]:
    """
    Which of a cell's level-set groups survive each filter, cumulatively:
    ``bulk`` out (c2b), then not stable (c2), then not ``moves`` (c3).
    ``classes`` are `move_text.group_classes` (own floor, EOD join).
    """
    nb = [s < BULK_SHARE * n_kept for s in sizes]
    st = [a and c not in NOT_STABLE for a, c in zip(nb, classes)]
    mv = [a and c == "moves" for a, c in zip(st, classes)]
    return {"c2b": nb, "c2": st, "c3": mv}


def readable(labels: Sequence[int], min_n: int = READABLE_MIN) -> bool:
    lab = np.asarray(labels)
    return bool(np.sum(lab >= 0) >= min_n and np.sum(lab < 0) >= min_n)


# ---------------------------------------------------------------------------
# One (step, prompt)
# ---------------------------------------------------------------------------

def _layer_groups(Y: np.ndarray, frame: str, mcs: int, shipped: bool):
    """Level-set groups (index lists) on ``Y`` in ``frame``, and the shipped call's labels if asked."""
    from p1d_cluster_ensemble.admit import layer_groups, level_set_hdbscan
    from p1d_cluster_ensemble.gaussian_null import frame_vectors
    from p1d_cluster_ensemble.methods import LayerData
    Z, _ = frame_vectors(Y, frame)
    if shipped:
        _, rows, check = layer_groups(Z, mcs)
        return [r["members"] for r in rows], check["labels"]
    _, rows = level_set_hdbscan(LayerData.from_normed(Z).cos_dist, mcs)
    return [r["members"] for r in rows], None


def shipped_f64(X: np.ndarray) -> List[int]:
    """The stored partition's call (shipped HDBSCAN, size 2) on float64 cosine distances of every row."""
    import hdbscan
    from core.metrics import cosine_distance_matrix
    return hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed").fit_predict(
        cosine_distance_matrix(X)).astype(int).tolist()


def build_prompt(step: str, prompt: str, u1: Dict, run_dir: Path, learned_rows: Optional[Dict]) -> Dict:
    """One (step, prompt): the columns per layer, or ``{"refused": why}``."""
    from tools.run.backfill_hdbscan import read_labels
    m = u1.get("p0_match") or {}
    if not m.get("ok"):
        return {"refused": f"unit 1's record has no passing p0_match: {m}"}
    if Path(m["run_dir"]) != Path(run_dir):
        return {"refused": f"p0_match was taken against {m['run_dir']}, the index gives {run_dir}"}
    a = np.load(run_dir / "activations.npz")["activations"]
    if a.shape[1] != u1["n_passage"]:
        return {"refused": f"Stage 0 has {a.shape[1]} tokens, unit 1's passage {u1['n_passage']}"}
    c0 = read_labels(run_dir)
    kept = np.asarray(u1["kept_offsets"], dtype=int)
    nk = int(kept.size)
    by_cell = {(lay["layer"], lay["frame"]): lay for lay in u1["layers"]}
    layers, refused = {}, {}
    for L in LAYERS:
        if L not in c0:
            refused[str(L)] = "no stored c0 labels"
            continue
        Y = a[L][kept]
        cols: Dict = {"c0": c0[L].astype(int).tolist(), "c0f": shipped_f64(a[L])}
        bad = []
        for col, frame, mcs in GROUP_CELLS:
            rec = by_cell[(L, frame)]["mcs"][str(mcs)]
            u1_groups = [g["offsets"] for g in rec["groups"]]
            groups, shipped = _layer_groups(Y, frame, mcs, shipped=(mcs == 2))
            off = [kept[np.asarray(g, dtype=int)].tolist() for g in groups]
            if not same_groups(off, u1_groups):
                bad.append(f"{frame}/{mcs}: {len(off)} groups here, {len(u1_groups)} in unit 1")
                continue
            # unit 1's order, so its classes line up with the groups
            pos = {frozenset(o): i for i, o in enumerate(off)}
            order = [pos[frozenset(g)] for g in u1_groups]
            groups = [groups[i] for i in order]
            keeps = ladder_keeps([len(g) for g in groups], group_classes(rec), nk)
            if shipped is not None:
                cols["c1" if frame == "raw" else "c1c"] = shipped
            if col == "c3":
                cols["c2a"] = labels_of(nk, groups)
                for k in ("c2b", "c2", "c3"):
                    cols[k] = labels_of(nk, groups, keeps[k])
                if step == LEARNED_STEP:
                    lr = learned_rows.get((prompt, L)) if learned_rows is not None else None
                    if lr is None:
                        bad.append("no unit 2 rows to join for learned")
                        continue
                    lrn = {}
                    for i, g in enumerate(u1_groups):
                        if keeps["c3"][i]:
                            st = lr.get(frozenset(g))
                            if st is None:
                                bad.append(f"c3 group {i} not in unit 2's rows")
                                break
                            lrn[str(i)] = st in ("learned+replicates", "learned only")
                    cols["learned"] = lrn
            else:
                cols[col] = labels_of(nk, groups, keeps["c3"])
        if not bad:
            layers[str(L)] = cols
        else:
            refused[str(L)] = "; ".join(bad)
    return {"kept": kept.tolist(), "n_positions": int(a.shape[1]), "stage0_run": str(run_dir),
            "p0_match": m, "layers": layers, "refused_layers": refused}


def _job(args) -> Tuple[str, Dict]:
    step, prompt, u1_path, run_dir, learned_rows = args
    u1 = json.loads(Path(u1_path).read_text())
    return prompt, build_prompt(step, prompt, u1, Path(run_dir), learned_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def unit1_hashes(unit1: Path, step: str) -> Dict[str, Optional[str]]:
    """sha256 (16 hex) of each passage's unit 1 record at ``step``, None if absent: a step file
    is reused on resume only if these are unchanged (CodeRabbit on #145)."""
    return {p: (_sha(unit1 / step / f"{p}.json") if (unit1 / step / f"{p}.json").exists() else None)
            for p in V1_PASSAGES}


def learned_index(rows_path: Path) -> Dict[Tuple[str, int], Dict[frozenset, str]]:
    """``{(prompt, layer): {members: status}}`` of unit 2's seed-0 groups at step 143000, centred size 2."""
    rows = json.loads(rows_path.read_text())[LEARNED_STEP]
    out: Dict = {}
    for r in rows:
        if r["frame"] == "centred" and r["size"] == 2:
            out.setdefault((r["prompt"], r["layer"]), {})[frozenset(r["members"])] = r["status"]
    return out


def build(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="p10_label_source build")
    ap.add_argument("--unit1", type=Path, required=True, help="move_text output, run with --kept-from and --stage0-index")
    ap.add_argument("--token-sets", type=Path, required=True, help="the token_sets.json unit 1 was run on")
    ap.add_argument("--rows", type=Path, required=True, help="`candidates read`'s candidate_rows.json (learned, step 143000)")
    ap.add_argument("--index", type=Path, default=DATA / "phase12" / "stage0_logs" / "stage0_index.json")
    ap.add_argument("--steps", nargs="+", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args(argv)
    held = sorted(set(V1_PASSAGES) & set(HELD_OUT_PROMPT_KEYS))
    if held:   # the 12 v2 prompts stay held out on 410m (`core/holdout.py`)
        print(f"refusing: held-out prompts in the passage list: {held}", file=sys.stderr)
        return 1
    ts_sha = _sha(args.token_sets)
    sets = json.loads(args.token_sets.read_text())["sets"]
    runs = stage0_runs(args.index)
    lrows = None
    args.out.mkdir(parents=True, exist_ok=True)
    status = 0
    for step in args.steps:
        f = args.out / f"{step}.json"
        if f.exists():
            m = json.loads(f.read_text())["meta"]
            same = (m["token_sets"]["sha256"] == ts_sha and m["index"]["sha256"] == _sha(args.index)
                    and m["unit1"] == str(args.unit1) and tuple(m["columns"]) == COLUMNS
                    and m.get("unit1_records") == unit1_hashes(args.unit1, step))
            if not same:
                print(f"refusing: {f} was built on other inputs or columns; move it aside to rebuild",
                      file=sys.stderr)
                status = 1
            else:
                print(f"already {f}", flush=True)
            continue
        n = int(step.removeprefix("step"))
        jobs, refused = [], {}
        for p in V1_PASSAGES:
            up = args.unit1 / step / f"{p}.json"
            if not up.exists():
                refused[p] = f"no unit 1 record {up}"
                continue
            u1 = json.loads(up.read_text())
            kf = u1["meta"].get("kept_from") or {}
            if kf.get("sha256") != ts_sha or u1["kept_offsets"] != sets[p]["kept"]:
                refused[p] = "unit 1's record is not on the given token set"
                continue
            if (n, p) not in runs:
                refused[p] = "no Stage 0 run in the index"
                continue
            if step == LEARNED_STEP and lrows is None:
                lrows = learned_index(args.rows)
            lr = None if step != LEARNED_STEP else {k: v for k, v in lrows.items() if k[0] == p}
            jobs.append((step, p, str(up), str(runs[(n, p)]), lr))
        if args.workers > 1 and len(jobs) > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
                done = pool.map(_job, jobs, chunksize=1)
        else:
            done = [_job(j) for j in jobs]
        prompts = {}
        for p, rec in done:
            if "refused" in rec:
                refused[p] = rec["refused"]
            else:
                prompts[p] = rec
        res = {"meta": {"git": _git_head(), "step": step, "columns": COLUMNS,
                        "unit1": str(args.unit1), "unit1_records": unit1_hashes(args.unit1, step),
                        "token_sets": {"path": str(args.token_sets), "sha256": ts_sha},
                        "index": {"path": str(args.index), "sha256": _sha(args.index)},
                        "rows": {"path": str(args.rows), "sha256": _sha(args.rows)} if step == LEARNED_STEP else None,
                        "bulk_share": BULK_SHARE, "readable_min": READABLE_MIN},
               "prompts": prompts, "refused": refused}
        f.write_text(json.dumps(res) + "\n")
        n_ref = sum(len(v["refused_layers"]) for v in prompts.values())
        print(f"done {f}: {len(prompts)} prompts, refused prompts {refused or 'none'}, refused layers {n_ref}",
              flush=True)
        if refused or n_ref:
            status = 1
    return status


def load_step(src: Path, step: str) -> Dict:
    f = Path(src) / f"{step}.json"
    if not f.exists():
        raise LabelSourceError(f"no label source for {step} in {src}")
    return json.loads(f.read_text())


def load_column(src, step: str, prompt: str, layer: int, column: str,
                cache: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    ``(positions, labels)`` of one column at one (step, prompt, layer):
    c0 over every stored position, every other column over the kept positions
    only (its domain). Refuses an unknown column, ``learned`` (read
    `load_learned`), and a refused or missing record.
    """
    if column not in ALL_COLUMNS:
        raise LabelSourceError(f"unknown column {column!r}; one of {ALL_COLUMNS}")
    d = (cache if cache is not None else {}).get(step) or load_step(src, step)
    if cache is not None:
        cache[step] = d
    if prompt not in d["prompts"]:
        raise LabelSourceError(f"{step}/{prompt} refused or absent: {d['refused'].get(prompt, 'absent')}")
    p = d["prompts"][prompt]
    if str(layer) not in p["layers"]:
        raise LabelSourceError(f"{step}/{prompt}/L{layer} refused or absent: "
                               f"{p['refused_layers'].get(str(layer), 'absent')}")
    if column not in p["layers"][str(layer)]:
        raise LabelSourceError(f"{step}/{prompt}/L{layer}: no {column} in {src} (written by `extend`)")
    lab = np.asarray(p["layers"][str(layer)][column], dtype=int)
    pos = np.arange(p["n_positions"]) if column in ALL_POSITIONS else np.asarray(p["kept"], dtype=int)
    if lab.size != pos.size:
        raise LabelSourceError(f"{step}/{prompt}/L{layer}/{column}: {lab.size} labels for {pos.size} positions")
    return pos, lab


def load_learned(src, step: str, prompt: str, layer: int) -> Dict[int, bool]:
    """Unit 2's ``learned`` per c3 group id; step 143000 only."""
    if step != LEARNED_STEP:
        raise LabelSourceError(f"learned is written at {LEARNED_STEP} only (module docstring)")
    p = load_step(src, step)["prompts"].get(prompt)
    if p is None or str(layer) not in p["layers"]:
        raise LabelSourceError(f"{step}/{prompt}/L{layer} refused or absent")
    return {int(k): v for k, v in p["layers"][str(layer)]["learned"].items()}


# ---------------------------------------------------------------------------
# For the readers (R1–R3): one column as full-length labels
# ---------------------------------------------------------------------------

#: A position outside a column's domain (not kept). −1 stays "kept, in no group".
OUTSIDE = -2
#: c3 (and c3x, R9) split by unit 2's learned bar (step 143000 only): the other groups' members
#: join the rest. ``learned`` is keyed by c3's ids, and c3x's groups are c3 groups.
SPLITS = {"c3_learned": ("c3", True), "c3_unlearned": ("c3", False),
          "c3x_learned": ("c3x", True), "c3x_unlearned": ("c3x", False)}
READER_COLUMNS = ALL_COLUMNS + tuple(SPLITS)


def add_reader_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--labels", type=Path, default=None,
                    help="the R0 label source (`p10_label_source build`'s --out): the re-read's only input")
    ap.add_argument("--column", choices=READER_COLUMNS, default=None,
                    help="the ladder column to read (`design-10.md` \"The ladder\")")


def reader_input(src, column: str) -> Dict:
    """
    One column for a reader: ``runs`` {(step, prompt): Stage 0 run dir}, ``labels``
    {(step, prompt): {layer: labels over every stored position}} with ``OUTSIDE`` off the
    column's domain, and ``records`` {step: [n, n_readable]}. Only **readable** (prompt,
    layer) records carry labels (`readable` on the column's own domain, `design-10.md`
    "Readable"): an unreadable record is counted, never read as "no effect". Every step file
    under ``src`` is read (a learned column: step 143000 only); a refused prompt or layer
    refuses the whole read.
    """
    src = Path(src)
    if column not in READER_COLUMNS:
        raise LabelSourceError(f"unknown column {column!r}; one of {READER_COLUMNS}")
    split = column in SPLITS
    steps = [LEARNED_STEP] if split else sorted(
        (s for s in MODELS if (src / f"{s}.json").exists()), key=lambda s: int(s.removeprefix("step")))
    if not steps:
        raise LabelSourceError(f"no label source step files in {src}")
    runs, labels, records, gits = {}, {}, {}, {}
    for step in steps:
        cache: Dict = {}
        d = load_step(src, step)
        cache[step] = d
        if d["refused"]:
            raise LabelSourceError(f"{step}: refused prompts {sorted(d['refused'])}")
        n_step = int(step.removeprefix("step"))
        gits[step] = d["meta"]["git"]
        rec = [0, 0]
        for prompt in sorted(d["prompts"]):
            p = d["prompts"][prompt]
            runs[(n_step, prompt)] = Path(p["stage0_run"])
            layers = {}
            for L in LAYERS:
                pos, lab = load_column(src, step, prompt, L, SPLITS[column][0] if split else column, cache)
                if split:
                    lrn = {int(k): v for k, v in p["layers"][str(L)]["learned"].items()}
                    want = SPLITS[column][1]
                    lab = np.where([x >= 0 and lrn[int(x)] == want for x in lab], lab, -1)
                full = np.full(p["n_positions"], OUTSIDE, dtype=int)
                full[pos] = lab
                rec[0] += 1
                if readable(lab):
                    rec[1] += 1
                    layers[L] = full
            labels[(n_step, prompt)] = layers
        records[n_step] = rec
    meta = {"labels": str(src), "column": column, "label_source_git": gits,
            "summary_sha256": _sha(src / "summary.json") if (src / "summary.json").exists() else None,
            "readable_min": READABLE_MIN, "outside": OUTSIDE}
    return {"runs": runs, "labels": labels, "records": records, "meta": meta}


def step_summary(d: Dict) -> Dict:
    """Per column: group-layer records by band, records (prompt, layer) readable, and refusals."""
    out = {"refused_prompts": sorted(d["refused"]),
           "refused_layers": sum(len(p["refused_layers"]) for p in d["prompts"].values()), "columns": {}}
    present = [c for c in ALL_COLUMNS
               if all(c in cols for p in d["prompts"].values() for cols in p["layers"].values())]
    for col in present:
        groups, n_rec, n_read = Counter(), 0, 0
        for p in d["prompts"].values():
            kept_n = len(p["kept"])
            for L, cols in p["layers"].items():
                lab = np.asarray(cols[col])
                if col in ALL_POSITIONS:
                    lab = lab[np.asarray(p["kept"])]   # read on the kept domain, for comparability
                ids = set(lab[lab >= 0].tolist())
                groups[band(int(L))] += len(ids)
                n_rec += 1
                n_read += readable(lab)
                assert lab.size == kept_n
        out["columns"][col] = {"groups": sum(groups.values()), "by_band": dict(sorted(groups.items())),
                               "records": n_rec, "readable": n_read}
    return out


def c3_member_mismatches(d: Dict, rows: Sequence[Dict]) -> List[Tuple[str, int]]:
    """(prompt, layer) records whose c3 member sets are not the rows' non-bulk ``moves`` groups (centred, size 2)."""
    want: Dict[Tuple[str, int], List] = {}
    for r in rows:
        if r["frame"] == "centred" and r["size"] == 2 and not r["bulk"] and r["class"] == "moves":
            want.setdefault((r["prompt"], r["layer"]), []).append(r["members"])
    out = []
    for prompt, p in d["prompts"].items():
        kept = np.asarray(p["kept"])
        for L, cols in p["layers"].items():
            lab = np.asarray(cols["c3"])
            got = [kept[lab == i].tolist() for i in sorted(set(lab[lab >= 0].tolist()))]
            if not same_groups(got, want.get((prompt, int(L)), [])):
                out.append((prompt, int(L)))
    return out


def summary(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="p10_label_source summary")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--definitions", type=Path, default=None,
                    help="`candidates definitions`' output: c3 at steps 0 / 143000 must give its 'moves' records")
    ap.add_argument("--rows", type=Path, default=None,
                    help="`candidates read`'s candidate_rows.json: c3's member sets at steps 0 / 143000 must be "
                         "its non-bulk 'moves' rows (centred, size 2)")
    args = ap.parse_args(argv)
    res, status = {}, 0
    for step in MODELS:
        f = args.src / f"{step}.json"
        if f.exists():
            d = json.loads(f.read_text())
            res[step] = step_summary(d) | {"git": d["meta"]["git"]}
    if args.definitions is not None:
        defs = json.loads(args.definitions.read_text())["steps"]
        res["definitions_check"] = {}
        for step in ("step0", LEARNED_STEP):
            if step not in res:
                continue
            want = defs[step]["moves"]["centred/2"]
            got = res[step]["columns"]["c3"]
            ok = got["groups"] == want["records"] and got["by_band"] == want["records_by_band"]
            res["definitions_check"][step] = {"c3": got["by_band"], "definitions": want["records_by_band"], "ok": ok}
            status |= 0 if ok else 1
    if args.rows is not None:
        rows = json.loads(args.rows.read_text())
        res["members_check"] = {}
        for step in ("step0", LEARNED_STEP):
            f = args.src / f"{step}.json"
            if not f.exists():
                continue
            mism = c3_member_mismatches(json.loads(f.read_text()), rows[step])
            res["members_check"][step] = {"mismatched_records": len(mism), "first": mism[:3], "ok": not mism}
            status |= 0 if not mism else 1
    (args.src / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print("group-layer records per column; then readable (prompt, layer) records on c2 / c3, of all")
    print(f"{'step':>11s} " + " ".join(f"{c:>6s}" for c in COLUMNS) + "   readable c2 / c3")
    for step, v in res.items():
        if step in ("definitions_check", "members_check"):
            continue
        c = v["columns"]
        print(f"{step:>11s} " + " ".join(f"{c[x]['groups']:6d}" for x in COLUMNS)
              + f"   {c['c2']['readable']} / {c['c3']['readable']} of {c['c3']['records']}"
              + (f"; c3x {c['c3x']['groups']} groups, {c['c3x']['readable']} readable" if "c3x" in c else "")
              + (f"  refused {v['refused_prompts']} layers {v['refused_layers']}"
                 if v["refused_prompts"] or v["refused_layers"] else ""))
    if "definitions_check" in res:
        print("definitions check: " + json.dumps(res["definitions_check"]))
    if "members_check" in res:
        print("members check: " + json.dumps(res["members_check"]))
    return status


def c3x_layer(cols: Dict, rows: Sequence[Dict], where: str) -> List[int]:
    """c3x over the kept positions: c3's labels with the groups R8x does not mark ``c3x`` moved to −1."""
    lab = np.asarray(cols["c3"], dtype=int)
    c3_ids = set(lab[lab >= 0].tolist())
    marked = {r["id"] for r in rows if r["c3"]}
    if marked != c3_ids:
        raise LabelSourceError(f"{where}: R8x marks c3 groups {sorted(marked)}, the source's c3 holds {sorted(c3_ids)}")
    keep = set()
    for r in rows:
        if r["c3x"]:
            if r["id"] not in c3_ids:
                raise LabelSourceError(f"{where}: R8x's c3x group {r['id']} is not a c3 group")
            if int((lab == r["id"]).sum()) != r["size"]:
                raise LabelSourceError(f"{where}: c3x group {r['id']} has {int((lab == r['id']).sum())} "
                                       f"members here, R8x's size {r['size']}")
            keep.add(r["id"])
    return np.where(np.isin(lab, sorted(keep)), lab, -1).tolist()


def extend(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="p10_label_source extend")
    ap.add_argument("--labels", type=Path, required=True, help="R0's label source")
    ap.add_argument("--r8x", type=Path, required=True, help="R8x's output dir (rows/<step>/<prompt>.json)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.resolve() == args.labels.resolve():
        raise SystemExit("refusing: --out is the source itself")
    if not (args.r8x / "r8x.json").exists():
        raise SystemExit(f"refusing: no r8x.json in {args.r8x}")
    r8x_sha = _sha(args.r8x / "r8x.json")
    args.out.mkdir(parents=True, exist_ok=True)
    missing = [s for s in MODELS if not (args.labels / f"{s}.json").exists()]
    if missing:
        raise SystemExit(f"refusing: no label source for {missing} in {args.labels}")
    for step in MODELS:
        f = args.labels / f"{step}.json"
        d = json.loads(f.read_text())
        rows_sha = hashlib.sha256()
        if d["refused"]:
            raise SystemExit(f"refusing: {step} has refused prompts {sorted(d['refused'])}")
        n = [0, 0]
        for prompt, p in d["prompts"].items():
            rf = args.r8x / "rows" / step / f"{prompt}.json"
            if not rf.exists():
                raise SystemExit(f"refusing: no R8x rows for {step}/{prompt}")
            rows_sha.update(rf.read_bytes())
            by_layer: Dict[int, List[Dict]] = {}
            for r in json.loads(rf.read_text())["rows"]:
                by_layer.setdefault(int(r["layer"]), []).append(r)
            for L, cols in p["layers"].items():
                cols["c3x"] = c3x_layer(cols, by_layer.get(int(L), []), f"{step}/{prompt}/L{L}")
                lab = np.asarray(cols["c3x"])
                n[0] += len(set(np.asarray(cols["c3"])[np.asarray(cols["c3"]) >= 0].tolist()))
                n[1] += len(set(lab[lab >= 0].tolist()))
        # the readers name a source by meta.git: the commit that wrote these files (/challenge-pr on #166, finding 3)
        d["meta"]["extended"] = {"from": str(args.labels), "from_sha256": _sha(f), "from_git": d["meta"]["git"],
                                 "r8x": str(args.r8x), "r8x_sha256": r8x_sha,
                                 "r8x_rows_sha256": rows_sha.hexdigest()[:16], "columns": list(EXTENDED)}
        d["meta"]["git"] = _git_head()
        out = args.out / f"{step}.json"
        out.write_text(json.dumps(d) + "\n")
        back, src = json.loads(out.read_text()), json.loads(f.read_text())
        for prompt, p in src["prompts"].items():                # the copy check, on what was written
            for L, cols in p["layers"].items():
                if any(back["prompts"][prompt]["layers"][L][c] != v for c, v in cols.items()):
                    raise SystemExit(f"refusing: {out} differs from {f} at {prompt}/L{L}")
        print(f"{step}: c3 {n[0]} group-layer records, c3x {n[1]}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"build": build, "summary": summary, "extend": extend}
    if not argv or argv[0] not in cmds:
        print(f"usage: p10_label_source {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
