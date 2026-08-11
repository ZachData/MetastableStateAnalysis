"""
p5_single_mstate_analysis/particle_join.py — the sweep as one particle table,
and the questions that become groupbys once it exists.

Why this module is the point of the rebuild
-------------------------------------------
`core/particles.py` says it plainly: "cluster- and population-level results
become aggregations (groupby / filter) over this table rather than separate
code paths." Everything W1-W5 built produces per-particle numbers; this joins
them into one table keyed by
(model, checkpoint_step, prompt_key, layer, token_position), with the token
set's role and the sweep geometry as extra columns.

What that buys, concretely
--------------------------
status-1's sharpest unexplained result: across all 27 checkpoints the maximum
number of simultaneously-alive clusters holds at 50-55, while mean cluster
lifespan falls 7.0 -> 4.5 and births rise 113 -> 164. Carrying capacity is
invariant; turnover is not.

Cluster-level statistics cannot distinguish the two readings:

  (a) the SAME particles are cycling through clusters faster, or
  (b) DIFFERENT particles are clustering at late checkpoints.

Both produce identical births/deaths/lifespan curves. They are distinguishable
only per particle, and only if particle identity carries across checkpoints —
which is what the frozen token positions and the byte-identical NeoX tokenizer
give. `turnover_decomposition` below is that test, and it is a groupby, not a
new experiment.

`particle_biography` is §7 item 2: per token position, the layer at which it
first joins a stable cluster and how that date moves across the sweep. It is
the same aggregation, read down a different axis.

Complement retained
-------------------
Every token gets a row at every layer, including tokens in no role and tokens
in no cluster (D-10). The unclustered population is what design-5c is about —
"not a failure mode but a distinct phase" — and a table restricted to cluster
members would make it unrecoverable.

Pure numpy, no pandas (matching core/particles.py's stated choice).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.run_discovery import RunRef, sweep_for_prompt
from .token_sets import TokenSet

__all__ = [
    "GEOMETRY_COLUMNS",
    "ROLE_COLUMNS",
    "build_layer_table",
    "build_sweep_particle_table",
    "particle_biography",
    "clustered_set_overlap",
    "turnover_decomposition",
    "biography_report_lines",
    "turnover_report_lines",
]


# Per-particle geometry carried as `extra` columns. Every table built here
# carries all of them — `ParticleTable.concat` fills a missing extra with
# np.nan, which would silently coerce a string column ("primary") to the
# literal text "nan" on concatenation. Uniform extras avoid that entirely.
GEOMETRY_COLUMNS = (
    "cos_to_centroid",
    "particle_norm",
    "norm_z",
    "rank_contribution_normed",
    "rank_contribution_raw",
)
ROLE_COLUMNS = ("token_set_role", "in_token_set")

_ROLE_NONE = "none"


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _role_array(n_tokens: int, token_set: TokenSet) -> Tuple[np.ndarray, np.ndarray]:
    roles = np.full(n_tokens, _ROLE_NONE, dtype=object)
    for role, positions in (
        ("control", token_set.control_positions),
        ("sibling", token_set.sibling_positions),
        ("primary", token_set.positions),      # last wins on any overlap
    ):
        for p in positions:
            if 0 <= int(p) < n_tokens:
                roles[int(p)] = role
    roles = np.array([str(r) for r in roles])   # fixed-width unicode, not object
    in_set = (roles == "primary").astype(np.int64)
    return roles, in_set


def build_layer_table(
    model: str,
    checkpoint_step: Optional[int],
    prompt_key: str,
    layer: int,
    cluster_labels: Sequence[int],
    token_set: TokenSet,
    geometry_rows: Optional[Sequence[dict]] = None,
    token_str: Optional[Sequence[str]] = None,
    ParticleTable=None,
):
    """
    One (model, checkpoint, prompt, layer) worth of rows.

    `geometry_rows` are `sweep_geometry.layer_geometry`'s per-particle dicts
    for this layer. Positions absent from them — every token outside the three
    roles — get NaN, not 0.0. Those tokens were not measured; a zero would
    read as "measured, contributes nothing", and the difference matters most
    for exactly the unclustered complement this table exists to retain.
    """
    if ParticleTable is None:
        from core.particles import ParticleTable  # noqa: F811

    labels = np.asarray(cluster_labels, dtype=np.int64)
    n = int(labels.shape[0])

    roles, in_set = _role_array(n, token_set)

    geom: Dict[str, np.ndarray] = {
        c: np.full(n, np.nan, dtype=np.float64) for c in GEOMETRY_COLUMNS
    }
    for row in (geometry_rows or []):
        p = int(row.get("token_position", -1))
        if not (0 <= p < n):
            continue
        geom["cos_to_centroid"][p] = row.get("cos_to_centroid", np.nan)
        geom["particle_norm"][p] = row.get("norm", np.nan)
        geom["norm_z"][p] = row.get("norm_z", np.nan)
        geom["rank_contribution_normed"][p] = row.get(
            "rank_contribution_normed", np.nan)
        geom["rank_contribution_raw"][p] = row.get(
            "rank_contribution_raw", np.nan)

    tokens = list(token_str) if token_str is not None else None
    if tokens is not None and len(tokens) != n:
        tokens = None

    return ParticleTable.from_layer(
        model=model,
        prompt_key=prompt_key,
        layer=int(layer),
        cluster_labels=labels,
        checkpoint_step=checkpoint_step,
        token_str=tokens,
        extra={"token_set_role": roles, "in_token_set": in_set, **geom},
    )


def build_sweep_particle_table(
    token_set: TokenSet,
    refs: Sequence[RunRef],
    run_loader: Callable[[Path], dict],
    geometry: Optional[dict] = None,
    steps: Optional[Sequence[int]] = None,
    ParticleTable=None,
):
    """
    One table for a frozen token set across the whole sweep.

    `run_loader` is `anchors.load_run_for_selection` (labels + tokens, no
    activations). `geometry` is `sweep_geometry.sweep_geometry`'s output, or
    None to build the table with role columns only and NaN geometry — useful
    because the labels-only table costs seconds while the geometry pass costs
    minutes, and the turnover question below needs only the labels.

    Returns (table, skipped). A checkpoint whose token count disagrees with
    the anchor's is skipped: the frozen positions no longer identify the same
    particles, and rows built from them would silently be about other tokens.
    """
    if ParticleTable is None:
        from core.particles import ParticleTable  # noqa: F811

    geom_by_step: Dict[int, List[dict]] = {}
    if geometry:
        for rec in geometry.get("records", []):
            geom_by_step[int(rec["step"])] = rec["layers"]

    sweep = sweep_for_prompt(refs, token_set.prompt_key)
    if steps is not None:
        want = {int(s) for s in steps}
        sweep = [r for r in sweep if r.step in want]

    tables = []
    skipped: List[dict] = []

    for ref in sweep:
        run = run_loader(Path(ref.run_dir))
        if not run:
            skipped.append({"step": ref.step,
                            "reason": f"no usable clustering in {ref.run_dir}"})
            continue
        labels = run["hdbscan_labels"]
        n_tokens = int(run["n_tokens"])
        if token_set.n_tokens_prompt and n_tokens != token_set.n_tokens_prompt:
            skipped.append({
                "step": ref.step,
                "reason": (f"token count {n_tokens} != anchor's "
                           f"{token_set.n_tokens_prompt}; positions no longer "
                           "identify the same particles"),
            })
            continue

        layer_geoms = geom_by_step.get(int(ref.step) if ref.step is not None else -1)
        for li, lab in enumerate(labels):
            rows = None
            if layer_geoms is not None and li < len(layer_geoms):
                rows = layer_geoms[li].get("particles")
            tables.append(build_layer_table(
                model=ref.model,
                checkpoint_step=ref.step,
                prompt_key=token_set.prompt_key,
                layer=li,
                cluster_labels=lab,
                token_set=token_set,
                geometry_rows=rows,
                token_str=run.get("tokens") or None,
                ParticleTable=ParticleTable,
            ))

    if not tables:
        return ParticleTable.concat([]), skipped
    return ParticleTable.concat(tables), skipped


# ---------------------------------------------------------------------------
# Biography — the groupby
# ---------------------------------------------------------------------------

def _longest_run(mask: np.ndarray) -> int:
    """Longest consecutive True run. Distinct from the total count: a particle
    clustered at layers 2,3,4 and one clustered at 2,7,15 have the same total
    and very different behaviour, and only the run length distinguishes a
    persistent membership from a flickering one."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def particle_biography(table) -> List[dict]:
    """
    One record per (checkpoint_step, token_position): when this particle was
    in a cluster, for how long, and how many distinct clusters it passed
    through.

    Fields
    ------
    first_clustered_layer / last_clustered_layer : None when never clustered.
        None, not -1 — a sentinel that sorts before layer 0 would silently
        make never-clustered particles look like the earliest joiners in any
        aggregate that forgets to filter.
    n_layers_clustered : total layers with a non-noise label.
    longest_run        : longest consecutive stretch (see `_longest_run`).
    n_distinct_labels  : how many different clusters it belonged to. High
        with a high n_layers_clustered is a particle passing through many
        clusters; that is the per-particle form of the turnover status-1
        measures only in aggregate.
    role               : its token-set role, carried through.
    """
    if len(table) == 0:
        return []

    steps = np.asarray(table.columns["checkpoint_step"])
    pos = np.asarray(table.columns["token_position"])
    layer = np.asarray(table.columns["layer"])
    labels = np.asarray(table.columns["cluster_label"])
    roles = np.asarray(table.extra.get(
        "token_set_role", np.array([_ROLE_NONE] * len(table))))
    tokens = np.asarray(table.columns.get(
        "token_str", np.array([""] * len(table))))

    order = np.lexsort((layer, pos, steps))
    steps, pos, layer = steps[order], pos[order], layer[order]
    labels, roles, tokens = labels[order], roles[order], tokens[order]

    out: List[dict] = []
    keys = np.stack([steps, pos], axis=1)
    boundaries = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(steps)]])

    for s, e in zip(starts, ends):
        lab = labels[s:e]
        lay = layer[s:e]
        clustered = lab >= 0
        idx = np.flatnonzero(clustered)
        out.append({
            "checkpoint_step": int(steps[s]),
            "token_position": int(pos[s]),
            "token_str": str(tokens[s]),
            "role": str(roles[s]),
            "n_layers": int(e - s),
            "n_layers_clustered": int(clustered.sum()),
            "first_clustered_layer": (int(lay[idx[0]]) if idx.size else None),
            "last_clustered_layer": (int(lay[idx[-1]]) if idx.size else None),
            "longest_run": _longest_run(clustered),
            "n_distinct_labels": int(len(set(lab[clustered].tolist()))),
        })
    return out


def clustered_set_overlap(
    bio: Sequence[dict],
    step_a: int,
    step_b: int,
    min_layers_clustered: int = 1,
) -> dict:
    """
    Jaccard between the sets of particles that clustered at two checkpoints.

    `min_layers_clustered` raises the bar from "touched a cluster once" to
    "was a cluster member for at least k layers". At k=1 nearly every token
    qualifies at a trained checkpoint and the Jaccard saturates near 1,
    telling you nothing; the interesting comparison is at a k above the noise
    floor. Reported at several k by `turnover_decomposition` rather than
    fixed at one, because where the Jaccard starts to fall IS the answer.

    **Empty side means None, not 0.0.** If k exceeds a checkpoint's maximum
    per-particle persistence, that side's set is empty and the Jaccard is
    mechanically 0 — which reads identically to "completely different
    particles" while actually meaning "the threshold is above this
    checkpoint's ceiling". That confusion is exactly the failure mode this
    decomposition exists to avoid, and it showed up on the first worked
    example: a sweep where the SAME particles clustered throughout reported
    J(k=8)=0.000 between step 0 and step 512, purely because nobody at 512
    persisted for 8 layers. `degenerate` names which side emptied.
    """
    def _set(step):
        return {r["token_position"] for r in bio
                if r["checkpoint_step"] == step
                and r["n_layers_clustered"] >= min_layers_clustered}

    a, b = _set(step_a), _set(step_b)
    union = a | b
    degenerate = None
    if not a and not b:
        degenerate = "both"
    elif not a:
        degenerate = "a"
    elif not b:
        degenerate = "b"
    return {
        "step_a": step_a, "step_b": step_b,
        "min_layers_clustered": min_layers_clustered,
        "n_a": len(a), "n_b": len(b), "n_intersection": len(a & b),
        "degenerate": degenerate,
        "jaccard": (round(len(a & b) / len(union), 4)
                    if union and degenerate is None else None),
    }


def turnover_decomposition(
    bio: Sequence[dict],
    thresholds: Sequence[int] = (1, 3, 5, 8),
) -> dict:
    """
    The §7 item 1 test: same particles cycling faster, or different particles?

    status-1 reports carrying capacity flat at 50-55 alive clusters while mean
    lifespan falls 7.0 -> 4.5. Two readings produce that identically at the
    cluster level. Per particle they do not:

      **same particles, faster cycling** — the SET of clustering particles is
          stable across checkpoints (Jaccard stays high at every threshold),
          per-particle `n_layers_clustered` falls, and `n_distinct_labels`
          rises. The same tokens are passing through more clusters, each for
          less time.

      **different particles** — the Jaccard falls with checkpoint distance,
          especially at higher thresholds. Late checkpoints are clustering a
          different subset of the prompt.

    Returned per consecutive checkpoint pair, plus first-vs-last. The
    per-particle rank correlation of `n_layers_clustered` is the sharpest
    single number: high means the same particles are the persistent ones even
    as absolute persistence drops (reading 1); low means the persistent set
    itself is being reshuffled (reading 2).

    No verdict is returned. The two readings are not exhaustive and the
    numbers can land between them; naming which one "won" would be the kind
    of premature collapse this rebuild keeps finding in the old results.
    """
    steps = sorted({r["checkpoint_step"] for r in bio})
    if len(steps) < 2:
        return {"steps": steps, "pairs": [], "note": "need >= 2 checkpoints"}

    by_step: Dict[int, Dict[int, dict]] = {}
    for r in bio:
        by_step.setdefault(r["checkpoint_step"], {})[r["token_position"]] = r

    def _pair(a: int, b: int) -> dict:
        common = sorted(set(by_step[a]) & set(by_step[b]))
        la = np.array([by_step[a][p]["n_layers_clustered"] for p in common],
                      dtype=np.float64)
        lb = np.array([by_step[b][p]["n_layers_clustered"] for p in common],
                      dtype=np.float64)
        # Distinct-label means over EVERY token are diluted by the
        # never-clustered complement, which contributes a structural 0 and
        # drags the mean toward it — on the first worked example both
        # checkpoints reported 0.40 while the clustered particles' actual
        # values were unchanged at 1.0. "Cycling through more clusters" is a
        # claim about particles that cluster, so it is measured on them.
        da = np.array([by_step[a][p]["n_distinct_labels"] for p in common],
                      dtype=np.float64)
        db = np.array([by_step[b][p]["n_distinct_labels"] for p in common],
                      dtype=np.float64)
        ca = np.array([by_step[a][p]["n_distinct_labels"] for p in common
                       if by_step[a][p]["n_layers_clustered"] > 0],
                      dtype=np.float64)
        cb = np.array([by_step[b][p]["n_distinct_labels"] for p in common
                       if by_step[b][p]["n_layers_clustered"] > 0],
                      dtype=np.float64)
        ka = np.array([by_step[a][p]["n_layers_clustered"] for p in common
                       if by_step[a][p]["n_layers_clustered"] > 0],
                      dtype=np.float64)
        kb = np.array([by_step[b][p]["n_layers_clustered"] for p in common
                       if by_step[b][p]["n_layers_clustered"] > 0],
                      dtype=np.float64)
        return {
            "step_a": a, "step_b": b,
            "n_common_particles": len(common),
            "jaccard_by_threshold": {
                int(k): clustered_set_overlap(bio, a, b, k)["jaccard"]
                for k in thresholds
            },
            "mean_layers_clustered_a": round(float(la.mean()), 3) if la.size else None,
            "mean_layers_clustered_b": round(float(lb.mean()), 3) if lb.size else None,
            "mean_distinct_labels_a": round(float(da.mean()), 3) if da.size else None,
            "mean_distinct_labels_b": round(float(db.mean()), 3) if db.size else None,
            "n_clustered_a": int(ca.size),
            "n_clustered_b": int(cb.size),
            "mean_distinct_labels_among_clustered_a": (
                round(float(ca.mean()), 3) if ca.size else None),
            "mean_distinct_labels_among_clustered_b": (
                round(float(cb.mean()), 3) if cb.size else None),
            "mean_layers_among_clustered_a": (
                round(float(ka.mean()), 3) if ka.size else None),
            "mean_layers_among_clustered_b": (
                round(float(kb.mean()), 3) if kb.size else None),
            "rank_corr_layers_clustered": _spearman(la, lb),
        }

    pairs = [_pair(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]
    return {
        "steps": steps,
        "thresholds": list(thresholds),
        "pairs": pairs,
        "first_vs_last": _pair(steps[0], steps[-1]),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Spearman rank correlation, average-ranked so ties (very common in
    integer layer counts) do not bias it. None when either side is constant —
    the correlation is undefined there, and 0.0 would read as "unrelated"
    rather than "not computable"."""
    if a.size < 3 or b.size < 3:
        return None
    ra, rb = _avg_rank(a), _avg_rank(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def _avg_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # Average ranks within tied groups.
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i, j + 1))
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def biography_report_lines(bio: Sequence[dict],
                           role: Optional[str] = "primary") -> List[str]:
    """Per-checkpoint summary of when particles join clusters and for how
    long. Restricted to one role by default — averaged over every token in the
    prompt the numbers are dominated by the complement."""
    rows = [r for r in bio if role is None or r["role"] == role]
    if not rows:
        return [f"(no particles with role={role!r})"]

    steps = sorted({r["checkpoint_step"] for r in rows})
    lines = [
        f"particle biography — role={role} n_particles="
        f"{len({r['token_position'] for r in rows})}",
        f"{'step':>8} {'n_clust':>8} {'mean_layers':>12} {'mean_first':>11} "
        f"{'mean_run':>9} {'mean_labels':>12}",
    ]
    for s in steps:
        rs = [r for r in rows if r["checkpoint_step"] == s]
        firsts = [r["first_clustered_layer"] for r in rs
                  if r["first_clustered_layer"] is not None]
        n_clust = sum(1 for r in rs if r["n_layers_clustered"] > 0)
        lines.append(
            f"{s:>8} {n_clust:>8} "
            f"{np.mean([r['n_layers_clustered'] for r in rs]):>12.2f} "
            + (f"{np.mean(firsts):>11.2f} " if firsts else f"{'n/a':>11} ")
            + f"{np.mean([r['longest_run'] for r in rs]):>9.2f} "
            f"{np.mean([r['n_distinct_labels'] for r in rs]):>12.2f}"
        )
    return lines


def turnover_report_lines(turn: dict) -> List[str]:
    """The carrying-capacity question, laid out. Deliberately ends without a
    verdict: the two readings are not exhaustive."""
    pairs = turn.get("pairs", [])
    if not pairs:
        return [f"(turnover not computable: {turn.get('note', 'no pairs')})"]

    ks = turn["thresholds"]
    lines = [
        "turnover decomposition — same particles cycling faster, or "
        "different particles?",
        f"{'a':>8} {'b':>8} " + " ".join(f"{'J(k=' + str(k) + ')':>10}" for k in ks)
        + f" {'n_cl_a':>7} {'n_cl_b':>7} {'lay_a':>7} {'lay_b':>7}"
          f" {'lab_a':>7} {'lab_b':>7} {'rho':>7}",
    ]

    def _row(p):
        js = " ".join(
            (f"{p['jaccard_by_threshold'][k]:>10.3f}"
             if p["jaccard_by_threshold"].get(k) is not None else f"{'n/a':>10}")
            for k in ks)
        rho = p["rank_corr_layers_clustered"]

        def _f(key):
            v = p.get(key)
            return f"{v:>7.2f}" if v is not None else f"{'n/a':>7}"

        return (
            f"{p['step_a']:>8} {p['step_b']:>8} {js} "
            f"{p['n_clustered_a']:>7} {p['n_clustered_b']:>7} "
            + _f("mean_layers_among_clustered_a") + " "
            + _f("mean_layers_among_clustered_b") + " "
            + _f("mean_distinct_labels_among_clustered_a") + " "
            + _f("mean_distinct_labels_among_clustered_b") + " "
            + (f"{rho:>7.3f}" if rho is not None else f"{'n/a':>7}")
        )

    for p in pairs:
        lines.append(_row(p))
    lines.append("first vs last:")
    lines.append(_row(turn["first_vs_last"]))

    degen = sorted({
        k for p in pairs + [turn["first_vs_last"]]
        for k in ks
        if p["jaccard_by_threshold"].get(k) is None
    })
    if degen:
        lines.append(
            f"  [note] J is n/a at k={degen} for at least one pair: the "
            "threshold exceeds a checkpoint's maximum per-particle "
            "persistence, so that side's set is empty. Not evidence of "
            "turnover."
        )
    return lines
