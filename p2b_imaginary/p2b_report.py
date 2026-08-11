"""
p2b_imaginary/p2b_report.py — cross-checkpoint aggregation.

`run_2b.sweep_summary_lines` prints one row per checkpoint. This module asks
the questions those rows exist to answer, and it is deliberately conservative
about what a 27-point series with no repeats can support.

WHAT THIS IS FOR
----------------
Phase 1 and Phase 2 dated five events between them. Phase 2b's Block 1a is a
weights-only per-layer scalar, so for the first time there is a candidate
MECHANISM series that can be laid against those dates:

  - Phase 1's step 8->16 collapse is "unexplained and could be an LR-warmup
    artifact" (status-1 open item 4). If the OV spectrum does nothing there,
    that is evidence for the artifact reading.
  - Phase 2's `frac_repulsive` decay (1.00 -> 0.50 -> 0.80 over ~90k steps
    with violation COUNT flat) is open item 5: something reorganises which
    subspace violations occupy without changing how many there are. Henrici
    non-normality measures exactly how much S and A interact, and is
    available at every checkpoint for free.

WHAT THIS REFUSES TO DO
-----------------------
**Categorical change-point verdicts.** status-2's own headline carries the
warning: of the 13 `mixed_or_unattributed` runs in the 40000-100000 window,
five sit at `frac_repulsive` exactly 0.500 against a strict `> 0.5` guard, so
"the verdict label is an artifact of where the threshold happens to fall."
Every function here returns a continuous quantity and a rank, never a
boolean "moved at step N".

**Change points without a dispersion scale.** With 27 checkpoints and no
repeats, "the sharpest interval is X" is a statement about one draw.
`interval_deltas` reports every interval's change in units of the
ACROSS-LAYER spread at the same checkpoint, which is a real error bar that
Block 1a supplies for free (24 layers per checkpoint). An interval that moves
less than the layer-to-layer scatter is not a transition.

**Log-vs-linear confusion.** Pythia's schedule is log-spaced then linear, and
`p1_mstate_tracking/visualization/checkpoints.py` already settled the
convention: `log(step+1)`, with step 0 kept as its own object rather than
folded into the colormap. Interval sizes here are computed in the same
coordinate, so "8->16" and "40000->60000" are not compared as if they were
equal-length intervals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# The dated events, as data
# ---------------------------------------------------------------------------
#
# Transcribed from status-1.md and status-2.md so a Block 1a trajectory can be
# laid against them without anyone re-reading a prose table. `span` is
# inclusive of both endpoints. `source` is the document that dated it — if one
# of these moves, this table is what has to move with it.

KNOWN_TRANSITIONS = (
    {
        "key": "late_layer_collapse",
        "span": (8, 16),
        "source": "status-1.md",
        "quantity": "raw effective rank 6.5 -> 2.1; max IP-mass near 1 0.016 -> 0.58",
        "note": "Confined to layers 21-23. Fully recovered by step 512. "
                "Unpredicted; status-1 open item 4 asks whether it is a "
                "training event or an LR-warmup artifact.",
    },
    {
        "key": "energy_break",
        "span": (256, 512),
        "source": "status-2.md",
        "quantity": "violations 21 -> 64 in one interval (Study B: 33 -> 68)",
        "note": "Theorem 3.4 holds exactly at steps 8-64 (9/9 prompts, 4 "
                "consecutive checkpoints) and is broken by 512.",
    },
    {
        "key": "plateau_onset_flip",
        "span": (256, 512),
        "source": "status-1.md",
        "quantity": "plateau-onset SD 0.00 -> 3.31",
        "note": "Weight-level to content-driven, in one interval.",
    },
    {
        "key": "fiedler_sign_change",
        "span": (1000, 3000),
        "source": "status-1.md",
        "quantity": "Fiedler deviation crosses zero, saturating near -0.023 by 40k",
        "note": None,
    },
    {
        "key": "effective_rank_peak",
        "span": (3000, 5000),
        "source": "status-1.md",
        "quantity": "mean effective rank 40.4 (individual runs to 60.4), then "
                    "monotone decline for 140k steps",
        "note": "status-1: 'its own dynamics, an order of magnitude earlier "
                "than the other two'. The rank peak sits unbracketed between "
                "1000 and 3000.",
    },
    {
        "key": "frac_repulsive_decay",
        "span": (7000, 100000),
        "source": "status-2.md",
        "quantity": "mean frac_repulsive 1.00 -> 0.50 with violation count flat",
        "note": "status-2 open item 5. Count saturates at 512 and never moves "
                "again; attribution decays for ~90k steps. Two curves, two "
                "timescales.",
    },
    {
        "key": "frac_repulsive_rebound",
        "span": (120000, 143000),
        "source": "status-2.md",
        "quantity": "mean frac_repulsive 0.50 -> 0.80, then 0.72",
        "note": "8/8 prompts move the same direction.",
    },
)

#: Block 1a scalars worth a trajectory, and what each would mean if it moved.
TRACKED_STATISTICS = {
    "complex_energy_fraction_mean": (
        "If flat from step 0, the 84-97% headline is a fact about square "
        "matrices rather than about training."
    ),
    "complex_energy_fraction_legacy_mean": (
        "The pre-rewrite per-block convention. Tracked only so the historical "
        "figure stays checkable against the corrected one."
    ),
    "dim_complex_fraction_mean": (
        "How many DIMENSIONS rotate, as against how much energy is in "
        "rotation. A different question from the line above."
    ),
    "theta_mean_across_layers": (
        "Mean rotation angle on [0, pi]. Near pi/2 is what a Gaussian gives."
    ),
    "frac_repulsive_real_part_mean": (
        "Fraction of rotation planes with Re(lambda) < 0 — the directions "
        "e^{-V} grows in. The weights-side analogue of Phase 2's "
        "`frac_repulsive`, which is measured on violations."
    ),
    "henrici_relative_mean": (
        "How much S and A interact. The candidate mechanism for Phase 2 open "
        "item 5: attribution reorganises while the count stays flat."
    ),
    "henrici_relative_max": (
        "Same, at the layer where it is largest."
    ),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_combined(path) -> dict:
    """Read `phase2b_results.json`, or the directory containing it."""
    path = Path(path)
    if path.is_dir():
        from p2b_imaginary.p2b_io import COMBINED_RESULTS
        path = path / COMBINED_RESULTS
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def collect_trajectory(combined: dict, statistic: str) -> dict:
    """
    One Block 1a summary scalar across checkpoints, with a dispersion band.

    `spread` is the standard deviation ACROSS LAYERS at each checkpoint,
    taken from `per_layer`. It is the only error bar available without
    re-running anything, and it is the right scale for "did this move": a
    change between checkpoints smaller than the layer-to-layer scatter within
    a checkpoint is not a transition.

    Returns dict(steps, values, spread, n_layers, statistic, missing_steps).
    """
    per_layer_key = _per_layer_key_for(statistic)

    rows = []
    for stem, r in combined.get("results", {}).items():
        step = r.get("checkpoint_step")
        block = r.get("block1a")
        if step is None or not block:
            continue
        summary = block.get("summary") or {}
        if statistic not in summary:
            continue
        vals = [rec.get(per_layer_key) for rec in (block.get("per_layer") or [])
                if per_layer_key and rec.get(per_layer_key) is not None]
        rows.append((int(step), float(summary[statistic]),
                     float(np.std(vals)) if vals else float("nan"),
                     len(block.get("per_layer") or [])))

    rows.sort()
    return {
        "statistic": statistic,
        "steps": [r[0] for r in rows],
        "values": [r[1] for r in rows],
        "spread": [r[2] for r in rows],
        "n_layers": [r[3] for r in rows],
        "missing_steps": list(combined.get("missing_checkpoints") or []),
        "meaning": TRACKED_STATISTICS.get(statistic),
    }


def _per_layer_key_for(statistic: str) -> Optional[str]:
    """Map a summary key back to the per-layer key it aggregates."""
    for suffix in ("_mean_across_layers", "_across_layers", "_mean", "_max"):
        if statistic.endswith(suffix):
            return statistic[: -len(suffix)]
    return None


def interval_deltas(traj: dict) -> list:
    """
    Per-interval change, in three units, ranked by the third.

    `delta`             raw change
    `delta_per_log_step` change divided by the interval's width in
                        `log(step+1)`, so 8->16 and 40000->60000 are not
                        compared as equal-length intervals
    `delta_in_spreads`  change divided by the mean across-layer spread of the
                        two endpoints. THE ranking key: an interval that moves
                        less than the within-checkpoint layer scatter is not a
                        transition, however large the raw number looks.

    Returns a list of dicts sorted by `|delta_in_spreads|` descending.
    """
    steps = traj["steps"]
    vals = traj["values"]
    spread = traj["spread"]
    out = []
    for i in range(1, len(steps)):
        s0, s1 = steps[i - 1], steps[i]
        d = vals[i] - vals[i - 1]
        width = float(np.log(s1 + 1) - np.log(s0 + 1))
        sp = float(np.nanmean([spread[i - 1], spread[i]]))
        out.append({
            "span": (int(s0), int(s1)),
            "delta": float(d),
            "log_width": width,
            "delta_per_log_step": float(d / width) if width > 0 else float("nan"),
            "spread": sp,
            "delta_in_spreads": (float(d / sp) if np.isfinite(sp) and sp > 1e-12
                                 else float("nan")),
        })
    out.sort(key=lambda r: (-abs(r["delta_in_spreads"])
                            if np.isfinite(r["delta_in_spreads"]) else 0.0))
    return out


def expected_range_under_noise(n: int, n_draws: int = 20000,
                               seed: int = 0) -> float:
    """
    Expected range of `n` iid standard normals — the reference a trajectory
    has to beat before "it moved" means anything.

    For n = 21 this is ~3.78 and for n = 27 it is ~4.00, so a 27-checkpoint
    series drawn from pure noise has a range of about four standard errors by
    construction. Comparing a range against ONE standard error, which is the
    obvious thing to do, therefore calls almost every flat trajectory a
    transition. Simulated rather than tabulated so the constant cannot drift
    from the n actually used.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(int(n_draws), int(n)))
    return float((x.max(axis=1) - x.min(axis=1)).mean())


def flatness(traj: dict) -> dict:
    """
    Whether the statistic moves at all, relative to within-checkpoint scatter.

    This is the actual test behind open question 1. "Does the complex fraction
    have a developmental trajectory" is not answered by the range being
    nonzero — it is answered by the range being large compared with how much
    the quantity varies across layers of a single checkpoint.

    `range_in_spreads < 1` means the whole 143000-step trajectory moves less
    than one checkpoint's layer-to-layer scatter: flat, and the headline is a
    statement about square matrices.
    """
    v = np.asarray(traj["values"], dtype=np.float64)
    sp = np.asarray(traj["spread"], dtype=np.float64)
    nl = np.asarray(traj.get("n_layers") or [], dtype=np.float64)
    if v.size == 0:
        return {"status": "no_data"}

    # An external series carries NaN spread by design (`external_trajectory`),
    # so an all-NaN slice here is expected rather than a defect.
    typical = (float(np.nanmedian(sp)) if np.isfinite(sp).any()
               else float("nan"))
    rng = float(np.nanmax(v) - np.nanmin(v))

    # `values` are MEANS over layers, so their sampling scale is
    # spread / sqrt(n_layers), not spread. Dividing a range by the raw layer
    # spread understates movement by ~sqrt(24) ~ 4.9x — conservative, but the
    # wrong scale, and it makes a real move look marginal. Both are reported:
    # `range_in_spreads` compares the trajectory against DEPTH variation,
    # which is a substantive comparison; `range_in_standard_errors` compares
    # it against the sampling noise of the mean, which is the statistical one.
    n_layers = float(np.nanmedian(nl)) if nl.size else float("nan")
    se = (typical / np.sqrt(n_layers)
          if np.isfinite(typical) and np.isfinite(n_layers) and n_layers > 0
          else float("nan"))
    range_in_se = (float(rng / se) if np.isfinite(se) and se > 1e-12
                   else float("nan"))
    expected = expected_range_under_noise(int(v.size))

    return {
        "statistic": traj["statistic"],
        "n_checkpoints": int(v.size),
        "first": float(v[0]),
        "last": float(v[-1]),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "range": rng,
        "typical_across_layer_spread": typical,
        "median_n_layers": n_layers,
        "standard_error_of_mean": se,
        "range_in_spreads": (float(rng / typical)
                             if np.isfinite(typical) and typical > 1e-12
                             else float("nan")),
        "range_in_standard_errors": range_in_se,
        "expected_range_in_se_under_noise": expected,
        # THE number. Below 1.0 the trajectory's range is no larger than a
        # series of this length drawn from pure noise would give, so "it
        # moves" is not supported however nonzero the range is.
        "range_excess_over_noise": (float(range_in_se / expected)
                                    if np.isfinite(range_in_se) and expected > 0
                                    else float("nan")),
        "monotone_rank_corr_with_log_step": _spearman(
            np.log(np.asarray(traj["steps"], dtype=np.float64) + 1.0), v),
    }


def _spearman(a, b) -> float:
    """Rank correlation, ties averaged. No scipy.stats dependency."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = _rankdata(a[m]), _rankdata(b[m])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 1e-12 else float("nan")


def _rankdata(x) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Alignment with the dated events
# ---------------------------------------------------------------------------

def align_to_transitions(traj: dict,
                         transitions: Sequence[dict] = KNOWN_TRANSITIONS) -> list:
    """
    For each dated event, how much this statistic moves across it.

    Deliberately NOT a hit/miss verdict. Each row reports the change in
    spread units and the RANK of that interval among all intervals, so a
    reader can see both "it moved 3.2 layer-spreads there" and "that was the
    2nd largest of 26 intervals" — which are different claims, and the second
    is the one that guards against a statistic that moves everywhere.

    An event whose span covers several checkpoints is scored on the total
    change from the first checkpoint at or after the span start to the last at
    or before the span end.
    """
    steps = np.asarray(traj["steps"], dtype=np.float64)
    vals = np.asarray(traj["values"], dtype=np.float64)
    sp = np.asarray(traj["spread"], dtype=np.float64)
    deltas = interval_deltas(traj)
    n_intervals = len(deltas)
    rank_of = {d["span"]: i + 1 for i, d in enumerate(deltas)}

    rows = []
    for ev in transitions:
        lo, hi = ev["span"]
        inside = np.where((steps >= lo) & (steps <= hi))[0]
        if inside.size < 2:
            rows.append({
                "key": ev["key"], "span": ev["span"], "source": ev["source"],
                "status": "not_bracketed",
                "note": "the sweep has fewer than two checkpoints in this span; "
                        "nothing can be said about it",
                "n_checkpoints_in_span": int(inside.size),
            })
            continue

        i0, i1 = int(inside[0]), int(inside[-1])
        d = float(vals[i1] - vals[i0])
        s = float(np.nanmean(sp[[i0, i1]]))
        span_key = (int(steps[i0]), int(steps[i1]))

        rows.append({
            "key": ev["key"],
            "span": ev["span"],
            "observed_span": span_key,
            "source": ev["source"],
            "status": "scored",
            "delta": d,
            "delta_in_spreads": (float(d / s) if np.isfinite(s) and s > 1e-12
                                 else float("nan")),
            "interval_rank": rank_of.get(span_key),
            "n_intervals": n_intervals,
            "n_checkpoints_in_span": int(inside.size),
            "event_quantity": ev["quantity"],
        })
    return rows


def co_movement(traj_a: dict, traj_b: dict) -> dict:
    """
    Rank correlation between two trajectories on their shared step grid.

    Built for one specific question — does Henrici non-normality track Phase
    2's `frac_repulsive` decay — but it is the wrong tool for a causal claim
    and says so. With 27 points on a shared monotone-ish schedule, two
    quantities that both drift with training will correlate. The output
    therefore carries `n_shared` and a `caveat`, and `interval_agreement`
    (fraction of intervals where both move the same direction), which is less
    sensitive to a common trend than the level correlation is.
    """
    a = {int(s): v for s, v in zip(traj_a["steps"], traj_a["values"])}
    b = {int(s): v for s, v in zip(traj_b["steps"], traj_b["values"])}
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return {"status": "insufficient_overlap", "n_shared": len(shared)}

    va = np.array([a[s] for s in shared], dtype=np.float64)
    vb = np.array([b[s] for s in shared], dtype=np.float64)
    da, db = np.diff(va), np.diff(vb)
    same = np.sign(da) == np.sign(db)

    return {
        "status": "ok",
        "a": traj_a["statistic"],
        "b": traj_b.get("statistic", "external"),
        "n_shared": len(shared),
        "spearman_levels": _spearman(va, vb),
        "spearman_deltas": _spearman(da, db),
        "interval_agreement": float(same.mean()) if same.size else float("nan"),
        "caveat": (
            "Two quantities that both drift with training will correlate at "
            "the level. `spearman_deltas` and `interval_agreement` are the "
            "less trend-sensitive readings; neither is a causal claim."
        ),
    }


def external_trajectory(name: str, steps: Sequence[int],
                        values: Sequence[float]) -> dict:
    """
    Wrap a series from another phase (e.g. Phase 2's `frac_repulsive`) in the
    trajectory shape, so `co_movement` can take it.

    `spread` is NaN: an external series arrives without Phase 2b's
    across-layer dispersion, so `flatness` and `interval_deltas` will report
    NaN in spread units for it rather than silently substituting a different
    scale.
    """
    order = np.argsort(np.asarray(steps))
    return {
        "statistic": name,
        "steps": [int(steps[i]) for i in order],
        "values": [float(values[i]) for i in order],
        "spread": [float("nan")] * len(order),
        "n_layers": [0] * len(order),
        "missing_steps": [],
        "meaning": None,
        "external": True,
    }


# ---------------------------------------------------------------------------
# Block 1b across checkpoints
# ---------------------------------------------------------------------------

def block1b_trajectory(combined: dict) -> dict:
    """
    Verdicts, elimination rates and refusals per checkpoint.

    `n_refused` is reported next to `n_scored` at every step, because a
    checkpoint where every run refused looks identical to one where every run
    said "inert" if only the verdict tally is read. Steps 8-64 are `no
    violations` on 9/9 prompts in Study B, so this is the expected shape at
    the early end and must not read as a finding.
    """
    per_step: dict = {}
    for stem, r in combined.get("results", {}).items():
        step = r.get("checkpoint_step")
        if step is None:
            continue
        row = per_step.setdefault(int(step), {
            "step": int(step), "verdicts": {}, "elim_full": [],
            "elim_signed": [], "n_runs": 0, "n_refused": 0,
            "n_truncated_frames": 0, "n_invariance_broken": 0,
        })
        for prompt, js in (r.get("block1b") or {}).items():
            row["n_runs"] += 1
            interp = js.get("interpretation")
            if not interp:
                row["n_refused"] += 1
                continue
            v = interp["overall"]
            row["verdicts"][v] = row["verdicts"].get(v, 0) + 1
            if v in ("no_violations", "not_comparable"):
                row["n_refused"] += 1
            ref = str(interp["reference_beta"])
            comp = js.get("comparison", {}).get(ref, {})
            for key in ("elim_full", "elim_signed"):
                rate = comp.get(key, {}).get("rate")
                if rate is not None:
                    row[key].append(float(rate))
            row["n_truncated_frames"] += sum(
                1 for f in js.get("frames", {}).values() if f.get("truncated"))
            if (js.get("invariance") or {}).get("status") == "identity_broken":
                row["n_invariance_broken"] += 1

    for row in per_step.values():
        for key in ("elim_full", "elim_signed"):
            vals = row.pop(key)
            row[f"{key}_n"] = len(vals)
            row[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{key}_min"] = float(np.min(vals)) if vals else None
            row[f"{key}_max"] = float(np.max(vals)) if vals else None

    return {"per_step": [per_step[s] for s in sorted(per_step)]}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(combined: dict,
                 statistics: Optional[Sequence[str]] = None) -> dict:
    """Every trajectory, its flatness, and its alignment with the dated events."""
    stats = list(statistics or TRACKED_STATISTICS)
    trajectories = {s: collect_trajectory(combined, s) for s in stats}
    trajectories = {k: v for k, v in trajectories.items() if v["steps"]}

    return {
        "phase": "2b",
        "base": combined.get("base"),
        "n_checkpoints": combined.get("n_checkpoints"),
        "missing_checkpoints": combined.get("missing_checkpoints") or [],
        "n_failed": combined.get("n_failed", 0),
        "trajectories": trajectories,
        "flatness": {k: flatness(v) for k, v in trajectories.items()},
        "intervals": {k: interval_deltas(v)[:5] for k, v in trajectories.items()},
        "alignment": {k: align_to_transitions(v) for k, v in trajectories.items()},
        "block1b": block1b_trajectory(combined),
        "known_transitions": [dict(t) for t in KNOWN_TRANSITIONS],
    }


def report_lines(report: dict) -> list:
    """LLM-consumable report."""
    L = [
        "=== Phase 2b cross-checkpoint report ===",
        f"Base: {report.get('base')}   checkpoints: {report.get('n_checkpoints')}"
        f"   failures: {report.get('n_failed')}"
        f"   missing: {report.get('missing_checkpoints') or 'none'}",
        "",
        "--- Does anything move? ---",
        "excess  = the trajectory's range in standard errors of the",
        "          per-checkpoint mean, divided by the range a series of this",
        "          length drawn from PURE NOISE would give (~3.8 at n=21,",
        "          ~4.0 at n=27). Below 1.0, 'it moves' is not supported.",
        "in_sd   = the same range against the across-LAYER scatter within a",
        "          checkpoint. A different comparison: how the trajectory",
        "          compares with depth variation, not with sampling noise.",
        "rho_log = rank correlation with log(step+1).",
        "",
        f"{'statistic':<42} {'first':>8} {'last':>8} {'range':>8} "
        f"{'excess':>7} {'in_sd':>7} {'rho_log':>8}",
    ]
    for key, f in report["flatness"].items():
        if f.get("status") == "no_data":
            continue
        L.append(
            f"{key:<42} {f['first']:8.4f} {f['last']:8.4f} {f['range']:8.4f} "
            f"{f['range_excess_over_noise']:7.2f} "
            f"{f['range_in_spreads']:7.2f} "
            f"{f['monotone_rank_corr_with_log_step']:8.3f}"
        )

    L += ["", "--- Sharpest intervals (in across-layer spreads) ---"]
    for key, rows in report["intervals"].items():
        if not rows:
            continue
        top = rows[0]
        L.append(
            f"  {key:<42} {str(top['span']):>16}  "
            f"{top['delta']:+.4f}  ({top['delta_in_spreads']:+.2f} sd)"
        )

    L += ["", "--- Alignment with dated transitions ---",
          "  interval_rank is out of all intervals: rank 1 with a large",
          "  delta_in_spreads is a co-located move; rank 20 of 26 is not.", ""]
    for key, rows in report["alignment"].items():
        scored = [r for r in rows if r.get("status") == "scored"]
        if not scored:
            continue
        L.append(f"  {key}")
        for r in scored:
            rank = (f"{r['interval_rank']}/{r['n_intervals']}"
                    if r.get("interval_rank") else "spans multiple")
            L.append(
                f"    {r['key']:<24} {str(r['span']):>16}  "
                f"{r['delta']:+.4f}  ({r['delta_in_spreads']:+.2f} sd)  "
                f"rank {rank}"
            )
        for r in rows:
            if r.get("status") == "not_bracketed":
                L.append(f"    {r['key']:<24} {str(r['span']):>16}  "
                         f"NOT BRACKETED ({r['n_checkpoints_in_span']} checkpoints)")
        L.append("")

    b = report.get("block1b", {}).get("per_step") or []
    if b:
        L += [
            "--- Block 1b across checkpoints ---",
            f"{'step':>8} {'runs':>5} {'refused':>8} {'trunc':>6} "
            f"{'elim_full':>10} {'elim_signed':>12}  verdicts",
        ]
        for row in b:
            ef = row["elim_full_mean"]
            es = row["elim_signed_mean"]
            verdicts = ", ".join(f"{k}={v}" for k, v in sorted(row["verdicts"].items()))
            L.append(
                f"{row['step']:>8} {row['n_runs']:>5} {row['n_refused']:>8} "
                f"{row['n_truncated_frames']:>6} "
                f"{'n/a' if ef is None else f'{ef:+.4f}':>10} "
                f"{'n/a' if es is None else f'{es:+.4f}':>12}  {verdicts}"
            )
        broken = sum(r["n_invariance_broken"] for r in b)
        L.append("")
        L.append(f"  Invariance control broken in {broken} run(s). "
                 "Nonzero is a numerical failure of e^{-A}'s orthogonality, "
                 "not a result about rotation.")
        L.append("  `refused` counts no_violations and not_comparable. Steps "
                 "8-64 are clean on 9/9 prompts in Study B, so a high count "
                 "there is the expected shape, not a finding.")

    return L


def write_report(combined: dict, out_dir,
                 statistics: Optional[Sequence[str]] = None) -> dict:
    """Build the report and write `phase2b_report.json` / `.txt`."""
    from p2b_imaginary.p2b_io import json_default

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(combined, statistics)

    with open(out_dir / "phase2b_report.json", "w") as f:
        json.dump(report, f, indent=2, default=json_default, allow_nan=False)
    with open(out_dir / "phase2b_report.txt", "w") as f:
        f.write("\n".join(report_lines(report)) + "\n")
    return report


__all__ = [
    "KNOWN_TRANSITIONS",
    "expected_range_under_noise",
    "TRACKED_STATISTICS",
    "load_combined",
    "collect_trajectory",
    "interval_deltas",
    "flatness",
    "align_to_transitions",
    "co_movement",
    "external_trajectory",
    "block1b_trajectory",
    "build_report",
    "report_lines",
    "write_report",
]
