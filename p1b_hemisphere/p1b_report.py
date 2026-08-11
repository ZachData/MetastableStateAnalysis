"""
p1b_report.py — Phase 1b's summaries, verdicts, and cross-run synthesis.

Split out of run_1b.py, matching the project's own convention
(reporting_p1.py, reporting_p2.py, p5b_report.py, report_6.py). The split is
not cosmetic: run_1b.py needs torch to load a model, and none of the logic
here does, so extracting it makes every verdict rule testable in a plain
numpy environment. All four of the reporting bugs below were in code that
had no test because it could not be imported without a GPU stack.

Four corrections carried in this module
---------------------------------------

**1. An inverted verdict field.** `_global_verdict` computed

    cone_at_long = mean(cone_collapse_fractions) < 0.5

and published it as `cone_collapse_regime_at_long_prompts`. It is True
exactly when cone-collapse is RARE. The derived `paper_alignment` string was
right (it read the variable, not the name), so the published verdict was
correct while the boolean printed beside it in the JSON said the opposite.
Renamed to `split_regime_at_long_prompts`, with
`cone_collapse_regime_at_long_prompts` retained as its correct negation so
existing readers get the truth rather than a KeyError.

**2. Swapped crossref keys.** `_build_summary` assigned
`mean_crossing_at_violation` to a field named
`mean_crossing_at_merge_events`, and `mean_axis_rotation_at_merge` to
`mean_axis_rotation_at_violation_layers`. Names and values were crossed. Both
now carry their own names, and both off-event baselines are reported
alongside — a mean crossing count at violation layers means nothing without
the mean at non-violation layers next to it, and the old summary published
the first without the second.

**3. A hardcoded pre-run narrative.** `_write_cross_run_md` ended with three
paragraphs asserting that long prompts enter a split regime at mid-depth and
that the bipartition is real geometric structure. Both are false against the
run that was actually performed (status-1b.md blocker 2 records this). The
text is now generated from the results.

**4. `strong_bipartition` as the sole headline.** Under cone-collapse the
antipodal classifier cannot report a strong bipartition, so a 0% figure is
close to uninformative on its own. Every summary now carries the relative
classifier's `separated_fraction` beside it, and the verdict distinguishes
"no bipartition" from "a bipartition that is not antipodal".

Also new here: `aggregate_by_checkpoint`. The cross-run digest grouped only
by model and by prompt, so a 27-checkpoint Pythia pilot rendered as 27
unrelated models with no step axis anywhere. Checkpoint families are grouped
and reported against log10(step + 1), for the reason core/checkpoint_frames.py
gives: Pythia's checkpoints are log-spaced to step 512 and linear after, so
anything differenced over checkpoint index peaks wherever the release
schedule changes spacing rather than wherever training does.
"""

from __future__ import annotations

import re

import numpy as np


#: Fallback for p1_mstate_tracking.visualization.checkpoints._STEP_RE, which
#: is the canonical definition but lives in a module that imports matplotlib.
#: `checkpoint_step` prefers the canonical one and falls back to this.
_STEP_RE = re.compile(r"^(?P<base>.+)-step(?P<step>\d+)$")

#: A prompt is "long" above this many tokens. Was hardcoded as `n_tokens > 100`
#: inside the verdict. Kept as a named default because the number is
#: tokenizer-dependent: the same battery under the NeoX BPE does not produce
#: the same token counts as under GPT-2 BPE, so which prompts count as long
#: silently changes with the model family. core/battery_structure.py is where
#: that gets measured rather than assumed; this constant is the reporting
#: convention, and every verdict that uses it records the value and the
#: resulting sample size.
LONG_PROMPT_TOKENS = 100


def checkpoint_step(model: str):
    """'pythia-410m-step2000' -> 2000; None for non-checkpoint names."""
    try:
        from p1_mstate_tracking.visualization.checkpoints import _checkpoint_step
        return _checkpoint_step(model)
    except Exception:
        m = _STEP_RE.match(str(model))
        return int(m.group("step")) if m else None


def checkpoint_base(model: str):
    """'pythia-410m-step2000' -> 'pythia-410m'; None for non-checkpoint names."""
    try:
        from p1_mstate_tracking.visualization.checkpoints import _checkpoint_base
        return _checkpoint_base(model)
    except Exception:
        m = _STEP_RE.match(str(model))
        return m.group("base") if m else None


# ---------------------------------------------------------------------------
# Per-run summary
# ---------------------------------------------------------------------------

def build_summary(block0: dict, block1: dict, mem_json: dict,
                  block3: dict | None, block4: dict,
                  axis: dict | None = None) -> dict:
    """Assemble one run's summary from the block outputs."""
    n_layers = block0["n_layers"]

    regime_arr  = np.array([str(r) for r in block0["regime"]])
    strong_frac = float((regime_arr == "strong_bipartition").sum() / n_layers)

    rel = block0.get("regime_relative")
    if rel is not None:
        rel_arr = np.array([str(r) for r in rel])
        separated_frac = float((rel_arr == "separated").sum() / n_layers)
        graded_frac    = float((rel_arr == "graded").sum() / n_layers)
    else:
        separated_frac = graded_frac = None

    if block3 is not None:
        cone_arr  = np.array([str(r) for r in block3["cone_regime"]])
        cone_frac = float((cone_arr == "cone_collapse").sum() / n_layers)
        nm = np.asarray(block3["normalized_margin"], dtype=np.float64)
        nm = nm[np.isfinite(nm)]
        mean_norm_margin = float(nm.mean()) if nm.size else None
        n_escalated = int(np.asarray(block3.get("escalated", [])).sum())
    else:
        cone_frac = mean_norm_margin = None
        n_escalated = 0

    axis_rot  = block1["axis_rotation"]
    valid_rot = axis_rot[np.isfinite(axis_rot)]
    mean_axis_rot = float(valid_rot.mean()) if valid_rot.size else None

    event_counts: dict = {}
    for ev in block1["events"]:
        t = ev.get("type", "unknown")
        event_counts[t] = event_counts.get(t, 0) + 1

    mem_summary = mem_json.get("summary", {})
    nesting = mem_json.get("hdbscan_nesting") or {}
    bvn     = mem_json.get("border_vs_noise") or {}
    crossref = block1.get("crossref", {})

    summary = {
        "strong_bipartition_layer_fraction": strong_frac,
        "separated_layer_fraction":          separated_frac,
        "graded_layer_fraction":             graded_frac,
        "cone_collapse_layer_fraction":      cone_frac,
        "mean_normalized_cone_margin":       mean_norm_margin,
        "n_layers_escalated_to_full_d":      n_escalated,
        "mean_axis_rotation":                mean_axis_rot,
        "mean_asymmetry_strong":             block4.get("mean_asymmetry_strong"),
        "event_counts":                      event_counts,
        "mean_stability_score":              mem_summary.get("mean_stability_score"),
        "fraction_never_stable":             mem_summary.get("fraction_never_stable"),
        "hdbscan_nesting_overall":           nesting.get("overall"),
        "border_vs_noise_mean_auc":          (bvn.get("overall") or {}).get("mean_auc"),
        # Each value under the name of the thing it measures, with its
        # off-event baseline beside it. The old summary crossed the two and
        # published neither baseline.
        "crossref_with_phase1": {
            "mean_axis_rotation_at_merge":      crossref.get("mean_axis_rotation_at_merge"),
            "mean_axis_rotation_off_merge":     crossref.get("mean_axis_rotation_off_merge"),
            "mean_crossing_at_violation":       crossref.get("mean_crossing_at_violation"),
            "mean_crossing_off_violation":      crossref.get("mean_crossing_off_violation"),
            "n_merges_in_run":                  crossref.get("n_merges_in_run"),
            "n_violations_in_run":              crossref.get("n_violations_in_run"),
        },
    }

    if axis is not None:
        summary["axis_modal_redundancy"] = axis.get("modal_redundancy")
        summary["mean_cos_axis_mean"]    = _mean(axis.get("cos_axis_mean"))
        summary["mean_cos_axis_pc1"]     = _mean(axis.get("cos_axis_pc1"))
        summary["mean_cos_mean_pc1"]     = _mean(axis.get("cos_mean_pc1"))

    return summary


# ---------------------------------------------------------------------------
# Cross-run aggregation
# ---------------------------------------------------------------------------

AGGREGATED_FIELDS = (
    "strong_bipartition_layer_fraction",
    "separated_layer_fraction",
    "graded_layer_fraction",
    "cone_collapse_layer_fraction",
    "mean_normalized_cone_margin",
    "mean_axis_rotation",
    "mean_asymmetry_strong",
    "mean_stability_score",
    "fraction_never_stable",
    "border_vs_noise_mean_auc",
    "mean_cos_axis_mean",
    "mean_cos_axis_pc1",
)


def aggregate(runs: list) -> dict:
    """Mean of each aggregated field over a list of run dicts."""
    out: dict = {"n_runs": len(runs)}
    for f in AGGREGATED_FIELDS:
        vals = [r["summary"].get(f) for r in runs
                if r.get("summary", {}).get(f) is not None]
        out[f"mean_{f}"] = float(np.mean(vals)) if vals else None
    return out


def aggregate_by_checkpoint(all_results: list) -> dict:
    """
    Group runs into checkpoint families and report each family against the
    training-step axis.

    Returns {base_model: {"steps": [...], "log_step": [...],
                          "per_step": {step: aggregate(...)}}}.

    Non-checkpoint models (gpt2, albert-base-v2, pythia-1.4b-random) produce
    no family — pythia-1.4b-random deliberately carries no step and must not
    be placed on the step axis, per core/pythia_registry.py.
    """
    fams: dict = {}
    for r in all_results:
        model = r.get("model", "")
        step  = checkpoint_step(model)
        base  = checkpoint_base(model)
        if step is None or base is None:
            continue
        fams.setdefault(base, {}).setdefault(int(step), []).append(r)

    out: dict = {}
    for base, by_step in fams.items():
        steps = sorted(by_step)
        out[base] = {
            "steps":    steps,
            "log_step": [float(np.log10(s + 1.0)) for s in steps],
            "per_step": {int(s): aggregate(by_step[s]) for s in steps},
        }
    return out


# ---------------------------------------------------------------------------
# Global verdict
# ---------------------------------------------------------------------------

def global_verdict(all_results: list,
                   long_prompt_tokens: int = LONG_PROMPT_TOKENS) -> dict:
    """
    Derive the run-level verdicts.

    Every boolean here is named for what being True means. The previous
    version's `cone_collapse_regime_at_long_prompts` was True when
    cone-collapse was rare; see the module docstring.
    """
    strong_fracs = [r["summary"].get("strong_bipartition_layer_fraction")
                    for r in all_results]
    strong_fracs = [f for f in strong_fracs if f is not None]
    antipodal_bipartition_universal = bool(
        strong_fracs and all(f > 0.0 for f in strong_fracs))

    sep_fracs = [r["summary"].get("separated_layer_fraction")
                 for r in all_results]
    sep_fracs = [f for f in sep_fracs if f is not None]
    separated_majority = (bool(float(np.mean(sep_fracs)) > 0.5)
                          if sep_fracs else None)

    overlaps = [entry["match_overlap"]
                for r in all_results for entry in r.get("per_layer", [])
                if entry.get("match_overlap") is not None]
    identity_persistent = (bool(float(np.mean(overlaps)) > 0.5)
                           if overlaps else False)

    nested_fracs = []
    for r in all_results:
        n = r["summary"].get("hdbscan_nesting_overall") or {}
        v = n.get("fully_nested_fraction")
        if v is not None:
            nested_fracs.append(v)
    hdbscan_nested = (bool(float(np.mean(nested_fracs)) > 0.5)
                      if nested_fracs else None)

    long_runs = [r for r in all_results
                 if r.get("n_tokens", 0) > long_prompt_tokens]
    cone_vals = [r["summary"].get("cone_collapse_layer_fraction")
                 for r in long_runs
                 if r["summary"].get("cone_collapse_layer_fraction") is not None]

    if cone_vals:
        mean_cone = float(np.mean(cone_vals))
        split_at_long = bool(mean_cone < 0.5)
        paper_alignment = "split" if split_at_long else "cone_collapse"
    else:
        mean_cone = None
        split_at_long = None
        paper_alignment = "mixed"

    # Is cone-collapse more than dimension counting? Only answerable where
    # nulls were run; None otherwise, never silently True.
    uniform_fracs = []
    for r in all_results:
        v = r["summary"].get("mean_uniform_cone_fraction")
        if v is not None:
            uniform_fracs.append(v)
    cone_above_dimension_null = (
        bool(float(np.mean(uniform_fracs)) < 0.5) if uniform_fracs else None)

    redundancies = [r["summary"].get("axis_modal_redundancy")
                    for r in all_results]
    redundancies = [x for x in redundancies if x]
    if redundancies:
        counts: dict = {}
        for x in redundancies:
            counts[x] = counts.get(x, 0) + 1
        axis_verdict = max(counts, key=counts.get)
    else:
        counts = {}
        axis_verdict = None

    return {
        # The bipartition, under both classifiers.
        "antipodal_bipartition_present_universally": antipodal_bipartition_universal,
        "separated_under_relative_classifier":       separated_majority,
        "bipartition_identity_persistent":           identity_persistent,
        "hdbscan_nested_in_bipartition":             hdbscan_nested,
        # Containment. Named for what True means.
        "split_regime_at_long_prompts":              split_at_long,
        "cone_collapse_regime_at_long_prompts":
            (None if split_at_long is None else (not split_at_long)),
        "mean_cone_collapse_fraction_long_prompts":  mean_cone,
        "n_long_prompt_runs":                        len(long_runs),
        "long_prompt_token_threshold":               int(long_prompt_tokens),
        "cone_collapse_above_dimension_null":        cone_above_dimension_null,
        # What the axis is.
        "axis_redundancy":                           axis_verdict,
        "axis_redundancy_counts":                    counts,
        "paper_alignment":                           paper_alignment,
    }


# ---------------------------------------------------------------------------
# Cross-run markdown, generated from results
# ---------------------------------------------------------------------------

def cross_run_markdown(cross_run: dict, by_model: dict, by_prompt: dict) -> str:
    """
    One-page synthesis. Every claim below is conditioned on the verdict dict;
    nothing is asserted that the run did not measure.
    """
    v = cross_run.get("global_verdict", {})

    def pct(x):  return f"{x * 100:.1f}%" if x is not None else "—"
    def num(x, dp=3): return f"{x:.{dp}f}" if x is not None else "—"

    lines = [
        "# Phase 1b — Cross-Run Synthesis",
        "",
        "## Regime counts by model",
        "",
        "| Model | Strong bipartition % (antipodal) | Separated % (relative) "
        "| Cone-collapse % | Mean norm. margin | Mean axis rotation (rad) |",
        "|---|---|---|---|---|---|",
    ]
    for model, agg in cross_run.get("by_model", {}).items():
        lines.append(
            f"| {model} "
            f"| {pct(agg.get('mean_strong_bipartition_layer_fraction'))} "
            f"| {pct(agg.get('mean_separated_layer_fraction'))} "
            f"| {pct(agg.get('mean_cone_collapse_layer_fraction'))} "
            f"| {num(agg.get('mean_mean_normalized_cone_margin'))} "
            f"| {num(agg.get('mean_mean_axis_rotation'))} |"
        )

    lines += [
        "",
        "## Token stability by model",
        "",
        "| Model | Mean stability | Never stable | Border-vs-noise AUC |",
        "|---|---|---|---|",
    ]
    for model, agg in cross_run.get("by_model", {}).items():
        lines.append(
            f"| {model} | {num(agg.get('mean_mean_stability_score'))} "
            f"| {pct(agg.get('mean_fraction_never_stable'))} "
            f"| {num(agg.get('mean_border_vs_noise_mean_auc'))} |"
        )

    by_ckpt = cross_run.get("by_checkpoint") or {}
    if by_ckpt:
        lines += ["", "## Checkpoint families", ""]
        for base, fam in by_ckpt.items():
            lines += [
                f"### {base}",
                "",
                "| Step | log10(step+1) | Cone-collapse % | Separated % | "
                "Mean axis rotation | cos(axis, mean) |",
                "|---|---|---|---|---|---|",
            ]
            for s, lx in zip(fam["steps"], fam["log_step"]):
                a = fam["per_step"][int(s)]
                lines.append(
                    f"| {s} | {lx:.2f} "
                    f"| {pct(a.get('mean_cone_collapse_layer_fraction'))} "
                    f"| {pct(a.get('mean_separated_layer_fraction'))} "
                    f"| {num(a.get('mean_mean_axis_rotation'))} "
                    f"| {num(a.get('mean_mean_cos_axis_mean'))} |"
                )
            lines.append("")

    lines += [
        "",
        "## Global verdict",
        "",
        f"- Antipodal bipartition present in every run: "
        f"{v.get('antipodal_bipartition_present_universally')}",
        f"- Separated under the relative classifier: "
        f"{v.get('separated_under_relative_classifier')}",
        f"- Bipartition identity persistent across layers: "
        f"{v.get('bipartition_identity_persistent')}",
        f"- HDBSCAN clusters nested in the bipartition: "
        f"{v.get('hdbscan_nested_in_bipartition')}",
        f"- Split regime at long prompts (>{v.get('long_prompt_token_threshold')} "
        f"tokens, n={v.get('n_long_prompt_runs')}): "
        f"{v.get('split_regime_at_long_prompts')}",
        f"- Cone-collapse exceeds the matched dimension null: "
        f"{v.get('cone_collapse_above_dimension_null')}",
        f"- Axis redundancy: {v.get('axis_redundancy')}",
        f"- Paper alignment: {v.get('paper_alignment')}",
        "",
    ]

    lines += ["## What the bipartition is", ""] + _bipartition_paragraph(v, cross_run)
    lines += ["", "## Relationship to the paper's hemisphere", ""] + _cone_paragraph(v)
    lines += ["", "## What the axis is", ""] + _axis_paragraph(v)

    return "\n".join(lines)


def _bipartition_paragraph(v: dict, cross_run: dict) -> list:
    antipodal = v.get("antipodal_bipartition_present_universally")
    separated = v.get("separated_under_relative_classifier")

    if antipodal:
        body = (
            "The antipodal classifier reports a strong bipartition in every "
            "run: the Fiedler partition produces two populated halves whose "
            "centroids are at least pi/2 apart and which are internally "
            "compact."
        )
    elif separated:
        body = (
            "No run reaches the antipodal classifier's strong_bipartition "
            "state, but the relative classifier reports separated in a "
            "majority of layers. Cross-half pairs are measurably less "
            "similar than same-half pairs while both halves remain inside "
            "one open half-space. These are not contradictory readings: the "
            "antipodal threshold (centroid angle >= pi/2) cannot be met "
            "under cone-collapse, so its null is close to structural. The "
            "axis carries contrast; it does not separate antipodes."
        )
    elif separated is False:
        body = (
            "Neither classifier finds a bipartition. The Fiedler sign split "
            "carries no similarity contrast: cross-half pairs are about as "
            "similar as same-half pairs. The k=2 eigengap is not marking a "
            "partition of the token set."
        )
    else:
        body = (
            "The relative classifier was not run or produced no verdict, so "
            "only the antipodal reading is available, and it reports no "
            "strong bipartition. That is close to uninformative on its own "
            "under cone-collapse — see the module docstring in "
            "p1b_report.py."
        )
    return [body]


def _cone_paragraph(v: dict) -> list:
    split = v.get("split_regime_at_long_prompts")
    above = v.get("cone_collapse_above_dimension_null")
    n     = v.get("n_long_prompt_runs")

    if split is None:
        return [
            "Block 3 produced no verdict at long prompts — either it was not "
            f"run, or no run exceeded {v.get('long_prompt_token_threshold')} "
            "tokens. Note that the token threshold is tokenizer-dependent, so "
            "a battery that was long under one tokenizer may not be under "
            "another."
        ]

    if split:
        base = (
            f"Across {n} long-prompt runs, most layers admit no enclosing "
            "half-space. Theorem 6.3's precondition — all tokens in a single "
            "open hemisphere — fails there, so the cone-collapse result does "
            "not apply to those layers."
        )
    else:
        base = (
            f"Across {n} long-prompt runs, every layer admits an enclosing "
            "open half-space. Theorem 6.3's precondition holds throughout, "
            "and the k=2 eigengap is not an antipodal split."
        )

    if above is True:
        base += (
            " This is not a dimension-counting artifact: matched uniform "
            "draws at the same n and d_eff do not reproduce it."
        )
    elif above is False:
        base += (
            " Read with care — matched uniform draws at the same n and d_eff "
            "reproduce the containment, so this is at least partly a "
            "statement about dimension rather than about the model."
        )
    else:
        base += (
            " No null was run, so how much of this is transformer geometry "
            "and how much is n versus d_eff is not established. Pass "
            "--n-null to settle it."
        )
    return [base]


def _axis_paragraph(v: dict) -> list:
    r = v.get("axis_redundancy")
    if r == "pc1":
        return [
            "At most layers the activation-space Fiedler axis IS the top "
            "principal component (|cos| >= 0.9). The k=2 structure is the "
            "leading variance direction, recovered through a more expensive "
            "route. Anything downstream using this axis as a probe feature — "
            "Phase 5's hemisphere centroids, Phase 6's Fiedler-difference "
            "vector — is using PC1 and should say so."
        ]
    if r == "top_pc_block":
        return [
            "The axis is not any single principal component but lies almost "
            "entirely inside the top-k principal subspace. It is not new "
            "information relative to the cloud's leading variance geometry, "
            "though it is not identifiable with one component either — which "
            "is the expected picture when the top eigenvalues are close "
            "together."
        ]
    if r == "distinct":
        return [
            "The axis leaves the top-k principal subspace. It carries "
            "structure the cloud's leading variance geometry does not supply, "
            "and is worth a probe of its own."
        ]
    if r == "degenerate":
        return [
            "The axis is degenerate at most layers. Note that this verdict "
            "also fires when the Fiedler vector is not cleanly orthogonal to "
            "the Laplacian's trivial eigenvector, which indicates a "
            "disconnected graph or an unconverged eigensolve rather than an "
            "interesting geometry — check the connectivity floor."
        ]
    return [
        "Axis identity was not computed for this run, so whether the Fiedler "
        "axis is distinguishable from the cloud's leading variance directions "
        "is unknown."
    ]


def _mean(arr):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None
