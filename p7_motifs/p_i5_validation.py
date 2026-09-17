"""
p7_motifs/p_i5_validation.py — P-I5's gate, PART FOUR: validating the
pipeline `p_i5_ablation.py` built, before its `L3H6` reading (§3.32,
p = 0.0234 under the min-rank statistic; 0.0312 re-scored under the
intersection-union statistic that replaced it on 2026-09-17, see
`p_i5_gate.py` section 2) is trusted as more than a first look.

STATISTIC NOTE (2026-09-17). The p-values quoted below were computed with
`joint_rank_pvalue`, which controls only the complete null and is no longer
P-I5's statistic; the scripts now report `intersection_union_pvalue` as
`gate` and the min-rank value beside it as `*_min_rank_superseded`. Where
the committed record stores the per-prompt deltas the re-scored value is
given; the 2026-09-16 negative-control records store only p-values and
cannot be re-scored without a model run (later records store the deltas).

**THE VALIDATION FOUND A REAL PROBLEM. §3.32's reading is NOT specific to
induction, and should not be read as evidence for `P-I5` until this is
resolved.** Stated first because it is the headline result, not a caveat
buried under four checks that mostly passed.

1. NEGATIVE CONTROLS FAIL. Ablating `L4H6` ("below the +0.05 print
   threshold" in `status-8.md`'s own cascade table — the closest thing to
   a documented near-zero head, from layer 4, not layer 3) gives
   `joint_rank_pvalue` **p = 0.0039** — the exact floor, MORE extreme than
   `L3H6`'s own p = 0.0234. Ablating `L5H3` (arbitrary, final-layer,
   named in no induction-cascade table anywhere in this project) gives
   **p = 0.0391** — same order of magnitude as `L3H6`. Neither of these
   heads has any documented relationship to induction, and both "pass"
   the gate `L3H6` passes.

2. ONE RANDOM-VS-RANDOM DRAW, REPORTED AS WHAT IT IS.
   `run_random_vs_random_diagnostic` runs the identical pipeline with BOTH
   arms drawn as matched-magnitude random directions (seed 100 against
   seed 200, neither one a real ablation). Because both arms are random,
   `seed_a - seed_b` has no predeclared direction, so only the two-sided
   reading means anything: min-rank two-sided = **0.031**, intersection-
   union two-sided = **0.156** (re-scored from the stored deltas); the
   one-sided 0.930 / 0.965 says nothing either way. One draw is one
   observation from the null, not a calibration of it — it neither
   establishes nor rules out a pipeline defect. Establishing that the
   real-activation pipeline is calibrated needs REPEATED seed-pair draws
   (a rejection rate over many random-vs-random pairs), which has not
   been run; the synthetic calibration in `p_i5_gate.py` supports a
   separate claim about the statistic under its synthetic null. What
   finding 1 shows, on the two controls measured: **mean-ablation of
   `L4H6` and of `L5H3` each beats an ISOTROPIC random direction of
   matched magnitude as decisively as `L3H6` does.** Whether that holds
   for heads generally is the natural reading and is NOT measured — a
   representative head sweep would be needed to say so. A real,
   trained direction is structured; concentration of measure in
   `d_head = 64` dimensions makes a uniformly random direction generically
   nearly orthogonal to whatever subspace a downstream reading is
   sensitive to, so "structured vs isotropic-random" is close to a
   free win for the structured arm regardless of whether the structure is
   induction-relevant. The control this project's registry names
   ("matched-magnitude random-direction ablation") is under-specified in
   exactly the way that matters: matching the NORM is not enough when the
   comparison needs to isolate a DIRECTION'S relevance, not merely
   confirm it is a direction at all.

3. Seed sensitivity, checkpoint replication and a `cosine_distance`
   cross-check all landed as expected (stable across 5 seeds, p = 0.021
   mean; replicates at `step64000`, p = 0.0078; agrees with `raw_distance`
   under `cosine_distance`, p = 0.0273) — worth keeping on record, but
   they answer "is the reading stable," not "is the reading specific,"
   and specificity is what finding 1 shows is missing.

WHAT THIS MEANS FOR `P-I5`. `claims/registry.json` is unchanged — nothing
here was ever close to being registered as an adjudication, and this
finding is exactly why that discipline exists. The real next step is not
running the existing pipeline further (more prompts, more checkpoints)
but fixing what the control compares against: a null distribution for
"an unstructured direction" needs to be unstructured RELATIVE TO WHATEVER
THE READING IS SENSITIVE TO, not merely isotropic in the ambient
`d_head`-dimensional space — e.g. drawn from the empirical distribution
of other heads' own output directions, or from a random combination of
directions the residual stream already occupies, rather than a fresh
Gaussian draw. That redesign is not attempted here; naming it precisely
is this module's contribution.

WHAT IS STILL NOT DONE, beyond the control redesign above: a power
analysis (would a fixed design reliably detect a real effect of some given
size at n=8?) needs either an analytical treatment of `joint_rank_pvalue`'s
power (unbuilt) or a planted-effect simulation inside the real-activation
pipeline (expensive). Moot until the control itself is fixed — there is no
point measuring the power of a test that isn't measuring what it claims to.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

NEGATIVE_CONTROL_HEADS = [(4, 6), (5, 3)]
SENSITIVITY_SEEDS = [20260916, 1, 2, 3, 4]
SECOND_CHECKPOINT_MODEL = "pythia-70m-step64000"


# ---------------------------------------------------------------------------
# 1. Negative controls
# ---------------------------------------------------------------------------

def run_negative_controls(seed: int = 20260916) -> dict:
    from core.lm_loading import load_causal_lm
    from p7_motifs.p_i5_ablation import MODEL_NAME, TARGET_HEAD, run_all_on_loaded_model

    model, tokenizer = load_causal_lm(MODEL_NAME)

    out = {}
    for head in [TARGET_HEAD] + NEGATIVE_CONTROL_HEADS:
        label = f"L{head[0]}H{head[1]}"
        result = run_all_on_loaded_model(model, tokenizer, target_head=head, seed=seed)
        out[label] = {
            "is_target": head == TARGET_HEAD,
            "p_value": result["gate"]["p_value"] if result["gate"] else None,
            "p_value_min_rank_superseded": (result["gate_min_rank_superseded"]["p_value"]
                                            if result.get("gate_min_rank_superseded") else None),
            "n_prompts": result["n_prompts"],
            "mean_delta_geometric": float(np.mean(result["delta_geometric"])),
            "mean_delta_logit": float(np.mean(result["delta_logit"])),
            # Stored so a later statistic can re-score the record without a
            # model run; the 2026-09-16 records lack these and cannot be.
            "delta_geometric": list(map(float, result["delta_geometric"])),
            "delta_logit": list(map(float, result["delta_logit"])),
        }
    return out


# ---------------------------------------------------------------------------
# 2. Seed sensitivity (one random direction per prompt -- how much does
#    the draw move the reading?)
# ---------------------------------------------------------------------------

def run_seed_sensitivity(seeds=SENSITIVITY_SEEDS) -> dict:
    from core.lm_loading import load_causal_lm
    from p7_motifs.p_i5_ablation import MODEL_NAME, TARGET_HEAD, run_all_on_loaded_model

    model, tokenizer = load_causal_lm(MODEL_NAME)

    p_values = []
    for seed in seeds:
        result = run_all_on_loaded_model(model, tokenizer, target_head=TARGET_HEAD, seed=seed)
        p_values.append(result["gate"]["p_value"])

    return {
        "seeds": list(seeds),
        "p_values": p_values,
        "min": float(np.min(p_values)),
        "max": float(np.max(p_values)),
        "mean": float(np.mean(p_values)),
    }


# ---------------------------------------------------------------------------
# 2b. Diagnostic: two random directions against each other (neither arm
#     real) — is the statistic itself sound, or does anything beat anything?
# ---------------------------------------------------------------------------

def run_random_vs_random_diagnostic(seed_a: int = 100, seed_b: int = 200) -> dict:
    """
    Both arms are matched_magnitude_random_ablation draws (independent
    directions, seed_a vs seed_b) -- NEITHER is a real ablation. Under H0
    (no structured difference between two isotropic random directions)
    joint_rank_pvalue should NOT reject. If it does, the statistic itself
    (not just the choice of control) is suspect; if it doesn't, the
    negative-control failure (see module docstring) is about what the
    control is compared against, not about joint_rank_pvalue's validity.
    """
    from core.config import PROMPTS
    from core.lm_loading import load_causal_lm
    from core.battery_structure import induction_candidates
    from core.dual_reading import pairwise_geometric_reading
    from core.intervention import next_token_kl
    from p7_motifs.p_i5_ablation import (
        MODEL_NAME, TARGET_HEAD, HIDDEN_STATE_INDEX,
        draw_unit_direction, matched_magnitude_random_ablation, _forward,
    )
    from p7_motifs.p_i5_gate import DEGENERATE_PROMPT, intersection_union_pvalue, joint_rank_pvalue
    from tools.run.induction_rank_sweep import arch_dims, head_means
    import torch

    model, tokenizer = load_causal_lm(MODEL_NAME)
    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)

    deltas_geo, deltas_logit = [], []
    for key, text in PROMPTS.items():
        if key == DEGENERATE_PROMPT:
            continue
        clean = _forward(model, tokenizer, text)
        pairs = induction_candidates(clean["ids"])
        if not pairs:
            continue
        ids_tensor = torch.tensor([clean["ids"]])
        means = head_means(model, ids_tensor, [TARGET_HEAD])
        _, d_head, _ = arch_dims(model)

        dir_a = draw_unit_direction(rng_a, d_head)
        with matched_magnitude_random_ablation(model, TARGET_HEAD, dir_a, means):
            arm_a = _forward(model, tokenizer, text)
        dir_b = draw_unit_direction(rng_b, d_head)
        with matched_magnitude_random_ablation(model, TARGET_HEAD, dir_b, means):
            arm_b = _forward(model, tokenizer, text)

        h_clean = clean["hidden"][HIDDEN_STATE_INDEX]
        h_a = arm_a["hidden"][HIDDEN_STATE_INDEX]
        h_b = arm_b["hidden"][HIDDEN_STATE_INDEX]

        geo_a, geo_b, logit_a, logit_b = [], [], [], []
        for q, k in pairs:
            d_clean = pairwise_geometric_reading(h_clean[q], h_clean[k])["raw_distance"]
            d_a = pairwise_geometric_reading(h_a[q], h_a[k])["raw_distance"]
            d_b = pairwise_geometric_reading(h_b[q], h_b[k])["raw_distance"]
            geo_a.append(abs(d_a - d_clean))
            geo_b.append(abs(d_b - d_clean))
            pos = q - 1
            logit_a.append(next_token_kl(clean["logits"], arm_a["logits"], position=pos))
            logit_b.append(next_token_kl(clean["logits"], arm_b["logits"], position=pos))

        deltas_geo.append(float(np.mean(geo_a) - np.mean(geo_b)))
        deltas_logit.append(float(np.mean(logit_a) - np.mean(logit_b)))

    dg = np.array(deltas_geo)
    dl = np.array(deltas_logit)
    gate_greater = intersection_union_pvalue(dg, dl, alternative="greater")
    gate_two_sided = intersection_union_pvalue(dg, dl, alternative="two-sided")
    min_rank_greater = joint_rank_pvalue(dg, dl, alternative="greater")
    min_rank_two_sided = joint_rank_pvalue(dg, dl, alternative="two-sided")
    return {
        "seed_a": seed_a,
        "seed_b": seed_b,
        "n_prompts": len(deltas_geo),
        "delta_geometric": dg.tolist(),
        "delta_logit": dl.tolist(),
        "p_value_greater": gate_greater["p_value"],
        "p_value_two_sided": gate_two_sided["p_value"],
        "p_value_greater_min_rank_superseded": min_rank_greater["p_value"],
        "p_value_two_sided_min_rank_superseded": min_rank_two_sided["p_value"],
    }


# ---------------------------------------------------------------------------
# 3. Checkpoint replication
# ---------------------------------------------------------------------------

def run_checkpoint_replication(seed: int = 20260916) -> dict:
    from core.lm_loading import load_causal_lm
    from p7_motifs.p_i5_ablation import TARGET_HEAD, run_all_on_loaded_model

    model, tokenizer = load_causal_lm(SECOND_CHECKPOINT_MODEL)
    result = run_all_on_loaded_model(model, tokenizer, target_head=TARGET_HEAD, seed=seed)
    return {
        "model": SECOND_CHECKPOINT_MODEL,
        "p_value": result["gate"]["p_value"] if result["gate"] else None,
        "n_prompts": result["n_prompts"],
        "mean_delta_geometric": float(np.mean(result["delta_geometric"])),
        "mean_delta_logit": float(np.mean(result["delta_logit"])),
    }


# ---------------------------------------------------------------------------
# 4. raw_distance vs cosine_distance
# ---------------------------------------------------------------------------

def run_cosine_cross_check(seed: int = 20260916) -> dict:
    """
    Re-derives delta_geometric using cosine_distance instead of
    raw_distance, on the SAME forward passes p_i5_ablation.run_prompt
    already does the hard work for -- reimplemented here rather than
    threading a `metric` parameter through run_prompt, because this is a
    one-off cross-check, not a second production reading.
    """
    from core.config import PROMPTS
    from core.lm_loading import load_causal_lm
    from core.battery_structure import induction_candidates
    from core.dual_reading import pairwise_geometric_reading
    from p7_motifs.p_i5_ablation import (
        MODEL_NAME, TARGET_HEAD, ABLATION_MODE, HIDDEN_STATE_INDEX,
        draw_unit_direction, matched_magnitude_random_ablation, _forward,
    )
    from p7_motifs.p_i5_gate import DEGENERATE_PROMPT, intersection_union_pvalue, joint_rank_pvalue, attainable_floor
    from tools.run.induction_rank_sweep import arch_dims, head_means, ablate_heads
    import torch

    model, tokenizer = load_causal_lm(MODEL_NAME)
    rng = np.random.default_rng(seed)

    deltas_geometric = []
    deltas_logit = []
    for key, text in PROMPTS.items():
        if key == DEGENERATE_PROMPT:
            continue
        clean = _forward(model, tokenizer, text)
        pairs = induction_candidates(clean["ids"])
        if not pairs:
            continue

        ids_tensor = torch.tensor([clean["ids"]])
        means = head_means(model, ids_tensor, [TARGET_HEAD])
        with ablate_heads(model, [TARGET_HEAD], mode=ABLATION_MODE, means=means):
            real = _forward(model, tokenizer, text)
        _, d_head, _ = arch_dims(model)
        direction = draw_unit_direction(rng, d_head)
        with matched_magnitude_random_ablation(model, TARGET_HEAD, direction, means):
            control = _forward(model, tokenizer, text)

        h_clean = clean["hidden"][HIDDEN_STATE_INDEX]
        h_real = real["hidden"][HIDDEN_STATE_INDEX]
        h_control = control["hidden"][HIDDEN_STATE_INDEX]

        geo_real, geo_control = [], []
        for query, key_pos in pairs:
            d_clean = pairwise_geometric_reading(h_clean[query], h_clean[key_pos])["cosine_distance"]
            d_real = pairwise_geometric_reading(h_real[query], h_real[key_pos])["cosine_distance"]
            d_control = pairwise_geometric_reading(h_control[query], h_control[key_pos])["cosine_distance"]
            if d_clean is None or d_real is None or d_control is None:
                continue
            geo_real.append(abs(d_real - d_clean))
            geo_control.append(abs(d_control - d_clean))

        if not geo_real:
            continue
        deltas_geometric.append(float(np.mean(geo_real) - np.mean(geo_control)))

    n = len(deltas_geometric)
    dg = np.array(deltas_geometric)
    # No independent logit re-read needed -- reuse the already-committed
    # real-run's delta_logit (the logit readout doesn't depend on which
    # geometric metric is used); this cross-check is about the geometric
    # axis only.
    with open(ROOT / "claims" / "calibration" / "p_i5_real_ablation.json") as f:
        committed = json.load(f)
    dl = np.array(committed["delta_logit"])
    if len(dl) != n:
        gate = gate_min_rank_superseded = None
    else:
        gate = intersection_union_pvalue(dg, dl, alternative="greater")
        gate_min_rank_superseded = joint_rank_pvalue(dg, dl, alternative="greater")

    return {
        "metric": "cosine_distance",
        "n_prompts": n,
        "delta_geometric": dg.tolist(),
        "p_value": gate["p_value"] if gate else None,
        "p_value_min_rank_superseded": (gate_min_rank_superseded["p_value"]
                                        if gate_min_rank_superseded else None),
        "note": ("delta_logit reused from claims/calibration/p_i5_real_ablation.json "
                 "(same seed, same run) -- the logit readout is unaffected by which "
                 "geometric metric is used."),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "claims" / "calibration" / "p_i5_validation.json"
    result = {}

    def _checkpoint():
        # Written after each step -- a killed process (this machine's
        # memory watchdog is a documented risk, CLAUDE.md) still leaves
        # every step completed so far on disk, not just the last print.
        if args.write:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            tmp_path.replace(out_path)

    print("=== 1. Negative controls ===")
    result["negative_controls"] = run_negative_controls()
    for label, r in result["negative_controls"].items():
        tag = " (TARGET)" if r["is_target"] else ""
        print(f"  {label}{tag:9s} p={r['p_value']:.4f} "
              f"dG={r['mean_delta_geometric']:+.4f} dL={r['mean_delta_logit']:+.4f}")
    _checkpoint()

    print("=== 1b. Diagnostic: random vs random (neither arm real) ===")
    result["random_vs_random"] = run_random_vs_random_diagnostic()
    rvr = result["random_vs_random"]
    print(f"  p(greater)={rvr['p_value_greater']:.4f}  p(two-sided)={rvr['p_value_two_sided']:.4f}")
    _checkpoint()

    print("=== 2. Seed sensitivity (L3H6) ===")
    result["seed_sensitivity"] = run_seed_sensitivity()
    sens = result["seed_sensitivity"]
    print(f"  p-values: {sens['p_values']}")
    print(f"  min={sens['min']:.4f} max={sens['max']:.4f} mean={sens['mean']:.4f}")
    _checkpoint()

    print("=== 3. Checkpoint replication ===")
    result["checkpoint_replication"] = run_checkpoint_replication()
    ckpt = result["checkpoint_replication"]
    print(f"  {ckpt['model']}: p={ckpt['p_value']:.4f} "
          f"dG={ckpt['mean_delta_geometric']:+.4f} dL={ckpt['mean_delta_logit']:+.4f}")
    _checkpoint()

    print("=== 4. cosine_distance cross-check ===")
    result["cosine_cross_check"] = run_cosine_cross_check()
    cos = result["cosine_cross_check"]
    print(f"  p={cos['p_value']} (n={cos['n_prompts']})")
    _checkpoint()

    if args.write:
        print(f"wrote {out_path.relative_to(ROOT)}")
