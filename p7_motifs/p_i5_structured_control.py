"""
p7_motifs/p_i5_structured_control.py — P-I5's gate, PART FIVE: the
redesigned control §3.33/`POPPER_PLAN.md` 6z named — and its own failure,
which points at a deeper problem than the control alone.

**BOTH CONSTRUCTIONS IN THIS MODULE ALSO FAIL TO DISCRIMINATE, and the
second one's failure is diagnosed down to the geometric readout's own
mechanics, not just the control's.** Read this before the two
constructions below — the headline is the diagnosis, not either fix.

`p_i5_ablation.py`'s control (an isotropic random direction, matched only
in magnitude) does not discriminate `L3H6` from heads with no relationship
to induction (§3.33) — mean-ablation of essentially any head beats
isotropic noise, because a real trained direction is structured and a
uniformly random direction of the same magnitude is generically almost
orthogonal to whatever a downstream reading is sensitive to.

CONSTRUCTION 1: OTHER-HEAD-DIRECTION, MAGNITUDE-MATCHED. Draw the control
direction from OTHER HEADS' OWN OUTPUT DIRECTIONS instead of `N(0, I)`:
one other head drawn uniformly from the model's remaining
`n_layers * n_heads - 1` heads (47 at pythia-70m), its own mean output
vector normalized to a unit direction, used exactly where the isotropic
draw was — same per-position displacement magnitude
(`other_head_direction_ablation`). This is "random direction" read as "a
real direction, drawn at random from the population of real directions
this site actually sees," the same move `P-ST1`'s retired null made
replacing a matched-dimension random subspace with a matched-occupancy one
(`POPPER_PLAN.md` 6m).

**Result: still fails.** `L4H6` p = 0.0078 (MORE significant than before),
`L5H3` p = 0.0156, `L3H6` p = 0.0234 — same pattern as the isotropic
control, structure alone does not fix it.

CONSTRUCTION 2 (DIAGNOSTIC): CONSTANT-SUBSTITUTION SWAP
(`run_constant_substitution_diagnostic`). Both arms use full constant
substitution (`ablate_heads(mode="mean")`, exactly the real arm's own
mechanism) — real substitutes `target_head`'s own mean, control substitutes
a randomly drawn OTHER head's mean. Not magnitude-matched at all; it
isolates a different, sharper question: with intervention TYPE held fixed,
does WHICH constant matters?

**Result: `delta_geometric` is ~1e-8 for every prompt — floating-point
noise, not a null finding about induction.** This is a property of the
READOUT, not the control. `pairwise_geometric_reading` reads the full
512-dim residual stream at two positions; the ablated head is one fixed
64-dim slice of it. Under constant substitution, THAT SLICE BECOMES
IDENTICAL AT EVERY POSITION — real or donor constant, it doesn't matter —
so its contribution to the (query, key) PAIRWISE DIFFERENCE is exactly
zero either way. `raw_distance` on the full residual stream cannot tell
"the right constant" from "a wrong constant" under this intervention
type; it can only detect "was a slice unified across positions or not,"
which both arms do identically. This is a real, mechanistic reason
§3.33's original construction (which does NOT do constant substitution —
it displaces each position by its own magnitude, so this specific
cancellation does not apply there) still leaves the field open: the
readout itself needs to be sensitive to *content*, not just *whether
variance was removed*.

`delta_logit` under construction 2 is also informative and goes the WRONG
way for `P-I5`: negative on every single prompt (donor-substitution is
MORE disruptive to next-token prediction than the target's own mean,
consistently) — plausibly because a foreign head's mean is further
out-of-distribution for whatever the downstream computation expects than
the target's own mean is, which is a statement about how "unusual" the
substitute is, not about induction relevance.

WHAT THIS ADDS UP TO. Three constructions tried, three failures to
discriminate, and the third one's failure has a clean mechanistic cause
rather than an unexplained number. The open problem is no longer only
"what control isolates directional relevance" (§3.33's framing) — it now
also includes "does `raw_distance` on the ablated layer's own residual
stream even have the sensitivity this test needs, once the intervention
removes cross-position variance in the ablated slice." A geometric
readout at a LATER layer (downstream of where the ablated slice's
information would need to propagate through further mixing) may not
suffer the same exact-cancellation, and is the next thing to try — named
here, not built.

STILL NOT A REGISTRATION. `claims/registry.json`'s `P-I5` entry stays
untouched regardless of what this module finds — see §3.33's own
discipline note.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The other-head-direction control
# ---------------------------------------------------------------------------

def all_heads(model) -> list:
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    return [(L, H) for L in range(n_layers) for H in range(n_heads)]


def draw_other_head_direction(rng: np.random.Generator, target_head: tuple, heads_pool: list, means: dict) -> tuple:
    """
    Pick one head from `heads_pool` other than `target_head` uniformly at
    random and return (unit_direction, chosen_head). `means` must already
    contain every head in `heads_pool` (one `head_means` call over the
    whole pool, not one call per draw — the clean forward pass is shared).
    """
    candidates = [h for h in heads_pool if h != target_head]
    idx = int(rng.integers(len(candidates)))
    chosen = candidates[idx]
    vec = means[chosen].numpy()
    return vec / np.linalg.norm(vec), chosen


@contextlib.contextmanager
def other_head_direction_ablation(model, target_head: tuple, direction: np.ndarray, target_means: dict):
    """
    Same mechanics as `p_i5_ablation.matched_magnitude_random_ablation` —
    replace `target_head`'s output slice with
    `clean + direction * ||mean(target_head) - clean||` at every position —
    but `direction` here is another head's own (normalized) mean vector,
    not an isotropic draw. `target_means` is `target_head`'s OWN mean (for
    the displacement-magnitude computation), not the donor head's.
    """
    import torch
    from tools.run.induction_rank_sweep import arch_dims

    L, H = target_head
    _, d_head, _ = arch_dims(model)
    sl = slice(H * d_head, (H + 1) * d_head)
    mean_vec = torch.as_tensor(target_means[(L, H)])
    dir_vec = torch.as_tensor(direction)

    def hook(mod, args):
        x = args[0].clone()
        clean = x[..., sl]
        disp_norm = torch.linalg.norm(mean_vec.to(x.dtype) - clean, dim=-1, keepdim=True)
        x[..., sl] = clean + dir_vec.to(x.dtype) * disp_norm
        return (x,) + args[1:]

    handle = model.gpt_neox.layers[L].attention.dense.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# One prompt, one target head
# ---------------------------------------------------------------------------

def run_prompt(model, tokenizer, text: str, rng: np.random.Generator, target_head: tuple) -> Optional[dict]:
    from core.battery_structure import induction_candidates
    from core.dual_reading import pairwise_geometric_reading
    from core.intervention import next_token_kl
    from p7_motifs.p_i5_ablation import ABLATION_MODE, _forward
    from tools.run.induction_rank_sweep import head_means, ablate_heads
    import torch

    hidden_state_index = target_head[0] + 1
    clean = _forward(model, tokenizer, text)
    pairs = induction_candidates(clean["ids"])
    if not pairs:
        return None

    ids_tensor = torch.tensor([clean["ids"]])
    pool = all_heads(model)
    means = head_means(model, ids_tensor, pool)  # every head, one clean pass

    with ablate_heads(model, [target_head], mode=ABLATION_MODE, means=means):
        real = _forward(model, tokenizer, text)

    direction, donor_head = draw_other_head_direction(rng, target_head, pool, means)
    with other_head_direction_ablation(model, target_head, direction, means):
        control = _forward(model, tokenizer, text)

    h_clean = clean["hidden"][hidden_state_index]
    h_real = real["hidden"][hidden_state_index]
    h_control = control["hidden"][hidden_state_index]

    geo_real, geo_control = [], []
    logit_real, logit_control = [], []
    for query, key in pairs:
        d_clean = pairwise_geometric_reading(h_clean[query], h_clean[key])["raw_distance"]
        d_real = pairwise_geometric_reading(h_real[query], h_real[key])["raw_distance"]
        d_control = pairwise_geometric_reading(h_control[query], h_control[key])["raw_distance"]
        geo_real.append(abs(d_real - d_clean))
        geo_control.append(abs(d_control - d_clean))

        pos = query - 1
        logit_real.append(next_token_kl(clean["logits"], real["logits"], position=pos))
        logit_control.append(next_token_kl(clean["logits"], control["logits"], position=pos))

    return {
        "n_pairs": len(pairs),
        "donor_head": list(donor_head),
        "delta_geometric": float(np.mean(geo_real) - np.mean(geo_control)),
        "delta_logit": float(np.mean(logit_real) - np.mean(logit_control)),
    }


# ---------------------------------------------------------------------------
# Full run, one target head, every informative prompt
# ---------------------------------------------------------------------------

def run_all_on_loaded_model(model, tokenizer, target_head: tuple, seed: int = 20260916) -> dict:
    from core.config import PROMPTS
    from p7_motifs.p_i5_gate import DEGENERATE_PROMPT, joint_rank_pvalue, attainable_floor

    rng = np.random.default_rng(seed)
    per_prompt = {}
    for key, text in PROMPTS.items():
        if key == DEGENERATE_PROMPT:
            continue
        result = run_prompt(model, tokenizer, text, rng, target_head)
        if result is not None:
            per_prompt[key] = result

    delta_geometric = np.array([r["delta_geometric"] for r in per_prompt.values()])
    delta_logit = np.array([r["delta_logit"] for r in per_prompt.values()])
    n = len(per_prompt)
    gate = joint_rank_pvalue(delta_geometric, delta_logit, alternative="greater") if n >= 1 else None

    return {
        "target_head": list(target_head),
        "seed": seed,
        "n_prompts": n,
        "per_prompt": per_prompt,
        "delta_geometric": delta_geometric.tolist(),
        "delta_logit": delta_logit.tolist(),
        "gate": gate,
        "attainable_floor_one_sided": attainable_floor(n, "greater") if n >= 1 else None,
    }


# ---------------------------------------------------------------------------
# Diagnostic: constant-substitution swap. Same intervention TYPE as real
# mean-ablation (the whole slice becomes one fixed vector at every
# position), but the fixed vector is a randomly drawn OTHER head's mean
# instead of the target's own -- see module docstring's addendum below for
# what this found and why.
# ---------------------------------------------------------------------------

def run_constant_substitution_diagnostic(target_head: tuple, seed: int = 20260916) -> dict:
    """
    Both arms use `ablate_heads(mode="mean")` -- the real arm with
    `target_head`'s own mean (exactly `p_i5_ablation.py`'s real arm), the
    control arm with a randomly drawn OTHER head's mean substituted INTO
    `target_head`'s slice. This is not magnitude-matched at all (a
    donor's mean can have any norm) -- it isolates a different question:
    does WHICH constant is substituted matter, once the intervention TYPE
    (full constant substitution, not a magnitude-matched displacement) is
    held fixed?

    FINDING (documented in the module docstring's addendum): delta_geometric
    is ~1e-8 (floating-point noise, not signal) for every prompt. This is
    NOT a null result about induction -- it is a property of the geometric
    readout's construction. `pairwise_geometric_reading` reads the FULL
    residual stream (512-dim) at two positions; the ablated head occupies
    one fixed 64-dim slice of it. Under constant substitution, that slice
    becomes IDENTICAL at every position (= the substituted constant, real
    OR donor), so its contribution to the (query, key) PAIRWISE DIFFERENCE
    is exactly zero regardless of which constant was used -- the
    difference is a property of the OTHER 448 dimensions, which no head
    ablation here touches. `raw_distance` on the full residual stream
    cannot distinguish "the right constant" from "a wrong constant" under
    this intervention type; it can only detect "was a slice UNIFIED across
    positions or not."
    """
    from core.lm_loading import load_causal_lm
    from core.config import PROMPTS
    from core.battery_structure import induction_candidates
    from core.dual_reading import pairwise_geometric_reading
    from core.intervention import next_token_kl
    from p7_motifs.p_i5_ablation import MODEL_NAME, ABLATION_MODE, _forward
    from p7_motifs.p_i5_gate import DEGENERATE_PROMPT, joint_rank_pvalue
    from tools.run.induction_rank_sweep import head_means, ablate_heads
    import torch

    model, tokenizer = load_causal_lm(MODEL_NAME)
    rng = np.random.default_rng(seed)
    hidden_state_index = target_head[0] + 1
    pool = [h for h in all_heads(model) if h != target_head]

    deltas_geo, deltas_logit = [], []
    for key, text in PROMPTS.items():
        if key == DEGENERATE_PROMPT:
            continue
        clean = _forward(model, tokenizer, text)
        pairs = induction_candidates(clean["ids"])
        if not pairs:
            continue
        ids_tensor = torch.tensor([clean["ids"]])
        means = head_means(model, ids_tensor, pool + [target_head])

        with ablate_heads(model, [target_head], mode=ABLATION_MODE, means=means):
            real = _forward(model, tokenizer, text)

        donor = pool[int(rng.integers(len(pool)))]
        with ablate_heads(model, [target_head], mode=ABLATION_MODE, means={target_head: means[donor]}):
            control = _forward(model, tokenizer, text)

        h_clean = clean["hidden"][hidden_state_index]
        h_real = real["hidden"][hidden_state_index]
        h_control = control["hidden"][hidden_state_index]

        geo_real, geo_control, logit_real, logit_control = [], [], [], []
        for query, key_pos in pairs:
            d_clean = pairwise_geometric_reading(h_clean[query], h_clean[key_pos])["raw_distance"]
            d_real = pairwise_geometric_reading(h_real[query], h_real[key_pos])["raw_distance"]
            d_control = pairwise_geometric_reading(h_control[query], h_control[key_pos])["raw_distance"]
            geo_real.append(abs(d_real - d_clean))
            geo_control.append(abs(d_control - d_clean))
            pos = query - 1
            logit_real.append(next_token_kl(clean["logits"], real["logits"], position=pos))
            logit_control.append(next_token_kl(clean["logits"], control["logits"], position=pos))

        deltas_geo.append(float(np.mean(geo_real) - np.mean(geo_control)))
        deltas_logit.append(float(np.mean(logit_real) - np.mean(logit_control)))

    dg = np.array(deltas_geo)
    dl = np.array(deltas_logit)
    gate = joint_rank_pvalue(dg, dl, alternative="greater") if len(dg) >= 1 else None
    return {
        "target_head": list(target_head),
        "n_prompts": len(dg),
        "delta_geometric": dg.tolist(),
        "delta_logit": dl.tolist(),
        "geometric_is_near_zero": bool(np.max(np.abs(dg)) < 1e-6) if len(dg) else None,
        "p_value": gate["p_value"] if gate else None,
    }


# ---------------------------------------------------------------------------
# The same three heads §3.33 tested, under the new control
# ---------------------------------------------------------------------------

def run_negative_controls(seed: int = 20260916) -> dict:
    from core.lm_loading import load_causal_lm
    from p7_motifs.p_i5_ablation import MODEL_NAME, TARGET_HEAD
    from p7_motifs.p_i5_validation import NEGATIVE_CONTROL_HEADS

    model, tokenizer = load_causal_lm(MODEL_NAME)
    out = {}
    for head in [TARGET_HEAD] + NEGATIVE_CONTROL_HEADS:
        label = f"L{head[0]}H{head[1]}"
        result = run_all_on_loaded_model(model, tokenizer, target_head=head, seed=seed)
        out[label] = {
            "is_target": head == TARGET_HEAD,
            "p_value": result["gate"]["p_value"] if result["gate"] else None,
            "n_prompts": result["n_prompts"],
            "mean_delta_geometric": float(np.mean(result["delta_geometric"])),
            "mean_delta_logit": float(np.mean(result["delta_logit"])),
        }
    return out


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "claims" / "calibration" / "p_i5_structured_control.json"
    result = {}

    def _checkpoint():
        if args.write:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("=== Negative controls, under the other-head-direction control ===")
    result["negative_controls"] = run_negative_controls()
    for label, r in result["negative_controls"].items():
        tag = " (TARGET)" if r["is_target"] else ""
        print(f"  {label}{tag:9s} p={r['p_value']:.4f} "
              f"dG={r['mean_delta_geometric']:+.4f} dL={r['mean_delta_logit']:+.4f}")
    _checkpoint()

    print("=== Diagnostic: constant-substitution swap (L3H6) ===")
    from p7_motifs.p_i5_ablation import TARGET_HEAD
    result["constant_substitution_diagnostic"] = run_constant_substitution_diagnostic(TARGET_HEAD)
    csd = result["constant_substitution_diagnostic"]
    print(f"  geometric_is_near_zero={csd['geometric_is_near_zero']} p={csd['p_value']}")
    _checkpoint()

    if args.write:
        print(f"wrote {out_path.relative_to(ROOT)}")
