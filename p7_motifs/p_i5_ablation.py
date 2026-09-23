"""
p7_motifs/p_i5_ablation.py — P-I5's gate, PART THREE: the matched-magnitude
random-direction ablation control, on real activations. First real numbers
for the joint statistic `p7_motifs/p_i5_gate.py` built and validated on
synthetic data.

STATUS: exploratory, not an adjudication. `claims/registry.json`'s `P-I5`
entry is untouched by this module — running the constructed gate for the
first time on real data is not the same thing as the pipeline being
validated the way `P-AB1`/`P-I3` validated theirs (a positive/negative
control pair, several checkpoints, a power check). See "WHAT THIS PASS
DOES NOT VALIDATE" at the end of this docstring for exactly what is still
open before a number from this module could be read as more than a first
look.

TARGET: `L3H6` on `pythia-70m`, at `step143000`. `p8_scale_ladder/
status-8.md` identifies `L3H6` as pythia-70m's induction/matcher head (the
70m analogue of 410m's `L7H8`) and `L2H1` as its previous-token partner
(the `L5H2` analogue) — see that file's "`L3H6` as pythia-70m's induction
head and `L2H1` as its previous-token partner" table. `P-I5`'s statement is
about ablating "an induction head", and the matcher is that head: it is
the one whose attention pattern IS the induction match, not the relay that
merely supplies it a previous-token signal.

ABLATION MODE: mean, not zero. `archive/PROJECT-start-here.md`'s resume block states the rule
this module follows without re-deriving it: "Zero-ablation's off-distribution
bias scales as 1/n_heads, so it distorts at 70m (8 heads/layer)... a
prediction... should name mean-ablation." `tools/run/induction_rank_sweep.py::
ablate_heads` is the shared instrument every `p7d_redundancy` script already
uses for this; this module imports it rather than reimplementing head
ablation a second time.

THE MATCHED-MAGNITUDE RANDOM-DIRECTION CONTROL, defined here for the first
time in this project (found not to exist anywhere else — see
`p_i5_gate.py`'s own history of the phrase). "Matched magnitude" is read as:
the control's per-position displacement from the clean value has the SAME
NORM the real (mean) ablation's displacement has AT THAT POSITION, applied
in a FIXED RANDOM DIRECTION instead of "regress to the mean". Concretely,
at every position the real arm replaces the head's `d_head`-dim output slice
with `mean_vector`; the control arm replaces the same slice with
`clean_value + direction * ||mean_vector - clean_value||` — same
displacement magnitude, different (unstructured) direction. `direction` is
drawn ONCE per (prompt, ablation site) — one random direction per prompt,
not per matched pair — matching `p_i5_gate.py`'s own design choice (§3.31,
POPPER_PLAN.md 6x): pairs inside one prompt share a forward pass and would
share the real arm's one ablation, so the control's one random draw per
prompt keeps the two arms symmetric rather than introducing a
finer-grained randomness the real arm doesn't have.

WHAT "delta_geometric" AND "delta_logit" MEAN HERE, per prompt (the
`(Dg_i, Dl_i)` `joint_rank_pvalue` consumes):

  Dg_i = mean_over_matched_pairs(|raw_distance(real) - raw_distance(clean)|)
       - mean_over_matched_pairs(|raw_distance(control) - raw_distance(clean)|)

  raw_distance is `core.dual_reading.pairwise_geometric_reading`'s field on
  the (query, key) matched-position pair, read from the residual stream
  immediately after the ablated layer (`hidden_states[L+1]`, standard
  HuggingFace convention: `hidden_states[0]` is the embedding,
  `hidden_states[k]` is the output after transformer block `k-1`). Absolute
  value because the prediction is that ablation moves the particles more
  than a magnitude-matched random push does — not a claim about which
  direction they move, which neither `P-I5`'s statement nor its falsifier
  names.

  Dl_i = mean_over_matched_pairs(KL(clean, real) at position query-1)
       - mean_over_matched_pairs(KL(clean, control) at position query-1)

  via `core.intervention.next_token_kl` (NOT `core.functional_distance` —
  see `p_i5_gate.py`'s corrected docstring for where that misattribution
  came from and was fixed). Position `query - 1`, not `query`:
  `core.battery_structure.induction_candidates`' pairs satisfy
  `ids[key-1] == ids[query-1]`, so the token induction PREDICTS is
  `ids[query]`, and the logits that predict it (HuggingFace's
  `logits[i]` predicts `token[i+1]`) live at position `query - 1` — "the
  logit at the copied token" in `P-I5`'s own wording is this position's
  distribution, not the query position's own.

Both deltas are non-negative-magnitude readings by construction, matching
`joint_rank_pvalue`'s "greater" alternative — H1 predicts the real arm's
magnitude exceeds the control arm's on both axes.

WHAT THIS PASS DOES NOT VALIDATE, stated so a p-value from it is not
mistaken for an adjudication:

  - No positive/negative control pair (e.g. ablating a head with a known
    null effect, to confirm the pipeline reads ~0 there before trusting a
    nonzero reading on `L3H6`).
  - One checkpoint only (`step143000`) — no replication across the training
    axis the rest of this project's `L3H6`/`L2H1` work already uses.
  - One random direction per prompt — no check of how much the result moves
    under a different draw (a nuisance-randomness sensitivity check, the
    kind `P-AB1`'s own construction needed before it was trusted).
  - No power analysis: whether this design could detect a real effect of
    plausible size is unmeasured, only the null's calibration is
    (`p_i5_gate.py`'s synthetic work).
  - `raw_distance` alone, not `cosine_distance` or a projector-restricted
    reading — the scale-sensitive one, chosen because it is the literal
    "pairwise-distance" `P-I5`'s statement names, but not cross-checked
    against the alternative here.

Building the missing validation is the natural next PR if this module's
first reading looks worth pursuing further; it is not built here because
doing so before looking at whether `L3H6` even produces a readable signal
would be exactly the kind of depth-before-breadth this project's own
working agreements (§3.29) flag as the standing hazard.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "pythia-70m-step143000"
TARGET_HEAD = (3, 6)  # L3H6: pythia-70m's induction/matcher head, status-8.md
ABLATION_MODE = "mean"  # required at 8 heads/layer per status-8.md's ablation-mode A/B
HIDDEN_STATE_INDEX = TARGET_HEAD[0] + 1  # residual stream right after the ablated layer
SEED = 20260916


# ---------------------------------------------------------------------------
# The matched-magnitude random-direction control
# ---------------------------------------------------------------------------

def draw_unit_direction(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


@contextlib.contextmanager
def matched_magnitude_random_ablation(model, head: tuple, direction: np.ndarray, means: dict):
    """
    Replace `head`'s output slice with `clean + direction * ||mean - clean||`
    at every position — same per-position displacement magnitude
    `tools.run.induction_rank_sweep.ablate_heads(mode='mean')` uses, same
    head, but a FIXED RANDOM direction instead of "regress to the mean".
    See module docstring for why this is the control P-I5's registered
    null_construction names.

    `direction` : (d_head,) unit vector. `means` : from
    `tools.run.induction_rank_sweep.head_means`, same shape contract that
    function's own callers already use.
    """
    import torch
    from tools.run.induction_rank_sweep import arch_dims

    L, H = head
    _, d_head, _ = arch_dims(model)
    sl = slice(H * d_head, (H + 1) * d_head)
    mean_vec = torch.as_tensor(means[(L, H)])
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
# One forward pass, clean or under a hook
# ---------------------------------------------------------------------------

def _forward(model, tokenizer, text: str) -> dict:
    import torch

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    ids = [int(i) for i in inputs["input_ids"][0]]
    hidden = [h[0].float().numpy() for h in out.hidden_states]
    logits = out.logits[0].float().numpy()
    return {"ids": ids, "hidden": hidden, "logits": logits}


# ---------------------------------------------------------------------------
# One prompt: clean, real-ablated, control-ablated -> (delta_geometric, delta_logit)
# ---------------------------------------------------------------------------

def run_prompt(
    model, tokenizer, text: str, rng: np.random.Generator,
    target_head: tuple = TARGET_HEAD, hidden_state_index: Optional[int] = None,
) -> Optional[dict]:
    """
    One prompt's contribution to the joint null: the real ablation arm vs
    the matched-magnitude random-direction control arm, aggregated over
    every induction-matched (query, key) pair in the prompt (mean absolute
    change per pair). Returns None if the prompt has no matched pairs
    (nothing to aggregate).

    `target_head` / `hidden_state_index` default to the module's own
    `L3H6` target — overridden by callers validating the pipeline against
    other heads (see `p_i5_validation.py`, e.g. a negative-control head
    with no real relationship to induction).
    """
    from core.battery_structure import induction_candidates
    from core.dual_reading import pairwise_geometric_reading
    from core.intervention import next_token_kl
    from tools.run.induction_rank_sweep import arch_dims, head_means, ablate_heads

    if hidden_state_index is None:
        hidden_state_index = target_head[0] + 1

    clean = _forward(model, tokenizer, text)
    pairs = induction_candidates(clean["ids"])
    if not pairs:
        return None

    inputs_ids = [clean["ids"]]
    import torch
    ids_tensor = torch.tensor(inputs_ids)
    means = head_means(model, ids_tensor, [target_head])

    with ablate_heads(model, [target_head], mode=ABLATION_MODE, means=means):
        real = _forward(model, tokenizer, text)

    _, d_head, _ = arch_dims(model)
    direction = draw_unit_direction(rng, d_head)
    with matched_magnitude_random_ablation(model, target_head, direction, means):
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

        pos = query - 1  # see module docstring: this is "the logit at the copied token"
        logit_real.append(next_token_kl(clean["logits"], real["logits"], position=pos))
        logit_control.append(next_token_kl(clean["logits"], control["logits"], position=pos))

    return {
        "n_tokens": len(clean["ids"]),
        "n_pairs": len(pairs),
        "delta_geometric": float(np.mean(geo_real) - np.mean(geo_control)),
        "delta_logit": float(np.mean(logit_real) - np.mean(logit_control)),
        "geo_real_mean": float(np.mean(geo_real)),
        "geo_control_mean": float(np.mean(geo_control)),
        "logit_real_mean": float(np.mean(logit_real)),
        "logit_control_mean": float(np.mean(logit_control)),
    }


# ---------------------------------------------------------------------------
# Full run: every informative prompt -> joint_rank_pvalue
# ---------------------------------------------------------------------------

def run_all_on_loaded_model(
    model, tokenizer, target_head: tuple = TARGET_HEAD, seed: int = SEED,
) -> dict:
    """
    Same as run_all, but against an already-loaded model — the form
    callers comparing several target heads (`p_i5_validation.py`) actually
    want, so the (slow-ish) model load happens once rather than once per
    head.
    """
    from core.config import PROMPTS
    from p7_motifs.p_i5_gate import DEGENERATE_PROMPT, intersection_union_pvalue, joint_rank_pvalue, attainable_floor

    hidden_state_index = target_head[0] + 1
    rng = np.random.default_rng(seed)

    per_prompt = {}
    for key, text in PROMPTS.items():
        if key == DEGENERATE_PROMPT:
            continue
        result = run_prompt(model, tokenizer, text, rng, target_head=target_head,
                             hidden_state_index=hidden_state_index)
        if result is not None:
            per_prompt[key] = result

    delta_geometric = np.array([r["delta_geometric"] for r in per_prompt.values()])
    delta_logit = np.array([r["delta_logit"] for r in per_prompt.values()])
    n = len(per_prompt)

    gate = intersection_union_pvalue(delta_geometric, delta_logit, alternative="greater") if n >= 1 else None
    gate_min_rank_superseded = joint_rank_pvalue(delta_geometric, delta_logit, alternative="greater") if n >= 1 else None

    return {
        "target_head": list(target_head),
        "ablation_mode": ABLATION_MODE,
        "seed": seed,
        "n_prompts": n,
        "per_prompt": per_prompt,
        "delta_geometric": delta_geometric.tolist(),
        "delta_logit": delta_logit.tolist(),
        "gate": gate,
        "gate_min_rank_superseded": gate_min_rank_superseded,
        "attainable_floor_one_sided": attainable_floor(n, "greater") if n >= 1 else None,
    }


def run_all(seed: int = SEED, target_head: tuple = TARGET_HEAD) -> dict:
    from core.lm_loading import load_causal_lm

    model, tokenizer = load_causal_lm(MODEL_NAME)
    result = run_all_on_loaded_model(model, tokenizer, target_head=target_head, seed=seed)
    result["model"] = MODEL_NAME
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    result = run_all(seed=args.seed)
    print(f"n_prompts={result['n_prompts']}")
    for key, r in result["per_prompt"].items():
        print(f"  {key:22s} n_pairs={r['n_pairs']:5d} "
              f"dG={r['delta_geometric']:+.4f} dL={r['delta_logit']:+.4f}")
    if result["gate"] is not None:
        print(f"joint_rank_pvalue: p={result['gate']['p_value']:.4f} "
              f"(floor={result['attainable_floor_one_sided']:.4f})")

    if args.write:
        out_path = ROOT / "claims" / "calibration" / "p_i5_real_ablation.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_path.relative_to(ROOT)}")
