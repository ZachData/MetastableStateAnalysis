# Phase 2b — STATUS

**Naming reconciliation (v2 plan, item 1):** this phase's own README and its result files
(`phase2i_results.json`, `phase2i_summary.txt`) call it "Phase 2i." The directory is
`p2b_imaginary/`, and the transition plan keeps directory names as canonical. Going forward,
**"Phase 2b" is the name to use in documentation and cross-phase references** ("Phase 2b's
global rotation-neutral result," not "Phase 2i's"). On-disk artifact filenames
(`phase2i_results.json`) are left as they are for now — renaming those is a separate,
unscoped action, not part of this documentation pass.

**Last verified:** 2026-04-29
**Overall:** Complete. All 7 models × 5 prompts = 35 combinations run. Block 2 skipped
universally (its precondition never triggered).

## Verdict table

| Test | Result |
|---|---|
| Block 1a — rotational spectrum | OV structurally dominated by complex eigenvalue pairs everywhere: 84–97.5% rotational energy across all 7 architectures. Universal, not model-specific. |
| Block 1b — causal isolation | **`rotation_neutral` in all 35/35 runs.** `elim_signed = 1.0` always; `elim_rotation = 0.0` always. |
| Block 2 — hemispheric tracking | Not run (conditional on Block 1b showing `rotation_contributes`; never triggered). |

**Headline finding:** the rotational component (84–97% of OV's spectral energy) is a
structural red herring for clustering — it has zero causal weight for energy violations.
The signed component (2–16% of spectral energy) carries 100% of causal weight. This does
not undermine Phase 2's conclusions; it confirms they aren't confounded by the imaginary
structure.

## Known issues / caveats

1. **ALBERT full-rescaling overcorrection.** `elim_full` is unreliable for ALBERT (shared
   weights) — can go negative (more violations after full rescaling than before). Use
   signed-only rescaling as the definitive test for ALBERT. GPT-2 full rescaling is clean.
2. **`gpt2` entry aggregation bug.** The `gpt2` (small) entry in `phase2i_results.json`
   appears to aggregate results from multiple GPT-2 sizes due to a naming collision in the
   runner. Per-model entries (`gpt2-large`, `gpt2-medium`, `gpt2-xl`) are authoritative.
3. Block 2 remains available if a future model/prompt shows rotational contribution — not
   removed, just unexercised.

## Not yet done

Nothing blocking. This phase is closed (see DESIGN.md for why it isn't reopened for Phase 2c).
