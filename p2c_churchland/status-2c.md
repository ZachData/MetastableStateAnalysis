# Phase 2c — STATUS

**Last verified:** 2026-05-02 (run dirs `results/p2c_{model}_2026-05-02_*`)
**Overall: DOCUMENTATION GAP.** `readme-phase2c.md` header says "Not started." The results
file (`p2c-results.txt`) shows the run actually happened, across 7 models. Treat the header
as stale — this STATUS reflects the actual run output, not the README text.

## Verdict table (from `p2c-results.txt`, 7 models: albert-base-v2, albert-xlarge-v2,
bert-base-uncased, gpt2, gpt2-large, gpt2-medium, gpt2-xl)

| ID | Prediction | Result across models |
|---|---|---|
| P2c-J1 | jPCA R² ratio > 0.5 in ≥1 model | HOLDS for albert-base-v2 only (0.664); FAILS elsewhere (0.08–0.46) |
| P2c-J2 | jPCA planes within 30° of $U_A$ | FAILS universally (mean angles 86–87°, i.e. orthogonal) |
| P2c-T1 | A-channel tangling < S-channel | HOLDS for 5/7 models; FAILS for albert-base-v2, gpt2-medium |
| P2c-T2 | Induction prompts lower tangling | HOLDS for 6/7; FAILS for albert-base-v2 |
| P2c-K1 | Invariant variance in A, specific in S | FAILS universally |
| P2c-K2 | Stereotyped invariant change at merge layers | HOLDS only for albert-xlarge-v2; FAILS elsewhere |
| P2c-S1 | Plateau Jacobians more symmetric than V | FAILS universally |
| P2c-S2 | Merge Jacobians less symmetric than plateau | FAILS universally |
| P2c-M1 | A-channel magnitude scales with k | HOLDS for 4/7 (albert-xlarge-v2, gpt2, gpt2-large, gpt2-medium); FAILS for albert-base-v2, bert-base-uncased, gpt2-xl |
| P2c-M2 | A-channel direction is task-specific | HOLDS for gpt2-large, gpt2-xl only |
| P2c-M3 | Context-paired prompts diverge in A, agree in S | HOLDS for 4/7 |

**Reading:** the operator-side null from Phase 2b (rotation dynamically neutral for
clustering) largely reproduces on the trajectory side too — jPCA planes are orthogonal to
$U_A$ almost everywhere, and most of the specific division-of-labor predictions (K1, S1, S2)
fail outright. Signal is not zero (T1/T2/M1/M3 hold in a majority of models) but does not
cohere into the clean "rotational subspace does relational computation" story the phase was
built to test.

## Known blockers

1. **README status line is wrong** — says "Not started," contradicted by results data.
   Needs correcting once this STATUS is adopted as the source of truth.
2. `albert-xlarge-v2` appears twice in the results file (duplicate run block, identical
   numbers) — dedupe when re-parsing this file programmatically.
3. HDR follow-up (`P2c-J1-HDR`) mostly NOT RUN — only attempted for bert-base-uncased and
   gpt2-large (both FAILS), despite the design calling for HDR as a fallback when jPCA is
   borderline in any model.
4. No falsification-table retrofit note yet — this phase already has one (per v2 plan item
   12, was item 10 in v1; phases 1/2/5 need this format added, 2c already has it).

## Not yet done

C4 (slow points) and C1 (jPCA) were meant to run first per the recommended order in
DESIGN.md — actual results file doesn't indicate whether that order was followed; not
verifiable from the flat text report alone.
