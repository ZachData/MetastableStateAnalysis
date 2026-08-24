# Phase 6 — STATUS

**Last verified:** run date recorded only as "2026-04-xx" in source (incomplete timestamp —
backfill exact date from `phase6_report.txt` file metadata if needed).
**Overall: PARTIAL RUN, MISLABELED AS "NOT STARTED."** `README_phase6.md`'s header says "Not
started," but the same file's own "First Run Results" section and `phase6_report.txt`
confirm one model (albert-xlarge-v2 only) has been run. Treat header as stale.

## Verdict table (albert-xlarge-v2 only — no other model tested)

**0/6 tested predictions passed. 6/12 predictions did not run at all.**

| Track | Status |
|---|---|
| Track A (imaginary subspace / relational computation: `head_classify`, `qk_decompose`, P6-C1, P6-DD1/DD2) | **No data.** Skipped due to missing prerequisites (`qk_logit_matrices`, `qk_matrices` never populated) and un-threaded `model`/`tokenizer`/`text`/`hook_targets` for the dissociation test. |
| P6-R2 / P6-R4 (LDA alignment, linear probes) | **Ran — result inverted from prediction.** Mean LDA alignment with $U_A$ (imaginary) = 0.887 vs. $U_\text{neg}$ (real repulsive) = 0.067. 0/49 layers show the predicted direction. Probe accuracy: real-only 0.152 (chance level), imaginary-only 0.564 (near full-activation 0.590). |
| P6-R5 (local contraction) | **Partial pass.** 29/44 plateau steps contract in real subspace; 28/44 show neutral rotation in imaginary — as predicted. Merge destabilization fails: only 121/341 merge steps show the predicted real-subspace expansion. |

## Known blockers

1. `_compute_qk_logit_matrices` in `run_6.py` not confirmed to populate `ctx["qk_matrices"]`
   / `ctx["qk_logit_matrices"]` correctly — blocks all of Track A.
2. `model`, `tokenizer`, `text`, `hook_targets` not threaded into `ctx` before the
   dissociation subexperiment registers — blocks the double-dissociation causal test (the
   single most falsifiable prediction in the phase).
3. **`eigenspace_degeneracy.py` — NameError, `d` undefined** (project-wide known bug list;
   fixed in v2 execution-order item 4, "Freeze-for-deletion, bug fixes, oracle + smoke
   tiers" — alongside phase 3's freeze, not as a phase-6-specific task. Needs a named
   regression test first, per policy, and is a candidate for the oracle-tier suite since a
   degeneracy-ratio computation has a known-correct answer on planted synthetic clusters).
4. **`write_subspace.py` — `channel_orthogonality` called with unsupported `top_r` kwarg**
   (project-wide known bug list; same item-4 fix location as above).
5. ~~Two competing explanations for the R2/R4 inversion, neither ruled out yet:~~ **SETTLED
   2026-08-24, and by a third explanation that was not on this list.** Kept verbatim below,
   because the archive's third rule is that archiving the code does not retract the findings
   and that includes retracting a diagnosis honestly.

   Original: (a) a projector-construction error in `subspace_build.py` (Schur block
   mislabeling, swapping $U_\text{neg}$ and $U_A$) that would invert all four geometry tests
   together; (b) the real/imaginary functional-separation hypothesis genuinely doesn't hold
   under ALBERT's weight-tying, where one OV matrix implements both channels.

   **(a) is RULED OUT.** `tools/audit_p6_projector_labels.py` runs this file's Schur
   partition against matrices with planted real/imaginary structure (every bucket recovers
   its own span to 3.3e-08 rad) and against a classification taken from `np.linalg.eigvals`
   without touching the Schur form (bucket sizes match exactly). Two deliberate
   mislabelings are caught. Record: `claims/audits/p6_projector_labels.json`.

   **(b) cannot be assessed from these numbers, and neither could (a) have been.**
   `p6_subspace/math-6.md` §7.2's explanation (c) — the comparison is not
   dimension-normalized — is the binding one. $\mathbb{E}[\lVert P_U v\rVert^2] = \dim U/d$,
   and this file's own resolution order (§2: $U_A$ loses span($U_S$), $U_\text{neg}$ loses
   span($U_\text{pos}$)) makes $U_\text{neg}$ the doubly-shrunk bucket. At
   `albert-xlarge-v2`'s exact shape the audit measures $\dim U_A/\dim U_\text{neg} = 24.9$
   against the observed alignment ratio of $0.887/0.067 = 13.2$. **The correction is larger
   than the effect.** Chance-normalized the numbers read 0.960 for $U_A$ and 1.805 for
   $U_\text{neg}$ — the *predicted* direction — though those dims come from random OV
   matrices at ALBERT's shape, not ALBERT's weights, so they bound the correction rather
   than reporting a result.

   **What would settle it is one number this phase already computed and never reported:**
   the actual per-layer $\dim U_A$ and $\dim U_\text{neg}$. `_build_for_layer` returns both
   on every run.

   Blocker 6 below stands and is now the second-order concern, not the first.
6. ALBERT-specific caveat on P6-R5: the 0/49-layer inversion is not 49 independent
   measurements — same projector, 49 activation snapshots from the same shared OV weights.
   Result is weaker evidence than it would be for a non-weight-tied model.

7. **`dissociation.py::run_intervened_forward` not migrated to `core/intervention.py`'s
   `run_model_with_hook` (item 3, complete).** The safe half of that migration — one
   architecture, no ALBERT dispatch, unlike `causal_tests.py`'s — described but not done.
   Migrating it also fixes a latent embedding-index mismatch as a side effect:
   `dissociation.py` currently skips the embedding layer (`hidden_states[1:]`), while
   `core/models.py`'s own extraction functions (and `run_model_with_hook`) include it at
   index 0. This was internally consistent within `dissociation.py` alone (baseline and
   intervention runs were always compared to each other, never to externally-supplied
   labels), but is a real misalignment risk if `ctx["baseline_labels"]` is ever supplied
   from genuine Phase 1 output. Note this does not by itself fix blocker 2 above
   (`model`/`tokenizer`/`text`/`hook_targets` not threaded into `ctx`) — that's separate
   wiring work in `run_6.py`.

## Not yet done

- Re-run with Track A prerequisites fixed, before drawing further conclusions from R2/R4.
- Run on any non-ALBERT model — current result is ALBERT-only, and the ALBERT-specific
  weight-tying caveat (blocker 6) means the inversion needs a per-layer-weight model to be
  read as a real functional-separation failure rather than an ALBERT artifact.
- Falsification-table retrofit not needed here — this phase already has one (12-row table
  in the README), unlike Phases 1/2/5.
- Migrate `dissociation.py` onto `run_model_with_hook` (blocker 7) — lower risk than
  `causal_tests.py`'s migration, no dispatch logic needed.
