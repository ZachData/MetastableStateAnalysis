<!-- p7_motifs/status-7.md -->
# Phase 7 — STATUS

<!-- phase-card -->
## Card

- **Question:** Is a named mechinterp phenomenon, induction, the same object as a motif in the particles' typed interaction graph (a two-stage relay of attention forces, split into the value operator's sign channels), and does that motif form on the checkpoint axis when the behaviour does?
- **Inputs:** `pythia-410m`, the registered 19 checkpoints (step 0 to 143000), the v1 8-prompt battery (`camus_letranger+hdbscan_code+homer_iliad+latex_monograph+paper_excerpt+repeated_tokens+sullivan_ballou+wiki_paragraph`), `schur` sign channel, top 16 edges per target, rotational channel absent: `data/phase7/step*/interaction_table.npz` (generated 2026-09-01, 17 of 19 recompressed 2026-09-03: `archive/docs/results_provenance_audit_2026-09-05.md`). `P-I1` scored on `data/analysis/relay_null_series.json` and `data/analysis/behavioural_series.json`, hashed in `claims/audits/p_i1_real_run.json`. `P-I5` ran on `pythia-70m` step 143000, `L3H6`, `mean` ablation, 8 prompts (`claims/calibration/p_i5_real_ablation.json`)
- **Results:**
  - `P-I1` on the real 19-step sweep: INSUFFICIENT, 116 forming heads, mean distance about 2 log-steps, floor 0.0005; its p moves with the replicate count and is not quotable — `claims/audits/p_i1_real_run.json`, §3.7
  - The raw relay count is almost entirely the prompt's own induction-pair supply, so the gate scores an above-null excess from a relay-count null built for it — §3
  - `P-I5` built, calibrated and run, then parked: the control does not tell `L3H6` from other heads, and a single-head diagnostic explains the readout — §3.33, §3.34
  - The build, oracle tier through the driver: artifact contract, typed-edge primitive, motif alphabet, statistics, IO, events, producer, `run_7.py` — `p7_motifs/status-7.md` "Build order"
  - `P-I3`'s registered null permutes a label that is a function of the correlated score; replaced by a matched-control null that enumerates — `p7_motifs/status-7.md` "Findings from implementation"
  - A mean+kσ hub rule cannot find a hub at n ≤ 4; an absent projector is not a zero channel; a missing null is not a cleared one — `p7_motifs/status-7.md` "Findings from implementation"
- **Superseded / wrong:**
  - This file said nothing had run against a model and listed `P-I1` as "not run"; all 19 tables were written by 2026-09-03 and `P-I1` scored 2026-09-04 — `p7_motifs/status-7.md` "Corrections received"
  - "p = 0.1414" was quoted as `P-I1`'s result in four documents; it is 0.14 at K = 50 and 0.89 at K = 100 — §3.45
  - `P-I5`'s min-rank statistic controlled only the complete null; replaced by intersection-union, `L3H6` 0.0234 → 0.0312, five overclaims withdrawn — §3.38
  - `design-7.md` carries "cone collapse is universal" from a 1b result that 1b says not to cite — `p1b_hemisphere/status-1b.md`
  - `P-I5`'s target `L3H6` was chosen as 70m's `L7H8` analogue, but 70m has no relay-fed matcher (its matcher is `L0H3`, layer 0) — `p8_scale_ladder/status-8.md` "The self-repair chain at 70m"
- **Registry:** nine `H-BRIDGE` rows, all `active`, none adjudicated. `P-I1` e-value, real run recorded, INSUFFICIENT; `P-ST1`, `P-AB1`, `P-I3` e-value, built, calibrated, unrun (`P-AB1` and `P-I3` have no known-answer dry run); `P-I5` needs-null with a gate and a real-run record, parked, and it iterates the live battery (`STATE.md` Blocked 2); `P-I2`, `P-I4`, `P-I7` nothing built; `P-SA1` instrument frozen — `claims/EXPERIMENTS.md`
- **Depends on:** 1@54c1f37216, 2@5e9fc59e62, 8@8cc3fb223c
- **Feeds:** none
- **Open threads:**
  - The rotational channel is not wired: `U_S`/`U_A` absent and `real_frac`/`imag_frac` NaN in every table — `p7_motifs/status-7.md` "Build order"
  - Whether the battery supplies enough induction pairs on the NeoX tokenizer was never checked (`core/battery_structure.py`) — `p7_motifs/status-7.md` "Known blockers"
  - `P-I5`'s problem moved from the control to the readout; no construction found yet — §3.34
  - `P-ST1`, `P-AB1`, `P-I3` each need a sweep meeting its own pre-computed requirement — `claims/EVALUABILITY.md`
- **After Phase 10:**
  - Check whether the 19 tables on disk meet `P-I3`'s and `P-AB1`'s pre-computed requirements, without scoring them (free)
  - Decide `P-I5`'s battery (pin to the v1 keys or re-register) and its target (`L3H6` is not an induction head by attention score) (free, the user's decision)
  - Wire the rotational channel from 2b's Schur blocks and rebuild the tables (forward pass: `run_7.py`, about 16 min per checkpoint × 19)
- **Reviewed:** 2026-09-24 · body `01ebf7f87d`
<!-- /phase-card -->

**Registered predictions (9):** e-value — `P-ST1` (`steering_gate.py`),
`P-AB1` (`patching_gate.py`), `P-I1` (`formation_gate.py` +
`relay_count_null.py`), `P-I3` (`cross_head_gate.py`), each with a null built
and calibrated. `P-ST1`, `P-AB1` and `P-I3` have no run against real artifacts. `P-I1` **has**
been scored on the real 19-step sweep (INSUFFICIENT, 2026-09-04; the p is K-dependent and not quotable — 0.14143 at K = 50, 0.89355 at K = 100, §3.7), and the run is recorded in `claims/audits/p_i1_real_run.json`, which the registry's
`real_run_record` points to (2026-09-19; "E-value audit" below).
needs-null — `P-I5` (built, calibrated, run on real `L3H6`, and **parked**:
two head-comparison controls failed to discriminate it from `L4H6`/`L5H3` and a
single-head diagnostic explained the readout, `PROJECT.md` §3.31–§3.34; statistic
corrected to intersection-union 2026-09-17),
`P-I2`, `P-I4`, `P-I7` (nothing built), `P-SA1` (instrument frozen). Nulls and
evidence paths are `claims/registry.json`; the per-phase view is
`claims/EXPERIMENTS.md`.

**Last verified:** 2026-08-31 for the build; the runs since are in "Corrections received".
**Overall:** Design plus build steps 1-8, both halves. The artifact contract, the typed-edge
primitive, the motif alphabet, the statistics, the IO layer, the event level, the
producer and now the driver exist and pass their oracle tier. The driver was then run
on real Pythia artifacts, the full 19-step sweep, 2026-08-31 to 09-03 (`data/phase7/`),
and `P-I1` was scored on it. No prediction is adjudicated.

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-24.

- 2026-09-04 · the driver ran on all 19 `pythia-410m` checkpoints (`data/phase7/step*/`, v1 battery) and `P-I1` was scored; this file kept saying nothing had run · §2
- 2026-09-05 · `P-I1`'s p is not quotable: 0.14143 at K = 50, 0.89355 at K = 100 · §3.7
- 2026-09-05 · the tables' provenance: generated 2026-09-01 under `f395127` / `bbf7c0c`, 17 of 19 rewritten 2026-09-03 by a recompression pass · `archive/docs/results_provenance_audit_2026-09-05.md`
- 2026-09-17 · `P-I5`'s statistic is intersection-union, not min-rank; five overclaims withdrawn · §3.38
- 2026-09-19 · `P-I1`'s run record committed; four documents had quoted p = 0.1414 · §3.45
- 2026-09-19 · `P-I5` iterates the live battery, so since v2 it gates on 20 prompts, not the 8 it was calibrated on · `LESSONS.md`
- 2026-09-23 · `design-7.md`'s "cone collapse is universal" rests on a 1b result 1b says not to cite · `p1b_hemisphere/status-1b.md`
- 2026-09-24 · `P-I5`'s target, 70m `L3H6`, was picked as "the `L7H8` analogue" (§3.32, 2026-09-16), but Phase 8 had found on 2026-09-13 that 70m has no relay-fed matcher: no head above +1.0 has induction score over 0.024, and the only strong matcher is `L0H3`, in layer 0. `L3H6`'s induction score is 0.009 at both 16000 and 143000, `P-I5`'s own step (`wide`, `data/analysis/prev_token_profile_pythia-70m.json`); `P-I5` says "ablating an induction head". Found by the Phase 8 card review; recorded, not acted on · `p8_scale_ladder/status-8.md` "The self-repair chain at 70m"

## E-value audit, Phase 7 (2026-09-19)

The audit's **last** unit (`PROJECT.md` §3.45), after phases 1, 1c, 2/2d and
5b/6. Nine registered rows, all `active` — the largest live phase in the
registry and the only one where gates have been pointed at real artifacts.

| row | evaluable | gate | calibrated | run on real artifacts |
|---|---|---|---|---|
| `P-I1` | e-value | `formation_gate:p_value_p_i1` | ✓ | **✓ — recorded 2026-09-19** |
| `P-ST1` | e-value | `steering_gate:p_value_p_st1` | ✓ | — |
| `P-AB1` | e-value | `patching_gate:p_value_p_ab1` | ✓ | — |
| `P-I3` | e-value | `cross_head_gate:p_value_p_i3` | ✓ | — |
| `P-I5` | **needs-null** | `p_i5_gate:intersection_union_pvalue` | ✓ | ✓ (exploratory) |
| `P-SA1`, `P-I2`, `P-I4`, `P-I7` | needs-null | — | — | — |

**1. `P-I1`'s record now exists, and writing it is this unit's one change to
the tree.** `tools/score_p_i1.py` printed its result and returned — so the only
p-value this project had ever produced against real artifacts lived in stdout
and in `PROJECT.md`'s prose, with both inputs under git-ignored `data/`. That
is precisely the gap §3.36 named. The script now writes
`claims/audits/p_i1_real_run.json` with the full result, the sweep, the seed,
the replicate count and a sha256 of each input, following
`tools/score_claim_c.py`'s record-either-way convention; `real_run_record` is
set. Re-run today it reproduces: **verdict INSUFFICIENT, 116 forming heads, 0
skipped, floor 0.00050, mean distance 2.018 log-step.**

*`check_registry` refused the entry until the record was `git add`-ed* —
"an artifact only on this machine is not evidence a later reader can check" —
which is the §3.36 rule doing exactly its job, caught live.

**2. Four documents quoted a number §3.7 says is not quotable.**
`claims/EVALUABILITY.md`, this file, `PREDICTIONS.md` and §3.36 all carried
"p = 0.1414" as *the* `P-I1` result. §3.7 measured the p at two null sizes and
found **0.14143 at K = 50 against 0.89355 at K = 100 — same verdict** —
because 36 heads share one coset of the relay axis, and concluded in terms:
"§3.6's p = 0.1414 is not a quotable number, and neither is 0.8936." The four
citations are annotated rather than deleted; what they should have carried is
the verdict, the mean distance and the floor, all of which are robust. **A
p-value that moves by 0.75 with the replicate count is a statement about K.**

**3. `P-I5` is the registry's one internal contradiction, and it is
deliberate.** It is classified `needs-null` while naming a gate *and* carrying
a `real_run_record`; `check_registry` warns that `core/adjudication.py` will
refuse it, so the gate cannot reach a claim's e-process. That is correct and
wanted — §3.31–§3.34 built the joint statistic, then found the control does not
discriminate, so the row is parked with its instrument visible rather than
quietly downgraded. Worth stating plainly because the warning will keep
appearing: **it is a flag, not a defect**, and it clears only when the parked
control problem is solved or the row is retired.

**4. The four `needs-null` rows without gates are the honest ones.** `P-SA1`,
`P-I2`, `P-I4` and `P-I7` name no gate and claim no calibration, which is the
classification agreeing with the tree — the opposite of phases 5b/6's four
dormant `e-value` rows with nothing behind them (§3.44).

**What this unit did not do.** `P-ST1`, `P-AB1` and `P-I3` were not run.
Each is built, calibrated and unrun, and each needs a sweep satisfying its own
pre-computed requirement (`EVALUABILITY.md`, "What the pilot must produce") —
running them to see what comes out is the peek every other unit of this audit
has refused.

## Verdict table

No adjudicated verdicts. The table covers the four original rows of the nine (P-I1, P-I2, P-I3, P-I4 — see `PREDICTIONS.md`); all nine
are outstanding, and by construction they were registered before any Phase 7 code was
written, matching the discipline used for Phase 1c and Phase 2d.

| Prediction | State |
|---|---|
| P-I1 (co-emergence of motif and behavior) | scored 2026-09-04, INSUFFICIENT, not adjudicated (`claims/audits/p_i1_real_run.json`) |
| P-I2 (channel asymmetry between stages) | not run |
| P-I3 (cross-head association, control arm required) | not run — gate built 2026-08-30 |
| P-I4 (event consequence) | not run |

## Build order

Each step gated on the one before it, because each is what makes the next step's output
interpretable.

1. **Artifact contract first** — DONE. A `phase7` entry in `core/artifacts.py`'s `REGISTRY`
   (`interaction_table`, `motif_counts`, `formation_curve`), written *before* the producer
   existed. This is the v2 rule that exists to kill the producer/consumer-mismatch bug class;
   Phase 5's blockers 2 and 3 are both instances of what happens without it.
   `tests/test_core_interactions.py::TestArtifactContract` is what keeps declaration and
   producer from drifting.
2. **`core/interactions.py`** — DONE. The typed-edge primitive (`InteractionTable`,
   `projection_fractions`, `classify_pair_types`). Lives in `core/` and not here for the same
   reason `ParticleTable` does: it is a project-level object, not one phase's private
   structure. 33 tests.
3. **`motif_alphabet.py`** — DONE. The seven named motifs, `find_relays` / `relay_strength`
   for the two-stage composition. 36 tests including the planted-relay oracle, six negative
   controls, and random-graph null calibration.
4. **`motif_stats.py`** — DONE. Per-head rates, N1/N2/N3 gating, P-I3's control arm, and
   the `motif_counts.json` assembly. `cross_head_association` here reports the association
   descriptively; the p-value and its refusals live in `cross_head_gate.py` (finding 8). Verdicts go through `core/nulls.py`'s `nsigma_verdict`.
   31 tests. The null *values* still have to be produced by `core/qk_offset_null.py` against
   real weights — this module adjudicates them, it does not generate them.
5. **`p7_io.py`** — DONE. Reads Phase 2's sign-channel projectors and Phase 2b's rotation
   planes into the shapes the interaction table needs; writes `motif_counts.json` and
   `formation_curve.json` against their registered contracts. 22 tests.
6. **`events.py`** — DONE. capture / hold / hold_run / escape / relay_target /
   moved_fraction as `extra__` columns on `ParticleTable`. `hold_run` is Phase 5c's
   never-built `noise_tracking.py` primitive, and it is a groupby on the particle table
   exactly as the plan predicted. 29 tests.
7. **`interaction_graph.py`** — DONE (oracle tier). The producer: typed edges from
   activations, attention and the composed OV circuit, with the projectors `p7_io` supplies.
   31 tests, including an end-to-end planted relay that goes producer → alphabet → event
   level, which is the seam a individually-correct producer can still fail at (transposed
   indices, wrong offset sign, pair types keyed the other way). When written it had not been
   run against a real forward pass; step 10's sweep has since run it ("Corrections received").
8. **`run_7.py`** — DONE (2026-08-31). The driver, and the first thing in this phase
   with a caller: it joins a Phase 1 run directory (activations on the L2 sphere plus
   their norms, and the attention tensor) to a Phase 2 decomposition (composed OV
   circuits and sign-channel projectors) and writes `interaction_table.npz` against the
   contract step 1 registered. Until it existed nothing in the repository wrote that
   file outside a test, so `motif_stats`, `formation_gate` and `cross_head_gate` — all
   built, all passing — had no input. 26 tests, every refusal checked by constructing
   the mismatch rather than asserting the message.

   Three frames are resolved rather than assumed, and two of them refuse: the raw
   residual stream is reconstructed by multiplying the stored norms back and the run
   stops when a Phase 1 artifact predates that field, and which stored state is the
   input to layer *l* is read from `geometry.json`'s extraction convention, stopping
   when it is unrecorded. The rotational channel is NOT wired — `U_S`/`U_A` are None,
   `real_frac`/`imag_frac` are NaN, and the manifest records
   `rotational_channel: "absent"` so that a table missing the channel cannot be read as
   one measured to have none (finding 2). Supplying it needs Phase 2b's
   `extract_schur_blocks` for the same checkpoint.

   **`formation_curve.py`** — DONE (2026-08-31). Turns a checkpoint series of
   interaction tables into `formation_curve.json`. Two arguments are required with no
   default because both are the author's: `relay_owner`, because `relay_strength` is
   keyed by (layer_1, head_1, layer_2, head_2) while `P_I1_UNIT` is the head, so the
   collapse onto a head axis is a definition of what the motif measures; and
   `independence_source`, which the contract already required. 21 tests.

   It does NOT emit the above-null excess the gate requires, and says so:
   `above_null_excess` is stamped into the artifact and `assert_gate_ready` refuses a
   raw series. `core/qk_offset_null.py` computes N1/N2 for the QK antisymmetry
   statistic, not for relay counts, and **a relay-count null does not exist in this
   repository** — that is now the single named blocker between the sweep and P-I1's
   p-value.

   The behavioural score is computed from the attention tensor rather than from the
   table's `weight` column on induction rows. The table is thinned by a top-k-by-force
   cutoff, so averaging over it would select on force magnitude — the quantity the
   motif side is built from — and the two arms would share a selection step the
   pairing null cannot separate.
9. **Smoke tier** — one prompt, two checkpoints, tiny GPT-NeoX, end to end.
10. **The checkpoint sweep.** Not before. **Done** 2026-09-01 to 09-03, all 19 steps (`data/phase7/`).

## Findings from implementation

1. **A `mean + k·σ` hub rule cannot detect a hub in a small population, and the failure is
   structural rather than conservative.** The candidate inflates the very statistic it is
   compared against, and for *n* values the largest achievable z-score is
   (*n*−1)/√*n* — below 2 for every *n* ≤ 4. A single dominant attractor against four
   background particles scores *exactly* at a 2σ cutoff and is missed. `hub_mask` now
   excludes the candidate from its own baseline (leave-one-out mean and standard deviation),
   with `hub_flat_multiple` as the fallback when the leave-one-out spread is exactly zero.
   Found by the planted-attractor oracle test, not reasoned out in advance — which is the
   argument for having written the oracle first.
2. **"No projector supplied" and "no component in that channel" must not collapse.**
   `projection_fractions` returns NaN for an absent projector and 0.0 for a zero-magnitude
   force, and `motif_mask` reports `unknown_channel` alongside every count so a zero from
   "we never loaded the Phase 2 projectors" is distinguishable from an honest zero
   (standing rules 3 and 4).
3. **A missing null must not read as a cleared null.** `compare_against_nulls` returns
   REFUSED, not CONFIRMED, when a gating null is absent or empty. Phase 6's P6-I2 was broken
   in exactly this way — the stated null was not a floor and nothing in the output said so.
   Relatedly, P-I3's `independence_source` is a required positional argument rather than a
   keyword with a default: a result that cannot name what makes it independent of the
   behavioural induction score has measured that score twice.
4. **The two channels arrive in two incompatible shapes, and neither is the one the
   primitive originally assumed.** Phase 2's `weights.py` stores the attractive/repulsive
   split as (d, d) *symmetric idempotent projector matrices* (`P = Z @ Z.T`), while Phase
   2b's `top_rotation_planes` returns a *list of (d, 2) orthonormal plane bases* and
   deliberately never forms the projector — its own docstring records that doing so costs
   ~7 GB at d=1024 and ~27 GB at d=2048. `projection_fractions` was written expecting
   (d, r) orthonormal columns. It happens to return the right answer for a valid projector
   (‖Pᵀf‖² = ‖Pf‖² when P is symmetric idempotent) and would have returned a plausible
   wrong answer for any square matrix that is not one — `archive/UPDATE_PLAN.md` §5.6's failure mode
   exactly. It now accepts all three forms and *validates* rather than assuming, refusing a
   square matrix that is neither basis nor projector.
5. **`schur_*` vs `sym_*` is a choice, not a default.** Phase 2 stores both splits; Phase 2b's
   finding is that the symmetric part carries 100% of violation causality while the
   antisymmetric part is dynamically neutral, so they are not interchangeable and which one
   a result used changes what it means. `p7_io.load_sign_channel`'s `sign_channel` is a
   required argument, and it is stamped into every record.
6. **`moved_fraction` is a signed projection, not a magnitude ratio.** The obvious
   ‖motif_force‖ / ‖displacement‖ scores a large force *orthogonal* to the actual motion as
   highly explanatory (50× in the pinned test case) while it moved the particle nowhere
   along its path, and cannot distinguish a force driving the motion from one opposing it.
   ⟨force, displacement⟩ / ‖displacement‖² reads ~1 for aligned, ~0 for orthogonal, and
   negative when the motif pushed against where the particle actually went — a real and
   reportable outcome rather than an error to clip.
7. **An absent edge is not a zero-force edge.** Edge tables are `n_tokens²` per head per
   layer per checkpoint and will be thinned. `InteractionTable.retention` carries the cutoff
   in the artifact itself, and `concat` refuses to merge tables thinned differently rather
   than silently picking one — two such tables cannot be counted together without a row
   meaning different things in each.

8. **P-I3's registered null cannot be used, and its own tautology risk is what says so**
   (2026-08-30, `p7_motifs/cross_head_gate.py`, `POPPER_PLAN.md` §6s). An induction head is
   one whose behavioural induction score clears a cutoff, so "permutation over the head
   classification" permutes a label that is a deterministic function of the variable the
   prediction correlates against: exactly one of its 1.09e16 draws is a classification the
   definition permits, and measured, neither reading of the statistic can tell a planted
   effect from its absence. What replaces it compares each induction head against control
   heads matched on its own score, straddled above and below, and permutes the label within
   a matched set — a null that enumerates, so the p-value is exact. The tautology finding 3
   names then falls out as arithmetic: when the classification IS the thresholded score, no
   head can be straddled and the design floor is 1.000, decidable before an edge is counted.
   `p7_motifs/patching_gate.py` (P-AB1, 2026-08-27) is the other gate built since this list
   was last written.

## Two things the producer settled

**The (n², d) force tensor is never built.** Materializing every `f_ij = A_ij · (x_j @ OV_h)`
is ~4 GB for a single head at n=512, d=2048 in float64. It is also unnecessary: because
`A_ij ≥ 0` after softmax, `‖f_ij‖ = A_ij · ‖x_j @ OV_h‖`, so every edge's magnitude follows
from one (n, d) matmul and a row-norm — O(n·d²) once, then an outer product. Selection
happens on magnitudes before any force vector exists, and only retained edges' vectors are
formed. Same reasoning `top_rotation_planes` used to stop building (d, d) projectors, applied
on the other axis. `test_matches_the_brute_force_tensor` pins the identity against the naive
computation.

**Top-k is per target, not global.** A global cutoff lets a few high-norm particles consume
the whole budget and leaves others with no incoming edges — which does not read as "this
particle was not moved much", it reads as "this particle was not moved", and every per-target
motif (`hub`, `mutual`, both relay stages) would then be counted against a denominator that
silently varies by particle.

## Known blockers

1. **No Pythia artifacts on disk in this working tree.** Every step through the oracle tier
   is runnable without them; the smoke tier needs a tiny model; the real sweep needs the
   anchor-schedule runs. This is a sequencing fact rather than a defect.
2. **Phase 2/2b projectors are required for the force decomposition** (`U_pos`/`U_neg` for
   the sign channel, `U_S`/`U_A` for the rotational channel). They exist, and Phase 2's
   Pythia rerun is the active work — so the projectors this phase consumes are being
   regenerated concurrently. Do not build against a stale projector artifact; read the
   revision out of the manifest and refuse on a mismatch, per standing rule 4.
3. **Prompt-battery coverage for induction is not yet established on the NeoX tokenizer.**
   `core/battery_structure.py` exists precisely to answer this and has not been run against
   the current battery for this purpose. If too few prompts survive the four degeneracy
   checks, the first study is underpowered before it starts — this should be checked early,
   it is cheap, and it can change what prompts the sweep needs.

## Relationship to the rest of the project

- **`PREDICTIONS.md` claim (b)** — "resistance emerges at circuit-formation events" — is
  directly served by this phase's formation curve, which measures where one such event sits
  on the checkpoint axis rather than assuming the literature's anchors are right.
- **Not blocked on Phase 1c-B**, unlike Phase 2d. This phase needs Phase 2/2b projectors and
  the checkpoint anchors, both of which exist. It can run in parallel with the 1c work rather
  than queueing behind it.
- **Phase 5c's descoped Group B** — "what does GPT-2's fixed content-independent routing
  actually compute (induction, n-gram completion, skip-trigrams)?" — was deferred as a
  candidate "Phase 7" with `induction_ov.py` / `head_classify.py` named as starting points.
  This phase is not that phase. It shares the subject but not the frame: the question here is
  whether a motif of particle interactions is what the name "induction head" picks out, not
  what the routing computes. The archived modules are readable as prior art;
  per `archive/README.md` rule 2 they are not lifted.

## Not yet done

Everything past `design-7.md` and the `PREDICTIONS.md` addendum.
