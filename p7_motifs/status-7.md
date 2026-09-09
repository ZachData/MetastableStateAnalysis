<!-- p7_motifs/status-7.md -->
# Phase 7 — STATUS

**Last verified:** 2026-08-31 (oracle tier plus the driver — still nothing has executed
against any model).
**Overall:** Design plus build steps 1-8, both halves. The artifact contract, the typed-edge
primitive, the motif alphabet, the statistics, the IO layer, the event level, the
producer and now the driver exist and pass their oracle tier. The driver has been run
end to end against synthetic Phase 1 / Phase 2 artifacts and a real tokenizer, and
produces a contract-valid `interaction_table.npz`; it has not been run against a real
forward pass, which is step 9. No prediction is adjudicated.

## Verdict table

No verdicts. The four registered predictions (P-I1, P-I2, P-I3, P-I4 — see `PREDICTIONS.md`)
are outstanding, and by construction they were registered before any Phase 7 code was
written, matching the discipline used for Phase 1c and Phase 2d.

| Prediction | State |
|---|---|
| P-I1 (co-emergence of motif and behavior) | not run |
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
   indices, wrong offset sign, pair types keyed the other way). It has not been run against a
   real forward pass — that is the smoke tier.
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
10. **The checkpoint sweep.** Not before.

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
   wrong answer for any square matrix that is not one — `UPDATE_PLAN.md` §5.6's failure mode
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
