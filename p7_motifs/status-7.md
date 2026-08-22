<!-- p7_motifs/status-7.md -->
# Phase 7 — STATUS

**Last verified:** 2026-08-22 (oracle tier only — nothing has executed against any model).
**Overall:** Design plus the first two build steps. The artifact contract, the typed-edge
primitive and the motif alphabet exist and pass their oracle tier; no run has touched a
model, and no prediction is adjudicated.

## Verdict table

No verdicts. The four registered predictions (P-I1, P-I2, P-I3, P-I4 — see `PREDICTIONS.md`)
are outstanding, and by construction they were registered before any Phase 7 code was
written, matching the discipline used for Phase 1c and Phase 2d.

| Prediction | State |
|---|---|
| P-I1 (co-emergence of motif and behavior) | not run |
| P-I2 (channel asymmetry between stages) | not run |
| P-I3 (cross-head association, control arm required) | not run |
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
   the `motif_counts.json` assembly. Verdicts go through `core/nulls.py`'s `nsigma_verdict`.
   31 tests. The null *values* still have to be produced by `core/qk_offset_null.py` against
   real weights — this module adjudicates them, it does not generate them.
5. **`events.py`** — capture / hold / escape / relay_target / moved_fraction as `extra__`
   columns on `ParticleTable`.
6. **`interaction_graph.py`** — the producer: build typed edges from activations plus
   Phase 2/2b projectors. Needs a real forward pass, so it is the first module that cannot
   be fully verified without a model.
7. **`formation_curve.py`**, **`p7_io.py`**, **`run_7.py`**.
8. **Smoke tier** — one prompt, two checkpoints, tiny GPT-NeoX, end to end.
9. **The checkpoint sweep.** Not before.

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
4. **An absent edge is not a zero-force edge.** Edge tables are `n_tokens²` per head per
   layer per checkpoint and will be thinned. `InteractionTable.retention` carries the cutoff
   in the artifact itself, and `concat` refuses to merge tables thinned differently rather
   than silently picking one — two such tables cannot be counted together without a row
   meaning different things in each.

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
