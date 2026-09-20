<!-- INDEX.md -->
# Project index

**This file is the structural map: which phase lives in which directory, what is
archived, and what is referenced but absent. It is not the current state of the
work — that is `PROJECT.md`, which is the file to read first.**

## Current priority (updated 2026-09-20)

- **Start from `docs/AXES.md`** — new, project-wide: which axes exist, which cells are populated, six producers that do not exist, ten rules for combining rungs, and fourteen questions never asked (ten of them free). `INDEX.md` maps phases to directories; **that file maps questions to data.** Its headline: **`attentions.npz` is in 152/152 directories of the 410m sweep** and is largely unexploited, and the **`beta_eff` producer needs no forward pass** yet unblocks two registered predictions — gated on β's undecided unit convention, worth a factor of 8.
- **Before quoting the attention flip, read `p10_cluster_function/attention-10.md` and `math-10.md` §1 (`PROJECT.md` §3.49–§3.50).** Content-free, the causal mask already gives `received(j) = H_n − H_j` with layer mean exactly 1 — **6.155× at position 0 and 0.0038× at the last token, a ~1 600× tilt**, and the baseline hits 1.6× at position ≈53 and 0.5× at ≈160. **The observed flip is reproducible with zero content until that is divided out.** Row A0 gates everything.
- **`2411.04990` is READ — `docs/readings/2411.04990.md` (`PROJECT.md` §3.51).** The paper two phases depended on. **Thm 4.1: all tokens converge to `x₁(0)`, the first token's initial position, for arbitrary `Q, K`** — so no QK-side intervention, γ included, prevents collapse under `V = Id`. **Lemma C.1's centre count saturates in `n`** — that is Phase 1's carrying-capacity finding (50–55 invariant) with a formula. **The `d_eff` regression is the paper's own open conjecture**, with `d₁ = dim L` computable from Phase 2's projectors. **RMSNorm's diagonal is absorbable into `K, Q, V`**, so a γ-patch is not a read-side lever and `plan-9.md` §4.7 needs a `W_V` arm.
- **New convention: `docs/readings/<arxiv-id>.md`** for a paper read as primary text, marked **[R]** (`docs/LITERATURE.md` §0.1). First entry is `2411.04990`; the queue is `p10_cluster_function/lit-10.md` §10.
- **Phase 10 is open and is the current front (`PROJECT.md` §3.48).** `p10_cluster_function/notes-10.md`, then `lit-10.md`. *What clusters are and what they do* — the prerequisite Phase 9 could not skip. **Two rows of its ladder are free and unblocked today**: the Rényi-parking cluster-count prediction against 27 checkpoints of counts already on disk (rated the project's best cheap experiment by two independent reviews, and `claims/adjudications/` still holds zero entries), and the transport observables that no runner calls. **Verify the Jacobian-lens artifacts on the research machine** (`huggingface.co` is blocked from cloud sessions) — `notes-10.md` §4.5 says what may and may not be claimed from a `pythia-70m-deduped` lens.
- **Phase 9 is PARKED on Phase 10, not closed.** `p9_metric_intervention/notes-9.md` then `plan-9.md`; `notes-10.md` §9 states what each phase owes the other. Two amendments are marked inline in `plan-9.md` (§2.1a, §5.1a).
- **Scanning method changed (`docs/LITERATURE.md` §0.1):** `github.com` and `raw.githubusercontent.com` are reachable from a cloud session while `arxiv.org`, `transformer-circuits.pub`, `huggingface.co` and `neuronpedia.org` are not. **A paper's companion code is primary text even when the paper is not** — try it before settling for `[S]`, and mark what was actually read **[R]**.
- **The e-value audit is COMPLETE (`PROJECT.md` §3.36, §3.40, §3.43–§3.45).**
  Five units, thirty-nine registered predictions, **zero e-values**. One gate
  has been run on real checkpoints — `CLAIM-C`, three times, refusing three
  different ways (§3.41, §3.46) — and one p-value is recorded (`P-I1`,
  INSUFFICIENT, its p not quotable). **The pattern: the instruments are in
  better shape than the inputs, and wherever the inputs exist the blocker is a
  decision nobody has taken** — β's scale convention, `P-T1`'s wording,
  `P6-R2`/`R4`'s exchangeable unit, `P-S1`'s matched-k clustering. None costs a
  forward pass. `claims/EXPERIMENTS.md` is the per-phase view;
  `claims/EVALUABILITY_LOG.md` is the construction diary.
- **Direction (`PROJECT.md` §3.29, 2026-09-13, user): the programme is the
  particle/OT reading — mathematical perspective + optimal transport applied
  to mechanisms the network builds.** Induction heads (the whole thread
  below, Phase 7/7d/7e/8) are the *current instance*, not the object, and are
  now past diminishing returns as a place to keep digging. The same pass —
  why did this form, what were the mechanisms, what happened after — is to
  be run on SAE features, clusters, and other objects next. Read §3.29 before
  choosing where to work.
- **Live construction thread, PARKED not closed: `P-I5`'s joint permutation
  null.** `p7_motifs/p_i5_gate.py` (the statistic — the intersection-union
  test as of 2026-09-17; the earlier min-rank fix to an over-rejecting
  AND-corner controlled only the complete null, not the union null a
  conjunction needs), `p_i5_ablation.py`/`p_i5_validation.py`/
  `p_i5_structured_control.py` (on pythia-70m `L3H6`: two head-comparison
  controls that failed to discriminate it from `L4H6`/`L5H3`, and one
  `L3H6`-only diagnostic with a mechanistic explanation, `PROJECT.md`
  §3.31–§3.34). Next step named
  there (a later-layer geometric readout) when this becomes the priority
  again; `claims/registry.json` untouched throughout.
- **Just run: §2.5's isometric path on `L7H8` (`PROJECT.md` §3.35,
  `MATH_SPECTRAL_OT.md` §2.5.6).** The only *designed* particle
  intervention in this project. Real finding: `t=0` and `t=1` have
  identical singular values (exact isometry) but the transpose (`t=1`)
  stays as broken as the fully-symmetric midpoint rather than recovering —
  read/write alignment carries real causal weight beyond the spectral
  sign. `core/isometric_path.py` (pure math) / `tools/run/
  isometric_path_sweep.py` (real-model sweep). Next step named: §2.5.4's
  second family, holding both subspaces fixed and rotating only the
  correspondence.
- **Active work before the direction changed: Phase 8, the Pythia scale
  ladder.** `p8_scale_ladder/`, opened 2026-09-10 — the 7d/7e measurements
  repeated across model sizes so `n = 1` observations can become population
  claims. **Rung policy: explore on 70m and 410m, RESERVE 1b and 1.4b** (no
  induction measurement there until a prediction naming them is
  registered). Read `p8_scale_ladder/literature-8.md` first (2026-09-12 —
  the verified scan; the phase's original headline question is answered by
  `2407.10827` and the phase reframes onto the **geometry of the
  substitution**, with invariant 4 the load-bearing card), then
  `design-8.md`, then `status-8.md`. **First rung has run** (2026-09-11):
  the 48-head catalogue and invariants 2/4/5/6 on pythia-70m, each under
  two ablation modes. **Invariant 5 does not replicate; 2 and 4 do; 6 needs
  its ceiling-immune instrument.** `compare_rungs.py` holds the
  threshold-free cross-rung statistics — no absolute bar transfers between
  rungs, and the first write-up broke that rule before the A/B caught it.
  Not the current priority per §3.29, but not abandoned either — this
  thread is still where a new rung or invariant would be measured.
- **Read before running any ablation anywhere:** `status-8.md`'s
  ablation-mode A/B. Zero-ablation's off-distribution bias scales as
  `1/n_heads`, so it distorts at 70m (8 heads/layer) and barely touches 410m
  (16) — and **pythia-1b is also 8 heads/layer**, so a prediction registered
  against that reserved rung should name mean-ablation.
- **Phase 7d/7e — the 410m body of work the ladder generalises.**
  `p7d_redundancy/` (redundancy catalogue, both open axes closed 2026-09-10) and
  `p7e_consolidation/` (`L11H14`; whether the set collapses into one head). Read
  their `status-7d.md` / `status-7e.md`, then `PROJECT.md` §3.12-V.
- **Literature review, per phase, 2026-09-16** — **`docs/LITERATURE.md`** is the
  index; **`<phase>/lit-N.md`** is the review, and **every phase now has one**. Leads,
  not readings: `arxiv.org` and every other scholarly host are blocked by the session
  egress proxy, so nothing was read and every citation is marked `[S]` (search summary)
  or `[N]` (title only). **Read it before building on any phase's headline.** Three
  results are scooped — Phase 1's developmental arc (`2509.23024`), 7d's second-order
  ablation instrument (`2607.01940`), and 7d's "behavioural proxies fail" thesis
  (`2606.05378`) — and two of Phase 1's framing statements are stale. See
  `PROJECT.md` §3.37.
- **Literature scan, 2026-09-10** — `docs/literature_scan_2026-09-10.md`. The earlier,
  narrower scan; covers §3.12-V only. **Superseded in scope, not overturned** — all
  four of its verdicts are confirmed or sharpened by the 2026-09-16 review.
- **Phase 7 (original)** — the mechinterp/particle bridge, `P-I1`
  (induction-head formation as a two-stage `relay` motif). See `PROJECT.md` for
  where it stands and what is blocking, and `p7_motifs/design-7.md` for the
  translation table.
- **Phase 2's Pythia rerun is complete.** The 19-step registered sweep is on
  disk; `PROJECT.md` §1 has the layout.
- **Everything after Phase 2 has been archived.** Phases 3, 4, 5, 5b, 5c and 6 moved to
  `archive/` on 2026-08-22. They are not maintained, not imported by anything live, and
  their tests are not collected. Their findings stand and stay citable — see
  `archive/README.md`, which states the policy once so it does not have to be re-derived.
- **Checkpoint schedule: anchors only, pilot sweep not run.** Checkpoints at known event
  locations (the plan's provisional-anchor steps) are in use directly, not the dense item-8
  pilot sweep. Whether item 8 happens as its own pass is still undecided; not blocking.
- **SLT anchor and LLC: not current work.** The plan's "cheap SLT anchor" (per-checkpoint
  training-loss / weight-norm logging) is not implemented and is not being added. LLC stays
  dropped, with the same conditional revisit (good checkpoint data *and* an actual need).
  Flagged where it would otherwise get built: `p1_visualization/checkpoints.py` and
  `p2_eigenspectra/status-2.md`.

## Live phases

| Phase | Directory | State |
|---|---|---|
| 1 | `p1_mstate_tracking/` | Complete |
| 1b | `p1b_hemisphere/` | Complete |
| 1c | `p1c_frames/` | Implemented and validated on synthetic data. **Audited 2026-09-19 (§3.40): E run against Pythia artifacts (`P-H1` measured), A/B/F blocked on inputs no run directory carries** |
| 2 | `p2_eigenspectra/` | Complete; the Pythia rerun is done and the 19-step sweep is on disk |
| 2b | `p2b_imaginary/` | Complete *(directory name is canonical; on-disk artifacts still say "2i")* |
| 2d | `p2d_operator_activation/` | Implemented and validated on constructed operators; **not run**. **Audited 2026-09-19 (§3.43): inputs all present and the join verified on real artifacts** — blocked on Phase 1c-B by design, and on `P-T1`'s wording |
| 6 | `p6_subspace/` (live) / `archive/p6_subspace/` (frozen) | Ten rows dormant; **`P6-R2` and `P6-R4` active**, their projector path rebuilt live 2026-08-24. **Audited 2026-09-19 (§3.44): `P6-R2` needs `U_A`, which no artifact carries; `P6-R4`'s inputs exist and it needs an exchangeable unit the registry never named.** `status-6.md` |
| 7 | `p7_motifs/` | New — mechinterp phenomena as particle motifs. **Audited 2026-09-19 (§3.45): `P-I1`'s run is recorded at last; `P-ST1`/`P-AB1`/`P-I3` built, calibrated, unrun; `P-I5` parked.** See "Phase 7" below |
| 7d | `p7d_redundancy/` | **Active.** The redundancy set — which heads hold the induction regime and when each formed. Q1/Q2/Q3 answered; results in `PROJECT.md` §3.12-S/T/U. Every runner now takes `--model` and `--ablation` |
| 7e | `p7e_consolidation/` | **Active.** Whether the set collapses into one head; `L11H14`. Gate measurement answered — `status-7e.md` |
| 8 | `p8_scale_ladder/` | **Active.** The same measurements across the Pythia ladder. 70m rung run 2026-09-11; `compare_rungs.py` is the cross-rung reader. **Read `literature-8.md` before the next measurement** — the verified scan reframes the phase. See `status-8.md`, `PROJECT.md` §3.15–§3.16 |
| 9 | `p9_metric_intervention/` | **Notes and a plan. PARKED, pre-design, nothing frozen, nothing registered** — intervening on the *metric* the dynamics is read in rather than on the weights. Read `notes-9.md` then `plan-9.md`. **Parked on Phase 10**, whose answer it needs: `plan-9.md` §2–§3 moved there, and §2.1a / §5.1a carry two amendments found there — the functional partition is measurable per layer after all, and the Wasserstein-Hessian framing is at risk under the causal-mask theory. §4 onward is unaffected. Trigger 1 **not** discharged |
| 10 | `p10_cluster_function/` | **Notes, math, an audit and a two-pass scan. Pre-design, nothing frozen, nothing registered** — *what clusters are and what they do* (`PROJECT.md` §3.48–§3.50). `notes-10.md` (the trash-collection hypothesis, the J-lens), `attention-10.md` (the flip audited), **`math-10.md`** (five derivations, four `tools/math_checks/` files, 28 checks — three of them corrections to statements this repo makes), `lit-10.md` (two scan passes; §10 is the reading queue). Trigger 1 **not** discharged. No experiment code yet |

Shared code lives in `core/`; one-off scripts in `tools/`; tests in `tests/`.

**Project-wide maps in `docs/`:** `docs/AXES.md` — questions to data (which axes exist, what is
populated, which producers do not exist, and what has never been asked). `docs/LITERATURE.md` —
the per-phase prior-work index, and §0.1's note on which hosts a scan can actually reach.
`docs/readings/` — papers read as primary text, one file per arXiv id, marked **[R]**.

`p7d_redundancy/` and `p7_motifs/` are different programmes and are easy to
confuse. `p7_motifs/` is the motif/relay line behind `P-I1`; `p7d_redundancy/`
is the causal-ablation line behind §3.12. In particular
`p7_motifs/formation_curve.py` is a **behavioural** relay curve and
`p7d_redundancy/member_formation_curves.py` is the **causal** ablation curve —
similar names, different instruments. `7a`/`7b`/`7c` remain labels in
`PROJECT.md` §3.14.1 with no directory of their own; only `7d` grew one, when it
went active and started producing results.

Read `status-N.md` for the current state of a phase. Read `design-N.md` for the reasoning
that is not visible from the code. Read **`lit-N.md`** for what has been done before,
where the phase is scooped, and what is left to grow — `docs/LITERATURE.md` indexes all
sixteen. `PREDICTIONS.md` is the project-level falsification record, separate from any
single phase's.

**Which phase carries which e-value: `claims/EXPERIMENTS.md`.** Generated from
`claims/registry.json`, it is the phase → experiment → prediction → gate join that this
index (phases to directories) and `claims/FALSIFICATION.md` (claims to evidence) each hold
half of. It is also where the three phases with a live instrument and no registered
prediction are named — **7d, 7e and 8, which are the active work** — together with the two
adjudicable gates that never had a known-answer dry run (`P-AB1`, `P-I3`) and the one
declared claim nothing feeds (`H-BUDGET`). See `POPPER_PLAN.md` §6zb.

## Archived phases

Moved to `archive/` on 2026-08-22. Reason in every case: the project moved to Pythia
checkpoints and to the "particles first" framing, and this code predates both. Not a
verdict on the work.

Each archived phase also carries a `lit-N.md`: `archive/p3_crosscoder/lit-3.md`,
`archive/p4_mstate_features/lit-4.md`, `archive/p5c_unclustered/lit-5c.md`. Phases 5,
5b and 6 keep their reviews beside their study notes in the live top-level directories
(`p5_single_mstate_analysis/lit-5.md`, `p5b_manifold_steering/lit-5b.md`,
`p6_subspace/lit-6.md`). **A `lit-N.md` is not a reintroduction trigger** — it records
what the field has done since, so that a trigger in `FROZEN.md` is evaluated against
2026-09, not against the date the phase was frozen.

| Phase | Directory | What it found |
|---|---|---|
| 3 | `archive/p3_crosscoder/` | **Null.** Sparse crosscoder decoder directions align with V at chance (0.484 / 0.501), both models. `FROZEN.md` carries the reintroduction trigger |
| 4 | `archive/p4_mstate_features/` | **Not null.** Track 3's dense low-rank AE recovered V-alignment for ALBERT (33 bottleneck directions on V-attractive vs 0 for GPT-2): "sparsity was the confound." `FROZEN.md` |
| 5 | `archive/p5_single_mstate_analysis/` | Complete for 6 models; 6 code-level blockers. Carries the tuned-lens skip-to-output note |
| 5b | `archive/p5b_manifold_steering/` | Built and tested, never run |
| 5c | `archive/p5c_unclustered/` | Docs only, no code. Its attention-flip result is cited in `PREDICTIONS.md` claim (a) |
| 6 | `archive/p6_subspace/` | Partial run, ALBERT only. The LDA-alignment inversion (0.887 imaginary vs 0.067 real repulsive) is unresolved, two live explanations |

Phase 2c was described in earlier versions of this index and has never existed on disk.

## Referenced, not present

These files are cited by name in live code and docs and **do not exist in this repository**.
Recorded rather than quietly tolerated: the project's own standing rule 4 is "refuse rather
than degrade," and a documentation reference that silently resolves to nothing is the
documentation instance of that bug class. Nothing here has been invented to fill the gap.

| Referenced file | Cited by | Notes |
|---|---|---|
| `MATH.md` | 12 live files — `PREDICTIONS.md`, `UPDATE_PLAN.md`, `core/metrics.py`, `p1_mstate_tracking/{design-1,status-1}.md`, `p1c_frames/*` | The most load-bearing absence. Cited for §3.2's collapse-time table, §8's step-size definition, §9.1/§9.3 — all of which Phase 1c validates against |
| `DESIGN_pythia_frames.md` | 11 live files, all in `core/` — `frames.py`, `rope.py`, `qk_offset_null.py`, `battery_structure.py`, `frame_card.py`, `sink_audit.py`, and others | Cited by item number ("see items 5, 8, 12"), so the numbering is load-bearing too |
| `CHANGES_jlens_adjacent.md` | `p2_eigenspectra/lens_band.py` | |
| A "2026-07-22 addendum" to `PREDICTIONS.md` | `core/qk_offset_null.py:12` | `PREDICTIONS.md` has no such addendum. The one it does carry (P-T1) is undated in the body and describes a different change |
| `POPPER_PLAN.md` §6x | `POPPER_PLAN.md:4030`, `PROJECT.md:1636` | Both forward-reference a section that stops at §6w. `PROJECT.md` calls it "where the [design] is" and `POPPER_PLAN.md` calls it "the proposal for getting [a fresh artifact]"; the audit pass that would otherwise have taken the next letter used §6y instead, so the reference stays open rather than silently resolving to the wrong section |

## In flight on other branches

Not on `main`, and not reflected in the tables above:

- **Phase 1d** — clusterer comparison, `origin/claude/particle-methods-comparison-vpuads`.
- **Cross-phase visualization CLI** — `origin/claude/visualize-mets-results-sl2ya5`.

**These two are the only branches carrying work that exists nowhere else, and a
2026-09-10 cleanup deleted 21 branches around them.** Both are from August and
~93 commits behind `main`, so branch age and commit count will keep suggesting
they are stale; this section is the reason they are not.
`docs/deleted-branches-2026-09-10.md` records what went and how to restore it.

## Phase 7 — the mechinterp/particle bridge

New as of 2026-08-22. The goal is to describe mechinterp phenomena — induction heads,
steering, activation patching, prompt injection, SAEs — as statements about what particles
are doing, **without going through natural language**, and then to test whether recurring
structures (motifs) in the interaction graph are what those names actually pick out.

First study: **induction-head formation**, restated as a two-stage `relay` motif and tracked
across the checkpoint axis. It bears directly on `PREDICTIONS.md` claim (b) — that
collapse-resistance emerges at circuit-formation events.

SAEs are an **object of study, never an instrument**: the standing rule from
`core/DESIGN_dual_reading.md` (no SAE/LRAE features in any measurement path) is unchanged.

See `p7_motifs/design-7.md` for the translation table and the motif alphabet,
`p7_motifs/status-7.md` for state, and `PREDICTIONS.md` for the pre-registered predictions.

## Dates

Recovered from run-directory names / report timestamps where present: Phase 1 — 2026-04-23;
Phase 2 — 2026-04-28; Phase 2b, Phase 3 — 2026-04-29; Phase 4 — 2026-05-04; Phase 5 — not
recorded (after 2026-05-04); Phase 5b — never run; Phase 5c — not recorded; Phase 6 —
recorded only as "2026-04-xx", itself a gap. Every future run carries a real timestamp via
the run-manifest infrastructure, so this should stop recurring.
