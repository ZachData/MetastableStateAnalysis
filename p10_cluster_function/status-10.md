<!-- p10_cluster_function/status-10.md -->
# Phase 10 — STATUS

<!-- phase-card -->
## Card

- **Question:** What are the particles' metastable clusters made of and what do they do: is a cluster a parking lot that the metric makes cheap to leave but where particles stay, or a category with a functional and causal role, and can the drive that forms clusters be used as an instrument?
- **Inputs:** `pythia-410m` only. The Phase 1 sweep (152 dirs, 19 checkpoints × 8 v1 prompts, battery `1e47918ef77a`), with HDBSCAN labels backfilled from its activations, and the pilot sweep on `HDD_1TB` (243 dirs, 27 checkpoints, native labels) as the second measurement. Seven records at 2 000 permutations: `data/analysis/p10_*.json`. Stage 0 (option B): 20 v2 prompts × 19 checkpoints = 380 runs, battery `06790b90dcfe`, pin `64a4087`, selected only through `data/phase12/stage0_logs/stage0_index.json`; progress lives in `p10_cluster_function/handoff-10.md` §0.3. The 12 new prompts are held out (Registry)
- **Results:**
  - The trained-model attention flip toward unclustered tokens is all causal mask at initialisation on 410m (corrected gap 0.004 against a raw 0.26). Its trained part, and the "~94 % mask" share, are two tokens per prompt (R3, row below) — `p10_cluster_function/status-10.md` §1.1, `p10_cluster_function/status-10.md` §1.17
  - F0: the earliest member of a density cluster sits late, not early, against the prediction, and the within-cluster restricted null leaves the effect in place. It measured a proxy for the paper's strong Rényi centre, so it is not evidence on the parking account — `p10_cluster_function/status-10.md` §1.2, `p10_cluster_function/lit-10.md` §11.4
  - F1: the identity coupling is the optimal transport plan at 3 630 of 3 648 layer boundaries, so every per-layer displacement on record is the true `W_2`. `swap_fraction` counts repeated tokens, not motion. Clustered particles move less than noise only in a window, steps 32–512 — `p10_cluster_function/status-10.md` §1.3
  - F12: raw `log Z` is almost all position. The sink is the minimum of raw `Z` and the maximum of corrected `Z` at every β tried, so the β-unit convention does not touch it. The raw clustered-minus-noise sign is already present at step 0, because HDBSCAN clusters by density — `p10_cluster_function/status-10.md` §1.4
  - F1 and F12 together read "parked, not pinned" at steps 32–64, a window the trained model passes through. This rests on reading `Z` as a metric, and the density confound is argued from step 0, not controlled — `p10_cluster_function/status-10.md` §1.5
  - F1 and F12 on the pilot's partition: per-step means within 0.005 and 0.008 at the shared steps, and no changed unit flips sign at step 32, the least stable step (50/50, 159/159). Below-baseline `Z` lasts past 512 on both sweeps (to 4000 on WDS, non-monotonically to 11 000 on the pilot), which F12's baseline table had left out. Same activations clustered twice — `p10_cluster_function/status-10.md` §1.13
  - The 410m sweep had no density partition in 152 of 152 dirs, and `pair_agreement`, the only semantic instrument, wrote well-formed zero records. The backfill re-derives the labels, bit-checked against the pilot, and does not rerun the analysis — `p10_cluster_function/status-10.md` §2, `p10_cluster_function/handoff-10.md` §1.2
  - The stored HDBSCAN partition's run-to-run floor (ARI 5th percentile 0.347, almost all `repeated_tokens`) is float32 cosine distances: refit on float64 from the same activations, the two sweeps agree at ARI p5 1.000 over all 2 600 layer-records (99.1 % identical), and Stage 0's stored `repeated_tokens` labels are rounding (ARI to float64, mean 0.23); the other 7 prompts' stored labels hold (mean ≥ 0.998). Stored labels are still float32-derived, so every reader pooling `repeated_tokens` includes one noise prompt in eight (`p1d_cluster_ensemble/status-1d.md` "Float64 distances, and Phase 10 §3's floor re-run on them"). A0 and F0 hold on the second sweep; per-layer claims stay exposed — `p10_cluster_function/status-10.md` §3, `p10_cluster_function/status-10.md` §3.1, `p10_cluster_function/status-10.md` §1.11
  - Stage 1 steps 1 and 2 on the pilot agree with Stage 0 to ≤ 0.004 at all 13 shared steps; the handoff's first-look table averaged 9 prompts, not 8. No null — `p10_cluster_function/status-10.md` §1.11
  - The per-layer co-membership and lexical-carry claims hold on the pilot's partition: 3 103 of 3 120 and 383 of 384 per-layer readings agree, all but one disagreement a flip at the ±0.05 floor, and every quoted cell at a shared step (the trained model: 143000 only) is within 0.005. Most cells are bit-identical: 100 of 2 600 run-layers have a different partition. Same activations clustered twice, so this bounds HDBSCAN's instability only. No null — `p10_cluster_function/status-10.md` §1.12
  - 410m's step 0 and step 1 are the same weights, so the checkpoint axis has 18 distinct points — §3.51.3
  - `pair_agreement`'s "ext_semantic" count is mostly a repeat count on Pythia (repeats have cosine 1 at layer 0). Over training the repeat share of mutual-NN pairs falls, and the non-repeat pairs that replace them become similar in the trained embedding. No null — `p10_cluster_function/status-10.md` §1.6
  - Which tokens are clustered is mostly copy count at init and in shallow layers (`min_cluster_size=2`, copies coincide at layer 0), and a moderate effect in the trained model's deep layers. Among unique tokens BPE rank does not predict it; class does, weakly and against trash collection. No null — `p10_cluster_function/status-10.md` §1.7
  - At step 0 layer 0 the partition is what HDBSCAN makes of Gaussian noise with planted duplicates (twins noise at 0.14, singletons clustered at 0.43). At layer 0 the cluster count tracks the prompt's repeated token types at every checkpoint (7-prompt mean ratio 0.92–1.01; per run 0.78–1.24), and ≥ 63 % of clusters at any layer hold a repeat. Phase 1's `max_alive` falls at layer 0 in 92 of 133 runs, mean 57–63 at every step against 59.7 repeated types: on these runs the carrying capacity is mostly a repeat count. `repeated_tokens` is a different mechanism (~50 deep-layer clusters). No null — `p10_cluster_function/status-10.md` §1.8
  - Clustered unique tokens, against step 0: at init, clusters at depth already follow each token's own (random) embedding carried in the residual. Training adds their own class (Δ +0.20 over a random draw, by step 512) and moves them away from copy groups. Own-embedding similarity adds only +0.12, and adjacency a depth-only share. No null — `p10_cluster_function/status-10.md` §1.9
  - At step 512 the network computes that class effect (+0.20 beyond own-embedding similarity; a purely lexical cluster scores ≈ 0). Trained, it is mostly what the embedding already groups: L12 sits on the lexical control, L24 is +0.05 above it. The control was added post hoc. Clustered tokens keep less of their individual layer-0 direction. Context vs per-token feature is not separated. No null — `p10_cluster_function/status-10.md` §1.10
  - Lemma C.1's saturation is a formula for Phase 1's 50–55 carrying capacity — `p10_cluster_function/math-10.md` §5.4, §3.53
  - The re-read's label source (R0, an instrument, nothing re-read yet): 1d's working definition and its ladder (c0 → c3, two arms) at all 18 checkpoints on the 7 v1 passages, every P = 0 pass ≤ 2.9e-7 from Stage 0, nothing refused, reproducing the definition's 62 / 810 records exactly, member set by member set. A float64 refit of the stored partition (c0f) is identical to it in 2,924 of 3,024 records. Steps 0–32 have too few readable c3 records and read on c2, so the c2 / c3 switch falls inside F1's 32–512 window — `p10_cluster_function/status-10.md` §1.14, `p10_cluster_function/design-10.md`
  - R1, the token-composition (unique tokens), co-membership and lexical-carry rows re-read on the definition, c0 reproducing the published records exactly. No row holds at every cell; the quoted class claims keep their labels where they were quoted (rank does nothing trained; class co-membership +0.21; step 512's class effect +0.15 beyond a matched lexical cluster, ~0 trained at L12), and class against trash collection comes earlier, from step 512 rather than 16000, which flips its sign at 512. c0's 64–512 dip in the embedding lifts is the old partition's. Adjacency's Δ flips because stable step-0 groups are runs of neighbours (the c2 baseline) while c3's raw trained lift is larger than c0's. On the rules' own cells 16 / 18, 170 / 255 and 16 / 18 labels agree with c0 — `p10_cluster_function/status-10.md` §1.15
  - R2, F1 and F12's gap re-read on the definition, c0 reproducing the published records at every unit. F1 does not hold by its per-step rule (9 of 18 steps): the strong 64–512 difference is there on c3, and the late tail keeps its size but drops to none on c3's smaller member sets; c0 under the rule stays negative to 54000, so the published window was a magnitude reading. F12's gap reads "below" at every step on c3 because its baseline, c2's step 0, is the densest population on the axis (+1.32); c3's raw sign keeps c0's shape (below 32–256, above from 4000) and its own step 0 shows no density confound (−0.15, 15 records). Every changed label first changes at the token rules or the centred frame. The parked window (F1 negative with F12 below) is 64–512 on the primary. F12's label is replaced by R2m (next line) — `p10_cluster_function/status-10.md` §1.16
  - R2m, F12 against a matched control (each step's member set scored on step 0's activations at the same positions). On the definition (c3), members are less dense than the same tokens were at init at 32–256 (every prompt at 32–128), as dense at 512–1000, and denser from 2000 (+0.12 to +0.27, 6–7 of 7 prompts, sign p 0.016–0.125; mostly the learned groups at 143000); c2, the primary at 8–32, reads below from 8. c3's tokens were not dense at init (control −0.08 to +0.13 from 256 on), so R2's "below at every step" was c2's step-0 baseline. On the old partition the control is +0.09 to +0.35, part of the old step-0 density confound; its crossover lies between 1000–2000 and 8000–16000 (the control and the step-0 baseline bracket the selection effect). The parked window is 64–256 on the primary; c3 agrees with c0 at 16 of 17 steps (the primary at 14, two of them c2's). Selection, context and norm not controlled — `p10_cluster_function/status-10.md` §1.18
  - R3, A0 re-read under T4 (the sink and massive-token columns dropped, rows renormalised), c0 reproducing the published record at every unit. A0 does not hold. Its raw flip is position 0 and one massive token per prompt: out of the means on the old labels, the sweep's raw gap falls from +0.49 to +0.06 and reverses from step 16000, and the late residual goes with them; T4's renormalisation adds little. Under T4 the sweep-pooled raw gap is ≤ 0 in every column, and per step the mask explains under 0.4 of any flip from 1000 on. On the definition, members against the rest at the same positions leave a residual window, +0.05 to +0.09 at 2000–16000, gone from 32000 (the pooled late gap is position); never significant per unit — `p10_cluster_function/status-10.md` §1.17
  - R5c, 5c's flip on `gpt2-large` with the sink out (stored runs, 21 prompts, trained and random, 5c's partition). It does not survive. With position 0 in, random weights give the same raw gap as trained (+1.52 against +1.59), all causal mask, so the flip was never trained-specific on this random arm. Out of the means, trained's raw gap is −0.03 (7 of 21 prompts positive). The trained-only gap beyond the mask (+0.23, 16 of 21) is position 0. Position 0 is the only T1–T2 token and is unclustered in every trained unit. ALBERT not read — `p10_cluster_function/status-10.md` §1.19
  - R6, the cross-checkpoint matcher (an instrument: adjacent steps, same prompt and layer, containment ≥ 0.5, MONIC's kinds, on R0's labels). From 4000 to 54000, 56–67 % of the definition's groups survive each boundary, with changed members (median Jaccard 0.75–0.83). Few of its births and deaths are new groups: most are groups that persist in c2a and fail or pass a filter, or that c2a splits or merges. No 143000 group's unbroken lineage reaches back past 256, and 89 % start after 2000, though chains last about 3× longer than independent breaks would give. The old partition's 17 % of 143000 clusters traced back to step 0 lie off the definition's positions: the same call on the kept positions has none. Its null (permuted labels) only rules out chance overlap — `p10_cluster_function/status-10.md` §1.20
  - R6w, whether the co-membership and lexical-carry drift happens within lineages (each step's change split exactly into the persisting groups' change and entry / exit terms, on R6's links). On the definition, from 512 to 143000, replacement does not carry the drifts measured in the embedding: entering and leaving groups sit where the persisting ones do, and the persisting groups move, mostly by 2000 (class beyond a lexical cluster −0.12, own-embedding similarity +0.18). A shift of the layer-0 frame under every group would read the same way, and where the drift is, groups with identical members move at least as much as changed ones (few links). Same on the unfiltered groups and the old partition, and on both record sets. Layer mean only; descriptive, no null — `p10_cluster_function/status-10.md` §1.21
  - R6f, R6w's within split into members against frame (every step's groups re-scored in one fixed layer 0, step 512's and step 143000's). For own-embedding and within-class embedding similarity it is the frame, pooled and in 5–7 of 7 prompts: the token embedding moves under the definition's groups, mostly 512 → 2000, so there "trained, mostly what the embedding groups" reads as the embedding coming to group what the groups held. For class beyond a lexical cluster the prompts split, so whether the groups became lexical stays open (the pooled members part ≤ 0.03 is partly prompts cancelling). Same on the unfiltered groups; on the old partition about a third of the class-beyond-lexical drift is membership. Layer mean only; descriptive, no null — `p10_cluster_function/status-10.md` §1.22
  - R7, the context-shuffle test (each group's tokens re-clustered with their passage block-shuffled at four grains, and each token alone after the sink). From 512 on, the label depends on the bar: on each group's own floor it is mixed (39 % at 512 and 48–61 % after survive with their token alone, a third break under a token shuffle); on a fixed bar, and among the groups whose floor clears chance, breaking under a shuffle is the largest share (about half to two thirds). The own floor is below chance for a third or more of the groups, and those come out token-alone. Deeper groups survive alone less. Before 512 no shuffle moves any group: the model starts using order between 256 and 1000. The token-alone input moves states even at init, and a shuffle also breaks some token-alone groups, so neither label is a pure reading of context. At 512, groups that break under a shuffle are as same-class as token-alone ones, so the class grouping there is not only a per-token feature. Descriptive, no null — `p10_cluster_function/status-10.md` §1.23
  - R8, the own-floor check on the definition's "moves" filter (a chance level per group from 2000 random same-size sets against the passage's own partition, a proxy for the moved passages'). Requiring every moved passage's match to beat chance as well as the group's own floor keeps 87–91 % of the definition's records at steps 64–1000 and 93–97 % after: at the placed 90 % bar and not resolved by it (over 7 prompts the intervals straddle it; one passage carries 64–256). What holds: at init "moves" is mostly chance (step 0's 62 records halve), and the drop is about twice as large at 64–1000. Most groups whose floor is below chance still pass, intact. The stricter set is a candidate; the choice is the user's — `p10_cluster_function/status-10.md` §1.24
  - R8x, the exact check: the same test with each moved passage's own partition in place of the proxy. The proxy was close (exact in the median, a few records per step classified differently), and the stricter set keeps 0.88 and 0.89 of the definition's records at steps 128 and 1000, below the placed 90 % bar (93–98 % from 2000), so by the rule the drop holds and the stricter set is recommended as the definition; 7 prompts still cannot resolve the bar or the drop's size, and pooled over 64–1000 the stricter set keeps 0.905 — `p10_cluster_function/status-10.md` §1.25
- **Superseded / wrong:**
  - Row A0's "~94 % causal mask, with a learned residual from ~2000–4000 that persists": from step 4000 both the flip and the residual are position 0 and one massive token per prompt in the unclustered population — `p10_cluster_function/status-10.md` §1.17
  - F0's reading as a test of the parking account: a density cluster's earliest member is not the paper's strong Rényi centre; F13 is the real row — `p10_cluster_function/lit-10.md` §11.4, `p10_cluster_function/status-10.md` §5.1
  - The Rényi packing law "as a function of `n`": the law is in β and dimension, not `n` — `p10_cluster_function/lit-10.md` §5, `docs/AXES.md` §7
  - §3.53's "no gradient-flow structure": Lemma 5.3 makes the causal dynamics a sequential gradient flow — §3.52.5
  - `handoff-10.md` §0.4's "Stages 1–5 re-run on all 20 prompts for free", which contradicted registering on prompts chosen blind: the 12 are held out on 410m — `p10_cluster_function/handoff-10.md` §0.4, `docs/PHASE_REVIEW.md` "Decisions"
  - §5's order: replaced by §5.1 after the papers were read (F13 first) — `p10_cluster_function/status-10.md` §5.1
  - `handoff-10.md` §1.1's reading that neighbourhoods go "lexical → contextual": on Stage 0's v1 runs they go from repeats to repeats plus embedding-similar tokens — `p10_cluster_function/status-10.md` §1.6
- **Registry:** none, because the phase is pre-design and deliberately unregistered (`claims/EXPERIMENTS.md`). F14 is named as the one to register (`handoff-10.md` "Standing constraints"). The 12 new v2 prompts are held out on 410m as a confirmation set, only partly blind (`CLAIM-C` ran them on 1.4b and gpt2-large); whether F14 is scored on them, on all 20, and in what order is undecided and the user's (`docs/PHASE_REVIEW.md` "Open" 1–4)
- **Depends on:** 1@30c0d54cc5, 5c@574eba0e4a, 7d@19b7d835b7, 7e@0c1071db50, 8@8cc3fb223c, 9@e70efd632b
- **Feeds:** 9, 1e
- **Open threads:**
  - F5, the four-signature concordance, is the phase's central test. It needs the J-lens (F2 → F3 → F4), and F2 needs HF access — `p10_cluster_function/status-10.md` §4
  - F13, the strong-Rényi centre scan, which F0 stood in for; it is free and needs no partition. F14 needs it, and so do F15 and F20 — `p10_cluster_function/status-10.md` §5.1
  - `CLAIM-C`'s two HDBSCAN metrics have never been compared against the reproducibility floor. This is the only open item here that bears on a registered prediction — `p10_cluster_function/status-10.md` §5
  - No norm-matched random twin per checkpoint. F12's density confound is now controlled for tokens and positions (R2m), not for norm or the init's context — `p10_cluster_function/status-10.md` §5, `p10_cluster_function/status-10.md` §1.18
  - Is the class grouping computed early in training context, or a per-token feature from an early layer? R7 ran the context-shuffle test on the definition: at 512 not only per-token; the token-alone arm's position (every token at position 1) and a word-level shuffle are not controlled. The pilot sweep is unread — `p10_cluster_function/status-10.md` §1.23, `p10_cluster_function/handoff-10.md` "Parked"
  - Which set the re-read rows read after the exact check: c3, or c3x (its "moves" must also beat each moved passage's chance level), which keeps 88–98 % of c3 from step 64 and half its step-0 floor; by the rule the drop holds and c3x is recommended, the user's call. Beside it, whether a well of the theory's own density at the measured β is a better cluster than a level-set group (a probe only, `tools/run/p10_phi_wells_probe.py`, no stored result) — `p10_cluster_function/status-10.md` §1.24, `p10_cluster_function/handoff-10.md` "Parked"
  - Does the pilot's 27-checkpoint "50–55" `max_alive` also fall at layer 0? And what sets `repeated_tokens`' ~50 deep-layer clusters? (Its stored partitions are mostly float32 rounding at steps 32 / 512; float64 gives 24–30 clusters at L6–18, so re-ask after the fix) — `p10_cluster_function/status-10.md` §1.8, `p1d_cluster_ensemble/status-1d.md` "Matched k on `repeated_tokens`, and the float32 defect"
  - Is 410m spent on the induction axis for any Phase 10 registration that joins 7d's causal sweep? — `docs/PHASE_REVIEW.md` "Parked"
  - Position is a confound in every row here, with three separate corrections and no shared one in `core/` — `p10_cluster_function/handoff-10.md` "Standing constraints on all of it"
  - About two in five of the HDBSCAN groups every row reads exist only through hdbscan's tie order (never a component of the mutual-reachability graph; mostly pairs); 1d's level-set HDBSCAN is the tie-free version, and no row here is re-read on it — `p1d_cluster_ensemble/status-1d.md` "Admission"
  - The `p10_*` readers' default glob now mixes the Phase 1 sweep with Stage 0's v1 dirs. The holdout guard (`core/holdout.py`) removes only the 12. Stage 1's reader selects through Stage 0's index; the older readers still glob — `p10_cluster_function/handoff-10.md` "Parked", `p10_cluster_function/status-10.md` §0
- **After Phase 10:**
  - Rebuild a cluster ensemble (1d's intent): drift measured 2026-09-25. Tuning HDBSCAN makes it move less often, not less far. The consensus is stable, but its stable families pick coarse k or group identical strings, so the ensemble waits on 1d's scale design (free: stored activations). Matched k done: the drift is float32 distances, upstream of every method, and k-means at fine k is seed-dependent — `p1d_cluster_ensemble/status-1d.md` "Float-noise drift", "Matched k on `repeated_tokens`, and the float32 defect"
  - Everything in Stages 1–5 again on all 20 prompts once registrations are frozen (free: Stage 0's dirs)
  - F20, the frozen-centre intervention, as the phase's known-answer dry run (forward pass: 410m or 70m, needs F13)
- **Reviewed:** 2026-10-08 · body `b1f07a07e9`
<!-- /phase-card -->

**Registered predictions:** none, and none yet can be. `claims/registry.json` is
untouched and `claims/EXPERIMENTS.md` lists this phase as *pre-design and
deliberately unregistered*. **Nothing below may be quoted as an adjudication.**

**Last verified:** 2026-09-20.
**Overall:** **four rows of `notes-10.md` §8's ladder have RUN on real
checkpoints** — F0, F1, F11-A0 and F12 — plus a producer that unblocked them
and a measurement that was not on the ladder and matters more than two of the
rows. All tier 1, exploratory. Two headline rows were **replicated on a second
sweep with an independently-derived partition** and both hold.

Read `notes-10.md` for what the phase is for, `math-10.md` for the derivations
these rows test, `attention-10.md` for row A0's audit, **`handoff-10.md` for the
cluster-function thread's ordered plan and `questions-10.md` for its
hypotheses**, and `PROJECT.md` §3.51 for the cross-cutting findings (three of which are about the project's
instruments rather than about clusters).

> **2026-09-20, a scoped thread opened out of this one: `handoff-10.md`, and
> its first action is COMPUTE — Stage 0, taking the Phase-1 battery from 8
> prompts to 20.** Eight prompts is the power ceiling under which every e-value
> in this phase sits; the 12 unused v2 prompts were already chosen blind under a
> committed rule, so running them retires no selection risk that has not already
> been retired.**
> The cluster-function question — what clusters are made of, what they do, and
> whether the drive that forms them is usable as an instrument — has its own
> ordered plan in **`handoff-10.md`** and its hypotheses in
> **`questions-10.md`**. It starts at Stage 0, *what is actually in a cluster*,
> and **that stage found a fourth casualty of the HDBSCAN outage §2 records**:
> `pair_agreement` — the project's only semantic instrument — wrote a
> well-formed record of zeros into all 152 WDS directories rather than failing.
> The pilot sweep's copy is populated in 6 066 of 6 075 layer-records and had
> never been reported. First read in `handoff-10.md` §1.1.

> **2026-09-20, later the same day: five papers were READ as primary text**
> (`lit-10.md` §11–§15) — `2411.04990`, `2501.10573`, `2605.12765`,
> `2505.16831`, `2601.02932`. **No number below changes. One interpretation
> does, and it is F0's.** The paper's cluster nucleus is a token separated by
> more than `δ` from *every preceding token* — a geometric acceptance rule on
> positions and distances, with no partition in it — not the earliest member of
> a density cluster. **F0 therefore measured a proxy, and its failure is not
> evidence against the parking account** (`lit-10.md` §11.4). The real row is
> **F13** and it is free. §4 and §5 below are updated accordingly.

**Decision taken 2026-09-20 (user):** F0 runs **exploratory first**, which caps
it at tier 1 — the answer was seen before any wording was frozen. If it is ever
to be the project's first adjudication it needs a fresh axis: 70m, or the
battery prompts that have never been through Phase 1.

**Corrected 2026-09-20: it is 12, not 13** — `core/prompts.py`'s v2 battery is
21 prompts, Phase 1's sweep used 8, and the thirteenth (`short_heterogeneous`,
115 characters) is almost certainly too short to cluster. **Running those 12 is
now the first action of the cluster-function thread** — `handoff-10.md` Stage 0,
which carries the measured storage budget and the four checks. Because they were
chosen blind under a rule committed ahead of the text (`PROJECT.md` §3.42),
running them is not a new selection decision, and **the enlarged battery can
carry a registered prediction where the current one cannot.**

---

## 0. The instruments, and where they live

| what | where | tier |
|---|---|---|
| the holdout guard | `core/holdout.py` — `refuse_held_out`, `add_holdout_args`; every `tools/run/p10_*.py` calls it, and `tests/test_holdout.py` fails any that does not | pure |
| statistics + content-free baselines | `core/parking.py` | pure |
| the arbitrary-dependence e-merger | `core/evalues.py` — `average`, `average_p`, `max_attainable_average_E` | pure |
| tie-tolerant p, restricted null | `core/nulls.py` — `p_from_null_tolerant`, `label_permutation_null_within` | pure |
| the HDBSCAN backfill | `tools/run/backfill_hdbscan.py` | deps |
| row A0 | `tools/run/p10_attention_baseline.py` | — |
| F0 | `tools/run/p10_anchor.py` | — |
| Stage 1 steps 1, 2, co-membership | `tools/run/p10_ext_sem_threshold.py`, `tools/run/p10_token_composition.py`, `tools/run/p10_comembership.py` (read only `stage0_index.json`) | — |
| F1 | `tools/run/transport.py` | — |
| F12 | `tools/run/p10_partition_function.py` | — |
| the reproducibility floor | `tools/run/p10_partition_stability.py` | — |

**Records** under `data/analysis/`, every one at **2 000 permutations** with
`max_attainable_E` **22.37** and `design_can_reject` on its face:
`p10_row_a0.json` (3 646 units), `p10_row_a0_pilot.json` (5 593),
`p10_f0_anchor.json` (3 800), `p10_f0_anchor_pilot.json` (5 835),
`p10_f1_transport.json` (3 648), `p10_f12_z.json` (11 400), and
`p10_partition_stability.json` (2 600 — a floor, **no p-value by design**).

**How to re-run anything here.** From the worktree. Every `tools/run/*.py`
derives `sys.path` from `METS_REPO`, which since 2026-09-24 defaults to the
runner's own checkout (it used to default to the main tree; `docs/PHASE_REVIEW.md`
Parked 2). `METS_DATA` still has to point at the main tree's `data/`:

```bash
METS_REPO=$PWD METS_DATA=/run/media/system/WDS_500/Mets/data \
  /run/media/system/WDS_500/miniforge3/envs/mets/bin/python tools/run/<runner>.py
```

**The conda `mets` env, not `.venv`,** for anything touching HDBSCAN — §2.

**The holdout guard (2026-09-24).** Readers (the four `p10_*` and F1's
`transport.py`) refuse held-out inputs by default. `--v1-only` drops per-prompt ones
and `--allow-holdout` reads them; both are recorded under `"holdout"` in the output
record. It holds out: a run dir for one of the 12 keys (any model; by
`manifest.json` `prompt_key`, else by name), any file named for one, and three
**pooled** kinds that `--v1-only` refuses rather than drops, because they hold v1
values too: a file beside such a run dir (`pair_agreement.json`),
`claim_c_real_run.json`, and `stage0_logs/*.out|*.log`. P-I1's `behavioural.py` refuses
the 12 outright. The other runners over `data/phase12` are exempt, each for a reason
listed in `tests/test_holdout.py`. Dry run on `data/phase12` at 12:40, chunk 2 mid-step512,
reading names and manifest `prompt_key` only: 632 `pythia-410m-*` dirs, 193 held
out, 439 kept. Of the 8 `2026-09-19_*` `CLAIM-C` dirs, the four v2 ones (21 run dirs
each, 48 held out in total) are flagged, and so is each one's `pair_agreement.json`; the four
v1-only ones (9 each) are not. **The records above predate the guard and Stage 0.**
Re-running a reader on the default `--pattern pythia-410m-*` now also globs Stage 0's
v1 dirs beside the Phase 1 sweep (handoff Parked).

---

## 1. What is answered

### 1.1 Row A0 — **the attention flip is ~94 % causal mask**

*Re-read 2026-10-05 (§1.17): the step-0 reading stands; from step 4000 the flip and the residual
below are position 0 and one massive token per prompt in the unclustered population. Out of the
means, the raw gap reverses from 16000 and the residual goes. On `gpt2-large` (§1.19, R5c) the
flip does not survive the sink either, and its random arm already has it in full.*

152 directories, **3 646 layer-units**. `math-10.md` §1's content-free baseline
divided out per layer, with the partition/attention pairing matching
`noise_importance_proxy` exactly (a test plants a signature in one attention
index and checks it lands on the right partition layer).

| | unclustered | clustered | gap |
|---|---|---|---|
| raw, as this project reports it | 1.417× | 0.825× | 0.592 |
| causal-mask-corrected | 1.034× | 0.998× | **0.036** |

**6.2 % of the gap survives**, and the sweep mean hides the finding:

| step | raw gap | corrected gap | position bias |
|---|---|---|---|
| 0–16 | 0.26–0.39 | **0.003–0.004** | 0.061–0.088 |
| 32–512 | 0.10–0.36 | −0.006 to −0.164 | 0.038–0.083 |
| 4000 | 0.949 | **0.245** | 0.035 |
| 143000 | 1.689 | **0.172** | 0.013 |

**Entirely mask at initialisation** — corrected gap 0.004 against a raw 0.26,
exactly as derived. A residual appears at **step ~2000–4000** and persists,
while the position-bias confound *falls* with training. The learned effect is
real in the means and about a sixth the size the uncorrected number implies.

**Verified against an independent reimplementation** written from the closed
form on one layer of `step143000_wiki_paragraph`: raw 1.6685 / 0.4490,
corrected 1.0925 / 0.9238 — exact to four decimals against the runner.

**What may NOT be said.** The 5c number was measured on `gpt2-large` and
ALBERT; this re-measures the same statistic on the 410m sweep rather than
refuting it. And **no arm here is a norm-matched random twin** — step 0 is an
untrained checkpoint, the right baseline for a developmental read but a weaker
control than the `-random` arms, because it shares the initialisation scheme
rather than being matched to a trained model's norms.

### 1.2 F0, the anchor test — **nuclei are LATE, and it is not the confound**

`2411.04990`'s mechanism says early tokens are the nuclei of cluster formation.
Direction fixed in code, before the sweep was read: **"less"**.

- observed **0.3164** against an ordinary-permutation null mean of **0.2077**
- median p **1.00**, fraction below 0.05 **0.18 %**, merged E **0.536**
- the design *could* have rejected at 2 000 draws and did not

**The first version was confounded, and the fix is the result's main
methodological content.** The ordinary permutation is free to move whole
clusters between the early and late halves of a sequence. The sweep has
clustered tokens sitting later than unclustered ones (`position_bias` +0.051),
so a late clustered population produces late cluster minima with nothing said
about nucleation. `label_permutation_null_within` holds the clustered/noise
split **fixed** and shuffles only which clustered token carries which id.

**Both nulls agree.** The restricted null's mean moves only 0.2077 → **0.2311**
against an observed 0.3164, median p 0.9995, E 0.556. The confound accounts for
about a fifth of the gap and the effect survives it.

**AMENDMENT 2026-09-20 — what this row does and does not bear on.** Both
numbers stand; both nulls stand; the confound analysis stands and is still this
row's main methodological content. What does not stand is the inference from
here to the parking account. `2411.04990` §5.1 defines a *strong Rényi centre*
as a token `δ`-separated from **every** preceding token, `δ = cβ^{−1/2}`, and
its Lemma 5.1 remark claims only that the accepted indices `s_j` *"are mostly
small"* and that the early ones among them are near-stationary. **A density
cluster's earliest member is a different object**: it need not be separated
from anything, and a strong centre need not be in any cluster. §6's line "F0 is
not an adjudication of the parking account" was written for a weaker reason and
is true for a stronger one.

### 1.3 F1, transport — **the identity coupling IS optimal, and parking is a window**

3 648 layer boundaries.

**`swap_absorbed_fraction` is 0 to machine precision in 3 630 of 3 648** (mean
5.2e-05, largest 7.0e-02). Displacement across a layer is genuinely motion of
the measure, not tokens changing places, so **every per-layer displacement
number this project has recorded — all identity-coupling — IS the true `W_2`**
rather than the upper bound it was known to be. A validation of a whole class
of existing numbers, with no forward pass.

**`swap_fraction` (0.023 → 0.041, rising with training) is NOT motion.** At
layer 0 a Pythia hidden state is the token embedding alone, so repeated tokens
have identical vectors and swapping their assignments is an exact tie in cost.
Quoting it as transport means quoting repeated tokens. A test fixture pins
this, and pins that the tie argument needs a *small perturbation* — under a
random target the optimal matching genuinely beats identity.

**Straightness falls with training**, 0.151 → **0.122**: net displacement is a
sixth of arc length, and the mature model wanders more per unit of progress.

**The kinematic signature of §3.1 exists and is a window:**

| step | clustered − noise step | median p | merged E |
|---|---|---|---|
| 0–16 | +0.02 to +0.03 | 0.15–0.17 | 5.2–6.9 |
| **32–512** | **−0.29 to −0.49** | **0.0005 (floor)** | **12.9–17.6** |
| 16000–32000 | −0.13 to −0.15 | 0.024–0.039 | 8.7–8.8 |
| 143000 | −0.01 | 0.092 | 7.3 |

### 1.4 F12 — **`math-10.md` §2 confirmed, β-independently; the raw sign is a trap**

11 400 units, three betas.

| | β = 1 | β = 2 | β = 4 |
|---|---|---|---|
| raw `log Z` variance explained by `log(i+1)` | **0.995** | 0.979 | 0.893 |
| percentile of position 0 in **raw** `Z` | **0.0014** | 0.0014 | 0.0014 |
| percentile of position 0 in **corrected** `Z` | **0.9986** | 0.9986 | 0.9986 |

The raw partition function is **99.5 % position** at β = 1. The sink is the
**minimum** of raw `Z` and the **maximum** of corrected `Z`, identical to four
decimals across the grid — so `docs/AXES.md` §4's factor-of-8 convention
question **does not touch this result**. `math-1.md` §1A.6's "high-`Z` = sink"
is right about the corrected quantity and inverted about the raw one.

`position_r2_corrected` of 0.32 is the **known** unit-diagonal residual —
sphere-projected states give `Z_i = e^β + i e^{βγ}`, affine in `i` rather than
proportional to `(i+1)` — not unexplained structure. Both forms are test
fixtures.

**The raw clustered-minus-noise sign is mostly definitional.** It is +0.364
standardised (median p at the floor, 79 % of units below 0.05, E 14.60), which
on the metric reading is *pinned*. But HDBSCAN clusters by density and
`Z_i/(i+1)` IS a density, so a clustered token has high `Z` because that is what
put it in a cluster — and **the effect is +0.465 at step 0, under random
weights, before any training**, with 86 % of units below 0.05.

**Against the step-0 baseline:**

| step | clustered − noise | vs baseline |
|---|---|---|
| 0 (random init) | +0.465 | — |
| **32–64** | **−0.12 to −0.14** | **−0.59 to −0.61** |
| 256–512 | +0.12 to +0.22 | −0.25 to −0.35 |
| 1000–4000 *(rows added 2026-09-25, §1.13)* | +0.34 to +0.37 | −0.09 to −0.13 |
| 8000 | +0.469 | +0.004 |
| 16000–54000 | +0.57 to +0.64 | +0.11 to +0.17 |
| 143000 | +0.676 | +0.21 |

### 1.5 F1 + F12 together — **parked, not pinned, and a window**

At **steps 32–64** clustered particles **move less (F1) AND sit below the
untrained baseline in corrected `Z` (F12)**. Low `Z` on §1A.6's reading means
the metric makes them *cheap* to move — and they do not move.

> **That is the PARKED signature, not the pinned one.** It is a phase the model
> passes through rather than a property of the trained model: by step143000 the
> kinematic difference is −0.01 and the metric difference +0.21 over baseline.

`attention-10.md` §5 named this as the distinction displacement alone cannot
make. It took both rows.

**Where the window ends (2026-09-25, §1.13).** The negative raw sign is 32–64
only. Below-baseline `Z` lasts to 4000 on WDS (back at baseline by 8000) and,
not monotonically, to 11 000 on the pilot; F1's strong difference ends at 512.
Which of these bounds "the window" is not decided here.

**Held as a hazard, not a result.** The density confound is argued from the
step-0 value, not proved — the proper control is a norm-matched random twin per
checkpoint, which the sweep does not carry. And the whole reading rests on
§1A.6's **interpretation** of `Z` as a metric, not on a causal measurement.
**`notes-10.md` §3.1's functional and causal columns are untouched and F5 is
still what discriminates.**

### 1.6 Stage 1 step 1, the `ext_sem_threshold` sweep — **mostly repeats; the rest become embedding-similar**

Tier 1, descriptive, no null. Reader `tools/run/p10_ext_sem_threshold.py
--v1-only`, record `data/analysis/p10_s1_ext_sem_threshold.json`. **Input:**
Stage 0 through `stage0_index.json` (pin `64a4087`, battery `06790b90dcfe`),
8 v1 prompts × 17 steps = 136 runs (16000 and 54000 were not yet indexed;
chunk 2 was running), inputs sha256 `61127896b17d`. Every run reproduced its
stored `n_ext_semantic` at 0.5 exactly, per layer. **Re-run at 152/152 on
2026-09-24** (inputs `c558b210c08f`, the same set as §1.7's re-run): all 2036
summary numbers of the 136-run record are unchanged, 236 are new (steps 16000
and 54000), and the verdicts are the same (frozen dead, self mixed). The table
below has no row for either new step, so it stands.

Means over 8 prompts × 25 layers. "Repeat": the pair is two copies of one token.
Non-repeat columns use the **frozen** frame (layer 0 of step 143000), against
the non-repeat pairs of the same prompt. Base rate of cos > 0.2 in that frame: 0.145.

| step | repeat share | `ext_semantic_fraction` @0.5 (self) | non-repeat: cos > 0.2 | non-repeat: mean percentile | same-cluster among repeats |
|---|---|---|---|---|---|
| 0 | 0.875 | 0.875 | 0.010 | 0.37 | 0.700 |
| 64 | 0.914 | 0.914 | 0.079 | 0.44 | 0.704 |
| 512 | 0.839 | 0.839 | 0.311 | 0.69 | 0.743 |
| 2000 | 0.769 | 0.769 | 0.425 | 0.79 | 0.722 |
| 8000 | 0.733 | 0.747 | 0.477 | 0.80 | 0.703 |
| 143000 | 0.696 | 0.735 | 0.483 | 0.78 | 0.697 |

- **The stored statistic is mostly a repeat count.** Pythia adds no position
  embedding before layer 0, so repeats have cosine exactly 1 in every frame.
  In the self frame, no non-repeat mutual pair passes 0.5 before step 4000,
  so through step 2000 `ext_semantic_fraction` @0.5 *equals* the repeat share.
  After that the cut starts to act: at step 143000 the 0.1 and 0.5 cuts
  differ by 0.15 and self vs frozen by up to 0.03 in between.
- **The repeat share falls 0.875 → 0.696**, mostly between steps 64 and 4000,
  after a rise to 0.914 at step 64. The same-cluster rate among repeats is flat.
- **The non-repeat pairs that replace them become embedding-similar.** Take
  the trained model's layer-0 embedding as the reference. At init, the
  non-repeat mutual pairs sit *below* the typical pair (percentile 0.37). From
  step ~1000 they sit near the 80th percentile, and half have cos > 0.2 against
  a 0.145 base rate. The rise is steps 64–2000. **§1.1's "lexical →
  contextual" is backwards on this measure:** neighbourhoods go from "same
  token" to "same token or a token the trained embedding calls similar".
  Whether that similarity is semantic is step 2's question.
- **The pre-stated verdicts.** Self is "mixed" because of the 0.1 cut, below
  the non-repeat range. Frozen is "dead" because of the all-pairs median cut,
  which sits in the continuous part. Both are in the record. Neither answers
  §1.3's scale question, which the split above answers. The all-pairs quantile
  and rank columns land in the cosine-1 pile when a prompt repeats a lot, so
  the non-repeat block replaces them (`LESSONS.md` lesson 6).
- **Caveats.** No null. Layer 0 is 1 of the 25 layers averaged, and there the
  mutual-NN graph comes from the frame's own Gram. The stored-count gate at 0.5
  could not have failed on the cosines, since none sit near 0.5; it checks the
  tags only (the tokens-vs-pairs check is separate). Re-run when Stage 0
  completes (152 v1 runs). The pilot sweep (§1.1's source): §1.11, which
  agrees to ≤ 0.002 at every shared step.

### 1.7 Stage 1 step 2, the token-composition table — **copy count dominates at init and in shallow layers; among unique tokens rank does nothing**

*Re-read on 1d's working definition (unique tokens): §1.15.*

Tier 1, descriptive, no null. Reader `tools/run/p10_token_composition.py
--v1-only`, record `data/analysis/p10_s1_token_composition.json` (full cells:
feature × level × copies, per layer per step, pooled and prompt-balanced).
**Input:** Stage 0 through `stage0_index.json` (pin `64a4087`, battery
`06790b90dcfe`), 149 v1 runs: 8 prompts × 19 steps less 3 at step 54000
(chunk 2 running). Inputs sha256 `564020cf46e1`, tokenizer.json `c24618a1b3e6`,
native labels only. **Re-run at 152/152 on 2026-09-24** (inputs `c558b210c08f`,
with §1.8's columns): the tables below show no step-54000 row, so they are
unchanged, and so are the verdict counts.

"Clustered" = HDBSCAN label ≠ −1. Rates are per prompt per layer, then averaged.
Copies: *unique* (one copy in the prompt), *first* (first of several), *repeat*
(has an earlier copy; §1.3's column). Rank = token id = BPE merge index + 245, a
frequency proxy.

| step | noise rate | unique | 2 copies | 3–5 copies | 6–20 copies | unique, rank < 1k | unique, rank ≥ 20k | unique word_start | unique punct |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.389 | 0.179 | 0.381 | **1.000** | 0.945 | 0.171 | 0.209 | 0.196 | 0.237 |
| 64 | 0.416 | 0.117 | 0.298 | 0.999 | 0.953 | 0.157 | 0.102 | 0.145 | 0.108 |
| 512 | 0.340 | 0.250 | 0.416 | 0.955 | 0.975 | 0.270 | 0.276 | 0.268 | 0.289 |
| 2000 | 0.355 | 0.277 | 0.489 | 0.900 | 0.882 | 0.242 | 0.310 | 0.321 | 0.290 |
| 8000 | 0.392 | 0.258 | 0.472 | 0.861 | 0.820 | 0.243 | 0.262 | 0.288 | 0.233 |
| 143000 | 0.397 | 0.257 | 0.456 | 0.801 | 0.782 | 0.238 | 0.271 | 0.291 | **0.146** |

(2 copies: the first copy; the later one is within 0.03. 3–5 and 6–20: later copies.)

| step | pre-stated, non-repeat: freq / class contrast | post hoc, unique only: freq / class |
|---|---|---|
| 0 | +0.132 / +0.059 | −0.015 / +0.078 |
| 64 | +0.222 / +0.035 | +0.076 / −0.022 |
| 512 | +0.172 / +0.161 | +0.031 / +0.060 |
| 2000 | +0.088 / +0.095 | −0.037 / +0.007 |
| 8000 | +0.113 / +0.133 | +0.015 / −0.027 |
| 143000 | +0.089 / +0.108 | +0.000 / −0.130 |

By layer (pooled over prompts; added after `/challenge-pr` on #90). Later copies
of 3–5-copy tokens / first copy of a 2-copy token / unique; then the unique-only
contrasts, freq / class:

| step | L0 | L12 | L24 | unique contrasts L0 · L12 · L24 |
|---|---|---|---|---|
| 0 | 1.00 / 0.87 / 0.43 | 1.00 / 0.33 / 0.17 | 1.00 / 0.33 / 0.19 | +0.01/−0.20 · −0.03/+0.09 · −0.02/+0.32 |
| 512 | 1.00 / 0.85 / 0.43 | 0.97 / 0.45 / 0.34 | 0.91 / 0.38 / 0.33 | |
| 2000 | 1.00 / 0.74 / 0.38 | 0.92 / 0.44 / 0.30 | 0.84 / 0.48 / 0.41 | |
| 143000 | 1.00 / 0.74 / 0.34 | 0.80 / 0.45 / 0.26 | **0.70 / 0.48 / 0.44** | +0.03/−0.29 · +0.01/−0.16 · −0.05/−0.05 |

- **Clustered vs noise is mostly copy count at init and in shallow layers.**
  HDBSCAN runs at `min_cluster_size=2` (`p1_mstate_tracking/clustering.py`),
  and copies of one token coincide at layer 0. At step 0 every token with 3–5
  copies is clustered at every layer. Training erodes the copy groups in depth:
  at layer 24 of step 143000 it is 0.70 against 0.44 for unique tokens, and a
  2-copy token is only 0.04 above unique. In the deep layers of the trained
  model, copy count is a moderate effect, not the whole story. The whole-prompt
  noise rate barely moves (0.34–0.42).
- **The pre-stated trash-collection criterion reads "consistent" at 11 of 18
  distinct checkpoints (step 0 = step 1, §3.51.3), but twins carry it.** Its
  non-repeat set keeps first copies, which have cosine-1 twins later in the
  prompt, and high-frequency tokens recur more. Among **unique** tokens the
  verdict is "unclear" at all 18.
- **Among unique tokens, rank does not predict clustering.** The frequency
  contrast is within ±0.08 at every step (0.000 at step 143000) and within
  ±0.05 at L0, L12 and L24. Rank < 1k is the *least* clustered unique bin from
  step 2000 on. This is the solid result.
- **Class does predict, weakly and against trash collection.** Unique
  punctuation is less clustered than word starts late in training (0.146 vs
  0.291 at step 143000), and the class contrast is negative at every layer
  shown there. At step 0 its sign flips with depth (−0.20 at L0, +0.32 at L24).
  The verdict reads "unclear" because the two contrasts disagree, not because
  nothing predicts. Thin: 28 unique punctuation and 3 unique whitespace tokens
  across the 8 prompts. Unique numerals climb to 0.55, the highest class (30
  tokens, 4 prompts).
- **What this does to the semantic question.** It narrows it. The clustered
  unique tokens are the residual to explain, but part of it is there at random
  init (the "unique" column at step 0), so the quantity is trained minus step 0,
  not the trained rate. The next step asks what they cluster *with* (their
  co-members' copies, classes, positions), against step 0 as the baseline. That
  is a new reader, not this table. How much of the copy effect is the
  instrument's: §1.8 (at step 0 layer 0 all of it is).
- **Caveats.** No null. The first two tables average layers.
  The HDBSCAN floor (§3, ARI p5 0.347) applies to every rate. Rank is merge
  order, not corpus frequency. `repeated_tokens` has no unique word starts, so
  the unique contrasts rest on 7 prompts. The pilot sweep agrees to ≤ 0.004
  at every shared step (§1.11).

### 1.8 Two checks on §1.7 — **at step 0 layer 0 the partition is what HDBSCAN makes of noise with duplicates; the carrying capacity is mostly a repeat count at layer 0**

Tier 1, descriptive, no null. Both were parked by `/challenge-pr` on #90.

**(a) Known answer: duplicates planted in Gaussian noise.**
`tools/run/p10_hdbscan_planted.py`, record `data/analysis/p10_hdbscan_planted.json`.
**Input:** 250 background points N(0, I₁₀₂₄) plus 40 groups of 2 exact copies,
10 of 3, 6 of 4 and 4 of 5 (≈ 404 points), seeds 0–19, partitioned by the exact
call `clustering.py` makes (cosine, `min_cluster_size=2`). Conda `mets`,
hdbscan 0.8.41. Step 0's embeddings are a random Gaussian init, so this is close
to the real layer-0 input.

| | planted (20 seeds) | real, step 0 layer 0 (§1.7) |
|---|---|---|
| 2-copy groups: noise | 0.141 (113/800) | 0.13 (26/205) |
| 3–5-copy groups: noise | 0.000 (0/400) | 0.00 |
| singletons clustered | **0.43** (2159/5000 background) | **0.43** (unique tokens) |
| clusters / groups of ≥ 2 | 1.09 (65.3 / 60) | 0.99 (below) |

- **The step 0 layer 0 rates are the instrument's.** Twins labelled noise,
  every 3+-copy group clustered, and 43 % of singletons clustered all come out
  of structureless noise at the rates §1.7 measured.
- **HDBSCAN at `min_cluster_size=2` clusters structureless points.** The known
  answer for the background is 0; it gets 0.43. Of 1307 clusters, 902 hold
  exactly one planted group, 88 hold several, and 317 hold background points
  only (207 of those are pairs). A planted group is its own cluster in only 271
  of 1200 cases; 806 absorb 1–16 background points, and 10 are split between
  labels despite identical coordinates (ties).
- **Pure noise gives no stable count.** The second arm (404 points, nothing
  planted) gives 2–55 clusters per seed, mean 14.7: a few large clusters, not
  a floor near 50. Duplicates are what make the count track the groups.
- **So step 0 is the right baseline for the next reader and cannot be skipped:**
  a clustered unique token at init is a noise point glued to a copy group.

**(b) Cluster count vs repeated token types.** New columns in
`tools/run/p10_token_composition.py` (the §1.7 record). **Input:** the 152 v1 runs,
inputs `c558b210c08f`. Per run per layer: HDBSCAN cluster count, token types
with ≥ 2 copies (a property of the prompt, the same at every checkpoint), and
the share of clusters holding ≥ 2 copies of some token. Means over the 7
prompts other than `repeated_tokens` (59.7 repeated types per prompt):

| step | L0: clusters / repeated types · hold a repeat | L12 | L24 |
|---|---|---|---|
| 0 | 0.99 · 0.89 | 0.85 · 0.91 | 0.83 · 0.87 |
| 512 | 0.98 · 0.87 | 0.75 · 0.83 | 0.72 · 0.82 |
| 2000 | 0.96 · 0.87 | 0.84 · 0.76 | 0.61 · 0.77 |
| 16000 | 0.97 · 0.89 | 0.79 · 0.74 | 0.51 · 0.73 |
| 143000 | 0.93 · 0.89 | 0.79 · 0.74 | 0.59 · 0.70 |

- **At layer 0 the cluster count tracks repeated types at every checkpoint**:
  the 7-prompt mean ratio is 0.92–1.01 over all 19 steps. Per run it is
  0.78–1.24, so "tracks", not "equals". At every layer and step, ≥ 63 % of
  clusters hold a repeat (minimum: step 143000, L19).
- **Training lowers the deep-layer count** (L24: 0.83 at step 0, 0.46–0.59 from
  step 16000). It does not raise it: the trained model merges or dissolves copy
  groups rather than forming clusters without copies.
- **`repeated_tokens` is a different mechanism, not evidence against.** Its
  text has 1 repeated token type; HDBSCAN finds 2 clusters at L0 (ratio 2.00 at
  every step) but 52 at L12 and 55 at L24 at step 0 (43 / 42 at step 143000).
  Its `max_alive` is 41–65 and peaks anywhere from L2 to L22, so it is neither
  fixed nor at the embedding. Something else (position is the first suspect)
  makes ~50 clusters there; pure noise does not (arm 2 of (a)).
- **Phase 1's carrying capacity, on these runs, is mostly a repeat count taken
  at the embedding layer** (measured after `/challenge-pr` on #91 pointed out
  the record already held it). `max_alive` (`cluster_tracking.py:264`) is the
  most clusters at any one layer, from these same labels. Over the 7 prompts it
  falls at L0 (the embedding lookup, before any attention) in 92 of 133 runs,
  and its mean is 57.0–63.0 at every one of the 19 steps, against 59.7 repeated
  types per prompt, which training cannot change. So its invariance across
  training is what a repeat count predicts, and Lemma C.1's formula
  (`math-10.md` §5.4) is not needed to explain it here. **Still open:** the
  "50–55" came from `max_alive` on the pilot's 27 checkpoints, and what it
  averaged over is not recorded; on these runs the 8-prompt mean is 56–63.
  Whether the pilot behaves the same is one pass over its labels.
- **Caveats.** No null. The HDBSCAN floor (§3) applies. Means over 7 prompts
  hide their spread. "Holds a repeat" does not mean "is a copy group": at L0
  only 0.33–0.49 of clusters are a single token type.

### 1.9 What clustered unique tokens cluster with — **trained, their own class and away from copy groups; embedding similarity is mostly there at init**

*Re-read on 1d's working definition: §1.15 (class holds; adjacency turns on the baseline).*

Tier 1, descriptive, no null. Reader `tools/run/p10_comembership.py --v1-only`,
record `data/analysis/p10_s1_comembership.json`. **Input:** the 152 v1 runs
through `stage0_index.json` (pin `64a4087`, battery `06790b90dcfe`), inputs
`c558b210c08f`, tokenizer.json `c24618a1b3e6`, native labels only. Record
schema 2, after `/challenge-pr` on #92 (below).

Focal token: a clustered unique token (one copy in the prompt) at position > 0.
For each, properties of its co-members against a uniform random draw of the
same size (exact expectations): from the rest of the prompt (pre-stated), and
from its other clustered positions (`_cl`, added). Lift = observed − expected,
per run-layer, then the mean over prompts. The reading, fixed in the docstring
before the first run: Δ = lift(step) − lift(step 0), "above" or "below" step 0
past ±0.05. `repeated_tokens` has 1 unique token, never clustered, so 7
prompts contribute.

**What the review changed.** The first version's headline property, `emb_pct`,
scores co-members in the *trained* layer-0 embedding (the frozen frame). Step
0's clusters come from an unrelated random embedding, so their lift there is
~0 by construction, and the Δ only tracks the embedding moving toward its
final form. Its Δ is now not read. `emb_pct_own` scores them in each run's
**own** layer 0, which has a valid step-0 baseline. At layer 0 it is circular
(the partition was made from that geometry). The first version's reading ("trained, they cluster with tokens the
trained embedding calls similar, emerging over steps 64–2000") was that
artifact, and it is withdrawn.

Lift at layer 0 / 12 / 24 · mean over 25 layers, all-positions draw. Expected at L12: copy_share 0.65, no_copy ≈ 0.13, adjacent 0.02, same_class ≈ 0.51, emb_pct_own 0.50.

| step | same_class | copy_share | no_copy | adjacent | emb_pct_own |
|---|---|---|---|---|---|
| 0 | +0.04 / +0.07 / +0.02 · +0.06 | −0.05 / −0.02 / +0.00 · −0.03 | +0.15 / +0.14 / +0.11 · +0.13 | +0.02 / +0.10 / +0.10 · +0.08 | +0.39 / +0.22 / +0.15 · +0.20 |
| 64 | +0.04 / +0.19 / +0.22 · +0.17 | −0.06 / −0.03 / −0.13 · −0.05 | +0.16 / +0.10 / +0.21 · +0.13 | +0.00 / +0.02 / +0.02 · +0.04 | +0.39 / +0.11 / +0.05 · +0.14 |
| 512 | +0.07 / +0.33 / +0.33 · +0.33 | −0.07 / −0.19 / −0.20 · −0.17 | +0.18 / +0.28 / +0.21 · +0.24 | +0.02 / +0.04 / +0.09 · +0.06 | +0.39 / +0.10 / +0.11 · +0.15 |
| 2000 | +0.29 / +0.31 / +0.28 · +0.32 | −0.13 / −0.23 / −0.17 · −0.21 | +0.23 / +0.35 / +0.26 · +0.32 | −0.01 / +0.26 / +0.17 · +0.17 | +0.43 / +0.34 / +0.21 · +0.32 |
| 16000 | +0.30 / +0.29 / +0.24 · +0.30 | −0.13 / −0.24 / −0.13 · −0.17 | +0.25 / +0.33 / +0.18 · +0.28 | −0.01 / +0.20 / +0.19 · +0.14 | +0.46 / +0.34 / +0.17 · +0.35 |
| 143000 | +0.29 / +0.26 / +0.23 · +0.27 | −0.13 / −0.19 / −0.08 · −0.15 | +0.25 / +0.31 / +0.12 · +0.26 | −0.01 / +0.20 / +0.19 · +0.13 | +0.47 / +0.33 / +0.17 · +0.32 |

Δ at step 143000, layer mean, all-positions draw / clustered draw: same_class
+0.20 / +0.20, copy_share −0.12 / −0.06, no_copy +0.13 / +0.11, adjacent
+0.04 / +0.04 (L12: +0.10 / +0.10), emb_pct_own +0.12 / +0.11 (L12 +0.10, L24
+0.02). Share of unique tokens clustered (layer mean): 0.21 at step 0, 0.30 at
step 143000. Distinct clusters behind the focal tokens: 16–21 per run-layer at
the layer mean, so no single cluster carries a cell.

- **At init, clusters at depth follow each token's own embedding.**
  Step 0's `emb_pct_own` lift is +0.22 at L12 and +0.15 at L24 (L0 is
  circular). The embedding is random, so this is the residual stream carrying
  the token's own vector, not meaning. Copy share is at chance (−0.03), and
  24 % of focal tokens sit in a cluster with no copies, which planted noise
  also makes (§1.8(a): 317 of 1307 clusters are background only).
- **Training adds class, the clearest effect.** `same_class` Δ is +0.20 at
  the layer mean, and the same under the clustered draw, so it is not
  §1.7's class-predicts-clustered effect leaking into the expectation. Per
  prompt it varies widely (+0.03 to +0.60 at L12, step 143000). It moves
  first: +0.19 at L12 by step 64, full size by step 512.
- **Training moves them away from copy groups.** Copy share falls below
  chance (Δ −0.12; −0.06 under the clustered draw, whose pool is copy-heavy),
  and clusters of unique tokens only rise (Δ +0.13 / +0.11).
- **Embedding similarity adds a little.** `emb_pct_own` rises from +0.20 to
  +0.32 at the layer mean (Δ +0.10 at L12, +0.02 at L24). At L12 Δ is
  positive in 6 of 7 prompts (−0.07 in `hdbscan_code`). It dips at steps
  64–512 (+0.10 at L12) and comes back by step 2000, when the embedding
  settles (§1.6). Class and embedding similarity are not separated.
- **Position is a depth-only part.** Adjacent co-members: Δ +0.10 at L12 and
  +0.09 at L24, none at L0, near the floor at the layer mean. At L12, 22 % of
  focal tokens have a neighbour in their cluster against 2 % by chance, so
  most co-members are not neighbours.
- **Caveats.** No null beyond the random draw; no e-value. The ±0.05 floor is
  placed, not calibrated. At trained L24 one cluster holds most of the prompt
  in some runs (mean k 76 at step 143000; `homer_iliad` has 190 focal tokens
  there), so their lifts are near 0 and the L24 column is weaker than it
  looks. Whether trained depth clusters are still the token's own embedding
  (lexical) or contextual is not answered here (§1.10 answers part of it).
  The HDBSCAN floor (§3) applies. Stage 0's v1 runs; on the pilot sweep, §1.12.

### 1.10 §1.9's class effect, lexical or not — **early in training the network computes it; trained, it is mostly what the embedding already groups**

*Re-read on 1d's working definition: §1.15 (the step-512 and trained readings hold). Within
lineages: §1.21 (replacement does not carry the 512 → 143000 fall; the persisting groups move
with the whole population, consistent with the layer-0 frame moving).*

Tier 1, descriptive, no null. Reader `tools/run/p10_lexical_carry.py --v1-only`,
record `data/analysis/p10_s1_lexical_carry.json` (schema 4, after
`/challenge-pr` on #93 and #94). **Input:** as §1.9: 152 v1 runs through
`stage0_index.json`, pin `64a4087`, battery `06790b90dcfe`, inputs
`c558b210c08f`, tokenizer `c24618a1b3e6`. 4 min on 16 cores. Same focal tokens
and all-positions draw as §1.9. Both checks were parked in `handoff-10.md`.
"Class" is §1.7's orthographic category (word start, continuation, punct,
numeric, whitespace, byte fragment), not meaning. Word start vs continuation
depends partly on the previous token.

- **Carry** (`self_pct`): the percentile of cos(x_L[i], x_0[i]) among
  cos(x_L[i], x_0[j]). This removes the common direction. Raw `self_cos` is
  0.05 at L12, trained.
- **Split**: `class_given_emb` is the same-class lift with each co-member
  redrawn from its own-embedding similarity bin, so class is measured beyond
  similarity. `emb_given_class` is the reverse: similarity with each
  co-member redrawn within its own class, so similarity is measured beyond
  class.

**Post hoc, and it decides the reading.** The pre-stated control used
deciles. It cannot work: a purely lexical cluster of the same size at the same
layer (each focal token's k nearest in its own layer 0, `*_knn`) scores +0.21
at trained L12 under it, more than the observed +0.15. At 40 bins that lexical
control is +0.04 at L12. The observed value is read against it, and beside a
class-only cluster's expected score (`*_classonly`: same-class tokens, the
embedding ignored; a reference, not a maximum: one run exceeds it, and per
prompt it runs 0.04–0.44). The pre-stated 10-bin reading ("above step 0",
Δ +0.13 at the layer mean) is recorded but does not settle it. Trained L0, the
first reference used, agrees with the kNN control at 40 bins (+0.03 / +0.03).

`class_given_emb` at 40 bins: observed / lexical control (kNN) / class-only
reference. Step 0 is observed −0.04 to +0.01, kNN −0.04 to −0.01, class-only
0.28–0.30 at every layer. 10 bins, observed / kNN, in brackets:

| step | L12 | L24 | layer mean |
|---|---|---|---|
| 512 | **+0.20** / −0.00 / 0.32 (+0.28 / +0.04) | **+0.20** / −0.00 / 0.30 (+0.27 / +0.05) | +0.20 / +0.00 / 0.31 |
| 2000 | +0.12 / +0.04 / 0.27 (+0.20 / +0.14) | +0.13 / +0.02 / 0.30 (+0.19 / +0.11) | +0.13 / +0.04 / 0.28 |
| 143000 | **+0.04** / +0.04 / 0.26 (+0.15 / +0.21) | **+0.07** / +0.02 / 0.26 (+0.15 / +0.13) | +0.05 / +0.03 / 0.25 |

Carry, focal (clustered unique) / unclustered unique, `self_pct` · `self_top1`,
over the 7 prompts with both groups (`repeated_tokens` has no focal token):

| step | L12 | L24 |
|---|---|---|
| 0 | 0.97 / 0.97 · 0.43 / 0.44 | 0.90 / 0.91 · 0.20 / 0.20 |
| 512 | 0.82 / 0.82 · 0.08 / 0.12 | 0.66 / 0.65 · 0.02 / 0.02 |
| 143000 | 0.84 / 0.88 · 0.13 / 0.20 | 0.61 / 0.72 · 0.00 / 0.05 |

`emb_given_class` Δ at step 143000: +0.07 L12, −0.02 L24, +0.08 mean. `emb_same`
Δ +0.11 L12, `emb_cross` Δ +0.02.

- **Step 512: the network computes the class grouping.** The embedding
  barely groups by class yet: §1.9's same_class lift at L0 is +0.07, and a
  lexical kNN cluster scores +0.04 / +0.05 at L12 / L24 under deciles, against
  +0.21 / +0.13 trained. Depth clusters are +0.20 same-class beyond embedding
  similarity at 40 bins, about two thirds of the class-only reference (0.31). This check cannot
  say whether the source is context or a per-token feature computed after
  layer 0.
- **Trained: mostly what the embedding already groups.** At 40 bins, L12's
  +0.04 is on the lexical control (+0.04). L24's +0.07 is +0.05 above its
  control (+0.02), on the floor. Taken the pre-stated way, each against its
  step-0 value, the L24 excess is +0.04, inside the floor. The class-only
  reference stays 0.26, so the measure could have shown more. Two to three of 7 prompts carry the L24 excess
  (`latex_monograph` +0.23, `hdbscan_code` +0.16, `camus_letranger` +0.08; the
  rest −0.03 to +0.01).
- **Not individual carry, which is a narrower claim.** Clustered unique tokens
  keep less of their *own* layer-0 direction than unclustered ones at trained
  depth (`self_pct` −0.04 at L12, −0.10 at L24; no gap at steps 0 and 512).
  `self_pct` does not look at class, so it cannot rule out tokens carrying a
  direction their class shares in the embedding. That would still be
  lexical, and the kNN control above is consistent with it.
- **Within class, embedding neighbours at mid-depth only.** `emb_given_class`
  Δ is above step 0 at L12 (+0.07) and as step 0 at L24. The embedding
  preference is among same-class co-members (`emb_same` +0.11, `emb_cross`
  +0.02).
- **Caveats.** Binning always leaves some similarity inside a bin, and each
  co-member's own s shrinks a 40-bin lift by ~10 %. "Beyond the own layer-0
  vector" is not "contextual". Position, attention and per-token computed
  features are not separated. A context-shuffle test would separate them
  (Parked in `handoff-10.md`). The floor, no null and HDBSCAN (§3) apply as in
  §1.9. The first version's carry table averaged 7 prompts against 8 and is
  replaced; its "17 %" was 5 % on matched runs.

### 1.11 Stage 1 steps 1 and 2 on the pilot sweep — **the pilot agrees with Stage 0 to ≤ 0.004 at every shared step; its 14 extra steps fill the gaps smoothly**

Tier 1, descriptive, no null. `handoff-10.md` §1.3 step 3. **Inputs:** the pilot
sweep `HDD_1TB/Mets_archive/2026-08-12_05-01-35` (native labels, battery
`1e47918ef77a`, git `3726289`), read with `--run-root` and `--prompts` set to Stage 0's
8 v1 keys: 8 × 27 steps = 216 runs, inputs sha256 `c183a7fcbd9e`, beside Stage 0's
152 v1 runs (`c558b210c08f`, §1.6–§1.7). Records `data/analysis/p10_s1_ext_sem_threshold_pilot.json`,
`p10_s1_token_composition_pilot.json`. Side by side: `tools/run/p10_s1_compare.py`
(prints every row; it reproduces §1.6 and §1.7's Stage 0 columns exactly; its
`--raw` mode produces every label and activation number in this section). Every
pilot run reproduced its stored `n_ext_semantic` at 0.5.

**The inputs are the same, so only the labels can differ.** On the 104 (step,
prompt) runs both sweeps hold (13 steps × 8 prompts), `tokens.txt` is identical
and activations differ by ≤ 2.5e-7 up to step 1000 and ≤ 7.9e-5 at step 143000.
Stage 0's native labels are **bit-identical to the WDS backfill's** in all 3 800
layer-records of the 152 v1 runs, and so are its activations. So "the WDS sweep"
and Stage 0 are one measurement, and the pilot is the only second one. Pilot vs
Stage 0 labels: 2 165 of 2 600 layer-records identical (104/104 at layer 0,
79–90 at every other layer), ARI p5 0.347, min 0.166, which is §3's floor on the
same pairs. `max_alive` is equal in 92 of 104 runs.

| on the 13 shared steps, max \|pilot − Stage 0\| | value |
|---|---|
| §1.6: repeat share, `ext_semantic_fraction` @0.5, non-repeat cos > 0.2, mean percentile | 0.000 |
| §1.6: same-cluster among repeats | 0.002 |
| §1.7: noise rate, unique, 2 / 3–5 / 6–20 copies, unique by rank and class | ≤ 0.003 |
| §1.7: unique freq / class contrast | 0.001 / 0.004 |
| pre-stated verdicts (frozen / self) | dead / mixed in both |

Pilot-only steps (3000, 5000 … 19000, 40000 … 120000), so the 14 not in Stage 0:

| step | repeat share | nr cos > 0.2 | unique clustered | 3–5 copies | unique punct | unique class contrast |
|---|---|---|---|---|---|---|
| 3000 | 0.751 | 0.451 | 0.269 | 0.884 | 0.229 | −0.045 |
| 9000 | 0.730 | 0.468 | 0.243 | 0.847 | 0.179 | −0.067 |
| 19000 | 0.731 | 0.518 | 0.224 | 0.830 | 0.139 | −0.092 |
| 60000 | 0.718 | 0.510 | 0.240 | 0.809 | 0.147 | −0.100 |
| 120000 | 0.695 | 0.487 | 0.247 | 0.793 | 0.144 | −0.116 |

- **§1.6 and §1.7 hold on the independent sweep.** Every Stage 0 number they
  quote is within 0.004 on the pilot, and the pilot's extra steps sit between
  Stage 0's neighbours: the repeat share keeps falling slowly after step 4000
  (0.74 → 0.70), non-repeat similarity is flattest at steps 17 000–40 000 (0.52–0.53),
  and unique punctuation falls from 0.23 at step 3000 to ~0.14 by step 15 000.
  The class contrast goes negative after step 2000 and stays there.
- **What the agreement tests, and what it does not** (after `/challenge-pr`
  on #95). Four of §1.6's five columns never read a label (they read the
  mutual pairs and the embedding), so they could not have differed. The label
  columns (same-cluster among repeats, all of §1.7) average over layers. The
  partitions differ in 17 % of layer-records, but the differences are small
  (mean |Δ n_clusters| ≤ 0.8 per layer), none are at layer 0, and every column
  is a mean over thousands of tokens. §3.1 found the same for A0 and F0. The
  per-layer claims, which are §1.9–§1.10's, are **not** tested here.
- **§1.1's source table is 9 prompts, not 8.** Run over all 9 pilot prompts
  (`short_heterogeneous` included; inputs `1b32908600b1`,
  `p10_s1_ext_sem_threshold_pilot_all9.json`), the reader reproduces every
  §1.1 value: 0.833 / 0.856 / 0.788 / 0.752 / 0.692 / 0.723 / 0.708, and the
  same-cluster column. `handoff-10.md` §1.1 said 8. The 8-prompt pilot starts at
  0.875, as Stage 0 does.
- **The floor is HDBSCAN moving under float noise** (corrected after
  `/challenge-pr` on #95; the first version said the floor was "between the
  pilot and today", which was wrong). At steps 0–1000 the two sweeps'
  activations differ by ≤ 2.5e-7, about one float32 rounding step, and 24–60
  of each step's 200 layer-records still differ. The toolchain did not change:
  the backfill re-clustered the pilot's own activations with today's code and
  matched its labels (§2). That today's pipeline repeats itself bit for bit
  (Stage 0 vs the WDS backfill) shows only that it is deterministic on this
  machine. A change of thread count, library build or hardware would bring
  the floor back.
- **Caveats.** No null. The pilot's 9th prompt is left out of the tables so the
  step means compare. The co-membership and lexical-carry readers (§1.9–§1.10)
  carry the per-layer claims; §1.12 runs them on the pilot.

### 1.12 §1.9–§1.10 on the pilot sweep — **the per-layer claims hold on the second partition: 3 103 of 3 120 and 383 of 384 readings agree**

Tier 1, descriptive, no null. `handoff-10.md` Parked (from `/challenge-pr` on
#95). **Inputs:** the pilot as in §1.11 (216 runs, inputs `c183a7fcbd9e`,
`--run-root` + `--prompts` = the 8 v1 keys), tokenizer `c24618a1b3e6`,
against Stage 0's §1.9–§1.10 records (`c558b210c08f`). Records
`data/analysis/p10_s1_comembership_pilot.json`, `p10_s1_lexical_carry_pilot.json`
(1 min 52 s and 5 min 53 s on 16 cores). Side by side and agreement:
`tools/run/p10_s1_compare.py --per-layer`. A reading is §1.9's "above / below /
as step 0" at ±0.05 on Δ = lift(step) − lift(step 0), counted over the 12
shared steps > 0. For §1.9 it is re-derived from the lifts at all 25 layers and
the mean, because the record's `reading` holds only L0/L12/L24/mean; §1.10's
record holds those four layers only.

| | readings agree | max \|Δ value\|, pilot vs Stage 0 |
|---|---|---|
| §1.9: 5 properties × 2 draws × 26 layers × 12 steps | 3 103 / 3 120 | ≤ 0.021; copy_share and no_copy up to 0.052 |
| §1.10: 8 quantities × 4 layers × 12 steps | 383 / 384 | ≤ 0.020 |

- **The numbers §1.9 and §1.10 quote are the same on the pilot, at the steps
  both sweeps have.** Shared: 0, 1, 2, 4 … 1000 and 143000. The steps §1.9 and
  §1.10 also quote (2000, 16000) are not in the pilot. At steps 0, 64, 512 and
  143000, every quoted cell is within 0.005: same_class lift
  +0.29 / +0.26 / +0.23 at L0 / 12 / 24, `class_given_emb` at 40 bins +0.20 at
  step 512 against kNN ≈ 0, trained L12 +0.04 on +0.04, L24 +0.07 on +0.01–0.02,
  carry `self_pct` 0.84 / 0.88 at L12 and 0.61 / 0.72 at L24. 11 of the 12
  shared steps > 0 are ≤ 1000, so the trained-model claims get one step
  (143000) on the second partition.
- **Most of the agreement is identical cells.** 100 of 2 600 run-layers (13
  steps × 8 prompts × 25 layers; none in `repeated_tokens`) have a different
  partition; the rest are bit-identical. Where one does change, a single prompt's value moves by up to
  0.36 (`latex_monograph`, step 32, L12, `copy_share`). The per-prompt numbers
  §1.9–§1.10 quote hold within 0.027 at 143000 (`/challenge-pr` on #96).
- **The 18 disagreements are threshold flips.** In 17 both Δs are within
  0.011 of ±0.05. The one real gap is step 32 L12 (no_copy Δ −0.071 vs −0.030;
  copy_share_cl and emb_cross flip in the same cell). Step 32 is the least
  stable step (`/challenge-pr` on #96): 29 run-layers differ there, in 16 of 25
  layers, against 15 at step 16, 13 at step 64 and ≤ 8 at every other shared
  step. It is inside §1.9's 32–512 window and the
  32–64 window §1.3 and §1.5 rest on (parked, `handoff-10.md`).
- **§1.10 at all 26 layers** (`/challenge-pr` on #96, the reviewer's own code
  over the records): 7 disagreements in 2 184 cells, all at the floor. The
  table above counts the 4 layers the record's reading holds.
- **The pilot's extra steps fill in §1.10's decline.** `class_given_emb` at 40
  bins, layer mean: 0.20 at step 512, 0.09 at 3000, 0.055 at 13 000–19 000,
  0.05 from 40 000 on; the kNN control is 0.015–0.03 throughout. L24 is
  0.06–0.10 over steps 3000–120 000 against kNN 0.005–0.04, so its "+0.05 over
  the control, on the floor" reads the same on both sweeps.
- **What this shows and what it does not.** It is the same activations
  clustered twice (§1.11), so it measures how far HDBSCAN's float-noise
  instability moves these claims, and here it moves them little. It is not a
  second model, a second prompt set or a null. The ±0.05 floor stays placed,
  not calibrated.

### 1.13 F1 and F12 on the pilot sweep — **the 32–64 window holds on the second partition**

Tier 1, the same statistics and nulls as §1.3–§1.4 (2 000 permutations).
`handoff-10.md` Parked (from `/challenge-pr` on #96). **Inputs:** the pilot's
216 v1 runs, the same set as §1.11 (inputs `c183a7fcbd9e`: 8 v1 keys × 27 steps
of `HDD_1TB/Mets_archive/2026-08-12_05-01-35`, native labels). Git `10e44ea`,
`--v1-only`. `transport.py` and `p10_partition_function.py` take only
`--root`/`--pattern`, so they read a symlink root, which also keeps Stage 0's
dirs out (handoff Parked, "default glob"). The records' `root` is that
throwaway path. Rebuild it and re-run:

```bash
P=/run/media/system/HDD_1TB/Mets_archive/2026-08-12_05-01-35; R=<scratch>/pilot_v1/$(basename $P)
mkdir -p $R; for d in $P/pythia-410m-step*/; do case $d in *short_heterogeneous/) ;; *) ln -sfn ${d%/} $R/;; esac; done
<env as §0> python tools/run/transport.py --root <scratch>/pilot_v1 --v1-only --out data/analysis/p10_f1_transport_pilot.json
<env as §0> python tools/run/p10_partition_function.py --root <scratch>/pilot_v1 --v1-only --out data/analysis/p10_f12_z_pilot.json
```

Records `data/analysis/p10_f1_transport_pilot.json` (5 184 boundaries, 9 min) and
`p10_f12_z_pilot.json` (16 200 units, 21 min, 16 cores), beside §1.3–§1.4's
`p10_f1_transport.json` and `p10_f12_z.json` (152 runs, the same 8 prompts).
Per-unit counts below drop step 1 (step 0's weights, §3.51.3) and count only
units whose value differs between the sweeps: the rest are identical and
cannot disagree.

| on the 12 distinct shared steps | F1 clustered − noise step | F12 clustered − noise |
|---|---|---|
| max \|pilot − WDS\|, per-step mean | 0.005 | 0.008 |
| step 32, WDS / pilot | −0.294 / −0.290 | −0.122 / −0.118 (vs step 0: −0.587 / −0.588) |
| step 64, WDS / pilot | −0.317 / −0.318 | −0.141 / −0.145 (vs step 0: −0.606 / −0.615) |
| units whose value differs, same sign | 361 / 366 (step 32: 50 / 50) | 1 126 / 1 126 (step 32: 159 / 159) |
| largest single-unit move | 0.37 (step 64) | 0.60 (step 64) |

- **§1.3's and §1.5's numbers hold on the second partition.** At step 32, the least
  stable step between the partitions (§1.12), no changed unit flips sign in either
  row. Single units move by up to 0.4–0.6; the per-step means do not.
- **Per-unit p is not compared.** Both runners draw every null from one
  generator seeded once per sweep, so a unit's p depends on how many directories
  ran before it; most p disagreements are in units with bit-identical
  statistics (`/challenge-pr` on #97). Parked in the handoff.
- **Below the step-0 `Z` baseline lasts past 512 on both sweeps**, which §1.4's
  table left out (now added there). WDS: −0.10 / −0.09 / −0.13 at 1000 / 2000 /
  4000, +0.004 at 8000. Pilot: −0.19 at 3000, −0.13 at 5000, −0.04 at 7000,
  +0.04 at 9000, −0.05 at 11 000, then +0.06 to +0.21 from 13 000. F1's weaker
  late dip was in §1.3's 16000–32000 row; the pilot's 13 steps from 3000 to
  100 000 all sit between +0.012 and −0.153, then +0.044 at 120 000 and −0.017
  at 143 000.
- **What this shows.** The same activations clustered twice (§1.11): it bounds
  HDBSCAN's float-noise instability on these two rows, and it is small. It is
  not a second model, a second prompt set, or the missing control.

### 1.14 The re-read, R0: the label source on 1d's working definition — **built at 18 checkpoints; nothing refused; it reproduces the definition's counts exactly**

**What it is** (`design-10.md` "Order" R0; branch `claude/p10-r0`). An instrument, not a
reading: no Phase 10 row is re-read yet. Unit 1 (`move_text`) ran at all 18 distinct Stage 0
steps on unit 2's token set, and each P = 0 pass was checked against the stored Stage 0 run;
then the label source (`tools/run/p10_label_source.py`) was built from it. Per (step, prompt,
layer L1–24), the source holds c0 (stored labels, float32 distances) and c0f (the same call on float64; *added after `/challenge-pr` on #145*), both on every position, and c1, c1c, c2a, c2b, c2,
c3, plus the arms c3_c4 (centred, size 4) and c3_r2 (raw, size 2), all on the kept positions.
At step 143000 it also holds "learned" per c3 group. Its refusal rules are in the module
docstring.

**Inputs.** 410m, seed 0; the 7 v1 passages; token set `arch_null_trained_2026-10-02/step0/
token_sets.json` (sha `b1eaa3ab`; corrected in `design-10.md`); Stage 0 through
`stage0_index.json` only; learned from `candidates_2026-10-02/candidate_rows.json`. Code
`be309f8` (every record's `meta.git`). Output `data/p10/reread_r0_2026-10-05/` (`unit1/`, 84 MB; `labels/`, 23 MB;
`labels/summary.json` md5 `d9d41708`, rebuilt with c0f and record hashes at `ce37e1b`; logs `run_*.log`). 18 × 7 × 115 = 14,490 forward passes
on CPU (14 workers), ~6.5 min per checkpoint, ~2 h in all.

**Checks, all passed.**

| check | result |
|---|---|
| P = 0 against Stage 0 (126 (step, passage)) | max 2.9e-7 (direction), 1.6e-7 (relative norm); tolerance 1e-5 |
| token set at every step (`--kept-from` refuses otherwise) | no refusal: besides position 0, the only massive tokens are the two `\n` (`hdbscan_code` 34, `latex_monograph` 10), from step 8000 |
| re-run against unit 1's stored records, steps 0 and 143000 | identical in 672 / 672 cells each (groups, floors, classes, best Jaccards) |
| groups recomputed from Stage 0's activations against unit 1's (centred 2 and 4, raw 2) | equal at every (step, prompt, layer): 0 refused of 3,024 |
| c3 against `candidates definitions`' "moves" (centred / 2) | step 0: 35 / 21 / 6 = 62; step 143000: 317 / 336 / 157 = 810; exact |
| c3's member sets against `candidate_rows.json`'s non-bulk moving groups (*added after `/challenge-pr` on #145, finding 3*: counts alone cannot see a class on the wrong group) | 0 mismatched records at steps 0 and 143000 |
| c0 (stored, float32 distances) against c0f (same call, float64), every position | identical in 2,924 of 3,024 records; ARI mean 0.9995, p5 1.000, min 0.92 (`latex_monograph` L14, step 0): precision moves little on these 7 prompts (as `status-1d.md` "Float64 distances" found for 7 of 8) |
| learned at 143000 | 325 of c3's 810 records learned (= the moves ∧ learned set) |
| first checks (step 0 opening, designed prompts) | pass, as unit 1 |

**Group-layer records per column, and the readable (prompt, layer) records on c2 / c3 (of 168).**

| step | c1 | c1c | c2a | c2b | c2 | c3 | c3_c4 | c3_r2 | readable c2 / c3 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 2937 | 1930 | 1925 | 520 | **62** | 1 | 22 | 160 / 15 |
| 2 | 997 | 2917 | 1936 | 1931 | 521 | 61 | 1 | 22 | 160 / 15 |
| 4 | 903 | 2928 | 1955 | 1950 | 527 | 63 | 2 | 17 | 161 / 10 |
| 8 | 568 | 3056 | 1944 | 1939 | 505 | 55 | 3 | 11 | 160 / 10 |
| 16 | 495 | 3116 | 2020 | 2016 | 512 | 96 | 2 | 13 | 161 / 24 |
| 32 | 445 | 1200 | 865 | 857 | 349 | 67 | 8 | 57 | 150 / 34 |
| 64 | 610 | 1369 | 1004 | 968 | 445 | 248 | 70 | 56 | 156 / 110 |
| 128 | 972 | 1895 | 1421 | 1389 | 621 | 307 | 94 | 102 | 156 / 107 |
| 256 | 1295 | 1996 | 1591 | 1546 | 822 | 326 | 106 | 154 | 148 / 95 |
| 512 | 1194 | 2370 | 1857 | 1815 | 947 | 598 | 237 | 214 | 160 / 119 |
| 1000 | 1653 | 2638 | 1974 | 1927 | 869 | 670 | 245 | 290 | 128 / 111 |
| 2000 | 2216 | 3117 | 2260 | 2229 | 1055 | 817 | 282 | 570 | 143 / 128 |
| 4000 | 2549 | 3144 | 2440 | 2419 | 985 | 849 | 258 | 612 | 146 / 139 |
| 8000 | 2074 | 3323 | 2365 | 2337 | 1004 | 872 | 226 | 441 | 144 / 132 |
| 16000 | 1690 | 3329 | 2472 | 2448 | 1050 | 932 | 242 | 324 | 145 / 136 |
| 32000 | 1484 | 3294 | 2518 | 2495 | 1045 | 967 | 237 | 286 | 143 / 140 |
| 54000 | 1452 | 3431 | 2515 | 2493 | 1020 | 934 | 244 | 302 | 142 / 138 |
| 143000 | 1596 | 3031 | 2243 | 2209 | 926 | **810** | 211 | 328 | 138 / 130 |

**What it means for R1–R3 (counts only; no row is read).** (1) Under `design-10.md`'s
readability rule (half the records readable on c3), **steps 0–32 read on c2, not c3**: c3 has
10–34 readable records of 168 there, and from step 64 on it has 95–140. The switch is at
**32 → 64, inside F1's 32–512 window** (*corrected after `/challenge-pr` on #145, finding 2*;
the first version put it at F1's 16 → 32 edge), so R2's design must say how a window that
spans the switch is read. (2) The step-0
floor holds through step 16 (55–96 records against 62) and rises from step 64. (3) c1c
(shipped, centred) gives ~1.3–1.6× c2a's groups at most steps, the tie artefacts of
`status-1d.md` "Admission" on this token set; the bulk filter (c2a → c2b) removes ≤ 4 % (most at step 64). (4) c2
→ c3 (the move filter) removes 81–89 % at steps 0–16 and 7–23 % from step 2000 on, as unit 1
found at the two ends.

**Caveats.** "Learned" exists at step 143000 only: unit 2's bars are stored, but each group's
`s` needs Gaussian draws at the step's clouds, which were not run (Parked in `handoff-10.md`).
One passage set (7); one seed. Each R-unit adds its reader's `--labels <dir> --column <c>`
and refuses without it (R1's three readers: §1.15) (`load_column` refuses a refused record,
an unknown column, and a column's positions outside its domain).

**Re-run:** `data/p10/reread_r0_2026-10-05/run_r0.sh step0 step512 step54000 …` from the
worktree root (resumable; step 0 first, since its first checks gate the rest), then
`python tools/run/p10_label_source.py summary --src <out>/labels --definitions
data/p1d/candidates_2026-10-02/definitions.json --rows data/p1d/candidates_2026-10-02/candidate_rows.json`.
Tests: `tests/test_p10_label_source.py` (13),
`tests/test_phase1d_move_text.py`. **Next: R1** (§1.7 unique tokens, §1.9, §1.10), after
this PR merges.

### 1.15 The re-read, R1: §1.7 (unique tokens), §1.9 and §1.10 down the ladder — **no row holds by `design-10.md`'s rule (one label at every cell it names); the quoted class claims keep their labels at the steps they were quoted, except §1.7's class sign at step 512; adjacency and the 64–512 embedding dip change, and adjacency turns on the step-0 baseline**

**What ran** (`design-10.md` "Order" R1). The three readers take one column by `--labels
<R0 source> --column <c>` and refuse without it (`--old-partition` reads the stored labels, to
reproduce §1.7–§1.12). Under a column, the draw, `adjacent`'s neighbours, the kNN control and
carry's reference set range over the column's domain (kept tokens; all positions for c0 / c0f),
and unreadable (prompt, layer) records are left out and counted. Each column is one record;
`tools/run/p10_r1_ladder.py` applies each row's own rule to every column, c3 taking c2's step-0
baseline and c0–c2 their own, and names for each label that differs from c0 the first column
that changed it and the column it **settled at** (the first from which every column through
the primary carries the primary's label; added here because c1, the raw frame, swings and later
columns swing back).

**Inputs.** The R0 label source (`data/p10/reread_r0_2026-10-05/labels/`, `summary.json` md5
`d9d41708`); 7 v1 passages × 18 steps, L1–24; 12 columns (the ladder, the two arms, and c3 split
by learned at step 143000). Code at this PR. Output `data/p10/reread_r1_2026-10-05/` (36
records, `ladder.json` md5 `ddd48bbb`), 32 min on 6 processes. **Primary column:** c2 at steps
0–32, c3 from 64 (c3 readable in 15–34 of 168 records at 0–32). **Floor:** 62 c3 group-layer
records at step 0, 15 of 168 records readable, 1–2 prompts at each of 11 layers and none at L12
or L24. **What the quoted values rest on:** c0 168 / 168 readable records (7 prompts at L12 and
L24); c3 119 at step 512 and 130 at 143000 (6 prompts at L12 / L24); c3_learned 84 (3 prompts
at L12, 5 at L24), c3_unlearned 114 (6 / 6).
**Baselines** (*departure from `design-10.md`, flagged after `/challenge-pr` on #146*): c3, the
arms and the learned split subtract c2's step-0 value, as the design says; c0–c2b subtract their
own step 0, so c0 stays the published reading, where the design's words ("at every column")
read literally put c2's step 0 under every column. Both are counted below; the literal reading
changes §1.9 most (144 against 170 agreeing). For the user (`STATE.md` Blocked 24).

**Checks, all passed.** c0 reproduces the published per-run records of all three readers
exactly on the shared runs (3,015 readable of 3,024 records each; same 126 run dirs). c3's
readable counts equal the label source's. **BLAS threads matter at the record level:** a first
run at `OMP_NUM_THREADS=1` (kept in `omp1/`) differed from the published §1.9 records by ≤ 0.003
and from §1.10's by up to 0.27 in one record's 40-bin kNN control, because the float32 layer-0
Grams change with the thread count (5e-7 between 1 and 16) and copies sit on exact ties; at the
default threads c0 is identical. **No reading label changes between the two runs** (0 of 648
primary labels; `ladder.py --against`).

**Labels that agree with c0** (primary vs c0; by the design's rule a row holds only at all of
its cells, so **none holds**). "Rule cells" are the cells the row's own rule names (§1.7: its
verdict per step; §1.9: every Δ; §1.10: readings 1–3 at step 143000 and the 40-bin kNN excess at
512, 2000, 143000; *added after `/challenge-pr` on #146*, finding 2: the first count used extra
quantities and a three-way carry label where rule 3 is binary at +0.05).

| row | all cells | the rest settled at | rule cells | literal baseline: all / rule |
|---|---|---|---|---|
| §1.7 (freq, class, verdict per step) | 41 of 54 | c1c 6, c2 5, c1 1, c2a 1 | 16 of 18 (32 "consistent", 2000 "against") | 41 / 16 |
| §1.9 (5 lifts × L12, L24, mean × 17 steps) | 170 of 255 | c2 38, c3 16, c1 16, c1c 7, c2a 7, c2b 1 | 170 of 255 | 144 / 144 |
| §1.10 (6 readings × 3 layers × 17 steps) | 232 of 306 | c1 23, c2 21, c3 10, c1c 8, c2a 6, c2b 6 | 16 of 18 (L24 at 143000: `emb_given_class`, kNN excess) | 222 / 17 |

**§1.7, unique tokens** (contrasts per step; ±0.05; c2 at ≤ 32):

| step | 0 | 8 | 32 | 64 | 128 | 512 | 2000 | 8000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|
| freq c0 / primary | −0.02 / −0.00 | −0.05 / −0.00 | −0.03 / +0.08 | +0.08 / +0.21 | +0.12 / +0.15 | +0.03 / +0.03 | −0.03 / −0.08 | +0.01 / −0.04 | −0.00 / +0.01 |
| class c0 / primary | +0.09 / +0.29 | +0.01 / +0.27 | −0.03 / +0.32 | −0.02 / +0.02 | +0.02 / +0.02 | +0.07 / −0.11 | +0.01 / −0.11 | −0.03 / −0.13 | −0.13 / −0.07 |

- **Rank does nothing, trained: same label** (within ±0.05 from 4000 on). On c3 rank matters at
  64–256 (+0.21, +0.15, +0.07), where c0 already read "+" (+0.08, +0.12, +0.14), and at 1000–2000
  (−0.10, −0.08).
- **Class predicts, against trash collection: same label at 143000 (−0.07 / −0.13), and it
  comes earlier, which flips the sign at step 512** (c0 +0.07, c3 −0.11). c3's class contrast is
  −0.07 to −0.14 at every step from 512 on but 1000 (−0.03); c0's turns only at 16000. At 0–32 (on
  c2) unique punctuation is *more* often a member than word starts (+0.27 to +0.32). The rule's
  verdict changes at 2 of 18 steps: "consistent" at 32 (c2), "against" at 2000.

**§1.9, Δ against step 0** (L12 / L24 / mean; c0 against its own step 0, c3 against c2's):

| lift | 512 c0 | 512 c3 | 143000 c0 | 143000 c3 | c3 floor (step 0; mean of 11 layers, 15 records) | 143000 learned / not |
|---|---|---|---|---|---|---|
| same_class | +0.25 / +0.31 / +0.28 | +0.22 / +0.20 / +0.24 | +0.19 / +0.21 / +0.20 | +0.17 / +0.16 / +0.21 | +0.01 | +0.27 / +0.16 |
| copy_share | −0.17 / −0.21 / −0.14 | −0.11 / −0.05 / −0.09 | −0.17 / −0.08 / −0.12 | −0.08 / −0.19 / −0.12 | −0.10 | −0.13 / −0.09 |
| no_copy | +0.14 / +0.09 / +0.11 | −0.04 / −0.04 / −0.00 | +0.16 / +0.00 / +0.13 | +0.04 / +0.25 / +0.14 | −0.06 | +0.13 / +0.07 |
| adjacent | −0.05 / −0.00 / −0.02 | −0.18 / −0.21 / −0.16 | +0.10 / +0.09 / +0.05 | +0.10 / −0.14 / −0.11 | −0.28 | −0.14 / −0.09 |
| emb_pct_own | −0.12 / −0.05 / −0.06 | −0.01 / +0.04 / +0.02 | +0.10 / +0.02 / +0.12 | +0.20 / +0.12 / +0.21 | +0.07 | +0.23 / +0.19 |

- **Class: same label at every cell from step 64 on** (3 differ, at 8–32). Learned groups carry
  more of it (+0.27 against +0.16; learned rests on 3 prompts at L12).
- **Away from copy groups: same label at the layer mean** (−0.12 both); L12 and L24 trade places.
  **Unique-only clusters: same label trained at the mean and L24** (+0.14, +0.25; L12 +0.04), but
  on c3 they rise at 2000, not 512.
- **Own-embedding similarity: same label trained, and larger** (+0.21 against +0.12 at the mean);
  **c0's dip at 64–512 is the old partition's** (as step 0 on c3; most settle at c2).
- **Adjacency: the label changes, and the reason is the baseline.** 29 of 51 cells differ, 23
  settle at c2. c3's raw trained lift is *larger* than c0's (L12 0.44 observed against 0.05
  expected, +0.39; c0 +0.20), but stable step-0 groups (c2) are runs of neighbours (lift +0.29 /
  +0.37 at L12 / L24), so against c2's step 0 the trained L24 reads "below". **c3's own step 0
  cannot stand in** (*corrected after `/challenge-pr` on #146*, finding 1): it has no readable
  record at L12 or L24; its layer mean (0.03 against 0.03) rests on 15 records, 1–2 prompts at
  each of 11 layers, and the design calls the floor "a count, not a baseline" (`STATE.md`
  Blocked 24).

**§1.10** (L12 / L24 / mean; the deciding reading first, a level; the rest Δ as §1.9):

| reading | 512 c0 | 512 c3 | 143000 c0 | 143000 c3 |
|---|---|---|---|---|
| class beyond a lexical cluster, 40 bins (CGE40 − kNN) | +0.20 / +0.20 / +0.20 | +0.15 / +0.15 / +0.16 | −0.01 / +0.05 / +0.02 | +0.04 / +0.07 / +0.04 |
| class_given_emb (10 bins) Δ | +0.24 / +0.27 / +0.25 | +0.21 / +0.19 / +0.23 | +0.11 / +0.15 / +0.13 | +0.09 / +0.11 / +0.12 |
| emb_given_class Δ | −0.13 / −0.06 / −0.07 | −0.02 / +0.02 / +0.01 | +0.07 / −0.02 / +0.08 | +0.16 / +0.09 / +0.15 |
| carry gap (focal − unclustered `self_pct`) | −0.00 / +0.01 / −0.01 | +0.01 / −0.02 / +0.00 | −0.04 / −0.10 / −0.04 | +0.00 / −0.06 / −0.01 |

- **Step 512, the network computes the class grouping: same label** (+0.15 beyond the matched
  lexical cluster at L12 and L24). **Trained, mostly what the embedding groups: same label at L12**
  (+0.04); L24 is +0.07 on c3 against +0.05 on c0, which crosses the floor (one of the two rule
  cells that differ).
- **Within class, embedding neighbours: now at L12 and L24 trained** (+0.16 / +0.09; c0 only
  L12; the other rule cell that differs). c0's 64–512 dip below step 0 (here and in `emb_same`,
  `emb_cross`) is the old partition's.
- **Carry, rule 3 (focal − unclustered `self_pct` > +0.05): "not" on both.** The gap is smaller
  on c3: −0.06 against −0.10 at L24, +0.00 against −0.04 at L12.

**Arms** (where a label differs from c3's, steps read on c3): c3_c4 in 9 / 41 / 51 of 36 / 180
/ 216 cells (§1.7 / §1.9 / §1.10), c3_r2 in 11 / 53 / 60; in `ladder.json`, not read here.

**Caveats.** Tier 1, no null; ±0.05 placed. 7 prompts, one seed. The c2 / c3 switch at 32 → 64
makes the step-64 Δs the first on c3 against a c2 baseline. "Settled at" is post hoc. Learned
exists at 143000 only. **Re-run:** `data/p10/reread_r1_2026-10-05/run_r1.sh` from the worktree
root (default BLAS threads), then `python tools/run/p10_r1_ladder.py --dir <out> --labels <R0
labels> [--against <out>/omp1]`. Tests: `tests/test_p10_r1_ladder.py`, the readers' tests,
`tests/test_p10_label_source.py`.

### 1.16 The re-read, R2: F1, F12's gap and §1.5 down the ladder — **F1 does not hold by the rule (9 of 18 steps): the strong 64–512 difference is there on the definition, and the late tail keeps its size but loses significance on c3's smaller member sets; F12's gap reads "below" at every step because c2's step 0 is the densest population on the axis, so its label is the baseline's; §1.5's parked window is F1's, 64–512**

**What ran** (`design-10.md` "Order" R2). `transport.py` (F1) and `p10_partition_function.py`
(F12) take one column by `--labels <R0 source> --column <c>` and refuse without it
(`--old-partition` reads the stored labels). Under a column the per-token quantity is unchanged
(each particle's step L → L+1; corrected log Z over **every** stored position, the model's own
context), so a column changes the partition and not the quantity; **members − rest** compares the
column's members with the kept tokens in no group, permuted among kept tokens, 2,000 draws, one
generator per unit (closes the "per-directory seeds" Parked item for these two readers). F1 reads
boundaries from L1 (L1–23); F12 pools β = 1, 2, 4 per step, as §1.4. `tools/run/p10_r2_ladder.py`
applies the rules: F1 negative / positive / none (median p ≤ 0.05 and the mean's sign); F12 Δ vs
step 0, below / as / above at ±0.05, **raw value and baseline printed beside every Δ** (Blocked 24
(a), user 2026-10-05); c3, the arms and the learned split against c2's step 0, c0–c2b against their
own (Blocked 24 (b), user 2026-10-05: the departure is accepted); §1.5 parked iff F1 negative ∧ F12
below. The 32 → 64 c2 / c3 switch (§1.14) is handled by reading every window on the primary and
also on c2 and c3 throughout (table below).

**Inputs.** The R0 label source (`summary.json` md5 `d9d41708`); 7 v1 passages × 18 steps; 12
columns. Code at this PR (base `88ead8e`). Output `data/p10/reread_r2_2026-10-05/` (24 records,
`ladder.json` md5 `220c972d`, 15 MB), 14 min on 14 processes at one BLAS thread each.

**Check, passed.** c0 against the published records, matched by run-dir name, layer and β: **2,891
of 2,891 F1 units and 9,045 of 9,045 F12 units identical**. The published records read the earlier
WDS sweep (2026-08-31 dirs), not Stage 0, so on these 7 prompts the two sweeps' activations and
stored labels agree at every unit. c3's readable counts equal the source's.

**Per step** (primary: c2 at 0–32, c3 from 64; F1 mean / median p; F12 raw value, baseline):

| step | F1 c0 | F1 primary | F12 c0 raw (base +0.35) | F12 c2 raw (base +1.32) | F12 c3 raw |
|---|---|---|---|---|---|
| 0 | +0.01 / 0.21 | −0.20 / 0.21 (c2) | +0.35 | +1.32 | −0.15 (15 records) |
| 16 | −0.12 / 0.08 | **+0.41 / 0.009 (c2)** | +0.33 | +0.44 | −0.06 |
| 32 | −0.48 / 0.0005 | −0.31 / 0.064 (c2); c3 −0.86 / 0.0005 | −0.35 | −0.10 | −0.29 |
| 64 | −0.49 / 0.0005 | −0.54 / 0.0035 | −0.38 | −0.28 | **−0.75** |
| 128 | −0.48 / 0.0005 | −0.51 / 0.0005 | −0.16 | +0.08 | −0.38 |
| 256 | −0.60 / 0.0005 | −0.46 / 0.0055 | −0.05 | −0.10 | −0.27 |
| 512 | −0.33 / 0.0005 | −0.25 / 0.035 | +0.08 | +0.01 | −0.01 |
| 1000 | −0.15 / 0.012 | −0.03 / 0.065 | +0.25 | +0.13 | −0.03 |
| 4000 | −0.01 / 0.013 | −0.02 / 0.14 | +0.21 | +0.25 | +0.17 |
| 16000 | −0.11 / 0.036 | −0.13 / 0.17 | +0.54 | +0.35 | +0.31 |
| 143000 | −0.01 / 0.10 | −0.08 / 0.11 | +0.59 | +0.27 | +0.24 |

| window (steps whose label is…) | c0 | c2 | c3 | primary |
|---|---|---|---|---|
| F1 negative | 32–54000 | 64–512 | 32–512 | 64–512 |
| F12 below baseline | 32–4000 | 8–143000 | 0–143000 | 8–143000 |
| §1.5 parked | 32–4000 | 64–512 | 32–512 | 64–512 |

Labels that agree with c0 (primary vs c0): **F1 9 of 18, F12 10 of 17, §1.5 13 of 17**; every
differing label first changes at c1 (the token rules) or c1c (the centred frame), none at the
definition's own filters (c2a–c3).

- **F1: does not hold by the rule; the strong window is there, the late tail is a power change.**
  On the primary the label is negative at 64–512 (−0.25 to −0.54) and none from 1000 on (median p
  0.065–0.17); c0 under the same rule is negative from 32 to 54000. **The rule reads significance,
  not size:** the published 8-prompt record gives the same long tail (median p 0.005–0.039 at
  1000–54000, means −0.15 to +0.04), so §1.3's "window" was a magnitude reading (published −0.29 to
  −0.49 at 32–512, −0.15 or smaller after) that `design-10.md`'s rule did not encode. **The late
  tail's size is about the same on c3** (−0.02 to −0.13 at 1000–143000, against c0's −0.01 to
  −0.15) and its label drops to none because c3's records are smaller: median 50 members against
  c0's 252 (*corrected after `/challenge-pr` on #147*, finding 1; the first version said the tail
  "goes"). At 32 the primary is c2 (none, −0.31, median p 0.064); c3 alone reads −0.86 on ≤ 34
  readable records. At step 16 stable groups (c2) move *more* than the rest (+0.41).
- **F12: the gap's label is the baseline's.** c2's step-0 value is +1.32 at every β (1.27 / 1.31 /
  1.37): stable step-0 groups are the densest population on the axis in corrected Z. Against it, c3 reads "below" at every step (Δ −2.07 at 64,
  −1.08 at 143000). Each column's own step 0 rises down the ladder (c0 +0.35, c1 +0.91, c1c +0.58,
  c2a +0.70, c2b +0.69, c2 +1.32) while trained values fall (143000: +0.59, +0.92, +0.38, +0.38,
  +0.32, +0.27), so c0's "above baseline from 16000" reverses at c1c, the centred frame. **The raw
  trajectory keeps c0's shape on c3:** members sit below the rest at 32–256 (−0.29 to −0.75, median
  p ≤ 0.043), at the rest at 512–2000, above from 4000 (+0.17 to +0.36); the β = 1, 2, 4 means differ by ≤ 0.16 (most at step 32).
  c3's own step 0 is −0.15 (median p 0.19, 15 records, 1–2 prompts per layer): **on the definition
  the step-0 density confound is not visible** (§1.4's +0.465 was c0's), but 15 records are the
  floor, "a count, not a baseline" (`design-10.md`). *Replaced by §1.18* (Blocked 25 (iv), user
  2026-10-05): against a matched control the definition reads below at 32–256, above from 2000.
- **§1.5: parked, not pinned, at 64–512 on the primary.** With F12 below at every step on c3, the
  parked window is F1's, so it inherits F1's power caveat. On the raw sign instead (members below the rest in Z by more
  than 0.05), it is 32–256 on c3: §1.5's 32–64 holds on both readings, and the raw window ends at
  256 on c3 and on the c0 re-read (−0.05 there; the published 8 prompts ended at 64).
- **Arms:** F12's labels identical to c3's at every step on both arms; F1's differ at 5 (c3_c4,
  negative at 1000, 2000, 32000–143000) and 3 (c3_r2) of 12. **Learned at 143000:** F1 −0.08 / −0.06
  (learned / not), F12 raw +0.43 / +0.09.

**Caveats.** Tier 1; the permutation p is per unit and the merge is a median, not an e-value. 7
prompts, one seed. c3 at 0–32 is too thin to read (primary c2). "Settled at" is post hoc. Z on every
position includes position 0 and the two `\n` tokens in every context sum; only the members − rest
split is on the kept tokens. **Re-run:** `data/p10/reread_r2_2026-10-05/run_r2.sh` from the
worktree root, then `python tools/run/p10_r2_ladder.py --dir <out> --labels <R0 labels> --published
data/analysis`. Tests: `tests/test_p10_r2_ladder.py`, `tests/test_p10_transport.py`,
`tests/test_p10_partition_function.py`.

---

### 1.17 The re-read, R3: A0 under T4 down the ladder — **A0 does not hold: the raw flip (§1.1's "~94 % mask") is two tokens per prompt, position 0 and one massive token. Out of the means on c0's own labels, the sweep's raw gap falls from +0.49 to +0.06 and reverses from 16000 on; c0's late "learned residual" goes with them. Under T4 the sweep-pooled raw gap is ≤ 0 in every column, and where a step has a flip the mask explains under 0.4 of it from 1000 on. On the definition a residual beyond the mask and beyond position (members against the rest in the same position bins) is a window, +0.05 to +0.09 at 2000–16000, gone from 32000; never significant per unit**

**What ran** (`design-10.md` "Order" R3). `p10_attention_baseline.py --labels <R0 source>` reads
all 12 columns on one load of each run's attention (`--old-partition` reads the stored labels; one
or the other, else it refuses). c0 and c0f run the published `measure_layer` unchanged. From c1 on,
**T4**: the columns of the prompt's T1–T2 positions (position 0 and the token set's massive token,
one per prompt: `\n` in `hdbscan_code` / `latex_monograph`, the first `.` in the prose; unit 2's
`token_sets.json`, sha `b1eaa3ab`, checked) are dropped and rows renormalised
(`core.parking.t4_attention`); the mask baseline is the content-free one under the same rule
(`t4_received_baseline`, closed form checked in `tools/math_checks/causal_mask_attention_baseline.py`
§5 and against the explicit matrix in `tests/test_core_parking.py`). Means and nulls are on the
column's domain (the kept tokens), position bias in the prompt's own n − 1. One generator per unit
(step, prompt, layer), the same in every column. `tools/run/p10_r3_ladder.py` applies the rule
(its docstring): **gap** = rest − members (A0's published sign); sweep **mask share** 1 − corrected
/ raw, ≥ 0.9 or < 0.9, **no flip** where the raw gap is ≤ 0; per step **residual** iff corrected gap
≥ 0.05 (placed; checked first on the published record, where it gives 2000–4000 and "persists", as
§1.1 reads it, `LESSONS.md` 6); **appears** = the interval before the first residual step;
**persists** iff residual at every step after. Judged on the primary against **c1** (T4 changes
the statistic). Significance beside, never in a label.

**Inputs.** R0's label source (`summary.json` md5 `d9d41708`), 7 v1 passages × 18 steps, L1–23
(L24 has no block above it, as published). Code at this PR (base `5b58411`). Output
`data/p10/reread_r3_2026-10-05/` (`a0.json` md5 `4b2ec6ab`, `ladder.json` `09df92da`,
`sink_split.json` `5fef0bed`; 11 MB), 5 min on 14 processes.

**Check, passed.** c0 against the published record (`data/analysis/p10_row_a0.json`, the 2026-08-31
WDS sweep, 8 prompts) by run-dir name and layer: **2,891 of 2,891 units identical** in all seven
per-unit values. c0 on the 7 prompts reads as §1.1 does: sweep share **0.96** (≥ 0.9), residual
appears 2000–4000 and persists.

**Which half of c0 → c1 removes the flip** (`tools/run/p10_r3_sink_split.py`, c0's own labels,
no permutations; gap rest − members, raw / corrected):

| step | all positions (c0) | T1–T2 out of the means | out, and T4 |
|---|---|---|---|
| 0 | +0.14 / +0.003 | +0.12 / +0.003 | +0.12 / +0.003 |
| 256 | +0.17 / −0.085 | +0.15 / −0.085 | +0.16 / −0.085 |
| 2000 | +0.07 / +0.024 | +0.06 / +0.030 | +0.06 / +0.047 |
| 4000 | **+0.74 / +0.154** | +0.10 / +0.027 | +0.16 / +0.072 |
| 16000 | +1.10 / +0.133 | −0.08 / −0.054 | −0.02 / −0.008 |
| 143000 | **+1.58 / +0.105** | −0.24 / −0.150 | −0.16 / −0.082 |
| sweep | **+0.49 / +0.018** | +0.06 / −0.045 | +0.09 / −0.027 |

Both T1–T2 tokens are noise in 42 % of c0's units. Up to 2000 the two tokens move the raw gap by
≤ 0.02 and the corrected gap not at all: §1.1's "entirely mask at initialisation" stands. From 4000,
where §1.1 found the flip at its largest and the residual, both are these two tokens; T4's
renormalisation changes little beside taking them out.

**Per step** (primary: c2 at 0–32, c3 from 64; corrected gap (raw gap)):

| step | c0 | c1 | c2 | primary | primary median p |
|---|---|---|---|---|---|
| 0 | +0.00 (+0.14) | +0.00 (−0.52) | +0.00 (−1.18) | (c2) | 0.36 |
| 256 | −0.09 (+0.17) | +0.01 (+0.07) | −0.03 (−0.18) | −0.08 (−0.03) | 0.96 |
| 512 | −0.19 (+0.02) | +0.06 (+0.22) | −0.04 (−0.13) | −0.08 (−0.10) | 0.85 |
| 1000 | −0.10 (+0.02) | +0.22 (+0.21) | +0.03 (−0.14) | +0.02 (+0.02) | 0.45 |
| 2000 | +0.02 (+0.07) | +0.39 (+0.50) | +0.08 (−0.01) | **+0.07** (+0.11) | 0.34 |
| 4000 | +0.15 (+0.74) | +0.24 (+0.14) | +0.10 (+0.03) | **+0.08** (+0.05) | 0.28 |
| 16000 | +0.13 (+1.10) | +0.09 (−0.25) | +0.10 (−0.07) | **+0.10** (−0.06) | 0.35 |
| 143000 | +0.10 (+1.58) | −0.04 (−0.44) | +0.05 (−0.21) | **+0.11** (−0.14) | 0.29 |

| label | c0 | c1 | primary | holds (primary = c1) |
|---|---|---|---|---|
| sweep share | 0.96, ≥ 0.9 | no flip (raw −0.02) | no flip (raw −0.48) | same |
| residual appears | 2000–4000 | 256–512 | 1000–2000 | **no** |
| persists to 143000 | yes | no (none from 32000) | yes | **no** |

Per-step residual labels: 13 of 18 agree with c1. The five that differ first change at c1c (512,
1000: the centred frame), c2a (32000), c2 (54000) and c3 (143000).

- **The flip.** The share label reads the sweep pooled over training (as §1.1 did), which
  `aggregate`'s own docstring warns against; per step there are flips (*corrected after
  `/challenge-pr` on #148, finding 1*; the first version said "no column has a flip"). c1's raw gap
  is positive at 9 of 18 steps: the mask explains 0.98–1.00 of it at 8–32, 0.75–0.88 at 64–512,
  0.22 at 2000 (+0.50 raw), and less than none at 1000 and 4000. The primary's is positive only at
  1000–4000 (+0.02 to +0.11; shares 0.36 or below), and its pooled −0.48 is mostly c2's steps 0–32
  (−1.18; stable step-0 groups sit early, position bias −0.23). c3 alone is positive at 0–16
  (+0.05 to +0.21, share ≥ 0.96, on too few records to be primary). §1.1's "~94 % mask" was a share
  of a gap that two tokens per prompt carried from 4000.
- **The residual on the definition is mostly position after 16000** (*added after `/challenge-pr`
  on #148, finding 2*). The mask correction divides out uniform attention only, so a trained
  model's own position profile stays in it, and from 8000 the definition's members get *more* raw
  attention than the rest while sitting earlier. `tools/run/p10_r3_position_bins.py` (sizes only)
  reads the corrected gap inside 8 equal-count position bins of the domain (placed), members
  against the rest at the same positions (`position_bins.json`, md5 `c3891a72`):

  | corrected gap, pooled / binned | 2000 | 4000 | 8000 | 16000 | 32000 | 54000 | 143000 |
  |---|---|---|---|---|---|---|---|
  | c0 | +0.02 / +0.02 | +0.15 / +0.16 | +0.12 / +0.12 | +0.13 / +0.13 | +0.13 / +0.12 | +0.12 / +0.11 | +0.11 / +0.09 |
  | c1 | +0.39 / +0.32 | +0.24 / +0.14 | +0.22 / +0.07 | +0.09 / −0.09 | +0.03 / −0.16 | −0.01 / −0.20 | −0.05 / −0.21 |
  | c3 | +0.07 / +0.09 | +0.08 / +0.09 | +0.12 / +0.06 | +0.10 / +0.05 | +0.12 / +0.00 | +0.10 / −0.00 | +0.11 / +0.01 |

  On c3 the binned residual is a window, +0.05 to +0.09 at 2000–16000, ≤ 0.014 from 32000: by the
  rule on binned values, residual appears 1000–2000 and does **not** persist, at floors 0.03, 0.05
  and 0.075. c0's residual is not position (binned ≈ pooled); it is the two tokens (split table
  above). c1's large 1000–2000 gap (+0.39 at 2000, not position) is shipped HDBSCAN on the
  uncentred kept tokens; its label drops at 1000 in c1c, the frame column (c2 pooled: +0.03 / +0.08 at 1000 / 2000).
  Not significant per unit: median p 0.28–0.39 from 2000, merged E 2.7 over c3's sweep (2.2 on
  c0; rejection at 20), and the permutation null does not hold position either.
- **The 0.05 floor** (*finding 3*): "A0 does not hold" is the same at every floor from 0.024 to
  0.105 (where c0 reproduces the published labels). The primary's pooled "appears" moves with it
  (1000–2000 up to 0.06, 2000–4000 at 0.075, 4000–8000 from 0.09), so that interval is not a
  finding; the binned one above is steadier.
- **Arms** agree with c3 at all 12 steps read on c3. **Learned at 143000:** pooled corrected gap
  +0.18 on learned groups (79 units), +0.04 on the rest (108), but binned −0.05 and +0.01: the
  learned groups' gap is their position (the first version read it as "the residual sits on
  learned groups").

**Caveats.** Tier 1; the per-unit p is a permutation p and the merge is the mean. 7 prompts, one
seed. The 0.05 floor and the 0.9 bar are placed. The split table is sizes only, on c0's labels,
not a re-read. T4's content-free baseline is uniform causal attention through the rule, not a model
of where a trained network sends the sink's mass. **Re-run:**
`data/p10/reread_r3_2026-10-05/run_r3.sh` from the worktree root, then `python
tools/run/p10_r3_ladder.py --record <out>/a0.json --labels <R0 labels> --published
data/analysis/p10_row_a0.json` and `python tools/run/p10_r3_sink_split.py --labels <R0 labels>
--out <out>/sink_split.json --jobs 14`, and `p10_r3_position_bins.py` the same way
(`<out>/position_bins.json`). Tests: `tests/test_p10_attention_baseline.py`,
`tests/test_p10_r3_ladder.py`, `tests/test_core_parking.py`.

### 1.18 The re-read, R2m: F12 against a matched control — **on the definition (c3), members are less dense than the same tokens were at init at 32–256, as dense at 512–1000, and denser from 2000 (+0.12 to +0.27, 6–7 of 7 prompts; sign p 0.016 at 4000, 8000, 16000, 54000, 0.125 at the rest); c2, the primary at 8–32, already reads below from 8; c3's members were not dense at init (control −0.08 to +0.13 from 256 on), so R2's "below at every step" was c2's step-0 baseline; §1.5's parked window on the primary is 64–256**

**What ran** (`design-10.md` "R2m", rule fixed before any control output was read; `STATE.md`
Blocked 25 (iv), user 2026-10-05). `p10_partition_function.py --activations-step 0` scores
each column's labels at step s on step 0's activations of the same prompt (`matched_runs`
refuses unless `tokens.txt` matches); everything else is R2's (statistic, β grid, unit
generator, domain). `tools/run/p10_r2m_ladder.py` pairs every R2 unit with its control
(refuses an unpaired unit, and any Δ ≠ 0 at step 0: none), labels the mean matched Δ below / as /
above control at ±0.05, with the raw sign (Blocked 25 (ii)) and the control beside; §1.5 = R2's F1
negative ∧ F12m below. *Added after `/challenge-pr` on #149 (finding 3), beside and not a label:*
a two-sided sign-test p over the 7 prompts (`sign_p`; a prompt at Δ = 0 has no sign; 0.016 is
7 of 7, 0.125 is 6 of 7).

**Inputs.** The R0 label source (`summary.json` md5 `d9d41708`); R2's records
(`data/p10/reread_r2_2026-10-05/`, unchanged); 7 v1 passages × 18 steps; 12 columns. Output
`data/p10/reread_r2m_2026-10-05/` (12 records + `ladder.json` md5 `9300a087`, 12 MB), 11 min on
11 processes at one BLAS thread each. Code at this PR (base `f8672a8`).

**Per step** (Δ = trained − control, mean over units; prompts with mean Δ < 0, of 7):

| step | c0 Δ (trained, control) | primary | primary Δ (trained, control) | prompts Δ<0 | label |
|---|---|---|---|---|---|
| 0–4 | 0.00 (+0.35) | c2 | ≤ 0.01 (+1.32) | — | as |
| 8 | +0.07 (+0.35, +0.28) | c2 | −0.22 (+1.14, +1.36) | 7 | below |
| 16 | +0.12 (+0.33, +0.20) | c2 | −0.88 (+0.44, +1.33) | 7 | below |
| 32 | −0.57 (−0.35, +0.22) | c2 | −0.99 (−0.10, +0.89) | 7 | below |
| 64 | −0.64 (−0.38, +0.25) | c3 | −0.84 (−0.75, +0.10) | 7 | below |
| 128 | −0.39 (−0.16, +0.23) | c3 | −0.44 (−0.38, +0.06) | 7 | below |
| 256 | −0.23 (−0.05, +0.18) | c3 | −0.19 (−0.27, −0.08) | 6 | below |
| 512 | −0.09 (+0.08, +0.17) | c3 | +0.04 (−0.01, −0.06) | 1 | as |
| 1000 | +0.01 (+0.25, +0.23) | c3 | +0.01 (−0.03, −0.03) | 4 | as |
| 2000 | +0.06 (+0.25, +0.19) | c3 | +0.12 (+0.06, −0.05) | 1 | above |
| 4000 | +0.13 (+0.21, +0.09) | c3 | +0.18 (+0.17, −0.02) | 0 | above |
| 16000 | +0.41 (+0.54, +0.13) | c3 | +0.27 (+0.31, +0.04) | 0 | above |
| 143000 | +0.46 (+0.59, +0.12) | c3 | +0.18 (+0.24, +0.06) | 1 | above |

| window (steps whose label is…) | c0 | c2 | c3 | primary |
|---|---|---|---|---|
| F12m below control | 32–512 | 8–256 | 32–256 | 8–256 |
| F12 raw below the rest | 32–256 | 32, 64, 256 | 0–256 (thin at 0–32) | 32–256 |
| §1.5 parked | 32–512 | 64–256 | 32–256 | 64–256 |

Labels that agree with c0 (primary vs c0): **F12m 14 of 17, raw sign 15 of 17, §1.5 15 of 17.**
F12m differs at 8 and 16 (c0 above, c2 below; changes at c1c, the centred frame) and at 512 (c0
below, c3 as; settles at c2); §1.5 differs at 32 (c2's F1 is none there) and 512. **Two of
F12m's three are c2's** (*added after `/challenge-pr` on #149, finding 1*): c3 itself reads above
its control at 8 and 16 (+0.05 on 30 units, +0.12 on 72; 2 of 5 and 0 of 5 prompts below), as
c0 does, so **c3 against c0 agrees at 16 of 17, differing only at 512**, and the definition's
own below-window is 32–256 (−0.68 at 32, 7 of 7 prompts, on ≤ 34 readable records).

- **The density confound, on the definition, is the baseline's, not the tokens'.** The control
  (members' tokens on init activations) is +0.89 to +1.36 on c2 at 0–32, where c2's members are
  step-0-like groups, but **−0.08 to +0.13 on c3 from 256 on** (median control p 0.17–0.25; +0.10
  and +0.06 at 64–128, p 0.35), so
  the tokens training puts in c3 were not denser at init. c0's control is +0.09 to +0.35 at every
  step: part of the old partition's raw sign is token-level density present at init (§1.4).
- **The late "above" is training's.** From 2000, c3's members are denser than the same tokens at
  init by +0.12 to +0.27, in 6–7 of 7 prompts at every step (sign p 0.016 at 4000, 8000, 16000,
  54000; 0.125 at 2000, 32000, 143000, so the window is weakest at its first and last steps). On c0
  the crossover is **between 1000–2000 (this control) and 8000–16000 (§1.4, its step-0 baseline)**:
  the two bracket the selection effect from either side (next bullet).
- **What the control does not remove** (*added after `/challenge-pr` on #149, finding 2*): members
  are chosen by closeness at step s, and Z is a density, so a member set is denser than the rest
  partly by selection. At step 0 the control equals the trained value and holds that selection in
  full; later it holds it only as far as the members' tokens stay close at init, so what it
  subtracts changes along the trajectory. On c3 this is smaller (its own step 0 is −0.15, 15
  records). The reviewer's position checks on c3 (states unit-norm; position 0 out: Δ +0.10 to
  +0.24 late; each token's own term out: +0.26 to +0.40; members' mean position within ~15 tokens
  of the rest's) say the late "above" is not R3's position mechanism; they are the review's, not
  re-run here.
- **Learned at 143000:** Δ +0.37 (learned) against +0.07 (not), so the late excess is mostly the
  learned groups'.
- **Arms:** c3_r2 identical to c3 on every row; c3_c4 differs at 512 (below) and 1000 (above),
  and is parked at 512.

**Caveats.** Tier 1; the only per-step test is the sign test over 7 prompts (minimum p 0.016);
the units within a prompt are not independent. c3 at 0–32 is thin (10–34 readable records), so the primary there is
c2, whose members are a different population (step-0-like stable groups); the "below" at 8–32 is
c2's. The control holds tokens and positions, not context: Z at init is computed from init
activations everywhere, so a Δ is training's change in those positions' density, not in their
neighbours'. No norm-matched random twin; one init. **Re-run:**
`data/p10/reread_r2m_2026-10-05/run_r2m.sh` from the worktree root, then `python
tools/run/p10_r2m_ladder.py --dir <out> --r2 <R2 out> --labels <R0 labels>`. Tests:
`tests/test_p10_r2m_ladder.py`, `tests/test_p10_partition_function.py` (incl. that step s's
labels are scored on step 0's run, *after `/challenge-pr` on #149, finding 4*).

### 1.19 R5c: 5c's flip on `gpt2-large` with the sink out — **does not survive. On this random arm it was never trained-specific: with position 0 in, random weights give the same raw gap (+1.52 against trained +1.59; trained above random in 11 of 21 prompts). With position 0 out of the means, trained's raw gap is −0.03 (7 of 21 positive) and random's stays +1.44, all of it causal mask. The one trained-specific part, the gap beyond the mask (+0.23, 16 of 21), is position 0: out, −0.02 (9 of 21)**

**What ran** (`design-10.md` "R5c", rule fixed before any output was read; `STATE.md` Blocked
26 (e), user 2026-10-05). `tools/run/p10_r5c_gpt2_sink.py` applies R3's sink split
(`p10_r3_sink_split.run`) to each stored run's own HDBSCAN labels (5c's partition), per
(prompt, layer), with the prompt as the unit (mean over its 36 layers).

**Inputs.** `gpt2-large` `data/phase12/2026-09-19_13-40-48/`, `gpt2-large-random`
`…_15-47-26/` (CLAIM-C's arms; battery v2 `06790b90dcfe`, the 8 v1 prompts, `short_heterogeneous`
and the 12 v2 prompts held out on 410m only; numpy / torch seed 0, so one random draw; the
battery corrected after `/challenge-pr` on #150, finding 4), 21 prompts × 36 layers = 756 units per arm, all readable; per-file sha256 in the
output's `inputs`. The 9-prompt pair (10-51-00 / 11-52-10) is byte-identical on labels and tokens
over its overlap. Output `data/p10/reread_r5c_2026-10-05/r5c.json` (md5 `b16433a1`, 388 KB),
90 s on 6 processes. Code at this PR (base `a1f89b6`).

**T1–T2 is position 0 alone, in every prompt.** At layers 2–32 trained position 0's norm is
25.5–28.5× its layer median (min, median over prompts); no other position exceeds 2.03×, so the
10× bar is not near a decision. In the random arm position 0 is 1.1–1.4×: there it is dropped by
T1, not as a massive token. **Position 0 is unclustered in 756 of 756 trained units** (752 of 756
random): the Parked question's answer is yes, the sink sat in 5c's unclustered population.

| arm (gap = unclustered − clustered enrichment) | trained | random | trained > 0 | trained > random | reading |
|---|---|---|---|---|---|
| `all`, raw (5c as published) | +1.59 | +1.52 | 21 | 11 | does not survive |
| `all`, mask-corrected | +0.23 | +0.00 | 16 (p 0.013) | 16 (p 0.013) | survives |
| **`drop`, raw (primary)** | **−0.03** | **+1.44** | **7** | **4** | **does not survive** |
| `drop`, mask-corrected | −0.02 | +0.00 | 9 | 9 | does not survive |
| `drop+T4`, raw | +0.02 | +1.53 | 7 | 4 | does not survive |
| `drop+T4`, mask-corrected | +0.01 | +0.00 | 11 | 11 | does not survive |

(of 21 prompts; one-sided sign test.)

- **5c's sign flip is not reproduced even with the sink in.** `all`'s enrichments: trained
  unclustered 2.18×, clustered 0.61× (5c: ~1.6×, ~0.5×, same order); random 2.50× and 0.98×,
  not "near parity or clustered-favored". Random's gap is entirely mask (corrected +0.004; +0.004
  pooled, +0.004 in R3's position bins): its unclustered tokens sit early, where a causal mask
  routes attention. 5c's random arm cannot have been this one: neither `gpt2-large-random` nor
  `albert-base-v2-random` could load through `run_1` before 2026-09-17 (`e191d77`; the handler
  skipped the model and the sweep finished without it). So **both of 5c's random arms have no
  recorded run**, and the disagreement is not resolved here. *(ALBERT added after `/challenge-pr`
  on #150, finding 1.)*
- **What is trained-specific is the gap beyond the mask, and it is position 0.** With it in, the
  corrected gap is +0.23 against random's +0.00 (16 of 21 both ways); out, −0.02 (pooled) and
  −0.016 within position bins, 9 of 21. As on 410m (§1.17), where the trained part was position 0
  and one massive token.
- **By depth, `drop` raw (thirds of 12 layers):** trained +0.00, −0.09, +0.01; random +1.08,
  +1.67, +1.57. No third holds a trained flip once position 0 is out.
- **Does not control:** token frequency (`archive/p5c_unclustered/lit-5c.md` §1); a random arm
  matched in norm (this one is HF's default re-init); ALBERT (no run survives; Parked in
  `handoff-10.md`). One HDBSCAN partition (`min_cluster_size=2`), as 5c read: not 1d's
  definition, which would need gpt2-large's own label source.

Re-run: `python tools/run/p10_r5c_gpt2_sink.py --trained <…_13-40-48> --random <…_15-47-26>
--out <file> --jobs 6` with `METS_DATA` set. Tests: `tests/test_p10_r5c_gpt2_sink.py`.

### 1.20 R6: the cross-checkpoint matcher — **built; between adjacent checkpoints from 4000 to 54000, 56–67 % of the definition's groups (c3) survive into the next step (51 % over the long last step), mostly with changed members (median Jaccard 0.75–0.83, identical 22–30 %). Few of c3's births and deaths are new groups (from 4000, 14–49 per boundary): the group is in c2a, either unchanged and failing or passing a filter (332–388) or split or merged by c2a (169–313). No 143000 c3 group traces an unbroken chain back past 256 and 89 % start after 2000, yet chains last longer than independent breaks would give (11.5 % reach 2000 against 3.9 %). The old partition's 17 % of 143000 clusters back to step 0 sit off the definition's positions: the same call on the kept positions (c1) has none**

**What ran** (`design-10.md` "R6", rule fixed before any matcher output was read; `STATE.md`
Blocked 26 (b), user 2026-10-05). `tools/run/p10_r6_matcher.py` links each step's groups to the
next step's at the same (prompt, layer) with `merge_tree.link_layer_pair` (containment ≥ 0.5;
components stable / split / merge / tangle / birth / death), on columns c0, c1, c1c, c2a, c2,
c3 (c1 and c1c added after `/challenge-pr` on #151, finding 2; the other four columns reproduce
the first run exactly). No forward pass.

**Inputs.** R0's label source `data/p10/reread_r0_2026-10-05/labels/` (summary sha256
`28498b12`); 7 v1 passages × L1–24 × 18 steps (17 boundaries); every record read, none refused;
each column's domain identical at every step of a prompt (checked; the run refuses otherwise).
Output `data/p10/reread_r6_2026-10-05/r6.json` (md5 `8a956856`; the first run, before the
review, kept as `r6_v1_620e9f7.json`), ~1 min on 14 processes.
Null: 200 draws, seed (0, prompt index, layer). Code at this PR (base `86bd460`).

**First check passes:** c2a at 0 → 2, stable share 0.9995 (bar 0.9); 1,925 of 1,929 stable links
identical. *Corrected at reading:* the rule called 0 → 2 **and 2 → 4** the matcher's floor; 2 → 4
already moves (c2a stable 0.86, 82–89 % per prompt; R0's c1 count goes 997 → 903 there), so only
0 → 2 is a floor.

**Per boundary, pooled over 168 (prompt, layer) chains.** Stable = share of the earlier step's
groups whose component is 1–1; death = share with no link; J = the stable links' Jaccard (median;
share identical); c3's births + deaths by the same id's fate in c2a's matching: **kept** (c2a
1–1: the group persisted and passed or failed a filter), **restr.** (c2a split, merged or tangled
it), **new** (no c2a link). *Kept and restructured split after `/challenge-pr` on #151, finding
1: the first version called both "filter flips".*

| from → to | c3 groups | c3 stable | c3 death | c3 J (med; ident) | c3 flips kept / restr. / new | c2a stable | c2a merge | c2a J (med; ident) | c1 stable | c0 stable |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 → 2 | 62 → 61 | 0.97 | 0.03 | 1.00; 1.00 | 3 / 0 / 0 | 1.00 | 0.00 | 1.00; 0.99 | 0.87 | 0.94 |
| 2 → 4 | 61 → 63 | 0.64 | 0.36 | 1.00; 0.92 | 39 / 6 / 1 | 0.86 | 0.06 | 1.00; 0.72 | 0.41 | 0.89 |
| 4 → 8 | 63 → 55 | 0.29 | 0.71 | 1.00; 0.72 | 55 / 25 / 2 | 0.67 | 0.13 | 0.83; 0.44 | 0.20 | 0.84 |
| 8 → 16 | 55 → 96 | 0.22 | 0.78 | 0.88; 0.33 | 74 / 39 / 14 | 0.54 | 0.13 | 0.67; 0.28 | 0.18 | 0.81 |
| 16 → 32 | 96 → 67 | 0.06 | 0.94 | 0.30; 0.00 | 56 / 37 / 58 | 0.21 | 0.16 | 0.43; 0.19 | 0.16 | 0.78 |
| 32 → 64 | 67 → 248 | 0.36 | 0.63 | 0.40; 0.08 | 115 / 64 / 85 | 0.41 | 0.19 | 0.42; 0.15 | 0.18 | 0.79 |
| 64 → 128 | 248 → 307 | 0.35 | 0.62 | 0.65; 0.06 | 136 / 121 / 102 | 0.44 | 0.13 | 0.54; 0.12 | 0.29 | 0.77 |
| 128 → 256 | 307 → 326 | 0.33 | 0.66 | 0.67; 0.10 | 257 / 124 / 44 | 0.54 | 0.19 | 0.64; 0.20 | 0.43 | 0.74 |
| 256 → 512 | 326 → 598 | 0.40 | 0.55 | 0.62; 0.28 | 252 / 236 / 142 | 0.45 | 0.20 | 0.60; 0.20 | 0.28 | 0.59 |
| 512 → 1000 | 598 → 670 | 0.39 | 0.54 | 0.52; 0.08 | 267 / 273 / 166 | 0.37 | **0.34** | 0.56; 0.15 | 0.34 | 0.64 |
| 1000 → 2000 | 670 → 817 | 0.44 | 0.48 | 0.57; 0.08 | 284 / 414 / 79 | 0.43 | **0.30** | 0.60; 0.18 | 0.39 | 0.64 |
| 2000 → 4000 | 817 → 849 | 0.50 | 0.43 | 0.67; 0.15 | 388 / 285 / 72 | 0.55 | 0.22 | 0.67; 0.23 | 0.42 | 0.66 |
| 4000 → 8000 | 849 → 872 | 0.56 | 0.36 | 0.75; 0.22 | 344 / 268 / 19 | 0.57 | 0.23 | 0.75; 0.33 | 0.35 | 0.68 |
| 8000 → 16000 | 872 → 932 | 0.62 | 0.33 | 0.78; 0.24 | 367 / 217 / 31 | 0.65 | 0.16 | 0.80; 0.36 | 0.37 | 0.68 |
| 16000 → 32000 | 932 → 967 | 0.67 | 0.30 | 0.80; 0.26 | 388 / 169 / 32 | 0.68 | 0.14 | 0.80; 0.37 | 0.40 | 0.71 |
| 32000 → 54000 | 967 → 934 | 0.64 | 0.32 | 0.83; 0.30 | 384 / 197 / 14 | 0.69 | 0.16 | 0.83; 0.41 | 0.44 | 0.73 |
| 54000 → 143000 | 934 → 810 | 0.51 | 0.44 | 0.80; 0.28 | 332 / 313 / 49 | 0.55 | 0.24 | 0.80; 0.40 | 0.46 | 0.70 |

**Lineage origin of the groups at 143000** (cumulative share whose unbroken stable chain starts at
or before the step; c3 on its own links and, by id, on c2a's). In brackets, *added after
`/challenge-pr` on #151, finding 3*: the share if every boundary broke chains independently at
its own pooled rate (the product of stable components / later groups over the boundaries
crossed; a baseline, not a null):

| column (groups) | ≤ 0 | ≤ 256 | ≤ 2000 | ≤ 16000 | ≤ 54000 |
|---|---|---|---|---|---|
| c0, every position (7161) | 0.168 (0.006) | 0.217 (0.032) | 0.376 (0.128) | 0.531 (0.381) | 0.711 (0.711) |
| c1, kept positions, c0's call on float64 (1596) | 0.000 (0.000) | 0.007 (0.000) | 0.075 (0.006) | 0.183 (0.084) | 0.415 (0.415) |
| c1c (3031) | 0.000 (0.000) | 0.011 (0.000) | 0.081 (0.013) | 0.245 (0.138) | 0.512 (0.512) |
| c2a (2243) | 0.000 (0.000) | 0.021 (0.003) | 0.140 (0.053) | 0.383 (0.282) | 0.617 (0.617) |
| c3, own links (810) | 0.000 (0.000) | 0.007 (0.001) | 0.115 (0.039) | 0.351 (0.251) | 0.594 (0.594) |
| c3, by id on c2a (810) | 0.000 | 0.021 | 0.173 | 0.443 | 0.668 |

- **The null is weak, as expected for small groups:** every boundary's stable count beats all
  200 permuted chains in every column (p = 1/201, the floor); c3's draws average 0.2–30 stable
  links against 6–620 observed. It rules out chance overlap and nothing more.
- **Few of c3's births and deaths are new groups.** From 4000 on, 14–49 per boundary have no c2a
  link; of the rest, 51–70 % are the same group unchanged in c2a passing or failing a filter, and
  the others ride a split or merge in c2a (at 54000 → 143000, 332 against 313; at 1000 → 2000 the
  restructured are the majority, 414 against 284). At 32–1000 new / gone is a larger share
  (44–166 per boundary), as c2a forms. On c2a itself, groups survive at 0.55–0.69 per boundary
  from 2000.
- **Chains last longer than independent breaks give** (lineage table, brackets): on c3, 11.5 %
  reach back to 2000 against 3.9 % expected, 35 % to 16000 against 25 %; on c2a 14.0 % against
  5.3 %. A group that survived one boundary is more likely to survive the next. "No lineage past
  256" stands as an absolute; "young" does not, without the baseline.
- **c0's long lineages are off the definition's positions** (finding 2). c1 is c0's call
  (`HDBSCAN(min_cluster_size=2)`) on float64 over the kept positions only, and none of its 1,596
  groups at 143000 traces back to step 0, against c0's 1,204 of 7,161. Precision alone moves c0
  little (c0 → c0f identical in 2,924 of 3,024 records, §1.14), so the difference is the positions
  c0 adds: position 0, repeats and the first delimiter. *The reviewer's probe:* none of the 1,204
  lies wholly inside the kept set, about 164 would by chance, and they sit in early layers.
- **Survival means changed members.** A stable link at 4000–54000 keeps a median 75–83 % of its
  union; 22–30 % (c3) and 33–41 % (c2a) are identical sets. Over many steps this compounds: the lineage table.
- **c2a's merges peak at 512 → 2000** (0.30–0.34 of groups absorbed), the coarsening window of
  F1 (`status-10.md` §1.16: 64–512 strong difference, late tail).
- **16 → 32 is a break in every filtered column** (c2a 2020 → 865 groups, stable 0.21; c3 0.06):
  R0's group counts fall there too (§1.14 table). Not explained here.
- **Spread over prompts** (stable share, c3): 0.44–0.78 per prompt at 8000–54000, 0.00–0.71 below
  1000, where several prompts have under 10 c3 groups.
- **Does not:** follow a group across layers, read §1.9–§1.10's drift within lineages (the
  Parked question: from these numbers, adjacent steps keep most groups with changed members, and
  over the whole axis lineages are replaced; which of the two carries the drift is that row's),
  or say why a group changed. One seed, 7 passages, containment 0.5 placed. A lineage ends at
  any split, merge or filter flip, so the lineage table is a lower bound on how long content
  persists.

Re-run: `python tools/run/p10_r6_matcher.py --labels data/p10/reread_r0_2026-10-05/labels --out
<file> --workers 14` with `METS_DATA` set (conda `mets`); it refuses without the label source's
`summary.json`. Tests: `tests/test_p10_r6_matcher.py` (11).

### 1.21 R6w: §1.9–§1.10's drift within lineages — **the drifts measured in the embedding are not carried by replacement: on the definition (c3), from 512 to 143000, the groups entering and leaving at each boundary sit where the persisting ones do, and the persisting groups move, mostly at 512 → 2000. Class beyond a lexical cluster: −0.12 on R1's records, within −0.11. Own-embedding similarity: +0.18, within +0.17. Within-class embedding similarity: +0.15, within +0.12. The same holds on the rule's fixed records, on c2a and on c0. Where the drift happens, groups with identical members move at least as much as changed ones (17 and 11 links), so the drift is consistent with each step's own layer 0 moving under every group; a shift shared by all groups would read this way. Unique-only clusters (`no_copy`) read "within" on R1's records, "both" on the fixed set. `same_class` does not drift (the control). c3's single layers are too thin to read**

**What ran** (`design-10.md` "R6w", rule committed at `1a96524` before any output; `STATE.md`
Blocked 26 (b′), user 2026-10-06). `tools/run/p10_r6w_drift.py`: R1's per-token lifts, each focal
token weighted as R1 aggregates it, each tagged with its group's kind from R6's `link` at every
boundary it borders. Each boundary's change in the weighted mean is split exactly into **within**
(the persisting groups' mean change) and one term per kind of entry and exit, measured against the
persisting groups. On c3 those kinds are restructured, filter flip kept, filter flip restructured,
and new / gone. Per term, the boundaries are summed over the span. No forward pass.

**Inputs.** R0's labels (`data/p10/reread_r0_2026-10-05/labels/`, summary sha256 `28498b12`); R1's
records `data/p10/reread_r1_2026-10-05/{cm,lc}_{c3,c2a,c0}.json`; R6's `r6.json` (md5 `8a956856`).
Records: those readable **with ≥ 1 focal token at every step of the span**. The rule said
"readable"; a record with no focal token has no value, so this is what "the same population on
both boundaries" needs. c3 has 51 records over 512–143000 (L12 5, L24 2) and 39 over 64–512 (L12 3,
L24 0); c2a 132 / 168; c0 160 / 168. Output `data/p10/reread_r6w_2026-10-06/r6w.json` (md5
`e53a1a2f`), 1 min on 8 processes, at `1cbfecb`, runner clean. The first output (`215693d`,
kept as `r6w_v1_215693d.json`) recorded the rule commit `1a96524` as its git, but its runner was
uncommitted (`/challenge-pr` on #152, finding 4). Its fixed-set numbers are identical to the
re-run's.

**First checks, all passed:**
- (a) per-record values equal R1's stored records to 1.0e-15;
- (b) the terms sum to the total within 1e-9 at every boundary and level;
- (c) on identical links the four composition lifts' paired change is exactly 0 (381 links on c3);
- (d) stable components per boundary equal R6's, every column;
- (e) *added after `/challenge-pr` on #152:* on R1's own records (below), the span total equals
  R1's change within 1e-9 at every level and statistic.

**512 → 143000, layer mean** (span sums; "share" = within / total; prompts = those whose within
term has the sign of their own total, of those with |total| ≥ 0.05):

| statistic | column | total | within | restr. | flip kept | flip restr. | new / gone | label (share) | prompts | R1's change |
|---|---|---|---|---|---|---|---|---|---|---|
| CGE40 − kNN40 (class beyond a lexical cluster) | **c3** | −0.192 | −0.184 | +0.010 | +0.009 | −0.034 | +0.008 | **within** (0.96) | 4 of 5 | −0.121 |
| | c2a | −0.102 | −0.159 | +0.055 | | | +0.002 | within (1.57) | 4 of 4 | −0.093 |
| | c0 | −0.189 | −0.217 | +0.028 | | | −0.000 | within (1.15) | 6 of 6 | −0.182 |
| `emb_pct_own` | **c3** | +0.185 | +0.163 | +0.018 | −0.008 | +0.009 | +0.002 | **within** (0.88) | 6 of 6 | +0.184 |
| | c2a | +0.167 | +0.172 | +0.002 | | | −0.008 | within (1.03) | 7 of 7 | +0.166 |
| | c0 | +0.174 | +0.226 | −0.028 | | | −0.024 | within (1.30) | 7 of 7 | +0.177 |
| `emb_given_class` | **c3** | +0.110 | +0.086 | +0.012 | −0.001 | +0.009 | +0.004 | **within** (0.78) | 6 of 6 | +0.148 |
| | c2a | +0.144 | +0.141 | +0.008 | | | −0.004 | within (0.98) | 7 of 7 | +0.143 |
| | c0 | +0.146 | +0.189 | −0.025 | | | −0.019 | within (1.30) | 7 of 7 | +0.150 |
| `no_copy` | **c3** | +0.176 | +0.068 | +0.030 | +0.027 | +0.028 | +0.023 | **both** (0.38) | 3 of 5 | +0.144 |
| `copy_share` | **c3** | −0.063 | +0.018 | −0.017 | −0.034 | −0.018 | −0.012 | **replacement** (−0.29) | 2 of 4 | −0.025 |
| `same_class` | **c3** | −0.029 | −0.022 | +0.019 | −0.006 | −0.029 | +0.010 | no drift | | −0.027 |
| `adjacent` | **c3** | +0.042 | +0.164 | −0.055 | −0.021 | −0.034 | −0.012 | no drift | | +0.050 |

On c2a and c0, `no_copy` reads "replacement" (+0.092) and "no drift" (+0.019), and `copy_share`
"no drift" on both; `same_class` is "no drift" on c2a and "within" on c0 (−0.077).

**On R1's own records** (*added after `/challenge-pr` on #152, finding 2*). Each step reads every
record readable with ≥ 1 focal token there (c3: 111–140 per step, against the fixed set's 51). A
record readable at one end of a boundary only is its own kind, **records**, so the span total is
R1's change exactly (check (e)). c3, layer mean:

| statistic | total (= R1) | within | records | restr. | flip kept | flip restr. | new / gone | label (share) | prompts | fixed set's label |
|---|---|---|---|---|---|---|---|---|---|---|
| CGE40 − kNN40 | −0.121 | −0.112 | +0.021 | −0.011 | −0.005 | −0.015 | +0.002 | **within** (0.93) | 3 of 4 | within |
| `emb_pct_own` | +0.184 | +0.174 | +0.002 | +0.001 | +0.004 | +0.004 | −0.001 | **within** (0.95) | 7 of 7 | within |
| `emb_given_class` | +0.148 | +0.121 | +0.013 | +0.003 | +0.004 | +0.005 | +0.001 | **within** (0.82) | 7 of 7 | within |
| `no_copy` | +0.144 | +0.145 | −0.045 | +0.023 | +0.001 | +0.020 | +0.000 | **within** (1.01) | 4 of 5 | both |
| `copy_share` | −0.025 | −0.037 | +0.034 | −0.012 | −0.003 | −0.002 | −0.006 | no drift | | replacement |
| `same_class` | −0.027 | +0.017 | +0.004 | −0.016 | −0.017 | −0.019 | +0.004 | no drift | | no drift |
| `adjacent` | +0.050 | +0.143 | −0.045 | −0.023 | −0.014 | −0.009 | −0.003 | no drift, **opposing** | | no drift |

c2a and c0 on their own records give the same labels as on their fixed sets at the layer mean,
for all seven statistics. Records entering and leaving barely move the embedding statistics
(≤ 0.021). R1's records also give c3 readings at L12 and L24, but there within and the
replacement kinds have opposite signs in several rows (`r6w.json`, `levels_r1_set`); not read.

**Opposing terms** (*added after `/challenge-pr` on #152, finding 5*: within and the rest both
≥ 0.05 in size with opposite signs; flagged `opposing` in `r6w.json`). At c3's layer mean only
`adjacent`: within +0.164 against −0.122, net "no drift" on both record sets. The labels stay as
the rule fixed them; such a row is a cancellation, not an absence.

**Where on the axis** (c3, layer mean, total / within per boundary): CGE40 − kNN40 −0.106 / −0.097
at 1000 → 2000 and −0.049 / −0.045 at 2000 → 4000, ≤ 0.02 at every other boundary;
`emb_pct_own` +0.074 / +0.074 at 512 → 1000 and +0.073 / +0.069 at 1000 → 2000, then ≤ 0.03 in size.

**Identical against changed links** (c3, fixed set, the mean paired change per stable link,
identical / changed; per boundary *after `/challenge-pr` on #152, finding 3*, because the drift
sits at 512 → 2000):

| boundary | links (ident. / changed) | `emb_pct_own` | `emb_given_class` | CGE40 − kNN40 |
|---|---|---|---|---|
| 512 → 1000 | 17 / 120 | +0.108 / +0.052 | +0.030 / +0.044 | −0.126 / −0.015 |
| 1000 → 2000 | 11 / 167 | +0.092 / +0.056 | +0.067 / +0.051 | −0.142 / −0.123 |
| 2000 → 4000 | 35 / 183 | +0.025 / +0.017 | +0.008 / +0.004 | −0.069 / −0.039 |
| 4000 → 143000 (each of 5) | 51–77 / 164–195 | −0.001 to +0.013 / −0.022 to +0.018 | −0.002 to +0.013 / −0.011 to +0.016 | −0.019 to +0.031 / −0.020 to +0.015 |
| pooled, 8 boundaries | 381 / 1,369 | +0.013 / +0.015 | +0.007 / +0.013 | −0.014 / −0.025 |

c2a and c0, pooled: `emb_pct_own` +0.018 / +0.021 and +0.018 / +0.037; CGE40 − kNN40 −0.016 /
−0.017 and −0.001 / −0.030 (per boundary in `r6w.json`).

- **Replacement does not carry the drift §1.10 read as "trained, mostly what the embedding
  groups"** (wording *changed after `/challenge-pr` on #152, finding 1*; the first version said
  "is a within-lineage change"). At each boundary the entering and leaving groups sit where the
  persisting ones do, so every entry and exit term is ≤ 0.04 and within carries 0.82–0.96 on c3,
  on both record sets and all three columns. What this does **not** show is that lineage matters.
  These statistics are measured in each step's own layer 0. If that frame shifted every group
  alike, the shift would cancel out of the entry and exit terms (they compare groups at the same
  step) and all of it would land in within. So "within" here means the population moved as a
  whole, and the persisting groups with it. It can read "replacement" only if entrants differ
  from the persisting groups at their own step.
- **Consistent with the frame moving under fixed groups.** On a link whose members are identical,
  any change is that layer 0 moving (the rule's construction). At 512 → 1000 and 1000 → 2000, where
  the drift is, identical links move at least as much as changed ones in `emb_pct_own` and CGE40 −
  kNN40, but they are few (17 and 11). The reading this suggests, untested: the embedding comes to
  group what the depth groups already held. The per-link means are not decomposition terms and
  give no share; (b″) would (`handoff-10.md` Parked (R6w)).
- **Unique-only clusters (`no_copy`)** rise "within" on R1's records (+0.145 of +0.144) and "both"
  on the fixed set (0.38 of +0.18). The two record sets disagree, so neither is read. On c2a it is
  replacement (restructured +0.066). Per prompt the within term swings in sign (−0.57 to +0.30,
  fixed set).
- **The control holds:** `same_class` has no span drift on c3 (−0.03), matching R1 (−0.03).
- **The fixed record set changes c3's single layers.** With L12 at 5 records and L24 at 2, four
  c3 labels at L24 differ from R1's (e.g. `no_copy` +0.03 against R1's +0.29), so only the layer
  mean is read on c3; c2a and c0 have 6–7 records per layer. At the mean one label differs:
  `copy_share`'s −0.063 crosses the floor where R1's −0.025 does not, so its "replacement" is the
  fixed set's.
- **64 → 512 (secondary) is not readable on c3.** Its fixed set (39 records) does not reproduce
  R1's change: `same_class` +0.05 against +0.12, CGE40 − kNN40 +0.04 against +0.09. On c0, the
  rise of `same_class` (+0.16) reads "within" (0.67, at the bar), and `copy_share` and `no_copy`
  read "replacement" by new / gone groups (−0.14 / +0.14). In the network's first 512 steps, the
  old partition's copy-group drift is groups replaced.
- **Does not:** test anything (no null; 0.05, 2/3 placed); separate member turnover from the
  frame beyond the per-link split; follow groups across layers. "Within" includes changed members
  (R6: median Jaccard 0.75–0.83). The reference (persisting groups) is one choice (`design-10.md`
  "R6w"). One seed, 7 passages.

Re-run: `data/p10/reread_r6w_2026-10-06/run_r6w.sh` from the worktree root (default BLAS threads,
as R1). It refuses on any first check. Tests: `tests/test_p10_r6w_drift.py` (10).

### 1.22 R6f: R6w's within split into members against frame — **frame on the two embedding-similarity statistics, pooled and per prompt; mixed per prompt on class beyond a lexical cluster. Scored in one fixed layer 0 (step 512's and step 143000's), the pooled members part of the definition's (c3) within is ≤ 0.03 on all three statistics, and the rest is layer 0 moving under the groups, mostly at 512 → 2000. Per prompt the members part is not that small (−0.12 to +0.07 on the two frames' mean), so the pooled 0.03 is partly prompts cancelling. Still, `emb_pct_own` and `emb_given_class` read frame in 5–7 of 7 prompts on both record sets. CGE40 − kNN40 splits (fixed set: 3 frame, 1 both, 1 members, 2 below the floor). So §1.10's "trained, mostly what the embedding groups" reads, for embedding similarity, as the embedding coming to group what the groups held; for class beyond a lexical cluster this does not decide it. c2a reads frame per prompt on all three; c0's class-beyond-lexical is frame-dependent (members about a third)**

**What ran** (`design-10.md` "R6f", rule committed at `6f73b39` before any output; `STATE.md`
Blocked 26 (b″), user 2026-10-06). `tools/run/p10_r6f_frame.py`: every step's focal tokens
re-scored with **frame F's layer-0 Gram** (F = 512, 143000; same prompt, tokens checked) and R6w's
members, pools, weights and kinds unchanged; each R6w term T splits exactly into **members** = T in
frame F and **frame** = T − that. Layer 0 here is the token embedding (GPT-NeoX's rotary positions
add nothing there), so "frame" is the embedding matrix training under fixed groups. No forward pass.

**Inputs.** R0's labels (summary sha256 `28498b12`), R1's records, R6's `r6.json` (md5
`8a956856`), R6w's `r6w.json` (md5 `e53a1a2f`). Span 512 → 143000; c3 51 fixed records (R6w's
set), R1's own records beside. Output `data/p10/reread_r6f_2026-10-06/r6f.json` (md5 `bae452e6`),
3 min on 8 processes, at `269e88b`, runner clean. The first output (`6d068d3`, kept as
`r6f_v1_6d068d3.json`) stored per-prompt labels only; its span terms and shares are identical to
the re-run's, which adds per-prompt values (`/challenge-pr` on #153, finding 1).

**First checks, all passed:** (a) own-frame per-record values equal R1's to 1.0e-15; (a′) in frame
143000, each record's `emb_pct_own` equals R1's frozen-frame `emb_pct` to 4.4e-16; (b) at step F,
frame F's values equal the own frame's exactly; (c) on identical stable links the three
statistics' paired change in a fixed frame is exactly 0 (381 links on c3); (d) R6's stable counts;
(e) the own-frame span terms equal `r6w.json`'s within 1e-9, both record sets, every column.

**c3, layer mean, 512 → 143000, fixed set** (span sums of within; share = members / within; prompts
= per-prompt labels members / both / frame among those with |within| ≥ 0.05):

| statistic | within (R6w) | F = 512: members / frame (share) | prompts | F = 143000: members / frame (share) | prompts | label | Shapley |
|---|---|---|---|---|---|---|---|
| `emb_pct_own` | +0.163 | −0.009 / +0.173 (−0.06) | 0 / 0 / 7 | +0.010 / +0.154 (+0.06) | 2 / 2 / 3 | **frame** | 0.00 |
| `emb_given_class` | +0.086 | −0.013 / +0.099 (−0.15) | 0 / 0 / 7 | −0.004 / +0.090 (−0.04) | 2 / 3 / 2 | **frame** | −0.09 |
| CGE40 − kNN40 | −0.184 | −0.012 / −0.172 (+0.07) | 2 / 2 / 1 | −0.030 / −0.154 (+0.16) | 1 / 1 / 3 | **frame** | +0.12 |

**Per prompt, on the two frames' mean** (*added after `/challenge-pr` on #153, findings 1–2*: each
fixed frame favours its own step's groups, members mostly negative in frame 512 and positive in
143000, so the per-frame prompt columns above are not support; label = members part of the
two-frame mean against the prompt's own within; opposing = members and frame both ≥ 0.05 with
opposite signs):

| c3, statistic | record set | prompts' within | prompts' members (mean of frames) | members / both / frame / no drift | opposing |
|---|---|---|---|---|---|
| `emb_pct_own` | fixed | +0.11 to +0.30 | −0.12 to +0.07 | 0 / 1 / 6 / 0 | 2 |
| | R1's | +0.06 to +0.25 | −0.06 to +0.03 | 0 / 0 / 7 / 0 | 2 |
| `emb_given_class` | fixed | +0.06 to +0.29 | −0.11 to +0.07 | 0 / 2 / 5 / 0 | 1 |
| | R1's | +0.06 to +0.24 | −0.06 to +0.04 | 0 / 0 / 7 / 0 | 1 |
| CGE40 − kNN40 | fixed | −0.27 to +0.08 | −0.11 to +0.13 | 1 / 1 / 3 / 2 | 2 |
| | R1's | −0.26 to +0.11 | −0.07 to +0.14 | 2 / 1 / 3 / 1 | 0 |

c2a per prompt: frame in 6 of 6 drifting prompts on `emb_pct_own` and `emb_given_class`, both sets;
CGE40 − kNN40 6 of 6 (fixed), 4 frame / 1 both of 5 (R1's). c0: frame in 7 of 7 on the first two;
CGE40 − kNN40 1 / 2 / 3 / 1 on both sets.

On R1's own records the c3 labels are the same (shares −0.20 to +0.03; members ≤ 0.023 in size).
**c2a**: frame on all three, both sets (shares −0.19 to +0.24). **c0**: frame on `emb_pct_own` and
`emb_given_class` (shares −0.32 to +0.18), but its members parts are larger (−0.06 at F = 512,
+0.03 at 143000); CGE40 − kNN40 is **frame-dependent** on the fixed set (0.34 both / 0.26 frame) and
frame on R1's (0.32 / 0.26). The total splits the same way (c3 members part of the total ≤ 0.035).

**Where** (c3, fixed set, within per boundary, members / frame): `emb_pct_own` in frame 512,
+0.002 / +0.072 at 512 → 1000 and −0.012 / +0.081 at 1000 → 2000; CGE40 − kNN40 −0.016 / −0.081
at 1000 → 2000. From 4000 on, both parts ≤ 0.03 at every boundary in both frames. Changed stable
links, pooled: members part −0.013 to −0.002 per link in either frame (1,369 links on c3).

- **For embedding similarity, the embedding came to hold the groups.** Summed over boundaries,
  the groups persisting across each boundary change little in a frame that does not move, while in
  each step's own layer 0 they drift by 0.09–0.18; per prompt this holds in 5–7 of 7. It is a sum
  over persisting links, not one lineage followed from 512 (89 % of 143000's lineages start after
  2000, §1.20; wording *corrected after `/challenge-pr` on #153, finding 3*). R6w's identical
  links (whose whole change is frame by construction) were the visible edge of this.
- **For class beyond a lexical cluster, undecided.** Pooled it reads frame, but per prompt the
  members part runs −0.11 to +0.13 against withins of −0.27 to +0.08, and prompts split. This is
  the statistic §1.10's "lexical or not" turns on, so §1.22 does not say the groups did or did not
  become lexical.
- **The two frames differ by the interaction** (share at 512 minus share at 143000: −0.10 to
  −0.12 on c3), never enough to change a pooled label on c3 or c2a. Each frame favours its own
  step's groups, so per-prompt labels are read on the two frames' mean (table above).
- **On the old partition (c0) a third of the class-beyond-lexical drift is membership**: its
  groups did change toward what a fixed embedding calls "class beyond lexical" (−0.07 / −0.06
  members). The definition's filters remove that part.
- **Does not:** test anything (no null; 0.05, 2/3, 1/3 placed); say which frame is right; read
  before 512 (c3's 64 → 512 is not readable, R6w), L12 / L24, or the composition lifts (no frame);
  say why the embedding moved (gradient from the depth groups, or from unigram statistics, `design-10.md`
  Parked n-gram control). One seed, 7 passages.

Re-run: `data/p10/reread_r6f_2026-10-06/run_r6f.sh` from the worktree root (default BLAS threads,
as R1). It refuses on any first check. Tests: `tests/test_p10_r6f_frame.py` (13).

### 1.23 R7: the context-shuffle test — **the label depends on the bar, and most of the definition's groups that clear chance break under a token shuffle. From 512, on each group's own floor (the rule's yardstick) the step label is mixed: 39 % of c3's group-layer records survive with their token alone at 512, 48–61 % after; 31–39 % break under a full token shuffle. On the fixed bar 0.5 it reads context at every step (token-alone 0.21–0.31). Among groups whose own floor clears its chance level (54–68 % of records), breaking under a shuffle is the largest share (0.46–0.59) and token-alone 0.26–0.45; the floors below chance come out token-alone 68–91 %, and they lift the own-floor share. Deeper groups survive alone less (from 2000: L1–8 0.62–0.74, L17–24 0.34–0.46, own floor). Before 512 no shuffle moves any group (survival 0.94–0.99 at every block size): the model starts using order between 256 and 1000. Two limits on the labels: the 2-token `alone` input moves states even at init (cosine 0.74 at step 0, where a full shuffle leaves 0.99), so failing alone is not by itself "needs its context"; and breaking under a shuffle includes scrambling damage (11–24 % of token-alone groups also break). At 512, groups that break under a shuffle are as same-class as token-alone ones (0.83 against 0.86, own floor), so §1.10's class grouping there is not only a per-token feature**
*Title and bullets rewritten after `/challenge-pr` on #154 (findings 1–3): the first headline gave the own-floor label alone; the split by floor is new in `r7.json`, the rule's fields unchanged.*

**What ran** (`design-10.md` "R7", rule committed at `9949ed9` before any shuffled pass; `STATE.md`
Blocked 26 (c), user 2026-10-06). `tools/run/p10_r7_shuffle.py`: per (step, passage), the passage
as read (`orig`), its offsets 1 … n−1 block-shuffled (b = 64, 16, 4, 1; 5 permutations each, the
same at every step; offset 0 fixed), and each kept token `alone` as [offset 0, token]; in each, the
same kept tokens clustered (centred, level-set size 2, all groups), and each c2 group of R0 (c3 a
subset) scored by best-match Jaccard against its own unit 1 floor `J0` (median over the 5
permutations). **Token-borne** = survives `alone`; **bag-borne** = fails `alone`, survives b = 1;
**order-borne** = fails both. 22 forward passes per (step, passage), CPU.

**Inputs.** R0's labels and unit 1 records (`data/p10/reread_r0_2026-10-05/`; per-step label sha
in each record's `meta`); Stage 0 index sha `b013ab4e371936ed`; 7 v1 passages × 18 steps = 126
records, `data/p10/reread_r7_2026-10-06/records/`; reading `r7.json` (md5 `2bfedf3e`; the first
reading, `r7_v1_3a1de2a.json`, md5 `572cea82`, has every field but the split by floor, identical).
Runner at `233d67e`; ~17 s per (step, passage), 40 min in all.

**First checks, all passed:** (a) `orig` against Stage 0, max 2.9e-7 (tolerance 1e-5); (b) `orig`'s
groups are R0's c2a member sets at every layer (0 refused of 3,024) and R0's members are unit 1's;
(c) every shuffle a permutation with offset 0 fixed, tokens at their mapped positions; (d) `alone`'s
offset 1 against `orig`'s position 1, max 1.1e-5. *Departure at the gate:* (d)'s tolerance was
borrowed from (a) at 1e-5 and refused `latex_monograph` at step 143000 (1.06e-5); an unbatched
2-token pass is 1.5e-5 from the passage's pass there, so the gap is float32 sequence-length noise.
(d) now uses 1e-4 (`design-10.md` "R7"). Steps 0–512 are at ≤ 1.3e-7. T2 drops under shuffles: 0
before 1000, up to 240 token-conditions per passage at 143000 (`camus_letranger`).

**c3 (c2 before 64, where c3 is not readable), pooled over prompts and L1–24, own floor:**

| step | col | n | token / bag / order | label | survival b64 / b16 / b4 / b1 / alone | token share, fixed bar 0.5 | J0 < chance |
|---|---|---|---|---|---|---|---|
| 0 | c2 | 520 | 107 / 173 / 240 | context | 0.51 / 0.49 / 0.49 / 0.51 / 0.21 | 0.00 (c3) | 52 / 62 (c3) |
| 2–16 | c2 | 505–527 | 103–117 / 144–181 / 214–248 | context | 0.47–0.58 at every b / 0.20–0.23 | 0.00–0.02 | |
| 32 | c2 | 349 | 60 / 92 / 197 | context | 0.39 / 0.40 / 0.32 / 0.41 / 0.17 | 0.09 | |
| 64 | c3 | 248 | 140 / 102 / 6 | mixed | 0.97 / 0.99 / 0.98 / 0.98 / 0.56 | 0.38 | 100 |
| 128 | c3 | 307 | 133 / 157 / 17 | mixed | 0.94 / 0.95 / 0.96 / 0.94 / 0.43 | 0.19 | 110 |
| 256 | c3 | 326 | 94 / 213 / 19 | context | 0.94 / 0.93 / 0.96 / 0.94 / 0.29 | 0.09 | 100 |
| 512 | c3 | 598 | 235 / 157 / 206 | mixed | 0.94 / 0.84 / 0.70 / 0.61 / 0.39 | 0.23 | 190 |
| 1000 | c3 | 670 | 343 / 98 / 229 | mixed | 0.91 / 0.82 / 0.69 / 0.55 / 0.51 | 0.21 | 307 |
| 2000 | c3 | 817 | 390 / 125 / 302 | mixed | 0.91 / 0.77 / 0.62 / 0.53 / 0.48 | 0.24 | 307 |
| 4000 | c3 | 849 | 491 / 59 / 299 | mixed | 0.92 / 0.78 / 0.66 / 0.56 / 0.58 | 0.29 | 331 |
| 8000 | c3 | 872 | 463 / 71 / 338 | mixed | 0.92 / 0.80 / 0.63 / 0.54 / 0.53 | 0.25 | 328 |
| 16000 | c3 | 932 | 532 / 73 / 327 | mixed | 0.93 / 0.84 / 0.67 / 0.55 / 0.57 | 0.28 | 372 |
| 32000 | c3 | 967 | 516 / 100 / 351 | mixed | 0.94 / 0.85 / 0.68 / 0.54 / 0.53 | 0.23 | 368 |
| 54000 | c3 | 934 | 484 / 98 / 352 | mixed | 0.95 / 0.80 / 0.63 / 0.52 / 0.52 | 0.24 | 360 |
| 143000 | c3 | 810 | 498 / 58 / 254 | mixed | 0.93 / 0.82 / 0.69 / 0.54 / 0.61 | 0.31 | 309 |

c2 beside, from 512: token share 0.33 (512), 0.45–0.59 (1000–143000), label context at 512, mixed
after. Step 0's c2 baseline: 0.21 token-borne. Step 0's c3 floor (62 records) has 52 with `J0` below
chance: its groups are near-chance to begin with, as unit 1 found.

**Beside** (c3, from 512): per band, token share L1–8 / L9–16 / L17–24: 0.42 / 0.37 / 0.40 at 512,
0.62–0.74 / 0.49–0.59 / 0.34–0.46 from 2000. Per prompt (≥ 5 records): mixed in 3–6 of 7, token in
0–3, context in 0–2 at every step. Token-borne groups that break under b = 1: 25 at 512, 63–118
from 1000 (11–24 % of token-borne groups: a scrambled context breaks what no context keeps). Split
by floor (`by_floor`), from 512: `J0` at or above chance, n 363–599, token / bag / order
0.26–0.45 / 0.07–0.28 / 0.46–0.59; `J0` below chance, n 190–372, token-borne 0.68–0.91. Same-class share of
member pairs, token / bag / order: 0.86 / 0.88 / 0.83 at 512; from 2000, 0.79–0.86 / 0.82–0.86 /
0.67–0.73. Self-similarity (centred cosine to `orig`, median over records, members / rest): b = 1
0.97–0.99 to 256, 0.82 / 0.80 at 512, 0.47–0.59 from 1000; `alone` 0.49–0.81, members ≈ rest
throughout.

- **The model starts using its context's order between 256 and 1000.** Before 512 every group
  survives every shuffle and a full shuffle leaves each token's state at cosine ≥ 0.97 to its
  original; at 512 the dose–response appears (0.94 → 0.61 from b = 64 to 1) and by 1000 a full shuffle
  moves every token (0.58). From 1000 the curve barely changes. This fits Pythia's unigram-first stage
  (`lit-10.md` §16 row 3) and puts the change in the window where R6f found the embedding moving
  (512 → 2000, §1.22).
- **A token-only account is ruled out for a large part of the groups, and how large depends on the
  bar.** From 512, 31–39 % break under a full token shuffle on their own floor, 0.46–0.59 among
  floors that clear chance, 0.54–0.69 on the fixed bar. "Mixed" is the own-floor reading; the
  stricter readings lean to context. The floor check (`handoff-10.md` Parked (R7)) decides which
  bar the definition should use; R7's shares are not to be quoted in a later row before it.
- **Breaking under a shuffle is not only using order.** 11–24 % of token-alone groups also break
  under b = 1, so a scrambled context breaks some groups that need none; "order-borne" is read as
  "breaks under a token shuffle", which includes that damage.
- **Failing `alone` is not only needing context.** At step 0 a full shuffle leaves states at 0.99
  and `alone` moves them to 0.74; at 64–256, groups survive every shuffle yet 44–71 % fail `alone`.
  The 2-token input changes the geometry by itself, so "bag-borne" (fails `alone`, survives b = 1)
  mixes needing the other tokens present with that. Separating them needs an arm with a growing
  number of the passage's own tokens as context (not run).
- **Token-borne is front-loaded in depth.** L1–8 groups are mostly token-borne from 2000, L17–24
  mostly not. This agrees with early layers building tokens into words (`lit-10.md` §17 row 3).
- **For §1.10's class question at 512 (the Parked question):** order-borne and token-borne groups are
  equally same-class there (0.83, 0.86), so the class grouping at 512 is not only a per-token feature.
  From 2000, order-borne groups are less same-class (0.67–0.73 against 0.79–0.86). Class is
  orthographic: neither reading makes it semantic.
- **The own floor is lenient.** 32–46 % of c3's groups from 512 have `J0` below the chance level of a
  random same-size set; on the fixed bar the token share halves. The labels use the yardstick "moves"
  used; the fixed-bar row is the stricter reading.
- **Does not:** say what a group carries (token-borne ≠ lexical, order-borne ≠ semantic); control
  position in `alone` (every token at position 1 beside the sink; a token-borne miss can be position
  1, not context); keep words whole at b = 1 (Parked in `handoff-10.md`); test anything (no null; the
  2/3 and 1/3 thresholds placed). One seed, 7 passages.

Re-run: `data/p10/reread_r7_2026-10-06/run_r7.sh <steps>` from the worktree root (resumable; gate
512 and 143000 first). It refuses a (step, passage) on any first check. Tests:
`tests/test_p10_r7_shuffle.py` (12).

### 1.24 R8: the own-floor check on "moves" — **at the bar, not resolved by it: under a chance-aware bar (every EOD condition's best Jaccard ≥ max(`J0`, the 95 % chance level of a random same-size set, a P = 0 proxy)) c3 keeps 0.87–0.91 of its records at 64–1000, prompt-bootstrap intervals 0.77–0.97 straddling the placed 0.90, and one passage (`camus_letranger`) carries 64–256 below it; from 2000 it keeps 0.93–0.97 (intervals 0.87–0.99). What holds: step 0's floor halves (62 → 31; at init a random set clears the median group's `J0` in one condition 66–74 % of the time), and the drop is about twice as large at 64–1000 as after. Most groups whose `J0` is below chance (35–52 % of c3's records from 32) still pass, intact. Which set the rows read is the user's (Blocked 27)**

**What.** `design-10.md` "R8" (rule fixed before any chance level was computed; Blocked 26 (f),
user 2026-10-06). `tools/run/p10_r8_floor.py`: per (step, passage, layer), centred, size 2, a
chance level `Jc` per group size from 2000 random same-size sets of the kept offsets, scored by
best-match Jaccard against the P = 0 partition; c3c = c2 groups whose unit 1 class is still
`moves` when every EOD condition's J must reach max(`J0`, `Jc`). No forward pass, 8 min CPU.

**Inputs.** R0's labels (sha over the 18 step files in `r8.json`'s `meta`) and unit 1 records
(`data/p10/reread_r0_2026-10-05/`); output `data/p10/reread_r8_2026-10-06/r8.json` (md5
`95d915ec`), run on `fc1ab92` plus this PR's tool. **First checks, all passed:** the stored P = 0
groups are R0's c2a at every layer; the recomputed c2 and c3 are R0's; c3's totals are R0's (step
0: 62, 143000: 810). Beside: R7's 20-draw flag agrees with `J0` < `Jc` on 8,224 of 8,734 records.

**c3's group-layer records, pooled over prompts and L1–24:**

| step | c3 | c3c | keep | `J0` < `Jc` (share) | of those dropped | median share of random sets ≥ `J0` | `Jc` (median) |
|---|---|---|---|---|---|---|---|
| 0 | 62 | 31 | 0.50 | 55 (0.89) | 31 | 0.73 | 0.20 |
| 2 / 4 / 8 / 16 | 61 / 63 / 55 / 96 | 30 / 31 / 29 / 58 | 0.49–0.60 | 0.83–0.90 | 26–38 | 0.66–0.74 | 0.20 |
| 32 | 67 | 56 | 0.84 | 30 (0.45) | 11 | 0.005 | 0.18 |
| **64** | 248 | 223 | **0.90** (0.899) | 108 (0.44) | 25 | 0.000 | 0.17 |
| **128** | 307 | 268 | **0.87** | 135 (0.44) | 39 | 0.006 | 0.18 |
| **256** | 326 | 288 | **0.88** | 125 (0.38) | 38 | 0.002 | 0.19 |
| 512 | 598 | 545 | 0.91 | 211 (0.35) | 53 | 0.000 | 0.19 |
| **1000** | 670 | 595 | **0.89** | 349 (0.52) | 75 | 0.047 | 0.20 |
| 2000 | 817 | 757 | 0.93 | 351 (0.43) | 60 | 0.003 | 0.20 |
| 4000–54000 | 849–967 | 812–931 | 0.95–0.96 | 0.42–0.47 | 36–43 | 0.005–0.016 | 0.20–0.21 |
| 143000 | 810 | 789 | 0.97 | 364 (0.45) | 21 | 0.010 | 0.21 |

**Reading.**
- **By the letter of the placed rule c3 fails** (four primary steps below 0.90, lowest 0.87 at
  128; 64 misses by one record, 223 of 248), **but 7 prompts cannot resolve the bar** (*after
  `/challenge-pr` on #155, finding 1*). Bootstrapping prompts (5,000 draws), the 95 % interval of
  keep is 0.79–0.96 (64), 0.78–0.93 (128), 0.77–0.95 (256), 0.80–0.97 (512), 0.79–0.95 (1000),
  0.87–0.96 (2000), 0.91–0.99 (4000–54000), 0.95–0.99 (143000). Per prompt keep runs 0.29–1.0.
  Without `camus_letranger` (keep 0.29–0.65 at 64–256) those three steps keep 0.906–0.918; 1000
  stays 0.888. A rule needing 0.90 at all 12 steps would likely fail at a true share near 0.93
  too. So the reading is: the drop is ~10 % at 64–1000 and 3–7 % from 2000, where F1's and F12's
  windows are.
- **At init, "moves" is mostly chance.** Steps 0–16 keep about half: for the median c3 group a
  random set clears `J0` in one condition 66–74 % of the time. Unit 1's "trained groups move, step
  0's do not" stands, and c3c widens it (step 0 31 against 789 at 143000).
- **A floor below chance is not a chance pass.** Of the 35–52 % of records whose `J0` is below
  chance from 32, 6–25 % drop: most of those groups come through every preamble at J well above
  `Jc`. R7's "lenient floor" matters for R7's labels, much less for "moves". (*A check, not a
  finding (finding 4):* every dropped record has `J0` < `Jc` by construction, since only those
  groups get a higher bar. The dropped are not a random sample, so a re-read on c3c can shift
  small effects at any step, not only at 64–1000 (finding 3).)
- c3's groups are almost all larger than 2 (size-2 groups are ≤ 3 records at any step): the
  definition's "size 2" is `min_cluster_size`, not the groups' size.
- Band and prompt splits are in `r8.json`; from 64 no band keeps less than 0.76.

**Does not.** Use the real P > 0 partitions: unit 1 stored their group count `k`, not labels, so
P = 0's partition stands in. They agree in the median (`k` ratio 1.0) but not everywhere (p10 0.29,
p90 1.4, over 23,328 condition-cells), so `Jc` is a proxy **of unmeasured error**: `Jc` rises as a
partition gets finer (≈ 0.19 → 0.40 in the reviewer's toy, finding 2) and 64 hangs on one record,
so the exact check (unit 1's P > 0 passes re-run with labels kept) can flip the rule's letter. Ask whether `J0` (a placed 10th percentile) is the right
yardstick. Test anything. One seed, 7 passages.

Re-run: `data/p10/reread_r8_2026-10-06/run_r8.sh [--steps …]` from the worktree root. Tests:
`tests/test_p10_r8_floor.py` (13).

*The exact check has run (§1.25): the proxy was accurate, and the drop holds at 128 and 1000.*

### 1.25 R8x: the exact check — **the drop at 64–1000 holds by the rule's letter: with each moved passage's own partition in place of R8's proxy, the chance-aware set (c3x) keeps 0.876 of c3 at step 128 and 0.893 at 1000, below the placed 0.90; 64, 256 and 512 clear it (0.927, 0.911, 0.921), and from 2000 it keeps 0.93–0.98. The proxy was close: its chance level is exact in the median, within ±0.03–0.05 (p10–p90), and c3c and c3x differ on ≤ 11 records per step. So R8's picture stands: a drop at 64–1000 that is there but whose size 7 prompts do not pin (point 7–12 % at 128 and 1000, intervals 5–21 %; pooled over 64–1000 c3x keeps 0.905, clearing the bar), 2–7 % after, and at init "moves" is mostly chance**

**What.** `design-10.md` "R8x" (rule `ced45f5`, before any P > 0 label was written; Blocked 27
(c), user 2026-10-08). `tools/run/p10_r8x_exact.py` (`a0af469`): unit 1's EOD-join P > 0 passes re-run in
R0's environment (CPU, float32, 14 threads), each condition's partition of R0's kept offsets
(centred, size 2) recomputed and written; per condition a chance level `Jc` from 2000 random
same-size sets against **that** partition; c3x = c2 groups whose class is still `moves` when each
EOD condition's J ≥ max(`J0`, its `Jc`). R8's proxy recomputed beside (c3c). 972 forward passes (6 EOD conditions for
the three passages that are also preamble sources, 9 for the other four; × 18 steps), ~50 min
CPU. *Corrected after `/challenge-pr` on #163 (finding 5): this line first said 1,080.*

**Inputs.** R0's labels and unit 1 records (`data/p10/reread_r0_2026-10-05/`); output
`data/p10/reread_r8x_2026-10-08/` (`labels/`: each condition's partition per (step, passage,
layer); `rows/`; `r8x.json`, md5 `8c8bdb91`, whose `meta` names the input paths, R0's labels hash
`0d6da9cb` (R8's) and the run's git `a0af469` and environment; `run.log`). **First checks, all passed:** R8's three
(P = 0 groups are R0's c2a, recomputed c2 and c3 are R0's); every condition's ids and start are
unit 1's; at all 126 (step, passage) × 24 layers × 6–9 conditions the recomputed partition
reproduces unit 1's stored group count and every stored best Jaccard exactly (none refused);
step 64 opened and checked populated before the batch. The recomputed c3c equals R8's at every
step.

**c3's group-layer records, pooled over prompts and L1–24** (bootstrap: 7 prompts, 5,000 draws):

| step | c3 | c3c (R8) | c3x | keep c3c | **keep c3x** | c3x 95 % interval | c3c only / c3x only | `Jc` error p10 / p90 | \|error\| > 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 62 | 31 | 29 | 0.50 | 0.47 | 0.34–0.64 | 2 / 0 | −0.05 / +0.06 | 0.23 |
| 2 / 4 / 8 / 16 | 61 / 63 / 55 / 96 | 30 / 31 / 29 / 58 | 28 / 33 / 29 / 58 | 0.49–0.60 | 0.46–0.60 | 0.27–0.74 | ≤ 2 / ≤ 3 | | |
| 32 | 67 | 56 | 56 | 0.84 | 0.84 | 0.73–0.94 | 1 / 1 | | |
| 64 | 248 | 223 | 230 | 0.90 | **0.93** | 0.86–0.97 | 2 / 9 | −0.03 / +0.05 | 0.16 |
| **128** | 307 | 268 | 269 | 0.87 | **0.876** | 0.79–0.93 | 2 / 3 | −0.05 / +0.03 | 0.14 |
| 256 | 326 | 288 | 297 | 0.88 | **0.91** | 0.83–0.96 | 1 / 10 | −0.03 / +0.03 | 0.09 |
| 512 | 598 | 545 | 551 | 0.91 | **0.92** | 0.81–0.98 | 2 / 8 | −0.03 / +0.03 | 0.07 |
| **1000** | 670 | 595 | 598 | 0.89 | **0.893** | 0.80–0.95 | 2 / 5 | −0.03 / +0.03 | 0.05 |
| 2000 | 817 | 757 | 757 | 0.93 | 0.93 | 0.87–0.96 | 1 / 1 | −0.02 / +0.02 | 0.04 |
| 4000–54000 | 849–967 | 812–931 | 810–934 | 0.95–0.96 | 0.95–0.97 | 0.90–0.99 | ≤ 2 / ≤ 3 | ±0.02 | 0.03–0.04 |
| 143000 | 810 | 789 | 792 | 0.97 | 0.98 | 0.96–0.99 | 0 / 3 | −0.03 / +0.02 | 0.04 |

`Jc` error = exact − proxy per (c3 record, condition); median 0.000 at every step. The
condition's group count over P = 0's, per (c2 record, condition): median 1.0, p10 0.47, p90 1.25
(103,224; R8's 0.29 / 1.4 counted condition cells, not records).

**Reading.**
- **By the rule, the drop holds** (128 and 1000 below 0.90), so Blocked 27's recommendation applies:
  **(a), c3x as the definition's column, every re-read row run again on it.** The intervals still
  straddle 0.90 at every step 64–1000, so this is the rule's letter, not a resolved share. What is
  resolved is that a drop exists there (every upper bound below 1); its size is not (128: 7–21 %,
  1000: 5–20 %). *Corrected after `/challenge-pr` on #163 (finding 1): the first version called
  the size resolved.*
- **The rule is fragile, and it chooses the headline, not the work** (*after `/challenge-pr` on
  #163, finding 2*). It fires if any one of five steps falls below 0.90. Pooled over 64–1000 c3x
  keeps 1945 / 2149 = 0.905, which clears it. Resampling prompts, the rule fires in 87 % of draws
  on these data, and by the reviewer's simulation about half the time at a true 0.93 everywhere.
  (a) and (b) both put c3x on every re-read row; the verdict decides only which column leads.
- **Step 128's miss is one passage; step 1000's is not** (*corrected after `/challenge-pr` on #163,
  finding 3*; the first version read "not one passage"). Without `camus_letranger`, 128 keeps
  251 / 277 = 0.906 and clears the bar; 1000 keeps 584 / 654 = 0.893, five records short, so
  without that passage the verdict rests on 1000 alone. 1000's drop is L17–24 (0.83), 128's L1–16
  (0.85).
- **The proxy was not the problem.** Exact and proxy chance levels agree in the median at every
  step, and the records they classify differently are ≤ 11 per step, mostly c3x keeping what the
  proxy dropped (64, 256, 512). R8's "proxy of unmeasured error" is now measured: small.
- At init, "moves" is mostly chance: step 0's 62 records keep 29.

**Does not.** Re-examine `J0` (a placed 10th percentile); read the `\n\n` join; re-read any row on
c3x (that is (a)); test anything. Pin the whole P > 0 partition: the reproduction check fixes each
condition's group count and every best Jaccard of a P = 0 group, so a group touching no P = 0
group could in principle differ, and the chance draws use it (*`/challenge-pr` on #163, finding
7*; at the same code, inputs and environment as R0, not expected). One seed, 7 passages.

Re-run: `data/p10/reread_r8x_2026-10-08/run_r8x.sh [--steps …]` from the worktree root (CPU;
resumes; `summary` at the end). Tests: `tests/test_p10_r8x_exact.py` (9).

---

## 2. The blocker that had to be cleared first

**The 410m sweep had no density partition at all.** `hdbscan_labels.json` is
`{}` in **152 of 152** directories and `clustering.json` beside it reads
`"nesting_summary": "HDBSCAN not available"`. The outage `PROJECT.md` §3.41
records was never only `CLAIM-C`'s arms, and `docs/AXES.md` listed the artifact
as present. F0, F5 and F11-A4 all read it.

**A fourth row was hit and nobody noticed — found 2026-09-20, `handoff-10.md`
§0.2.** `pair_agreement` (`pair_hdbscan_agreement`, the project's **only**
semantic instrument: mutual-NN pairs tagged against the embedding Gram) is
computed only when `"labels" in hdb_data`. Through the outage that branch was
never taken, and the `else` wrote a **well-formed record of zeros and nulls**
into all 152 directories rather than failing. A silent zero record looks exactly
like a real one, which is why it survived every check. **The pilot sweep's copy
is populated — 6 066 of 6 075 layer-records — and had never been reported in any
markdown file in this repository.** Same bug class as `docs/AXES.md` listing an
absent artifact as present, and as standing rule 4.

`tools/run/backfill_hdbscan.py` re-derives it from `activations.npz` — no
forward pass, **152 directories in 69 s**. **It does not re-run the analysis**,
by design (item 3 below), so the WDS sweep still carries no semantic record. Mean 47.1 clusters/layer, mean noise
fraction 0.389, in line with the pilot's 45–69 and finding 4's 50–55.

Three properties make it usable rather than merely fast:

1. **It guards on the TOOLCHAIN, not the interpreter path** — the opposite of
   every other runner here, and correct for this one. `clustering.py` records,
   measured, that HDBSCAN's output is a property of the install: conda `mets`
   (py3.10.20, hdbscan 0.8.41, sklearn 1.7.2) reproduces the pilot sweep
   exactly and `.venv` does not.
2. **It re-verifies on every invocation** — `--verify-pilot` replays N pilot
   directories that *do* carry labels and refuses to write unless they come
   back bit-identical. This run: 3 directories × 25 layers, clean.
3. **It writes a separate file and touches nothing.** Filling the canonical one
   would leave `clustering.json` still saying `null` beside it — the shape of
   inconsistency `b55375e` had to un-write. `read_labels` owns the precedence.

---

## 3. The measurement that was not on the ladder

**The HDBSCAN partition is not reproducible run to run** — *confirmed
2026-09-25 (§1.11): up to step 1000 the activations differ by ≤ 2.5e-7, and
24–60 of each step's 200 layer-records still differ. Today's pipeline repeats
itself bit for bit only because it is deterministic on one machine.* Two independent
Phase-1 sweeps cover the same checkpoints and prompts; their `tokens.txt` are
identical in all 104 overlapping directories and their activations differ by at
most **7.9e-05**. Over **2 600 layer-pairs**:

| | value |
|---|---|
| label vectors identical | **83.3 %** |
| ARI median / mean | 1.0 / 0.933 |
| **ARI 5th percentile / minimum** | **0.347 / 0.166** |
| ARI noise-dropped, p05 / min | 0.585 / 0.327 |
| cluster-count \|Δ\| mean / **max** | 0.58 / **20** |

**Superseded 2026-09-26: this floor is the float32-distance defect.** Refit on
float64 distances from the same activations, the same 2 600 pairs agree at ARI
p5 1.000 (99.1 % identical), and Stage 0's stored `repeated_tokens` labels are
rounding (ARI to their float64 refit, mean 0.23). The table above is the stored,
float32-derived labels. Numbers and re-run: `p1d_cluster_ensemble/status-1d.md`
"Float64 distances, and Phase 10 §3's floor re-run on them".

**A measurement-reproducibility floor, not a null** — no hypothesis, no
p-value, by design. No null this project has built accounts for it:
`notes-10.md` §4.4's size-profile null is about ARI's variance under random
*labelling*, a different quantity from its variance under *re-measurement*.

**What it bears on.** `CLAIM-C` reads `cluster_count` and `cluster_membership`
from HDBSCAN and nothing else, and §3.41 scored them 2/8 and 5/8 — the two
weakest of six. **It does NOT follow that the gate's result is noise**: those
arms differ by model and training, not by a re-run, and this measures only the
re-run. What follows is that the comparison has never been made.

### 3.1 Both headline rows were re-run against it, and both hold

*A0's "hold" here is two sweeps sharing one confound: both keep position 0 and the massive token
in the unclustered population, which carry the late flip (§1.17).*

| | WDS (backfilled) | pilot (native) |
|---|---|---|
| directories / units, A0 | 152 / 3 646 | 243 / 5 593 |
| **A0** corrected gap, sweep mean | 0.037 | 0.100 |
| **F0** nucleus position | **0.3164** | **0.3157** |
| F0 ordinary / restricted null mean | 0.2077 / 0.2311 | 0.2053 / 0.2220 |
| F0 median p, ordinary / restricted | 1.00 / 0.9995 | 1.00 / 0.9995 |
| F0 merged E, ordinary / restricted | 0.536 / 0.556 | 0.550 / 0.569 |

A0's sweep means differ because the pilot's checkpoint grid is weighted toward
late training, where §1.1 already showed the residual lives.

A0's per-checkpoint corrected gap: 0.004/0.004 at step 0, −0.072/−0.055 at 256,
−0.164/−0.121 at 512, **0.172/0.230 at 143000**. F0 agrees to **three decimal
places** on the statistic itself — and the pilot finds **fewer** clusters per
layer (42.8 against 47.1), so the agreement is not an artifact of a matching
size profile.

**The floor still binds any PER-LAYER claim.** What these checks establish is
that a statistic aggregated over thousands of units is far less exposed to it.

---

## 4. Ladder status

| row | state |
|---|---|
| **F0** anchor test | **RUN**, both sweeps. Fails in the predicted direction |
| **F0b** slope test | not run — needs the `beta_eff` producer (`AXES.md` §4) |
| **F1** transport | **RUN** |
| **F2** verify J-lens artifacts | not run — needs HF access from this machine |
| **F3** fit a J-lens | blocked on F2 |
| **F4** per-layer functional partition + ARI | blocked on F3 |
| **F5** four-signature concordance | **the phase's central test.** Blocked on F4 |
| **F6** `turnover_decomposition` rebuilt | not run. No new experiment — a groupby |
| **F7** centroid substitution | blocked on F4 |
| **F8** operator decomposition of `J_l` | blocked on F3 |
| **F9** MLP object collapse | not run; needs a forward pass |
| **F10** neuron-basis collapse | blocked on F4 |
| **F11** attention audit | **A0 RUN.** A1–A8 not run; A0 gated them and has now cleared |
| **F12** `Z_beta,i` per token | **RUN** |
| **F13** the centre scan (Rényi + strong Rényi, swept in `δ`) | **new 2026-09-20.** Free, unblocked, and the row F0 was a proxy for |
| **F14** observed count vs Lemma C.1 | new. Free; needs F13 |
| **F15** the coverage curve (the paper's Fig. 3, and its own open problem) | new. Free; needs F13 |
| **F16** intrinsic dimension per layer, as the `d_eff` candidate | new. Free |
| **F17** the graded block-shuffle null | new. Free |
| **F18** `V`-spectrum atlas against Table 1 | new. Free |
| **F19** depth-axis merge intervals and splits | new. Free; needs the particle table |
| **F20** frozen-centre intervention (Thm 5.2) | new. **Forward pass**; needs F13 |

---

## 5. What a next session should do, in order

1. **`CLAIM-C`'s two HDBSCAN metrics against the reproducibility floor** (§3).
   **The only open item here that bears on a *registered* prediction**, which
   is why it is first. `tools/run/p10_partition_stability.py` supplies one side
   — the amount a partition moves when the same measurement is re-run — and
   what is missing is the other: how far that gate's four arms actually differ
   on `cluster_count` and `cluster_membership`. §3.41 scored them 2/8 and 5/8,
   the two weakest of six. Free. **This ordering matches `docs/AXES.md` §7.**
2. **F11 rows A1–A8.** A0 gated them and A0 is done. `attention-10.md` §6 rates
   **A2** (the checkpoint axis) and **A4** (the population×population mass
   matrix) highest, and calls A4 a direct H-PARK vs H-CAT test. Free.
3. **F6, `turnover_decomposition`.** Validated on synthetic sweeps in 2026 and
   awaiting real data ever since; a rebuild against `core/particles.py`, not a
   lift (`archive/README.md` rule 2). No forward pass.
4. **A norm-matched random twin per checkpoint**, which is what would turn
   §1.4's step-0 baseline from an argument into a control. Needs forward
   passes, and it is the single thing that would most strengthen §1.5.
5. **Only then F2/F3**, the J-lens, which is what unblocks F4, F5, F7, F8, F10
   — the functional and causal columns, and the phase's central test.

### 5.1 REVISED ORDERING, 2026-09-20, after the papers were read

The list above was written when `2411.04990` was `[S]`. It is now `[R]`
(`lit-10.md` §11), and **one free row moved to the front.**

1. **F13, the centre scan.** Greedy sequential acceptance over token positions,
   **both rules**, **swept in `δ`**, per layer. It needs positions and a
   distance and **not the HDBSCAN partition at all** — so it is the one row in
   this phase immune to §3's reproducibility floor, and it is the test F0 was
   standing in for. Free.
2. **F14 beside it.** `E[#strong centres] = E_{x∼μ}[1/μ(B_δ(x))]` estimated on
   the same cloud, against the observed count. The phase's
   packing-versus-content discriminant **with an exact i.i.d. null and no free
   parameter but `δ`** (`math-10.md` §7.2) — and the `δ` where the two meet
   reads back `c²/β`, turning `PROJECT.md` §3.40's undecided convention into a
   measurement. Free.
3. **`CLAIM-C`'s two HDBSCAN metrics against the reproducibility floor** —
   item 1 above, unchanged, and still the only open item bearing on a
   *registered* prediction. Free.
4. **F11 rows A1–A8, plus the new A9** (are the strong centres the sinks?
   `attention-10.md` §6). A2 and A4 still carry the most information per unit of
   work. Free.
5. **F16 and F17**, because both are cheap and both improve every row after
   them: the manifold `d_eff` (`math-10.md` §7.3), and a graded calibrated null
   to replace the binary controls (`lit-10.md` §12.2).
6. Then F6, the norm-matched random twin, and only then F2/F3.

**F20 is the phase's natural known-answer dry run** — the only experiment here
whose predicted outcome is a theorem — and `claims/EXPERIMENTS.md` records that
two adjudicable gates never had one. It costs a forward pass and it waits on
F13.

**Before any of this becomes `design-10.md`:** `notes-10.md` §12 as amended.
`2411.04990` **has now been read**; what remains unread and could still change a
construction is `2303.06562` (ContraNorm, before any Phase 9 spreading arm) and
`2607.15495` (the J-lens paper itself) — `lit-10.md` §15.

---

## 6. What this phase has NOT established

- **Nothing is registered and nothing is adjudicated.** Tier 1 throughout.
- **H-PARK is not confirmed.** §1.5 reads parked rather than pinned in one
  window, on an interpretation of `Z`, with the density confound argued rather
  than controlled.
- **H-CAT is not refuted.** The functional and causal columns are untouched.
- **The attention flip is not refuted** — it is re-measured on a different
  model from the one that produced the published number, and found to be mostly
  structural there.
- **F0 is not an adjudication of the parking account.** It is a tier-1
  exploratory result that came back against the prediction, on a statistic
  whose wording was never frozen — **and, since 2026-09-20, on a statistic that
  is a proxy for the account's object rather than an instance of it**
  (`lit-10.md` §11.4). F13 is the row that would adjudicate, and it has not run.
- **Nothing in the five papers read on 2026-09-20 is a theorem about Pythia.**
  `2411.04990` ties weights across layers, omits the MLP, and proves its
  meta-stability results at `V = I`, `Q = K = I`, `d = 2`. The correspondence is
  a hypothesis to test on a real model; that is the point of testing it.

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-24 when the card was written.

- 2026-09-20 · 410m's step 0 and step 1 are the same weights, so every count or pooled statistic over checkpoints here counts one checkpoint twice: the unit counts (3 648 = 19 × 8 × 24, 11 400 = 3 × 19 × 8 × 25, A0's 3 646), sweep means and merged E. Min–max ranges such as the "0–16" rows are unchanged · §3.51.3
- 2026-09-23 · the 12 new v2 prompts are held out on 410m, not pooled into Stages 1–5, so the header's "the enlarged battery can carry a registered prediction" holds only for the 12; they are partly seen already via `CLAIM-C` on 1.4b and gpt2-large · `docs/PHASE_REVIEW.md` "Decisions", `p10_cluster_function/handoff-10.md` §0.4
- 2026-09-24 · §0's "`METS_REPO` defaults to the MAIN tree" stopped being true: every runner now defaults to its own checkout (fixed in place) · `docs/PHASE_REVIEW.md` "Parked"
- 2026-09-25 · §3's floor is almost all one prompt, `repeated_tokens`; the other 7 v1 prompts' minimum ARI is far higher. The per-layer caution in §3.1 binds `repeated_tokens`, and much less the rest. Per-prompt table and producer (`tools/run/p1d_drift_checks.py --baseline`) · `p1d_cluster_ensemble/status-1d.md` "Float-noise drift"
- 2026-09-25 · §3's `repeated_tokens` drift is float32 cancellation in the cosine-distance step (`clustering.py`), not HDBSCAN's sensitivity: refit on float64 from the same activations, the sweeps agree at ARI 1 in 8 of 9 records (0.982), and the stored partitions at steps 32 / 512 share ARI 0.07–0.66 with the float64 ones. Every row that reads `repeated_tokens`' stored labels reads rounding; `wiki_paragraph`'s are exact; the other six prompts unchecked · `p1d_cluster_ensemble/status-1d.md` "Matched k on `repeated_tokens`, and the float32 defect"
- 2026-09-26 · `math-10.md` §5.4's inversion: its "scaled" β = 0.50 is the theory's β ÷ 8 (the fit already sees the model's `1/√d_h`; `status-1c.md` "Corrections received"), and "Lemma 5.1 needs c > 1" is sufficient only for β > 1; the bound is `c_min(β)`, 0.809 at β = 0.5 and 0.978 at β = 4.0. The table's reading survives (no row clears at 0.5, margin −0.016 at d_eff 22; d_eff ≥ 5 clears at 4.0) · `tools/math_checks/lemma51_c_bound.py`, `p1d_cluster_ensemble/status-1d.md` "Attention communities against three nulls"
- 2026-09-26 · §3's floor is the float32-distance defect, not HDBSCAN: on float64 distances ARI p5 is 1.000 over the same 2 600 pairs. Stage 0's stored `repeated_tokens` labels are rounding (mean ARI 0.23 to float64), so every reader here that pools that prompt's stored labels carries one noise prompt in eight; not re-run · `p1d_cluster_ensemble/status-1d.md` "Float64 distances, and Phase 10 §3's floor re-run on them"
- 2026-10-01 · the partition every row here reads, hdbscan `min_cluster_size=2`, orders tied mutual-reachability edges by processing order, and ties are the rule: on deduped v1 tokens 42 % of step143000's groups (53 % of step 0's) are never a connected component of the graph at any distance, two thirds of them pairs; median ARI to tie-merged (level-set) HDBSCAN 0.82, p10 −0.06. Deterministic for fixed input order, so §3's floor stands; group counts and group-level rows carry the artefacts; not re-run · `p1d_cluster_ensemble/status-1d.md` "Admission"
