<!-- trackT_timaeus/status-T.md -->
# Track T — STATUS

<!-- phase-card -->
## Card

- **Question:** Do Timaeus's released checkpoints (Dodecahedron, a fourth training schedule) and the local learning coefficient give C2's energy break a second schedule and a degeneracy curve to sit beside, and which released model and settings make the LLC a calibrated instrument here?
- **Inputs:** T1 ran nothing: it read Hugging Face metadata (cards, config files, branch and file listings, LFS hashes) and papers through a summarising fetch, listed in `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"; T2 unit 1 ran `dodecahedron-32m` and `-67m` at steps 1 and 9536, 3 re-inits each, and `pythia-410m` steps 0–4000, the 7 v1 prompts (`976b62d677bc`), code `4fd84525dd37`, listed in `trackT_timaeus/status-T.md` "T2 unit 1"; T5 U0 read weights only (`timaeus/gpt-2-small-init-seed-5` `4f072cfb2d`, no forward pass), listed in `trackT_timaeus/status-T.md` "T5 U0"; T5 U1 ran `pythia-70m`, `-160m`, `-410m` step 0, `dodecahedron-32m` step 1, `timaeus/gpt-2-small-init-seed-5` and 680 random models (32 cells × 20 seeds, 2 controls × 20), the 7 v1 prompts (`976b62d677bc`), code `f6b205d00f5c`, listed in `trackT_timaeus/status-T.md` "T5 U1"; T5 U1b ran 360 random models (Llama at 32m's shape without biases, base and 8 swaps × 2 inits × 20 seeds), the 7 v1 prompts (`976b62d677bc`), code `d205f0b180e3`, listed in `trackT_timaeus/status-T.md` "T5 U1b"; T5 U2 ran `dodecahedron-32m`, `-67m` at their 53 steps, `pythia-70m`, `-160m`, `-410m` at C2's 18 and `timaeus/gpt-2-small` at its init and 19 checkpoints, the 7 v1 prompts (`976b62d677bc`), code `3d425b32db90`, listed in `trackT_timaeus/status-T.md` "T5 U2"; T5 U2b ran `dodecahedron-32m` at steps 491, 671, 917 and 9536 with its 3 re-inits, the 7 v1 prompts (`976b62d677bc`), code `b01ad9e3e4d8`, listed in `trackT_timaeus/status-T.md` "T5 U2b"; T5 U3's first read ran `dodecahedron-32m` at the 16 grid steps 917–9536 with its 3 re-inits, the 7 v1 prompts (`976b62d677bc`), code `b01ad9e3e4d8`, reader `t5e.py`, listed in `trackT_timaeus/status-T.md` "T5 U3, first read"; T5 U3's split ran `dodecahedron-32m` at 9 steps (1, 917–9536) with a planted o_proj bias at 9536 and `pythia-410m` at C2's 8 split steps (0–143000), the 7 v1 prompts (`976b62d677bc`), code `ef81a441e7dc`, listed in `trackT_timaeus/status-T.md` "T5 U3's split, the batch"; T5 U4s ran 3,160 random models (Llama at 32m's shape without biases, 79 cells: the 64 subsets of U1b's six swaps, `untied`, the `gelu` split, the reverse ablation, × 2 inits × 20 seeds), the 7 v1 prompts (`976b62d677bc`), code `2d7d55d4e4f9`, listed in `trackT_timaeus/status-T.md` "T5 U4s"; T5 U4a ran released `dodecahedron-32m` at 1, 917, 4369, 9536 and `pythia-410m` seeds 0–9 at 4000 and 143000 through U2's reader, built FineWeb `sample-10BT`'s stream (revision `9bb295dd`, files 000–003, main `a7c1b38b`), and timed 60 updates of A0's configuration (discarded), code `b9e5724ac9d2`, listed in `trackT_timaeus/status-T.md` "T5 U4a, before A0"; T5 U4a's A0 trained the Llama-32m (init F, seed 0, no biases) on that stream's main part to 4369 at 524,288 tokens a step and read its 49 grid checkpoints on the 7 v1 prompts (`976b62d677bc`), code `66af413c31fe`, listed in `trackT_timaeus/status-T.md` "T5 U4a: Blocked 133 decided; A0"
- **Results:**
  - Opened 2026-10-29 (Blocked 117 (a), the user's: T1 and T2 first) with T1's questions, grades and a one-unit budget fixed before reading — `trackT_timaeus/design-T1.md`
  - T1 closed at its budget: Dodecahedron's 40B tokens fits a global batch of 8 × 524k (per GPU), not 524k, and under it T2's token window overlaps both its step and its LR-dose windows, so T2 separates LR dose from step either way and tests tokens only if the batch is 524k; `-v2` is a rename and `-archived` the same weights, but `-pilot-dense` is a different run; 2402.02364's LLC settings are read, its model is most likely `tetrahedron-3m-og`; no released model deeper than 2 layers has a published whole-model LLC curve (`L4` has one head's, 2507.21449); `gpt-2-small` is a GPT-NeoX of Pythia-160m's shape; the Pythia negative stands for the two papers read — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
  - T2 unit 1 (design frozen on D1–D4 (a)): the producer reproduces A5's seed-0 records on `pythia-410m` exactly and its planted effect and null pass, and Dodecahedron loads as trained, its hidden states bit-identical to transformers 5.2's; but every centred arm fails its gate on both models (plant A turns 0.21–0.51 of the free units), and the uncentred count departs as far (plant A 0.04–0.07 against 0.79–0.98 on 70m, 160m, GPT-2); beside, not shown to be the cause, the unit cloud is nearly one direction (middle-state mean cosine 0.82–0.84 at init, 0.90–0.91 at step 9536, against 410m's 0.47 and 0.17); so neither count is the one dated on Pythia, and the design goes back to the user (Blocked 120) — `trackT_timaeus/status-T.md` "T2 unit 1"
  - T2 closed 2026-10-30 at unit 1, before any scored record (Blocked 121 (i), the user's, as recommended): a count fit to Llama was proposed (Blocked 120 (b)) and not frozen, since at equal token norms the raw-centred count equals the centred one exactly (L4) and differs from it only through the token-norm spread (L2), while centring leaves the rows a mean squared norm of about 0.09 at cosine 0.9 (L3); whether it would pass was not computed; no Dodecahedron date, and C2's F1 gets no fourth schedule from it — `trackT_timaeus/status-T.md` "T2 on one page (closed 2026-10-30)"
  - T5 opened 2026-10-30 (Blocked 122 (a)); U0, weights only: Timaeus's `gpt-2-small` starts from 0.02 on every weight with zero biases, so it is Pythia-160m's architecture under Dodecahedron's flat init, a released run whose architecture matches a model that passed and whose init does not (its data and the rest of its schedule unknown) — `trackT_timaeus/status-T.md` "T5 U0"
  - T5 U1 (gate passes): the init alone does not concentrate the cloud: `pythia-70m` step 0 (mean cosine over states 1–5, 0.28) and `gpt-2-small`'s flat init (0.23) both sit at or below the falsifying 0.30; on random weights the flat init raises the middle-state mean cosine in Llama by 0.18–0.37 at every shape (at matched depth 0.20–0.42, smallest at 70m's dose, as H1 predicts), and in GPT-NeoX not at all (−0.06 to 0.02, lower than Pythia's init at 3 of 4 shapes), and Pythia's init takes a Llama at 32m's shape 0.84 of the way to a GPT-NeoX's, so H1 at init reads "partly": an interaction of init and architecture, the architecture's part not identified (its factor carries the tokenizer) — `trackT_timaeus/status-T.md` "T5 U1"
  - T5 U1b (Blocked 124 (a), one unit beyond the budget; gate passes): no single part of the Llama carries the init's effect at 32m's shape: swapping its SwiGLU MLP for GeLU at 4d removes 0.16 of the 0.36 (about half on logit or atanh scales), and LayerNorm, the parallel residual, 0.25 rotary, 6 KV heads and GPT-NeoX's tokenizer each remove under 0.01 (the tokenizer is not R2's architecture factor); all six at once remove 0.43, so much of the effect appears only with parts together (its size depends on the scale of ρ), which set not tested — `trackT_timaeus/status-T.md` "T5 U1b"
  - T5 U2 (Blocked 125 (a); gate passes): Dodecahedron's cloud is not concentrated throughout training: on both models the middle-state mean cosine rises to 1.00 by step 30–47, falls to 0.10–0.11 at 671–917 (lower than any Pythia minimum read, 0.17–0.21) and rises again to 0.90 by 9536, so T2 read it only at its two peaks; Pythia has the same rise, fall and rise, with lower ends; `gpt-2-small` spreads (0.34 → 0.17) and stays spread; position 0 decides none of it; so the trained cloud's 0.90 is not the init's concentration persisting, and arises after step ~1253 — `trackT_timaeus/status-T.md` "T5 U2"
  - T5 U2b (Blocked 126 (a), a second extension; known answer exact): in the trough (steps 491–917, ρ_mid 0.11–0.12) T2's B-matched centred arm passes T2's full gate at every step (plant A 0.00, re-init 1.5 %; its B ≥ 0.4 holds by its derivation) and the uncentred count sees plant A as on Pythia (0.93–1.00, against 0.04 at 9536), so the counts fail where the cloud is concentrated (co-occurrence: the trough is also in the LR warmup, at higher loss and a small position-0 norm); post hoc, at fixed τ 1e-3–1.2e-2 plant A is ≤ 0.10 in the trough and 0.32–0.58 at 9536; the other three centred arms still fail, on the re-inits (fixed by the init) and on plant A (0.12–0.45) — `trackT_timaeus/status-T.md` "T5 U2b"
  - T5 U3's first read (Blocked 130 (a); both known answers exact): along the late rise every count stops passing with the rise, none before it: the uncentred count leaves its Pythia behaviour at 1713 (plant A 0.89 → 0.72; ρ_mid 0.32), the fixed-τ centred arm (τ 3e-3) leaks plant A from 2002 (0.12; ρ 0.43), and T2's B-matched arm passes T2's full gate to 3738 (ρ 0.76; at 3198 and 3738 only with its τ search widened past the code's 1e-2, post hoc) and every arm fails from 4369 (ρ 0.81); at 2340 and 2735 the uncentred count sees plant B at no dose, so no arm is defined; ρ, position 0's norm, the loss, steps since warmup and each count's operating point (base rate, matched τ) move together there, so which makes a count stop is not told apart — `trackT_timaeus/status-T.md` "T5 U3, first read"
  - T5 U3's split refused by its planted-bias gate (2026-11-02; the sum and known-answer gates pass at `-32m` 9536): a constant o_proj bias at half the median middle-state norm in every block (as large as block 1's whole stream) collapses the cloud at state 1 (mean cosine 1.00), and the MLPs then write a new constant, near-orthogonal to the bias and much larger, which takes 95 % of the planted component, so the bias path holds 0.125 against the bar 0.9; the split is exact, the gate's plant moves the model; no other step run; Blocked 131 — `trackT_timaeus/status-T.md` "T5 U3's split"
  - T5 U3's split, the batch (Blocked 131 (a), the user's: the planted gate retired on the record before output; sum and known-answer gates pass, 9536 reproduced exactly): read as direct writes, through `-32m`'s rise attention's ordinary keys (share 0.43–0.60) and the MLP write's pooled mean (0.25–0.44) carry the shared component about equally, the sink grows 0.006 → 0.13 with position 0's norm; on shares no path changes abruptly where a count stops, while on sizes the MLP mean carries most of the write's growth to 4369, fastest over 1253 → 1713; Dodecahedron's step 1 is written by attention (0.87), 410m's step 0 by the MLP mean (0.58); 410m's biases write 0.31 at 143000; post hoc, neither MLP writes one vector to every token (0.06–0.15 of `-32m`'s MLP write in its mean), while `-32m`'s attention write grows more uniform (0.14 → 0.42); what raises ρ is not measured; no H2 reading — `trackT_timaeus/status-T.md` "T5 U3's split, the batch"
  - T5 U4s, the screen at init (D6 (b), the user's; gate passes, U1b reproduced exactly): at 32m's shape the flat init's concentrating effect needs GPT-NeoX's parallel residual and its ungated MLP together to go (the two remove 1.15 of it, every other pair's interaction ≤ 0.08 of it); put back into GPT-NeoX, SwiGLU alone brings all of it back (E 1.15 of the Llama's) and the sequential residual about half (0.53); in the MLP, ungating removes about half (not told apart from the larger write it brings), GeLU for SiLU does not, and the width amplifies it; the norm, rotary fraction, KV heads and tokenizer do nothing in any combination; `gelu` with 6 KV heads also carries, on the line (0.52, 82 % of a seed bootstrap); at init only — `trackT_timaeus/status-T.md` "T5 U4s"
  - T5 U4a's preparation (no arm trained): the reader's known answer through the new checkpoint format is exact, the training loop equals a hand-written AdamW in float64 (in float32 rounding exceeds the frozen bar), and the throughput fits all five arms and every continuation in the cap; but over ten `pythia-410m` seeds the end-of-training spread of the middle-state mean cosine is wider than the rule's floor (two seeds end well below the rest), so the arms waited on Blocked 133, decided 2026-11-03 (the floor raised to the largest seed-pair difference, 0.2, and a third seed of A0 added) — `trackT_timaeus/status-T.md` "T5 U4a, before A0", "T5 U4a: Blocked 133 decided; A0"
  - T5 U4a's A0, the positive control: Dodecahedron's recipe retrained on FineWeb at 524k tokens a step reproduces its fall (mean cosine 0.14 at 671, against 0.11) and its late level (0.79 against 0.79, the mean of 3738 and 4369), so the one-change arms are read against it; the rise starts earlier than in the released run (0.31 against 0.13 at 1253, a description after output, one seed) — `trackT_timaeus/status-T.md` "T5 U4a: Blocked 133 decided; A0"
  - T5 U4b's replicates: two more seeds of A0 land within 0.046 of it at the late level (0.781, 0.748 against 0.794), so the noise bar stays at its floor 0.2, under half A0's rise (0.328), and the one-change arms are read with all four labels; the earlier rise is not A0's seed (all three 0.13–0.23 above the released run at 1253–2002, spread 0.04–0.075; after output) — `trackT_timaeus/status-T.md` "T5 U4b"
- **Superseded / wrong:**
  - The ROADMAP's T2 windows (from #282) read the card's per-GPU batch as the global one and used seed 0's break bracket, not A5's nine-seed (128, 512]; re-derived in T1 — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
- **Registry:** none, because Track T is tier 1
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - Which seed of `tetrahedron-3m` the paper's curve is from, and why `-og`'s checkpoint spacing differs from the `icl` sweep's — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
  - Why the Pythia counts fail on Dodecahedron (H1 the init, H2 no biases): T5, opened 2026-10-30, design frozen 2026-10-30 (Blocked 123, D1–D3 (a)); U1 done (H1 at init partly); U1b done (no single part); U2 done (the cloud spreads in the middle of training and re-concentrates late); U2b done (the counts pass, on the trained side, where the cloud is spread); U3 revised toward the late rise (Blocked 130 (a)); U3's first read done (every count stops with the rise, 1713–4369; `c2_seeds`' τ search bound, Parked 3); U3 done 2026-11-02 (the split read as direct writes; its planted gate retired, Blocked 131 (a)); U4's design frozen 2026-11-03 (Blocked 132: Dodecahedron's recipe at 524k a step to 4369, five arms with A3, A0 as the positive control, the cap 4 GPU-days; U4s, the screen at init, first); U4s done 2026-11-03 (at init the effect needs the parallel residual and the ungated MLP together to go; SwiGLU brings it all back, the sequential residual half); U4a's preparation done 2026-11-03: the 410m noise floor refused at 0.1; Blocked 133 decided (floor 0.2, a second replicate seed A0″, the float64 loop gate); U4a done 2026-11-04 (A0 passes its positive control, L 0.794); U4b started 2026-11-04 (the replicates first, then the seed-noise gate before A1–A3); U4b's replicates done 2026-11-05 (readable: τ 0.046, bar 0.2 < R_A0 / 2 0.328; A1–A3 next) — `trackT_timaeus/status-T.md` "T5 U4b"
- **After Phase 10:**
  - The `L0`–`L6` attention-only depth series at constant LR with no warmup, as a schedule without the warmup confound *(forward pass, small models)*
  - `gpt-2-small` as a Pythia-160m-shaped run checkpointed every 100 steps *(forward pass; its data and schedule unread)*
- **Reviewed:** 2026-11-05 · body `4884f2f92f`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| T1. the inventory | **done 2026-10-29** (one unit, at budget) | `design-T1.md`; "T1, the inventory" below |
| T2. V and E_β on Dodecahedron | **closed 2026-10-30 at unit 1** (Blocked 121 (i)), before the batch: design frozen 2026-10-29 (Blocked 119), unit 1's centred arms fail their gate on both models (Blocked 120), the Llama-fit count proposed (Blocked 120 (b)) and not frozen | "T2 on one page" below; `design-T2.md` |
| T3. LLC positive control | candidate | `docs/ROADMAP.md` §7b |
| T4. LLC on C2's checkpoints | candidate, after T3 | `docs/ROADMAP.md` §7b |
| T5. why the Pythia counts fail on Dodecahedron (the cloud's geometry; H1 the init, H2 no biases) | **opened 2026-10-30** (Blocked 122 (a)); U0 done: design frozen 2026-10-30 (Blocked 123: D1–D3 all (a)), scan, `gpt-2-small`'s init read (flat 0.02, biases on); homes "T2 on one page"'s "why the counts fail" | `design-T5.md`; "T5 U0", "T5's proposal" below |
| T5 U1. step 0 and random weights (H1 at init) | **done 2026-10-30**: gate passes; R1a, R1b falsified; R2 partly (the init matters in Llama only) | "T5 U1" below |
| T5 U1b. the Llama's parts at init (added, Blocked 124 (a)) | **done 2026-10-30**: gate passes; no single part (`gelu` partly, 0.16 of 0.36; the rest none) | "T5 U1b" below |
| T5 U2. the trajectories | **done 2026-10-30** (Blocked 125 (a)): gate passes; both Dodecahedron models spread (Δ 0.73–0.74, trough 0.10–0.11 at 671–917) and re-concentrate to 0.90; `gpt-2-small` spreads (0.17) | "T5 U2" below |
| T5 U2b. T2's gates in the trough (added, Blocked 126 (a)) | **done 2026-11-01**: known answer exact; **passes in the trough**, on the B-matched arm alone (full gate at all 3 steps); the uncentred count as on Pythia (plant A 0.93–1.00) | "T5 U2b" below |
| T5 U3. where the shared direction is written | **done 2026-11-02.** Revised 2026-11-01 (Blocked 130 (a), docs only); first read: known answers exact; every count stops with the rise (uncentred 1713, fixed τ 3e-3 2002, T2's B-matched arm 4369, last pass 3738 with its τ search widened post hoc), none before it; the split's planted-bias gate refused, then retired on the record (Blocked 131 (a)); **the batch: gates pass; through the rise attention's ordinary keys and the MLP write's pooled mean write the shared component about equally, the sink grows 0.006 → 0.13; on shares no path changes abruptly where a count stops, on sizes the MLP mean grows fastest over 1253 → 1713; neither MLP writes one vector to every token (post hoc)** | "T5 U3's split, the batch", "T5 U3's split", "T5 U3, first read" and "T5 U3's revision" below; `design-T5.md` "U3 revised", "U3's split, the details", "The planted-bias gate retired on the record" |
| T5 U4. the causal arm (aimed at the rise) | **design frozen 2026-11-03** (Blocked 132, the user's: D4 (a) steps at 524k; D5 (b) A3, biases and P together, with the cap raised to 4 GPU-days; D6 (b) U4s, the screen at init, run first; D7 (a) U4a then U4b); U4s done; U4a's preparation done; Blocked 133 decided ((i) (d): bar's floor 0.2 and a second replicate seed A0″; (ii) float64 loop gate). **U4a done 2026-11-04: A0 passes its positive control (L 0.794). U4b started 2026-11-04: the first change (Parked 2–3, code `40289679fc36`). Replicates done 2026-11-05: A0′, A0″ to 4369 (L 0.781, 0.748); the seed-noise gate passes, readable (τ 0.046, bar 0.2 < 0.328), all four labels in use; next A1–A3** | "T5 U4b" and "T5 U4's design" below; `design-T5.md` "U4, the causal arm, designed" |
| T5 U4s. the screen at init (D6 (b)) | **done 2026-11-03**: gate passes (U1b reproduced exactly); minimal carrying sets `par`+`gelu` (1.15 of E0) and `gelu`+`mha` (0.52, on the line); reverse: SwiGLU brings back all of it, the sequential residual about half; within `gelu`, ungating (or the write's size) | "T5 U4s" below |
| T5 U4a. the loop, the shard, θ, the noise floor, A0 | **preparation done 2026-11-03; no arm run**: known answer exact (1.7e-9); the loop exact in float64 (1e-14), not at the frozen float32 1e-5 (rounding, 2.4e-4); θ 104.8k tokens/s (all five arms and continuations fit); **the 410m noise floor refuses** (2·SD 0.153 at 143000, bar 0.1; 0.021 at 4000). **Blocked 133 decided 2026-11-03, the user's: (i) (d), floor 0.2 and A0″; (ii) the float64 gate.** Floor at 0.2 placed from its own data and reproduced (not a pass that could fail; /challenge-pr on #310, finding 1). **Done 2026-11-04: A0 passes its positive control** (T 0.137 at 671, L 0.794 ≥ 0.60, R 0.657; Dodecahedron 0.111, 0.788); its rise starts earlier than Dodecahedron's (0.31 against 0.13 at 1253, post hoc; Parked 1 to U4b) | "T5 U4a: Blocked 133 decided; A0" and "T5 U4a, before A0" below; `design-T5.md` "U4a's details" |

## T5 U4b: the replicates first (A0′, A0″), the seed-noise gate before A1–A3 (2026-11-04)

The order proposed on #310 and taken by the user (this session's prompt): A0′ and A0″ to 4369, then the
design's last gate row, **bar < R_A0 / 2** (R_A0 0.657, so readable iff τ < 0.164), before A1–A3 start.
A0″'s continuation to 5970 waits for the continuations' stage (it enters no label; design "Throughput").

**The first change** (commit `6f01a3f`, before any replicate output; `t5h` code `66af413c31fe` →
**`40289679fc36`**; `t5c`, `t5`, `t2`, `c2_aniso` untouched, so the loop, stream, checkpoints and read are
the same bytes):

| item | change |
|---|---|
| Parked 2 on #310 (defect) | `t5h train` with `--until` ≤ the resume step prints `nothing_to_train` and returns 0 without a `segment` line (not a refusal, so the stage scripts stay resumable) |
| Parked 3 | `FLOOR_MAX`'s comment says *placed* |
| the resume across the change | `t5h.RESUMABLE` named `66af413c31fe` with what changed, so A0's 4369 state continues by name, not by `--allow-code-change`; each `segment` line now records `from_code`. Keyed on the old code alone, it let that state resume under any later code; since the second change (below) a crossing is a (from, to) pair |
| the reading | `trackT_timaeus/t5i.py` (outside `t5h`'s hash, as `t5g` left `t5b`'s): `replicates` reads L, T, R of A0, A0′, A0″, τ, bar, bar_I and readability from `t5h`'s rule functions; refuses unless A0's late gate and the replicates' start gates passed; beside (Parked 1): their ρ_mid at 1253, 1713, 2002 and its spread, the v1 loss at 4369, the codes read at |
| the batch | `scripts/t5u4b_replicates.sh`: per arm, train to 103, read, start gate, train to 4369, read; then `t5i replicates` (exit 3: not readable, back to the user) |

Tests: `test_trackT_t5i.py` (readable at small noise, unreadable at τ 0.17 and readable at 0.16, the
refusals), `test_trackT_t5h.py` (empty segment, the named crossing); `check.sh` 3254 passed; the loop's
smoke tests 4 passed. **Launched 2026-11-04** (≈ 6 h an arm at θ ≈ 105k).

**The replicates read: readable (2026-11-05).** Input: arms A0′ (init F, seed 1) and A0″ (F, seed 2),
no biases, Dodecahedron's recipe at 524,288 tokens a step on the shard `a7c1b38b…`, trained and read at
`t5h` `40289679fc36` throughout (A0's records at `66af413c31fe`; the reader's code, `t5c`/`t5`/`t2`, is
the same bytes at both); 7 v1 prompts `976b62d677bc`; `t5i` `74f766b6a53b` (re-read at `74cdc66d32f6`; below). Records
`<HDD_1TB>/mets_data/trackT/t5u4/` (`replicates.json`; `A0p/`, `A0pp/` with `gate_start.json`, `read/`
at the 49 grid steps, `ckpt/`). **The pause:** the user stopped the batch overnight after A0″'s update
2622; its state was at 2540, so the rerun (2026-11-05, `logs/u4b_replicates_resume.log`, exit 0, 9116 s
for 1829 updates) trained 2541–2622 again. `A0pp/train_log.jsonl` holds both, split by `segment` lines;
the 82 repeated losses agree to ≤ 8.8e-4 (mean 1.5e-4; bf16 autocast and compile are not bit-exact), and
the kept model is the rerun's. A0′ was done before the pause; its stages returned at once.

| arm | T (step) | L (3738, 4369 mean) | R | v1 loss at 4369 |
|---|---|---|---|---|
| A0 (seed 0) | 0.137 (671) | 0.794 | 0.657 | 3.920 |
| A0′ (seed 1) | 0.119 (785) | 0.781 | 0.662 | 3.910 |
| A0″ (seed 2) | 0.119 (785) | 0.748 | 0.630 | 3.899 |

| gate (`design-T5.md` "U4", Gate, last row) | value | bar | pass |
|---|---|---|---|
| pairs of L | A0 − A0′ 0.013, A0 − A0″ **0.046**, A0′ − A0″ 0.033 | | |
| τ = max(0.02, largest pair) | 0.046 | | |
| bar = max(0.2, 2τ); bar_I = √2 · bar | **0.2** (the floor binds; 2τ 0.092); 0.283 | | |
| readable | bar 0.2 | < R_A0 / 2 = 0.328 | **yes**: all four labels in use (lowers ≥ 0.328, partly 0.2–0.328, does not, raises) |

Start gates passed for both. **What it established:** three seeds of A0's recipe land within 0.046 of
each other at the late level, under a quarter of the floor, so the seed noise leaves U4's rule readable
with "partly" as its own label, and A1–A3 start under the rule as frozen. **Not established:** whether
the floor of 0.2 (placed from 410m's seed pairs, U4a) is larger than this recipe needs; it stays as
frozen. Each replicate would also pass A0's trough and late bars (T ≤ 0.25, L ≥ 0.60); they carry no
late gate record, so this is description.

Beside, no label (U4a's Parked 1; after output):

| step | 1253 | 1713 | 2002 |
|---|---|---|---|
| ρ_mid A0 / A0′ / A0″ | 0.305 / 0.284 / 0.263 | 0.548 / 0.543 / 0.474 | 0.635 / 0.619 / 0.561 |
| spread (max − min) | 0.041 | 0.075 | 0.074 |
| Dodecahedron | 0.129 | 0.32 | 0.43 |

So A0's earlier rise is not its seed: all three are 0.13–0.23 above Dodecahedron at 1253–2002, against
a three-seed spread of 0.04–0.075. Its cause (the data, the tokens a step, Dodecahedron's own seed) is
still not told apart. For timing reads between arms at these steps, differences under ≈ 0.075 are within
three seeds' spread; no timing rule was frozen, so this is a scale, not a bar. The three differ in the
init seed only (one shard, one order), so τ and this spread are the noise *between arms*, which share
that stream; against another run (Dodecahedron: other data, order and batch) they understate it, and
"not its seed" means not its init seed (/challenge-pr on #311, finding 2). **U4a's Parked 1: done**
(the spread recorded; the onset's cause stays under U4a's "Not established").

**The second change** (/challenge-pr on #311, finding 1, after output; nothing trained at it yet; `t5h`
`40289679fc36` → **`81b2e03bdcf7`**, `t5c`/`t5`/`t2`/`c2_aniso` untouched): `RESUMABLE` keyed only on the
old code, so A0's 4369 state would have resumed under *any* later `t5h`, the guard it was meant to
narrow. Crossings are now (from, to) pairs in a sidecar outside the hash, `trackT_timaeus/t5h_resumable.json`
(a pair names the code that reads it, so it cannot sit in a hashed file): 66af… → 4028… (history),
66af… → 81b2… (A0, for 5970), 4028… → 81b2… (A0′, A0″). Tests flipped (`test_trackT_t5h.py`: a pair
carries a state into its one code only); the loop's smoke tests 4 passed; `t5i`'s docstring gains
finding 2's clause. CodeRabbit on #311: `t5i` took a saved gate's `pass` without checking it was read on
the current records; it now refuses unless each gate's stored ρ_mid series equals the records (tests: a
changed value and a missing series refuse). `replicates.json` re-read at `t5i` `74cdc66d32f6`, `t5h_now`
`81b2e03bdcf7`: the three gates match their records, every number equal, only those two fields changed.
A1–A3 train at `81b2e03bdcf7`.

Re-run: `METS_DATA=<main>/data scripts/t5u4b_replicates.sh` (resumable; each stage gated); `python -m
trackT_timaeus.t5i replicates` alone rewrites `replicates.json` from the records. **Next: A1–A3 to 4369**
(a batch script to write), then the continuations and the delay check.

## T5 U4a: Blocked 133 decided; A0 (2026-11-03; A0 passes, 2026-11-04)

**Blocked 133, the user's, 2026-11-03, before any arm output** (`design-T5.md` "U4" rows τ/bar,
the arms, the gate, the throughput; "U4a's details" (t8), (t10); `tools/math_checks/t5_u4_budget.py` B11):

| item | decided | applied |
|---|---|---|
| (i) the noise floor | **(d)**, as recommended: (a′) and (c) | bar = max(**0.2**, 2τ), bar_I = √2 · bar (floor 0.283); "partly" from bar up to R_A0 / 2. **A0″** (F, seed 2) trained to 5970 with its continuation; τ = max(0.02, the largest of the three pairs among A0, A0′, A0″). The floor gate: 2·SD and the largest pair difference both ≤ 0.2 at 4000 and 143000. Rows: six arms and every continuation at θ ≥ 51.9k, six arms ≥ 39.8k; at 104.8k, 47.6 h of 96 (39.3 h before) |
| (ii) the loop's gate row | **float64 at rel 1e-10**, as recommended | the gate row and (t10) say so; float32 stays beside as a sanity bound |

Code `66af413c31fe` (commit `cddd7b2`). Re-run there:

| check | result | pass |
|---|---|---|
| the floor (`floor/floor.json`, overwritten; the 0.1 record's numbers are in "before A0" below) | 2·SD 0.021 / 0.153, largest pair 0.032 / 0.1996 at 4000 / 143000, against 0.2 | **placed, reproduced** (/challenge-pr on #310, finding 1): 0.2 was set from this pair, so the row cannot fail on these data; it shows the records were re-read unchanged. The 0.1996 is the gap between two groups (seeds 3, 4 at 0.23; the other eight 0.385–0.428), not a typical pair |
| known answer (`known/known.json`) | 6.8e-10 at 9536 | yes |
| the loop, float64 and float32 (`tests/test_trackT_t5h_smoke.py`) | 4 passed | yes |
| the interaction's planted null, six arms, τ from three pairs, 1e5 draws | σ 0.03: "interaction" 0.0000 against the pair's 0.00001; σ 0.05: **0.0024 against 0.0013** (bound +0.02). At floor 0.2 both rates are near zero, so the test now says little | yes |
| the rule on planted series | "partly" at a shift of 0.25 (0.15 now reads "does not"); τ from the largest pair (A0′ − A0″ in the test) | yes |

**A0 passes its positive control** (`scripts/t5u4a_A0.sh` at `66af413c31fe` throughout; launched
2026-11-03, paused by the user at update 3346, resumed 2026-11-04 from the state at 3198, so
3199–3348 were trained twice and `train_log.jsonl` holds both, split by its `segment` lines; the
second segment 5836 s for 1171 updates, exit 0). Input: Dodecahedron's recipe at 524,288 tokens a
step on the shard `a7c1b38b…` (FineWeb `sample-10BT` `9bb295dd`), arm A0 (init F, seed 0, no
biases); read on the 7 v1 prompts (`976b62d677bc`) through U2's reader. Records
`<HDD_1TB>/mets_data/trackT/t5u4/A0/` (`gate_start.json`, `gate_trough.json`, `gate_late.json`,
`read/`, `ckpt/` at the 49 grid steps to 4369).

| gate (`design-T5.md` "U4", rows 6–7) | A0 | bar | pass |
|---|---|---|---|
| start: init SDs, step-0 band, first output, the loss falls over 1–100 | ρ_mid 0.848 at 0 (band 0.787–0.881); loss 9.60 → 7.08 | as frozen | yes |
| trough: T_A0 by 1253 | **0.137 at 671** | ≤ 0.25 | yes |
| late: L_A0 = mean ρ_mid at 3738 and 4369 | **0.794** (0.787, 0.802) | ≥ 0.60 | yes |

R_A0 = L_A0 − T_A0 = **0.657**, so U4b's "lowers" is ΔL ≥ 0.328 and "partly" from bar (≥ 0.2) up
to it. Beside, not gated: the same steps on released `dodecahedron-32m` (`known/known.json`; U3's
first read):

| step | 671 | 917 | 1072 | 1253 | 1713 | 2002 | 3738 | 4369 |
|---|---|---|---|---|---|---|---|---|
| A0 ρ_mid | 0.137 | 0.156 | 0.197 | 0.305 | 0.548 | 0.635 | 0.787 | 0.802 |
| Dodecahedron ρ_mid | 0.10–0.11 | 0.111 | 0.116 | 0.129 | 0.32 | 0.43 | 0.76 | 0.812 |
| A0 v1 loss | 4.55 | 4.45 | 4.34 | 4.25 | 4.15 | 4.12 | 3.96 | 3.92 |
| Dodecahedron v1 loss | | 4.160 | | | | | | 3.766 |

**What it established.** At Dodecahedron's steps and LR schedule, at 524k tokens a step,
Dodecahedron's recipe on FineWeb reproduces the fall (0.137 against 0.111) and the late level
(0.794 against 0.788, R 0.657 against 0.677). So U4b's arms are read under the rule, with the
delay check (the design's "passes" row). By that row the rise is not set by tokens alone, if
Dodecahedron's batch was 4.19M (T1 Q3, leaning).

**Not established.** The rise's *onset* is not reproduced at Dodecahedron's steps: A0 is above
Dodecahedron by 0.08 at 1072, 0.18 at 1253 and 0.23 at 1713, and they meet by 3738. The gate reads
the trough and the late level only, so this is a description, read after output. Its cause is not
told apart: the data, the tokens a step, seed and order, A0's higher loss (0.29 above at 917, 0.15
at 4369; the v1 losses are not on one stream). One seed: A0′ and A0″ come in U4b.

**Parked** (why / cost / which decision it could change): 1. A0's rise starts earlier than
Dodecahedron's (above), and its lead (0.18 at 1253, 0.23 at 1713) is as large as bar / nothing new:
A0′ and A0″ (U4b) show whether it is the seed; record their spread at 1253, 1713 and 2002 beside τ
(/challenge-pr on #310, finding 2), so any timing read in U4b has a noise scale / whether a timing
difference between arms can be read at all (the labels, at 3738 and 4369, are unaffected; the
reviewer's arithmetic puts the delay check's 5970 as long enough). Carried to U4b.
2. **Defect, fix with U4b's first code change** (/challenge-pr on #310, finding 3): `t5h train`
with `--until` at or below the resume step trains nothing, exits 0 and writes a `segment` line, so
the resumed script left two empty segments (`until` 103 and 1253 from 3198) in A0's
`train_log.jsonl`. Not fixed here because any edit to `t5h.py` changes the code hash, and A0's
continuation to 5970 (U4b) resumes from A0's state at `66af413c31fe`; U4b changes the code anyway
and will say how the resume is allowed across it. Fix: refuse, or say so and return without a line.
3. `FLOOR_MAX`'s comment (`t5h.py:84`) should say *placed* (finding 1); with 2, for the same reason.

Re-run: `METS_DATA=<main>/data scripts/t5u4a_A0.sh` (resumable; each stage is gated; a resume state
continues only under its own code or under a (from, to) pair in `trackT_timaeus/t5h_resumable.json`:
`66af413c31fe`'s continues under `40289679fc36` and `81b2e03bdcf7`). Parked 2–3 fixed there ("T5 U4b" above); Parked 1 is read by `t5i replicates`.

## T5 U4a, before A0: the loop, the shard, θ, the known answer; the noise floor refuses (2026-11-03)

Code `trackT_timaeus/t5h.py` `b9e5724ac9d2` (commit `d8927fb`) for every output below; each was first produced at `24f9a6c81ed3`, which no commit holds, and re-run at the committed code after `/challenge-pr` on #309 (finding 4): known answer and floor equal to the last digit, the shard's sha256s equal, θ 2.4 % lower with the shard rebuild running beside it (the first records kept, suffixed `_24f9a6c81ed3`); details fixed before any
arm output in `design-T5.md` "U4a's details" (d1)–(t12). **No arm has run.** The 410m noise
floor, a gate row before the arms, fails, and the loop's gate row was changed from float32 to
float64; both are Blocked 133, yours. Records `<HDD_1TB>/mets_data/trackT/t5u4/` (linked from
`data/trackT/t5u4`).

| gate row | result | pass |
|---|---|---|
| known answer: released `-32m` 1 and 9536 saved and reloaded through t5h's format, read, against U2 (`known/known.json`) | max \|Δρ_ℓ\| 1.7e-9, 6.8e-10 (917 and 4369 beside: 1.1e-9, 3.7e-9; v1 loss 4.160, 3.766, as U3's 4.16, 3.77) | yes |
| the loop (`tests/test_trackT_t5h_smoke.py`) | float64, the loop's two micro-batches against a hand-written AdamW's full batch, clip off and binding: rel 1e-14 (bar 1e-10). **As written, float32 at rel 1e-5: 1.2–2.5e-5, fails**; each float32 side is 2.4e-4 from the float64 result, so the bar is below float32's rounding (LESSONS 4) | float64 yes; as written no |
| **the noise floor** (`floor/floor.json`): ρ_mid SD over `pythia-410m` seeds 0–9 | 4000: SD 0.0104, 2·SD 0.021. **143000: SD 0.0765, 2·SD 0.153** (seeds 3, 4 at 0.233, 0.228; the other eight 0.385–0.428). Seed 0 equals U2's record at both (0.0) | **no** (bar 0.1) |
| θ, A0's configuration, 10 + 50 updates of the real stream (`theta.json`) | **104.8k tokens/s** (107.4k at the first run, with the CPU idle; 4.3 GB; `torch.compile`, micro-batch 4); the loss 9.77 → 7.60 over the 60 | row: five arms and every continuation |

Beside: the interaction's planted null (a unit test before U4b, `tests/test_trackT_t5h.py`), read
two-sided ((t11)), at 1e5 draws ((t11); the design's 1,000 passed about 76 % of seeds, `/challenge-pr`
on #309, finding 2): "interaction" 0.0121 against the pair's 0.0096 at σ 0.03, **0.0895 against
0.0749 at σ 0.05** (bound 0.0949; at 1,000 draws it had read 0.095 against 0.095). The rule's arithmetic on planted series passes (a
lowered-by-R/2 series reads "lowers", A0′'s "does not"; the delay check and the H labels).

**The shard** (`data/meta.json`): FineWeb `sample-10BT` at `9bb295dd`, files 000–003, 4,133,057
documents, 1,528,320 sequences of 2048 (3.13B tokens): main 4369 × 256 (sha256 `a7c1b38b…`), the
delay check's tail 1601 × 256 (`18f023eb…`); `<|endoftext|>` 0.13 % of tokens (about 740 tokens a
document); 19 min on the CPU. First sequences decode as FineWeb text.

**What θ does to the budget.** A0 to 4369 is 6.1 h; the five arms and four continuations
14.8B tokens, 39 h of the 96 h cap; a further replicate seed with its continuation 8.3 h.

**The floor's shape** (`/challenge-pr` on #309, finding 1): at 143000 the ten seeds sit in two
groups, so 2·SD (0.153) understates what one pair can differ by: the largest pair difference is
0.200 (at 4000, 0.032). A pair drawn from 410m's split lands in one group 68 % of the time, three
seeds 52 %. Blocked 133's options carry both numbers.

**Not established.** Anything about training: no arm ran. Whether 410m's late bimodality
(two seeds of ten end 0.17 below the rest) has any counterpart at Dodecahedron's 4369, which is
2.29B tokens against 410m's 8.4B at 4000 and 300B at 143000.

**Parked** (why / cost / which decision it could change): 1. `pythia-410m` seeds 3 and 4 end at
ρ_mid 0.23 against 0.39–0.43 / U5 reads these seeds at the 18 steps anyway / whether the floor's
step is the one Blocked 133 should take; U5 describes it.

Re-run (`HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data`): `python -m trackT_timaeus.t5h
shard`, `known`, `floor`, `theta --out <root>/theta.json`;
`SMOKE_REAL_DEPS=1 python -m pytest tests/test_trackT_t5h_smoke.py tests/test_trackT_t5h.py`. A0,
once Blocked 133 is decided: `scripts/t5u4a_A0.sh` (stages 103, 1253, 4369, each gated; resumable;
the configuration (t12) is fixed in `t5h` and refused if a segment differs, finding 3).

## T5 U4s: the screen at init (2026-11-03)

D6 (b), the user's (Blocked 132). Rule: `design-T5.md` "U4s, the screen at init", frozen 2026-11-03;
implementation (i)–(v) written there and pushed with the producer before any output (`e947ded`).
Producer `trackT_timaeus/t5g.py` (on `t5b`; code `2d7d55d4e4f9`, reader `f19874efeb15`, the
post-hoc `boot` added after the reading); records `data/trackT/t5u4s/2026-11-03/` (main tree;
3,160 models, 626 s on the RTX 3080). **Input:** the 7 v1 prompts (`976b62d677bc`), ≤ 512 tokens,
float32, eager, TF32 off, torch 2.10.0, transformers 4.44.2; Llama at 32m's shape without biases,
U1b's base and swaps; 79 cells × 2 inits × 20 seeds (the design's count, 77, left out two gate
cells; implementation (i)).

**Gate: passes on every check.**

| check | result | pass at |
|---|---|---|
| U1b's records reproduced: ∅, the six singles, `untied`, `all`, same seeds | max \|Δρ_ℓ\| 0.0 over 360 models | ≤ 1e-6 |
| `gate` + `act` + `width` built together against `gelu`'s cell | −0.001 (P), −0.006 (F) | ≤ 0.05 |
| the reverse's all-six against the base: *by implementation (ii) this cell is `untied`'s construction on fresh seeds, so it tests seed noise, not the reverse's build* (`/challenge-pr` on #308, finding 4) | −0.010 (P), −0.003 (F) | ≤ 0.05 |
| the init: per role, seed 0 of every cell | largest relative gap 0.13 % | ≤ 2 % |

τ 0.02 (`untied`'s M −0.003, −0.001, as U1b), bar 0.1, **E0 0.363**, E0 / 2 = 0.182.

**Reading: the minimal carrying sets are `par`+`gelu` and `gelu`+`mha`; put back into GPT-NeoX,
SwiGLU and the sequential residual each suffice alone.** 17 of 63 subsets carry, 15 partly, 31 do
not, none amplifies. Every carrying set holds `gelu`; all but `gelu`+`mha` hold `par`+`gelu`.

| cell | P | F | E | ΔE (share of E0) | label |
|---|---|---|---|---|---|
| base (Llama) | 0.478 | 0.842 | 0.363 | 0 | |
| `par` | 0.411 | 0.815 | 0.404 | −0.040 (−0.11) | does not |
| `gelu` | 0.535 | 0.735 | 0.199 | 0.164 (0.45) | partly |
| **`par`+`gelu`** | 0.424 | 0.371 | −0.054 | **0.417 (1.15)** | **carries** |
| `gelu`+`mha` | 0.549 | 0.722 | 0.173 | 0.190 (0.52) | carries (by 0.008) |
| `all` (GPT-NeoX) | 0.418 | 0.347 | −0.070 | 0.434 (1.19) | carries |

- **`par`+`gelu` is the effect.** Their pair interaction is 0.81 of E0 on ρ (0.67 logit, 0.57
  atanh); every other pair's is ≤ 0.08. Adding `ln`, `rot`, `mha` or `tok` to it moves ΔE by ≤ 0.02.
- **`gelu`+`mha` is on the line**, not a second mechanism: 0.026 above `gelu` alone, below the bar.
  A seed bootstrap (post hoc, `t5g boot`, 2,000 resamples) puts its share at 0.49–0.55 (5–95 %),
  at or above 0.5 in 82 %. `gate`+`act` (0.49) sits on the same line from below (47 %).
- **The 2 × 2** (E = med F − med P; the other four parts Llama's / GPT-NeoX's):

  | | SwiGLU | GeLU 4d |
  |---|---|---|
  | sequential residual | 0.363 / 0.379 | 0.199 / 0.191 |
  | parallel residual | 0.404 / 0.417 | −0.054 / −0.070 |

  **Reverse** G = E_k − E_all (the frozen labels): SwiGLU 0.487 (1.33 of E0), the sequential
  residual 0.261 (0.72), both "suffices" (bootstrap 5 % ends 1.27 and 0.66); RMSNorm 0.007, full
  rotary 0.016, 2 KV heads 0.017, Dodecahedron's tokenizer −0.004, "does not". **G overstates both
  against the Llama's effect**, since GPT-NeoX's own E is below zero (−0.070, 0.19 of E0): measured
  as E_k / E0, SwiGLU put back gives **1.15** (all of it) and the sequential residual **0.53** (about
  half, on the line as `gelu`+`mha` is) (`/challenge-pr` on #308, finding 2). So the gated MLP
  carries the whole effect whatever the residual; the residual's form matters once the MLP is
  ungated. To remove it from the Llama both must go.
- **Within `gelu`, ungating carries it.** `gate` (SiLU 1024, ungated) 0.163 (0.45), the same as
  `gelu`'s 0.164; `act` (GeLU, gated) −0.053 "does not"; `width` (1536, gated SiLU) −0.125
  **"amplifies"** (P falls to 0.310); `gate`+`act` 0.180 and `gate`+`width` 0.118 "partly",
  `act`+`width` −0.159 "amplifies". All three 0.169. **The gate is not told apart from the MLP's
  write size** (`/challenge-pr` on #308, finding 3): each role's SD is fixed whatever the form, so
  at init the ungated cells write 1.6–2.1× the base's RMS under P and 2.6–3.3× under F, the gated
  ones 1.06–1.33× (`tools/math_checks/t5_u4s_mlp_size.py`, a Gaussian-input estimate, not the
  records). Size is not monotone in the share (gated 1.2–1.3× amplify, ungated 1.6–3.3× remove about
  half), but every cell that removes it is ungated *and* writes more.
- **Not additive:** mean ΔE by subset size 0.05, 0.17, 0.33, 0.55, 0.84, 1.19 of E0 (logit and atanh
  within 0.07 of these).
- **Two draws of one construction** (the screen's five-part subsets against the reverse cells,
  independent seeds): E within 0.023 for all six, the size of τ.

**What it says.** At init, in Dodecahedron's architecture at 32m's shape, the flat init concentrates
the cloud fully when the block has a gated MLP (E 0.36–0.42, 1.0–1.15 of E0, whatever the
residual), and about half when it has a sequential residual with GPT-NeoX's ungated MLP (0.19–0.20);
with both of GPT-NeoX's (parallel residual, ungated GeLU MLP) not at all (E −0.05 to −0.07). In the
MLP, ungating removes about half (with the larger write it brings; not told apart), GeLU for SiLU
does not, and the wider gated MLP amplifies it. RMSNorm, the rotary fraction, the KV heads and the
tokenizer do nothing in any combination screened. This explains U1b's "no single part": no single
swap removes the effect because the gated MLP produces it and the sequential residual half of it.

**Not established.** Why either part makes the flat init concentrate. U1b's candidate (an ungated
activation's positive mean writes one vector every token shares; a gated product's mean is 0) names
the gate, but read plainly it predicts ungating raises ρ under both inits, and dropping the gate
raises it under P (0.478 → 0.583) and lowers it under F (0.842 → 0.783); not tested. Why the
residual's form matters is not addressed by any candidate here. Whether the gate or the write's
size it sets carries the MLP's part (Parked 2). Whether other shapes agree (one shape read).
Anything through training: U2 found the init's concentration gone by 671–917, and U4's arms
(biases, the init) are frozen and not changed by this (the design).

**Parked** (why / cost / which decision it could change): 1. A trained arm that swaps the MLP's
form, and the residual with it (A0 with an ungated MLP and a parallel residual; the MLP is the part
that carries the whole effect alone, finding 2) / one U4-sized arm, about 11 h / whether the late
rise, which the init's concentration does not carry (U2), needs these parts. U4's arms are frozen
without it, so it is a candidate after U4b, not a change to U4. 2. A size-matched control: the
ungated cells with `down_proj` scaled to the base's write RMS, and the base scaled up to theirs
(finding 3) / about 2 GPU-minutes and a rule / whether 1 swaps the gate or the MLP's scale; it goes
before 1 is costed.

Re-run: `HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t5g run --out <dir>`
(skips records present), then `METS_DATA=<main>/data python -m trackT_timaeus.t5g read --out <dir>`
(gate first, against U1b's records; writes `reading.json`) and `python -m trackT_timaeus.t5g boot
--out <dir>` (post hoc; `boot.json`).

## T5 U4's design frozen (Blocked 132, 2026-11-03)

The user's: **D4 (a)** Dodecahedron's steps and LR at 524k a step to 4369; **D5 (b)**, over the
recommended (a), "but add to the budget": A3 (biases and Pythia's init together, A2 plus zero
biases) runs, under the arm rule and the delay check, with its own label and an interaction
I = ΔL_A3 − ΔL_A1 − ΔL_A2 at bar √2 × max(0.1, 2τ) (B10; the floor scaled after `/challenge-pr` on #307, finding 3); the cap goes from 3 to 4 GPU-days so
the delay check keeps its budget (B9: five arms and four continuations at θ ≥ 42.9k tokens/s, five
arms ≥ 33.1k, else A0 alone and back to the user); **D6 (b)**, over the recommended (a): U4s runs
first, the full screen of U1b's six swaps (64 subsets), `gelu` split into its three changes, and the
reverse ablation (one Llama part into GPT-NeoX), 3,080 random models, about 10 GPU-minutes,
labelled with U1b's bar, context at init that does not change U4's arms; **D7 (a)** U4a, then U4b.
So U4 is three units (U4s, U4a, U4b), two beyond T5's budget, on the record. The rules A3, the
interaction and U4s were written at the freeze, before any code (`design-T5.md` "U4, the causal
arm, designed" and "U4s, the screen at init"); they were not in the design `/challenge-pr` saw on
#306. Re-run the arithmetic: `python tools/math_checks/t5_u4_budget.py`.

## T5 U4's design: the causal arm, aimed at the rise (2026-11-02)

Docs only; nothing trained or run except `tools/math_checks/t5_u4_budget.py` (arithmetic). The
design is `design-T5.md` "U4, the causal arm, designed", **proposed, not frozen: Blocked 132**
(D4–D7). In short: Dodecahedron's recipe trained from scratch at the card's 524,288 tokens a step
under its steps and LR schedule, stopped at 4369, where every count has stopped (U3's first read).
Four arms: A0 the recipe, A0′ a second seed, A1 biases on (H2), A2 Pythia's init (H1 through
training); no fifth (D5 (a)). Rule: per arm the late level L (mean ρ_mid at 3738, 4369); a change
**lowers the late level** if it lowers L by at least half of A0's rise L − T, **partly** from the
bar max(0.1, 2τ), where τ comes from A0′; then **the delay check**: A0 and every arm that lowers L
continue to 5970, and the label holds only if the gap stays (≥ bar and ≥ half of it); else
"delays", read "not told apart". The floor 0.1 is checked first against `pythia-410m`'s ten seeds. Gate: A0 runs first and alone, and must reproduce the fall
(T ≤ 0.25 by 1253) and the rise (L ≥ 0.60); otherwise U4 refuses with A0's series as the reading
on steps against tokens. The arms that fit follow a measured throughput (B7: the four arms and
every continuation need ≥ 45.1k tokens/s; below 35.4k only A0 runs and the rest returns to the
user; ROADMAP's 60k is never measured and about 34 % of the GPU's dense rate, B8). Not matched: Dodecahedron's FineWeb subset and order,
packing and BOS, LR floor (0.662 against 0.696 of the peak at 4369, B3), weight-decay exclusions.
Proposed dropped: Blocked 125 (a)'s pair screen and reverse ablation (D6). U3's Parked 3 (the τ
search bound) is not hit by U4, which derives no T2 arm; carried to T5's close-out. `/challenge-pr` on #306: accept with changes; findings 1–5 taken (the
delay check and the renamed label; no fifth arm, LR's case and why it is not readable here (B6);
the throughput rows; the 410m noise floor; the step-0 bands written out, A2's added).

## T5 U3's split, the batch: which paths write the shared direction (2026-11-02)

Rule: Blocked 131 (a), the user's, as recommended: the planted-bias gate retired on the record
before any batch output (`design-T5.md` "The planted-bias gate retired on the record", pushed in
`3e9550c`; re-run with token constancy beside in `88e9ca4`, code `ef81a441e7dc`, after `/challenge-pr` on #305); the sum gate, the known answer and the tiny-model n · b·u test
stay. **Input:** `dodecahedron-32m` at 1, 917, 1253, 1713, 2340, 3198, 4369, 6979, 9536 and
`pythia-410m` at 0, 16, 128, 512, 1000, 4000, 32000, 143000, the 7 v1 prompts (`976b62d677bc`),
≤ 512 tokens, float32 forward (eager, TF32 off), float64 split, RTX 3080, 44 s (`runner`, no
edits). Records `data/trackT/t5u3s/2026-11-02-batch/` (main tree; `reading.json` `e46f387c85c4`,
32m 9536 `c0b998628088`, 410m 143000 `8e43364bb224`; the first run, code `55a6714b242c`, gave
every share and size the same). Re-run: `HF_HOME=<HDD_1TB>/mets_data/hf
METS_DATA=<main>/data python -m trackT_timaeus.t5f run --out <dir>`, then `t5f read --out <dir>
--reproduce <first output's 32m/step9536.json>`.

| check | bar | result |
|---|---|---|
| the parts sum to the block's write (max over tokens, blocks, prompts, 17 records) | ≤ 1e-4 | **passes**, 1.2e-5 |
| known answer: ρ_ℓ against T2's `diag` at 32m 1, 9536 and 410m 0, 4000 | ≤ 1e-6 | **passes**, 0.0 at all four |
| tiny models: bias path n · b·u exactly (tests) | rel 1e-6 | passes (`tests/test_trackT_t5f_smoke.py`) |
| beside: 9536 against the first output (`c8c966ac571c`, code `ffab0d6aff56`) | 1e-12 | **equal**, 0.0 |
| beside, retired: the plant at 9536, its bias path's share of the model-level planted component | (0.9, retired) | 0.125, as the first output |

**Shares** of the summed component on u (over blocks, prompts, tokens; they sum to 1) and **the
write's size** (mean over blocks of the mean per-token component over the block's RMS token norm).
Each rise step beside its first-read row (`status-T.md` "T5 U3, first read"). The biases are zero on
Dodecahedron by architecture, so its attn_bias and mlp_bias are 0.000 throughout. **"MLP const" is
the MLP write's pooled mean** (over every token of the 7 prompts), and "MLP rest" each prompt's
deviation from it: on summed components every path counts only through token averages, so the
split does not say the MLP writes one vector to every token (`/challenge-pr` on #305, finding 1;
the token constancy below says it does not).

| `-32m` step | ρ_mid | n0 | sink | attn rest | MLP mean (const) | MLP rest | write size | centred P | fixed τ A | uncentred A |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (0.85) | | 0.006 | **0.872** | 0.023 | 0.099 | 0.234 | | | |
| 917 | 0.111 | 2.2 | 0.006 | 0.450 | 0.342 | 0.201 | 0.093 | yes | −0.01 | 0.93 |
| 1253 | 0.129 | 4.1 | 0.011 | 0.444 | 0.400 | 0.145 | 0.099 | yes | 0.01 | 0.91 |
| 1713 | 0.315 | 4.8 | 0.022 | 0.433 | 0.442 | 0.103 | 0.117 | yes | 0.06 | **0.72** |
| 2340 | 0.546 | 9.0 | 0.039 | 0.448 | 0.439 | 0.074 | 0.129 | refused | **0.22** | 0.19 |
| 3198 | 0.706 | 11.3 | 0.054 | 0.459 | 0.431 | 0.056 | 0.137 | refused | 0.43 | 0.13 |
| 4369 | 0.812 | 15.1 | 0.078 | 0.479 | 0.401 | 0.042 | 0.144 | **no** | 0.46 | 0.16 |
| 6979 | 0.888 | 18.4 | 0.088 | 0.508 | 0.375 | 0.029 | 0.149 | no | 0.55 | 0.21 |
| 9536 | 0.903 | 23.8 | 0.128 | 0.596 | 0.245 | 0.032 | 0.147 | no | 0.56 | 0.04 |

(Step 1's ρ_mid is U2's, 0.85; the first read did not cover it. In **bold**, each count's first
step past its stop: uncentred 1713, fixed τ 2002 (not a split step; 2340 shown), centred 4369.)

**Sizes per path** (mean over blocks, as above) and, beside, post hoc (finding 1 on #305; fixed in
`88e9ca4` before its re-run), **token constancy**: per block the fraction of a module's write's
squared norm in its pooled mean, N ‖m‖² / Σ‖w_i‖² (1 iff every token gets the same vector), mean
over blocks. Bias paths 0 on `-32m`.

| `-32m` step | sink | attn rest | MLP mean | MLP rest | write | const. MLP | const. attn |
|---|---|---|---|---|---|---|---|
| 1 | 0.0014 | 0.1918 | 0.0083 | 0.0321 | 0.234 | 0.122 | 0.173 |
| 917 | 0.0005 | 0.0397 | 0.0280 | 0.0242 | 0.093 | 0.056 | 0.138 |
| 1253 | 0.0010 | 0.0420 | 0.0359 | 0.0205 | 0.099 | 0.056 | 0.140 |
| 1713 | 0.0018 | 0.0472 | 0.0547 | 0.0137 | 0.117 | 0.073 | 0.208 |
| 2340 | 0.0027 | 0.0524 | 0.0645 | 0.0092 | 0.129 | 0.086 | 0.222 |
| 3198 | 0.0039 | 0.0564 | 0.0693 | 0.0070 | 0.137 | 0.097 | 0.248 |
| 4369 | 0.0056 | 0.0586 | 0.0750 | 0.0051 | 0.144 | 0.109 | 0.300 |
| 6979 | 0.0068 | 0.0616 | 0.0774 | 0.0035 | 0.149 | 0.135 | 0.362 |
| 9536 | 0.0100 | 0.0672 | 0.0668 | 0.0033 | 0.147 | 0.146 | 0.417 |

| 410m step | sink | attn bias | attn rest | MLP bias | MLP mean | MLP rest | const. MLP | const. attn |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0005 | 0 | 0.0368 | 0 | 0.0689 | 0.0055 | 0.405 | 0.781 |
| 16 | 0.0006 | 0 | 0.0446 | 0 | 0.0846 | 0.0024 | 0.707 | 0.916 |
| 128 | 0.0005 | 0.0001 | 0.0291 | 0.0001 | 0.0563 | 0.0277 | 0.281 | 0.413 |
| 512 | 0.0001 | 0.0001 | 0.0222 | 0.0002 | 0.0579 | 0.0329 | 0.307 | 0.233 |
| 1000 | 0.0006 | 0.0008 | 0.0344 | 0.0008 | 0.0318 | 0.0162 | 0.100 | 0.184 |
| 4000 | 0.0009 | 0.0047 | 0.0477 | 0.0045 | 0.0099 | 0.0032 | 0.073 | 0.112 |
| 32000 | 0.0014 | 0.0074 | 0.0333 | 0.0051 | 0.0279 | 0.0028 | 0.132 | 0.111 |
| 143000 | −0.0013 | 0.0184 | 0.0113 | 0.0140 | 0.0337 | 0.0062 | 0.204 | 0.157 |

| `pythia-410m` step | sink | attn bias | attn rest | MLP bias | MLP mean (const) | MLP rest | write size |
|---|---|---|---|---|---|---|---|
| 0 | 0.005 | 0.000 | 0.364 | 0.000 | **0.580** | 0.051 | 0.112 |
| 16 | 0.005 | 0.000 | 0.363 | 0.000 | 0.620 | 0.011 | 0.132 |
| 128 | 0.004 | 0.001 | 0.260 | 0.001 | 0.552 | 0.182 | 0.114 |
| 512 | 0.000 | 0.002 | 0.157 | 0.002 | 0.502 | 0.338 | 0.114 |
| 1000 | 0.008 | 0.010 | 0.440 | 0.010 | 0.310 | 0.221 | 0.085 |
| 4000 | 0.014 | 0.061 | 0.677 | 0.058 | 0.119 | 0.071 | 0.071 |
| 32000 | 0.026 | 0.076 | 0.383 | 0.061 | 0.385 | 0.068 | 0.078 |
| 143000 | −0.027 | 0.165 | 0.197 | 0.143 | 0.414 | 0.109 | 0.082 |

**At the end states** (`-32m` 9536, 410m 143000; the write 0.147 against 0.082): attention's
ordinary keys 0.067 against 0.011; the MLP write's pooled mean 0.067 against 0.034; 410m's biases
(token-constant by construction) 0.018 + 0.014; the sink 0.010 against −0.001.

What the batch shows (a description, read as direct writes; no rule, no label, no H2 reading):

1. **Through the rise, attention's ordinary keys and the MLP write's pooled mean carry the shared
   component about equally** (shares 0.43–0.60 and 0.25–0.44 at 917–9536; at 9536 equal in size,
   0.067 each, but 0.60 against 0.25 in share, since the share sums signed components over every
   token and the size averages per block). The write's size rises 0.093 → 0.149 as ρ_mid rises
   0.11 → 0.90. The MLP mean's share peaks at 1713–2340 (0.44) and falls to 0.25 at 9536.
2. **The sink path grows steadily, 0.006 → 0.128 in share**, with position 0's norm (2.2 → 23.8);
   it stays the smallest attention path. Dropping position 0 from the sums moves no share by more
   than 0.001 (410m 0.008), so position 0's own tokens do not carry the shares.
3. **On shares, no path changes abruptly at a count's stop** (1713, 2002, 4369): between
   neighbouring split steps every share moves by ≤ 0.13, the largest step MLP mean 0.375 → 0.245 at
   6979 → 9536, after every count has stopped. **On sizes**, the MLP mean carries most of the write's
   growth to 4369 (0.028 → 0.075 while the write goes 0.093 → 0.144), its fastest relative growth
   (+52 %) over 1253 → 1713, the interval ending at the uncentred count's stop; attention's token
   constancy rises 0.14 → 0.21 over the same interval. Side by side only; "abruptly" had no
   threshold fixed before output (corrected after `/challenge-pr` on #305, finding 2: this
   section first read statement 3 on shares alone).
4. **Dodecahedron's first checkpoint (step 1) is written by attention**: attention's ordinary keys
   0.87, the MLP mean 0.02. 410m's step 0 is the other way round: the MLP mean 0.58, attention 0.36
   (wording after finding 4: step 1 is not an init).
5. **410m's biases write a growing part late** (both together 0.00 at step 0, 0.12 at 4000, 0.31 at
   143000). The first version of this section read "the constant writes are equal in size at the
   end states" (the MLP mean 0.067 against MLP mean plus biases 0.066): withdrawn, since it grouped
   a pooled mean with token-constant biases by label (finding 1).
6. **Neither model's MLP writes one vector to every token** (post hoc): on `-32m` 0.06–0.15 of the
   MLP write's squared norm lies in its pooled mean through the rise, rising with ρ; the attention
   write's fraction rises more, 0.14 → 0.42. On 410m both are high at steps 0–16 (MLP 0.41–0.71,
   attention 0.78–0.92) and 0.07–0.31 after.

**Not established:** **what raises ρ**: the split measures the growth of the component along the
shared direction, not the shrinking of each token's own part, and ρ_mid goes 0.11 → 0.90 while the
write's size goes only 0.093 → 0.149 (finding 3 on #305; U4 is aimed at the rise); what a path's
removal would do (direct writes only; the plant showed that a model answers a constant with
another); whether a module writes near-constant vectors to subsets of tokens (token constancy pools
every token); `-67m`; any split step between 1713 and 2340 (where the
fixed-τ count stops at 2002); step 1's first-read row (not read there).

**Parked** (why / cost / which decision it could change): (1) the model's response to a constant
(Blocked 131 (a)): whether a large MLP const can answer an upstream constant, as at the plant;
minutes on the GPU (a dose series of the plant); it could change how the MLP's write is worded in
T5's close-out. Carried to T5's close-out. (2) The end states' equal constant sizes: **dropped**
(why: finding 5's grouping withdrawn, the MLP write is not token-constant, 0.15 / 0.20).

## T5 U3's split: the planted-bias gate refuses (2026-11-02)

Rule: `design-T5.md` "U3" and "U3's split, the details", with the producer `trackT_timaeus/t5f.py`,
pushed before any split output (`08c7a19`, code `ffab0d6aff56`). **Input:** `dodecahedron-32m` at
9536 only (the first output, opened before the batch as CLAUDE.md asks), the 7 v1 prompts
(`976b62d677bc`), ≤ 512 tokens, float32 forward (eager, TF32 off), float64 split, RTX 3080, 2 s.
Records `data/trackT/t5u3s/2026-11-02/32m/` (main tree): `step9536.json` `c8c966ac571c`,
`step9536_planted.json` `dd74c0e25dcd`. Re-run: `HF_HOME=<HDD_1TB>/mets_data/hf
METS_DATA=<main>/data python -m trackT_timaeus.t5f run --out <dir> --models 32m --steps 9536`.

| gate (design-T5.md "Calibration gate") | bar | at 9536 |
|---|---|---|
| the parts sum to the block's write | ≤ 1e-4 | **passes**, 1.1e-5 (all of it float32 against float64: the recomputed attention is 8.4e-7 from the module's output; the hooks' attention + MLP is 1.1e-5 from x_out − x_in) |
| known answer: ρ_ℓ from the split's forward against T2's `diag` | ≤ 1e-6 | **passes**, 0.0 (32m 1 and 410m 0, 4000 not run) |
| planted o_proj bias (‖b‖ 345, 0.5 × the median middle-state norm 690, in all 16 blocks): the bias path's share of the planted component | ≥ 0.9 | **fails, 0.125** (bias path 6.30e6 of 5.04e7) |

**Where the planted component went** (planted less unplanted, per path's summed component; their
sum is the planted component): attn_bias 0.125, **mlp_const 0.950**, attn_rest −0.044, attn_sink
−0.024, mlp_rest −0.006. The plant sends ρ_ℓ to 1.00 already at state 1 (unplanted: the 0.90
cloud): ‖b‖ 345 is about the whole stream at block 1 (its output's RMS norm 390, unplanted) and
0.13 of it at block 16 (2669), since it was sized on the middle states' median (690), so the
collapse is block 1's, not a build-up over blocks. **The MLPs do not relay b; they write a new,
larger constant** (corrected after `/challenge-pr` on #303, finding 1; this PR first wrote
"relays"): the token-constant MLP vector is near-orthogonal to b in every block (cos −0.16 to 0.05;
its norm 190–11951 against ‖b‖ 345; from a re-run of the planted forward, not stored), the token
RMS norms grow 2.8–7.3 ×, and b lies at cos 0.20–0.43 to the planted u in blocks 1–15 (0.73 at
16), so u follows the MLPs' new constant, not b. Blocks 1, 7, 8 and 16 write the most. **The split
is exact** (the sum gate; the bias path is n · b·u per block, tested on tiny models,
`tests/test_trackT_t5f_smoke.py`); **the gate's plant moves the model** into another regime, and
the "planted component" (fixed as the model-level difference in `design-T5.md` "U3's split, the
details", before output) counts that response. Checking ‖b‖ against each block's norm before output
would have shown the plant as large as block 1's stream (finding 2). So the gate tests the model's
response to a large constant, not the split. And (finding 3): with a random direction in 384
dimensions, b's own component on the unplanted u is small and of either sign, and the two runs are
each measured on their own u, so ΔT also carries u's rotation; a plant along the unplanted u, both
runs measured on it, avoids both.

**Seen before the gate's revision:** the unplanted 9536 shares (sink 0.13, attn_rest 0.60,
mlp_const 0.24, mlp_rest 0.03; biases 0 by architecture), printed by the run. No other step run.

**Not established:** any path's share or size at any other step or on 410m; whether a smaller plant
stays in the linear regime. **Decision: Blocked 131** (`STATE.md`), decided (a) 2026-11-02: the gate
retired on the record, the batch run ("T5 U3's split, the batch" above). LESSONS 6.

## T5 U3, first read: T2's gates along the late rise (2026-11-02)

Blocked 130 (a), the user's. Rule: `design-T5.md` "U3 revised toward the late rise" (#301, before
any output). Producer `t2.py preflight --model 32m --trained-step <s>` (code `b01ad9e3e4d8`, as
U2b), reader `trackT_timaeus/t5e.py`, pushed before the records were read (`c7189da`). Records
`data/trackT/t5u3/2026-11-02/` (main tree): `reading.json` `e12c7679fddf` (after `/challenge-pr` on #302; first `22f1f46a916b`); per step
`32m-step<s>/preflight_plants.json` (9536 in `32m/`, `75af52727e6e`; 917 `c96699cfde27`).
**Input:** `dodecahedron-32m` at the 16 grid steps 917, 1072, …, 8158, 9536, its 3 re-inits, the
7 v1 prompts (`976b62d677bc`), ≤ 512 tokens, float32, eager, TF32 off; 112 trained and 336 re-init
units per step. About 40 s a step on the RTX 3080 (11 min in all).

**Gate: passes.** `--trained-step 9536` reproduces T2 unit 1's record and `--trained-step 917`
U2b's: `derived`, `arms`, `gate` max |Δ| 0.0, no mismatch (the same file hashes as U2b's).

**A refusal the rule did not foresee** (a gap in the design, filled post hoc after the batch and
before the reading, `34b0b7c`): at 4 of the 14 steps T2's preflight refuses to derive its arms.
At 2340 and 2735 the uncentred count sees plant B at no dose up to 0.5 (`DoseRefused`), so B's dose
is undefined: the count's own refusal. At 3198 and 3738 a dose is found, but `c2_seeds.b_matched_tol`
searches τ only up to 1e-2 (a code default, in no frozen design) and the B-matched τ lies just
above it (`SeedsRefused`): **a limit of the search, not of the count** (`/challenge-pr` on #302,
finding 1; this PR first wrote "no τ exists"). Searched to 1e-1 (post hoc, beside, `t5e.WIDE_HI`;
the same search reproduces the recorded B-matched τ at the other 12 steps to 3 figures) the arm is
1.12e-2 at 3198 and 1.06e-2 at 3738 and **passes T2's full gate at both** (plant A 0.04, 0.08; B
0.42, 0.45; re-init 0.3 %). (ii) and (iii) do not need the derivation and are read at all 16
steps; (iii) is computed from the units and equals the records at the other 12 steps exactly (max
|Δ| 0.0). A refused step's units file names no step and carries no checks (`t2.py` writes both
only with the plants record), so the reader requires its log to show that step loaded (finding 2).
(i) is read both ways at the refused steps: counted as no arm with P, and left out.

| step | ρ_mid | n0 / median | loss | (i) B dose, B-matched τ: A, B share | (i) any P (full gate) | (ii) τ 3e-3, dose 0.5: A | (iii) uncentred A (base rate) |
|---|---|---|---|---|---|---|---|
| 917 | 0.111 | 2.2 | 4.16 | 0.5, 3.37e-3: 0.00, 0.77 | yes (yes) | −0.01 | 0.93 (0.19) |
| 1072 | 0.116 | 3.5 | 4.11 | 0.5, 3.85e-3: 0.00, 0.76 | yes (yes) | 0.00 | 0.92 (0.20) |
| 1253 | 0.129 | 4.1 | 4.01 | 0.5, 3.89e-3: 0.01, 0.72 | yes (yes) | 0.01 | 0.91 (0.28) |
| 1465 | 0.198 | 4.4 | 3.99 | 0.5, 5.57e-3: 0.00, 0.65 | yes (yes) | 0.03 | 0.89 (0.45) |
| 1713 | 0.315 | 4.8 | 3.90 | 0.5, 6.30e-3: 0.00, 0.60 | yes (yes) | 0.06 | **0.72** (0.55) |
| 2002 | 0.431 | 7.7 | 3.86 | 0.25, 5.73e-3: 0.01, 0.41 | yes (yes) | **0.12** | 0.50 (0.62) |
| 2340 | 0.546 | 9.0 | | refused (no B dose) | — | 0.22 | 0.19 |
| 2735 | 0.639 | 9.2 | | refused (no B dose) | — | 0.36 | 0.13 |
| 3198 | 0.706 | 11.3 | | refused (τ > 1e-2); searched to 1e-1, post hoc: 0.5, 1.12e-2: 0.04, 0.42 | — (post hoc: yes (yes)) | 0.43 | 0.13 |
| 3738 | 0.763 | 13.2 | | refused (τ > 1e-2); searched to 1e-1, post hoc: 0.5, 1.06e-2: 0.08, 0.45 | — (post hoc: yes (yes)) | 0.46 | 0.16 |
| 4369 | 0.812 | 15.1 | 3.77 | 0.5, 9.88e-3: 0.14, 0.45 | **no** (no) | 0.46 | 0.16 (0.78) |
| 5108 – 8158 | 0.847 – 0.900 | 16.1 – 19.8 | 3.75 – 3.68 | 0.25–0.5, 7.9–9.2e-3: 0.27–0.42, 0.46–0.53 | no (no) | 0.49 – 0.56 | 0.14 – 0.25 (0.74–0.75) |
| 9536 | 0.903 | 23.8 | 3.66 | 0.25, 8.82e-3: 0.42, 0.47 | no (no) | 0.56 | 0.04 (0.75) |

ρ_mid and n0 from U2's record; loss from the preflight's `checks` (not written at a refused step);
LR in decay from 1072 on (warmup ends at 1000). The other three centred arms (null-matched 5.6e-4,
1e-4, 3e-4) never have P (plant A 0.11–0.46 at every populated step), as in U2b.

**Where each count stops** (the rule's description):

| count | last pass | stops at | ρ_mid there | the one-sided reading |
|---|---|---|---|---|
| (iii) uncentred, A ≥ 0.79 | 1465 | **1713**, monotone | 0.315 | with the rise |
| (ii) fixed τ 3e-3, A ≤ 0.10 | 1713 | **2002**, monotone | 0.431 | with the rise |
| (i) T2's arms as run (τ ≤ 1e-2), some arm with P | 2002 | 2340 (4 refusals = no P) or 4369 (left out) | 0.546 / 0.812 | with the rise, either way |
| **(i) with τ searched to 1e-1** (post hoc, beside) | **3738** (ρ 0.763) | **4369** (2340, 2735 left out: monotone; counted as no P: not monotone, fails at 2340, 2735 and passes at 3198, 3738) | 0.812 | with the rise |

The as-run row's 2340 comes from the search's limit at 3198 and 3738, so **the B-matched arm's
stop is read as 4369, last pass 3738**, with the post hoc label on it (the bound was raised after
seeing the refusal). Stated before output, and held: 917 passes and 9536 fails on (i)–(iii), so
each stop lies in (917, 9536]. None stops at 1072 or 1253, so no count stops before ρ rises; each
stops after it has started to rise (ρ_mid 0.20 at 1465). From 1465 on ρ, position 0's norm (4.4 →
23.8), the steps since warmup, the loss (3.99 → 3.66) **and each count's own operating point**
move together: the uncentred base rate 0.19 → 0.78, the B-matched τ 3.4e-3 → 1.1e-2, the
fixed-τ passing band moving up (below). So **the read cannot say which of them makes a count
stop**, and a fixed tolerance falling out of scale along training is one of the candidates
(`/challenge-pr` on #302, finding 3); the split's paths would not show that one. The order
(uncentred first, T2's B-matched arm last) describes series at the bars fixed for them; it does
not rank the counts, since each stop sits at its own bar.

**Beside, the fixed-τ grid** (τ 1e-3 … 1.2e-2 at dose 0.5, T2's full gate per cell): τ 2e-3–5e-3
pass at 917–1253; the passing band moves up and narrows along the rise (3e-3–5e-3 at 1465,
3e-3–8.82e-3 at 1713, 5e-3–8.82e-3 at 2002–2340, 8.82e-3 alone at 2735–3198), and no cell passes
from 3738 on (plant A ≥ 0.14 wherever B ≥ 0.4). Plant A's share is the same at both B doses in
every cell. The uncentred base rate rises from 0.19 to 0.78 across the rise.

**What it says.** Both Pythia counts behave as on Pythia into the rise and stop with it: the
uncentred count to ρ_mid 0.20 (1465), the fixed-τ centred arm to 0.32 (1713), T2's B-matched arm
to 0.76 (3738, post hoc on the search's bound; 0.43 at 2002 as run). They stop between 1713 and
4369, as ρ_mid goes 0.32 → 0.81. The trough's pass (U2b) does not end with the warmup: the counts
keep passing 465–2738 steps into the decay.

**What it does not say.** No cause: ρ, the norm, the loss, the LR schedule and each count's
operating point are not told apart (the design's one-sided reading). (i)'s stop as run depends on
how a refused derivation counts (2340 or 4369), which the rule did not fix; with the search
widened it is 4369 either way; all read "with the rise". No date, no V, no H2 label, `-67m` not
read. The split (paths) is not run.

**Parked** (why / cost / which decision it could change): (1) T2's early plateau (1253–3738, "T5
U2b") lies where T2's B-matched arm still passes (through 3738 with the search widened; 2340 and
2735 refused for want of a B dose), so much of the window T2 would have dated on is readable / a
T2 reopen (a design and a batch) / whether T2 reopens on 1253–3738, decided at T5's close-out.
(2) The uncentred count's sight of plant B lapses at 2340–2735 and is back from 3198 / a re-read of
the units, minutes / none in T5; for the split to look at. (3) `c2_seeds.b_matched_tol`'s τ ≤ 1e-2
bound is shared with C2 and T2; a miss outside it reads as "no τ" / widening it moves no recorded
number (every recorded τ is inside), but a re-derivation in the split or U4 could hit it again /
whether the bound widens or the refusal names the bound, before U4 derives arms (finding 1).

Re-run: `HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t2 preflight
--model 32m --trained-step <s> --out <dir>` for the 16 steps (4 exit on their refusals), then
`METS_DATA=<main>/data python -m trackT_timaeus.t5e read --pre <dir> --ref
<main>/data/trackT/t2u1/2026-10-29-preflight --ref-917 <main>/data/trackT/t5u2b/2026-11-01 --u2
<main>/data/trackT/t5u2/2026-10-30/reading.json`. Tests: `tests/test_trackT_t5e.py`.

## T5 U3's revision: toward the late rise (2026-11-01)

Blocked 130 (a), the user's, as recommended; docs only, nothing run. `design-T5.md` "U3 revised
toward the late rise" fixes, before any U3 output: `-32m` at 917, 1253, 1713, 2340, 3198, 4369,
6979, 9536 (ρ_mid 0.11 → 0.90, spread over the rise), plus step 1 (410m's 8 kept); as U3's
first read, T2's preflight at all 14 grid steps in (917, 9536) (917 and 9536 from their records,
which also gate it), read as T2's arms, the fixed-τ arms (U2b's post hoc sweep, now fixed) and the
uncentred count, with ρ, position 0's norm, the loss and the LR phase beside, and "where each count
stops" as a description; the split's paths then sit beside it. Stated before output: every
crossing lies in (917, 9536] unless not monotone; a crossing at 1072 (ρ_mid 0.116, inside the
trough's passing range, after the warmup ends and position 0's norm grows) reads "stops before ρ
rises"; from 1465 on ρ, the norm, the loss and the LR move together and the read cannot say which
makes a count stop. U4 is aimed at the rise; whether its cap reaches the rise depends on
Dodecahedron's batch (ROADMAP row; T1 Q3 "leaning"), and its design states whether it matches
steps, tokens or the LR schedule. `/challenge-pr` on #301: accept with changes, findings 1–4 taken
(1072, step 1 kept, the batch dependence, one home for the compute number). U2b's Parked 1 is taken
into U3's first read; Parked 2 stays with T5's close-out.

## T5 U2b: T2's planted gates in the trough (2026-11-01)

Blocked 126 (a), decided by Claude on the user's delegation ("Blocked 115 or 126 are yours"), as
recommended. Rule: `design-T5.md` "U2b", fixed and pushed before output (`cc66296`). Producer
`t2.py preflight --model 32m --trained-step <s>` (T2's preflight, trained population moved; code
`b01ad9e3e4d8`), reader `trackT_timaeus/t5d.py`. Records `data/trackT/t5u2b/2026-11-01/` (main
tree): `32m/preflight_plants.json` `75af52727e6e` (9536), `32m-step{491,671,917}/…` `03717e7823c3`,
`d0a99c815e6d`, `c96699cfde27`; `reading.json` `2ba04d6002bd`. A few minutes on the RTX 3080.
**Input:** `dodecahedron-32m` at steps 491, 671, 917 and 9536, its 3 re-inits, the 7 v1 prompts
(`976b62d677bc`), ≤ 512 tokens, float32, eager, TF32 off; 112 trained and 336 re-init units per step.

**Gate: passes.** `--trained-step 9536` reproduces unit 1's `derived`, `arms` and `gate` exactly
(max |Δ| 0.0, no mismatch).

**Stated before output** (from unit 1's record): the null-matched, 1e-4 and 3e-4 arms' re-init
rates (6.2 %, 8.9 %, 8.3 %) do not depend on the trained step, so those arms fail T2's full gate
at every step. The label reads the trained part P = (A ≤ 0.1 and B ≥ 0.4).

| step (ρ_mid, U2) | B's dose | arm | re-init rate | B share | **A share** | base rate (c) | full gate | P |
|---|---|---|---|---|---|---|---|---|
| 491 (0.124) | 0.5 | null-matched 5.6e-4 | 6.2 % | 1.00 | 0.12 | 0.35 | no | no |
| | | 1e-4 / 3e-4 | 8.9 / 8.3 % | 1.00 / 1.00 | 0.15 / 0.20 | 0.77 / 0.55 | no | no |
| | | **B-matched 3.61e-3** | 1.5 % | 0.81 | **0.00** | 0.12 | **yes** | **yes** |
| 671 (0.116) | 0.5 | null-matched | 6.2 % | 1.00 | 0.21 | 0.32 | no | no |
| | | 1e-4 / 3e-4 | 8.9 / 8.3 % | 1.00 / 1.00 | 0.27 / 0.32 | 0.73 / 0.50 | no | no |
| | | **B-matched 3.19e-3** | 1.5 % | 0.81 | **0.00** | 0.13 | **yes** | **yes** |
| 917 (0.111) | 0.5 | null-matched | 6.2 % | 1.00 | 0.27 | 0.35 | no | no |
| | | 1e-4 / 3e-4 | 8.9 / 8.3 % | 1.00 / 1.00 | 0.23 / 0.45 | 0.73 / 0.46 | no | no |
| | | **B-matched 3.37e-3** | 1.5 % | 0.77 | **0.00** | 0.14 | **yes** | **yes** |
| 9536 (0.903; unit 1) | 0.25 | B-matched 8.82e-3 | 0.3 % | 0.47 | 0.42 | 0.05 | no | no |

**Label: passes in the trough** (P on some arm at each of the 3 steps); the full gate passes at
every trough step, on the B-matched arm only. **Uncentred at 1e-3, beside:** plant A share 1.00,
0.97, 0.93 (**as on Pythia**, ≥ 0.79; 0.04 at 9536); trained base rate 0.26, 0.24, 0.19 (70m–GPT-2
0.24–0.50; 9536 0.75); re-init rate 6.2 % throughout. Mean loss 4.57, 4.36, 4.16 nats.

**What it says.** Where Dodecahedron's cloud is spread (ρ_mid 0.11–0.12), both counts behave as
on Pythia on the trained side: the uncentred count sees a common vector as it does on 70m, 160m and
GPT-2 small, and one centred arm (B-matched) keeps plant A out entirely while seeing B. At 9536
(ρ 0.90) neither does. So the counts' failure **co-occurs with** the concentrated cloud, which
answers T5's question as asked (`/challenge-pr` on #300, finding 2: not "tracks", a causal word;
the design's outcome table used it). By that table, U3 and U4 are revised toward the late rise
before U3 runs (Blocked 130).

**The B-matched arm's B share is not an observation** (`/challenge-pr` on #300, finding 1): its τ
is set so the centred count sees plant B as often as the uncentred one does, at a B dose chosen
so that rate is ≥ 0.4 (`t2.py` preflight), so "B ≥ 0.4" holds by construction and its pass rests
on plant A alone. **Post hoc, beside, chosen after the reading** (`t5d.py sweep`,
`posthoc_sweep.json` `6544e4d489c4`): with τ fixed across steps (1e-3, 2e-3, 3e-3, 5e-3, 8.82e-3,
1.2e-2) and B's dose fixed (0.25 or 0.5), plant A's share is ≤ 0.10 at every τ in the trough and
0.32–0.58 at 9536. At B dose 0.5, τ 2e-3–5e-3 passes T2's full gate at all three trough steps and
9536 fails at every τ, on plant A. So the leak's size depends on the step at the same arm, not
only on the arm's re-derivation.

**What it does not say.** T2's design gate needs *every* arm to pass; in the trough three of four
still fail: on the re-inits (fixed by Dodecahedron's init cloud, U1) and also on plant A (0.12–0.45
at the small τ 1e-4 to 5.6e-4, which hold 32–77 % of the trained units free). Above τ ≈ 1e-3 the
leak is ≤ 0.10 (the sweep). **The trough differs from 9536 in more than ρ** (finding 2): it is
inside Dodecahedron's 1000-step LR warmup, its loss is 4.2–4.6 nats against 3.66, and position 0's
norm over the median is 1.4–2.2 against 23.8; none of these is told apart from the cloud's
concentration here. Nothing here dates anything: the trough is 308–1253 and T2's early plateau
(1253–3738) spans ρ_mid 0.13 → 0.76, so a date there would be read while the cloud
re-concentrates. `-67m` not read.

**Parked** (why / cost / which decision it could change): (1) The gate at U3's checkpoints along
the late rise (1253, 1713, 2340, 3198, 4369, 9536), pairing where each count stops passing with
ρ_mid / minutes / whether U3's revision aims at a ρ threshold. Carried to Blocked 130; **taken
into U3's first read** (Blocked 130 (a), `design-T5.md` "U3 revised"). (2) Whether
T2 reopens on the trough / its plateau sits on the rise (above), so a date there is confounded by
the re-concentration / — / carried to Blocked 130 (c) and T5's close-out.

Re-run: `HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t2
preflight --model 32m --trained-step <s> --out <dir>` for s in 9536 491 671 917, then
`python -m trackT_timaeus.t5d read --pre <dir> --ref <main>/data/trackT/t2u1/2026-10-29-preflight`
(refuses the reading unless 9536 reproduces unit 1, and any record without its step or with
failed checks), then `… t5d sweep --pre <dir>` (post hoc). Tests: `tests/test_trackT_t5d.py`.
The `--trained-step` branch of `t2.py` has no unit test of its own (adding one now would move
`t2.code_sha` past the records'); the known answer at 9536 and each record's `trained_step` cover it.

## T5 U2: the cloud's trajectories through training (2026-10-30)

Blocked 125 (a), the user's, as recommended: U2 as frozen; the pair screen and the reverse
ablation go to U4's design as its first step. Rule: `design-T5.md` "U2" and "U2's details", fixed
and pushed before output (`727b89c`). Producer `trackT_timaeus/t5c.py` (code `3d425b32db90`);
records `data/trackT/t5u2/2026-10-30/` (main tree; 180 checkpoints, 474 s on the RTX 3080 after a
~10 min download, runner). **Input:** the 7 v1 prompts (`976b62d677bc`), ≤ 512 tokens, float32,
eager, TF32 off, torch 2.10.0, transformers 4.44.2; `dodecahedron-32m`, `-67m` at their 53 steps,
`pythia-70m`, `-160m`, `-410m` at C2's 18, `timaeus/gpt-2-small` at its init and 19 checkpoints
(stored bf16, upcast).

**Gate: passes.** Known answer (410m 0, 4000; 32m 1, 9536 against T2's `diag`): max |Δρ_ℓ| 0.0.
The rule's synthetic at s 0.0072 (32m step 9536's prompt spread): planted 0.15 "spreads" in 94.5 %
(≥ 90 %), flat in 0.0 % (≤ 5 %). Beside: at step 1's s (0.026) 95 % and 0 %; planted 0.10 0 %,
planted 0.20 100 %.

**Reading.** Δ = ρ_mid(first) − min, with its prompt bootstrap (5th, 95th percentiles).

| trajectory | first | min (step) | last | Δ | 5–95 % | label |
|---|---|---|---|---|---|---|
| **`dodecahedron-32m`** (scored) | 0.847 (1) | **0.111 (917)** | 0.903 (9536) | **0.736** | 0.669–0.794 | **spreads** |
| **`dodecahedron-67m`** (scored) | 0.834 (1) | **0.100 (671)** | 0.911 (9536) | **0.734** | 0.672–0.778 | **spreads** |
| **`gpt-2-small`** (scored) | 0.335 (init) | 0.167 (6000) | 0.19 (65000) | **0.168** | 0.153–0.182 | **spreads** |
| `pythia-70m` (beside) | 0.30 (0) | 0.19 (2000) | 0.37 | 0.101 | 0.073–0.130 | does not spread |
| `pythia-160m` (beside) | 0.38 (0) | 0.21 (2000) | 0.44 | 0.171 | 0.146–0.193 | spreads |
| `pythia-410m` (beside) | 0.46 (0) | 0.17 (4000) | 0.41 | 0.298 | 0.277–0.316 | spreads |

ρ_mid⁻⁰ gives the same label and the same minimum step on every trajectory (Δ within 0.004), so
position 0 does not decide any of it. **H1 through training:** neither branch applies (`gpt-2-small`'s
init 0.335 < 0.75, as U1 found).

**The shape** (ρ_mid; `reading.json` holds every series, with ρ_1:5, m and n0 at middle states):

| | early peak | trough | end | n0 (middle states) at trough → end |
|---|---|---|---|---|
| `-32m` | 1.00 at 30–40 | ≤ 0.15 at 308–1253 | 0.90 | 2.2 → 23.8 |
| `-67m` | 1.00 at 40–47 | ≤ 0.17 at 359–1072 | 0.91 | 3–4 → 13.9 |
| `pythia-410m` | 0.81 at 16 | 0.17–0.18 at 2000–4000 | 0.41 | 1.8–5.3 → 28.3 |
| `pythia-160m` | 0.80 at 32 | 0.21 at 2000 | 0.44 | 1.4 → 11.1 |
| `gpt-2-small` | 0.65 at 100 | 0.17 at 6000 | 0.19 | 1.4 → 8.5 |

**What it says.** Dodecahedron's cloud is not concentrated throughout training. T2 read it at step 1
and step 9536 only, and both sit at peaks. The trajectory has Pythia's shape: a rise to an early
peak, a long fall, a rise again. The minimum (0.111 at 917 on `-32m`, 0.100 at 671 on `-67m`) is
lower than any Pythia model's minimum read here (0.17–0.21). What differs is the two ends: the start (0.85 against
0.30–0.46, U1's init × architecture interaction) and the late rise (to 0.90 against 0.37–0.44),
which comes with the position-0 norm growing on every model. Since ρ_mid⁻⁰ rises the same way,
the late re-concentration is in the other tokens, not position 0's direction. `gpt-2-small`
(Pythia-160m's architecture under the flat init, biases on) spreads and stays spread to 65000.
**For H1** (description, not a label; `/challenge-pr` on #293, finding 1): the cloud is at
0.10–0.11 by 671–917, so the 0.90 at 9536 is not the init's concentration persisting. Whatever
holds it there arises after ~1253 in training. H1 as an account of the trained cloud is
not supported by this trajectory; U1's init effect explains step 1 only.
**D3:** D1's question (does a flat init keep the cloud concentrated through training) is not
answered, since `gpt-2-small` does not start concentrated (0.335); U4's flat-init arms carry it,
as under the design's "if it spreads" branch (finding 3).

**Post hoc, against the rule's choices** (`/challenge-pr` on #293, findings 2–3; not labels):
- At matched depth (ρ_1:5, the design's statistic across depths) the minima are `-32m` 0.085 at
  917, `-67m` 0.087 at 671, 410m **0.116 at 128**, 160m 0.175 and 70m 0.199 at 2000,
  `gpt-2-small` 0.176 at 6000. Dodecahedron's trough is still the lowest. 410m's minimum moves
  to 128, and Dodecahedron's grid (53 steps against 18) gives its minimum more chances to fall.
- "Spreads" is measured from the first checkpoint, so it reads the starting level. From each
  trajectory's early peak the falls are `-32m` 0.89, `-67m` 0.90, 410m 0.64, 160m 0.59, 70m 0.46
  (its "does not spread" is from 0.30 at step 0), `gpt-2-small` 0.48. `gpt-2-small`'s Δ 0.168
  sits within its step-to-step jitter of the bar.

**Not established.** Why Dodecahedron re-concentrates after ~1253 and Pythia much less (H2, the
data, LR or tokenizer: not told apart; U3 describes which paths write the shared direction, U4
tests). Whether the counts would pass in Dodecahedron's trough: not computed (Parked). `gpt-2-small`'s
Δ is 0.168, its 5th percentile 0.153, so "spreads" is near the bar; its data are unknown.

**Parked** (why / cost / which decision it could change): T2's counts at Dodecahedron's trough
steps (308–1253), where its cloud is as spread as 410m's at 2000–4000; T2 closed on the cloud
read at steps 1 and 9536 only, and its planted gate was never run in the trough / one unit (the
gate at 2–3 trough steps of `-32m`, minutes) / whether T2 reopens on the window where the cloud
is Pythia-like (Blocked 121's reason). Blocked 126; **done** as U2b (2026-11-01), "T5 U2b" above.

Re-run: `HF_HOME=<HDD_1TB>/mets_data/hf python -m trackT_timaeus.t5c fetch`, then
`HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t5c run --out <dir>`
(skips records present), then `METS_DATA=<main>/data python -m trackT_timaeus.t5c read --out <dir>`
(gate first; refuses the reading unless both checks pass; writes `reading.json`).

## T5 U1b: which part of the Llama makes the init matter (2026-10-30)

Added by Blocked 124 (a), the user's, as recommended (one unit beyond T5's budget, on the record).
Rule and gate: `design-T5.md` "U1b", frozen and pushed before output (`e92f8f6`). Producer
`trackT_timaeus/t5b.py` (code `d205f0b180e3`); records `data/trackT/t5u1b/2026-10-30/` (main tree;
360 models, 69 s on the RTX 3080). *Re-run after `/challenge-pr` on #292 (finding 1): the first run's
code `c8c454d4ea52` broke under transformers 4.57 (CI's) and left the rotary check out of `gate()`;
the fix changes no number (every cell median identical, labels identical).* **Input:** the 7 v1 prompts (`976b62d677bc`), ≤ 512 tokens,
float32, eager, TF32 off, torch 2.10.0, transformers 4.44.2; Llama at 32m's shape without biases
(U1's cell, U1's seeds for the base), Dodecahedron's tokenizer except `tok`; 18 cells × 20 seeds.

**Gate: passes on every check.**

| check | result | pass at |
|---|---|---|
| known answer: the base through `t5b`'s code against U1's `32m-llama-{P,F}-nob` | max \|Δρ_ℓ\| 0.0 over 40 models | ≤ 1e-6 |
| the init: per role, seed 0 of every cell against P's / F's formula | largest relative gap 0.10 % | ≤ 2 % |
| the rotary swap against NeoX's own partial rotary, permuted dims (`t5b.rotary_check`, in the gate; smoke test too, 4.44 and 4.57) | 8.6e-7 (full rotary differs by 34) | ≤ 1e-5 |
| planted null, `untied` | M −0.003 (P), −0.001 (F); τ 0.02, bar 0.1 | ≤ 0.05 |
| `all` against U1's `32m-neox-{P,F}-nob` | 0.418 vs 0.407 (P), 0.347 vs 0.354 (F) | ≤ 0.05 |
| positive control: `all` read as a swap | ΔE 0.434 of E0 0.363: "carries" | "carries" |

**Reading: no single part.** Medians of ρ_mid over 20 seeds. Base: P 0.478, F 0.842, **E0 = 0.363**.

| swap to NeoX's | P | F | M_P | M_F | E_k | ΔE_k | label |
|---|---|---|---|---|---|---|---|
| `ln` LayerNorm | 0.474 | 0.834 | −0.004 | −0.008 | 0.360 | 0.004 | does not |
| `par` parallel residual | 0.411 | 0.815 | −0.067 | −0.027 | 0.404 | −0.040 | does not |
| **`gelu` GeLU MLP at 4d** | 0.535 | 0.735 | **+0.057** | **−0.107** | 0.199 | **0.164** | **partly** |
| `rot` rotary on 0.25 | 0.470 | 0.839 | −0.009 | −0.003 | 0.369 | −0.006 | does not |
| `mha` 6 KV heads | 0.477 | 0.842 | −0.002 | 0.000 | 0.365 | −0.002 | does not |
| `tok` GPT-NeoX's tokenizer | 0.478 | 0.846 | −0.001 | +0.004 | 0.368 | −0.005 | does not |
| `untied` (null) | 0.476 | 0.841 | −0.003 | −0.001 | 0.365 | −0.002 | does not |
| `all` | 0.418 | 0.347 | −0.061 | −0.494 | −0.070 | 0.434 | carries |

- No swap "moves the cloud at both inits" (beside). `gelu` moves it in opposite directions under the
  two inits (up under P, down under F), closing the gap from both sides.
- **Not additive, by a scale-dependent amount** (beside, not scored; `/challenge-pr` on #292,
  finding 2): as shares of E0, the single swaps sum to 0.32 on ρ, 0.45 on logit ρ, 0.52 on atanh ρ,
  against 1.19, 1.17, 1.12 for all at once. All at once exceeds 1 partly because GPT-NeoX's own
  init effect is negative (E_all −0.07).
- **`gelu`'s share depends on the scale:** 0.452 on ρ ("partly" under the frozen rule, which reads
  ρ), 0.501 on logit ρ, 0.517 on atanh ρ (both would read "carries"). A seed bootstrap (2,000
  resamples of the 20 seeds per cell) puts the ρ share at 0.40–0.50 (5–95 %), at or above 0.5 in
  4 %. The label stays as frozen; the share is about a half on every scale.
- **The tokenizer is not R2's architecture factor:** `tok` moves nothing (|M| ≤ 0.004).
- ρ_1:5 beside (same depth, so no fairness change): base E 0.416, `gelu` 0.249, `par` 0.338, `all`
  −0.049. ρ_mid⁻⁰ within 0.003 of ρ_mid in every cell.

**What it says.** At init, in Dodecahedron's architecture, the flat init's concentrating effect does
not live in one part. The MLP's form (SwiGLU 1024 against GeLU 1536) carries about half of it
(0.45–0.52 by scale), and the norm, the rotary fraction, the KV heads and the tokenizer carry none
alone. The rest appears only when parts are swapped together. Which combination was not tested.

**Not established.** Which set of parts carries the rest. Whether one Llama part put into
GPT-NeoX is enough (the reverse ablation). Which of `gelu`'s three changes matters: the gate dropped,
SiLU for GeLU, width 1024 for 1536 (finding 3). Why it raises ρ under P; one untested candidate is that
GeLU of a centred Gaussian has a positive mean (1/(2√π) at unit variance), which the output map turns into one vector every
token shares, while SwiGLU's gated product has mean 0. Anything through training (U2, U4).

**Parked** (why / cost / which decision it could change): a full screen of the six swaps (all 64
subsets, so a three-part effect is not missed; finding 4) with `gelu` split into its three changes,
and the reverse ablation (NeoX-32m with one Llama part), all under both inits / about 10 GPU-minutes
and a unit of writing / which architecture contrast U4 trains (D3). Blocked 125.

Re-run: `HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t5b run --out <dir>`
(skips records present), then `METS_DATA=<main>/data python -m trackT_timaeus.t5b read --out <dir>`
(gate first, against U1's records; refuses the reading unless every check passes; writes `reading.json`).

## T5 U1: the cloud at step 0 and on random weights (2026-10-30)

Rules and gate as `design-T5.md` "U1" fixes them (frozen before output). Producer
`trackT_timaeus/t5.py` (commit `ad31ac4`, code `f6b205d00f5c`, pushed before the batch);
records `data/trackT/t5u1/2026-10-30/` (main tree; 685 models, 202 s on the RTX 3080).
**Input:** the 7 v1 prompts (`976b62d677bc`), ≤ 512 tokens, float32, eager, TF32 off; GPT-NeoX's
tokenizer (`pythia-410m` step143000) for every NeoX, Dodecahedron's (`step9536`) for every Llama;
the 5 real checkpoints below; 32 cells × 20 seeds, each model from its own CUDA generator (seed
hashed from cell and index, so cells are independent draws).

**Gate: passes on every check.**

| check | result | pass at |
|---|---|---|
| known answer (T2's `diag`: 410m steps 0, 4000; 32m steps 1, 9536) | max \|Δρ_ℓ\| 0.0 on all four | ≤ 1e-6 |
| init implementation: per-role SD, seed 0 of the cell against the real checkpoint | largest relative gap 0.08 % (70m), 0.04 %, 0.02 %, 0.04 % (32m step 1), 0.03 % (`gpt-2-small-init`); every real bias 0 | ≤ 2 % |
| real ρ_mid inside its cell's 20-seed range ± 0.03 | 70m 0.296 in [0.254, 0.319]; 160m 0.377 in [0.343, 0.416]; 410m 0.465 in [0.438, 0.530]; 32m step 1 0.847 in [0.817, 0.851]; `gpt-2-small-init` 0.335 in [0.317, 0.389] | yes, all five |
| planted null: \|E_bias\| over the 16 cells | max 0.012, so τ = 0.02 and the bar max(0.1, 2τ) = 0.1 | ≤ 0.05 |
| positive control: Llama-32m, F, attention only | ρ at the last state: median 0.999994, min 0.999989 (20 seeds) | median ≥ 0.99 |
| zeroed output projections, Llama-32m F | every ρ_ℓ equals state 0's to 5.5e-11; E(F − zeroed) 0.80 | 1e-6; ≥ 0.1 |

**R1a, `pythia-70m` step 0: ρ_1:5 = 0.278, falsified** (≤ 0.30). The per-block form of H1 (70m's
r 7.5 as large as Dodecahedron's) does not concentrate 70m's cloud. `pythia-160m` step 0 beside: 0.256.

**R1b, `timaeus/gpt-2-small-init-seed-5`: ρ_1:5 = 0.226, falsified in both forms** (r 15.4, r·L
184, the largest of all). Beside: its ρ_mid 0.335 against `pythia-160m` step 0's 0.377, the same
shape: the flat init sits slightly *lower*. **A GPT-NeoX under Dodecahedron's flat init is not
concentrated at init.**

**R2, the factorial: H1 at init partly.** Medians of ρ_mid over 20 seeds (biases pooled for E_init):

| shape (dose r_F/r_P) | NeoX P | NeoX F | **E_init NeoX** | Llama P | Llama F | **E_init Llama** |
|---|---|---|---|---|---|---|
| 70m (1.36) | 0.282 | 0.228 | −0.054 | 0.490 | 0.723 | **0.232** |
| 160m (3.33) | 0.384 | 0.364 | −0.020 | 0.531 | 0.766 | **0.234** |
| 410m (7.68) | 0.472 | 0.496 | 0.024 | 0.574 | 0.754 | **0.181** |
| 32m (3.14) | 0.406 | 0.348 | −0.058 | 0.475 | 0.842 | **0.366** |

- **Failed:** size in NeoX at 160m's, 410m's and 32m's shapes (E_init −0.06 … 0.02 against the bar
  0.1); the dose order in both architectures (70m's E_init not the smallest: in Llama 410m's is).
- **Held:** size in Llama at every shape (0.18–0.37); in Llama at 32m's shape, P brings ρ_mid
  **0.84 of the way** from Llama-F (0.842) to NeoX-P (0.406), against the rule's half.
- Beside (not scored): the architecture's effect, NeoX − Llama with inits pooled, −0.37, −0.28,
  −0.18, −0.30 at 70m's, 160m's, 410m's, 32m's shapes. ρ_mid⁻⁰ is within 0.003 of ρ_mid in every
  cell and real model: **position 0 is not what concentrates the cloud at init.** m_ℓ tracks ρ_ℓ
  (as it must at near-equal norms; n0 at middle states 1.18–1.24 on the real models).
- **Post hoc, at matched depth** (`/challenge-pr` on #291, finding 1; the frozen rule scores the
  order on ρ_mid, which `design-T5.md` itself calls unfair across depths, so the label stays
  "partly"). On ρ_1:5, E_init in Llama is **0.20, 0.32, 0.40, 0.42** at 70m's, 160m's, 410m's, 32m's
  shapes (doses 1.36, 3.33, 7.68, 3.14): 70m's the smallest, as H1's dose predicts; "halfway" 1.02.
  In NeoX −0.05, −0.02, 0.02, −0.05. The ρ_mid order failure in Llama is a depth artefact.

**What it says.** At init the cloud's concentration is an **interaction**: the flat init
concentrates a Llama (Dodecahedron's architecture), with the size ordered by H1's dose at matched
depth, and in a GPT-NeoX (Pythia's) it leaves ρ_mid *slightly lower* than Pythia's init at 3 of 4
shapes (by up to 0.058, 2.9 τ; R1b points the same way), the opposite of H1's direction
(`/challenge-pr` on #291, finding 3). Dodecahedron's step-1 ρ_mid 0.85 is reproduced by its own
architecture under 0.02 and mostly removed (to 0.48) by Pythia's init in that architecture. H1 as
recorded (the init alone, per block or over depth) is falsified on the two real checkpoints that
test it.

**Not established.** Which part of the Llama makes the init matter: RMSNorm against LayerNorm
(LayerNorm removes each token's feature mean), the sequential against the parallel residual,
SwiGLU, full rotary, tied embeddings, **and the tokenizer**, since in R2 the architecture factor
carries it (NeoX with GPT-NeoX's tokenizer, Llama with Dodecahedron's). Anything through training
(U2). H2 (biases are zero at init, so E_bias is null by construction and says nothing of H2).

**Consequence for U2 (D1).** U2's H1-through-training branch for `gpt-2-small` needs its init
ρ_mid ≥ 0.75; it is 0.335, so neither branch applies: the run does not start concentrated and
cannot show a flat init "keeping" the cloud concentrated. Whether to read it anyway, and whether
to find the Llama component first, is **Blocked 124**.

**Parked** (why / cost / which decision it could change): the Llama component ablation at init,
one component of Llama-32m swapped to NeoX's at a time (norm, residual, MLP, rotary fraction,
tying), **under both inits** so a part that moves the cloud whatever the init is told from one
that makes the init matter (`/challenge-pr` on #291, finding 2), plus one cell of Llama-32m fed
GPT-NeoX's tokenizer (random weights see the tokenizer only through the token sequence), 20 seeds
each, under a rule fixed before output / about 5 GPU-minutes / which factor U4 trains (D3) —
Blocked 124. *Carried to U1b (Blocked 124 (a)), done: "T5 U1b" above.*

**Environment** (`/challenge-pr` on #291, finding 4): the run's `meta.json` records torch only;
transformers was 4.44.2 (the `mets` env at the run, read from the env, not the record). `t5.py`
records it from now on (a later edit, so `read` on these records reports a different reader sha).

Re-run: `HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t5 run --out <dir>`
(skips records present), then `python -m trackT_timaeus.t5 read --out <dir>` (gate first; refuses
the reading unless every check passes; writes `reading.json`).

## T5's proposal: the init scales measured (2026-10-30)

Read from the weights before T5 opens (no forward pass; `trackT_timaeus/t5_init_sd.py`,
SD of each 2-D weight, its range over layers within 0.5 %). Closed forms and the per-block ratio
r: `tools/math_checks/t5_init.py` I1–I2 (sympy; what it does not prove is in its docstring).

| checkpoint | embedding, QKV / `c_attn`, MLP in | output projections (attention out, MLP out) | biases |
|---|---|---|---|
| `pythia-410m` step 0 | 0.0198 (small_init) | **0.0026** (wang_init) | all 0 |
| `pythia-70m` step 0 | 0.0280 | **0.0147** | all 0 |
| Mistral GPT-2 small (`alias-gpt2-small-x21`) checkpoint 0 | 0.0200 | **0.0041** (0.02 / √(2L)) | all 0 |
| `dodecahedron-32m` step 1 | 0.0200 | **0.0200** | none in the config (`-67m` too) |

Not read: `pythia-160m` (its formula only), any step-0 cloud. `timaeus/gpt-2-small`: "T5 U0" below.
Re-run: `python -I trackT_timaeus/t5_init_sd.py <checkpoint file>`; `python3 tools/math_checks/t5_init.py`.

## T5 U0: the design, the scan, `gpt-2-small`'s init (2026-10-30)

Opened by Blocked 122 (a), the user's. Design frozen 2026-10-30 (Blocked 123: D1–D3 all (a), the user's):
`trackT_timaeus/design-T5.md`; scan `docs/lit-T.md` §4. No forward pass.

**`timaeus/gpt-2-small`'s init is flat.** `timaeus/gpt-2-small-init-seed-5` (commit
`4f072cfb2d`, `model.safetensors` sha256 `6dc399b6ed6d`, float32, `HDD_1TB/mets_data/hf`), read by
`t5_init_sd.py`: **every 2-D weight at SD 0.0200** (embedding, QKV, attention out, MLP in and out,
`embed_out`; range over layers 0.01997–0.02004), every bias and LayerNorm shift 0, gains 1. The
config is GPT-NeoX, 12 × 768, 12 heads, rotary 0.25, parallel residual, untied, biases on, vocab
50,304: Pythia-160m's shape. So **it is Pythia-160m under Dodecahedron's flat init**, and its r is
the largest of the seven shapes, 15.4, 3.33× Pythia-160m's (r · L 184; `tools/math_checks/t5_init.py` I3).

**Its training arguments** (`checkpoint-100/training_args.bin`, read with `pickletools`, never
unpickled): peak LR 6e-4, cosine, warmup 1 % of the total steps (`warmup_steps` 0), weight decay
0.1, Adam (0.9, 0.95), clip 1.0, per-device batch 32, gradient accumulation 1, bf16, seed 5, one
epoch, `hub_model_id` `georgeyw/gpt-2-small`. Not in the file: the data, the GPU count, the
sequence length, the total steps. Checkpoints: `checkpoint-100` … `checkpoint-65000` every 100
(325 MB each); no step 0 except the init repo. **The init repo is the run's start**
(`/challenge-pr` on #290, finding 6): its weights correlate 0.9995–0.9997 with `checkpoint-100`'s
(`embed_in`, layer 0's QKV, layer 11's MLP out), SDs equal to 4 places. HF's GPT-NeoX default init
is 0.02 on every weight, which this matches.

Re-run: `HF_HOME=<HDD_1TB>/mets_data/hf python -c "from huggingface_hub import hf_hub_download as h;
print(h('timaeus/gpt-2-small-init-seed-5','model.safetensors'))"`, then `t5_init_sd.py` on the path.

## T2 on one page (closed 2026-10-30)

**Asked** (`design-T2.md`): on Dodecahedron (`-32m`, `-67m`; 53 log-spaced checkpoints, peak LR
3.5e-3, warmup 1000), a fourth schedule, does C2's energy break fall in step's band or the LR
dose's (tokens only if the batch is 524k)? **Closed at unit 1, before the scored batch:** Blocked
121 (i), the user's, as Claude recommended after `/challenge-pr` on #287. The exit's
"a recorded refusal" branch: no scored record, no reading. Downloaded: steps 1 and 9536 only
(0.76 GB). Writing only at close.

**Established (tier 1; `pythia-410m` 0–4000, Dodecahedron steps 1 and 9536 and 3 re-inits each,
v1's 7 prompts; no date read):**

1. **The producer reads a Llama-style model through C2's path.** `t2.py` equals A5's seed-0
   records on 410m exactly, its planted effect and null pass, and Dodecahedron's hidden states are
   bit-identical to transformers 5.2's ("T2 unit 1").
2. **Neither count dated on Pythia is that count on Dodecahedron.** Every centred arm fails its
   gate on both models (plant A 0.21–0.51 of the free units, bar ≤ 0.1). The uncentred count's
   plant A share is 0.04 / 0.07 against 0.79–0.98 on 70m, 160m and GPT-2 small.
3. **Beside (post hoc): the unit cloud is nearly one direction.** Middle-state mean cosine
   0.82–0.84 at init and 0.90–0.91 at 9536, against 410m's 0.47 and 0.17. The largest token
   norm, position 0 on every prompt, is 13.9–23.8× the median at 9536. Neither this nor position
   0's norm is shown to be why the gates fail (they were not told apart; "Not established").
4. **On paper: no reordering removes the small residual (L3, L4); whether one passes the gate was
   not computed** (`tools/math_checks/t2_llama_count.py`, L1–L4, sympy; heading corrected after
   `/challenge-pr` on #288, finding 1). Centring after normalising leaves the rows a mean squared
   norm (n − 1)(1 − ρ)/n, 0.09 at ρ 0.91 (L3; a mean over rows, not each row). Centring before
   normalising (e1r) cancels plant A exactly (L1), but at equal token norms e1r = e1c exactly
   (L4). So e1r differs only through the norm spread, mostly position 0, and a common direction on
   the sphere (A_s) reaches it by that route (L2). That A_s would fail on the models is a
   prediction from an isotropic toy (position 0 at 20× the median; `/challenge-pr` on #287,
   finding 2), not measured; option (ii)'s pre-check on step 9536's norm spread was not run.
5. **The grid separates step from cumulative LR under either batch; fraction does not split from
   step** (`tools/math_checks/t2_grid.py`, 6 of 6). That holds for a count whose Pythia window is
   no wider than E₁'s or the centred count's (their windows already abut at the shared step).

**Not established:**

- Any date on Dodecahedron, so C2's F1 (step or LR dose) gets no fourth schedule here. It stands
  as `PROJECT.md` §3.68–3.69 leave it.
- Whether e1r, or any count, would pass the gate on Dodecahedron: no e1r value was computed there.
- Why the counts fail their gates. The cloud's cosine, position 0's norm (plant A is 0.5 of the
  median norm, so it barely moves position 0), Llama's init (0.84 already on the re-inits), and
  architecture, tokenizer and data were not told apart. No plant-A share was recomputed with
  position 0 dropped (`/challenge-pr` on #288, finding 2).
- Why the cloud sits at cosine 0.84 from init (Llama's init, RMSNorm and the embeddings were not
  told apart), and whether step 9536 is the plateau's geometry.
- Dodecahedron's global batch (inferred from the card, never asked); tokens never scored.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| design | #284 | frozen 2026-10-29 (Blocked 119, D1–D4 (a), `ab6826c`) |
| 1. producer, gate, preflight | #285 | centred arms fail on both models; E₁ off its Pythia behaviour (Blocked 120) |
| 1b's design (Blocked 120 (b)) | #287 | proposed, never frozen; L4 found by its `/challenge-pr` (Blocked 121) |
| close-out | this | writing only (Blocked 121 (i)) |

Budget at close: 4 units + close-out allowed (Blocked 120 (b)); unit 1 and the two designs spent.

**Parked at close:**

| item | fate |
|---|---|
| unit 1's Parked 1 (plant A through e1r, preflight re-read) | taken into Blocked 120 (b), not frozen; **dropped**: L1 says plant A cannot fail on e1r, and L4 says what it would add (norm spread) |
| unit 1's Parked 2 (the uncentred re-init rate) | explained beside by the shared direction; **dropped**, nothing left to read |
| Blocked 121's other options: (ii) a synthetic pre-check of A_s and P0, (iii) freeze with plant P0, (iv) position 0 dropped on both sides | **not taken** (Blocked 121 (i)); written in `design-T2.md` "After `/challenge-pr` on #287" for any later Llama-style reading |
| the batch question to Timaeus (Blocked 118 (b), never sent) | **dropped**: T2 was its only consumer; the draft stays in `design-T2.md` |
| 3c's Parked 0 (the zero-rate refusal), fixed in `t2.py` only | still carried to C4 (`trackC_writeup/status-C.md`), as before T2 |

## T2 unit 1: the producer, the calibration gate, the preflight (2026-10-29)

Design frozen first (`ab6826c`, pushed before any Dodecahedron weight was fetched). **Input:** the
7 v1 prompts (`prompts_hash` `976b62d677bc`), ≤ 512 tokens, float32, eager, TF32 off, RTX 3080;
`timaeus/dodecahedron-32m` and `-67m` at steps 1 and 9536 only (`model.safetensors`, stored
float32; `HDD_1TB/mets_data/hf`, 0.76 GB); `pythia-410m` steps 0–4000 from `<main>/data/hf`.
Code `4fd84525dd37` (`trackT_timaeus/t2.py`, after `/challenge-pr` on #285; the first run at
`c6d56ab1085e` read the same numbers); outputs `data/trackT/t2u1/2026-10-29-{gate,known,preflight,
diag}/` (main tree): `gate.json` `0bf7c93a3c61`, `known_answer.json` `08a502d6a440`,
`preflight_plants.json` `9b141c42cff7` (32m), `672a740517bf` (67m), `diag_{410m,32m,67m}.json`
`56977086d9d0`, `7d465b91f6b9`, `c4a86406d558`. Resolved commits: 32m step 1 `5e1db647f7`, 9536
`2e976151cc`; 67m `f25fa3d8f4`, `e767b1e55c`.

| check | result |
|---|---|
| known answer (410m through this producer's path) | **pass**: e1 equals A5's seed-0 records exactly (max relative deviation 0.0, tokens equal); V 0 through 128, 19 at 256, 56 at 512, plateau 57, half-rise 512 |
| planted effect (crossing at each band's geometric middle) | **pass**, 4 scored schedules × 2 counts: inst. reads 16 / 35, cum. 65 / 141, frac. 193 / 359, step 263 / 574; each consistent set is the design's outcome-table row (fraction and step not split, as the design says); `joint` keeps the planted schedule. With tokens scored (tested, not in force) tok524k passes too, its centred crossing (2024) inside the early window, which reads 2002 beside the late 2340 (both tok524k's) |
| planted null (20,000 draws, crossings uniform in log step over (1, 4369], 4 cells independent) | "neither" 98.2 %; each scored schedule survives by chance 0.50–0.54 %; every "consistent with …" label ≤ 0.49 % (not a p). 28.5 % of draws have an unreadable cell (a crossing at 1–2), which `joint` passes over: about a third of each schedule's chance survival. **Over E₁ alone (2 cells): each schedule survives 5.4–6.6 %, "neither" 79.6 %** (`/challenge-pr` on #285, finding 2; `planted_null(counts=…)`) |
| the models load as trained | step 1's mean loss 9.77 / 9.80 nats (ln 16384 = 9.70), step 9536's 3.66 / 3.46 (placed bar < 6) |
| the state convention | 17 / 21 states; the first equals the embeddings and the last the final norm of the last block's output, exactly (max difference 0.0), as GPT-NeoX and GPT-2 |
| against transformers 5.2.0, the library that saved them (`t2_v5_check.py`, finding 4) | step 9536, both models, 7 prompts, CPU float32, 5.2's own tokenizer (tokenizers 0.22, merges read natively) and loader: **token ids equal and every hidden state bit-identical** (max difference 0.0) to this producer's 4.44.2 path |
| tokenizer | one `tokenizer.json` (`4edef5ce3cb6`) at steps 1 and 9536 on both; saved by tokenizers ≥ 0.20 (merges as pairs), read by the env's 0.19 with each pair joined (no piece holds a space; the producer refuses otherwise); no BOS, as Pythia |
| the re-inits as the init | HF's Llama init at 0.02: every weight's SD within 0.989–1.012 (32m) and 0.994–1.007 (67m) of step 1's |
| `tools.design_preflight` (both counts, both models) | `floor` only (E 9.18, 10.3 < 20), as every C2 arm: C2 and T2 compute no E. Reports `981d68aecc0e` (32m u), `5351ef132934` (32m c), `460a89a4c486` (67m u), `504e6f9c98f6` (67m c) |

**The centred arms (D2 (a)), gated as 3b: every arm fails, on both models** (the bars: re-init
rate ≤ 5 %, plant B share ≥ 0.4, plant A share ≤ 0.1; B's dose 0.25 on both):

| model | arm | re-init rate | B share | **A share** |
|---|---|---|---|---|
| 32m (112 trained units, 336 re-init) | null-matched 5.60e-4 | 6.2 % | 0.98 | **0.38** |
| | 1e-4 | 8.9 % | 0.98 | **0.21** |
| | 3e-4 | 8.3 % | 0.98 | **0.23** |
| | B-matched 8.82e-3 | 0.3 % | 0.47 | **0.43** |
| 67m (140, 420) | null-matched | **refused**: −3.05e-4 ≤ 0 | — | — |
| | 1e-4 | 5.5 % | 1.00 | **0.43** |
| | 3e-4 | 4.5 % | 1.00 | **0.49** |
| | B-matched 4.05e-3 | 0.2 % | 0.78 | **0.51** |

Plant A, a common vector at 0.5 of the median token norm that centring should remove, turns
0.21–0.51 of the centred count's free units on Dodecahedron (on 410m, 70m, 160m and GPT-2 small it
passed). So the centred count is not the token-centred statistic it was on those models, and
`TAU_ARMS` is not frozen: `run` refuses. By the design ("Any arm failing sends the design back"),
**Blocked 120**, the user's, before any scored record.

**The uncentred count at D2's 1e-3 departs from Pythia's and GPT-2's as far** (never gated, as
A5, 3b, 3c; the report above cited; the comparison is `/challenge-pr` on #285's finding 1, checked
against the C2 preflights in `data/trackC/c2u3{b,c}/2026-10-22-preflight/`):

| at 1e-3 | 70m (3b) | 160m (3b) | GPT-2 small (3c) | 32m | 67m |
|---|---|---|---|---|---|
| plant A share | 0.79 | 0.88 | 0.98 | **0.04** | **0.07** |
| re-init rate | 0.0 | 0.004 | 0.0 | **0.062** | **0.081** |
| trained base rate | 0.31 | 0.24 | 0.50 | **0.75** | **0.71** |

**Why, beside (post hoc; `t2.py diag`, steps no reading uses):** the unit cloud E₁ is read on is
nearly one direction on Dodecahedron, at init and trained. Mean cosine over the middle states
(median over states 1 … L−1, 7 prompts): 410m 0.47 at step 0 and **0.17** at 4000; 32m 0.84
(re-init), 0.85 (step 1), **0.90** (9536); 67m 0.82, 0.83, **0.91**. The largest token norm over
the median: 410m 1.2 and 5.4; 32m and 67m 1.2 at init, **23.8 and 13.9** at 9536, and it is
position 0 on every prompt at every middle state. So a common vector barely moves E₁ (already near
its maximum), and centring after normalising leaves a small residual that a common plant does move.
**E₁ on Dodecahedron is not the count that was dated on Pythia**: its "break", if any, would date
a change on a cloud at cosine 0.9, not Pythia's descent to 0.17. Why the init cloud is already at
0.84 (Llama's init, RMSNorm, tied embeddings) is not established.

**Defects fixed in this unit:** the null-matched derivation now refuses a zero re-init uncentred
rate (3c's Parked 0, carried here as the next unit deriving a centred arm; in `t2.py`, not in the
frozen `c2_seeds.py`, so C4 and C5 still need their own) and a tolerance ≤ 0 (67m's, found here).

**Not established:** the mechanism of the shared direction (the diagnostic shows it, not why);
whether step 9536 (end of the cosine decay) is the plateau's geometry (the scored grid is not run;
`/challenge-pr` on #285, question 4); anything about a date on Dodecahedron.

**Parked:** (1) Plant A read through `e1r` (raw states centred before normalising, in every record)
/ a preflight re-read, minutes / only if Blocked 120 (b) is taken; and (b) also needs a rule change
to the arm family, since the null-matched arm's re-init rate equals the uncentred one (6.2 %,
8.1 % > 5 %) by construction (finding 3). Carried to Blocked 120 (b); **taken 2026-10-30** as
`design-T2.md` "Blocked 120 (b)" D5–D7. Plant A cancels exactly from e1r (`tools/math_checks/
t2_llama_count.py` L1), so plant A_s is added as its leak test. (2) The uncentred count's re-init
rate / now explained beside by the shared direction / carried to Blocked 120; **taken** as D9 (E₁
beside, not in `joint`).

**Re-run:** `python -m trackT_timaeus.t2 gate --out <dir>`; `METS_DATA=<main>/data
HF_HOME=<main>/data/hf python -m trackT_timaeus.t2 known --out <dir>` (2 min); `HF_HOME=<HDD_1TB>/
mets_data/hf python -m trackT_timaeus.t2 fetch --steps 1 9536`, then `… preflight --model {32m,67m}
--out <dir>` (1–2 min each), then `python -m tools.design_preflight <dir>/<m>/preflight_input_{u,c}.json
--out …`; `… diag --model {410m,32m,67m} --out <dir>` (410m with `HF_HOME=<main>/data/hf`); the 5.2
check: `trackT_timaeus/t2_v5_check.py` docstring (a throwaway venv, `--system-site-packages` on the
`mets` env, `transformers==5.2.0`). Tests: `tests/test_trackT_t2.py` (pure, 17, one of them `read`
end to end from record files, finding 5); `python3 tools/math_checks/t2_grid.py` (T1–T6; T6 checks
the producer's constants and `predicted`).

## T2, the design (proposed 2026-10-29; frozen 2026-10-29)

`design-T2.md`, opened by Blocked 118 (b), the user's. Nothing downloaded or run; the bands and
outcome table come from `tools/math_checks/t2_grid.py` (6 of 6 pass; re-run:
`python3 tools/math_checks/t2_grid.py`). What it fixes and what it found on paper:

| item | value |
|---|---|
| grid | the 53 released steps 1–9536 (same on `-32m` and `-67m`); 9536 the preflight's trained step |
| step against cumulative LR | separated under either batch: E₁ 35–121 against 165–574 (shared 141); centred 65–225 against 308–1072 (shared 263) |
| fraction against step | not split: 0.7 × step; they part only at 420–574 (E₁), 917–1072 (centred) |
| tokens at 4.19M | overlaps cumulative and step (the step band one octave down); beside, not scored |
| tokens at 524k | scored only if a reply confirming it is quoted here before the first scored record (D4 (a), proposed) |
| plateau (D3 (a), proposed; revised after `/challenge-pr` on #284) | early 1253–3738 unless tokens is scored, then late 5108–8158; the union of both plateaus' dates. V rises late on all 9 A5 seeds (1000–4000 to 8000–16000, by 3–20), so a late window leans dates later |
| width-scaled dose (beside) | cumulative with LR × d_model / 1024 lands on step's band at 141–205 (E₁): a step reading excludes only the raw dose |
| downloads | about 6.7 GB (`-32m`) and 14.3 GB (`-67m`) for the 53 steps |

D1–D4 decided all (a) (Blocked 119, 2026-10-29). T2 closed 2026-10-30 at unit 1 ("T2 on one
page" above); the message to Timaeus was not sent.

## T1, the inventory (2026-10-29)

Sources: Hugging Face API (`/api/models`, `/refs`, `/paths-info`, `/commits`, raw cards and
yamls; 1,943 `timaeus/*` repos, 1,797 of them `rl-lm-pythia*` fine-tunes, not read further);
GitHub `timaeus-research/icl`, `timaeus-research/paper-rllcs-2024`; arXiv HTML of 2402.02364v3
(App. A.4.1, F.1.1, Table 1), 2507.21449 (App. E), 2606.22389v2, the abstract page of
2402.03698; the AF post "Timaeus in 2024" (2025-02-20). The arXiv HTML was read through a
summarising fetch that quotes the named sections, not as a full reading (not [R]); T3's design
re-reads A.4.1 in full before it freezes. No weights downloaded. No `.bin` or `.pth` pickle was
opened. Grades as `design-T1.md`, with "confirmed" from the papers resting on that fetch.

### Q1. A released model with a published LLC curve, deeper than 2 layers — **no, leaning**

| model | published LLC | depth | source |
|---|---|---|---|
| 2402.02364's LM (Q2) | whole-model, along training | 2 (and 1) | the paper (confirmed) |
| `timaeus/L4` = 2507.21449's model | **per head, one head shown** (layer 3, head 2), Fig. 4; sampler benchmark; γ, nβ, chains not stated | 4, attention-only | App. E matches `L4`'s yamls field for field (layers, d_model 256, 8 × 32, ctx 1024, vocab 5000, Shortformer, AdamW 1e-3, wd 0.05, 75,000 steps, batch 32, DSIR Pile); the paper names no repo, so leaning |
| anything else | none found | — | search, and the 2024 review (no curve) |

So the cheapest overlay (ROADMAP §7b, E_β over a published curve, no LLC estimated) has no
deeper model to use. `L4`'s one-head curve is not a whole-model control.

### Q2. 2402.02364's model and settings — **settings confirmed; model leaning `tetrahedron-3m-og`**

- **Settings** (App. A.4.1, Table 1, two-layer): SGLD, 20 chains × 200 steps, 1 draw a step,
  ε = 0.001, γ = 100, nβ = 21.7, minibatch m = 100, data lines 10m–11m of the DSIR-filtered Pile
  (10m–10.1m for validation); burn-in not stated for the LM. One-layer: ε = 0.003, γ = 300.
  nβ is m / log m (100 / ln 100 = 21.7), not n / log n (A.4.1). Table 1 also lists μ = 13m,
  "dataset size for gradient minibatches", beside A.4.1's 1M-line SGLD set; which set the
  minibatches are drawn from is not stated, and T3's design settles it before it freezes
  (`/challenge-pr` on #283, finding 2).
- **Model** (F.1.1): 2 layers attention-only, d_model 256, 8 heads, ctx 1024, Shortformer,
  vocab 5000, 3,355,016 parameters. The paper links only `github.com/timaeus-research/icl`.
- **Repo**: `icl`'s `sweeps/language/training-runs/tetrahedron-3m.yaml` is a 2-layer, 50,000-step
  run, seeds 0–4, data `timaeus/dsir-pile-10m` (public); its config defaults match F.1.1 (d_head
  32, batch 100, AdamW 1e-3, wd 0.05). `timaeus/tetrahedron-3m-og`: 501 checkpoints 0–50,000
  every 100, `losses.txt`, uploaded 2023-12-23, no card. Not confirmed: the sweep's checkpoint
  rule (log 100 + linear 2,000) does not give `-og`'s every-100 spacing, and which seed the
  published curve used is not stated.

### Q3. Dodecahedron's provenance — **mostly confirmed; the global batch leaning**

| item | answer | grade |
|---|---|---|
| `-v2` | the API redirects `dodecahedron-32m-v2` and `-67m-v2` (307) to `-32m` and `-67m`: a rename | confirmed |
| `-archived` | same weights: LFS sha256 equal at steps 1, 103, 491, 9536 (32m), one folder per step | confirmed at 4 of 53 |
| steps | `-32m` and `-67m` both carry the card's 53 steps (1 … 9536) | confirmed |
| `-32m` `sgd-relaxed` branch | "SGD-relaxed w0 fixed point (relaxation of step9536; fp32 patterning base)", 2026-06-16 | confirmed (commit) |
| `-pilot-dense` | **a different run**: weights differ from `-32m` at steps 1, 2, 103, 141; every step 1–160 released, `main` = step 500, no card; same architecture and tokenizer size | confirmed different; its schedule unknown |
| architecture | Llama, RMSNorm, RoPE θ 1e4, SiLU MLP, GQA (32m: 16 layers, 384, 6 heads / 2 KV; 67m: 20, 512, 8 / 4), tied embeddings, vocab 16,384 | confirmed (`config.json`) |
| LR | 3.5e-3, 1,000 warmup, cosine; AdamW β 0.9 / 0.95, wd 0.01, clip 1.0; bf16 | card only, leaning |
| tokens a step | the card's "64 micro × 4 accum × 2048" is 524,288; with its 8 × H100, 8 × 524,288 = 4,194,304, and 9,536 × 4,194,304 = 40.0B, the card's total (524k alone gives 5.0B) | **leaning 4.19M**: arithmetic on the card, no training log |

**What this does to T2's mapping.** The break window is A5's registered (128, 512] over 9
seeds (7 at 512, 2 at 256: `trackA_force/status-A.md` "A5 units 2 and 3"), not seed 0's
256–512 (corrected by `/challenge-pr` on #283, finding 1). Pythia: 2,097,152 tokens a step, LR
3e-4 over a 1,430-step warmup; Dodecahedron's warmup linear from 0. This table is the one home
of these crossing windows (the checkpoints *inside* each); the grid dates each schedule
*predicts*, which add the checkpoint just past a window's top (step's 574, cumulative's 141) and
so overlap at 141, are `design-T2.md`'s outcome table (`/challenge-pr` on #284, finding 3).

| reading | Dodecahedron steps | checkpoints inside |
|---|---|---|
| cumulative LR | (31, 125] | 35, 40, 47, 55, 65, 75, 88, 103, 121 |
| step | (128, 512] | 141, 165, 193, 225, 263, 308, 359, 420, 491 |
| tokens at 4.19M a step (leaning) | (64, 256] | 65 … 121 and 141 … 225: **overlaps both rows above** |
| tokens at 524k a step (the card's figure) | (512, 2048] | 574, 671, 785, 917, 1072, 1253, 1465, 1713, 2002 |

**What T2 can separate.** Cumulative LR from step under either batch (the windows abut at
125 / 128). Tokens only under 524k a step: at 4.19M, Dodecahedron takes twice Pythia's tokens
a step, so the token window is the step window one octave down and covers half of each of the
other two. So the batch decides whether T2 tests tokens at all (Blocked 118).
`-pilot-dense` covers steps 1–160 at every step, finer in the LR window, but it is a different
run with no stated schedule.

### Q4. `timaeus/gpt-2-small` — **architecture confirmed; data, schedule and LLC unknown**

GPT-NeoX (`config.json`): 12 layers, 768, 12 heads, rotary 0.25, parallel residual, vocab 50,304:
**Pythia-160m's shape, not GPT-2's**; initialised from `gpt-2-small-init-seed-5`; a copy of
`georgeyw/gpt-2-small` (2024-03-14). Checkpoints every 100, steps 100–65,000 (650);
`-log-spacing` 430 checkpoints to step 800,000. Data, LR and batch sit in `training_args.bin`
(a pickle, not opened). No card; no published LLC on it found.

### Q5. No published LLC along training on any Pythia model — **stands, leaning**

2402.03698 (withdrawn, merged into 2308.12108): deep linear networks only. 2606.22389v2: 3-layer
MLPs on MNIST, probes started at each checkpoint with a Gaussian tether, stated as not stationary
LLC estimates; code `github.com/yingjiacai/SIVE`. 2507.21449: a 4-layer attention-only model (Q1),
not Pythia. The 2024 review: LLC "scaled to models with billions of parameters", no model or
curve named. Still a search result: unindexed or unpublished work is not ruled out.

### Parked (why / cost / which decision it could change)

1. The `L0`–`L6` series (attention-only, d_model 256, `pile13m`, constant LR 1e-3, **no warmup**,
   75,000 steps, checkpoints dense early then every 300; the `H*`, `dm*` repos vary heads and
   width) / forward passes on 3M models / a break here could not be the warmup, which C2's F1
   cannot separate on Pythia. Carried to ROADMAP §7b.
2. `gpt-2-small` / its schedule needs the pickle read in isolation (`python -I`, own directory),
   then forward passes at 160m / a Pythia-architecture run on a different schedule. Carried to
   ROADMAP §7b.
3. `L4`'s one-head LLC curve as a partial positive control for a refined LLC / T3's cost / only
   if T4 goes per-head. Carried to ROADMAP §7b (T3).

## Corrections received
