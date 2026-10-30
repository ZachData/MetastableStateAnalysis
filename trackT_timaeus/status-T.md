<!-- trackT_timaeus/status-T.md -->
# Track T — STATUS

<!-- phase-card -->
## Card

- **Question:** Do Timaeus's released checkpoints (Dodecahedron, a fourth training schedule) and the local learning coefficient give C2's energy break a second schedule and a degeneracy curve to sit beside, and which released model and settings make the LLC a calibrated instrument here?
- **Inputs:** T1 ran nothing: it read Hugging Face metadata (cards, config files, branch and file listings, LFS hashes) and papers through a summarising fetch, listed in `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"; T2 unit 1 ran `dodecahedron-32m` and `-67m` at steps 1 and 9536, 3 re-inits each, and `pythia-410m` steps 0–4000, the 7 v1 prompts (`976b62d677bc`), code `4fd84525dd37`, listed in `trackT_timaeus/status-T.md` "T2 unit 1"; T5 U0 read weights only (`timaeus/gpt-2-small-init-seed-5` `4f072cfb2d`, no forward pass), listed in `trackT_timaeus/status-T.md` "T5 U0"; T5 U1 ran `pythia-70m`, `-160m`, `-410m` step 0, `dodecahedron-32m` step 1, `timaeus/gpt-2-small-init-seed-5` and 680 random models (32 cells × 20 seeds, 2 controls × 20), the 7 v1 prompts (`976b62d677bc`), code `f6b205d00f5c`, listed in `trackT_timaeus/status-T.md` "T5 U1"; T5 U1b ran 360 random models (Llama at 32m's shape without biases, base and 8 swaps × 2 inits × 20 seeds), the 7 v1 prompts (`976b62d677bc`), code `d205f0b180e3`, listed in `trackT_timaeus/status-T.md` "T5 U1b"; T5 U2 ran `dodecahedron-32m`, `-67m` at their 53 steps, `pythia-70m`, `-160m`, `-410m` at C2's 18 and `timaeus/gpt-2-small` at its init and 19 checkpoints, the 7 v1 prompts (`976b62d677bc`), code `3d425b32db90`, listed in `trackT_timaeus/status-T.md` "T5 U2"
- **Results:**
  - Opened 2026-10-29 (Blocked 117 (a), the user's: T1 and T2 first) with T1's questions, grades and a one-unit budget fixed before reading — `trackT_timaeus/design-T1.md`
  - T1 closed at its budget: Dodecahedron's 40B tokens fits a global batch of 8 × 524k (per GPU), not 524k, and under it T2's token window overlaps both its step and its LR-dose windows, so T2 separates LR dose from step either way and tests tokens only if the batch is 524k; `-v2` is a rename and `-archived` the same weights, but `-pilot-dense` is a different run; 2402.02364's LLC settings are read, its model is most likely `tetrahedron-3m-og`; no released model deeper than 2 layers has a published whole-model LLC curve (`L4` has one head's, 2507.21449); `gpt-2-small` is a GPT-NeoX of Pythia-160m's shape; the Pythia negative stands for the two papers read — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
  - T2 unit 1 (design frozen on D1–D4 (a)): the producer reproduces A5's seed-0 records on `pythia-410m` exactly and its planted effect and null pass, and Dodecahedron loads as trained, its hidden states bit-identical to transformers 5.2's; but every centred arm fails its gate on both models (plant A turns 0.21–0.51 of the free units), and the uncentred count departs as far (plant A 0.04–0.07 against 0.79–0.98 on 70m, 160m, GPT-2); beside, not shown to be the cause, the unit cloud is nearly one direction (middle-state mean cosine 0.82–0.84 at init, 0.90–0.91 at step 9536, against 410m's 0.47 and 0.17); so neither count is the one dated on Pythia, and the design goes back to the user (Blocked 120) — `trackT_timaeus/status-T.md` "T2 unit 1"
  - T2 closed 2026-10-30 at unit 1, before any scored record (Blocked 121 (i), the user's, as recommended): a count fit to Llama was proposed (Blocked 120 (b)) and not frozen, since at equal token norms the raw-centred count equals the centred one exactly (L4) and differs from it only through the token-norm spread (L2), while centring leaves the rows a mean squared norm of about 0.09 at cosine 0.9 (L3); whether it would pass was not computed; no Dodecahedron date, and C2's F1 gets no fourth schedule from it — `trackT_timaeus/status-T.md` "T2 on one page (closed 2026-10-30)"
  - T5 opened 2026-10-30 (Blocked 122 (a)); U0, weights only: Timaeus's `gpt-2-small` starts from 0.02 on every weight with zero biases, so it is Pythia-160m's architecture under Dodecahedron's flat init, a released run whose architecture matches a model that passed and whose init does not (its data and the rest of its schedule unknown) — `trackT_timaeus/status-T.md` "T5 U0"
  - T5 U1 (gate passes): the init alone does not concentrate the cloud: `pythia-70m` step 0 (mean cosine over states 1–5, 0.28) and `gpt-2-small`'s flat init (0.23) both sit at or below the falsifying 0.30; on random weights the flat init raises the middle-state mean cosine in Llama by 0.18–0.37 at every shape (at matched depth 0.20–0.42, smallest at 70m's dose, as H1 predicts), and in GPT-NeoX not at all (−0.06 to 0.02, lower than Pythia's init at 3 of 4 shapes), and Pythia's init takes a Llama at 32m's shape 0.84 of the way to a GPT-NeoX's, so H1 at init reads "partly": an interaction of init and architecture, the architecture's part not identified (its factor carries the tokenizer) — `trackT_timaeus/status-T.md` "T5 U1"
  - T5 U1b (Blocked 124 (a), one unit beyond the budget; gate passes): no single part of the Llama carries the init's effect at 32m's shape: swapping its SwiGLU MLP for GeLU at 4d removes 0.16 of the 0.36 (about half on logit or atanh scales), and LayerNorm, the parallel residual, 0.25 rotary, 6 KV heads and GPT-NeoX's tokenizer each remove under 0.01 (the tokenizer is not R2's architecture factor); all six at once remove 0.43, so much of the effect appears only with parts together (its size depends on the scale of ρ), which set not tested — `trackT_timaeus/status-T.md` "T5 U1b"
  - T5 U2 (Blocked 125 (a); gate passes): Dodecahedron's cloud is not concentrated throughout training: on both models the middle-state mean cosine rises to 1.00 by step 30–47, falls to 0.10–0.11 at 671–917 (lower than any Pythia minimum read, 0.17–0.21) and rises again to 0.90 by 9536, so T2 read it only at its two peaks; Pythia has the same rise, fall and rise, with lower ends; `gpt-2-small` spreads (0.34 → 0.17) and stays spread; position 0 decides none of it — `trackT_timaeus/status-T.md` "T5 U2"
- **Superseded / wrong:**
  - The ROADMAP's T2 windows (from #282) read the card's per-GPU batch as the global one and used seed 0's break bracket, not A5's nine-seed (128, 512]; re-derived in T1 — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
- **Registry:** none, because Track T is tier 1
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - Which seed of `tetrahedron-3m` the paper's curve is from, and why `-og`'s checkpoint spacing differs from the `icl` sweep's — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
  - Why the Pythia counts fail on Dodecahedron (H1 the init, H2 no biases): T5, opened 2026-10-30, design frozen 2026-10-30 (Blocked 123, D1–D3 (a)); U1 done (H1 at init partly); U1b done (no single part); U2 done (the cloud spreads in the middle of training and re-concentrates late); U3 waits on Blocked 126 — `trackT_timaeus/status-T.md` "T5 U2"
- **After Phase 10:**
  - The `L0`–`L6` attention-only depth series at constant LR with no warmup, as a schedule without the warmup confound *(forward pass, small models)*
  - `gpt-2-small` as a Pythia-160m-shaped run checkpointed every 100 steps *(forward pass; its data and schedule unread)*
- **Reviewed:** 2026-10-30 · body `e059d23386`
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
| T5 U2. the trajectories | **done 2026-10-30** (Blocked 125 (a)): gate passes; both Dodecahedron models spread (Δ 0.73–0.74, trough 0.10–0.11 at 671–917) and re-concentrate to 0.90; `gpt-2-small` spreads (0.17); U3 waits on Blocked 126 | "T5 U2" below |

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
peak, a long fall, a rise again. The trough (0.10–0.12, at 491–1072) is lower than any Pythia
model's minimum read here (0.17–0.21). What differs is the two ends: the start (0.85 against
0.30–0.46, U1's init × architecture interaction) and the late rise (to 0.90 against 0.37–0.44),
which comes with the position-0 norm growing on every model. Since ρ_mid⁻⁰ rises the same way,
the late re-concentration is in the other tokens, not position 0's direction. `gpt-2-small`
(Pythia-160m's architecture under the flat init, biases on) spreads and stays spread to 65000.
D3: U4's flat-init arms carry its question (the design's "if it spreads" branch).

**Not established.** Why Dodecahedron re-concentrates after ~1253 and Pythia much less (H2, the
data, LR or tokenizer: not told apart; U3 describes which paths write the shared direction, U4
tests). Whether the counts would pass in Dodecahedron's trough: not computed (Parked). `gpt-2-small`'s
Δ is 0.168, its 5th percentile 0.153, so "spreads" is near the bar; its data are unknown.

**Parked** (why / cost / which decision it could change): T2's counts at Dodecahedron's trough
steps (308–1253), where its cloud is as spread as 410m's at 2000–4000; T2 closed on the cloud
read at steps 1 and 9536 only, and its planted gate was never run in the trough / one unit (the
gate at 2–3 trough steps of `-32m`, minutes) / whether T2 reopens on the window where the cloud
is Pythia-like (Blocked 121's reason). Blocked 126.

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
