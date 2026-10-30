<!-- trackT_timaeus/design-T5.md -->
# Track T — design: T5, why the Pythia counts fail on Dodecahedron (proposed 2026-10-30; frozen 2026-10-30)

Opened 2026-10-30 by `STATE.md` Blocked 122 (a), the user's, as recommended: all units, U4 after
U3. Row: `docs/ROADMAP.md` §7b T5, which states H1 and H2 and the recorded 70m prediction; this
file fixes the rules. Scan: `docs/lit-T.md` §4 (2026-10-30). U0's reads (no forward pass):
`status-T.md` "T5 U0". Tier 1, description throughout. **Frozen 2026-10-30 on the user's calls, D1–D3
all (a) (Blocked 123), as revised after `/challenge-pr` on #290, before any forward pass of T5**;
the "Decisions" table keeps the options as written.

## What T5 is for

T2 unit 1 found the unit cloud on Dodecahedron at mean cosine ρ 0.84 at init and 0.90 at step
9536, against 410m's 0.47 → 0.17. That is why the Pythia counts do not carry over, and it is not
explained (`status-T.md` "T2 on one page", "Not established"). If the cloud's concentration is
Pythia's init leaving (H1), C2's break may partly be an init effect, a limit the paper rewrite needs.

**What it cannot show.** U1–U3 and U5 are forward passes on released or random weights: they say
where the cloud sits and which paths write its shared direction, not what training would do under
another init or with biases. Only U4 (training from scratch) or a released run that differs in one
factor tests a cause. Dodecahedron and Pythia also differ in data, tokenizer, LR and norm (ROADMAP
row, "Not told apart").

**U0's read changes the plan** (`status-T.md` "T5 U0"): `timaeus/gpt-2-small` starts from
`timaeus/gpt-2-small-init-seed-5`, which has **0.02 on every weight, the output projections
included, and zero biases**. It is Pythia-160m's GPT-NeoX shape with biases, under Dodecahedron's flat
init, so it is a released "biases on, flat init" run with 650 checkpoints (steps 100–65,000). Its
r is the largest of all shapes, 15.4 (`tools/math_checks/t5_init.py` I3). Its data, the GPU count
and the sequence length are unknown (peak LR 6e-4, cosine, warmup 1 % of the total steps).

## The statistics (fixed now)

Input to every read: the 7 v1 prompts (`prompts_hash` `976b62d677bc`, checked at run time), each
model's own tokenizer, ≤ 512 tokens, float32, eager, GPU. States are as T2 defines them (every
state HF returns; state 0 the embeddings, the last after the final norm). The **middle states** are
1 … L−1.

| symbol | definition | as |
|---|---|---|
| ρ_ℓ | mean pairwise cosine of the unit token vectors at state ℓ, averaged over the 7 prompts | T2's `diag` (`c.mean_cos(c.unit(h))`) |
| **ρ_mid** | the median of ρ_ℓ over the middle states | T2's 0.84 / 0.90 / 0.47 / 0.17 |
| **ρ_1:5** | the mean of ρ_ℓ over states 1 … 5, the same block count on every model (70m has 6 blocks); used wherever depths differ | added after `/challenge-pr` on #290 (finding 1): at init ρ_ℓ rises with ℓ (32m step 1: 0.30, 0.63, 0.75, 0.79, 0.82), so ρ_mid reads a 6-block model at its 3rd state and a 16-block one at its 8th. T2's records give ρ_1:5 = 0.656 (32m step 1), 0.661 (32m re-init), 0.666 / 0.641 (67m), **0.252** (410m step 0) |
| ρ_mid⁻⁰ | ρ_mid with position 0 dropped from every prompt | ROADMAP U2 |
| m_ℓ | ‖x̄_ℓ‖² / mean_i ‖x_iℓ‖² on raw states (the mean's share of the energy); also without position 0 | ROADMAP U1 |
| n0_ℓ | ‖x_0ℓ‖ / median_i ‖x_iℓ‖ | T2's norm ratio, at position 0 |

## The units and their rules (fixed before any output)

**U1, step 0 and random weights (H1 at init).** Three reads, in this order.

| read | input | rule |
|---|---|---|
| R1a (recorded in ROADMAP before any output; **restated at matched depth** after `/challenge-pr` on #290, finding 1, still before any 70m cloud) | `pythia-70m` step 0 | per-block H1 **supported** if ρ_1:5 ≥ 0.60, **falsified** if ≤ 0.30, else **not told apart**. The bands: Dodecahedron's four init values (0.641–0.666) less their spread (0.025), and 410m's step 0 (0.252) plus 0.05. `pythia-160m` step 0 beside |
| R1b (recorded here, before any output) | `timaeus/gpt-2-small-init-seed-5` | both forms mark it (r 15.4, r·L 184). H1 **supported** if ρ_1:5 ≥ 0.60, **falsified in both forms** if ≤ 0.30, else not told apart. Beside, ρ_mid against `pythia-160m` step 0 (the same shape, so ρ_mid is fair): only the init changed (the embedding SD 0.020 against 0.0228 too). **Its weight:** HF's GPT-NeoX default init is 0.02 on every weight, so this is one more seed of R2's NeoX-160m-F cell (finding 6); it is read because it is the start of a trained run (U2), confirmed by its weights' correlation 0.9995 with `checkpoint-100`'s (3 tensors; `status-T.md` "T5 U0") |
| R2, the factorial | random weights: shapes {70m, 160m, 410m, 32m} × architecture {GPT-NeoX as Pythia, Llama as Dodecahedron} × init {**P** Pythia's: small_init on the embedding and input maps, wang_init on the output projections; **F** 0.02 on everything} × biases {on, off}, 20 seeds each (640 models) | below |

R2's details: each architecture keeps its own conventions at every shape (NeoX: LayerNorm, GeLU,
4d MLP, parallel residual, rotary 0.25, untied; Llama: RMSNorm, SwiGLU at 8d/3 rounded up to 64,
full rotary, tied, MHA except 2 KV heads at 32m's shape, as Dodecahedron). Norm gains 1, biases 0.
Per cell, the median ρ_mid over seeds. **E_init**(shape, arch) = median under F − median under P
(biases pooled, 40 a side). **E_bias**(shape, arch, init) = median with biases − without: **a
planted null by construction**, since biases are 0 at init. τ = max(0.02, the largest |E_bias| of
the 16).

**H1's dose differs by shape** (revised after `/challenge-pr` on #290, finding 2). Under P the
embedding and input maps share small_init, so r_F / r_P = 0.02 / wang_init(d, L): **1.36 at 70m's
shape**, 3.33 at 160m's, 7.68 at 410m's, 3.14 at 32m's (`tools/math_checks/t5_init.py` I4). H1
predicts a small E_init at 70m's shape, so the rule asks for size only where the dose is ≥ 3, and a
dose order at 70m's.

- **H1 at init supported:** E_init ≥ max(0.1, 2τ) in the 6 cells at 160m's, 410m's and 32m's shapes;
  E_init at 70m's shape the smallest of the four shapes, in each architecture; **and** in Llama at
  32m's shape (Dodecahedron's own), init P brings ρ_mid at least halfway from Llama-F toward NeoX-P
  at the same shape (Pythia's start at that depth; finding 1: no comparison across depths).
- **Not supported:** E_init < 2τ in Llama at 32m's shape.
- **Partly** otherwise, naming which condition failed. The architecture's effect (F and P pooled)
  is reported beside, not scored.

**U2, the trajectories.** Every statistic above per state at all 53 checkpoints of `-32m` and
`-67m`, plus references at their on-disk steps: `pythia-70m`, `-160m`, `-410m` (C2's 18 steps);
`timaeus/gpt-2-small` under D1. **"Spreads"** for one trajectory: Δ = ρ_mid(first checkpoint) −
min over t of ρ_mid(t), with a 7-prompt bootstrap (1,000 resamples, the minimum retaken in each).
**Spreads** if Δ ≥ 0.15 (half 410m's 0 → 4000 descent) and the bootstrap's 5th percentile is >
0.05; **does not spread** if the 95th percentile is < 0.15; else not told apart. The reading uses
ρ_mid (position 0 kept, as T2); ρ_mid⁻⁰ is read under the same rule beside, and a disagreement is
reported as position 0's.

- **H1 through training**, from `gpt-2-small` (D1): if its init ρ_mid is ≥ 0.75 (R1b) and it
  **spreads**, a flat init does not keep the cloud concentrated in a NeoX with biases: H1 explains
  at most init, and Dodecahedron's 0.90 at 9536 is something else (H2, or the rest). If it **does
  not spread**, then H1 is consistent through training, and H2 is not needed for that run.

**U3, where the shared direction is written (H2).** At 8 checkpoints a model (Dodecahedron 1, ~10,
~100, ~300, ~1000, ~3000, ~6000, 9536 on its grid; 410m 0, 16, 128, 512, 1000, 4000, 32000,
143000). u_ℓ = x̄_ℓ / ‖x̄_ℓ‖ at the block's output. For each block and token, the write's component
on u_ℓ is split exactly. On attention it splits into the sink path (a_i0 · W_O v_0) and the rest.
On Pythia it also splits into the bias terms (b_O, W_O b_V, since attention rows sum to 1, and the
MLP's output bias) and the rest. The MLP's write also splits into its token-constant part (its
mean over tokens and prompts) and the rest, on both models. Per path, two numbers: **its share** of
the summed component over tokens and blocks, and **its size**, that component over the mean squared
token norm, so the two models compare in size and not only in which parameter writes (finding 4).

**U3 describes; it reads no H2 label** (revised after `/challenge-pr` on #290, findings 3–4). A
share rule could read "H2 consistent" with the shared write equally large on both models, and would
miss a bias-free MLP writing a near-constant vector. Pythia's linear biases also give no sink slot:
the key bias cancels in the softmax, and the value and output biases are a constant write. So
Sun et al.'s explicit attention bias, an extra key–value slot, is not what Pythia has (finding 3;
`docs/lit-T.md` §4). **H2 is read only by U4**, or by a released run differing in biases alone.

**U4, the causal arm.** Its design is written after U2 and the `gpt-2-small` read, as a section
of this file and its own Blocked item, before any training (D3). Cap: 3 GPU-days.

**U5, the controls.** U2's statistics on Mistral GPT-2 small from checkpoint 0 (on disk), on
410m's seeds 1–9 at the 18 steps (on disk), and on `gpt-2-small` if D1 (b). They describe, with no rule.

## Calibration gate (each unit, before its reading)

| unit | check | pass |
|---|---|---|
| all | **known answer:** the producer's ρ_ℓ on 410m steps 0 and 4000 and 32m steps 1 and 9536 equal T2's `diag_{410m,32m}.json` | max difference ≤ 1e-6 |
| U1 | **the init implementation:** NeoX under P at 70m's, 160m's and 410m's shapes, Llama under F at 32m's, NeoX under F at 160m's; the weight SDs match the measured (`status-T.md` "T5's proposal", "T5 U0") | within 2 %; and the real checkpoint's ρ_mid within ±0.03 of its cell's 20-seed range (pythia-70m / -160m / -410m step 0, 32m step 1, `gpt-2-small-init`) |
| U1 | **planted null:** E_bias, biases being 0 at init | every \|E_bias\| ≤ 0.05; else the seed noise is too large to read E_init, and U1 refuses |
| U1 | **positive control, a known collapse:** Llama at 32m's shape under F, attention only (no MLP, no residual), the norms kept: pure attention loses rank (Dong et al. 2021, 2103.03404) | ρ at the last state ≥ 0.99 |
| U1 | **known-answer contrast at the rule's size** (finding 5): Llama at 32m's shape under F with every output projection (`o_proj`, `down_proj`) zeroed, 20 seeds, so every state is the embeddings | its ρ_ℓ equal state 0's to 1e-6, and E(F − zeroed) ≥ max(0.1, 2τ): the factorial sees an effect of the size the rule asks for |
| U2 | **the rule:** synthetic ρ_mid series on the 53-step grid with the 7 prompts' measured prompt spread (from step 9536): a planted descent of 0.15 reads "spreads", a flat series "does not spread" | planted descent "spreads" in ≥ 90 % of 1,000 draws; flat "spreads" in ≤ 5 % |
| U3 | **the split:** the parts sum to the block's write | max relative error ≤ 1e-4 |
| U3 | **planted bias:** a vector at 0.5 × the median token norm added as an o_proj bias in every block of `-32m` at 9536 | the bias path's share of the planted component ≥ 0.9 |

Any failure goes back to you before that unit's reading.

## Decisions (each yours; recommendation first)

| # | choice | options | recommended, and why |
|---|---|---|---|
| D1 | where `timaeus/gpt-2-small`'s trajectory goes | (a) in U2: its init and about 20 log-spaced checkpoints of its 650 (100 … 65,000; about 6.5 GB at 325 MB each), read under U2's "spreads" rule, and its result sizes U4; (b) in U5 as the ROADMAP row has it, described with no rule | **(a).** It is the only released run with a passing model's architecture (Pythia-160m) under the flat init (its data and LR differ, unknown), so it answers "does a flat init keep the cloud concentrated through training" before any GPU-day is spent. It cannot separate H2, since it has biases. (b) keeps the plan as proposed but reads the strongest control without a rule |
| D2 | U1's bands | (a) as above: ρ_1:5 ≥ 0.60 / ≤ 0.30 for R1a, R1b; E_init ≥ max(0.1, 2τ) where the dose is ≥ 3, 70m's E_init the smallest; Llama-32m under P at least halfway to NeoX-P; (b) bands from the 20-seed spread alone (E_init beyond the 95 % seed interval), no fixed size | **(a).** A fixed size says the init matters by an amount that bears on the counts. (b) would call a 0.03 shift "supported" with 20 seeds. The R1 bands come from T2's records (Dodecahedron's four init values less their spread; 410m's plus 0.05); 0.1 is a judgement. Revised after `/challenge-pr` on #290 (findings 1, 2): the first proposal's 0.75 on ρ_mid sat where a Dodecahedron-like 70m would land, and its reason (the 32m–67m spread) was wrong (0.016, not 0.09) |
| D3 | U4 | (a) designed after U2 and D1's read, own Blocked item, 3 GPU-day cap; (b) committed now to the ROADMAP's 4 arms ({biases on, off} × {0.02, depth-scaled}) at 32m's shape | **(a).** If `gpt-2-small` does not spread, the flat-init arm is half answered and U4 can drop to the biases contrast; if it spreads, U4's flat arms carry the question. (b) spends up to 3 GPU-days before that is known |

## Exit and budget

**Exit** (ROADMAP): H1 and H2 each read supported, not supported or not told apart (H2 only from
U4; "not told apart" if U4 does not run) under the rules above, and "does Dodecahedron's cloud spread at any
checkpoint" answered on the 53. **Budget: U0 (this design and scan) + U1–U5 + close-out**, U4 ≤ 3
GPU-days. **No lean:** no step-0 cloud of 70m, 160m or `gpt-2-small` has been computed; the only
clouds read are T2's (410m 0, 4000; Dodecahedron 1, 9536 and the re-inits).

## Worth challenging (for the PR)

- The bands (D2): 0.75 and 0.655 are judgement, not derived; a reviewer may prefer bands from a
  model of the init (none is derived: `t5_init.py`'s r is a rough order, "NOT proved").
- ρ_mid as the one statistic: it is T2's diagnostic, not either count. A cloud could pass the
  counts at ρ 0.6 and fail at 0.5 for another reason; T5 explains the cloud, not the gates.
- The bias factor as U1's planted null: it is null in distribution, not seed for seed, because
  creating a biased Linear consumes the RNG differently.
- U3's split credits W_O b_V to the bias, though it reaches tokens through attention. The rows sum
  to 1, so it is constant, but a reviewer may count it as attention.
- Promoting `gpt-2-small` into U2 (D1) and letting it size U4: its data are unknown, so a spread
  there may be the data, not the init's leaving.
