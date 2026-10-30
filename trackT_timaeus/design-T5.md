<!-- trackT_timaeus/design-T5.md -->
# Track T — design: T5, why the Pythia counts fail on Dodecahedron (proposed 2026-10-30; not frozen)

Opened 2026-10-30 by `STATE.md` Blocked 122 (a), the user's, as recommended: all units, U4 after
U3. Row: `docs/ROADMAP.md` §7b T5, which states H1 and H2 and the recorded 70m prediction; this
file fixes the rules. Scan: `docs/lit-T.md` §4 (2026-10-30). U0's reads (no forward pass):
`status-T.md` "T5 U0". Tier 1, description throughout. **Nothing is frozen until you take D1–D3
(Blocked 123); no forward pass of T5 has run.**

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
| ρ_mid⁻⁰ | ρ_mid with position 0 dropped from every prompt | ROADMAP U2 |
| m_ℓ | ‖x̄_ℓ‖² / mean_i ‖x_iℓ‖² on raw states (the mean's share of the energy); also without position 0 | ROADMAP U1 |
| n0_ℓ | ‖x_0ℓ‖ / median_i ‖x_iℓ‖ | T2's norm ratio, at position 0 |

## The units and their rules (fixed before any output)

**U1, step 0 and random weights (H1 at init).** Three reads, in this order.

| read | input | rule |
|---|---|---|
| R1a (recorded in ROADMAP before any output) | `pythia-70m` step 0 | per-block H1 **supported** if ρ_mid ≥ 0.75, **falsified** if ≤ 0.47 (410m's step 0), else **not told apart**. `pythia-160m` step 0 beside |
| R1b (recorded here, before any output) | `timaeus/gpt-2-small-init-seed-5` | both forms mark it (r 15.4, r·L 184). H1 **supported** if ρ_mid ≥ 0.75, **falsified in both forms** if ≤ 0.47, else not told apart. Beside, the difference from `pythia-160m` step 0: the same architecture with only the init changed (the embedding SD 0.020 against 0.0228 too) |
| R2, the factorial | random weights: shapes {70m, 160m, 410m, 32m} × architecture {GPT-NeoX as Pythia, Llama as Dodecahedron} × init {**P** Pythia's: small_init on the embedding and input maps, wang_init on the output projections; **F** 0.02 on everything} × biases {on, off}, 20 seeds each (640 models) | below |

R2's details: each architecture keeps its own conventions at every shape (NeoX: LayerNorm, GeLU,
4d MLP, parallel residual, rotary 0.25, untied; Llama: RMSNorm, SwiGLU at 8d/3 rounded up to 64,
full rotary, tied, MHA except 2 KV heads at 32m's shape, as Dodecahedron). Norm gains 1, biases 0.
Per cell, the median ρ_mid over seeds. **E_init**(shape, arch) = median under F − median under P
(biases pooled, 40 a side). **E_bias**(shape, arch, init) = median with biases − without: **a
planted null by construction**, since biases are 0 at init. τ = max(0.02, the largest |E_bias| of
the 16).

- **H1 at init supported:** E_init ≥ max(0.1, 2τ) in all 8 shape × architecture cells, **and** in
  Llama at 32m's shape (Dodecahedron's own) init P brings ρ_mid to ≤ 0.655, the midpoint of 410m's
  step 0 (0.47) and Dodecahedron's (0.84). So the init alone takes Dodecahedron's architecture most of
  the way to Pythia's start.
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
MLP's output bias) and the rest. **The sink share** S is the sink path's part of the summed
component over tokens and blocks; **the bias share** B is the same for the biases.

- **H2 consistent:** S ≥ 0.5 on Dodecahedron and B ≥ 0.5 on Pythia at the trained checkpoints
  (≥ 1000 on Dodecahedron, ≥ 4000 on Pythia), **not supported** if S < 0.25 or B < 0.25, else not
  told apart. "Consistent", not "supported": it is a correlation in released weights. Only U4, or a
  released run differing in biases alone, tests H2.

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
| U2 | **the rule:** synthetic ρ_mid series on the 53-step grid with the 7 prompts' measured prompt spread (from step 9536): a planted descent of 0.15 reads "spreads", a flat series "does not spread" | planted descent "spreads" in ≥ 90 % of 1,000 draws; flat "spreads" in ≤ 5 % |
| U3 | **the split:** the parts sum to the block's write | max relative error ≤ 1e-4 |
| U3 | **planted bias:** a vector at 0.5 × the median token norm added as an o_proj bias in every block of `-32m` at 9536 | B of the planted component ≥ 0.9 |

Any failure goes back to you before that unit's reading.

## Decisions (each yours; recommendation first)

| # | choice | options | recommended, and why |
|---|---|---|---|
| D1 | where `timaeus/gpt-2-small`'s trajectory goes | (a) in U2: its init and about 20 log-spaced checkpoints of its 650 (100 … 65,000; about 6.5 GB at 325 MB each), read under U2's "spreads" rule, and its result sizes U4; (b) in U5 as the ROADMAP row has it, described with no rule | **(a).** It is the only released run with a passing model's architecture (Pythia-160m) under the flat init (its data and LR differ, unknown), so it answers "does a flat init keep the cloud concentrated through training" before any GPU-day is spent. It cannot separate H2, since it has biases. (b) keeps the plan as proposed but reads the strongest control without a rule |
| D2 | U1's bands | (a) as above: 0.75 / 0.47 for R1a, R1b; E_init ≥ max(0.1, 2τ); 0.655 for "most of the way"; (b) bands from the 20-seed spread alone (E_init beyond the 95 % seed interval), no fixed size | **(a).** A fixed size says the init matters by an amount that bears on the counts. (b) would call a 0.03 shift "supported" with 20 seeds. The 0.75 band is a judgement: within 0.09 of 0.84, about the 32m–67m spread at init |
| D3 | U4 | (a) designed after U2 and D1's read, own Blocked item, 3 GPU-day cap; (b) committed now to the ROADMAP's 4 arms ({biases on, off} × {0.02, depth-scaled}) at 32m's shape | **(a).** If `gpt-2-small` does not spread, the flat-init arm is half answered and U4 can drop to the biases contrast; if it spreads, U4's flat arms carry the question. (b) spends up to 3 GPU-days before that is known |

## Exit and budget

**Exit** (ROADMAP): H1 and H2 each read supported, not supported or not told apart (H2 at most
"consistent" without U4) under the rules above, and "does Dodecahedron's cloud spread at any
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
