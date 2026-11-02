<!-- trackT_timaeus/design-T5.md -->
# Track T — design: T5, why the Pythia counts fail on Dodecahedron (proposed 2026-10-30; frozen 2026-10-30; U3 revised 2026-11-01)

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

**U1b, the Llama's parts at init (added 2026-10-30 by Blocked 124 (a), the user's, as
recommended: one unit beyond the budget, on the record; rule and gate fixed here before any U1b
output).** U1 found the flat init concentrates a Llama and not a GPT-NeoX (`status-T.md` "T5 U1").
U1b asks which part of the Llama makes the init matter, so U4 trains that part. **Base:** U1's
cell Llama-32m without biases (Dodecahedron's architecture: RMSNorm, sequential residual, SwiGLU
1024, full rotary, 2 KV heads, tied, its tokenizer), under P and under F. **Swaps**, one at a time
to GPT-NeoX's, each under both inits (`/challenge-pr` on #291, finding 2), 20 seeds:

| swap | to | how |
|---|---|---|
| `ln` | LayerNorm (gain 1, shift 0, eps 1e-5) for RMSNorm, every norm | modules replaced |
| `par` | parallel residual, x + attn(ln₁ x) + mlp(ln₂ x) | the block's forward |
| `gelu` | GeLU MLP at 4d (1536) for SwiGLU 1024 | module replaced; P puts small_init on its input, wang_init on its output |
| `rot` | rotary on 0.25 of each head (8 of 32 pairs, NeoX's frequencies, base 10⁴) for all 32 | pairs 8 … 31 at frequency 0. A permutation of head dimensions from NeoX's layout, so the same in distribution under i.i.d. weights (checked: a test against NeoX's `apply_rotary_pos_emb`) |
| `mha` | 6 KV heads for 2 | config |
| `tok` | GPT-NeoX's tokenizer (vocab 50,304) for Dodecahedron's | random weights see it only through the token sequence |
| `untied` | untied unembedding | **planted null**: the unembedding never reaches a hidden state |
| `all` | every swap above at once | GPT-NeoX-32m without biases, in distribution (U1's cell) |

Statistic: ρ_mid (every cell has 32m's depth, so ρ_mid is fair); ρ_1:5 beside. Per cell, the median
over seeds. B_i = the base's median under init i; **E0 = B_F − B_P**; per swap k, M_k,i = med(k, i) −
B_i, **E_k = med(k, F) − med(k, P)** and **ΔE_k = E0 − E_k** (what the swap removes of the init's
effect). τ = max(0.02, |M_untied,P|, |M_untied,F|); bar = max(0.1, 2τ), as U1's.

- Swap k **carries the init's effect** if ΔE_k ≥ E0 / 2; **partly** if bar ≤ ΔE_k < E0 / 2; **does
  not** if |ΔE_k| < bar; **amplifies** if ΔE_k ≤ −bar.
- Beside, not a label of the init: k **moves the cloud at both inits** if M_k,P and M_k,F share a
  sign and both |M| ≥ bar (a part that moves the cloud whatever the init).
- The unit's reading names the swaps that carry; **none carries** reads "no single part": the
  effect is spread over parts or needs two together. Beside: Σ_k ΔE_k against ΔE_all (additivity).

Gate (before the reading; any failure goes back to the user):

| check | pass |
|---|---|
| **known answer:** the base cells, built through U1b's code with no swap, against U1's records `32m-llama-{P,F}-nob` (same seeds) | max \|Δρ_ℓ\| ≤ 1e-6 over the 40 models |
| **the init:** per role, seed 0's weight SD in every cell against P's or F's formula | within 2 % |
| **the rotary swap:** attention logits of the patched Llama against NeoX's partial rotary on permuted dimensions (frozen as a test; put in `gate()` as `t5b.rotary_check` after `/challenge-pr` on #292, finding 1, which found it missing there; the bar unchanged) | ≤ 1e-5 |
| **planted null:** \|M_untied,i\| at both inits | ≤ 0.05 |
| **`all` is GPT-NeoX:** \|med(all, i) − U1's `32m-neox-i-nob`\| at both inits | ≤ 0.05 |
| **positive control at the rule's size:** `all` read as a swap | "carries" |

Not in U1b: the reverse ablation (one Llama part into GPT-NeoX, whether a part suffices), pairs of
swaps, biases (null at init, U1). It cannot say what training does with the part (U4).

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

**U2's details (fixed 2026-10-30 before any U2 output; Blocked 125 (a), the user's, as
recommended: U2 next as frozen, the pair screen and the reverse ablation go to U4's design as its
first step).** Producer `trackT_timaeus/t5c.py`.

| item | fixed as |
|---|---|
| trajectories | `dodecahedron-32m`, `-67m` at `t2.GRID` and 9536 (53 steps); `pythia-70m`, `-160m`, `-410m` at C2's 18 steps (0 … 143000); `timaeus/gpt-2-small`: its init (`-init-seed-5`, as step 0) and `checkpoint-N` for N in 100, 200, 300, 400, 500, 800, 1100, 1500, 2100, 3000, 4300, 6000, 8400, 11800, 16600, 23400, 32900, 46200, 65000 (`geomspace(100, 65000, 20)` rounded to 100, distinct: 19), each with its own `config.json`, GPT-NeoX's tokenizer as in U1, stored weights upcast to float32 |
| first checkpoint | the earliest: Dodecahedron step 1, Pythia step 0, `gpt-2-small` its init |
| the bootstrap | one resample of the 7 prompts (with replacement) is used at every step of a trajectory: ρ_ℓ averaged over the resampled prompts, ρ_mid per step, Δ_b = first − min; `numpy.random.default_rng(20261030)` |
| the label | "spreads" is checked first (Δ ≥ 0.15 and 5th percentile > 0.05), then "does not spread" (95th < 0.15), else "not told apart" |
| scored | Dodecahedron `-32m` and `-67m`, each its own label (the exit's question), and `gpt-2-small` (D1). Pythia's three are read under the rule beside, a real descent (410m's 0.47 → 0.17, T2) the rule should see; not gating |
| H1 through training | `gpt-2-small`'s init ρ_mid is 0.335 (U1), below 0.75, so **neither branch applies**: its label is reported and sizes U4 (D3), with no H1 reading |
| gate, the rule's synthetic | per prompt and step ρ_p(t) = μ(t) + N(0, s²), i.i.d.; s = the SD (ddof 1) of the 7 prompts' own ρ_mid at `-32m` step 9536 (from U2's record; U1's equal record gives 0.0072); flat μ = 0.90; planted μ(t) = 0.90 − 0.15 · min(1, ln t / ln 1072), linear in log step to 1072, flat after (15 steps at the bottom); 1,000 draws, each with its own 1,000-resample bootstrap through the reading's code. Beside, not gating: s from step 1 (0.026), and planted 0.10 and 0.20 |

What the synthetic implies, stated before it runs: the planted size equals the bar, so under
i.i.d. noise a planted descent reads Δ ≥ 0.15 when the first point lies above the least of the 15
at the bottom, probability about 1 − 1/16 = 0.94 whatever s is. The gate's 90 % is then met by the
trough's length, and a planted dip at 0.15 lasting one step would read "spreads" about half the
time. The bar applies to descents that stay down.

**U2b, T2's planted gates in the trough (added 2026-11-01 by Blocked 126 (a), decided by
Claude on the user's delegation ("Blocked 115 or 126 are yours"), as recommended; a second
extension of T5's budget, one unit; fixed before any U2b output).** The question as asked: do the
counts fail where the cloud is concentrated? Producer: `t2.py preflight --trained-step <s>`, T2's
preflight with its trained population moved from 9536 to step s and nothing else changed.

| item | fixed as |
|---|---|
| input | `dodecahedron-32m` at s ∈ {491, 671, 917} (U2's ρ_mid 0.124, 0.116, 0.111; position 0's norm over the median 1.4–2.2), the same 3 re-inits, the 7 v1 prompts, plants and seed as T2 unit 1 |
| known answer (gate) | `--trained-step 9536` reproduces unit 1's `preflight_plants.json` (`9b141c42cff7`): `derived`, `arms` and every `gate` number equal to 1e-12. Fails → refuse the reading |
| arms | T2's family, derived as frozen at each s: null-matched (from the re-inits, so the same τ at every s), 1e-4, 3e-4, B-matched (re-derived at s, with B's dose) |
| **stated before output** | the null part of the gate reads the re-inits only, so it does not move with s: on 32m the null-matched, 1e-4 and 3e-4 arms have re-init rates 6.2 %, 8.9 %, 8.3 % > 5 % (unit 1's record) and **fail T2's full gate at every s**. Only the B-matched arm can pass it in full. So the full gate is reported, and the label reads the part that depends on the trained cloud |
| per s, per arm | T2's full gate as frozen (null ≤ 5 %, B ≥ 0.4, A ≤ 0.1), and **the trained part** P = (A share ≤ 0.1 and B share ≥ 0.4) |
| **the label (centred)** | **passes in the trough** if at each of the 3 steps some arm has P; **fails in the trough** if no arm has P at any step; else **mixed**. Beside: whether the full gate passes at any s (the B-matched arm only) |
| the uncentred count, beside | at D2's 1e-3: plant A share, against 70m, 160m and GPT-2 small's 0.79–0.98 and 9536's 0.04: "as on Pythia" if ≥ 0.79, "as at 9536" if ≤ 0.10, else "between". No label (T2 never gated it) |
| what it changes | **passes**: the leak of a common vector into the centred count tracks the trained cloud's concentration, so the count fails where the cloud is concentrated (T5's question as asked); U3 and U4 are revised toward the late rise (1253 → 9536) as their own Blocked item, before U3 runs. **fails**: the leak is there at ρ ≈ 0.11 too, so concentration is not its cause on its own; U3 as frozen. **mixed**: described, U3 as frozen |
| not done | no date, no V, no scored grid, `-67m` not read (its null-matched τ refused in unit 1) |
| exit, budget | the label computed or the gate refuses; one unit, minutes of GPU |

**U3, where the shared direction is written (H2).** **Dodecahedron `-32m` at step 1 and at 917,
1253, 1713, 2340, 3198, 4369, 6979, 9536, on the late rise** (revised 2026-11-01, below; first
frozen as 1, ~10, ~100, ~300, ~1000, ~3000, ~6000, 9536); 410m at 0, 16, 128, 512, 1000, 4000,
32000, 143000 (kept). u_ℓ = x̄_ℓ / ‖x̄_ℓ‖ at the block's output. For each block and token, the write's component
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

**U3 revised toward the late rise (2026-11-01, Blocked 130 (a), the user's, as recommended;
fixed before any U3 output).** U2b read "passes in the trough", so by U2b's table U3 aims at the
rise, where Dodecahedron's cloud re-concentrates and the counts stop passing: ρ_mid 0.11 at 917,
0.13 at 1253, 0.90 at 9536 (U2's record). The 8 steps sit on U2's grid, spread over ρ_mid:

| step | 917 | 1253 | 1713 | 2340 | 3198 | 4369 | 6979 | 9536 |
|---|---|---|---|---|---|---|---|---|
| ρ_mid (U2) | 0.111 | 0.129 | 0.315 | 0.546 | 0.706 | 0.812 | 0.888 | 0.903 |
| position 0's norm / median, middle states (U2) | 2.2 | 4.1 | 4.8 | 9.0 | 11.3 | 15.1 | 18.4 | 23.8 |
| LR phase (linear warmup to 3.5e-3 over 1000; cosine after, floor unknown) | warmup, 0.92 | decay | decay | decay | decay | decay | decay | end of decay |

**Step 1 stays in the split** (revised after `/challenge-pr` on #301, finding 2): U1 and U2 read
its ρ, not its paths, and it is the one comparison of what writes the init cloud against the trained
one; 410m keeps its step 0. So the split reads 9 Dodecahedron steps. `-67m` not read.

**The first read takes every grid step on the rise, not only the split's 8** (revised after
`/challenge-pr` on #301, finding 1): from 491 to 1072 ρ_mid stays at 0.111–0.127 while position
0's norm goes 1.4 → 3.5 and the warmup ends at 1000, so the steps 1072 and 1253 can tell a count
that stops before ρ rises from one that stops with it. The extra steps (U2's ρ_mid): 1072 0.116,
1465 0.198, 2002 0.431, 2735 0.639, 3738 0.763, 5108 0.847, 5970 0.872, 8158 0.900.

| item | fixed as |
|---|---|
| **U3's first read: the gates along the rise** (U2b's Parked 1) | `t2.py preflight --model 32m --trained-step s` at the 14 grid steps in (917, 9536): 1072, 1253, 1465, 1713, 2002, 2340, 2735, 3198, 3738, 4369, 5108, 5970, 6979, 8158 (new, minutes each); 917 from U2b's record, 9536 from T2 unit 1's; T2's family, re-inits, prompts, plants and seed as in U2b |
| its gate | `--trained-step 9536` reproduces unit 1's `preflight_plants.json` (`9b141c42cff7`) and `--trained-step 917` reproduces U2b's (`c96699cfde27`): `derived`, `arms` and every `gate` number equal to 1e-12. Fails → refuse the reading |
| per step | (i) T2's four centred arms as derived at s: the full gate and P (as U2b); (ii) **the fixed-τ arms, fixed now** (post hoc in U2b): τ ∈ {1e-3, 2e-3, 3e-3, 5e-3, 8.82e-3, 1.2e-2} × B dose {0.25, 0.5}, plant A and B shares (`t5d sweep`'s code over the 16 steps); (iii) the uncentred count at 1e-3: plant A share, binned as U2b ("as on Pythia" ≥ 0.79, "as at 9536" ≤ 0.10, else "between") |
| beside each step | ρ_mid and ρ_mid⁻⁰ and position 0's norm (U2's record), the mean loss (the preflight's `checks`), the LR phase (table above) |
| **where each count stops** (description, no label) | for (i), the last step where some arm has P, and the first step from which no arm has P at it and every later step; for (ii) at τ 3e-3, dose 0.5 (U2b's passing band), the first step from which plant A > 0.10 at it and every later step; for (iii), the first step from which A < 0.79 at it and every later step. A series that crosses back is reported "not monotone", with its steps |
| stated before output | 917 passes and 9536 fails on every one of (i)–(iii) (the records), so each crossing lies in (917, 9536] unless a series is not monotone |
| **the one-sided reading** (fixed now, after `/challenge-pr` on #301, finding 1) | a crossing at **1072** reads **"stops before ρ rises"**: its ρ_mid (0.116) lies inside the range where the counts passed in U2b (0.111–0.124 at 491–917), so ρ_mid's rise is not needed for that stop; the end of the warmup and position 0's norm (2.2 → 3.5) are not told apart from each other. A crossing at **1253** (ρ_mid 0.129, just above that range) reads "at the rise's start", not told apart. A crossing at 1465 or later reads "with the rise": ρ_mid, position 0's norm and the steps since warmup all rise together there and the loss falls, so **the read cannot say which makes it stop** |
| then the split | as above, at its 9 Dodecahedron steps, each rise step's paths reported beside its row of the first read, so where the shared write grows sits beside where each count stops |
| not done | no date, no V, no H2 label, `-67m` not read |

**U4, the causal arm.** Its design is written after U2 and the `gpt-2-small` read, as a section
of this file and its own Blocked item, before any training (D3). Cap: 3 GPU-days. **Aimed at the
rise** (Blocked 130 (a)): its arms are read on whether they re-concentrate after their trough, not
at init. The cap's reach in Dodecahedron steps is the ROADMAP row's compute line (`docs/ROADMAP.md`
§7b T5). It depends on the batch, which T1 Q3 grades only "leaning" 4.19M tokens a step: at that
batch an arm stops before the rise (1253 → 9536), and at the card's 524k it reaches into the rise.
Whether the rise is fixed in steps, in tokens or by the LR schedule is not known, so a run with
another batch or schedule may not rise where Dodecahedron did (`/challenge-pr` on #301, finding 3).
U4's design states which of the three it matches, with U3's crossings in hand.

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
| U3 | **the first read's known answers** (added 2026-11-01, Blocked 130 (a)): `--trained-step` 9536 and 917 reproduce T2 unit 1's and U2b's records | `derived`, `arms`, `gate` equal to 1e-12 |
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
GPU-days, **plus U1b** (extended on the record, Blocked 124 (a), 2026-10-30), **plus U2b** (Blocked 126 (a),
2026-11-01, on the user's delegation), **plus U3's revision** (Blocked 130 (a), 2026-11-01, the
user's: one docs unit; U3 then runs in budget, its first read minutes of GPU). **No lean:** no step-0 cloud of 70m, 160m or `gpt-2-small` has been computed; the only
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
- U2's synthetic plants the descent at the bar and draws noise i.i.d. over steps: its pass is
  set by the trough's length (15 steps), not by s; a real prompt's offset persists over steps, which
  this noise model does not have. A plant at 0.20 is reported beside.
- U3's revision (Blocked 130 (a)): the split's rise steps are spread over U2's ρ_mid, not over log
  step, so the rise's middle (1713–3198) gets four of eight and the trough one (917); the first read
  takes all 14 grid steps on the rise. Rejected: Blocked 130 (b), U3 as frozen (~1000, ~3000,
  ~6000, 9536 already on the rise, but it overrides U2b's outcome table); fixing a ρ threshold for
  "stops" (no derivation picks one; the crossing is read at the step instead).
- "Where each count stops" is one-sided: only a crossing at 1072 separates ρ_mid from the rest (it
  stops before ρ rises). From 1465 on, ρ, position 0's norm, the loss and the LR move together and
  the read cannot rank them; its use there is to place the split's paths beside it and to tell U4
  where its arms must reach. The first version claimed they moved together everywhere, which 1072
  contradicts (`/challenge-pr` on #301, finding 1).
- The fixed-τ arms (ii) were chosen after U2b's reading (its post hoc sweep). They are fixed here
  before any rise output, but the choice of 3e-3 at dose 0.5 as the crossing's arm is informed by
  where U2b's trough passed.
