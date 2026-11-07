<!-- trackT_timaeus/design-T5.md -->
# Track T — design: T5, why the Pythia counts fail on Dodecahedron (proposed 2026-10-30; frozen 2026-10-30; U3 revised 2026-11-01; U4 proposed 2026-11-02, frozen 2026-11-03)

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

**U3's split, the details (fixed 2026-11-02 before any split output; code `trackT_timaeus/t5f.py`).**
The rows above leave these open; each is a choice, not derived.

| item | fixed as |
|---|---|
| u_ℓ | per prompt: the mean over its tokens, position 0 included, of block ℓ's raw output (the residual; at the last block before the final norm, so not T2's last state), normalised |
| the paths | attn_sink Σ_h a^h_i0 W_O^h W_V^h x̃_0 (bias-free value); attn_bias b_O + W_O b_V; attn_rest the keys j ≥ 1; mlp_bias the output bias; mlp_const the bias-free MLP write's mean over every token of the 7 prompts, per block; mlp_rest the bias-free write less it. Six paths, summing to the block's write |
| share | Σ over blocks, prompts and tokens of the path's component on u / the same sum of the write's; per block beside; position 0 dropped beside |
| size | per block, the path's mean component per token over the **root** mean square token norm at the block's output, then the mean over blocks. The rows above say "over the mean squared token norm", which has units 1 / norm and would not compare the two models; read as RMS (dimensionless) |
| the sum's gate | per token ‖Σ paths − (x_out − x_in)‖ / ‖x_out − x_in‖, max over tokens, blocks, prompts, every model and step ≤ 1e-4 (float32 forward, float64 split) |
| the plant | one vector, norm 0.5 × the median token norm over the middle states (1 … L−1) of the unplanted 9536, pooled over the prompts; direction N(0, I) normalised, seed 20261102; the same vector as the o_proj bias of all 16 blocks |
| "the planted component" | the planted model's Σ write component less the unplanted one's (each on its own u); the bias path's share of it ≥ 0.9. One-sided as frozen: a share above 1 passes |
| the known answer ("all") | ρ_ℓ from the split's own forward (`output_attentions` on) at 32m 1, 9536 and 410m 0, 4000 equal T2's `diag` to 1e-6 |
| reported | per model and step, the six shares and sizes and the write's size; each rise step beside its first-read row (any centred arm with P, fixed-τ 3e-3 / 0.5 plant A, uncentred plant A, ρ_mid, position 0's norm) and the first read's stops. A description: no rule, no label |

**The planted-bias gate retired on the record (2026-11-02, Blocked 131 (a), the user's, as
recommended; fixed before any batch output, code `t5f.py` `gate`).** At `-32m` 9536 the plant sent
ρ to 1.00 at state 1 and the MLPs wrote a new constant that took 0.95 of the planted component, so
the bias path held 0.125 (`status-T.md` "T5 U3's split"). The gate was mis-specified: on a trained
model, the model-level planted component includes the model's response to the plant, and no bar was
derived for that response. What the split claims is **direct attribution**: each path's own write
on u, the paths summing to the block's write. Three checks test that and stay as gates:

| check | bar |
|---|---|
| the parts sum to the block's write, every record | ≤ 1e-4 |
| known answer: ρ_ℓ from the split's forward, 32m 1, 9536 and 410m 0, 4000, against T2's `diag` | ≤ 1e-6 |
| tiny Llama and GPT-NeoX (tests): the bias path is n · b·u per block exactly, the MLP bias path 0 | exact (rel 1e-6) |

The plant is still run at 9536 and reported beside, not gating. **The paths are read as direct
writes**: a path's share says which parameters write the component, not what the component would
be without them (a path's removal can move every other path, as the plant showed). Not taken:
(b) a small plant along the unplanted u (still a test of the model's response, so it may fail the
same way); (c) reading past a failed gate; (d) the bias's own component as the denominator (1.0 by
construction); (e) closing U3 at its first read. Parked: the model's response to a constant
(`status-T.md` "T5 U3's split, the batch").

**The batch, fixed before output:** all 17 records (`-32m` 9 steps, 410m 8) and the plant re-run
under one code hash into a fresh directory, `data/trackT/t5u3s/2026-11-02-batch/`; 9536 re-run there
must equal the first output's record (`c8c966ac571c`) in every share, size and the write to 1e-12, a
reproduction check beside the gates (the first output is kept as it was).

**Token constancy, beside (post hoc, after `/challenge-pr` on #305, finding 1; fixed before its
re-run).** On summed components every path counts only through its average over tokens, so
"mlp_const" against "mlp_rest" is the MLP write's pooled mean against each prompt's deviation from
it; it does not say the MLP writes one vector to every token. Beside, per block and module (the
bias-free MLP write, the attention write), the fraction of its squared norm in its pooled mean,
N ‖m‖² / Σ_i ‖w_i‖² (1 iff every token gets the same vector), mean over blocks. The batch is re-run
with `--force` into the same directory (the paths' numbers must not move: the 9536 reproduction
check again); no rule, no label.

**U4, the causal arm.** Its design is written after U2 and the `gpt-2-small` read, as a section
of this file and its own Blocked item, before any training (D3). Cap: 3 GPU-days (raised to 4 at the freeze, Blocked 132 D5 (b)). **Aimed at the
rise** (Blocked 130 (a)): its arms are read on whether they re-concentrate after their trough, not
at init. The cap's reach in Dodecahedron steps is the ROADMAP row's compute line (`docs/ROADMAP.md`
§7b T5). It depends on the batch, which T1 Q3 grades only "leaning" 4.19M tokens a step: at that
batch an arm stops before the rise (1253 → 9536), and at the card's 524k it reaches into the rise.
Whether the rise is fixed in steps, in tokens or by the LR schedule is not known, so a run with
another batch or schedule may not rise where Dodecahedron did (`/challenge-pr` on #301, finding 3).
U4's design states which of the three it matches, with U3's crossings in hand.

### U4, the causal arm, designed (proposed 2026-11-02; frozen 2026-11-03 on Blocked 132, the user's: D4 (a), D5 (b) with the cap raised to 4 GPU-days, D6 (b), D7 (a); nothing trained)

Written with U3's crossings and split in hand (`status-T.md` "T5 U3, first read", "T5 U3's
split, the batch"). Arithmetic: `tools/math_checks/t5_u4_budget.py` (B1–B10; B9, B10 added at the freeze). **The question:**
in Dodecahedron's own recipe, do biases (H2), or Pythia's init (H1 through training), remove the
cloud's late rise? The rise is read where every count has already stopped.

| from U3 | so U4 |
|---|---|
| every count stops by 4369 (uncentred 1713, fixed τ 2002, T2's B-matched arm 4369; ρ_mid 0.32–0.81), none before ρ rises | reads its late level at 3738 and 4369 and trains no further |
| the shared component is written by attention's ordinary keys and the MLP write's pooled mean, about equally; the sink grows with position 0's norm; Dodecahedron has no bias path | H2 asks whether a model given biases rises less; the split on each arm at 4369, beside, shows whether its biases take the write |
| 410m's biases write 0.12 of the component at 4000, 0.31 at 143000 | a model that has biases uses them, late |
| what raises ρ is not measured (the write's size 0.093 → 0.149 while ρ_mid 0.11 → 0.90) | the labels read ρ itself, not a path |

**The schedule it matches: Dodecahedron's steps and LR schedule, at the card's 524,288 tokens a
step.** Matching tokens is out of reach: at T1 Q3's leaning 4.19M a step, a single arm gets to
step 3707 in the whole cap at the ROADMAP's guessed 60k tokens/s (B2), short of 4369, so no
comparison of arms is possible. At 524k each arm reaches 4369 in 2.29B tokens (B1; 10.6 h at
60k). The cost: if the batch was 4.19M, the arms see an eighth of Dodecahedron's tokens at every
step. A0 then tests whether the rise appears at Dodecahedron's steps and LR with an eighth of the
tokens. Either outcome is a reading (below).

| item | fixed as |
|---|---|
| architecture | `dodecahedron-32m`'s `config.json`: Llama, 16 × 384, 6 heads / 2 KV of 64, SwiGLU 1024, RMSNorm eps 1e-5, rotary θ 1e4, tied, vocab 16,384; 31,469,952 parameters (B5); built by `t5.build` (U1's code, which its init gate checked) |
| tokenizer | Dodecahedron's (`step9536`), as U1 |
| data | FineWeb `sample-10BT` (`HuggingFaceFW/fineweb`), tokenized with Dodecahedron's tokenizer and packed into 2048-token sequences, its files read in order until 4369 × 256 sequences (2.29B tokens); shuffled once (`default_rng(20261102)`), one stream in the same order for every arm; uint16, outside git |
| batch | 2048 tokens × 256 sequences = 524,288 a step (the card's 64 × 4 × 2048); micro-batch and accumulation set by memory in U4a |
| optimizer | AdamW β 0.9 / 0.95, weight decay 0.01 on matrices (not on norm gains or biases), gradient clip 1.0 (card) |
| LR | linear from 0 to 3.5e-3 over 1000 steps, then cosine toward 9536 with floor 0 (the card gives none: 0.662 of the peak at 4369, 0.696 with floor 0.1, B3); stopped at 4369 |
| precision | bf16 autocast, float32 master weights |
| checkpoints | step 0 and every `t2.GRID` step ≤ 4369 (48), float32 (~126 MB each, ~6.2 GB an arm), the optimizer state at 4369 (so an arm can be continued); `$METS_DATA/trackT/t5u4/<arm>/`, on the HDD |
| read | U2's statistics per checkpoint through `t5c`'s per-state code (the 7 v1 prompts, float32, eager), so U2's known answer covers the reader |

Not matched, and so checked together by A0: Dodecahedron's FineWeb subset and order, its
packing and BOS convention, its LR floor, its weight-decay exclusions, eight GPUs' numerics.
U4a fixes the packing and BOS details before any arm runs, as U3's split did.

**The arms** (each read against A0; the data stream is the same for all):

| arm | change | init | reads |
|---|---|---|---|
| A0 | none: Dodecahedron's recipe | F (0.02 everywhere), seed 0 | positive control: the rise reproduced |
| A0′ | none | F, seed 1 | seed noise τ |
| A0″ | none; trained to 5970 with its continuation (added 2026-11-03, Blocked 133 (i) (d), the user's, before any arm output) | F, seed 2 | seed noise τ (with A0, A0′); beside, a seed pair at 5970 (A0, A0″) that enters no label |
| A1 | biases on every linear (q, k, v, o, gate, up, down), zero at init | A0's weights plus zero biases, so A1 is A0 at step 0 exactly | **H2** |
| A2 | Pythia's init P (U1's: small_init on the embedding and input maps, wang_init(384, 16) on `o_proj`, `down_proj`) | P, seed 0 | **H1 through training** |
| A3 | biases on every linear, as A1 (D5 (b), the user's, with the cap raised so it does not take the delay check's budget) | A2's weights plus zero biases, so A3 is A2 at step 0 exactly | biases and P together; the interaction, beside H1 and H2 |

**The statistic and the rule (fixed before any arm runs).** For arm k, from its ρ_mid series on the grid:

| symbol | definition | Dodecahedron `-32m` (B4) |
|---|---|---|
| L_k | the late level: mean ρ_mid at 3738 and 4369 (two steps against step-to-step jitter) | 0.788 |
| T_k | the trough: min ρ_mid over grid steps 100 … 4369, with its step | 0.111 (917) |
| R_k | the re-concentration, L_k − T_k | 0.677 |
| ΔL_k | L_A0 − L_k: how much of A0's late level the change removes | |
| τ, bar | τ = max(0.02, the largest of \|L_A0 − L_A0′\|, \|L_A0 − L_A0″\|, \|L_A0′ − L_A0″\|); **bar = max(0.2, 2τ)** (Blocked 133 (i) (d), the user's, 2026-11-03, before any arm output: the floor is 410m's largest seed-pair difference at 143000, 0.1996, so it sits outside the whole measured spread; first frozen τ from A0, A0′ alone and bar = max(0.1, 2τ), U1b's form; B11) | |

- At 4369 the change **lowers the late level** if ΔL_k ≥ R_A0 / 2; **partly** if bar ≤ ΔL_k <
  R_A0 / 2; **does not** if |ΔL_k| < bar; **raises it** if ΔL_k ≤ −bar. (First written "removes
  the rise"; renamed after `/challenge-pr` on #306, finding 1: a read at one step cannot tell a
  removed rise from a later one.)
- **The delay check** (finding 1; fixed before output). A0 and every arm that lowers the late
  level, fully or partly, continue from their saved 4369 state (weights and optimizer) to **5970**
  on the same schedule and stream (Dodecahedron's ρ_mid 0.872 there; 0.84B tokens an arm, B7).
  L′_k = mean ρ_mid at 5108 and 5970; ΔL′_k = L′_A0 − L′_k. The label **holds** if ΔL′_k ≥ bar
  and ΔL′_k ≥ ΔL_k / 2 (the gap has not closed by more than half); otherwise the change
  **delays** the rise.
- **H2** from A1, **H1 through training** from A2: lowers and holds → **supported**; partly and
  holds → **partly**; delays → **not told apart** (named "delays the rise"); does not, or raises →
  **not supported**. An arm whose continuation does not fit the cap reads "lowers the late level
  at 4369; removes or delays not told apart", never "supported".
- **A3, together** (D5 (b)): read under the same rule and the same delay check, labelled "biases
  and Pythia's init together: lowers / partly / delays / does not / raises". It does not enter H1's
  or H2's label, which come from A1 and A2 alone. **The interaction** I = ΔL_A3 − ΔL_A1 − ΔL_A2
  (= L_A1 + L_A2 − L_A0 − L_A3), at 4369, with **bar_I = √2 · bar = max(0.2√2, 2√2 τ)** (floor 0.1√2 until Blocked 133 (i) (d) raised bar's floor to 0.2; B10: I's noise is √2
  times a pair's, so the whole bar scales, floor included; first frozen as max(0.1, 2√2 τ), which
  left the floor unscaled and read a no-interaction null as "interaction" 6–16 % of the time at
  σ 0.03–0.05 against the main effect's 1–8 %; corrected 2026-11-03 before any code or output,
  `/challenge-pr` on #307, finding 3; the threshold is *placed*, not calibrated). I ≥ bar_I: **together lower it more than the sum**; I ≤ −bar_I: **less than the
  sum** (one change takes what the other would); |I| < bar_I: **no interaction seen at this noise**
  (not "additive": three seeds' largest pair sets the bar). I is read at 4369 only; at 5970 it is beside,
  and only if all three of A1, A2, A3 continued. If neither A1 nor A2 lowers L and A3 lowers and
  holds, the reading says so in those words: "neither alone; together".

Beside, with no label: each arm's full series (ρ_mid, ρ_mid⁻⁰, ρ_1:5, m, n0), T_k and its step,
R_k (also against A0's: a change that leaves L but raises T reads here, not in the label); the 7-prompt bootstrap of each ΔL_k (U2's: one resample shared across arms and steps, 1,000,
`default_rng(20261030)`); the training loss and the v1 prompts' loss at each checkpoint
(Dodecahedron's: 4.16 at 917, 3.77 at 4369). At 4369 also U3's split (`t5f`: shares and sizes,
and on A1 the bias paths), plus the two counts that need no derivation: (ii) the fixed-τ arm
at 3e-3 / dose 0.5 and (iii) the uncentred count at 1e-3 (plant A share), provided T2's preflight
accepts a local checkpoint. No T2 arm is derived, so the τ search bound (U3 first read, Parked 3)
is not hit; it is carried to T5's close-out.

**Gate (before the reading; any failure goes back to the user).** A0 runs first and alone; the
other arms start only after A0 passes, so a recipe that does not rise costs one arm.

| check | when | pass |
|---|---|---|
| known answer: the reader on released `-32m` steps 1 and 9536 against U2's record | before any arm | max \|Δρ_ℓ\| ≤ 1e-6 |
| the loop: on a tiny Llama, 2 steps on fixed tokens equal a hand-written AdamW with the schedule, **in float64** (Blocked 133 (ii), the user's, 2026-11-03: first frozen as float32 at rel 1e-5, which sits below float32's own rounding here, details (t10)); float32 beside as a sanity bound | before any arm (a test) | rel 1e-10 |
| the init: per-role SDs at step 0 against F (A0, A0′, A0″, A1) and P (A2, A3) | each arm's step 0 | within 2 % |
| step-0 ρ_mid inside U1's 20-seed range of its cell (without biases) ± 0.03: A0, A0′, A0″ in Llama-32m F's [0.817, 0.851], so **[0.787, 0.881]**; A2 in Llama-32m P's [0.445, 0.514], so **[0.415, 0.544]** (finding 5) | each arm's step 0 | inside |
| A1 is A0 at step 0, A3 is A2 at step 0: ρ_ℓ equal | step 0 | ≤ 1e-6 |
| **the noise floor, from outside** (finding 4): ρ_mid's SD over `pythia-410m` seeds 0–9 at steps 4000 and 143000 (on disk; U5's input, minutes) | U4a, before any arm | 2 × SD and the largest pair difference ≤ bar's floor **0.2** at both steps, else back to the user before the arms: the bar's floor would sit inside a known seed spread (first frozen 2 × SD ≤ 0.1; it refused at 143000, 2 × SD 0.153, largest pair 0.1996, and Blocked 133 (i) (d) raised the floor to 0.2) |
| first output populated: A0's step-1 and step-10 checkpoints load and give finite ρ_ℓ; the training loss falls over steps 1–100 | A0, before it continues | yes |
| **positive control: A0 reproduces the rise** | A0 at 1253, then at 4369 | T_A0 ≤ 0.25 by 1253 (every trough read is 0.10–0.21), else A0 stops there; L_A0 ≥ 0.60 (every Pythia end level read is 0.37–0.44; Dodecahedron 0.79) |
| the seed noise leaves the rule readable | A0′, A0″ at 4369 | bar < R_A0 / 2 (on Dodecahedron's R, τ < 0.169; B11); else "removes" and "partly" merge, and U4 reads only removes against does not, saying so |

There is no trained positive control at the rule's size: no recipe is known to remove the rise.
**The interaction's planted null** (finding 3 on #307; a unit test before U4b): six arms' L drawn
i.i.d. at σ 0.03, 0.05 (1,000 draws each, 1e5 since (t11); τ from the largest of the A0, A0′, A0″ pairs since Blocked 133) read "interaction" at most as
often as one main-effect pair reads "partly" or more, plus 0.02.
The rule's arithmetic is tested on planted series instead (A0's series with L lowered by R/2 reads
"removes"; A0′'s reads "does not"; as a unit test).

**Throughput, and the arms that fit (fixed now).** U4a measures θ on A0's configuration over 50
timed steps after 10 warm-up steps (discarded):

| θ (tokens/s) | arms, in a 4 GPU-day cap (B11, with A0″ (Blocked 133); B9's five-arm rows were 42.9k and 33.1k; the 3-day rows 45.1k and 35.4k, B2, B7) |
|---|---|
| ≥ 51.9k | the six arms and every continuation (A0, A0″, A1, A2, A3) |
| 39.8k – 51.9k | the six arms; continuations as the cap allows, A0's first, then A1's, A2's, A3's, A0″'s last (it enters no label); an arm without one reads as the rule says (never "supported") |
| < 39.8k | A0 alone runs (any version of U4 needs its control); the rest goes back to the user with θ and A0's result. Revised after finding 3: the first version dropped A0′ below its row and read with unmeasured noise, which degrades instead of refusing. A3 is not dropped to fit either: you added it with the budget |

60k tokens/s (the ROADMAP's guess) is about 34 % of the RTX 3080's dense bf16 rate at 340 MFLOP a
token (B8, a ceiling argument), and d 384 makes the matmuls small, so the middle and last rows
are live.

**Outcomes, stated before output.**

| A0 | then |
|---|---|
| no trough ≤ 0.25 by 1253 | refused: the recipe at 524k a step does not reproduce the fall; nothing read; A0's series reported |
| spreads, L_A0 < 0.60 | refused: the rise is not reproduced at 524k by 4369 (it may need the tokens, if the batch was 4.19M, or something else in the recipe). H1 and H2 "not told apart". Continuing A0 to 9536 or a larger batch would be a new Blocked item |
| passes | the arms are read under the rule, with the delay check. The rise appears at Dodecahedron's steps and LR schedule at 524k a step, so it is not set by tokens alone if the batch was 4.19M |

**What it cannot show.** Each arm is Dodecahedron's recipe with one change. So a label says what
biases or Pythia's init do to a Llama trained on FineWeb at LR 3.5e-3, not why Pythia's cloud stays
lower: the data, tokenizer, LR, norm and MLP differ there too (ROADMAP row, "Not told apart").
The delay check reaches 5970 only, so a rise delayed beyond it reads "holds". A2's smaller output
init may move the trough's timing (T_k beside). There is one run per arm and two replicate seeds
for the noise (A0′, A0″; checked from outside on 410m's seeds, where 2 of 10 seeds end in a lower group: if Dodecahedron's recipe splits as often, three seeds all land in one group 52 % of the time, B11). **The records do not separate biases from
LR:** every run read that ends low (Pythia's three, `gpt-2-small`) has both biases and a lower peak
LR (6e-4 or less against 3.5e-3) (finding 2). If A1 reads "not supported", LR is the next
candidate. It needs its own design: within this schedule an LR arm never reaches A0's cumulative
LR at the read (B6), so it needs a constant LR or a longer schedule.
**D3's branch, stated** (finding 2): `gpt-2-small` spread but started at 0.335, so D1's question
was not answered, and U2 put it on U4's flat-init arms ("if it spreads"). Here A0 and A1 are
flat-init, and A2 is the contrast.

**Budget (frozen 2026-11-03).** Three units: U4s (the screen at init, below; D6 (b)), U4a (the
loop, the data shard, the tests, θ, the 410m noise floor, and A0 through its positive control) and
U4b (the other arms, the delay check and the reading). That is two units beyond T5's budget, on the
record (D6 (b), D7 (a)). **The cap is 4 GPU-days** (D5 (b), the user's: A3 and its continuation
added to the budget rather than taken from the delay check; 3 GPU-days before).

**U4s, the screen at init (D6 (b), the user's: Blocked 125 (a)'s first step, run anyway; rule
fixed here before any output).** Context at step 0: U4's arms are fixed above and the screen does
not change them (U2 found the init's concentration gone by 671–917). It finishes what U1b left
(`status-T.md` "T5 U1b", Parked). Code: `t5b.py` extended, U1b's base, swaps, seeds (20), inits
(P, F), statistic (ρ_mid, per-cell median), E0, τ and bar = max(0.1, 2τ).

| arm | cells | label, per cell (U1b's `label_of`) |
|---|---|---|
| full screen | every subset S of U1b's six swaps (`ln`, `par`, `gelu`, `rot`, `mha`, `tok`): 64, with ∅ the base and the six-set `all` | ΔE_S = E0 − E_S: **carries** ≥ E0 / 2; **partly** ≥ bar; **does not** \|ΔE_S\| < bar; **amplifies** ≤ −bar |
| `gelu` split | its three changes, alone and in pairs: `gate` (the gate dropped, SiLU 1024 ungated), `act` (GeLU for SiLU, gated, 1024), `width` (1536, gated SiLU); the three together are `gelu` | as above, against E0 |
| reverse ablation | GPT-NeoX-32m without biases (U1b's `all`) with one Llama part put back: RMSNorm, sequential residual, SwiGLU 1024, full rotary, 2 KV heads, Dodecahedron's tokenizer; and all six (the base) | gain G_k = E_k − E_all: **suffices** ≥ E0 / 2; **partly** ≥ bar; **does not** \|G_k\| < bar; **opposes** ≤ −bar |

The reading names the **minimal carrying sets** (subsets that carry with no carrying proper
subset); if only `all` carries, "only all six together". Beside, no label: each pair's
ΔE_jk − ΔE_j − ΔE_k, the mean ΔE by subset size, and the same on logit and atanh ρ (U1b's
scale caveat). Gate before the reading: ∅, the six singles, `untied` and `all` reproduce U1b's
records (same seeds; ≤ 1e-6); `gate` + `act` + `width` built together equals `gelu`'s cell, and
the reverse ablation's all-six equals the base, each within 0.05 of the median at both inits (U1b's
bar for one model built another way, since the seeds' draws need not line up); the reverse ∅ is
`all`'s cell itself; the init SDs within 2 %. 77 cells × 2 inits × 20 seeds = 3,080 models, about 10
minutes on the 3080 at U1b's rate (360 in 69 s). It cannot say what training does with a part
(U4's arms do that for biases and the init only).

*Implementation, fixed before any output (2026-11-03, U4s's code `t5g.py`, which imports `t5b`
and leaves its code hash alone; none of these moves the rule):* (i) **79 cells, not 77**: the 64
subsets, `untied` (the gate's and τ's), the split's six plus the three built together, the reverse's
six plus its all-six; 77 left out two of these. 3,160 models. (ii) **The reverse cells are U1b's
`all` with the part's swap removed, under their own key (`+rev-<part>`, `+rev-all6`)**, so their
seeds are independent of the screen's five-part subsets, which are the same construction: a second
draw of those six cells, compared beside. Building them from `GPTNeoXForCausalLM` was not chosen:
NeoX has no grouped KV heads, so `mha` could not be put back. (iii) **The six-set is U1b's `all`
record for record**, `untied` included, as in U1b (drawn last, it never reaches a state); the other
subsets are tied. (iv) **The split's MLP** is one module: `gate_proj` (when gated), `up_proj`,
`down_proj`, no biases, P's small_init in and wang_init out; gated: down(act(gate x) ⊙ up x),
ungated: down(act(up x)); `act` is exact GeLU as NeoX's, else SiLU; `width` 1536, else 1024. With
gated SiLU at 1024 it equals Llama's MLP on the same weights (a test). (v) **The scales beside:**
per cell the median over seeds of logit ρ_mid and of atanh ρ_mid, then the same differences.

**U4a's details, fixed before any arm output (2026-11-03; code `trackT_timaeus/t5h.py`; none of
these moves the rule).**

| item | fixed as |
|---|---|
| (d1) documents | FineWeb `sample-10BT` at revision `9bb295dd`, its parquet files in name order, rows in order; each `text` encoded by Dodecahedron's tokenizer with no special tokens (it adds none; the reader adds none either), then `<|endoftext|>` (id 0) |
| (d2) packing | the documents concatenated, cut into consecutive 2048-token sequences; no mask across documents; no BOS (the tokenizer has none) |
| (d3) the stream | the first 4369 × 256 sequences permuted by `default_rng(20261102)` (the design's), stored in that order; update t reads rows 256(t − 1) … 256t − 1 |
| (d4) the delay check's tail | the next 1601 × 256 sequences of the same packing, permuted by `default_rng(20261103)`, read by updates 4370 … 5970; built now so every arm's continuation sees one stream |
| (d5) storage | uint16 on the HDD (`<HDD_1TB>/mets_data/trackT/t5u4/data/`, linked from `$METS_DATA/trackT/t5u4`), with each part's sha256 in `meta.json` |
| (t1) the update | update t (1-indexed) uses B3's LR, peak · t / 1000 to 1000, then cosine, and makes checkpoint t; step 0 is the init |
| (t2) the loss | mean next-token cross-entropy over each sequence's 2047 predictions, in float32 from bf16 logits; the 256 sequences in equal micro-batches, summed gradients ÷ their number |
| (t3) the optimizer | `torch.optim.AdamW` (foreach), ε 1e-8 (the card gives none; PyTorch's default), weight decay on every 2-D weight (the tied embedding included), none on norm gains or biases; clip by the global norm |
| (t4) attention | SDPA in training (bf16 autocast); every checkpoint is read eager in float32 (U2's reader), so the training kernel does not enter the read |
| (t5) checkpoints | float32 safetensors without the tied `lm_head.weight`; the reader rebuilds `t5.config_of`'s eager model, loads them and checks the tie. A resume state (weights, optimizer, step) every 200 updates and at each grid step, overwritten; the 4369 state kept |
| (t6) A0's first output | the step-1 and step-10 checkpoints read with finite ρ_ℓ; the training loss falls: mean over updates 91–100 below the mean over 1–10 |
| (t7) the trough stage | T over the grid steps 103 … 1253 (100 is not on the grid), read once A0 reaches 1253; A0 stops there if T > 0.25 |
| (t8) the noise floor | U2's reader on `pythia-410m` (seed 0) and `-seed1` … `-seed9` at 4000 and 143000; SD with ddof 1, per step; passes if 2 · SD ≤ 0.1 at both; seed 0 against U2's record beside. **Refused** (2 · SD 0.153 at 143000); Blocked 133 (i) (d), the user's: floor 0.2, 2 · SD and the largest pair difference both ≤ it (passes: 0.153, 0.1996), and A0″ |
| (t9) the known answer | released `-32m` steps 1 and 9536 saved and reloaded through (t5)'s format, then read: the gate covers the save, the load and the reader (917 and 4369 beside) |
| (t10) the loop's test | **deviates from the gate row as written**: in float32 the loop and the hand-written AdamW differ by 1.2–2.5e-5 relative (bar 1e-5), while each is 2.4e-4 from the float64 result, so the bar is below float32's own rounding; in float64 they agree to 1e-14. The gate is the float64 test at 1e-10 (the loop's two micro-batches against the reference's full batch, clip off and binding); float32 is a sanity bound beside. Put to the user before A0: **confirmed, Blocked 133 (ii), 2026-11-03** |
| (t12) the configuration | `torch.compile`, micro-batch 4 (64 accumulations); chosen by a probe on random tokens (5 updates each: 69k tokens/s eager at 4, 106k compiled; 8 and 16 run out of memory); fixed in `t5h` (`N_MICRO`, `COMPILE`), refused if an arm's segment differs from its `meta.json` (after `/challenge-pr` on #309, finding 3) |
| (t11) the interaction's planted null | "reads 'partly' or more" read two-sided, \|ΔL\| ≥ bar ('raises' is the other side), since I's label is two-sided; seed `default_rng(20261103 + 1000σ)`, fixed before the test first ran; **1e5 draws, not the design's 1,000** (`/challenge-pr` on #309, finding 2: at 1,000 the outcome is about a coin flip on the seed, 76 % pass), so the test reads the rates |

**U4's beside stage at 4369 (Blocked 134 (a), the user's, 2026-11-06; fixed before any output; code
`trackT_timaeus/t5j.py`).** Description, no label: U4's labels are final. The continuations do not
run (134 (a)); LR goes to ROADMAP §7b, not opened.

| item | fixed |
|---|---|
| models | the six arms' `ckpt/step4369.safetensors` (A0, A0′, A0″, A1, A2, A3), loaded by `t5h.load_weights`; for the onsets each arm's 49 checkpoints 0 … 4369. Reference row: released `-32m` at 4369 from U3's records |
| counts (i)–(iii) | `t2.py preflight` unchanged at trained step 4369, the trained population loaded from the arm's checkpoint (its loader swapped in-process, so `t2.py`'s hash and `t5h`'s stay); **T2's re-inits, plants, `u` seed and doses, as U3**. (i) T2's centred arms as derived: P and the full gate per arm, and the wide B-matched search beside (U3's post hoc `WIDE_HI`); (ii) the fixed-τ grid (`t5e.TAUS` × `DOSES`), read at 3e-3 / 0.5: plant A ≤ 0.10; (iii) the uncentred plant-A share at 1e-3, binned as U2b. P, (ii)'s plant A and (iii) do not read the re-inits; the full gate and the null-matched arm do, against Llama re-inits that are A0's init in distribution and not A1–A3's (stated beside, not corrected) |
| the split | `t5f.read_model` on each arm at 4369: the six shares and sizes, the write's size, token constancy; on A1 and A3 the bias paths (attn_bias, mlp_bias). Gate: each record's `err_sum` ≤ `t5f.SUM_TOL` (1e-4) |
| sink metric S(s) | Gu et al. 2410.10781's Sink^ε to the first token, ε = 0.3: the share of (block, head) whose attention to position 0, averaged over every query position, exceeds ε; mean over the 7 prompts |
| massive activation M(s) | per prompt and block output ℓ = 1 … L − 1 (the last is seen only after the final norm): the largest \|coordinate\| over tokens and dimensions, and its ratio to the median \|coordinate\|; per ℓ the median over prompts; M(s) the largest ratio over ℓ, with that ℓ's magnitude. **Sun et al. 2402.17762's criterion**: some ℓ with magnitude ≥ 100 and ratio ≥ 1000 |
| position 0's norm n0(s) | the read records' n0 (position 0's norm over the median token norm), the median over the middle states, as U2's `n0_mid`; no new forward |
| onset (each of S, M, n0) | the **half-rise step**: the first grid step s from which x ≥ x(0) + ½ (x(4369) − x(0)) at s and every later step; "no rise" if x(4369) ≤ x(0). Sun et al.'s criterion: the first step from which it holds at every later step, else "not met by 4369". Beside: ρ_mid's half-rise from its trough (x(T_k) in place of x(0), steps after T_k) |
| beside the onsets | the share of (prompt, ℓ) whose largest coordinate sits at position 0, and the dimension index most often holding it |

**Gates (before the reading; any failure goes back to the user).**

| check | pass |
|---|---|
| counts, the code path: released `-32m` 4369 through `t5j`'s harness with `t2.load` itself, against U3's `32m-step4369/preflight_plants.json` | `t5d.known_answer` at 1e-12 |
| counts, the load path: `t5u4/known/step4369.safetensors` (released 4369 saved by (t9)) through the swapped loader, against the same record | `t5d.known_answer` at 1e-6 (the reader's known-answer bar, design "Gate") |
| the split's load path: the same file through `t5f.read_model`, against U3's batch record `t5u3s/2026-11-02-batch/32m/step4369.json` | shares, sizes, `rho` within 1e-6 |
| the onsets' forward: n0 per state from `t5j`'s forward on each arm's checkpoints equals the read records' n0 | 1e-6 at every step |
| tests: S on a hand-built attention tensor, M and the criterion on planted arrays, the half-rise rule on planted series (with "no rise" and a series that crosses back) | pass |

**What the comparisons can say, stated before output** (one run per arm; the replicates' spread of
each reading at 4369 is the only noise on them). Dodecahedron's records: (i) has P at 3738 (ρ_mid
0.76, wide search) and none at 4369 (0.81). At 4369 A1 sits at A0's step and v1 loss with ρ_mid 0.66
against 0.80, and the replicates at 0.75–0.80.

| (i) at 4369 | reads (description) |
|---|---|
| A1 has P; A0, A0′, A0″ none | the count's stop goes with ρ (or with what biases change), not with step or loss, in this recipe |
| A1 none; A0 none | not ρ alone at 0.66: step, loss or something the arms share |
| A0 (or a replicate) has P | the count passes at ρ ≈ 0.75–0.80 in this recipe, unlike Dodecahedron at 0.81; ρ alone is not its stop |

The same table is read for (ii) and (iii) beside. **Not shown by it:** one run per arm; A1 differs from
A0 in biases as well as ρ, so "goes with ρ" cannot be told from "goes with biases".

**U5, the controls.** U2's statistics on Mistral GPT-2 small from checkpoint 0 (on disk), on
410m's seeds 1–9 at the 18 steps (on disk), and on `gpt-2-small` if D1 (b). They describe, with no rule.

### U5's details (fixed 2026-11-06 before any U5 output; Blocked 135 (a), the user's)

**Inputs.** `EleutherAI/pythia-410m-seed1` … `-seed9` at C2's 18 steps; `pythia-410m` (seed 0) re-read
at the 18; `stanford-crfm/alias-gpt2-small-x21` at its 36 checkpoint branches on disk (0 … 20000) and
`battlestar-gpt2-small-x49` at its 35 (0 … 16000; the same recipe, another seed). Every checkpoint read as
U2 reads one (`t5c.read_model`), plus the mean next-token loss on the 7 prompts. Tokenizers: Pythia's;
for both Mistral runs the GPT-2 BPE from alias's checkpoint 0 (battlestar's branches hold none). Code
`t5c.py` `u5`, `scripts/t5u5.sh`; records `data/trackT/t5u5/2026-11-06/` (main tree).

**Gate** (any failure back to the user before the description):

| check | pass |
|---|---|
| seed 0 re-read against U2's records, all 18 steps (run first; the batch stops on a refusal) | ρ_ℓ, ρ_ℓ⁻⁰, m, m⁻⁰, n0 at every state within 1e-6 (absolute below 1, relative above) |
| seeds 1–9 at 4000 and 143000 against U4a's floor records (`HDD_1TB/…/t5u4/floor`) | the same, 1e-6 |
| Mistral load: both runs | checkpoint 0's loss within 1 nat of ln 50257 (10.82), the last checkpoint's ≥ 3 nats below it |

**Described** (no rule, no label; `u5.json`): per trajectory the series ρ_mid, ρ_mid⁻⁰, ρ_1:5, m and n0
at the middle states (median), loss; U2's shape on ρ_mid, ρ_mid⁻⁰ and ρ_1:5, fixed: **early peak** the
maximum over steps in (0, 1000], **trough** the minimum at or after it, **end** the last step, fall = peak −
trough, late rise = end − trough. Over the ten 410m seeds, per step: mean, SD, min, max, which seeds
sit at the ends, seed 0's rank; and each shape feature's range over the seeds. Mistral: the two runs'
|Δρ_mid| at their 35 common checkpoints. The 410m late level is quoted as the seeds' range at 143000.

**Not shown by it.** Mistral's checkpoints stop at 20000 of 400000 steps, so its "end" is not a trained
level; its recipe (data, LR, batch) differs from every other run here, so it is one more architecture under
a flat init with depth-scaled outputs, not a control for any one factor. Ten seeds of one Pythia size
bound that size's seed spread, not the spread across sizes or recipes.

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
| U3 | ~~**planted bias:** a vector at 0.5 × the median token norm added as an o_proj bias in every block of `-32m` at 9536~~ **retired on the record 2026-11-02 (Blocked 131 (a))**: it refused at 0.125 by testing the model's response, not the split; reported beside. The split's gates are the sum, the known answer and the tiny-model test ("The planted-bias gate retired on the record") | ~~the bias path's share of the planted component ≥ 0.9~~ |
| U4s | the screen's gate in "U4s, the screen at init" (frozen 2026-11-03): U1b's records reproduced, the `gelu` split and the reverse all-six against their cells, the init | as there |
| U4 | the gate in "U4, the causal arm, designed" (frozen 2026-11-03, Blocked 132): the reader's known answer, the loop's test, the init, A1 = A0 and A3 = A2 at step 0, the 410m noise floor, A0's first output, **A0 reproduces the rise**, the seed noise | as there |

Any failure goes back to you before that unit's reading.

## Decisions (each yours; recommendation first)

| # | choice | options | recommended, and why |
|---|---|---|---|
| D1 | where `timaeus/gpt-2-small`'s trajectory goes | (a) in U2: its init and about 20 log-spaced checkpoints of its 650 (100 … 65,000; about 6.5 GB at 325 MB each), read under U2's "spreads" rule, and its result sizes U4; (b) in U5 as the ROADMAP row has it, described with no rule | **(a).** It is the only released run with a passing model's architecture (Pythia-160m) under the flat init (its data and LR differ, unknown), so it answers "does a flat init keep the cloud concentrated through training" before any GPU-day is spent. It cannot separate H2, since it has biases. (b) keeps the plan as proposed but reads the strongest control without a rule |
| D2 | U1's bands | (a) as above: ρ_1:5 ≥ 0.60 / ≤ 0.30 for R1a, R1b; E_init ≥ max(0.1, 2τ) where the dose is ≥ 3, 70m's E_init the smallest; Llama-32m under P at least halfway to NeoX-P; (b) bands from the 20-seed spread alone (E_init beyond the 95 % seed interval), no fixed size | **(a).** A fixed size says the init matters by an amount that bears on the counts. (b) would call a 0.03 shift "supported" with 20 seeds. The R1 bands come from T2's records (Dodecahedron's four init values less their spread; 410m's plus 0.05); 0.1 is a judgement. Revised after `/challenge-pr` on #290 (findings 1, 2): the first proposal's 0.75 on ρ_mid sat where a Dodecahedron-like 70m would land, and its reason (the 32m–67m spread) was wrong (0.016, not 0.09) |
| D3 | U4 | (a) designed after U2 and D1's read, own Blocked item, 3 GPU-day cap; (b) committed now to the ROADMAP's 4 arms ({biases on, off} × {0.02, depth-scaled}) at 32m's shape | **(a).** If `gpt-2-small` does not spread, the flat-init arm is half answered and U4 can drop to the biases contrast; if it spreads, U4's flat arms carry the question. (b) spends up to 3 GPU-days before that is known |
| D4 | U4's schedule (Blocked 132; **decided (a), 2026-11-03, the user's**) | (a) Dodecahedron's steps and LR schedule at the card's 524,288 tokens a step, read to 4369 (2.29B tokens an arm); (b) tokens matched at T1's leaning 4.19M a step; (c) a compressed schedule (warmup and cosine shortened so 4.19M-token steps fit the cap) | **(a).** Every count stops by 4369, and (a) reaches it in every arm within the cap if θ ≥ 26.5k (B2). (b) gets one arm to step 3707 in the whole cap at the guessed 60k, so no arm is compared. (c) matches neither steps, tokens nor the schedule, so a rise there or its absence says nothing about Dodecahedron's. (a)'s cost: an eighth of the tokens if the batch was 4.19M, which A0's positive control reads |
| D5 | U4's fifth arm (Blocked 132; revised after `/challenge-pr` on #306, finding 2; **decided (b), 2026-11-03, the user's, over the recommended (a), "but add to the budget": A3 runs and the cap goes to 4 GPU-days, so the delay check keeps its budget**) | (a) none: its budget goes to the delay check (A0 and the arms that lower L continued to 5970); (b) biases and Pythia's init together (the ROADMAP's 2 × 2), beside, for the interaction; (c) peak LR lowered in Dodecahedron's schedule | **(a).** The delay check is what lets an arm read "supported" at all (finding 1), and a fifth arm outside the exit competes with it for the cap (B7). (b) reads an interaction that matters only if a main effect shows. (c) is the strongest candidate the records leave (every run that ends low also has a lower LR), but within this schedule an LR arm never reaches A0's cumulative LR at the read (B6: half the peak gives 2384 peak-steps over the whole run, against A0's 3470 by 4369), so it cannot be read at a fixed step or at a matched dose. The first version rejected (c) because a fixed step cannot tell removes from delays; that is true of A1 and A2 too, so it was not a reason. LR goes to its own design (constant LR or a longer schedule) if A1 reads "not supported" |
| D6 | the pair screen and the reverse ablation (Blocked 125 (a) made them U4's first step; **decided (b), 2026-11-03, the user's, over the recommended (a)**: U4s, a full screen) | (a) dropped, recorded in T5's close-out with why; (b) run first anyway (about 10 GPU-minutes and a unit of writing), as context at init | **(a).** They rank the Llama's parts at init, and U2 found the init's concentration gone by 671–917. U4 is aimed at the rise (Blocked 130 (a)), so a ranking at init cannot choose what it trains |
| D7 | U4's budget (**decided (a), 2026-11-03, the user's**; with D6 (b), U4s makes it two units beyond) | (a) two units, U4a (loop, data, tests, θ, the 410m noise floor, A0 through its positive control) and U4b (the other arms, the delay check and the reading), one beyond T5's budget, on the record; the 3 GPU-day cap unchanged; (b) one unit, A0 and the arms back to back | **(a).** It puts a stop after A0, so a recipe that does not rise costs one arm (about 11 h) and not four. (b) keeps the budget but launches four arms before the positive control is known |

## Exit and budget

**Exit** (ROADMAP): H1 and H2 each read supported, not supported or not told apart (H2 only from
U4; "not told apart" if U4 does not run) under the rules above, and "does Dodecahedron's cloud spread at any
checkpoint" answered on the 53. **Budget: U0 (this design and scan) + U1–U5 + close-out**, U4 ≤ 3
GPU-days, **plus U1b** (extended on the record, Blocked 124 (a), 2026-10-30), **plus U2b** (Blocked 126 (a),
2026-11-01, on the user's delegation), **plus U3's revision** (Blocked 130 (a), 2026-11-01, the
user's: one docs unit; U3 then runs in budget, its first read minutes of GPU). **plus U4s and U4's second unit** (Blocked 132, 2026-11-03, the user's: D6 (b), D7 (a); U4s, U4a, U4b, two beyond), **U4 ≤ 4 GPU-days** (D5 (b): A3 added with its budget). **No lean:** no step-0 cloud of 70m, 160m or `gpt-2-small` has been computed; the only
clouds read are T2's (410m 0, 4000; Dodecahedron 1, 9536 and the re-inits).

**Closed 2026-11-07 at the exit** (`status-T.md` "T5 on one page (closed 2026-11-07)"): the cloud
spreads on both models; H1 at init partly; H1 through training and H2 not supported at bar 0.2;
the bar's floor rests on a PolyPythias outlier pair, Blocked 136 (decided (a), 2026-11-07: labels
kept, nothing carried).

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
- U4's late level L is read at a fixed step (mean of 3738, 4369), with a delay check to 5970
  only (added after `/challenge-pr` on #306, finding 1); a rise delayed past 5970 still reads
  "holds". Rejected: reading each arm to its own plateau (an open horizon the cap cannot pay
  for), and reading at matched cumulative LR (the same as matched steps for every arm here).
- U4's bands are judgement: the positive control's T ≤ 0.25 and L ≥ 0.60 sit between the
  troughs and end levels read so far (0.10–0.21; Pythia 0.37–0.44, Dodecahedron 0.79), and
  "lowers" at half of A0's rise copies U1b's "carries". The noise comes from two replicate seeds
  (A0″ added by Blocked 133), so τ could still be underestimated; the floor, 0.2 since Blocked 133
  (0.1 first), is 410m's largest seed-pair difference (another model and recipe).
- A3 (D5 (b)) gets a label of its own and an interaction bar, √2 × the main bar (floor included),
  scaled from the replicates' largest pair by an independence assumption (B10; the floor first left unscaled,
  finding 3 on #307). Rejected: I beside with no label (the user
  added the arm to read the 2 × 2). LR, the candidate the records leave beside biases, is still not
  tested in U4.
- U4s (D6 (b)) labels 77 cells with U1b's bar and no correction for the number of cells; its
  minimal carrying sets are a description at init, and it cannot change U4's arms.
- U4 matches steps at 524k (D4) because tokens cannot be matched within the cap. If T1 Q3's 4.19M
  is right, every arm sees an eighth of Dodecahedron's tokens, and A0's control may refuse for
  that reason alone; that refusal is itself the reading on steps against tokens.
- A1's biases are zero at init and trained with no weight decay. A reviewer may prefer them decayed
  like the matrices (Pythia's own setting is not read here).
