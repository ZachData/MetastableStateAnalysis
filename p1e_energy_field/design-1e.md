<!-- p1e_energy_field/design-1e.md -->
# Phase 1e — the energy landscape: wells, crests and the force (design, FROZEN 2026-10-06)

**Status:** frozen 2026-10-06 (user, Blocked 28, "follow your recommendations"): (a) as proposed;
(b) U2's block arm first, then Phase 10's Blocked 27 (c) before U1; (c) U4 stays fenced until P-S1
is scored; (d) U5 stays parked. Opening scan:
`lit-1e.md`. Tier 1: exploratory, unregistered. Opened on the user's direction (2026-10-06): read the
residual stream as a field, not as clusters of particles.

## Why a new phase

1d asked "what is a cluster" with level-set groups, and every check since (R7, R8) lands on a
placed bar: `min_cluster_size`, the floor's 10th percentile, 0.5, the bulk share. The theory has a
field with one parameter the model supplies: `φ_β(x) = Σ_j exp(β⟨x, x_j⟩)` on the sphere, with
β = 3.5 [1.6, 5.6] measured (`STATE.md` Blocked 9). Its wells are clusters, its crests and saddles
are the gaps between them, and its gradient is the force the theory says each token feels. 1e
reads those three things directly, and reads the model's actual update against the force. It does
not need a cluster definition; where it produces one (a well), that definition has one measured
knob.

## The math (each line checked in `tools/math_checks/energy_field_1e.py`; what it does not prove is said there)

| | claim | used for |
|---|---|---|
| M1 | `E_β = (1/2β) Σ_i φ_β(x_i)`: the global energy is the sum of per-token local energies, so the deviations `φ_β(x_i) − mean` sum to 0 | "deviation from the global energy" is a per-token quantity, `e_i = log φ_β(x_i) − mean_k log φ_β(x_k)` |
| M2 | on the unit sphere, `∇ log φ_β(x) = β P_x^⊥ m(x)`, `m(x) = Σ_j softmax_j(β⟨x, x_j⟩) x_j` | the force: mean shift is ascent of the field; one attention step with `Q = K = V = I` is that step |
| M3 | for n ≤ d + 1, `Σ_{i≠j} exp(β⟨x_i, x_j⟩) ≥ n(n − 1) e^{−β/(n−1)}`, equality at the regular simplex | the repulsive (`V = −I`) end: at Pythia's n < d, "packing" is pairwise near-orthogonality (⟨·,·⟩ → −1/(n − 1)), not a lattice |
| M4 | as β → 0, `m(x)` → the plain mean, correction `(β/n) Σ_j (⟨x, x_j⟩ − mean)x_j` | at small β the force says only "towards the centroid"; its **local part** is `m_β − m_0` |
| M5 | `exp(β⟨x, y⟩) = e^β exp(−(β/2)‖x − y‖²)` on the sphere | the global energy is Wang & Isola's uniformity at `t = β/2` (`lit-1e.md` §3) |
| M6 | a critical point of `φ_β` lies in `span(x_j)` **or has `m(x) = 0`**. *Corrected after `/challenge-pr` on #156, finding 4:* for x ⊥ span every weight is equal, so `m(x)` is the plain mean of the `x_j`; in the **cloud-centred** frame that mean is 0 before the rows are renormalised (and small after), so the whole subsphere ⊥ span is critical, or nearly flat (the field's floor, φ = n). On unit LN1 rows (primary) the rows are generically independent at n ≈ 200–500 < d, their mean ≠ 0, and every critical point is in the span | wells, crests and saddles worth reading live in the ≤ n-dimensional subsphere of the tokens; the interesting voids are the low points *between* tokens, inside the span. The centred arm has a degenerate floor off the span, reported as such |

**On the user's framing (2026-10-06).** The energy `E_β` is always positive (a sum of exponentials).
What the theory ties to clustering is the **direction** the dynamics moves it: with `V = I` the
flow ascends `E_β` and tokens cluster; with `V = −I` it descends and tokens spread to M3's simplex
(2312.10794 §9.1's sharp configurations when the count allows). Trained weights are neither, `Q^⊤K`
is not symmetric, so there is no single energy the flow must follow, and the project measured it
not following one: monotone at steps 8–64, broken at 128–512 (`p2_eigenspectra/status-2.md`
"Headline result"), and an OV spectrum that turns repulsive at 1000–2000 (`PROJECT.md` §3.12 A).
So the user's "negative energy / repulsive particles / sphere packing" reads here as: **where, per
token and layer, does the update descend the field instead of ascending it, and do those tokens
move towards M3's near-orthogonal configuration.** That is U2 and U4.

## Inputs, fixed here

| choice | value | why, and what was rejected |
|---|---|---|
| model, checkpoints | `pythia-410m`, the 18 distinct Stage 0 steps | the activations already on disk (`p10_cluster_function/status-10.md` §1.14's index); the 12 v2 prompts stay held out |
| passages | *Changed 2026-10-06 (user, after the freeze, before any output): "move towards only using the larger prompts, that's just more information".* **Primary: 8 long passages**, every one at all 18 steps: 1d's 4 (`p1d_cluster_ensemble/long_prompts.py`: `wiki_paragraph`, `sullivan_ballou`, `hdbscan_code`, `latex_monograph`, 1,032–2,036 tokens) and 4 new (`p1e_energy_field/long_prompts_1e.py`: Butler's *Odyssey* VII, *Le Horla*, *Origin* ch. IV, *Hamlet* I.i; the user chose these four). **Beside: the 7 v1 passages**, kept offsets as R0 (T1–T3: position 0, massive tokens, repeats out): a **continuity check with the record, not a replication** (*after `/challenge-pr` on #157, finding 4*): they are stored CPU runs, and 4 of the 7 are the opening tokens of 4 long passages. Each unit's targets on a long passage are fixed in its own rule, before its output (U2's below) | more tokens per passage, and 8 passages, not 4: a sign test over 4 cannot pass below p = 1/16. Rejected: the 4 existing long passages alone (that floor); v1 as primary (512 positions, the user's direction); the 12 v2 prompts (held out on 410m, and ≤ 512 tokens) |
| **frame** | **unit LN1 rows** of each layer (LN1 recomputed from the stored residual and the checkpoint's LN1 weights), the frame β was measured in (`p1d_cluster_ensemble/status-1d.md` "β refit"); raw and cloud-centred unit rows beside | a field with the measured β is only defined in β's frame. *The 2026-10-06 probe used raw and cloud-centred rows, so its numbers are not on this frame* |
| β | 3.5 primary, 1.6 and 5.6 (the interval's ends) beside; a sweep 0.5–100 for the landscape only | measured; the sweep shows where wells shatter |
| sum | *Changed after `/challenge-pr` on #156, finding 2:* **causal** (j ≤ i, the field token i can feel; `math-10.md` §7.5's sequential flow) **primary for U2 and U4**, which compare against the model's move; **full** (every kept token: the cloud's landscape) **primary for U1 and U3**, which map the landscape. Each reported beside the other. The full sum adds later tokens no attention update sees, a gap that grows with position | F12 read the causal sum and found position, hence the position control below |
| the sink (position 0) | in U2's causal field it is a **source** (every token attends to it), never a target (T1 removes it from the targets); U2 reports the field with and without it as a source (*finding 5*) | the model's update includes its attention to the sink; leaving it out would compare against a field the model does not feel |
| the move in β's frame | *After finding (frame), adopted from the review:* `u_i = unit(LN1_ℓ(x_i^ℓ))` and `u'_i = unit(LN1_ℓ(x_i^{ℓ+1}))` (layer ℓ's own LN1 applied to the next residual); the move is `P^⊥_{u_i}(u'_i − u_i)`. Stays in the frame β was measured in, with no Jacobian | rejected: carrying Δx through LN1's Jacobian (first order only); reading the field on residual rows (β not measured there) |
| position control | `e_i` regressed on log offset per (step, passage, layer); residual reported beside the raw value | `math-10.md` §2 (99.5 % position on raw rows, causal) |

## Units (each its own PR; each opens with its first output checked as populated; order U2's block arm → U1 → U3, U4 fenced)

| unit | question | reads | cost |
|---|---|---|---|
| **U1, the field at the tokens** | where is density and where is void: `e_i` per token (deviation from the global energy, M1), wells by mean shift at β (each token's basin), and **what the 2–4 wells of the probe are** (position, the opening, content: class share, R0's c3 overlap) | stored activations, LN1 weights | free, minutes per step |
| **U2, the force against the move** | does each token's update go where the field says: `a_i = cos(P^⊥Δx_i, g_i)` with `g_i` the field's tangential gradient (M2) and its local part (M4), per token, layer, step. `a_i > 0` ascends (attractive), `< 0` descends (repulsive). Two arms: the whole block's update (stored residuals, free) and attention's output alone (forward hooks, GPU), and an arm with **each head's own kernel** (its real `softmax(QK^⊤)` rows) in place of `φ_β`, since the idealised head is a poor model of the real one (`lit-1e.md` §2) | stored activations; hooks for the attention arm | free; the attention arms ~1 forward pass per (step, passage), GPU |
| **U3, crests and saddles** | between each pair of U1's wells, inside the span (M6): the minimum-energy path and its saddle height (nudged elastic band); persistence of each well; the low points of `φ_β` along it are the "interesting voids" | U1's wells | free |
| **U4, the repulsive regions** | where U2 reads repulsive, do those tokens move towards near-orthogonality (pairwise ⟨·,·⟩ against −1/(n − 1), M3) over layers and steps; set beside the OV repulsive phase (1000–2000) and the energy break (128–512). **Fenced until P-S1 is scored or the user waives** (below: its statistic is P-S1's degree-1 moment) | U2 | free |
| **U5, correlated against anti-correlated (Parked)** | the user's superposition question at the token level: substitutes (paradigmatic, anti-correlated in co-occurrence) against co-occurring pairs (syntagmatic) — which are closer in the field, per layer and step (`lit-1e.md` §4) | needs co-occurrence counts from a corpus; none is local | a corpus download and a count; then free |

**Nulls and controls, every unit.** Step 0 (init) on the same tokens; Pythia-architecture random
inits (unit 2's re-init recipe; activations are not stored, so a forward pass per init, GPU);
position (above); the β interval's ends. **U2's null** (*changed after finding 3*): a uniformly
random tangent direction (`cos ~ N(0, 1/D)`, spread ≈ 0.03 at this D) cannot separate "follows the
field" from any update component common to all tokens (shared anisotropy), so the primary null is
a **within-passage token shuffle** (token i's field against token k's move, same layer and step),
and the cosine is also reported with the **mean move across tokens subtracted**. The random
direction stays beside as the floor.

**The 1e page** (*proposed 2026-10-07; the user asked where it belongs; theirs to move*): a
findings artifact at **U1's close**, with U2's settled reading (Blocked 29 decided) as its first
section and U1's per-token energy and wells as its centre, since U1 is the first output that is a
picture rather than a table of signs; updated in place at U3's close (saddles between the same
wells). Not at U2's close: U2 alone is sign tables, and its headline turns on Blocked 29.
Rejected: one page per unit (1d had two pages that went stale); waiting for the phase's end (1d
paused without one).

## U2's block arm: the rule (fixed 2026-10-06, before any output)

| | |
|---|---|
| cells | *Changed 2026-10-06 with the passages (above): this unit runs after the long passages are extracted; the counts below for 8 passages are the label row's, the v1 reading beside keeps 7 of 7 / 6 of 7.* 18 steps × 8 long passages (v1's 7 beside) × blocks ℓ = 0–22. **Block 23 refused**: the stored last hidden state is after `final_layer_norm` (HF GPT-NeoX; its norms ≈ 75–100 against ≈ 60–430 at L23), so its "move" is not the block's. Bands: L0 (beside; β there ≈ 0, `status-1d.md` "β refit"), L1–8, L9–16, L17–22 |
| frame, move | as "Inputs": `u_i = unit(LN1_ℓ(x_i^ℓ))`, `u'_i = unit(LN1_ℓ(x_i^{ℓ+1}))`, stored rows × stored norms; `d_i = P^⊥_{u_i}(u'_i − u_i)` |
| targets | *Fixed after `/challenge-pr` on #157, finding 1, before any field was read.* **Long, primary: T1 + T2**, every position but 0 and the massive tokens (`move_text.massive_positions` at any of the 18 steps, the union, so one set across steps): 1,031–2,039 per passage. Repeats stay: they were dropped for clustering (identical strings co-locate at L0), but U2 scores each token's own move against its own force, and a repeat has its own context. **Beside: T1–T3** (R0's rule, `move_text.kept_offsets`): 17–40 % of each passage, front-loaded (`hdbscan_code_long` keeps 6–11 % of its last three quarters), so position and word rarity are confounded there; the label is reported on both, and where they differ T1 + T2 is the label and the difference is said. Counts: `status-1e.md` "Targets on the long passages". v1 (beside): R0's kept offsets (`data/p10/reread_r0_2026-10-05/labels`) |
| sources | **causal, every stored position j ≤ i** (primary: the sink, massive tokens and repeats included, since attention reads them); beside: the sink out; the **full** sum over every stored position; the **local** part (causal, `m_β − m_0`, M4) |
| force | `g_i = P^⊥_{u_i} m_β(u_i)`, `m_β` the softmax-weighted mean of the sources (M2). The self term changes only the length of `g_i`, not its direction |
| per token | `a_i = cos(d_i, g_i)`; tokens with `‖d_i‖` or `‖g_i‖` < 1e-9 left out and counted |
| per cell | `A` = mean `a_i` over targets. **Null: 1,000 within-passage permutations** of the moves among the targets (token i's force against token k's move projected onto i's tangent space: `d_k·g_i / (‖P^⊥_{u_i} d_k‖ ‖g_i‖)`, since `g_i ⊥ u_i`); excess `X = A − mean(A_null)` and two one-sided ranks. Beside: `Ã`, the cosine with the mean move across targets subtracted first (`P^⊥_{u_i}(d_i − d̄)`); the random-direction floor 1/√(D−1) ≈ 0.031; Spearman of `a_i` with log offset (position) |
| label per (step, band) | each passage's `X` averaged over the band's blocks, 8 values. **ascends** if all 8 > 0 (sign test p = 1/256), **leans ascends** 7 of 8 (p = 0.035); **descends** / **leans descends** the same below 0; else **mixed**. Beside: cells with rank ≤ 0.05 each way. *After `/challenge-pr` on #157, finding 3:* both cut-offs are **placed**, not calibrated. **Expected by chance** (no effect anywhere, passages independent), per β over the 54 primary cells (18 steps × L1–8, L9–16, L17–22): **0.42 ascends / descends, 3.4 leans** (2 · 8/256 each cell); printed beside the label table, and no "leans" is reported without that count. A "leans" not shared by an adjacent step is marked **isolated** (adjacent steps share weights, so chance labels can run together into what looks like a window) |
| memorization (beside) | *After `/challenge-pr` on #157, finding 5:* each passage's mean per-token NLL at each step, from the stored last hidden state (after `final_layer_norm`) × its norms through the checkpoint's `embed_out`, no forward pass; the labels recomputed without the lowest-NLL passage at 143000, beside |
| β | 3.5 primary; a label is **β-robust** if 1.6 and 5.6 give the same, otherwise β-dependent |
| step 0 | read by the same rule beside every trained label; a trained label that step 0 also carries is not called learned |
| first checks (refuse) | stored rows unit to 1e-4; each exported block's revision is the run manifest's; every long run's manifest carries `LONG8_HASH`; R0's kept offsets index the stored positions; the first cell (step143000, `wiki_paragraph_long`) opened and checked populated (finite `a_i` for ≥ 90 % of targets at every block) before the rest; a synthetic cloud moved one exact mean-shift step reads `A` ≈ +1 and the reversed step ≈ −1 (test) |
| shared update (beside) | *Added 2026-10-07 after the output (`/challenge-pr` on #158, finding 1); not part of the frozen label.* The shuffle null and `Ã` cannot see an update added to every token, so beside every U2 arm: `r1out` (each token's residual update less its component along the shared direction, through LN1; added after #159's review), `residout` (less the mean update), `resid` (the mean alone), and the β = 0 field. Which one is the headline is `STATE.md` Blocked 29 (`status-1e.md` "The shared update") |
| not computed | step lengths, or any sum of them over layers (P-γ fence); any pairwise ⟨·,·⟩ statistic of the cloud (P-S1 fence); per-head kernels (the per-head arm, its own PR) |

## U2's attention arm: the rule (fixed 2026-10-07, before any output)

*Blocked 29 decided (c) (user, 2026-10-07): run this arm, carry `r1out` and `resid` beside the
frozen label, and decide 29 on its output.* Everything not named here is the block arm's rule.

| | |
|---|---|
| pass | one forward pass per (step, passage) on the GPU, the long batch's settings (float32, eager, TF32 off), with hooks: each layer's attention output, MLP output, attention weights to key 0 and the value at position 0. Nothing but the records is stored |
| components | Pythia's residual is parallel, so block ℓ's update is exactly **attention + MLP**. Split further: **`sink`** = attention to key 0, `Σ_h A_h,i0 W_O^h (v_h0 − b_V^h)`; **`keys`** = attention to keys 1…i, the rest of attention less its constants (the mean pull); **`mlpx`** = the MLP less its output bias; **`bias`** = `b_O + W_O b_V + b_mlp`, one vector added to every token (`Σ_j A_ij = 1` makes `b_V`'s part constant). `attn` (sink + keys + its constants) and `mlp` whole are read too; `block` = attn + mlp is the check row. `sink + keys + mlpx + bias = block` exactly |
| move per component | `c_i` added alone, in the block arm's frame: `u'_i = unit(LN1_ℓ(x_i + c_i))`, `d_i = P⊥_{u_i}(u'_i − u_i)`. Not additive across components (LN1 and the sphere); `block` is |
| readings, per component | **frozen** (causal at β 1.6, 3.5, 5.6; the β = 0 field `mean0`), **`r1out`** (`c_i` less its component along its own shared direction `ĉ = unit(mean_t c)`; causal 3.5 and `mean0`), **`resid`** (`mean_t c` alone; causal 3.5), `residout` (`c_i − mean_t c`; causal 3.5). `bias` reads frozen only (causal 3.5, `mean0`: it is its own `resid`); `block` frozen causal 3.5 only |
| shared shares (beside) | per (step, passage, block): the block's shared update `c̄ = mean_t(attn + mlp)` split by component, `share_k = (mean_t c_k)·ĉ / |c̄|` for sink, keys, mlpx, bias (sum to 1); each component's **sharedness** `|mean_t c_k| / mean_t |c_k|`; the mean attention to key 0 over targets and heads. Reported per band as the median over passages |
| targets, cells, label, nulls, β, step 0 | the block arm's: T1 + T2 on the 8 long passages (v1's R0 kept offsets beside), blocks 0–22, bands, 1,000 within-passage permutations, the sign rule over passages with its chance counts and isolated leans, β-robust across 1.6 / 3.5 / 5.6 for the frozen reading |
| **reading for Blocked 29** (fixed before output) | in each trained (step, band) where the block's `resid` ascends (`status-1e.md` "The shared update"), the component with the largest share of `c̄` whose own `resid` carries the same direction: **`keys`** → the shared ascent is attention to the other tokens, a mean-field pull through the cloud, so it **counts as following the field** (option (a) for that window); **`sink`, `bias` or `mlpx`** → a vector that does not depend on where the other tokens are, so **not** (option (b), `r1out` the headline there); none or tied (shares within 0.1) → said so, undecided. The verdict is per window, and 29 is the user's on that table |
| first checks (refuse) | the pass's hidden states against the stored run's (unit rows × norms): long ≤ 1e-5 (same device and settings), v1 ≤ 1e-4 (stored on the CPU; the 512-token GPU gap was 7.1e-6); `x^ℓ + attn + mlp = x^{ℓ+1}` and `sink + keys + mlpx + bias = attn + mlp` to 1e-4 relative to the update's norm; the `block` row's `X` equals the block arm's stored frozen record (causal 3.5, `t12`) to 1e-6 on the first cell; the first cell (step143000, `wiki_paragraph_long`) populated (finite `a_i` for ≥ 90 % of targets, every block and component); a tiny random GPT-NeoX checks the hooked split against the explicit sum over keys (test) |
| not computed | per-head kernels (the per-head arm); step lengths; pairwise ⟨·,·⟩ of the cloud (fences unchanged) |

## Fences (registered predictions and the registry)

| registered | what 1e will not compute before it is scored, or the user decides |
|---|---|
| **P-S1** (1c-F: trained cluster centroids closer to a spherical t-design) | no Gegenbauer moment or design statistic on any centroid, well centre or token cloud. *Corrected after `/challenge-pr` on #156, finding 1:* **U4 as proposed breaches this.** Its mean pairwise inner product is an exact affine function of the degree-1 Gegenbauer moment, and the pairwise distribution is the input to `p1c_frames/design_test.py`'s `gegenbauer_moments`. So U4 is fenced until P-S1 is scored, unless the user waives this clause knowingly (Blocked 28 (c)); U2's attract / repel sign does not touch it |
| **P-γ1 / P-γ2** (1c: `ip_mean` against γ_β(T_eff); T_eff ≪ 4.2) | U2 needs each tangential step `P^⊥Δx_i`, P-γ2's ingredient. 1e reports directions (cosines) only, never the step sizes summed over layers |
| **P-M1** (2d: energy-monotonicity violations concentrate in heads far from QK symmetric) | U2's per-head arm reads each head's own kernel; it does not tabulate per-head energy violations against QK symmetry |

`claims/registry.json` is untouched. Nothing in 1e is registered.

## Relation to the other threads

- **Phase 10 (Blocked 27)** waits: if the user takes 1e, the c3 / c3c choice can stand until U1
  says whether a well is the better unit; R0–R8 stay as they are.
- **1d** stays the record of the level-set route; 1e's U1 reads R0's c3 groups beside its wells,
  so the two definitions are compared on the same tokens.
- The 2026-10-06 probe (`tools/run/p10_phi_wells_probe.py`; `p10_cluster_function/handoff-10.md`
  Parked) is the reason for U1's order; its numbers are on the wrong frame and are not used.

## Worth challenging

- **The passages** (changed after the freeze, before any output; user's direction and sources).
  Long passages give each passage more tokens, but the passage stays the unit a label counts, so
  8 is still few; and 8 is a mixed set: 1d's 4 continue v1 texts (so their first 242–482 tokens
  are v1's), the new 4 do not. `sullivan_ballou_long` is half length. Two are verse / dialogue or
  French, and v1's French passage was a different author. Rejected: more passages now (12, the
  user chose 8); v1 primary; v2 (held out). The Gutenberg three are
  probably in Pythia's training data (the Pile includes Project Gutenberg), as several v1 texts
  are; not checked.
- **The frame.** Settled after the review: layer ℓ's LN1 applied to both residuals ("Inputs").
  What it does not settle: LN1's gain and bias are learned per layer, so the frame itself moves
  over training; a step-to-step change in `a_i` can be the frame.
- **Order** (*finding 6, a defensible difference*): the review argues 27 (c) (~1 h) before U1,
  since U1 compares wells against c3, and U2's free block arm before U1. Taken: U2's block arm
  needs no clusters and goes first; U1's c3 comparison reads whichever set 27 settles.
- **The idealised field.** `φ_β` is a single head with `Q = K = V = I`; Pythia has 16 heads with
  learned `Q, K, V` and an MLP. U2's per-head arm is the hedge; if the real kernel and `φ_β`
  disagree, the theory's field is the wrong object and 1e says so.
- **Ghost modes** (`lit-1e.md` §2): mean shift from the tokens can miss wells away from data, and
  the count can rise with β in high d. U1 reports basins of tokens, not a global mode count.
- **U5 parked** for lack of a corpus; the alternative is Pythia's own next-token distribution as
  the co-occurrence proxy (free, but it measures the model, not the data).
