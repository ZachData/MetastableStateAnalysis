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

## U2's per-head arm: the rule (fixed 2026-10-07, before any output)

*User, 2026-10-07: "U2's per-head arm (GPU, real softmax(QK^⊤) rows per head), saving each part's
mean update per block" (Blocked 29's two recommendations).* The attention arm read attention's
token-specific move against `φ_β` (one head, `Q = K = V = I`) and found it descends. This arm
asks the same of each head's **own** kernel: if the move still descends, attention pushes tokens
away from what its heads attend to; if it ascends, `φ_β` was the wrong field. Everything not
named here is the attention arm's rule.

| | |
|---|---|
| pass | the attention arm's (one hooked forward pass per (step, passage), GPU float32, eager, TF32 off), plus each layer's full attention weights `A_h` (16 heads, n × n) and values, read inside that layer's hook (CUDA float64) and dropped before the next layer. Only records and the saved means are kept |
| the real kernel's field | head h at token i: `g^h_i = P⊥_{u_i} Σ_{j ≥ 1} A_h,ij u_j`, the head's own causal `softmax(QK^⊤/√d_h)` row (with the model's rotary) in place of `softmax(β⟨u_i, u_j⟩)`, over keys 1…i, values `u_j` (unit LN1 rows: `V = I` in β's frame, the frame the head reads). Whole attention: **`kernns`** = `P⊥ Σ_h Σ_{j ≥ 1} A_h,ij u_j`, what attention to keys 1…i would add with every `V = I` (each head weighted by `1 − A_h,i0` as in the model, no renormalisation); **`kern`** = the same with key 0 in (the sink as a source). No β: the kernel is the model's |
| moves | head h's keys part `c^h_i = W_O^h Σ_{j ≥ 1} A_h,ij (v_hj − b_V^h)` (`Σ_h c^h = keys`), and the whole head `c^h + ` its sink part (`Σ_h` = sink + keys); `keys` and `attn` as in the attention arm. Each added alone in β's frame; readings frozen and `r1out` (less its component along its own `unit(mean_t c)`) |
| cells, per block | **primary**: `kernns:keys:r1out`. Beside: `kernns:keys:frozen`, `kern:attn:r1out`, `kern:attn:frozen`; check row `causal:keys:r1out` (`φ_β`, β 3.5: the attention arm's). Per head: `kernns_h:keys_h:r1out` (head h's keys part against its own field) and `kern_h:head_h:frozen` (whole head, sink in, against its kernel with key 0). The attention arm's cell (1,000 within-passage permutations, `X`, `Xt`, ranks) |
| beside, per (block, head) | alignment of the real field with `φ_β`'s: `mean_t cos(g^h_i, g_i)` (`g` causal, β 3.5), and `kernns`'s; the head's mean attention to key 0; the sharedness `|mean_t c^h| / mean_t |c^h|` |
| saved (Blocked 29; `/challenge-pr` on #160, finding 1) | per (run, block), float64, over the targets: `mean_t c` for sink, keys, mlpx, bias, attn, mlp, block, and each head's keys part and sink part; `means/<kind>/step…_<passage>.npz` beside the records |
| ascent share (CPU, from the saved means and the stored rows) | per (run, block): `F(S) = mean_t ⟨P⊥_{u_i}(unit(LN1_ℓ(x_i + Σ_{k∈S} c̄_k)) − u_i), ĝ_i⟩`, `ĝ` the unit `φ_β` force (causal, β 3.5), `c̄_k` part k's saved mean; each part's **Shapley value** over sink, keys, mlpx, bias (16 subsets, exact; they sum to `F(all)`, the block's `resid` move projected on the force). Per (step, band): each passage's band sum of part k's value over its band sum of `F(all)`, the median over passages. A raw projection with no null: beside the labels, not a label |
| labels | whole-attention rows: the block arm's sign rule per (step, band), with chance counts and isolated leans. **Per head there is no band** (head h in two layers is two unrelated heads): per (step, block, head) the sign of `X` over passages by the same rule; per (step, band) the count of (block, head) units that ascend / descend / lean either way (a unit whose keys cell is short of 90 % of targets in any passage, sink-only, is kept out and counted beside), beside chance (`units · 2/256`, `units · 16/256`; L1–8 has 128 units) |
| **reading** (fixed before output) | in each trained (step, band) where the attention arm's `keys:r1out` against `φ_β` descends or leans descends (long: 30 of 54 cells), `kernns:keys:r1out`: **descends / leans descends** → attention's token-specific move goes **against what its own heads attend to** (repulsion under the real kernel); **ascends / leans ascends** → it follows its own heads' kernel, and the descent was `φ_β` being the wrong field there; **mixed** → the real kernel does not sign it. Per window; the per-head counts say whether many heads carry it or few, `kern:attn:r1out` beside (if it differs, the sink as a source decides it, and that is said) |
| **Blocked 29's ascent reading** (fixed before output) | in each trained window where the block's `resid` ascends: the part with the largest **ascent** share; `keys` → (a) follows the field, sink / bias / mlpx → (b) not interaction, top two within 0.1 → tied. This is finding 1's reading (who supplies the ascent); the attention arm's length share stays beside |
| first checks (refuse) | the attention arm's (the pass against the stored rows, the split adds up); `Σ_h (c^h + sink_h) + b_O + W_O b_V = attn` to 1e-4 relative to the update's norm; every `A_h` row sums to 1 to 1e-5 with no weight above the diagonal; the check row's `X` equals the attention arm's `causal:keys:r1out` record to 1e-6 on the first cell; the first cell (step143000, `wiki_paragraph_long`) populated (finite `a_i` for ≥ 90 % of targets in every cell, every block and head; *changed after that check refused, 2026-10-07, before any reading:* a head's keys cell may fall short only by its **sink-only** targets, where the head's keys part or keys field is below `TINY` (its weight on keys 1…i underflows), and only if the head puts ≥ 0.9999 of its weight on key 0 and the token itself at every such target (attending to yourself adds nothing to a `V = I` field; token 1's one key in 1…i is itself); recorded per head: the count and the largest weight elsewhere. L18 head 3 at 143000 is sink-only at 1,817 of 1,839 targets; any other gap refuses); the saved means' `sink + keys + mlpx + bias = block` to 1e-6 relative; tests: per-head parts and kernel fields against explicit sums on a tiny random GPT-NeoX, Shapley values summing to `F(all)` |
| not computed | P-M1's table (per-head energy violations against `Q^⊤K` symmetry; fence), or any per-head reading set against QK symmetry; step lengths; pairwise ⟨·,·⟩ of the cloud |

## U1: the rule (fixed 2026-10-08, before any output)

*User, 2026-10-08: "Start 1e's U1 (the field at the tokens)", after #163 (Blocked 27 → (a)).* The
field is read where the tokens are, with the states held fixed (one layer's landscape, not the
dynamics). No forward pass. Everything not named here is the block arm's rule.

| | |
|---|---|
| cells | 18 steps × 8 long passages (v1's 7 beside) × layers ℓ = 0–23: hidden state ℓ in layer ℓ's LN1 frame, unit rows (`u_i = unit(LN1_ℓ(x_i^ℓ))`). Hidden state 24 refused (after `final_layer_norm`). Bands L0 (beside), L1–8, L9–16, **L17–23** (U2's L17–22 plus 23: U1 needs no next state) |
| tokens | **T1 + T2** on the long passages, sources and targets alike (primary); T1–T3 beside (repeats out: identical strings co-locate at L0 and are density by construction). v1: R0's kept offsets |
| field | **full**, `φ_β(x) = Σ_{j ∈ T} exp(β⟨x, u_j⟩)` (primary for U1, "Inputs"); β 3.5 primary, 1.6 and 5.6 beside; a **sweep** β ∈ {0.5, 1, 2.5, 10, 20, 50, 100} for the well count only, at layers 4, 12, 20 (the probe's) |
| density | `e_i = log φ_β^{−i}(u_i) − mean_k log φ_β^{−k}(u_k)` over the targets, **self term out** (with it every φ gains `e^β` and an isolated token reads `e^β`, not void); M1's identity holds either way. Per cell: `sd(e)`; Spearman of `e_i` with log offset and the R² of `e_i` on log offset (the position control, "Inputs"), the residual's sd beside; R² of `e_i` on `⟨u_i, ū⟩` (the β → 0 term, M4's "towards the centroid"; `ū` used, not reported). Beside: the **causal** `e_i` as log of the *mean* over sources j < i (every stored position, the sink in), so the source count does not drive it, and its Spearman with log offset. Who is dense / void: the token-class mix (`tools/run/p10_token_composition.token_class`) of the top and bottom deciles of `e_i` against all targets |
| wells | mean shift on `φ_β` from every target (`y ← unit(Σ_j exp(β⟨y, u_j⟩) u_j)`), GPU float32 until each row moves < 1e-6 in cosine distance, rows within 1e-6 merged as they coincide, then CPU float64 on the merged rows until < 1e-10 (cap 5,000 iterations; unconverged rows counted); final modes within **1e-3** are one well (placed, the probe's; the count at 1e-4 and 1e-2 beside). A target's well is its basin. Per cell: wells `k`, wells holding ≥ 2 targets `k2`, `k_eff = exp(H)` of the basin shares, the largest share. These are **basins of tokens, not a global mode count** (ghost modes, "Worth challenging") |
| what the wells are | where `k2 ≥ 2`: **AMI** (adjusted mutual information, 0 at chance) of the well partition against **position** (8 equal-count bins of offset; 8 placed) and against **token class**; on v1 also the **purity** of R0's groups that pass **c3x** (Blocked 27 (a)): the share of each group's members in its modal well, mean over groups, against 200 permutations of the well labels. **The opening:** the share of the first 64 targets that share position 1's well, and that well's share of all targets |
| nulls | **matched Gaussian**: per cell, 4 draws of n rows `ū + G (U − ū) / √(n − 1)` (G standard normal n × n, so the same mean and covariance as the targets' unit rows), each row then put on the sphere; `e` and wells read the same way at β 3.5 (seed `SEED`, crc32 of passage, step, layer, draw). *Amended 2026-10-08 after the first check, before any reading (the rule asked β-robustness of labels (1)–(2) with a null drawn at 3.5 only):* the same draws give `sd(e_G)` at 1.6 and 5.6 too, so (1) is read at all three β; the Gaussian's wells stay at 3.5, so (2)'s β-robustness is **not read** (the real cloud's `k_eff` at 1.6 / 5.6 and the sweep beside). **Step 0** by the same rule beside every trained label. Pythia random inits (the design's "Nulls" paragraph) **not run**, as in U2: step 0 is the init null here |
| labels per (step, band) | each passage's value the mean over the band's layers, 8 values; U2's sign rule (**all 8** / **7 of 8**), the chance counts (0.42 / 3.4 per 54 primary cells) and isolated leans. **(1) lumpy:** `Xe = sd(e) − mean_draws sd(e_G)` > 0 → *lumpier than its Gaussian*, < 0 → *smoother*. **(2) wells:** `Xw = log k_eff − mean_draws log k_eff_G` > 0 → *more wells than its Gaussian*, < 0 → *fewer*. **(3) density and position:** the sign of Spearman(`e_i`, log offset) → *denser later* / *denser early*. **(4) what the wells are**, over passages with `k2 ≥ 2` (fewer than 7 such → *not read*): `AMI_pos − AMI_cls` > 0 → *position*, < 0 → *content*, else *mixed*; the two AMIs' medians beside. β-robust if 1.6 and 5.6 give the same label (1) and (3) (*(2): see the amendment under nulls*); a trained label step 0 also carries is not called learned |
| saved | per cell, float32: `e_i` (β 3.5, full; causal beside) and the well label of every target at β 1.6 / 3.5 / 5.6, so the 1e page can draw them; no well centres |
| first checks (refuse) | U2's (stored rows unit to 1e-4, manifests carry `LONG8_HASH` and the step, targets index the stored positions, R0's kept offsets and R8x's c3x rows index R0's labels); the first cell (step143000, `wiki_paragraph_long`) populated before the rest: `e_i` finite for every target at every layer and β, every target in a well, ≥ 99 % of rows converged; and its GPU wells at β 3.5, layers 4 / 12 / 20, equal a CPU float64 run's (same partition for ≥ 99.9 % of targets). Tests: three planted von Mises–Fisher groups give three wells and their labels; LOO `e_i` against an explicit loop; the Gaussian draw's covariance; AMI 0 under permutation |
| not computed | inner products between well centres or any design / Gegenbauer statistic of centres or cloud, `|ū|` or the mean pairwise inner product (P-S1 fence); step lengths (P-γ); per-head anything |
| **(1′) calibrated lumpiness** (beside; *added 2026-10-08 after the output, `/challenge-pr` on #164, finding 2; fixed before it was computed*) | putting the matched Gaussian on the sphere shrinks its density spread once the covariance is concentrated, so `Xe` > 0 for a cloud with no structure beyond its moments. Per cell and β: 4 **null clouds** `Y_k` (matched Gaussians of the targets, seed `SEED + 1`), each scored as the data against 4 matched Gaussians of its own; `bias = mean_k [sd(e_{Y_k}) − mean_j sd(e_{G_j(Y_k)})]`; **`Xe′ = Xe − bias`**, labelled by the same sign rule (*lumpier / smoother than a structureless cloud*), step 0 beside. GPU float32 Grams; the first cell's `bias` against CPU float64 within 1e-4 (refuse). `Xe` and label (1) stay as recorded. *Finding 4, same commit:* the GPU-against-CPU wells check repeated where there are several wells (`wiki_paragraph_long`, step 128 L4 at β 5.6 and 10, step 143000 L12 at β 10), same ≥ 99.9 % bar |

## U1 beside: β ≈ 10 and the cloud-centred frames, the rule (fixed 2026-10-08, before any output)

*User, 2026-10-08, after #165 merged with Blocked 30 (a) confirmed: "work on all three", i.e. (a)
stands and (b) and (c) are read **beside** it.* Both choose a β or a frame where wells were seen,
the placed-bar problem "Blocked 30 decided" named, so neither replaces U1's reading at the
measured β; each is labelled as chosen. Everything not named here is U1's rule (cells, tokens,
targets, field, density, wells, what the wells are, nulls, labels, first checks), with one choice
changed per run.

| run | the one choice changed | β (primary; beside) | sweep |
|---|---|---|---|
| **(b) β 10** | β: **10**, the sweep's one value with a few wells in many cells (196 of 432 with 2–20); **7 and 14** beside (÷ and × √2, placed: no interval is measured here) | 10; 7, 14 | none (U1's sweep stands) |
| **(c) centred** | frame: **`v_i = unit(u_i − ū)`**, `u_i` U1's unit LN1 rows, `ū` their mean over the target set (each set its own `ū`), applied to every stored position (the causal density's sources) | 3.5; 1.6, 5.6 | U1's, at L4 / 12 / 20 |
| **(c′) raw centred** (beside (c)) | frame: **`unit(x_i − x̄)`** on the stored residual `x` (no LN1), `x̄` over the target set: the 2026-10-06 probe's frame, so its "2–4 wells" are read with U1's rule on every cell | 3.5; 1.6, 5.6 | U1's |

| | |
|---|---|
| nulls | U1's matched Gaussians of the **rows read** (the centred rows in (c), (c′)); their wells at the run's primary β |
| labels | U1's (1)–(4), read at the run's primary β where U1 reads 3.5; (1) and (3) β-robust if both beside β give the same label; the c3x purity on v1 at the primary β |
| caveats, fixed now | (c), (c′): β is not measured in a centred frame, so 3.5 there is the measured value carried over, not measured. M6: in a centred frame the subsphere ⊥ span is a near-flat floor; mean shift from the targets stays in their span and does not see it. (b): `e_i` at β 10 is dominated by each token's nearest neighbours (M5: a Gaussian kernel of width `1/√10` ≈ 0.32 in chord) |
| first checks | U1's, the first cell's GPU wells against CPU float64 in the run's frame at its primary β; records carry `opts` (frame, β set, primary β, sweep), and a resume refuses a record read with other `opts` |
| not computed | (1′)'s calibration (U1's bias is for 3.5 in β's frame; not carried over); U1's fences |

*Amended 2026-10-08, a defect, after (c)'s first check refused and U3 refused on (b), before any
label of (b), (c) or U3 was read.* (c)'s first cell matched CPU float64 on 0.9984 of targets at
L12 (bar 0.999). The cause is U1's float32 phase: it merged rows within `MERGE_RUN` (1e-6) by a
float32 dot, and a CUDA float32 dot of 1024-d unit rows is off by up to ~1e-6. So a row still
moving near a ridge could join a neighbour bound for another well. Without that merge the GPU
equals the reference. (b)'s step 2000 `hdbscan_code_long` L12 had 185 wells against float64's
189 (agreement 0.9886), and 8 targets sat above their own well's mode, which is U3's negative
persistence. The first check passed on (a) and (b) because step 143000 has few wells. **Now:** merges
compare in float64 (`dedup`, `dedup_t`; a test with pairs 1.5e-6 apart). (b), (c), (c′) and U3
run on the fixed producer, and the first build of each is set aside unread. **Added check:** a run is
read only if its audit (`python -m p1e_energy_field.u1_audit`; every record, its
read-at-every-β target set, L4 / 12 / 20, the primary β) has its wells equal to CPU float64's on
≥ 0.999 of targets in every cell; otherwise it is refused. U1 (a), merged in #165 on the old
producer, gets the same audit; what it finds is a correction to (a) (`status-1e.md`).

U3 at β 10, if (b) finds wells to join, gets its own rule before its output.

## U3 at β 10, crests and saddles: the rule (fixed 2026-10-08, before any U3 output and before (b)'s labels were read)

*Blocked 30 (b), the user's. (b)'s first cell (step 143000, `wiki_paragraph_long`) was opened as
its populated check: 1–31 wells at β 10, one holding 86–100 % of the targets.* The field is U1's
(full, `φ_β` over the targets, unit LN1 rows, states fixed), at **β 10 only**; cells as (b): long
passages, T1 + T2, L0–L23, 18 steps; v1 (R0's kept offsets) beside. The saddles live in the
targets' span (M6): every path below starts and stays there. *Amended 2026-10-08 after the first
cell's check, before any label was read (the check refused on NEB convergence; the cell took 648 s
at 24 layers, ~45 h for the batch):* **layers 4, 12, 20 only** (U1's sweep layers, one per trained
band, so a band's value is its one layer), **long passages only** (v1 not read). Every pass height
below, graph or band, converged or not, is the lowest point of a feasible path between the two
wells, so a **lower bound on the pass**, and each persistence an upper bound.

| | |
|---|---|
| wells | (b)'s, recomputed (same mean shift, its modes kept) and **refused unless they equal (b)'s saved labels** (agreement ≥ 0.999 per cell). Every well counts, singletons too (a lone token's bump is a maximum of `φ_β`); `k2` wells beside |
| heights | `h(x) = log φ_β(x)`, the full sum, self term in (a path over a token crosses its bump); a well's peak is `h` at its mode |
| merge tree (primary) | a graph over the targets: each target's **15** nearest neighbours by `⟨u_i, u_j⟩` (placed). *Amended 2026-10-08 after the first cell refused, before any U3 output was written: the rule first doubled k up to 120 until the graph joined every well, and on one matched-Gaussian draw 120 did not. Now:* where the 15-NN graph leaves wells apart, each apart component's targets are also joined to their 15 nearest targets outside it, repeated until every well is joined (rounds counted, `bridges`; 50 rounds, else refused), edge height `min(h(u_i), h(unit(u_i + u_j)), h(u_j))`; edges added from the highest (Kruskal), and when an edge joins two components whose highest wells differ, the lower one **dies** there (elder rule): its **persistence** `p = peak − edge height` (nats), its **saddle** that edge. `k − 1` deaths per cell |
| NEB (beside; the cell's **8 most persistent deaths**, placed for cost: cells at β 10 reach hundreds of wells) | climbing-image nudged elastic band between the dying well's mode and the mode of the well across the edge, 24 images, started on the geodesic path mode → `u_i` → `u_j` → mode, on the sphere (forces projected to each image's tangent; ascent of `h` off the path, a spring along it); float32 for up to 3,000 iterations, the path's heights then in float64. *Converged (amended after the first cell, as above): the climbing image's height moves < 1e-4 nats over 500 iterations;* the rule first asked the largest force on the band < 1e-4·β, which oscillates over token spikes (to 0.5–1.0 at 12,000 iterations) while the pass height is fixed to 1e-4 by iteration 500 (6 bands, step 143000 L4 / L12); the last largest force is reported beside. Reported: its saddle height against the graph's (`Δ_NEB = h_NEB − h_graph`, ≥ 0 when the band finds the higher pass) and unconverged bands. Beside, since a band from a graph start finds *a* pass, not provably the lowest |
| per cell | `P = Σ p` over deaths, `p_max`, deaths with `p` > 1 nat; the token-class mix of the saddle edges' endpoints against all targets (the "interesting voids": who sits on the crests) |
| null | U1's 4 matched Gaussians of the cell (U1's seeds, so the same draws (b) read), their wells and merge tree the same way (graph only, no NEB) |
| label per (step, band) | U1's sign rule over the passages on **(5) `Xp = log(1 + P) − mean_draws log(1 + P_G)`**: > 0 *a deeper landscape than its Gaussian*, < 0 *shallower*; step 0 beside, a trained label step 0 also carries is not called learned. `p_max` and the NEB gap beside, no label |
| first checks (refuse) | the first cell's wells equal (b)'s; its merge tree has `k − 1` deaths and every persistence ≥ 0; two planted vMF groups give one death at the pass the band also finds (test); the first cell's NEB on L4 / L12 / L20 converges for ≥ 90 % of the bands run |
| not computed | inner products between well centres or any statistic of the modes' configuration (P-S1); U1's other fences. The modes are used as path endpoints only, never stored |

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
