<!-- p1e_energy_field/design-1e.md -->
# Phase 1e — the energy landscape: wells, crests and the force (design, PROPOSED 2026-10-06, not frozen)

**Status:** proposed; freezes when the user decides `STATE.md` Blocked 28. Opening scan:
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
| M6 | every critical point of `φ_β` lies in `span(x_j)` | wells, crests and saddles live in the ≤ n-dimensional subsphere of the tokens; the rest of the sphere is featureless "void" by construction. The interesting voids are the low points *between* tokens, inside the span |

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
| passages | the 7 v1 passages, kept offsets as R0 (T1–T3: position 0, massive tokens, repeats out) | one token set across every phase that reads them |
| **frame** | **unit LN1 rows** of each layer (LN1 recomputed from the stored residual and the checkpoint's LN1 weights), the frame β was measured in (`p1d_cluster_ensemble/status-1d.md` "β refit"); raw and cloud-centred unit rows beside | a field with the measured β is only defined in β's frame. *The 2026-10-06 probe used raw and cloud-centred rows, so its numbers are not on this frame* |
| β | 3.5 primary, 1.6 and 5.6 (the interval's ends) beside; a sweep 0.5–100 for the landscape only | measured; the sweep shows where wells shatter |
| sum | over the passage's kept tokens (**full**: the cloud's landscape); the **causal** sum (j ≤ i, the model's own sequential flow, `math-10.md` §7.5) beside | F12 read the causal sum and found position; both are reported |
| position control | `e_i` regressed on log offset per (step, passage, layer); residual reported beside the raw value | `math-10.md` §2 (99.5 % position on raw rows, causal) |

## Units (each its own PR; each opens with its first output checked as populated)

| unit | question | reads | cost |
|---|---|---|---|
| **U1, the field at the tokens** | where is density and where is void: `e_i` per token (deviation from the global energy, M1), wells by mean shift at β (each token's basin), and **what the 2–4 wells of the probe are** (position, the opening, content: class share, R0's c3 overlap) | stored activations, LN1 weights | free, minutes per step |
| **U2, the force against the move** | does each token's update go where the field says: `a_i = cos(P^⊥Δx_i, g_i)` with `g_i` the field's tangential gradient (M2) and its local part (M4), per token, layer, step. `a_i > 0` ascends (attractive), `< 0` descends (repulsive). Two arms: the whole block's update (stored residuals, free) and attention's output alone (forward hooks, GPU), and an arm with **each head's own kernel** (its real `softmax(QK^⊤)` rows) in place of `φ_β`, since the idealised head is a poor model of the real one (`lit-1e.md` §2) | stored activations; hooks for the attention arm | free; the attention arms ~1 forward pass per (step, passage), GPU |
| **U3, crests and saddles** | between each pair of U1's wells, inside the span (M6): the minimum-energy path and its saddle height (nudged elastic band); persistence of each well; the low points of `φ_β` along it are the "interesting voids" | U1's wells | free |
| **U4, the repulsive regions** | where U2 reads repulsive, do those tokens move towards near-orthogonality (pairwise ⟨·,·⟩ against −1/(n − 1), M3) over layers and steps; set beside the OV repulsive phase (1000–2000) and the energy break (128–512). **Fenced** (below): no Gegenbauer or design statistic | U2 | free |
| **U5, correlated against anti-correlated (Parked)** | the user's superposition question at the token level: substitutes (paradigmatic, anti-correlated in co-occurrence) against co-occurring pairs (syntagmatic) — which are closer in the field, per layer and step (`lit-1e.md` §4) | needs co-occurrence counts from a corpus; none is local | a corpus download and a count; then free |

**Nulls and controls, every unit.** Step 0 (init) on the same tokens; Pythia-architecture random
inits (unit 2's re-init recipe; activations are not stored, so a forward pass per init, GPU); a
uniformly random tangent direction for U2 (`cos ~ N(0, 1/D)`, D the tangent dimension); position
(above); the β interval's ends.

## Fences (registered predictions and the registry)

| registered | what 1e will not compute before it is scored, or the user decides |
|---|---|
| **P-S1** (1c-F: trained cluster centroids closer to a spherical t-design) | no Gegenbauer moment or design statistic on any centroid, well centre or token cloud. U4's pairwise inner products against −1/(n − 1) are the M3 statistic, not P-S1's; **whether even that is too close is the user's call** (Blocked 28 (c)) |
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

- **The frame.** Unit LN1 rows are β's frame, but the residual stream is what the stored update
  `Δx` lives in; LN1 is a per-token map, so U2's update has to be carried into LN1's frame (the
  Jacobian of LN1 at `x_i`) or the field carried into the residual's. Rejected: reading the field
  on raw residual rows (β was not measured there).
- **The idealised field.** `φ_β` is a single head with `Q = K = V = I`; Pythia has 16 heads with
  learned `Q, K, V` and an MLP. U2's per-head arm is the hedge; if the real kernel and `φ_β`
  disagree, the theory's field is the wrong object and 1e says so.
- **Ghost modes** (`lit-1e.md` §2): mean shift from the tokens can miss wells away from data, and
  the count can rise with β in high d. U1 reports basins of tokens, not a global mode count.
- **U5 parked** for lack of a corpus; the alternative is Pythia's own next-token distribution as
  the co-occurrence proxy (free, but it measures the model, not the data).
