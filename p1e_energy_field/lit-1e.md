<!-- p1e_energy_field/lit-1e.md -->
# Phase 1e — literature scan (opening scan, 2026-10-06)

`CLAUDE.md` trigger 1: the phase opens, before `design-1e.md` freezes. Tags: **[P]** primary text
read (abstract page at least, as here), **[S]** secondary or summary only, **[M]** from memory,
unopened this session. Scholarly PDF hosts are blocked from the session; arXiv abstract pages were
read with WebFetch. Rows the project already read are pointed to, not repeated.

## 1. The field, its wells and its force (the theory)

| work | what it gives 1e | tag |
|---|---|---|
| Geshkovski, Letrouit, Polyanskiy, Rigollet, 2305.05465 / 2312.10794 | `E_β`, the attractive case (`V = I`: ascent of `E_β`, clustering) and the repulsive case (`V = −I`, §3.2 / §9.1: descent, *sharp configurations* and spherical designs). Already read: `p2d_operator_activation/lit-2d.md` row on §3.2/§9.1, `PREDICTIONS.md` P-S1 | [P] (earlier) |
| `p1d_cluster_ensemble/lit-1d.md` §3 | `φ_β` as an unnormalised vMF density, mean shift = one attention step with `Q = K = V = I`, a cluster = a mode's basin, β from the model so no bandwidth is fitted | project |
| Geshkovski, Rigollet, Sun, 2412.09080 | the count of KDE modes at bandwidth `β^{-1/2}` (1D, `Θ(√(β log β))`); the theory's own cluster count. PDF still missing (`STATE.md` Blocked 13) | [S] |
| Gao, Yang, Chen, 2609.28448 (Sept 2026) | `Q = K = I`, `V = −I`, non-causal: for `d = N → ∞` from Gaussian starts, phases include **diffuse simplex-like states**, consensus flips, condensed routing, fragmented cluster flips; condensation at β = O(1) in high d. The repulsive side of the user's question, in the idealised model only | [P] abstract |
| Zimin, Polyanskiy, Rigollet, 2601.23236 (YuriiFormer); Xu et al., 2605.07588 (Causal Energy Minimization) | attention = a gradient step on an interaction energy, MLP = a step on a potential energy (Lie–Trotter split); both **design** architectures from it and train them. Neither measures whether a *pretrained* model's updates follow the field: that measurement (1e's U2) has no precedent found | [P] abstracts |
| Pham et al., 2512.03058 | conditions under which tokens of a pretrained model's continuous-time limit converge or diverge; theory from the weights, not a measured alignment of updates | [P] abstract |
| Karagodin, Ge, Polyanskiy, Rigollet, 2510.22026 | Pre-LN's contraction law (already in `docs/readings/meanfield_reading_list_2026-10-01.md` row 7) | project |

**Not found:** a measurement, in a pretrained LM, of whether each layer's update points up or down
the attention energy's gradient at each token. The closest are energy-*derived* architectures
(trained from scratch) and weight-side conditions. 1e's U2 would be new as stated; the scan is
abstracts only, so "not found" is weak.

## 2. Wells, crests and saddles of a kernel density in high dimension

| work | what it gives 1e | tag |
|---|---|---|
| Edelsbrunner, Fasy, Rote, *Discrete Comput. Geom.* 2013, "Add isotropic Gaussian kernels at own risk" | a sum of isotropic Gaussians can have **more modes than kernels** ("ghost" modes) in d ≥ 2, and the bandwidth range where they survive grows like √d. Mean shift started at the data can miss modes not near data, and mode counts need not fall monotonically with bandwidth in high d. A hazard for counting wells at d = 1024 | [S] |
| Chazal, Guibas, Oudot, Skraba, *J. ACM* 2013 (ToMATo) | modes merged by persistence: the barrier (saddle) height between two wells sets how distinct they are. Already in `lit-1d.md` §1 | [S] |
| Henkelman, Jónsson et al. (nudged elastic band, dimer method) | finding saddles and minimum-energy paths between two minima on a high-dimensional surface: how to find the "crest" between two wells | [S] |
| `math-10.md` §2, `status-10.md` §1.4 | `log Z_{β,i}` (= `log φ_β(x_i)` under the causal sum) is 99.5 % position on raw rows. Any field read at the tokens has to separate position first | project |
| `p1d_cluster_ensemble/status-1d.md` "Attention communities against three nulls", row 6 | the idealised cosine head on the real unit LN1 rows reproduces little of the trained block's community structure (Q 0.156 vs 0.407) and none of its locality. `φ_β` is the theory's field, not necessarily the model's | project |

## 3. Energy, uniformity and packing on the sphere

| work | what it gives 1e | tag |
|---|---|---|
| Wang & Isola, ICML 2020 (2005.10242) | the "uniformity" of features on the hypersphere is `log E exp(−t‖x − y‖²)`, the log of the mean pairwise Gaussian potential: on the unit sphere this is `log mean φ_β − β` at `t = β/2` (`tools/math_checks/energy_field_1e.py` M5). So the global energy is a known representation statistic under another name; 2403.00642 lists its failure modes (sensitive to duplicates, blind to dimensional collapse) | [P] abstract; [S] |
| Cohn & Kumar, *JAMS* 2007 (universal optimality) | which configurations minimise every completely monotone pair potential; for n ≤ d + 1 points the minimiser of `Σ exp(β⟨x_i, x_j⟩)` is the regular simplex (pairwise ⟨·,·⟩ = −1/(n − 1); M3 checks the two-line proof). Pythia's prompts have n ≈ 200–500 < d = 1024: "packing" here means near-orthogonality within the tokens' span, not a lattice | [M] |
| `PREDICTIONS.md` P-S1 (registered) | trained cluster centroids closer to a spherical t-design than step 0's (Gegenbauer moments). **1e computes no design or Gegenbauer statistic until P-S1 is scored or the user decides** (`design-1e.md` "Fences") | project |
| Timkey & van Schijndel, EMNLP 2021 ("rogue dimensions"); Ethayarajh 2019 | a few dimensions dominate cosine in LM representations (anisotropy). The raw-frame probe's single well is likely this | [M] |

## 4. Correlated and anti-correlated features (the user's question)

| work | what it gives 1e | tag |
|---|---|---|
| Elhage et al., *Toy Models of Superposition*, 2209.10652 (2022) | in the toy model, **correlated features are placed orthogonally** (they co-occur, so interference would cost), **anti-correlated features are placed antipodally or share a subspace** (they rarely co-occur, so sharing is cheap). The user recalled this as the monosemanticity paper; it is this one | [P] abstract + [S] |
| Li, Michaud, Baek, Engels, Sun, Tegmark, 2410.19750 | on real SAE dictionaries, **co-occurring features cluster spatially** ("lobes": math and code together) at coarse scale, far more than random geometry gives. At the scale of semantic regions the real geometry goes with co-occurrence, the opposite direction to the toy's local rule | [P] abstract |
| Bal, 2607.12166 | antipodal pairs are present in well-trained SAEs on toy data ("structural inertness") | [P] abstract |
| Sahlgren 2006 (the distributional hypothesis); Mickus, SCiL 2024 ("Language models and the paradigmatic axis") | **paradigmatic** pairs (substitutes, cat / dog) share contexts but rarely co-occur, **syntagmatic** pairs co-occur; embeddings mostly encode paradigmatic similarity. This is the token-level analogue of anti-correlated vs correlated features, and the one 1e could measure | [P] search summary; [S] |

**Reading for 1e.** A particle (a token's state) is not a feature, as the user says. The testable
analogue is at the level of token types: substitutes (anti-correlated in co-occurrence) against
co-occurring pairs (correlated), and whether the residual stream puts the first closer than the
second, layer by layer and over training. The toy rule predicts substitutes close and co-occurring
pairs orthogonal; Li et al. find co-occurring features grouped at coarse scale. Both can hold at
different scales, which is 1e's scale question again. It needs co-occurrence counts from a corpus
(none is local: the Pile is not on the box), so it is parked (`design-1e.md` U5).

## 5. Queue (not read)

- 2412.09080 as primary text (Blocked 13), and whether a d = 1024 version of its count exists.
- 2609.28448 in full: its high-d phase diagram against β = 3.5, and whether "diffuse simplex-like"
  has a statistic 1e can read without touching P-S1's.
- Edelsbrunner–Fasy–Rote as primary text: the √d law's constants at d = 1024 and β ≈ 3.5.
- Any measurement of local density / local intrinsic dimension of LM token states over training.
