<!-- docs/readings/meanfield_reading_list_2026-10-01.md -->
# Mean-field reading list (supplied 2026-10-01) — cross-referenced, not read

**Provenance.** The user pasted the list in §6 on 2026-10-01. It was produced outside this
repo; its own "verified" means checked through citing pages, not read. **Nothing here is
`[R]`.** Every scholarly host is blocked from the cloud session (arxiv, export.arxiv,
semanticscholar, openreview, alphaxiv, crossref, papers.nips.cc, proceedings.iclr.cc,
proceedings.mlr.press, mlanthology, pith.science, bytez, researchgate, github.io: all
refused on 2026-10-01). Marks: **[S]**, **[N]**, **[R]** as in `docs/LITERATURE.md` §0 ([S] here: a search
summary was read this session). **[L]** is defined here only: the claim comes from the
supplied list, and no summary of the paper's page was seen (weaker than [S]).

The project's verification queue is `docs/LITERATURE.md` §5; this file's §4 is the
mean-field addition to it, and is not repeated there.

---

## 1. What it changes for this project

Ranked by how directly each bears on the active thread (Phase 1d, `STATE.md` Blocked 11′
and 9). None of it is a result; each row names what reading the paper could decide.

| # | Papers | Project item | What reading could decide | Mark |
|---|---|---|---|---|
| 1 | **2604.01978** *Homogenized Transformers* (Koubbi, Geshkovski, Rigollet): weights resampled independently across layers and heads, "as at initialization"; joint depth/step/heads scaling gives an Itô SDE on the sphere with logistic collapse. **2601.21942** *Clustering in Deep Stochastic Transformers* (Fedorov, Sander, Elie, Marion, Laurière; ICML 2026): random value matrices at init; for two tokens a phase transition in interaction strength and dimension, antipodal configurations attract with positive probability. **2604.26898** (Agazzi et al.): synchronization by common noise | 1d's **step-0 control** (`p1d_cluster_ensemble/status-1d.md` "Position"). Step 0 *is* the random-weight model these papers analyse | What an untrained Pythia should do at d = 1024, n ≈ 467, β ≈ 0, stated before looking. Secondary for Blocked 11′: `status-1d.md` already has the mechanism (near-uniform attention, position 0's share 1/(t+1)), and a cut derived from it, or step 0 at M = 32 on the long prompts, is cheaper than reading (`/challenge-pr` on #124) | [S] 2604.01978, 2601.21942; [L] 2604.26898 |
| 2 | **2605.09213** *Kinetic theory for Transformers and the lost-in-the-middle phenomenon* (Duerinckx, Geshkovski, Rossi): causal attention as a non-exchangeable particle system; for iid uniform tokens the correlation equation is solved in closed form, retrieval profile U-shaped (primacy, recency, one interior minimum under a smallness condition). With **2411.04990** Thm 4.1 (`docs/readings/2411.04990.md`, [R]): all tokens collapse to `x₁(0)` | 1d's **position finding**: step 0's admitted groups are the prompt's opening; the cut M = 32 was picked on the control (Blocked 11′) | Whether a closed-form primacy profile agrees with the prefix-average account `status-1d.md` already gives. Not the route to Blocked 11′: that account yields a rule-fixed cut without the paper. Open: whether iid-uniform tokens with fixed weights resemble Pythia at step 0 | [S] |
| 3 | **2412.09080** *On the number of modes of Gaussian kernel density estimators* (Geshkovski, Rigollet, Sun): Gaussian KDE with bandwidth `β^{-1/2}`; expected modes on the line `Θ(√(β log β))` for `n^c ≲ β ≲ n^{2−c}`; stated motive: how many clusters a transformer is drawn to in a metastable state (mean-shift ↔ self-attention) | 1d's **"the theory's own definition"** (`STATE.md` Now, last line; `p1d_cluster_ensemble/lit-1d.md` §3 `φ_β` wells, C's `δ = cβ^{-1/2}`) | A published definition of a metastable cluster as a KDE mode at bandwidth `β^{-1/2}`, with a count to compare against. 1D and Gaussian samples only; whether a d = 1024 version exists is a reading question | [S] |
| 4 | **2410.23228**, **2509.25040** (Bruno, Pasqualotto, Agazzi): metastable clusters from the linearisation around uniform, periodicity set by β; with β ∝ N, collapse, clustering, slow pairwise merging | 1d count; already queued in `lit-1d.md` §6 | Whether their count ("~√β clusters", [L]) has a d = 1024 form with known constants; `lit-1d.md` §6 already asks this of the Gegenbauer index. Any comparison with 1d's counts also needs β's convention (**Blocked 9**). No number is set beside a measurement here until both are known | [S] 2410.23228; repo [S] 2509.25040 |
| 5 | **2601.21366** *Perceptrons and localization of attention's mean-field landscape* (Álvarez-López, Geshkovski, Ruiz-Balet; ICML 2026 spotlight): with the MLP block, critical points are generically atomic and localised. [L]: cluster mass ≤ ≈ 0.5742, heavy atoms ∝ √β | 1d merge tree: "one cluster plus outliers" in 126 of 192 layer-records (`status-1d.md` "Merge tree…") | If the mass bound holds in their model, Pythia's longest-lived scale is far from its stationary points. A contrast to state, not a test: their object is stationary measures of an idealised flow, ours is layer snapshots | [S]; bound [L] |
| 6 | **2510.05554** *Critical attention scaling* (S. Chen, Z. Lin, Polyanskiy, Rigollet; ICLR 2026): critical `β_n = log n`; below, all tokens collapse; above, attention tends to identity. **2605.08505** *Scaling Limits of Long-Context Transformers* (Bruno, S. Chen, Z. Lin, Polyanskiy, Rigollet): `β*_n ≍ n^{2/(d−1)}` for uniform keys on `S^{d−1}`; sub-critical, critical, super-critical regimes | 1d long prompts (`status-1d.md` "Long prompts"): β 3.12 on v1's offsets at ~2000 tokens vs 3.46 at v1 length; the blob halves with length (69 → 43 of 96 layers) | Which critical scale applies to a model with no length-dependent scaling, and in which units. The two disagree at d = 1024 (`log n` is 6–8 at these n, `n^{2/(d−1)}` ≈ 1), so neither places Pythia's β until the constants and the role of dimension are read. Also bears on `p1c_frames/lit-1c.md` §5 item 3's attribution | [S] |
| 7 | **2510.22026** *Normalization in Attention Dynamics* (Karagodin, Ge, Polyanskiy, Rigollet; NeurIPS 2025): normalization as a speed factor in one particle ODE across Post-, Pre-, Mix-, Peri-LN, nGPT. [L]: Post-LN exponential, Pre-LN ~1/t² contraction | Any depth-rate statement (Phase 1 energy, 1c's `T_eff`). Pythia is Pre-LN with parallel attention and MLP | Which contraction law a depth-rate fit should be compared with | [S]; rates [L] |
| 8 | Training-dynamics mean field (list §b–d): **2608.25055** Herty–Liu, **2605.17660** Barboni–de Hoop–Furuya–Peyré, 2410.23610, 2402.01258, 2606.10469, 2405.15712, 2607.05735 | `FUTURE_IDEAS.md` D7; `docs/TRIAGE_2026-09.md` flagship (training axis) | Not the active thread. Only 2608.25055 and 2605.17660 are in the repo | [L] |

**Not on the list, found by the same searches:** **2609.28448** *Nonequilibrium Phases of
Repulsive Self-Attention: Chaos, Attention Condensation, and Emergent Locality* [N]; bears on
the OV repulsive phase (`PROJECT.md` §3.12 A).

**In the repo, missing from the list** (so the list is not a superset): 2605.07772
(Isobe–Inoue–Imaizumi, trained FFN leaves the clustered regime near the last layers),
2406.07247 (DMFT of self-attention), 2605.18870 (multi-head as time-dependent Wasserstein
flows), 2604.23740 (Euler discretisation on the sphere), 2605.04279 (gradient-flow
structure of multi-head attention), 2501.10573 (intrinsic dimension of prompts, [R]),
2507.00683 (spin-bath view).

## 2. Cross-check against the repo

22 of the list's 61 arXiv ids were already cited (grep of `*.md`, `*.py`, `*.json`,
2026-10-01). Where:

| id | where |
|---|---|
| 2312.10794 | 16 files (project-wide citation) |
| 2411.04990 | 29 files; `docs/readings/2411.04990.md` is [R] |
| 2410.06833 | `PROJECT.md`, `p1_mstate_tracking/lit-1.md`, `PUBLICATION_IDEAS.md` (+3) |
| 2410.23228 | `p1d_cluster_ensemble/lit-1d.md` §6 queue, `p10_cluster_function/lit-10.md`, `PROJECT.md`, `PUBLICATION_IDEAS.md`, `docs/readings/2411.04990.md` |
| 2509.25040 | `p1d_cluster_ensemble/lit-1d.md` only |
| 2601.21366, 2608.08922, 2601.21942 | `p1_mstate_tracking/lit-1.md` [N]; first two also `lit-1d.md` §6 |
| 2605.10931 | `p1c_frames/lit-1c.md`, `PUBLICATION_IDEAS.md` |
| 2605.08505 | `p1c_frames/lit-1c.md` only |
| 2604.26085 | `p2_eigenspectra/lit-2.md`, `p2b_imaginary/lit-2b.md` [N] |
| 2607.24502 | `p2b_imaginary/lit-2b.md`, `p6_subspace/lit-6.md` [N], `docs/LITERATURE.md` §5, `PUBLICATION_IDEAS.md` |
| 2305.05465 | `lit-1.md`, `lit-1d.md`, `PUBLICATION_IDEAS.md` |
| 2512.01868, 2110.11773, 2501.18322, 2411.04551, 2510.22026, 2510.05554, 2605.09213 | `PUBLICATION_IDEAS.md` only |
| 2608.25055, 2605.17660 | `FUTURE_IDEAS.md` only |

The other 39 are new to the repo. **Corrections and confirmations made by this unit:**

- `p1c_frames/lit-1c.md` §5 item 3 flagged `β*_n ≍ n^{2/(d−1)}` → 2605.08505 as an
  unreliable attribution. A search summary of 2605.08505's own page now gives that
  result and the authors. Consistent at [S], not independent confirmation: both are
  search summaries. Edited there to say so; it stays a reading item.
- `p1d_cluster_ensemble/lit-1d.md` names 2608.08922's authors "Gao, Yang & Chen", which
  the list leaves unchecked. Search: Qucheng Gao, Zuyi Yang, Xiao Chen [S]. Agrees.
- `FUTURE_IDEAS.md` cited 2608.25055 without authors and 2605.17660 as "Barboni" alone;
  authors added from the list ([L]).
- The list's own corrections ("Hardy" → Herty; "Sra" → Jadbabaie; Ziang vs Shi Chen)
  concern names the repo never carried.

## 3. Search log (2026-10-01)

WebSearch, 11 queries: 2604.01978; 2601.21366; 2412.09080; 2605.08505; 2605.09213; GitHub
for companion code (only `borjanG/2023-transformers`, for 2305.05465); 2601.21942;
2510.05554; 2608.08922; 2510.22026; 2410.23228. No companion repository found for rows 1–7.

## 4. Ask: PDFs, ranked

Supplied PDFs are read in full and recorded as `docs/readings/<id>.md` (as 2411.04990 was).

| rank | id | why first |
|---|---|---|
| 1 | 2604.01978 Homogenized Transformers | Theory of the step-0 control (row 1); secondary for Blocked 11′ |
| 2 | 2605.09213 Kinetic theory / lost-in-the-middle | Position under causal attention, against `status-1d.md`'s prefix-average account (row 2) |
| 3 | 2412.09080 Modes of Gaussian KDEs | A theory definition and count for "what a cluster is" (row 3) |
| 4 | 2509.25040, 2410.23228 Bruno–Pasqualotto–Agazzi | The count's constants; already in `lit-1d.md` §6 (row 4) |
| 5 | 2601.21366 Perceptrons and localization | Mass bound and atom count with the MLP (row 5) |
| 6 | 2512.01868 Rigollet survey | Its bibliography is the map; one read replaces many [L] marks |
| 7 | 2601.21942 Deep stochastic transformers | Random V at init (row 1) |
| 8 | 2605.08505, 2510.05554 | Length scaling (row 6) |

## 5. Caveats carried from the list

Many 2026 entries are unrefereed preprints; several were checked only by title and id. The
results are mostly for `QᵀK = βI`, `V = I`, fixed weights, unit sphere, no MLP. Several of
the papers say their theorems describe tendencies, not trained-LLM behaviour.

---

## 6. The list as supplied (verbatim, 2026-10-01)

# Reading List: Mean-Field Dynamics of Transformers and Related Architectures (as of 1 Oct 2026)

The tokens-as-particles program started by Geshkovski, Letrouit, Polyanskiy and Rigollet has grown into a large, fast-moving literature. Its core is clustering and metastability on the sphere. Since late 2025 it has branched into normalization, causal masking and kinetic theory, MLP blocks, stochastic/random-weight limits, and long-context scaling. It is now merging with a second line of work, "neurons/heads-as-particles" training dynamics, in coupled data–parameter mean-field models; the Herty–Liu paper you already know is the most recent example.

## TL;DR
- **Your three anchors are verified, with two corrections.** Paper 2 is by **Michael Herty and Hailiang Liu** (not "Michael Hardy"), arXiv:2608.25055, submitted 25 Aug 2026. Paper 3 is **"Width-Robust Learnability in Mean-Field Bayesian Neural Networks"** by **Dmitry Vaintrob and Kaarel Hänni**, arXiv:2607.05735 (Principles of Intelligence). Rigollet's paper is arXiv:2512.01868 (v4, 30 Jan 2026), to appear in the ICM 2026 Proceedings.
- **Read first (tokens-as-particles):** Quantitative clustering (Chen–Lin–Polyanskiy–Rigollet), Bruno–Pasqualotto–Agazzi's two papers (ICLR 2025; NeurIPS 2025 oral), Normalization in attention dynamics, Perceptrons and localization (Álvarez-López–Geshkovski–Ruiz-Balet), Homogenized transformers, Kinetic theory/lost-in-the-middle, and Stochastic scaling limits (Agazzi et al.).
- **Read first (training-dynamics mean field for transformers):** Gao et al. "Global Convergence in Training Large-Scale Transformers" (NeurIPS 2024), Barboni–de Hoop–Furuya–Peyré "Training Infinitely Deep and Wide Transformers" (2026), Kim–Suzuki (ICML 2024 oral), Bordelon–Chaudhry–Pehlevan DMFT (NeurIPS 2024), and Huan–Yuan heads-as-particles (2026). These are the direct predecessors and competitors of Herty–Liu.

## Already known (anchors, verified)
- **P. Rigollet, "The Mean-Field Dynamics of Transformers,"** arXiv:2512.01868 (Dec 2025; v4 Jan 2026; ICM 2026 Proceedings). A survey-style synthesis covering the SA/USA flows on the sphere, Wasserstein gradient-flow structure, the Kuramoto connection, metastability, the equiangular model, normalization and long context, and noisy transformers. Its bibliography is the best map of the literature below.
- **M. Herty, H. Liu, "A Mean-Field Theory of Transformers: Well-Posedness of the Coupled Data–Parameter Dynamics and Global Convergence of Training,"** arXiv:2608.25055 (math.AP, 25 Aug 2026). It couples a McKean–Vlasov token flow (N→∞ tokens) with a Wasserstein-gradient-flow/Fokker–Planck evolution of the attention-head distribution (H→∞ heads). It proves global exponential convergence only for a single mean-field attention layer under log-Sobolev, and local linear convergence for deep models under NTK non-degeneracy.
- **D. Vaintrob, K. Hänni, "Width-Robust Learnability in Mean-Field Bayesian Neural Networks,"** arXiv:2607.05735 (stat.ML, July 2026; Vaintrob affiliated with Principles of Intelligence). The main theorem: at fixed depth, a family of Boolean-cube targets is learnable from polynomially many samples at infinite width iff it is learnable at polynomial width, iff its reduced entropy is polynomially bounded. In other words, the mean-field (γ=1, critical feature-learning) Bayesian limit adds no spurious width-dependent generalization power. This is a neurons-as-particles Bayesian/Gibbs-posterior result, not a token-dynamics one, which matches your "different but relevant" intuition.

## (a) Tokens-as-particles: clustering, metastability, normalization, masking

**Foundations (Geshkovski–Letrouit–Polyanskiy–Rigollet core)**
- **Geshkovski, Letrouit, Polyanskiy, Rigollet — "A Mathematical Perspective on Transformers,"** arXiv:2312.10794; *Bull. AMS* 62(3):427–479 (2025). The founding framework: tokens are particles on S^{d−1}, layers are time, and the dynamics is a continuity equation that is a gradient flow (Hessian-type metric for SA, Wasserstein for USA). It includes long-time clustering, the equiangular model, and the ALBERT empirics.
- **Geshkovski, Letrouit, Polyanskiy, Rigollet — "The Emergence of Clusters in Self-Attention Dynamics,"** arXiv:2305.05465; NeurIPS 2023. Shows that the limiting geometry of clustering (points, hyperplanes, polytopes) depends on the spectrum of the value matrix V, with time-independent weights.
- **Sander, Ablin, Blondel, Peyré — "Sinkformers: Transformers with Doubly Stochastic Attention,"** arXiv:2110.11773; AISTATS 2022. The original particle/measure view of attention without normalization. Sinkhorn-normalized attention yields a Wasserstein gradient flow in the infinite-depth, mean-field limit.
- **Castin, Ablin, Carrillo, Peyré — "A Unified Perspective on the Dynamics of Deep Transformers,"** arXiv:2501.18322 (2025). Treats the "Transformer PDE" as a Vlasov equation, with well-posedness and analysis of compactly supported and Gaussian initial data (Gaussians are invariant, which reduces the dynamics to a mean/covariance system).
- **Geshkovski, Rigollet, Ruiz-Balet — "Measure-to-Measure Interpolation Using Transformers,"** arXiv:2411.04551 (2024). A controllability/expressivity counterpart: transformers viewed as maps between probability measures.

**Clustering rates and metastability**
- **Geshkovski, Koubbi, Polyanskiy, Rigollet — "Dynamic Metastability in the Self-Attention Model,"** arXiv:2410.06833 (2024). Proves exponentially long-lived multi-cluster metastable states (Otto–Reznikoff slow-motion framework) and saddle-to-saddle "staircase" energy profiles.
- **S. Chen, Z. Lin, Polyanskiy, Rigollet — "Quantitative Clustering in Mean-Field Transformer Models,"** arXiv:2504.14697 (2025). Exponential W₂ convergence of the mean-field PDE to a Dirac mass from regular initial data for small β. It extends Morales–Poyato's Kuramoto analysis to the sphere, and gives a counterexample showing that multi-cluster limits are possible for large β.
- **Bruno, Pasqualotto, Agazzi — "Emergence of Meta-Stable Clustering in Mean-Field Transformer Models,"** arXiv:2410.23228; ICLR 2025. Linearizes the mean-field PDE around the uniform measure and shows perturbations grow into ~√β periodic clusters (the first metastable state).
- **Bruno, Pasqualotto, Agazzi — "A Multiscale Analysis of Mean-Field Transformers in the Moderate Interaction Regime,"** arXiv:2509.25040; NeurIPS 2025 (oral). With β scaling with N, they find fast collapse onto a low-dimensional set, then intermediate clustering, then slow sequential merging (a hardmax-like pairing phase). The large-β limit gives porous-medium-type PDEs.
- **Alcalde, Bungert, Riedl, Roith — "Quantifying Concentration Phenomena of Mean-Field Transformers in the Low-Temperature Regime,"** arXiv:2605.10931 (2026). For general K/Q/V, the token law concentrates onto a pushforward under a projection map at rate √(log(β+1)/β)·e^{Ct}+e^{−ct}, then stays metastable. It uses consensus-based-optimization tools.
- **Polyanskiy, Rigollet, Yao — "Synchronization of Mean-Field Models on the Circle,"** arXiv:2507.22857 (2025). Closes the d=2 gap left by sphere-synchronization theorems. The abstract reports synchronization for all β ≥ −0.16, which "significantly extends the previous bound of 0≤β≤1" from Criscitiello et al. (2024), and shows that global synchronization does not occur when β < −2/3.
- **Criscitiello, Rebjock, McRae, Boumal — "Synchronization on Circles and Spheres with Nonlinear Interactions,"** arXiv:2405.18273 (2024). The general synchronization theorem underlying almost-sure single-cluster convergence for d ≥ 3.
- **Z. Chen, Polyanskiy, Rigollet — "Clustering in Self-Attention Dynamics with Wasserstein–Fisher–Rao Gradient Flows,"** NeurIPS 2025 workshop "Dynamics at the Frontiers of Optimization, Sampling, and Games." Weighted tokens are transported and reweighted, and clustering still holds. *No arXiv ID found. Note the first author is Ziang Chen, not Shi Chen.*
- **Geshkovski, Rigollet, Sun — "On the Number of Modes of Gaussian Kernel Density Estimators,"** arXiv:2412.09080 (2024). Uses the mean-shift ↔ self-attention equivalence to count metastable clusters, getting E[M] ≍ √(β log β) in 1D.

**Normalization, masking, positional encodings, architecture variants**
- **Karagodin, Ge, Polyanskiy, Rigollet — "Normalization in Attention Dynamics,"** arXiv:2510.22026; NeurIPS 2025. Shows that normalization acts as speed regulation, in a unified analysis of Post-/Pre-/Mix-/Peri-LN, nGPT and LN-scaling. Post-LN gives exponential contraction while Pre-LN gives polynomial contraction (~1/t²). The paper favors Peri-LN.
- **Karagodin, Polyanskiy, Rigollet — "Clustering in Causal Attention Masking,"** arXiv:2411.04990; NeurIPS 2024. Studies decoder-style (non-exchangeable) causal dynamics; the result is clustering toward the first token, with connections to Rényi parking.
- **Wu, Ajorlou, Wang, Jegelka, Jadbabaie — "On the Role of Attention Masks and LayerNorm in Transformers,"** arXiv:2405.18781; NeurIPS 2024. A graph-theoretic analysis of rank collapse: masks slow collapse, and LayerNorm with suitable V admits equilibria of any rank. *Correction to your list: authors are Wu–Ajorlou–Wang–Jegelka–Jadbabaie, not "…–Sra."*
- **Burger, Kabri, Korolev, Roith, Weigand — "Analysis of Mean-Field Models Arising from Self-Attention Dynamics in Transformer Architectures with Layer Normalization,"** arXiv:2501.03096; *Phil. Trans. R. Soc. A* 383(2298):20240233 (2025). A rigorous gradient-flow framework on the sphere with an aggregation-equation viewpoint. It characterizes stationary points and the cluster-vs-uniform regimes.
- **Duerinckx, Geshkovski, Rossi — "Kinetic Theory for Transformers and the Lost-in-the-Middle Phenomenon,"** arXiv:2605.09213 (May 2026). Treats causal attention as a non-exchangeable particle system using cumulant expansions. The closed-form correlation equation gives a U-shaped retrieval profile, a rigorous account of lost-in-the-middle.
- **Álvarez-López, Geshkovski, Ruiz-Balet — "Perceptrons and Localization of Attention's Mean-Field Landscape,"** arXiv:2601.21366 (Jan 2026). Adds the MLP block as an external potential drift. Stationary measures become generically atomic even in the repulsive regime, with cluster mass bounded by ≈0.5742 and the number of heavy atoms ∝ √β.
- **Alcalde, Fantuzzi, Zuazua — "Clustering in Pure-Attention Hardmax Transformers and Its Role in Sentiment Analysis,"** *SIAM J. Math. Data Sci.* 7(3):1367–1393 (2025). The β→∞ (hardmax) dynamics; this is the limiting rule used in Bruno et al.'s pairing phase.
- **Alcalde, Geshkovski, Ruiz-Balet — "Attention's Forward Pass and Frank–Wolfe,"** arXiv:2508.09628 (2025). An optimization-algorithm interpretation of the attention forward pass.
- **Abella, Silvestre, Tabuada — "Consensus Is All You Get: The Role of Attention in Transformers,"** ICML 2025 (companion preprint arXiv:2412.02682, "The Asymptotic Behavior of Attention in Transformers"). A control-theoretic consensus proof for general attention dynamics.
- **2026 preprints on extensions** (titles verified via citations; authors not individually checked):
  - "Krause Synchronization Transformers" (arXiv:2602.11534): bounded-confidence attention that supports multi-cluster states and reduces attention sinks.
  - "Self-Attention Dynamics with Rotary Position Embeddings: Twisted States and Explicit Consensus Rates on the Sphere" (arXiv:2607.24502): RoPE.
  - "Spectral Selection in Symmetric Self-Attention Dynamics" (arXiv:2604.26085).
  - "On the Diverse Dynamical Behaviors Arising in Deep Linear Transformers" (arXiv:2607.18584).
  - "Clustered Attractor Manifolds and Dynamical Condensation in Self-Attention" (arXiv:2608.08922): a statistical-physics/REM angle.
  - Altafini, "Multistability of Self-Attention Dynamics in Transformers" (arXiv:2511.11553).

**Long-context scaling**
- **S. Chen, Z. Lin, Polyanskiy, Rigollet — "Critical Attention Scaling in Long-Context Transformers,"** arXiv:2510.05554; ICLR 2026. Shows a phase transition at β_n = γ log n (the equiangular model) and justifies log-n scaling as used in Qwen/SSMax.
- **Bruno, S. Chen, Z. Lin, Polyanskiy, Rigollet — "Scaling Limits of Long-Context Transformers,"** arXiv:2605.08505 (May 2026). Uses order statistics and extreme-value theory of the closest keys; the critical scale is β*_n ≍ n^{2/(d−1)} for uniform keys. In the subcritical regime, attention approximately implements a backward heat equation.

**Noise and random weights (stochastic mean-field limits)**
- **Balasubramanian, Banerjee, Rigollet — "On the Structure of Stationary Solutions to McKean–Vlasov Equations with Applications to Noisy Transformers,"** arXiv:2510.20094 (2025). Bifurcations of the noisy transformer Fokker–Planck equation on the circle.
- **Shalova, Schlichting — "Solutions of Stationary McKean–Vlasov Equation on a High-Dimensional Sphere and Other Riemannian Manifolds,"** arXiv:2412.14813 (2024). Bifurcation structure of noisy attention on spheres.
- **Koubbi, Geshkovski, Rigollet — "Homogenized Transformers,"** arXiv:2604.01978 (Apr 2026). Weights are resampled across layers and heads, as at initialization. Under joint depth/step/heads scalings there is a homogenized (deterministic or SDE) limit, which gives trade-offs between dimension, context length and temperature for avoiding collapse.
- **Agazzi, Bruno, Mosig García, Saviozzi, Romito — "Stochastic Scaling Limits and Synchronization by Noise in Deep Transformer Models,"** arXiv:2604.26898 (Apr 2026). Pathwise convergence, including MLP blocks, to a stochastic interacting particle system and SPDE. It proves propagation of chaos with commuting limits and synchronization by common noise.
- **Fedorov, Sander, Elie, Marion, Laurière — "Clustering in Deep Stochastic Transformers,"** arXiv:2601.21942; ICML 2026. Random value matrices give an SDE limit on the sphere with a noise-driven phase transition, including antipodal configurations.
- **Gibson — "Uniform Scaling Limits in AdamW-Trained Transformers,"** arXiv:2605.11059 (2026). Large-depth/heads IPS limit for *trained* (AdamW) attention-only transformers.
- **Related:** "Random Quadratic Form with Random Forcing: Metastable Synchronization by Noise" (arXiv:2608.16664) and Engel–Shalova "Random Quadratic Form on a Sphere: Synchronization by Common Noise" (arXiv:2603.06187).
- **"Understanding Catastrophic Forgetting in LoRA via Mean-Field Attention Dynamics,"** arXiv:2402.15415 (v2). According to the abstract, there are two phase transitions in representation drift: "one phase transition appears with respect to the norm of the perturbation, and the other with respect to the depth of the Transformers." The paper also bounds time-to-deviation in terms of perturbation size and spectral quantities. *v1 was titled "The Impact of LoRA on the Emergence of Clusters in Transformers"; check the authors.*

**Signal propagation (adjacent, physics-style)**
- **Cowsik, Nebabu, Qi, Ganguli — "Geometric Dynamics of Signal Propagation Predict Trainability of Transformers,"** arXiv:2403.02579 (2024). Finds "not 2 but 4 distinct phases," including "an ordered phase where the n token representations converge and collapse to a line, and a chaotic phase where the n token representations chaotically repulse each other and converge to a regular n-simplex." According to Álvarez-López et al., the resulting scaling laws were used in OLMo-2.
- **Giorlandino, Goldt — "Two Failure Modes of Deep Transformers and How to Avoid Them,"** arXiv:2505.24333 (2025). A unified signal-propagation theory at initialization.
- **Noci et al. — "Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse,"** NeurIPS 2022.

## (b) Mean-field training dynamics for attention/transformers (parameters as particles)
- **Gao, Cao, Li, He, Wang, H. Liu, Klusowski, Fan — "Global Convergence in Training Large-Scale Transformers,"** arXiv:2410.23610; NeurIPS 2024. Width/depth→∞ mean-field limit; gradient flow converges to a Wasserstein-gradient-flow PDE and reaches a global minimum for small weight decay. The abstract contrasts this with Lu et al. (2020), which "demand homogeneity and global Lipschitz smoothness"; Gao et al. assume "only partial homogeneity and local Lipschitz smoothness."
- **Barboni, de Hoop, Furuya, Peyré — "Training Infinitely Deep and Wide Transformers,"** arXiv:2605.17660 (May 2026). Extends conditional-OT ResNet theory to transformers: token distributions follow a continuity equation and the head distribution is trained by a conditional Wasserstein gradient flow ("neural PDE" control). Global minimum from small initial loss under attention-NTK injectivity. Herty–Liu name this as their closest precursor.
- **Kim, Suzuki — "Transformers Learn Nonlinear Features in Context: Nonconvex Mean-field Dynamics on the Attention Landscape,"** arXiv:2402.01258; ICML 2024 (oral). MLP followed by linear attention in a mean-field, two-timescale limit. The landscape is benign, and the authors (U. Tokyo/RIKEN AIP; PMLR 235) prove that the "Wasserstein gradient flow almost always avoids saddle points," calling this "the first saddle point analysis of mean-field dynamics in general."
- **Huan, Yuan — "A Mean-Field Analysis of Multi-Head Self-Attention under Cross-Entropy Training,"** arXiv:2606.10469 (2026). Heads are the particles. Results: an O(N^{-1/2}) static approximation, a support condition for global minimizers, finite-time propagation of chaos for SGD, KL-rate convergence to stationarity, and Dirac stability/instability via a "translation Hessian."
- **Bordelon, Chaudhry, Pehlevan — "Infinite Limits of Multi-head Transformer Dynamics,"** arXiv:2405.15712; NeurIPS 2024. DMFT for feature-learning (μP-type) limits in key/query dimension, heads, and depth. The N→∞ limit collapses multi-head to single-head behavior, while H→∞ gives a deterministic distribution over heads.
- **"Convergent Stochastic Training of Multi-Headed Attention and Understanding LoRA,"** arXiv:2605.07959 (2026). Langevin-type convergence for multi-head attention. *Authors not verified.*

## (c) Background: mean-field limits of two-layer, multilayer, and ResNet networks
- **Mei, Montanari, Nguyen — "A Mean Field View of the Landscape of Two-Layer Neural Networks,"** arXiv:1804.06561; PNAS 2018. Distributional dynamics PDE for SGD.
- **Chizat, Bach — "On the Global Convergence of Gradient Descent for Over-parameterized Models using Optimal Transport,"** arXiv:1805.09545; NeurIPS 2018. Wasserstein gradient flow and global convergence for homogeneous models.
- **Rotskoff, Vanden-Eijnden — "Trainability and Accuracy of Neural Networks: An Interacting Particle System Approach,"** arXiv:1805.00915; CPAM 2022.
- **Sirignano, Spiliopoulos — "Mean Field Analysis of Neural Networks: A Law of Large Numbers,"** arXiv:1805.01053; SIAM J. Appl. Math. 2020. The CLT companion is arXiv:1808.09372.
- **Mei, Misiakiewicz, Montanari — "Mean-Field Theory of Two-Layers Neural Networks: Dimension-Free Bounds and Kernel Limit,"** COLT 2019.
- **Nguyen, Pham — "A Rigorous Framework for the Mean Field Limit of Multilayer Neural Networks,"** arXiv:2001.11443 (2020).
- **Pham, Nguyen — "Global Convergence of Three-Layer Neural Networks in the Mean Field Regime,"** arXiv:2105.05228; ICLR 2021.
- **Lu, Ma, Lu, Lu, Ying — "A Mean-field Analysis of Deep ResNet and Beyond: Towards Provable Optimization via Overparameterization from Depth,"** arXiv:2003.05508; ICML 2020.
- **Ding, Chen, Li, Wright — "Overparameterization of Deep ResNet: Zero Loss and Mean-Field Analysis,"** arXiv:2105.14417; JMLR 23(48) (2022). A depth- and width-limit PDE converging to zero loss.
- **Barboni, Peyré, Vialard — "Understanding the Training of Infinitely Deep and Wide ResNets with Conditional Optimal Transport,"** arXiv:2403.12887; *Comm. Pure Appl. Math.* 78(11) (2025). The direct template for the 2026 transformer extension.
- **E, Han, Li — "A Mean-Field Optimal Control Formulation of Deep Learning,"** *Res. Math. Sci.* 6:10 (2019). The mean-field-control/PMP viewpoint.
- **Yang, Hu — "Feature Learning in Infinite-Width Neural Networks" (Tensor Programs IV),** arXiv:2011.14522; ICML 2021. Defines μP, the feature-learning limit.
- **Bordelon, Pehlevan — "Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks,"** arXiv:2205.09653; NeurIPS 2022. The DMFT basis for the transformer limits above.

## (d) Bayesian and other mean-field variants (Principles of Intelligence context)
- **Vaintrob, Hänni (2026)**, arXiv:2607.05735: already known, see above. The abstract also states that "the infinite-width mean-field limit gives a clean analytic description of learning without introducing spurious width-dependent generalization power."
- **Principles of Intelligence (PrincInt, formerly PIBBSS) — PIRAMID research group** (Lauren Greenspan, Dmitry Vaintrob, Ari Brill, Andrew Mack, Nischal Mainali, Jennifer Lin). The PIRAMID page lists "Mean field sequence," "A tale of three theories," and the Vaintrob–Hänni preprint as recent learning-theory work. These posts are not peer-reviewed:
  - **Vaintrob, Greenspan — "Mean Field Sequence: An Introduction,"** LessWrong, 4 Apr 2026. Presents "adaptive mean field theory," which treats neurons as gas-like particles, as a tool for interpretability.
  - **Vaintrob — "A Tale of Three Theories: Sparsity, Frustration, and Statistical Field Theory,"** LessWrong. Connects computation in superposition with mean-field and frustration physics; a formal paper is said to be planned.
- **Related statistical-mechanics Bayesian work** (not verified in detail): kernel-adaptation and "critical feature learning" theories (Seroussi–Naveh–Ringel, *Nat. Commun.* 2023; Fischer et al., ICML 2024; Rubin et al., ICML 2025). These are the closest academic neighbors of the PrincInt approach.

## (e) Surveys, notes, entry points
- Rigollet (2025/26, ICM), arXiv:2512.01868, and Geshkovski et al. *Bull. AMS* (2025): read these two first.
- Chewi, Niles-Weed, Rigollet — *Statistical Optimal Transport*, Springer LNM 2364 (2025). Background on Wasserstein gradient flows.
- Ambrosio–Gigli–Savaré, *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. The standard reference.

## Recommendations
1. **For the Herty–Liu style coupled problem**, read in this order: Barboni–Peyré–Vialard (2024) → Gao et al. (2024) → Barboni–de Hoop–Furuya–Peyré (2026) → Herty–Liu. Herty–Liu name the gap between shallow global and deep local convergence as the central open question. This is the most promising place for new results.
2. **For tokens-as-particles depth**, read the Bull. AMS paper, then Quantitative clustering, then both Bruno–Pasqualotto–Agazzi papers, then Perceptrons/localization. The MLP drift and random-weight (homogenized/stochastic) limits are the 2026 frontier: they are where single-cluster collapse stops being the generic outcome.
3. **For PrincInt alignment**, pair Vaintrob–Hänni with Yang–Hu (μP) and Bordelon–Pehlevan DMFT. The PrincInt line is neurons-as-particles Bayesian mean field aimed at interpretability, not token dynamics. Showing fluency in both would set you apart.

## Caveats
- Many 2026 entries are recent preprints that have not been peer-reviewed. For several, only the title and ID were verified through citing papers; they are marked above.
- The Chen–Polyanskiy–Rigollet WFR paper has no arXiv ID that I could locate.
- Some venue details (AISTATS, ICLR, CPAM, ICML) come from citing documents rather than publisher pages.
- Results throughout are for idealized models: often Q^⊤K = βI and V = I, fixed weights, unit sphere, and no MLP. Several papers stress that clustering theorems describe tendencies (collapse vs. metastable multi-cluster states), not trained-LLM behavior.
