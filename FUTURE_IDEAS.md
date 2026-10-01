# FUTURE_IDEAS — how training chooses the weights the particle picture runs on

**Opened:** 2026-09-29. **Tier 1**: a literature scan and a question log. Nothing
here was run or registered. The user asked for it in a docs-only session
("don't run anything, we're just thinking").

**How it was read:** about 30 web searches. arXiv and timaeus.co were blocked
by this container's egress proxy, so every 2025–26 entry below was read
**at abstract/search-snippet level only**. Before a claim here is used in a
design, open the paper.

---

## 0. The question, in one paragraph

The theory this project tests (Geshkovski et al.) proves clustering for
**chosen** weights: `Q = K = V = I`, or more generally weights satisfying a
theorem's hypotheses, tied across layers, no MLP (`p10_cluster_function/lit-10.md`
§11.1). Pythia's weights are not chosen. They start random (step 0) and
training (Adam on the Pile) moves them into a small subset of weight space.
The project measures what happens to tokens (activation space) at checkpoints
along that path, but treats each checkpoint's weights as given. **The missing
object is the map from (initialisation, data, optimiser, architecture) to the
weights training ends up with, and from there to token dynamics.** Below it is
called the *selection map*. The theory covers one point of weight space
(identity). Signal-propagation theory covers another (random init). Nothing in
this repo covers the path between them.

## 1. What the user is proposing, split into claims that can be checked

| # | Claim | Status | How to check it |
|---|---|---|---|
| C1 | "Identity weights ≈ random weights ≈ trained weights" for the clustering result | Assumed, untested | Recast as: *do trained weights stay inside the hypothesis class of the theorem being tested?* That is a per-head, per-checkpoint measurement on weights alone (§3 D1) |
| C2 | The optimiser (not only data and architecture) shapes which weights are reached | Well supported in general (§2 A); untested for token clustering | Compare seeds (D2) and optimisers (D5) at matched data |
| C3 | The shaping can be described by a few principles, the way two axioms fixed special relativity | Open. Several partial languages exist (§2) and none is complete | §4 |
| C4 | Activation space is where interpretation should happen; weight space is the cause | A position, not a result. Partly backed by weight-space symmetries (§2 E) and representation convergence (§2 F) | §4 |
| C5 | Biology uses a different learning rule, so SGD's selection may not be the only way to reach similar activations | Supported in outline (§2 G) | Out of scope for Pythia; a framing check |

## 2. What is known about how training restricts weights

Five non-equivalent languages. Each answers a different question. None has
been connected to token clustering in a real language model; that gap is the
opening for this project.

### A. Implicit bias: *which* minimum the optimiser picks

| Mechanism | What it selects | Applies to Pythia? | Source |
|---|---|---|---|
| Adam(W) ≈ smoothed sign descent | KKT points under an **ℓ∞ bound** `‖θ‖∞ ≤ 1/λ` (full batch, if it converges) | Pythia trained with Adam; verify its weight decay and schedule from the Pythia config before using | Xie & Li, ICML 2024 |
| Adam vs Muon | Adam → max-norm margin; Muon → **spectral-norm** margin; Muon weights have higher stable rank | Contrast case only; Pythia is Adam | 2602.16340; 2506.15054; 2502.04664 |
| SGD with label noise, after zero loss | Drifts along the zero-loss manifold minimising **tr(Hessian)** (flatness) | Weak: LM training never reaches zero loss | Li, Wang & Arora, ICLR 2022 |
| Adam's diagonal preconditioning | **Outlier features / rogue coordinates**; non-diagonal preconditioners (Shampoo, K-FAC) reduce them | **Direct**: 1d's Gaussian null removes "3 rogue coordinates" (`p1d_cluster_ensemble/status-1d.md`) | He et al., NeurIPS 2024; Puccetti et al. 2022 (frequency-driven) |
| Weight decay + learning rate | **Attention sinks** appear after ~1–2k steps; more weight decay → more sink heads | **Direct**: 1d's token-0 results | Gu et al., ICLR 2025 |

### B. Dynamics: *which path* training takes

| Finding | Content | Source |
|---|---|---|
| Edge of stability / central flows | The oscillating optimiser's time-average follows an ODE; adaptive optimisers seek regions that allow larger steps | Cohen et al., ICLR 2025; Adam version 2605.06821; perturbative derivation 2609.01034 |
| Gradient lives in a tiny subspace | After early training, gradients concentrate in the top-Hessian eigenspace, which drifts slowly | Gur-Ari, Roberts & Dyer 2018 |
| Incremental rank | `ΔW = W_t − W_0` grows in stable rank step by step (GPT-2 on Wikitext: from ~1 upward) | Boix-Adserà et al., NeurIPS 2023 |
| Weight spectral phases | Layer ESDs go random-like → bleeding-out → bulk+spikes → heavy-tailed → rank collapse | Martin & Mahoney, JMLR 2021 |
| **QK geometry from the objective** | Bidirectional training → **symmetric** `W_QK`; autoregressive → **directional, column-dominant** `W_QK` | Saponati et al., ICML 2025 (2502.10927) |
| Representation phases on Pythia | Hidden-state spectrum: collapse → expansion (n-gram memorisation) → anisotropic compression | 2509.23024; 2602.15997 |

**Saponati et al. already agrees with a result in this repo.** A self-attention
system is a gradient flow of an interaction energy when `QᵀK` is symmetric.
PROJECT.md §3.12 block J (2026-09-09, `tools/run/qk_symmetry_sweep.py`,
410m, 384 heads × 19 steps) found the median QK
symmetric fraction at **0.5005 → 0.5231** from step 512 to 143000, with 11 of
384 heads above 0.7. Autoregressive training does **not** move most heads
towards the symmetric case the gradient-flow theorems use. The heads that
become symmetric are content matchers. This is the most concrete known part
of the selection map for this project.

### C. Bayesian degeneracy (singular learning theory): *where* the posterior puts mass

| Item | Content | Source |
|---|---|---|
| LLC and stages | The local learning coefficient splits transformer training into stages that coincide with changes in internal structure | Hoogland et al., 2402.02364 |
| Susceptibilities | Response of posterior expectations to data perturbations = a posterior covariance (fluctuation–dissipation); decomposes over modes of the data | Primer 2605.07980 |
| **Spectroscopy on Pythia** | Tokens clustered by **weight-space susceptibility**: 510 clusters on Pythia-14M; follow-up on **Pythia-1.4B** (April 2026; page blocked, title only) | Gordon et al., 2601.12703 |
| Dead directions | Directions where the Fisher metric degenerates; per-direction KL order read at a frozen checkpoint; a LayerNorm-transformer variant at LLM scale | 2606.05957, 2607.00603, 2606.19491 (unvetted series) |

**Caveat that matters here:** SLT describes the Bayesian **posterior**, not the
SGD/Adam **trajectory**. Using it to explain what Adam selects assumes the two
agree locally (the LLC is estimated by SGLD around the trained point). That
assumption is the bridge the user is asking about; it is not a result.

### D. Mean-field training: optimal transport **in weight space**

| Item | Content | Source |
|---|---|---|
| Classic | Wide-network training = Wasserstein gradient flow of the parameter distribution | Mei, Montanari & Nguyen 2018; Chizat & Bach 2018 |
| **Coupled flow for transformers** | Two mean-field objects: token distribution `μ_t` moving through **depth** (McKean–Vlasov, the Geshkovski object) and attention-parameter distribution `ρ_s` moving through **training time** (Wasserstein gradient flow of the risk). Well-posed; global convergence for shallow single-layer attention under log-Sobolev | 2608.25055 |
| Deep and wide training | Training infinitely deep and wide transformers | Barboni, de Hoop, Furuya, Peyré, 2605.17660 |
| ICL, mean-field | Non-convex mean-field dynamics on the attention landscape; saddles avoided | Kim & Suzuki, ICML 2024 |
| Training breaks clustering | Trained FFN makes tokens leave the clustered regime near the last layers | Isobe, Inoue & Imaizumi, 2605.07772 (already in PUBLICATION_IDEAS.md) |

**2608.25055 is the closest existing formalisation of the user's idea:** OT in
activations (depth) coupled to OT in weights (training time). It is proven only
in idealised limits (infinitely many tokens, heads and layers; shallow case for
convergence).

### E. Symmetry: *which weights are the same function*

- Transformers have a large group of function-preserving weight transforms:
  neuron and head permutations, residual-stream rotation and scaling, norm
  absorption, head-internal `W_Q → W_Q A`, `W_K → W_K A^{-T}`
  (Theus et al., 2506.22712; LMC at billion scale, 2606.23607).
- So "the set of weights SGD picks" is only defined **up to this group**.
  Measurements that are invariant under it (token Gram matrices, cosine
  geometry up to rotation, attention patterns) see the function; raw weight
  coordinates do not. This is the physicist's **gauge** picture: weights carry
  gauge freedom, activation geometry is gauge-invariant. It supports C4 as a
  *choice of observables*, not as a claim about phenomenology.

### F. Convergence of representations across runs

| Item | Content | Source |
|---|---|---|
| Platonic representation hypothesis | Models with different architectures, objectives and modalities drift towards similar representational similarity structure as scale grows | Huh et al., ICML 2024; critiques 2602.14486, 2604.17960 |
| **PolyPythias** | 9 extra seeds × 5 sizes (14M–410M), ~7k checkpoints; training dynamics highly consistent across seeds, with identifiable outlier runs | van der Wal et al., ICLR 2025 (2503.09543) |

### G. Learning rules other than backprop

| Item | Content | Source |
|---|---|---|
| Predictive coding | Local, Hebbian updates converge to backprop gradients on arbitrary computation graphs, under conditions | Millidge et al. 2022; Whittington & Bogacz 2017 |
| **Prospective configuration** | The network first infers the activity learning *should* produce, then weights change to consolidate it: activity first, weights second | Song et al., Nature Neuroscience 2024 |

Prospective configuration is an existing formal learning rule in which the
activation state is primary and the weights follow, which is the user's C4
turned into an algorithm.

### H. Interpreting in weight space, and beyond linear features

| Item | Content | Source |
|---|---|---|
| APD / SPD | Decompose weights into parameter components that sum to the model; few active per input | Braun et al. 2025; Bushnaq, Braun & Sharkey 2506.20790; small transformers 2511.08854 |
| Weight-sparse transformers | Train sparse so circuits are readable in the weights | 2511.13653 |
| Feature manifolds | Irreducible multi-dimensional features (circles for days/months) | Engels et al., 2405.14860 |
| Counting manifolds | Character counts on curved low-dim manifolds; attention heads "twist" them | Gurnee et al., transformer-circuits.pub 2025 |
| Origin of manifolds | Why representation manifolds arise | 2505.18235 |
| Field-level | "Learning mechanics": solvable settings, limits, simple laws, hyperparameter theories, universal behaviours | Simon, Kunin et al., 2604.21691 |

## 3. Directions, cheapest first

None is scheduled. Order and scope are the user's. "Weights only" means CPU
and no forward passes.

| # | Question | Cost | What it could change | Blocker / first step |
|---|---|---|---|---|
| **D1** | **Hypothesis-class audit.** For each theorem this project leans on, list its conditions on `W_QK`, `W_V` (symmetry, definiteness, spectrum, β scale). Measure each per head across the 19 steps. QK symmetry is already done (block J); add `W_V`'s symmetric part and definiteness, and Saponati's column dominance | weights only, hours | Turns C1 from an assumption into a per-head table: which heads the theorems can speak about, at which step | Write the conditions list from `p10_cluster_function/lit-10.md` and `math-10.md`; reuse `tools/run/qk_symmetry_sweep.py`'s algebra |
| D2 | **Seeds (PolyPythias).** Do clustering statistics at matched step agree across 9 seeds? If so, the selection map, not the init, sets them | forward passes, 14M–410M | Whether clustering is a property of (data, optimiser, arch) or of one run | 410m's v2 prompts are held out on 410m (`core/holdout.py`); whether PolyPythias-410m counts as "410m" is the user's call. 14M/70M/160M avoid it |
| D3 | **`ΔW` rank vs clustering onset.** Stable rank of `W_t − W_0` per layer vs step; does it jump where block J's symmetry starts moving (step 1000) and where the geometry literature's phases fall? | weights only | Links a known training-dynamics law to the checkpoints this repo already reads | none |
| D4 | **Weight ESD phases** (Martin–Mahoney) per layer × step, beside 1d's clustering metrics | weights only | Whether layers where clusters sharpen are the heavy-tailed ones | none |
| D5 | **Optimiser contrast.** Same architecture and data, Adam vs Muon (or Shampoo) | depends on public checkpoints | Separates optimiser artefacts (rogue coordinates, sinks) from generic clustering | Discovery: no matched open pair located yet |
| D6 | **Two definitions of a token cluster.** Timaeus's susceptibility clusters (weight-space response) vs this repo's residual-stream clusters on the same Pythia model and tokens | SGLD; moderate GPU; `devinterp` is open source | Bears on 1d's "what a cluster is": a weight-side definition to compare against | Read 2601.12703 in full; pick a model both sides can run (14M is theirs; 70m/410m are ours) |
| D7 | **Read 2608.25055 properly.** Does its coupled `(μ_t, ρ_s)` system predict how training moves the clustering rate or the metastable time scale? | reading; `sympy` check for any closed form used | Might give the first theoretical prediction for how training changes the clustering picture this repo measures | arXiv access |
| D8 | **Optimiser artefacts as a 1d confound.** The rogue coordinates and the token-0 sink have optimiser-level explanations (He et al.; Gu et al.). Part of what 1d calls structure may be Adam-induced coordinates rather than content | reading, then a null that removes outlier coordinates (1d already does 3) | Attach to 1d as a confound, not a new thread | The user decides whether to attach |
| D9 | **SPD on a Pythia layer:** do clusters line up with active parameter components? | long | A weight-side account of a cluster | After D6 |

## 4. On "axioms"

Physics fixed its frame by naming invariances first. Candidate invariances for
this problem, stated as things to test rather than to assume:

1. **Gauge:** observables must be invariant under the weight-symmetry group (§2 E).
   Most of this repo's activation metrics already are, up to rotation.
2. **Seed invariance:** at matched (data, optimiser, architecture, step), the
   token-geometry statistics should not depend on the initialisation draw. D2 tests it.
3. **Optimiser covariance:** a change of optimiser changes the weight norm
   geometry (ℓ∞ vs spectral, §2 A). The prediction is that the effect on token
   geometry is limited to specific coordinates (outliers, sinks). D5 tests it.

If 2 holds and 3 holds in that form, the activation geometry is the invariant
object and the weights are one representative of it. That would support C4 on
evidence rather than preference. If either fails, the weights carry information
the activations do not show, and weight-space methods (§2 C, H) are needed
alongside.

## 5. What this file does not establish

- Most 2025–26 entries were read at abstract level (see top).
- Implicit-bias theorems are for separable data, homogeneous networks or full
  batch. None covers Pythia's regime.
- SLT's objects are posterior quantities. Their link to the Adam trajectory is
  the open bridge (§2 C).
- Nothing here has been checked against this repo's token-clustering data
  except the QK-symmetry cross-reference (§2 B), which cites an existing result.

## 6. Triage

All of §3 is a **discovery** under CLAUDE.md's tangent triage, parked here and
not followed. D8 is the exception: it may be a **confound** on Phase 1d. The
decisions it could change are 1d's cluster definition (D6, D8) and whether
Phase 10 rows may cite identity-weight theorems at all (D1).

## References (links)

- Xie & Li, *Implicit Bias of AdamW: ℓ∞-Norm Constrained Optimization*, [ICML 2024](https://proceedings.mlr.press/v235/xie24e.html)
- *The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks*, [2602.16340](https://arxiv.org/abs/2602.16340); *Muon Optimizes Under Spectral Norm Constraints*, [2506.15054](https://arxiv.org/abs/2506.15054); *Implicit Bias of Spectral Descent and Muon*, [2502.04664](https://arxiv.org/abs/2502.04664)
- Li, Wang & Arora, *What Happens after SGD Reaches Zero Loss?*, [2110.06914](https://arxiv.org/abs/2110.06914)
- He et al., *Understanding and Minimising Outlier Features in Transformer Training*, [2405.19279](https://arxiv.org/abs/2405.19279); Puccetti et al., *Outlier Dimensions … Are Driven by Frequency*, [2205.11380](https://arxiv.org/abs/2205.11380)
- Gu et al., *When Attention Sink Emerges in Language Models*, [2410.10781](https://arxiv.org/abs/2410.10781)
- Cohen et al., *Understanding Optimization in Deep Learning with Central Flows*, [2410.24206](https://arxiv.org/abs/2410.24206); *A Rod Flow Model for Adam at the Edge of Stability*, [2605.06821](https://arxiv.org/abs/2605.06821); *The Multiple Timescales of Gradient Descent on the Edge of Stability*, [2609.01034](https://arxiv.org/abs/2609.01034)
- Gur-Ari, Roberts & Dyer, *Gradient Descent Happens in a Tiny Subspace*, [1812.04754](https://arxiv.org/abs/1812.04754)
- Boix-Adserà et al., *Transformers Learn Through Gradual Rank Increase*, [2306.07042](https://arxiv.org/abs/2306.07042)
- Martin & Mahoney, *Implicit Self-Regularization in Deep Neural Networks*, [1810.01075](https://arxiv.org/abs/1810.01075)
- Saponati et al., *The Underlying Structures of Self-Attention: Symmetry, Directionality, and Emergent Dynamics in Transformer Training*, [2502.10927](https://arxiv.org/abs/2502.10927)
- *Tracing the Representation Geometry of Language Models from Pretraining to Post-training*, [2509.23024](https://arxiv.org/abs/2509.23024); *The Geometric Anatomy of Capability Acquisition in Transformers*, [2602.15997](https://arxiv.org/abs/2602.15997)
- Hoogland et al., *Loss Landscape Degeneracy and Stagewise Development in Transformers*, [2402.02364](https://arxiv.org/abs/2402.02364)
- *Susceptibilities and Patterning: A Primer on Linear Response in Bayesian Learning*, [2605.07980](https://arxiv.org/abs/2605.07980)
- Gordon et al., *Towards Spectroscopy: Susceptibility Clusters in Language Models*, [2601.12703](https://arxiv.org/abs/2601.12703); Timaeus, *Finding Interpretable Structure in Pythia-1.4B* (2026-04-21, timaeus.co/research)
- *Dead Directions: Geometric Singular Learning*, [2606.05957](https://arxiv.org/abs/2606.05957); follow-ups [2607.00603](https://arxiv.org/abs/2607.00603), [2606.21158](https://arxiv.org/abs/2606.21158), 2606.19491
- Herty, Liu, *A Mean-Field Theory of Transformers: Well-Posedness of the Coupled Data–Parameter Dynamics and Global Convergence of Training*, [2608.25055](https://arxiv.org/abs/2608.25055) (authors from the 2026-10-01 reading list, `docs/readings/meanfield_reading_list_2026-10-01.md`)
- Barboni, de Hoop, Furuya, Peyré, *Training Infinitely Deep and Wide Transformers*, [2605.17660](https://arxiv.org/abs/2605.17660)
- Kim & Suzuki, *Transformers Learn Nonlinear Features In Context: Nonconvex Mean-field Dynamics on the Attention Landscape*, [ICML 2024](https://proceedings.mlr.press/v235/kim24af.html)
- Isobe, Inoue & Imaizumi, *Training-Induced Escape from Token Clustering*, [2605.07772](https://arxiv.org/abs/2605.07772)
- Noci et al., *Signal Propagation in Transformers: … Rank Collapse*, [2206.03126](https://arxiv.org/abs/2206.03126); *The Shaped Transformer*, [2306.17759](https://arxiv.org/abs/2306.17759) — the random-init end: token covariance follows an SDE in depth
- Theus et al., *Generalized Linear Mode Connectivity for Transformers*, [2506.22712](https://arxiv.org/abs/2506.22712); *Scaling Linear Mode Connectivity … to Billion Parameter Pretrained Transformers*, [2606.23607](https://arxiv.org/abs/2606.23607)
- Huh et al., *The Platonic Representation Hypothesis*, [2405.07987](https://arxiv.org/abs/2405.07987); *Revisiting the PRH: An Aristotelian View*, [2602.14486](https://arxiv.org/abs/2602.14486); *The Umwelt Representation Hypothesis*, [2604.17960](https://arxiv.org/abs/2604.17960)
- van der Wal et al., *PolyPythias*, [2503.09543](https://arxiv.org/abs/2503.09543)
- Millidge, Tschantz & Buckley, *Predictive Coding Approximates Backprop along Arbitrary Computation Graphs*, [2006.04182](https://arxiv.org/abs/2006.04182)
- Song et al., *Inferring Neural Activity Before Plasticity as a Foundation for Learning Beyond Backpropagation*, [Nature Neuroscience 2024](https://www.nature.com/articles/s41593-023-01514-1)
- Bushnaq, Braun & Sharkey, *Stochastic Parameter Decomposition*, [2506.20790](https://arxiv.org/abs/2506.20790); *Decomposition of Small Transformer Models*, [2511.08854](https://arxiv.org/abs/2511.08854); *Weight-sparse Transformers Have Interpretable Circuits*, [2511.13653](https://arxiv.org/abs/2511.13653)
- Engels et al., *Not All Language Model Features Are Linear*, [2405.14860](https://arxiv.org/abs/2405.14860); *When Models Manipulate Manifolds*, [transformer-circuits.pub 2025](https://transformer-circuits.pub/2025/linebreaks/index.html); *The Origins of Representation Manifolds in LLMs*, [2505.18235](https://arxiv.org/abs/2505.18235)
- Simon, Kunin et al., *There Will Be a Scientific Theory of Deep Learning*, [2604.21691](https://arxiv.org/abs/2604.21691)
