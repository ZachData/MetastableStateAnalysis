# Phase 2c — Trajectory-Side Dynamical Systems Analysis

**Status:** Not started. Sister phase to Phase 2i.

---

## Placement

Phase 2i analyzed the operator: Schur decomposition of $V$, causal rescaling by $e^{-tS}$ and $e^{-tA}$. It established that $A$ is *dynamically neutral for clustering* (`elim_rotation = 0.0` across all 35 model × prompt combinations). It did not test whether $A$ is used by any other computation.

Phase 2c analyzes the trajectory. It imports neuroscience methods developed for population recordings — jPCA, tangling, CIS, slow points, context-dependent subspace selection — and applies them to layer-by-layer activation trajectories. The aim is to characterize the observed flow field, independent of the operator-spectrum analysis, and check whether transformer layer dynamics carry the autonomous-dynamical-system signatures Churchland's lab found in motor cortex.

This is a separate directory (`p2c_trajectory/`) rather than an expansion of `p2b_imaginary/` because:

1. **Operator-side vs data-side.** p2b decomposes weights. p2c fits dynamical models to activations. Different objects, different machinery.
2. **p2b is closed.** Phase 2i was a complete experiment with a unanimous null. Reopening it for new metrics blurs that conclusion.
3. **Imports, not refactors.** The Schur projectors ($U_A$, $U_S$) from `p2b_imaginary/rotational_schur.py` and the displacement helpers from `phase2/trajectory.py` are read-only inputs to p2c. No existing file needs modification.
4. **Falsifiable separation.** If p2c finds rotational signatures in trajectories that p2b's operator analysis does not predict, that misalignment is itself the result. Mixing them obscures it.

---

## Core question

The operator $V$ has 84–97% of its spectral energy in 2×2 rotation blocks, but the antisymmetric component does not drive clustering. This leaves three possibilities:

1. **Trajectories use $V$'s rotational capacity for non-clustering computation.** Observed $\Delta x$ projects onto $U_A$ planes; jPCA fits recover the same planes; rotational signatures (low tangling, oscillatory CIS) are present.
2. **Trajectories use rotation, but in planes orthogonal to $U_A$.** jPCA fits give clean rotation, but the planes are unrelated to the operator's antisymmetric structure. Implies the rotational dynamics emerge from softmax + FFN composition rather than from $V$ itself.
3. **No rotational structure in trajectories.** $A$'s 97% spectral energy is a representational accident of high-dimensional non-normal matrices, not a computation. Trajectories are predominantly translational toward the cluster attractor, with the rotational operator capacity unused.

These have different implications for in-context learning, induction, and the Phase 6 division-of-labor hypothesis.

---

## Background: methods imported

Each method below has a section: (i) the original neuroscience finding, (ii) what it would tell us about transformers, (iii) the falsifiable prediction, (iv) the failure mode that would invalidate the import.

### C1 — jPCA fit to layer-to-layer Δx

**Original (Churchland et al. 2012):** Reduce population activity $X \in \mathbb{R}^{n \times T}$ to its top PCs, then fit $\dot X = M X$ with $M$ constrained skew-symmetric. The eigenvectors of $M$ define rotation planes; the eigenvalues are pure imaginary; the $R^2$ ratio of the constrained vs unconstrained fit measures how rotational the dynamics actually are. Motor cortex during reach: ~84% ratio.

**Transformer port:** For each prompt, $X \in \mathbb{R}^{(L-1) \times d}$ is the per-layer mean activation (or per-token, stacked) and $\dot X$ is approximated by $\Delta X^{(L)} = X^{(L+1)} - X^{(L)}$. Fit jPCA in the top-6 PC subspace, extract rotation planes and frequencies.

**Comparison to Phase 2i:** The data-driven rotation planes from jPCA can be compared directly to the operator-side $U_A$ planes via principal-angle analysis. Three diagnostic outcomes:

| jPCA planes vs $U_A$ | Interpretation |
|---|---|
| Coincide (small principal angles) | $V$'s rotational structure is exercised. Phase 2i's null was about clustering specifically. |
| Orthogonal | Rotational dynamics are real but emerge from softmax/FFN composition, not from $V$. |
| Coincide only at specific layers or token populations | Localized rotational computation; identify where. |

**Falsifiable prediction P2c-J1:** Constrained-vs-unconstrained $R^2$ ratio for layer-to-layer $\Delta X$ jPCA fit exceeds 0.5 in at least one model. Failure: ratio < 0.3 universally → trajectories are not rotational regardless of operator structure.

**Falsifiable prediction P2c-J2:** Top jPCA planes have mean principal angle < 30° to $U_A$ planes from $V$'s Schur decomposition. Failure: angles uniform on $[0, 90°]$ → operator and trajectory rotations are unrelated.

**Caveat:** jPCA requires removing the cross-condition mean. Its successor HDR (Lara, Cunningham, Churchland 2018) sidesteps this and is preferred when interpretability is critical. Run jPCA first; rerun with HDR if jPCA flags weak rotation that might be inflated by the centering.

### C2 — Trajectory tangling (Russo et al. 2018)

**Original:** $Q(t) = \max_{t'} \frac{\|x(t) - x(t')\|^2}{\|\dot x(t) - \dot x(t')\|^2 + \epsilon}$. Two states close in position but with different derivatives produce high $Q$. Smooth autonomous dynamics keep $Q$ low; input-driven systems (somatosensory cortex, EMG, kinematics) have high $Q$. Motor cortex during cycling: low $Q$, consistent with autonomous pattern generation.

**Transformer port:** Compute $Q$ on per-token trajectories across layers, with $x(t) \to x^{(L)}_i$ and $\dot x(t) \to \Delta x^{(L)}_i$. Compute separately on:
- Full activation trajectories
- $S$-channel projection only (real subspace)
- $A$-channel projection only (imaginary subspace)

**Prediction P2c-T1:** $A$-channel trajectories have lower tangling than $S$-channel trajectories. The real subspace contracts toward a small number of attractors — distinct prompts will visit similar positions with different forward derivatives, yielding high $Q$. The imaginary subspace, if it carries autonomous-style sequence-generation, should be smoother.

**Prediction P2c-T2:** Induction-pattern prompts (`A B ... A → B`) have lower full-trajectory tangling than non-induction matched-length prompts. Failure of P2c-T2 with success of P2c-T1 is informative: it would suggest the imaginary channel is autonomous-like, but induction is implemented elsewhere.

**Caveat — regime mismatch:** Tangling was developed for continuous-time autonomous recurrent systems. Transformers are layer-discrete and non-autonomous (attention re-weights at every layer based on the full token state). $Q$ is still computable, but its theoretical grounding does not transfer. Treat as a descriptive statistic, not a test of autonomy. The interesting comparison is *relative* $Q$ across channels and prompt types, not the absolute magnitude.

### C3 — Condition-invariant signal (Kaufman et al. 2016)

**Original:** Decompose population activity across conditions (reaches in different directions) into condition-invariant components (CIS — common across all reaches, dominates variance, undergoes a stereotyped change at movement onset) and condition-specific components (CS — task content). CIS occupies > 50% of variance in motor cortex around movement onset.

**Transformer port:** Across $K$ prompts of matched length, decompose per-layer activations into prompt-invariant and prompt-specific components via demixed PCA (dPCA) or a simpler variance-decomposition: at each layer, compute the cross-prompt mean trajectory $\bar X^{(L)} = \frac{1}{K} \sum_k X_k^{(L)}$ and the residual $\tilde X_k^{(L)} = X_k^{(L)} - \bar X^{(L)}$.

**Prediction P2c-K1:** The prompt-invariant component projects predominantly onto the $A$ subspace; the prompt-specific component projects predominantly onto the $S$ subspace. This is a sharper, variance-decomposed version of the Phase 6 division-of-labor claim and uses the same projectors.

**Prediction P2c-K2:** The prompt-invariant component shows a stereotyped change at the layers where Phase 1 detected merge events. CIS in motor cortex is associated with a transition between dynamical regimes (preparatory → movement-generating); merge events are transitions between metastable states. If both reflect a regime change, a stereotyped condition-invariant change at merge layers is the analog.

**Caveat — what counts as a "condition"?** In motor cortex, conditions are reach directions on a controlled task. Prompts vary along too many axes to be true conditions. The cleanest version of this analysis uses a single task with controlled variation (e.g., `A B C D E ... A → ?` for varying `A`-`E` token sets) rather than the heterogeneous prompt set Phase 1 used. C3 should be run on a purpose-built prompt grid, not the existing five prompts.

### C4 — Slow points around metastable states (Sussillo & Barak 2013)

**Original:** For an RNN with state update $h_{t+1} = F(h_t)$, find points where $\|F(h) - h\|^2$ is locally minimized but nonzero. These are "slow points" — ghosts of fixed points that govern dynamics over relevant timescales. Linearize $F$ at each slow point; the local Jacobian's spectrum reveals the local computation.

**Transformer port:** For each layer's per-layer map $F_L: x \mapsto \text{LayerNorm}(x + \text{Attn}_L(x) + \text{FFN}_L(x))$, identify metastable states from Phase 1 (cluster centroids in plateau windows) and treat them as candidate slow points. Compute the local Jacobian $J_L = \partial F_L / \partial x$ at the centroid. Decompose into symmetric and antisymmetric parts; compare the local $S/A$ ratio to $V$'s global $S/A$ ratio.

**Prediction P2c-S1:** Local Jacobians at metastable states are *more* symmetric than $V$ (the global operator). The interpretation: at a stable plateau, the local flow is contractive (real-eigenvalue dominated); rotation lives elsewhere in the trajectory.

**Prediction P2c-S2:** Local Jacobians at merge layers (between plateaus) are *less* symmetric than at plateau interiors. If true, this localizes rotational computation to transition periods, paralleling the motor-cortex finding that rotational dynamics emerge after preparation and during the movement-generating epoch.

**Why this matters for Phase 2i's null:** Phase 2i's rescaling uses the global $V$. If local Jacobians differ substantially from $V$, the global rescaling's null result is silent on what the local linearization would have shown. This is the most important methodological correction available — global S/A and local S/A can in principle disagree, and only the local picture controls the trajectory near a metastable state.

**Caveat:** Slow-point search in the original paper uses gradient descent on $q(h) = \|F(h)-h\|^2$. For transformers, the natural targets are already known (Phase 1 centroids), so the search is unnecessary; what we need is the Jacobian computation at known points. This is a JVP/VJP loop in the model's forward pass, not a separate optimization.

### C5 — Context-dependent subspace selection (Mante et al. 2013) — the ICL test

**Original:** Macaque PFC during a context-dependent decision task: same network, same noisy sensory inputs, different context cues → different computations performed via projection of input onto context-selected subspaces. Untargeted dimensions still receive activity, they are just not read out. This is the cleanest neuroscientific analog to in-context learning.

**Transformer port — direct ICL test:** For a fixed task (e.g., word-pair reversal, simple function inference) construct a graded prompt series:

- Zero-shot: task description only
- 1-shot: one example
- $k$-shot: $k$ examples, $k \in \{0, 1, 2, 4, 8, 16\}$

For each prompt, compute the layer-wise activation projections onto $U_A$ and $U_S$ separately. Sum squared projections per layer per channel.

**Prediction P2c-M1:** $A$-channel projection magnitude on the answer-position token grows monotonically with $k$. $S$-channel projection magnitude does not (or grows less). This would be positive evidence that ICL-relevant context information is routed through the rotational subspace.

**Prediction P2c-M2:** The $A$-channel direction along which projection scales with $k$ is *task-specific* — reversal tasks vs translation tasks vs arithmetic tasks select different directions within $U_A$. Failure would mean either the rotational subspace is task-general (a generic ICL machinery) or unused.

**Prediction P2c-M3 (Mante's specific finding):** Across two tasks differing only in instruction/context (e.g., "answer in English" vs "answer in French", same input), the $S$-channel trajectory is similar; the $A$-channel trajectory diverges in a context-selected subspace. This is the closest possible analog to the Mante et al. result.

**Caveat:** Mante's task was tightly controlled (two integration tasks, same stimuli, only context flipped). Most ICL benchmarks confound task content with context length. Construct prompt pairs where only the instruction differs.

---

## Module structure

```
p2c_trajectory/
  README_phase2c.md             — this file
  jpca_fit.py                   — C1: jPCA on layer Δx, plane extraction, frequency, R² ratio
  jpca_alignment.py             — C1: principal angles between jPCA planes and U_A
  hdr_fit.py                    — C1 follow-up: HDR if jPCA results are borderline
  tangling.py                   — C2: Q metric on full / S-only / A-only projections
  cis_decompose.py              — C3: dPCA-style invariant/specific split, projection onto S/A
  local_jacobian.py             — C4: per-centroid Jacobian via JVP, S/A spectral decomposition
  slow_point_compare.py         — C4: local vs global S/A ratio, layer-wise comparison
  icl_subspace_scaling.py       — C5: A/S projection magnitude vs k-shot
  context_selection.py          — C5: subspace divergence between matched-context prompt pairs
  prompt_grids/                 — purpose-built prompt sets for C3 and C5
    matched_length.json
    icl_kshot.json
    context_pairs.json
  run_2c.py                     — CLI entry point
  report_2c.py                  — flat text report
```

---

## Dependencies

**Read-only imports from existing phases:**

- `p2b_imaginary/rotational_schur.py` — `extract_schur_blocks`, `build_rotation_plane_projectors` for $U_A$ planes and $U_S$ vectors
- `p2b_imaginary/subspace_build.py` — `build_global_projectors` for $P_A$, $P_S$ residual-stream projectors
- `phase2/trajectory.py` — displacement helpers for $\Delta x$ extraction
- `phase2/weights.py` — per-layer / shared $V_\text{eff}$ matrices (only needed for jPCA-vs-$U_A$ alignment in C1)
- Phase 1 outputs: `activations.npz`, `clusters.npz`, `metrics.json` (centroid trajectories, plateau/merge layer indices)

**New external dependencies:**

- jPCA reference implementation (Churchland lab, Columbia) — port to Python or reimplement
- dPCA (Kobak et al. 2016) — pip package available
- For C4 Jacobian: `torch.func.jacrev` or `torch.autograd.functional.jacobian`; the Phase 1 forward pass needs to be re-run with autograd enabled at the centroid layer

**No modifications to p2b, phase1, phase2, or any earlier phase.**

---

## ALBERT-specific notes

ALBERT shares weights, so the Schur decomposition gives a single $U_A$ per model, and "layer" in p2c means iteration index. For C4, the same forward function $F$ is applied at every iteration — local Jacobians at different metastable states differ only because the input state differs, not because the operator differs. This is actually the cleanest test bed for the local-vs-global comparison: any divergence between local Jacobian S/A and global $V$ S/A is purely an effect of where in state space the linearization is taken.

For per-layer models (GPT-2, BERT), each layer has its own $V_L$ and its own $F_L$. C1 jPCA can be fit per-layer-window or globally; both are informative. C4 requires per-layer Jacobians.

---

## Falsification table

| ID | Prediction | Failure |
|---|---|---|
| P2c-J1 | jPCA $R^2$ ratio > 0.5 in at least one model | < 0.3 universally |
| P2c-J2 | Top jPCA planes within 30° of $U_A$ | Uniform on $[0, 90°]$ |
| P2c-T1 | $A$-channel tangling < $S$-channel tangling | $A$-channel $\geq$ $S$-channel |
| P2c-T2 | Induction prompts have lower tangling than matched non-induction | No difference, or higher |
| P2c-K1 | Prompt-invariant variance lives in $A$; specific in $S$ | Reversed, or both in same channel |
| P2c-K2 | Stereotyped invariant change at Phase 1 merge layers | No layer-locked change |
| P2c-S1 | Plateau Jacobians more symmetric than $V$ | Equal or less symmetric |
| P2c-S2 | Merge Jacobians less symmetric than plateau Jacobians | No difference |
| P2c-M1 | $A$-channel magnitude scales with $k$, $S$ does not | Both scale, or $S$ scales more |
| P2c-M2 | Task-specific direction within $U_A$ scales with $k$ | Same direction for all tasks |
| P2c-M3 | Context-paired prompts diverge in $A$, agree in $S$ | Reversed, or no divergence |

---

## Order of operations

The five tracks are not equally costly or equally informative. Recommended order:

1. **C4 first.** Local Jacobians settle whether Phase 2i's global rescaling missed a localized effect. Cheapest of the five (no new prompts; reuses Phase 1 centroids), and the most likely to overturn or refine an existing conclusion.
2. **C1 second.** jPCA on existing Phase 1 trajectories. Tests whether the operator's rotational structure is exercised by the data without needing new prompts.
3. **C5 third.** Requires new prompt construction but is the most direct ICL test in the plan. The result is binary-ish (does $A$-channel scale with $k$ or not).
4. **C2 fourth.** Tangling is a descriptive statistic; useful framing, low independent inferential weight.
5. **C3 last.** Requires the most careful prompt construction (matched-length, controlled-variation grid). Most informative when C1 and C5 already point somewhere specific.

---

## Regime-mismatch caveat (overall)

All five methods were developed for continuous-time autonomous recurrent dynamics observed via population spike-rate recordings. Transformers are layer-discrete, non-autonomous (attention re-weights every layer based on the full token state), have a known operator (we have the weights — neuroscience does not), and the "trajectory" is across depth, not time, with no physical interpretation of $dt$.

Numerical computability is not the issue. Theoretical grounding is. Each prediction in this README should be read as: *if the transformer dynamics share the relevant structural features with motor-cortex dynamics, this is what we would observe.* A confirming result is suggestive of the analogy, not a proof. A disconfirming result is more informative than a confirming one — it rules out specific mechanisms the analogy predicts, regardless of whether the analogy holds in general.

The failure mode to actively avoid is over-interpreting confirmations. The success mode is treating each method as a directed probe with a precommitted falsifier.
