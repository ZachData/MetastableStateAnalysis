# Phase 6 — Real/Imaginary Subspace Decomposition of Transformer Computation

**Status:** Not started.

---

## What Phase 2i Left Open

Phase 2i established that the antisymmetric/imaginary component $A = (V-V^\top)/2$ is
dynamically neutral: removing it from the OV circuit leaves energy violations unchanged.
The signed/real component $S = (V+V^\top)/2$ carries 100% of violation causality.

This answers one question: does rotation drive the clustering dynamics? No.

It does not answer: **what does the rotational subspace do?** A subspace orthogonal to
the clustering dynamics is not inert — it is free to carry other computation without
interfering with the attractor structure.

Phase 6 tests a specific hypothesis about this division of labor:

> **Real subspace ($S$, 1×1 Schur blocks):** semantic similarity and metastable state
> organization. Tokens in the same cluster are degenerate in $S$'s eigenspace — they
> follow identical trajectories under the Geshkovski dynamics and converge to the same
> attractor.
>
> **Imaginary subspace ($A$, 2×2 Schur blocks):** relational computation. Operations
> that require comparing tokens across positions, encoding positional offsets, or
> implementing algorithmic circuits that go beyond static inner-product similarity.

---

## Part A — What Does the Imaginary Subspace Compute?

### A.1 The Taxonomy of Rotational Operations

Induction is the clearest candidate but not uniquely so. The class of operations that
*require* the antisymmetric component is precisely the class of operations that cannot be
reduced to pairwise inner products $\langle x_i, x_j \rangle$ on current positions. The
Geshkovski dynamics are inner-product dynamics. Everything the transformer computes that
escapes that framework is a candidate for the rotational subspace.

**Operations classifiable as potentially rotational:**

| Operation | Why rotation could be involved |
|---|---|
| **Induction** | Must match $x_i \approx x_{j-1}$ and attend to $x_j$. The "shift by one" is a position-relative offset. |
| **Previous-token heads** | Attend to $i-1$ regardless of content. Pure positional offset. If position is encoded as angle, antisymmetric QK is required. |
| **Copy / name-mover heads** | Address by position, not by content similarity. Requires a directional "pointer" into the past. |
| **Anti-similarity heads** | Attend to tokens *unlike* the query (e.g. function words attending to content words). Antisymmetric QK repels similar and attracts dissimilar by construction. |
| **Coreference / name binding** | Associate pronoun with antecedent across distance. Relational, positional, not a static inner product. |

The unifying principle: **self-similarity operations use real structure; relational
operations use imaginary structure.** The Geshkovski framework fully describes the
self-similarity dynamics. The imaginary subspace carries everything beyond it.

### A.2 Head Classification: Self-Similarity vs Relational

For each head $h$, define:

**Content-coupling score:**
$$\text{CC}(h) = \text{Spearman}\bigl(A^{(h)}[i,j],\ \langle q^{(h)}_i, k^{(h)}_j \rangle\bigr)_{(i,j)}$$

High CC: head attends based on query-key inner product — content-based, self-similarity.

**Positional-coupling score:**
$$\text{PC}(h) = \text{Spearman}\bigl(A^{(h)}[i,j],\ f(i-j)\bigr)_{(i,j)}$$

for a positional function $f$ (indicator on $j = i-1$ for previous-token, Gaussian for
local, flat for non-local).

This produces a 2D (CC, PC) space per head. Predicted locations:
- Semantic heads: high CC, low PC → high $S$-fraction in OV
- Previous-token / positional heads: low CC, high PC → high $A$-fraction in OV
- Induction heads: moderate CC (content match) + moderate PC (shift by one) → elevated $A$-fraction
- Anti-similarity heads: negative CC, low PC → elevated $A$-fraction (antisymmetric by definition)

**Falsifiable prediction P6-A2:** Rotational energy fraction $f_\text{rot}(h)$ (from OV
Schur decomposition) decreases monotonically with CC and increases with PC. Cross-head
Spearman $\rho(f_\text{rot}, -\text{CC}) > 0.4$ across all heads at the model's most
expressive layers.

### A.3 Anti-Similarity Heads as a Second Computation Class

A head with high imaginary OV fraction but low induction score is not doing induction. If
it has negative CC, it is an anti-similarity head. Test: for candidate heads, compute
correlation between $A^{(h)}[i,j]$ and $-\langle x_i, x_j \rangle$. Significantly
negative correlation at non-trivial attention weights = anti-similarity head.

Anti-similarity heads may drive merge events by bridging dissimilar clusters. If they fire
at merge layers specifically, they are a candidate mechanism for cluster dissolution,
complementary to the V-repulsive mechanism Phase 2 identified but acting through the
*attention routing* channel (Phase 2's Hypothesis 3, which was tested but not fully
resolved for all models).

### A.4 QK Antisymmetry Analysis (from original Phase 6 Block 3)

For induction specifically: decompose $W_Q^\top W_K = S_{QK} + A_{QK}$ and partition the
attention logit for induction pairs $(i,j) \in \mathcal{P}$:

$$\langle q_i, k_j \rangle = x_i^\top S_{QK} x_j + x_i^\top A_{QK} x_j$$

Measure $A_{QK}$ fraction for induction pairs vs same-content non-induction pairs
($\text{token}[i] = \text{token}[j]$, no offset). The positional shift should appear as
elevated $A_{QK}$ in induction pairs but not same-content pairs.

---

## Part B — The Eigenspace Degeneracy Argument for the Real Subspace

### B.1 The Core Claim

A metastable cluster in the Geshkovski framework is a set of tokens that share an
eigenspace basin of $S$. Tokens whose projections onto $S$'s dominant attractive
eigenvectors are nearly identical are dynamically equivalent — they are driven toward the
same attractor at the same rate. This is **eigenspace degeneracy**.

This is the direct, SAE-free, theory-grounded definition of what a metastable cluster is.
It is testable without any learned features: does HDBSCAN cluster membership align with
$S$'s eigenspectrum structure?

### B.2 Eigenspace Degeneracy Test

Let $U_+ \in \mathbb{R}^{d \times k}$ be the top-$k$ attractive eigenvectors of $S$.
For each token $i$ at layer $L$, compute $z_i^{(L)} = U_+^\top x_i^{(L)} \in \mathbb{R}^k$.

- **Within-cluster variance:** $\sigma_W^2 = \text{mean}_C \frac{1}{|C|} \sum_{i \in C} \|z_i - \bar{z}_C\|^2$
- **Between-cluster variance:** $\sigma_B^2 = \frac{1}{K} \sum_C \|\bar{z}_C - \bar{z}\|^2$
- **Degeneracy ratio:** $R = \sigma_B^2 / \sigma_W^2$

**Prediction P6-R1:** $R \geq 5$ at plateau layers, drops at merge events, near 1 for
random projections and for $A$'s rotational planes.

Sweep $k \in \{1, 2, 4, 8, 16, 32\}$ and plot $R(k)$. If the first few eigenvectors
achieve high $R$, $S$'s dominant structure organizes the clusters with low-rank geometry.

### B.3 LDA Alignment with S Eigenvectors

At each plateau layer, compute the LDA direction $w_\text{LDA}$ separating the two
clusters with the highest subsequent merge probability.

Compute alignment of $w_\text{LDA}$ against:
- $U_-$ (repulsive subspace of $S$): should be highest (the separating direction = the direction tokens are pushed apart before merging)
- $U_+$ (attractive subspace): lower (attractive directions are shared by both clusters — they're already converged on those)
- $U_A$ (imaginary subspace): should be near null-distribution level

**Prediction P6-R2:** $\text{align}(w_\text{LDA}, U_-) > \text{align}(w_\text{LDA}, U_A)$.
The cluster-separating direction is in $S$'s repulsive subspace, not the imaginary subspace.

This also tests a specific Phase 2 prediction more cleanly than the energy violation
analysis: the repulsive subspace should be the *geometric* source of cluster separation,
not just the energetic source.

### B.4 Centroid Velocity Decomposition

At each layer $L \to L+1$, decompose cluster centroid displacement:

$$\Delta \bar{x}_C^{(L)} = \Pi_S \Delta \bar{x}_C^{(L)} + \Pi_A \Delta \bar{x}_C^{(L)}$$

**Predictions P6-R3:**
- Plateau layers: $\|\Pi_S \Delta \bar{x}_C\|$ is small (centroid settled near attractor in real subspace); $\|\Pi_A \Delta \bar{x}_C\|$ can be nonzero (rotation, harmless per Phase 2i)
- Merge layers: $\|\Pi_S \Delta \bar{x}_C\|$ spikes in the merge direction
- Ratio $r_S(L) = \|\Pi_S \Delta \bar{x}_C^{(L)}\| / \|\Delta \bar{x}_C^{(L)}\|$ is elevated at violation layers

This is a direct measurement of which subspace drives centroid motion at each layer type,
without any learned features.

### B.5 Local Contraction Analysis

During a plateau, the cluster is near an attractor of $S$ and the local dynamics should
be contracting in the real subspace. For each cluster $C$ at plateau layer $L$, fit:

$$x_i^{(L+1)} \approx W_C x_i^{(L)},\quad W_C = \text{argmin} \sum_{i \in C} \|W x_i^{(L)} - x_i^{(L+1)}\|^2$$

Decompose $W_C = W_C^S + W_C^A$.

**Predictions P6-R5:**
- $W_C^S$ has all eigenvalues inside the unit disk during plateau (local contraction in real subspace)
- $W_C^S$ has at least one eigenvalue near or outside the unit disk at merge events (attractor destabilizing)
- $W_C^A$ has eigenvalues near the unit circle throughout (rotation, norm-preserving, neither contracting nor expanding)

This connects to the known Phase 1 limitation: ALBERT-xlarge's spectral radius of $V >
1$ predicts collapse. The per-cluster local map gives a layer-resolved, cluster-specific
version — and importantly, it can separate the real (destabilizing) from the imaginary
(neutral) component.

---

## Part C — Residual Stream Subspace Channels

### C.1 Per-Head Write Subspace Alignment

The mechanistic interpretability picture (Elhage et al.) treats each head as writing into
a specific subspace of the residual stream. The real/imaginary hypothesis predicts that
semantic heads write into the real channel and relational heads write into the imaginary
channel.

For each head $h$, let $e_1, \ldots, e_r$ be the top-$r$ left singular vectors of
$W_O^{(h)}$ (the head's dominant write directions). Compute:

$$\text{align}_\text{rot}(h) = \frac{1}{r} \sum_{k=1}^r \left\| \Pi_A e_k \right\|^2, \quad \text{align}_\text{real}(h) = \frac{1}{r} \sum_{k=1}^r \left\| \Pi_S e_k \right\|^2$$

**Prediction P6-C1:** $\text{align}_\text{rot}$ and CC (content-coupling score from A.2)
are negatively correlated across heads. Heads classified as relational by attention
patterns write into the imaginary channel; heads classified as self-similarity write into
the real channel.

### C.2 Write Subspace Orthogonality

If the real/imaginary channel separation is real, the write subspaces of real-channel
heads should be approximately orthogonal to those of imaginary-channel heads.

Partition heads into real-channel ($\text{align}_\text{rot} < 0.4$) and imaginary-channel
($\text{align}_\text{rot} > 0.6$). Compute principal angles between:

$$\text{span}\bigl(\bigcup_{h \in \text{real-ch}} W_O^{(h)}\bigr) \quad \text{vs} \quad \text{span}\bigl(\bigcup_{h \in \text{imag-ch}} W_O^{(h)}\bigr)$$

Small principal angles = channels overlap = weak hypothesis. Large principal angles =
channels occupy orthogonal residual stream subspaces = strong channel separation.

### C.3 Double Dissociation Causal Test

The strongest test in the phase. Two surgical interventions on the residual stream during
inference.

**Intervention 1 — zero the imaginary channel:**
Before adding the attention output to the residual stream at each layer:
$$h_\text{attn} \leftarrow h_\text{attn} - \Pi_A h_\text{attn}$$

Measure: induction score → **should drop**; cluster structure (HDBSCAN) → **should be
preserved**; $E_\beta$ violation profile → **should be unchanged**.

**Intervention 2 — zero the real channel:**
$$h_\text{attn} \leftarrow h_\text{attn} - \Pi_S h_\text{attn}$$

Measure: induction score → **should be preserved**; cluster structure → **should be
disrupted**; $E_\beta$ violations → **should be eliminated**.

**Intervention 3 — control (random subspace of matching dimension):**
Both induction and clusters should degrade somewhat — the structured interventions should
produce qualitatively different, dissociated patterns that this control cannot replicate.

The double dissociation (both arms) is the single most falsifiable prediction in the
phase. Either arm failing is a strong falsification of the functional separation
hypothesis.

**Note on relationship to Phase 2i:** Phase 2i's rescaled frames removed the operator
from the dynamics but did not intervene on activations during inference. This test is
more direct: it surgically removes one channel from what each head writes into the
residual stream, while leaving the softmax routing intact.

---

## Part D — Metastable States Without SAEs

### D.1 Why SAEs Are Insufficient Here

Phase 3 found crosscoder decoder directions geometrically random w.r.t. $V$'s eigensubspaces
(SNR 0.18×). Phase 4 tests whether activation patterns (not directions) track cluster
membership. Both use the SAE as an intermediary tool whose properties are imperfectly
understood.

The fundamental problem: sparsity pressure allocates dictionary capacity to frequent,
independent features (syntax, frequency, position). Metastable cluster membership is a
structural property that may not decompose into independent sparse features. The SAE will
not find it — not because the structure is absent, but because sparse coding is the wrong
prior for this kind of structure.

The direct geometric tests in Parts B and C test the structure itself without any
representational prior. They require only HDBSCAN labels (Phase 1), $S/A$ projectors
(Phase 2), and centroid trajectories (Phase 1).

### D.2 Direct Geometric Test Suite

Five tests, each independently falsifiable, none requiring SAEs:

**D.2.1** Eigenspace degeneracy (B.2) — within/between cluster variance in $S$ eigenspace.

**D.2.2** LDA–$S$ alignment (B.3) — discriminant direction vs attractive/repulsive subspaces.

**D.2.3** Centroid velocity decomposition (B.4) — real vs imaginary contribution per layer type.

**D.2.4** Linear probes on $S$-only projections: train a linear classifier on $z_i^S = U_+^\top x_i$
only. Compare accuracy to probes on the full $x_i$ and on $z_i^A = U_A^\top x_i$ only.
**Prediction P6-D4:** probe on $z_i^S$ ≈ probe on $x_i$; probe on $z_i^A$ near chance.
If cluster membership is entirely in the real subspace, the compressed representation
$z_i^S$ (dimension $k \ll d$) preserves it.

**D.2.5** Merge event geometry: track inter-centroid distance in $S$ and $A$ subspaces separately
across layers approaching a merge. **Prediction P6-D5:** $d_S^{(L)} = \|U_+^\top(\bar{x}_C - \bar{x}_{C'})\|$
decreases monotonically as merge approaches; $d_A^{(L)}$ oscillates or does not decrease
systematically. The merge is driven by real-subspace convergence.

### D.3 Connecting Geometric and SAE Results

Once the direct geometric tests are run, Phase 3/4 results can be reinterpreted:

- **Geometric tests succeed, SAE tests fail:** metastable structure is geometrically real
  but not decomposable into sparse features. The SAE prior is wrong for this structure.
  Sparse coding finds syntax; the real subspace finds clusters. These are different things.
- **Both fail:** cluster structure is not linearly encoded in the residual stream. This
  would be surprising given HDBSCAN's success and would imply the clusters are a
  nonlinear phenomenon.
- **Both succeed:** SAE features are proxies for $S$-subspace projections. Phase 4
  MI tests should then show high correlation between feature activations and $z_i^S$ quantiles.

---

## Falsification Table

| ID | Claim | Test | Falsified if |
|---|---|---|---|
| P6-R1 | Clusters are degenerate in $S$ eigenspace | B.2 degeneracy ratio | $\sigma_B^2 / \sigma_W^2 \approx 1$ for $U_+$ |
| P6-R2 | Cluster-separating direction aligns with $S$ repulsive subspace | B.3 LDA–$S$ alignment | LDA aligns with $U_A$ or is random |
| P6-R3 | Centroid motion at merges is real-dominated | B.4 velocity decomposition | $\|\Pi_A \Delta\bar{x}\|$ exceeds $\|\Pi_S \Delta\bar{x}\|$ at merge layers |
| P6-R4 | $S$-only projection preserves cluster membership | D.2.4 linear probe | Probe on $z_i^S \ll$ probe on $x_i$ |
| P6-R5 | Plateau dynamics locally contracting in $S$, rotating in $A$ | B.5 local contraction | $W_C^A$ has eigenvalues outside unit circle during plateau |
| P6-A2 | $f_\text{rot}(h)$ predicts head type (self-similarity vs relational) | A.2 CC regression | No significant correlation across heads |
| P6-I1 | Induction write directions project onto rotational Schur planes | Induction OV decomposition | $p_\text{rot}(\text{induction}) \approx p_\text{rot}(\text{null})$ |
| P6-I2 | QK antisymmetry carries positional offset for induction | A.4 $A_{QK}$ contribution | Equal $A_{QK}$ fraction for induction and same-content pairs |
| P6-C1 | Head write subspace aligns with matching channel | C.1 write subspace alignment | $\text{align}_\text{rot}$ uncorrelated with CC |
| P6-DD1 | Zeroing imaginary channel reduces induction, preserves clusters | C.3 Intervention 1 | Clusters disrupted; or induction preserved |
| P6-DD2 | Zeroing real channel disrupts clusters, preserves induction | C.3 Intervention 2 | Induction disrupted; or clusters preserved |

---

## Module Structure

```
p6_induction_imaginary/
  README_phase6.md              — this file
  head_classify.py              — A.2: CC/PC scores, 2D head map
  induction_detect.py           — A.3: induction score, IS threshold calibration
  ov_subspace.py                — A.3: per-head rotational energy fraction, write direction projection
  qk_decompose.py               — A.4: S/A of QK, logit partitioning for induction pairs
  eigenspace_degeneracy.py      — B.2: within/between cluster variance in S eigenspace, degeneracy ratio
  centroid_velocity.py          — B.4: centroid displacement decomposed into real vs imaginary components
  local_contraction.py          — B.5: per-cluster local linear map, S and A spectral radii
  write_subspace.py             — C.1/C.2: per-head write subspace alignment, principal angles
  dissociation.py               — C.3: forward-pass interventions zeroing real or imaginary channel
  probe_subspace.py             — D.2.4: linear probes on real-only vs imaginary-only projections
  run_6.py                      — CLI entry point
  report_6.py                   — flat text report
```

**External dependencies:**
- `p2b_imaginary/rotational_schur.py` — Schur block extraction, $U_A$ planes, $U_S$ vectors
- `p2_eigenspectra/weights.py` — per-head OV, $W_Q$, $W_K$, $W_O$
- `p2_eigenspectra/decompose.py` — attn/FFN residual deltas (Intervention setup)
- Phase 1: HDBSCAN labels, plateau/violation layers, centroid trajectories
- Phase 4: LDA directions (optional B.3 enhancement)

---

## ALBERT-Specific Notes

ALBERT's shared weights mean the same $W_Q, W_K, W_V, W_O$ are reused at every
iteration. Consequently: Schur decomposition is computed once; head classification is
fixed; eigenspace degeneracy (B.2) must be computed per-iteration using Phase 1's
iteration-level activations. The double dissociation (C.3) operates on the residual
stream at each iteration, not on the weights.

This is actually a strengthening: if the same weights implement both channels, the
functional separation must arise from which subspace the incoming activation occupies,
not from separate weight matrices. This is a cleaner test of whether the residual stream
itself is partitioned into real and imaginary channels.
