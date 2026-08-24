# Phase 6 — MATH (study notes)

## 0. What this document is

Companion to `math-2b.md` (which supplies the S/A decomposition) and `math-5.md`. Phase 6 asks the
question Phase 2b's null left open:

> Phase 2b answered *"does rotation drive clustering?"* — no. It did **not** answer **what the
> rotational subspace does.** A subspace orthogonal to clustering dynamics is free to carry other
> computation without interfering with the attractor structure.

**Before anything else: this phase's stated premise has been withdrawn.** `design-6.md` opens with
"Phase 2b established that the antisymmetric component $A$ is dynamically neutral — removing it
from OV leaves energy violations unchanged; the symmetric component $S$ carries 100% of violation
causality." That is the `elim_rotation = 0.0` result, and `math-2b.md` §3.1 shows it was **an
orthogonal-invariance identity, forced by construction before any data was read.**

The good news is that **Phase 6 survives its own premise being void.** Its hypothesis is a
*positive* claim about what $A$ does, testable on its own terms — it never needed "rotation is
causally neutral for clustering" as a lemma, only as motivation. But the framing needs rewriting:
Phase 6 is not building on an established null, it is asking an open question that Phase 2b's
instrument could not have answered either way.

Status: **partial run, one model (albert-xlarge-v2), 0/6 tested predictions passed, 6/12 never
ran.** The header in `README_phase6.md` says "Not started" and is stale.

---

## 1. The division-of-labour hypothesis

$$
\textbf{Real subspace } (S):\quad \text{semantic similarity, metastable-state organization}
$$
$$
\textbf{Imaginary subspace } (A):\quad \text{relational computation}
$$

Concretely: tokens in the same cluster are **degenerate in $S$'s eigenspace**; while $A$ carries
operations that *cannot reduce to pairwise inner products on current positions* — induction,
previous-token heads, copy/name-mover heads, anti-similarity heads, coreference.

$$
\textbf{Unifying principle: self-similarity operations use real structure; relational operations
use imaginary structure.}
$$

This is a much stronger and more falsifiable claim than "rotation exists and might do something."
It predicts **a specific correlation structure** (Part A), **a specific geometric signature**
(Part B), and licenses **a direct causal test** (Part C).

The intuition is sound and worth stating in the language of `math-2b.md` §2.2: a $2\times2$ Schur
block acts as a rotation by $\theta$ in a plane. A rotation *cannot* be expressed as a function of
$\langle x_i, x_j\rangle$ alone — it is inherently a map between *different* directions, which is
exactly the algebraic signature of a relation rather than a similarity. Symmetric operators, by
contrast, are diagonalizable in an orthogonal basis and act by scaling along fixed directions —
which is what "measure similarity to a template" looks like.

---

## 2. Building the projectors

`subspace_build.py` partitions $d_{\rm model}$ using the real Schur decomposition of each head's
$\mathrm{OV} = W_VW_O$:

| Schur block | condition | destination |
|---|---|---|
| $1\times1$ | $\lambda > \texttt{eig\_tol}$ | attractive → $U_{\rm pos}$ → $U_S$ |
| $1\times1$ | $\lambda < -\texttt{eig\_tol}$ | repulsive → $U_{\rm neg}$ → $U_S$ |
| $1\times1$ | $|\lambda| \le \texttt{eig\_tol}$ | **kernel — excluded from $S$** |
| $2\times2$ | $|\det|^{1/2} > \texttt{eig\_tol}$ | rotation → $U_A$ |
| $2\times2$ | $|\det|^{1/2} \le \texttt{eig\_tol}$ | **kernel** (`n_kernel_rot`) — also excluded |

Both `eig_tol` and `block_tol` are **relative to $\lVert T\rVert_F$**, not absolute — which handles
OV matrices whose operator norm differs substantially across layers and models. (Same fix as
`math-2b.md` §2.2's subdiagonal threshold; the absolute version is the recurring defect in this
codebase.)

After per-head extraction, cross-head unions are orthonormalized with an explicit **resolution
order**:

$$
U_A := U_A \cap \mathrm{span}(U_S)^\perp,
\qquad
U_{\rm neg} := U_{\rm neg}\cap\mathrm{span}(U_{\rm pos})^\perp
$$

i.e. **$S$ wins over $A$**, and $U_{\rm pos}$ wins over $U_{\rm neg}$. Exclusive projectors
$P_{S\setminus A}$, $P_{A\setminus S}$ expose the clean partition, and **principal angles and
overlap diagnostics are stored per layer** — which is the right instinct, and §7 argues they are
the key to reading the phase's headline result.

---

## 3. Part A — behavioural evidence

### 3.1 The CC/PC map

Heads are classified on two axes: **content-coupling** (CC — attends by inner-product similarity)
versus **positional-coupling** (PC — attends by relative offset). The prediction is that rotational
energy fraction correlates with position on that 2D map.

**Anti-similarity heads are included as a *second* relational class distinct from induction** —
high imaginary OV fraction, low induction score, negative CC. Collapsing all relational computation
into "induction" would miss a mechanism that could drive merge events through the
attention-*routing* channel specifically, complementary to (not the same as) Phase 2's V-repulsive
*energy* mechanism. That is a genuinely useful distinction: routing and energy are different
channels and the project has mostly measured the second.

### 3.2 The QK antisymmetry test (P6-I2)

$$
\mathrm{logit}(i,j) \;=\; \underbrace{x_i^\top S_{QK}\,x_j}_{\text{content similarity}}
\;+\; \underbrace{x_i^\top A_{QK}\,x_j}_{\text{positional offset}}
$$

For an induction head attending from $i$ to $j$ where $\mathrm{token}[j{-}1]\approx
\mathrm{token}[i{-}1]$, the offset "$j$ is one ahead of the matching position" must be encoded
somewhere. The prediction: **the $A_{QK}$ fraction of the logit is elevated for induction pairs
relative to same-content non-induction pairs**, with effect size $> 0.05$ absolute.

This is elegant — the symmetric part *cannot* distinguish $(i,j)$ from $(j,i)$, so any
order-sensitive information must live in the antisymmetric part. It follows from the algebra, not
from a modelling assumption.

### 3.3 …and on Pythia it has a null it does not yet have

`math-1.md` §2.4: **RoPE supplies offset-dependent antisymmetry by construction.** On each rotary
plane $R(\Delta) = \cos(\Delta\omega_t)I + \sin(\Delta\omega_t)J$ exactly, so

$$
a_{\rm frac}(\Delta) = \frac{2\sum_t \sin^2(\Delta\,\omega_t)}{d_{\rm head}}
$$

which is $0$ at $\Delta = 0$ and rises with $|\Delta|$. `core/rope.py` states the consequence
directly: *"this is why P6-I2 needs a new null model — rotary supplies offset-dependent antisymmetry
by construction, so 'a_frac is elevated for induction pairs' is not evidence of anything until it
is measured against this baseline."*

**The live question is whether the *content* bilinear $W_QW_K^\top$ carries antisymmetry beyond
what rotary contributes at the same offsets.** On GPT-2/ALBERT (no rotary) P6-I2 is clean as
written; on Pythia it is confounded, and the exact closed-form baseline already exists in
`rope_sa_fractions`. Since induction pairs have a *systematically different offset distribution*
than same-content pairs — that is what makes them induction pairs — this is not a small
correction: the two populations being compared differ in exactly the variable that drives the
architectural baseline.

---

## 4. Part B — the SAE-free definition of a metastable cluster

This is the phase's most valuable conceptual contribution:

> Tokens whose projections onto $S$'s dominant attractive eigenvectors are **nearly identical** are
> **dynamically equivalent.**

That is a *theory-grounded* definition of a cluster — derived from the operator that generates the
dynamics — rather than an artifact of HDBSCAN's hyperparameters. It reframes *"is HDBSCAN finding
something real?"* from an act of faith into **a testable geometric claim**, and it is the natural
answer to `math-1.md` §10.3's complaint that the plateau detector has no connection to the paper's
actual characterization of metastability.

Four independent angles on the same underlying claim: **B.2** degeneracy ratio, **B.3** LDA-vs-$S$
alignment, **B.4** centroid-velocity decomposition, **B.5** local contraction.

**B.5 is the one result with positive signal.** 29/44 plateau steps contract in the real subspace
and 28/44 show neutral rotation in the imaginary — as predicted. The merge-destabilization half
fails (only 121/341 merge steps show the predicted real-subspace expansion), **possibly a
layer-type-label mismatch** between Phase 1's per-layer plateau/merge classification and this
phase's per-step dynamic labelling rather than a hypothesis failure. That is a plausible and
checkable alternative, and it is the same producer/consumer class as `math-5.md` §9.

*(`eigenspace_degeneracy.py` currently raises `NameError: d undefined` — a good oracle-tier
candidate, since degeneracy ratio has a known-correct answer on planted synthetic clusters.)*

---

## 5. Part C — the double dissociation

The single most falsifiable prediction in the phase. Three surgical interventions on the residual
stream **during inference** — more direct than Phase 2b's rescaled frames, which removed the
operator's contribution retroactively:

| arm | intervention | prediction |
|---|---|---|
| 1 | $h_{\rm attn} \leftarrow h_{\rm attn} - \Pi_A h_{\rm attn}$ | induction score **drops**; clusters **preserved**; $E_\beta$ violations unchanged |
| 2 | $h_{\rm attn} \leftarrow h_{\rm attn} - \Pi_S h_{\rm attn}$ | induction **preserved**; clusters **disrupted**; violations eliminated |
| 3 | zero a **random subspace of matching dimension** | both degrade — replicating neither pattern |

The falsification conditions are enumerated per arm rather than left implicit, which is the right
discipline for a double dissociation (the whole point is that *both* arms must confirm *their
own* prediction; either one failing kills it).

**Note arm 3 gets the dimension control right** — "random subspace of *matching dimension*." Hold
that thought for §7, because Part B does not.

**Blocked**, on wiring rather than on data: `model`, `tokenizer`, `text`, `hook_targets` are never
threaded into `ctx` before the subexperiment registers. And `run_intervened_forward` has not been
migrated to `core/intervention.py::run_model_with_hook` — the *safe* half of that migration (one
architecture, no ALBERT dispatch, unlike `causal_tests.py`'s). Migrating it also fixes a latent
**embedding-index mismatch**: `dissociation.py` skips the embedding (`hidden_states[1:]`) while
`core/models.py` and `run_model_with_hook` include it at index 0. Internally consistent while
baseline and intervention are only ever compared to each other — **a real misalignment the moment
`ctx["baseline_labels"]` comes from genuine Phase 1 output.**

---

## 6. Part D — metastable states without SAEs

Positions the phase's geometric tests as the answer to a limitation named back in Phases 3/4:
**sparsity pressure allocates dictionary capacity to frequent independent features, and metastable
cluster membership may not decompose into independent sparse features — not because the structure
is absent, but because sparse coding is the wrong prior.**

Three readings are laid out in advance:

| geometric | SAE | reading |
|---|---|---|
| succeeds | fails | the SAE prior was wrong for this structure |
| fails | fails | clusters aren't linearly encoded — **surprising**, given HDBSCAN's success |
| succeeds | succeeds | SAE features are proxies for $S$-projections |

Registering all three, including the one that would embarrass the phase, is the same discipline as
`math-5c.md` §4's three-outcome rank test.

---

## 7. The first-run result — and a third explanation nobody has listed

### 7.1 The inversion

| measurement | predicted | observed |
|---|---|---|
| LDA alignment with $U_{\rm neg}$ (real repulsive) | high | **0.067** |
| LDA alignment with $U_A$ (imaginary) | low | **0.887** |
| layers showing predicted direction | 49/49 | **0/49** |
| probe accuracy, real-only | high | **0.152** (chance) |
| probe accuracy, imaginary-only | low | **0.564** |
| probe accuracy, full activation | — | 0.590 |

Cluster-separating directions align overwhelmingly with the **imaginary** subspace, and
imaginary-only probes nearly match full-activation accuracy while real-only probes sit at chance.
This is the exact inverse of the hypothesis.

`status-6.md` lists two live explanations: **(a)** a projector-construction/labelling bug in
`subspace_build.py` swapping $U_{\rm neg}$ and $U_A$ (which would invert all four geometry tests
*together*, exactly as observed), or **(b)** the hypothesis is wrong under ALBERT's weight-tying.
The design correctly prioritizes ruling out (a) first, since a labelling bug produces exactly this
pattern and is checkable in isolation.

### 7.2 Explanation (c): the comparison is not dimension-normalized

**For a random unit vector $v\in\mathbb R^d$ and a $k$-dimensional subspace $U$:**

$$
\mathbb E\big[\lVert P_U v\rVert^2\big] = \frac{k}{d}
$$

So **alignment with a subspace scales with that subspace's dimension**, and comparing raw alignment
against $U_A$ versus $U_{\rm neg}$ measures dimension at least as much as content.

Now: **is $\dim U_A \gg \dim U_{\rm neg}$?** Almost certainly yes, but the *size* of the gap is
not something the existing numbers settle, and it is important not to repeat the error
`math-2b.md` §2.3 names. The 84–97.5% figure is **spectral energy**, not dimension, and the two
diverge sharply per head — `head_circuits.py` measures one head at **5.5% of dimensions** rotating
in full space against **87.5%** within its own core subspace. So the energy fraction cannot be
read off as a dimension fraction, and §2's resolution order ($S$ wins ties) shrinks $U_A$ further
by the overlap.

$$
\frac{0.887}{0.067} \approx 13
$$

**A dimension ratio of ~13 between $U_A$ and $U_{\rm neg}$ would reproduce the observed alignment
ratio exactly, with no content explanation at all.** Whether the true ratio is near 13 is
**directly measurable and unreported** — `subspace_build.py` knows both dimensions per layer.
That single number decides whether explanation (c) is the whole story, part of it, or none. The probe result has the same structure: a probe fit on a
high-dimensional subspace has more capacity than one fit on a low-dimensional subspace, so
"imaginary-only 0.564 vs real-only 0.152" may be a statement about **available dimensions**, not
about where cluster identity lives.

**The fix is cheap and the phase already knows the pattern** — `dissociation.py`'s arm 3 is
precisely a matched-dimension random control (§5). Three options, in order of directness:

1. Report $\lVert P_U v\rVert^2 \big/ (\dim U / d)$ — alignment relative to chance.
2. Compare against a **random subspace of matched dimension** drawn per layer, exactly as arm 3
   does.
3. Report **principal angles** between the LDA direction and each subspace rather than a scalar
   projection — these are already stored per layer (§2).

**Until one of these is done, the inversion is not evidence for or against the hypothesis**, and
neither (a) nor (b) can be adjudicated. This is the same defect as `math-5b.md` §8.1's missing
dimension-matched control in Sub-exp D and `math-1b.md` §6's un-nulled cone verdict — third
instance, and here it is sitting on the phase's only substantive result.

### 7.3 The ALBERT caveat is stronger than stated

$0/49$ layers is **not 49 independent measurements**. ALBERT ties weights: one OV matrix, one Schur
decomposition, one projector pair, 49 activation snapshots. The design doc frames weight-tying as a
*strengthening* test in principle — if the same weights implement both channels, functional
separation must arise from which subspace the incoming activation occupies rather than from
separate weight matrices, which is a cleaner test of whether the residual stream itself is
partitioned. That argument is good. But empirically it cuts the other way here: **the inversion
needs a per-layer-weight model before it can be read as a functional-separation failure rather than
an ALBERT artifact.**

---

## 8. Code map

| File | Part | Role |
|---|---|---|
| `subspace_build.py` / `write_subspace.py` | — | Schur partition into $U_{\rm pos}/U_{\rm neg}/U_A$/kernel; cross-head orthogonalization; exclusive projectors; per-layer principal angles. (`write_subspace.channel_orthogonality` currently called with an unsupported `top_r` kwarg) |
| `head_classify.py` | A.2 | CC/PC classification, anti-similarity class |
| `induction_ov.py` | A.3 | Induction scoring from OV structure |
| `qk_decompose.py` | A.4 | $S_{QK}/A_{QK}$ split, logit partitioning, P6-I2 |
| `eigenspace_degeneracy.py` | B.2 | Degeneracy ratio (`NameError`) |
| `probe_subspace.py` | B.3 / D.2.4 | LDA alignment, subspace-restricted linear probes |
| `centroid_velocity.py` | B.4 | Velocity decomposition across subspaces |
| `local_contraction.py` | B.5 | Contraction/rotation per subspace per step — the one positive result |
| `dissociation.py` | C.3 | The double dissociation; the only module needing a live forward pass |
| `p6_io.py`, `report_6.py`, `run_6.py` | — | IO, reporting, CLI |

Read-only imports: `p2b_imaginary/rotational_schur.py` (Schur blocks, $U_A/U_S$),
`p2_eigenspectra/weights.py` (per-head OV, $W_Q/W_K/W_O$), `p2_eigenspectra/decompose.py`
(attn/FFN deltas for intervention setup).

---

## 9. Open questions

Tracked: Track A's prerequisites (`qk_matrices`/`qk_logit_matrices` never populated); the
dissociation wiring; the two `NameError`/kwarg bugs; the two competing explanations for the
inversion; and the need for a non-ALBERT run.

Surfaced by writing this document:

1. **Explanation (c) — dimension — should be ruled out before (a) or (b)** (§7.2). It is cheaper
   than the projector-labelling audit, it predicts the observed ratio quantitatively, and if it
   holds then both listed explanations are moot. **This is the single highest-value action in the
   phase**, and every ingredient (stored principal angles, arm 3's matched-dimension control) is
   already built.

2. **The phase's premise needs rewriting, and doing so may change what Part C tests** (§0). With
   `elim_rotation = 0.0` withdrawn as an identity, "A is dynamically neutral for clustering" is
   **unestablished** — Phase 2b's Gram-based instrument could not have detected it either way
   (`math-2b.md` §3.4). But arm 1 of the double dissociation predicts *"clusters preserved"* when
   $A$ is zeroed, which is precisely the withdrawn claim, **now tested by an instrument that can
   actually see it**: a forward-pass intervention is not orthogonally invariant, because attention
   logits are computed in a fixed basis and the unembedding is fixed. **So Part C is not merely
   Phase 6's causal test — it is the real version of the test Phase 2b could not run**, and it
   should be labelled as such. That reframing raises its priority considerably.

3. **P6-I2 needs the rotary null before it runs on Pythia, and the confound is correlated with the
   contrast** (§3.3). Induction pairs differ from same-content pairs *in offset*, which is exactly
   the variable driving rotary's architectural antisymmetry. The closed form exists; the corrected
   statistic is $a_{\rm frac}^{\rm observed}(\Delta) - a_{\rm frac}^{\rm rotary}(\Delta)$ matched
   per pair.

4. **Part B's definition of a cluster should be run *against* HDBSCAN, not just alongside it.**
   §4 gives a theory-grounded cluster definition — degeneracy in $S$'s dominant eigenspace — and
   the phase uses it to *test* HDBSCAN's labels. But it can also **replace** them: cluster tokens
   directly on their $S$-projections and compare the resulting partition to HDBSCAN's with the ARI
   machinery `core/functional_distance.py` already provides (`math-1.md` §3.8). That would give the
   project a third, mechanism-derived labelling to sit beside the geometric and functional ones in
   `frame_agreement` — **and it is the only one of the three derived from the operator that
   generates the dynamics.** Given `math-1.md` §10.3's concern that the plateau detector is a
   flatness heuristic with no theoretical grounding, this is the closest thing to a principled
   cluster definition anywhere in the project.

5. **The kernel bucket is excluded and never reported.** §2 routes near-zero modes — both
   $1\times1$ blocks with $|\lambda| \le \texttt{eig\_tol}$ and $2\times2$ blocks with
   $|\det|^{1/2} \le \texttt{eig\_tol}$ — to the kernel and excludes them from both $U_S$ and
   $U_A$; correct, since a null direction is neither attractive, repulsive, nor meaningfully
   rotational. But its **dimension is a quantity worth having**: it is the part of the
   residual stream this head's OV circuit *does not write to at all*, and its size across layers and
   checkpoints is a direct measure of how much of the space each head ignores. Given `math-2d.md`
   §2's operator-conditioned rank asks a closely related question from the activation side, the two
   should be compared.

6. **Nothing checks whether $U_S$ and $U_A$ are stable across checkpoints.** Every Phase 6 test
   conditions on projectors built from one set of weights. If the S/A partition itself reorganizes
   during training — which `math-2b.md` §8's untested "does the complex fraction have a
   developmental trajectory?" question directly bears on — then per-checkpoint results are not
   comparable, because the coordinate system moves underneath them. Principal angles between
   consecutive checkpoints' $U_A$ bases would settle it, are weights-only, and reuse the diagnostic
   §2 already stores per layer.
