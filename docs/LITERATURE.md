<!-- docs/LITERATURE.md -->
# LITERATURE — the cross-phase index

**Every phase now has a `lit-N.md`.** This file is the map, the shared method, the
cross-phase findings that no single phase's file could state, and the single ranked
queue of what to read.

**Produced 2026-09-16.** Extends `docs/literature_scan_2026-09-10.md`, which covered
§3.12-V (the 7d/7e/8 territory) only and which remains valid — every one of its four
verdicts is confirmed or sharpened below.

---

## 0. The constraint this whole review was run under, stated first

**`arxiv.org`, `semanticscholar.org`, `openreview.net`, `huggingface.co` and every
other scholarly host are blocked by this session's egress proxy.** Not one abstract
page could be fetched. `WebSearch` runs server-side and works; `WebFetch` and `curl`
do not.

So **every arXiv id, author list, date and finding in every `lit-N.md` came from a
search engine's summary of a page**, which is weaker than an abstract and much weaker
than a paper. The marks used throughout:

| mark | meaning |
|---|---|
| **[S]** | a search engine's summary of the paper's page was read; the paper was not |
| **[N]** | id and title appeared in a result list only; no summary text was seen |

**Nothing is marked "read", because nothing was.** This is the same discipline
`docs/literature_scan_2026-09-10.md` set, for the same reason, and it is not optional:
a search summary can misattribute a finding to the wrong paper, and at least one
attribution in these files is flagged as unreliable for exactly that reason
(`lit-1c.md` §5.3).

**Anyone with network access should work §5 top to bottom.** That is the highest-value
unblocked task in the project.

### 0.1 Amendment 2026-09-20 — `github.com` IS reachable, and it is a primary source

Measured this time rather than assumed, from a cloud session, and it changes how
future scans should be run. §5 item 18 guessed it (*"GitHub may be reachable where
arXiv is not — this is the fastest route"*) and the guess is correct.

| host | `WebFetch` from a cloud session |
|---|---|
| `github.com`, `raw.githubusercontent.com` | **reachable** — full README and source text |
| `arxiv.org` | blocked |
| `transformer-circuits.pub` | blocked |
| `huggingface.co` | blocked |
| `neuronpedia.org` | blocked |
| `WebSearch` (titles, snippets, summaries) | works |

**So a paper's companion-code repository is readable primary text even when the
paper is not**, and a finding taken from a README is not `[S]`. Worked in practice:
`p10_cluster_function/lit-10.md` §1 takes the Jacobian lens's defining equation, its
fitting corpus size, its API and its licence from `anthropics/jacobian-lens`'s README
and marks them **[R]**, while everything about the *published lens artifacts* stays
`[S]` because `huggingface.co` is blocked. **Try the companion repo before settling
for a search summary**, and use a third mark:

| mark | meaning |
|---|---|
| **[R]** | primary text was read — a README, source file or docstring, not a summary |

§5 item 18 should be read as a general method, not as one route to items 2 and 4.

---

## 1. The files

| Phase | File | State of the phase |
|---|---|---|
| 1 | `p1_mstate_tracking/lit-1.md` | Live, complete |
| 1b | `p1b_hemisphere/lit-1b.md` | Live, complete |
| 1c | `p1c_frames/lit-1c.md` | Live, not yet run against Pythia |
| 2 | `p2_eigenspectra/lit-2.md` | Live, complete |
| 2b | `p2b_imaginary/lit-2b.md` | Live, mid-rewrite |
| 2d | `p2d_operator_activation/lit-2d.md` | Live, not run |
| 3 | `archive/p3_crosscoder/lit-3.md` | Archived |
| 4 | `archive/p4_mstate_features/lit-4.md` | Archived |
| 5 | `p5_single_mstate_analysis/lit-5.md` | Archived code, live notes |
| 5b | `p5b_manifold_steering/lit-5b.md` | Built, **never run** |
| 5c | `archive/p5c_unclustered/lit-5c.md` | No code, framing promoted |
| 6 | `p6_subspace/lit-6.md` | Rebuilt live |
| 7 | `p7_motifs/lit-7.md` | Live |
| 7d | `p7d_redundancy/lit-7d.md` | Active |
| 7e | `p7e_consolidation/lit-7e.md` | Active |
| 8 | `p8_scale_ladder/lit-8.md` | Active |
| 9 | — | Pre-design, **no scan yet**; `p9_metric_intervention/notes-9.md` §11 and `plan-9.md` §12 name the searches |
| 10 | `p10_cluster_function/lit-10.md` | Pre-design. **PARTIAL** — one question settled as **[R]** from companion code, the rest named in `notes-10.md` §12. Does not follow the shape below |

Each file has the same shape: what the phase rests on → finding-by-finding verdicts →
what survives → directions to grow → a verification queue → the search log.

---

## 2. What is scooped, ranked by how much it costs us

**Read this section before writing anything up.**

1. **Phase 1's developmental arc.** *Tracing the Representation Geometry of Language
   Models from Pretraining to Post-training*, **2509.23024** **[S]**, NeurIPS 2025 —
   **Pythia 160M–12B**, three-phase non-monotonic sequence: warmup collapse →
   entropy-seeking expansion → compression-seeking consolidation. That is
   `status-1.md`'s "collapse, recovery, overshoot, slow decline", which it calls "the
   phase's main new object". **Strike or rewrite that sentence.**
2. **7d's second-order interaction instrument.** *Conditional Co-Ablation (CoAx)*,
   **2607.01940** **[S]**, July 2026. The 2026-09-10 scan named it as the nearest
   scoop; it is real and it is what it feared.
3. **7d's "behavioural proxies fail" thesis.** *Pattern Selectivity is Not Task-Causal
   Structure*, **2606.05378** **[S]**, June 2026 — the thesis in those words. **The
   developmental half (selectivity inverts *during formation*) survives; the
   cross-model half does not.**
4. **Phase 1's raw-effective-rank defect, as a phenomenon.** *Attention Sinks and
   Compression Valleys in LLMs are Two Sides of the Same Coin*, **2510.06477** **[S]**,
   ICLR 2026 — massive activations *provably* produce representational compression.
5. **Phase 1b's cone-collapse result.** The anisotropy / common-direction literature
   has had it since NAACL 2021.
6. **Phase 2's OV sign spectrum.** Elhage et al. 2021's copying-head statistic
   `Σλ/Σ|λ|`. **The project already adopted the field's instrument** — §3.12-N4
   identified it and §3.12-O ran it across 384 heads and eight checkpoints — which
   `p2_eigenspectra/lit-2.md`'s first draft missed and has been corrected for.
7. **Phase 2b's S/A decomposition and the `exp(A)` orthogonality identity.**
8. **Phase 1c's Euler-discretisation construction.** Standard since ~2018, and
   **2604.23740** **[S]** does it on the sphere.
9. **7e's "SVD order is a poor proxy for importance".** FWSVD, **2207.00112**,
   ICLR 2022.
10. **Phase 7d's super-additivity.** The Hydra effect, **2307.15771**, 2023.

**Two framing statements are stale rather than scooped**, and both are in Phase 1:
metastability may no longer be an open problem (**2410.06833** **[S]**), and the
theory now has a causal-mask version (**2411.04990** **[S]**) that Pythia, being
decoder-only, should have been compared against all along.

---

## 3. What survives, ranked

1. **The measured-null discipline, and specifically the anisotropy correction.**
   §3.12-V3's ambient participation ratio of **22 of 1024** kills the isotropic `k/d`
   baseline that the adjacent subspace literature appears to use. This was the
   2026-09-10 scan's rank-2 survivor; it holds, and `lit-6.md` §3 shows the live Phase
   6 rebuild already depends on it.
2. **Causal membership over every head, with the demonstration that structural *and*
   behavioural proxies fail.** Still the strongest card, now narrowed by
   **2606.05378** to its developmental half — which is the sharper half. See
   `lit-7d.md` §4.1.
3. **Force-typed interaction edges** (`lit-7.md` §3.1). Every attention-graph paper
   found uses the attention weight as the edge; Phase 7 uses `A_ij · V x_j`, typed by
   sign and rotational channel. Novel in composition, from parts the field trusts.
4. **The `γ_β` residual as a null for a trained network** (`lit-1c.md` §2). The field
   builds ODE transformers; nobody uses the ODE as a baseline to subtract.
5. **`L11H14`'s anti-optimality** — bottom-`r` > matched-random > top-`r` at every
   rank, negative at `r = 1`. A strengthening of FWSVD, conditional on a read.
6. **The operator–activation pairing `PR_M`** (`lit-2d.md` §3), narrowed but not
   closed by a targeted search.
7. **The energy axis on a checkpoint trajectory** (`lit-1.md` §3.2). No neighbour
   found for `E_β` violations tracked across training.
8. **The heterogeneity contrast** — a rank-1 gain-ordered member and a full-rank
   anti-ordered one doing the same job (`lit-7e.md` §3.2).
9. **The ambient-budget inversion** — the set writes into directions the residual
   stream barely uses (`lit-7e.md` §4.3). Plausibly unoccupied, unproven.

---

## 4. The cross-phase finding no single file could state

Three phases independently produced **a clean, uniform null that turned out to be
forced by the instrument**:

| phase | the null | why it was forced |
|---|---|---|
| **2b** | `elim_rotation = 0.0` in 35/35 runs, ~1e-15 residual | `exp(A)` is orthogonal and the readout is a function of `X Xᵀ`, which orthogonal maps preserve exactly (`lit-2b.md` §3) |
| **3** | crosscoder decoder directions align with `V` at chance (0.484 / 0.501) | the sparse objective drives `Dᵀ D → I`, so alignment with any fixed subspace is chance by construction (`lit-3.md` §1) |
| **5** | two of six cluster-selection criteria contributing 0.0 on every model | `load_phase1_run` reads the wrong event schema and returns a shape with no `merges` key — silently, forever (`lit-5.md` §2) |

And two more where a **metric was confounded by a single token or a single frame
choice**: Phase 1's raw effective rank dragged toward 2 by outlier-norm tokens
(`lit-1.md` §3.1), and Phase 1's Fiedler classification made vacuous by thresholds
calibrated on a different architecture (`status-1.md` D2).

The generalisation, which the searches found nowhere:

> **An intervention whose readout is invariant under it returns a clean null at
> machine precision, and perfect cross-architecture uniformity is the tell.** A
> scoring function whose terms can silently evaluate to zero still returns a ranked
> list, and the ranking still looks like a selection.

**This project has five worked instances from five phases, with numbers.** It is the
most publishable thing the literature review found, it requires no new compute, and
no other group is positioned to write it because no other group has kept the failures
in the repository. It is also a natural companion to `claims/EVALUABILITY.md` and to
standing rule 4 ("refuse rather than degrade").

---

## 5. The unified verification queue

Every phase file has its own; this is the ranked cross-phase order. **Work it top
down from a machine with arXiv access.**

| # | Paper | Decides |
|---|---|---|
| 1 | **2607.01940** *Conditional Co-Ablation (CoAx)* | How much of 7d's interaction matrix is left. Score vs decomposition; seed-set vs full matrix; ceiling handling |
| 2 | **2606.02378** *When Do Attention Circuits Form?* | The selectivity screen (`lit-7d.md` §4.1), the Pythia-1B rung (`lit-8.md` §3), Phase 7's `sink` motif |
| 3 | **2509.23024** *Tracing the Representation Geometry…* | Whether Phase 1's arc is a replication, and **whether they normalise before RankMe** (`lit-1.md` §3.1) |
| 4 | **2606.05378** *Pattern Selectivity is Not Task-Causal Structure* | How much of 7d's headline thesis survives |
| 5 | **2410.06833** *Dynamic metastability in the self-attention model* | Whether "Problem 1 is open" must be retracted from `design-1.md` |
| 6 | ~~**2411.04990** *Clustering in Causal Attention Masking*~~ **— READ 2026-09-20, `docs/readings/2411.04990.md` [R]**. Not a gradient flow (verbatim, §4); Thm 4.1 collapses all tokens to `x₁(0)` for arbitrary `Q,K`; the parking law is `Θ(β^((d−1)/2))` with the `d_eff` form as the paper's own open conjecture; Lemma C.1's count saturates in `n`; RMSNorm's diagonal is absorbable into `K,Q,V`. Opens five new pointers, **Castin–Ablin–Peyré 2024** the most consequential |
| 7 | **2510.06477** *Attention Sinks and Compression Valleys…* | Whether our normed-frame correction is a contribution (`lit-1.md` §3.1) |
| 8 | **2207.00112** FWSVD | Whether anti-optimality is a contribution (`lit-7e.md` §5.1) |
| 9 | **2511.16893** *Predicting the Formation of Induction Heads* | A free external adjudication on the formation window (`lit-7.md` §4.1, `lit-8.md` §4.1) |
| 10 | **2605.05115** *Manifold Steering…* | **Blocking for all of Phase 5b** |
| 11 | **2605.04279** *Gradient Flow Structure… Multi-Head Self-Attention* | Phase 2d's D1 referent and the aggregation problem |
| 12 | **2310.04625** *Copy Suppression* | The mechanism for 7d's super-additivity; Phase 2's repulsive subspace |
| 13 | **2607.24502** *Self-Attention Dynamics with Rotary Position Embeddings* | Phase 6's opening and Phase 2b's item 14 |
| 14 | **2601.10266** *Projection Kernel* head-subspace affinity | Whether the anisotropy correction is ours |
| 15 | **2604.23740** *Transformer as an Euler Discretization…* | Phase 1c's construction and the effective step size |
| 16 | **2307.15771** *The Hydra Effect* | The ~70 % restoration figure, for a like-for-like comparison |
| 17 | **2312.10794** *A mathematical perspective on Transformers*, **Bull. AMS 62(3) 2025** | Project-wide citation fix; whether theorem numbers moved |
| 18 | `github.com/skydancerosel/spectral-probe-circuits` | **Code, not a paper.** GitHub may be reachable where arXiv is not — this is the fastest route to items 2 and 4 |

---

## 6. The cheap, high-value experiments this review surfaced

All of these are **re-analysis of artifacts already on disk, or weights-only**, and
none needs new forward passes unless marked.

| # | Experiment | Phase | Why |
|---|---|---|---|
| 1 | **Rényi-parking prediction** vs measurements at 27 checkpoints — **restated 2026-09-20**, see `p10_cluster_function/lit-10.md` §5: the law is `Θ(β^((d−1)/2))`, in **β and dimension, not in n**, and the 0.7476 constant does not appear. The two tests that survive are a position-indexed **anchor** test (free) and a log-log **slope** regression returning `d_eff` (`math-10.md` §5) | 1, 10 | A published quantitative theory prediction, never checked on a trained model. `claims/adjudications/` holds zero entries |
| 2 | **Re-report the developmental arc in the normed frame** | 1 | Turns a scooped finding into a frame correction on two published papers |
| 3 | **Formation-point equation** (batch size, context size) vs the measured `(512, 2000]` window, at 70m and 410m | 7, 8 | Free external adjudication; two rungs of evidence already exist |
| 4 | **`PR_M` / `coupling_efficiency` vs 7d's 384-head causal sweep** | 2d, 7d | The one instrument that could rescue a structural proxy for causal effect, after `‖OV‖_F` failed at r² = 0.001 |
| 5 | **The copying score vs `frac_repulsive`** (token basis vs residual basis) | 2 | The score itself is **already computed** — `PROJECT.md` §3.12-O, all 384 heads, eight checkpoints. What is uncrossed is the residual-basis column, which §3.12-N4 says need not share a sign. Both are on disk |
| 6 | **`schur` vs `svd` basis on `L11H14`** | 7e | Weights-only, already named in `PROJECT.md`'s resume block, still unrun, and a stated hold sits on an instrument until it is done |
| 7 | **Selectivity screen vs causal membership, per checkpoint** (confusion matrix) | 7d | `lit-7d.md` §4.1 — the project's sharpest live disagreement with a published method |
| 8 | **Re-run `cone_collapse.py` with BOS excluded** | 1b | If universal cone-collapse is one token's norm, that is a publishable negative on our own result |
| 9 | **Run `border_vs_noise` and read it** | 1b, 5c | The Phase 5c boundary question has an answer waiting in a Phase 1b output |
| 10 | **Regress cluster membership on log token frequency** | 5c | The third story 5c never listed, and a confound for both it did |
| 11 | **Copy-suppression signature on 7d's compensating heads** | 7d | Turns a known phenomenon into a specific mechanism claim about our set |
| 12 | **The φ question** — antisymmetric fraction, QK vs OV halves | 6, 2b | Weights-only; closes an item handed over from the sister project |
| 13 | **Ordered-concept positive control for Phase 5b** (needs forward passes) | 5b | **Blocking.** Without it a negative result is uninterpretable |

---

## 7. What this review changes about the plan

- **`design-1.md` and `status-1.md` need edits**, not just annotations: the
  "Problem 1 is open" framing and the "main new object" sentence are both stale.
- **The rung policy needs a rule 4** — *a rung may be externally spent.* Pythia-1B is
  measured on the induction axis by two June 2026 papers. `lit-8.md` §3 proposes the
  amendment and argues it cuts in favour of keeping the reserve, with the registration
  recording what external measurements exist. **This is a human call.**
  **Pythia-1.4b is unaffected and is now the cleaner reserved rung** — prefer it for
  the first registration.
- **Phase 2d's sequencing gains a second reason to wait**, and D3 is the sub-experiment
  to descope if time is short.
- **Phase 5b must not run its main arm before its positive control.**
- **Phase 3 should not be reintroduced to re-test alignment at chance.** The null is
  explained; the live question is the sparse-vs-dense gap on a checkpoint axis.
- **§4 is a paper.** It costs no compute and nobody else can write it.
