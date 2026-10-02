<!-- p1d_cluster_ensemble/lit-1d.md -->
# Phase 1d — LITERATURE (trigger 1: before `design-1d.md` changes, 2026-09-25)

**Why:** 1d's first run picked each family's most extreme scale (`status-1d.md`
"First real run"). Fixing the scale reopens `design-1d.md`, and a design that reopens
needs a literature scan first (`CLAUDE.md` "Literature scans", trigger 1). The user
added three questions to the scan (2026-09-25): clusters defined by attention over a
window of layers, clusters as potential wells, and whether prompt length or prompt
count limits either.

**Marks:** **[R]** primary text read; **[S]** search summary or abstract only;
**[P]** read earlier in this project (pointer given). Web search and arXiv
abstracts, 2026-09-25. Nothing here is registered; no data was opened.

## 0. Headline

| # | finding | mark | changes |
|---|---|---|---|
| 1 | 1d's stage 1 ranks candidates by **raw** subsample stability. The literature's standard warning is that raw instability "trivially scales with k, regardless of what the underlying data structure is", so the argmax is "meaningless" unless each candidate is **normalised** by its own null (von Luxburg 2010 §2) | [R] | a standard remedy exists, but **on 1d's data it does not work as stated** (row 1a). `design-1d.md` §"The gate is asymmetric" keeps raw stability on purpose, because its null reaches the ceiling |
| 1a | *Added after `/challenge-pr` on #99.* Since #98, a degenerate null draw scores stability 0, and the null mean is exactly 0 for agglomerative at 0.05 / 0.25 and for HDBSCAN `mcs=5` at L12 / L18. The ratio is then infinite, and ranking by significance ties them at the floor p (1/21). On the stored quick-grid candidates, ranking by observed/null moves k-means off k = 2, but twice onto k = 4, **the other end of the grid** (the review's table, PR #99) | data, smoke run | normalisation can trade one extreme scale for the other. The grid must extend past both ends before A means anything |
| 2 | *Revised after `/challenge-pr` on #99.* Agglomerative's perfect stability is **not** von Luxburg's "perturbation too small". At L12 its pick is k = 452 of 467 tokens, 95 % singletons, and that passes `selection.py`'s trivial filter, which bounds the dominant cluster but not the share of singletons | checked in `p1d_results.json` | **a code defect**, cheaper to fix than any of A–D (option A0) |
| 3 | Consensus clustering finds "apparently stable clusters" in unimodal data with no clusters (Şenbabaoğlu et al. 2014) | [S] | 1d's per-family null gate is necessary. A consensus-level statistic (PAC) also needs its own null |
| 4 | Multiscale methods do not pick one scale. They sweep a resolution parameter and keep the scales where partitions are **robust** (Markov stability; Jeub et al. 2018; ToMATo persistence; Fred–Jain lifetime) | [S] | option D below |
| 5 | The theory supplies its own scale: `δ = c β^{-1/2}` (2411.04990 §5.1; the paper's simulations use `c = 4`) | [P] `docs/readings/2411.04990.md` | option C, but it inherits β's open ×8 convention (STATE Blocked 9) |
| 6 | The user's "potential well" definition already exists in the theory. With `Q = K = V = I`, one self-attention step is a mean-shift step on the von Mises–Fisher potential `φ_β(x) = Σ_j exp(β⟨x, x_j⟩)`. Modern Hopfield networks name the fixed points that average a subset of patterns "metastable states", present for intermediate β (Ramsauer et al. 2020) | [S] | a candidate definition with a theory-given scale (§3) |
| 7 | Attention as a Markov chain has metastable sets that group tokens ("Attention (as Discrete-Time Markov) Chains", 2507.17657). Multiplying attention across layers is attention rollout (Abnar & Zuidema 2020) | [S] | the user's "interaction over 2–3 layers" is Markov time on the layer chain (§2) |
| 8 | Not found in this scan: any paper that defines clusters in a **trained** decoder's residual stream by the model's own kernel and scale. That comes from about 20 searches, which is weak evidence of novelty | [S] | — |

## 1. Scale selection: what the stability literature says

| source | claim | mark | bearing on 1d |
|---|---|---|---|
| Ben-David, von Luxburg & Pál, COLT 2006, *A sober look at clustering stability* | For idealised k-means, stability holds when the objective has a unique global minimiser and fails only under symmetry of the distribution, **whether k is right or wrong** | [S], via von Luxburg 2010 §3.1 [R] | stability cannot *select* k on its own. It is at most a filter |
| Shamir & Tishby 2008–2010 (COLT, NIPS, MLJ) | Stability does not "break down" as n grows: suitably rescaled, instability converges to a distribution, and relative stability across k stays discernible | [S] | the 2006 result is asymptotic. At n ≈ 300–500 tokens, rescaled comparisons are still informative |
| von Luxburg, *Clustering stability: an overview*, FnT ML 2010 (`1007.1075`) §2 | Two normalisations. (a) Reference null: `Instab_norm = Instab / Instab_null`, where the null scrambles each dimension (Fridlyand & Dudoit 2001), **which is 1d's shuffled-dimension null**. (b) Random labels: divide by the instability of permuted labels (Lange et al. 2004). Or a test: choose the k whose stability is most significant against the null. She adds that "normalization often has no effect", untested | [R] | 1d already computes (a) for its `top_m` candidates only. Ranking on it, for **every** grid point, is the textbook protocol |
| Lange, Roth, Braun & Buhmann, *Neural Comput.* 2004 | Stability read as a classification risk on a second sample, normalised by a random predictor | [S] | the cheaper normalisation (no refits for the null) |
| Hennig, *CSDA* 2007 | Cluster-wise stability: the mean bootstrap Jaccard of each cluster to its best match; ≥ 0.85 "highly stable" | [S] | per-cluster, not per-partition: fits a graded annotation better than one ARI |
| Şenbabaoğlu, Michailidis & Li, *Sci. Rep.* 2014 | Consensus clustering reports chance partitions of unimodal data as stable; PAC (the share of ambiguous pairs) infers k better | [S] | the consensus matrix itself needs a null |

## 2. Matched-scale and multiscale ensembles

| source | idea | mark | bearing |
|---|---|---|---|
| Fred & Jain (evidence accumulation, 2002–2005) | Many fine k-means runs → co-association → hierarchical clustering; the number of clusters is taken where a level has the longest **lifetime** | [S] | the ensemble defines a hierarchy, and the scale is read from it rather than set per family |
| Jeub, Sporns & Fortunato, *Sci. Rep.* 2018 (`1710.02249`) | Sample the whole resolution range, then build a **hierarchical** consensus | [S] | a matched-scale ensemble: members are compared at equal resolution |
| Delvenne, Yaliraki & Barahona, *PLoS ONE* 2012 (`1109.5593`); PyGenStability | Markov time of a random walk is the resolution. Robust scales are plateaus with low variation of information across optimisations | [S] | applies directly to attention, which is row-stochastic (§2.1) |
| Chazal, Guibas, Oudot & Skraba, *J. ACM* 2013 (ToMATo) | Mode-seeking on a density, then merge modes by topological persistence; the persistence diagram shows how many modes are prominent | [S] | option D's selection rule, and the machinery for §3's wells |

### 2.1 The user's question: clusters as "who interacts with whom" over a window

| point | status |
|---|---|
| The data exist: every run stores `attentions.npz`, shape `(24, 16, n, n)` float32 (checked on `pythia-410m-step143000_wiki_paragraph`, n = 467, 158 MB) | checked |
| Attention is row-stochastic, so each layer is a Markov chain on tokens. A window of w layers is the product `Π (½I + ½Ā_ℓ)` (rollout, with the residual as the ½I term). w is Markov time, so "2 or 3 layers" is a resolution parameter, and Markov stability's plateau rule selects among windows | literature [S]; not built |
| **Sink.** GPT-NeoX adds no BOS token, so position 0 takes the sink role (`p10_cluster_function/attention-10.md` §2.1). Raw attention communities would be a star around it. Drop the sink column and renormalise, using `core/sink_audit.py`'s machinery | known here |
| **Causal.** Aᵢⱼ = 0 for j > i, so the graph is directed and early tokens can only receive. Symmetrise (A + Aᵀ), or use co-attention A Aᵀ ("attend to the same tokens"). These answer different questions. **Directed Markov stability is degenerate as it stands:** a lower-triangular row-stochastic chain has its whole stationary distribution on token 0, so it needs a teleportation term (`/challenge-pr` on #99, finding 5) | open choice |
| **Attention weight is not force.** In the particle picture, j moves i by `Aᵢⱼ · P_{xᵢ}(V xⱼ)`. The weight alone ignores V and the 16 heads' different V | caveat |
| **What it adds over distance.** Attention is `softmax(⟨Q xᵢ, K xⱼ⟩/√d_h)`, a function of geometry. If `QᵀK` were a scaled identity, attention communities would equal cosine clusters at scale β^{-1/2}, so **where they differ measures what QK does beyond cosine**. That makes it a family with a different bias (an interaction, not a distance), and the one closest to the theory's own coupling | recommendation |

## 3. The user's question: clusters as potential wells

| point | status |
|---|---|
| In the idealised model (`Q = K = V = I`, one head), the energy is `E_β = (1/2β) Σᵢⱼ exp(β⟨xᵢ, xⱼ⟩)`, and a test point feels `φ_β(x) = Σⱼ exp(β⟨x, xⱼ⟩)`: an unnormalised von Mises–Fisher kernel density on the sphere with concentration β. `φ_β(xᵢ)` is `math-10.md` §2's `Z_{β,i}` | theory (Geshkovski et al., `2305.05465`) |
| Gradient ascent on `φ_β` projected to the sphere is mean shift. A cluster is then the **basin** of a mode, a well, whose depth and persistence are defined even for one token. That is the user's "a well, not a bag of particles" | standard (Carreira-Perpiñán review [S]) |
| **Letting the particles move too** (blurring mean shift) is the self-attention dynamics itself, and it ends in one cluster; a stopping rule is needed (Carreira-Perpiñán 2006 [S]). Freezing the landscape at a layer and following mean shift to its modes is the "wells of this layer" reading | distinction to keep |
| **Enough particles?** Estimating a density nonparametrically from n ≈ 300–500 points in 1024 dimensions is hopeless **if the bandwidth has to be learned**. Here it does not: β comes from the model, so `φ_β` is a defined function at any n. What n limits is resolution: at most n wells, and if β^{-1/2} is below typical nearest-neighbour cosine distances, every token is its own well. That is checkable, by counting wells against β | answer |
| **Trained model:** 16 heads, each with its own `φ_h(x) = Σⱼ exp(⟨Q_h x, K_h xⱼ⟩/√d_h)`, plus the MLP. There is no single energy, and the causal system is a *sequential* gradient flow, not a mean-field one (`math-10.md` §7.5). The landscape is exact for the idealised model and a per-head proxy for Pythia | caveat |
| Prediction to test against: Ramsauer et al. (2020) report early layers averaging globally and later layers averaging over subsets (metastable) [S]. Bruno, Pasqualotto & Agazzi (`2410.23228`; `2509.25040`, NeurIPS 2025) predict the metastable cluster structure from β (a Gegenbauer index) and, with β ∝ N, three phases: low-dimensional collapse, then clusters, then sequential merging [S] | leads |
| Link to option C: strong Rényi centres at `δ = cβ^{-1/2}` are the theory's predicted nuclei. Whether each nucleus sits in its own well of `φ_β` is a direct, cheap consistency check between the two definitions | proposal |

## 4. The user's question: context length and more prompts

| point | reading |
|---|---|
| Current n: 265–512 tokens per v1 prompt (line counts of `tokens.txt`, approximate); Pythia's context is 2048 | checked |
| Lemma C.1 (2411.04990, [P]): the expected number of strong Rényi centres has **no n in the limit** and saturates from below. Longer prompts test whether 300–500 is already saturated | the theory's own reason for longer prompts |
| Bruno et al.'s regime is β ∝ N; their phases need large N | longer prompts are closer to the theory's regime |
| `core/config.py` already has `LENGTH_SWEEP_TOKENS = [50, …, 400]` on `wiki_paragraph`. Extending it to ~1000–2000 is a config change plus forward passes | cheap to add |
| **Cost scales as n²** for stored attention: 158 MB at n = 467, so ~2.9 GB per run at n = 2000 (twice that with `plateau_attentions.npz`). Stability subsampling costs rise too | the binding cost |
| **Prompt count** matters for generalising the definition and for registered tests, not for defining it on one cloud. The only usable prompts are the 8 v1 (the 12 v2 are held out). New prompts need fresh forward passes (~200 s each on 410m, CPU) | **recommend: length first, count later** |

## 5. Options for fixing 1d's scale

| option | what | from | cost | risk |
|---|---|---|---|---|
| A0 | *Added after `/challenge-pr` on #99.* Bound the share of singletons in `selection.py`'s trivial filter, then re-run the smoke | row 2 | minutes | the bound is a new constant to choose; it fixes agglomerative's end only. **Done 2026-09-25** (bound 0.5; `status-1d.md` "A0") |
| A | Keep 1d; rank every grid point by **normalised** stability (observed ÷ null, or most significant against it) instead of raw | von Luxburg §2; Fridlyand–Dudoit | null for every grid point: ×(n_grid / top_m) of today's stage 2 | **undefined where the null mean is 0, and ties at the floor p** (row 1a). It can move k to the grid's other end. Needs a wider grid and more null draws. Still one scale per family |
| B | Matched k: compare families at equal cluster count | Jeub et al.; `docs/PHASE_SYNTHESIS.md` (`P-S1` at matched k) | cheap | HDBSCAN and modularity have no k; matching on the output count is post hoc |
| C | Theory scale: every distance-taking family (agglomerative threshold, Rényi centres, vMF mean shift) at `δ = cβ_eff^{-1/2}` | 2411.04990 | cheap | β's ×8 convention is undecided (Blocked 9), a factor √8 ≈ 2.8 in δ; `c` is free (the paper uses 4) |
| D | Hierarchy: report the merge tree over a scale sweep and keep the robust scales (plateaus, persistence, lifetime), with C's δ marked on it | §2 | moderate | a set of scales, not one answer. The Phase 10 readers would need to say which level they read |

**Recommendation (for the user to decide; reordered after `/challenge-pr` on #99).**
First, **A0**, which is a defect fix, not a design choice. Second, the question 1d was
revived for, whether tuning reduces the float-noise drift, which none of A–D blocks.
Third, **D** as the frame and **C** to mark the theory's point on it: §3's `φ_β`
landscape swept over β around `β_eff`, with its wells and merge tree, plus §2.1's
attention-window communities (symmetrised) as the interaction family. The seven families
become checks at matched scale. **A** is not recommended as stated (row 1a).
**Blocked on the user before C and §3: β's convention (Blocked 9).** Both read β, and
the ×8 changes the answer. D + C turns 1d from "is HDBSCAN a good choice" into "what
is a cluster". That change is the user's to make, not the scan's.

## 6. Queue (not read)

- Ben-David, von Luxburg & Pál 2006 as primary text (the PDF host failed); Lange et al. 2004 (PubMed blocked by a captcha).
- `2509.25040` and `2410.23228` as primary text: whether the Gegenbauer index gives a count for d = 1024 at Pythia's β.
- `2507.17657`: how it handles causal masks, and whether its metastable sets use PCCA or eigengaps.
- `2601.21366` (localisation of the mean-field landscape with the MLP) and `2608.08922` (clustered attractors; its "overlap gap" condition may be a testable separation criterion). Both were [N] in `p1_mstate_tracking/lit-1.md`.
- Fridlyand & Dudoit 2002 (Clest): the test form of normalisation.

## 7. A matched-covariance Gaussian null (added 2026-09-26)

Scan for the unit "are tokens lumpier than their covariance explains" (user,
2026-09-26; `status-1d.md` "Matched-covariance Gaussian null"). Web search, 3 queries.

| # | finding | mark | changes |
|---|---|---|---|
| 1 | SigClust defines "one cluster" as data from a single Gaussian. It compares the 2-means cluster index (within-SS / total-SS) to its distribution under a Gaussian with the data's covariance. For HDLSS data it does not use the plug-in covariance: invariance plus a factor model with a background-noise floor (Liu, Hayes, Nobel & Marron 2008, JASA 103:1281) | [S] | the null and the primary statistic (`ci2`). 1d uses the plug-in covariance and measures its bias (`--calibrate`) instead of estimating the floor |
| 2 | SHC applies SigClust's test at every node of a hierarchical tree from the root down, with FWER control (Kimes, Liu, Hayes & Marron 2017, Biometrics 73:811; R `sigclust2`) | [S] | the natural next step for stage F's merge tree; not done here |
| 3 | 1–3 "rogue dimensions" dominate cosine similarity in contextual LMs. Their contribution to the expected cosine is `E[u_i v_i]`. Standardising or removing them changes similarity-based conclusions (Timkey & van Schijndel 2021, EMNLP) | [S] | the `centred_norogue` frame and its rogue measure `m_i²` |

## 8. Testing one cluster at a time (added 2026-09-30)

Scan for `design-1d.md`'s revision (Blocked 10 decided: the graded vote is retired, so a
cluster is admitted one group at a time against a null). Web search, 3 queries.

| # | finding | mark | changes |
|---|---|---|---|
| 1 | HDBSCAN's per-cluster `cluster_persistence_` has been gated against noise: run HDBSCAN on many noise datasets and admit a real cluster whose persistence exceeds a high percentile (99.7th) of the noise persistences (`2501.16294`, stellar populations) | [S] | the shape of the admission rule `design-1d.md` proposes, with three changes: the noise is the matched-covariance Gaussian (§7), not uniform; the threshold is a quantile of each draw's **maximum**, so the error rate is per record, not per cluster; and the statistic is `S_C / |C|`, not `cluster_persistence_`, which hdbscan divides by the tree-wide maximum λ (`/challenge-pr` on #121) |
| 2 | SHC tests each node of the tree top-down and stops at the first non-significant node (Kimes et al. 2017; §7 row 2) | [S] | rejected as the definition: 1d's local excess (HDBSCAN groups, `status-1d.md`) sits under a root that is Gaussian-typical at L1–16, and a top-down stop would never reach it |
| 3 | Selective inference after hierarchical clustering gives an exact test of two clusters' mean difference, but under an isotropic Gaussian with known variance (Gao, Bien & Witten 2022, JASA; k-means extension Chen & Witten) | [S] | not usable as stated: the residual's covariance is far from σ²I (rogue coordinates, §7 row 3). Named so a reviewer can propose it with that caveat |
| 4 | A permutation test of each dendrogram split against permuted memberships (Park et al. 2009, PMC3023458) | [S] | a membership permutation keeps the geometry, so it tests the algorithm's consistency, not structure beyond covariance |

## 9. The identity-weights positive control (added 2026-10-01)

Scan before `design-1d.md`'s "Identity-weights positive control" freezes (trigger 1). The
user named two papers: read `2605.09213`, re-read `2411.04990`'s causal results. Both
came from arXiv HTML through a fetch summarizer (mark [H]: primary text, but read through
a summarizer, so equations are as quoted and the rest is paraphrase). Plus 2 web searches.

| # | finding | mark | changes |
|---|---|---|---|
| 1 | `2411.04990`'s (CSA): `ẋ_k = P_{x_k}( Σ_{j≤k} e^{β⟨Qx_k,Kx_j⟩} V x_j / Z_k )`, **self included**, `Z_k` the softmax partition over `j ≤ k`. The first token is autonomous. Thm 4.1: with `V = I` and any `Q, K, β`, every token converges to `x₁(0)`. No rate, no early-vs-late speed statement (checked: "not addressed") | [H], agrees with [P] `docs/readings/2411.04990.md` | the simulator is this equation with `Q = K = V = I`. Thm 4.1 is a test (`x₁` never moves; all tokens approach `x₁(0)`). Nothing in the paper predicts *which* tokens join `x₁` first, so the opening question is open, not answered by the theory |
| 2 | Its simulations are `d = 2`, `n = 200`, `β = 64` (Figs 2–3) and `d = 3`, `n = 32`, `β = 9`, `T = 5000` (Fig 1). Metastable clusters are seeded by strong Rényi centres at `δ = 4β^{-1/2}`. Nothing is shown in high `d` | [H] | at Pythia's β (0.43 or 3.46, Blocked 9) `δ = 4β^{-1/2}` is 6.1 or 2.2 rad. That is past the sphere's diameter, or near it, so the theory predicts **one centre, `x₁`**, at real β. A grid that only spans real β cannot test whether admission recovers multi-cluster states, so it extends to 64 |
| 3 | `2605.09213` (Duerinckx, Geshkovski, Rossi, v1 2026-05-09) studies a **different** causal model: angles on the circle, `θ̇_j = (1/Z_{N,j}) Σ_{k<j} e^{−λ(j−k)/N} w'_β(θ_j − θ_k)`, `w_β = e^{β cos θ}`, **no self term**, and `Z` sums only the ALiBi weights (no softmax partition). Proves a mean-field limit `f(t, σ, θ)` indexed by relative position `σ = j/N`, rate `N^{-δ∧1/2} e^{Ct}` | [H] | its closed forms (Bessel `I₁` correlations) are for that model, so they **cannot** test our simulator. What transfers is the frame: position `σ` is a coordinate of the limiting object, i.e. in the theory, causal dynamics *are* position-indexed |
| 4 | Its primacy mechanism: early tokens are "repeatedly reused by later ones", `sup_k Σ_j ω_{j,k} ≃ log N`. Lost-in-the-middle (U-shaped retrieval: primacy, recency, a unique interior minimum) holds under `t · sup_n a_n ≤ min{3 − √3, 2(1 − e^{−λ})}`, i.e. **short times**. Its Fig. 3 (`N = 64`, `β = λ = 1`) shows drift toward `θ₁` plus coherence among the latest tokens | [H] | a named prediction for the opening question: at small β and short `t`, the first tokens' pull dominates. The simulator records `cos(x_k(t), x₁(t))` by position, so it can show primacy (and, without ALiBi, no recency term) directly |
| 5 | The open problem it states is our object: `ẋ_j = (1/Z) Σ_{k<j} exp(β⟨Q(t)x_j, K(t)x_k⟩ − λ(j−k)/N) P^⊥_{x_j}(V(t)x_k)` with general, time-dependent `Q, K, V`; "RoPE-type encodings fall beyond the present theory" | [H] | Pythia (RoPE, MLP, 24 untied layers) is outside both papers. The positive control tests the definition on the case the theory covers, not Pythia |
| 6 | Attention sinks trace to a "variance discrepancy" from value aggregation under the causal mask: the first token attends only to itself, later tokens average a growing prefix, so the first token stays a high-variance outlier (`2605.06611`, Li, Jiang, Sun, Hu, 2026-05). Random-init transformers already have "extreme token preferences" and an attention-sink-linked "positional discrepancy" (`2602.05927`, Li, Tong, Wang, Hu, 2026-02) | [S] | the same mechanism #123 read off step 0's stored attention (near-uniform attention, so early positions share the first tokens' value vectors). The simulator at β = 0 is that mechanism with nothing else in it |
| 7 | Not found: any paper that runs the identity-weight dynamics from a real model's embeddings and asks whether a cluster definition recovers the clusters it makes (2 searches; weak evidence) | [S] | — |

## 10. The Blocked 11⁗ programme: intervention, architecture null, scale spectrum (added 2026-10-02)

Scan for unit 0 of the programme in `status-1d.md` "Blocked 11⁗ decided" (trigger 1:
`design-1d.md`'s "Programme" section freezes after it). 8 web searches, 5 fetches
(marks as §9; **[M]** = measured in this repo for this scan, input named). No forward
pass was run and no admission output was opened.

| # | finding | mark | changes |
|---|---|---|---|
| 1 | PolyPythias (`2503.09543`, ICLR 2025): 9 extra seeds of 14m, 31m, 70m, 160m and **410m**, same code, hyperparameters and standard (non-deduplicated) Pile as Pythia; a seed changes **both** the weight init and the data order (decoupled seeds only at 160m); the original run is seed 0. At 410m, **seeds 3 and 4 are outliers** (loss spikes, ≥ 2 SD below the mean on downstream tasks) | [H] | unit 2 has 10 real 410m inits and 10 trained endpoints. Seeds 3, 4 are kept and flagged, not dropped |
| 1a | `EleutherAI/pythia-410m-seed{1,9}` each list 155 revisions on the Hub, `step0` and `step143000` among them; the repo's `pythia-410m` is the standard-Pile model (same as PolyPythias) | [M] Hub API, 2026-10-02 | both checkpoints reachable from the local box; ~18 more checkpoints in the HF cache |
| 2 | Pythia's init: `small_init` σ = √(2 / 5d) for the embeddings, QKV and the MLP's first matrix; `wang_init` σ = 2 / (L√d) for the attention output (`dense`) and the MLP's `dense_4h_to_h` (Pythia / GPT-NeoX-20B papers) | [S] | a re-init must use these two σ |
| 2a | Measured on the cached `pythia-410m` `step0`: σ 0.01974–0.01978 (embed, QKV, `h_to_4h`, unembed; √(2/5120) = 0.019764) and 0.00260–0.00261 (`dense`, `4h_to_h`; 2/(24·32) = 0.002604), every bias 0, every LayerNorm (1, 0). **transformers' `GPTNeoXPreTrainedModel._init_weights` draws σ = `initializer_range` = 0.02 for every Linear**, so `model.init_weights()` gives the two output projections 7.7× Pythia's σ | [M] | unit 2's re-init writes the two σ itself; `_init_weights` is not a Pythia init. Normality of the draws is assumed, not checked |
| 3 | Attention sinks form at **absolute position 0**, not on a token: re-sampling the first token keeps the sink, and fixing the first two moves it to position 2. Pythia is in the study; the sink is present at 14m and stronger with size in Pythia. The first token's hidden-state norm is large from an early block (Gu et al., `2410.10781`, ICLR 2025) | [H] | unit 1 moves the passage off position 0, so the sink stays with the preamble. The token-0 rule below excludes a position, not a string |
| 4 | Massive activations: a few activations orders of magnitude above the rest, on the starting token **and the first delimiter (`.` or `\n`)**, acting as input-independent biases (Sun, Chen, Kolter & Liu, `2402.17762`, COLM 2024) | [S] | look for a second massive token, not just position 0 (row 4a) |
| 4a | Norm over the layer's median (step143000: median over positions ≥ 1; step 0: all positions), the 7 deduped-batch v1 prompts, `data/phase12/2026-09-01_18-25-12` (step143000) and `2026-09-01_13-30-37` (step 0): **position 0 is 20–50× at L8–20 in all 7 trained prompts; the first `\n` (Ċ) is a second massive token in 2 of 7** (`hdbscan_code` position 34, `latex_monograph` position 10; 18–41× at L8–20); every other token ≤ ~3×. **Step 0: maximum 1.30×, at position 0, in all 7** | [M] | the rule is a norm bound, placed at 10× in the gap between ~3× and 18× |
| 5 | Random-init transformers already show a first-token "positional discrepancy" (variance decays ∝ 1/√i along the sequence, seed-independent) and a **seed-dependent** token preference (contraction along a random direction; different seeds favour different tokens) (`2602.05927`; RoPE GPT-2 and LLaMA-2 nano / 1.2B, not Pythia) | [H] | the positional part is what step 0's opening showed; the seed-dependent part is why unit 2 needs several inits rather than one step 0 |
| 6 | Random-weight models as the control for interpretability claims: automated interpretability metrics score random and trained transformers alike (`2501.17727`, ICLR 2026) | [S] | precedent for unit 2's question; no paper found that uses a set of re-initialisations as the null for clusters in the residual stream (3 searches; weak evidence) |
| 7 | Position invariance by intervention: SHAPE (`2109.05644`) feeds one input at offsets k ∈ {0, 100, 250, 500} and averages, per position, the cosine of hidden states across offsets | [S] | unit 1's per-token readout (cosine of a passage token's state at preamble P against P = 0). No paper found that asks whether *clusters* move with the text (2 searches) |
| 8 | Pythia's training sequences are packed 2049-token windows; documents are joined by an end-of-document token, a window rarely starts at a document start, and attention crosses document boundaries | [S] (Pythia repo / paper) | text at a non-zero position after an EOD token is the training distribution. Unit 1's primary join is `<|endoftext|>`; a plain `\n\n` join is the arm |
| 9 | Markov stability on point clouds (Liu & Barahona, `1909.04491`): build a CkNN graph (k = 7, δ ≈ 1.5–2.4), scan Markov time, call a scale robust where the VI between partitions at nearby times is a low block and the VI across Louvain runs at one time is low; results stable over a range of the graph parameter. PyGenStability (`2303.05385`) implements it. Persistent homology of a multiscale clustering (`2305.04281`) and hierarchical planted-partition benchmarks (Jeub et al. 2018, §2) test such methods on nested planted structure | [S]; Liu & Barahona's parameters from its PDF's method lines (grepped, not read whole) | the alternative to the merge tree for unit 3; rejected as primary in `design-1d.md` (its scale is Markov time, not an angle, and it adds a graph parameter and Louvain randomness) |
| 10 | Attention-as-Markov-chain metastability (`2507.17657`): λ₂ of the row-stochastic matrix, products across layers; **causal masks are not discussed**; vision models only | [H] | closes §6's queue item: nothing in it handles a lower-triangular chain, so §2.1's teleportation caveat stands |

## Sources

- von Luxburg 2010 — https://arxiv.org/abs/1007.1075
- Ben-David, von Luxburg & Pál 2006 — https://link.springer.com/chapter/10.1007/11776420_4
- Shamir & Tishby 2010 — https://link.springer.com/article/10.1007/s10994-010-5177-8
- Lange et al. 2004 — https://direct.mit.edu/neco/article/16/6/1299/6841/Stability-Based-Validation-of-Clustering-Solutions
- Hennig 2007 — https://www.homepages.ucl.ac.uk/~ucakche/papers/clusta.pdf
- Şenbabaoğlu et al. 2014 — https://www.biorxiv.org/content/10.1101/002642v3.full
- Jeub, Sporns & Fortunato 2018 — https://arxiv.org/abs/1710.02249
- Delvenne, Yaliraki & Barahona 2012 — https://arxiv.org/abs/1109.5593
- Chazal et al. 2013 — https://dl.acm.org/doi/10.1145/2535927
- Carreira-Perpiñán, mean-shift review — https://faculty.ucmerced.edu/mcarreira-perpinan/papers/mean-shift-review.pdf
- Ramsauer et al. 2020 — https://arxiv.org/abs/2008.02217
- Bruno, Pasqualotto & Agazzi — https://arxiv.org/abs/2410.23228, https://arxiv.org/abs/2509.25040
- Abnar & Zuidema 2020 — https://arxiv.org/abs/2005.00928
- Erel et al. 2025 — https://arxiv.org/abs/2507.17657
- Álvarez-López, Geshkovski & Ruiz-Balet 2026 — https://arxiv.org/abs/2601.21366
- Gao, Yang & Chen 2026 — https://arxiv.org/abs/2608.08922
- Liu, Hayes, Nobel & Marron 2008 (SigClust) — https://www.tandfonline.com/doi/abs/10.1198/016214508000000454
- Kimes, Liu, Hayes & Marron 2017 (SHC) — https://academic.oup.com/biometrics/article/73/3/811/7537682
- Timkey & van Schijndel 2021 — https://aclanthology.org/2021.emnlp-main.372/
- HDBSCAN persistence against noise (stellar populations) — https://arxiv.org/abs/2501.16294
- Gao, Bien & Witten 2022 (selective inference, hierarchical clustering) — https://www.semanticscholar.org/paper/181289c336f649356d91af4af3548eab84bb8b2e
- Park et al. 2009 (permutation test for cluster significance) — https://pmc.ncbi.nlm.nih.gov/articles/PMC3023458/
- Duerinckx, Geshkovski & Rossi 2026 (kinetic theory, lost in the middle) — https://arxiv.org/abs/2605.09213 (HTML v1)
- Karagodin, Polyanskiy & Rigollet 2024 (causal attention masking) — https://arxiv.org/html/2411.04990v2
- Li, Jiang, Sun & Hu 2026 (attention sink, variance discrepancy) — https://arxiv.org/abs/2605.06611
- Li, Tong, Wang & Hu 2026 (transformers born biased) — https://arxiv.org/abs/2602.05927
- van der Wal et al. 2025 (PolyPythias) — https://arxiv.org/abs/2503.09543
- Biderman et al. 2023 (Pythia; init, packing) — https://arxiv.org/abs/2304.01373, https://github.com/EleutherAI/pythia
- Black et al. 2022 (GPT-NeoX-20B; small_init, wang_init) — https://arxiv.org/abs/2204.06745
- Gu et al. 2025 (when attention sink emerges) — https://arxiv.org/abs/2410.10781
- Sun, Chen, Kolter & Liu 2024 (massive activations) — https://arxiv.org/abs/2402.17762
- Automated interpretability metrics, trained vs random — https://arxiv.org/abs/2501.17727
- Kiyono et al. 2021 (SHAPE) — https://arxiv.org/abs/2109.05644
- Liu & Barahona 2020 (graph-based clustering via Markov stability) — https://arxiv.org/abs/1909.04491
- Arnaudon et al. 2023 (PyGenStability) — https://arxiv.org/abs/2303.05385
- Schindler & Barahona 2023 (persistent homology of multiscale clustering) — https://arxiv.org/abs/2305.04281
