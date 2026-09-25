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
| A0 | *Added after `/challenge-pr` on #99.* Bound the share of singletons in `selection.py`'s trivial filter, then re-run the smoke | row 2 | minutes | the bound is a new constant to choose; it fixes agglomerative's end only |
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
