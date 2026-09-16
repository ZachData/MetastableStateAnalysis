<!-- p1b_hemisphere/lit-1b.md -->
# Phase 1b — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy, so not even an abstract page could be fetched. Marks: **[S]**
a search engine's summary of the page was read, **[N]** title and id only.

**The headline: Phase 1b's positive finding — universal cone-collapse — is a
rediscovery of the anisotropy / common-direction literature, which has had it since
2021. Phase 1b's *instrument* is better than that literature's, and the instrument
is the part worth defending.**

---

## 1. What the phase rests on

- Geshkovski et al., **2312.10794** (*Bulletin of the AMS* **62(3)**, 2025) **[S]**,
  Theorem 6.3 / Lemma 6.4: if all tokens start in one open hemisphere
  `{x : ⟨w,x⟩ > 0}`, convergence to a single point is exponential. `cone_collapse.py`
  implements that containment condition as an LP feasibility test. Phase 1's `k = 2`
  eigengap result motivated the competing "split" hypothesis.

Note the same staleness flagged in `lit-1.md` §1.3: **2411.04990** *Clustering in
Causal Attention Masking* **[S]** proves convergence for **arbitrary QK with V = I**
under a causal mask. Whether the hemisphere hypothesis is still what the causal
theorem needs is unchecked, and Pythia is causal.

---

## 2. Finding by finding

| Phase 1b finding | Nearest prior work | Verdict |
|---|---|---|
| **Cone-collapse holds in 100 % of tested layers/models; no antipodal split** | The anisotropy / representation-degeneration literature says this in different words. *Too Much in Common: Shifting of Embeddings in Transformer Language Models and its Implications*, **NAACL 2021** (`aclanthology.org/2021.naacl-main.403`) **[S]**; *The Shape of Learning*, **2311.05928** **[S]**; *Revisiting Anisotropy in Language Transformers: The Geometry of Learning Dynamics*, **2604.08764** **[S]** | **NOT NEW as a phenomenon.** "All embeddings lie in one open half-space" is the cone statement; "embeddings occupy a narrow cone / share a dominant common component" is the anisotropy statement. They are the same geometry |
| The Fiedler axis is real, stable, identity-preserving, but is **not a separator** | Nothing found stating exactly this. The nearest object is *Hidden Coalitions in Multi-Agent AI: A Spectral Diagnostic from Internal Representations*, **2605.06696** **[S]**, which takes a **global Fiedler bipartition of internal representations and refines it into nested sub-coalitions** where within/across contrast survives | **Contested — read 2605.06696.** Its "refine only where contrast survives" gate is structurally Phase 1b's `classify_regime_relative`, arrived at independently |
| Per-layer Fiedler value / algebraic connectivity as a transformer diagnostic | *Training-Free Spectral Fingerprints of Voice Processing in Transformers*, **2510.19131** **[S]** — attention induces dynamic graphs over tokens, Fiedler value per layer as "a single interpretable endpoint per layer" | **NOT NEW as an instrument.** Their framing (graph signal processing over attention-induced graphs) is the field's name for what `sinkhorn.py` does |
| The Fiedler vector is orthogonal to the cloud's mean direction **by construction**, so testing it against the mean token direction is unreachable (measured `|cos|` 0.000–0.085) | The common-direction literature explains *why* the mean direction dominates: the "common enemies effect" — every non-target embedding receives the same gradient direction under cross-entropy, so a shared component accumulates (**NAACL 2021** **[S]**) | **Our observation is correct and the reason is published.** Block A's correction is sound; its cause is not ours |
| Boundary-vs-noise: is the unclustered population the Fiedler-boundary population? (rank AUC) | Nothing found | **Looks new.** Also Phase 5c's question — see `archive/p5c_unclustered/lit-5c.md` |
| Two matched nulls (shuffled-dimension, uniform-sphere) for "n points in d dimensions separate for free" | Standard, but not found applied to the cone question | **Looks new in application** |

---

## 3. What survives

### 3.1 The LP is a better instrument than a cosine average, and that is the defensible claim

The anisotropy literature establishes cone-like concentration with **average pairwise
cosine similarity** and with PCA of the common component. Phase 1b answers a
**containment** question — *does there exist a `w` with `min_i ⟨w, x_i⟩ > 0`* — by
linear programming, which is exact rather than aggregate. A cloud can have low mean
cosine and still be contained; it can have high mean cosine and still straddle.
**The two measurements are not interchangeable, and no source found runs the exact
one.**

The `normalized_margin` quantity is the real deliverable, not the regime label —
`design-1b.md`'s errata already says so. Against the anisotropy literature that
distinction is the entire contribution: margin is a *graded* containment measure with
a matched null, where "anisotropy" is a scalar summary with no null at all.

### 3.2 The PCA lifting asymmetry

`design-1b.md`'s errata records that a reduced-space **cone_collapse** verdict lifts
exactly (`w = Vt[:k].T @ w_r`) while a reduced-space **split** verdict may be a
projection artifact, so `escalate_on_split=True` re-solves at full `d`. Nothing found
states this asymmetry. It is small, it is correct, and any paper doing half-space
tests on PCA-reduced activations is exposed to it.

### 3.3 The Hungarian tie hazard

Exact ties let the assignment solver return either pairing on 4 of 500 random label
pairs, and anchor chaining propagates each flip. A methods note, not a finding — but
it is the kind of thing that silently corrupts a published trajectory, and it is
worth a paragraph in any write-up of Fiedler-axis tracking across layers.

---

## 4. Directions to grow

1. **Read 2605.06696 first.** It is the closest neighbour and it is 2026. If its
   nested-refinement gate is Phase 1b's relative rule, the phase's methodological
   card is already played by someone else and the remaining card is the LP.
2. **Cross-check the cone result against the attention-sink account.** `lit-1.md`
   §2 records **2510.06477** **[S]**, which traces mid-layer compression to the
   BOS token's massive norm. A single huge-norm token would dominate any
   containment test. **Re-run `cone_collapse.py` with BOS excluded.** If universal
   cone-collapse survives BOS removal it is a fact about the token cloud; if it does
   not, it is a fact about one token, and that is a publishable negative on our own
   result. **Cheapest item in this file.**
3. **Put the margin on the checkpoint axis with a null.** `axis_settling_step` and
   `cross_checkpoint_axis_rotation` exist and have never been crossed with the
   anisotropy literature's training-dynamics claim (**2604.08764** **[S]**: "the
   intrinsic dimension of embeddings increases in the initial phases of training").
   A settling step for the Fiedler direction, against an anisotropy onset step, on
   the same checkpoints, is a two-curve figure nobody has.
4. **Ask the causal question.** 2411.04990's masked dynamics may make the hemisphere
   condition unnecessary or replace it. If the containment condition is no longer
   the relevant hypothesis, Phase 1b's headline is answering a question the current
   theory does not ask.

---

## 5. Verification queue

1. **2605.06696** — is its Fiedler bipartition on token representations or on agent
   embeddings, and is the refinement gate the same test as `classify_regime_relative`?
2. **NAACL 2021 `2021.naacl-main.403`** — does it state containment (all in one
   half-space) or only concentration (high mean cosine)? Decides §3.1 entirely.
3. **2510.19131** — what exactly is Fiedler-valued: post-softmax attention,
   Sinkhorn-normalised attention, or a token Gram matrix? Determines whether our
   baseline-subtraction problem (defect D2) is theirs too.
4. **2604.08764** — the learning-dynamics half; what onset step, on what models.
5. **2311.05928** — the bell-shaped anisotropy profile, against our layer profile.

---

## 6. Search log (2026-09-16)

- `Fiedler vector spectral bipartition token representations transformer layers attention graph Laplacian analysis`
- `anisotropy common direction embeddings all vectors in narrow cone representation degeneration language models`
- (inherited from `lit-1.md`) `Clustering in Causal Attention Masking arXiv 2411.04990 decoder-only self-attention dynamics`

Surfaced, not pursued: **2607.00063** *Spectral Geometry and Bosonic-Bloch Probes*
**[N]**; **2408.08073** *Extracting Sentence Embeddings from Pretrained Transformer
Models* **[N]**.
