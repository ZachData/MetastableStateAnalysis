<!-- p5b_manifold_steering/lit-5b.md -->
# Phase 5b — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; scholarly hosts are blocked by this session's egress proxy.
Marks: **[S]** search-engine summary read, **[N]** title and id only.

**Phase 5b is fully specified and has never been run** (`math-5b.md` §0), which makes
this the one phase where a literature review can still change the design rather than
grade it.

---

## 1. The paper the phase is built on — **confirmed to exist**

*Manifold Steering Reveals the Shared Geometry of Neural Network Representation and
Behavior*, arXiv **2605.05115** **[S]**, Daniel Wurgaft and collaborators.
`math-5b.md` cites it as "Wurgaft et al. (2026)" without an id; **the id is
2605.05115** and it should be added to the doc.

What the summary establishes, and two things `math-5b.md` does not record:

- The link is a **scaled isometry**: pairwise geodesic distance on the activation
  manifold correlates with geodesic distance between the corresponding output
  distributions. `math-5b.md` says "approximately isometric"; **"scaled" is
  load-bearing** — an isometry up to a global scale factor is a weaker and more
  achievable claim, and it changes what a Phase 5b null would look like.
- **The correlations are ≈ 0.999**, on **weekday, month, letter and age** tasks.

### 1.1 The risk this creates for Phase 5b's substitution, stated plainly

`math-5b.md` §1.2 is correct that the substitution — HDBSCAN cluster centroids in
place of labelled concept centroids — **is the entire test.** The tasks behind the
0.999 figures sharpen what that test is up against:

> **Weekday, month, letter and age are totally ordered, low-cardinality, and
> densely sampled.** A manifold threaded with a spline through such centroids is
> nearly a 1-D curve, and geodesic distance along it is nearly arc length. Correlations
> of 0.999 are what that regime produces.

Metastable-state cluster centroids from natural text have **no ordering, no fixed
cardinality, and membership that changes layer to layer**. So:

- A **negative** result for Phase 5b would be weak evidence — it could mean the
  substitution fails, or merely that unordered clusters do not admit a spline.
- **Therefore the phase needs an ordered-concept positive control**, run through our
  own pipeline, before its main arm means anything. Build one of Wurgaft's four tasks
  (weekday is cheapest), run HDBSCAN on *its* activations, and check that our
  unsupervised centroids recover their supervised manifold. **If they do not, the
  substitution has failed on the easy case and the main arm should not be run.**

This is the single most important change this file proposes to any phase's design.

---

## 2. The rest of the steering literature moved

- *Don't Lose Focus: Activation Steering via Key-Orthogonal Projections*,
  **2605.06342** **[N]**.
- *High-Dimensional Random Projection for Activation Steering in Language Models*,
  **2606.15092** **[N]**.
- *Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects*,
  **2606.08365** **[N]** — predicting side effects before intervening; relevant to
  5b's falsification design.
- `archive/p3_crosscoder/steering.py` already implements a steering intervention with
  a merge-event readout and a recorded null. `design-7.md` says to read it before
  writing new steering work. **Still true.**

---

## 3. The causal-chain claim, and the link that broke

`math-5b.md` §1.2 proposes the chain

> `V` eigenstructure → metastable attractor landscape → `M_h` ↔(isometry)↔ `M_y`

and already flags that its first link rests on Phase 2's claim, which Phase 2b
withdrew (`lit-2b.md`). This scan adds nothing that rescues it and one thing that
weakens it further: `lit-2.md` §1.1 shows the `V`-eigenstructure half is the
**copying-head eigenvalue statistic** from 2021, so the first link is not only
unsupported here but well-studied elsewhere, where it is *not* connected to attractor
landscapes.

**Recommendation:** run B, C and D (the downstream links) as designed, and **drop the
causal reading of the first link from the phase's stated goal** until Phase 2's
rescaled frame works on Pythia. The isometry question is worth answering on its own.

---

## 4. Directions to grow

1. **Build the ordered-concept positive control** (§1.1). Blocking; everything else
   waits.
2. **Add the id `2605.05115` to `math-5b.md`** and correct "isometric" to "scaled
   isometry" wherever the distinction matters to a threshold.
3. **The one thing Phase 5b could give that 2605.05115 cannot:** an *unsupervised*
   route to `M_h`. Their manifold needs concept labels; ours needs none. If the
   substitution works even partially, "manifold steering without concept labels" is a
   real contribution and a short paper.
4. **Check whether 2605.05115 has a checkpoint axis.** If it does not — and nothing in
   the summary suggests it does — then *when the isometry forms* is unclaimed, and
   this project has 27 Pythia checkpoints and a cluster tracker.

## 5. Verification queue

1. **2605.05115** — the task list, the manifold-fitting procedure, the scale factor,
   and whether any unsupervised variant is attempted. **Blocking for the whole phase.**
2. **2606.08365** — side-effect prediction, for 5b's falsification thresholds.
3. **2605.06342**, **2606.15092** — current steering practice.

## 6. Search log (2026-09-16)

- `Wurgaft manifold steering shared geometry neural network representation and behavior isometry concept centroids`
- `crosscoder sparse autoencoder decoder directions alignment weight matrix eigenvectors geometry features 2026`
