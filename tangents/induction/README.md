# Tangent: Induction Heads and Particle Interaction

## Question

Phase 1 found mutual nearest-neighbor pairs at plateau layers that split into two categories:
- **Semantic** (category 1): mutual NNs that share the same HDBSCAN cluster. Dense-region attraction. Examples: `novelist ↔ poet`, `failed ↔ rejected`.
- **Artifact** (category 2): mutual NNs in *different* HDBSCAN clusters. Local attraction without density support. Examples: `▁he ↔ ger` (cross-position subword completion of "Heger").

Are category-2 pairs driven by induction heads? If yes, this connects two literatures: the Geshkovski et al. particle-interaction framework (energy, attraction, repulsion on S^{d-1}) and the Olsson et al. induction head framework (attention pattern composition, in-context learning).

## Predictions (falsifiable)

1. **Induction × Fiedler correlation is negative.** Induction heads route narrowly between specific prior-occurrence pairs → low Fiedler value (near-disconnected attention components). Phase 2 found V-repulsive heads have *high* Fiedler (broad mixing). These should be distinct populations.

2. **Category-2 pairs are preferentially attended by induction heads.** For each mutual-NN pair at plateau layers, the heads with the strongest bidirectional attention should have high induction scores for category-2 pairs and low induction scores for category-1 pairs.

3. **At plateau layers, induction-mediated pairs have increasing per-pair energy** (local attraction) even when global energy drops (V-repulsive through FFN). The metastable plateau is the regime where these forces balance.

If prediction 1 fails (positive or null correlation), induction heads are not the narrow-routing population and the NN structure at plateau layers has a different origin.

If prediction 2 fails (no separation between category-1 and category-2 in top-head induction score), the category-2 pairs are not attention-routing artifacts — they are genuine geometric structure driven by something other than induction.

## Connection to the energy functional

The paper's energy is $E_\beta = \frac{1}{2\beta n^2} \sum_{i,j} e^{\beta \langle x_i, x_j \rangle}$.

Phase 2 showed V's negative eigenvalues create a global repulsive force (energy drops) delivered primarily through FFN. This experiment asks about the *attractive* side: what holds specific token pairs together against that repulsive force?

If induction heads are the mechanism, the picture is:
- **Global repulsion** (V-repulsive eigensubspace → FFN) pushes the token cloud apart, decreasing global energy.
- **Local attraction** (induction heads routing between specific token pairs) pulls those pairs together, increasing their pairwise energy contribution.
- **Metastable plateau** = the layer range where these forces balance. The NN graph locks in (induction routing is stable) before the density structure locks in (HDBSCAN clusters still shifting). This is the SLT inflection point described in a different vocabulary.

## Usage

Single run:
```bash
python -m tangents.induction.run results/2026-03-15_18-55-33/gpt2_wiki_paragraph
```

All GPT-2/BERT runs in an experiment:
```bash
python -m tangents.induction.run --scan results/2026-03-15_18-55-33
```

Options:
- `--threshold 0.03` — lower induction score threshold (default 0.04)
- `--beta 2.0` — interaction energy beta for pair tracking (default 1.0)

Outputs go to `tangent_results/induction/<run_stem>/`.

## Input requirements

Requires Phase 1 run directories containing:
- `metrics.json` (results dict with pair_agreement, sinkhorn Fiedler data)
- `attentions.npz` (full attention matrices, all layers)
- `activations.npz` (L2-normed hidden states, for energy trajectory)

## Files

- `induction.py` — five measurement functions, no I/O
- `run.py` — CLI entry point, loads Phase 1 data, runs analysis, writes results
