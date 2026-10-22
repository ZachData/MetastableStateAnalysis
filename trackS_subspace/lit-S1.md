<!-- trackS_subspace/lit-S1.md -->
# S1, the shared-subspace atlas: literature scan before its design freezes (2026-10-20)

Trigger (`CLAUDE.md` "Literature scans"): Track S opens with S1 (`docs/ROADMAP.md` §7a, order
§9 row 6). Scope: (1) has anyone measured subspaces shared across many heads, from the weights,
along a model's training axis; (2) what the head-level spectral literature already says about
OV / QK eigenvalues (copying, suppression, rotation) and their emergence; (3) how a mean of
subspaces is read and what its null is; (4) known global directions (outlier dimensions, massive
activations, anisotropy) that would read as "shared" for a trivial reason. Twelve web searches,
eight abstract pages (2026-10-20). Not repeated: `trackB_circuit/lit-B.md` (2607.06621 on
previous-token heads), `p2b_imaginary/lit-2b.md` and `p2_eigenspectra/lit-2.md` (complex
fraction, 2607.06621 as positional scheme → spectral algebra), `p6_subspace/lit-6.md`,
`trackA_force/lit-C2.md` (anisotropy along Pythia, 2405.15471, Bernas et al.).

**Marks.** [A] the arXiv abstract page was fetched and read this session; [S] a search engine's
summary only. Nothing was read in full.

## 1. Verdict on "nobody maps shared head subspaces along training"

**Not refuted, with one near neighbour.** No paper found builds a population-level map of
directions shared across all heads and layers from the weights, against a random-subspace null,
on a model's checkpoints. The nearest:

| work | what it measures | bearing on S1 |
|---|---|---|
| Shaikh et al., 2603.13314 [A] (Mar 2026) | a head's Q, K, V *activations* are linear combinations of 2–5 peers, mostly in the same layer; absent at init, grows over OLMo-2 pretraining, tied to growing intra-layer alignment of **key** projection subspaces | the nearest neighbour: intra-layer key sharing grows with training. S1's step-8000 preflight read agrees (every layer's key pool is above the layer-shuffle null). S1 differs in reading weights, all seven subspace families, across layers, on Pythia's early steps (0–4000), with a Haar null. Cite it for the key row; do not lead with "keys are shared" |
| Merullo, Eickhoff, Pavlick, 2406.09519 [A] (NeurIPS 2024) | low-rank write / read channels between pairs of heads from weight SVDs (GPT-2 and one other LM) | pairwise channels, endpoints only; B2's C6 already used this view. S1 is the population view of the same singular subspaces |
| Yamagiwa, Takase, Shimodaira, 2601.10266 [A] (Jan 2026) | the projection kernel (principal-angle overlap) between heads' weight subspaces, with a reference distribution from random orthogonal subspaces; GPT-2 small, endpoint | S1's mean pairwise overlap is their kernel averaged over a pool (tr P̄² identity, `tools/math_checks/s1_projector_null.py`); their random-orthogonal reference is S1's Haar null. Prior art for the null, not for the atlas |
| Cordonnier, Loukas, Jaggi, 2006.16362 [A] | trained heads' key / query projections are largely redundant; compared through the product W_Q W_Kᵀ, since per-head rotations hide it | supports reading QK through M_0's singular subspaces, not W_Q or W_K alone (S1 does) |
| Ahmad, Joshi, Modi, 2511.20273 [A] (NeurIPS 2025) | singular directions inside one head carry separate subfunctions (IOI, Greater-Than) | a head is not the unit; within-head, not shared across heads |
| Yunis et al., 2408.11804 [A]; workshop version [S] | top singular vectors stabilise early in training, top singular values grow; [S] neighbouring layers' top singular vectors align as they grow | cross-layer alignment of top singular vectors along training is someone's (at the level of whole weight matrices). S1's cross-layer pool is per head, by subspace family, against a null; cite as the matrix-level precedent |
| Liu, 2604.22778 [A] (Apr 2026) | full SVDs of every weight during pretraining (30M–285M custom models; GPT-2, Pythia for pruning): stable-rank compression waves; Q/K carry the depth-dependent dynamics, V/O compress uniformly | spectra per matrix, not shared subspaces; its Q/K–V asymmetry is a prediction S1's families can show or not (QK shares more than OV at 8000 in the preflight read) |

## 2. Head-level OV / QK spectra and their emergence

| work | finding | bearing |
|---|---|---|
| Elhage et al. 2021 (framework) [S]; 2310.04625 copy suppression [S] | OV eigenvalue score Σλ / Σ|λ|: positive = copying, negative = suppression (GPT-2 L10H7); complex eigenvalues drive the score to 0 | S1's `ov_pos` / `ov_neg` families are where copying and suppression live; a shared negative subspace is the population form of "suppression", which the particle picture reads as repulsion |
| Li Hengyu, 2607.06621 [A] | QK of previous-token / induction heads: rotational under RoPE; on Pythia checkpoints every head starts at the Ginibre null, and the rotational signature emerges *with* the behaviour | per head and QK only; S1's `ov_cplx` family asks the same of OV at the population level. Its "starts at the null" is S1's step-0 check |
| Li Hengyu, 2607.09279 [A] | non-normality (pseudospectra, Kreiss) of the attention propagator on GPT-2, Pythia-410m, Llama-3-8B: a routing minority carries transient reserve that tracks previous-token function; checkpoint censuses date the sink / routing organisation after circuit formation | already in `PUBLICATION_IDEAS.md` as a positioning caution; for S1, a reminder that eigenvalues of a non-normal OV miss transient growth (the S1 radial-sign reading says what V does to a direction, not how it amplifies) |
| antonym-heads post (LessWrong) [S] | top-scoring heads in 13 models: the largest eigenvalues are negative real | a head-level precedent for strong `ov_neg` content in trained heads |

## 3. Means of subspaces and their null

The average of orthogonal projectors (the extrinsic or chordal mean) is the standard summary of
a set of subspaces: its eigenvectors give a central subspace and thresholding its eigenvalues
gives a dimension, for subspaces of different dimensions too (Santamaría, Scharf et al. on
order selection by the average projector [S]; Ramírez, Santamaría, Scharf, book chapter [S]).
No source found sets that threshold by a simulated random-subspace null (parallel analysis on
projectors); free-probability edges for sums of fixed-rank random projections are not tabulated
in what was found [S]. **S1 draws the null by simulation** (independent Haar subspaces of the
heads' own dimensions), which 2601.10266 also uses as a reference.

## 4. Directions that would read as "shared" for a trivial reason

| source | what | how S1 handles it |
|---|---|---|
| massive activations, 2026 [S] (Pythia dims 130, 357, 966 named) | a few residual coordinates of order 10³ | the preflight read at 8000: the top shared direction's largest coordinate holds ≤ 1.8 % of it in every family, so the shared space is not those coordinates; S1 records each shared direction's top coordinates beside |
| Machina, Mercer, NAACL 2024 [S]; Bernas et al. (`lit-C2.md`) | anisotropy along Pythia's training; large Pythias end isotropic | the ROADMAP's beside check: the shared directions' share of the residual stream's top-variance directions |
| LN gain (`b2a.ln1`) | a per-layer diagonal every head of the layer reads through | S1's primary frame is raw (as stored); the gain would make a layer's read spaces share its large-gain coordinates. Folded frame beside |

## What the scan changes in the design

| item | change |
|---|---|
| 2603.13314 | the key row is expected to share within layers; S1 does not lead with it, and states it as agreeing with an OLMo-2 result |
| 2601.10266 | S1's per-pool scalar is their projection kernel averaged; cite, do not rename |
| 2604.22778 | QK vs OV asymmetry stated as a known shape S1 can confirm or not, not as a finding |
| §3 | the threshold is a simulated Haar null per pool shape, max-eigenvalue based (family-wise) |

## Parked

1. **2603.13314's full text** (why: which OLMo-2 steps, and whether value subspaces stay near
   orthogonal, which S1's `ov_read` row would contradict at 8000; cost: one read; could change:
   how S1's key and value rows are stated against it). Carry to S1's close. **At S1's close
   (2026-10-21): carried to C2's draft** (`status-S.md` "S1 on one page"). **Done at C2 unit 1
   (2026-10-22):** weight subspaces on OLMo-2 7B from step 100; keys rise most, values "largely
   flat"; appendix A states S1's `ov_read` beside it as a different statistic, not a contradiction
   (`trackC_writeup/lit-C2-unit1.md` §2).
2. **The massive-activation dims in 410m specifically** (why: the [S] source names "Pythia" with
   no size; cost: one forward pass, beside S1's residual check; could change: nothing in the
   count, only the beside column). Carry to S1 unit 3. **Done 2026-10-21:** 357, 130, 966 from
   block 8 (278, 335, 914 at 1–4), and at 143000 the write, negative and complex shared spaces
   are partly these coordinates, so §4's "≤ 1.8 %" holds for step 8000's top direction only
   (`status-S.md` "S1 unit 3").
