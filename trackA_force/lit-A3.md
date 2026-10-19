<!-- trackA_force/lit-A3.md -->
# A3 — literature scan before the design freezes (2026-10-19)

Trigger (`CLAUDE.md` "Literature scans"): A3 opens (`docs/ROADMAP.md` §9 row 6). Narrow on purpose:
`lit-C2.md` (2026-10-18) already places the developmental stages and the LR confound on Pythia's
axis, and is not repeated. This scan covers what A3 adds over C2's: the n-gram milestones (triage
M2), the weights-only rows (`FUTURE_IDEAS.md` D3, D4), and C2's Parked 3, carried to A3.
Three web searches, two arXiv pages.

**Marks.** [A] abstract page read; [H] the arXiv HTML full text read through a fetch summary;
[S] a search engine's summary only.

## 1. Rows A3 lays down, and who has read them on Pythia

| row | prior work on Pythia's axis | bearing on A3 |
|---|---|---|
| n-gram milestones (M2) | Belrose et al., 2402.04362 [S] (already `lit-10.md` row 3): Pythia scored by cross-entropy on max-entropy n-gram samples from the Pile; some unigram in-context learning by step 1000, more after 8000; bigram after 8000 | the published milestones are *later* than the energy break (128–512). Their measure is loss on synthetic sequences, not KL of the model to n-gram predictors on real text, so A3's M2 row is a different observable; Belrose is quoted beside it, not replaced by it |
| loss-space trajectory | Kishino et al., 2505.15353 [S] (Findings ACL 2026): KL between Pythia checkpoints in log-likelihood space; behaviour stabilises early while weights keep drifting | a candidate row beside M2, not in the family (§2 of the design) |
| ΔW stable rank (D3) | 2306.07042 [S]: stable rank of `W_t − W_0` grows gradually, ends low against the init spectrum (vision and language models, not Pythia's checkpoints); 2510.06954 [S]: a stage where `softmax(QKᵀ)` barely moves, then QK moves | method only; nobody found reads it on Pythia's checkpoint axis. 2510.06954's "QK stagnant, then moves" is a prediction the D3 row can show or not on `W_Q`, `W_K` |
| weight ESD α (D4) | Martin–Mahoney HTSR (1901.08276 [S]); AlphaPruning 2410.10912 [S] applies it per transformer block to LLM endpoints | no per-checkpoint Pythia ESD found. Fits per matrix (QKV, O, MLP in, MLP out), not per block, since the blocks mix |
| geometry collapse and recovery | Billa, 2602.15997 [H] (C2 Parked 3): Pythia-deduped 160m, 410m (154 checkpoints each), 2.8b; task RankMe drops 50–90 % from step 0 to steps 32–256, then partially recovers by about 1000 (the paper's windows disagree: 0–128 in App. E, 0–512 in Fig. 8); the logical-deduction floor on 410m at step 32. Top-down layer order is from its toy transformers only, **not** measured on Pythia; warmup not discussed | the collapse window (32–256) sits **on** the energy break (128–512) and on Phase 1's 8 → 16 late-layer collapse; a row in A3, quoted from the paper, with C2's line that the early collapse may be a warmup event still open (Pythia warms up to about step 1430, so 32–256 is deep in it) |

## 2. Verdict

Nothing found puts these rows on one axis. Each row exists alone (rank, ID, anisotropy, n-gram
loss, induction) somewhere; A3's contribution is the joint table and the stated alignment, and,
new to the literature as far as abstracts show, the energy row and the weights-only rows on
Pythia's checkpoints.

## Parked

1. **Belrose's Pythia milestone figure** (why: the [S] numbers are the search engine's; cost: one
   full-text read; could change: whether M2 is still worth a unit, if their curve already has a
   unigram step at 128–512). Read in unit 1's design review or before unit 4 opens.
2. **2602.15997's per-checkpoint RankMe on 410m** (why: would give a measured 410m row instead of
   a quoted window; cost: their code or tables, if released; could change: the row's location).
