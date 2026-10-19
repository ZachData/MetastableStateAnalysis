<!-- trackA_force/lit-C2.md -->
# C2 and A5 — literature scan before C2's inputs open and before A5 registers (2026-10-18)

Triggers (`CLAUDE.md` "Literature scans"): C2 was re-scoped at the 2026-10-18 check-in with "a
literature scan first", since "nobody reads this energy across training" is unverified
(`docs/ROADMAP.md` §8); A5 needs one before an entry lands in `claims/registry.json`. Scope:
(1) has anyone measured a Geshkovski-type interaction energy, or token clustering, along a real
model's training axis; (2) where the developmental literature puts the stages C2 sequences;
(3) what PolyPythias already says about seeds. Fifteen web searches and eight abstract pages
(2026-10-18). Not repeated: `lit-A.md`, `lit-A2.md`, `trackB_circuit/lit-B.md` (previous-token
heads at 512 → 1000, 2511.16893, 2404.07129), `p1_mstate_tracking/lit-1.md` (2509.23024 scooped
Phase 1's arc), `p10_cluster_function/lit-10.md` (2209.11895 at Pythia ~1200–2400, 2402.04362).

**Marks.** [A] the arXiv abstract page (or proceedings page) was fetched and read this session;
[S] a search engine's summary only. Nothing was read in full. **`WebFetch` of `arxiv.org/abs/…`
worked today**, which `docs/LITERATURE.md` §0 and `lit-A2.md` say is blocked; full texts may now
be readable (Parked below).

## 1. Verdict on "nobody reads this energy across training"

**Not refuted.** No paper found measures the interaction energy E_β, its layer-monotonicity, or
token clustering on real checkpoints along training. The nearest work measures other geometry
along training, or has an energy and training but no checkpoints:

| work | what it measures | bearing on C2 |
|---|---|---|
| Li et al., 2509.23024 [S] (NeurIPS 2025) | RankMe, αReQ of hidden states on Pythia 160m–12b and OLMo checkpoints: warmup collapse → entropy-seeking expansion → compression-seeking | already known (`lit-1.md`): Phase 1's rank arc is theirs. C2 must not lead with rank; the energy break is a different observable at a different step |
| Billa, 2602.15997 [A] (2026) | RankMe and probes along training, toy tasks and Pythia 160m–2.8b: representations collapse, recover, then capability rises; the collapse propagates top-down | the top-down collapse is the shape of Phase 1's step 8 → 16 late-layer collapse (layers 21–23 first); a precedent for "geometry before behaviour", not for an energy |
| Cheng et al., 2405.15471 [A] (ICLR 2025) | intrinsic dimension by layer on Pythia steps 512, 4000, 16000, 64000, 143000: ID grows and its peak sharpens with training; no onset step given | another global geometry series on the same axis; a candidate A3 row, not prior art for the energy |
| Bernas et al., ICML 2026 (PMLR 306) [A] | anisotropy of embeddings along Pythia, EuroBERT, SmolLM2 checkpoints, against the entropy of the distribution | anisotropy over training is someone's; C2's E_β depends on it (E_β is a moment of the Gram matrix), so C2 states E_β's anisotropy share (ROADMAP S1's residual-variance check is the same worry) |
| Diehl Martinez et al., 2410.11451 [A] | per-layer attention / MLP convergence to the final state across Pythia sizes; most layers settle in the first 20 % | beside; says nothing at ≤ 4000 steps |
| Isobe, Inoue, Imaizumi, 2605.07772 [A] (May 2026) | theory: a noisy mean-field transformer with only an FFN trained; an entropy-regularised interaction energy; tokens cluster, plateau, then escape near the final layers | the closest *theory* (already in `PUBLICATION_IDEAS.md`): training breaks the energy picture near the top. A predicted shape, no checkpoints, FFN-only |
| Herty, Liu, 2608.25055 [S]; Rigollet et al., 2512.01868 [S] | coupled token / parameter flows (theory); inner products concentrating along depth in trained models | known (`FUTURE_IDEAS.md`, `lit-A.md`); depth axis, not training |
| the user's LessWrong sequence ("Part 1: Do metastable token clusters exist in trained transformers?", "Transformers resist their own architecture") [S] | energy violations in every model tested (GPT-2 era, endpoints) | the only hit claiming non-monotone energy in trained models is the user's own (Blog 1, tier 0). C2 cites it and supersedes its "under random weights too" line (`math-1.md` §13.2) |

**What C2 can claim as new, one line each:** E_β's layer-monotonicity read along Pythia's
training axis; its break dated at 128–512 (`p1_mstate_tracking/status-1.md` "The developmental
arc"); its order against the previous-token, induction and OV repulsive windows. **Not new:** a
non-monotone geometric arc along training (2509.23024, 2602.15997), stagewise development
(below).

## 2. The stages C2's sequence is placed against

| stage | where | source |
|---|---|---|
| bigrams, then n-grams, then previous-token heads, then induction heads, then a fifth stage | 3M attention-only LM | Hoogland et al., 2402.02364 [S]; "Embryology of a language model" 2508.00331 [S] lists the five |
| induction heads form in a phase change with the in-context score | Pythia ~1200–2400 | Olsson et al., 2209.11895 (`lit-10.md` row 3) |
| previous-token behaviour 0.37 → 0.95 over 512 → 1000, ahead of induction | Pythia 160m, 410m | 2607.06621 (`lit-B.md` §1) |
| induction heads at ~1000, function-vector heads at ~16000 | 7 Pythia sizes | 2502.14010 [S] |
| head-level stages staggered by thousands of steps (rLLC) | 2-layer toy | 2410.02984 [S] |
| induction formation 10–20× (tokens) before the BOS attractor in DCLM models; Pythia 1b's BOS fraction a gradual ramp | Pythia 1b, OLMo 1b, OLMoE | Xu, 2606.02378 [A] |
| task components emerge at similar token counts across scale | Pythia 70m–2.8b | Tigges et al., 2407.10827 [A] |

**Bearing.** The energy break (128–512) falls where the LLC literature puts the n-gram stages,
before previous-token heads (512–1000) and induction ((512, 4000]). So C2's sequence is the
known developmental order with E_β laid on it; the reading C2 can add is whether E_β turns with
the n-gram stage (the A3 row from Phase 10's R6f already asks this) or with the warmup.

## 3. The confound the scan sharpens: the LR schedule

Pythia warms up its learning rate over the first 1 % of training (about step 1430), so the
energy break, previous-token and induction formation all fall inside warmup (ROADMAP §6 risks).
2509.23024 names its first phase after warmup. Every PolyPythias seed shares the schedule, so
**A5 on the seeds can show the break's step is not seed 0's; it cannot show it is not the
schedule's.** Separating the two needs a different schedule (S4, the toy model, can vary warmup;
nothing public on Pythia does).

## 4. PolyPythias (for A5)

- van der Wal et al., 2503.09543 [A] (ICLR 2025): 45 new runs (9 seeds × 5 sizes, 14m–410m),
  ~7k checkpoints; "highly consistent training dynamics" across seeds, with outlier runs that
  the abstract does not name. A seed changes init and data order (`lit-1d.md` §10 row 1).
- At abstract level, neither PolyPythias nor 2407.10827 reports a per-seed induction or
  phase-transition step on 410m. So A5's descriptive arm (break and induction onset together,
  per seed) is not already published, as far as abstracts show.
- **Outlier seeds are the risk for A5's count**: the repo already records that 410m seeds 3 and 4
  are outliers (loss spikes, ≥ 2 SD below the mean downstream; `lit-1d.md` §10 row 1, [H]), so
  K = 9 may be out of reach for reasons unrelated to the energy (`design-A5.md` D7; Parked 2).

## Parked

1. **Full texts are fetchable now** (why: §0's "blocked" was true on 2026-09-16 and 2026-10-16;
   cost: a re-check per paper; could change: any [S] row above, and `lit-1.md`'s open question of
   whether 2509.23024 normalises before RankMe). One line in `docs/LITERATURE.md` §0 when next
   edited.
2. *PolyPythias's outlier runs, by size and seed.* **Done, already in the repo** (#227 finding
   1): `p1d_cluster_ensemble/lit-1d.md` §10 row 1 names 410m seeds 3 and 4 (loss spikes, ≥ 2 SD
   below the mean downstream). A5's D7 is decided from it (`design-A5.md`).
3. **2602.15997's Pythia checkpoints and its top-down collapse** (why: may date Phase 1's step
   8 → 16 collapse as a known warmup event; cost: one read; could change: C2's line on the early
   collapse). Carry to A3.
