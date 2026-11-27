<!-- trackC_writeup/lit-C9.md -->
# C9's scan: does the developmental order hold across models and seeds? (2026-11-25)

The scan for C9 opening (CLAUDE.md "Literature scans"), before `design-C9.md` freezes. **Depth:
abstracts and search snippets only, plus what earlier scans already read** (`trackA_force/lit-A3.md`
for the timeline's rows; `lit-C4*.md` for the schedule). No full text read. Each line says what it
changes in the design.

## 1. What is read across sizes and seeds

| paper | what it reports (abstract level) | what it changes here |
|---|---|---|
| Tigges, Hanna, Yu, Biderman 2407.10827 (NeurIPS 2024) | circuits in Pythia 70m–2.8b across 300B tokens; functional components (induction among them) emerge at similar **token counts** across sizes; the same algorithm can move between heads over training | head identity is not stable, so U2 names heads per model by one fixed rule (D2), not by index; and sizes are compared in tokens beside steps |
| van der Wal, Lesci et al. 2503.09543 (PolyPythias, ICLR 2025) | 9 seeds × 5 sizes (14m–410m); language modelling largely stable across seeds; outlier runs identified; training phases studied (an early version: a critical phase between 10⁴ and 10⁵ steps, not confirmed in the final text) | the 160m seeds (U3) are the published resource; its outlier seeds are flagged as A5 flagged 410m's 3 and 4 (whether 160m has named outliers: **Parked 1**) |
| 2509.22831 ("Toward a theory of generalizability in LLM mechanistic interpretability research") | 1-back (previous-token) attention on the PolyPythias 14m–410m seeds at 16 checkpoints; larger models' seeds show earlier onsets, steeper slopes and higher peaks | a prior for E4 on 160m against 410m (earlier on the larger); U2's E4 on the harness arms (70m) is expected later than 410m's, not equal |
| Aoyama, Wilcox, Schneider 2511.16893 (ICML 2026) | induction heads' emergence point predicted by an equation in batch size and context size, independent of model size (small GPT-2-style models, natural and synthetic data) | GPT-2 small (Mistral: batch 512 × 1024 tokens) and Pythia (1024 × 2048) differ in both, so their induction dates in steps are not expected equal; the design reports steps and tokens and reads order within a model, not dates across models (D4) |
| Karamcheti et al. 2021 (Mistral, via 2503.09543) | 10 GPT-2 runs (124M and 355M) with ~600 checkpoints each | the 5 small seeds C2 3c used are the resource; nothing new |
| ACL Anthology 2026.findings-emnlp.2052 (title not retrieved; one search snippet) | a mature circuit can hide an earlier transient configuration; similar early-to-late reversals in two GPT-2 models and OLMo | the locator dates a row's first move only (A3 §4); a reversal is described, not dated (unchanged) |

## 2. Verdict

Nothing scooped: no paper found reads the interaction energy's break, or its mean term, across
models, and none places a previous-token / induction / OV-phase order against it. Two inputs to
the design: (i) heads must be chosen per model by a rule (2407.10827); (ii) dates across models
are compared in tokens or not at all (2511.16893), so C9 reads **orders within each model** and
reports dates beside.

## Parked

1. **PolyPythias' full text for 160m** (why: which 160m seeds, if any, it names as outliers, and
   its training-phase section; cost: one reading; could change: which seeds U3 flags). Read
   before U3's records are read. **Done 2026-11-26** (U3, before any seed 1–9 record; arXiv HTML
   v2, §1–5 verbatim, read by a fetch agent; §6–7, tables 2–3 and the appendices only as the
   agent's summary). §3 names outliers only at 410m: "We (remarkably) find only two such
   combinations: 410 M seed 3 and 4". §5: "The only exception is 410 M, for which we observe
   variability caused by outlier seeds." No 160m seed is named, so **U3 flags none**. The phases
   are given across sizes ("an initial learning phase between steps 10^3–10^4 and a critical
   learning phase between steps 10^4–10^5"). They begin at 10^3, at or after 160m seed 0's
   two dates (256, 1000), so nothing moves. Table 2's 160m transition steps (about 2k, 18k, 61k, 100k, from the
   summary) are unverified and not used.
2. **2509.22831's full text** (why: its 1-back onset steps on 160m and 70m seeds; cost: one
   reading; could change: E4's lean on the harness arms). Read before U2's rows are read.
   **Done 2026-11-26** (U2b, before any U2b record; arXiv HTML v1, read by a fetch agent; quotes
   checked only against its report). The score is §4.1's Eq. 1, each head's mean attention from
   i to i − 1 on the Natural Stories corpus. The figures take **the max over heads** at each
   checkpoint, which is D2 (a)'s E4 (on other text). Onset is the first step where Δ(max) / Δ log
   step passes a threshold (averaged over 0.01–0.3). **No onset step is given for 70m, 160m or
   410m**: only a mixed-model slope, "larger models … earlier onsets" (β −0.18 on log
   parameters, units not stated), and on all models pooled "an expected onset … between 512 and
   2000 steps". It does not compare with induction. **What it changes:** E4's lean stays as
   written (70m's previous-token onset no earlier than 410m's). There is no number to read
   against, and 512–2000 covers both. No rule moves.
