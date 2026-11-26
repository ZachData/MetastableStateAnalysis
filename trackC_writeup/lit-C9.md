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
   before U3's records are read.
2. **2509.22831's full text** (why: its 1-back onset steps on 160m and 70m seeds; cost: one
   reading; could change: E4's lean on the harness arms). Read before U2's rows are read.
