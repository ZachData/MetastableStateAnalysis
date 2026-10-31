<!-- trackC_writeup/lit-C2p-fulltext.md -->
# The draft's citations against the full texts in `docs/readings/` (2026-10-30)

The user populated `docs/readings/` (PDFs and arXiv LaTeX sources, untracked). Every claim
`paper/main.tex` (at `dffcb83`) attributes to a paper was checked against that paper's own
source, mostly by targeted reads of the LaTeX (section, theorem, caption), not cover to cover.
Earlier rows were [A]/[S] in `lit-C2-unit1.md` §3, §7 and `lit-C5.md` §1.

**Marks.** [T] the LaTeX source read at the lines named; [P] the PDF text or a figure read by eye;
[R] a Hugging Face repository queried (`git ls-remote`, the models API), 2026-10-30.

## 1. Claims that are wrong or need narrowing

| # | draft (`main.tex`) | the source says | fix |
|---|---|---|---|
| W1 | §2.1 line 56: "trained transformers satisfy the symmetry and tying conditions nowhere exactly \citep[§10]{pendharkar2026}" | §10 ("Approximate Head Orthogonality") says **Condition 6, orthogonal head subspaces**, is never exactly satisfied in trained transformers (principal angles 70–85°) [P]. Tying (Condition 4) is "the main idealization of our framework" (Remark 5, §2.3: QK and V are independent in general transformers). Nothing on symmetry (Condition 3) in trained models | cite Remark 5 for tying; drop the symmetry clause or state it as ours; §10 only for orthogonality |
| W2 | §2.1: "a single head's own energy can fall through cross-head terms [Theorem 17]" | Theorem 17 *proves* per-head monotonicity on the sphere under Conditions 3, 4, 6, 7 and 8 (radial dominance), on [0, T_crit); the failure without Condition 8 is the "radial shadow obstruction" of §4.2 and the contributions list [P] | "per head, monotonicity on the sphere needs a further radial-dominance condition, because cross-head radial terms enter it (§4.2, Theorem 17)" |
| W3 | §2.1: "No result covers MLP blocks, values untied from the scores, weights that change from layer to layer, or discrete layers"; "none says a randomly initialised one's should" | Koubbi, Geshkovski, Rigollet, **2604.01978** [T]: attention-only, multi-head, **weights resampled independently across layers and heads "as at initialization"**, discrete time on the sphere, with a homogenized limit. Its Theorem (small β, `thm:clustering_small_beta`): the mean overlap m(t) = (1/n²)Σᵢⱼ⟨xᵢ,xⱼ⟩, **which is the draft's ‖m‖²**, follows u̇ = u(1−u) with high probability, i.e. rises in depth; the simplex theorem gives the overlap's ODE at β²n/d = O(1). Fedorov et al. 2601.21942 (cited there) treat values that vary randomly by layer. Not monotonicity of E_β, and no MLP (a remark says it adds drift or common noise) | narrow the sentence: per-layer random weights (at initialisation, attention only) are covered for the mean overlap, not for E_β; cite 2604.01978 beside the step-0 fact (§2.1, §4 "Beside the count") |
| W4 | §8: "decoders' anisotropy peaks in their middle layers \citep{razzhigaev2023}; §4.1's profile of ‖m‖² … is compatible with such a peak forming at the break" | Razzhigaev's anisotropy is σ₁²/Σσᵢ² of the **centred** embedding matrix (§3) [T]: the top principal direction's variance share, which excludes the mean by construction. It is not ‖m‖² | say their measure is centred; the comparison is with the centred cloud, or drop "compatible" |
| W5 | §8: "Contextual representations are anisotropic in every layer \citep{ethayarajh2019}" | "anisotropic in all **non-input** layers" (ELMo's input layer the exception) [T] | "every non-input layer" |
| W6 | §8: Gao et al. "the regular polygon loses stability to period-two motion, chaos and flipping clusters" | true for d = 2, the circle (abstract) [T] | add "on the circle (d = 2)" |

Minor, wording only: E6's "Our quantity differs from that loss" (appendix D): Belrose et al. also
report **D_KL(n-gram model ‖ Pythia)**, the same direction as E6, in its appendix figures, Fig. 13 among them (on sequences
sampled from the n-gram model, N = 4096) [T, P]; cite that panel, not only the loss. PolyPythias
calls the second seed factor "batch composition", not "data order" (footnote, §2) [T].

## 2. Confirmed (no change)

geshkovski2023 (Prop. 3.4, §3.2 with V = −I, §3.4, §3.4.1's V-weighted energy and Z-weighted
metric; E11's "one cluster at a long horizon" is the theorem credited to Markdahl and Criscitiello et al.
in "The high-dimensional case", d ≥ 3, every β); pendharkar2026 Theorems 11–12; gao2026 Sec. V B; biderman2023 (Table 1 rates,
warmup 0.01, cosine to 0.1, checkpoint grid, parallel attention, rotary 0.25); vanderwal2025
(9 seeds × 5 sizes, seeds 3 and 4 the 410m outliers with loss spikes, consistent dynamics);
belrose2024 (trough 2⁶–2⁸, rise to 2¹⁰: the Pythia n-gram loss figure's caption); hoogland2024 (LM1–LM5 as worded; stages
bounded by LLC critical points); olsson2022; tigges2024; li2025 (RankMe, αReQ on the last layer's
last token; first phase "coinciding with learning rate ramp-up"); billa2026 (its released
`data/fig6_pythia_410m.csv`: Pile RankMe 277 at 0, 18 at 32, 34 at 256, 209 at 1000, 348 at 2000);
cheng2024 (Pythia-6.9B, steps 512, 4000, 16 000, 64 000, final); queipo2025 (steps 1, 1k, 2k, 4k,
8k, 10k, 20k, 30k, 143k; one GSM8K prompt; p₁ = σ₁²/‖X‖²_F, X not centred); bernas2026 (Pythia
from 10k–22k); benartzy2026 (PC1 share; Pythia-1.4B from step 1000, so no step in 1–1000);
godey2024; timkey2021 (1–5 dimensions); sun2024; xiao2023; dong2021; noci2022; liu2019;
kalra2024; martin2019; vovk2019 (κp^{κ−1}); yin2025 (induction heads evolve into FV heads);
chen2023; isobe2026 (abstract as worded); karamcheti2021 (5 seeds, 610 checkpoints).
**The novelty sentence stands:** no source here reads a step in (1, 1000) on Pythia's
representations except billa2026 (RankMe, which §8 cites) and belrose2024 (n-gram loss).

## 3. What the draft missed

| # | what | source | bearing |
|---|---|---|---|
| M1 | **A released Pythia run that differs only in warmup.** `EleutherAI/pythia-70m-warmup01` and `pythia-14m-warmup01` [R]: GPT-NeoX 6 × 512 (70m's config: parallel residual, rotary 0.25), 156 branches, steps 0, 1, 2, …, 512, 1000, then every 1000 to 143 000. Belrose et al. trained them "with an extended learning rate warmup period of 14,300 steps to isolate the effect of LR warmup" (Pythia experiments) [T]. Peak rate and data order are not documented on the Hub | 2402.04362 §5; HF | §7 says the separating runs are other rates, other suites, or continued training. This is a fourth, on 410m's family: same data, size and architecture as the 70m the draft already reads, warmup ×10. Predicted bands on it for 70m's dates (break 256, centred 1000): step 256 / 1000; cumulative dose ×√10 ≈ 810 / 3200 (grid 1000 / 4000); instantaneous ×10 ≈ 2560 / 10 000. The grid separates all three. Not run |
| M2 | **The n-gram minima move with the warmup on Pythia.** Belrose Fig. 13 (by eye): 70M D_KL(1-gram ‖ Pythia) minimum ≈ 2⁷ (fast) → 2⁸ (slow); 2-gram ≈ 2⁸ → 2⁹–2¹⁰; 14M 1–2 octaves later as well [P]. Their text: the warmup "did not consistently affect their overall shape" (shape, not date) | 2402.04362 App. Fig. 13 | E6u, E6b's analogues are schedule-sensitive on 70m, shifted 1–2 octaves, nearest the cumulative dose (√10 ≈ 1.7 octaves). Supports §5's "inside a linear ramp 'within one grid step' includes 'driven by the same ramp'" with outside evidence. A figure read, not a number |
| M3 | Hoogland et al. trained **with no learning-rate scheduler** (App., training details) [T] | 2402.02364 | the stage order §2.2 quotes arises there without any warmup: in that model the order is not the schedule's. One clause in §2.2 or §8 "Warmup" |
| M4 | Tigges et al.: "most models' induction heads emerge soon after they have seen 2×10⁹ tokens" (component-emergence results) [T], about step 950 at 2 097 152 tokens a step | 2407.10827 | an outside date for E3 (1000), on Pythia; a positive-control-like agreement for the locator |
| M5 | Precedents on the warmup confound. Olsson et al. (Model Analysis, small models): the phase change at 1–3B tokens overlaps the warmup (first 1.5e9 tokens), and they say the window does not "correspond to a scheduled change" [P]. Chen et al. (appendix `sec:scaling_xaxis`): transition well after warmup (10k steps), and re-plotted against distance from initialisation and path length [T] | 2209.11895; 2309.07311 | §8 "Warmup": the confound is known; Chen's alternative clocks are cheap here (E7 is already ‖W − W₀‖/‖W₀‖) |
| M6 | PolyPythias' own phases: first training-map transition "around step 10³"; "SSA roughly follows the learning rate schedule: peaking around step 2000 (end of the warm-up phase)" (representations and training-map sections) [T] | 2503.09543 | the break (256–512) precedes PolyPythias' first phase; a published representation statistic on the same seeds already tracks the schedule |
| M7 | Isobe et al.: escape from clustering "near the final layers", produced by "a single gradient-descent update" [T] | 2605.07772 | contrast with appendix C: drops sit mid-stack (transitions 10–12 at 512) and are absent through 128. A sentence in §8, not a test (their model is noisy, FFN-only trained) |
| M8 | Ethayarajh's and Timkey's anisotropy is the expected cosine of random pairs [T]; for unit tokens the mean off-diagonal cosine is (n‖m‖² − 1)/(n − 1), exactly affine in ‖m‖² | 1909.00512; 2109.04404 | makes §2.1's "‖m‖², a standard measure of anisotropy" exact, with a citation; a one-line sympy check if used |
| M9 | Godey et al. ("Queries and keys: training dynamics" and the section before it): on untrained Transformers, pushing a common drift (anisotropy) sharpens self-attention; Q/K distributions along training (MultiBERTs) [T] | 2401.12143 | a mechanism beside §4.1 (the mean and attention sharpness), not read here |

Not cited by the draft and not bearing on it: 1411.5259 (SigClust), 1904.02447 (`1904.02447.md`),
2412.09080, 2605.09213, clusta_hennig. `2512.09080v3_GlobalMinimumCut.pdf` is still the unrelated
min-cut paper (STATE Blocked 13); `arXiv-2412.09080v3` is the right source and is present.

## Parked

1. **Run the draft's two counts on `pythia-70m-warmup01`** (why: M1, the one released run on the
   draft's own family whose warmup differs (its other settings unpublished); cost: 70m forward passes at ~10 checkpoints,
   under an hour on the GPU, plus a design with frozen bands before any output; could change: §7's
   "nothing separates the schedule from training" and the abstract's last claims). *Carried to C6*
   (ROADMAP §8; opened 2026-10-30, the user's).
2. **Confirm `warmup01`'s schedule** (why: the Hub has no model card; only Belrose's text gives
   14 300 steps; cost: one read of the EleutherAI config or a log, if published; could change: M1's
   bands). *Carried to C6*, before its design freezes.

## In `paper/` (this unit)

W1–W6 fixed; M1, M2 in §1 and §7; M3, M4 in §2.2; M5, M6 and M7 in §8; M8 in §2.1 (sympy
`tools/math_checks/c2p_mean_cosine.py`); 2604.01978 cited in §2.1 and §8 (`refs.bib` `koubbi2026`).
M9 not taken (nothing here reads it). `trackC_writeup/status-C.md` "C2p readings audit (2026-10-30)".
