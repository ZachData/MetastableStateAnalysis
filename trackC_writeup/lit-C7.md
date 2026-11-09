<!-- trackC_writeup/lit-C7.md -->
# C7 (i), Isobe et al.'s depth profile on Pythia-410m: literature scan before its design freezes (2026-11-09)

Trigger (`CLAUDE.md` "Literature scans"): C7 (i) opens (STATE Blocked 153, revised 2026-11-09).
Scope: (1) what 2605.07772 predicts, exactly, and which of its quantities exist on a pretrained
model; (2) whether anyone has looked for its three phases (clustering, a plateau, a terminal
escape) in a pretrained language model; (3) which "spectral gap" the escape is tied to, and who
has measured that object per layer. Read: 2605.07772 v2 (2026-10-16, NeurIPS 2026) in full from
its PDF (text extracted locally; the HTML is a 404); six web searches. Not repeated:
`trackA_force/lit-C2.md` (its row on 2605.07772, from the abstract), `trackC_writeup/lit-C2p-fulltext.md`
M7, `trackS_subspace/lit-S4.md` (2506.13688).

**Marks.** [F] full text read this session; [A] abstract page; [S] a search engine's summary only.

## 1. What 2605.07772 says, in the parts a measurement can use [F]

| item | the paper | where |
|---|---|---|
| model | noisy S-USA: `dXᵢ = P⊥(u_W(t)(Xᵢ) + (1/N)Σⱼ e^{⟨Xᵢ,AXⱼ⟩} AXⱼ) dt + √(2ε) dBᵢ` on S^{d−1}; A = QᵀK = V symmetric, **shared across depth**; β = 1; only a parameter-linear FFN u_W = Wσ(x) trained, L² penalty (≈ weight decay) | (2.1), (2.3), Assumption 3.3 |
| energy | E_{ε,A}(µ) = ε Ent(µ) − ½∬e^{⟨x,Ay⟩}dµdµ; clustering **lowers** it; Δ = E − min E is the "distance to the clustered state" | (3.1), Figure 1 |
| three phases | 1: attention lowers E in the first layers; 2: a plateau near the clustered state through the interior; 3: in the final layers **the FFN dominates attention** and moves µ toward lower loss, raising Δ | §3 "Phase 1–3" |
| the gap | **Assumption 3.1: A positive definite, λ₁ > λ₂ (simple top eigenvalue).** The Hessian at the minimiser is bounded below by (ε/2)(1 − λ₂/λ₁); the turnpike rate a depends on A "through 1 − λ₂/λ₁ and λ₁e^{−λ₁}" | Prop. 3.2, Remark 3.10, A.53 |
| the escape's rate | after one GD step, Δ near T is bounded below by ∝ exp(−(T−t) R_ρ(g_ℓ)), R_ρ the **Rayleigh quotient of the loss gradient** at the clustered state, not the gap itself | Prop. 3.5 |
| minimisers | Δ(µ*_t) ≤ C(e^{−at} + e^{−a(T−t)}): small except in two boundary layers of **T-independent width**; W*_t exponentially small away from the terminal layer | Theorem 3.9, §3.3 |
| needs | the input distribution already biased toward one well (Assumption 3.6 excludes near-uniform inputs); under global random initialisation 11 of 20 runs show the profile | Assumption 3.6, B.3 |
| weak attention | E ≈ ε Ent − (β/2)⟨m, Am⟩ to first order: **only the mean m interacts** | Remark 3.11 |
| experiments | particles only: N = 64, d = 2, T = 80, A = diag(1, λ₂), start in a cap; smaller λ₂ gives "a deeper and more persistent interior plateau" and a sharper terminal lift; the fitted terminal rate a tracks R_ρ; "neither verify the quantitative hypotheses … nor constitute measurements on pretrained Transformers" | §4, Figure 4 |
| **the criterion** | "one fixed three-phase criterion applied without retuning": with R_E = max E − min E and E_pl = median of E on [0.4T, 0.6T]: (E(0) − E_pl)/R_E ≥ 0.1, SD_{[0.4T,0.6T]}E / R_E ≤ 0.05, (E(T) − E_pl)/R_E ≥ 0.1; t_esc the first time in [0.5T, T] reaching 10 % of the plateau-to-terminal rise; a fitted on [0.7T, T] from E − E_pl ≈ Ce^{−a(T−t)} | Appendix B.3 |
| pointers to pretrained models | interior blocks removable with little damage, early and final layers fragile (Gromov et al. 2025; Lad et al. 2025); sharp final-layer shifts in multilingual models (Wendler et al. 2024; Shibata et al. 2026). "We do not claim that our FFN-only noisy mean-field model directly explains these phenomena" | §5 |

**Two corrections to how C7 (i) was worded** (ROADMAP §8 C7, STATE Blocked 153; `/challenge-pr` on
#340, finding 3):

1. **The spectral gap is A's, the weight matrix QᵀK, not the attention pattern's.** The row said
   "attention's spectral gap (`attentions.npz`)". The paper's gap is 1 − λ₂/λ₁ of the symmetric
   interaction matrix (Assumption 3.1). It is read off the checkpoints' weights, with no forward
   pass. On a causal model the pattern's own eigen-gap is empty anyway. The pattern P is lower
   triangular, so its eigenvalues are its diagonal (each token's self-weight), with P₁₁ = 1. The
   pattern's singular-value gap (2410.07799's object, below) can be read from `attentions.npz` as
   a different mechanism, rank collapse, and goes beside.
2. **The escape's rate is not the gap.** In the paper the gap sets the plateau's persistence and
   depth (rate a; Figure 4's λ₂ sweep). The terminal lift's rate is R_ρ(g_ℓ), a property of the
   loss at the clustered state, and R_ρ has no analogue we can read. So C7 (i) can ask two
   things: whether the gap goes with the plateau, and whether the FFN takes over in the final
   blocks. It cannot ask whether the gap sets the escape's rate.

## 2. Pretrained models: has anyone looked for the three phases?

**No direct test found.** Nearest work:

| work | what | bearing |
|---|---|---|
| Krishna Kumar, 2511.21635 (2025) [S] | ViT-S/B/L on ImageNet: a "Cliff-Plateau-Climb" three-phase profile of token similarity by depth; plateau 6, 12, 14 layers, growing with depth; neural collapse in the climb | the same shape in another modality, measured on similarity, not an energy; plateau length grows with depth, as Theorem 3.9 says (boundary widths fixed) |
| Machina & Mercer, NAACL 2024 [S] | large Pythia models isotropic; the final LayerNorm optimised for isotropy; anisotropy not monotone along training (21 checkpoints) | the post-final-LN state (our state 24) is a readout, not dynamics: it is beside, not in the profile |
| Timkey & van Schijndel 2021 [S] | GPT-2's final layer: any two words almost perfectly similar (rogue dimensions) | the final block can *raise* similarity, the opposite of an escape; a reason the criterion's terminal half may fail |
| Skean et al., 2502.02013 (2025) [S] | intermediate layers beat final ones downstream, Pythia-410m included | context for "the interior is the plateau" |
| Gopalani & Hu, 2506.13688 (NeurIPS 2025) [S, via `lit-S4.md`] | Pythia's last-layer mean cosine above 0.9 within a few steps, "followed by a decrease around step 1000" | a dated terminal fall along training: if the criterion's terminal half ever passes on 410m, this says when to look |
| the user's Blog 1 (LessWrong) | metastable plateaus by depth in GPT-2, BERT, ALBERT | our own, cited as context |

## 3. Who has measured the gap per layer?

| work | object | bearing |
|---|---|---|
| Naderi, Nait Saada, Tanner, 2410.07799 (ICML 2025) [S] | the gap between the attention **pattern's** two largest singular values; causes rank collapse in width; at initialisation, random Markov matrices | the pattern's gap: beside, a different mechanism |
| eigenspectrum-variance work on WqWkᵀ (2402.02098) [S]; MP diagnostics of WqWkᵀ per layer (2507.09394) [S] | singular values or the spread of QK's spectrum, not the symmetric part's λ₂/λ₁ | no published λ₂/λ₁ of sym(QᵀK) per layer found |
| B2 (this project, `trackB_circuit/status-B.md` "B2 on one page") | each head's sym(W_QᵀW_K)'s positive share (C3) on 410m along training | the instrument exists (`b2.m_factors`, `b2.sym_eigs`). C3 = 1 exactly when a head's form is positive semi-definite, so Assumption 3.1 can be checked per head from B2's own statistic. That it fails on 410m is likely but unread here |

## What the scan changes in the design

1. **The profile rule is the paper's own criterion** (B.3), applied without retuning, with
   E ↦ −E₁ and −‖m‖² (our energies rise with clustering). No threshold is placed by us.
2. **The gap is read from the weights** (sym QᵀK per layer, λ₂/λ₁ of its positive part), and
   Assumption 3.1 itself (positive-definiteness, OV = QK) is reported layer by layer. A mismatch
   in profile then has a stated reason, not a refutation.
3. **Phase 3's mechanism is read directly.** Pythia's blocks are parallel-residual
   (x_{ℓ+1} = x_ℓ + attn(LN₁x_ℓ) + mlp(LN₂x_ℓ)), so each block's FFN update is a function of the
   stored state alone, and attention's is the remainder. The paper's claim that "the FFN
   dominates attention in the final layers" becomes a per-block ratio of tangential speeds.
4. **The escape's rate against the gap is not read** (§1, correction 2).

## Parked

1. **2511.21635's figures** (why: a published three-phase depth profile on trained transformers;
   cost: a full-text read; could change: whether §8 can say "seen in ViTs, not tested in LMs").
   For the close-out's §8 sentence.
2. **Wendler et al. 2402.10588 and Shibata et al. 2026's final-layer shifts** (why: the paper's
   own pointers to a terminal escape in pretrained models; cost: two reads; could change: the
   email's framing). For the close-out.
