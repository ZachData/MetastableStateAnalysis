<!-- trackC_writeup/design-C7.md -->
# Track C — design: C7 (i), Isobe et al.'s three phases on Pythia-410m

Opened 2026-11-09 by STATE Blocked 153 as revised (#340; Claude's choice at the user's
"reconsider Blocked 153 with C7 on the list"). The user says the rewrite of `paper/` is done.
Row: `docs/ROADMAP.md` §8 C7 (i) only; (ii) and (iii) stay unscheduled. Scan:
`trackC_writeup/lit-C7.md` (2026-11-09; 2605.07772 v2 read in full). Unit 0 is writing only and
nothing ran. **Proposed; the decisions are Blocked 154.**

## What C7 (i) is for

Isobe, Inoue and Imaizumi (2605.07772) prove that, in a noisy mean-field transformer with only
the FFN trained, the trained depth dynamics have three phases. Attention clusters the tokens in
the first layers (1). The distribution stays near the clustered state through the interior, the
"turnpike plateau" (2). In the final layers the FFN dominates attention and moves the tokens out
of the cluster, the "terminal escape" (3). The plateau's persistence is set by the attention
matrix's spectral gap 1 − λ₂/λ₁. Their evidence is particles in d = 2, and they say it is not a
measurement on pretrained transformers. **C7 (i) reads, on Pythia-410m's stored runs,
(P) whether the energy's depth profile has the three phases by their own criterion, (F) whether
the FFN takes over from attention in the final blocks, and (Q) what the interaction matrix's gap
is per layer.** It also checks (J) whether the gap goes with the plateau along training, as their
λ₂ sweep found. All of this is description at tier 1. A mismatch refutes nothing of theirs,
because their model shares A across depth, trains only the FFN, adds noise and starts clustered.
The result goes to `paper/` §8 as a sentence and to the user's email to the authors (outreach,
not posting).

## The inputs (no forward pass of the model; CPU, because nothing here is a batch of GPU work)

| item | fixed as | why |
|---|---|---|
| runs | the kept seed-0 410m runs, `trackA_force/a1a.RUNS`: 18 steps × the 7 v1 prompts, `data/phase12/<run>/pythia-410m-step<k>_<prompt>/` | the records C2's figure 13 and appendix C read; all on disk (checked 2026-11-09: `activations.npz` (25, 465, 1024), `attentions.npz` (24, 16, 465, 465), `energies.json`) |
| states | `activations[s]`, s = 0 … 23 = the input to block s; **s = 24 (after the final LayerNorm) is beside, not in the profile** | the final LN is the readout's normaliser (Machina & Mercer: optimised for isotropy), not a step of the dynamics. Block 23's own output before the LN is not stored, so the profile's last state is the input to block 23 |
| weights | the 18 checkpoints in `data/hf` (cached, `HF_HUB_OFFLINE=1`), loaded by `trackB_circuit.b2.load_weights`' loader | Q and F need W_Q, W_K, W_O, W_V and the MLP; B2's loader is tested |
| known answers (refuse on a miss) | (k1) `c2_grid.check_a5`: the runs reproduce A5's seed-0 energies; (k2) for blocks 2, 12, 22 at steps 512 and 143000 on `wiki_paragraph`, HF's block forward on stored x_s reproduces x_{s+1} to relative 1e-4; (k3) `b2.sym_eigs` against a dense `eigvalsh` on 3 heads, 1e-8 | k2 is what makes F's decomposition exact |

## The readings (the family: 18 steps × 7 prompts × states 0–23 / blocks 0–23; seed 0 only)

**P, the profile, by the paper's own criterion (2605.07772 Appendix B.3), without retuning.**
For each (step, prompt) and each of the two statistics f ∈ {E₁ (`energies.json`, β = 1), ‖m‖²
(the mean of the unit-normalised tokens, from `activations`)}, set F(s) = −f(s). Our energies
rise with clustering and theirs fall, and the criterion is invariant to shift and scale. Then
T = 23, R = max F − min F, F_pl = the median of F over s ∈ [0.4T, 0.6T] = states 10–13, and:

| part | condition | in our sign |
|---|---|---|
| initial clustering | (F(0) − F_pl)/R ≥ 0.1 | the embedding's f below the plateau |
| plateau | SD_{10–13}(F)/R ≤ 0.05 | f flat on states 10–13 |
| terminal response | (F(23) − F_pl)/R ≥ 0.1 | f at block 23's input below the plateau |
| full profile | all three | |

Reported per step: the count of prompts passing each part (k of 7), the same four on the mean
over prompts of f, and, where the terminal part passes, t_esc and the fitted rate a. Here t_esc
is the first state in [11.5, 23] reaching 10 % of the plateau-to-terminal rise, and a comes from
F − F_pl ≈ Ce^{−a(23−s)} on states 17–23. Position 0 is kept, as `energies.json` has it. ‖m‖² is
the energy's leading term at weak attention (their Remark 3.11) and at β = 1 for us
(E₁ = ½ + ½‖m‖² + R).

**F, who moves the tokens, block by block.** Pythia's blocks are parallel-residual, so for block
s = 0 … 22 and token i, the FFN update u_i = mlp_s(LN₂(x_{s,i})) is a function of the stored state
alone, and attention's update is a_i = x_{s+1,i} − x_{s,i} − u_i. Their dynamics move tokens on
the sphere through P⊥_x, so the reading is each update's tangential speed:
‖P⊥_{x̂} u‖ / ‖x‖ and ‖P⊥_{x̂} a‖ / ‖x‖. Per block, take the median over tokens (position 0
out, the sink) and **ρ_s = FFN speed / attention speed**. Block 23 gets the FFN speed only, since
its output is stored only after the LN. The label, fixed now: **"the FFN takes over at the end"**
when ρ_s > 1 on each of blocks 20–22 and the median of ρ over blocks 10–13 is < 1. The onset is
the first block from which ρ > 1 holds through 22, reported against t_esc.

**Q, the interaction matrix per layer.** For block s, S_s = Σ_h sym(W_Q^hᵀ W_K^h)/√64. This is the
offset-0 form, with rotary dimensions included and QK biases ignored. The sum over heads is the
layer's energy at the paper's first order (Remark 3.11: each head contributes ⟨m, A_h m⟩, and
these add). From its eigenvalues λ₁ ≥ λ₂ ≥ … we read:
- **g_s = 1 − λ₂/λ₁**, the paper's gap factor, if λ₁ > 0 (else "not read");
- the positive share Σλ₊/Σ|λ| (B2's C3), which is 1 exactly when Assumption 3.1's
  positive-definiteness holds up to its null space;
- λ₁ in the unit-LN₁ frame (λ₁ · r̄_s², with r̄_s the tokens' median ‖LN₁(x)‖), for the
  paper's λ₁e^{−λ₁} factor.

Beside: the per-head medians of g and the positive share; the unrotated dimensions alone (B2's
`C3_unrot` convention); and S-USA's V = A as the Frobenius cosine between Σ_h sym(W_O^h W_V^h)
and S_s. Also beside, a different mechanism: the attention pattern's σ₂/σ₁ (2410.07799) per
head, median per layer, from `attentions.npz`. The pattern's eigen-gap is not read: P is lower
triangular, so its eigenvalues are its diagonal (`lit-C7.md` §1, correction 1).

**J, the gap against the plateau (description, post hoc on 18 points).** The paper's λ₂ sweep has
a smaller λ₂/λ₁ give a flatter, longer plateau and a sharper terminal lift. On the prompt-mean
E₁, we report two Spearman correlations over the 18 steps: of the interior gap (the median of
g_s over blocks 10–13) with the plateau's SD/R, and with the terminal lift (F(23) − F_pl)/R.
Steps are not independent and everything changes along training, so J says only whether the two
move together. Not read: whether the gap sets the escape's *rate*. In the paper that rate is
R_ρ(g_ℓ), a property of the loss, with no analogue here (`lit-C7.md` §1, correction 2).

**Step 0 is the no-train control** (their Appendix B.2: the profile appears only after training).
Its P, F and Q are reported in the same table, and a part that passes at step 0 is not read as
learned.

## The gate (U1, before any Pythia profile is labelled)

| arm | plant | pass |
|---|---|---|
| G0 | the known answers k1–k3 | all, else refuse |
| G1 | the criterion on 24-state profiles: their template F(s) = e^{−as} + e^{−a(23−s)}, a ∈ {0.3, 0.6, 1.0}; a monotone fall (linear; logistic at s = 12); a mid-stack step (flat to 11, then a fall) | template: full profile; monotone: not full; the mid-stack step: no plateau or no initial part (the shape appendix C's drops suggest); else refuse |
| G1n | the template at a = 0.6 with Gaussian noise of SD 0.01 and 0.02 of R, 200 draws each | reported, no bar: how much noise the plateau part survives |
| G2 | F's split on a synthetic block: u planted, a planted, x_{s+1} = x_s + a + u | ρ recovered to 1e-10 |

G1 tests that the code implements the criterion. It is not a power analysis on Pythia.

## Leans, written before any output

- Appendix C and figure 13 have shown E₁'s change at every depth. **P's labels are therefore
  post hoc on a profile we have seen.** The protection is that the criterion is the paper's own,
  taken without retuning.
- C2's drops sit mid-stack (transitions 10–12 at 512; `lit-C2p-fulltext.md` M7). Where they do,
  the plateau part (states 10–13) fails before the terminal part is reached. The lean is that the
  full profile fails at 512, on the plateau.
- Timkey & van Schijndel's final-layer similarity in GPT-2 points the other way from an escape.

## Not this design

R_ρ and the escape's rate; any seed but 0, or another size (Theorem 3.9's T-independent
boundary width could be read on 70m / 160m / 410m. Parked: it needs per-layer states on those
sizes, unchecked); noise ε, which has no analogue; the post-LN state as dynamics; any causal
statement (F is a decomposition of the stored update, not an intervention); C7 (ii), (iii).

## Exit condition and budget

**Exit:** P, F, Q and J read on seed 0's 18 steps with step 0 as the control, or a recorded
refusal (a known answer missed, G1 failed). **Budget: 3 units** (#340): this one, U1, the
close-out. At the budget it closes at what it knows, unless the user extends it on the record.

| unit | what | done when |
|---|---|---|
| 0 | the scan and this design (writing only) | Blocked 154 decided; the design frozen |
| 1 | `trackC_writeup/c7_profile.py` and pure tests; G0–G2; the run (CPU; 18 weight loads, 126 state reads); P, F, Q, J read; status section | every reading, or a refusal |
| close-out | one page; a §8 sentence for `paper/`; the email's paragraph for the user; Parked items carried or dropped; the card current | the card current |

## Decisions (each yours; recommendation first)

| # | decision | options | recommendation |
|---|---|---|---|
| 154 | the design | **D1, the profile rule:** (a) the paper's B.3 criterion on −E₁ and −‖m‖², states 0–23, state 24 beside; (b) the same on states 0–24. **D2, a layer's A:** (a) the summed sym QᵀK over heads (Remark 3.11), per-head medians beside; (b) the per-head median primary; (c) the head of largest λ₁. **D3, Phase 3's mechanism:** (a) F, the tangential speeds from the stored states, known-answer gated; (b) the MLPs' weight norms only, with no block computed. **D4, scope:** (a) seed 0's 18 steps, as above; (b) also the 9 PolyPythias seeds at 512 and 143000 (forward passes, about an hour on the 3080) | **all (a).** D1 (a): the readout's normaliser is not dynamics. D2 (a): the layer's energy at the paper's order is the summed form. D3 (a): it reads the paper's mechanism, not a proxy, and parallel residuals make it exact. D4 (a): the result is a description for a sentence and an email; seeds would turn it into a claim, which is not asked for |

## Worth challenging (for the PR)

- **Adopting their criterion.** It was built for d = 2 particles over T = 80 steps. With
  T = 23 its interior window holds 4 states, so the plateau part is a test on 4 numbers. The
  alternative was a criterion of our own, gated on plants, and it was rejected because every
  threshold in it would be placed after seeing figure 13.
- **Summing heads into one A.** Softmax acts per head, so the sum is the layer's form only at
  first order in β. The per-head reading is beside, and D2 is yours.
- **Ignoring rotary and biases in A.** S_s is the offset-0 form. On 410m, 16 of each head's 64
  dimensions rotate, and the unrotated part is read beside.
- **ρ against 1.** "Dominates" read as a speed ratio above 1 on blocks 20–22 is our
  translation of the paper's words. Both speeds are reported, so a reader can draw another line.
- **No seeds** (D4). One seed's profile for a sentence in §8 and an email. If the authors want
  a claim, that is a new row.
