# Track A, A2 — literature scan at opening (2026-10-16)

Trigger 1 (`CLAUDE.md` "Literature scans"; ROADMAP §6 risks: "literature trigger 1 before A2's
design freezes"): A2 opens, so this precedes `design-A2.md`'s freeze. Scope: what A2 (a β refit,
per-component dissipation, mean-ablation of the components that push apart) must not get wrong.
Eight web searches (2026-10-16), read at abstract / summary level (scholarly hosts are blocked
from the session, so no full text). Not repeated: `lit-A.md` (collapse time, normalisation,
Euler steps, β over training), `p10_cluster_function/lit-10.md` (2504.02732 is there),
`trackB_circuit/lit-B.md` (patching practice, Heimersheim), `p7_motifs/lit-7.md` (2510.06477).

## 1. What resists collapse: the residual, the MLP, inactive heads

| source | claim (abstract level) | what it means for A2 |
|---|---|---|
| Dong, Cordonnier, Loukas 2103.03404 | pure attention loses rank doubly exponentially; skip connections stop it; MLPs slow it by their Lipschitz constant; LayerNorm does not help | the residual itself is an anti-collapse term with no component to ablate: A2's ablations remove a component's *update*, never the skip, so the skip's share is not A2's to measure |
| 2604.23681 (Apr 2026) | residuals generically obstruct rank collapse without the MLP; the MLP's distinct role is directions outside the embeddings' span; per-head contributions are not identifiable after W_O sums them ("head-channel non-identifiability") | (a) A1's ">90 % off the field" may be the MLP's new directions, a reading A2's per-component split can test; (b) per-head shares are defined here as each head's own `W_O^h` slice of the output, which is exact as a sum but is one choice of split (`design-A2.md` "Worth challenging" 4) |
| Barbero et al. 2504.02732 | first-token sinks keep heads inactive and so limit over-mixing; deeper models need more inactive heads | the sink is anti-collapse by *not moving*: it lowers a head's update, so it appears as a small component, not a negative one. A2 reads the sink's part as received attention × value (Phase 10's row A5), and a head ablated to its mean keeps its sink pattern only if the mean does |
| Queipo-de-Llano et al. 2510.06477 | sinks and compression valleys both come from one massive-norm token at mid depth; mix, compress, refine by depth | 410m's mid-layer compression, if present, is a shared move the field's β = 0 term carries (1e finding 1); A2 reads every component with the shared drift removed beside (A1's `T_proj_c`) |

## 2. Ablation

| source | claim | what it means for A2 |
|---|---|---|
| Li and Janson 2409.09951 (NeurIPS 2024) | zero, mean and resample are all "total" ablations; mean ablation injects inconsistent information and overstates importance; the optimal constant minimises loss | A2's ablation is mean ablation (the ROADMAP row), with the constant taken on held-apart text (the long passages) so v1 is not ablated to its own mean; its effect is quoted against random matched ablations, so the overstatement both carry cancels in the comparison, not in the raw number. Optimal ablation is beside only if U5's budget allows (it needs a loss fit per component) |

## 3. β's estimator

No source found comparing least squares on log attention with the softmax's own cross-entropy for
an attention temperature (two searches). The standard statistical point stands without a
citation: least squares on log A weights every pair equally, including the near-zero entries
whose logs carry the most noise; the softmax likelihood weights by attention mass. B2a found the
two disagree in sign in 6 of 14 heads, all 6 with the least-squares β positive
(`trackB_circuit/status-B.md` "A1a's β against M1's"). So unit 2 does not lean on the literature;
its rule is fixed in `design-A2.md` before it runs.

## 4. Per-component energy

No source found that measures dissipation per block and splits it into attention and MLP in a
trained model (two searches); the theory treats attention alone as gradient flow (2505.19458,
2605.04279, symmetric weights or single heads). The project's own split is `PROJECT.md` §3.8.3
(Tier B: exact attention / FFN / per-head split, at β = 1 in the sphere frame, 2026-09-06); A2
re-reads it at the measured β in A1's frame. Heterogeneous heads cluster super-additively in a
scalar-head model (2601.21942, already in `lit-1d.md`): the "cooperating heads or a few large
ones" question (1e at close) has that as its prior.

## What changed in the design because of this scan

- The skip's anti-collapse share is out of scope by construction (§1, 2103.03404): ablations
  remove updates only.
- Per-head shares are named as the `W_O^h` split and its non-identifiability is a stated limit
  (§1, 2604.23681).
- The ablation constant comes from the long passages, and every ablation effect is quoted
  against random matched ablations (§2, 2409.09951).
- The off-field share is read per component, MLP first (§1, 2604.23681's "new directions").
