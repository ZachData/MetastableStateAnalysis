"""
p6_subspace — the real/imaginary residual-stream channels, rebuilt live.

Phase 6 moved to `archive/` on 2026-08-22 and its twelve predictions went
`dormant`: pre-registered, falsifier intact, but with no live instrument able to
produce a p-value. `P6-R2` and `P6-R4` are revived here, which is what taking
them out of `dormant` requires -- `archive/README.md` rule 2 is that nothing is
salvaged by copying, so this is a rebuild against `core/particles.py` and
`core/nulls.py` rather than a lift of `archive/p6_subspace/subspace_build.py`.

What that buys, and what it does not: the apparatus becomes live, so the two
predictions can carry a p-value. No p-value is emitted, because this repository
holds no run artifacts. That is the same position `CLAIM-C`, `P-S1`, `P-T1` and
`P-M1` are in -- apparatus validated on synthetic inputs with known answers, and
`claims/adjudications/` still empty.

Read `p6_subspace/math-6.md` §7 before using any of this. The archived run's
headline numbers (alignment 0.887 with U_A against 0.067 with U_neg) are NOT
inputs to these functions and are not adjudicable by them, because the statistic
they came from is not dimension-normalized and `claims/audits/
p6_projector_labels.json` measures the correction as larger than the effect.
The rebuilt statistics are normalized by construction; they supersede that
comparison rather than adjudicating it.
"""
