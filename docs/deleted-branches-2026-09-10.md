# Branches deleted 2026-09-10

Deleted after confirming each was fully merged into `origin/main` (`git
branch -r --merged`), except the last, which was 93 commits stale and whose
one unmerged file was rescued first.

Any of these can be restored without a local copy:

```bash
git push origin <sha>:refs/heads/<name>
```

| sha | branch |
|---|---|
| `1cf825f35676` | `claude/claim-c-cell-drop-curve-jpklpx` |
| `d5b410d4a715` | `claude/claim-c-homogeneity-calibration-7glywo` |
| `2ff45dc9f7ae` | `claude/claim-c-null-construction-gjvh8k` |
| `92f6ef35d454` | `claude/eigenvalue-eigenvector-physics-i79ov0` |
| `117f6b2967f3` | `claude/fix-main-test-errors` |
| `2de3f365e119` | `claude/grid-feasibility-claim-b-rvne1x` |
| `34536f878123` | `claude/metastable-popperian-ci-cmbgtx` |
| `a735c0754b89` | `claude/metastable-popperian-ci-dw60l5` |
| `73f566c1b9a4` | `claude/metastable-popperian-ci-m3j5sy` |
| `6f01f6b5c29c` | `claude/p1-p2-terminal-args-v0rdaw` |
| `93326a6288db` | `claude/p6-r2-r4-null-adjudication-4bvrqq` |
| `77098465e657` | `claude/p6-runtime-fix` |
| `7af117f8653b` | `claude/particle-interpretation-framework-0ky1x1` |
| `de412eb073d5` | `claude/phase-1b-visualization-lu93v6` |
| `0ba8109ae663` | `claude/phase-1c-visualization-uh94ix` |
| `9510d8f1cb32` | `claude/pi3-pi4-null-construction-uopfb5` |
| `99f69180e71d` | `claude/popperian-ci-dry-runs-8mblvn` |
| `98502330e3c6` | `claude/popperian-ci-pilot-setup-rqfzdt` |
| `186295bdd500` | `claude/popperian-cicd-planning-3ptr53` |
| `ce119be53eb6` | `claude/publication-ideas-transformers-d9ze9x` |
| `19e52546e8e7` | `claude/phase-2b-visualization-31iu5o (heads.py rescued in 86b7082)` |

`claude/phase-2b-visualization-31iu5o` was the only one not merged. Everything
it carried was already on `main` except
`p2b_imaginary/visualization/heads.py`, taken onto `claude/p2b-per-head-figure`
in commit `86b7082` before the branch was deleted.

Not deleted, and not to be deleted without checking `INDEX.md` first: 
`claude/particle-methods-comparison-vpuads` (all of Phase 1d,
`p1d_cluster_ensemble/`, on no other branch) and
`claude/visualize-mets-results-sl2ya5` (`tools/visualize_latest.py`). `INDEX.md`
lists both under "In flight on other branches".
