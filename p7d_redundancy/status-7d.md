<!-- p7d_redundancy/status-7d.md -->
# Phase 7d — STATUS

**Last verified:** 2026-09-10.
**Overall:** Q1, Q2 and Q3 are answered on pythia-410m. Q4 is mostly on disk and
unread along this axis; Q5 needs Q4 first. Nothing is registered and nothing can
be — every measurement here is on an artifact spent under `check_registry`
rule 3. Restore checks are exact (`0.0e+00`) on every run reported below.

## What is answered

### Q1 — membership. **~4 substantial members, ~10 with any effect, of 384.**

`redundancy_catalog.py`, single-head OV ablation at step 16000, 8 sequences, all
384 heads. The distribution is brutally heavy-tailed — the **median head moves
the readout by 0.001** — so the set is a real, small, identifiable object rather
than a gradient.

| | |
|---|---|
| median | +0.00107 |
| p99 | +0.1935 |
| above +0.05 / +0.2 / +1.0 | 10 / 4 / 2 heads |
| max / min | +1.9662 (`L5H2`) / −0.0522 (`L10H7`) |

`L5H2` +1.966, `L7H8` +1.019, **`L12H5` +0.420**, `L8H6` +0.212, `L11H14`
+0.190, `L8H9` +0.131, `L15H14` +0.087, `L7H1` +0.067, `L9H13` +0.064, `L10H9`
+0.050. **`L12H5` and `L8H6` were entirely unknown** before this sweep, and the
38-head sample of §3.12-R had missed both — which is why the full sweep was run
instead of a proxy screen.

Members are spread across layers 5, 7, 8, 8, 11, 12, 15 — **not** clustered in
the 9–20 band where the token-identity copiers live. Membership and copying are
different properties.

### Q2 — **no, they did not form together.** Q3 — **yes for `L5H2`, no for `L7H8`.**

`member_formation_curves.py`, 23 checkpoints, 16 sequences, six members plus the
joint `L5H2`+`L7H8` arm and the residual-delta cosine. `L7H8` reproduces
§3.11-A throughout and step 16000 reproduces §3.12-S to four decimals.

| member | forms in | peak | at step | at 143000 | retained |
|---|---|---|---|---|---|
| `L5H2` | **(512, 1000]** | +8.43 | 2000 | +1.26 | 15 % |
| `L11H14` | **(512, 1000]** | +3.57 | 1000 | +0.17 | 5 % |
| `L12H5` | (1000, 2000] | +0.99 | 3000 | +0.11 | 11 % |
| `L8H9` | (1000, 2000] | +0.60 | 3000 | −0.01 | −1 % |
| `L8H6` | (512, 1000] | +0.70 | 3000 | +0.15 | 22 % |
| `L7H8` | **(2000, 3000]** | +1.54 | 54000 | +1.22 | **79 %** |

`L5H2` appears in the **same interval the model acquires induction at all**
(second-copy NLL 12.63 → 4.91), and Pythia publishes no checkpoint between 512
and 1000, so that is the finest interval this axis can resolve. `L7H8` — the
head §3.11–§3.12 was largely spent on — is the **last** member to arrive, and
the only one of the six that rises monotonically.

**The redundancy postdates both heads.** The interaction is ≈0 while `L5H2` is
at its maximum (−0.03 at 1000, +0.02 at 2000), then +0.39 (3000), +1.72 (4000),
+4.15 (16000), +4.17 (32000). Dating the set by dating its members would have
been wrong by 2000 steps. The alignment behaves the same way and is not a
converged endpoint but `L7H8`'s **entry condition**: −0.17 → +0.01 → +0.26 →
**+0.83 at step 4000** → +0.91, flat after.

Full table and caveats: `PROJECT.md` §3.12-U.

## What is open

- **Q4 — structure per member.** Largely on disk and unread along this axis:
  `qk_symmetry_sweep.json` (384 heads × 19 steps), `ov_per_head_series.json`,
  `copying_score_sweep.json`, `behavioural_series.json`. §3.12-U says which
  *steps* matter — the action is in `(512, 4000]`, not at the endpoints.
- **Q5 — classes.** Needs Q4. Cluster on (formation step, spectral signature, QK
  symmetry trajectory, copying score, causal magnitude); report both views
  per §3.13.
- **Pass 2 of the catalogue** — the pairwise interaction matrix over the top
  members (6 arms for the top 4, 45 for the top 10) at a fixed 16 sequences.
  One redundancy set, or several disjoint ones?
- **The step-1000 circuit, which is a different circuit.** At step 1000 the
  mechanism is `L5H2` (+4.97) and `L11H14` (+3.57), with `L7H8` absent and every
  other member under +0.06. `L11H14` is the top copier in the model at 143000
  but does not enter the copying top-5 until step 16000, and its step-1000 score
  is **not on disk** — `copying_score_sweep.json` keeps only the top ten and a
  named set. Measuring it is a weights-only step and comes first.

## The pilot that must not be over-read

`--heads L5H2,L11H14 --pair L5H2,L11H14 --steps 1000`, restore exact: singles
+4.973 and +3.567, joint **+6.859**, interaction **−1.682**, δ-cosine +0.857.
Sub-additive — the serial signature, opposite in sign to the `L7H8` pair.

**The sign is not evidence.** That joint arm lands at NLL 11.77 against a uniform
ceiling of 10.83, so it is outside the readout entirely, and §3.12-M5 warns that
this readout's compression biases independent contributions toward exactly this
apparent sub-additivity. Settle it with the graded readout (§3.12-M's KL / λ
scale) before running more steps at raw `dNLL`. See design-7d.md, "The readout's
ceiling".

## Reproducing

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf METS_RESULTS_DIR=$PWD/data/phase12
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

python -u p7d_redundancy/redundancy_catalog.py                    # Q1,  ~25 min
python -u p7d_redundancy/member_formation_curves.py --top 6 --chunk 4
python -u p7d_redundancy/member_formation_curves.py --append \
        --steps 3000,5000,7000,9000                               # Q2/Q3, ~20 min
python -u p7d_redundancy/two_big_heads.py                         # §3.12-S
```

Outputs land in `data/analysis/*.json`, which is git-ignored — see PROJECT.md's
resume block for the full list. Use `--out` when running a different `--pair` or
`--heads`: a partial run replaces the main six-member curve otherwise.
