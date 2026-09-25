# Phase 1d predictions (written before the code; never registered)

> **Not in `claims/registry.json`.** Written in the August branch's
> `PREDICTIONS.md` and never entered in the registry; the v1 runs have since
> been examined, so they cannot be scored blind on them
> (`p1d_cluster_ensemble/status-1d.md` "Registry").

Committed **before any Phase 1d code is written or run**, same rule as above: this is a
re-analysis of artifacts already on disk, and the outcome that would be most convenient — "the
other methods agree with HDBSCAN once tuned, so nothing we have published changes" — is
exactly the one it would be easiest to arrive at by choosing hyperparameters after seeing the
answer. No edits after results; dated addenda only.

## What Phase 1d is for

Every cluster-conditioned claim in this project rests on one clusterer at one setting:
`hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed")` on cosine distance
(`p1_mstate_tracking/clustering.py`). Three other partitions of the same tokens are computed
and persisted, and `p1_visualization/cluster_methods.py` measures whether they agree — but
none of the four was ever tuned, and the agreement statistic therefore compares four sets of
library defaults, not four methods. Phase 1d tunes each family against criteria that do not
presuppose HDBSCAN's answer, and asks what a graded, method-independent annotation says that
the categorical label cannot.

| ID | Prediction | Falsifier | Instrument | Cost |
|---|---|---|---|---|
| **P-C1** | At layers where Phase 1 already reports method agreement (counts within ±1), the *tuned* ensemble's consensus strength stays ≥ 0.9 — tuning does not dissolve the agreement the defaults showed | Consensus strength drops under tuning ⇒ the reported cross-method agreement was a coincidence of library defaults, not a property of the geometry, and every "the methods agree" sentence in Phase 1 needs restating | 1d-B | [R] |
| **P-C2** | `min_cluster_size=2` is **not** the stability-optimal HDBSCAN setting: the selected setting is larger, and ARI between the tuned and the shipped HDBSCAN partition is < 0.9 at the majority of mid-depth layers | The shipped default is already stability-optimal ⇒ every existing HDBSCAN-conditioned result stands unchanged and this phase's premise is void — a real and welcome outcome, recorded as such | 1d-A | [R] |
| **P-C3** | The unclustered population is not homogeneous: within HDBSCAN noise, at least 20% of particles sit above the null-calibrated consensus-confidence threshold, i.e. other methods place them in structure HDBSCAN's density criterion refused | Noise-token confidence is indistinguishable from the matched null ⇒ "unclustered" is one population, and Phase 5c's binary clustered/unclustered split is the right object after all | 1d-C | [R] |
| **P-C4** | Graded consensus confidence beats the binary clustered/noise flag at predicting layer-to-layer cluster persistence: ΔAUC > 0 and outside the permutation null's 2σ band | No improvement ⇒ the categorical label loses nothing the ensemble recovers, and the conglomeration is presentation rather than information. **This is the prediction that decides whether the phase produced anything.** | 1d-D | [R] |

## Why each one is worth pre-committing

**P-C2 is stated against our own interest.** The convenient result is that the default was
right all along. The prediction says it was not, which — if it holds — means every downstream
cluster-conditioned number was computed on a partition nobody chose deliberately. Stating it
the other way round would make the phase unfalsifiable: any disagreement could be attributed
to a "badly tuned" alternative.

**P-C1 and P-C2 can both hold, and that combination is the interesting one.** The methods can
agree on the *coarse* partition (P-C1) while HDBSCAN's specific setting is wrong about the
*boundary* (P-C2). Registering them separately keeps that from being resolved by narrative
afterwards.

**P-C3 targets the phase whose object of study this is.** Phase 5c's framing — every particle,
clustering as an annotation — makes "unclustered" a population rather than a residue. If that
population turns out to have a structured sub-part visible to five other methods, the binary
tag is hiding it. The 20% floor is **placed, not calibrated**: there is no distribution to
derive it from before the phase runs, and it is written here rather than chosen later
precisely because it is arbitrary. The threshold it is measured against is calibrated (95th
percentile of the matched-null co-association); only the 20% is placed.

**P-C4 is the phase's own falsification.** Everything else Phase 1d produces is descriptive: a
prettier annotation is not a result. The claim that earns the phase is that the graded
annotation carries information the categorical one does not, and the only honest way to test
that is to have both predict the same held-out thing and compare. If ΔAUC lands inside the
null band, the correct write-up is "the ensemble adds nothing measurable" — and that sentence
is easier to write with the prediction already on record.

## Adjudication constraints, fixed now

1. **One vote per family.** The ensemble takes one tuned partition from each method family.
   Six agglomerative linkages voting against one HDBSCAN is a rigged consensus, and which
   families are included must not be a post-hoc choice.
2. **The comparison partition for P-C2 is the shipped one**, `min_cluster_size=2` with library
   defaults, not a re-run with different defaults.
3. **A family that fails the null gate at a layer does not vote there**, and the fact that it
   abstained is reported alongside the consensus, not silently absorbed into it.
4. **P-C4's persistence target is computed from the consensus partition, not from HDBSCAN**,
   and the same target is used for both predictors. Scoring the graded annotation against a
   target HDBSCAN defined would be a rigged comparison in the other direction.

---

Not yet adjudicated. Phase 1c, Phase 1d and Phase 2d code exists and is validated on synthetic
data; no sub-experiment has been run against Pythia artifacts.
