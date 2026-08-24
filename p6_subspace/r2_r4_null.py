"""
p6_subspace/r2_r4_null.py — nulls for P6-R2 and P6-R4.

THE TWO PREDICTIONS, AND WHY THEY SHARE A CONSTRUCTION

    P6-R2  LDA aligns more with the real repulsive subspace U_neg than with
           the imaginary subspace U_A.
    P6-R4  S-only projection preserves cluster membership.

They are not the `P5b-B1`/`P5b-B3` pattern -- not one test with two thresholds.
R2 compares alignments of a cluster-separating direction; R4 compares the
accuracy of probes fit inside different subspaces. Two statistics, two
instruments. What they share is the PROJECTOR, so a projector defect moves both,
which is why they are built together (the lesson `EVALUABILITY.md` draws for
`CLAIM-B`/`P-I1`) and why the registry records them as dependent factors.

THE NULL IS A MATCHED-DIMENSION RANDOM SUBSPACE, NOT A SIGN TEST

H0-OPERATOR is "the operator classification carries no information about
activation geometry". Realise it directly: replace the operator-derived
subspaces with random subspaces OF THE SAME DIMENSION and recompute the same
statistic. Everything the statistic could be reading off dimension is held
fixed, and only the operator content moves.

This matters more than it looks, and it changes the answer to a question the
plan had already framed. `status-6.md` records that 49 ALBERT layers are not 49
independent observations, and the obvious repair -- `CLAIM-C`'s, which is to
choose an exchangeable unit COARSER than the measurement unit -- lands on "one
model", n = 1. Under a SIGN-FLIP null over units, n = 1 has an attainable floor
of 2/(2^1 + 1) = 0.667 and the test refuses however clean the data is. Under
this null it does not, because the randomisation is over SUBSPACES rather than
over units, and its resolution floor is 1/(n_draws + 1) -- a number the caller
controls.

So the binding constraint here was never the exchangeable unit. It was the
choice of null. `attainable_floor_report` prints both framings side by side
rather than leaving that as a claim in a docstring.

WHAT THE EXCHANGEABLE UNIT DOES CONTROL

It controls whether one null draw is SHARED across layers or drawn
INDEPENDENTLY per layer:

    unit="model"  one draw of the subspaces, applied to every layer. Correct
                  when the layers share a projector -- which under ALBERT's
                  weight-tying they literally do: one OV matrix, one Schur
                  decomposition, one projector pair, 49 activation snapshots.

    unit="layer"  a fresh draw per layer. Treats the layers as independent
                  observations, which is the error status-6.md names. Kept
                  BUILDABLE and refused at adjudication rather than deleted,
                  because the size of the gap between the two is a measurement
                  worth having, and a reader who cannot run the wrong one has
                  to take the right one on faith.

NO UNIT IS REGISTERED, AND SO NOTHING HERE ADJUDICATES

`REGISTERED_EXCHANGEABLE_UNIT` is None. Adjudication refuses while it is, and
passing `unit=` does not route around that -- the argument selects which null to
COMPUTE, and the module constant is what decides which one may enter a claim's
e-process. Choosing it is a scientific decision of the same class as CLAIM-C's
criterion, and making it after seeing a p-value would void the guarantee. When
it is made it belongs here, in the registry's `null_construction`, and in
POPPER_PLAN -- in that order, before any record is written.

WHAT THIS INSTRUMENT DOES NOT DO

It does not adjudicate the 2026-04 ALBERT run. That run reported RAW alignments
(0.887 with U_A, 0.067 with U_neg) and `claims/audits/p6_projector_labels.json`
measures the chance ratio at ALBERT's shape as 24.9 against an observed 13.2 --
the dimension correction is larger than the effect. The recorded numbers are
superseded as evidence rather than turned into a p-value; see
`p6_subspace/math-6.md` §7.2 and POPPER_PLAN.md §6h.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from core.nulls import p_from_null

from .subspace_geometry import (
    LayerChannels,
    normalized_alignment,
    random_orthogonal_subspace_pair,
    random_subspace,
)

#: The two units this construction can express. Not a free-text field: an
#: unrecognised unit refuses rather than falling back on a default, because a
#: default here would silently pick the anticonservative one.
EXCHANGEABLE_UNITS = ("model", "layer")

#: Which unit may enter a claim's e-process. None means "not decided", and
#: adjudication refuses while it is None. See the module docstring.
REGISTERED_EXCHANGEABLE_UNIT: Optional[str] = None

#: Monte-Carlo null size. A module constant and not a parameter, so a per-run
#: choice is not available -- the same convention P-S1, P-T1, P-M1 and CLAIM-C
#: follow. 2000 puts the resolution floor at 1/2001 = 0.0005, two orders below
#: the registry's alpha, so the floor is not what decides any verdict.
N_NULL_DRAWS = 2000

#: One-sided, fixed in advance. P6-R2 predicts MORE alignment with the real
#: repulsive channel; P6-R4 predicts the real channel PRESERVES cluster
#: membership. Both make the predicted outcome the large one.
ALTERNATIVE = "greater"

#: Ridge coefficient for the probe and the discriminant, as a fraction of the
#: mean eigenvalue of the within-class scatter. PLACED, not calibrated: it
#: exists to make a singular scatter matrix invertible in a k-dimensional
#: subspace where k can exceed the token count, and no distribution was
#: consulted. It is applied IDENTICALLY to the observed and null arms, so it
#: cannot move the contrast -- which is why a placed value is tolerable here
#: and would not be if it entered only one arm.
RIDGE_FRACTION = 1e-3

#: Cross-validation folds for the probe. PLACED.
N_FOLDS = 5

_SEED = 20260824


class NullRefused(Exception):
    """
    Raised when a p-value would have to be computed from inputs that cannot
    support one. Standing rule 4: a number from mismatched inputs is worse than
    no number, because it is unfalsifiable from the output alone.
    """


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

def _check_unit(unit: str) -> str:
    if unit not in EXCHANGEABLE_UNITS:
        raise NullRefused(
            f"unknown exchangeable unit {unit!r}; expected one of "
            f"{list(EXCHANGEABLE_UNITS)}. Refusing rather than defaulting: the "
            f"two units differ by orders of magnitude in the p they produce, "
            f"and a default would silently pick one.")
    return unit


def _draw_rngs(unit: str, n_layers: int, draw: int) -> List[np.random.Generator]:
    """
    One generator per layer for this draw.

    unit="model" gives every layer the SAME generator, so the subspaces drawn
    are the same object across layers -- which is what "the layers share one
    projector" means when written as a randomisation. unit="layer" gives each
    its own.
    """
    if unit == "model":
        # Separate generator OBJECTS on one seed, not one shared object: a
        # shared object would advance between layers and hand each a different
        # subspace, which is the independent draw wearing the coarse unit's
        # name. Seeded identically, two layers with the same channel dimensions
        # get the SAME random subspace -- which is what "one projector, 49
        # activation snapshots" means when written as a randomisation.
        return [np.random.default_rng(_SEED + draw) for _ in range(n_layers)]
    return [np.random.default_rng(_SEED + draw * 100_003 + i)
            for i in range(n_layers)]


def attainable_floor_report(n_units: int,
                            n_draws: Optional[int] = None) -> dict:
    """
    The smallest p each candidate design can express, before any data is seen.

    `EVALUABILITY.md`: check the attainable floor BEFORE building the null, not
    after a result comes back null. Both framings are reported because the
    comparison is the finding -- the same n = 1 that makes a sign-flip design
    unusable is no obstacle at all to a subspace-randomisation design.
    """
    # Resolved here and NOT as a default argument. Python binds defaults once
    # at definition time, so `n_draws: int = N_NULL_DRAWS` would freeze the
    # value the module had when it was first imported -- and the floor REFUSAL
    # reads this, so a smaller null would have been reported as safe. Found by
    # a test that set N_NULL_DRAWS = 5 and got no refusal.
    n_draws = N_NULL_DRAWS if n_draws is None else n_draws
    return {
        "n_units": int(n_units),
        "n_null_draws": int(n_draws),
        "sign_flip_floor": 2.0 / (2 ** int(n_units) + 1.0),
        "subspace_randomisation_floor": 1.0 / (int(n_draws) + 1.0),
        "_note": (
            "sign_flip_floor is what a CLAIM-C-style enumeration over n "
            "exchangeable units could express; subspace_randomisation_floor is "
            "what THIS construction expresses. At n_units=1 the first is 0.667 "
            "and refuses at any sensible alpha while the second is unaffected, "
            "because the randomisation is over subspaces rather than over "
            "units. The choice of null, not the choice of unit, is what decides "
            "whether a one-model design can reject."),
    }


def _finish(observed: float, null_values: Sequence[float], unit: str,
            alpha: float, extra: dict) -> dict:
    null_values = np.asarray(list(null_values), dtype=np.float64)
    if not np.isfinite(observed):
        raise NullRefused(
            f"observed statistic is not finite ({observed!r}); refusing rather "
            f"than dropping it and reporting the rest")
    finite = null_values[np.isfinite(null_values)]
    if finite.size < N_NULL_DRAWS:
        raise NullRefused(
            f"only {finite.size} of {null_values.size} null draws are finite; "
            f"a null thinned by failures is not the null that was designed")

    floor = attainable_floor_report(extra.get("n_units", 1))
    if floor["subspace_randomisation_floor"] > alpha:
        raise NullRefused(
            f"attainable floor {floor['subspace_randomisation_floor']:.4f} "
            f"exceeds alpha={alpha}: this design cannot reject on a perfect "
            f"result, and reporting 'not significant' on nothing is worse than "
            f"reporting nothing. Raise N_NULL_DRAWS.")

    out = p_from_null(observed, finite, alternative=ALTERNATIVE)
    out.update({
        "exchangeable_unit": unit,
        "registered_exchangeable_unit": REGISTERED_EXCHANGEABLE_UNIT,
        "attainable_floor": floor,
        "n_null_draws": N_NULL_DRAWS,
        "alternative": ALTERNATIVE,
    })
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# P6-R2 — alignment of the cluster-separating direction
# ---------------------------------------------------------------------------

def cluster_separating_direction(X: np.ndarray, labels: Sequence[int],
                                 ridge_fraction: float = RIDGE_FRACTION
                                 ) -> np.ndarray:
    """
    The leading multiclass Fisher discriminant direction, ridge-regularised.

    Noise points (`cluster_label` < 0, Phase 1's and HDBSCAN's convention,
    carried by `core.particles`) are excluded: they are the absence of a
    cluster, not a class, and admitting them makes the between-class scatter
    partly a scatter of things that were never claimed to separate.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    keep = labels >= 0
    X, labels = X[keep], labels[keep]
    classes = np.unique(labels)
    if classes.size < 2:
        raise NullRefused(
            f"a cluster-separating direction needs at least two clusters; "
            f"{classes.size} present after dropping noise (label < 0)")

    mu = X.mean(axis=0)
    d = X.shape[1]
    S_w = np.zeros((d, d))
    S_b = np.zeros((d, d))
    for c in classes:
        Xc = X[labels == c]
        mc = Xc.mean(axis=0)
        Dc = Xc - mc
        S_w += Dc.T @ Dc
        diff = (mc - mu).reshape(-1, 1)
        S_b += Xc.shape[0] * (diff @ diff.T)

    ridge = ridge_fraction * float(np.trace(S_w)) / max(d, 1)
    if ridge <= 0:
        ridge = ridge_fraction
    S_w = S_w + ridge * np.eye(d)

    w, V = np.linalg.eigh(np.linalg.solve(S_w, S_b))
    v = V[:, int(np.argmax(w))]
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        raise NullRefused("the discriminant direction degenerated to zero")
    return v / n


def r2_layer_contrast(v: np.ndarray, u_neg: np.ndarray, u_a: np.ndarray,
                      d_model: int) -> float:
    """
    Chance-normalized alignment with U_neg minus chance-normalized alignment
    with U_A. Positive is P6-R2's predicted direction.

    Normalized on BOTH arms, so the contrast is between two numbers that mean
    the same thing. The archived comparison subtracted two raw alignments whose
    scales differ by the dimension ratio.
    """
    return (normalized_alignment(v, u_neg, d_model)
            - normalized_alignment(v, u_a, d_model))


def p_value_p6_r2(directions: Sequence[np.ndarray],
                  channels: Sequence[LayerChannels],
                  unit: str,
                  alpha: float = 0.05) -> dict:
    """
    P6-R2's p-value under a matched-dimension random-subspace null.

    `directions[i]` is layer i's cluster-separating direction and `channels[i]`
    its subspaces. The statistic is the mean contrast over layers; the null
    redraws both subspaces at their own dimensions and recomputes it, sharing
    or not sharing the draw according to `unit`.
    """
    _check_unit(unit)
    directions = list(directions)
    channels = list(channels)
    if len(directions) != len(channels):
        raise NullRefused(
            f"{len(directions)} directions against {len(channels)} layers of "
            f"channels; these index the same layers and must match")
    if not directions:
        raise NullRefused("no layers supplied")

    for i, ch in enumerate(channels):
        if ch.u_neg.shape[1] == 0 or ch.u_a.shape[1] == 0:
            raise NullRefused(
                f"layer {i} has an empty channel "
                f"(dim U_neg={ch.u_neg.shape[1]}, dim U_A={ch.u_a.shape[1]}). "
                f"Normalized alignment is undefined against an empty subspace, "
                f"and treating it as zero would report 'orthogonal' for "
                f"'absent'.")

    observed = float(np.mean([
        r2_layer_contrast(v, ch.u_neg, ch.u_a, ch.d_model)
        for v, ch in zip(directions, channels)]))

    null_values: List[float] = []
    for draw in range(N_NULL_DRAWS):
        rngs = _draw_rngs(unit, len(channels), draw)
        vals = []
        for v, ch, rng in zip(directions, channels, rngs):
            # Mutually orthogonal, because the observed pair is. Drawing them
            # independently leaves the null pairs overlapping where the observed
            # pair cannot be, and that alone made the test anticonservative --
            # measured at 0.0875 against a nominal 0.05 before the fix.
            rn, ra = random_orthogonal_subspace_pair(
                ch.d_model, ch.u_neg.shape[1], ch.u_a.shape[1], rng)
            vals.append(r2_layer_contrast(v, rn, ra, ch.d_model))
        null_values.append(float(np.mean(vals)))

    return _finish(observed, null_values, unit, alpha, {
        "prediction_id": "P6-R2",
        "statistic": "mean over layers of (normalized alignment with U_neg "
                     "minus normalized alignment with U_A)",
        "n_layers": len(channels),
        "n_units": 1 if unit == "model" else len(channels),
        "dims": [ch.dims() for ch in channels],
    })


# ---------------------------------------------------------------------------
# P6-R4 — cluster membership recoverable from a projection
# ---------------------------------------------------------------------------

def _stratified_folds(labels: np.ndarray, n_folds: int,
                      rng: np.random.Generator) -> List[np.ndarray]:
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % n_folds].append(int(i))
    return [np.asarray(sorted(f), dtype=int) for f in folds]


def probe_accuracy(X: np.ndarray, labels: Sequence[int],
                   n_folds: int = N_FOLDS,
                   ridge_fraction: float = RIDGE_FRACTION,
                   seed: int = _SEED) -> float:
    """
    Cross-validated accuracy of a ridge one-vs-rest linear probe.

    NOT sklearn's LogisticRegression, which is what the archived
    `probe_subspace.py` used. Two reasons, and the second is the real one.
    First, `sklearn` is a heavy-tier dependency and this module is pure tier.
    Second, the archived accuracies are not comparable to anything computed
    here anyway -- they were measured on raw subspaces of unequal dimension,
    which is the defect this construction exists to remove -- so matching the
    classifier would buy a comparability that is not available. What matters is
    that the SAME probe scores the observed and null arms, and it does.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    keep = labels >= 0
    X, labels = X[keep], labels[keep]
    classes = np.unique(labels)
    if classes.size < 2:
        raise NullRefused(
            f"a probe needs at least two clusters; {classes.size} present "
            f"after dropping noise (label < 0)")
    if X.shape[1] == 0:
        raise NullRefused(
            "a probe cannot be fit in a zero-dimensional subspace; an empty "
            "channel is a finding about the OV circuit, not chance accuracy")

    counts = np.array([np.sum(labels == c) for c in classes])
    folds = min(n_folds, int(counts.min()))
    if folds < 2:
        raise NullRefused(
            f"the smallest cluster has {int(counts.min())} member(s); "
            f"cross-validation needs at least two per class")

    rng = np.random.default_rng(seed)
    fold_idx = _stratified_folds(labels, folds, rng)
    Y = (labels[:, None] == classes[None, :]).astype(np.float64)

    correct = 0
    total = 0
    for f in range(folds):
        test = fold_idx[f]
        train = np.concatenate([fold_idx[g] for g in range(folds) if g != f])
        Xtr, Xte = X[train], X[test]
        mu = Xtr.mean(axis=0)
        Xtr, Xte = Xtr - mu, Xte - mu
        G = Xtr.T @ Xtr
        ridge = ridge_fraction * float(np.trace(G)) / max(X.shape[1], 1)
        if ridge <= 0:
            ridge = ridge_fraction
        W = np.linalg.solve(G + ridge * np.eye(X.shape[1]), Xtr.T @ Y[train])
        pred = classes[np.argmax(Xte @ W, axis=1)]
        correct += int(np.sum(pred == labels[test]))
        total += test.size
    return correct / total if total else float("nan")


def r4_layer_accuracy(X: np.ndarray, labels: Sequence[int], U: np.ndarray
                      ) -> float:
    """Probe accuracy inside subspace U. Named an accuracy and not a contrast
    because that is what it is -- the contrast is formed against the null arm,
    not inside this function. Coordinates, not the ambient embedding
    -- fitting in R^d after projecting would give the probe back the dimensions
    the projection was supposed to remove."""
    return probe_accuracy(np.asarray(X, dtype=np.float64) @ U, labels)


def p_value_p6_r4(activations: Sequence[np.ndarray],
                  labels: Sequence[Sequence[int]],
                  channels: Sequence[LayerChannels],
                  unit: str,
                  alpha: float = 0.05) -> dict:
    """
    P6-R4's p-value: does the real channel carry cluster membership that a
    matched-dimension random subspace does not?

    Statistic: mean over layers of accuracy(U_S) minus accuracy(random U of the
    same dimension). The null redraws the random arm, so what is being tested is
    the operator content of U_S and nothing about how many dimensions it has.
    """
    _check_unit(unit)
    activations = list(activations)
    labels = list(labels)
    channels = list(channels)
    if not (len(activations) == len(labels) == len(channels)):
        raise NullRefused(
            f"{len(activations)} activation blocks, {len(labels)} label blocks "
            f"and {len(channels)} channel sets index the same layers and must "
            f"match")
    if not activations:
        raise NullRefused("no layers supplied")
    for i, ch in enumerate(channels):
        if ch.u_s.shape[1] == 0:
            raise NullRefused(
                f"layer {i} has an empty real channel (dim U_S = 0); there is "
                f"no subspace to probe")

    observed = float(np.mean([
        r4_layer_accuracy(X, y, ch.u_s)
        for X, y, ch in zip(activations, labels, channels)]))

    null_values: List[float] = []
    for draw in range(N_NULL_DRAWS):
        rngs = _draw_rngs(unit, len(channels), draw)
        vals = []
        for X, y, ch, rng in zip(activations, labels, channels, rngs):
            U = random_subspace(ch.d_model, ch.u_s.shape[1], rng)
            vals.append(r4_layer_accuracy(X, y, U))
        null_values.append(float(np.mean(vals)))

    return _finish(observed, null_values, unit, alpha, {
        "prediction_id": "P6-R4",
        "statistic": "mean over layers of cross-validated probe accuracy "
                     "inside U_S",
        "n_layers": len(channels),
        "n_units": 1 if unit == "model" else len(channels),
        "dims": [ch.dims() for ch in channels],
    })


# ---------------------------------------------------------------------------
# Adjudication (opt-in, and currently refused by construction)
# ---------------------------------------------------------------------------

def adjudicate_p6_r2_r4(result: dict,
                        artifact_hashes: Sequence[str] = (),
                        run_manifest: Optional[dict] = None,
                        adjudicate: bool = False,
                        adjudications_dir=None) -> Optional[dict]:
    """
    Write `result` into the falsification ledger, if asked and if allowed.

    Opt-in behind a flag, like every other adjudicating site in this project:
    these functions are exercised by tests and fixtures, and
    `core.adjudication.adjudicate` refuses to overwrite an existing record, so
    one accidental fixture run would permanently occupy P6-R2's slot with a
    synthetic p-value.

    It refuses on top of that while `REGISTERED_EXCHANGEABLE_UNIT` is None, and
    the refusal is deliberately NOT satisfiable by an argument. Which unit may
    enter a claim's e-process is a scientific decision, it has not been made,
    and letting a caller supply it would make it a per-run choice -- which is
    the selection this whole apparatus exists to prevent.
    """
    if not adjudicate:
        return None
    if result.get("p_value") is None:
        return None

    if REGISTERED_EXCHANGEABLE_UNIT is None:
        raise NullRefused(
            f"{result.get('prediction_id')} cannot be adjudicated: no "
            f"exchangeable unit is registered. This construction can compute a "
            f"p under either unit and they differ by orders of magnitude, so "
            f"which one counts as evidence has to be fixed BEFORE a number is "
            f"seen. Set REGISTERED_EXCHANGEABLE_UNIT here, record it in the "
            f"registry's null_construction, and note it in POPPER_PLAN -- in "
            f"that order. Passing unit= does not substitute: that argument "
            f"chooses what to COMPUTE, not what may enter an e-process.")
    if result.get("exchangeable_unit") != REGISTERED_EXCHANGEABLE_UNIT:
        raise NullRefused(
            f"result was computed under unit "
            f"{result.get('exchangeable_unit')!r} but "
            f"{REGISTERED_EXCHANGEABLE_UNIT!r} is registered")

    from core.adjudication import adjudicate_if_registered
    return adjudicate_if_registered(
        result["prediction_id"],
        result["p_value"],
        artifact_hashes=artifact_hashes,
        run_manifest=run_manifest,
        test_name=(f"matched-dimension random-subspace null, one-sided "
                   f"{ALTERNATIVE}, unit={result['exchangeable_unit']}"),
        adjudications_dir=adjudications_dir,
    )
