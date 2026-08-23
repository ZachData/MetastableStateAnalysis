"""
core/evalues.py — the Popperian adjudication kernel (POPPER_PLAN.md item B1).

Implements the statistical core of POPPER (Huang, Jin, Li, Li, Candès &
Leskovec, "Automated Hypothesis Validation with Agentic Sequential
Falsifications", arXiv:2502.09858): a p-to-e calibrator, a product e-process
over a sequence of falsification experiments, and the anytime-valid decision
rule that turns accumulated evidence into a validation status with a Type-I
error guarantee.

What this module is for
-----------------------
This project already pre-registers predictions with falsifiers and instruments
(`PREDICTIONS.md`), which is the hard part and the part most work skips. What
prose cannot supply is a *rate*: how often a procedure that adjudicates many
predictions against many artifacts declares a claim supported when it is not.
An e-process supplies exactly that, and supplies it under the two conditions
that actually hold here and break the classical alternatives:

  1. **Optional stopping.** This project stops when it stops. `PREDICTIONS.md`
     attaches a hard stop to claim (c); `INDEX.md` records three phases going
     out of scope in a single day. A p-value combination (Fisher, Brown) is
     invalid the moment the number of tests depends on the results, which is
     the project's actual operating mode. An e-process is valid under any
     stopping time (Assumption 3 in the paper; Doob's optional stopping).
  2. **Dependent experiments.** Predictions here are re-analyses of overlapping
     artifacts, so their p-values are not independent. E-values need no
     independence -- only that each is conditionally calibrated given what came
     before (Assumption 2). See the WARNING in `EProcess` about what that does
     and does not permit.

Scope: pure numpy + stdlib, no project imports, no torch. This module knows
nothing about predictions, artifacts, or phases -- `core/adjudication.py`
(item B4) is the layer that knows those things and calls this one. Keeping the
arithmetic separate from the bookkeeping is deliberate: this is the only piece
of the workstream with a proof attached, and it should be testable without
constructing a registry.

The mathematics, stated once
----------------------------
An **e-value** is a non-negative random variable with E[e] <= 1 under the null.
Large values are evidence against the null. Given a p-value that is valid under
the null (P(p <= t) <= t for all t in [0,1]), Vovk & Wang's calibrator

    e = kappa * p ** (kappa - 1),    kappa in (0, 1)

is an e-value for any fixed kappa. Proof, since it is two lines and worth
having next to the code: for p ~ U(0,1),

    E[e] = int_0^1 kappa * u^(kappa-1) du = [u^kappa]_0^1 = 1,

and for a p-value that is merely valid (stochastically larger than uniform)
the expectation is <= 1 because e is decreasing in p. Equality at uniform is
what makes this calibrator admissible rather than merely conservative.

Aggregation is by product: E_i = prod_{s<=i} e_s. Under the assumptions above
{E_i} is a non-negative supermartingale, so by Doob's optional stopping theorem
E[E_tau] <= 1 at any stopping time tau, and Markov's inequality gives

    P(E_tau >= 1/alpha) <= alpha    under the null.

That last line is the whole guarantee: declaring a claim supported when the
accumulated evidence crosses 1/alpha has Type-I error at most alpha, whenever
you stop and however you chose to stop.

Two consequences that are easy to miss and change how results read
------------------------------------------------------------------
* **A non-falsification counts against the claim.** At p = 1, e = kappa < 1, so
  the running product *shrinks*. An experiment that fails to falsify a null is
  not neutral; it is evidence the hypothesis is wrong. This is the property
  that makes the process honest, and it is visible in the paper's own Table 1
  (round 1: p = 1.0, cumulative e-value 0.5), which is also how kappa = 0.5 is
  confirmed as their setting.
* **The product is only as valid as its weakest factor.** One e-value derived
  from a mis-specified null voids the guarantee for the entire claim, not just
  for itself. The paper measures this: removing the relevance checker, which
  exists solely to keep irrelevant nulls out, raises Type-I error from 0.082 to
  0.340 on TargetVal-IL2. `core/adjudication.py` must therefore refuse to emit
  an e-value for a prediction whose null is not established, rather than
  emitting a neutral one -- an instance of the project's standing rule 4,
  "refuse rather than degrade" (`UPDATE_PLAN.md` §6).

References
----------
Vovk & Wang (2021), "E-values: Calibration, combination, and applications",
Ann. Statist. 49(3) -- the calibrator.
Grünwald, de Heide & Koolen (2020), "Safe testing" -- anytime validity.
Huang et al. (2025), arXiv:2502.09858 -- the framework this instantiates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Fixed calibrator parameter. MUST be chosen before seeing any p-value:
#: picking kappa after the fact turns a calibrated e-value into a selected
#: one and voids E[e] <= 1. 0.5 matches POPPER's own setting (their Table 1
#: shows p = 1.0 -> cumulative e = 0.5). Recorded per registry entry so a
#: future change is visible in the record rather than silent in a default.
DEFAULT_KAPPA = 0.5

#: Nominal Type-I error level. POPPER's evaluations use 0.1; this project's
#: claims are load-bearing enough to warrant the stricter conventional level.
#: Declared here so every caller reads one number, and overridable per call.
DEFAULT_ALPHA = 0.05


class EValueError(ValueError):
    """Raised on inputs that would silently invalidate the guarantee."""


# ---------------------------------------------------------------------------
# The calibrator
# ---------------------------------------------------------------------------

def calibrate(p: float, kappa: float = DEFAULT_KAPPA) -> float:
    """
    Vovk-Wang p-to-e calibrator: ``e = kappa * p ** (kappa - 1)``.

    Parameters
    ----------
    p : float
        A p-value that is *valid under the null* -- P(p <= t) <= t. Validity is
        this function's precondition and it cannot be checked here; it is
        established by whatever produced p (a permutation null, an exact test,
        a bootstrap with its resolution floor stated). Feeding this an
        uncalibrated "score that looks like a p-value" produces a number with
        no guarantee attached and is the single most likely way to break the
        whole procedure.
    kappa : float
        Calibrator parameter in (0, 1). Must be fixed in advance; see
        DEFAULT_KAPPA.

    Returns
    -------
    float
        The e-value. ``math.inf`` when ``p == 0``.

    Raises
    ------
    EValueError
        On p outside [0, 1], kappa outside the open interval (0, 1), or NaN in
        either. Refusing rather than clipping is deliberate: a clipped p-value
        yields a finite e-value that looks ordinary in the artifact and cannot
        be distinguished later from a real one.

    Notes
    -----
    ``p == 0`` returns infinity rather than raising. An exact zero is a real
    outcome of some exact tests, and infinity is the mathematically correct
    e-value; it is also a loud signal in any downstream report. A permutation
    test, by contrast, should never report exactly 0 -- its resolution floor is
    1/(n_perm + 1) -- so a 0 arriving from one is a bug in the caller, which
    `p_from_null` (item B6) enforces at the source rather than here.
    """
    if kappa is None or not math.isfinite(kappa) or not (0.0 < kappa < 1.0):
        raise EValueError(f"kappa must be a finite value in (0, 1); got {kappa!r}")
    if p is None or (isinstance(p, float) and math.isnan(p)):
        raise EValueError("p-value is NaN; refusing to calibrate")
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise EValueError(f"p-value must lie in [0, 1]; got {p!r}")
    if p == 0.0:
        return math.inf
    return kappa * p ** (kappa - 1.0)


def log_calibrate(p: float, kappa: float = DEFAULT_KAPPA) -> float:
    """
    ``log(calibrate(p, kappa))``, computed without forming the ratio.

    Long e-processes overflow in the direct product well before they lose
    precision in the log, and this project's claims are expected to accumulate
    many predictions each. `EProcess` accumulates in log space for that reason
    and exposes the linear value only where it is representable.
    """
    if kappa is None or not math.isfinite(kappa) or not (0.0 < kappa < 1.0):
        raise EValueError(f"kappa must be a finite value in (0, 1); got {kappa!r}")
    if p is None or (isinstance(p, float) and math.isnan(p)):
        raise EValueError("p-value is NaN; refusing to calibrate")
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise EValueError(f"p-value must lie in [0, 1]; got {p!r}")
    if p == 0.0:
        return math.inf
    return math.log(kappa) + (kappa - 1.0) * math.log(p)


def sufficient_evidence(E: float, alpha: float = DEFAULT_ALPHA) -> bool:
    """
    POPPER's decision rule: reject the null (validate the hypothesis) when the
    accumulated evidence reaches ``1 / alpha``.

    Valid at any stopping time, which is the point -- see the module docstring.
    """
    if not (0.0 < alpha < 1.0):
        raise EValueError(f"alpha must lie in (0, 1); got {alpha!r}")
    return bool(E >= 1.0 / alpha)


def required_p_for_rejection(
    alpha: float = DEFAULT_ALPHA,
    kappa: float = DEFAULT_KAPPA,
    log_E_prior: float = 0.0,
) -> float:
    """
    The largest p-value the *next* experiment could return and still push the
    accumulated evidence across the rejection threshold.

    This is a planning instrument, not an inferential one. It is cheap, it
    needs no data, and it answers the question worth asking before spending a
    forward-pass budget: given what has accumulated so far, is the experiment
    being contemplated even capable of settling the claim? An experiment whose
    best attainable p-value -- a permutation test's 1/(n_perm + 1) floor, say --
    exceeds this number cannot cross the threshold no matter how the data fall,
    and should be re-powered or re-scoped before it is run rather than after.

    Parameters
    ----------
    log_E_prior : float
        Natural log of the evidence accumulated so far. 0.0 for a fresh claim.

    Returns
    -------
    float
        The threshold p, clipped into [0, 1]. Returns 1.0 when the accumulated
        evidence already crosses the threshold on its own (any p would do), and
        0.0 when no attainable p-value could (the claim cannot be settled by
        one further experiment).
    """
    if not (0.0 < alpha < 1.0):
        raise EValueError(f"alpha must lie in (0, 1); got {alpha!r}")
    if not (0.0 < kappa < 1.0):
        raise EValueError(f"kappa must lie in (0, 1); got {kappa!r}")

    # Need: log_E_prior + log(kappa) + (kappa - 1) log p >= log(1/alpha)
    # (kappa - 1) < 0, so dividing flips the inequality:
    #   log p <= (log(1/alpha) - log_E_prior - log(kappa)) / (kappa - 1)
    needed = math.log(1.0 / alpha) - log_E_prior - math.log(kappa)
    log_p = needed / (kappa - 1.0)
    if log_p >= 0.0:
        return 1.0
    try:
        return float(min(1.0, max(0.0, math.exp(log_p))))
    except OverflowError:                       # pragma: no cover - defensive
        return 0.0


# ---------------------------------------------------------------------------
# The e-process
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Adjudication:
    """One falsification experiment's contribution to a claim."""
    prediction_id: str
    p_value: float
    e_value: float
    log_e_value: float
    kappa: float


@dataclass
class EProcess:
    """
    A product e-process over the falsification experiments for one claim.

    Accumulates in log space; ``E`` reconstitutes the linear value where it is
    representable and returns ``inf`` where it is not.

    WARNING -- what this class cannot check, and what breaks if you get it wrong
    ---------------------------------------------------------------------------
    Validity requires ``E[e_i | D_{i-1}] <= 1`` (the paper's Assumption 2): each
    e-value must be calibrated *conditional on everything already seen*. Two
    ways to violate it, neither detectable from the numbers here:

    * **Choosing a prediction after seeing the artifact it will be tested on.**
      Re-using one artifact across predictions is fine on its own -- e-values
      need no independence. What is not fine is registering a prediction after
      looking at that artifact and then adjudicating it against the same
      artifact. `tools/check_preregistration.py` (item B3) is the only thing
      that detects this, by comparing git history rather than data.
    * **Admitting an e-value from a null that is not valid.** The product is
      only as valid as its weakest factor. `core/adjudication.py` (item B4)
      gates this by refusing predictions the evaluability audit
      (`claims/EVALUABILITY.md`, item B5) did not classify as ``e-value``.

    Both are enforced upstream of this class, deliberately: this one owns the
    arithmetic and should stay small enough to be obviously correct.
    """

    claim: str
    alpha: float = DEFAULT_ALPHA
    adjudications: List[Adjudication] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise EValueError(f"alpha must lie in (0, 1); got {self.alpha!r}")

    # -- accumulation ------------------------------------------------------

    def add(
        self,
        prediction_id: str,
        p_value: float,
        kappa: float = DEFAULT_KAPPA,
    ) -> Adjudication:
        """
        Calibrate one p-value and append it to the process.

        Order matters for the record (an e-process is a sequence, and the
        interim ``E`` after each step is what optional stopping is valid
        against), though the final product does not depend on it.

        Raises
        ------
        EValueError
            If ``prediction_id`` already contributed. A prediction adjudicated
            twice would multiply its own evidence in, which inflates E without
            any new experiment behind it -- the arithmetic form of double
            counting. Re-adjudicating against new data belongs to a *new*
            prediction id with its own registry entry and its own
            pre-registration timestamp.
        """
        if any(a.prediction_id == prediction_id for a in self.adjudications):
            raise EValueError(
                f"prediction {prediction_id!r} has already contributed to claim "
                f"{self.claim!r}; re-adjudication needs a new registered "
                f"prediction id, not a second entry for this one"
            )
        adj = Adjudication(
            prediction_id=prediction_id,
            p_value=float(p_value),
            e_value=calibrate(p_value, kappa),
            log_e_value=log_calibrate(p_value, kappa),
            kappa=float(kappa),
        )
        self.adjudications.append(adj)
        return adj

    # -- readout -----------------------------------------------------------

    @property
    def log_E(self) -> float:
        """Natural log of the accumulated evidence. 0.0 when empty."""
        total = 0.0
        for a in self.adjudications:
            if math.isinf(a.log_e_value):
                return math.inf
            total += a.log_e_value
        return total

    @property
    def E(self) -> float:
        """
        Accumulated evidence. ``inf`` when the log overflows the float range,
        which is a real outcome once enough strong experiments accumulate and
        is not an error.
        """
        lg = self.log_E
        if math.isinf(lg):
            return math.inf
        try:
            return math.exp(lg)
        except OverflowError:                   # pragma: no cover - defensive
            return math.inf

    @property
    def trajectory(self) -> List[float]:
        """
        Running ``log_E`` after each experiment, in order.

        This is the object worth plotting -- POPPER's Figure 3 panel 4 shows
        exactly this, and the shape carries information the endpoint does not:
        evidence that accumulates steadily reads differently from evidence
        that arrives entirely from one experiment, even at the same final E.
        """
        out: List[float] = []
        total = 0.0
        for a in self.adjudications:
            if math.isinf(a.log_e_value) or math.isinf(total):
                total = math.inf
            else:
                total += a.log_e_value
            out.append(total)
        return out

    def decision(self, alpha: Optional[float] = None) -> str:
        """
        ``"reject_null"`` when the evidence crosses 1/alpha, else
        ``"insufficient_evidence"``.

        Deliberately not ``"accept_null"``. Failing to accumulate evidence
        against a null is not evidence for it, and naming the outcome
        "insufficient evidence" keeps the Popperian asymmetry visible in the
        artifact rather than only in the prose around it.
        """
        a = self.alpha if alpha is None else alpha
        return "reject_null" if sufficient_evidence(self.E, a) else "insufficient_evidence"

    def next_p_needed(self, alpha: Optional[float] = None, kappa: float = DEFAULT_KAPPA) -> float:
        """``required_p_for_rejection`` given what has accumulated so far."""
        a = self.alpha if alpha is None else alpha
        return required_p_for_rejection(alpha=a, kappa=kappa, log_E_prior=self.log_E)

    # -- serialization -----------------------------------------------------

    def to_record(self) -> dict:
        """
        JSON-serialisable record. Shape chosen so CI can recompute ``E`` from
        the committed p-values and fail if the stored decision disagrees
        (item B7) -- the committed artifact verifies itself rather than being
        taken on trust.
        """
        return {
            "claim": self.claim,
            "alpha": self.alpha,
            "threshold": 1.0 / self.alpha,
            "n_experiments": len(self.adjudications),
            "log_E": self.log_E,
            "E": self.E,
            "decision": self.decision(),
            "experiments": [
                {
                    "prediction_id": a.prediction_id,
                    "p_value": a.p_value,
                    "e_value": a.e_value,
                    "log_e_value": a.log_e_value,
                    "kappa": a.kappa,
                }
                for a in self.adjudications
            ],
        }

    @classmethod
    def from_record(cls, record: dict) -> "EProcess":
        """
        Rebuild from ``to_record`` output, recalibrating each p-value rather
        than trusting the stored e-value.

        Recalibrating is the point: a record whose stored ``e_value`` does not
        match ``calibrate(p_value, kappa)`` has been edited by hand, and a
        round-trip that silently preserved the stored number would hide exactly
        that.
        """
        proc = cls(claim=record["claim"], alpha=float(record.get("alpha", DEFAULT_ALPHA)))
        for exp in record.get("experiments", []):
            proc.add(
                prediction_id=exp["prediction_id"],
                p_value=float(exp["p_value"]),
                kappa=float(exp.get("kappa", DEFAULT_KAPPA)),
            )
        return proc


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def combine(
    p_values: Sequence[float],
    kappa: float = DEFAULT_KAPPA,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[float, bool]:
    """
    ``(E, reject)`` for a sequence of p-values. Convenience for tests and for
    one-off checks; real adjudication goes through `EProcess` so the per-step
    trajectory and the prediction ids are recorded.
    """
    log_E = 0.0
    for p in p_values:
        le = log_calibrate(p, kappa)
        if math.isinf(le):
            return math.inf, True
        log_E += le
    E = math.inf if log_E > 709.0 else math.exp(log_E)
    return E, sufficient_evidence(E, alpha)


def simulate_type_i_error(
    n_trials: int = 20_000,
    n_experiments: int = 5,
    alpha: float = DEFAULT_ALPHA,
    kappa: float = DEFAULT_KAPPA,
    seed: int = 0,
) -> float:
    """
    Empirical Type-I error of the procedure under the null, by simulation with
    uniform p-values.

    Exists so the guarantee is checked rather than cited: the test suite asserts
    the returned rate is at or below ``alpha``. Mirrors the paper's own
    sensitivity analysis (their Figure 4, panel 1), which varies the nominal
    level and confirms empirical control at each.

    Note this simulates the *fixed-horizon* case. The optional-stopping case is
    covered separately in the tests by stopping at the first crossing, which is
    the adversarial choice for a procedure that claims anytime validity.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=(n_trials, n_experiments))
    log_e = math.log(kappa) + (kappa - 1.0) * np.log(p)
    log_E = log_e.sum(axis=1)
    return float(np.mean(log_E >= math.log(1.0 / alpha)))
