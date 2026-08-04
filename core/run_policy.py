"""
core/run_policy.py — the policy decisions as one declared, recorded,
verified object (DESIGN_pythia_frames.md policies P1-P5, plus item 13).

A policy item has no patch. It is a decision currently being made
implicitly, and possibly differently, at every call site. Code cannot make
the decision. It can do the three things that turn a decision into a result
worth trusting:

  1. refuse to run until the decision is stated      -> require_declared
  2. carry the statement into every artifact         -> to_manifest_extra
  3. refuse to compare runs that decided differently -> verify_same_policy

Deliberately NOT a second frame ledger
--------------------------------------
pos0 already lives in `core.frames.FrameSpec`, because whether position 0
is in or out is a per-analysis choice: the fidelity oracle and the Delta-x
exactness checks need every token by construction, while clustering and
energy may not want the sink. `RunPolicy` carries the run-level DEFAULT and
`assert_agrees_with_frame` pins the two together, so a FrameSpec written
under a policy the run never declared is a caught error. Two ledgers free
to disagree would be the same bug class this module exists to close.

Why UNSET rather than a default
-------------------------------
`FrameSpec.pos0_policy` defaults to "included". Every record written today
therefore asserts inclusion whether or not anyone considered the question,
and a call site that starts masking without threading the field writes a
record that lies in the one direction the ledger cannot catch. RunPolicy
has no defaults for the four live decisions. A run that has not stated them
does not start.

See also
--------
core.sink_audit       — the measurement that decides P1
core.precision_policy — the measurement that decides P2 and item 13
core.checkpoint_frames— P3's detector (log-step intervals)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict, replace
from typing import Any, Mapping, Sequence

#: Sentinel for "this decision has not been made". Not a valid value; every
#: consumer either sees a real choice or raises.
UNSET = "__unset__"

POS0_POLICIES = ("included", "excluded")

#: P5. "published_step0" is the real init checkpoint and is a MODEL, not a
#: randomization scheme — it is listed here so a run can declare that its
#: untrained arm is step 0 rather than a re-randomized final checkpoint.
#: They are different objects and answer different claims (PREDICTIONS.md
#: (a) wants step 0; (c) wants whatever gpt2-large-random was).
RANDOM_SCHEMES = ("orthogonal", "gaussian", "published_step0", "none")

#: The fallback every call site applies today:
#:     cfg.get("random_init_scheme", "orthogonal")
#: `gpt2-large-random` omits the key, so Blog 1's random control IS
#: orthogonal randomization of trained weights. That fact, not a judgment
#: call, settles what the Pythia continuity control has to be.
DEFAULT_RANDOM_SCHEME = "orthogonal"

WEIGHT_PRECISIONS = ("as_stored", "float64_from_fp32")
ACTIVATION_PRECISIONS = ("autocast_bf16", "float32_no_autocast")
TRANSITION_AXES = ("log_step", "checkpoint_index")

#: The four fields a gated run must state. `transition_axis` is not here:
#: it has a correct answer (P3), so it carries a default and is recorded
#: only so an index-based run is visible rather than assumed away.
REQUIRED_FIELDS = ("pos0", "random_scheme", "weight_precision",
                   "activation_precision")


class PolicyError(ValueError):
    """Base for both failure modes below."""


class PolicyUndeclared(PolicyError):
    """A run reached a gated entry point without stating its policy."""


class PolicyMismatch(PolicyError):
    """Two records made different policy choices and were about to be compared."""


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPolicy:
    """
    One run's answers to P1-P5 and item 13.

    pos0                 : P1. Run-level default; individual FrameSpecs may
                           differ (exactness checks need "included"), but a
                           differing spec must be deliberate — see
                           `assert_agrees_with_frame`.
    random_control       : P5. MODEL_CONFIGS key of the untrained arm, or
                           None when this run is not a trained-vs-random
                           contrast. A string, so the artifact names the
                           object rather than describing it.
    random_scheme        : P5. What that arm actually is. "norm_matched" is
                           not a member of RANDOM_SCHEMES because
                           core.models.randomize_weights does not implement
                           it and never did.
    weight_precision     : P2.
    activation_precision : item 13.
    transition_axis      : P3.
    """

    pos0: str = UNSET
    random_control: Any = UNSET          # str | None
    random_scheme: str = UNSET
    weight_precision: str = UNSET
    activation_precision: str = UNSET
    transition_axis: str = "log_step"
    notes: str = ""

    def __post_init__(self):
        self._check("pos0", POS0_POLICIES)
        self._check("random_scheme", RANDOM_SCHEMES)
        self._check("weight_precision", WEIGHT_PRECISIONS)
        self._check("activation_precision", ACTIVATION_PRECISIONS)
        self._check("transition_axis", TRANSITION_AXES)

    def _check(self, field: str, allowed: Sequence[str]) -> None:
        value = getattr(self, field)
        if value == UNSET:
            return
        if value not in allowed:
            raise PolicyError(
                f"RunPolicy.{field}: expected one of {tuple(allowed)} or UNSET, "
                f"got {value!r}"
            )

    # -- declaration --------------------------------------------------------

    def undeclared_fields(self) -> list:
        return [f for f in REQUIRED_FIELDS if getattr(self, f) == UNSET]

    def require_declared(self, context: str = "") -> "RunPolicy":
        """
        Raise unless every gated decision has been stated. Call this at the
        top of every `run_*.py` main, before any model is loaded — the cost
        of stopping here is seconds, and the cost of not stopping here is a
        sweep whose arms are not comparable.
        """
        missing = self.undeclared_fields()
        if missing:
            where = f" [{context}]" if context else ""
            raise PolicyUndeclared(
                f"RunPolicy: undeclared {missing}{where}. These are decisions, "
                f"not defaults — see DESIGN_pythia_frames.md policies P1, P2, "
                f"P5 and item 13. Pass them explicitly (core.run_policy."
                f"add_policy_args wires the CLI flags)."
            )
        return self

    # -- identity -----------------------------------------------------------

    def policy_key(self) -> str:
        """
        Short deterministic digest of the decisions, for `manifest_id`.

        Without this, the two arms of an S3 dual-run differ in no field that
        `io.compute_manifest_id` hashes, so both arms get the SAME manifest id
        and their stamped figures collide — the diff that is supposed to
        settle the policy overwrites itself.
        """
        parts = [str(getattr(self, f)) for f in
                 REQUIRED_FIELDS + ("random_control", "transition_axis")]
        return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:8]

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_manifest_extra(self) -> dict:
        """Merge into `core.io.write_manifest(..., extra=...)`."""
        return {"policy": self.to_dict(), "policy_key": self.policy_key()}

    @classmethod
    def from_dict(cls, d: Mapping) -> "RunPolicy":
        known = {k: v for k, v in dict(d).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_manifest(cls, manifest: Mapping) -> "RunPolicy":
        block = (manifest or {}).get("policy")
        if block is None:
            raise PolicyUndeclared(
                "manifest has no 'policy' block. A run written before "
                "core.run_policy landed cannot have its policy verified; "
                "re-run it rather than assuming the current default."
            )
        return cls.from_dict(block)

    def replace(self, **kw) -> "RunPolicy":
        return replace(self, **kw)

    # -- guards -------------------------------------------------------------

    def assert_agrees_with_frame(self, spec, context: str = "") -> None:
        """
        A FrameSpec's pos0_policy must match the run's, unless the caller
        deliberately built a differing spec and said so in `notes`.

        The intended exception is narrow and real: `tests/test_forward_
        fidelity.py` and the Delta-x exactness checks need every token even
        in an "excluded" run, because they reconstruct the forward pass.
        Anything else that differs is a bug.
        """
        got = getattr(spec, "pos0_policy", None)
        if got is None:
            raise PolicyMismatch(
                f"assert_agrees_with_frame: {spec!r} carries no pos0_policy"
                + (f" [{context}]" if context else "")
            )
        if got != self.pos0:
            where = f" [{context}]" if context else ""
            raise PolicyMismatch(
                f"FrameSpec pos0_policy={got!r} but the run declared "
                f"{self.pos0!r}{where}. Exactness checks may legitimately "
                f"differ — construct those specs through an explicitly named "
                f"helper and record why in RunPolicy.notes, so the exception "
                f"is visible in the artifact."
            )


def verify_same_policy(a: RunPolicy, b: RunPolicy,
                       fields: Sequence[str] = REQUIRED_FIELDS,
                       context: str = "") -> None:
    """
    Refuse to compare two runs that answered a policy question differently.

    Written on the model of `core.frames.verify_same_frame` and
    `core.prompts.verify_same_battery`: raises rather than warns. The
    invariant this protects is the one every policy item shares — the same
    choice, applied to BOTH arms of every contrast and every checkpoint. A
    trained-vs-random energy contrast where one arm excluded the sink and
    the other did not is a sink contrast wearing a training label.
    """
    diffs = [(f, getattr(a, f), getattr(b, f)) for f in fields
             if getattr(a, f) != getattr(b, f)]
    if diffs:
        where = f" [{context}]" if context else ""
        detail = "; ".join(f"{f}: {x!r} vs {y!r}" for f, x, y in diffs)
        raise PolicyMismatch(
            f"verify_same_policy: records made different decisions{where} — "
            f"{detail}. These records are not comparable."
        )


def verify_all_same_policy(records: Sequence[Mapping], context: str = "") -> RunPolicy:
    """Every manifest in a sweep must share one policy. Returns it."""
    if not records:
        raise PolicyError("verify_all_same_policy: no records")
    first = RunPolicy.from_manifest(records[0])
    for r in records[1:]:
        verify_same_policy(first, RunPolicy.from_manifest(r), context=context)
    return first


# ---------------------------------------------------------------------------
# P5 — continuity of the random control, derived rather than asserted
# ---------------------------------------------------------------------------

def effective_random_scheme(cfg: Mapping) -> str:
    """
    The scheme a run WOULD actually use for this MODEL_CONFIGS entry,
    including the fallback every call site applies:

        cfg.get("random_init_scheme", "orthogonal")

    `gpt2-large-random` omits the key. Its effective scheme is therefore
    "orthogonal", and that is the fact P5 turns on: continuity with Blog 1
    means matching the object that ran, not choosing the better control.
    """
    return str(cfg.get("random_init_scheme", DEFAULT_RANDOM_SCHEME))


def assert_matched_random_scheme(model_configs: Mapping,
                                 reference: str,
                                 candidate: str,
                                 context: str = "") -> str:
    """
    The Blog-1 continuity claim, mechanized: the Pythia random control must
    be the same KIND of object as the GPT-2 one, or PREDICTIONS.md claim (c)
    is comparing two different controls and calling the difference
    architecture.

    Derives the requirement from the registry rather than hardcoding
    "orthogonal", so changing the GPT-2 control moves the requirement with
    it instead of silently breaking the claim.
    """
    for name in (reference, candidate):
        if name not in model_configs:
            raise PolicyError(
                f"assert_matched_random_scheme: {name!r} is not in "
                f"MODEL_CONFIGS. The continuity control has to exist before "
                f"a prediction can name it."
            )
    ref = effective_random_scheme(model_configs[reference])
    cand = effective_random_scheme(model_configs[candidate])
    if ref != cand:
        where = f" [{context}]" if context else ""
        raise PolicyMismatch(
            f"random control mismatch{where}: {reference!r} uses {ref!r} but "
            f"{candidate!r} uses {cand!r}. Blog 1's trained-vs-random contrast "
            f"is only continuous across architectures if both untrained arms "
            f"are the same construction."
        )
    return ref


# ---------------------------------------------------------------------------
# CLI boundary — where the human is, and therefore where the default dies
# ---------------------------------------------------------------------------

def add_policy_args(parser, required: bool = True) -> None:
    """
    Add the policy flags to a `run_*.py` argparse parser.

    `required=True` on --pos0-policy is intentional and WILL break existing
    invocations. That is the point: a bare `python -m p1_mstate_tracking.run_1`
    currently makes the P1 decision silently, in the "included" direction,
    for every figure the sweep produces. Pass `required=False` only for
    exploratory local runs whose output is not going into a result record.
    """
    g = parser.add_argument_group("policy (DESIGN_pythia_frames.md P1-P5)")
    g.add_argument("--pos0-policy", choices=POS0_POLICIES, required=required,
                   help="P1: include or exclude position 0, the NeoX attention sink.")
    g.add_argument("--random-control", default=None,
                   help="P5: MODEL_CONFIGS key of the untrained arm, if any.")
    g.add_argument("--random-scheme", choices=RANDOM_SCHEMES, default="none",
                   help="P5: what that arm is. 'published_step0' is a model, not a scheme.")
    g.add_argument("--weight-precision", choices=WEIGHT_PRECISIONS,
                   default="as_stored", help="P2.")
    g.add_argument("--activation-precision", choices=ACTIVATION_PRECISIONS,
                   default="autocast_bf16", help="Item 13.")
    g.add_argument("--policy-notes", default="",
                   help="Free text; use it to record a deliberate exception.")


def policy_from_args(args) -> RunPolicy:
    return RunPolicy(
        pos0=getattr(args, "pos0_policy", UNSET) or UNSET,
        random_control=getattr(args, "random_control", None),
        random_scheme=getattr(args, "random_scheme", UNSET),
        weight_precision=getattr(args, "weight_precision", UNSET),
        activation_precision=getattr(args, "activation_precision", UNSET),
        notes=getattr(args, "policy_notes", ""),
    )


def policy_summary_lines(p: RunPolicy) -> list:
    """Paste into a status doc or a run log."""
    ctrl = p.random_control if p.random_control not in (None, UNSET) else "none"
    return [
        "Run policy:",
        f"  P1 pos0                : {p.pos0}",
        f"  P5 random control      : {ctrl}  (scheme={p.random_scheme})",
        f"  P2 weight precision    : {p.weight_precision}",
        f"  13 activation precision: {p.activation_precision}",
        f"  P3 transition axis     : {p.transition_axis}",
        f"  key                    : {p.policy_key()}",
    ] + ([f"  notes                  : {p.notes}"] if p.notes else [])
