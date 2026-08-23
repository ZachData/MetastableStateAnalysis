"""
tests/test_pythia_registry.py — pure-logic tests for
core/pythia_registry.build_pythia_model_configs.

No model loading, no network: transformers' GPTNeoXModel/AutoTokenizer are
only imported for their identity (used as a dict value), never
instantiated here. Runs under the same stub-heavy-deps session as the rest
of the non-smoke suite.

Two cross-module contracts are checked by reading source rather than by
importing, and both reasons are session artifacts, not preferences:

  1. `core.models.randomize_weights` — conftest installs a hand-built
     `core.models` stub exposing four names, and this is not one of them.
  2. `p1_mstate_tracking.visualization.checkpoints` — importing it runs
     that package's `__init__`, which raises ModuleNotFoundError on
     `.style`. See the note at the bottom of this file; that breakage
     predates this file and is not fixed here.

Source-reading keeps the drift these tests exist to catch catchable. A
`pytest.importorskip` would turn both into silent no-ops, which is the
same failure mode as the bug being guarded against.
"""
import re
from pathlib import Path

import pytest

from core.pythia_registry import (
    build_pythia_model_configs,
    PYTHIA_410M_PILOT_STEPS,
    PYTHIA_1_4B_ANCHOR_STEPS,
    PYTHIA_1_4B_EXPENSIVE_STEPS,
    PYTHIA_ALL_STEPS,
    PYTHIA_RANDOM_MATCH_STEP,
)

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

STEP_RE = re.compile(r"^step\d+$")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINTS_PY = "p1_mstate_tracking/visualization/checkpoints.py"


def _read_source(relpath: str) -> str:
    path = _PROJECT_ROOT / relpath
    assert path.exists(), (
        f"{relpath} not found under {_PROJECT_ROOT} — this test asserts a "
        "cross-module contract and cannot pass vacuously if the other "
        "module moved"
    )
    return path.read_text()


def _step_re_pattern() -> str:
    """
    The checkpoint-step grammar, as the real compiled object.

    This used to scrape the `_STEP_RE = re.compile(...)` literal out of
    checkpoints.py source, because importing that module pulled in
    matplotlib through its package `__init__`. The grammar has since moved
    to `core/model_family.py` — which is stdlib-only for exactly this
    reason — and checkpoints.py now re-exports it as `_STEP_RE`. So the
    scrape found nothing and the assert fired.

    Importing the real object is strictly better than a regex over text:
    it tests the contract rather than the spelling.
    """
    from core.model_family import CHECKPOINT_STEP_RE
    return CHECKPOINT_STEP_RE.pattern


def _checkpoints_reexports_step_re() -> bool:
    """checkpoints.py must still expose the grammar under `_STEP_RE` — its
    four figure consumers call that name."""
    src = _read_source(_CHECKPOINTS_PY)
    return re.search(r"CHECKPOINT_STEP_RE as _STEP_RE", src) is not None


def _family_baselines(base: str, models):
    """
    Mirror of `checkpoints.family_baselines`, guarded against drift.

    The mirrored rule is exact-match on '{base}-random' and '{base}-step0',
    with no cross-family substitution. The assertions below fail if the
    real implementation stops being that rule, so this stays a test of the
    resolver rather than of a private copy of it.
    """
    src = _read_source(_CHECKPOINTS_PY)
    body = src[src.index("def family_baselines"):]
    body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    for expected in ('f"{base}-random"', 'f"{base}-step0"'):
        assert expected in body, (
            f"family_baselines no longer resolves via {expected}; this "
            "mirror is stale and the test must be rewritten against the "
            "new rule"
        )
    return {
        "random": f"{base}-random" if f"{base}-random" in models else None,
        "step0":  f"{base}-step0" if f"{base}-step0" in models else None,
    }


@pytest.fixture(scope="module")
def cfgs():
    return build_pythia_model_configs()


class TestRegistryCoverage:

    def test_one_entry_per_pilot_step(self, cfgs):
        for step in PYTHIA_410M_PILOT_STEPS:
            assert f"pythia-410m-step{step}" in cfgs

    def test_one_entry_per_anchor_step(self, cfgs):
        for step in PYTHIA_1_4B_ANCHOR_STEPS:
            assert f"pythia-1.4b-step{step}" in cfgs

    def test_expensive_tier_is_subset_of_anchors(self):
        """Plan: expensive-tier checkpoints are drawn from the anchor
        schedule, not a separate set — a mismatch here means the two
        tables drifted apart."""
        assert set(PYTHIA_1_4B_EXPENSIVE_STEPS) <= set(PYTHIA_1_4B_ANCHOR_STEPS)

    def test_anchor_steps_are_subset_of_all_published_steps(self):
        """Every anchor must be a step Pythia actually checkpointed at —
        an anchor not in PYTHIA_ALL_STEPS would 404 against the real repo."""
        assert set(PYTHIA_1_4B_ANCHOR_STEPS) <= set(PYTHIA_ALL_STEPS)

    def test_pilot_steps_are_subset_of_all_published_steps(self):
        assert set(PYTHIA_410M_PILOT_STEPS) <= set(PYTHIA_ALL_STEPS)


class TestRegistryEntryShape:

    def test_every_entry_has_required_keys(self, cfgs):
        """
        `checkpoint_step` is deliberately NOT in the common set. The random
        control is not a point on the training trajectory, so it carries no
        step; requiring one here would force a fabricated value onto the
        step axis of every checkpoint figure. It is required of every
        checkpoint entry below.
        """
        common = {"model_class", "tokenizer_class", "is_albert",
                  "random_init", "hf_repo", "revision"}
        for name, entry in cfgs.items():
            missing = common - entry.keys()
            assert not missing, f"{name} missing keys: {missing}"

    def test_every_checkpoint_entry_has_a_step(self, cfgs):
        for name, entry in cfgs.items():
            if "-step" in name:
                assert "checkpoint_step" in entry, f"{name} missing checkpoint_step"

    def test_revision_format(self, cfgs):
        for name, entry in cfgs.items():
            assert STEP_RE.match(entry["revision"]), (
                f"{name} has malformed revision {entry['revision']!r}"
            )

    def test_revision_matches_checkpoint_step(self, cfgs):
        for name, entry in cfgs.items():
            if "checkpoint_step" not in entry:
                continue
            assert entry["revision"] == f"step{entry['checkpoint_step']}"

    def test_is_albert_always_false(self, cfgs):
        """Pythia has no shared-weight iterated-map mode — this must never
        route through run_1.py's ALBERT-extended branch."""
        assert all(not e["is_albert"] for e in cfgs.values())

    def test_410m_entries_point_at_410m_repo(self, cfgs):
        for name, entry in cfgs.items():
            if name.startswith("pythia-410m"):
                assert entry["hf_repo"] == "EleutherAI/pythia-410m"

    def test_1_4b_entries_point_at_1_4b_repo(self, cfgs):
        for name, entry in cfgs.items():
            if name.startswith("pythia-1.4b"):
                assert entry["hf_repo"] == "EleutherAI/pythia-1.4b"

class TestRandomBaseline:
    """
    REPLACES TestNoRandomBaseline (2026-08-02).

    That class asserted the registry contained no `random_init` entry at
    all. It was written when the crash was fresh: the registry named a
    "norm_matched" scheme `randomize_weights` did not implement, and
    `run_2.run_full` calls `randomize_weights` outside its per-model
    `try`, so the raise killed the queued sweep rather than skipping one
    model. Deleting the entry stopped the crash.

    It also deleted an object five other artifacts depend on —
    `family_baselines` (two baseline slots), `STEP0_STYLE` vs
    `RANDOM_BASELINE_STYLE`, `compute_distance_from_random`,
    `design-5c.md` ("norm-matched, not fresh-init"), and PREDICTIONS.md
    claim (c), which carries a hard stop on the whole checkpoint sweep.
    The step-0 checkpoint is a different object: GPT-NeoX's own init at
    its own variance scaling, on the training trajectory. The random
    control is the final checkpoint's scale with its structure destroyed,
    not on the trajectory at all.

    The scheme is implemented now, so the entry is back and these tests
    assert the properties that make it safe rather than its absence.

    NOTE: this class does NOT establish that the crash path is fixed.
    `run_2.py`'s try-scope defect is still open (ISSUES_p2.md A3) and this
    entry is the first thing in a sweep that can raise into it again.
    """

    def test_exactly_one_random_init_entry(self, cfgs):
        randoms = [n for n, e in cfgs.items() if e.get("random_init")]
        assert randoms == ["pythia-1.4b-random"]

    def test_every_other_entry_is_a_published_checkpoint(self, cfgs):
        for name, entry in cfgs.items():
            if name == "pythia-1.4b-random":
                continue
            assert entry["random_init"] is False, f"{name} unexpectedly randomized"
            assert "random_init_scheme" not in entry

    def test_requested_scheme_is_one_randomize_weights_accepts(self, cfgs):
        """
        The exact drift that killed the sweep: the registry named a scheme
        the implementation did not accept. Read the guard out of
        core/models.py source rather than hardcoding the accepted set, so
        the two cannot drift apart again.

        Source-read, not import: conftest replaces `core.models` with a
        stub exposing four names, and `randomize_weights` is not among
        them. Importing it here would fail in the stubbed session for a
        reason unrelated to what is being tested.
        """
        src = _read_source("core/models.py")
        m = re.search(r"if scheme not in \(([^)]*)\)", src)
        assert m, "could not locate the scheme guard in core/models.py"
        accepted = set(re.findall(r'"([^"]+)"', m.group(1)))
        assert accepted, "scheme guard parsed but contained no schemes"

        scheme = cfgs["pythia-1.4b-random"]["random_init_scheme"]
        assert scheme in accepted, (
            f"registry requests {scheme!r}; randomize_weights accepts "
            f"{sorted(accepted)}"
        )

    def test_random_control_is_matched_to_the_final_checkpoint(self, cfgs):
        """
        Not step 0. The control asks what a structureless model at the
        *trained* model's scale does, so the scale it borrows has to be the
        trained one. Matching to step 0 would make it a second copy of the
        developmental-origin object.
        """
        entry = cfgs["pythia-1.4b-random"]
        assert entry["revision"] == f"step{PYTHIA_RANDOM_MATCH_STEP}"
        assert PYTHIA_RANDOM_MATCH_STEP == max(PYTHIA_1_4B_ANCHOR_STEPS)

    def test_random_control_carries_no_checkpoint_step(self, cfgs):
        assert "checkpoint_step" not in cfgs["pythia-1.4b-random"]

    def test_random_control_is_excluded_from_the_step_axis(self):
        """
        `checkpoints._STEP_RE` is what puts a model on the step axis. The
        control must not match it, or it would be drawn as a checkpoint.

        The grammar itself now lives in `core/model_family.py` (stdlib-only,
        so it imports here); checkpoints.py re-exports it under the
        `_STEP_RE` name its figure consumers use. Both halves are asserted,
        so moving the grammar again without re-exporting it fails loudly
        rather than silently taking the control off the step axis.
        """
        pattern = _step_re_pattern()
        assert re.compile(pattern).match("pythia-1.4b-random") is None
        assert re.compile(pattern).match("pythia-1.4b-step143000") is not None
        assert _checkpoints_reexports_step_re(), (
            "checkpoints.py no longer re-exports CHECKPOINT_STEP_RE as "
            "_STEP_RE — its four figure consumers call that name"
        )

    def test_both_baseline_slots_resolve_and_are_distinct(self, cfgs):
        """The two-baseline policy end to end, against the resolver's own rule."""
        models = list(cfgs)
        slots = _family_baselines("pythia-1.4b", models)
        assert slots["random"] == "pythia-1.4b-random"
        assert slots["step0"] == "pythia-1.4b-step0"
        assert slots["random"] != slots["step0"]

    def test_410m_family_does_not_borrow_the_1_4b_control(self, cfgs):
        """
        A different-size random control is a different object.
        `family_baselines` documents that refusal; this entry is the first
        one that makes violating it possible, so pin it here.
        """
        slots = _family_baselines("pythia-410m", list(cfgs))
        assert slots["random"] is None
        assert slots["step0"] == "pythia-410m-step0"

class TestNoKeyCollisions:

    def test_no_overlap_between_410m_and_1_4b_keys(self, cfgs):
        m410 = {k for k in cfgs if k.startswith("pythia-410m")}
        m14b = {k for k in cfgs if k.startswith("pythia-1.4b")}
        assert not (m410 & m14b)

    def test_step_count_matches_key_count(self, cfgs):
        n_410m = sum(1 for k in cfgs if k.startswith("pythia-410m"))
        n_14b  = sum(1 for k in cfgs if k.startswith("pythia-1.4b-step"))
        assert n_410m == len(set(PYTHIA_410M_PILOT_STEPS))
        assert n_14b == len(set(PYTHIA_1_4B_ANCHOR_STEPS))

    def test_random_control_is_the_only_non_step_key(self, cfgs):
        non_step = {k for k in cfgs if "-step" not in k}
        assert non_step == {"pythia-1.4b-random"}


# ---------------------------------------------------------------------------
# Open, unrelated to this file's subject (recorded 2026-08-02)
# ---------------------------------------------------------------------------
# `p1_mstate_tracking/visualization/__init__.py` line 44 does
# `from .style import ...` and raises ModuleNotFoundError: no module named
# `p1_mstate_tracking.visualization.style`. Consequence:
# `tests/test_checkpoint_viz.py` — which imports checkpoints,
# checkpoint_scalars and checkpoint_filmstrip from that package — does not
# run. It is absent from the collected suite rather than reported as an
# error, so the checkpoint-visualization logic currently has no test
# coverage and the suite does not say so.
#
# That is why the two cross-module contracts above are checked by reading
# source. Once the package imports again, both helpers should be replaced
# by direct imports of `_STEP_RE` and `family_baselines`.

