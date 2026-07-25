"""
tests/test_pythia_registry.py — pure-logic tests for
core/pythia_registry.build_pythia_model_configs.

No model loading, no network: transformers' GPTNeoXModel/AutoTokenizer are
only imported for their identity (used as a dict value), never
instantiated here. Runs under the same stub-heavy-deps session as the rest
of the non-smoke suite.
"""
import re

import pytest

from core.pythia_registry import (
    build_pythia_model_configs,
    PYTHIA_410M_PILOT_STEPS,
    PYTHIA_1_4B_ANCHOR_STEPS,
    PYTHIA_1_4B_EXPENSIVE_STEPS,
    PYTHIA_ALL_STEPS,
)

STEP_RE = re.compile(r"^step\d+$")


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
        required = {"model_class", "tokenizer_class", "is_albert",
                    "random_init", "hf_repo", "revision", "checkpoint_step"}
        for name, entry in cfgs.items():
            missing = required - entry.keys()
            assert not missing, f"{name} missing keys: {missing}"

    def test_revision_format(self, cfgs):
        for name, entry in cfgs.items():
            assert STEP_RE.match(entry["revision"]), (
                f"{name} has malformed revision {entry['revision']!r}"
            )

    def test_revision_matches_checkpoint_step(self, cfgs):
        for name, entry in cfgs.items():
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

class TestNoRandomBaseline:

    def test_no_random_init_entries(self, cfgs):
        """Every Pythia entry is a real published checkpoint. A random_init
        entry here would route into core.models.randomize_weights, which
        accepts only "orthogonal"/"gaussian" and raises outside run_all's
        try/except — killing the sweep rather than skipping one model."""
        assert all(e["random_init"] is False for e in cfgs.values())

    def test_no_random_init_scheme_keys(self, cfgs):
        assert all("random_init_scheme" not in e for e in cfgs.values())

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
