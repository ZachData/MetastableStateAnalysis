"""
tests/test_core_prompts.py — Tests for core/prompts.py (core foundations
item 5: the versioned, hash-pinned prompt battery).
"""

import pytest

from core.prompts import (
    compute_prompt_battery_hash, PROMPT_BATTERY_HASH, PROMPT_BATTERY_VERSION,
    verify_same_battery, assert_same_battery,
)


class TestComputePromptBatteryHash:
    def test_deterministic(self):
        a = compute_prompt_battery_hash()
        b = compute_prompt_battery_hash()
        assert a == b

    def test_matches_module_level_constant(self):
        assert compute_prompt_battery_hash() == PROMPT_BATTERY_HASH

    def test_short_and_hex(self):
        h = compute_prompt_battery_hash()
        assert len(h) == 12
        int(h, 16)  # raises if not valid hex

    def test_independent_of_dict_insertion_order(self):
        prompts_a = {"x": "hello", "y": "world"}
        prompts_b = {"y": "world", "x": "hello"}
        assert compute_prompt_battery_hash(prompts_a) == compute_prompt_battery_hash(prompts_b)

    def test_changes_when_text_changes(self):
        prompts = {"x": "hello"}
        a = compute_prompt_battery_hash(prompts)
        b = compute_prompt_battery_hash({"x": "hello world"})
        assert a != b

    def test_changes_when_key_set_changes(self):
        a = compute_prompt_battery_hash({"x": "hello"})
        b = compute_prompt_battery_hash({"x": "hello", "y": "extra"})
        assert a != b

    def test_changes_when_version_changes(self):
        a = compute_prompt_battery_hash({"x": "hello"}, version="v1")
        b = compute_prompt_battery_hash({"x": "hello"}, version="v2")
        assert a != b, "a version bump must change the hash even with identical text"

    def test_defaults_to_current_version(self):
        assert compute_prompt_battery_hash({"x": "hello"}) == compute_prompt_battery_hash(
            {"x": "hello"}, version=PROMPT_BATTERY_VERSION
        )


class TestVerifySameBattery:
    def test_equal_hashes_true(self):
        assert verify_same_battery("abc123", "abc123") is True

    def test_different_hashes_false(self):
        assert verify_same_battery("abc123", "def456") is False


class TestAssertSameBattery:
    def test_does_not_raise_on_match(self):
        assert_same_battery("abc123", "abc123")  # no exception

    def test_raises_on_mismatch(self):
        with pytest.raises(ValueError, match="Prompt battery mismatch"):
            assert_same_battery("abc123", "def456")

    def test_context_included_in_message(self):
        with pytest.raises(ValueError, match="replication gate"):
            assert_same_battery("abc123", "def456", context="replication gate")
