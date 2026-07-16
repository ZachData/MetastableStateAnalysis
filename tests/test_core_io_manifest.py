"""
tests/test_core_io_manifest.py — Tests for the run-manifest infrastructure
appended to core/io.py (core foundations item 1).
"""
import json
import time

import pytest

from core.io import (
    compute_manifest_id, get_git_sha, RunTimer,
    write_manifest, load_manifest, stamp_figure_name,
)
from core.artifacts import validate_artifact


class TestComputeManifestId:
    def test_deterministic(self):
        a = compute_manifest_id("gpt2", "abc123", checkpoint_step=None, seeds={"numpy": 1})
        b = compute_manifest_id("gpt2", "abc123", checkpoint_step=None, seeds={"numpy": 1})
        assert a == b

    def test_differs_on_model(self):
        a = compute_manifest_id("gpt2", "abc123")
        b = compute_manifest_id("pythia-1.4b-step0", "abc123")
        assert a != b

    def test_differs_on_checkpoint_step(self):
        a = compute_manifest_id("pythia-1.4b-step0", "abc123", checkpoint_step=0)
        b = compute_manifest_id("pythia-1.4b-step0", "abc123", checkpoint_step=1000)
        assert a != b

    def test_ignores_timestamp_by_construction(self):
        # No timestamp param exists at all — this documents why: the id
        # must be stable across re-runs of the same logical run.
        import inspect
        sig = inspect.signature(compute_manifest_id)
        assert "timestamp" not in sig.parameters

    def test_short_and_hex(self):
        mid = compute_manifest_id("gpt2", "abc123")
        assert len(mid) == 12
        int(mid, 16)  # raises if not valid hex


class TestGetGitSha:
    def test_returns_none_or_string_never_raises(self, tmp_path):
        # tmp_path is not a git repo -> must return None, not raise.
        result = get_git_sha(tmp_path)
        assert result is None or isinstance(result, str)


class TestRunTimer:
    def test_measures_positive_elapsed(self):
        with RunTimer() as t:
            time.sleep(0.01)
        assert t.elapsed is not None
        assert t.elapsed > 0

    def test_elapsed_none_before_exit(self):
        t = RunTimer()
        with t:
            assert t.elapsed is None


class TestWriteAndLoadManifest:
    def test_write_creates_file(self, tmp_path):
        m = write_manifest(
            tmp_path, model="gpt2", prompt_battery_hash="deadbeef",
            wall_time_seconds=1.23,
        )
        assert (tmp_path / "manifest.json").exists()
        assert m["model"] == "gpt2"

    def test_required_keys_all_present(self, tmp_path):
        write_manifest(
            tmp_path, model="pythia-1.4b-step1000", prompt_battery_hash="deadbeef",
            wall_time_seconds=4.0, hf_revision="step1000", checkpoint_step=1000,
            seeds={"numpy": 42}, config={"beta": [0.1, 1.0]},
        )
        result = validate_artifact(tmp_path, "manifest", "manifest")
        assert result["ok"] is True, result

    def test_load_manifest_roundtrip(self, tmp_path):
        written = write_manifest(
            tmp_path, model="gpt2", prompt_battery_hash="deadbeef", wall_time_seconds=1.0,
        )
        loaded = load_manifest(tmp_path)
        assert loaded == written

    def test_load_manifest_missing_returns_none(self, tmp_path):
        assert load_manifest(tmp_path) is None

    def test_extra_does_not_override_required_keys(self, tmp_path):
        m = write_manifest(
            tmp_path, model="gpt2", prompt_battery_hash="deadbeef", wall_time_seconds=1.0,
            extra={"model": "SHOULD_NOT_WIN", "custom_field": 123},
        )
        assert m["model"] == "gpt2"
        assert m["custom_field"] == 123

    def test_manifest_id_matches_compute_manifest_id(self, tmp_path):
        m = write_manifest(
            tmp_path, model="gpt2", prompt_battery_hash="deadbeef", wall_time_seconds=1.0,
            checkpoint_step=None, seeds={"numpy": 42},
        )
        expected = compute_manifest_id("gpt2", "deadbeef", checkpoint_step=None, seeds={"numpy": 42})
        assert m["manifest_id"] == expected

    def test_timestamp_defaults_to_now_iso(self, tmp_path):
        m = write_manifest(
            tmp_path, model="gpt2", prompt_battery_hash="deadbeef", wall_time_seconds=1.0,
        )
        # Should parse as ISO-8601 without raising.
        from datetime import datetime
        datetime.fromisoformat(m["timestamp"])

    def test_config_must_be_json_serialisable(self, tmp_path):
        with pytest.raises(TypeError):
            write_manifest(
                tmp_path, model="gpt2", prompt_battery_hash="deadbeef",
                wall_time_seconds=1.0, config={"bad": object()},
            )


class TestStampFigureName:
    def test_basic(self):
        assert stamp_figure_name("energy_curve.png", "a1b2c3d4e5f6") == "energy_curve__a1b2c3d4e5f6.png"

    def test_no_extension(self):
        assert stamp_figure_name("energy_curve", "abc123") == "energy_curve__abc123"

    def test_multi_dot_name_keeps_true_extension(self):
        assert stamp_figure_name("fig.v2.png", "abc123") == "fig.v2__abc123.png"
