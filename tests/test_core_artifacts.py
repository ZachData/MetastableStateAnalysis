"""
tests/test_core_artifacts.py — Tests for core/artifacts.py (core
foundations item 2: the artifact contract).
"""
import json
import numpy as np
import pytest

from core.artifacts import (
    ArtifactSpec, REGISTRY, PHASE1, MANIFEST,
    get_spec, artifact_path, validate_artifact,
)


class TestArtifactSpec:
    def test_valid_kinds_accepted(self):
        for kind in ("json", "npz", "txt"):
            ArtifactSpec(kind=kind, filename="x")

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError):
            ArtifactSpec(kind="csv", filename="x")

    def test_frozen(self):
        spec = ArtifactSpec(kind="json", filename="x.json")
        with pytest.raises(Exception):
            spec.filename = "y.json"


class TestGetSpec:
    def test_known_phase1_artifact(self):
        spec = get_spec("phase1", "activations")
        assert spec.filename == "activations.npz"
        assert "activations" in spec.required_keys

    def test_unknown_phase_raises_with_options(self):
        with pytest.raises(KeyError, match="Known phases"):
            get_spec("phase99", "x")

    def test_unknown_artifact_raises_with_options(self):
        with pytest.raises(KeyError, match="Known artifacts"):
            get_spec("phase1", "not_a_real_artifact")

    def test_manifest_registered(self):
        spec = get_spec("manifest", "manifest")
        assert spec is MANIFEST


class TestArtifactPath:
    def test_builds_expected_path(self, tmp_path):
        p = artifact_path(tmp_path, "phase1", "geometry")
        assert p == tmp_path / "geometry.json"


class TestValidateArtifact:
    def test_missing_file_reports_not_ok(self, tmp_path):
        result = validate_artifact(tmp_path, "phase1", "geometry")
        assert result["ok"] is False
        assert "does not exist" in result["error"]

    def test_json_with_all_required_keys_ok(self, tmp_path):
        (tmp_path / "trajectory.json").write_text(
            json.dumps({"cluster_tracking": {"trajectories": []}, "plateau_layers": []})
        )
        result = validate_artifact(tmp_path, "phase1", "trajectory")
        assert result["ok"] is True
        assert result["missing_keys"] == []

    def test_json_missing_a_required_key_reports_it(self, tmp_path):
        (tmp_path / "trajectory.json").write_text(
            json.dumps({"cluster_tracking": {"trajectories": []}})
        )
        result = validate_artifact(tmp_path, "phase1", "trajectory")
        assert result["ok"] is False
        assert result["missing_keys"] == ["plateau_layers"]

    def test_npz_with_required_key_ok(self, tmp_path):
        np.savez(tmp_path / "activations.npz", activations=np.zeros((2, 3, 4)))
        result = validate_artifact(tmp_path, "phase1", "activations")
        assert result["ok"] is True

    def test_npz_wrong_key_name_reports_missing(self, tmp_path):
        # This is exactly the "miskeyed" bug class the contract exists to
        # catch: producer wrote under the wrong array key.
        np.savez(tmp_path / "activations.npz", hidden_states=np.zeros((2, 3, 4)))
        result = validate_artifact(tmp_path, "phase1", "activations")
        assert result["ok"] is False
        assert result["missing_keys"] == ["activations"]

    def test_txt_artifact_no_required_keys_just_existence(self, tmp_path):
        (tmp_path / "tokens.txt").write_text("0\thello\n")
        result = validate_artifact(tmp_path, "phase1", "tokens")
        assert result["ok"] is True

    def test_malformed_json_reports_error_not_crash(self, tmp_path):
        (tmp_path / "trajectory.json").write_text("{not valid json")
        result = validate_artifact(tmp_path, "phase1", "trajectory")
        assert result["ok"] is False
        assert result["error"] is not None

    def test_unregistered_pair_raises(self, tmp_path):
        with pytest.raises(KeyError):
            validate_artifact(tmp_path, "phase1", "nope")


class TestRegistryConsistency:
    def test_every_phase1_filename_unique(self):
        filenames = [spec.filename for spec in PHASE1.values()]
        assert len(filenames) == len(set(filenames))

    def test_registry_top_level_matches_expected_phases(self):
        assert {"phase1", "phase1_session", "manifest"} <= set(REGISTRY)
