"""
tests/test_p_i5_validation_smoke.py — Tier smoke test for
p7_motifs/p_i5_validation.py.

Run for real against the cached pythia-70m checkpoint
(SMOKE_REAL_DEPS=1 pytest -m smoke). Kept lean relative to the module's
own full battery (claims/calibration/p_i5_validation.json is the real,
committed evidence for the actual finding — negative controls fail) --
this file checks the PLUMBING (schema, no crashes, the diagnostic's own
sanity) rather than re-deriving every number.

    SMOKE_REAL_DEPS=1 HF_HUB_OFFLINE=1 pytest -m smoke tests/test_p_i5_validation_smoke.py -v
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


class TestNegativeControlsSchema:

    def test_returns_one_entry_per_head_with_expected_keys(self, monkeypatch):
        import p7_motifs.p_i5_validation as val
        # One negative-control head instead of two, to keep this fast --
        # the full pair is already run for real in
        # claims/calibration/p_i5_validation.json.
        monkeypatch.setattr(val, "NEGATIVE_CONTROL_HEADS", [(4, 6)])
        result = val.run_negative_controls()
        assert set(result.keys()) == {"L3H6", "L4H6"}
        for label, r in result.items():
            for key in ("is_target", "p_value", "n_prompts",
                        "mean_delta_geometric", "mean_delta_logit"):
                assert key in r
            assert 0.0 < r["p_value"] <= 1.0
        assert result["L3H6"]["is_target"] is True
        assert result["L4H6"]["is_target"] is False


class TestRandomVsRandomDiagnostic:

    def test_schema_and_sane_range(self):
        from p7_motifs.p_i5_validation import run_random_vs_random_diagnostic
        result = run_random_vs_random_diagnostic(seed_a=1, seed_b=2)
        for key in ("seed_a", "seed_b", "n_prompts", "delta_geometric",
                    "delta_logit", "p_value_greater", "p_value_two_sided"):
            assert key in result
        assert 0.0 < result["p_value_greater"] <= 1.0
        assert 0.0 < result["p_value_two_sided"] <= 1.0
        assert result["n_prompts"] == 8  # 9 prompts minus repeated_tokens

    def test_different_seed_pairs_give_different_draws(self):
        """Not a claim about the p-value (that's the committed real run's
        job) -- just that the two draws are actually different directions,
        so this isn't silently comparing a direction to itself."""
        from p7_motifs.p_i5_validation import run_random_vs_random_diagnostic
        a = run_random_vs_random_diagnostic(seed_a=1, seed_b=2)
        b = run_random_vs_random_diagnostic(seed_a=3, seed_b=4)
        assert a["delta_geometric"] != b["delta_geometric"]


class TestSeedSensitivitySchema:

    def test_two_seeds_returns_expected_keys(self):
        from p7_motifs.p_i5_validation import run_seed_sensitivity
        result = run_seed_sensitivity(seeds=[1, 2])
        assert result["seeds"] == [1, 2]
        assert len(result["p_values"]) == 2
        for key in ("min", "max", "mean"):
            assert key in result
        assert result["min"] <= result["mean"] <= result["max"]
