"""
tests/test_p2_viz_smoke.py — p2_eigenspectra/visualization.

Two tiers, matching tests/SMOKE_TESTS_NOTES.md's split:

  pure  — loader grammar and the zone adapter. These catch the failure
          modes that matter: a layer ordering that silently scrambles
          every depth profile, a JSON null read as a dropped layer
          instead of a NaN, and a zone adapter that has quietly grown
          logic of its own instead of delegating to layer_v_events.

  smoke — the CLI end to end against a synthetic Phase 2 output directory,
          asserting the figures and transitions_p2_*.json actually appear.
          Slow-ish (renders ~25 PNGs) but it is the only thing that
          exercises the p1 convention imports, which is where this package
          is most exposed to a refactor elsewhere.

Analysis logic is imported from p2_eigenspectra, not copied here, so the
tests assert delegation rather than agreement between two implementations.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from p2_eigenspectra.visualization._fixture import build_fixture
from p2_eigenspectra.visualization.loaders import (
    discover_weight_summaries, discover_p2_runs, layer_keys, layer_field,
    per_head_field, eigen_cloud,
)
from p2_eigenspectra.visualization.spectra import (
    zone_counts, weight_families,
    WEIGHT_METRICS, zone_series,
)
from p2_eigenspectra.visualization.p2_scalars import (
    WEIGHT_SCALARS, RUN_SCALARS, _r_rescaled_frac, ffn_channel_available,
)


# ─────────────────────────────────────────────────────────────────────────────
# pure — loaders
# ─────────────────────────────────────────────────────────────────────────────

def test_layer_keys_sort_numerically_not_lexically():
    """layer_10 must come after layer_2. A lexical sort here scrambles
    every depth profile in the package without raising."""
    summary = {"layers": {f"layer_{i}": {} for i in [0, 1, 2, 10, 11, 20, 3]}}
    assert layer_keys(summary) == [
        "layer_0", "layer_1", "layer_2", "layer_3",
        "layer_10", "layer_11", "layer_20",
    ]


def test_layer_field_maps_json_null_to_nan_and_keeps_length():
    summary = {"layers": {
        "layer_0": {"frac_repulsive": 0.5},
        "layer_1": {"frac_repulsive": None},
        "layer_2": {},
    }}
    v = layer_field(summary, "frac_repulsive")
    assert v.size == 3
    assert v[0] == 0.5
    assert np.isnan(v[1]) and np.isnan(v[2])


def test_layer_field_missing_field_is_all_nan_not_empty():
    summary = {"layers": {"layer_0": {"a": 1.0}, "layer_1": {"a": 2.0}}}
    v = layer_field(summary, "does_not_exist")
    assert v.size == 2 and np.all(np.isnan(v))


def test_per_head_field_rejects_ragged_rather_than_padding():
    summary = {"layers": {
        "layer_0": {"qk_spectral_norms_per_head": [1.0, 2.0]},
        "layer_1": {"qk_spectral_norms_per_head": [1.0, 2.0, 3.0]},
    }}
    assert per_head_field(summary) is None


# ─────────────────────────────────────────────────────────────────────────────
# pure — zone logic
# ─────────────────────────────────────────────────────────────────────────────

def test_zone_counts_delegates_to_layer_v_events():
    """
    The adapter must pass through to the analysis module unchanged — its
    only job is accepting a bare array where classify_layers wants the
    v_profile dict. If this ever grows logic of its own, the thresholds
    have been forked.
    """
    from p2_eigenspectra.layer_v_events import classify_layers

    rng = np.random.default_rng(0)
    for _ in range(20):
        rep = rng.uniform(0, 1, int(rng.integers(2, 32)))
        assert zone_counts(rep) == classify_layers({"repulsive_frac": rep})


def test_zone_counts_thresholds_are_the_documented_ones():
    rep = np.array([0.9, 0.8, 0.56, 0.50, 0.44, 0.1])
    z = zone_counts(rep)
    assert z["n_repulsive"] == 3          # > 0.55
    assert z["n_attractive"] == 2         # < 0.45
    assert z["n_transition"] == 1
    assert z["crossover_layer"] == 3


def test_zone_counts_no_crossover_when_repulsive_to_the_end():
    assert zone_counts(np.array([0.9, 0.9, 0.9]))["crossover_layer"] is None
    assert zone_counts(np.array([0.1, 0.1]))["crossover_layer"] is None


# ─────────────────────────────────────────────────────────────────────────────
# pure — scalar semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_rescaled_frac_is_nan_not_one_when_there_are_no_violations():
    """A checkpoint with nothing to explain must be a gap in the curve, not
    a perfect score — otherwise every untrained checkpoint reads as maximal
    evidence for the rescaled-frame result."""
    assert np.isnan(_r_rescaled_frac(
        {"beta1.0_n_violations": 0, "rescaled_improvement_beta1.0": 0}))
    assert _r_rescaled_frac(
        {"beta1.0_n_violations": 4, "rescaled_improvement_beta1.0": 3}) == 0.75


def test_registries_are_nonempty_and_well_formed():
    for name, spec in WEIGHT_METRICS.items():
        assert callable(spec["fn"]), name
        assert spec["ylabel"] and spec["title"], name
        if spec["cmap"] in ("coolwarm", "PuOr"):
            assert spec["null"] is not None, f"{name}: diverging cmap needs a null"
    for reg in (WEIGHT_SCALARS, RUN_SCALARS):
        for name, (fn, label) in reg.items():
            assert callable(fn) and label, name


# ─────────────────────────────────────────────────────────────────────────────
# smoke — fixture + CLI
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("p2out"),
                         steps=[0, 8, 128, 512, 1000, 9000, 143000],
                         prompts=["wiki_paragraph", "short_heterogeneous",
                                  "homer_iliad"],
                         n_layers=8, d_model=64)


def test_fixture_is_discoverable(fixture_dir):
    summaries = discover_weight_summaries(fixture_dir)
    runs = discover_p2_runs(fixture_dir)
    assert len(summaries) == 7
    assert len(runs) == 21
    # Model name is recovered from the filename: the real artifact's
    # "model" field is always null.
    assert "pythia-410m-step1000" in summaries
    assert ("pythia-410m-step1000", "homer_iliad") in runs


def test_families_group_by_base_and_sort_by_step(fixture_dir):
    fams = weight_families(discover_weight_summaries(fixture_dir))
    assert list(fams) == ["pythia-410m"]
    steps = [s for s, _ in fams["pythia-410m"]]
    assert steps == sorted(steps) and steps[0] == 0


def test_eigen_cloud_reads_only_the_requested_layer(fixture_dir):
    cloud = eigen_cloud(fixture_dir, "pythia-410m-step1000", 4)
    assert cloud is not None
    re_, im_, sym = cloud
    assert re_.shape == im_.shape and re_.size == 64
    assert sym is not None and sym.size == 64
    # A layer with no arrays written returns None rather than raising.
    assert eigen_cloud(fixture_dir, "pythia-410m-step1000", 5) is None


def test_ffn_channel_reported_unavailable_on_parallel_residual(fixture_dir):
    """The fixture reproduces the real Pythia case: decompose.py writes no
    ffn deltas, so the v-score's FFN term is identically zero and the
    figure must say so."""
    assert ffn_channel_available(discover_p2_runs(fixture_dir)) is False


def test_zone_series_lengths_match(fixture_dir):
    summaries = discover_weight_summaries(fixture_dir)
    fam = weight_families(summaries)["pythia-410m"]
    steps, fracs, crossover = zone_series(summaries, fam)
    assert len(steps) == len(crossover) == len(fam)
    for k in ("repulsive", "transition", "attractive"):
        assert len(fracs[k]) == len(steps)
    total = np.array(fracs["repulsive"]) + np.array(fracs["transition"]) \
        + np.array(fracs["attractive"])
    assert np.allclose(total, 1.0)


def test_cli_end_to_end(fixture_dir, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from p2_eigenspectra.visualization.pipeline import generate_p2_figures

    out = tmp_path / "figs"
    generate_p2_figures(fixture_dir, out,
                        metrics=["rep_frac", "non_normality"],
                        filmstrip_k=4)

    names = {p.name for p in out.iterdir()}
    assert "p2_weight_scalars_pythia-410m.png" in names
    assert "p2_run_scalars_pythia-410m.png" in names
    assert "spectra_heatmap_rep_frac_pythia-410m.png" in names
    assert "spectra_sweep_non_normality_pythia-410m.png" in names
    assert "zone_bands_pythia-410m.png" in names
    assert any(n.startswith("eigen_cloud_") for n in names)

    payload = json.loads((out / "transitions_p2_pythia-410m.json").read_text())
    assert payload["base_model"] == "pythia-410m"
    assert payload["phase"] == "p2_eigenspectra"
    assert payload["ranked_by_jump"]
    # Run-side metrics are namespaced so they can't collide with the
    # weight-side ones of the same name.
    assert any(k.startswith("run__") for k in payload["per_metric"])


def test_no_checkpoint_family_is_a_noop(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from p2_eigenspectra.visualization.pipeline import generate_p2_figures

    d = tmp_path / "p2_gpt2"
    d.mkdir()
    (d / "ov_summary_gpt2-large.json").write_text(json.dumps(
        {"model": None, "d_model": 8, "d_head": 2, "n_heads": 4,
         "is_per_layer": True,
         "layers": {"layer_0": {"frac_repulsive": 0.5}}}))
    out = tmp_path / "figs_none"
    generate_p2_figures(d, out)
    assert not out.exists() or not list(out.iterdir())
