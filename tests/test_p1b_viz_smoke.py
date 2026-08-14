"""
tests/test_p1b_viz_smoke.py — p1b_hemisphere/visualization.

Two tiers, matching tests/SMOKE_TESTS_NOTES.md's split and
tests/test_p2_viz_smoke.py's shape. Neither tier needs torch or a model: this
package reads artifacts only, and the fixture writes them.

  pure  — loader grammar and the class vocabularies. These catch the failure
          modes that actually bite: a transition field that is null at the
          last layer being dropped instead of NaN-ed (which shifts every
          depth profile in the package by one), a JSON string key not
          normalized back to a layer int, a regime class that exists in
          `bipartition_detect` but has no color and would render as invalid
          gray, and a particle column reshaped to the wrong grid.

  smoke — the CLI end to end against the synthetic Phase 1b directory,
          asserting the figures appear, that a run stripped of its optional
          inputs degrades to skips rather than errors, and that the package
          imports without Phase 1's visualization dependencies.

Analysis logic is imported from `p1b_hemisphere`, never restated here, so
the tests assert delegation rather than agreement between two copies.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from p1b_hemisphere.visualization._fixture import build_fixture, build_run
from p1b_hemisphere.visualization.loaders import (
    checkpoint_families, discover_runs, layer_field, layer_pair_field,
    layer_strings, load_cross_run, particle_grid,
)
from p1b_hemisphere.visualization.style import (
    CONE_COLORS, REDUNDANCY_COLORS, REGIME_COLORS, REGIME_REL_COLORS,
    fiedler_norm,
)


# ─────────────────────────────────────────────────────────────────────────────
# pure — loaders
# ─────────────────────────────────────────────────────────────────────────────

def test_layer_field_maps_json_null_to_nan_and_keeps_length():
    """
    The transition fields (crossing_count, axis_rotation, match_overlap) are
    null at the last layer by the writer's own convention. Dropping that row
    instead of NaN-ing it shifts every depth profile by one, silently, and
    only at the end of the axis where nobody looks.
    """
    per_layer = [{"axis_rotation": 0.1}, {"axis_rotation": None},
                 {"axis_rotation": 0.3}, {}]
    v = layer_field(per_layer, "axis_rotation")
    assert v.size == 4
    assert v[0] == 0.1 and v[2] == 0.3
    assert np.isnan(v[1]) and np.isnan(v[3])


def test_layer_field_missing_field_is_all_nan_not_empty():
    v = layer_field([{"a": 1.0}, {"a": 2.0}], "does_not_exist")
    assert v.size == 2 and np.all(np.isnan(v))


def test_layer_pair_field_rejects_a_non_pair_rather_than_padding():
    per_layer = [{"within_half_ip": [0.8, 0.7]},
                 {"within_half_ip": [0.8]},
                 {"within_half_ip": None}]
    v = layer_pair_field(per_layer, "within_half_ip")
    assert v.shape == (3, 2)
    assert v[0].tolist() == [0.8, 0.7]
    assert np.all(np.isnan(v[1])) and np.all(np.isnan(v[2]))


def test_layer_strings_defaults_missing_to_invalid():
    assert layer_strings([{"regime": "collapsed"}, {"regime": None}, {}],
                         "regime") == ["collapsed", "invalid", "invalid"]


def test_json_string_keys_are_normalized_back_to_layer_ints(tmp_path):
    """
    hdbscan_nesting / border_vs_noise are keyed by layer int in memory and by
    "7" after a round trip through JSON. Assuming either convention at a call
    site is the artifact-contract bug class core/artifacts.py exists to kill —
    and the one that already bit this phase once (status-1b R2).
    """
    build_run(tmp_path, model="gpt2", prompt="p", n_layers=4, n_tokens=16)
    run = discover_runs(tmp_path)[0]
    assert set(run.nesting["per_layer"]) == {0, 1, 2, 3}
    assert set(run.border_vs_noise["per_layer"]) == {0, 1, 2, 3}


# ─────────────────────────────────────────────────────────────────────────────
# pure — class vocabularies
# ─────────────────────────────────────────────────────────────────────────────

def test_every_regime_class_the_classifier_can_emit_has_a_color():
    """
    A class present in `bipartition_detect` but missing from the palette
    renders as invalid gray — a real verdict displayed as "no data". Both
    vocabularies are wider than they look: `diffuse` and `uniform` exist.
    """
    from p1b_hemisphere.bipartition_detect import (
        classify_regime, classify_regime_relative,
    )

    rng = np.random.default_rng(0)
    for _ in range(400):
        minority = float(rng.uniform(0, 0.5))
        angle = float(rng.uniform(0, np.pi))
        a, b = float(rng.uniform(0, 1)), float(rng.uniform(0, 1))
        assert classify_regime(minority, angle, a, b) in REGIME_COLORS
        assert classify_regime_relative(minority, float(rng.uniform(0, 1.5))) \
            in REGIME_REL_COLORS


def test_cone_and_redundancy_vocabularies_are_covered():
    for label in ("cone_collapse", "split", "borderline", "invalid"):
        assert label in CONE_COLORS
    for label in ("pc1", "top_pc_block", "distinct", "degenerate"):
        assert label in REDUNDANCY_COLORS


def test_fiedler_norm_is_centered_on_zero_not_on_the_data():
    """
    With an 80/20 split matplotlib's default would put the neutral color
    inside the majority hemisphere, drawing a boundary that is not there.
    Zero is the boundary; the normalizer has to be anchored to it.
    """
    lopsided = np.concatenate([np.full(80, -0.9), np.full(20, 0.2)])
    norm = fiedler_norm(lopsided)
    assert norm.vcenter == 0.0
    assert norm.vmin == -norm.vmax
    assert norm(0.0) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# smoke — fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("p1bout"))


def test_fixture_is_discoverable(fixture_dir):
    runs = discover_runs(fixture_dir)
    # 2 models x 2 prompts + one 10-checkpoint family.
    assert len(runs) == 14
    stems = {r.stem for r in runs}
    assert "gpt2-large_wiki_paragraph" in stems
    assert "pythia-410m-step1000_wiki_paragraph" in stems
    assert all(not r.missing for r in runs), \
        f"fixture should write every optional input: {runs[0].missing}"


def test_discovery_filters_by_model_and_prompt(fixture_dir):
    runs = discover_runs(fixture_dir, models=["gpt2-large"],
                         prompts=["wiki_paragraph"])
    assert len(runs) == 1 and runs[0].model == "gpt2-large"


def test_cross_run_digest_is_the_phases_own_shape(fixture_dir):
    """
    The fixture builds its digest through `p1b_report`'s aggregator, so this
    asserts the reader and the phase agree — including on fields a future
    revision adds.
    """
    from p1b_hemisphere.p1b_report import AGGREGATED_FIELDS

    cross = load_cross_run(fixture_dir)
    assert cross is not None
    assert set(cross) == {"by_model", "by_prompt", "by_checkpoint",
                          "global_verdict"}
    for agg in cross["by_model"].values():
        for field in AGGREGATED_FIELDS:
            assert f"mean_{field}" in agg
    assert cross["global_verdict"]["paper_alignment"] in (
        "cone_collapse", "split", "mixed")


def test_checkpoint_families_exclude_models_with_no_step(fixture_dir):
    """
    gpt2-large and albert-base-v2 carry no step and must not be placed on a
    step axis — the same rule pythia-1.4b-random relies on.
    """
    fams = checkpoint_families(discover_runs(fixture_dir))
    assert list(fams) == ["pythia-410m"]
    assert len(fams["pythia-410m"]) == 10


def test_particle_grid_matches_the_per_layer_scalars(fixture_dir):
    """
    The particle table and the per-layer JSON are two views of one object in
    the fixture, exactly as in a real run. If the grid reshape is wrong, the
    hemisphere counts recovered from it will not match the ones Block 0
    recorded — and the barcode figure would still look plausible.
    """
    run = next(r for r in discover_runs(fixture_dir)
               if r.stem == "gpt2-large_wiki_paragraph")
    hemi = particle_grid(run, "hemisphere")
    assert hemi.shape == (run.n_layers, run.n_tokens)

    sizes = layer_pair_field(run.per_layer, "hemisphere_sizes")
    for L in range(run.n_layers):
        assert int((hemi[L] == 1).sum()) == int(sizes[L, 1])


def test_particle_grid_refuses_a_mismatched_reshape(fixture_dir, capsys):
    run = next(iter(discover_runs(fixture_dir)))
    run.data["n_tokens"] = run.n_tokens + 1        # lie about the shape
    assert particle_grid(run, "hemisphere") is None
    assert "not reshaping" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# smoke — figures
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_all_produces_every_class(fixture_dir, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from p1b_hemisphere.visualization.pipeline import CLASSES, generate_all

    runs = discover_runs(fixture_dir, models=["gpt2-large"],
                         prompts=["wiki_paragraph"])
    produced = generate_all(fixture_dir, tmp_path, runs=runs)

    for cls in CLASSES:
        if cls == "checkpoints":
            continue          # no family in this one-run subset
        assert produced.get(cls), f"{cls} produced no figures"
    for paths in produced.values():
        for p in paths:
            assert p.exists() and p.stat().st_size > 0

    assert (tmp_path / "gpt2-large_wiki_paragraph").is_dir()
    assert (tmp_path / "_cross").is_dir()


def test_unknown_class_is_rejected_rather_than_silently_skipped(fixture_dir,
                                                               tmp_path):
    from p1b_hemisphere.visualization.pipeline import generate_all
    with pytest.raises(ValueError, match="unknown figure class"):
        generate_all(fixture_dir, tmp_path, classes=["regmie"])


def test_missing_optional_inputs_degrade_to_skips(tmp_path):
    """
    Runs written before the per-layer emissions landed carry none of
    `cone_per_layer`, `hdbscan_nesting`, `border_vs_noise`,
    `persistence_length`, or the axes npz. Those runs must still draw
    everything that does not need them, and report the rest — never raise.
    """
    import matplotlib
    matplotlib.use("Agg")
    from p1b_hemisphere.visualization.pipeline import generate_all

    src = tmp_path / "old_run"
    build_run(src, model="gpt2", prompt="wiki_paragraph", n_layers=8,
              n_tokens=48)
    path = next(src.glob("phase1b_*.json"))
    data = json.loads(path.read_text())
    for key in ("cone_per_layer", "hdbscan_nesting", "border_vs_noise",
                "persistence_length"):
        data.pop(key)
    path.write_text(json.dumps(data))
    next(src.glob("phase1b_*_axes.npz")).unlink()

    run = discover_runs(src)[0]
    assert len(run.missing) >= 5
    assert any("axes npz" in m for m in run.missing)

    produced = generate_all(src, tmp_path / "figs")
    assert produced["regime"], "regime needs none of the optional inputs"
    assert produced["cone"], "C1/C2/C4 need only the per-layer JSON"
    # The three that genuinely cannot be drawn.
    names = {p.stem for p in produced["membership"]}
    assert "nesting_r_c" not in names
    assert "border_vs_noise_auc" not in names
    assert "stability_hist" in names


def test_package_imports_without_phase1_visualization_deps(monkeypatch):
    """
    Only the `checkpoints` class needs Phase 1's step-axis helpers, and
    reaching them imports that package's whole figure surface. Importing
    p1b's package — or drawing any other class — must not require it.
    """
    import builtins
    import importlib

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("p1_mstate_tracking.visualization"):
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    mod = importlib.reload(
        importlib.import_module("p1b_hemisphere.visualization.checkpoints_1b"))
    # The class reports the missing dependency against itself and returns [].
    assert mod.generate_checkpoint_figures([], Path("/tmp")) == []
