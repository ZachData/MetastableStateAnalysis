"""
tests/test_p1c_viz_smoke.py — p1c_frames/visualization.

Two tiers, matching tests/SMOKE_TESTS_NOTES.md's split and
tests/test_p1b_viz_smoke.py's shape. Neither tier needs torch or a model:
this package reads artifacts only, and the fixture writes them through
`p1c_io.save_p1c` rather than hand-rolling JSON, so what is tested is the
real artifact grammar.

  pure  — the loader grammar and the verdict vocabulary. These catch the
          failure modes that actually bite this phase:

          * `save_p1c` sends arrays with MORE THAN 8 entries to the npz and
            leaves shorter ones in the JSON, so the same quantity lives in
            different files depending on the model's depth. A loader that
            reads one place works on Pythia-410M and silently returns
            nothing on a short run.
          * F's `per_layer` is not one entry per layer — layers whose
            centroids failed are absent — so a column read positionally is
            a depth profile shifted left at every gap.
          * a verdict class the palette has no colour for renders grey, i.e.
            a real verdict displayed as "not classified".

  smoke — the CLI end to end against the synthetic directory, asserting the
          figures appear, that a degraded run (no `norms`, no `beta_eff`,
          no `clusters.npz`) produces skips rather than errors, and that the
          package imports and draws without Phase 1's visualization
          dependencies.

Analysis logic is imported from `p1c_frames`, never restated here, so the
tests assert delegation rather than agreement between two copies.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from p1c_frames.visualization._fixture import FIXTURE_STEPS, build_fixture, build_run
from p1c_frames.visualization.loaders import (
    SUBEXPS, checkpoint_families, discover_runs, floats, record_field,
    record_matrix, records,
)
from p1c_frames.visualization.style import (
    STEP_COLORS, STEP_DEFS, VERDICT_COLORS, residual_norm, verdict_word,
)

# Tier: deps -- needs the heavy tier importable (torch / transformers /
# scikit-learn / matplotlib). No model download, no run artifacts.
# Measured, not assumed; see pyproject.toml markers.
pytestmark = pytest.mark.deps


# ─────────────────────────────────────────────────────────────────────────────
# pure — the json / npz split
# ─────────────────────────────────────────────────────────────────────────────

def test_series_resolves_arrays_from_either_file(tmp_path):
    """
    `save_p1c` splits on SIZE, not on meaning: an ndarray with more than 8
    entries goes to `p1c_curves.npz` and a shorter one stays in the JSON as
    a list. So `A.h_calibrated` is in the npz for a 12-layer run (11
    entries) and in the JSON for an 8-layer one (7 entries), and a loader
    that reads one place is silently empty on half the sweep.
    """
    build_run(tmp_path, model="deep", prompt="p", n_layers=12, n_tokens=32)
    build_run(tmp_path, model="shallow", prompt="p", n_layers=8, n_tokens=32)
    runs = {r.model: r for r in discover_runs(tmp_path)}

    deep, shallow = runs["deep"], runs["shallow"]
    assert "A.h_calibrated" in deep.arrays(), "12-layer run should use the npz"
    assert "A.h_calibrated" not in shallow.arrays(), \
        "8-layer run's 7-entry array should stay in the JSON"

    # Same call, same contract, regardless of which file it came from.
    assert deep.series("A.h_calibrated").size == 11
    assert shallow.series("A.h_calibrated").size == 7
    assert np.isfinite(shallow.series("A.h_calibrated")).all()


def test_series_of_an_absent_key_is_empty_not_an_exception(tmp_path):
    build_run(tmp_path, model="m", prompt="p", n_layers=10, n_tokens=24)
    run = discover_runs(tmp_path)[0]
    assert run.series("B.does_not_exist").size == 0
    assert run.series("Z.nothing.at.all").size == 0
    assert not run.has_series("B.does_not_exist")


def test_json_null_maps_to_nan_and_keeps_length():
    """
    `_NpEncoder` writes non-finite floats as JSON null for 1-D arrays.
    Dropping those rows rather than NaN-ing them shifts every depth profile
    by one, silently, at whichever end nobody looks.
    """
    v = floats([0.1, None, 0.3, "x", 0.5])
    assert v.size == 5
    assert v[0] == 0.1 and v[2] == 0.3 and v[4] == 0.5
    assert np.isnan(v[1]) and np.isnan(v[3])


def test_nan_survives_the_round_trip_for_h_attn_only(tmp_path):
    """
    `h_attn_only` is nan on every run without sublayer streams (G3), and it
    is the frame-correct definition. It must arrive as a NaN series of the
    right length, not as an absent key — the difference is a printed skip
    against a silently missing figure.
    """
    build_run(tmp_path, model="m", prompt="p", n_layers=12, attn_streams=False)
    run = discover_runs(tmp_path)[0]
    v = run.series("A.h_attn_only")
    assert v.size == 11 and np.isnan(v).all()
    assert not run.has_series("A.h_attn_only")
    assert any("h_attn_only" in m for m in run.missing)


# ─────────────────────────────────────────────────────────────────────────────
# pure — per-layer records
# ─────────────────────────────────────────────────────────────────────────────

def test_f_records_are_scattered_back_onto_the_depth_axis(tmp_path):
    """
    Sub-experiment F skips layers whose centroids could not be loaded, so
    its `per_layer` list is shorter than the stack and each entry carries
    its own `layer`. Reading a column positionally would draw a 12-layer
    profile as an 11-layer one with every feature moved left by one from
    the gap onward.
    """
    build_run(tmp_path, model="m", prompt="p", n_layers=12, n_tokens=32)
    run = discover_runs(tmp_path)[0]
    recs = records(run, "F")
    assert len(recs) == 11, "the fixture drops layer 1 on purpose"

    scattered = record_field(recs, "sharp_score", n=run.n_layers)
    assert scattered.size == 12
    assert np.isnan(scattered[1]), "the dropped layer must stay a hole"
    assert np.isfinite(scattered[0]) and np.isfinite(scattered[2])


def test_record_matrix_pads_short_rows_rather_than_raising():
    """`--f-tmax` is a CLI choice, so one directory can hold runs at two."""
    recs = [{"layer": 0, "Q_ratio": [0.9, 0.8, 0.7]},
            {"layer": 1, "Q_ratio": [0.9]},
            {"layer": 2, "Q_ratio": None}]
    m = record_matrix(recs, "Q_ratio", 3, n=3)
    assert m.shape == (3, 3)
    assert m[0].tolist() == [0.9, 0.8, 0.7]
    assert m[1, 0] == 0.9 and np.isnan(m[1, 1])
    assert np.isnan(m[2]).all()


# ─────────────────────────────────────────────────────────────────────────────
# pure — the verdict vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def test_every_verdict_the_phase_can_emit_is_classified():
    """
    A verdict string whose class the palette does not recognize renders in
    the invalid grey — a real result displayed as "not classified". The
    vocabularies are wider than they look, and two of them share a leading
    word: `sphere_license` emits both "SPHERE LICENSED" and "SPHERE NOT
    LICENSED", which is why the classifier scans past the first token.
    """
    from p1c_frames.frame_table import sphere_license
    from p1c_frames.gamma_null import adjudicate_p_gamma1
    from p1c_frames.moments import adjudicate_sink_hypothesis

    def panels(raw, npr, nrm):
        return [{"shannon_raw": a, "norm_pr": b, "shannon_normed": c}
                for a, b, c in zip(raw, npr, nrm)]

    sinks = adjudicate_sink_hypothesis(panels([5, 4, 3, 2], [5, 4, 3, 2],
                                              [40, 9, 30, 8]))
    directional = adjudicate_sink_hypothesis(panels([5, 4, 3, 2],
                                                    [9, 2, 8, 3],
                                                    [5, 4, 3, 2]))
    for v in (sinks["verdict"], directional["verdict"]):
        assert verdict_word(v) in VERDICT_COLORS, v

    for by_step in ({0: 0.01, 1: 0.05, 2: 0.2},      # CONFIRMED
                    {0: 0.2, 1: 0.05, 2: 0.01},      # FALSIFIED
                    {0: 0.01, 1: 0.3, 2: 0.05}):     # PARTIAL
        v = adjudicate_p_gamma1(by_step)["verdict"]
        assert verdict_word(v) in VERDICT_COLORS, v

    def gamma_stats(cv):
        return {"cv": cv, "condition_number": 1 + cv, "mean": 0.44}

    for cv in (0.01, 0.05, 0.5):                     # LICENSED / MARGINAL / NOT
        v = sphere_license([gamma_stats(cv)] * 4)["verdict"]
        assert verdict_word(v) in VERDICT_COLORS, v


def test_sphere_not_licensed_is_not_coloured_as_licensed():
    """The two verdicts share a first word and mean opposite things."""
    assert verdict_word("SPHERE LICENSED — gamma dispersion is within 2x") \
        == "LICENSED"
    assert verdict_word("SPHERE NOT LICENSED — gamma dispersion is 16.6x") \
        == "NOT"
    assert VERDICT_COLORS["LICENSED"] != VERDICT_COLORS["NOT"]


def test_unclassifiable_verdict_falls_back_rather_than_guessing():
    assert verdict_word("insufficient checkpoints") == ""
    assert verdict_word("") == ""


def test_every_step_definition_has_a_colour_and_a_linestyle():
    """
    The three definitions differ by ~5.7x and are read together at thumbnail
    size (status-1c finding 1), so hue alone is not enough — each needs its
    own linestyle too.
    """
    from p1c_frames.visualization.style import STEP_LABELS, STEP_STYLES

    for key in STEP_DEFS:
        assert key in STEP_COLORS and key in STEP_STYLES and key in STEP_LABELS
    assert len({STEP_STYLES[k]["linestyle"] for k in STEP_DEFS}) == 3


def test_residual_norm_is_centred_on_zero_not_on_the_data():
    """
    The residual is usually all one sign. Matplotlib would put the neutral
    colour in the middle of the data, drawing a boundary between "resisting"
    and "resisting" that is not there. Zero is the boundary.
    """
    all_negative = np.array([-0.9, -0.6, -0.4, -0.05])
    norm = residual_norm(all_negative)
    assert norm.vcenter == 0.0
    assert norm.vmin == -norm.vmax
    assert norm(0.0) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# smoke — the fixture directory
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("p1cout"))


def test_fixture_is_discoverable(fixture_dir):
    runs = discover_runs(fixture_dir)
    # 2 models x 2 prompts + a 6-checkpoint family + one degraded run.
    assert len(runs) == 4 + len(FIXTURE_STEPS) + 1
    stems = {r.stem for r in runs}
    assert "gpt2-large_wiki_paragraph" in stems
    assert "pythia-410m-step143000_wiki_paragraph" in stems
    assert all(r.model != "?" for r in runs), \
        "identity comes from the artifact, not from the directory name"


def test_discovery_filters_by_model_and_prompt(fixture_dir):
    runs = discover_runs(fixture_dir, models=["gpt2-large"],
                         prompts=["wiki_paragraph"])
    assert len(runs) == 1 and runs[0].model == "gpt2-large"


def test_the_degraded_run_reports_the_drivers_own_reasons(fixture_dir):
    """
    A run made before the `norms` fix has A, B, C and F skipped by the
    driver with its own messages, and those messages — not this package's
    paraphrase — are what a skipped figure prints.
    """
    run = next(r for r in discover_runs(fixture_dir)
               if r.stem == "gpt2_short_prompt")
    assert run.subexps == ["E"]
    assert any("norms" in m for m in run.missing)
    assert any("clusters.npz" in m for m in run.missing)
    assert run.beta_source == "fallback_flag"


def test_d_is_absent_everywhere_but_the_one_fixture_run(fixture_dir):
    """
    No driver writes a D block (FIGURES-1c.md G1). The fixture writes one
    for a single run so the `frames` class stays exercised, and the skip
    path stays exercised by every other run.
    """
    runs = discover_runs(fixture_dir)
    with_d = [r.stem for r in runs if r.has("D")]
    assert with_d == ["gpt2-large_wiki_paragraph"]
    for r in runs:
        if not r.has("D"):
            assert any("run_1c has no D branch" in m for m in r.missing)


def test_checkpoint_families_group_by_prompt_and_exclude_static_models(
        fixture_dir):
    """
    t* is n-dependent and the prompts span 20-512 tokens, so a step axis
    pooled over prompts would compare each checkpoint against a different
    collapse time. gpt2-large and albert carry no step and must not appear
    on a step axis at all.
    """
    fams = checkpoint_families(discover_runs(fixture_dir))
    assert list(fams) == ["pythia-410m"]
    assert list(fams["pythia-410m"]) == ["wiki_paragraph"]
    assert sorted(fams["pythia-410m"]["wiki_paragraph"]) == sorted(FIXTURE_STEPS)


def test_availability_is_read_from_what_landed_not_from_what_was_asked(
        fixture_dir):
    for run in discover_runs(fixture_dir):
        for sub in SUBEXPS:
            assert run.has(sub) == (sub in run.subexps)


# ─────────────────────────────────────────────────────────────────────────────
# smoke — figures
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_all_produces_every_class(fixture_dir, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from p1c_frames.visualization.pipeline import PER_RUN_CLASSES, generate_all

    runs = discover_runs(fixture_dir, models=["gpt2-large"],
                         prompts=["wiki_paragraph"])
    produced = generate_all(fixture_dir, tmp_path, runs=runs,
                            classes=list(PER_RUN_CLASSES) + ["crossrun",
                                                             "theory"],
                            cheap=True)
    for cls in PER_RUN_CLASSES:
        assert produced.get(cls), f"{cls} produced no figures"
    assert produced["crossrun"] and produced["theory"]
    for paths in produced.values():
        for p in paths:
            assert p.exists() and p.stat().st_size > 0

    assert (tmp_path / "gpt2-large_wiki_paragraph").is_dir()
    assert (tmp_path / "_cross").is_dir()
    assert (tmp_path / "_theory").is_dir()


def test_theory_needs_no_runs_at_all(tmp_path):
    """
    Phase 1c has not been run against Pythia artifacts yet (status-1c), so
    the class that draws the null model has to work with no directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    from p1c_frames.visualization.pipeline import generate_all

    produced = generate_all(None, tmp_path, classes=["theory"], cheap=True)
    assert len(produced["theory"]) == 7
    assert all(p.exists() for p in produced["theory"])


def test_unknown_class_is_rejected_rather_than_silently_skipped(fixture_dir,
                                                               tmp_path):
    from p1c_frames.visualization.pipeline import generate_all
    with pytest.raises(ValueError, match="unknown figure class"):
        generate_all(fixture_dir, tmp_path, classes=["integraton"])


def test_a_degraded_run_degrades_to_skips_rather_than_errors(fixture_dir,
                                                             tmp_path):
    """
    The run with no `norms`, no `beta_eff` and no `clusters.npz` must still
    draw everything that does not need them — which is E and the fingerprint
    — and report the rest. Never raise.
    """
    import matplotlib
    matplotlib.use("Agg")
    from p1c_frames.visualization.pipeline import generate_all

    runs = [r for r in discover_runs(fixture_dir)
            if r.stem == "gpt2_short_prompt"]
    produced = generate_all(fixture_dir, tmp_path, runs=runs,
                            classes=["integration", "null", "moments",
                                     "feasibility", "designs", "frames",
                                     "curiosities"])
    assert produced["feasibility"], "E ran on this artifact and must draw"
    assert not produced["integration"] and not produced["null"]
    assert not produced["moments"] and not produced["designs"]
    assert not produced["frames"]
    names = {p.stem for p in produced["curiosities"]}
    assert names == {"run_fingerprint"}, \
        "only the fingerprint survives a run with no A and no B"


def test_missing_envelope_skips_the_two_band_figures_with_the_runs_own_note(
        fixture_dir, tmp_path, capsys):
    """
    Without `beta_eff_per_head` the residual is a point estimate whose error
    bar is unreported, and `run_1c` records `envelope_note` saying exactly
    that (G4). The skip must print the run's own note rather than a
    paraphrase.
    """
    import matplotlib
    matplotlib.use("Agg")
    from p1c_frames.visualization.null_model import generate_null_figures

    run = next(r for r in discover_runs(fixture_dir)
               if r.stem == "albert-base-v2_wiki_paragraph")
    paths = {p.stem for p in generate_null_figures(run, tmp_path / "albert")}
    assert "residual_curve" in paths
    assert "beta_envelope" not in paths and "residual_bracket" not in paths
    assert "no per-head beta_eff" in capsys.readouterr().out


def test_checkpoint_class_reports_its_own_missing_dependency(monkeypatch,
                                                            tmp_path):
    """
    Only the `checkpoints` class needs Phase 1's step-axis helpers, and
    reaching them imports that package's whole figure surface (sklearn
    included). Every other class must draw without it.
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
        importlib.import_module("p1c_frames.visualization.checkpoints_1c"))
    assert mod.generate_checkpoint_figures([], tmp_path) == []


def test_no_figure_module_recomputes_an_analysis():
    """
    Every verdict this package prints is imported from `p1c_frames` or read
    from an artifact. A figure module defining its own threshold constant is
    how a figure starts disagreeing with the phase it illustrates — so the
    modules that draw verdicts must import them.
    """
    import p1c_frames.visualization.checkpoints_1c as k
    import p1c_frames.visualization.feasibility as e
    import p1c_frames.visualization.theory as t

    src = Path(t.__file__).read_text()
    assert "from p1c_frames.gamma_ode import" in src
    assert "def integrate_gamma" not in src, \
        "the theory class must call the phase's ODE, not restate it"

    assert Path(e.__file__).read_text().count(
        "from p1c_frames.hemisphere_feasibility import") == 1

    ksrc = Path(k.__file__).read_text()
    assert "adjudicate_p_gamma1" in ksrc and "adjudicate_p_s1_banded" in ksrc
