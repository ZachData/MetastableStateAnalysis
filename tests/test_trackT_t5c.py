"""T5 U2, the trajectories (`trackT_timaeus/t5c.py`, design-T5.md "U2", "U2's details"): the frozen
lists and the "spreads" rule (pure)."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports, transformers
from trackT_timaeus import t2, t5c


def test_the_frozen_lists():
    assert len(t5c.DODECA_STEPS) == 53 and t5c.DODECA_STEPS[0] == 1 and t5c.DODECA_STEPS[-1] == 9536
    g = sorted({int(round(x / 100)) * 100 for x in np.geomspace(100, 65000, 20)})
    assert t5c.GPT2_STEPS == (0,) + tuple(g) and len(g) == 19
    assert t5c.TRAJ["pythia-410m"][1][0] == 0 and len(t5c.TRAJ["pythia-410m"][1]) == 18
    assert t5c.SCORED == ("dodecahedron-32m", "dodecahedron-67m", "gpt-2-small")
    assert (t5c.SPREAD_MIN, t5c.BOOT_P5_MIN, t5c.N_BOOT, t5c.KNOWN_TOL) == (0.15, 0.05, 1000, 1e-6)
    assert t2.SCALE_STEP == t5c.SPREAD_STEP


def test_label_order():
    assert t5c.label(0.15, 0.06, 0.30) == "spreads"
    assert t5c.label(0.15, 0.05, 0.30) == "not told apart"            # 5th percentile must exceed 0.05
    assert t5c.label(0.10, 0.02, 0.149) == "does not spread"
    assert t5c.label(0.10, 0.02, 0.15) == "not told apart"
    assert t5c.label(0.20, 0.10, 0.14) == "spreads"                   # spreads is checked first


def test_mid_series_is_prompt_mean_then_median_of_middle_states():
    P = np.zeros((2, 3, 5))
    P[0, :, 1:4] = [[0.1, 0.5, 0.9], [0.3, 0.5, 0.7], [0.2, 0.5, 0.2]]
    P[0, :, 0] = P[0, :, 4] = 9.0                                   # embeddings and final state ignored
    assert abs(t5c.mid_series(P)[0] - np.median([0.2, 0.5, 0.6])) < 1e-15


def test_spread_on_noiseless_series():
    steps = 6
    X = np.tile(np.array([0.9, 0.8, 0.7, 0.6, 0.65, 0.7])[:, None], (1, 7))
    r = t5c.spread(t5c.scalar_as_states(X), np.random.default_rng(0), n_boot=50)
    assert abs(r["delta"] - 0.3) < 1e-12 and r["argmin"] == 3 and r["label"] == "spreads"
    assert abs(r["p5"] - 0.3) < 1e-12 and abs(r["p95"] - 0.3) < 1e-12
    flat = t5c.spread(t5c.scalar_as_states(np.full((steps, 7), 0.9)), np.random.default_rng(0), n_boot=50)
    assert flat["delta"] == 0 and flat["label"] == "does not spread"


def test_planted_shape():
    mu = t5c.planted_mu(t5c.DODECA_STEPS, 0.15)
    assert mu[0] == 0.90 and abs(mu.min() - 0.75) < 1e-12
    assert int(np.sum(np.isclose(mu, 0.75))) == 15                    # 1072 … 9536 at the bottom


def test_synthetic_small():
    rng = np.random.default_rng(1)
    planted = t5c.synthetic(0.0072, 0.15, rng, draws=40)
    flat = t5c.synthetic(0.0072, 0.0, rng, draws=40)
    assert planted["spreads"] >= 0.75 and flat["spreads"] == 0 and flat["does not spread"] == 1


# ---------------------------------------------------------------- U5 (design-T5.md "U5's details"; pure)

def test_u5_lists():
    assert t5c.U5_SEEDS == tuple(range(1, 10))
    assert all(t5c.U5_TRAJ[f"pythia-410m-seed{k}"][1] == t5c.STEPS for k in t5c.U5_SEEDS)
    assert len(t5c.U5_TRAJ["mistral-alias"][1]) == 36 and t5c.U5_TRAJ["mistral-alias"][1][-1] == 20000
    assert len(t5c.U5_TRAJ["mistral-battlestar"][1]) == 35 and t5c.MISTRAL_STEPS[0] == 0
    assert list(t5c.MISTRAL_STEPS) == sorted(set(t5c.MISTRAL_STEPS))
    assert t5c.revision_of("mistral-alias", 40) == "checkpoint-40"
    assert t5c.revision_of("pythia-410m-seed3", 40) == "step40"
    assert set(t5c.TRAJ) <= set(t5c.ALL_TRAJ) and not set(t5c.TRAJ) & set(t5c.U5_TRAJ)
    assert (t5c.LOSS_INIT_TOL, t5c.LOSS_DROP_MIN, t5c.PEAK_MAX_STEP) == (1.0, 3.0, 1000)


def test_shape_peak_window_and_trough_after_peak():
    steps = [0, 2, 16, 512, 1000, 2000, 4000, 143000]
    s = [0.46, 0.05, 0.81, 0.60, 0.40, 0.17, 0.20, 0.41]
    r = t5c.shape(steps, s)
    assert (r["peak_step"], r["trough_step"], r["end_step"]) == (16, 2000, 143000)
    assert abs(r["fall"] - 0.64) < 1e-12 and abs(r["late_rise"] - 0.24) < 1e-12 and r["first"] == 0.46
    r = t5c.shape(steps, [0.9, 0.1, 0.2, 0.3, 0.4, 0.95, 0.5, 0.6])     # the 2000 high is outside the window
    assert r["peak_step"] == 1000 and r["trough_step"] == 1000 and r["late_rise"] == pytest.approx(0.2)
    r = t5c.shape([0, 20, 100, 1000, 20000], [0.42, 0.98, 0.5, 0.35, 0.26])  # still falling at the end
    assert r["trough_is_last"] and r["late_rise"] is None and r["trough"] == r["end"]
    with pytest.raises(t5c.T5cRefused):
        t5c.shape([0, 2000], [0.1, 0.2])


def test_rec_diff_absolute_below_one_relative_above():
    ref = {k: [0.5, 0.5] for k in t5c.GATE_KEYS}
    ref["n0"] = [20.0, 1.0]
    got = {k: list(v) for k, v in ref.items()}
    got["n0"] = [20.0 + 2e-5, 1.0]
    assert t5c.rec_diff(got, ref) == pytest.approx(1e-6)
    got["rho"] = [0.5, 0.5 + 3e-6]
    assert t5c.rec_diff(got, ref) == pytest.approx(3e-6)
    got["m"] = [0.5]
    assert t5c.rec_diff(got, ref) == float("inf")
    got["m"] = [0.5, 0.5]
    got["n0"] = [float("nan"), 1.0]
    assert t5c.rec_diff(got, ref) == float("inf")                     # NaN refuses, not 0
    assert t5c.rec_diff(ref, got) == float("inf")


def _rec(step, rho_mid, loss=None):
    r = {"step": step, "rho_mid": rho_mid, "rho_mid_no0": rho_mid, "rho_15": rho_mid,
         "m": [0, 0.1, 0.2, 0], "n0": [1, 2.0, 3.0, 1], "rho": [0.1, rho_mid, rho_mid, 0.2],
         "rho_no0": [0.1, rho_mid, rho_mid, 0.2], "m_no0": [0, 0.1, 0.2, 0]}
    if loss is not None:
        r["loss"] = loss
    return r


def test_describe_and_gate_on_fabricated_records(tmp_path):
    recs = {}
    for i, n in enumerate(t5c.u5_names()):
        steps = t5c.ALL_TRAJ[n][1]
        loss = n in t5c.MISTRAL
        recs[n] = [_rec(s, 0.3 + 0.01 * i + (0.4 if 0 < s <= 32 else 0.0), (10.9 if s == 0 else 4.0) if loss else None)
                   for s in steps]
    u2, fl = tmp_path / "u2" / "pythia-410m", tmp_path / "floor"
    u2.mkdir(parents=True)
    fl.mkdir()
    import json
    for r in recs["pythia-410m"]:
        (u2 / f"step{r['step']}.json").write_text(json.dumps(r))
    for k in t5c.U5_SEEDS:
        for s in t5c.FLOOR_STEPS:
            (fl / f"seed{k}_step{s}.json").write_text(json.dumps(recs[f"pythia-410m-seed{k}"][t5c.STEPS.index(s)]))
    g = t5c.u5_gate(recs, tmp_path / "u2", fl)
    assert g["pass"] and g["seed0_vs_u2"]["pass"] and g["mistral_load"]["pass"]
    d = t5c.describe_u5(recs)
    t = d["pythia_410m_seeds"]["rho_mid"]
    assert t["min"][0] == pytest.approx(0.30) and t["max"][0] == pytest.approx(0.39)
    assert t["seed0_rank_from_low"][0] == 0 and t["argmax_seed"][0] == 9
    assert d["shapes"]["pythia-410m"]["rho_mid"]["peak_step"] == 2
    assert d["mistral_pair"]["max_abs_diff"] == pytest.approx(0.01) and len(d["mistral_pair"]["common_steps"]) == 35
    assert d["shapes"]["mistral-alias"]["rho_mid"]["late_rise"] == pytest.approx(0.0)
    recs["mistral-battlestar"][-1]["loss"] = 8.5                         # did not learn: the load check refuses
    assert not t5c.u5_gate(recs, tmp_path / "u2", fl)["pass"]
    recs["mistral-battlestar"][-1]["loss"] = 4.0
    bad = dict(recs["pythia-410m-seed4"][t5c.STEPS.index(143000)], n0=[1, 2.0, 3.0 + 1e-4, 1])
    (fl / "seed4_step143000.json").write_text(json.dumps(bad))           # a floor record that differs: refuses
    g = t5c.u5_gate(recs, tmp_path / "u2", fl)
    assert not g["pass"] and not g["seeds_vs_floor"]["pass"] and g["seed0_vs_u2"]["pass"]
    assert g["seeds_vs_floor"]["max_diff"]["seed4@143000"] == pytest.approx(1e-4 / (3.0 + 1e-4))  # relative to the floor record
