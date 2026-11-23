"""C4′ (d)+(f)'s pure rules (trackC_writeup/c4pdf_rules.py; design-C4pdf.md "The readings")."""
import numpy as np
import pytest

from trackC_writeup import c4pdf_rules as rl

pytestmark = pytest.mark.deps  # a1a's imports


def _mat(rows):
    return np.asarray(rows, int)


def test_grid_d():
    assert len(rl.GRID_D) == 65 and rl.GRID_D[0] == 0 and rl.GRID_D[-1] == 1000
    assert 128 in rl.GRID_D and 124 in rl.GRID_D and 144 in rl.GRID_D and 288 in rl.GRID_D


def test_transient_span_counts_updates():
    states = [0, 32, 48, 64, 80]
    blip = _mat([[0] * 6, [0] * 6, [0, 0, 0, 7, 0, 0], [0] * 6, [0] * 6])
    assert not rl.transient(blip, states)["present"]                  # one state, span 0
    run = _mat([[0] * 6, [0] * 6, [0, 0, 0, 4, 0, 0], [0, 7, 7, 7, 7, 6], [0] * 6])
    t = rl.transient(run, states)
    assert t["present"] and t["onset"] == 48 and t["end"] == 64       # span 16
    endset = _mat([[0] * 6, [0, 7, 7, 0, 0, 0], [0, 7, 7, 0, 0, 0], [0, 7, 7, 0, 0, 0], [0] * 6])
    assert not rl.transient(endset, states)["present"]                 # {T1, T2} only
    short = _mat([[0] * 6, [0, 0, 0, 7, 0, 0], [0, 0, 0, 7, 0, 0], [0] * 6, [0] * 6])
    assert not rl.transient(short, [0, 4, 8, 12, 16])["present"]       # span 4 < 16


def test_bands_and_refusal():
    assert not rl.bands(32, 48, 0.5)["separable"]                     # C4's grid at ½
    assert rl.bands(44, 48, 0.5)["separable"] and rl.bands(32, 48, 0.1)["separable"]
    b = rl.bands(44, 48, 0.5)
    assert b["step"] == (35.2, 60.0) and b["dose"][1] == 120.0
    assert rl.label_onset({"present": True, "onset": 48}, (32, 48), 0.5, 1000)["label"] == "refused"
    assert rl.label_onset({"present": True, "onset": 96}, (44, 48), 0.5, 1000)["label"] == "dose"
    assert rl.label_onset({"present": True, "onset": 48}, (44, 48), 0.5, 1000)["label"] == "step"
    assert rl.label_onset({"present": True, "onset": 200}, (44, 48), 0.5, 1000)["label"] == "neither"
    assert rl.label_onset({"present": False}, (60, 64), 0.1, 1000)["label"] == "none"
    assert rl.label_onset({"present": False}, (60, 64), 0.1, 768)["label"].startswith("unreadable")


def test_joint():
    per = {"a": {"Fh": {"label": "dose"}, "Ft": {"label": "none"}},
           "b": {"Fh": {"label": "dose"}, "Ft": {"label": "none"}}}
    assert rl.d1_joint(per, ["a", "b"])["joint"].startswith("a threshold between a tenth and a half")
    per["b"]["Ft"]["label"] = "dose"
    assert rl.d1_joint(per, ["a", "b"])["joint"] == "mixed"


def test_induction_labels():
    assert rl.label_induction(640, 0.5, 1000)["label"] == "dose"
    assert rl.label_induction(320, 0.5, 1000)["label"] == "step"
    assert rl.label_induction(None, 0.5, 1000)["label"].startswith("not formed")
    assert rl.label_induction(None, 0.1, 1000)["label"] == "not step"
    assert rl.label_induction(512, 0.1, 1000)["label"] == "neither"
    assert rl.bar_date([0.1, 0.4, 0.5], [0, 4, 8], 0.45) == 8


def test_f_and_stretch():
    assert rl.f_label({"present": True}, {"present": False}) == "not the early state's"
    assert rl.f_label({"present": False}, {"present": False}) == "the early state's"
    src = [0, 48, 64]
    m = _mat([[0] * 6, [0, 0, 0, 7, 0, 0], [0, 0, 0, 7, 0, 0]])
    st = rl.stretch(m, src, [0, 96, 100, 128], 0.5)
    assert st[1].tolist() == m[1].tolist() and st[3].tolist() == m[2].tolist()
    with pytest.raises(rl.C4pdfRuleRefused):
        rl.previous([0, 4], 0)
