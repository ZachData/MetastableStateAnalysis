"""C4′ (d)+(f)'s pure rules (trackC_writeup/c4pdf_rules.py; design-C4pdf.md "The readings")."""
import json

import numpy as np
import pytest

from trackC_writeup import c4pdf_rules as rl

pytestmark = pytest.mark.deps  # a1a's imports


def _mat(rows):
    return np.asarray(rows, int)


def test_grid_d():
    assert len(rl.GRID_D) == 77 and rl.GRID_D[0] == 0 and rl.GRID_D[-1] == 1000
    assert {124, 128, 144, 400, 416, 640, 656, 688, 976} <= set(rl.GRID_D)
    assert 672 not in rl.GRID_D and 130 not in rl.GRID_D


def test_interval_labels():
    # a point at 416 is in the dose band [409.6, 800], but (400, 416] is not wholly inside it
    assert rl.label_induction(416, 0.5, 1000)["label"] == "dose"
    assert rl.label_induction(416, 0.5, 1000, prev=400)["label"] == "neither"
    assert rl.label_induction(432, 0.5, 1000, prev=416)["label"] == "dose"
    assert rl.label_onset({"present": True, "onset": 96}, (44, 48), 0.5, 1000, prev=92)["label"] == "dose"
    assert rl.label_onset({"present": True, "onset": 64}, (44, 48), 0.5, 1000, prev=60)["label"] == "neither"


def test_d1_presence():
    cell = lambda h, t: {"Fh": {"transient": {"present": h}}, "Ft": {"transient": {"present": t}}}
    assert rl.d1_presence({"a": cell(True, False), "b": cell(True, False)}, ["a", "b"])["holds"]
    assert not rl.d1_presence({"a": cell(True, False), "b": cell(True, True)}, ["a", "b"])["holds"]


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


def test_registration_guard(tmp_path):
    from trackC_writeup import c4pdf_train as pt
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"predictions": [{"id": "X-1"}]}))
    ent = tmp_path / "entries.json"
    assert pt.registration(ent, reg, check_git=False) == {"d1": "pending", "d2": "pending"}
    with pytest.raises(pt.C4pdfRefused):
        pt.require_decided(entries=ent, registry=reg, check_git=False)
    ent.write_text(json.dumps({"d1": "X-1", "d2": "X-2"}))           # X-2 not in the registry
    assert pt.registration(ent, reg, check_git=False) == {"d1": "registered as X-1", "d2": "pending"}
    with pytest.raises(pt.C4pdfRefused):
        pt.require_decided(entries=ent, registry=reg, check_git=False)
    ent.write_text(json.dumps({"d1": "X-1", "d2": "declined"}))
    assert pt.require_decided(entries=ent, registry=reg, check_git=False) == {"d1": "registered as X-1", "d2": "declined"}
    assert set(pt.GUARDED) == {"Fh", "Ft"}


def test_k_rule_and_gate():
    """U3b's rule K and G-K (trackC_writeup/c4pdf_bk.py), frozen before the rebuild or B_K ran."""
    from trackC_writeup import c4pdf_bk as bk
    assert bk.UNTIL == 384 and bk.SAVES == (256, 257, 258, 260, 264, 268, 272, 280, 288, 304, 320, 336, 352, 384)
    yes, no = {"present": True}, {"present": False}
    assert bk.k_label(yes, no) == "the jump's" and bk.k_label(no, no) == "the restart's"
    assert bk.k_label(yes, yes).startswith("unreadable") and bk.k_label(no, yes).startswith("unreadable")
    assert bk.k_joint({"a": "the jump's", "b": "the jump's"}) == "the jump's"
    assert bk.k_joint({"a": "the jump's", "b": "the restart's"}) == "mixed"
    st = list(bk.SAVES) + [416, 448, 512]
    hot = np.zeros((len(st), 7), int)
    hot[7:, 3] = 7                                   # a run 280 → 512 at T3
    cold = np.zeros((len(st), 7), int)
    assert bk.g_k({"t": hot}, st, {"t": cold}, st, ["t"])["pass"]
    late = np.zeros((len(st), 7), int)
    late[-3:, 3] = 7                                 # a run 416 → 512: cut at 384, nothing
    assert not bk.g_k({"t": late}, st, {"t": cold}, st, ["t"])["pass"]
    assert not bk.g_k({"t": hot}, st, {"t": hot}, st, ["t"])["pass"]
