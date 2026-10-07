"""
`p1e_energy_field/long_prompts_1e.py` — the four new long passages, built as its rule says.
Synthetic sources in a temporary cache; no network, no tokenizer download.
"""
import json

import pytest

from core.holdout import HELD_OUT_PROMPT_KEYS
from p1e_energy_field import long_prompts_1e as lp

# Tier: stdlib only (the 8-passage loader, which imports p1d's package, is not called here).
pytestmark = pytest.mark.pure


def _pg(tmp_path, n, header, body):
    (tmp_path / f"pg{n}.txt").write_text("\r\n".join(header + ["", "*** START", ""] + body
                                                      + ["*** END OF THE PROJECT GUTENBERG EBOOK X"]))


def test_footnote_markers_go_numbers_after_a_space_stay():
    s = "as though he were a god.57 She said, Alcinous;58 and sleep64 during 20 days."
    assert lp.FOOTNOTE.sub("", s) == "as though he were a god. She said, Alcinous; and sleep during 20 days."


def test_odyssey_starts_after_heading_and_summary_and_drops_later_headings(tmp_path):
    n, header, heading = lp.PG["odyssey_butler_long"]
    _pg(tmp_path, n, list(header), ["BOOK VI", "", "Earlier.", "", "", "BOOK VII", "", "",
                                    "RECEPTION OF ULYSSES AT THE PALACE OF KING ALCINOUS.", "", "",
                                    "Thus, then, did Ulysses", "wait.57", "", "Second para.", "",
                                    "BOOK VIII", "", "SUMMARY LINE.", "", "Third para."])
    assert lp.units("odyssey_butler_long", tmp_path) == (
        " ", ["Thus, then, did Ulysses wait.", "Second para.", "Third para."])


def test_heading_must_be_found_exactly_once(tmp_path):
    n, header, _ = lp.PG["odyssey_butler_long"]
    _pg(tmp_path, n, list(header), ["BOOK VI", "", "Text."])
    with pytest.raises(ValueError, match="found 0 times"):
        lp.units("odyssey_butler_long", tmp_path)


def test_wrong_header_refuses(tmp_path):
    n, header, heading = lp.PG["darwin_origin_long"]
    _pg(tmp_path, n, ["Title: Something else", header[1]], list(heading) + ["", "Summary.", "", "Body."])
    with pytest.raises(ValueError, match="header lacks"):
        lp.units("darwin_origin_long", tmp_path)


def test_darwin_drops_the_summary_paragraph(tmp_path):
    n, header, heading = lp.PG["darwin_origin_long"]
    _pg(tmp_path, n, list(header), ["CHAPTER IV.", "NATURAL SELECTION.", "", "",
                                    "Natural Selection: its power", "compared.", "", "How will the", "struggle"])
    assert lp.units("darwin_origin_long", tmp_path)[1] == ["How will the struggle"]


def test_hamlet_keeps_lines_strips_indent_and_scene_headings(tmp_path):
    n, header, heading = lp.PG["hamlet_long"]
    _pg(tmp_path, n, list(header), ["ACT I", "", heading[0], "", "", "Enter Francisco.", "",
                                    "BARNARDO.", "   Who’s there?", "", "SCENE II. A room of state.", "",
                                    "KING."])
    joiner, us = lp.units("hamlet_long", tmp_path)
    assert joiner == ""
    assert "".join(us).rstrip("\n") == "Enter Francisco.\n\nBARNARDO.\nWho’s there?\n\n\nKING."


def test_horla_starts_on_8_mai_and_collapses_whitespace(tmp_path):
    page = ('<p>LE HORLA</p><p>. . . . .\n</p><p>8 <i>mai</i>. — Quelle journée&#160;! aux '
            '<span class="pagenum"></span>&#32;\nnourritures.\n</p><p>12 mai. — Suite.</p>')
    (tmp_path / "horla.json").write_text(json.dumps({"parse": {"text": page, "revid": 1}}))
    assert lp.units("horla_long", tmp_path) == (
        " ", ["8 mai. — Quelle journée\xa0! aux nourritures.", "12 mai. — Suite."])


def test_build_one_stops_on_the_cap_with_whole_units(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "units", lambda key, cache: (" ", ["a b", "c d e", "f"]))
    monkeypatch.setattr(lp, "MAX_TOKENS", 5)
    r = lp.build_one("darwin_origin_long", lambda s: s.split(), tmp_path)
    assert r["text"] == "a b c d e" and r["units_used"] == 2 and not r["source_exhausted"]


def test_keys_are_not_held_out():
    for k in lp.KEYS:
        assert not any(h in k for h in HELD_OUT_PROMPT_KEYS)


def test_built_texts_match_provenance():
    prov = lp.load_provenance()
    assert sorted(prov["prompts"]) == sorted(lp.KEYS)
    assert all(0 < v["n_tokens"] <= lp.MAX_TOKENS for v in prov["prompts"].values())
    assert sorted(lp.load()) == sorted(lp.KEYS) and all(lp.load().values())
