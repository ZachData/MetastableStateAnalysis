"""core/library.py: the prompt library loads only through its pins; its sealed split refuses."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from core import library
from core.holdout import HoldoutError, SealedSplitError

pytestmark = pytest.mark.pure

REPO = Path(__file__).resolve().parents[1]


def _fake(tmp_path, texts=("alpha beta", "gamma delta", "epsilon"), split="val"):
    items = [[10 + i, "Pile-CC" if i == 0 else "ArXiv", len(t),
              hashlib.sha256(t.encode()).hexdigest(), 3] for i, t in enumerate(texts)]
    split_entry = {"columns": list(library.COLUMNS), "items": items,
                   "file": {"sha256": "f" * 64}, "sealed": split == "test"}
    man = {"version": "v1", "splits": {split: split_entry}}
    mpath = tmp_path / "library_v1.json"
    mpath.write_text(json.dumps(man))
    d = tmp_path / "data"
    d.mkdir()
    (d / f"{split}.jsonl").write_text(
        "".join(json.dumps({"rank": i, "text": t}) + "\n" for i, t in enumerate(texts)))
    return mpath, d


def test_load_a_prefix(tmp_path):
    m, d = _fake(tmp_path)
    got = library.load("val", 2, data_dir=d, manifest_path=m)
    assert [it.text for it in got] == ["alpha beta", "gamma delta"]
    assert [it.key for it in got] == ["val_0000", "val_0001"]
    assert got[0].web and not got[1].web


def test_hash_names_the_prefix(tmp_path):
    m, _ = _fake(tmp_path)
    h2, h3 = (library.library_hash("val", k, manifest_path=m) for k in (2, 3))
    assert h2 != h3 and len(h2) == 64
    # The first k entries alone set the hash: changing item 3 leaves k = 2's.
    man = json.loads(m.read_text())
    man["splits"]["val"]["items"][2][3] = "0" * 64
    m.write_text(json.dumps(man))
    assert library.library_hash("val", 2, manifest_path=m) == h2
    assert library.library_hash("val", 3, manifest_path=m) != h3


@pytest.mark.parametrize("k", [0, 4, -1, 1.0, True])
def test_k_outside_the_library_refuses(tmp_path, k):
    m, d = _fake(tmp_path)
    with pytest.raises(library.LibraryError):
        library.load("val", k, data_dir=d, manifest_path=m)


def _rewrite(d, i, text):
    p = d / "val.jsonl"
    lines = p.read_text().splitlines()
    lines[i] = json.dumps({"rank": i, "text": text})
    p.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("text,match", [
    ("gamma delta!", "characters"),   # long
    ("gamma", "characters"),          # short
    ("gamma Delta", "sha256"),        # same length, drifted
    ("   ", "empty"),
])
def test_drift_refuses(tmp_path, text, match):
    m, d = _fake(tmp_path)
    _rewrite(d, 1, text)
    assert library.load("val", 1, data_dir=d, manifest_path=m)   # item 0 is still fine
    with pytest.raises(library.LibraryError, match=match):
        library.load("val", 2, data_dir=d, manifest_path=m)


def test_missing_or_truncated_file_refuses(tmp_path):
    m, d = _fake(tmp_path)
    p = d / "val.jsonl"
    p.write_text(p.read_text().splitlines()[0] + "\n")
    with pytest.raises(library.LibraryError, match="holds 1 items"):
        library.load("val", 2, data_dir=d, manifest_path=m)
    p.unlink()
    with pytest.raises(library.LibraryError, match="missing"):
        library.load("val", 1, data_dir=d, manifest_path=m)


def test_the_sealed_split_refuses(tmp_path):
    m, d = _fake(tmp_path, split="test")
    with pytest.raises(SealedSplitError, match="sealed"):
        library.load("test", 1, data_dir=d, manifest_path=m)
    assert issubclass(SealedSplitError, HoldoutError)
    assert library.load("test", 1, allow_sealed=True, data_dir=d, manifest_path=m)


# Only the builder may pass allow_sealed=True (design-L.md "Fixed now", the sealed split).
SEALED_SCOPE = ("tools", "trackA_force", "trackT_timaeus", "prompt_library")
SEALED_ALLOWED = {"prompt_library/build_l1.py"}


def test_only_the_builder_unseals():
    # Any value but a literal False counts: `allow_sealed=a.allow_sealed` unseals too.
    pat = re.compile(r"allow_sealed\s*=\s*(?!False\b)[\w.]")
    hits = {str(p.relative_to(REPO)) for d in SEALED_SCOPE
            for p in (REPO / d).rglob("*.py") if pat.search(p.read_text(errors="ignore"))}
    assert hits <= SEALED_ALLOWED, f"unseal the confirmation split: {sorted(hits - SEALED_ALLOWED)}"
    assert SEALED_ALLOWED <= hits, "the builder no longer reads its sealed split back"


def test_the_committed_manifest():
    """The pinned manifest is whole: two splits of 1024 distinct rows, test sealed."""
    man = library.manifest()
    assert man["rule"]["seed"] == 20261107 and man["rule"]["cut_tokens"] == 512
    assert set(man["splits"]) == {"val", "test"}
    for split, e in man["splits"].items():
        assert tuple(e["columns"]) == library.COLUMNS
        assert e["n_items"] == len(e["items"]) == 1024
        assert e["sealed"] == (split in library.SEALED_SPLITS)
        rows = [it[0] for it in e["items"]]
        assert len(set(rows)) == 1024 and all(0 <= r < e["n_rows"] for r in rows)
        assert all(re.fullmatch(r"[0-9a-f]{64}", it[3]) for it in e["items"])
        # The cut is at the document's 512th token; re-tokenized alone, a prefix ending in a
        # whitespace run or inside a multi-byte character gives 511 or 513 (status-L.md "U1").
        assert all(it[2] > 0 and 511 <= it[4] <= 513 for it in e["items"])
    # As built 2026-11-07 (status-L.md "U1"): a change to any of these is a different library.
    counts = {s: Counter(it[4] for it in e["items"]) for s, e in man["splits"].items()}
    assert counts == {"val": {512: 1010, 511: 14}, "test": {512: 1008, 511: 14, 513: 2}}
    assert library.library_hash("val", 1024).startswith("47feaa321fa0")
    assert library.library_hash("test", 1024).startswith("538e14844925")
