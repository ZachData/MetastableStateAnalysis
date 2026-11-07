"""
core/library.py — the shared prompt library, loaded only through its pins.

The library (`prompt_library/design-L.md`, built by `prompt_library/build_l1.py`) is 1024 single
Pile documents per split, each cut to its first 512 GPT-NeoX tokens, in a seeded order. Its
manifest, `prompt_library/library_v1.json`, is in git; the texts are not
(`$METS_DATA/library/v1/{split}.jsonl`), and are rebuilt from the pinned source file.

A reader names a prefix k in its design and calls `load(split, k)`: the first k items in library
order, each re-hashed against the manifest. It refuses (LibraryError) on a missing file, fewer
lines than k, an empty item, a text whose length or sha256 differs from the manifest, or a k
outside 1..n_items. `library_hash(split, k)` hashes the manifest's first k entries, so a record
names its exact input (CLAUDE.md "Name the input"); a prefix of the order is itself a random draw,
so every k samples the same population.

`test` is the sealed confirmation split (design-L.md D4): `load` refuses it without
`allow_sealed=True` (`core/holdout.py`), and `tests/test_library.py` checks that only the
builder passes that.

Kept apart from `core.config.PROMPTS`: the battery, its hash and its readers are unchanged.
Torch- and numpy-free.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.holdout import SEALED_LIBRARY_SPLITS, refuse_sealed_split

__all__ = ["MANIFEST", "COLUMNS", "SEALED_SPLITS", "WEB_SUBSETS", "LibraryError", "Item",
           "default_data_dir", "manifest", "library_hash", "load"]

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "prompt_library" / "library_v1.json"
COLUMNS = ("row", "subset", "char_end", "sha256", "n_tokens")
SEALED_SPLITS = SEALED_LIBRARY_SPLITS
# Pile-CC is the subset most likely to overlap a web-crawl training set (FineWeb); a flag,
# not a guarantee: any crawl can hold a page from another subset (design-L.md D1).
WEB_SUBSETS = frozenset({"Pile-CC"})


class LibraryError(RuntimeError):
    """The library on disk does not match its manifest, or a request is outside it."""


@dataclass(frozen=True)
class Item:
    key: str          # "<split>_<rank:04d>", stable across builds
    rank: int         # position in library order
    row: int          # row index in the pinned source file
    subset: str       # meta.pile_set_name, or "unknown"
    text: str
    sha256: str
    n_tokens: int     # GPT-NeoX tokens of `text` (pinned tokenizer, no special tokens)

    @property
    def web(self) -> bool:
        return self.subset in WEB_SUBSETS


def default_data_dir() -> Path:
    return Path(os.environ.get("METS_DATA", str(REPO / "data"))) / "library" / "v1"


def manifest(manifest_path: Optional[Path] = None) -> dict:
    path = Path(manifest_path or MANIFEST)
    try:
        man = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise LibraryError(f"cannot read the library manifest {path}: {e}") from e
    return man


def _split(man: dict, split: str) -> dict:
    try:
        e = man["splits"][split]
    except KeyError:
        raise LibraryError(f"no split {split!r} in the manifest "
                           f"(has {sorted(man.get('splits', {}))})") from None
    if tuple(e["columns"]) != COLUMNS:
        raise LibraryError(f"manifest columns {e['columns']} != {COLUMNS}")
    return e


def _check_k(e: dict, split: str, k: int) -> None:
    n = len(e["items"])
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
        raise LibraryError(f"k = {k!r} outside 1..{n} for split {split!r}: "
                           "a reader names its prefix k (design-L.md D3)")


def library_hash(split: str, k: int, *, manifest_path: Optional[Path] = None) -> str:
    """sha256 over the version, split, source file and the manifest's first k entries."""
    man = manifest(manifest_path)
    e = _split(man, split)
    _check_k(e, split, k)
    body = {"version": man["version"], "split": split, "file_sha256": e["file"]["sha256"],
            "columns": list(COLUMNS), "items": e["items"][:k]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def load(split: str, k: int, *, allow_sealed: bool = False, context: str = "",
         data_dir: Optional[Path] = None, manifest_path: Optional[Path] = None) -> List[Item]:
    """The first k items of `split`, each checked against the manifest."""
    refuse_sealed_split(split, allow_sealed=allow_sealed, context=context)
    e = _split(manifest(manifest_path), split)
    _check_k(e, split, k)
    path = Path(data_dir or default_data_dir()) / f"{split}.jsonl"
    if not path.is_file():
        raise LibraryError(f"{path} is missing: rebuild it with prompt_library/build_l1.py "
                           "(it re-draws from the pinned source file and must reproduce the "
                           "manifest's hashes)")
    out: List[Item] = []
    with open(path, encoding="utf-8") as f:
        for rank, line in zip(range(k), f):
            d = json.loads(line)
            row, subset, char_end, sha, n_tok = e["items"][rank]
            text = d.get("text", "")
            if d.get("rank") != rank:
                raise LibraryError(f"{path}: line {rank} carries rank {d.get('rank')!r}")
            if not text.strip():
                raise LibraryError(f"{path}: item {rank} is empty")
            if len(text) != char_end:
                raise LibraryError(f"{path}: item {rank} has {len(text)} characters, "
                                   f"the manifest {char_end} (short or long)")
            got = hashlib.sha256(text.encode()).hexdigest()
            if got != sha:
                raise LibraryError(f"{path}: item {rank} sha256 {got[:12]} != manifest "
                                   f"{sha[:12]}: the text drifted; rebuild, do not read it")
            out.append(Item(f"{split}_{rank:04d}", rank, row, subset, text, sha, n_tok))
    if len(out) < k:
        raise LibraryError(f"{path} holds {len(out)} items, {k} asked")
    return out
