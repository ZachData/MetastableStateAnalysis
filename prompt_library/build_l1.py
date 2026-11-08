"""
prompt_library/build_l1.py — the prompt library's U1: draw it, pin it, check it.

Implements `prompt_library/design-L.md` D1–D4 as decided (Blocked 140, all (a), 2026-11-07):
one shared source, `monology/pile-uncopyrighted` at a pinned revision; an item is one document
with >= 512 GPT-NeoX tokens, cut to its first 512; the split's rows permuted once by
`default_rng(20261107)` and the first 1024 eligible rows kept, in that order; `test` drawn the
same way and sealed (`core/holdout.py`). No model reads anything here: text and a tokenizer only.

Writes
  prompt_library/library_v1.json   the manifest (in git): pins, the rule, per item its row,
                                   Pile subset, character end, sha256 of its text, its token count
  $METS_DATA/library/v1/val.jsonl          the item texts, one JSON line per item, in order
  $METS_DATA/library/v1/build_report.json  eligibility and mixture per subset, the checks

The cut. Each item is the document's prefix up to the end character of its 512th token, from
the fast tokenizer's offset mapping, with the ByteLevel post-processor's `trim_offsets` off:
trimming moves a span's edges inward past whitespace, so a token ending in whitespace after other
characters (3 in the vocabulary) would be cut short; untrimmed, each end is the token's own (ids
are unchanged either way). Eligibility reads the first
PREFIX_CHARS characters, exact for shorter documents; a longer one that falls short there is
re-tokenized whole, and every drawn item's cut is taken from its whole document.

    python prompt_library/build_l1.py --first      # item 0 of val only, printed; writes nothing
    python prompt_library/build_l1.py              # both splits, the manifest, the report
    python prompt_library/build_l1.py --data-dir D --manifest M   # a second build, to compare
    python prompt_library/build_l1.py --from-manifest val          # texts from the manifest alone

The sealed split's texts are not written: its manifest entries pin them, and
`--from-manifest test --allow-sealed` rebuilds them once a registry entry names the split.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core import library  # noqa: E402

DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))

SOURCE_REPO = "monology/pile-uncopyrighted"
SOURCE_REVISION = "3be90335b66f24456a5d6659d9c8d208c0357119"
SOURCE_FILES = {   # LFS sha256 at SOURCE_REVISION (docs/lit-L.md §1; full values from the HF API)
    "val": ("val.jsonl.zst", 338045152,
            "db5e5d1532bf8dc33a6589b50ecba1a8c96f7b4b9cb343d168e603c393007c26"),
    "test": ("test.jsonl.zst", 337771483,
             "a6cafe820e9c350af95b57ec75d35d7c8b479de2eb87caabe6fb6b567e639c3a"),
}
TOKENIZER_REPO = "EleutherAI/pythia-410m"
TOKENIZER_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

SEED = 20261107
N_ITEMS = 1024
CUT_TOKENS = 512
PREFIX_CHARS = 32768
BAND = 0.99


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(split: str) -> Path:
    from huggingface_hub import hf_hub_download
    name, size, sha = SOURCE_FILES[split]
    path = Path(hf_hub_download(SOURCE_REPO, name, repo_type="dataset", revision=SOURCE_REVISION))
    if path.stat().st_size != size or sha256_file(path) != sha:
        raise SystemExit(f"REFUSE: {path} does not match its pin ({size} bytes, sha256 {sha}); "
                         "no fallback: the pin is the point (design-L.md)")
    return path


def tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer, processors
    paths = {f: Path(hf_hub_download(TOKENIZER_REPO, f, revision=TOKENIZER_REVISION))
             for f in TOKENIZER_FILES}
    tok = Tokenizer.from_file(str(paths["tokenizer.json"]))
    pp = json.loads(paths["tokenizer.json"].read_text())["post_processor"]
    if pp.get("type") != "ByteLevel":
        raise SystemExit(f"REFUSE: expected a ByteLevel post-processor, found {pp}")
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    pin = {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
           "files_sha256": {f: sha256_file(p) for f, p in paths.items()},
           "add_special_tokens": False,
           "offsets": "ByteLevel post-processor with trim_offsets=False (ids unchanged)"}
    return tok, pin


def read_rows(path: Path):
    """(texts, subsets) of every row, in file order."""
    proc = subprocess.Popen(["zstd", "-dc", str(path.resolve())], stdout=subprocess.PIPE)
    texts, subsets = [], []
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        d = json.loads(line)
        texts.append(d["text"])
        meta = d.get("meta") or {}
        subsets.append(meta.get("pile_set_name") or "unknown")
    if proc.wait() != 0:
        raise SystemExit(f"REFUSE: zstd failed on {path}")
    return texts, subsets


def count_tokens(tok, texts, batch=4096):
    """Token count of each text's first PREFIX_CHARS characters; a longer text short there is
    counted whole. Returns (counts, n_recounted)."""
    counts = np.empty(len(texts), dtype=np.int64)
    for s in range(0, len(texts), batch):
        enc = tok.encode_batch([t[:PREFIX_CHARS] for t in texts[s:s + batch]],
                               add_special_tokens=False)
        counts[s:s + batch] = [len(e.ids) for e in enc]
    redo = [i for i, t in enumerate(texts) if len(t) > PREFIX_CHARS and counts[i] < CUT_TOKENS]
    for i in redo:
        counts[i] = len(tok.encode(texts[i], add_special_tokens=False).ids)
    return counts, len(redo)


def cut(tok, text):
    """(item text, its char end, ids of the whole document's first CUT_TOKENS tokens)."""
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc.ids) < CUT_TOKENS:
        raise ValueError(f"{len(enc.ids)} tokens < {CUT_TOKENS}")
    end = enc.offsets[CUT_TOKENS - 1][1]
    return text[:end], end, enc.ids[:CUT_TOKENS]


def binom_band(n, p):
    from scipy.stats import binom
    a = (1 - BAND) / 2
    return int(binom.ppf(a, n, p)), int(binom.ppf(1 - a, n, p))


def draw(split, tok, first_only=False):
    path = fetch(split)
    texts, subsets = read_rows(path)
    n_rows = len(texts)
    print(f"[{split}] {n_rows} rows read from {path.name}", flush=True)
    order = np.random.default_rng(SEED).permutation(n_rows)

    if first_only:
        for r in order:
            if len(texts[r]) >= 1 and len(tok.encode(texts[r][:PREFIX_CHARS],
                                                     add_special_tokens=False).ids) >= CUT_TOKENS:
                item, end, ids = cut(tok, texts[r])
                n = len(tok.encode(item, add_special_tokens=False).ids)
                print(json.dumps({"split": split, "rank": 0, "row": int(r), "subset": subsets[r],
                                  "char_end": end, "n_tokens": n,
                                  "sha256": hashlib.sha256(item.encode()).hexdigest(),
                                  "head": item[:300], "tail": item[-200:]}, indent=1))
                return None
        raise SystemExit("REFUSE: no eligible row")

    counts, n_recounted = count_tokens(tok, texts)
    eligible = counts >= CUT_TOKENS
    print(f"[{split}] {int(eligible.sum())} eligible ({eligible.mean():.3f}); "
          f"{n_recounted} long rows recounted whole", flush=True)

    rows = [int(r) for r in order if eligible[r]][:N_ITEMS]
    if len(rows) < N_ITEMS:
        raise SystemExit(f"REFUSE: only {len(rows)} eligible rows")
    items, lines, mismatch, ntok = [], [], [], []
    for rank, r in enumerate(rows):
        item, end, ids = cut(tok, texts[r])
        re_ids = tok.encode(item, add_special_tokens=False).ids
        if re_ids != ids:          # whole equality: a 513 with the right first 512 is not equal
            mismatch.append(rank)
        if not item.strip():
            raise SystemExit(f"REFUSE: {split} item {rank} (row {r}) is empty")
        items.append([r, subsets[r], end, hashlib.sha256(item.encode()).hexdigest(), len(re_ids)])
        lines.append(json.dumps({"rank": rank, "text": item}, ensure_ascii=False))
        ntok.append(len(re_ids))

    rows_by = Counter(subsets)
    elig_by = Counter(s for s, e in zip(subsets, eligible) if e)
    drawn_by = Counter(it[1] for it in items)
    n_elig = int(eligible.sum())
    mixture = {}
    for s in sorted(rows_by, key=lambda s: -elig_by[s]):
        p = elig_by[s] / n_elig
        lo, hi = binom_band(N_ITEMS, p)
        mixture[s] = {"rows": rows_by[s], "eligible": elig_by[s],
                      "eligible_share_of_rows": round(elig_by[s] / rows_by[s], 4),
                      "share_of_eligible": round(p, 5), "drawn": drawn_by[s],
                      "band99": [lo, hi], "in_band": lo <= drawn_by[s] <= hi}
    ntok = np.asarray(ntok)
    report = {"n_rows": n_rows, "n_eligible": n_elig, "n_recounted_whole": n_recounted,
              "last_permuted_position_used": int(np.flatnonzero(np.isin(order, rows[-1:]))[0]),
              "retokenized_alone_equals_document_ids": N_ITEMS - len(mismatch),
              "mismatch_ranks": mismatch,
              "item_tokens_retokenized": {"min": int(ntok.min()), "max": int(ntok.max()),
                                          "n_not_512": int((ntok != CUT_TOKENS).sum())},
              "mixture": mixture,
              "subsets_out_of_band": [s for s, m in mixture.items() if not m["in_band"]]}
    entry = {"n_rows": n_rows, "n_eligible": n_elig, "n_items": N_ITEMS,
             "sealed": split in library.SEALED_SPLITS,
             "file": {"path": SOURCE_FILES[split][0], "bytes": SOURCE_FILES[split][1],
                      "sha256": SOURCE_FILES[split][2]},
             "columns": list(library.COLUMNS), "items": items}
    return entry, lines, report


def tokenizers_version() -> str:
    import tokenizers
    return tokenizers.__version__


def from_manifest(split: str, data_dir: Path, manifest_path: Path, allow_sealed: bool):
    """Rebuild a split's texts by slicing the pinned file at the manifest's rows and char ends,
    each checked against its sha256; no random stream and no tokenizer enter."""
    from core.holdout import refuse_sealed_split
    refuse_sealed_split(split, allow_sealed=allow_sealed, context="build_l1 --from-manifest")
    e = library.manifest(manifest_path)["splits"][split]
    texts, _ = read_rows(fetch(split))
    if len(texts) != e["n_rows"]:
        raise SystemExit(f"REFUSE: {len(texts)} rows, the manifest says {e['n_rows']}")
    lines = []
    for rank, (row, _, end, sha, _) in enumerate(e["items"]):
        item = texts[row][:end]
        if hashlib.sha256(item.encode()).hexdigest() != sha:
            raise SystemExit(f"REFUSE: {split} item {rank} (row {row}) does not hash to the manifest")
        lines.append(json.dumps({"rank": rank, "text": item}, ensure_ascii=False))
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    n = len(library.load(split, len(lines), allow_sealed=allow_sealed, data_dir=data_dir,
                         manifest_path=manifest_path, context="build_l1 --from-manifest"))
    print(f"[{split}] {n} items rebuilt from the manifest into {data_dir}")


def write_manifest(man: dict, path: Path):
    """One item per line, so a diff shows which items moved."""
    head = {k: v for k, v in man.items() if k != "splits"}
    out = json.dumps(head, indent=1)[:-2] + ',\n "splits": {\n'
    parts = []
    for split, e in man["splits"].items():
        meta = {k: v for k, v in e.items() if k != "items"}
        body = json.dumps(meta, indent=2)[:-2]
        rows = ",\n".join("    " + json.dumps(it, ensure_ascii=False) for it in e["items"])
        parts.append(f'  "{split}": ' + body.replace("\n", "\n  ")
                     + ',\n   "items": [\n' + rows + "\n   ]\n  }")
    out += ",\n".join(parts) + "\n }\n}\n"
    json.loads(out)
    path.write_text(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--first", action="store_true", help="item 0 of val only; writes nothing")
    ap.add_argument("--data-dir", type=Path, default=DATA / "library" / "v1")
    ap.add_argument("--manifest", type=Path, default=library.MANIFEST)
    ap.add_argument("--from-manifest", metavar="SPLIT",
                    help="rebuild SPLIT's texts from the manifest alone (row, char end, sha256): "
                         "no random draw, no tokenizer")
    ap.add_argument("--allow-sealed", action="store_true",
                    help="with --from-manifest test: only once a registry entry names it")
    a = ap.parse_args()

    if a.from_manifest:
        from_manifest(a.from_manifest, a.data_dir, a.manifest, a.allow_sealed)
        return
    tok, tok_pin = tokenizer()
    if a.first:
        draw("val", tok, first_only=True)
        return

    man = {"version": "v1",
           "design": "prompt_library/design-L.md (D1-D4 (a), Blocked 140, 2026-11-07)",
           "builder": "prompt_library/build_l1.py",
           "source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION},
           "tokenizer": tok_pin,
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "tokenizers": tokenizers_version()},
           "rule": {"seed": SEED, "order": "numpy default_rng(seed).permutation(n_rows), "
                    "then the first n_items rows with >= min_tokens tokens",
                    "min_tokens": CUT_TOKENS, "cut_tokens": CUT_TOKENS,
                    "cut": "document prefix to the end character of its cut_tokens-th token",
                    "eligibility_prefix_chars": PREFIX_CHARS, "normalisation": "none",
                    "web_subsets": sorted(library.WEB_SUBSETS)},
           "splits": {}}
    a.data_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for split in ("val", "test"):
        entry, lines, report = draw(split, tok)
        man["splits"][split] = entry
        reports[split] = report
        path = a.data_dir / f"{split}.jsonl"
        if entry["sealed"]:
            # The sealed split's texts are not kept: its manifest entries pin them, and
            # --from-manifest test --allow-sealed rebuilds them once a registry entry names it.
            if path.exists():
                path.unlink()
                print(f"[{split}] removed {path}: sealed texts are not kept on disk", flush=True)
        else:
            path.write_text("\n".join(lines) + "\n")
        print(f"[{split}] {len(lines)} items{' (sealed, not written)' if entry['sealed'] else ''}; out of band: "
              f"{report['subsets_out_of_band']}; prefix mismatches: {len(report['mismatch_ranks'])}",
              flush=True)
    write_manifest(man, a.manifest)

    # Read back through the loader every reader uses: hashes, lengths, emptiness.
    # The sealed split has no texts on disk to read back; `--from-manifest test --allow-sealed`
    # into a scratch directory is its read-back (status-L.md "U1").
    for split in ("val", "test"):
        rb = {"library_hash": library.library_hash(split, N_ITEMS, manifest_path=a.manifest)}
        if not man["splits"][split]["sealed"]:
            rb["n"] = len(library.load(split, N_ITEMS, data_dir=a.data_dir,
                                       manifest_path=a.manifest, context="build_l1 read-back"))
        reports[split]["read_back"] = rb
    (a.data_dir / "build_report.json").write_text(json.dumps(reports, indent=1) + "\n")
    print(json.dumps({s: {k: v for k, v in r.items() if k != "mixture"}
                      for s, r in reports.items()}, indent=1))


if __name__ == "__main__":
    main()
