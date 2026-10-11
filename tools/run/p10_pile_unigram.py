#!/usr/bin/env python3
"""
Unigram counts under Pythia's tokenizer over the first 10^8 tokens of the Pile (`design-10.md`
"F13r", the frequency input). `p10_token_composition` found no corpus unigram count on disk and used
the merge rank as a proxy; this is the count.

The source is `monology/pile-uncopyrighted` `train/00.jsonl.zst` (the Pile less its copyrighted
parts), streamed through `curl | zstd -dc` and never stored. Each document's text is tokenized
whole (no separator, no special tokens) with the local `tokenizer.json`
(`p10_token_composition.find_tokenizer`); documents are taken in file order until the total
reaches N, the last one cut at N.

    python tools/run/p10_pile_unigram.py --out <dir> [--tokens 100000000]

Writes `<dir>/counts.npy` (int64, one per id) and `<dir>/meta.json` (total, documents, source,
tokenizer sha, counts sha, the five commonest tokens).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

URL = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/00.jsonl.zst"
N_TOKENS = 10 ** 8
BATCH = 2000                                                # documents per encode_batch (placed)
COMMON = ("Ġthe", ",", ".")                                 # must be among the five commonest (the rule's check)


class UnigramError(RuntimeError):
    pass


def count_tokens(id_lists: Iterable[Sequence[int]], vocab_size: int, limit: int) -> tuple:
    """(counts, total, documents) over id lists in order, the last list cut so total ≤ limit."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    total = docs = 0
    for ids in id_lists:
        if total >= limit:
            break
        ids = np.asarray(ids[: limit - total], dtype=np.int64)
        counts += np.bincount(ids, minlength=vocab_size)
        total += int(ids.size)
        docs += 1
    return counts, total, docs


def check_counts(counts: np.ndarray, total: int, id_to_tok: dict) -> List[str]:
    if int(counts.sum()) != total:
        raise UnigramError(f"counts sum to {int(counts.sum())}, not {total}")
    top = [id_to_tok.get(int(i), "?") for i in np.argsort(-counts, kind="stable")[:5]]
    missing = [t for t in COMMON if t not in top]
    if missing:
        raise UnigramError(f"five commonest {top}: {missing} missing (not populated?)")
    return top


def stream_texts(url: str, proc_out) -> Iterable[str]:
    for line in proc_out:
        yield json.loads(line)["text"]


def batched(it: Iterable[str], n: int) -> Iterable[List[str]]:
    buf: List[str] = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tokenizers import Tokenizer
    from tools.run.p10_token_composition import find_tokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=N_TOKENS)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(DATA / "hf")))
    ap.add_argument("--url", default=URL)
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    tok_path = find_tokenizer(Path(a.hf_home))
    tok = Tokenizer.from_file(str(tok_path))
    vsize = tok.get_vocab_size(with_added_tokens=True)
    id_to_tok = {i: t for t, i in tok.get_vocab(with_added_tokens=True).items()}
    a.out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(f"curl -sfL {a.url} | zstd -dc", shell=True, stdout=subprocess.PIPE,
                            text=True, encoding="utf-8")

    def ids():
        done = 0
        for batch in batched(stream_texts(a.url, proc.stdout), BATCH):
            for e in tok.encode_batch(batch, add_special_tokens=False):
                yield e.ids
                done += len(e.ids)
            print(f"  {done:,} tokens, {time.monotonic() - t0:.0f}s", flush=True)

    counts, total, docs = count_tokens(ids(), vsize, a.tokens)
    proc.stdout.close()                                     # curl and zstd stop on SIGPIPE
    proc.kill()
    if total < a.tokens:
        raise UnigramError(f"stream ended at {total} tokens, short of {a.tokens}")
    top = check_counts(counts, total, id_to_tok)
    np.save(a.out / "counts.npy", counts)
    meta = {"rule": 'design-10.md "F13r"', "source": a.url, "total": total, "documents": docs,
            "vocab_size": vsize, "zero_count_ids": int((counts == 0).sum()), "top5": top,
            "tokenizer": str(tok_path),
            "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest(),
            "counts_sha256": hashlib.sha256((a.out / "counts.npy").read_bytes()).hexdigest(),
            "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12],
            "seconds": round(time.monotonic() - t0, 1)}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
