"""trackA_force/a3_ngram.py — A3 unit 4, the n-gram rows' producer (`design-A3.md` §2 E6u, E6b;
§4 "Placed before any unit 4 output"; §5's corpus).

- **Corpus:** the Pile's validation split (`val.jsonl.zst`, sha256 ``CORPUS_SHA``), documents in
  file order, each tokenized with 410m's tokenizer and followed by ``<|endoftext|>``; a document
  sharing any ``OVERLAP_N``-token run with a v1 prompt is dropped. The **scored set** is
  ``N_BLOCKS`` blocks of ``BLOCK`` tokens, each starting at a document's start and running into
  its successors (the rest of its last document discarded), so no document is in two blocks; the
  **table set** is the documents after them, until ``N_TABLE`` tokens.
- **Predictors:** q_u, the table set's token frequencies (MLE); q_b(·|v), Witten–Bell
  interpolated with q_u, (c(v, ·) + T(v) q_u) / (c(v) + T(v)), q_u where c(v) = 0.
- **E6u, E6b:** KL(q ‖ p_model) in nats at every position of a block, mean over the block; one
  value per block and step (the bootstrap's units in `a3.py`). Beside: the model's next-token
  loss per block.
- **The full table (unit 5, Blocked 74 (a); `design-A3.md` §4 "Placed before any unit 5
  output"):** with ``--scored <unit 4 dir>``, the table set′ is every kept document of the split
  that is not in that dir's scored set, in file order; its first ``N_TABLE`` tokens must be unit
  4's table, and the blocks are unit 4's. The records are written as in unit 4 (E6u′, E6b′).

KL(q ‖ p) = Σ q log q − Σ q log p. The first term does not depend on the step and is computed
once per context token; the second is ``logp @ q_u`` for the unigram and, for the bigram,
(Σ_w c(v, w) log p(w) + T(v) · logp @ q_u) / (c(v) + T(v)), from the bigram counts' sparse row.

Every step's record is written as it finishes, so a stopped run resumes. Model float32 on the GPU
(`docs/compute_profile.md` "The GPU": no stored CPU cloud is compared), TF32 off, KL in float64.

Tier 1: exploratory, unregistered. v1 prompts are read only for the overlap check.
    python -m trackA_force.a3_ngram run --corpus <val.jsonl.zst> --out <dir> [--steps 0 143000]
        [--scored <unit 4 dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .a1a import STEPS, V1

CORPUS_SHA = "264c875d8bbd355d8daa9d032b75fd8fb91606218bb84dd1155b203fcd5fab92"
N_BLOCKS = 64
BLOCK = 1024
N_TABLE = 2_000_000
OVERLAP_N = 13          # placed: the contamination run length
VOCAB = 50304           # 410m's logits (the tokenizer's 50,277 padded)
EOS = 0                 # <|endoftext|>
LOSS_143000 = (1.5, 4.0)        # placed check: the trained model's loss on the scored set, nats
STEP0_TOL = 0.10                # placed check: E6u at step 0 against KL(q_u ‖ uniform)
CHUNK = 25_000_000              # tokens per bigram count chunk (memory only; counts are exact)
BATCH_DOCS = 1000               # documents per tokenizer call in the full pass


class NgramRefused(RuntimeError):
    pass


# ---------------------------------------------------------------- corpus (pure parts)

def ngrams(ids: Sequence[int], n: int = OVERLAP_N) -> set:
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


_P = np.uint64(1_000_003)


def _window_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """A polynomial hash (mod 2⁶⁴) of every n-token window."""
    from numpy.lib.stride_tricks import sliding_window_view
    a = np.asarray(ids, np.uint64)
    if len(a) < n:
        return np.zeros(0, np.uint64)
    pw = np.cumprod(np.full(n, _P, np.uint64)) // _P              # P^0 … P^(n−1), wrapping
    return (sliding_window_view(a, n) * pw).sum(axis=1, dtype=np.uint64)


def overlap_check(bad: set, n: int = OVERLAP_N):
    """A function equal to ``lambda ids: bool(ngrams(ids, n) & bad)``, fast on long documents:
    a hash hit is confirmed on the tokens themselves."""
    bad_h = np.array([_window_hashes(np.array(b), n)[0] for b in bad], np.uint64)

    def hit(ids: Sequence[int]) -> bool:
        h = _window_hashes(np.asarray(ids), n)
        return any(tuple(ids[i:i + n]) in bad for i in np.flatnonzero(np.isin(h, bad_h)))
    return hit


def pack_blocks(docs: Iterable[Tuple[int, List[int]]], n_blocks: int, block: int, n_table: int
                ) -> dict:
    """Blocks from document starts, then the table set from the documents after them.

    ``docs`` yields (document index, its tokens with the trailing EOS), already filtered."""
    it = iter(docs)
    blocks, block_docs = [], []
    cur, cur_docs = [], []
    while len(blocks) < n_blocks:
        i, d = next(it)
        cur.extend(d); cur_docs.append(i)
        if len(cur) >= block:
            blocks.append(cur[:block]); block_docs.append(cur_docs)
            cur, cur_docs = [], []
    table, table_docs = [], []
    while len(table) < n_table:
        i, d = next(it)
        table.extend(d); table_docs.append(i)
    return {"blocks": np.asarray(blocks, np.int64), "block_docs": block_docs,
            "table": np.asarray(table[:n_table], np.int64), "table_docs": table_docs}


def iter_docs(path: Path) -> Iterator[Tuple[int, str, str]]:
    """(index, text, Pile subset) in file order, streamed through ``zstd -dc``."""
    p = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    try:
        for i, line in enumerate(p.stdout):
            d = json.loads(line)
            yield i, d["text"], d.get("meta", {}).get("pile_set_name", "")
    finally:
        p.stdout.close(); p.kill(); p.wait()


def _check_corpus(path: Path) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != CORPUS_SHA:
        raise NgramRefused(f"{path}: sha256 {h.hexdigest()[:12]}, expected {CORPUS_SHA[:12]}")


def build_corpus(path: Path, tok) -> dict:
    _check_corpus(path)
    from core.config import PROMPTS
    bad = set().union(*(ngrams(tok(PROMPTS[k])["input_ids"]) for k in V1))
    dropped, subset = [], {}

    def kept():
        for i, text, name in iter_docs(path):
            ids = tok(text)["input_ids"] + [EOS]
            if ngrams(ids) & bad:
                dropped.append(i)
                continue
            subset[i] = name
            yield i, ids
    c = pack_blocks(kept(), N_BLOCKS, BLOCK, N_TABLE)
    sb = {d for ds in c["block_docs"] for d in ds}
    if sb & set(c["table_docs"]):
        raise NgramRefused("a document is in both the scored and the table set")
    count = lambda ds: dict(sorted({n: sum(subset[d] == n for d in ds) for n in
                                    {subset[d] for d in ds}}.items()))
    c["manifest"] = {"corpus_sha256": CORPUS_SHA, "dropped_overlap": dropped,
                     "scored_docs": len(sb), "table_docs": len(c["table_docs"]),
                     "scored_subsets": count(sb), "table_subsets": count(c["table_docs"]),
                     "blocks_sha": hashlib.sha256(c["blocks"].tobytes()).hexdigest()[:12],
                     "table_sha": hashlib.sha256(c["table"].tobytes()).hexdigest()[:12]}
    return c


def _sha12(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:12]


def scored_set(scored: Path) -> dict:
    """Unit 4's corpus (blocks, table, the scored documents), its arrays checked against its
    manifest."""
    man = json.loads((scored / "corpus.json").read_text())
    z = np.load(scored / "corpus.npz")
    for key in ("blocks", "table"):
        if _sha12(z[key]) != man[f"{key}_sha"]:
            raise NgramRefused(f"{scored}: {key} is {_sha12(z[key])}, manifest {man[f'{key}_sha']}")
    return {"blocks": z["blocks"], "table": z["table"], "manifest": man,
            "docs": {d for ds in man["block_docs"] for d in ds}}


def build_full(path: Path, tok, scored: Path) -> dict:
    """Table set′ (unit 5, Blocked 74 (a)): every kept document of the split not in unit 4's
    scored set, in file order, each ended with EOS; its first N_TABLE tokens must be unit 4's
    table."""
    u4 = scored_set(scored)
    _check_corpus(path)
    from core.config import PROMPTS
    hit = overlap_check(set().union(*(ngrams(tok(PROMPTS[k])["input_ids"]) for k in V1)))
    dropped, parts, subsets, batch = [], [], {}, []

    def flush():
        for (i, _, name), ids in zip(batch, tok([b[1] for b in batch])["input_ids"]):
            ids = ids + [EOS]
            if hit(ids):
                dropped.append(i)
                continue
            parts.append(np.asarray(ids, np.int32))
            subsets[name] = subsets.get(name, 0) + 1
        batch.clear()
    n_seen = 0
    for i, text, name in iter_docs(path):
        n_seen += 1
        if i in u4["docs"]:
            continue
        batch.append((i, text, name))
        if len(batch) == BATCH_DOCS:
            flush()
    flush()
    table = np.concatenate(parts)
    first = _sha12(table[:N_TABLE].astype(np.int64))
    if first != u4["manifest"]["table_sha"]:
        raise NgramRefused(f"table set′'s first {N_TABLE} tokens are {first}, unit 4's table is "
                           f"{u4['manifest']['table_sha']}")
    return {"blocks": u4["blocks"], "table": table,
            "manifest": {"corpus_sha256": CORPUS_SHA, "scored_from": str(scored),
                         "docs_in_split": n_seen, "scored_docs_excluded": len(u4["docs"]),
                         "dropped_overlap": dropped, "table_docs": len(parts),
                         "table_tokens": int(len(table)), "table_subsets": dict(sorted(
                             subsets.items())), "first_n_table_sha": first,
                         "blocks_sha": _sha12(u4["blocks"]), "table_sha": _sha12(table)}}


# ---------------------------------------------------------------- predictors (pure)

def tables(table: np.ndarray, V: int = VOCAB, chunk: int = CHUNK) -> dict:
    """q_u, the bigram counts (CSR, V × V), c(v) and T(v); counted ``chunk`` tokens at a time
    (the chunks overlap by one token, so every adjacent pair is counted once)."""
    from scipy.sparse import coo_matrix, csr_matrix
    if table.max() >= V:
        raise NgramRefused(f"a token id {table.max()} is outside the {V} logits")
    uni = np.bincount(table, minlength=V).astype(np.float64)
    B = csr_matrix((V, V), dtype=np.float64)
    for a in range(0, len(table) - 1, chunk):
        s = table[a:a + chunk + 1]
        B = B + coo_matrix((np.ones(len(s) - 1), (s[:-1], s[1:])), shape=(V, V)).tocsr()
    B.sum_duplicates(); B.sort_indices()
    return {"q_u": uni / uni.sum(), "B": B, "c": np.asarray(B.sum(axis=1)).ravel(),
            "T": np.diff(B.indptr).astype(np.float64)}


def q_b_row(t: dict, v: int) -> np.ndarray:
    """q_b(·|v), dense (for checks and the entropy term)."""
    c, T = t["c"][v], t["T"][v]
    if c == 0:
        return t["q_u"].copy()
    row = np.zeros(len(t["q_u"]))
    s = slice(t["B"].indptr[v], t["B"].indptr[v + 1])
    row[t["B"].indices[s]] = t["B"].data[s]
    return (row + T * t["q_u"]) / (c + T)


def neg_entropy(q: np.ndarray) -> float:
    q = q[q > 0]
    return float(np.sum(q * np.log(q)))


def neg_entropy_b(t: dict, contexts: Iterable[int]) -> dict:
    return {int(v): neg_entropy(q_b_row(t, int(v))) for v in contexts}


def cross_terms(logp, ctx: np.ndarray, t: dict, q_u_dev):
    """Σ q log p per position for q_u and q_b(·|ctx[i]); ``logp`` (n, V) float64 torch."""
    import torch
    cu = logp @ q_u_dev                                             # (n,)
    sub = t["B"][ctx]                                               # (n, V) counts, CSR
    rows = np.repeat(np.arange(len(ctx)), np.diff(sub.indptr))
    dev = logp.device
    r = torch.from_numpy(rows).to(dev)
    vals = logp[r, torch.from_numpy(sub.indices.astype(np.int64)).to(dev)]
    vals = vals * torch.from_numpy(sub.data.astype(np.float64)).to(dev)
    sparse = torch.zeros(len(ctx), dtype=logp.dtype, device=dev).index_add_(0, r, vals)
    c = torch.from_numpy(t["c"][ctx]).to(dev)
    T = torch.from_numpy(t["T"][ctx]).to(dev)
    cb = torch.where(c > 0, (sparse + T * cu) / torch.clamp(c + T, min=1), cu)
    return cu, cb


# ---------------------------------------------------------------- run

def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def prepare(corpus: Path, out: Path, tok, scored: Optional[Path] = None) -> dict:
    """The corpus and the step-independent terms, cached in ``out``. With ``scored`` (unit 4's
    dir), the table is table set′ and the blocks are unit 4's."""
    f = out / "corpus.npz"
    if not f.exists():
        if scored is None:
            c = build_corpus(corpus, tok)
            extra = {"block_docs": c["block_docs"]}
        else:
            c = build_full(corpus, tok, scored)
            extra = {"block_docs": scored_set(scored)["manifest"]["block_docs"]}
        np.savez(f, blocks=c["blocks"], table=c["table"])
        (out / "corpus.json").write_text(json.dumps(
            {**c["manifest"], **extra, "code": code_sha()}) + "\n")
    z = np.load(f)
    blocks, table = z["blocks"], z["table"]
    if blocks.shape != (N_BLOCKS, BLOCK) or not (len(table) == N_TABLE if scored is None
                                                 else len(table) > N_TABLE):
        raise NgramRefused(f"corpus shapes {blocks.shape}, {len(table)}")
    t = tables(table)
    for v in (EOS, int(blocks[0, 0]), int(np.argmax(t["c"]))):     # a few rows sum to 1
        if abs(q_b_row(t, v).sum() - 1) > 1e-9:
            raise NgramRefused(f"q_b(·|{v}) does not sum to 1")
    t["negH_u"] = neg_entropy(t["q_u"])
    t["negH_b"] = neg_entropy_b(t, np.unique(blocks))
    return {"blocks": blocks, "t": t}


def run(corpus: Path, out: Path, steps, device: str, scored: Optional[Path] = None) -> None:
    import torch
    from transformers import AutoTokenizer
    from tools.run.behavioural import TOKENIZER_ID
    from trackB_circuit.b0_gate import load
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True, exist_ok=True)
    code = code_sha()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    t0 = time.time()
    P = prepare(corpus, out, tok, scored)
    blocks, t = P["blocks"], P["t"]
    print(f"corpus and tables ready ({time.time() - t0:.0f} s)", flush=True)
    if not (out / "predictor_losses.json").exists():
        (out / "predictor_losses.json").write_text(json.dumps(
            {**predictor_losses(out, t), "code": code}) + "\n")
    q_u = torch.from_numpy(t["q_u"]).to(device)
    negH_b = np.array([[t["negH_b"][int(v)] for v in b] for b in blocks])     # (N_BLOCKS, BLOCK)
    for s in steps:
        f = out / f"step{s}.json"
        if f.exists():
            if json.loads(f.read_text()).get("code") != code:
                raise NgramRefused(f"refusing to resume: {f} was written by other code")
            continue
        t1 = time.time()
        m, _ = load(s, device, torch.float32)
        rec = {"step": s, "code": code, "device": torch.cuda.get_device_name(0)
               if device.startswith("cuda") else "cpu", "kl_u": [], "kl_b": [], "loss": []}
        with torch.no_grad():
            for k, b in enumerate(blocks):
                ids = torch.from_numpy(b).to(device)[None]
                logits = m(ids).logits[0]
                if logits.shape[-1] != VOCAB:
                    raise NgramRefused(f"step {s}: {logits.shape[-1]} logits")
                logp = torch.log_softmax(logits.double(), dim=-1)
                cu, cb = cross_terms(logp, b, t, q_u)
                kl_u = t["negH_u"] - cu.cpu().numpy()
                kl_b = negH_b[k] - cb.cpu().numpy()
                nll = -logp[torch.arange(BLOCK - 1, device=device), ids[0, 1:]].mean()
                rec["kl_u"].append(float(kl_u.mean())); rec["kl_b"].append(float(kl_b.mean()))
                rec["loss"].append(float(nll))
        del m
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        f.write_text(json.dumps(rec) + "\n")
        print(f"step {s}: KL_u {np.mean(rec['kl_u']):.4f} KL_b {np.mean(rec['kl_b']):.4f} "
              f"loss {np.mean(rec['loss']):.4f} ({time.time() - t1:.0f} s)", flush=True)
    print("done", flush=True)


def predictor_losses(out: Path, t: Optional[dict] = None) -> dict:
    """Beside: the two predictors' next-token loss on the scored set (positions where q > 0),
    and how many next tokens they give probability 0."""
    z = np.load(out / "corpus.npz")
    t = tables(z["table"]) if t is None else t
    res = {}
    for name in ("unigram", "bigram"):
        nll, zeros = [], 0
        for b in z["blocks"]:
            for v, w in zip(b[:-1], b[1:]):
                q = t["q_u"][w] if name == "unigram" else q_b_row_at(t, int(v), int(w))
                if q > 0:
                    nll.append(-np.log(q))
                else:
                    zeros += 1
        res[name] = {"loss": float(np.mean(nll)), "zero_prob_tokens": zeros}
    return res


def q_b_row_at(t: dict, v: int, w: int) -> float:
    c, T = t["c"][v], t["T"][v]
    if c == 0:
        return float(t["q_u"][w])
    return float((t["B"][v, w] + T * t["q_u"][w]) / (c + T))


def load_steps(out: Path, steps=STEPS) -> dict:
    recs = {}
    for s in steps:
        f = out / f"step{s}.json"
        if not f.exists():
            raise NgramRefused(f"n-gram rows: no record {f}")
        recs[s] = json.loads(f.read_text())
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--corpus", type=Path, required=True)
    r.add_argument("--steps", nargs="*", type=int)
    r.add_argument("--device", default="cuda")
    r.add_argument("--scored", type=Path, help="unit 4's dir: count table set′ (unit 5)")
    a = ap.parse_args(argv)
    steps = tuple(a.steps) if a.steps else STEPS
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"refusing: steps {bad} are not on the 18-step axis")
    run(a.corpus, a.out, steps, a.device, a.scored)


if __name__ == "__main__":
    main()
