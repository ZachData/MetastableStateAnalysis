"""
data.py — Activation extraction, disk caching, and Dataset.

Responsibilities:
  - Stream text from C4 or TinyStories via HuggingFace datasets
  - Extract residual stream activations from ALBERT-xlarge or GPT-2-large
  - L2-normalize onto the unit sphere
  - Cache to disk as float16 numpy shards
  - Provide a PyTorch Dataset that loads from the cache

This is the only module that knows about model architecture differences
(ALBERT shared-weight iteration vs GPT-2 native layers).

Storage format:
  cache_dir/
    meta.json          — config, layer_indices, d_model, n_tokens_total
    shard_0000.npy     — (n_tokens_in_shard, n_sampled_layers, d_model) float16
    ...
    eval_prompts/      — cached Phase 1 prompts for analysis
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYER_PRESETS = {
    "albert-xlarge-v2": [0, 6, 12, 18, 24, 30, 36, 42, 46, 48],
    "gpt2-large":       [0, 4, 8, 12, 16, 20, 24, 28, 32, 35],
}

# 4x expansion fits on 10GB VRAM with mixed precision.
# Scale to 16x (32768 / 20480) with 24GB+.
FEATURE_PRESETS = {
    "albert-xlarge-v2": 8192,    # 4 * 2048
    "gpt2-large":       5120,    # 4 * 1280
}

SUPPORTED_MODELS = ["albert-xlarge-v2", "gpt2-large"]


@dataclass
class ExtractionConfig:
    model_name: str
    layer_indices: list[int]
    max_seq_len: int = 512
    shard_size: int = 50_000
    cache_dir: str = "activation_cache"
    dtype: str = "float16"

    @property
    def is_albert(self) -> bool:
        return "albert" in self.model_name.lower()

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "layer_indices": self.layer_indices,
            "max_seq_len": self.max_seq_len,
            "shard_size": self.shard_size,
            "dtype": self.dtype,
        }


# ---------------------------------------------------------------------------
# Text data sources
# ---------------------------------------------------------------------------

def stream_c4(n_texts: int, max_length: int = 2000) -> list[str]:
    """Stream from C4 (allenai/c4) via HuggingFace datasets streaming."""
    from datasets import load_dataset

    print(f"  Streaming {n_texts} texts from C4...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    texts = []
    for example in ds:
        text = example["text"].strip()
        if len(text) >= 100:
            texts.append(text[:max_length])
        if len(texts) >= n_texts:
            break
        if len(texts) % 10_000 == 0 and len(texts) > 0:
            print(f"    {len(texts)}/{n_texts}")
    print(f"    Collected {len(texts)} texts")
    return texts


def stream_tinystories(n_texts: int, max_length: int = 2000) -> list[str]:
    """Stream from TinyStories — simple semantics, good sanity check."""
    from datasets import load_dataset

    print(f"  Streaming {n_texts} texts from TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    texts = []
    for example in ds:
        text = example["text"].strip()
        if len(text) >= 50:
            texts.append(text[:max_length])
        if len(texts) >= n_texts:
            break
    print(f"    Collected {len(texts)} texts")
    return texts


def load_texts(
    source: str = "c4",
    n_texts: int = 50_000,
    data_dir: Optional[str] = None,
) -> list[str]:
    """Load texts.  source: "c4", "tinystories", or "local"."""
    if source == "c4":
        return stream_c4(n_texts)
    elif source == "tinystories":
        return stream_tinystories(n_texts)
    elif source == "local":
        if data_dir is None:
            raise ValueError("source='local' requires data_dir")
        return _load_local(data_dir, n_texts)
    else:
        raise ValueError(f"Unknown source: {source}")


def _load_local(data_dir: str, max_texts: int) -> list[str]:
    data_path = Path(data_dir)
    texts = []
    if data_path.is_file():
        with open(data_path) as f:
            content = f.read()
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        texts = chunks[:max_texts]
    elif data_path.is_dir():
        for txt_file in sorted(data_path.glob("*.txt"))[:max_texts]:
            with open(txt_file) as f:
                texts.append(f.read().strip())
    else:
        raise FileNotFoundError(f"Not found: {data_dir}")
    return texts[:max_texts]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def extract_albert(model, tokenizer, texts, layer_indices, max_seq_len=512, device="cuda"):
    max_iter = max(layer_indices)
    layer_set = set(layer_indices)
    results = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)

        with torch.no_grad(), torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            emb = model.embeddings(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs.get("token_type_ids"),
            )
            hidden = model.encoder.embedding_hidden_mapping_in(emb)
            attn_mask = model.get_extended_attention_mask(
                inputs["attention_mask"], inputs["input_ids"].shape
            )
            albert_layer = model.encoder.albert_layer_groups[0].albert_layers[0]

            snaps = {}
            if 0 in layer_set:
                snaps[0] = _l2_normalize(hidden[0].float()).cpu().numpy()

            for step in range(1, max_iter + 1):
                out = albert_layer(hidden, attention_mask=attn_mask)
                hidden = out[0]
                if step in layer_set:
                    snaps[step] = _l2_normalize(hidden[0].float()).cpu().numpy()

        results.append(np.stack([snaps[i] for i in sorted(layer_indices)], axis=1))
    return results


def extract_gpt2(model, tokenizer, texts, layer_indices, max_seq_len=512, device="cuda"):
    results = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)

        with torch.no_grad(), torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            outputs = model(**inputs)

        selected = [
            _l2_normalize(outputs.hidden_states[i][0].float()).cpu().numpy()
            for i in sorted(layer_indices)
        ]
        results.append(np.stack(selected, axis=1))
    return results


def extract_activations(model, tokenizer, texts, config, device="cuda"):
    fn = extract_albert if config.is_albert else extract_gpt2
    return fn(model, tokenizer, texts, config.layer_indices, config.max_seq_len, device)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def cache_activations(model, tokenizer, texts, config, device="cuda", batch_size=8):
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    buffer, buffer_tokens, total_tokens = [], 0, 0

    def _flush():
        nonlocal shard_idx, buffer, buffer_tokens
        if not buffer:
            return
        arr = np.concatenate(buffer, axis=0)
        np.save(cache_dir / f"shard_{shard_idx:04d}.npy", arr.astype(np.float16))
        print(f"  Wrote shard_{shard_idx:04d}.npy: {arr.shape[0]} tokens")
        shard_idx += 1
        buffer.clear()
        buffer_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for arr in extract_activations(model, tokenizer, batch, config, device):
            buffer.append(arr)
            buffer_tokens += arr.shape[0]
            total_tokens += arr.shape[0]
        if buffer_tokens >= config.shard_size:
            _flush()
        if (i // batch_size + 1) % 100 == 0:
            print(f"  {i + len(batch)}/{len(texts)} texts, {total_tokens} tokens")
    _flush()

    d_model = getattr(model.config, "hidden_size", None) or model.config.n_embd
    meta = config.to_dict()
    meta.update({"n_tokens_total": total_tokens, "n_shards": shard_idx, "d_model": d_model})
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Cache: {total_tokens} tokens, {shard_idx} shards → {cache_dir}")
    return cache_dir


def is_cache_valid(cache_dir: str | Path) -> bool:
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("n_tokens_total", 0) > 0 and meta.get("n_shards", 0) > 0
    except Exception:
        return False


def is_trained(checkpoint_dir: str | Path) -> bool:
    p = Path(checkpoint_dir) / "final"
    return (p / "model.pt").exists() and (p / "config.json").exists()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ActivationDataset(Dataset):
    def __init__(self, cache_dir: str | Path):
        cache_dir = Path(cache_dir)
        with open(cache_dir / "meta.json") as f:
            self.meta = json.load(f)

        self.cache_dir = cache_dir
        self.n_tokens = self.meta["n_tokens_total"]
        self.n_layers = len(self.meta["layer_indices"])
        self.d_model = self.meta["d_model"]

        self._shard_paths = sorted(cache_dir.glob("shard_*.npy"))
        self._shard_sizes = []
        self._shard_offsets = []
        offset = 0
        for sp in self._shard_paths:
            shape = self._peek_shape(sp)
            self._shard_sizes.append(shape[0])
            self._shard_offsets.append(offset)
            offset += shape[0]

        self._loaded_shard_idx: Optional[int] = None
        self._loaded_shard_data: Optional[np.ndarray] = None

    @staticmethod
    def _peek_shape(path):
        with open(path, "rb") as f:
            ver = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, ver)
        return shape

    def _load_shard(self, idx):
        if self._loaded_shard_idx != idx:
            self._loaded_shard_data = np.load(self._shard_paths[idx]).astype(np.float32)
            self._loaded_shard_idx = idx

    def _find_shard(self, global_idx):
        lo, hi = 0, len(self._shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._shard_offsets[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, global_idx - self._shard_offsets[lo]

    def __len__(self):
        return self.n_tokens

    def __getitem__(self, idx):
        si, li = self._find_shard(idx)
        self._load_shard(si)
        return torch.from_numpy(self._loaded_shard_data[li])


class PromptActivationStore:
    def __init__(self):
        self.prompts: dict[str, dict] = {}

    def add(self, prompt_key, activations, tokens, layer_indices):
        self.prompts[prompt_key] = {
            "activations": activations, "tokens": tokens,
            "layer_indices": layer_indices,
        }

    def get_stacked_tensor(self, prompt_key):
        return torch.from_numpy(self.prompts[prompt_key]["activations"].astype(np.float32))

    def keys(self):
        return self.prompts.keys()

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {}
        for key, data in self.prompts.items():
            safe = key.replace("/", "_")
            np.save(path / f"{safe}.npy", data["activations"])
            meta[key] = {"tokens": data["tokens"], "layer_indices": data["layer_indices"],
                         "n_tokens": data["activations"].shape[0]}
        with open(path / "prompt_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path / "prompt_meta.json") as f:
            meta = json.load(f)
        store = cls()
        for key, info in meta.items():
            arr = np.load(path / f"{key.replace('/', '_')}.npy")
            store.add(key, arr, info["tokens"], info["layer_indices"])
        return store
