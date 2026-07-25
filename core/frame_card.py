"""
core/frame_card.py — The model-level frame card (frames item 5a).

Why this module exists
----------------------
Phase 6 cannot currently build an LN frame. It reconstructs its context
from Phase 1 activations and the Phase 2 NPZ, and Phase 2 persists OV/QK
matrices only — no LayerNorm gamma/beta, no rotary geometry, no revision
string. The same gap blocks the beta_eff fix in Phase 5 and the
per-checkpoint frame requirement in the sweep.

The obvious patch — put the LN parameters in the Phase 2 NPZ — is wrong.
LN parameters, rotary geometry, vocabulary width, and extraction
conventions are facts about the *model*, not about a weight
decomposition. Filing them under Phase 2 gives every frame-aware phase a
dependency on Phase 2 having been run, including Phase 1, which has none
today and which is where the position-0 and layer-0 questions live.

So: one artifact per (model, revision), written once by whatever loads the
model, read by every phase.

    frame_card.json      metadata (below)
    frame_card_ln.npz    ln_gamma_attn / ln_beta_attn / ln_gamma_mlp /
                         ln_beta_mlp, each (n_blocks, d_model), plus
                         final_ln_gamma / final_ln_beta

The quiet win is `embedding_stripped` and `last_is_post_final_ln`. Those
extraction conventions are today carried in prose across the status docs
and re-derived at call sites; a mismatch between the extraction path used
and the convention assumed is unfalsifiable from the artifacts alone. Here
it is one field, checked.

Single-home constraint
----------------------
The hidden-state off-by-one lives in
core.ln_frame.resolve_frame_index and is delegated to by both
frame_for_hidden_state (model present) and FrameCard.frame_spec_for
(artifacts only). This module must never re-derive it.

See DESIGN_pythia_frames.md, item 5a.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.frames import FrameSpec, UNKNOWN_REV
from core.ln_frame import (
    DEFAULT_LN_EPS,
    get_final_ln_params,
    get_ln_params,
    n_blocks as _n_blocks,
    resolve_frame_index,
)
from core.rope import DEFAULT_ROPE_BASE, rope_config_from_model, model_uses_rope


CARD_VERSION = "v1"
CARD_JSON = "frame_card.json"
CARD_NPZ = "frame_card_ln.npz"

_LN_KEYS = (
    "ln_gamma_attn", "ln_beta_attn",
    "ln_gamma_mlp", "ln_beta_mlp",
    "final_ln_gamma", "final_ln_beta",
)


class FrameCardError(ValueError):
    """Raised when a card is missing, malformed, or does not match its consumer."""


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameCard:
    """
    Everything needed to reconstruct the frame a number lives in, without
    loading the model.

    `revision` is required in spirit: it defaults to UNKNOWN_REV so a card
    can be built from a model that does not expose one, but
    core.frames.verify_same_revision refuses to verify two unknowns, so an
    unset revision fails at comparison time rather than passing silently.
    """

    # identity
    model_name: str
    revision: str = UNKNOWN_REV
    card_version: str = CARD_VERSION

    # shape
    d_model: int = 0
    n_heads: int = 0
    head_size: int = 0
    n_blocks: int = 0

    # attention geometry
    rotary_ndims: int = 0
    rotary_pct: float = 0.0
    rope_base: float = DEFAULT_ROPE_BASE
    attn_scale: float = 1.0
    parallel_residual: bool = False

    # normalisation
    ln_eps: float = DEFAULT_LN_EPS

    # vocabulary — the padded/real split is why the KL arbiter needs this
    vocab_size_padded: int = 0
    vocab_size_real: int = 0

    # tokenizer behaviour
    tokenizer_name: str = ""
    prepends_bos: bool = False

    # extraction conventions (see module docstring)
    embedding_stripped: bool = True
    last_is_post_final_ln: bool = False

    # ---------------------------------------------------------------- derived

    @property
    def uses_rope(self) -> bool:
        return self.rotary_ndims > 0

    @property
    def rev_key(self) -> str:
        """Stable revision key. Distinguishes deduped from non-deduped runs."""
        return f"{self.model_name}@{self.revision}"

    def vocab_mask(self) -> np.ndarray:
        """
        Boolean (vocab_size_padded,) keep-mask over logit rows.

        Pythia's embed_out is padded to 50304 while the tokenizer has ~50277
        real tokens; the remainder are untrained and can emit arbitrary
        logits. Every softmax consumer must mask first — most of all
        functional_distance.kl_matrix, which is the arbiter when two frames
        disagree.
        """
        if self.vocab_size_padded <= 0:
            raise FrameCardError("vocab_mask: vocab_size_padded not recorded on card")
        real = self.vocab_size_real or self.vocab_size_padded
        if real > self.vocab_size_padded:
            raise FrameCardError(
                f"vocab_mask: vocab_size_real {real} exceeds padded "
                f"{self.vocab_size_padded}"
            )
        keep = np.zeros(self.vocab_size_padded, dtype=bool)
        keep[:real] = True
        return keep

    @property
    def n_padding_rows(self) -> int:
        return max(0, self.vocab_size_padded - self.vocab_size_real)

    # ------------------------------------------------------------ frame spec

    def frame_spec_for(
        self,
        hidden_layer_idx: int,
        n_hidden_states: int,
        which: str = "attn",
        rope_applied: bool | None = None,
        pos0_policy: str = "included",
    ) -> FrameSpec:
        """
        Resolve the FrameSpec for a hidden-state index, from artifacts alone.

        Delegates the off-by-one to core.ln_frame.resolve_frame_index — the
        single home — and supplies the conventions from the card rather than
        from a call site's assumption.

        rope_applied defaults to the card's own answer (True iff the model
        uses rotary). Pass it explicitly only to record a deliberate
        omission, e.g. labelling a legacy quantity as a proxy.
        """
        res = resolve_frame_index(
            hidden_layer_idx, n_hidden_states, self.n_blocks,
            embedding_stripped=self.embedding_stripped,
            last_is_post_final_ln=self.last_is_post_final_ln,
        )
        return FrameSpec.from_ln_resolution(
            res,
            layer_idx=hidden_layer_idx,
            model_rev=self.rev_key,
            which=which,
            rope_applied=self.uses_rope if rope_applied is None else rope_applied,
            pos0_policy=pos0_policy,
            ln_eps=self.ln_eps,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "FrameCard":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise FrameCardError(
                f"FrameCard.from_dict: unrecognised fields {sorted(unknown)}. "
                f"A card written by a newer schema must not be read as if it "
                f"were this one."
            )
        return cls(**{k: v for k, v in d.items() if k in known})

    def summary_lines(self) -> list:
        return [
            "Frame card:",
            f"  model         {self.model_name} @ {self.revision}",
            f"  shape         d_model={self.d_model} heads={self.n_heads} "
            f"head_size={self.head_size} blocks={self.n_blocks}",
            f"  rotary        {self.rotary_ndims}/{self.head_size} dims "
            f"({self.rotary_pct:.2f}), base={self.rope_base:g}"
            if self.uses_rope else "  rotary        none",
            f"  attn scale    {self.attn_scale:.6g}",
            f"  residual      {'parallel' if self.parallel_residual else 'sequential'}",
            f"  vocab         {self.vocab_size_real} real / "
            f"{self.vocab_size_padded} padded ({self.n_padding_rows} dead rows)",
            f"  tokenizer     {self.tokenizer_name} "
            f"(BOS {'prepended' if self.prepends_bos else 'NOT prepended'})",
            f"  extraction    embedding_stripped={self.embedding_stripped} "
            f"last_is_post_final_ln={self.last_is_post_final_ln}",
        ]


# ---------------------------------------------------------------------------
# LN parameter store
# ---------------------------------------------------------------------------

class LNStore:
    """
    The gamma/beta arrays that go with a card. Kept separate from the frozen
    dataclass because they are bulk numeric data, not metadata.

    Arrays are (n_blocks, d_model) per (which, param), plus the final LN.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray], card: FrameCard | None = None):
        missing = [k for k in _LN_KEYS if k not in arrays]
        if missing:
            raise FrameCardError(f"LNStore: missing arrays {missing}")
        self.arrays = {k: np.asarray(arrays[k], dtype=np.float64) for k in _LN_KEYS}
        self.card = card
        if card is not None:
            self._validate_against(card)

    def _validate_against(self, card: FrameCard) -> None:
        for k in ("ln_gamma_attn", "ln_beta_attn", "ln_gamma_mlp", "ln_beta_mlp"):
            want = (card.n_blocks, card.d_model)
            if self.arrays[k].shape != want:
                raise FrameCardError(
                    f"LNStore: {k} has shape {self.arrays[k].shape}, "
                    f"card says {want}"
                )
        for k in ("final_ln_gamma", "final_ln_beta"):
            if self.arrays[k].shape != (card.d_model,):
                raise FrameCardError(
                    f"LNStore: {k} has shape {self.arrays[k].shape}, "
                    f"card says ({card.d_model},)"
                )

    def params_for(self, spec: FrameSpec) -> dict:
        """
        The {"gamma","beta","eps"} dict apply_frame needs, for a FrameSpec.

        Non-LN frames return None: a caller that passes an l2_sphere spec
        here is asking for something that does not exist, and should get
        None rather than a plausible default.
        """
        if not spec.is_ln():
            return None
        which = "attn" if spec.kind == "ln_attn" else "mlp"
        eps = spec.ln_eps
        if dict(spec.extras).get("ln_source") == "final_layer_norm":
            return {
                "gamma": self.arrays["final_ln_gamma"],
                "beta": self.arrays["final_ln_beta"],
                "eps": eps,
            }
        b = spec.reader_block
        n = self.arrays[f"ln_gamma_{which}"].shape[0]
        if b is None or not (0 <= b < n):
            raise FrameCardError(
                f"LNStore.params_for: reader_block {b} out of range [0, {n})"
            )
        return {
            "gamma": self.arrays[f"ln_gamma_{which}"][b],
            "beta": self.arrays[f"ln_beta_{which}"][b],
            "eps": eps,
        }


# ---------------------------------------------------------------------------
# Extraction (duck-typed; torch never imported)
# ---------------------------------------------------------------------------

def build_frame_card(
    model,
    model_name: str,
    revision: str = UNKNOWN_REV,
    tokenizer=None,
    embedding_stripped: bool = True,
    last_is_post_final_ln: bool = False,
) -> tuple:
    """
    Build (FrameCard, LNStore) from a live GPT-NeoX/Pythia model.

    Duck-typed on the module structure the way core/pythia_weights.py and
    core/ln_frame.py are, so this runs against SimpleNamespace fakes in the
    stubbed test session.

    The extraction-convention arguments are NOT guessed. They describe the
    path the caller used to record hidden states, and the caller is the only
    party that knows; a default that silently differed from the actual path
    is precisely the failure this artifact exists to close.
    """
    cfg = getattr(model, "config", model)
    nb = _n_blocks(model)

    rope_uses = model_uses_rope(model)
    if rope_uses:
        rc = rope_config_from_model(model)
    else:
        d_model = int(getattr(cfg, "hidden_size"))
        n_heads = int(getattr(cfg, "num_attention_heads"))
        head_size = d_model // n_heads
        rc = dict(d_model=d_model, n_heads=n_heads, head_size=head_size,
                  rotary_ndims=0, rotary_pct=0.0, base=DEFAULT_ROPE_BASE,
                  scale=1.0 / np.sqrt(head_size))

    ln_eps = float(getattr(cfg, "layer_norm_eps", DEFAULT_LN_EPS))

    gam_a, bet_a, gam_m, bet_m = [], [], [], []
    for b in range(nb):
        pa = get_ln_params(model, b, which="attn")
        pm = get_ln_params(model, b, which="mlp")
        gam_a.append(np.asarray(pa["gamma"], dtype=np.float64))
        bet_a.append(np.asarray(pa["beta"], dtype=np.float64))
        gam_m.append(np.asarray(pm["gamma"], dtype=np.float64))
        bet_m.append(np.asarray(pm["beta"], dtype=np.float64))
    pf = get_final_ln_params(model)

    vocab_padded = int(getattr(cfg, "vocab_size", 0))
    vocab_real = int(len(tokenizer)) if tokenizer is not None else vocab_padded
    tok_name = getattr(tokenizer, "name_or_path", "") if tokenizer is not None else ""
    prepends_bos = _detect_prepends_bos(tokenizer)

    card = FrameCard(
        model_name=model_name,
        revision=revision,
        d_model=int(rc["d_model"]),
        n_heads=int(rc["n_heads"]),
        head_size=int(rc["head_size"]),
        n_blocks=int(nb),
        rotary_ndims=int(rc["rotary_ndims"]),
        rotary_pct=float(rc.get("rotary_pct", 0.0)),
        rope_base=float(rc["base"]),
        attn_scale=float(rc["scale"]),
        parallel_residual=bool(getattr(cfg, "use_parallel_residual", False)),
        ln_eps=ln_eps,
        vocab_size_padded=vocab_padded,
        vocab_size_real=vocab_real,
        tokenizer_name=str(tok_name),
        prepends_bos=prepends_bos,
        embedding_stripped=embedding_stripped,
        last_is_post_final_ln=last_is_post_final_ln,
    )
    store = LNStore(
        {
            "ln_gamma_attn": np.stack(gam_a) if gam_a else np.zeros((0, card.d_model)),
            "ln_beta_attn": np.stack(bet_a) if bet_a else np.zeros((0, card.d_model)),
            "ln_gamma_mlp": np.stack(gam_m) if gam_m else np.zeros((0, card.d_model)),
            "ln_beta_mlp": np.stack(bet_m) if bet_m else np.zeros((0, card.d_model)),
            "final_ln_gamma": np.asarray(pf["gamma"], dtype=np.float64),
            "final_ln_beta": np.asarray(pf["beta"], dtype=np.float64),
        },
        card=card,
    )
    return card, store


def _detect_prepends_bos(tokenizer) -> bool:
    """
    Whether the tokenizer puts a BOS in front. NeoX tokenizers do not, which
    is why position 0 becomes the attention sink on Pythia (policy P1).

    Detected empirically where possible rather than assumed from the class
    name; falls back to False, which is the Pythia-correct answer and the
    one that keeps the sink question visible.
    """
    if tokenizer is None:
        return False
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return False
    try:
        ids = tokenizer("a")["input_ids"]
        return bool(len(ids) > 0 and ids[0] == bos)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_frame_card(out_dir, card: FrameCard, store: LNStore) -> dict:
    """Write frame_card.json + frame_card_ln.npz. Returns the written paths."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    jp, np_ = d / CARD_JSON, d / CARD_NPZ
    with open(jp, "w") as f:
        json.dump(card.to_dict(), f, indent=2, sort_keys=True)
    np.savez_compressed(np_, **store.arrays)
    return {"json": jp, "npz": np_}


def load_frame_card(in_dir) -> tuple:
    """
    Read (FrameCard, LNStore) back. Raises FrameCardError when absent —
    a phase that needs a frame and finds no card must stop, not guess.
    """
    d = Path(in_dir)
    jp, np_ = d / CARD_JSON, d / CARD_NPZ
    if not jp.exists():
        raise FrameCardError(
            f"load_frame_card: no {CARD_JSON} in {d}. The frame card is written "
            f"at extraction time; a run without one cannot have its frames "
            f"verified."
        )
    with open(jp) as f:
        card = FrameCard.from_dict(json.load(f))
    if not np_.exists():
        raise FrameCardError(f"load_frame_card: {CARD_JSON} present but {CARD_NPZ} missing in {d}")
    with np.load(np_) as z:
        store = LNStore({k: z[k] for k in _LN_KEYS}, card=card)
    return card, store


def verify_card_for_run(card: FrameCard, model_name: str, revision: str | None = None,
                        context: str = "") -> None:
    """
    Assert a loaded card belongs to the run consuming it.

    The failure this catches is a cached card from the final model being
    applied to a step-1000 checkpoint — LN gamma/beta from the wrong point in
    training, silently. That is item 11's whole content.
    """
    where = f" [{context}]" if context else ""
    if card.model_name != model_name:
        raise FrameCardError(
            f"Frame card is for {card.model_name!r}, run is {model_name!r}{where}"
        )
    if revision is not None and card.revision != revision:
        raise FrameCardError(
            f"Frame card revision {card.revision!r} != run revision "
            f"{revision!r}{where}. LN parameters must come from the "
            f"checkpoint being analysed, never from the final model."
        )
