"""
low_rank_ae.py — Track 3: low-rank autoencoder (no sparsity).

The crosscoder imposes sparsity. Phase 3 showed its decoder directions
are random w.r.t. V. Track 3 asks: was sparsity the wrong prior? A
low-rank bottleneck autoencoder with rank matching the cluster count
should recover V-aligned directions if the dissociation was caused by
the sparsity constraint.

Architecture:
  input (L*d) → encoder (L*d → r) → decoder (r → L*d)

where r is the bottleneck rank, set to match the number of HDBSCAN
clusters at each plateau layer. No activation function in the bottleneck
— this is a linear autoencoder, which at convergence recovers the top-r
PCA directions of the data. The point is to compare those directions
against V's eigensubspaces and against the crosscoder's decoder directions.

If the low-rank AE directions align with V but the crosscoder's don't,
sparsity was the problem. If neither aligns, the dissociation is deeper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LowRankAEConfig:
    d_input: int          # L * d_model
    rank: int             # bottleneck dimension (= cluster count)
    n_layers: int         # number of sampled layers (for reshape)
    d_model: int          # model hidden dim

    def to_dict(self) -> dict:
        return {
            "d_input": self.d_input,
            "rank": self.rank,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LowRankAEConfig":
        return cls(**d)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LowRankAE(nn.Module):
    """
    Linear autoencoder with rank-r bottleneck.

    At convergence this recovers the top-r principal components of the
    input distribution (up to rotation in the bottleneck). The encoder
    and decoder weight matrices span the same r-dimensional subspace
    as the top-r eigenvectors of the data covariance.

    We train it anyway (rather than just doing PCA) because:
    1. Streaming — PCA on 1M tokens × 20k dims doesn't fit in memory
    2. Per-layer reconstruction loss lets us weight layers differently
    3. The trained weights are directly comparable to the crosscoder's
    """

    def __init__(self, cfg: LowRankAEConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.d_input
        r = cfg.rank

        self.encoder = nn.Linear(d_in, r, bias=True)
        self.decoder = nn.Linear(r, d_in, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.encoder.weight)
        nn.init.orthogonal_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x_flat: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x_flat : (batch, L*d)

        Returns
        -------
        dict with x_hat, z (bottleneck activations), loss
        """
        z = self.encoder(x_flat)         # (B, r)
        x_hat = self.decoder(z)          # (B, L*d)
        loss = F.mse_loss(x_hat, x_flat)
        return {"x_hat": x_hat, "z": z, "loss": loss}

    def bottleneck_directions(self) -> np.ndarray:
        """
        Extract the r directions spanning the bottleneck subspace.

        The decoder weight matrix (d_in, r) columns span this subspace.
        We orthonormalize for clean projection tests.

        Returns
        -------
        directions : (r, d_input) orthonormal rows
        """
        W = self.decoder.weight.detach().cpu().numpy()  # (d_in, r)
        Q, _ = np.linalg.qr(W)  # (d_in, r)
        return Q.T  # (r, d_in)

    def per_layer_directions(self) -> np.ndarray:
        """
        Reshape bottleneck directions into per-layer components.

        Returns
        -------
        directions : (r, n_layers, d_model)
        """
        dirs = self.bottleneck_directions()  # (r, L*d)
        r = dirs.shape[0]
        L = self.cfg.n_layers
        d = self.cfg.d_model
        return dirs.reshape(r, L, d)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class LRAETrainingConfig:
    lr: float = 1e-3
    total_steps: int = 20000
    batch_size: int = 512
    log_interval: int = 1000
    checkpoint_dir: Optional[str] = None
    warmup_steps: int = 500


def train_low_rank_ae(
    cfg: LowRankAEConfig,
    train_cfg: LRAETrainingConfig,
    data_iter,
    device: str = "cpu",
) -> LowRankAE:
    """
    Train a low-rank AE on streaming activation data.

    Parameters
    ----------
    cfg : LowRankAEConfig
    train_cfg : LRAETrainingConfig
    data_iter : iterable yielding (batch, L*d) tensors
    device : torch device string

    Returns
    -------
    Trained LowRankAE
    """
    model = LowRankAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    # Linear warmup
    def lr_schedule(step):
        if step < train_cfg.warmup_steps:
            return step / max(train_cfg.warmup_steps, 1)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    model.train()
    running_loss = 0.0
    step = 0

    for batch in data_iter:
        if step >= train_cfg.total_steps:
            break

        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch).float()
        batch = batch.to(device)

        # Flatten if (B, L, d)
        if batch.dim() == 3:
            batch = batch.reshape(batch.shape[0], -1)

        out = model(batch)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % train_cfg.log_interval == 0:
            avg = running_loss / train_cfg.log_interval
            print(f"    [LRAE] step {step}/{train_cfg.total_steps}  "
                  f"loss={avg:.6f}")
            running_loss = 0.0

    model.eval()

    if train_cfg.checkpoint_dir:
        ckpt_path = Path(train_cfg.checkpoint_dir) / "low_rank_ae"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path / "model.pt")
        import json
        with open(ckpt_path / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        print(f"    [LRAE] Saved to {ckpt_path}")

    return model


def load_low_rank_ae(path: Path, device: str = "cpu") -> LowRankAE:
    """Load a trained low-rank AE from checkpoint."""
    import json
    with open(path / "config.json") as f:
        cfg = LowRankAEConfig.from_dict(json.load(f))
    model = LowRankAE(cfg)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Analysis: V-alignment of bottleneck directions
# ---------------------------------------------------------------------------

def bottleneck_v_alignment(
    model: LowRankAE,
    v_projectors: dict,
) -> dict:
    """
    Project the low-rank AE's bottleneck directions onto V's
    eigensubspaces. Compare to the crosscoder's decoder direction
    alignment (which was null in Phase 3).

    Parameters
    ----------
    model : trained LowRankAE
    v_projectors : dict with 'repulsive'/'attractive' projectors

    Returns
    -------
    dict with per-direction alignment and summary
    """
    dirs = model.bottleneck_directions()  # (r, L*d)
    r = dirs.shape[0]

    P_rep = v_projectors.get("repulsive")
    P_att = v_projectors.get("attractive")
    if P_rep is None or P_att is None:
        return {"error": "Missing V projectors"}

    # Projectors are in per-layer d_model space. We need to handle
    # the fact that bottleneck directions span L*d space.
    # Project per-layer components separately.
    per_layer_dirs = model.per_layer_directions()  # (r, L, d)
    L = per_layer_dirs.shape[1]

    per_direction = []
    for i in range(r):
        rep_total = 0.0
        att_total = 0.0
        for l in range(L):
            d_l = per_layer_dirs[i, l]
            norm = np.linalg.norm(d_l)
            if norm < 1e-10:
                continue
            d_l = d_l / norm
            # Handle per-layer or global projectors
            if isinstance(P_rep, dict):
                Pr = P_rep.get(l, P_rep.get(str(l)))
                Pa = P_att.get(l, P_att.get(str(l)))
            else:
                Pr, Pa = P_rep, P_att

            if Pr is None or Pa is None:
                continue

            rep_total += float(d_l @ Pr @ d_l)
            att_total += float(d_l @ Pa @ d_l)

        rep_avg = rep_total / max(L, 1)
        att_avg = att_total / max(L, 1)

        per_direction.append({
            "direction_idx": i,
            "repulsive": rep_avg,
            "attractive": att_avg,
            "dominance": "repulsive" if rep_avg > att_avg else "attractive",
        })

    rep_vals = [d["repulsive"] for d in per_direction]
    att_vals = [d["attractive"] for d in per_direction]

    return {
        "per_direction": per_direction,
        "mean_repulsive": float(np.mean(rep_vals)) if rep_vals else 0.0,
        "mean_attractive": float(np.mean(att_vals)) if att_vals else 0.0,
        "n_repulsive_dominant": sum(
            1 for d in per_direction if d["dominance"] == "repulsive"
        ),
        "n_attractive_dominant": sum(
            1 for d in per_direction if d["dominance"] == "attractive"
        ),
    }


# ---------------------------------------------------------------------------
# Analysis: reconstruction quality vs crosscoder
# ---------------------------------------------------------------------------

def compare_reconstruction(
    lrae: LowRankAE,
    crosscoder,
    prompt_store,
    device: str = "cpu",
) -> dict:
    """
    Compare reconstruction loss between the low-rank AE and the
    crosscoder on the same eval prompts.

    Parameters
    ----------
    lrae : trained LowRankAE
    crosscoder : trained Crosscoder from Phase 3
    prompt_store : PromptActivationStore

    Returns
    -------
    dict with per-prompt MSE for both models
    """
    results = {}
    for pk in prompt_store.keys():
        x = prompt_store.get_stacked_tensor(pk)  # (T, L, d)
        T, L, d = x.shape
        x_flat = x.reshape(T, -1)

        # Low-rank AE
        with torch.no_grad():
            lrae_out = lrae(x_flat.to(device))
        lrae_mse = float(F.mse_loss(
            lrae_out["x_hat"].cpu(), x_flat
        ))

        # Crosscoder
        with torch.no_grad():
            cc_out = crosscoder(x.to(device))
        cc_hat = cc_out["x_hat"].cpu()  # (T, L, d)
        cc_mse = float(F.mse_loss(cc_hat.reshape(T, -1), x_flat))

        results[pk] = {
            "lrae_mse": lrae_mse,
            "crosscoder_mse": cc_mse,
            "ratio": lrae_mse / max(cc_mse, 1e-10),
        }

    return results


# ---------------------------------------------------------------------------
# Streaming data adapter
# ---------------------------------------------------------------------------

class ActivationBufferAdapter:
    """
    Wraps Phase 3's ActivationBuffer to yield flat (B, L*d) batches
    for low-rank AE training.
    """

    def __init__(self, buffer, batch_size: int = 512):
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for item in self.buffer:
            # item is (L, d) or (L*d,)
            if isinstance(item, torch.Tensor):
                item = item.numpy()
            if item.ndim == 2:
                item = item.reshape(-1)
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield np.stack(batch, axis=0)
                batch = []
        if batch:
            yield np.stack(batch, axis=0)
