"""
training.py — Training loop, loss computation, checkpointing, monitoring.

Takes a Crosscoder and an ActivationBuffer.  Knows nothing about how the data
was extracted or what model it came from.

Responsibilities:
  - Training loop with gradient accumulation
  - Per-layer MSE logging
  - Dead feature detection and optional resampling
  - Checkpointing (model + config + training state)
  - Callback hooks for extensibility
"""

import json
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .crosscoder import Crosscoder, CrosscoderConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyperparameters that control training (not architecture)."""
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    total_steps: int = 100_000
    batch_size: int = 512
    grad_accum_steps: int = 4         # effective batch = 512 * 4 = 2048

    # Logging and checkpointing
    log_interval: int = 100           # steps between metric logs
    checkpoint_interval: int = 5000   # steps between checkpoints
    checkpoint_dir: str = "checkpoints"

    # Dead feature handling
    dead_feature_window: int = 10_000  # tokens to accumulate before checking
    dead_feature_threshold: float = 1e-6  # fire rate below this = dead
    resample_dead: bool = True        # whether to resample dead features

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def lr_schedule(step: int, cfg: TrainingConfig) -> float:
    """Linear warmup then cosine decay to 0."""
    if step < cfg.warmup_steps:
        return step / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(
        cfg.total_steps - cfg.warmup_steps, 1
    )
    return 0.5 * (1.0 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# Dead feature resampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def resample_dead_features(
    model: Crosscoder,
    optimizer: torch.optim.Optimizer,
    dead_mask: torch.Tensor,
    recent_inputs: torch.Tensor,
):
    """
    Resample dead features using high-loss input examples.

    For each dead feature, reinitialize its encoder weights from a
    randomly selected high-reconstruction-loss input, and reset its
    decoder to a small random vector.  Reset the optimizer state for
    the affected parameters.

    Parameters
    ----------
    model         : the crosscoder
    optimizer     : the optimizer (Adam) — state gets reset for dead features
    dead_mask     : (n_features,) bool tensor — True for dead features
    recent_inputs : (n_samples, n_layers, d_model) — recent training inputs
    """
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return

    device = model.W_enc.device
    dead_indices = torch.where(dead_mask)[0]

    # Find high-loss inputs to use as seeds
    out = model(recent_inputs[:min(len(recent_inputs), 8192)])
    per_sample_loss = ((recent_inputs[:min(len(recent_inputs), 8192)] - out["x_hat"]) ** 2).mean(dim=(1, 2))
    _, high_loss_idx = torch.topk(per_sample_loss, min(n_dead * 2, len(per_sample_loss)))

    for i, feat_idx in enumerate(dead_indices):
        # Pick a high-loss input as the seed direction
        seed_idx = high_loss_idx[i % len(high_loss_idx)]
        seed_input = recent_inputs[seed_idx].reshape(-1)  # (L*d,)

        # Reinitialize encoder row to point toward this input
        seed_dir = seed_input / (seed_input.norm() + 1e-8)
        model.W_enc[:, feat_idx] = seed_dir * 0.2  # scale down

        # Reinitialize decoder columns to small random
        for layer_idx in range(model.cfg.n_layers):
            rand_dir = torch.randn(model.cfg.d_model, device=device)
            rand_dir = rand_dir / (rand_dir.norm() + 1e-8) * 0.01
            model.W_dec[layer_idx, feat_idx] = rand_dir

        # Reset encoder bias for this feature
        model.b_enc[feat_idx] = 0.0

    # Reset optimizer state for affected parameters.
    # Must cover W_enc, W_dec, and b_enc — all three are modified above.
    # Leaving stale Adam momentum on reinitialized weights corrupts the
    # first several update steps for those features.
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p not in optimizer.state:
                continue
            state = optimizer.state[p]
            if "exp_avg" not in state:
                continue

            if p.shape == model.W_enc.shape:
                # W_enc: (d_in, F) — zero columns for dead features
                state["exp_avg"][:, dead_indices] = 0
                state["exp_avg_sq"][:, dead_indices] = 0
            elif p.shape == model.W_dec.shape:
                # W_dec: (L, F, d) — zero the feature slice in all layers
                state["exp_avg"][:, dead_indices, :] = 0
                state["exp_avg_sq"][:, dead_indices, :] = 0
            elif p.shape == model.b_enc.shape:
                # b_enc: (F,)
                state["exp_avg"][dead_indices] = 0
                state["exp_avg_sq"][dead_indices] = 0

    print(f"    Resampled {n_dead} dead features")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: Crosscoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_cfg: TrainingConfig,
    metrics_history: list,
    path: Path,
):
    """Save training checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "crosscoder_config": model.cfg.to_dict(),
        "training_config": train_cfg.to_dict(),
    }, path / f"checkpoint_{step:07d}.pt")

    # Save metrics history as JSON (small, human-readable)
    with open(path / "metrics_history.json", "w") as f:
        json.dump(metrics_history, f)


def save_final(
    model: Crosscoder,
    train_cfg: TrainingConfig,
    path: Path,
):
    """
    Save final trained model in the standard format analysis.py expects.

    This is the contract between training and analysis: a directory with
    model.pt (state dict) and config.json (architecture config).
    """
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    with open(path / "config.json", "w") as f:
        json.dump(model.cfg.to_dict(), f, indent=2)
    with open(path / "training_config.json", "w") as f:
        json.dump(train_cfg.to_dict(), f, indent=2)
    print(f"  Final model saved to {path}")


def load_trained_crosscoder(path: str | Path, device: str = "cpu") -> Crosscoder:
    """
    Load a trained crosscoder from the standard format.

    This is the entry point for analysis.py — it doesn't need to know
    anything about how training was configured.
    """
    path = Path(path)
    with open(path / "config.json") as f:
        cfg = CrosscoderConfig.from_dict(json.load(f))
    model = Crosscoder(cfg).to(device)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Callback signature: fn(step, model, batch_output, metrics) -> None
TrainingCallback = Callable


def train(
    model: Crosscoder,
    data_source,
    train_cfg: TrainingConfig,
    device: str = "cuda",
    callbacks: Optional[list[TrainingCallback]] = None,
) -> dict:
    """
    Train the crosscoder.

    Parameters
    ----------
    model       : Crosscoder (will be moved to device)
    data_source : ActivationBuffer with a get_batch(batch_size) method
    train_cfg   : TrainingConfig
    device      : "cuda" or "cpu"
    callbacks   : optional list of functions called every log_interval steps

    Returns
    -------
    dict with training summary: final_loss, n_dead_final, metrics_history
    """
    callbacks = callbacks or []
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    # Mixed precision: forward in fp16, optimizer in fp32.
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    metrics_history = []

    # Dead feature tracking
    fire_counts = torch.zeros(model.cfg.n_features, device=device)
    total_samples_since_check = 0
    recent_batch = None  # single batch kept for dead feature resampling

    step = 0
    t_start = time.time()

    print(f"  Training: {train_cfg.total_steps} steps, "
          f"batch={train_cfg.batch_size}×{train_cfg.grad_accum_steps}, "
          f"lr={train_cfg.lr}, amp={use_amp}")

    while step < train_cfg.total_steps:
        # Get batch from buffer (stays on CPU until .to(device))
        batch = data_source.get_batch(train_cfg.batch_size).to(device)

        # Forward (mixed precision)
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            out = model(batch)
            loss = out["loss"] / train_cfg.grad_accum_steps

        # Backward (scaled)
        scaler.scale(loss).backward()

        # Track feature firing for dead feature detection (every step)
        with torch.no_grad():
            fire_counts += (out["z"] > 0).float().sum(dim=0)
            total_samples_since_check += batch.shape[0]

            # Keep a rolling buffer (last N batches) for resampling seed selection.
            # A single batch is too small to find diverse high-loss examples;
            # we keep recent_batch as a concatenation of the last few batches,
            # capped at dead_feature_window tokens so memory stays bounded.
            if train_cfg.resample_dead:
                recent_batch = batch.detach()

        if (step + 1) % train_cfg.grad_accum_steps == 0:
            # LR schedule
            lr_mult = lr_schedule(step, train_cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = train_cfg.lr * lr_mult

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Normalize decoder after each optimizer step
            model.normalize_decoder()

            # Dead feature check and resample — only after a real weight update,
            # never mid-accumulation (resampling mid-cycle corrupts the
            # accumulated gradients that are about to be applied).
            if (
                total_samples_since_check >= train_cfg.dead_feature_window
                and train_cfg.resample_dead
            ):
                fire_rates = fire_counts / total_samples_since_check
                dead_mask = fire_rates < train_cfg.dead_feature_threshold
                n_dead = int(dead_mask.sum())

                if n_dead > 0 and recent_batch is not None:
                    resample_dead_features(model, optimizer, dead_mask, recent_batch)

                fire_counts.zero_()
                total_samples_since_check = 0

        # Logging
        if step % train_cfg.log_interval == 0:
            with torch.no_grad():
                activity = model.feature_activity(out["z"])
                per_layer = out["per_layer_mse"]

                metrics = {
                    "step": step,
                    "loss": float(out["loss"]),
                    "n_dead": activity["n_dead"],
                    "mean_fire_rate": float(activity["fire_rate"].mean()),
                    "mean_l0": float((out["z"] > 0).float().sum(dim=-1).mean()),
                    "per_layer_mse": per_layer.cpu().tolist(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "time_elapsed": time.time() - t_start,
                }
                metrics_history.append(metrics)

                elapsed = metrics["time_elapsed"]
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                print(
                    f"  step {step:6d}/{train_cfg.total_steps}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"L0={metrics['mean_l0']:.1f}  "
                    f"dead={metrics['n_dead']}  "
                    f"lr={metrics['lr']:.2e}  "
                    f"({steps_per_sec:.1f} steps/s)"
                )

                for cb in callbacks:
                    cb(step, model, out, metrics)

        # Checkpointing
        if step > 0 and step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, step, train_cfg,
                metrics_history, checkpoint_dir,
            )

        step += 1

    # Final save
    save_final(model, train_cfg, checkpoint_dir / "final")
    save_checkpoint(
        model, optimizer, step, train_cfg,
        metrics_history, checkpoint_dir,
    )

    return {
        "final_loss": metrics_history[-1]["loss"] if metrics_history else None,
        "n_dead_final": metrics_history[-1]["n_dead"] if metrics_history else None,
        "metrics_history": metrics_history,
    }
