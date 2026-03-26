# Phase 3 — Crosscoder Training

**Status:** not yet started

## Goal
Train a crosscoder spanning all layers on the residual stream.
Features stratified by activation lifetime correspond to
metastable configurations.

## Falsification criterion
If crosscoder features don't stratify by layer lifetime,
representations aren't organized along dynamical structure.

## Planned modules
- `crosscoder.py`  — architecture and training loop
- `dataset.py`     — activation dataset builder from saved .npz files
- `analysis.py`
- `run.py`
