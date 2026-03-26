# Phase 4 — Metastable Feature Identification

**Status:** not yet started

## Goal
Per feature: compute activation trajectory variance across layers.
Low variance over a window then a spike = metastable feature candidate.
Find coordinated reorganization events (merge events).

## Falsification criterion
If feature plateaus don't align with Phase 1 cluster count plateaus,
features aren't tracking metastable configurations.
