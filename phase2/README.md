# Phase 2 — V Eigenspectrum Characterization

**Status:** not yet started

## Goal
Extract value matrices from each layer, compute eigenspectra,
record sign distribution.  Cross-reference mixed-sign layers with
Phase 1 plateau windows.

## Falsification criterion
If mixed-sign V layers don't correlate with plateau locations,
the V eigenspectrum is not the driver of metastability.

## Planned modules
- `v_spectrum.py`  — extraction and decomposition
- `analysis.py`    — cross-reference with Phase 1 results
- `plots.py`
- `run.py`
