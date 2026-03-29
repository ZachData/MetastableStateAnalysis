# Finding: Late-layer oscillation in large GPT-2 models

**Observed:** GPT-2-xl (and larger models) oscillate around a
metastable state density in late layers rather than monotonically
collapsing as the paper predicts.

**Source run:** see metastability_results/2026-03-15_18-55-33

## Questions
- Is the oscillation period regular?
- Does it scale with model depth?
- Is energy non-monotone at the same layers?
- Does the Fiedler value oscillate in phase?

## Planned analysis
Load saved run artifacts (no model needed) and characterize the
oscillation: period, amplitude, which metrics co-oscillate, and
whether it's present in gpt2-large but absent in gpt2-medium.
