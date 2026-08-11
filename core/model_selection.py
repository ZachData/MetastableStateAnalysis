"""
core/model_selection.py — resolving --models arguments, once.

`run_1.py::resolve_model_names` and `print_model_catalogue` were written to
fix a real problem: MODEL_CONFIGS grew from 10 entries to 47 when the Pythia
checkpoint registry was merged in, so `--models` defaulting to
`list(MODEL_CONFIGS.keys())` turned a bare invocation into tens of gigabytes
of downloads across 37 checkpoints. run_1 pinned the original meaning with
DEFAULT_MODELS and made the Pythia schedules opt-in by group name.

Phase 1b never got that fix. `run_1b.py` still carried
`default=list(MODEL_CONFIGS.keys())` with `choices=` over all 47 keys — the
exact two things run_1 deliberately removed, including the `choices=` list
that made a typo render as an unreadable error.

Copying run_1's resolver into run_1b would have been the third copy of a
selection rule that has already drifted once. It lives here instead, and
run_1.py should delegate to it (its own versions are unchanged for now, and
are byte-equivalent in behaviour — the delegation is a separate, mechanical
edit that should not ride along with a Phase 1b change).

Deliberately importable without torch at the function level: the module-level
`core.config` import already pulls torch in via DEVICE, so this does not make
things worse, but nothing here adds a new heavy edge.
"""

from __future__ import annotations

import sys


def resolve_model_names(requested, model_configs, model_groups) -> list:
    """
    Expand group names and validate registry keys, preserving order.

    A group name and a registry key are accepted in the same position, so
    `--models replication-gate gpt2-xl` works. Unknown names are fatal rather
    than skipped: a typo that silently produces an empty selection looks
    exactly like a run that legitimately had nothing to do.
    """
    resolved, unknown = [], []
    for name in requested:
        if name in model_groups:
            for m in model_groups[name]:
                if m not in resolved:
                    resolved.append(m)
        elif name in model_configs:
            if name not in resolved:
                resolved.append(name)
        else:
            unknown.append(name)

    if unknown:
        sys.exit(
            f"Unknown model/group: {', '.join(unknown)}\n"
            f"Groups:   {', '.join(sorted(model_groups))}\n"
            f"Run --list-models for the full registry."
        )
    return resolved


def print_model_catalogue(model_configs, model_groups, default_models) -> None:
    print("\nGroups (usable directly as --models arguments):")
    for group, members in sorted(model_groups.items()):
        print(f"  {group:<24} {len(members):>3} models")
        for m in members:
            print(f"      {m}")
    print(f"\nDefault when --models is omitted: {', '.join(default_models)}")
    print(f"\nAll registry keys ({len(model_configs)}):")
    for key in sorted(model_configs):
        print(f"  {key}")
