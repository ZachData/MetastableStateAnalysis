"""
tools/token_counts.py — token count per prompt, per registered model.

How many tokens each entry of core.config.PROMPTS becomes under each
model's own tokenizer. Worth having because n (the particle count) is the
independent variable behind most of this project's geometry, and it is a
property of the tokenizer, not of the prompt text: the same string is a
different n on GPT-2 BPE and on the NeoX tokenizer. See
core/battery_structure.py for the stronger version of this check — whether
the battery still has the *structure* it was designed to have after
tokenization, not just how long it is.

Promoted from a scratch `test.py` at the repo root (which pytest could
collect by name) when Phases 3-6 were archived. Needs transformers, and
downloads each registered tokenizer.

Usage:  python -m tools.token_counts
"""

from core.config import PROMPTS, MODEL_CONFIGS


def main() -> None:
    from transformers import AutoTokenizer

    print(f"{'prompt':<25} {'model':<30} {'tokens':>6}")
    print("-" * 65)

    for model_name, cfg in MODEL_CONFIGS.items():
        tok = AutoTokenizer.from_pretrained(model_name)
        for key, text in PROMPTS.items():
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            print(f"{key:<25} {model_name:<30} {n:>6}")
        print()


if __name__ == "__main__":
    main()
