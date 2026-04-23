"""
constants.py — Shape constants shared by conftest.py and every test module.

Keeping them here avoids importing from conftest.py directly, which can
trigger double-load warnings in some pytest plugin configurations.
"""

N_LAYERS = 6    # number of synthetic layers
N_TOKENS = 40   # tokens per layer
D        = 16   # embedding dimension
