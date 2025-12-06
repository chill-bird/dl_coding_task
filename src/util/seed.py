"""
seed.py
---

Provides utility functions for setting seeds.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to use (default: 42)
    """

    print(f"Setting seed {seed}...\n")

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
