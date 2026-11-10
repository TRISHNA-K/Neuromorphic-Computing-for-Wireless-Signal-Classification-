"""Helper utilities for seed, serialization, and device configuration."""

import os
import json
import random
import torch
import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: dict, filepath: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_device() -> torch.device:
    """Select best available device (cuda or cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
