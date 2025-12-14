"""Seed management for reproducibility."""
import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # For better reproducibility with GPU operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def get_seed_info(seed: int) -> dict:
    """
    Get seed information for metadata.

    Args:
        seed: The seed value used

    Returns:
        Dictionary with seed information
    """
    return {
        "seed": seed,
        "deterministic_ops_enabled": True,
        "libraries_seeded": ["python", "numpy", "tensorflow"]
    }
