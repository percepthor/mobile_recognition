"""Versioning and system metadata utilities."""
import sys
import platform
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import tensorflow as tf
import numpy as np


def get_python_version() -> str:
    """Get Python version string."""
    return sys.version.split()[0]


def get_tensorflow_version() -> str:
    """Get TensorFlow version string."""
    return tf.__version__


def get_numpy_version() -> str:
    """Get NumPy version string."""
    return np.__version__


def get_platform_info() -> Dict[str, str]:
    """
    Get platform information.

    Returns:
        Dictionary with platform details
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }


def get_pip_freeze() -> Optional[str]:
    """
    Get pip freeze output as string.

    Returns:
        String with pip packages or None if error
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash if in a git repository.

    Returns:
        Commit hash or None
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def create_run_metadata(
    training_time_seconds: Optional[float] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive run metadata.

    Args:
        training_time_seconds: Total training time in seconds
        additional_info: Additional metadata to include

    Returns:
        Dictionary with run metadata
    """
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": get_python_version(),
        "tensorflow_version": get_tensorflow_version(),
        "numpy_version": get_numpy_version(),
        "platform": get_platform_info(),
    }

    # Add git commit if available
    git_commit = get_git_commit()
    if git_commit:
        metadata["git_commit"] = git_commit

    # Add pip freeze
    pip_freeze = get_pip_freeze()
    if pip_freeze:
        metadata["pip_freeze_hash"] = hash(pip_freeze)

    # Add training time
    if training_time_seconds is not None:
        metadata["training_time_seconds"] = round(training_time_seconds, 2)
        metadata["training_time_minutes"] = round(training_time_seconds / 60, 2)

    # Add any additional info
    if additional_info:
        metadata.update(additional_info)

    return metadata
