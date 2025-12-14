"""I/O utilities for saving and loading artifacts."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary as JSON file.

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        filepath: Input file path

    Returns:
        Dictionary with loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_text_lines(lines: List[str], filepath: str) -> None:
    """
    Save list of strings as text file (one per line).

    Args:
        lines: List of strings
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def load_text_lines(filepath: str) -> List[str]:
    """
    Load text file as list of lines.

    Args:
        filepath: Input file path

    Returns:
        List of lines (stripped)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def get_file_size_mb(filepath: str) -> float:
    """
    Get file size in MB.

    Args:
        filepath: File path

    Returns:
        Size in MB
    """
    return os.path.getsize(filepath) / (1024 * 1024)
