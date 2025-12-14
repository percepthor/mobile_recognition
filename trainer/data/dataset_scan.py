"""Dataset scanning and class mapping."""
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def scan_dataset(
    data_dir: str,
    output_dir: str
) -> Tuple[List[str], Dict[str, int], List[str]]:
    """
    Scan dataset directory and extract class information.

    Args:
        data_dir: Root directory containing class subdirectories
        output_dir: Directory to save manifest and bad files list

    Returns:
        Tuple of (class_names, class_counts, bad_files)
        - class_names: List of class names in alphabetical order
        - class_counts: Dictionary mapping class name to image count
        - bad_files: List of corrupted/invalid image paths
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Get all subdirectories (classes) - only first level
    class_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if not class_dirs:
        raise ValueError(f"No class directories found in {data_dir}")

    # Extract class names in alphabetical order
    class_names = [d.name for d in class_dirs]
    logger.info(f"Found {len(class_names)} classes: {class_names}")

    # Scan images in each class
    class_counts = {}
    all_images = []
    bad_files = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = []

        # Find all supported images
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(class_dir.glob(f"*{ext}"))
            image_files.extend(class_dir.glob(f"*{ext.upper()}"))

        # Validate each image
        valid_count = 0
        for img_path in image_files:
            if _is_valid_image(img_path):
                all_images.append(str(img_path))
                valid_count += 1
            else:
                bad_files.append(str(img_path))
                logger.warning(f"Corrupted/invalid image: {img_path}")

        class_counts[class_name] = valid_count
        logger.info(f"Class '{class_name}': {valid_count} valid images")

    # Save manifest
    manifest = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_counts": class_counts,
        "total_images": sum(class_counts.values()),
        "bad_files_count": len(bad_files)
    }

    from trainer.utils.io import save_json, save_text_lines, ensure_dir
    ensure_dir(output_dir)

    save_json(manifest, os.path.join(output_dir, "dataset_manifest.json"))
    save_text_lines(class_names, os.path.join(output_dir, "labels.txt"))

    if bad_files:
        save_text_lines(bad_files, os.path.join(output_dir, "bad_files.txt"))
        logger.warning(f"Found {len(bad_files)} corrupted files - saved to bad_files.txt")

    logger.info(f"Total valid images: {sum(class_counts.values())}")

    return class_names, class_counts, bad_files


def _is_valid_image(image_path: Path) -> bool:
    """
    Check if image file is valid and can be opened.

    Args:
        image_path: Path to image file

    Returns:
        True if valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        # Open again to actually load it (verify() doesn't load the image)
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception as e:
        logger.debug(f"Invalid image {image_path}: {e}")
        return False


def get_class_to_index(class_names: List[str]) -> Dict[str, int]:
    """
    Create mapping from class name to index.

    Args:
        class_names: List of class names (ordered)

    Returns:
        Dictionary mapping class name to index
    """
    return {name: idx for idx, name in enumerate(class_names)}
