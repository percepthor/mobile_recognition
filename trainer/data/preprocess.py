"""
Preprocessing functions - EXACT letterbox implementation.

This module is the SOURCE OF TRUTH for preprocessing.
The letterbox implementation must match exactly what will be used in C inference.
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Union


def letterbox_resize_pad(
    image: Union[tf.Tensor, np.ndarray],
    target: int = 240
) -> tf.Tensor:
    """
    Letterbox resize with padding (EXACT algorithm for reproducibility).

    This function implements the EXACT letterbox algorithm specified in requirements:
    1. Calculate scale = target / max(H, W)
    2. Resize to (round(H*scale), round(W*scale)) using BILINEAR
    3. Pad with black (0,0,0) to center the image in target x target canvas

    Args:
        image: Input tensor [H, W, 3] - can be uint8 or float32
        target: Target size (default 240)

    Returns:
        Tensor [target, target, 3] with same dtype as input
    """
    # Ensure tensor
    if isinstance(image, np.ndarray):
        image = tf.constant(image)

    original_dtype = image.dtype

    # Get original dimensions
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    # Step 1: Calculate scale
    max_dim = tf.maximum(h, w)
    scale = tf.cast(target, tf.float32) / max_dim

    # Step 2: Calculate new dimensions (rounded)
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)

    # Step 3: Resize using BILINEAR interpolation
    resized = tf.image.resize(
        image,
        [new_h, new_w],
        method=tf.image.ResizeMethod.BILINEAR
    )

    # Step 4-7: Calculate padding (centered)
    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left

    # Step 8: Apply padding with black (0,0,0)
    paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    padded = tf.pad(resized, paddings, mode='CONSTANT', constant_values=0)

    # Ensure exact shape
    padded = tf.ensure_shape(padded, [target, target, 3])

    # Convert back to original dtype if needed
    if original_dtype == tf.uint8:
        padded = tf.cast(tf.round(padded), tf.uint8)

    return padded


def decode_and_preprocess_image(
    image_bytes: bytes,
    target_size: int = 240,
    return_range_0_255: bool = True
) -> tf.Tensor:
    """
    Decode image bytes and apply preprocessing.

    This handles:
    - Decoding to RGB (3 channels)
    - Converting grayscale to RGB if needed
    - Discarding alpha channel if present
    - Applying letterbox resize
    - Returning in [0, 255] range as float32

    Args:
        image_bytes: Raw image bytes
        target_size: Target size for letterbox (default 240)
        return_range_0_255: If True, return float32 in [0,255], else uint8

    Returns:
        Preprocessed tensor [target_size, target_size, 3]
    """
    # Decode image
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)

    # Ensure RGB (decode_image with channels=3 should handle this, but let's be explicit)
    if image.shape[-1] != 3:
        # This shouldn't happen with channels=3, but safety check
        if image.shape[-1] == 1:
            image = tf.tile(image, [1, 1, 3])
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

    # Apply letterbox
    image = letterbox_resize_pad(image, target_size)

    # Convert to desired range
    if return_range_0_255:
        if image.dtype == tf.uint8:
            image = tf.cast(image, tf.float32)
        # Already in [0, 255] range
    else:
        if image.dtype != tf.uint8:
            image = tf.cast(tf.round(image), tf.uint8)

    return image


def preprocess_image_bytes_to_letterbox_rgb(
    image_bytes: bytes,
    target: int = 240
) -> np.ndarray:
    """
    Public API for preprocessing (for contract testing).

    This function provides a simple interface that can be compared
    against C implementation.

    Args:
        image_bytes: Raw image bytes
        target: Target size (default 240)

    Returns:
        NumPy array [target, target, 3] uint8 in RGB format
    """
    tensor = decode_and_preprocess_image(
        image_bytes,
        target_size=target,
        return_range_0_255=False  # Return uint8
    )
    return tensor.numpy()


def preprocess_from_path(
    image_path: str,
    target_size: int = 240,
    return_range_0_255: bool = True
) -> tf.Tensor:
    """
    Load and preprocess image from file path.

    Args:
        image_path: Path to image file
        target_size: Target size for letterbox
        return_range_0_255: If True, return float32 in [0,255]

    Returns:
        Preprocessed tensor
    """
    image_bytes = tf.io.read_file(image_path)
    return decode_and_preprocess_image(
        image_bytes,
        target_size=target_size,
        return_range_0_255=return_range_0_255
    )


def get_preprocessing_metadata(target_size: int = 240) -> dict:
    """
    Get preprocessing metadata for model_metadata.json.

    Args:
        target_size: Target image size

    Returns:
        Dictionary with preprocessing configuration
    """
    return {
        "letterbox": True,
        "keep_aspect_ratio": True,
        "pad_color_rgb": [0, 0, 0],
        "interpolation": "bilinear",
        "resize_rule": "scale = target/max(h,w); new=round(h*scale,w*scale); pad centered",
        "target_size": target_size,
        "output_range": [0, 255],
        "output_dtype": "float32",
        "color_space": "RGB",
        "channels": 3
    }
