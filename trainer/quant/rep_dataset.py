"""Representative dataset generator for TFLite quantization."""
import logging
from typing import Iterator
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


def create_representative_dataset_generator(
    train_ds: tf.data.Dataset,
    num_batches: int = 50,
    force_input_range_0_255: bool = True
) -> Iterator:
    """
    Create representative dataset generator for TFLite quantization.

    The representative dataset is used by the TFLite converter to calibrate
    quantization parameters.

    Args:
        train_ds: Training dataset (preprocessed)
        num_batches: Number of batches to use from training set
        force_input_range_0_255: If True, add synthetic examples at 0 and 255
                                 to ensure full range quantization

    Returns:
        Generator function that yields representative samples
    """
    logger.info(f"Creating representative dataset with {num_batches} batches")

    # Collect samples from training dataset
    samples = []

    for batch_idx, (images, _) in enumerate(train_ds):
        if batch_idx >= num_batches:
            break

        # Add each image in the batch
        for img in images:
            samples.append(img.numpy())

    logger.info(f"Collected {len(samples)} samples from training data")

    # Add synthetic examples to force full [0, 255] range if requested
    if force_input_range_0_255 and len(samples) > 0:
        logger.info("Adding synthetic examples to force [0, 255] input range")

        # Get shape from first sample
        sample_shape = samples[0].shape

        # Create all-zero image (black)
        zero_image = np.zeros(sample_shape, dtype=np.float32)
        samples.insert(0, zero_image)

        # Create all-255 image (white)
        max_image = np.full(sample_shape, 255.0, dtype=np.float32)
        samples.insert(1, max_image)

        logger.info("Added 2 synthetic examples (all-0 and all-255)")

    def representative_dataset_gen():
        """Generator function for TFLite converter."""
        for sample in samples:
            # Yield as list with single element (model expects batch dimension will be added)
            yield [sample[np.newaxis, :].astype(np.float32)]

    # Store count for reference
    representative_dataset_gen.num_samples = len(samples)

    logger.info(f"Representative dataset generator ready with {len(samples)} samples")

    return representative_dataset_gen


def create_simple_representative_dataset(
    image_array: np.ndarray,
    num_samples: int = 100
) -> Iterator:
    """
    Create simple representative dataset from a single image array.

    Useful for testing or when full dataset isn't available.

    Args:
        image_array: Single image array [H, W, C]
        num_samples: Number of samples to generate (with slight variations)

    Returns:
        Generator function
    """
    samples = []

    # Add original
    samples.append(image_array.astype(np.float32))

    # Add variations with slight noise
    for i in range(num_samples - 1):
        noisy = image_array.astype(np.float32) + np.random.randn(*image_array.shape) * 5
        noisy = np.clip(noisy, 0, 255)
        samples.append(noisy)

    def representative_dataset_gen():
        for sample in samples:
            yield [sample[np.newaxis, :].astype(np.float32)]

    return representative_dataset_gen
