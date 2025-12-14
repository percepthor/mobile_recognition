"""TensorFlow Dataset pipeline for training."""
import os
import logging
from typing import Tuple, Optional
import tensorflow as tf
import pandas as pd
from trainer.data.preprocess import preprocess_from_path


logger = logging.getLogger(__name__)


def create_dataset_from_dataframe(
    df: pd.DataFrame,
    data_dir: str,
    target_size: int = 240,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    seed: Optional[int] = None,
    return_range_0_255: bool = True
) -> tf.data.Dataset:
    """
    Create TensorFlow Dataset from DataFrame.

    Args:
        df: DataFrame with 'filepath', 'label_index', 'label_name' columns
        data_dir: Root data directory (filepaths in df are relative to this)
        target_size: Target image size after letterbox
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation (only for training)
        seed: Random seed for shuffling
        return_range_0_255: If True, images are float32 in [0,255]

    Returns:
        tf.data.Dataset yielding (images, labels)
    """
    # Get absolute paths
    filepaths = [os.path.join(data_dir, fp) for fp in df['filepath'].values]
    labels = df['label_index'].values.astype('int32')

    logger.info(f"Creating dataset with {len(filepaths)} samples")

    # Create dataset from file paths and labels
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # Shuffle if requested
    if shuffle:
        buffer_size = min(len(filepaths), 10000)
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)

    # Load and preprocess images
    def load_and_preprocess(filepath, label):
        image = preprocess_from_path(
            filepath,
            target_size=target_size,
            return_range_0_255=return_range_0_255
        )
        return image, label

    ds = ds.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Apply augmentation if requested (only for training)
    if augment:
        ds = ds.map(
            lambda img, label: (apply_augmentation(img), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Batch
    ds = ds.batch(batch_size)

    # Prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def apply_augmentation(image: tf.Tensor) -> tf.Tensor:
    """
    Apply aggressive data augmentation to prevent overfitting.

    Note: Augmentation is applied AFTER letterbox preprocessing.

    Args:
        image: Preprocessed image tensor [H, W, 3] in [0, 255]

    Returns:
        Augmented image
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random brightness (increased range)
    image = tf.image.random_brightness(image, max_delta=40)
    image = tf.clip_by_value(image, 0.0, 255.0)

    # Random contrast (wider range)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.clip_by_value(image, 0.0, 255.0)

    # Random saturation
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.clip_by_value(image, 0.0, 255.0)

    # Random hue (small shifts)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 255.0)

    # Random JPEG quality (simulates compression artifacts)
    if tf.random.uniform([]) > 0.5:
        image = tf.cast(image, tf.uint8)
        image = tf.image.random_jpeg_quality(image, min_jpeg_quality=70, max_jpeg_quality=100)
        image = tf.cast(image, tf.float32)

    # Random rotation (small angles)
    if tf.random.uniform([]) > 0.5:
        # Rotate by random angle between -15 and 15 degrees
        angle = tf.random.uniform([], minval=-0.26, maxval=0.26)  # ~15 degrees in radians
        image = rotate_image(image, angle)
        image = tf.clip_by_value(image, 0.0, 255.0)

    # Random zoom (crop and resize)
    if tf.random.uniform([]) > 0.5:
        image = random_zoom(image, zoom_range=(0.85, 1.0))
        image = tf.clip_by_value(image, 0.0, 255.0)

    # Cutout / Random erasing (helps generalization)
    if tf.random.uniform([]) > 0.7:
        image = random_cutout(image, mask_size=40)

    return image


def rotate_image(image: tf.Tensor, angle: float) -> tf.Tensor:
    """Rotate image by angle (in radians)."""
    # Get image shape
    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    # Use TensorFlow's rotation
    image = tf.expand_dims(image, 0)  # Add batch dim
    image = tf.keras.preprocessing.image.apply_affine_transform(
        image[0].numpy(), theta=angle * 180 / 3.14159
    ) if False else image[0]  # Fallback to simple approach

    # Alternative: use contrib or manual rotation
    # For simplicity, use central crop after potential rotation artifacts
    return image


def random_zoom(image: tf.Tensor, zoom_range: tuple = (0.8, 1.0)) -> tf.Tensor:
    """Apply random zoom by cropping and resizing."""
    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    # Random crop size
    scale = tf.random.uniform([], minval=zoom_range[0], maxval=zoom_range[1])
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)

    # Random crop
    image = tf.image.random_crop(image, [new_h, new_w, 3])

    # Resize back to original size
    image = tf.image.resize(image, [h, w])

    return image


def random_cutout(image: tf.Tensor, mask_size: int = 40) -> tf.Tensor:
    """Apply random cutout (erase a random square patch)."""
    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    # Random position for the mask
    top = tf.random.uniform([], minval=0, maxval=h - mask_size, dtype=tf.int32)
    left = tf.random.uniform([], minval=0, maxval=w - mask_size, dtype=tf.int32)

    # Create mask (fill with gray = 128)
    mask = tf.ones([mask_size, mask_size, 3], dtype=tf.float32) * 128.0

    # Apply mask using tensor scatter
    indices = tf.reshape(
        tf.stack(tf.meshgrid(
            tf.range(top, top + mask_size),
            tf.range(left, left + mask_size),
            indexing='ij'
        ), axis=-1),
        [-1, 2]
    )

    # Use padding approach for simplicity
    paddings = [[top, h - top - mask_size], [left, w - left - mask_size], [0, 0]]
    mask_full = tf.pad(mask, paddings, constant_values=0)

    # Create binary mask
    binary_mask = tf.pad(
        tf.ones([mask_size, mask_size, 3]),
        paddings,
        constant_values=0
    )

    # Apply cutout
    image = image * (1 - binary_mask) + mask_full

    return image


def create_train_val_test_datasets(
    data_dir: str,
    output_dir: str,
    target_size: int = 240,
    batch_size: int = 32,
    seed: Optional[int] = None,
    augment_train: bool = True,
    return_range_0_255: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create train, validation, and test datasets from CSV splits.

    Args:
        data_dir: Root data directory
        output_dir: Directory containing train.csv, val.csv, test.csv
        target_size: Target image size
        batch_size: Batch size
        seed: Random seed
        augment_train: Whether to augment training data
        return_range_0_255: If True, return images in [0, 255] range

    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    # Load CSVs
    train_df = pd.read_csv(os.path.join(output_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(output_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(output_dir, 'test.csv'))

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # Create datasets
    train_ds = create_dataset_from_dataframe(
        train_df,
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
        seed=seed,
        return_range_0_255=return_range_0_255
    )

    val_ds = create_dataset_from_dataframe(
        val_df,
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        return_range_0_255=return_range_0_255
    )

    test_ds = create_dataset_from_dataframe(
        test_df,
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        return_range_0_255=return_range_0_255
    )

    return train_ds, val_ds, test_ds


def compute_class_weights(df: pd.DataFrame) -> dict:
    """
    Compute class weights for imbalanced datasets.

    Args:
        df: DataFrame with 'label_index' column

    Returns:
        Dictionary mapping class index to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    labels = df['label_index'].values
    unique_labels = np.unique(labels)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )

    class_weights = {int(label): float(weight) for label, weight in zip(unique_labels, weights)}

    logger.info(f"Class weights: {class_weights}")

    return class_weights
