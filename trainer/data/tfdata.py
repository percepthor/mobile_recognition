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
    Apply data augmentation.

    Note: Augmentation is applied AFTER letterbox preprocessing.
    Keep augmentations that don't break the letterbox format.

    Args:
        image: Preprocessed image tensor [H, W, 3] in [0, 255]

    Returns:
        Augmented image
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random brightness (keep in valid range)
    image = tf.image.random_brightness(image, max_delta=25)
    image = tf.clip_by_value(image, 0.0, 255.0)

    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 255.0)

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
