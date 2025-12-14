"""Dataset splitting with stratification."""
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def create_stratified_split(
    data_dir: str,
    class_names: List[str],
    output_dir: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test split.

    Args:
        data_dir: Root directory with class subdirectories
        class_names: List of class names (ordered)
        output_dir: Directory to save split CSV files
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Validate fractions
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    data_path = Path(data_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Collect all valid images
    all_data = []
    supported_exts = {'.jpg', '.jpeg', '.png'}

    for class_name in class_names:
        class_dir = data_path / class_name
        class_idx = class_to_idx[class_name]

        # Find all images
        image_files = []
        for ext in supported_exts:
            image_files.extend(class_dir.glob(f"*{ext}"))
            image_files.extend(class_dir.glob(f"*{ext.upper()}"))

        for img_path in image_files:
            # Store relative path to data_dir for portability
            rel_path = img_path.relative_to(data_path)
            all_data.append({
                'filepath': str(rel_path),
                'label_index': class_idx,
                'label_name': class_name
            })

    if not all_data:
        raise ValueError("No valid images found for splitting")

    df = pd.DataFrame(all_data)
    logger.info(f"Total images to split: {len(df)}")

    # Check if we have enough samples per class for stratification
    class_counts = df['label_name'].value_counts()
    min_samples = class_counts.min()

    if min_samples < 3:
        logger.warning(
            f"Some classes have very few samples (min={min_samples}). "
            "Stratification may not be perfect."
        )

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df['label_name'],
        random_state=seed
    )

    # Second split: val vs test
    # Calculate relative proportion of val in the remaining data
    val_relative = val_frac / (val_frac + test_frac)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        stratify=temp_df['label_name'],
        random_state=seed
    )

    # Log split statistics
    logger.info(f"Train set: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Val set: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")

    # Log per-class distribution
    logger.info("\nPer-class distribution:")
    for class_name in class_names:
        train_count = (train_df['label_name'] == class_name).sum()
        val_count = (val_df['label_name'] == class_name).sum()
        test_count = (test_df['label_name'] == class_name).sum()
        total = train_count + val_count + test_count
        logger.info(
            f"  {class_name}: train={train_count}, val={val_count}, "
            f"test={test_count}, total={total}"
        )

    # Save to CSV
    from trainer.utils.io import ensure_dir
    ensure_dir(output_dir)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    logger.info("Split CSVs saved successfully")

    return train_df, val_df, test_df


def load_split_from_csv(
    output_dir: str,
    split: str = 'train'
) -> pd.DataFrame:
    """
    Load split DataFrame from CSV.

    Args:
        output_dir: Directory containing split CSV files
        split: One of 'train', 'val', 'test'

    Returns:
        DataFrame with split data
    """
    csv_path = os.path.join(output_dir, f'{split}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    return pd.read_csv(csv_path)
