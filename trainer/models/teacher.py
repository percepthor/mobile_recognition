"""Teacher model implementation (EfficientNet-B3 or ResNet101)."""
import logging
import os
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras


logger = logging.getLogger(__name__)


def create_teacher_model(
    num_classes: int,
    architecture: str = 'efficientnet_b3',
    input_size: int = 300,
    pretrained: bool = True
) -> keras.Model:
    """
    Create teacher model.

    Args:
        num_classes: Number of output classes
        architecture: 'efficientnet_b3' or 'resnet101'
        input_size: Input image size
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        Keras model (returns logits)
    """
    if architecture not in ['efficientnet_b3', 'resnet101']:
        raise ValueError(f"Unsupported architecture: {architecture}")

    logger.info(f"Creating teacher model: {architecture} with input size {input_size}")

    # Input layer - expects float32 in [0, 255]
    inputs = keras.Input(shape=(input_size, input_size, 3), dtype=tf.float32)

    # Normalize to [0, 1] for pretrained backbones
    x = inputs / 255.0

    # Create backbone
    weights = 'imagenet' if pretrained else None

    if architecture == 'efficientnet_b3':
        backbone = keras.applications.EfficientNetB3(
            include_top=False,
            weights=weights,
            input_shape=(input_size, input_size, 3),
            pooling=None
        )
    elif architecture == 'resnet101':
        backbone = keras.applications.ResNet101(
            include_top=False,
            weights=weights,
            input_shape=(input_size, input_size, 3),
            pooling=None
        )

    # Apply backbone - let training mode be determined dynamically
    x = backbone(x)

    # Head with stronger regularization
    x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = keras.layers.Dropout(0.5, name='dropout')(x)  # Increased dropout to prevent overfitting
    outputs = keras.layers.Dense(
        num_classes,
        name='logits',
        kernel_regularizer=keras.regularizers.l2(0.01)  # L2 regularization
    )(x)  # No softmax

    model = keras.Model(inputs=inputs, outputs=outputs, name=f'teacher_{architecture}')

    logger.info(f"Teacher model created with {model.count_params():,} parameters")

    return model


def train_teacher(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 10,
    lr: float = 1e-3,
    class_weights: Optional[dict] = None,
    output_dir: str = None
) -> Tuple[keras.Model, dict]:
    """
    Train teacher model with frozen backbone to prevent overfitting.
    Uses early stopping to avoid memorization.

    Args:
        model: Teacher model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Maximum epochs for training
        lr: Learning rate
        class_weights: Optional class weights for imbalanced data
        output_dir: Directory to save model and metrics

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info("Starting teacher training")

    # KEEP BACKBONE FROZEN to prevent overfitting on small datasets
    logger.info("Training with frozen backbone (no fine-tuning)")
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower() or 'resnet' in layer.name.lower():
            layer.trainable = False

    # Early stopping callback - crucial to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [early_stopping, reduce_lr]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Get best metrics (from restored weights)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    stopped_epoch = len(history.history['accuracy'])

    logger.info(f"Teacher training complete - Train acc: {final_train_acc:.4f}, Val acc: {final_val_acc:.4f}")
    logger.info(f"Best val acc: {best_val_acc:.4f}, Stopped at epoch: {stopped_epoch}")

    # Save model if output_dir provided
    if output_dir:
        model_path = os.path.join(output_dir, 'teacher_saved_model')
        model.save(model_path)
        logger.info(f"Teacher model saved to {model_path}")

        # Save metrics
        metrics = {
            'final_train_accuracy': float(final_train_acc),
            'final_val_accuracy': float(final_val_acc),
            'best_val_accuracy': float(best_val_acc),
            'stopped_epoch': stopped_epoch,
            'max_epochs': epochs,
            'learning_rate': lr,
            'backbone_frozen': True
        }

        from trainer.utils.io import save_json
        save_json(metrics, os.path.join(output_dir, 'teacher_metrics.json'))

        return model, metrics

    return model, {}
