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

    # Apply backbone
    x = backbone(x, training=False)  # Initially frozen

    # Head
    x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = keras.layers.Dropout(0.2, name='dropout')(x)
    outputs = keras.layers.Dense(num_classes, name='logits')(x)  # No softmax

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
    Train teacher model with two-phase approach:
    1. Train only head with frozen backbone (2-3 epochs)
    2. Fine-tune last layers of backbone

    Args:
        model: Teacher model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Total epochs for training
        lr: Learning rate
        class_weights: Optional class weights for imbalanced data
        output_dir: Directory to save model and metrics

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info("Starting teacher training")

    # Phase 1: Train head only (freeze backbone)
    logger.info("Phase 1: Training head with frozen backbone")
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower() or 'resnet' in layer.name.lower():
            layer.trainable = False

    phase1_epochs = min(3, epochs // 3)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        class_weight=class_weights,
        verbose=1
    )

    # Phase 2: Fine-tune last layers
    logger.info("Phase 2: Fine-tuning last layers")

    # Unfreeze last layers
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower() or 'resnet' in layer.name.lower():
            # Unfreeze last 20% of layers
            total_layers = len(layer.layers) if hasattr(layer, 'layers') else 1
            unfreeze_from = int(total_layers * 0.8)
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers[unfreeze_from:]:
                    sublayer.trainable = True
            layer.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr * 0.1),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    phase2_epochs = epochs - phase1_epochs

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        class_weight=class_weights,
        verbose=1
    )

    # Combine histories
    final_train_acc = history2.history['accuracy'][-1]
    final_val_acc = history2.history['val_accuracy'][-1]

    logger.info(f"Teacher training complete - Train acc: {final_train_acc:.4f}, Val acc: {final_val_acc:.4f}")

    # Save model if output_dir provided
    if output_dir:
        model_path = os.path.join(output_dir, 'teacher_saved_model')
        model.save(model_path)
        logger.info(f"Teacher model saved to {model_path}")

        # Save metrics
        metrics = {
            'final_train_accuracy': float(final_train_acc),
            'final_val_accuracy': float(final_val_acc),
            'phase1_epochs': phase1_epochs,
            'phase2_epochs': phase2_epochs,
            'total_epochs': epochs,
            'learning_rate': lr
        }

        from trainer.utils.io import save_json
        save_json(metrics, os.path.join(output_dir, 'teacher_metrics.json'))

        return model, metrics

    return model, {}
