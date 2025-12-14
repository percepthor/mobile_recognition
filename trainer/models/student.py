"""Student model implementation (EfficientNet-Lite1)."""
import logging
import os
from typing import Optional
import tensorflow as tf
from tensorflow import keras


logger = logging.getLogger(__name__)


def create_student_model(
    num_classes: int,
    input_size: int = 240
) -> keras.Model:
    """
    Create student model using EfficientNet-Lite1.

    The student model is designed for mobile deployment:
    - Input: 240x240x3 float32 in [0, 255]
    - Output: logits (no softmax)
    - Lightweight architecture suitable for TFLite conversion

    Args:
        num_classes: Number of output classes
        input_size: Input image size (default 240)

    Returns:
        Keras model (returns logits)
    """
    logger.info(f"Creating student model: EfficientNet-Lite1 with input size {input_size}")

    # Input layer - expects float32 in [0, 255]
    inputs = keras.Input(shape=(input_size, input_size, 3), dtype=tf.float32, name='input')

    # Normalize to [0, 1] - this normalization will be part of the model
    # so it's included in TFLite export
    x = inputs / 255.0

    # Try to use TensorFlow Hub for EfficientNet-Lite1
    try:
        import tensorflow_hub as hub

        # EfficientNet-Lite1 from TF Hub
        hub_url = "https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2"
        logger.info(f"Loading EfficientNet-Lite1 from TF Hub: {hub_url}")

        hub_layer = hub.KerasLayer(
            hub_url,
            trainable=True,
            name='efficientnet_lite1'
        )
        x = hub_layer(x)

    except Exception as e:
        logger.warning(f"Could not load from TF Hub: {e}")
        logger.info("Using EfficientNetB0 as fallback (similar to Lite1)")

        # Fallback to EfficientNetB0 which is similar to Lite1
        backbone = keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(input_size, input_size, 3),
            pooling='avg'  # Global average pooling
        )
        # Let training mode be determined dynamically
        x = backbone(x)

    # Head with regularization to prevent overfitting
    x = keras.layers.Dropout(0.4, name='dropout')(x)  # Increased dropout
    outputs = keras.layers.Dense(
        num_classes,
        name='logits',
        kernel_regularizer=keras.regularizers.l2(0.01)  # L2 regularization
    )(x)  # No softmax

    model = keras.Model(inputs=inputs, outputs=outputs, name='student_efficientnet_lite1')

    logger.info(f"Student model created with {model.count_params():,} parameters")

    return model


def train_student_standalone(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 20,
    lr: float = 1e-3,
    class_weights: Optional[dict] = None,
    output_dir: Optional[str] = None
) -> tuple:
    """
    Train student model without distillation (baseline).

    This is primarily for testing purposes. In production,
    student should be trained via distillation.

    Args:
        model: Student model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of epochs
        lr: Learning rate
        class_weights: Optional class weights
        output_dir: Directory to save model

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info("Training student model (standalone, no distillation)")

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
        verbose=1
    )

    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    logger.info(f"Student training complete - Train acc: {final_train_acc:.4f}, Val acc: {final_val_acc:.4f}")

    metrics = {
        'final_train_accuracy': float(final_train_acc),
        'final_val_accuracy': float(final_val_acc),
        'epochs': epochs,
        'learning_rate': lr,
        'trained_with_distillation': False
    }

    if output_dir:
        model_path = os.path.join(output_dir, 'student_fp32_saved_model')
        model.save(model_path)
        logger.info(f"Student model saved to {model_path}")

        from trainer.utils.io import save_json
        save_json(metrics, os.path.join(output_dir, 'student_fp32_metrics.json'))

    return model, metrics
