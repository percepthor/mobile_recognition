"""Quantization Aware Training (QAT) implementation."""
import logging
import os
from typing import Optional
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot


logger = logging.getLogger(__name__)


def apply_qat(student_model: keras.Model) -> keras.Model:
    """
    Apply Quantization Aware Training to student model.

    Args:
        student_model: FP32 student model (pre-trained/distilled)

    Returns:
        QAT-ready model with fake quantization nodes
    """
    logger.info("Applying QAT to student model")

    try:
        # Apply quantization to the entire model
        qat_model = tfmot.quantization.keras.quantize_model(student_model)
        logger.info("QAT applied successfully using quantize_model")

    except Exception as e:
        logger.warning(f"Could not apply quantize_model directly: {e}")
        logger.info("Trying quantize_annotate + quantize_apply approach")

        try:
            # Alternative approach: annotate then apply
            annotated_model = tfmot.quantization.keras.quantize_annotate_model(student_model)
            qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
            logger.info("QAT applied using annotate + apply approach")

        except Exception as e2:
            logger.error(f"Could not apply QAT: {e2}")
            logger.warning("Falling back to original model without QAT")
            qat_model = student_model

    logger.info(f"QAT model parameters: {qat_model.count_params():,}")

    return qat_model


def fine_tune_qat_model(
    qat_model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 10,
    lr: float = 1e-4,
    class_weights: Optional[dict] = None,
    teacher_model: Optional[keras.Model] = None,
    distill_alpha: float = 0.5,
    distill_temperature: float = 4.0,
    output_dir: Optional[str] = None,
    teacher_input_size: int = 300
) -> tuple:
    """
    Fine-tune QAT model.

    Can optionally continue using distillation during QAT fine-tuning
    for better accuracy preservation.

    Args:
        qat_model: QAT model to fine-tune
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of QAT fine-tuning epochs
        lr: Learning rate (should be low, e.g., 1e-4)
        class_weights: Optional class weights
        teacher_model: Optional teacher for continued distillation during QAT
        distill_alpha: Alpha for distillation (if teacher provided)
        distill_temperature: Temperature for distillation (if teacher provided)
        output_dir: Directory to save QAT model
        teacher_input_size: Input size expected by teacher model

    Returns:
        Tuple of (qat_model, metrics_dict)
    """
    logger.info(f"Fine-tuning QAT model for {epochs} epochs with lr={lr}")

    # Determine if we use distillation during QAT
    use_distillation = teacher_model is not None

    if use_distillation:
        logger.info("Using distillation during QAT fine-tuning")

        # Create a custom distiller for QAT
        # Note: This might not always work with QAT wrapper
        # If it fails, we fall back to standard training
        try:
            from trainer.models.distiller import Distiller

            distiller = Distiller(
                student=qat_model,
                teacher=teacher_model,
                alpha=distill_alpha,
                temperature=distill_temperature,
                teacher_input_size=teacher_input_size
            )

            distiller.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
            )

            history = distiller.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=1
            )

            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]

            metrics = {
                'qat_with_distillation': True,
                'final_train_accuracy': float(final_train_acc),
                'final_val_accuracy': float(final_val_acc),
                'epochs': epochs,
                'learning_rate': lr
            }

        except Exception as e:
            logger.warning(f"Could not use distillation with QAT: {e}")
            logger.info("Falling back to standard QAT fine-tuning (hard loss only)")
            use_distillation = False

    if not use_distillation:
        # Standard QAT fine-tuning with hard loss only
        logger.info("Using standard hard loss for QAT fine-tuning")

        qat_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        history = qat_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            verbose=1
        )

        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        metrics = {
            'qat_with_distillation': False,
            'final_train_accuracy': float(final_train_acc),
            'final_val_accuracy': float(final_val_acc),
            'epochs': epochs,
            'learning_rate': lr
        }

    logger.info(f"QAT fine-tuning complete - Train acc: {final_train_acc:.4f}, Val acc: {final_val_acc:.4f}")

    # Save QAT model
    if output_dir:
        qat_path = os.path.join(output_dir, 'student_qat_saved_model')
        qat_model.save(qat_path)
        logger.info(f"QAT model saved to {qat_path}")

        from trainer.utils.io import save_json
        save_json(metrics, os.path.join(output_dir, 'qat_metrics.json'))

    return qat_model, metrics
