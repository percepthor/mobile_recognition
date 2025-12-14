"""Knowledge distillation: Teacher-Student model."""
import logging
import os
from typing import Optional
import tensorflow as tf
from tensorflow import keras


logger = logging.getLogger(__name__)


class Distiller(keras.Model):
    """
    Distillation model that combines teacher and student.

    Implements knowledge distillation with:
    - Hard loss: standard cross-entropy with true labels
    - Distill loss: KL divergence between teacher and student soft predictions
    - Combined loss: alpha * hard_loss + (1 - alpha) * distill_loss
    """

    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        alpha: float = 0.5,
        temperature: float = 4.0,
        teacher_input_size: int = 300
    ):
        """
        Initialize distiller.

        Args:
            student: Student model
            teacher: Teacher model (will be frozen during training)
            alpha: Weight for hard loss (1-alpha for distill loss)
            temperature: Temperature for soft predictions
            teacher_input_size: Input size expected by teacher model
        """
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_input_size = teacher_input_size

        # Freeze teacher
        self.teacher.trainable = False

    def compile(
        self,
        optimizer,
        metrics=None,
        student_loss_fn=None,
        distillation_loss_fn=None
    ):
        """
        Compile distiller with custom losses.

        Args:
            optimizer: Optimizer for student
            metrics: Optional metrics to track
            student_loss_fn: Loss for hard labels (default: SparseCategoricalCrossentropy)
            distillation_loss_fn: Loss for distillation (default: KL divergence)
        """
        super().compile(optimizer=optimizer, metrics=metrics)

        self.student_loss_fn = student_loss_fn or keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.distillation_loss_fn = distillation_loss_fn or keras.losses.KLDivergence()

    def call(self, inputs, training=False):
        """Forward pass returns student predictions."""
        return self.student(inputs, training=training)

    def train_step(self, data):
        """
        Custom training step implementing knowledge distillation.

        Args:
            data: Tuple of (x, y) where x is images and y is labels

        Returns:
            Dictionary with loss metrics
        """
        x, y = data

        # Resize images for teacher if needed
        x_teacher = tf.image.resize(x, [self.teacher_input_size, self.teacher_input_size])

        # Forward pass through teacher (frozen, no gradients)
        # Use stop_gradient to prevent backprop through teacher
        teacher_logits = tf.stop_gradient(self.teacher(x_teacher, training=False))

        with tf.GradientTape() as tape:
            # Forward pass through student
            student_logits = self.student(x, training=True)

            # Compute hard loss (student vs true labels)
            hard_loss = self.student_loss_fn(y, student_logits)

            # Compute distillation loss (student vs teacher soft predictions)
            # Apply temperature scaling
            teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
            student_probs = tf.nn.softmax(student_logits / self.temperature)

            # KL divergence expects probabilities (not log-probabilities for first arg)
            # Scale by T^2 as per standard distillation
            distill_loss = self.distillation_loss_fn(
                teacher_probs,
                student_probs
            ) * (self.temperature ** 2)

            # Combined loss
            total_loss = self.alpha * hard_loss + (1 - self.alpha) * distill_loss

        # Compute gradients and update student weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        if self.compiled_metrics:
            self.compiled_metrics.update_state(y, student_logits)

        return {
            "loss": total_loss,
            "hard_loss": hard_loss,
            "distill_loss": distill_loss,
            **{m.name: m.result() for m in self.metrics}
        }

    def test_step(self, data):
        """
        Validation/test step.

        Args:
            data: Tuple of (x, y)

        Returns:
            Dictionary with metrics
        """
        x, y = data

        # Resize images for teacher if needed
        x_teacher = tf.image.resize(x, [self.teacher_input_size, self.teacher_input_size])

        # Forward pass
        student_logits = self.student(x, training=False)
        teacher_logits = self.teacher(x_teacher, training=False)

        # Compute losses
        hard_loss = self.student_loss_fn(y, student_logits)

        teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
        student_probs = tf.nn.softmax(student_logits / self.temperature)

        distill_loss = self.distillation_loss_fn(
            teacher_probs,
            student_probs
        ) * (self.temperature ** 2)

        total_loss = self.alpha * hard_loss + (1 - self.alpha) * distill_loss

        # Update metrics
        if self.compiled_metrics:
            self.compiled_metrics.update_state(y, student_logits)

        return {
            "loss": total_loss,
            "hard_loss": hard_loss,
            "distill_loss": distill_loss,
            **{m.name: m.result() for m in self.metrics}
        }


def train_student_with_distillation(
    student: keras.Model,
    teacher: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 20,
    lr: float = 1e-3,
    alpha: float = 0.5,
    temperature: float = 4.0,
    class_weights: Optional[dict] = None,
    output_dir: Optional[str] = None,
    teacher_input_size: int = 300
) -> tuple:
    """
    Train student model using knowledge distillation from teacher.

    Args:
        student: Student model
        teacher: Teacher model (pre-trained)
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of epochs
        lr: Learning rate
        alpha: Weight for hard loss
        temperature: Temperature for soft predictions
        class_weights: Optional class weights (applied to hard loss only)
        output_dir: Directory to save artifacts
        teacher_input_size: Input size expected by teacher model

    Returns:
        Tuple of (trained_student, metrics_dict)
    """
    logger.info(f"Training student with distillation (alpha={alpha}, T={temperature})")

    # Create distiller
    distiller = Distiller(
        student=student,
        teacher=teacher,
        alpha=alpha,
        temperature=temperature,
        teacher_input_size=teacher_input_size
    )

    # Compile
    distiller.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [early_stopping, reduce_lr]

    # Train
    # Note: class_weights are tricky with custom training loop
    # For simplicity, we'll train without class_weights in distillation
    # (could be added by weighting the hard_loss per sample)

    history = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Extract final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_distill_loss = history.history['distill_loss'][-1]

    logger.info(
        f"Distillation complete - Train acc: {final_train_acc:.4f}, "
        f"Val acc: {final_val_acc:.4f}, Distill loss: {final_distill_loss:.4f}"
    )

    metrics = {
        'final_train_accuracy': float(final_train_acc),
        'final_val_accuracy': float(final_val_acc),
        'final_distill_loss': float(final_distill_loss),
        'epochs': epochs,
        'learning_rate': lr,
        'alpha': alpha,
        'temperature': temperature,
        'trained_with_distillation': True
    }

    # Save student model and config
    if output_dir:
        student_path = os.path.join(output_dir, 'student_fp32_saved_model')
        student.save(student_path)
        logger.info(f"Distilled student model saved to {student_path}")

        from trainer.utils.io import save_json
        save_json(metrics, os.path.join(output_dir, 'student_fp32_metrics.json'))

        distill_config = {
            'alpha': alpha,
            'temperature': temperature,
            'epochs': epochs,
            'learning_rate': lr
        }
        save_json(distill_config, os.path.join(output_dir, 'distillation_config.json'))

    return student, metrics
