"""Training configuration management."""
import os
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Complete training configuration matching CLI arguments."""

    # Required paths
    data_dir: str
    output_dir: str

    # Model architecture
    teacher_arch: str = 'efficientnet_b3'
    student_arch: str = 'efficientnet_lite1'
    num_classes: Optional[int] = None
    student_img_size: int = 240
    teacher_img_size: str = 'auto'  # 'auto' or int

    # Dataset and training
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    batch_size: int = 32
    epochs_teacher: int = 10
    epochs_student: int = 20
    epochs_qat: int = 10
    lr_teacher: float = 1e-3
    lr_student: float = 1e-3
    lr_qat: float = 1e-4
    weight_decay: float = 0.0
    use_class_weights: bool = True

    # Distillation
    distill_alpha: float = 0.5
    distill_temperature: float = 4.0

    # Quantization / TFLite
    qat: bool = True
    tflite_inference_input_type: str = 'int8'
    tflite_inference_output_type: str = 'int8'
    rep_data_num_batches: int = 50
    force_input_range_0_255: bool = True

    # Threshold
    threshold_target_accept_accuracy: float = 0.95
    threshold_min_coverage: float = 0.60
    threshold_penalty_incorrect: float = 3.0

    # Logging
    log_level: str = 'INFO'
    save_tensorboard: bool = True

    def __post_init__(self):
        """Validate and resolve configuration."""
        # Resolve teacher image size
        if self.teacher_img_size == 'auto':
            if self.teacher_arch == 'efficientnet_b3':
                self.teacher_img_size = 300
            elif self.teacher_arch == 'resnet101':
                self.teacher_img_size = 224
            else:
                self.teacher_img_size = 224  # Default
        else:
            self.teacher_img_size = int(self.teacher_img_size)

        # Validate fractions
        total_frac = self.train_frac + self.val_frac + self.test_frac
        if abs(total_frac - 1.0) > 1e-6:
            raise ValueError(f"train_frac + val_frac + test_frac must equal 1.0, got {total_frac}")

        # Validate architectures
        if self.teacher_arch not in ['efficientnet_b3', 'resnet101']:
            raise ValueError(f"teacher_arch must be 'efficientnet_b3' or 'resnet101', got {self.teacher_arch}")

        if self.student_arch != 'efficientnet_lite1':
            raise ValueError(f"student_arch must be 'efficientnet_lite1', got {self.student_arch}")

        # Validate types
        if self.tflite_inference_input_type not in ['int8', 'uint8']:
            raise ValueError(f"tflite_inference_input_type must be 'int8' or 'uint8'")

        if self.tflite_inference_output_type != 'int8':
            raise ValueError(f"tflite_inference_output_type must be 'int8'")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to save JSON
        """
        from trainer.utils.io import save_json
        save_json(self.to_dict(), filepath)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            TrainingConfig instance
        """
        return cls(**config_dict)

    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            TrainingConfig instance
        """
        from trainer.utils.io import load_json
        config_dict = load_json(filepath)
        return cls.from_dict(config_dict)


def create_default_config(data_dir: str, output_dir: str) -> TrainingConfig:
    """
    Create configuration with default values.

    Args:
        data_dir: Data directory path
        output_dir: Output directory path

    Returns:
        TrainingConfig with defaults
    """
    return TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir
    )
