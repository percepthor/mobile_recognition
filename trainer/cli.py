"""Command-line interface for training pipeline."""
import argparse
import logging
import os
import time
import pandas as pd

from trainer.config import TrainingConfig
from trainer.utils.seed import set_global_seed
from trainer.utils.logging import setup_logging
from trainer.utils.io import save_json, ensure_dir
from trainer.utils.versioning import create_run_metadata

# Import data modules
from trainer.data.dataset_scan import scan_dataset, get_class_to_index
from trainer.data.split import create_stratified_split
from trainer.data.tfdata import create_train_val_test_datasets, compute_class_weights

# Import model modules
from trainer.models.teacher import create_teacher_model, train_teacher
from trainer.models.student import create_student_model
from trainer.models.distiller import train_student_with_distillation

# Import quantization modules
from trainer.quant.qat import apply_qat, fine_tune_qat_model
from trainer.quant.rep_dataset import create_representative_dataset_generator
from trainer.quant.tflite_export import export_tflite_full_integer, create_model_metadata

# Import evaluation modules
from trainer.eval.tflite_runner import evaluate_tflite_on_dataset
from trainer.eval.metrics import compute_metrics
from trainer.eval.confusion_matrix import create_confusion_matrix, create_normalized_confusion_matrix
from trainer.eval.threshold import find_optimal_threshold, save_threshold_analysis
from trainer.eval.calibration import create_reliability_diagram


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train image recognition model with teacher-student distillation and QAT'
    )

    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset root directory with class subdirectories')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for all artifacts')

    # Model architecture
    parser.add_argument('--teacher_arch', type=str, default='efficientnet_b3',
                        choices=['efficientnet_b3', 'resnet101'],
                        help='Teacher architecture')
    parser.add_argument('--student_arch', type=str, default='efficientnet_lite1',
                        help='Student architecture (fixed: efficientnet_lite1)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (inferred from data if not provided)')
    parser.add_argument('--student_img_size', type=int, default=240,
                        help='Student input image size')
    parser.add_argument('--teacher_img_size', default='auto',
                        help='Teacher input image size (auto or int)')

    # Dataset and training
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='Training set fraction')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Validation set fraction')
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='Test set fraction')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs_teacher', type=int, default=10,
                        help='Epochs for teacher training')
    parser.add_argument('--epochs_student', type=int, default=20,
                        help='Epochs for student distillation')
    parser.add_argument('--epochs_qat', type=int, default=10,
                        help='Epochs for QAT fine-tuning')
    parser.add_argument('--lr_teacher', type=float, default=1e-3,
                        help='Teacher learning rate')
    parser.add_argument('--lr_student', type=float, default=1e-3,
                        help='Student learning rate')
    parser.add_argument('--lr_qat', type=float, default=1e-4,
                        help='QAT learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--use_class_weights', type=lambda x: x.lower() == 'true',
                        default=True, help='Use class weights for imbalanced data')

    # Distillation
    parser.add_argument('--distill_alpha', type=float, default=0.5,
                        help='Weight for hard loss in distillation')
    parser.add_argument('--distill_temperature', type=float, default=4.0,
                        help='Temperature for distillation')

    # Quantization / TFLite
    parser.add_argument('--qat', type=lambda x: x.lower() == 'true',
                        default=True, help='Apply QAT')
    parser.add_argument('--tflite_inference_input_type', type=str, default='int8',
                        choices=['int8', 'uint8'], help='TFLite input type')
    parser.add_argument('--tflite_inference_output_type', type=str, default='int8',
                        help='TFLite output type')
    parser.add_argument('--rep_data_num_batches', type=int, default=50,
                        help='Number of batches for representative dataset')
    parser.add_argument('--force_input_range_0_255', type=lambda x: x.lower() == 'true',
                        default=True, help='Force input range [0,255] in quantization')

    # Threshold
    parser.add_argument('--threshold_target_accept_accuracy', type=float, default=0.95,
                        help='Target accuracy for accepted predictions')
    parser.add_argument('--threshold_min_coverage', type=float, default=0.60,
                        help='Minimum coverage requirement')
    parser.add_argument('--threshold_penalty_incorrect', type=float, default=3.0,
                        help='Penalty for incorrect accepted predictions')

    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--save_tensorboard', type=lambda x: x.lower() == 'true',
                        default=True, help='Save TensorBoard logs')

    return parser.parse_args()


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()

    # Ensure output directory exists
    ensure_dir(args.output_dir)

    # Setup logging
    log_file = os.path.join(args.output_dir, 'training.log')
    global logger
    logger = setup_logging(args.log_level, log_file, name='trainer')

    logger.info("="*80)
    logger.info("IMAGE RECOGNITION TRAINING PIPELINE")
    logger.info("="*80)

    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        teacher_arch=args.teacher_arch,
        student_arch=args.student_arch,
        num_classes=args.num_classes,
        student_img_size=args.student_img_size,
        teacher_img_size=args.teacher_img_size,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        batch_size=args.batch_size,
        epochs_teacher=args.epochs_teacher,
        epochs_student=args.epochs_student,
        epochs_qat=args.epochs_qat,
        lr_teacher=args.lr_teacher,
        lr_student=args.lr_student,
        lr_qat=args.lr_qat,
        weight_decay=args.weight_decay,
        use_class_weights=args.use_class_weights,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        qat=args.qat,
        tflite_inference_input_type=args.tflite_inference_input_type,
        tflite_inference_output_type=args.tflite_inference_output_type,
        rep_data_num_batches=args.rep_data_num_batches,
        force_input_range_0_255=args.force_input_range_0_255,
        threshold_target_accept_accuracy=args.threshold_target_accept_accuracy,
        threshold_min_coverage=args.threshold_min_coverage,
        threshold_penalty_incorrect=args.threshold_penalty_incorrect,
        log_level=args.log_level,
        save_tensorboard=args.save_tensorboard
    )

    # Save config
    config.save(os.path.join(args.output_dir, 'training_config.json'))
    logger.info(f"Configuration saved to {args.output_dir}/training_config.json")

    # Set global seed
    set_global_seed(config.seed)
    logger.info(f"Global seed set to {config.seed}")

    # Start timer
    start_time = time.time()

    # === PHASE 1: Dataset Scanning and Splitting ===
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: Dataset Scanning and Splitting")
    logger.info("="*80)

    class_names, class_counts, bad_files = scan_dataset(config.data_dir, config.output_dir)

    # Infer num_classes if not provided
    if config.num_classes is None:
        config.num_classes = len(class_names)
        logger.info(f"Inferred num_classes = {config.num_classes}")
    elif config.num_classes != len(class_names):
        logger.warning(
            f"Provided num_classes ({config.num_classes}) != detected ({len(class_names)}). "
            f"Using detected value."
        )
        config.num_classes = len(class_names)

    # Create splits
    train_df, val_df, test_df = create_stratified_split(
        config.data_dir,
        class_names,
        config.output_dir,
        train_frac=config.train_frac,
        val_frac=config.val_frac,
        test_frac=config.test_frac,
        seed=config.seed
    )

    # Compute class weights if requested
    class_weights_dict = None
    if config.use_class_weights:
        class_weights_dict = compute_class_weights(train_df)

    # === PHASE 2: Create Datasets ===
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: Creating TensorFlow Datasets")
    logger.info("="*80)

    # Teacher datasets (larger images)
    logger.info(f"Creating teacher datasets (size={config.teacher_img_size})")
    train_ds_teacher, val_ds_teacher, test_ds_teacher = create_train_val_test_datasets(
        config.data_dir,
        config.output_dir,
        target_size=config.teacher_img_size,
        batch_size=config.batch_size,
        seed=config.seed,
        augment_train=True,
        return_range_0_255=True
    )

    # Student datasets (240x240)
    logger.info(f"Creating student datasets (size={config.student_img_size})")
    train_ds_student, val_ds_student, test_ds_student = create_train_val_test_datasets(
        config.data_dir,
        config.output_dir,
        target_size=config.student_img_size,
        batch_size=config.batch_size,
        seed=config.seed,
        augment_train=True,
        return_range_0_255=True
    )

    # === PHASE 3: Train Teacher ===
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: Training Teacher Model")
    logger.info("="*80)

    teacher = create_teacher_model(
        num_classes=config.num_classes,
        architecture=config.teacher_arch,
        input_size=config.teacher_img_size,
        pretrained=True
    )

    teacher, teacher_metrics = train_teacher(
        teacher,
        train_ds_teacher,
        val_ds_teacher,
        epochs=config.epochs_teacher,
        lr=config.lr_teacher,
        class_weights=class_weights_dict,
        output_dir=config.output_dir
    )

    # === PHASE 4: Train Student with Distillation ===
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: Training Student with Knowledge Distillation")
    logger.info("="*80)

    student = create_student_model(
        num_classes=config.num_classes,
        input_size=config.student_img_size
    )

    student, student_metrics = train_student_with_distillation(
        student,
        teacher,
        train_ds_student,
        val_ds_student,
        epochs=config.epochs_student,
        lr=config.lr_student,
        alpha=config.distill_alpha,
        temperature=config.distill_temperature,
        output_dir=config.output_dir,
        teacher_input_size=config.teacher_img_size
    )

    # === PHASE 5: QAT (if enabled) ===
    if config.qat:
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: Quantization Aware Training (QAT)")
        logger.info("="*80)

        qat_model = apply_qat(student)

        qat_model, qat_metrics = fine_tune_qat_model(
            qat_model,
            train_ds_student,
            val_ds_student,
            epochs=config.epochs_qat,
            lr=config.lr_qat,
            teacher_model=teacher,
            distill_alpha=config.distill_alpha,
            distill_temperature=config.distill_temperature,
            output_dir=config.output_dir,
            teacher_input_size=config.teacher_img_size
        )

        final_model = qat_model
    else:
        logger.info("\nQAT disabled, using FP32 student model")
        final_model = student

    # === PHASE 6: TFLite Export ===
    logger.info("\n" + "="*80)
    logger.info("PHASE 6: Exporting to TFLite (Full Integer)")
    logger.info("="*80)

    # Create representative dataset
    rep_dataset_gen = create_representative_dataset_generator(
        train_ds_student,
        num_batches=config.rep_data_num_batches,
        force_input_range_0_255=config.force_input_range_0_255
    )

    # Export to TFLite
    tflite_path = os.path.join(config.output_dir, 'model_qat_int8.tflite')
    export_result = export_tflite_full_integer(
        final_model,
        rep_dataset_gen,
        tflite_path,
        inference_input_type=config.tflite_inference_input_type,
        inference_output_type=config.tflite_inference_output_type
    )

    logger.info(f"TFLite model exported: {tflite_path}")

    # Create model metadata
    model_metadata = create_model_metadata(
        export_result['quantization_params'],
        target_size=config.student_img_size
    )
    save_json(model_metadata, os.path.join(config.output_dir, 'model_metadata.json'))
    logger.info("Model metadata saved")

    # === PHASE 7: Evaluation on Test Set ===
    logger.info("\n" + "="*80)
    logger.info("PHASE 7: Evaluating TFLite Model on Test Set")
    logger.info("="*80)

    y_true_test, y_pred_test, prob_max_test, all_probs_test = evaluate_tflite_on_dataset(
        tflite_path,
        config.data_dir,
        test_df,
        target_size=config.student_img_size
    )

    # Compute metrics
    test_metrics = compute_metrics(y_true_test, y_pred_test, class_names)
    save_json(test_metrics, os.path.join(config.output_dir, 'metrics.json'))
    logger.info("Test metrics saved")

    # Create confusion matrix
    create_confusion_matrix(y_true_test, y_pred_test, class_names, config.output_dir)
    create_normalized_confusion_matrix(y_true_test, y_pred_test, class_names, config.output_dir)

    # === PHASE 8: Threshold Calibration on Validation Set ===
    logger.info("\n" + "="*80)
    logger.info("PHASE 8: Threshold Calibration on Validation Set")
    logger.info("="*80)

    y_true_val, y_pred_val, prob_max_val, all_probs_val = evaluate_tflite_on_dataset(
        tflite_path,
        config.data_dir,
        val_df,
        target_size=config.student_img_size
    )

    recommended_threshold, threshold_metrics, threshold_curve_df = find_optimal_threshold(
        y_true_val,
        all_probs_val,
        target_accept_accuracy=config.threshold_target_accept_accuracy,
        min_coverage=config.threshold_min_coverage,
        penalty_incorrect=config.threshold_penalty_incorrect
    )

    save_threshold_analysis(threshold_curve_df, threshold_metrics, config.output_dir)

    # === PHASE 9: Calibration Analysis ===
    logger.info("\n" + "="*80)
    logger.info("PHASE 9: Calibration Analysis")
    logger.info("="*80)

    calibration_metrics = create_reliability_diagram(
        y_true_test,
        y_pred_test,
        prob_max_test,
        config.output_dir
    )

    # === PHASE 10: Save Run Metadata ===
    logger.info("\n" + "="*80)
    logger.info("PHASE 10: Saving Run Metadata")
    logger.info("="*80)

    end_time = time.time()
    training_time = end_time - start_time

    run_metadata = create_run_metadata(
        training_time_seconds=training_time,
        additional_info={
            'num_classes': config.num_classes,
            'total_train_samples': len(train_df),
            'total_val_samples': len(val_df),
            'total_test_samples': len(test_df),
            'teacher_architecture': config.teacher_arch,
            'student_architecture': config.student_arch,
            'qat_enabled': config.qat
        }
    )

    save_json(run_metadata, os.path.join(config.output_dir, 'run_metadata.json'))

    # === COMPLETE ===
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {training_time/60:.2f} minutes")
    logger.info(f"Test accuracy: {test_metrics['accuracy_top1']:.4f}")
    logger.info(f"Recommended threshold: {recommended_threshold:.3f}")
    logger.info(f"All artifacts saved to: {config.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
