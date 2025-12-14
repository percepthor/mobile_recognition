# Image Recognition Training Pipeline

A reproducible Docker-based training pipeline for image recognition models using knowledge distillation and quantization-aware training (QAT). Produces optimized TFLite models for mobile deployment.

## Features

- **Knowledge Distillation**: Teacher-student architecture for model compression
- **Quantization-Aware Training (QAT)**: Full integer quantization (int8) for efficient mobile inference
- **Automatic Threshold Calibration**: Confidence threshold recommendation for "unknown" class detection
- **Reproducible**: Fully containerized with fixed random seeds
- **Complete Evaluation**: Confusion matrices, calibration curves, and comprehensive metrics

## Architecture

- **Teacher Model**: EfficientNet-B3 or ResNet101 (for knowledge transfer)
- **Student Model**: EfficientNet-Lite1 (mobile-optimized, 240×240 input)
- **Output**: Full integer TFLite model with int8 quantization

## Quick Start

### Prerequisites

- Docker
- Dataset organized as:
  ```
  dataset/
    class_1/
      image_001.jpg
      image_002.jpg
      ...
    class_2/
      image_001.jpg
      ...
  ```

### Build and Run

```bash
# Build Docker image
docker build -t image_recognition_training:latest .

# Run training
./scripts/run_in_docker.sh \
  --data_dir /path/to/dataset \
  --output_dir /path/to/output \
  --build

# Run smoke test
./scripts/smoke_test.sh
```

### Python CLI (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python -m trainer.cli \
  --data_dir /path/to/dataset \
  --output_dir /path/to/output
```

## Output Artifacts

The pipeline generates the following artifacts in `--output_dir`:

### Required Outputs

- **`model_qat_int8.tflite`**: Quantized TFLite model (full integer, int8)
- **`labels.txt`**: Class names in alphabetical order
- **`model_metadata.json`**: Preprocessing and quantization parameters
- **`confusion_matrix.png`**: Confusion matrix visualization
- **`threshold_recommendation.json`**: Recommended confidence threshold

### Additional Outputs

- **`metrics.json`**: Test set accuracy, F1 scores, per-class metrics
- **`training_config.json`**: Complete training configuration
- **`run_metadata.json`**: System info, versions, training time
- **`threshold_curve.csv`**: Threshold analysis data
- **`threshold_analysis.png`**: Threshold selection visualization
- **`reliability_diagram.png`**: Model calibration analysis
- **`confidence_histogram.png`**: Distribution of prediction confidences
- **`dataset_manifest.json`**: Dataset statistics
- **`train.csv`, `val.csv`, `test.csv`**: Dataset splits

## Configuration Options

### Model Architecture

```bash
--teacher_arch efficientnet_b3  # or resnet101
--teacher_img_size auto         # auto or integer (auto: 300 for B3, 224 for ResNet)
--student_img_size 240          # Student input size (default: 240)
```

### Training Parameters

```bash
--epochs_teacher 10       # Teacher training epochs
--epochs_student 20       # Student distillation epochs
--epochs_qat 10          # QAT fine-tuning epochs
--batch_size 32          # Batch size
--lr_teacher 1e-3        # Teacher learning rate
--lr_student 1e-3        # Student learning rate
--lr_qat 1e-4            # QAT learning rate
--seed 42                # Random seed
```

### Dataset Split

```bash
--train_frac 0.8         # Training set fraction
--val_frac 0.1           # Validation set fraction
--test_frac 0.1          # Test set fraction
```

### Distillation

```bash
--distill_alpha 0.5          # Weight for hard loss (1-alpha for distillation loss)
--distill_temperature 4.0    # Temperature for soft targets
```

### Quantization

```bash
--qat true                           # Enable QAT (default: true)
--tflite_inference_input_type int8   # Input dtype: int8 or uint8
--tflite_inference_output_type int8  # Output dtype (must be int8)
--rep_data_num_batches 50           # Representative dataset size
--force_input_range_0_255 true      # Force [0,255] range calibration
```

### Threshold Calibration

```bash
--threshold_target_accept_accuracy 0.95  # Target accuracy for accepted samples
--threshold_min_coverage 0.60            # Minimum coverage requirement
--threshold_penalty_incorrect 3.0        # Penalty for accepting incorrect predictions
```

## Preprocessing

The pipeline uses **letterbox preprocessing** to maintain aspect ratio:

1. Scale image so `max(height, width)` fits target size
2. Resize using bilinear interpolation
3. Pad with black pixels (0,0,0) to center image
4. Output: 240×240×3 RGB in [0, 255] range

This preprocessing is the **source of truth** for mobile deployment and must be replicated exactly in the inference code.

## Pipeline Phases

1. **Dataset Scanning**: Validate images, create class mapping
2. **Data Splitting**: Stratified train/val/test split
3. **Teacher Training**: Train large teacher model
4. **Student Distillation**: Train student using teacher knowledge
5. **QAT**: Quantization-aware training (if enabled)
6. **TFLite Export**: Convert to full-integer TFLite
7. **Test Evaluation**: Compute metrics on test set
8. **Threshold Calibration**: Find optimal confidence threshold on validation set
9. **Calibration Analysis**: Analyze model calibration

## Model Metadata

The `model_metadata.json` file contains everything needed for mobile deployment:

```json
{
  "model_file": "model_qat_int8.tflite",
  "labels_file": "labels.txt",
  "input": {
    "width": 240,
    "height": 240,
    "channels": 3,
    "dtype": "int8",
    "quantization": {
      "scale": 1.0,
      "zero_point": -128
    }
  },
  "preprocess": {
    "letterbox": true,
    "keep_aspect_ratio": true,
    "pad_color_rgb": [0, 0, 0],
    "interpolation": "bilinear"
  },
  "output": {
    "dtype": "int8",
    "quantization": {
      "scale": 0.02,
      "zero_point": -3
    },
    "is_logits": true,
    "softmax_in_model": false
  }
}
```

## Reproducibility

- Fixed random seeds across Python, NumPy, and TensorFlow
- Deterministic operations enabled
- Complete configuration saved
- System information and package versions logged

## Project Structure

```
image_recognition_mobile/
├── Dockerfile
├── requirements.txt
├── README.md
├── trainer/
│   ├── __init__.py
│   ├── cli.py              # Main entry point
│   ├── config.py           # Configuration management
│   ├── data/
│   │   ├── dataset_scan.py # Dataset scanning
│   │   ├── split.py        # Train/val/test splitting
│   │   ├── preprocess.py   # Preprocessing (letterbox)
│   │   └── tfdata.py       # TensorFlow Dataset pipeline
│   ├── models/
│   │   ├── teacher.py      # Teacher model
│   │   ├── student.py      # Student model
│   │   └── distiller.py    # Knowledge distillation
│   ├── quant/
│   │   ├── qat.py          # Quantization-aware training
│   │   ├── tflite_export.py # TFLite conversion
│   │   └── rep_dataset.py  # Representative dataset
│   ├── eval/
│   │   ├── tflite_runner.py    # TFLite inference
│   │   ├── metrics.py          # Metrics computation
│   │   ├── confusion_matrix.py # Confusion matrix
│   │   ├── threshold.py        # Threshold calibration
│   │   └── calibration.py      # Calibration analysis
│   └── utils/
│       ├── seed.py         # Random seed management
│       ├── io.py           # I/O utilities
│       ├── logging.py      # Logging setup
│       └── versioning.py   # Version tracking
└── scripts/
    ├── run_in_docker.sh    # Docker run helper
    └── smoke_test.sh       # Automated testing
```

## Examples

### Minimal Training

```bash
python -m trainer.cli \
  --data_dir ./dataset \
  --output_dir ./output
```

### Custom Configuration

```bash
python -m trainer.cli \
  --data_dir ./dataset \
  --output_dir ./output \
  --teacher_arch resnet101 \
  --epochs_teacher 15 \
  --epochs_student 25 \
  --batch_size 64 \
  --distill_temperature 5.0
```

### Quick Test (1 epoch each)

```bash
python -m trainer.cli \
  --data_dir ./dataset \
  --output_dir ./output \
  --epochs_teacher 1 \
  --epochs_student 1 \
  --epochs_qat 1
```

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Use smaller teacher: `--teacher_arch efficientnet_b3` (instead of resnet101)

### Poor Accuracy

- Increase epochs: `--epochs_teacher 20 --epochs_student 30`
- Adjust distillation: `--distill_alpha 0.3 --distill_temperature 5.0`
- Enable class weights: `--use_class_weights true`

### TFLite Conversion Fails

- Check that input is in [0, 255] range
- Ensure QAT is enabled: `--qat true`
- Try `--force_input_range_0_255 true`

## License

Desarrollado por Felipe Lara felipe@lara.ac

## Support

For issues and questions, please check the training logs in `output_dir/training.log`.
