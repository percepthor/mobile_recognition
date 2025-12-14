# Model Assets Directory

Place your trained model files here:

## Required Files

1. **model_qat_int8.tflite**
   - Quantized TFLite model (int8)
   - Output from the training pipeline
   - From: `trainer/output/model_qat_int8.tflite`

2. **labels.txt**
   - Class labels (one per line, alphabetical order)
   - From: `trainer/output/labels.txt`

3. **threshold_recommendation.json**
   - Recommended confidence threshold
   - From: `trainer/output/threshold_recommendation.json`

4. **runtime_config.json**
   - Runtime configuration
   - Already provided in this directory

## How to Get These Files

After training your model with the training pipeline:

```bash
# Copy from training output
cp /path/to/trainer/output/model_qat_int8.tflite ./
cp /path/to/trainer/output/labels.txt ./
cp /path/to/trainer/output/threshold_recommendation.json ./
```

## Directory Structure

After adding the files, you should have:

```
assets/image_recognition/
├── model_qat_int8.tflite       # Your trained model
├── labels.txt                   # Your class labels
├── threshold_recommendation.json # Auto-generated threshold
└── runtime_config.json          # Runtime settings
```
