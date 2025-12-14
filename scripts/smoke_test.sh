#!/bin/bash
# Smoke test for training pipeline
# Creates a minimal synthetic dataset and runs a quick training pass

set -e

echo "=================================="
echo "Running Smoke Test"
echo "=================================="

# Configuration
TEST_DIR="./smoke_test_temp"
DATA_DIR="$TEST_DIR/data"
OUTPUT_DIR="$TEST_DIR/output"
IMAGE_NAME="image_recognition_training:latest"

# Clean up previous test
if [ -d "$TEST_DIR" ]; then
    echo "Cleaning up previous test..."
    rm -rf "$TEST_DIR"
fi

# Create test directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Create synthetic dataset (2 classes, 10 images each)
echo "Creating synthetic dataset..."

create_synthetic_images() {
    local class_name=$1
    local num_images=$2
    local class_dir="$DATA_DIR/$class_name"

    mkdir -p "$class_dir"

    for i in $(seq 1 $num_images); do
        # Create a simple colored image using Python
        python3 -c "
from PIL import Image
import numpy as np
import random

# Random color for this class
if '$class_name' == 'class_a':
    color = (255, 0, 0)  # Red
else:
    color = (0, 0, 255)  # Blue

# Create 100x100 image with some noise
img_array = np.full((100, 100, 3), color, dtype=np.uint8)
noise = np.random.randint(-30, 30, (100, 100, 3), dtype=np.int16)
img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

img = Image.fromarray(img_array, 'RGB')
img.save('$class_dir/image_${i}.jpg')
        "
    done

    echo "  Created $num_images images for $class_name"
}

create_synthetic_images "class_a" 10
create_synthetic_images "class_b" 10

echo "Synthetic dataset created at: $DATA_DIR"

# Build Docker image if it doesn't exist
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Building Docker image..."
    docker build -t "$IMAGE_NAME" .
fi

# Run smoke test with minimal epochs
echo ""
echo "Running training pipeline with minimal configuration..."
echo ""

docker run --rm \
    -v "$(pwd)/$DATA_DIR":/data \
    -v "$(pwd)/$OUTPUT_DIR":/out \
    "$IMAGE_NAME" \
    --data_dir /data \
    --output_dir /out \
    --epochs_teacher 1 \
    --epochs_student 1 \
    --epochs_qat 1 \
    --batch_size 4 \
    --rep_data_num_batches 2

echo ""
echo "=================================="
echo "Validating outputs..."
echo "=================================="

# Validate required outputs exist
REQUIRED_FILES=(
    "model_qat_int8.tflite"
    "labels.txt"
    "confusion_matrix.png"
    "threshold_recommendation.json"
    "model_metadata.json"
    "metrics.json"
    "training_config.json"
    "run_metadata.json"
)

ALL_PASS=true

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file MISSING"
        ALL_PASS=false
    fi
done

# Validate TFLite model properties
echo ""
echo "Validating TFLite model..."

python3 -c "
import tensorflow as tf
import numpy as np
import sys

try:
    interpreter = tf.lite.Interpreter(model_path='$OUTPUT_DIR/model_qat_int8.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details['dtype']
    output_dtype = output_details['dtype']

    print(f'Input dtype: {input_dtype}')
    print(f'Output dtype: {output_dtype}')

    # Validate quantization
    if input_dtype not in [np.int8, np.uint8]:
        print('✗ Input is not quantized (expected int8 or uint8)')
        sys.exit(1)
    else:
        print('✓ Input is quantized')

    if output_dtype != np.int8:
        print('✗ Output is not int8')
        sys.exit(1)
    else:
        print('✓ Output is quantized (int8)')

    print('✓ TFLite model validation passed')

except Exception as e:
    print(f'✗ TFLite validation failed: {e}')
    sys.exit(1)
" || ALL_PASS=false

echo ""
echo "=================================="

if [ "$ALL_PASS" = true ]; then
    echo "✓ SMOKE TEST PASSED"
    echo "=================================="
    echo ""
    echo "Cleaning up test directory..."
    rm -rf "$TEST_DIR"
    exit 0
else
    echo "✗ SMOKE TEST FAILED"
    echo "=================================="
    echo ""
    echo "Test artifacts preserved at: $TEST_DIR"
    exit 1
fi
