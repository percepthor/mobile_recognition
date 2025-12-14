#!/bin/bash
# Script to run training pipeline in Docker container

set -e

# Default values
IMAGE_NAME="image_recognition_training:latest"
DATA_DIR=""
OUTPUT_DIR=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --build)
            BUILD=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --data_dir /path/to/data --output_dir /path/to/output [--build] [additional training args]"
    echo ""
    echo "Required:"
    echo "  --data_dir      Path to dataset directory"
    echo "  --output_dir    Path to output directory"
    echo ""
    echo "Optional:"
    echo "  --build         Build Docker image before running"
    echo "  [additional]    Any additional arguments passed to trainer CLI"
    echo ""
    echo "Example:"
    echo "  $0 --data_dir ./dataset --output_dir ./output --build --epochs_teacher 5"
    exit 1
fi

# Build image if requested
if [ "$BUILD" = true ]; then
    echo "Building Docker image: $IMAGE_NAME"
    docker build -t "$IMAGE_NAME" .
fi

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Docker image $IMAGE_NAME not found. Building..."
    docker build -t "$IMAGE_NAME" .
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Convert to absolute paths
DATA_DIR=$(cd "$DATA_DIR" && pwd)
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

echo "=================================="
echo "Running Image Recognition Training"
echo "=================================="
echo "Data directory:   $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Extra arguments:  $EXTRA_ARGS"
echo "=================================="

# Run container
docker run --rm \
    -v "$DATA_DIR":/data \
    -v "$OUTPUT_DIR":/out \
    "$IMAGE_NAME" \
    --data_dir /data \
    --output_dir /out \
    $EXTRA_ARGS

echo ""
echo "Training complete! Artifacts saved to: $OUTPUT_DIR"
