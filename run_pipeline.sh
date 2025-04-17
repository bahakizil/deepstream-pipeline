#!/bin/bash
set -e

# Validate the environment setup
./validate_setup.sh

# Check if help is requested
if [[ "$1" == "--help" ]]; then
    echo "DeepStream Pipeline Usage:"
    echo "  ./run_pipeline.sh [options]"
    echo ""
    echo "Options:"
    echo "  --input=PATH      Path to input video (default: /videos/input.mp4)"
    echo "  --output=PATH     Path to output video (default: /videos/output.mp4)"
    echo "  --config=PATH     Path to DeepStream config (default: /app/deepstream_app_config.txt)"
    echo "  --model=MODEL     Model to use (default: yolov8)"
    echo "  --no-display      Run without display"
    exit 0
fi

# Default parameters
INPUT_VIDEO="/videos/input.mp4"
OUTPUT_VIDEO="/videos/output.mp4"
CONFIG_FILE="/app/deepstream_app_config.txt"
MODEL="yolov8"
DISPLAY_FLAG="--display"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --input=*)
        INPUT_VIDEO="${arg#*=}"
        ;;
        --output=*)
        OUTPUT_VIDEO="${arg#*=}"
        ;;
        --config=*)
        CONFIG_FILE="${arg#*=}"
        ;;
        --model=*)
        MODEL="${arg#*=}"
        ;;
        --no-display)
        DISPLAY_FLAG=""
        ;;
    esac
done

# Check if input video exists
if [ ! -f "$INPUT_VIDEO" ] && [ "$INPUT_VIDEO" != "/dev/video0" ]; then
    echo "Error: Input video not found at $INPUT_VIDEO"
    echo "Please mount a video directory to /videos or specify a different path"
    exit 1
fi

# Run the pipeline
echo "Starting DeepStream Pipeline..."
echo "Input: $INPUT_VIDEO"
echo "Output: $OUTPUT_VIDEO"
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL"

python3 /app/integrated_pipeline.py \
    --input "$INPUT_VIDEO" \
    --output "$OUTPUT_VIDEO" \
    --config "$CONFIG_FILE" \
    --model "$MODEL" \
    $DISPLAY_FLAG 