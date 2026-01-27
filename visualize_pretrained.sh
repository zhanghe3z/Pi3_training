#!/bin/bash

# Visualize depth predictions using pretrained Pi3 model
# Example usage for pretrained checkpoint downloaded from web

# Default paths (modify these based on your setup)
PRETRAINED_CKPT="ckpts/pi3_pretrained.pth"  # Path to your downloaded pretrained checkpoint
DATA_ROOT="/mnt/localssd/tartanair_tools/tartanair_data/hospital"
OUTPUT_DIR="./outputs/pi3_pretrained_visualizations"
NUM_SAMPLES=5
FRAME_NUM=8

# Run visualization with depth scaling
python visualize_depth.py \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --frame_num $FRAME_NUM \
    --scale_depth \
    --device cuda

echo "Visualization complete! Check $OUTPUT_DIR/depth_visualizations/"
