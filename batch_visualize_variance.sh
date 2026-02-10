#!/bin/bash
# Batch visualize depth variance for multiple frames
# This script demonstrates how to visualize variance for several frames

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Batch Depth Variance Visualization - Hospital Dataset        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Create output directory
OUTPUT_DIR="variance_visualizations"
mkdir -p ${OUTPUT_DIR}

# Frames to visualize
FRAMES=(0 50 100 150 200 250 300)

echo "Visualizing frames: ${FRAMES[@]}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Visualize each frame
for FRAME in "${FRAMES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Processing Frame ${FRAME}..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python visualize_depth_variance.py \
        --frame ${FRAME} \
        --save ${OUTPUT_DIR}/variance_frame_${FRAME}.png \
        2>&1 | grep -E "(Loaded|Depth range|Variance range|saved)"

    echo ""
done

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Batch Visualization Complete!                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Visualizations saved to: ${OUTPUT_DIR}/"
ls -lh ${OUTPUT_DIR}/*.png | wc -l | xargs echo "Total images:"
echo ""
echo "To view all images:"
echo "  ls ${OUTPUT_DIR}/*.png"
