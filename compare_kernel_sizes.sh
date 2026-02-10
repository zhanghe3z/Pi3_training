#!/bin/bash
# Compare different kernel sizes on the same frame
# Shows how variance computation changes with different window sizes

set -e

FRAME=100
OUTPUT_DIR="variance_kernel_comparison"
mkdir -p ${OUTPUT_DIR}

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Kernel Size Comparison - Frame ${FRAME}                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Different kernel sizes to compare
KERNEL_SIZES=(5 7 11 15 21 31)

for K in "${KERNEL_SIZES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Kernel size: ${K}×${K}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python visualize_depth_variance.py \
        --frame ${FRAME} \
        --kernel_size ${K} \
        --save ${OUTPUT_DIR}/variance_frame${FRAME}_k${K}.png \
        2>&1 | grep -E "Variance (range|mean|median)"

    echo ""
done

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Comparison Complete!                                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Comparison images saved to: ${OUTPUT_DIR}/"
echo ""
echo "Observations:"
echo "  • Smaller kernels (5×5) → more local variance, sharper edges"
echo "  • Larger kernels (31×31) → smoother variance, global patterns"
echo "  • Training uses 7×7 (good balance between local and global)"
