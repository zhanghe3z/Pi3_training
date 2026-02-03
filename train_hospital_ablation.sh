#!/bin/bash

# Pi3 Ablation Study Training Script
# This script trains the Pi3 model with ablations:
# 1. Pure Linear Head (no Transformer decoder for point prediction)
# 2. Simple L1 Loss (no scale-invariant alignment)

set -e  # Exit on error

# Configuration
NUM_GPUS=8  # Adjust based on your available GPUs
NUM_MACHINES=1
DATA_CONFIG="tartanair_hospital"

# Output directory
OUTPUT_DIR="outputs/pi3_hospital_ablation"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Pi3 ABLATION STUDY Training"
echo "=========================================="
echo "Ablations:"
echo "  1. Pure Linear Head (no Transformer decoder)"
echo "  2. Simple L1 Loss (no scale-invariant alignment)"
echo "=========================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Data Config: ${DATA_CONFIG}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

# Stage 1: Low-Resolution Training (224x224)
echo ""
echo "[Stage 1/3] Starting Low-Resolution Training (ABLATION)..."
echo "Resolution: 224x224"
echo "Epochs: 80"
echo "Depth Visualization: Every 200 steps"
echo "Number of Samples: 2"
echo ""

accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes ${NUM_GPUS} \
    --num_machines ${NUM_MACHINES} \
    scripts/train_pi3.py \
    train=train_pi3_lowres \
    data=${DATA_CONFIG} \
    model=pi3_ablation \
    name=pi3_hospital_lowres_ablation \
    log.use_wandb=true \
    log.use_tensorboard=false \
    viz_interval=200 \
    num_viz_samples=2

# Check if Stage 1 completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "[Stage 1/3] Low-Resolution Training Completed Successfully!"
    echo ""
else
    echo ""
    echo "[ERROR] Stage 1 failed. Please check the logs."
    exit 1
fi

# Find the latest checkpoint from Stage 1
LOWRES_CKPT=$(find outputs/pi3_hospital_lowres_ablation/ckpts -name "*.pth" | sort -V | tail -n 1)

if [ -z "${LOWRES_CKPT}" ]; then
    echo "[ERROR] No checkpoint found from Stage 1. Cannot proceed to Stage 2."
    exit 1
fi

echo "Found Stage 1 checkpoint: ${LOWRES_CKPT}"
echo ""

# Stage 2: High-Resolution Training
echo "[Stage 2/3] Starting High-Resolution Training (ABLATION)..."
echo "Loading checkpoint: ${LOWRES_CKPT}"
echo "Epochs: 40"
echo "Depth Visualization: Every 200 steps"
echo "Number of Samples: 4"
echo ""

accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes ${NUM_GPUS} \
    --num_machines ${NUM_MACHINES} \
    scripts/train_pi3.py \
    train=train_pi3_highres \
    data=${DATA_CONFIG} \
    model=pi3_ablation \
    name=pi3_hospital_highres_ablation \
    model.ckpt=${LOWRES_CKPT} \
    log.use_wandb=true \
    log.use_tensorboard=false \
    viz_interval=200 \
    num_viz_samples=4

# Check if Stage 2 completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "[Stage 2/3] High-Resolution Training Completed Successfully!"
    echo ""
else
    echo ""
    echo "[ERROR] Stage 2 failed. Please check the logs."
    exit 1
fi

# Find the latest checkpoint from Stage 2
HIGHRES_CKPT=$(find outputs/pi3_hospital_highres_ablation/ckpts -name "*.pth" | sort -V | tail -n 1)

if [ -z "${HIGHRES_CKPT}" ]; then
    echo "[ERROR] No checkpoint found from Stage 2."
    exit 1
fi

echo "Found Stage 2 checkpoint: ${HIGHRES_CKPT}"
echo ""

echo ""
echo "=========================================="
echo "ABLATION Training Complete!"
echo "=========================================="
echo "Ablations Applied:"
echo "  1. Pure Linear Head (no Transformer decoder)"
echo "  2. Simple L1 Loss (no scale-invariant alignment)"
echo "=========================================="
echo "Checkpoints are saved in:"
echo "  - Stage 1: outputs/pi3_hospital_lowres_ablation/ckpts/"
echo "  - Stage 2: outputs/pi3_hospital_highres_ablation/ckpts/"
echo "=========================================="
echo ""
echo "To compare with baseline:"
echo "  Baseline: outputs/pi3_hospital_lowres/ckpts/"
echo "  Ablation: outputs/pi3_hospital_lowres_ablation/ckpts/"
echo "=========================================="
