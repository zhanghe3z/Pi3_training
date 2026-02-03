#!/bin/bash

# Pi3 Training Script with Transformer Decoder + Simple Loss
# - Transformer decoder: ENABLED
# - Simple L1 Loss (no scale-invariant alignment)

set -e  # Exit on error

# Configuration
NUM_GPUS=8  # Adjust based on your available GPUs
NUM_MACHINES=1
DATA_CONFIG="tartanair_hospital"

# Output directory
OUTPUT_DIR="outputs/pi3_hospital_decoder_simpleloss"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Pi3 Training: Decoder + Simple Loss"
echo "=========================================="
echo "Configuration:"
echo "  - Transformer decoder: ENABLED"
echo "  - Simple L1 Loss"
echo "=========================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Data Config: ${DATA_CONFIG}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

# Stage 1: Low-Resolution Training (224x224)
echo ""
echo "[Stage 1/2] Starting Low-Resolution Training..."
echo "Resolution: 224x224"
echo "Epochs: 80"
echo ""

accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes ${NUM_GPUS} \
    --num_machines ${NUM_MACHINES} \
    scripts/train_pi3.py \
    train=train_pi3_lowres \
    data=${DATA_CONFIG} \
    model=pi3 \
    loss.train_loss._target_=pi3.models.loss_ablation.Pi3LossAblation \
    loss.test_loss._target_=pi3.models.loss_ablation.Pi3LossAblation \
    name=pi3_hospital_lowres_decoder_simpleloss \
    log.use_wandb=true \
    log.use_tensorboard=false \
    viz_interval=200 \
    num_viz_samples=2

# Check if Stage 1 completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "[Stage 1/2] Low-Resolution Training Completed Successfully!"
    echo ""
else
    echo ""
    echo "[ERROR] Stage 1 failed. Please check the logs."
    exit 1
fi

# Find the latest checkpoint from Stage 1
LOWRES_CKPT=$(find outputs/pi3_hospital_lowres_decoder_simpleloss/ckpts -name "*.pth" | sort -V | tail -n 1)

if [ -z "${LOWRES_CKPT}" ]; then
    echo "[ERROR] No checkpoint found from Stage 1. Cannot proceed to Stage 2."
    exit 1
fi

echo "Found Stage 1 checkpoint: ${LOWRES_CKPT}"
echo ""

# Stage 2: High-Resolution Training
echo "[Stage 2/2] Starting High-Resolution Training..."
echo "Loading checkpoint: ${LOWRES_CKPT}"
echo "Epochs: 40"
echo ""

accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes ${NUM_GPUS} \
    --num_machines ${NUM_MACHINES} \
    scripts/train_pi3.py \
    train=train_pi3_highres \
    data=${DATA_CONFIG} \
    model=pi3 \
    loss.train_loss._target_=pi3.models.loss_ablation.Pi3LossAblation \
    loss.test_loss._target_=pi3.models.loss_ablation.Pi3LossAblation \
    name=pi3_hospital_highres_decoder_simpleloss \
    model.ckpt=${LOWRES_CKPT} \
    log.use_wandb=true \
    log.use_tensorboard=false \
    viz_interval=200 \
    num_viz_samples=4

# Check if Stage 2 completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "[Stage 2/2] High-Resolution Training Completed Successfully!"
    echo ""
else
    echo ""
    echo "[ERROR] Stage 2 failed. Please check the logs."
    exit 1
fi

# Find the latest checkpoint from Stage 2
HIGHRES_CKPT=$(find outputs/pi3_hospital_highres_decoder_simpleloss/ckpts -name "*.pth" | sort -V | tail -n 1)

if [ -z "${HIGHRES_CKPT}" ]; then
    echo "[ERROR] No checkpoint found from Stage 2."
    exit 1
fi

echo "Found Stage 2 checkpoint: ${HIGHRES_CKPT}"
echo ""

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Configuration:"
echo "  - Transformer decoder: ENABLED"
echo "  - Simple L1 Loss"
echo "=========================================="
echo "Checkpoints are saved in:"
echo "  - Stage 1: outputs/pi3_hospital_lowres_decoder_simpleloss/ckpts/"
echo "  - Stage 2: outputs/pi3_hospital_highres_decoder_simpleloss/ckpts/"
echo "=========================================="
