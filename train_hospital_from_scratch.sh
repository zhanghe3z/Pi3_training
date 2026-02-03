#!/bin/bash

# Pi3 Training Script for TartanAir Hospital Dataset (From Scratch)
# This script trains the Pi3 model from scratch on the hospital dataset

set -e  # Exit on error

# Configuration
NUM_GPUS=8  # Adjust based on your available GPUs
NUM_MACHINES=1
DATA_CONFIG="tartanair_hospital"

# Output directory
OUTPUT_DIR="outputs/pi3_hospital"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Pi3 Training on TartanAir Hospital Dataset"
echo "=========================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Data Config: ${DATA_CONFIG}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

# Stage 1: Low-Resolution Training (224x224)
echo ""
echo "[Stage 1/3] Starting Low-Resolution Training..."
echo "Resolution: 224x224"
echo "Epochs: 80"
echo "Depth Visualization: Every 10 steps (starting from step 0)"
echo "Number of Samples: 4"
echo ""

accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes ${NUM_GPUS} \
    --num_machines ${NUM_MACHINES} \
    scripts/train_pi3.py \
    train=train_pi3_lowres \
    data=${DATA_CONFIG} \
    name=pi3_hospital_lowres \
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
LOWRES_CKPT=$(find outputs/pi3_hospital_lowres/ckpts -name "*.pth" | sort -V | tail -n 1)

if [ -z "${LOWRES_CKPT}" ]; then
    echo "[ERROR] No checkpoint found from Stage 1. Cannot proceed to Stage 2."
    exit 1
fi

echo "Found Stage 1 checkpoint: ${LOWRES_CKPT}"
echo ""

# Stage 2: High-Resolution Training
echo "[Stage 2/3] Starting High-Resolution Training..."
echo "Loading checkpoint: ${LOWRES_CKPT}"
echo "Epochs: 40"
echo "Depth Visualization: Every 10 steps"
echo "Number of Samples: 4"
echo ""

accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes ${NUM_GPUS} \
    --num_machines ${NUM_MACHINES} \
    scripts/train_pi3.py \
    train=train_pi3_highres \
    data=${DATA_CONFIG} \
    name=pi3_hospital_highres \
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
HIGHRES_CKPT=$(find outputs/pi3_hospital_highres/ckpts -name "*.pth" | sort -V | tail -n 1)

if [ -z "${HIGHRES_CKPT}" ]; then
    echo "[ERROR] No checkpoint found from Stage 2. Cannot proceed to Stage 3."
    exit 1
fi

echo "Found Stage 2 checkpoint: ${HIGHRES_CKPT}"
echo ""

# Stage 3: Confidence Branch Training (Optional)
echo "[Stage 3/3] Starting Confidence Branch Training..."
echo "Loading checkpoint: ${HIGHRES_CKPT}"
echo ""
echo "NOTE: This stage requires Segformer checkpoint at ckpts/segformer.b0.512x512.ade.160k.pth"
echo "If you haven't downloaded it, the training will fail."
echo ""
read -p "Continue with Stage 3? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if Segformer checkpoint exists
    if [ ! -f "ckpts/segformer.b0.512x512.ade.160k.pth" ]; then
        echo "[WARNING] Segformer checkpoint not found at ckpts/segformer.b0.512x512.ade.160k.pth"
        echo "Please download it from: https://github.com/NVlabs/SegFormer"
        echo "Skipping Stage 3..."
    else
        accelerate launch --config_file configs/accelerate/ddp.yaml \
            --num_processes ${NUM_GPUS} \
            --num_machines ${NUM_MACHINES} \
            scripts/train_pi3.py \
            train=train_pi3_conf \
            data=${DATA_CONFIG} \
            name=pi3_hospital_conf \
            model.ckpt=${HIGHRES_CKPT} \
            log.use_wandb=true \
            log.use_tensorboard=false \
            viz_interval=200 \
            num_viz_samples=4

        if [ $? -eq 0 ]; then
            echo ""
            echo "[Stage 3/3] Confidence Branch Training Completed Successfully!"
            echo ""
        else
            echo ""
            echo "[ERROR] Stage 3 failed. Please check the logs."
            exit 1
        fi
    fi
else
    echo "Skipping Stage 3."
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoints are saved in:"
echo "  - Stage 1: outputs/pi3_hospital_lowres/ckpts/"
echo "  - Stage 2: outputs/pi3_hospital_highres/ckpts/"
if [[ $REPLY =~ ^[Yy]$ ]] && [ -f "ckpts/segformer.b0.512x512.ade.160k.pth" ]; then
    echo "  - Stage 3: outputs/pi3_hospital_conf/ckpts/"
fi
echo "=========================================="
