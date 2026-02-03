#!/bin/bash

# Resume training from checkpoint_54 and complete in 20 more epochs
# Learning rate starts at 3e-5 and decays to 1e-7 using CosineAnnealingLR

set -e  # Exit on error

# Configuration
NUM_GPUS=8
NUM_MACHINES=1
DATA_CONFIG="tartanair_hospital"
OUTPUT_DIR="outputs/pi3_hospital_lowres"

echo "=========================================="
echo "Resuming Pi3 Training from Checkpoint 54"
echo "=========================================="
echo "Resume checkpoint: outputs/pi3_hospital_lowres/ckpts/checkpoint_54"
echo "Total epochs: 74 (54 completed + 20 more)"
echo "Remaining epochs: 20"
echo "Initial learning rate: 3e-5"
echo "Learning rate schedule: CosineAnnealingLR (3e-5 -> 1e-7 over 20 epochs)"
echo "=========================================="
echo ""

# Verify checkpoint exists
if [ ! -d "outputs/pi3_hospital_lowres/ckpts/checkpoint_54" ]; then
    echo "[ERROR] Checkpoint not found: outputs/pi3_hospital_lowres/ckpts/checkpoint_54"
    echo "Please ensure the checkpoint directory exists."
    exit 1
fi

echo "Checkpoint verified. Starting training..."
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

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training Completed Successfully!"
    echo "=========================================="
    echo "Final checkpoint saved at:"
    echo "  outputs/pi3_hospital_lowres/ckpts/checkpoint_73"
    echo "=========================================="
else
    echo ""
    echo "[ERROR] Training failed. Please check the logs."
    exit 1
fi
