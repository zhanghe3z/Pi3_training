#!/bin/bash

# Simple Pi3 Training Script for TartanAir Hospital Dataset
# This script runs a single stage of training (low-resolution)
# Ideal for quick testing or limited resources

set -e  # Exit on error

# Configuration
NUM_GPUS=${1:-1}  # Default to 1 GPU if not specified
DATA_CONFIG="tartanair_hospital"
EXPERIMENT_NAME="pi3_hospital_lowres"

echo "=========================================="
echo "Pi3 Low-Resolution Training"
echo "Dataset: TartanAir Hospital"
echo "=========================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Output: outputs/${EXPERIMENT_NAME}"
echo "=========================================="
echo ""

# Run training
accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes ${NUM_GPUS} \
    --num_machines 1 \
    scripts/train_pi3.py \
    train=train_pi3_lowres \
    data=${DATA_CONFIG} \
    name=${EXPERIMENT_NAME}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Checkpoints saved in: outputs/${EXPERIMENT_NAME}/ckpts/"
echo "=========================================="
