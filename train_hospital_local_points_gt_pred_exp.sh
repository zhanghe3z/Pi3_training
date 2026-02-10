#!/bin/bash

# Pi3 Training Script - Experimental Version
# - Softplus activation for depth output
# - mip-NeRF inspired variance weighting for depth loss
# - GT-only normalization
# - No scale alignment

set -e  # Exit on error

# Configuration
NUM_GPUS=8  # Adjust based on your available GPUs
NUM_MACHINES=1
DATA_CONFIG="tartanair_hospital"

# Output directory
OUTPUT_DIR="outputs/pi3_hospital_local_points_exp"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Pi3 Training: Experimental Configuration"
echo "=========================================="
echo "Configuration:"
echo "  - Exp activation: ENABLED"
echo "  - mip-NeRF variance weighting: ENABLED"
echo "  - Normalize predicted points: DISABLED"
echo "  - Normalize GT points: ENABLED (using global_points scale)"
echo "  - Scale alignment: DISABLED"
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
    train_dataset.TarTanAir.data_root=/mnt/localssd/data \
    test_dataset.TarTanAir.data_root=/mnt/localssd/data \
    model=pi3_local_pts_gt_pred \
    model.depth_activation=exp \
    loss.train_loss._target_=pi3.models.loss_ablation.Pi3LossMipNeRFVariance \
    ~loss.train_loss.use_local_alignment_normalize \
    +loss.train_loss.normalize_pred=false \
    +loss.train_loss.normalize_gt=true \
    +loss.train_loss.loss_type=weighted_l1 \
    +loss.train_loss.window=5 \
    +loss.train_loss.lambda_prior=0.5 \
    +loss.train_loss.alpha_clamp=0.3 \
    loss.test_loss._target_=pi3.models.loss_ablation.Pi3LossMipNeRFVariance \
    ~loss.test_loss.use_local_alignment_normalize \
    +loss.test_loss.normalize_pred=false \
    +loss.test_loss.normalize_gt=true \
    +loss.test_loss.loss_type=weighted_l1 \
    +loss.test_loss.window=5 \
    +loss.test_loss.lambda_prior=0.5 \
    +loss.test_loss.alpha_clamp=0.3 \
    name=pi3_hospital_lowres_exp \
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
LOWRES_CKPT=$(find outputs/pi3_hospital_lowres_exp/ckpts -name "*.pth" | sort -V | tail -n 1)

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
    train_dataset.TarTanAir.data_root=/mnt/localssd/data \
    test_dataset.TarTanAir.data_root=/mnt/localssd/data \
    model=pi3_local_pts_gt_pred \
    model.depth_activation=exp \
    loss.train_loss._target_=pi3.models.loss_ablation.Pi3LossMipNeRFVariance \
    ~loss.train_loss.use_local_alignment_normalize \
    +loss.train_loss.normalize_pred=false \
    +loss.train_loss.normalize_gt=true \
    +loss.train_loss.loss_type=weighted_l1 \
    +loss.train_loss.window=5 \
    +loss.train_loss.lambda_prior=0.5 \
    +loss.train_loss.alpha_clamp=0.3 \
    loss.test_loss._target_=pi3.models.loss_ablation.Pi3LossMipNeRFVariance \
    ~loss.test_loss.use_local_alignment_normalize \
    +loss.test_loss.normalize_pred=false \
    +loss.test_loss.normalize_gt=true \
    +loss.test_loss.loss_type=weighted_l1 \
    +loss.test_loss.window=5 \
    +loss.test_loss.lambda_prior=0.5 \
    +loss.test_loss.alpha_clamp=0.3 \
    name=pi3_hospital_highres_exp \
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
HIGHRES_CKPT=$(find outputs/pi3_hospital_highres_exp/ckpts -name "*.pth" | sort -V | tail -n 1)

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
echo "  - Exp activation: ENABLED"
echo "  - mip-NeRF variance weighting: ENABLED"
echo "  - Normalize predicted points: DISABLED"
echo "  - Normalize GT points: ENABLED (using global_points scale)"
echo "  - Scale alignment: DISABLED"
echo "=========================================="
echo "Checkpoints are saved in:"
echo "  - Stage 1: outputs/pi3_hospital_lowres_exp/ckpts/"
echo "  - Stage 2: outputs/pi3_hospital_highres_exp/ckpts/"
echo "=========================================="
