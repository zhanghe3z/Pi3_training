# Wandb Logging and Depth Visualization Changes

## Summary

Modified the training pipeline to enable Weights & Biases (wandb) logging and add periodic depth map visualizations during training.

## Changes Made

### 1. Training Script (`train_hospital_from_scratch.sh`)

Added wandb configuration flags to all three training stages:
- `log.use_wandb=true` - Enable wandb logging
- `log.use_tensorboard=false` - Disable tensorboard (optional)

These flags are now passed to all three training stages:
- Stage 1: Low-resolution training (224x224)
- Stage 2: High-resolution training
- Stage 3: Confidence branch training

### 2. Pi3 Trainer (`trainers/pi3_trainer.py`)

Added depth visualization capabilities:

**New imports:**
- `matplotlib` (with Agg backend for non-interactive plotting)
- `numpy` for array operations
- `wandb` for logging

**New methods:**
- `create_depth_visualization()`: Creates side-by-side comparison plots of:
  - RGB input image
  - Ground truth depth map
  - Predicted depth map
  - All depth maps use the 'turbo' colormap with consistent scaling

- `log_depth_visualizations()`: Logs depth visualizations to wandb
  - Extracts predicted depth from model output (local_points[:,:,:,:,2])
  - Creates visualizations for multiple samples
  - Includes error handling to prevent training interruption
  - Only runs on main process to avoid duplicate logging

**New configuration attributes:**
- `viz_interval`: Controls how often to log visualizations (default: 500 steps)
- `num_viz_samples`: Number of samples to visualize per logging event (default: 2)

### 3. Base Trainer (`trainers/base_trainer_accelerate.py`)

Added visualization hooks:

**Training loop:**
- Calls `log_depth_visualizations()` every `viz_interval` steps during training
- Only executes if the trainer has visualization methods

**Validation loop:**
- Logs depth visualizations at the start of each validation epoch
- Visualizes first batch to track progress over time

### 4. Configuration (`configs/general/default.yaml`)

Added new configuration parameters:
```yaml
viz_interval: 500  # Log depth visualizations every N steps
num_viz_samples: 2  # Number of samples to visualize per batch
```

## Installation

First, install the required packages:
```bash
pip install wandb matplotlib
```

Or reinstall from requirements:
```bash
pip install -r requirements.txt
```

Then, log in to wandb (first time only):
```bash
wandb login
```

## Usage

### Basic Usage

Simply run the modified training script:
```bash
bash train_hospital_from_scratch.sh
```

The script will now:
1. Log all metrics to wandb
2. Upload depth visualizations every 500 steps during training
3. Upload depth visualizations at the start of each validation epoch

### Customizing Visualization Frequency

You can override the visualization interval when launching training:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes 8 \
    scripts/train_pi3.py \
    train=train_pi3_lowres \
    data=tartanair_hospital \
    name=pi3_hospital_lowres \
    log.use_wandb=true \
    viz_interval=1000 \
    num_viz_samples=4
```

### Wandb Project Configuration

Update the wandb configuration in `configs/general/default.yaml`:
```yaml
log:
  wandb_init_conf:
    name: ${name}
    entity: your-entity-name  # Your wandb username or team
    project: ${name}  # Project name
```

## Visualization Output

Each visualization shows:
1. **RGB Image**: Original input image
2. **Ground Truth Depth**: Target depth map from dataset
3. **Predicted Depth**: Model's depth prediction

All three images are displayed side-by-side with consistent color scaling (using min/max from ground truth) for easy comparison.

## Notes

- Visualizations only run on the main process to avoid duplicate uploads
- Error handling is included to prevent training interruption if visualization fails
- The matplotlib backend is set to 'Agg' (non-interactive) to work in headless environments
- Figures are closed after logging to prevent memory leaks

## Based On

The visualization code is based on `/mnt/localssd/Pi3_training/visualize_depth.py`, adapted for integration into the training loop with wandb logging.
