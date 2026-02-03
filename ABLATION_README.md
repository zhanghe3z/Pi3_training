# Pi3 Ablation Study

## Overview

This ablation study investigates the contribution of two key components in the Pi3 model:

1. **Transformer Decoder Head** vs **Pure Linear Head**
2. **Scale-Invariant L1 Loss** vs **Simple L1 Loss**

## Files Created

### Model Files
- `pi3/models/pi3_training_ablation.py` - Ablation model with pure linear head
- `pi3/models/loss_ablation.py` - Simple L1 loss without scale alignment

### Configuration
- `configs/model/pi3_ablation.yaml` - Model config for ablation study

### Training Script
- `train_hospital_ablation.sh` - Training script for ablation experiments

## Key Differences

### 1. Pure Linear Head (vs Transformer Decoder)

**Original (pi3_training.py:111-118):**
```python
self.point_decoder = TransformerDecoder(
    in_dim=2*self.dec_embed_dim,
    dec_embed_dim=1024,
    dec_num_heads=16,
    out_dim=1024,
    rope=self.rope,
)
self.point_head = LinearPts3d(...)
```

**Ablation (pi3_training_ablation.py:125-127):**
```python
self.point_projection = nn.Linear(2*self.dec_embed_dim, 1024)
self.point_head = LinearPts3d(...)
```

The ablation removes the multi-layer Transformer decoder and replaces it with a single linear projection layer.

### 2. Simple L1 Loss (vs Scale-Invariant)

**Original (loss.py:126-137):**
```python
# Compute optimal scale alignment
S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)

# Apply scale to align predictions
aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

# Compute L1 loss on aligned points
local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks], ...)
```

**Ablation (loss_ablation.py:83-89):**
```python
# No scale alignment - use dummy scale of 1.0
S_opt_local = torch.ones(B, device=pred_local_pts.device)

# Direct L1 loss without alignment
local_pts_loss = self.criteria_local(pred_local_pts[valid_masks], ...)
```

The ablation removes the scale-invariant alignment step and directly computes L1 loss.

## Running the Ablation Study

### Prerequisites
Same as the baseline Pi3 training:
- 8 GPUs (configurable)
- TartanAir Hospital dataset
- VGGT checkpoint (optional, set `load_vggt: false` in config)

### Execute Training
```bash
./train_hospital_ablation.sh
```

### Training Stages
1. **Stage 1**: Low-res (224x224) training for 80 epochs
2. **Stage 2**: High-res training for 40 epochs

## Output Locations

### Ablation Results
- Stage 1: `outputs/pi3_hospital_lowres_ablation/ckpts/`
- Stage 2: `outputs/pi3_hospital_highres_ablation/ckpts/`

### Baseline Results (for comparison)
- Stage 1: `outputs/pi3_hospital_lowres/ckpts/`
- Stage 2: `outputs/pi3_hospital_highres/ckpts/`

## Expected Impact

### Transformer Decoder Ablation
Removing the Transformer decoder may result in:
- **Reduced capacity** for modeling complex 3D point relationships
- **Faster inference** due to fewer parameters and computations
- **Lower accuracy** on challenging scenes with occlusions or ambiguities

### Scale-Invariant Loss Ablation
Removing scale-invariant alignment may result in:
- **Scale ambiguity issues** in depth prediction
- **Difficulty handling scenes at different scales**
- **Potential training instability** due to absolute scale requirements

## Metrics to Compare

When comparing baseline vs ablation, monitor:
1. **Point reconstruction error** (L1 distance)
2. **Normal estimation accuracy**
3. **Camera pose accuracy** (rotation/translation errors)
4. **Training stability** (loss curves)
5. **Inference speed**

## Notes

- The camera decoder still uses Transformer (only point decoder is ablated)
- Normal loss is kept the same in both versions
- Confidence training is disabled in ablation (not implemented)
- Both models use the same encoder and main decoder architecture
