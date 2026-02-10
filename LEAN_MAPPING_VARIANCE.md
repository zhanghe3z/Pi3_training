# Lean Mapping Variance Loss Implementation

## Summary

Created a new variance calculation method using Lean Mapping / VSM-style kernel-based local moments as an alternative to the mip-NeRF variance approach.

## Changes Made

### 1. New Module: `pi3/models/lean_variance_loss.py`

Implements Lean Mapping variance calculation:

**Key Features:**
- **Local Moments**: Computes M1 = E[z], M2 = E[z^2], Var = M2 - M1^2
- **Kernel Types**: Supports Gaussian and box kernels
- **Configurable Window Size**: Default 7x7 (vs 5x5 in mip-NeRF)
- **Prior Variance**: For low-support regions (< min_valid_count samples)
  - var_prior = (prior_rel * mean)^2 + prior_abs^2
  - Default prior_rel = 0.1 (10% of local mean depth)
- **Padding Modes**: reflect, replicate, or zeros
- **Optional Winsorization**: Clamp depth values to prevent outlier explosion

**Main Functions:**
- `lean_depth_moments_and_variance()`: Full computation with all statistics
- `lean_variance_from_depth()`: Simplified interface returning only variance
- `weighted_l1_from_variance()`: L1 loss with inverse-std weighting
- `laplace_nll_from_variance()`: Laplace NLL loss

### 2. Loss Classes: `pi3/models/loss_ablation.py`

Added two new classes:

#### `PointLossLeanMapping`
- Point loss using Lean Mapping variance
- Separates depth loss (variance-weighted) from XY loss (unweighted)
- No scale alignment
- Logs detailed statistics (variance, std, weights, count)

#### `Pi3LossLeanMapping`
- Full training loss combining point and camera losses
- GT-only normalization (normalize_gt=true, normalize_pred=false)
- Camera loss with 0.1 weight

**Configuration Parameters:**
```python
loss_type='weighted_l1'      # or 'laplace_nll'
kernel_size=7                # local window size
kernel='gaussian'            # or 'box'
gaussian_sigma=None          # default: kernel_size/6
min_valid_count=8            # prior threshold
prior_rel=0.1                # relative prior (10%)
prior_abs=0.0                # absolute prior
std_min=0.1                  # min std to prevent weight explosion
```

### 3. Training Script: `train_hospital_local_points_gt_lean_mapping.sh`

New training script with Lean Mapping configuration:

**Stage 1 (Low-Res 224x224, 80 epochs):**
- Output: `outputs/pi3_hospital_lowres_lean/`
- WandB run: `pi3_hospital_lowres_lean`

**Stage 2 (High-Res, 40 epochs):**
- Output: `outputs/pi3_hospital_highres_lean/`
- WandB run: `pi3_hospital_highres_lean`

**Configuration:**
- Exp activation for depth
- Gaussian 7x7 kernel
- Prior: 10% of local mean
- GT-only normalization
- No scale alignment

## Key Differences: Lean Mapping vs mip-NeRF Variance

| Aspect | mip-NeRF | Lean Mapping |
|--------|----------|--------------|
| **Method** | Quantile-based (Q95-Q05) | Moment-based (E[z²] - E[z]²) |
| **Window** | 5x5 fixed | Configurable (default 7x7) |
| **Kernel** | Unfold + sort | Gaussian convolution |
| **Ray Geometry** | Uses camera intrinsics, ray dirs | Direct depth variance |
| **Prior** | Depth + footprint dependent | Simple relative (% of mean) |
| **Edge Detection** | Quantile range in t-space | Variance in z-space |
| **Computational Cost** | Higher (sorting) | Lower (convolution) |

## Advantages of Lean Mapping

1. **Simpler**: Direct depth variance without ray-space conversion
2. **Flexible Kernels**: Gaussian smoothing reduces grid artifacts
3. **Larger Windows**: 7x7 captures more context than 5x5
4. **No Intrinsics Required**: Works directly with depth values
5. **Differentiable**: Could support learnable kernels in future

## Usage

Run the training:
```bash
bash train_hospital_local_points_gt_lean_mapping.sh
```

Monitor on WandB:
- Check `variance_mean`, `std_mean`, `weights_mean` for sanity
- Compare `depth_loss` with mip-NeRF version
- Look for `m1_mean` (local mean depth) and `count_mean` (valid samples)

## Next Steps

Potential experiments:
1. **Kernel Size**: Try 5x5 vs 9x9
2. **Prior Strength**: Vary prior_rel (0.05, 0.1, 0.15)
3. **Kernel Type**: Compare Gaussian vs Box
4. **Loss Type**: weighted_l1 vs laplace_nll
5. **Larger std_min**: If weights still explode, increase from 0.1 to 0.2
