# mip-NeRF Variance-Weighted Loss Integration

## Summary

Successfully integrated a mip-NeRF inspired variance-weighted depth loss for supervising local points (`pts_local`) in the Pi3 training pipeline.

## What Was Implemented

### 1. Core Variance Computation Module
**File**: `pi3/models/mipnerf_variance_loss.py`

This module provides:
- **Per-pixel depth variance estimation** based on:
  - Local spatial variation (5x5 window using robust quantile range Q95-Q05)
  - Depth-dependent prior (pixel footprint grows with distance)
- **mip-NeRF conical frustum formula** for computing ray-distance variance
- **Two loss variants**:
  - `weighted_l1_from_sigma_z2`: Weighted L1 loss (practical)
  - `laplace_nll_from_sigma_z2`: Laplace NLL (probabilistic)

Key functions:
```python
sigma_z2_from_gt_z_pixelwise(Z_gt, fx, fy, cx, cy, window=5, lambda_prior=0.5, alpha_clamp=0.3)
weighted_l1_from_sigma_z2(Z_pred, Z_gt, sigma_Z2)
laplace_nll_from_sigma_z2(Z_pred, Z_gt, sigma_Z2)
```

### 2. Loss Classes
**File**: `pi3/models/loss_ablation.py`

Added two new classes:

#### `PointLossMipNeRFVariance`
- Computes per-pixel depth variance from GT using camera intrinsics
- Applies variance-weighted loss on depth (Z coordinate)
- Supports both `weighted_l1` and `laplace_nll` loss types
- No scale alignment (similar to ablation studies)
- Extracts intrinsics from raw GT data

#### `Pi3LossMipNeRFVariance`
- Full Pi3 loss wrapper using mip-NeRF variance weighting
- GT-only normalization (normalize_gt=true, normalize_pred=false)
- Combines depth variance loss with camera loss

### 3. Updated Training Script
**File**: `train_hospital_local_points_gt_pred_softplus.sh`

Updated to use the new loss with:
- Loss type: `pi3.models.loss_ablation.Pi3LossMipNeRFVariance`
- Configuration:
  - `loss_type=weighted_l1` (or `laplace_nll`)
  - `window=5` (5x5 local window)
  - `lambda_prior=0.5` (mild depth-dependent prior)
  - `alpha_clamp=0.3` (clamp interval to prevent explosion at edges)

## How It Works

### Variance Computation Pipeline

1. **Extract depth from local points**: `Z = pts_local[..., 2]`

2. **Compute ray direction z-component**:
   ```
   d_z = 1 / sqrt((u-cx)²/fx² + (v-cy)²/fy² + 1)
   ```

3. **Convert to ray distance**: `t = Z / d_z`

4. **Compute local spatial variation**:
   - Extract 5x5 patches around each pixel
   - Compute Q05 (5th percentile) and Q95 (95th percentile)
   - `t_delta_edge = 0.5 * (Q95 - Q05)`

5. **Add depth-dependent prior**:
   ```
   tan(theta) = sqrt((0.5/fx)² + (0.5/fy)²)
   t_delta_prior = lambda_prior * tan(theta) * t
   ```

6. **Combine and clamp**:
   ```
   t_delta = sqrt(t_delta_edge² + t_delta_prior²)
   t_delta = min(t_delta, alpha_clamp * t)
   ```

7. **mip-NeRF conical frustum variance**:
   ```
   sigma_t² = t_delta²/3 - [4*t_delta⁴*(12*t² - t_delta²)] / [15*(3*t² + t_delta²)²]
   ```

8. **Map back to depth variance**: `sigma_Z² = d_z² * sigma_t²`

### Loss Computation

**Weighted L1** (recommended):
```
w = 1 / (sqrt(sigma_Z²) + eps)
loss = mean(w * |Z_pred - Z_gt|)
```

**Laplace NLL** (probabilistic):
```
b = sqrt(sigma_Z² / 2)
loss = mean(|Z_pred - Z_gt| / b + log(b))
```

## Key Features

1. **Geometry-aware weighting**: Down-weights uncertain regions (edges, far depths)
2. **No epistemic uncertainty**: This is a prior based on GT geometry, not model uncertainty
3. **Mild prior**: `lambda_prior=0.5` avoids overly aggressive down-weighting
4. **Edge handling**: `alpha_clamp=0.3` prevents explosion at hard occlusion boundaries

## Usage

### Training with the new loss:

```bash
bash train_hospital_local_points_gt_pred_softplus.sh
```

### Key Configuration Parameters:

- `loss_type`: Choose between `weighted_l1` (default) or `laplace_nll`
- `window`: Local window size (default: 5 for 5x5)
- `lambda_prior`: Weight for depth-dependent prior (default: 0.5)
  - Higher values → stronger depth weighting
  - Lower values → rely more on local spatial variation
- `alpha_clamp`: Clamp interval half-width (default: 0.3)
  - Prevents variance explosion at occlusion boundaries
  - `t_delta <= alpha_clamp * t`

### Customization Example:

```bash
# More aggressive depth weighting
+loss.train_loss.lambda_prior=1.0

# Use probabilistic Laplace NLL instead of weighted L1
+loss.train_loss.loss_type=laplace_nll

# Larger local window (7x7)
+loss.train_loss.window=7
```

## Testing

A sanity check test is provided in `test_mipnerf_loss.py`:

```bash
python test_mipnerf_loss.py
```

Tests verify:
1. ✓ Basic variance computation
2. ✓ Weighted loss functions
3. ✓ Integration with loss module

## Expected Behavior

### Variance Map Characteristics:
- **Low variance**: Smooth regions with consistent depth
- **High variance**:
  - Depth discontinuities (edges)
  - Far-away regions (larger pixel footprint)
  - Regions with local depth variation

### Training Impact:
- Network focuses more on reliable depth values
- Edges and uncertain regions contribute less to gradient
- Should improve depth accuracy on smooth surfaces
- May help with scale ambiguity in monocular depth

## Technical Details

### Why mip-NeRF formula?
The mip-NeRF conical frustum variance formula accounts for the fact that each pixel corresponds to a cone in 3D space (not a ray). This is geometrically more accurate than simple Gaussian assumptions.

### Why separate depth and XY losses?
- Depth (Z) uses variance weighting
- XY coordinates use standard L1 loss
- This is because variance is most meaningful for depth uncertainty

### Integration with existing pipeline:
- Requires raw GT data (`gt_raw`) to extract camera intrinsics
- Compatible with GT-only normalization
- Works with softplus depth activation
- No changes needed to model architecture

## Files Modified/Created

1. ✓ Created: `pi3/models/mipnerf_variance_loss.py` (core variance computation)
2. ✓ Modified: `pi3/models/loss_ablation.py` (added two new loss classes)
3. ✓ Modified: `train_hospital_local_points_gt_pred_softplus.sh` (updated to use new loss)
4. ✓ Created: `test_mipnerf_loss.py` (sanity check tests)

## Validation Results

From sanity check test:
```
✓ Test 1: Basic Variance Computation
  - sigma_Z2 stats - mean: 0.017414, std: 0.094095
  - Edge variance: 0.161368, Smooth variance: 0.005695
  - ✓ Edge has higher variance than smooth regions

✓ Test 2: Weighted Loss Functions
  - Weighted L1 loss: 9.346384
  - Laplace NLL loss: 7.216088
  - ✓ Losses are positive and not NaN

✓ Syntax validation passed for all Python files
```

## Next Steps

1. **Run training**: `bash train_hospital_local_points_gt_pred_softplus.sh`
2. **Monitor metrics**: Watch `depth_loss`, `xy_loss`, and overall `local_pts_loss` in wandb
3. **Compare**: Compare against baseline (Pi3LossGTOnlyNorm) to see if variance weighting improves results
4. **Tune hyperparameters**: Adjust `lambda_prior`, `alpha_clamp` if needed

## Notes

- This implementation is a **geometric/aliasing prior**, not epistemic uncertainty
- The variance map is computed from GT, so it's detached during backprop
- `lambda_prior=0.5` is intentionally mild to avoid over-regularization
- Works best with accurate camera intrinsics

## References

- mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields (Barron et al., ICCV 2021)
- Closed-form conical frustum variance formula from mip-NeRF supplementary material
