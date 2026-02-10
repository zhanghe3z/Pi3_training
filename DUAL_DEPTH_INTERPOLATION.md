# Dual Depth Interpolation Implementation

## Problem
Previously, the same depth map was used for both:
1. **Variance calculation** - in Lean mapping
2. **GT supervision** - in loss computation

The depth map was resized using `cv2.INTER_NEAREST`, which is appropriate for GT supervision but suboptimal for variance calculation.

## Solution
Implemented a dual depth interpolation system:
1. **`depthmap`** - Uses `INTER_NEAREST` for GT supervision (preserves exact depth values)
2. **`depthmap_linear`** - Uses `INTER_LINEAR` for variance calculation (smoother gradients)

## Changes Made

### 1. `pi3/utils/cropping.py`
Modified three functions to handle both depth maps:
- `rescale_image_depthmap()` - Added `depthmap_linear` parameter, applies `INTER_LINEAR` to it
- `center_crop_image_depthmap()` - Added `depthmap_linear` parameter, crops both depth maps
- `crop_image_depthmap()` - Added `depthmap_linear` parameter, crops both depth maps

**Key change in `rescale_image_depthmap()` (lines 77-84):**
```python
if depthmap is not None:
    depthmap = cv2.resize(depthmap, tuple(output_resolution), fx=scale_final,
                          fy=scale_final, interpolation=cv2.INTER_NEAREST)

# Resize depthmap_linear using linear interpolation for variance calculation
if depthmap_linear is not None:
    depthmap_linear = cv2.resize(depthmap_linear, tuple(output_resolution), fx=scale_final,
                                 fy=scale_final, interpolation=cv2.INTER_LINEAR)
```

### 2. `datasets/base/base_dataset.py`
Modified `_crop_resize_if_necessary()` to:
- Create `depthmap_linear` as a copy of `depthmap`
- Pass both depth maps through all cropping/resizing operations
- Return `depthmap_linear` in the results

**Key changes (lines 127-177):**
- Line 137: `depthmap_linear = depthmap.copy() if depthmap is not None else None`
- Lines 147, 165, 169, 174: Pass `depthmap_linear` to all cropping functions
- Line 176: Append `depthmap_linear` to return values

### 3. `datasets/tartanair_hospital_dataset.py`
Modified `_get_views()` to:
- Extract `depthmap_linear` from the result tuple
- Add `depthmap_linear` to the view dictionary

**Key changes (lines 144-160):**
```python
result = self._crop_resize_if_necessary(
    rgb_image, depthmap, self.intrinsics, resolution, rng=rng, info=impath)

rgb_image = result[0]
depthmap = result[1]
intrinsics = result[2]
# Extract depthmap_linear (last element in result)
depthmap_linear = result[-1] if len(result) > 3 else None

views.append(dict(
    img=rgb_image,
    depthmap=depthmap,
    depthmap_linear=depthmap_linear,  # NEW
    ...
))
```

### 4. `pi3/models/loss_ablation.py`
Modified `PointLoss.forward()` to:
- Extract both `depthmap` and `depthmap_linear` from `gt_raw`
- Use `gt_depth_linear` for variance calculation via Lean mapping
- Use `gt_depth` (nearest neighbor) for GT supervision in loss computation

**Key changes (lines 790-832):**
```python
# Use nearest-neighbor interpolated depth for GT supervision
gt_depth = torch.stack([view['depthmap'] for view in gt_raw], dim=1)

# Use linear-interpolated depth for variance calculation
gt_depth_linear = torch.stack([view['depthmap_linear'] for view in gt_raw], dim=1)

# ... normalize both depths ...

# Use linear interpolated depth for variance calculation
gt_depth_linear_flat = gt_depth_linear.reshape(B*N, 1, H, W)

# Compute variance using Lean mapping with linear interpolated depth
m1, m2, variance, count = lean_depth_moments_and_variance(
    depth=gt_depth_linear_flat,  # LINEAR for variance
    ...
)

# Later in loss computation (lines 855, 884)
depth_diff = torch.abs(pred_depth - gt_depth)  # NEAREST for supervision
```

## Benefits

1. **Better variance estimation**: Linear interpolation provides smoother depth transitions, leading to more accurate variance estimation in Lean mapping
2. **Accurate GT supervision**: Nearest neighbor interpolation preserves exact depth values for supervision
3. **Backward compatible**: Falls back gracefully if `depthmap_linear` is not available

## Testing
To verify the changes work correctly:
```bash
# Run training with the modified code
bash train_hospital_local_points_gt_pred.sh
```

Expected behavior:
- Training should run without errors
- Variance values should be smoother (check wandb logs: `variance_mean`, `variance_min`, `variance_max`)
- Depth loss should maintain accuracy (check `depth_loss` in wandb)
