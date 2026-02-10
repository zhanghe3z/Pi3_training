# Loss Comparison: Original vs mip-NeRF Variance

## Quick Reference

### Original Loss (Pi3LossGTOnlyNorm)

**Location**: `pi3.models.loss_ablation.Pi3LossGTOnlyNorm`

**Depth weighting**:
```python
weights_ = gt_local_pts[..., 2]  # Use GT depth directly
weights_ = weights_.clamp_min(0.1 * mean_depth)
weights_ = 1 / (weights_ + 1e-6)  # Inverse depth weighting
```

**Loss computation**:
```python
local_pts_loss = L1(pred, gt) * weights_
```

**Characteristics**:
- Simple inverse-depth weighting (near points weighted more)
- Uniform weighting across spatial extent at same depth
- No edge/aliasing awareness

---

### New Loss (Pi3LossMipNeRFVariance)

**Location**: `pi3.models.loss_ablation.Pi3LossMipNeRFVariance`

**Depth variance computation**:
```python
# Extract intrinsics
fx, fy, cx, cy = extract_from_gt_raw(...)

# Compute variance from GT geometry
sigma_Z2 = sigma_z2_from_gt_z_pixelwise(
    Z_gt, fx, fy, cx, cy,
    window=5,          # Local spatial window
    lambda_prior=0.5,  # Depth-dependent weight
    alpha_clamp=0.3    # Clamp at edges
)
```

**Loss computation**:
```python
# Weighted L1 (default)
weights = 1 / (sqrt(sigma_Z2) + eps)
depth_loss = mean(weights * |Z_pred - Z_gt|)
xy_loss = mean(|XY_pred - XY_gt|)

# OR Laplace NLL (probabilistic)
b = sqrt(sigma_Z2 / 2)
depth_loss = mean(|Z_pred - Z_gt| / b + log(b))
```

**Characteristics**:
- **Geometry-aware**: Considers local spatial variation
- **Edge-sensitive**: Higher variance at depth discontinuities
- **Physics-based**: Uses mip-NeRF conical frustum formula
- **Depth-dependent**: Pixel footprint grows with distance

---

## Key Differences

| Aspect | Original | mip-NeRF Variance |
|--------|----------|-------------------|
| **Weighting basis** | Inverse depth | Depth variance (geometry) |
| **Spatial awareness** | None | 5x5 local window |
| **Edge handling** | Same as smooth | Higher uncertainty at edges |
| **Intrinsics** | Not used | Required (from GT) |
| **XY vs Z** | Same weighting | Separate (Z weighted, XY not) |
| **Depth prior** | Linear (1/Z) | Mild (lambda=0.5) + spatial |

---

## When to Use Each

### Use Original (Pi3LossGTOnlyNorm) when:
- ✓ Simple inverse-depth weighting is sufficient
- ✓ Don't have accurate camera intrinsics
- ✓ Want faster training (no variance computation)
- ✓ Dataset has mostly smooth surfaces

### Use New (Pi3LossMipNeRFVariance) when:
- ✓ Dataset has significant depth discontinuities/edges
- ✓ Have accurate camera intrinsics
- ✓ Want geometry-aware weighting
- ✓ Monocular depth estimation with scale ambiguity
- ✓ Aliasing/sub-pixel mixture is a concern

---

## Configuration Comparison

### Original:
```bash
loss.train_loss._target_=pi3.models.loss_ablation.Pi3LossGTOnlyNorm
+loss.train_loss.normalize_pred=false
+loss.train_loss.normalize_gt=true
+loss.train_loss.use_scale_align=false
```

### New (Weighted L1):
```bash
loss.train_loss._target_=pi3.models.loss_ablation.Pi3LossMipNeRFVariance
+loss.train_loss.normalize_pred=false
+loss.train_loss.normalize_gt=true
+loss.train_loss.loss_type=weighted_l1
+loss.train_loss.window=5
+loss.train_loss.lambda_prior=0.5
+loss.train_loss.alpha_clamp=0.3
```

### New (Laplace NLL - Probabilistic):
```bash
loss.train_loss._target_=pi3.models.loss_ablation.Pi3LossMipNeRFVariance
+loss.train_loss.normalize_pred=false
+loss.train_loss.normalize_gt=true
+loss.train_loss.loss_type=laplace_nll  # <-- Change this
+loss.train_loss.window=5
+loss.train_loss.lambda_prior=0.5
+loss.train_loss.alpha_clamp=0.3
```

---

## Hyperparameter Tuning Guide

### `lambda_prior` (depth-dependent weight)
- **Default**: 0.5 (mild)
- **Range**: 0.0 - 2.0
- **Lower (0.1-0.3)**: Rely mostly on local spatial variation
- **Higher (1.0-2.0)**: Stronger depth-dependent weighting
- **Effect**: Controls balance between spatial and depth priors

### `alpha_clamp` (edge clamp)
- **Default**: 0.3 (conservative)
- **Range**: 0.1 - 0.5
- **Lower (0.1-0.2)**: Tighter clamp, less variance at edges
- **Higher (0.4-0.5)**: Allow more variance at edges
- **Effect**: Prevents variance explosion at occlusion boundaries

### `window` (local window size)
- **Default**: 5 (5x5 window)
- **Options**: 3, 5, 7
- **Smaller (3)**: More local, faster computation
- **Larger (7)**: Smoother variance, more context
- **Effect**: Spatial extent of variance estimation

### `loss_type`
- **weighted_l1**: Practical, simpler gradient behavior
- **laplace_nll**: Probabilistic, better calibrated uncertainty
- **Recommendation**: Start with weighted_l1

---

## Expected Training Behavior

### Original Loss:
```
local_pts_loss: 0.XXX
  - Uniform contribution across each depth layer
  - May struggle with edges
```

### New Loss:
```
depth_loss: 0.XXX    # Variance-weighted depth
xy_loss: 0.XXX       # Standard XY loss
local_pts_loss: 0.XXX  # Combined
  - Edges contribute less (higher variance)
  - Smooth regions drive optimization
  - Better depth accuracy on reliable areas
```

---

## Migration Guide

**From**: `train_hospital_decoder_simpleloss.sh`
**To**: `train_hospital_local_points_gt_pred_softplus.sh`

Changes needed:
1. Update loss class: `Pi3LossGTOnlyNorm` → `Pi3LossMipNeRFVariance`
2. Add loss parameters: `loss_type`, `window`, `lambda_prior`, `alpha_clamp`
3. Update output directory names
4. No model changes needed!

**Backward compatible**: Can switch back to original loss anytime

---

## Performance Notes

### Computational Cost:
- **Original**: ~0 extra cost
- **New**:
  - Variance computation: ~5-10% overhead (done once per batch, @no_grad)
  - Negligible impact on training speed
  - All operations are GPU-accelerated (unfold, sort)

### Memory:
- Extra tensors: `sigma_Z2`, `sigma_t2`, `t_mu`, `t_delta` (all same size as depth map)
- Temporary: 5x5 patches for unfold operation
- Impact: Minimal (~1% memory increase)

---

## Debugging Tips

### Check variance map statistics:
Add to your code:
```python
print(f"sigma_Z2 - mean: {sigma_Z2.mean():.6f}, std: {sigma_Z2.std():.6f}")
print(f"Edge variance: {sigma_Z2[:,:,edge_region].mean():.6f}")
print(f"Smooth variance: {sigma_Z2[:,:,smooth_region].mean():.6f}")
```

Expected behavior:
- Edge variance > Smooth variance
- Far-depth variance > Near-depth variance
- sigma_Z2 always >= 0

### Visualize variance:
```python
import matplotlib.pyplot as plt
plt.imshow(sigma_Z2[0, 0].cpu().numpy())
plt.colorbar()
plt.title('Depth Variance Map')
plt.savefig('variance_map.png')
```

### Monitor loss components:
Track these in wandb:
- `depth_loss` - Should decrease over training
- `xy_loss` - Should decrease over training
- Ratio `depth_loss / xy_loss` - Should stabilize

---

## FAQ

**Q: Can I use this with other depth activations (exp, sigmoid)?**
A: Yes! Works with any depth activation. Just ensure positive depth values.

**Q: What if I don't have accurate intrinsics?**
A: Use approximate intrinsics or fall back to original loss. Variance weighting benefits from accurate intrinsics.

**Q: Can I visualize the variance map during training?**
A: Yes! Add visualization code in your training loop (see debugging tips above).

**Q: Does this help with scale ambiguity in monocular depth?**
A: Yes! By down-weighting uncertain regions, it can help the network learn more consistent scales.

**Q: Should I use weighted_l1 or laplace_nll?**
A: Start with weighted_l1. Try laplace_nll if you want better calibrated uncertainty estimates.
