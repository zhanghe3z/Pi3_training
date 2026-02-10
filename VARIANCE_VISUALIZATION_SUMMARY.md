# Hospital Depth Variance Visualization - Summary

## Created Files

1. **`visualize_depth_variance.py`** - Main visualization script
2. **`test_visualize_depth_variance.py`** - Test script with synthetic data
3. **`VISUALIZE_DEPTH_VARIANCE_README.md`** - Comprehensive documentation

## Quick Start

### 1. List Available Sequences
```bash
python visualize_depth_variance.py --list_sequences
```

**Output:**
```
Available sequences in hospital dataset:
  Easy     / P000       -  563 frames
Total: 1 sequences
```

### 2. Visualize a Frame
```bash
python visualize_depth_variance.py --frame 0
```

This will:
- Load depth map from `/mnt/localssd/data/Easy/P000/depth_left/000000_left_depth.npy`
- Load RGB image from `/mnt/localssd/data/Easy/P000/image_left/000000_left.png`
- Compute local variance using Lean Mapping (matching training config)
- Generate an 8-panel visualization showing:
  1. RGB Image
  2. Ground Truth Depth Map
  3. Local Mean Depth (M1)
  4. Local Variance Map
  5. Standard Deviation Map
  6. Valid Sample Count
  7. Coefficient of Variation
  8. Log10 Variance

### 3. Test Results

**Frame 0:**
- Depth range: [0.164, 7.820] meters
- Variance range: [0.0, 11.797]
- Mean variance: 0.0206
- Valid pixels: 100%

**Frame 100 (kernel_size=15):**
- Depth range: [0.487, 2.512] meters
- Variance range: [0.0, 0.140]
- Mean variance: 0.00067
- More uniform depth → lower variance

## Configuration Matching Training

The script defaults match your training configuration from `train_hospital_local_points_gt_lean_mapping.sh`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `data_root` | `/mnt/localssd/data` | Dataset location |
| `kernel_size` | `7` | Local window size |
| `kernel` | `gaussian` | Kernel type |
| `min_valid_count` | `8` | Min samples for variance |
| `prior_rel` | `0.1` | Relative prior |
| `prior_abs` | `0.0` | Absolute prior |

## Advanced Usage

### Different Kernel Sizes
```bash
# Smaller kernel (more local detail)
python visualize_depth_variance.py --frame 50 --kernel_size 5

# Larger kernel (smoother variance)
python visualize_depth_variance.py --frame 50 --kernel_size 15

# Very large kernel (global patterns)
python visualize_depth_variance.py --frame 50 --kernel_size 31
```

### Box vs Gaussian Kernel
```bash
# Gaussian (default - smooth, weighted)
python visualize_depth_variance.py --kernel gaussian

# Box (uniform weighting)
python visualize_depth_variance.py --kernel box
```

### Custom Prior Settings
```bash
# Higher prior (more conservative uncertainty estimates)
python visualize_depth_variance.py --prior_rel 0.2

# Lower prior (trust computed variance more)
python visualize_depth_variance.py --prior_rel 0.05
```

## Understanding the Outputs

### 1. Variance Map (Hot Colormap)
- **Red/White**: High variance (uncertain depth)
- **Dark/Black**: Low variance (certain depth)
- High variance typically appears at:
  - Depth discontinuities (edges)
  - Texture-less regions
  - Far surfaces

### 2. Coefficient of Variation (CoV)
- Normalized uncertainty: `std / mean`
- Range: [0, 0.3] (clipped for visualization)
- Higher values = higher relative uncertainty
- Useful for comparing uncertainty across different depth ranges

### 3. Valid Sample Count
- Shows effective number of valid pixels in local window
- For Gaussian kernel: weighted count
- For Box kernel: exact count
- Low count regions get higher prior variance

## Integration with Training

This visualization shows you the **exact variance weights** used during training with the Lean Mapping loss:

```python
# In training (pi3/models/loss_ablation.py)
m1, m2, var, count = lean_depth_moments_and_variance(
    depth=Z_gt,
    kernel_size=7,
    kernel='gaussian',
    min_valid_count=8,
    prior_rel=0.1,
)

# Weighted L1 loss
std = torch.sqrt(var + eps)
std = torch.clamp(std, min=0.1)  # std_min
weight = 1.0 / (std + eps)
loss = (weight * |Z_pred - Z_gt|).mean()
```

The variance map shows where the model is **allowed more error** (high variance = low weight) vs **penalized heavily** (low variance = high weight).

## Troubleshooting

### Dataset Not Found
If you get "FileNotFoundError", check:
```bash
ls -la /mnt/localssd/data/Easy/P000/depth_left/
ls -la /mnt/localssd/data/Easy/P000/image_left/
```

### Frame Index Out of Range
Check available frames:
```bash
python visualize_depth_variance.py --list_sequences
# Shows: Easy/P000 - 563 frames
# Valid frame indices: 0 to 562
```

### Memory Issues
If processing large images, reduce kernel size:
```bash
python visualize_depth_variance.py --kernel_size 5
```

## Next Steps

1. **Visualize different scenes** to understand variance patterns
2. **Compare variance across frames** to see temporal consistency
3. **Experiment with kernel sizes** to find optimal local window
4. **Analyze edge regions** where variance is typically high
5. **Correlate with training loss** to validate uncertainty-weighted training

## Files Generated

Each run creates:
- `variance_visualization_{difficulty}_{sequence}_{frame:06d}.png`
- Example: `variance_visualization_Easy_P000_000000.png`
- Size: ~500KB per image
- Resolution: 3200×1600 pixels (high quality)

---

**Script Created**: February 9, 2026
**Dataset**: TartanAir Hospital (Easy/P000, 563 frames)
**Purpose**: Visualize ground truth depth variance for Lean Mapping training
