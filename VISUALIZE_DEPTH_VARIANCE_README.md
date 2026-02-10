# Depth Variance Visualization

This script visualizes ground truth depth maps from the TartanAir Hospital dataset and computes local variance using the Lean Mapping / VSM-style variance computation.

## Script: `visualize_depth_variance.py`

### Features

- Loads ground truth depth maps from the hospital dataset
- Computes local depth variance using `lean_depth_moments_and_variance`
- Visualizes 8 different views:
  1. **RGB Image** - Original color image
  2. **Ground Truth Depth** - Raw depth map
  3. **Local Mean Depth (M1)** - First moment E[z]
  4. **Local Variance** - Var[z] = E[z²] - E[z]²
  5. **Standard Deviation** - sqrt(variance)
  6. **Valid Sample Count** - Number of valid pixels in local window
  7. **Coefficient of Variation** - Std/Mean (normalized uncertainty)
  8. **Log10 Variance** - Better visualization of small variance values

### Usage

#### 1. List available sequences

```bash
python visualize_depth_variance.py --list_sequences
```

#### 2. Visualize a specific frame

```bash
# Basic usage (default parameters matching training config)
python visualize_depth_variance.py --difficulty Easy --sequence P000 --frame 0

# Custom kernel size and parameters
python visualize_depth_variance.py \
    --difficulty Easy \
    --sequence P000 \
    --frame 100 \
    --kernel_size 15 \
    --kernel gaussian \
    --gaussian_sigma 2.5 \
    --prior_rel 0.1 \
    --save my_variance_map.png
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | `/mnt/localssd/data` | Root directory of hospital dataset |
| `--difficulty` | `Easy` | Difficulty level (`Easy` or `Hard`) |
| `--sequence` | `P000` | Sequence name (e.g., `P000`, `P001`) |
| `--frame` | `0` | Frame index to visualize |
| `--kernel_size` | `7` | Size of local window (matches training) |
| `--kernel` | `gaussian` | Kernel type (`box` or `gaussian`) |
| `--gaussian_sigma` | `kernel_size/6` | Standard deviation for Gaussian kernel |
| `--min_valid_count` | `8` | Minimum valid samples in window |
| `--prior_rel` | `0.1` | Relative prior (matches training config) |
| `--prior_abs` | `0.0` | Absolute prior variance |
| `--padding_mode` | `replicate` | Padding mode (`reflect`, `replicate`, or `zeros`) |
| `--save` | Auto-generated | Path to save visualization |

**Note**: Default parameters match the training configuration in `train_hospital_local_points_gt_lean_mapping.sh`

### Examples

```bash
# List all sequences and frames
python visualize_depth_variance.py --list_sequences

# Visualize frame 0 with default parameters (matching training)
python visualize_depth_variance.py --frame 0

# Visualize with larger kernel for smoother variance
python visualize_depth_variance.py --kernel_size 15 --frame 50

# Use box kernel instead of gaussian
python visualize_depth_variance.py --kernel box --kernel_size 11

# Higher prior for low-support regions
python visualize_depth_variance.py --prior_rel 0.2 --min_valid_count 12

# Save to specific location
python visualize_depth_variance.py --frame 123 --save ~/my_results/variance_123.png
```

### Test Results

Tested on TartanAir Hospital dataset:
- **Dataset**: Easy/P000 (563 frames)
- **Frame 0**:
  - Depth range: [0.164, 7.820] meters
  - Variance range: [0.0, 11.797]
  - 100% valid pixels
- **Frame 100**:
  - Depth range: [0.487, 2.512] meters
  - Variance range: [0.0, 0.140]
  - More uniform depth, lower variance

### Output

The script generates a figure with 8 subplots showing:
- Original RGB and depth data
- Local statistics (mean, variance, std dev)
- Uncertainty metrics (CoV, count)

The figure is saved as a PNG file (default: `variance_visualization_{difficulty}_{sequence}_{frame:06d}.png`)

### Testing

Test with synthetic data (no dataset required):

```bash
python test_visualize_depth_variance.py
```

This creates a synthetic depth map with multiple depth regions and computes variance with different kernel sizes.

## Algorithm: Lean Depth Variance

The variance computation uses the Lean Mapping / VSM approach:

1. **Local Moments**:
   - M1 = E[z] (mean depth in local window)
   - M2 = E[z²] (second moment)
   - Var = M2 - M1² (variance)

2. **Weighted by kernel**: Gaussian or box kernel for local averaging

3. **Prior for low support**: When valid samples < threshold, inject prior:
   - var_prior = (prior_rel × M1)² + prior_abs²
   - final_var = max(var, var_prior)

4. **Handles invalid depths**: Automatically masks invalid/infinite values

## Notes

- The dataset path in the script defaults to `/mnt/localssd/tartanair_tools/tartanair_data/hospital`
- Depth values > 80m are filtered as invalid (same as training)
- All computations use float32 for numerical stability
- The coefficient of variation (CoV) is clipped to [0, 0.3] for better visualization
