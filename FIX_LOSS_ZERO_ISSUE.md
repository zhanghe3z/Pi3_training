# Fix for Loss Becoming Zero Issue

## Problem Diagnosis

All losses are showing 0.0000 because of the loss clipping logic in `trainers/base_trainer_accelerate.py:379-380`:

```python
if loss > self.cfg.train.clip_loss:
    loss = loss * 0.0  # This sets loss to 0 when it exceeds clip_loss (default: 10)
```

The real issue is that **loss is becoming very large (likely NaN or Inf)**, which triggers the clipping.

## Root Causes

### 1. Depth Activation Overflow (MOST LIKELY)

In `pi3/models/pi3_training.py:287`:

```python
z = torch.exp(z)  # ← If z is large, exp(z) will overflow to Inf!
```

**Example:**
- If raw z output is 30, then exp(30) ≈ 1.07e13
- If raw z output is 50, then exp(50) ≈ 5.18e21
- If raw z output > 88, then exp() → Inf

### 2. Normalization Division by Zero

In normalization steps, if all valid pixels are zero or very small, division can produce NaN/Inf.

### 3. Variance Computation (if using mipnerf loss)

The variance loss computation might produce NaN if intrinsics or depth values are problematic.

## Solutions

### Solution 1: Replace exp with softplus (RECOMMENDED)

Softplus is more stable than exp:

```bash
# Use the softplus training script you already have
./train_hospital_local_points_gt_pred_softplus.sh
```

Or modify the depth activation in config:

```yaml
model:
    depth_activation: softplus  # instead of null (which uses exp)
```

### Solution 2: Add Diagnostic Logging

Modify `trainers/base_trainer_accelerate.py` to print diagnostic info when loss is clipped:

```python
# Around line 379-380
if loss > self.cfg.train.clip_loss:
    print(f"\n⚠️ WARNING: Loss clipping triggered!")
    print(f"  Loss value: {loss.item()}")
    print(f"  Clip threshold: {self.cfg.train.clip_loss}")

    # Print detailed loss breakdown
    print(f"  Loss details:")
    for key in batch_output:
        if 'loss' in key and isinstance(batch_output[key], torch.Tensor):
            val = batch_output[key].item()
            print(f"    {key}: {val}")

    # Check for NaN/Inf
    if torch.isnan(loss):
        print(f"  Loss is NaN!")
    if torch.isinf(loss):
        print(f"  Loss is Inf!")

    # Check model outputs
    if 'local_points' in forward_output[0]:
        local_pts = forward_output[0]['local_points']
        print(f"  local_points stats:")
        print(f"    shape: {local_pts.shape}")
        print(f"    has NaN: {torch.isnan(local_pts).any().item()}")
        print(f"    has Inf: {torch.isinf(local_pts).any().item()}")
        if not torch.isnan(local_pts).any() and not torch.isinf(local_pts).any():
            print(f"    range: [{local_pts.min().item():.2f}, {local_pts.max().item():.2f}]")

        # Check depth channel specifically
        depth = local_pts[..., 2]
        print(f"  depth stats:")
        print(f"    has NaN: {torch.isnan(depth).any().item()}")
        print(f"    has Inf: {torch.isinf(depth).any().item()}")
        if not torch.isnan(depth).any() and not torch.isinf(depth).any():
            print(f"    range: [{depth.min().item():.2f}, {depth.max().item():.2f}]")

    loss = loss * 0.0
```

### Solution 3: Add Safety Checks in Model

Modify `pi3/models/pi3_training.py` around line 280-290 to add clipping before exp:

```python
# Before applying activation
xy, z = ret.split([2, 1], dim=-1)

# Clip z before activation to prevent overflow
z = torch.clamp(z, min=-10, max=10)  # exp(10) ≈ 22026, exp(-10) ≈ 0.000045

# Apply depth activation based on configuration
if self.depth_activation == 'softplus':
    z = torch.nn.functional.softplus(z)
else:
    # Default: use exp activation
    z = torch.exp(z)

local_points = torch.cat([xy * z, z], dim=-1)
```

### Solution 4: Increase clip_loss Threshold

If losses are legitimately large (not NaN/Inf), increase the threshold in your config:

```yaml
train:
    clip_loss: 100  # or even higher
```

**But first diagnose WHY the loss is large!**

## Recommended Action Plan

1. **First**: Add diagnostic logging (Solution 2) to see what's happening
2. **Then**: Switch to softplus activation (Solution 1) - this is the safest fix
3. **Optional**: Add safety clipping (Solution 3) as an extra safeguard

## Testing the Fix

After applying fixes, you should see:
- Loss values that are non-zero
- Gradients that are non-zero (grad_norm > 0)
- Model actually learning (loss decreasing over time)

If issues persist, use the diagnostic script:

```bash
python diagnose_loss_nan.py
```

And integrate it into your training loop to catch where NaN/Inf first appears.
