#!/usr/bin/env python3
"""Diagnose variance and weight distribution in mip-NeRF loss."""

import torch
import numpy as np
from pi3.models.mipnerf_variance_loss import sigma_z2_from_gt_z_pixelwise

# Simulate a typical depth map
B, H, W = 1, 224, 224
depth = torch.rand(B, 1, H, W) * 1.0 + 0.3  # Range [0.3, 1.3]

# Typical intrinsics
fx = fy = torch.tensor([300.0])
cx = torch.tensor([H/2])
cy = torch.tensor([W/2])

# Compute variance with default parameters
sigma_Z2, sigma_t2, t_mu, t_delta = sigma_z2_from_gt_z_pixelwise(
    depth, fx, fy, cx, cy,
    window=5,
    lambda_prior=0.5,
    alpha_clamp=0.3
)

# Compute weights as in the loss function
sigma_std = torch.sqrt(sigma_Z2 + 1e-6)
weights = 1.0 / (sigma_std + 1e-6)

print("=" * 80)
print("Variance and Weight Statistics")
print("=" * 80)
print(f"\nDepth Statistics:")
print(f"  Range: [{depth.min():.4f}, {depth.max():.4f}]")
print(f"  Mean: {depth.mean():.4f}")

print(f"\nsigma_Z2 (variance) Statistics:")
print(f"  Range: [{sigma_Z2.min():.6f}, {sigma_Z2.max():.6f}]")
print(f"  Mean: {sigma_Z2.mean():.6f}")
print(f"  Std: {sigma_Z2.std():.6f}")
print(f"  Median: {sigma_Z2.median():.6f}")

# Percentiles
percentiles = [5, 25, 50, 75, 95]
for p in percentiles:
    val = torch.quantile(sigma_Z2, p/100.0)
    print(f"  P{p}: {val:.6f}")

print(f"\nsigma_std (sqrt of variance) Statistics:")
print(f"  Range: [{sigma_std.min():.6f}, {sigma_std.max():.6f}]")
print(f"  Mean: {sigma_std.mean():.6f}")
print(f"  Median: {sigma_std.median():.6f}")

print(f"\nWeights (1 / sigma_std) Statistics:")
print(f"  Range: [{weights.min():.2f}, {weights.max():.2f}]")
print(f"  Mean: {weights.mean():.2f}")
print(f"  Median: {weights.median():.2f}")

# Check for extreme weights
extreme_thresh = 100
n_extreme = (weights > extreme_thresh).sum().item()
print(f"\n  Number of pixels with weights > {extreme_thresh}: {n_extreme} / {weights.numel()}")
print(f"  Percentage: {100 * n_extreme / weights.numel():.2f}%")

# Simulate loss with typical depth error
depth_error = 0.05  # 5cm typical error
loss_unweighted = depth_error
loss_weighted = (weights * depth_error).mean().item()

print(f"\nLoss Comparison (with {depth_error:.2f}m typical depth error):")
print(f"  Unweighted loss: {loss_unweighted:.4f}")
print(f"  Weighted loss: {loss_weighted:.4f}")
print(f"  Amplification factor: {loss_weighted / loss_unweighted:.2f}x")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
if weights.mean() > 50:
    print("⚠️  Weights are TOO LARGE on average!")
    print("   This will cause the loss to explode.")
    print("   Solution: Increase eps or clamp maximum weight")
elif weights.max() > 500:
    print("⚠️  Some weights are EXTREMELY LARGE!")
    print("   This will cause gradient instability.")
    print("   Solution: Clamp maximum weight")
else:
    print("✓  Weights are in reasonable range.")

print("\nRecommended fixes:")
print("1. Increase eps in weight computation: weights = 1 / (sigma_std + 0.01)  # instead of 1e-6")
print("2. Or clamp maximum weight: weights = torch.clamp(weights, max=50.0)")
print("3. Or use different weighting: weights = 1 / (sigma_Z2 + 0.01)  # no sqrt")
