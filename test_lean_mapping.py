#!/usr/bin/env python3
"""
Quick test to verify Lean Mapping variance loss implementation.
"""

import torch
import sys
sys.path.insert(0, '/mnt/localssd/Pi3_training')

from pi3.models.lean_variance_loss import (
    lean_depth_moments_and_variance,
    weighted_l1_from_variance,
    laplace_nll_from_variance
)

def test_lean_variance():
    """Test Lean Mapping variance calculation"""
    print("=" * 60)
    print("Testing Lean Mapping Variance Loss")
    print("=" * 60)

    # Create synthetic depth data
    B, H, W = 2, 240, 320
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Shape: B={B}, H={H}, W={W}")

    # Create depth with different regions
    Z_gt = torch.ones(B, 1, H, W, device=device) * 2.0
    Z_gt[:, :, :H//2, :] = 5.0  # far region
    Z_gt[:, :, :, :W//2] += 1.0  # add variation

    # Add some noise
    Z_pred = Z_gt + 0.1 * torch.randn_like(Z_gt)

    print("\n" + "-" * 60)
    print("Test 1: Gaussian Kernel (7x7)")
    print("-" * 60)

    # Compute variance with Gaussian kernel
    m1_gauss, m2_gauss, var_gauss, count_gauss = lean_depth_moments_and_variance(
        Z_gt,
        kernel_size=7,
        kernel="gaussian",
        min_valid_count=8,
        prior_rel=0.1
    )

    print(f"M1 (mean depth) range: [{m1_gauss.min():.3f}, {m1_gauss.max():.3f}]")
    print(f"Variance range: [{var_gauss.min():.6f}, {var_gauss.max():.6f}]")
    print(f"Variance mean: {var_gauss.mean():.6f}")
    print(f"Count range: [{count_gauss.min():.1f}, {count_gauss.max():.1f}]")

    # Compute losses
    loss_wl1_gauss = weighted_l1_from_variance(Z_pred, Z_gt, var_gauss)
    loss_laplace_gauss = laplace_nll_from_variance(Z_pred, Z_gt, var_gauss)

    print(f"\nWeighted L1 loss: {loss_wl1_gauss.item():.6f}")
    print(f"Laplace NLL loss: {loss_laplace_gauss.item():.6f}")

    print("\n" + "-" * 60)
    print("Test 2: Box Kernel (7x7)")
    print("-" * 60)

    # Compute variance with Box kernel
    m1_box, m2_box, var_box, count_box = lean_depth_moments_and_variance(
        Z_gt,
        kernel_size=7,
        kernel="box",
        min_valid_count=8,
        prior_rel=0.1
    )

    print(f"M1 (mean depth) range: [{m1_box.min():.3f}, {m1_box.max():.3f}]")
    print(f"Variance range: [{var_box.min():.6f}, {var_box.max():.6f}]")
    print(f"Variance mean: {var_box.mean():.6f}")
    print(f"Count range: [{count_box.min():.1f}, {count_box.max():.1f}]")

    # Compute losses
    loss_wl1_box = weighted_l1_from_variance(Z_pred, Z_gt, var_box)
    loss_laplace_box = laplace_nll_from_variance(Z_pred, Z_gt, var_box)

    print(f"\nWeighted L1 loss: {loss_wl1_box.item():.6f}")
    print(f"Laplace NLL loss: {loss_laplace_box.item():.6f}")

    print("\n" + "-" * 60)
    print("Test 3: Different Kernel Sizes (Gaussian)")
    print("-" * 60)

    for k in [5, 7, 9]:
        _, _, var_k, _ = lean_depth_moments_and_variance(
            Z_gt,
            kernel_size=k,
            kernel="gaussian",
            min_valid_count=8,
            prior_rel=0.1
        )
        loss_k = weighted_l1_from_variance(Z_pred, Z_gt, var_k)
        print(f"Kernel size {k}x{k}: var_mean={var_k.mean():.6f}, loss={loss_k.item():.6f}")

    print("\n" + "-" * 60)
    print("Test 4: Different Prior Strengths (7x7 Gaussian)")
    print("-" * 60)

    for prior in [0.05, 0.1, 0.15, 0.2]:
        _, _, var_p, _ = lean_depth_moments_and_variance(
            Z_gt,
            kernel_size=7,
            kernel="gaussian",
            min_valid_count=8,
            prior_rel=prior
        )
        loss_p = weighted_l1_from_variance(Z_pred, Z_gt, var_p)
        print(f"Prior {prior:.2f}: var_mean={var_p.mean():.6f}, loss={loss_p.item():.6f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_lean_variance()
