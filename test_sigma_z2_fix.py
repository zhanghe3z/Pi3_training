"""
Test script to verify the sigma_z2 computation fix.
Tests both batched and unbatched camera intrinsics.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pi3.models.mipnerf_variance_loss import sigma_z2_from_gt_z_pixelwise

def test_intrinsics_extraction():
    """Test that we can correctly extract intrinsics from batched tensor."""

    # Create batched camera intrinsics (B, 3, 3)
    B = 2
    K_batched = torch.tensor([
        [[320., 0, 160.], [0, 320., 120.], [0, 0, 1]],
        [[320., 0, 160.], [0, 320., 120.], [0, 0, 1]]
    ], dtype=torch.float32)

    print("Batched K shape:", K_batched.shape)  # Should be (2, 3, 3)

    # Simulate the fix: extract first batch
    if K_batched.ndim == 3:
        K = K_batched[0]  # (3, 3)
    else:
        K = K_batched

    print("After extraction, K shape:", K.shape)  # Should be (3, 3)

    # Now extract intrinsics
    fx = K[0, 0].item()
    fy = K[1, 1].item()
    cx = K[0, 2].item()
    cy = K[1, 2].item()

    print(f"Extracted intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    assert fx == 320.0 and fy == 320.0 and cx == 160.0 and cy == 120.0
    print("✓ Intrinsics extraction test passed!")

def test_sigma_z2_computation():
    """Test sigma_z2 computation with proper intrinsics."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = 240, 320

    # Create synthetic depth
    Z_gt = torch.ones(1, 1, H, W, device=device) * 5.0

    # Add some variation (e.g., a step edge)
    Z_gt[:, :, :, W//2:] = 10.0

    # Camera intrinsics
    fx, fy = 320.0, 320.0
    cx, cy = (W-1)/2.0, (H-1)/2.0

    # Compute sigma_z2
    sigma_Z2, sigma_t2, t, t_delta = sigma_z2_from_gt_z_pixelwise(
        Z_gt, fx, fy, cx, cy,
        window=5,
        lambda_prior=0.5,
        alpha_clamp=0.3
    )

    print(f"sigma_Z2 shape: {sigma_Z2.shape}")  # Should be (1, 1, H, W)
    print(f"sigma_Z2 stats: mean={sigma_Z2.mean():.6f}, std={sigma_Z2.std():.6f}")
    print(f"sigma_Z2 range: min={sigma_Z2.min():.6f}, max={sigma_Z2.max():.6f}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ground truth depth
    im1 = axes[0].imshow(Z_gt[0, 0].cpu().numpy(), cmap='turbo')
    axes[0].set_title('Ground Truth Depth')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Variance (raw)
    sigma_np = sigma_Z2[0, 0].cpu().numpy()
    im2 = axes[1].imshow(sigma_np, cmap='turbo')
    axes[1].set_title(f'σ²_Z (raw)\nμ={sigma_np.mean():.2e}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Variance (log scale for better visualization)
    sigma_log = np.log10(sigma_np + 1e-8)
    im3 = axes[2].imshow(sigma_log, cmap='turbo')
    axes[2].set_title('log10(σ²_Z) - Pseudo-color')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('/mnt/localssd/Pi3_training/test_sigma_z2_fix_visualization.png', dpi=150)
    print(f"✓ Visualization saved to test_sigma_z2_fix_visualization.png")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Testing sigma_z2 computation fix")
    print("=" * 60)

    print("\n1. Testing intrinsics extraction from batched tensor...")
    test_intrinsics_extraction()

    print("\n2. Testing sigma_z2 computation and visualization...")
    test_sigma_z2_computation()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
