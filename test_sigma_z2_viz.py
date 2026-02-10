"""
Test sigma_z2 visualization in trainer.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from pi3.models.mipnerf_variance_loss import sigma_z2_from_gt_z_pixelwise

def test_sigma_z2_visualization():
    """Test creating sigma_z2 visualization."""
    print("\n" + "="*60)
    print("Test: Sigma_Z2 Visualization")
    print("="*60)

    # Create synthetic depth with edges
    H, W = 240, 320
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create depth with different regions
    depth_gt = torch.ones(1, 1, H, W, device=device) * 2.0
    # Add a closer region (edge)
    depth_gt[:, :, 60:120, :] = 1.0
    # Add a farther region
    depth_gt[:, :, 150:200, :] = 5.0

    # Camera intrinsics (TartanAir defaults)
    fx, fy, cx, cy = 320.0, 320.0, 160.0, 120.0

    print(f"Depth shape: {depth_gt.shape}")
    print(f"Depth range: [{depth_gt.min():.3f}, {depth_gt.max():.3f}]")

    # Compute sigma_z2
    sigma_z2, sigma_t2, t_mu, t_delta = sigma_z2_from_gt_z_pixelwise(
        depth_gt, fx, fy, cx, cy,
        window=5,
        lambda_prior=0.5,
        alpha_clamp=0.3
    )

    print(f"sigma_z2 shape: {sigma_z2.shape}")
    print(f"sigma_z2 stats - mean: {sigma_z2.mean():.6f}, std: {sigma_z2.std():.6f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Plot 1: GT Depth
    depth_np = depth_gt[0, 0].cpu().numpy()
    im1 = axes[0, 0].imshow(depth_np, cmap='turbo')
    axes[0, 0].set_title('GT Depth')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Plot 2: Sigma_Z2 (linear scale)
    sigma_np = sigma_z2[0, 0].cpu().numpy()
    im2 = axes[0, 1].imshow(sigma_np, cmap='viridis')
    axes[0, 1].set_title(f'σ²_Z (linear)\nmean={sigma_z2.mean():.6f}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Plot 3: Sigma_Z2 (log scale)
    sigma_log = np.log10(sigma_np + 1e-8)
    im3 = axes[1, 0].imshow(sigma_log, cmap='viridis')
    axes[1, 0].set_title('log10(σ²_Z)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Plot 4: t_delta (interval half-width)
    t_delta_np = t_delta[0, 0].cpu().numpy()
    im4 = axes[1, 1].imshow(t_delta_np, cmap='plasma')
    axes[1, 1].set_title(f't_delta\nmean={t_delta.mean():.6f}')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('test_sigma_z2_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: test_sigma_z2_visualization.png")
    plt.close()

    # Check variance at different regions
    edge1_var = sigma_z2[:, :, 55:65, :].mean()  # Edge between 2.0 and 1.0
    edge2_var = sigma_z2[:, :, 115:125, :].mean()  # Edge between 1.0 and 2.0
    edge3_var = sigma_z2[:, :, 145:155, :].mean()  # Edge between 2.0 and 5.0
    smooth1_var = sigma_z2[:, :, 80:100, :].mean()  # Smooth region at 1.0
    smooth2_var = sigma_z2[:, :, 170:190, :].mean()  # Smooth region at 5.0

    print(f"\nVariance at different regions:")
    print(f"  Edge 1 (2.0→1.0): {edge1_var:.6f}")
    print(f"  Edge 2 (1.0→2.0): {edge2_var:.6f}")
    print(f"  Edge 3 (2.0→5.0): {edge3_var:.6f}")
    print(f"  Smooth 1 (1.0):   {smooth1_var:.6f}")
    print(f"  Smooth 2 (5.0):   {smooth2_var:.6f}")

    assert edge1_var > smooth1_var, "Edge variance should be higher than smooth region"
    assert edge2_var > smooth1_var, "Edge variance should be higher than smooth region"
    assert smooth2_var > smooth1_var, "Far region should have higher variance (depth-dependent)"

    print("\n✓ All checks passed!")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_sigma_z2_visualization()
