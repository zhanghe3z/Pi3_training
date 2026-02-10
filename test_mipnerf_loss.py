"""
Sanity check test for mip-NeRF variance-weighted loss implementation.
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from pi3.models.mipnerf_variance_loss import (
    sigma_z2_from_gt_z_pixelwise,
    weighted_l1_from_sigma_z2,
    laplace_nll_from_sigma_z2
)

def test_basic_variance_computation():
    """Test basic variance computation."""
    print("\n" + "="*60)
    print("Test 1: Basic Variance Computation")
    print("="*60)

    B, H, W = 2, 240, 320
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create synthetic depth with some variation
    Z_gt = torch.ones(B, 1, H, W, device=device) * 2.0
    # Add some edges
    Z_gt[:, :, 100:140, :] = 5.0

    # Camera intrinsics (TartanAir defaults)
    fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0

    print(f"Input shape: {Z_gt.shape}")
    print(f"Z_gt range: [{Z_gt.min():.3f}, {Z_gt.max():.3f}]")

    # Compute variance
    sigma_Z2, sigma_t2, t_mu, t_delta = sigma_z2_from_gt_z_pixelwise(
        Z_gt, fx, fy, cx, cy,
        window=5,
        lambda_prior=0.5,
        alpha_clamp=0.3
    )

    print(f"sigma_Z2 shape: {sigma_Z2.shape}")
    print(f"sigma_Z2 stats - mean: {sigma_Z2.mean():.6f}, std: {sigma_Z2.std():.6f}, max: {sigma_Z2.max():.6f}")
    print(f"t_delta stats - mean: {t_delta.mean():.6f}, std: {t_delta.std():.6f}, max: {t_delta.max():.6f}")

    # Check that variance is higher near edges
    edge_variance = sigma_Z2[:, :, 95:105, :].mean()
    smooth_variance = sigma_Z2[:, :, 0:50, :].mean()
    print(f"Edge variance: {edge_variance:.6f}, Smooth variance: {smooth_variance:.6f}")

    assert sigma_Z2.shape == Z_gt.shape, "Shape mismatch"
    assert sigma_Z2.min() >= 0, "Variance should be non-negative"
    assert edge_variance > smooth_variance, "Edge should have higher variance"

    print("✓ Test passed!")
    return sigma_Z2

def test_weighted_losses(sigma_Z2):
    """Test weighted loss functions."""
    print("\n" + "="*60)
    print("Test 2: Weighted Loss Functions")
    print("="*60)

    B, H, W = 2, 240, 320
    device = sigma_Z2.device

    Z_gt = torch.ones(B, 1, H, W, device=device) * 2.0
    Z_pred = Z_gt + 0.05 * torch.randn_like(Z_gt)

    # Test weighted L1
    loss_wl1 = weighted_l1_from_sigma_z2(Z_pred, Z_gt, sigma_Z2)
    print(f"Weighted L1 loss: {loss_wl1.item():.6f}")

    # Test Laplace NLL
    loss_laplace = laplace_nll_from_sigma_z2(Z_pred, Z_gt, sigma_Z2)
    print(f"Laplace NLL loss: {loss_laplace.item():.6f}")

    assert loss_wl1.item() > 0, "Loss should be positive"
    assert loss_laplace.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss_wl1), "Loss should not be NaN"
    assert not torch.isnan(loss_laplace), "Loss should not be NaN"

    print("✓ Test passed!")

def test_integration_with_loss_module():
    """Test integration with Pi3LossMipNeRFVariance."""
    print("\n" + "="*60)
    print("Test 3: Integration with Loss Module")
    print("="*60)

    from pi3.models.loss_ablation import Pi3LossMipNeRFVariance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, H, W = 2, 3, 224, 224

    # Create synthetic data
    pred = {
        'local_points': torch.randn(B, N, H, W, 3, device=device),
        'camera_poses': torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1),
    }

    # Create GT data structure
    gt_raw = []
    for b in range(B):
        batch_views = []
        for n in range(N):
            view = {
                'pts3d': np.random.randn(H, W, 3).astype(np.float32),
                'valid_mask': np.ones((H, W), dtype=bool),
                'camera_pose': np.eye(4, dtype=np.float32),
                'camera_intrinsics': np.array([[320., 0, 320./2], [0, 320., 224./2], [0, 0, 1]], dtype=np.float32),
                'img': torch.randn(3, H, W, device=device),
                'dataset': 'test',
            }
            batch_views.append(view)
        gt_raw.append(batch_views)

    # Initialize loss
    loss_fn = Pi3LossMipNeRFVariance(
        normalize_pred=False,
        normalize_gt=True,
        loss_type='weighted_l1'
    ).to(device)

    print(f"Loss function created: {loss_fn.__class__.__name__}")

    # Compute loss
    try:
        total_loss, details = loss_fn(pred, gt_raw)
        print(f"Total loss: {total_loss.item():.6f}")
        print("Loss details:")
        for k, v in details.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.6f}")
            else:
                print(f"  {k}: {v}")

        assert not torch.isnan(total_loss), "Total loss should not be NaN"
        assert total_loss.item() > 0, "Total loss should be positive"

        print("✓ Test passed!")
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("mip-NeRF Variance Loss - Sanity Check Tests")
    print("="*60)

    try:
        # Test 1: Basic variance computation
        sigma_Z2 = test_basic_variance_computation()

        # Test 2: Weighted losses
        test_weighted_losses(sigma_Z2)

        # Test 3: Integration with loss module
        success = test_integration_with_loss_module()

        if success:
            print("\n" + "="*60)
            print("✓ All tests passed!")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("✗ Some tests failed!")
            print("="*60 + "\n")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
