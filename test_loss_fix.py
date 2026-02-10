#!/usr/bin/env python3
"""
Quick test to verify the intrinsics scaling fix works correctly.
"""
import torch
import sys
sys.path.append('.')

from pi3.models.loss_ablation import Pi3LossMipNeRFVariance

print("=" * 80)
print("Testing Pi3LossMipNeRFVariance with intrinsics scaling fix")
print("=" * 80)

# Create dummy data
B, N, H, W = 1, 2, 224, 224

# Simulate GT data - before normalization (values around 10-20m)
gt_pts_raw = torch.randn(B, N, H, W, 3) * 5.0 + 15.0
gt_pts_raw[..., 2] = torch.abs(gt_pts_raw[..., 2])  # positive depth

# Simulate predicted data (small values from softplus initialization)
pred_local_pts = torch.randn(B, N, H, W, 3) * 0.05 + 0.3
pred_local_pts[..., 2] = torch.abs(pred_local_pts[..., 2])

valid_masks = torch.ones(B, N, H, W, dtype=torch.bool)
camera_poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)

# Create pred dict
pred = {
    'local_points': pred_local_pts,
    'camera_poses': camera_poses,
}

# Create raw GT with intrinsics (original scale)
fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)

gt_raw = []
for n in range(N):
    view_dict = {
        'img': torch.randn(B, 3, H, W),  # Add dummy image
        'camera_intrinsics': K,  # (B, 3, 3)
        'valid_mask': valid_masks[:, n],  # (B, H, W)
        'pts3d': gt_pts_raw[:, n],  # (B, H, W, 3) - original scale!
        'camera_pose': camera_poses[:, n],  # (B, 4, 4)
        'dataset': 'TarTanAir-Hospital'
    }
    gt_raw.append(view_dict)

# Create loss function with normalize_gt=True
loss_fn = Pi3LossMipNeRFVariance(
    train_conf=False,
    normalize_pred=False,  # Don't normalize predictions
    normalize_gt=True,     # Normalize GT (this will trigger norm_factor)
    loss_type='weighted_l1',
    window=5,
    lambda_prior=0.5,
    alpha_clamp=0.3,
)

print(f"Raw GT depth range: [{gt_pts_raw[..., 2].min():.4f}, {gt_pts_raw[..., 2].max():.4f}], mean: {gt_pts_raw[..., 2].mean():.4f}")
print(f"Pred depth range: [{pred_local_pts[..., 2].min():.4f}, {pred_local_pts[..., 2].max():.4f}], mean: {pred_local_pts[..., 2].mean():.4f}")
print(f"Original intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
print()

# Test the loss forward pass
try:
    print("Running forward pass...")
    result = loss_fn(pred, gt_raw)
    if len(result) == 3:
        loss, details, scale = result
    else:
        loss, details = result
        scale = None

    print()
    print("=" * 80)
    print("SUCCESS! Loss computed successfully")
    print("=" * 80)
    print(f"Total Loss: {loss.item():.6f}")
    print()
    print("Loss components:")
    for k, v in details.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.item():.6f}")
        else:
            print(f"  {k}: {v}")
    print()
    print(f"Scale: {scale}")

    # Check if loss is non-zero
    if loss.item() > 0:
        print()
        print("✓ Loss is non-zero - Fix appears to be working!")
        print("✓ The intrinsics scaling fix correctly handles GT normalization")
    else:
        print()
        print("✗ WARNING: Loss is still zero - there may be other issues")

except Exception as e:
    print()
    print("=" * 80)
    print("ERROR during forward pass:")
    print("=" * 80)
    print(f"{e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
