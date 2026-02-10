"""
Diagnostic script to check for NaN/Inf in loss computation.
This script adds hooks to the model and loss functions to identify where NaN/Inf appears.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

def check_tensor(tensor, name="tensor"):
    """Check if a tensor contains NaN or Inf."""
    if tensor is None:
        print(f"  [{name}] is None")
        return

    if not isinstance(tensor, torch.Tensor):
        print(f"  [{name}] is not a tensor: {type(tensor)}")
        return

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        print(f"  ⚠️ [{name}] has {'NaN' if has_nan else ''} {'Inf' if has_inf else ''}")
        print(f"     Shape: {tensor.shape}")
        print(f"     Min: {tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].min().item() if tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].numel() > 0 else 'N/A'}")
        print(f"     Max: {tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].max().item() if tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].numel() > 0 else 'N/A'}")
        print(f"     NaN count: {torch.isnan(tensor).sum().item()}")
        print(f"     Inf count: {torch.isinf(tensor).sum().item()}")
        return True
    else:
        print(f"  ✓ [{name}] OK - Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}], Mean: {tensor.mean().item():.6f}")
        return False


def add_forward_hooks(model):
    """Add forward hooks to check for NaN/Inf in model outputs."""

    def hook_fn(module, input, output):
        module_name = module.__class__.__name__

        # Check inputs
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    check_tensor(inp, f"{module_name}.input[{i}]")

        # Check outputs
        if isinstance(output, torch.Tensor):
            check_tensor(output, f"{module_name}.output")
        elif isinstance(output, dict):
            for key, val in output.items():
                if isinstance(val, torch.Tensor):
                    check_tensor(val, f"{module_name}.output[{key}]")
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    check_tensor(out, f"{module_name}.output[{i}]")

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    return hooks


def diagnose_model_output(pred):
    """Diagnose model predictions."""
    print("\n" + "="*80)
    print("DIAGNOSING MODEL OUTPUT")
    print("="*80)

    has_issue = False

    if 'local_points' in pred:
        local_pts = pred['local_points']
        print(f"\nLocal Points Shape: {local_pts.shape}")
        has_issue |= check_tensor(local_pts, "local_points")

        # Check individual coordinates
        if len(local_pts.shape) == 5:  # (B, N, H, W, 3)
            has_issue |= check_tensor(local_pts[..., 0], "local_points[x]")
            has_issue |= check_tensor(local_pts[..., 1], "local_points[y]")
            has_issue |= check_tensor(local_pts[..., 2], "local_points[z/depth]")

    if 'camera_poses' in pred:
        has_issue |= check_tensor(pred['camera_poses'], "camera_poses")

    if 'global_points' in pred and pred['global_points'] is not None:
        has_issue |= check_tensor(pred['global_points'], "global_points")

    return has_issue


def diagnose_gt(gt):
    """Diagnose ground truth data."""
    print("\n" + "="*80)
    print("DIAGNOSING GROUND TRUTH")
    print("="*80)

    has_issue = False

    for key in ['local_points', 'global_points', 'valid_masks', 'camera_poses']:
        if key in gt and gt[key] is not None:
            has_issue |= check_tensor(gt[key], f"gt.{key}")

    # Check valid mask statistics
    if 'valid_masks' in gt:
        masks = gt['valid_masks']
        num_valid = masks.sum().item()
        num_total = masks.numel()
        print(f"\nValid Masks: {num_valid}/{num_total} ({100*num_valid/num_total:.2f}%)")

        if num_valid == 0:
            print(f"  ⚠️ WARNING: No valid pixels! All masks are False.")
            has_issue = True

    return has_issue


def diagnose_normalize(local_points, masks):
    """Diagnose normalization process."""
    print("\n" + "="*80)
    print("DIAGNOSING NORMALIZATION")
    print("="*80)

    has_issue = False

    B, N, H, W, _ = local_points.shape

    # Calculate norm_factor
    all_pts = local_points.clone()
    all_pts[~masks] = 0
    all_pts = all_pts.reshape(B, N, -1, 3)
    all_dis = all_pts.norm(dim=-1)

    has_issue |= check_tensor(all_dis, "all_dis (norms)")

    # Check for division by zero
    num_valid_per_batch = masks.float().sum(dim=[-1, -2, -3])
    print(f"\nNum valid pixels per batch: {num_valid_per_batch}")

    if (num_valid_per_batch == 0).any():
        print(f"  ⚠️ WARNING: Some batches have no valid pixels!")
        has_issue = True

    norm_factor = all_dis.sum(dim=[-1, -2]) / (num_valid_per_batch + 1e-8)
    has_issue |= check_tensor(norm_factor, "norm_factor")

    # Check normalized points
    local_points_normalized = local_points / norm_factor[..., None, None, None, None]
    has_issue |= check_tensor(local_points_normalized, "local_points_normalized")

    return has_issue


def diagnose_depth_activation(z_before_activation, activation='exp'):
    """Diagnose depth activation function."""
    print("\n" + "="*80)
    print(f"DIAGNOSING DEPTH ACTIVATION ({activation})")
    print("="*80)

    has_issue = False

    print(f"\nBefore activation:")
    has_issue |= check_tensor(z_before_activation, "z_before_activation")

    if activation == 'exp':
        # Check if values are too large for exp
        max_safe_exp = 20  # exp(20) ≈ 4.8e8
        too_large = (z_before_activation > max_safe_exp).any()

        if too_large:
            print(f"  ⚠️ WARNING: z values > {max_safe_exp} detected! This will cause exp() to overflow.")
            print(f"     Max z: {z_before_activation.max().item():.2f}")
            has_issue = True

        z_after = torch.exp(z_before_activation)
    elif activation == 'softplus':
        z_after = torch.nn.functional.softplus(z_before_activation)
    else:
        z_after = z_before_activation

    print(f"\nAfter {activation} activation:")
    has_issue |= check_tensor(z_after, f"z_after_{activation}")

    return has_issue


if __name__ == "__main__":
    print("Diagnostic utility loaded.")
    print("\nUsage in training script:")
    print("```python")
    print("from diagnose_loss_nan import diagnose_model_output, diagnose_gt, diagnose_normalize")
    print("")
    print("# After model forward")
    print("pred = model(imgs)")
    print("diagnose_model_output(pred)")
    print("")
    print("# After GT preparation")
    print("gt = loss_fn.prepare_gt(batch)")
    print("diagnose_gt(gt)")
    print("")
    print("# Before normalization")
    print("diagnose_normalize(pred['local_points'], gt['valid_masks'])")
    print("```")
