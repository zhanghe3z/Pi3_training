"""
Lean Mapping / VSM-style local depth variance for uncertainty-aware depth loss.

This module computes local depth variance using kernel-based moments:
- M1 = E[z], M2 = E[z^2], Var = M2 - M1^2
- Supports box and Gaussian kernels
- Includes prior variance for low-support regions
- Optional winsorization to prevent outlier explosion
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Literal


def _as_4d(depth: torch.Tensor) -> torch.Tensor:
    """Accept (B,H,W) or (B,1,H,W) or (H,W); return (B,1,H,W)."""
    if depth.dim() == 2:
        depth = depth[None, None, ...]
    elif depth.dim() == 3:
        depth = depth[:, None, ...]
    elif depth.dim() == 4:
        assert depth.size(1) == 1, "depth should be (B,1,H,W) if 4D."
    else:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")
    return depth


def _make_box_kernel(k: int, device, dtype) -> torch.Tensor:
    """Create a box (uniform) kernel of size k x k."""
    ker = torch.ones((1, 1, k, k), device=device, dtype=dtype)
    return ker


def _make_gaussian_kernel(k: int, sigma: float, device, dtype) -> torch.Tensor:
    """Create a 2D Gaussian kernel of size k x k with standard deviation sigma."""
    # 2D separable gaussian
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g1 = torch.exp(-(ax * ax) / (2 * sigma * sigma)).clamp_min(0)
    g1 = g1 / g1.sum().clamp_min(1e-12)
    g2 = g1[:, None] * g1[None, :]
    ker = g2[None, None, ...]  # (1,1,k,k)
    return ker


@torch.no_grad()
def lean_depth_moments_and_variance(
    depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    kernel_size: int = 3,
    kernel: Literal["box", "gaussian"] = "gaussian",
    gaussian_sigma: Optional[float] = None,
    padding_mode: Literal["reflect", "replicate", "zeros"] = "replicate",
    min_valid_count: int = 8,
    # prior for low support regions: var_prior = (prior_rel * mean)^2 + prior_abs^2
    prior_rel: float = 0.05,
    prior_abs: float = 0.0,
    # optional: clamp depth to global range to prevent z^2 exploding
    winsorize: bool = False,
    winsor_min: float = 0.0,
    winsor_max: float = 1e6,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Lean/VSM-style local moments:
      M1 = E[z], M2 = E[z^2], Var = M2 - M1^2.

    Args:
        depth: Input depth map (B,H,W) or (B,1,H,W)
        valid_mask: Optional boolean mask for valid pixels (B,H,W) or (B,1,H,W)
        kernel_size: Size of local window (default: 7)
        kernel: Type of kernel - "box" or "gaussian" (default: "gaussian")
        gaussian_sigma: Standard deviation for Gaussian kernel (default: kernel_size/6)
        padding_mode: Padding mode - "reflect", "replicate", or "zeros" (default: "reflect")
        min_valid_count: Minimum valid samples in window to trust variance (default: 8)
        prior_rel: Relative prior for variance as fraction of mean depth (default: 0.05)
        prior_abs: Absolute prior for variance (default: 0.0)
        winsorize: Whether to clamp depth values to prevent outliers (default: False)
        winsor_min: Minimum depth value for winsorization (default: 0.0)
        winsor_max: Maximum depth value for winsorization (default: 1e6)
        eps: Small value for numerical stability (default: 1e-6)

    Returns:
        m1: (B,1,H,W) Local mean depth E[z]
        m2: (B,1,H,W) Local second moment E[z^2]
        var: (B,1,H,W) Local variance Var[z] = E[z^2] - E[z]^2
        count: (B,1,H,W) Local valid sample count (weighted for gaussian kernel)
    """
    depth = _as_4d(depth)
    B, _, H, W = depth.shape
    device = depth.device
    # use float32 internally for stability
    z = depth.to(torch.float32)

    # Apply 5x5 Gaussian filter to depth BEFORE computing moments
    # This smooths the depth map to reduce noise
    # z_pad_gauss = F.pad(z, (2, 2, 2, 2), mode='replicate')
    # z = F.conv2d(z_pad_gauss, gauss_kernel_5x5, padding=0)

    if winsorize:
        z = z.clamp(min=winsor_min, max=winsor_max)

    if valid_mask is None:
        # valid if finite and > 0
        valid_mask = torch.isfinite(z) & (z > 0)
    else:
        valid_mask = _as_4d(valid_mask).to(torch.bool)
        valid_mask = valid_mask & torch.isfinite(z) & (z > 0)

    # zero-out invalid to avoid polluting sums
    z = torch.where(valid_mask, z, torch.zeros_like(z))
    z2 = z * z

    k = int(kernel_size)
    assert k % 2 == 1 and k >= 1, "kernel_size should be odd and >= 1."

    if kernel == "box":
        ker = _make_box_kernel(k, device=device, dtype=torch.float32)
    elif kernel == "gaussian":
        if gaussian_sigma is None:
            # a common default: sigma ~ k/6 (covers ~3 sigma in half width)
            gaussian_sigma = max(k / 6.0, 1e-3)
        ker = _make_gaussian_kernel(k, gaussian_sigma, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    pad = k // 2
    if padding_mode in ("reflect", "replicate"):
        z_pad  = F.pad(z,  (pad, pad, pad, pad), mode=padding_mode)
        z2_pad = F.pad(z2, (pad, pad, pad, pad), mode=padding_mode)
        m_pad  = F.pad(valid_mask.to(torch.float32), (pad, pad, pad, pad), mode=padding_mode)
    elif padding_mode == "zeros":
        z_pad, z2_pad = z, z2
        m_pad = valid_mask.to(torch.float32)
    else:
        raise ValueError(f"Unknown padding_mode: {padding_mode}")

    # weighted sums
    sum_w  = F.conv2d(m_pad,  ker, padding=0 if padding_mode != "zeros" else pad)
    sum_z  = F.conv2d(z_pad,  ker, padding=0 if padding_mode != "zeros" else pad)
    sum_z2 = F.conv2d(z2_pad, ker, padding=0 if padding_mode != "zeros" else pad)

    # moments
    m1 = sum_z  / sum_w.clamp_min(eps)
    m2 = sum_z2 / sum_w.clamp_min(eps)

    var = (m2 - m1 * m1).clamp_min(0.0)

    # Edge case: too few valid samples in window -> inject prior variance
    # For gaussian kernel, sum_w is "weighted count"; for box it's exact count if ker is ones.
    # We'll threshold by approximate effective count.
    # Effective count estimate:
    if kernel == "box":
        count = sum_w  # exact count
        low_support = count < float(min_valid_count)
    else:
        # gaussian: sum_w is in [0,1] * count-ish but not integer; approximate by scaling
        # scale so that full-valid window has value ~= 1
        full = ker.sum().item()
        count = sum_w / max(full, eps) * (k * k)  # rough "count-like" measure
        low_support = count < float(min_valid_count)

    var_prior = (prior_rel * m1).pow(2) + (prior_abs ** 2)
    var = torch.where(low_support, torch.maximum(var, var_prior), var)

    # cast back to original dtype if you want (usually keep float32 for variance)
    return m1, m2, var, count


@torch.no_grad()
def lean_variance_from_depth(
    depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    kernel_size: int = 7,
    kernel: Literal["box", "gaussian"] = "gaussian",
    gaussian_sigma: Optional[float] = None,
    min_valid_count: int = 8,
    prior_rel: float = 0.1,
    prior_abs: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Simplified interface that returns only variance.

    Args:
        depth: Input depth map (B,H,W) or (B,1,H,W)
        valid_mask: Optional boolean mask for valid pixels
        kernel_size: Size of local window (default: 7)
        kernel: Type of kernel - "box" or "gaussian" (default: "gaussian")
        gaussian_sigma: Standard deviation for Gaussian kernel (default: kernel_size/6)
        min_valid_count: Minimum valid samples in window (default: 8)
        prior_rel: Relative prior for variance (default: 0.1)
        prior_abs: Absolute prior for variance (default: 0.0)
        eps: Small value for numerical stability (default: 1e-6)

    Returns:
        var: (B,1,H,W) Local variance
    """
    _, _, var, _ = lean_depth_moments_and_variance(
        depth=depth,
        valid_mask=valid_mask,
        kernel_size=kernel_size,
        kernel=kernel,
        gaussian_sigma=gaussian_sigma,
        min_valid_count=min_valid_count,
        prior_rel=prior_rel,
        prior_abs=prior_abs,
        eps=eps,
    )
    return var


def weighted_l1_from_variance(
    Z_pred: torch.Tensor,
    Z_gt: torch.Tensor,
    variance: torch.Tensor,
    eps: float = 1e-6,
    detach_variance: bool = True,
    std_min: float = 0.1,
) -> torch.Tensor:
    """
    Weighted L1 loss using variance:
      w = 1 / (sqrt(variance) + eps)
      loss = mean( w * |Z_pred - Z_gt| )

    Args:
        Z_pred: Predicted depth
        Z_gt: Ground truth depth
        variance: Depth variance
        eps: Small value for numerical stability
        detach_variance: Whether to detach variance from gradient computation
        std_min: Minimum value for standard deviation (default: 0.1) to prevent extremely large weights
    """
    assert Z_pred.shape == Z_gt.shape == variance.shape
    if detach_variance:
        variance = variance.detach()
    std = torch.sqrt(variance + eps)
    # Clip standard deviation to minimum value to prevent extremely large weights
    std = torch.clamp(std, min=std_min)
    w = 1.0 / (std + eps)
    return (w * (Z_pred - Z_gt).abs()).mean()


def laplace_nll_from_variance(
    Z_pred: torch.Tensor,
    Z_gt: torch.Tensor,
    variance: torch.Tensor,
    eps: float = 1e-6,
    b_min: float = 1e-3,
    detach_variance: bool = True,
    std_min: float = 0.1,
) -> torch.Tensor:
    """
    Laplace NLL loss using variance:
    Laplace variance: Var = 2*b^2  =>  b = sqrt(Var/2)
    NLL: L = |r|/b + log b

    Args:
        Z_pred: Predicted depth
        Z_gt: Ground truth depth
        variance: Depth variance
        eps: Small value for numerical stability
        b_min: Minimum value for b
        detach_variance: Whether to detach variance from gradient computation
        std_min: Minimum value for standard deviation (default: 0.1)
    """
    assert Z_pred.shape == Z_gt.shape == variance.shape
    if detach_variance:
        variance = variance.detach()
    # Clip standard deviation first
    std = torch.sqrt(variance + eps)
    std = torch.clamp(std, min=std_min)
    # Convert to Laplace parameter b = sigma/sqrt(2)
    b = std / torch.sqrt(torch.tensor(2.0, device=std.device, dtype=std.dtype))
    b = torch.clamp(b, min=b_min)
    r = (Z_pred - Z_gt).abs()
    return (r / b + torch.log(b)).mean()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    B, H, W = 2, 240, 320
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create synthetic depth with some structure
    Z_gt = torch.ones(B, 1, H, W, device=device) * 2.0
    Z_gt[:, :, :H//2, :] = 5.0  # far region
    Z_pred = Z_gt + 0.1 * torch.randn_like(Z_gt)

    # Compute variance using Lean mapping
    m1, m2, var, count = lean_depth_moments_and_variance(
        Z_gt, kernel_size=7, kernel="gaussian",
        min_valid_count=8, prior_rel=0.1
    )

    # Compute losses
    loss_wl1 = weighted_l1_from_variance(Z_pred, Z_gt, var)
    loss_laplace = laplace_nll_from_variance(Z_pred, Z_gt, var)

    print(f"Weighted L1 loss: {loss_wl1.item():.6f}")
    print(f"Laplace NLL loss: {loss_laplace.item():.6f}")
    print(f"Variance stats - mean: {var.mean():.6f}, std: {var.std():.6f}")
    print(f"M1 (mean depth) range: [{m1.min():.3f}, {m1.max():.3f}]")
    print(f"Count range: [{count.min():.1f}, {count.max():.1f}]")
