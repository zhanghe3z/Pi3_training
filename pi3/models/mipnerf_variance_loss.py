"""
mip-NeRF inspired variance-weighted depth loss for local points.

This module provides per-pixel depth variance estimation based on:
1. Local spatial variation (5x5 window) - captures edge aliasing/sub-pixel mixture
2. Depth-dependent prior - pixel footprint grows with distance

The variance is used to weight the depth loss, down-weighting uncertain regions.
"""

import torch
import torch.nn.functional as F

def _as_broadcast_intr(x, device, dtype, B):
    """Return tensor shaped (B,1,1) broadcastable to (B,1,H,W)."""
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
    else:
        t = torch.tensor(x, device=device, dtype=dtype)
    if t.numel() == 1:
        t = t.view(1).repeat(B)
    if t.numel() == B:
        t = t.view(B, 1, 1)
    else:
        # allow already (B,1,1) or (B,) etc.
        t = t.view(B, 1, 1)
    return t

@torch.no_grad()
def ray_dir_z(H, W, fx, fy, cx, cy, device, dtype, B):
    """
    Compute per-pixel unit ray direction z-component d_z for normalized rays:
    dir = normalize([ (u-cx)/fx, (v-cy)/fy, 1 ])
    d_z = 1 / sqrt(x^2 + y^2 + 1)
    Return shape: (B,1,H,W)
    """
    fx = _as_broadcast_intr(fx, device, dtype, B)
    fy = _as_broadcast_intr(fy, device, dtype, B)
    cx = _as_broadcast_intr(cx, device, dtype, B)
    cy = _as_broadcast_intr(cy, device, dtype, B)

    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")  # (H,W)

    uu = uu.view(1, 1, H, W)  # broadcast over batch
    vv = vv.view(1, 1, H, W)

    x = (uu - cx.view(B,1,1,1)) / fx.view(B,1,1,1)
    y = (vv - cy.view(B,1,1,1)) / fy.view(B,1,1,1)

    dz = 1.0 / torch.sqrt(x * x + y * y + 1.0)
    return dz

@torch.no_grad()
def mipnerf_sigma_t2_from_t_mu_delta(t_mu, t_delta, eps=1e-8):
    """
    mip-NeRF stable closed-form for ray-distance variance sigma_t^2 given:
    t_mu:    (B,1,H,W) mean distance
    t_delta: (B,1,H,W) half-width of interval [t0,t1]
    Formula (stable form used in mip-NeRF paper):
      sigma_t^2 = t_delta^2/3 - [4 t_delta^4 (12 t_mu^2 - t_delta^2)] / [15 (3 t_mu^2 + t_delta^2)^2]
    """
    t_mu = torch.clamp(t_mu, min=eps)
    t_delta = torch.clamp(t_delta, min=eps)

    t_mu2 = t_mu * t_mu
    td2 = t_delta * t_delta
    denom = (3.0 * t_mu2 + td2)
    sigma_t2 = (td2 / 3.0) - (4.0 * (td2 * td2) * (12.0 * t_mu2 - td2)) / (15.0 * (denom * denom) + eps)
    return torch.clamp(sigma_t2, min=0.0)

@torch.no_grad()
def _gaussian_blur_5x5(x: torch.Tensor):
    """Simple separable 5x5 Gaussian blur, sigma≈1, fixed kernel."""
    # [1,4,6,4,1] / 16  (binomial)
    k = torch.tensor([1.,4.,6.,4.,1.], device=x.device, dtype=x.dtype)
    k = k / k.sum()
    # depthwise conv
    B, C, H, W = x.shape
    kx = k.view(1,1,1,5).repeat(C,1,1,1)
    ky = k.view(1,1,5,1).repeat(C,1,1,1)

    x = F.pad(x, (2,2,0,0), mode="replicate")
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0,0,2,2), mode="replicate")
    x = F.conv2d(x, ky, groups=C)
    return x

@torch.no_grad()
def _local_edge_delta_moment(t: torch.Tensor, k=5, eps=1e-6, moment_scale=1.645, prefilter="none"):
    """
    Continuous alternative to quantiles: use local variance (second moment).
    moment_scale:
      If you want to mimic 0.5*(Q95-Q05) for roughly Gaussian noise,
      0.5*(Q95-Q05) ≈ 1.645 * std, hence default 1.645.
      You can reduce it (e.g., 1.0) if you want milder edge deltas.
    prefilter:
      'none' | 'avg3' | 'gauss5'
    """
    if prefilter == "avg3":
        t = F.avg_pool2d(F.pad(t, (1,1,1,1), mode="replicate"), 3, stride=1, padding=0)
    elif prefilter == "gauss5":
        t = _gaussian_blur_5x5(t)

    pad = k // 2
    t_pad = F.pad(t, (pad,pad,pad,pad), mode="replicate")

    mean = F.avg_pool2d(t_pad, k, stride=1, padding=0)
    mean2 = F.avg_pool2d(t_pad * t_pad, k, stride=1, padding=0)
    var = torch.clamp(mean2 - mean * mean, min=0.0)
    std = torch.sqrt(var + eps)
    return moment_scale * std  # (B,1,H,W)

@torch.no_grad()
def _local_edge_delta_quantile(t: torch.Tensor, k=5, eps=1e-6, prefilter="gauss5"):
    """
    Keep your quantile idea but fix the two main issues:
      - replicate padding instead of zero padding
      - prefilter to suppress GT tile/upscale fingerprints before sorting
    """
    if prefilter == "avg3":
        t = F.avg_pool2d(F.pad(t, (1,1,1,1), mode="replicate"), 3, stride=1, padding=0)
    elif prefilter == "gauss5":
        t = _gaussian_blur_5x5(t)

    pad = k // 2
    t_pad = F.pad(t, (pad,pad,pad,pad), mode="replicate")
    patches = F.unfold(t_pad, kernel_size=k, padding=0, stride=1)  # (B, k*k, H*W)

    patches_sorted, _ = torch.sort(patches, dim=1)
    # for 25 values: Q05~index1, Q95~index23
    q05 = patches_sorted[:, 1, :]
    q95 = patches_sorted[:, 23, :]
    t_delta_edge = 0.5 * (q95 - q05)
    B, _, H, W = t.shape
    return torch.clamp(t_delta_edge.view(B,1,H,W), min=0.0)

@torch.no_grad()
def sigma_z2_from_gt_z_pixelwise_fixed(
    Z_gt: torch.Tensor,
    fx, fy, cx, cy,
    window: int = 5,
    lambda_prior: float = 0.5,
    alpha_clamp: float = 0.3,
    eps: float = 1e-6,
    edge_method: str = "moment",      # "moment" (recommended) or "quantile"
    prefilter: str = "gauss5",        # "none" | "avg3" | "gauss5"
    moment_scale: float = 1.645,      # map std -> half (Q95-Q05) approx
):
    """
    Pixel-wise sigma_Z^2 from GT z-depth, with grid-artifact fixes.

    Fixes:
      1) replicate padding (no zero padding contamination)
      2) avoid hard-quantile jumps by default (moment-based local variance)
      3) optional prefilter to suppress GT tile/upscale fingerprints
    """
    assert Z_gt.dim() == 4 and Z_gt.size(1) == 1
    B, _, H, W = Z_gt.shape
    device, dtype = Z_gt.device, Z_gt.dtype
    assert window == 5

    # 1) d_z and ray-distance t
    d_z = ray_dir_z(H, W, fx, fy, cx, cy, device, dtype, B)
    Z = torch.clamp(Z_gt, min=eps)
    t = Z / (d_z + eps)

    # 2) edge/aliasing term (the usual source of grid artifacts)
    if edge_method == "moment":
        t_delta_edge = _local_edge_delta_moment(t, k=window, eps=eps, moment_scale=moment_scale, prefilter=prefilter)
    elif edge_method == "quantile":
        t_delta_edge = _local_edge_delta_quantile(t, k=window, eps=eps, prefilter=prefilter)
    else:
        raise ValueError(f"Unknown edge_method={edge_method}")

    # 3) mild footprint prior (smooth;不会产生网格)
    fx_t = _as_broadcast_intr(fx, device, dtype, B)
    fy_t = _as_broadcast_intr(fy, device, dtype, B)
    tan_theta = torch.sqrt((0.5 / (fx_t + eps))**2 + (0.5 / (fy_t + eps))**2).view(B,1,1,1)
    t_delta_prior = lambda_prior * tan_theta * t

    # 4) combine + clamp
    t_delta = torch.sqrt(t_delta_edge * t_delta_edge + t_delta_prior * t_delta_prior + eps)
    if alpha_clamp is not None and alpha_clamp > 0:
        t_delta = torch.minimum(t_delta, alpha_clamp * t)

    # 5) mip-NeRF sigma_t^2 -> sigma_Z^2
    sigma_t2 = mipnerf_sigma_t2_from_t_mu_delta(t, t_delta, eps=eps)
    sigma_Z2 = torch.clamp((d_z * d_z) * sigma_t2, min=0.0)

    return sigma_Z2, sigma_t2, t, t_delta, t_delta_edge, t_delta_prior

@torch.no_grad()
def sigma_z2_from_gt_z_pixelwise(
    Z_gt: torch.Tensor,
    fx, fy, cx, cy,
    window: int = 5,
    lambda_prior: float = 0.5,
    alpha_clamp: float = 0.3,
    eps: float = 1e-6,
):
    """
    Compute per-pixel z-depth variance sigma_Z^2 from GT z-depth using:
      - local 5x5 robust range (Q95-Q05) to capture edge/aliasing mixture
      - mild footprint prior: t_delta_prior = lambda_prior * tan(theta) * t

    Args:
        Z_gt: (B,1,H,W) GT z-depth in camera coordinates (positive values)
        fx, fy, cx, cy: Camera intrinsics (can be scalars or tensors of shape (B,))
        window: Size of local window for spatial variance (default: 5)
        lambda_prior: Weight for depth-dependent prior (default: 0.5)
        alpha_clamp: Clamp t_delta to alpha_clamp * t to prevent explosion (default: 0.3)
        eps: Small value for numerical stability

    Steps:
      1) d_z from intrinsics
      2) t = Z / d_z
      3) local quantile range in 5x5 on t: t_delta_edge = 0.5*(Q95-Q05)
      4) tan(theta) ~ sqrt((0.5/fx)^2 + (0.5/fy)^2)
         t_delta_prior = lambda_prior * tan(theta) * t
      5) t_delta = sqrt(edge^2 + prior^2)
         clamp t_delta <= alpha_clamp * t to prevent blow-up at occlusions
      6) mip-NeRF sigma_t^2 from (t, t_delta)
      7) sigma_Z^2 = d_z^2 * sigma_t^2

    Returns:
      sigma_Z2: (B,1,H,W) depth variance
      sigma_t2: (B,1,H,W) ray distance variance
      t_mu: (B,1,H,W) ray distance (=t)
      t_delta: (B,1,H,W) interval half-width
    """
    assert Z_gt.dim() == 4 and Z_gt.size(1) == 1, "Z_gt must be (B,1,H,W)"
    B, _, H, W = Z_gt.shape
    device, dtype = Z_gt.device, Z_gt.dtype
    assert window == 5, "This implementation assumes window=5 (can generalize if needed)."

    # 1) d_z and t
    d_z = ray_dir_z(H, W, fx, fy, cx, cy, device, dtype, B)  # (B,1,H,W)
    Z = torch.clamp(Z_gt, min=eps)
    t = Z / (d_z + eps)  # ray distance

    # 2) local quantiles on t via unfold + sort
    k = window
    pad = k // 2
    patches = F.unfold(t, kernel_size=k, padding=pad, stride=1)  # (B, k*k, H*W) with k*k=25
    patches_sorted, _ = torch.sort(patches, dim=1)               # sort 25 values

    # Q05 and Q95 for n=25 -> indices around 1 and 23 (0-based), mild robustification
    q05 = patches_sorted[:, 1, :]   # (B, H*W)
    q95 = patches_sorted[:, 23, :]  # (B, H*W)

    t_delta_edge = 0.5 * (q95 - q05)  # (B, H*W)
    t_delta_edge = t_delta_edge.view(B, 1, H, W)
    t_delta_edge = torch.clamp(t_delta_edge, min=0.0)

    # 3) mild footprint prior: tan(theta) from intrinsics (per-batch allowed)
    fx_t = _as_broadcast_intr(fx, device, dtype, B)  # (B,1,1)
    fy_t = _as_broadcast_intr(fy, device, dtype, B)
    tan_theta = torch.sqrt((0.5 / (fx_t + eps))**2 + (0.5 / (fy_t + eps))**2)  # (B,1,1)
    tan_theta = tan_theta.view(B, 1, 1, 1)
    t_delta_prior = lambda_prior * tan_theta * t

    # 4) combine and clamp
    t_delta = torch.sqrt(t_delta_edge * t_delta_edge + t_delta_prior * t_delta_prior + eps)
    if alpha_clamp is not None and alpha_clamp > 0:
        t_delta = torch.minimum(t_delta, alpha_clamp * t)

    # 5) mip-NeRF sigma_t^2 and map back
    sigma_t2 = mipnerf_sigma_t2_from_t_mu_delta(t, t_delta, eps=eps)
    sigma_Z2 = (d_z * d_z) * sigma_t2
    sigma_Z2 = torch.clamp(sigma_Z2, min=0.0)

    return sigma_Z2, sigma_t2, t, t_delta

def weighted_l1_from_sigma_z2(
    Z_pred: torch.Tensor,
    Z_gt: torch.Tensor,
    sigma_Z2: torch.Tensor,
    eps: float = 1e-6,
    detach_sigma: bool = True,
    sigma_std_min: float = 1.0,
):
    """
    Simple weighted L1:
      w = 1 / (sqrt(sigma_Z2) + eps)
      loss = mean( w * |Z_pred - Z_gt| )

    This is not a full likelihood, but works as a practical weighting.

    Args:
        Z_pred: Predicted depth
        Z_gt: Ground truth depth
        sigma_Z2: Depth variance
        eps: Small value for numerical stability
        detach_sigma: Whether to detach sigma_Z2 from gradient computation
        sigma_std_min: Minimum value for standard deviation (default: 1.0) to prevent extremely large weights
    """
    assert Z_pred.shape == Z_gt.shape == sigma_Z2.shape
    if detach_sigma:
        sigma_Z2 = sigma_Z2.detach()
    sigma_std = torch.sqrt(sigma_Z2 + eps)
    # Clip standard deviation to minimum value to prevent extremely large weights
    sigma_std = torch.clamp(sigma_std, min=sigma_std_min)
    w = 1.0 / (sigma_std + eps)
    return (w * (Z_pred - Z_gt).abs()).mean()

def laplace_nll_from_sigma_z2(
    Z_pred: torch.Tensor,
    Z_gt: torch.Tensor,
    sigma_Z2: torch.Tensor,
    eps: float = 1e-6,
    b_min: float = 1e-3,
    detach_sigma: bool = True,
    sigma_std_min: float = 1.0,
):
    """
    Probabilistic version (heteroscedastic Laplace NLL) using variance prior.

    Laplace variance: Var = 2*b^2  =>  b = sqrt(Var/2)
    NLL (ignoring constant log 2):
      L = |r|/b + log b

    Args:
        Z_pred: Predicted depth
        Z_gt: Ground truth depth
        sigma_Z2: Depth variance
        eps: Small value for numerical stability
        b_min: Minimum value for b to prevent division by zero (legacy, use sigma_std_min instead)
        detach_sigma: Whether to detach sigma_Z2 from gradient computation
        sigma_std_min: Minimum value for standard deviation (default: 1.0) to prevent extremely large weights
    """
    assert Z_pred.shape == Z_gt.shape == sigma_Z2.shape
    if detach_sigma:
        sigma_Z2 = sigma_Z2.detach()
    # Clip standard deviation first
    sigma_std = torch.sqrt(sigma_Z2 + eps)
    sigma_std = torch.clamp(sigma_std, min=sigma_std_min)
    # Convert to Laplace parameter b = sigma/sqrt(2)
    b = sigma_std / torch.sqrt(torch.tensor(2.0, device=sigma_std.device, dtype=sigma_std.dtype))
    b = torch.clamp(b, min=b_min)  # Keep legacy b_min for extra safety
    r = (Z_pred - Z_gt).abs()
    return (r / b + torch.log(b)).mean()

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    B, H, W = 2, 240, 320
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Z_gt = torch.ones(B,1,H,W, device=device) * 2.0
    Z_pred = Z_gt + 0.05 * torch.randn_like(Z_gt)

    fx, fy, cx, cy = 500.0, 500.0, (W-1)/2.0, (H-1)/2.0

    sigma_Z2, sigma_t2, t, t_delta = sigma_z2_from_gt_z_pixelwise(
        Z_gt, fx, fy, cx, cy, window=5, lambda_prior=0.5, alpha_clamp=0.3
    )

    loss_wl1 = weighted_l1_from_sigma_z2(Z_pred, Z_gt, sigma_Z2)
    loss_laplace = laplace_nll_from_sigma_z2(Z_pred, Z_gt, sigma_Z2)

    print(f"Weighted L1 loss: {loss_wl1.item():.6f}")
    print(f"Laplace NLL loss: {loss_laplace.item():.6f}")
    print(f"Sigma_Z2 stats - mean: {sigma_Z2.mean():.6f}, std: {sigma_Z2.std():.6f}")
