import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from .mipnerf_variance_loss import (
    sigma_z2_from_gt_z_pixelwise,
    weighted_l1_from_sigma_z2,
    laplace_nll_from_sigma_z2
)
from .lean_variance_loss import (
    lean_depth_moments_and_variance,
    weighted_l1_from_variance,
    laplace_nll_from_variance
)

from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__

# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

# ---------------------------------------------------------------------------
# Visualization helper function
# ---------------------------------------------------------------------------

def save_depth_comparison(rgb, depth_gt, depth_pred, save_path, variance=None):
    """Save side-by-side comparison of RGB, GT depth, predicted depth, and variance."""
    num_plots = 4 if variance is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    # RGB
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if rgb.shape[0] == 3:  # C, H, W -> H, W, C
        rgb = rgb.transpose(1, 2, 0)
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    # GT Depth
    if isinstance(depth_gt, torch.Tensor):
        depth_gt = depth_gt.cpu().numpy()
    valid_gt = depth_gt > 0
    vmin = depth_gt[valid_gt].min() if valid_gt.any() else 0
    vmax = depth_gt[valid_gt].max() if valid_gt.any() else 80

    im1 = axes[1].imshow(depth_gt, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground Truth Depth')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Predicted Depth
    if isinstance(depth_pred, torch.Tensor):
        depth_pred = depth_pred.detach().cpu().numpy()

    im2 = axes[2].imshow(depth_pred, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[2].set_title('Predicted Depth')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Variance (if provided)
    if variance is not None:
        if isinstance(variance, torch.Tensor):
            variance = variance.detach().cpu().numpy()

        # Scale std range [std_min, std_max] to [1, 6]
        std = np.sqrt(variance + 1e-6)

        # Get valid std values
        valid_std = std[valid_gt] if valid_gt.any() else std
        if valid_std.size > 0:
            std_min_val = valid_std.min()
            std_max_val = valid_std.max()

            # Linear mapping: std_min -> 1, std_max -> 6
            if std_max_val > std_min_val:
                std_scaled = 1.0 + (std - std_min_val) / (std_max_val - std_min_val) * 5.0
            else:
                std_scaled = np.full_like(std, 3.5)
        else:
            std_scaled = np.full_like(std, 3.5)

        weights = 1 / std_scaled

        # Compute weight statistics
        weight_min = weights[valid_gt].min() if valid_gt.any() else 1/6
        weight_max = weights[valid_gt].max() if valid_gt.any() else 1
        weight_mean = weights[valid_gt].mean() if valid_gt.any() else 0

        im3 = axes[3].imshow(weights, cmap='plasma', vmin=1/6, vmax=1)
        axes[3].set_title(f'Weight (mean={weight_mean:.4f})')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

# ---------------------------------------------------------------------------
# ABLATION: Simple L1 PointLoss (No Scale-Invariant Alignment)
# ---------------------------------------------------------------------------

class PointLossAblation(nn.Module):
    """
    Ablation Study: Simple L1 Loss without Scale-Invariant Alignment
    - Removes scale alignment step
    - Directly computes L1 loss between predicted and GT points
    """
    def __init__(self, train_conf=False):
        super().__init__()
        self.criteria_local = nn.L1Loss(reduction='none')
        self.train_conf = train_conf

        if self.train_conf:
            raise NotImplementedError("Confidence training not supported in ablation loss")

    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        mean_depth = weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True)
        # 限制深度范围：最小0.1倍平均深度，最大10倍平均深度，防止远处点衰减太多
        weights_ = weights_.clamp(0.1 * mean_depth.clamp_min(1e-6), 10.0 * mean_depth.clamp_min(1e-6))
        weights_ = 1 / (weights_ + 1e-6)

        # ABLATION: No scale alignment, directly compute L1 loss
        # Using a dummy scale of 1.0 for compatibility with camera loss
        S_opt_local = torch.ones(B, device=pred_local_pts.device)

        # Direct L1 loss without alignment
        local_pts_loss = self.criteria_local(pred_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss (still use it for high quality datasets)
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * pred_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(pred_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']
            pred_global_pts = pred['global_points']
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local


# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose (Keep Original)
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:
            R_err: rotation angular error
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]

    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)

        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)

        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)

        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)

        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]

        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)

        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3),
            R_gt.reshape(-1, 3, 3)
        )

        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

# ---------------------------------------------------------------------------
# Final Loss for Ablation
# ---------------------------------------------------------------------------

class Pi3LossAblation(nn.Module):
    """
    Ablation Loss:
    - Uses simple L1 loss without scale-invariant alignment
    """
    def __init__(
        self,
        train_conf=False,
    ):
        super().__init__()
        self.point_loss = PointLossAblation(train_conf=train_conf)
        self.camera_loss = CameraLoss()

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]

        dataset_names = gt[0]['dataset']

        return dict(
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            dataset_names=dataset_names
        )

    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def forward(self, pred, gt_raw):
        gt = self.prepare_gt(gt_raw)
        pred = self.normalize_pred(pred, gt)

        final_loss = 0.0
        details = dict()

        # Local Point Loss (ABLATION: Simple L1, no scale alignment)
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        final_loss += point_loss
        details.update(point_loss_details)

        # Camera Loss (keep original)
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        final_loss += camera_loss * 0.1
        details.update(camera_loss_details)

        return final_loss, details

# ---------------------------------------------------------------------------
# GT-Only Normalization Loss
# ---------------------------------------------------------------------------

class Pi3LossGTOnlyNorm(nn.Module):
    """
    Loss variant that allows independent control over:
    - Normalizing predicted points
    - Normalizing GT points
    - Using scale alignment in point loss
    """
    def __init__(
        self,
        train_conf=False,
        normalize_pred=True,
        normalize_gt=True,
        use_scale_align=True,
        local_align_res=4096,
    ):
        super().__init__()
        # Import PointLoss from main loss module
        from .loss import PointLoss
        self.point_loss = PointLoss(
            train_conf=train_conf,
            local_align_res=local_align_res,
            use_scale_align=use_scale_align
        )
        self.camera_loss = CameraLoss()
        self.normalize_pred = normalize_pred
        self.normalize_gt = normalize_gt

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # Transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # Optionally normalize GT points
        if self.normalize_gt:
            valid_batch = masks.sum([-1, -2, -3]) > 0
            if valid_batch.sum() > 0:
                B_ = valid_batch.sum()
                all_pts = gt_pts[valid_batch].clone()
                all_pts[~masks[valid_batch]] = 0
                all_pts = all_pts.reshape(B_, N, -1, 3)
                all_dis = all_pts.norm(dim=-1)
                norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

                gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
                poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]

        dataset_names = gt[0]['dataset']

        return dict(
            imgs=torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            dataset_names=dataset_names
        )

    def normalize_prediction(self, pred, gt):
        """Normalize predicted points and camera poses"""
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # Normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def forward(self, pred, gt_raw):
        gt = self.prepare_gt(gt_raw)

        # Optionally normalize predicted points
        if self.normalize_pred:
            pred = self.normalize_prediction(pred, gt)

        final_loss = 0.0
        details = dict()

        # Point Loss (with optional scale alignment)
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        final_loss += point_loss
        details.update(point_loss_details)

        # Camera Loss
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        final_loss += camera_loss * 0.1
        details.update(camera_loss_details)

        return final_loss, details


# ---------------------------------------------------------------------------
# Lean Mapping Variance Loss (Kernel-Based Local Moments)
# ---------------------------------------------------------------------------

class PointLossLeanMapping(nn.Module):
    """
    Lean Mapping / VSM-style variance weighting for depth loss.

    Uses kernel-based local moments to compute variance:
    - M1 = E[z], M2 = E[z^2], Var = M2 - M1^2
    - Supports Gaussian or box kernels
    - Prior variance for low-support regions
    - No scale alignment (similar to other ablation losses)
    """
    def __init__(
        self,
        train_conf=False,
        loss_type='weighted_l1',  # 'weighted_l1' or 'laplace_nll'
        kernel_size=5,
        kernel='gaussian',  # 'gaussian' or 'box'
        gaussian_sigma=None,  # default: kernel_size/6
        min_valid_count=8,
        prior_rel=0.1,
        prior_abs=0.0,
        std_min=0.1,
    ):
        super().__init__()
        self.criteria_local = nn.L1Loss(reduction='none')
        self.train_conf = train_conf
        self.loss_type = loss_type
        self.kernel_size = kernel_size
        self.kernel = kernel
        self.gaussian_sigma = gaussian_sigma
        self.min_valid_count = min_valid_count
        self.prior_rel = prior_rel
        self.prior_abs = prior_abs
        self.std_min = std_min

        if self.train_conf:
            raise NotImplementedError("Confidence training not supported in ablation loss")

    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt, gt_raw=None):
        """
        Args:
            pred: dict with 'local_points' (B, N, H, W, 3)
            gt: dict with 'local_points', 'valid_masks', etc.
            gt_raw: raw GT data with depthmap for each view
        """
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        # ABLATION: No scale alignment
        S_opt_local = torch.ones(B, device=pred_local_pts.device)

        # Extract predicted depth
        pred_depth = pred_local_pts[..., 2]  # (B, N, H, W)

        # Extract GT depth from gt_raw (original depthmap) instead of local_points
        if gt_raw is not None:
            # gt_raw is a list of N view dicts, each with batched tensors
            # Stack all depthmaps: (B, N, H, W)
            # Use nearest-neighbor interpolated depth for GT supervision
            gt_depth = torch.stack([view['depthmap'] for view in gt_raw], dim=1)

            # Use linear-interpolated depth for variance calculation
            gt_depth_linear = torch.stack([view['depthmap_linear'] for view in gt_raw], dim=1)

            # Normalize each depth map by its median (vectorized)
            # Replace invalid values with nan for nanmedian computation
            valid_mask_depth = (gt_depth > 0) & torch.isfinite(gt_depth)
            gt_depth_masked = gt_depth.clone()
            gt_depth_masked[~valid_mask_depth] = float('nan')

            # Compute median along spatial dimensions for each (B, N)
            # Reshape to (B, N, H*W) and compute nanmedian
            B, N, H, W = gt_depth.shape
            depth_flat = gt_depth_masked.reshape(B, N, H*W)
            medians = torch.nanmedian(depth_flat, dim=-1).values  # (B, N)

            # Handle cases where all values are nan (no valid pixels)
            medians = torch.where(torch.isnan(medians), torch.ones_like(medians), medians)

            # Normalize both depths: (B, N, H, W) / (B, N, 1, 1)
            gt_depth_linear = gt_depth_linear / (medians[:, :, None, None] + 1e-6)
        else:
            # Fallback to using local_points if gt_raw is not available
            gt_depth = gt_local_pts[..., 2]  # (B, N, H, W)
            gt_depth_linear = gt_depth.clone()

        # Compute Lean mapping variance for each view independently
        # Reshape to (B*N, 1, H, W) for processing
        # Use linear interpolated depth for variance calculation
        gt_depth_linear_flat = gt_depth_linear.reshape(B*N, 1, H, W)
        valid_masks_flat = valid_masks.reshape(B*N, 1, H, W)

        # Compute variance using Lean mapping with linear interpolated depth
        m1, m2, variance, count = lean_depth_moments_and_variance(
            depth=gt_depth_linear_flat,
            valid_mask=valid_masks_flat,
            kernel_size=self.kernel_size,
            kernel=self.kernel,
            gaussian_sigma=self.gaussian_sigma,
            min_valid_count=self.min_valid_count,
            prior_rel=self.prior_rel,
            prior_abs=self.prior_abs,
        )

        # Reshape back to (B, N, H, W)
        variance = variance.reshape(B, N, H, W)
        m1 = m1.reshape(B, N, H, W)
        count = count.reshape(B, N, H, W)

        # Debug: Check for issues
        num_valid = valid_masks.sum().item()
        if num_valid == 0:
            print(f"[WARNING] No valid pixels! All masks are False.")
            depth_loss = 0.0 * pred_depth.mean()
            xy_loss = 0.0 * pred_depth.mean()
        else:
            # Apply variance-weighted depth loss only on valid pixels
            # NOTE: gt_depth (nearest neighbor) is used for GT supervision
            # while variance is computed from gt_depth_linear (linear interpolation)
            if self.loss_type == 'weighted_l1':
                # Compute weighted L1 for valid pixels
                depth_diff = torch.abs(pred_depth - gt_depth)  # (B, N, H, W)
                std = torch.sqrt(variance + 1e-6)

                # Scale std range [std_min, std_max] to [1, 6] for each image
                # Compute min/max per (B, N) image using matrix operations
                B, N, H, W = std.shape
                std_flat = std.reshape(B, N, -1)  # (B, N, H*W)
                valid_flat = valid_masks.reshape(B, N, -1)  # (B, N, H*W)

                # For min: set invalid to inf, then take min over spatial dimension
                std_for_min = std_flat.clone()
                std_for_min[~valid_flat] = float('inf')
                std_min_vals = std_for_min.min(dim=-1).values  # (B, N)

                # For max: set invalid to -inf, then take max over spatial dimension
                std_for_max = std_flat.clone()
                std_for_max[~valid_flat] = float('-inf')
                std_max_vals = std_for_max.max(dim=-1).values  # (B, N)

                # Reshape for broadcasting: (B, N, 1, 1)
                std_min_vals = std_min_vals.view(B, N, 1, 1)
                std_max_vals = std_max_vals.view(B, N, 1, 1)

                # Linear mapping: std_min -> 1, std_max -> 6
                std_range = std_max_vals - std_min_vals
                std_scaled = torch.where(
                    std_range > 1e-6,
                    1.0 + (std - std_min_vals) / std_range * 9.0,
                    torch.full_like(std, 5)  # If all same, use middle value
                )

                weights = 1 / std_scaled 
                weighted_depth_loss = (weights * depth_diff)[valid_masks]

                # Add statistics to details for wandb logging
                details['variance_min'] = variance[valid_masks].min()
                details['variance_max'] = variance[valid_masks].max()
                details['variance_mean'] = variance[valid_masks].mean()
                details['std_min'] = std[valid_masks].min()
                details['std_max'] = std[valid_masks].max()
                details['std_mean'] = std[valid_masks].mean()
                details['weights_min'] = weights[valid_masks].min()
                details['weights_max'] = weights[valid_masks].max()
                details['weights_mean'] = weights[valid_masks].mean()
                details['depth_diff_mean'] = depth_diff[valid_masks].mean()
                details['depth_diff_max'] = depth_diff[valid_masks].max()
                details['m1_mean'] = m1[valid_masks].mean()
                details['count_mean'] = count[valid_masks].mean()
                details['weighted_depth_loss_mean'] = weighted_depth_loss.mean()

                depth_loss = weighted_depth_loss.mean()
            elif self.loss_type == 'laplace_nll':
                # Laplace NLL for valid pixels
                std = torch.sqrt(variance + 1e-6)
                std = torch.clamp(std, min=self.std_min)
                b = std / torch.sqrt(torch.tensor(2.0, device=std.device, dtype=std.dtype))
                b = b.clamp(min=1e-3)
                depth_diff = torch.abs(pred_depth - gt_depth)
                laplace_loss = (depth_diff / b + torch.log(b))[valid_masks]
                depth_loss = laplace_loss.mean()
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # Also add XY loss (unweighted L1)
            xy_loss = self.criteria_local(
                pred_local_pts[..., :2][valid_masks].float(),
                gt_local_pts[..., :2][valid_masks].float()
            ).mean()

        final_loss += depth_loss + xy_loss
        details['depth_loss'] = depth_loss if torch.is_tensor(depth_loss) else torch.tensor(depth_loss, device=pred_local_pts.device)
        details['xy_loss'] = xy_loss if torch.is_tensor(xy_loss) else torch.tensor(xy_loss, device=pred_local_pts.device)
        details['local_pts_loss'] = details['depth_loss'] + details['xy_loss']

        # normal loss (still use it for high quality datasets)
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss = 0.0 * pred_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(pred_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += 0.0 * normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        return final_loss, details, S_opt_local


class Pi3LossLeanMapping(nn.Module):
    """
    Loss variant using Lean Mapping variance-weighted depth loss:
    - Uses GT-only normalization (normalize_gt=true, normalize_pred=false)
    - Applies Lean Mapping (kernel-based) variance weighting on depth
    - No scale alignment
    """
    def __init__(
        self,
        train_conf=False,
        normalize_pred=False,
        normalize_gt=True,
        loss_type='weighted_l1',  # 'weighted_l1' or 'laplace_nll'
        kernel_size=7,
        kernel='gaussian',  # 'gaussian' or 'box'
        gaussian_sigma=None,
        min_valid_count=8,
        prior_rel=0.1,
        prior_abs=0.0,
        std_min=0.1,
    ):
        super().__init__()
        self.point_loss = PointLossLeanMapping(
            train_conf=train_conf,
            loss_type=loss_type,
            kernel_size=kernel_size,
            kernel=kernel,
            gaussian_sigma=gaussian_sigma,
            min_valid_count=min_valid_count,
            prior_rel=prior_rel,
            prior_abs=prior_abs,
            std_min=std_min,
        )
        self.camera_loss = CameraLoss()
        self.normalize_pred = normalize_pred
        self.normalize_gt = normalize_gt

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # Transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # Optionally normalize GT points
        if self.normalize_gt:
            valid_batch = masks.sum([-1, -2, -3]) > 0
            if valid_batch.sum() > 0:
                B_ = valid_batch.sum()
                all_pts = gt_pts[valid_batch].clone()
                all_pts[~masks[valid_batch]] = 0
                all_pts = all_pts.reshape(B_, N, -1, 3)
                all_dis = all_pts.norm(dim=-1)
                norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

                gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
                poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]

        dataset_names = gt[0]['dataset']

        return dict(
            imgs=torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            dataset_names=dataset_names
        )

    def normalize_prediction(self, pred, gt):
        """Normalize predicted points and camera poses"""
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # Normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def forward(self, pred, gt_raw):
        gt = self.prepare_gt(gt_raw)

        final_loss = 0.0
        details = dict()

        # Point Loss with Lean mapping variance weighting
        point_loss, point_loss_details, scale = self.point_loss(pred, gt, gt_raw=gt_raw)
        final_loss += point_loss
        details.update(point_loss_details)

        # Camera Loss
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        final_loss += camera_loss * 0.1
        details.update(camera_loss_details)

        return final_loss, details
