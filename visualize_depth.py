#!/usr/bin/env python3
"""
Visualize depth predictions from Pi3 model on TartanAir Hospital dataset.
"""
import sys
sys.path.append('.')

import torch
import numpy as np
import argparse
from pathlib import Path
from pi3.models.pi3_training import Pi3
from datasets.tartanair_hospital_dataset import TarTanAirHospitalDataset
from datasets.base.transforms import ImgToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

def save_depth_visualization(depth, save_path, vmin=None, vmax=None):
    """Save depth map as a colorized image."""
    plt.figure(figsize=(10, 6))

    # Handle invalid depths
    valid_mask = depth > 0
    if vmin is None and valid_mask.any():
        vmin = depth[valid_mask].min()
    if vmax is None and valid_mask.any():
        vmax = depth[valid_mask].max()

    plt.imshow(depth, cmap='turbo', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Depth (m)')
    plt.title('Depth Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def compute_depth_scale(depth_pred, depth_gt, valid_mask=None):
    """Compute optimal scale to align predicted depth with ground truth."""
    if isinstance(depth_pred, torch.Tensor):
        depth_pred = depth_pred.cpu().numpy()
    if isinstance(depth_gt, torch.Tensor):
        depth_gt = depth_gt.cpu().numpy()

    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)

    if not valid_mask.any():
        return 1.0

    # Compute scale using least squares: scale = sum(gt * pred) / sum(pred^2)
    pred_valid = depth_pred[valid_mask]
    gt_valid = depth_gt[valid_mask]
    scale = np.sum(gt_valid * pred_valid) / (np.sum(pred_valid * pred_valid) + 1e-8)

    return scale

def save_comparison(rgb, depth_gt, depth_pred, save_path, depth_pred_scaled=None):
    """Save side-by-side comparison of RGB, GT depth, predicted depth, and scaled predicted depth."""
    num_cols = 4 if depth_pred_scaled is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(6*num_cols, 6))

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

    # Predicted Depth (unscaled)
    if isinstance(depth_pred, torch.Tensor):
        depth_pred = depth_pred.cpu().numpy()
    pred_valid = depth_pred > 0
    pred_vmin = depth_pred[pred_valid].min() if pred_valid.any() else 0
    pred_vmax = depth_pred[pred_valid].max() if pred_valid.any() else 1

    im2 = axes[2].imshow(depth_pred, cmap='turbo', vmin=pred_vmin, vmax=pred_vmax)
    axes[2].set_title(f'Predicted Depth (unscaled)\nRange: [{pred_vmin:.2f}, {pred_vmax:.2f}]')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Scaled Predicted Depth
    if depth_pred_scaled is not None:
        if isinstance(depth_pred_scaled, torch.Tensor):
            depth_pred_scaled = depth_pred_scaled.cpu().numpy()
        im3 = axes[3].imshow(depth_pred_scaled, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[3].set_title('Predicted Depth (scaled)')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize depth predictions from Pi3 model")
    parser.add_argument("--ckpt_dir", type=str,
                       default='/mnt/localssd/Pi3_training/outputs/pi3_hospital_lowres/ckpts/best_model',
                       help="Path to checkpoint directory (for trained models)")
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                       help="Path to pretrained checkpoint file (e.g., downloaded from web)")
    parser.add_argument("--data_root", type=str,
                       default='/mnt/localssd/tartanair_tools/tartanair_data/hospital',
                       help="Path to TartanAir hospital data")
    parser.add_argument("--output_dir", type=str,
                       default='/mnt/localssd/Pi3_training/outputs/pi3_hospital_lowres',
                       help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default='cuda',
                       help="Device to run inference on")
    parser.add_argument("--frame_num", type=int, default=8,
                       help="Number of frames to use for inference")
    parser.add_argument("--scale_depth", action='store_true',
                       help="Scale predicted depth to match ground truth range")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir) / "depth_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    device = torch.device(args.device)

    if args.pretrained_ckpt is not None:
        # Load from pretrained checkpoint (downloaded from web)
        print(f"Loading pretrained model from {args.pretrained_ckpt}...")
        model = Pi3(
            pos_type='rope100',
            decoder_size='large',
            load_vggt=False,
            freeze_encoder=False,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=args.pretrained_ckpt
        ).to(device).eval()
        print("Pretrained model loaded successfully!")
    else:
        # Load from trained checkpoint directory
        print(f"Loading model from {args.ckpt_dir}...")
        model = Pi3(
            pos_type='rope100',
            decoder_size='large',
            load_vggt=False,
            freeze_encoder=True,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None
        ).to(device).eval()

        ckpt_path = Path(args.ckpt_dir) / "pytorch_model.bin"
        if not ckpt_path.exists():
            print(f"Error: Checkpoint not found at {ckpt_path}")
            return

        weight = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(weight)
        print("Model loaded successfully!")

    # Load dataset
    print(f"Loading dataset from {args.data_root}...")
    dataset = TarTanAirHospitalDataset(
        data_root=args.data_root,
        z_far=80,
        frame_num=args.frame_num,
        resolution=[[518, 336]],
        transform=ImgToTensor,
        mode='test',
        verbose=True
    )
    print(f"Dataset loaded! Total sequences: {len(dataset)}")

    # Run inference on samples
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    num_samples = min(args.num_samples, len(dataset))

    print(f"\nVisualizing {num_samples} samples...")
    all_scales = []  # Track scales for statistics

    for idx in tqdm(range(num_samples)):
        views = dataset[idx]  # dataset returns views list directly

        # Prepare input images: stack all views
        # views contain transformed images (already tensors from ImgToTensor)
        imgs = torch.stack([view['img'] for view in views]).to(device)  # (N, 3, H, W)

        # Run inference
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(imgs[None])  # Add batch dimension -> (1, N, 3, H, W)

        # Get predicted depth from local_points (z coordinate)
        # local_points shape: (B, N, H, W, 3)
        depth_pred = res['local_points'][0, :, :, :, 2].cpu()  # (N, H, W)

        # Save visualizations for each view
        scene_label = views[0]['label']
        for view_idx in range(len(views)):
            view = views[view_idx]
            rgb = view['img']
            depth_gt = view['depthmap']
            depth_p = depth_pred[view_idx].numpy()

            # Compute depth scale if requested
            depth_p_scaled = None
            if args.scale_depth:
                scale = compute_depth_scale(depth_p, depth_gt)
                all_scales.append(scale)
                depth_p_scaled = depth_p * scale

            # Save comparison
            save_name = f"sample_{idx:03d}_view_{view_idx:02d}_{scene_label}_instance_{view['instance']}.png"
            save_path = output_dir / save_name
            save_comparison(rgb, depth_gt, depth_p, save_path, depth_p_scaled)

    print(f"\nVisualization complete! Images saved to: {output_dir}")
    print(f"Total images generated: {num_samples * args.frame_num}")

    if args.scale_depth and all_scales:
        print(f"\nDepth scaling statistics:")
        print(f"  Mean scale: {np.mean(all_scales):.4f}")
        print(f"  Median scale: {np.median(all_scales):.4f}")
        print(f"  Std scale: {np.std(all_scales):.4f}")
        print(f"  Min scale: {np.min(all_scales):.4f}")
        print(f"  Max scale: {np.max(all_scales):.4f}")

if __name__ == '__main__':
    main()
