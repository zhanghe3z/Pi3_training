#!/usr/bin/env python3
"""
Visualize depth predictions from Pi3 pretrained model on TartanAir Hospital dataset.
"""
import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from Pi3_training datasets first (before Pi3)
import datasets.tartanair_hospital_dataset
from datasets.tartanair_hospital_dataset import TarTanAirHospitalDataset
from datasets.base.transforms import ImgToTensor

# Add Pi3 eval code path for pretrained model
sys.path.insert(0, '/mnt/localssd/Pi3')
from pi3.models.pi3 import Pi3

def save_comparison(rgb, depth_gt, depth_pred, save_path):
    """Save side-by-side comparison of RGB, GT depth, and predicted depth."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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
        depth_pred = depth_pred.cpu().numpy()
    im2 = axes[2].imshow(depth_pred, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[2].set_title('Predicted Depth (Pi3 Pretrained)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize depth predictions from Pi3 pretrained model")
    parser.add_argument("--model_name", type=str,
                       default='yyfz233/Pi3',
                       help="HuggingFace model name or local path")
    parser.add_argument("--data_root", type=str,
                       default='/mnt/localssd/tartanair_tools/tartanair_data/hospital',
                       help="Path to TartanAir hospital data")
    parser.add_argument("--output_dir", type=str,
                       default='/mnt/localssd/Pi3_training/outputs/pretrained_visualizations',
                       help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default='cuda:0',
                       help="Device to run inference on")
    parser.add_argument("--frame_num", type=int, default=8,
                       help="Number of frames to use for inference")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load pretrained model
    print(f"Loading pretrained Pi3 model from {args.model_name}...")
    device = torch.device(args.device)

    try:
        model = Pi3.from_pretrained(args.model_name).to(device).eval()
        print("Pretrained model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to download from HuggingFace...")
        model = Pi3.from_pretrained(args.model_name, force_download=True).to(device).eval()
        print("Model downloaded and loaded successfully!")

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
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    num_samples = min(args.num_samples, len(dataset))

    print(f"\nVisualizing {num_samples} samples with pretrained model...")
    for idx in tqdm(range(num_samples)):
        views = dataset[idx]  # dataset returns views list directly

        # Prepare input images: stack all views
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

            # Save comparison
            save_name = f"pretrained_sample_{idx:03d}_view_{view_idx:02d}_{scene_label}_instance_{view['instance']}.png"
            save_path = output_dir / save_name
            save_comparison(rgb, depth_gt, depth_p, save_path)

    print(f"\nVisualization complete! Images saved to: {output_dir}")
    print(f"Total images generated: {num_samples * args.frame_num}")

if __name__ == '__main__':
    main()
