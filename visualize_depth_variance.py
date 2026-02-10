#!/usr/bin/env python3
"""
Visualize ground truth depth maps and their variance maps from hospital dataset.
Uses lean_depth_moments_and_variance to compute local variance.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.append('.')

from pi3.models.lean_variance_loss import lean_depth_moments_and_variance


def load_hospital_depth(
    data_root='/mnt/localssd/data',
    difficulty='Easy',
    sequence='P000',
    frame_idx=0,
):
    """
    Load a depth map and RGB image from the hospital dataset.

    Args:
        data_root: Root directory of hospital dataset
        difficulty: 'Easy' or 'Hard'
        sequence: Sequence name (e.g., 'P000', 'P001', etc.)
        frame_idx: Frame index

    Returns:
        depth: numpy array of shape (H, W)
        rgb: numpy array of shape (H, W, 3)
    """
    depth_path = os.path.join(
        data_root, difficulty, sequence, 'depth_left',
        f'{frame_idx:06d}_left_depth.npy'
    )
    img_path = os.path.join(
        data_root, difficulty, sequence, 'image_left',
        f'{frame_idx:06d}_left.png'
    )

    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    depth = np.load(depth_path)
    rgb = np.array(Image.open(img_path))

    # Filter invalid depth (same as in dataset)
    depth[depth > 80] = 0

    print(f"Loaded depth from: {depth_path}")
    print(f"Loaded RGB from: {img_path}")
    print(f"Depth shape: {depth.shape}")
    print(f"RGB shape: {rgb.shape}")
    print(f"Depth range: [{depth[depth > 0].min():.3f}, {depth[depth > 0].max():.3f}]")
    print(f"Valid pixels: {(depth > 0).sum()} / {depth.size} ({100 * (depth > 0).sum() / depth.size:.1f}%)")

    return depth, rgb


def compute_variance_map(
    depth,
    kernel_size=7,
    kernel='gaussian',
    gaussian_sigma=None,
    min_valid_count=8,
    prior_rel=0.1,
    prior_abs=0.0,
    padding_mode='replicate',
):
    """
    Compute variance map from depth using lean_depth_moments_and_variance.

    Args:
        depth: numpy array of shape (H, W)
        kernel_size: Size of local window (default: 7)
        kernel: 'box' or 'gaussian' (default: 'gaussian')
        gaussian_sigma: Sigma for Gaussian kernel (default: kernel_size/6)
        min_valid_count: Minimum valid samples in window (default: 8)
        prior_rel: Relative prior for variance (default: 0.05)
        prior_abs: Absolute prior for variance (default: 0.0)
        padding_mode: 'reflect', 'replicate', or 'zeros' (default: 'replicate')

    Returns:
        m1: mean depth (H, W)
        m2: second moment (H, W)
        variance: local variance (H, W)
        count: valid sample count (H, W)
    """
    # Convert to torch tensor
    depth_tensor = torch.from_numpy(depth).float()

    # Create valid mask (depth > 0)
    valid_mask = depth_tensor > 0

    # Compute variance
    m1, m2, var, count = lean_depth_moments_and_variance(
        depth=depth_tensor,
        valid_mask=valid_mask,
        kernel_size=kernel_size,
        kernel=kernel,
        gaussian_sigma=gaussian_sigma,
        padding_mode=padding_mode,
        min_valid_count=min_valid_count,
        prior_rel=prior_rel,
        prior_abs=prior_abs,
    )

    # Convert back to numpy
    m1_np = m1.squeeze().numpy()
    m2_np = m2.squeeze().numpy()
    var_np = var.squeeze().numpy()
    count_np = count.squeeze().numpy()

    valid_mask_np = valid_mask.numpy()

    print(f"\nVariance computation statistics:")
    print(f"  Mean depth (m1) range: [{m1_np[valid_mask_np].min():.3f}, {m1_np[valid_mask_np].max():.3f}]")
    print(f"  Variance range: [{var_np[valid_mask_np].min():.6f}, {var_np[valid_mask_np].max():.6f}]")
    print(f"  Variance mean: {var_np[valid_mask_np].mean():.6f}")
    print(f"  Variance median: {np.median(var_np[valid_mask_np]):.6f}")
    print(f"  Std deviation range: [{np.sqrt(var_np[valid_mask_np]).min():.3f}, {np.sqrt(var_np[valid_mask_np]).max():.3f}]")
    print(f"  Count range: [{count_np.min():.1f}, {count_np.max():.1f}]")

    return m1_np, m2_np, var_np, count_np


def visualize_depth_and_variance(
    rgb,
    depth,
    m1,
    m2,
    variance,
    count,
    save_path=None,
    title_prefix='',
):
    """
    Visualize RGB, depth map and variance map side by side.

    Args:
        rgb: RGB image (H, W, 3)
        depth: Original depth map (H, W)
        m1: Mean depth (H, W)
        m2: Second moment (H, W)
        variance: Variance map (H, W)
        count: Valid count map (H, W)
        save_path: Path to save figure (optional)
        title_prefix: Prefix for title
    """
    valid_mask = depth > 0

    # Create figure with subplots
    fig = plt.figure(figsize=(24, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: RGB, Original Depth, Mean Depth, Variance
    # 1. RGB Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    ax1.set_title(f'{title_prefix}RGB Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Original depth map
    ax2 = fig.add_subplot(gs[0, 1])
    depth_vis = np.copy(depth)
    depth_vis[~valid_mask] = np.nan
    im2 = ax2.imshow(depth_vis, cmap='jet')
    ax2.set_title(f'{title_prefix}Ground Truth Depth Map', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Depth (m)')

    # 3. Mean depth (m1)
    ax3 = fig.add_subplot(gs[0, 2])
    m1_vis = np.copy(m1)
    m1_vis[~valid_mask] = np.nan
    im3 = ax3.imshow(m1_vis, cmap='jet')
    ax3.set_title(f'{title_prefix}Local Mean Depth (M1)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Mean Depth (m)')

    # 4. Variance map
    ax4 = fig.add_subplot(gs[0, 3])
    var_vis = np.copy(variance)
    var_vis[~valid_mask] = np.nan
    im4 = ax4.imshow(var_vis, cmap='hot')
    ax4.set_title(f'{title_prefix}Local Variance Map', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Variance (m²)')

    # Row 2: Standard Deviation, Valid Count, Coefficient of Variation, Log Variance
    # 5. Standard deviation map (sqrt of variance)
    ax5 = fig.add_subplot(gs[1, 0])
    std_vis = np.sqrt(var_vis)
    im5 = ax5.imshow(std_vis, cmap='hot')
    ax5.set_title(f'{title_prefix}Standard Deviation Map', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='Std Dev (m)')

    # 6. Valid count map
    ax6 = fig.add_subplot(gs[1, 1])
    count_vis = np.copy(count)
    count_vis[~valid_mask] = np.nan
    im6 = ax6.imshow(count_vis, cmap='viridis')
    ax6.set_title(f'{title_prefix}Valid Sample Count', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='Count')

    # 7. Coefficient of variation (std / mean)
    ax7 = fig.add_subplot(gs[1, 2])
    cov_vis = np.zeros_like(variance)
    valid_mean = (m1 > 0) & valid_mask
    cov_vis[valid_mean] = std_vis[valid_mean] / m1_vis[valid_mean]
    cov_vis[~valid_mean] = np.nan
    im7 = ax7.imshow(cov_vis, cmap='plasma', vmin=0, vmax=0.3)
    ax7.set_title(f'{title_prefix}Coefficient of Variation (Std/Mean)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, label='CoV')

    # 8. Log variance (for better visualization of low values)
    ax8 = fig.add_subplot(gs[1, 3])
    log_var_vis = np.copy(variance)
    log_var_vis[valid_mask] = np.log10(variance[valid_mask] + 1e-8)
    log_var_vis[~valid_mask] = np.nan
    im8 = ax8.imshow(log_var_vis, cmap='hot')
    ax8.set_title(f'{title_prefix}Log10 Variance', fontsize=12, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04, label='log10(Var)')

    plt.suptitle(f'Depth Variance Analysis - {title_prefix}', fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()

    return fig


def list_available_sequences(data_root):
    """List all available sequences in the hospital dataset."""
    sequences = []
    for difficulty in ['Easy', 'Hard']:
        difficulty_path = os.path.join(data_root, difficulty)
        if not os.path.exists(difficulty_path):
            continue

        trajectory_names = [name for name in os.listdir(difficulty_path)
                          if os.path.isdir(os.path.join(difficulty_path, name))]

        for name in trajectory_names:
            seq_path = os.path.join(difficulty_path, name)
            if (os.path.exists(os.path.join(seq_path, 'image_left')) and
                os.path.exists(os.path.join(seq_path, 'depth_left'))):

                # Count frames
                depth_dir = os.path.join(seq_path, 'depth_left')
                num_frames = len([f for f in os.listdir(depth_dir) if f.endswith('_depth.npy')])
                sequences.append((difficulty, name, num_frames))

    return sorted(sequences)


def visualize_all_data(
    data_root,
    output_dir='visualization',
    difficulties=['Easy'],
    sequences=None,
    max_frames_per_seq=None,
    frame_stride=1,
    kernel_size=7,
    kernel='gaussian',
    gaussian_sigma=None,
    min_valid_count=8,
    prior_rel=0.1,
    prior_abs=0.0,
    padding_mode='replicate',
):
    """
    Visualize all data in the dataset (RGB, depth, variance) in training style.

    Args:
        data_root: Root directory of hospital dataset
        output_dir: Output directory for visualizations (default: 'visualization')
        difficulties: List of difficulties to process (default: ['Easy'])
        sequences: List of sequences to process (default: all available)
        max_frames_per_seq: Maximum frames per sequence (default: all)
        frame_stride: Process every Nth frame (default: 1)
        kernel_size: Kernel size for variance (default: 7, matches training)
        kernel: Kernel type (default: 'gaussian', matches training)
        gaussian_sigma: Gaussian sigma (default: None = kernel_size/6)
        min_valid_count: Minimum valid samples (default: 8, matches training)
        prior_rel: Relative prior (default: 0.1, matches training)
        prior_abs: Absolute prior (default: 0.0, matches training)
        padding_mode: Padding mode (default: 'replicate', matches training)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Batch Visualization - Training Style Parameters")
    print("="*80)
    print(f"Parameters (matching train_hospital_local_points_gt_lean_mapping.sh):")
    print(f"  kernel_size: {kernel_size}")
    print(f"  kernel: {kernel}")
    print(f"  gaussian_sigma: {gaussian_sigma if gaussian_sigma else 'auto (kernel_size/6)'}")
    print(f"  min_valid_count: {min_valid_count}")
    print(f"  prior_rel: {prior_rel}")
    print(f"  prior_abs: {prior_abs}")
    print(f"  padding_mode: {padding_mode}")
    print(f"\nOutput directory: {output_dir}")
    print("="*80)

    # Get all available sequences
    all_sequences = list_available_sequences(data_root)

    # Filter by difficulty
    filtered_sequences = [
        (diff, seq, num_frames)
        for diff, seq, num_frames in all_sequences
        if diff in difficulties
    ]

    # Filter by sequence names if specified
    if sequences is not None:
        filtered_sequences = [
            (diff, seq, num_frames)
            for diff, seq, num_frames in filtered_sequences
            if seq in sequences
        ]

    if not filtered_sequences:
        print("No sequences found matching the criteria!")
        return

    print(f"\nProcessing {len(filtered_sequences)} sequences:")
    for diff, seq, num_frames in filtered_sequences:
        print(f"  {diff:8s} / {seq:10s} - {num_frames:4d} frames")
    print("="*80)

    total_processed = 0

    # Process each sequence
    for seq_idx, (difficulty, sequence, total_frames) in enumerate(filtered_sequences):
        print(f"\n[{seq_idx+1}/{len(filtered_sequences)}] Processing {difficulty}/{sequence}...")

        # Create subdirectory for this sequence
        seq_output_dir = os.path.join(output_dir, difficulty, sequence)
        os.makedirs(seq_output_dir, exist_ok=True)

        # Determine frames to process
        num_frames = total_frames
        if max_frames_per_seq is not None:
            num_frames = min(num_frames, max_frames_per_seq)

        frames_to_process = list(range(0, num_frames, frame_stride))

        print(f"  Processing {len(frames_to_process)} frames (stride={frame_stride})...")

        # Process each frame
        for frame_idx in frames_to_process:
            try:
                # Load data
                depth, rgb = load_hospital_depth(
                    data_root=data_root,
                    difficulty=difficulty,
                    sequence=sequence,
                    frame_idx=frame_idx,
                )

                # Compute variance
                m1, m2, variance, count = compute_variance_map(
                    depth,
                    kernel_size=kernel_size,
                    kernel=kernel,
                    gaussian_sigma=gaussian_sigma,
                    min_valid_count=min_valid_count,
                    prior_rel=prior_rel,
                    prior_abs=prior_abs,
                    padding_mode=padding_mode,
                )

                # Create visualization
                title_prefix = f"{difficulty}/{sequence} Frame {frame_idx:06d}\n"
                save_path = os.path.join(
                    seq_output_dir,
                    f"frame_{frame_idx:06d}.png"
                )

                # Turn off interactive plotting
                plt.ioff()

                visualize_depth_and_variance(
                    rgb=rgb,
                    depth=depth,
                    m1=m1,
                    m2=m2,
                    variance=variance,
                    count=count,
                    save_path=save_path,
                    title_prefix=title_prefix,
                )

                plt.close('all')  # Close figure to free memory

                total_processed += 1

                if (frame_idx + 1) % 10 == 0 or frame_idx == frames_to_process[-1]:
                    print(f"    Processed {frame_idx+1}/{len(frames_to_process)} frames")

            except Exception as e:
                print(f"    Error processing frame {frame_idx}: {e}")
                continue

        print(f"  Completed {difficulty}/{sequence}")

    print("\n" + "="*80)
    print("Batch Visualization Complete!")
    print("="*80)
    print(f"Total frames processed: {total_processed}")
    print(f"Output directory: {output_dir}")
    print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize hospital GT depth maps and variance')
    parser.add_argument('--data_root', type=str,
                        default='/mnt/localssd/data',
                        help='Root directory of hospital dataset')
    parser.add_argument('--difficulty', type=str, default='Easy',
                        choices=['Easy', 'Hard'],
                        help='Difficulty level (for single frame mode)')
    parser.add_argument('--sequence', type=str, default='P000',
                        help='Sequence name (e.g., P000, P001) (for single frame mode)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index (for single frame mode)')
    parser.add_argument('--kernel_size', type=int, default=7,
                        help='Kernel size for variance computation (default: 7, matches training)')
    parser.add_argument('--kernel', type=str, default='gaussian',
                        choices=['box', 'gaussian'],
                        help='Kernel type (default: gaussian, matches training)')
    parser.add_argument('--gaussian_sigma', type=float, default=None,
                        help='Gaussian sigma (default: kernel_size/6)')
    parser.add_argument('--min_valid_count', type=int, default=8,
                        help='Minimum valid samples in window (default: 8, matches training)')
    parser.add_argument('--prior_rel', type=float, default=0.1,
                        help='Relative prior for variance (default: 0.1, matches training)')
    parser.add_argument('--prior_abs', type=float, default=0.0,
                        help='Absolute prior for variance (default: 0.0, matches training)')
    parser.add_argument('--padding_mode', type=str, default='replicate',
                        choices=['reflect', 'replicate', 'zeros'],
                        help='Padding mode (default: replicate)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save visualization (for single frame mode)')
    parser.add_argument('--output_dir', type=str, default='visualization',
                        help='Output directory for batch mode (default: visualization)')
    parser.add_argument('--list_sequences', action='store_true',
                        help='List all available sequences and exit')

    # Batch processing options
    parser.add_argument('--all', action='store_true',
                        help='Visualize all data (batch mode)')
    parser.add_argument('--difficulties', type=str, nargs='+', default=['Easy'],
                        choices=['Easy', 'Hard'],
                        help='Difficulties to process in batch mode (default: Easy)')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Specific sequences to process (default: all)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames per sequence (default: all)')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='Process every Nth frame (default: 1)')

    args = parser.parse_args()

    # List sequences if requested
    if args.list_sequences:
        print("="*80)
        print("Available sequences in hospital dataset:")
        print("="*80)
        sequences = list_available_sequences(args.data_root)
        for difficulty, name, num_frames in sequences:
            print(f"  {difficulty:8s} / {name:10s} - {num_frames:4d} frames")
        print(f"\nTotal: {len(sequences)} sequences")
        return

    # Batch mode: visualize all data
    if args.all:
        visualize_all_data(
            data_root=args.data_root,
            output_dir=args.output_dir,
            difficulties=args.difficulties,
            sequences=args.sequences,
            max_frames_per_seq=args.max_frames,
            frame_stride=args.frame_stride,
            kernel_size=args.kernel_size,
            kernel=args.kernel,
            gaussian_sigma=args.gaussian_sigma,
            min_valid_count=args.min_valid_count,
            prior_rel=args.prior_rel,
            prior_abs=args.prior_abs,
            padding_mode=args.padding_mode,
        )
        return

    # Single frame mode (original behavior)
    # Load depth
    print("="*80)
    print("Loading ground truth depth map and RGB image...")
    print("="*80)
    depth, rgb = load_hospital_depth(
        data_root=args.data_root,
        difficulty=args.difficulty,
        sequence=args.sequence,
        frame_idx=args.frame,
    )

    # Compute variance
    print("\n" + "="*80)
    print("Computing variance map...")
    print("="*80)
    print(f"Parameters:")
    print(f"  kernel_size: {args.kernel_size}")
    print(f"  kernel: {args.kernel}")
    print(f"  gaussian_sigma: {args.gaussian_sigma if args.gaussian_sigma else 'auto (kernel_size/6)'}")
    print(f"  min_valid_count: {args.min_valid_count}")
    print(f"  prior_rel: {args.prior_rel}")
    print(f"  prior_abs: {args.prior_abs}")
    print(f"  padding_mode: {args.padding_mode}")

    m1, m2, variance, count = compute_variance_map(
        depth,
        kernel_size=args.kernel_size,
        kernel=args.kernel,
        gaussian_sigma=args.gaussian_sigma,
        min_valid_count=args.min_valid_count,
        prior_rel=args.prior_rel,
        prior_abs=args.prior_abs,
        padding_mode=args.padding_mode,
    )

    # Visualize
    print("\n" + "="*80)
    print("Creating visualization...")
    print("="*80)

    title_prefix = f"{args.difficulty}/{args.sequence} Frame {args.frame:06d}\n"

    save_path = args.save
    if save_path is None:
        # Save to visualization directory by default
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(
            args.output_dir,
            f"variance_visualization_{args.difficulty}_{args.sequence}_{args.frame:06d}.png"
        )

    visualize_depth_and_variance(
        rgb=rgb,
        depth=depth,
        m1=m1,
        m2=m2,
        variance=variance,
        count=count,
        save_path=save_path,
        title_prefix=title_prefix,
    )

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
