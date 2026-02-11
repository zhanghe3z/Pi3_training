from trainers.base_trainer_accelerate import BaseTrainer
from easydict import EasyDict
import torch
import torch.nn.functional as F
from datasets.base.base_dataset import sample_resolutions
import hydra
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import wandb

from pi3.models.loss import Pi3Loss
from pi3.models.mipnerf_variance_loss import sigma_z2_from_gt_z_pixelwise
from pi3.models.lean_variance_loss import lean_depth_moments_and_variance

class Pi3Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.train_loss)

        # Configuration for depth visualization
        self.viz_interval = cfg.get('viz_interval', 50)  # Log every N steps
        self.num_viz_samples = cfg.get('num_viz_samples', 2)  # Number of samples to visualize

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            encoder_params = [param for param in model_.encoder.named_parameters()]
            other_params = [
                (name, param) for name, param in model_.named_parameters()
                if not name.startswith("encoder.") and not '.encoder.' in name
            ]

            print(f'Number of trainable encoder parameters:', sum(p.numel() for _, p in encoder_params if p.requires_grad))
            print(f'Length of trainable others:', sum(p.numel() for _, p in other_params if p.requires_grad))

            def handle_weight_decay(params, weight_decay, lr):
                decay = []
                no_decay = []
                for name, param in params:
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                return [
                    {"params": no_decay, "weight_decay": 0.0, 'lr': lr},
                    {"params": decay, "weight_decay": weight_decay, 'lr': lr},
                ]

            res = []
            res.extend(handle_weight_decay(encoder_params, cfg_optimizer.weight_decay, cfg_optimizer.encoder_lr))
            res.extend(handle_weight_decay(other_params, cfg_optimizer.weight_decay, cfg_optimizer.lr))

            return res
        
        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def before_epoch(self, epoch):
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'batch_sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        

        if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
            self.test_loader.dataset.set_epoch(0, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.test_loader.batch_sampler, 'batch_sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)

        if 'random_reslution' in self.cfg.train and self.cfg.train.random_reslution and self.cfg.train.num_resolution > 0:
            seed = epoch + self.cfg.train.base_seed
            resolutions = sample_resolutions(aspect_ratio_range=self.cfg.train.aspect_ratio_range, pixel_count_range=self.cfg.train.pixel_count_range, patch_size=self.cfg.train.patch_size, num_resolutions=self.cfg.train.num_resolution, seed=seed)
            print('[Pi3 Trainer] Sampled new resolutions:', resolutions)
            datasets = []
            recursive_get_dataset(self.train_loader.dataset, datasets)
            for dataset in datasets:
                dataset._set_resolutions(resolutions)
            
    def forward_batch(self, batch, mode='train'):
        imgs = torch.stack([view['img'] for view in batch], dim=1)
        pred = self.model(imgs)

        return [pred, batch]
    
    def calculate_loss(self, output, batch, mode='train'):
        output, batch = output

        if mode == 'train':
            result = self.train_loss(output, batch)
        else:
            result = self.test_loss(output, batch)

        # Extract loss, details, and scale from Pi3Loss
        if isinstance(result, tuple) and len(result) == 3:
            loss, details, scale = result
            details['depth_scale'] = scale  # Store scale for visualization
        else:
            loss, details = result
            details['depth_scale'] = None

        return EasyDict(
            loss=loss,
            **details
        )

    def align_depth_to_gt(self, depth_pred, depth_gt):
        """
        Align predicted depth to ground truth using scale and shift.
        Solves: depth_aligned = scale * depth_pred + shift
        to minimize ||depth_aligned - depth_gt||^2 over valid pixels.

        Returns:
            depth_aligned: aligned depth map
            scale: computed scale factor
            shift: computed shift value
        """
        # Ensure numpy arrays
        if isinstance(depth_pred, torch.Tensor):
            depth_pred = depth_pred.cpu().detach().numpy()
        if isinstance(depth_gt, torch.Tensor):
            depth_gt = depth_gt.cpu().detach().numpy()

        # Find valid pixels (gt > 0)
        valid_mask = depth_gt > 0

        if not valid_mask.any():
            return depth_pred, 1.0, 0.0

        # Get valid depth values
        pred_valid = depth_pred[valid_mask].flatten()
        gt_valid = depth_gt[valid_mask].flatten()

        # Solve least squares: [pred, 1] * [scale, shift]^T = gt
        # Using closed form solution
        A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
        b = gt_valid

        # Solve: A^T A x = A^T b
        try:
            params = np.linalg.lstsq(A, b, rcond=None)[0]
            scale, shift = params[0], params[1]
        except:
            # Fallback to simple median-based alignment
            scale = np.median(gt_valid / (pred_valid + 1e-8))
            shift = np.median(gt_valid - scale * pred_valid)

        # Apply alignment
        depth_aligned = scale * depth_pred + shift

        return depth_aligned, scale, shift

    def create_depth_visualization(self, rgb, depth_gt, depth_pred, vmin=None, vmax=None):
        """Create a side-by-side visualization of RGB, GT depth, and predicted depth."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # RGB
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().detach().numpy()
        if rgb.shape[0] == 3:  # C, H, W -> H, W, C
            rgb = rgb.transpose(1, 2, 0)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        axes[0].imshow(rgb)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')

        # GT Depth
        if isinstance(depth_gt, torch.Tensor):
            depth_gt = depth_gt.cpu().detach().numpy()
        valid_gt = depth_gt > 0
        if vmin is None:
            vmin = depth_gt[valid_gt].min() if valid_gt.any() else 0
        if vmax is None:
            vmax = depth_gt[valid_gt].max() if valid_gt.any() else 80

        im1 = axes[1].imshow(depth_gt, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='bilinear')
        axes[1].set_title('Ground Truth Depth')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Predicted Depth - Interpolate to GT resolution, then align
        if isinstance(depth_pred, torch.Tensor):
            depth_pred_tensor = depth_pred
            depth_pred = depth_pred.cpu().detach().numpy()
        else:
            depth_pred_tensor = torch.from_numpy(depth_pred)

        # Interpolate predicted depth to match GT resolution using bilinear interpolation
        if depth_pred.shape != depth_gt.shape:
            # Add batch and channel dims if needed
            if depth_pred_tensor.ndim == 2:
                depth_pred_tensor = depth_pred_tensor.unsqueeze(0).unsqueeze(0)
            elif depth_pred_tensor.ndim == 3:
                depth_pred_tensor = depth_pred_tensor.unsqueeze(0)

            depth_pred_tensor = F.interpolate(
                depth_pred_tensor,
                size=depth_gt.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            depth_pred = depth_pred_tensor.squeeze().cpu().numpy()

        # Align predicted depth to GT using scale and shift
        depth_pred_aligned, scale, shift = self.align_depth_to_gt(depth_pred, depth_gt)

        im2 = axes[2].imshow(depth_pred_aligned, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='bilinear')
        axes[2].set_title(f'Predicted Depth (Aligned)\nscale={scale:.3f}, shift={shift:.3f}')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig

    def create_weight_visualization(self, variance_map, depth_gt=None, depth_pred=None, std_min=0.1):
        """
        Create visualization of weight map = 1/std, where std is computed from variance and clipped.

        Weight computation follows the loss function:
        - std = sqrt(variance + eps)
        - std = clip(std, std_min, inf)
        - weight = 1 / (std + eps)

        Args:
            variance_map: Variance map tensor (B, N, H, W) or (H, W) - computed from GT depth
            depth_gt: Optional ground truth depth for comparison
            depth_pred: Optional predicted depth for comparison
            std_min: Minimum value for standard deviation (default: 0.1)
        """
        # Convert to numpy if needed
        if isinstance(variance_map, torch.Tensor):
            variance_map = variance_map.cpu().detach().numpy()

        # Remove extra dimensions
        if variance_map.ndim > 2:
            variance_map = variance_map.squeeze()

        # Compute std from variance
        eps = 1e-6
        std_map = np.sqrt(variance_map + eps)

        # Scale std range [std_min, std_max] to [1, 6] for each image
        std_min_val = std_map.min()
        std_max_val = std_map.max()

        # Linear mapping: std_min -> 1, std_max -> 6
        if std_max_val > std_min_val:
            std_scaled = 1.0 + (std_map - std_min_val) / (std_max_val - std_min_val) * 9.0
        else:
            # If all values are the same, map to middle value
            std_scaled = np.full_like(std_map, 5)

        # Compute weight = 1 / std
        weight_map = 1.0 / std_scaled

        # Calculate statistics
        var_mean = variance_map.mean()
        var_std = variance_map.std()
        var_min = variance_map.min()
        var_max = variance_map.max()

        std_mean = std_map.mean()
        std_std = std_map.std()
        std_min_val = std_map.min()
        std_max = std_map.max()

        weight_mean = weight_map.mean()
        weight_min = weight_map.min()
        weight_max = weight_map.max()

        # Create figure
        num_plots = 1
        if depth_gt is not None:
            num_plots += 1
        if depth_pred is not None:
            num_plots += 1

        fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Plot weight map (1/std)
        im = axes[plot_idx].imshow(weight_map, cmap='turbo', interpolation='bilinear')
        axes[plot_idx].set_title(
            f'Weight Map (1/std) from GT Variance\n'
            f'var: mean={var_mean:.3e}, range=[{var_min:.3e}, {var_max:.3e}]\n'
            f'std: mean={std_mean:.3f}, range=[{std_min_val:.3f}, {std_max:.3f}]\n'
            f'std clipped to [{std_min:.1f}, inf)\n'
            f'weight: mean={weight_mean:.3f}, range=[{weight_min:.3f}, {weight_max:.3f}]'
        )
        axes[plot_idx].axis('off')
        plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1

        # Plot GT Depth if provided
        if depth_gt is not None:
            if isinstance(depth_gt, torch.Tensor):
                depth_gt = depth_gt.cpu().detach().numpy()
            if depth_gt.ndim > 2:
                depth_gt = depth_gt.squeeze()

            valid_gt = depth_gt > 0
            vmin = depth_gt[valid_gt].min() if valid_gt.any() else 0
            vmax = depth_gt[valid_gt].max() if valid_gt.any() else 80

            im_gt = axes[plot_idx].imshow(depth_gt, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='bilinear')
            axes[plot_idx].set_title('GT Depth')
            axes[plot_idx].axis('off')
            plt.colorbar(im_gt, ax=axes[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1

        # Plot Predicted Depth if provided
        if depth_pred is not None:
            if isinstance(depth_pred, torch.Tensor):
                depth_pred_tensor = depth_pred
                depth_pred = depth_pred.cpu().detach().numpy()
            else:
                depth_pred_tensor = torch.from_numpy(depth_pred)

            if depth_pred.ndim > 2:
                depth_pred = depth_pred.squeeze()

            # Interpolate if needed
            if depth_gt is not None and depth_pred.shape != depth_gt.shape:
                if depth_pred_tensor.ndim == 2:
                    depth_pred_tensor = depth_pred_tensor.unsqueeze(0).unsqueeze(0)
                elif depth_pred_tensor.ndim == 3:
                    depth_pred_tensor = depth_pred_tensor.unsqueeze(0)

                depth_pred_tensor = F.interpolate(
                    depth_pred_tensor,
                    size=depth_gt.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                depth_pred = depth_pred_tensor.squeeze().cpu().numpy()

            # Align if GT is available
            if depth_gt is not None:
                depth_pred_aligned, scale, shift = self.align_depth_to_gt(depth_pred, depth_gt)
                valid_gt = depth_gt > 0
                vmin = depth_gt[valid_gt].min() if valid_gt.any() else 0
                vmax = depth_gt[valid_gt].max() if valid_gt.any() else 80

                im_pred = axes[plot_idx].imshow(depth_pred_aligned, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='bilinear')
                axes[plot_idx].set_title(f'Pred Depth (Aligned)\nscale={scale:.3f}')
            else:
                im_pred = axes[plot_idx].imshow(depth_pred, cmap='turbo', interpolation='bilinear')
                axes[plot_idx].set_title('Pred Depth')

            axes[plot_idx].axis('off')
            plt.colorbar(im_pred, ax=axes[plot_idx], fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig

    def create_sigma_z2_visualization(self, depth_gt, variance=None, depth_pred=None, variance_name="σ²_Z", view=None, valid_mask=None):
        """
        Create visualization of depth variance.

        Args:
            depth_gt: Ground truth depth (H, W) or (1, H, W)
            variance: Depth variance (1, 1, H, W) or (H, W). If None, will compute based on loss type
            depth_pred: Optional predicted depth for comparison
            variance_name: Name to display for the variance (e.g., "σ²_Z" or "Var")
            view: Optional view dict for computing variance (needed if variance is None)
            valid_mask: Optional valid mask for lean variance computation
        """
        # Compute variance if not provided
        if variance is None and view is not None:
            # Use depthmap_linear for variance computation instead of nn-interpolated depth
            depth_for_variance = view.get('depthmap_linear', depth_gt)
            if isinstance(depth_for_variance, torch.Tensor) and depth_for_variance.ndim == 3:
                depth_for_variance = depth_for_variance[0]  # (B, H, W) -> (H, W)

            # Get valid mask for variance computation
            if valid_mask is None:
                valid_mask = view.get('valid_mask', None)
                if valid_mask is not None and isinstance(valid_mask, torch.Tensor):
                    if valid_mask.ndim == 3:
                        valid_mask = valid_mask[0]  # (B, H, W) -> (H, W)

            # Detect which variance method to use based on loss type
            loss_class_name = self.train_loss.__class__.__name__
            use_lean_variance = 'LeanMapping' in loss_class_name

            if use_lean_variance:
                # Use Lean Mapping variance (kernel-based)
                variance = self.compute_lean_variance_from_view(view, depth_for_variance, valid_mask)
                variance_name = "Var_lean"
            else:
                # Use mip-NeRF variance (geometry-based)
                variance = self.compute_sigma_z2_from_view(view, depth_for_variance)
                variance_name = "σ²_Z"

        # Convert to numpy if needed
        if isinstance(depth_gt, torch.Tensor):
            depth_gt = depth_gt.cpu().detach().numpy()
        if isinstance(variance, torch.Tensor):
            variance = variance.cpu().detach().numpy()

        # Remove extra dimensions
        if depth_gt.ndim > 2:
            depth_gt = depth_gt.squeeze()
        if variance.ndim > 2:
            # If variance has multiple channels, take the last one (z channel)
            if variance.shape[0] > 1 and variance.ndim == 3:
                variance = variance[-1, :, :]  # Take last channel
            else:
                variance = variance.squeeze()

        # Create figure with 2 or 3 subplots
        num_plots = 3 if depth_pred is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))

        # Plot 1: GT Depth
        valid_gt = depth_gt > 0
        vmin = depth_gt[valid_gt].min() if valid_gt.any() else 0
        vmax = depth_gt[valid_gt].max() if valid_gt.any() else 80

        im1 = axes[0].imshow(depth_gt, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='bilinear')
        axes[0].set_title('GT Depth')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot 2: Variance (Depth Variance)
        # Use log scale for better visualization of variance range
        variance_vis = np.log10(variance + 1e-8)

        # Use turbo colormap for better visual distinction (pseudo-color)
        im2 = axes[1].imshow(variance_vis, cmap='turbo', interpolation='bilinear')

        # Compute statistics on real variance values (not log scale)
        var_mean = variance[valid_gt].mean() if valid_gt.any() else 0
        var_std = variance[valid_gt].std() if valid_gt.any() else 0
        var_min = variance[valid_gt].min() if valid_gt.any() else 0
        var_max = variance[valid_gt].max() if valid_gt.any() else 0

        # Title shows real variance statistics for interpretation
        axes[1].set_title(
            f'log10({variance_name}) - Depth Variance\n'
            f'Real Stats: mean={var_mean:.3e}, std={var_std:.3e}\n'
            f'Range: [{var_min:.3e}, {var_max:.3e}]'
        )
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot 3: Predicted Depth (optional)
        if depth_pred is not None:
            if isinstance(depth_pred, torch.Tensor):
                depth_pred_tensor = depth_pred
                depth_pred = depth_pred.cpu().detach().numpy()
            else:
                depth_pred_tensor = torch.from_numpy(depth_pred)

            if depth_pred.ndim > 2:
                depth_pred = depth_pred.squeeze()
                depth_pred_tensor = depth_pred_tensor.squeeze()

            # Interpolate predicted depth to match GT resolution using bilinear interpolation
            if depth_pred.shape != depth_gt.shape:
                # Add batch and channel dims if needed
                if depth_pred_tensor.ndim == 2:
                    depth_pred_tensor = depth_pred_tensor.unsqueeze(0).unsqueeze(0)
                elif depth_pred_tensor.ndim == 3:
                    depth_pred_tensor = depth_pred_tensor.unsqueeze(0)

                depth_pred_tensor = F.interpolate(
                    depth_pred_tensor,
                    size=depth_gt.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                depth_pred = depth_pred_tensor.squeeze().cpu().numpy()

            # Align predicted depth to GT
            depth_pred_aligned, scale, shift = self.align_depth_to_gt(depth_pred, depth_gt)

            im3 = axes[2].imshow(depth_pred_aligned, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='bilinear')
            axes[2].set_title(f'Pred Depth (Aligned)\nscale={scale:.3f}')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig

    def compute_sigma_z2_from_view(self, view, depth_gt):
        """
        Compute sigma_z2 for a single view using camera intrinsics (mip-NeRF method).

        Args:
            view: Dictionary containing camera_intrinsics
            depth_gt: Ground truth depth tensor (H, W) or (1, H, W)

        Returns:
            sigma_z2: Depth variance tensor (1, 1, H, W)
        """
        try:
            # Get intrinsics from view
            K = view['camera_intrinsics']  # Could be (B, 3, 3) or (3, 3)
            if isinstance(K, np.ndarray):
                K = torch.from_numpy(K).to(depth_gt.device)

            # If batched, select the first sample
            if K.ndim == 3:
                K = K[0]  # (B, 3, 3) -> (3, 3)

            fx = K[0, 0].item()
            fy = K[1, 1].item()
            cx = K[0, 2].item()
            cy = K[1, 2].item()

            # Ensure depth_gt has shape (1, 1, H, W)
            # If depth_gt has 3 channels (x, y, z), extract only z channel
            if depth_gt.ndim == 4 and depth_gt.shape[1] == 3:
                depth_gt = depth_gt[:, -1:, :, :]  # Take last channel (z)
            elif depth_gt.ndim == 3 and depth_gt.shape[0] == 3:
                depth_gt = depth_gt[-1:, :, :]  # Take last channel (z)

            if depth_gt.ndim == 2:
                depth_gt = depth_gt.unsqueeze(0).unsqueeze(0)
            elif depth_gt.ndim == 3:
                depth_gt = depth_gt.unsqueeze(0)

            # Compute sigma_z2
            with torch.no_grad():
                sigma_z2, _, _, _ = sigma_z2_from_gt_z_pixelwise(
                    depth_gt,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    window=5,
                    lambda_prior=0.5,
                    alpha_clamp=0.3
                )

            # If sigma_z2 has multiple channels (x, y, z), extract only the z channel (last one)
            if sigma_z2.ndim == 4 and sigma_z2.shape[1] > 1:
                sigma_z2 = sigma_z2[:, -1:, :, :]  # Take the last channel (z)

            return sigma_z2
        except Exception as e:
            self.log_info(f"Error computing sigma_z2: {e}")
            return None

    def compute_lean_variance_from_view(self, view, depth_gt, valid_mask=None):
        """
        Compute lean mapping variance for a single view using kernel-based moments.

        Args:
            view: Dictionary (not used for lean mapping, kept for interface consistency)
            depth_gt: Ground truth depth tensor (H, W) or (1, H, W)
            valid_mask: Optional boolean mask for valid pixels

        Returns:
            variance: Depth variance tensor (1, 1, H, W)
        """
        try:
            # Ensure depth_gt has shape (1, 1, H, W)
            if depth_gt.ndim == 4 and depth_gt.shape[1] == 3:
                depth_gt = depth_gt[:, -1:, :, :]  # Take last channel (z)
            elif depth_gt.ndim == 3 and depth_gt.shape[0] == 3:
                depth_gt = depth_gt[-1:, :, :]  # Take last channel (z)

            if depth_gt.ndim == 2:
                depth_gt = depth_gt.unsqueeze(0).unsqueeze(0)
            elif depth_gt.ndim == 3:
                depth_gt = depth_gt.unsqueeze(0)

            # Create valid mask if not provided
            if valid_mask is None:
                valid_mask = depth_gt > 0
            else:
                if valid_mask.ndim == 2:
                    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
                elif valid_mask.ndim == 3:
                    valid_mask = valid_mask.unsqueeze(0)

            # Get loss configuration parameters
            # Check if train_loss has the parameters (for Lean Mapping loss)
            kernel_size = getattr(self.train_loss.point_loss if hasattr(self.train_loss, 'point_loss') else self.train_loss, 'kernel_size', 7)
            kernel = getattr(self.train_loss.point_loss if hasattr(self.train_loss, 'point_loss') else self.train_loss, 'kernel', 'gaussian')
            gaussian_sigma = getattr(self.train_loss.point_loss if hasattr(self.train_loss, 'point_loss') else self.train_loss, 'gaussian_sigma', None)
            min_valid_count = getattr(self.train_loss.point_loss if hasattr(self.train_loss, 'point_loss') else self.train_loss, 'min_valid_count', 8)
            prior_rel = getattr(self.train_loss.point_loss if hasattr(self.train_loss, 'point_loss') else self.train_loss, 'prior_rel', 0.1)
            prior_abs = getattr(self.train_loss.point_loss if hasattr(self.train_loss, 'point_loss') else self.train_loss, 'prior_abs', 0.0)

            # Compute variance using Lean mapping
            with torch.no_grad():
                m1, m2, variance, count = lean_depth_moments_and_variance(
                    depth=depth_gt,
                    valid_mask=valid_mask,
                    kernel_size=kernel_size,
                    kernel=kernel,
                    gaussian_sigma=gaussian_sigma,
                    min_valid_count=min_valid_count,
                    prior_rel=prior_rel,
                    prior_abs=prior_abs,
                )

            return variance
        except Exception as e:
            self.log_info(f"Error computing lean variance: {e}")
            import traceback
            self.log_info(traceback.format_exc())
            return None

    def log_depth_visualizations(self, output, batch, step, mode='train', depth_scale=None):
        """Log depth visualizations to wandb."""
        if not self.cfg.log.use_wandb or not self.accelerator.is_main_process:
            return

        try:
            pred, batch_views = output

            # Get predicted depth from local_points (z coordinate)
            # local_points shape: (B, N, H, W, 3)
            if 'local_points' not in pred:
                return

            depth_pred = pred['local_points'][:, :, :, :, 2]  # (B, N, H, W)

            # Apply scale to predicted depth if available
            if depth_scale is not None:
                # depth_scale shape: (B,)
                # Reshape to (B, 1, 1, 1) for broadcasting
                depth_pred = depth_pred * depth_scale.view(-1, 1, 1, 1)

            # Detect which variance method to use based on loss type
            loss_class_name = self.train_loss.__class__.__name__
            use_lean_variance = 'LeanMapping' in loss_class_name

            # Log which method is being used
            if step % (self.viz_interval * 10) == 0:  # Log occasionally
                variance_method = "Lean Mapping (kernel-based)" if use_lean_variance else "mip-NeRF (geometry-based)"
                self.log_info(f"Using {variance_method} variance visualization for {loss_class_name}")

            # batch_views is a list of views
            # Visualize samples from the first view
            wandb_images = []
            wandb_weight_images = []
            num_views = min(self.num_viz_samples, len(batch_views))

            # Get std_min from loss configuration for consistent weight computation
            std_min = getattr(self.train_loss.point_loss if hasattr(self.train_loss, 'point_loss') else self.train_loss, 'std_min', 0.1)

            for view_idx in range(num_views):
                view = batch_views[view_idx]

                # Select first sample from batch (batch_size is the first dimension)
                rgb = view['img'][0]  # (B, 3, H, W) -> (3, H, W)
                depth_gt = view['depthmap'][0]  # (B, H, W) -> (H, W)

                # Use depthmap_linear for variance computation instead of nn-interpolated depth
                depth_linear = view.get('depthmap_linear', None)
                if depth_linear is not None and isinstance(depth_linear, torch.Tensor):
                    depth_linear = depth_linear[0]  # (B, H, W) -> (H, W)
                else:
                    depth_linear = depth_gt  # Fallback to depthmap if depthmap_linear not available

                valid_mask = view.get('valid_mask', None)
                if valid_mask is not None and isinstance(valid_mask, torch.Tensor):
                    valid_mask = valid_mask[0]  # (B, H, W) -> (H, W)

                # Select first batch item and corresponding view
                depth_p = depth_pred[0, view_idx] if depth_pred.shape[0] > 0 else depth_pred[0, 0]

                # Create depth visualization
                fig = self.create_depth_visualization(rgb, depth_gt, depth_p)

                # Convert to wandb Image
                label = view.get('label', 'unknown')
                wandb_images.append(wandb.Image(fig, caption=f"{mode}_view_{view_idx}_{label}"))
                plt.close(fig)

                # Compute variance from GT depth using the appropriate method
                if use_lean_variance:
                    # Use Lean Mapping variance (kernel-based)
                    variance = self.compute_lean_variance_from_view(view, depth_linear, valid_mask)
                    variance_name = "Var_lean"
                    caption_suffix = "lean_weight"
                else:
                    # Use mip-NeRF variance (geometry-based)
                    variance = self.compute_sigma_z2_from_view(view, depth_linear)
                    variance_name = "σ²_Z"
                    caption_suffix = "mipnerf_weight"

                # Visualize weight map (1/std) computed from GT variance
                if variance is not None:
                    fig_weight = self.create_weight_visualization(variance, depth_gt, depth_p, std_min=std_min)
                    wandb_weight_images.append(
                        wandb.Image(fig_weight, caption=f"{mode}_{caption_suffix}_view_{view_idx}_{label}")
                    )
                    plt.close(fig_weight)

            # Log to wandb
            if len(wandb_images) > 0:
                wandb.log({f"{mode}/depth_visualizations": wandb_images}, step=step)
            if len(wandb_weight_images) > 0:
                wandb.log({f"{mode}/weight_visualizations": wandb_weight_images}, step=step)
        except Exception as e:
            self.log_info(f"Error in depth visualization: {e}")
            import traceback
            self.log_info(traceback.format_exc())



def recursive_get_dataset(dataset, res=[]):
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            recursive_get_dataset(ds, res)
    else:
        if hasattr(dataset, 'dataset'):
            res.append(dataset.dataset)
    return res
