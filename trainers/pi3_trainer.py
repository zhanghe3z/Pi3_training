from trainers.base_trainer_accelerate import BaseTrainer
from easydict import EasyDict
import torch
from datasets.base.base_dataset import sample_resolutions
import hydra
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import wandb

from pi3.models.loss import Pi3Loss

class Pi3Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.train_loss)

        # Configuration for depth visualization
        self.viz_interval = cfg.get('viz_interval', 500)  # Log every N steps
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
            loss, details = self.train_loss(output, batch)
        else:
            loss, details = self.test_loss(output, batch)

        return EasyDict(
            loss=loss,
            **details
        )

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

        im1 = axes[1].imshow(depth_gt, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[1].set_title('Ground Truth Depth')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Predicted Depth
        if isinstance(depth_pred, torch.Tensor):
            depth_pred = depth_pred.cpu().detach().numpy()
        im2 = axes[2].imshow(depth_pred, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[2].set_title('Predicted Depth')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig

    def log_depth_visualizations(self, output, batch, step, mode='train'):
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

            # batch_views is a list of views
            # Visualize samples from the first view
            wandb_images = []
            num_views = min(self.num_viz_samples, len(batch_views))

            for view_idx in range(num_views):
                view = batch_views[view_idx]

                # Select first sample from batch (batch_size is the first dimension)
                rgb = view['img'][0]  # (B, 3, H, W) -> (3, H, W)
                depth_gt = view['depthmap'][0]  # (B, H, W) -> (H, W)
                # Select first batch item and corresponding view
                depth_p = depth_pred[0, view_idx] if depth_pred.shape[0] > 0 else depth_pred[0, 0]

                # Create visualization
                fig = self.create_depth_visualization(rgb, depth_gt, depth_p)

                # Convert to wandb Image
                label = view.get('label', 'unknown')
                wandb_images.append(wandb.Image(fig, caption=f"{mode}_view_{view_idx}_{label}"))
                plt.close(fig)

            # Log to wandb
            if len(wandb_images) > 0:
                wandb.log({f"{mode}/depth_visualizations": wandb_images}, step=step)
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
