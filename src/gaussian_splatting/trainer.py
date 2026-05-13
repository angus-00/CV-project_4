"""
3D Gaussian Splatting Trainer with densification and pruning
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from tqdm import tqdm
import json

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.renderer import GaussianRenderer
from common.camera import Camera
from common.metrics import compute_psnr, compute_ssim, compute_lpips


class GaussianTrainer:
    """Trainer for 3D Gaussian Splatting with adaptive density control"""

    def __init__(
        self,
        model: GaussianModel,
        renderer: GaussianRenderer,
        lr_xyz: float = 0.00016,
        lr_features: float = 0.0025,
        lr_opacity: float = 0.05,
        lr_scaling: float = 0.005,
        lr_rotation: float = 0.001,
        device: str = 'cuda',
        confidence_maps: Optional[List[np.ndarray]] = None,
        view_weights: Optional[List[float]] = None,
        pixel_masks: Optional[List[np.ndarray]] = None,   # per-view uint8 mask (255=compute loss, 0=skip)
        depth_loss_fn: Optional[Callable] = None,          # callback(camera, model, renderer, iteration, idx) -> loss tensor or None
    ):
        self.model = model
        self.renderer = renderer
        self.device = device
        self.confidence_maps = confidence_maps
        self.view_weights    = view_weights
        self.pixel_masks     = pixel_masks   # if set, only compute loss where mask > 0
        self.depth_loss_fn   = depth_loss_fn

        # Move model to device
        self.model.to(device)

        # Setup optimizers
        self.optimizers = self._setup_optimizers(
            lr_xyz, lr_features, lr_opacity, lr_scaling, lr_rotation
        )

        # Densification parameters
        self.densify_grad_threshold = 0.0002
        self.densify_from_iter = 500
        self.densify_until_iter = 10000  # stop densifying at 10k, let last 5k iters converge
        self.densify_interval = 100
        self.opacity_reset_interval = 3000
        self.max_gaussians = 100000  # hard cap to prevent OOM

        # Tracking for densification
        self.xyz_gradient_accum = None
        self.denom = None
        self.max_radii2D = None

    def _setup_optimizers(self, lr_xyz, lr_features, lr_opacity, lr_scaling, lr_rotation):
        """Setup parameter-specific optimizers"""
        params = [
            {'params': [self.model._xyz], 'lr': lr_xyz, 'name': 'xyz'},
            {'params': [self.model._features_dc], 'lr': lr_features, 'name': 'f_dc'},
            {'params': [self.model._features_rest], 'lr': lr_features / 20.0, 'name': 'f_rest'},
            {'params': [self.model._opacity], 'lr': lr_opacity, 'name': 'opacity'},
            {'params': [self.model._scaling], 'lr': lr_scaling, 'name': 'scaling'},
            {'params': [self.model._rotation], 'lr': lr_rotation, 'name': 'rotation'}
        ]
        return torch.optim.Adam(params, lr=0.0, eps=1e-15)

    def train(
        self,
        cameras: List[Camera],
        images: List[np.ndarray],
        iterations: int = 30000,
        test_cameras: Optional[List[Camera]] = None,
        test_images: Optional[List[np.ndarray]] = None,
        save_path: Optional[Path] = None,
        log_interval: int = 100,
        eval_interval: int = 1000
    ) -> Dict:
        """Train 3DGS model with adaptive densification

        Args:
            cameras: List of training cameras
            images: List of training images (numpy arrays)
            iterations: Number of training iterations
            test_cameras: Optional test cameras for evaluation
            test_images: Optional test images
            save_path: Optional path to save model
            log_interval: Interval for logging training metrics
            eval_interval: Interval for evaluation on test set

        Returns:
            Dictionary with training results and metrics
        """
        # Convert images to tensors
        train_images = []
        for img in images:
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).float().to(self.device)
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(-1).repeat(1, 1, 3)
            train_images.append(img_tensor)

        # Initialize tracking for densification
        self.xyz_gradient_accum = torch.zeros((self.model._xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.model._xyz.shape[0], 1), device=self.device)

        metrics_history = {
            'loss': [],
            'psnr': [],
            'num_gaussians': [],
            'iteration': []
        }
        best_psnr = 0.0

        # Training loop
        pbar = tqdm(range(iterations), desc="Training 3DGS")
        for iteration in pbar:
            # Sample random camera
            idx = np.random.randint(0, len(cameras))
            camera = cameras[idx]
            gt_image = train_images[idx]

            # Render
            try:
                rendered = self.renderer.render(camera, self.model)

                # Compute loss (confidence-weighted if available)
                if self.confidence_maps is not None and idx < len(self.confidence_maps):
                    conf = torch.from_numpy(self.confidence_maps[idx]).float().to(self.device)
                    if conf.dim() == 2:
                        conf = conf.unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
                    else:
                        conf = conf.permute(2, 0, 1).unsqueeze(0)
                    rH, rW = rendered.shape[0], rendered.shape[1]
                    conf = F.interpolate(conf, size=(rH, rW), mode='bilinear', align_corners=False)
                    conf = conf.squeeze(0).permute(1, 2, 0)  # H x W x 1
                    if gt_image.shape[0] != rH or gt_image.shape[1] != rW:
                        gt_r = gt_image.permute(2, 0, 1).unsqueeze(0)
                        gt_r = F.interpolate(gt_r, size=(rH, rW), mode='bilinear', align_corners=False)
                        gt_image_loss = gt_r.squeeze(0).permute(1, 2, 0)
                    else:
                        gt_image_loss = gt_image

                    # pixel_mask: only compute loss on masked pixels (e.g. LaMa-inpainted regions)
                    if self.pixel_masks is not None and idx < len(self.pixel_masks) and self.pixel_masks[idx] is not None:
                        pmask = torch.from_numpy((self.pixel_masks[idx] > 0).astype(np.float32)).to(self.device)
                        pmask = pmask.unsqueeze(0).unsqueeze(0)
                        pmask = F.interpolate(pmask, size=(rH, rW), mode='nearest')
                        pmask = pmask.squeeze(0).permute(1, 2, 0)  # H x W x 1
                        weight = conf * pmask
                        denom  = weight.sum() + 1e-8
                        loss   = (F.l1_loss(rendered, gt_image_loss, reduction='none') * weight).sum() / denom
                    else:
                        l1 = (F.l1_loss(rendered, gt_image_loss, reduction='none') * conf).mean()
                        vw = self.view_weights[idx] if (self.view_weights and idx < len(self.view_weights)) else 1.0
                        if vw >= 1.0:
                            loss = 0.6 * l1 + 0.4 * self._ssim_loss(rendered, gt_image_loss)
                        else:
                            loss = l1
                else:
                    rH, rW = rendered.shape[0], rendered.shape[1]
                    if gt_image.shape[0] != rH or gt_image.shape[1] != rW:
                        gt_r = gt_image.permute(2, 0, 1).unsqueeze(0)
                        gt_r = F.interpolate(gt_r, size=(rH, rW), mode='bilinear', align_corners=False)
                        gt_image = gt_r.squeeze(0).permute(1, 2, 0)
                    l1 = F.l1_loss(rendered, gt_image)
                    vw = self.view_weights[idx] if (self.view_weights and idx < len(self.view_weights)) else 1.0
                    if vw >= 1.0:
                        loss = 0.6 * l1 + 0.4 * self._ssim_loss(rendered, gt_image)
                    else:
                        loss = l1

                # Apply per-view scalar weight (1.0 for real views, 0.3 for pseudo-views)
                if self.view_weights is not None and idx < len(self.view_weights):
                    loss = loss * self.view_weights[idx]

                # Backward
                self.optimizers.zero_grad()

                # Opacity × scale regularization: penalise Gaussians that are
                # simultaneously large AND present (opacity > near-zero).
                # These become overexposed "floater" splats at novel viewpoints.
                # Loss = mean(sigmoid(opacity) * max_scale), scaled to be small
                # relative to the photometric loss (weight 1e-4).
                opacity_sig  = torch.sigmoid(self.model._opacity).squeeze(-1)   # (N,)
                scales       = torch.exp(self.model._scaling)                    # (N, 3)
                max_scale    = scales.max(dim=1).values                          # (N,)
                min_scale    = scales.min(dim=1).values                          # (N,)
                reg_loss     = (opacity_sig * max_scale).mean() * 1e-4

                # Aspect ratio regularization: penalise needle-like Gaussians
                # log(max/min) is 0 for spheres, large for elongated splats
                aspect_ratio = torch.log1p(max_scale / (min_scale + 1e-7))
                aspect_reg   = (opacity_sig * aspect_ratio).mean() * 5e-4

                total_loss = loss + reg_loss + aspect_reg

                # Optional depth supervision callback
                if self.depth_loss_fn is not None:
                    d_loss = self.depth_loss_fn(camera, self.model, self.renderer, iteration, idx)
                    if d_loss is not None:
                        total_loss = total_loss + d_loss

                total_loss.backward()

                # Update gradients for densification
                if iteration < self.densify_until_iter:
                    self._update_densification_stats(iteration)

                # Optimizer step
                self.optimizers.step()

                # SH degree scheduling: increase every 1000 iters up to max
                if iteration > 0 and iteration % 1000 == 0:
                    if self.model.active_sh_degree < self.model.max_sh_degree:
                        self.model.active_sh_degree += 1

                # Densification
                if iteration >= self.densify_from_iter and iteration <= self.densify_until_iter:
                    if iteration % self.densify_interval == 0:
                        self._densify_and_prune(iteration)

                # Scale pruning after densification ends (prune giant Gaussians every 1000 iters)
                if iteration > self.densify_until_iter and iteration % 1000 == 0:
                    self._prune_points()

                # Opacity reset (skip iteration 0 to let training start)
                if iteration > 0 and iteration % self.opacity_reset_interval == 0:
                    self._reset_opacity()

                # Memory guard: abort if VRAM exceeds limit
                if iteration % 500 == 0 and torch.cuda.is_available():
                    used_gb = torch.cuda.memory_allocated() / 1e9
                    reserved_gb = torch.cuda.memory_reserved() / 1e9
                    if used_gb > 15.7:
                        print(f"\n[MEM] VRAM alloc={used_gb:.1f}GB reserved={reserved_gb:.1f}GB > 15.7GB limit at iter {iteration}, stopping.")
                        import sys; sys.exit(1)

                # Logging
                if iteration % log_interval == 0:
                    with torch.no_grad():
                        psnr = compute_psnr(rendered, gt_image)
                        metrics_history['loss'].append(loss.item())
                        metrics_history['psnr'].append(psnr)
                        metrics_history['num_gaussians'].append(self.model._xyz.shape[0])
                        metrics_history['iteration'].append(iteration)
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'psnr': f'{psnr:.2f}',
                            'gaussians': self.model._xyz.shape[0]
                        })

                        if psnr > best_psnr:
                            best_psnr = psnr

                # Evaluation
                if test_cameras and test_images and iteration % eval_interval == 0 and iteration > 0:
                    eval_metrics = self.evaluate(test_cameras, test_images)
                    torch.cuda.empty_cache()
                    print(f"\nIteration {iteration} - Test PSNR: {eval_metrics['psnr']:.2f}, "
                          f"SSIM: {eval_metrics['ssim']:.4f}, LPIPS: {eval_metrics['lpips']:.4f}")

            except Exception as e:
                print(f"\nError at iteration {iteration}: {e}")
                continue

        # Final prune: remove any remaining floaters before saving.
        # This runs even when densification is disabled (no_densify mode), ensuring
        # the saved model never contains large-scale, low-opacity Gaussians.
        self._prune_points()
        torch.cuda.empty_cache()

        # Final evaluation
        results = {'metrics': metrics_history, 'best_psnr': best_psnr}
        if test_cameras and test_images:
            results['test_metrics'] = self.evaluate(test_cameras, test_images)

        # Save model
        if save_path:
            self.model.save(save_path)

        return results

    def _update_densification_stats(self, iteration: int):
        """Update statistics for adaptive densification"""
        if self.model._xyz.grad is None:
            return

        # Accumulate gradients
        grad = self.model._xyz.grad
        self.xyz_gradient_accum += grad.norm(dim=-1, keepdim=True)
        self.denom += 1

    def _densify_and_prune(self, iteration: int):
        """Adaptive densification and pruning"""
        # Compute average gradient
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Densify only if under cap
        if self.model._xyz.shape[0] < self.max_gaussians:
            self._densify_and_clone(grads)
            self._densify_and_split(grads)

        # Prune
        self._prune_points()

        # Reset stats
        self.xyz_gradient_accum = torch.zeros((self.model._xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.model._xyz.shape[0], 1), device=self.device)
        torch.cuda.empty_cache()

    def _densify_and_clone(self, grads: torch.Tensor):
        """Clone Gaussians in high-gradient regions"""
        # Select Gaussians with high gradient and small scale
        selected = torch.where(grads.squeeze() >= self.densify_grad_threshold)[0]
        if len(selected) == 0:
            return

        scales = self.model.get_scaling[selected]
        # Use median scale as threshold (small = below median scale)
        median_scale = self.model.get_scaling.max(dim=1).values.median()
        selected = selected[scales.max(dim=1).values <= median_scale]

        if len(selected) > 0:
            self._densify_clone(selected)

    def _densify_and_split(self, grads: torch.Tensor):
        """Split large Gaussians in high-gradient regions"""
        # Select Gaussians with high gradient and large scale
        selected = torch.where(grads.squeeze() >= self.densify_grad_threshold)[0]
        if len(selected) == 0:
            return

        scales = self.model.get_scaling[selected]
        # Use median scale as threshold (large = above median scale)
        median_scale = self.model.get_scaling.max(dim=1).values.median()
        selected = selected[scales.max(dim=1).values > median_scale]

        if len(selected) > 0:
            self._densify_split(selected)

    def _densify_clone(self, selected: torch.Tensor):
        """Clone selected Gaussians"""
        new_xyz = self.model._xyz[selected]
        new_features_dc = self.model._features_dc[selected]
        new_features_rest = self.model._features_rest[selected]
        new_opacity = self.model._opacity[selected]
        new_scaling = self.model._scaling[selected]
        new_rotation = self.model._rotation[selected]

        self._concat_gaussians(new_xyz, new_features_dc, new_features_rest,
                              new_opacity, new_scaling, new_rotation)

    def _densify_split(self, selected: torch.Tensor):
        """Split selected Gaussians into two"""
        # Sample new positions
        stds = self.model.get_scaling[selected]
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)

        new_xyz = self.model._xyz[selected] + samples
        new_features_dc = self.model._features_dc[selected]
        new_features_rest = self.model._features_rest[selected]
        new_opacity = self.model._opacity[selected]
        new_scaling = self.model._scaling[selected] - torch.log(torch.tensor(1.6))
        new_rotation = self.model._rotation[selected]

        self._concat_gaussians(new_xyz, new_features_dc, new_features_rest,
                              new_opacity, new_scaling, new_rotation)

    def _concat_gaussians(self, xyz, features_dc, features_rest, opacity, scaling, rotation):
        """Concatenate new Gaussians to the model"""
        self.model._xyz = torch.nn.Parameter(torch.cat([self.model._xyz.data, xyz.data], dim=0))
        self.model._features_dc = torch.nn.Parameter(torch.cat([self.model._features_dc.data, features_dc.data], dim=0))
        self.model._features_rest = torch.nn.Parameter(torch.cat([self.model._features_rest.data, features_rest.data], dim=0))
        self.model._opacity = torch.nn.Parameter(torch.cat([self.model._opacity.data, opacity.data], dim=0))
        self.model._scaling = torch.nn.Parameter(torch.cat([self.model._scaling.data, scaling.data], dim=0))
        self.model._rotation = torch.nn.Parameter(torch.cat([self.model._rotation.data, rotation.data], dim=0))

        self._rebuild_optimizer()

    def _prune_points(self):
        """Prune Gaussians with low opacity or excessive scale."""
        opacity = self.model.get_opacity.squeeze()

        # 1) Near-transparent: standard opacity threshold
        low_opacity_mask = opacity < 0.005

        # 2) Absolute scale too large (covers entire image)
        max_scale = self.model.get_scaling.max(dim=1).values
        abs_scale_mask = max_scale > 3.0

        # 3) Large scale + low opacity: the floater pattern.
        #    Only target the top-1% by scale (very conservative) with near-zero opacity,
        #    to avoid removing Gaussians that are genuinely contributing to the scene.
        scale_p99      = torch.quantile(max_scale, 0.99)
        floater_mask   = (max_scale > scale_p99) & (opacity < 0.02)

        prune_mask = low_opacity_mask | abs_scale_mask | floater_mask
        if prune_mask.sum() > 0:
            self._prune_gaussians(prune_mask)

    def _prune_gaussians(self, mask: torch.Tensor):
        """Remove Gaussians according to mask"""
        valid_mask = ~mask

        self.model._xyz = torch.nn.Parameter(self.model._xyz.data[valid_mask])
        self.model._features_dc = torch.nn.Parameter(self.model._features_dc.data[valid_mask])
        self.model._features_rest = torch.nn.Parameter(self.model._features_rest.data[valid_mask])
        self.model._opacity = torch.nn.Parameter(self.model._opacity.data[valid_mask])
        self.model._scaling = torch.nn.Parameter(self.model._scaling.data[valid_mask])
        self.model._rotation = torch.nn.Parameter(self.model._rotation.data[valid_mask])

        self._rebuild_optimizer()

    def _reset_opacity(self):
        """Reset opacity to avoid saturation"""
        opacities_new = torch.min(self.model.get_opacity, torch.ones_like(self.model.get_opacity) * 0.01)
        self.model._opacity = torch.nn.Parameter(torch.logit(opacities_new))

    def _ssim_loss(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Differentiable SSIM loss (returns 1 - SSIM as a scalar tensor).
        Inputs: (H, W, 3) float tensors in [0, 1].
        """
        x = img1.permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)
        y = img2.permute(2, 0, 1).unsqueeze(0)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = window_size // 2
        mu1 = F.avg_pool2d(x, window_size, stride=1, padding=pad)
        mu2 = F.avg_pool2d(y, window_size, stride=1, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu2_sq
        sigma12   = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu1_mu2
        ssim_map  = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()

    def _rebuild_optimizer(self):
        """Rebuild optimizer with current parameters"""
        params = [
            {'params': [self.model._xyz], 'lr': self.optimizers.param_groups[0]['lr'], 'name': 'xyz'},
            {'params': [self.model._features_dc], 'lr': self.optimizers.param_groups[1]['lr'], 'name': 'f_dc'},
            {'params': [self.model._features_rest], 'lr': self.optimizers.param_groups[2]['lr'], 'name': 'f_rest'},
            {'params': [self.model._opacity], 'lr': self.optimizers.param_groups[3]['lr'], 'name': 'opacity'},
            {'params': [self.model._scaling], 'lr': self.optimizers.param_groups[4]['lr'], 'name': 'scaling'},
            {'params': [self.model._rotation], 'lr': self.optimizers.param_groups[5]['lr'], 'name': 'rotation'}
        ]
        self.optimizers = torch.optim.Adam(params, lr=0.0, eps=1e-15)

    def evaluate(self, cameras: List[Camera], images: List[np.ndarray]) -> Dict:
        """Evaluate on test set"""
        self.model.eval()
        metrics = {'psnr': [], 'ssim': [], 'lpips': []}

        with torch.no_grad():
            for camera, img in zip(cameras, images):
                # Convert image to tensor
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                gt = torch.from_numpy(img).float().to(self.device)
                if gt.dim() == 2:
                    gt = gt.unsqueeze(-1).repeat(1, 1, 3)

                # Render
                try:
                    rendered = self.renderer.render(camera, self.model)

                    # Compute metrics
                    metrics['psnr'].append(compute_psnr(rendered, gt))
                    metrics['ssim'].append(compute_ssim(rendered, gt))
                    metrics['lpips'].append(compute_lpips(rendered, gt))
                except Exception as e:
                    print(f"Evaluation error: {e}")
                    continue

        return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}
