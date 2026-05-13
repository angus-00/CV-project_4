"""
3D Gaussian Splatting Renderer using official diff-gaussian-rasterization
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from gaussian_splatting.model import GaussianModel
from common.camera import Camera


class GaussianRenderer:
    """Differentiable renderer for 3D Gaussians using official rasterizer"""

    def __init__(self, device: str = 'cuda', bg_color: Tuple[float, float, float] = (0, 0, 0)):
        self.device = device
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)

    def render(self, camera: Camera, model: GaussianModel,
               return_alpha: bool = False, return_depth: bool = False):
        """Render image from camera viewpoint

        Args:
            camera: Camera object with R, t, K, width, height
            model: GaussianModel with Gaussian parameters
            return_alpha: if True, return (image, alpha) tuple instead of just image
            return_depth: if True, return (image, invdepth) tuple

        Returns:
            Rendered image tensor of shape (H, W, 3), or
            tuple (image (H,W,3), alpha/invdepth (H,W,1)) if return_alpha/return_depth=True
        """
        # Get Gaussian parameters
        means3D = model._xyz
        opacity = model.get_opacity
        scales = model.get_scaling
        rotations = model.get_rotation
        shs = model.get_features

        # Compute view and projection matrices
        viewmatrix = self._get_view_matrix(camera).to(self.device)
        projmatrix = self._get_projection_matrix(camera).to(self.device)
        campos = self._get_camera_center(camera).to(self.device)

        # Compute FOV
        tanfovx = camera.width / (2 * camera.K[0, 0])
        tanfovy = camera.height / (2 * camera.K[1, 1])

        # Setup rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=float(tanfovx),
            tanfovy=float(tanfovy),
            bg=self.bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=model.active_sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

        # Create rasterizer
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize visible Gaussians to image
        # Returns: (rendered_image, radii, rendered_alpha)
        rendered_image, radii, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=torch.zeros_like(means3D, dtype=torch.float32, device=self.device),
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

        # Convert from (3, H, W) to (H, W, 3)
        rendered_image = rendered_image.permute(1, 2, 0)

        if return_alpha or return_depth:
            # rendered_alpha is actually inverse depth from the CUDA rasterizer
            invdepth = rendered_alpha.permute(1, 2, 0)
            return rendered_image, invdepth

        return rendered_image

    def _get_view_matrix(self, camera: Camera) -> torch.Tensor:
        """Compute view matrix from camera parameters

        View matrix transforms from world space to camera space
        """
        R = camera.R  # (3, 3)
        t = camera.t  # (3,)

        # Build 4x4 view matrix
        viewmatrix = torch.eye(4, dtype=torch.float32)
        viewmatrix[:3, :3] = torch.from_numpy(R).float()
        viewmatrix[:3, 3] = torch.from_numpy(t).float()

        return viewmatrix.transpose(0, 1)  # Transpose for column-major order

    def _get_projection_matrix(self, camera: Camera) -> torch.Tensor:
        """Compute projection matrix from camera intrinsics"""
        K = camera.K
        width = camera.width
        height = camera.height

        znear = 0.01
        zfar = 100.0

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Build OpenGL-style projection matrix
        projmatrix = torch.zeros(4, 4, dtype=torch.float32)

        projmatrix[0, 0] = 2.0 * fx / width
        projmatrix[1, 1] = 2.0 * fy / height
        projmatrix[0, 2] = (width - 2.0 * cx) / width
        projmatrix[1, 2] = (2.0 * cy - height) / height
        projmatrix[2, 2] = -(zfar + znear) / (zfar - znear)
        projmatrix[2, 3] = -2.0 * zfar * znear / (zfar - znear)
        projmatrix[3, 2] = -1.0

        return projmatrix.transpose(0, 1)  # Transpose for column-major order

    def _get_camera_center(self, camera: Camera) -> torch.Tensor:
        """Get camera center in world coordinates"""
        R = camera.R
        t = camera.t

        # Camera center: C = -R^T * t
        C = -np.dot(R.T, t)

        return torch.from_numpy(C).float()
