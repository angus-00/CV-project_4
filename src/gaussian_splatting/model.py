"""3D Gaussian Splatting Model Wrapper"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


class GaussianModel:
    """Wrapper for 3D Gaussian Splatting model

    This is a simplified interface that can work with existing 3DGS implementations
    like gaussian-splatting or gsplat libraries.
    """

    def __init__(self, sh_degree: int = 3):
        self.sh_degree = sh_degree
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # Gaussian parameters (will be initialized during training)
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def create_from_pcd(self, points: np.ndarray, colors: np.ndarray):
        """Initialize Gaussians from point cloud

        Args:
            points: (N, 3) array of 3D points
            colors: (N, 3) array of RGB colors [0, 1]
        """
        n_points = points.shape[0]

        # Use nn.Parameter for all trainable parameters
        self._xyz = torch.nn.Parameter(torch.tensor(points, dtype=torch.float32))
        self._features_dc = torch.nn.Parameter(torch.tensor(colors, dtype=torch.float32).unsqueeze(1))
        self._features_rest = torch.nn.Parameter(torch.zeros((n_points, (self.max_sh_degree + 1) ** 2 - 1, 3)))

        # Initialize scaling, rotation, opacity
        dist = torch.clamp_min(self._compute_nearest_distances(self._xyz), 0.0000001)
        self._scaling = torch.nn.Parameter(torch.log(dist.unsqueeze(-1).repeat(1, 3)))
        self._rotation = torch.nn.Parameter(torch.zeros((n_points, 4)))
        self._rotation.data[:, 0] = 1.0  # Identity quaternion
        self._opacity = torch.nn.Parameter(torch.logit(torch.ones((n_points, 1)) * 0.1))

    def _compute_nearest_distances(self, points: torch.Tensor) -> torch.Tensor:
        """Compute mean k-nearest distance for each point to estimate initial scale."""
        pts = points.detach().float()
        n = pts.shape[0]
        if n <= 3:
            return torch.ones(n) * 0.1

        # Subsample for large point clouds to avoid OOM
        MAX_N = 2000
        if n > MAX_N:
            idx = torch.randperm(n)[:MAX_N]
            pts_sub = pts[idx]
        else:
            idx = None
            pts_sub = pts

        # Compute pairwise distances on subsampled set
        # d2[i, j] = ||pts_sub[i] - pts_sub[j]||^2
        d2 = torch.cdist(pts_sub, pts_sub)  # (m, m)
        # Set diagonal to inf
        d2.fill_diagonal_(float('inf'))
        # Take min distance (k=1 NN)
        min_d = d2.min(dim=1).values  # (m,)

        if idx is not None:
            # Expand back: for each original point, find distance to closest subsampled point
            d_all = torch.cdist(pts, pts_sub)
            d_all.fill_diagonal_(float('inf')) if n == MAX_N else None
            min_d_all = d_all.min(dim=1).values
            return torch.clamp(min_d_all, min=1e-6)

        return torch.clamp(min_d, min=1e-6)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    def to(self, device):
        """Move model to device, preserving nn.Parameter status"""
        if self._xyz is not None:
            self._xyz = torch.nn.Parameter(self._xyz.data.to(device))
        if self._features_dc is not None:
            self._features_dc = torch.nn.Parameter(self._features_dc.data.to(device))
        if self._features_rest is not None:
            self._features_rest = torch.nn.Parameter(self._features_rest.data.to(device))
        if self._scaling is not None:
            self._scaling = torch.nn.Parameter(self._scaling.data.to(device))
        if self._rotation is not None:
            self._rotation = torch.nn.Parameter(self._rotation.data.to(device))
        if self._opacity is not None:
            self._opacity = torch.nn.Parameter(self._opacity.data.to(device))
        return self

    def eval(self):
        """Set model to evaluation mode"""
        # For compatibility with PyTorch models
        return self

    def save(self, path: Path):
        """Save model parameters"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            'xyz': self._xyz.detach().cpu().numpy() if self._xyz is not None else None,
            'features_dc': self._features_dc.detach().cpu().numpy() if self._features_dc is not None else None,
            'features_rest': self._features_rest.detach().cpu().numpy() if self._features_rest is not None else None,
            'scaling': self._scaling.detach().cpu().numpy() if self._scaling is not None else None,
            'rotation': self._rotation.detach().cpu().numpy() if self._rotation is not None else None,
            'opacity': self._opacity.detach().cpu().numpy() if self._opacity is not None else None,
            'sh_degree': self.sh_degree,
            'active_sh_degree': self.active_sh_degree
        }

        np.savez(path / 'model.npz', **state)
        print(f"Model saved to: {path / 'model.npz'}")
