"""Confidence Fusion and Consistency-Aware Optimization

This module implements advanced features for Part 3:
1. Confidence fusion: Weight pseudo-views based on reliability
2. Consistency-aware optimization: Use optical flow and reprojection error
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import cv2


class ConfidenceFusion:
    """Compute confidence scores for pseudo-views"""

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def compute_confidence(
        self,
        pseudo_view: np.ndarray,
        reference_views: List[np.ndarray],
        method: str = 'rendering_error'
    ) -> np.ndarray:
        """Compute confidence map for a pseudo-view

        Args:
            pseudo_view: Generated pseudo-view
            reference_views: List of reference (real) views
            method: Confidence computation method

        Returns:
            Confidence map (0-1, higher = more confident)
        """
        if method == 'rendering_error':
            return self._confidence_from_rendering_error(pseudo_view, reference_views)
        elif method == 'gradient':
            return self._confidence_from_gradient(pseudo_view)
        elif method == 'combined':
            conf1 = self._confidence_from_rendering_error(pseudo_view, reference_views)
            conf2 = self._confidence_from_gradient(pseudo_view)
            return (conf1 + conf2) / 2
        else:
            raise ValueError(f"Unknown method: {method}")

    def _confidence_from_rendering_error(
        self,
        pseudo_view: np.ndarray,
        reference_views: List[np.ndarray]
    ) -> np.ndarray:
        """Confidence based on similarity to reference views"""
        if len(reference_views) == 0:
            return np.ones((pseudo_view.shape[0], pseudo_view.shape[1]))

        # Compute minimum error to any reference view
        min_error = None

        for ref_view in reference_views:
            # Resize if needed
            if ref_view.shape != pseudo_view.shape:
                ref_view = cv2.resize(
                    ref_view,
                    (pseudo_view.shape[1], pseudo_view.shape[0])
                )

            # Compute L2 error
            error = np.sqrt(
                ((pseudo_view.astype(float) - ref_view.astype(float)) ** 2).sum(axis=2)
            )

            if min_error is None:
                min_error = error
            else:
                min_error = np.minimum(min_error, error)

        # Convert error to confidence (lower error = higher confidence)
        # Normalize to [0, 1]
        min_error = min_error / (min_error.max() + 1e-8)
        confidence = 1.0 - min_error

        # Apply sigmoid to make it smoother
        confidence = 1.0 / (1.0 + np.exp(-10 * (confidence - 0.5)))

        return confidence

    def _confidence_from_gradient(self, image: np.ndarray) -> np.ndarray:
        """Confidence based on gradient magnitude

        High gradients may indicate artifacts or uncertain regions
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)

        # Lower gradient = higher confidence (smooth regions are more reliable)
        confidence = 1.0 - grad_mag

        # Apply threshold
        confidence = np.clip(confidence, 0.3, 1.0)

        return confidence


class ConsistencyOptimizer:
    """Consistency-aware optimization using optical flow and reprojection"""

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def compute_optical_flow_consistency(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        img3: np.ndarray
    ) -> np.ndarray:
        """Compute optical flow consistency for three consecutive frames

        Args:
            img1: Frame t-1
            img2: Frame t (middle frame to evaluate)
            img3: Frame t+1

        Returns:
            Consistency map (0-1, higher = more consistent)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        gray3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY) if len(img3.shape) == 3 else img3

        # Compute forward flow (1 -> 2)
        flow_12 = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Compute forward flow (2 -> 3)
        flow_23 = cv2.calcOpticalFlowFarneback(
            gray2, gray3,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Compute backward flow (2 -> 1)
        flow_21 = cv2.calcOpticalFlowFarneback(
            gray2, gray1,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Check forward-backward consistency
        # flow_12 should be opposite of flow_21
        flow_consistency = np.sqrt(
            (flow_12[:, :, 0] + flow_21[:, :, 0])**2 +
            (flow_12[:, :, 1] + flow_21[:, :, 1])**2
        )

        # Normalize
        flow_consistency = flow_consistency / (flow_consistency.max() + 1e-8)

        # Convert to confidence (lower inconsistency = higher confidence)
        confidence = 1.0 - flow_consistency

        # Apply threshold
        confidence = np.clip(confidence, 0.0, 1.0)

        return confidence

    def compute_reprojection_error(
        self,
        points_3d: np.ndarray,
        camera_params: Dict,
        image: np.ndarray
    ) -> np.ndarray:
        """Compute reprojection error for 3D points

        Args:
            points_3d: 3D points (N, 3)
            camera_params: Camera parameters (intrinsics + extrinsics)
            image: Rendered image

        Returns:
            Error map
        """
        # This is a simplified version
        # In practice, you would project 3D Gaussians to 2D

        # For now, return uniform confidence
        return np.ones((image.shape[0], image.shape[1]))

    def compute_combined_confidence(
        self,
        pseudo_view: np.ndarray,
        prev_view: np.ndarray,
        next_view: np.ndarray,
        reference_views: List[np.ndarray]
    ) -> np.ndarray:
        """Combine multiple confidence measures

        Args:
            pseudo_view: Generated pseudo-view
            prev_view: Previous frame
            next_view: Next frame
            reference_views: Reference (real) views

        Returns:
            Combined confidence map
        """
        # Compute optical flow consistency
        flow_conf = self.compute_optical_flow_consistency(
            prev_view, pseudo_view, next_view
        )

        # Compute rendering-based confidence
        conf_fusion = ConfidenceFusion(self.device)
        render_conf = conf_fusion.compute_confidence(
            pseudo_view, reference_views, method='combined'
        )

        # Combine (weighted average)
        combined = 0.6 * flow_conf + 0.4 * render_conf

        return combined
