"""Pseudo-View Generation using Diffusion-based Inpainting

This module implements pseudo-view generation following the BRPO/Difix3D approach:
1. Render intermediate views from Part 2 model
2. Use diffusion inpainting to complete missing/low-quality regions
3. Generate high-quality pseudo-views to fill gaps
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PseudoViewGenerator:
    """Generate pseudo-views using diffusion-based inpainting"""

    def __init__(self, device: str = 'cuda'):
        """Initialize the pseudo-view generator

        Args:
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.inpainting_model = None
        self._load_model()

    def _load_model(self):
        """Load the diffusion inpainting model"""
        try:
            from diffusers import StableDiffusionInpaintPipeline

            print("Loading Stable Diffusion Inpainting model...")
            print("This may take a few minutes on first run...")

            # Use SD 2.0 inpainting model
            model_id = "stabilityai/stable-diffusion-2-inpainting"

            self.inpainting_model = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(self.device)

            # Enable memory optimizations
            self.inpainting_model.enable_attention_slicing()

            print(f"Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Warning: Could not load diffusion model: {e}")
            print("Falling back to simple interpolation method...")
            self.inpainting_model = None

    def generate_intermediate_views(
        self,
        sparse_images: List[np.ndarray],
        sparse_cameras: List[Dict],
        num_intermediate: int = 2
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Generate intermediate views between sparse views

        Args:
            sparse_images: List of sparse view images
            sparse_cameras: List of sparse view camera parameters
            num_intermediate: Number of intermediate views to generate between each pair

        Returns:
            Tuple of (generated_images, generated_cameras)
        """
        generated_images = []
        generated_cameras = []

        print(f"Generating {num_intermediate} intermediate views between each pair...")

        for i in range(len(sparse_images) - 1):
            # Add current sparse view
            generated_images.append(sparse_images[i])
            generated_cameras.append(sparse_cameras[i])

            # Generate intermediate views
            for j in range(num_intermediate):
                alpha = (j + 1) / (num_intermediate + 1)

                # Interpolate camera parameters
                interp_camera = self._interpolate_camera(
                    sparse_cameras[i],
                    sparse_cameras[i + 1],
                    alpha
                )

                # Generate pseudo-view
                if self.inpainting_model is not None:
                    pseudo_view = self._generate_with_inpainting(
                        sparse_images[i],
                        sparse_images[i + 1],
                        alpha
                    )
                else:
                    # Fallback: simple blending
                    pseudo_view = self._simple_blend(
                        sparse_images[i],
                        sparse_images[i + 1],
                        alpha
                    )

                generated_images.append(pseudo_view)
                generated_cameras.append(interp_camera)

        # Add last sparse view
        generated_images.append(sparse_images[-1])
        generated_cameras.append(sparse_cameras[-1])

        return generated_images, generated_cameras

    def _interpolate_camera(
        self,
        cam1: Dict,
        cam2: Dict,
        alpha: float
    ) -> Dict:
        """Interpolate between two cameras

        Args:
            cam1: First camera parameters
            cam2: Second camera parameters
            alpha: Interpolation factor (0 = cam1, 1 = cam2)

        Returns:
            Interpolated camera parameters
        """
        interp_cam = {}

        # Interpolate translation
        if 'tvec' in cam1 and 'tvec' in cam2:
            tvec1 = np.array(cam1['tvec'])
            tvec2 = np.array(cam2['tvec'])
            interp_cam['tvec'] = ((1 - alpha) * tvec1 + alpha * tvec2).tolist()

        # Interpolate rotation (SLERP for quaternions)
        if 'qvec' in cam1 and 'qvec' in cam2:
            qvec1 = np.array(cam1['qvec'])
            qvec2 = np.array(cam2['qvec'])
            interp_cam['qvec'] = self._slerp(qvec1, qvec2, alpha).tolist()

        # Copy intrinsics (assume same for intermediate views)
        for key in ['width', 'height', 'fx', 'fy', 'cx', 'cy']:
            if key in cam1:
                interp_cam[key] = cam1[key]

        return interp_cam

    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation for quaternions"""
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        # Compute dot product
        dot = np.dot(q1, q2)

        # If negative, negate one quaternion
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        # Compute angle
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)

        # Compute interpolation
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        return w1 * q1 + w2 * q2

    def _generate_with_inpainting(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """Generate pseudo-view using diffusion inpainting

        Args:
            img1: First image
            img2: Second image
            alpha: Interpolation factor

        Returns:
            Generated pseudo-view
        """
        # Simple blend as initial guess
        blended = ((1 - alpha) * img1 + alpha * img2).astype(np.uint8)

        # Create mask for uncertain regions (high gradient areas)
        mask = self._create_uncertainty_mask(img1, img2, alpha)

        # Convert to PIL
        blended_pil = Image.fromarray(blended)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

        # Resize to model input size (512x512 for SD 2.0)
        target_size = (512, 512)
        blended_resized = blended_pil.resize(target_size, Image.LANCZOS)
        mask_resized = mask_pil.resize(target_size, Image.NEAREST)

        # Generate with inpainting
        prompt = "high quality photograph, detailed, sharp"
        negative_prompt = "blurry, low quality, distorted"

        with torch.no_grad():
            result = self.inpainting_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=blended_resized,
                mask_image=mask_resized,
                num_inference_steps=20,  # Reduced for speed
                guidance_scale=7.5
            ).images[0]

        # Resize back to original size
        result_resized = result.resize(
            (img1.shape[1], img1.shape[0]),
            Image.LANCZOS
        )

        return np.array(result_resized)

    def _create_uncertainty_mask(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """Create mask for uncertain regions

        Args:
            img1: First image
            img2: Second image
            alpha: Interpolation factor

        Returns:
            Binary mask (1 = uncertain, 0 = certain)
        """
        # Compute difference
        diff = np.abs(img1.astype(float) - img2.astype(float)).mean(axis=2)

        # Normalize
        diff = diff / (diff.max() + 1e-8)

        # Threshold: high difference = uncertain
        mask = (diff > 0.3).astype(float)

        return mask

    def _simple_blend(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """Simple linear blending (fallback)"""
        return ((1 - alpha) * img1 + alpha * img2).astype(np.uint8)
