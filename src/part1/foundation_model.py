"""Foundation Model (DUSt3R/MASt3R) integration for pose estimation"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

# Add MASt3R and its dust3r submodule to Python path
MAST3R_PATH = Path(__file__).parent.parent.parent / "mast3r"
if MAST3R_PATH.exists():
    if str(MAST3R_PATH) not in sys.path:
        sys.path.insert(0, str(MAST3R_PATH))
    DUST3R_PATH = MAST3R_PATH / "dust3r"
    if DUST3R_PATH.exists() and str(DUST3R_PATH) not in sys.path:
        sys.path.insert(0, str(DUST3R_PATH))


class FoundationModelRunner:
    """Wrapper for Foundation Models (DUSt3R or MASt3R)"""

    def __init__(self, model_name: str = 'dust3r', device: str = 'cuda'):
        """Initialize Foundation Model

        Args:
            model_name: 'dust3r' or 'mast3r'
            device: 'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the foundation model"""
        try:
            if self.model_name == 'dust3r':
                from dust3r.inference import inference
                from dust3r.model import AsymmetricCroCo3DStereo
                from dust3r.utils.device import to_numpy

                self.model = AsymmetricCroCo3DStereo.from_pretrained(
                    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
                ).to(self.device)
                self.inference_fn = inference
                self.to_numpy = to_numpy

            elif self.model_name == 'mast3r':
                # MASt3R uses its own model class
                from mast3r.model import AsymmetricMASt3R
                from dust3r.inference import inference
                from dust3r.utils.device import to_numpy

                self.model = AsymmetricMASt3R.from_pretrained(
                    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                ).to(self.device)
                self.inference_fn = inference
                self.to_numpy = to_numpy
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

        except ImportError as e:
            print(f"Warning: Could not load {self.model_name}: {e}")
            print("Please install MASt3R from: https://github.com/naver/mast3r")
            self.model = None

    def estimate_poses(self, image_paths: List[str], output_path: str, batch_size: int = None) -> Dict:
        """Estimate camera poses from images

        Args:
            image_paths: List of image file paths
            output_path: Directory to save results
            batch_size: Process images in batches to save memory (None = process all at once)
        """
        if self.model is None:
            raise RuntimeError(f"{self.model_name} model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # If batch_size specified and we have many images, use batch processing
        if batch_size and len(image_paths) > batch_size:
            print(f"Processing {len(image_paths)} images in batches of {batch_size}")
            return self._estimate_poses_batched(image_paths, output_path, batch_size)

        if self.model_name == 'mast3r':
            return self._estimate_poses_mast3r(image_paths, output_path)
        else:
            return self._estimate_poses_dust3r(image_paths, output_path)

    def _estimate_poses_mast3r(self, image_paths: List[str], output_path: Path) -> Dict:
        """Estimate poses using MASt3R sparse global alignment"""
        from dust3r.utils.image import load_images
        from mast3r.image_pairs import make_pairs
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

        # Load images using dust3r loader (returns properly formatted dicts)
        imgs = load_images(list(image_paths), size=256, verbose=True)

        # Create pairs (sequential scene graph for large datasets)
        scene_graph = 'swin-5'  # sliding window with 5 neighbors
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

        cache_dir = str(output_path / 'cache')
        import os; os.makedirs(cache_dir, exist_ok=True)

        # Run sparse global alignment (MASt3R's pipeline)
        scene = sparse_global_alignment(
            list(image_paths), pairs, cache_dir,
            self.model, lr1=0.07, niter1=500, lr2=0.014, niter2=200,
            device=self.device, opt_depth=True, shared_intrinsics=False,
            matching_conf_thr=5.0
        )

        return self._extract_from_scene(scene, image_paths)

    def _estimate_poses_dust3r(self, image_paths: List[str], output_path: Path) -> Dict:
        """Estimate poses using DUSt3R global aligner"""
        from dust3r.utils.image import load_images
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        imgs = load_images(list(image_paths), size=256, verbose=True)

        # Create all pairs
        pairs = []
        for i in range(len(imgs)):
            for j in range(i + 1, min(i + 5, len(imgs))):
                pairs.append((imgs[i], imgs[j]))

        output = inference(pairs, self.model, self.device, batch_size=8)

        scene = global_aligner(output, device=self.device,
                               mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init='mst', niter=300)

        return self._extract_from_scene(scene, image_paths)

    def _estimate_poses_batched(self, image_paths: List[str], output_path: Path, batch_size: int) -> Dict:
        """Estimate poses in overlapping batches, then align and merge results."""
        from dust3r.utils.image import load_images
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        import torch

        overlap = 10  # overlapping images between consecutive batches
        step = batch_size - overlap
        all_cameras = {}
        all_images_dict = {}
        all_points = []

        starts = list(range(0, len(image_paths), step))
        num_batches = len(starts)
        print(f"Processing {len(image_paths)} images in {num_batches} batches "
              f"(batch_size={batch_size}, overlap={overlap})")

        # Reference poses from the first batch (global coordinate frame)
        ref_c2w = None  # shape (overlap, 4, 4) from previous batch's tail

        for b_idx, start_idx in enumerate(starts):
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            n = len(batch_paths)
            print(f"\nBatch {b_idx+1}/{num_batches}: images [{start_idx}..{end_idx-1}] ({n} images)")

            imgs = load_images(batch_paths, size=256, verbose=False)
            pairs = [(imgs[i], imgs[j])
                     for i in range(n) for j in range(i+1, min(i+5, n))]

            output = inference(pairs, self.model, self.device, batch_size=8, verbose=False)
            scene = global_aligner(output, device=self.device,
                                   mode=GlobalAlignerMode.PointCloudOptimizer)
            scene.compute_global_alignment(init='mst', niter=300)

            focals  = scene.get_focals().detach().cpu().numpy()
            c2w_arr = scene.get_im_poses().detach().cpu().numpy()   # (n, 4, 4) cam-to-world

            # Align this batch into the global frame using Procrustes on the overlap region
            if b_idx == 0:
                transform = np.eye(4)
            else:
                # curr overlap: first `overlap` poses of this batch
                # prev overlap: last `overlap` poses stored from previous batch
                curr_ov = c2w_arr[:overlap, :3, 3]       # (overlap, 3)
                prev_ov = ref_c2w[:overlap, :3, 3]       # (overlap, 3)
                transform = self._procrustes_transform(curr_ov, prev_ov)

            # Apply transformation to all poses in this batch
            aligned_poses = np.zeros((n, 4, 4))
            for i in range(n):
                aligned_poses[i] = transform @ c2w_arr[i]

            # Save the tail of this batch as reference overlap for next batch
            ref_c2w = aligned_poses[-overlap:] if n >= overlap else aligned_poses

            # Write results (skip leading overlap for b_idx>0 to avoid duplicates)
            write_start = overlap if b_idx > 0 else 0
            for i in range(write_start, n):
                global_idx = start_idx + i
                f = float(focals[i]) if focals[i].ndim == 0 else float(focals[i][0])
                all_cameras[global_idx] = {
                    'model_id': 1, 'width': 256, 'height': 256,
                    'params': [f, f, 128.0, 128.0]
                }
                w2c = np.linalg.inv(aligned_poses[i])
                all_images_dict[global_idx] = {
                    'qvec': self._rotation_to_quaternion(w2c[:3, :3]).tolist(),
                    'tvec': w2c[:3, 3].tolist(),
                    'camera_id': global_idx,
                    'name': Path(batch_paths[i]).name
                }

            # Collect point cloud (transformed)
            for pts in scene.get_pts3d():
                arr = pts.detach().cpu().numpy().reshape(-1, 3)
                ones = np.ones((len(arr), 1))
                arr_h = np.concatenate([arr, ones], axis=1)   # (N,4)
                arr_t = (transform @ arr_h.T).T[:, :3]
                all_points.append(arr_t)

            del scene, output, imgs, pairs
            torch.cuda.empty_cache()

        points3d = np.vstack(all_points) if all_points else np.zeros((0, 3))
        return {'cameras': all_cameras, 'images': all_images_dict, 'points3d': points3d}

    def _procrustes_transform(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Compute scale*R+t that maps src points -> dst points (Umeyama)."""
        assert src.shape == dst.shape and src.shape[1] == 3
        mu_s, mu_d = src.mean(0), dst.mean(0)
        src_c, dst_c = src - mu_s, dst - mu_d
        H = src_c.T @ dst_c
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        scale = np.sum(S) / (np.sum(src_c ** 2) + 1e-8)
        t = mu_d - scale * R @ mu_s
        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = t
        return T

    def _quaternion_to_rotation(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w,x,y,z) to rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])

    def _extract_from_scene(self, scene, image_paths: List[str]) -> Dict:
        """Extract cameras, poses, and points from an optimized scene"""
        cameras = {}
        images = {}

        focals = scene.get_focals().detach().cpu().numpy()
        im_poses = scene.get_im_poses().detach().cpu().numpy()  # (N, 4, 4) cam2world

        for i, img_path in enumerate(image_paths):
            f = float(focals[i]) if focals[i].ndim == 0 else float(focals[i][0])
            cameras[i] = {
                'model_id': 1,  # PINHOLE
                'width': 256,
                'height': 256,
                'params': [f, f, 128.0, 128.0]
            }

            # im_poses is cam2world; invert to world2cam for COLMAP convention
            c2w = im_poses[i]
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = self._rotation_to_quaternion(R)

            images[i] = {
                'qvec': qvec.tolist(),
                'tvec': t.tolist(),
                'camera_id': i,
                'name': Path(img_path).name
            }

        # Extract point cloud
        pts3d_fn = getattr(scene, 'get_sparse_pts3d', None) or getattr(scene, 'get_pts3d')
        pts3d = pts3d_fn()
        all_points = []
        for pts in pts3d:
            if pts is None:
                continue
            arr = pts.detach().cpu().numpy() if hasattr(pts, 'detach') else np.array(pts)
            arr = arr.reshape(-1, 3)
            if len(arr) > 0:
                all_points.append(arr)
        points3d = np.vstack(all_points) if all_points else np.zeros((0, 3))

        return {
            'cameras': cameras,
            'images': images,
            'points3d': points3d
        }

    def _rotation_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        return np.array([w, x, y, z])
