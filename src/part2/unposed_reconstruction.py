"""Unposed Sparse Reconstruction using MASt3R (Option A - RegGS style)

Pipeline:
1. Sub-sample input frames to create sparse views (1/10 or 1/30)
2. Run MASt3R pairwise inference (no poses needed)
3. Global alignment via PointCloudOptimizer
4. Extract camera poses + point cloud
5. Train 3DGS on the result
6. Evaluate PSNR/SSIM/LPIPS on held-out test views
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import time

# Add DUSt3R/MASt3R to path
MAST3R_PATH = Path(__file__).parent.parent.parent / 'mast3r'
if MAST3R_PATH.exists():
    if str(MAST3R_PATH) not in sys.path:
        sys.path.insert(0, str(MAST3R_PATH))
    if str(MAST3R_PATH / 'dust3r') not in sys.path:
        sys.path.insert(0, str(MAST3R_PATH / 'dust3r'))

from common.camera import Camera, CameraParameters


class UnposedReconstructor:
    """Unposed sparse 3D reconstruction using MASt3R.

    This implements Option A from the project PDF:
    - No camera poses required as input
    - Uses MASt3R for geometry prediction & registration (upgraded from DUSt3R)
    - Fuses aligned point clouds into a global scene
    - Trains 3DGS for rendering quality evaluation
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load MASt3R model (upgraded from DUSt3R for better reconstruction quality)"""
        try:
            from mast3r.model import AsymmetricMASt3R
            self.model = AsymmetricMASt3R.from_pretrained(
                "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            ).to(self.device).eval()
            print("MASt3R model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load MASt3R, falling back to DUSt3R: {e}")
            try:
                from dust3r.model import AsymmetricCroCo3DStereo
                self.model = AsymmetricCroCo3DStereo.from_pretrained(
                    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
                ).to(self.device).eval()
                print("DUSt3R model loaded as fallback")
            except Exception as e2:
                print(f"Warning: Could not load DUSt3R either: {e2}")
                self.model = None

    def reconstruct(self, image_paths: List[str], output_path: Path,
                    niter: int = 500, batch_size: int = 8) -> Dict:
        """Run full unposed reconstruction pipeline.

        Args:
            image_paths: Sparse input frames (already sub-sampled)
            output_path: Directory to save cache and results
            niter: Global alignment iterations
            batch_size: DUSt3R inference batch size

        Returns:
            Dict with 'cameras', 'images', 'points3d' in COLMAP-like format
        """
        if self.model is None:
            raise RuntimeError("DUSt3R model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        n = len(image_paths)
        print(f"Reconstructing {n} sparse frames (no poses provided)...")
        t0 = time.time()

        # If too many images for single pass, use batched processing
        if n > 60:
            result = self._reconstruct_batched(image_paths, output_path, niter, batch_size)
        else:
            result = self._reconstruct_single(image_paths, output_path, niter, batch_size)

        elapsed = time.time() - t0
        print(f"Reconstruction done in {elapsed:.1f}s")
        return result

    def _reconstruct_single(self, image_paths: List[str], output_path: Path,
                             niter: int, batch_size: int) -> Dict:
        """Single-pass reconstruction for small image sets."""
        from dust3r.utils.image import load_images as dust3r_load_images
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        imgs = dust3r_load_images(list(image_paths), size=512, verbose=True)
        n = len(imgs)

        # Create sliding-window pairs (window=5)
        pairs = []
        for i in range(n):
            for j in range(i + 1, min(i + 6, n)):
                pairs.append((imgs[i], imgs[j]))
        print(f"  Created {len(pairs)} image pairs")

        output = inference(pairs, self.model, self.device,
                           batch_size=batch_size, verbose=False)

        scene = global_aligner(output, device=self.device,
                                mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule='cosine', lr=0.01)
        print(f"  Global alignment loss: {float(loss):.4f}")

        return self._extract_result(scene, image_paths, imgs)

    def _reconstruct_batched(self, image_paths: List[str], output_path: Path,
                              niter: int, batch_size: int) -> Dict:
        """Batched reconstruction for large image sets with overlap + Procrustes merge."""
        from dust3r.utils.image import load_images as dust3r_load_images
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        BATCH = 50
        OVERLAP = 10
        step = BATCH - OVERLAP

        all_cameras = {}
        all_images_dict = {}
        all_points = []
        ref_c2w = None

        starts = list(range(0, len(image_paths), step))
        print(f"Batched: {len(image_paths)} images, {len(starts)} batches (size={BATCH}, overlap={OVERLAP})")

        for b_idx, start in enumerate(starts):
            end = min(start + BATCH, len(image_paths))
            bpaths = image_paths[start:end]
            n = len(bpaths)
            print(f"\nBatch {b_idx+1}/{len(starts)}: [{start}..{end-1}] ({n} images)")

            imgs = dust3r_load_images(bpaths, size=512, verbose=False)
            pairs = [(imgs[i], imgs[j])
                     for i in range(n) for j in range(i+1, min(i+6, n))]

            out = inference(pairs, self.model, self.device, batch_size=batch_size, verbose=False)
            scene = global_aligner(out, device=self.device,
                                   mode=GlobalAlignerMode.PointCloudOptimizer)
            scene.compute_global_alignment(init='mst', niter=niter, schedule='cosine', lr=0.01)

            focals  = scene.get_focals().detach().cpu().numpy()
            c2w_arr = scene.get_im_poses().detach().cpu().numpy()

            # Procrustes alignment to global frame
            if b_idx == 0:
                T = np.eye(4)
            else:
                curr_ov = c2w_arr[:OVERLAP, :3, 3]
                prev_ov = ref_c2w[:OVERLAP, :3, 3]
                T = self._umeyama(curr_ov, prev_ov)

            aligned = np.stack([T @ c2w_arr[i] for i in range(n)])
            ref_c2w = aligned[-OVERLAP:] if n >= OVERLAP else aligned

            write_start = OVERLAP if b_idx > 0 else 0
            for i in range(write_start, n):
                gidx = start + i
                f = float(focals[i].flat[0])
                all_cameras[gidx] = {
                    'model_id': 1, 'width': 512, 'height': 512,
                    'params': [f, f, 256.0, 256.0]
                }
                w2c = np.linalg.inv(aligned[i])
                all_images_dict[gidx] = {
                    'qvec': self._rot_to_quat(w2c[:3, :3]).tolist(),
                    'tvec': w2c[:3, 3].tolist(),
                    'camera_id': gidx,
                    'name': Path(bpaths[i]).name
                }

            # Accumulate point cloud - limit per frame
            MAX_PER_FRAME = 2000
            for pts in scene.get_pts3d():
                arr = pts.detach().cpu().numpy().reshape(-1, 3)
                if len(arr) > MAX_PER_FRAME:
                    idx = np.random.choice(len(arr), MAX_PER_FRAME, replace=False)
                    arr = arr[idx]
                ones = np.ones((len(arr), 1))
                arr_g = (T @ np.concatenate([arr, ones], 1).T).T[:, :3]
                all_points.append(arr_g)

            del scene, out, imgs, pairs
            torch.cuda.empty_cache()

        pts3d = np.vstack(all_points) if all_points else np.zeros((0, 3))
        return {'cameras': all_cameras, 'images': all_images_dict, 'points3d': pts3d}

    def _extract_result(self, scene, image_paths: List[str], imgs=None) -> Dict:
        """Extract cameras and points from aligned scene."""
        cameras = {}
        images = {}

        focals  = scene.get_focals().detach().cpu().numpy()
        c2w_arr = scene.get_im_poses().detach().cpu().numpy()

        for i, img_path in enumerate(image_paths):
            f = float(focals[i].flat[0])
            # Get actual image size from loaded imgs (DUSt3R may resize)
            if imgs is not None and i < len(imgs):
                h, w = imgs[i]['img'].shape[-2], imgs[i]['img'].shape[-1]
            else:
                w, h = 512, 512
            cx, cy = w / 2.0, h / 2.0
            cameras[i] = {
                'model_id': 1, 'width': w, 'height': h,
                'params': [f, f, cx, cy]
            }
            w2c = np.linalg.inv(c2w_arr[i])
            images[i] = {
                'qvec': self._rot_to_quat(w2c[:3, :3]).tolist(),
                'tvec': w2c[:3, 3].tolist(),
                'camera_id': i,
                'name': Path(img_path).name
            }

        # Extract point cloud - limit per-frame points to avoid OOM
        pts_fn = getattr(scene, 'get_sparse_pts3d', None) or scene.get_pts3d
        all_pts = []
        MAX_PER_FRAME = 2000  # limit per frame to keep total manageable
        for pts in pts_fn():
            if pts is None:
                continue
            arr = pts.detach().cpu().numpy() if hasattr(pts, 'detach') else np.array(pts)
            arr = arr.reshape(-1, 3)
            if len(arr) > MAX_PER_FRAME:
                idx = np.random.choice(len(arr), MAX_PER_FRAME, replace=False)
                arr = arr[idx]
            all_pts.append(arr)
        points3d = np.vstack(all_pts) if all_pts else np.zeros((0, 3))

        return {'cameras': cameras, 'images': images, 'points3d': points3d}

    # ---- Helpers ----

    def _umeyama(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Umeyama (scale+rotation+translation) alignment: src -> dst."""
        assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3
        mu_s, mu_d = src.mean(0), dst.mean(0)
        sc, dc = src - mu_s, dst - mu_d
        H = sc.T @ dc
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        scale = S.sum() / (np.sum(sc**2) + 1e-8)
        t = mu_d - scale * R @ mu_s
        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = t
        return T

    def _rot_to_quat(self, R: np.ndarray) -> np.ndarray:
        """Rotation matrix -> quaternion (w, x, y, z)."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            return np.array([0.25/s, (R[2,1]-R[1,2])*s,
                             (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                             (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                             0.25*s, (R[1,2]+R[2,1])/s])
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                             (R[1,2]+R[2,1])/s, 0.25*s])

    def compute_ate(self, estimated_poses: Dict, gt_cameras: List) -> float:
        """Compute ATE (Absolute Trajectory Error) RMSE.

        Aligns estimated trajectory to ground truth via Umeyama, then
        computes position RMSE.

        Args:
            estimated_poses: Dict with 'images' key (from reconstruct())
            gt_cameras: List of CameraParameters with ground truth poses
        Returns:
            ATE RMSE in scene units
        """
        est_keys = sorted(estimated_poses['images'].keys())
        n = min(len(est_keys), len(gt_cameras))
        if n < 2:
            return float('nan')

        est_t = []
        gt_t  = []
        for k in est_keys[:n]:
            img_data = estimated_poses['images'][k]
            qvec = np.array(img_data['qvec'])
            tvec = np.array(img_data['tvec'])
            # world2cam -> cam position in world = -R^T @ t
            R = self._quat_to_rot(qvec)
            cam_pos = -R.T @ tvec
            est_t.append(cam_pos)

        for cam in gt_cameras[:n]:
            if hasattr(cam, 'translation') and cam.translation is not None:
                R_gt = cam.rotation if hasattr(cam, 'rotation') else np.eye(3)
                gt_t.append(-R_gt.T @ cam.translation)
            elif hasattr(cam, 't'):
                gt_t.append(-cam.R.T @ cam.t)
            else:
                gt_t.append(np.zeros(3))

        est_t = np.array(est_t)
        gt_t  = np.array(gt_t)

        # Align estimated -> GT
        T_align = self._umeyama(est_t, gt_t)
        ones = np.ones((n, 1))
        est_aligned = (T_align @ np.concatenate([est_t, ones], 1).T).T[:, :3]

        ate = float(np.sqrt(np.mean(np.sum((est_aligned - gt_t)**2, axis=1))))
        return ate

    def _quat_to_rot(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
