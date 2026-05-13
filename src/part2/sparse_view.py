"""Part 2: Sparse View Reconstruction Experiment

Pipeline:
1. Load dataset (DL3DV, RE10K, Waymo)
2. Sub-sample to sparse views (1/10 or 1/30 frames)
3. Run unposed reconstruction (DUSt3R)
4. Train 3DGS on reconstructed poses
5. Evaluate on held-out test views
6. Compute ATE for pose accuracy
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import time
from PIL import Image

from common.dataset import SceneDataset
from common.camera import Camera, CameraParameters
from common.metrics import compute_psnr, compute_ssim, compute_lpips
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.renderer import GaussianRenderer
from gaussian_splatting.trainer import GaussianTrainer
from part2.unposed_reconstruction import UnposedReconstructor


def run_sparse_view_experiment(
    data_path: str,
    output_path: str,
    sparsity: int = 10,
    iterations: int = 3000,
    device: str = 'cuda',
    temporal_range: str = 'full'  # 'full' or 'front_half'
) -> Dict:
    """Run complete sparse view reconstruction experiment.

    Args:
        data_path: Path to dataset (DL3DV/RE10K/Waymo)
        output_path: Output directory for results
        sparsity: Sub-sampling rate (1/N frames)
        iterations: 3DGS training iterations
        device: cuda or cpu
        temporal_range: 'full' for all frames, 'front_half' for first 50% only

    Returns:
        Dict with metrics: PSNR, SSIM, LPIPS, ATE
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Part 2: Sparse View Reconstruction")
    print(f"Dataset: {data_path}")
    print(f"Sparsity: 1/{sparsity} frames")
    print(f"Temporal range: {temporal_range}")
    print(f"{'='*60}\n")

    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    dataset = SceneDataset(data_path, dataset_type='auto')
    print(f"  Loaded {len(dataset)} frames")

    # Step 2: Sub-sample to sparse views
    print(f"\nStep 2: Sub-sampling to 1/{sparsity} frames (range: {temporal_range})...")
    sparse_indices, test_indices = _subsample_dataset(len(dataset), sparsity, temporal_range)
    print(f"  Sparse views: {len(sparse_indices)}")
    print(f"  Test views: {len(test_indices)}")

    sparse_images = [dataset.images[i] for i in sparse_indices]
    sparse_cameras_gt = [dataset.cameras[i] for i in sparse_indices]

    # Step 3: Unposed reconstruction
    print(f"\nStep 3: Running unposed reconstruction (DUSt3R)...")
    reconstructor = UnposedReconstructor(device=device)
    recon_result = reconstructor.reconstruct(
        sparse_images,
        output_path / 'reconstruction',
        niter=300,
        batch_size=8
    )

    # Save reconstruction
    _save_reconstruction(recon_result, output_path / 'reconstruction.json')
    print(f"  Reconstructed {len(recon_result['cameras'])} cameras")
    print(f"  Point cloud: {len(recon_result['points3d'])} points")

    # Compute ATE
    ate = reconstructor.compute_ate(recon_result, sparse_cameras_gt)
    print(f"  ATE RMSE: {ate:.4f}")

    # Step 4: Train 3DGS
    print(f"\nStep 4: Training 3DGS ({iterations} iterations)...")

    # Split sparse views: use 80% for training, 20% as hold-out test
    # Both sets are in MASt3R coordinate system (consistent coordinate frame)
    all_sparse_cameras = _convert_to_camera_list(recon_result)
    all_sparse_keys = sorted(recon_result['cameras'].keys())

    n_sparse = len(all_sparse_cameras)
    # Hold out every 5th sparse view as test (20%)
    test_sparse_idx = list(range(0, n_sparse, 5))
    train_sparse_idx = [i for i in range(n_sparse) if i not in test_sparse_idx]

    # Ensure at least 3 training views
    if len(train_sparse_idx) < 3:
        train_sparse_idx = list(range(n_sparse))
        test_sparse_idx = []

    train_cameras = [all_sparse_cameras[i] for i in train_sparse_idx]
    test_cameras_sparse = [all_sparse_cameras[i] for i in test_sparse_idx]
    print(f"  Training cameras: {len(train_cameras)}, Hold-out test cameras: {len(test_cameras_sparse)}")

    train_images_np = []
    for i in train_sparse_idx:
        k = all_sparse_keys[i]
        p = sparse_images[i]
        cam_w = recon_result['cameras'][k]['width']
        cam_h = recon_result['cameras'][k]['height']
        train_images_np.append(_load_image_np_sized(p, cam_w, cam_h))

    test_images_sparse_np = []
    for i in test_sparse_idx:
        k = all_sparse_keys[i]
        p = sparse_images[i]
        cam_w = recon_result['cameras'][k]['width']
        cam_h = recon_result['cameras'][k]['height']
        test_images_sparse_np.append(_load_image_np_sized(p, cam_w, cam_h))

    # Initialize 3DGS from point cloud
    model = GaussianModel()
    pts = recon_result['points3d']
    # Downsample point cloud to avoid OOM (max 50k points)
    MAX_PTS = 50000
    if len(pts) > MAX_PTS:
        idx = np.random.choice(len(pts), MAX_PTS, replace=False)
        pts = pts[idx]
        print(f"  Downsampled point cloud to {MAX_PTS} points")
    if len(pts) < 100:
        print(f"  Warning: Only {len(pts)} points, adding random points")
        pts = np.vstack([pts, np.random.randn(1000, 3) * 0.5])
    colors = np.ones((len(pts), 3)) * 0.5
    model.create_from_pcd(pts, colors)

    renderer = GaussianRenderer(device=device)
    trainer = GaussianTrainer(model, renderer, device=device)

    # Train
    train_result = trainer.train(
        train_cameras,
        train_images_np,
        iterations=iterations,
        test_cameras=test_cameras_sparse if test_cameras_sparse else None,
        test_images=test_images_sparse_np if test_images_sparse_np else None,
        save_path=output_path / 'model.npz',
        log_interval=100,
        eval_interval=500
    )

    # Step 5: Final evaluation on hold-out sparse views (same MASt3R coord system)
    print(f"\nStep 5: Final evaluation on {len(test_cameras_sparse)} hold-out sparse views...")
    if test_cameras_sparse:
        test_metrics = trainer.evaluate(test_cameras_sparse, test_images_sparse_np)
    else:
        # Fallback: evaluate on training views
        test_metrics = trainer.evaluate(train_cameras, train_images_np)

    # Also compute Sim3-aligned GT test metrics for reference
    print(f"\nComputing Sim3-aligned GT test metrics ({len(test_indices)} views)...")
    T_gt2mast3r = _compute_sim3_gt_to_recon(recon_result, sparse_cameras_gt)
    TARGET_W, TARGET_H = 512, 384
    gt_test_cameras = []
    gt_test_images = []
    for i in test_indices:
        gt_cam = dataset.cameras[i]
        img = _load_image_np_sized(dataset.images[i], TARGET_W, TARGET_H)
        gt_test_images.append(img)
        gt_test_cameras.append(_gt_cam_to_recon_space(gt_cam, T_gt2mast3r, TARGET_W, TARGET_H))
    gt_test_metrics = trainer.evaluate(gt_test_cameras, gt_test_images)
    print(f"  GT-aligned test: PSNR={gt_test_metrics['psnr']:.2f}, SSIM={gt_test_metrics['ssim']:.4f}")

    # Render some training views for visualization (same coordinate system as 3DGS)
    print(f"\nRendering sample views for visualization...")
    num_vis_samples = min(4, len(train_cameras))
    rendered_samples = []
    gt_samples = []
    for i in range(num_vis_samples):
        rendered = renderer.render(train_cameras[i], model)
        rendered_samples.append(rendered.detach().cpu().numpy())
        gt_samples.append(train_images_np[i])

    # Save visualization samples separately (not in JSON)
    vis_data = {
        'rendered_samples': rendered_samples,
        'gt_samples': gt_samples
    }
    np.savez(output_path / 'visualization_samples.npz', **vis_data)

    # Compile results
    results = {
        'dataset': str(data_path),
        'sparsity': sparsity,
        'num_sparse_views': len(sparse_indices),
        'num_train_cameras': len(train_cameras),
        'num_holdout_test': len(test_cameras_sparse),
        'num_gt_test_views': len(test_indices),
        'num_points': len(recon_result['points3d']),
        'ate_rmse': float(ate),
        # Primary metrics: hold-out sparse views in MASt3R coordinate system
        'psnr': test_metrics['psnr'],
        'ssim': test_metrics['ssim'],
        'lpips': test_metrics['lpips'],
        # Secondary metrics: Sim3-aligned GT test views
        'psnr_gt_aligned': gt_test_metrics['psnr'],
        'ssim_gt_aligned': gt_test_metrics['ssim'],
        'lpips_gt_aligned': gt_test_metrics['lpips'],
        'training_iterations': iterations,
        'best_train_psnr': train_result['best_psnr']
    }

    # Save results
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  ATE RMSE: {ate:.4f}")
    print(f"  PSNR (hold-out sparse): {test_metrics['psnr']:.2f} dB")
    print(f"  SSIM (hold-out sparse): {test_metrics['ssim']:.4f}")
    print(f"  LPIPS (hold-out sparse): {test_metrics['lpips']:.4f}")
    print(f"  PSNR (GT-aligned): {gt_test_metrics['psnr']:.2f} dB")
    print(f"{'='*60}\n")

    return results


def _subsample_dataset(n: int, sparsity: int, temporal_range: str = 'full') -> Tuple[List[int], List[int]]:
    """Sub-sample dataset into sparse training + test views.

    Args:
        n: Total number of frames
        sparsity: Sub-sampling rate (1/N)
        temporal_range: 'full' for all frames, 'front_half' for first 50% only

    Returns:
        (sparse_indices, test_indices)
    """
    # Determine frame range
    if temporal_range == 'front_half':
        max_frame = n // 2
        print(f"  Using front half only: frames 0-{max_frame} (out of {n} total)")
    else:
        max_frame = n

    all_indices = np.arange(max_frame)
    sparse_indices = all_indices[::sparsity].tolist()

    # Test set: frames not in sparse set (within the same temporal range)
    test_indices = [i for i in all_indices if i not in sparse_indices]

    # Limit test set size for efficiency
    if len(test_indices) > 50:
        test_indices = np.random.choice(test_indices, 50, replace=False).tolist()

    return sparse_indices, test_indices


def _save_reconstruction(recon: Dict, path: Path):
    """Save reconstruction result to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    save_dict = {
        'cameras': recon['cameras'],
        'images': recon['images'],
        'num_points': len(recon['points3d'])
    }
    with open(path, 'w') as f:
        json.dump(save_dict, f, indent=2)


def _convert_to_camera_list(recon: Dict) -> List[Camera]:
    """Convert reconstruction result to Camera list."""
    cameras = []
    for i in sorted(recon['cameras'].keys()):
        cam_data = recon['cameras'][i]
        img_data = recon['images'][i]

        # Extract intrinsics
        params = cam_data['params']
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Extract extrinsics (world2cam)
        qvec = np.array(img_data['qvec'])
        tvec = np.array(img_data['tvec'])
        R = _quat_to_rot(qvec)
        t = tvec.flatten()  # Ensure (3,) shape

        cameras.append(Camera(R, t, K, cam_data['width'], cam_data['height']))

    return cameras


def _camera_params_to_camera(cam: CameraParameters, target_w: int = 512, target_h: int = 384) -> Camera:
    """Convert CameraParameters to Camera, scaled to target render size."""
    # Scale intrinsics to target size
    sx = target_w / cam.width if cam.width > 0 else 1.0
    sy = target_h / cam.height if cam.height > 0 else 1.0
    K = np.array([
        [cam.fx * sx, 0, cam.cx * sx],
        [0, cam.fy * sy, cam.cy * sy],
        [0, 0, 1]
    ])
    if cam.rotation is not None and cam.translation is not None:
        c2w = np.eye(4)
        c2w[:3, :3] = cam.rotation
        c2w[:3, 3] = cam.translation.flatten()
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
    else:
        R = np.eye(3)
        t = np.zeros(3)

    return Camera(R, t, K, target_w, target_h)


def _load_image_np_sized(path: str, target_w: int, target_h: int) -> np.ndarray:
    """Load and resize image to exact (target_w, target_h)."""
    img = Image.open(path).convert('RGB').resize((target_w, target_h), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama Sim3 alignment: find T such that T @ src ≈ dst."""
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


def _compute_sim3_gt_to_recon(recon_result: Dict, sparse_cameras_gt) -> np.ndarray:
    """Compute Sim3 transform from GT coordinate space to MASt3R reconstruction space.

    Uses sparse training cameras as correspondences.
    Returns 4x4 Sim3 matrix that maps GT camera positions to MASt3R camera positions.
    """
    # Get MASt3R estimated camera positions (cam center in world = -R^T @ t)
    est_positions = []
    est_keys = sorted(recon_result['images'].keys())
    for k in est_keys:
        img_data = recon_result['images'][k]
        R_est = _quat_to_rot(np.array(img_data['qvec']))
        t_est = np.array(img_data['tvec'])
        # Camera center in world space
        cam_pos_est = -R_est.T @ t_est
        est_positions.append(cam_pos_est)

    # Get GT camera positions for the same sparse views
    # All datasets store w2c convention: cam_center = -R^T @ t
    n = min(len(est_positions), len(sparse_cameras_gt))
    gt_positions = []
    for cam in sparse_cameras_gt[:n]:
        if cam.rotation is not None and cam.translation is not None:
            R_gt = cam.rotation
            t_gt = cam.translation.flatten()
            # w2c convention: camera center in world = -R^T @ t
            gt_positions.append(-R_gt.T @ t_gt)
        else:
            gt_positions.append(np.zeros(3))

    est_arr = np.array(est_positions[:n])
    gt_arr  = np.array(gt_positions[:n])

    # T maps GT positions -> MASt3R positions
    if n < 3:
        return np.eye(4)
    return _umeyama(gt_arr, est_arr)


def _gt_cam_to_recon_space(gt_cam, T_gt2mast3r: np.ndarray,
                            target_w: int, target_h: int) -> Camera:
    """Transform a GT camera into MASt3R reconstruction space.

    All datasets store w2c convention: rotation=R_w2c, translation=t_w2c.
    Camera center in world = -R^T @ t.
    T_gt2mast3r maps GT world positions -> MASt3R world positions.
    """
    # Build GT w2c matrix
    w2c_gt = np.eye(4)
    w2c_gt[:3, :3] = gt_cam.rotation
    w2c_gt[:3, 3]  = gt_cam.translation.flatten()

    # Build GT c2w
    c2w_gt = np.linalg.inv(w2c_gt)

    # Apply Sim3: new_c2w = T_gt2mast3r @ old_c2w
    c2w_recon = T_gt2mast3r @ c2w_gt

    # Convert to w2c for Camera
    w2c_recon = np.linalg.inv(c2w_recon)
    R = w2c_recon[:3, :3]
    t = w2c_recon[:3, 3]

    # Scale intrinsics to target resolution
    sx = target_w / gt_cam.width  if gt_cam.width  > 0 else 1.0
    sy = target_h / gt_cam.height if gt_cam.height > 0 else 1.0
    K = np.array([
        [gt_cam.fx * sx, 0, gt_cam.cx * sx],
        [0, gt_cam.fy * sy, gt_cam.cy * sy],
        [0, 0, 1]
    ])
    return Camera(R, t, K, target_w, target_h)
