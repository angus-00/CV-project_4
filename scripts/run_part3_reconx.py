"""Part 3 ReconX Route: DynamiCrafter Pseudo-Views + Hybrid 3DGS Optimization

Uses pre-generated DynamiCrafter video interpolation frames as pseudo-views
for 3DGS fine-tuning. Confidence is endpoint-weighted (frames near real sparse
views are more reliable than midpoint free-generation).

Key differences from Difix3D v4:
- No rendering + inpainting step (pseudo-views come pre-generated from DynamiCrafter)
- Camera intrinsics from dc_output (512x320), real views keep native (512x384)
- Confidence: render-agreement * endpoint-bias (NOT midpoint-bias)
- Simpler pipeline: load dc_output -> confidence -> train
"""

import sys
import os
import json
import time
import torch
import cv2
import numpy as np
from pathlib import Path
import torch.nn.functional as F_depth
from typing import Dict, List, Tuple, Optional
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from common.memory_monitor import start_memory_monitor, stop_memory_monitor
from common.dataset import SceneDataset
from common.camera import Camera
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.renderer import GaussianRenderer
from gaussian_splatting.trainer import GaussianTrainer
from common.metrics import compute_psnr, compute_ssim, compute_lpips


MEMORY_LIMIT_GB = 15.7
MAX_RENDER_DIM = 512

# Dataset mapping: dc_output dataset name -> part2 path
DATASET_MAP = {
    'dl3dv': 'outputs/part2/dl3dv_s10_front_half',
    're10k': 'outputs/part2/re10k',
    'waymo': 'outputs/part2/waymo',
}


def check_mem(label: str = ""):
    if not torch.cuda.is_available():
        return
    used = torch.cuda.memory_reserved() / 1e9
    if used > MEMORY_LIMIT_GB:
        print(f"\n{'!'*70}")
        print(f"  MEMORY LIMIT EXCEEDED  ({used:.2f} GB > {MEMORY_LIMIT_GB} GB)")
        if label:
            print(f"  Location: {label}")
        print(f"{'!'*70}")
        sys.exit(1)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),      1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y),      2*(y*z+w*x),    1-2*(x**2+y**2)]
    ])


def build_camera_from_dc(cam_data: Dict, frame: Dict) -> Camera:
    """Build Camera from dc_output camera dict + frame pose."""
    params = cam_data['params']
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    orig_w, orig_h = cam_data['width'], cam_data['height']
    scale = min(1.0, MAX_RENDER_DIM / max(orig_w, orig_h))
    w = max(1, int(orig_w * scale))
    h = max(1, int(orig_h * scale))
    K = np.array([[fx*scale, 0, cx*scale], [0, fy*scale, cy*scale], [0, 0, 1]])
    R = quat_to_rot(np.array(frame['qvec']))
    t = np.array(frame['tvec'])
    return Camera(R, t, K, w, h)


def build_camera_from_recon(cam_data: Dict, img_data: Dict) -> Camera:
    """Build Camera from reconstruction camera dict + image pose."""
    params = cam_data['params']
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    orig_w, orig_h = cam_data['width'], cam_data['height']
    scale = min(1.0, MAX_RENDER_DIM / max(orig_w, orig_h))
    w = max(1, int(orig_w * scale))
    h = max(1, int(orig_h * scale))
    K = np.array([[fx*scale, 0, cx*scale], [0, fy*scale, cy*scale], [0, 0, 1]])
    R = quat_to_rot(np.array(img_data['qvec']))
    t = np.array(img_data['tvec'])
    return Camera(R, t, K, w, h)


# ─────────────────────────────────────────────────────────
#  Step 0: Pre-flight sanity check
# ─────────────────────────────────────────────────────────

def preflight_check(
    dc_data: Dict,
    pseudo_images_dir: Path,
    reconstruction: Dict,
    sparse_image_paths: List[str],
    train_local_idx: List[int],
    model: GaussianModel,
    renderer: GaussianRenderer,
    output_path: Path,
) -> Dict:
    """Render Part2 model from a near-endpoint pseudo-view and check PSNR vs dc image.

    Endpoint frames (small or large t_norm) should be close to a real sparse view,
    so PSNR > 14 dB indicates pose/intrinsic mapping is consistent.
    """
    frames = dc_data['frames']
    cameras = dc_data['cameras']

    # Pick an endpoint-ish frame (smallest t_norm)
    sorted_frames = sorted(frames, key=lambda f: f['t_norm'])
    test_frame = sorted_frames[0]
    cam_id = str(test_frame['camera_id'])
    cam_data = cameras[cam_id]
    derived = cam_data.get('derived_from_camera_id', -1)

    pseudo_cam = build_camera_from_dc(cam_data, test_frame)
    img_path = pseudo_images_dir / test_frame['name']
    pseudo_img = np.array(Image.open(img_path).convert('RGB'))
    pseudo_img_rs = cv2.resize(pseudo_img, (pseudo_cam.width, pseudo_cam.height))

    with torch.no_grad():
        rendered_t = renderer.render(pseudo_cam, model)
    rendered_np = (np.clip(rendered_t.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)

    # PSNR vs dc image
    rend_t = torch.from_numpy(rendered_np.astype(np.float32) / 255.0)
    pseu_t = torch.from_numpy(pseudo_img_rs.astype(np.float32) / 255.0)
    psnr_vs_dc = float(compute_psnr(rend_t, pseu_t))

    # PSNR vs nearest real sparse view (camera derived_from)
    psnr_vs_real = -1.0
    if 0 <= derived < len(train_local_idx):
        # Find which local index matches derived
        try:
            local_pos = train_local_idx.index(derived)
            real_img_path = sparse_image_paths[derived]
            real_img = np.array(Image.open(real_img_path).convert('RGB'))
            real_img_rs = cv2.resize(real_img, (pseudo_cam.width, pseudo_cam.height))
            real_t = torch.from_numpy(real_img_rs.astype(np.float32) / 255.0)
            psnr_vs_real = float(compute_psnr(pseu_t, real_t))
        except ValueError:
            pass

    report = {
        'test_frame': test_frame['name'],
        't_norm': test_frame['t_norm'],
        'camera_id': cam_id,
        'derived_from_camera_id': derived,
        'pseudo_cam_size': [pseudo_cam.width, pseudo_cam.height],
        'pseudo_cam_fx': float(cam_data['params'][0]),
        'psnr_render_vs_dc': psnr_vs_dc,
        'psnr_dc_vs_real_endpoint': psnr_vs_real,
        'recon_cam_sample': {
            'width':  reconstruction['cameras'][list(reconstruction['cameras'].keys())[0]]['width'],
            'height': reconstruction['cameras'][list(reconstruction['cameras'].keys())[0]]['height'],
            'fx':     reconstruction['cameras'][list(reconstruction['cameras'].keys())[0]]['params'][0],
        },
    }

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / '_preflight.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  [preflight] frame={test_frame['name']} t_norm={test_frame['t_norm']:.3f}")
    print(f"  [preflight] PSNR(part2_render vs dc_image) = {psnr_vs_dc:.2f} dB (expect > 12 dB)")
    if psnr_vs_real > 0:
        print(f"  [preflight] PSNR(dc_image vs real_endpoint) = {psnr_vs_real:.2f} dB")
    if psnr_vs_dc < 8.0:
        print(f"  [preflight] WARNING: PSNR very low - possible pose/intrinsic mismatch")
    return report


# ─────────────────────────────────────────────────────────
#  Confidence estimation
# ─────────────────────────────────────────────────────────

def compute_confidence(
    pseudo_img: np.ndarray,
    rendered_img: np.ndarray,
    t_norm: float,
) -> Tuple[np.ndarray, float]:
    """Per-pixel confidence map + scalar weight.

    Components:
      c_render: per-pixel (1 - normalized render error)
      c_endpoint: scalar = |2*t_norm - 1|  (high near endpoints, low at midpoint)
    """
    err = np.abs(pseudo_img.astype(float) - rendered_img.astype(float)).mean(axis=2)
    err_n = err / (err.max() + 1e-8)
    c_render = 1.0 - err_n  # (H, W) in [0, 1]

    c_endpoint = float(abs(2.0 * t_norm - 1.0))  # in [0, 1]

    # combined per-pixel confidence (kept high; the scalar weight controls blast radius)
    combined = np.clip(c_render, 0.0, 1.0).astype(np.float32)
    return combined, c_endpoint


# ─────────────────────────────────────────────────────────
#  Pseudo-view generation from DynamiCrafter outputs
# ─────────────────────────────────────────────────────────

def generate_pseudo_views_reconx(
    model: GaussianModel,
    renderer: GaussianRenderer,
    dc_data: Dict,
    pseudo_images_dir: Path,
) -> Tuple[List[Camera], List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    """Build pseudo-views from pre-generated DynamiCrafter frames.

    Returns: cameras, images (np), confidence_maps (np), t_norms, c_endpoints
    """
    frames = dc_data['frames']
    cameras_dict = dc_data['cameras']

    out_cams, out_imgs, out_confs, out_tnorms, out_endpoints = [], [], [], [], []

    print(f"  Loading {len(frames)} DynamiCrafter pseudo-views...")
    for i, frame in enumerate(frames):
        cam_id = str(frame['camera_id'])
        cam_data = cameras_dict[cam_id]
        pseudo_cam = build_camera_from_dc(cam_data, frame)

        img_path = pseudo_images_dir / frame['name']
        pseudo_img = np.array(Image.open(img_path).convert('RGB'))
        if pseudo_img.shape[1] != pseudo_cam.width or pseudo_img.shape[0] != pseudo_cam.height:
            pseudo_img = cv2.resize(pseudo_img, (pseudo_cam.width, pseudo_cam.height))

        with torch.no_grad():
            rendered_t = renderer.render(pseudo_cam, model)
        rendered_np = (np.clip(rendered_t.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)

        conf_map, c_endpoint = compute_confidence(pseudo_img, rendered_np, frame['t_norm'])

        out_cams.append(pseudo_cam)
        out_imgs.append(pseudo_img)
        out_confs.append(conf_map)
        out_tnorms.append(float(frame['t_norm']))
        out_endpoints.append(c_endpoint)

        if i % 10 == 0:
            print(f"    [{i:03d}] {frame['name']} t_norm={frame['t_norm']:.3f} "
                  f"c_endpoint={c_endpoint:.3f} render_conf_mean={conf_map.mean():.3f}")

    return out_cams, out_imgs, out_confs, out_tnorms, out_endpoints


# ─────────────────────────────────────────────────────────
#  Main experiment
# ─────────────────────────────────────────────────────────

def run_reconx_experiment(
    dataset_name: str,
    part2_output_path: str,
    dc_output_path: str,
    output_path: str,
    iterations: int = 15000,
    device: str = 'cuda',
    num_rounds: int = 3,
    max_gaussians: int = 40000,
    no_densify: bool = True,
    weight_cap: float = 0.45,
) -> Dict:
    t_start = time.time()
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    part2_path = Path(part2_output_path)
    dc_path = Path(dc_output_path)
    pseudo_images_dir = dc_path / 'pseudo_images'

    print(f"\n{'='*70}")
    print(f"Part 3 ReconX: DynamiCrafter Pseudo-View Training")
    print(f"Dataset:         {dataset_name}")
    print(f"Part 2 path:     {part2_path}")
    print(f"DC output path:  {dc_path}")
    print(f"Iterations:      {iterations}  (rounds={num_rounds})")
    print(f"Max Gaussians:   {max_gaussians}")
    print(f"Weight cap:      {weight_cap}")
    print(f"Memory limit:    {MEMORY_LIMIT_GB} GB")
    print(f"{'='*70}\n")

    check_mem("start")

    # Load dc_output metadata (try aligned version first)
    print("Loading DynamiCrafter metadata...")
    aligned_file = dc_path / 'pseudo_poses_aligned.json'
    poses_file = aligned_file if aligned_file.exists() else (dc_path / 'pseudo_poses.json')
    print(f"  Using poses: {poses_file.name}")
    with open(poses_file) as f:
        dc_data = json.load(f)
    n_pseudo_total = len(dc_data['frames'])
    print(f"  {n_pseudo_total} pseudo-views available")

    # Load Part 2 results
    print("Loading Part 2 results...")
    with open(part2_path / 'reconstruction.json') as f:
        reconstruction = json.load(f)
    with open(part2_path / 'results.json') as f:
        part2_results = json.load(f)

    baseline_psnr = part2_results['psnr']
    baseline_ssim = part2_results['ssim']
    baseline_lpips = part2_results['lpips']
    print(f"  Baseline (Part 2): PSNR={baseline_psnr:.2f}, SSIM={baseline_ssim:.4f}, LPIPS={baseline_lpips:.4f}")

    # Load dataset
    print("Loading dataset...")
    dataset = SceneDataset(str(part2_results['dataset']))
    n_total = len(dataset)
    n_recon = len(reconstruction['images'])
    sparsity = part2_results['sparsity']

    all_indices = np.arange(n_total)
    sparse_indices = all_indices[::sparsity].tolist()
    sparse_image_paths = [dataset.images[i] for i in sparse_indices[:n_recon]]

    # Train/test split (same as v4: every 5th sparse view is holdout)
    test_sparse_local_idx = list(range(0, n_recon, 5))
    train_sparse_local_idx = [i for i in range(n_recon) if i not in test_sparse_local_idx]
    if len(train_sparse_local_idx) < 3:
        train_sparse_local_idx = list(range(n_recon))
        test_sparse_local_idx = []

    n_train_sparse = len(train_sparse_local_idx)
    print(f"  Sparse views: {n_recon} total, {n_train_sparse} train, "
          f"{len(test_sparse_local_idx)} test")

    # Load Part 2 3DGS model
    print("Loading Part 2 3DGS model...")
    model_path = part2_path / 'model.npz'
    if model_path.is_dir():
        model_path = model_path / 'model.npz'
    pts_data = np.load(str(model_path))

    renderer = GaussianRenderer(device=device)

    MAX_G = 150000
    xyz = pts_data['xyz']
    opacity_np = 1.0 / (1.0 + np.exp(-pts_data['opacity'].squeeze()))
    if len(xyz) > MAX_G:
        top_idx = np.argsort(opacity_np)[::-1][:MAX_G]
    else:
        top_idx = np.arange(len(xyz))

    def _load_model_from_pts(idx_arr):
        m = GaussianModel()
        m._xyz           = torch.nn.Parameter(torch.tensor(pts_data['xyz'][idx_arr], dtype=torch.float32).cuda())
        m._features_dc   = torch.nn.Parameter(torch.tensor(pts_data['features_dc'][idx_arr], dtype=torch.float32).cuda())
        m._features_rest = torch.nn.Parameter(torch.tensor(pts_data['features_rest'][idx_arr], dtype=torch.float32).cuda())
        m._scaling       = torch.nn.Parameter(torch.tensor(pts_data['scaling'][idx_arr], dtype=torch.float32).cuda())
        m._rotation      = torch.nn.Parameter(torch.tensor(pts_data['rotation'][idx_arr], dtype=torch.float32).cuda())
        m._opacity       = torch.nn.Parameter(torch.tensor(pts_data['opacity'][idx_arr], dtype=torch.float32).cuda())
        m.sh_degree        = int(pts_data['sh_degree'])
        m.active_sh_degree = int(pts_data['active_sh_degree'])
        return m

    current_model = _load_model_from_pts(top_idx)
    print(f"  Loaded {len(top_idx)} Gaussians from Part 2 model")
    check_mem("after model load")

    # Step 0: Preflight
    print("\n--- Step 0: Preflight sanity check ---")
    preflight = preflight_check(
        dc_data, pseudo_images_dir, reconstruction,
        sparse_image_paths, train_sparse_local_idx,
        current_model, renderer, output_path,
    )

    # Build test cameras (native 512x384)
    img_keys = sorted(reconstruction['images'].keys(), key=lambda x: int(x))
    test_cameras = []
    test_images_np = []
    for local_i in test_sparse_local_idx:
        k = img_keys[local_i]
        img_data = reconstruction['images'][k]
        cam_id = str(img_data.get('camera_id', k))
        cam_data = reconstruction['cameras'][cam_id]
        test_cam = build_camera_from_recon(cam_data, img_data)
        test_cameras.append(test_cam)
        img = Image.open(sparse_image_paths[local_i]).convert('RGB')
        img = img.resize((test_cam.width, test_cam.height), Image.LANCZOS)
        test_images_np.append(np.array(img).astype(np.float32) / 255.0)

    # Build real training cameras (native 512x384)
    train_cameras_real = []
    train_images_real = []
    for local_i in train_sparse_local_idx:
        k = img_keys[local_i]
        img_data = reconstruction['images'][k]
        cam_id = str(img_data.get('camera_id', k))
        cam_data = reconstruction['cameras'][cam_id]
        cam = build_camera_from_recon(cam_data, img_data)
        train_cameras_real.append(cam)
        img = Image.open(sparse_image_paths[local_i]).convert('RGB')
        img = img.resize((cam.width, cam.height), Image.LANCZOS)
        train_images_real.append(np.array(img))

    # Depth supervision (same as v4)
    print("Pre-computing Depth Anything V2 depth maps...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    depth_model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_processor = AutoImageProcessor.from_pretrained(depth_model_name)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).to(device)
    depth_model.eval()

    depth_maps_gt = {}
    for i, cam in enumerate(train_cameras_real):
        img_pil = Image.fromarray(train_images_real[i])
        inputs = depth_processor(images=img_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        depth_resized = F_depth.interpolate(
            predicted_depth.unsqueeze(0), size=(cam.height, cam.width),
            mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        depth_maps_gt[i] = 1.0 / (depth_resized + 1e-6)

    del depth_model, depth_processor
    torch.cuda.empty_cache()
    print(f"  Computed {len(depth_maps_gt)} depth maps, freed depth model VRAM")
    check_mem("after depth pre-computation")

    DEPTH_LAMBDA = 0.01
    DEPTH_START_ITER = 500

    def depth_loss_fn(camera, model_obj, renderer_obj, iteration, view_idx):
        if iteration < DEPTH_START_ITER:
            return None
        if view_idx >= n_train_sparse:
            return None
        _, invdepth = renderer_obj.render(camera, model_obj, return_depth=True)
        pred_invdepth = invdepth[:, :, 0]
        gt_invdepth_np = depth_maps_gt[view_idx]
        gt_invdepth = torch.tensor(gt_invdepth_np, dtype=torch.float32, device=pred_invdepth.device)
        valid = (pred_invdepth > 0.01) & (gt_invdepth > 0.01) & (gt_invdepth < 1e4)
        if valid.sum() < 100:
            return None
        log_pred = torch.log(pred_invdepth[valid])
        log_gt = torch.log(gt_invdepth[valid])
        diff = log_pred - log_gt
        si_loss = (diff ** 2).mean() - 0.5 * (diff.mean() ** 2)
        return DEPTH_LAMBDA * si_loss

    # Iterative refinement loop
    iters_per_round = max(1000, iterations // num_rounds)

    pseudo_weights_final = []
    pseudo_tnorms_final = []
    pseudo_endpoints_final = []

    for round_idx in range(num_rounds):
        print(f"\n{'-'*60}")
        print(f"  Round {round_idx+1}/{num_rounds}  ({iters_per_round} iterations)")
        print(f"{'-'*60}")

        pseudo_cams, pseudo_imgs, pseudo_confs, pseudo_tnorms, pseudo_endpoints = \
            generate_pseudo_views_reconx(current_model, renderer, dc_data, pseudo_images_dir)

        # Weight formula v2: additive blend of endpoint-bias + render-agreement,
        # higher floor so middle-section frames still contribute non-trivially.
        # w = clip(0.4 * c_endpoint + 0.4 * c_render + 0.1, floor, cap)
        WEIGHT_FLOOR = 0.25
        pseudo_weights = []
        for i in range(len(pseudo_cams)):
            c_render_mean = float(pseudo_confs[i].mean())
            c_endpoint = pseudo_endpoints[i]
            w = float(np.clip(0.4 * c_endpoint + 0.4 * c_render_mean + 0.1,
                              WEIGHT_FLOOR, weight_cap))
            pseudo_weights.append(w)

        pw_arr = np.array(pseudo_weights)
        print(f"  Pseudo weights: mean={pw_arr.mean():.3f} min={pw_arr.min():.3f} "
              f"max={pw_arr.max():.3f} median={np.median(pw_arr):.3f}")

        pseudo_weights_final = pseudo_weights
        pseudo_tnorms_final = pseudo_tnorms
        pseudo_endpoints_final = pseudo_endpoints

        all_cameras = list(train_cameras_real) + list(pseudo_cams)
        all_images_np = list(train_images_real) + list(pseudo_imgs)
        all_conf_maps = [np.ones((c.height, c.width), dtype=np.float32) for c in train_cameras_real] + pseudo_confs
        view_weights = [1.0] * n_train_sparse + pseudo_weights

        n_pseudo = len(pseudo_cams)
        print(f"  Total training views: {len(all_cameras)} ({n_train_sparse} real + {n_pseudo} pseudo)")
        check_mem(f"after pseudo-view generation round {round_idx+1}")

        model_enhanced = _load_model_from_pts(top_idx)
        train_images_float = [img.astype(np.float32) / 255.0 for img in all_images_np]

        renderer_enhanced = GaussianRenderer(device=device)
        trainer = GaussianTrainer(
            model_enhanced, renderer_enhanced, device=device,
            confidence_maps=all_conf_maps,
            view_weights=view_weights,
            pixel_masks=None,
            depth_loss_fn=depth_loss_fn,
            lr_xyz=0.00004, lr_features=0.0005,
            lr_opacity=0.005, lr_scaling=0.001, lr_rotation=0.0002,
        )
        trainer.densify_from_iter = iterations + 1 if no_densify else 200
        trainer.densify_until_iter = iterations + 1 if no_densify else max(500, iters_per_round // 4)
        trainer.max_gaussians = max_gaussians
        trainer.opacity_reset_interval = iterations + 1

        check_mem(f"before retrain round {round_idx+1}")

        save_path = output_path / (
            'model_enhanced.npz' if round_idx == num_rounds - 1
            else f'model_round{round_idx+1}.npz'
        )
        train_result = trainer.train(
            all_cameras, train_images_float,
            iterations=iters_per_round,
            test_cameras=test_cameras,
            test_images=test_images_np,
            save_path=save_path,
            log_interval=200,
            eval_interval=500,
        )
        check_mem(f"after retrain round {round_idx+1}")

        if round_idx < num_rounds - 1:
            del current_model
            torch.cuda.empty_cache()
            current_model = model_enhanced
            print(f"  Round {round_idx+1} complete - using updated model for next round")

    # Final evaluation
    print("\nFinal evaluation on test set...")
    test_metrics = trainer.evaluate(test_cameras, test_images_np)

    # Save weights log
    weights_log = {
        'pseudo_weights': pseudo_weights_final,
        't_norms': pseudo_tnorms_final,
        'c_endpoints': pseudo_endpoints_final,
        'weight_cap': weight_cap,
    }
    with open(output_path / 'weights.json', 'w') as f:
        json.dump(weights_log, f, indent=2)

    total_time = time.time() - t_start
    results = {
        'dataset': str(part2_results['dataset']),
        'sparsity': sparsity,
        'num_sparse_views': n_recon,
        'num_train_sparse': n_train_sparse,
        'num_holdout_test': len(test_sparse_local_idx),
        'num_pseudo_views': n_pseudo_total,
        'num_total_views': n_train_sparse + n_pseudo_total,
        'baseline_psnr': baseline_psnr,
        'baseline_ssim': baseline_ssim,
        'baseline_lpips': baseline_lpips,
        'enhanced_psnr': test_metrics['psnr'],
        'enhanced_ssim': test_metrics['ssim'],
        'enhanced_lpips': test_metrics['lpips'],
        'delta_psnr': test_metrics['psnr'] - baseline_psnr,
        'delta_ssim': test_metrics['ssim'] - baseline_ssim,
        'delta_lpips': test_metrics['lpips'] - baseline_lpips,
        'training_iterations': iterations,
        'best_train_psnr': train_result['best_psnr'],
        'total_time': total_time,
        'version': 'reconx_v1',
        'weight_cap': weight_cap,
        'num_rounds': num_rounds,
        'preflight': preflight,
    }

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Part 3 ReconX Results:")
    print(f"{'='*70}")
    print(f"  {'':30s} {'Baseline':>12} {'Enhanced':>12} {'Delta':>8}")
    print(f"  {'-'*65}")
    print(f"  {'PSNR (dB)':30s} {baseline_psnr:>12.2f} {test_metrics['psnr']:>12.2f} {results['delta_psnr']:>+8.2f}")
    print(f"  {'SSIM':30s} {baseline_ssim:>12.4f} {test_metrics['ssim']:>12.4f} {results['delta_ssim']:>+8.4f}")
    print(f"  {'LPIPS':30s} {baseline_lpips:>12.4f} {test_metrics['lpips']:>12.4f} {results['delta_lpips']:>+8.4f}")
    print(f"{'='*70}")
    print(f"  Total time: {total_time:.0f}s  |  Output: {output_path}")
    print(f"{'='*70}\n")

    return results


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Part 3 ReconX: DynamiCrafter Pseudo-Views')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['dl3dv', 're10k', 'waymo'])
    parser.add_argument('--iterations', type=int, default=15000)
    parser.add_argument('--num_rounds', type=int, default=3)
    parser.add_argument('--max_gaussians', type=int, default=40000)
    parser.add_argument('--no_densify', action='store_true', default=True)
    parser.add_argument('--weight_cap', type=float, default=0.45)
    parser.add_argument('--part2_path', type=str, default=None)
    parser.add_argument('--dc_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    ds = args.dataset
    part2_path = args.part2_path or DATASET_MAP.get(ds, f"outputs/part2/{ds}")
    dc_path = args.dc_path or f"outputs/dc_output/{ds}"
    output_path = args.output_path or f"outputs/part3_reconx/{ds}"

    start_memory_monitor(threshold_gb=MEMORY_LIMIT_GB, check_interval=1.0)

    try:
        run_reconx_experiment(
            dataset_name=ds,
            part2_output_path=part2_path,
            dc_output_path=dc_path,
            output_path=output_path,
            iterations=args.iterations,
            device=args.device,
            num_rounds=args.num_rounds,
            max_gaussians=args.max_gaussians,
            no_densify=args.no_densify,
            weight_cap=args.weight_cap,
        )
    finally:
        stop_memory_monitor()
