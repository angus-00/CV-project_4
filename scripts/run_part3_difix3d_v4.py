"""Part 3 Difix3D v4: Alpha-Blended DiFix3D + Monocular Depth Supervision

Changes vs v3:
1. All v3 changes retained (70/30 alpha blend, skip blended-only, 3% mask, real_weights=1.0, max_per_pair=3)
2. NEW: Depth Anything V2 monocular depth maps pre-computed for training frames
3. NEW: Scale-invariant depth loss on real views (lambda=0.1, after iter 500)
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
from typing import Dict, List, Tuple
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


# ─────────────────────────────────────────────────────────
#  Camera helpers
# ─────────────────────────────────────────────────────────

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),      1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y),      2*(y*z+w*x),    1-2*(x**2+y**2)]
    ])

def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0:
        q2, dot = -q2, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        r = q1 + t * (q2 - q1)
        return r / np.linalg.norm(r)
    theta = np.arccos(dot)
    return (np.sin((1-t)*theta)/np.sin(theta))*q1 + (np.sin(t*theta)/np.sin(theta))*q2

MAX_RENDER_DIM = 512

def build_camera(cam_data: Dict, img_data: Dict) -> Camera:
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

def interp_camera(cam1_data: Tuple, cam2_data: Tuple, alpha: float) -> Tuple[Dict, Dict]:
    c1_cam, c1_img = cam1_data
    c2_cam, c2_img = cam2_data
    q_interp = slerp(np.array(c1_img['qvec']), np.array(c2_img['qvec']), alpha)
    t_interp = (1-alpha)*np.array(c1_img['tvec']) + alpha*np.array(c2_img['tvec'])
    cam_dict = c1_cam.copy()
    img_dict = {
        'qvec': q_interp.tolist(),
        'tvec': t_interp.tolist(),
        'camera_id': 0,
        'name': f'pseudo_{alpha:.3f}'
    }
    return cam_dict, img_dict

def quat_angle_deg(q1, q2):
    q1 = np.array(q1) / np.linalg.norm(q1)
    q2 = np.array(q2) / np.linalg.norm(q2)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0, 1))
    return float(np.degrees(np.arccos(dot)))


# ─────────────────────────────────────────────────────────
#  Confidence maps  (same as v3)
# ─────────────────────────────────────────────────────────

def compute_confidence_map(
    pseudo_img: np.ndarray,
    prev_img: np.ndarray,
    next_img: np.ndarray,
    rendered_img: np.ndarray,
    inpaint_mask: np.ndarray = None,
) -> np.ndarray:
    H, W = pseudo_img.shape[:2]

    render_err = np.abs(pseudo_img.astype(float) - rendered_img.astype(float)).mean(axis=2)
    render_err = render_err / (render_err.max() + 1e-8)
    render_conf = 1.0 - render_err

    def to_gray(img):
        return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    g_prev = to_gray(prev_img)
    g_cur  = to_gray(pseudo_img)

    flow_fwd = cv2.calcOpticalFlowFarneback(
        g_prev, g_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_bwd = cv2.calcOpticalFlowFarneback(
        g_cur, g_prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    fb_err = np.sqrt(
        (flow_fwd[:,:,0] + flow_bwd[:,:,0])**2 +
        (flow_fwd[:,:,1] + flow_bwd[:,:,1])**2
    )
    fb_err = fb_err / (np.percentile(fb_err, 95) + 1e-8)
    flow_conf = 1.0 - np.clip(fb_err, 0, 1)

    if inpaint_mask is not None:
        inpaint_region = (inpaint_mask > 0).astype(np.float32)
        inpaint_conf = 1.0 - 0.5 * inpaint_region
    else:
        inpaint_conf = np.ones((H, W), dtype=np.float32)

    combined = 0.4 * render_conf + 0.3 * flow_conf + 0.3 * inpaint_conf
    return np.clip(combined, 0.0, 1.0)


# ─────────────────────────────────────────────────────────
#  Mask: near-black hole detection (same as v3)
# ─────────────────────────────────────────────────────────

def _render_to_mask(rendered_np: np.ndarray, threshold: float = 15.0) -> np.ndarray:
    gray = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2GRAY).astype(float)
    hole_mask = (gray < threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hole_mask = cv2.dilate(hole_mask, kernel, iterations=2)
    return hole_mask



# ─────────────────────────────────────────────────────────
#  v4: Trajectory-aligned pseudo-view generation
# ─────────────────────────────────────────────────────────

# Same constants as render_demo_optionb.py
TARGET_DEG_PER_FRAME = 0.5
MIN_FRAMES_PER_PAIR  = 5
MAX_FRAMES_PER_PAIR  = 45

def compute_pair_alphas(angle_deg: float, max_per_pair: int,
                        alpha_min: float = 0.20, alpha_max: float = 0.70) -> List[float]:
    """Compute alpha values along the video trajectory for one pair.

    Returns interior alpha values filtered to [alpha_min, alpha_max] to avoid
    near-endpoint pseudo-views that are almost identical to real views.
    Uses the same adaptive interpolation as render_demo_optionb.py.
    """
    nframes = int(np.clip(
        int(np.ceil(angle_deg / TARGET_DEG_PER_FRAME)),
        MIN_FRAMES_PER_PAIR, MAX_FRAMES_PER_PAIR
    ))
    # interior alphas: j/nframes for j=1..nframes-1, filtered to middle range
    all_alphas = [j / nframes for j in range(1, nframes)
                  if alpha_min <= j / nframes <= alpha_max]
    if not all_alphas:
        # fallback: just use midpoint
        all_alphas = [0.5]
    if len(all_alphas) <= max_per_pair:
        return all_alphas
    # Uniformly subsample to max_per_pair
    indices = np.linspace(0, len(all_alphas) - 1, max_per_pair, dtype=int)
    return [all_alphas[i] for i in indices]


def generate_pseudo_views_v4(
    model: GaussianModel,
    renderer: GaussianRenderer,
    reconstruction: Dict,
    dataset_image_paths: List[str],
    max_per_pair: int = 8,
    local_indices: List[int] = None,
    output_path: Path = None,
    dataset_name_hint: str = "",
) -> Tuple[List[Camera], List[np.ndarray], List[np.ndarray], List, List[bool]]:
    """Generate trajectory-aligned pseudo-views.

    For each adjacent pair of sparse cameras, computes the same alpha sequence
    as the demo video and samples at most `max_per_pair` of those positions.
    """
    all_cameras      = []
    all_images       = []
    all_conf_maps    = []
    all_pixel_masks  = []
    all_inpainted    = []   # True if Difix3D was applied, False otherwise

    all_img_keys = sorted(reconstruction['images'].keys(), key=lambda x: int(x))
    all_cam_img_pairs = []
    for k in all_img_keys:
        img_data = reconstruction['images'][k]
        cam_id   = str(img_data.get('camera_id', k))
        cam_data = reconstruction['cameras'][cam_id]
        all_cam_img_pairs.append((cam_data, img_data))

    if local_indices is not None:
        cam_img_pairs = [all_cam_img_pairs[i] for i in local_indices]
    else:
        cam_img_pairs = all_cam_img_pairs

    n = len(cam_img_pairs)

    # Pre-compute pair angles and alpha lists
    pair_alphas = []
    total_pseudo = 0
    for i in range(n - 1):
        _, img1 = cam_img_pairs[i]
        _, img2 = cam_img_pairs[i + 1]
        ang = quat_angle_deg(img1['qvec'], img2['qvec'])
        alphas = compute_pair_alphas(ang, max_per_pair)
        pair_alphas.append(alphas)
        total_pseudo += len(alphas)
    print(f"  {n} training sparse views, trajectory-aligned pseudo-views: "
          f"{total_pseudo} total ({total_pseudo/(n-1):.1f}/pair avg, max {max_per_pair}/pair)")

    # Load + resize real sparse images
    sparse_imgs = []
    for idx_p, p in enumerate(dataset_image_paths):
        img = Image.open(p).convert('RGB')
        cam_data_ref, _ = cam_img_pairs[idx_p]
        orig_w, orig_h = cam_data_ref['width'], cam_data_ref['height']
        scale = min(1.0, MAX_RENDER_DIM / max(orig_w, orig_h))
        tw = max(1, int(orig_w * scale))
        th = max(1, int(orig_h * scale))
        if img.width != tw or img.height != th:
            img = img.resize((tw, th), Image.LANCZOS)
        sparse_imgs.append(np.array(img))

    # ── Render all pseudo-views ────────────────────────────
    pseudo_info = []
    render_idx  = 0

    for i in range(n):
        cam_data, img_data = cam_img_pairs[i]
        real_cam = build_camera(cam_data, img_data)
        all_cameras.append(real_cam)
        all_images.append(sparse_imgs[i])
        all_conf_maps.append(np.ones((real_cam.height, real_cam.width), dtype=np.float32))
        all_pixel_masks.append(None)
        all_inpainted.append(None)   # real view: not a pseudo-view

        if i < n - 1:
            check_mem(f"pseudo-view render loop i={i}")
            alphas = pair_alphas[i]
            for alpha in alphas:
                c_cam, c_img = interp_camera(cam_img_pairs[i], cam_img_pairs[i+1], alpha)
                pseudo_cam   = build_camera(c_cam, c_img)

                with torch.no_grad():
                    rendered_t = renderer.render(pseudo_cam, model)
                rendered_np = (np.clip(rendered_t.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)

                prev_img_rs = cv2.resize(sparse_imgs[i],   (pseudo_cam.width, pseudo_cam.height))
                next_img_rs = cv2.resize(sparse_imgs[i+1], (pseudo_cam.width, pseudo_cam.height))
                blended_np  = ((1-alpha)*prev_img_rs + alpha*next_img_rs).astype(np.uint8)

                mask_np  = _render_to_mask(rendered_np)
                coverage = (mask_np > 0).mean()
                if render_idx % 10 == 0:
                    print(f"    pseudo {render_idx:03d} (pair {i}-{i+1}, α={alpha:.3f}): "
                          f"mask coverage={coverage*100:.1f}%")

                pseudo_info.append({
                    'cam':          pseudo_cam,
                    'alpha':        alpha,
                    'rendered':     rendered_np,
                    'blended':      blended_np,
                    'mask':         mask_np,
                    'prev_img':     prev_img_rs,
                    'next_img':     next_img_rs,
                    'idx':          render_idx,
                    'inpaint_mask': mask_np,
                    'pair_i':       i,
                })
                render_idx += 1

    # ── Save inputs for Difix3D processing ──────────────────
    # output_path here is per-round (e.g. .../round_0), use parent for difix3d_io
    experiment_dir = output_path.parent if output_path and "round_" in output_path.name else output_path
    difix3d_base = experiment_dir / "difix3d_io" if experiment_dir else Path(f"outputs/part3_difix3d/{dataset_name_hint}/difix3d_io")
    ref_in = difix3d_base / "input"
    ref_in.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for info in pseudo_info:
        idx = info['idx']
        mask_frac = (info['mask'] > 0).mean()
        if mask_frac > 0.03:
            Image.fromarray(info['blended']).save(ref_in / f"image_{idx:04d}.png")
            Image.fromarray(info['mask']).save(ref_in / f"mask_{idx:04d}.png")
            # Save nearest training frame as reference for difix_ref
            alpha = info['alpha']
            ref_img = info['prev_img'] if alpha <= 0.5 else info['next_img']
            Image.fromarray(ref_img).save(ref_in / f"ref_{idx:04d}.png")
            n_saved += 1
    print(f"  Saved {n_saved} input/mask/ref triplets to {ref_in}")

    # ── Load pre-computed Difix3D inpainting results ─────────
    inpainted_results = {}
    difix3d_dir = difix3d_base / "output"

    if difix3d_dir.exists() and len(pseudo_info) > 0:
        for f in difix3d_dir.glob("inpainted_*.png"):
            idx = int(f.stem.split("_")[1])
            inpainted_results[idx] = np.array(Image.open(f).convert("RGB"))
        print(f"  Loaded {len(inpainted_results)} Difix3D inpainted views from {difix3d_dir}")
    else:
        print(f"  Difix3D outputs not found at {difix3d_dir}, using rendered images only")

    # ── Assemble final pseudo-views ────────────────────────
    # For non-LaMa views: keep at most 1 per pair (the one closest to alpha=0.5).
    # This prevents low-quality blended views from flooding training.
    from collections import defaultdict
    pair_best_blend = defaultdict(list)   # pair_key -> list of (dist_to_0.5, info)
    for info in pseudo_info:
        if info['idx'] not in inpainted_results:
            pair_key = (info.get('pair_i', info['idx']), )
            pair_best_blend[info['idx']].append(info)   # placeholder, handled below

    # Group non-inpainted by pair
    pair_noninpaint = defaultdict(list)
    for info in pseudo_info:
        if info['idx'] not in inpainted_results:
            pair_noninpaint[info.get('pair_i', -1)].append(info)

    # Build set of non-inpainted indices to keep (best alpha per pair)
    keep_noninpaint = set()
    for pair_i, infos in pair_noninpaint.items():
        best = min(infos, key=lambda x: abs(x['alpha'] - 0.5))
        keep_noninpaint.add(best['idx'])

    n_kept_blend = 0
    n_kept_lama  = 0
    for info in pseudo_info:
        idx         = info['idx']
        was_inpainted = idx in inpainted_results

        # v3: skip ALL non-inpainted views — blended-only adds noise
        if not was_inpainted:
            continue

        pcam        = info['cam']
        rendered_np = info['rendered']
        prev_img_rs = info['prev_img']
        next_img_rs = info['next_img']

        difix_img = cv2.resize(inpainted_results[idx], (pcam.width, pcam.height))
        # v3: 70% DiFix3D + 30% blended to reduce style drift
        final_img = (0.7 * difix_img.astype(np.float32) +
                     0.3 * info['blended'].astype(np.float32))
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        n_kept_lama += 1

        inpaint_mask = info['inpaint_mask']
        conf = compute_confidence_map(final_img, prev_img_rs, next_img_rs,
                                      info['blended'], inpaint_mask=inpaint_mask)

        all_cameras.append(pcam)
        all_images.append(final_img)
        all_conf_maps.append(conf.astype(np.float32))
        all_pixel_masks.append(None)
        all_inpainted.append(was_inpainted)

    print(f"  Pseudo-view selection: {n_kept_lama} Difix3D + {n_kept_blend} blended-only (1/pair max) = {n_kept_lama+n_kept_blend} total")
    return all_cameras, all_images, all_conf_maps, all_pixel_masks, all_inpainted


# ─────────────────────────────────────────────────────────
#  Main experiment
# ─────────────────────────────────────────────────────────

def run_part3_experiment(
    dataset_name: str,
    part2_output_path: str,
    output_path: str,
    max_per_pair: int = 8,
    iterations: int = 10000,
    device: str = 'cuda',
    num_rounds: int = 2,
    max_gaussians: int = 40000,
    no_densify: bool = False,
    sparsity_override: int = None,
) -> Dict:
    t_start     = time.time()
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    part2_path  = Path(part2_output_path)

    print(f"\n{'='*70}")
    print(f"Part 3 Difix3D: Trajectory-Aligned Pseudo-View Training")
    print(f"Dataset:         {dataset_name}")
    print(f"Part 2 path:     {part2_path}")
    print(f"Max pseudo/pair: {max_per_pair}")
    print(f"Iterations:      {iterations}  (rounds={num_rounds})")
    print(f"Max Gaussians:   {max_gaussians}")
    print(f"Memory limit:    {MEMORY_LIMIT_GB} GB")
    print(f"{'='*70}\n")

    check_mem("start")

    # ── Load Part 2 results ──────────────────────────────
    print("Loading Part 2 results...")
    with open(part2_path / 'reconstruction.json') as f:
        reconstruction = json.load(f)
    with open(part2_path / 'results.json') as f:
        part2_results = json.load(f)

    baseline_psnr  = part2_results['psnr']
    baseline_ssim  = part2_results['ssim']
    baseline_lpips = part2_results['lpips']
    print(f"  Baseline (Part 2): PSNR={baseline_psnr:.2f}, SSIM={baseline_ssim:.4f}, LPIPS={baseline_lpips:.4f}")

    # ── Load dataset ─────────────────────────────────────
    print("Loading dataset...")
    dataset  = SceneDataset(str(part2_results['dataset']))
    n_total  = len(dataset)

    n_recon  = len(reconstruction['images'])
    sparsity = part2_results['sparsity']

    all_indices        = np.arange(n_total)
    sparse_indices     = all_indices[::sparsity].tolist()
    if sparsity_override and sparsity_override != sparsity:
        override_indices   = all_indices[::sparsity_override].tolist()
        sparse_image_paths = [dataset.images[i] for i in override_indices[:n_recon]]
        print(f"  sparsity_override={sparsity_override}: using denser image files "
              f"(first {n_recon} of {len(override_indices)})")
    else:
        sparse_image_paths = [dataset.images[i] for i in sparse_indices[:n_recon]]

    n_sparse = n_recon
    test_sparse_local_idx  = list(range(0, n_sparse, 5))
    train_sparse_local_idx = [i for i in range(n_sparse) if i not in test_sparse_local_idx]
    if len(train_sparse_local_idx) < 3:
        train_sparse_local_idx = list(range(n_sparse))
        test_sparse_local_idx  = []

    n_train_sparse = len(train_sparse_local_idx)
    print(f"  Sparse views: {n_sparse} total, {n_train_sparse} train, {len(test_sparse_local_idx)} test")

    # ── Load Part 2 3DGS model ───────────────────────────
    print("Loading Part 2 3DGS model...")
    model_path = part2_path / 'model.npz'
    if model_path.is_dir():
        model_path = model_path / 'model.npz'
    pts_data = np.load(str(model_path))

    renderer = GaussianRenderer(device=device)
    check_mem("after model load")

    # ── Build test cameras ───────────────────────────────
    img_keys = sorted(reconstruction['images'].keys(), key=lambda x: int(x))
    test_cameras    = []
    test_images_np  = []
    for local_i in test_sparse_local_idx:
        k        = img_keys[local_i]
        img_data = reconstruction['images'][k]
        cam_id   = str(img_data.get('camera_id', k))
        cam_data = reconstruction['cameras'][cam_id]
        test_cam = build_camera(cam_data, img_data)
        test_cameras.append(test_cam)
        img = Image.open(sparse_image_paths[local_i]).convert('RGB')
        img = img.resize((test_cam.width, test_cam.height), Image.LANCZOS)
        test_images_np.append(np.array(img).astype(np.float32) / 255.0)

    train_sparse_image_paths = [sparse_image_paths[i] for i in train_sparse_local_idx]

    # ── v4: Pre-compute Depth Anything V2 depth maps ─────
    print("Pre-computing Depth Anything V2 depth maps for training frames...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    depth_model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_processor = AutoImageProcessor.from_pretrained(depth_model_name)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).to(device)
    depth_model.eval()

    # Build training cameras for depth map generation
    train_cameras_for_depth = []
    for local_i in train_sparse_local_idx:
        k = img_keys[local_i]
        img_data = reconstruction['images'][k]
        cam_id = str(img_data.get('camera_id', k))
        cam_data = reconstruction['cameras'][cam_id]
        train_cameras_for_depth.append(build_camera(cam_data, img_data))

    depth_maps_gt = {}  # idx -> (H, W) numpy array of inverse depth
    for i, local_i in enumerate(train_sparse_local_idx):
        img_pil = Image.open(sparse_image_paths[local_i]).convert('RGB')
        cam = train_cameras_for_depth[i]
        img_pil = img_pil.resize((cam.width, cam.height), Image.LANCZOS)
        inputs = depth_processor(images=img_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, h, w)
        # Resize to camera resolution
        depth_resized = F_depth.interpolate(
            predicted_depth.unsqueeze(0), size=(cam.height, cam.width),
            mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()  # (H, W), relative depth (larger = farther)
        # Convert to inverse depth (larger = closer) to match rasterizer output
        depth_maps_gt[i] = 1.0 / (depth_resized + 1e-6)
        print(f"  [{i}] depth range: {depth_resized.min():.3f} - {depth_resized.max():.3f}")

    # Free depth model VRAM
    del depth_model, depth_processor
    torch.cuda.empty_cache()
    print(f"  Computed {len(depth_maps_gt)} depth maps, freed depth model VRAM")
    check_mem("after depth pre-computation")

    # v4: depth loss callback
    DEPTH_LAMBDA = 0.01
    DEPTH_START_ITER = 500
    _depth_loss_log = []

    def depth_loss_fn(camera, model, renderer_obj, iteration, view_idx):
        """Scale-invariant depth loss for real training views only."""
        if iteration < DEPTH_START_ITER:
            return None
        if view_idx >= n_train_sparse:
            return None  # only real views have depth GT

        # Render inverse depth
        _, invdepth = renderer_obj.render(camera, model, return_depth=True)
        pred_invdepth = invdepth[:, :, 0]  # (H, W)

        # Get GT inverse depth
        gt_invdepth_np = depth_maps_gt[view_idx]
        gt_invdepth = torch.tensor(gt_invdepth_np, dtype=torch.float32, device=pred_invdepth.device)

        # Mask out invalid regions — use stricter threshold
        valid = (pred_invdepth > 0.01) & (gt_invdepth > 0.01) & (gt_invdepth < 1e4)
        if valid.sum() < 100:
            return None

        log_pred = torch.log(pred_invdepth[valid])
        log_gt = torch.log(gt_invdepth[valid])
        diff = log_pred - log_gt
        # Scale-invariant loss: var(diff) = mean(diff^2) - mean(diff)^2
        si_loss = (diff ** 2).mean() - 0.5 * (diff.mean() ** 2)

        weighted = DEPTH_LAMBDA * si_loss
        # Log every 500 iters
        if iteration % 500 == 0 and view_idx == 0:
            _depth_loss_log.append((iteration, si_loss.item(), weighted.item()))
            print(f"    [depth] iter={iteration} si_loss={si_loss.item():.4f} weighted={weighted.item():.6f} valid={valid.sum().item()}")

        return weighted

    # ── Iterative refinement loop ────────────────────────
    iters_per_round = max(1000, iterations // num_rounds)

    MAX_G = 150000
    xyz        = pts_data['xyz']
    opacity_np = 1.0 / (1.0 + np.exp(-pts_data['opacity'].squeeze()))
    if len(xyz) > MAX_G:
        top_idx = np.argsort(opacity_np)[::-1][:MAX_G]
    else:
        top_idx = np.arange(len(xyz))

    def _load_model_from_pts(top_idx):
        m = GaussianModel()
        m._xyz           = torch.nn.Parameter(torch.tensor(pts_data['xyz'][top_idx],           dtype=torch.float32).cuda())
        m._features_dc   = torch.nn.Parameter(torch.tensor(pts_data['features_dc'][top_idx],   dtype=torch.float32).cuda())
        m._features_rest = torch.nn.Parameter(torch.tensor(pts_data['features_rest'][top_idx], dtype=torch.float32).cuda())
        m._scaling       = torch.nn.Parameter(torch.tensor(pts_data['scaling'][top_idx],       dtype=torch.float32).cuda())
        m._rotation      = torch.nn.Parameter(torch.tensor(pts_data['rotation'][top_idx],      dtype=torch.float32).cuda())
        m._opacity       = torch.nn.Parameter(torch.tensor(pts_data['opacity'][top_idx],       dtype=torch.float32).cuda())
        m.sh_degree        = int(pts_data['sh_degree'])
        m.active_sh_degree = int(pts_data['active_sh_degree'])
        return m

    current_render_model = _load_model_from_pts(top_idx)
    print(f"  Loaded {len(top_idx)} Gaussians from Part 2 model")

    for round_idx in range(num_rounds):
        print(f"\n{'─'*60}")
        print(f"  Round {round_idx+1}/{num_rounds}  ({iters_per_round} iterations)")
        print(f"{'─'*60}")

        round_output = output_path / f"round_{round_idx}"
        all_cameras, all_images, all_conf_maps, all_pixel_masks, all_inpainted = generate_pseudo_views_v4(
            current_render_model, renderer, reconstruction,
            train_sparse_image_paths,
            max_per_pair=max_per_pair,
            local_indices=train_sparse_local_idx,
            output_path=round_output,
            dataset_name_hint=dataset_name,
        )

        n_pseudo = len(all_cameras) - n_train_sparse
        print(f"  Total training views: {len(all_cameras)} ({n_train_sparse} real + {n_pseudo} pseudo)")

        # Differentiated weights:
        #   real views:          3.0  (dominant: real GT drives texture/appearance)
        #   DiFix3D-inpainted:   clip(conf*0.8, 0.2, 0.8)  — same as LaMa v3
        #   blended-only:        0.15 (suppressed)
        real_weights   = [1.0] * n_train_sparse   # v3: match LaMa v3 weight balance
        pseudo_weights = []
        n_inpainted = 0
        for i in range(n_pseudo):
            conf_mean = float(all_conf_maps[n_train_sparse + i].mean())
            was_inpainted = all_inpainted[n_train_sparse + i]
            if was_inpainted:
                w = float(np.clip(conf_mean * 0.8, 0.2, 0.8))
                n_inpainted += 1
            else:
                w = 0.15
            pseudo_weights.append(w)
        view_weights  = real_weights + pseudo_weights
        pseudo_w_mean = float(np.mean(pseudo_weights)) if pseudo_weights else 0.0
        print(f"  Pseudo view weights: mean={pseudo_w_mean:.3f}  "
              f"min={min(pseudo_weights, default=0):.3f}  max={max(pseudo_weights, default=0):.3f}  "
              f"({n_inpainted} Difix3D-inpainted, {n_pseudo-n_inpainted} blended-only)")

        check_mem(f"after pseudo-view generation round {round_idx+1}")

        model_enhanced = _load_model_from_pts(top_idx)
        train_images_np = [img.astype(np.float32) / 255.0 for img in all_images]

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
        trainer.densify_from_iter  = iterations + 1 if no_densify else 200
        trainer.densify_until_iter = iterations + 1 if no_densify else max(500, iters_per_round // 4)
        trainer.max_gaussians      = max_gaussians
        trainer.opacity_reset_interval = iterations + 1  # disable opacity reset to preserve Part 2 opacity

        check_mem(f"before retrain round {round_idx+1}")

        save_path = output_path / (
            'model_enhanced.npz' if round_idx == num_rounds - 1
            else f'model_round{round_idx+1}.npz'
        )
        train_result = trainer.train(
            all_cameras, train_images_np,
            iterations=iters_per_round,
            test_cameras=test_cameras,
            test_images=test_images_np,
            save_path=save_path,
            log_interval=200,
            eval_interval=500,
        )
        check_mem(f"after retrain round {round_idx+1}")

        if round_idx < num_rounds - 1:
            del current_render_model
            torch.cuda.empty_cache()
            current_render_model = model_enhanced
            print(f"  Round {round_idx+1} complete — using updated model for next round")

    # ── Final evaluation ─────────────────────────────────
    print("\nFinal evaluation on test set...")
    test_metrics = trainer.evaluate(test_cameras, test_images_np)

    # ── Compile results ───────────────────────────────────
    total_time = time.time() - t_start
    results = {
        'dataset':              str(part2_results['dataset']),
        'sparsity':             sparsity_override or sparsity,
        'num_sparse_views':     len(sparse_indices),
        'num_train_sparse':     n_train_sparse,
        'num_holdout_test':     len(test_sparse_local_idx),
        'num_pseudo_views':     n_pseudo,
        'max_per_pair':         max_per_pair,
        'num_total_views':      len(all_cameras),
        'baseline_psnr':        baseline_psnr,
        'baseline_ssim':        baseline_ssim,
        'baseline_lpips':       baseline_lpips,
        'enhanced_psnr':        test_metrics['psnr'],
        'enhanced_ssim':        test_metrics['ssim'],
        'enhanced_lpips':       test_metrics['lpips'],
        'delta_psnr':           test_metrics['psnr']  - baseline_psnr,
        'delta_ssim':           test_metrics['ssim']  - baseline_ssim,
        'delta_lpips':          test_metrics['lpips'] - baseline_lpips,
        'training_iterations':  iterations,
        'best_train_psnr':      train_result['best_psnr'],
        'total_time':           total_time,
        'version':              'difix3d_v4',
    }

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Part 3 Difix3D Results:")
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

    parser = argparse.ArgumentParser(description='Part 3 Difix3D v4: Pseudo-Views + Depth Supervision')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['dl3dv', 're10k', 'waymo', 'flowers', 'treehill'])
    parser.add_argument('--max_per_pair', type=int, default=3,
                        help='Max pseudo-views per pair (sampled from video trajectory)')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--num_rounds', type=int, default=2)
    parser.add_argument('--max_gaussians', type=int, default=40000)
    parser.add_argument('--no_densify', action='store_true')
    parser.add_argument('--sparsity', type=int, default=None)
    parser.add_argument('--part2_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    ds          = args.dataset
    part2_path  = args.part2_path  or f"outputs/part2/{ds}"
    output_path = args.output_path or f"outputs/part3_difix3d/{ds}"

    start_memory_monitor(threshold_gb=MEMORY_LIMIT_GB, check_interval=1.0)

    try:
        run_part3_experiment(
            dataset_name=ds,
            part2_output_path=part2_path,
            output_path=output_path,
            max_per_pair=args.max_per_pair,
            iterations=args.iterations,
            device=args.device,
            num_rounds=args.num_rounds,
            max_gaussians=args.max_gaussians,
            no_densify=args.no_densify,
            sparsity_override=args.sparsity,
        )
    finally:
        stop_memory_monitor()
