"""Render a side-by-side demo video: Part2 (sparse-only) vs Part3 (enhanced).

Layout per frame:
  [Part2 render | Part3 render | Ground-truth blend]

Camera trajectory: SLERP interpolation between consecutive sparse cameras.

Usage:
    python scripts/render_demo_video.py --dataset waymo
    python scripts/render_demo_video.py --dataset waymo --num_interp 20 --fps 30
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from common.camera import Camera
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.renderer import GaussianRenderer


# ─────────────────────────────────────────────────────────
#  Helpers (reuse patterns from run_part3.py)
# ─────────────────────────────────────────────────────────

MAX_RENDER_DIM = 768

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),      1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y),      2*(y*z+w*x),    1-2*(x**2+y**2)]
    ])

def slerp(q1, q2, t):
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

def build_camera(cam_data, img_data):
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

def interp_camera(c1_cam, c1_img, c2_cam, c2_img, alpha):
    """Return interpolated Camera object."""
    q_interp = slerp(np.array(c1_img['qvec']), np.array(c2_img['qvec']), alpha)
    t_interp = (1-alpha)*np.array(c1_img['tvec']) + alpha*np.array(c2_img['tvec'])
    fake_img = {'qvec': q_interp.tolist(), 'tvec': t_interp.tolist(), 'camera_id': 0}
    return build_camera(c1_cam, fake_img)

def load_model(model_path: str, prune_floaters: bool = True) -> GaussianModel:
    p = Path(model_path)
    if p.is_dir():
        p = p / 'model.npz'
    data = np.load(str(p))

    xyz           = data['xyz']
    features_dc   = data['features_dc']
    features_rest = data['features_rest']
    scaling       = data['scaling']
    rotation      = data['rotation']
    opacity_raw   = data['opacity']

    if prune_floaters:
        # Remove floater Gaussians that cause "light explosion" at novel viewpoints.
        # Three-tier strategy (video rendering only — does not affect PSNR metrics):
        #   Tier 1: absolute scale > 3.0 → always a floater, remove unconditionally
        #   Tier 2: scale > p95 AND opacity < 0.1 → large + low-opacity floater
        #   Tier 3: scale > p99 → unconditionally remove extreme outliers
        scales     = np.exp(scaling)
        max_scale  = scales.max(axis=1)
        opacity    = 1.0 / (1.0 + np.exp(-opacity_raw.squeeze()))

        abs_mask   = max_scale > 3.0
        scale_p95  = np.percentile(max_scale, 95)
        scale_p99  = np.percentile(max_scale, 99)
        tier2_mask = (max_scale > scale_p95) & (opacity < 0.1)
        tier3_mask = max_scale > scale_p99
        floater_mask = abs_mask | tier2_mask | tier3_mask
        keep         = ~floater_mask

        n_before = len(xyz)
        xyz           = xyz[keep]
        features_dc   = features_dc[keep]
        features_rest = features_rest[keep]
        scaling       = scaling[keep]
        rotation      = rotation[keep]
        opacity_raw   = opacity_raw[keep]
        print(f"    [prune] removed {floater_mask.sum()}/{n_before} Gaussians "
              f"(abs>3: {abs_mask.sum()}, p95+opacity<0.1: {tier2_mask.sum()}, p99: {tier3_mask.sum()})")

    model = GaussianModel()
    model._xyz           = torch.nn.Parameter(torch.tensor(xyz,           dtype=torch.float32).cuda())
    model._features_dc   = torch.nn.Parameter(torch.tensor(features_dc,   dtype=torch.float32).cuda())
    model._features_rest = torch.nn.Parameter(torch.tensor(features_rest, dtype=torch.float32).cuda())
    model._scaling       = torch.nn.Parameter(torch.tensor(scaling,       dtype=torch.float32).cuda())
    model._rotation      = torch.nn.Parameter(torch.tensor(rotation,      dtype=torch.float32).cuda())
    model._opacity       = torch.nn.Parameter(torch.tensor(opacity_raw,   dtype=torch.float32).cuda())
    model.sh_degree        = int(data['sh_degree'])
    model.active_sh_degree = int(data['active_sh_degree'])
    model.eval()
    return model

def render_np(model, renderer, camera):
    with torch.no_grad():
        img = renderer.render(camera, model)
    img_np = img.detach().cpu().numpy()
    # Detect overexposed frames (mean > 0.85 in float space = likely floater artifact).
    # Apply Reinhard only to those frames; leave normal frames untouched.
    if img_np.mean() > 0.85:
        img_np = img_np / (1.0 + img_np)
    return np.clip(img_np, 0, 1)

def to_uint8(img_float):
    return (img_float * 255).astype(np.uint8)

def add_label(arr: np.ndarray, text: str, color=(255,255,255)) -> np.ndarray:
    """Burn a text label into the top-left corner of a uint8 HxWx3 array."""
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # Shadow for readability
    draw.text((11, 11), text, fill=(0, 0, 0))
    draw.text((10, 10), text, fill=color)
    return np.array(img)

def make_side_by_side(p2_np, p3_np, gt_np, p2_psnr, p3_psnr):
    """Stack [Part2 | Part3 | GT] horizontally with labels."""
    h, w = p2_np.shape[:2]
    gt_np = gt_np if gt_np is not None else np.zeros_like(p2_np)

    p2_u8 = add_label(to_uint8(p2_np), f"Part2 Sparse\nPSNR {p2_psnr:.1f}dB")
    p3_u8 = add_label(to_uint8(p3_np), f"Part3 Enhanced\nPSNR {p3_psnr:.1f}dB", color=(100,255,100))
    gt_u8 = add_label(to_uint8(gt_np),  "GT Blend")

    # 2-pixel separator
    sep = np.zeros((h, 2, 3), dtype=np.uint8)
    frame = np.concatenate([p2_u8, sep, p3_u8, sep, gt_u8], axis=1)
    return frame


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def render_video(dataset: str, num_interp: int, fps: int, device: str,
                 part2_path: str = None, part3_path: str = None,
                 ema_alpha: float = 0.5):
    part2_dir = Path(part2_path) if part2_path else Path(f"outputs/part2/{dataset}")
    part3_dir = Path(part3_path) if part3_path else Path(f"outputs/part3/{dataset}")
    frames_dir = part3_dir / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reconstruction and results for {dataset}...")
    with open(part2_dir / 'reconstruction.json') as f:
        recon = json.load(f)
    with open(part3_dir / 'results.json') as f:
        p3_results = json.load(f)
    with open(part2_dir / 'results.json') as f:
        p2_results = json.load(f)

    renderer = GaussianRenderer(device=device)

    # Build ordered list of reconstruction cameras
    img_keys = sorted(recon['images'].keys(), key=lambda x: int(x))
    cam_pairs = []
    for k in img_keys:
        img_data = recon['images'][k]
        cam_id   = str(img_data.get('camera_id', k))
        cam_data = recon['cameras'][cam_id]
        cam_pairs.append((cam_data, img_data))

    n_cams = len(cam_pairs)

    # Load dataset images for GT blend
    from common.dataset import SceneDataset
    dataset_obj = SceneDataset(str(p2_results['dataset']))
    sparsity    = p2_results['sparsity']
    sparse_imgs = [dataset_obj.images[i] for i in range(0, len(dataset_obj), sparsity)]

    # ── Compute angular distance between consecutive sparse cameras ──
    def quat_angle_deg(q1, q2):
        q1 = np.array(q1) / np.linalg.norm(q1)
        q2 = np.array(q2) / np.linalg.norm(q2)
        dot = float(np.clip(abs(np.dot(q1, q2)), 0, 1))
        return float(np.degrees(np.arccos(dot)))

    pair_angles = []
    for i in range(n_cams - 1):
        ang = quat_angle_deg(cam_pairs[i][1]['qvec'], cam_pairs[i+1][1]['qvec'])
        pair_angles.append(ang)

    # ── Adaptive interp: target ≤ 0.5° per frame, capped at [5, 45] frames ──
    TARGET_DEG_PER_FRAME = 0.5
    pair_nframes = []
    for ang in pair_angles:
        n = int(np.ceil(ang / TARGET_DEG_PER_FRAME))
        n = int(np.clip(n, 5, 45))
        pair_nframes.append(n)

    total_frames = sum(pair_nframes) + 1
    print(f"  {n_cams} cameras, adaptive interp: {pair_nframes} frames/pair "
          f"(angles: {[f'{a:.1f}' for a in pair_angles]})")
    print(f"  Total frames: {total_frames}")
    print(f"  Rendering {total_frames} frames (sequential: Part2 then Part3 to save VRAM)...")

    # ── Build trajectory with equal-arc-length sampling ──────────────
    traj_cameras = []
    traj_gt      = []
    for i in range(n_cams - 1):
        c1_cam, c1_img = cam_pairs[i]
        c2_cam, c2_img = cam_pairs[i + 1]
        gt1 = np.array(Image.open(sparse_imgs[min(i,   len(sparse_imgs)-1)]).convert('RGB')).astype(np.float32)/255.0
        gt2 = np.array(Image.open(sparse_imgs[min(i+1, len(sparse_imgs)-1)]).convert('RGB')).astype(np.float32)/255.0
        nf = pair_nframes[i]
        for j in range(nf):
            alpha = j / nf
            traj_cameras.append(interp_camera(c1_cam, c1_img, c2_cam, c2_img, alpha))
            traj_gt.append((gt1, gt2, alpha))
    # Last camera
    c_last_cam, c_last_img = cam_pairs[-1]
    traj_cameras.append(build_camera(c_last_cam, c_last_img))
    gt_last = np.array(Image.open(sparse_imgs[-1]).convert('RGB')).astype(np.float32)/255.0
    traj_gt.append((gt_last, gt_last, 0.0))

    def render_all(model_path, label, ema_alpha=0.5):
        """Render all trajectory frames with one model, return list of float32 HxWx3.

        ema_alpha: exponential moving average weight for temporal smoothing.
                   Each frame = ema_alpha * prev_smoothed + (1 - ema_alpha) * current.
                   Set to 0.0 to disable.
        """
        print(f"  Loading {label} model...")
        model = load_model(model_path)
        renders = []
        valid_count = 0   # frames actually rendered (not zero-padded)
        ema_frame = None
        for idx, cam in enumerate(traj_cameras):
            raw = render_np(model, renderer, cam)
            # EMA temporal smoothing to reduce flicker
            if ema_frame is None:
                ema_frame = raw
            else:
                ema_frame = ema_alpha * ema_frame + (1 - ema_alpha) * raw
            renders.append(ema_frame.copy())
            valid_count += 1
            if idx % 20 == 0:
                used = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                print(f"    {label} frame {idx}/{len(traj_cameras)}  VRAM={used:.1f}GB")
                if used > 15.5:
                    print(f"  [STOP] VRAM={used:.1f}GB > 15.5GB, truncating at frame {idx}")
                    # Mark remaining as invalid (None) instead of zero-padding
                    renders.extend([None] * (len(traj_cameras) - len(renders)))
                    break
        del model
        torch.cuda.empty_cache()
        print(f"    {label}: {valid_count}/{len(traj_cameras)} frames rendered")
        return renders

    # Render both models sequentially (never both in VRAM at once)
    renders_p2 = render_all(str(part2_dir / 'model.npz'),         "Part2", ema_alpha=ema_alpha)
    renders_p3 = render_all(str(part3_dir / 'model_enhanced.npz'), "Part3", ema_alpha=ema_alpha)

    # Composite frames from pre-rendered lists, skipping invalid (None) frames
    def psnr(a, b):
        mse = np.mean((a - b) ** 2)
        return 100.0 if mse < 1e-10 else float(20 * np.log10(1.0 / np.sqrt(mse)))

    frame_idx = 0
    skipped = 0
    for fi in range(len(traj_cameras)):
        p2_np = renders_p2[fi]
        p3_np = renders_p3[fi]

        # Skip frames where either model was truncated (VRAM limit)
        if p2_np is None or p3_np is None:
            skipped += 1
            continue

        gt1, gt2, alpha = traj_gt[fi]

        h, w = p2_np.shape[:2]
        gt1_r = np.array(Image.fromarray((gt1*255).astype(np.uint8)).resize((w, h), Image.LANCZOS)).astype(np.float32)/255.0
        gt2_r = np.array(Image.fromarray((gt2*255).astype(np.uint8)).resize((w, h), Image.LANCZOS)).astype(np.float32)/255.0
        gt_blend = (1 - alpha) * gt1_r + alpha * gt2_r

        p2_psnr = psnr(p2_np, gt_blend)
        p3_psnr = psnr(p3_np, gt_blend)

        frame = make_side_by_side(p2_np, p3_np, gt_blend, p2_psnr, p3_psnr)
        Image.fromarray(frame).save(str(frames_dir / f"frame_{frame_idx:04d}.png"))

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"  Composited frame {frame_idx}  "
                  f"Part2 PSNR={p2_psnr:.1f} Part3 PSNR={p3_psnr:.1f}")

    if skipped:
        print(f"  Skipped {skipped} truncated frames (VRAM limit)")

    print(f"\nSaved {frame_idx} frames to {frames_dir}")

    # Encode with ffmpeg — try common locations
    ffmpeg_exe = "ffmpeg"
    for candidate in [
        r"C:\Users\Administrator\AppData\Local\oopz\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]:
        if Path(candidate).exists():
            ffmpeg_exe = candidate
            break

    video_path = part3_dir / f"demo_{dataset}.mp4"
    ffmpeg_cmd = [
        ffmpeg_exe, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-crf", "18",
        "-vf", "unsharp=5:5:0.8:3:3:0.0",
        "-pix_fmt", "yuv420p",
        str(video_path)
    ]
    print(f"Encoding video: {' '.join(ffmpeg_cmd)}")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        size_mb = video_path.stat().st_size / 1e6
        print(f"Video saved: {video_path}  ({size_mb:.1f} MB)")
    else:
        print(f"ffmpeg failed: {result.stderr[-300:]}")
        print(f"Frames are available at: {frames_dir}")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render Part2 vs Part3 demo video')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['waymo', 'dl3dv', 're10k', 'flowers', 'treehill'])
    parser.add_argument('--num_interp', type=int, default=15,
                        help='Interpolated frames between each pair of sparse cameras (default 15)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video frame rate (default 30)')
    parser.add_argument('--part2_path', type=str, default=None,
                        help='Override Part2 directory (default: outputs/part2/{dataset})')
    parser.add_argument('--part3_path', type=str, default=None,
                        help='Override Part3 directory (default: outputs/part3/{dataset})')
    parser.add_argument('--no_ema', action='store_true',
                        help='Disable EMA temporal smoothing (removes motion blur)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    render_video(args.dataset, args.num_interp, args.fps, args.device,
                 part2_path=args.part2_path, part3_path=args.part3_path,
                 ema_alpha=0.0 if args.no_ema else 0.5)
