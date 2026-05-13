"""Generate visual comparison figures for Part 1 and Part 3.

Part 1: [COLMAP render | DUSt3R render | GT]  x N test views
Part 3: [Part2 baseline render | Part3 enhanced render | GT]  x N test views

Usage:
    python scripts/visualize_comparison.py
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from common.camera import Camera
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.renderer import GaussianRenderer

DEVICE = 'cuda'
MAX_DIM = 512   # cap render resolution


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])


def build_camera_from_dict(cam_data, img_data):
    params = cam_data['params']
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    orig_w, orig_h = cam_data['width'], cam_data['height']
    scale = min(1.0, MAX_DIM / max(orig_w, orig_h))
    w = max(1, int(orig_w * scale))
    h = max(1, int(orig_h * scale))
    K = np.array([[fx*scale, 0, cx*scale],
                  [0, fy*scale, cy*scale],
                  [0, 0, 1]])
    R = quat_to_rot(np.array(img_data['qvec']))
    t = np.array(img_data['tvec'])
    return Camera(R, t, K, w, h)


def load_gaussian_model(model_path: str):
    p = Path(model_path)
    if p.is_dir():
        p = p / 'model.npz'
    data = np.load(str(p))
    model = GaussianModel()
    model._xyz           = torch.nn.Parameter(torch.tensor(data['xyz'],          dtype=torch.float32).cuda())
    model._features_dc   = torch.nn.Parameter(torch.tensor(data['features_dc'],  dtype=torch.float32).cuda())
    model._features_rest = torch.nn.Parameter(torch.tensor(data['features_rest'],dtype=torch.float32).cuda())
    model._scaling       = torch.nn.Parameter(torch.tensor(data['scaling'],       dtype=torch.float32).cuda())
    model._rotation      = torch.nn.Parameter(torch.tensor(data['rotation'],      dtype=torch.float32).cuda())
    model._opacity       = torch.nn.Parameter(torch.tensor(data['opacity'],       dtype=torch.float32).cuda())
    model.sh_degree        = int(data['sh_degree'])
    model.active_sh_degree = int(data['active_sh_degree'])
    return model


def render_to_np(model, renderer, camera):
    with torch.no_grad():
        r = renderer.render(camera, model)
    return np.clip(r.detach().cpu().numpy(), 0, 1)


def psnr_np(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


# ─────────────────────────────────────────────────────────
#  Part 3 visualisation
# ─────────────────────────────────────────────────────────

def make_part3_figure(dataset_name: str,
                      part2_dir: str,
                      part3_dir: str,
                      out_path: str,
                      n_views: int = 4):
    """
    For each of n_views test cameras render:
      col0: Part-2 baseline model
      col1: Part-3 enhanced model
      col2: GT image
    Plus PSNR annotations.
    """
    p2 = Path(part2_dir)
    p3 = Path(part3_dir)

    with open(p2 / 'results.json')      as f: p2_res = json.load(f)
    with open(p3 / 'results.json')      as f: p3_res = json.load(f)
    with open(p2 / 'reconstruction.json') as f: recon = json.load(f)

    from common.dataset import SceneDataset
    dataset   = SceneDataset(str(p2_res['dataset']))
    sparsity  = p2_res['sparsity']
    all_idx   = np.arange(len(dataset))
    sparse_idx = all_idx[::sparsity].tolist()
    n_sparse   = len(sparse_idx)

    # hold-out test = every 5th sparse view
    test_local = list(range(0, n_sparse, 5))
    test_global = [sparse_idx[i] for i in test_local]

    if not test_global:
        print(f"  [warn] no holdout test views for {dataset_name}")
        return

    n_views = min(n_views, len(test_global))

    renderer = GaussianRenderer(device=DEVICE)
    model_p2 = load_gaussian_model(str(p2 / 'model.npz'))
    model_p3 = load_gaussian_model(str(p3 / 'model_enhanced.npz'))
    model_p2.eval(); model_p3.eval()

    img_keys = sorted(recon['images'].keys(), key=lambda x: int(x))

    fig_rows = n_views
    fig, axes = plt.subplots(fig_rows, 3,
                             figsize=(12, 4 * fig_rows),
                             squeeze=False)

    for row, local_i in enumerate(test_local[:n_views]):
        k = img_keys[local_i]
        img_data = recon['images'][k]
        cam_id   = str(img_data.get('camera_id', k))
        cam_data = recon['cameras'][cam_id]
        cam      = build_camera_from_dict(cam_data, img_data)

        # GT
        gt_path = dataset.images[sparse_idx[local_i]]
        gt_img  = np.array(Image.open(gt_path).convert('RGB').resize(
            (cam.width, cam.height), Image.LANCZOS)).astype(np.float32) / 255.0

        # Renders
        r_p2 = render_to_np(model_p2, renderer, cam)
        r_p3 = render_to_np(model_p3, renderer, cam)

        psnr_p2 = psnr_np(r_p2, gt_img)
        psnr_p3 = psnr_np(r_p3, gt_img)

        for col, (img, label) in enumerate([
            (r_p2, f'Part2 Baseline\nPSNR={psnr_p2:.1f}dB'),
            (r_p3, f'Part3 Enhanced\nPSNR={psnr_p3:.1f}dB'),
            (gt_img, 'Ground Truth'),
        ]):
            ax = axes[row][col]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(label, fontsize=9)
            ax.axis('off')

    fig.suptitle(f'Part 3: Sparse-Only vs Sparse+Generated — {dataset_name}',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Clean up GPU memory
    del model_p2, model_p3
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────
#  Part 1 visualisation
# ─────────────────────────────────────────────────────────

def make_part1_figure(dataset_name: str,
                      part1_dir: str,
                      out_path: str,
                      n_views: int = 3):
    """
    Visualise Part 1 results: COLMAP vs DUSt3R metrics bar chart +
    training PSNR curves on one figure.
    If rendered models are not stored, show only the metrics / training curves.
    """
    p1 = Path(part1_dir)
    comp_json = p1 / 'comparison.json'
    if not comp_json.exists():
        print(f"  [skip] {comp_json} not found"); return

    with open(comp_json) as f:
        comp = json.load(f)

    colmap_ok = comp['colmap'].get('success', False)
    found_ok  = comp['foundation'].get('success', False)

    # ── figure layout ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1) Metrics bar chart ───────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 0])
    metrics = ['PSNR (dB)', 'SSIM', 'LPIPS']
    qc = comp.get('summary', {}).get('quality_comparison', {})
    c_vals = [qc.get('colmap_psnr', 0),  qc.get('colmap_ssim', 0),  qc.get('colmap_lpips', 0)]
    f_vals = [qc.get('foundation_psnr', 0), qc.get('foundation_ssim', 0), qc.get('foundation_lpips', 0)]

    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax_bar.bar(x - w/2, c_vals, w, label='COLMAP',  color='steelblue', alpha=0.85)
    b2 = ax_bar.bar(x + w/2, f_vals, w, label='DUSt3R',  color='tomato',    alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metrics, fontsize=8)
    ax_bar.set_title('Rendering Quality', fontsize=10)
    ax_bar.legend(fontsize=8); ax_bar.grid(axis='y', alpha=0.3)

    # ── 2) Training PSNR curve ─────────────────────────────
    ax_psnr = fig.add_subplot(gs[0, 1])
    c_hist = comp['colmap'].get('training_history', {}) or {}
    f_hist = comp['foundation'].get('training_history', {}) or {}
    if c_hist.get('psnr') and c_hist.get('iteration'):
        ax_psnr.plot(c_hist['iteration'], c_hist['psnr'],
                     'b-', label='COLMAP', linewidth=1.8)
    if f_hist.get('psnr') and f_hist.get('iteration'):
        ax_psnr.plot(f_hist['iteration'], f_hist['psnr'],
                     'r-', label='DUSt3R', linewidth=1.8)
    ax_psnr.set_xlabel('Iteration', fontsize=8)
    ax_psnr.set_ylabel('Train PSNR (dB)', fontsize=8)
    ax_psnr.set_title('Convergence Curve', fontsize=10)
    ax_psnr.legend(fontsize=8); ax_psnr.grid(alpha=0.3)

    # ── 3) Gaussian count curve ────────────────────────────
    ax_gauss = fig.add_subplot(gs[0, 2])
    if c_hist.get('num_gaussians') and c_hist.get('iteration'):
        ax_gauss.plot(c_hist['iteration'], c_hist['num_gaussians'],
                      'b-', label='COLMAP', linewidth=1.8)
    if f_hist.get('num_gaussians') and f_hist.get('iteration'):
        ax_gauss.plot(f_hist['iteration'], f_hist['num_gaussians'],
                      'r-', label='DUSt3R', linewidth=1.8)
    ax_gauss.set_xlabel('Iteration', fontsize=8)
    ax_gauss.set_ylabel('# Gaussians', fontsize=8)
    ax_gauss.set_title('Gaussian Count', fontsize=10)
    ax_gauss.legend(fontsize=8); ax_gauss.grid(alpha=0.3)

    # ── 4) Point cloud / camera count summary bar ──────────
    ax_init = fig.add_subplot(gs[1, 0])
    rc = comp.get('summary', {}).get('reconstruction_comparison', {})
    tc = comp.get('summary', {}).get('time_comparison', {})
    labels2 = ['Cameras', 'Points (k)', 'Time (s)']
    c2 = [rc.get('colmap_cameras', 0),
          rc.get('colmap_points', 0) / 1000,
          tc.get('colmap', 0)]
    f2 = [rc.get('foundation_cameras', 0),
          rc.get('foundation_points', 0) / 1000,
          tc.get('foundation', 0)]
    x2 = np.arange(len(labels2))
    b3 = ax_init.bar(x2 - w/2, c2, w, label='COLMAP',  color='steelblue', alpha=0.85)
    b4 = ax_init.bar(x2 + w/2, f2, w, label='DUSt3R',  color='tomato',    alpha=0.85)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax_init.text(bar.get_x() + bar.get_width()/2, h + max(h*0.01, 0.5),
                     f'{h:.0f}', ha='center', va='bottom', fontsize=7)
    ax_init.set_xticks(x2); ax_init.set_xticklabels(labels2, fontsize=8)
    ax_init.set_title('Initialization Stats', fontsize=10)
    ax_init.legend(fontsize=8); ax_init.grid(axis='y', alpha=0.3)

    # ── 5) Loss curve ──────────────────────────────────────
    ax_loss = fig.add_subplot(gs[1, 1])
    if c_hist.get('loss') and c_hist.get('iteration'):
        ax_loss.plot(c_hist['iteration'], c_hist['loss'],
                     'b-', label='COLMAP', linewidth=1.8)
    if f_hist.get('loss') and f_hist.get('iteration'):
        ax_loss.plot(f_hist['iteration'], f_hist['loss'],
                     'r-', label='DUSt3R', linewidth=1.8)
    ax_loss.set_xlabel('Iteration', fontsize=8)
    ax_loss.set_ylabel('L1 Loss', fontsize=8)
    ax_loss.set_title('Training Loss', fontsize=10)
    ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)

    # ── 6) Summary text ────────────────────────────────────
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis('off')
    lines = [
        f'Dataset: {dataset_name}',
        '',
        f"COLMAP: {'OK' if colmap_ok else 'FAILED'}",
        f"DUSt3R: {'OK' if found_ok  else 'FAILED'}",
        '',
    ]
    if qc:
        dpsnr = qc.get('psnr_delta', 0)
        lines += [
            f"PSNR delta: {dpsnr:+.2f} dB",
            f"(DUSt3R vs COLMAP)",
            '',
            f"COLMAP time: {tc.get('colmap', 0):.1f}s",
            f"DUSt3R time: {tc.get('foundation', 0):.1f}s",
        ]
    ax_txt.text(0.05, 0.95, '\n'.join(lines),
                transform=ax_txt.transAxes,
                fontsize=10, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax_txt.set_title('Summary', fontsize=10)

    fig.suptitle(f'Part 1: COLMAP vs DUSt3R Initialization — {dataset_name}',
                 fontsize=14)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────
#  Part 3 multi-dataset summary bar chart
# ─────────────────────────────────────────────────────────

def make_part3_summary(datasets, out_path):
    """Bar chart: baseline PSNR vs enhanced PSNR for each dataset."""
    names, base_psnr, enh_psnr, base_ssim, enh_ssim = [], [], [], [], []
    for ds in datasets:
        p3 = Path(f'outputs/part3/{ds}')
        r_file = p3 / 'results.json'
        if not r_file.exists():
            continue
        with open(r_file) as f:
            r = json.load(f)
        names.append(ds)
        base_psnr.append(r['baseline_psnr'])
        enh_psnr.append(r['enhanced_psnr'])
        base_ssim.append(r['baseline_ssim'])
        enh_ssim.append(r['enhanced_ssim'])

    if not names:
        print("  [skip] no Part 3 results found"); return

    x = np.arange(len(names))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    b1 = ax.bar(x - w/2, base_psnr, w, label='Part2 (Sparse Only)', color='steelblue', alpha=0.85)
    b2 = ax.bar(x + w/2, enh_psnr,  w, label='Part3 (+Generated)',   color='tomato',   alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f'{h:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('PSNR (dB)'); ax.set_title('PSNR: Sparse vs Sparse+Generated')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    b3 = ax.bar(x - w/2, base_ssim, w, label='Part2 (Sparse Only)', color='steelblue', alpha=0.85)
    b4 = ax.bar(x + w/2, enh_ssim,  w, label='Part3 (+Generated)',   color='tomato',   alpha=0.85)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('SSIM'); ax.set_title('SSIM: Sparse vs Sparse+Generated')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Part 3: Effect of Generative View Completion', fontsize=13)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────
#  Part 1 multi-dataset summary
# ─────────────────────────────────────────────────────────

def make_part1_summary(datasets, out_path):
    """Summary bar chart: COLMAP vs DUSt3R across datasets."""
    names, c_psnr, f_psnr, c_ssim, f_ssim = [], [], [], [], []
    for ds in datasets:
        p1 = Path(f'outputs/part1/{ds}')
        cj = p1 / 'comparison.json'
        if not cj.exists():
            continue
        with open(cj) as f:
            c = json.load(f)
        if not c['colmap'].get('success') or not c['foundation'].get('success'):
            continue
        qc = c.get('summary', {}).get('quality_comparison', {})
        names.append(ds)
        c_psnr.append(qc.get('colmap_psnr', 0))
        f_psnr.append(qc.get('foundation_psnr', 0))
        c_ssim.append(qc.get('colmap_ssim', 0))
        f_ssim.append(qc.get('foundation_ssim', 0))

    if not names:
        print("  [skip] no complete Part 1 comparison data"); return

    x = np.arange(len(names))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (c_vals, f_vals, ylabel, title) in zip(axes, [
        (c_psnr, f_psnr, 'PSNR (dB)', 'PSNR Comparison'),
        (c_ssim, f_ssim, 'SSIM',      'SSIM Comparison'),
    ]):
        b1 = ax.bar(x - w/2, c_vals, w, label='COLMAP',  color='steelblue', alpha=0.85)
        b2 = ax.bar(x + w/2, f_vals, w, label='DUSt3R',  color='tomato',    alpha=0.85)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Part 1: COLMAP vs DUSt3R Initialization Comparison', fontsize=13)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_render', action='store_true',
                        help='Skip GPU rendering (only generate metric figures)')
    args = parser.parse_args()

    print("="*60)
    print("Generating Part 1 & Part 3 Visualizations")
    print("="*60)

    # ── Part 1: per-dataset detail figures ───────────────────
    print("\n[Part 1] Dataset detail figures...")
    for ds in ['re10k', 'dl3dv', 'waymo']:
        make_part1_figure(ds, f'outputs/part1/{ds}',
                          f'outputs/part1/vis_{ds}_comparison.png')

    print("\n[Part 1] Multi-dataset summary...")
    make_part1_summary(['re10k', 'dl3dv', 'waymo'],
                       'outputs/part1/summary_comparison.png')

    # ── Part 3: per-dataset render comparison ────────────────
    if not args.skip_render:
        print("\n[Part 3] Render comparison figures (requires GPU)...")
        for ds in ['waymo', 'dl3dv', 're10k']:
            p2_dir = f'outputs/part2/{ds}'
            p3_dir = f'outputs/part3/{ds}'
            if Path(p2_dir).exists() and Path(p3_dir).exists():
                make_part3_figure(ds, p2_dir, p3_dir,
                                  f'outputs/part3/{ds}/sparse_vs_enhanced.png',
                                  n_views=4)
            else:
                print(f"  [skip] {ds}: missing part2 or part3 output")

    print("\n[Part 3] Multi-dataset summary...")
    make_part3_summary(['waymo', 'dl3dv', 're10k', 'flowers', 'treehill'],
                       'outputs/part3/summary_comparison.png')

    print("\nAll visualizations generated.")
