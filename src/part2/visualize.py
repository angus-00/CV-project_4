"""Visualization utilities for Part 2 results

Generates high-quality visualizations:
- Camera trajectory plots (3D)
- Reconstruction quality comparisons
- Novel view synthesis results
- Side-by-side comparisons
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Dict, Optional
import json
from PIL import Image
import torch


def visualize_camera_trajectory(
    cameras_dict: Dict,
    output_path: Path,
    title: str = "Camera Trajectory"
):
    """Plot 3D camera trajectory.

    Args:
        cameras_dict: Dict with 'images' key containing pose data
        output_path: Path to save figure
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    positions = []
    for i in sorted(cameras_dict['images'].keys()):
        img_data = cameras_dict['images'][i]
        qvec = np.array(img_data['qvec'])
        tvec = np.array(img_data['tvec'])

        # world2cam -> cam position in world
        R = _quat_to_rot(qvec)
        cam_pos = -R.T @ tvec
        positions.append(cam_pos)

    positions = np.array(positions)

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='Camera Path')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=np.arange(len(positions)), cmap='viridis', s=50)

    # Mark start and end
    ax.scatter(*positions[0], c='green', s=200, marker='o', label='Start')
    ax.scatter(*positions[-1], c='red', s=200, marker='s', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved trajectory plot: {output_path}")


def visualize_point_cloud(
    points: np.ndarray,
    output_path: Path,
    title: str = "Reconstructed Point Cloud",
    max_points: int = 10000
):
    """Plot 3D point cloud.

    Args:
        points: (N, 3) point cloud
        output_path: Path to save figure
        title: Plot title
        max_points: Downsample to this many points for visualization
    """
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=1, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title} ({len(points)} points)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved point cloud plot: {output_path}")


def visualize_rendering_comparison(
    gt_images: List[np.ndarray],
    rendered_images: List[np.ndarray],
    output_path: Path,
    num_samples: int = 4
):
    """Create side-by-side comparison of GT vs rendered images.

    Args:
        gt_images: List of ground truth images (H, W, 3) in [0, 1]
        rendered_images: List of rendered images
        output_path: Path to save figure
        num_samples: Number of samples to show
    """
    n = min(num_samples, len(gt_images), len(rendered_images))
    indices = np.linspace(0, len(gt_images)-1, n, dtype=int)

    fig, axes = plt.subplots(n, 2, figsize=(10, 3*n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        gt = gt_images[idx]
        rendered = rendered_images[idx]

        # Normalize to [0, 1]
        if gt.max() > 1.0:
            gt = gt / 255.0
        rendered = np.clip(rendered, 0, 1)

        axes[i, 0].imshow(np.clip(gt, 0, 1))
        axes[i, 0].set_title(f"Ground Truth {idx}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(rendered)
        axes[i, 1].set_title(f"Rendered {idx}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved rendering comparison: {output_path}")


def plot_metrics_comparison(
    results_list: List[Dict],
    labels: List[str],
    output_path: Path
):
    """Plot bar chart comparing metrics across experiments.

    Args:
        results_list: List of result dicts with 'psnr', 'ssim', 'lpips'
        labels: Labels for each experiment
        output_path: Path to save figure
    """
    metrics = ['psnr', 'ssim', 'lpips']
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(metrics):
        values = [r.get(metric, 0) for r in results_list]
        axes[i].bar(x, values, width, label=metric.upper())
        axes[i].set_xlabel('Experiment')
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)

        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved metrics comparison: {output_path}")


def create_summary_report(
    results: Dict,
    output_path: Path
):
    """Create a text summary report.

    Args:
        results: Result dict from experiment
        output_path: Path to save report
    """
    report = []
    report.append("="*60)
    report.append("Part 2: Sparse View Reconstruction - Summary Report")
    report.append("="*60)
    report.append("")
    report.append(f"Dataset: {results.get('dataset', 'N/A')}")
    report.append(f"Sparsity: 1/{results.get('sparsity', 'N/A')} frames")
    report.append(f"Sparse Views: {results.get('num_sparse_views', 'N/A')}")
    report.append(f"Test Views: {results.get('num_test_views', 'N/A')}")
    report.append(f"Reconstructed Points: {results.get('num_points', 'N/A')}")
    report.append("")
    report.append("Pose Accuracy:")
    report.append(f"  ATE RMSE: {results.get('ate_rmse', 0):.4f}")
    report.append("")
    report.append("Rendering Quality:")
    report.append(f"  PSNR: {results.get('psnr', 0):.2f} dB")
    report.append(f"  SSIM: {results.get('ssim', 0):.4f}")
    report.append(f"  LPIPS: {results.get('lpips', 0):.4f}")
    report.append("")
    report.append("Training:")
    report.append(f"  Iterations: {results.get('training_iterations', 'N/A')}")
    report.append(f"  Best Train PSNR: {results.get('best_train_psnr', 0):.2f} dB")
    report.append("")
    report.append("="*60)

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"  Saved summary report: {output_path}")


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])
