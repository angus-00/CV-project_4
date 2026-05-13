"""
Visualization utilities for Part 1
Camera trajectories, point clouds, training curves, and convergence analysis
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Optional
import json

from common.camera import Camera


def visualize_training_curves(
    colmap_history: Dict,
    foundation_history: Dict,
    output_path: str
):
    """Visualize training loss and PSNR curves for convergence comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    ax = axes[0, 0]
    if colmap_history.get('loss') and colmap_history.get('iteration'):
        ax.plot(colmap_history['iteration'], colmap_history['loss'],
                'b-', label='COLMAP', linewidth=2)
    if foundation_history.get('loss') and foundation_history.get('iteration'):
        ax.plot(foundation_history['iteration'], foundation_history['loss'],
                'r-', label='DUSt3R', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PSNR curves
    ax = axes[0, 1]
    if colmap_history.get('psnr') and colmap_history.get('iteration'):
        ax.plot(colmap_history['iteration'], colmap_history['psnr'],
                'b-', label='COLMAP', linewidth=2)
    if foundation_history.get('psnr') and foundation_history.get('iteration'):
        ax.plot(foundation_history['iteration'], foundation_history['psnr'],
                'r-', label='DUSt3R', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Training PSNR Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Number of Gaussians
    ax = axes[1, 0]
    if colmap_history.get('num_gaussians') and colmap_history.get('iteration'):
        ax.plot(colmap_history['iteration'], colmap_history['num_gaussians'],
                'b-', label='COLMAP', linewidth=2)
    if foundation_history.get('num_gaussians') and foundation_history.get('iteration'):
        ax.plot(foundation_history['iteration'], foundation_history['num_gaussians'],
                'r-', label='DUSt3R', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Gaussians')
    ax.set_title('Gaussian Count Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convergence speed (PSNR improvement rate)
    ax = axes[1, 1]
    if colmap_history.get('psnr') and len(colmap_history['psnr']) > 1:
        c_psnr = np.array(colmap_history['psnr'])
        c_iter = np.array(colmap_history['iteration'])
        c_rate = np.gradient(c_psnr, c_iter)
        ax.plot(c_iter, c_rate, 'b-', label='COLMAP', linewidth=2, alpha=0.7)
    if foundation_history.get('psnr') and len(foundation_history['psnr']) > 1:
        f_psnr = np.array(foundation_history['psnr'])
        f_iter = np.array(foundation_history['iteration'])
        f_rate = np.gradient(f_psnr, f_iter)
        ax.plot(f_iter, f_rate, 'r-', label='DUSt3R', linewidth=2, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR Improvement Rate (dB/iter)')
    ax.set_title('Convergence Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {output_path}")


def visualize_camera_trajectories(
    colmap_poses: Dict,
    foundation_poses: Dict,
    output_path: str,
    title: str = "Camera Trajectory Comparison"
):
    """Visualize camera trajectories from both methods"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract camera positions
    colmap_positions = _extract_camera_positions(colmap_poses)
    foundation_positions = _extract_camera_positions(foundation_poses)

    # Align foundation model scale to COLMAP using Procrustes
    if len(colmap_positions) > 0 and len(foundation_positions) > 0:
        # Compute scale factor (ratio of standard deviations)
        colmap_scale = np.std(colmap_positions)
        foundation_scale = np.std(foundation_positions)
        scale_factor = colmap_scale / foundation_scale if foundation_scale > 1e-6 else 1.0

        # Center and scale foundation positions
        foundation_center = foundation_positions.mean(axis=0)
        colmap_center = colmap_positions.mean(axis=0)
        foundation_positions_aligned = (foundation_positions - foundation_center) * scale_factor + colmap_center
    else:
        foundation_positions_aligned = foundation_positions

    # Plot trajectories
    if len(colmap_positions) > 0:
        ax.plot(colmap_positions[:, 0], colmap_positions[:, 1],
                colmap_positions[:, 2], 'b-o', label='COLMAP', linewidth=2)

    if len(foundation_positions_aligned) > 0:
        ax.plot(foundation_positions_aligned[:, 0], foundation_positions_aligned[:, 1],
                foundation_positions_aligned[:, 2], 'r-s', label='DUSt3R (aligned)', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Camera trajectory saved to: {output_path}")


def visualize_point_clouds(
    colmap_poses: Dict,
    foundation_poses: Dict,
    output_path: str,
    max_points: int = 10000
):
    """Visualize point clouds from both methods"""
    fig = plt.figure(figsize=(15, 6))

    # COLMAP point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    colmap_points = colmap_poses.get('points3d', [])
    if len(colmap_points) > 0:
        # Extract xyz coordinates from dict format
        if isinstance(colmap_points[0], dict):
            points = np.array([p['xyz'] for p in colmap_points])
        else:
            points = np.array(colmap_points)

        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=points[:, 2], cmap='viridis', s=1)
        ax1.set_title(f'COLMAP ({len(colmap_points)} points)')
    else:
        ax1.set_title('COLMAP (no points)')

    # DUSt3R point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    foundation_points = foundation_poses.get('points3d', [])
    if len(foundation_points) > 0:
        # Extract xyz coordinates from dict format
        if isinstance(foundation_points[0], dict):
            points = np.array([p['xyz'] for p in foundation_points])
        else:
            points = np.array(foundation_points)

        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=points[:, 2], cmap='viridis', s=1)
        ax2.set_title(f'DUSt3R ({len(foundation_points)} points)')
    else:
        ax2.set_title('DUSt3R (no points)')

    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Point clouds saved to: {output_path}")


def visualize_metrics_comparison(
    colmap_metrics: Dict,
    foundation_metrics: Dict,
    output_path: str
):
    """Visualize metrics comparison"""
    metrics = ['psnr', 'ssim', 'lpips']
    colmap_values = [colmap_metrics.get(m, 0) for m in metrics]
    foundation_values = [foundation_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, colmap_values, width, label='COLMAP', color='blue', alpha=0.7)
    ax.bar(x + width/2, foundation_values, width, label='DUSt3R',
           color='red', alpha=0.7)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Pose Estimation Quality Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison saved to: {output_path}")


def _extract_camera_positions(poses: Dict) -> np.ndarray:
    """Extract camera positions from pose dictionary"""
    positions = []
    for img_id, img_data in poses.get('images', {}).items():
        tvec = np.array(img_data['tvec'])
        positions.append(tvec)
    return np.array(positions) if positions else np.array([])


def visualize_initialization_quality(
    colmap_results: Dict,
    foundation_results: Dict,
    output_path: str
):
    """Visualize initialization quality comparison (sparse vs dense, point distribution)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Point cloud density
    ax = axes[0]
    labels = ['COLMAP', 'DUSt3R']
    points = [
        colmap_results.get('num_points', 0),
        foundation_results.get('num_points', 0)
    ]
    colors = ['blue', 'red']
    bars = ax.bar(labels, points, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Points')
    ax.set_title('Initial Point Cloud Density\n(Sparse vs Dense)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, points):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom')

    # Camera count
    ax = axes[1]
    cameras = [
        colmap_results.get('num_cameras', 0),
        foundation_results.get('num_cameras', 0)
    ]
    bars = ax.bar(labels, cameras, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Cameras')
    ax.set_title('Reconstructed Cameras')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, cameras):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom')

    # Reconstruction time
    ax = axes[2]
    times = [
        colmap_results.get('time', 0),
        foundation_results.get('time', 0)
    ]
    bars = ax.bar(labels, times, color=colors, alpha=0.7)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Reconstruction Time')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Initialization quality comparison saved to: {output_path}")


def create_analysis_report(
    summary: Dict,
    output_path: str
):
    """Create a detailed text analysis report"""
    report = []
    report.append("="*80)
    report.append("Part 1: Camera Pose Initialization Analysis Report")
    report.append("="*80)
    report.append("")

    # Success status
    report.append("1. RECONSTRUCTION SUCCESS")
    report.append("-" * 80)
    report.append(f"   COLMAP:          {'✓ Success' if summary.get('colmap_success') else '✗ Failed'}")
    report.append(f"   DUSt3R:          {'✓ Success' if summary.get('foundation_success') else '✗ Failed'}")
    report.append("")

    if summary.get('colmap_success') and summary.get('foundation_success'):
        # Time comparison
        tc = summary.get('time_comparison', {})
        report.append("2. TIME COMPARISON")
        report.append("-" * 80)
        report.append(f"   COLMAP:          {tc.get('colmap', 0):.2f}s")
        report.append(f"   DUSt3R:          {tc.get('foundation', 0):.2f}s")
        report.append(f"   Speedup:         {tc.get('speedup', 0):.2f}x")
        report.append("")

        # Reconstruction comparison
        rc = summary.get('reconstruction_comparison', {})
        report.append("3. INITIALIZATION QUALITY (Sparse vs Dense)")
        report.append("-" * 80)
        report.append(f"   COLMAP Cameras:          {rc.get('colmap_cameras', 0)}")
        report.append(f"   DUSt3R Cameras:       {rc.get('foundation_cameras', 0)}")
        report.append(f"   COLMAP Points:           {rc.get('colmap_points', 0):,} (sparse)")
        report.append(f"   DUSt3R Points:        {rc.get('foundation_points', 0):,} (dense)")
        report.append("")
        report.append("   Analysis: COLMAP produces sparse but accurate point clouds from")
        report.append("   feature matching, while DUSt3R generates dense predictions")
        report.append("   directly from image pairs.")
        report.append("")

        # Quality comparison
        qc = summary.get('quality_comparison', {})
        report.append("4. RENDERING QUALITY (Final 3DGS Results)")
        report.append("-" * 80)
        report.append(f"   Metric          COLMAP      DUSt3R        Delta")
        report.append(f"   PSNR (dB)       {qc.get('colmap_psnr', 0):6.2f}      {qc.get('foundation_psnr', 0):6.2f}        {qc.get('psnr_delta', 0):+.2f}")
        report.append(f"   SSIM            {qc.get('colmap_ssim', 0):6.4f}      {qc.get('foundation_ssim', 0):6.4f}      {qc.get('ssim_delta', 0):+.4f}")
        report.append(f"   LPIPS           {qc.get('colmap_lpips', 0):6.4f}      {qc.get('foundation_lpips', 0):6.4f}      {qc.get('lpips_delta', 0):+.4f}")
        report.append("")

        # Convergence analysis
        ca = summary.get('convergence_analysis', {})
        if ca:
            report.append("5. CONVERGENCE ANALYSIS")
            report.append("-" * 80)
            report.append(f"   COLMAP Final Train PSNR:      {ca.get('colmap_final_train_psnr', 0):.2f} dB")
            report.append(f"   DUSt3R Final Train PSNR:   {ca.get('foundation_final_train_psnr', 0):.2f} dB")
            c_iters = ca.get('colmap_iters_to_20db', -1)
            f_iters = ca.get('foundation_iters_to_20db', -1)
            report.append(f"   Iterations to 20dB (COLMAP):   {c_iters if c_iters > 0 else 'N/A'}")
            report.append(f"   Iterations to 20dB (DUSt3R): {f_iters if f_iters > 0 else 'N/A'}")
            report.append("")
            report.append("   Analysis: Convergence speed indicates how quickly each initialization")
            report.append("   method allows 3DGS to reach acceptable quality. Faster convergence")
            report.append("   suggests better initialization.")
            report.append("")

        # Key findings
        report.append("6. KEY FINDINGS")
        report.append("-" * 80)
        report.append("   • Initialization Impact:")
        if qc.get('psnr_delta', 0) > 0:
            report.append("     DUSt3R initialization leads to HIGHER final quality")
        elif qc.get('psnr_delta', 0) < -0.5:
            report.append("     COLMAP initialization leads to HIGHER final quality")
        else:
            report.append("     Both methods achieve SIMILAR final quality")

        if tc.get('speedup', 0) > 1.5:
            report.append(f"     DUSt3R is {tc.get('speedup', 0):.1f}x FASTER than COLMAP")
        elif tc.get('speedup', 0) < 0.7:
            report.append(f"     COLMAP is {1/tc.get('speedup', 1):.1f}x FASTER than DUSt3R")

        report.append("")
        report.append("   • Sparse vs Dense:")
        report.append(f"     COLMAP: {rc.get('colmap_points', 0):,} points (sparse, feature-based)")
        report.append(f"     DUSt3R: {rc.get('foundation_points', 0):,} points (dense, learning-based)")
        report.append("")

    report.append("="*80)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"Analysis report saved to: {output_path}")


def generate_all_visualizations(
    colmap_results: Dict,
    foundation_results: Dict,
    output_dir: Path,
    summary: Dict = None
):
    """Generate all visualizations for comparison"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualizations...")

    # Camera trajectories
    if colmap_results.get('success') and foundation_results.get('success'):
        visualize_camera_trajectories(
            colmap_results.get('poses', {}),
            foundation_results.get('poses', {}),
            str(output_dir / 'camera_trajectories.png')
        )

        visualize_point_clouds(
            colmap_results.get('poses', {}),
            foundation_results.get('poses', {}),
            str(output_dir / 'point_clouds.png')
        )

        visualize_metrics_comparison(
            colmap_results.get('metrics', {}),
            foundation_results.get('metrics', {}),
            str(output_dir / 'metrics_comparison.png')
        )

        # NEW: Training curves
        c_hist = colmap_results.get('training_history', {})
        f_hist = foundation_results.get('training_history', {})
        if c_hist and f_hist:
            visualize_training_curves(
                c_hist, f_hist,
                str(output_dir / 'training_curves.png')
            )

        # NEW: Initialization quality
        visualize_initialization_quality(
            colmap_results, foundation_results,
            str(output_dir / 'initialization_quality.png')
        )

        # NEW: Analysis report
        if summary:
            create_analysis_report(
                summary,
                str(output_dir / 'analysis_report.txt')
            )

    print("Visualizations complete!")
