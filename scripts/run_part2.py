"""Run Part 2: Sparse View Unposed Reconstruction

Usage:
    # Run on DL3DV with 1/30 sparsity (as per PDF)
    python scripts/run_part2.py --dataset dl3dv --data_path data/dl3dv-2

    # Run on all 3 mandatory datasets
    python scripts/run_part2.py --all_datasets

    # Quick test with fewer iterations
    python scripts/run_part2.py --dataset re10k --iterations 1000
"""

import argparse
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from part2.sparse_view import run_sparse_view_experiment
from part2.visualize import (
    visualize_camera_trajectory,
    visualize_point_cloud,
    plot_metrics_comparison,
    create_summary_report
)
from common.memory_monitor import start_memory_monitor, stop_memory_monitor


# Dataset configs: sparsity per PDF Section 6
DATASET_CONFIGS = {
    'dl3dv': {
        'data_path': 'data/dl3dv-2',
        'sparsity': 30,        # 1/30 frames
        'output': 'outputs/part2/dl3dv',
        'iterations': 3000,
    },
    're10k': {
        'data_path': 'data/re10k-1',
        'sparsity': 30,        # 1/30 frames
        'output': 'outputs/part2/re10k',
        'iterations': 3000,
    },
    'waymo': {
        'data_path': 'data/waymo-405841',
        'sparsity': 10,        # 1/10 frames
        'output': 'outputs/part2/waymo',
        'iterations': 3000,
    },
    'flowers': {
        'data_path': 'data/flowers',
        'sparsity': 20,        # 1/20 frames (extra dataset)
        'output': 'outputs/part2/flowers',
        'iterations': 3000,
    },
    'treehill': {
        'data_path': 'data/treehill',
        'sparsity': 20,        # 1/20 frames (extra dataset)
        'output': 'outputs/part2/treehill',
        'iterations': 3000,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Part 2: Sparse View Reconstruction')
    parser.add_argument('--dataset', choices=['dl3dv', 're10k', 'waymo', 'flowers', 'treehill'],
                        help='Dataset to run on')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override dataset path')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Override output path')
    parser.add_argument('--sparsity', type=int, default=None,
                        help='Sub-sampling rate (1/N). Overrides dataset default.')
    parser.add_argument('--iterations', type=int, default=None,
                        help='3DGS training iterations (default per dataset config)')
    parser.add_argument('--temporal_range', type=str, default='full', choices=['full', 'front_half'],
                        help='Temporal sampling range: full or front_half')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--all_datasets', action='store_true',
                        help='Run on all 3 mandatory datasets sequentially')
    parser.add_argument('--extra_datasets', action='store_true',
                        help='Run on extra datasets (flowers, treehill)')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Skip visualization generation')
    return parser.parse_args()


def run_single_dataset(cfg: dict, args) -> dict:
    """Run experiment on a single dataset."""
    data_path = args.data_path if args.data_path else cfg['data_path']
    output_path = args.output_path if args.output_path else cfg['output']
    sparsity = args.sparsity if args.sparsity else cfg['sparsity']
    iterations = args.iterations if args.iterations else cfg['iterations']
    temporal_range = args.temporal_range if hasattr(args, 'temporal_range') else 'full'

    t0 = time.time()
    results = run_sparse_view_experiment(
        data_path=data_path,
        output_path=output_path,
        sparsity=sparsity,
        iterations=iterations,
        device=args.device,
        temporal_range=temporal_range
    )
    results['total_time'] = time.time() - t0

    return results, output_path


def generate_visualizations(results: dict, output_path: str, recon_json: str):
    """Generate all visualizations for a single dataset result."""
    out = Path(output_path)
    vis_dir = out / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    # Load reconstruction data
    recon_path = out / 'reconstruction.json'
    if recon_path.exists():
        with open(recon_path) as f:
            recon_data = json.load(f)

        # Camera trajectory
        visualize_camera_trajectory(
            recon_data,
            vis_dir / 'camera_trajectory.png',
            title=f"Camera Trajectory ({Path(output_path).name})"
        )

        # Point cloud visualization
        if 'points3d' in recon_data and len(recon_data['points3d']) > 0:
            from part2.visualize import visualize_point_cloud
            points = np.array(recon_data['points3d'])
            visualize_point_cloud(
                points,
                vis_dir / 'point_cloud.png',
                title=f"Reconstructed Point Cloud ({Path(output_path).name})"
            )

    # Rendering comparison (load from npz file)
    vis_samples_path = out / 'visualization_samples.npz'
    if vis_samples_path.exists():
        from part2.visualize import visualize_rendering_comparison
        vis_data = np.load(vis_samples_path)
        visualize_rendering_comparison(
            vis_data['gt_samples'],
            vis_data['rendered_samples'],
            vis_dir / 'rendering_comparison.png',
            num_samples=4
        )

    # Summary report
    create_summary_report(results, vis_dir / 'summary.txt')
    print(f"  Visualizations saved to: {vis_dir}")


def main():
    args = parse_args()

    # Start memory monitor
    print("\n" + "="*70)
    print("Starting GPU Memory Monitor")
    print("="*70)
    start_memory_monitor(threshold_gb=15.5, check_interval=1.0)
    print()

    try:
        if args.all_datasets:
            # Run on all 3 mandatory datasets
            datasets = ['dl3dv', 're10k', 'waymo']
            print(f"\nRunning on all {len(datasets)} mandatory datasets: {datasets}")
            all_results = {}
            all_labels = []

            for dataset_name in datasets:
                print(f"\n{'='*70}")
                print(f"Dataset: {dataset_name}")
                print(f"{'='*70}")
                cfg = DATASET_CONFIGS[dataset_name]

                results, output_path = run_single_dataset(cfg, args)
                all_results[dataset_name] = results
                all_labels.append(dataset_name)

                # Generate visualizations
                if not args.no_visualize:
                    recon_json = str(Path(output_path) / 'reconstruction.json')
                    generate_visualizations(results, output_path, recon_json)

            # Save combined results
            combined_path = Path('outputs/part2/all_results.json')
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            with open(combined_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n{'='*70}")
            print(f"Combined results saved to: {combined_path}")
            print(f"{'='*70}")

        elif args.extra_datasets:
            # Run on extra datasets
            datasets = ['flowers', 'treehill']
            print(f"\nRunning on extra datasets: {datasets}")
            all_results = {}

            for dataset_name in datasets:
                print(f"\n{'='*70}")
                print(f"Dataset: {dataset_name}")
                print(f"{'='*70}")
                cfg = DATASET_CONFIGS[dataset_name]

                results, output_path = run_single_dataset(cfg, args)
                all_results[dataset_name] = results

                # Generate visualizations
                if not args.no_visualize:
                    recon_json = str(Path(output_path) / 'reconstruction.json')
                    generate_visualizations(results, output_path, recon_json)

            # Save combined results
            combined_path = Path('outputs/part2/extra_results.json')
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            with open(combined_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n{'='*70}")
            print(f"Combined results saved to: {combined_path}")
            print(f"{'='*70}")

        elif args.dataset:
            # Single dataset
            cfg = DATASET_CONFIGS[args.dataset]
            print(f"\nRunning Part 2 on: {args.dataset}")
            results, output_path = run_single_dataset(cfg, args)

            if not args.no_visualize:
                recon_json = str(Path(output_path) / 'reconstruction.json')
                generate_visualizations(results, output_path, recon_json)

            print(f"\nOutput: {output_path}")

        else:
            print("Please specify --dataset, --all_datasets, or --extra_datasets")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        stop_memory_monitor()
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        stop_memory_monitor()
        sys.exit(1)
    finally:
        stop_memory_monitor()


if __name__ == '__main__':
    main()
