"""Run Part 1: Camera Pose Initialization Comparison

Usage:
    python scripts/run_part1.py --dataset dl3dv --data_path data/dl3dv-2
    python scripts/run_part1.py --config configs/part1_default.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from part1.compare import run_comparison_experiment
from part1.visualize import generate_all_visualizations
from common.memory_monitor import start_memory_monitor, stop_memory_monitor


def parse_args():
    parser = argparse.ArgumentParser(description='Part 1: Pose Initialization Comparison')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['dl3dv', 're10k', 'waymo'],
                       help='Dataset type (overrides config)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset (overrides config)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--colmap_path', type=str, default=None,
                       help='Path to COLMAP executable (overrides config)')
    parser.add_argument('--foundation_model', type=str, default=None,
                       choices=['dust3r', 'mast3r'],
                       help='Foundation model to use (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (overrides config)')
    parser.add_argument('--max_images', type=int, default=50,
                       help='Maximum number of images to use (default: 50 for memory constraints)')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip visualization generation')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Load config from file or use defaults
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            'dataset': {'type': 'dl3dv', 'path': 'data/dl3dv-2'},
            'colmap': {'path': 'E:/Colmap/COLMAP.bat', 'camera_model': 'PINHOLE', 'use_gpu': True},
            'foundation_model': {'name': 'dust3r', 'device': 'cuda'},
            'training': {'iterations': 3000, 'train_test_split': 0.8},
            'output': {'base_path': 'outputs/part1', 'save_visualizations': True},
            'device': 'cuda'
        }

    # Override with command line arguments
    if args.dataset:
        config['dataset']['type'] = args.dataset
    if args.data_path:
        config['dataset']['path'] = args.data_path
    if args.colmap_path:
        config['colmap']['path'] = args.colmap_path
    if args.foundation_model:
        config['foundation_model']['name'] = args.foundation_model
    if args.device:
        config['device'] = args.device
        config['foundation_model']['device'] = args.device

    # Set output path
    dataset_type = config['dataset']['type']
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = f"{config['output']['base_path']}/{dataset_type}"

    # Build experiment config
    experiment_config = {
        'colmap_path': config['colmap']['path'],
        'foundation_model': config['foundation_model']['name'],
        'device': config['device'],
        'train_iterations': config['training']['iterations']
    }

    print("\n" + "="*70)
    print("Part 1: Camera Pose Initialization Comparison")
    print("="*70)
    print(f"Dataset: {dataset_type}")
    print(f"Data path: {config['dataset']['path']}")
    print(f"Output path: {output_path}")
    print(f"COLMAP path: {config['colmap']['path']}")
    print(f"Foundation model: {config['foundation_model']['name']}")
    print(f"Device: {config['device']}")
    print(f"Training iterations: {config['training']['iterations']}")
    print("="*70 + "\n")

    # Start memory monitor
    print("Starting memory monitor (threshold: 15.5GB)...")
    start_memory_monitor(threshold_gb=15.5, check_interval=1.0)
    print()

    # Run comparison experiment
    try:
        results = run_comparison_experiment(
            config=experiment_config,
            dataset_path=config['dataset']['path'],
            output_path=output_path,
            max_images=args.max_images
        )

        # Print summary
        print("\n" + "="*70)
        print("Experiment Summary")
        print("="*70)

        summary = results.get('summary', {})

        print(f"\nCOLMAP:")
        if summary.get('colmap_success'):
            colmap = results['colmap']
            print(f"  [OK] Success")
            print(f"  Time:     {colmap['time']:.2f}s")
            print(f"  Cameras:  {colmap['num_cameras']}")
            print(f"  Points:   {colmap['num_points']:,}")
            m = colmap.get('metrics', {})
            print(f"  PSNR:     {m.get('psnr', 0):.2f} dB")
            print(f"  SSIM:     {m.get('ssim', 0):.4f}")
            print(f"  LPIPS:    {m.get('lpips', 0):.4f}")
        else:
            print(f"  [FAILED] {results['colmap'].get('error', 'Unknown error')}")

        print(f"\nFoundation Model (DUSt3R):")
        if summary.get('foundation_success'):
            foundation = results['foundation']
            print(f"  [OK] Success")
            print(f"  Time:     {foundation['time']:.2f}s")
            print(f"  Cameras:  {foundation['num_cameras']}")
            print(f"  Points:   {foundation['num_points']:,}")
            m = foundation.get('metrics', {})
            print(f"  PSNR:     {m.get('psnr', 0):.2f} dB")
            print(f"  SSIM:     {m.get('ssim', 0):.4f}")
            print(f"  LPIPS:    {m.get('lpips', 0):.4f}")
        else:
            print(f"  [FAILED] {results['foundation'].get('error', 'Unknown error')}")

        if 'time_comparison' in summary:
            tc = summary['time_comparison']
            print(f"\nTime Comparison:")
            print(f"  COLMAP:     {tc['colmap']:.2f}s")
            print(f"  Foundation: {tc['foundation']:.2f}s")
            print(f"  Speedup:    {tc['speedup']:.2f}x")

        if 'quality_comparison' in summary:
            qc = summary['quality_comparison']
            print(f"\nQuality Comparison (COLMAP vs Foundation):")
            print(f"  {'Metric':<10} {'COLMAP':>10} {'Foundation':>12} {'Delta':>8}")
            print(f"  {'-'*42}")
            print(f"  {'PSNR':<10} {qc.get('colmap_psnr',0):>10.2f} {qc.get('foundation_psnr',0):>12.2f} {qc.get('psnr_delta',0):>+8.2f}")
            print(f"  {'SSIM':<10} {qc.get('colmap_ssim',0):>10.4f} {qc.get('foundation_ssim',0):>12.4f} {qc.get('ssim_delta',0):>+8.4f}")
            print(f"  {'LPIPS':<10} {qc.get('colmap_lpips',0):>10.4f} {qc.get('foundation_lpips',0):>12.4f} {qc.get('lpips_delta',0):>+8.4f}")

        if 'convergence_analysis' in summary:
            ca = summary['convergence_analysis']
            print(f"\nConvergence Analysis:")
            print(f"  COLMAP final train PSNR:      {ca.get('colmap_final_train_psnr',0):.2f} dB")
            print(f"  Foundation final train PSNR:   {ca.get('foundation_final_train_psnr',0):.2f} dB")

        # Generate visualizations
        if not args.no_visualize and config['output'].get('save_visualizations', True):
            if summary.get('colmap_success') and summary.get('foundation_success'):
                print("\n" + "="*70)
                print("Generating Visualizations")
                print("="*70)
                generate_all_visualizations(
                    colmap_results=results['colmap'],
                    foundation_results=results['foundation'],
                    output_dir=Path(output_path) / 'visualizations',
                    summary=summary
                )

        print("\n" + "="*70)
        print(f"Results saved to: {output_path}/comparison.json")
        if not args.no_visualize:
            print(f"Visualizations saved to: {output_path}/visualizations/")
        print("="*70 + "\n")

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
