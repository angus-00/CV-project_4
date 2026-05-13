"""Part 1: Comparison experiment between COLMAP and Foundation Model"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import time
from tqdm import tqdm

from common.dataset import SceneDataset
from common.metrics import compute_psnr, compute_ssim, compute_lpips
from common.camera import Camera
from part1.colmap_runner import COLMAPRunner
from part1.foundation_model import FoundationModelRunner
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.renderer import GaussianRenderer
from gaussian_splatting.trainer import GaussianTrainer


class ComparisonExperiment:
    """Compare COLMAP vs Foundation Model for pose initialization"""

    def __init__(self, config: Dict):
        self.config = config
        self.colmap_runner = COLMAPRunner(
            colmap_path=config.get('colmap_path', 'E:/Colmap/COLMAP.bat')
        )
        self.foundation_runner = FoundationModelRunner(
            model_name=config.get('foundation_model', 'dust3r'),
            device=config.get('device', 'cuda')
        )

    def run_comparison(self, dataset_path: str, output_path: str, max_images: int = None) -> Dict:
        """Run complete comparison experiment

        Args:
            dataset_path: Path to dataset
            output_path: Path to save results
            max_images: Maximum number of images to use (None = use all)

        Returns:
            Dictionary with comparison results
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Part 1: Camera Pose Initialization Comparison")
        print(f"Dataset: {dataset_path}")
        print(f"{'='*60}\n")

        # Load dataset
        dataset = SceneDataset(dataset_path)
        total_images = len(dataset)

        # Subsample if needed
        if max_images and total_images > max_images:
            print(f"Subsampling {max_images} images from {total_images} total images")
            # Evenly spaced sampling
            indices = np.linspace(0, total_images - 1, max_images, dtype=int)
            dataset.images = [dataset.images[i] for i in indices]
            dataset.cameras = [dataset.cameras[i] for i in indices]
            print(f"Using {len(dataset)} images (every ~{total_images // max_images} frames)")
        else:
            print(f"Loaded {len(dataset)} images")

        # Method 1: COLMAP
        print("\n[1/2] Running COLMAP SfM...")
        colmap_results = self._run_colmap(dataset, output_path / 'colmap')

        # Method 2: Foundation Model
        print("\n[2/2] Running Foundation Model...")
        foundation_results = self._run_foundation_model(
            dataset, output_path / 'foundation'
        )

        # Compare results
        print("\n" + "="*60)
        print("Comparison Results")
        print("="*60)

        comparison = {
            'colmap': colmap_results,
            'foundation': foundation_results,
            'summary': self._generate_summary(colmap_results, foundation_results)
        }

        # Save results
        self._save_results(comparison, output_path / 'comparison.json')

        return comparison

    def _run_colmap(self, dataset: SceneDataset, output_path: Path) -> Dict:
        """Run COLMAP and evaluate"""
        output_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Prepare images directory
        images_dir = output_path / 'images'
        images_dir.mkdir(exist_ok=True)

        # Copy/link images
        for i, img_path in enumerate(dataset.images):
            src = Path(img_path)
            dst = images_dir / src.name
            if not dst.exists():
                import shutil
                shutil.copy(src, dst)

        # Run COLMAP SfM
        try:
            poses = self.colmap_runner.run_sfm(
                image_path=str(images_dir),
                output_path=str(output_path)
            )
        except Exception as e:
            print(f"COLMAP failed: {e}")
            return {'success': False, 'error': str(e)}

        elapsed_time = time.time() - start_time

        # Evaluate (placeholder - actual 3DGS training needed)
        metrics = self._evaluate_poses(poses, dataset)

        return {
            'success': True,
            'time': elapsed_time,
            'num_cameras': len(poses['cameras']),
            'num_points': len(poses['points3d']) if isinstance(poses['points3d'], np.ndarray) else len(poses['points3d']),
            'metrics': metrics,
            'poses': poses,
            'training_history': metrics.get('training_history', {})
        }

    def _run_foundation_model(self, dataset: SceneDataset,
                             output_path: Path) -> Dict:
        """Run Foundation Model and evaluate"""
        output_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Run pose estimation with batching for large datasets
        batch_size = 50 if len(dataset.images) > 100 else None
        try:
            poses = self.foundation_runner.estimate_poses(
                image_paths=dataset.images,
                output_path=str(output_path),
                batch_size=batch_size
            )
        except Exception as e:
            print(f"Foundation Model failed: {e}")
            return {'success': False, 'error': str(e)}

        elapsed_time = time.time() - start_time

        # Evaluate
        metrics = self._evaluate_poses(poses, dataset)

        return {
            'success': True,
            'time': elapsed_time,
            'num_cameras': len(poses['cameras']),
            'num_points': len(poses['points3d']) if isinstance(poses['points3d'], np.ndarray) else len(poses['points3d']),
            'metrics': metrics,
            'poses': poses,
            'training_history': metrics.get('training_history', {})
        }

    def _evaluate_poses(self, poses: Dict, dataset: SceneDataset) -> Dict:
        """Evaluate pose estimation quality by training 3DGS"""
        try:
            # Convert poses to Camera objects
            cameras = self._poses_to_cameras(poses)

            # Split train/test
            n_train = int(len(cameras) * 0.8)
            train_cameras = cameras[:n_train]
            test_cameras = cameras[n_train:]

            # Load images - keep original resolution, don't resize
            # Each camera already has correct intrinsics for its resolution
            from PIL import Image as PILImage

            def load_img(p, cam):
                img = PILImage.open(p).convert('RGB')
                # Resize to match camera resolution (Foundation Model uses 256px internally)
                if img.width != cam.width or img.height != cam.height:
                    img = img.resize((cam.width, cam.height), PILImage.LANCZOS)
                return np.array(img)

            train_images = [load_img(dataset.images[i], train_cameras[i]) for i in range(n_train)]
            test_images  = [load_img(dataset.images[i], test_cameras[i - n_train]) for i in range(n_train, len(cameras))]

            # Initialize 3DGS model
            model = GaussianModel(sh_degree=3)
            points3d_raw = poses.get('points3d', [])
            if len(points3d_raw) > 0:
                if isinstance(points3d_raw, np.ndarray):
                    pts = points3d_raw
                    colors = np.ones((len(pts), 3)) * 0.5
                else:
                    pts = np.array([p['xyz'] for p in points3d_raw], dtype=np.float32)
                    colors = np.array([p['rgb'] for p in points3d_raw], dtype=np.float32) / 255.0

                # Limit point cloud size
                MAX_PTS = 50000
                if len(pts) > MAX_PTS:
                    idx = np.random.choice(len(pts), MAX_PTS, replace=False)
                    pts = pts[idx]
                    colors = colors[idx]
                model.create_from_pcd(pts, colors)
            else:
                model.create_from_pcd(np.random.randn(1000, 3) * 0.5,
                                      np.ones((1000, 3)) * 0.5)

            # Train
            renderer = GaussianRenderer(device=self.config.get('device', 'cuda'))
            trainer = GaussianTrainer(model, renderer, device=self.config.get('device', 'cuda'))

            results = trainer.train(
                cameras=train_cameras,
                images=train_images,
                iterations=self.config.get('train_iterations', 3000),
                test_cameras=test_cameras,
                test_images=test_images,
                log_interval=50,
                eval_interval=500
            )

            test_metrics = results.get('test_metrics', {})
            test_metrics['training_history'] = results.get('metrics', {})
            return test_metrics

        except Exception as e:
            import traceback
            print(f"Evaluation failed: {e}")
            traceback.print_exc()
            return {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'error': str(e)}

    def _poses_to_cameras(self, poses: Dict) -> List[Camera]:
        """Convert pose dict to Camera objects"""
        cameras = []
        for img_id, img_data in poses['images'].items():
            cam_id = img_data['camera_id']
            cam_data = poses['cameras'][cam_id]

            # Build Camera object
            qvec = np.array(img_data['qvec'])
            tvec = np.array(img_data['tvec'])

            # Convert quaternion to rotation matrix
            R = self._quat_to_rotation(qvec)

            # Build intrinsic matrix
            params = cam_data['params']
            K = np.array([
                [params[0], 0, params[2]],
                [0, params[1], params[3]],
                [0, 0, 1]
            ])

            camera = Camera(
                R=R, t=tvec, K=K,
                width=cam_data['width'],
                height=cam_data['height']
            )
            cameras.append(camera)

        return cameras

    def _quat_to_rotation(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = quat
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])

    def _generate_summary(self, colmap_results: Dict,
                         foundation_results: Dict) -> Dict:
        """Generate comparison summary with convergence analysis"""
        summary = {
            'colmap_success': colmap_results.get('success', False),
            'foundation_success': foundation_results.get('success', False),
        }

        if summary['colmap_success'] and summary['foundation_success']:
            # Time comparison
            summary['time_comparison'] = {
                'colmap': colmap_results['time'],
                'foundation': foundation_results['time'],
                'speedup': colmap_results['time'] / max(foundation_results['time'], 1e-6)
            }

            # Reconstruction comparison
            summary['reconstruction_comparison'] = {
                'colmap_cameras': colmap_results['num_cameras'],
                'foundation_cameras': foundation_results['num_cameras'],
                'colmap_points': colmap_results['num_points'],
                'foundation_points': foundation_results['num_points'],
            }

            # Rendering quality comparison
            cm = colmap_results.get('metrics', {})
            fm = foundation_results.get('metrics', {})
            summary['quality_comparison'] = {
                'colmap_psnr':  cm.get('psnr', 0.0),
                'colmap_ssim':  cm.get('ssim', 0.0),
                'colmap_lpips': cm.get('lpips', 0.0),
                'foundation_psnr':  fm.get('psnr', 0.0),
                'foundation_ssim':  fm.get('ssim', 0.0),
                'foundation_lpips': fm.get('lpips', 0.0),
                'psnr_delta':  fm.get('psnr', 0.0) - cm.get('psnr', 0.0),
                'ssim_delta':  fm.get('ssim', 0.0) - cm.get('ssim', 0.0),
                'lpips_delta': fm.get('lpips', 0.0) - cm.get('lpips', 0.0),
            }

            # Convergence analysis
            c_hist = colmap_results.get('training_history', {})
            f_hist = foundation_results.get('training_history', {})
            if c_hist.get('psnr') and f_hist.get('psnr'):
                c_psnr = c_hist['psnr']
                f_psnr = f_hist['psnr']
                # Iteration to reach 20dB PSNR
                def iters_to_threshold(psnr_list, iters_list, threshold=20.0):
                    for p, it in zip(psnr_list, iters_list):
                        if p >= threshold:
                            return it
                    return -1
                summary['convergence_analysis'] = {
                    'colmap_final_train_psnr': float(c_psnr[-1]) if c_psnr else 0.0,
                    'foundation_final_train_psnr': float(f_psnr[-1]) if f_psnr else 0.0,
                    'colmap_iters_to_20db': iters_to_threshold(c_psnr, c_hist.get('iteration', [])),
                    'foundation_iters_to_20db': iters_to_threshold(f_psnr, f_hist.get('iteration', [])),
                }

        return summary

    def _save_results(self, results: Dict, output_file: Path):
        """Save comparison results to JSON"""
        import numpy as np

        def convert_to_serializable(obj):
            """Convert numpy arrays and other non-serializable objects"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        serializable_results = convert_to_serializable(results)
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def run_comparison_experiment(config: Dict, dataset_path: str,
                             output_path: str, max_images: int = None) -> Dict:
    """Main entry point for Part 1 comparison experiment"""
    experiment = ComparisonExperiment(config)
    return experiment.run_comparison(dataset_path, output_path, max_images=max_images)
