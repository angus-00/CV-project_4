"""Dataset loader for different scene formats"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

from .camera import CameraParameters


class SceneDataset(Dataset):
    """Generic scene dataset loader"""

    def __init__(self, data_path: str, dataset_type: str = 'auto'):
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type

        if dataset_type == 'auto':
            self.dataset_type = self._detect_dataset_type()

        self.images = []
        self.cameras = []
        self._load_data()

    def _detect_dataset_type(self) -> str:
        """Auto-detect dataset type"""
        if (self.data_path / 'cameras.json').exists():
            if (self.data_path / 'rgb').exists():
                return 'dl3dv'
            elif (self.data_path / 'images').exists():
                return 're10k'
        elif (self.data_path / 'FRONT').exists():
            return 'waymo'
        elif (self.data_path / 'poses_bounds.npy').exists():
            return 'llff'
        raise ValueError(f"Unknown dataset type at {self.data_path}")

    def _load_data(self):
        """Load images and camera parameters"""
        if self.dataset_type == 'dl3dv':
            self._load_dl3dv()
        elif self.dataset_type == 're10k':
            self._load_re10k()
        elif self.dataset_type == 'waymo':
            self._load_waymo()
        elif self.dataset_type == 'llff':
            self._load_llff()

    def _load_dl3dv(self):
        """Load DL3DV dataset"""
        with open(self.data_path / 'cameras.json', 'r') as f:
            camera_data = json.load(f)

        with open(self.data_path / 'intrinsics.json', 'r') as f:
            intrinsics = json.load(f)

        for cam in camera_data:
            img_path = self.data_path / 'rgb' / cam['image_name']
            if img_path.exists():
                self.images.append(str(img_path))
                self.cameras.append(self._parse_dl3dv_camera(cam, intrinsics))

    def _parse_dl3dv_camera(self, cam: Dict, intrinsics: Dict) -> CameraParameters:
        """Parse DL3DV camera parameters"""
        quat = np.array(cam['cam_quat'])
        rotation = self._quat_to_rotation(quat)
        translation = np.array(cam['cam_trans'])

        img = Image.open(self.data_path / 'rgb' / cam['image_name'])
        width, height = img.size

        return CameraParameters(
            fx=cam['fx'] * width,
            fy=cam['fy'] * height,
            cx=cam['cx'] * width,
            cy=cam['cy'] * height,
            width=width,
            height=height,
            rotation=rotation,
            translation=translation
        )

    def _load_re10k(self):
        """Load RE10K dataset"""
        with open(self.data_path / 'cameras.json', 'r') as f:
            camera_data = json.load(f)

        for cam in camera_data:
            img_path = self.data_path / 'images' / cam['image_name']
            if img_path.exists():
                self.images.append(str(img_path))
                self.cameras.append(self._parse_re10k_camera(cam))

    def _parse_re10k_camera(self, cam: Dict) -> CameraParameters:
        """Parse RE10K camera parameters"""
        quat = np.array(cam['cam_quat'])
        rotation = self._quat_to_rotation(quat)
        translation = np.array(cam['cam_trans'])

        img = Image.open(self.data_path / 'images' / cam['image_name'])
        width, height = img.size

        return CameraParameters(
            fx=cam['fx'] * width,
            fy=cam['fy'] * height,
            cx=cam['cx'] * width,
            cy=cam['cy'] * height,
            width=width,
            height=height,
            rotation=rotation,
            translation=translation
        )

    def _load_waymo(self):
        """Load Waymo FRONT camera dataset.

        Structure:
            FRONT/
                rgb/000000.png ...
                calib/000000.txt  (Tr_velo_to_cam, fx, fy, cx, cy, k1, k2, ...)
                gt/000000.txt     (4x4 camera-to-world matrix, row-major, 4 rows)
        """
        front_dir = self.data_path / 'FRONT'
        rgb_dir = front_dir / 'rgb'
        calib_dir = front_dir / 'calib'
        gt_dir = front_dir / 'gt'

        frame_files = sorted(rgb_dir.glob('*.png'))

        for img_file in frame_files:
            stem = img_file.stem
            calib_file = calib_dir / f'{stem}.txt'
            gt_file = gt_dir / f'{stem}.txt'

            if not calib_file.exists() or not gt_file.exists():
                continue

            # Parse calibration
            calib = self._parse_waymo_calib(calib_file)
            # Parse ground truth pose (camera-to-world 4x4)
            c2w = self._parse_waymo_gt(gt_file)

            img = Image.open(img_file)
            width, height = img.size

            # c2w -> w2c extrinsics (rotation R, translation t in world2cam convention)
            w2c = np.linalg.inv(c2w)
            rotation = w2c[:3, :3]
            translation = w2c[:3, 3]

            self.images.append(str(img_file))
            self.cameras.append(CameraParameters(
                fx=calib['fx'],
                fy=calib['fy'],
                cx=calib['cx'],
                cy=calib['cy'],
                width=width,
                height=height,
                rotation=rotation,
                translation=translation
            ))

    def _parse_waymo_calib(self, path: Path) -> dict:
        """Parse Waymo calibration file."""
        with open(path, 'r') as f:
            text = f.read()

        import re
        calib = {}
        for key in ['fx', 'fy', 'cx', 'cy']:
            match = re.search(rf'{key}:\s+([\d.e+-]+)', text)
            if match:
                calib[key] = float(match.group(1))
        return calib

    def _parse_waymo_gt(self, path: Path) -> np.ndarray:
        """Parse Waymo ground truth pose (4x4 c2w matrix, 4 space-separated rows)."""
        rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    vals = [float(v) for v in line.split()]
                    if len(vals) == 4:
                        rows.append(vals)
        if len(rows) == 4:
            return np.array(rows)
        return np.eye(4)

    def _load_llff(self):
        """Load LLFF/Mip-NeRF 360 dataset.

        Structure:
            images/*.JPG or *.png
            poses_bounds.npy: (N, 17) array
                - [:15] -> 3x5 pose matrix [R|t|hwf]
                - [15:17] -> [near, far] bounds
        """
        poses_bounds = np.load(self.data_path / 'poses_bounds.npy')
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

        # Get image dimensions and focal from first pose
        H, W, focal = poses[0, :, 4]
        H, W = int(H), int(W)

        # Find images directory
        img_dir = self.data_path / 'images'
        if not img_dir.exists():
            raise ValueError(f"No images directory found in {self.data_path}")

        # Get sorted image files
        img_files = sorted(img_dir.glob('*'))
        img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        if len(img_files) != len(poses):
            raise ValueError(f"Mismatch: {len(img_files)} images but {len(poses)} poses")

        for i, img_file in enumerate(img_files):
            # Parse pose: LLFF uses [down, right, back] convention
            # Convert to OpenCV [right, down, forward]
            c2w_llff = poses[i, :, :4]  # 3x4 camera-to-world

            # LLFF to OpenCV coordinate transform
            # LLFF: x=right, y=down, z=back
            # OpenCV: x=right, y=down, z=forward
            # Transform: flip z axis
            transform = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ])

            R_llff = c2w_llff[:, :3]
            t_llff = c2w_llff[:, 3]

            # Apply transform
            R_opencv = R_llff @ transform
            t_opencv = t_llff

            # Build c2w in OpenCV convention
            c2w = np.eye(4)
            c2w[:3, :3] = R_opencv
            c2w[:3, 3] = t_opencv

            # Convert to w2c for CameraParameters
            w2c = np.linalg.inv(c2w)
            rotation = w2c[:3, :3]
            translation = w2c[:3, 3]

            # Get actual image size (LLFF images may be downsampled)
            img = Image.open(img_file)
            actual_w, actual_h = img.size

            # Scale focal length if image was resized
            scale_w = actual_w / W
            scale_h = actual_h / H
            fx = focal * scale_w
            fy = focal * scale_h
            cx = actual_w / 2.0
            cy = actual_h / 2.0

            self.images.append(str(img_file))
            self.cameras.append(CameraParameters(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=actual_w,
                height=actual_h,
                rotation=rotation,
                translation=translation
            ))

    def _quat_to_rotation(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = quat
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, CameraParameters]:
        """Get image and camera parameters"""
        img = Image.open(self.images[idx]).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor, self.cameras[idx]
