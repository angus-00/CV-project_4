"""COLMAP initialization wrapper"""

import subprocess
import os
import struct
from pathlib import Path
from typing import Optional, Dict
import json


class COLMAPRunner:
    """Wrapper for COLMAP SfM pipeline"""

    def __init__(self, colmap_path: str = "E:/Colmap/COLMAP.bat"):
        self.colmap_path = colmap_path

    def run_sfm(self,
                image_path: str,
                output_path: str,
                camera_model: str = 'PINHOLE',
                use_gpu: bool = True) -> Dict:
        """Run complete COLMAP SfM pipeline"""

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        database_path = output_path / 'database.db'
        sparse_path = output_path / 'sparse'
        sparse_path.mkdir(exist_ok=True)

        # Feature extraction
        self._run_feature_extractor(image_path, database_path, camera_model, use_gpu)

        # Feature matching
        self._run_matcher(database_path, use_gpu)

        # Sparse reconstruction
        self._run_mapper(image_path, database_path, sparse_path)

        return self._parse_results(sparse_path / '0')

    def _run_feature_extractor(self, image_path, database_path, camera_model, use_gpu):
        """Extract features from images"""
        cmd = [
            self.colmap_path, 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(image_path),
            '--ImageReader.camera_model', camera_model,
        ]
        # Note: --SiftExtraction.use_gpu is not supported in all COLMAP versions
        # Removed to ensure compatibility
        subprocess.run(cmd, check=True)

    def _run_matcher(self, database_path, use_gpu):
        """Match features between images"""
        cmd = [
            self.colmap_path, 'exhaustive_matcher',
            '--database_path', str(database_path),
        ]
        # Note: --SiftMatching.use_gpu is not supported in all COLMAP versions
        # Removed to ensure compatibility
        subprocess.run(cmd, check=True)

    def _run_mapper(self, image_path, database_path, sparse_path):
        """Run sparse reconstruction"""
        cmd = [
            self.colmap_path, 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(image_path),
            '--output_path', str(sparse_path),
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0',
            '--Mapper.ba_refine_extra_params', '0',
            '--Mapper.ba_global_max_num_iterations', '50',
            '--Mapper.ba_local_max_num_iterations', '25'
        ]
        subprocess.run(cmd, check=True)

    def _parse_results(self, sparse_model_path: Path) -> Dict:
        """Parse COLMAP results and convert to standard format"""

        cameras_file = sparse_model_path / 'cameras.bin'
        images_file = sparse_model_path / 'images.bin'
        points3d_file = sparse_model_path / 'points3D.bin'

        if not cameras_file.exists():
            raise FileNotFoundError(f"COLMAP output not found at {sparse_model_path}")

        cameras = self._read_cameras_binary(cameras_file)
        images = self._read_images_binary(images_file)
        points3d = self._read_points3d_binary(points3d_file)

        return {
            'cameras': cameras,
            'images': images,
            'points3d': points3d
        }

    def _read_cameras_binary(self, path):
        """Read COLMAP cameras.bin file"""
        cameras = {}
        with open(path, 'rb') as f:
            num_cameras = struct.unpack('Q', f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack('I', f.read(4))[0]
                model_id = struct.unpack('i', f.read(4))[0]
                width = struct.unpack('Q', f.read(8))[0]
                height = struct.unpack('Q', f.read(8))[0]
                params = struct.unpack('d' * 4, f.read(8 * 4))
                cameras[camera_id] = {
                    'model_id': model_id,
                    'width': width,
                    'height': height,
                    'params': params
                }
        return cameras

    def _read_images_binary(self, path):
        """Read COLMAP images.bin file"""
        images = {}
        with open(path, 'rb') as f:
            num_images = struct.unpack('Q', f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack('I', f.read(4))[0]
                qw, qx, qy, qz = struct.unpack('d' * 4, f.read(8 * 4))
                tx, ty, tz = struct.unpack('d' * 3, f.read(8 * 3))
                camera_id = struct.unpack('I', f.read(4))[0]

                name_len = 0
                name_bytes = b''
                while True:
                    char = f.read(1)
                    if char == b'\x00':
                        break
                    name_bytes += char
                name = name_bytes.decode('utf-8')

                num_points2d = struct.unpack('Q', f.read(8))[0]
                f.read(24 * num_points2d)  # Skip point2D data

                images[image_id] = {
                    'qvec': [qw, qx, qy, qz],
                    'tvec': [tx, ty, tz],
                    'camera_id': camera_id,
                    'name': name
                }
        return images

    def _read_points3d_binary(self, path):
        """Read COLMAP points3D.bin file"""
        points3d = []
        with open(path, 'rb') as f:
            num_points = struct.unpack('Q', f.read(8))[0]
            for _ in range(num_points):
                point_id = struct.unpack('Q', f.read(8))[0]
                x, y, z = struct.unpack('d' * 3, f.read(8 * 3))
                r, g, b = struct.unpack('B' * 3, f.read(3))
                error = struct.unpack('d', f.read(8))[0]
                track_len = struct.unpack('Q', f.read(8))[0]
                f.read(8 * track_len)  # Skip track data

                points3d.append({
                    'xyz': [x, y, z],
                    'rgb': [r/255.0, g/255.0, b/255.0]
                })
        return points3d
