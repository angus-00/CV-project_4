"""Camera parameter handling"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def get_extrinsic_matrix(self) -> np.ndarray:
        """Get 4x4 extrinsic matrix [R|t]"""
        if self.rotation is None or self.translation is None:
            return np.eye(4)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = self.rotation
        extrinsic[:3, 3] = self.translation.flatten()
        return extrinsic


class Camera:
    """Camera class with R, t, K interface for compatibility"""
    def __init__(self, R: np.ndarray, t: np.ndarray, K: np.ndarray, width: int, height: int):
        self.R = R
        self.t = t
        self.K = K
        self.width = width
        self.height = height
