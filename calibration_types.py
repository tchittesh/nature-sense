"""Utilities for camera calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import yaml

from constants import CALIBRATION_YAML_PATH


@dataclass
class CameraMatrix:
    """Camera intrinsic matrix parameters."""

    fx: float
    fy: float
    cx: float
    cy: float

    def to_matrix(self) -> np.ndarray:
        """Convert to OpenCV camera matrix format."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> CameraMatrix:
        """Create from OpenCV camera matrix."""
        return cls(
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
        )


@dataclass
class ImageSize:
    """Image dimensions in pixels."""

    width: int
    height: int


@dataclass
class DistortionCoefficients:
    """Lens distortion coefficients (Brown-Conrady model)."""

    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    def to_array(self) -> np.ndarray:
        """Convert to OpenCV distortion coefficients format."""
        return np.array([[self.k1, self.k2, self.p1, self.p2, self.k3]], dtype=np.float64)

    @classmethod
    def from_array(cls, dist: np.ndarray) -> DistortionCoefficients:
        """Create from OpenCV distortion coefficients."""
        return cls(
            k1=float(dist[0, 0]),
            k2=float(dist[0, 1]),
            p1=float(dist[0, 2]),
            p2=float(dist[0, 3]),
            k3=float(dist[0, 4]),
        )


@dataclass
class CalibrationResult:
    """Complete camera calibration result."""

    camera_matrix: CameraMatrix
    distortion: DistortionCoefficients
    image_size: ImageSize
    rms_error_pixels: float

    def get_opencv_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Get camera matrix and distortion coefficients for OpenCV functions."""
        return self.camera_matrix.to_matrix(), self.distortion.to_array()

    def save(self) -> None:
        """Save calibration to YAML file at CALIBRATION_YAML_PATH."""
        with open(CALIBRATION_YAML_PATH, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def load(cls) -> CalibrationResult:
        """Load camera calibration from YAML file at CALIBRATION_YAML_PATH."""
        with open(CALIBRATION_YAML_PATH) as f:
            data = yaml.safe_load(f)

        return cls(
            camera_matrix=CameraMatrix(**data["camera_matrix"]),
            distortion=DistortionCoefficients(**data["distortion"]),
            image_size=ImageSize(**data["image_size"]),
            rms_error_pixels=data["rms_error_pixels"],
        )
