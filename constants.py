"""Configuration constants for nature sense tools."""

from __future__ import annotations

from pathlib import Path
import acoular as ac

CAMERA_INDEX = 0
"""Default camera index to use for video capture."""

MIC_GEOMETRY_PATH = Path(ac.__file__).parent / "xml" / "minidsp_uma-16_mirrored.xml"
"""Microphone geometry file path (absolute path within acoular package)."""

BEAMFORMING_INCREMENT_M =  0.1
"""Beamforming grid spacing in meters."""

BEAMFORMING_FREQUENCY_HZ = 4000.0
"""Beamforming frequency in Hz."""

BEAMFORMING_XMIN_M = -2.5
"""Beamforming grid minimum X in meters."""

BEAMFORMING_XMAX_M = 2.5
"""Beamforming grid maximum X in meters."""

BEAMFORMING_YMIN_M = -1.5
"""Beamforming grid minimum Y in meters."""

BEAMFORMING_YMAX_M = 1.5
"""Beamforming grid maximum Y in meters."""

BEAMFORMING_Z_M = 2.0
"""Beamforming plane distance in meters."""

CALIBRATION_YAML_PATH = Path(__file__).parent / "calibration.yaml"
"""Path to the calibration yaml file."""
