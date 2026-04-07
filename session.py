"""Session file path tracker for nature sense recording sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SessionFiles:
    """Canonical file paths for a single recording session directory.

    All paths are derived from ``session_dir`` and reflect the filenames that
    the recording, stitching, and reprocessing scripts produce and consume.

    Args:
        session_dir: Path to the session directory (e.g. ``results/2026-04-02_16-47-02``).
    """

    session_dir: Path

    # Derived paths — set in __post_init__, not constructor arguments.
    audio: Path = field(init=False, repr=False)
    """Raw 16-channel audio recorded by the mic array (HDF5)."""

    video: Path = field(init=False, repr=False)
    """Raw video captured by the camera (may be variable frame-rate)."""

    sync: Path = field(init=False, repr=False)
    """Per-frame timestamp CSV written alongside the video."""

    output: Path = field(init=False, repr=False)
    """Stitched H.264 MP4 produced by stitch.py (30 fps, browser-compatible)."""

    visualization: Path = field(init=False, repr=False)
    """Optional beamforming overlay video produced by reprocess.py --visualize."""

    beamforming_results: Path = field(init=False, repr=False)
    """Stacked beamforming heatmaps saved by reprocess.py (NumPy .npy)."""

    def __post_init__(self) -> None:
        self.session_dir = Path(self.session_dir)
        self.audio = self.session_dir / "audio.h5"
        self.video = self.session_dir / "video.mp4"
        self.sync = self.session_dir / "sync.csv"
        self.output = self.session_dir / "output.mp4"
        self.visualization = self.session_dir / "visualization.mp4"
        self.beamforming_results = self.session_dir / "beamforming_results.npy"

    @property
    def name(self) -> str:
        """Session directory name (e.g. ``2026-04-02_16-47-02``)."""
        return self.session_dir.name
