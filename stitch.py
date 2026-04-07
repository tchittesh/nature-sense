"""Stitch audio from audio.h5 with video into a combined 30 fps MP4."""

from __future__ import annotations

import argparse
import bisect
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import acoular as ac
import cv2
import numpy as np
import scipy.io.wavfile

from session import SessionFiles

TARGET_FPS = 30.0
"""Output frame rate for all stitched videos."""


def _parse_frame_timestamps(sync_path: Path) -> list[float]:
    """Return a list of real-world elapsed times (seconds) for each frame.

    The Nth entry corresponds to the Nth frame read sequentially from the
    video file.  Times are normalised so the first frame is at t=0.

    Args:
        sync_path: Path to sync.csv.

    Returns:
        List of elapsed times in seconds, one per captured frame.
    """
    raw: list[float] = []
    with open(sync_path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "frame_idx":
                continue
            raw.append(float(row[1]))

    if not raw:
        return []

    start = raw[0]
    return [t - start for t in raw]


def _reencode_at_target_fps(
    input_video_path: Path,
    elapsed_times: list[float],
    output_video_path: Path,
) -> bool:
    """Re-encode video at TARGET_FPS using per-frame elapsed times as truth.

    For each output frame at time t = i / TARGET_FPS, selects the nearest
    input frame by real elapsed time.  This correctly handles variable capture
    rates without loading all frames into memory at once.

    Args:
        input_video_path: Source video (any frame rate).
        elapsed_times: Real elapsed time for each input frame (seconds from 0).
        output_video_path: Destination video path.

    Returns:
        True on success, False on failure.
    """
    if not elapsed_times:
        return False

    total_duration = elapsed_times[-1]
    total_output_frames = int(total_duration * TARGET_FPS) + 1

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_video_path), fourcc, TARGET_FPS, (frame_width, frame_height)
    )

    # Precompute which input frame index each output frame should use.
    # bisect gives us O(log n) nearest-neighbour lookups.
    nearest_input_indices: list[int] = []
    n_input = len(elapsed_times)
    for i in range(total_output_frames):
        target_time = i / TARGET_FPS
        if target_time > total_duration:
            break
        pos = bisect.bisect_right(elapsed_times, target_time)
        if pos == 0:
            nearest = 0
        elif pos >= n_input:
            nearest = n_input - 1
        else:
            before, after = pos - 1, pos
            nearest = (
                before
                if abs(elapsed_times[before] - target_time)
                <= abs(elapsed_times[after] - target_time)
                else after
            )
        nearest_input_indices.append(nearest)

    # Read input frames sequentially (never seek backwards) and write output.
    # Since nearest_input_indices is non-decreasing, each input frame is read
    # at most once.
    input_pos = 0
    current_frame: np.ndarray | None = None

    for nearest in nearest_input_indices:
        # Advance the input reader to the required frame.
        while input_pos <= nearest:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame
            input_pos += 1

        if current_frame is not None:
            out.write(current_frame)

    cap.release()
    out.release()
    return True


def stitch_session(session_dir: Path) -> None:
    """Combine audio.h5 with the session video into a single 30 fps MP4.

    Re-encodes the video at TARGET_FPS using sync.csv timestamps so that
    playback speed matches real-world time regardless of the declared capture
    rate.  Mixes all microphone channels to mono for the audio track.
    Output is encoded as H.264 for broad browser compatibility.

    Args:
        session_dir: Path to the session directory.
    """
    session = SessionFiles(session_dir)

    if not session.audio.exists():
        print(f"  Skipping {session.name}: audio.h5 not found")
        return

    if not session.video.exists():
        print(f"  Skipping {session.name}: no video file found")
        return

    if not session.sync.exists():
        print(f"  Skipping {session.name}: sync.csv not found")
        return

    elapsed_times = _parse_frame_timestamps(session.sync)
    if not elapsed_times:
        print(f"  Skipping {session.name}: no frame timestamps in sync.csv")
        return

    actual_fps = (len(elapsed_times) - 1) / elapsed_times[-1] if elapsed_times[-1] > 0 else 0
    print(f"  Video source : {session.video.name}")
    print(f"  Capture fps  : {actual_fps:.2f}  →  re-encoding at {TARGET_FPS:.0f} fps")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video_f:
        tmp_video_path = Path(tmp_video_f.name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_f:
        tmp_wav_path = Path(tmp_wav_f.name)

    try:
        # Re-encode video at TARGET_FPS with correct frame timing.
        print(f"  Re-encoding video...")
        if not _reencode_at_target_fps(session.video, elapsed_times, tmp_video_path):
            print(f"  ERROR: video re-encoding failed")
            return

        # Load audio and mix 16 channels to mono.
        time_samples = ac.TimeSamples(file=str(session.audio))
        sample_rate = int(time_samples.sample_freq)
        mono_audio = time_samples.data[:].mean(axis=1)

        peak = np.abs(mono_audio).max()
        if peak > 0:
            mono_audio = mono_audio / peak * 0.9

        scipy.io.wavfile.write(tmp_wav_path, sample_rate, (mono_audio * 32767).astype(np.int16))

        # Mux re-encoded video with audio via ffmpeg, encoding to H.264 for
        # browser compatibility.
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(tmp_video_path),
                "-i", str(tmp_wav_path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac",
                "-movflags", "+faststart",
                "-shortest",
                str(session.output),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ERROR running ffmpeg:\n{result.stderr}")
            return

        print(f"  Saved: {session.output}")

    finally:
        tmp_video_path.unlink(missing_ok=True)
        tmp_wav_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch audio.h5 with video for one or more session directories."
    )
    parser.add_argument(
        "session_dirs",
        nargs="*",
        help="Session directories to process. Defaults to all folders in results/.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing session folders (default: results/).",
    )
    args = parser.parse_args()

    if args.session_dirs:
        session_dirs = [Path(d) for d in args.session_dirs]
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: results directory not found: {results_dir}")
            sys.exit(1)
        session_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())

    if not session_dirs:
        print("No session directories found.")
        sys.exit(0)

    print(f"Processing {len(session_dirs)} session(s)...")
    for session_dir in session_dirs:
        print(f"\n[{session_dir.name}]")
        stitch_session(session_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
