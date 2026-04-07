"""
Reprocess recorded session data through the beamforming pipeline.
"""

import argparse
import csv
import sys
from pathlib import Path

import acoular as ac
import cv2
import numpy as np
from tqdm import tqdm

from constants import (
    BEAMFORMING_XMIN_M,
    BEAMFORMING_XMAX_M,
    BEAMFORMING_YMIN_M,
    BEAMFORMING_YMAX_M,
    BEAMFORMING_INCREMENT_M,
    BEAMFORMING_FREQUENCY_HZ,
    BEAMFORMING_Z_M,
    MIC_GEOMETRY_PATH,
)
from session import SessionFiles


def parse_sync_csv(sync_path):
    """Parse sync.csv and extract metadata and frame timestamps.

    Args:
        sync_path: Path to the sync.csv file.

    Returns:
        Tuple of (metadata dict, list of (frame_idx, timestamp) tuples).
    """
    metadata = {}
    frame_timestamps = []

    with open(sync_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith('# '):
                # Metadata row
                key = row[0][2:]  # Remove '# ' prefix
                if len(row) > 1:
                    metadata[key] = row[1]
            elif row[0] == 'frame_idx':
                # Header row, skip
                continue
            else:
                # Frame timestamp row
                frame_idx = int(row[0])
                timestamp = float(row[1])
                frame_timestamps.append((frame_idx, timestamp))

    return metadata, frame_timestamps


def _create_visualization_video(
    video_path,
    heatmaps,
    frame_timestamps,
    start_timestamp,
    sample_rate,
    num_per_average,
    output_path,
):
    """Create a video with beamforming heatmap and max point overlaid.

    Maps video frames to beamforming frames using timestamps.

    Args:
        video_path: Path to input video file.
        heatmaps: List of 2D beamforming heatmap arrays.
        frame_timestamps: List of (frame_idx, timestamp) tuples.
        start_timestamp: Recording start timestamp.
        sample_rate: Audio sample rate in Hz.
        num_per_average: Number of samples averaged per beamforming frame.
        output_path: Path for output visualization video.
    """
    print("\nGenerating visualization video...")

    # Calculate time per beamforming frame
    # TimeAverage with num_per_average=N outputs 1 sample per N input audio samples
    # So each beamforming frame represents num_per_average/sample_rate seconds of audio
    time_per_bf_frame = num_per_average / sample_rate
    print(f"  Beamforming frame rate: {1/time_per_bf_frame:.1f} Hz ({time_per_bf_frame*1000:.2f} ms per frame)")

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # Build timestamp lookup from frame_timestamps
    frame_to_timestamp = {idx: ts for idx, ts in frame_timestamps}

    # Find global min/max for consistent colormap
    all_levels = np.concatenate(heatmaps)
    level_min = np.percentile(all_levels, 5)
    level_max = np.percentile(all_levels, 99)

    # Calculate expected duration coverage
    total_bf_duration = len(heatmaps) * time_per_bf_frame
    print(f"  Beamforming covers {total_bf_duration:.2f}s ({len(heatmaps)} frames)")
    print(f"  Video has {total_frames} frames at {frame_width}x{frame_height}, {fps:.1f} fps ({total_frames/fps:.2f}s)")

    frame_idx = 0
    last_bf_idx = 0  # Track last used beamforming frame for interpolation

    with tqdm(total=total_frames, desc="Rendering video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get timestamp for this video frame
            bf_idx = last_bf_idx  # Default to last known if missing
            if frame_idx in frame_to_timestamp:
                video_timestamp = frame_to_timestamp[frame_idx]
                elapsed = video_timestamp - start_timestamp

                # Map to beamforming frame index
                bf_idx = int(elapsed / time_per_bf_frame)
                bf_idx = max(0, min(bf_idx, len(heatmaps) - 1))
                last_bf_idx = bf_idx

            # Get heatmap and result for this frame
            heatmap = heatmaps[bf_idx]

            # Normalize heatmap to 0-255
            heatmap_norm = np.clip((heatmap - level_min) / (level_max - level_min), 0, 1)
            heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

            # Apply colormap (jet)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Resize heatmap to match video frame
            heatmap_resized = cv2.resize(heatmap_color, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

            # Blend heatmap with video frame
            alpha = 0.4
            frame = cv2.addWeighted(frame, 1 - alpha, heatmap_resized, alpha, 0)

            # Map from world coords to normalized [0,1]
            max_idx = np.argmax(heatmap)
            row, col = np.unravel_index(max_idx, heatmap.shape)
            num_rows, num_cols = heatmap.shape
            x_norm = col / num_cols
            y_norm = row / num_rows

            # Map to pixel coords (flip y for image coordinates)
            px = int(x_norm * frame_width)
            py = int(y_norm * frame_height)

            # Draw crosshair at max point
            cv2.circle(frame, (px, py), 10, (0, 255, 0), 2)
            cv2.line(frame, (px - 15, py), (px + 15, py), (0, 255, 0), 2)
            cv2.line(frame, (px, py - 15), (px, py + 15), (0, 255, 0), 2)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    print(f"Visualization saved to: {output_path}")


def reprocess(session_dir, output_path=None, visualize=False):
    """
    Reprocess a session directory through the beamforming pipeline.

    Args:
        session_dir: Path to session directory (contains audio.h5, video.mp4, sync.csv)
        output_path: Path for output NPY file (optional, auto-generated if not provided)
        visualize: If True, generate a video with heatmap overlay

    Returns:
        List of result dictionaries with keys: time_idx, x, y, level
    """
    session = SessionFiles(session_dir)

    # Verify session directory structure
    if not session.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {session.audio}")
    if not session.sync.exists():
        raise FileNotFoundError(f"Sync file not found: {session.sync}")
    if visualize and not session.video.exists():
        raise FileNotFoundError(f"Video file not found: {session.video}")

    # Parse sync data
    metadata, frame_timestamps = parse_sync_csv(session.sync)
    start_timestamp = float(metadata.get('start_timestamp', 0))
    sample_rate = float(metadata.get('measured_sample_rate', metadata.get('nominal_sample_rate', 48000)))

    print(f"Session: {session.name}")
    print(f"Audio: {session.audio}")
    print(f"Frequency: {BEAMFORMING_FREQUENCY_HZ} Hz")
    print(f"Z distance: {BEAMFORMING_Z_M} m")
    print(f"Grid: x=[{BEAMFORMING_XMIN_M}, {BEAMFORMING_XMAX_M}], y=[{BEAMFORMING_YMIN_M}, {BEAMFORMING_YMAX_M}], increment={BEAMFORMING_INCREMENT_M}")
    print(f"Sample rate: {sample_rate:.2f} Hz")

    # Calculate grid dimensions
    grid_dim = (
        int((BEAMFORMING_XMAX_M - BEAMFORMING_XMIN_M) / BEAMFORMING_INCREMENT_M + 1),
        int((BEAMFORMING_YMAX_M - BEAMFORMING_YMIN_M) / BEAMFORMING_INCREMENT_M + 1)
    )

    # Set up acoular pipeline
    mic_geometry = ac.MicGeom(file=MIC_GEOMETRY_PATH)
    ts = ac.TimeSamples(file=str(session.audio))

    print(f"Audio samples: {ts.num_samples}")
    print(f"Audio duration: {ts.num_samples / ts.sample_freq:.2f} s")

    # Beamforming grid
    grid = ac.RectGrid(
        x_min=BEAMFORMING_XMIN_M, x_max=BEAMFORMING_XMAX_M,
        y_min=BEAMFORMING_YMIN_M, y_max=BEAMFORMING_YMAX_M,
        z=BEAMFORMING_Z_M, increment=BEAMFORMING_INCREMENT_M
    )

    # Steering vector
    steer = ac.SteeringVector(env=ac.Environment(c=343), grid=grid, mics=mic_geometry)

    # Processing pipeline
    bf = ac.BeamformerTime(source=ts, steer=steer)
    filt = ac.FiltOctave(source=bf, band=BEAMFORMING_FREQUENCY_HZ, fraction='Third octave')
    power = ac.TimePower(source=filt)
    bf_out = ac.Average(source=power, num_per_average=512)

    # Process and collect results
    heatmaps = []
    for res in tqdm(bf_out.result(num=1), desc="Running beamforming"):
        res = ac.L_p(res)
        res = res.reshape(grid_dim)
        res = np.transpose(res)
        res = res[::-1, ::-1]
        heatmaps.append(res)

    # Save heatmaps as numpy array
    if output_path is None:
        output_path = session.beamforming_results

    heatmaps = np.stack(heatmaps)
    np.save(output_path, heatmaps)

    print(f"\nHeatmaps saved to: {output_path}")
    print(f"Shape: {heatmaps.shape} (frames, height, width)")

    # Generate visualization video if requested
    if visualize:
        _create_visualization_video(
            video_path=session.video,
            heatmaps=heatmaps,
            frame_timestamps=frame_timestamps,
            start_timestamp=start_timestamp,
            sample_rate=sample_rate,
            num_per_average=512,  # Must match the num_per_average used in TimeAverage above
            output_path=session.visualization,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Reprocess session data through the beamforming pipeline'
    )
    parser.add_argument('session_dir', help='Path to session directory')
    parser.add_argument('--output', '-o', help='Output .npy file path')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualization video with heatmap overlay')

    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"Error: Session directory not found: {session_dir}")
        sys.exit(1)
    if not session_dir.is_dir():
        print(f"Error: Not a directory: {session_dir}")
        sys.exit(1)

    reprocess(
        session_dir,
        output_path=args.output,
        visualize=args.visualize,
    )


if __name__ == '__main__':
    main()
