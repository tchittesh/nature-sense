"""
Standalone synchronized audio + video recorder for the acoustic camera.

Records:
  - 16-channel audio from UMA-16 microphone array → HDF5
  - Video from webcam → MP4
  - Synchronization timestamps → CSV
"""

import argparse
import sys
import time
import datetime
import csv
import signal
from pathlib import Path
from threading import Thread, Event
import cv2
import acoular as ac
import numpy as np
import matplotlib.pyplot as plt

from utils import SoundDeviceSamplesGeneratorFp64, get_uma16_index


CAMERA_INDEX = 0
"""Default camera index to use."""

VIDEO_FPS = 45
"""Arbitrary FPS to record video at. In reality, the frame rate will differ from this value."""


class SyncedRecorder:
    """Records synchronized audio and video streams.

    Captures 16-channel audio from UMA-16 microphone array and video from webcam,
    maintaining precise timestamp synchronization between the two streams.

    Args:
        output_dir: Base directory for saving session recordings.
    """

    def __init__(self, output_dir):
        # Timestamp for this recording session
        self.session_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.session_dir = Path(output_dir) / self.session_time
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        self.audio_path = self.session_dir / 'audio.h5'
        self.video_path = self.session_dir / 'video.mp4'
        self.sync_path = self.session_dir / 'sync.csv'

        # Device indices
        self.device_index = get_uma16_index()

        # Recording state
        self.stop_event = Event()
        self.start_timestamp = None
        self.video_timestamps = []
        self.audio_blocks_written = 0

        # Audio settings
        self.sample_rate = 48000
        self.num_channels = 16
        self.audio_block_size = 1024

    def _setup_audio(self):
        """Set up acoular audio capture pipeline."""
        self.audio_source = SoundDeviceSamplesGeneratorFp64(
            device=self.device_index,
            numchannels=self.num_channels,
        )

        # Volt to Pascal conversion
        self.source_mixer = ac.SourceMixer(
            sources=[self.audio_source],
            weights=np.array([1/0.0016])
        )

        # HDF5 writer
        self.h5_writer = ac.WriteH5(
            source=self.source_mixer,
            file=str(self.audio_path)
        )

    def _setup_video(self):
        """Set up OpenCV video capture and writer."""
        self.video_capture = cv2.VideoCapture(CAMERA_INDEX)
        # Buffer size of 1 ensures we always get the most recent frame,
        # avoiding latency from stale queued frames
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get actual dimensions (may differ from requested)
        actual_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            VIDEO_FPS,
            (actual_width, actual_height)
        )

        print(f"Video: {actual_width}x{actual_height} @ {actual_fps} fps")

    def _audio_thread(self):
        """Thread for recording audio."""
        gen = self.h5_writer.result(num=self.audio_block_size)

        while not self.stop_event.is_set():
            try:
                next(gen)
                self.audio_blocks_written += 1
            except StopIteration:
                break

    def _video_loop(self):
        """Main thread video capture, recording, and preview with matplotlib."""
        frame_count = 0

        # Set up matplotlib figure
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title("Recording - Close window to stop")
        ax.axis('off')
        img_plot = None

        while not self.stop_event.is_set():
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                continue

            # Record timestamp
            timestamp = time.time()
            self.video_timestamps.append((frame_count, timestamp))

            # TODO: These operations (writing video frames and showing in UI)
            # are pretty heavy and they cause us to drop frames. Figure out
            # how to optimize them

            # Write frame to video file
            self.video_writer.write(frame)
            frame_count += 1

            if frame_count % 60 == 0:
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Add recording indicator text
                elapsed = timestamp - self.start_timestamp if self.start_timestamp else 0
                cv2.putText(
                    frame_rgb,
                    f"REC {elapsed:.1f}s  |  Frames: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

                # Update matplotlib display
                if img_plot is None:
                    img_plot = ax.imshow(frame_rgb)
                else:
                    img_plot.set_data(frame_rgb)

                # Check if window was closed
                if not plt.fignum_exists(fig.number):
                    self.stop_event.set()
                    break
                plt.pause(0.0001)

        plt.close(fig)

    def _save_sync_data(self):
        """Save synchronization data to CSV."""
        end_timestamp = time.time()
        total_samples = self.audio_blocks_written * self.audio_block_size
        actual_duration = end_timestamp - self.start_timestamp

        # Calculate actual sample rate (to detect clock drift)
        if actual_duration > 0:
            measured_sample_rate = total_samples / actual_duration
            drift_ppm = ((measured_sample_rate - self.sample_rate) / self.sample_rate) * 1e6
        else:
            measured_sample_rate = self.sample_rate
            drift_ppm = 0

        with open(self.sync_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header with metadata
            writer.writerow(['# Acoustic Camera Sync Data'])
            writer.writerow(['# start_timestamp', self.start_timestamp])
            writer.writerow(['# end_timestamp', end_timestamp])
            writer.writerow(['# nominal_sample_rate', self.sample_rate])
            writer.writerow(['# measured_sample_rate', f'{measured_sample_rate:.4f}'])
            writer.writerow(['# drift_ppm', f'{drift_ppm:.2f}'])
            writer.writerow(['# audio_block_size', self.audio_block_size])
            writer.writerow(['# audio_blocks_written', self.audio_blocks_written])
            writer.writerow(['# audio_total_samples', total_samples])
            writer.writerow([])

            # Video frame timestamps
            writer.writerow(['frame_idx', 'timestamp'])
            for frame_idx, ts in self.video_timestamps:
                writer.writerow([frame_idx, ts])

        # Store for summary output
        self.end_timestamp = end_timestamp
        self.measured_sample_rate = measured_sample_rate
        self.drift_ppm = drift_ppm

    def record(self):
        """Start recording audio and video.

        Blocks until the preview window is closed or Ctrl+C is pressed.
        Creates three output files: audio.h5, video.mp4, and sync.csv.
        """
        print(f"\nOutput directory: {self.session_dir}\n")

        # Setup
        print("Initializing audio...")
        self._setup_audio()

        print("Initializing video...")
        self._setup_video()

        # Record start timestamp
        self.start_timestamp = time.time()
        start_dt = datetime.datetime.fromtimestamp(self.start_timestamp)
        print(f"\nRecording started at {start_dt.strftime('%H:%M:%S')}")
        print("Close the preview window or Ctrl+C to stop\n")

        # Start audio thread in background
        audio_thread = Thread(target=self._audio_thread, daemon=True)
        audio_thread.start()

        # Run video capture/preview in main thread (required for matplotlib)
        try:
            self._video_loop()
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop_event.set()

        # Cleanup
        audio_thread.join(timeout=2)

        self.video_capture.release()
        self.video_writer.release()

        # Save sync data
        self._save_sync_data()

        # Summary
        total_samples = self.audio_blocks_written * self.audio_block_size
        audio_duration = total_samples / self.sample_rate

        print(f"\nRecording complete!")
        print(f"  Audio: {audio_duration:.2f}s ({self.audio_blocks_written} blocks, {total_samples} samples)")
        print(f"  Video: {len(self.video_timestamps)} frames")
        print(f"  Clock drift: {self.drift_ppm:+.1f} ppm (measured rate: {self.measured_sample_rate:.2f} Hz)")
        if abs(self.drift_ppm) > 100:
            print(f"  ⚠️  High drift detected - consider using measured_sample_rate for sync")
        print(f"\nFiles saved:")
        print(f"  {self.audio_path}")
        print(f"  {self.video_path}")
        print(f"  {self.sync_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Record synchronized audio and video for acoustic camera. ' \
        'Close the preview window or Ctrl+C to stop recording.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='results',
        help='Output directory (default: results)'
    )

    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    recorder = None
    def signal_handler(sig, frame):
        if recorder:
            recorder.stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)

    # Record
    try:
        recorder = SyncedRecorder(output_dir=args.output_dir)
        recorder.record()

    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
