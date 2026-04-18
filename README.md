# Nature Sense

Tools for recording and analyzing synchronized audio and video with acoustic beamforming.

## Scripts

| Script | Description |
|--------|-------------|
| `record.py` | Captures synchronized 16-channel audio (UMA-16) and video, saving `audio.h5`, `video.mp4`, and `sync.csv` to a timestamped session directory. |
| `stitch.py` | Re-encodes a session's video to 30 fps (using `sync.csv` timestamps) and muxes it with mixed-down mono audio into `output.mp4`. |
| `reprocess.py` | Runs acoustic beamforming on a recorded session and overlays a sound-source heatmap on the video, producing `visualization.mp4`. |
| `calibrate_april_tag.py` | Interactive AprilTag-board calibration guided by an 18-stage pose grid; auto-saves frames and writes `calibration.yaml` + `calibration_detections.npz`. |
| `calibration_metrics.py` | Loads `calibration_detections.npz`, computes per-frame reprojection residuals, and saves a PNG with spatial and per-stage error heatmaps. Optional `--overwrite-yaml` writes the stored calibration back to `calibration.yaml`. |
| `viewer.py` | Frontend viewer for browsing session results including spectrograms and beamforming visualizations. |
| `biodenoising_modal.py` | Modal serverless script that denoises animal vocalizations in a session audio file using the biodenoising model on a cloud GPU. |
| `sam_audio_modal.py` | Modal serverless script that isolates a target sound (by text description) from a session recording using Meta's SAM-Audio model on a cloud GPU. |

## Hardware Requirements

- miniDSP UMA-16 microphone array (16-channel USB audio interface)
- Webcam (for video recording)

## Installation

### 1. Create Conda Environment

```bash
conda create -n nature_sense python=3.11
conda activate nature_sense
```

### 2. Install Dependencies

First install binary dependencies via conda, then pip install the rest:

```bash
conda install -c conda-forge numba
pip install -r requirements.txt
```

### 3. Install Pre-Commit Hooks

This project uses [`prek`](https://github.com/j178/prek) (a fast pre-commit replacement) to run linting (`ruff`), formatting (`ruff-format`), and type checking (`mypy`) on every commit. Install the git hooks once after cloning:

```bash
prek install
```

To run all hooks across the repo manually:

```bash
prek run --all-files
```
