"""Interactive web viewer for nature sense session data.

Serves a timeline spectrogram synced to video playback.  Users can drag a
rectangle on the spectrogram to select a time-frequency region, then run
beamforming on that region and see the result overlaid on the video.

Usage:
    conda run -n nature_sense python viewer.py [--results-dir results] [--port 5000]
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import acoular as ac
import h5py
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.image
import numpy as np
import scipy.signal
from flask import Flask, jsonify, render_template, request, send_file

matplotlib.use("Agg")

from constants import (
    BEAMFORMING_INCREMENT_M,
    BEAMFORMING_XMAX_M,
    BEAMFORMING_XMIN_M,
    BEAMFORMING_YMAX_M,
    BEAMFORMING_YMIN_M,
    BEAMFORMING_Z_M,
    MIC_GEOMETRY_PATH,
)
from session import SessionFiles

# Beamforming grid: acoular result is indexed as [x_col, y_row] before transpose.
# After reshape(GRID_DIM) + transpose + flip, the array is [row, col] matching
# the video frame (top-left origin, row=y, col=x).
GRID_DIM = (
    int((BEAMFORMING_XMAX_M - BEAMFORMING_XMIN_M) / BEAMFORMING_INCREMENT_M + 1),  # x → cols
    int((BEAMFORMING_YMAX_M - BEAMFORMING_YMIN_M) / BEAMFORMING_INCREMENT_M + 1),  # y → rows
)

SPECTROGRAM_FREQ_MAX_HZ = 12_000
"""Upper frequency limit for the displayed spectrogram."""

app = Flask(__name__)
RESULTS_DIR = Path("results")


# ── Session list ──────────────────────────────────────────────────────────────

@app.get("/api/sessions")
def list_sessions():
    sessions = sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir())
    return jsonify(sessions)


# ── Session info ──────────────────────────────────────────────────────────────

@app.get("/api/session/<name>/info")
def session_info(name: str):
    session = SessionFiles(RESULTS_DIR / name)
    ts = ac.TimeSamples(file=str(session.audio))
    return jsonify({
        "duration_s": float(ts.num_samples / ts.sample_freq),
        "sample_rate": float(ts.sample_freq),
        "n_channels": int(ts.num_channels),
        "freq_max_hz": SPECTROGRAM_FREQ_MAX_HZ,
    })


# ── Video ─────────────────────────────────────────────────────────────────────

@app.get("/api/session/<name>/video")
def serve_video(name: str):
    """Serve the stitched H.264 output video for a session.

    Expects stitch.py to have already produced output.mp4 (H.264).
    Falls back to video.mp4 if output.mp4 is not present.
    """
    session = SessionFiles(RESULTS_DIR / name)
    for candidate in (session.output, session.video):
        if candidate.exists():
            return send_file(str(candidate), mimetype="video/mp4", conditional=True)
    return "Video not found", 404


# ── Spectrogram ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _spectrogram_cached(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute (and cache) the spectrogram for a session.

    Returns:
        (Sxx_db, times, freqs, sample_rate)
        Sxx_db shape: (n_freq_bins, n_time_bins)  — freq axis trimmed to FREQ_MAX
    """
    session = SessionFiles(RESULTS_DIR / name)
    ts = ac.TimeSamples(file=str(session.audio))
    sample_rate = float(ts.sample_freq)
    mono = ts.data[:].mean(axis=1)

    freqs, times, Sxx = scipy.signal.spectrogram(
        mono, fs=sample_rate, nperseg=4096, noverlap=3072, window="hann"
    )
    mask = freqs <= SPECTROGRAM_FREQ_MAX_HZ
    Sxx_db = 10.0 * np.log10(Sxx[mask] + 1e-10)
    return Sxx_db, times, freqs[mask], sample_rate


@app.get("/api/session/<name>/spectrogram.png")
def get_spectrogram(name: str):
    """Return spectrogram as a raw PNG image (no axes).

    Response headers carry duration and max-frequency metadata so the
    frontend can map canvas pixels to time/frequency without a separate
    request.
    """
    Sxx_db, times, freqs, _ = _spectrogram_cached(name)

    vmin = float(np.percentile(Sxx_db, 5))
    vmax = float(np.percentile(Sxx_db, 99))

    # Flip freq axis so row-0 of the image = high frequency (top of display).
    rgba = cm.inferno(mcolors.Normalize(vmin=vmin, vmax=vmax)(Sxx_db[::-1]))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    buf = io.BytesIO()
    matplotlib.image.imsave(buf, rgb, format="png")
    buf.seek(0)

    resp = send_file(buf, mimetype="image/png")
    resp.headers["X-Duration-S"] = f"{float(times[-1]):.4f}"
    resp.headers["X-Freq-Max-Hz"] = f"{float(freqs[-1]):.1f}"
    resp.headers["Access-Control-Expose-Headers"] = "X-Duration-S, X-Freq-Max-Hz"
    return resp


# ── Beamforming ───────────────────────────────────────────────────────────────

def _beamform_segment(
    audio_segment: np.ndarray,
    sample_rate: float,
    f_center: float,
) -> np.ndarray:
    """Run time-domain beamforming on an audio segment at f_center Hz.

    Uses the same acoular pipeline as reprocess.py so results are consistent.

    Args:
        audio_segment: shape (n_samples, n_channels), float32
        sample_rate: samples per second
        f_center: centre frequency for the 1/3-octave filter (Hz)

    Returns:
        2D dB heatmap, shape (grid_rows, grid_cols), top-left origin.
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            tmp_path = f.name

        # Pad to a multiple of num_per_average (512).  The SamplesBuffer inside
        # BeamformerTime.result() looks ahead by max_sample_delay (~465 samples
        # at 44.1 kHz).  When the last partial block has fewer samples than that
        # look-ahead, acoular computes a negative block size and crashes.
        # Padding to the next multiple of 512 guarantees the last block is
        # exactly 512 samples, which is safely larger than max_sample_delay.
        NUM_PER_AVG = 512
        n_samples = audio_segment.shape[0]
        remainder = n_samples % NUM_PER_AVG
        if remainder != 0:
            pad_len = NUM_PER_AVG - remainder
            audio_segment = np.pad(
                audio_segment, ((0, pad_len), (0, 0)), mode="constant"
            )

        with h5py.File(tmp_path, "w") as hf:
            ds = hf.create_dataset("time_data", data=audio_segment.astype(np.float32))
            ds.attrs["sample_freq"] = float(sample_rate)

        ts = ac.TimeSamples(file=tmp_path)
        mic_geom = ac.MicGeom(file=str(MIC_GEOMETRY_PATH))
        grid = ac.RectGrid(
            x_min=BEAMFORMING_XMIN_M, x_max=BEAMFORMING_XMAX_M,
            y_min=BEAMFORMING_YMIN_M, y_max=BEAMFORMING_YMAX_M,
            z=BEAMFORMING_Z_M, increment=BEAMFORMING_INCREMENT_M,
        )
        steer = ac.SteeringVector(env=ac.Environment(c=343), grid=grid, mics=mic_geom)

        bf = ac.BeamformerTime(source=ts, steer=steer)
        filt = ac.FiltOctave(source=bf, band=f_center, fraction="Third octave")
        power = ac.TimePower(source=filt)
        avg = ac.Average(source=power, num_per_average=512)

        heatmaps: list[np.ndarray] = []
        for res in avg.result(num=1):
            h = ac.L_p(res).reshape(GRID_DIM)  # (x_cols, y_rows)
            h = np.transpose(h)                  # (y_rows, x_cols)
            h = h[::-1, ::-1]                    # flip to video top-left origin
            heatmaps.append(h)

        if not heatmaps:
            return np.zeros((GRID_DIM[1], GRID_DIM[0]))

        return np.mean(heatmaps, axis=0)

    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/session/<name>/beamform")
def beamform(name: str):
    import traceback
    body = request.get_json()
    t1: float = float(body["t1"])
    t2: float = float(body["t2"])
    f1: float = float(body["f1"])
    f2: float = float(body["f2"])

    session = SessionFiles(RESULTS_DIR / name)

    # Use h5py directly to avoid acoular holding an open file handle that
    # conflicts with the pipeline inside _beamform_segment.
    with h5py.File(str(session.audio), "r") as hf:
        sample_rate = float(hf["time_data"].attrs.get("sample_freq", 48000))
        n_total = hf["time_data"].shape[0]
        s1 = max(0, int(t1 * sample_rate))
        s2 = min(n_total, int(t2 * sample_rate))
        if s2 - s1 < 512:
            return jsonify({"error": "Time selection too short (need > ~10 ms)"}), 400
        audio_segment = hf["time_data"][s1:s2]

    # Use geometric mean as band centre frequency (avoids 0 Hz corner case).
    f_center = float(np.sqrt(max(f1, 1.0) * max(f2, 1.0)))

    try:
        heatmap = _beamform_segment(audio_segment, sample_rate, f_center)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500

    grid_rows, grid_cols = heatmap.shape

    return jsonify({
        "heatmap": heatmap.tolist(),
        "rows": grid_rows,
        "cols": grid_cols,
    })


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("viewer.html")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nature Sense web viewer")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    RESULTS_DIR = Path(args.results_dir)
    app.run(host=args.host, port=args.port, debug=False)
