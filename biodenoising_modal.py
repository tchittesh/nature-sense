from __future__ import annotations

import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal image: CUDA base + biodenoising
# ---------------------------------------------------------------------------

biodenoising_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "build-essential")
    .pip_install("biodenoising")
)

app = modal.App("nature-sense-biodenoising")

MODEL = "biodenoising16k_dns48"


# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------

@app.function(
    image=biodenoising_image,
    gpu="L4",
    timeout=600,
)
def denoise_audio(audio_bytes: bytes, filename: str = "input.mp4") -> bytes:
    """
    Denoise animal vocalizations in an audio/video file using biodenoising.

    Accepts any ffmpeg-readable file (mp4, wav, etc.).
    Returns denoised mono WAV bytes at the model's native 16 kHz.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        input_path = tmp_path / filename
        input_path.write_bytes(audio_bytes)

        # Extract mono WAV from input (mp4 or audio file)
        wav_path = tmp_path / "input.wav"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-ac", "1",          # mono
                "-ar", "16000",      # resample to 16 kHz for model
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )

        noisy_dir = tmp_path / "noisy"
        noisy_dir.mkdir()
        (noisy_dir / "input.wav").write_bytes(wav_path.read_bytes())

        out_dir = tmp_path / "denoised"
        out_dir.mkdir()

        print(f"Running {MODEL}...", flush=True)
        subprocess.run(
            [
                "biodenoise",
                "--method", MODEL,
                "--noisy_dir", str(noisy_dir),
                "--out_dir", str(out_dir),
                "--device", "cuda",
            ],
            check=True,
            capture_output=False,
        )

        denoised_files = list(out_dir.rglob("*.wav"))
        if not denoised_files:
            raise RuntimeError(f"No output WAV found in {out_dir}")

        print(f"Done. Output: {denoised_files[0].name}", flush=True)
        return denoised_files[0].read_bytes()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    input_path: str,
    output_dir: str | None = None,
) -> None:
    """
    Denoise animal vocalizations in a nature-sense session recording.

    Args:
        input_path:  Path to output.mp4 (or any audio/video file).
        output_dir:  Where to save the denoised WAV. Defaults to same
                     directory as input_path.
    """
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        print(f"Error: {input_file} does not exist", file=sys.stderr)
        sys.exit(1)

    save_dir = Path(output_dir).expanduser().resolve() if output_dir else input_file.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{input_file.stem}_denoised.wav"

    print(f"Uploading {input_file.name} ({input_file.stat().st_size / 1e6:.1f} MB)...")
    audio_bytes = input_file.read_bytes()

    print("Running biodenoising on GPU...")
    denoised_bytes = denoise_audio.remote(
        audio_bytes=audio_bytes,
        filename=input_file.name,
    )

    output_path.write_bytes(denoised_bytes)
    print(f"Saved: {output_path}")
