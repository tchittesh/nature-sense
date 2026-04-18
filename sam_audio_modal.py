from __future__ import annotations

import io
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal image: CUDA base + PyTorch + SAM-Audio from GitHub
# ---------------------------------------------------------------------------

sam_audio_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torchaudio",
        "huggingface_hub",
        "transformers",
    )
    .pip_install(
        "git+https://github.com/facebookresearch/sam-audio.git"
    )
)

# ---------------------------------------------------------------------------
# Modal app + HuggingFace model cache volume
# ---------------------------------------------------------------------------

app = modal.App("nature-sense-sam-audio")

hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL_ID = "facebook/sam-audio-large"
HF_CACHE_DIR = "/root/.cache/huggingface"


# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------

@app.function(
    image=sam_audio_image,
    gpu="L4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={HF_CACHE_DIR: hf_cache_volume},
    memory=16384,
)
def separate_audio(
    audio_bytes: bytes,
    description: str,
    audio_filename: str = "input.mp4",
) -> tuple[bytes, bytes]:
    """
    Run SAM-Audio on the provided audio/video bytes.

    Returns (target_wav_bytes, residual_wav_bytes).
    """
    import tempfile

    import torch
    import torchaudio
    from sam_audio import SAMAudio, SAMAudioProcessor

    device = torch.device("cuda")

    print(f"Loading model {MODEL_ID}...", flush=True)
    model = SAMAudio.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR)
    hf_cache_volume.commit()
    print("Model loaded.", flush=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / audio_filename
        input_path.write_bytes(audio_bytes)

        print(f"Processing: '{description}'", flush=True)
        batch = processor(
            audios=[str(input_path)],
            descriptions=[description],
        ).to(device)

        with torch.inference_mode():
            result = model.separate(batch, predict_spans=False, reranking_candidates=1)

        sample_rate = processor.audio_sampling_rate

        target_buf = io.BytesIO()
        residual_buf = io.BytesIO()
        torchaudio.save(target_buf, result.target.cpu(), sample_rate, format="wav")
        torchaudio.save(residual_buf, result.residual.cpu(), sample_rate, format="wav")

    print("Done.", flush=True)
    return target_buf.getvalue(), residual_buf.getvalue()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    input_path: str,
    description: str = "bird",
    output_dir: str = ".",
) -> None:
    """
    Run SAM-Audio on an mp4 (or audio) file from a nature-sense session.

    Args:
        input_path:  Path to output.mp4 (or any audio/video file).
        description: Sound to isolate, e.g. "bird singing", "wind", "insect".
        output_dir:  Where to save target.wav and residual.wav.
    """
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        print(f"Error: {input_file} does not exist", file=sys.stderr)
        sys.exit(1)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem
    target_path = output_path / f"{stem}_target_{description.replace(' ', '_')}.wav"
    residual_path = output_path / f"{stem}_residual_{description.replace(' ', '_')}.wav"

    print(f"Uploading {input_file.name} ({input_file.stat().st_size / 1e6:.1f} MB)...")
    audio_bytes = input_file.read_bytes()

    print(f"Running SAM-Audio on GPU (description: '{description}')...")
    target_bytes, residual_bytes = separate_audio.remote(
        audio_bytes=audio_bytes,
        description=description,
        audio_filename=input_file.name,
    )

    target_path.write_bytes(target_bytes)
    residual_path.write_bytes(residual_bytes)

    print(f"Saved:")
    print(f"  target:   {target_path}")
    print(f"  residual: {residual_path}")
