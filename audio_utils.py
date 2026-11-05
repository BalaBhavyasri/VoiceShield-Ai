import os
import subprocess
import tempfile
import wave
from typing import Any, Dict


def inspect_audio(file_path: str) -> Dict[str, Any]:
    """
    Inspect an audio file to assess duration, volume/energy, and quality.
    Identifies unclear or very short recordings to prevent confident false conclusions.
    """
    if not os.path.exists(file_path):
        return {
            "duration_sec": 0.0,
            "status": "missing_file",
            "is_uncertain": True,
            "warning": "Audio file not found."
        }

    file_size = os.path.getsize(file_path)
    if file_size <= 44:  # standard empty wav header
        return {
            "duration_sec": 0.0,
            "status": "empty",
            "is_uncertain": True,
            "warning": "The recorded audio file is empty or corrupted."
        }

    duration_sec = 0.0
    channels = 1
    sample_rate = 16000

    try:
        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            duration_sec = frames / float(sample_rate) if sample_rate > 0 else 0.0
    except wave.Error:
        duration_sec = estimate_duration_with_ffmpeg(file_path)

    warnings = []
    is_uncertain = False

    if duration_sec < 1.8:
        warnings.append(
            f"Recording is very short ({duration_sec:.1f}s). Voice patterns may be clipped."
        )
        is_uncertain = True
    elif duration_sec > 300:
        warnings.append(
            f"Recording is lengthy ({duration_sec / 60:.1f} min). Transcription may take longer."
        )

    return {
        "duration_sec": round(duration_sec, 2),
        "file_size_kb": round(file_size / 1024, 1),
        "channels": channels,
        "sample_rate": sample_rate,
        "is_uncertain": is_uncertain,
        "status": "warning" if is_uncertain else "healthy",
        "warning": " ".join(warnings) if warnings else None
    }


def estimate_duration_with_ffmpeg(file_path: str) -> float:
    """Fallback duration estimation using ffmpeg stdout/stderr."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_exe, "-i", file_path]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=5)
        for line in res.stderr.splitlines():
            if "Duration:" in line:
                time_str = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = time_str.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        pass
    return max(1.0, round(os.path.getsize(file_path) / 16000, 2))


def convert_to_pcm_wav(file_bytes: bytes, original_filename: str = "audio.wav") -> str:
    """
    Converts incoming audio bytes (MP3/M4A/OGG/WAV) to standard 16kHz mono PCM WAV.
    Returns path to a temporary WAV file. The caller is responsible for cleaning up.
    """
    import imageio_ffmpeg

    suffix = os.path.splitext(original_filename)[1].lower() or ".audio"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(file_bytes)
        temp_input = src.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
        temp_wav = dst.name

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", temp_input,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            temp_wav,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {res.stderr[-300:]}")
        return temp_wav
    finally:
        if os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except OSError:
                pass
