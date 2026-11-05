import io
import os
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv(override=True)


def transcribe_gnani(audio_path: str, language_code: Optional[str] = None) -> str:
    """
    Transcribe an audio file using Gnani Vachana STT REST API v3.
    Requires GNANI_STT_API_KEY_ID in .env or environment.
    """
    url = os.getenv("GNANI_STT_URL", "https://api.vachana.ai/stt/v3").strip()
    key_id = os.getenv("GNANI_STT_API_KEY_ID", "").strip()
    lang = language_code or os.getenv("GNANI_LANGUAGE_CODE", "en-IN").strip()

    if not key_id:
        raise RuntimeError(
            "Gnani API Key ID is missing. Set GNANI_STT_API_KEY_ID in .env."
        )

    raw_model = os.getenv("GNANI_MODEL", "prisma-v3.0").strip()
    if raw_model in ("prisma-3.0", "prisma3.0", "v3", "3.0"):
        model_name = "prisma-v3.0"
    elif raw_model in ("prisma-2.5", "prisma2.5", "v2.5", "2.5"):
        model_name = "prisma-v2.5"
    else:
        model_name = raw_model or "prisma-v3.0"

    payload = {
        "language_code": lang,
        "preferred_language": lang,
        "format": "transcribe",
        "itn_native_numerals": "true",
        "model": model_name,
    }

    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                url,
                headers={"X-API-Key-ID": key_id},
                data=payload,
                files={
                    "audio_file": (
                        os.path.basename(audio_path),
                        audio_file,
                        "audio/wav",
                    )
                },
                timeout=120,
            )

        response.raise_for_status()
        result = response.json()

    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Gnani transcription timed out. Try a shorter audio sample.") from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = exc.response.text[:300] if exc.response is not None else ""
        raise RuntimeError(f"Gnani STT returned HTTP {status}: {body}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not connect to Gnani STT API: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Gnani returned a response that was not valid JSON.") from exc

    if result.get("success") is False:
        raise RuntimeError(f"Gnani STT reported an error: {result.get('message', result)}")

    transcript = result.get("transcript") or result.get("transcription") or result.get("text")
    if not isinstance(transcript, str):
        # Check if segments/data exist
        data = result.get("data")
        if isinstance(data, dict) and "transcript" in data:
            transcript = data["transcript"]
        elif isinstance(result.get("response"), str):
            transcript = result.get("response")
        else:
            raise RuntimeError(f"Gnani response did not contain a transcript: {result}")

    return transcript.strip()


def transcribe_local_fallback(audio_path: str) -> str:
    """Local fallback using faster-whisper if installed, else informative message."""
    try:
        from faster_whisper import WhisperModel
        model_name = os.getenv("WHISPER_MODEL", "base")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except ImportError:
        raise RuntimeError(
            "Local faster-whisper is not installed. Please ensure Gnani STT API key "
            "is set in .env or install faster-whisper."
        )


def transcribe_audio(audio_path: str, language_code: Optional[str] = None) -> str:
    """
    Main transcription entry point:
    Prioritizes Gnani STT API with automatic fallback to local speech engine.
    """
    provider = os.getenv("STT_PROVIDER", "gnani").lower().strip()
    key_id = os.getenv("GNANI_STT_API_KEY_ID", "").strip()

    if provider == "gnani" or key_id:
        try:
            return transcribe_gnani(audio_path, language_code=language_code)
        except Exception as exc:
            # If Gnani is unavailable, attempt local fallback
            try:
                return transcribe_local_fallback(audio_path)
            except Exception:
                # Re-raise the primary Gnani exception with helpful diagnostic info
                raise exc

    return transcribe_local_fallback(audio_path)


def synthesize_speech(text: str, language: str = "en") -> bytes:
    """
    Generate spoken safety advice.
    Prioritizes Gnani TTS if configured in .env; otherwise uses gTTS for high-quality
    spoken output in English, Telugu, and Hindi.
    """
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Cannot synthesize empty text.")

    # 1. Try Gnani TTS if API key is configured
    gnani_key = os.getenv("GNANI_API_KEY", "").strip()
    if gnani_key:
        try:
            url = os.getenv("GNANI_TTS_URL", "https://api.gnani.ai/v1/tts/inference").strip()
            payload = {
                "model": "timbre-2.5",
                "text": clean_text,
                "language": "te-IN" if language.startswith("te") else ("hi-IN" if language.startswith("hi") else "en-IN"),
                "voice": os.getenv("GNANI_TTS_VOICE", "Yashvi"),
                "sample_rate": 24000,
            }
            res = requests.post(
                url,
                headers={"Authorization": f"Bearer {gnani_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if res.status_code == 200 and ("audio" in res.headers.get("Content-Type", "").lower() or len(res.content) > 100):
                return res.content
        except Exception:
            pass  # Fallback to gTTS seamlessly

    # 2. Robust fallback using gTTS
    try:
        from gtts import gTTS
        lang_code = "te" if language.startswith("te") else ("hi" if language.startswith("hi") else "en")
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        raise RuntimeError(f"Speech synthesis error: {exc}")
