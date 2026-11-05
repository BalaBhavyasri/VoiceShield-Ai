
import os
from pathlib import Path
from dotenv import load_dotenv
from speech import transcribe_audio

load_dotenv(override=True)

print("=== VoiceShield AI | Prisma STT Test ===")
print("Provider:", os.getenv("STT_PROVIDER", "local"))

audio_input = input(
    "Enter the full path to a test audio file (WAV/MP3/M4A/OGG/FLAC): "
).strip().strip('"')

audio_path = Path(audio_input)

if not audio_path.is_file():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

print(f"\nTesting file: {audio_path.name}")
print("Sending audio to Gnani. Please wait...")

try:
    transcript = transcribe_audio(str(audio_path))

    if not transcript:
        print("\nWARNING: The API returned an empty transcript.")
    else:
        print("\nSUCCESS: Transcription returned!")
        print("-" * 45)
        print(transcript)
        print("-" * 45)

except Exception as exc:
    print("\nTRANSCRIPTION FAILED")
    print(f"{type(exc).__name__}: {exc}")
    print("\nCheck your .env settings, API access, and audio file.")
