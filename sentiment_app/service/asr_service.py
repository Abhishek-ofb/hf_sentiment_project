import os, threading
from transformers import pipeline

ASR_MODEL_NAME = os.getenv("ASR_MODEL", "Oriserve/Whisper-Hindi2Hinglish-Swift")

_asr_pipeline = None
_lock = threading.Lock()


def initialize_asr():
    global _asr_pipeline
    if _asr_pipeline is None:
        with _lock:
            if _asr_pipeline is None:
                print(f"Loading ASR model: {ASR_MODEL_NAME}")
                _asr_pipeline = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME)


def transcribe(audio_path: str) -> str:
    if _asr_pipeline is None:
        initialize_asr()
    try:
        # Enable return_timestamps for long audio
        result = _asr_pipeline(audio_path, return_timestamps=True)

        # Some models return segments instead of plain "text"
        if "text" in result:
            return result["text"]
        elif "chunks" in result:
            return " ".join([c.get("text", "") for c in result["chunks"]])
        else:
            return str(result)
    except Exception as e:
        print(f"ASR transcription failed: {e}")
        return ""
