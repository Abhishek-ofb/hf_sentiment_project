import os, threading
from . import translation_service
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


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribes the audio and translates it to English.
    Returns:
        {
            "transcribed_text": "...",
            "translated_text": "..."
        }
    """
    if _asr_pipeline is None:
        initialize_asr()

    try:
        result = _asr_pipeline(audio_path, return_timestamps=True)

        # Get plain text from segments
        if "text" in result:
            transcribed_text = result["text"]
        elif "chunks" in result:
            transcribed_text = " ".join([c.get("text", "") for c in result["chunks"]])
        else:
            transcribed_text = str(result)

        # Translate to English
        translated_text = translation_service.translate_to_english(transcribed_text)

        return {
            "transcribed_text": transcribed_text,
            "translated_text": translated_text
        }

    except Exception as e:
        print(f"ASR transcription failed: {e}")
        return {
            "transcribed_text": "",
            "translated_text": ""
        }
