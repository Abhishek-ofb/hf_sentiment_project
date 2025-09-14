import subprocess
import os
from . import asr_service, translation_service, sentiment_service, summarization_service

def convert_to_wav(input_path: str, output_path: str):
    """
    Converts any audio file to mono 16kHz WAV using ffmpeg.
    """
    command = [
        "ffmpeg",
        "-y",                # overwrite output
        "-i", input_path,    # input file
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz
        output_path
    ]
    subprocess.run(command, check=True)

    
def process_call(audio_path: str) -> dict:
    try:
        wav_path = audio_path.rsplit(".", 1)[0] + "_converted.wav"
        convert_to_wav(audio_path, wav_path)

        asr_result = asr_service.transcribe_audio(wav_path)
        english_text = asr_result.get("translated_text", "").strip()

        if not english_text or len(english_text) < 5:
            return {
                "transcribed_text": asr_result.get("transcribed_text", ""),
                "translated_text": "",
                "sentiment": {"label": "unknown", "score": 0.0},
                "summary": "⚠️ No valid transcription available"
            }

        sentiment = sentiment_service.analyze_sentiment(english_text)
        summary = summarization_service.summarize_call(english_text)

        return {
            "transcribed_text": asr_result["transcribed_text"],
            "translated_text": english_text,
            "sentiment": sentiment,
            "summary": summary
        }
    except Exception as e:
        print("process_call failed:", e)
        return {
            "transcribed_text": "",
            "translated_text": "",
            "sentiment": {"label": "unknown", "score": 0.0},
            "summary": f"❌ Processing failed: {e}"
        }