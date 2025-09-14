from . import asr_service, translation_service, sentiment_service, summarization_service


def process_audio(audio_path: str) -> dict:
    """Full pipeline for audio input"""
    # Step 1: Transcribe Hindi/Hinglish
    transcribed = asr_service.transcribe(audio_path)

    # Step 2: Translate Hinglish → English
    english_text = translation_service.translate_to_english(transcribed)
    print(f"Translated Text: {english_text}")
    # Step 3: Run Sentiment + Summary in parallel (could use threads if needed)
    sentiment = sentiment_service.analyze_sentiment(english_text)
    summary = summarization_service.summarize_call(english_text)

    return {
        "transcribed_text": transcribed,
        "translated_text": english_text,
        "sentiment": sentiment,
        "summary": summary
    }


def process_text(text: str) -> dict:
    """Pipeline for raw text input"""
    english_text = translation_service.translate_to_english(text)
    sentiment = sentiment_service.analyze_sentiment(english_text)
    summary = summarization_service.summarize_call(english_text)

    return {
        "input_text": text,
        "translated_text": english_text,
        "sentiment": sentiment,
        "summary": summary
    }
