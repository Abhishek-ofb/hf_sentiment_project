from transformers import pipeline

_translator = None


def initialize_translator():
    global _translator
    if _translator is None:
        print("Loading translation model (Hinglish → English)...")
        _translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")


def translate_to_english(text: str) -> str:
    if _translator is None:
        initialize_translator()
    try:
        out = _translator(text)
        return out[0]["translation_text"]
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # fallback
