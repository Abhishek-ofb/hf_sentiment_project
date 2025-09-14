# translation_service.py
from transformers import pipeline
import re

_translator = None

def initialize_translator():
    global _translator
    if _translator is None:
        print("Loading translation model (Hindi → English)...")
        _translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")

def translate_to_english(text: str) -> str:
    if not text.strip():
        return ""

    # Always translate if contains Hindi/Hinglish words (not just ASCII)
    # crude check: if any Hindi Unicode character
    if re.search(r'[\u0900-\u097F]', text) or any(word.lower() in ["kya","ho","rahi","hai","bhai","main","phir"] for word in text.split()):
        initialize_translator()
        try:
            out = _translator(text, max_length=512)
            return out[0]["translation_text"]
        except Exception as e:
            print(f"Translation failed: {e}")
            return text
    else:
        # Likely already English
        return text
