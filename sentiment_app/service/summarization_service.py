# summarization_service.py
from transformers import pipeline
import threading

_summarizer = None
_summarizer_lock = threading.Lock()

def initialize_summarizer():
    global _summarizer
    with _summarizer_lock:
        if _summarizer is None:
            print("Loading summarizer...")
            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_summarizer():
    if _summarizer is None:
        initialize_summarizer()
    return _summarizer

def summarize_call(english_text: str) -> str:
    summarizer = get_summarizer()

    prompt_text = (
        "Customer support call transcript:\n"
        f"{english_text}\n\n"
        "Provide a **short summary in English** highlighting:\n"
        "- Customer's issue\n"
        "- Emotional tone\n"
        "- Resolution steps (if any)"
    )

    input_length = len(english_text.split())
    max_len = min(120, int(input_length * 0.6))  # max 60% of transcript
    min_len = max(30, int(input_length * 0.2))   # min 20% of transcript

    result = summarizer(
        prompt_text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )
    return result[0]["summary_text"]
