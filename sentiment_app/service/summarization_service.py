from transformers import pipeline
import threading

_summarizer = None
_summarizer_lock = threading.Lock()


def initialize_summarizer():
    """Initialize the summarizer pipeline once (thread-safe)."""
    global _summarizer
    with _summarizer_lock:
        if _summarizer is None:
            print("Loading summarizer...")
            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def get_summarizer():
    """Ensure summarizer is loaded before use."""
    if _summarizer is None:
        initialize_summarizer()
    return _summarizer


def summarize_call(english_text: str) -> str:
    summarizer = get_summarizer()

    # Use a clear but minimal summarization request
    prompt_text = (
        f"Customer support call transcript:\n\n{english_text}\n\n"
        "Provide a short summary in English highlighting the customer's issue, emotional tone, and any resolution."
    )

    input_length = len(english_text.split())
    max_len = min(200, int(input_length * 0.6))
    min_len = min(60, int(input_length * 0.2))

    result = summarizer(
        prompt_text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )
    return result[0]["summary_text"]

