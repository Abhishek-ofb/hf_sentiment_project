# sentiment_service.py
import threading
from transformers import pipeline
import statistics

_sentiment_analyzer = None
_sentiment_lock = threading.Lock()

def initialize_sentiment_service():
    global _sentiment_analyzer
    with _sentiment_lock:
        if _sentiment_analyzer is None:
            print("Loading sentiment model...")
            _sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )

def analyze_sentiment(english_text: str) -> dict:
    initialize_sentiment_service()

    # Break into ~500 word chunks for better accuracy
    words = english_text.split()
    chunk_size = 500
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    results = _sentiment_analyzer(chunks)

    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]

    # Pick majority label
    final_label = max(set(labels), key=labels.count)
    avg_score = statistics.mean(scores)

    return {"label": final_label, "score": avg_score}
