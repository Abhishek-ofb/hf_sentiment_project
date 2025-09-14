from transformers import pipeline

_sentiment_pipeline = None


def initialize_sentiment():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("Loading customer support sentiment model...")
        _sentiment_pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )


def analyze_sentiment(text: str) -> dict:
    if _sentiment_pipeline is None:
        initialize_sentiment()
    result = _sentiment_pipeline(text, truncation=True)[0]

    # Map to business-friendly labels
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    return {
        "label": label_map.get(result["label"], result["label"]),
        "score": float(result["score"])
    }
