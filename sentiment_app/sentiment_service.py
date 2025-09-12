import os, threading

MODE = os.getenv("SENTIMENT_MODE", "local")
MODEL_NAME = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
HF_API_URL = os.getenv("HF_API_URL")
HF_API_KEY = os.getenv("HF_API_KEY")

_pipeline = None
_pipeline_lock = threading.Lock()

def initialize_service():
    global _pipeline
    if _pipeline is not None:
        return
    with _pipeline_lock:
        if _pipeline is not None:
            return
        if MODE == "local":
            from transformers import pipeline
            _pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
        elif MODE == "hf_api":
            _pipeline = "hf_api"
        else:
            raise RuntimeError(f"Unknown SENTIMENT_MODE: {MODE}")

def analyze_text(text: str):
    if _pipeline is None:
        initialize_service()
    if MODE == "local":
        results = _pipeline(text, truncation=True)
        top = results[0]
        return {"label": top.get("label"), "score": float(top.get("score")), "raw": results}
    else:
        import requests
        if not HF_API_URL or not HF_API_KEY:
            raise RuntimeError("HF_API_URL and HF_API_KEY must be set for hf_api mode")
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": text}
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return {"label": data[0].get("label"), "score": float(data[0].get("score")), "raw": data}
