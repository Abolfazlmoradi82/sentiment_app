# app\utils.py
import re


def clean_text(text: str) -> str:
    """Clean up extra spaces and unnecessary characters."""
    text = re.sub(r"\s+", " ", text).strip()    # remove URLs
    text = re.sub(r"[^A-Za-zÀ-ÿ0-9\s.,!?']+", " ", text)    # Remove special characters (keep most letters)
    text = re.sub(r"\s+", " ", text).strip()     # normalize spaces
    return text

def format_output(result: dict) -> str:
    """Format the model output for display."""
    sentiment = result.get("sentiment", "Unknown")
    confidence = result.get("confidence", 0.0)
    return f"Sentiment: {sentiment} | Confidence: {confidence:.2f}"