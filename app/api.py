# app\api.py
from fastapi import FastAPI
from pydantic import BaseModel
from .model import analyze_sentiment
from .utils import clean_text

app = FastAPI(title="Sentiment Analysis API")

class InputText(BaseModel):
    text: str

@app.post("/analyze")
def analyze(data: InputText):
    """Receive text input, clean it, and return sentiment prediction."""
    text = clean_text(data.text)
    result = analyze_sentiment(text)
    return result
