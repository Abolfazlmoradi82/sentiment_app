# app/model.py
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("tabularisai/multilingual-sentiment-analysis")

def analyze_sentiment(text: str):
    """Analyze the sentiment of a given text using a multilingual transformer model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Disable gradient calculation (faster inference)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label_id = torch.argmax(probs, dim=1).item()
        label = model.config.id2label[label_id]
        score = probs[0][label_id].item()

    return {"sentiment": label, "confidence": round(score, 3)}