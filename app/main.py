import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("tabularisai/multilingual-sentiment-analysis")

st.title("🌍 Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment (multilingual supported!)")

# Input text
text = st.text_area("Your text here:")

if st.button("Analyze"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label_id = torch.argmax(probs, dim=1).item()
            label = model.config.id2label[label_id]
            score = probs[0][label_id].item()

        st.success(f"**Sentiment:** {label}")
        st.info(f"**Confidence:** {score:.2f}")
    else:
        st.warning("Please enter some text first!")