import streamlit as st
import torch
import re
import emoji
from ftfy import fix_text
from transformers import DebertaV2TokenizerFast
from Model.model import EmotionClassifier   # ← import your model class

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 256
MODEL_PATH = "../best_deberta_model.pt"   # ← correct relative path from App/app.py

# ---------------------- TEXT CLEANING ----------------------
def clean_text(text):
    text = fix_text(text)
    text = re.sub(r'http\S+|www.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------- LOAD MODEL -------------------------
tokenizer = DebertaV2TokenizerFast.from_pretrained(MODEL_NAME)

@st.cache_resource
def load_model():
    model = EmotionClassifier(model_name=MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()
st.success("✅ Model Loaded Successfully!")

# ---------------------- STREAMLIT UI ------------------------
st.title("🔍 Emotion Detection (DeBERTa-v3-Large)")
st.write("Enter your text below, and the model will predict emotions.")

user_input = st.text_area("Enter text")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Text cannot be empty!")
    else:
        cleaned = clean_text(user_input)

        encoded = tokenizer(
            cleaned,
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )

        probs = torch.sigmoid(logits).numpy()[0]

        emotions = ["anger", "fear", "joy", "sadness", "surprise"]

        st.subheader("🧠 Predictions")
        for emo, score in zip(emotions, probs):
            st.write(f"**{emo}**: {score:.4f}")

        st.subheader("📊 Emotion Scores")
        st.bar_chart({emo: probs[i] for i, emo in enumerate(emotions)})
