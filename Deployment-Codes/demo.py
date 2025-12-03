import torch
import os
from transformers import DebertaV2Tokenizer
from model import EmotionClassifier


MODEL_NAME = "microsoft/deberta-v3-large"
MODEL_PATH = "./best_model_deberta.pt"
MAX_LEN = 64

print(f"\n📌 Loading model from: {MODEL_PATH}\n")


# -------------------- CLEAN TEXT FUNCTION --------------------
def clean_text(text):
    import re, emoji
    from ftfy import fix_text

    text = fix_text(text)
    text = re.sub(r"http\S+|www.\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = emoji.demojize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------- LOAD TOKENIZER --------------------
tokenizer = DebertaV2Tokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)  # FIRST TIME: allow download


# -------------------- LOAD MODEL --------------------
def load_model():
    try:
        model = EmotionClassifier(model_name=MODEL_NAME)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print("❌ Error loading model:", e)
        return None


model = load_model()


# -------------------- RUN INFERENCE --------------------
sample_text = "I am really tensed about the results!"
cleaned_text = clean_text(sample_text)

encoded = tokenizer(
    cleaned_text,
    padding="max_length",
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

print("\n🔮 Predicted Emotion Probabilities:")
for emo, score in zip(emotions, probs):
    print(f"{emo}: {score:.4f}")
