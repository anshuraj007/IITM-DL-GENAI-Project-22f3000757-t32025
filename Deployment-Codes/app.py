import os
import torch
from model import EmotionClassifier
from transformers import DebertaV2Tokenizer
import requests

# ---------------- CONFIG ----------------
MODEL_NAME = "microsoft/deberta-v3-large"
MODEL_DRIVE_URL = "YOUR_DRIVE_DOWNLOAD_LINK_HERE"  # <- replace with direct file link
MODEL_PATH = "./best_model_deberta.pt"
MAX_LEN = 64

# ---------------- DOWNLOAD MODEL IF NOT EXISTS ----------------
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Drive... ⬇️")
    r = requests.get(MODEL_DRIVE_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        total = 0
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
                print(f"\rDownloaded {total//1024//1024} MB", end="")
    print("\n✅ Model download complete!")

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    import re, emoji
    from ftfy import fix_text
    text = fix_text(text)
    text = re.sub(r'http\S+|www.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- LOAD TOKENIZER ----------------
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, local_files_only=True, use_fast=True)

# ---------------- LOAD MODEL ----------------
def load_model():
    try:
        model = EmotionClassifier(model_name=MODEL_NAME)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        print("❌ Error loading model:", e)
        return None

model = load_model()
print("✅ Model loaded successfully!")

# ---------------- INFERENCE ----------------
sample_text = "I am really tensed about the results!"
cleaned_text = clean_text(sample_text)

encoded = tokenizer(
    cleaned_text,
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

print("Predicted Emotion Probabilities:")
for emo, score in zip(emotions, probs):
    print(f"{emo}: {score:.4f}")