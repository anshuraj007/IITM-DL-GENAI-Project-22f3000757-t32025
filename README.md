# Project Title: Sentiment Analysis
# Name: Anshu Raj
# Roll Number: 22f3000757

# 🧠 Emotion Classification using DeBERTa & RoBERTa

## 📌 Abstract

This project focuses on building a **robust emotion classification system** capable of identifying multiple emotional states expressed in text. Leveraging **large-scale Transformer-based architectures** such as *DeBERTa-v3 Large* and *RoBERTa Large*, the model is fine-tuned on a labeled dataset of social-media style text. The aim is to handle noisy, informal, and context-heavy input efficiently while producing accurate multi-label predictions. Advanced techniques such as **mean pooling**, **layer normalization**, **mixed-precision training**, and **learning rate warmup** are used to stabilize and enhance training.

---

## 🔧 Models Used

### **1. DeBERTaV3-Large**

* Architecture: Disentangled attention + Enhanced mask decoder
* Tokenizer: **DeBERTaV2TokenizerFast**
* Hidden Size: 1024
* Strengths:

  * Better handling of word content vs. positional information
  * Superior performance on low-data and noisy text settings

### **2. RoBERTa-Large**

* Architecture: Optimized version of BERT with dynamic masking & large-scale training
* Tokenizer: **RobertaTokenizerFast (Byte-Level BPE)**
* Strengths:

  * Highly robust on informal text
  * Byte-level subword handling → emojis, slang, rare words

### **3. Custom Classifier Head**

* Mean-pooled final hidden states
* LayerNorm + Dropout
* Linear classifier for 5 emotional labels

---

## 📁 Repository Structure

```
📦 IITM-DL-GENAI-PROJECT-22f3000757-T3025
│
├── 📂 Notebooks
│     ├── dl-22f3000757_Base_Model.ipynb
│     ├── dl-22f3000757_dummy_submission.ipynb
│     ├── dl-22f3000757_DeBERTa-Large-Model.ipynb
│     ├── dl-22f3000757_EDA.ipynb
│     ├── dl-22f3000757_milestone_1.ipynb
│     ├── dl-22f3000757_milestone_4.ipynb
│     └── dl-22f3000757_RoBERTa-Large-Model.ipynb
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitattributes 
│
├── 📂 Data
│   ├── train.csv
│   └── test.csv
│
│
└── 📂 Deployment-Codes
    ├── app.py
    ├── model.py
    ├── Dockerfile
    ├── requirements.txt    

```

---

## 📚 References

* DeBERTaV3 Paper: [https://arxiv.org/abs/2111.09543](https://arxiv.org/abs/2111.09543)
* RoBERTa Paper: [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
* HuggingFace Model Card (DeBERTa): [https://huggingface.co/microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)
* HuggingFace Model Card (RoBERTa): [https://huggingface.co/roberta-large](https://huggingface.co/roberta-large)
* PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

---

---
# Deployment

* The model is deployed on HuggingFace.
* Link to the deployed mode: <a>https://huggingface.co/spaces/rajanshu22f3/deploy-sentiment-analysis-app</a>
---


## ✨ Closing Note

This project demonstrates how combining **state-of-the-art Transformer architectures** with a **clean training pipeline** can produce a highly reliable emotion classification system. The repository is structured for clarity, allowing easy extension into areas like explainability, quantization, deployment (API/Streamlit), and model ensembling.
