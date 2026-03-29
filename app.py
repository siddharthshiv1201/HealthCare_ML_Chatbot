from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
import os
from huggingface_hub import login, hf_hub_download

app = FastAPI()

# 🔐 Safe login (only if token exists)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

# 🔥 Hugging Face model
model_name = "siddharth-1201/healthcare-chatbot-model"

# 🔹 Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 🔥 Download label_encoder from HF
label_path = hf_hub_download(
    repo_id=model_name,
    filename="label_encoder.pkl"
)

label_encoder = pickle.load(open(label_path, "rb"))

@app.get("/")
def home():
    return {"message": "BERT Chatbot API running 🚀"}

@app.post("/predict")
def predict(data: dict):
    text = data.get("message")

    if not text:
        return {"error": "No message provided"}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    top3 = torch.topk(probs, 3)

    indices = top3.indices[0].tolist()
    scores = top3.values[0].tolist()

    diseases = label_encoder.inverse_transform(indices)

    results = []
    for i in range(3):
        results.append({
            "disease": diseases[i],
            "confidence": round(scores[i] * 100, 2)
        })

    return {
        "input": text,
        "top_3_predictions": results
    }