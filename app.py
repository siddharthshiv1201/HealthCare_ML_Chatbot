from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
import os
from huggingface_hub import login, hf_hub_download

app = FastAPI()

# 🔐 Safe login
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

# 🔥 Model name
model_name = "siddharth-1201/healthcare-chatbot-model"

# 🔹 Global variables (lazy loading)
tokenizer = None
bert_model = None
label_encoder = None

# 🔧 Reduce CPU usage
torch.set_num_threads(1)

def load_model():
    global tokenizer, bert_model, label_encoder

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if bert_model is None:
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,        # 🔥 RAM reduce
            low_cpu_mem_usage=True            # 🔥 RAM reduce
        )

    if label_encoder is None:
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

    # 🔥 Load model only when needed
    load_model()

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