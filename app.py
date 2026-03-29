from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
from huggingface_hub import login

app = FastAPI()

# 🔐 (Optional but recommended if private model)
# login("your_token_here")

# 🔥 Hugging Face model
model_name = "siddharth-1201/healthcare-chatbot-model"

# 🔹 Load model from HF
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 🔹 Load LabelEncoder (local hi rahega)
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "BERT Chatbot API running 🚀"}

@app.post("/predict")
def predict(data: dict):
    text = data.get("message")

    if not text:
        return {"error": "No message provided"}

    # 🔹 Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # 🔹 Prediction
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # 🔥 Softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    # 🔥 Top 3 predictions (same as tera logic)
    top3 = torch.topk(probs, 3)

    indices = top3.indices[0].tolist()
    scores = top3.values[0].tolist()

    # 🔥 Convert to disease names
    diseases = label_encoder.inverse_transform(indices)

    # 🔥 Response format (unchanged)
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