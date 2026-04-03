from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
import os
from huggingface_hub import login, hf_hub_download
from rapidfuzz import fuzz

app = FastAPI()

# 🔐 Safe login
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

# 🔥 Model name
model_name = "siddharth-1201/healthcare-chatbot-model"

# 🔹 Global variables
tokenizer = None
bert_model = None
label_encoder = None
all_symptoms = None   # 🔥 NEW

# 🔧 Reduce CPU usage
torch.set_num_threads(1)


# =========================
# LOAD MODEL
# =========================
def load_model():
    global tokenizer, bert_model, label_encoder, all_symptoms

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if bert_model is None:
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    if label_encoder is None:
        label_path = hf_hub_download(
            repo_id=model_name,
            filename="label_encoder.pkl"
        )
        label_encoder = pickle.load(open(label_path, "rb"))

    # 🔥 Load symptoms list (IMPORTANT)
    if all_symptoms is None:
        try:
            sym_path = hf_hub_download(
                repo_id=model_name,
                filename="symptoms_list.pkl"
            )
            all_symptoms = pickle.load(open(sym_path, "rb"))
        except:
            all_symptoms = []


# =========================
# INPUT VALIDATION
# =========================
def is_valid_input(text):
    text = text.lower()

    match_count = 0

    for symptom in all_symptoms:
        score = fuzz.partial_ratio(symptom, text)

        if score > 80:
            match_count += 1

        if match_count >= 1:
            return True

    return False


# =========================
# AUTO CORRECTION
# =========================
def correct_input(text):
    words = text.lower().split()
    corrected_words = []

    for word in words:
        best_match = word
        best_score = 0

        for symptom in all_symptoms:
            score = fuzz.ratio(word, symptom)

            if score > best_score:
                best_score = score
                best_match = symptom

        if best_score > 80:
            corrected_words.append(best_match)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "BERT Chatbot API running 🚀"}


@app.post("/predict")
def predict(data: dict):
    text = data.get("message")

    if not text:
        return {"error": "No message provided"}

    load_model()

    # 🔥 INPUT VALIDATION
    if not is_valid_input(text):
        return {
            "error": "Please enter valid medical symptoms"
        }

    # 🔥 AUTO CORRECTION
    corrected_text = correct_input(text)

    inputs = tokenizer(corrected_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    top3 = torch.topk(probs, 3)

    indices = top3.indices[0].tolist()
    scores = top3.values[0].tolist()

    diseases = label_encoder.inverse_transform(indices)

    # 🔥 CONFIDENCE CHECK
    if scores[0] * 100 < 40:
        return {
            "error": "Symptoms unclear. Please provide more specific symptoms"
        }

    results = []
    for i in range(3):
        results.append({
            "disease": diseases[i].title(),
            "confidence": round(scores[i] * 100, 2)
        })

    return {
        "input": text,
        "interpreted_input": corrected_text,
        "top_3_predictions": results
    }