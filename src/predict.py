import torch
import joblib
import pandas as pd

from normalizer import normalize_text
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import fuzz


# =========================
# LOAD PRECAUTIONS DATASET
# =========================
precautions_df = pd.read_csv("data/precautions.csv")
precautions_df['Disease'] = precautions_df['Disease'].str.strip().str.lower()


def show_precautions(disease):
    disease = disease.strip().lower()

    row = precautions_df.loc[precautions_df['Disease'] == disease]

    if not row.empty:
        print("\nPrecautions:")
        print(row['Precautions'].values[0])
    else:
        print("\nPrecautions: Not available")


# =========================
# LOAD BERT MODEL
# =========================
df = pd.read_csv("data/nlp_training_dataset.csv")
df['disease'] = df['disease'].str.strip().str.lower()

le = LabelEncoder()
le.fit(df['disease'])

model_path = "model/bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertForSequenceClassification.from_pretrained(model_path)
bert_model.eval()


# =========================
# LOAD SVM MODEL
# =========================
svm_model = joblib.load("model/svm_model.pkl")
all_symptoms = joblib.load("model/symptoms_list.pkl")


# =========================
# INPUT VALIDATION (FUZZY)
# =========================
def is_valid_input(text):
    text = normalize_text(text)

    match_count = 0

    for symptom in all_symptoms:
        score = fuzz.partial_ratio(symptom, text)

        if score > 80:
            match_count += 1

        if match_count >= 1:
            return True

    return False


# =========================
# AUTO CORRECTION (NEW 🔥)
# =========================
def correct_input(text):
    text = normalize_text(text)
    words = text.split()

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
# BERT PREDICTION (TOP 3)
# =========================
def predict_bert(text):
    text = normalize_text(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    top_probs, top_indices = torch.topk(probs, 3)

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        disease = le.inverse_transform([idx.item()])[0]
        confidence = prob.item() * 100
        results.append((disease, confidence))

    return results


# =========================
# SVM PREDICTION
# =========================
def symptoms_to_vector(symptoms):
    vector = [0] * len(all_symptoms)

    for s in symptoms:
        s = s.strip().lower()
        if s in all_symptoms:
            idx = all_symptoms.index(s)
            vector[idx] = 1

    return [vector]


def predict_svm(text):
    symptoms = text.split(",")
    vec = symptoms_to_vector(symptoms)
    prediction = svm_model.predict(vec)[0]
    return prediction


# =========================
# HYBRID LOOP
# =========================
print("\nHealthcare Chatbot Ready")
print("Type symptoms (comma separated) OR a sentence")
print("Type 'exit' to quit\n")

while True:
    user_input = input("Enter symptoms: ")

    if user_input.lower() == "exit":
        break

    # =====================
    # STRUCTURED INPUT → SVM
    # =====================
    if "," in user_input:

        if not is_valid_input(user_input):
            print("\nPlease enter valid medical symptoms.")
            continue

        prediction = predict_svm(user_input)

        print("\nPredicted Disease (SVM):", prediction.title())
        show_precautions(prediction)

    # =====================
    # SENTENCE INPUT → BERT
    # =====================
    else:

        if not is_valid_input(user_input):
            print("\nPlease enter valid medical symptoms (e.g., fever, headache, rash).")
            continue

        # 🔥 AUTO CORRECTION
        corrected_text = correct_input(user_input)

        print("\nInterpreted symptoms:", corrected_text)

        predictions = predict_bert(corrected_text)

        if predictions[0][1] < 40:
            print("\nSymptoms not sufficient or unclear. Please provide more specific symptoms.")
            continue

        print("\nTop Possible Diseases:")
        for disease, conf in predictions:
            print(f"{disease.title()}  →  {conf:.2f}%")

        show_precautions(predictions[0][0])

        print("\nNote: Please provide key symptoms for more accurate prediction.")
        print("\nDisclaimer: This system provides preliminary guidance and is not a substitute for professional medical advice.")