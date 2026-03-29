import torch
import joblib
import pandas as pd

from normalizer import normalize_text
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder


# =========================
# LOAD PRECAUTIONS DATASET
# =========================
precautions_df = pd.read_csv("data/precautions.csv")

# Clean precautions disease column once
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

# Clean labels before encoder
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

    # Structured input → SVM
    if "," in user_input:
        prediction = predict_svm(user_input)

        print("\nPredicted Disease (SVM):", prediction.title())
        show_precautions(prediction)

    # Sentence input → BERT
    else:
        predictions = predict_bert(user_input)

        print("\nTop Possible Diseases:")
        for disease, conf in predictions:
            print(f"{disease.title()}  →  {conf:.2f}%")

        # Show precautions for most probable disease
        show_precautions(predictions[0][0])

        print("\nNote: Please provide key symptoms for more accurate prediction.")
        print("\nDisclaimer: This system provides preliminary guidance and is not a substitute for professional medical advice.")
