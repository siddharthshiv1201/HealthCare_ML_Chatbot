import streamlit as st
import torch
import joblib
import pandas as pd

from normalizer import normalize_text
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Healthcare Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Healthcare Chatbot")
st.write("Enter symptoms to predict possible diseases.")


# ----------------------------
# LOAD PRECAUTIONS
# ----------------------------
precautions_df = pd.read_csv("data/precautions.csv")
precautions_df['Disease'] = precautions_df['Disease'].str.strip().str.lower()


def show_precautions(disease):
    disease = disease.strip().lower()
    row = precautions_df.loc[precautions_df['Disease'] == disease]

    if not row.empty:
        return row['Precautions'].values[0]
    return "Precautions not available"


# ----------------------------
# LOAD DATASET FOR LABELS
# ----------------------------
df = pd.read_csv("data/nlp_training_dataset.csv")
df['disease'] = df['disease'].str.strip().str.lower()

le = LabelEncoder()
le.fit(df['disease'])


# ----------------------------
# LOAD MODELS
# ----------------------------
model_path = "model/bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertForSequenceClassification.from_pretrained(model_path)
bert_model.eval()

svm_model = joblib.load("model/svm_model.pkl")
all_symptoms = joblib.load("model/symptoms_list.pkl")


# ----------------------------
# BERT PREDICTION
# ----------------------------
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


# ----------------------------
# SVM PREDICTION
# ----------------------------
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


# ----------------------------
# USER INPUT
# ----------------------------
user_input = st.text_input("Enter symptoms (comma separated OR sentence):")


if st.button("Predict"):

    if user_input:

        # Structured input → SVM
        if "," in user_input:
            disease = predict_svm(user_input)

            st.success(f"Predicted Disease (SVM): {disease.title()}")

            st.subheader("Precautions")
            st.info(show_precautions(disease))

        # Sentence input → BERT
        else:
            predictions = predict_bert(user_input)

            st.subheader("Top Possible Diseases")

            for disease, conf in predictions:
                st.write(f"**{disease.title()}** — {conf:.2f}%")

            st.subheader("Precautions")
            st.info(show_precautions(predictions[0][0]))

        st.warning("Note: Please provide key symptoms for more accurate prediction.")
        st.caption("Disclaimer: This system provides preliminary guidance and is not a substitute for professional medical advice.")
