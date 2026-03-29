import pandas as pd
import random

df = pd.read_csv("data/dataset.csv")
symptom_cols = df.columns[1:]

df['symptom_list'] = df[symptom_cols].apply(
    lambda row: [str(sym).strip().replace('_', ' ') for sym in row if pd.notna(sym)],
    axis=1
)

df = df.drop_duplicates(subset=['symptom_list'])

# Key symptoms per disease (VERY IMPORTANT)
KEY_SYMPTOMS = {
    "Migraine": "sensitivity to light",
    "Malaria": "shivering",
    "Dengue": "joint pain",
    "Fungal infection": "skin rash",
    "Chronic cholestasis": "yellowish skin",
}

templates = [
    "I am experiencing {}.",
    "I have {}.",
    "I am suffering from {}.",
    "Symptoms include {}.",
    "I feel {}."
]

def build_sentence(symptoms, disease):
    key = KEY_SYMPTOMS.get(disease, symptoms[0])

    others = [s for s in symptoms if s != key]
    k = random.randint(1, min(3, len(others))) if others else 0
    chosen = random.sample(others, k) if k > 0 else []

    final = [key] + chosen

    text = ", ".join(final[:-1]) + " and " + final[-1] if len(final) > 1 else final[0]
    template = random.choice(templates)

    return template.format(text)

rows = []

for _, row in df.iterrows():
    symptoms = row['symptom_list']
    disease = row['Disease']

    for _ in range(20):
        sentence = build_sentence(symptoms, disease)
        rows.append([sentence, disease])

nlp_df = pd.DataFrame(rows, columns=['text', 'disease'])
nlp_df = nlp_df.drop_duplicates(subset=['text'])

nlp_df.to_csv("data/nlp_training_dataset.csv", index=False)
print("New improved NLP dataset created:", nlp_df.shape)
