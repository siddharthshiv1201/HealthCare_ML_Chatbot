import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib
import os

# ================================
# STEP 1: LOAD DATASET
# ================================
df = pd.read_csv("data/dataset.csv")

print("Original shape:", df.shape)
print(df.head())

# ================================
# STEP 2: SEPARATE LABEL
# ================================
y = df['Disease']

# All symptom columns
symptom_cols = df.columns[1:]

# ================================
# STEP 3: CREATE SYMPTOM LIST
# ================================
df['symptom_list'] = df[symptom_cols].apply(
    lambda row: [str(sym).strip().lower() for sym in row if pd.notna(sym)],
    axis=1
)

# ================================
# STEP 4: REMOVE DUPLICATES
# ================================
df = df.drop_duplicates(subset=['symptom_list'])

print("\nAfter removing duplicate symptom patterns:", df.shape)

# Update labels after removing duplicates
y = df['Disease']

# ================================
# STEP 5: BINARY SYMPTOM ENCODING
# ================================
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['symptom_list'])

print("Symptom Vector Shape:", X.shape)
print("Total unique symptoms:", len(mlb.classes_))

# Save symptom list for prediction time
all_symptoms = list(mlb.classes_)

# ================================
# STEP 6: TRAIN–TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# STEP 7: MODEL TRAINING
# ================================
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM": LinearSVC()
}

print("\nModel Accuracies:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc*100:.2f}%")

# ================================
# STEP 8: SAVE SVM MODEL
# ================================
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

joblib.dump(svm_model, "model/svm_model.pkl")
joblib.dump(all_symptoms, "model/symptoms_list.pkl")

print("\nSVM model and symptom list saved successfully.")
