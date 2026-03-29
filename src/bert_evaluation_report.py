import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset

# ================= LOAD DATA =================
df = pd.read_csv("data/nlp_training_dataset.csv")



le = LabelEncoder()
df['label'] = le.fit_transform(df['disease'])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# ================= LOAD MODEL =================
model_path = "model/bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# ================= DATASET CLASS =================
class EvalDataset(Dataset):
    def __init__(self, texts):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

val_dataset = EvalDataset(val_texts)

# ================= PREDICTIONS =================
preds = []
with torch.no_grad():
    for i in range(len(val_dataset)):
        inputs = {k: v.unsqueeze(0) for k, v in val_dataset[i].items()}
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        preds.append(pred)

# ================= METRICS =================
accuracy = accuracy_score(val_labels, preds)
print(f"\n✅ BERT Test Accuracy: {accuracy*100:.2f}%")

report = classification_report(val_labels, preds, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\n📋 Classification Report:\n")
print(report_df)

# Save report to CSV
report_df.to_csv("bert_classification_report.csv")

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(val_labels, preds)

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("bert_confusion_matrix.png")
plt.close()

# ================= F1 SCORE BAR CHART =================
f1_scores = report_df[:-3]['f1-score']  # remove avg rows

plt.figure(figsize=(12,6))
f1_scores.plot(kind='bar')
plt.title("F1 Score per Disease")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("bert_f1_scores.png")
plt.close()

print("\n📊 Charts saved:")
print(" - bert_confusion_matrix.png")
print(" - bert_f1_scores.png")
print(" - bert_classification_report.csv")
