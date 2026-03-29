import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score


# Load dataset
# df = pd.read_csv("data/nlp_training_dataset.csv")
df = df = pd.read_csv("data/nlp_training_dataset.csv")





# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['disease'])

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SymptomDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True)
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SymptomDataset(train_texts, train_labels)
val_dataset = SymptomDataset(val_texts, val_labels)

# Model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(le.classes_)
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./bert_results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=10
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained("model/bert_model")
tokenizer.save_pretrained("model/bert_model")
import pickle

pickle.dump(le, open("model/label_encoder.pkl", "wb"))

print("BERT training completed and model saved.")



# ================= EVALUATION =================

predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(axis=1)

acc = accuracy_score(val_labels, preds)
print(f"\nBERT Validation Accuracy: {acc*100:.2f}%")
