import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM

# STEP 1: Load dataset
df = pd.read_csv("data/nlp_training_dataset.csv")

texts = df['text']
labels = df['disease']

# STEP 2: Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# STEP 3: Tokenize text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=50)

# STEP 4: Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_classes = len(set(y))

# ================= CNN MODEL =================
cnn_model = Sequential([
    Embedding(5000, 64, input_length=50),
    Conv1D(64, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nTraining CNN...")
cnn_model.fit(X_train, y_train, epochs=3, verbose=1)

loss, acc = cnn_model.evaluate(X_test, y_test)
print("CNN Accuracy:", acc)


# ================= RNN MODEL =================
rnn_model = Sequential([
    Embedding(5000, 64, input_length=50),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

rnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nTraining RNN...")
rnn_model.fit(X_train, y_train, epochs=3, verbose=1)

loss, acc = rnn_model.evaluate(X_test, y_test)
print("RNN Accuracy:", acc)
