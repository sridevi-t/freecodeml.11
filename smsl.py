!wget -q https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget -q https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential

train_df = pd.read_csv("train-data.tsv", sep='\t', header=None, names=['label', 'message'])
valid_df = pd.read_csv("valid-data.tsv", sep='\t', header=None, names=['label', 'message'])

label_map = {'ham': 0, 'spam': 1}
train_df['label'] = train_df['label'].map(label_map)
valid_df['label'] = valid_df['label'].map(label_map)

max_tokens = 10000
seq_length = 100

vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=seq_length)
vectorizer.adapt(train_df['message'])

model = Sequential([
    vectorizer,
    Embedding(max_tokens, 64, mask_zero=True),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    train_df['message'],
    train_df['label'],
    validation_data=(valid_df['message'], valid_df['label']),
    epochs=5,
    verbose=1
)
def predict_message(message):
    prob = float(model.predict([message])[0][0])
    label = 'spam' if prob > 0.5 else 'ham'
    return [prob, label]

examples = [
    "Hey, are you free today?",
    "Win a brand new car! Call now!",
    "Don't forget the meeting at 5 PM",
    "Congratulations! You have won a $1000 gift card!"
]

for text in examples:
    print(f"Message: {text}\nPrediction: {predict_message(text)}\n")
