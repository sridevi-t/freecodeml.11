import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
df = pd.read_csv(url)

df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=42)

train_labels = train_dataset.pop("expenses")
test_labels = test_dataset.pop("expenses")

scaler = StandardScaler()
train_dataset_scaled = scaler.fit_transform(train_dataset)
test_dataset_scaled = scaler.transform(test_dataset)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[train_dataset.shape[1]]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

history = model.fit(train_dataset_scaled, train_labels, epochs=100, validation_split=0.2, verbose=0)

loss, mae = model.evaluate(test_dataset_scaled, test_labels, verbose=1)
print(f"\n✅ Mean Absolute Error on test set: ${mae:.2f}")

import matplotlib.pyplot as plt

predicted = model.predict(test_dataset_scaled).flatten()

plt.figure(figsize=(10,6))
plt.scatter(test_labels, predicted, alpha=0.6)
plt.plot([0, max(test_labels)], [0, max(test_labels)], 'r')
plt.xlabel('True Expenses')
plt.ylabel('Predicted Expenses')
plt.title('Actual vs Predicted Healthcare Costs')
plt.grid(True)
plt.show()
