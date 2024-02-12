import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.data import Dataset
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
strategy = tf.distribute.MirroredStrategy()

def load_data(file_path):
    texts, labels = [], []
    positive, negative = 0, 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                review = json.loads(line)
                if 'reviewText' in review and 'overall' in review:
                    texts.append(review['reviewText'])
                    if review['overall'] > 3:
                        labels.append(1)
                        positive += 1
                    else:
                        labels.append(0)
                        negative += 1
            except json.JSONDecodeError:
                print(f"Warnung: Eine Zeile konnte nicht in JSON umgewandelt werden und wird übersprungen. Inhalt: {line}")
    return texts, labels, positive, negative


def create_dataset(texts, labels, batch_size=128):
    tokenizer = Tokenizer(num_words=10000)
    sequences = tokenizer.texts_to_sequences(texts)
    max_length = 20
    data = pad_sequences(sequences, maxlen=max_length)
    data_array = np.array(data)
    labels_array = np.array(labels)
    dataset = Dataset.from_tensor_slices((data_array, labels_array))
    dataset = dataset.cache().shuffle(len(data_array)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def create_model():
    with strategy.scope():
        model = Sequential([
            Embedding(10000, 16, input_length=20),
            GlobalAveragePooling1D(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


print(tf.__version__)

train_texts, train_labels, train_pos, train_neg = load_data('train_reviews.json')
test_texts, test_labels, test_pos, test_neg = load_data('test_reviews.json')

print(f"Training data: {len(train_texts)} reviews ({train_pos} positive, {train_neg} negative)")
print(f"Testing data: {len(test_texts)} reviews ({test_pos} positive, {test_neg} negative)")

batch_size = 128
# Erstellen des Trainingsdatensatzes
train_dataset = create_dataset(train_texts, train_labels, batch_size=batch_size)
# Erstellen des Testdatensatzes
test_dataset = create_dataset(test_texts, test_labels, batch_size=batch_size)


model_path = 'mein_rezensionsmodell.h5'
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Training new model...")
    model = create_model()
    model.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[ModelCheckpoint("model_checkpoint.h5", monitor='val_loss', save_best_only=True, verbose=1), EarlyStopping(monitor='val_loss', patience=5)])
    model.save("final_model.h5")

plot_model(model, to_file='model_diagram.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=300)

# Evaluierung mit dem test_dataset
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Vorhersage mit dem test_dataset
predicted_probabilities = model.predict(test_dataset)
predicted_labels = (predicted_probabilities >= 0.6).astype(int)

# Für eine detaillierte Auswertung müssen Sie tatsächliche Labels extrahieren
# Dies erfordert das Iterieren über test_dataset, um Labels zu sammeln
actual_labels = []
for _, labels in test_dataset.unbatch().batch(1):
    actual_labels.extend(labels.numpy())

# Umwandlung in Numpy-Array für einfache Handhabung
actual_labels = np.array(actual_labels)

# Berechnung der Korrektheit der Vorhersagen
correct_predictions = (predicted_labels.flatten() == actual_labels)
incorrect_predictions = ~correct_predictions

# Zählung korrekter und inkorrekter Vorhersagen
correct_counts = np.sum(correct_predictions)
incorrect_counts = np.sum(incorrect_predictions)

print("Correct predictions count:", correct_counts)
print("Incorrect predictions count:", incorrect_counts)

# Generate a bar chart to represent the model's performance
labels = ['Negative Reviews', 'Positive Reviews']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, correct_counts, width, label='Correctly Predicted', color='green')
rects2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrectly Predicted', color='red')

ax.set_xlabel('Review Type')
ax.set_ylabel('Number of Predictions')
ax.set_title('Model Performance on Test Data by Review Type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.show()

# Generate a histogram to visualize the predicted probabilities
plt.figure(figsize=(10, 5))
plt.hist(predicted_probabilities[actual_labels == 0], bins=50, alpha=0.7, label='Negative Reviews')
plt.hist(predicted_probabilities[actual_labels == 1], bins=50, alpha=0.7, label='Positive Reviews')
plt.xlabel('Predicted Probability of Positive Review')
plt.ylabel('Number of Samples')
plt.title('Histogram of Predicted Probabilities by Review Type')
plt.legend()
plt.show()