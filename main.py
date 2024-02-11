import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LSTM 
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def read_json_data(file_path):
    """
    Read data from a JSON file and extract review texts and labels.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        texts (list): A list of review texts.
        labels (list): A list of corresponding labels.
    """
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            review = json.loads(line)
            if 'reviewText' in review:
                texts.append(review['reviewText'])
                labels.append(1 if review['overall'] > 3 else 0)
    return texts, labels

print(tf.__version__)

# Check if TensorFlow is built with ROCm support
if tf.test.is_built_with_rocm():
    print("TensorFlow wurde mit ROCm Unterstützung gebaut.")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPUs gefunden:", gpus)
        except RuntimeError as e:
            print(e)
    else:
        print("Keine GPUs gefunden, die von TensorFlow genutzt werden können.")
else:
    print("TensorFlow wurde nicht mit ROCm Unterstützung gebaut.")

train_path = 'train_reviews.json'
test_path = 'test_reviews.json'

# Read training and test data from JSON files
train_texts, train_labels = read_json_data(train_path)
test_texts, test_labels = read_json_data(test_path)

# Tokenize the text data
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
max_length = 100 
train_data = pad_sequences(train_sequences, maxlen=max_length)
test_data = pad_sequences(test_sequences, maxlen=max_length)

# Convert data to numpy arrays
train_data_array = np.array(train_data)
train_labels_array = np.array(train_labels)
test_data_array = np.array(test_data)
test_labels_array = np.array(test_labels)

plt.figure(figsize=(8, 6))
unique, counts = np.unique(train_labels_array, return_counts=True)
plt.bar(unique, counts, tick_label=['Negative', 'Positive'])
plt.title('Verteilung der Labels im Trainingsdatensatz')
plt.xlabel('Label')
plt.ylabel('Anzahl der Beispiele')
plt.show()

# Define the model architecture
model = Sequential()
model.add(Embedding(20000, 32, input_length=max_length))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer is now recognized
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
plot_model(model, to_file='model_diagram.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=300)


# Define class weights
class_weights = {0: 1.0, 1: 4.275}
# Train the model
model.fit(train_data_array, train_labels_array, epochs=10, validation_split=0.2, class_weight=class_weights)
model.save('mein_rezensionsmodell.h5')

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_data_array, test_labels_array)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Set a threshold for classification
threshold = 0.6

# Predict probabilities for test data
predicted_probabilities = model.predict(test_data_array)

# Print some predicted probabilities
print("Some predicted probabilities:", predicted_probabilities[:10])

# Predict labels based on the threshold
predicted_labels = (predicted_probabilities >= threshold).astype(int)

# Print some predicted labels
print("Some predicted labels:", predicted_labels[:10])

# Calculate correct and incorrect predictions
correct_predictions = (predicted_labels.flatten() == test_labels_array)
incorrect_predictions = ~correct_predictions

# Count correct and incorrect predictions
correct_counts = [np.sum(correct_predictions & (test_labels_array == i)) for i in [0, 1]]
incorrect_counts = [np.sum(incorrect_predictions & (test_labels_array == i)) for i in [0, 1]]

# Print correct and incorrect counts
print("Correct predictions count:", np.sum(correct_predictions))
print("Incorrect predictions count:", np.sum(incorrect_predictions))
print("Correct counts:", correct_counts)
print("Incorrect counts:", incorrect_counts)

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
plt.hist(predicted_probabilities[test_labels_array == 0], bins=50, alpha=0.7, label='Negative Reviews')
plt.hist(predicted_probabilities[test_labels_array == 1], bins=50, alpha=0.7, label='Positive Reviews')
plt.xlabel('Predicted Probability of Positive Review')
plt.ylabel('Number of Samples')
plt.title('Histogram of Predicted Probabilities by Review Type')
plt.legend()
plt.show()
