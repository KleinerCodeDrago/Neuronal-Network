import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()

train_true_path = 'train_true.txt'
train_not_true_path = 'train_not_true.txt'
test_true_path = 'test_true.txt'
test_not_true_path = 'test_not_true.txt'

train_true_titles = read_data(train_true_path)
train_not_true_titles = read_data(train_not_true_path)
test_true_titles = read_data(test_true_path)
test_not_true_titles = read_data(test_not_true_path)

train_titles = train_true_titles + train_not_true_titles
train_labels = [1] * len(train_true_titles) + [0] * len(train_not_true_titles)

test_titles = test_true_titles + test_not_true_titles
test_labels = [1] * len(test_true_titles) + [0] * len(test_not_true_titles)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_titles)
train_sequences = tokenizer.texts_to_sequences(train_titles)
train_data = pad_sequences(train_sequences, maxlen=20)

test_sequences = tokenizer.texts_to_sequences(test_titles)
test_data = pad_sequences(test_sequences, maxlen=20)

train_data_array = np.array(train_data)
train_labels_array = np.array(train_labels)

test_data_array = np.array(test_data)
test_labels_array = np.array(test_labels)

model = Sequential()
model.add(Embedding(10000, 16, input_length=20))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data_array, train_labels_array, epochs=10, validation_split=0.2)
loss, accuracy = model.evaluate(test_data_array, test_labels_array)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

threshold = 0.5

predicted_probabilities = model.predict(test_data_array)

print("Some predicted probabilities:", predicted_probabilities[:10])

predicted_labels = (predicted_probabilities >= threshold).astype(int)

print("Some predicted labels:", predicted_labels[:10])

correct_predictions = (predicted_labels.flatten() == test_labels_array)
incorrect_predictions = ~correct_predictions

print("Correct predictions count:", np.sum(correct_predictions))
print("Incorrect predictions count:", np.sum(incorrect_predictions))

correct_counts = [np.sum(correct_predictions & (test_labels_array == i)) for i in [0, 1]]
incorrect_counts = [np.sum(incorrect_predictions & (test_labels_array == i)) for i in [0, 1]]

print("Correct counts:", correct_counts)
print("Incorrect counts:", incorrect_counts)


labels = ['Not true', 'True']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, correct_counts, width, label='Correct', color='green')
rects2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='red')

ax.set_xlabel('Category')
ax.set_ylabel('Number of Predictions')
ax.set_title('Model Performance on Test Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.show()


predicted_probabilities = model.predict(test_data_array)

plt.figure(figsize=(10, 5))
plt.hist(predicted_probabilities[test_labels_array == 0], bins=50, alpha=0.7, label='Not true')
plt.hist(predicted_probabilities[test_labels_array == 1], bins=50, alpha=0.7, label='True')
plt.xlabel('Predicted Probability of Being True')
plt.ylabel('Number of Samples')
plt.title('Histogram of Predicted Probabilities')
plt.legend()
plt.show()
