from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

titles = ["Beispiel Titel 1", "Beispiel Titel 2", "Weiterer Titel"]
labels = [0, 1, 0]

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(titles)
sequences = tokenizer.texts_to_sequences(titles)
data = pad_sequences(sequences, maxlen=20)

data_array = np.array(data)
labels_array = np.array(labels)

model = Sequential()
model.add(Embedding(10000, 16, input_length=20))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(data_array, labels_array, epochs=10, validation_split=0.2)
