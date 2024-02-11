import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.utils import plot_model

# Definieren Sie Ihr Modell hier (wie oben im bereitgestellten Code)
model = Sequential()
model.add(Embedding(10000, 16, input_length=20))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Modell mit plot_model visualisieren
plot_model(model, to_file='model_diagram.png', show_shapes=True, show_layer_names=True)
