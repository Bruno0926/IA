# Criação e Treinamento da Rede Neural
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(layers, neurons):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_res.shape[1], activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Exemplo de treinamento
model = create_model(layers=3, neurons=64)
history = model.fit(X_res, y_res, epochs=50, batch_size=32, validation_split=0.2)
