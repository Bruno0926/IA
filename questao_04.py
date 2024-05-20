# Ajustando Outros Hiperparâmetros
from tensorflow.keras.layers import Dropout

def create_advanced_model(layers, neurons, dropout_rate):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_res.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Exemplo de treinamento com dropout
model = create_advanced_model(layers=3, neurons=64, dropout_rate=0.5)
history = model.fit(X_res, y_res, epochs=50, batch_size=32, validation_split=0.2)
