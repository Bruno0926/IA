# Testando Diferentes Taxas de Aprendizado
from tensorflow.keras.optimizers import Adam

def create_model_with_lr(lr):
    model = Sequential()
    model.add(Dense(64, input_dim=X_res.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

learning_rates = [0.01, 0.001, 0.0001]
histories = {}

for lr in learning_rates:
    model = create_model_with_lr(lr)
    history = model.fit(X_res, y_res, epochs=50, batch_size=32, validation_split=0.2)
    histories[lr] = history
