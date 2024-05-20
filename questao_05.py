# Grid Search
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_model(layers=1, neurons=32):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_res.shape[1], activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)
param_grid = {'layers': [1, 2, 3], 'neurons': [32, 64, 128]}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_res, y_res)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
