import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y):
        for epoch in range(self.epochs):
            # Forward Propagation
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            output_layer_output = self.sigmoid(output_layer_input)

            # Backward Propagation
            output_error = y - output_layer_output
            output_delta = output_error * self.sigmoid_derivative(output_layer_output)

            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += self.learning_rate * np.dot(hidden_layer_output.T, output_delta)
            self.weights_input_hidden += self.learning_rate * np.dot(X.T, hidden_delta)
            self.bias_output += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
            self.bias_hidden += self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output_layer_output = self.sigmoid(output_layer_input)
        return np.round(output_layer_output)

def generate_boolean_data(n, function):
    data = []
    labels = []
    for i in range(2**n):
        inputs = [int(x) for x in bin(i)[2:].zfill(n)]
        if function == "AND":
            label = int(all(inputs))
        elif function == "OR":
            label = int(any(inputs))
        elif function == "XOR":
            label = int(sum(inputs) % 2)
        data.append(np.array(inputs))
        labels.append(label)
    return np.array(data), np.array(labels).reshape(-1, 1)

def plot_results(nn, n, function):
    data, labels = generate_boolean_data(n, function)
    predictions = nn.predict(data)

    plt.figure()
    for inputs, label, prediction in zip(data, labels, predictions):
        color = 'r' if prediction == 0 else 'b'
        if n == 2:
            plt.scatter(inputs[0], inputs[1], c=color)
        plt.text(inputs[0], inputs[1], f'{inputs}', ha='center', va='center', color='white', bbox=dict(facecolor=color, edgecolor=color, boxstyle='round,pad=0.3'))
    plt.title(f'{function} Function with {n} inputs')
    plt.show()

def main():
    n = int(input("Enter the number of inputs: "))
    function = input("Enter the function (AND/OR/XOR): ")

    data, labels = generate_boolean_data(n, function)
    nn = NeuralNetwork(input_size=n, hidden_size=2*n, output_size=1)
    nn.train(data, labels)
    plot_results(nn, n, function)

if __name__ == "__main__":
    main()
