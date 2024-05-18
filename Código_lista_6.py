import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1, epochs=1000):
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(n_inputs + 1)  # +1 para o bias

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                
def generate_boolean_data(n, function):
    data = []
    labels = []
    for i in range(2**n):
        inputs = [int(x) for x in bin(i)[2:].zfill(n)]
        if function == "AND":
            label = int(all(inputs))
        elif function == "OR":
            label = int(any(inputs))
        data.append(np.array(inputs))
        labels.append(label)
    return np.array(data), np.array(labels)

def plot_decision_boundary(p, n, function):
    plt.figure()
    data, labels = generate_boolean_data(n, function)
    for inputs, label in zip(data, labels):
        color = 'r' if label == 0 else 'b'
        if n == 2:
            plt.scatter(inputs[0], inputs[1], c=color)
        plt.text(inputs[0], inputs[1], f'{inputs}', ha='center', va='center', color='white', bbox=dict(facecolor=color, edgecolor=color, boxstyle='round,pad=0.3'))

    x_values = np.array([0, 1])
    if n == 2:
        y_values = -(p.weights[0] + p.weights[1] * x_values) / p.weights[2]
        plt.plot(x_values, y_values, label="Decision Boundary")
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
    plt.title(f'{function} Function with {n} inputs')
    plt.legend()
    plt.show()

def main():
    n = int(input("Enter the number of inputs: "))
    function = input("Enter the function (AND/OR): ")

    data, labels = generate_boolean_data(n, function)
    perceptron = Perceptron(n_inputs=n)
    perceptron.train(data, labels)
    plot_decision_boundary(perceptron, n, function)

if __name__ == "__main__":
    main()
