import matplotlib.pyplot as plt
import numpy as np


class Perceptron():
    """Implements a single general perceptron"""

    def __init__(self, input_dimensions,  Weights=np.array([]), w_=False, learning_rate=0.01, epochs=8):
        if not w_:
            # An extra one for bias
            self.Weights = np.zeros(input_dimensions + 1, dtype=np.float128)
        else:
            self.Weights = Weights
        self.epochs = epochs
        self.eta = learning_rate

    # The activation function
    def activation_fn(self, y):
        return 1 if y >= 0 else 0

    def find_output(self, input_matrix):
        z = self.Weights.T.dot(input_matrix)
        return self.activation_fn(z)

    def learn(self, input_vector, desired_output):
        errors = []
        for _ in range(self.epochs):
            total_error = 0
            for i in range(desired_output.shape[0]):
                # Insert the weight 1 for every input for the bais
                x = np.insert(input_vector[i], 0, 1)
                actual_output = self.find_output(x)
                error = desired_output[i] - actual_output
                # Weight update rules
                self.Weights = self.Weights + self.eta * error * x
                if error <= 0:
                    total_error += int(error != 0.0)
            errors.append(total_error)
        return errors

    def predict(self, X):
        return 1 if self.find_output(X) > 0.0 else 0


def fit():
    perceptron = Perceptron(input_dimensions=2)
    # AND data
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    data = np.array(data)
    desired = [0, 1, 1, 1]
    desired = np.array(desired)
    # Fit a boundary
    perceptron.learn(data, desired)

    # Plot the AND data
    plt.scatter(x=[1, 0, 1], y=[1, 1, 0], color='blue',
                marker='x')
    plt.scatter(x=[0], y=[0], color='red',
                marker='o')
    # Sample points
    i = np.linspace(np.amin(data, axis=0)[0], np.amax(data, axis=0)[0], 2000)

    # To plot the boundary learned by the perceptron algorithm
    weights = perceptron.Weights
    slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
    intercept = -weights[0]/weights[2]
    y1 = (slope * i) + intercept
    plt.plot(i, y1, color='green')
    plt.show()
