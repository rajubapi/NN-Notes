import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Perceptron():
    """Implements a single general perceptron"""

    def __init__(self, input_dimensions,  Weights=np.array([]),
                 w_=False, learning_rate=1, epochs=500, fn='binary'):
        if not w_:
            # An extra one for bias
            self.Weights = np.zeros(input_dimensions)
        else:
            self.Weights = Weights
        self.epochs = epochs
        self.eta = learning_rate
        self.fn = fn

    # The activation function
    def activation_fn(self, y):
        if self.fn == 'binary':
            return 1 if y >= 0 else 0
        elif self.fn == 'linear':
            return y
        else:  # fn == 'sigmoid'
            return sigmoid(y)

    def find_output(self, input_matrix):
        z = self.Weights.T.dot(input_matrix)
        return self.activation_fn(z)

    def learn(self, input_vector, desired_output):
        errors = []
        for _ in range(self.epochs):
            total_error = 0
            for i in range(desired_output.shape[0]):
                # Insert the weight 1 for every input for the bais
                x = input_vector[i]
                actual_output = self.find_output(x)
                error = desired_output[i] - actual_output
                # Weight update rules
                if self.fn == 'binary':
                    self.Weights = self.Weights + self.eta * error * x
                    if error <= 0:
                        total_error += int(error != 0.0)
                elif self.fn == 'linear':
                    self.Weights = self.Weights + self.eta * error * x
                    total_error += (error ** 2) / 2
                else:
                    self.Weights = self.Weights + self.eta * error * \
                        x * actual_output * (1 - actual_output)
                    total_error += (error ** 2) / 2
            errors.append(total_error)
        return errors

    def predict(self, X):
        if self.fn == 'binary':
            return 1 if self.find_output(X) > 0.0 else 0
        elif self.fn == 'linear':
            return 1 if self.find_output(X) >= 0.0 else -1
        else:
            return 1 if self.find_output(X) >= 0.5 else 0


def draw(vectors):
    vectors = np.array(vectors)
    x1 = [vectors[0][0], vectors[1][0]]
    x2 = [vectors[0][1], vectors[1][1]]
    plt.quiver([0, 0], [0, 0], x1, x2, color=['r', 'b'],
               angles='xy', scale_units='xy', scale=1)
    clf = Perceptron(input_dimensions=2)
    clf.learn(vectors, np.array([0, 1]))
    plt.quiver([0], [0], clf.Weights[0], clf.Weights[1], color='g',
               angles='xy', scale_units='xy', scale=1)
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.show()
