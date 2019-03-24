import numpy as np
import matplotlib.pyplot as plt
import imageio



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Perceptron():
    """Implements a single general perceptron"""

    def __init__(self, input_dimensions,  Weights=np.array([]), w_=False, learning_rate=1, epochs=33, fn='binary'):
        if not w_:
            # An extra one for bias
            self.Weights = np.zeros(input_dimensions + 1, dtype=np.float128)
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
        total_error = 0
        for i in range(desired_output.shape[0]):
            # Insert the weight 1 for every input for the bais
            x = np.insert(input_vector[i], 0, 1)
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
        return drawBaisGif(self.Weights)

    def predict(self, X):
        if self.fn == 'binary':
            return 1 if self.find_output(X) > 0.0 else 0
        elif self.fn == 'linear':
            return 1 if self.find_output(X) >= 0.0 else -1
        else:
            return 1 if self.find_output(X) >= 0.5 else 0


# Create a data set
dataset1_x = np.array([1, 7, 8, 9, 4, 8], dtype=np.float128)
dataset1_y = np.array([6, 2, 9, 9, 8, 5], dtype=np.float128)
dataset2_x = np.array([2, 3, 2, 7, 1, 5], dtype=np.float128)
dataset2_y = np.array([1, 3, 4, 1, 3, 2], dtype=np.float128)

# Create Set A for all points with output one
set_A = np.column_stack((dataset1_x, dataset1_y))
set_A_outputs = np.ones(len(set_A))

# Create Set B similarly with output zero
set_B = np.column_stack((dataset2_x, dataset2_y))
set_B_outputs = np.zeros(len(set_B))

# Mash everything together
data = np.concatenate((set_A, set_B))
desired = np.concatenate((set_A_outputs, set_B_outputs))




def drawBaisGif(weights):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(True)
    ax.scatter(x=dataset1_x, y=dataset1_y, color='blue',
               marker='x', label='Data Set 1')
    ax.scatter(x=dataset2_x, y=dataset2_y, color='red',
               marker='o', label='Data Set 2')

    # Sample points
    i = np.linspace(-10,10, 2000)

    # To plot the boundary learned by the perceptron algorithm
    slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
    intercept = -weights[0]/weights[2]
    y1 = (slope * i) + intercept
    ax.plot(i, y1, color='green', label='Mch-Pitt')
    plt.legend()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


# Binary threshlod perceptron
perceptron_binary = Perceptron(input_dimensions=2)
kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./Gifs/WithBias.gif', [perceptron_binary.learn(data, desired) for i in range(perceptron_binary.epochs)], fps=1)

