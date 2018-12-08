import matplotlib.pyplot as plt
import math
import numpy as np


def sigmoid(x):
    """The Sigmoid function"""
    return 1 / (1 + math.exp(-x))


# Plotting Sigmoid
def draw():
    global sigmoid
    x = np.arange(-200, 200, 0.001)
    w = 0.1
    z = x * w
    sigmoid = np.vectorize(sigmoid)
    y1 = sigmoid(z)
    y2 = sigmoid(-z)
    plt.plot(x, y1, color='r', lw=2, label="+ve Weights")
    plt.plot(x, y2, color='b', lw=2, label="-ve Weights")
    plt.title("Sigmoid with +ve and -ve weights")
    plt.legend()
    plt.grid()
    plt.show()
