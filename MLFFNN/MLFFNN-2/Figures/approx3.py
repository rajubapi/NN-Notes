import matplotlib.pyplot as plt
import math
import numpy as np


def f(x):
    return 1 if x >= 2 and x <= 6 else 0


def sigmoid(x):
    """The Sigmoid function"""
    return 1 / (1 + math.exp(-x))


def draw():
    global sigmoid, f
    x1 = np.linspace(0, 10, 8000)
    f = np.vectorize(f)
    y1 = f(x1)
    x = np.arange(-20, 20, 0.01)
    w = 3
    bais1 = 18.5
    bais2 = 5.5
    z = x * w
    z1 = np.copy(z)
    z2 = np.copy(z)
    z1 -= bais1
    z2 -= bais2
    sigmoid = np.vectorize(sigmoid)
    y2 = sigmoid(-z1)
    y3 = sigmoid(z2)
    plt.plot(x, y3, color='g', lw=2, label="Sigmoid +ve")
    plt.plot(x, y2, color='r', lw=2, label="Sigmoid -ve")
    plt.plot(x1, y1, color='b', lw=2, label="f(x)")
    # Get values
    plt.fill_between(x, [0], y2, where=(y3 >= y2), facecolor='red')
    plt.fill_between(x, [0], y3, where=(y2 >= y3), facecolor='green')
    plt.fill_between
    plt.xlim(0, 10)
    plt.ylim(0, 2)
    plt.title("Approximation f(x)")
    plt.legend()
    plt.grid()
    plt.show()
