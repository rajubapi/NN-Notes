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
    bais = 5.5
    z = x * w
    z -= bais
    sigmoid = np.vectorize(sigmoid)
    y2 = sigmoid(z)
    plt.plot(x, y2, color='r', lw=2, label="Sigmoid")
    plt.plot(x1, y1, color='b', lw=2, label="f(x)")
    plt.fill_between(x, [0], y2)
    plt.xlim(0, 10)
    plt.title("Approximation 1")
    plt.legend()
    plt.grid()
    plt.show()
