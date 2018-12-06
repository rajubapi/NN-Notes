import matplotlib.pyplot as plt
import math
import numpy as np


def f(x):
    return 1 if x >= 4 and x <= 6 else 0


def sigmoid(x):
    """The Sigmoid function"""
    return 1 / (1 + math.exp(-x))


# Plot both the f(x) and sigmoid
plt.xlim(0, 10)
plt.ylim(0, 2)
x1 = np.linspace(0, 10, 8000)
f = np.vectorize(f)
y1 = f(x1)
x2 = np.linspace(5, 7, 2000)  # Large weights
y2 = f(x2)
x3 = np.linspace(3, 5, 2000)
y3 = f(x3)
plt.plot(x1, y1, color='b', lw=2)
plt.plot(x2, y2, color='g', lw=2)
plt.plot(x3, y3, color='r', lw=2)
plt.title("Approximating f(x)")
plt.show()
