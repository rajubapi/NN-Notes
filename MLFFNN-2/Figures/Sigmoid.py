import matplotlib.pyplot as plt
import math
import numpy as np


def sigmoid(x):
    """The Sigmoid function"""
    return 1 / (1 + math.exp(-x))


# Plotting Sigmoid
x = np.linspace(-5, 5, 2000)
func = np.vectorize(sigmoid)
y = func(x)
# y = func(-x) to plot the negative variant
plt.plot(x, y, color='r', lw=2)
plt.title("Sigmoid with positive weights")
plt.show()
