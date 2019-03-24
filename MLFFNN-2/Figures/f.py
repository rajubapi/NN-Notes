import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 1 if x >= 2 and x <= 6 else 0


def draw():
    global f
    x = np.linspace(0, 10, 8000)
    f = np.vectorize(f)
    y = f(x)
    plt.plot(x, y, color='b', lw=2)
    plt.xlim(0, 8)
    plt.ylim(0, 2)
    plt.title("Plot for f(x)")
    plt.show()
