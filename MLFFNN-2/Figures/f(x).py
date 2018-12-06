import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 1 if x >= 4 and x <= 6 else 0


x = np.linspace(0, 10, 8000)
f = np.vectorize(f)
y = f(x)
plt.plot(x, y, color='b', lw=2)
plt.title("Plot for f(x)")
plt.show()
