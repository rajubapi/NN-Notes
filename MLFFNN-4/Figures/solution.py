import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-10, 10, 2000)
y1 = -x1 + 0.6
x2 = np.linspace(-10, 10, 2000)
y2 = -x2 + 1.2

# Plot line
plt.plot(x1, y1, color="g")
plt.plot(x2, y2, color="g")
plt.title("Solution")

# Plot points
x3 = [0, 1]
y3 = [0, 1]
x4 = [0, 1]
y4 = [1, 0]
plt.scatter(x3, y3, color="r", marker='o')
plt.scatter(x4, y4, color="b", marker='x')

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
