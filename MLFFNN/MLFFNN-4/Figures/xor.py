import matplotlib.pyplot as plt

x1 = [0, 1]
y1 = [0, 1]
x2 = [0, 1]
y2 = [1, 0]


def draw():
    plt.scatter(x1, y1, color="r", marker='o')
    plt.scatter(x2, y2, color="b", marker='x')
    plt.grid()
    plt.title("XOR")
    plt.show()
