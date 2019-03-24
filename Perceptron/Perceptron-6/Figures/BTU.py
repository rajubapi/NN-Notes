import matplotlib.pyplot as plt
import numpy as np
import math
def display(xaxis,yaxis):
    plt.plot(xaxis,yaxis)
    plt.grid()
    plt.show()

def BTU(input):
    if input>=0:
        return 1
    else:
        return 0
def draw():
    x = list(np.arange(-10.0,10.0,0.0001))
    y = []
    for val in x:
        y.append(BTU(val))
    display(x,y)
