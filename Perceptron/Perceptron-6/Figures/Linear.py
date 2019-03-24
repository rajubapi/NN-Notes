import matplotlib.pyplot as plt
import numpy as np
import math
def display(xaxis,yaxis):
    plt.plot(xaxis,yaxis)
    plt.grid()
    plt.show()

def linear(input):
        return input
def draw():
        x = list(np.arange(-10.0,10.0,0.0001))
        y = []
        for val in x:
                y.append(linear(val))
        display(x,y)