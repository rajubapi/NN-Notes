import matplotlib.pyplot as plt
import numpy as np
import math
def display(xaxis,yaxis):
    plt.plot(xaxis,yaxis)
    plt.grid()
    plt.show()

def Relu(input):
    return max(0,input)
    
def draw():
    x = list(np.arange(-10.0,10.0,0.0001))
    y = []
    for val in x:
        y.append(Relu(val))
    display(x,y)