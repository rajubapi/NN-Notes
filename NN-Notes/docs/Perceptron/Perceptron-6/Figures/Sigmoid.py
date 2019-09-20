import matplotlib.pyplot as plt
import numpy as np
import math
def display(xaxis,yaxis,title):
    plt.plot(xaxis,yaxis)
    plt.grid()
    plt.title(title)
    plt.show()

def sigmoid(input):
    return 1/(1+math.exp(-input))

def draw():
    x = list(np.arange(-10,10,0.0001))
    y = []
    for val in x:
        y.append(sigmoid(val))
    display(x,y,"Sigmoid")

def drawDerivative():
    x = list(np.arange(-10,10,0.0001))
    y = []
    for val in x:
        y.append(sigmoid(val)*(1-sigmoid(val)))
    display(x,y,"Sigmoid Derivative")