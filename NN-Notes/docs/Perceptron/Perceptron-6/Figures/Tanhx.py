import matplotlib.pyplot as plt
import numpy as np
import math
def display(xaxis,yaxis,title):
    plt.plot(xaxis,yaxis)
    plt.grid()
    plt.title(title)
    plt.show()

def tanhx(input):
    return (math.exp(input)-math.exp(-input))/(math.exp(input)+math.exp(-input))

def draw():
    x = list(np.arange(-10,10,0.0001))
    y = []
    for val in x:
        y.append(tanhx(val))
    display(x,y,"Tangent Hyperbolic")

def drawDerivative():
    x = list(np.arange(-10,10,0.0001))
    y = []
    for val in x:
        y.append(1-tanhx(val)**2)
    display(x,y,"Tangent Hyperbolic Derivative")