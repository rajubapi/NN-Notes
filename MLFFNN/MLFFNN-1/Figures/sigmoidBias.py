import matplotlib.pyplot as plt
import numpy as np
import math
labels=["No Bias","-ve Bias","+ve Bias"]

def display(xaxis,yaxis):
    for i in range(len(yaxis)):
        plt.plot(xaxis,yaxis[i],label=labels[i])
    plt.title("z= sigma(xi*wi)-b")
    plt.grid()
    plt.legend()
    plt.show()

def sigmoid(input):
    return 1/(1+math.exp(-input))

def draw():
    x = np.arange(-200,200,0.001)
    weight = 0.05
    bias = 5
    y = [[0 for p in range(len(x))] for m in range(len(labels))] 
    for i in range(len(x)):
        z = x[i]*weight
        zPositiveBias = z + bias
        zNegativeBias = z - bias
        y[0][i]=sigmoid(z)
        y[1][i]=sigmoid(zPositiveBias)
        y[2][i]=sigmoid(zNegativeBias)
    display(x,y)