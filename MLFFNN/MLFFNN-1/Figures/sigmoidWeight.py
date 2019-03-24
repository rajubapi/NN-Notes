import matplotlib.pyplot as plt
import numpy as np
import math

x = np.arange(-200,200,0.001)
w = [0.01,0.05,0.1,0.2,2]
bias = 5

def display(xaxis,yaxis):
    for i in range(len(yaxis)):
        plt.plot(xaxis,yaxis[i],label=("Weight ="+str(w[i])))
    plt.grid()
    plt.legend()
    plt.show()

def sigmoid(input):
    return 1/(1+math.exp(-input))

def draw():
        y = [[0 for p in range(len(x))] for m in range(len(w))] 
        for j in range(len(w)):
                for k in range(len(x)):
                        z= x[k]*w[j]
                        y[j][k] = sigmoid(z)
        display(x,y)