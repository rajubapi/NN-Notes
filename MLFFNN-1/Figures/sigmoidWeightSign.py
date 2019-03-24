import matplotlib.pyplot as plt
import numpy as np
import math
x = np.arange(-200,200,0.001)
w = [0.05,-0.05]

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
        for i in range(len(w)):
                for j in range(len(x)):
                        z = x[j]*w[i]
                        y[i][j]=sigmoid(z)
        display(x,y)
