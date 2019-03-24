import numpy as np
import matplotlib.pyplot as plt
import math

data = np.array([[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])

target = np.array([0, 1, 1, 0])
def printPrediction(mlp):
    print("Data\t:","\tTarget\t:","\tPredicted")
    print("-----------------------------------------")
    for i in range(len(data)):
        dataWithBias=np.insert(np.array([data[i]],ndmin=2),2,1)
        print(data[i],"\t:\t",target[i],"\t:\t",mlp.feedforward(dataWithBias)[0][0])

class MLFFNN:
    def __init__(self,eta=0.2,inNodes=2,outNodes=1,hiddenNodes=3,activation='sigmoid',epochs=100000):
        self.eta=eta
        self.inNodes=inNodes
        self.outNodes=outNodes
        self.hiddenNodes=hiddenNodes
        self.weightsitoh = np.random.normal(0, 1, (self.inNodes+1, self.hiddenNodes))
        self.weightshtoo = np.random.normal(0, 1, (self.hiddenNodes, self.outNodes))
        self.activation=activation
        self.epochs=epochs
        self.xj=np.array([])

    def actvationFn(self,input):
        if self.activation=='btu':
            return 1 if input>0 else 0
        elif  self.activation=='sigmoid':
            return 1/(1+np.e**-input)
        elif  self.activation=='relu':
            return max(0,input)
        elif  self.activation=='tanhx':
            return np.tanh(input)
        else:
            return input
    
    def derivative(self,val):
        if self.activation=='sigmoid':
            return val*(1-val)

        elif self.activation=='tanhx':
            return 1.0-val**2

    def feedforward(self,input):
        self.xj=self.actvationFn(np.dot(input[:,None].T,self.weightsitoh))
        return self.actvationFn(np.dot(self.xj,self.weightshtoo))

    def backPropagation(self,input,target,actual):
        error=target- actual
        deltai=error*self.derivative(actual)
        deltaWij=self.eta*deltai*self.xj
        self.weightshtoo+=deltaWij.T
        deltaj=np.dot(deltai,self.weightshtoo.T)*self.derivative(self.xj)
        deltaWjk=self.eta*np.dot(input[:,None],deltaj)
        self.weightsitoh+=deltaWjk

    def train(self,input,target):
        for _ in range(self.epochs):
            for i in range(len(input)):
                inputWithBias=np.insert(np.array([input[i]],ndmin=2),2,1)
                self.backPropagation(inputWithBias,target[i],self.feedforward(inputWithBias))


def fit():
    mlp=MLFFNN()
    mlp.train(data,target)
    return mlp