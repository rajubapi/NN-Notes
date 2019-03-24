import numpy as np
import matplotlib.pyplot as plt
import math

n_samples = 30
n_samples_test = 100

data = np.sort(np.random.rand(n_samples))
target=np.cos(1.5*np.pi*data) + np.random.randn(n_samples) * 0.1
plt.plot(data,target,c="b",marker="*")
class MLFFNN:
    def __init__(self,eta=0.2,inNodes=1,outNodes=1,hiddenNodes=7,activation='tanhx',epochs=5000):
        self.eta=eta
        self.inNodes=inNodes
        self.outNodes=outNodes
        self.hiddenNodes=hiddenNodes
        self.weightsitoh=np.random.randn(self.inNodes+1, self.hiddenNodes)
        self.weightshtoo=np.random.randn(self.hiddenNodes, self.outNodes)
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
                dataWithBias=np.insert(np.array([input[i]],ndmin=2),1,1)
                self.backPropagation(dataWithBias,target[i],self.feedforward(dataWithBias))

def plotPredictions(mlp):
    predictedData1,predictedData2=[],[]
    for i in range(len(data)):
        dataWithBias=np.insert(np.array([data[i]],ndmin=2),1,1)
        predictedData1.append(mlp.feedforward(dataWithBias))

    subplt=plt.subplot(1,2,1)
    subplt.set_title("Predicting the Training Data")
    plt.plot(data, np.array(predictedData1).reshape(n_samples,),label="Predicted")
    plt.plot(data, target, label="Target")
    plt.legend()
    X_test=np.linspace(0,1,n_samples_test)
    for i in range(len(X_test)):
        xWithBias=np.insert(np.array([X_test[i]],ndmin=2),1,1)
        predictedData2.append(mlp.feedforward(xWithBias))
    subplt=plt.subplot(1,2,2)
    subplt.set_title("Predicting the Test Data")
    plt.plot(X_test,np.array(predictedData2).reshape(n_samples_test,), label="Model")
    plt.plot(X_test, np.cos(1.5*np.pi*X_test), label="True function")
    plt.legend()
    plt.show()

def overFitting():
    mlp=MLFFNN(hiddenNodes=20)
    mlp.train(data,target)
    plotPredictions(mlp)

def withoutOverFitting():
    mlp=MLFFNN()
    mlp.train(data,target)
    plotPredictions(mlp)
# overFitting()
# withoutOverFitting()